//! `rl_dash` — a native desktop dashboard for `rl_agent` training telemetry.
//!
//! Reads the same per-run files the trainer writes
//! (`runs/<run_id>/run.json` + `runs/<run_id>/metrics.jsonl`, see
//! `rl_agent/src/telemetry.rs`), live-follows the newest run while
//! training runs, and lets you overlay any number of runs for comparison —
//! on iteration / env-steps / wall-clock axes, as a run×iteration heatmap,
//! or in a per-run summary table. Extra views: best-so-far envelopes,
//! log-scale, eval/checkpoint guides, and seed mean±std bands.
//!
//! ```bash
//! cargo run -p rl_dash --release                 # watches ./runs
//! cargo run -p rl_dash -- --runs /path/to/runs   # explicit run root
//! cargo run -p rl_dash -- --metric policy_loss    # start on a metric
//! ```

#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

use std::collections::BTreeMap;
use std::io::{Read, Seek, SeekFrom};
use std::path::{Path, PathBuf};
use std::time::{Duration, SystemTime};

use eframe::egui;
use sysinfo::System;

use egui_plot::{Legend, Line, LineStyle, Plot, PlotPoints, Points, Polygon, VLine};
use serde::Deserialize;

/// Metric registry — keys mirror `metrics.jsonl` fields (and
/// `godot/scripts/rl_chart.gd` / `rl_agent/dashboard.html`).
const METRICS: &[(&str, &str)] = &[
    ("mean_reward_per_step", "Reward / step"),
    ("policy_loss", "Policy loss"),
    ("value_loss", "Value loss"),
    ("entropy", "Entropy"),
    ("advantage_mean", "Advantage mean"),
    ("advantage_std", "Advantage std"),
    ("steps_per_s", "Env steps / s"),
    ("crash_rate", "Crash rate"),
    ("mean_episode_steps", "Episode steps"),
    ("episodes_completed", "Episodes completed"),
    // Mission-mode counters (`rl_agent` mission runs only).
    ("mission_captures", "Mission captures"),
    ("mission_returns", "Mission returns"),
    ("mission_successes", "Mission successes"),
    ("alt_err", "Eval alt err (m)"),
    ("spd_err", "Eval spd err (m/s)"),
    ("eval_reward", "Eval mean reward"),
    // Ground-truth flight state (`env` block, `rl_agent.telemetry` schema).
    ("env_alt_mean", "Env altitude (m)"),
    ("env_airspeed_mean", "Env airspeed (m/s)"),
    ("env_climb_mean", "Env climb (m/s)"),
    ("env_ground_speed_mean", "Env ground speed (m/s)"),
    ("env_roll_abs_mean", "Env |roll| (rad)"),
    ("env_pitch_abs_mean", "Env |pitch| (rad)"),
    ("env_aoa_mean", "Env AoA (rad)"),
    ("env_wind_mean", "Env wind (m/s)"),
    ("env_throttle_mean", "Env throttle"),
    ("env_surface_effort_mean", "Env surface effort (deg)"),
    ("env_baro_err_abs_mean", "Env baro err (m)"),
    ("env_battery_pct_min", "Env battery min (%)"),
    // Weather (`env.weather`, only present on runs with a weather system).
    ("weather_wind_speed_mean", "Weather wind (m/s)"),
    ("weather_gust_rms_max", "Weather gust max (m/s)"),
    ("weather_turbulence_mean", "Weather turbulence"),
    ("weather_severity_mean", "Weather severity"),
    ("weather_visibility_min", "Weather visibility (m)"),
];

/// Colors for vertical guide lines (eval / checkpoint markers) on the chart.
/// Excluded from `PALETTE` so run colors never collide with guides.
const EVAL_MARKER: egui::Color32 = egui::Color32::from_rgb(150, 160, 180);
const CKPT_MARKER: egui::Color32 = egui::Color32::from_rgb(242, 204, 77);

/// Per-run line colors, assigned by position in the run list.
const PALETTE: [egui::Color32; 8] = [
    egui::Color32::from_rgb(115, 217, 134),
    egui::Color32::from_rgb(242, 154, 51),
    egui::Color32::from_rgb(77, 166, 242),
    egui::Color32::from_rgb(191, 115, 242),
    egui::Color32::from_rgb(242, 204, 77),
    egui::Color32::from_rgb(242, 115, 102),
    egui::Color32::from_rgb(77, 217, 217),
    egui::Color32::from_rgb(140, 217, 115),
];

/// Cap per-run records so a pathological file can't grow memory forever.
const MAX_RECORDS: usize = 300_000;

/// How often the UI repaints and re-reads telemetry files.
const POLL_MS: u64 = 200;

/// Re-scan the runs dir every N polls (~1 s) so a trainer that spawns a run
/// dir *after* the dashboard opens shows up without a manual "Rescan" click.
const RESCAN_EVERY_POLLS: u8 = 5;

// ── Telemetry schema (mirror of rl_agent::telemetry) ─────────────────

// Fields mirror `rl_agent::telemetry`; unread ones are kept so the schema
// stays recognizable next to the writer. `serde` ignores unknown fields, so
// extra writer fields never break the reader.
#[allow(dead_code)]
#[derive(Debug, Clone, Default, Deserialize)]
struct RunManifest {
    #[serde(default)]
    run_id: Option<String>,
    #[serde(default)]
    status: Option<String>,
    #[serde(default)]
    started_utc: Option<String>,
    #[serde(default)]
    ended_utc: Option<String>,
    #[serde(default)]
    algo: Option<String>,
    #[serde(default)]
    backend: Option<String>,
    #[serde(default)]
    git_commit: Option<String>,
    /// Original argv. The optional `--name` value is the human model label
    /// shown on the dashboard's model tabs.
    #[serde(default)]
    command_line: Vec<String>,
    /// Raw hyper-parameter blocks, retained so runs that share a config
    /// (same algo/backend/network/algorithm/trainer — seed excluded) can be
    /// grouped into mean±std seed-replication bands.
    #[serde(default)]
    trainer: Option<serde_json::Value>,
    #[serde(default)]
    network: Option<serde_json::Value>,
    #[serde(default)]
    algorithm: Option<serde_json::Value>,
}

#[allow(dead_code)]
#[derive(Debug, Clone, Default, Deserialize)]
struct EvalMetrics {
    #[serde(default)]
    alt_err: Option<f64>,
    #[serde(default)]
    spd_err: Option<f64>,
    #[serde(default)]
    total_reward: Option<f64>,
    #[serde(default)]
    crashes: Option<u64>,
    #[serde(default)]
    episodes: Option<u64>,
}

#[allow(dead_code)]
#[derive(Debug, Clone, Default, Deserialize)]
struct MetricsRecord {
    iteration: i64,
    #[serde(default)]
    elapsed_s: Option<f64>,
    #[serde(default)]
    mean_reward_per_step: Option<f64>,
    #[serde(default)]
    policy_loss: Option<f64>,
    #[serde(default)]
    value_loss: Option<f64>,
    #[serde(default)]
    entropy: Option<f64>,
    #[serde(default)]
    advantage_mean: Option<f64>,
    #[serde(default)]
    steps_per_s: Option<f64>,
    #[serde(default)]
    crash_rate: Option<f64>,
    #[serde(default)]
    advantage_std: Option<f64>,
    #[serde(default)]
    mean_episode_steps: Option<f64>,
    /// Episodes completed *this iteration*; the dashboard accumulates it for
    /// the `episodes_completed` metric.
    #[serde(default)]
    episodes_completed: Option<u64>,
    /// Mission-mode counters. Absent on avionics runs, so they stay `None`.
    #[serde(default)]
    mission_captures: Option<f64>,
    #[serde(default)]
    mission_returns: Option<f64>,
    #[serde(default)]
    mission_successes: Option<f64>,
    #[serde(default)]
    checkpoint: bool,
    #[serde(default)]
    eval: Option<EvalMetrics>,
    /// Ground-truth flight-state summary over the rollout (`env` key in
    /// `metrics.jsonl`). Absent on runs recorded before env telemetry existed.
    #[serde(default)]
    env: Option<EnvSummary>,
}

/// The `env` block of an iteration record. Every field is optional so older
/// runs (or a schema change) degrade to "no data" instead of failing to parse.
#[allow(dead_code)]
#[derive(Debug, Clone, Default, Deserialize)]
struct EnvSummary {
    #[serde(default)]
    alt_mean: Option<f64>,
    #[serde(default)]
    alt_min: Option<f64>,
    #[serde(default)]
    alt_max: Option<f64>,
    #[serde(default)]
    airspeed_mean: Option<f64>,
    #[serde(default)]
    climb_mean: Option<f64>,
    #[serde(default)]
    ground_speed_mean: Option<f64>,
    #[serde(default)]
    roll_abs_mean: Option<f64>,
    #[serde(default)]
    pitch_abs_mean: Option<f64>,
    #[serde(default)]
    aoa_mean: Option<f64>,
    #[serde(default)]
    wind_mean: Option<f64>,
    #[serde(default)]
    throttle_mean: Option<f64>,
    #[serde(default)]
    surface_effort_mean: Option<f64>,
    #[serde(default)]
    baro_err_abs_mean: Option<f64>,
    #[serde(default)]
    airspeed_err_abs_mean: Option<f64>,
    #[serde(default)]
    battery_pct_min: Option<f64>,
    #[serde(default)]
    weather: Option<WeatherSummary>,
}

/// The nested `env.weather` block, when the run had a weather system.
#[allow(dead_code)]
#[derive(Debug, Clone, Default, Deserialize)]
struct WeatherSummary {
    #[serde(default)]
    wind_speed_mean: Option<f64>,
    #[serde(default)]
    gust_rms_mean: Option<f64>,
    #[serde(default)]
    gust_rms_max: Option<f64>,
    #[serde(default)]
    turbulence_mean: Option<f64>,
    #[serde(default)]
    severity_mean: Option<f64>,
    #[serde(default)]
    rain_mean: Option<f64>,
    #[serde(default)]
    visibility_min: Option<f64>,
    #[serde(default)]
    updraft_abs_mean: Option<f64>,
    #[serde(default)]
    shear_abs_mean: Option<f64>,
}

impl MetricsRecord {
    /// Flatten a record into the value for one metric key (`None` = no data
    /// point, e.g. eval metrics on non-eval iterations).
    fn metric(&self, key: &str) -> Option<f64> {
        match key {
            "mean_reward_per_step" => self.mean_reward_per_step,
            "policy_loss" => self.policy_loss,
            "value_loss" => self.value_loss,
            "entropy" => self.entropy,
            "advantage_mean" => self.advantage_mean,
            "advantage_std" => self.advantage_std,
            "steps_per_s" => self.steps_per_s,
            "crash_rate" => self.crash_rate,
            "mean_episode_steps" => self.mean_episode_steps,
            "mission_captures" => self.mission_captures,
            "mission_returns" => self.mission_returns,
            "mission_successes" => self.mission_successes,
            "alt_err" => self.eval.as_ref().and_then(|e| e.alt_err),
            "spd_err" => self.eval.as_ref().and_then(|e| e.spd_err),
            "eval_reward" => self.eval.as_ref().and_then(|e| e.total_reward),
            // Ground-truth flight state (`env` block).
            "env_alt_mean" => self.env.as_ref().and_then(|e| e.alt_mean),
            "env_airspeed_mean" => self.env.as_ref().and_then(|e| e.airspeed_mean),
            "env_climb_mean" => self.env.as_ref().and_then(|e| e.climb_mean),
            "env_ground_speed_mean" => self.env.as_ref().and_then(|e| e.ground_speed_mean),
            "env_roll_abs_mean" => self.env.as_ref().and_then(|e| e.roll_abs_mean),
            "env_pitch_abs_mean" => self.env.as_ref().and_then(|e| e.pitch_abs_mean),
            "env_aoa_mean" => self.env.as_ref().and_then(|e| e.aoa_mean),
            "env_wind_mean" => self.env.as_ref().and_then(|e| e.wind_mean),
            "env_throttle_mean" => self.env.as_ref().and_then(|e| e.throttle_mean),
            "env_surface_effort_mean" => self.env.as_ref().and_then(|e| e.surface_effort_mean),
            "env_baro_err_abs_mean" => self.env.as_ref().and_then(|e| e.baro_err_abs_mean),
            "env_battery_pct_min" => self.env.as_ref().and_then(|e| e.battery_pct_min),
            // Weather (`env.weather` block, only when the run had a weather system).
            "weather_wind_speed_mean" => self
                .env
                .as_ref()
                .and_then(|e| e.weather.as_ref())
                .and_then(|w| w.wind_speed_mean),
            "weather_gust_rms_max" => self
                .env
                .as_ref()
                .and_then(|e| e.weather.as_ref())
                .and_then(|w| w.gust_rms_max),
            "weather_turbulence_mean" => self
                .env
                .as_ref()
                .and_then(|e| e.weather.as_ref())
                .and_then(|w| w.turbulence_mean),
            "weather_severity_mean" => self
                .env
                .as_ref()
                .and_then(|e| e.weather.as_ref())
                .and_then(|w| w.severity_mean),
            "weather_visibility_min" => self
                .env
                .as_ref()
                .and_then(|e| e.weather.as_ref())
                .and_then(|w| w.visibility_min),
            _ => None,
        }
    }
}

// ── Run model ────────────────────────────────────────────────────────

/// Horizontal axis of the chart: what each record's x-coordinate means.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum XMode {
    /// PPO iteration number (the index each telemetry row is stamped with).
    Iteration,
    /// Cumulative environment steps (sample-efficiency view).
    Steps,
    /// Wall-clock seconds since the run started.
    Elapsed,
}

impl XMode {
    fn label(self) -> &'static str {
        match self {
            XMode::Iteration => "iteration",
            XMode::Steps => "cumulative env steps",
            XMode::Elapsed => "wall clock (s)",
        }
    }
}

/// Top-level layout: main dashboard or detailed views.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum ViewMode {
    Main,
    Detail,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum DetailTab {
    Chart,
    Heatmap,
    Table,
    Bar,
    Scatter,
}

/// How runs are grouped into the dashboard's tabs.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum GroupMode {
    /// Human model name (`--name`), falling back to the architecture signature
    /// for runs trained before names were used.
    Model,
    /// One tab per `runs/<run_id>` directory.
    Run,
    /// Algorithm + network shape + backend.
    Architecture,
}

impl GroupMode {
    const ALL: [GroupMode; 3] = [GroupMode::Model, GroupMode::Run, GroupMode::Architecture];

    fn label(self) -> &'static str {
        match self {
            GroupMode::Model => "Model",
            GroupMode::Run => "Run",
            GroupMode::Architecture => "Architecture",
        }
    }

    fn hint(self) -> &'static str {
        match self {
            GroupMode::Model => "One tab per --name label (unnamed runs group by architecture)",
            GroupMode::Run => "One tab per training run",
            GroupMode::Architecture => "One tab per algorithm + network shape + backend",
        }
    }
}

/// One selectable tab in the dashboard header.
#[derive(Clone, Debug)]
struct GroupTab {
    key: String,
    label: String,
    count: usize,
    live: bool,
}

struct RunModel {
    dir: PathBuf,
    manifest: Option<RunManifest>,
    records: Vec<MetricsRecord>,
    /// Byte offset of the last read in `metrics.jsonl` (incremental tail).
    file_len: u64,
    /// Inode of `metrics.jsonl` at the last read (Unix only). A different
    /// inode at the same path means a writer renamed a fresh file over it.
    file_ino: Option<u64>,
    /// Modification time of `metrics.jsonl` at the last read. Catches an
    /// in-place rewrite that keeps both the path and the length.
    file_mtime: Option<SystemTime>,
    error: Option<String>,
}

/// Inode for replacement detection, on Unix only.
#[cfg(unix)]
fn inode_of(meta: &std::fs::Metadata) -> Option<u64> {
    use std::os::unix::fs::MetadataExt;
    Some(meta.ino())
}

/// Non-Unix filesystems have no stable inode: rely on mtime alone.
#[cfg(not(unix))]
fn inode_of(_meta: &std::fs::Metadata) -> Option<u64> {
    None
}

/// Read + parse `run.json`; missing/unparseable → `None`.
fn read_manifest(dir: &Path) -> Option<RunManifest> {
    std::fs::read_to_string(dir.join("run.json"))
        .ok()
        .and_then(|s| serde_json::from_str::<RunManifest>(&s).ok())
}

impl RunModel {
    fn new(dir: PathBuf) -> Self {
        let manifest = read_manifest(&dir);
        Self {
            dir,
            manifest,
            records: Vec::new(),
            file_len: 0,
            file_ino: None,
            file_mtime: None,
            error: None,
        }
    }

    /// Re-read `run.json` (status may have flipped to `completed` since we
    /// last looked) without touching the metrics we've already loaded.
    fn reload_manifest(&mut self) {
        if let Some(m) = read_manifest(&self.dir) {
            self.manifest = Some(m);
        }
    }

    /// Append whatever is new since the last read. Truncation restarts from 0,
    /// as does a replaced or rewritten file (detected by inode/mtime).
    fn poll(&mut self) {
        let path = self.dir.join("metrics.jsonl");
        let mut f = match std::fs::File::open(&path) {
            Ok(f) => f,
            Err(e) if e.kind() == std::io::ErrorKind::NotFound && self.records.is_empty() => {
                // Brand-new run whose metrics.jsonl hasn't been created yet —
                // benign (discovery already treats run.json-only dirs as runs).
                return;
            }
            Err(_) => {
                // If we have records already, don't clear them on transient errors
                if self.records.is_empty() {
                    self.error = Some("metrics.jsonl not found or unreadable".into());
                }
                return;
            }
        };
        let Ok(meta) = f.metadata() else {
            return;
        };
        let len = meta.len();
        let ino = inode_of(&meta);
        let mtime = meta.modified().ok();

        // A same-length file can still be a different file: a writer that
        // renames a fresh `metrics.jsonl` over the path, or rewrites it in
        // place, leaves `len == file_len`. The inode catches the rename and
        // the mtime catches the rewrite; a shortened file needs no detective
        // work. In all three cases the old rows are stale, so start over.
        let replaced = matches!((ino, self.file_ino), (Some(new), Some(old)) if new != old);
        let truncated = len < self.file_len;
        let rewritten = len == self.file_len && mtime != self.file_mtime;
        if replaced || truncated || rewritten {
            self.file_len = 0;
            self.records.clear();
        }
        self.file_ino = ino;
        self.file_mtime = mtime;
        if len == self.file_len {
            return;
        }
        if f.seek(SeekFrom::Start(self.file_len)).is_err() {
            return;
        }
        let mut buf = String::new();
        if f.read_to_string(&mut buf).is_err() {
            if self.records.is_empty() {
                self.error = Some("failed to read metrics.jsonl".into());
            }
            return;
        }
        // Only consume *complete* lines: a read can race a partially-flushed
        // write, and re-reading the tail next poll is free (seek + tiny read).
        let (complete, consumed) = if buf.ends_with('\n') {
            (buf.as_str(), buf.len())
        } else {
            match buf.rfind('\n') {
                Some(i) => (&buf[..=i], i + 1),
                None => ("", 0), // only a partial line so far — retry next poll
            }
        };
        if consumed > 0 {
            self.file_len += consumed as u64;
        }
        self.error = None;
        for line in complete.lines() {
            let line = line.trim();
            if line.is_empty() {
                continue;
            }
            if let Ok(rec) = serde_json::from_str::<MetricsRecord>(line) {
                self.records.push(rec);
            }
        }
        if self.records.len() > MAX_RECORDS {
            let drop = self.records.len() - MAX_RECORDS / 2;
            self.records.drain(..drop);
        }
    }

        /// Quick sparkline (last n points) for a metric in iteration mode
    fn sparkline(&self, key: &str, n: usize) -> Vec<f64> {
        let s = self.series_for(key, XMode::Iteration);
        let start = if s.len() > n { s.len() - n } else { 0 };
        s[start..].iter().map(|p| p[1]).collect()
    }

    /// Latest value for a metric
    fn latest(&self, key: &str) -> Option<f64> {
        self.series_for(key, XMode::Iteration).last().map(|p| p[1])
    }

    /// Best (max) value for a metric
    fn best(&self, key: &str) -> Option<f64> {
        let s = self.series_for(key, XMode::Iteration);
        if s.is_empty() {
            None
        } else {
            Some(s.iter().map(|p| p[1]).fold(f64::NEG_INFINITY, f64::max))
        }
    }

    /// X-coordinate for every logged record under the selected axis mode.
    /// `Steps` reconstructs cumulative environment steps from the logged
    /// throughput × wall-clock deltas (the writer logs no per-iteration step
    /// counts); a missing clock/rate falls back to a 1 s slot.
    fn x_values(&self, mode: XMode) -> Vec<f64> {
        match mode {
            XMode::Iteration => self.records.iter().map(|r| r.iteration as f64).collect(),
            XMode::Elapsed => self
                .records
                .iter()
                .map(|r| r.elapsed_s.unwrap_or(r.iteration as f64))
                .collect(),
            XMode::Steps => {
                let mut cum = 0.0;
                let mut prev: Option<f64> = None;
                self.records
                    .iter()
                    .map(|r| {
                        let rate = r.steps_per_s.unwrap_or(0.0);
                        let dt = match (r.elapsed_s, prev) {
                            (Some(e), Some(p)) => (e - p).max(0.0),
                            (Some(e), None) => e,
                            _ => 1.0,
                        };
                        prev = r.elapsed_s;
                        cum += rate * dt;
                        cum
                    })
                    .collect()
            }
        }
    }

    /// Value series for one metric in the given axis mode. `episodes_completed`
    /// is cumulative across the run (the writer logs per-iteration counts).
    fn series_for(&self, key: &str, mode: XMode) -> Vec<[f64; 2]> {
        let xs = self.x_values(mode);
        let mut cum_episodes = 0.0;
        self.records
            .iter()
            .zip(&xs)
            .filter_map(|(r, &x)| {
                if key == "episodes_completed" {
                    let n = r.episodes_completed?;
                    cum_episodes += n as f64;
                    Some([x, cum_episodes])
                } else {
                    r.metric(key).map(|v| [x, v])
                }
            })
            .collect()
    }

    /// X positions (current mode) of every eval iteration (vertical guides).
    fn eval_xs(&self, mode: XMode) -> Vec<f64> {
        self.x_values(mode)
            .iter()
            .zip(&self.records)
            .filter(|(_, r)| r.eval.is_some())
            .map(|(&x, _)| x)
            .collect()
    }

    /// X positions (current mode) of every checkpoint write (vertical guides).
    fn checkpoint_xs(&self, mode: XMode) -> Vec<f64> {
        self.x_values(mode)
            .iter()
            .zip(&self.records)
            .filter(|(_, r)| r.checkpoint)
            .map(|(&x, _)| x)
            .collect()
    }

    /// `(x, total_reward)` for every eval — the sparse dots overlaid on the
    /// reward curve.
    fn eval_reward_dots(&self, mode: XMode) -> Vec<[f64; 2]> {
        self.x_values(mode)
            .iter()
            .zip(&self.records)
            .filter_map(|(&x, r)| r.eval.as_ref().and_then(|e| e.total_reward).map(|v| [x, v]))
            .collect()
    }

    /// Config identity for seed-replication grouping: algo, backend, network,
    /// algorithm, and trainer config (the `seed` value excluded). Runs sharing
    /// it are treated as replications of the same experiment. `None` for
    /// manifests that lack the config blocks (those stay individual lines).
    fn group_key(&self) -> Option<String> {
        let m = self.manifest.as_ref()?;
        let mut trainer = m.trainer.clone()?;
        if let Some(obj) = trainer.as_object_mut() {
            obj.remove("seed");
        }
        let algo = m.algo.clone().unwrap_or_default();
        let backend = m.backend.clone().unwrap_or_default();
        let network = m.network.as_ref().map(|v| v.to_string()).unwrap_or_default();
        let algorithm = m.algorithm.as_ref().map(|v| v.to_string()).unwrap_or_default();
        Some(format!("{algo}|{backend}|{network}|{algorithm}|{trainer}"))
    }

    /// Directory name (`runs/<run_id>`), the dashboard's stable run identity.
    fn run_id(&self) -> String {
        self.dir
            .file_name()
            .map(|s| s.to_string_lossy().into_owned())
            .unwrap_or_else(|| self.dir.to_string_lossy().into_owned())
    }

    /// Human model label passed to `rl_agent --name`, if the manifest kept it.
    fn explicit_model_name(&self) -> Option<String> {
        let args = &self.manifest.as_ref()?.command_line;
        for (i, arg) in args.iter().enumerate() {
            let value = if arg == "--name" {
                args.get(i + 1).map(String::as_str)
            } else {
                arg.strip_prefix("--name=")
            };
            if let Some(v) = value {
                let v = v.trim();
                if !v.is_empty() {
                    return Some(v.to_string());
                }
            }
        }
        None
    }

    /// Older manifests may not expose `--name` cleanly; the label is also the
    /// suffix of `..._<algo>_seed<digits>_<label>`.
    fn run_id_label(&self) -> Option<String> {
        let id = self.run_id();
        let after = &id[id.rfind("_seed")? + 5..];
        let digits = after.find(|c: char| !c.is_ascii_digit())?;
        let label = after[digits..].strip_prefix('_')?;
        (!label.is_empty()).then(|| label.to_string())
    }

    fn hidden_label(&self) -> Option<String> {
        match self.manifest.as_ref()?.network.as_ref()?.get("hidden")? {
            serde_json::Value::Number(n) => Some(n.to_string()),
            serde_json::Value::Array(layers) => {
                Some(layers.iter().map(|v| v.to_string()).collect::<Vec<_>>().join("×"))
            }
            _ => None,
        }
    }

    /// Compact config signature used when a run has no explicit name.
    fn architecture_label(&self) -> String {
        let m = self.manifest.as_ref();
        let algo = m.and_then(|m| m.algo.as_deref()).unwrap_or("unknown");
        let backend = short_backend(m.and_then(|m| m.backend.as_deref()));
        match self.hidden_label() {
            Some(h) => format!("{algo} · h{h} · {backend}"),
            None => format!("{algo} · {backend}"),
        }
    }

    /// Tab key for the selected grouping mode.
    fn tab_key(&self, mode: GroupMode) -> String {
        match mode {
            GroupMode::Model => self
                .explicit_model_name()
                .or_else(|| self.run_id_label())
                .unwrap_or_else(|| self.architecture_label()),
            GroupMode::Run => self.run_id(),
            GroupMode::Architecture => self.architecture_label(),
        }
    }

    /// Total wall-clock seconds at the last record (`None` when unlogged).
    fn last_elapsed(&self) -> Option<f64> {
        self.records.last().and_then(|r| r.elapsed_s)
    }

    /// Mean `steps_per_s` across records that carry the field.
    fn mean_sps(&self) -> Option<f64> {
        let vals: Vec<f64> = self.records.iter().filter_map(|r| r.steps_per_s).collect();
        if vals.is_empty() {
            None
        } else {
            Some(vals.iter().sum::<f64>() / vals.len() as f64)
        }
    }

    /// Cumulative environment steps at the last record (via [`Self::x_values`]).
    fn total_env_steps(&self) -> Option<f64> {
        self.x_values(XMode::Steps).last().copied()
    }

    fn status(&self) -> &str {
        self.manifest
            .as_ref()
            .and_then(|m| m.status.as_deref())
            .unwrap_or("unknown")
    }

    fn is_live(&self) -> bool {
        self.status() == "running"
    }
}

/// Run ids sort lexicographically by start time, so the last entry is the
/// current/most recent run (same convention the Godot panel uses).
fn discover_runs(root: &Path) -> Vec<PathBuf> {
    let Ok(entries) = std::fs::read_dir(root) else {
        return Vec::new();
    };
    let mut dirs: Vec<PathBuf> = entries
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| p.is_dir() && (p.join("metrics.jsonl").exists() || p.join("run.json").exists()))
        .collect();
    dirs.sort_by(|a, b| {
        let an = a.file_name().unwrap_or_default().to_string_lossy();
        let bn = b.file_name().unwrap_or_default().to_string_lossy();
        an.cmp(&bn)
    });
    dirs
}

/// Exponential smoothing (EMA) over the value column; `alpha = 1/(window+1)`,
/// `window = 0` disables smoothing. Keeps sparse points (eval) as-is.
fn smooth(pts: &[[f64; 2]], window: u32) -> Vec<[f64; 2]> {
    if window == 0 {
        return pts.to_vec();
    }
    let alpha = 1.0 / (window as f64 + 1.0);
    let mut out = Vec::with_capacity(pts.len());
    let mut ema = pts.first().map(|p| p[1]).unwrap_or(0.0);
    for p in pts {
        ema = ema + alpha * (p[1] - ema);
        out.push([p[0], ema]);
    }
    out
}

/// Running max (or running min for loss metrics) — the "best-so-far" envelope.
fn best_so_far(pts: &[[f64; 2]], lower_better: bool) -> Vec<[f64; 2]> {
    let mut best = if lower_better {
        f64::INFINITY
    } else {
        f64::NEG_INFINITY
    };
    let mut out = Vec::with_capacity(pts.len());
    for &p in pts {
        best = if lower_better { best.min(p[1]) } else { best.max(p[1]) };
        out.push([p[0], best]);
    }
    out
}

/// Metrics where a *lower* value is better (drives the best-so-far direction).
fn lower_is_better(key: &str) -> bool {
    matches!(key, "policy_loss" | "value_loss")
}

/// Map a series through log10(y) for the log-scale toggle; y ≤ 0 is dropped
/// (the logarithm is undefined there).
fn log10_pts(pts: &[[f64; 2]]) -> Vec<[f64; 2]> {
    pts.iter()
        .filter(|p| p[1].is_finite() && p[1] > 0.0)
        .map(|&[x, y]| [x, y.log10()])
        .collect()
}

/// Linear interpolation between two colors (heatmap scale).
fn lerp_color(a: egui::Color32, b: egui::Color32, t: f32) -> egui::Color32 {
    let t = t.clamp(0.0, 1.0);
    let mix = |x: u8, y: u8| (x as f32 + (y as f32 - x as f32) * t).round() as u8;
    egui::Color32::from_rgb(mix(a.r(), b.r()), mix(a.g(), b.g()), mix(a.b(), b.b()))
}

/// Heatmap cell color for a normalized value `t ∈ [0, 1]`: cold → warm → hot.
fn value_color(t: f64) -> egui::Color32 {
    let (cold, mid, hot) = (
        egui::Color32::from_rgb(24, 62, 120),
        egui::Color32::from_rgb(240, 200, 80),
        egui::Color32::from_rgb(230, 80, 56),
    );
    let t = t.clamp(0.0, 1.0) as f32;
    if t < 0.5 {
        lerp_color(cold, mid, t * 2.0)
    } else {
        lerp_color(mid, hot, (t - 0.5) * 2.0)
    }
}

/// Short human label for a burn backend path (`..::WgpuDevice` → `WgpuDevice`).
fn short_backend(backend: Option<&str>) -> &str {
    backend
        .and_then(|b| b.rsplit("::").next())
        .filter(|s| !s.is_empty())
        .unwrap_or("?")
}

/// Compact tab label for a run id:
/// `20260923_104623_870_ppo_seed42_smoke` → `10:46:23 · ppo_seed42_smoke`.
fn short_run_label(id: &str) -> String {
    let mut parts = id.splitn(4, '_');
    let (Some(date), Some(time), Some(_millis), Some(rest)) =
        (parts.next(), parts.next(), parts.next(), parts.next())
    else {
        return id.to_string();
    };
    if date.len() == 8 && time.len() == 6 && time.chars().all(|c| c.is_ascii_digit()) {
        format!("{}:{}:{} · {rest}", &time[0..2], &time[2..4], &time[4..6])
    } else {
        id.to_string()
    }
}

/// `seconds` → `3m 24s` / `1h 05m`, `—` when unknown.
fn fmt_duration(secs: Option<f64>) -> String {
    let Some(s) = secs.filter(|s| s.is_finite() && *s >= 0.0) else {
        return "—".into();
    };
    let s = s.round() as u64;
    let (h, m, s) = (s / 3600, (s % 3600) / 60, s % 60);
    if h > 0 {
        format!("{h}h {m:02}m")
    } else if m > 0 {
        format!("{m}m {s:02}s")
    } else {
        format!("{s}s")
    }
}

/// Tick formatter: plain but not absurdly long.
fn fmt(v: f64) -> String {
    let a = v.abs();
    if !v.is_finite() {
        "—".into()
    } else if a >= 10_000.0 {
        format!("{v:.0}")
    } else if a >= 100.0 {
        format!("{v:.1}")
    } else if a >= 1.0 {
        format!("{v:.3}")
    } else {
        format!("{v:.4}")
    }
}

// ── The app ──────────────────────────────────────────────────────────

use std::time::Instant;

struct DashApp {
    runs_root: PathBuf,
    /// keyed by run id, kept sorted (BTreeMap iterates in run-id/time order)
    runs: BTreeMap<String, RunModel>,
    visible: BTreeMap<String, bool>,
    metric: usize,
    window: u32,
    follow_live: bool,
    banner: Option<String>,
    /// Polls since the last runs-dir rescan (auto-discovery of new runs).
    polls_since_rescan: u8,
    /// Layout + chart switches (see the side panel).
    view: ViewMode,
    detail_tab: DetailTab,
    /// Tab grouping mode and selected group (`None` = the All tab).
    group_mode: GroupMode,
    active_group: Option<String>,
    x_mode: XMode,
    log_y: bool,
    show_best: bool,
    show_evals: bool,
    show_checkpoints: bool,
    seed_bands: bool,
    show_summary: bool,
    /// Last time we polled (for status bar)
    last_poll: Instant,
    /// Track last modification time per run for staleness detection
    last_update: BTreeMap<String, Instant>,
    /// Run list sorting
    sort_by: SortKey,
    sort_desc: bool,
    /// Filters
    filter_text: String,
    hide_completed: bool,
    hide_interrupted: bool,
    /// KPI strip
    show_kpi: bool,
    kpi_expanded: Option<String>,
}

/// Sort key for run list
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum SortKey {
    Name,
    Iters,
    LatestReward,
    Status,
}

impl DashApp {
    fn new(runs_root: PathBuf, metric: usize) -> Self {
        let mut app = Self {
            runs_root,
            runs: BTreeMap::new(),
            visible: BTreeMap::new(),
            metric,
            window: 0,
            follow_live: true,
            banner: None,
            polls_since_rescan: 0,
            view: ViewMode::Main,
            detail_tab: DetailTab::Chart,
            group_mode: GroupMode::Model,
            active_group: None,
            x_mode: XMode::Iteration,
            log_y: false,
            show_best: false,
            show_evals: false,
            show_checkpoints: false,
            seed_bands: false,
            show_summary: true,
            last_poll: Instant::now(),
            last_update: BTreeMap::new(),
            sort_by: SortKey::Name,
            sort_desc: false,
            filter_text: String::new(),
            hide_completed: false,
            hide_interrupted: false,
            show_kpi: true,
            kpi_expanded: None,
        };
        app.rescan();
        app
    }

    fn rescan(&mut self) {
        let dirs = discover_runs(&self.runs_root);
        if dirs.is_empty() {
            self.banner = Some(format!(
                "No runs found in {} — start training (rl_agent) or pass --runs <dir>.",
                self.runs_root.display()
            ));
        } else {
            self.banner = None;
        }

        // Keep existing models (their loaded records survive), add new ones,
        // and refresh manifests so a finished run's status shows up.
        let mut seen = BTreeMap::new();
        let mut new_ids: Vec<String> = Vec::new();
        for dir in dirs {
            let id = dir
                .file_name()
                .map(|s| s.to_string_lossy().into_owned())
                .unwrap_or_else(|| dir.to_string_lossy().into_owned());
            seen.insert(id.clone(), ());
            if !self.runs.contains_key(&id) {
                self.visible.insert(id.clone(), true);
                self.runs.insert(id.clone(), RunModel::new(dir));
                new_ids.push(id);
            }
        }
        for (id, run) in self.runs.iter_mut() {
            if seen.contains_key(id) {
                run.reload_manifest();
            }
        }

        // Poll runs that appeared on this scan immediately. Waiting for the
        // next UI tick made a run that had already written records look empty
        // on the tick it was discovered.
        for id in &new_ids {
            let has_records = if let Some(run) = self.runs.get_mut(id) {
                run.poll();
                !run.records.is_empty()
            } else {
                false
            };
            if has_records {
                self.last_update.insert(id.clone(), Instant::now());
            }
        }

        // Drop runs that disappeared from disk.
        let gone: Vec<String> = self
            .runs
            .keys()
            .filter(|k| !seen.contains_key(*k))
            .cloned()
            .collect();
        for k in gone {
            self.runs.remove(&k);
            self.visible.remove(&k);
        }
    }

    fn poll_all(&mut self) {
        for (id, run) in self.runs.iter_mut() {
            let before = run.records.len();
            run.poll();
            if run.records.len() != before {
                self.last_update.insert(id.clone(), Instant::now());
            }
        }
        // Discover runs that landed after launch (the RL tab's trainer spawns
        // its run dir within the first second or two) without a manual click.
        self.polls_since_rescan += 1;
        if self.polls_since_rescan >= RESCAN_EVERY_POLLS {
            self.polls_since_rescan = 0;
            self.rescan();
        }
        if self.follow_live {
            if let Some(newest) = self.active_run_ids().last().cloned() {
                self.visible.insert(newest, true);
            }
        }
        self.last_poll = Instant::now();
    }

    fn newest_id(&self) -> Option<String> {
        self.active_run_ids().last().cloned()
    }

    /// True when a run belongs to the active tab (checkboxes are applied too).
    fn run_in_active_group(&self, run: &RunModel) -> bool {
        self.active_group
            .as_deref()
            .is_none_or(|group| run.tab_key(self.group_mode) == group)
    }

    /// Checked *and* inside the active tab — the filter every view uses.
    fn run_is_selected(&self, id: &str, run: &RunModel) -> bool {
        self.visible.get(id).copied().unwrap_or(false) && self.run_in_active_group(run)
    }

    /// Run ids inside the active tab, in chronological (BTreeMap) order.
    fn active_run_ids(&self) -> Vec<String> {
        self.runs
            .iter()
            .filter(|(_, run)| self.run_in_active_group(run))
            .map(|(id, _)| id.clone())
            .collect()
    }

    /// Tabs for the current grouping mode, ordered by each group's first run.
    fn group_tabs(&self) -> Vec<GroupTab> {
        let mut tabs: Vec<GroupTab> = Vec::new();
        let mut index: BTreeMap<String, usize> = BTreeMap::new();
        for (id, run) in self.runs.iter() {
            let key = run.tab_key(self.group_mode);
            let label = match self.group_mode {
                GroupMode::Run => short_run_label(id),
                _ => key.clone(),
            };
            if let Some(&i) = index.get(&key) {
                tabs[i].count += 1;
                tabs[i].live |= run.is_live();
            } else {
                index.insert(key.clone(), tabs.len());
                tabs.push(GroupTab {
                    key,
                    label,
                    count: 1,
                    live: run.is_live(),
                });
            }
        }
        tabs
    }

    /// Fall back to the All tab when a rescan removes the selected group.
    fn ensure_active_group(&mut self) {
        let Some(group) = self.active_group.clone() else {
            return;
        };
        if !self.runs.values().any(|run| run.tab_key(self.group_mode) == group) {
            self.active_group = None;
        }
    }

    fn metric_key(&self) -> &'static str {
        METRICS[self.metric].0
    }

    fn metric_label(&self) -> &'static str {
        METRICS[self.metric].1
    }
}

impl eframe::App for DashApp {
    // egui ≥0.35 hands the app a root `Ui` instead of a bare `Context`.
    fn ui(&mut self, ui: &mut egui::Ui, _frame: &mut eframe::Frame) {
        let ctx = ui.ctx().clone();
        self.poll_all();
        self.ensure_active_group();
        ctx.request_repaint_after(Duration::from_millis(POLL_MS));

        // Keyboard shortcuts
        if ctx.input(|i| i.key_pressed(egui::Key::Space)) {
            self.follow_live = !self.follow_live;
        }
        if ctx.input(|i| i.key_pressed(egui::Key::R)) {
            self.rescan();
        }
        if ctx.input(|i| i.key_pressed(egui::Key::H)) {
            self.view = match self.view {
                ViewMode::Main => ViewMode::Detail,
                ViewMode::Detail => ViewMode::Main,
            };
        }
        if ctx.input(|i| i.key_pressed(egui::Key::ArrowLeft)) {
            if self.metric == 0 {
                self.metric = METRICS.len() - 1;
            } else {
                self.metric -= 1;
            }
        }
        if ctx.input(|i| i.key_pressed(egui::Key::ArrowRight)) {
            self.metric = (self.metric + 1) % METRICS.len();
        }

        egui::Panel::left("runs")
            .resizable(true)
            .default_size(320.0)
            .show(ui, |ui| self.side_panel(ui));

        egui::Panel::bottom("status").show(ui, |ui| {
            ui.horizontal(|ui| {
                // System resource monitor (CPU / RAM)
                let mut sys = System::new_all();
                sys.refresh_memory();
                let cpu_usage = if let Some(c) = sys.cpus().first() { c.cpu_usage() } else { 0.0 };
                let ram_used = sys.total_memory() - sys.available_memory();
                let ram_total = sys.total_memory();
                ui.add(egui::ProgressBar::new(cpu_usage as f32 / 100.0).text(format!("CPU {:.0}%", cpu_usage)));
                ui.add(egui::ProgressBar::new(ram_used as f32 / ram_total as f32).text(format!("RAM {:.0}MB / {:.0}MB", ram_used / 1024 / 1024, ram_total / 1024 / 1024)));

                let tab_ids = self.active_run_ids();
                let tab_total = tab_ids.len();
                let tab_checked = tab_ids
                    .iter()
                    .filter(|id| self.visible.get(*id).copied().unwrap_or(false))
                    .count();
                let checked_total = self.visible.values().filter(|&&v| v).count();
                let total = self.runs.len();
                let scope = self.active_group.as_deref().unwrap_or("All");
                ui.label(
                    egui::RichText::new(format!(
                        "Runs: {tab_checked}/{tab_total} in tab • {checked_total} checked / {total} total • Tab: {scope} • Last poll: {}ms ago",
                        self.last_poll.elapsed().as_millis()
                    ))
                    .small()
                    .weak(),
                );
                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    ui.label(
                        egui::RichText::new("Space=follow • R=rescan • H=Main/Detail • ←/→=metric")
                            .small()
                            .weak(),
                    );
                });
            });
        });

        egui::CentralPanel::default().show(ui, |ui| {
            self.group_tab_bar(ui);
            ui.separator();
            ui.horizontal(|ui| {
                ui.selectable_value(&mut self.view, ViewMode::Main, "Main Dashboard");
                ui.selectable_value(&mut self.view, ViewMode::Detail, "Detailed Analysis");
            });
            ui.separator();
            if self.show_kpi && self.view == ViewMode::Main {
                self.kpi_strip(ui);
                ui.separator();
            }
            if self.view == ViewMode::Main {
                // Main screen: compact overview - show chart + quick summaries
                ui.horizontal(|ui| {
                    ui.heading("Overview");
                    ui.label(egui::RichText::new("Click 'Detailed Analysis' for Heatmap/Table/Bar/Scatter").small().weak());
                });
                self.plot_panel(ui);
            } else {
                ui.horizontal_wrapped(|ui| {
                    ui.selectable_value(&mut self.detail_tab, DetailTab::Chart, "Chart");
                    ui.selectable_value(&mut self.detail_tab, DetailTab::Heatmap, "Heatmap");
                    ui.selectable_value(&mut self.detail_tab, DetailTab::Table, "Table");
                    ui.selectable_value(&mut self.detail_tab, DetailTab::Bar, "Bar");
                    ui.selectable_value(&mut self.detail_tab, DetailTab::Scatter, "Scatter");
                });
                ui.separator();
                match self.detail_tab {
                    DetailTab::Chart => self.plot_panel(ui),
                    DetailTab::Heatmap => self.heatmap_panel(ui),
                    DetailTab::Table => self.table_panel(ui),
                    DetailTab::Bar => self.bar_panel(ui),
                    DetailTab::Scatter => self.scatter_panel(ui),
                }
            }
        });
    }
}

impl DashApp {
    /// Horizontal tab bar: All + one tab per group in the selected mode.
    fn group_tab_bar(&mut self, ui: &mut egui::Ui) {
        let tabs = self.group_tabs();
        let active = self.active_group.clone();
        let total = self.runs.len();
        let mut pick: Option<Option<String>> = None;
        let mut mode = self.group_mode;

        ui.horizontal(|ui| {
            ui.label(egui::RichText::new("Tabs").small().weak());
            egui::ComboBox::from_id_salt("group_mode")
                .selected_text(mode.label())
                .width(120.0)
                .show_ui(ui, |ui| {
                    for m in GroupMode::ALL {
                        ui.selectable_value(&mut mode, m, m.label())
                            .on_hover_text(m.hint());
                    }
                });
            ui.separator();
            if ui
                .selectable_label(active.is_none(), format!("All ({total})"))
                .on_hover_text("Show every checked run")
                .clicked()
            {
                pick = Some(None);
            }
            egui::ScrollArea::horizontal()
                .id_salt("group_tabs")
                .show(ui, |ui| {
                    ui.horizontal(|ui| {
                        ui.spacing_mut().item_spacing.x = 4.0;
                        for tab in &tabs {
                            let selected = active.as_deref() == Some(tab.key.as_str());
                            let text = if tab.live {
                                format!("● {} ({})", tab.label, tab.count)
                            } else {
                                format!("{} ({})", tab.label, tab.count)
                            };
                            if ui
                                .selectable_label(selected, text)
                                .on_hover_text(format!(
                                    "{} · {} run(s) · grouped by {}",
                                    tab.label,
                                    tab.count,
                                    mode.label().to_lowercase()
                                ))
                                .clicked()
                            {
                                pick = Some(Some(tab.key.clone()));
                            }
                        }
                    });
                });
        });

        if mode != self.group_mode {
            self.group_mode = mode;
            self.active_group = None;
        } else if let Some(sel) = pick {
            self.active_group = sel;
        }
    }

    fn side_panel(&mut self, ui: &mut egui::Ui) {
        ui.add_space(4.0);
        ui.heading("rl_dash");
        ui.label(
            egui::RichText::new(format!("runs: {}", self.runs_root.display()))
                .weak()
                .small(),
        );
        ui.separator();

        egui::ComboBox::from_label("Metric")
            .selected_text(self.metric_label())
            .width(260.0)
            .show_ui(ui, |ui| {
                for (i, (_, label)) in METRICS.iter().enumerate() {
                    ui.selectable_value(&mut self.metric, i, *label);
                }
            });

        ui.horizontal(|ui| {
            ui.label("View:");
            ui.selectable_value(&mut self.view, ViewMode::Main, "Main");
            ui.selectable_value(&mut self.view, ViewMode::Detail, "Detail");
        });

        egui::ComboBox::from_label("X-axis")
            .selected_text(match self.x_mode {
                XMode::Iteration => "Iteration",
                XMode::Steps => "Env steps",
                XMode::Elapsed => "Wall clock",
            })
            .width(260.0)
            .show_ui(ui, |ui| {
                ui.selectable_value(&mut self.x_mode, XMode::Iteration, "Iteration");
                ui.selectable_value(&mut self.x_mode, XMode::Steps, "Env steps");
                ui.selectable_value(&mut self.x_mode, XMode::Elapsed, "Wall clock");
            });

        let smooth_label = if self.window == 0 {
            "Smoothing: off".to_string()
        } else {
            format!("Smoothing: window {}", self.window)
        };
        ui.add(egui::Slider::new(&mut self.window, 0..=50).text(smooth_label));

        ui.horizontal_wrapped(|ui| {
            ui.checkbox(&mut self.show_best, "Best-so-far");
            ui.checkbox(&mut self.log_y, "Log y");
            ui.checkbox(&mut self.show_evals, "Evaluations");
            ui.checkbox(&mut self.show_checkpoints, "Checkpoints");
        });
        ui.horizontal_wrapped(|ui| {
            ui.checkbox(&mut self.seed_bands, "Seed bands");
            ui.checkbox(&mut self.show_summary, "Run summary");
        });

        ui.horizontal_wrapped(|ui| {
            let follow = &mut self.follow_live;
            ui.checkbox(follow, "Follow live run");
            ui.checkbox(&mut self.show_kpi, "KPIs");
            if ui.button("Rescan").clicked() {
                self.rescan();
            }
            if ui.button("Clear data").clicked() {
                self.runs.clear();
                self.visible.clear();
                self.last_update.clear();
                self.active_group = None;
                self.banner = Some("Data cleared. Click Rescan to reload runs from disk.".to_string());
            }
            if ui.button("Select all").clicked() {
                for id in self.active_run_ids() {
                    self.visible.insert(id, true);
                }
            }
            if ui.button("Uncheck all").clicked() {
                for id in self.active_run_ids() {
                    self.visible.insert(id, false);
                }
            }
        });

        if let Some(banner) = self.banner.clone() {
            ui.colored_label(egui::Color32::YELLOW, banner);
        }

        ui.separator();
        ui.collapsing("Sort & Filter", |ui| {
            egui::ComboBox::from_label("Sort by")
                .selected_text(match self.sort_by {
                    SortKey::Name => "Name",
                    SortKey::Iters => "Iterations",
                    SortKey::LatestReward => "Latest reward",
                    SortKey::Status => "Status",
                })
                .show_ui(ui, |ui| {
                    ui.selectable_value(&mut self.sort_by, SortKey::Name, "Name");
                    ui.selectable_value(&mut self.sort_by, SortKey::Iters, "Iterations");
                    ui.selectable_value(&mut self.sort_by, SortKey::LatestReward, "Latest reward");
                    ui.selectable_value(&mut self.sort_by, SortKey::Status, "Status");
                });
            ui.checkbox(&mut self.sort_desc, "Descending");
            ui.text_edit_singleline(&mut self.filter_text);
            ui.horizontal_wrapped(|ui| {
                ui.checkbox(&mut self.hide_completed, "Hide completed");
                ui.checkbox(&mut self.hide_interrupted, "Hide interrupted");
            });
        });
        ui.separator();
        ui.label(egui::RichText::new("Runs").strong());
        if let Some(group) = &self.active_group {
            ui.label(
                egui::RichText::new(format!("tab: {group}"))
                    .small()
                    .weak(),
            );
        }

        egui::ScrollArea::vertical().show(ui, |ui| {
            if self.runs.is_empty() {
                ui.label(
                    egui::RichText::new("No runs found.\n\nStart training with rl_agent or click Rescan.\nCheck the runs path above.")
                        .weak()
                        .small(),
                );
                return;
            }
            let newest = self.newest_id();
            let filter_lower = self.filter_text.to_lowercase();
            let mut ids: Vec<String> = self.runs.keys().cloned().collect();
            ids.retain(|id| {
                let Some(run) = self.runs.get(id) else {
                    return false;
                };
                if !self.run_in_active_group(run) {
                    return false;
                }
                if !self.filter_text.is_empty() && !id.to_lowercase().contains(&filter_lower) {
                    return false;
                }
                let status = run.status();
                if self.hide_completed && status == "completed" {
                    return false;
                }
                if self.hide_interrupted && status == "interrupted" {
                    return false;
                }
                true
            });
            ids.sort_by(|a, b| {
                let ord = match self.sort_by {
                    SortKey::Name => a.cmp(b),
                    SortKey::Iters => {
                        let la = self.runs.get(a).map(|r| r.records.len()).unwrap_or(0);
                        let lb = self.runs.get(b).map(|r| r.records.len()).unwrap_or(0);
                        la.cmp(&lb)
                    }
                    SortKey::LatestReward => {
                        let fa = self
                            .runs
                            .get(a)
                            .and_then(|r| r.series_for("mean_reward_per_step", XMode::Iteration).last().map(|p| p[1]))
                            .unwrap_or(f64::NEG_INFINITY);
                        let fb = self
                            .runs
                            .get(b)
                            .and_then(|r| r.series_for("mean_reward_per_step", XMode::Iteration).last().map(|p| p[1]))
                            .unwrap_or(f64::NEG_INFINITY);
                        fa.partial_cmp(&fb).unwrap_or(std::cmp::Ordering::Equal)
                    }
                    SortKey::Status => {
                        let sa = self.runs.get(a).map(|r| r.status()).unwrap_or("");
                        let sb = self.runs.get(b).map(|r| r.status()).unwrap_or("");
                        sa.cmp(sb)
                    }
                };
                if self.sort_desc { ord.reverse() } else { ord }
            });
            for id in ids {
                let run = &self.runs[&id];
                let status = run.status().to_string();
                let live = run.is_live();
                let stale_secs = self
                    .last_update
                    .get(&id)
                    .map(|t| t.elapsed().as_secs())
                    .unwrap_or(u64::MAX);
                let color = if live && stale_secs > 30 {
                    egui::Color32::from_rgb(242, 204, 77) // stale live
                } else if live {
                    egui::Color32::from_rgb(115, 217, 134)
                } else if status == "completed" {
                    egui::Color32::from_rgb(150, 170, 200)
                } else {
                    egui::Color32::from_rgb(242, 154, 51)
                };
                let is_newest = Some(id.clone()) == newest;
                let series_len = run.records.len();

                ui.horizontal(|ui| {
                    let vis = self.visible.entry(id.clone()).or_insert(true);
                    ui.checkbox(vis, "");
                    let label = if is_newest {
                        format!("{id} ◀")
                    } else {
                        id.clone()
                    };
                    ui.label(egui::RichText::new(label).monospace());
                    let status_text = if live && stale_secs > 30 {
                        "● stale"
                    } else if live {
                        "● live"
                    } else {
                        status.as_str()
                    };
                    ui.colored_label(color, status_text);
                });
                if run.error.is_none() {
                    ui.label(
                        egui::RichText::new(format!(
                            "    {} · {} pts{}{}",
                            run.manifest
                                .as_ref()
                                .and_then(|m| m.algo.clone())
                                .unwrap_or_else(|| "?".into()),
                            series_len,
                            run.manifest
                                .as_ref()
                                .and_then(|m| m.git_commit.clone())
                                .map(|c| format!(" · {c}"))
                                .unwrap_or_default(),
                            if live && stale_secs < u64::MAX {
                                format!(" · {}s ago", stale_secs)
                            } else {
                                String::new()
                            }
                        ))
                        .weak()
                        .small(),
                    );
                } else if let Some(err) = &run.error {
                    ui.label(egui::RichText::new(format!("    {err}")).weak().small());
                }
            }
        });
    }

    fn plot_panel(&mut self, ui: &mut egui::Ui) {
        let key = self.metric_key();
        let label = self.metric_label();
        let window = self.window;
        let mode = self.x_mode;
        let log_y = self.log_y;
        let lower_better = lower_is_better(key);

        // Pre-built plot items so the seed-band grouping / log / best-so-far
        // transforms live next to each other instead of inline in the show fn.
        #[derive(Clone)]
        struct SeriesItem {
            name: String,
            color: egui::Color32,
            pts: Vec<[f64; 2]>,
            dashed: bool,
            width: f32,
        }
        struct BandItem {
            name: String,
            color: egui::Color32,
            mean: Vec<[f64; 2]>,
            lower: Vec<[f64; 2]>,
            upper: Vec<[f64; 2]>,
        }

        let mut items: Vec<SeriesItem> = Vec::new();
        let mut bands: Vec<BandItem> = Vec::new();
        let mut eval_dots: Vec<(String, egui::Color32, Vec<[f64; 2]>)> = Vec::new();
        let mut eval_xs: Vec<f64> = Vec::new();
        let mut checkpoint_xs: Vec<f64> = Vec::new();
        let mut shown_runs: Vec<(String, egui::Color32, f64)> = Vec::new();
        // Visible runs grouped by manifest config (for mean±std bands).
        type GroupMembers = Vec<(String, egui::Color32, Vec<[f64; 2]>)>;
        let mut groups: BTreeMap<String, GroupMembers> = BTreeMap::new();
        let mut palette_i = 0usize;

        for (id, run) in self.runs.iter() {
            if !self.run_is_selected(id, run) {
                continue;
            }
            let raw = run.series_for(key, mode);
            if raw.is_empty() {
                continue;
            }
            let color = PALETTE[palette_i % PALETTE.len()];
            palette_i += 1;
            if let Some(&p) = raw.last() {
                shown_runs.push((id.clone(), color, p[1]));
            }

            if key == "mean_reward_per_step" && self.show_evals {
                let dots = run.eval_reward_dots(mode);
                if !dots.is_empty() {
                    eval_dots.push((id.clone(), color, dots));
                }
            }
            if self.show_checkpoints {
                checkpoint_xs.extend(run.checkpoint_xs(mode));
            }
            if self.show_evals {
                eval_xs.extend(run.eval_xs(mode));
            }

            let pts = smooth(&raw, window);
            let best = best_so_far(&pts, lower_better);
            if self.seed_bands {
                if let Some(g) = run.group_key() {
                    groups.entry(g).or_default().push((id.clone(), color, pts));
                    continue;
                }
            }
            items.push(SeriesItem {
                name: id.clone(),
                color,
                pts,
                dashed: false,
                width: 2.0,
            });
            if self.show_best && !best.is_empty() {
                items.push(SeriesItem {
                    name: format!("{id} best-so-far"),
                    color,
                    pts: best,
                    dashed: true,
                    width: 1.0,
                });
            }
        }

        // Collapse config groups (<2 members → plain lines; ≥2 → mean±std band).
        for (g, members) in groups {
            if members.len() < 2 {
                for (mid, color, pts) in members {
                    let best = best_so_far(&pts, lower_better);
                    items.push(SeriesItem {
                        name: mid.clone(),
                        color,
                        pts,
                        dashed: false,
                        width: 2.0,
                    });
                    if self.show_best && !best.is_empty() {
                        items.push(SeriesItem {
                            name: format!("{mid} best-so-far"),
                            color,
                            pts: best,
                            dashed: true,
                            width: 1.0,
                        });
                    }
                }
                continue;
            }
            // Bands align members by iteration index (records are one per PPO
            // iteration), with x = mean of the members' x at that index.
            let n = members.iter().map(|(_, _, s)| s.len()).max().unwrap_or(0);
            let (mut mean_pts, mut lower, mut upper) = (Vec::new(), Vec::new(), Vec::new());
            for i in 0..n {
                let vs: Vec<(f64, f64)> = members
                    .iter()
                    .filter_map(|(_, _, s)| s.get(i).map(|&p| (p[0], p[1])))
                    .collect();
                if vs.is_empty() {
                    continue;
                }
                let mean = vs.iter().map(|(_, y)| *y).sum::<f64>() / vs.len() as f64;
                let var = vs.iter().map(|(_, y)| (*y - mean).powi(2)).sum::<f64>() / vs.len() as f64;
                let sd = var.sqrt();
                let x = vs.iter().map(|(x, _)| *x).sum::<f64>() / vs.len() as f64;
                mean_pts.push([x, mean]);
                lower.push([x, mean - sd]);
                upper.push([x, mean + sd]);
            }
            let algo = g.split('|').next().unwrap_or("group");
            let short = if algo.is_empty() { "group" } else { algo };
            bands.push(BandItem {
                name: format!("{short} ± std (n={})", members.len()),
                color: members[0].1,
                mean: mean_pts,
                lower,
                upper,
            });
        }

        if self.show_summary {
            self.summary_table(ui);
            ui.separator();
        }

        ui.horizontal(|ui| {
            ui.heading(label);
            let total = items.len() + bands.len();
            if total == 0 {
                ui.label(
                    egui::RichText::new("  no visible data — tick a run on the left (or train!)")
                        .weak(),
                );
            } else {
                ui.label(
                    egui::RichText::new(format!(
                        "  {} item(s) · x = {}",
                        total,
                        mode.label()
                    ))
                    .weak(),
                );
            }
        });

        if items.is_empty() && bands.is_empty() {
            return;
        }

        let y_axis_label = if log_y {
            format!("log₁₀({label})")
        } else {
            label.to_string()
        };
        // Keep all series for crosshair lookup
        let mut all_series_for_hover: Vec<(String, egui::Color32, Vec<[f64; 2]>)> = Vec::new();
        for band in &bands {
            let mean = if log_y { log10_pts(&band.mean) } else { band.mean.clone() };
            if !mean.is_empty() {
                all_series_for_hover.push((band.name.clone(), band.color, mean));
            }
        }
        for item in &items {
            let pts = if log_y { log10_pts(&item.pts) } else { item.pts.clone() };
            if !pts.is_empty() {
                all_series_for_hover.push((item.name.clone(), item.color, pts));
            }
        }

        let mut hover: Option<egui_plot::PlotPoint> = None;
        let _plot_resp = Plot::new("metrics")
            .legend(Legend::default())
            .x_axis_label(mode.label())
            .y_axis_label(y_axis_label)
            .show(ui, |plot_ui| {
                // Vertical guides first so data draws on top of them.
                for (i, &x) in eval_xs.iter().enumerate() {
                    plot_ui.vline(
                        VLine::new("", x)
                            .id(egui::Id::new(("eval_vline", i)))
                            .color(EVAL_MARKER)
                            .width(1.0)
                            .style(LineStyle::dotted_dense()),
                    );
                }
                for (i, &x) in checkpoint_xs.iter().enumerate() {
                    plot_ui.vline(
                        VLine::new("", x)
                            .id(egui::Id::new(("ckpt_vline", i)))
                            .color(CKPT_MARKER)
                            .width(1.0)
                            .style(LineStyle::dashed_loose()),
                    );
                }
                // Seed-replication bands
                for band in &bands {
                    let upper = if log_y { log10_pts(&band.upper) } else { band.upper.clone() };
                    let lower = if log_y { log10_pts(&band.lower) } else { band.lower.clone() };
                    let mut poly = Vec::with_capacity(upper.len() + lower.len());
                    poly.extend(upper.iter().copied());
                    poly.extend(lower.iter().rev().copied());
                    if poly.len() >= 3 {
                        plot_ui.polygon(
                            Polygon::new(format!("{} band", band.name), PlotPoints::from(poly))
                                .fill_color(band.color.gamma_multiply(0.22))
                                .stroke(egui::Stroke::NONE),
                        );
                    }
                    let mean = if log_y { log10_pts(&band.mean) } else { band.mean.clone() };
                    if !mean.is_empty() {
                        plot_ui.line(
                            Line::new(band.name.clone(), PlotPoints::from(mean))
                                .color(band.color)
                                .width(2.0)
                                .style(LineStyle::dashed_dense()),
                        );
                    }
                }
                // Per-run series
                for item in &items {
                    let pts = if log_y { log10_pts(&item.pts) } else { item.pts.clone() };
                    if pts.is_empty() {
                        continue;
                    }
                    let style = if item.dashed {
                        LineStyle::dotted_loose()
                    } else {
                        LineStyle::Solid
                    };
                    plot_ui.line(
                        Line::new(item.name.clone(), PlotPoints::from(pts))
                            .color(item.color)
                            .width(item.width)
                            .style(style),
                    );
                }
                // Sparse eval dots
                for (id, color, dots) in &eval_dots {
                    let pts = if log_y { log10_pts(dots) } else { dots.clone() };
                    if pts.is_empty() {
                        continue;
                    }
                    plot_ui.points(
                        Points::new(format!("{id} eval"), PlotPoints::from(pts))
                            .color(*color)
                            .radius(4.0)
                            .filled(true),
                    );
                }
                hover = plot_ui.pointer_coordinate();
            });

        if let Some(p) = hover {
            ui.label(
                egui::RichText::new(format!("Crosshair @ {} × {}", fmt(p.x), fmt(p.y)))
                    .monospace()
                    .weak(),
            );
            egui::ScrollArea::vertical()
                .max_height(150.0)
                .show(ui, |ui| {
                    egui::Grid::new("crosshair_values")
                        .striped(true)
                        .spacing([12.0, 2.0])
                        .show(ui, |ui| {
                            ui.strong("Series");
                            ui.strong("Value");
                            ui.end_row();
                            for (name, color, pts) in &all_series_for_hover {
                                // Find nearest point by x
                                if let Some(best) = pts.iter().min_by(|a, b| {
                                    ((a[0] - p.x).abs())
                                        .partial_cmp(&((b[0] - p.x).abs()))
                                        .unwrap_or(std::cmp::Ordering::Equal)
                                }) {
                                    ui.colored_label(*color, name);
                                    ui.label(fmt(best[1]));
                                    ui.end_row();
                                }
                            }
                        });
                });
        } else {
            ui.label(
                egui::RichText::new("Hover for crosshair values across all series • drag to pan • scroll to zoom")
                    .weak()
                    .small(),
            );
        }

        // Latest-value strip for the visible runs.
        if !shown_runs.is_empty() {
            ui.separator();
            ui.label(egui::RichText::new("Latest values").strong());
            ui.horizontal_wrapped(|ui| {
                for (id, color, v) in &shown_runs {
                    ui.colored_label(*color, format!("{id}: {}", fmt(*v)));
                }
            });
        }

        // Run details for the first visible run (its manifest).
        if let Some((id, _, _)) = shown_runs.first() {
            if let Some(run) = self.runs.get(id) {
                if let Some(m) = &run.manifest {
                    ui.separator();
                    ui.collapsing(format!("Details · {id}"), |ui| {
                        ui.monospace(format!(
                            "status    {}\nstarted   {}\nended     {}\nalgo      {}\nbackend   {}\ngit       {}",
                            m.status.as_deref().unwrap_or("?"),
                            m.started_utc.as_deref().unwrap_or("?"),
                            m.ended_utc.as_deref().unwrap_or("—"),
                            m.algo.as_deref().unwrap_or("?"),
                            m.backend.as_deref().unwrap_or("?"),
                            m.git_commit.as_deref().unwrap_or("—"),
                        ));
                    });
                }
            }
        }
    }

    /// Per-run leaderboard: status, progress, throughput, reward extremes.
    /// Newest run first; only runs ticked in the side panel are listed.
    fn summary_table(&mut self, ui: &mut egui::Ui) {
        let rows: Vec<(String, &RunModel)> = self
            .runs
            .iter()
            .rev() // newest (lexicographically last) first
            .filter(|(id, run)| self.run_is_selected(id.as_str(), run))
            .map(|(id, run)| (id.clone(), run))
            .collect();
        if rows.is_empty() {
            return;
        }

        egui::CollapsingHeader::new("Run summary")
            .default_open(true)
            .show(ui, |ui| {
                egui::Grid::new("run_summary")
                    .striped(true)
                    .spacing([14.0, 4.0])
                    .show(ui, |ui| {
                        for h in [
                            "run", "status", "iters", "wall", "env steps", "avg SPS", "final",
                            "best", "crash", "backend",
                        ] {
                            ui.strong(h);
                        }
                        ui.end_row();

                        for (id, run) in rows {
                            let status = run.status();
                            let live = run.is_live();
                            let status_color = if live {
                                egui::Color32::from_rgb(115, 217, 134)
                            } else if status == "completed" {
                                egui::Color32::from_rgb(150, 170, 200)
                            } else {
                                egui::Color32::from_rgb(242, 154, 51)
                            };

                            ui.monospace(id);
                            ui.colored_label(status_color, if live { "● live" } else { status });
                            ui.label(run.records.len().to_string());
                            let wall = run.last_elapsed();
                            ui.label(fmt_duration(wall));
                            ui.label(
                                run.total_env_steps()
                                    .map(|v| format!("{v:.0}"))
                                    .unwrap_or_else(|| "—".into()),
                            );
                            ui.label(
                                run.mean_sps()
                                    .map(|v| format!("{v:.0}"))
                                    .unwrap_or_else(|| "—".into()),
                            );
                            let series = run.series_for("mean_reward_per_step", XMode::Iteration);
                            let final_v = series.last().map(|p| p[1]);
                            let best_v = series.iter().map(|p| p[1]).fold(f64::NAN, f64::max);
                            let v = if best_v.is_nan() { None } else { Some(best_v) };
                            ui.label(final_v.map(fmt).unwrap_or_else(|| "—".into()));
                            ui.label(v.map(fmt).unwrap_or_else(|| "—".into()));
                            ui.label(
                                run.records
                                    .iter()
                                    .rev()
                                    .find_map(|r| r.crash_rate)
                                    .map(fmt)
                                    .unwrap_or_else(|| "—".into()),
                            );
                            ui.label(
                                run.manifest
                                    .as_ref()
                                    .map(|m| short_backend(m.backend.as_deref()).to_string())
                                    .unwrap_or_else(|| "?".into()),
                            );
                            ui.end_row();
                        }
                    });
            });
    }

    /// Iteration × run heatmap for the selected metric. Every row (run)
    /// normalizes over its own value range so per-run trends stay readable;
    /// each column buckets a span of iterations into the mean value.
    fn heatmap_panel(&mut self, ui: &mut egui::Ui) {
        ui.horizontal(|ui| {
            ui.heading(format!("Heatmap — {}", self.metric_label()));
            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                ui.label(egui::RichText::new("Rows normalized • Buckets by iteration").small().weak());
            });
        });
        ui.separator();
        let label = self.metric_label();
        let rows: Vec<(String, Vec<[f64; 2]>)> = self
            .runs
            .iter()
            .rev() // newest run on top
            .filter(|(id, run)| {
                self.run_is_selected(id.as_str(), run)
                    && !run.series_for(self.metric_key(), XMode::Iteration).is_empty()
            })
            .map(|(id, run)| (id.clone(), run.series_for(self.metric_key(), XMode::Iteration)))
            .collect();
        if rows.is_empty() {
            ui.label(
                egui::RichText::new("no visible data — tick a run on the left (or train!)").weak(),
            );
            return;
        }

        let max_iter = rows.iter().map(|(_, s)| s.len()).max().unwrap_or(1);
        const MAX_COLS: usize = 96;
        let cols = max_iter.min(MAX_COLS);
        let bucket = (max_iter as f64 / cols as f64).ceil().max(1.0) as usize;

        // Bucket each row into per-column means.
        let mut matrix: Vec<Vec<Option<f64>>> = Vec::with_capacity(rows.len());
        let mut norms: Vec<Option<(f64, f64)>> = Vec::with_capacity(rows.len());
        for (_id, series) in &rows {
            let mut counts = vec![0usize; cols];
            let mut sums = vec![0.0f64; cols];
            for &[x, v] in series {
                let col = ((x as usize) / bucket).min(cols - 1);
                counts[col] += 1;
                sums[col] += v;
            }
            let row: Vec<Option<f64>> = counts
                .iter()
                .zip(&sums)
                .map(|(&n, &s)| if n == 0 { None } else { Some(s / n as f64) })
                .collect();
            let vals: Vec<f64> = row.iter().filter_map(|c| *c).collect();
            norms.push(if vals.is_empty() {
                None
            } else {
                let (mut mn, mut mx) = (f64::INFINITY, f64::NEG_INFINITY);
                for v in &vals {
                    mn = mn.min(*v);
                    mx = mx.max(*v);
                }
                Some((mn, mx))
            });
            matrix.push(row);
        }

        // Layout: run-id labels on the left, colored cells to the right.
        let label_w = 280.0f32;
        let gap = 1.0f32;
        let cell_h = 18.0f32;
        let row_h = cell_h + gap;
        let cell_w = ((ui.available_width() - label_w) / cols as f32).clamp(2.0, 14.0);
        let matrix_w = cell_w * cols as f32 + (cols as f32 - 1.0) * gap;
        let legend_h = 46.0f32;
        let height = rows.len() as f32 * row_h + legend_h;

        egui::ScrollArea::both().show(ui, |ui| {
            let (response, painter) = ui.allocate_painter(
                egui::Vec2::new(label_w + matrix_w, height),
                egui::Sense::hover(),
            );
            let origin = response.rect.min;
            let text = egui::FontId::monospace(11.0);
            let weak = egui::Color32::from_gray(170);

            for (r, (((id, _), row), norm)) in rows.iter().zip(&matrix).zip(&norms).enumerate() {
                let y0 = origin.y + r as f32 * row_h;
                painter.text(
                    egui::pos2(origin.x + label_w - 8.0, y0 + cell_h / 2.0),
                    egui::Align2::RIGHT_CENTER,
                    id,
                    text.clone(),
                    weak,
                );
                for (c, cell) in row.iter().enumerate() {
                    let rect = egui::Rect::from_min_size(
                        egui::pos2(
                            origin.x + label_w + c as f32 * (cell_w + gap),
                            y0,
                        ),
                        egui::vec2(cell_w, cell_h),
                    );
                    let fill = match cell {
                        None => egui::Color32::from_rgb(22, 26, 34), // missing data
                        Some(v) => {
                            let t = match norm {
                                Some((mn, mx)) => (*v - *mn) / (*mx - *mn),
                                None => 0.5,
                            };
                            value_color(t)
                        }
                    };
                    painter.rect_filled(rect, 0.0, fill);
                }
            }

            // Legend: value scale + "per-row normalized" note.
            let leg_y = origin.y + rows.len() as f32 * row_h + 8.0;
            let leg_w = 200.0f32;
            let leg = egui::Rect::from_min_size(
                egui::pos2(origin.x + label_w, leg_y),
                egui::vec2(leg_w, 10.0),
            );
            for i in 0..leg_w as usize {
                let t = i as f64 / leg_w as f64;
                painter.rect_filled(
                    egui::Rect::from_min_size(
                        egui::pos2(leg.left() + i as f32, leg.top()),
                        egui::vec2(1.0, leg.height()),
                    ),
                    0.0,
                    value_color(t),
                );
            }
            painter.text(
                egui::pos2(leg.left(), leg.bottom() + 4.0),
                egui::Align2::LEFT_TOP,
                "Low → High (per-run normalized)  •  darker = missing",
                egui::FontId::proportional(11.0),
                weak,
            );

            // Hover tooltip: run, iteration span, and bucketed value.
            if let Some(hp) = response.hover_pos() {
                let dx = hp.x - (origin.x + label_w);
                let dy = hp.y - origin.y;
                if dx >= 0.0 && dy >= 0.0 {
                    let c = (dx / (cell_w + gap)) as usize;
                    let r = (dy / row_h) as usize;
                    if c < cols && r < rows.len() {
                        let (id, _) = &rows[r];
                        let i0 = c * bucket;
                        let i1 = (i0 + bucket).saturating_sub(1);
                        let txt = matrix[r][c]
                            .map(fmt)
                            .unwrap_or_else(|| "no data".to_string());
                        response.clone().on_hover_ui(|ui| {
                            ui.monospace(format!("{id}\niter {i0}–{i1} · {label}: {txt}"));
                        });
                    }
                }
            }
        });
    }

    fn kpi_strip(&mut self, ui: &mut egui::Ui) {
        // Find primary run (newest selected run inside the active tab)
        let mut primary_id: Option<String> = None;
        for (id, run) in self.runs.iter().rev() {
            if self.run_is_selected(id, run) {
                primary_id = Some(id.clone());
                break;
            }
        }
        let Some(pid) = primary_id else {
            ui.label(egui::RichText::new("No visible runs for KPIs").weak());
            return;
        };
        let Some(run) = self.runs.get(&pid) else { return; };
        let stale_secs = self.last_update.get(&pid).map(|t| t.elapsed().as_secs()).unwrap_or(u64::MAX);
        let is_stale = run.is_live() && stale_secs > 30;

        ui.horizontal_wrapped(|ui| {
            let kpis = vec![
                ("Reward/step", "mean_reward_per_step", ""),
                ("Best Reward", "mean_reward_per_step", "best"),
                ("Iterations", "", "iters"),
                ("Env Steps", "", "steps"),
                ("Avg SPS", "", "sps"),
                ("Crash Rate", "crash_rate", ""),
                ("Eval Reward", "eval_reward", ""),
                ("Entropy", "entropy", ""),
                ("Wall Time", "", "wall"),
                ("Policy Loss", "policy_loss", ""),
                ("Env Alt", "env_alt_mean", ""),
                ("Env Airspeed", "env_airspeed_mean", ""),
                ("Env Throttle", "env_throttle_mean", ""),
            ];
            for (label, key, mode) in kpis {
                let (val_str, color) = match mode {
                    "iters" => (run.records.len().to_string(), egui::Color32::from_gray(200)),
                    "steps" => (run.total_env_steps().map(|v| format!("{:.1}M", v/1e6)).or_else(|| run.total_env_steps().map(|v| format!("{:.0}k", v/1e3))).unwrap_or_else(|| "—".into()), egui::Color32::from_gray(200)),
                    "sps" => (run.mean_sps().map(fmt).unwrap_or_else(|| "—".into()), egui::Color32::from_gray(200)),
                    "wall" => (fmt_duration(run.last_elapsed()), egui::Color32::from_gray(200)),
                    "best" => {
                        if let Some(v) = run.best(key) {
                            (fmt(v), if v >= 0.0 { egui::Color32::from_rgb(115, 217, 134) } else { egui::Color32::from_rgb(242, 154, 51) })
                        } else { ("—".into(), egui::Color32::GRAY) }
                    }
                    _ => {
                        if let Some(v) = run.latest(key) {
                            let c = if key == "crash_rate" {
                                if v > 0.1 { egui::Color32::from_rgb(242, 115, 102) } else if v > 0.05 { egui::Color32::from_rgb(242, 204, 77) } else { egui::Color32::from_rgb(115, 217, 134) }
                            } else if key == "mean_reward_per_step" || key == "eval_reward" {
                                if v >= 0.0 { egui::Color32::from_rgb(115, 217, 134) } else { egui::Color32::from_rgb(242, 154, 51) }
                            } else {
                                egui::Color32::from_gray(220)
                            };
                            if key == "crash_rate" { (format!("{:.1}%", v*100.0), c) } else { (fmt(v), c) }
                        } else { ("—".into(), egui::Color32::GRAY) }
                    }
                };
                let frame = egui::Frame::group(ui.style()).inner_margin(6.0);
                frame.show(ui, |ui| {
                    ui.set_min_width(110.0);
                    ui.vertical(|ui| {
                        ui.horizontal(|ui| {
                            ui.label(egui::RichText::new(label).small().weak());
                            if is_stale && label.contains("Reward") {
                                ui.label(egui::RichText::new("STALE").small().color(egui::Color32::YELLOW));
                            }
                        });
                        ui.label(egui::RichText::new(&val_str).strong().monospace().color(color));
                        // mini sparkline
                        if !key.is_empty() && mode != "best" {
                            let sp = run.sparkline(key, 20);
                            if !sp.is_empty() {
                                let minv = sp.iter().copied().fold(f64::INFINITY, f64::min);
                                let maxv = sp.iter().copied().fold(f64::NEG_INFINITY, f64::max);
                                let span = (maxv - minv).max(1e-9);
                                let w = 90.0f32; let h = 16.0f32;
                                let (rect, _) = ui.allocate_exact_size(egui::vec2(w, h), egui::Sense::hover());
                                let painter = ui.painter();
                                for i in 0..sp.len().saturating_sub(1) {
                                    let x1 = rect.left() + (i as f32 / (sp.len()-1).max(1) as f32) * rect.width();
                                    let x2 = rect.left() + ((i+1) as f32 / (sp.len()-1).max(1) as f32) * rect.width();
                                    let y1 = rect.bottom() - ((sp[i]-minv)/span) as f32 * rect.height();
                                    let y2 = rect.bottom() - ((sp[i+1]-minv)/span) as f32 * rect.height();
                                    painter.line_segment([egui::pos2(x1,y1), egui::pos2(x2,y2)], egui::Stroke::new(1.0, color));
                                }
                            }
                        }
                    });
                });
                ui.add_space(2.0);
            }
            ui.label(egui::RichText::new(format!("· {}", pid)).small().weak());
        });
        if let Some(exp) = &self.kpi_expanded.clone() {
            ui.separator();
            ui.collapsing(format!("KPI Detail: {}", exp), |ui| {
                if let Some(r) = self.runs.get(&pid) {
                    let series = r.series_for(exp, XMode::Iteration);
                    if !series.is_empty() {
                        ui.label(format!("Points: {} | Min: {} | Max: {} | Last: {}",
                            series.len(), fmt(series.iter().map(|p|p[1]).fold(f64::INFINITY,f64::min)),
                            fmt(series.iter().map(|p|p[1]).fold(f64::NEG_INFINITY,f64::max)),
                            fmt(series.last().unwrap()[1])));
                    } else {
                        ui.label("No data");
                    }
                }
            });
        }
    }

    fn table_panel(&mut self, ui: &mut egui::Ui) {
        let key = self.metric_key();
        let label = self.metric_label();
        ui.heading(format!("Metrics table — {}", label));
        ui.separator();
        egui::ScrollArea::both().show(ui, |ui| {
            egui::Grid::new("metrics_table")
                .striped(true)
                .spacing([16.0, 4.0])
                .show(ui, |ui| {
                    for h in ["run", "status", "iters", "latest", "best", "final", "wall", "env steps", "avg SPS"] {
                        ui.strong(h);
                    }
                    ui.end_row();
                    let rows: Vec<String> = self.runs.keys().cloned().collect();
                    for id in rows {
                        let Some(run) = self.runs.get(&id) else {
                            continue;
                        };
                        if !self.run_is_selected(&id, run) {
                            continue;
                        }
                        let series = run.series_for(key, XMode::Iteration);
                        let latest = series.last().map(|p| p[1]);
                        let best = series.iter().map(|p| p[1]).fold(f64::NAN, f64::max);
                        let best_v = if best.is_nan() { None } else { Some(best) };
                        ui.monospace(&id);
                        ui.label(run.status());
                        ui.label(run.records.len().to_string());
                        ui.label(latest.map(fmt).unwrap_or_else(|| "—".into()));
                        ui.label(best_v.map(fmt).unwrap_or_else(|| "—".into()));
                        ui.label(latest.map(fmt).unwrap_or_else(|| "—".into()));
                        ui.label(fmt_duration(run.last_elapsed()));
                        ui.label(run.total_env_steps().map(|v| format!("{v:.0}")).unwrap_or_else(|| "—".into()));
                        ui.label(run.mean_sps().map(|v| format!("{v:.0}")).unwrap_or_else(|| "—".into()));
                        ui.end_row();
                    }
                });
        });
    }

    fn bar_panel(&mut self, ui: &mut egui::Ui) {
        let key = self.metric_key();
        let label = self.metric_label();
        ui.heading(format!("Bar chart — {}", label));
        ui.separator();
        let mut bars: Vec<(String, f64, egui::Color32)> = Vec::new();
        let mut i = 0;
        for (id, run) in self.runs.iter() {
            if !self.run_is_selected(id, run) {
                continue;
            }
            let series = run.series_for(key, XMode::Iteration);
            if let Some(last) = series.last() {
                bars.push((id.clone(), last[1], PALETTE[i % PALETTE.len()]));
                i += 1;
            }
        }
        if bars.is_empty() {
            ui.label(egui::RichText::new("No data").weak());
            return;
        }
        let max_v = bars.iter().map(|(_, v, _)| v.abs()).fold(0.0f64, f64::max).max(1e-9);
        egui::ScrollArea::vertical().show(ui, |ui| {
            for (name, val, color) in &bars {
                ui.horizontal(|ui| {
                    ui.label(egui::RichText::new(name).monospace().small());
                    let w = (ui.available_width() * 0.6) as f32;
                    let (rect, _) = ui.allocate_exact_size(egui::vec2(w, 18.0), egui::Sense::hover());
                    let painter = ui.painter();
                    let pct = ((val.abs() / max_v) as f32).clamp(0.0, 1.0);
                    painter.rect_filled(
                        egui::Rect::from_min_size(rect.min, egui::vec2(rect.width() * pct, rect.height())),
                        4.0,
                        *color,
                    );
                    painter.text(
                        rect.right_center() + egui::vec2(4.0, 0.0),
                        egui::Align2::LEFT_CENTER,
                        fmt(*val),
                        egui::FontId::proportional(11.0),
                        egui::Color32::from_gray(200),
                    );
                });
            }
        });
    }

    fn scatter_panel(&mut self, ui: &mut egui::Ui) {
        ui.horizontal(|ui| {
            ui.heading("Scatter");
            ui.label(egui::RichText::new("Latest Reward vs Env Steps").weak());
        });
        ui.separator();
        let mut pts: Vec<(f64, f64, String, egui::Color32)> = Vec::new();
        let mut i = 0;
        for (id, run) in self.runs.iter() {
            if !self.run_is_selected(id, run) {
                continue;
            }
            let steps = run.total_env_steps().unwrap_or(0.0);
            let reward = run.series_for("mean_reward_per_step", XMode::Iteration).last().map(|p| p[1]).unwrap_or(0.0);
            pts.push((steps, reward, id.clone(), PALETTE[i % PALETTE.len()]));
            i += 1;
        }
        if pts.is_empty() {
            ui.label(egui::RichText::new("No data").weak());
            return;
        }
        let mut hover: Option<egui_plot::PlotPoint> = None;
        Plot::new("scatter")
            .legend(Legend::default())
            .x_axis_label("Env steps")
            .y_axis_label("Reward / step")
            .show(ui, |plot_ui| {
                for (x, y, name, color) in &pts {
                    plot_ui.points(
                        Points::new(name.clone(), PlotPoints::new(vec![[*x, *y]]))
                            .color(*color)
                            .radius(7.0)
                            .filled(true),
                    );
                }
                hover = plot_ui.pointer_coordinate();
            });
        if let Some(p) = hover {
            ui.label(egui::RichText::new(format!("cursor → steps {} · reward {}", fmt(p.x), fmt(p.y))).monospace().weak());
        }
    }
}

// ── Entry point ──────────────────────────────────────────────────────

fn parse_args() -> (PathBuf, usize) {
    let mut root = PathBuf::from("runs");
    let mut metric = 0usize;
    let mut it = std::env::args().skip(1);
    while let Some(arg) = it.next() {
        match arg.as_str() {
            "--runs" => {
                if let Some(v) = it.next() {
                    root = PathBuf::from(v);
                }
            }
            "--metric" => {
                if let Some(v) = it.next() {
                    metric = METRICS.iter().position(|(k, _)| *k == v).unwrap_or(0);
                }
            }
            "--help" | "-h" => {
                println!(
                    "rl_dash — rl_agent training telemetry dashboard\n\n\
                     USAGE:\n  rl_dash [--runs DIR] [--metric KEY]\n\n\
                     METRICS:\n  {}",
                    METRICS
                        .iter()
                        .map(|(k, l)| format!("  {k:20} {l}"))
                        .collect::<Vec<_>>()
                        .join("\n")
                );
                std::process::exit(0);
            }
            other => {
                eprintln!("rl_dash: unknown argument `{other}` (see --help)");
                std::process::exit(2);
            }
        }
    }
    (root, metric)
}

fn main() -> eframe::Result<()> {
    let (runs_root, metric) = parse_args();
    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_inner_size([1280.0, 800.0])
            .with_title("rl_dash — training telemetry"),
        ..Default::default()
    };
    eframe::run_native(
        "rl_dash",
        options,
        Box::new(move |_cc| Ok(Box::new(DashApp::new(runs_root, metric)))),
    )
}

// ── Tests ────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    const SAMPLE: &str = r#"{"iteration":2,"elapsed_s":16.4,"steps_per_s":180.7,"policy_loss":-0.032,"value_loss":8.58,"entropy":3.67,"advantage_mean":-0.46,"advantage_std":2.88,"mean_reward_per_step":0.059,"mean_episode_steps":0.0,"crash_rate":0.0,"episodes_completed":0,"eval":{"alt_err":99.4,"spd_err":11.1,"total_reward":-4.26,"crashes":0,"episodes":1,"mean_steps":2000.0},"checkpoint":true,"env":{"samples":2048,"alt_mean":1200.5,"alt_min":1100.0,"alt_max":1300.0,"airspeed_mean":60.2,"climb_mean":0.4,"ground_speed_mean":61.0,"roll_abs_mean":0.1,"pitch_abs_mean":0.05,"aoa_mean":0.02,"wind_mean":4.5,"wind_max":9.0,"throttle_mean":0.55,"surface_effort_mean":2.5,"baro_err_abs_mean":0.8,"airspeed_err_abs_mean":0.3,"battery_pct_min":91.0,"weather":{"samples":2048,"wind_speed_mean":5.0,"gust_rms_mean":1.2,"gust_rms_max":3.4,"turbulence_mean":1.0,"severity_mean":1.0,"rain_mean":0.2,"visibility_min":5000.0,"updraft_abs_mean":0.4,"shear_abs_mean":0.01}}}"#;

    #[test]
    fn parses_full_record_and_flattens_metrics() {
        let rec: MetricsRecord = serde_json::from_str(SAMPLE).expect("sample must parse");
        assert_eq!(rec.iteration, 2);
        assert_eq!(rec.elapsed_s, Some(16.4));
        assert!((rec.metric("mean_reward_per_step").unwrap() - 0.059).abs() < 1e-9);
        assert!((rec.metric("alt_err").unwrap() - 99.4).abs() < 1e-9);
        assert!((rec.metric("eval_reward").unwrap() + 4.26).abs() < 1e-9);
        // Ground-truth env block flattens to `env_*` / `weather_*` keys.
        assert!((rec.metric("env_alt_mean").unwrap() - 1200.5).abs() < 1e-9);
        assert!((rec.metric("env_airspeed_mean").unwrap() - 60.2).abs() < 1e-9);
        assert!((rec.metric("env_throttle_mean").unwrap() - 0.55).abs() < 1e-9);
        assert!((rec.metric("env_battery_pct_min").unwrap() - 91.0).abs() < 1e-9);
        assert!((rec.metric("weather_gust_rms_max").unwrap() - 3.4).abs() < 1e-9);
        assert!((rec.metric("weather_visibility_min").unwrap() - 5000.0).abs() < 1e-9);
        // Non-eval iteration has no eval metric.
        let no_eval: MetricsRecord =
            serde_json::from_str(r#"{"iteration":3,"mean_reward_per_step":0.5}"#).unwrap();
        assert_eq!(no_eval.metric("alt_err"), None);
        assert_eq!(no_eval.metric("mean_reward_per_step"), Some(0.5));
        assert_eq!(no_eval.metric("entropy"), None);
        // Pre-env-telemetry runs parse fine and report no env data.
        assert_eq!(no_eval.metric("env_alt_mean"), None);
        assert_eq!(no_eval.metric("weather_wind_speed_mean"), None);
        // A run without a weather system still reports flight env metrics.
        let no_weather: MetricsRecord = serde_json::from_str(
            r#"{"iteration":4,"env":{"samples":10,"alt_mean":950.0,"throttle_mean":0.4}}"#,
        )
        .unwrap();
        assert!((no_weather.metric("env_alt_mean").unwrap() - 950.0).abs() < 1e-9);
        assert_eq!(no_weather.metric("weather_gust_rms_max"), None);
    }

    #[test]
    fn mission_counters_parse_and_appear_in_the_metric_list() {
        let rec: MetricsRecord = serde_json::from_str(
            r#"{"iteration":0,"mission_captures":2,"mission_returns":1,"mission_successes":1}"#,
        )
        .unwrap();
        assert_eq!(rec.metric("mission_captures"), Some(2.0));
        assert_eq!(rec.metric("mission_returns"), Some(1.0));
        assert_eq!(rec.metric("mission_successes"), Some(1.0));
        for key in ["mission_captures", "mission_returns", "mission_successes"] {
            assert!(
                METRICS.iter().any(|(k, _)| *k == key),
                "rl_dash must offer the `{key}` series"
            );
        }

        // Avionics records have no mission keys and must stay `None`.
        let avionics: MetricsRecord = serde_json::from_str(r#"{"iteration":0}"#).unwrap();
        assert_eq!(avionics.metric("mission_captures"), None);
        assert_eq!(avionics.metric("mission_returns"), None);
        assert_eq!(avionics.metric("mission_successes"), None);
    }

    #[test]
    fn incremental_poll_reads_appended_lines_once() {
        let dir = std::env::temp_dir().join(format!("rl_dash_poll_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("metrics.jsonl");
        std::fs::write(&path, SAMPLE.lines().next().unwrap().to_string() + "\n").unwrap();

        let mut run = RunModel::new(dir.clone());
        run.poll();
        assert_eq!(run.records.len(), 1);
        run.poll(); // nothing new → no duplicate rows
        assert_eq!(run.records.len(), 1);

        std::fs::write(
            &path,
            format!(
                "{}{}\n{}\n",
                SAMPLE.lines().next().unwrap(),
                SAMPLE.lines().next().unwrap().trim_end(),
                SAMPLE.lines().next().unwrap()
            ),
        )
        .unwrap();
        run.poll();
        assert_eq!(
            run.records.len(),
            2,
            "append must be picked up exactly once"
        );

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn replaced_metrics_file_reloads_from_zero() {
        let dir = std::env::temp_dir().join(format!("rl_dash_replace_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("metrics.jsonl");

        // Same byte length, different value: only identity can distinguish
        // the replacement from an unchanged file.
        let old = r#"{"iteration":0,"mean_reward_per_step":1.0}"#;
        let new = r#"{"iteration":0,"mean_reward_per_step":2.0}"#;
        assert_eq!(old.len(), new.len());
        std::fs::write(&path, format!("{old}\n")).unwrap();

        let mut run = RunModel::new(dir.clone());
        run.poll();
        assert_eq!(run.records.len(), 1);
        assert_eq!(run.records[0].metric("mean_reward_per_step"), Some(1.0));

        // Simulate a fresh writer replacing the file by rename: same path,
        // same length, different inode.
        let staged = dir.join("metrics.jsonl.new");
        std::fs::write(&staged, format!("{new}\n")).unwrap();
        std::fs::rename(&staged, &path).unwrap();

        run.poll();
        assert_eq!(run.records.len(), 1, "replaced file must not append");
        assert_eq!(
            run.records[0].metric("mean_reward_per_step"),
            Some(2.0),
            "the dashboard must show the replacement, not the stale row"
        );

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn same_length_in_place_rewrite_reloads_by_mtime() {
        let dir = std::env::temp_dir().join(format!("rl_dash_rewrite_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("metrics.jsonl");

        let old = r#"{"iteration":0,"mean_reward_per_step":1.0}"#;
        let new = r#"{"iteration":0,"mean_reward_per_step":3.0}"#;
        assert_eq!(old.len(), new.len());
        std::fs::write(&path, format!("{old}\n")).unwrap();

        let mut run = RunModel::new(dir.clone());
        run.poll();
        assert_eq!(run.records[0].metric("mean_reward_per_step"), Some(1.0));

        // Same inode and same length; only the mtime marks the rewrite. Set it
        // explicitly instead of sleeping, so a coarse timestamp granularity
        // cannot make this test flaky.
        std::fs::write(&path, format!("{new}\n")).unwrap();
        std::fs::OpenOptions::new()
            .write(true)
            .open(&path)
            .unwrap()
            .set_modified(SystemTime::now() + Duration::from_secs(1))
            .unwrap();

        run.poll();
        assert_eq!(run.records.len(), 1);
        assert_eq!(run.records[0].metric("mean_reward_per_step"), Some(3.0));

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn rescan_polls_a_newly_discovered_run_on_the_same_tick() {
        let root = std::env::temp_dir().join(format!("rl_dash_rescan_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&root);
        std::fs::create_dir_all(&root).unwrap();

        let mut app = DashApp::new(root.clone(), 0);
        assert!(app.runs.is_empty(), "empty root has no runs");

        let run_dir = root.join("20260101_000000_000_ppo_seed42");
        std::fs::create_dir_all(&run_dir).unwrap();
        std::fs::write(
            run_dir.join("metrics.jsonl"),
            format!("{}\n", SAMPLE.lines().next().unwrap()),
        )
        .unwrap();

        app.rescan();
        let run = app.runs.values().next().expect("run discovered");
        assert_eq!(
            run.records.len(),
            1,
            "rescan must poll the new run before returning"
        );

        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn discover_runs_sorts_lexicographically_newest_last() {
        let root = std::env::temp_dir().join(format!("rl_dash_disc_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&root);
        std::fs::create_dir_all(root.join("20260101_000000_000_ppo_seed42")).unwrap();
        std::fs::create_dir_all(root.join("20260102_000000_000_ppo_seed42")).unwrap();
        std::fs::create_dir_all(root.join("20260103_000000_000_ppo_seed42")).unwrap();
        std::fs::write(
            root.join("20260101_000000_000_ppo_seed42/metrics.jsonl"),
            "",
        )
        .unwrap();
        std::fs::write(
            root.join("20260102_000000_000_ppo_seed42/metrics.jsonl"),
            "",
        )
        .unwrap();
        // A run that only has run.json so far (metrics.jsonl not written yet)
        // is still a run.
        std::fs::write(
            root.join("20260103_000000_000_ppo_seed42/run.json"),
            "{\"status\":\"running\"}",
        )
        .unwrap();
        // A dir with neither file is not a run.
        std::fs::create_dir_all(root.join("not_a_run")).unwrap();

        let found = discover_runs(&root);
        assert_eq!(found.len(), 3);
        let last = found.last().unwrap().file_name().unwrap().to_string_lossy();
        assert_eq!(
            last, "20260103_000000_000_ppo_seed42",
            "newest run sorts last"
        );
        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn poll_tolerates_a_run_whose_metrics_file_has_not_appeared_yet() {
        let dir = std::env::temp_dir().join(format!("rl_dash_fresh_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        // run.json exists (freshly-spawned trainer), metrics.jsonl not yet.
        std::fs::write(dir.join("run.json"), "{\"status\":\"running\"}").unwrap();

        let mut run = RunModel::new(dir.clone());
        run.poll();
        assert!(run.records.is_empty());
        assert!(
            run.error.is_none(),
            "missing file on a fresh run is not an error"
        );

        // Once the file lands the same RunModel picks it up.
        std::fs::write(
            dir.join("metrics.jsonl"),
            "{\"iteration\":1,\"elapsed_s\":1.0,\"mean_reward_per_step\":0.5}\n",
        )
        .unwrap();
        run.poll();
        assert_eq!(run.records.len(), 1);
        assert!(run.error.is_none());
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn smoothing_preserves_length_and_keeps_first_value() {
        let pts = [[0.0, 1.0], [1.0, 3.0], [2.0, 2.0]];
        let s = smooth(&pts, 4);
        assert_eq!(s.len(), pts.len());
        assert_eq!(s[0][1], 1.0, "first EMA seed equals the first value");
        assert!(s[1][1] > 1.0 && s[1][1] < 3.0, "EMA moves toward the point");
        assert_eq!(smooth(&pts, 0), pts.to_vec(), "window 0 = off");
    }

    #[test]
    fn auto_rescan_picks_up_a_run_spawned_after_launch() {
        // The RL tab spawns the trainer, which creates its run dir *after* the
        // dashboard's first scan — that run must appear without a click.
        let root = std::env::temp_dir().join(format!("rl_dash_auto_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&root);
        std::fs::create_dir_all(root.join("20260920_100000_000_ppo_seed42")).unwrap();
        std::fs::write(
            root.join("20260920_100000_000_ppo_seed42/metrics.jsonl"),
            "{\"iteration\":1,\"elapsed_s\":1.0,\"mean_reward_per_step\":0.1}\n",
        )
        .unwrap();

        let mut app = DashApp::new(root.clone(), 0);
        assert_eq!(
            app.runs.len(),
            1,
            "dashboard opens with only the completed run"
        );

        // Trainer spawns its live run after launch.
        std::fs::create_dir_all(root.join("20260923_120000_000_ppo_seed42")).unwrap();
        std::fs::write(
            root.join("20260923_120000_000_ppo_seed42/metrics.jsonl"),
            "{\"iteration\":1,\"elapsed_s\":2.0,\"mean_reward_per_step\":0.7}\n",
        )
        .unwrap();

        for _ in 0..4 {
            app.poll_all();
        }
        assert_eq!(
            app.runs.len(),
            1,
            "rescan only triggers every 5 polls (~1 s)"
        );
        app.poll_all();
        assert_eq!(
            app.runs.len(),
            2,
            "post-launch run is discovered without a manual rescan"
        );
        let newest = app.runs.keys().next_back().unwrap().clone();
        assert_eq!(newest, "20260923_120000_000_ppo_seed42");
        assert!(app.visible.get(&newest).copied().unwrap_or(false));
        app.poll_all();
        assert_eq!(
            app.runs[&newest].records.len(),
            1,
            "its telemetry is tailed on the following poll"
        );
        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn run_manifest_parses_trainer_schema() {
        let json = r#"{"schema":"rl_agent.telemetry.v1","run_id":"20260923_104623_870_ppo_seed42_smoke","started_utc":"2026-09-23T10:46:23Z","ended_utc":"2026-09-23T10:46:56Z","status":"completed","algo":"ppo","backend":"burn_ndarray::backend::NdArrayDevice","git_commit":"d48a301"}"#;
        let m: RunManifest = serde_json::from_str(json).expect("manifest must parse");
        assert_eq!(m.status.as_deref(), Some("completed"));
        assert_eq!(m.algo.as_deref(), Some("ppo"));
        assert_eq!(m.git_commit.as_deref(), Some("d48a301"));
    }

    #[test]
    fn compare_view_renders_every_metric_with_real_data() {
        // End-to-end guard against schema drift between rl_agent's telemetry
        // writer and this reader: copy the repo's newest run into a temp root,
        // build the app, and render the Compare panel headlessly for every
        // selectable metric. Any metric that has data must produce at least
        // one drawn line path inside a sizable plot rect. Skipped when the
        // repo has no runs.
        let repo_runs = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .expect("rl_dash sits at repo root")
            .join("runs");
        let Some(newest) = discover_runs(&repo_runs).into_iter().next_back() else {
            return; // no runs on this machine — nothing to verify against
        };
        let id = newest
            .file_name()
            .map(|s| s.to_string_lossy().into_owned())
            .unwrap();
        let root = std::env::temp_dir().join(format!("rl_dash_cmp_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&root);
        let dst = root.join(&id);
        std::fs::create_dir_all(&dst).unwrap();
        if newest.join("metrics.jsonl").exists() {
            std::fs::copy(newest.join("metrics.jsonl"), dst.join("metrics.jsonl")).unwrap();
        }
        if newest.join("run.json").exists() {
            std::fs::copy(newest.join("run.json"), dst.join("run.json")).unwrap();
        }

        for (_i, (key, _)) in METRICS.iter().enumerate() {
            // The panel would show "no visible data" for metrics this run never
            // wrote; only assert against metrics that actually have points.
            let mut app = DashApp::new(root.clone(), _i);
            for _ in 0..3 {
                app.poll_all();
            }
            let has_data = app
                .runs
                .values()
                .any(|r| !r.series_for(key, XMode::Iteration).is_empty());
            if !has_data {
                continue;
            }

            let ctx = egui::Context::default();
            let out = ctx.run_ui(Default::default(), |ctx| {
                // Emulate the eframe root layout (side runs panel + central view).
                egui::CentralPanel::default().show(ctx, |ui| {
                    egui::Panel::left("runs_test")
                        .resizable(true)
                        .default_size(320.0)
                        .show(ui, |ui| {
                            app.side_panel(ui);
                        });
                    egui::CentralPanel::default().show(ui, |ui| {
                        app.plot_panel(ui);
                    });
                });
            });
            let paths = out
                .shapes
                .iter()
                .filter(|c| {
                    matches!(c.shape, egui::Shape::Path(_))
                        || matches!(c.shape, egui::Shape::LineSegment { .. })
                })
                .count();
            let max_clip_w = out
                .shapes
                .iter()
                .map(|c| c.clip_rect.width())
                .fold(0.0f32, f32::max);
            let max_clip_h = out
                .shapes
                .iter()
                .map(|c| c.clip_rect.height())
                .fold(0.0f32, f32::max);
            assert!(
                paths >= 1,
                "compare view must draw a line path for `{key}` on {id} (drew {paths})"
            );
            assert!(
                max_clip_w > 100.0 && max_clip_h > 80.0,
                "compare plot rect must be sizable for `{key}` (clip {max_clip_w:.0}x{max_clip_h:.0})"
            );
        }

        // Overlay constellation (log y + best-so-far + evals + checkpoints +
        // seed bands on the steps axis) and the heatmap must render without
        // panicking on the same real data.
        {
            let mut app = DashApp::new(root.clone(), 0);
            for _ in 0..3 {
                app.poll_all();
            }
            if app
                .runs
                .values()
                .any(|r| !r.series_for("mean_reward_per_step", XMode::Iteration).is_empty())
            {
                app.log_y = true;
                app.show_best = true;
                app.show_evals = true;
                app.show_checkpoints = true;
                app.seed_bands = true;
                app.x_mode = XMode::Steps;
                app.view = ViewMode::Detail;
                app.detail_tab = DetailTab::Chart;
                let _ = egui::Context::default().run_ui(Default::default(), |ctx| {
                    egui::CentralPanel::default().show(ctx, |ui| {
                        app.plot_panel(ui);
                    });
                });
                app.detail_tab = DetailTab::Heatmap;
                let _ = egui::Context::default().run_ui(Default::default(), |ctx| {
                    egui::CentralPanel::default().show(ctx, |ui| {
                        app.heatmap_panel(ui);
                    });
                });
            }
        }
        let _ = std::fs::remove_dir_all(&root);
    }

    fn make_run(records: &[&str], manifest: Option<&str>) -> RunModel {
        let dir = std::env::temp_dir().join(format!(
            "rl_dash_feat_{}_{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::create_dir_all(&dir).unwrap();
        if let Some(m) = manifest {
            std::fs::write(dir.join("run.json"), m).unwrap();
        }
        std::fs::write(
            dir.join("metrics.jsonl"),
            records.join("\n") + "\n",
        )
        .unwrap();
        let mut run = RunModel::new(dir.clone());
        run.poll();
        let _ = std::fs::remove_dir_all(&dir);
        run
    }

    #[test]
    fn x_modes_and_cumulative_episodes_series() {
        // 0 s → 10 s at 100 sps → 30 s at 150 sps.
        let run = make_run(
            &[
                r#"{"iteration":0,"elapsed_s":0.0,"steps_per_s":100.0,"episodes_completed":2}"#,
                r#"{"iteration":1,"elapsed_s":10.0,"steps_per_s":100.0,"episodes_completed":3}"#,
                r#"{"iteration":2,"elapsed_s":30.0,"steps_per_s":150.0,"episodes_completed":4}"#,
            ],
            None,
        );
        assert_eq!(
            run.x_values(XMode::Iteration),
            vec![0.0, 1.0, 2.0],
            "iteration axis is the record index"
        );
        assert_eq!(run.x_values(XMode::Elapsed), vec![0.0, 10.0, 30.0]);
        assert_eq!(
            run.x_values(XMode::Steps),
            vec![0.0, 1000.0, 4000.0],
            "steps = Σ throughput × wall-clock delta (10×100 + 20×150)"
        );
        let eps = run.series_for("episodes_completed", XMode::Iteration);
        assert_eq!(
            eps,
            vec![[0.0, 2.0], [1.0, 5.0], [2.0, 9.0]],
            "episodes_completed accumulates across the run"
        );
    }

    #[test]
    fn best_so_far_respects_metric_direction() {
        let pts = [[0.0, 1.0], [1.0, 3.0], [2.0, 2.0], [3.0, 4.0]];
        assert_eq!(
            best_so_far(&pts, false),
            vec![[0.0, 1.0], [1.0, 3.0], [2.0, 3.0], [3.0, 4.0]],
            "running max keeps the envelope"
        );
        assert_eq!(
            best_so_far(&pts, true),
            vec![[0.0, 1.0], [1.0, 1.0], [2.0, 1.0], [3.0, 1.0]],
            "running min for loss metrics"
        );
    }

    #[test]
    fn log10_pts_drops_nonpositive_and_transforms() {
        let pts = [[0.0, 100.0], [1.0, -5.0], [2.0, 0.0], [3.0, 1.0]];
        let out = log10_pts(&pts);
        assert_eq!(out.len(), 2, "non-positive values are dropped");
        assert!((out[0][0] - 0.0).abs() < 1e-9 && (out[0][1] - 2.0).abs() < 1e-9);
        assert!((out[1][0] - 3.0).abs() < 1e-9 && out[1][1].abs() < 1e-9);
    }

    #[test]
    fn group_key_ignores_seed_but_keeps_config() {
        let base = r#"{
            "status":"completed","algo":"ppo","backend":"burn_ndarray::backend::NdArrayDevice",
            "trainer":{"iterations":100,"rollout_steps":2048,"seed":42},
            "network":{"obs_dim":9,"hidden":[32,32],"action_dim":2},
            "algorithm":{"name":"ppo","clip":0.2}
        }"#;
        let a = make_run(&[r#"{"iteration":0}"#], Some(base));
        let b = make_run(
            &[r#"{"iteration":0}"#],
            Some(&base.replace("\"seed\":42", "\"seed\":7")),
        );
        assert_eq!(
            a.group_key(),
            b.group_key(),
            "different seeds but same config → one replication group"
        );
        let c = make_run(
            &[r#"{"iteration":0}"#],
            Some(&base.replace("\"clip\":0.2", "\"clip\":0.15")),
        );
        assert_ne!(
            a.group_key(),
            c.group_key(),
            "a real config change splits the group"
        );
        let nodata = make_run(&[r#"{"iteration":0}"#], None);
        assert_eq!(nodata.group_key(), None, "no manifest config → no group");
    }

    #[test]
    fn summary_adjectives_compute_from_records() {
        let run = make_run(
            &[
                r#"{"iteration":0,"elapsed_s":2.0,"steps_per_s":100.0,"mean_reward_per_step":-0.5}"#,
                r#"{"iteration":1,"elapsed_s":7.0,"steps_per_s":120.0,"mean_reward_per_step":0.3}"#,
            ],
            None,
        );
        assert_eq!(run.last_elapsed(), Some(7.0));
        assert_eq!(run.mean_sps(), Some(110.0));
        assert_eq!(run.total_env_steps(), Some(800.0));
        assert_eq!(fmt_duration(Some(125.0)), "2m 05s");
        assert_eq!(fmt_duration(None), "—");
        assert_eq!(short_backend(Some("burn_ndarray::backend::NdArrayDevice")), "NdArrayDevice");
    }

    // ---- Tab grouping -------------------------------------------------

    fn temp_root(tag: &str) -> PathBuf {
        let root = std::env::temp_dir().join(format!(
            "rl_dash_tabs_{tag}_{}_{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        let _ = std::fs::remove_dir_all(&root);
        std::fs::create_dir_all(&root).unwrap();
        root
    }

    /// Create `root/<id>` with the given manifest and one metric record.
    fn make_run_at(root: &Path, id: &str, manifest: Option<&str>) -> RunModel {
        let dir = root.join(id);
        std::fs::create_dir_all(&dir).unwrap();
        if let Some(m) = manifest {
            std::fs::write(dir.join("run.json"), m).unwrap();
        }
        std::fs::write(
            dir.join("metrics.jsonl"),
            "{\"iteration\":0,\"mean_reward_per_step\":0.1}\n",
        )
        .unwrap();
        let mut run = RunModel::new(dir);
        run.poll();
        run
    }

    fn manifest(name: Option<&str>, hidden: u32, backend: &str) -> String {
        let args = match name {
            Some(n) => format!(r#"["rl_agent","--name","{n}"]"#),
            None => r#"["rl_agent"]"#.to_string(),
        };
        format!(
            r#"{{"status":"completed","algo":"ppo","backend":"{backend}",
                "network":{{"obs_dim":19,"hidden":{hidden},"action_dim":4}},
                "command_line":{args}}}"#
        )
    }

    #[test]
    fn model_name_reads_both_flag_forms() {
        let root = temp_root("flagforms");
        let spaced = make_run_at(
            &root,
            "20260101_000000_000_ppo_seed42",
            Some(&manifest(Some("alpha"), 128, "burn_ndarray::backend::NdArrayDevice")),
        );
        assert_eq!(spaced.explicit_model_name().as_deref(), Some("alpha"));

        let eq = make_run_at(
            &root,
            "20260102_000000_000_ppo_seed42",
            Some(
                r#"{"command_line":["rl_agent","--name=beta"],"algo":"ppo",
                    "backend":"burn_ndarray::backend::NdArrayDevice"}"#,
            ),
        );
        assert_eq!(eq.explicit_model_name().as_deref(), Some("beta"));

        let unnamed = make_run_at(&root, "20260103_000000_000_ppo_seed42", None);
        assert_eq!(unnamed.explicit_model_name(), None);
        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn run_id_label_recovers_the_optional_name_suffix() {
        let root = temp_root("idlabel");
        let labelled = make_run_at(
            &root,
            "20260923_104623_870_ppo_seed42_smoke",
            Some(&manifest(None, 128, "burn_ndarray::backend::NdArrayDevice")),
        );
        assert_eq!(labelled.run_id_label().as_deref(), Some("smoke"));

        let plain = make_run_at(
            &root,
            "20260923_104623_870_ppo_seed42",
            Some(&manifest(None, 128, "burn_ndarray::backend::NdArrayDevice")),
        );
        assert_eq!(plain.run_id_label(), None, "no suffix → no label");
        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn model_tabs_prefer_explicit_names_then_fall_back() {
        let root = temp_root("modeltabs");
        make_run_at(
            &root,
            "20260101_000000_000_ppo_seed42_old",
            Some(&manifest(None, 128, "burn_ndarray::backend::NdArrayDevice")),
        );
        make_run_at(
            &root,
            "20260102_000000_000_ppo_seed42",
            Some(&manifest(Some("alpha"), 128, "burn_ndarray::backend::NdArrayDevice")),
        );
        make_run_at(
            &root,
            "20260103_000000_000_ppo_seed42",
            Some(&manifest(Some("alpha"), 128, "burn_ndarray::backend::NdArrayDevice")),
        );
        make_run_at(
            &root,
            "20260104_000000_000_ppo_seed42",
            Some(&manifest(Some("beta"), 256, "burn_ndarray::backend::NdArrayDevice")),
        );

        let app = DashApp::new(root.clone(), 0);
        let tabs = app.group_tabs();
        let keys: Vec<&str> = tabs.iter().map(|t| t.key.as_str()).collect();
        assert_eq!(keys, vec!["old", "alpha", "beta"], "one tab per model, in first-seen order");
        assert_eq!(tabs[1].count, 2, "both alpha runs share one tab");
        assert_eq!(tabs[0].label, "old", "run-id suffix becomes the model label");

        // Unnamed runs with no suffix group by their architecture signature.
        let root2 = temp_root("modelfallback");
        make_run_at(
            &root2,
            "20260101_000000_000_ppo_seed42",
            Some(&manifest(None, 128, "burn_ndarray::backend::NdArrayDevice")),
        );
        let fallback = DashApp::new(root2.clone(), 0);
        assert_eq!(
            fallback.group_tabs()[0].key,
            "ppo · h128 · NdArrayDevice",
            "no name → architecture label"
        );
        let _ = std::fs::remove_dir_all(&root);
        let _ = std::fs::remove_dir_all(&root2);
    }

    #[test]
    fn architecture_tabs_split_on_shape_and_backend() {
        let root = temp_root("archtabs");
        make_run_at(
            &root,
            "20260101_000000_000_ppo_seed42",
            Some(&manifest(Some("alpha"), 128, "burn_ndarray::backend::NdArrayDevice")),
        );
        make_run_at(
            &root,
            "20260102_000000_000_ppo_seed42",
            Some(&manifest(Some("beta"), 128, "burn_ndarray::backend::NdArrayDevice")),
        );
        make_run_at(
            &root,
            "20260103_000000_000_ppo_seed42",
            Some(&manifest(Some("gamma"), 256, "burn_ndarray::backend::NdArrayDevice")),
        );
        make_run_at(
            &root,
            "20260104_000000_000_ppo_seed42",
            Some(&manifest(Some("delta"), 128, "cubecl_wgpu::device::WgpuDevice")),
        );

        let mut app = DashApp::new(root.clone(), 0);
        app.group_mode = GroupMode::Architecture;
        let tabs = app.group_tabs();
        assert_eq!(tabs.len(), 3, "shape/backend variants get separate tabs");
        assert_eq!(tabs[0].count, 2, "alpha + beta share PPO/h128/CPU");
        assert_eq!(tabs[0].key, "ppo · h128 · NdArrayDevice");
        assert_eq!(tabs[2].key, "ppo · h128 · WgpuDevice");
        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn run_mode_uses_compact_labels_and_one_tab_per_run() {
        let root = temp_root("runtabs");
        make_run_at(&root, "20260101_101112_000_ppo_seed42", Some(&manifest(None, 128, "burn_ndarray::backend::NdArrayDevice")));
        make_run_at(&root, "20260102_131415_000_ppo_seed42", Some(&manifest(None, 128, "burn_ndarray::backend::NdArrayDevice")));

        let mut app = DashApp::new(root.clone(), 0);
        app.group_mode = GroupMode::Run;
        let tabs = app.group_tabs();
        assert_eq!(tabs.len(), 2);
        assert_eq!(tabs[0].label, "10:11:12 · ppo_seed42");
        assert_eq!(tabs[1].label, "13:14:15 · ppo_seed42");
        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn active_tab_filters_selection_for_every_view() {
        let root = temp_root("filter");
        make_run_at(
            &root,
            "20260101_000000_000_ppo_seed42",
            Some(&manifest(Some("alpha"), 128, "burn_ndarray::backend::NdArrayDevice")),
        );
        make_run_at(
            &root,
            "20260102_000000_000_ppo_seed42",
            Some(&manifest(Some("beta"), 128, "burn_ndarray::backend::NdArrayDevice")),
        );

        let mut app = DashApp::new(root.clone(), 0);
        assert_eq!(app.active_run_ids().len(), 2, "All tab sees every run");

        app.active_group = Some("beta".into());
        let beta_id = "20260102_000000_000_ppo_seed42".to_string();
        let alpha_id = "20260101_000000_000_ppo_seed42".to_string();
        assert_eq!(app.active_run_ids(), vec![beta_id.clone()]);
        assert!(app.run_is_selected(&beta_id, &app.runs[&beta_id]));
        assert!(
            !app.run_is_selected(&alpha_id, &app.runs[&alpha_id]),
            "runs outside the active tab are filtered even when checked"
        );

        // Select all / Uncheck all only touch the active tab's runs.
        for id in app.active_run_ids() {
            app.visible.insert(id, true);
        }
        app.visible.insert(alpha_id.clone(), true);
        for id in app.active_run_ids() {
            app.visible.insert(id, false);
        }
        assert!(!app.visible[&beta_id]);
        assert!(app.visible[&alpha_id], "other tabs keep their checkbox state");
        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn switching_group_mode_and_missing_group_fall_back_to_all() {
        let root = temp_root("fallback");
        make_run_at(
            &root,
            "20260101_000000_000_ppo_seed42",
            Some(&manifest(Some("alpha"), 128, "burn_ndarray::backend::NdArrayDevice")),
        );

        let mut app = DashApp::new(root.clone(), 0);
        app.active_group = Some("alpha".into());
        app.ensure_active_group();
        assert_eq!(app.active_group.as_deref(), Some("alpha"), "live group is kept");

        app.active_group = Some("deleted-model".into());
        app.ensure_active_group();
        assert_eq!(app.active_group, None, "removed group falls back to All");

        app.active_group = Some("alpha".into());
        app.group_mode = GroupMode::Run;
        app.ensure_active_group();
        assert_eq!(app.active_group, None, "a key from another mode is not valid");
        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn short_run_label_handles_nonstandard_ids() {
        assert_eq!(
            short_run_label("20260923_104623_870_ppo_seed42_smoke"),
            "10:46:23 · ppo_seed42_smoke"
        );
        assert_eq!(short_run_label("not-a-timestamp"), "not-a-timestamp");
    }

    #[test]
    fn tab_bar_and_all_filtered_views_render_for_the_active_tab() {
        let root = temp_root("render");
        make_run_at(
            &root,
            "20260101_000000_000_ppo_seed42",
            Some(&manifest(Some("alpha"), 128, "burn_ndarray::backend::NdArrayDevice")),
        );
        make_run_at(
            &root,
            "20260102_000000_000_ppo_seed42",
            Some(&manifest(Some("beta"), 128, "burn_ndarray::backend::NdArrayDevice")),
        );

        let mut app = DashApp::new(root.clone(), 0);
        app.active_group = Some("alpha".into());
        let alpha_id = "20260101_000000_000_ppo_seed42";
        let beta_id = "20260102_000000_000_ppo_seed42";

        let out = egui::Context::default().run_ui(Default::default(), |ctx| {
            egui::CentralPanel::default().show(ctx, |ui| {
                app.group_tab_bar(ui);
                app.kpi_strip(ui);
                app.plot_panel(ui);
                app.summary_table(ui);
                app.heatmap_panel(ui);
                app.table_panel(ui);
                app.bar_panel(ui);
                app.scatter_panel(ui);
            });
        });
        let texts: Vec<String> = out
            .shapes
            .iter()
            .filter_map(|c| match &c.shape {
                egui::Shape::Text(t) => Some(t.galley.text().to_string()),
                _ => None,
            })
            .collect();
        let joined = texts.join("\n");
        assert!(joined.contains("All (2)"), "All tab is always available");
        assert!(joined.contains("alpha (1)") && joined.contains("beta (1)"));
        assert!(
            joined.contains(alpha_id),
            "summary/table views list the active tab's run"
        );
        assert!(
            !joined.contains(beta_id),
            "views must not list runs from another tab"
        );
        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn env_metrics_render_and_drive_the_kpi_strip() {
        let root = temp_root("env");
        let id = "20260101_000000_000_ppo_seed42_env";
        let dir = root.join(id);
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(
            dir.join("metrics.jsonl"),
            concat!(
                r#"{"iteration":0,"mean_reward_per_step":0.1,"env":{"samples":10,"#,
                r#""alt_mean":1200.5,"alt_min":1100.0,"alt_max":1300.0,"#,
                r#""airspeed_mean":60.2,"climb_mean":0.4,"ground_speed_mean":61.0,"#,
                r#""roll_abs_mean":0.1,"pitch_abs_mean":0.05,"aoa_mean":0.02,"#,
                r#""wind_mean":4.5,"wind_max":9.0,"throttle_mean":0.55,"#,
                r#""surface_effort_mean":2.5,"baro_err_abs_mean":0.8,"#,
                r#""airspeed_err_abs_mean":0.3,"battery_pct_min":91.0,"#,
                r#""weather":{"samples":10,"wind_speed_mean":5.0,"#,
                r#""gust_rms_mean":1.2,"gust_rms_max":3.4,"turbulence_mean":1.0,"#,
                r#""severity_mean":1.0,"rain_mean":0.2,"visibility_min":5000.0,"#,
                r#""updraft_abs_mean":0.4,"shear_abs_mean":0.01}}}"#,
                "\n",
            ),
        )
        .unwrap();

        let env_idx = METRICS
            .iter()
            .position(|(k, _)| *k == "env_alt_mean")
            .expect("env altitude is registered");
        let mut app = DashApp::new(root.clone(), env_idx);
        let out = egui::Context::default().run_ui(Default::default(), |ctx| {
            egui::CentralPanel::default().show(ctx, |ui| {
                app.kpi_strip(ui);
                app.plot_panel(ui);
            });
        });
        let texts: Vec<String> = out
            .shapes
            .iter()
            .filter_map(|c| match &c.shape {
                egui::Shape::Text(t) => Some(t.galley.text().to_string()),
                _ => None,
            })
            .collect();
        let joined = texts.join("\n");
        assert!(
            joined.contains("Env altitude (m)"),
            "plot heading should name the env series, got: {joined}"
        );
        assert!(
            joined.contains("Env Alt"),
            "KPI strip should show the env altitude card, got: {joined}"
        );
        let _ = std::fs::remove_dir_all(&root);
    }
}
