//! Structured training telemetry.
//!
//! Every training run writes an organized, machine-readable record under
//! `run_root/<run_id>/`:
//!
//! ```text
//! runs/
//! └── 20260923_143512_123_ppo_seed42_smoke/
//!     ├── run.json        # manifest: config, timestamps, git commit, argv
//!     ├── metrics.jsonl   # one JSON object per PPO iteration (append-only)
//!     └── episodes.jsonl  # one JSON object per finished rollout episode
//! ```
//!
//! - `run_id` is a UTC timestamp + algorithm + seed (+ an optional human
//!   label), so runs sort lexicographically by start time and never collide.
//! - `run.json` is written on start (status `running`), then rewritten on
//!   completion with end time + `completed`, or by [`Drop`] with `interrupted`
//!   if the process exits abnormally.
//! - `metrics.jsonl` is flushed after every iteration, so a live consumer
//!   (the Godot RL panel, a shell `tail -f`, an HTML dashboard) can watch the
//!   file as training progresses — no schema changes needed to add fields.
//!   Each iteration carries an optional [`EnvSummary`] (`env` key) describing
//!   the flight state seen during that rollout.
//! - `episodes.jsonl` gets one [`EpisodeRecord`] per episode that ends during
//!   the rollout, with its termination cause and the same environment summary.
//!   Eval episodes are deliberately excluded: they are a periodic probe, not
//!   part of the training distribution.
//!
//! The writer is plain `std` (no external deps): UTC timestamps are computed
//! from `SystemTime` with the Howard Hinnant civil-from-days algorithm.

use crate::algo::AlgoSpec;
use crate::net::NetConfig;
use crate::trainer::{EvalMetrics, TrainerConfig};
use serde::{Deserialize, Serialize};
use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};
use std::time::{Instant, SystemTime, UNIX_EPOCH};

/// Telemetry schema version — bump on breaking `metrics.jsonl` layouts.
pub const TELEMETRY_SCHEMA: &str = "rl_agent.telemetry.v1";

const MANIFEST: &str = "run.json";
const METRICS: &str = "metrics.jsonl";
const EPISODES: &str = "episodes.jsonl";

/// Optional label added to the run id; the label is slugified
/// (lowercased, non-alphanumerics → `_`).
const DEFAULT_RUN_ROOT: &str = "runs";

/// Controls where (and whether) telemetry is written.
#[derive(Debug, Clone)]
pub struct TelemetryConfig {
    /// Root directory for per-run subdirectories.
    pub run_root: String,
    /// Optional human label embedded in the run id.
    pub name: Option<String>,
}

impl Default for TelemetryConfig {
    fn default() -> Self {
        Self {
            run_root: DEFAULT_RUN_ROOT.to_string(),
            name: None,
        }
    }
}

impl TelemetryConfig {
    /// No telemetry at all — used by library callers and tests.
    pub fn disabled() -> Self {
        Self {
            run_root: String::new(),
            name: None,
        }
    }

    fn enabled(&self) -> bool {
        !self.run_root.is_empty()
    }
}

/// Static metadata captured once per run and persisted in `run.json`.
pub struct RunMeta<'a> {
    pub algo: &'a str,
    pub backend: &'a str,
    pub tcfg: &'a TrainerConfig,
    pub ncfg: &'a NetConfig,
    pub spec: &'a AlgoSpec,
    pub mission: Option<&'a crate::contracts::MissionSummary>,
    pub reward_profile: Option<&'a str>,
    pub specialization: Option<&'a str>,
    pub contract: &'a str,
    pub expert: Option<&'a crate::policy::ExpertCard>,
}

/// An append-only writer for one training run.
///
/// Call [`record`][Self::record] once per PPO iteration. [`finish`][Self::finish]
/// must be called on a normal exit; if the writer is dropped unfinished the
/// manifest is rewritten as `interrupted`.
pub struct TelemetryWriter {
    run_id: String,
    run_dir: PathBuf,
    metrics: Option<BufWriter<File>>,
    episodes: Option<BufWriter<File>>,
    started: Instant,
    finished: bool,
}

#[derive(Serialize)]
struct Manifest<'a> {
    schema: &'static str,
    run_id: &'a str,
    started_utc: String,
    ended_utc: Option<String>,
    status: &'a str,
    algo: &'a str,
    backend: &'a str,
    command_line: Vec<String>,
    git_commit: Option<String>,
    trainer: &'a TrainerConfig,
    network: &'a NetConfig,
    algorithm: &'a AlgoSpec,
    #[serde(skip_serializing_if = "Option::is_none")]
    mission: Option<&'a crate::contracts::MissionSummary>,
    #[serde(skip_serializing_if = "Option::is_none")]
    reward_profile: Option<&'a str>,
    #[serde(skip_serializing_if = "Option::is_none")]
    specialization: Option<&'a str>,
    contract: &'a str,
    #[serde(skip_serializing_if = "Option::is_none")]
    expert: Option<&'a crate::policy::ExpertCard>,
}

/// Per-iteration telemetry. Serde-flattened eval metrics ride along when an
/// evaluation ran this iteration.
#[derive(Debug, Clone, Serialize)]
pub struct IterationRecord<'a> {
    /// Iteration index (0-based; the console reports 1-based).
    pub iteration: usize,
    /// Wall time since the run started (seconds).
    pub elapsed_s: f64,
    /// Environment steps per second since the previous iteration.
    pub steps_per_s: f64,
    /// Mean policy surrogate loss (negative for an improving policy).
    pub policy_loss: f32,
    /// Mean squared value error.
    pub value_loss: f32,
    /// Mean policy entropy.
    pub entropy: f32,
    /// Mean advantage over the rollout (before minibatch whitening).
    pub advantage_mean: f32,
    /// Standard deviation of the advantages.
    pub advantage_std: f32,
    /// Mean per-step reward over the rollout (crash penalties included).
    pub mean_reward_per_step: f32,
    /// Mean steps per episode completed during this rollout. Episodes can
    /// span iteration boundaries, so this is the true episode length, not
    /// just the steps collected since the boundary.
    pub mean_episode_steps: f32,
    /// Fraction of completed episodes that ended in a crash (0..=1).
    pub crash_rate: f32,
    /// Episodes finished during this rollout (crash or step-budget).
    pub episodes_completed: usize,
    /// Deterministic-evaluation metrics, present on eval iterations.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub eval: Option<&'a EvalMetrics>,
    /// Ground-truth flight state summarised over this rollout, present when
    /// the environment can report it (see [`crate::trainer::EnvSpec::frame`]).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub env: Option<EnvSummary>,
    /// Whether a checkpoint was written this iteration.
    pub checkpoint: bool,
    /// Live-stream counters, present when frame streaming is enabled.
    ///
    /// `dropped` is the honest part: a viewer that missed frames would
    /// otherwise be indistinguishable from an aircraft that sat still, so the
    /// loss is reported rather than hidden.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stream: Option<StreamRecord>,
    /// Mission-mode captures during this rollout. `None` for avionics runs,
    /// so their records are byte-for-byte unchanged.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub mission_captures: Option<usize>,
    /// Mission-mode returns home during this rollout.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub mission_returns: Option<usize>,
    /// Mission-mode successes (captured *and* returned) during this rollout.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub mission_successes: Option<usize>,
    /// Mean steps per completed episode spent in
    /// `[outbound, capture, return]`. `return` folds in the terminal `Done`
    /// phase, so the three entries cover every step of a mission.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub mean_phase_steps: Option<[f64; 3]>,
}

/// Frame-streaming counters attached to an iteration record.
#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize)]
pub struct StreamRecord {
    /// Frames handed to the writer thread so far.
    pub frames: u64,
    /// Frames dropped because the consumer could not keep up.
    pub dropped: u64,
    /// Episodes flushed to `episodes/`.
    pub episodes: u64,
}

/// Environment telemetry summarised over a window of rollout steps.
///
/// This is the ground-truth counterpart to the agent's noisy observation: the
/// reward curve says *whether* the policy improved, this says *how* it flew.
/// Values are plain `f64` in SI units so a JSON consumer can plot them without
/// knowing the simulation's internals.
#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize)]
pub struct EnvSummary {
    /// Steps accumulated into this summary (0 means "no data").
    pub samples: u64,
    /// Mean/min/max true altitude (m).
    pub alt_mean: f64,
    pub alt_min: f64,
    pub alt_max: f64,
    /// Mean/min/max true airspeed (m/s).
    pub airspeed_mean: f64,
    pub airspeed_min: f64,
    pub airspeed_max: f64,
    /// Mean/min/max climb rate (m/s, positive climbing).
    pub climb_mean: f64,
    pub climb_min: f64,
    pub climb_max: f64,
    /// Mean ground speed (m/s).
    pub ground_speed_mean: f64,
    /// Mean absolute bank / pitch (rad).
    pub roll_abs_mean: f64,
    pub pitch_abs_mean: f64,
    /// Mean angle of attack (rad) and mean absolute sideslip (rad).
    pub aoa_mean: f64,
    pub sideslip_abs_mean: f64,
    /// Mean and max wind magnitude (m/s).
    pub wind_mean: f64,
    pub wind_max: f64,
    /// Mean commanded throttle `[0, 1]`.
    pub throttle_mean: f64,
    /// Mean absolute surface deflection (deg) across elevator/aileron/rudder.
    pub surface_effort_mean: f64,
    /// Mean absolute barometric altitude error (m) — perceived vs true.
    pub baro_err_abs_mean: f64,
    /// Mean absolute indicated-airspeed error (m/s) — perceived vs true.
    pub airspeed_err_abs_mean: f64,
    /// Lowest battery charge seen (%), or 0 when the stack has no battery.
    pub battery_pct_min: f64,
    /// Weather summary, present only when the env has a weather system.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub weather: Option<WeatherSummary>,
}

/// Weather telemetry summarised over the same window as [`EnvSummary`].
#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize)]
pub struct WeatherSummary {
    /// Steps accumulated into this summary.
    pub samples: u64,
    /// Mean steady wind speed of the active scene (m/s).
    pub wind_speed_mean: f64,
    /// Mean and max effective gust RMS (m/s).
    pub gust_rms_mean: f64,
    pub gust_rms_max: f64,
    /// Mean turbulence band index (`Light = 0` … `Extreme = 3`).
    pub turbulence_mean: f64,
    /// Mean severity rank (`Clear = 0` … `Severe = 3`).
    pub severity_mean: f64,
    /// Mean rain intensity `[0, 1]`.
    pub rain_mean: f64,
    /// Lowest visibility seen (m).
    pub visibility_min: f64,
    /// Mean absolute updraft strength (m/s).
    pub updraft_abs_mean: f64,
    /// Mean absolute wind-shear gradient ((m/s)/m).
    pub shear_abs_mean: f64,
}

/// One finished rollout episode, written to `episodes.jsonl`.
#[derive(Debug, Clone, Serialize)]
pub struct EpisodeRecord<'a> {
    /// Monotonic episode counter across the run.
    pub episode: u32,
    /// Iteration during which the episode ended.
    pub iteration: usize,
    /// Environment steps in the episode.
    pub steps: usize,
    /// Episode wall-clock duration implied by `steps * env_dt` (s).
    pub duration_s: f64,
    /// Cumulative shaped reward, crash penalty included.
    pub reward_sum: f64,
    /// Ended on a terminal state (ground impact or weather limit).
    pub terminated: bool,
    /// Ended because the step budget ran out.
    pub truncated: bool,
    /// Why the episode ended: `ground_impact`, `gust_limit`,
    /// `visibility_limit`, `severe_exposure`, `step_budget`, or `terminated`.
    pub termination: &'a str,
    /// Environment telemetry accumulated over the episode.
    pub env: EnvSummary,
}

impl TelemetryWriter {
    /// Start a run: create the run directory and write the manifest.
    ///
    /// Returns `None` when telemetry is disabled.
    pub fn start(
        cfg: &TelemetryConfig,
        meta: &RunMeta<'_>,
    ) -> Result<Option<Self>, Box<dyn std::error::Error>> {
        if !cfg.enabled() {
            return Ok(None);
        }
        let now = SystemTime::now().duration_since(UNIX_EPOCH).unwrap_or_default();
        let run_id = run_id(now, meta.algo, meta.tcfg.seed, cfg.name.as_deref());
        let run_dir = Path::new(&cfg.run_root).join(&run_id);
        std::fs::create_dir_all(&run_dir)?;

        let manifest = Manifest {
            schema: TELEMETRY_SCHEMA,
            run_id: &run_id,
            started_utc: iso8601_utc(now.as_secs()),
            ended_utc: None,
            status: "running",
            algo: meta.algo,
            backend: meta.backend,
            command_line: std::env::args().collect(),
            git_commit: git_head_short(),
            trainer: meta.tcfg,
            network: meta.ncfg,
            algorithm: meta.spec,
            mission: meta.mission,
            reward_profile: meta.reward_profile,
            specialization: meta.specialization,
            contract: meta.contract,
            expert: meta.expert,
        };
        write_json(&run_dir.join(MANIFEST), &serde_json::to_string_pretty(&manifest)?)?;

        let file = File::options().create(true).append(true).open(run_dir.join(METRICS))?;
        let episodes =
            File::options().create(true).append(true).open(run_dir.join(EPISODES))?;
        Ok(Some(Self {
            run_id,
            run_dir,
            metrics: Some(BufWriter::new(file)),
            episodes: Some(BufWriter::new(episodes)),
            started: Instant::now(),
            finished: false,
        }))
    }

    /// Identifier of this run; used by consumers to locate the telemetry files.
    pub fn run_id(&self) -> &str {
        &self.run_id
    }

    /// Directory this run's telemetry lives in.
    pub fn run_dir(&self) -> &Path {
        &self.run_dir
    }

    /// Append one iteration record and flush (so `tail -f` / the Godot panel
    /// see it immediately). Marks the manifest as `completed` on success.
    pub fn record(&mut self, rec: &IterationRecord<'_>) -> Result<(), Box<dyn std::error::Error>> {
        let Some(w) = self.metrics.as_mut() else {
            return Ok(());
        };
        serde_json::to_writer(&mut *w, rec)?;
        w.write_all(b"\n")?;
        w.flush()?;
        Ok(())
    }

    /// Append one finished-episode record and flush.
    ///
    /// Called from inside the rollout as soon as an episode ends, so a live
    /// consumer sees episode boundaries without waiting for the iteration to
    /// finish. The cost is one flush per episode (thousands of steps), not per
    /// step, so it stays off the hot path.
    pub fn record_episode(
        &mut self,
        rec: &EpisodeRecord<'_>,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let Some(w) = self.episodes.as_mut() else {
            return Ok(());
        };
        serde_json::to_writer(&mut *w, rec)?;
        w.write_all(b"\n")?;
        w.flush()?;
        Ok(())
    }

    /// Mark the run as completed with the final wall time.
    pub fn finish(&mut self) -> Result<(), Box<dyn std::error::Error>> {
        if self.finished {
            return Ok(());
        }
        self.finished = true;
        let now = SystemTime::now().duration_since(UNIX_EPOCH).unwrap_or_default();
        self.rewrite_manifest("completed", Some(now.as_secs()))
    }

    /// Wall-clock seconds since the run started.
    pub fn elapsed_secs(&self) -> f64 {
        self.started.elapsed().as_secs_f64()
    }

    fn rewrite_manifest(&self, status: &str, ended_utc: Option<u64>) -> Result<(), Box<dyn std::error::Error>> {
        let json = std::fs::read_to_string(self.run_dir.join(MANIFEST))?;
        let mut manifest: serde_json::Value = serde_json::from_str(&json)?;
        manifest["status"] = serde_json::Value::String(status.to_string());
        manifest["ended_utc"] = match ended_utc {
            Some(secs) => serde_json::Value::String(iso8601_utc(secs)),
            None => serde_json::Value::Null,
        };
        write_json(
            &self.run_dir.join(MANIFEST),
            &serde_json::to_string_pretty(&manifest)?,
        )
    }
}

impl Drop for TelemetryWriter {
    fn drop(&mut self) {
        if !self.finished {
            // Best-effort: mark interrupted so run.json stays an honest record.
            let _ = self.rewrite_manifest("interrupted", None);
        }
    }
}

/// Human + machine readable run id, e.g. `20260923_143512_123_ppo_seed42`.
/// Milliseconds keep same-second runs distinct; the optional label is
/// slugified and appended last.
fn run_id(now: std::time::Duration, algo: &str, seed: u64, name: Option<&str>) -> String {
    let secs = now.as_secs();
    let millis = now.subsec_millis();
    let (y, mo, d, h, mi, s) = civil_from_secs(secs);
    let mut id = format!("{y:04}{mo:02}{d:02}_{h:02}{mi:02}{s:02}_{millis:03}_{algo}_seed{seed}");
    if let Some(label) = name.map(slugify).filter(|s| !s.is_empty()) {
        id.push('_');
        id.push_str(&label);
    }
    id
}

/// `YYYY-MM-DDTHH:MM:SSZ` from seconds since the Unix epoch (UTC).
fn iso8601_utc(secs: u64) -> String {
    let (y, mo, d, h, mi, s) = civil_from_secs(secs);
    format!("{y:04}-{mo:02}-{d:02}T{h:02}:{mi:02}:{s:02}Z")
}

fn civil_from_secs(secs: u64) -> (i64, u32, u32, u32, u32, u32) {
    let days = (secs / 86_400) as i64;
    let rem = secs % 86_400;
    let (h, mi, s) = (rem / 3600, (rem % 3600) / 60, rem % 60);
    let (y, m, d) = civil_from_days(days);
    (y, m, d, h as u32, mi as u32, s as u32)
}

/// Howard Hinnant's civil-from-days algorithm; days since 1970-01-01.
fn civil_from_days(z: i64) -> (i64, u32, u32) {
    let z = z + 719_468;
    let era = if z >= 0 { z } else { z - 146_096 } / 146_097;
    let doe = (z - era * 146_097) as u64; // [0, 146096]
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365; // [0, 399]
    let y = yoe as i64 + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100); // [0, 365]
    let mp = (5 * doy + 2) / 153; // [0, 11]
    let d = (doy - (153 * mp + 2) / 5 + 1) as u32; // [1, 31]
    let m = if mp < 10 { mp + 3 } else { mp - 9 } as u32; // [1, 12]
    (if m <= 2 { y + 1 } else { y }, m, d)
}

fn slugify(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    for c in s.chars() {
        if c.is_alphanumeric() {
            out.extend(c.to_lowercase());
        } else if !out.ends_with('_') {
            out.push('_');
        }
    }
    out.trim_matches('_').to_string()
}

fn git_head_short() -> Option<String> {
    let out = std::process::Command::new("git")
        .args(["rev-parse", "--short", "HEAD"])
        .output()
        .ok()?;
    if out.status.success() {
        Some(String::from_utf8_lossy(&out.stdout).trim().to_string())
    } else {
        None
    }
}

fn write_json(path: &Path, contents: &str) -> Result<(), Box<dyn std::error::Error>> {
    std::fs::write(path, contents)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn iso8601_utc_matches_epoch_and_recent() {
        assert_eq!(iso8601_utc(0), "1970-01-01T00:00:00Z");
        assert_eq!(iso8601_utc(86_400), "1970-01-02T00:00:00Z");
        // 2026-09-23T00:00:00Z (a fixed epoch value).
        assert_eq!(iso8601_utc(1_790_121_600), "2026-09-23T00:00:00Z");
    }

    #[test]
    fn run_id_is_unique_and_lexicographic() {
        let a = run_id(std::time::Duration::from_secs(1_789_862_401), "ppo", 42, None);
        let b = run_id(std::time::Duration::from_secs(1_789_862_402), "ppo", 42, None);
        // Same-second runs differ by millis field.
        let c = run_id(std::time::Duration::from_millis(1_789_862_401_123), "ppo", 42, None);
        assert!(a < b, "run ids must sort by start time: {a} vs {b}");
        assert_ne!(a, c);
        assert!(a.contains("_ppo_seed42"));
        assert!(c.ends_with("_123_ppo_seed42"));
    }

    #[test]
    fn run_id_slugifies_labels() {
        let id = run_id(std::time::Duration::from_millis(1), "ppo", 7, Some("  Smoke A/B Run! "));
        assert!(id.ends_with("_smoke_a_b_run"), "got {id}");
    }

    #[test]
    fn writer_records_and_finishes() {
        use crate::net::NetConfig;
        use crate::trainer::TrainerConfig;

        let root = std::env::temp_dir().join("rl_agent_telemetry_test");
        let _ = std::fs::remove_dir_all(&root);
        let cfg = TelemetryConfig {
            run_root: root.to_str().unwrap().to_string(),
            name: Some("unit".to_string()),
        };
        let tcfg = TrainerConfig::default();
        let ncfg = NetConfig {
            obs_dim: 19,
            hidden: 32,
            action_dim: 4,
            log_std_init: -0.5,
        };
        let spec = AlgoSpec::Ppo {
            config: Default::default(),
        };
        let meta = RunMeta {
            algo: "ppo",
            backend: "cpu",
            tcfg: &tcfg,
            ncfg: &ncfg,
            spec: &spec,
            mission: None,
            reward_profile: None,
            specialization: None,
            contract: "avionics",
            expert: None,
        };
        let mut w = TelemetryWriter::start(&cfg, &meta)
            .expect("start")
            .expect("enabled run writes telemetry");
        assert!(w.run_id().contains("_ppo_seed42"));

        let ev = EvalMetrics {
            alt_err: 12.5,
            spd_err: 3.25,
            total_reward: 40.0,
            crashes: 1,
            episodes: 2,
            mean_steps: 512.0,
        };
        let rec = IterationRecord {
            iteration: 0,
            elapsed_s: 1.0,
            steps_per_s: 2048.0,
            policy_loss: -0.1,
            value_loss: 0.5,
            entropy: 1.2,
            advantage_mean: 0.05,
            advantage_std: 0.9,
            mean_reward_per_step: 0.02,
            mean_episode_steps: 256.0,
            crash_rate: 0.25,
            episodes_completed: 4,
            eval: Some(&ev),
            env: Some(EnvSummary {
                samples: 2048,
                alt_mean: 1200.0,
                alt_min: 1100.0,
                alt_max: 1300.0,
                airspeed_mean: 60.0,
                airspeed_min: 55.0,
                airspeed_max: 65.0,
                climb_mean: 0.5,
                climb_min: -2.0,
                climb_max: 3.0,
                ground_speed_mean: 61.0,
                roll_abs_mean: 0.1,
                pitch_abs_mean: 0.05,
                aoa_mean: 0.02,
                sideslip_abs_mean: 0.01,
                wind_mean: 4.0,
                wind_max: 9.0,
                throttle_mean: 0.55,
                surface_effort_mean: 2.5,
                baro_err_abs_mean: 0.8,
                airspeed_err_abs_mean: 0.3,
                battery_pct_min: 91.0,
                weather: Some(WeatherSummary {
                    samples: 2048,
                    wind_speed_mean: 5.0,
                    gust_rms_mean: 1.2,
                    gust_rms_max: 3.4,
                    turbulence_mean: 1.0,
                    severity_mean: 1.0,
                    rain_mean: 0.2,
                    visibility_min: 5000.0,
                    updraft_abs_mean: 0.4,
                    shear_abs_mean: 0.01,
                }),
            }),
            checkpoint: false,
            stream: None,
            mission_captures: None,
            mission_returns: None,
            mission_successes: None,
            mean_phase_steps: None,
        };
        w.record(&rec).expect("record");
        w.record_episode(&EpisodeRecord {
            episode: 7,
            iteration: 0,
            steps: 512,
            duration_s: 17.07,
            reward_sum: 42.5,
            terminated: false,
            truncated: true,
            termination: "step_budget",
            env: rec.env.expect("env summary"),
        })
        .expect("record episode");
        w.finish().expect("finish");

        // Files exist and contain the expected shape.
        let metrics = std::fs::read_to_string(w.run_dir().join(METRICS)).expect("metrics.jsonl");
        let line: serde_json::Value = serde_json::from_str(metrics.trim()).expect("json line");
        assert_eq!(line["iteration"], 0);
        assert_eq!(line["eval"]["episodes"], 2);
        assert_eq!(line["mean_reward_per_step"], 0.02);
        assert_eq!(line["env"]["alt_mean"], 1200.0);
        assert_eq!(line["env"]["weather"]["gust_rms_max"], 3.4);

        // Episode rows are a separate append-only file, one JSON object each.
        let episodes =
            std::fs::read_to_string(w.run_dir().join(EPISODES)).expect("episodes.jsonl");
        let ep: serde_json::Value = serde_json::from_str(episodes.trim()).expect("episode json");
        assert_eq!(ep["episode"], 7);
        assert_eq!(ep["termination"], "step_budget");
        assert_eq!(ep["env"]["samples"], 2048);

        let manifest: serde_json::Value = serde_json::from_str(
            &std::fs::read_to_string(w.run_dir().join(MANIFEST)).expect("run.json"),
        )
        .expect("manifest json");
        assert_eq!(manifest["status"], "completed");
        assert_eq!(manifest["algo"], "ppo");
        assert!(manifest["started_utc"].as_str().unwrap().ends_with('Z'));

        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn disabled_config_writes_nothing() {
        let cfg = TelemetryConfig::disabled();
        let tcfg = TrainerConfig::default();
        let ncfg = NetConfig {
            obs_dim: 4,
            hidden: 8,
            action_dim: 2,
            log_std_init: -0.5,
        };
        let spec = AlgoSpec::Ppo {
            config: Default::default(),
        };
        let meta = RunMeta {
            algo: "ppo",
            backend: "cpu",
            tcfg: &tcfg,
            ncfg: &ncfg,
            spec: &spec,
            mission: None,
            reward_profile: None,
            specialization: None,
            contract: "avionics",
            expert: None,
        };
        let w = TelemetryWriter::start(&cfg, &meta).expect("start");
        assert!(w.is_none());
    }

    /// Minimal record so the mission-key serialization tests stay focused on
    /// the new fields rather than the existing schema.
    fn minimal_record<'a>(
        captures: Option<usize>,
        returns: Option<usize>,
        successes: Option<usize>,
        phase_steps: Option<[f64; 3]>,
    ) -> IterationRecord<'a> {
        IterationRecord {
            iteration: 0,
            elapsed_s: 1.0,
            steps_per_s: 10.0,
            policy_loss: 0.0,
            value_loss: 0.0,
            entropy: 0.0,
            advantage_mean: 0.0,
            advantage_std: 0.0,
            mean_reward_per_step: 0.0,
            mean_episode_steps: 0.0,
            crash_rate: 0.0,
            episodes_completed: 0,
            eval: None,
            env: None,
            checkpoint: false,
            stream: None,
            mission_captures: captures,
            mission_returns: returns,
            mission_successes: successes,
            mean_phase_steps: phase_steps,
        }
    }

    #[test]
    fn mission_records_carry_capture_counters() {
        let rec = minimal_record(Some(2), Some(1), Some(1), Some([10.0, 4.0, 6.0]));
        let json: serde_json::Value = serde_json::to_value(&rec).expect("serialize");
        assert_eq!(json["mission_captures"], 2);
        assert_eq!(json["mission_returns"], 1);
        assert_eq!(json["mission_successes"], 1);
        assert_eq!(json["mean_phase_steps"][0], 10.0);
        assert_eq!(json["mean_phase_steps"][2], 6.0);
    }

    #[test]
    fn avionics_records_omit_mission_counters() {
        let rec = minimal_record(None, None, None, None);
        let json: serde_json::Value = serde_json::to_value(&rec).expect("serialize");
        for key in [
            "mission_captures",
            "mission_returns",
            "mission_successes",
            "mean_phase_steps",
        ] {
            assert!(
                json.get(key).is_none(),
                "avionics records must not grow a `{key}` key"
            );
        }
    }

    #[test]
    fn manifest_records_mission_metadata() {
        use crate::contracts::MissionSummary;
        let dir = std::env::temp_dir().join(format!(
            "rl_agent_manifest_mission_{}",
            std::process::id()
        ));
        let _ = std::fs::remove_dir_all(&dir);
        let tcfg = TrainerConfig {
            iterations: 1,
            rollout_steps: 8,
            ..Default::default()
        };
        let ncfg = NetConfig { obs_dim: 32, ..Default::default() };
        let spec = AlgoSpec::from_id("ppo").unwrap();
        let summary = MissionSummary {
            airframe: "m_sweep_mini.toml".into(),
            contract: crate::contracts::MISSION_CONTRACT.into(),
            reward_profile: "balanced".into(),
            cruise_altitude: 1000.0,
            cruise_airspeed: 55.0,
            pickup_radius: 40.0,
            pickup_vertical_tolerance: 15.0,
            capture_speed: 11.5,
            capture_dwell_steps: 15,
            max_steps: 4500,
        };
        let mut w = TelemetryWriter::start(
            &TelemetryConfig { run_root: dir.to_string_lossy().into(), name: Some("mt".into()) },
            &RunMeta {
                algo: "ppo",
                backend: "cpu",
                tcfg: &tcfg,
                ncfg: &ncfg,
                spec: &spec,
                mission: Some(&summary),
                reward_profile: Some("balanced"),
                specialization: Some("full"),
                contract: crate::contracts::MISSION_CONTRACT,
                expert: None,
            },
        )
        .unwrap()
        .unwrap();
        w.finish().unwrap();
        let json: serde_json::Value =
            serde_json::from_str(&std::fs::read_to_string(w.run_dir().join("run.json")).unwrap()).unwrap();
        assert_eq!(json["mission"]["contract"], "mission.v1");
        assert_eq!(json["mission"]["capture_speed"], 11.5);
        assert_eq!(json["contract"], "mission.v1");
        let _ = std::fs::remove_dir_all(&dir);
    }
}