//! `rl_dash` — a native desktop dashboard for `rl_agent` training telemetry.
//!
//! Reads the same per-run files the trainer writes
//! (`runs/<run_id>/run.json` + `runs/<run_id>/metrics.jsonl`, see
//! `rl_agent/src/telemetry.rs`), live-follows the newest run while
//! training runs, and lets you overlay any number of runs for comparison.
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
use std::time::Duration;

use eframe::egui;
use egui_plot::{Legend, Line, Plot, PlotPoints};
use serde::Deserialize;

/// Metric registry — keys mirror `metrics.jsonl` fields (and
/// `godot/scripts/rl_chart.gd` / `rl_agent/dashboard.html`).
const METRICS: &[(&str, &str)] = &[
    ("mean_reward_per_step", "Reward / step"),
    ("policy_loss", "Policy loss"),
    ("value_loss", "Value loss"),
    ("entropy", "Entropy"),
    ("advantage_mean", "Advantage mean"),
    ("alt_err", "Eval alt err (m)"),
    ("spd_err", "Eval spd err (m/s)"),
    ("eval_reward", "Eval mean reward"),
];

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
    checkpoint: bool,
    #[serde(default)]
    eval: Option<EvalMetrics>,
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
            "alt_err" => self.eval.as_ref().and_then(|e| e.alt_err),
            "spd_err" => self.eval.as_ref().and_then(|e| e.spd_err),
            "eval_reward" => self.eval.as_ref().and_then(|e| e.total_reward),
            _ => None,
        }
    }
}

// ── Run model ────────────────────────────────────────────────────────

struct RunModel {
    dir: PathBuf,
    manifest: Option<RunManifest>,
    records: Vec<MetricsRecord>,
    /// Byte offset of the last read in `metrics.jsonl` (incremental tail).
    file_len: u64,
    error: Option<String>,
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

    /// Append whatever is new since the last read. Truncation (file smaller
    /// than remembered) restarts from 0.
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
                self.error = Some("metrics.jsonl not found or unreadable".into());
                return;
            }
        };
        let Ok(len) = f.metadata().map(|m| m.len()) else {
            return;
        };
        if len < self.file_len {
            self.file_len = 0;
            self.records.clear();
        }
        if len == self.file_len {
            return;
        }
        if f.seek(SeekFrom::Start(self.file_len)).is_err() {
            return;
        }
        let mut buf = String::new();
        if f.read_to_string(&mut buf).is_err() {
            self.error = Some("failed to read metrics.jsonl".into());
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
        self.file_len += consumed as u64;
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

    /// Value series for one metric: `[(iteration, value), …]`.
    fn series(&self, key: &str) -> Vec<[f64; 2]> {
        self.records
            .iter()
            .filter_map(|r| r.metric(key).map(|v| [r.iteration as f64, v]))
            .collect()
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
        for dir in dirs {
            let id = dir
                .file_name()
                .map(|s| s.to_string_lossy().into_owned())
                .unwrap_or_else(|| dir.to_string_lossy().into_owned());
            seen.insert(id.clone(), ());
            if !self.runs.contains_key(&id) {
                self.visible.insert(id.clone(), true);
                self.runs.insert(id, RunModel::new(dir));
            }
        }
        for (id, run) in self.runs.iter_mut() {
            if seen.contains_key(id) {
                run.reload_manifest();
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
        for run in self.runs.values_mut() {
            run.poll();
        }
        // Discover runs that landed after launch (the RL tab's trainer spawns
        // its run dir within the first second or two) without a manual click.
        self.polls_since_rescan += 1;
        if self.polls_since_rescan >= RESCAN_EVERY_POLLS {
            self.polls_since_rescan = 0;
            self.rescan();
        }
        if self.follow_live {
            if let Some(newest) = self.runs.keys().next_back().cloned() {
                self.visible.insert(newest, true);
            }
        }
    }

    fn newest_id(&self) -> Option<String> {
        self.runs.keys().next_back().cloned()
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
        ctx.request_repaint_after(Duration::from_millis(POLL_MS));

        egui::Panel::left("runs")
            .resizable(true)
            .default_size(320.0)
            .show(ui, |ui| self.side_panel(ui));

        egui::CentralPanel::default().show(ui, |ui| {
            self.plot_panel(ui);
        });
    }
}

impl DashApp {
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

        let smooth_label = if self.window == 0 {
            "Smoothing: off".to_string()
        } else {
            format!("Smoothing: window {}", self.window)
        };
        ui.add(egui::Slider::new(&mut self.window, 0..=50).text(smooth_label));

        ui.horizontal(|ui| {
            let follow = &mut self.follow_live;
            ui.checkbox(follow, "Follow live run");
            if ui.button("Rescan").clicked() {
                self.rescan();
            }
        });

        if let Some(banner) = self.banner.clone() {
            ui.colored_label(egui::Color32::YELLOW, banner);
        }

        ui.separator();
        ui.label(egui::RichText::new("Runs").strong());

        egui::ScrollArea::vertical().show(ui, |ui| {
            let newest = self.newest_id();
            let ids: Vec<String> = self.runs.keys().cloned().collect();
            for id in ids {
                let run = &self.runs[&id];
                let status = run.status().to_string();
                let live = run.is_live();
                let color = if live {
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
                    let status_text = if live { "● live" } else { status.as_str() };
                    ui.colored_label(color, status_text);
                });
                if run.error.is_none() {
                    ui.label(
                        egui::RichText::new(format!(
                            "    {} · {} points{}",
                            run.manifest
                                .as_ref()
                                .and_then(|m| m.algo.clone())
                                .unwrap_or_else(|| "?".into()),
                            series_len,
                            run.manifest
                                .as_ref()
                                .and_then(|m| m.git_commit.clone())
                                .map(|c| format!(" · {c}"))
                                .unwrap_or_default()
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

        // Gather the visible, non-empty series.
        let mut lines: Vec<Line> = Vec::new();
        let mut shown_runs: Vec<(String, f64)> = Vec::new();
        for (i, (id, run)) in self.runs.iter().enumerate() {
            if !self.visible.get(id).copied().unwrap_or(false) {
                continue;
            }
            let pts = smooth(&run.series(key), window);
            if pts.is_empty() {
                continue;
            }
            if let Some(last) = pts.last() {
                shown_runs.push((id.clone(), last[1]));
            }
            lines.push(
                Line::new(id.as_str(), PlotPoints::from(pts))
                    .color(PALETTE[i % PALETTE.len()])
                    .width(2.0),
            );
        }

        ui.horizontal(|ui| {
            ui.heading(label);
            if lines.is_empty() {
                ui.label(
                    egui::RichText::new("  no visible data — tick a run on the left (or train!)")
                        .weak(),
                );
            } else {
                ui.label(egui::RichText::new(format!("  {} run(s)", lines.len())).weak());
            }
        });

        let mut hover: Option<egui_plot::PlotPoint> = None;
        if !lines.is_empty() {
            // Auto-bounds are on by default, so new points always fit the plot.
            let _plot_resp = Plot::new("metrics")
                .legend(Legend::default())
                .x_axis_label("iteration")
                .y_axis_label(label)
                .show(ui, |plot_ui| {
                    for line in lines {
                        plot_ui.line(line);
                    }
                    hover = plot_ui.pointer_coordinate();
                });

            if let Some(p) = hover {
                ui.label(
                    egui::RichText::new(format!("cursor → iter {:.0} · {}", p.x, fmt(p.y)))
                        .monospace()
                        .weak(),
                );
            } else {
                ui.label(
                    egui::RichText::new("hover the plot for values · drag to pan · scroll to zoom")
                        .weak()
                        .small(),
                );
            }
        }

        // Latest-value strip for the visible runs.
        if !shown_runs.is_empty() {
            ui.separator();
            ui.label(egui::RichText::new("Latest values").strong());
            ui.horizontal_wrapped(|ui| {
                for (i, (id, v)) in shown_runs.iter().enumerate() {
                    let color = PALETTE[i % PALETTE.len()];
                    ui.colored_label(color, format!("{id}: {}", fmt(*v)));
                }
            });
        }

        // Run details for the first visible run (its manifest).
        if let Some((id, _)) = shown_runs.first().cloned() {
            if let Some(run) = self.runs.get(&id) {
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

    const SAMPLE: &str = r#"{"iteration":2,"elapsed_s":16.4,"steps_per_s":180.7,"policy_loss":-0.032,"value_loss":8.58,"entropy":3.67,"advantage_mean":-0.46,"advantage_std":2.88,"mean_reward_per_step":0.059,"mean_episode_steps":0.0,"crash_rate":0.0,"episodes_completed":0,"eval":{"alt_err":99.4,"spd_err":11.1,"total_reward":-4.26,"crashes":0,"episodes":1,"mean_steps":2000.0},"checkpoint":true}"#;

    #[test]
    fn parses_full_record_and_flattens_metrics() {
        let rec: MetricsRecord = serde_json::from_str(SAMPLE).expect("sample must parse");
        assert_eq!(rec.iteration, 2);
        assert_eq!(rec.elapsed_s, Some(16.4));
        assert!((rec.metric("mean_reward_per_step").unwrap() - 0.059).abs() < 1e-9);
        assert!((rec.metric("alt_err").unwrap() - 99.4).abs() < 1e-9);
        assert!((rec.metric("eval_reward").unwrap() + 4.26).abs() < 1e-9);
        // Non-eval iteration has no eval metric.
        let no_eval: MetricsRecord =
            serde_json::from_str(r#"{"iteration":3,"mean_reward_per_step":0.5}"#).unwrap();
        assert_eq!(no_eval.metric("alt_err"), None);
        assert_eq!(no_eval.metric("mean_reward_per_step"), Some(0.5));
        assert_eq!(no_eval.metric("entropy"), None);
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
            let has_data = app.runs.values().any(|r| !r.series(key).is_empty());
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
        let _ = std::fs::remove_dir_all(&root);
    }
}
