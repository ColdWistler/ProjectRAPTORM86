//! Milestone 7 — Weather Exam Suite
//!
//! Generates 10 mission scenarios that reuse the same geometry contract as the
//! mission exam (`MissionConfig`, `MissionScenario`) but vary the weather
//! scene through [`WeatherConfig`] presets (`Clear`, `LightTurb`, `Rain`,
//! `Storm`, `ExtremeStorm`). Each preset appears twice for a 10-episode exam.
//!
//! The exam measures:
//! - `success_rate` (mission completed: captured + returned)
//! - `capture_rate`
//! - `crash_rate` (ground impact or out-of-bounds)
//! - Wilson 95 % confidence intervals per preset group
//!
//! No new physics is added; weather scenes pass through the existing
//! [`MissionEnvironment`] via `EnvConfig.weather_config`.

use serde::{Serialize, Deserialize};
use flight_core::{
    MissionConfig, MissionEnvironment, MissionPhase, MissionScenario, WeatherConfig,
    WeatherPreset,
};
use std::collections::BTreeMap;

/// Number of weather exam episodes (2 per preset × 5 presets = 10).
pub const WEATHER_SCENARIO_COUNT: usize = 10;

/// Presets covered by the exam, in display order.
pub const PRESETS: [WeatherPreset; 5] = [
    WeatherPreset::Clear,
    WeatherPreset::LightTurb,
    WeatherPreset::Rain,
    WeatherPreset::Storm,
    WeatherPreset::ExtremeStorm,
];

/// A single weather exam entry: mission scenario + weather preset + label.
#[derive(Debug, Clone)]
pub struct WeatherExamEntry {
    /// Human-readable label: `weather_{preset}_{i}`.
    pub label: String,
    /// Fixed mission scenario (same spread geometry as mission exam).
    pub scenario: MissionScenario,
    /// Weather preset applied to this episode.
    pub preset: WeatherPreset,
    /// Reproducible seed derived from label hash so same label = same weather.
    pub seed: u64,
}

/// Build the 10 weather-varied exam scenarios.
///
/// Geometry mirrors the default 30-scenario exam (10 bearings × 3 distances ×
/// 3 altitudes) but takes the first entry from each of the 10 bearings (one
/// distance/altitude cycle) so all 10 episodes have distinct pickup points
/// while using the same seed-controlled spread logic.
pub fn weather_scenarios() -> Vec<WeatherExamEntry> {
    let mut entries = Vec::with_capacity(WEATHER_SCENARIO_COUNT);
    // Cycle through bearings (36° increments) and preset pairs.
    for (preset_idx, preset) in PRESETS.iter().enumerate() {
        for cycle in 0..2 {
            let scenario_index = preset_idx * 2 + cycle;
            let bearing = scenario_index as f64 * (2.0 * std::f64::consts::TAU / 10.0);
            let dist = if cycle == 0 { 1200.0 } else { 1800.0 };
            let alt = if cycle == 0 { 900.0 } else { 1100.0 };
            let label = format!("weather_{:?}_{}", preset, cycle);
            let seed = (
                label.bytes().map(|b| b as u64).sum::<u64>() % 10_000
            ) + (preset_idx as u64 * 1_000 + 7);
            entries.push(WeatherExamEntry {
                label,
                scenario: MissionScenario {
                    id: format!("weather_{:?}_{:02}", preset, cycle),
                    pickup_north: dist * bearing.cos(),
                    pickup_east: dist * bearing.sin(),
                    pickup_altitude: alt,
                    initial_phase: MissionPhase::Outbound,
                    initial_north: None,
                    initial_east: None,
                    initial_altitude: None,
                    initial_airspeed: None,
                },
                preset: *preset,
                seed,
            });
        }
    }
    // Shuffle deterministically by seed so display order = preset rotation.
    entries.sort_by(|a, b| {
        a.preset
            .name()
            .cmp(b.preset.name())
            .then_with(|| a.cycle().cmp(&b.cycle()))
    });
    // Re-label after sort to keep sequential indexing.
    for (i, e) in entries.iter_mut().enumerate() {
        e.label = format!("w{:02}_{}", i + 1, e.preset.name());
    }
    entries
}

/// Helper to recover the 0/1 cycle index from a sorted entry label.
impl WeatherExamEntry {
    pub fn preset_name(&self) -> &'static str {
        self.preset.name()
    }

    /// Cycle index within the preset pair (0 or 1).
    pub fn cycle(&self) -> usize {
        match self.preset {
            WeatherPreset::Clear => match self.scenario.id.as_str() {
                s if s.contains("_0") || s.contains("_00") => 0,
                _ => 1,
            },
            _ => {
                // Derive from distance/altitude split: 1200/900 = cycle 0.
                if self.scenario.pickup_altitude == 900.0 {
                    0
                } else {
                    1
                }
            }
        }
    }
}

/// Run a weather exam episode through the mission environment with the
/// configured weather preset. Returns the raw mission termination / signals.
pub fn run_weather_episode(
    entry: &WeatherExamEntry,
    mission_cfg: &MissionConfig,
) -> Result<(bool, bool, bool, String), Box<dyn std::error::Error>> {
    let weather_cfg = WeatherConfig::from_preset(entry.preset, entry.seed);
    let mut env = MissionEnvironment::new_with_weather(
        if std::path::Path::new("../m_sweep_mini.toml").exists() {
            "../m_sweep_mini.toml"
        } else {
            "m_sweep_mini.toml"
        },
        mission_cfg.clone(),
        entry.seed,
        Some(weather_cfg),
    )?;

    env.reset_with_scenario(&entry.scenario);
    // Short scripted trajectory (heuristic-level) to measure outcome without
    // requiring a full trained policy. Uses a neutral cruise action.
    let action = flight_core::AvionicsAction::level_cruise(0.5);

    let max_steps = mission_cfg.max_steps.min(600);
    let mut last_termination = String::new();
    for _ in 0..max_steps {
        let out = env.step(action);
        if out.terminated || out.truncated {
            last_termination = format!("{:?}", out.termination);
            break;
        }
    }

    // Determine outcomes from the final mission phase / termination.
    let terminated = env.termination();
    let success = env.phase() == MissionPhase::Done
        || terminated == flight_core::MissionTermination::CapturedAndReturned;
    let captured = terminated == flight_core::MissionTermination::CapturedAndReturned
        || env.phase() == MissionPhase::Return || env.phase() == MissionPhase::Done;
    let crash = matches!(
        terminated,
        flight_core::MissionTermination::GroundImpact
            | flight_core::MissionTermination::OutOfBounds
    );

    Ok((success, captured, crash, last_termination))
}

/// Aggregate statistics with Wilson 95 % interval for binary outcomes.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct WeatherExamSummary {
    /// Per-preset breakdown.
    pub preset_summary: BTreeMap<String, PresetStats>,
    /// Overall 10-episode aggregates.
    pub overall: AggregateStats,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct PresetStats {
    pub preset: String,
    pub n: usize,
    pub successes: usize,
    pub captures: usize,
    pub crashes: usize,
    pub success_rate: f64,
    pub capture_rate: f64,
    pub crash_rate: f64,
    pub success_ci_low: f64,
    pub success_ci_high: f64,
    pub capture_ci_low: f64,
    pub capture_ci_high: f64,
    pub crash_ci_low: f64,
    pub crash_ci_high: f64,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct AggregateStats {
    pub n: usize,
    pub successes: usize,
    pub captures: usize,
    pub crashes: usize,
    pub success_rate: f64,
    pub success_ci_low: f64,
    pub success_ci_high: f64,
    pub capture_rate: f64,
    pub crash_rate: f64,
}

/// Wilson score interval (z = 1.96 for 95 % confidence).
pub fn wilson_interval(successes: usize, n: usize, z: f64) -> (f64, f64) {
    if n == 0 {
        return (0.0, 0.0);
    }
    let p = successes as f64 / n as f64;
    let denom = 1.0 + z * z / n as f64;
    let centre = (p + z * z / (2.0 * n as f64)) / denom;
    let delta = z
        * (p * (1.0 - p) / n as f64 + z * z / (4.0 * n as f64 * n as f64)).sqrt()
        / denom;
    ((centre - delta).max(0.0), (centre + delta).min(1.0))
}

/// Build a [`WeatherExamSummary`] from a vector of per-episode results.
pub fn build_summary(
    results: &[(String, bool, bool, bool, String)],
) -> WeatherExamSummary {
    let mut preset_stats: BTreeMap<String, Vec<(bool, bool, bool)>> = BTreeMap::new();
    for (label, success, captured, crash, _termination) in results {
        // Extract preset key from label: `w01_Clear` etc.
        let preset_key = label
            .split('_')
            .last()
            .unwrap_or("unknown")
            .to_string();
        preset_stats
            .entry(preset_key.clone())
            .or_default()
            .push((*success, *captured, *crash));
    }

    let mut preset_summary = BTreeMap::new();
    let mut total_successes = 0usize;
    let mut total_captures = 0usize;
    let mut total_crashes = 0usize;
    let mut total_n = 0usize;

    for (preset_key, outcomes) in preset_stats {
        let n = outcomes.len();
        let successes = outcomes.iter().filter(|(s, _, _)| *s).count();
        let captures = outcomes.iter().filter(|(_, c, _)| *c).count();
        let crashes = outcomes.iter().filter(|(_, _, cr)| *cr).count();
        total_successes += successes;
        total_captures += captures;
        total_crashes += crashes;
        total_n += n;

        let (s_lo, s_hi) = wilson_interval(successes, n, 1.96);
        let (c_lo, c_hi) = wilson_interval(captures, n, 1.96);
        let (cr_lo, cr_hi) = wilson_interval(crashes, n, 1.96);

        preset_summary.insert(
            preset_key.clone(),
            PresetStats {
                preset: preset_key,
                n,
                successes,
                captures,
                crashes,
                success_rate: if n > 0 { successes as f64 / n as f64 } else { 0.0 },
                capture_rate: if n > 0 { captures as f64 / n as f64 } else { 0.0 },
                crash_rate: if n > 0 { crashes as f64 / n as f64 } else { 0.0 },
                success_ci_low: s_lo,
                success_ci_high: s_hi,
                capture_ci_low: c_lo,
                capture_ci_high: c_hi,
                crash_ci_low: cr_lo,
                crash_ci_high: cr_hi,
            },
        );
    }

    let (s_lo, s_hi) = wilson_interval(total_successes, total_n, 1.96);

    WeatherExamSummary {
        preset_summary,
        overall: AggregateStats {
            n: total_n,
            successes: total_successes,
            captures: total_captures,
            crashes: total_crashes,
            success_rate: if total_n > 0 { total_successes as f64 / total_n as f64 } else { 0.0 },
            success_ci_low: s_lo,
            success_ci_high: s_hi,
            capture_rate: if total_n > 0 { total_captures as f64 / total_n as f64 } else { 0.0 },
            crash_rate: if total_n > 0 { total_crashes as f64 / total_n as f64 } else { 0.0 },
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn weather_scenarios_has_ten_entries_and_spans_all_presets() {
        let scenarios = weather_scenarios();
        assert_eq!(scenarios.len(), WEATHER_SCENARIO_COUNT);
        let presets: std::collections::BTreeSet<_> =
            scenarios.iter().map(|e| e.preset.name()).collect();
        assert!(presets.contains("clear"));
        assert!(presets.contains("light_turb"));
        assert!(presets.contains("rain"));
        assert!(presets.contains("storm"));
        assert!(presets.contains("extreme_storm"));
        assert_eq!(presets.len(), 5);
    }

    #[test]
    fn weather_scenarios_each_preset_appears_twice() {
        let scenarios = weather_scenarios();
        for preset in PRESETS.iter() {
            let count = scenarios.iter().filter(|e| e.preset == *preset).count();
            assert_eq!(count, 2, "preset {:?} should appear twice", preset);
        }
    }

    #[test]
    fn wilson_interval_computes_within_bounds() {
        let (lo, hi) = wilson_interval(5, 10, 1.96);
        assert!(lo >= 0.0 && hi <= 1.0);
        assert!(lo <= hi);
    }

    #[test]
    fn build_summary_aggregates_correctly() {
        let results = vec![
            ("w01_Clear".into(), true, true, false, "CapturedAndReturned".into()),
            ("w02_Clear".into(), false, false, true, "GroundImpact".into()),
        ];
        let summary = build_summary(&results);
        assert_eq!(summary.overall.n, 2);
        assert_eq!(summary.overall.successes, 1);
        assert_eq!(summary.preset_summary.len(), 1);
    }

    #[test]
    fn weather_config_preset_roundtrip() {
        let cfg = WeatherConfig::from_preset(WeatherPreset::Clear, 42);
        assert_eq!(cfg.seed, 42);
        assert!(cfg.deterministic == false); // default recipe
    }
}
