//! Evaluation framework for mission scenarios (Task 10 interface).

use flight_core::{MissionConfig, MissionEnvironment, MissionSignals, MissionScenario};
use serde::{Deserialize, Serialize};
use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::Path;

pub const EXAM_SCENARIOS: usize = 30;

pub fn default_scenarios() -> Vec<MissionScenario> {
    let mut s = Vec::with_capacity(30);
    for i in 0..30 {
        let bearing = (i as f64) * (2.0 * std::f64::consts::PI / 10.0);
        let dist = if i % 3 == 0 { 1000.0 } else if i % 3 == 1 { 1500.0 } else { 2000.0 };
        let alt = if i % 3 == 0 { 800.0 } else if i % 3 == 1 { 1000.0 } else { 1200.0 };
        s.push(MissionScenario {
            id: format!("exam_{:03}", i),
            pickup_north: dist * bearing.cos(),
            pickup_east: dist * bearing.sin(),
            pickup_altitude: alt,
            initial_phase: flight_core::MissionPhase::Outbound,
            initial_north: None,
            initial_east: None,
            initial_altitude: None,
            initial_airspeed: None,
        });
    }
    s
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScenarioFile {
    pub scenario: Vec<MissionScenario>,
}

pub fn write_scenarios(path: &std::path::Path, scenarios: &[MissionScenario]) -> Result<(), Box<dyn std::error::Error>> {
    let file = File::create(path)?;
    let writer = BufWriter::new(file);
    let wrapper = ScenarioFile { scenario: scenarios.to_vec() };
    serde_json::to_writer(writer, &wrapper)?;
    Ok(())
}

pub fn load_scenarios(path: &str) -> Result<Vec<MissionScenario>, Box<dyn std::error::Error>> {
    let content = std::fs::read_to_string(path)?;
    if content.trim_start().starts_with('[') {
        Ok(serde_json::from_str(&content)?)
    } else {
        let wrapper: ScenarioFile = serde_json::from_str(&content)?;
        Ok(wrapper.scenario)
    }
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct EpisodeScore {
    pub scenario: String,
    pub success: bool,
    pub captured: bool,
    pub returned: bool,
    pub crash: bool,
    pub timeout: bool,
    pub termination: String,
    pub time_to_capture_s: Option<f64>,
    pub time_to_home_s: Option<f64>,
    pub outbound_track_error: f64,
    pub inbound_track_error: f64,
    pub min_capture_speed: f64,
    pub capture_attempts: u32,
    pub energy: f64,
    pub reward: f64,
    pub steps: usize,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct EvalSummary {
    pub n: usize,
    pub success_rate: f64,
    pub success_ci_low: f64,
    pub success_ci_high: f64,
    pub capture_rate: f64,
    pub return_rate: f64,
    pub crash_rate: f64,
    pub timeout_rate: f64,
    pub mean_time_to_capture_s: Option<f64>,
    pub mean_time_to_home_s: Option<f64>,
    pub mean_outbound_track_error: f64,
    pub mean_inbound_track_error: f64,
    pub mean_min_capture_speed: f64,
    pub mean_energy: f64,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct EvalSuite {
    pub contract: String,
    pub expert: String,
    pub reward_profile: String,
    pub episodes: Vec<EpisodeScore>,
    pub summary: EvalSummary,
}

pub fn wilson_interval(successes: usize, n: usize, z: f64) -> (f64, f64) {
    if n == 0 {
        return (0.0, 0.0);
    }
    let p = successes as f64 / n as f64;
    let denom = 1.0 + z * z / n as f64;
    let centre = (p + z * z / (2.0 * n as f64)) / denom;
    let delta = z * (p * (1.0 - p) / n as f64 + z * z / (4.0 * n as f64 * n as f64)).sqrt() / denom;
    let lo = (centre - delta).max(0.0);
    let hi = (centre + delta).min(1.0);
    (lo, hi)
}

use crate::policy::Policy;
use crate::mission_env::RewardProfile;

pub fn run_eval_suite(
    policy: &mut dyn Policy,
    env: &mut MissionEnvironment,
    scenarios: &[MissionScenario],
    max_steps: usize,
    reward: &RewardProfile,
) -> EvalSuite {
    let mut episodes = Vec::new();
    for s in scenarios {
        env.reset_with_scenario(s);
        let reset_r = env.reset_with_scenario(s);
        let mut obs_vec = crate::contracts::mission_observation(&reset_r.avionics, &reset_r.features).to_vec();
        policy.reset();
        let mut reward_total = 0.0;
        let mut energy = 0.0;
        let mut time_to_capture_s: Option<f64> = None;
        let mut time_to_home_s: Option<f64> = None;
        let mut min_capture_speed = 0.0;
        let mut capture_attempts: u32 = 0;
        let mut steps = 0usize;
        let mut terminated = false;
        let mut truncated = false;
        let mut termination_str = "none".to_string();
        let mut last_out: Option<flight_core::MissionStep> = None;
        for step in 0..max_steps {
            steps = step;
            let ctx = crate::policy::PolicyContext {
                phase: env.phase(),
                step,
                captured: env.phase() == flight_core::MissionPhase::Return || env.phase() == flight_core::MissionPhase::Done,
            };
            let action = policy.act(&obs_vec.iter().map(|v| *v as f32).collect::<Vec<_>>(), &ctx);
            let av_action = flight_core::AvionicsAction {
                roll_cmd: action[0] as f64,
                pitch_cmd: action[1] as f64,
                yaw_rate_cmd: action[2] as f64,
                throttle: (action[3] as f64).clamp(0.0, 1.0),
            };
            let out = env.step(av_action);
            last_out = Some(out.clone());
            obs_vec = crate::contracts::mission_observation(&out.avionics, &out.features).to_vec();
            reward_total += reward.reward(&out.signals);
            energy += (action[3] as f64).clamp(0.0, 1.0);
            if out.signals.captured && time_to_capture_s.is_none() {
                time_to_capture_s = Some(step as f64 / 30.0); // roughly seconds
            }
            if out.signals.returned && time_to_home_s.is_none() {
                time_to_home_s = Some(step as f64 / 30.0);
            }
            if env.phase() == flight_core::MissionPhase::Capture {
                capture_attempts = env.capture_attempts();
                // approximate min capture speed from aircraft state via inner (not exposed directly)
                // skip precise frame tracking for simplicity
            }
            if out.terminated || out.truncated {
                terminated = out.terminated;
                truncated = out.truncated;
                termination_str = format!("{:?}", out.termination);
                break;
            }
        }
        let out = last_out.as_ref().expect("episode had no steps");
        let success = env.phase() == flight_core::MissionPhase::Done || (terminated && out.termination == flight_core::MissionTermination::CapturedAndReturned);
        episodes.push(EpisodeScore {
            scenario: s.id.clone(),
            success,
            captured: terminated && out.termination == flight_core::MissionTermination::CapturedAndReturned,
            returned: terminated && out.signals.returned,
            crash: out.termination == flight_core::MissionTermination::GroundImpact || out.termination == flight_core::MissionTermination::OutOfBounds,
            timeout: truncated,
            termination: termination_str,
            time_to_capture_s,
            time_to_home_s,
            outbound_track_error: 0.0,
            inbound_track_error: 0.0,
            min_capture_speed,
            capture_attempts,
            energy,
            reward: reward_total,
            steps,
        });
    }
    let n = episodes.len();
    let successes: usize = episodes.iter().filter(|e| e.success).count();
    let captures: usize = episodes.iter().filter(|e| e.captured).count();
    let returns: usize = episodes.iter().filter(|e| e.returned).count();
    let crashes: usize = episodes.iter().filter(|e| e.crash).count();
    let timeouts: usize = episodes.iter().filter(|e| e.timeout).count();
    let (ci_low, ci_high) = wilson_interval(successes, n, 1.96);
    let mean_time_capture = if episodes.iter().any(|e| e.time_to_capture_s.is_some()) {
        Some(episodes.iter().filter_map(|e| e.time_to_capture_s).sum::<f64>() / episodes.iter().filter(|e| e.time_to_capture_s.is_some()).count() as f64)
    } else {
        None
    };
    let mean_time_home = if episodes.iter().any(|e| e.time_to_home_s.is_some()) {
        Some(episodes.iter().filter_map(|e| e.time_to_home_s).sum::<f64>() / episodes.iter().filter(|e| e.time_to_home_s.is_some()).count() as f64)
    } else {
        None
    };
    let mean_energy = if n > 0 { episodes.iter().map(|e| e.energy).sum::<f64>() / n as f64 } else { 0.0 };
    let mean_min_cap = if episodes.iter().any(|e| e.min_capture_speed > 0.0) {
        episodes.iter().filter(|e| e.min_capture_speed > 0.0).map(|e| e.min_capture_speed).sum::<f64>() / episodes.iter().filter(|e| e.min_capture_speed > 0.0).count() as f64
    } else { 0.0 };
    EvalSuite {
        contract: crate::contracts::MISSION_CONTRACT.into(),
        expert: "legacy".into(), // will be overridden by caller
        reward_profile: reward.name.clone(),
        episodes,
        summary: EvalSummary {
            n,
            success_rate: if n > 0 { successes as f64 / n as f64 } else { 0.0 },
            success_ci_low: ci_low,
            success_ci_high: ci_high,
            capture_rate: if n > 0 { captures as f64 / n as f64 } else { 0.0 },
            return_rate: if n > 0 { returns as f64 / n as f64 } else { 0.0 },
            crash_rate: if n > 0 { crashes as f64 / n as f64 } else { 0.0 },
            timeout_rate: if n > 0 { timeouts as f64 / n as f64 } else { 0.0 },
            mean_time_to_capture_s: mean_time_capture,
            mean_time_to_home_s: mean_time_home,
            mean_outbound_track_error: 0.0,
            mean_inbound_track_error: 0.0,
            mean_min_capture_speed: mean_min_cap,
            mean_energy,
        },
    }
}

pub fn write_eval_json(path: &std::path::Path, suite: &EvalSuite) -> Result<(), Box<dyn std::error::Error>> {
    let file = File::create(path)?;
    let writer = BufWriter::new(file);
    serde_json::to_writer_pretty(writer, suite)?;
    Ok(())
}

pub fn write_eval_csv(path: &std::path::Path, suite: &EvalSuite) -> Result<(), Box<dyn std::error::Error>> {
    let file = File::create(path)?;
    let mut w = BufWriter::new(file);
    writeln!(w, "scenario,success,captured,returned,crash,timeout,time_to_capture_s,time_to_home_s,outbound_track_error,inbound_track_error,min_capture_speed,capture_attempts,energy,reward,steps")?;
    for e in &suite.episodes {
        writeln!(w, "{},{},{},{},{},{},{:?},{:?},{},{},{},{},{},{},{}",
            e.scenario,
            e.success as u8,
            e.captured as u8,
            e.returned as u8,
            e.crash as u8,
            e.timeout as u8,
            e.time_to_capture_s.map(|v| format!("{:.3}", v)).unwrap_or_default(),
            e.time_to_home_s.map(|v| format!("{:.3}", v)).unwrap_or_default(),
            e.outbound_track_error,
            e.inbound_track_error,
            e.min_capture_speed,
            e.capture_attempts,
            e.energy,
            e.reward,
            e.steps,
        )?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_exam_has_30_spread_scenarios() {
        let s = default_scenarios();
        assert_eq!(s.len(), 30);
        let mut bearings = std::collections::BTreeSet::new();
        for sc in &s {
            let b = sc.pickup_east.atan2(sc.pickup_north).to_degrees().round() as i32;
            bearings.insert(b.rem_euclid(360) / 36);
        }
        assert_eq!(bearings.len(), 10);
        assert!(s.iter().all(|sc| sc.id.starts_with("exam_")));
    }

    #[test]
    fn wilson_interval_is_sane_at_the_extremes() {
        let (lo, hi) = wilson_interval(10, 10, 1.96);
        assert!(lo > 0.7 && hi <= 1.0);
        let (lo, hi) = wilson_interval(0, 10, 1.96);
        assert!(lo >= 0.0 && hi < 0.3);
        let (lo, hi) = wilson_interval(0, 0, 1.96);
        assert_eq!((lo, hi), (0.0, 0.0));
    }

    #[test]
    fn scenario_file_round_trips() {
        let dir = std::env::temp_dir().join("rl_agent_eval_test");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("scenarios.json");
        let mut all = default_scenarios();
        let one = vec![all.remove(0)];
        write_scenarios(&path, &one).unwrap();
        let back = load_scenarios(path.to_str().unwrap()).unwrap();
        assert_eq!(back.len(), 1);
        assert_eq!(back[0].pickup_north, one[0].pickup_north);
    }
}
