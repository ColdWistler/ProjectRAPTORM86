//! CLI-level end-to-end checks for the `eval-suite` mode (Task 10).
//! These are integration tests built from the public `run_eval_suite` +
//! `write_eval_json` / `write_eval_csv` surface, plus a heuristic policy run.

use flight_core::{MissionConfig, MissionEnvironment};
use rl_agent::contracts::MISSION_CONTRACT;
use rl_agent::eval::{default_scenarios, run_eval_suite, write_eval_json, write_eval_csv};
use rl_agent::mission_env::RewardProfile;
use rl_agent::policy::{HeuristicPolicy, PolicyContext};
use flight_core::MissionPhase;

#[test]
fn eval_suite_writes_json_and_csv_for_the_heuristic() {
    let cfg = MissionConfig { max_steps: 400, ..Default::default() };
    let mut env = MissionEnvironment::new(
        if std::path::Path::new("m_sweep_mini.toml").exists() { "m_sweep_mini.toml" } else { "../m_sweep_mini.toml" },
        cfg.clone(),
        7,
    )
    .unwrap();
    let v_stall = flight_core::stall_speed(env.aircraft_config(), cfg.cruise_altitude);
    let mut policy = HeuristicPolicy::new(cfg.cruise_airspeed, cfg.resolve_capture_speed(env.aircraft_config()), v_stall);
    let suite = run_eval_suite(&mut policy, &mut env, &default_scenarios()[..1], cfg.max_steps, &RewardProfile::balanced());
    let dir = std::env::temp_dir().join(format!("rl_agent_eval_cli_{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    write_eval_json(&dir.join("eval.json"), &suite).unwrap();
    write_eval_csv(&dir.join("eval.csv"), &suite).unwrap();
    assert!(dir.join("eval.json").metadata().unwrap().len() > 0);
    let csv = std::fs::read_to_string(dir.join("eval.csv")).unwrap();
    assert!(csv.starts_with("scenario,success,captured"));
    // sanity: suite carries the mission contract through the run
    assert_eq!(suite.contract, MISSION_CONTRACT);
    assert_eq!(suite.reward_profile, "balanced");
    let _ = std::fs::remove_dir_all(&dir);
}
