//! Restored mission-heuristic regression gate.
//!
//! `task-7-report.md` and `task-8-report.md` referenced this file, but it was
//! never committed. The bounds check protects the avionics action contract;
//! the completion gate tracks whether the heuristic can actually finish a
//! mission (gated on T06 until the capture-approach profile lands).

use flight_core::{MissionConfig, MissionPhase};
use rl_agent::mission_env::{MissionEnv, RewardProfile};
use rl_agent::policy::{HeuristicPolicy, Policy, PolicyContext};
use rl_agent::trainer::EnvSpec;

fn mission_config() -> String {
    if std::path::Path::new("m_sweep_mini.toml").exists() {
        "m_sweep_mini.toml".into()
    } else {
        "../m_sweep_mini.toml".into()
    }
}

fn env() -> MissionEnv {
    MissionEnv::new(
        &mission_config(),
        MissionConfig::default(),
        RewardProfile::balanced(),
        7,
    )
    .expect("mission env")
}

fn policy() -> HeuristicPolicy {
    let cfg = MissionConfig::default();
    let env = env();
    let v_stall = flight_core::stall_speed(env.aircraft_config(), cfg.cruise_altitude);
    HeuristicPolicy::new(
        cfg.cruise_airspeed,
        cfg.resolve_capture_speed(env.aircraft_config()),
        v_stall,
    )
}

#[test]
fn heuristic_actions_stay_inside_the_contract_bounds() {
    let mut p = policy();
    let mut env = env();
    let obs: Vec<f32> = env.reset().iter().map(|&x| x as f32).collect();
    for phase in [
        MissionPhase::Outbound,
        MissionPhase::Capture,
        MissionPhase::Return,
    ] {
        let ctx = PolicyContext {
            phase,
            step: 0,
            captured: false,
            features: None,
        };
        let a = p.act(&obs, &ctx);
        assert_eq!(a.len(), 4);
        // contracts: roll ±0.6, pitch ±0.4, yaw ±0.8, throttle 0..1
        assert!(a[0].abs() <= 0.6, "roll {}", a[0]);
        assert!(a[1].abs() <= 0.4, "pitch {}", a[1]);
        assert!(a[2].abs() <= 0.8, "yaw {}", a[2]);
        assert!((0.0..=1.0).contains(&a[3]), "throttle {}", a[3]);
    }
}

#[test]
fn heuristic_starts_bleeding_speed_before_the_pickup() {
    use rl_agent::contracts::{MISSION_OBS_DIM, OBS_AIRSPEED_INDICATED, OBS_PICKUP_DISTANCE};
    let mut p = policy();
    let ctx = PolicyContext {
        phase: MissionPhase::Outbound,
        step: 0,
        captured: false,
        features: None,
    };

    // Compare at the same airspeed: the difference must come from the distance
    // to the pickup, not from being faster. On approach the heuristic targets
    // the lower capture speed, so it commands less throttle and a more nose-up
    // attitude to bleed speed; far out it holds the cruise target.
    let airspeed = 18.0;
    let mut near = vec![0.0f32; MISSION_OBS_DIM];
    near[OBS_AIRSPEED_INDICATED] = airspeed;
    near[OBS_PICKUP_DISTANCE] = 0.12; // ~300 m out of 2500 m
    let a = p.act(&near, &ctx);

    let mut far = vec![0.0f32; MISSION_OBS_DIM];
    far[OBS_AIRSPEED_INDICATED] = airspeed;
    far[OBS_PICKUP_DISTANCE] = 0.9; // ~2250 m out
    let b = p.act(&far, &ctx);

    assert!(
        a[3] + 1e-6 < b[3],
        "approach throttle {} should be below cruise throttle {}",
        a[3],
        b[3]
    );
    assert!(
        a[1] < b[1],
        "approach pitch {} should be more nose-up than cruise pitch {}",
        a[1],
        b[1]
    );
}

#[test]
fn heuristic_completes_at_least_one_of_the_first_three_exam_scenarios() {
    let mut p = policy();
    let mut env = env();
    let scenarios = rl_agent::eval::default_scenarios();
    let mut successes = 0;
    for s in scenarios.iter().take(3) {
        let mut obs = env.reset_with_scenario_pub(s);
        let mut done = false;
        for _ in 0..env.mission_config().max_steps {
            let ctx = PolicyContext {
                phase: env.phase_public(),
                step: 0,
                captured: false,
                features: None,
            };
            let a = p.act(&obs, &ctx);
            let out = env.step_physical(&a);
            obs = out.obs.iter().map(|&x| x as f32).collect();
            if out.terminated || out.truncated {
                done = true;
                break;
            }
        }
        if done && env.terminated_success() {
            successes += 1;
        }
    }
    assert!(successes >= 1, "heuristic completed {successes}/3; see Task 6");
}
