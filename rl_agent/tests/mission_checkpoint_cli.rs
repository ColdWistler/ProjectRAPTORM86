//! Checkpoint metadata contract for mission and avionics training.
//!
//! Mission runs must persist the mission contract, reward profile, and expert
//! card beside the weights so downstream tooling can reconstruct them without
//! guessing. Avionics checkpoints must stay byte-compatible with 0.2.0, so the
//! same run must not grow any of the new files.

use std::process::Command;

fn bin() -> &'static str {
    env!("CARGO_BIN_EXE_rl_agent")
}

/// Config path resolved relative to the crate dir: cargo runs integration
/// tests with `rl_agent/` as the working directory, while the CLI's mission
/// default (`m_sweep_mini.toml`) assumes the repository root.
fn mission_config() -> String {
    if std::path::Path::new("m_sweep_mini.toml").exists() {
        "m_sweep_mini.toml".into()
    } else {
        "../m_sweep_mini.toml".into()
    }
}

/// Avionics counterpart: `aircraft.toml` lives at the repository root, while
/// cargo runs this test from `rl_agent/`.
fn avionics_config() -> String {
    if std::path::Path::new("aircraft.toml").exists() {
        "aircraft.toml".into()
    } else {
        "../aircraft.toml".into()
    }
}

fn temp_dir(label: &str) -> std::path::PathBuf {
    let dir = std::env::temp_dir().join(format!("raptor-{label}-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    dir
}

fn train_mission(dir: &std::path::Path, extra: &[&str]) {
    let mut cmd = Command::new(bin());
    cmd.args([
        "--mode", "train", "--mission", "mission", "--algo", "ppo",
        "--backend", "cpu", "--iterations", "1", "--rollout", "64",
        "--env-max-steps", "64", "--seed", "42", "--telemetry", "off",
        "--save-every", "1", "--config",
    ])
    .arg(mission_config())
    .args(extra)
    .arg("--checkpoint")
    .arg(dir);
    let out = cmd.output().expect("run rl_agent");
    assert!(
        out.status.success(),
        "stdout:\n{}\nstderr:\n{}",
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr)
    );
}

fn read_json(dir: &std::path::Path, name: &str) -> serde_json::Value {
    serde_json::from_str(&std::fs::read_to_string(dir.join(name)).unwrap()).unwrap()
}

#[test]
fn mission_training_writes_expert_mission_and_reward_json() {
    let dir = temp_dir("meta");
    train_mission(&dir, &[]);

    for name in ["expert.json", "mission.json", "reward.json"] {
        assert!(dir.join(name).exists(), "missing {name} in {}", dir.display());
    }

    let expert = read_json(&dir, "expert.json");
    assert_eq!(expert["contract"], "mission.v1");
    assert_eq!(expert["model"], "ppo");
    assert_eq!(expert["reward_profile"], "balanced");
    assert_eq!(expert["specialization"], "full");
    assert_eq!(expert["obs_mask"].as_array().unwrap().len(), 32);

    let mission = read_json(&dir, "mission.json");
    assert_eq!(mission["contract"], "mission.v1");
    assert_eq!(mission["max_steps"], 64);
    assert_eq!(mission["reward_profile"], "balanced");

    let reward = read_json(&dir, "reward.json");
    assert_eq!(reward["name"], "balanced");

    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn mission_specialization_is_persisted_and_card_round_trips() {
    let dir = temp_dir("capture");
    train_mission(&dir, &["--specialization", "capture"]);

    let expert = read_json(&dir, "expert.json");
    assert_eq!(expert["specialization"], "capture");

    let card = rl_agent::policy::ExpertCard::from_checkpoint(dir.to_str().unwrap(), "ppo").unwrap();
    assert_eq!(card.specialization, rl_agent::policy::Specialization::Capture);
    assert_eq!(card.contract, "mission.v1");
    assert_eq!(card.reward_profile, "balanced");
    assert_eq!(card.obs_mask.len(), 32);
    assert_eq!(card.model, rl_agent::policy::ModelKind::Ppo);

    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn avionics_training_writes_no_mission_metadata() {
    let dir = temp_dir("avionics");
    let out = Command::new(bin())
        .args([
            "--mode", "train", "--backend", "cpu", "--iterations", "1",
            "--rollout", "64", "--env-max-steps", "64", "--seed", "42",
            "--telemetry", "off", "--save-every", "1", "--config",
        ])
        .arg(avionics_config())
        .arg("--checkpoint")
        .arg(&dir)
        .output()
        .expect("run rl_agent");
    assert!(
        out.status.success(),
        "stdout:\n{}\nstderr:\n{}",
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr)
    );

    for name in ["expert.json", "mission.json", "reward.json"] {
        assert!(
            !dir.join(name).exists(),
            "avionics checkpoint unexpectedly wrote {name}"
        );
    }
    assert!(dir.join("net.json").exists());

    let _ = std::fs::remove_dir_all(&dir);
}
