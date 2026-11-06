//! End-to-end check that `--mode train --mission mission` trains on the
//! 32-channel mission adapter instead of the 19-channel avionics environment.
//!
//! The bug this guards: the CLI built a 32-channel network (because
//! `--mission mission` raised `ncfg.obs_dim`) but still handed the trainer an
//! `AvionicsEnv`, so the first normalizer update panicked with an
//! observation-width mismatch.

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

#[test]
fn mission_training_run_completes_and_writes_a_32_channel_checkpoint() {
    let dir = std::env::temp_dir().join(format!("raptor-mission-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    let out = Command::new(bin())
        .args([
            "--mode", "train", "--mission", "mission", "--algo", "ppo",
            "--backend", "cpu", "--iterations", "1", "--rollout", "64",
            "--env-max-steps", "64", "--seed", "42", "--telemetry", "off",
            "--config",
        ])
        .arg(mission_config())
        .arg("--checkpoint")
        .arg(&dir)
        .output()
        .expect("run rl_agent");
    let stdout = String::from_utf8_lossy(&out.stdout);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(out.status.success(), "stdout:\n{stdout}\nstderr:\n{stderr}");
    assert!(
        !stderr.contains("observation dimension mismatch"),
        "stderr:\n{stderr}"
    );
    assert!(stdout.contains("training complete"), "stdout:\n{stdout}");
    let net: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(dir.join("net.json")).unwrap()).unwrap();
    assert_eq!(net["obs_dim"], 32);
    let _ = std::fs::remove_dir_all(&dir);
}
