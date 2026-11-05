//! End-to-end checks that every registered algorithm can actually **train**.
//!
//! The unit tests inside each algorithm module prove the maths. These tests
//! prove the wiring: that the shared trainer loop feeds the algorithm, that
//! the update reaches the weights, and that nothing panics on the way.
//!
//! They are the criteria the milestone briefs dropped ("bounded actions,
//! replay samples") in favour of the spec's behavioural acceptance tests
//! (spec §14 / §18): an algorithm whose metrics never move is not learning.

use std::process::Command;

/// Aircraft config, resolved relative to the crate dir (cargo runs
/// integration tests with the package directory as the working directory).
fn config_path() -> String {
    if std::path::Path::new("aircraft.toml").exists() {
        "aircraft.toml".into()
    } else {
        "../aircraft.toml".into()
    }
}

struct Run {
    success: bool,
    stdout: String,
    stderr: String,
}

fn run_algo(algo: &str, extra: &[&str]) -> Run {
    let exe = env!("CARGO_BIN_EXE_rl_agent");
    let mut args: Vec<String> = vec![
        "--mode".into(),
        "train".into(),
        "--algo".into(),
        algo.into(),
        "--iterations".into(),
        "2".into(),
        "--rollout".into(),
        "64".into(),
        "--env-max-steps".into(),
        "64".into(),
        "--print-every".into(),
        "1".into(),
        "--telemetry".into(),
        "off".into(),
        "--stream".into(),
        "off".into(),
        "--control".into(),
        "off".into(),
        "--backend".into(),
        "cpu".into(),
        "--config".into(),
        config_path(),
    ];
    args.extend(extra.iter().map(|s| s.to_string()));
    let out = Command::new(exe)
        .args(&args)
        .output()
        .expect("rl_agent binary must be runnable");
    Run {
        success: out.status.success(),
        stdout: String::from_utf8_lossy(&out.stdout).into_owned(),
        stderr: String::from_utf8_lossy(&out.stderr).into_owned(),
    }
}

/// Pull a named metric out of the `[rl_agent] iter ...` line.
fn metric(stdout: &str, key: &str) -> Option<f64> {
    let idx = stdout.find(key)?;
    let rest = &stdout[idx + key.len()..];
    let token: String = rest
        .chars()
        .take_while(|c| c.is_ascii_digit() || matches!(c, '-' | '+' | '.' | 'e' | 'E'))
        .collect();
    token.parse::<f64>().ok()
}

#[test]
fn d3qn_training_run_completes_without_panicking() {
    let r = run_algo("d3qn", &[]);
    assert!(
        r.success,
        "d3qn training run failed:\nstdout:\n{}\nstderr:\n{}",
        r.stdout,
        r.stderr
    );
    assert!(
        !r.stdout.contains("Tensor Operation Error"),
        "d3qn hit a burn tensor shape error:\n{}",
        r.stdout
    );
    assert!(
        r.stdout.contains("training complete"),
        "d3qn did not finish:\n{}",
        r.stdout
    );
}

#[test]
fn sac_training_reports_a_nonzero_critic_loss() {
    let r = run_algo("sac", &[]);
    assert!(
        r.success,
        "sac training run failed:\nstdout:\n{}\nstderr:\n{}",
        r.stdout,
        r.stderr
    );
    let value = metric(&r.stdout, "value ").expect("sac must print a value loss");
    assert!(
        value.abs() > 1e-6,
        "sac critic loss is {value}; the critic never trained:\n{}",
        r.stdout
    );
    let entropy = metric(&r.stdout, "entropy ").expect("sac must print an entropy");
    assert!(
        entropy.is_finite() && entropy.abs() > 1e-6,
        "sac entropy is {entropy}; the policy entropy was never computed:\n{}",
        r.stdout
    );
}

#[test]
fn td3_training_reports_a_nonzero_critic_loss() {
    let r = run_algo("td3", &[]);
    assert!(
        r.success,
        "td3 training run failed:\nstdout:\n{}\nstderr:\n{}",
        r.stdout,
        r.stderr
    );
    let value = metric(&r.stdout, "value ").expect("td3 must print a value loss");
    assert!(
        value.abs() > 1e-6,
        "td3 critic loss is {value}; the critic never trained:\n{}",
        r.stdout
    );
}

#[test]
fn a2c_training_run_reports_finite_metrics() {
    let r = run_algo("a2c", &[]);
    assert!(
        r.success,
        "a2c training run failed:\nstdout:\n{}\nstderr:\n{}",
        r.stdout,
        r.stderr
    );
    let value = metric(&r.stdout, "value ").expect("a2c must print a value loss");
    assert!(
        value.is_finite(),
        "a2c value loss is not finite ({value}); the run diverged:\n{}",
        r.stdout
    );
    let entropy = metric(&r.stdout, "entropy ").expect("a2c must print an entropy");
    assert!(
        entropy.is_finite(),
        "a2c entropy is not finite ({entropy}); the policy diverged:\n{}",
        r.stdout
    );
}

/// Train with `--save-every 1` into a scratch directory, then evaluate the
/// checkpoint. A checkpoint that cannot be reloaded is a broken model: the
/// weights would be silently re-randomized on the next run.
fn scratch_dir(tag: &str) -> std::path::PathBuf {
    static NEXT: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
    let n = NEXT.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    let p = std::env::temp_dir().join(format!("rl_algo_cp_{tag}_{}_{n}", std::process::id()));
    let _ = std::fs::remove_dir_all(&p);
    p
}

#[test]
fn off_policy_checkpoints_round_trip_through_the_cli() {
    for algo in ["sac", "td3", "d3qn"] {
        let dir = scratch_dir(algo);
        let dir_s = dir.to_str().expect("utf-8 temp dir").to_string();

        let train = run_algo(algo, &["--save-every", "1", "--checkpoint", &dir_s]);
        assert!(
            train.success,
            "{algo} training run failed:\nstdout:\n{}\nstderr:\n{}",
            train.stdout,
            train.stderr
        );
        assert!(
            dir.join("algo.json").exists(),
            "{algo} did not write algo.json"
        );

        let exe = env!("CARGO_BIN_EXE_rl_agent");
        let eval_run = Command::new(exe)
            .args([
                "--mode",
                "eval",
                "--checkpoint",
                &dir_s,
                "--backend",
                "cpu",
                "--config",
                &config_path(),
                // Keep the eval short: this test checks that the checkpoint
                // *loads*, not that the policy is good.
                "--eval-episodes",
                "1",
                "--env-max-steps",
                "64",
            ])
            .output()
            .expect("rl_agent binary must be runnable");
        assert!(
            eval_run.status.success(),
            "{algo} checkpoint could not be evaluated:\nstdout:\n{}\nstderr:\n{}",
            String::from_utf8_lossy(&eval_run.stdout),
            String::from_utf8_lossy(&eval_run.stderr)
        );

        let _ = std::fs::remove_dir_all(&dir);
    }
}

#[test]
fn ppo_first_epoch_policy_loss_is_zero() {    // With one epoch and an unchanged policy, the importance ratio is exactly
    // 1.0 and the (whitened) surrogate averages to zero. A non-zero policy
    // loss means the rollout and the update normalized the observations with
    // different statistics, so the ratio at epoch 1 was not 1.
    let r = run_algo("ppo", &["--epochs", "1"]);
    assert!(
        r.success,
        "ppo training run failed:\nstdout:\n{}\nstderr:\n{}",
        r.stdout,
        r.stderr
    );
    let policy = metric(&r.stdout, "policy ").expect("ppo must print a policy loss");
    assert!(
        policy.abs() < 1e-3,
        "ppo first-epoch policy loss is {policy}; the importance ratio was not 1.0:\n{}",
        r.stdout
    );
}
