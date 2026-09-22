//! Training loop and environment plumbing.
//!
//! [`run`] is the generic (backend-agnostic) training loop over an
//! [`EnvSpec`] (by default the [`AvionicsEnv`] wrapper around
//! [`AvionicsEnvironment`]):
//!
//! 1. **Rollout**: collect `rollout_steps` transitions, observing the
//!    normalizer during resets, sampling stochastic actions host-side.
//! 2. **Values**: predict V(s) for the whole normalized batch and for the
//!    trailing observation, then compute GAE advantages host-side.
//! 3. **Update**: one [`Algorithm::update`] pass over shuffled minibatches.
//! 4. Periodically evaluate and save a checkpoint
//!    (weights + JSON configs; optimizer state is not persisted in v1).

use crate::algo::{make_algorithm, Algorithm, AlgoSpec, UpdateBatch};
use crate::buffer::{RolloutBuffer, Transition};
use crate::gae::compute_gae;
use crate::net::NetConfig;
use crate::normalize::RunningNormalizer;
use crate::ppo::PpoConfig;
use crate::{ACTION_DIM, AVIONICS_OBS_DIM};
use burn::tensor::backend::AutodiffBackend;
use flight_core::{AvionicsAction, AvionicsEnvironment, EnvConfig};
use serde::{Deserialize, Serialize};
use std::path::Path;

/// Outcome of a single environment step (generic over the concrete env).
#[derive(Debug, Clone)]
pub struct StepOutcome {
    /// Next observation.
    pub obs: Vec<f64>,
    /// Scalar reward (before any crash penalty).
    pub reward: f64,
    /// Episode ended by termination (crash).
    pub terminated: bool,
    /// Episode ended by exhausting the step budget.
    pub truncated: bool,
}

/// Minimal environment interface the trainer needs.
pub trait EnvSpec {
    /// Observation dimension.
    fn obs_dim(&self) -> usize;
    /// Action dimension.
    fn action_dim(&self) -> usize;
    /// Reset to a fresh episode; returns the initial observation.
    fn reset(&mut self) -> Vec<f64>;
    /// Apply a continuous action (tanh-space), advancing the sim one `dt`.
    fn step(&mut self, action: &[f32]) -> StepOutcome;
    /// Penalty added when a step ends in termination (≤ 0).
    fn crash_penalty(&self) -> f64;
    /// Target cruise altitude (m) — used by the evaluator.
    fn target_altitude(&self) -> f64;
    /// Target cruise airspeed (m/s) — used by the evaluator.
    fn target_airspeed(&self) -> f64;
}

/// Map a tanh-squashed neural action to an avionics attitude command.
/// Channel scales keep every component inside the FC's ±45° setpoint clamps.
fn action_to_avionics(action: &[f32]) -> AvionicsAction {
    AvionicsAction {
        roll_cmd: f64::from(action[0].tanh()) * 0.6,
        pitch_cmd: f64::from(action[1].tanh()) * 0.4,
        yaw_rate_cmd: f64::from(action[2].tanh()) * 0.8,
        throttle: 0.5 * (f64::from(action[3].tanh()) + 1.0),
    }
}

/// The default training target: `flight_core::AvionicsEnvironment`.
pub struct AvionicsEnv {
    inner: AvionicsEnvironment,
}

impl AvionicsEnv {
    /// Load the aircraft config and build an avionics environment with the
    /// given physics step and episode budget.
    pub fn new(config_path: &str, dt: f64, max_steps: usize) -> Result<Self, Box<dyn std::error::Error>> {
        let env = AvionicsEnvironment::with_config(
            config_path,
            EnvConfig {
                dt,
                max_steps,
                ..Default::default()
            },
        )?;
        Ok(Self { inner: env })
    }
}

impl EnvSpec for AvionicsEnv {
    fn obs_dim(&self) -> usize {
        AVIONICS_OBS_DIM
    }

    fn action_dim(&self) -> usize {
        ACTION_DIM
    }

    fn reset(&mut self) -> Vec<f64> {
        self.inner.reset().0.to_vec()
    }

    fn step(&mut self, action: &[f32]) -> StepOutcome {
        let out = self.inner.step(action_to_avionics(action));
        StepOutcome {
            obs: out.observation.to_vec(),
            reward: out.reward,
            terminated: out.terminated,
            truncated: out.truncated,
        }
    }

    fn crash_penalty(&self) -> f64 {
        self.inner.crash_penalty()
    }

    fn target_altitude(&self) -> f64 {
        self.inner.config.target_altitude
    }

    fn target_airspeed(&self) -> f64 {
        self.inner.config.target_airspeed
    }
}

/// Trainer-level hyper-parameters.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainerConfig {
    /// Number of PPO iterations (rollout + update).
    pub iterations: usize,
    /// Steps collected per iteration (episodes continue across the boundary).
    pub rollout_steps: usize,
    /// Per-episode step budget.
    pub env_max_steps: usize,
    /// Physics time step per env step (seconds).
    pub env_dt: f64,
    /// Seed for the agent RNG + deterministic net init.
    pub seed: u64,
    /// Save a checkpoint every N iterations (0 = never).
    pub save_every: usize,
    /// Print per-iteration stats every N iterations (0 = silent).
    pub print_every: usize,
    /// Run this many deterministic eval episodes every `eval_every`.
    pub eval_episodes: usize,
    /// Evaluate every N iterations (0 = never).
    pub eval_every: usize,
    /// Directory for checkpoints.
    pub checkpoint_dir: String,
}

impl Default for TrainerConfig {
    fn default() -> Self {
        Self {
            iterations: 1000,
            rollout_steps: 2048,
            env_max_steps: 2000,
            env_dt: 1.0 / 30.0,
            seed: 42,
            save_every: 100,
            print_every: 1,
            eval_episodes: 3,
            eval_every: 100,
            checkpoint_dir: "checkpoints".to_string(),
        }
    }
}

/// Aggregated evaluation metrics.
#[derive(Debug, Clone, Copy, Default)]
pub struct EvalMetrics {
    /// Mean per-step |baro_altitude − target| over all eval steps.
    pub alt_err: f32,
    /// Mean per-step |airspeed_indicated − target| over all eval steps.
    pub spd_err: f32,
    /// Mean per-episode cumulative reward (crash penalties included).
    pub total_reward: f32,
    /// Number of episodes that ended by crashing.
    pub crashes: usize,
    /// Number of episodes evaluated.
    pub episodes: usize,
    /// Mean steps per episode.
    pub mean_steps: f32,
}

/// What `run` hands back for tests / the CLI.
pub struct TrainingOutcome<B: AutodiffBackend<FloatElem = f32>> {
    /// Trained algorithm (final params after all iterations).
    pub algorithm: Box<dyn Algorithm<B>>,
    /// Normalizer with the final running stats.
    pub normalizer: RunningNormalizer,
    /// Last completed eval, if any eval episode ran.
    pub last_eval: Option<EvalMetrics>,
}

/// Run `tcfg.iterations` training iterations on the avionics environment.
///
/// The algorithm is built from `spec` through the [`make_algorithm`]
/// registry, so the loop itself stays algorithm-agnostic.
pub fn run<B: AutodiffBackend<FloatElem = f32>>(
    tcfg: &TrainerConfig,
    ncfg: &NetConfig,
    spec: &AlgoSpec,
    device: &B::Device,
    config_path: &str,
) -> Result<TrainingOutcome<B>, Box<dyn std::error::Error>> {
    assert!(tcfg.rollout_steps > 0, "rollout_steps must be > 0");

    let mut env = AvionicsEnv::new(config_path, tcfg.env_dt, tcfg.env_max_steps)?;
    let mut agent: Box<dyn Algorithm<B>> = make_algorithm(spec, ncfg, tcfg.seed, device);
    let mut normalizer = RunningNormalizer::new(ncfg.obs_dim);
    let (gamma, lambda) = agent.td_params();

    let mut obs = env.reset();
    normalizer.observe(&obs);

    let mut last_eval: Option<EvalMetrics> = None;

    for it in 0..tcfg.iterations {
        let mut buffer = RolloutBuffer::new(ncfg.obs_dim, ncfg.action_dim);
        let mut final_obs: Option<Vec<f32>> = None;

        // ---- 1. Rollout ----
        for _ in 0..tcfg.rollout_steps {
            let obs_f32: Vec<f32> = obs.iter().map(|&x| x as f32).collect();
            let norm = normalizer.normalize_f32(&obs_f32);
            let (action, log_prob) = agent.act(&norm);
            let outcome = env.step(&action);

            let mut reward = outcome.reward;
            if outcome.terminated {
                reward += env.crash_penalty();
            }
            buffer.push(Transition {
                obs: obs_f32,
                action,
                log_prob,
                reward: reward as f32,
                terminated: outcome.terminated,
                truncated: outcome.truncated,
            });

            let next_f32: Vec<f32> = outcome.obs.iter().map(|&x| x as f32).collect();
            final_obs = Some(next_f32);

            if outcome.terminated || outcome.truncated {
                obs = env.reset();
                normalizer.observe(&obs);
            } else {
                obs = outcome.obs;
            }
        }

        let final_obs = final_obs.expect("rollout_steps > 0 guarantees one transition");

        // ---- 2. Values + GAE ----
        let norm_batch = normalizer.normalize_batch(&buffer.obs_flat());
        let values = agent.value_all(&norm_batch);
        let last_value = match buffer.last() {
            Some(last) if last.terminated => 0.0,
            Some(_) => agent.value_single(&normalizer.normalize_f32(&final_obs)),
            None => 0.0,
        };
        let (advantages, returns) = compute_gae(
            &buffer.rewards(),
            &values,
            &buffer.terminated(),
            last_value,
            gamma,
            lambda,
        );

        // ---- 3. Update ----
        let batch = UpdateBatch {
            obs: norm_batch,
            actions: buffer.actions_flat(),
            old_log_probs: buffer.log_probs(),
            advantages,
            returns,
        };
        let stats = agent.update(&batch);

        if tcfg.print_every > 0 && (it % tcfg.print_every == 0 || it == tcfg.iterations - 1) {
            println!(
                "[rl_agent] iter {:>4}/{}  policy {:.4}  value {:.4}  entropy {:.4}  adv {:.3}±{:.3}",
                it + 1,
                tcfg.iterations,
                stats.policy_loss,
                stats.value_loss,
                stats.entropy,
                stats.advantage_mean,
                stats.advantage_std,
            );
        }

        if tcfg.eval_every > 0 && it % tcfg.eval_every == 0 {
            last_eval = Some(evaluate(
                agent.as_ref(),
                &mut env,
                &normalizer,
                tcfg.eval_episodes,
                tcfg.env_max_steps,
            ));
            if let Some(ev) = last_eval {
                println!(
                    "[rl_agent] eval       alt_err {:.1} m  spd_err {:.1} m/s  reward {:.1}  crashes {}/{}",
                    ev.alt_err, ev.spd_err, ev.total_reward, ev.crashes, ev.episodes,
                );
            }
        }

        if tcfg.save_every > 0 && (it % tcfg.save_every == 0 || it == tcfg.iterations - 1) {
            save_checkpoint(agent.as_ref(), &normalizer, tcfg, ncfg, &tcfg.checkpoint_dir)?;
        }
    }

    Ok(TrainingOutcome {
        algorithm: agent,
        normalizer,
        last_eval,
    })
}

/// Run `episodes` deterministic episodes and aggregate metrics.
pub fn evaluate<B: AutodiffBackend<FloatElem = f32>>(
    agent: &dyn Algorithm<B>,
    env: &mut dyn EnvSpec,
    normalizer: &RunningNormalizer,
    episodes: usize,
    max_steps: usize,
) -> EvalMetrics {
    let mut m = EvalMetrics {
        episodes,
        ..Default::default()
    };
    if episodes == 0 {
        return m;
    }
    let target_alt = env.target_altitude();
    let target_spd = env.target_airspeed();

    for _ in 0..episodes {
        let mut obs = env.reset();
        let mut total_reward = 0.0f64;
        let mut steps = 0usize;
        let mut crashed = false;
        let mut alt_err = 0.0f64;
        let mut spd_err = 0.0f64;
        let mut done = false;

        while !done && steps < max_steps {
            let obs_f32: Vec<f32> = obs.iter().map(|&x| x as f32).collect();
            let action = agent.deterministic_action(&normalizer.normalize_f32(&obs_f32));
            let out = env.step(&action);
            total_reward += out.reward;
            if out.terminated {
                total_reward += env.crash_penalty();
                crashed = true;
            }
            alt_err += (out.obs[10] - target_alt).abs();
            spd_err += (out.obs[11] - target_spd).abs();
            done = out.terminated || out.truncated;
            obs = out.obs;
            steps += 1;
        }

        m.alt_err += (alt_err / steps.max(1) as f64) as f32;
        m.spd_err += (spd_err / steps.max(1) as f64) as f32;
        m.total_reward += (total_reward / steps.max(1) as f64) as f32;
        m.mean_steps += steps as f32;
        m.crashes += usize::from(crashed);
    }

    let n = episodes as f32;
    m.alt_err /= n;
    m.spd_err /= n;
    m.total_reward /= n;
    m.mean_steps /= n;
    m
}

/// Save the model weights (bin) plus JSON configs and normalizer.
///
/// The algorithm id + hyper-parameters are persisted as `algo.json`; the
/// network architecture as `net.json`; weights as `net.bin` (via the
/// algorithm's own recorder).
pub fn save_checkpoint<B: AutodiffBackend<FloatElem = f32>>(
    agent: &dyn Algorithm<B>,
    normalizer: &RunningNormalizer,
    tcfg: &TrainerConfig,
    ncfg: &NetConfig,
    dir: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    std::fs::create_dir_all(dir)?;
    agent.save_weights(Path::new(dir))?;
    write_json(Path::new(dir).join("net.json"), &serde_json::to_string_pretty(ncfg)?)?;
    write_json(Path::new(dir).join("algo.json"), &agent.spec().to_json()?)?;
    write_json(
        Path::new(dir).join("trainer.json"),
        &serde_json::to_string_pretty(tcfg)?,
    )?;
    write_json(
        Path::new(dir).join("normalizer.json"),
        &serde_json::to_string_pretty(normalizer)?,
    )?;
    Ok(())
}

fn write_json(path: std::path::PathBuf, contents: &str) -> Result<(), Box<dyn std::error::Error>> {
    std::fs::write(path, contents)?;
    Ok(())
}

/// Loaded checkpoint: the trained algorithm plus the stats needed to run it.
pub struct Checkpoint<B: AutodiffBackend<FloatElem = f32>> {
    /// Trained algorithm (weights on `device`), algorithm-agnostic.
    pub algorithm: Box<dyn Algorithm<B>>,
    /// Normalizer stats to apply before feeding observations to the net.
    pub normalizer: RunningNormalizer,
}

/// Load a checkpoint from `dir`. Optimizer state is not stored in v1 and is
/// rebuilt fresh from the algorithm config by the factory.
///
/// Checkpoints written by v0.1 trainers (before the algorithm layer) used
/// `ppo.json` instead of `algo.json`; those are still loadable.
pub fn load_checkpoint<B: AutodiffBackend<FloatElem = f32>>(
    dir: &str,
    device: &B::Device,
    seed: u64,
) -> Result<Checkpoint<B>, Box<dyn std::error::Error>> {
    let net_cfg: NetConfig = serde_json::from_str(&std::fs::read_to_string(
        Path::new(dir).join("net.json"),
    )?)?;
    let spec: AlgoSpec = {
        let algo_path = Path::new(dir).join("algo.json");
        if algo_path.exists() {
            AlgoSpec::from_json(&std::fs::read_to_string(algo_path)?)?
        } else {
            // Legacy < 0.2 checkpoints: PPO only.
            let ppo_cfg: PpoConfig = serde_json::from_str(&std::fs::read_to_string(
                Path::new(dir).join("ppo.json"),
            )?)?;
            AlgoSpec::Ppo { config: ppo_cfg }
        }
    };
    let normalizer: RunningNormalizer = serde_json::from_str(
        &std::fs::read_to_string(Path::new(dir).join("normalizer.json"))?,
    )?;
    let mut algorithm = make_algorithm(&spec, &net_cfg, seed, device);
    algorithm.load_weights(Path::new(dir), device)?;
    Ok(Checkpoint {
        algorithm,
        normalizer,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::Cpu;

    fn config_path() -> String {
        if Path::new("../aircraft.toml").exists() {
            "../aircraft.toml".to_string()
        } else {
            "aircraft.toml".to_string()
        }
    }

    fn trained() -> TrainingOutcome<Cpu> {
        let device = crate::backend::cpu_device();
        let tcfg = TrainerConfig {
            iterations: 2,
            rollout_steps: 64,
            env_max_steps: 300,
            env_dt: 0.05,
            seed: 1234,
            save_every: 0,
            print_every: 0,
            eval_episodes: 1,
            eval_every: 1,
            checkpoint_dir: "checkpoints".to_string(),
        };
        let ncfg = NetConfig {
            obs_dim: AVIONICS_OBS_DIM,
            hidden: 32,
            action_dim: ACTION_DIM,
            log_std_init: -0.5,
        };
        let spec = AlgoSpec::Ppo {
            config: PpoConfig {
                lr: 1e-3,
                gamma: 0.99,
                lambda: 0.95,
                clip_epsilon: 0.2,
                value_coef: 0.5,
                entropy_coef: 0.0,
                max_grad_norm: 5.0,
                epochs: 1,
                minibatch_size: 32,
            },
        };
        run::<Cpu>(&tcfg, &ncfg, &spec, &device, &config_path()).expect("run should succeed")
    }

    #[test]
    fn training_run_produces_finite_results_and_eval() {
        let outcome = trained();
        assert!(outcome.normalizer.count > 0);
        // Rollouts + an eval episode must move the running stats.
        assert!(outcome.normalizer.mean.iter().all(|m| m.is_finite()));
        let ev = outcome.last_eval.expect("eval_every=1 must produce metrics");
        assert!(ev.alt_err.is_finite());
        assert!(ev.spd_err.is_finite());
        assert!(ev.total_reward.is_finite());
        assert_eq!(ev.episodes, 1);
    }

    #[test]
    fn same_seed_reproduces_identical_training() {
        let a = trained();
        let b = trained();
        assert_eq!(a.normalizer.mean, b.normalizer.mean, "normalizer must match run-to-run");
        let obs = vec![0.0f32; AVIONICS_OBS_DIM];
        // Deterministic actions on a fixed input must match exactly.
        let act_a = a.algorithm.deterministic_action(&obs);
        let act_b = b.algorithm.deterministic_action(&obs);
        assert_eq!(act_a, act_b, "same seed must reproduce the trained policy");
    }

    #[test]
    fn action_mapping_stays_inside_fc_limits() {
        let action = [0.999f32, -0.999, 0.999, -0.999];
        let av = action_to_avionics(&action);
        assert!(av.roll_cmd.abs() <= 0.6 + 1e-6);
        assert!(av.pitch_cmd.abs() <= 0.4 + 1e-6);
        assert!(av.yaw_rate_cmd.abs() <= 0.8 + 1e-6);
        assert!((0.0..=1.0).contains(&av.throttle));
    }

    #[test]
    fn checkpoint_round_trip() {
        use crate::net::NetConfig;
        let device = crate::backend::cpu_device();
        let tcfg = TrainerConfig {
            iterations: 2,
            rollout_steps: 16,
            env_max_steps: 100,
            env_dt: 0.05,
            seed: 99,
            save_every: 1,
            print_every: 0,
            eval_episodes: 0,
            eval_every: 0,
            checkpoint_dir: std::env::temp_dir().join("rl_agent_cp").to_str().unwrap().to_string(),
        };
        let ncfg = NetConfig {
            obs_dim: AVIONICS_OBS_DIM,
            hidden: 16,
            action_dim: ACTION_DIM,
            log_std_init: -0.5,
        };
        let spec = AlgoSpec::Ppo {
            config: PpoConfig {
                minibatch_size: 16,
                epochs: 1,
                ..Default::default()
            },
        };
        run::<Cpu>(&tcfg, &ncfg, &spec, &device, &config_path()).expect("run");

        let cp = load_checkpoint::<Cpu>(&tcfg.checkpoint_dir, &device, tcfg.seed).expect("load");
        assert_eq!(cp.algorithm.net_config().hidden, 16);
        assert_eq!(cp.algorithm.id(), crate::algo::ALGO_PPO);
        match cp.algorithm.spec() {
            AlgoSpec::Ppo { config } => assert_eq!(config.minibatch_size, 16),
        }
        assert_eq!(cp.normalizer.dim, AVIONICS_OBS_DIM);

        let _ = std::fs::remove_dir_all(&tcfg.checkpoint_dir);
    }

    #[test]
    fn loads_legacy_ppo_checkpoint_without_algo_json() {
        // Checkpoints written before the algorithm layer stored PPO config as
        // `ppo.json` and had no `algo.json`; they must still load.
        use crate::net::NetConfig;
        let device = crate::backend::cpu_device();
        let dir = std::env::temp_dir()
            .join("rl_agent_cp_legacy")
            .to_str()
            .unwrap()
            .to_string();
        let tcfg = TrainerConfig {
            iterations: 1,
            rollout_steps: 16,
            env_max_steps: 100,
            env_dt: 0.05,
            seed: 7,
            save_every: 1,
            print_every: 0,
            eval_episodes: 0,
            eval_every: 0,
            checkpoint_dir: dir.clone(),
        };
        let ncfg = NetConfig {
            obs_dim: AVIONICS_OBS_DIM,
            hidden: 16,
            action_dim: ACTION_DIM,
            log_std_init: -0.5,
        };
        let spec = AlgoSpec::Ppo {
            config: PpoConfig::default(),
        };
        run::<Cpu>(&tcfg, &ncfg, &spec, &device, &config_path()).expect("run");

        // Rewrite the checkpoint as a legacy one: drop algo.json, add ppo.json.
        std::fs::remove_file(std::path::Path::new(&dir).join("algo.json")).expect("remove algo.json");
        let ppo_cfg = match spec {
            AlgoSpec::Ppo { config } => config,
        };
        std::fs::write(
            std::path::Path::new(&dir).join("ppo.json"),
            serde_json::to_string_pretty(&ppo_cfg).expect("serialize ppo.json"),
        )
        .expect("write ppo.json");

        let cp = load_checkpoint::<Cpu>(&dir, &device, tcfg.seed).expect("legacy load");
        assert_eq!(cp.algorithm.id(), crate::algo::ALGO_PPO);
        match cp.algorithm.spec() {
            AlgoSpec::Ppo { config } => assert_eq!(config.minibatch_size, PpoConfig::default().minibatch_size),
        }
        let _ = std::fs::remove_dir_all(&dir);
    }
}