//! Pluggable training-algorithm layer.
//!
//! The trainer is algorithm-agnostic: everything it needs from an RL model
//! lives behind the [`Algorithm`] trait, and *which* algorithm + hyper-
//! parameters to run comes from the serializable [`AlgoSpec`] enum. The
//! trainer loop that consumes this seam (rollout → GAE → `update`) lives in
//! [`crate::trainer`].
//!
//! # Adding a new model
//!
//! 1. Write a module (e.g. `dqn.rs`) owning the agent struct, its config,
//!    its update math and its checkpoint (weights + config) bookkeeping.
//! 2. `impl Algorithm<B>` for that struct.
//! 3. Add an [`AlgoSpec`] variant and one arm in [`make_algorithm`] (the
//!    registry).
//! 4. Register the id string wherever selections come from: the CLI
//!    (`rl_agent --algo <id>` in `main.rs`) and the Godot menu dropdown
//!    (`ALGORITHMS` in `godot/scripts/rl_trainer.gd`).

use crate::net::NetConfig;
use crate::ppo::{PpoAgent, PpoConfig};
use crate::a2c::{A2cAgent, A2cConfig};
use crate::d3qn::{D3qnConfig, D3qnAgent};
use crate::sac::{SacConfig, SacAgent};
use crate::td3::{Td3Agent, Td3Config};
use burn::tensor::backend::AutodiffBackend;
use serde::{Deserialize, Serialize};
use std::path::Path;

/// Algorithm id used by `--algo ppo` and the menu dropdown.
pub const ALGO_PPO: &str = "ppo";
pub const ALGO_A2C: &str = "a2c";
pub const ALGO_D3QN: &str = "d3qn";
pub const ALGO_SAC: &str = "sac";
pub const ALGO_TD3: &str = "td3";

/// Every registered algorithm id. Selections (CLI helpers, menu dropdowns)
/// are drawn from this list.
pub const ALGORITHM_IDS: &[&str] = &[ALGO_PPO, ALGO_A2C, ALGO_D3QN, ALGO_SAC, ALGO_TD3];

/// Serializable selection of *which* algorithm + hyper-parameters to run.
/// Persisted into every checkpoint as `algo.json`.
///
/// This is the extension seam: adding a variant here, plus a module and a
/// [`make_algorithm`] arm, is all it takes to make a new model selectable.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
#[serde(tag = "id", rename_all = "snake_case")]
pub enum AlgoSpec {
    /// Proximal Policy Optimization (clipped surrogate objective).
    Ppo {
        /// PPO hyper-parameters.
        config: PpoConfig,
    },
    /// Advantage Actor-Critic (simpler actor-critic, no clipping).
    A2c {
        /// A2C hyper-parameters.
        config: A2cConfig,
    },
    /// Continuous D3QN — dueling Q + prioritized replay + tanh continuous actions.
    D3qn {
        /// D3QN hyper-parameters.
        config: D3qnConfig,
    },
    /// Soft Actor-Critic — off-policy twin critics + entropy bonus + Polyak tau.
    Sac {
        /// SAC hyper-parameters.
        config: SacConfig,
    },
    /// TD3 — off-policy twin-delayed DDPG with deterministic actor, twin critics,
    /// target noise smoothing, and delayed policy updates.
    Td3 {
        /// TD3 hyper-parameters.
        config: Td3Config,
    },
}

impl AlgoSpec {
    /// Stable identifier used on the CLI, in `algo.json`, and in menus.
    pub fn id(&self) -> &'static str {
        match self {
            AlgoSpec::Ppo { .. } => ALGO_PPO,
            AlgoSpec::A2c { .. } => ALGO_A2C,
            AlgoSpec::D3qn { .. } => ALGO_D3QN,
            AlgoSpec::Sac { .. } => ALGO_SAC,
            AlgoSpec::Td3 { .. } => ALGO_TD3,
        }
    }

    /// Build a spec with default hyper-parameters for `id`.
    pub fn from_id(id: &str) -> Result<Self, String> {
        match id {
            ALGO_PPO => Ok(AlgoSpec::Ppo {
                config: PpoConfig::default(),
            }),
            ALGO_A2C => Ok(AlgoSpec::A2c { config: A2cConfig::default() }),
            ALGO_D3QN => Ok(AlgoSpec::D3qn { config: D3qnConfig::default() }),
            ALGO_SAC => Ok(AlgoSpec::Sac { config: SacConfig::default() }),
            ALGO_TD3 => Ok(AlgoSpec::Td3 { config: Td3Config::default() }),
            other => Err(format!(
                "unknown algorithm `{other}` (registered: {})",
                ALGORITHM_IDS.join(", ")
            )),
        }
    }

    /// Serialized JSON for `algo.json`.
    pub fn to_json(&self) -> Result<String, serde_json::Error> {
        serde_json::to_string_pretty(self)
    }

    /// Parse an [`AlgoSpec`] from `algo.json`.
    pub fn from_json(json: &str) -> Result<Self, serde_json::Error> {
        serde_json::from_str(json)
    }
}

/// Algorithm-agnostic update pass result (aggregated for monitoring/logging).
#[derive(Debug, Clone, Copy, Default)]
#[allow(clippy::module_name_repetitions)]
pub struct UpdateStats {
    /// Mean policy surrogate loss (negative for an improving policy).
    pub policy_loss: f32,
    /// Mean squared value error.
    pub value_loss: f32,
    /// Mean policy entropy.
    pub entropy: f32,
    /// Mean advantage over the whole batch (before whitening).
    pub advantage_mean: f32,
    /// Standard deviation of the advantages over the whole batch.
    pub advantage_std: f32,
}

/// One transition batch handed to [`Algorithm::update`] after a rollout and a
/// host-side GAE pass. On-policy algorithms (PPO) consume everything;
/// off-policy algorithms can build their own targets and ignore the fields
/// they do not need.
#[derive(Debug, Clone)]
pub struct UpdateBatch {
    /// Flattened `n·obs_dim` observations, already normalized.
    pub obs: Vec<f32>,
    /// Flattened `n·action_dim` actions (tanh-space).
    pub actions: Vec<f32>,
    /// Per-step `log π_old(a|s)` (from rollout sampling).
    pub old_log_probs: Vec<f32>,
    /// Per-step GAE advantages.
    pub advantages: Vec<f32>,
    /// Per-step `returns = advantage + value`.
    pub returns: Vec<f32>,
}

/// Everything the trainer needs from an RL algorithm.
///
/// The shared trainer loop rolls out with [`Algorithm::act`] (stochastic
/// actions + log-probs), bootstraps the batch tail with
/// [`Algorithm::value_single`], computes GAE on the host with the discount /
/// trace-decay from [`Algorithm::td_params`], calls [`Algorithm::update`],
/// and persists weights via [`Algorithm::save_weights`] /
/// [`Algorithm::load_weights`].
pub trait Algorithm<B: AutodiffBackend> {
    /// Stable algorithm id (see [`ALGORITHM_IDS`]).
    fn id(&self) -> &'static str;
    /// Architecture config (observation/action dims, hidden width).
    fn net_config(&self) -> &NetConfig;
    /// Serializable algorithm spec (id + hyper-parameters).
    fn spec(&self) -> AlgoSpec;
    /// Discount factor and GAE trace-decay used to compute bootstrapped
    /// returns in the shared loop. Value-free / off-policy algorithms that
    /// rebuild the loop can return `(0.99, 0.95)` and ignore them.
    fn td_params(&self) -> (f32, f32) {
        (0.99, 0.95)
    }
    /// Sample one stochastic action for a normalized observation.
    /// Returns `(action, log π(a|s))` in tanh-space.
    fn act(&mut self, obs_norm: &[f32]) -> (Vec<f32>, f32);
    /// Greedy deterministic action for a normalized observation (evaluation).
    fn deterministic_action(&self, obs_norm: &[f32]) -> Vec<f32>;
    /// Value of a single normalized observation — used to bootstrap the GAE
    /// tail. Value-free algorithms return 0.0 and rely on `truncated` flags.
    fn value_single(&self, obs_norm: &[f32]) -> f32;
    /// Values for a flattened batch of normalized observations.
    fn value_all(&self, obs_norm_flat: &[f32]) -> Vec<f32>;
    /// One update pass over the collected batch.
    fn update(&mut self, batch: &UpdateBatch) -> UpdateStats;
    /// Persist the model weights under `dir` (e.g. `dir/net`).
    fn save_weights(&self, dir: &Path) -> Result<(), Box<dyn std::error::Error>>;
    /// Load the model weights under `dir` onto `device`, replacing this
    /// algorithm's parameters.
    fn load_weights(
        &mut self,
        dir: &Path,
        device: &B::Device,
    ) -> Result<(), Box<dyn std::error::Error>>;
}

/// Build a boxed algorithm for a concrete backend — the algorithm registry.
///
/// Adding a model = adding an [`AlgoSpec`] variant and one arm here (plus an
/// `Algorithm` implementation).
pub fn make_algorithm<B: AutodiffBackend<FloatElem = f32>>(
    spec: &AlgoSpec,
    net_cfg: &NetConfig,
    seed: u64,
    device: &B::Device,
) -> Box<dyn Algorithm<B>> {
    match *spec {
        AlgoSpec::Ppo { config } => Box::new(PpoAgent::<B>::new(device, net_cfg, &config, seed)),
        AlgoSpec::A2c { config } => Box::new(A2cAgent::<B>::new(device, net_cfg, &config, seed)),
        AlgoSpec::D3qn { config } => Box::new(D3qnAgent::<B>::new(device, net_cfg, &config, seed)),
        AlgoSpec::Sac { config } => Box::new(SacAgent::<B>::new(device, net_cfg, &config, seed)),
        AlgoSpec::Td3 { config } => Box::new(Td3Agent::<B>::new(device, net_cfg, &config, seed)),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::Cpu;

    #[test]
    fn algo_spec_serde_round_trip() {
        let spec = AlgoSpec::Ppo {
            config: PpoConfig::default(),
        };
        let json = spec.to_json().expect("serialize");
        let back = AlgoSpec::from_json(&json).expect("deserialize");
        assert_eq!(back.id(), ALGO_PPO);
        match back {
            AlgoSpec::Ppo { config } => assert_eq!(config.lr, PpoConfig::default().lr),
            _ => (),
        }
    }

    #[test]
    fn from_id_resolves_registered_algorithms() {
        assert_eq!(
            AlgoSpec::from_id(ALGO_PPO).expect("ppo").id(),
            ALGO_PPO
        );
        assert_eq!(
            AlgoSpec::from_id(ALGO_A2C).expect("a2c").id(),
            ALGO_A2C
        );
        assert!(AlgoSpec::from_id("dqn").is_err());
        assert!(AlgoSpec::from_id("a2c_beta").is_err());
    }

    #[test]
    fn algo_spec_serde_round_trip_a2c() {
        use crate::a2c::A2cConfig;
        let spec = AlgoSpec::A2c {
            config: A2cConfig::default(),
        };
        let json = spec.to_json().expect("serialize");
        let back = AlgoSpec::from_json(&json).expect("deserialize");
        assert_eq!(back.id(), ALGO_A2C);
        match back {
            AlgoSpec::A2c { config } => assert_eq!(config, A2cConfig::default()),
            _ => panic!("expected A2c spec"),
        }
    }

    #[test]
    fn from_id_resolves_d3qn() {
        assert_eq!(AlgoSpec::from_id(ALGO_D3QN).expect("d3qn").id(), ALGO_D3QN);
    }

    #[test]
    fn d3qn_spec_serde_round_trip() {
        use crate::d3qn::D3qnConfig;
        let spec = AlgoSpec::D3qn { config: D3qnConfig::default() };
        let json = spec.to_json().expect("serialize");
        let back = AlgoSpec::from_json(&json).expect("deserialize");
        assert_eq!(back.id(), ALGO_D3QN);
        match back {
            AlgoSpec::D3qn { config } => assert_eq!(config.hidden, D3qnConfig::default().hidden),
            _ => panic!("expected D3qn spec"),
        }
    }

    #[test]
    fn from_id_resolves_td3() {
        assert_eq!(AlgoSpec::from_id(ALGO_TD3).expect("td3").id(), ALGO_TD3);
    }

    #[test]
    fn td3_spec_serde_round_trip() {
        use crate::td3::Td3Config;
        let spec = AlgoSpec::Td3 { config: Td3Config::default() };
        let json = spec.to_json().expect("serialize");
        let back = AlgoSpec::from_json(&json).expect("deserialize");
        assert_eq!(back.id(), ALGO_TD3);
        match back {
            AlgoSpec::Td3 { config } => assert_eq!(config, Td3Config::default()),
            _ => panic!("expected Td3 spec"),
        }
    }

    #[test]
    fn factory_builds_td3_and_acts() {
        use crate::td3::Td3Config;
        let device = crate::backend::cpu_device();
        let ncfg = NetConfig {
            obs_dim: 4,
            hidden: 16,
            action_dim: 4,
            log_std_init: -0.5,
        };
        let mut algo: Box<dyn Algorithm<Cpu>> = make_algorithm(
            &AlgoSpec::Td3 { config: Td3Config::default() },
            &ncfg,
            1,
            &device,
        );
        assert_eq!(algo.id(), ALGO_TD3);
        let (action, logp) = algo.act(&[0.1, -0.2, 0.3, 0.4]);
        assert_eq!(action.len(), 4);
        assert!(logp.is_finite());
        assert!(action.iter().all(|a| a.abs() < 1.0));
    }

    #[test]
    fn factory_builds_a2c_and_acts() {
        use crate::a2c::A2cConfig;
        let device = crate::backend::cpu_device();
        let ncfg = NetConfig {
            obs_dim: 6,
            hidden: 16,
            action_dim: 2,
            log_std_init: -0.5,
        };
        let mut algo: Box<dyn Algorithm<Cpu>> = make_algorithm(
            &AlgoSpec::A2c {
                config: A2cConfig::default(),
            },
            &ncfg,
            2,
            &device,
        );
        assert_eq!(algo.id(), ALGO_A2C);
        let (action, logp) = algo.act(&[0.0f32; 6]);
        assert_eq!(action.len(), 2);
        assert!(logp.is_finite());
        assert!(action.iter().all(|a| a.abs() < 1.0));
    }

    #[test]
    fn factory_builds_ppo_and_acts() {
        let device = crate::backend::cpu_device();
        let ncfg = NetConfig {
            obs_dim: 6,
            hidden: 16,
            action_dim: 2,
            log_std_init: -0.5,
        };
        let mut algo: Box<dyn Algorithm<Cpu>> = make_algorithm(
            &AlgoSpec::Ppo {
                config: Default::default(),
            },
            &ncfg,
            1,
            &device,
        );
        assert_eq!(algo.id(), ALGO_PPO);
        let (action, logp) = algo.act(&[0.0f32; 6]);
        assert_eq!(action.len(), 2);
        assert!(logp.is_finite());
    }
}