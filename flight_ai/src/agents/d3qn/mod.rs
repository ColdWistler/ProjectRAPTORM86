//! Dueling Double DQN (D3QN) agent skeleton.
//!
//! Provides the [`Agent`] trait implementation with a placeholder
//! action-selection strategy. Replace the TODO bodies with your
//! neural-network inference (e.g. via [`tch-rs`] or [`burn`]).
//!
//! # Architecture
//! The D3QN splits the Q-value into a state-value stream and an
//! advantage stream, combined as `Q = V + A - mean(A)`. Two identical
//! networks (online / target) provide stable bootstrapping targets.
//!
//! # Hyper-parameters (read from [`AgentConfig::params`])
//! | Key               | Default  | Description                            |
//! |-------------------|----------|----------------------------------------|
//! | `obs_dim`         | 12       | Observation vector length              |
//! | `n_actions`       | 5        | Discrete action count per channel      |
//! | `hidden_sizes`    | [256,256]| Fully-connected hidden layer widths    |
//! | `learning_rate`   | 1e-4     | Adam learning rate                     |
//! | `gamma`           | 0.99     | Discount factor                        |
//! | `epsilon_start`   | 1.0      | Initial ε for ε-greedy exploration     |
//! | `epsilon_end`     | 0.05     | Final ε                                 |
//! | `epsilon_decay`   | 500      | Episodes to linearly anneal ε           |
//! | `batch_size`      | 64       | Mini-batch size for replay sampling     |
//! | `buffer_size`     | 100_000  | Experience replay buffer capacity       |
//! | `tau`             | 0.005    | Polyak averaging rate for target update |
//! | `target_update`   | 1000     | Steps between hard target-network syncs |

use crate::config::AgentConfig;
use crate::Agent;
use flight_core::ControlAction;

/// A placeholder D3QN agent. The action-selection and training logic
/// are stubs that return deterministic defaults — replace with real
/// tensor inference when a deep-learning backend is integrated.
pub struct D3QNAgent {
    obs_dim: usize,
    n_actions: usize,
    #[allow(dead_code)]
    hidden_sizes: Vec<usize>,
    #[allow(dead_code)]
    learning_rate: f64,
    #[allow(dead_code)]
    gamma: f64,
    epsilon_start: f64,
    epsilon_end: f64,
    epsilon_decay: f64,
    #[allow(dead_code)]
    batch_size: usize,
    #[allow(dead_code)]
    buffer_size: usize,
    #[allow(dead_code)]
    tau: f64,
    target_update: usize,

    // --- runtime state ---
    episode: usize,
    step: usize,
    current_epsilon: f64,
}

impl D3QNAgent {
    /// Build from TOML config, falling back to documented defaults.
    pub fn from_config(cfg: &AgentConfig) -> Self {
        let epsilon_start = cfg.get_f64("epsilon_start", 1.0);
        Self {
            obs_dim: cfg.get_usize("obs_dim", 12),
            n_actions: cfg.get_usize("n_actions", 5),
            hidden_sizes: cfg.get_usize_vec("hidden_sizes", vec![256, 256]),
            learning_rate: cfg.get_f64("learning_rate", 1e-4),
            gamma: cfg.get_f64("gamma", 0.99),
            epsilon_start,
            epsilon_end: cfg.get_f64("epsilon_end", 0.05),
            epsilon_decay: cfg.get_f64("epsilon_decay", 500.0),
            batch_size: cfg.get_usize("batch_size", 64),
            buffer_size: cfg.get_usize("buffer_size", 100_000),
            tau: cfg.get_f64("tau", 0.005),
            target_update: cfg.get_usize("target_update", 1000),

            episode: 0,
            step: 0,
            current_epsilon: epsilon_start,
        }
    }

    /// Current exploration rate.
    pub fn epsilon(&self) -> f64 {
        self.current_epsilon
    }

    /// Linearly anneal ε from `epsilon_start` → `epsilon_end` over
    /// `epsilon_decay` episodes.
    fn update_epsilon(&mut self) {
        let progress = (self.episode as f64 / self.epsilon_decay).min(1.0);
        self.current_epsilon =
            self.epsilon_start + progress * (self.epsilon_end - self.epsilon_start);
    }

    /// TODO: Replace with actual Q-value forward pass through the
    /// online network. Returns the Q-value for each action given `obs`.
    fn q_values(&self, _obs: &[f64]) -> Vec<f64> {
        // Placeholder: equal Q-value for every action → random uniform.
        vec![0.0; self.n_actions]
    }

    /// TODO: Replace with a mini-batch sample from the replay buffer,
    /// a forward + backward pass on the online network, and a soft
    /// (Polyak) or hard update of the target network.
    fn train_step(&mut self) {
        // No-op skeleton.
    }
}

impl Agent for D3QNAgent {
    fn name(&self) -> &str {
        "d3qn"
    }

    fn obs_dim(&self) -> usize {
        self.obs_dim
    }

    fn act_dim(&self) -> usize {
        self.n_actions
    }

    fn act(&self, obs: &[f64]) -> ControlAction {
        // --- ε-greedy action selection ---
        let _rng_roll: f64 = 0.5; // TODO: replace with real RNG
        let _explore = _rng_roll < self.current_epsilon;

        let _q = self.q_values(obs);

        // TODO: if explore → random action index
        //        else     → argmax(_q)
        // For now: return neutral controls.
        let _action_idx: usize = 0;

        // TODO: map action_idx to a meaningful ControlAction.
        // With 5 discrete channels × N bins each, discretise to:
        //   elevator ∈ [-0.35, 0.35]
        //   aileron  ∈ [-0.35, 0.35]
        //   rudder   ∈ [-0.35, 0.35]
        //   throttle ∈ [0.0, 1.0]
        //   flaps    ∈ [0.0, 0.7]
        ControlAction {
            elevator: 0.0,
            aileron: 0.0,
            rudder: 0.0,
            throttle: 0.5,
            flaps: 0.0,
        }
    }

    fn reset(&mut self) {
        self.update_epsilon();
        self.episode += 1;
        self.step = 0;
    }

    fn on_step(
        &mut self,
        _observation: &[f64],
        _action: &ControlAction,
        _reward: f64,
        _next_observation: &[f64],
        _done: bool,
    ) {
        // TODO: push transition into the replay buffer.
        self.step += 1;

        // Periodic training.
        if self.step % 4 == 0 {
            self.train_step();
        }

        // Periodic target-network sync.
        if self.step % self.target_update == 0 {
            // TODO: hard-copy online → target, or Polyak averaging.
        }
    }
}
