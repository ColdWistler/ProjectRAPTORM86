//! Pluggable AI agent framework for the RAPTOR M86 flight simulator.
//!
//! Provides a trait-based abstraction (`Agent`) that lets different
//! reinforcement-learning algorithms (D3QN, PPO, SAC, …) and classical
//! controllers (PID) share a common interface for inference and episode
//! management. An `AgentRegistry` maps string keys to factory functions so
//! the active agent can be swapped at runtime without recompilation.
//!
//! # Quick start
//! ```rust,no_run
//! use flight_ai::{AgentConfig, AgentRegistry, Agent};
//!
//! let mut registry = AgentRegistry::new();
//! // Built-in agents are registered by default.
//! let config = AgentConfig::new("d3qn");
//! let mut agent = registry.build(&config);
//! // agent.reset();
//! // let action = agent.act(&observation);
//! ```

pub mod agents;
mod config;
mod registry;

pub use config::AgentConfig;
pub use registry::AgentRegistry;

use flight_core::ControlAction;
use std::path::Path;

/// The core interface every AI agent must implement.
///
/// Methods are split into *inference* (`act`, `observe`) and *lifecycle*
/// (`reset`, `on_step`, `save`, `load`). Implementations may keep
/// internal state (neural-network weights, replay buffers, PID
/// integrators, …) that is reset on each episode via [`Agent::reset`].
///
/// # Object safety
/// This trait is object-safe: all methods take `&self` or `&mut self` and
/// return sized types. Store agents as `Box<dyn Agent>`.
pub trait Agent: Send {
    /// Human-readable name of the agent (e.g. `"d3qn"`, `"pid"`).
    fn name(&self) -> &str;

    /// Dimensionality of the observation vector this agent expects.
    fn obs_dim(&self) -> usize;

    /// Dimensionality of the action vector this agent produces.
    fn act_dim(&self) -> usize;

    /// Select a control action given a flat observation vector.
    ///
    /// The observation layout must match [`Agent::obs_dim`]. Implementations
    /// may scale, clip, or discretise internally before returning a
    /// [`ControlAction`].
    fn act(&self, observation: &[f64]) -> ControlAction;

    /// Map a full [`flight_core::AircraftState`] into the agent's
    /// observation vector. The default implementation calls
    /// `AircraftState::to_observation_array` and truncates/pads to
    /// `obs_dim`.
    fn observe(&self, state: &flight_core::AircraftState) -> Vec<f64> {
        let full = state.to_observation_array();
        let dim = self.obs_dim();
        if dim <= full.len() {
            full[..dim].to_vec()
        } else {
            let mut obs = full.to_vec();
            obs.resize(dim, 0.0);
            obs
        }
    }

    /// Called at the beginning of every episode. Reset any internal state
    /// (integrators, hidden states, exploration noise, …).
    fn reset(&mut self);

    /// Optional per-step hook called *after* the environment advances.
    ///
    /// `reward` is the scalar reward for this transition, `done` indicates
    /// whether the episode has ended, and `truncated` indicates a time-limit
    /// truncation. Implementations can use this for experience replay,
    /// advantage estimation, curriculum updates, etc.
    fn on_step(
        &mut self,
        observation: &[f64],
        action: &ControlAction,
        reward: f64,
        next_observation: &[f64],
        done: bool,
    ) {
        let _ = (observation, action, reward, next_observation, done);
    }

    /// Persist the agent's learnable parameters (weights, biases, …) to
    /// `path`. The format is algorithm-specific.
    fn save(&self, _path: &Path) -> std::io::Result<()> {
        Ok(())
    }

    /// Load learnable parameters from `path`. If the file does not exist
    /// the agent should keep its current (random / default) initialisation.
    fn load(&mut self, _path: &Path) -> std::io::Result<()> {
        Ok(())
    }
}
