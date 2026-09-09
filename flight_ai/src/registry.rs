//! Dynamic agent registry and factory.
//!
//! `AgentRegistry` maps string keys to factory closures so agents can be
//! created (and swapped) at runtime without recompilation. Built-in agents
//! are registered automatically; user-defined agents can be added via
//! [`AgentRegistry::register`].
//!
//! # Example
//! ```rust,no_run
//! use flight_ai::{AgentConfig, AgentRegistry};
//!
//! let mut registry = AgentRegistry::new();
//!
//! // List available agents:
//! for name in registry.available() {
//!     println!("{name}");
//! }
//!
//! // Build from config:
//! let config = AgentConfig::new("pid");
//! let mut agent = registry.build(&config);
//! agent.reset();
//! ```

use crate::agents::d3qn::D3QNAgent;
use crate::agents::rule_based::PidAgent;
use crate::config::AgentConfig;
use crate::Agent;
use std::collections::HashMap;

/// A factory function that creates a boxed agent from a config.
type AgentFactory = Box<dyn Fn(&AgentConfig) -> Box<dyn Agent> + Send + Sync>;

/// A registry that maps agent type names to their factory functions.
///
/// Call [`AgentRegistry::new`] to get a pre-populated registry with the
/// built-in agents. Custom agents can be added at any time.
pub struct AgentRegistry {
    factories: HashMap<String, AgentFactory>,
}

impl AgentRegistry {
    /// Create a new registry pre-loaded with all built-in agents
    /// (`pid`, `d3qn`).
    pub fn new() -> Self {
        let mut reg = Self {
            factories: HashMap::new(),
        };
        reg.register_builtin();
        reg
    }

    /// Register a built-in agent under its canonical name.
    fn register_builtin(&mut self) {
        self.register("pid", |cfg| Box::new(PidAgent::from_config(cfg)));
        self.register("d3qn", |cfg| Box::new(D3QNAgent::from_config(cfg)));
    }

    /// Register a new agent type under `name`.
    ///
    /// The factory closure receives an [`AgentConfig`] and must return a
    /// freshly-constructed boxed agent. Overwriting an existing name is
    /// allowed (for hot-swapping at development time).
    pub fn register<F>(&mut self, name: &str, factory: F)
    where
        F: Fn(&AgentConfig) -> Box<dyn Agent> + Send + Sync + 'static,
    {
        self.factories
            .insert(name.to_string(), Box::new(factory));
    }

    /// Build an agent from `config`. The `config.agent` field selects
    /// which factory is invoked.
    ///
    /// # Panics
    /// Panics if `config.agent` is not a registered key. Use
    /// [`AgentRegistry::contains`] to check first.
    pub fn build(&self, config: &AgentConfig) -> Box<dyn Agent> {
        let factory = self.factories.get(&config.agent).unwrap_or_else(|| {
            panic!(
                "AgentRegistry: unknown agent type '{}'. Available: {:?}",
                config.agent,
                self.available()
            )
        });
        factory(config)
    }

    /// Try to build an agent; returns `Err` with the list of available
    /// agents when the key is unknown.
    pub fn try_build(&self, config: &AgentConfig) -> Result<Box<dyn Agent>, Vec<String>> {
        let factory = self
            .factories
            .get(&config.agent)
            .ok_or_else(|| self.available())?;
        Ok(factory(config))
    }

    /// Returns `true` if `name` is a registered agent type.
    pub fn contains(&self, name: &str) -> bool {
        self.factories.contains_key(name)
    }

    /// Sorted list of all registered agent type names.
    pub fn available(&self) -> Vec<String> {
        let mut names: Vec<String> = self.factories.keys().cloned().collect();
        names.sort();
        names
    }
}

impl Default for AgentRegistry {
    fn default() -> Self {
        Self::new()
    }
}
