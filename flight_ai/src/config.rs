//! TOML-loadable agent configuration.
//!
//! `AgentConfig` describes *which* agent to instantiate and carries
//! algorithm-specific hyper-parameters in a flat, serialisable map.
//!
//! # Example TOML
//! ```toml
//! agent = "d3qn"
//!
//! [params]
//! learning_rate = 1e-4
//! gamma = 0.99
//! epsilon_start = 1.0
//! epsilon_end = 0.05
//! hidden_sizes = [256, 256]
//! ```

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs;
use std::path::Path;

/// Configuration for an AI agent, loadable from TOML.
///
/// The `agent` field selects the algorithm family (e.g. `"d3qn"`,
/// `"pid"`). The `params` map carries algorithm-specific tuning knobs
/// that are forwarded to the agent's constructor.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentConfig {
    /// Identifier of the agent type. Must match a key registered in
    /// [`crate::AgentRegistry`].
    pub agent: String,

    /// Algorithm-specific hyper-parameters. Each agent decides which
    /// keys it reads; unknown keys are silently ignored.
    #[serde(default)]
    pub params: HashMap<String, toml::Value>,
}

impl AgentConfig {
    /// Create a minimal config that selects `agent_type` with no extra
    /// parameters (the agent will use its defaults).
    pub fn new(agent_type: &str) -> Self {
        Self {
            agent: agent_type.to_string(),
            params: HashMap::new(),
        }
    }

    /// Look up a typed parameter, returning `default` when missing or
    /// the wrong type.
    pub fn get_f64(&self, key: &str, default: f64) -> f64 {
        self.params
            .get(key)
            .and_then(|v| v.as_float())
            .unwrap_or(default)
    }

    /// Look up a typed parameter, returning `default` when missing or
    /// the wrong type.
    pub fn get_usize(&self, key: &str, default: usize) -> usize {
        self.params
            .get(key)
            .and_then(|v| v.as_integer())
            .map(|v| v as usize)
            .unwrap_or(default)
    }

    /// Look up a typed parameter as `bool`, returning `default` when
    /// missing or the wrong type.
    pub fn get_bool(&self, key: &str, default: bool) -> bool {
        self.params
            .get(key)
            .and_then(|v| v.as_bool())
            .unwrap_or(default)
    }

    /// Look up a typed parameter as a string, returning `default` when
    /// missing or the wrong type.
    pub fn get_str<'a>(&'a self, key: &str, default: &'a str) -> &'a str {
        self.params
            .get(key)
            .and_then(|v| v.as_str())
            .unwrap_or(default)
    }

    /// Look up a parameter as a vector of `f64`, returning `default`
    /// when missing or the wrong type.
    pub fn get_f64_vec(&self, key: &str, default: Vec<f64>) -> Vec<f64> {
        self.params
            .get(key)
            .and_then(|v| v.as_array())
            .map(|arr| {
                arr.iter()
                    .filter_map(|v| v.as_float())
                    .collect()
            })
            .filter(|v: &Vec<f64>| !v.is_empty())
            .unwrap_or(default)
    }

    /// Look up a parameter as a `usize` vector, returning `default`
    /// when missing or the wrong type. Useful for neural-network hidden
    /// layer sizes.
    pub fn get_usize_vec(&self, key: &str, default: Vec<usize>) -> Vec<usize> {
        self.params
            .get(key)
            .and_then(|v| v.as_array())
            .map(|arr| {
                arr.iter()
                    .filter_map(|v| v.as_integer())
                    .map(|v| v as usize)
                    .collect()
            })
            .filter(|v: &Vec<usize>| !v.is_empty())
            .unwrap_or(default)
    }

    /// Deserialise an `AgentConfig` from a TOML file.
    pub fn from_file(path: &Path) -> Result<Self, Box<dyn std::error::Error>> {
        let contents = fs::read_to_string(path)?;
        let config: AgentConfig = toml::from_str(&contents)?;
        Ok(config)
    }

    /// Serialise this config to a TOML string.
    pub fn to_toml(&self) -> Result<String, Box<dyn std::error::Error>> {
        Ok(toml::to_string_pretty(self)?)
    }
}
