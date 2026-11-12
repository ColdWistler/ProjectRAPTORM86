//! Registry of named expert policies for the hybrid phase selector.
//!
//! [`SelectorConfig`](crate::selector::SelectorConfig) maps a mission phase to
//! an expert label. The registry owns the policies those labels refer to, so
//! the selector can dispatch without cloning or rebuilding them.

use crate::policy::Policy;
use std::collections::HashMap;

/// Named expert policies, keyed by label.
#[derive(Default)]
pub struct ExpertRegistry {
    by_label: HashMap<String, Box<dyn Policy>>,
}

impl ExpertRegistry {
    pub fn new() -> Self {
        Self::default()
    }

    /// Insert (or replace) an expert under `label`, returning the previous
    /// policy when the label was already bound.
    pub fn insert(
        &mut self,
        label: impl Into<String>,
        policy: Box<dyn Policy>,
    ) -> Option<Box<dyn Policy>> {
        self.by_label.insert(label.into(), policy)
    }

    /// Borrow an expert for dispatch. Returning `Option<&mut Box<dyn Policy>>`
    /// keeps the selector from cloning the policy on every step.
    pub fn get_mut(&mut self, label: &str) -> Option<&mut Box<dyn Policy>> {
        self.by_label.get_mut(label)
    }

    /// Registered labels, sorted for stable reporting.
    pub fn labels(&self) -> Vec<&str> {
        let mut labels: Vec<&str> = self.by_label.keys().map(String::as_str).collect();
        labels.sort_unstable();
        labels
    }

    pub fn contains(&self, label: &str) -> bool {
        self.by_label.contains_key(label)
    }

    pub fn len(&self) -> usize {
        self.by_label.len()
    }

    pub fn is_empty(&self) -> bool {
        self.by_label.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::contracts::MISSION_OBS_DIM;
    use crate::policy::{ExpertCard, PolicyContext};
    use flight_core::MissionPhase;

    struct FixedPolicy {
        action: Vec<f32>,
        card: ExpertCard,
    }

    impl Policy for FixedPolicy {
        fn act(&mut self, _obs: &[f32], _ctx: &PolicyContext) -> Vec<f32> {
            self.action.clone()
        }

        fn deterministic_action(&mut self, _obs: &[f32], _ctx: &PolicyContext) -> Vec<f32> {
            self.action.clone()
        }

        fn reset(&mut self) {}

        fn card(&self) -> &ExpertCard {
            &self.card
        }
    }

    fn fixed(action: Vec<f32>) -> Box<dyn Policy> {
        Box::new(FixedPolicy {
            action,
            card: ExpertCard::for_training("ppo", MISSION_OBS_DIM, "balanced", "full", 1),
        })
    }

    #[test]
    fn registry_insert_get_and_labels() {
        let mut reg = ExpertRegistry::new();
        assert!(reg.is_empty());
        assert_eq!(reg.len(), 0);

        reg.insert("expert_b", fixed(vec![0.0; 4]));
        reg.insert("expert_a", fixed(vec![1.0; 4]));
        assert_eq!(reg.len(), 2);
        assert_eq!(reg.labels(), vec!["expert_a", "expert_b"]);
        assert!(reg.contains("expert_a"));
        assert!(!reg.contains("missing"));

        let ctx = PolicyContext {
            phase: MissionPhase::Outbound,
            step: 0,
            captured: false,
            features: None,
        };
        let action = reg.get_mut("expert_a").unwrap().act(&[], &ctx);
        assert_eq!(action, vec![1.0; 4]);
        assert!(reg.get_mut("missing").is_none());
    }

    #[test]
    fn insert_replaces_in_place() {
        let mut reg = ExpertRegistry::new();
        assert!(reg.insert("a", fixed(vec![0.0; 4])).is_none());
        assert!(reg.insert("a", fixed(vec![0.5; 4])).is_some());
        assert_eq!(reg.len(), 1);
    }
}
