//! Rollout storage.
//!
//! The trainer collects one transition per environment step into a
//! [`RolloutBuffer`]. Observations are kept **raw** (un-normalized `f32`) —
//! normalization happens lazily at update time once all episode stats are
//! final — while actions and per-step log-densities are stored as computed
//! during sampling so PPO's importance ratio can compare old vs new policies.

/// One environment step: the state before the step, the action taken, the
/// log-density of that action under the sampling policy, and the outcome.
#[derive(Debug, Clone)]
pub struct Transition {
    /// Raw observation (normalized later), `obs_dim` floats.
    pub obs: Vec<f32>,
    /// Continuous action in tanh-space, `action_dim` floats.
    pub action: Vec<f32>,
    /// `log π_old(action | obs)` from the sampling policy.
    pub log_prob: f32,
    /// Scalar reward for this step.
    pub reward: f32,
    /// `true` if the episode ended by termination (crash / weather).
    pub terminated: bool,
    /// `true` if the episode ended because the step budget ran out.
    pub truncated: bool,
}

/// Flat arrays backing an iteration of rollouts.
#[derive(Debug, Default)]
pub struct RolloutBuffer {
    transitions: Vec<Transition>,
    obs_dim: usize,
    action_dim: usize,
}

impl RolloutBuffer {
    /// New empty buffer for the given observation / action dimensions.
    pub fn new(obs_dim: usize, action_dim: usize) -> Self {
        Self {
            transitions: Vec::new(),
            obs_dim,
            action_dim,
        }
    }

    /// Append one transition.
    pub fn push(&mut self, t: Transition) {
        debug_assert_eq!(t.obs.len(), self.obs_dim);
        debug_assert_eq!(t.action.len(), self.action_dim);
        self.transitions.push(t);
    }

    /// Number of stored transitions (all are kept until [`Self::clear`]).
    pub fn len(&self) -> usize {
        self.transitions.len()
    }

    /// Whether the buffer is empty.
    pub fn is_empty(&self) -> bool {
        self.transitions.is_empty()
    }

    /// Observation dimension.
    pub fn obs_dim(&self) -> usize {
        self.obs_dim
    }

    /// Action dimension.
    pub fn action_dim(&self) -> usize {
        self.action_dim
    }

    /// Reference to the last transition, if any.
    pub fn last(&self) -> Option<&Transition> {
        self.transitions.last()
    }

    /// Drop all collected rollouts (start a new iteration).
    pub fn clear(&mut self) {
        self.transitions.clear();
    }

    /// Flattened `n·obs_dim` raw observations.
    pub fn obs_flat(&self) -> Vec<f32> {
        let mut out = Vec::with_capacity(self.transitions.len() * self.obs_dim);
        for t in &self.transitions {
            out.extend_from_slice(&t.obs);
        }
        out
    }

    /// Flattened `n·action_dim` actions.
    pub fn actions_flat(&self) -> Vec<f32> {
        let mut out = Vec::with_capacity(self.transitions.len() * self.action_dim);
        for t in &self.transitions {
            out.extend_from_slice(&t.action);
        }
        out
    }

    /// Per-step old log-densities.
    pub fn log_probs(&self) -> Vec<f32> {
        self.transitions.iter().map(|t| t.log_prob).collect()
    }

    /// Per-step rewards.
    pub fn rewards(&self) -> Vec<f32> {
        self.transitions.iter().map(|t| t.reward).collect()
    }

    /// Per-step termination flags.
    pub fn terminated(&self) -> Vec<bool> {
        self.transitions.iter().map(|t| t.terminated).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn buffer_flat_arrays_line_up() {
        let mut buf = RolloutBuffer::new(3, 2);
        for i in 0..4 {
            buf.push(Transition {
                obs: vec![i as f32, i as f32 + 1.0, i as f32 + 2.0],
                action: vec![i as f32, -i as f32],
                log_prob: i as f32 * 0.1,
                reward: 1.0 - i as f32 * 0.25,
                terminated: i == 3,
                truncated: false,
            });
        }
        assert_eq!(buf.len(), 4);
        assert_eq!(buf.obs_flat(), vec![0.0, 1.0, 2.0, 1.0, 2.0, 3.0, 2.0, 3.0, 4.0, 3.0, 4.0, 5.0]);
        assert_eq!(buf.actions_flat(), vec![0.0, 0.0, 1.0, -1.0, 2.0, -2.0, 3.0, -3.0]);
        assert_eq!(buf.log_probs(), vec![0.0, 0.1, 0.2, 0.3]);
        assert_eq!(buf.rewards(), vec![1.0, 0.75, 0.5, 0.25]);
        assert_eq!(buf.terminated(), vec![false, false, false, true]);
        assert!(buf.last().unwrap().terminated);

        buf.clear();
        assert!(buf.is_empty());
    }
}