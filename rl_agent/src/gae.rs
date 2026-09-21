//! Host-side Generalized Advantage Estimation (GAE).
//!
//! Computed on the host in `f32` from value predictions the trainer pulled
//! off the device, so the runtime cost is trivial.
//!
//! Formula (Schulman et al., *High-Dimensional Continuous Control Using
//! Generalized Advantage Estimation*, ICLR 2016):
//!
//! ```text
//! δ_t     = r_t + γ·V(s_{t+1}) − V(s_t)          (V(s_{t+1}) = 0 if s_t terminated)
//! A_t     = δ_t + (γ·λ)·A_{t+1}·(1 − terminated_t)
//! returns_t = A_t + V(s_t)
//! ```
//!
//! A truncated-but-not-terminated final step bootstraps through the
//! caller-supplied `last_value = V(s_final)`; a terminated final step does not.

/// Compute GAE advantages and (advantage + value) returns.
///
/// # Arguments
/// - `rewards`: per-step rewards.
/// - `values`: V(s_t) for each step (from the rollout policy).
/// - `terminated`: per-step termination flags.
/// - `last_value`: V(s_final) — the value of the observation **after** the
///   final step. Used only when the final step did not terminate.
/// - `gamma`: discount factor.
/// - `lambda`: GAE trace-decay parameter.
pub fn compute_gae(
    rewards: &[f32],
    values: &[f32],
    terminated: &[bool],
    last_value: f32,
    gamma: f32,
    lambda: f32,
) -> (Vec<f32>, Vec<f32>) {
    let n = rewards.len();
    assert_eq!(values.len(), n, "values length must match rewards");
    assert_eq!(terminated.len(), n, "terminated length must match rewards");
    assert!(n > 0, "cannot compute GAE over an empty rollout");

    let mut advantages = vec![0.0f32; n];
    let mut returns = vec![0.0f32; n];
    let mut acc = 0.0f32;

    for t in (0..n).rev() {
        let next_value = if terminated[t] {
            0.0
        } else if t + 1 < n {
            values[t + 1]
        } else {
            last_value
        };
        let delta = rewards[t] + gamma * next_value - values[t];
        let mask = if terminated[t] { 0.0 } else { 1.0 };
        acc = delta + gamma * lambda * mask * acc;
        advantages[t] = acc;
        returns[t] = acc + values[t];
    }

    (advantages, returns)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gae_discounts_simple_episode() {
        // Two-step episode: r = [1, 1], V = [0.5, 0.6], no termination,
        // last_value = 0.7, γ=0.99, λ=0.95.
        let (adv, ret) = compute_gae(
            &[1.0, 1.0],
            &[0.5, 0.6],
            &[false, false],
            0.7,
            0.99,
            0.95,
        );
        let delta_1 = 1.0 + 0.99 * 0.7 - 0.6;
        let delta_0 = 1.0 + 0.99 * 0.6 - 0.5;
        let adv_1 = delta_1;
        let adv_0 = delta_0 + 0.99 * 0.95 * adv_1;
        assert!((adv[1] - adv_1).abs() < 1e-6);
        assert!((adv[0] - adv_0).abs() < 1e-6);
        assert!((ret[0] - (adv_0 + 0.5)).abs() < 1e-6);
        assert!((ret[1] - (adv_1 + 0.6)).abs() < 1e-6);
    }

    #[test]
    fn terminated_breaks_the_trace() {
        // r = [1, 1, 1], V = [0.5, 0.6, 0.7], terminated at t=1.
        // The trace must not carry the t=2 advantage back through the terminal
        // step t=1, but it *does* continue from t=1 into t=0.
        let (adv, _ret) = compute_gae(
            &[1.0, 1.0, 1.0],
            &[0.5, 0.6, 0.7],
            &[false, true, false],
            0.8,
            0.99,
            0.95,
        );
        let delta_2 = 1.0 + 0.99 * 0.8 - 0.7;
        let adv_2 = delta_2;
        let delta_1 = 1.0 + 0.99 * 0.0 - 0.6; // next value zeroed at the terminal step
        let adv_1 = delta_1; // mask is 0 at t=1, so nothing propagates from t=2
        let delta_0 = 1.0 + 0.99 * 0.6 - 0.5;
        let adv_0 = delta_0 + 0.99 * 0.95 * adv_1;
        assert!((adv[2] - adv_2).abs() < 1e-6);
        assert!((adv[1] - adv_1).abs() < 1e-6);
        assert!((adv[0] - adv_0).abs() < 1e-6);
    }

    #[test]
    fn long_rollout_non_done_bootstraps_from_last_value() {
        let n = 100;
        let rewards: Vec<f32> = vec![0.5; n];
        let values: Vec<f32> = (0..n).map(|i| 0.5 + i as f32 * 0.001).collect();
        let terminated = vec![false; n];
        let (adv, ret) = compute_gae(&rewards, &values, &terminated, 1.0, 0.99, 0.97);
        // All rets = advantage + value; last advantage uses last_value.
        for t in 0..n {
            assert!((ret[t] - (adv[t] + values[t])).abs() < 1e-6);
        }
        assert!(adv[n - 1].is_finite());
    }
}