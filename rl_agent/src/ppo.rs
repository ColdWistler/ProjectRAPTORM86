//! PPO-Clip agent (Schulman et al., *Proximal Policy Optimization
//! Algorithms*, 2017).
//!
//! The agent owns the actor-critic network, an AdamW optimizer (with
//! gradient-norm clipping baked into the optimizer config), and a seeded host
//! RNG used for stochastic action sampling. Training is orchestrated by the
//! trainer module; this file owns the network-adjacent math:
//!
//! - `act` samples a tanh-squashed Gaussian action on the host from the
//!   device-computed mean.
//! - `update` runs the clipped surrogate objective over shuffled minibatches
//!   with per-minibatch advantage whitening, a value loss, and an optional
//!   entropy bonus:
//!
//! ```text
//! L = −E[ min(r·Â, clip(r, 1±ε)·Â) ] + c_v·E[(V−R)²] − c_e·E[H]
//! r = exp(log π_θ(a|s) − log π_old(a|s))
//! ```

use crate::net::{gaussian_log_prob_tanh, ActorCritic, NetConfig, ACTION_CLAMP};
use burn::grad_clipping::GradientClippingConfig;
use burn::optim::adaptor::OptimizerAdaptor;
use burn::optim::{AdamW, AdamWConfig, GradientsParams, Optimizer};
use burn::prelude::*;
use burn::tensor::backend::AutodiffBackend;
use rand::rngs::StdRng;
use rand::seq::SliceRandom;
use rand::SeedableRng;
use rand_distr::Distribution;
use serde::{Deserialize, Serialize};

/// PPO hyper-parameters (serialized into every checkpoint).
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct PpoConfig {
    /// AdamW learning rate.
    pub lr: f32,
    /// Discount factor.
    pub gamma: f32,
    /// GAE trace-decay parameter.
    pub lambda: f32,
    /// PPO ratio clip bound.
    pub clip_epsilon: f32,
    /// Coefficient of the squared value loss.
    pub value_coef: f32,
    /// Coefficient of the policy entropy bonus.
    pub entropy_coef: f32,
    /// Maximum global gradient norm (clipped by the optimizer).
    pub max_grad_norm: f32,
    /// Number of passes over the rollout buffer per update.
    pub epochs: usize,
    /// Number of transitions per gradient step (samples are reshuffled each epoch).
    pub minibatch_size: usize,
}

impl Default for PpoConfig {
    fn default() -> Self {
        Self {
            lr: 3e-4,
            gamma: 0.99,
            lambda: 0.95,
            clip_epsilon: 0.2,
            value_coef: 0.5,
            entropy_coef: 0.0,
            max_grad_norm: 0.5,
            epochs: 10,
            minibatch_size: 64,
        }
    }
}

/// One iteration's worth of data, pre-normalized on the host.
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

/// Averaged loss terms from one `update` pass (helpful for monitoring).
#[derive(Debug, Clone, Copy, Default)]
pub struct UpdateStats {
    /// Mean clipped policy-surrogate loss (negative for an improving policy).
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

/// The PPO agent: network + AdamW + sampling RNG.
pub struct PpoAgent<B: AutodiffBackend> {
    /// Actor-critic network (policy + value).
    pub net: ActorCritic<B>,
    /// Architecture config (kept for checkpoint metadata).
    pub net_cfg: NetConfig,
    /// PPO hyper-parameters.
    pub ppo_cfg: PpoConfig,
    device: B::Device,
    optimizer: OptimizerAdaptor<AdamW, ActorCritic<B>, B>,
    rng: StdRng,
}

impl<B: AutodiffBackend<FloatElem = f32>> PpoAgent<B> {
    /// Build an agent. Seeds the backend (deterministic weight init) and the
    /// sampling RNG.
    pub fn new(device: &B::Device, net_cfg: &NetConfig, ppo_cfg: &PpoConfig, seed: u64) -> Self {
        let net = ActorCritic::new(device, net_cfg, seed);
        Self::from_parts(net, *net_cfg, *ppo_cfg, seed, device)
    }

    /// Rebuild the optimizer/RNG scaffolding around an existing network
    /// (used when loading a checkpoint for evaluation or resume).
    pub fn from_parts(
        net: ActorCritic<B>,
        net_cfg: NetConfig,
        ppo_cfg: PpoConfig,
        seed: u64,
        device: &B::Device,
    ) -> Self {
        let optimizer = AdamWConfig::new()
            .with_grad_clipping(Some(GradientClippingConfig::Norm(ppo_cfg.max_grad_norm)))
            .init::<B, ActorCritic<B>>();
        Self {
            net,
            net_cfg,
            ppo_cfg,
            device: device.clone(),
            optimizer,
            rng: StdRng::seed_from_u64(seed),
        }
    }

    /// Sample one stochastic action for a normalized observation. Returns
    /// `(action, log π(a|s))` in tanh-space.
    pub fn act(&mut self, obs_norm: &[f32]) -> (Vec<f32>, f32) {
        let x = Tensor::<B, 1>::from_floats(obs_norm, &self.device)
            .reshape([1, self.net_cfg.obs_dim]);
        let mu = self.net.mean(x);
        let mu_host: Vec<f32> = mu
            .into_data()
            .to_vec::<f32>()
            .expect("mean must be f32");
        let log_std = self.net.log_std_host();

        let mut action = Vec::with_capacity(self.net_cfg.action_dim);
        let mut logp = 0.0f32;
        for d in 0..self.net_cfg.action_dim {
            let std = log_std[d].exp();
            let dist = rand_distr::Normal::new(mu_host[d], std)
                .expect("policy std must be positive");
            let x_host: f32 = dist.sample(&mut self.rng);
            let a = x_host.tanh().clamp(-ACTION_CLAMP, ACTION_CLAMP);
            action.push(a);
            logp += gaussian_log_prob_tanh(a, mu_host[d], log_std[d]);
        }
        (action, logp)
    }

    /// Greedy deterministic action for a normalized observation.
    pub fn deterministic_action(&self, obs_norm: &[f32]) -> Vec<f32> {
        let x = Tensor::<B, 1>::from_floats(obs_norm, &self.device)
            .reshape([1, self.net_cfg.obs_dim]);
        self.net
            .deterministic_action(x)
            .into_data()
            .to_vec::<f32>()
            .expect("action must be f32")
    }

    /// Value predictions for a flattened batch of normalized observations.
    pub fn value_all(&self, obs_norm_flat: &[f32]) -> Vec<f32> {
        let n = obs_norm_flat.len() / self.net_cfg.obs_dim;
        assert!(n > 0, "value_all needs at least one observation");
        let x = Tensor::<B, 1>::from_floats(obs_norm_flat, &self.device)
            .reshape([n, self.net_cfg.obs_dim]);
        self.net
            .value(x)
            .into_data()
            .to_vec::<f32>()
            .expect("value must be f32")
    }

    /// Value of a single normalized observation.
    pub fn value_single(&self, obs_norm: &[f32]) -> f32 {
        let x = Tensor::<B, 1>::from_floats(obs_norm, &self.device)
            .reshape([1, self.net_cfg.obs_dim]);
        self.net.value(x).into_scalar()
    }

    /// Run `epochs` passes of clipped PPO over the rollout batch. Each
    /// minibatch whitens its advantages, computes the clipped surrogate +
    /// value + entropy losses, and takes one AdamW step.
    pub fn update(&mut self, batch: &UpdateBatch) -> UpdateStats {
        let eps = self.ppo_cfg.clip_epsilon;
        let value_coef = self.ppo_cfg.value_coef;
        let entropy_coef = self.ppo_cfg.entropy_coef;
        let lr = self.ppo_cfg.lr as f64;
        let obs_dim = self.net_cfg.obs_dim;
        let action_dim = self.net_cfg.action_dim;
        let n = batch.old_log_probs.len();
        assert_eq!(batch.obs.len(), n * obs_dim, "obs count mismatch");
        assert_eq!(batch.actions.len(), n * action_dim, "action count mismatch");
        assert_eq!(batch.advantages.len(), n, "advantage count mismatch");
        assert_eq!(batch.returns.len(), n, "returns count mismatch");
        assert!(n > 0, "cannot update from an empty batch");

        let mb_size = self.ppo_cfg.minibatch_size.min(n).max(1);
        let mut indices: Vec<usize> = (0..n).collect();

        let mut accum = UpdateStats::default();
        let mut mb_count = 0usize;

        for _ in 0..self.ppo_cfg.epochs {
            indices.shuffle(&mut self.rng);
            for idx in indices.chunks(mb_size) {
                let m = idx.len();
                let mut mb_obs = Vec::with_capacity(m * obs_dim);
                let mut mb_actions = Vec::with_capacity(m * action_dim);
                let mut mb_old = Vec::with_capacity(m);
                let mut mb_adv = Vec::with_capacity(m);
                let mut mb_ret = Vec::with_capacity(m);
                for &i in idx {
                    mb_obs.extend_from_slice(&batch.obs[i * obs_dim..(i + 1) * obs_dim]);
                    mb_actions.extend_from_slice(&batch.actions[i * action_dim..(i + 1) * action_dim]);
                    mb_old.push(batch.old_log_probs[i]);
                    mb_adv.push(batch.advantages[i]);
                    mb_ret.push(batch.returns[i]);
                }

                // Whitening: per-minibatch mean/std subtraction.
                let adv_mean = mb_adv.iter().sum::<f32>() / m as f32;
                let adv_var =
                    mb_adv.iter().map(|a| (a - adv_mean).powi(2)).sum::<f32>() / m as f32;
                let adv_std = adv_var.sqrt() + 1e-8;

                let obs_t = Tensor::<B, 1>::from_floats(mb_obs.as_slice(), &self.device)
                    .reshape([m, obs_dim]);
                let actions_t = Tensor::<B, 1>::from_floats(mb_actions.as_slice(), &self.device)
                    .reshape([m, action_dim]);
                let old_logp_t = Tensor::<B, 1>::from_floats(mb_old.as_slice(), &self.device);
                let returns_t = Tensor::<B, 1>::from_floats(mb_ret.as_slice(), &self.device);
                let adv_host: Vec<f32> =
                    mb_adv.iter().map(|a| (a - adv_mean) / adv_std).collect();
                let adv_t = Tensor::<B, 1>::from_floats(adv_host.as_slice(), &self.device);

                let logp = self.net.log_prob(obs_t.clone(), actions_t);
                let ratio = (logp - old_logp_t).exp();

                // Clipped surrogate: r_max where Â ≥ 0 else r_min (see module docs).
                let ratio_max = ratio.clone().clamp_max(1.0 + eps);
                let ratio_min = ratio.clone().clamp_min(1.0 - eps);
                let chosen = ratio_min.mask_where(adv_t.clone().greater_equal_elem(0.0), ratio_max);
                let surr = chosen.mul(adv_t);
                let policy_loss = surr.mean().neg();

                let values = self.net.value(obs_t.clone());
                let value_loss = (values - returns_t).powf_scalar(2.0).mean();
                let entropy = self.net.entropy(obs_t).mean();

                let loss = policy_loss.clone()
                    + value_loss.clone().mul_scalar(value_coef)
                    - entropy.clone().mul_scalar(entropy_coef);

                let grads = loss.backward();
                let grads = GradientsParams::from_grads(grads, &self.net);
                self.net = self.optimizer.step(lr, self.net.clone(), grads);

                accum.policy_loss += policy_loss.into_scalar();
                accum.value_loss += value_loss.into_scalar();
                accum.entropy += entropy.into_scalar();
                mb_count += 1;
            }
        }

        let (adv_mean, adv_std) = {
            let mean = batch.advantages.iter().sum::<f32>() / n as f32;
            let var = batch.advantages.iter().map(|a| (a - mean).powi(2)).sum::<f32>() / n as f32;
            (mean, var.sqrt())
        };
        accum.policy_loss /= mb_count as f32;
        accum.value_loss /= mb_count as f32;
        accum.entropy /= mb_count as f32;
        accum.advantage_mean = adv_mean;
        accum.advantage_std = adv_std;
        accum
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::Cpu;

    fn device() -> burn::backend::ndarray::NdArrayDevice {
        crate::backend::cpu_device()
    }

    fn agent() -> PpoAgent<Cpu> {
        let cfg = PpoConfig {
            lr: 3e-3,
            gamma: 0.99,
            lambda: 0.95,
            clip_epsilon: 0.2,
            value_coef: 0.5,
            entropy_coef: 0.01,
            max_grad_norm: 10.0,
            epochs: 2,
            minibatch_size: 8,
        };
        let net_cfg = NetConfig {
            obs_dim: 4,
            hidden: 16,
            action_dim: 2,
            log_std_init: -0.5,
        };
        PpoAgent::new(&device(), &net_cfg, &cfg, 7)
    }

    #[test]
    fn act_is_deterministic_under_same_seed() {
        let mut a = agent();
        let mut b = agent();
        let obs = vec![0.1f32, -0.2, 0.3, 0.4];
        let (act_a, lp_a) = a.act(&obs);
        let (act_b, lp_b) = b.act(&obs);
        assert_eq!(act_a, act_b, "same seed must reproduce sampled actions");
        assert_eq!(lp_a, lp_b, "same seed must reproduce log probs");
        assert!(lp_a.is_finite());
        for &x in &act_a {
            assert!(x.abs() < 1.0, "tanh action out of range: {x}");
        }
    }

    #[test]
    fn act_changes_with_obs() {
        let mut a = agent();
        let (a1, _) = a.act(&[0.0f32; 4]);
        let (a2, _) = a.act(&[0.7f32, -0.3, 0.2, -0.9]);
        assert!(a1 != a2, "different observations should change the action");
    }

    #[test]
    fn update_smoke_finite_and_changes_policy() {
        let mut ag = agent();
        let n = 24;
        let obs_dim = 4;
        let action_dim = 2;
        let mut batch = UpdateBatch {
            obs: vec![0.0; n * obs_dim],
            actions: vec![0.0; n * action_dim],
            old_log_probs: vec![0.0; n],
            advantages: vec![0.0; n],
            returns: vec![0.0; n],
        };
        for i in 0..n {
            for d in 0..obs_dim {
                batch.obs[i * obs_dim + d] = (i as f32) * 0.1 + d as f32;
            }
            for d in 0..action_dim {
                batch.actions[i * action_dim + d] = ((i + d) as f32).sin() * 0.5;
            }
            batch.old_log_probs[i] = -1.0 - 0.01 * i as f32;
            batch.advantages[i] = 1.0 - (i as f32) / n as f32;
            batch.returns[i] = 0.5;
        }
        let mut norm = crate::normalize::RunningNormalizer::new(obs_dim);
        for i in 0..n {
            norm.observe(
                &batch.obs[i * obs_dim..(i + 1) * obs_dim]
                    .iter()
                    .map(|&x| x as f64)
                    .collect::<Vec<f64>>(),
            );
        }
        let normalized = norm.normalize_batch(&batch.obs);
        let before = ag.value_all(&normalized);
        let stats = ag.update(&batch);
        assert!(stats.policy_loss.is_finite());
        assert!(stats.value_loss.is_finite());
        assert!(stats.entropy.is_finite());
        assert!(stats.advantage_mean.is_finite());
        assert!(stats.advantage_std.is_finite());
        assert!(stats.advantage_std > 0.0);

        let after = ag.value_all(&normalized);
        assert_ne!(before, after, "optimizer step must change the value function");
    }

    #[test]
    fn deterministic_action_matches_mean_tanh() {
        let ag = agent();
        let obs = vec![0.2f32, -0.1, 0.4, 0.05];
        let act = ag.deterministic_action(&obs);
        assert_eq!(act.len(), 2);
        assert!(act.iter().all(|a| a.abs() < 1.0));
    }
}