//! Advantage Actor-Critic (A2C) agent — simpler on-policy actor-critic.
//!
//! No clipping / importance ratio; policy loss uses the host-computed GAE
//! advantages passed in [`UpdateBatch`], value loss targets [`returns`].
//! Single-epoch update over the full rollout batch (minibatched if large).

use crate::algo::{Algorithm, AlgoSpec, UpdateBatch, UpdateStats, ALGO_A2C};
use crate::net::{ActorCritic, NetConfig, ACTION_CLAMP};
use burn::grad_clipping::GradientClippingConfig;
use burn::optim::adaptor::OptimizerAdaptor;
use burn::optim::{AdamW, AdamWConfig, GradientsParams, Optimizer};
use burn::prelude::*;
use burn::record::{BinFileRecorder, FullPrecisionSettings};
use burn::tensor::backend::AutodiffBackend;
use rand::rngs::StdRng;
use rand::SeedableRng;
use rand_distr::Distribution;
use serde::{Deserialize, Serialize};
use std::path::Path;

/// A2C hyper-parameters (serialized into every checkpoint).
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq)]
pub struct A2cConfig {
    pub lr: f32,
    pub gamma: f32,
    pub value_coef: f32,
    pub entropy_coef: f32,
    pub max_grad_norm: f32,
    pub minibatch_size: usize,
    pub epochs: usize,
}

impl Default for A2cConfig {
    fn default() -> Self {
        Self {
            lr: 3e-4,
            gamma: 0.99,
            value_coef: 0.5,
            entropy_coef: 0.01,
            max_grad_norm: 0.5,
            minibatch_size: 64,
            epochs: 4,
        }
    }
}

pub struct A2cAgent<B: AutodiffBackend> {
    pub net: ActorCritic<B>,
    pub net_cfg: NetConfig,
    pub cfg: A2cConfig,
    device: B::Device,
    optimizer: OptimizerAdaptor<AdamW, ActorCritic<B>, B>,
    rng: StdRng,
}

impl<B: AutodiffBackend<FloatElem = f32>> A2cAgent<B> {
    pub fn new(device: &B::Device, net_cfg: &NetConfig, cfg: &A2cConfig, seed: u64) -> Self {
        let net = ActorCritic::new(device, net_cfg, seed);
        Self::from_parts(net, *net_cfg, *cfg, seed, device)
    }

    pub fn from_parts(
        net: ActorCritic<B>,
        net_cfg: NetConfig,
        cfg: A2cConfig,
        seed: u64,
        device: &B::Device,
    ) -> Self {
        let optimizer = AdamWConfig::new()
            .with_grad_clipping(Some(GradientClippingConfig::Norm(cfg.max_grad_norm)))
            .init::<B, ActorCritic<B>>();
        Self {
            net,
            net_cfg,
            cfg,
            device: device.clone(),
            optimizer,
            rng: StdRng::seed_from_u64(seed),
        }
    }

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
            logp += crate::net::gaussian_log_prob_tanh(a, mu_host[d], log_std[d]);
        }
        (action, logp)
    }

    pub fn deterministic_action(&self, obs_norm: &[f32]) -> Vec<f32> {
        let x = Tensor::<B, 1>::from_floats(obs_norm, &self.device)
            .reshape([1, self.net_cfg.obs_dim]);
        self.net.deterministic_action(x).into_data()
            .to_vec::<f32>().expect("action must be f32")
    }

    pub fn value_all(&self, obs_norm_flat: &[f32]) -> Vec<f32> {
        let n = obs_norm_flat.len() / self.net_cfg.obs_dim;
        assert!(n > 0, "value_all needs at least one observation");
        let x = Tensor::<B, 1>::from_floats(obs_norm_flat, &self.device)
            .reshape([n, self.net_cfg.obs_dim]);
        self.net.value(x).into_data().to_vec::<f32>().expect("value must be f32")
    }

    pub fn value_single(&self, obs_norm: &[f32]) -> f32 {
        let x = Tensor::<B, 1>::from_floats(obs_norm, &self.device)
            .reshape([1, self.net_cfg.obs_dim]);
        self.net.value(x).into_scalar()
    }

    pub fn update(&mut self, batch: &UpdateBatch) -> UpdateStats {
        let obs_dim = self.net_cfg.obs_dim;
        let action_dim = self.net_cfg.action_dim;
        let n = batch.advantages.len();
        assert_eq!(batch.obs.len(), n * obs_dim, "obs count mismatch");
        assert_eq!(batch.actions.len(), n * action_dim, "action count mismatch");
        assert_eq!(batch.advantages.len(), n, "advantage count mismatch");
        assert_eq!(batch.returns.len(), n, "returns count mismatch");
        assert!(n > 0, "cannot update from an empty batch");

        let mb_size = self.cfg.minibatch_size.min(n).max(1);
        let lr = self.cfg.lr as f64;
        let value_coef = self.cfg.value_coef;
        let entropy_coef = self.cfg.entropy_coef;

        let mut accum = UpdateStats::default();
        let mut mb_count = 0usize;

        for _ in 0..self.cfg.epochs {
            let mut indices: Vec<usize> = (0..n).collect();
            use rand::seq::SliceRandom;
            indices.shuffle(&mut self.rng);
            for idx in indices.chunks(mb_size) {
                let m = idx.len();
                let mut mb_obs = Vec::with_capacity(m * obs_dim);
                let mut mb_actions = Vec::with_capacity(m * action_dim);
                let mut mb_adv = Vec::with_capacity(m);
                let mut mb_ret = Vec::with_capacity(m);
                for &i in idx {
                    mb_obs.extend_from_slice(&batch.obs[i * obs_dim..(i + 1) * obs_dim]);
                    mb_actions.extend_from_slice(&batch.actions[i * action_dim..(i + 1) * action_dim]);
                    mb_adv.push(batch.advantages[i]);
                    mb_ret.push(batch.returns[i]);
                }

                let obs_t = Tensor::<B, 1>::from_floats(mb_obs.as_slice(), &self.device)
                    .reshape([m, obs_dim]);
                let actions_t = Tensor::<B, 1>::from_floats(mb_actions.as_slice(), &self.device)
                    .reshape([m, action_dim]);

                // Policy gradient: maximize log_prob * adv
                let logp = self.net.log_prob(obs_t.clone(), actions_t.clone());
                let adv_t = Tensor::<B, 1>::from_floats(mb_adv.as_slice(), &self.device);
                let policy_loss = logp.mul(adv_t.clone()).mean().neg();

                let values = self.net.value(obs_t.clone());
                let returns_t = Tensor::<B, 1>::from_floats(mb_ret.as_slice(), &self.device);
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

impl<B: AutodiffBackend<FloatElem = f32>> Algorithm<B> for A2cAgent<B> {
    fn id(&self) -> &'static str {
        ALGO_A2C
    }
    fn net_config(&self) -> &NetConfig {
        &self.net_cfg
    }
    fn spec(&self) -> AlgoSpec {
        AlgoSpec::A2c { config: self.cfg }
    }
    fn td_params(&self) -> (f32, f32) {
        (self.cfg.gamma, 0.95)
    }
    fn act(&mut self, obs_norm: &[f32]) -> (Vec<f32>, f32) {
        A2cAgent::act(self, obs_norm)
    }
    fn deterministic_action(&self, obs_norm: &[f32]) -> Vec<f32> {
        A2cAgent::deterministic_action(self, obs_norm)
    }
    fn value_single(&self, obs_norm: &[f32]) -> f32 {
        A2cAgent::value_single(self, obs_norm)
    }
    fn value_all(&self, obs_norm_flat: &[f32]) -> Vec<f32> {
        A2cAgent::value_all(self, obs_norm_flat)
    }
    fn update(&mut self, batch: &UpdateBatch) -> UpdateStats {
        A2cAgent::update(self, batch)
    }
    fn save_weights(&self, dir: &Path) -> Result<(), Box<dyn std::error::Error>> {
        let recorder = BinFileRecorder::<FullPrecisionSettings>::default();
        self.net.clone().save_file(dir.join("net"), &recorder)?;
        Ok(())
    }
    fn load_weights(&mut self, dir: &Path, device: &B::Device) -> Result<(), Box<dyn std::error::Error>> {
        let recorder = BinFileRecorder::<FullPrecisionSettings>::default();
        self.net = self.net.clone().load_file(dir.join("net"), &recorder, device)?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::Cpu;

    fn device() -> burn::backend::ndarray::NdArrayDevice {
        crate::backend::cpu_device()
    }

    fn agent() -> A2cAgent<Cpu> {
        let cfg = A2cConfig::default();
        let net_cfg = NetConfig {
            obs_dim: 4,
            hidden: 16,
            action_dim: 2,
            log_std_init: -0.5,
        };
        A2cAgent::new(&device(), &net_cfg, &cfg, 7)
    }

    #[test]
    fn act_bounded() {
        let mut a = agent();
        let (act, lp) = a.act(&[0.1f32, -0.2, 0.3, 0.4]);
        assert_eq!(act.len(), 2);
        for &x in &act {
            assert!(x.abs() < 1.0, "tanh action out of range: {x}");
        }
        assert!(lp.is_finite());
    }

    #[test]
    fn update_smoke() {
        let mut ag = agent();
        let n = 24;
        let obs_dim = 4;
        let action_dim = 2;
        let mut batch = UpdateBatch {
            obs: vec![0.0; n * obs_dim],
            actions: vec![0.0; n * action_dim],
            old_log_probs: vec![0.0; n],
            advantages: vec![0.0; n],
            returns: vec![0.5; n],
        };
        for i in 0..n {
            for d in 0..obs_dim {
                batch.obs[i * obs_dim + d] = (i as f32) * 0.05 + d as f32;
            }
            for d in 0..action_dim {
                batch.actions[i * action_dim + d] = ((i + d) as f32).sin() * 0.3;
            }
            batch.advantages[i] = 1.0 - (i as f32) / n as f32;
        }
        let stats = ag.update(&batch);
        assert!(stats.policy_loss.is_finite());
        assert!(stats.value_loss.is_finite());
        assert!(stats.entropy.is_finite());
    }
}
