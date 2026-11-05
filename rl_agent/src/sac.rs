//! Soft Actor-Critic (SAC) — off-policy twin-critic with entropy bonus.
//! Continuous tanh 4-D actions, uniform replay, Polyak target averaging.
use crate::algo::{Algorithm, AlgoSpec, UpdateBatch, UpdateStats, ALGO_SAC};
use crate::buffer::Transition;
use crate::net::{polyak_linear, NetConfig, LN_2PI};
use crate::normalize::RunningNormalizer;
use burn::grad_clipping::GradientClippingConfig;
use burn::nn::{Linear, LinearConfig};
use burn::optim::{AdamW, AdamWConfig, GradientsParams, Optimizer};
use burn::prelude::*;
use rand::Rng;
use rand::SeedableRng;
use rand::rngs::StdRng;
use rand_distr::Distribution;
use serde::{Deserialize, Serialize};
use std::path::Path;
use burn::record::{BinFileRecorder, FullPrecisionSettings};
use burn::tensor::backend::AutodiffBackend;

pub const ACTION_CLAMP: f32 = 0.999_999;

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq)]
pub struct SacConfig {
    pub gamma: f32,
    pub tau: f32,
    pub hidden: usize,
    pub buffer_capacity: usize,
    pub batch_size: usize,
    pub max_grad_norm: f32,
    pub lr: f32,
    pub alpha: f32, // entropy temperature (fixed)
}

impl Default for SacConfig {
    fn default() -> Self {
        Self {
            gamma: 0.99,
            tau: 0.005,
            hidden: 256,
            buffer_capacity: 8192,
            batch_size: 64,
            max_grad_norm: 0.5,
            lr: 3e-4,
            alpha: 0.2,
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ReplayItem {
    obs: Vec<f32>,
    action: Vec<f32>,
    reward: f32,
    terminated: bool,
    truncated: bool,
    next_obs: Vec<f32>,
}

/// Uniform replay (no prioritization).
pub struct UniformReplay {
    capacity: usize,
    count: usize,
    items: Vec<Option<ReplayItem>>,
}

impl UniformReplay {
    fn new(cap: usize) -> Self {
        Self { capacity: cap, count: 0, items: vec![None; cap] }
    }
    fn len(&self) -> usize { self.items.iter().filter(|o| o.is_some()).count() }
    fn push(&mut self, item: ReplayItem) {
        let idx = self.count % self.capacity;
        self.items[idx] = Some(item);
        self.count += 1;
    }
    fn sample(&mut self, n: usize, rng: &mut StdRng) -> Vec<ReplayItem> {
        let filled: Vec<usize> = self.items.iter().enumerate()
            .filter(|(_, v)| v.is_some())
            .map(|(i, _)| i)
            .collect();
        if filled.is_empty() { return Vec::new(); }
        let n = n.min(filled.len());
        let mut out = Vec::with_capacity(n);
        let mut chosen = filled.clone();
        for _ in 0..n {
            let idx = rng.gen_range(0..chosen.len());
            let item_idx = chosen.remove(idx);
            if let Some(ref item) = self.items[item_idx % self.capacity] {
                out.push(item.clone());
            }
        }
        out
    }
}

#[derive(Module, Debug)]
pub struct ActorNet<B: Backend> {
    trunk: Linear<B>,
    mean_head: Linear<B>,
    log_std_head: Linear<B>,
    device: B::Device,
    obs_dim: usize,
    act_dim: usize,
}

impl<B: Backend> ActorNet<B> {
    pub fn new(device: &B::Device, obs_dim: usize, act_dim: usize, hidden: usize) -> Self {
        let trunk = LinearConfig::new(obs_dim, hidden).with_bias(true).init(device);
        let mean_head = LinearConfig::new(hidden, act_dim).with_bias(true).init(device);
        let log_std_head = LinearConfig::new(hidden, act_dim).with_bias(true).init(device);
        Self { trunk, mean_head, log_std_head, device: device.clone(), obs_dim, act_dim }
    }

    pub fn forward(&self, obs: Tensor<B, 2>) -> (Tensor<B, 2>, Tensor<B, 2>) {
        let h = self.trunk.forward(obs);
        let h = h.tanh();
        let mean = self.mean_head.forward(h.clone()).tanh();
        let log_std = self.log_std_head.forward(h).tanh().clamp(-20.0, 2.0);
        (mean, log_std)
    }

    /// Reparameterized tanh-Gaussian sample.
    ///
    /// `eps` is a host-supplied standard-normal sample that is **constant**
    /// with respect to the parameters, so `a = tanh(μ + σ·ε)` stays
    /// differentiable w.r.t. `μ` and `log_std`. That is what makes SAC's
    /// actor loss trainable: a host round-trip (`into_data`) would sever the
    /// autodiff graph and freeze the actor forever.
    ///
    /// Returns `(action, log π(a|s), H(π))` where `H` is the differential
    /// entropy of the *squashed* policy.
    pub fn sample(
        &self,
        obs: Tensor<B, 2>,
        eps: Tensor<B, 2>,
    ) -> (Tensor<B, 2>, Tensor<B, 1>, Tensor<B, 1>) {
        let (mean, log_std) = self.forward(obs);
        let std = log_std.clone().exp();
        let pre = mean.clone().add(std.mul(eps));
        let a = pre.clone().tanh();

        // log N(pre; μ, σ) summed over action dimensions.
        let diff = pre.clone().sub(mean);
        let variance = log_std.clone().mul_scalar(2.0).exp().add_scalar(1e-6);
        let gauss_logp = diff
            .powf_scalar(2.0)
            .mul_scalar(-0.5)
            .div(variance)
            .sub(log_std.clone())
            .sub_scalar(0.5 * LN_2PI)
            .sum_dim(1)
            .squeeze_dim::<1>(1);

        // tanh change of variables: subtract log |da/du| = log(1 - a²).
        let one = Tensor::ones_like(&a);
        let log_det = one
            .clone()
            .sub(a.clone().powf_scalar(2.0))
            .clamp(1e-6, 1.0)
            .log()
            .sum_dim(1)
            .squeeze_dim::<1>(1);
        let logp = gauss_logp.clone().sub(log_det.clone());

        // Differential entropy of the squashed Gaussian:
        // H(u) - E[log |da/du|].
        let gauss_entropy = log_std
            .add_scalar(0.5 * (LN_2PI + 1.0))
            .sum_dim(1)
            .squeeze_dim::<1>(1);
        let entropy = gauss_entropy.sub(log_det);
        (a, logp, entropy)
    }
}

/// Polyak-average a critic network towards the online critic.
fn polyak_critic<B: Backend>(
    target: &CriticNet<B>,
    online: &CriticNet<B>,
    tau: f32,
) -> CriticNet<B> {
    CriticNet {
        trunk: polyak_linear(&target.trunk, &online.trunk, tau),
        fc2: polyak_linear(&target.fc2, &online.fc2, tau),
        q_head: polyak_linear(&target.q_head, &online.q_head, tau),
        device: target.device.clone(),
        obs_dim: target.obs_dim,
        act_dim: target.act_dim,
    }
}

#[derive(Module, Debug)]
pub struct CriticNet<B: Backend> {
    trunk: Linear<B>,
    fc2: Linear<B>,
    q_head: Linear<B>,
    device: B::Device,
    obs_dim: usize,
    act_dim: usize,
}

impl<B: Backend> CriticNet<B> {
    pub fn new(device: &B::Device, obs_dim: usize, act_dim: usize, hidden: usize) -> Self {
        let trunk = LinearConfig::new(obs_dim + act_dim, hidden).with_bias(true).init(device);
        let fc2 = LinearConfig::new(hidden, hidden).with_bias(true).init(device);
        let q_head = LinearConfig::new(hidden, 1).with_bias(true).init(device);
        Self { trunk, fc2, q_head, device: device.clone(), obs_dim, act_dim }
    }

    pub fn forward(&self, obs_act: Tensor<B, 2>) -> Tensor<B, 1> {
        let h = self.trunk.forward(obs_act);
        let h = h.tanh();
        let h = self.fc2.forward(h);
        let h = h.tanh();
        let qv = self.q_head.forward(h);
        qv.squeeze_dim::<1>(1)
    }
}

pub struct SacAgent<B: AutodiffBackend<FloatElem = f32>> {
    pub actor: ActorNet<B>,
    pub q1: CriticNet<B>,
    pub q2: CriticNet<B>,
    pub q1_target: CriticNet<B>,
    pub q2_target: CriticNet<B>,
    pub cfg: SacConfig,
    pub device: B::Device,
    pub actor_opt: burn::optim::adaptor::OptimizerAdaptor<AdamW, ActorNet<B>, B>,
    pub critic_opt: burn::optim::adaptor::OptimizerAdaptor<AdamW, CriticNet<B>, B>,
    pub replay: UniformReplay,
    pub rng: StdRng,
    pub net_cfg: NetConfig,
    /// Trainer-owned observation statistics. The replay buffer stores **raw**
    /// observations; they are scaled with these statistics at update time so
    /// that `obs` and `next_obs` of one transition always use the same
    /// scaling (otherwise the Bellman target is inconsistent).
    pub normalizer: RunningNormalizer,
}

impl<B: AutodiffBackend<FloatElem = f32>> SacAgent<B> {
    pub fn new(device: &B::Device, net_cfg: &NetConfig, cfg: &SacConfig, seed: u64) -> Self {
        // Seed the backend and materialize every parameter under one lock:
        // burn's ndarray backend shares one process-wide RNG and initializes
        // `Linear` weights lazily, so an unlocked build is neither
        // reproducible nor safe to clone.
        let (actor, q1, q2) = {
            let _guard = crate::net::init_lock();
            B::seed(device, seed);
            let actor = ActorNet::new(device, net_cfg.obs_dim, net_cfg.action_dim, cfg.hidden);
            let q1 = CriticNet::new(device, net_cfg.obs_dim, net_cfg.action_dim, cfg.hidden);
            let q2 = CriticNet::new(device, net_cfg.obs_dim, net_cfg.action_dim, cfg.hidden);
            crate::net::materialize(&actor);
            crate::net::materialize(&q1);
            crate::net::materialize(&q2);
            (actor, q1, q2)
        };
        // The target critics start as an **exact copy** of the online ones.
        // `polyak_*` with tau = 1 writes the online values into fresh
        // parameters (new ids, same numbers).
        let q1_target = polyak_critic(&q1, &q1, 1.0);
        let q2_target = polyak_critic(&q2, &q2, 1.0);
        let actor_opt = AdamWConfig::new()
            .with_grad_clipping(Some(GradientClippingConfig::Norm(cfg.max_grad_norm)))
            .init::<B, ActorNet<B>>();
        let critic_opt = AdamWConfig::new()
            .with_grad_clipping(Some(GradientClippingConfig::Norm(cfg.max_grad_norm)))
            .init::<B, CriticNet<B>>();
        Self {
            actor,
            q1,
            q2,
            q1_target,
            q2_target,
            cfg: *cfg,
            device: device.clone(),
            actor_opt,
            critic_opt,
            replay: UniformReplay::new(cfg.buffer_capacity),
            rng: StdRng::seed_from_u64(seed),
            net_cfg: *net_cfg,
            normalizer: RunningNormalizer::new(net_cfg.obs_dim),
        }
    }

    fn sample_action(&mut self, obs_norm: &[f32]) -> (Vec<f32>, f32) {
        let obs_t = Tensor::<B, 1>::from_floats(obs_norm, &self.device).reshape([1, self.net_cfg.obs_dim]);
        let (mean, log_std) = self.actor.forward(obs_t.clone());
        let mean_v = mean.into_data().to_vec::<f32>().expect("mean f32");
        let std_v = log_std.into_data().to_vec::<f32>().expect("std f32").iter().map(|s| s.exp()).collect::<Vec<_>>();
        let mut action = Vec::with_capacity(self.net_cfg.action_dim);
        let mut log_prob = 0.0f32;
        for d in 0..self.net_cfg.action_dim {
            let noise: f32 = rand_distr::StandardNormal.sample(&mut self.rng);
            let u = mean_v[d] + std_v[d] * noise as f32;
            let a = u.tanh();
            action.push(a.clamp(-ACTION_CLAMP, ACTION_CLAMP));
            // log prob correction for tanh squash
            let logp = -0.5 * (noise as f32).powi(2) - std_v[d].ln() - 0.5 * (2.0f32 * std::f32::consts::PI).ln();
            let correction = (1.0 - a * a).ln();
            log_prob += logp + correction;
        }
        (action, log_prob)
    }

    pub fn act(&mut self, obs_norm: &[f32]) -> (Vec<f32>, f32) {
        self.sample_action(obs_norm)
    }

    pub fn deterministic_action(&self, obs_norm: &[f32]) -> Vec<f32> {
        let obs_t = Tensor::<B, 1>::from_floats(obs_norm, &self.device).reshape([1, self.net_cfg.obs_dim]);
        let (mean, _log_std) = self.actor.forward(obs_t);
        let v: Vec<f32> = mean.into_data().to_vec::<f32>().expect("mean f32");
        v.into_iter().map(|a| a.tanh().clamp(-ACTION_CLAMP, ACTION_CLAMP)).collect()
    }

    pub fn value_single(&self, obs_norm: &[f32]) -> f32 {
        let obs = obs_norm.to_vec();
        let zeros = vec![0.0f32; self.net_cfg.action_dim];
        let mut flat = Vec::with_capacity(self.net_cfg.obs_dim + self.net_cfg.action_dim);
        flat.extend_from_slice(&obs);
        flat.extend_from_slice(&zeros);
        let t = Tensor::<B, 1>::from_floats(flat.as_slice(), &self.device).reshape([1, self.net_cfg.obs_dim + self.net_cfg.action_dim]);
        let q1 = self.q1.forward(t.clone());
        let q2 = self.q2.forward(t);
        let q1_v = q1.into_scalar();
        let q2_v = q2.into_scalar();
        q1_v.min(q2_v)
    }

    pub fn value_all(&self, obs_norm_flat: &[f32]) -> Vec<f32> {
        let n = obs_norm_flat.len() / self.net_cfg.obs_dim;
        let mut flat = Vec::with_capacity(n * (self.net_cfg.obs_dim + self.net_cfg.action_dim));
        for i in 0..n {
            flat.extend_from_slice(&obs_norm_flat[i * self.net_cfg.obs_dim..(i + 1) * self.net_cfg.obs_dim]);
            flat.extend(std::iter::repeat(0.0f32).take(self.net_cfg.action_dim));
        }
        let t = Tensor::<B, 1>::from_floats(flat.as_slice(), &self.device).reshape([n, self.net_cfg.obs_dim + self.net_cfg.action_dim]);
        let q1 = self.q1.forward(t.clone()).into_data().to_vec::<f32>().expect("q f32");
        let q2 = self.q2.forward(t).into_data().to_vec::<f32>().expect("q f32");
        q1.iter().zip(q2.iter()).map(|(a, b)| (*a).min(*b)).collect()
    }

    pub fn replay_push(&mut self, item: ReplayItem) {
        self.replay.push(item);
    }

    pub fn replay_sample(&mut self, n: usize) -> Vec<ReplayItem> {
        self.replay.sample(n, &mut self.rng)
    }

    /// Number of transitions currently held in the replay buffer.
    pub fn replay_len(&self) -> usize {
        self.replay.len()
    }

    /// Polyak-average the target critics towards the online critics.
    ///
    /// A hard sync (`target = online`) is *not* a Polyak update: it makes the
    /// target track the online network one step behind at best and destroys
    /// the slowly-moving target the Bellman target relies on.
    fn polyak_target(&mut self) {
        let tau = self.cfg.tau as f32;
        self.q1_target = polyak_critic(&self.q1_target, &self.q1, tau);
        self.q2_target = polyak_critic(&self.q2_target, &self.q2, tau);
    }

    /// Host-side standard-normal samples for the reparameterization trick.
    fn sample_eps(&mut self, batch: usize) -> Tensor<B, 2> {
        let n = batch * self.net_cfg.action_dim;
        let mut v = Vec::with_capacity(n);
        for _ in 0..n {
            let noise: f32 = rand_distr::StandardNormal.sample(&mut self.rng);
            v.push(noise);
        }
        Tensor::<B, 1>::from_floats(v.as_slice(), &self.device)
            .reshape([batch, self.net_cfg.action_dim])
    }

    /// One SAC gradient step.
    ///
    /// The on-policy `UpdateBatch` is **ignored**: SAC is off-policy and
    /// builds its own targets from its replay buffer (spec §9.3).
    pub fn update(&mut self, _batch: &UpdateBatch) -> UpdateStats {
        let n = self.cfg.batch_size.min(self.replay.len()).max(1);
        let replay_batch = self.replay_sample(n);
        if replay_batch.is_empty() {
            return UpdateStats::default();
        }
        let m = replay_batch.len();
        let obs_dim = self.net_cfg.obs_dim;
        let act_dim = self.net_cfg.action_dim;

        // ---- Gather the batch, then scale it with one set of statistics ----
        let mut obs_raw = Vec::with_capacity(m * obs_dim);
        let mut next_obs_raw = Vec::with_capacity(m * obs_dim);
        let mut act_flat = Vec::with_capacity(m * act_dim);
        let mut rewards = Vec::with_capacity(m);
        let mut alive = Vec::with_capacity(m);
        for item in &replay_batch {
            obs_raw.extend_from_slice(&item.obs);
            next_obs_raw.extend_from_slice(&item.next_obs);
            act_flat.extend_from_slice(&item.action);
            rewards.push(item.reward);
            // Only a *termination* ends the episode for bootstrapping purposes.
            // A truncation (step budget) still has a valid next state.
            alive.push(if item.terminated { 0.0 } else { 1.0 });
        }
        let obs = self.normalizer.normalize_batch(&obs_raw);
        let next_obs = self.normalizer.normalize_batch(&next_obs_raw);

        let obs_t = Tensor::<B, 1>::from_floats(obs.as_slice(), &self.device)
            .reshape([m, obs_dim]);
        let next_obs_t = Tensor::<B, 1>::from_floats(next_obs.as_slice(), &self.device)
            .reshape([m, obs_dim]);
        let act_t = Tensor::<B, 1>::from_floats(act_flat.as_slice(), &self.device)
            .reshape([m, act_dim]);
        let rewards_t = Tensor::<B, 1>::from_floats(rewards.as_slice(), &self.device);
        let alive_t = Tensor::<B, 1>::from_floats(alive.as_slice(), &self.device);

        // ---- Critic target: r + γ·(1-done)·(min(Q1',Q2') - α·log π'(a'|s')) ----
        let eps_next = self.sample_eps(m);
        let (a_next, logp_next, _) = self.actor.sample(next_obs_t.clone(), eps_next);
        let next_input = Tensor::cat(vec![next_obs_t.clone(), a_next], 1);
        let q1_next = self.q1_target.forward(next_input.clone());
        let q2_next = self.q2_target.forward(next_input);
        let min_q_next = q1_next.min_pair(q2_next);
        let soft_v = min_q_next.sub(logp_next.mul_scalar(self.cfg.alpha));
        let targets = rewards_t.add(soft_v.mul_scalar(self.cfg.gamma).mul(alive_t));

        // ---- Critic update ----
        let q_input = Tensor::cat(vec![obs_t.clone(), act_t], 1);
        let preds1 = self.q1.forward(q_input.clone());
        let loss1 = (preds1 - targets.clone()).powf_scalar(2.0).mean();
        let grads1 = loss1.backward();
        let grads1_params = GradientsParams::from_grads(grads1, &self.q1);
        self.q1 = self.critic_opt.step(self.cfg.lr as f64, self.q1.clone(), grads1_params);

        let preds2 = self.q2.forward(q_input.clone());
        let loss2 = (preds2 - targets.clone()).powf_scalar(2.0).mean();
        let grads2 = loss2.backward();
        let grads2_params = GradientsParams::from_grads(grads2, &self.q2);
        self.q2 = self.critic_opt.step(self.cfg.lr as f64, self.q2.clone(), grads2_params);

        // ---- Actor update (fully in-tensor) ----
        // The policy gradient needs `∂Q/∂a`, so the actor's action must stay
        // attached to the graph. Rounding the action through the host would
        // detach it and the actor would never move.
        let eps = self.sample_eps(m);
        let (a_pi, logp_pi, entropy) = self.actor.sample(obs_t.clone(), eps);
        let pi_input = Tensor::cat(vec![obs_t.clone(), a_pi], 1);
        let q1_pi = self.q1.forward(pi_input.clone());
        let q2_pi = self.q2.forward(pi_input);
        let min_q_pi = q1_pi.min_pair(q2_pi);
        let actor_loss = min_q_pi
            .mul_scalar(-1.0)
            .add(logp_pi.mul_scalar(self.cfg.alpha))
            .mean();
        let grads_a = actor_loss.backward();
        let grads_a_params = GradientsParams::from_grads(grads_a, &self.actor);
        self.actor = self.actor_opt.step(self.cfg.lr as f64, self.actor.clone(), grads_a_params);

        // ---- Target critics ----
        self.polyak_target();

        let critic_loss = (loss1.into_data().to_vec::<f32>().expect("f32")[0]
            + loss2.into_data().to_vec::<f32>().expect("f32")[0])
            / 2.0;
        UpdateStats {
            policy_loss: actor_loss.into_data().to_vec::<f32>().expect("f32")[0],
            value_loss: critic_loss,
            entropy: entropy.into_data().to_vec::<f32>().expect("f32")[0],
            advantage_mean: 0.0,
            advantage_std: 0.0,
        }
    }

    pub fn save_weights(&self, dir: &Path) -> Result<(), Box<dyn std::error::Error>> {
        let recorder = BinFileRecorder::<FullPrecisionSettings>::default();
        self.actor.clone().save_file(dir.join("actor"), &recorder)?;
        self.q1.clone().save_file(dir.join("q1"), &recorder)?;
        self.q2.clone().save_file(dir.join("q2"), &recorder)?;
        Ok(())
    }

    pub fn load_weights(&mut self, dir: &Path, device: &B::Device) -> Result<(), Box<dyn std::error::Error>> {
        let recorder = BinFileRecorder::<FullPrecisionSettings>::default();
        self.actor = self.actor.clone().load_file(dir.join("actor"), &recorder, device)?;
        self.q1 = self.q1.clone().load_file(dir.join("q1"), &recorder, device)?;
        self.q2 = self.q2.clone().load_file(dir.join("q2"), &recorder, device)?;
        self.q1_target = self.q1.clone();
        self.q2_target = self.q2.clone();
        Ok(())
    }
}

impl<B: AutodiffBackend<FloatElem = f32>> Algorithm<B> for SacAgent<B> {
    fn id(&self) -> &'static str { ALGO_SAC }
    fn net_config(&self) -> &NetConfig { &self.net_cfg }
    fn spec(&self) -> AlgoSpec { AlgoSpec::Sac { config: self.cfg } }
    fn td_params(&self) -> (f32, f32) { (self.cfg.gamma, 0.95) }
    fn act(&mut self, obs_norm: &[f32]) -> (Vec<f32>, f32) { self.act(obs_norm) }
    fn deterministic_action(&self, obs_norm: &[f32]) -> Vec<f32> { self.deterministic_action(obs_norm) }
    fn value_single(&self, obs_norm: &[f32]) -> f32 { self.value_single(obs_norm) }
    fn value_all(&self, obs_norm_flat: &[f32]) -> Vec<f32> { self.value_all(obs_norm_flat) }
    fn update(&mut self, batch: &UpdateBatch) -> UpdateStats { self.update(batch) }
    fn is_off_policy(&self) -> bool { true }
    fn observe_transition(&mut self, transition: &Transition) {
        self.replay_push(ReplayItem {
            obs: transition.obs.clone(),
            action: transition.action.clone(),
            reward: transition.reward,
            terminated: transition.terminated,
            truncated: transition.truncated,
            next_obs: transition.next_obs.clone(),
        });
    }
    fn gradient_steps_per_env_step(&self) -> f32 { 1.0 }
    fn set_normalizer(&mut self, normalizer: RunningNormalizer) { self.normalizer = normalizer; }
    fn save_weights(&self, dir: &Path) -> Result<(), Box<dyn std::error::Error>> { self.save_weights(dir) }
    fn load_weights(&mut self, dir: &Path, device: &B::Device) -> Result<(), Box<dyn std::error::Error>> { self.load_weights(dir, device) }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::algo::{AlgoSpec, ALGO_SAC};
    use crate::backend::{Cpu, cpu_device};
    use crate::buffer::Transition;
    use crate::net::NetConfig;

    fn agent() -> SacAgent<Cpu> {
        let device = cpu_device();
        let ncfg = NetConfig { obs_dim: 4, hidden: 16, action_dim: 4, log_std_init: -0.5 };
        SacAgent::new(&device, &ncfg, &SacConfig::default(), 1)
    }

    #[test]
    fn sac_act_bounded_continuous() {
        let mut a = agent();
        let (action, _lp) = a.act(&[0.1, -0.2, 0.3, 0.4]);
        assert_eq!(action.len(), 4);
        for &x in &action { assert!(x.abs() <= 1.0, "action out of tanh bound: {x}"); }
    }

    #[test]
    fn replay_sample_uniform() {
        let mut a = agent();
        let item = ReplayItem { obs: vec![0.1;4], action: vec![0.2, -0.3, 0.1, 0.4], reward: 1.0, terminated: false, truncated: false, next_obs: vec![0.1;4] };
        for _ in 0..100 { a.replay_push(item.clone()); }
        let samples = a.replay_sample(2);
        assert!(!samples.is_empty());
        for s in &samples {
            assert_eq!(s.action.len(), 4, "continuous action must be 4-dim f32");
        }
    }

    #[test]
    fn algo_spec_from_id_sac() {
        let spec = AlgoSpec::from_id(ALGO_SAC).expect("sac id");
        assert_eq!(spec.id(), ALGO_SAC);
    }

    // ------------------------------------------------------------------
    // Behavioural acceptance tests (spec §9.3).
    //
    // These are the criteria the milestone brief dropped: an off-policy
    // algorithm that does not move its weights is not an algorithm.
    // ------------------------------------------------------------------

    fn param_vec(p: &burn::module::Param<burn::tensor::Tensor<Cpu, 2>>) -> Vec<f32> {
        p.val().into_data().to_vec::<f32>().expect("f32")
    }

    fn max_delta(a: &[f32], b: &[f32]) -> f32 {
        a.iter()
            .zip(b)
            .map(|(x, y)| (x - y).abs())
            .fold(0.0f32, f32::max)
    }

    fn fill_replay(a: &mut SacAgent<Cpu>, n: usize) {
        for i in 0..n {
            let t = i as f32 / n as f32;
            a.replay_push(ReplayItem {
                obs: vec![t, t * 0.5, -t, 1.0 - t],
                action: vec![0.2 * t, -0.3, 0.1 * t, 0.4],
                reward: (i % 7) as f32 - 3.0,
                terminated: false,
                truncated: false,
                next_obs: vec![t + 0.01, t * 0.5, -t, 1.0 - t],
            });
        }
    }

    fn empty_batch() -> UpdateBatch {
        UpdateBatch {
            obs: vec![],
            actions: vec![],
            old_log_probs: vec![],
            advantages: vec![],
            returns: vec![],
        }
    }

    #[test]
    fn sac_actor_and_log_std_train_on_a_non_empty_replay() {
        let mut a = agent();
        fill_replay(&mut a, 64);

        let mean_before = param_vec(&a.actor.mean_head.weight);
        let log_std_before = param_vec(&a.actor.log_std_head.weight);

        a.update(&empty_batch());

        let mean_after = param_vec(&a.actor.mean_head.weight);
        let log_std_after = param_vec(&a.actor.log_std_head.weight);

        assert!(
            max_delta(&mean_before, &mean_after) > 0.0,
            "SAC actor mean_head did not train (gradient was severed)"
        );
        assert!(
            max_delta(&log_std_before, &log_std_after) > 0.0,
            "SAC actor log_std_head did not train (entropy term was constant)"
        );
    }

    #[test]
    fn sac_target_critics_are_polyak_averaged_not_hard_synced() {
        let mut a = agent();
        fill_replay(&mut a, 64);

        let online_before = param_vec(&a.q1.trunk.weight);
        let target_before = param_vec(&a.q1_target.trunk.weight);
        assert_eq!(
            online_before, target_before,
            "SAC target critic must start as an exact copy of the online critic"
        );

        a.update(&empty_batch());

        let online_after = param_vec(&a.q1.trunk.weight);
        let target_after = param_vec(&a.q1_target.trunk.weight);

        assert_ne!(
            online_after, target_after,
            "SAC target critic was hard-synced; spec requires Polyak averaging with tau"
        );
        assert_ne!(
            target_before, target_after,
            "SAC target critic never moved"
        );
        let online_delta = max_delta(&online_before, &online_after);
        let target_delta = max_delta(&target_before, &target_after);
        assert!(
            target_delta < online_delta,
            "target moved ({target_delta}) at least as far as online ({online_delta})"
        );
    }

    #[test]
    fn sac_collects_transitions_through_the_algorithm_trait() {
        let mut a = agent();
        assert!(a.is_off_policy());
        assert!(a.gradient_steps_per_env_step() > 0.0);
        let t = Transition {
            obs: vec![0.1; 4],
            action: vec![0.2; 4],
            log_prob: 0.0,
            reward: 1.0,
            terminated: false,
            truncated: false,
            next_obs: vec![0.3; 4],
        };
        Algorithm::observe_transition(&mut a, &t);
        assert_eq!(a.replay_len(), 1, "SAC replay did not receive the transition");
    }
}
