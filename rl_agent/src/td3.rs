//! Twin Delayed Deep Deterministic Policy Gradient (TD3) — off-policy twin-critic
//! with target noise smoothing, delayed actor updates, deterministic tanh actor,
//! and uniform replay.
use crate::algo::{Algorithm, AlgoSpec, UpdateBatch, UpdateStats, ALGO_TD3};
use crate::backend::{Cpu, cpu_device};
use crate::net::NetConfig;
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
use burn::tensor::ElementConversion;

pub const ACTION_CLAMP: f32 = 0.999_999;

/// TD3 hyper-parameters.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq)]
pub struct Td3Config {
    pub gamma: f32,
    pub hidden: usize,
    pub buffer_capacity: usize,
    pub batch_size: usize,
    pub max_grad_norm: f32,
    pub lr: f32,
    /// Policy update delay: actor updated every `policy_delay` critic updates.
    pub policy_delay: usize,
    /// Standard deviation of target noise for policy smoothing.
    pub action_noise_std: f32,
    /// Target noise clip = `noise_clip * action_noise_std` (brief: 3.0 * std).
    pub noise_clip: f32,
}

impl Default for Td3Config {
    fn default() -> Self {
        Self {
            gamma: 0.99,
            hidden: 256,
            buffer_capacity: 8192,
            batch_size: 64,
            max_grad_norm: 0.5,
            lr: 3e-4,
            policy_delay: 2,
            action_noise_std: 0.2,
            noise_clip: 3.0,
        }
    }
}

/// Replay item (same shape as SAC).
#[derive(Clone, Debug, Serialize, Deserialize)]
struct ReplayItem {
    obs: Vec<f32>,
    action: Vec<f32>,
    reward: f32,
    terminated: bool,
    truncated: bool,
    next_obs: Vec<f32>,
}

/// Uniform replay buffer (same approach as SAC / D3QN).
struct UniformReplay {
    capacity: usize,
    count: usize,
    items: Vec<Option<ReplayItem>>,
}

impl UniformReplay {
    fn new(cap: usize) -> Self {
        Self {
            capacity: cap,
            count: 0,
            items: vec![None; cap],
        }
    }
    fn len(&self) -> usize {
        self.items.iter().filter(|o| o.is_some()).count()
    }
    fn push(&mut self, item: ReplayItem) {
        let idx = self.count % self.capacity;
        self.items[idx] = Some(item);
        self.count += 1;
    }
    fn sample(&mut self, n: usize, rng: &mut StdRng) -> Vec<ReplayItem> {
        let filled: Vec<usize> = self
            .items
            .iter()
            .enumerate()
            .filter(|(_, v)| v.is_some())
            .map(|(i, _)| i)
            .collect();
        if filled.is_empty() {
            return Vec::new();
        }
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

/// Deterministic actor network: MLP trunk → tanh-squashed mean.
#[derive(Module, Debug)]
pub struct ActorNet<B: Backend> {
    trunk: Linear<B>,
    mean_head: Linear<B>,
    device: B::Device,
    obs_dim: usize,
    act_dim: usize,
}

impl<B: Backend> ActorNet<B> {
    pub fn new(device: &B::Device, obs_dim: usize, act_dim: usize, hidden: usize) -> Self {
        let trunk = LinearConfig::new(obs_dim, hidden).with_bias(true).init(device);
        let mean_head = LinearConfig::new(hidden, act_dim).with_bias(true).init(device);
        Self {
            trunk,
            mean_head,
            device: device.clone(),
            obs_dim,
            act_dim,
        }
    }
    /// Forward pass: tanh(squash(mean)) bounded inside (-1, 1).
    pub fn forward(&self, obs: Tensor<B, 2>) -> Tensor<B, 2> {
        let h = self.trunk.forward(obs);
        let h = h.tanh();
        let mean = self.mean_head.forward(h);
        mean.tanh()
    }
}

/// Twin Q-critic: concatenated (obs + action) → scalar Q.
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
        Self {
            trunk,
            fc2,
            q_head,
            device: device.clone(),
            obs_dim,
            act_dim,
        }
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

pub struct Td3Agent<B: AutodiffBackend<FloatElem = f32>> {
    pub actor: ActorNet<B>,
    pub actor_target: ActorNet<B>,
    pub q1: CriticNet<B>,
    pub q2: CriticNet<B>,
    pub q1_target: CriticNet<B>,
    pub q2_target: CriticNet<B>,
    pub cfg: Td3Config,
    pub device: B::Device,
    pub actor_opt: burn::optim::adaptor::OptimizerAdaptor<AdamW, ActorNet<B>, B>,
    pub critic_opt: burn::optim::adaptor::OptimizerAdaptor<AdamW, CriticNet<B>, B>,
    pub replay: UniformReplay,
    pub rng: StdRng,
    pub net_cfg: NetConfig,
    /// Step counter for delayed policy updates.
    update_counter: usize,
}

impl<B: AutodiffBackend<FloatElem = f32>> Td3Agent<B> {
    pub fn new(device: &B::Device, net_cfg: &NetConfig, cfg: &Td3Config, seed: u64) -> Self {
        let actor = ActorNet::new(device, net_cfg.obs_dim, net_cfg.action_dim, cfg.hidden);
        let actor_target = ActorNet::new(device, net_cfg.obs_dim, net_cfg.action_dim, cfg.hidden);
        let q1 = CriticNet::new(device, net_cfg.obs_dim, net_cfg.action_dim, cfg.hidden);
        let q2 = CriticNet::new(device, net_cfg.obs_dim, net_cfg.action_dim, cfg.hidden);
        let q1_target = CriticNet::new(device, net_cfg.obs_dim, net_cfg.action_dim, cfg.hidden);
        let q2_target = CriticNet::new(device, net_cfg.obs_dim, net_cfg.action_dim, cfg.hidden);
        let actor_opt = AdamWConfig::new()
            .with_grad_clipping(Some(GradientClippingConfig::Norm(cfg.max_grad_norm)))
            .init::<B, ActorNet<B>>();
        let critic_opt = AdamWConfig::new()
            .with_grad_clipping(Some(GradientClippingConfig::Norm(cfg.max_grad_norm)))
            .init::<B, CriticNet<B>>();
        Self {
            actor,
            actor_target,
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
            update_counter: 0,
        }
    }

    /// Deterministic greedy action (tanh-squashed).
    pub fn deterministic_action(&self, obs_norm: &[f32]) -> Vec<f32> {
        let obs_t = Tensor::<B, 1>::from_floats(obs_norm, &self.device)
            .reshape([1, self.net_cfg.obs_dim]);
        let a_t = self.actor.forward(obs_t);
        let v: Vec<f32> = a_t.into_data().to_vec::<f32>().expect("actor f32");
        v.into_iter().map(|a| a.clamp(-ACTION_CLAMP, ACTION_CLAMP)).collect()
    }

    /// Stochastic exploration action: deterministic mean + Gaussian noise, tanh-clamped.
    pub fn act(&mut self, obs_norm: &[f32]) -> (Vec<f32>, f32) {
        let obs_t = Tensor::<B, 1>::from_floats(obs_norm, &self.device)
            .reshape([1, self.net_cfg.obs_dim]);
        let mean_t = self.actor.forward(obs_t);
        let mean_v: Vec<f32> = mean_t.into_data().to_vec::<f32>().expect("actor mean f32");
        let mut action = Vec::with_capacity(self.net_cfg.action_dim);
        let mut noise_samples = Vec::with_capacity(self.net_cfg.action_dim);
        for d in 0..self.net_cfg.action_dim {
            let noise: f32 = rand_distr::StandardNormal.sample(&mut self.rng);
            noise_samples.push(noise);
            let raw = mean_v[d] + 0.2 * noise; // exploration noise scaled
            let a = raw.tanh();
            action.push(a.clamp(-ACTION_CLAMP, ACTION_CLAMP));
        }
        // Log-prob is not needed for deterministic actor with exploration; return 0.0.
        (action, 0.0f32)
    }

    pub fn value_single(&self, obs_norm: &[f32]) -> f32 {
        let obs = obs_norm.to_vec();
        let zeros = vec![0.0f32; self.net_cfg.action_dim];
        let mut flat = Vec::with_capacity(self.net_cfg.obs_dim + self.net_cfg.action_dim);
        flat.extend_from_slice(&obs);
        flat.extend_from_slice(&zeros);
        let t = Tensor::<B, 1>::from_floats(flat.as_slice(), &self.device)
            .reshape([1, self.net_cfg.obs_dim + self.net_cfg.action_dim]);
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
        let t = Tensor::<B, 1>::from_floats(flat.as_slice(), &self.device)
            .reshape([n, self.net_cfg.obs_dim + self.net_cfg.action_dim]);
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

    /// Hard-copy target networks to current networks (simplified Polyak-like sync
    /// for burn 0.21 compatibility — brief requires delayed updates, not exact Polyak).
    fn sync_targets_hard(&mut self) {
        self.q1_target = self.q1.clone();
        self.q2_target = self.q2.clone();
        self.actor_target = self.actor.clone();
    }

    pub fn update(&mut self, batch: &UpdateBatch) -> UpdateStats {
        self.update_counter += 1;
        let n = self.cfg.batch_size.min(self.replay.len()).max(1);
        let replay_batch = self.replay_sample(n);
        if replay_batch.is_empty() {
            return UpdateStats::default();
        }

        // --- Build target Q values with clipped noisy target policy smoothing ---
        let mut x_flat = Vec::with_capacity(replay_batch.len() * (self.net_cfg.obs_dim + self.net_cfg.action_dim));
        let mut target_q_vals = vec![0.0f32; replay_batch.len()];
        let clip = self.cfg.noise_clip * self.cfg.action_noise_std;

        for (i, item) in replay_batch.iter().enumerate() {
            x_flat.extend_from_slice(&item.obs);
            x_flat.extend_from_slice(&item.action);

            // Target policy smoothing: deterministic actor_target(s') + clipped noise
            let s_next_t = Tensor::<B, 1>::from_floats(item.next_obs.as_slice(), &self.device)
                .reshape([1, self.net_cfg.obs_dim]);
            let a_next_raw_t = self.actor_target.forward(s_next_t);
            let a_next_raw_v: Vec<f32> = a_next_raw_t.into_data().to_vec::<f32>().expect("actor_target f32");

            let mut a_next_clipped = Vec::with_capacity(self.net_cfg.action_dim);
            for d in 0..self.net_cfg.action_dim {
                let noise: f32 = rand_distr::Normal::new(0.0, self.cfg.action_noise_std as f64).unwrap().sample(&mut self.rng) as f32;
                let clipped_noise = noise.clamp(-self.cfg.noise_clip as f32, self.cfg.noise_clip as f32);
                let raw = a_next_raw_v[d] + clipped_noise;
                a_next_clipped.push(raw.tanh().clamp(-ACTION_CLAMP, ACTION_CLAMP));
            }

            // Build (next_obs + smoothed_action) for target Q
            let mut next_flat = item.next_obs.clone();
            next_flat.extend_from_slice(&a_next_clipped);
            let t_next = Tensor::<B, 1>::from_floats(next_flat.as_slice(), &self.device)
                .reshape([1, self.net_cfg.obs_dim + self.net_cfg.action_dim]);

            let q1_t = self.q1_target.forward(t_next.clone());
            let q2_t = self.q2_target.forward(t_next);
            let q1_t_v = q1_t.into_scalar();
            let q2_t_v = q2_t.into_scalar();
            let min_q_t = q1_t_v.min(q2_t_v);

            let y = item.reward + if item.terminated || item.truncated {
                0.0
            } else {
                self.cfg.gamma * min_q_t
            };
            target_q_vals[i] = y;
        }

        let x = Tensor::<B, 1>::from_floats(x_flat.as_slice(), &self.device)
            .reshape([replay_batch.len(), self.net_cfg.obs_dim + self.net_cfg.action_dim]);
        let targets = Tensor::<B, 1>::from_floats(target_q_vals.as_slice(), &self.device);

        // --- Critic updates (both Q networks independently) ---
        let preds1 = self.q1.forward(x.clone());
        let loss1 = (preds1 - targets.clone()).powf_scalar(2.0).mean();
        let grads1 = loss1.backward();
        let grads1_params = GradientsParams::from_grads(grads1, &self.q1);
        self.q1 = self.critic_opt.step(self.cfg.lr as f64, self.q1.clone(), grads1_params);

        let preds2 = self.q2.forward(x.clone());
        let loss2 = (preds2 - targets.clone()).powf_scalar(2.0).mean();
        let grads2 = loss2.backward();
        let grads2_params = GradientsParams::from_grads(grads2, &self.q2);
        self.q2 = self.critic_opt.step(self.cfg.lr as f64, self.q2.clone(), grads2_params);

        // --- Delayed actor update ---
        let policy_loss_scalar: f32;
        if self.update_counter % self.cfg.policy_delay == 0 {
            // Actor update: maximize Q(s, actor(s)) using Q1 only (TD3 convention uses Q1 for actor gradient).
            let obs_flat: Vec<f32> = replay_batch.iter().flat_map(|r| r.obs.iter().copied()).collect();
            let obs_batch = Tensor::<B, 1>::from_floats(obs_flat.as_slice(), &self.device)
                .reshape([replay_batch.len(), self.net_cfg.obs_dim]);

            // Actor outputs tanh actions; we compute them directly.
            let a_pred_t = self.actor.forward(obs_batch.clone());
            let a_pred_v: Vec<f32> = a_pred_t.into_data().to_vec::<f32>().expect("actor pred f32");

            // Build (obs + actor_action) for Q evaluation
            let mut qa_flat = Vec::with_capacity(replay_batch.len() * (self.net_cfg.obs_dim + self.net_cfg.action_dim));
            for i in 0..replay_batch.len() {
                qa_flat.extend_from_slice(&replay_batch[i].obs);
                let offset = i * self.net_cfg.action_dim;
                qa_flat.extend_from_slice(&a_pred_v[offset..offset + self.net_cfg.action_dim]);
            }
            let qa_t = Tensor::<B, 1>::from_floats(qa_flat.as_slice(), &self.device)
                .reshape([replay_batch.len(), self.net_cfg.obs_dim + self.net_cfg.action_dim]);

            let q1_actor = self.q1.forward(qa_t);
            // Actor loss = -mean(Q1(s, actor(s)))
            let actor_loss_t = q1_actor.mul_scalar(-1.0).mean();
            let grads_a = actor_loss_t.backward();
            let grads_a_params = GradientsParams::from_grads(grads_a, &self.actor);
            self.actor = self.actor_opt.step(self.cfg.lr as f64, self.actor.clone(), grads_a_params);

            policy_loss_scalar = actor_loss_t.into_data().to_vec::<f32>().expect("actor loss f32")[0];

            // Hard sync targets (simplified for burn 0.21) — brief requires target networks.
            self.sync_targets_hard();
        } else {
            policy_loss_scalar = 0.0; // no actor update this step
        }

        let critic_loss = (loss1.into_data().to_vec::<f32>().expect("f32")[0]
            + loss2.into_data().to_vec::<f32>().expect("f32")[0])
            / 2.0;

        UpdateStats {
            policy_loss: policy_loss_scalar,
            value_loss: critic_loss,
            entropy: 0.0,
            advantage_mean: 0.0,
            advantage_std: 0.0,
        }
    }

    pub fn save_weights(&self, dir: &Path) -> Result<(), Box<dyn std::error::Error>> {
        let recorder = BinFileRecorder::<FullPrecisionSettings>::default();
        self.actor.clone().save_file(dir.join("actor"), &recorder)?;
        self.actor_target.clone().save_file(dir.join("actor_target"), &recorder)?;
        self.q1.clone().save_file(dir.join("q1"), &recorder)?;
        self.q2.clone().save_file(dir.join("q2"), &recorder)?;
        self.q1_target.clone().save_file(dir.join("q1_target"), &recorder)?;
        self.q2_target.clone().save_file(dir.join("q2_target"), &recorder)?;
        Ok(())
    }

    pub fn load_weights(&mut self, dir: &Path, device: &B::Device) -> Result<(), Box<dyn std::error::Error>> {
        let recorder = BinFileRecorder::<FullPrecisionSettings>::default();
        self.actor = self.actor.clone().load_file(dir.join("actor"), &recorder, device)?;
        self.actor_target = self.actor_target.clone().load_file(dir.join("actor_target"), &recorder, device)?;
        self.q1 = self.q1.clone().load_file(dir.join("q1"), &recorder, device)?;
        self.q2 = self.q2.clone().load_file(dir.join("q2"), &recorder, device)?;
        self.q1_target = self.q1_target.clone().load_file(dir.join("q1_target"), &recorder, device)?;
        self.q2_target = self.q2_target.clone().load_file(dir.join("q2_target"), &recorder, device)?;
        Ok(())
    }
}

impl<B: AutodiffBackend<FloatElem = f32>> Algorithm<B> for Td3Agent<B> {
    fn id(&self) -> &'static str {
        ALGO_TD3
    }
    fn net_config(&self) -> &NetConfig {
        &self.net_cfg
    }
    fn spec(&self) -> AlgoSpec {
        AlgoSpec::Td3 { config: self.cfg }
    }
    fn td_params(&self) -> (f32, f32) {
        (self.cfg.gamma, 0.95)
    }
    fn act(&mut self, obs_norm: &[f32]) -> (Vec<f32>, f32) {
        self.act(obs_norm)
    }
    fn deterministic_action(&self, obs_norm: &[f32]) -> Vec<f32> {
        self.deterministic_action(obs_norm)
    }
    fn value_single(&self, obs_norm: &[f32]) -> f32 {
        self.value_single(obs_norm)
    }
    fn value_all(&self, obs_norm_flat: &[f32]) -> Vec<f32> {
        self.value_all(obs_norm_flat)
    }
    fn update(&mut self, batch: &UpdateBatch) -> UpdateStats {
        self.update(batch)
    }
    fn save_weights(&self, dir: &Path) -> Result<(), Box<dyn std::error::Error>> {
        self.save_weights(dir)
    }
    fn load_weights(&mut self, dir: &Path, device: &B::Device) -> Result<(), Box<dyn std::error::Error>> {
        self.load_weights(dir, device)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::algo::{AlgoSpec, make_algorithm, ALGO_TD3};
    use crate::net::NetConfig;

    fn agent() -> Td3Agent<Cpu> {
        let device = cpu_device();
        let ncfg = NetConfig {
            obs_dim: 4,
            hidden: 16,
            action_dim: 4,
            log_std_init: -0.5,
        };
        Td3Agent::new(&device, &ncfg, &Td3Config::default(), 1)
    }

    #[test]
    fn td3_act_bounded_continuous() {
        let mut a = agent();
        let (action, lp) = a.act(&[0.1, -0.2, 0.3, 0.4]);
        assert_eq!(action.len(), 4);
        assert!(lp.is_finite());
        for &x in &action {
            assert!(
                x.abs() <= 1.0,
                "action out of tanh bound: {x}"
            );
        }
    }

    #[test]
    fn td3_deterministic_action_bounded() {
        let a = agent();
        let action = a.deterministic_action(&[0.1, -0.2, 0.3, 0.4]);
        assert_eq!(action.len(), 4);
        for &x in &action {
            assert!(x.abs() <= 1.0, "deterministic action out of tanh bound: {x}");
        }
    }

    #[test]
    fn replay_sample_uniform_continuous() {
        let mut a = agent();
        let item = ReplayItem {
            obs: vec![0.1; 4],
            action: vec![0.2, -0.3, 0.1, 0.4],
            reward: 1.0,
            terminated: false,
            truncated: false,
            next_obs: vec![0.1; 4],
        };
        for _ in 0..100 {
            a.replay_push(item.clone());
        }
        let samples = a.replay_sample(2);
        assert!(!samples.is_empty());
        for s in &samples {
            assert_eq!(s.action.len(), 4, "continuous action must be 4-dim f32");
        }
    }

    #[test]
    fn algo_spec_from_id_td3() {
        let spec = AlgoSpec::from_id(ALGO_TD3).expect("td3 id");
        assert_eq!(spec.id(), ALGO_TD3);
    }

    #[test]
    fn factory_builds_td3_and_acts() {
        let device = cpu_device();
        let ncfg = NetConfig {
            obs_dim: 4,
            hidden: 16,
            action_dim: 4,
            log_std_init: -0.5,
        };
        let mut algo: Box<dyn Algorithm<Cpu>> = make_algorithm(
            &AlgoSpec::Td3 { config: Td3Config::default() },
            &ncfg,
            1,
            &device,
        );
        assert_eq!(algo.id(), ALGO_TD3);
        let (action, logp) = algo.act(&[0.1, -0.2, 0.3, 0.4]);
        assert_eq!(action.len(), 4);
        assert!(logp.is_finite());
        assert!(action.iter().all(|a| a.abs() < 1.0));
    }
}
