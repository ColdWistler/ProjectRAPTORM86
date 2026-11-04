//! Continuous D3QN — dueling Q + prioritized replay + tanh continuous actions.
use crate::algo::{Algorithm, AlgoSpec, UpdateBatch, UpdateStats, ALGO_D3QN};
use crate::backend::{Cpu, cpu_device};
use crate::net::NetConfig;
use burn::grad_clipping::GradientClippingConfig;
use burn::nn::{Linear, LinearConfig};
use burn::optim::{AdamW, AdamWConfig, GradientsParams, Optimizer};
use burn::prelude::*;
use rand::Rng;
use rand_distr::Distribution;
use burn::record::{BinFileRecorder, FullPrecisionSettings};
use burn::tensor::backend::AutodiffBackend;
use burn::tensor::ElementConversion;
use rand::rngs::StdRng;
use rand::SeedableRng;
use serde::{Deserialize, Serialize};
use std::path::Path;

pub const ACTION_CLAMP: f32 = 0.999_999;

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq)]
pub struct D3qnConfig {
    pub gamma: f32,
    pub epsilon: f32,
    pub alpha: f32,
    pub beta_min: f32,
    pub tau: f32,
    pub n_steps: usize,
    pub hidden: usize,
    pub buffer_capacity: usize,
    pub batch_size: usize,
    pub max_grad_norm: f32,
    pub lr: f32,
    pub k_candidates: usize,
}

impl Default for D3qnConfig {
    fn default() -> Self {
        Self {
            gamma: 0.99,
            epsilon: 1e-5,
            alpha: 0.6,
            beta_min: 0.4,
            tau: 0.005,
            n_steps: 3,
            hidden: 256,
            buffer_capacity: 8192,
            batch_size: 64,
            max_grad_norm: 0.5,
            lr: 3e-4,
            k_candidates: 32,
        }
    }
}

#[derive(Module, Debug)]
pub struct DuelingQNet<B: Backend> {
    trunk1: Linear<B>,
    trunk2: Linear<B>,
    fc3: Linear<B>,
    value_head: Linear<B>,
    advantage_head: Linear<B>,
    mu_head: Linear<B>,
    pub obs_dim: usize,
    pub action_dim: usize,
    device: B::Device,
}

impl<B: Backend> DuelingQNet<B> {
    pub fn new(device: &B::Device, obs_dim: usize, action_dim: usize, hidden: usize) -> Self {
        let t1 = LinearConfig::new(obs_dim + action_dim, hidden).with_bias(true).init(device);
        let t2 = LinearConfig::new(hidden, hidden).with_bias(true).init(device);
        let fc3 = LinearConfig::new(hidden, hidden).with_bias(true).init(device);
        let value_head = LinearConfig::new(hidden, 1).with_bias(true).init(device);
        let advantage_head = LinearConfig::new(hidden, action_dim).with_bias(true).init(device);
        let mu_head = LinearConfig::new(hidden, action_dim).with_bias(true).init(device);
        Self { trunk1: t1, trunk2: t2, fc3, value_head, advantage_head, mu_head, obs_dim, action_dim, device: device.clone() }
    }

    /// Shared trunk from concatenated (obs+action).
    pub fn q_shared(&self, x: Tensor<B, 2>) -> Tensor<B, 2> {
        let h = self.trunk1.forward(x);
        let h = h.tanh();
        let h = self.fc3.forward(h);
        h.tanh()
    }

    /// Policy mean proposal from obs only (zero-padded action channel).
    pub fn mu(&self, obs: Tensor<B, 2>) -> Tensor<B, 2> {
        let n = obs.shape().dims::<2>()[0];
        let data = obs.into_data().to_vec::<f32>().expect("f32");
        let rows: Vec<Vec<f32>> = (0..n).map(|i| {
            let m = data[i*self.obs_dim..(i+1)*self.obs_dim].to_vec();
            let mut arr = m;
            arr.extend(std::iter::repeat(0.0f32).take(self.action_dim));
            arr
        }).collect();
        let flat: Vec<f32> = rows.iter().flatten().copied().collect();
        let t = Tensor::<B, 1>::from_floats(flat.as_slice(), &self.device).reshape([n, self.obs_dim + self.action_dim]);
        let h = self.q_shared(t);
        self.mu_head.forward(h)
    }

    pub fn q(&self, x: Tensor<B, 2>) -> Tensor<B, 1> {
        let shape = x.shape().dims::<2>();
        let batch = shape[0];
        let data: Vec<f32> = x.into_data().to_vec::<f32>().expect("f32");
        let mut vals = Vec::with_capacity(batch);
        for i in 0..batch {
            let off = i * (self.obs_dim + self.action_dim);
            let row = data[off..off+self.obs_dim+self.action_dim].to_vec();
            let row = row;
            let h = self.q_shared(Tensor::<B, 2>::from_floats(row.as_slice(), &self.device).reshape([1, self.obs_dim + self.action_dim]));
            let v = self.value_head.forward(h.clone()).into_data().to_vec::<f32>().expect("f")[0];
            let a = self.advantage_head.forward(h.clone()).into_data().to_vec::<f32>().expect("f");
            let avg_a = a.iter().sum::<f32>() / a.len() as f32;
            vals.push(v + a.iter().map(|av| av - avg_a).sum::<f32>());
        }
        Tensor::<B, 1>::from_floats(vals.as_slice(), &self.device)
    }

}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct ReplayItem {
    obs: Vec<f32>,
    action: Vec<f32>,
    reward: f32,
    terminated: bool,
    truncated: bool,
    next_obs: Vec<f32>,
}

/// Minimal prioritized replay with SumTree-style priority tracking.
struct PrioritizedReplay {
    capacity: usize,
    count: usize,
    items: Vec<Option<ReplayItem>>,
    tree_size: usize,
    tree: Vec<u64>,
    priorities: Vec<f32>,
    alpha: f32,
}

impl PrioritizedReplay {
    fn new(cap: usize, alpha: f32) -> Self {
        let ts = cap * 2;
        Self { capacity: cap, count: 0, items: vec![None; cap], tree_size: ts, tree: vec![0u64; ts], priorities: vec![0.0f32; cap], alpha }
    }

    fn push(&mut self, item: ReplayItem, priority: f32) {
        let idx = self.count % self.capacity;
        self.items[idx] = Some(item);
        self.priorities[idx] = priority.max(self.alpha);
        self.count += 1;
        let pos = idx + self.capacity;
        self.tree[pos] = priority as u64;
        let mut p = pos >> 1;
        while p > 0 {
            self.tree[p] = self.tree[p*2] + self.tree[p*2+1];
            p >>= 1;
        }
    }

    fn total(&self) -> u64 { self.tree[1] }
    fn len(&self) -> usize { self.items.iter().filter(|o| o.is_some()).count() }

    fn sample_indices(&self, n: usize, rng: &mut StdRng) -> Vec<usize> {
        let mut out = Vec::with_capacity(n);
        let total = self.total().max(1);
        let cap = self.capacity;
        for _ in 0..n {
            let v = (rng.gen::<u64>() % total) + 1;
            let mut idx = 1usize;
            while idx < cap {
                if self.tree[idx*2] >= v {
                    idx *= 2;
                } else {
                    idx = idx*2 + 1;
                }
            }
            out.push(idx - cap);
        }
        out
    }

    fn replay_sample(&mut self, n: usize, rng: &mut StdRng) -> Vec<ReplayItem> {
        let indices = self.sample_indices(n, rng);
        let mut out = Vec::with_capacity(indices.len());
        for idx in indices {
            if let Some(ref item) = self.items[idx % self.capacity] {
                out.push(item.clone());
            }
        }
        out
    }
}

pub struct D3qnAgent<B: AutodiffBackend<FloatElem = f32>> {
    pub net: DuelingQNet<B>,
    pub tgt: DuelingQNet<B>,
    pub cfg: D3qnConfig,
    pub device: B::Device,
    pub optimizer: burn::optim::adaptor::OptimizerAdaptor<AdamW, DuelingQNet<B>, B>,
    pub rng: StdRng,
    replay: PrioritizedReplay,
    net_cfg: NetConfig,
}

impl<B: AutodiffBackend<FloatElem = f32>> D3qnAgent<B> {
    pub fn new(device: &B::Device, net_cfg: &NetConfig, cfg: &D3qnConfig, seed: u64) -> Self {
        let net = DuelingQNet::new(device, net_cfg.obs_dim, net_cfg.action_dim, cfg.hidden);
        let tgt = DuelingQNet::new(device, net_cfg.obs_dim, net_cfg.action_dim, cfg.hidden);
        let optimizer = AdamWConfig::new()
            .with_grad_clipping(Some(GradientClippingConfig::Norm(cfg.max_grad_norm)))
            .init::<B, DuelingQNet<B>>();
        Self {
            net,
            tgt,
            cfg: *cfg,
            device: device.clone(),
            optimizer,
            rng: StdRng::seed_from_u64(seed),
            replay: PrioritizedReplay::new(cfg.buffer_capacity, cfg.alpha),
            net_cfg: *net_cfg,
        }
    }

    pub fn deterministic_action(&self, obs_norm: &[f32]) -> Vec<f32> {
        let x = Tensor::<B, 1>::from_floats(obs_norm, &self.device).reshape([1, self.net_cfg.obs_dim]);
        let mu = self.net.mu(x);
        let v: Vec<f32> = mu.into_data().to_vec::<f32>().expect("mu f32");
        v.into_iter().map(|a| a.tanh().clamp(-ACTION_CLAMP, ACTION_CLAMP)).collect()
    }

    pub fn act(&mut self, obs_norm: &[f32]) -> (Vec<f32>, f32) {
        let x = Tensor::<B, 1>::from_floats(obs_norm, &self.device).reshape([1, self.net_cfg.obs_dim]);
        let mu = self.net.mu(x);
        let v: Vec<f32> = mu.into_data().to_vec::<f32>().expect("mu f32");
        let mut action = Vec::with_capacity(self.net_cfg.action_dim);
        for d in 0..self.net_cfg.action_dim {
            let noise = rand_distr::Normal::new(0.0, 0.15).unwrap().sample(&mut self.rng);
            let a = (v[d] + noise).tanh().clamp(-ACTION_CLAMP, ACTION_CLAMP);
            action.push(a);
        }
        (action, 0.0f32)
    }

    pub fn value_single(&self, obs_norm: &[f32]) -> f32 {
        let n = 1;
        let obs = obs_norm.to_vec();
        let zeros = vec![0.0f32; self.net_cfg.action_dim];
        let mut flat = Vec::with_capacity(n * (self.net_cfg.obs_dim + self.net_cfg.action_dim));
        for i in 0..n {
            flat.extend_from_slice(&obs[i*self.net_cfg.obs_dim..(i+1)*self.net_cfg.obs_dim]);
            flat.extend_from_slice(&zeros);
        }
        let t = Tensor::<B, 2>::from_floats(flat.as_slice(), &self.device).reshape([n, self.net_cfg.obs_dim + self.net_cfg.action_dim]);
        self.net.q(t).into_scalar()
    }

    pub fn value_all(&self, obs_norm_flat: &[f32]) -> Vec<f32> {
        let n = obs_norm_flat.len() / self.net_cfg.obs_dim;
        let mut flat = Vec::with_capacity(n * (self.net_cfg.obs_dim + self.net_cfg.action_dim));
        for i in 0..n {
            flat.extend_from_slice(&obs_norm_flat[i*self.net_cfg.obs_dim..(i+1)*self.net_cfg.obs_dim]);
            flat.extend(std::iter::repeat(0.0f32).take(self.net_cfg.action_dim));
        }
        let t = Tensor::<B, 2>::from_floats(flat.as_slice(), &self.device).reshape([n, self.net_cfg.obs_dim + self.net_cfg.action_dim]);
        self.net.q(t).into_data().to_vec::<f32>().expect("q f32")
    }

    pub fn replay_sample(&mut self, n: usize) -> Vec<ReplayItem> {
        self.replay.replay_sample(n, &mut self.rng)
    }

    pub fn replay_push(&mut self, item: ReplayItem, priority: f32) {
        self.replay.push(item, priority);
    }

    fn make_target_batch(&self, batch: &[ReplayItem]) -> (Tensor<B, 2>, Tensor<B, 1>) {
        let k = self.cfg.k_candidates;
        let n = batch.len();
        let obs_dim = self.net_cfg.obs_dim;
        let act_dim = self.net_cfg.action_dim;
        let mut x_flat = Vec::with_capacity(n * (obs_dim + act_dim));
        let mut target_q = vec![0.0f32; n];
        for (i, item) in batch.iter().enumerate() {
            x_flat.extend_from_slice(&item.obs);
            x_flat.extend_from_slice(&item.action);
            let s_next = item.next_obs.clone();
            let candidates: Vec<Vec<f32>> = (0..k).map(|_j| {
                (0..act_dim).map(|_| (rand::random::<f32>() * 1.8 - 0.9)).collect()
            }).collect();
            let mut best_q = f32::NEG_INFINITY;
            for cand in candidates {
                let mut c_flat = s_next.clone();
                c_flat.extend(cand);
                let t = Tensor::<B, 2>::from_floats(c_flat.as_slice(), &self.device).reshape([1, obs_dim + act_dim]);
                let q_val = self.tgt.q(t).into_scalar();
                if q_val > best_q { best_q = q_val; }
            }
            target_q[i] = item.reward + self.cfg.gamma * best_q;
            if item.terminated || item.truncated { target_q[i] = item.reward; }
        }
        (Tensor::<B, 2>::from_floats(x_flat.as_slice(), &self.device).reshape([n, obs_dim + act_dim]),
         Tensor::<B, 1>::from_floats(target_q.as_slice(), &self.device))
    }

    pub fn update(&mut self, batch: &UpdateBatch) -> UpdateStats {
        let replay_batch = self.replay_sample(1);
        if replay_batch.is_empty() {
            return UpdateStats { policy_loss: 0.0, value_loss: 0.0, entropy: 0.0, advantage_mean: 0.0, advantage_std: 0.0 };
        }
        let batch_size = replay_batch.len();
        let x_rows: Vec<f32> = replay_batch.iter()
            .flat_map(|item| {
                let mut v = item.obs.clone();
                v.extend(item.action.clone());
                v
            }).collect();
        let x_shape = [batch_size, self.net_cfg.obs_dim + self.net_cfg.action_dim];
        let x = Tensor::<B, 2>::from_floats(x_rows.as_slice(), &self.device)
            .reshape(x_shape);
        let preds = self.net.q(x.clone());
        let target_q = Tensor::<B, 1>::zeros([batch_size], &self.device);
        let loss = (preds - target_q.clone()).powf_scalar(2.0).mean();
        let grads = loss.backward();
        let grads = GradientsParams::from_grads(grads, &self.net);
        self.net = self.optimizer.step(self.cfg.lr as f64, self.net.clone(), grads);
        self.tgt = DuelingQNet::new(&self.device, self.net_cfg.obs_dim, self.net_cfg.action_dim, self.cfg.hidden);
        UpdateStats { policy_loss: loss.into_data().to_vec::<f32>().expect("f32")[0], value_loss: 0.0, entropy: 0.0, advantage_mean: 0.0, advantage_std: 0.0 }
    }

    pub fn save_weights(&self, dir: &Path) -> Result<(), Box<dyn std::error::Error>> {
        let recorder = BinFileRecorder::<FullPrecisionSettings>::default();
        self.net.clone().save_file(dir.join("net"), &recorder)?;
        Ok(())
    }

    pub fn load_weights(&mut self, dir: &Path, device: &B::Device) -> Result<(), Box<dyn std::error::Error>> {
        let recorder = BinFileRecorder::<FullPrecisionSettings>::default();
        self.net = self.net.clone().load_file(dir.join("net"), &recorder, device)?;
        Ok(())
    }
}

impl<B: AutodiffBackend<FloatElem = f32>> Algorithm<B> for D3qnAgent<B> {
    fn id(&self) -> &'static str { ALGO_D3QN }
    fn net_config(&self) -> &NetConfig { &self.net_cfg }
    fn spec(&self) -> AlgoSpec { AlgoSpec::D3qn { config: self.cfg } }
    fn td_params(&self) -> (f32, f32) { (self.cfg.gamma, 0.95) }
    fn act(&mut self, obs_norm: &[f32]) -> (Vec<f32>, f32) { self.act(obs_norm) }
    fn deterministic_action(&self, obs_norm: &[f32]) -> Vec<f32> { self.deterministic_action(obs_norm) }
    fn value_single(&self, obs_norm: &[f32]) -> f32 { self.value_single(obs_norm) }
    fn value_all(&self, obs_norm_flat: &[f32]) -> Vec<f32> { self.value_all(obs_norm_flat) }
    fn update(&mut self, batch: &UpdateBatch) -> UpdateStats { self.update(batch) }
    fn save_weights(&self, dir: &Path) -> Result<(), Box<dyn std::error::Error>> { self.save_weights(dir) }
    fn load_weights(&mut self, dir: &Path, device: &B::Device) -> Result<(), Box<dyn std::error::Error>> { self.load_weights(dir, device) }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::algo::{AlgoSpec, make_algorithm, ALGO_D3QN};
    use crate::net::NetConfig;

    fn agent() -> D3qnAgent<Cpu> {
        let device = cpu_device();
        let ncfg = NetConfig { obs_dim: 4, hidden: 16, action_dim: 4, log_std_init: -0.5 };
        D3qnAgent::new(&device, &ncfg, &D3qnConfig::default(), 1)
    }

    #[test]
    fn d3qn_act_bounded_continuous() {
        let mut a = agent();
        let (action, _lp) = a.act(&[0.1, -0.2, 0.3, 0.4]);
        assert_eq!(action.len(), 4);
        for &x in &action { assert!(x.abs() <= 1.0, "action out of tanh bound: {x}"); }
    }

    #[test]
    fn replay_sample_continuous_tuples() {
        let mut a = agent();
        a.replay_push(ReplayItem { obs: vec![0.1;4], action: vec![0.2, -0.3, 0.1, 0.4], reward: 1.0, terminated: false, truncated: false, next_obs: vec![0.1;4] }, 1.0);
        let samples = a.replay_sample(2);
        assert!(!samples.is_empty());
        for s in &samples {
            assert_eq!(s.action.len(), 4, "continuous action must be 4-dim f32, not int");
        }
    }
}
