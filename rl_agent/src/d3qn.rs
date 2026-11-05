//! Continuous D3QN — dueling critic, double-Q target, deterministic policy
//! head and prioritized replay.
//!
//! The action space is continuous (`mission.v1`, 4-D), so the discrete
//! "argmax over the action set" of the original D3QN paper is replaced by a
//! candidate search: the policy's proposal plus seeded perturbations is
//! scored by the **online** critic (the *double* part) and the winner is
//! evaluated by the **target** critic.
//!
//! The critic is *dueling*: `Q(s,a) = V(s,a) + A(s,a) − A(s,0)`. The
//! advantage is centred on the zero action so the decomposition has a fixed
//! origin and the Q value still depends on the action that was taken.

use crate::algo::{Algorithm, AlgoSpec, UpdateBatch, UpdateStats, ALGO_D3QN};
use crate::buffer::Transition;
use crate::net::{polyak_linear, NetConfig};
use crate::normalize::RunningNormalizer;
use burn::grad_clipping::GradientClippingConfig;
use burn::nn::{Linear, LinearConfig};
use burn::optim::{AdamW, AdamWConfig, GradientsParams, Optimizer};
use burn::prelude::*;
use burn::record::{BinFileRecorder, FullPrecisionSettings};
use burn::tensor::backend::AutodiffBackend;
use rand::Rng;
use rand::rngs::StdRng;
use rand::SeedableRng;
use rand_distr::Distribution;
use serde::{Deserialize, Serialize};
use std::path::Path;

pub const ACTION_CLAMP: f32 = 0.999_999;

/// Default exploration noise added to the policy proposal.
fn default_exploration_noise() -> f32 {
    0.15
}

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
    /// Standard deviation of the noise added to the policy proposal when
    /// exploring and when building candidate actions.
    #[serde(default = "default_exploration_noise")]
    pub exploration_noise: f32,
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
            k_candidates: 8,
            exploration_noise: default_exploration_noise(),
        }
    }
}

/// Dueling Q critic over `(observation, action)` pairs.
#[derive(Module, Debug)]
pub struct DuelingCriticNet<B: Backend> {
    trunk: Linear<B>,
    fc2: Linear<B>,
    value_head: Linear<B>,
    advantage_head: Linear<B>,
    pub obs_dim: usize,
    pub act_dim: usize,
    device: B::Device,
}

impl<B: Backend> DuelingCriticNet<B> {
    pub fn new(device: &B::Device, obs_dim: usize, act_dim: usize, hidden: usize) -> Self {
        let trunk = LinearConfig::new(obs_dim + act_dim, hidden)
            .with_bias(true)
            .init(device);
        let fc2 = LinearConfig::new(hidden, hidden).with_bias(true).init(device);
        let value_head = LinearConfig::new(hidden, 1).with_bias(true).init(device);
        let advantage_head = LinearConfig::new(hidden, 1).with_bias(true).init(device);
        Self {
            trunk,
            fc2,
            value_head,
            advantage_head,
            obs_dim,
            act_dim,
            device: device.clone(),
        }
    }

    /// Shared trunk over the concatenated `(obs, action)` vector.
    fn trunk_of(&self, x: Tensor<B, 2>) -> Tensor<B, 2> {
        let h = self.trunk.forward(x);
        let h = h.tanh();
        let h = self.fc2.forward(h);
        h.tanh()
    }

    /// Dueling heads: `(V(s,a), A(s,a))`, both rank-1.
    fn heads(&self, x: Tensor<B, 2>) -> (Tensor<B, 1>, Tensor<B, 1>) {
        let h = self.trunk_of(x);
        let v = self.value_head.forward(h.clone()).squeeze_dim::<1>(1);
        let a = self.advantage_head.forward(h).squeeze_dim::<1>(1);
        (v, a)
    }

    /// Dueling Q-value of `(s, a)`:
    /// `Q(s,a) = V(s,a) + A(s,a) − A(s,0)`.
    ///
    /// The reference term is detached: it is a constant offset, not a
    /// quantity the critic loss should push around.
    pub fn q(&self, obs: Tensor<B, 2>, action: Tensor<B, 2>) -> Tensor<B, 1> {
        let n = obs.shape().dims::<2>()[0];
        let zeros = Tensor::<B, 2>::zeros([n, self.act_dim], &self.device);
        let ref_x = Tensor::cat(vec![obs.clone(), zeros], 1);
        let x = Tensor::cat(vec![obs, action], 1);
        let (v, a) = self.heads(x);
        let (_, a_ref) = self.heads(ref_x);
        v.add(a.sub(a_ref.detach()))
    }

    /// The value head alone, `V(s,a)` — used for state-value estimates.
    pub fn value_of(&self, obs: Tensor<B, 2>, action: Tensor<B, 2>) -> Tensor<B, 1> {
        let x = Tensor::cat(vec![obs, action], 1);
        let (v, _) = self.heads(x);
        v
    }
}

/// Deterministic policy head: `μ(s) = tanh(head(trunk(s)))`.
#[derive(Module, Debug)]
pub struct PolicyNet<B: Backend> {
    trunk: Linear<B>,
    fc2: Linear<B>,
    mu_head: Linear<B>,
    pub act_dim: usize,
    device: B::Device,
}

impl<B: Backend> PolicyNet<B> {
    pub fn new(device: &B::Device, obs_dim: usize, act_dim: usize, hidden: usize) -> Self {
        let trunk = LinearConfig::new(obs_dim, hidden).with_bias(true).init(device);
        let fc2 = LinearConfig::new(hidden, hidden).with_bias(true).init(device);
        let mu_head = LinearConfig::new(hidden, act_dim).with_bias(true).init(device);
        Self {
            trunk,
            fc2,
            mu_head,
            act_dim,
            device: device.clone(),
        }
    }

    /// Policy proposal in action space (`tanh`-squashed to `(-1, 1)`).
    pub fn mu(&self, obs: Tensor<B, 2>) -> Tensor<B, 2> {
        let h = self.trunk.forward(obs);
        let h = h.tanh();
        let h = self.fc2.forward(h);
        let h = h.tanh();
        self.mu_head.forward(h).tanh()
    }
}

/// Polyak-average the target critic towards the online critic.
fn polyak_critic<B: Backend>(
    target: &DuelingCriticNet<B>,
    online: &DuelingCriticNet<B>,
    tau: f32,
) -> DuelingCriticNet<B> {
    DuelingCriticNet {
        trunk: polyak_linear(&target.trunk, &online.trunk, tau),
        fc2: polyak_linear(&target.fc2, &online.fc2, tau),
        value_head: polyak_linear(&target.value_head, &online.value_head, tau),
        advantage_head: polyak_linear(&target.advantage_head, &online.advantage_head, tau),
        obs_dim: target.obs_dim,
        act_dim: target.act_dim,
        device: target.device.clone(),
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReplayItem {
    obs: Vec<f32>,
    action: Vec<f32>,
    reward: f32,
    terminated: bool,
    truncated: bool,
    next_obs: Vec<f32>,
}

/// Priorities are stored in the sum tree as integers, so they are scaled to
/// keep sub-unit priorities (a raw `as u64` cast would truncate every
/// priority below 1 to zero and the tree would always sample index 0).
const PRIORITY_SCALE: f64 = 1_000_000.0;

/// Minimal prioritized replay with a sum-tree priority index.
struct PrioritizedReplay {
    capacity: usize,
    count: usize,
    items: Vec<Option<ReplayItem>>,
    tree: Vec<u64>,
    priorities: Vec<f32>,
    alpha: f32,
}

impl PrioritizedReplay {
    fn new(cap: usize, alpha: f32) -> Self {
        Self {
            capacity: cap,
            count: 0,
            items: vec![None; cap],
            tree: vec![0u64; cap * 2],
            priorities: vec![0.0f32; cap],
            alpha,
        }
    }

    fn push(&mut self, item: ReplayItem, priority: f32) {
        let idx = self.count % self.capacity;
        let p = priority.max(self.alpha).max(1e-6);
        self.items[idx] = Some(item);
        self.priorities[idx] = p;
        self.count += 1;
        let pos = idx + self.capacity;
        self.tree[pos] = (p as f64 * PRIORITY_SCALE) as u64;
        let mut parent = pos >> 1;
        while parent > 0 {
            self.tree[parent] = self.tree[parent * 2] + self.tree[parent * 2 + 1];
            parent >>= 1;
        }
    }

    /// Re-prioritise sampled items by their (absolute) TD error.
    fn update_priorities(&mut self, indices: &[usize], errors: &[f32]) {
        for (i, &idx) in indices.iter().enumerate() {
            let Some(&e) = errors.get(i) else { break };
            let slot = idx % self.capacity;
            let p = e.abs().max(self.alpha).max(1e-6);
            self.priorities[slot] = p;
            let pos = slot + self.capacity;
            self.tree[pos] = (p as f64 * PRIORITY_SCALE) as u64;
            let mut parent = pos >> 1;
            while parent > 0 {
                self.tree[parent] = self.tree[parent * 2] + self.tree[parent * 2 + 1];
                parent >>= 1;
            }
        }
    }

    fn total(&self) -> u64 {
        self.tree[1]
    }

    fn len(&self) -> usize {
        self.items.iter().filter(|o| o.is_some()).count()
    }

    fn sample_indices(&self, n: usize, rng: &mut StdRng) -> Vec<usize> {
        let mut out = Vec::with_capacity(n);
        let total = self.total().max(1);
        let cap = self.capacity;
        for _ in 0..n {
            let v = (rng.gen::<u64>() % total) + 1;
            let mut idx = 1usize;
            while idx < cap {
                if self.tree[idx * 2] >= v {
                    idx *= 2;
                } else {
                    idx = idx * 2 + 1;
                }
            }
            out.push(idx - cap);
        }
        out
    }

    /// Sample `n` items, returning their buffer indices alongside them so
    /// the caller can re-prioritise them after the update.
    fn sample_with_indices(&mut self, n: usize, rng: &mut StdRng) -> (Vec<usize>, Vec<ReplayItem>) {
        let indices = self.sample_indices(n, rng);
        let mut out = Vec::with_capacity(indices.len());
        let mut kept = Vec::with_capacity(indices.len());
        for &idx in &indices {
            if let Some(ref item) = self.items[idx % self.capacity] {
                out.push(item.clone());
                kept.push(idx);
            }
        }
        (kept, out)
    }
}

pub struct D3qnAgent<B: AutodiffBackend<FloatElem = f32>> {
    pub net: DuelingCriticNet<B>,
    pub tgt: DuelingCriticNet<B>,
    pub policy: PolicyNet<B>,
    pub cfg: D3qnConfig,
    pub device: B::Device,
    pub optimizer: burn::optim::adaptor::OptimizerAdaptor<AdamW, DuelingCriticNet<B>, B>,
    pub policy_optimizer: burn::optim::adaptor::OptimizerAdaptor<AdamW, PolicyNet<B>, B>,
    pub rng: StdRng,
    replay: PrioritizedReplay,
    net_cfg: NetConfig,
    /// Trainer-owned observation statistics (the replay stores raw values).
    pub normalizer: RunningNormalizer,
}

impl<B: AutodiffBackend<FloatElem = f32>> D3qnAgent<B> {
    pub fn new(device: &B::Device, net_cfg: &NetConfig, cfg: &D3qnConfig, seed: u64) -> Self {
        // Seed the backend and materialize every parameter under one lock:
        // burn's ndarray backend shares one process-wide RNG and initializes
        // `Linear` weights lazily, so an unlocked build is neither
        // reproducible nor safe to clone.
        let (net, policy) = {
            let _guard = crate::net::init_lock();
            B::seed(device, seed);
            let net = DuelingCriticNet::new(device, net_cfg.obs_dim, net_cfg.action_dim, cfg.hidden);
            let policy = PolicyNet::new(device, net_cfg.obs_dim, net_cfg.action_dim, cfg.hidden);
            crate::net::materialize(&net);
            crate::net::materialize(&policy);
            (net, policy)
        };
        // The target critic starts as an exact copy of the online critic.
        // Building it from scratch gives the target an unrelated function,
        // which makes every bootstrap term meaningless.
        let tgt = polyak_critic(&net, &net, 1.0);
        let optimizer = AdamWConfig::new()
            .with_grad_clipping(Some(GradientClippingConfig::Norm(cfg.max_grad_norm)))
            .init::<B, DuelingCriticNet<B>>();
        let policy_optimizer = AdamWConfig::new()
            .with_grad_clipping(Some(GradientClippingConfig::Norm(cfg.max_grad_norm)))
            .init::<B, PolicyNet<B>>();
        Self {
            net,
            tgt,
            policy,
            cfg: *cfg,
            device: device.clone(),
            optimizer,
            policy_optimizer,
            rng: StdRng::seed_from_u64(seed),
            replay: PrioritizedReplay::new(cfg.buffer_capacity, cfg.alpha),
            net_cfg: *net_cfg,
            normalizer: RunningNormalizer::new(net_cfg.obs_dim),
        }
    }

    pub fn deterministic_action(&self, obs_norm: &[f32]) -> Vec<f32> {
        let x = Tensor::<B, 1>::from_floats(obs_norm, &self.device)
            .reshape([1, self.net_cfg.obs_dim]);
        let mu = self.policy.mu(x);
        let v: Vec<f32> = mu.into_data().to_vec::<f32>().expect("mu f32");
        v.into_iter()
            .map(|a| a.clamp(-ACTION_CLAMP, ACTION_CLAMP))
            .collect()
    }

    pub fn act(&mut self, obs_norm: &[f32]) -> (Vec<f32>, f32) {
        let x = Tensor::<B, 1>::from_floats(obs_norm, &self.device)
            .reshape([1, self.net_cfg.obs_dim]);
        let mu = self.policy.mu(x);
        let v: Vec<f32> = mu.into_data().to_vec::<f32>().expect("mu f32");
        let mut action = Vec::with_capacity(self.net_cfg.action_dim);
        for d in 0..self.net_cfg.action_dim {
            let noise: f32 = rand_distr::StandardNormal.sample(&mut self.rng);
            let a = (v[d] + self.cfg.exploration_noise * noise)
                .clamp(-ACTION_CLAMP, ACTION_CLAMP);
            action.push(a);
        }
        (action, 0.0f32)
    }

    pub fn value_single(&self, obs_norm: &[f32]) -> f32 {
        let x = Tensor::<B, 1>::from_floats(obs_norm, &self.device)
            .reshape([1, self.net_cfg.obs_dim]);
        self.state_value(x).into_scalar()
    }

    pub fn value_all(&self, obs_norm_flat: &[f32]) -> Vec<f32> {
        let n = obs_norm_flat.len() / self.net_cfg.obs_dim;
        let x = Tensor::<B, 1>::from_floats(obs_norm_flat, &self.device)
            .reshape([n, self.net_cfg.obs_dim]);
        self.state_value(x).into_data().to_vec::<f32>().expect("value f32")
    }

    /// `V(s, μ(s))` — the critic's value at the policy's own action.
    fn state_value(&self, obs: Tensor<B, 2>) -> Tensor<B, 1> {
        let mu = self.policy.mu(obs.clone());
        self.net.value_of(obs, mu)
    }

    pub fn replay_sample(&mut self, n: usize) -> Vec<ReplayItem> {
        self.replay
            .sample_with_indices(n, &mut self.rng)
            .1
    }

    pub fn replay_push(&mut self, item: ReplayItem, priority: f32) {
        self.replay.push(item, priority);
    }

    /// Number of transitions currently held in the replay buffer.
    pub fn replay_len(&self) -> usize {
        self.replay.len()
    }

    /// Greedy action under the **online** critic (the *double* part of D3QN).
    ///
    /// The candidates are the policy's proposal at `s'` plus seeded
    /// perturbations. The online critic picks the winner; the *target* critic
    /// then scores it. Using the target critic for both steps would let the
    /// target's own errors compound.
    fn greedy_action(&mut self, next_obs_host: &[f32], next_mu_host: &[f32], m: usize) -> Vec<f32> {
        let obs_dim = self.net_cfg.obs_dim;
        let act_dim = self.net_cfg.action_dim;
        let k = self.cfg.k_candidates.max(1);
        let mut cand_obs = Vec::with_capacity(m * k * obs_dim);
        let mut cand_act = Vec::with_capacity(m * k * act_dim);
        for i in 0..m {
            let obs_row = &next_obs_host[i * obs_dim..(i + 1) * obs_dim];
            for _ in 0..k {
                cand_obs.extend_from_slice(obs_row);
                for d in 0..act_dim {
                    let noise: f32 = rand_distr::StandardNormal.sample(&mut self.rng);
                    let v = (next_mu_host[i * act_dim + d] + self.cfg.exploration_noise * noise)
                        .clamp(-ACTION_CLAMP, ACTION_CLAMP);
                    cand_act.push(v);
                }
            }
        }
        let cand_obs_t = Tensor::<B, 1>::from_floats(cand_obs.as_slice(), &self.device)
            .reshape([m * k, obs_dim]);
        let cand_act_t = Tensor::<B, 1>::from_floats(cand_act.as_slice(), &self.device)
            .reshape([m * k, act_dim]);
        let q_cand = self
            .net
            .q(cand_obs_t, cand_act_t)
            .into_data()
            .to_vec::<f32>()
            .expect("q f32");

        let mut a_star = Vec::with_capacity(m * act_dim);
        for i in 0..m {
            let mut best = 0usize;
            let mut best_q = f32::NEG_INFINITY;
            for j in 0..k {
                let qv = q_cand[i * k + j];
                if qv > best_q {
                    best_q = qv;
                    best = j;
                }
            }
            let base = (i * k + best) * act_dim;
            a_star.extend_from_slice(&cand_act[base..base + act_dim]);
        }
        a_star
    }

    /// TD target for a batch: `r + γ·(1−done)·Q_target(s', argmax Q_online(s',·))`.
    fn compute_targets(
        &mut self,
        next_obs: &[f32],
        next_obs_host: &[f32],
        rewards: &[f32],
        alive: &[f32],
        m: usize,
    ) -> Tensor<B, 1> {
        let obs_dim = self.net_cfg.obs_dim;
        let act_dim = self.net_cfg.action_dim;
        let next_obs_t = Tensor::<B, 1>::from_floats(next_obs, &self.device)
            .reshape([m, obs_dim]);
        let mu = self.policy.mu(next_obs_t.clone());
        let mu_host = mu.into_data().to_vec::<f32>().expect("mu f32");
        let a_star = self.greedy_action(next_obs_host, &mu_host, m);
        let a_star_t = Tensor::<B, 1>::from_floats(a_star.as_slice(), &self.device)
            .reshape([m, act_dim]);
        let q_next = self.tgt.q(next_obs_t, a_star_t);
        let rewards_t = Tensor::<B, 1>::from_floats(rewards, &self.device);
        let alive_t = Tensor::<B, 1>::from_floats(alive, &self.device);
        rewards_t.add(q_next.mul_scalar(self.cfg.gamma).mul(alive_t))
    }

    /// TD target for a single replay item (diagnostics / tests).
    pub fn compute_target(&mut self, item: &ReplayItem) -> f32 {
        let next_obs = self.normalizer.normalize_f32(&item.next_obs);
        let alive = if item.terminated { 0.0 } else { 1.0 };
        self.compute_targets(
            &next_obs,
            &next_obs,
            &[item.reward],
            &[alive],
            1,
        )
        .into_scalar()
    }

    /// One D3QN gradient step.
    ///
    /// The on-policy `UpdateBatch` is **ignored**: D3QN is off-policy and
    /// builds its own targets from its replay buffer.
    pub fn update(&mut self, _batch: &UpdateBatch) -> UpdateStats {
        let n = self.cfg.batch_size.min(self.replay.len()).max(1);
        let (indices, replay_batch) = self.replay.sample_with_indices(n, &mut self.rng);
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
            alive.push(if item.terminated { 0.0 } else { 1.0 });
        }
        let obs = self.normalizer.normalize_batch(&obs_raw);
        let next_obs = self.normalizer.normalize_batch(&next_obs_raw);

        let obs_t = Tensor::<B, 1>::from_floats(obs.as_slice(), &self.device)
            .reshape([m, obs_dim]);
        let act_t = Tensor::<B, 1>::from_floats(act_flat.as_slice(), &self.device)
            .reshape([m, act_dim]);

        // ---- Critic target (double Q: online picks, target scores) ----
        let targets = self.compute_targets(&next_obs, &next_obs, &rewards, &alive, m);

        // ---- Critic update ----
        let preds = self.net.q(obs_t.clone(), act_t);
        let loss = (preds.clone() - targets.clone()).powf_scalar(2.0).mean();
        let grads = loss.backward();
        let grads = GradientsParams::from_grads(grads, &self.net);
        self.net = self.optimizer.step(self.cfg.lr as f64, self.net.clone(), grads);

        // ---- Policy update: maximise Q(s, μ(s)) ----
        let mu_pi = self.policy.mu(obs_t.clone());
        let q_pi = self.net.q(obs_t.clone(), mu_pi);
        let policy_loss = q_pi.mul_scalar(-1.0).mean();
        let grads_p = policy_loss.backward();
        let grads_p_params = GradientsParams::from_grads(grads_p, &self.policy);
        self.policy = self
            .policy_optimizer
            .step(self.cfg.lr as f64, self.policy.clone(), grads_p_params);

        // ---- Re-prioritise the sampled items by their TD error ----
        let td: Vec<f32> = (preds - targets)
            .into_data()
            .to_vec::<f32>()
            .expect("f32");
        self.replay.update_priorities(&indices, &td);

        // ---- Target critic: Polyak average, never re-randomise ----
        self.tgt = polyak_critic(&self.tgt, &self.net, self.cfg.tau as f32);

        UpdateStats {
            policy_loss: policy_loss.into_data().to_vec::<f32>().expect("f32")[0],
            value_loss: loss.into_data().to_vec::<f32>().expect("f32")[0],
            entropy: 0.0,
            advantage_mean: 0.0,
            advantage_std: 0.0,
        }
    }

    pub fn save_weights(&self, dir: &Path) -> Result<(), Box<dyn std::error::Error>> {
        let recorder = BinFileRecorder::<FullPrecisionSettings>::default();
        self.net.clone().save_file(dir.join("net"), &recorder)?;
        self.tgt.clone().save_file(dir.join("tgt"), &recorder)?;
        self.policy.clone().save_file(dir.join("policy"), &recorder)?;
        Ok(())
    }

    pub fn load_weights(&mut self, dir: &Path, device: &B::Device) -> Result<(), Box<dyn std::error::Error>> {
        let recorder = BinFileRecorder::<FullPrecisionSettings>::default();
        self.net = self.net.clone().load_file(dir.join("net"), &recorder, device)?;
        self.tgt = self.tgt.clone().load_file(dir.join("tgt"), &recorder, device)?;
        self.policy = self.policy.clone().load_file(dir.join("policy"), &recorder, device)?;
        Ok(())
    }
}

impl<B: AutodiffBackend<FloatElem = f32>> Algorithm<B> for D3qnAgent<B> {
    fn id(&self) -> &'static str {
        ALGO_D3QN
    }
    fn net_config(&self) -> &NetConfig {
        &self.net_cfg
    }
    fn spec(&self) -> AlgoSpec {
        AlgoSpec::D3qn { config: self.cfg }
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
    fn is_off_policy(&self) -> bool {
        true
    }
    fn observe_transition(&mut self, transition: &Transition) {
        self.replay_push(
            ReplayItem {
                obs: transition.obs.clone(),
                action: transition.action.clone(),
                reward: transition.reward,
                terminated: transition.terminated,
                truncated: transition.truncated,
                next_obs: transition.next_obs.clone(),
            },
            1.0,
        );
    }
    fn gradient_steps_per_env_step(&self) -> f32 {
        1.0
    }
    fn set_normalizer(&mut self, normalizer: RunningNormalizer) {
        self.normalizer = normalizer;
    }
    fn save_weights(&self, dir: &Path) -> Result<(), Box<dyn std::error::Error>> {
        self.save_weights(dir)
    }
    fn load_weights(
        &mut self,
        dir: &Path,
        device: &B::Device,
    ) -> Result<(), Box<dyn std::error::Error>> {
        self.load_weights(dir, device)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::{Cpu, cpu_device};
    use crate::buffer::Transition;
    use crate::net::NetConfig;

    fn agent() -> D3qnAgent<Cpu> {
        let device = cpu_device();
        let ncfg = NetConfig {
            obs_dim: 4,
            hidden: 16,
            action_dim: 4,
            log_std_init: -0.5,
        };
        D3qnAgent::new(&device, &ncfg, &D3qnConfig::default(), 1)
    }

    #[test]
    fn d3qn_act_bounded_continuous() {
        let mut a = agent();
        let (action, _lp) = a.act(&[0.1, -0.2, 0.3, 0.4]);
        assert_eq!(action.len(), 4);
        for &x in &action {
            assert!(x.abs() <= 1.0, "action out of tanh bound: {x}");
        }
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

    // ------------------------------------------------------------------
    // Behavioural acceptance tests (spec §9.3 / the D3QN milestone).
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

    fn fill_replay(a: &mut D3qnAgent<Cpu>, n: usize) {
        for i in 0..n {
            let t = i as f32 / n as f32;
            a.replay_push(
                ReplayItem {
                    obs: vec![t, t * 0.5, -t, 1.0 - t],
                    action: vec![0.2 * t, -0.3, 0.1 * t, 0.4],
                    reward: (i % 7) as f32 - 3.0,
                    terminated: false,
                    truncated: false,
                    next_obs: vec![t + 0.01, t * 0.5, -t, 1.0 - t],
                },
                1.0,
            );
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
    fn d3qn_value_all_returns_one_value_per_observation() {
        let a = agent();
        let obs: Vec<f32> = (0..12).map(|i| i as f32 * 0.1).collect();
        let values = a.value_all(&obs);
        assert_eq!(
            values.len(),
            3,
            "value_all must return one value per observation"
        );
        assert!(values.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn d3qn_value_single_matches_value_all() {
        let a = agent();
        let obs: Vec<f32> = (0..8).map(|i| i as f32 * 0.1).collect();
        let one = a.value_single(&obs[0..4]);
        let all = a.value_all(&obs);
        assert!(
            (one - all[0]).abs() < 1e-4,
            "value_single {one} != value_all[0] {}",
            all[0]
        );
    }

    #[test]
    fn d3qn_update_trains_the_q_network() {
        let mut a = agent();
        fill_replay(&mut a, 64);

        let before = param_vec(&a.net.value_head.weight);
        a.update(&empty_batch());
        let after = param_vec(&a.net.value_head.weight);

        assert!(
            max_delta(&before, &after) > 0.0,
            "D3QN Q-network did not train (gradient was severed)"
        );
    }

    #[test]
    fn d3qn_target_network_starts_as_a_copy_and_is_not_rerandomized() {
        let mut a = agent();
        assert_eq!(
            param_vec(&a.net.value_head.weight),
            param_vec(&a.tgt.value_head.weight),
            "D3QN target network must start as an exact copy of the online network"
        );

        fill_replay(&mut a, 64);
        let target_before = param_vec(&a.tgt.value_head.weight);
        a.update(&empty_batch());
        let target_after = param_vec(&a.tgt.value_head.weight);

        assert_ne!(
            target_before, target_after,
            "D3QN target network was re-randomized instead of Polyak-averaged"
        );
    }

    #[test]
    fn d3qn_td_target_uses_the_target_network_and_is_not_zero() {
        let mut a = agent();
        fill_replay(&mut a, 8);
        let item = a.replay_sample(1).remove(0);
        let target = a.compute_target(&item);
        assert!(
            target.is_finite(),
            "D3QN TD target must be finite, got {target}"
        );
        assert!(
            target.abs() > 1e-6,
            "D3QN TD target is {target}; the target network was not used"
        );
    }

    #[test]
    fn d3qn_collects_transitions_through_the_algorithm_trait() {
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
        assert_eq!(a.replay_len(), 1, "D3QN replay did not receive the transition");
    }
}
