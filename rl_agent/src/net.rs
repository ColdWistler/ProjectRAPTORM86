//! Feed-forward actor-critic network with a tanh-squashed Gaussian policy.
//!
//! The shared trunk is a two-hidden-layer MLP with tanh activations. It fans
//! out into a mean head (action dimension), a value head (scalar), and a
//! learnable per-channel `log_std` parameter. The policy is a
//! diagonal-Gaussian with tanh squashing: `a = tanh(x), x ~ N(μ, σ)`.
//!
//! Sampling happens on the **host** (seeded `StdRng`) from the device-
//! computed mean and the host copy of `log_std`, so rollouts are
//! reproducible. The log-density used at update time on the device
//! ([`ActorCritic::log_prob`]) and the host-side density
//! ([`gaussian_log_prob_tanh`]) implement *exactly* the same formula so the
//! PPO importance ratio stays consistent.

use burn::module::Param;
use burn::nn::{Linear, LinearConfig};
use burn::prelude::*;
use serde::{Deserialize, Serialize};
use std::sync::Mutex;

/// Serializes backend seeding + lazy parameter materialization.
///
/// burn's ndarray backend keeps its RNG seed in a process-wide static, and
/// `Linear` parameters initialize lazily on first use. Without this lock two
/// agents created on different threads could consume the shared RNG stream
/// between one agent's `seed` and its first forward pass, breaking
/// seed-reproducibility.
static INIT_LOCK: Mutex<()> = Mutex::new(());

/// Clamp applied to sampled/action values so `atanh` stays finite.
pub const ACTION_CLAMP: f32 = 0.999_999;

/// `ln(2π)`, shared by the Gaussian entropy and log-density formulas.
const LN_2PI: f32 = 1.837_877_1;

/// Architecture hyper-parameters for the actor-critic.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct NetConfig {
    /// Observation dimension (19 for the noisy avionics bus).
    pub obs_dim: usize,
    /// Width of both shared hidden layers.
    pub hidden: usize,
    /// Action dimension (4).
    pub action_dim: usize,
    /// Initial value of the learnable per-channel `log_std` parameter.
    pub log_std_init: f32,
}

impl Default for NetConfig {
    fn default() -> Self {
        Self {
            obs_dim: crate::AVIONICS_OBS_DIM,
            hidden: 128,
            action_dim: crate::ACTION_DIM,
            log_std_init: -0.5,
        }
    }
}

/// Actor-critic network.
///
/// `#[derive(Module)]` registers the `Linear` layers and the `log_std`
/// parameter for gradient tracking; the plain `usize` fields are metadata.
#[derive(Module, Debug)]
pub struct ActorCritic<B: Backend> {
    fc1: Linear<B>,
    fc2: Linear<B>,
    mu_head: Linear<B>,
    value_head: Linear<B>,
    log_std: Param<Tensor<B, 1>>,
    /// Observation dimension (metadata, not a parameter).
    pub obs_dim: usize,
    /// Action dimension (metadata, not a parameter).
    pub action_dim: usize,
}

impl<B: Backend> ActorCritic<B> {
    /// Build the network on `device` from `cfg`.
    ///
    /// Reproducible *and* thread-safe: takes [`INIT_LOCK`], reseeds the
    /// backend, builds, then forces every lazy parameter to materialize
    /// before releasing the lock. Callers get weights that depend only on
    /// `seed` regardless of concurrent agent creation.
    pub fn new(device: &B::Device, cfg: &NetConfig, seed: u64) -> Self {
        let _guard = INIT_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        B::seed(device, seed);
        let net = Self::build(device, cfg);
        net.materialize();
        net
    }

    /// Build the network without seeding or forcing materialization.
    fn build(device: &B::Device, cfg: &NetConfig) -> Self {
        let fc1 = LinearConfig::new(cfg.obs_dim, cfg.hidden).with_bias(true).init(device);
        let fc2 = LinearConfig::new(cfg.hidden, cfg.hidden).with_bias(true).init(device);
        let mu_head = LinearConfig::new(cfg.hidden, cfg.action_dim).with_bias(true).init(device);
        let value_head = LinearConfig::new(cfg.hidden, 1).with_bias(true).init(device);

        let log_std_vec = vec![cfg.log_std_init; cfg.action_dim];
        let log_std_data =
            Tensor::<B, 1>::from_floats(log_std_vec.as_slice(), device).into_data();
        let log_std = Param::from_data(log_std_data, device);

        Self {
            fc1,
            fc2,
            mu_head,
            value_head,
            log_std,
            obs_dim: cfg.obs_dim,
            action_dim: cfg.action_dim,
        }
    }

    /// Force every lazily-initialized parameter to materialize now.
    fn materialize(&self) {
        let _ = self.fc1.weight.val();
        let _ = self.fc1.bias.as_ref().map(|b| b.val());
        let _ = self.fc2.weight.val();
        let _ = self.fc2.bias.as_ref().map(|b| b.val());
        let _ = self.mu_head.weight.val();
        let _ = self.mu_head.bias.as_ref().map(|b| b.val());
        let _ = self.value_head.weight.val();
        let _ = self.value_head.bias.as_ref().map(|b| b.val());
        let _ = self.log_std.val();
    }

    /// Shared feature trunk: `tanh(fc2(tanh(fc1(x))))`.
    pub fn shared(&self, x: Tensor<B, 2>) -> Tensor<B, 2> {
        let h = self.fc1.forward(x);
        let h = h.tanh();
        let h = self.fc2.forward(h);
        h.tanh()
    }

    /// Gaussian mean (pre-tanh) from pre-computed shared features.
    pub fn mean_from_shared(&self, h: Tensor<B, 2>) -> Tensor<B, 2> {
        self.mu_head.forward(h)
    }

    /// State value `V(s)` from pre-computed shared features.
    pub fn value_from_shared(&self, h: Tensor<B, 2>) -> Tensor<B, 1> {
        self.value_head.forward(h).squeeze_dim::<1>(1)
    }

    /// Gaussian mean (pre-tanh) for a batch of observations.
    pub fn mean(&self, x: Tensor<B, 2>) -> Tensor<B, 2> {
        self.mean_from_shared(self.shared(x))
    }

    /// State value for a batch of observations (rank-1).
    pub fn value(&self, x: Tensor<B, 2>) -> Tensor<B, 1> {
        self.value_from_shared(self.shared(x))
    }

    /// Deterministic greedy action `tanh(μ)` for a batch.
    pub fn deterministic_action(&self, x: Tensor<B, 2>) -> Tensor<B, 2> {
        self.mean(x).tanh()
    }

    /// Broadcast `log_std` to `[batch, action_dim]`.
    fn log_std_2d(&self, batch: usize) -> Tensor<B, 2> {
        self.log_std
            .val()
            .unsqueeze_dim::<2>(0)
            .expand([batch, self.action_dim])
    }

    /// Host copy of the per-channel `log_std` parameters.
    pub fn log_std_host(&self) -> Vec<f32> {
        self.log_std
            .val()
            .into_data()
            .to_vec::<f32>()
            .expect("log_std must be f32 data")
    }

    /// Per-sample Gaussian entropy `Σ_a (log σ + 0.5·ln(2π·e))`.
    pub fn entropy(&self, x: Tensor<B, 2>) -> Tensor<B, 1> {
        let batch = x.shape().dims::<2>()[0];
        self.log_std_2d(batch)
            .add_scalar(0.5 * (LN_2PI + 1.0))
            .sum_dim(1)
            .squeeze_dim::<1>(1)
    }

    /// Tensor log-density `log π(a | s)` (tanh-squashed diagonal Gaussian).
    pub fn log_prob(&self, x: Tensor<B, 2>, action: Tensor<B, 2>) -> Tensor<B, 1> {
        let batch = x.shape().dims::<2>()[0];
        let h = self.shared(x);
        let mu = self.mean_from_shared(h);
        let log_std_2d = self.log_std_2d(batch);

        // atanh(a), with `a` clamped so it is numerically inside (-1, 1).
        let a = action.clamp(-ACTION_CLAMP, ACTION_CLAMP);
        let one = Tensor::ones_like(&a);
        let atanh = ((one.clone() + a.clone()) / (one.clone() - a.clone()))
            .log()
            .mul_scalar(0.5);

        let variance = log_std_2d.clone().mul_scalar(2.0).exp().add_scalar(1e-6);
        let diff = atanh - mu;

        let logp_per_dim = diff
            .powf_scalar(2.0)
            .mul_scalar(-0.5)
            .div(variance)
            .sub(log_std_2d)
            .sub_scalar(0.5 * LN_2PI)
            .sub((one.clone() + a.clone()).log())
            .sub((one - a).log());

        logp_per_dim.sum_dim(1).squeeze_dim::<1>(1)
    }
}

/// Host-side log-density of a tanh-squashed Gaussian.
///
/// Must stay bit-for-bit consistent with [`ActorCritic::log_prob`]:
/// `x = atanh(a)`, `log p = −½(x−μ)²/(σ²+1e-6) − log σ − ½ln(2π) − ln(1+a) − ln(1−a)`.
pub fn gaussian_log_prob_tanh(action: f32, mean: f32, log_std: f32) -> f32 {
    let a = action.clamp(-ACTION_CLAMP, ACTION_CLAMP);
    let x = 0.5 * ((1.0 + a).ln() - (1.0 - a).ln());
    let variance = (2.0 * log_std).exp() + 1e-6;
    -0.5 * (x - mean).powi(2) / variance - log_std - 0.5 * LN_2PI
        - (1.0 + a).ln()
        - (1.0 - a).ln()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::Cpu;

    fn cpu() -> burn::backend::ndarray::NdArrayDevice {
        crate::backend::cpu_device()
    }

    fn net() -> ActorCritic<Cpu> {
        let device = cpu();
        let cfg = NetConfig {
            obs_dim: 4,
            hidden: 8,
            action_dim: 2,
            log_std_init: -1.0,
        };
        ActorCritic::new(&device, &cfg, 0)
    }

    #[test]
    fn forward_shapes_are_correct() {
        let device = cpu();
        let n = net();
        let x = Tensor::<Cpu, 2>::ones([5, 4], &device);
        let mu = n.mean(x.clone());
        let v = n.value(x.clone());
        let a = n.deterministic_action(x.clone());
        let ent = n.entropy(x);
        assert_eq!(mu.shape().dims::<2>(), [5, 2]);
        assert_eq!(v.shape().dims::<1>(), [5]);
        assert_eq!(a.shape().dims::<2>(), [5, 2]);
        assert_eq!(ent.shape().dims::<1>(), [5]);
        let v_host: Vec<f32> = v.into_data().to_vec().unwrap();
        assert!(v_host.iter().all(|&x| x.is_finite()));
        let a_host: Vec<f32> = a.into_data().to_vec().unwrap();
        assert!(a_host.iter().all(|&a| a.abs() < 1.0), "tanh must squash actions");
    }

    #[test]
    fn host_and_tensor_log_prob_agree() {
        let device = cpu();
        let n = net();
        let x = Tensor::<Cpu, 1>::from_floats(
            [0.1f32, -0.2, 0.3, 0.4, -0.1, 0.2, 0.5, -0.3],
            &device,
        )
        .reshape([2, 4]);
        let a = Tensor::<Cpu, 1>::from_floats(
            [0.3f32, -0.4, 0.7, 0.2],
            &device,
        )
        .reshape([2, 2]);
        let tensor_lp = n.log_prob(x, a.clone());
        let a_host: Vec<f32> = a.into_data().to_vec().unwrap();
        let mu_host: Vec<f32> = n
            .mean(Tensor::<Cpu, 1>::from_floats([0.1f32, -0.2, 0.3, 0.4, -0.1, 0.2, 0.5, -0.3], &device).reshape([2, 4]))
            .into_data()
            .to_vec()
            .unwrap();
        let log_std = n.log_std_host();
        let batch = 2;
        let mut host_lp = vec![0.0f32; batch];
        for i in 0..batch {
            for d in 0..2 {
                host_lp[i] += gaussian_log_prob_tanh(
                    a_host[i * 2 + d],
                    mu_host[i * 2 + d],
                    log_std[d],
                );
            }
        }
        let tlp: Vec<f32> = tensor_lp.into_data().to_vec().unwrap();
        for i in 0..batch {
            let t = tlp[i];
            let h = host_lp[i];
            assert!(
                (t - h).abs() < 1e-5,
                "tensor {t:.6} != host {h:.6} at sample {i}"
            );
        }
    }

    #[test]
    fn save_load_round_trip_preserves_weights() {
        use burn::record::{BinFileRecorder, FullPrecisionSettings};
        let device = cpu();
        let n = net();
        let x = Tensor::<Cpu, 1>::from_floats([0.5f32; 8], &device).reshape([2, 4]);
        let before = n.value(x.clone());
        let before_host: Vec<f32> = before.into_data().to_vec().unwrap();

        let dir = std::env::temp_dir().join("rl_agent_net_roundtrip");
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let recorder = BinFileRecorder::<FullPrecisionSettings>::default();
        n.clone()
            .save_file(dir.join("net"), &recorder)
            .expect("save should succeed");

        let loaded = ActorCritic::<Cpu>::new(&device, &NetConfig {
            obs_dim: 4,
            hidden: 8,
            action_dim: 2,
            log_std_init: -1.0,
        }, 0)
        .load_file(dir.join("net"), &recorder, &device)
        .expect("load should succeed");

        let after = loaded.value(x);
        let after_host: Vec<f32> = after.into_data().to_vec().unwrap();
        assert_eq!(before_host, after_host);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn backward_step_updates_params() {
        use burn::optim::{AdamWConfig, GradientsParams, Optimizer};

        let device = cpu();
        let mut n = net();
        let log_std_before = n.log_std_host();
        let mut optim = AdamWConfig::new().init::<Cpu, ActorCritic<Cpu>>();
        let x = Tensor::<Cpu, 1>::from_floats([0.5f32; 8], &device).reshape([2, 4]);
        let a = Tensor::<Cpu, 1>::from_floats([0.1f32; 4], &device).reshape([2, 2]);
        let target: Tensor<Cpu, 1> = Tensor::from_floats([1.0f32, 0.0], &device);
        let z = Tensor::<Cpu, 1>::from_floats([0.3f32; 8], &device).reshape([2, 4]);
        let value_loss = (n.value(z) - target).powf_scalar(2.0).mean();
        let policy_loss = n.log_prob(x, a).mul_scalar(-1.0).mean();
        let loss = policy_loss + value_loss;
        let grads = loss.backward();
        let grads = GradientsParams::from_grads(grads, &n);
        n = optim.step(1e-3, n, grads);
        let log_std_after = n.log_std_host();
        assert_ne!(log_std_before, log_std_after);
    }

    #[test]
    fn normalization_does_not_break_shapes() {
        use crate::normalize::RunningNormalizer;
        let dim = 4;
        let mut norm = RunningNormalizer::new(dim);
        for _ in 0..10 {
            norm.observe(&[1.0, 2.0, 3.0, 4.0]);
        }
        let batch = norm.normalize_batch(&[1.0, 2.0, 3.0, 4.0, 0.5, 1.0, 1.5, 2.0]);
        assert_eq!(batch.len(), 8);
    }
}