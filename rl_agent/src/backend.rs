//! Runtime-selected tensor backends for the RL agent.
//!
//! Both a CPU backend (burn's `ndarray` backend, no GPU required) and a GPU
//! backend (burn's `wgpu` backend, Vulkan compute) are compiled into the same
//! binary. The user picks one at runtime (`--backend cpu|gpu|auto`), and in
//! `auto` mode a tiny tensor op probes each candidate GPU device — falling
//! back to CPU if the GPU is unavailable, while `gpu` mode errors instead.
//!
//! `auto` goes further than a micro matmul: a device that survives the tiny
//! op must also pass [`probe_gpu_train`], which runs the real training
//! workload (autodiff updates + rollout-style synced forwards) under a
//! wall-clock timeout imposed by the caller. Some wgpu/Vulkan runtimes pass a
//! micro matmul yet stall in autodiff/fusion codegen or per-call dispatch;
//! those must not be selected by `auto`.
//!
//! Every training routine is generic over `B: AutodiffBackend`, so this
//! module is the only place that names concrete backend types.

use burn::backend::autodiff::Autodiff;
use burn::backend::ndarray::NdArrayDevice;
use burn::backend::wgpu::WgpuDevice;
use burn::backend::{NdArray, Wgpu};
use burn::nn::{Linear, LinearConfig};
use burn::optim::{AdamWConfig, GradientsParams, Optimizer};
use burn::prelude::*;

/// CPU backend: autodiff layered on the ndarray backend.
pub type Cpu = Autodiff<NdArray>;

/// GPU backend: autodiff layered on the wgpu (Vulkan) backend.
pub type Gpu = Autodiff<Wgpu>;

/// Raw wgpu backend — no autodiff graph, used only to probe GPU devices.
pub type RawGpu = Wgpu;

/// Which concrete backend the user asked for.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BackendKind {
    /// `Autodiff<NdArray>` on `NdArrayDevice::Cpu`.
    Cpu,
    /// `Autodiff<Wgpu>` on the first working GPU.
    Gpu,
}

impl BackendKind {
    /// Lowercase CLI token for this backend.
    pub fn as_str(&self) -> &'static str {
        match self {
            BackendKind::Cpu => "cpu",
            BackendKind::Gpu => "gpu",
        }
    }
}

/// Parse the `--backend` CLI value. Unknown values fall back to `cpu`.
pub fn parse_backend(s: &str) -> BackendKind {
    match s.to_ascii_lowercase().as_str() {
        "gpu" => BackendKind::Gpu,
        _ => BackendKind::Cpu,
    }
}

/// Probe candidate GPU devices with a tiny tensor op under `catch_unwind`.
/// Returns the first device that completes a small matmul + reduction
/// with finite output, or `None` when no GPU works.
pub fn try_gpu_device() -> Option<WgpuDevice> {
    for device in [WgpuDevice::DefaultDevice, WgpuDevice::DiscreteGpu(0)] {
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let x: Tensor<RawGpu, 2> =
                Tensor::<RawGpu, 1>::from_floats([1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0], &device)
                    .reshape([2, 3]);
            let y = x.clone().matmul(x.transpose());
            let value: f32 = y.sum().into_scalar();
            (value.is_finite(), device.clone())
        }));
        if let Ok((ok, device)) = result {
            if ok {
                return Some(device);
            }
        }
    }
    None
}

/// CPU device used for the ndarray backend.
pub fn cpu_device() -> NdArrayDevice {
    NdArrayDevice::Cpu
}

// ── Functional GPU probe ─────────────────────────────────────────────

/// Autodiff updates the probe runs to exercise the compile/fusion path.
const GPU_PROBE_UPDATES: usize = 16;
/// Rollout-style per-step forwards the probe runs (each syncs to the host,
/// mirroring `act` during a rollout).
const GPU_PROBE_STEPS: usize = 128;

/// Tiny feed-forward net used *only* by the `auto` backend probe.
///
/// It deliberately **does not** take `net::INIT_LOCK`: the probe runs on a
/// separate thread that may be abandoned if it hangs (the caller times it
/// out), and dying while holding the global init lock would deadlock the CPU
/// fallback that runs next on the main thread.
///
/// The dimensions mirror the real [`ActorCritic`](crate::net::ActorCritic)
/// (19 inputs → 128 → 128 → action/value) because the wgpu/Vulkan stall seen
/// on some machines only shows up with the real workload — a toy net that is
/// 8→16→1 passes where a real one crawls.
#[derive(Module, Debug)]
struct ProbeNet<B: Backend> {
    fc1: Linear<B>,
    fc2: Linear<B>,
    mu_head: Linear<B>,
    value_head: Linear<B>,
}

impl<B: Backend> ProbeNet<B> {
    fn new(device: &B::Device) -> Self {
        const OBS: usize = 19;
        const HIDDEN: usize = 128;
        const ACTION: usize = 4;
        Self {
            fc1: LinearConfig::new(OBS, HIDDEN).with_bias(true).init(device),
            fc2: LinearConfig::new(HIDDEN, HIDDEN).with_bias(true).init(device),
            mu_head: LinearConfig::new(HIDDEN, ACTION).with_bias(true).init(device),
            value_head: LinearConfig::new(HIDDEN, 1).with_bias(true).init(device),
        }
    }

    /// Shared trunk: `tanh(fc2(tanh(fc1(x))))`.
    fn shared(&self, x: Tensor<B, 2>) -> Tensor<B, 2> {
        let h = self.fc1.forward(x).tanh();
        self.fc2.forward(h).tanh()
    }

    /// Policy mean (pre-tanh) and state value for `x`.
    fn mean(&self, x: Tensor<B, 2>) -> Tensor<B, 2> {
        self.mu_head.forward(self.shared(x))
    }

    fn value(&self, x: Tensor<B, 2>) -> Tensor<B, 1> {
        self.value_head.forward(self.shared(x)).squeeze_dim::<1>(1)
    }
}

/// Functional GPU probe: the *actual* trainer workload, not a micro matmul.
///
/// wgpu/Vulkan on some machines survives a one-shot matmul yet stalls in
/// autodiff/fusion kernel codegen or per-call dispatch — exactly the failure
/// mode here where `--backend auto` produced `running` runs that never wrote
/// a telemetry record (or crawled at <1 step/s through a real-sized net).
/// This probe runs a few real autodiff updates followed by a rollout-style
/// sequence of synced single-transition forwards through the real 19→128→128
/// architecture; a device in that broken state will hang (or panic) here.
/// The caller runs it under a wall-clock timeout and falls back to CPU.
pub fn probe_gpu_train(device: WgpuDevice) -> Result<(), String> {
    // Silence the default panic hook for the duration of the probe: a broken
    // wgpu runtime dumps a long Tensor Operation Error / IR trace here, which
    // would flood the training log. The diagnostic survives in the returned
    // `Err` payload and is printed as a one-line warning by the caller.
    let prev_hook = std::panic::take_hook();
    std::panic::set_hook(Box::new(|_| {}));
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let mut net = ProbeNet::<Gpu>::new(&device);
        let mut optim = AdamWConfig::new().init::<Gpu, ProbeNet<Gpu>>();
        let target: Tensor<Gpu, 1> = Tensor::from_floats([0.5f32], &device);

        // Phase 1: several full autodiff update steps (codegen + autodiff +
        // optimizer path used by every PPO iteration).
        for i in 0..GPU_PROBE_UPDATES {
            let f = (i as f32 + 1.0) * 0.05;
            let x = Tensor::<Gpu, 1>::from_floats([f; 19], &device).reshape([1, 19]);
            let value_loss = (net.value(x.clone()) - target.clone()).powf_scalar(2.0).mean();
            let policy_loss = net.mean(x).powf_scalar(2.0).mean();
            let loss = value_loss + policy_loss;
            let grads = loss.backward();
            let grads = GradientsParams::from_grads(grads, &net);
            net = optim.step(1e-3, net, grads);
        }

        // Phase 2: rollout-style per-step forwards that sync to the host
        // every step (the per-call dispatch overhead that stalls `act`).
        for i in 0..GPU_PROBE_STEPS {
            let f = (i as f32 + 1.0) * 0.01;
            let x = Tensor::<Gpu, 1>::from_floats([f; 19], &device).reshape([1, 19]);
            let v: f32 = net.value(x.clone()).into_scalar();
            let m: f32 = net.mean(x).into_scalar();
            if !v.is_finite() || !m.is_finite() {
                return Err("GPU probe produced non-finite values".to_string());
            }
        }
        Ok(())
    }));
    std::panic::set_hook(prev_hook);
    result
        .map_err(|e| {
            let msg = e
                .downcast_ref::<String>()
                .cloned()
                .or_else(|| e.downcast_ref::<&str>().map(|s| s.to_string()))
                .unwrap_or_else(|| "unknown panic payload".to_string());
            format!("GPU probe panicked: {msg}")
        })?
}