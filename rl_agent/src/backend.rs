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
//! those must not be selected by `auto`. The probe additionally times its
//! synced per-step forwards and rejects devices at ≥0.5 ms/step — such a path
//! trains slower than the ndarray CPU backend on this tiny 19→128→128 net
//! (measured ~48 sps GPU vs ~2 000 sps CPU on the reference box).
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
use std::time::Duration;

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
///
/// The discrete GPU is preferred: on laptops wgpu's `DefaultDevice` ("high
/// power" adapter) can land on the integrated iGPU, which crawls on real
/// training loads (~15 sps vs ~1100 sps CPU here) even though it passes the
/// micro probe. `RL_GPU_DEVICE=discrete|integrated|default` overrides the
/// candidate order for diagnosis.
pub fn try_gpu_device() -> Option<WgpuDevice> {
    let order: &[WgpuDevice] = match std::env::var("RL_GPU_DEVICE").as_deref() {
        Ok("integrated") => &[
            WgpuDevice::IntegratedGpu(0),
            WgpuDevice::DefaultDevice,
            WgpuDevice::DiscreteGpu(0),
        ],
        Ok("default") => &[WgpuDevice::DefaultDevice, WgpuDevice::DiscreteGpu(0)],
        _ => &[
            WgpuDevice::DiscreteGpu(0),
            WgpuDevice::IntegratedGpu(0),
            WgpuDevice::DefaultDevice,
        ],
    };
    for device in order {
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let x: Tensor<RawGpu, 2> =
                Tensor::<RawGpu, 1>::from_floats([1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0], device)
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
/// Per-step budget for the probe's synced forwards. A *healthy* GPU finishes
/// one tiny forward + host sync in well under 0.1 ms; anything at or above
/// 0.5 ms/step is not going to beat the CPU backend on this 19→128→128
/// workload (on this box the wgpu path measures ≥1 ms/step and trains at
/// ~48 sps vs ~2 000 sps on CPU), so `auto` should never select it.
const GPU_PROBE_STEP_BUDGET: Duration = Duration::from_micros(500);

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
            fc2: LinearConfig::new(HIDDEN, HIDDEN)
                .with_bias(true)
                .init(device),
            mu_head: LinearConfig::new(HIDDEN, ACTION)
                .with_bias(true)
                .init(device),
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
            let value_loss = (net.value(x.clone()) - target.clone())
                .powf_scalar(2.0)
                .mean();
            let policy_loss = net.mean(x).powf_scalar(2.0).mean();
            let loss = value_loss + policy_loss;
            let grads = loss.backward();
            let grads = GradientsParams::from_grads(grads, &net);
            net = optim.step(1e-3, net, grads);
        }

        // Phase 2: rollout-style per-step forwards that sync to the host
        // every step (the per-call dispatch overhead that stalls `act`).
        // Time it: a device that only survives phase 1 but takes ≥0.5 ms per
        // synced forward would train slower than the CPU backend (the very
        // stall this probe exists to catch), so reject it here.
        let phase2_started = std::time::Instant::now();
        for i in 0..GPU_PROBE_STEPS {
            let f = (i as f32 + 1.0) * 0.01;
            let x = Tensor::<Gpu, 1>::from_floats([f; 19], &device).reshape([1, 19]);
            let v: f32 = net.value(x.clone()).into_scalar();
            // `mean` is [1, action_dim] — reduce before `into_scalar` (which only
            // accepts 1-element tensors; on a [1, 4] it panics with a Tensor
            // Operation Error and every `auto` run falls back to CPU).
            let m: f32 = net.mean(x.clone()).mean().into_scalar();
            if !v.is_finite() || !m.is_finite() {
                return Err("GPU probe produced non-finite values".to_string());
            }
        }
        let per_step = phase2_started.elapsed().div_f32(GPU_PROBE_STEPS as f32);
        if per_step > GPU_PROBE_STEP_BUDGET {
            return Err(format!(
                "GPU probe too slow for training: {:.1} ms/synced-forward vs \
                 {:.1} ms budget (wgpu dispatch overhead makes it slower than CPU)",
                per_step.as_secs_f64() * 1000.0,
                GPU_PROBE_STEP_BUDGET.as_secs_f64() * 1000.0
            ));
        }
        Ok(())
    }));
    std::panic::set_hook(prev_hook);
    result.map_err(|e| {
        let msg = e
            .downcast_ref::<String>()
            .cloned()
            .or_else(|| e.downcast_ref::<&str>().map(|s| s.to_string()))
            .unwrap_or_else(|| "unknown panic payload".to_string());
        format!("GPU probe panicked: {msg}")
    })?
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::{Duration, Instant};

    /// Time `probe_gpu_train` on each candidate device under a per-device
    /// wall-clock budget. `#[ignore]` because it initialises wgpu/Vulkan and
    /// can be slow or environment-dependent; run with
    /// `cargo test -p rl_agent -- --ignored --nocapture` to diagnose which
    /// physical GPU is actually trainable on a given machine. Results go to
    /// stderr so they show live even if a later device hangs.
    #[test]
    #[ignore]
    fn probe_timing_all_devices() {
        const DEVICE_BUDGET: Duration = Duration::from_secs(40);
        for device in [WgpuDevice::DefaultDevice, WgpuDevice::DiscreteGpu(0)] {
            eprintln!("probe_timing: starting {device:?}");
            let d = device.clone();
            let t = std::time::Instant::now();
            let (tx, rx) = std::sync::mpsc::channel();
            std::thread::spawn(move || {
                let _ = tx.send(probe_gpu_train(d));
            });
            match rx.recv_timeout(DEVICE_BUDGET) {
                Ok(Ok(())) => eprintln!(
                    "probe_timing: {device:?} OK in {:.2}s",
                    t.elapsed().as_secs_f64()
                ),
                Ok(Err(e)) => eprintln!(
                    "probe_timing: {device:?} FAILED after {:.2}s: {e}",
                    t.elapsed().as_secs_f64()
                ),
                Err(_) => eprintln!(
                    "probe_timing: {device:?} HUNG (>{}s)",
                    DEVICE_BUDGET.as_secs()
                ),
            }
        }
    }

    /// Isolate where the wgpu backend loses time: raw matmul throughput (no
    /// host sync) vs per-op `into_scalar` sync latency vs a real single-step
    /// forward + sync (the rollout `act` cost). `#[ignore]` like the other
    /// wgpu tests; run with `--ignored --nocapture` (use `--release` for
    /// honest CPU-side op-submission timings).
    #[test]
    #[ignore]
    fn gpu_op_latency() {
        let device = WgpuDevice::DiscreteGpu(0);
        let a = Tensor::<RawGpu, 2>::ones([128, 128], &device);

        // 1) Raw matmul throughput: 200 fused matmuls, single sync at the end.
        let t = Instant::now();
        let mut y = a.clone();
        for _ in 0..200 {
            y = y.matmul(a.clone());
        }
        let _hold: f32 = y.sum().into_scalar();
        eprintln!(
            "gpu_lat: 200x[128,128] matmuls (1 sync total): {:.3}s",
            t.elapsed().as_secs_f64()
        );

        // 2) Per-op host-sync latency: the cost paid on every rollout step.
        let t = Instant::now();
        for _ in 0..200 {
            let _v: f32 = a.clone().sum().into_scalar();
        }
        let e = t.elapsed();
        eprintln!(
            "gpu_lat: 200x sum+into_scalar syncs: {:.3}s ({:.2} ms/op)",
            e.as_secs_f64(),
            e.as_secs_f64() * 1000.0 / 200.0
        );

        // 3) Real net forward + two syncs per call: the rollout `act()` shape.
        let net = ProbeNet::<RawGpu>::new(&device);
        let x = Tensor::<RawGpu, 1>::from_floats([0.5f32; 19], &device).reshape([1, 19]);
        let t = Instant::now();
        for _ in 0..200 {
            let _v: f32 = net.value(x.clone()).into_scalar();
            let _m: f32 = net.mean(x.clone()).mean().into_scalar();
        }
        let e = t.elapsed();
        eprintln!(
            "gpu_lat: 200x net fwd + 2 syncs: {:.3}s ({:.2} ms/step)",
            e.as_secs_f64(),
            e.as_secs_f64() * 1000.0 / 200.0
        );
    }
}
