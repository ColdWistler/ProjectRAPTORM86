//! Runtime-selected tensor backends for the RL agent.
//!
//! Both a CPU backend (burn's `ndarray` backend, no GPU required) and a GPU
//! backend (burn's `wgpu` backend, Vulkan compute) are compiled into the same
//! binary. The user picks one at runtime (`--backend cpu|gpu|auto`), and in
//! `auto` mode a tiny tensor op probes each candidate GPU device — falling
//! back to CPU if the GPU is unavailable, while `gpu` mode errors instead.
//!
//! Every training routine is generic over `B: AutodiffBackend`, so this
//! module is the only place that names concrete backend types.

use burn::backend::autodiff::Autodiff;
use burn::backend::ndarray::NdArrayDevice;
use burn::backend::wgpu::WgpuDevice;
use burn::backend::{NdArray, Wgpu};
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