# `rl_agent` Guide

`rl_agent` is a **pure-Rust, GPU-optimized PPO trainer** for the
`flight_core` flight-dynamics engine. It trains the **avionics
attitude-command environment** (`AvionicsEnvironment`): the agent emits
roll/pitch setpoints, a yaw-rate command, and a throttle command, and a
simulated PID flight controller inside `flight_core` turns those into
servo/ESC outputs. The agent sees the **noisy 19-channel sensor bus**
(gyro/accel/GPS/baro/airspeed/battery).

Everything is built on [burn](https://burn.dev) v0.21.0 — tensors and
autodiff — with both a CPU backend (`ndarray`) and a GPU backend
(`wgpu`/Vulkan) compiled into the same binary and selected at **runtime**.
There is no compile-time feature gating: `--backend auto` probes the GPU
and transparently falls back to CPU.

---

## 1. Quick start

```bash
# Train from the repo root (aircraft.toml is found automatically):
cargo run -p rl_agent --release -- --mode train --backend auto

# Or a quick CPU smoke run:
cargo run -p rl_agent -- --mode train --backend cpu --iterations 3 --rollout 128

# Evaluate a checkpoint with deterministic actions:
cargo run -p rl_agent --release -- --mode eval --backend auto --checkpoint checkpoints
```

`--help` prints the full CLI.

## 2. CLI reference (`cargo run -p rl_agent -- --help`)

| Flag | Default | Meaning |
|---|---|---|
| `--mode train\|eval` | `train` | Train, or evaluate a checkpointed policy |
| `--backend cpu\|gpu\|auto` | `auto` | Tensor backend at runtime (see §3) |
| `--iterations N` | `1000` | PPO iterations (one rollout + update each) |
| `--rollout N` | `2048` | Env steps collected per iteration |
| `--env-max-steps N` | `2000` | Per-episode step budget (truncation) |
| `--env-dt SECONDS` | `0.0333` | Physics step size (1/30 s) |
| `--hidden N` | `128` | Width of both shared hidden layers |
| `--log-std-init F` | `-0.5` | Initial per-channel policy `log_std` |
| `--epochs N` | `10` | PPO epochs per update |
| `--minibatch N` | `64` | Transitions per gradient step |
| `--lr F` | `3e-4` | AdamW learning rate |
| `--gamma F` | `0.99` | Discount factor |
| `--lambda F` | `0.95` | GAE trace-decay |
| `--clip F` | `0.2` | PPO ratio clip bound ε |
| `--value-coef F` | `0.5` | Value-loss coefficient |
| `--entropy-coef F` | `0.0` | Entropy bonus coefficient |
| `--max-grad-norm F` | `0.5` | Global gradient-norm clip |
| `--seed N` | `42` | Weights init + host sampler RNG |
| `--save-every N` | `100` | Checkpoint every N iterations (0 = never) |
| `--print-every N` | `1` | Log per-iteration stats every N (0 = silent) |
| `--eval-every N` | `100` | Deterministic eval every N iterations (0 = never) |
| `--eval-episodes N` | `3` | Episodes per eval run |
| `--config PATH` | `aircraft.toml` | Aircraft config for `flight_core` |
| `--checkpoint DIR` | `checkpoints` | Checkpoint directory |

## 3. Backends (`--backend cpu | gpu | auto`)

| Value | Behavior |
|---|---|
| `cpu` | `Autodiff<NdArray>` on `NdArrayDevice::Cpu`. No GPU required; fast to start and fully deterministic with a seed. |
| `gpu` | `Autodiff<Wgpu>` (Vulkan compute). **Errors** if no working GPU is found or the GPU panics mid-run. |
| `auto` (default) | Probes `[WgpuDevice::DefaultGpu, WgpuDevice::DiscreteGpu(0)]` with a tiny matmul+sum under `catch_unwind`; uses the first device that yields a finite result, else falls back to CPU (with a warning). If GPU execution panics later, the **entire run** restarts on CPU. |

Both backends are compiled into one binary — switching is a flag, not a
rebuild. The only place concrete backend types are named is
`rl_agent/src/backend.rs`; every training routine is generic over
`B: AutodiffBackend`.

## 4. Training target

`flight_core::AvionicsEnvironment` with `EnvConfig { dt, max_steps, .. }`
(the file from `--config` supplies the rest). Observation layout
(19 channels, noisy):

```text
 0  gyro_p (rad/s)         10 baro_altitude (m)
 1  gyro_q (rad/s)         11 airspeed_indicated (m/s)
 2  gyro_r (rad/s)         12 battery_voltage (V)
 3  accel_x (m/s²)         13 battery_capacity_remaining_pct (%)
 4  accel_y (m/s²)         14 actual_elevator_deg (°)
 5  accel_z (m/s²)         15 actual_aileron_deg (°)
 6  gps_north (m, NED)     16 actual_rudder_deg (°)
 7  gps_east (m, NED)      17 actual_esc_output (0..1)
 8  gps_altitude (m)       18 sim_time (s)
 9  gps_fix_quality (0..3)
```

Actions are 4-dim **tanh-squashed** and mapped to attitude commands:

```rust
roll_cmd    = tanh(a0) * 0.6        // rad   (±45° clamp: 0.785 rad)
pitch_cmd   = tanh(a1) * 0.4        // rad
yaw_rate_cmd= tanh(a2) * 0.8        // rad/s
throttle    = 0.5 * (tanh(a3) + 1)  // 0..1
```

All components stay inside the flight controller's `±45°` setpoint clamps.
The service attempts attitude control toward a **target altitude**
(1000 m) and **target airspeed** (60 m/s) drawn from `aircraft.toml` /
`EnvConfig`; evaluation metrics compare the noisy `baro_altitude` and
`airspeed_indicated` channels (indices 10/11) against those targets.

Episodes end by **termination** (crash — the env's `crash_penalty()` is
added to the reward) or **truncation** (step budget exhausted).

## 5. Algorithm and components

Training implements PPO-Clip (Schulman et al., 2017) with per-minibatch
advantage whitening:

```text
L = −E[ min(r·Â, clip(r, 1±ε)·Â) ] + c_v·E[(V−R)²] − c_e·E[H]
r = exp(log π_θ(a|s) − log π_old(a|s))
```

| Piece | File | Notes |
|---|---|---|
| Network | `src/net.rs` | 2-hidden-layer tanh MLP (`fc1`/`fc2`) → `mu_head` + `value_head`, plus a learnable per-channel `log_std` `Param`. Diagonal Gaussian, tanh-squashed. |
| Sampling | `src/ppo.rs` | Action sampled on the **host** with a seeded `StdRng` from the device-computed mean, so rollouts are reproducible. The tensor `log_prob` and the host `gaussian_log_prob_tanh` implement the **identical formula** (checked to 1e-5 in tests). |
| Buffer | `src/buffer.rs` | Flat `f32` arrays: raw obs, actions, log-probs, rewards, `terminated`/`truncated` flags. |
| GAE | `src/gae.rs` | Host-side generalized advantage estimation; `terminated` zeroes the next value **and** masks the accumulator. |
| Normalizer | `src/normalize.rs` | Welford running mean/variance observed during rollout; applied with **final** stats at update/eval time; clipped to ±5. |
| Optimizer | `src/ppo.rs` | AdamW with `GradientClippingConfig::Norm(max_grad_norm)` baked into the optimizer config. |
| Trainer | `src/trainer.rs` | Generic `run<B>` loop: rollout → values → GAE → update → eval/checkpoint. |

**Rollout flow** (per iteration):

1. For `rollout_steps` transitions: normalize the current obs, `agent.act()`
   produces `(action, log_prob)`, the env steps, reward += `crash_penalty()`
   on termination, transition is pushed (with the **pre-reset** observation
   as `final_obs`), and on `terminated`/`truncated` the env resets and the
   normalizer observes the fresh initial state.
2. Values are predicted over the whole normalized batch; the last value is
   `0` if the final rollout step terminated, else V(final obs).
3. GAE advantages + returns; one `agent.update` over shuffled minibatches.
4. Periodic deterministic eval and checkpoint save.

## 6. Determinism

- Same `--seed` ⇒ same backend init (`B::seed`) and same host sampler
  stream ⇒ byte-identical normalizer stats and identical trained policies
  (asserted in `same_seed_reproduces_identical_training`).
- The sensor bus is deterministic for a fixed seed (`xorshift64*` with a
  fixed default), so two fresh environments reproduce identically — which is
  what makes the reproducibility test meaningful.
- `cpu` and `gpu` won't give *bit-identical* numbers to each other (different
  floating-point kernels), but each is reproducible within its backend.

## 7. Checkpoints

Saved to `--checkpoint DIR` every `save_every` iterations (and at the end):

| File | Contents |
|---|---|
| `net` (bin) | Policy + critic weights, `BinFileRecorder<FullPrecisionSettings>` (FP32, portable across backends). |
| `net.json` | `NetConfig` (obs/hidden/action dims, log_std init). |
| `ppo.json` | `PpoConfig`. |
| `trainer.json` | `TrainerConfig`. |
| `normalizer.json` | Running Welford stats + clip bound. |

Optimizer state is **not** persisted in v1 — on eval / resumption from a
checkpoint a fresh AdamW optimizer is built from `ppo.json`. Configure
`save_every: 0` / `eval_every: 0` / `print_every: 0` to disable those
`.json` write paths entirely in tests/harnesses.

Eval mode loads the checkpoint and runs `--eval-episodes` deterministic
episodes, reporting mean steps, altitude error, airspeed error, mean reward,
and crash count.

## 8. Using the crate programmatically

```rust
use rl_agent::backend::{cpu_device, Cpu};
use rl_agent::net::NetConfig;
use rl_agent::ppo::{PpoAgent, PpoConfig};
use rl_agent::trainer::{evaluate, load_checkpoint, run, TrainerConfig};

// Train:
let device = cpu_device();
let outcome = run::<Cpu>(
    &TrainerConfig::default(),
    &NetConfig::default(),
    &PpoConfig::default(),
    &device,
    "aircraft.toml",
)?;
// outcome.agent, outcome.normalizer, outcome.last_eval

// Evaluate a checkpoint:
let cp = load_checkpoint::<Cpu>("checkpoints", &device)?;
let agent = PpoAgent::<Cpu>::from_parts(cp.net, cp.net_cfg, cp.ppo_cfg, seed, &device);
// feed normalized obs to agent.deterministic_action(&obs) ...
```

## 9. Tests

```bash
# From the repo root (tests locate ../aircraft.toml):
cargo test -p rl_agent
```

The suite covers the network (host/tensor log-prob parity to 1e-5), GAE
math (terminated bootstrapping), the normalizer, the rollout buffer, PPO
updates, and end-to-end trainer smoke tests — including a real-environment
two-iteration run, same-seed reproducibility, FC-limit action mapping, and a
checkpoint round-trip to a temp directory.

---

### Reference

- Schulman et al., *Proximal Policy Optimization Algorithms*, 2017
  (the clipped surrogate objective).
- Schulman et al., *High-Dimensional Continuous Control Using Generalized
  Advantage Estimation*, 2015 (GAE-λ).
- burn v0.21.0 docs: `burn::module`, `burn::optim`, `burn::record`,
  `burn::backend::{ndarray, wgpu}`.
- `flight_core::AvionicsEnvironment` / `EnvConfig` / `AVIONICS_OBS_DIM`
  (`flight_core/src/env.rs`).