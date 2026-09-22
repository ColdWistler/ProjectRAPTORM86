//! # rl_agent
//!
//! A pure-Rust, GPU-optimized reinforcement-learning agent that trains the
//! [`flight_core`] flight-dynamics engine with Proximal Policy Optimization.
//!
//! The training target is [`AvionicsEnvironment`](flight_core::AvionicsEnvironment):
//! the agent emits *attitude-command* actions (roll/pitch setpoints, yaw rate,
//! throttle) that a simulated PID flight controller turns into servo/ESC
//! outputs, while observations come from the *noisy* avionics sensor bus
//! (gyro/accel/GPS/baro/airspeed/battery).
//!
//! # Design
//!
//! - **Backends**: [`burn`] is the tensor/autodiff engine. Both the
//!   ndarray (CPU) and wgpu/Vulkan (GPU) backends are compiled into one
//!   binary and selected at runtime ([`backend`]).
//! - **Policy**: tanh-squashed diagonal-Gaussian actor + MLP critic
//!   ([`net`]). The action is sampled on the host from the device-computed
//!   mean, so a seeded [`rand`] RNG gives reproducible rollouts; the
//!   log-density uses the identical formula on both host and tensor.
//! - **PPO-Clip** ([`ppo`]) with a Generalized Advantage Estimator computed
//!   host-side ([`gae`]), per-minibatch advantage whitening, AdamW with
//!   gradient-norm clipping, and a Welford observation normalizer ([`normalize`]).
//!
//! # Training target
//!
//! `AvionicsEnvironment`, 19-dim noisy sensor observation:
//!
//! ```text
//!  0  gyro_p (rad/s)         10 baro_altitude (m)
//!  1  gyro_q (rad/s)         11 airspeed_indicated (m/s)
//!  2  gyro_r (rad/s)         12 battery_voltage (V)
//!  3  accel_x (m/s²)         13 battery_capacity_remaining_pct (%)
//!  4  accel_y (m/s²)         14 actual_elevator_deg (°)
//!  5  accel_z (m/s²)         15 actual_aileron_deg (°)
//!  6  gps_north (m, NED)     16 actual_rudder_deg (°)
//!  7  gps_east (m, NED)      17 actual_esc_output (0..1)
//!  8  gps_altitude (m)       18 sim_time (s)
//!  9  gps_fix_quality (0..3)
//! ```
//!
//! Actions (4-dim, tanh-squashed, before mapping):
//! `[roll_cmd, pitch_cmd, yaw_rate_cmd, throttle]` with per-channel scales —
//! roll `±0.6 rad`, pitch `±0.4 rad`, yaw rate `±0.8 rad/s`, throttle
//! `0.5·(tanh(a)+1)` — all inside the flight controller's `±45°` setpoint
//! clamps.
//!
//! Evaluation metrics compare the noisy `baro_altitude` / `airspeed_indicated`
//! channels against the environment's `target_altitude` / `target_airspeed`.

pub mod algo;
pub mod backend;
pub mod buffer;
pub mod gae;
pub mod net;
pub mod normalize;
pub mod ppo;
pub mod trainer;

pub use algo::{Algorithm, AlgoSpec};

/// Number of continuous action dimensions
/// (`roll_cmd`, `pitch_cmd`, `yaw_rate_cmd`, `throttle`).
pub const ACTION_DIM: usize = 4;

/// Observation dimension of the [`AvionicsEnvironment`](flight_core::AvionicsEnvironment)
/// training target (19 noisy bus channels).
pub const AVIONICS_OBS_DIM: usize = flight_core::AVIONICS_OBS_DIM;