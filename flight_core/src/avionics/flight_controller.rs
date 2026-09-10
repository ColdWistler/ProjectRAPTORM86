//! PID flight controller.
//!
//! Modeled after ArduPilot's `APM_Control` architecture. The controller
//! runs nested PID loops:
//!
//! ```text
//! Attitude outer loop                Rate inner loop
//! ┌───────────────────────┐          ┌─────────────────────┐
//! │ cmd_roll → φ error →  │          │  p_cmd → p error →  │
//! │ p_cmd (rate command)  │ ───────► │  servo deflection   │──► elevator/
//! │ cmd_pitch → θ error → │          │  q_cmd → q error →  │   aileron/
//! │ q_cmd (rate command)  │          │  r_cmd → r error →  │   rudder
//! └───────────────────────┘          │  rudder cmd        │   ESC
//!                                    └─────────────────────┘
//! ```
//!
//! # Bus contract
//! Reads: `gyro` (rate feedback), IMU attitude estimate, agent commands
//!       (`cmd_roll`, `cmd_pitch`, `cmd_yaw_rate`, `cmd_throttle`)
//! Writes: `fc_servo_*`, `fc_esc_throttle`, `fc_mode`
//!
//! # Modes
//! - [`FcMode::Rate`] — agent commands angular rates directly
//! - [`FcMode::Attitude`] — agent commands roll/pitch/yaw_rate, FC runs outer + inner loops
//! - [`FcMode::Manual`] — passthrough: agent commands directly drive servos
//! - [`FcMode::Stabilize`] — FC holds wings level, agent gives heading/throttle
//!
//! # Standards & Traceability
//! Structure follows ArduPilot's `AC_AttitudeControl` + `APM_Control`.
//! Butterfly/PID limit handling: each stage saturates and anti-windup
//! clamps the integrator.

use serde::Deserialize;

use super::bus::{AvionicsBus, FcMode};
use super::traits::{AvionicsComponent, Controller};

/// A single PID loop (P, I, D with anti-windup).
#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "snake_case")]
pub struct PidConfig {
    pub kp: f64,
    pub ki: f64,
    pub kd: f64,
    /// Output limit (defined by caller).
    #[serde(default = "default_out_limit")]
    pub out_limit: f64,
    /// Integral limit (saturation clamp for anti-windup).
    #[serde(default = "default_i_limit")]
    pub i_limit: f64,
}

fn default_out_limit() -> f64 {
    1.0
}
fn default_i_limit() -> f64 {
    0.5
}

impl Default for PidConfig {
    fn default() -> Self {
        Self {
            kp: 0.0,
            ki: 0.0,
            kd: 0.0,
            out_limit: default_out_limit(),
            i_limit: default_i_limit(),
        }
    }
}

/// PID controller with integrator anti-windup.
#[derive(Debug, Clone)]
pub struct PidController {
    config: PidConfig,
    /// Running integral term.
    integral: f64,
    /// Previous error (for derivative).
    prev_error: f64,
    /// Output of the previous step (for derivative kick filtering).
    prev_output: f64,
    /// Derivative low-pass filter state.
    d_filter: f64,
    /// Whether to use derivative-on-measurement (avoids derivative kick).
    derivative_on_measurement: bool,
}

impl PidController {
    pub fn new(config: PidConfig) -> Self {
        Self {
            config,
            integral: 0.0,
            prev_error: 0.0,
            prev_output: 0.0,
            d_filter: 0.0,
            derivative_on_measurement: false,
        }
    }

    /// Enable derivative-on-measurement (recommended for noisy sensors).
    pub fn set_derivative_on_measurement(&mut self, enabled: bool) {
        self.derivative_on_measurement = enabled;
    }

    /// Compute the PID output given the target and measurement.
    pub fn compute(&mut self, target: f64, measurement: f64, dt: f64) -> f64 {
        let error = target - measurement;

        // Proportional
        let p = self.config.kp * error;

        // Integral with anti-windup clamp
        self.integral += error * dt;
        self.integral = self
            .integral
            .clamp(-self.config.i_limit, self.config.i_limit);
        let i = self.config.ki * self.integral;

        // Derivative (either on error or on measurement)
        let d = if self.derivative_on_measurement {
            // Derivative on measurement: -(measurement change) / dt
            let dterm = -self.config.kd * (measurement - self.prev_output) / dt.max(1e-9);
            // Simple first-order smoothing
            let alpha = dt / (0.001 + dt); // 1ms time constant
            self.d_filter = self.d_filter + alpha * (dterm - self.d_filter);
            self.d_filter
        } else {
            let dterm = self.config.kd * (error - self.prev_error) / dt.max(1e-9);
            let alpha = dt / (0.001 + dt);
            self.d_filter = self.d_filter + alpha * (dterm - self.d_filter);
            self.d_filter
        };

        self.prev_error = error;
        self.prev_output = measurement;

        // Sum and clamp
        (p + i + d).clamp(-self.config.out_limit, self.config.out_limit)
    }

    /// Reset the PID state.
    pub fn reset(&mut self) {
        self.integral = 0.0;
        self.prev_error = 0.0;
        self.prev_output = 0.0;
        self.d_filter = 0.0;
    }
}

/// Configuration for the whole flight controller.
#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "snake_case")]
pub struct FlightControllerConfig {
    /// Roll attitude loop → roll rate command.
    #[serde(default)]
    pub roll_attitude: PidConfig,
    /// Pitch attitude loop → pitch rate command.
    #[serde(default)]
    pub pitch_attitude: PidConfig,
    /// Yaw-rate command (feed-forward, no outer loop).
    #[serde(default)]
    pub yaw_rate_ff: f64,
    /// Roll rate loop → aileron.
    #[serde(default)]
    pub roll_rate: PidConfig,
    /// Pitch rate loop → elevator.
    #[serde(default)]
    pub pitch_rate: PidConfig,
    /// Yaw rate loop → rudder.
    #[serde(default)]
    pub yaw_rate: PidConfig,
    /// Maximum commanded roll/pitch angle (rad).
    #[serde(default = "default_max_attitude_rad")]
    pub max_attitude_rad: f64,
}

fn default_max_attitude_rad() -> f64 {
    45.0_f64.to_radians()
}

impl Default for FlightControllerConfig {
    fn default() -> Self {
        Self {
            // Outer loops: attitude → rate
            roll_attitude: PidConfig {
                kp: 3.0,
                out_limit: 1.0,
                ..Default::default()
            },
            pitch_attitude: PidConfig {
                kp: 3.0,
                out_limit: 1.0,
                ..Default::default()
            },
            yaw_rate_ff: 1.0,
            // Inner loops: rate → surface
            roll_rate: PidConfig {
                kp: 30.0,
                out_limit: 30.0,
                ..Default::default()
            },
            pitch_rate: PidConfig {
                kp: 30.0,
                out_limit: 30.0,
                ..Default::default()
            },
            yaw_rate: PidConfig {
                kp: 20.0,
                out_limit: 30.0,
                ..Default::default()
            },
            max_attitude_rad: default_max_attitude_rad(),
        }
    }
}

/// Flight controller running nested attitude → rate PID loops.
pub struct FlightController {
    config: FlightControllerConfig,
    mode: FcMode,
    roll_att: PidController,
    pitch_att: PidController,
    roll_rate: PidController,
    pitch_rate: PidController,
    yaw_rate_pid: PidController,
    /// Smoothed throttle (1st-order lag).
    throttle_filter: f64,
}

impl FlightController {
    pub fn new(config: FlightControllerConfig) -> Self {
        // Enable derivative-on-measurement on rate loops by default
        // to avoid derivative kick from noisy gyro.
        let mut roll_rate = PidController::new(config.roll_rate.clone());
        roll_rate.set_derivative_on_measurement(true);
        let mut pitch_rate = PidController::new(config.pitch_rate.clone());
        pitch_rate.set_derivative_on_measurement(true);
        let mut yaw_rate_pid = PidController::new(config.yaw_rate.clone());
        yaw_rate_pid.set_derivative_on_measurement(true);

        Self {
            roll_att: PidController::new(config.roll_attitude.clone()),
            pitch_att: PidController::new(config.pitch_attitude.clone()),
            roll_rate,
            pitch_rate,
            yaw_rate_pid,
            throttle_filter: 0.0,
            mode: FcMode::Attitude,
            config,
        }
    }

    /// Current estimated attitude (roll/pitch) in radians.
    /// Derived from gyro integration, or (future) the EKF estimate.
    fn estimate_attitude(&self, bus: &AvionicsBus) -> (f64, f64) {
        // In the initial implementation we use the true quaternion from
        // the bus converted to Euler. A real FC would run an estimator.
        let q = bus.true_quat;
        let (w, x, y, z) = (q[0], q[1], q[2], q[3]);
        // Roll (φ) about X: atan2(2(wx + yz), 1 - 2(x² + y²))
        let roll = (2.0 * (w * x + y * z))
            .atan2(1.0 - 2.0 * (x * x + y * y));
        // Pitch (θ) about Y: asin(2(wy - zx))
        let pitch = (2.0 * (w * y - z * x)).clamp(-1.0, 1.0).asin();
        (roll, pitch)
    }
}

impl AvionicsComponent for FlightController {
    fn name(&self) -> &str {
        "FC"
    }

    fn init(&mut self, _dt: f64) {
        self.roll_att.reset();
        self.pitch_att.reset();
        self.roll_rate.reset();
        self.pitch_rate.reset();
        self.yaw_rate_pid.reset();
        self.throttle_filter = 0.0;
    }

    fn step(&mut self, bus: &mut AvionicsBus, dt: f64) {
        let thr = bus.cmd_throttle.clamp(0.0, 1.0);

        match self.mode {
            FcMode::Manual => {
                // Passthrough: agent commands drive surfaces directly
                bus.fc_servo_aileron = bus.cmd_roll.clamp(-30.0, 30.0);
                bus.fc_servo_elevator = bus.cmd_pitch.clamp(-30.0, 30.0);
                bus.fc_servo_rudder = bus.cmd_yaw_rate.clamp(-30.0, 30.0);
                bus.fc_esc_throttle = thr;
            }
            FcMode::Rate => {
                // Rate loop only: cmd_roll = roll rate (rad/s), etc.
                let p_cmd = bus.cmd_roll;
                let q_cmd = bus.cmd_pitch;
                let r_cmd = bus.cmd_yaw_rate;

                bus.fc_servo_aileron = self.roll_rate.compute(p_cmd, bus.gyro.x, dt);
                bus.fc_servo_elevator = self.pitch_rate.compute(q_cmd, bus.gyro.y, dt);
                bus.fc_servo_rudder = self.yaw_rate_pid.compute(r_cmd, bus.gyro.z, dt);
                bus.fc_esc_throttle = Self::smooth_throttle(
                    self.throttle_filter,
                    thr,
                    dt,
                );
            }
            FcMode::Attitude => {
                // Outer loop: attitude (roll/pitch angle) → rate command
                let (phi, theta) = self.estimate_attitude(bus);
                let max_att = self.config.max_attitude_rad;

                let roll_cmd = bus.cmd_roll.clamp(-max_att, max_att);
                let pitch_cmd = bus.cmd_pitch.clamp(-max_att, max_att);

                let p_cmd = self.roll_att.compute(roll_cmd, phi, dt);
                let q_cmd = self.pitch_att.compute(pitch_cmd, theta, dt);
                let r_cmd = bus.cmd_yaw_rate * self.config.yaw_rate_ff;

                // Inner loop: rate → surface
                bus.fc_servo_aileron = self.roll_rate.compute(p_cmd, bus.gyro.x, dt);
                bus.fc_servo_elevator = self.pitch_rate.compute(q_cmd, bus.gyro.y, dt);
                bus.fc_servo_rudder = self.yaw_rate_pid.compute(r_cmd, bus.gyro.z, dt);
                bus.fc_esc_throttle = Self::smooth_throttle(
                    self.throttle_filter,
                    thr,
                    dt,
                );
            }
            FcMode::Stabilize => {
                // Hold wings-level: zero roll command, damp yaw, keep pitch
                let (phi, theta) = self.estimate_attitude(bus);
                let p_cmd = self.roll_att.compute(0.0, phi, dt);
                let q_cmd = self.pitch_att.compute(bus.cmd_pitch, theta, dt);
                let r_cmd = bus.cmd_yaw_rate * self.config.yaw_rate_ff;

                bus.fc_servo_aileron = self.roll_rate.compute(p_cmd, bus.gyro.x, dt);
                bus.fc_servo_elevator = self.pitch_rate.compute(q_cmd, bus.gyro.y, dt);
                bus.fc_servo_rudder = self.yaw_rate_pid.compute(r_cmd, bus.gyro.z, dt);
                bus.fc_esc_throttle = Self::smooth_throttle(
                    self.throttle_filter,
                    thr,
                    dt,
                );
            }
        }
    }

    fn reset(&mut self) {
        self.init(0.0);
    }
}

impl FlightController {
    /// 1st-order low-pass on throttle command.
    ///
    /// `tau` is hardcoded to 0.05s (50ms) for the initial implementation.
    fn smooth_throttle(filter: f64, target: f64, dt: f64) -> f64 {
        let tau = 0.05;
        let alpha = dt / (tau + dt);
        (target - filter) * alpha + filter
    }
}

impl Controller for FlightController {
    fn mode(&self) -> FcMode {
        self.mode
    }

    fn set_mode(&mut self, mode: FcMode) {
        // Mode transitions reset the PID state to avoid transients
        self.init(0.0);
        self.mode = mode;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use nalgebra::Vector3;

    fn make_fc() -> (FlightController, AvionicsBus) {
        let fc = FlightController::new(FlightControllerConfig::default());
        let bus = AvionicsBus::default();
        (fc, bus)
    }

    #[test]
    fn pid_hits_target() {
        let mut pid = PidController::new(PidConfig {
            kp: 1.0,
            ki: 0.1,
            kd: 0.0,
            out_limit: 100.0,
            i_limit: 10.0,
        });
        // Drive from 0 toward target 5.0
        let mut meas = 0.0;
        for _ in 0..20_000 {
            meas += pid.compute(5.0, meas, 0.001) * 0.001;
        }
        assert!((meas - 5.0).abs() < 0.1, "PID should converge to 5.0, got {meas}");
    }

    #[test]
    fn pid_respects_output_limit() {
        let mut pid = PidController::new(PidConfig {
            kp: 10.0,
            out_limit: 2.0,
            ..Default::default()
        });
        let out = pid.compute(5.0, 0.0, 0.001);
        assert!((out - 2.0).abs() < 1e-9, "output should clamp at 2.0, got {out}");
    }

    #[test]
    fn pid_anti_windup() {
        let mut pid = PidController::new(PidConfig {
            kp: 0.1,
            ki: 1.0,
            out_limit: 1.0,
            i_limit: 5.0,
            ..Default::default()
        });
        // Large sustained error → integral saturates, not grows forever
        for _ in 0..10_000 {
            pid.compute(10.0, 0.0, 0.001);
        }
        assert!(
            pid.integral <= 5.0 + 1e-9,
            "integral should saturate at i_limit, got {}",
            pid.integral
        );
    }

    #[test]
    fn fc_manual_passthrough() {
        let (mut fc, mut bus) = make_fc();
        fc.init(0.0);
        fc.set_mode(FcMode::Manual);

        bus.cmd_roll = 10.0;
        bus.cmd_pitch = -5.0;
        bus.cmd_yaw_rate = 3.0;
        bus.cmd_throttle = 0.7;

        fc.step(&mut bus, 0.01);

        assert!((bus.fc_servo_aileron - 10.0).abs() < 1e-9);
        assert!((bus.fc_servo_elevator - (-5.0)).abs() < 1e-9);
        assert!((bus.fc_servo_rudder - 3.0).abs() < 1e-9);
        // Throttle passes through (smoothed, so near-instant for large dt)
        assert!((bus.fc_esc_throttle - 0.7).abs() < 0.05);
    }

    #[test]
    fn fc_attitude_tracks_error() {
        let (mut fc, mut bus) = make_fc();
        fc.init(0.0);
        fc.set_mode(FcMode::Attitude);

        // Wings level, command a roll of ~0.2 rad
        bus.true_quat = [1.0, 0.0, 0.0, 0.0];
        bus.gyro = Vector3::new(0.0, 0.0, 0.0);
        bus.cmd_roll = 0.2;
        bus.cmd_pitch = 0.0;
        bus.cmd_yaw_rate = 0.0;
        bus.cmd_throttle = 0.6;

        fc.step(&mut bus, 0.01);

        // Roll error positive → aileron output should be positive (right roll)
        assert!(
            bus.fc_servo_aileron > 1.0,
            "positive roll error should produce aileron deflection, got {}",
            bus.fc_servo_aileron
        );
        // No pitch error → elevator ≈ 0
        assert!(
            bus.fc_servo_elevator.abs() < 1.0,
            "no pitch error should produce ~0 elevator, got {}",
            bus.fc_servo_elevator
        );
    }

    #[test]
    fn fc_rate_mode_direct() {
        let (mut fc, mut bus) = make_fc();
        fc.init(0.0);
        fc.set_mode(FcMode::Rate);

        bus.gyro = Vector3::new(0.0, 0.0, 0.0);
        bus.cmd_roll = 1.0; // rad/s
        bus.cmd_pitch = 0.0;
        bus.cmd_yaw_rate = 0.0;

        fc.step(&mut bus, 0.01);

        // Positive roll rate command → positive aileron
        assert!(
            bus.fc_servo_aileron > 0.5,
            "roll rate cmd → aileron, got {}",
            bus.fc_servo_aileron
        );

        // Now rate satisfied → output should drop toward 0
        bus.gyro = Vector3::new(1.0, 0.0, 0.0);
        fc.step(&mut bus, 0.01);
        assert!(
            bus.fc_servo_aileron.abs() < 0.5,
            "rate satisfied → ~0 aileron, got {}",
            bus.fc_servo_aileron
        );
    }

    #[test]
    fn fc_attitude_clamps_commands() {
        let (mut fc, mut bus) = make_fc();
        fc.init(0.0);
        fc.set_mode(FcMode::Attitude);

        bus.cmd_roll = 100.0; // absurd
        bus.cmd_pitch = 100.0;
        fc.step(&mut bus, 0.01);

        // Should not produce NaN or unbounded output
        assert!(bus.fc_servo_aileron.is_finite());
        assert!(bus.fc_servo_elevator.is_finite());
    }
}