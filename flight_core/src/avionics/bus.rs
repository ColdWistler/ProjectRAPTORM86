//! Shared data bus (blackboard) for the avionics system.
//!
//! All avionics components communicate exclusively through this bus.
//! No component imports or calls another component directly — they
//! read the fields they need and write only their own outputs.
//!
//! # Design
//! The bus is split into logical sections:
//! - **True state** — written by the physics engine, not exposed to the RL agent
//! - **Sensor outputs** — written by sensor modules, read by FC and agent
//! - **Flight controller outputs** — written by the FC, read by actuators
//! - **Agent commands** — written by the RL agent, read by the FC
//! - **Actuator feedback** — written by actuators, read by physics and agent
//!
//! Each component is responsible for reading/writing only the fields
//! documented in its own module header. This contract is enforced by
//! convention and testing, not by the type system, to keep the bus
//! ergonomics flat and simple.

use nalgebra::Vector3;

/// Fault flags carried on the bus for failure injection.
#[derive(Debug, Clone, Copy, Default)]
pub struct FaultFlags {
    pub imu_failed: bool,
    pub gps_failed: bool,
    pub baro_failed: bool,
    pub mag_failed: bool,
    pub airspeed_failed: bool,
    pub servo_elevator_failed: bool,
    pub servo_aileron_failed: bool,
    pub servo_rudder_failed: bool,
    pub esc_failed: bool,
    pub battery_depleted: bool,
}

impl FaultFlags {
    /// True when no fault is registered.
    pub fn all_clear(&self) -> bool {
        !self.imu_failed
            && !self.gps_failed
            && !self.baro_failed
            && !self.mag_failed
            && !self.airspeed_failed
            && !self.servo_elevator_failed
            && !self.servo_aileron_failed
            && !self.servo_rudder_failed
            && !self.esc_failed
            && !self.battery_depleted
    }

    /// Set or clear the single fault referenced by `flag`.
    pub fn set(&mut self, flag: FaultFlag, on: bool) {
        *flag.field_of(self) = on;
    }

    /// Read the single fault referenced by `flag`.
    pub const fn has(&self, flag: FaultFlag) -> bool {
        match flag {
            FaultFlag::Imu => self.imu_failed,
            FaultFlag::Gps => self.gps_failed,
            FaultFlag::Baro => self.baro_failed,
            FaultFlag::Mag => self.mag_failed,
            FaultFlag::Airspeed => self.airspeed_failed,
            FaultFlag::ServoElevator => self.servo_elevator_failed,
            FaultFlag::ServoAileron => self.servo_aileron_failed,
            FaultFlag::ServoRudder => self.servo_rudder_failed,
            FaultFlag::Esc => self.esc_failed,
            FaultFlag::BatteryDepleted => self.battery_depleted,
        }
    }
}

/// A single injectable fault source. Each variant maps exactly one bit of
/// [`FaultFlags`]; components self-check the shared register so failure
/// behaviour lives with the failing device (same blackboard contract).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FaultFlag {
    Imu,
    Gps,
    Baro,
    Mag,
    Airspeed,
    ServoElevator,
    ServoAileron,
    ServoRudder,
    Esc,
    BatteryDepleted,
}

impl FaultFlag {
    /// Borrow the exact `bool` field of `flags` that this flag controls.
    fn field_of(self, flags: &mut FaultFlags) -> &mut bool {
        match self {
            Self::Imu => &mut flags.imu_failed,
            Self::Gps => &mut flags.gps_failed,
            Self::Baro => &mut flags.baro_failed,
            Self::Mag => &mut flags.mag_failed,
            Self::Airspeed => &mut flags.airspeed_failed,
            Self::ServoElevator => &mut flags.servo_elevator_failed,
            Self::ServoAileron => &mut flags.servo_aileron_failed,
            Self::ServoRudder => &mut flags.servo_rudder_failed,
            Self::Esc => &mut flags.esc_failed,
            Self::BatteryDepleted => &mut flags.battery_depleted,
        }
    }
}

/// Flight controller mode — determines which loop hierarchy is active.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum FcMode {
    /// Rate-only: agent commands angular rates directly, FC mixes to servos.
    Rate,
    /// Attitude: agent commands roll/pitch/yaw-rate, FC runs attitude→rate→mixer.
    #[default]
    Attitude,
    /// Manual: FC is bypassed, agent commands pass through directly to servos.
    Manual,
    /// Stabilize: FC holds wings-level + altitude, agent gives heading/throttle.
    Stabilize,
}

/// The shared avionics data bus.
///
/// All fields are public for direct access by components. The naming
/// convention separates logical sections with comments; components read
/// the sections they need and write only the sections documented in
/// their module header.
#[derive(Debug, Clone)]
pub struct AvionicsBus {
    // --- True state (from physics, not exposed to agent) ---
    /// Earth-frame position NED (m).
    pub true_position_ned: Vector3<f64>,
    /// Body-frame velocity (m/s): u=forward, v=right, w=down.
    pub true_velocity_body: Vector3<f64>,
    /// Quaternion orientation (Earth → body).
    pub true_quat: [f64; 4],
    /// Body-frame angular rates (rad/s): p=roll, q=pitch, r=yaw.
    pub true_angular_rates: Vector3<f64>,
    /// True airspeed (m/s).
    pub true_airspeed: f64,
    /// True altitude above sea level (m).
    pub true_altitude: f64,
    /// True angle of attack (rad).
    pub true_alpha: f64,
    /// True sideslip angle (rad).
    pub true_beta: f64,
    /// Wind vector in Earth NED frame (m/s).
    pub true_wind: Vector3<f64>,
    /// True dynamic pressure (Pa).
    pub true_q_dynamic: f64,

    // --- IMU outputs (written by imu.rs) ---
    pub gyro: Vector3<f64>,
    pub accel: Vector3<f64>,
    pub imu_sample_time: f64,

    // --- GPS outputs (written by gps.rs) ---
    pub gps_position_ned: Vector3<f64>,
    pub gps_velocity_ned: Vector3<f64>,
    pub gps_fix_quality: u8,
    pub gps_hdop: f64,
    pub gps_sample_time: f64,

    // --- Barometer outputs (written by baro.rs) ---
    pub baro_altitude: f64,
    pub baro_sample_time: f64,

    // --- Magnetometer outputs (written by magnetometer.rs) ---
    pub mag_field_body: Vector3<f64>,
    pub mag_sample_time: f64,

    // --- Airspeed sensor outputs (written by airspeed.rs) ---
    pub airspeed_indicated: f64,
    pub airspeed_sample_time: f64,

    // --- Battery outputs (written by battery.rs) ---
    pub battery_voltage: f64,
    pub battery_current: f64,
    pub battery_capacity_remaining_pct: f64,

    // --- Flight controller outputs (written by flight_controller.rs) ---
    pub fc_mode: FcMode,
    pub fc_servo_elevator: f64,
    pub fc_servo_aileron: f64,
    pub fc_servo_rudder: f64,
    pub fc_esc_throttle: f64,
    pub fc_fault_flags: FaultFlags,

    // --- Agent commands (written by RL agent) ---
    pub cmd_roll: f64,
    pub cmd_pitch: f64,
    pub cmd_yaw_rate: f64,
    pub cmd_throttle: f64,

    // --- Actuator feedback (written by actuator.rs) ---
    pub actual_elevator_deg: f64,
    pub actual_aileron_deg: f64,
    pub actual_rudder_deg: f64,
    pub actual_esc_output: f64,

    // --- Simulation time ---
    pub sim_time: f64,
}

impl Default for AvionicsBus {
    fn default() -> Self {
        Self {
            true_position_ned: Vector3::zeros(),
            true_velocity_body: Vector3::zeros(),
            true_quat: [1.0, 0.0, 0.0, 0.0],
            true_angular_rates: Vector3::zeros(),
            true_airspeed: 0.0,
            true_altitude: 0.0,
            true_alpha: 0.0,
            true_beta: 0.0,
            true_wind: Vector3::zeros(),
            true_q_dynamic: 0.0,

            gyro: Vector3::zeros(),
            accel: Vector3::new(0.0, 0.0, -9.80665),
            imu_sample_time: 0.0,

            gps_position_ned: Vector3::zeros(),
            gps_velocity_ned: Vector3::zeros(),
            gps_fix_quality: 3,
            gps_hdop: 1.0,
            gps_sample_time: 0.0,

            baro_altitude: 0.0,
            baro_sample_time: 0.0,

            mag_field_body: Vector3::new(0.0, 0.0, 0.0),
            mag_sample_time: 0.0,

            airspeed_indicated: 0.0,
            airspeed_sample_time: 0.0,

            battery_voltage: 16.8,
            battery_current: 0.0,
            battery_capacity_remaining_pct: 100.0,

            fc_mode: FcMode::Attitude,
            fc_servo_elevator: 0.0,
            fc_servo_aileron: 0.0,
            fc_servo_rudder: 0.0,
            fc_esc_throttle: 0.0,
            fc_fault_flags: FaultFlags::default(),

            cmd_roll: 0.0,
            cmd_pitch: 0.0,
            cmd_yaw_rate: 0.0,
            cmd_throttle: 0.5,

            actual_elevator_deg: 0.0,
            actual_aileron_deg: 0.0,
            actual_rudder_deg: 0.0,
            actual_esc_output: 0.0,

            sim_time: 0.0,
        }
    }
}

impl AvionicsBus {
    /// Write the true aircraft state from the physics engine into the bus.
    /// Called once per step before any avionics component runs.
    ///
    /// Airspeed / angle-of-attack / sideslip are computed **air-relative**
    /// (relative wind) from the state and the Earth-frame wind vector, so
    /// downstream sensors see physically-consistent values.
    pub fn write_true_state(
        &mut self,
        state: &crate::state::AircraftState,
        wind_earth: &Vector3<f64>,
        q_dynamic: f64,
    ) {
        self.true_position_ned = Vector3::new(state.pos_x, state.pos_y, state.pos_z);
        self.true_velocity_body = Vector3::new(state.u, state.v, state.w);
        self.true_quat = [state.q0, state.q1, state.q2, state.q3];
        self.true_angular_rates = Vector3::new(state.p, state.q, state.r);
        let air = state.air_velocity(wind_earth);
        self.true_airspeed = air.norm();
        self.true_alpha = air.z.atan2(air.x);
        self.true_beta = air.y.atan2(air.x.max(crate::state::SIDESLIP_AXIAL_MIN));
        self.true_altitude = state.altitude();
        self.true_wind = *wind_earth;
        self.true_q_dynamic = q_dynamic;
    }

    /// Read the actuator outputs as control inputs for the physics engine.
    /// Returns (elevator_rad, aileron_rad, rudder_rad, throttle).
    pub fn read_actuator_outputs(&self) -> (f64, f64, f64, f64) {
        let elev = self.actual_elevator_deg.to_radians();
        let ail = self.actual_aileron_deg.to_radians();
        let rud = self.actual_rudder_deg.to_radians();
        let thr = self.actual_esc_output;
        (elev, ail, rud, thr)
    }

    /// Read agent commands as actuator targets (for Manual mode passthrough).
    pub fn read_agent_commands_as_actuators(&self) -> (f64, f64, f64, f64) {
        let elev = self.cmd_pitch.clamp(-30.0, 30.0);
        let ail = self.cmd_roll.clamp(-30.0, 30.0);
        let rud = self.cmd_yaw_rate.clamp(-30.0, 30.0);
        let thr = self.cmd_throttle.clamp(0.0, 1.0);
        (elev, ail, rud, thr)
    }

    /// Clear all sensor outputs (used on reset).
    pub fn clear_sensor_outputs(&mut self) {
        self.gyro = Vector3::zeros();
        self.accel = Vector3::new(0.0, 0.0, -9.80665);
        self.gps_position_ned = Vector3::zeros();
        self.gps_velocity_ned = Vector3::zeros();
        self.baro_altitude = 0.0;
        self.mag_field_body = Vector3::zeros();
        self.airspeed_indicated = 0.0;
        self.battery_voltage = 16.8;
        self.battery_current = 0.0;
        self.battery_capacity_remaining_pct = 100.0;
    }
}
