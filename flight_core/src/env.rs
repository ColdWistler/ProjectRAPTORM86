//! Reinforcement-learning environment wrapper around [`crate::Simulator`].
//!
//! Implements a Gymnasium-compatible single-agent interface
//! (`reset` / `step` / `observation` / `reward` / `terminated` / `truncated`)
//! so the flight dynamics engine can be trained with RL libraries
//! (e.g. stable-baselines3 via a Python bridge).
//!
//! # Standards References
//! - Environment API shape follows the OpenAI Gymnasium environment interface
//!   (Farama Foundation, gymnasium v1.0+ specification).

use crate::integrator::step;
use crate::{ControlInputs, Simulator, WindConfig, WindEnvironment};
use nalgebra::Vector3;

#[cfg(feature = "full-avionics")]
use crate::atmosphere::Atmosphere;
#[cfg(feature = "full-avionics")]
use crate::avionics::{
    ActuatorSuite, AirspeedConfig, AirspeedSensor, AvionicsSystem, BaroConfig, BaroSensor, Battery,
    BatteryConfig, Esc, EscConfig, FaultFlag, FaultFlags, FlightController, FlightControllerConfig,
    GpsConfig, GpsSensor, ImuConfig, ImuSensor, MagConfig, MagnetometerSensor, ServoConfig,
};

/// A 12-component observation vector (same layout as
/// [`crate::AircraftState::to_observation_array`]).
pub type Observation = [f64; 12];

/// The continuous control action applied to the aircraft on a single step.
#[derive(Debug, Clone, Copy)]
pub struct ControlAction {
    /// Elevator deflection in radians.
    pub elevator: f64,
    /// Aileron deflection in radians.
    pub aileron: f64,
    /// Rudder deflection in radians.
    pub rudder: f64,
    /// Throttle setting, clamped internally to `[0.0, 1.0]`.
    pub throttle: f64,
    /// Flap (trailing-edge) deflection in radians.
    pub flaps: f64,
}

impl ControlAction {
    /// A no-op / neutral action: all control surfaces centered, throttle mid.
    pub fn neutral() -> Self {
        Self {
            elevator: 0.0,
            aileron: 0.0,
            rudder: 0.0,
            throttle: 0.5,
            flaps: 0.0,
        }
    }
}

/// Outcome of a single [`Environment::step`].
#[derive(Debug, Clone, Copy)]
pub struct EnvStep {
    pub observation: Observation,
    pub reward: f64,
    /// `true` if the episode ended because the aircraft crashed.
    pub terminated: bool,
    /// `true` if the episode ended because the step budget was exhausted.
    pub truncated: bool,
}

/// Tuning parameters for the reward function and episode termination.
#[derive(Debug, Clone)]
pub struct EnvConfig {
    /// Target cruise altitude in meters.
    pub target_altitude: f64,
    /// Target cruise airspeed in m/s.
    pub target_airspeed: f64,
    /// Altitude (m) below which the aircraft is considered crashed.
    pub ground_altitude: f64,
    /// Maximum elevator deflection magnitude (radians) the agent may use.
    pub max_elevator: f64,
    /// Maximum aileron deflection magnitude (radians).
    pub max_aileron: f64,
    /// Maximum rudder deflection magnitude (radians).
    pub max_rudder: f64,
    /// Maximum number of steps per episode.
    pub max_steps: usize,
    /// Physics time step per `step()` call (seconds).
    pub dt: f64,

    /// Optional atmospheric wind configuration. `None` (default) disables
    /// wind entirely (still air), preserving prior behaviour.
    pub wind_config: Option<WindConfig>,

    /// Per-step survival bonus.
    pub w_time: f64,
    /// Altitude error weight.
    pub w_alt: f64,
    /// Scale (meters) dividing the altitude error.
    pub scale_alt: f64,
    /// Airspeed error weight.
    pub w_spd: f64,
    /// Scale (m/s) dividing the airspeed error.
    pub scale_spd: f64,
    /// Pitch-away-from-level penalty weight (per rad²).
    pub w_pitch: f64,
    /// Roll-away-from-wings-level penalty weight (per rad²).
    pub w_roll: f64,
    /// Crash termination penalty.
    pub w_crash: f64,
}

impl Default for EnvConfig {
    fn default() -> Self {
        Self {
            target_altitude: 1000.0,
            target_airspeed: 60.0,
            ground_altitude: 0.0,
            max_elevator: 0.35,
            max_aileron: 0.35,
            max_rudder: 0.35,
            max_steps: 2000,
            dt: 1.0 / 30.0,
            wind_config: None,
            w_time: 1.0,
            w_alt: 2.0,
            scale_alt: 50.0,
            w_spd: 1.0,
            scale_spd: 10.0,
            w_pitch: 1.0,
            w_roll: 1.0,
            w_crash: 200.0,
        }
    }
}

/// A Gymnasium-style single-agent environment.
pub struct Environment {
    sim: Simulator,
    pub config: EnvConfig,
    step_count: usize,
    wind: Option<WindEnvironment>,
}

impl Environment {
    /// Create an environment, loading the aircraft config from
    /// `config_path` and using default reward tuning.
    pub fn new(config_path: &str) -> Result<Self, Box<dyn std::error::Error>> {
        Self::with_config(config_path, EnvConfig::default())
    }

    /// Create an environment with a custom [`EnvConfig`].
    pub fn with_config(config_path: &str, config: EnvConfig) -> Result<Self, Box<dyn std::error::Error>> {
        let sim = Simulator::new(config_path)?;
        let wind = config.wind_config.clone().map(WindEnvironment::new);
        Ok(Self {
            sim,
            config,
            step_count: 0,
            wind,
        })
    }

    /// Reset the aircraft to steady level flight and return the initial
    /// observation.
    pub fn reset(&mut self) -> (Observation, usize) {
        let prev_steps = self.step_count;
        self.sim.reset();
        self.step_count = 0;
        (self.sim.state.to_observation_array(), prev_steps)
    }

    /// Apply an action, advance the physics by `dt`, and return the next
    /// observation, reward, and episode-done flags.
    pub fn step(&mut self, action: ControlAction) -> EnvStep {
        let alt_before = self.sim.state.altitude();

        let elevator = action
            .elevator
            .clamp(-self.config.max_elevator, self.config.max_elevator);
        let aileron = action
            .aileron
            .clamp(-self.config.max_aileron, self.config.max_aileron);
        let rudder = action
            .rudder
            .clamp(-self.config.max_rudder, self.config.max_rudder);
        let throttle = action.throttle.clamp(0.0, 1.0);
        let flaps = action.flaps.clamp(0.0, 0.7);

        // Compute the total wind (steady + turbulence) in the Earth NED frame.
        let mut wind_vec: Option<Vector3<f64>> = None;
        if let Some(wind_env) = &mut self.wind {
            let vt_air = self.sim.state.true_airspeed(&Vector3::zeros());
            wind_vec = Some(wind_env.total_wind(&self.sim.state, vt_air, self.config.dt));
        }

        let controls = ControlInputs {
            elevator,
            aileron,
            rudder,
            throttle,
            flap: flaps,
        };
        step(
            &mut self.sim.state,
            &self.sim.config,
            controls,
            wind_vec.as_ref(),
            self.config.dt,
            None,
        );

        self.step_count += 1;
        let observation = self.sim.state.to_observation_array();

        let reward = self.compute_reward(alt_before);

        // Termination: ground impact.
        let terminated = self.sim.state.altitude() <= self.config.ground_altitude;
        // Truncation: step budget exhausted.
        let truncated = self.step_count >= self.config.max_steps;

        EnvStep {
            observation,
            reward,
            terminated,
            truncated,
        }
    }

    /// Shaped reward for the *current* aircraft state plus a crash penalty
    /// handled by the caller via `terminated`.
    fn compute_reward(&self, _alt_before: f64) -> f64 {
        shaped_reward(&self.sim.state, &self.config)
    }

    /// The crash penalty applied when the episode terminates by impact.
    pub fn crash_penalty(&self) -> f64 {
        -self.config.w_crash
    }

    /// Current step index within the episode (0-based).
    pub fn step_count(&self) -> usize {
        self.step_count
    }
}

/// Shaped reward for the current aircraft state: survival time bonus minus
/// weighted errors on altitude, airspeed, and away-from-level attitude.
/// Shared by [`Environment`] and [`AvionicsEnvironment`].
fn shaped_reward(s: &crate::AircraftState, cfg: &EnvConfig) -> f64 {
    let alt = s.altitude();
    let airspeed = s.airspeed();
    let (roll, pitch, _yaw) = s.euler_angles();

    let mut reward = cfg.w_time;
    reward -= cfg.w_alt * (alt - cfg.target_altitude).abs() / cfg.scale_alt;
    reward -= cfg.w_spd * (airspeed - cfg.target_airspeed).abs() / cfg.scale_spd;
    reward -= cfg.w_pitch * pitch * pitch;
    reward -= cfg.w_roll * roll * roll;
    reward
}

/// Size of [`AvionicsObservation`].
#[cfg(feature = "full-avionics")]
pub const AVIONICS_OBS_DIM: usize = 19;

/// Observation vector from the *noisy* avionics sensors on the bus (not the
/// true physics state), so the agent learns to fly on what hardware would
/// actually report.
///
/// Layout (SI unless noted):
/// ```text
///  0  gyro_p (rad/s)         10 baro_altitude (m)
///  1  gyro_q (rad/s)         11 airspeed_indicated (m/s)
///  2  gyro_r (rad/s)         12 battery_voltage (V)
///  3  accel_x (m/s²)         13 battery_capacity_remaining_pct (%)
///  4  accel_y (m/s²)         14 actual_elevator_deg (°)
///  5  accel_z (m/s²)         15 actual_aileron_deg (°)
///  6  gps_north (m, NED)     16 actual_rudder_deg (°)
///  7  gps_east (m, NED)      17 actual_esc_output (0..1)
///  8  gps_altitude (m)       18 sim_time (s)
///  9  gps_fix_quality (0..3)
/// ```
#[cfg(feature = "full-avionics")]
pub type AvionicsObservation = [f64; AVIONICS_OBS_DIM];

/// The continuous control action applied to the **flight controller**, not
/// directly to the aircraft. Commands are attitude setpoints:
/// roll/pitch angles, yaw rate, and throttle.
#[cfg(feature = "full-avionics")]
#[derive(Debug, Clone, Copy)]
pub struct AvionicsAction {
    /// Commanded roll attitude offset (radians), clamped internally to ±45°.
    pub roll_cmd: f64,
    /// Commanded pitch attitude offset (radians), clamped internally to ±45°.
    pub pitch_cmd: f64,
    /// Commanded yaw rate (radians/second).
    pub yaw_rate_cmd: f64,
    /// Throttle setting, clamped internally to `[0.0, 1.0]`.
    pub throttle: f64,
}

#[cfg(feature = "full-avionics")]
impl AvionicsAction {
    /// A level-flight cruise command: wings level, zero yaw rate, mid throttle.
    pub fn level_cruise(throttle: f64) -> Self {
        Self {
            roll_cmd: 0.0,
            pitch_cmd: 0.0,
            yaw_rate_cmd: 0.0,
            throttle,
        }
    }
}

/// Outcome of a single [`AvionicsEnvironment::step`] with a full avionics
/// observation.
#[cfg(feature = "full-avionics")]
#[derive(Debug, Clone, Copy)]
pub struct AvionicsEnvStep {
    pub observation: AvionicsObservation,
    pub reward: f64,
    /// `true` if the episode ended because the aircraft crashed.
    pub terminated: bool,
    /// `true` if the episode ended because the step budget was exhausted.
    pub truncated: bool,
}

/// A GPS-style continuous-control environment that flies through the simulated
/// avionics stack: the action is an attitude command, sensors observe the
/// noisy bus, the PID flight controller drives servos/ESC, and the actuator
/// outputs feed the 6-DOF physics. Failures can be injected with
/// [`AvionicsEnvironment::inject_fault`].
#[cfg(feature = "full-avionics")]
pub struct AvionicsEnvironment {
    sim: Simulator,
    avionics: AvionicsSystem,
    pub config: EnvConfig,
    step_count: usize,
    wind: Option<WindEnvironment>,
}

#[cfg(feature = "full-avionics")]
impl AvionicsEnvironment {
    /// Load the aircraft config from `config_path` and build a fully-connected
    /// avionics stack (IMU/GPS/baro/mag/airspeed sensors, attitude PID FC,
    /// servo suite, ESC, battery) with default hardware characteristics.
    pub fn new(config_path: &str) -> Result<Self, Box<dyn std::error::Error>> {
        Self::with_config(config_path, EnvConfig::default())
    }

    /// Create an avionics environment with a custom [`EnvConfig`].
    pub fn with_config(
        config_path: &str,
        config: EnvConfig,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        let sim = Simulator::new(config_path)?;
        let wind = config.wind_config.clone().map(WindEnvironment::new);
        let mut avionics = Self::make_avionics(config.dt);
        avionics.init();
        Ok(Self {
            sim,
            avionics,
            config,
            step_count: 0,
            wind,
        })
    }

    /// Build the default hardware stack. Register order matters: ESC and
    /// battery are generic devices that run after the actuator suite.
    fn make_avionics(dt: f64) -> AvionicsSystem {
        let mut sys = AvionicsSystem::new(dt);
        sys.add_sensor(Box::new(ImuSensor::new(ImuConfig::default())));
        sys.add_sensor(Box::new(GpsSensor::new(GpsConfig::default())));
        sys.add_sensor(Box::new(BaroSensor::new(BaroConfig::default())));
        sys.add_sensor(Box::new(MagnetometerSensor::new(MagConfig::default())));
        sys.add_sensor(Box::new(AirspeedSensor::new(AirspeedConfig::default())));
        sys.add_controller(Box::new(FlightController::new(
            FlightControllerConfig::default(),
        )));
        // Servos/ESC/battery are generic components (not trait-typed actuators);
        // they run in insertion order after the controller stage.
        sys.add_component(Box::new(ActuatorSuite::new(
            ServoConfig::default(),
            ServoConfig::default(),
            ServoConfig::default(),
        )));
        sys.add_component(Box::new(Esc::new(EscConfig::default())));
        sys.add_component(Box::new(Battery::new(BatteryConfig::default())));
        sys
    }

    /// Reset to trimmed level flight and clear all injected faults.
    pub fn reset(&mut self) -> (AvionicsObservation, usize) {
        let prev_steps = self.step_count;
        self.sim.reset();
        self.avionics.reset();
        self.step_count = 0;
        self.avionics.bus_mut().fc_fault_flags = FaultFlags::default();
        (self.observation(), prev_steps)
    }

    /// Apply an attitude command, run the avionics stack, advance the physics
    /// by `dt`, and return the next noisy observation / reward / done-flags.
    pub fn step(&mut self, action: AvionicsAction) -> AvionicsEnvStep {
        // 1. Feed the agent command into the bus (the FC will read it).
        let bus = self.avionics.bus_mut();
        bus.cmd_roll = action.roll_cmd;
        bus.cmd_pitch = action.pitch_cmd;
        bus.cmd_yaw_rate = action.yaw_rate_cmd;
        bus.cmd_throttle = action.throttle.clamp(0.0, 1.0);

        // 2. Compute wind + dynamic pressure for the true-state write.
        let mut wind_vec = Vector3::zeros();
        if let Some(wind_env) = &mut self.wind {
            let vt_air = self.sim.state.true_airspeed(&Vector3::zeros());
            wind_vec = wind_env.total_wind(&self.sim.state, vt_air, self.config.dt);
        }
        let tas = self.sim.state.true_airspeed(&wind_vec);
        let q_dynamic = Atmosphere::at_altitude(self.sim.state.altitude()).dynamic_pressure(tas);

        // 3. Sample the sim state, run sensors → FC → actuators → devices.
        self.avionics
            .bus_mut()
            .write_true_state(&self.sim.state, &wind_vec, q_dynamic);
        self.avionics.step();

        // 4. Feed actual actuator outputs back into the physics.
        let (elev, ail, rud, thr) = self.avionics.bus().read_actuator_outputs();
        let controls = ControlInputs {
            elevator: elev,
            aileron: ail,
            rudder: rud,
            throttle: thr,
            flap: 0.0,
        };
        step(
            &mut self.sim.state,
            &self.sim.config,
            controls,
            self.wind.as_ref().map(|_| &wind_vec),
            self.config.dt,
            None,
        );

        // 5. Bookkeeping, observation, reward, termination.
        self.step_count += 1;
        let observation = self.observation();
        let reward = shaped_reward(&self.sim.state, &self.config);
        let terminated = self.sim.state.altitude() <= self.config.ground_altitude;
        let truncated = self.step_count >= self.config.max_steps;

        AvionicsEnvStep {
            observation,
            reward,
            terminated,
            truncated,
        }
    }

    /// Build the current observation array from the noisy bus sensor outputs.
    pub fn observation(&self) -> AvionicsObservation {
        let bus = self.avionics.bus();
        [
            bus.gyro.x,
            bus.gyro.y,
            bus.gyro.z,
            bus.accel.x,
            bus.accel.y,
            bus.accel.z,
            bus.gps_position_ned.x,
            bus.gps_position_ned.y,
            -bus.gps_position_ned.z,
            bus.gps_fix_quality as f64,
            bus.baro_altitude,
            bus.airspeed_indicated,
            bus.battery_voltage,
            bus.battery_capacity_remaining_pct,
            bus.actual_elevator_deg,
            bus.actual_aileron_deg,
            bus.actual_rudder_deg,
            bus.actual_esc_output,
            bus.sim_time,
        ]
    }

    /// Inject a fault; components apply their own failure behaviour.
    pub fn inject_fault(&mut self, flag: FaultFlag) {
        self.avionics.inject_fault(flag, true);
    }

    /// Clear a previously injected fault.
    pub fn clear_fault(&mut self, flag: FaultFlag) {
        self.avionics.inject_fault(flag, false);
    }

    /// Snapshot of the currently injected faults.
    pub fn faults(&self) -> FaultFlags {
        self.avionics.faults()
    }

    /// The crash penalty applied when the episode terminates by impact.
    pub fn crash_penalty(&self) -> f64 {
        -self.config.w_crash
    }

    /// Current step index within the episode (0-based).
    pub fn step_count(&self) -> usize {
        self.step_count
    }
}

/// Verification & Validation (V&V) for the OpenAI-gym-style environment
/// wrapper: config loading from disk, reset to trimmed level flight, state/
/// observation mapping and step indexing.
#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::load_config;

    fn test_env() -> Environment {
        let path = if std::path::Path::new("../aircraft.toml").exists() {
            "../aircraft.toml"
        } else {
            "aircraft.toml"
        };
        Environment::with_config(
            path,
            EnvConfig {
                dt: 0.01,
                max_steps: 10_000,
                ..Default::default()
            },
        )
        .expect("aircraft.toml should load")
    }

    #[test]
    fn reset_returns_level_flight_observation() {
        let mut env = test_env();
        let (obs, prev) = env.reset();
        assert_eq!(obs.len(), 12);
        assert_eq!(prev, 0);
        assert!((env.sim.state.altitude() - 1000.0).abs() < 1e-9);
        assert!((env.sim.state.airspeed() - 60.0).abs() < 1e-9);
        let (roll, pitch, _) = env.sim.state.euler_angles();
        assert!(roll.abs() < 1e-6 && pitch.abs() < 0.02);
    }

    #[test]
    fn no_op_action_stays_near_target() {
        let mut env = test_env();
        env.reset();
        let mut total = 0.0;
        let action = ControlAction::neutral();
        for _ in 0..500 {
            let r = env.step(action);
            total += r.reward;
            assert!(!r.terminated, "aircraft crashed");
            assert!(!r.truncated);
        }
        assert!(
            total > 0.0,
            "expected positive cumulative reward, got {total}"
        );
    }

    #[test]
    fn crashing_gives_very_negative_reward() {
        let mut env = test_env();
        env.reset();
        env.sim.state.pos_z = -50.0;
        env.sim.state.w = 20.0;
        let action = ControlAction {
            elevator: 0.5,
            aileron: 0.0,
            rudder: 0.0,
            throttle: 0.0,
            flaps: 0.0,
        };
        let mut crashed = false;
        let mut total = 0.0;
        for _ in 0..500 {
            let r = env.step(action);
            total += r.reward;
            if r.terminated {
                total += env.crash_penalty();
                crashed = true;
                break;
            }
        }
        assert!(crashed, "expected the aircraft to have crashed");
        assert!(total < 0.0, "expected negative cumulative reward, got {total}");
    }

    #[test]
    fn config_deserializes_from_disk() {
        let path = if std::path::Path::new("../aircraft.toml").exists() {
            "../aircraft.toml"
        } else {
            "aircraft.toml"
        };
        let cfg = load_config(path).expect("aircraft.toml should load");
        assert!(cfg.mass > 0.0 && cfg.thrust_max > 0.0);
    }

    #[cfg(feature = "full-avionics")]
    fn test_avionics_env() -> AvionicsEnvironment {
        let path = if std::path::Path::new("../aircraft.toml").exists() {
            "../aircraft.toml"
        } else {
            "aircraft.toml"
        };
        AvionicsEnvironment::with_config(
            path,
            EnvConfig {
                dt: 0.05,
                max_steps: 100_000,
                ..Default::default()
            },
        )
        .expect("aircraft.toml should load")
    }

    #[cfg(feature = "full-avionics")]
    #[test]
    fn avionics_reset_returns_sensor_observation() {
        let mut env = test_avionics_env();
        let (obs, prev) = env.reset();
        assert_eq!(obs.len(), AVIONICS_OBS_DIM);
        assert_eq!(prev, 0);
        // Sensors start idle (first sample fires on the first step) so the bus
        // still carries its nominal defaults.
        assert_eq!(obs[9], 3.0, "GPS starts with a 3D fix");
        assert!((obs[12] - 16.8).abs() < 0.1, "battery starts near full voltage");
        assert!((obs[13] - 100.0).abs() < 0.1, "battery starts fully charged");
    }

    #[cfg(feature = "full-avionics")]
    #[test]
    fn avionics_cruise_command_holds_flight() {
        let mut env = test_avionics_env();
        let (_, trim_thr) = env.sim.trim_level_flight(1000.0, 60.0);
        env.reset();
        let action = AvionicsAction::level_cruise(trim_thr);
        let mut alt_min = f64::INFINITY;
        let mut alt_max = f64::NEG_INFINITY;
        let mut last = None;
        for _ in 0..600 {
            // 600 * 0.05 s = 30 s of FC-stabilized cruise
            let r = env.step(action);
            assert!(!r.terminated, "FC-stabilized cruise crashed");
            alt_min = alt_min.min(env.sim.state.altitude());
            alt_max = alt_max.max(env.sim.state.altitude());
            last = Some(r);
        }
        let r = last.expect("stepped the environment");
        assert!(
            alt_min > 850.0 && alt_max < 1150.0,
            "FC must hold cruise altitude (range [{alt_min:.0}, {alt_max:.0}])"
        );
        assert!(
            r.observation[17] > 0.2,
            "ESC must be producing thrust at cruise, got {}",
            r.observation[17]
        );
        assert!(
            (r.observation[13] - 100.0).abs() < 2.0,
            "battery should be nearly undrained after 30 s (capacity {:.3}%, volt {:.3} V, esc {:.3})",
            r.observation[13],
            r.observation[12],
            r.observation[17]
        );
    }

    #[cfg(feature = "full-avionics")]
    #[test]
    fn avionics_battery_drains_over_cruise() {
        let mut env = test_avionics_env();
        let (_, trim_thr) = env.sim.trim_level_flight(1000.0, 60.0);
        env.reset();
        let action = AvionicsAction::level_cruise(trim_thr);
        let mut last = None;
        for _ in 0..1200 {
            // 60 s of cruise draws on the pack
            let r = env.step(action);
            assert!(!r.terminated);
            last = Some(r);
        }
        let r = last.unwrap();
        assert!(
            r.observation[13] < 99.99,
            "battery capacity must drain under cruise load, got {}%",
            r.observation[13]
        );
        assert!(
            r.observation[12] < 16.799,
            "battery voltage must sag under load, got {} V",
            r.observation[12]
        );
    }

    #[cfg(feature = "full-avionics")]
    #[test]
    fn avionics_gps_failure_degrades_observation() {
        let mut env = test_avionics_env();
        env.reset();
        let action = AvionicsAction::level_cruise(0.6);

        env.step(action);
        assert_eq!(env.observation()[9], 3.0, "GPS fix healthy before failure");

        env.inject_fault(FaultFlag::Gps);
        // GPS self-checks the flag on its next sample; a couple of steps
        // guarantee a 10 Hz sample elapses at dt=50 ms.
        env.step(action);
        env.step(action);
        let obs = env.observation();
        assert_eq!(obs[9], 0.0, "failed GPS must report no fix");
        assert_eq!(obs[6], 0.0, "failed GPS must zero the north position");
        assert_eq!(obs[7], 0.0, "failed GPS must zero the east position");

        env.clear_fault(FaultFlag::Gps);
        for _ in 0..4 {
            env.step(action);
        }
        assert_eq!(
            env.observation()[9],
            3.0,
            "GPS fix must recover after clearing the fault"
        );
    }

    #[cfg(feature = "full-avionics")]
    #[test]
    fn avionics_imu_failure_zeroes_gyro_observation() {
        let mut env = test_avionics_env();
        env.reset();
        env.inject_fault(FaultFlag::Imu);

        let r = env.step(AvionicsAction::level_cruise(0.6));
        assert_eq!(r.observation[0], 0.0, "gyro p must read zero on IMU failure");
        assert_eq!(r.observation[1], 0.0, "gyro q must read zero on IMU failure");
        assert_eq!(r.observation[2], 0.0, "gyro r must read zero on IMU failure");
    }

    #[cfg(feature = "full-avionics")]
    #[test]
    fn avionics_battery_depletion_cuts_thrust_and_glides() {
        let mut env = test_avionics_env();
        let (_, trim_thr) = env.sim.trim_level_flight(1000.0, 60.0);
        env.reset();
        env.inject_fault(FaultFlag::BatteryDepleted);
        let action = AvionicsAction::level_cruise(trim_thr.max(0.8));

        for _ in 0..1200 {
            // 60 s of "cruise" with a dead pack
            let r = env.step(action);
            assert!(
                r.observation[17] < 0.05,
                "depleted battery must produce no thrust, got {}",
                r.observation[17]
            );
        }
        assert!(
            env.sim.state.altitude() < 950.0,
            "with no thrust the aircraft must descend (alt {:.0} m)",
            env.sim.state.altitude()
        );
    }
}


