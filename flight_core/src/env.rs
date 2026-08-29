//! Reinforcement-learning environment wrapper around [`crate::Simulator`].

use crate::integrator::step;
use crate::{Simulator, WindConfig, WindEnvironment};
use nalgebra::Vector3;

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
    pub fn new(config_path: &str) -> Self {
        Self::with_config(config_path, EnvConfig::default())
    }

    /// Create an environment with a custom [`EnvConfig`].
    pub fn with_config(config_path: &str, config: EnvConfig) -> Self {
        let sim = Simulator::new(config_path);
        let wind = config.wind_config.clone().map(WindEnvironment::new);
        Self {
            sim,
            config,
            step_count: 0,
            wind,
        }
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

        step(
            &mut self.sim.state,
            &self.sim.config,
            elevator,
            aileron,
            rudder,
            throttle,
            flaps,
            wind_vec.as_ref(),
            self.config.dt,
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
        let cfg = &self.config;
        let s = &self.sim.state;

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

    /// The crash penalty applied when the episode terminates by impact.
    pub fn crash_penalty(&self) -> f64 {
        -self.config.w_crash
    }

    /// Current step index within the episode (0-based).
    pub fn step_count(&self) -> usize {
        self.step_count
    }
}

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
        let cfg = load_config(path);
        assert!(cfg.mass > 0.0 && cfg.thrust_max > 0.0);
    }
}
