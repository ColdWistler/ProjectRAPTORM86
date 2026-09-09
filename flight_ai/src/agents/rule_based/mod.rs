//! Classical PID rule-based agent.
//!
//! Serves as a deterministic baseline and a reference implementation of
//! the [`Agent`] trait. The PID controller holds altitude and airspeed
//! targets by driving elevator and throttle, with optional roll damping
//! via aileron.
//!
//! # Observation layout (matches [`flight_core::AircraftState::to_observation_array`])
//! | Index | Field           |
//! |-------|-----------------|
//! | 0     | pos_x (north)   |
//! | 1     | pos_z (down)    |
//! | 2     | u (body fwd)    |
//! | 3     | v (body right)  |
//! | 4     | w (body down)   |
//! | 5–8   | quaternion      |
//! | 9     | p (roll rate)   |
//! | 10    | q (pitch rate)  |
//! | 11    | r (yaw rate)    |

use crate::config::AgentConfig;
use crate::Agent;
use flight_core::ControlAction;

/// A proportional-integral-derivative controller for altitude and
/// airspeed tracking.
pub struct PidAgent {
    // --- targets ---
    target_alt: f64,
    target_airspeed: f64,

    // --- PID gains: altitude → elevator ---
    kp_alt: f64,
    ki_alt: f64,
    kd_alt: f64,

    // --- PID gains: airspeed → throttle ---
    kp_spd: f64,
    ki_spd: f64,
    kd_spd: f64,

    // --- roll damping gain ---
    kp_roll: f64,

    // --- integrator state (reset every episode) ---
    alt_integral: f64,
    spd_integral: f64,
    prev_alt_err: f64,
    prev_spd_err: f64,
}

impl PidAgent {
    /// Construct from TOML config. Missing keys fall back to sensible
    /// defaults for a light fixed-wing UAV at 1000 m / 60 m/s.
    pub fn from_config(cfg: &AgentConfig) -> Self {
        Self {
            target_alt: cfg.get_f64("target_altitude", 1000.0),
            target_airspeed: cfg.get_f64("target_airspeed", 60.0),

            kp_alt: cfg.get_f64("kp_alt", 0.015),
            ki_alt: cfg.get_f64("ki_alt", 0.001),
            kd_alt: cfg.get_f64("kd_alt", 0.008),

            kp_spd: cfg.get_f64("kp_spd", 0.04),
            ki_spd: cfg.get_f64("ki_spd", 0.005),
            kd_spd: cfg.get_f64("kd_spd", 0.01),

            kp_roll: cfg.get_f64("kp_roll", 0.30),

            alt_integral: 0.0,
            spd_integral: 0.0,
            prev_alt_err: 0.0,
            prev_spd_err: 0.0,
        }
    }
}

impl Agent for PidAgent {
    fn name(&self) -> &str {
        "pid"
    }

    fn obs_dim(&self) -> usize {
        12
    }

    fn act_dim(&self) -> usize {
        5
    }

    fn act(&self, obs: &[f64]) -> ControlAction {
        let altitude = -obs[1]; // pos_z is down
        let u = obs[2];
        let w = obs[4];
        let p = obs[9];
        let _q_rate = obs[10];

        let airspeed = (u * u + obs[3] * obs[3] + w * w).sqrt();

        // Altitude error (positive = too low → need to pitch up → negative elevator).
        let alt_err = self.target_alt - altitude;
        let elevator = -(self.kp_alt * alt_err
            + self.ki_alt * self.alt_integral
            + self.kd_alt * (alt_err - self.prev_alt_err));

        // Airspeed error (positive = too slow → increase throttle).
        let spd_err = self.target_airspeed - airspeed;
        let throttle = (0.5
            + self.kp_spd * spd_err
            + self.ki_spd * self.spd_integral
            + self.kd_spd * (spd_err - self.prev_spd_err))
            .clamp(0.0, 1.0);

        // Wings-level: oppose roll rate and roll angle.
        let roll = (2.0 * (obs[5] * obs[7] - obs[6] * obs[8]))
            .clamp(-1.0, 1.0)
            .asin();
        let aileron = (-self.kp_roll * (roll + 0.5 * p)).clamp(-0.35, 0.35);

        ControlAction {
            elevator: elevator.clamp(-0.35, 0.35),
            aileron,
            rudder: 0.0,
            throttle,
            flaps: 0.0,
        }
    }

    fn reset(&mut self) {
        self.alt_integral = 0.0;
        self.spd_integral = 0.0;
        self.prev_alt_err = 0.0;
        self.prev_spd_err = 0.0;
    }

    fn on_step(
        &mut self,
        observation: &[f64],
        _action: &ControlAction,
        _reward: f64,
        _next_observation: &[f64],
        done: bool,
    ) {
        let altitude = -observation[1];
        let u = observation[2];
        let w = observation[4];
        let airspeed = (u * u + observation[3] * observation[3] + w * w).sqrt();

        let alt_err = self.target_alt - altitude;
        let spd_err = self.target_airspeed - airspeed;

        if !done {
            self.alt_integral += alt_err;
            self.spd_integral += spd_err;
            // Anti-windup clamping.
            self.alt_integral = self.alt_integral.clamp(-200.0, 200.0);
            self.spd_integral = self.spd_integral.clamp(-50.0, 50.0);
        }

        self.prev_alt_err = alt_err;
        self.prev_spd_err = spd_err;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pid_produces_neutral_action_at_target() {
        let agent = PidAgent::from_config(&AgentConfig::new("pid"));
        // Craft an observation at exactly the target state.
        // pos_x=0, pos_z=-1000 (alt 1000), u=60, v=0, w≈0, quat=(1,0,0,0), rates=0
        let mut obs = [0.0f64; 12];
        obs[0] = 0.0;
        obs[1] = -1000.0; // altitude = 1000
        obs[2] = 60.0; // u = 60 m/s
        obs[3] = 0.0;
        obs[4] = 0.0;
        obs[5] = 1.0; // q0
        obs[6] = 0.0;
        obs[7] = 0.0;
        obs[8] = 0.0;
        obs[9] = 0.0;
        obs[10] = 0.0;
        obs[11] = 0.0;

        let action = agent.act(&obs);
        assert!(
            action.elevator.abs() < 0.02,
            "elevator should be near zero at target, got {}",
            action.elevator
        );
        assert!(
            (action.throttle - 0.5).abs() < 0.1,
            "throttle should be near 0.5 at target, got {}",
            action.throttle
        );
    }

    #[test]
    fn pid_climbs_when_below_target() {
        let agent = PidAgent::from_config(&AgentConfig::new("pid"));
        let mut obs = [0.0f64; 12];
        obs[1] = -800.0; // altitude = 800 (below 1000 target)
        obs[2] = 60.0;
        obs[5] = 1.0;
        let action = agent.act(&obs);
        // Below target → elevator should pitch up (negative elevator).
        assert!(
            action.elevator < 0.0,
            "should pitch up when below target, got {}",
            action.elevator
        );
    }
}
