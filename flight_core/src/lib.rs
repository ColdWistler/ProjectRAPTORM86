//! # flight_core
//!
//! A pure-Rust 6-DOF aircraft flight dynamics engine for reinforcement
//! learning training and real-time visualization. This crate implements
//! full rigid-body dynamics using a quaternion-based attitude representation,
//! 1976 US Standard Atmosphere, nonlinear post-stall aerodynamics, and an
//! RK4 numerical integrator.

pub mod aero;
pub mod atmosphere;
pub mod config;
pub mod env;
pub mod integrator;
pub mod state;

pub use atmosphere::Atmosphere;
pub use config::AircraftConfig;
pub use env::{ControlAction, Environment, EnvConfig, EnvStep, Observation};
pub use nalgebra;
pub use state::AircraftState;

use crate::config::load_config;
use crate::integrator::step;

/// A high-level convenience wrapper bundling the aircraft configuration
/// with its current dynamic state and providing the primary integration
/// API used by the Python and Bevy front-ends.
pub struct Simulator {
    pub config: AircraftConfig,
    pub state: AircraftState,
}

impl Simulator {
    /// Load the aircraft configuration from `config_path` and initialize
    /// a fresh trimmed level flight state (at 1000 m, 60 m/s).
    pub fn new(config_path: &str) -> Self {
        let config = load_config(config_path);
        let mut state = AircraftState::default();
        state.trim_level_flight(&config, 1000.0, 60.0);
        Self { config, state }
    }

    /// Reset the sim back to steady, level flight at 1000 m altitude and
    /// 60 m/s forward speed with exact aerodynamic equilibrium.
    pub fn reset(&mut self) -> (f64, f64) {
        self.state.trim_level_flight(&self.config, 1000.0, 60.0)
    }

    /// Trims the simulation to straight and level flight at specified altitude and speed.
    pub fn trim_level_flight(&mut self, altitude: f64, speed: f64) -> (f64, f64) {
        self.state.trim_level_flight(&self.config, altitude, speed)
    }

    /// Advance the simulation one time step of length `dt` seconds with
    /// full 6-DOF manual controls (elevator, aileron, rudder, throttle).
    pub fn step_6dof(
        &mut self,
        elevator: f64,
        aileron: f64,
        rudder: f64,
        throttle: f64,
        dt: f64,
    ) -> [f64; 12] {
        step(
            &mut self.state,
            &self.config,
            elevator,
            aileron,
            rudder,
            throttle,
            dt,
        );
        self.state.to_observation_array()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::integrator::step;

    #[test]
    fn level_flight_maintains_altitude_and_velocity() {
        let config = aircraft_default();
        let mut state = AircraftState::default();
        let (elev_trim, throttle_trim) = state.trim_level_flight(&config, 1000.0, 60.0);

        let dt = 0.01;
        let steps = 500; // 5 seconds of open-loop simulation

        let start = state.clone();
        for _ in 0..steps {
            step(&mut state, &config, elev_trim, 0.0, 0.0, throttle_trim, dt);
        }

        let alt_start = -start.pos_z;
        let alt_end = -state.pos_z;
        let speed_start = start.airspeed();
        let speed_end = state.airspeed();

        let tol_pos = 5.0; // meters over 300m traveled
        let tol_vel = 1.0; // m/s

        assert!(
            (alt_end - alt_start).abs() < tol_pos,
            "altitude drifted by {} m (start {}, end {})",
            (alt_end - alt_start).abs(),
            alt_start,
            alt_end
        );
        assert!(
            (speed_end - speed_start).abs() < tol_vel,
            "speed drifted by {} m/s (start {}, end {})",
            (speed_end - speed_start).abs(),
            speed_start,
            speed_end
        );
    }

    fn aircraft_default() -> AircraftConfig {
        AircraftConfig {
            mass: 1100.0,
            wing_area: 16.2,
            wing_span: 11.0,
            chord: 1.47,
            ixx: 1300.0,
            iyy: 1900.0,
            izz: 2700.0,
            cl0: 0.3,
            cla: 5.5,
            cd0: 0.025,
            k_drag: 0.04,
            cm0: 0.0,
            cma: -0.5,
            cmq: -12.0,
            cme: -1.0,
            thrust_max: 5000.0,
            oswald_e: 0.80,
            alpha_stall_pos: 16.0_f64.to_radians(),
            alpha_stall_neg: -12.0_f64.to_radians(),
            cd_max: 1.95,
            mach_crit: 0.65,
            cy_beta: -0.31,
            cy_dr: 0.15,
            cl_beta: -0.09,
            cl_p: -0.45,
            cl_r: 0.10,
            cl_da: 0.16,
            cl_dr: 0.01,
            cn_beta: 0.06,
            cn_p: -0.03,
            cn_r: -0.10,
            cn_da: -0.01,
            cn_dr: -0.07,
        }
    }
}
