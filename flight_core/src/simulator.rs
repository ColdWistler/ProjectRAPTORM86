//! High-level convenience wrapper for the flight dynamics engine.

use crate::config::{load_config, AircraftConfig};
use crate::integrator::step;
use crate::state::AircraftState;
use crate::terrain::Terrain;
use crate::aero::ControlInputs;

/// A high-level convenience wrapper bundling the aircraft configuration
/// with its current dynamic state and providing the primary integration
/// API used by the Godot GDExtension front-end.
pub struct Simulator {
    pub config: AircraftConfig,
    pub state: AircraftState,
}

impl Simulator {
    /// Load the aircraft configuration from `config_path` and initialize
    /// a fresh trimmed level flight state (at 1000 m, 60 m/s).
    pub fn new(config_path: &str) -> Result<Self, Box<dyn std::error::Error>> {
        let config = load_config(config_path)?;
        let mut state = AircraftState::default();
        state.trim_level_flight(&config, 1000.0, 60.0);
        Ok(Self { config, state })
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
    /// full 6-DOF manual controls (elevator, aileron, rudder, throttle)
    /// plus a trailing-edge flap deflection (radians) and an optional wind
    /// vector in the Earth NED frame (m/s). Pass `None` for still air.
    #[allow(clippy::too_many_arguments)]
    pub fn step_6dof(
        &mut self,
        elevator: f64,
        aileron: f64,
        rudder: f64,
        throttle: f64,
        flaps: f64,
        wind_earth: Option<&nalgebra::Vector3<f64>>,
        dt: f64,
        terrain: Option<&Terrain>,
    ) -> [f64; 12] {
        let controls = ControlInputs {
            elevator,
            aileron,
            rudder,
            throttle,
            flap: flaps,
            wing_sweep: 0.0,
        };
        step(
            &mut self.state,
            &self.config,
            controls,
            wind_earth,
            dt,
            terrain,
        );
        self.state.to_observation_array()
    }
}
