//! Out-and-back fixed-wing mission types, defaults, and stall-speed geometry.
//!
//! Provides [`MissionConfig`], [`MissionPhase`], [`MissionTermination`], and
//! seedable pickup sampling for the fixed-wing RL mission. Fixed-wing only —
//! no landing, taxi, or true hover.

use crate::atmosphere::Atmosphere;
use crate::config::AircraftConfig;
use rand::Rng;
use serde::{Deserialize, Serialize};

pub const MISSION_FEATURE_DIM: usize = 13;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum MissionPhase {
    #[default]
    Outbound,
    Capture,
    Return,
    Done,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum MissionTermination {
    #[default]
    None,
    GroundImpact,
    OutOfBounds,
    Timeout,
    CapturedAndReturned,
}

#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize)]
pub struct MissionSignals {
    pub progress_outbound: f64,
    pub progress_inbound: f64,
    pub altitude_error: f64,
    pub speed_error: f64,
    /// 0.0 when perfectly aligned; positive when outside the zone or too fast.
    pub capture_alignment: f64,
    pub dwell_fraction: f64,
    pub time_penalty: f64,
    pub energy_penalty: f64,
    pub crash: bool,
    pub captured: bool,
    pub returned: bool,
}

#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize)]
pub struct MissionFeatures {
    pub pickup_distance_norm: f64,
    pub pickup_bearing_sin: f64,
    pub pickup_bearing_cos: f64,
    pub pickup_altitude_error_norm: f64,
    pub home_distance_norm: f64,
    pub home_bearing_sin: f64,
    pub home_bearing_cos: f64,
    pub phase: f64,
    pub speed_margin_norm: f64,
    pub time_remaining_fraction: f64,
    pub capture_dwell_fraction: f64,
    pub horizontal_offset_norm: f64,
    pub vertical_offset_norm: f64,
}

impl MissionFeatures {
    pub fn to_array(self) -> [f64; MISSION_FEATURE_DIM] {
        [
            self.pickup_distance_norm,
            self.pickup_bearing_sin,
            self.pickup_bearing_cos,
            self.pickup_altitude_error_norm,
            self.home_distance_norm,
            self.home_bearing_sin,
            self.home_bearing_cos,
            self.phase,
            self.speed_margin_norm,
            self.time_remaining_fraction,
            self.capture_dwell_fraction,
            self.horizontal_offset_norm,
            self.vertical_offset_norm,
        ]
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MissionScenario {
    pub id: String,
    pub pickup_north: f64,
    pub pickup_east: f64,
    pub pickup_altitude: f64,
    #[serde(default)]
    pub initial_phase: MissionPhase,
    #[serde(default)]
    pub initial_north: Option<f64>,
    #[serde(default)]
    pub initial_east: Option<f64>,
    #[serde(default)]
    pub initial_altitude: Option<f64>,
    #[serde(default)]
    pub initial_airspeed: Option<f64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MissionConfig {
    pub cruise_altitude: f64,
    pub cruise_airspeed: f64,
    pub pickup_radius: f64,
    pub pickup_vertical_tolerance: f64,
    pub capture_speed: Option<f64>,
    pub capture_dwell_steps: usize,
    pub home_radius: f64,
    pub home_vertical_tolerance: f64,
    pub pickup_distance_min: f64,
    pub pickup_distance_max: f64,
    pub pickup_altitude_min: f64,
    pub pickup_altitude_max: f64,
    pub max_steps: usize,
    pub max_range: f64,
    pub min_altitude: f64,
    pub max_altitude: f64,
    pub ground_altitude: f64,
    pub dt: f64,
}

impl Default for MissionConfig {
    fn default() -> Self {
        Self {
            cruise_altitude: 1000.0,
            cruise_airspeed: 55.0,
            pickup_radius: 40.0,
            pickup_vertical_tolerance: 15.0,
            capture_speed: None,
            capture_dwell_steps: 15,
            home_radius: 100.0,
            home_vertical_tolerance: 20.0,
            pickup_distance_min: 1000.0,
            pickup_distance_max: 2000.0,
            pickup_altitude_min: 800.0,
            pickup_altitude_max: 1200.0,
            max_steps: 4500,
            max_range: 5000.0,
            min_altitude: 100.0,
            max_altitude: 3000.0,
            ground_altitude: 0.0,
            dt: 1.0 / 30.0,
        }
    }
}

impl MissionConfig {
    /// `capture_speed` when set, otherwise `1.15 × stall_speed(aircraft, cruise_altitude)`.
    pub fn resolve_capture_speed(&self, aircraft: &AircraftConfig) -> f64 {
        self.capture_speed
            .unwrap_or_else(|| 1.15 * stall_speed(aircraft, self.cruise_altitude))
    }
}

/// Clean-config stall speed at `altitude`, using `cl0 + cla·alpha_stall_pos`.
pub fn stall_speed(aircraft: &AircraftConfig, altitude: f64) -> f64 {
    let cl_max = (aircraft.cl0 + aircraft.cla * aircraft.alpha_stall_pos).max(0.1);
    let density = Atmosphere::at_altitude(altitude).density;
    (2.0 * aircraft.mass * 9.80665 / (density * aircraft.wing_area * cl_max)).sqrt()
}

/// Uniform pickup sample in polar coordinates. Returns `(north, east, altitude)`.
pub fn sample_pickup(config: &MissionConfig, rng: &mut impl Rng) -> (f64, f64, f64) {
    let bearing = rng.gen::<f64>() * std::f64::consts::TAU;
    let distance = rng.gen_range(config.pickup_distance_min..=config.pickup_distance_max);
    let altitude = rng.gen_range(config.pickup_altitude_min..=config.pickup_altitude_max);
    (distance * bearing.cos(), distance * bearing.sin(), altitude)
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;
    use rand::rngs::StdRng;

    fn mini() -> AircraftConfig {
        let path = if std::path::Path::new("../m_sweep_mini.toml").exists() {
            "../m_sweep_mini.toml"
        } else {
            "m_sweep_mini.toml"
        };
        AircraftConfig::from_file(path).expect("m_sweep_mini.toml should load")
    }

    #[test]
    fn mission_config_defaults_are_sane() {
        let c = MissionConfig::default();
        assert_eq!(c.cruise_altitude, 1000.0);
        assert_eq!(c.cruise_airspeed, 55.0);
        assert_eq!(c.pickup_radius, 40.0);
        assert_eq!(c.capture_dwell_steps, 15);
        assert_eq!(c.max_steps, 4500);
        assert_eq!(c.dt, 1.0 / 30.0);
        assert!(c.pickup_distance_min < c.pickup_distance_max);
        assert!(c.pickup_altitude_min < c.pickup_altitude_max);
        assert!(c.min_altitude < c.max_altitude);
    }

    #[test]
    fn stall_speed_matches_m_sweep_expectation() {
        let v = stall_speed(&mini(), 1000.0);
        assert!((9.0..=11.0).contains(&v), "stall speed was {v}");
    }

    #[test]
    fn capture_speed_defaults_to_stall_margin() {
        let c = MissionConfig::default();
        let v = c.resolve_capture_speed(&mini());
        assert!((v - 1.15 * stall_speed(&mini(), c.cruise_altitude)).abs() < 1e-9);
    }

    #[test]
    fn explicit_capture_speed_overrides_stall() {
        let c = MissionConfig {
            capture_speed: Some(13.0),
            ..Default::default()
        };
        assert_eq!(c.resolve_capture_speed(&mini()), 13.0);
    }

    #[test]
    fn seeded_pickup_is_reproducible_and_in_range() {
        let c = MissionConfig::default();
        let mut a = StdRng::seed_from_u64(7);
        let mut b = StdRng::seed_from_u64(7);
        let mut different = StdRng::seed_from_u64(8);
        let pa = sample_pickup(&c, &mut a);
        let pb = sample_pickup(&c, &mut b);
        let pc = sample_pickup(&c, &mut different);
        assert_eq!(pa, pb);
        assert_ne!(pa, pc);
        let dist = (pa.0 * pa.0 + pa.1 * pa.1).sqrt();
        assert!((c.pickup_distance_min..=c.pickup_distance_max).contains(&dist));
        assert!((c.pickup_altitude_min..=c.pickup_altitude_max).contains(&pa.2));
    }
}
