//! Out-and-back fixed-wing mission types, defaults, and stall-speed geometry.
//!
//! Provides [`MissionConfig`], [`MissionPhase`], [`MissionTermination`], and
//! seedable pickup sampling for the fixed-wing RL mission. Fixed-wing only —
//! no landing, taxi, or true hover.

use crate::atmosphere::Atmosphere;
use crate::config::AircraftConfig;
use nalgebra::Vector3;
use rand::Rng;
use serde::{Deserialize, Serialize};

pub const MISSION_FEATURE_DIM: usize = 13;

/// Wrap an angle to `(-pi, pi]`.
///
/// `rem_euclid(2pi) - pi` would map a straight-ahead bearing of `0` to `-pi`,
/// which reads as "the target is behind" and sends every steering law into a
/// permanent turn-around.
fn wrap_pi(angle: f64) -> f64 {
    let two_pi = 2.0 * std::f64::consts::PI;
    let wrapped = angle.rem_euclid(two_pi);
    if wrapped > std::f64::consts::PI {
        wrapped - two_pi
    } else {
        wrapped
    }
}

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
    fn wrap_pi_keeps_a_straight_ahead_bearing_at_zero() {
        // A target exactly ahead must report a bearing of 0, not +/-pi, or
        // every steering law reads it as "behind".
        let pi = std::f64::consts::PI;
        assert!(wrap_pi(0.0).abs() < 1e-12, "wrap_pi(0) = {}", wrap_pi(0.0));
        assert!((wrap_pi(pi) - pi).abs() < 1e-12);
        assert!((wrap_pi(-pi) - pi).abs() < 1e-12);
        assert!((wrap_pi(3.0 * pi) - pi).abs() < 1e-12);
        assert!((wrap_pi(2.5 * pi) - 0.5 * pi).abs() < 1e-12);
        assert!((wrap_pi(-0.25 * pi) + 0.25 * pi).abs() < 1e-12);
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

    fn mission_env() -> MissionEnvironment {
        MissionEnvironment::new(
            if std::path::Path::new("../m_sweep_mini.toml").exists() {
                "../m_sweep_mini.toml"
            } else {
                "m_sweep_mini.toml"
            },
            MissionConfig::default(),
            7,
        )
        .expect("mission env")
    }

    fn scenario_at_pickup(phase: MissionPhase, speed: f64) -> MissionScenario {
        MissionScenario {
            id: "test".into(),
            pickup_north: 0.0,
            pickup_east: 0.0,
            pickup_altitude: 1000.0,
            initial_phase: phase,
            initial_north: Some(0.0),
            initial_east: Some(0.0),
            initial_altitude: Some(1000.0),
            initial_airspeed: Some(speed),
        }
    }

    #[test]
    fn reset_reproduces_seed_geometry() {
        let mut a = mission_env();
        let mut b = mission_env();
        let ra = a.reset();
        let rb = b.reset();
        assert_eq!(ra.phase, MissionPhase::Outbound);
        assert_eq!(a.scenario().pickup_north, b.scenario().pickup_north);
        assert_eq!(a.scenario().pickup_east, b.scenario().pickup_east);
        assert!(ra.features.pickup_distance_norm.is_finite());
    }

    #[test]
    fn reset_with_scenario_pins_geometry() {
        let mut env = mission_env();
        let s = scenario_at_pickup(MissionPhase::Capture, 12.0);
        let r = env.reset_with_scenario(&s);
        assert_eq!(r.phase, MissionPhase::Capture);
        assert_eq!(env.scenario().pickup_north, 0.0);
        assert_eq!(env.capture_speed(), 1.15 * stall_speed(env.aircraft_config(), 1000.0));
    }

    #[test]
    fn entering_the_pickup_zone_switches_to_capture() {
        let mut env = mission_env();
        env.reset_with_scenario(&scenario_at_pickup(MissionPhase::Outbound, 55.0));
        let out = env.step(AvionicsAction::level_cruise(0.5));
        assert_eq!(env.phase(), MissionPhase::Capture);
        assert_eq!(env.capture_attempts(), 1);
        assert!(!out.signals.captured);
    }

    #[test]
    fn capture_requires_the_full_dwell() {
        let mut env = mission_env();
        let cfg = MissionConfig::default();
        let start = env.capture_speed() - 0.5;
        env.reset_with_scenario(&scenario_at_pickup(MissionPhase::Capture, start));
        for _ in 0..cfg.capture_dwell_steps - 1 {
            let out = env.step(AvionicsAction::level_cruise(0.0));
            assert!(!out.signals.captured);
        }
        let out = env.step(AvionicsAction::level_cruise(0.0));
        assert!(out.signals.captured);
        assert_eq!(out.signals.dwell_fraction, 1.0);
        assert_eq!(env.phase(), MissionPhase::Return);
    }

    #[test]
    fn captured_aircraft_returns_home_and_finishes() {
        let mut env = mission_env();
        let mut s = scenario_at_pickup(MissionPhase::Capture, 11.0);
        s.initial_phase = MissionPhase::Return;
        s.initial_north = Some(0.0);
        s.initial_east = Some(0.0);
        env.reset_with_scenario(&s);
        // Force the captured latch for the return leg; reset_with_scenario does it.
        let out = env.step(AvionicsAction::level_cruise(0.0));
        assert_eq!(env.phase(), MissionPhase::Done);
        assert_eq!(out.termination, MissionTermination::CapturedAndReturned);
        assert!(out.terminated);
        assert!(!out.truncated);
        assert!(out.signals.returned);
    }

    #[test]
    fn ground_impact_terminates() {
        let mut env = mission_env();
        let mut s = scenario_at_pickup(MissionPhase::Outbound, 55.0);
        s.initial_altitude = Some(0.0);
        env.reset_with_scenario(&s);
        let out = env.step(AvionicsAction::level_cruise(0.0));
        assert_eq!(out.termination, MissionTermination::GroundImpact);
        assert!(out.terminated);
        assert!(out.signals.crash);
    }

    #[test]
    fn out_of_bounds_terminates() {
        let mut env = mission_env();
        let mut s = scenario_at_pickup(MissionPhase::Outbound, 55.0);
        s.initial_north = Some(6000.0);
        env.reset_with_scenario(&s);
        let out = env.step(AvionicsAction::level_cruise(0.5));
        assert_eq!(out.termination, MissionTermination::OutOfBounds);
        assert!(out.signals.crash);
    }

    #[test]
    fn zero_budget_times_out_without_panicking() {
        let mut env = MissionEnvironment::new(
            if std::path::Path::new("../m_sweep_mini.toml").exists() {
                "../m_sweep_mini.toml"
            } else {
                "m_sweep_mini.toml"
            },
            MissionConfig { max_steps: 1, ..Default::default() },
            7,
        )
        .expect("mission env");
        env.reset();
        let out = env.step(AvionicsAction::level_cruise(0.5));
        assert_eq!(out.termination, MissionTermination::Timeout);
        assert!(out.truncated);
        assert!(!out.terminated);
    }

    #[test]
    fn on_target_features_stay_finite_at_zero_airspeed() {
        let mut env = mission_env();
        env.reset_with_scenario(&scenario_at_pickup(MissionPhase::Capture, 12.0));
        {
            // Zero body velocity makes both bearing and airspeed degenerate.
            let s = env.state_mut();
            s.u = 0.0;
            s.v = 0.0;
            s.w = 0.0;
        }
        let features = env.features();
        for v in features.to_array() {
            assert!(v.is_finite(), "non-finite feature: {v}");
        }
        assert_eq!(features.pickup_distance_norm, 0.0);
    }
}

#[derive(Debug, Clone)]
pub struct MissionReset {
    pub avionics: crate::env::AvionicsObservation,
    pub features: MissionFeatures,
    pub phase: MissionPhase,
}

#[derive(Debug, Clone, Copy)]
pub struct MissionStep {
    pub avionics: crate::env::AvionicsObservation,
    pub features: MissionFeatures,
    pub signals: MissionSignals,
    pub termination: MissionTermination,
    pub terminated: bool,
    pub truncated: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MissionContext {
    pub phase: MissionPhase,
    pub step: usize,
    pub captured: bool,
}

use crate::env::{AvionicsEnvironment, EnvConfig, AvionicsAction};
use rand::SeedableRng;

pub struct MissionEnvironment {
    inner: AvionicsEnvironment,
    config: MissionConfig,
    scenario: MissionScenario,
    phase: MissionPhase,
    captured: bool,
    dwell_steps: usize,
    step_count: usize,
    capture_attempts: u32,
    previous_pickup_distance: f64,
    previous_home_distance: f64,
    capture_speed: f64,
    v_stall: f64,
    rng: rand::rngs::StdRng,
    last_signals: MissionSignals,
    termination: MissionTermination,
    cached_features: MissionFeatures,
    tension: f64,
    tension_stable_steps: usize,
    /// Optional RL weather configuration threaded through to the inner
    /// [`AvionicsEnvironment`]. Set when the exam wants explicit weather
    /// scenes baked into the launch scenario.
    weather_config: Option<crate::weather::WeatherConfig>,
}

impl MissionEnvironment {
    pub fn new(config_path: &str, config: MissionConfig, seed: u64) -> Result<Self, Box<dyn std::error::Error>> {
        Self::new_with_weather(config_path, config, seed, None)
    }

    pub fn new_with_weather(
        config_path: &str,
        config: MissionConfig,
        seed: u64,
        weather_config: Option<crate::weather::WeatherConfig>,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        let env_config = EnvConfig {
            target_altitude: config.cruise_altitude,
            target_airspeed: config.cruise_airspeed,
            ground_altitude: config.ground_altitude,
            max_steps: config.max_steps,
            dt: config.dt,
            weather_config: weather_config.clone(),
            ..Default::default()
        };
        let avionics = AvionicsEnvironment::with_config(config_path, env_config)?;

        let capture_speed = config.resolve_capture_speed(avionics.aircraft_config());
        let v_stall = stall_speed(avionics.aircraft_config(), config.cruise_altitude);

        let scenario = MissionScenario {
            id: "random".into(),
            pickup_north: 0.0,
            pickup_east: 0.0,
            pickup_altitude: 1000.0,
            initial_phase: MissionPhase::Outbound,
            initial_north: None,
            initial_east: None,
            initial_altitude: None,
            initial_airspeed: None,
        };

        Ok(Self {
            inner: avionics,
            config,
            scenario,
            phase: MissionPhase::Outbound,
            captured: false,
            dwell_steps: 0,
            step_count: 0,
            capture_attempts: 0,
            previous_pickup_distance: 0.0,
            previous_home_distance: 0.0,
            capture_speed,
            v_stall,
            rng: rand::rngs::StdRng::seed_from_u64(seed),
            last_signals: MissionSignals::default(),
            termination: MissionTermination::None,
            cached_features: MissionFeatures::default(),
            tension: 0.0,
            tension_stable_steps: 0,
            weather_config: weather_config.clone(),
        })
    }


    pub fn reset(&mut self) -> MissionReset {
        let (north, east, altitude) = sample_pickup(&self.config, &mut self.rng);
        let s = MissionScenario {
            id: "random".into(),
            pickup_north: north,
            pickup_east: east,
            pickup_altitude: altitude,
            initial_phase: MissionPhase::Outbound,
            initial_north: Some(0.0),
            initial_east: Some(0.0),
            initial_altitude: Some(self.config.cruise_altitude),
            initial_airspeed: None,
        };
        self.reset_with_scenario(&s)
    }

    pub fn reset_with_scenario(&mut self, scenario: &MissionScenario) -> MissionReset {
        let start_north = scenario.initial_north.unwrap_or(0.0);
        let start_east = scenario.initial_east.unwrap_or(0.0);
        let start_alt = scenario.initial_altitude.unwrap_or(self.config.cruise_altitude);
        let start_speed = scenario.initial_airspeed.unwrap_or(self.capture_speed).max(1.0);
        self.inner.reset_trimmed(start_alt, start_speed);
        {
            let sm = self.inner.state_mut();
            sm.pos_x = start_north;
            sm.pos_y = start_east;
            sm.pos_z = -start_alt;
            sm.u = start_speed;
            sm.v = 0.0;
            sm.w = 0.0;
        }
        self.scenario = scenario.clone();
        self.phase = scenario.initial_phase;
        self.captured = self.phase == MissionPhase::Return;
        self.dwell_steps = 0;
        self.step_count = 0;
        self.capture_attempts = 0;
        self.last_signals = MissionSignals::default();
        self.termination = MissionTermination::None;
        {
            let s = self.inner.aircraft_state();
            self.previous_pickup_distance = ((s.pos_x - self.scenario.pickup_north).powi(2) + (s.pos_y - self.scenario.pickup_east).powi(2)).sqrt();
            self.previous_home_distance = (s.pos_x * s.pos_x + s.pos_y * s.pos_y).sqrt();
        }
        self.tension = 0.0;
        self.tension_stable_steps = 0;
        self.cached_features = self.compute_features();
        MissionReset { avionics: self.inner.observation(), features: self.cached_features.clone(), phase: self.phase }
    }

    /// Mission features for the current state.
    ///
    /// The reset observation used to carry a zeroed feature block, which told
    /// the agent it was already sitting on the pickup. The reset and every
    /// step now share this computation, so the features always describe the
    /// state the observation was taken in.
    fn compute_features(&self) -> MissionFeatures {
        let s = self.inner.aircraft_state();
        let heading = s.euler_angles().2;
        let pickup_bearing =
            f64::atan2(self.scenario.pickup_east - s.pos_y, self.scenario.pickup_north - s.pos_x)
                - heading;
        let pickup_bearing = wrap_pi(pickup_bearing);
        let home_bearing = f64::atan2(-s.pos_y, -s.pos_x) - heading;
        let home_bearing = wrap_pi(home_bearing);
        let pickup_distance = ((s.pos_x - self.scenario.pickup_north).powi(2)
            + (s.pos_y - self.scenario.pickup_east).powi(2))
        .sqrt();
        MissionFeatures {
            pickup_distance_norm: pickup_distance / 2500.0,
            pickup_bearing_sin: pickup_bearing.sin(),
            pickup_bearing_cos: pickup_bearing.cos(),
            pickup_altitude_error_norm: (self.scenario.pickup_altitude - s.altitude()) / 500.0,
            home_distance_norm: (s.pos_x * s.pos_x + s.pos_y * s.pos_y).sqrt() / self.config.home_radius,
            home_bearing_sin: home_bearing.sin(),
            home_bearing_cos: home_bearing.cos(),
            phase: match self.phase {
                MissionPhase::Outbound => 0.0,
                MissionPhase::Capture => 1.0,
                MissionPhase::Return => 2.0,
                MissionPhase::Done => 2.0,
            },
            speed_margin_norm: (s.airspeed() - self.v_stall) / 10.0,
            time_remaining_fraction: (1.0
                - (self.step_count as f64 / self.config.max_steps.max(1) as f64))
            .clamp(0.0, 1.0),
            capture_dwell_fraction: self.dwell_steps as f64 / self.config.capture_dwell_steps as f64,
            horizontal_offset_norm: pickup_distance / self.config.pickup_radius.max(1e-6),
            vertical_offset_norm: (s.altitude() - self.scenario.pickup_altitude).abs()
                / self.config.pickup_vertical_tolerance.max(1e-6),
        }
    }

    pub fn step(&mut self, action: AvionicsAction) -> MissionStep {
        let phase_before = self.phase;
        let inner_step = self.inner.step(action);
        self.step_count += 1;
        let inner_terminated = inner_step.terminated;
        let inner_truncated = inner_step.truncated;

        // Snapshot plane geometry up front (NED frame) so later code doesn't hold
        // an immutable borrow across the mutable cable-force application below.
        let (pickup_distance, pickup_altitude, current_speed, capture_speed) = {
            let s = self.inner.aircraft_state();
            (
                ((s.pos_x - self.scenario.pickup_north).powi(2)
                    + (s.pos_y - self.scenario.pickup_east).powi(2))
                    .sqrt(),
                self.scenario.pickup_altitude,
                s.airspeed(),
                self.capture_speed,
            )
        };

        let termination = if inner_terminated || self.inner.aircraft_state().altitude() <= self.config.ground_altitude {
            self.termination = MissionTermination::GroundImpact;
            MissionTermination::GroundImpact
        } else {
            let s_term = self.inner.aircraft_state();
            if s_term.pos_x.abs() > self.config.max_range
                || s_term.pos_y.abs() > self.config.max_range
            {
                self.termination = MissionTermination::OutOfBounds;
                MissionTermination::OutOfBounds
            } else if self.phase == MissionPhase::Done && phase_before == MissionPhase::Return {
                self.termination = MissionTermination::CapturedAndReturned;
                MissionTermination::CapturedAndReturned
            } else if inner_truncated {
                self.termination = MissionTermination::Timeout;
                MissionTermination::Timeout
            } else {
                match self.phase {
                    MissionPhase::Outbound => {
                        if pickup_distance <= self.config.pickup_radius {
                            self.phase = MissionPhase::Capture;
                            self.capture_attempts += 1;
                        }
                    }
                    MissionPhase::Capture => {
                        let within_capture = pickup_distance <= self.config.pickup_radius
                            && current_speed <= capture_speed;
                        if pickup_distance > 2.0 * self.config.pickup_radius {
                            self.phase = MissionPhase::Outbound;
                            self.dwell_steps = 0;
                        } else if within_capture {
                            self.dwell_steps += 1;
                        } else {
                            self.dwell_steps = 0;
                        }
                    }
                    MissionPhase::Return => {
                        let s = self.inner.aircraft_state();
                        let home_dist = (s.pos_x * s.pos_x + s.pos_y * s.pos_y).sqrt();
                        if home_dist <= self.config.home_radius
                            && (s.altitude() - self.config.cruise_altitude).abs() <= self.config.home_vertical_tolerance
                        {
                            self.phase = MissionPhase::Done;
                            self.termination = MissionTermination::CapturedAndReturned;
                        }
                    }
                    MissionPhase::Done => {}
                }
                self.termination
            }
        };

        // Cable-tension physics, driven by SkyhookConfig fields taken from
        // `m_sweep_mini.toml` [skyhook]: stiffness 1000, damping 65,
        // breakaway 2200 N. Applied as a force when inside the pickup zone
        // during Capture phase; capture fires only once the tension has been
        // stable for the full dwell window. If the tension exceeds the
        // breakaway load the mission terminates with a crash.
        if self.phase == MissionPhase::Capture {
            let skyhook = self.aircraft_config().skyhook.as_ref();
            if let Some(cfg) = skyhook {
                let rest_length = cfg.cable_length_max;
                let stiffness = cfg.cable_stiffness;
                let damping = cfg.cable_damping;
                let breakaway = cfg.tension_breakaway;
                let s = self.inner.aircraft_state();
                let within_zone = pickup_distance <= self.config.pickup_radius;
                let pickup_alt = self.scenario.pickup_altitude;
                if within_zone
                    && (s.altitude() - pickup_alt).abs() <= self.config.pickup_vertical_tolerance
                    && current_speed <= capture_speed
                {
                    let cable_vec = Vector3::new(self.scenario.pickup_north - s.pos_x,
                                                 self.scenario.pickup_east - s.pos_y,
                                                 -(pickup_alt - s.altitude()));
                    let d = cable_vec.norm();
                    if d > 1e-6 && rest_length > 0.0 {
                        let cable_dir = cable_vec / d;
                        let extension = d - rest_length;
                        let vel_along = s.velocity_ned().dot(&cable_dir);
                        let tension = (stiffness * extension - damping * vel_along).max(0.0);
                        self.tension = tension;
                        // Apply the tension as an acceleration over dt into the
                        // body-frame velocity.
                        let tension_force = cable_dir * tension;
                        let force_body = s.rotation_earth_to_body().inverse()
                            .transform_vector(&tension_force);
                        let dt = self.config.dt;
                        let mass = self.aircraft_config().mass.max(1e-6);
                        self.inner.state_mut().u += force_body.x * dt / mass;
                        self.inner.state_mut().v += force_body.y * dt / mass;
                        self.inner.state_mut().w += force_body.z * dt / mass;
                        if tension > breakaway {
                            self.termination = MissionTermination::GroundImpact;
                        }
                        if tension > 10.0 {
                            self.tension_stable_steps += 1;
                        } else {
                            self.tension_stable_steps = 0;
                        }
                    }
                }
            }
            // Transition to Return requires stable tension AND dwell completion.
            let stable = self.tension_stable_steps >= self.config.capture_dwell_steps
                || (self.tension == 0.0 && self.dwell_steps > 0);
            if self.dwell_steps >= self.config.capture_dwell_steps && stable {
                self.phase = MissionPhase::Return;
            }
        } else if self.phase == MissionPhase::Return {
            let s = self.inner.aircraft_state();
            let home_dist = (s.pos_x * s.pos_x + s.pos_y * s.pos_y).sqrt();
            if home_dist <= self.config.home_radius
                && (s.altitude() - self.config.cruise_altitude).abs() <= self.config.home_vertical_tolerance
            {
                self.phase = MissionPhase::Done;
                self.termination = MissionTermination::CapturedAndReturned;
            }
        }

        let signals = MissionSignals {
            progress_outbound: self.previous_pickup_distance.max(0.0) - pickup_distance.max(0.0),
            progress_inbound: 0.0,
            altitude_error: (pickup_altitude - self.inner.aircraft_state().altitude()).abs() / 500.0,
            speed_error: (self.inner.aircraft_state().airspeed() - capture_speed).abs() / capture_speed,
            capture_alignment: if pickup_distance > self.config.pickup_radius { 0.0 } else { ((pickup_distance - self.config.pickup_radius).max(0.0) / self.config.pickup_radius) + ((self.inner.aircraft_state().airspeed() - capture_speed).max(0.0) / capture_speed) },
            dwell_fraction: self.dwell_steps as f64 / self.config.capture_dwell_steps as f64,
            time_penalty: 1.0,
            energy_penalty: action.throttle.clamp(0.0, 1.0),
            crash: matches!(termination, MissionTermination::GroundImpact | MissionTermination::OutOfBounds),
            captured: (self.phase == MissionPhase::Capture && self.dwell_steps >= self.config.capture_dwell_steps)
                || self.phase == MissionPhase::Return
                || self.phase == MissionPhase::Done,
            returned: self.phase == MissionPhase::Done,
        };

        let features = self.compute_features();
        self.previous_pickup_distance = pickup_distance;
        self.cached_features = features.clone();
        self.last_signals = signals;
        MissionStep {
            avionics: self.inner.observation(),
            features,
            signals,
            termination,
            terminated: matches!(termination, MissionTermination::GroundImpact | MissionTermination::OutOfBounds | MissionTermination::CapturedAndReturned),
            truncated: termination == MissionTermination::Timeout,
        }
    }

    pub fn phase(&self) -> MissionPhase { self.phase }
    pub fn capture_attempts(&self) -> u32 { self.capture_attempts }
    pub fn capture_speed(&self) -> f64 { self.capture_speed }
    pub fn aircraft_config(&self) -> &AircraftConfig { self.inner.aircraft_config() }
    pub fn mission_config(&self) -> &MissionConfig { &self.config }
    pub fn scenario(&self) -> &MissionScenario { &self.scenario }
    pub fn pickup(&self) -> (f64, f64, f64) { let s = self.inner.aircraft_state(); (s.pos_x, s.pos_y, s.pos_z) }
    pub fn signals(&self) -> MissionSignals { self.last_signals }
    pub fn termination(&self) -> MissionTermination { self.termination }
    pub fn context(&self) -> MissionContext { MissionContext { phase: self.phase, step: self.step_count, captured: self.captured } }
    pub fn features(&self) -> MissionFeatures {
        self.cached_features.clone()
    }
    pub fn step_count(&self) -> usize { self.step_count }
    pub fn state(&self) -> &crate::state::AircraftState { self.inner.aircraft_state() }
    pub fn state_mut(&mut self) -> &mut crate::state::AircraftState { self.inner.state_mut() }
}
