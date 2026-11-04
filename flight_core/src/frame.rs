//! Per-step projection of the aircraft's **true** flight state.
//!
//! The reinforcement-learning observation vector ([`crate::env::AvionicsObservation`])
//! is deliberately impoverished: it carries only the *noisy* sensor bus the
//! agent is allowed to perceive (19 channels, no attitude, no velocity, no
//! position-derivative). That is correct for training — the agent must learn
//! to fly from what a real avionics package provides — but it makes the
//! aircraft impossible to *watch*.
//!
//! [`FlightFrame`] is the complement: a single step's ground-truth state,
//! captured from the same [`AircraftState`] the physics integrates, plus the
//! command the agent issued and the actuator feedback the airframe actually
//! produced. It is what a live training viewer draws.
//!
//! Two uses make the true/perceived split worth keeping:
//!
//! - **Flight visualisation** — position, attitude and airspeed are needed in
//!   Earth-frame terms, which the observation never provides.
//! - **Sensor-noise debugging** — `baro_altitude` vs [`Self::altitude`] and
//!   `airspeed_indicated` vs [`Self::airspeed`] let you see *why* an episode
//!   ended where it did. Crashes caused by noise are otherwise invisible.
//!
//! [`FlightFrame`] is `f64` and ergonomic; it is deliberately *not* the wire
//! format. Serialisation downcasts to a packed `f32` struct in the consumer
//! (see `rl_agent::stream`) so this type stays free to grow without breaking
//! a binary layout.

use crate::avionics::AvionicsBus;
use crate::state::AircraftState;
use nalgebra::Vector3;

/// Ground-truth state for one environment step.
///
/// Angles are radians, distances metres, velocities m/s, rates rad/s.
/// Position uses Earth-frame **NED**: north positive north, east positive
/// east, [`Self::down`] positive downward.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FlightFrame {
    /// Simulation time since the environment was reset.
    pub sim_time: f64,

    // ---- Earth-frame kinematics ----
    /// North coordinate (m).
    pub north: f64,
    /// East coordinate (m).
    pub east: f64,
    /// Down coordinate (m, positive downward).
    pub down: f64,
    /// Altitude above sea level (m) — `-down`, named for readability.
    pub altitude: f64,
    /// North velocity of the airframe over the ground (m/s).
    pub vel_north: f64,
    /// East velocity of the airframe over the ground (m/s).
    pub vel_east: f64,
    /// Down velocity of the airframe over the ground (m/s, positive falling).
    pub vel_down: f64,

    // ---- Attitude ----
    /// Bank angle (rad, positive right-wing-down).
    pub roll: f64,
    /// Pitch angle (rad, positive nose-up).
    pub pitch: f64,
    /// Heading (rad, clockwise from north).
    pub yaw: f64,

    // ---- Body rates ----
    /// Roll rate (rad/s).
    pub p: f64,
    /// Pitch rate (rad/s).
    pub q: f64,
    /// Yaw rate (rad/s).
    pub r: f64,

    // ---- Air data ----
    /// True airspeed including wind (m/s).
    pub airspeed: f64,
    /// Angle of attack (rad).
    pub alpha: f64,
    /// Sideslip angle (rad).
    pub beta: f64,
    /// Wind vector in Earth NED (m/s).
    pub wind_north: f64,
    pub wind_east: f64,
    pub wind_down: f64,

    // ---- What the agent commanded ----
    /// Commanded bank offset (rad, pre-PID).
    pub cmd_roll: f64,
    /// Commanded pitch offset (rad, pre-PID).
    pub cmd_pitch: f64,
    /// Commanded yaw rate (rad/s, pre-PID).
    pub cmd_yaw_rate: f64,
    /// Commanded throttle (0..1, pre-PID).
    pub cmd_throttle: f64,

    // ---- What the airframe actually did ----
    /// Elevator deflection (deg).
    pub elevator_deg: f64,
    /// Aileron deflection (deg).
    pub aileron_deg: f64,
    /// Rudder deflection (deg).
    pub rudder_deg: f64,
    /// ESC output / actual throttle (0..1).
    pub throttle: f64,

    // ---- What the agent believes ----
    /// Noisy barometric altitude (m) — compare against [`Self::altitude`].
    pub baro_altitude: f64,
    /// Noisy indicated airspeed (m/s) — compare against [`Self::airspeed`].
    pub airspeed_indicated: f64,
    /// Battery terminal voltage (V).
    pub battery_voltage: f64,
    /// Battery charge remaining (%).
    pub battery_pct: f64,
    /// GPS fix quality (0 = none, 3 = 3D).
    pub gps_fix_quality: u8,
}

impl FlightFrame {
    /// Capture the current true state and perceived sensor state.
    ///
    /// `state` must be the state the physics is currently at; `bus` the bus
    /// the avionics stack last wrote. Passing mismatched pairs yields a
    /// plausible-looking but wrong frame, so both should be sampled together
    /// at the end of a step.
    pub fn capture(state: &AircraftState, bus: &AvionicsBus) -> Self {
        // `rotation_earth_to_body` is Earth→body, so its inverse takes a
        // body-frame velocity to Earth frame. This is the same transform
        // `AircraftState::flight_path_angle` uses for the ground track.
        let body_to_earth = state.rotation_earth_to_body().inverse();
        let vel_earth = body_to_earth.transform_vector(&Vector3::new(state.u, state.v, state.w));

        // Reuse the tested Euler extraction rather than re-deriving the
        // quaternion convention — a silent sign flip here would show up as an
        // aircraft flying inverted, which is exactly the kind of bug that
        // costs an afternoon.
        let (roll, pitch, yaw) = state.euler_angles();

        Self {
            sim_time: bus.sim_time,

            north: state.pos_x,
            east: state.pos_y,
            down: state.pos_z,
            altitude: state.altitude(),
            vel_north: vel_earth.x,
            vel_east: vel_earth.y,
            vel_down: vel_earth.z,

            roll,
            pitch,
            yaw,

            p: state.p,
            q: state.q,
            r: state.r,

            airspeed: bus.true_airspeed,
            alpha: bus.true_alpha,
            beta: bus.true_beta,
            wind_north: bus.true_wind.x,
            wind_east: bus.true_wind.y,
            wind_down: bus.true_wind.z,

            cmd_roll: bus.cmd_roll,
            cmd_pitch: bus.cmd_pitch,
            cmd_yaw_rate: bus.cmd_yaw_rate,
            cmd_throttle: bus.cmd_throttle,

            elevator_deg: bus.actual_elevator_deg,
            aileron_deg: bus.actual_aileron_deg,
            rudder_deg: bus.actual_rudder_deg,
            throttle: bus.actual_esc_output,

            baro_altitude: bus.baro_altitude,
            airspeed_indicated: bus.airspeed_indicated,
            battery_voltage: bus.battery_voltage,
            battery_pct: bus.battery_capacity_remaining_pct,
            gps_fix_quality: bus.gps_fix_quality,
        }
    }

    /// Ground-track speed over the ground (m/s).
    pub fn ground_speed(&self) -> f64 {
        (self.vel_north * self.vel_north + self.vel_east * self.vel_east).sqrt()
    }

    /// Vertical speed (m/s, positive climbing).
    pub fn climb_rate(&self) -> f64 {
        -self.vel_down
    }

    /// Barometric altitude error against the true altitude (m) — a direct
    /// read on how much sensor noise is moving the aircraft's perceived
    /// altitude.
    pub fn baro_altitude_error(&self) -> f64 {
        self.baro_altitude - self.altitude
    }

    /// Indicated-airspeed error against true airspeed (m/s).
    pub fn airspeed_error(&self) -> f64 {
        self.airspeed_indicated - self.airspeed
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::AircraftConfig;
    use crate::state::AircraftState;
    use nalgebra::{Quaternion, UnitQuaternion};
    use std::f64::consts::FRAC_PI_2;

    fn config() -> AircraftConfig {
        let path = format!("{}/../aircraft.toml", env!("CARGO_MANIFEST_DIR"));
        AircraftConfig::from_file(&path).expect("aircraft.toml should load")
    }

    fn level_state() -> AircraftState {
        let mut s = AircraftState::default();
        s.trim_level_flight(&config(), 1000.0, 60.0);
        s
    }

    #[test]
    fn capture_reports_level_flight_consistently() {
        let f = FlightFrame::capture(&level_state(), &AvionicsBus::default());

        assert!(f.roll.abs() < 1e-6, "roll {} should be 0", f.roll);
        assert!(f.pitch.abs() < 0.02, "pitch {} should be ~0", f.pitch);
        // Altitude is `-down` by definition; the two must never drift apart.
        assert!((f.altitude + f.down).abs() < 1e-9);
        assert!((f.altitude - 1000.0).abs() < 1.0, "alt {}", f.altitude);
    }

    #[test]
    fn ground_track_is_parallel_to_the_nose() {
        let mut state = level_state();
        // Yaw the airframe off the north axis. The exact sign convention of the
        // Earth frame is not what is under test here — what matters is that
        // the projected velocity keeps pointing where the nose points, which
        // is what a viewer's ground track is drawn from.
        let yawed = UnitQuaternion::from_axis_angle(&Vector3::z_axis(), FRAC_PI_2)
            * state.rotation_earth_to_body();
        let q: Quaternion<f64> = *yawed;
        state.q0 = q.w;
        state.q1 = q.i;
        state.q2 = q.j;
        state.q3 = q.k;

        let f = FlightFrame::capture(&state, &AvionicsBus::default());
        let (forward, _, _) = state.body_axes_in_earth();
        let track = Vector3::new(f.vel_north, f.vel_east, f.vel_down).normalize();
        assert!(
            track.dot(&forward) > 0.999,
            "ground track {} is not along the nose {}",
            track,
            forward
        );
        // Yawing must not change the magnitude of the velocity.
        assert!(
            (f.ground_speed() - 60.0).abs() < 2.0,
            "ground speed drifted to {}",
            f.ground_speed()
        );
        // Sanity: the track really did swing off north, rather than the test
        // passing because the rotation was a no-op.
        let straight = FlightFrame::capture(&level_state(), &AvionicsBus::default());
        assert!(
            f.vel_east.abs() > 50.0 && straight.vel_east.abs() < 1.0,
            "yaw did not move the track: yawed east={} straight east={}",
            f.vel_east,
            straight.vel_east
        );
    }

    #[test]
    fn true_and_perceived_channels_are_distinct() {
        let state = level_state();
        let mut bus = AvionicsBus::default();
        // A real step writes ground truth onto the bus; do the same here so the
        // `true_*` fields hold sensible values. The sensor channels stay at
        // their un-written defaults, which is exactly the mismatch we want.
        bus.write_true_state(&state, &Vector3::zeros(), 0.0);

        let f = FlightFrame::capture(&state, &bus);
        assert!(
            (f.airspeed - 60.0).abs() < 2.0,
            "true airspeed {} should track the trimmed 60 m/s",
            f.airspeed
        );
        // Baro altitude reads 0 against a 1000 m state. If these fields ever
        // aliased the true values this assertion would pass trivially, so
        // checking they diverge is what makes it a real guard against
        // someone "simplifying" capture() to read truth twice.
        assert!(
            f.baro_altitude_error().abs() > 100.0,
            "baro error {} — perceived channel looks aliased",
            f.baro_altitude_error()
        );
        assert!(
            f.airspeed_error().abs() > 10.0,
            "ias error {} — perceived channel looks aliased",
            f.airspeed_error()
        );
    }

    #[test]
    fn climb_rate_inverts_the_down_axis() {
        let mut f = FlightFrame::capture(&level_state(), &AvionicsBus::default());
        f.vel_down = -3.0; // descending
        assert!((f.climb_rate() - 3.0).abs() < 1e-9);
    }
}
