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
pub mod shape;
pub mod state;
pub mod terrain;
pub mod wind;

pub use atmosphere::Atmosphere;
pub use config::AircraftConfig;
pub use env::{ControlAction, Environment, EnvConfig, EnvStep, Observation};
pub use nalgebra;
pub use state::AircraftState;
pub use terrain::{Terrain, TerrainHill};
pub use wind::{TurbulenceIntensity, WindConfig, WindEnvironment};

use crate::config::load_config;
use crate::integrator::step;

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
    /// full 6-DOF manual controls (elevator, aileron, rudder, throttle)
    /// plus a trailing-edge flap deflection (radians) and an optional wind
    /// vector in the Earth NED frame (m/s). Pass `None` for still air.
    ///
    /// `terrain` (optional) enables terrain ground collision, ground-effect
    /// lift and orographic vertical wind. Pass `None` to fly over the flat
    /// datum plane (historical behaviour).
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
        step(
            &mut self.state,
            &self.config,
            elevator,
            aileron,
            rudder,
            throttle,
            flaps,
            wind_earth,
            dt,
            terrain,
        );
        self.state.to_observation_array()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::aero::{
    compute_forces, compute_forces_moments, compute_forces_with_terrain, compute_moments,
    ground_effect_factor,
};
    use crate::integrator::step;
    use nalgebra::Vector3;

    #[test]
    fn level_flight_maintains_altitude_and_velocity() {
        let config = aircraft_default();
        let mut state = AircraftState::default();
        let (elev_trim, throttle_trim) = state.trim_level_flight(&config, 1000.0, 60.0);

        let dt = 0.01;
        let steps = 500; // 5 seconds of open-loop simulation

        let start = state.clone();
        for _ in 0..steps {
            step(&mut state, &config, elev_trim, 0.0, 0.0, throttle_trim, 0.0, None, dt, None);
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

    #[test]
    fn throttling_up_climbs_instead_of_diving() {
        let config = aircraft_default();
        let mut state = AircraftState::default();
        let (elev_trim, throttle_trim) = state.trim_level_flight(&config, 1000.0, 60.0);

        // Push throttle well above the level-flight trim setting.
        let throttle_high = throttle_trim + 0.25;

        let dt = 1.0 / 60.0;
        let steps = 1800; // 30 seconds
        for _ in 0..steps {
            step(
                &mut state,
                &config,
                elev_trim,
                0.0,
                0.0,
                throttle_high,
                0.0,
                None,
                dt,
                None,
            );
        }

        let alt_end = state.altitude();
        assert!(
            alt_end > 1000.0,
            "after 30s of extra throttle the aircraft dove to {} m instead of climbing",
            alt_end
        );
        // The climb exchange dips the airspeed slightly below the 60 m/s trim
        // value (phugoid); just ensure it never approaches the ~25 m/s stall.
        assert!(
            state.airspeed() > 45.0,
            "airspeed collapsed to {} m/s (stall/mush detected)",
            state.airspeed()
        );
    }

    #[test]
    fn sustained_throttle_climb_stays_bounded() {
        // A real fixed-stick, power-on airplane settles into a steady climb
        // with a lightly damped phugoid: airspeed wanders only a few m/s
        // around trim and the altitude gain per minute stays modest. This
        // guards against the runaway near-vertical zooms that result from
        // spurious energy leaks in the 6-DOF coupling.
        let config = aircraft_default();
        let mut state = AircraftState::default();
        let (elev_trim, throttle_trim) = state.trim_level_flight(&config, 1000.0, 60.0);
        let throttle_high = (throttle_trim + 0.35).min(1.0);

        let dt = 1.0 / 60.0;
        let mut tas_min = f64::INFINITY;
        let mut tas_max = f64::NEG_INFINITY;
        for _ in 0..3600 {
            step(
                &mut state,
                &config,
                elev_trim,
                0.0,
                0.0,
                throttle_high,
                0.0,
                None,
                dt,
                None,
            );
            tas_min = tas_min.min(state.airspeed());
            tas_max = tas_max.max(state.airspeed());
        }

        let climb_alt = state.altitude() - 1000.0;
        assert!(
            climb_alt > 0.0 && climb_alt < 400.0,
            "sustained throttle should climb steadily without runaway (gained {} m in 60 s)",
            climb_alt
        );
        assert!(
            tas_min > 45.0 && tas_max < 90.0,
            "airspeed swung beyond the bounded phugoid envelope (TAS [{tas_min:.1}, {tas_max:.1}])"
        );
    }

    #[test]
    fn pitch_oscillation_damps_out() {
        let config = aircraft_default();
        let mut state = AircraftState::default();
        let (elev_trim, throttle_trim) = state.trim_level_flight(&config, 1000.0, 60.0);
        let trim_alpha = state.angle_of_attack();

        let dt = 1.0 / 60.0;

        // Pull briefly (1 s, ~{pull} deg) to excite the short-period pitch
        // oscillation (AoA, pitch rate), then release back to trim. The
        // long-period phugoid keeps the nose slowly rising for ~38 s, so we
        // assert on the short-period response (alpha & pitch-rate return to
        // near-trim), not the pitch attitude.
        for _ in 0..60 {
            step(
                &mut state,
                &config,
                elev_trim - 0.06,
                0.0,
                0.0,
                throttle_trim,
                0.0,
                None,
                dt,
                None,
            );
        }
        assert!(
            state.angle_of_attack().to_degrees() < 10.0,
            "prod pushed the aircraft to stall (alpha {})",
            state.angle_of_attack().to_degrees()
        );

        // Integrate for 3 s of released flight; the short-period oscillation
        // (alpha, q) must damp out within this window, visible as a deep
        // minimum in |q| shortly after release. The long-period phugoid
        // (~38 s) makes the pitch rate slowly grow again afterwards, so we
        // assert on the damping event, not the final rate.
        let mut q_min = f64::INFINITY;
        for _ in 0..180 {
            step(
                &mut state,
                &config,
                elev_trim,
                0.0,
                0.0,
                throttle_trim,
                0.0,
                None,
                dt,
                None,
            );
            q_min = q_min.min(state.q.abs());
        }

        let (_, pitch, _) = state.euler_angles();
        assert!(
            q_min < 0.01,
            "short-period pitch oscillation not damped after release (min |q| in 3 s = {q_min})"
        );
        assert!(
            (state.angle_of_attack() - trim_alpha).abs() < 0.02,
            "angle of attack did not return to trim after release (alpha {})",
            state.angle_of_attack()
        );
        // The pitch attitude can legitimately drift on the slow phugoid;
        // just sanity-check we didn't diverge to vertical.
        assert!(
            pitch.abs() < 60.0_f64.to_radians(),
            "pitch diverged on the phugoid: {:.1} deg",
            pitch.to_degrees()
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
            cma: -1.1,
            cmq: -20.0,
            cme: -0.6,
            thrust_max: 1800.0,
            power_max: 119_000.0,
            oswald_e: 0.80,
            alpha_stall_pos: 16.0_f64.to_radians(),
            alpha_stall_neg: -12.0_f64.to_radians(),
            cd_max: 1.95,
            mach_crit: 0.65,
            cm_adot: -3.0,
            thrust_arm: 0.0,
            engine_count: 1,
            engine_lateral_arm: 0.0,
            prop_torque_coeff: 0.0,
            p_factor_coeff: 0.0,
            gyro_coeff: 0.0,
            throttle_split: 0.0,
            cy_beta: -0.31,
            cy_dr: -0.15,
            cl_beta: -0.09,
            cl_p: -0.45,
            cl_r: 0.10,
            cl_da: 0.07,
            cl_dr: 0.01,
            cn_beta: 0.12,
            cn_p: -0.02,
            cn_r: -0.16,
            cn_da: -0.004,
            cn_dr: 0.05,
            cl_flap: 1.10,
            cd_flap: 0.14,
            cm_flap: -0.20,
            flap_stall_shift: 6.0_f64.to_radians(),
            spiral_nose_drop_cm: 0.10,
            collision_panels: Vec::new(),
        }
    }

    #[test]
    fn positive_aileron_banks_right_and_turns_right() {
        // Positive aileron deflection must roll the right wing DOWN (positive
        // euler roll) and the banked lift must carry the heading rightward.
        let config = aircraft_default();
        let mut state = AircraftState::default();
        let (elev_trim, throttle_trim) = state.trim_level_flight(&config, 1000.0, 60.0);

        let yaw_start = state.euler_angles().2;
        let dt = 1.0 / 60.0;
        for _ in 0..120 {
            // 2 s of sustained right aileron (~11 deg deflection).
            step(&mut state, &config, elev_trim, 0.20, 0.0, throttle_trim, 0.0, None, dt, None);
        }

        let (roll, _, _) = state.euler_angles();
        let heading_change = state.euler_angles().2 - yaw_start;
        assert!(
            roll > 5f64.to_radians() && roll < 60f64.to_radians(),
            "right aileron should bank right-wing-down, got roll {:.1} deg",
            roll.to_degrees()
        );
        assert!(
            heading_change > 1.5f64.to_radians(),
            "banked drone should turn right, heading changed {:.1} deg",
            heading_change.to_degrees()
        );
    }

    #[test]
    fn positive_rudder_yaws_right() {
        // Positive rudder deflection must yaw the nose RIGHT (heading up).
        let config = aircraft_default();
        let mut state = AircraftState::default();
        let (elev_trim, throttle_trim) = state.trim_level_flight(&config, 1000.0, 60.0);

        let yaw_start = state.euler_angles().2;
        let dt = 1.0 / 60.0;
        for _ in 0..90 {
            // 1.5 s of sustained right rudder.
            step(&mut state, &config, elev_trim, 0.0, 0.30, throttle_trim, 0.0, None, dt, None);
        }

        let heading_change = state.euler_angles().2 - yaw_start;
        assert!(
            heading_change > 1.0f64.to_radians(),
            "positive rudder should yaw the nose right, heading changed {:.1} deg",
            heading_change.to_degrees()
        );
    }

    #[test]
    fn pull_elevator_pitches_up() {
        // Negative elevator deflection (= stick pull, trailing edge down)
        // must pitch the nose UP above the trimmed pitch attitude.
        let config = aircraft_default();
        let mut state = AircraftState::default();
        let (elev_trim, throttle_trim) = state.trim_level_flight(&config, 1000.0, 60.0);

        let dt = 1.0 / 60.0;
        for _ in 0..120 {
            // 2 s at 0.05 rad (~3 deg) stick pull beyond trim.
            let elevator = elev_trim - 0.05;
            step(&mut state, &config, elevator, 0.0, 0.0, throttle_trim, 0.0, None, dt, None);
        }

        let (_, pitch, _) = state.euler_angles();
        assert!(
            pitch > 2f64.to_radians(),
            "stick pull should pitch the nose up, got pitch {:.1} deg",
            pitch.to_degrees()
        );
    }

    #[test]
    fn flaps_add_lift_and_nose_down_moment() {
        // Deploying trailing-edge flaps must (a) increase the generated lift
        // and (b) add a nose-down pitching moment at otherwise identical
        // flight conditions. This verifies the flap effects actually reach
        // the aerodynamics (they were previously presentational only).
        let config = aircraft_default();
        let mut state = AircraftState::default();
        state.trim_level_flight(&config, 1000.0, 40.0);

        let flap = 30.0_f64.to_radians();
        let (f_clean, m_clean) =
            compute_forces_moments(&state, &config, 0.0, 0.0, 0.0, 0.5, 0.0, 0.0, &Vector3::zeros());
        let (f_flap, m_flap) =
            compute_forces_moments(&state, &config, 0.0, 0.0, 0.0, 0.5, 0.0, flap, &Vector3::zeros());

        // Flap lift acts upward = more negative body-Z force.
        assert!(
            f_flap.z < f_clean.z,
            "flaps should add lift (Fz clean {:.0} N, flap {:.0} N)",
            f_clean.z,
            f_flap.z
        );
        // Flap pitching moment is nose-down = more negative body-Y moment.
        assert!(
            m_flap.y < m_clean.y,
            "flaps should pitch the nose down (My clean {:.0}, flap {:.0})",
            m_clean.y,
            m_flap.y
        );
    }

    #[test]
    fn steep_bank_idle_descends_not_climbs() {
        // A hand-off aircraft banked 90 deg (wings vertical) has no vertical
        // lift, so gravity must pull the nose down into a dive. It must NOT
        // be able to pitch its own nose up and climb (a previous bug). This
        // guards the steep-bank spiral nose-drop behavior.
        use nalgebra::UnitQuaternion;

        let config = aircraft_default();
        let mut state = AircraftState::default();
        let (elev_trim, throttle_trim) = state.trim_level_flight(&config, 1000.0, 60.0);

        let (_, pitch0, yaw0) = state.euler_angles();
        let q = (UnitQuaternion::from_euler_angles(0.0, pitch0, yaw0)
            * UnitQuaternion::from_axis_angle(
                &nalgebra::Vector3::x_axis(),
                std::f64::consts::FRAC_PI_2,
            ))
        .normalize();
        state.q0 = q.w;
        state.q1 = q.i;
        state.q2 = q.j;
        state.q3 = q.k;
        state.p = 0.0;
        state.q = 0.0;
        state.r = 0.0;
        state.v = 0.0;

        let alt0 = state.altitude();
        let dt = 1.0 / 60.0;
        // 10 s of hand-off, idle, no inputs.
        for _ in 0..600 {
            step(&mut state, &config, elev_trim, 0.0, 0.0, throttle_trim, 0.0, None, dt, None);
        }

        let alt_change = state.altitude() - alt0;
        assert!(
            alt_change < -10.0,
            "90-deg-banked idle aircraft should dive, not climb (dAlt {:.1} m)",
            alt_change
        );
    }

    /// A twin-engine variant of the default aircraft.
    fn twin_default() -> AircraftConfig {
        let mut c = aircraft_default();
        c.engine_count = 2;
        c.engine_lateral_arm = 3.0;
        c.p_factor_coeff = -0.35;
        c.gyro_coeff = 0.05;
        c.throttle_split = 0.0;
        c
    }

    #[test]
    fn twin_engine_out_yaws_toward_dead_engine() {
        // Shutting down the right engine (split = +1) must produce a yawing
        // moment that pushes the nose toward the right (dead side), i.e. a
        // negative body-Z moment given the body frame (nose +X, right +Y).
        let mut config = twin_default();
        config.throttle_split = 1.0; // right engine out

        let mut state = AircraftState::default();
        let (elev_trim, throttle_trim) = state.trim_level_flight(&config, 1000.0, 60.0);

        let (_, moments) = compute_forces_moments(
            &state, &config, elev_trim, 0.0, 0.0, 0.8, 0.0, 0.0, &Vector3::zeros(),
        );
        // The asymmetric thrust yaws the nose toward the dead (right) engine.
        assert!(
            moments.z < 0.0,
            "right-engine-out should yaw the nose right (negative Mz), got {:.0}",
            moments.z
        );

        // Reversing the failed engine should reverse the yaw direction.
        let mut config_l = config.clone();
        config_l.throttle_split = -1.0; // left engine out
        let (_, m_l) = compute_forces_moments(
            &state, &config_l, elev_trim, 0.0, 0.0, 0.8, 0.0, 0.0, &Vector3::zeros(),
        );
        assert!(
            m_l.z > 0.0,
            "left-engine-out should yaw the nose left (positive Mz), got {:.0}",
            m_l.z
        );
    }

    #[test]
    fn twin_symmetric_thrust_matches_single_at_split_zero() {
        // With no asymmetric split, the twin must deliver the same total thrust
        // as the single-engine aircraft at the same throttle, so level-flight
        // trim and phugoid behaviour are unchanged.
        let single = aircraft_default();
        let twin = twin_default();
        let mut s_single = AircraftState::default();
        let mut s_twin = AircraftState::default();
        s_single.trim_level_flight(&single, 1000.0, 60.0);
        s_twin.trim_level_flight(&twin, 1000.0, 60.0);

        let zero = Vector3::zeros();
        let f_single = compute_forces(&s_single, &single, 0.0, 0.0, 0.0, 0.5, 0.0, &zero);
        let f_twin = compute_forces(&s_twin, &twin, 0.0, 0.0, 0.0, 0.5, 0.0, &zero);

        // Total axial thrust must be equal at the same throttle.
        assert!(
            (f_single.x - f_twin.x).abs() < 1.0,
            "twin symmetric thrust {:.1} should equal single {:.1}",
            f_twin.x,
            f_single.x
        );
    }

    #[test]
    fn twin_asymmetric_yaw_scales_with_engine_arm() {
        // The asymmetric-thrust yawing moment must scale linearly with the
        // engine lateral arm (it is `(T_left - T_right) * arm`).
        let mut state = AircraftState::default();
        let base = twin_default();

        let mut wide = base.clone();
        wide.engine_lateral_arm = 2.0;
        let mut narrow = base.clone();
        narrow.engine_lateral_arm = 1.0;

        let yaw_for = |c: &AircraftConfig, split: f64| {
            let mut cc = c.clone();
            cc.throttle_split = split;
            let (_, m) = compute_forces_moments(&state, &cc, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, &Vector3::zeros());
            m.z
        };
        let wide_yaw = yaw_for(&wide, 1.0);
        let narrow_yaw = yaw_for(&narrow, 1.0);
        assert!(
            (wide_yaw - 2.0 * narrow_yaw).abs() < 1e-6,
            "asymmetric yaw should scale with the engine arm ({wide_yaw} vs 2*{narrow_yaw})"
        );
    }

    #[test]
    fn twin_vmc_rudder_authority_grows_faster_than_asymmetric_yaw() {
        // Rudder + weathercock authority scale ~V^2 (via dynamic pressure),
        // while the static-limited asymmetric-thrust yaw is roughly constant
        // with speed. The ratio of rudder authority to asymmetric yaw must
        // therefore *increase* with speed — the Vmc mechanism: controllable
        // above a critical speed, uncontrollable below it.
        let base = twin_default();
        let speeds = [30.0f64, 60.0, 90.0];
        let mut prev_auth_ratio = f64::NEG_INFINITY;

        for speed in speeds {
            let mut c = base.clone();
            c.throttle_split = 0.0;
            let mut st = AircraftState::default();
            st.trim_level_flight(&c, 1000.0, speed);

            // Rudder yaw authority at full deflection (aero only: engine off).
            let mut c_aero = c.clone();
            c_aero.thrust_max = 1.0;
            c_aero.power_max = 1.0;
            let rud = compute_moments(&st, &c_aero, 0.0, 0.0, -0.35, 0.0, 0.0, 0.0, &Vector3::zeros()).z;

            // Asymmetric-thrust yaw at full throttle, one engine out.
            let mut c_out = c.clone();
            c_out.throttle_split = 1.0;
            let asym = compute_moments(&st, &c_out, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, &Vector3::zeros()).z;

            let ratio = (rud.abs() / asym.abs().max(1e-9));
            assert!(
                ratio > prev_auth_ratio,
                "rudder/asymmetric ratio must grow with speed (V={speed}: {ratio:.3} <= prev {prev_auth_ratio:.3})"
            );
            prev_auth_ratio = ratio;
        }
    }

    #[test]
    fn twin_long_range_high_ld_cruises_without_throttle_saturation() {
        // The TwinEngine is configured as a high-aspect-ratio long-range twin.
        // At high altitude it must trim at a strongly-improved L/D and sustain
        // cruise without the throttle saturating (range-optimal, not sprint).
        let cfg_path = format!("{}/../TwinEngine.toml", env!("CARGO_MANIFEST_DIR"));
        let c = AircraftConfig::from_file(&cfg_path)
            .expect("load TwinEngine.toml");
        assert_eq!(c.engine_count, 2, "TwinEngine must remain a twin");

        let mut st = AircraftState::default();
        let (_, thr) = st.trim_level_flight(&c, 8000.0, 48.0);
        assert!(
            thr < 1.0,
            "long-range cruise must not saturate throttle (got {thr:.2})"
        );

        // L/D from the trimmed conditions: CL = W/(qS), Cd from the polar.
        let rho = crate::Atmosphere::at_altitude(8000.0).density;
        let cl = 2.0 * c.mass * 9.80665 / (rho * 48.0 * 48.0 * c.wing_area);
        let ld = cl / (c.cd0 + c.k_drag * cl * cl);
        assert!(
            ld > 20.0,
            "long-range twin should beat L/D=20 at high-altitude cruise (got {ld:.1})"
        );
    }

    #[test]
    fn terrain_collision_stops_descent_on_a_mountain() {
        // A Gaussian 400 m mountain at the origin. The aircraft should NOT be
        // able to pierce it: it must rest clamped on the terrain surface at its
        // (x, y) position, i.e. pos_z = -h(x, y), even though its sea-level
        // altitude is still positive.
        let config = aircraft_default();
        let auth = Terrain::from_hills(vec![TerrainHill {
            centre: [0.0, 0.0],
            amplitude: 400.0,
            sigma: 500.0,
        }]);

        // Start right above the peak, diving straight down.
        let mut state = AircraftState::default();
        state.pos_x = 0.0;
        state.pos_y = 0.0;
        state.pos_z = -410.0; // altitude 410 m (10 m above the peak)
        state.w = 40.0; // 40 m/s down in body frame (level attitude)

        let dt = 1.0 / 120.0;
        for _ in 0..120 {
            step(&mut state, &config, 0.0, 0.0, 0.0, 0.0, 0.0, None, dt, Some(&auth));
        }

        // After 1 s of diving the aircraft must rest exactly on the peak's
        // surface, not below sea level.
        let ground_ned = -auth.height(state.pos_x, state.pos_y);
        assert!(
            (state.pos_z - ground_ned).abs() < 1.0,
            "aircraft did not rest on the terrain: pos_z {:.1} vs ground {:.1}",
            state.pos_z,
            ground_ned
        );
        assert!(
            state.pos_z < 0.0,
            "terrain should hold the aircraft above sea level (pos_z {:.1} < 0)",
            state.pos_z
        );
        // Vertical sink must be gone.
        let up_earth = Vector3::new(0.0, 0.0, -1.0);
        let up_body = state.rotation_earth_to_body().transform_vector(&up_earth);
        let bv = Vector3::new(state.u, state.v, state.w);
        assert!(
            bv.dot(&up_body) >= 0.0,
            "descent velocity into the ground was not killed"
        );
    }

    #[test]
    fn flat_terrain_keeps_historical_sea_level_floor() {
        let config = aircraft_default();
        let mut state = AircraftState::default();
        state.pos_z = 10.0; // 10 m *below* the datum (pos_z>0)
        state.w = 10.0;

        let dt = 1.0 / 120.0;
        let auth = Terrain::flat();
        for _ in 0..20 {
            step(&mut state, &config, 0.0, 0.0, 0.0, 0.0, 0.0, None, dt, Some(&auth));
        }
        assert!(
            state.pos_z <= 0.0,
            "flat terrain must clamp at sea level, got pos_z {:.1}",
            state.pos_z
        );
    }

    #[test]
    fn ground_effect_boosts_lift_near_the_surface() {
        let config = aircraft_default();
        let auth = Terrain::from_hills(vec![TerrainHill {
            centre: [0.0, 0.0],
            amplitude: 50.0,
            sigma: 1000.0,
        }]);
        let zero = Vector3::zeros();

        // Start from a trimmed level-flight state (net vertical force ~0) and
        // park it at the surface / 3 spans up. Terrain is the SAME peak, so the
        // two states differ only in their distance above it.
        let mut low = AircraftState::default();
        low.pos_x = 0.0;
        low.pos_y = 0.0;
        low.trim_level_flight(&config, 51.0, 60.0); // 1 m over the 50 m peak
        let ge_low = ground_effect_factor(
            auth.altitude_above_ground(0.0, 0.0, low.altitude()),
            config.wing_span,
        );

        let mut high = low.clone();
        high.pos_z -= 3.0 * config.wing_span; // 3*span above the surface
        let ge_high = ground_effect_factor(
            auth.altitude_above_ground(0.0, 0.0, high.altitude()),
            config.wing_span,
        );

        let f_low = compute_forces_with_terrain(&low, &config, 0.0, 0.0, 0.0, 0.5, 0.0, &zero, Some(&auth));
        let f_high = compute_forces_with_terrain(&high, &config, 0.0, 0.0, 0.0, 0.5, 0.0, &zero, Some(&auth));
        let f_ref = compute_forces(&high, &config, 0.0, 0.0, 0.0, 0.5, 0.0, &zero);

        // Body -Z is up: ground effect must raise the upward lift near the
        // surface, so f_low.z < f_high.z (both ~ = +/- gravity balance).
        assert!(ge_low > 0.5 && ge_high < 0.01, "ground-effect factors wrong: low {ge_low:.2}, high {ge_high:.3}");
        assert!(
            f_low.z < f_high.z,
            "ground effect should add lift: AGL~0 gave z {:.1}, 3·b gave z {:.1}",
            f_low.z,
            f_high.z
        );
        // 3 spans up the effect is negligible: the terrain force must match the
        // flat-ground reference to within a fraction of a percent.
        let rel = ((f_high.z - f_ref.z) / f_high.z.abs()).abs();
        assert!(
            rel < 0.005,
            "3·b above ground should have ~no ground effect (rel diff {rel:.3e})"
        );
    }

    #[test]
    fn orographic_wind_produces_updrafts_and_downdrafts() {
        // A north-facing windward slope: terrain rises to the north
        // (dh/dnorth > 0). A north wind is forced up it -> updraft
        // (negative NED vertical wind).
        let auth = Terrain::from_hills(vec![TerrainHill {
            centre: [1000.0, 0.0],
            amplitude: 400.0,
            sigma: 500.0,
        }]);

        // Just north of the datum (south of the peak), the slope climbs north.
        // The ground surface here is ~54 m ASL (Gaussian at 2 sigma), so fly at
        // 100 m to stay in the air while the decay term still bites.
        let north = 0.0;
        let east = 0.0;
        let wind = Vector3::new(10.0, 0.0, 0.0); // 10 m/s north
        let orog = auth.orographic_wind(&wind, north, east, 100.0, 300.0);

        assert!(
            orog.z < 0.0,
            "windward slope must push air up (NED z<0), got {:.3}",
            orog.z
        );

        // A wind blowing *south* down the same slope is a downdraft (NED z>0).
        let wind_south = Vector3::new(-10.0, 0.0, 0.0);
        let orog_south = auth.orographic_wind(&wind_south, north, east, 100.0, 300.0);
        assert!(
            orog_south.z > 0.0,
            "leeward/downwind side must push air down (NED z>0), got {:.3}",
            orog_south.z
        );

        // Effect decays away from the ground.
        let high = auth.orographic_wind(&wind, north, east, 1000.0, 300.0);
        assert!(
            high.z.abs() < orog.z.abs() * 0.1,
            "orographic wind must decay with altitude (high {:.3} vs near {:.3})",
            high.z,
            orog.z
        );
    }
}
