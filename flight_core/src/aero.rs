//! Aerodynamic force and moment computation with JSBSim-grade physics.
//!
//! Features:
//!   * Altitude-dependent density, dynamic pressure, and Mach number via
//!     the 1976 U.S. Standard Atmosphere model.
//!   * Viterna-Corrigan high-AoA nonlinear stall and post-stall model.
//!   * Prandtl-Glauert compressibility correction and Mach wave drag rise.
//!   * Downwash-lag (alpha-dot) pitch damping and stall center-of-pressure migration.
//!   * Full 6-DOF aerodynamic forces (Fx, Fy, Fz) and moments (L, M, N).

use nalgebra::{UnitQuaternion, Vector3};

use crate::atmosphere::Atmosphere;
use crate::config::AircraftConfig;
use crate::state::AircraftState;

/// Gravitational acceleration (m/s²).
const G: f64 = 9.80665;

/// Compute the net aerodynamic + propulsive + gravitational force and
/// moment acting on the aircraft in full 6-DOF with atmospheric lapse
/// and nonlinear stall aerodynamics.
///
/// Returns:
///   * `forces` — total force in the **body** frame (Newtons), `[Fx, Fy, Fz]`
///   * `moments` — total moment in the **body** frame (N·m), `[roll, pitch, yaw]` = `[L, M, N]`
/// Compute the net aerodynamic + propulsive + gravitational force acting on
/// the aircraft in full 6-DOF with atmospheric lapse and nonlinear stall
/// aerodynamics.
///
/// Returns the total force in the **body** frame (Newtons), `[Fx, Fy, Fz]`.
pub fn compute_forces(
    state: &AircraftState,
    config: &AircraftConfig,
    _elevator_deflection: f64,
    _aileron_deflection: f64,
    rudder_deflection: f64,
    throttle: f64,
    flap_deflection: f64,
    wind_earth: &Vector3<f64>,
) -> Vector3<f64> {
    // --- Air-relative velocity: the aero acts on the relative wind, not the
    //     ground-referenced velocity.
    let v_air = state.air_velocity(wind_earth);
    let v_tas = v_air.norm();
    let alpha = v_air.z.atan2(v_air.x);
    let beta = v_air.y.atan2(v_air.x.max(1e-6));

    // --- Atmospheric conditions at current aircraft altitude ---
    let altitude = state.altitude();
    let atm = Atmosphere::at_altitude(altitude);
    let q_dyn = atm.dynamic_pressure(v_tas);
    let mach = atm.mach_number(v_tas);

    // --- Compressibility correction (Prandtl-Glauert rule) ---
    let pg_factor = compressibility_factor(mach);
    let cla_effective = config.cla / pg_factor;

    // --- Flap (trailing-edge) increments: lift, induced-drag factor, drag ---
    let flap = flap_deflection.clamp(0.0, 0.7); // ~40 deg max
    let dcl_flap = config.cl_flap * flap;
    let dcd_flap = config.cd_flap * flap * flap.abs();
    // Flaps lower the positive stall angle.
    let alpha_stall_pos = config.alpha_stall_pos - config.flap_stall_shift * flap;
    let alpha_stall_neg = config.alpha_stall_neg;

    // --- Nonlinear Lift Coefficient CL(alpha) via Viterna blend ---
    let cl_linear = config.cl0 + cla_effective * alpha + dcl_flap;
    let cl = compute_viterna_lift(alpha, cl_linear, config, alpha_stall_pos, alpha_stall_neg);

    // --- Nonlinear Drag Coefficient CD(alpha, Mach) ---
    let k_induced = config.induced_drag_k();
    let cd_induced = k_induced * cl * cl;
    let cd_base =
        compute_viterna_drag(alpha, config.cd0 + cd_induced, config, alpha_stall_pos, alpha_stall_neg);

    // Mach wave drag divergence (drag rise above Mach_crit)
    let cd_mach = if mach > config.mach_crit {
        let dm = mach - config.mach_crit;
        20.0 * dm.powi(4)
    } else {
        0.0
    };
    let cd = cd_base + cd_mach + dcd_flap;

    // Lift and drag magnitudes (Newtons)
    let lift = q_dyn * config.wing_area * cl;
    let drag = q_dyn * config.wing_area * cd;

    // Convert lift & drag from wind frame into body axes. Lift acts in the
    // body X-Z plane; drag is directed along the (possibly sideslipped) body
    // velocity, so its component along the body X axis is reduced by cos(beta).
    // This is the standard sideslip-aware wind-to-body transformation, and
    // correctly reduces to the beta==0 special case at wings-level.
    let cb = beta.cos();
    let force_x = -drag * alpha.cos() * cb + lift * alpha.sin();
    let force_z = -lift * alpha.cos() - drag * alpha.sin() * cb;

    // --- Lateral Sideforce CY ---
    let cy = config.cy_beta * beta + config.cy_dr * rudder_deflection;
    let force_y = q_dyn * config.wing_area * cy;

    let mut forces = Vector3::new(force_x, force_y, force_z);

    // --- Engine Thrust along body +X ---
    // Engine thrust is limited by both the static (low-speed) thrust ceiling
    // and the constant shaft power delivered by the propeller:
    //   T(V) = min(thrust_max * δ, P_max * δ / V)
    // so somewhere above the corner speed the available thrust falls off with
    // airspeed. Both scale with the atmospheric density ratio (forced / turboshaft
    // losses), clamped to a sensible range.
    let throttle = throttle.clamp(0.0, 1.0);
    let density_factor = (atm.density_ratio).clamp(0.1, 1.2);
    let static_thrust = config.thrust_max * throttle * density_factor;
    let power_thrust = config.power_max * throttle * density_factor / v_tas.max(6.0);
    let thrust = static_thrust.min(power_thrust);
    forces.x += thrust;

    // --- Gravity rotated from Earth NED [0, 0, m*g] to body axes ---
    let gravity_earth = Vector3::new(0.0, 0.0, config.mass * G);
    let rot: UnitQuaternion<f64> = state.rotation_earth_to_body();
    let gravity_body = rot.transform_vector(&gravity_earth);
    forces += gravity_body;

    forces
}

/// Compute the net aerodynamic + propulsive moment acting on the aircraft.
///
/// `alpha_dot` (rad/s) feeds the downwash-lag pitch damping term `cm_adot`.
/// Returns the total moment in the **body** frame (N·m), `[L, M, N]`.
pub fn compute_moments(
    state: &AircraftState,
    config: &AircraftConfig,
    elevator_deflection: f64,
    aileron_deflection: f64,
    rudder_deflection: f64,
    throttle: f64,
    alpha_dot: f64,
    flap_deflection: f64,
    wind_earth: &Vector3<f64>,
) -> Vector3<f64> {
    // --- Air-relative velocity: the aero acts on the relative wind. ---
    let v_air = state.air_velocity(wind_earth);
    let v_tas = v_air.norm();
    let alpha = v_air.z.atan2(v_air.x);
    let beta = v_air.y.atan2(v_air.x.max(1e-6));

    // --- Atmospheric conditions at current aircraft altitude ---
    let altitude = state.altitude();
    let atm = Atmosphere::at_altitude(altitude);
    let q_dyn = atm.dynamic_pressure(v_tas);
    let mach = atm.mach_number(v_tas);

    // --- Compressibility correction (Prandtl-Glauert rule) ---
    let _pg_factor = compressibility_factor(mach);

    // --- Engine thrust (used for the thrust-line pitching moment) ---
    let throttle = throttle.clamp(0.0, 1.0);
    let density_factor = (atm.density_ratio).clamp(0.1, 1.2);
    let static_thrust = config.thrust_max * throttle * density_factor;
    let power_thrust = config.power_max * throttle * density_factor / v_tas.max(6.0);
    let thrust = static_thrust.min(power_thrust);

    // --- Dimensionless body angular rates ---
    let p_hat = if v_tas > 1e-6 {
        state.p * config.wing_span / (2.0 * v_tas)
    } else {
        0.0
    };
    let q_hat = if v_tas > 1e-6 {
        state.q * config.chord / (2.0 * v_tas)
    } else {
        0.0
    };
    let r_hat = if v_tas > 1e-6 {
        state.r * config.wing_span / (2.0 * v_tas)
    } else {
        0.0
    };
    let alpha_dot_hat = if v_tas > 1e-6 {
        alpha_dot * config.chord / (2.0 * v_tas)
    } else {
        0.0
    };

    // --- Pitching Moment Cm with downwash-lag damping & stall break ---
    // At post-stall, center-of-pressure shifts aft, adding a stabilizing nose-down pitch break.
    // Flaps lower the positive stall angle (incremental lift to the rear drops the break AoA).
    let flap = flap_deflection.clamp(0.0, 0.7);
    let alpha_stall_pos = config.alpha_stall_pos - config.flap_stall_shift * flap;
    let stall_pitch_break = if alpha > alpha_stall_pos {
        -0.45 * (alpha - alpha_stall_pos).min(0.4)
    } else if alpha < config.alpha_stall_neg {
        0.45 * (config.alpha_stall_neg - alpha).min(0.4)
    } else {
        0.0
    };

    // --- Steep-bank spiral nose-drop ---
    // A fixed-wing banked past ~45 deg has little/no vertical lift, so gravity
    // must pull the nose down into a dive. In a real aircraft the resulting
    // sideslip drives spiral divergence (nose drops); here we inject that
    // tendency directly with a bank-coupled nose-down moment, engaged only at
    // steep bank so normal turns are unaffected.
    let (bank, _, _) = state.euler_angles();
    let sin_bank = bank.sin().abs();
    // Engage between ~35 deg (sin=0.574) and ~90 deg (sin=1.0).
    let engage = ((sin_bank - 0.574) / (1.0 - 0.574)).clamp(0.0, 1.0);
    let spiral_nose_drop = -config.spiral_nose_drop_cm * engage * engage;

    // --- Inversion sign (proper 6-DOF pitch sense) ---
    // The body down-axis projects onto Earth +Z (NED) as +1 when upright and
    // -1 when inverted. A real elevator acts nose-toward-the-belly, so both
    // the elevator authority and the pitch static/dynamic stability reverse
    // sense when the aircraft is upside down: pulling "up" on a stick when
    // inverted pushes the nose toward the belly, i.e. down relative to the
    // world. Applying this sign to the pitch aerodynamic terms makes inverted
    // pull dive (and inverted attitude be trimmed/stabilised correctly)
    // instead of climbing. A tanh blend (rather than a hard sign) fades the
    // pitch authority smoothly to zero at knife-edge (~90 deg bank) and back,
    // so banking through 90 deg doesn't jerk the nose around. The bank-keyed
    // spiral nose-drop is left out: it is already a world-space nose-down term
    // rather than a body-flow term.
    let (_, _, body_down) = state.body_axes_in_earth();
    let pitch_sense = (8.0 * body_down.z).tanh();
    // The spiral nose-drop is a *world-space* gravity nose-down tendency. Its
    // body-frame pitch component reverses when inverted, but it must stay full
    // strength exactly at knife-edge (where pitch_sense ~ 0). Use a hard sign
    // here (not the smooth pitch_sense blend) so inverted bank keeps diving
    // instead of pushing the nose the wrong way and tumbling.
    let spiral_sense = if body_down.z >= 0.0 { 1.0 } else { -1.0 };

    let cm_core = config.cm0
        + config.cma * alpha
        + config.cmq * q_hat
        + config.cm_adot * alpha_dot_hat
        + config.cme * elevator_deflection
        + config.cm_flap * flap
        + stall_pitch_break;
    let cm = cm_core * pitch_sense + spiral_nose_drop * spiral_sense;
    // Thrust line offset from CG produces a pitching moment proportional to
    // thrust; its body-Z arm also reverses when inverted.
    let pitch_moment =
        q_dyn * config.wing_area * config.chord * cm + thrust * config.thrust_arm * pitch_sense;

    // --- Rolling Moment Cl (around body X) ---
    let cl_roll = config.cl_beta * beta
        + config.cl_p * p_hat
        + config.cl_r * r_hat
        + config.cl_da * aileron_deflection
        + config.cl_dr * rudder_deflection;
    let roll_moment = q_dyn * config.wing_area * config.wing_span * cl_roll;

    // --- Yawing Moment Cn (around body Z) ---
    let cn = config.cn_beta * beta
        + config.cn_p * p_hat
        + config.cn_r * r_hat
        + config.cn_da * aileron_deflection
        + config.cn_dr * rudder_deflection;
    let yaw_moment = q_dyn * config.wing_area * config.wing_span * cn;

    let moments = Vector3::new(roll_moment, pitch_moment, yaw_moment);

    moments
}

/// Compute the total body-frame force and moment.
///
/// `alpha_dot` (rad/s) feeds the downwash-lag pitch damping term.
pub fn compute_forces_moments(
    state: &AircraftState,
    config: &AircraftConfig,
    elevator_deflection: f64,
    aileron_deflection: f64,
    rudder_deflection: f64,
    throttle: f64,
    alpha_dot: f64,
    flap_deflection: f64,
    wind_earth: &Vector3<f64>,
) -> (Vector3<f64>, Vector3<f64>) {
    let forces = compute_forces(
        state,
        config,
        elevator_deflection,
        aileron_deflection,
        rudder_deflection,
        throttle,
        flap_deflection,
        wind_earth,
    );
    let moments = compute_moments(
        state,
        config,
        elevator_deflection,
        aileron_deflection,
        rudder_deflection,
        throttle,
        alpha_dot,
        flap_deflection,
        wind_earth,
    );
    (forces, moments)
}

/// Prandtl-Glauert compressibility correction factor.
fn compressibility_factor(mach: f64) -> f64 {
    if mach < 0.85 {
        (1.0 - mach * mach).max(0.04).sqrt()
    } else {
        0.20 // Subsonic limit clamp
    }
}

/// Viterna-Corrigan post-stall lift formulation with smooth hyperbolic blending.
fn compute_viterna_lift(
    alpha: f64,
    cl_linear: f64,
    config: &AircraftConfig,
    alpha_pos: f64,
    alpha_neg: f64,
) -> f64 {
    if alpha >= alpha_neg && alpha <= alpha_pos {
        // Pre-stall linear/attached flow
        cl_linear
    } else if alpha > alpha_pos {
        // Positive post-stall: smooth sigmoid transition to flat-plate separated flow
        let d_alpha = alpha - alpha_pos;
        let blend = (1.0 + (5.0 * d_alpha).tanh()) * 0.5;
        let cl_stall_peak = config.cl0 + config.cla * alpha_pos;
        let cl_separated = (config.cd_max * 0.5) * (2.0 * alpha).sin()
            + 0.1 * (alpha.cos()).powi(2) / alpha.sin().max(0.01);
        (1.0 - blend) * cl_stall_peak + blend * cl_separated
    } else {
        // Negative post-stall (inverted stall)
        let d_alpha = alpha_neg - alpha;
        let blend = (1.0 + (5.0 * d_alpha).tanh()) * 0.5;
        let cl_stall_neg_peak = config.cl0 + config.cla * alpha_neg;
        let cl_separated = (config.cd_max * 0.5) * (2.0 * alpha).sin();
        (1.0 - blend) * cl_stall_neg_peak + blend * cl_separated
    }
}

/// Viterna-Corrigan post-stall drag formulation with smooth transition.
fn compute_viterna_drag(
    alpha: f64,
    cd_attached: f64,
    config: &AircraftConfig,
    alpha_pos: f64,
    alpha_neg: f64,
) -> f64 {
    if alpha >= alpha_neg && alpha <= alpha_pos {
        cd_attached
    } else {
        let d_alpha = if alpha > alpha_pos {
            alpha - alpha_pos
        } else {
            alpha_neg - alpha
        };
        let blend = (1.0 + (6.0 * d_alpha).tanh()) * 0.5;
        let cd_flat_plate = config.cd_max * (alpha.sin()).powi(2) + config.cd0 * alpha.cos().abs();
        (1.0 - blend) * cd_attached + blend * cd_flat_plate
    }
}
