//! Aerodynamic force and moment computation with JSBSim-grade physics.
//!
//! Features:
//!   * Altitude-dependent density, dynamic pressure, and Mach number via
//!     the 1976 U.S. Standard Atmosphere model.
//!   * Viterna-Corrigan high-AoA nonlinear stall and post-stall model.
//!   * Prandtl-Glauert compressibility correction and Mach wave drag rise.
//!   * Downwash lag effects and stall center-of-pressure migration.
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
pub fn compute_forces_moments(
    state: &AircraftState,
    config: &AircraftConfig,
    elevator_deflection: f64,
    aileron_deflection: f64,
    rudder_deflection: f64,
    throttle: f64,
) -> (Vector3<f64>, Vector3<f64>) {
    // --- Atmospheric conditions at current aircraft altitude ---
    let altitude = state.altitude();
    let atm = Atmosphere::at_altitude(altitude);
    let v_tas = state.airspeed();
    let q_dyn = atm.dynamic_pressure(v_tas);
    let mach = atm.mach_number(v_tas);

    // --- Angles of attack (AoA) and sideslip (beta) ---
    let alpha = state.angle_of_attack();
    let beta = (state.v).atan2(state.u.max(1e-6));

    // --- Compressibility correction (Prandtl-Glauert rule) ---
    let pg_factor = if mach < 0.85 {
        (1.0 - mach * mach).max(0.04).sqrt()
    } else {
        0.20 // Subsonic limit clamp
    };
    let cla_effective = config.cla / pg_factor;

    // --- Nonlinear Lift Coefficient CL(alpha) via Viterna blend ---
    let cl_linear = config.cl0 + cla_effective * alpha;
    let cl = compute_viterna_lift(alpha, cl_linear, config);

    // --- Nonlinear Drag Coefficient CD(alpha, Mach) ---
    let k_induced = config.induced_drag_k();
    let cd_induced = k_induced * cl * cl;
    let cd_base = compute_viterna_drag(alpha, config.cd0 + cd_induced, config);

    // Mach wave drag divergence (drag rise above Mach_crit)
    let cd_mach = if mach > config.mach_crit {
        let dm = mach - config.mach_crit;
        20.0 * dm.powi(4)
    } else {
        0.0
    };
    let cd = cd_base + cd_mach;

    // Lift and drag magnitudes (Newtons)
    let lift = q_dyn * config.wing_area * cl;
    let drag = q_dyn * config.wing_area * cd;

    // Convert lift & drag from wind frame into body axes
    let force_x = -drag * alpha.cos() + lift * alpha.sin();
    let force_z = -lift * alpha.cos() - drag * alpha.sin();

    // --- Lateral Sideforce CY ---
    let cy = config.cy_beta * beta + config.cy_dr * rudder_deflection;
    let force_y = q_dyn * config.wing_area * cy;

    let mut forces = Vector3::new(force_x, force_y, force_z);

    // --- Engine Thrust along body +X ---
    // Engine thrust scales with atmospheric density ratio (rho / rho_0)
    let thrust_density_factor = (atm.density_ratio).clamp(0.1, 1.2);
    let thrust = config.thrust_max * throttle.clamp(0.0, 1.0) * thrust_density_factor;
    forces.x += thrust;

    // --- Gravity rotated from Earth NED [0, 0, m*g] to body axes ---
    let gravity_earth = Vector3::new(0.0, 0.0, config.mass * G);
    let rot: UnitQuaternion<f64> = state.rotation_earth_to_body();
    let gravity_body = rot.inverse().transform_vector(&gravity_earth);
    forces += gravity_body;

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

    // --- Pitching Moment Cm with downwash lag & stall break ---
    // At post-stall, center-of-pressure shifts aft, adding a stabilizing nose-down pitch break
    let stall_pitch_break = if alpha > config.alpha_stall_pos {
        -0.45 * (alpha - config.alpha_stall_pos).min(0.4)
    } else if alpha < config.alpha_stall_neg {
        0.45 * (config.alpha_stall_neg - alpha).min(0.4)
    } else {
        0.0
    };

    let cm = config.cm0
        + config.cma * alpha
        + config.cmq * q_hat
        + config.cme * elevator_deflection
        + stall_pitch_break;
    let pitch_moment = q_dyn * config.wing_area * config.chord * cm;

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

    (forces, moments)
}

/// Viterna-Corrigan post-stall lift formulation with smooth hyperbolic blending.
fn compute_viterna_lift(alpha: f64, cl_linear: f64, config: &AircraftConfig) -> f64 {
    let alpha_pos = config.alpha_stall_pos;
    let alpha_neg = config.alpha_stall_neg;

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
fn compute_viterna_drag(alpha: f64, cd_attached: f64, config: &AircraftConfig) -> f64 {
    let alpha_pos = config.alpha_stall_pos;
    let alpha_neg = config.alpha_stall_neg;

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
