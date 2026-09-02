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
use crate::terrain::Terrain;

/// Gravitational acceleration (m/s²).
const G: f64 = 9.80665;

/// Thrust produced by a given throttle fraction against a per-engine thrust and
/// power ceiling, using the constant-power propeller model `T = min(T_max·δ,
/// P_max·δ / V)`. `thr` is the per-engine power fraction in `[0,1]`.
fn engine_thrust_ceiling(
    thr_max: f64,
    pwr_max: f64,
    thr: f64,
    v_tas: f64,
    density_factor: f64,
) -> f64 {
    let static_thrust = thr_max * thr * density_factor;
    let power_thrust = pwr_max * thr * density_factor / v_tas.max(6.0);
    static_thrust.min(power_thrust)
}

/// The per-engine thrust vector for the configured propulsion layout.
///
/// `throttle_split` (`-1..=1`, from the config) skews the master throttle
/// between a left/right pair so an asymmetric thrust / engine-out can be
/// applied. Returns `(left, right, single)` in Newtons, where `single` is the
/// centreline thrust for a single-engine layout and is zero for a twin.
///
/// `thrust_max` / `power_max` describe the *total* installed propulsion, so for
/// a twin each engine ceilings at half the total. The skew preserves total
/// thrust at any split (one engine gains exactly what the other loses), and at
/// `split == ±1` the dead engine runs at zero while the live one carries its
/// full single-engine share — matching a real engine-out.
fn engine_thrusts(
    config: &AircraftConfig,
    throttle: f64,
    throttle_split: f64,
    v_tas: f64,
    density_factor: f64,
) -> (f64, f64, f64) {
    let split = throttle_split.clamp(-1.0, 1.0);

    if config.engine_count != 2 {
        // Single engine: thrust line on the centreline, no asymmetry.
        let t = engine_thrust_ceiling(
            config.thrust_max,
            config.power_max,
            throttle,
            v_tas,
            density_factor,
        );
        return (0.0, 0.0, t);
    }

    // Twin: per-engine ceiling is half the total installed thrust/power. The
    // master throttle is preserved at split==0; the skew shifts power from one
    // engine to the other up to a full single-engine share.
    let half_t = config.thrust_max * 0.5;
    let half_p = config.power_max * 0.5;
    let left_thr = (throttle * (1.0 - split)).clamp(0.0, 1.0);
    let right_thr = (throttle * (1.0 + split)).clamp(0.0, 1.0);
    let t_left = engine_thrust_ceiling(half_t, half_p, left_thr, v_tas, density_factor);
    let t_right = engine_thrust_ceiling(half_t, half_p, right_thr, v_tas, density_factor);
    (t_left, t_right, 0.0)
}

/// Compute the engine (propeller) force and moment contributions for the
/// configured propulsion layout, including the twin-specific couplings.
///
/// Returns `(force, moment)` in the body frame:
/// * `force.x`  — total axial thrust (sum of all engines).
/// * `moment.y` — thrust-line pitching moment (thrust × vertical arm).
/// * `moment.z` — asymmetric-thrust yawing moment `(T_left − T_right) · arm`
///   (the engine-out / Vmc driver) plus P-factor.
/// * `moment.x` — propeller-torque rolling moment.
/// * `moment`   — gyroscopic precession pitch/yaw couples.
///
/// `q`, `r` are the body angular rates (rad/s) and `alpha` is the
/// aerodynamic angle of attack (rad), both needed by the asymmetric couplings.
#[allow(clippy::too_many_arguments)]
fn engine_forces_moments(
    config: &AircraftConfig,
    throttle: f64,
    v_tas: f64,
    density_factor: f64,
    alpha: f64,
    q: f64,
    r: f64,
    pitch_sense: f64,
) -> (Vector3<f64>, Vector3<f64>) {
    let split = config.throttle_split;
    let (t_left, t_right, t_single) =
        engine_thrusts(config, throttle, split, v_tas, density_factor);

    let mut force = Vector3::zeros();
    let mut moment = Vector3::zeros();

    // Total axial thrust.
    let thrust_total = t_left + t_right + t_single;
    force.x += thrust_total;

    // Trust-line pitching moment (reverses when inverted).
    moment.y += thrust_total * config.thrust_arm * pitch_sense;

    if config.engine_count == 2 {
        let arm = config.engine_lateral_arm;

        // Asymmetric-thrust yawing moment: unequal left/right thrust yaws the
        // nose toward the dead engine. The rudder must counter this; below Vmc
        // it cannot, which drives the characteristic engine-out departure.
        moment.z += (t_left - t_right) * arm;

        // Propeller torque: each engine produces a rolling couple about body X
        // proportional to its thrust. Assumed counter-rotating for a symmetric
        // twin, so the net is small and scales with any residual asymmetry.
        let torque = config.prop_torque_coeff * (t_left - t_right) * arm;
        moment.x += torque;

        // P-factor: at high power and high AoA the descending blade thrusts
        // more than the ascending one, yawing the nose. Both props rotate the
        // same way (standard right-hand from behind), so they reinforce.
        let pf = config.p_factor_coeff * thrust_total * alpha;
        moment.z += pf;

        // Gyroscopic precession: the spinning propeller disc resists being
        // pitched or yawed, coupling the two axes. Proportional to the engine
        // angular momentum (modelled by the thrust proxy) and the body rates.
        let gyro = config.gyro_coeff * thrust_total;
        moment.y += gyro * r;
        moment.z += -gyro * q;
    }

    (force, moment)
}

// --- Ground effect (WIG / altitude-in-ground-effect) -------------------------

/// Ground-effect factor `σ ∈ [0, 1)` from the classic height-to-span ratio:
///
/// ```text
/// σ = 1 / (1 + (16·h/b)²)
/// ```
///
/// `h` is the altitude above the ground surface and `b` the wing span, both in
/// metres. `σ → 1` at the surface (full ground effect) and `σ → 0` beyond a
/// span or two, so the influence fades rapidly with height.
///
/// Returns the factor; pass it into [`ground_effect_factors`] to scale the
/// lift and induced drag coefficients.
pub fn ground_effect_factor(altitude_above_ground: f64, wing_span: f64) -> f64 {
    let x = 16.0 * (altitude_above_ground / wing_span.max(1e-6));
    1.0 / (1.0 + x * x)
}

/// Multiply the **lift coefficient** and the **induced-drag coefficient** by
/// these factors when the aircraft is in ground effect. Mapped from the raw
/// [`ground_effect_factor`]:
///
/// * `cl_mult`   — ground effect increases the lift-curve slope / effective
///   lift (~+30% max at the surface).
/// * `cd_induced_mult` — wingtip vortices are suppressed by the ground, so
///   induced drag falls (up to ~−40% at the surface).
pub fn ground_effect_factors(altitude_above_ground: f64, wing_span: f64) -> (f64, f64) {
    let ge = ground_effect_factor(altitude_above_ground, wing_span);
    let cl_mult = 1.0 + 0.30 * ge;
    let cd_induced_mult = 1.0 - 0.40 * ge;
    (cl_mult, cd_induced_mult)
}

/// Body-frame force (Newtons) produced by the aerodynamic lift/drag at the
/// current angle of attack if ground-effect multiplier `cl_mult` and
/// `cd_induced_mult` were applied. The extra lift `ΔL` acts perpendicular to
/// the relative wind in the body X–Z plane, exactly like the base lift.
///
/// This is a convenience for callers that already have the state/wind handy and
/// want the ground-effect delta without duplicating the coefficient math.
pub fn ground_effect_force_delta(
    q_dyn: f64,
    wing_area: f64,
    cl: f64,
    alpha: f64,
    ge_factors: (f64, f64),
) -> Vector3<f64> {
    let (cl_mult, _) = ge_factors;
    // The base lift acts along `L` in the body X–Z plane.
    let dlift = q_dyn * wing_area * cl * (cl_mult - 1.0);
    // Lift direction in body frame: with the wind in the X–Z plane at alpha,
    // lift points largely along -Z (and some -X when alpha is large).
    Vector3::new(dlift * alpha.sin(), 0.0, -dlift * alpha.cos())
}

/// Compute the net aerodynamic + propulsive + gravitational force acting on
/// the aircraft in full 6-DOF with atmospheric lapse and nonlinear stall
/// aerodynamics, optionally including terrain ground effect.
///
/// Returns the total force in the **body** frame (Newtons), `[Fx, Fy, Fz]`.
///
/// When `terrain` is `Some`, the aircraft's altitude above the terrain surface
/// is used to raise the lift-curve slope and lower the induced drag (ground
/// effect). With `None` the ground is assumed an infinite flat plane at the
/// datum, disabling the effect — the historical behaviour.
pub fn compute_forces(
    state: &AircraftState,
    config: &AircraftConfig,
    elevator_deflection: f64,
    aileron_deflection: f64,
    rudder_deflection: f64,
    throttle: f64,
    flap_deflection: f64,
    wind_earth: &Vector3<f64>,
) -> Vector3<f64> {
    compute_forces_impl(
        state,
        config,
        elevator_deflection,
        aileron_deflection,
        rudder_deflection,
        throttle,
        flap_deflection,
        wind_earth,
        None,
    )
}

/// `compute_forces` with the optional terrain ground effect.
#[allow(clippy::too_many_arguments)]
pub fn compute_forces_with_terrain(
    state: &AircraftState,
    config: &AircraftConfig,
    elevator_deflection: f64,
    aileron_deflection: f64,
    rudder_deflection: f64,
    throttle: f64,
    flap_deflection: f64,
    wind_earth: &Vector3<f64>,
    terrain: Option<&Terrain>,
) -> Vector3<f64> {
    compute_forces_impl(
        state,
        config,
        elevator_deflection,
        aileron_deflection,
        rudder_deflection,
        throttle,
        flap_deflection,
        wind_earth,
        terrain,
    )
}

#[allow(clippy::too_many_arguments)]
fn compute_forces_impl(
    state: &AircraftState,
    config: &AircraftConfig,
    _elevator_deflection: f64,
    _aileron_deflection: f64,
    rudder_deflection: f64,
    throttle: f64,
    flap_deflection: f64,
    wind_earth: &Vector3<f64>,
    terrain: Option<&Terrain>,
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
    let cl_clean = compute_viterna_lift(alpha, cl_linear, config, alpha_stall_pos, alpha_stall_neg);

    // --- Ground effect (when flying close above the terrain) ---
    // Raises the lift-curve slope (more effective CL) and suppresses induced
    // drag as the wing approaches the surface. Computed from the altitude
    // above the terrain surface at the aircraft's NED position.
    let (cl_mult, cd_ind_mult) = match terrain {
        Some(t) => {
            let agl = t.altitude_above_ground(state.pos_x, state.pos_y, altitude);
            let f = ground_effect_factors(agl, config.wing_span);
            eprintln!(
                "[GE] agl={agl:.2} cl_mult={:.3} cd_ind_mult={:.3}",
                f.0, f.1
            );
            f
        }
        None => (1.0, 1.0),
    };
    let cl = cl_clean * cl_mult;

    // --- Nonlinear Drag Coefficient CD(alpha, Mach) ---
    // Induced drag is built from the *unge* CL and then scaled by the ground
    // effect's induced-drag suppression, so the two corrections stay decoupled.
    let k_induced = config.induced_drag_k();
    let cd_induced = k_induced * cl_clean * cl_clean * cd_ind_mult;
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
    // losses), clamped to a sensible range. Multi-engine layouts (twin) split
    // and skew the total across left/right engines.
    let throttle = throttle.clamp(0.0, 1.0);
    let density_factor = (atm.density_ratio).clamp(0.1, 1.2);
    let (engine_force, _engine_moment) = engine_forces_moments(
        config,
        throttle,
        v_tas,
        density_factor,
        alpha,
        state.q,
        state.r,
        1.0, // force only; the thrust-line pitch moment is applied in compute_moments
    );
    forces += engine_force;

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
    // thrust; its body-Z arm also reverses when inverted. This (plus the twin
    // asymmetric couplings) is supplied by `engine_forces_moments` below.
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

    let mut moments = Vector3::new(roll_moment, pitch_moment, yaw_moment);

    // --- Engine (twin) thrust-line and asymmetric couplings ---
    // Adds the thrust-Line pitching moment plus, for a twin, the asymmetric
    // thrust yaw (Vmc), P-factor, prop torque and gyroscopic precession.
    let (_, engine_moment) = engine_forces_moments(
        config,
        throttle,
        v_tas,
        density_factor,
        alpha,
        state.q,
        state.r,
        pitch_sense,
    );
    moments += engine_moment;

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
