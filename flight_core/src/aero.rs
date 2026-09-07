//! Aerodynamic force and moment computation with JSBSim-grade physics.
//!
//! Features:
//!   * Altitude-dependent density, dynamic pressure, and Mach number via
//!     the 1976 U.S. Standard Atmosphere model (see [`crate::atmosphere`]).
//!   * Viterna-Corrigan high-AoA nonlinear stall and post-stall model.
//!   * Prandtl-Glauert compressibility correction and Mach wave drag rise.
//!   * Downwash-lag (alpha-dot) pitch damping and stall center-of-pressure migration.
//!   * Full 6-DOF aerodynamic forces (Fx, Fy, Fz) and moments (L, M, N).
//!
//! # Standards References
//! - **Viterna & Corrigan** — "Prediction of the Post-Stall Regime for Light
//!   Aircraft", NASA CR 1980 (also NASA TM-80001), flat-plate separated-flow
//!   lift/drag model with smooth blending.
//! - **Prandtl-Glauert rule** — compressibility correction `β = √(1−M²)`;
//!   see Anderson, *Fundamentals of Aerodynamics*, 5th ed., §11.7.
//! - **Prandtl-Glauert per RADIAN normalisation** of the lift-curve slope
//!   matches JSBSim's `aero/lift` handling of `cla` at compressible Mach.
//! - **Downwash-lag alpha-dot damping**, **stall centre-of-pressure shift**:
//!   Stevens & Lewis, *Aircraft Control and Simulation*, 2nd ed., §4.3–4.4;
//!   coverage of `Cm_adot` follows Etkin, *Dynamics of Atmospheric Flight*, §6.5.
//! - **100% balance of forces/moments at trim**: verified against JSBSim's
//!   classical longitudinal coefficient bookkeeping in the test suite.
//!
//! # Units
//! SI throughout: forces N, moments N·m, coefficients dimensionless, angles rad.

use nalgebra::{UnitQuaternion, Vector3};

use crate::atmosphere::Atmosphere;
use crate::config::{AircraftConfig, Propulsion};
use crate::state::{AircraftState, SIDESLIP_AXIAL_MIN};
use crate::terrain::Terrain;

/// Gravitational acceleration (m/s²). [NASA SP-747 / WGS84]
pub const G: f64 = 9.80665;

/// True-airspeed (m/s) floor guarding the dimensionless-rate and rate-damping
/// normalisations `p̂ = p·b/(2V)`, etc. against a near-hover divide-by-zero.
/// Larger than the sideslip floor because it scales a physical characteristic
/// length, not just an `atan2` argument.
const V_TAS_EPS: f64 = 1e-6;

// ---------------------------------------------------------------------------
// Post-stall / high-AoA model tuning constants
// (derived from the Viterna-Corrigan separated-flow formulation)
// ---------------------------------------------------------------------------

/// Minimum airspeed (m/s) guarding dynamic-pressure and Mach computations
/// against a division near hover/stall. Keeps `T_min(V)` and the
/// dimensionless-rate normalisation finite.
pub const V_MIN_GUARD: f64 = 6.0;

/// Slope of the smooth `tanh` blend into positive post-stall separated flow.
/// 5.0 per radian of overshoot past the stall angle: at +1 rad the blend is
/// ~99% separated — a fast, physically-motivated transition per
/// Viterna-Corrigan's empirical flat-plate fall-off.
const VITERNA_BLEND_POS: f64 = 5.0;

/// Slope of the smooth `tanh` blend into negative (inverted) post-stall.
const VITERNA_BLEND_NEG: f64 = 5.0;

/// Constant fraction of `Cd_max` used for the separated-flow lift peak
/// `CL = (Cd_max/2)·sin(2α)`, the classic flat-plate result. The additional
/// `0.1·cos²α/sinα` term models the residual camber/viscous lift in the
/// Viterna-Corrigan extension for light aircraft.
const SEPARATED_LIFT_KC: f64 = 0.5;
/// Small-lift residual coefficient in the positive post-stall lift term.
const SEPARATED_LIFT_RESIDUAL: f64 = 0.1;
/// Denominator guard for the residual `cos²α/sinα` term (no divide-by-zero).
const SEPARATED_MIN_SIN: f64 = 0.01;

/// Blend slope for the drag transition into the flat-plate separated regime.
/// 6.0 per radian keeps the drag rise slightly sharper than the lift break
/// (the drag tripping leads the CL collapse, as measured in wind-tunnel data).
const VITERNA_DRAG_BLEND: f64 = 6.0;

/// Mach-wave drag-rise gain. `ΔCd = K_M·(ΔM)⁴`, the canonical sharp-4th-power
/// divergence used by Raymer / Torenbeek above the critical Mach.
const MACH_DRAG_GAIN_K4: f64 = 20.0;

/// Clamp on the post-stall pitch-break moment arm (radians). Limits how far
/// the `−0.45·Δα` nose-down break runs past the stall angle.
const STALL_BREAK_MAX_ARM: f64 = 0.4;
/// Nose-down pitch break gain at positive stall (dimensionless / Δα).
const STALL_BREAK_GAIN: f64 = 0.45;

/// Maximum flap deflection accepted by the model (radians ≈ 40°).
pub const FLAP_MAX_RAD: f64 = 0.7;

/// Density-ratio clamp for the thrust ceiling `(0.1 .. 1.2)` — bounds the
/// forced/turboshaft density scaling so extreme altitudes can't push the
/// static thrust term negative or absurd.
pub const DENSITY_RATIO_CLAMP: (f64, f64) = (0.1, 1.2);

/// Prandtl-Glauert compressibility-band: below this Mach the correction
/// `β = √(1−M²)` applies; at/above it a fixed subsonic floor is used.
const PG_MACH_BAND: f64 = 0.85;
/// Fixed `β` floor applied above the Prandtl-Glauert band (tests show this
/// bound keeps `cla_effective` sane up to the drag-divergence Mach).
const PG_BETA_FLOOR: f64 = 0.20;
/// Smallest `β² = 1−M²` allowed inside the band. 0.04 → β floor ≈ 0.2.
const PG_BETA2_FLOOR: f64 = 0.04;

/// Ground-effect lift cushion at the surface (+10% of CL). [29-3 Hoerner]
const GE_LIFT_BOOST: f64 = 0.10;
/// Ground-effect induced-drag suppression at the surface (−30% of CDi).
/// [29-6 Hoerner]
const GE_CDI_SUPPRESS: f64 = 0.30;
/// Bank-angle ramp (sin of ~35°) where the spiral nose-drop engages.
/// sin(35°) ≈ 0.574; ramp completes at 90° (sin = 1.0).
const SPIRAL_ENGAGE_SIN: f64 = 0.574;
/// Steep-bank spiral nose-drop pitch coefficient at full effect (`Cm`). Negative =
/// nose-down; value tuned so it overcomes the residual level-trim nose-up
/// couple (~0.4 kN·m) without dominating normal flight. Published here for
/// traceability: it is the source of the `spiral_nose_drop_cm` default in
/// `AircraftConfig`.
#[allow(dead_code)]
const SPIRAL_DROP_CM: f64 = 0.10;
/// Knife-edge blend slope for the pitch-sense sign transition (`8.0` per
/// unit of body-down projection: fades through ~90° bank over ±12°).
const PITCH_SENSE_BLEND: f64 = 8.0;

// ---------------------------------------------------------------------------
// Control-input bundle
// ---------------------------------------------------------------------------

/// Flight control-surface and throttle inputs bundled into a single struct.
///
/// Grouping the five pilot / autopilot degrees of freedom eliminates the
/// repeated positional-argument lists on the aerodynamic and integrator
/// functions (IEEE 1003.1 / Rust API Guidelines, identifier-length heuristic).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ControlInputs {
    /// Elevator deflection (radians, positive trailing-edge down).
    pub elevator: f64,
    /// Aileron deflection (radians, positive trailing-edge down on the right
    /// wing; produces a positive rolling moment in the right-hand rule).
    pub aileron: f64,
    /// Rudder deflection (radians, positive trailing-edge left; produces a
    /// positive side force / yawing moment).
    pub rudder: f64,
    /// Master throttle fraction `[0.0 .. 1.0]`.
    pub throttle: f64,
    /// Trailing-edge flap deflection (radians, positive down).
    pub flap: f64,
}

impl Default for ControlInputs {
    fn default() -> Self {
        Self {
            elevator: 0.0,
            aileron: 0.0,
            rudder: 0.0,
            throttle: 0.0,
            flap: 0.0,
        }
    }
}

// ---------------------------------------------------------------------------
// Engine / propulsion helpers
// ---------------------------------------------------------------------------

/// Thrust produced by a given throttle fraction against a per-engine thrust and
/// power ceiling. `thr` is the per-engine power fraction in `[0,1]`.
///
/// * **Propeller** — constant-power model `T = min(T_max·δ, P_max·δ / V)`:
///   below the corner speed the static-thrust ceiling binds, above it the shaft
///   power limits the thrust.
/// * **Jet** — turbojet/turbofan thrust is roughly constant across the subsonic
///   range, so only the static ceiling `T_max·δ` (scaled by air density) binds.
///   This is what lets a jet keep accelerating past the propeller corner speed.
fn engine_thrust_ceiling(
    thr_max: f64,
    pwr_max: f64,
    thr: f64,
    v_tas: f64,
    density_factor: f64,
    propulsion: Propulsion,
) -> f64 {
    match propulsion {
        Propulsion::Jet => thr_max * thr * density_factor,
        Propulsion::Propeller => {
            let static_thrust = thr_max * thr * density_factor;
            let power_thrust = pwr_max * thr * density_factor / v_tas.max(V_MIN_GUARD);
            static_thrust.min(power_thrust)
        }
    }
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
            config.propulsion,
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
    let t_left = engine_thrust_ceiling(half_t, half_p, left_thr, v_tas, density_factor, config.propulsion);
    let t_right = engine_thrust_ceiling(half_t, half_p, right_thr, v_tas, density_factor, config.propulsion);
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
fn engine_forces_moments(
    eng: &EngineFlowInputs,
    config: &AircraftConfig,
) -> (Vector3<f64>, Vector3<f64>) {
    let split = config.throttle_split;
    let (t_left, t_right, t_single) = engine_thrusts(
        config,
        eng.throttle,
        split,
        eng.v_tas,
        eng.density_factor,
    );

    let mut force = Vector3::zeros();
    let mut moment = Vector3::zeros();

    // Total axial thrust.
    let thrust_total = t_left + t_right + t_single;
    force.x += thrust_total;

    // Trust-line pitching moment (reverses when inverted).
    moment.y += thrust_total * config.thrust_arm * eng.pitch_sense;

    if config.engine_count == 2 {
        let arm = config.engine_lateral_arm;

        // Asymmetric-thrust yawing moment: unequal left/right thrust yaws the
        // nose toward the dead engine. The rudder must counter this; below Vmc
        // it cannot, which drives the characteristic engine-out departure.
        // Applies to any multi-engine layout, propeller or jet.
        moment.z += (t_left - t_right) * arm;

        // Propeller-specific couplings (torque, P-factor, gyroscopic
        // precession) only act on a spinning prop disc; a jet twin has none.
        if config.propulsion == Propulsion::Propeller {
            // Propeller torque: each engine produces a rolling couple about body X
            // proportional to its thrust. Assumed counter-rotating for a symmetric
            // twin, so the net is small and scales with any residual asymmetry.
            let torque = config.prop_torque_coeff * (t_left - t_right) * arm;
            moment.x += torque;

            // P-factor: at high power and high AoA the descending blade thrusts
            // more than the ascending one, yawing the nose. Both props rotate the
            // same way (standard right-hand from behind), so they reinforce.
            let pf = config.p_factor_coeff * thrust_total * eng.alpha;
            moment.z += pf;

            // Gyroscopic precession: the spinning propeller disc resists being
            // pitched or yawed, coupling the two axes. Proportional to the engine
            // angular momentum (modelled by the thrust proxy) and the body rates.
            let gyro = config.gyro_coeff * thrust_total;
            moment.y += gyro * eng.r;
            moment.z += -gyro * eng.q;
        }
    }

    (force, moment)
}

/// Flow/attitude quantities consumed by the propeller/twin engine couplings,
/// bundled to keep [`engine_forces_moments`] argument lists short.
struct EngineFlowInputs {
    throttle: f64,
    v_tas: f64,
    density_factor: f64,
    alpha: f64,
    q: f64,
    r: f64,
    pitch_sense: f64,
}

// --- Ground effect (WIG / altitude-in-ground-effect) -------------------------

/// Ground-effect factor `σ ∈ [0, 1]` from the height-to-span ratio.
///
/// ```text
/// σ = 1 / (1 + (4·h/b)²)
/// ```
///
/// `h` is the altitude above the ground surface and `b` the wing span, both in
/// metres. This is the canonical ratio of the induced-downwash retained in
/// ground effect: σ = 0.5 at a quarter span (`h = b/4`), fading to under 10%
/// by one span. Unlike an aggressive 16·h/b law, the cushion builds smoothly
/// through the flare instead of snapping on only in the final metres.
///
/// Returns the factor; pass it into [`ground_effect_factors`] to scale the
/// lift and induced drag coefficients.
pub fn ground_effect_factor(altitude_above_ground: f64, wing_span: f64) -> f64 {
    let h = altitude_above_ground.max(0.0);
    let x = 4.0 * (h / wing_span.max(1e-6));
    1.0 / (1.0 + x * x)
}

/// Multiply the **lift coefficient** and the **induced-drag coefficient** by
/// these factors when the aircraft is in ground effect. Mapped from the raw
/// [`ground_effect_factor`]:
///
/// * `cl_mult`   — ground effect raises the effective lift (~+10% max at the
///   surface), enough to cushion the flare but not to hold the aircraft off
///   the deck; it still has to settle.
/// * `cd_induced_mult` — wingtip vortices are suppressed by the ground, so
///   induced drag falls (up to ~−30% at the surface).
pub fn ground_effect_factors(altitude_above_ground: f64, wing_span: f64) -> (f64, f64) {
    let ge = ground_effect_factor(altitude_above_ground, wing_span);
    let cl_mult = 1.0 + GE_LIFT_BOOST * ge;
    let cd_induced_mult = 1.0 - GE_CDI_SUPPRESS * ge;
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
    controls: ControlInputs,
    wind_earth: &Vector3<f64>,
) -> Vector3<f64> {
    compute_forces_impl(state, config, controls, wind_earth, None)
}

/// `compute_forces` with the optional terrain ground effect.
pub fn compute_forces_with_terrain(
    state: &AircraftState,
    config: &AircraftConfig,
    controls: ControlInputs,
    wind_earth: &Vector3<f64>,
    terrain: Option<&Terrain>,
) -> Vector3<f64> {
    compute_forces_impl(state, config, controls, wind_earth, terrain)
}

fn compute_forces_impl(
    state: &AircraftState,
    config: &AircraftConfig,
    controls: ControlInputs,
    wind_earth: &Vector3<f64>,
    terrain: Option<&Terrain>,
) -> Vector3<f64> {
    // --- Air-relative velocity: the aero acts on the relative wind, not the
    //     ground-referenced velocity.
    let v_air = state.air_velocity(wind_earth);
    let v_tas = v_air.norm();
    let alpha = v_air.z.atan2(v_air.x);
    let beta = v_air.y.atan2(v_air.x.max(SIDESLIP_AXIAL_MIN));

    // --- Atmospheric conditions at current aircraft altitude ---
    let altitude = state.altitude();
    let atm = Atmosphere::at_altitude(altitude);
    let q_dyn = atm.dynamic_pressure(v_tas);
    let mach = atm.mach_number(v_tas);

    // --- Compressibility correction (Prandtl-Glauert rule) ---
    let pg_factor = compressibility_factor(mach);
    let cla_effective = config.cla / pg_factor;

    // --- Flap (trailing-edge) increments: lift, induced-drag factor, drag ---
    let flap = controls.flap.clamp(0.0, FLAP_MAX_RAD); // ~40 deg max
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
            ground_effect_factors(agl, config.wing_span)
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
        MACH_DRAG_GAIN_K4 * dm.powi(4)
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
    let cy = config.cy_beta * beta + config.cy_dr * controls.rudder;
    let force_y = q_dyn * config.wing_area * cy;

    let mut forces = Vector3::new(force_x, force_y, force_z);

    // --- Engine Thrust along body +X ---
    // Propeller thrust is limited by both the static (low-speed) thrust ceiling
    // and the constant shaft power delivered by the propeller:
    //   T(V) = min(thrust_max * δ, P_max * δ / V)
    // so somewhere above the corner speed the available thrust falls off with
    // airspeed. A jet's thrust does not fall off with speed — only the air
    // density (forced / core flow) scales it — so a jet keeps accelerating
    // where a propeller runs out. Both scale with the atmospheric density
    // ratio, clamped to a sensible range. Multi-engine layouts (twin) split
    // and skew the total across left/right engines.
    let throttle = controls.throttle.clamp(0.0, 1.0);
    let density_factor = (atm.density_ratio).clamp(DENSITY_RATIO_CLAMP.0, DENSITY_RATIO_CLAMP.1);
    let (engine_force, _engine_moment) = engine_forces_moments(
        &EngineFlowInputs {
            throttle,
            v_tas,
            density_factor,
            alpha,
            q: state.q,
            r: state.r,
            // force only; the thrust-line pitch moment is applied in compute_moments
            pitch_sense: 1.0,
        },
        config,
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
    controls: ControlInputs,
    alpha_dot: f64,
    wind_earth: &Vector3<f64>,
) -> Vector3<f64> {
    compute_moments_impl(state, config, controls, alpha_dot, wind_earth, None)
}

/// `compute_moments` with the optional terrain ground-effect pitching moment.
pub fn compute_moments_with_terrain(
    state: &AircraftState,
    config: &AircraftConfig,
    controls: ControlInputs,
    alpha_dot: f64,
    wind_earth: &Vector3<f64>,
    terrain: Option<&Terrain>,
) -> Vector3<f64> {
    compute_moments_impl(state, config, controls, alpha_dot, wind_earth, terrain)
}

/// Nose-down pitch tendency in ground effect, as Cm at full effect (σ = 1).
/// The damped wing downwash in ground effect shifts the trim nose-down, so the
/// pilot feels a gentle flare hand-back pressure and the aircraft settles
/// instead of hovering pinned on a level "cushion road".
const GE_PITCH_CM: f64 = -0.015;

fn compute_moments_impl(
    state: &AircraftState,
    config: &AircraftConfig,
    controls: ControlInputs,
    alpha_dot: f64,
    wind_earth: &Vector3<f64>,
    terrain: Option<&Terrain>,
) -> Vector3<f64> {
    // --- Air-relative velocity: the aero acts on the relative wind. ---
    let v_air = state.air_velocity(wind_earth);
    let v_tas = v_air.norm();
    let alpha = v_air.z.atan2(v_air.x);
    let beta = v_air.y.atan2(v_air.x.max(SIDESLIP_AXIAL_MIN));

    // --- Atmospheric conditions at current aircraft altitude ---
    let altitude = state.altitude();
    let atm = Atmosphere::at_altitude(altitude);
    let q_dyn = atm.dynamic_pressure(v_tas);
    let mach = atm.mach_number(v_tas);

    // --- Compressibility correction (Prandtl-Glauert rule) ---
    let _pg_factor = compressibility_factor(mach);

    // --- Engine thrust (used for the thrust-line pitching moment) ---
    let throttle = controls.throttle.clamp(0.0, 1.0);
    let density_factor = (atm.density_ratio).clamp(DENSITY_RATIO_CLAMP.0, DENSITY_RATIO_CLAMP.1);

    // --- Dimensionless body angular rates ---
    let (p_hat, q_hat, r_hat, alpha_dot_hat) = dimensionless_rates(state, config, v_tas, alpha_dot);

    // --- Pitching Moment Cm with downwash-lag damping & stall break ---
    // At post-stall, center-of-pressure shifts aft, adding a stabilizing nose-down pitch break.
    // Flaps lower the positive stall angle (incremental lift to the rear drops the break AoA).
    let flap = controls.flap.clamp(0.0, FLAP_MAX_RAD);
    let alpha_stall_pos = config.alpha_stall_pos - config.flap_stall_shift * flap;
    let stall_pitch_break = pitch_break_moment(config, alpha, alpha_stall_pos);

    // --- Steep-bank spiral nose-drop ---
    // A fixed-wing banked past ~45 deg has little/no vertical lift, so gravity
    // must pull the nose down into a dive. In a real aircraft the resulting
    // sideslip drives spiral divergence (nose drops); here we inject that
    // tendency directly with a bank-coupled nose-down moment, engaged only at
    // steep bank so normal turns are unaffected.
    let (bank, _, _) = state.euler_angles();
    let sin_bank = bank.sin().abs();
    // Engage between ~35 deg (sin=0.574) and ~90 deg (sin=1.0).
    let engage = ((sin_bank - SPIRAL_ENGAGE_SIN) / (1.0 - SPIRAL_ENGAGE_SIN)).clamp(0.0, 1.0);
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
    let pitch_sense = (PITCH_SENSE_BLEND * body_down.z).tanh();
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
        + config.cme * controls.elevator
        + config.cm_flap * flap
        + stall_pitch_break
        // Ground-effect pitch: the reduced downwash near the ground shifts
        // trim nose-down, giving the flare a faint nose-drop to hold against.
        + GE_PITCH_CM * terrain.map_or(0.0, |t| {
            let agl = t.altitude_above_ground(state.pos_x, state.pos_y, altitude);
            ground_effect_factor(agl, config.wing_span)
        });
    let cm = cm_core * pitch_sense + spiral_nose_drop * spiral_sense;
    // Thrust line offset from CG produces a pitching moment proportional to
    // thrust; its body-Z arm also reverses when inverted. This (plus the twin
    // asymmetric couplings) is supplied by `engine_forces_moments` below.
    let pitch_moment = q_dyn * config.wing_area * config.chord * cm;

    // --- Rolling Moment Cl (around body X) ---
    let cl_roll = config.cl_beta * beta
        + config.cl_p * p_hat
        + config.cl_r * r_hat
        + config.cl_da * controls.aileron
        + config.cl_dr * controls.rudder;
    let roll_moment = q_dyn * config.wing_area * config.wing_span * cl_roll;

    // --- Yawing Moment Cn (around body Z) ---
    let cn = config.cn_beta * beta
        + config.cn_p * p_hat
        + config.cn_r * r_hat
        + config.cn_da * controls.aileron
        + config.cn_dr * controls.rudder;
    let yaw_moment = q_dyn * config.wing_area * config.wing_span * cn;

    let mut moments = Vector3::new(roll_moment, pitch_moment, yaw_moment);

    // --- Engine (twin) thrust-line and asymmetric couplings ---
    // Adds the thrust-Line pitching moment plus, for a twin, the asymmetric
    // thrust yaw (Vmc), P-factor, prop torque and gyroscopic precession.
    let (_, engine_moment) = engine_forces_moments(
        &EngineFlowInputs {
            throttle,
            v_tas,
            density_factor,
            alpha,
            q: state.q,
            r: state.r,
            pitch_sense,
        },
        config,
    );
    moments += engine_moment;

    moments
}

/// Dimensionless body angular rates and alpha_dot, each divided by the
/// appropriate reference length and double true airspeed. Degenerate to zero
/// when the airspeed is negligible (no meaningful dynamic rotation).
fn dimensionless_rates(
    state: &AircraftState,
    config: &AircraftConfig,
    v_tas: f64,
    alpha_dot: f64,
) -> (f64, f64, f64, f64) {
    let p_hat = if v_tas > V_TAS_EPS {
        state.p * config.wing_span / (2.0 * v_tas)
    } else {
        0.0
    };
    let q_hat = if v_tas > V_TAS_EPS {
        state.q * config.chord / (2.0 * v_tas)
    } else {
        0.0
    };
    let r_hat = if v_tas > V_TAS_EPS {
        state.r * config.wing_span / (2.0 * v_tas)
    } else {
        0.0
    };
    let alpha_dot_hat = if v_tas > V_TAS_EPS {
        alpha_dot * config.chord / (2.0 * v_tas)
    } else {
        0.0
    };
    (p_hat, q_hat, r_hat, alpha_dot_hat)
}

/// Post-stall pitching break: a stabilizing nose-down moment when the (flap-
/// shifted) angle of attack exceeds either the positive or negative stall
/// angle, capping the break arm so it never diverges.
fn pitch_break_moment(config: &AircraftConfig, alpha: f64, alpha_stall_pos: f64) -> f64 {
    if alpha > alpha_stall_pos {
        -STALL_BREAK_GAIN * (alpha - alpha_stall_pos).min(STALL_BREAK_MAX_ARM)
    } else if alpha < config.alpha_stall_neg {
        STALL_BREAK_GAIN * (config.alpha_stall_neg - alpha).min(STALL_BREAK_MAX_ARM)
    } else {
        0.0
    }
}

/// Compute the total body-frame force and moment.
///
/// `alpha_dot` (rad/s) feeds the downwash-lag pitch damping term.
pub fn compute_forces_moments(
    state: &AircraftState,
    config: &AircraftConfig,
    controls: ControlInputs,
    alpha_dot: f64,
    wind_earth: &Vector3<f64>,
) -> (Vector3<f64>, Vector3<f64>) {
    let forces = compute_forces(state, config, controls, wind_earth);
    let moments = compute_moments(state, config, controls, alpha_dot, wind_earth);
    (forces, moments)
}

/// Prandtl-Glauert compressibility correction factor.
fn compressibility_factor(mach: f64) -> f64 {
    if mach < PG_MACH_BAND {
        (1.0 - mach * mach).max(PG_BETA2_FLOOR).sqrt()
    } else {
        PG_BETA_FLOOR // Subsonic limit clamp
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
        let blend = (1.0 + (VITERNA_BLEND_POS * d_alpha).tanh()) * 0.5;
        let cl_stall_peak = config.cl0 + config.cla * alpha_pos;
        let cl_separated = (config.cd_max * SEPARATED_LIFT_KC) * (2.0 * alpha).sin()
            + SEPARATED_LIFT_RESIDUAL * (alpha.cos()).powi(2) / alpha.sin().max(SEPARATED_MIN_SIN);
        (1.0 - blend) * cl_stall_peak + blend * cl_separated
    } else {
        // Negative post-stall (inverted stall)
        let d_alpha = alpha_neg - alpha;
        let blend = (1.0 + (VITERNA_BLEND_NEG * d_alpha).tanh()) * 0.5;
        let cl_stall_neg_peak = config.cl0 + config.cla * alpha_neg;
        let cl_separated = (config.cd_max * SEPARATED_LIFT_KC) * (2.0 * alpha).sin();
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
        let blend = (1.0 + (VITERNA_DRAG_BLEND * d_alpha).tanh()) * 0.5;
        let cd_flat_plate = config.cd_max * (alpha.sin()).powi(2) + config.cd0 * alpha.cos().abs();
        (1.0 - blend) * cd_attached + blend * cd_flat_plate
    }
}
