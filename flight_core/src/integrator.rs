//! 4th-order Runge-Kutta (RK4) rigid-body integrator.
//!
//! The integrator advances a 13-element state vector consisting of:
//!   * Earth-frame position `(x, y, z)` — 3
//!   * Body-frame linear velocity `(u, v, w)` — 3
//!   * Quaternion orientation `(q0, q1, q2, q3)` — 4
//!   * Body-frame angular rates `(p, q, r)` — 3

use nalgebra::{UnitQuaternion, Vector3};

use crate::aero::{compute_forces_with_terrain, compute_moments};
use crate::config::AircraftConfig;
use crate::shape::compute_shape_wind;
use crate::state::AircraftState;
use crate::terrain::Terrain;

/// Compressed state used for RK4 derivative evaluation.
#[derive(Clone)]
struct DynState {
    pos_x: f64,
    pos_y: f64,
    pos_z: f64,
    u: f64,
    v: f64,
    w: f64,
    q0: f64,
    q1: f64,
    q2: f64,
    q3: f64,
    p: f64,
    q: f64,
    r: f64,
}

impl From<&AircraftState> for DynState {
    fn from(s: &AircraftState) -> Self {
        DynState {
            pos_x: s.pos_x,
            pos_y: s.pos_y,
            pos_z: s.pos_z,
            u: s.u,
            v: s.v,
            w: s.w,
            q0: s.q0,
            q1: s.q1,
            q2: s.q2,
            q3: s.q3,
            p: s.p,
            q: s.q,
            r: s.r,
        }
    }
}

impl DynState {
    /// Convert a dynamic state back into an `AircraftState`.
    fn into_state(self) -> AircraftState {
        AircraftState {
            pos_x: self.pos_x,
            pos_y: self.pos_y,
            pos_z: self.pos_z,
            u: self.u,
            v: self.v,
            w: self.w,
            q0: self.q0,
            q1: self.q1,
            q2: self.q2,
            q3: self.q3,
            p: self.p,
            q: self.q,
            r: self.r,
        }
    }

    /// Estimate the unit quaternion from the (possibly unnormalized) scalar
    /// components, used during intermediate RK4 stages where the norm is
    /// only approximately 1.
    fn rotation(&self) -> UnitQuaternion<f64> {
        UnitQuaternion::new_normalize(nalgebra::Quaternion::new(
            self.q0, self.q1, self.q2, self.q3,
        ))
    }
}

/// Compute the time-derivative of the full dynamics state.
fn derivatives(
    s: &DynState,
    config: &AircraftConfig,
    elevator: f64,
    aileron: f64,
    rudder: f64,
    throttle: f64,
    flap: f64,
    wind_earth: &Vector3<f64>,
    terrain: Option<&Terrain>,
) -> DynState {
    // --- Reconstruct an AircraftState to reuse aero/observer logic ------
    let aircraft = s.clone().into_state();
    let mut forces = compute_forces_with_terrain(
        &aircraft,
        config,
        elevator,
        aileron,
        rudder,
        throttle,
        flap,
        wind_earth,
        terrain,
    );

    // Shape-based wind interaction: the imposed wind pushes on the aircraft's
    // flat-plate collision-shape panels, adding a geometry-dependent force.
    let (shape_force, shape_moment) = compute_shape_wind(&aircraft, &config.collision_panels, wind_earth);
    forces += shape_force;

    // --- Translational kinematics ---------------------------------------
    let rot_earth_to_body: UnitQuaternion<f64> = s.rotation();
    let rot_body_to_earth = rot_earth_to_body.inverse();
    let body_vel = Vector3::new(s.u, s.v, s.w);
    let earth_vel = rot_body_to_earth.transform_vector(&body_vel);

    // --- Translational dynamics (Newton's 2nd law, rotating frame) ------
    let omega = Vector3::new(s.p, s.q, s.r);
    let accel = forces / config.mass - omega.cross(&body_vel);

    // --- Rotational kinematics: quaternion propagation ------------------
    // Positive q (pitch rate about body +Y) is a nose-up rotation for this
    // quaternion convention (see trim_level_flight / from_axis_angle(-Y)),
    // matching the -omega.cross(&body_vel) transport term above.
    let omega_quat = nalgebra::Quaternion::new(0.0, -s.p, -s.q, -s.r);
    let q_as_quat = nalgebra::Quaternion::new(s.q0, s.q1, s.q2, s.q3);
    let q_view_dot = 0.5 * q_as_quat * omega_quat;

    // --- Alpha-dot from the kinematic acceleration (for Cm_adot damping).
    //     alpha = atan2(w, u)  =>  d(alpha)/dt = (u*w_dot - w*u_dot) / (u² + w²).
    //     The body accelerations depend only on the forces, not the moments,
    //     so there is no circularity here.
    let v_t_sq = s.u * s.u + s.w * s.w;
    let alpha_dot = (s.u * accel.z - s.w * accel.x) / v_t_sq.max(1e-6);

    // --- Rotational dynamics (Euler's equations) ------------------------
    let mut moments =
        compute_moments(&aircraft, config, elevator, aileron, rudder, throttle, alpha_dot, flap, wind_earth);
    moments += shape_moment;
    let ixx = config.ixx;
    let iyy = config.iyy;
    let izz = config.izz;
    let p_dot = (moments.x + (iyy - izz) * s.q * s.r) / ixx;
    let q_dot = (moments.y + (izz - ixx) * s.p * s.r) / iyy;
    let r_dot = (moments.z + (ixx - iyy) * s.p * s.q) / izz;

    // --- Assemble the derivative vector ---------------------------------
    DynState {
        pos_x: earth_vel.x,
        pos_y: earth_vel.y,
        pos_z: earth_vel.z,
        u: accel.x,
        v: accel.y,
        w: accel.z,
        q0: q_view_dot.w,
        q1: q_view_dot.i,
        q2: q_view_dot.j,
        q3: q_view_dot.k,
        p: p_dot,
        q: q_dot,
        r: r_dot,
    }
}

/// Advance the aircraft state by `dt` seconds using RK4 with full 6-DOF control inputs.
///
/// `terrain` (optional) enables terrain ground collision, ground-effect force
/// augmentation and orographic (terrain-induced) vertical wind. With `None`
/// the aircraft flies over a flat infinite plane at the reference datum.
pub fn step(
    state: &mut AircraftState,
    config: &AircraftConfig,
    elevator: f64,
    aileron: f64,
    rudder: f64,
    throttle: f64,
    flap: f64,
    wind_earth: Option<&Vector3<f64>>,
    dt: f64,
    terrain: Option<&Terrain>,
) {
    let s0 = DynState::from(&*state);
    let half = dt * 0.5;

    // Resolve the wind field once for the entire step (quasi-static over dt),
    // adding any terrain-orographic vertical wind on top of the supplied field.
    let zero_wind = Vector3::zeros();
    let mut wind_buf = wind_earth.copied().unwrap_or(zero_wind);
    if let Some(t) = terrain {
        let wr = wind_earth.unwrap_or(&zero_wind);
        wind_buf += t.orographic_wind(
            wr,
            state.pos_x,
            state.pos_y,
            state.altitude(),
            300.0, // orographic decay height scale (m AGL)
        );
    }
    let wind = &wind_buf;

    let k1 = derivatives(&s0, config, elevator, aileron, rudder, throttle, flap, wind, terrain);

    let s2 = add_scaled(&s0, &k1, half);
    let k2 = derivatives(&s2, config, elevator, aileron, rudder, throttle, flap, wind, terrain);

    let s3 = add_scaled(&s0, &k2, half);
    let k3 = derivatives(&s3, config, elevator, aileron, rudder, throttle, flap, wind, terrain);

    let s4 = add_scaled(&s0, &k3, dt);
    let k4 = derivatives(&s4, config, elevator, aileron, rudder, throttle, flap, wind, terrain);

    // Combine the four stage slopes: (k1 + 2k2 + 2k3 + k4)/6.
    let one_sixth = dt / 6.0;
    let combined = combine_k(&k1, &k2, &k3, &k4, one_sixth);
    let next = add_scaled(&s0, &combined, 1.0);

    *state = next.into_state();
    state.normalize_quaternion();

    // --- Ground-plane collision ------------------------------------------
    // NED frame: altitude = -pos_z. Without a terrain the ground is a flat
    // plane at the reference datum (pos_z = 0). With terrain supplied the
    // ground follows the elevation surface `h(pos_x, pos_y)`, i.e. the aircraft
    // rests on pos_z = -h. In both cases clamp the position to the surface and
    // kill the descent so the aircraft "lands" rather than passing through.
    let ground_z_ned = match terrain {
        Some(t) => -t.height(state.pos_x, state.pos_y),
        None => 0.0,
    };
    if state.pos_z > ground_z_ned {
        state.pos_z = ground_z_ned;
        // Project out the downward component of the earth-frame velocity by
        // removing it from the body-frame velocity. NED up is -z.
        let rot_earth_to_body = UnitQuaternion::new_normalize(nalgebra::Quaternion::new(
            state.q0, state.q1, state.q2, state.q3,
        ));
        let up_earth = Vector3::new(0.0, 0.0, -1.0); // NED up is -z
        let up_body = rot_earth_to_body.transform_vector(&up_earth);
        let bv = Vector3::new(state.u, state.v, state.w);
        let velocity_along_up = bv.dot(&up_body);
        if velocity_along_up < 0.0 {
            let corrected = bv - up_body * velocity_along_up;
            state.u = corrected.x;
            state.v = corrected.y;
            state.w = corrected.z;
        }
    }
}

/// `a + b * scale` element-wise, used for RK4 stage construction.
fn add_scaled(a: &DynState, b: &DynState, scale: f64) -> DynState {
    DynState {
        pos_x: a.pos_x + scale * b.pos_x,
        pos_y: a.pos_y + scale * b.pos_y,
        pos_z: a.pos_z + scale * b.pos_z,
        u: a.u + scale * b.u,
        v: a.v + scale * b.v,
        w: a.w + scale * b.w,
        q0: a.q0 + scale * b.q0,
        q1: a.q1 + scale * b.q1,
        q2: a.q2 + scale * b.q2,
        q3: a.q3 + scale * b.q3,
        p: a.p + scale * b.p,
        q: a.q + scale * b.q,
        r: a.r + scale * b.r,
    }
}

/// Weighted combination `(k1 + 2*k2 + 2*k3 + k4) * w`.
fn combine_k(k1: &DynState, k2: &DynState, k3: &DynState, k4: &DynState, w: f64) -> DynState {
    DynState {
        pos_x: (k1.pos_x + 2.0 * k2.pos_x + 2.0 * k3.pos_x + k4.pos_x) * w,
        pos_y: (k1.pos_y + 2.0 * k2.pos_y + 2.0 * k3.pos_y + k4.pos_y) * w,
        pos_z: (k1.pos_z + 2.0 * k2.pos_z + 2.0 * k3.pos_z + k4.pos_z) * w,
        u: (k1.u + 2.0 * k2.u + 2.0 * k3.u + k4.u) * w,
        v: (k1.v + 2.0 * k2.v + 2.0 * k3.v + k4.v) * w,
        w: (k1.w + 2.0 * k2.w + 2.0 * k3.w + k4.w) * w,
        q0: (k1.q0 + 2.0 * k2.q0 + 2.0 * k3.q0 + k4.q0) * w,
        q1: (k1.q1 + 2.0 * k2.q1 + 2.0 * k3.q1 + k4.q1) * w,
        q2: (k1.q2 + 2.0 * k2.q2 + 2.0 * k3.q2 + k4.q2) * w,
        q3: (k1.q3 + 2.0 * k2.q3 + 2.0 * k3.q3 + k4.q3) * w,
        p: (k1.p + 2.0 * k2.p + 2.0 * k3.p + k4.p) * w,
        q: (k1.q + 2.0 * k2.q + 2.0 * k3.q + k4.q) * w,
        r: (k1.r + 2.0 * k2.r + 2.0 * k3.r + k4.r) * w,
    }
}
