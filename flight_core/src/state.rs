//! Rigid-body aircraft state representation.

use nalgebra::{Quaternion, UnitQuaternion, Vector3};

use crate::atmosphere::{Atmosphere, RHO0_SL};
use crate::config::AircraftConfig;

/// The full 6-DOF state of the aircraft.
///
/// Position is expressed in the Earth-fixed **NED** frame
/// (North–East–Down), while velocity and angular rates are expressed in
/// the **body** frame attached to the aircraft.
#[derive(Debug, Clone)]
pub struct AircraftState {
    /// Earth-frame position. NED: `pos_x` = North, `pos_y` = East,
    /// `pos_z` = Down (positive downward). Altitude = `-pos_z`.
    pub pos_x: f64,
    pub pos_y: f64,
    pub pos_z: f64,

    /// Body-frame linear velocity components.
    /// `u` = forward (body X), `v` = right (body Y), `w` = down (body Z).
    pub u: f64,
    pub v: f64,
    pub w: f64,

    /// Quaternion orientation `(w, x, y, z)` rotating Earth frame into
    /// body frame.
    pub q0: f64, // w
    pub q1: f64, // x
    pub q2: f64, // y
    pub q3: f64, // z

    /// Body-frame angular rates in radians per second.
    /// `p` = roll rate, `q` = pitch rate, `r` = yaw rate.
    pub p: f64,
    pub q: f64,
    pub r: f64,
}

impl Default for AircraftState {
    fn default() -> Self {
        Self {
            pos_x: 0.0,
            pos_y: 0.0,
            pos_z: 0.0,
            u: 0.0,
            v: 0.0,
            w: 0.0,
            q0: 1.0,
            q1: 0.0,
            q2: 0.0,
            q3: 0.0,
            p: 0.0,
            q: 0.0,
            r: 0.0,
        }
    }
}

impl AircraftState {
    /// Trims the aircraft state to exact 6-DOF steady, straight, wings-level
    /// flight equilibrium (Lift = Weight, Thrust = Drag, Moments = 0) at the
    /// specified target altitude and airspeed.
    ///
    /// Returns the required control trims: `(elevator_trim_rad, throttle_trim)`.
    pub fn trim_level_flight(
        &mut self,
        config: &AircraftConfig,
        altitude: f64,
        speed: f64,
    ) -> (f64, f64) {
        let atm = Atmosphere::at_altitude(altitude);
        let q_dyn = atm.dynamic_pressure(speed);
        let weight = config.mass * 9.80665;

        // Account for Prandtl-Glauert compressibility slope increase at cruise Mach
        let mach = atm.mach_number(speed);
        let pg_factor = if mach < 0.85 {
            f64::sqrt(f64::max(1.0 - mach * mach, 0.04))
        } else {
            0.20
        };
        let cla_effective = config.cla / pg_factor;

        // 1. Lift = Weight -> required CL
        let cl_req = weight / (q_dyn * config.wing_area);

        // 2. Required Angle of Attack (AoA)
        let alpha_trim = (cl_req - config.cl0) / cla_effective.max(1e-3);

        // 3. Drag = Thrust -> required thrust level
        let cd_req = config.cd0 + config.induced_drag_k() * cl_req * cl_req;
        let drag = q_dyn * config.wing_area * cd_req;
        let thrust_req = drag / alpha_trim.cos();

        // 4. Pitching moment = 0 -> required elevator trim, including the
        //    thrust-line pitching moment (engine offset from CG).
        let cm_thrust = (thrust_req * config.thrust_arm)
            / (q_dyn * config.wing_area * config.chord).max(1e-6);
        let elev_trim = -(config.cm0 + config.cma * alpha_trim + cm_thrust) / config.cme;

        // 5. Throttle setting needed to produce the required thrust. This
        //    must mirror the aero model's `min(static, power/V)` fall-off,
        //    otherwise trims requested above the corner speed would falsely
        //    assume more thrust than the constant-power propeller delivers.
        let density_factor = (atm.density / RHO0_SL).clamp(0.1, 1.2);
        let static_ceiling = config.thrust_max * density_factor;
        let power_ceiling = config.power_max * density_factor / speed.max(6.0);
        let thrust_avail = static_ceiling.min(power_ceiling);
        let throttle_trim = (thrust_req / thrust_avail.max(1.0)).clamp(0.0, 1.0);

        // 6. Populate rigid-body state
        self.pos_x = 0.0;
        self.pos_y = 0.0;
        self.pos_z = -altitude;

        // Body velocities: nose pitched up by alpha_trim relative to horizontal airflow
        self.u = speed * alpha_trim.cos();
        self.v = 0.0;
        self.w = speed * alpha_trim.sin();

        // Attitude quaternion: pitch up by alpha_trim around body Y.
        // The quaternion is (cos, 0, -sin, 0) under nalgebra's axis-angle
        // sense for a nose-up rotation; this makes the earth-frame velocity
        // of a level flight state exactly horizontal.
        let half_pitch = alpha_trim * 0.5;
        self.q0 = half_pitch.cos();
        self.q1 = 0.0;
        self.q2 = -half_pitch.sin();
        self.q3 = 0.0;

        self.p = 0.0;
        self.q = 0.0;
        self.r = 0.0;

        (elev_trim, throttle_trim)
    }

    /// Angle of attack (AoA), the angle between the body X axis and the
    /// relative wind projected into the body X–Z plane. Positive AoA means
    /// the nose is above the airflow direction.
    pub fn angle_of_attack(&self) -> f64 {
        (self.w).atan2(self.u)
    }

    /// Sideslip angle (beta) in radians, the angle between the relative wind
    /// and the body X axis in the body X–Y plane.
    pub fn sideslip_angle(&self) -> f64 {
        (self.v).atan2(self.u.max(1e-6))
    }

    /// Total airspeed magnitude, sqrt(u² + v² + w²) in m/s.
    pub fn airspeed(&self) -> f64 {
        (self.u * self.u + self.v * self.v + self.w * self.w).sqrt()
    }

    /// Normalize the orientation quaternion back to unit length. Called
    /// after every integration step to prevent drift.
    pub fn normalize_quaternion(&mut self) {
        let norm = (self.q0 * self.q0 + self.q1 * self.q1 + self.q2 * self.q2 + self.q3 * self.q3)
            .sqrt();
        if norm > 1e-12 {
            let inv = 1.0 / norm;
            self.q0 *= inv;
            self.q1 *= inv;
            self.q2 *= inv;
            self.q3 *= inv;
        }
    }

    /// Get the orientation as a nalgebra `UnitQuaternion` (Earth -> body).
    pub fn rotation_earth_to_body(&self) -> UnitQuaternion<f64> {
        UnitQuaternion::new_normalize(Quaternion::new(self.q0, self.q1, self.q2, self.q3))
    }

    /// Convert the full state into a flat observation array used by the
    /// reinforcement-learning agent.
    pub fn to_observation_array(&self) -> [f64; 12] {
        [
            self.pos_x,
            self.pos_z,
            self.u,
            self.v,
            self.w,
            self.q0,
            self.q1,
            self.q2,
            self.q3,
            self.p,
            self.q,
            self.r,
        ]
    }

    /// Altitude above the reference (sea-level) ground plane in meters.
    /// NED `pos_z` is positive downward, so altitude is `-pos_z`.
    pub fn altitude(&self) -> f64 {
        -self.pos_z
    }

    /// Extract the classical Tait–Bryan Euler angles (roll, pitch, yaw)
    /// from the Earth→body quaternion. Radians. Positive pitch is nose-up,
    /// positive roll is right-wing-down, computed from the body axes in the
    /// NED earth frame for a consistent convention.
    pub fn euler_angles(&self) -> (f64, f64, f64) {
        let (fwd, right, down) = self.body_axes_in_earth();
        let pitch = (-fwd.z).clamp(-1.0, 1.0).asin();
        let roll = right.z.atan2(down.z);
        let yaw = fwd.y.atan2(fwd.x);
        (roll, pitch, yaw)
    }

    /// Flight-path-climb angle: the pitch of the true velocity vector
    /// relative to the horizon. Positive = climbing.
    pub fn flight_path_angle(&self) -> f64 {
        let body = Vector3::new(self.u, self.v, self.w);
        let earth = self.rotation_earth_to_body().inverse().transform_vector(&body);
        let climb = -earth.z;
        let horiz = (earth.x * earth.x + earth.y * earth.y).sqrt().max(1e-6);
        climb.atan2(horiz)
    }

    /// Computes the aircraft body axes `(forward, right, down)` expressed
    /// in the Earth NED coordinate frame.
    pub fn body_axes_in_earth(&self) -> (Vector3<f64>, Vector3<f64>, Vector3<f64>) {
        let rot_body_to_earth = self.rotation_earth_to_body().inverse();
        let fwd = rot_body_to_earth.transform_vector(&Vector3::new(1.0, 0.0, 0.0));
        let right = rot_body_to_earth.transform_vector(&Vector3::new(0.0, 1.0, 0.0));
        let down = rot_body_to_earth.transform_vector(&Vector3::new(0.0, 0.0, 1.0));
        (fwd, right, down)
    }
}
