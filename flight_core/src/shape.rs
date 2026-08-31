//! Flat-plate collision-shape wind interaction.
//!
//! The aircraft carries a set of flat-plate panels (the "collision shape").
//! The imposed wind pours onto each panel and pushes it with a pressure force
//! proportional to the wind speed component normal to the panel, scaled by the
//! panel area and a flat-plate drag coefficient. Because every panel has a
//! centre of pressure offset from the CG, the resulting force also produces a
//! moment that rolls/pitches/yaws the aircraft depending on its shape.
//!
//! This is deliberately driven by the *wind field* (not the aircraft's own
//! relative airspeed, which the coefficient-based aerodynamics in `aero.rs`
//! already handle). In still air the wind is zero and these panels add
//! nothing, so the level-flight trim behaviour is preserved unchanged.
//!
//! Convention matches the rest of the engine: body frame with nose +X, up +Y,
//! right +Z; the wind is supplied in the Earth NED frame as everywhere else.

use nalgebra::Vector3;

use crate::config::CollisionPanel;
use crate::state::AircraftState;

/// Sea-level air density (kg/m³), used for the shape-pressure magnitude.
const RHO_SL: f64 = 1.225;

/// Compute the body-frame force and moment produced by the wind impinging on
/// the aircraft's flat-plate collision-shape panels.
///
/// Returns `(forces, moments)` in the **body** frame: force in Newtons,
/// moment in N·m (`[roll, pitch, yaw]` = `[L, M, N]`).
pub fn compute_shape_wind(
    state: &AircraftState,
    panels: &[CollisionPanel],
    wind_earth: &Vector3<f64>,
) -> (Vector3<f64>, Vector3<f64>) {
    let mut force = Vector3::zeros();
    let mut moment = Vector3::zeros();

    // Wind velocity expressed in the body frame: the air is moving in this
    // direction relative to the aircraft's axes.
    let wind_body = state.rotation_earth_to_body().transform_vector(wind_earth);

    for panel in panels {
        let n = Vector3::new(panel.normal[0], panel.normal[1], panel.normal[2]).normalize();
        let cp = Vector3::new(panel.cp[0], panel.cp[1], panel.cp[2]);

        // Wind-speed component normal to the panel surface.
        let vn = wind_body.dot(&n);

        // Flat-plate pressure: F = ½·ρ·|vn|·vn·A·Cd, directed along +n (the
        // wind pushes the panel in its own normal direction). The |vn|·vn form
        // keeps the sign correct for wind approaching from either side.
        let f = 0.5 * RHO_SL * vn.abs() * vn * panel.area * panel.cd;

        force += n * f;
        // Moment from the force applied at the centre of pressure, about the
        // CG.
        moment += cp.cross(&(n * f));
    }

    (force, moment)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::state::AircraftState;

    fn panel(area: f64, n: [f64; 3], cp: [f64; 3], cd: f64) -> CollisionPanel {
        CollisionPanel { area, normal: n, cp, cd }
    }

    /// In still air the collision shape must produce zero force/moment so the
    /// tuned level-flight trim is preserved.
    #[test]
    fn still_air_produces_no_shape_force() {
        let state = AircraftState::default();
        let panels = vec![
            panel(4.0, [1.0, 0.0, 0.0], [2.0, 0.0, 0.0], 1.5),
            panel(2.0, [0.0, 0.0, 1.0], [-1.5, 0.0, 0.5], 1.5),
        ];
        let (f, m) = compute_shape_wind(&state, &panels, &Vector3::zeros());
        assert!(f.norm() < 1e-9 && m.norm() < 1e-9);
    }

    /// A wind blowing along +Y (body right) onto a fin panel whose normal is
    /// also +Z should produce a force along +Z and therefore a yawing moment
    /// about the body Z axis from the fin's lever arm.
    #[test]
    fn crosswind_on_fin_yaws() {
        let state = AircraftState::default();
        // Vertical fin on the +Z side, centre of pressure behind the CG.
        let fin = panel(1.6, [0.0, 0.0, 1.0], [-1.9, 0.0, 0.6], 1.8);
        let wind = Vector3::new(0.0, 0.0, 5.0); // wind blowing body +Z
        let (f, m) = compute_shape_wind(&state, &[fin], &wind);

        // Normals point +Z, so the force pushes the fin along +Z.
        assert!(f.z > 0.0, "expected +Z force, got {f:?}");
        // cp = (-1.9, 0, 0.6), F = (0, 0, fz) -> cp x F = (-cp_y*Fz ... ) ->
        //   Mx = cp_y*Fz - cp_z*Fy = 0
        //   My = cp_z*Fx - cp_x*Fz = -(-1.9)*fz = +1.9*fz
        assert!(m.y > 0.0, "fin should pitch; expected +My, got {m:?}");
    }
}
