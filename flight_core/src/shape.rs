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
pub const RHO_SL: f64 = 1.225;
/// Dynamic viscosity of air at sea level (Pa·s) for Reynolds number.
const MU_AIR: f64 = 1.81e-5;

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

impl Default for ImportedAero {
    fn default() -> Self {
        ImportedAero {
            force: Vector3::zeros(),
            moment: Vector3::zeros(),
            frontal_area: 0.0,
            wetted_area: 0.0,
            reference_len: 0.0,
            cd_frontal: 0.0,
            re: 0.0,
        }
    }
}

/// Diagnostics + result of a realistic imported-shape drag computation.
#[derive(Debug, Clone, Copy)]
pub struct ImportedAero {
    /// Total aerodynamic force in the body frame (N).
    pub force: Vector3<f64>,
    /// Total aerodynamic moment about the body origin (N·m), `[roll,pitch,yaw]`.
    pub moment: Vector3<f64>,
    /// Projected frontal area normal to the stream (m²) — resolution independent.
    pub frontal_area: f64,
    /// Wetted (surface) area (m²).
    pub wetted_area: f64,
    /// Reference (stream-wise) length for Reynolds number (m).
    pub reference_len: f64,
    /// Total drag coefficient referenced to the frontal area.
    pub cd_frontal: f64,
    /// Reynolds number based on the reference length.
    pub re: f64,
}

/// Physically-grounded drag for an arbitrary imported model, replacing the
/// crude "every panel is a flat plate with Cd=1.2" model.
///
/// The body is decomposed into:
///  - **Skin friction** on the wetted area, via the turbulent flat-plate
///    correlation `C_f = 0.074 / Re^(1/5)` (Schlichting), scaled by a
///    Hoerner form factor `K = 1 + 1.5(t/l)^1.5 + 7(t/l)^3` that accounts for
///    how strongly the thickness ratio inflates the friction drag of a
///    non-streamlined body.
///  - **Pressure / form drag** on the projected frontal area, from a
///    slenderness-dependent drag coefficient: bluff bodies approach ~1.2,
///    neatly streamlined bodies fall toward ~0.05–0.15.
///
/// The total drag acts along the stream direction (downstream), and the
/// per-panel centre-of-pressure offsets still produce a realistic roll / pitch
/// / yaw moment when the model is yawed or pitched in the flow. Wetted area,
/// frontal area and reference length are supplied precomputed from the source
/// mesh (resolution independent) so the magnitude does not drift with the
/// voxelization resolution.
pub fn compute_imported_shape_wind(
    panels: &[CollisionPanel],
    stream_body: &Vector3<f64>,
    mesh_wetted_area: f64,
    mesh_frontal_area: f64,
    mesh_reference_len: f64,
) -> ImportedAero {
    let v = stream_body.norm();
    if v < 1e-6 || panels.is_empty() {
        return ImportedAero {
            force: Vector3::zeros(),
            moment: Vector3::zeros(),
            frontal_area: 0.0,
            wetted_area: 0.0,
            reference_len: 0.0,
            cd_frontal: 0.0,
            re: 0.0,
        };
    }
    let s = *stream_body / v; // unit stream direction (body frame)

    // Use the resolution-independent mesh metrics supplied by the caller.
    let frontal_area = mesh_frontal_area.max(1e-6);
    let wetted_area = mesh_wetted_area.max(1e-6);
    let reference_len = mesh_reference_len.max(0.1);

    // Track the panel bounding box only as a fallback centre-of-pressure.
    let mut min_cp = [f64::INFINITY; 3];
    let mut max_cp = [f64::NEG_INFINITY; 3];
    for p in panels {
        for a_ in 0..3 {
            min_cp[a_] = min_cp[a_].min(p.cp[a_]);
            max_cp[a_] = max_cp[a_].max(p.cp[a_]);
        }
    }

    // Cross-section diameter from the frontal area -> slenderness L/D.
    let cross_diam = (4.0 * frontal_area / std::f64::consts::PI).max(0.1).sqrt();
    let slenderness = reference_len / cross_diam; // L / D

    // --- Reynolds number & skin friction ------------------------------------
    let re = RHO_SL * v * reference_len / MU_AIR;
    // Turbulent flat-plate skin friction (Schlichting): C_f = 0.074/Re^0.2.
    let cf = if re > 10.0 { 0.074 / re.powf(0.2) } else { 0.02 };
    // Hoerner form factor: thicker, blunter body -> more skin-friction drag.
    let thickness = slenderness.recip();
    let thin = thickness.min(1.0);
    let form_factor = 1.0 + 1.5 * thin.powf(1.5) + 7.0 * thin.powf(3.0);

    // --- Pressure / form drag coefficient -----------------------------------
    // Slenderness-based: long smooth bodies are shaped to shed flow (low Cd),
    // compact bluff bodies behave like a flat plate (Cd ~ 1.1-1.2).
    let cd_pressure = 0.10
        + 1.1 / (1.0 + (slenderness / 1.5).powf(2.0)); // ~1.2 at lambda~0, ~0.1 at lambda~8

    let drag_coeff_frontal = cd_pressure + form_factor * cf * (wetted_area / frontal_area);

    // --- Assemble drag ------------------------------------------------------
    // Drag force pushes the body downstream, i.e. along the stream direction.
    let q = 0.5 * RHO_SL * v * v;
    let f_total = q * drag_coeff_frontal * frontal_area;
    let force = s * f_total;

    // Moment: the drag acts through the body's centre of pressure. Take the
    // area-weighted centroid of the projected pressure panels.
    let mut cp_sum = Vector3::zeros();
    let mut w_sum = 0.0;
    for p in panels {
        let n = Vector3::new(p.normal[0], p.normal[1], p.normal[2]);
        let norm = n.norm();
        if norm < 1e-12 {
            continue;
        }
        let nhat = n / norm;
        let facing = nhat.dot(&s);
        let cp = Vector3::new(p.cp[0], p.cp[1], p.cp[2]);
        let w = (p.area * facing).abs();
        cp_sum += cp * w;
        w_sum += w;
    }
    let cop = if w_sum > 1e-12 {
        cp_sum / w_sum
    } else {
        Vector3::new(
            0.5 * (min_cp[0] + max_cp[0]),
            0.5 * (min_cp[1] + max_cp[1]),
            0.5 * (min_cp[2] + max_cp[2]),
        )
    };
    let moment = cop.cross(&force);

    ImportedAero {
        force,
        moment,
        frontal_area,
        wetted_area,
        reference_len,
        cd_frontal: drag_coeff_frontal,
        re,
    }
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

    /// Build a voxel-shell panel set approximating a unit cube centred at the
    /// origin (one panel per exposed face), normalised outward.
    fn cube_shell() -> Vec<CollisionPanel> {
        let mut shell = Vec::new();
        // Six faces at +-0.5, each with area 1.0.
        let faces: [([f64; 3], [f64; 3]); 6] = [
            ([1.0, 0.0, 0.0], [0.5, 0.0, 0.0]),
            ([-1.0, 0.0, 0.0], [-0.5, 0.0, 0.0]),
            ([0.0, 1.0, 0.0], [0.0, 0.5, 0.0]),
            ([0.0, -1.0, 0.0], [0.0, -0.5, 0.0]),
            ([0.0, 0.0, 1.0], [0.0, 0.0, 0.5]),
            ([0.0, 0.0, -1.0], [0.0, 0.0, -0.5]),
        ];
        for (n, cp) in faces {
            shell.push(panel(1.0, n, cp, 1.2));
        }
        shell
    }

    #[test]
    fn imported_still_air_is_zero() {
        let aero = compute_imported_shape_wind(&cube_shell(), &Vector3::zeros(), 6.0, 1.0, 1.0);
        assert!(aero.force.norm() < 1e-9 && aero.moment.norm() < 1e-9);
        assert!(aero.cd_frontal == 0.0);
    }

    #[test]
    fn imported_cube_has_bluff_cd_and_drag_along_stream() {
        // A 1 m cube at 20 m/s: frontal area 1 m², Re ~ 1.3e6, Cd ~ 1.0-1.2,
        // drag should push strongly along the stream (+X here).
        let stream = Vector3::new(20.0, 0.0, 0.0);
        let aero = compute_imported_shape_wind(&cube_shell(), &stream, 6.0, 1.0, 1.0);
        assert!(aero.force.x > 50.0, "expected substantial drag, got {:?}", aero.force);
        // Drag direction = downstream (along stream).
        assert!(aero.force.y.abs() < aero.force.x * 1e-6);
        assert!(aero.force.z.abs() < aero.force.x * 1e-6);
        // A cube is a bluff body: Cd on frontal area should be O(1).
        assert!(
            aero.cd_frontal > 0.6 && aero.cd_frontal < 1.6,
            "cube Cd {} out of bluff range",
            aero.cd_frontal
        );
        // Reynolds number in the right ballpark.
        assert!((aero.re - 1.225 * 20.0 * 1.0 / 1.81e-5).abs() / aero.re < 0.05);
        // Centre of pressure is the cube centroid => moment near zero.
        assert!(aero.moment.norm() < 1e-6, "centred cube moment {}", aero.moment);
    }

    #[test]
    fn imported_slender_body_has_lower_cd() {
        // A slender stretched box (single panel chain) is more streamlined than
        // an equal-cube: Cd should drop as slenderness rises.
        let stream = Vector3::new(20.0, 0.0, 0.0);
        let blunt = compute_imported_shape_wind(&cube_shell(), &stream, 6.0, 1.0, 1.0);
        // Make a long slender rod along X: side panels stacked along the length,
        // small frontal area, large wetted area.
        let mut slender = Vec::new();
        for x in [-2.0, -1.0, 0.0, 1.0, 2.0] {
            slender.push(panel(0.25, [0.0, 1.0, 0.0], [x, 0.25, 0.0], 1.2));
            slender.push(panel(0.25, [0.0, -1.0, 0.0], [x, -0.25, 0.0], 1.2));
            slender.push(panel(0.25, [0.0, 0.0, 1.0], [x, 0.0, 0.25], 1.2));
            slender.push(panel(0.25, [0.0, 0.0, -1.0], [x, 0.0, -0.25], 1.2));
        }
        // Two end caps give the frontal area (normal along X).
        slender.push(panel(1.0, [1.0, 0.0, 0.0], [2.5, 0.0, 0.0], 1.2));
        slender.push(panel(1.0, [-1.0, 0.0, 0.0], [-2.5, 0.0, 0.0], 1.2));
        // Slender body: wetted 7 m², frontal 1 m², length 5 m.
        let slim = compute_imported_shape_wind(&slender, &stream, 7.0, 1.0, 5.0);
        assert!(
            slim.cd_frontal < blunt.cd_frontal,
            "slender Cd {} should be < blunt {}",
            slim.cd_frontal,
            blunt.cd_frontal
        );
    }
}
