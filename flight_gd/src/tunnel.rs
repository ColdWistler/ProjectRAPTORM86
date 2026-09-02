//! Wind-tunnel simulation bridge — VLM-grade flow visualization.
//!
//! The aircraft is held fixed at the origin while a mesh of smoke-streak
//! particles flows over/around it.  The flow field is now driven by a proper
//! **Vortex Lattice Method** (VLM) that solves for the spanwise circulation
//! distribution on the wing and tail, trailed by semi-infinite vortex
//! filaments whose induced velocities are evaluated with **Biot-Savart**
//! integration (with a viscous-core regularisation).  Behind bluff bodies
//! (fuselage, nacelle) an empirical **Von Kármán vortex street** is shed.
//! Surface **pressure coefficients** (Cp) are available to GDScript for
//! colour-mapping the aircraft mesh.
//!
//! World conventions: nose = +X, up = +Y, right = +Z.
//! Free stream travels from +X toward −X (head-on).

use flight_core::aero::compute_forces_moments;
use flight_core::config::CollisionPanel;
use flight_core::nalgebra::Vector3 as NVec3;
use flight_core::shape::{compute_imported_shape_wind, compute_shape_wind, ImportedAero};
use flight_core::{AircraftConfig, AircraftState};
use godot::builtin::{
    PackedFloat32Array, PackedFloat64Array, PackedVector3Array, Transform3D, Vector3,
};
use godot::classes::Node3D;
use godot::prelude::*;

use crate::sim::origin_and_basis;
use crate::voxel::{mesh_metrics, voxelize_panels};

// ─────────────────────────────────────────────────────────────────────────────
// Constants
// ─────────────────────────────────────────────────────────────────────────────

// Smoke-rake geometry
const RAKE_HEIGHTS: [f32; 11] = [
    -2.2, -1.6, -1.0, -0.55, -0.2, 0.1, 0.32, 0.62, 1.0, 1.6, 2.2,
];
const RAKE_ROWS: usize = RAKE_HEIGHTS.len();
const RAKE_COLS: usize = 6;
const RAKE_HALF_WIDTH: f32 = 5.0;
const RAKE_X: f32 = 9.0;

const TOP_RAKE_ROWS: usize = 6;
const TOP_RAKE_COLS: usize = 5;
const TOP_RAKE_Y: f32 = 3.4;
const TOP_RAKE_X_MIN: f32 = 3.5;
const TOP_RAKE_X_MAX: f32 = -3.5;
const TOP_RAKE_HALF_WIDTH: f32 = 2.8;

const PARTICLE_COUNT: usize = 8000;
const PARTICLE_LIFE: f32 = 2.0;
const RAKE_JITTER: f32 = 0.16;
const PARTICLE_BASE_RADIUS: f32 = 0.07;
const PARTICLE_GROW: f32 = 1.0;
#[allow(dead_code)]
pub const TRAIL_TUBE_RADIUS: f32 = 0.045;
const CL_REF: f32 = 0.6;

// Aircraft geometry
const WING_HALF_SPAN: f32 = 5.5;
const FUSE_HALF_LEN: f32 = 2.8;

// Propeller
const PROP_X: f32 = -2.38;
const PROP_Y: f32 = 0.08;
const PROP_RADIUS: f32 = 0.75;
const PROP_AXIAL: f32 = 1.7;
const PROP_SWIRL: f32 = 1.15;
const PROP_DECAY: f32 = 7.0;
const PROP_INFLOW: f32 = 2.0;

// ─────────────────────────────────────────────────────────────────────────────
// Vortex Lattice Method
// ─────────────────────────────────────────────────────────────────────────────

/// Viscous core radius (metres) for the Biot-Savart kernel.  Prevents the
/// 1/r² singularity and gives a realistic vortex structure.
const VLM_CORE: f32 = 0.15;
/// How far downstream trailing vortex filaments extend (metres from trailing edge).
const VLM_WAKE_LEN: f32 = 40.0;
/// Number of spanwise panels on the main wing.
const VLM_N_SPAN: usize = 14;
/// Number of spanwise panels on each V-tail fin.
const VLM_TAIL_SPAN: usize = 6;

/// A straight vortex filament segment in3D space.
#[derive(Clone, Copy, Default)]
struct VortexSegment {
    a: Vector3,
    b: Vector3,
    gamma: f32,
}

/// Pressure coefficient result at a surface control point.
#[derive(Clone, Copy, Default)]
struct CpSample {
    pos: Vector3,
    normal: Vector3,
    cp: f32,
}

/// Compute velocity induced at `p` by a finite vortex segment (A→B) with
/// viscous-core regularisation.
fn biot_savart_seg(p: Vector3, seg: VortexSegment) -> Vector3 {
    let r1 = p - seg.a;
    let r2 = p - seg.b;
    let r1_len = r1.length();
    let r2_len = r2.length();
    if r1_len < 1e-6 || r2_len < 1e-6 {
        return Vector3::ZERO;
    }
    let dl = seg.b - seg.a;
    let dl_len = dl.length();
    if dl_len < 1e-6 {
        return Vector3::ZERO;
    }
    let dl_hat = dl / dl_len;
    let cross = dl_hat.cross(r1);
    let h2 = cross.length_squared() + VLM_CORE * VLM_CORE; // viscous core
    if h2 < 1e-12 {
        return Vector3::ZERO;
    }
    let cos1 = dl_hat.dot(r1) / r1_len;
    let cos2 = dl_hat.dot(r2) / r2_len;
    let coeff = seg.gamma / (4.0 * std::f32::consts::PI);
    coeff * cross / h2 * (cos1 - cos2)
}

/// Compute velocity induced at `p` by a semi-infinite vortex filament
/// starting at `a` and extending to infinity in direction `dir`.
fn biot_savart_semi(p: Vector3, a: Vector3, dir: Vector3, gamma: f32) -> Vector3 {
    let r = p - a;
    let r_len = r.length();
    if r_len < 1e-6 {
        return Vector3::ZERO;
    }
    let cross = dir.cross(r);
    let h2 = cross.length_squared() + VLM_CORE * VLM_CORE;
    if h2 < 1e-12 {
        return Vector3::ZERO;
    }
    let dir_dot_rhat = dir.dot(r) / r_len;
    let coeff = gamma / (4.0 * std::f32::consts::PI);
    coeff * cross / h2 * (1.0 + dir_dot_rhat)
}

/// VLM lifting surface: defines the panel geometry for one lifting surface
/// (wing or tail).  Panels use horseshoe vortex elements: a bound segment at
/// quarter-chord, two semi-infinite trailing legs to downstream infinity.
#[derive(Default)]
struct VlmSurface {
    /// Span stations (y-coordinates, body frame, sorted ascending).
    span: Vec<f32>,
    /// Chord at each span station.
    #[allow(dead_code)]
    chord: Vec<f32>,
    /// Quarter-chord x at each span station (typically -chord/4).
    qc_x: Vec<f32>,
    /// Three-quarter-chord x at each span station (control point).
    tc_x: Vec<f32>,
    /// Surface normal at each control point (body frame).
    normals: Vec<Vector3>,
    /// Normal freestream component at each control point (recomputed each solve).
    rhs: Vec<f32>,
    /// Solved circulation at each span station.
    gamma: Vec<f32>,
    /// Downstream direction for trailing vortices.
    wake_dir: Vector3,
    /// Number of panels = span.len() - 1.
    n_panels: usize,
    /// Position offset (for V-tail placement).
    offset: Vector3,
}

impl VlmSurface {
    /// Build a flat wing surface.  `y_stations` are the spanwise y-coords,
    /// `root_chord` and `tip_chord` linearly interpolate along the span.
    fn new(y_stations: &[f32], root_chord: f32, tip_chord: f32, offset: Vector3) -> Self {
        let n = y_stations.len();
        let span = y_stations.to_vec();
        let y0 = span[0];
        let y1 = span.last().copied().unwrap_or(y0);
        let span_width = (y1 - y0).abs().max(1e-3);

        let mut chord = Vec::with_capacity(n);
        let mut qc_x = Vec::with_capacity(n);
        let mut tc_x = Vec::with_capacity(n);
        let mut normals = Vec::with_capacity(n - 1);

        for &y in &span {
            let t = ((y - y0) / span_width).clamp(0.0, 1.0);
            let c = root_chord + (tip_chord - root_chord) * t;
            chord.push(c);
            qc_x.push(offset.x - c * 0.25);
            tc_x.push(offset.x - c * 0.75);
        }

        // Compute control-point normals (perpendicular to chord, tilted by
        // small geometric incidence — here just (0,1,0) for a flat plate).
        for i in 0..n - 1 {
            let y_mid = (span[i] + span[i + 1]) * 0.5;
            let _ = y_mid;
            normals.push(Vector3::new(0.0, 1.0, 0.0));
        }

        let n_panels = n - 1;
        Self {
            span,
            chord,
            qc_x,
            tc_x,
            normals,
            rhs: vec![0.0; n_panels],
            gamma: vec![0.0; n],
            wake_dir: Vector3::new(-1.0, 0.0, 0.0),
            n_panels,
            offset,
        }
    }

    /// Set up the V-tail: two cantilevered surfaces at ±38° cant.
    fn new_vtail(root_chord: f32, tip_chord: f32) -> Vec<Self> {
        let cant = 38.0_f32.to_radians();
        let n = VLM_TAIL_SPAN;
        let mut surfaces = Vec::with_capacity(2);

        for side in [-1.0_f32, 1.0] {
            let mut ys = Vec::with_capacity(n);
            for i in 0..n {
                let t = i as f32 / (n - 1) as f32;
                let spanwise = t * 1.8; // 1.8 m fin span
                let y = 0.65 + side * spanwise * cant.sin();
                let _z = side * (0.55 + spanwise * cant.cos());
                ys.push(y); // store the local y; we use z offset
            }
            let mut s = Self::new(&ys, root_chord, tip_chord, Vector3::new(-2.05, 0.65, side * 0.55));
            // Tilt normals by cant angle
            for n in &mut s.normals {
                *n = Vector3::new(0.0, cant.cos(), side * cant.sin());
            }
            surfaces.push(s);
        }
        surfaces
    }

    /// Solve for the circulation distribution given the freestream and current
    /// aircraft angle of attack.  Uses a simple vortex-lattice influence matrix.
    fn solve(&mut self, alpha: f32, speed: f32) {
        if self.n_panels == 0 || speed < 0.5 {
            self.gamma.iter_mut().for_each(|g| *g = 0.0);
            return;
        }

        let n = self.n_panels;
        // Build the AIC matrix and RHS
        let mut aic = vec![0.0f32; n * n];

        for j in 0..n {
            // Vortex panel j: bound segment from (qc_x[j], span[j]) to (qc_x[j], span[j+1])
            let va = Vector3::new(self.qc_x[j], self.span[j], self.offset.z);
            let vb = Vector3::new(self.qc_x[j], self.span[j + 1], self.offset.z);
            let bound = VortexSegment {
                a: va,
                b: vb,
                gamma: 1.0,
            };

            // Two trailing semi-infinite legs
            let trail_a = vb;
            let trail_b = va;

            for i in 0..n {
                let cp = Vector3::new(self.tc_x[i], (self.span[i] + self.span[i + 1]) * 0.5, self.offset.z);

                // Velocity induced by bound segment
                let v_bound = biot_savart_seg(cp, bound);

                // Velocity induced by trailing semi-infinite legs
                let v_trail_a = biot_savart_semi(cp, trail_a, self.wake_dir, 1.0);
                let v_trail_b = biot_savart_semi(cp, trail_b, self.wake_dir, -1.0);

                let v_total = v_bound + v_trail_a + v_trail_b;

                // Flow tangency: V_total · n = 0  →  AIC[i][j] = v_total · n
                aic[i * n + j] = v_total.dot(self.normals[i]);
            }
        }

        // RHS = -V_inf · n  (flow tangency: V_inf + V_induced) · n = 0
        // For a flat plate at the origin with freestream from +X:
        // The effective normal component is speed * sin(alpha).
        let v_freestream = Vector3::new(speed * alpha.cos(), speed * alpha.sin(), 0.0);

        for i in 0..n {
            self.rhs[i] = -v_freestream.dot(self.normals[i]);
        }

        // Solve Ax = b with Gaussian elimination
        solve_linear(&aic, &self.rhs, &mut self.gamma, n);
    }

    /// Compute the trailing vortex filaments from the solved circulation.
    /// Returns semi-infinite vortex segments (for Biot-Savart evaluation).
    fn trailing_filaments(&self, te_x: f32) -> Vec<VortexSegment> {
        let mut filaments = Vec::with_capacity(self.n_panels + 1);

        // Trailing vortex from each span station: strength = dΓ/dy
        for j in 0..=self.n_panels {
            let y = self.span[j];
            let gamma = if j == 0 {
                self.gamma[1] - self.gamma[0]
            } else if j == self.n_panels {
                self.gamma[j] - self.gamma[j - 1]
            } else {
                self.gamma[j + 1] - self.gamma[j - 1]
            };

            let strength = gamma * 0.5; // average of adjacent panels
            if strength.abs() > 0.01 {
                let a = Vector3::new(te_x, y, self.offset.z);
                // Semi-infinite trailing vortex: we store it as a segment from
                // a to a far-downstream point, but use biot_savart_semi for evaluation.
                let b = a + self.wake_dir * VLM_WAKE_LEN;
                filaments.push(VortexSegment {
                    a,
                    b,
                    gamma: strength,
                });
            }
        }

        filaments
    }

    /// Compute the bound vortex segments (for upwash visualization).
    fn bound_segments(&self) -> Vec<VortexSegment> {
        let mut segs = Vec::with_capacity(self.n_panels);
        for j in 0..self.n_panels {
            let a = Vector3::new(self.qc_x[j], self.span[j], self.offset.z);
            let b = Vector3::new(self.qc_x[j], self.span[j + 1], self.offset.z);
            let strength = self.gamma[j + 1] - self.gamma[j];
            if strength.abs() > 0.01 {
                segs.push(VortexSegment {
                    a,
                    b,
                    gamma: strength,
                });
            }
        }
        segs
    }
}

/// Solve a linear system Ax = b via Gaussian elimination with partial pivoting.
/// `a` is a flat n×n matrix, `b` is length-n, `x` is length-n output.
fn solve_linear(a: &[f32], b: &[f32], x: &mut [f32], n: usize) {
    if n == 0 {
        return;
    }
    // Augmented matrix
    let mut aug = vec![0.0f32; n * (n + 1)];
    for i in 0..n {
        for j in 0..n {
            aug[i * (n + 1) + j] = a[i * n + j];
        }
        aug[i * (n + 1) + n] = b[i];
    }

    // Forward elimination
    for col in 0..n {
        // Partial pivoting
        let mut max_val = aug[col * (n + 1) + col].abs();
        let mut max_row = col;
        for row in (col + 1)..n {
            let val = aug[row * (n + 1) + col].abs();
            if val > max_val {
                max_val = val;
                max_row = row;
            }
        }
        if max_row != col {
            for k in 0..=n {
                let tmp = aug[col * (n + 1) + k];
                aug[col * (n + 1) + k] = aug[max_row * (n + 1) + k];
                aug[max_row * (n + 1) + k] = tmp;
            }
        }

        let pivot = aug[col * (n + 1) + col];
        if pivot.abs() < 1e-12 {
            continue; // singular
        }

        for row in (col + 1)..n {
            let factor = aug[row * (n + 1) + col] / pivot;
            for k in col..=n {
                aug[row * (n + 1) + k] -= factor * aug[col * (n + 1) + k];
            }
        }
    }

    // Back substitution
    for i in (0..n).rev() {
        let mut sum = aug[i * (n + 1) + n];
        for j in (i + 1)..n {
            sum -= aug[i * (n + 1) + j] * x[j];
        }
        let diag = aug[i * (n + 1) + i];
        x[i] = if diag.abs() > 1e-12 {
            sum / diag
        } else {
            0.0
        };
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Von Kármán vortex shedding
// ─────────────────────────────────────────────────────────────────────────────

/// A shed vortex element in the Von Kármán street.
#[derive(Clone, Copy)]
struct ShedVortex {
    pos: Vector3,
    gamma: f32,
    age: f32,
    lifespan: f32,
}

/// Strouhal number for a cylinder (Re ~ 10³–10⁵).
const STROUHAL: f32 = 0.2;
/// Decay rate for shed vortices.
const SHED_DECAY: f32 = 0.4;
/// Maximum shed vortices tracked.
const MAX_SHED: usize = 40;

// ─────────────────────────────────────────────────────────────────────────────
// Flow grid
// ─────────────────────────────────────────────────────────────────────────────

struct FlowGrid {
    origin: Vector3,
    cell: Vector3,
    nx: usize,
    ny: usize,
    nz: usize,
    data: Vec<f32>,
}

impl Default for FlowGrid {
    fn default() -> Self {
        Self::new()
    }
}

impl FlowGrid {
    /// Base grid: X ∈ [-24, 14], Y/Z ∈ [-6.5, 6.5], 0.5m cells.
    fn new() -> Self {
        Self {
            origin: Vector3::new(-24.0, -6.5, -6.5),
            cell: Vector3::splat(0.5),
            nx: 77,
            ny: 27,
            nz: 27,
            data: Vec::new(),
        }
    }

    fn build(
        &mut self,
        wind: Vector3,
        speed: f32,
        _cl: f32,
        t_sec: f32,
        alpha: f32,
        axes: (Vector3, Vector3, Vector3),
        filaments: &[VortexSegment],
        shed: &[ShedVortex],
    ) {
        let (fwd, up, right) = axes;
        let (nx, ny, nz) = (self.nx, self.ny, self.nz);
        self.data.clear();
        self.data.reserve(nx * ny * nz * 3);

        for k in 0..nz {
            let wz = self.origin.z + k as f32 * self.cell.z;
            for j in 0..ny {
                let wy = self.origin.y + j as f32 * self.cell.y;
                for i in 0..nx {
                    let wx = self.origin.x + i as f32 * self.cell.x;
                    let local = Vector3::new(
                        wx * fwd.x + wy * fwd.y + wz * fwd.z,
                        wx * up.x + wy * up.y + wz * up.z,
                        wx * right.x + wy * right.y + wz * right.z,
                    );

                    // Base analytical flow perturbation (solid body, prop,
                    // plus stall separation driven by post-stall alpha)
                    let mut pert = body_flow(local, speed, t_sec, alpha);

                    // VLM trailing-vortex induced velocity
                    pert += filaments_induced(local, filaments);

                    // Von Kármán shed vortices
                    pert += shed_induced(local, shed);

                    let vel = wind + fwd * pert.x + up * pert.y + right * pert.z;
                    self.data.push(vel.x);
                    self.data.push(vel.y);
                    self.data.push(vel.z);
                }
            }
        }
    }

    fn sample(&self, p: Vector3, fallback: Vector3) -> Vector3 {
        let dx = (p.x - self.origin.x) / self.cell.x;
        let dy = (p.y - self.origin.y) / self.cell.y;
        let dz = (p.z - self.origin.z) / self.cell.z;
        let maxx = (self.nx - 2) as f32;
        let maxy = (self.ny - 2) as f32;
        let maxz = (self.nz - 2) as f32;
        if dx < 0.0 || dy < 0.0 || dz < 0.0 || dx > maxx || dy > maxy || dz > maxz {
            return fallback;
        }
        let i0 = dx as usize;
        let j0 = dy as usize;
        let k0 = dz as usize;
        let fx = dx - i0 as f32;
        let fy = dy - j0 as f32;
        let fz = dz - k0 as f32;
        let cell = |i: usize, j: usize, k: usize| {
            let b = (k * self.ny + j) * self.nx * 3 + i * 3;
            Vector3::new(self.data[b], self.data[b + 1], self.data[b + 2])
        };
        let lerp = |a: Vector3, b: Vector3, f: f32| a + (b - a) * f;
        let c00 = lerp(cell(i0, j0, k0), cell(i0 + 1, j0, k0), fx);
        let c10 = lerp(cell(i0, j0 + 1, k0), cell(i0 + 1, j0 + 1, k0), fx);
        let c01 = lerp(cell(i0, j0, k0 + 1), cell(i0 + 1, j0, k0 + 1), fx);
        let c11 = lerp(cell(i0, j0 + 1, k0 + 1), cell(i0 + 1, j0 + 1, k0 + 1), fx);
        lerp(lerp(c00, c10, fy), lerp(c01, c11, fy), fz)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// WindTunnelNode
// ─────────────────────────────────────────────────────────────────────────────

#[derive(GodotClass)]
#[class(base = Node3D, init)]
struct WindTunnelNode {
    wind_speed: f32,
    wind_direction: f32,
    pitch: f32,
    roll: f32,
    yaw: f32,
    aileron: f32,
    rudder: f32,
    elevator: f32,
    throttle: f64,
    throttle_split: f64,
    imported_panels: Vec<CollisionPanel>,
    imported_aero: ImportedAero,
    imported_wetted: f64,
    imported_frontal: f64,
    imported_len: f64,
    flaps_deg: f32,
    particles: Vec<Particle>,
    sources: Vec<Vector3>,
    emit_i: usize,
    grid: FlowGrid,
    elapsed: f32,
    force: NVec3<f64>,
    moment: NVec3<f64>,
    cl: f64,
    config: Option<AircraftConfig>,
    base: Base<Node3D>,

    // VLM state
    vlm_wing: VlmSurface,
    vlm_tail: Vec<VlmSurface>,
    vlm_filaments: Vec<VortexSegment>,
    vlm_bound: Vec<VortexSegment>,
    vlm_cp: Vec<CpSample>,

    // Von Kármán shedding
    shed_vortices: Vec<ShedVortex>,
    shed_timer: f32,
    shed_phase: f32,
}

#[derive(Clone, Copy)]
struct Particle {
    pos: Vector3,
    age: f32,
    seed: f32,
}

#[godot_api]
impl WindTunnelNode {
    #[func]
    fn step(&mut self, dt: f64) {
        if self.particles.is_empty() {
            self.refill_particles();
        }
        self.compute_aero();
        self.elapsed += dt as f32;

        // Solve VLM for current angle of attack
        let alpha = self.pitch;
        self.vlm_wing.solve(alpha, self.wind_speed);
        for tail in &mut self.vlm_tail {
            tail.solve(alpha, self.wind_speed);
        }

        // Build trailing vortex filaments
        self.vlm_filaments.clear();
        self.vlm_bound.clear();
        self.vlm_filaments.extend(self.vlm_wing.trailing_filaments(0.25));
        self.vlm_bound.extend(self.vlm_wing.bound_segments());
        for tail in &self.vlm_tail {
            self.vlm_filaments.extend(tail.trailing_filaments(-1.8));
            self.vlm_bound.extend(tail.bound_segments());
        }

        // Update Von Kármán shedding
        self.update_shedding(dt as f32);

        // Compute surface Cp
        self.compute_cp();

        let wind = self.wind_velocity() * self.wind_speed;
        let cl = (self.cl as f32 / CL_REF).clamp(-3.0, 3.0);
        self.grid.build(
            wind,
            self.wind_speed,
            cl,
            self.elapsed,
            alpha,
            self.attitude_axes(),
            &self.vlm_filaments,
            &self.shed_vortices,
        );
        self.advance_particles(dt as f32);
    }

    #[func]
    fn set_wind_speed(&mut self, ms: f32) {
        self.wind_speed = ms.clamp(1.0, 120.0);
    }

    #[func]
    fn set_wind_direction(&mut self, degrees: f32) {
        self.wind_direction = degrees.to_radians();
    }

    #[func]
    fn set_attitude(&mut self, pitch_deg: f32, roll_deg: f32, yaw_deg: f32) {
        self.pitch = pitch_deg.to_radians();
        self.roll = roll_deg.to_radians();
        self.yaw = yaw_deg.to_radians();
    }

    #[func]
    fn set_controls(&mut self, aileron_deg: f32, rudder_deg: f32, elevator_deg: f32) {
        self.aileron = aileron_deg.to_radians().clamp(-0.35, 0.35);
        self.rudder = rudder_deg.to_radians().clamp(-0.35, 0.35);
        self.elevator = elevator_deg.to_radians().clamp(-0.35, 0.35);
    }

    #[func]
    fn set_flaps_deg(&mut self, degrees: f32) {
        self.flaps_deg = degrees;
    }

    #[func]
    fn set_throttle(&mut self, throttle: f64) {
        self.throttle = throttle.clamp(0.0, 1.0);
    }

    #[func]
    fn set_throttle_split(&mut self, split: f64) {
        self.throttle_split = split.clamp(-1.0, 1.0);
    }

    #[func]
    fn set_imported_shape(
        &mut self,
        vertices: PackedVector3Array,
        indices: PackedInt32Array,
        resolution: i64,
    ) -> i64 {
        if vertices.is_empty() || indices.is_empty() {
            self.imported_panels = Vec::new();
            return 0;
        }
        let mut verts: Vec<[f64; 3]> = Vec::with_capacity(vertices.len() as usize);
        for i in 0..vertices.len() {
            let v = vertices[i];
            verts.push([v.x as f64, v.y as f64, v.z as f64]);
        }
        let mut tris: Vec<[usize; 3]> = Vec::with_capacity(indices.len() as usize / 3);
        let chunks = indices.len() as usize / 3;
        for c in 0..chunks {
            tris.push([
                indices[c * 3] as usize,
                indices[c * 3 + 1] as usize,
                indices[c * 3 + 2] as usize,
            ]);
        }
        let panels = voxelize_panels(&verts, &tris, resolution as usize);
        let m = mesh_metrics(&verts, &tris);
        self.imported_frontal = m.frontal_area;
        self.imported_wetted = m.wetted_area;
        self.imported_len = m.size_x;
        self.imported_panels = panels;
        self.imported_panels.len() as i64
    }

    #[func]
    fn is_imported_shape(&self) -> bool {
        !self.imported_panels.is_empty()
    }

    #[func]
    fn switch_aircraft(&mut self, name: GString) -> bool {
        let name = name.to_string();
        let file_name = format!("{name}.toml");
        let Some(config) = resolve_config_named(&file_name) else {
            godot_error!("WindTunnelNode: no config for aircraft '{name}'");
            return false;
        };
        self.config = Some(config);
        self.imported_panels = Vec::new();
        self.rebuild_vlm_surfaces();
        true
    }

    #[func]
    fn reset_trails(&mut self) {
        if self.particles.is_empty() {
            self.refill_particles();
        }
        let n = self.particles.len();
        for (i, p) in self.particles.iter_mut().enumerate() {
            p.age = (i as f32 / (n - 1) as f32) * PARTICLE_LIFE;
            let src = self.sources[i % self.sources.len()];
            p.pos = jittered(src, p.seed);
        }
    }

    #[func]
    fn particle_count(&self) -> i64 {
        self.particles.len() as i64
    }

    #[func]
    fn get_particles(&self) -> PackedFloat32Array {
        let mut out = Vec::with_capacity(self.particles.len() * 6);
        for p in &self.particles {
            let f = (p.age / PARTICLE_LIFE).clamp(0.0, 1.0);
            let r = PARTICLE_BASE_RADIUS * (1.0 + PARTICLE_GROW * f * f) * (1.0 - 0.75 * f * f * f);
            out.extend_from_slice(&[p.pos.x, p.pos.y, p.pos.z, r, f, p.seed]);
        }
        PackedFloat32Array::from(out)
    }

    #[func]
    fn get_drone_transform(&self) -> Transform3D {
        let state = self.attitude_state();
        origin_and_basis(&state)
    }

    #[func]
    fn get_aero(&self) -> PackedFloat64Array {
        if self.force.x == 0.0 && self.force.y == 0.0 && self.force.z == 0.0 {
            return PackedFloat64Array::from(vec![0.0; 7]);
        }
        PackedFloat64Array::from(vec![
            -self.force.z,
            -self.force.x,
            self.force.y,
            self.moment.x,
            self.moment.y,
            self.moment.z,
            self.cl,
        ])
    }

    #[func]
    fn get_aero_magnitudes(&self) -> PackedFloat64Array {
        if self.force.x == 0.0 && self.force.y == 0.0 && self.force.z == 0.0 {
            return PackedFloat64Array::from(vec![0.0; 4]);
        }
        PackedFloat64Array::from(vec![
            (-self.force.z).abs(),
            (-self.force.x).abs(),
            self.force.y.abs(),
            self.cl.abs(),
        ])
    }

    #[func]
    fn get_imported_aero(&self) -> PackedFloat64Array {
        PackedFloat64Array::from(vec![
            self.imported_aero.cd_frontal,
            self.imported_aero.re,
            self.imported_aero.frontal_area,
            self.imported_aero.wetted_area,
            self.imported_aero.reference_len,
        ])
    }

    #[func]
    fn get_settings(&self) -> PackedFloat64Array {
        PackedFloat64Array::from(vec![
            self.wind_speed as f64,
            self.wind_direction.to_degrees().rem_euclid(360.0) as f64,
            self.pitch.to_degrees() as f64,
            self.roll.to_degrees() as f64,
            self.yaw.to_degrees() as f64,
            self.aileron.to_degrees() as f64,
            self.rudder.to_degrees() as f64,
            self.elevator.to_degrees() as f64,
            self.flaps_deg as f64,
        ])
    }

    #[func]
    fn has_aero(&self) -> bool {
        self.force.x != 0.0 || self.force.y != 0.0 || self.force.z != 0.0
    }

    /// Surface Cp data: flattened `[x, y, z, nx, ny, nz, cp, ...]` for all
    /// wing control points.  GDScript can use this to colour-map the mesh.
    #[func]
    fn get_surface_cp(&self) -> PackedFloat32Array {
        let mut out = Vec::with_capacity(self.vlm_cp.len() * 7);
        for s in &self.vlm_cp {
            out.extend_from_slice(&[
                s.pos.x, s.pos.y, s.pos.z, s.normal.x, s.normal.y, s.normal.z, s.cp,
            ]);
        }
        PackedFloat32Array::from(out)
    }

    /// Number of surface Cp samples.
    #[func]
    fn cp_count(&self) -> i64 {
        self.vlm_cp.len() as i64
    }

    /// VLM circulation at each span station (for diagnostics).
    /// Returns flattened `[y0, gamma0, y1, gamma1, ...]`.
    #[func]
    fn get_vlm_gamma(&self) -> PackedFloat32Array {
        let mut out = Vec::with_capacity(self.vlm_wing.span.len() * 2);
        for (y, g) in self.vlm_wing.span.iter().zip(self.vlm_wing.gamma.iter()) {
            out.extend_from_slice(&[*y, *g]);
        }
        PackedFloat32Array::from(out)
    }
}

impl WindTunnelNode {
    fn build_layout(&mut self) {
        self.sources = Vec::with_capacity(RAKE_ROWS * RAKE_COLS + TOP_RAKE_ROWS * TOP_RAKE_COLS);
        for row in 0..RAKE_ROWS {
            for col in 0..RAKE_COLS {
                self.sources.push(rake_slot(row, col));
            }
        }
        for row in 0..TOP_RAKE_ROWS {
            for col in 0..TOP_RAKE_COLS {
                self.sources.push(top_rake_slot(row, col));
            }
        }
        self.particles = (0..PARTICLE_COUNT)
            .map(|i| {
                let seed = prng(i);
                Particle {
                    pos: Vector3::ZERO,
                    age: (i as f32 / (PARTICLE_COUNT - 1) as f32) * PARTICLE_LIFE,
                    seed,
                }
            })
            .collect();
        self.emit_i = 0;
    }

    fn refill_particles(&mut self) {
        if self.sources.is_empty() {
            self.build_layout();
        }
        for (i, p) in self.particles.iter_mut().enumerate() {
            let src = self.sources[i % self.sources.len()];
            p.seed = prng(i * 2 + 1);
            p.pos = jittered(src, p.seed);
            p.age = 0.0;
        }
    }

    fn advance_particles(&mut self, dt: f32) {
        let dt = dt.clamp(0.001, 0.1);
        let wind = self.wind_velocity() * self.wind_speed;
        let nozzle_count = self.sources.len().max(1);
        let mut emit = self.emit_i;
        for p in self.particles.iter_mut() {
            p.age += dt;
            if p.age >= PARTICLE_LIFE {
                let src = self.sources[emit % nozzle_count];
                emit += 1;
                p.pos = jittered(src, p.seed);
                p.age = 0.0;
                continue;
            }
            let v1 = self.grid.sample(p.pos, wind);
            let mid = p.pos + v1 * (dt * 0.5);
            let v2 = self.grid.sample(mid, wind);
            p.pos += v2 * dt;
        }
        self.emit_i = emit % nozzle_count;
    }

    fn wind_velocity(&self) -> Vector3 {
        let dir = self.wind_direction;
        Vector3::new(-dir.cos(), 0.0, dir.sin())
    }

    fn attitude_state(&self) -> AircraftState {
        let mut state = AircraftState::default();
        let (w, x, y, z) = euler_to_quat(self.roll as f64, self.pitch as f64, self.yaw as f64);
        state.q0 = w;
        state.q1 = x;
        state.q2 = y;
        state.q3 = z;
        let dir = self.wind_direction as f64;
        let world_flow = NVec3::new(-dir.cos(), 0.0, dir.sin());
        let speed = self.wind_speed as f64;
        let rel = state
            .rotation_earth_to_body()
            .transform_vector(&(world_flow * speed));
        state.u = rel.x;
        state.v = rel.y;
        state.w = rel.z;
        state
    }

    fn compute_aero(&mut self) {
        let dir = self.wind_direction as f64;
        let speed = self.wind_speed as f64;
        let flow_wind = NVec3::new(-dir.cos(), 0.0, dir.sin()) * speed;

        if !self.imported_panels.is_empty() {
            let state = self.attitude_state();
            let stream_body = state
                .rotation_earth_to_body()
                .transform_vector(&flow_wind);
            let aero = compute_imported_shape_wind(
                &self.imported_panels,
                &stream_body,
                self.imported_wetted,
                self.imported_frontal,
                self.imported_len,
            );
            self.force = aero.force;
            self.moment = aero.moment;
            self.cl = 0.0;
            self.imported_aero = aero;
            return;
        }

        if self.config.is_none() {
            self.config = resolve_config();
            if self.config.is_none() {
                return;
            }
        }
        let config = self.config.as_ref().unwrap();
        let state = self.attitude_state();
        let mut config = config.clone();
        config.throttle_split = self.throttle_split;

        let wind_earth = NVec3::zeros();
        let (mut forces, mut moments) = compute_forces_moments(
            &state,
            &config,
            self.elevator as f64,
            self.aileron as f64,
            self.rudder as f64,
            self.throttle,
            0.0,
            self.flaps_deg.to_radians() as f64,
            &wind_earth,
        );

        let (shape_force, shape_moment) =
            compute_shape_wind(&state, &config.collision_panels, &flow_wind);
        forces += shape_force;
        moments += shape_moment;

        self.force = forces;
        self.moment = moments;

        let lift = -forces.z;
        let tas = state.u.hypot(state.v).hypot(state.w);
        let q_dyn = 0.5 * 1.225 * tas * tas;
        self.cl = (lift / (q_dyn * config.wing_area.max(1.0))).clamp(-3.0, 3.0);
    }

    fn attitude_axes(&self) -> (Vector3, Vector3, Vector3) {
        let state = self.attitude_state();
        let (f, r, d) = state.body_axes_in_earth();
        let world = |v: NVec3<f64>| Vector3::new(v.x as f32, -v.z as f32, v.y as f32);
        let fwd = world(f).normalized();
        let up = world(-d).normalized();
        let right = world(r).normalized();
        (fwd, up, right)
    }

    /// Build the VLM surfaces for the current aircraft configuration.
    fn rebuild_vlm_surfaces(&mut self) {
        // Main wing: 14 spanwise stations from -WING_HALF_SPAN to +WING_HALF_SPAN
        let n = VLM_N_SPAN;
        let mut ys = Vec::with_capacity(n);
        for i in 0..n {
            let t = i as f32 / (n - 1) as f32;
            ys.push(-WING_HALF_SPAN + t * 2.0 * WING_HALF_SPAN);
        }
        let root_chord = 1.0;
        let tip_chord = 0.6;
        self.vlm_wing = VlmSurface::new(&ys, root_chord, tip_chord, Vector3::ZERO);

        // V-tail
        self.vlm_tail = VlmSurface::new_vtail(0.8, 0.4);

        // Initial solve
        self.vlm_wing.solve(self.pitch, self.wind_speed);
        for tail in &mut self.vlm_tail {
            tail.solve(self.pitch, self.wind_speed);
        }
        self.vlm_filaments = self.vlm_wing.trailing_filaments(0.25);
        self.vlm_bound = self.vlm_wing.bound_segments();
    }

    /// Update Von Kármán vortex shedding behind bluff bodies.
    fn update_shedding(&mut self, dt: f32) {
        if self.wind_speed < 1.0 {
            return;
        }

        // Shed from fuselage (x=-FUSE_HALF_LEN, y=0, z=0)
        self.shed_timer += dt;
        let freq = STROUHAL * self.wind_speed / 0.9; // D ≈ 0.9m (fuselage dia)
        let period = 1.0 / freq.max(0.1);

        if self.shed_timer >= period {
            self.shed_timer -= period;
            self.shed_phase += 1.0;

            let side = if (self.shed_phase as i32) % 2 == 0 { 1.0 } else { -1.0 };
            let pos = Vector3::new(-FUSE_HALF_LEN, side * 0.45, 0.0);
            let gamma = side * self.wind_speed * 0.4;
            self.shed_vortices.push(ShedVortex {
                pos,
                gamma,
                age: 0.0,
                lifespan: 3.0,
            });

            // Also shed from nacelle (x=-2.25)
            let pos2 = Vector3::new(-2.25, side * 0.35, 0.0);
            self.shed_vortices.push(ShedVortex {
                pos: pos2,
                gamma: gamma * 0.6,
                age: 0.0,
                lifespan: 2.5,
            });
        }

        // Age and remove dead vortices
        for v in &mut self.shed_vortices {
            v.age += dt;
            // Convect downstream
            v.pos.x -= self.wind_speed * dt * 0.7;
        }
        self.shed_vortices.retain(|v| v.age < v.lifespan);
        if self.shed_vortices.len() > MAX_SHED {
            let drain = self.shed_vortices.len() - MAX_SHED;
            self.shed_vortices.drain(..drain);
        }
    }

    /// Compute surface pressure coefficients at VLM control points.
    fn compute_cp(&mut self) {
        self.vlm_cp.clear();
        let speed = self.wind_speed;
        if speed < 0.5 {
            return;
        }

        // Wing control points
        for i in 0..self.vlm_wing.n_panels {
            let cp = Vector3::new(
                self.vlm_wing.tc_x[i],
                (self.vlm_wing.span[i] + self.vlm_wing.span[i + 1]) * 0.5,
                self.vlm_wing.offset.z,
            );

            // Velocity at control point: freestream + VLM induced
            let v_induced = filaments_induced(cp, &self.vlm_filaments);
            let v_local = Vector3::new(speed, 0.0, 0.0) + v_induced;
            let v_sq = v_local.length_squared();
            let v_inf_sq = speed * speed;

            // Bernoulli: Cp = 1 - (V_local / V_inf)²
            let cp_val = (1.0 - v_sq / v_inf_sq.max(1e-6)).clamp(-3.0, 2.0);

            self.vlm_cp.push(CpSample {
                pos: cp,
                normal: self.vlm_wing.normals[i],
                cp: cp_val,
            });
        }

        // Tail control points
        for tail in &self.vlm_tail {
            for i in 0..tail.n_panels {
                let cp = Vector3::new(
                    tail.tc_x[i],
                    (tail.span[i] + tail.span[i + 1]) * 0.5,
                    tail.offset.z,
                );
                let v_induced = filaments_induced(cp, &self.vlm_filaments);
                let v_local = Vector3::new(speed, 0.0, 0.0) + v_induced;
                let v_sq = v_local.length_squared();
                let v_inf_sq = speed * speed;
                let cp_val = (1.0 - v_sq / v_inf_sq.max(1e-6)).clamp(-3.0, 2.0);

                self.vlm_cp.push(CpSample {
                    pos: cp,
                    normal: tail.normals[i],
                    cp: cp_val,
                });
            }
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Induced velocity helpers
// ─────────────────────────────────────────────────────────────────────────────

/// Compute the total velocity induced at point `p` by all vortex filaments
/// using Biot-Savart.
fn filaments_induced(p: Vector3, filaments: &[VortexSegment]) -> Vector3 {
    let mut v = Vector3::ZERO;
    for &seg in filaments {
        let dl = seg.b - seg.a;
        let dl_len = dl.length();
        // Use semi-infinite formula for long trailing filaments, finite for short
        if dl_len > VLM_WAKE_LEN * 0.9 {
            let dir = (seg.b - seg.a).normalized();
            v += biot_savart_semi(p, seg.a, dir, seg.gamma);
        } else {
            v += biot_savart_seg(p, seg);
        }
        // Limit per-segment contribution for stability
        let max_seg = 300.0;
        if v.length() > max_seg {
            v = v.normalized() * max_seg;
        }
    }
    v
}

/// Compute velocity induced at `p` by all shed Von Kármán vortices.
fn shed_induced(p: Vector3, shed: &[ShedVortex]) -> Vector3 {
    let mut v = Vector3::ZERO;
    for sv in shed {
        let age_frac = (sv.age / sv.lifespan).clamp(0.0, 1.0);
        let strength = sv.gamma * (1.0 - age_frac * SHED_DECAY);
        let r = p - sv.pos;
        let r2 = r.length_squared() + VLM_CORE * VLM_CORE;
        if r2 < 1e-4 {
            continue;
        }
        // 2D point vortex: velocity = Γ/(2πr) perpendicular to r
        let perp = Vector3::new(-r.z, r.y, r.x); // rotate 90° in YZ
        v += strength / (2.0 * std::f32::consts::PI * r2) * perp;
    }
    v
}

// ─────────────────────────────────────────────────────────────────────────────
// Analytical body flow (kept as fallback / supplement to VLM)
// ─────────────────────────────────────────────────────────────────────────────

const STANDOFF: f32 = 0.12;
/// Post-stall angle of attack (degrees) — beyond this, separation is full.
const STALL_ALPHA_DEG: f32 = 15.0;
/// Separation starts this many degrees before full stall.
const SEPARATION_SPAN_DEG: f32 = 4.0;

/// Body-frame flow perturbation — solid body deflection, prop slipstream,
/// time-evolving turbulence.  The VLM handles circulation / downwash /
/// tip vortices separately.
fn body_flow(local: Vector3, speed: f32, t_sec: f32, alpha: f32) -> Vector3 {
    let (x, y, z) = (local.x, local.y, local.z);
    let mut v = Vector3::ZERO;

    // Stall separation factor: 0 (attached) → 1 (fully separated) as alpha
    // exceeds the stall onset.  Smooth ramp in `SEPARATION_SPAN_DEG`.
    let alpha_deg = alpha.to_degrees().abs();
    let sep_ramp = ((alpha_deg - (STALL_ALPHA_DEG - SEPARATION_SPAN_DEG)) / SEPARATION_SPAN_DEG)
        .clamp(0.0, 1.0);
    let sep_ramp = sep_ramp * sep_ramp * (3.0 - 2.0 * sep_ramp); // smoothstep

    // --- 1) Solid-body interaction ---
    let mut d_over_max = 0.0f32;
    let mut push = Vector3::ZERO;

    // Fuselage
    if x.abs() <= FUSE_HALF_LEN + STANDOFF {
        let rn = ((y / 0.45).powi(2) + (z / 0.40).powi(2)).sqrt().max(1e-4);
        let pen = 1.0 + STANDOFF / 0.45 - rn;
        if pen > 0.0 && pen > d_over_max {
            d_over_max = pen;
            push = Vector3::new(0.0, y / (0.45 * 0.45), z / (0.40 * 0.40)) / rn;
        }
    }

    // Canopy
    let canopy_x = x - 0.9;
    if y >= 0.0 {
        let er = ((canopy_x / 0.95).powi(2) + ((y - 0.40) / 0.28).powi(2) + (z / 0.35).powi(2))
            .sqrt()
            .max(1e-4);
        let pen = 1.0 + STANDOFF / 0.28 - er;
        if pen > 0.0 && pen > d_over_max {
            d_over_max = pen;
            push = Vector3::new(
                canopy_x / (0.95 * 0.95),
                (y - 0.40) / (0.28 * 0.28),
                z / (0.35 * 0.35),
            ) / er;
        }
    }

    // Main wing
    if x.abs() <= 1.0 && z.abs() <= WING_HALF_SPAN + 0.2 {
        let w_pen = 0.5 + STANDOFF - (x - 0.25).abs();
        let w_thick = 0.10 + STANDOFF - (y - 0.15).abs();
        if w_pen > 0.0 && w_thick > 0.0 {
            let pen = w_pen.min(w_thick);
            if pen > d_over_max {
                d_over_max = pen;
                if w_thick < w_pen {
                    push = Vector3::new(0.0, (y - 0.15).signum(), 0.0);
                } else {
                    push = Vector3::new(-(x - 0.25).signum(), 0.0, 0.0);
                }
            }
        }
    }

    // V-tail fins
    for zt in [-0.55f32, 0.55] {
        if x >= -2.7 && x <= -1.4 {
            let cant = 38.0_f32.to_radians();
            let s = (-zt).signum();
            let n_nearest = s * ((y - 0.65) * cant.sin() + (z - zt) * cant.cos());
            let along = (y - 0.65) * cant.cos() - (z - zt) * cant.sin();
            let fin_pen = 0.06 + STANDOFF - n_nearest.abs();
            if fin_pen > 0.0 && along.abs() <= 0.45 && fin_pen > d_over_max {
                d_over_max = fin_pen;
                push = Vector3::new(
                    0.0,
                    n_nearest.signum() * cant.sin(),
                    n_nearest.signum() * cant.cos(),
                );
            }
        }
    }

    // Nacelle
    let nac_x = x + 2.25;
    if nac_x.abs() <= 1.0 {
        let nr = (y * y + z * z).sqrt().max(1e-4);
        let nac_pen = 0.42 + STANDOFF - nr;
        if nac_pen > 0.0 && nac_pen > d_over_max {
            d_over_max = nac_pen;
            push = Vector3::new(0.0, y, z) / nr.max(1e-4);
        }
    }

    if d_over_max > 0.0 {
        let intensity = (d_over_max / (STANDOFF * 3.0)).clamp(0.0, 1.0);
        v += push * (speed * 2.6 * intensity);
        let swirl = Vector3::new(0.0, -push.z, push.y);
        v += (if swirl.length() > 1e-6 {
            swirl.normalized()
        } else {
            Vector3::ZERO
        }) * (speed * 1.2 * intensity);
        if x > FUSE_HALF_LEN * 0.5 && x <= FUSE_HALF_LEN + STANDOFF {
            v.x += speed * 0.9 * intensity;
        }
    }

    // --- Post-stall flow separation ---
    // When alpha exceeds stall, the flow over the upper surface separates:
    // a strong recirculation zone forms above and behind the wing, with a
    // large downwash/backflow replacing the attached downwash.  This is the
    // visually distinctive "stalled wing" signature.
    if sep_ramp > 0.01 {
        // Recirculation cell: centred above the wing trailing edge, extending
        // downstream; rotating backward on the upper surface.
        let cell_cx = 0.2;
        let cell_cy = 0.8;
        let cell_rx = 2.2;
        let cell_ry = 1.4;
        let dx = (x - cell_cx) / cell_rx;
        let dy = (y - cell_cy) / cell_ry;
        let r2 = dx * dx + dy * dy;
        if r2 < 1.0 {
            let strength = sep_ramp * speed;
            // Rotating cell: on the upper surface flow moves upstream (backward),
            // on the lower it moves downstream — a rolling rotor.
            v.x += strength * (-dy * 1.2);
            v.y += strength * (dx * 2.0) * sep_ramp;
        }

        // Separated wake: large turbulent, decelerated bubble behind the wing
        // with roll-off and random wobble.
        if x < 1.0 {
            let wd = (1.0 - x).max(0.0);
            let wake_fall = (-wd / 8.0).exp() * sep_ramp;
            let alt = ((y - 0.3) * (y - 0.3) + z * z).sqrt();
            let spread = (-(alt * alt) / (3.0 * 3.0)).exp();
            // Backflow + strong vertical mixing
            v.x -= speed * 0.9 * wake_fall * spread;
            let wob = (x * 1.6 + t_sec * 2.0).sin();
            v.y += speed * 0.55 * wake_fall * spread * wob;
            v.z += speed * 0.5 * wake_fall * spread * (z / (alt + 0.5));
        }

        // Turbulence intensity rises strongly in the separated region.
        let tu_sep = speed * 0.5 * sep_ramp * (-(x * x) / 9.0).exp();
        v.y += tu_sep * (x * 3.0 + t_sec * 4.0).sin() * (z * 2.2).sin();
        v.z += tu_sep * (z * 2.8 - t_sec * 5.0).sin() * (y * 2.4).sin();
    }

    // --- 2) Turbulent wake behind the solid ---
    if x < -1.0 {
        let wake_d = -x - 1.0;
        let wake_breadth = (wake_d / 12.0).clamp(0.0, 1.0);
        let alt = (y * y + z * z).sqrt();
        let spread = (-(alt * alt) / (3.4 * 3.4)).exp();
        let wobble = ((x * 2.3).sin() + (y * 3.7).sin() + (z * 1.9).sin()) * 0.5;
        v.y += speed * 0.8 * wake_breadth * spread * (y / (alt + 0.6)) * wobble;
        v.z += speed * 0.8 * wake_breadth * spread * (z / (alt + 0.6)) * wobble;
        let core = (-(alt * alt) / (2.0 * 2.0)).exp();
        v.x -= speed * 1.0 * wake_breadth * core;
    }

    // --- 3) Rear-pusher propeller slipstream ---
    let prx = x - PROP_X;
    let pry = y - PROP_Y;
    let pr = (pry * pry + z * z).sqrt();
    let core_radius = 0.30 * PROP_RADIUS;
    let core_r2 = core_radius * core_radius;
    let inflow_prox = (-(prx * prx) / (PROP_INFLOW * PROP_INFLOW)).exp();
    let aft = if prx < 0.0 {
        (-(prx * prx) / (PROP_DECAY * PROP_DECAY)).exp()
    } else {
        0.0
    };
    let slip_rad = PROP_RADIUS * (1.0 + 0.05 * prx.clamp(-PROP_DECAY, 0.0).abs());
    let disk = (-(pr * pr) / (slip_rad * slip_rad)).exp();
    if disk > 0.02 {
        v.x -= speed * PROP_AXIAL * disk * (0.25 * inflow_prox + aft);
        let rg = (2.0 * core_radius * pr) / (pr * pr + core_r2);
        let swirl_mag = speed * PROP_AXIAL * PROP_SWIRL * aft * disk * rg;
        let r_safe = pr.max(1e-3);
        v.y += swirl_mag * (-z / r_safe);
        v.z += swirl_mag * (pry / r_safe);
    }

    // --- 4) Lattice turbulence ---
    let tu = speed * 0.30;
    let ph = x * 0.55 + z * 0.7 + t_sec * 0.9;
    let ph2 = y * 0.8 + t_sec * 0.6;
    v.y += tu * ph.sin() * ph2.sin();
    v.z += tu * (x * 0.4 - t_sec * 0.8).sin() * (z * 0.9).sin();
    v.x += speed * 0.06 * (z * 1.4 + t_sec * 1.3).sin();

    let max_mag = speed * 1.9;
    if v.length() > max_mag {
        v = v.normalized() * max_mag;
    }
    v
}

// ─────────────────────────────────────────────────────────────────────────────
// Utility functions
// ─────────────────────────────────────────────────────────────────────────────

fn prng(i: usize) -> f32 {
    let x = i as f32 * 0.1031;
    let s = (x * 12.9898 + 78.233).sin() * 43758.5453;
    s - s.floor()
}

fn jittered(src: Vector3, seed: f32) -> Vector3 {
    let a1 = seed * 6.2831853;
    let a2 = (seed * 37.0).fract() * 6.2831853;
    let jy = RAKE_JITTER * a1.sin() * a2.cos();
    let jz = RAKE_JITTER * a1.cos() * a2.sin();
    src + Vector3::new(0.0, jy, jz)
}

fn resolve_config() -> Option<AircraftConfig> {
    for p in [
        "aircraft.toml",
        "../aircraft.toml",
        "../../aircraft.toml",
        "../../../aircraft.toml",
    ] {
        if let Ok(cfg) = AircraftConfig::from_file(p) {
            return Some(cfg);
        }
    }
    godot_error!("WindTunnelNode: aircraft.toml not found for aero forces");
    None
}

fn resolve_config_named(file_name: &str) -> Option<AircraftConfig> {
    for p in ["", "..", "../..", "../../.."] {
        let candidate = if p.is_empty() {
            file_name.to_string()
        } else {
            format!("{p}/{file_name}")
        };
        if let Ok(cfg) = AircraftConfig::from_file(&candidate) {
            return Some(cfg);
        }
    }
    None
}

fn rake_slot(row: usize, col: usize) -> Vector3 {
    let row = row.min(RAKE_HEIGHTS.len() - 1);
    let width_frac = if RAKE_COLS > 1 {
        col as f32 / (RAKE_COLS - 1) as f32
    } else {
        0.5
    };
    Vector3::new(
        RAKE_X,
        RAKE_HEIGHTS[row],
        (width_frac - 0.5) * (2.0 * RAKE_HALF_WIDTH),
    )
}

fn top_rake_slot(row: usize, col: usize) -> Vector3 {
    let fx = if TOP_RAKE_ROWS > 1 {
        row as f32 / (TOP_RAKE_ROWS - 1) as f32
    } else {
        0.5
    };
    let fz = if TOP_RAKE_COLS > 1 {
        col as f32 / (TOP_RAKE_COLS - 1) as f32
    } else {
        0.5
    };
    Vector3::new(
        TOP_RAKE_X_MIN + fx * (TOP_RAKE_X_MAX - TOP_RAKE_X_MIN),
        TOP_RAKE_Y,
        (fz - 0.5) * (2.0 * TOP_RAKE_HALF_WIDTH),
    )
}

fn euler_to_quat(roll: f64, pitch: f64, yaw: f64) -> (f64, f64, f64, f64) {
    let (sr, cr) = (roll * 0.5).sin_cos();
    let (sp, cp) = (pitch * 0.5).sin_cos();
    let (sy, cy) = (yaw * 0.5).sin_cos();
    let w = cr * cp * cy + sr * sp * sy;
    let x = sr * cp * cy - cr * sp * sy;
    let y = cr * sp * cy + sr * cp * sy;
    let z = cr * cp * sy - sr * sp * cy;
    (w, x, y, z)
}
