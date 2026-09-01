//! Wind-tunnel simulation bridge.
//!
//! The aircraft is held fixed at the origin while a mesh of smoke-streak
//! particles flows over/around it. Rust owns the *entire* flow field: the
//! solid-body deflection, wing circulation (driven by the real `flight_core`
//! lift coefficient), wing-tip vortices, turbulent wake, the top smoke rake
//! and the rear-pusher propeller slipstream. Godot merely renders the
//! streaks (e.g. a `MultiMeshInstance3D`) and reads aero forces for the HUD.
//!
//! World conventions match the old Bevy visualizer so the math ports 1:1:
//! the drone nose points along world +X, up is +Y, right is +Z, and the
//! head-on free stream travels from +X (front) toward -X (tail). The aircraft
//! model traced by the flow field uses the same +X-nose convention as
//! `FlightSimNode`.

use flight_core::aero::compute_forces_moments;
use flight_core::nalgebra::Vector3 as NVec3;
use flight_core::shape::compute_shape_wind;
use flight_core::{AircraftConfig, AircraftState};
use godot::builtin::{PackedFloat32Array, PackedFloat64Array, Transform3D, Vector3};
use godot::classes::Node3D;
use godot::prelude::*;

use crate::sim::origin_and_basis;

// --- Smoke-rake geometry ----------------------------------------------------
/// Front rake rows across height (Y). The rows deliberately ENVELOPE the whole
/// airframe -- from below the belly, through the fuselage/body heights (those
/// emit into the stagnation region and get deflected around the skin), up past
/// the canopy to well above -- so the smoke traces the real interaction on
/// every side of the drone, not just in front and above it.
const RAKE_HEIGHTS: [f32; 11] = [
    -2.2, -1.6, -1.0, -0.55, -0.2, 0.1, 0.32, 0.62, 1.0, 1.6, 2.2,
];
const RAKE_ROWS: usize = RAKE_HEIGHTS.len();
const RAKE_COLS: usize = 6;
/// Lateral spread of the rake around the drone centreline (metres, ±).
const RAKE_HALF_WIDTH: f32 = 5.0;
/// Distance of the rake ahead of the origin (metres; nose sits at +2.1).
const RAKE_X: f32 = 9.0;

// --- Top rake (above the airframe) ------------------------------------------
const TOP_RAKE_ROWS: usize = 6;
const TOP_RAKE_COLS: usize = 5;
const TOP_RAKE_Y: f32 = 3.4;
const TOP_RAKE_X_MIN: f32 = 3.5;
const TOP_RAKE_X_MAX: f32 = -3.5;
const TOP_RAKE_HALF_WIDTH: f32 = 2.8;

/// Number of free air "molecules" (small seed particles) tracing the flow.
const PARTICLE_COUNT: usize = 8000;
/// Lifespan of each particle (s); after this it recycles to a rake nozzle.
const PARTICLE_LIFE: f32 = 2.0;
/// Emission jitter (metres) around each rake nozzle so particles don't land in
/// a rigid grid -- they read as a molecular mist rather than a dotted lattice.
const RAKE_JITTER: f32 = 0.16;
/// Starting radius of a fresh particle (metres) — a small air molecule.
const PARTICLE_BASE_RADIUS: f32 = 0.07;
/// How much the particle swells as it ages (dissipation downstream).
const PARTICLE_GROW: f32 = 1.0;
/// Half-thickness of each trail link (metres). Consumed by the GDScript
/// renderer when building the smoke MultiMesh box links.
#[allow(dead_code)]
pub const TRAIL_TUBE_RADIUS: f32 = 0.045;
/// Reference lift coefficient the visual circulation is normalized against.
const CL_REF: f32 = 0.6;

// --- Solid-body geometry (matches the aircraft model) -----------------------
const WING_HALF_SPAN: f32 = 5.5;
const FUSE_HALF_LEN: f32 = 2.8;
const STANDOFF: f32 = 0.12;
const CIRC_DECAY: f32 = 3.2;
const CIRC_STRENGTH: f32 = 3.6;
const TIP_STRENGTH: f32 = 6.0;
const TIP_CORE: f32 = 1.0;
const TIP_GROW: f32 = 2.5;

// --- Rear-pusher propeller slipstream ---------------------------------------
const PROP_X: f32 = -2.38;
const PROP_Y: f32 = 0.08;
const PROP_RADIUS: f32 = 0.75;
const PROP_AXIAL: f32 = 1.7;
const PROP_SWIRL: f32 = 1.15;
const PROP_DECAY: f32 = 7.0;
const PROP_INFLOW: f32 = 2.0;

#[derive(GodotClass)]
#[class(base = Node3D, init)]
struct WindTunnelNode {
    /// Free-stream wind speed (m/s).
    wind_speed: f32,
    /// Free-stream direction in the XZ plane (radians; 0 = head-on along -X).
    wind_direction: f32,
    /// Aircraft attitude (radians).
    pitch: f32,
    roll: f32,
    yaw: f32,
    /// Control-surface deflections (radians).
    aileron: f32,
    rudder: f32,
    elevator: f32,
    /// Flap deflection (degrees).
    flaps_deg: f32,
    /// Free air "molecule" seeds: small particles advected through the flow
    /// grid from the front rakes toward the rear.
    particles: Vec<Particle>,
    /// Rake nozzles the particles are recycled through (front sheet + top rake).
    sources: Vec<Vector3>,
    /// Round-robin emission cursor into `sources`.
    emit_i: usize,
    /// Smooth, trilinear-interpolated velocity field the particles ride.
    grid: FlowGrid,
    /// Wall-clock seconds — drives the time-evolving turbulence so the air
    /// never freezes into a static picture.
    elapsed: f32,
    /// Last computed aero force/moment (body frame) and lift coefficient.
    force: NVec3<f64>,
    moment: NVec3<f64>,
    cl: f64,
    config: Option<AircraftConfig>,
    base: Base<Node3D>,
}

/// A single free air "molecule": a small seed particle tracing the flow.
#[derive(Clone, Copy)]
struct Particle {
    /// World position.
    pos: Vector3,
    /// Age in seconds (recycled to a rake nozzle at `PARTICLE_LIFE`).
    age: f32,
    /// Per-particle stable pseudo-random in [0,1); pins jitter + shader phase.
    seed: f32,
}

/// A precomputed world-space velocity field sampled with trilinear
/// interpolation. Built fresh every step from the current aero state, so the
/// smoke rides a SMOOTH, continuous air flow around the drone — no per-particle
/// analytic evaluation, no filament-shredding discontinuities.
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
    /// Domain in world metres: X in [-24, 14] (rake at +9, long trails to -22),
    /// Y/Z in [-6.5, 6.5], one metre cells.
    fn new() -> Self {
        Self {
            origin: Vector3::new(-24.0, -6.5, -6.5),
            cell: Vector3::ONE,
            nx: 39,
            ny: 14,
            nz: 14,
            data: Vec::new(),
        }
    }

    fn build(&mut self, wind: Vector3, speed: f32, cl: f32, t_sec: f32, axes: (Vector3, Vector3, Vector3)) {
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
                    let pert = body_flow(local, speed, cl, t_sec);
                    let vel = wind + fwd * pert.x + up * pert.y + right * pert.z;
                    self.data.push(vel.x);
                    self.data.push(vel.y);
                    self.data.push(vel.z);
                }
            }
        }
    }

    /// Trilinear sample. Returns `fallback` (usually the plain free stream)
    /// outside the gridded domain.
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

#[godot_api]
impl WindTunnelNode {
    /// Step everything forward by `dt` seconds: recompute the aero
    /// forces/moments for the current attitude, rebuild the smooth flow grid
    /// from that real aero state, and advect every air particle one step
    /// through it (recycling spent ones to the front rakes).
    #[func]
    fn step(&mut self, dt: f64) {
        if self.particles.is_empty() {
            self.refill_particles();
        }
        self.compute_aero();
        self.elapsed += dt as f32;
        let wind = self.wind_velocity() * self.wind_speed;
        let cl = (self.cl as f32 / CL_REF).clamp(-3.0, 3.0);
        self.grid.build(wind, self.wind_speed, cl, self.elapsed, self.attitude_axes());
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

    /// Set aircraft attitude; all angles in degrees.
    #[func]
    fn set_attitude(&mut self, pitch_deg: f32, roll_deg: f32, yaw_deg: f32) {
        self.pitch = pitch_deg.to_radians();
        self.roll = roll_deg.to_radians();
        self.yaw = yaw_deg.to_radians();
    }

    /// Set control-surface deflections (degrees).
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

    /// Switch the active aircraft configuration (e.g. `"MQI"` or
    /// `"TwinEngine"`), reloading the flow-grid aero forces + collision-shape
    /// wind forces for the new airframe. Returns `true` on success.
    #[func]
    fn switch_aircraft(&mut self, name: GString) -> bool {
        let name = name.to_string();
        let file_name = format!("{name}.toml");
        let Some(config) = resolve_config_named(&file_name) else {
            godot_error!("WindTunnelNode: no config for aircraft '{name}'");
            return false;
        };
        self.config = Some(config);
        true
    }

    /// Recycle every particle back to its rake nozzle so the chamber visibly
    /// floods fresh air from the front on a reset.
    #[func]
fn reset_trails(&mut self) {
    if self.particles.is_empty() {
        self.refill_particles();
    }
    // Spread ages evenly across the lifespan so the trail is continuous
    // and does not "burst" when the whole cohort recycles at once.
    let n = self.particles.len();
    for (i, p) in self.particles.iter_mut().enumerate() {
        p.age = (i as f32 / (n - 1) as f32) * PARTICLE_LIFE;
        let src = self.sources[i % self.sources.len()];
        p.pos = jittered(src, p.seed);
    }
}

    /// Number of air particles.
    #[func]
    fn particle_count(&self) -> i64 {
        self.particles.len() as i64
    }

    /// All particle data flattened: for each particle,
    /// `[x, y, z, radius, age_norm (0=new .. 1=old), seed]`.
    /// Sized `particle_count * 6`.
    #[func]
    fn get_particles(&self) -> PackedFloat32Array {
        let mut out = Vec::with_capacity(self.particles.len() * 6);
        for p in &self.particles {
            let f = (p.age / PARTICLE_LIFE).clamp(0.0, 1.0);
            // Blown up mid-flight, then shrinks again just before recycle so
            // the tail "evaporates" instead of piling up (no per-instance
            // alpha fade in the engine-native billboard material).
            let r = PARTICLE_BASE_RADIUS * (1.0 + PARTICLE_GROW * f * f) * (1.0 - 0.75 * f * f * f);
            out.extend_from_slice(&[p.pos.x, p.pos.y, p.pos.z, r, f, p.seed]);
        }
        PackedFloat32Array::from(out)
    }

    /// Drone transform for the fixed aircraft (attitude only, at the origin).
    #[func]
    fn get_drone_transform(&self) -> Transform3D {
        let state = self.attitude_state();
        origin_and_basis(&state)
    }

    /// Aero telemetry: `[lift N, drag N, side N, roll-moment N·m, pitch-moment
    /// N·m, yaw-moment N·m, lift coefficient cl]`. Body frame, as computed by
    /// `flight_core` for the current attitude / control deflections.
    /// * lift is positive upward (+Y), drag is positive rearward (‑X),
    ///   side is positive right (+Y in body cross‑wind).
    #[func]
    fn get_aero(&self) -> PackedFloat64Array {
        if self.force.x == 0.0 && self.force.y == 0.0 && self.force.z == 0.0 {
            // No physics computed yet; return zeros so the HUD is well-formed.
            return PackedFloat64Array::from(vec![0.0; 7]);
        }
        PackedFloat64Array::from(vec![
            -self.force.z, // lift (N)
            -self.force.x, // drag (N)
            self.force.y,  // side (N)
            self.moment.x,
            self.moment.y,
            self.moment.z,
            self.cl,
        ])
    }

    /// Aero telemetry magnitudes: `[|lift| N, |drag| N, |side| N, cl]`.
    /// Always returns positive values regardless of aircraft attitude,
    /// so HUD displays can show “lift = 500 N” without sign confusion.
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

    /// Current settings, all degrees: `[wind_speed m/s, wind dir °, pitch °,
    /// roll °, yaw °, aileron °, rudder °, elevator °, flaps °]`.
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

    /// True once enough physics have been computed to display aero data.
    #[func]
    fn has_aero(&self) -> bool {
        self.force.x != 0.0 || self.force.y != 0.0 || self.force.z != 0.0
    }
}

impl WindTunnelNode {
    /// Build the rake nozzle list (front sheet + top rake) and the particle pool.
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

    /// Bin the whole pool onto rake nozzles with jitter so the chamber is
    /// pre-filled with fresh air molecules ready to stream rearward.
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

    /// Advect every particle through the flow grid with a 2nd-order midpoint
    /// (RK2) stepper. Particles past their lifespan recycle to the next rake
    /// nozzle, so the stock of air molecules is continuously replenished at
    /// the FRONT and streams toward the rear.
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

    /// Free-stream unit direction in the tunnel XZ plane. Head-on from the
    /// nose (+X) means the airstream travels from +X toward -X.
    fn wind_velocity(&self) -> Vector3 {
        let dir = self.wind_direction;
        Vector3::new(-dir.cos(), 0.0, dir.sin())
    }

    /// `AircraftState` freeze-framed at the current attitude with the world
    /// airstream rotated into the body frame (same approach as the legacy
    /// tunnel aero computation — the aircraft is fixed, the air moves).
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

    /// Run `flight_core`'s aero model for the current attitude and stash the
    /// forces + lift coefficient used by both the HUD and the flow field.
    fn compute_aero(&mut self) {
        if self.config.is_none() {
            self.config = resolve_config();
            if self.config.is_none() {
                return;
            }
        }
        let config = self.config.as_ref().unwrap();
        let state = self.attitude_state();

        let wind_earth = NVec3::zeros();
        let (mut forces, mut moments) = compute_forces_moments(
            &state,
            config,
            self.elevator as f64,
            self.aileron as f64,
            self.rudder as f64,
            0.0,
            0.0,
            self.flaps_deg.to_radians() as f64,
            &wind_earth,
        );

        // Collision-shape wind interaction: the tunnel flow pours onto the
        // aircraft's flat-plate panels, adding a geometry-dependent force.
        let dir = self.wind_direction as f64;
        let speed = self.wind_speed as f64;
        let flow_wind = NVec3::new(-dir.cos(), 0.0, dir.sin()) * speed;
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

    /// The drone's world-space `(forward, up, right)` axes for the current
    /// attitude, from `flight_core`'s body→Earth axes (NED mapped to Godot Y-up).
    fn attitude_axes(&self) -> (Vector3, Vector3, Vector3) {
        let state = self.attitude_state();
        let (f, r, d) = state.body_axes_in_earth();
        let world = |v: NVec3<f64>| Vector3::new(v.x as f32, -v.z as f32, v.y as f32);
        let fwd = world(f).normalized();
        let up = world(-d).normalized();
        let right = world(r).normalized();
        (fwd, up, right)
    }
}

/// Deterministic pseudo-random hash for a non-negative index → [0,1).
fn prng(i: usize) -> f32 {
    let x = i as f32 * 0.1031;
    let s = (x * 12.9898 + 78.233).sin() * 43758.5453;
    s - s.floor()
}

/// A rake nozzle position jittered in the YZ plane by a particle's seed.
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

/// Resolve a named aircraft config (e.g. `"MQI.toml"`) from the same candidate
/// directories as [`resolve_config`].
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

/// Smoke-rake nozzle position (world space), a vertical sheet ahead of the nose.
/// Row heights are the fixed near-body heights so streams graze the airframe.
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

/// Top-rake nozzle position: a horizontal sheet of sources above the airframe.
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

/// Euler (roll, pitch, yaw), ZYX intrinsic → quaternion `(w, x, y, z)` in the
/// `AircraftState` field order.
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

/// Body-frame flow-perturbation velocity (m/s) at a local position. The
/// drone's nose is +X and the airstream comes from the front (upstream +X,
/// downstream -X). `cl` (lift factor) scales circulation, tip vortices and
/// wake with the *actual* physics lift coefficient from `flight_core`. The
/// function is SMOOTH (everything is a sine/exp/smoothstep ramp) so streaks
/// stay continuous, and it ends with a light time-evolving turbulence term
/// that guarantees the air keeps animating even in perfectly uniform flow.
fn body_flow(local: Vector3, speed: f32, cl: f32, t_sec: f32) -> Vector3 {
    let (x, y, z) = (local.x, local.y, local.z);
    let mut v = Vector3::ZERO;

    // --- 1) Solid-body interaction -----------------------------------------
    let span_pos = (z / WING_HALF_SPAN).clamp(-1.0, 1.0);
    let span_taper = (1.0 - span_pos.abs()).clamp(0.0, 1.0);
    let span_taper = span_taper * span_taper;

    let mut d_over_max = 0.0f32;
    let mut push = Vector3::ZERO;

    // (a) Fuselage: elliptical cylinder on the X axis.
    if x.abs() <= FUSE_HALF_LEN + STANDOFF {
        let rn = ((y / 0.45).powi(2) + (z / 0.40).powi(2)).sqrt().max(1e-4);
        let pen = 1.0 + STANDOFF / 0.45 - rn;
        if pen > 0.0 && pen > d_over_max {
            d_over_max = pen;
            push = Vector3::new(0.0, y / (0.45 * 0.45), z / (0.40 * 0.40)) / rn;
        }
    }

    // Canopy hump: flattened ellipsoid on top of the fuselage at x≈0.9.
    let canopy_x = x - 0.9;
    if y >= 0.0 {
        let er = ((canopy_x / 0.95).powi(2) + ((y - 0.40) / 0.28).powi(2) + (z / 0.35).powi(2))
            .sqrt()
            .max(1e-4);
        let pen = 1.0 + STANDOFF / 0.28 - er;
        if pen > 0.0 && pen > d_over_max {
            d_over_max = pen;
            push = Vector3::new(canopy_x / (0.95 * 0.95), (y - 0.40) / (0.28 * 0.28), z / (0.35 * 0.35)) / er;
        }
    }

    // (b) Main wing: thin slab spanning the span at y≈0.15, chord centred x=0.25.
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

    // (c) V-tail fins (aft, canted ~38°): thin canted slabs at z≈±0.55.
    for zt in [-0.55f32, 0.55] {
        if x >= -2.7 && x <= -1.4 {
            let cant = 38.0_f32.to_radians();
            let s = (-zt).signum();
            let n_nearest = s * ((y - 0.65) * cant.sin() + (z - zt) * cant.cos());
            let along = (y - 0.65) * cant.cos() - (z - zt) * cant.sin();
            let fin_pen = 0.06 + STANDOFF - n_nearest.abs();
            if fin_pen > 0.0 && along.abs() <= 0.45 && fin_pen > d_over_max {
                d_over_max = fin_pen;
                push = Vector3::new(0.0, n_nearest.signum() * cant.sin(), n_nearest.signum() * cant.cos());
            }
        }
    }

    // (d) Rear engine nacelle: blunt cylinder at the tail (x≈-2.25).
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
        v += (if swirl.length() > 1e-6 { swirl.normalized() } else { Vector3::ZERO }) * (speed * 1.2 * intensity);
        if x > FUSE_HALF_LEN * 0.5 && x <= FUSE_HALF_LEN + STANDOFF {
            v.x += speed * 0.9 * intensity;
        }
    }

    // --- 2) Turbulent wake behind the solid ----------------------------------
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

    // --- 3) Wing circulation (lift): upwash ahead, downwash behind ----------
    let r_wing = (x * x + y * y).max(TIP_CORE * TIP_CORE).sqrt();
    let circ_decay = (-((r_wing - TIP_CORE).powi(2)) / (CIRC_DECAY * CIRC_DECAY)).exp();
    let w_circ = CIRC_STRENGTH * speed * cl * (x / r_wing) * circ_decay * span_taper;
    v.y += w_circ;

    // --- 4) Trailing wingtip vortices ---------------------------------------
    for (zt, s) in [(WING_HALF_SPAN, -1.0), (-WING_HALF_SPAN, 1.0)] {
        let aft = 0.5 - 0.5 * ((x + 1.2) / TIP_GROW).clamp(-1.0, 1.0);
        let ry = y;
        let rz = z - zt;
        let r2 = ry * ry + rz * rz + TIP_CORE * TIP_CORE;
        let k = TIP_STRENGTH * speed * cl * s * aft / r2;
        v.y += k * (-rz);
        v.z += k * ry;
    }

    // --- 5) Rear-pusher propeller slipstream --------------------------------
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

    // --- 6) Lattice turbulence (time-evolving, so the air never freezes) -----
    let tu = speed * 0.30;
    let ph = x * 0.55 + z * 0.7 + t_sec * 0.9;
    let ph2 = y * 0.8 + t_sec * 0.6;
    v.y += tu * ph.sin() * ph2.sin();
    v.z += tu * (x * 0.4 - t_sec * 0.8).sin() * (z * 0.9).sin();
    v.x += speed * 0.06 * (z * 1.4 + t_sec * 1.3).sin();

    // Cap total perturbation so strong shear never blows filaments apart.
    let max_mag = speed * 1.9;
    if v.length() > max_mag {
        v = v.normalized() * max_mag;
    }
    v
}