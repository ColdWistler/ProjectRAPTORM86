//! Wind-tunnel visualization mode.
//!
//! Holds the aircraft fixed in place while a field of streak particles (wind
//! streamlines) flows over and around it, showing how the air deflects around
//! the airframe. The user can rotate the aircraft (pitch/roll/yaw) and adjust
//! the wind speed and direction without moving the model, and the HUD reports
//! the aerodynamic forces/moments computed by `flight_core` for the current
//! orientation.

use bevy::input::mouse::{AccumulatedMouseMotion, AccumulatedMouseScroll};
use bevy::prelude::*;
use bevy::render::mesh::{Indices, PrimitiveTopology};
use bevy::render::primitives::Aabb;
use flight_core::nalgebra::Vector3;

use crate::aircraft::{spawn_aircraft, AircraftRoot};
use crate::AppState;

/// Radius (world metres) around the drone within which the full flow field is
/// sampled. Outside this region trails just ride the uniform free stream, so
/// we skip the (rotated-frame) field math entirely for distant points.
const FLOW_REGION: f32 = 22.0;

// --- Smoke-rake geometry --------------------------------------------------
// Real wind-tunnel testing injects thin smoke filaments from a small rake of
// nozzles ahead of the model. The air "trails" are exactly that: a compact
// grid of sources in front of the drone, each leaving a long, thin filament
// that bends over/around the airframe and trails far downstream. This keeps
// every generated element in the region that matters (next to the model)
// instead of scattering dots across the whole chamber.
/// Number of rake rows across height (Y).
const RAKE_ROWS: usize = 5;
/// Number of rake columns across width (Z).
const RAKE_COLS: usize = 6;
/// Vertical spread of the rake around the drone centreline (metres, ±). Sized
/// to bracket the actual airframe height envelope (~y = -0.5..+1.8) so the
/// middle rows pass directly over/around the body and visibly deflect.
const RAKE_HALF_HEIGHT: f32 = 2.0;
/// Lateral spread of the rake (metres, ±).
const RAKE_HALF_WIDTH: f32 = 5.0;
/// Distance of the rake ahead of the origin (metres; nose sits at +2.1).
const RAKE_X: f32 = 9.0;

// --- Top rake (above the airframe) ---------------------------------------
// A second smoke rake hangs above the model like smoke wands lowered from the
// tunnel ceiling. Those filaments sink down toward the aircraft as they are
// blown downstream, so the smoke visibly curtains over the canopy, wing tops
// and down through the pusher propeller.
/// Number of top-rake rows along the stream (X).
const TOP_RAKE_ROWS: usize = 6;
/// Number of top-rake columns across the stream (Z).
const TOP_RAKE_COLS: usize = 5;
/// Height of the top rake above the origin (metres).
const TOP_RAKE_Y: f32 = 4.5;
/// Downstream (X) extent of the top rake, from ahead of the nose to over the
/// tail, so smoke falls all along the airframe.
const TOP_RAKE_X_MIN: f32 = 3.5;
const TOP_RAKE_X_MAX: f32 = -3.5;
/// Lateral spread of the top rake (metres, ±).
const TOP_RAKE_HALF_WIDTH: f32 = 2.8;
/// How strongly top-rake filaments sink toward the airframe (fraction of the
/// free-stream speed, applied until they reach the aircraft height band).
const TOP_DESCENT: f32 = 0.55;
/// Number of rigid segments (thin tube links) that make up each trail. The
/// total trail length is roughly `TRAIL_SEGS * mean flow step` and extends
/// from the rake (x = +9) well past the drone's tail into the turbulent wake,
/// so the interaction with the airframe is fully visible.
const TRAIL_SEGS: usize = 56;
/// Half-thickness of each trail link (metres). Kept thin so the smoke reads as
/// fine filament lines and the aircraft stays clearly visible.
const TRAIL_TUBE_RADIUS: f32 = 0.045;
/// Reference lift coefficient the visual circulation is normalised against, so
/// the flow field matches the sign/magnitude of the main-flight-sim physics.
const CL_REF: f32 = 0.6;

/// User-adjustable wind-tunnel parameters.
#[derive(Resource)]
pub struct WindTunnelSettings {
    /// Wind speed (m/s).
    pub wind_speed: f32,
    /// Wind direction in the tunnel XZ plane, radians (0 = straight along +X).
    pub wind_direction: f32,
    /// Aircraft pitch (nose up/down), radians.
    pub pitch: f32,
    /// Aircraft roll, radians.
    pub roll: f32,
    /// Aircraft yaw, radians.
    pub yaw: f32,
    /// Aileron deflection for visualizing control force, radians.
    pub aileron: f32,
    /// Rudder deflection, radians.
    pub rudder: f32,
    /// Elevator deflection, radians.
    pub elevator: f32,
    /// Flap deflection in degrees.
    pub flaps_deg: f32,
}

impl Default for WindTunnelSettings {
    fn default() -> Self {
        Self {
            wind_speed: 20.0,
            wind_direction: 0.0,
            pitch: 4.0_f32.to_radians(),
            roll: 0.0,
            yaw: 0.0,
            aileron: 0.0,
            rudder: 0.0,
            elevator: 0.0,
            flaps_deg: 0.0,
        }
    }
}

/// One smoke filament born from a rake. Holds the streakline: the positions
/// of the air parcels born over the last few frames, in order from newest
/// (at the rake, upstream) to oldest (far downstream). All trails are baked
/// into a single combined mesh each frame (see `build_trail_mesh`), so there
/// is exactly one draw call for the entire smoke field.
#[derive(Component)]
struct TrailEmitter {
    /// Fixed nozzle position where a fresh smoke parcel is injected each frame.
    source: Vec3,
    /// Whether this filament comes from the top rake (above the airframe). Top
    /// filaments sink toward the aircraft as they travel, so the smoke visibly
    /// descends onto the canopy, wing tops and out through the propeller.
    top: bool,
    /// Streakline sample points, `points[0]` = newest (at the rake).
    points: Vec<Vec3>,
}

/// The single mesh + material entity that renders all smoke trails together.
#[derive(Component)]
struct TrailVisual {
    /// Handle to the per-frame rebuilt combined trail mesh.
    mesh: Handle<Mesh>,
}

/// Marker for the wind-tunnel HUD text.
#[derive(Component)]
struct TunnelHudText;

/// Computed aerodynamic force/moment for display. NED/body-frame as returned
/// by `flight_core`.
#[derive(Resource, Default)]
struct Aerodynamics {
    force: Vector3<f64>,
    moment: Vector3<f64>,
    /// Dimensionless lift coefficient from the main-flight-sim aero model,
    /// used to make the visual flow field follow the physics (e.g. pitch
    /// changes upwash/downwash strength exactly like real lift does).
    cl: f64,
}

/// Spherical-orbit camera state for the wind tunnel. The camera orbits the
/// fixed aircraft at the origin; the user drags with the mouse to rotate and
/// scrolls to zoom.
#[derive(Resource)]
struct TunnelCameraState {
    /// Orbit yaw around the drone (radians).
    yaw: f32,
    /// Orbit pitch above the drone (radians).
    pitch: f32,
    /// Distance from the origin (zoom).
    distance: f32,
}

impl Default for TunnelCameraState {
    fn default() -> Self {
        Self {
            yaw: -0.6,
            pitch: 0.35,
            distance: 24.0,
        }
    }
}

/// Plugin that only adds wind-tunnel systems (state-scoped setup/cleanup).
pub struct WindTunnelPlugin;

impl Plugin for WindTunnelPlugin {
    fn build(&self, app: &mut App) {
        app.add_systems(OnEnter(AppState::WindTunnel), tunnel_setup)
            .add_systems(OnEnter(AppState::WindTunnel), tunnel_camera_setup)
            .add_systems(OnExit(AppState::WindTunnel), tunnel_cleanup)
            .add_systems(
                PreUpdate,
                (tunnel_input, tunnel_update_aircraft)
                    .chain()
                    .run_if(in_state(AppState::WindTunnel)),
            )
            .add_systems(
                Update,
                (
                    (tunnel_compute_aero, advance_flow).chain(),
                    tunnel_camera_orbit,
                    tunnel_update_hud,
                )
                    .run_if(in_state(AppState::WindTunnel)),
            );
    }
}

/// Spawn the fixed aircraft, the streak-particle wind field, and the HUD.
fn tunnel_setup(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
    asset_server: Res<AssetServer>,
) {
    commands.insert_resource(WindTunnelSettings::default());
    commands.insert_resource(Aerodynamics::default());
    commands.insert_resource(TunnelCameraState::default());

    // --- The aircraft, held fixed at the tunnel origin. ---
    spawn_aircraft(&mut commands, &mut meshes, &mut materials, &asset_server);

    // --- Smoke trails (the moving wind). A compact rake of sources sits ahead
    // of the nose AND a second rake hangs above the airframe; each source
    // leaves a long, thin filament that bends over and around the airframe and
    // trails downstream — like the smoke wands used in real wind-tunnel flow
    // testing. The top rake's filaments sink onto the aircraft as they travel,
    // so the smoke visibly curtains down over the canopy, wings and propeller.
    //
    // All filaments are baked into ONE combined mesh (single draw call, GPU
    // instanced geometry) rather than thousands of separate entities. Only the
    // aerodynamically-visible regions cost anything.
    let trail_mat = materials.add(StandardMaterial {
        base_color: Color::srgb(0.85, 1.0, 1.0),
        emissive: LinearRgba::new(1.2, 1.9, 2.1, 1.0),
        unlit: true,
        // Double-sided so the thin square-stick linking segments always render.
        cull_mode: None,
        alpha_mode: AlphaMode::Blend,
        ..default()
    });

    // Downstream unit direction of the free stream, used to pre-build each
    // trail as a straight filament so it is fully visible on the very first
    // frame (the streakline then bends as it advects through the flow field).
    let flow_dir = wind_velocity(&WindTunnelSettings::default()).normalize();
    let init_step = WindTunnelSettings::default().wind_speed * (1.0 / 60.0);

    let mut emitters = Vec::with_capacity(RAKE_ROWS * RAKE_COLS + TOP_RAKE_ROWS * TOP_RAKE_COLS);
    let mut chains = Vec::with_capacity(RAKE_ROWS * RAKE_COLS + TOP_RAKE_ROWS * TOP_RAKE_COLS);

    // Front rake: a vertical fan of smoke ahead of the nose.
    for row in 0..RAKE_ROWS {
        for col in 0..RAKE_COLS {
            let source = rake_slot(row, col);
            // points[0] is pinned at the rake (upstream); the trail extends
            // downstream toward the drone.
            let points = (0..=TRAIL_SEGS)
                .map(|i| source + flow_dir * (init_step * i as f32))
                .collect::<Vec<_>>();
            chains.push(points.clone());
            emitters.push(TrailEmitter {
                source,
                top: false,
                points,
            });
        }
    }

    // Top rake: smoke wands hanging above the airframe.
    for row in 0..TOP_RAKE_ROWS {
        for col in 0..TOP_RAKE_COLS {
            let source = top_rake_slot(row, col);
            let points = (0..=TRAIL_SEGS)
                .map(|i| source + flow_dir * (init_step * i as f32))
                .collect::<Vec<_>>();
            chains.push(points.clone());
            emitters.push(TrailEmitter {
                source,
                top: true,
                points,
            });
        }
    }

    // Combined trail mesh (fixed large bounds so frustum culling never drops
    // the rebuilt per-frame mesh).
    let trail_mesh = meshes.add(build_trail_mesh(&chains));
    let bounds = Aabb::from_min_max(
        Vec3::new(-FLOW_REGION, -FLOW_REGION, -FLOW_REGION),
        Vec3::new(FLOW_REGION, FLOW_REGION, FLOW_REGION),
    );
    commands.spawn((
        TrailVisual {
            mesh: trail_mesh.clone(),
        },
        Mesh3d(trail_mesh),
        MeshMaterial3d(trail_mat),
        // Conservative culling bounds; the mesh itself is rewritten each frame
        // inside this box, but the entity `Aabb` is never updated automatically.
        bounds,
        Transform::default(),
    ));
    for em in emitters {
        commands.spawn((em,));
    }

    // --- Wind-tunnel HUD ---
    commands
        .spawn((
            TunnelHudText,
            Node {
                position_type: PositionType::Absolute,
                top: Val::Px(16.0),
                left: Val::Px(16.0),
                padding: UiRect::all(Val::Px(14.0)),
                flex_direction: FlexDirection::Column,
                ..default()
            },
            BackgroundColor(Color::srgba(0.03, 0.06, 0.10, 0.86)),
        ))
        .with_children(|parent| {
            parent.spawn((
                Text::new("WIND TUNNEL"),
                TextFont {
                    font_size: 15.0,
                    ..default()
                },
                TextColor(Color::srgb(0.95, 0.96, 0.98)),
            ));
        });
}

/// A dark, near-black tunnel background so the glowing smoke trails stand out
/// against the otherwise bright sky-blue simulator backdrop.
const TUNNEL_BG: Color = Color::srgb(0.02, 0.03, 0.05);

/// Position the persistent camera at the default orbit viewpoint on entry and
/// switch it to a dark wind-tunnel background.
/// Uses the default orbit (rather than the resource) so this system does not
/// depend on [`tunnel_setup`] having run yet — both run on `OnEnter`.
fn tunnel_camera_setup(
    mut camera: Query<(&mut Transform, &mut Camera), With<crate::MenuCamera>>,
) {
    if let Ok((mut tf, mut cam)) = camera.get_single_mut() {
        *tf = orbit_transform(&TunnelCameraState::default());
        cam.clear_color = ClearColorConfig::Custom(TUNNEL_BG);
    }
}

/// Mouse-drag orbit and scroll-zoom camera around the fixed drone.
fn tunnel_camera_orbit(
    mouse_buttons: Res<ButtonInput<MouseButton>>,
    mouse_motion: Res<AccumulatedMouseMotion>,
    mouse_wheel: Res<AccumulatedMouseScroll>,
    mut cam_state: ResMut<TunnelCameraState>,
    mut camera: Query<&mut Transform, With<crate::MenuCamera>>,
) {
    // Click-and-drag rotates the camera around the drone.
    if mouse_buttons.pressed(MouseButton::Left) {
        let delta = mouse_motion.delta;
        cam_state.yaw -= delta.x * 0.008;
        cam_state.pitch = (cam_state.pitch - delta.y * 0.008).clamp(-1.4, 1.4);
    }

    // Scroll wheel zooms in/out (clamped so the camera stays in a useful range).
    let scroll = mouse_wheel.delta.y;
    if scroll != 0.0 {
        cam_state.distance = (cam_state.distance - scroll * 2.5).clamp(8.0, 60.0);
    }

    if let Ok(mut tf) = camera.get_single_mut() {
        *tf = orbit_transform(&cam_state);
    }
}

/// Build the camera transform from the orbit state (always looking at the
/// origin where the drone sits).
fn orbit_transform(state: &TunnelCameraState) -> Transform {
    let (sy, cy) = state.yaw.sin_cos();
    let (sp, cp) = state.pitch.sin_cos();
    let pos = Vec3::new(
        state.distance * cp * cy,
        state.distance * sp,
        state.distance * cp * sy,
    );
    Transform::from_translation(pos).looking_at(Vec3::ZERO, Vec3::Y)
}

/// Remove everything created for the wind tunnel when leaving the mode.
fn tunnel_cleanup(
    mut commands: Commands,
    aircraft: Query<Entity, With<AircraftRoot>>,
    trails: Query<Entity, With<TrailEmitter>>,
    visuals: Query<Entity, With<TrailVisual>>,
    hud: Query<Entity, With<TunnelHudText>>,
    mut camera: Query<(&mut Transform, &mut Camera), With<crate::MenuCamera>>,
) {
    for e in aircraft
        .iter()
        .chain(trails.iter())
        .chain(visuals.iter())
        .chain(hud.iter())
    {
        commands.entity(e).despawn();
    }
    if let Ok((mut tf, mut cam)) = camera.get_single_mut() {
        *tf = Transform::from_xyz(-60.0, 1060.0, -120.0)
            .looking_at(Vec3::new(0.0, 1000.0, 0.0), Vec3::Y);
        cam.clear_color = ClearColorConfig::Default;
    }
    commands.remove_resource::<WindTunnelSettings>();
    commands.remove_resource::<Aerodynamics>();
    commands.remove_resource::<TunnelCameraState>();
}

/// Keyboard controls for the wind tunnel.
fn tunnel_input(
    keyboard: Res<ButtonInput<KeyCode>>,
    time: Res<Time>,
    mut settings: ResMut<WindTunnelSettings>,
    mut next_state: ResMut<NextState<AppState>>,
) {
    let dt = time.delta_secs();
    let ang_rate = 50.0 * dt; // deg/s

    if keyboard.pressed(KeyCode::KeyW) {
        settings.pitch += ang_rate.to_radians();
    }
    if keyboard.pressed(KeyCode::KeyS) {
        settings.pitch -= ang_rate.to_radians();
    }
    if keyboard.pressed(KeyCode::KeyA) {
        settings.yaw += ang_rate.to_radians();
    }
    if keyboard.pressed(KeyCode::KeyD) {
        settings.yaw -= ang_rate.to_radians();
    }
    if keyboard.pressed(KeyCode::ArrowUp) {
        settings.roll += ang_rate.to_radians();
    }
    if keyboard.pressed(KeyCode::ArrowDown) {
        settings.roll -= ang_rate.to_radians();
    }

    if keyboard.pressed(KeyCode::KeyZ) {
        settings.rudder = (settings.rudder - ang_rate.to_radians()).clamp(-0.35, 0.35);
    }
    if keyboard.pressed(KeyCode::KeyC) {
        settings.rudder = (settings.rudder + ang_rate.to_radians()).clamp(-0.35, 0.35);
    }
    if keyboard.pressed(KeyCode::KeyQ) {
        settings.aileron = (settings.aileron - ang_rate.to_radians()).clamp(-0.35, 0.35);
    }
    if keyboard.pressed(KeyCode::KeyE) {
        settings.aileron = (settings.aileron + ang_rate.to_radians()).clamp(-0.35, 0.35);
    }
    if keyboard.just_pressed(KeyCode::KeyF) {
        settings.flaps_deg = if settings.flaps_deg < 5.0 {
            15.0
        } else if settings.flaps_deg < 20.0 {
            30.0
        } else {
            0.0
        };
    }

    // Wind speed / direction.
    if keyboard.pressed(KeyCode::ShiftLeft) {
        settings.wind_speed = (settings.wind_speed + 25.0 * dt).min(120.0);
    }
    if keyboard.pressed(KeyCode::ControlLeft) {
        settings.wind_speed = (settings.wind_speed - 25.0 * dt).max(1.0);
    }
    if keyboard.pressed(KeyCode::KeyR) {
        settings.wind_direction += 30.0_f32.to_radians() * dt;
    }
    if keyboard.pressed(KeyCode::KeyT) {
        settings.wind_direction -= 30.0_f32.to_radians() * dt;
    }

    if keyboard.just_pressed(KeyCode::Space) {
        *settings = WindTunnelSettings::default();
    }

    if keyboard.just_pressed(KeyCode::Escape) {
        next_state.set(AppState::MainMenu);
    }
}

/// Rotate the fixed aircraft to the requested pitch/roll/yaw (in place).
fn tunnel_update_aircraft(
    settings: Res<WindTunnelSettings>,
    mut query: Query<&mut Transform, (With<AircraftRoot>, Without<TrailEmitter>)>,
) {
    let quat = Quat::from_euler(
        bevy::math::EulerRot::ZYX,
        -settings.yaw,
        -settings.pitch,
        -settings.roll,
    );
    for mut tf in query.iter_mut() {
        tf.translation = Vec3::ZERO;
        tf.rotation = quat;
    }
}

/// Compute the aerodynamic forces/moments for the current attitude using the
/// flight_core aero model, and store them for the HUD.
fn tunnel_compute_aero(settings: Res<WindTunnelSettings>, mut aero: ResMut<Aerodynamics>) {
    use flight_core::aero::compute_forces_moments;
    use flight_core::config::AircraftConfig;
    use flight_core::state::AircraftState;

    let config = AircraftConfig::from_file(&super::resolve_config_path())
        .or_else(|_| AircraftConfig::from_file("aircraft.toml"))
        .unwrap_or_else(|_| panic!("wind tunnel: aircraft config not found"));

    // Freeze the aircraft: fixed at the origin with zero angular rates. The
    // wind tunnel blows a uniform airstream and the aircraft is the one that
    // gets pitched/rolled/yawed about its CG, so we orient the body by the
    // user-selected angles and compute the air-relative velocity by rotating
    // the fixed world airstream into the body frame.
    let mut state = AircraftState::default();
    state.pos_z = 0.0;

    let (roll, pitch, yaw) = (settings.roll as f64, settings.pitch as f64, settings.yaw as f64);
    let q = nalgebra_euler_to_quat(roll, pitch, yaw);
    state.q0 = q.0;
    state.q1 = q.1;
    state.q2 = q.2;
    state.q3 = q.3;

    // World airstream direction (matches the particle visualization: head-on
    // from the nose, i.e. along -X).
    let dir = settings.wind_direction as f64;
    let world_flow = Vector3::new(-dir.cos(), 0.0, dir.sin());
    let speed = settings.wind_speed as f64;

    // Air-relative velocity in the body frame: rotate the fixed airstream so
    // aerodynamics see the correct angle of attack / sideslip for the chosen
    // attitude.
    let rel = state
        .rotation_earth_to_body()
        .transform_vector(&(world_flow * speed));
    state.u = rel.x;
    state.v = rel.y;
    state.w = rel.z;

    // No external wind beyond the constructed airstream expression above.
    let wind_earth = Vector3::zeros();

    let (forces, moments) = compute_forces_moments(
        &state,
        &config,
        settings.elevator as f64,
        settings.aileron as f64,
        settings.rudder as f64,
        0.0,
        0.0,
        settings.flaps_deg.to_radians() as f64,
        &wind_earth,
    );

    aero.force = forces;
    aero.moment = moments;

    // Derive the same dimensionless lift coefficient the main flight sim uses
    // (CL = lift / (q * S)) from the computed body-frame forces, so the visual
    // flow field can scale its circulation with the actual physics. The default
    // low-speed configuration is the reference point for `CL_REF`.
    let lift_n = -forces.z;
    let tas = rel.norm();
    let q_dyn = if tas > 1.0 {
        0.5 * 1.225 * tas * tas
    } else {
        0.5 * 1.225
    };
    aero.cl = (lift_n / (q_dyn * config.wing_area.max(1.0))).clamp(-3.0, 3.0);
}

/// Wind direction in world space. The drone sits at the origin with its nose
/// pointing along +X, so the airstream travels from +X (front) toward -X (tail):
/// particles are injected ahead of the nose and flow back over the airframe.
/// `wind_direction` rotates that in the tunnel XZ plane (0 = head-on along -X).
fn wind_velocity(settings: &WindTunnelSettings) -> Vec3 {
    let dir = settings.wind_direction;
    Vec3::new(-dir.cos(), 0.0, dir.sin())
}

// --- Fluid (smoke) flow field around the drone ---------------------------
// The particles are advected through a lightweight potential-flow-style field
// defined in the drone's local frame (X = forward, Y = up, Z = right). It
// combines three effects so the streamlines behave like real air over a
// lifting aircraft:
//   1. Body blockage — flow diverts around the fuselage/wing and decelerates
//      at the nose (stagnation).
//   2. Wing circulation (lift) — upwash ahead of the wing, downwash behind.
//   3. Trailing wingtip vortices — counter-rotating swirls shed from the tips.
//
// Tunable physical constants (in the drone's metres).
/// Wing half-span of the drone model (m).
const WING_HALF_SPAN: f32 = 5.5;
/// Fuselage half-length along the forward axis (m).
const FUSE_HALF_LEN: f32 = 2.8;
/// Standoff gap from aircraft surfaces where smoke parcels get deflected, so
/// traces wrap around the solid airframe instead of clipping through it (m).
const STANDOFF: f32 = 0.12;
/// Length scale over which the circulation decays away from the wing (m).
const CIRC_DECAY: f32 = 2.6;
/// Circulation (lift) strength factor.
const CIRC_STRENGTH: f32 = 1.8;
/// Wingtip vortex strength factor.
const TIP_STRENGTH: f32 = 3.0;
/// Vortex core radius (m), avoids dividing by zero.
const TIP_CORE: f32 = 1.2;
/// How quickly the wingtip vortices appear behind the wing (m).
const TIP_GROW: f32 = 2.5;

// --- Rear-pusher propeller slipstream ------------------------------------
// The pusher prop at the tail accelerates air aft inside a tube that slowly
// expands and swirls (classic prop wash / helical wake). Trails that pass the
// rotating blades get drawn in, stretched and made to spiral downstream.
/// Propeller disk centre (local X = -2.38, Y = +0.08 matches aircraft.rs).
const PROP_X: f32 = -2.38;
const PROP_Y: f32 = 0.08;
/// Propeller disk radius (m), matching the 1.45 m blades.
const PROP_RADIUS: f32 = 0.75;
/// Axial slipstream acceleration, as a multiple of the free-stream speed.
const PROP_AXIAL: f32 = 1.3;
/// Tangential slipstream swirl strength, as a multiple of the axial boost.
const PROP_SWIRL: f32 = 0.7;
/// E-folding length of the slipstream downstream of the prop (m).
const PROP_DECAY: f32 = 7.0;
/// Inflow region scale ahead of the disk (m) — air is gently sucked in first.
const PROP_INFLOW: f32 = 1.4;

/// Compute the drone's world-space body axes `(forward, up, right)` from the
/// current pitch/roll/yaw, matching the aircraft mesh orientation.
fn drone_axes(settings: &WindTunnelSettings) -> (Vec3, Vec3, Vec3) {
    let q = Quat::from_euler(
        bevy::math::EulerRot::ZYX,
        -settings.yaw,
        -settings.pitch,
        -settings.roll,
    );
    (q * Vec3::X, q * Vec3::Y, q * Vec3::Z)
}

/// Sample the flow-perturbation velocity (m/s) in the drone's local frame at a
/// local position, given the freestream wind speed. The drone's nose is +X and
/// the airstream comes from the front, so local upstream = +X and downstream
/// = -X.
///
/// `lift_factor` is derived from the *actual* lift coefficient computed by the
/// main flight-sim aero model (see `tunnel_compute_aero`), so the circulation
/// and tip-vortex strength track the real physics: higher AoA → more upwash/
/// downwash and stronger wing-tip swirls; zero/negative CL → circulation
/// disappears/reverses exactly like real lift.
fn local_flow(local: Vec3, speed: f32, lift_factor: f32) -> Vec3 {
    let (x, y, z) = (local.x, local.y, local.z);
    let mut v = Vec3::ZERO;

    // --- 1) Solid-body interaction -----------------------------------------
    // The airframe is treated as a real solid using the *actual* mesh
    // dimensions (see aircraft.rs): fuselage, canopy hump, main wing, V-tail
    // fins and the rear engine nacelle. Any smoke parcel that wanders inside
    // (or up to a standoff gap from) a surface is pushed out along the local
    // surface normal AND given a tangential swirl around the body, so the
    // traces visibly wrap over the leading edge, curl around the canopy and
    // flow over/under the wing — like smoke strokes in a real wind tunnel.
    let span_pos = (z / WING_HALF_SPAN).clamp(-1.0, 1.0);
    let span_taper = (1.0 - span_pos.abs()).clamp(0.0, 1.0);
    let span_taper = span_taper * span_taper;

    // Accumulate the largest penetration into the solid and the local outward
    // surface normal for that element (NULL means outside everywhere).
    let mut d_over_max = 0.0f32;
    let mut push = Vec3::ZERO;

    // (a) Fuselage: elliptical cylinder on the X axis (half-height ~0.45,
    //     half-width ~0.40, matching the Cuboid(4.2,0.75,0.75) plus slop).
    if x.abs() <= FUSE_HALF_LEN + STANDOFF {
        let rn = ((y / 0.45).powi(2) + (z / 0.40).powi(2)).sqrt().max(1e-4);
        let pen = 1.0 + STANDOFF / 0.45 - rn;
        if pen > 0.0 && pen > d_over_max {
            d_over_max = pen;
            // Elliptical gradient gives the outward surface normal.
            push = Vec3::new(0.0, y / (0.45 * 0.45), z / (0.40 * 0.40)) / rn;
        }
    }

    // Canopy hump: flattened ellipsoid on top of the fuselage at x≈0.9.
    let canopy_x = x - 0.9;
    if y >= 0.0 {
        let er = ((canopy_x / 0.95).powi(2) + ((y - 0.40) / 0.28).powi(2)
            + (z / 0.35).powi(2))
            .sqrt()
            .max(1e-4);
        let pen = 1.0 + STANDOFF / 0.28 - er;
        if pen > 0.0 && pen > d_over_max {
            d_over_max = pen;
            push = Vec3::new(canopy_x / (0.95 * 0.95), (y - 0.40) / (0.28 * 0.28), z / (0.35 * 0.35))
                / er;
        }
    }

    // (b) Main wing: thin slab spanning the span at y≈0.15, chord along X
    //     centered at x=0.25 (actual Cuboid(0.85,0.09,10.6)).
    if x.abs() <= 1.0 && z.abs() <= WING_HALF_SPAN + 0.2 {
        let w_pen = 0.5 + STANDOFF - (x - 0.25).abs();
        let w_thick = 0.10 + STANDOFF - (y - 0.15).abs();
        if w_pen > 0.0 && w_thick > 0.0 {
            let pen = w_pen.min(w_thick);
            if pen > d_over_max {
                d_over_max = pen;
                // Push normal to the nearest wing surface: top/bottom if closer
                // in Y, leading/trailing-edge normal otherwise.
                if w_thick < w_pen {
                    push = Vec3::new(0.0, (y - 0.15).signum(), 0.0);
                } else {
                    push = Vec3::new(-(x - 0.25).signum(), 0.0, 0.0);
                }
            }
        }
    }

    // (c) V-tail fins (aft, canted ~38°): thin canted slabs at z≈±0.55.
    for zt in [-0.55f32, 0.55] {
        if x >= -2.7 && x <= -1.4 {
            let cant = 38.0_f32.to_radians();
            let s = (-zt).signum();
            // Signed distance to the canted fin plane (about the X axis).
            let n_nearest = s * ((y - 0.65) * cant.sin() + (z - zt) * cant.cos());
            let along = (y - 0.65) * cant.cos() - (z - zt) * cant.sin();
            let fin_half = 0.45;
            let fin_pen = 0.06 + STANDOFF - n_nearest.abs();
            if fin_pen > 0.0 && along.abs() <= fin_half && fin_pen > d_over_max {
                d_over_max = fin_pen;
                push = Vec3::new(0.0, n_nearest.signum() * cant.sin(), n_nearest.signum() * cant.cos());
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
            push = Vec3::new(0.0, y, z) / nr.max(1e-4);
        }
    }

    // Apply the solid interaction: outward push keeps parcels off the skin;
    // the tangential swirl (X cross normal) makes them curl around the body so
    // traces flow over the nose, around the fuselage and over the wing rather
    // than simply ricocheting off.
    if d_over_max > 0.0 {
        let intensity = (d_over_max / (STANDOFF * 3.0)).clamp(0.0, 1.0);
        v += push * (speed * 2.5 * intensity);
        let swirl = Vec3::new(0.0, -push.z, push.y);
        v += swirl.normalize_or_zero() * (speed * 1.4 * intensity);
        // Mild downstream deceleration at the very nose (stagnation) so traces
        // visually pause and wrap around the rounded radome.
        if x > FUSE_HALF_LEN * 0.5 && x <= FUSE_HALF_LEN + STANDOFF {
            v.x += speed * 0.8 * intensity;
        }
    }

    // --- 2) Wake behind the solid ------------------------------------------
    // Downstream of the wing and fuselage the flow separates and re-energises:
    // traces visibly churn and spread — the classic turbulent wake. It is
    // deterministic (no time dependence) so it stays stable frame to frame.
    if x < -1.0 {
        let wake_d = -x - 1.0;
        let wake_len = 12.0;
        let wake_breadth = (wake_d / wake_len).clamp(0.0, 1.0);
        let alt = (y * y + z * z).sqrt();
        let spread = (-(alt * alt) / (3.4 * 3.4)).exp();
        let wobble = ((x * 2.3).sin() + (y * 3.7).sin() + (z * 1.9).sin()) * 0.5;
        v.y += speed * 0.5 * wake_breadth * spread * (y / (alt + 0.6)) * wobble;
        v.z += speed * 0.5 * wake_breadth * spread * (z / (alt + 0.6)) * wobble;
        // Slight deceleration directly behind the body (pressure drag).
        let core = (-(alt * alt) / (2.0 * 2.0)).exp();
        v.x -= speed * 0.7 * wake_breadth * core;
    }

    // --- 3) Wing circulation (lift): upwash ahead of the wing (x > 0, near
    //    the nose) and downwash behind (x < 0, toward the tail), decaying
    //    away from the wing line and tapering toward the tips. The magnitude
    //    is scaled by the physics-drawn `lift_factor` so it follows the real
    //    lift: sign flips under negative CL (inverted flight), and vanishes
    //    at zero lift.
    let r_wing = (x * x + y * y).max(TIP_CORE * TIP_CORE).sqrt();
    let circ_decay = (-((r_wing - TIP_CORE).powi(2)) / (CIRC_DECAY * CIRC_DECAY)).exp();
    let w_circ = CIRC_STRENGTH * speed * lift_factor * (x / r_wing) * circ_decay * span_taper;
    v.y += w_circ;

    // --- 4) Trailing wingtip vortices (counter-rotating swirls shed behind
    //    the wing, downstream toward the tail, x < 0). Their rotation mirrors
    //    the circulation sign from the physics, so they shrink with lift.
    for (zt, s) in [(WING_HALF_SPAN, -1.0), (-WING_HALF_SPAN, 1.0)] {
        // Vortex line lies along X, peaking aft of the wing (x < x_tip).
        let aft = 0.5 - 0.5 * ((x + 1.2) / TIP_GROW).clamp(-1.0, 1.0);
        let ry = y;
        let rz = z - zt;
        let r2 = ry * ry + rz * rz + TIP_CORE * TIP_CORE;
        let k = TIP_STRENGTH * speed * lift_factor * s * aft / r2;
        v.y += k * (-rz);
        v.z += k * (ry);
    }

    // --- 5) Rear-pusher propeller slipstream -------------------------------
    // The prop at the tail (x ≈ -2.38, y ≈ +0.08) drags air in ahead of the
    // disk and blows it out the back inside a tube that slowly expands and
    // swirls — the classic helical prop wash. Any smoke passing the spinning
    // blades gets drawn into the disk, stretched aft and made to spiral, so
    // the trails visibly "react" to the propeller.
    let prx = x - PROP_X;
    let pry = y - PROP_Y;
    let pr = (pry * pry + z * z).sqrt();
    let core_radius = 0.30 * PROP_RADIUS;
    let core_r2 = core_radius * core_radius;
    // Inflow region ahead of the disk (suck-in).
    let inflow_prox = (-(prx * prx) / (PROP_INFLOW * PROP_INFLOW)).exp();
    // Downstream slipstream envelope, slowly widening behind the prop.
    let aft = if prx < 0.0 {
        (-(prx * prx) / (PROP_DECAY * PROP_DECAY)).exp()
    } else {
        0.0
    };
    let slip_rad = PROP_RADIUS * (1.0 + 0.05 * prx.clamp(-PROP_DECAY, 0.0).abs());
    let disk = (-(pr * pr) / (slip_rad * slip_rad)).exp();
    if disk > 0.02 {
        // Axial momentum: gentle suck-in upstream, strong blow-out downstream.
        v.x -= speed * PROP_AXIAL * disk * (0.25 * inflow_prox + aft);
        // Rotational swirl (Rankine vortex so the centre and far field are
        // calm; tangential velocity peaks near the mid-blade radius).
        let rg = (2.0 * core_radius * pr) / (pr * pr + core_r2);
        let swirl_mag = speed * PROP_AXIAL * PROP_SWIRL * aft * disk * rg;
        let r_safe = pr.max(1e-3);
        v.y += swirl_mag * (-z / r_safe);
        v.z += swirl_mag * (pry / r_safe);
    }

    // Cap the total perturbation so near singularities never blow particles up.
    let max_mag = speed * 4.0;
    if v.length() > max_mag {
        v = v.normalize() * max_mag;
    }
    v
}

/// Smoke-rake nozzle position in world space for a given row/column, placed in
/// a vertical sheet ahead of the drone's nose. This is the fixed source point
/// from which each smoke filament is continuously born.
fn rake_slot(row: usize, col: usize) -> Vec3 {
    let height_frac = if RAKE_ROWS > 1 {
        row as f32 / (RAKE_ROWS - 1) as f32
    } else {
        0.5
    };
    let width_frac = if RAKE_COLS > 1 {
        col as f32 / (RAKE_COLS - 1) as f32
    } else {
        0.5
    };
    Vec3::new(
        RAKE_X,
        (height_frac - 0.5) * (2.0 * RAKE_HALF_HEIGHT),
        (width_frac - 0.5) * (2.0 * RAKE_HALF_WIDTH),
    )
}

/// Top-rake nozzle position: a horizontal sheet of sources hanging above the
/// airframe (y = [`TOP_RAKE_Y`]), spread along the stream (X) and across it (Z).
/// These filaments sink toward the aircraft as they are blown downstream, so
/// the smoke visibly curtains down over the canopy, wing tops and through the
/// pusher propeller.
fn top_rake_slot(row: usize, col: usize) -> Vec3 {
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
    Vec3::new(
        TOP_RAKE_X_MIN + fx * (TOP_RAKE_X_MAX - TOP_RAKE_X_MIN),
        TOP_RAKE_Y,
        (fz - 0.5) * (2.0 * TOP_RAKE_HALF_WIDTH),
    )
}

/// Generate the smoke-trail filaments: the classic wind-tunnel flow
/// visualization. A compact rake of smoke sources sits ahead of the nose; each
/// emits a long, thin filament that advects through the flow field, bending
/// over and around the airframe and trailing far downstream.
///
/// Realism: the upwash/downwash and wing-tip circulation strength are driven by
/// the dimensionless lift coefficient computed by the main flight-sim physics
/// (see `tunnel_compute_aero`), so increasing pitch angle (AoA) visibly
/// strengthens the trailing vortices exactly like real lift.
///
/// Performance:
/// * The expensive rotated-frame field is sampled only for streakline points
///   within [`FLOW_REGION`] of the drone; everything else rides the uniform
///   free stream.
/// * The advection runs on the CPU (it is only a few thousand simple samples
///   per frame), but the rendered result is a *single combined mesh* rewritten
///   each frame — one draw call for the whole smoke field, so the GPU does all
///   the heavy rasterisation.
/// * The steady-state cost does not depend on the full chamber volume; only a
///   small rake around the model generates geometry.
fn advance_flow(
    settings: Res<WindTunnelSettings>,
    time: Res<Time>,
    aero: Res<Aerodynamics>,
    mut meshes: ResMut<Assets<Mesh>>,
    mut emitters: Query<&mut TrailEmitter>,
    visual: Query<&TrailVisual>,
) {
    let dt = time.delta_secs();
    let wind = wind_velocity(&settings) * settings.wind_speed;
    let (fwd, up, right) = drone_axes(&settings);
    let region2 = FLOW_REGION * FLOW_REGION;

    // Physics-coupled lift factor: CL scaled around the cruise reference. This
    // sign+magnitude is what makes the visual flow match flight_core's forces.
    let lift_factor = ((aero.cl as f32) / CL_REF).clamp(-3.0, 3.0);

    // 1) Advect every existing streakline point one step downstream through
    //    the physics-driven flow, then inject a fresh parcel at the (fixed)
    //    rake slot and drop the oldest (furthest downstream) point. The stored
    //    chain is exactly the streakline of smoke born over the last frames.
    for mut em in emitters.iter_mut() {
        let source = em.source;
        let top = em.top;
        for p in em.points.iter_mut() {
            let mut vel = if p.length_squared() < region2 {
                let local = Vec3::new(p.dot(fwd), p.dot(up), p.dot(right));
                let field_local = local_flow(local, settings.wind_speed, lift_factor);
                let field_world =
                    fwd * field_local.x + up * field_local.y + right * field_local.z;
                wind + field_world
            } else {
                wind
            };
            // Top-rake filaments sink toward the airframe as they fly, so the
            // smoke curtains down over the canopy/wing tops instead of sailing
            // clean over everything at constant altitude.
            if top {
                let sink = ((p.y - TOP_DESCENT * 2.0) / (TOP_RAKE_Y * 0.9)).clamp(0.0, 1.0);
                vel.y -= settings.wind_speed * TOP_DESCENT * sink;
            }
            *p += vel * dt;
        }
        em.points.pop();
        em.points.insert(0, source);
    }

    // 2) Rebuild the single combined trail mesh from all streaklines.
    if let Ok(vis) = visual.get_single() {
        let chains: Vec<Vec<Vec3>> = emitters.iter().map(|em| em.points.clone()).collect();
        meshes.insert(&vis.mesh, build_trail_mesh(&chains));
    }
}

// --- Combined trail mesh --------------------------------------------------
// Each trail link becomes a thin square stick (oriented cuboid) spanning two
// consecutive streakline samples. All links are packed into a single indexed
// triangle mesh (one entity, one draw call) so the GPU rasterises the whole
// smoke field in a single pass.

/// Unit cube corners (±0.5). Ordering only matters for winding, which is
/// irrelevant here because the material is double-sided.
const BOX_CORNERS: [Vec3; 8] = [
    Vec3::new(-0.5, -0.5, 0.5),
    Vec3::new(0.5, -0.5, 0.5),
    Vec3::new(0.5, 0.5, 0.5),
    Vec3::new(-0.5, 0.5, 0.5),
    Vec3::new(-0.5, -0.5, -0.5),
    Vec3::new(0.5, -0.5, -0.5),
    Vec3::new(0.5, 0.5, -0.5),
    Vec3::new(-0.5, 0.5, -0.5),
];

/// Six faces as (corner indices, outward normal).
const BOX_FACES: [([usize; 4], Vec3); 6] = [
    ([3, 2, 6, 7], Vec3::Y),
    ([1, 0, 4, 5], Vec3::NEG_Y),
    ([1, 5, 6, 2], Vec3::X),
    ([0, 3, 7, 4], Vec3::NEG_X),
    ([0, 1, 2, 3], Vec3::Z),
    ([5, 4, 7, 6], Vec3::NEG_Z),
];

/// Pack every trail into a single indexed mesh of thin oriented sticks.
/// `chains[c]` holds the streakline points of rake source `c`; each pair of
/// consecutive points becomes one stick. Colors fade out toward the tail so
/// the recycled downstream end melts away smoothly.
fn build_trail_mesh(chains: &[Vec<Vec3>]) -> Mesh {
    let n_seg_total = chains.len() * TRAIL_SEGS;
    let mut positions = Vec::with_capacity(n_seg_total * 24);
    let mut normals = Vec::with_capacity(n_seg_total * 24);
    let mut uvs = Vec::with_capacity(n_seg_total * 24);
    let mut colors = Vec::with_capacity(n_seg_total * 24);
    let mut indices = Vec::with_capacity(n_seg_total * 36);

    for chain in chains {
        for index in 0..TRAIL_SEGS {
            let a = chain[index];
            let b = chain[index + 1];
            let seg = b - a;
            let len = seg.length();
            let mid = (a + b) * 0.5;

            let rot = if len > 1e-6 {
                Quat::from_rotation_arc(Vec3::Y, seg / len)
            } else {
                Quat::IDENTITY
            };
            let scale = Vec3::new(TRAIL_TUBE_RADIUS * 2.0, len.max(1e-4), TRAIL_TUBE_RADIUS * 2.0);

            // Fade brightness/alpha toward the tail so trails taper to nothing.
            let t = index as f32 / TRAIL_SEGS as f32;
            let color = [0.95 - 0.25 * t, 1.0, 1.0, 1.0 - t * 0.92];

            let base = indices.len() as u32;
            for (corners, n) in BOX_FACES {
                for &c in &corners {
                    let p = rot * (BOX_CORNERS[c] * scale) + mid;
                    positions.push([p.x, p.y, p.z]);
                    normals.push([n.x, n.y, n.z]);
                    uvs.push([0.0, 0.0]);
                    colors.push(color);
                }
            }
            indices.push(base);
            indices.push(base + 1);
            indices.push(base + 2);
            indices.push(base);
            indices.push(base + 2);
            indices.push(base + 3);
        }
    }

    use bevy::asset::RenderAssetUsages;

    let mut mesh = Mesh::new(
        PrimitiveTopology::TriangleList,
        RenderAssetUsages::RENDER_WORLD,
    );
    mesh.insert_attribute(Mesh::ATTRIBUTE_POSITION, positions);
    mesh.insert_attribute(Mesh::ATTRIBUTE_NORMAL, normals);
    mesh.insert_attribute(Mesh::ATTRIBUTE_UV_0, uvs);
    mesh.insert_attribute(Mesh::ATTRIBUTE_COLOR, colors);
    mesh.insert_indices(Indices::U32(indices));
    mesh
}

/// Refresh the wind-tunnel HUD with settings and computed forces.
fn tunnel_update_hud(
    settings: Res<WindTunnelSettings>,
    aero: Res<Aerodynamics>,
    mut text_query: Query<&mut Text, With<TunnelHudText>>,
) {
    // Body-frame: lift acts along +Z_air usually reported as -Fz; drag -Fx.
    let lift_n = -aero.force.z;
    let drag_n = -aero.force.x;
    let side_n = aero.force.y;
    let mom = aero.moment;

    let wind_dir_deg = settings.wind_direction.to_degrees().rem_euclid(360.0);

    for mut text in text_query.iter_mut() {
        **text = format!(
            "WIND TUNNEL\n\
             -----------------------------\n\
             Wind speed:  {:.1} m/s\n\
             Wind dir:    {:.0}\u{00b0}\n\
             Pitch:       {:.1}\u{00b0}\n\
             Roll:        {:.1}\u{00b0}\n\
             Yaw (beta):  {:.1}\u{00b0}\n\
             -----------------------------\n\
             Lift:  {:+7.0} N\n\
             Drag:  {:+7.0} N\n\
             Side:  {:+7.0} N\n\
             Roll M:  {:+6.0} N\u{b7}m\n\
             Pitch M: {:+6.0} N\u{b7}m\n\
             Yaw M:   {:+6.0} N\u{b7}m\n\n\
             CONTROLS\n\
             -----------------------------\n\
             Pitch:      [W] / [S]\n\
             Yaw:        [A] / [D]\n\
             Roll:       [\u{2191}] / [\u{2193}]\n\
             Aileron:    [Q] / [E]\n\
             Rudder:     [Z] / [C]\n\
             Flaps:      [F] cycle\n\
             Wind speed: [Shift] / [Ctrl]\n\
             Wind dir:   [R] / [T]\n\
             Reset:      [Space]   Exit: [Esc]",
            settings.wind_speed,
            wind_dir_deg,
            settings.pitch.to_degrees(),
            settings.roll.to_degrees(),
            settings.yaw.to_degrees(),
            lift_n,
            drag_n,
            side_n,
            mom.x,
            mom.y,
            mom.z,
        );
    }
}

/// Convert Euler (roll, pitch, yaw), ZYX intrinsic, to a quaternion
/// `(w, x, y, z)` matching `AircraftState` field order.
fn nalgebra_euler_to_quat(roll: f64, pitch: f64, yaw: f64) -> (f64, f64, f64, f64) {
    let (sr, cr) = (roll * 0.5).sin_cos();
    let (sp, cp) = (pitch * 0.5).sin_cos();
    let (sy, cy) = (yaw * 0.5).sin_cos();
    let w = cr * cp * cy + sr * sp * sy;
    let x = sr * cp * cy - cr * sp * sy;
    let y = cr * sp * cy + sr * cp * sy;
    let z = cr * cp * sy - sr * sp * cy;
    (w, x, y, z)
}
