//! Bevy-based visualization and interactive 6-DOF flight simulator.
//!
//! Spawns a rich environment (real-world 3D terrain/satellite textures, vast terrain,
//! farmlands, airport with runway and lights, downtown skyscrapers, mountain ranges,
//! forests, wind turbines, and low-poly 3D clouds) and a 3D tactical reconnaissance UAV drone
//! with animated ailerons, flaps, V-tail ruddervators, spinning propeller, full 6-DOF controls,
//! and real-time JSBSim HUD telemetry.

mod aircraft;
mod environment;
mod menu;

use bevy::prelude::*;
use flight_core::Simulator;

use aircraft::{animate_control_surfaces, spawn_aircraft, spin_propeller, AircraftRoot};
use environment::{animate_wind_turbines, spawn_environment};
use menu::MainMenuPlugin;

/// Top-level application screen. Each variant is a separate, independently
/// schedulable mode; add new component-simulator screens here as tabs.
#[derive(States, Debug, Clone, PartialEq, Eq, Hash, Default)]
pub enum AppState {
    #[default]
    MainMenu,
    FlightSim,
}

/// Candidate paths for the aircraft configuration file.
const CONFIG_PATHS: [&str; 2] = ["aircraft.toml", "../aircraft.toml"];
/// Physics time step in seconds.
const DT: f64 = 1.0 / 60.0;

/// 6-DOF Flight control surface deflections, pitch trim, and engine setting.
#[derive(Resource)]
pub struct FlightControls {
    pub elevator: f64,      // Manual stick pitch input (radians)
    pub elevator_trim: f64, // Baseline pitch trim tab setting (radians)
    pub aileron: f64,       // Roll stick input (radians)
    pub rudder: f64,        // Rudder pedal yaw input (radians)
    pub flaps_deg: f64,     // Trailing-edge flap deployment (0°, 15°, 30°)
    pub throttle: f64,      // Engine thrust setting (0.0 to 1.0)
    pub auto_level: bool,   // Wing-leveler / Altitude assist mode
    pub target_alt: f64,    // Target cruise altitude for auto-level
}

impl Default for FlightControls {
    fn default() -> Self {
        Self {
            elevator: 0.0,
            elevator_trim: -0.003,
            aileron: 0.0,
            rudder: 0.0,
            flaps_deg: 0.0,
            throttle: 0.21,
            auto_level: false,
            target_alt: 1000.0,
        }
    }
}

/// Marker for the HUD text UI entity.
#[derive(Component)]
struct HudText;

/// Marker for the HUD container (its clickable-free root node).
#[derive(Component)]
struct HudRoot;

/// Resolve the first existing aircraft config path.
fn resolve_config_path() -> String {
    for p in CONFIG_PATHS {
        if std::path::Path::new(p).exists() {
            return p.to_string();
        }
    }
    panic!(
        "aircraft config not found; tried {} (set CWD to the workspace root)",
        CONFIG_PATHS.join(", ")
    );
}

fn main() {
    App::new()
        .add_plugins(DefaultPlugins.set(WindowPlugin {
            primary_window: Some(Window {
                title: "Tactical UAV Drone Simulator - 6-DOF Bevy Visualizer".into(),
                ..default()
            }),
            ..default()
        }))
        .insert_resource(ClearColor(Color::srgb(0.55, 0.74, 0.94))) // Crisp sky blue
        .insert_resource(AmbientLight {
            color: Color::srgb(0.9, 0.94, 1.0),
            brightness: 450.0,
        })
        .insert_resource(FlightControls::default())
        .init_state::<AppState>()
        .enable_state_scoped_entities::<AppState>()
        .add_plugins(MainMenuPlugin)
        .add_systems(Startup, menu_env_setup)
        .add_systems(OnEnter(AppState::FlightSim), setup)
        .add_systems(OnExit(AppState::FlightSim), cleanup_sim_screen)
        .add_systems(
            PreUpdate,
            (handle_manual_input, update_aircraft)
                .chain()
                .run_if(in_state(AppState::FlightSim)),
        )
        .add_systems(
            Update,
            (
                spin_propeller,
                animate_control_surfaces,
                animate_wind_turbines,
                update_hud,
                print_telemetry,
            )
                .run_if(in_state(AppState::FlightSim)),
        )
        .run();
}

/// Injected simulation resource.
#[derive(Resource)]
struct Sim {
    physics: Simulator,
    wind: flight_core::WindEnvironment,
    /// Last wind vector (Earth NED) applied, for HUD/telemetry display.
    last_wind: flight_core::nalgebra::Vector3<f64>,
}

/// Marker for the chase camera entity.
#[derive(Component)]
struct ChaseCamera;

/// Marker for the main-menu camera (renders the menu UI + environment backdrop).
#[derive(Component)]
struct MenuCamera;

/// Timestamp accumulator for the per-second telemetry log.
#[derive(Resource, Default)]
struct TelemetryTimer(f32);

/// Build a [`WindConfig`] from optional environment variables so the air
/// simulation can be tweaked without recompiling:
///   RAPTOR_WIND_SPEED    m/s (default 0 = still air)
///   RAPTOR_WIND_DIR_DEG  true bearing the wind blows TOWARD (default 0)
///   RAPTOR_WIND_SHEAR    "1" enables altitude shear (default off)
///   RAPTOR_TURBULENCE    light | moderate | severe (default light)
fn wind_from_env() -> flight_core::WindConfig {
    use flight_core::TurbulenceIntensity;

    let env = |k: &str| std::env::var(k).ok();
    let speed = env("RAPTOR_WIND_SPEED")
        .and_then(|v| v.parse::<f64>().ok())
        .unwrap_or(0.0);
    let dir_deg = env("RAPTOR_WIND_DIR_DEG")
        .and_then(|v| v.parse::<f64>().ok())
        .unwrap_or(0.0);
    let shear = env("RAPTOR_WIND_SHEAR").map(|v| v == "1").unwrap_or(false);
    let turbulence = match env("RAPTOR_TURBULENCE").as_deref() {
        Some("moderate") => TurbulenceIntensity::Moderate,
        Some("severe") => TurbulenceIntensity::Severe,
        _ => TurbulenceIntensity::Light,
    };

    flight_core::WindConfig {
        wind_speed: speed,
        wind_direction: dir_deg.to_radians(),
        reference_altitude: 1000.0,
        wind_shear: shear,
        turbulence,
        turbulence_scale: 533.0,
        seed: 0x9E37_79B9_7F4A_7C15,
    }
}

fn setup(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
    mut controls: ResMut<FlightControls>,
    asset_server: Res<AssetServer>,
) {
    // Instantiate the flight simulator and calculate exact level-flight trim
    let mut sim = Simulator::new(&resolve_config_path());
    let (elev_trim, throttle_trim) = sim.trim_level_flight(1000.0, 60.0);

    controls.elevator = 0.0;
    controls.elevator_trim = elev_trim;
    controls.throttle = throttle_trim;
    controls.flaps_deg = 0.0;
    controls.target_alt = 1000.0;

    // Wind is configured through environment variables so it can be tuned
    // without recompiling. Defaults to still air (wind disabled).
    let wind_config = wind_from_env();
    commands.insert_resource(Sim {
        physics: sim,
        wind: flight_core::WindEnvironment::new(wind_config),
        last_wind: flight_core::nalgebra::Vector3::zeros(),
    });
    commands.insert_resource(TelemetryTimer::default());

    // --- Spawn Tactical Drone UAV Model (or custom model from assets/models/) -
    spawn_aircraft(&mut commands, &mut meshes, &mut materials, &asset_server);

    // --- On-Screen Flight HUD -----------------------------------------
    commands
        .spawn((
            HudRoot,
            Node {
                position_type: PositionType::Absolute,
                top: Val::Px(16.0),
                left: Val::Px(16.0),
                padding: UiRect::all(Val::Px(14.0)),
                flex_direction: FlexDirection::Column,
                row_gap: Val::Px(4.0),
                ..default()
            },
            BackgroundColor(Color::srgba(0.04, 0.07, 0.11, 0.84)),
        ))
        .with_children(|parent| {
            parent.spawn((
                HudText,
                Text::new("TACTICAL UAV DRONE TELEMETRY HUD\nInitializing..."),
                TextFont {
                    font_size: 13.5,
                    ..default()
                },
                TextColor(Color::srgb(0.95, 0.96, 0.98)),
            ));
        });
}

/// Runs once at startup: spawns the static sun and world environment. These
/// persist across all app states so the simulator does not rebuild (and
/// duplicate) a ~30-entity scene every time the user enters the simulator.
fn menu_env_setup(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
    asset_server: Res<AssetServer>,
) {
    // --- Sunlight (Directional Light with Shadows) --------------------
    commands.spawn((
        DirectionalLight {
            illuminance: 14000.0,
            shadows_enabled: true,
            ..default()
        },
        Transform::from_rotation(Quat::from_euler(
            EulerRot::ZYX,
            0.0,
            std::f32::consts::FRAC_PI_4,
            -std::f32::consts::FRAC_PI_4,
        )),
    ));

    // --- Spawn Environment (Terrain, Airport, City, Mountains, Clouds) ---
    spawn_environment(&mut commands, &mut meshes, &mut materials, &asset_server);

    // --- Single Persistent Camera -------------------------------------
    // One camera for the entire app. It is spawned here sitting at the raw
    // menu/environment view; while the simulator is active, `update_aircraft`
    // repositions it as the chase camera every frame. Using a single camera
    // (rather than separate menu + chase cameras) avoids Bevy's "camera order
    // ambiguity" warnings that occur when two cameras are briefly active
    // during state transitions.
    let mut position = Transform::from_xyz(-60.0, 1050.0, -120.0)
        .looking_at(Vec3::new(0.0, 1000.0, 0.0), Vec3::Y);
    position.translation.y = 1060.0;
    commands.spawn((
        ChaseCamera,
        MenuCamera,
        Camera3d::default(),
        Projection::Perspective(PerspectiveProjection {
            far: 35000.0,
            fov: 60.0_f32.to_radians(),
            ..default()
        }),
        DistanceFog {
            color: Color::srgba(0.68, 0.78, 0.90, 1.0),
            falloff: FogFalloff::Linear {
                start: 3500.0,
                end: 28000.0,
            },
            ..default()
        },
        position,
    ));
}

/// Runs when leaving the simulator back to the menu: despawns the aircraft
/// and HUD entities spawned in [`setup`], and removes the per-session
/// simulation resources. The persistent camera and static environment/sun
/// are left in place; the camera is simply repositioned to look at the world
/// from the menu vantage instead of chasing the (now-despawned) aircraft.
fn cleanup_sim_screen(
    mut commands: Commands,
    aircraft: Query<Entity, With<AircraftRoot>>,
    huds: Query<Entity, With<HudRoot>>,
    mut camera: Query<&mut Transform, (With<MenuCamera>, With<ChaseCamera>)>,
) {
    for e in aircraft.iter().chain(huds.iter()) {
        commands.entity(e).despawn_recursive();
    }
    if let Ok(mut tf) = camera.get_single_mut() {
        *tf = Transform::from_xyz(-60.0, 1060.0, -120.0)
            .looking_at(Vec3::new(0.0, 1000.0, 0.0), Vec3::Y);
    }
    commands.remove_resource::<Sim>();
    commands.remove_resource::<TelemetryTimer>();
}

/// Reads keyboard input to adjust 6-DOF controls, flaps, elevator trim, and autopilot.
fn handle_manual_input(
    keyboard: Res<ButtonInput<KeyCode>>,
    time: Res<Time>,
    mut controls: ResMut<FlightControls>,
    mut sim: ResMut<Sim>,
    mut next_state: ResMut<NextState<AppState>>,
) {
    let dt = time.delta_secs() as f64;

    // --- Pitch / Elevator: S/Down = Pull UP (climb), W/Up = Push DOWN (dive) ---
    // Physics convention: negative elevator deflection = nose-up (pull).
    let max_elevator = 0.35; // ~20 degrees
    let elevator_rate = 0.9; // rad/s
    let mut manual_pitch = false;
    if keyboard.pressed(KeyCode::KeyS) || keyboard.pressed(KeyCode::ArrowDown) {
        controls.elevator = (controls.elevator - elevator_rate * dt).max(-max_elevator);
        manual_pitch = true;
    } else if keyboard.pressed(KeyCode::KeyW) || keyboard.pressed(KeyCode::ArrowUp) {
        controls.elevator = (controls.elevator + elevator_rate * dt).min(max_elevator);
        manual_pitch = true;
    } else {
        // Auto-center manual stick input when released
        if controls.elevator.abs() < 0.01 {
            controls.elevator = 0.0;
        } else {
            controls.elevator -= controls.elevator.signum() * elevator_rate * 1.5 * dt;
        }
    }

    // --- Roll / Ailerons: D/Right = Roll RIGHT, A/Left = Roll LEFT ---
    let max_aileron = 0.35;
    let aileron_rate = 1.0;
    let mut manual_roll = false;
    if keyboard.pressed(KeyCode::KeyD) || keyboard.pressed(KeyCode::ArrowRight) {
        controls.aileron = (controls.aileron + aileron_rate * dt).min(max_aileron);
        manual_roll = true;
    } else if keyboard.pressed(KeyCode::KeyA) || keyboard.pressed(KeyCode::ArrowLeft) {
        controls.aileron = (controls.aileron - aileron_rate * dt).max(-max_aileron);
        manual_roll = true;
    } else {
        // Auto-center ailerons when released
        if controls.aileron.abs() < 0.01 {
            controls.aileron = 0.0;
        } else {
            controls.aileron -= controls.aileron.signum() * aileron_rate * 1.5 * dt;
        }
    }

    // --- Yaw / Rudder: E/C = Yaw RIGHT, Q/Z = Yaw LEFT ---
    // Physics convention: positive rudder deflection = yaw right.
    let max_rudder = 0.35;
    let rudder_rate = 0.9;
    if keyboard.pressed(KeyCode::KeyE) || keyboard.pressed(KeyCode::KeyC) {
        controls.rudder = (controls.rudder + rudder_rate * dt).min(max_rudder);
    } else if keyboard.pressed(KeyCode::KeyQ) || keyboard.pressed(KeyCode::KeyZ) {
        controls.rudder = (controls.rudder - rudder_rate * dt).max(-max_rudder);
    } else {
        if controls.rudder.abs() < 0.01 {
            controls.rudder = 0.0;
        } else {
            controls.rudder -= controls.rudder.signum() * rudder_rate * 1.5 * dt;
        }
    }

    // --- Flaps Deployment Cycle: 'F' Key (0° -> 15° -> 30° -> 0°) ---
    if keyboard.just_pressed(KeyCode::KeyF) {
        if controls.flaps_deg < 5.0 {
            controls.flaps_deg = 15.0; // Takeoff / Approach flaps
        } else if controls.flaps_deg < 20.0 {
            controls.flaps_deg = 30.0; // Full Landing flaps
        } else {
            controls.flaps_deg = 0.0;  // Retracted
        }
    }

    // --- Elevator Pitch Trim Adjustments: [ (Nose Down Trim) / ] (Nose Up Trim) ---
    let trim_rate = 0.05; // rad/s
    if keyboard.pressed(KeyCode::BracketRight) {
        controls.elevator_trim = (controls.elevator_trim + trim_rate * dt).min(0.15);
    }
    if keyboard.pressed(KeyCode::BracketLeft) {
        controls.elevator_trim = (controls.elevator_trim - trim_rate * dt).max(-0.15);
    }

    // --- Throttle: Left Shift / Equal = Increase, Left Ctrl / Minus = Decrease ---
    let throttle_rate = 0.35;
    if keyboard.pressed(KeyCode::ShiftLeft) || keyboard.pressed(KeyCode::Equal) {
        controls.throttle = (controls.throttle + throttle_rate * dt).min(1.0);
    }
    if keyboard.pressed(KeyCode::ControlLeft) || keyboard.pressed(KeyCode::Minus) {
        controls.throttle = (controls.throttle - throttle_rate * dt).max(0.0);
    }

    // --- Auto-Level / Wing-Leveler Autopilot Toggle: 'H' or 'T' ---
    if keyboard.just_pressed(KeyCode::KeyH) || keyboard.just_pressed(KeyCode::KeyT) {
        controls.auto_level = !controls.auto_level;
        if controls.auto_level {
            controls.target_alt = sim.physics.state.altitude();
        }
    }

    // --- Autopilot / Auto-Level Flight Stabilization Assist ---
    if controls.auto_level {
        let (roll, pitch, _) = sim.physics.state.euler_angles();
        let state = &sim.physics.state;

        // Wing leveler: roll wings back to level when stick is released
        if !manual_roll {
            let roll_cmd = (-0.75 * roll - 0.35 * state.p).clamp(-0.25, 0.25);
            controls.aileron = roll_cmd;
        }

        // Altitude & pitch hold: hold target altitude
        if !manual_pitch {
            let alt_error = controls.target_alt - state.altitude();
            let target_climb_pitch = (alt_error * 0.003).clamp(-0.12, 0.12);
            let pitch_error = target_climb_pitch - pitch;
            let pitch_cmd = (0.6 * pitch_error - 0.45 * state.q).clamp(-0.20, 0.20);
            controls.elevator = pitch_cmd;
        }
    }

    // --- Reset flight state: 'R' key ---
    if keyboard.just_pressed(KeyCode::KeyR) {
        let (trim_e, trim_t) = sim.physics.reset();
        controls.elevator = 0.0;
        controls.elevator_trim = trim_e;
        controls.aileron = 0.0;
        controls.rudder = 0.0;
        controls.flaps_deg = 0.0;
        controls.throttle = trim_t;
        controls.target_alt = 1000.0;
    }

    // --- Return to the main menu: 'Esc' key ---
    if keyboard.just_pressed(KeyCode::Escape) {
        next_state.set(AppState::MainMenu);
    }
}

/// Advance the 6-DOF physics and reposition the aircraft mesh + chase camera.
fn update_aircraft(
    controls: Res<FlightControls>,
    mut sim: ResMut<Sim>,
    mut query: Query<&mut Transform, With<AircraftRoot>>,
    mut cam_query: Query<&mut Transform, (With<ChaseCamera>, Without<AircraftRoot>)>,
) {
    // Total elevator deflection = manual stick + pitch trim tab
    let total_elevator = (controls.elevator + controls.elevator_trim).clamp(-0.35, 0.35);

    // Compute the total wind in the Earth NED frame (steady + turbulence),
    // then advance the 6-DOF simulation with that relative wind. Borrow the
    // three fields disjointly so the mutable wind advance can also read state.
    let Sim {
        physics,
        wind,
        last_wind,
    } = &mut *sim;
    let vt_air = physics.state.airspeed();
    let wind_earth = wind.total_wind(&physics.state, vt_air, DT);
    *last_wind = wind_earth;
    physics.step_6dof(
        total_elevator,
        controls.aileron,
        controls.rudder,
        controls.throttle,
        controls.flaps_deg.to_radians(),
        Some(&wind_earth),
        DT,
    );

    let state = &sim.physics.state;

    // Map NED Earth position to Bevy World position:
    let world_pos = Vec3::new(
        state.pos_x as f32,
        -state.pos_z as f32,
        state.pos_y as f32,
    );

    // Compute the aircraft body axes in Earth NED frame:
    let (fwd_ned, right_ned, down_ned) = state.body_axes_in_earth();

    // Convert each body axis vector into Bevy world coordinates:
    let fwd_bevy = Vec3::new(fwd_ned.x as f32, -fwd_ned.z as f32, fwd_ned.y as f32).normalize();
    let up_bevy = Vec3::new(-down_ned.x as f32, down_ned.z as f32, -down_ned.y as f32).normalize();
    let right_bevy = Vec3::new(right_ned.x as f32, -right_ned.z as f32, right_ned.y as f32).normalize();

    // In Bevy local mesh: +X = Forward, +Y = Up, +Z = Right
    let rot_mat = Mat3::from_cols(fwd_bevy, up_bevy, right_bevy);
    let quat = Quat::from_mat3(&rot_mat);

    // Update aircraft transform
    for mut tf in query.iter_mut() {
        tf.translation = world_pos;
        tf.rotation = quat;
    }

    // Chase camera behind (-X) and above (+Y) the aircraft, looking slightly ahead
    let chase_offset = Vec3::new(-38.0, 8.5, 0.0);
    for mut cam_tf in cam_query.iter_mut() {
        let pos = world_pos + quat * chase_offset;
        cam_tf.translation = pos;
        cam_tf.look_at(world_pos + quat * Vec3::new(25.0, 1.5, 0.0), up_bevy);
    }
}

/// Updates the on-screen heads-up display (HUD).
fn update_hud(
    sim: Res<Sim>,
    controls: Res<FlightControls>,
    mut text_query: Query<&mut Text, With<HudText>>,
) {
    let state = &sim.physics.state;
    let alt_m = state.altitude();
    let alt_ft = alt_m * 3.28084;
    let speed_tas_ms = state.true_airspeed(&sim.last_wind);
    let speed_tas_kts = speed_tas_ms * 1.94384;

    // Ground speed and horizontal wind relative to the aircraft.
    let speed_gs_ms = state.airspeed();
    let wind = sim.last_wind;
    let wind_ms = wind.norm();
    let wind_dir = (wind.y.atan2(wind.x).to_degrees() + 360.0) % 360.0;

    // Atmospheric calculations (1976 US Standard Atmosphere)
    let atm = flight_core::Atmosphere::at_altitude(alt_m);
    let mach = atm.mach_number(speed_tas_ms);
    let speed_ias_kts = atm.calibrated_airspeed(speed_tas_ms) * 1.94384;
    let q_dyn_pa = atm.dynamic_pressure(speed_tas_ms);
    let oat_c = atm.temperature_c;

    let alpha_deg = state.air_angle_of_attack(&sim.last_wind).to_degrees();
    let beta_deg = state.air_sideslip_angle(&sim.last_wind).to_degrees();

    let (roll, pitch, yaw) = state.euler_angles();
    let roll_deg = roll.to_degrees();
    let pitch_deg = pitch.to_degrees();
    let yaw_deg = (yaw.to_degrees() + 360.0) % 360.0;
    let climb_angle_deg = state.flight_path_angle().to_degrees();

    let aileron_deg = controls.aileron.to_degrees();
    let total_elev_deg = (controls.elevator + controls.elevator_trim).to_degrees();
    let rudder_deg = controls.rudder.to_degrees();
    let trim_deg = controls.elevator_trim.to_degrees();
    let flaps_deg = controls.flaps_deg;

    // Status flags
    let ap_status = if controls.auto_level { " [AUTOPILOT ON]" } else { "" };
    let stall_warning = if alpha_deg > 14.5 { " [! STALL !]" } else { "" };

    let flap_str = if flaps_deg < 1.0 {
        "UP (0°)".to_string()
    } else if flaps_deg < 20.0 {
        "TAKEOFF (15°)".to_string()
    } else {
        "LANDING (30°)".to_string()
    };

    for mut text in text_query.iter_mut() {
        **text = format!(
            "TACTICAL UAV DRONE TELEMETRY{ap_status}{stall_warning}\n\
             ------------------------------------\n\
             Altitude:       {alt_m:6.0} m ({alt_ft:6.0} ft)\n\
             Airspeed (TAS): {speed_tas_ms:6.1} m/s ({speed_tas_kts:5.0} kts)\n\
             Ground Speed:   {speed_gs_ms:6.1} m/s\n\
             Wind:           {wind_ms:5.1} m/s from {wind_dir:3.0}\u{00b0}\n\
             Airspeed (IAS): {speed_ias_kts:6.0} kts  (Mach {mach:4.2})\n\
             Dyn. Pressure:  {q_dyn_pa:6.0} Pa  (OAT: {oat_c:+4.1}\u{00b0}C)\n\
             AoA (\u{03b1}) / Slip (\u{03b2}):  {alpha_deg:+5.1}\u{00b0} / {beta_deg:+5.1}\u{00b0}\n\
             Pitch / Roll:   {pitch_deg:+5.1}\u{00b0} / {roll_deg:+5.1}\u{00b0}\n\
             Heading (Yaw):  {yaw_deg:5.1}\u{00b0} (Climb: {climb_angle_deg:+4.1}\u{00b0})\n\
             Throttle:       {throttle_pct:5.0}%  (Flaps: {flap_str})\n\
             Surfaces:       Ail: {aileron_deg:+4.1}\u{00b0} | Elev: {total_elev_deg:+4.1}\u{00b0} (Trim {trim_deg:+4.1}\u{00b0}) | Rud: {rudder_deg:+4.1}\u{00b0}\n\n\
             CONTROLS & DRONE SYSTEMS\n\
             ------------------------------------\n\
             Pitch:     [W] Down  /  [S] Up (or \u{2191}/\u{2193})\n\
             Roll:      [A] Left  /  [D] Right (or \u{2190}/\u{2192})\n\
             Rudder:    [Q] Left  /  [E] Right (or [Z]/[C])\n\
             Flaps:     [F] Deploy Flaps (0\u{00b0} \u{2192} 15\u{00b0} \u{2192} 30\u{00b0})\n\
             Trim:      [[] Nose Down  /  []] Nose Up\n\
             Throttle:  [Shift] Up / [Ctrl] Down\n\
             Autopilot: [H] or [T] (Toggle Level Flight Hold)\n\
             Reset:     [R] Reset to trimmed cruise",
            ap_status = ap_status,
            stall_warning = stall_warning,
            alt_m = alt_m,
            alt_ft = alt_ft,
            speed_tas_ms = speed_tas_ms,
            speed_tas_kts = speed_tas_kts,
            speed_ias_kts = speed_ias_kts,
            mach = mach,
            q_dyn_pa = q_dyn_pa,
            oat_c = oat_c,
            alpha_deg = alpha_deg,
            beta_deg = beta_deg,
            pitch_deg = pitch_deg,
            roll_deg = roll_deg,
            yaw_deg = yaw_deg,
            climb_angle_deg = climb_angle_deg,
            throttle_pct = controls.throttle * 100.0,
            flap_str = flap_str,
            aileron_deg = aileron_deg,
            total_elev_deg = total_elev_deg,
            trim_deg = trim_deg,
            rudder_deg = rudder_deg,
        );
    }
}

/// Print altitude (m) and airspeed (m/s) to the console once per second.
fn print_telemetry(
    sim: Res<Sim>,
    time: Res<Time>,
    mut timer: ResMut<TelemetryTimer>,
) {
    timer.0 += time.delta_secs();
    if timer.0 >= 1.0 {
        timer.0 = 0.0;
        let obs = sim.physics.state.to_observation_array();
        let alt = -obs[1];
        let airspeed = obs[2].hypot(obs[4]);
        println!("altitude: {:.1} m   airspeed: {:.1} m/s", alt, airspeed);
    }
}
