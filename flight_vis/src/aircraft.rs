use bevy::prelude::*;

/// Marker for the root aircraft entity that gets transformed by the simulator state.
#[derive(Component)]
pub struct AircraftRoot;

/// Marker for the rear pusher spinning propeller component.
#[derive(Component)]
pub struct Propeller;

/// Marker for Left Aileron control surface (outer wing trailing edge).
#[derive(Component)]
pub struct LeftAileron;

/// Marker for Right Aileron control surface (outer wing trailing edge).
#[derive(Component)]
pub struct RightAileron;

/// Marker for Left Inboard Flap (inner wing trailing edge).
#[derive(Component)]
pub struct LeftFlap;

/// Marker for Right Inboard Flap (inner wing trailing edge).
#[derive(Component)]
pub struct RightFlap;

/// Marker for Left V-Tail Ruddervator.
#[derive(Component)]
pub struct LeftRuddervator;

/// Marker for Right V-Tail Ruddervator.
#[derive(Component)]
pub struct RightRuddervator;

/// Spawns a modern tactical surveillance drone (UAV) model (MQ-9/Bayraktar style)
/// or automatically loads an external 3D model if placed in `assets/models/`.
pub fn spawn_aircraft(
    commands: &mut Commands,
    meshes: &mut ResMut<Assets<Mesh>>,
    materials: &mut ResMut<Assets<StandardMaterial>>,
    asset_server: &Res<AssetServer>,
) -> Entity {
    // 1. Check if user dropped a custom 3D model file in `assets/models/`
    let candidate_models = [
        "models/drone.glb",
        "models/aircraft.glb",
        "models/airplane.glb",
        "models/uav.glb",
        "models/drone.gltf",
    ];

    for rel_path in candidate_models {
        let full_path = format!("assets/{}", rel_path);
        if std::path::Path::new(&full_path).exists() {
            println!("-> Found custom 3D model: {}. Loading into simulator...", full_path);
            let scene_uri = format!("{}#Scene0", rel_path);
            return commands
                .spawn((
                    AircraftRoot,
                    SceneRoot(asset_server.load(scene_uri)),
                    Transform::default(),
                    Visibility::default(),
                ))
                .id();
        }
    }

    // 2. Built-in Procedural Tactical Military Drone Mesh
    let body_mat = materials.add(StandardMaterial {
        base_color: Color::srgb(0.72, 0.76, 0.80), // Tactical matte stealth gray
        metallic: 0.15,
        perceptual_roughness: 0.5,
        ..default()
    });

    let radome_mat = materials.add(StandardMaterial {
        base_color: Color::srgb(0.85, 0.88, 0.92), // Light composite satcom dome
        metallic: 0.1,
        perceptual_roughness: 0.4,
        ..default()
    });

    let dark_trim_mat = materials.add(StandardMaterial {
        base_color: Color::srgb(0.18, 0.20, 0.24), // Carbon-fiber leading edges / exhaust
        metallic: 0.4,
        perceptual_roughness: 0.35,
        ..default()
    });

    let control_mat = materials.add(StandardMaterial {
        base_color: Color::srgb(0.55, 0.59, 0.64), // Slightly darker contrast for moving surfaces
        metallic: 0.2,
        perceptual_roughness: 0.45,
        ..default()
    });

    let gimbal_turret_mat = materials.add(StandardMaterial {
        base_color: Color::srgb(0.12, 0.12, 0.14), // Dark anodized gimbal ball
        metallic: 0.7,
        perceptual_roughness: 0.25,
        ..default()
    });

    let sensor_lens_mat = materials.add(StandardMaterial {
        base_color: Color::srgb(0.05, 0.45, 0.65), // Optical coated FLIR lens
        metallic: 0.9,
        perceptual_roughness: 0.1,
        ..default()
    });

    let prop_mat = materials.add(StandardMaterial {
        base_color: Color::srgb(0.1, 0.1, 0.1), // Black composite propeller
        metallic: 0.6,
        perceptual_roughness: 0.3,
        ..default()
    });

    let nav_red = materials.add(StandardMaterial {
        base_color: Color::srgb(1.0, 0.1, 0.1),
        emissive: LinearRgba::new(4.0, 0.2, 0.2, 1.0),
        ..default()
    });

    let nav_green = materials.add(StandardMaterial {
        base_color: Color::srgb(0.1, 1.0, 0.2),
        emissive: LinearRgba::new(0.2, 4.0, 0.4, 1.0),
        ..default()
    });

    // Mesh assets
    let main_fuselage_mesh = meshes.add(Cuboid::new(4.2, 0.75, 0.75));
    let nose_radome_mesh = meshes.add(Sphere::new(0.48).mesh().ico(4).unwrap());
    let upper_satcom_hump = meshes.add(Cuboid::new(1.8, 0.42, 0.62));
    let sensor_ball_mesh = meshes.add(Sphere::new(0.24).mesh().ico(3).unwrap());
    let sensor_lens_mesh = meshes.add(Cylinder::new(0.10, 0.08));
    let pitot_tube_mesh = meshes.add(Cylinder::new(0.015, 0.6));

    // Wing meshes (High aspect ratio UAV wings, 11.2m span)
    let main_wing_mesh = meshes.add(Cuboid::new(0.85, 0.09, 10.6));
    let wing_leading_edge = meshes.add(Cuboid::new(0.20, 0.07, 10.6));
    let winglet_mesh = meshes.add(Cuboid::new(0.40, 0.35, 0.06));
    let nav_light_mesh = meshes.add(Sphere::new(0.04).mesh().ico(2).unwrap());

    // Movable Control Surfaces (hinge offsets positioned at leading edge of flap/aileron)
    let flap_mesh = meshes.add(Cuboid::new(0.32, 0.06, 2.2));
    let aileron_mesh = meshes.add(Cuboid::new(0.32, 0.06, 2.6));

    // V-Tail fins (Inverted or V-Tail configuration)
    let v_tail_fin_mesh = meshes.add(Cuboid::new(0.75, 1.35, 0.08));
    let ruddervator_mesh = meshes.add(Cuboid::new(0.28, 1.25, 0.06));

    // Pusher engine spinner and prop
    let pusher_spinner_mesh = meshes.add(Cylinder::new(0.20, 0.30));
    let pusher_blade_mesh = meshes.add(Cuboid::new(0.04, 1.45, 0.12));
    let engine_scoop_mesh = meshes.add(Cuboid::new(0.8, 0.25, 0.35));

    // Spawn drone root hierarchy
    let drone = commands
        .spawn((
            AircraftRoot,
            Transform::default(),
            Visibility::default(),
        ))
        .with_children(|parent| {
            // --- 1. Fuselage & SATCOM Bulge ---
            parent.spawn((
                Mesh3d(main_fuselage_mesh),
                MeshMaterial3d(body_mat.clone()),
                Transform::from_xyz(0.0, 0.0, 0.0),
            ));

            // Bulbous nose SATCOM dome
            parent.spawn((
                Mesh3d(nose_radome_mesh),
                MeshMaterial3d(radome_mat.clone()),
                Transform::from_xyz(2.1, 0.12, 0.0).with_scale(Vec3::new(1.3, 1.05, 1.0)),
            ));

            // SATCOM top fairing hump
            parent.spawn((
                Mesh3d(upper_satcom_hump),
                MeshMaterial3d(radome_mat.clone()),
                Transform::from_xyz(0.9, 0.42, 0.0),
            ));

            // --- 2. Chin Gimbal EO/IR Camera Turret ---
            parent.spawn((
                Mesh3d(sensor_ball_mesh),
                MeshMaterial3d(gimbal_turret_mat),
                Transform::from_xyz(1.7, -0.42, 0.0),
            ));
            // Sensor Optical Lens (pointing forward-down)
            parent.spawn((
                Mesh3d(sensor_lens_mesh),
                MeshMaterial3d(sensor_lens_mat),
                Transform::from_xyz(1.86, -0.46, 0.0)
                    .with_rotation(Quat::from_rotation_z(std::f32::consts::FRAC_PI_2 - 0.2)),
            ));

            // Nose Pitot Tube Probe
            parent.spawn((
                Mesh3d(pitot_tube_mesh),
                MeshMaterial3d(dark_trim_mat.clone()),
                Transform::from_xyz(2.85, 0.0, 0.0)
                    .with_rotation(Quat::from_rotation_z(std::f32::consts::FRAC_PI_2)),
            ));

            // Top Engine Air Intake Scoop
            parent.spawn((
                Mesh3d(engine_scoop_mesh),
                MeshMaterial3d(dark_trim_mat.clone()),
                Transform::from_xyz(-1.1, 0.46, 0.0),
            ));

            // --- 3. Main High-Aspect-Ratio Wings ---
            parent.spawn((
                Mesh3d(main_wing_mesh),
                MeshMaterial3d(body_mat.clone()),
                Transform::from_xyz(0.25, 0.15, 0.0),
            ));

            // Carbon-fiber Leading Edge Spars
            parent.spawn((
                Mesh3d(wing_leading_edge),
                MeshMaterial3d(dark_trim_mat.clone()),
                Transform::from_xyz(0.68, 0.15, 0.0),
            ));

            // Winglets (angled wingtips)
            parent.spawn((
                Mesh3d(winglet_mesh.clone()),
                MeshMaterial3d(dark_trim_mat.clone()),
                Transform::from_xyz(0.25, 0.30, 5.3)
                    .with_rotation(Quat::from_rotation_x(-0.25)),
            ));
            parent.spawn((
                Mesh3d(winglet_mesh),
                MeshMaterial3d(dark_trim_mat.clone()),
                Transform::from_xyz(0.25, 0.30, -5.3)
                    .with_rotation(Quat::from_rotation_x(0.25)),
            ));

            // Wingtip Navigation / Strobe Lights
            parent.spawn((
                Mesh3d(nav_light_mesh.clone()),
                MeshMaterial3d(nav_red),
                Transform::from_xyz(0.25, 0.18, 5.32),
            ));
            parent.spawn((
                Mesh3d(nav_light_mesh),
                MeshMaterial3d(nav_green),
                Transform::from_xyz(0.25, 0.18, -5.32),
            ));

            // --- 4. Trailing Edge Movable Surfaces (Flaps & Ailerons) ---
            // Left Inboard Flap
            parent
                .spawn((
                    LeftFlap,
                    Transform::from_xyz(-0.35, 0.14, 1.8),
                    Visibility::default(),
                ))
                .with_children(|f_parent| {
                    f_parent.spawn((
                        Mesh3d(flap_mesh.clone()),
                        MeshMaterial3d(control_mat.clone()),
                        Transform::from_xyz(-0.16, 0.0, 0.0),
                    ));
                });

            // Right Inboard Flap
            parent
                .spawn((
                    RightFlap,
                    Transform::from_xyz(-0.35, 0.14, -1.8),
                    Visibility::default(),
                ))
                .with_children(|f_parent| {
                    f_parent.spawn((
                        Mesh3d(flap_mesh),
                        MeshMaterial3d(control_mat.clone()),
                        Transform::from_xyz(-0.16, 0.0, 0.0),
                    ));
                });

            // Left Outboard Aileron
            parent
                .spawn((
                    LeftAileron,
                    Transform::from_xyz(-0.35, 0.14, 4.0),
                    Visibility::default(),
                ))
                .with_children(|a_parent| {
                    a_parent.spawn((
                        Mesh3d(aileron_mesh.clone()),
                        MeshMaterial3d(control_mat.clone()),
                        Transform::from_xyz(-0.16, 0.0, 0.0),
                    ));
                });

            // Right Outboard Aileron
            parent
                .spawn((
                    RightAileron,
                    Transform::from_xyz(-0.35, 0.14, -4.0),
                    Visibility::default(),
                ))
                .with_children(|a_parent| {
                    a_parent.spawn((
                        Mesh3d(aileron_mesh),
                        MeshMaterial3d(control_mat.clone()),
                        Transform::from_xyz(-0.16, 0.0, 0.0),
                    ));
                });

            // --- 5. V-Tail Empennage (Cant angle ~38 degrees) ---
            let v_angle = 38.0_f32.to_radians();
            parent.spawn((
                Mesh3d(v_tail_fin_mesh.clone()),
                MeshMaterial3d(body_mat.clone()),
                Transform::from_xyz(-1.85, 0.65, 0.55)
                    .with_rotation(Quat::from_rotation_x(v_angle)),
            ));
            parent
                .spawn((
                    LeftRuddervator,
                    Transform::from_xyz(-2.25, 0.65, 0.55)
                        .with_rotation(Quat::from_rotation_x(v_angle)),
                    Visibility::default(),
                ))
                .with_children(|rv_parent| {
                    rv_parent.spawn((
                        Mesh3d(ruddervator_mesh.clone()),
                        MeshMaterial3d(control_mat.clone()),
                        Transform::from_xyz(-0.14, 0.0, 0.0),
                    ));
                });

            parent.spawn((
                Mesh3d(v_tail_fin_mesh),
                MeshMaterial3d(body_mat.clone()),
                Transform::from_xyz(-1.85, 0.65, -0.55)
                    .with_rotation(Quat::from_rotation_x(-v_angle)),
            ));
            parent
                .spawn((
                    RightRuddervator,
                    Transform::from_xyz(-2.25, 0.65, -0.55)
                        .with_rotation(Quat::from_rotation_x(-v_angle)),
                    Visibility::default(),
                ))
                .with_children(|rv_parent| {
                    rv_parent.spawn((
                        Mesh3d(ruddervator_mesh),
                        MeshMaterial3d(control_mat.clone()),
                        Transform::from_xyz(-0.14, 0.0, 0.0),
                    ));
                });

            // --- 6. Rear Pusher Engine Propeller Assembly ---
            parent.spawn((
                Mesh3d(pusher_spinner_mesh),
                MeshMaterial3d(dark_trim_mat),
                Transform::from_xyz(-2.25, 0.08, 0.0)
                    .with_rotation(Quat::from_rotation_z(std::f32::consts::FRAC_PI_2)),
            ));

            parent
                .spawn((
                    Propeller,
                    Transform::from_xyz(-2.38, 0.08, 0.0),
                    Visibility::default(),
                ))
                .with_children(|prop_parent| {
                    prop_parent.spawn((
                        Mesh3d(pusher_blade_mesh),
                        MeshMaterial3d(prop_mat),
                        Transform::default(),
                    ));
                });
        })
        .id();

    drone
}

/// System to animate the rear pusher propeller rotation according to engine throttle.
pub fn spin_propeller(
    time: Res<Time>,
    controls: Res<crate::FlightControls>,
    mut query: Query<&mut Transform, With<Propeller>>,
) {
    let base_speed = 25.0; // rad/s at idle
    let speed = base_speed + (controls.throttle as f32) * 55.0;
    for mut tf in query.iter_mut() {
        tf.rotate_x(speed * time.delta_secs());
    }
}

/// System to animate all drone control surfaces (flaps, ailerons, and V-tail ruddervators)
/// in real time responding to user inputs and autopilot commands.
pub fn animate_control_surfaces(
    controls: Res<crate::FlightControls>,
    time: Res<Time>,
    mut left_aileron_q: Query<&mut Transform, (With<LeftAileron>, Without<RightAileron>, Without<LeftFlap>, Without<RightFlap>, Without<LeftRuddervator>, Without<RightRuddervator>)>,
    mut right_aileron_q: Query<&mut Transform, (With<RightAileron>, Without<LeftAileron>, Without<LeftFlap>, Without<RightFlap>, Without<LeftRuddervator>, Without<RightRuddervator>)>,
    mut left_flap_q: Query<&mut Transform, (With<LeftFlap>, Without<LeftAileron>, Without<RightAileron>, Without<RightFlap>, Without<LeftRuddervator>, Without<RightRuddervator>)>,
    mut right_flap_q: Query<&mut Transform, (With<RightFlap>, Without<LeftAileron>, Without<RightAileron>, Without<LeftFlap>, Without<LeftRuddervator>, Without<RightRuddervator>)>,
    mut left_ruddervator_q: Query<&mut Transform, (With<LeftRuddervator>, Without<LeftAileron>, Without<RightAileron>, Without<LeftFlap>, Without<RightFlap>, Without<RightRuddervator>)>,
    mut right_ruddervator_q: Query<&mut Transform, (With<RightRuddervator>, Without<LeftAileron>, Without<RightAileron>, Without<LeftFlap>, Without<RightFlap>, Without<LeftRuddervator>)>,
) {
    let lerp_rate = (15.0 * time.delta_secs()).min(1.0);

    // 1. Ailerons: differential deflection around lateral Z axis
    let aileron_cmd = controls.aileron as f32;
    for mut tf in left_aileron_q.iter_mut() {
        let target_rot = Quat::from_rotation_z(aileron_cmd * 1.1);
        tf.rotation = tf.rotation.slerp(target_rot, lerp_rate);
    }
    for mut tf in right_aileron_q.iter_mut() {
        let target_rot = Quat::from_rotation_z(-aileron_cmd * 1.1);
        tf.rotation = tf.rotation.slerp(target_rot, lerp_rate);
    }

    // 2. Flaps: trailing edge downward deflection
    let flap_angle = (controls.flaps_deg as f32).to_radians();
    for mut tf in left_flap_q.iter_mut() {
        let target_rot = Quat::from_rotation_z(flap_angle);
        tf.rotation = tf.rotation.slerp(target_rot, lerp_rate);
    }
    for mut tf in right_flap_q.iter_mut() {
        let target_rot = Quat::from_rotation_z(flap_angle);
        tf.rotation = tf.rotation.slerp(target_rot, lerp_rate);
    }

    // 3. V-Tail Ruddervators: combined pitch (elevator) and yaw (rudder) mixing
    let elev_cmd = (controls.elevator + controls.elevator_trim) as f32;
    let rud_cmd = controls.rudder as f32;

    let left_rv_angle = -elev_cmd * 1.0 + rud_cmd * 0.8;
    let right_rv_angle = -elev_cmd * 1.0 - rud_cmd * 0.8;

    let v_angle = 38.0_f32.to_radians();
    for mut tf in left_ruddervator_q.iter_mut() {
        let base_rot = Quat::from_rotation_x(v_angle);
        let deflection = Quat::from_rotation_z(left_rv_angle);
        let target_rot = base_rot * deflection;
        tf.rotation = tf.rotation.slerp(target_rot, lerp_rate);
    }
    for mut tf in right_ruddervator_q.iter_mut() {
        let base_rot = Quat::from_rotation_x(-v_angle);
        let deflection = Quat::from_rotation_z(right_rv_angle);
        let target_rot = base_rot * deflection;
        tf.rotation = tf.rotation.slerp(target_rot, lerp_rate);
    }
}
