use bevy::prelude::*;

/// Marker for wind turbine rotor assemblies that rotate over time.
#[derive(Component)]
pub struct WindTurbineRotor {
    pub speed: f32,
}

/// Marker for the root of the entire environment scene. Everything spawned by
/// [`spawn_environment`] is a descendant of this single entity, so the whole
/// world can be shown/hidden as one unit (e.g. hidden while in the wind
/// tunnel).
#[derive(Component)]
pub struct EnvironmentRoot;

/// Spawns the entire flight simulator environment:
/// - Real-world custom 3D terrain (if placed in `assets/models/terrain.glb` or `landscape.glb`)
/// - Satellite imagery ground texture (if placed in `assets/textures/satellite.png`)
/// - Vast procedural terrain and patchwork farmland
/// - Winding river
/// - Full airport complex (runway, markings, runway edge/threshold lights, taxiways, tower, hangars, terminal)
/// - Downtown city with skyscrapers, towers, and spires
/// - Left and right mountain ranges with snow caps
/// - Pine forests and tree groves
/// - Wind turbine farm with animated rotors
/// - 3D low-poly cloud formations at flight altitude
///
/// Returns the root [`EnvironmentRoot`] entity so callers can toggle its
/// [`Visibility`].
pub fn spawn_environment(
    commands: &mut Commands,
    meshes: &mut ResMut<Assets<Mesh>>,
    materials: &mut ResMut<Assets<StandardMaterial>>,
    asset_server: &Res<AssetServer>,
) -> Entity {
    commands
        .spawn((
            EnvironmentRoot,
            Transform::default(),
            Visibility::default(),
        ))
        .with_children(|parent| {
            // 1. Check if user provided a real-world 3D terrain model.
            let mut custom_terrain_loaded = false;
            for rel_path in [
                "models/terrain.glb",
                "models/landscape.glb",
                "models/mountains.glb",
                "models/real_world_terrain.glb",
                "models/terrain.gltf",
            ] {
                let full_path = format!("assets/{}", rel_path);
                if std::path::Path::new(&full_path).exists() {
                    println!("-> Loading real-world 3D terrain from {}", full_path);
                    parent.spawn((
                        SceneRoot(asset_server.load(format!("{}#Scene0", rel_path))),
                        Transform::from_xyz(0.0, 0.0, 0.0),
                    ));
                    custom_terrain_loaded = true;
                    break;
                }
            }

            spawn_terrain_and_fields(
                parent,
                meshes,
                materials,
                asset_server,
                custom_terrain_loaded,
            );
            spawn_river(parent, meshes, materials);
            spawn_airport(parent, meshes, materials);
            spawn_city(parent, meshes, materials);
            if !custom_terrain_loaded {
                spawn_mountains(parent, meshes, materials);
            }
            spawn_forests(parent, meshes, materials);
            spawn_wind_turbines(parent, meshes, materials);
            spawn_clouds(parent, meshes, materials);
        })
        .id()
}

// ---------------------------------------------------------------------------
// 1. Terrain & Farmland
// ---------------------------------------------------------------------------
fn spawn_terrain_and_fields(
    parent: &mut ChildBuilder,
    meshes: &mut ResMut<Assets<Mesh>>,
    materials: &mut ResMut<Assets<StandardMaterial>>,
    asset_server: &Res<AssetServer>,
    custom_terrain_loaded: bool,
) {
    // Vast base ground plane (50km x 50km)
    let ground_mesh = meshes.add(Plane3d::default().mesh().size(50000.0, 50000.0));

    // Check if user placed a real-world satellite ground map in assets/textures/
    let sat_candidates = [
        "textures/satellite.png",
        "textures/satellite.jpg",
        "textures/satellite_map.png",
        "textures/orthophoto.png",
    ];

    let mut sat_texture = None;
    for rel_path in sat_candidates {
        let full_path = format!("assets/{}", rel_path);
        if std::path::Path::new(&full_path).exists() {
            println!("-> Applying real-world satellite texture from {}", full_path);
            sat_texture = Some(asset_server.load(rel_path));
            break;
        }
    }

    let ground_mat = materials.add(StandardMaterial {
        base_color: if sat_texture.is_some() {
            Color::WHITE
        } else {
            Color::srgb(0.22, 0.48, 0.24) // Rich landscape green
        },
        base_color_texture: sat_texture.clone(),
        perceptual_roughness: 0.9,
        ..default()
    });

    parent.spawn((
        Mesh3d(ground_mesh),
        MeshMaterial3d(ground_mat),
        Transform::from_xyz(10000.0, 0.0, 0.0),
    ));

    // If no satellite texture is present and no custom terrain loaded, spawn procedural farmland patches
    if sat_texture.is_none() && !custom_terrain_loaded {
        let field_colors = [
            Color::srgb(0.28, 0.52, 0.26), // Deep pasture green
            Color::srgb(0.55, 0.48, 0.28), // Golden wheat / barley
            Color::srgb(0.40, 0.50, 0.22), // Olive crop
            Color::srgb(0.45, 0.38, 0.25), // Tilled loam / soil
            Color::srgb(0.35, 0.58, 0.32), // Bright meadow
        ];

        let field_materials: Vec<Handle<StandardMaterial>> = field_colors
            .iter()
            .map(|&c| {
                materials.add(StandardMaterial {
                    base_color: c,
                    perceptual_roughness: 0.95,
                    ..default()
                })
            })
            .collect();

        // Spawn 60 field patches scattered across the valley floor
        for i in 0..60 {
            let x = ((i * 397) % 24000) as f32 - 3000.0;
            let z = if i % 2 == 0 {
                -400.0 - ((i * 173) % 1800) as f32
            } else {
                400.0 + ((i * 227) % 1800) as f32
            };
            let width = 250.0 + ((i * 71) % 400) as f32;
            let length = 300.0 + ((i * 97) % 500) as f32;
            let mat_idx = i % field_materials.len();

            let field_mesh = meshes.add(Plane3d::default().mesh().size(width, length));
            parent.spawn((
                Mesh3d(field_mesh),
                MeshMaterial3d(field_materials[mat_idx].clone()),
                Transform::from_xyz(x, 0.2, z),
            ));
        }
    }
}

// ---------------------------------------------------------------------------
// 2. River
// ---------------------------------------------------------------------------
fn spawn_river(
    parent: &mut ChildBuilder,
    meshes: &mut ResMut<Assets<Mesh>>,
    materials: &mut ResMut<Assets<StandardMaterial>>,
) {
    let water_mat = materials.add(StandardMaterial {
        base_color: Color::srgba(0.12, 0.35, 0.65, 0.95),
        metallic: 0.8,
        perceptual_roughness: 0.15,
        ..default()
    });

    let segments = [
        (-4000.0, -1200.0, 3000.0, 140.0, 0.12),
        (-1000.0, -900.0, 3200.0, 150.0, -0.15),
        (2200.0, -1100.0, 3400.0, 160.0, 0.08),
        (5600.0, -850.0, 3600.0, 170.0, -0.18),
        (9200.0, -1250.0, 3800.0, 180.0, 0.14),
        (13000.0, -900.0, 4000.0, 190.0, -0.10),
        (17000.0, -1100.0, 4200.0, 200.0, 0.05),
        (21200.0, -950.0, 4500.0, 220.0, -0.12),
    ];

    for (x, z, len, width, rot) in segments {
        let river_mesh = meshes.add(Plane3d::default().mesh().size(len, width));
        parent.spawn((
            Mesh3d(river_mesh),
            MeshMaterial3d(water_mat.clone()),
            Transform::from_xyz(x, 0.4, z).with_rotation(Quat::from_rotation_y(rot)),
        ));
    }
}

// ---------------------------------------------------------------------------
// 3. Airport Complex
// ---------------------------------------------------------------------------
fn spawn_airport(
    parent: &mut ChildBuilder,
    meshes: &mut ResMut<Assets<Mesh>>,
    materials: &mut ResMut<Assets<StandardMaterial>>,
) {
    let asphalt_mat = materials.add(StandardMaterial {
        base_color: Color::srgb(0.18, 0.18, 0.20),
        perceptual_roughness: 0.8,
        ..default()
    });

    let taxiway_mat = materials.add(StandardMaterial {
        base_color: Color::srgb(0.24, 0.24, 0.26),
        perceptual_roughness: 0.85,
        ..default()
    });

    let marking_white = materials.add(StandardMaterial {
        base_color: Color::srgb(0.95, 0.95, 0.95),
        unlit: true,
        ..default()
    });

    let marking_yellow = materials.add(StandardMaterial {
        base_color: Color::srgb(0.95, 0.8, 0.1),
        unlit: true,
        ..default()
    });

    let building_concrete = materials.add(StandardMaterial {
        base_color: Color::srgb(0.72, 0.72, 0.75),
        perceptual_roughness: 0.6,
        ..default()
    });

    let glass_mat = materials.add(StandardMaterial {
        base_color: Color::srgba(0.2, 0.4, 0.6, 0.8),
        metallic: 0.9,
        perceptual_roughness: 0.1,
        ..default()
    });

    let metal_roof = materials.add(StandardMaterial {
        base_color: Color::srgb(0.3, 0.45, 0.6),
        metallic: 0.7,
        perceptual_roughness: 0.3,
        ..default()
    });

    let runway_light_green = materials.add(StandardMaterial {
        base_color: Color::srgb(0.1, 1.0, 0.2),
        emissive: LinearRgba::rgb(4.0, 15.0, 4.0),
        ..default()
    });

    let runway_light_red = materials.add(StandardMaterial {
        base_color: Color::srgb(1.0, 0.1, 0.1),
        emissive: LinearRgba::rgb(15.0, 2.0, 2.0),
        ..default()
    });

    let runway_light_white = materials.add(StandardMaterial {
        base_color: Color::srgb(1.0, 1.0, 0.9),
        emissive: LinearRgba::rgb(10.0, 10.0, 8.0),
        ..default()
    });

    let airport_x = 0.0;
    let airport_z = 0.0;

    // Runway 09/27 (3200m length x 60m width)
    let runway_len = 3200.0;
    let runway_width = 60.0;
    let runway_mesh = meshes.add(Plane3d::default().mesh().size(runway_len, runway_width));
    parent.spawn((
        Mesh3d(runway_mesh),
        MeshMaterial3d(asphalt_mat),
        Transform::from_xyz(airport_x, 0.5, airport_z),
    ));

    // Centerline stripes (every 50m)
    let stripe_mesh = meshes.add(Plane3d::default().mesh().size(30.0, 3.0));
    for i in -28..29 {
        let x = airport_x + (i as f32) * 50.0;
        parent.spawn((
            Mesh3d(stripe_mesh.clone()),
            MeshMaterial3d(marking_white.clone()),
            Transform::from_xyz(x, 0.52, airport_z),
        ));
    }

    // Runway Edge Lights
    let bulb_mesh = meshes.add(Sphere::new(0.4));
    for i in -32..=32 {
        let x = airport_x + (i as f32) * 50.0;
        parent.spawn((
            Mesh3d(bulb_mesh.clone()),
            MeshMaterial3d(runway_light_white.clone()),
            Transform::from_xyz(x, 0.8, airport_z - 31.0),
        ));
        parent.spawn((
            Mesh3d(bulb_mesh.clone()),
            MeshMaterial3d(runway_light_white.clone()),
            Transform::from_xyz(x, 0.8, airport_z + 31.0),
        ));
    }

    // Threshold Lights (Green for Approach, Red for End)
    for j in -5..=5 {
        let z = airport_z + (j as f32) * 5.0;
        parent.spawn((
            Mesh3d(bulb_mesh.clone()),
            MeshMaterial3d(runway_light_green.clone()),
            Transform::from_xyz(airport_x - 1600.0, 0.8, z),
        ));
        parent.spawn((
            Mesh3d(bulb_mesh.clone()),
            MeshMaterial3d(runway_light_red.clone()),
            Transform::from_xyz(airport_x + 1600.0, 0.8, z),
        ));
    }

    // Parallel Taxiway
    let taxiway_z = airport_z - 110.0;
    let taxiway_mesh = meshes.add(Plane3d::default().mesh().size(3200.0, 26.0));
    parent.spawn((
        Mesh3d(taxiway_mesh),
        MeshMaterial3d(taxiway_mat.clone()),
        Transform::from_xyz(airport_x, 0.48, taxiway_z),
    ));

    // Taxiway Centerline (Yellow)
    let taxiline_mesh = meshes.add(Plane3d::default().mesh().size(3200.0, 1.2));
    parent.spawn((
        Mesh3d(taxiline_mesh),
        MeshMaterial3d(marking_yellow),
        Transform::from_xyz(airport_x, 0.49, taxiway_z),
    ));

    // Taxiway Connectors
    for offset_x in [-1000.0, 0.0, 1000.0] {
        let connector_mesh = meshes.add(Plane3d::default().mesh().size(26.0, 110.0));
        parent.spawn((
            Mesh3d(connector_mesh),
            MeshMaterial3d(taxiway_mat.clone()),
            Transform::from_xyz(airport_x + offset_x, 0.48, airport_z - 55.0),
        ));
    }

    // Apron / Tarmac
    let apron_mesh = meshes.add(Plane3d::default().mesh().size(700.0, 180.0));
    parent.spawn((
        Mesh3d(apron_mesh),
        MeshMaterial3d(taxiway_mat),
        Transform::from_xyz(airport_x, 0.46, taxiway_z - 100.0),
    ));

    // ATC Tower
    let tower_base = meshes.add(Cuboid::new(22.0, 15.0, 22.0));
    let tower_shaft = meshes.add(Cylinder::new(6.5, 65.0));
    let tower_cab = meshes.add(Cylinder::new(11.0, 10.0));
    let tower_radome = meshes.add(Sphere::new(4.5));
    let tower_x = airport_x - 200.0;
    let tower_z = taxiway_z - 160.0;

    parent.spawn((
        Mesh3d(tower_base),
        MeshMaterial3d(building_concrete.clone()),
        Transform::from_xyz(tower_x, 7.5, tower_z),
    ));
    parent.spawn((
        Mesh3d(tower_shaft),
        MeshMaterial3d(building_concrete.clone()),
        Transform::from_xyz(tower_x, 47.5, tower_z),
    ));
    parent.spawn((
        Mesh3d(tower_cab),
        MeshMaterial3d(glass_mat.clone()),
        Transform::from_xyz(tower_x, 82.0, tower_z),
    ));
    parent.spawn((
        Mesh3d(tower_radome),
        MeshMaterial3d(marking_white.clone()),
        Transform::from_xyz(tower_x, 90.0, tower_z),
    ));

    // Aircraft Hangars (3 units)
    for h in 0..3 {
        let h_x = airport_x + 100.0 + (h as f32) * 160.0;
        let h_z = taxiway_z - 160.0;
        let hangar_body = meshes.add(Cuboid::new(120.0, 25.0, 80.0));
        let hangar_roof = meshes.add(Cylinder::new(42.0, 120.0));

        parent.spawn((
            Mesh3d(hangar_body),
            MeshMaterial3d(building_concrete.clone()),
            Transform::from_xyz(h_x, 12.5, h_z),
        ));
        parent.spawn((
            Mesh3d(hangar_roof),
            MeshMaterial3d(metal_roof.clone()),
            Transform::from_xyz(h_x, 25.0, h_z)
                .with_rotation(Quat::from_rotation_z(std::f32::consts::FRAC_PI_2)),
        ));
    }

    // Terminal Building
    let term_x = airport_x - 50.0;
    let term_z = taxiway_z - 220.0;
    let term_mesh = meshes.add(Cuboid::new(260.0, 22.0, 70.0));
    parent.spawn((
        Mesh3d(term_mesh),
        MeshMaterial3d(building_concrete),
        Transform::from_xyz(term_x, 11.0, term_z),
    ));

    // Windsock
    let pole_mesh = meshes.add(Cylinder::new(0.3, 14.0));
    let sock_mesh = meshes.add(Cone {
        radius: 1.8,
        height: 5.5,
    });
    let orange_mat = materials.add(StandardMaterial {
        base_color: Color::srgb(1.0, 0.4, 0.0),
        ..default()
    });

    let sock_x = airport_x - 600.0;
    let sock_z = airport_z + 60.0;
    parent.spawn((
        Mesh3d(pole_mesh),
        MeshMaterial3d(marking_white),
        Transform::from_xyz(sock_x, 7.0, sock_z),
    ));
    parent.spawn((
        Mesh3d(sock_mesh),
        MeshMaterial3d(orange_mat),
        Transform::from_xyz(sock_x + 2.0, 14.0, sock_z)
            .with_rotation(Quat::from_rotation_z(std::f32::consts::FRAC_PI_2)),
    ));
}

// ---------------------------------------------------------------------------
// 4. Downtown City / Skyscrapers
// ---------------------------------------------------------------------------
fn spawn_city(
    parent: &mut ChildBuilder,
    meshes: &mut ResMut<Assets<Mesh>>,
    materials: &mut ResMut<Assets<StandardMaterial>>,
) {
    let glass_blue = materials.add(StandardMaterial {
        base_color: Color::srgb(0.2, 0.35, 0.55),
        metallic: 0.85,
        perceptual_roughness: 0.15,
        ..default()
    });

    let glass_cyan = materials.add(StandardMaterial {
        base_color: Color::srgb(0.18, 0.45, 0.52),
        metallic: 0.9,
        perceptual_roughness: 0.12,
        ..default()
    });

    let glass_dark = materials.add(StandardMaterial {
        base_color: Color::srgb(0.12, 0.16, 0.22),
        metallic: 0.9,
        perceptual_roughness: 0.1,
        ..default()
    });

    let stone_grey = materials.add(StandardMaterial {
        base_color: Color::srgb(0.65, 0.65, 0.68),
        perceptual_roughness: 0.7,
        ..default()
    });

    let warm_brick = materials.add(StandardMaterial {
        base_color: Color::srgb(0.55, 0.38, 0.32),
        perceptual_roughness: 0.85,
        ..default()
    });

    let beacon_red = materials.add(StandardMaterial {
        base_color: Color::srgb(1.0, 0.1, 0.1),
        emissive: LinearRgba::rgb(5.0, 0.2, 0.2),
        ..default()
    });

    let road_mat = materials.add(StandardMaterial {
        base_color: Color::srgb(0.25, 0.25, 0.27),
        perceptual_roughness: 0.9,
        ..default()
    });

    let city_center_x = 5500.0;
    let city_center_z = -1800.0;

    let city_base = meshes.add(Plane3d::default().mesh().size(3200.0, 2400.0));
    parent.spawn((
        Mesh3d(city_base),
        MeshMaterial3d(road_mat),
        Transform::from_xyz(city_center_x, 0.3, city_center_z),
    ));

    let bldg_materials = [glass_blue, glass_cyan, glass_dark, stone_grey, warm_brick];
    let beacon_mesh = meshes.add(Sphere::new(1.2));

    for row in 0..8 {
        for col in 0..6 {
            let bx = city_center_x - 1200.0 + (row as f32) * 340.0;
            let bz = city_center_z - 900.0 + (col as f32) * 320.0;

            let seed = (row * 13 + col * 29) as f32;
            let width = 50.0 + ((seed * 17.0) % 45.0);
            let depth = 50.0 + ((seed * 23.0) % 45.0);
            let height = 70.0 + ((seed * 43.0) % 360.0);

            let mat_idx = ((seed as usize) + row + col) % bldg_materials.len();

            let bldg_mesh = meshes.add(Cuboid::new(width, height, depth));
            parent.spawn((
                Mesh3d(bldg_mesh),
                MeshMaterial3d(bldg_materials[mat_idx].clone()),
                Transform::from_xyz(bx, height * 0.5, bz),
            ));

            if height > 160.0 {
                let spire_height = 35.0;
                let spire_mesh = meshes.add(Cone {
                    radius: 2.0,
                    height: spire_height,
                });
                parent.spawn((
                    Mesh3d(spire_mesh),
                    MeshMaterial3d(bldg_materials[0].clone()),
                    Transform::from_xyz(bx, height + spire_height * 0.5, bz),
                ));
                parent.spawn((
                    Mesh3d(beacon_mesh.clone()),
                    MeshMaterial3d(beacon_red.clone()),
                    Transform::from_xyz(bx, height + spire_height + 1.0, bz),
                ));
            }
        }
    }
}

// ---------------------------------------------------------------------------
// 5. Alpine Mountain Ranges
// ---------------------------------------------------------------------------
fn spawn_mountains(
    parent: &mut ChildBuilder,
    meshes: &mut ResMut<Assets<Mesh>>,
    materials: &mut ResMut<Assets<StandardMaterial>>,
) {
    let rock_mat = materials.add(StandardMaterial {
        base_color: Color::srgb(0.38, 0.36, 0.35),
        perceptual_roughness: 0.95,
        ..default()
    });

    let snow_mat = materials.add(StandardMaterial {
        base_color: Color::srgb(0.96, 0.97, 1.0),
        perceptual_roughness: 0.6,
        ..default()
    });

    // Left Ridge
    for i in 0..14 {
        let mx = ((i as f32) * 1900.0) - 4000.0;
        let mz = -4800.0 - ((i as f32) * 180.0) % 900.0;
        let height = 1100.0 + ((i as f32 * 277.0) % 900.0);
        let base_radius = 1200.0 + ((i as f32 * 193.0) % 600.0);

        let cone_mesh = meshes.add(Cone {
            radius: base_radius,
            height,
        });
        parent.spawn((
            Mesh3d(cone_mesh),
            MeshMaterial3d(rock_mat.clone()),
            Transform::from_xyz(mx, height * 0.5, mz),
        ));

        let snow_height = height * 0.38;
        let snow_radius = base_radius * 0.38;
        let snow_mesh = meshes.add(Cone {
            radius: snow_radius,
            height: snow_height,
        });
        parent.spawn((
            Mesh3d(snow_mesh),
            MeshMaterial3d(snow_mat.clone()),
            Transform::from_xyz(mx, height - snow_height * 0.5, mz),
        ));
    }

    // Right Ridge
    for i in 0..14 {
        let mx = ((i as f32) * 1900.0) - 4000.0;
        let mz = 4800.0 + ((i as f32) * 210.0) % 900.0;
        let height = 1050.0 + ((i as f32 * 311.0) % 850.0);
        let base_radius = 1150.0 + ((i as f32 * 211.0) % 550.0);

        let cone_mesh = meshes.add(Cone {
            radius: base_radius,
            height,
        });
        parent.spawn((
            Mesh3d(cone_mesh),
            MeshMaterial3d(rock_mat.clone()),
            Transform::from_xyz(mx, height * 0.5, mz),
        ));

        let snow_height = height * 0.38;
        let snow_radius = base_radius * 0.38;
        let snow_mesh = meshes.add(Cone {
            radius: snow_radius,
            height: snow_height,
        });
        parent.spawn((
            Mesh3d(snow_mesh),
            MeshMaterial3d(snow_mat.clone()),
            Transform::from_xyz(mx, height - snow_height * 0.5, mz),
        ));
    }
}

// ---------------------------------------------------------------------------
// 6. Pine Forests & Trees
// ---------------------------------------------------------------------------
fn spawn_forests(
    parent: &mut ChildBuilder,
    meshes: &mut ResMut<Assets<Mesh>>,
    materials: &mut ResMut<Assets<StandardMaterial>>,
) {
    let trunk_mat = materials.add(StandardMaterial {
        base_color: Color::srgb(0.35, 0.22, 0.12),
        perceptual_roughness: 0.9,
        ..default()
    });

    let foliage_mat_1 = materials.add(StandardMaterial {
        base_color: Color::srgb(0.12, 0.32, 0.15),
        perceptual_roughness: 0.85,
        ..default()
    });

    let foliage_mat_2 = materials.add(StandardMaterial {
        base_color: Color::srgb(0.16, 0.40, 0.20),
        perceptual_roughness: 0.85,
        ..default()
    });

    let trunk_mesh = meshes.add(Cylinder::new(0.9, 10.0));
    let foliage_mesh_1 = meshes.add(Cone {
        radius: 6.5,
        height: 22.0,
    });
    let foliage_mesh_2 = meshes.add(Cone {
        radius: 5.2,
        height: 18.0,
    });

    for grove in 0..8 {
        let gx = ((grove * 2700) % 20000) as f32 - 2000.0;
        let gz = if grove % 2 == 0 {
            -1800.0 - ((grove * 450) % 1200) as f32
        } else {
            1800.0 + ((grove * 450) % 1200) as f32
        };

        for t in 0..24 {
            let tx = gx + ((t * 89) % 400) as f32 - 200.0;
            let tz = gz + ((t * 137) % 400) as f32 - 200.0;
            let scale = 0.8 + ((t as f32 * 19.0) % 40.0) / 100.0;

            parent.spawn((
                Mesh3d(trunk_mesh.clone()),
                MeshMaterial3d(trunk_mat.clone()),
                Transform::from_xyz(tx, 5.0 * scale, tz).with_scale(Vec3::splat(scale)),
            ));

            let fol_mesh = if t % 2 == 0 {
                foliage_mesh_1.clone()
            } else {
                foliage_mesh_2.clone()
            };
            let fol_mat = if t % 2 == 0 {
                foliage_mat_1.clone()
            } else {
                foliage_mat_2.clone()
            };

            parent.spawn((
                Mesh3d(fol_mesh),
                MeshMaterial3d(fol_mat),
                Transform::from_xyz(tx, (10.0 + 9.0) * scale, tz).with_scale(Vec3::splat(scale)),
            ));
        }
    }
}

// ---------------------------------------------------------------------------
// 7. Wind Turbines
// ---------------------------------------------------------------------------
fn spawn_wind_turbines(
    parent: &mut ChildBuilder,
    meshes: &mut ResMut<Assets<Mesh>>,
    materials: &mut ResMut<Assets<StandardMaterial>>,
) {
    let white_mat = materials.add(StandardMaterial {
        base_color: Color::srgb(0.92, 0.94, 0.96),
        perceptual_roughness: 0.35,
        ..default()
    });

    let red_tip_mat = materials.add(StandardMaterial {
        base_color: Color::srgb(0.9, 0.15, 0.1),
        ..default()
    });

    let tower_mesh = meshes.add(Cylinder::new(2.2, 90.0));
    let nacelle_mesh = meshes.add(Cuboid::new(12.0, 4.5, 4.5));
    let blade_mesh = meshes.add(Cuboid::new(0.6, 38.0, 1.8));

    for i in 0..12 {
        let tx = 8000.0 + ((i as f32) * 550.0);
        let tz = 1600.0 + if i % 2 == 0 { 0.0 } else { 350.0 };

        // Tower
        parent.spawn((
            Mesh3d(tower_mesh.clone()),
            MeshMaterial3d(white_mat.clone()),
            Transform::from_xyz(tx, 45.0, tz),
        ));

        // Nacelle (generator housing)
        parent.spawn((
            Mesh3d(nacelle_mesh.clone()),
            MeshMaterial3d(white_mat.clone()),
            Transform::from_xyz(tx, 90.0, tz),
        ));

        // Rotating 3-blade rotor hub
        let speed = 0.8 + ((i as f32 * 0.13) % 0.4);
        parent
            .spawn((
                WindTurbineRotor { speed },
                Transform::from_xyz(tx - 6.5, 90.0, tz),
                Visibility::default(),
            ))
            .with_children(|parent| {
                for b in 0..3 {
                    let angle = (b as f32) * (std::f32::consts::TAU / 3.0);
                    parent.spawn((
                        Mesh3d(blade_mesh.clone()),
                        MeshMaterial3d(white_mat.clone()),
                        Transform::from_xyz(0.0, 19.0, 0.0)
                            .with_rotation(Quat::from_rotation_x(angle)),
                    ));
                    parent.spawn((
                        Mesh3d(meshes.add(Cuboid::new(0.62, 6.0, 1.82))),
                        MeshMaterial3d(red_tip_mat.clone()),
                        Transform::from_xyz(0.0, 35.0, 0.0)
                            .with_rotation(Quat::from_rotation_x(angle)),
                    ));
                }
            });
    }
}

/// System to animate wind turbine rotor blades.
pub fn animate_wind_turbines(
    time: Res<Time>,
    mut query: Query<(&WindTurbineRotor, &mut Transform)>,
) {
    let dt = time.delta_secs();
    for (turbine, mut tf) in query.iter_mut() {
        tf.rotate_x(turbine.speed * dt);
    }
}

// ---------------------------------------------------------------------------
// 8. 3D Clouds
// ---------------------------------------------------------------------------
fn spawn_clouds(
    parent: &mut ChildBuilder,
    meshes: &mut ResMut<Assets<Mesh>>,
    materials: &mut ResMut<Assets<StandardMaterial>>,
) {
    let cloud_mat = materials.add(StandardMaterial {
        base_color: Color::srgba(0.98, 0.98, 1.0, 0.82),
        perceptual_roughness: 0.9,
        reflectance: 0.1,
        ..default()
    });

    let sphere_mesh = meshes.add(Sphere::new(60.0).mesh().ico(3).unwrap());

    for c in 0..40 {
        let cx = ((c * 1793) % 40000) as f32 - 10000.0;
        let cy = 1600.0 + ((c * 311) % 800) as f32;
        let cz = ((c * 2371) % 36000) as f32 - 18000.0;

        parent
            .spawn((
                Transform::from_xyz(cx, cy, cz),
                Visibility::default(),
            ))
            .with_children(|parent| {
                for p in 0..5 {
                    let ox = ((p * 73) % 180) as f32 - 90.0;
                    let oy = ((p * 47) % 60) as f32 - 30.0;
                    let oz = ((p * 109) % 180) as f32 - 90.0;
                    let scale = 1.0 + ((p as f32 * 0.3) % 0.9);

                    parent.spawn((
                        Mesh3d(sphere_mesh.clone()),
                        MeshMaterial3d(cloud_mat.clone()),
                        Transform::from_xyz(ox, oy, oz).with_scale(Vec3::splat(scale)),
                    ));
                }
            });
    }
}
