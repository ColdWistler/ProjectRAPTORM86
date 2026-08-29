# Asset Import Directory

This folder is the standard root for all runtime assets loaded by Bevy into the flight simulator.

## Directory Structure

```
assets/
├── models/       # 3D GLTF / GLB / OBJ models (e.g. drone.glb, cockpit.glb, runway.glb)
├── textures/     # Image files: PNG, JPG, KTX2, WebP (e.g. runway_pavement.png, grass.png)
├── audio/        # Sound effects: OGG, WAV, MP3, FLAC (e.g. turboprop_loop.ogg, stall_horn.wav)
└── fonts/        # HUD Fonts: TTF, OTF (e.g. FiraCode.ttf, AviationHUD.ttf)
```

## How to Load Assets in Bevy 0.15

### 1. Loading a 3D Model (GLTF / GLB)
Drop your `.glb` or `.gltf` model into `assets/models/my_drone.glb` and spawn it:

```rust
fn spawn_custom_model(mut commands: Commands, asset_server: Res<AssetServer>) {
    commands.spawn((
        SceneRoot(asset_server.load("models/my_drone.glb#Scene0")),
        Transform::from_xyz(0.0, 1000.0, 0.0),
    ));
}
```

### 2. Loading a Texture on a Material
```rust
fn setup_textured_material(
    mut materials: ResMut<Assets<StandardMaterial>>,
    asset_server: Res<AssetServer>,
) {
    let material_handle = materials.add(StandardMaterial {
        base_color_texture: Some(asset_server.load("textures/terrain.png")),
        perceptual_roughness: 0.8,
        ..default()
    });
}
```

### 3. Playing Spatial Engine Audio
```rust
fn play_engine_audio(mut commands: Commands, asset_server: Res<AssetServer>) {
    commands.spawn((
        AudioPlayer::new(asset_server.load("audio/engine_sound.ogg")),
        PlaybackSettings::LOOP,
    ));
}
```
