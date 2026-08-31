extends Node3D
## Wind-tunnel mode. The drone is fixed at the origin; Rust (`WindTunnelNode`)
## owns the flow-field advection + aero forces. This script renders thousands
## of free "air molecules" (Rust particle pool) into a single MultiMeshInstance
## (one draw call for the whole field), orbits the camera, and shows the
## computed aero telemetry.

var cam_yaw := -0.6
var cam_pitch := 0.35
var cam_dist := 24.0
var wind_speed := 20.0
var wind_dir_deg := 0.0
var pitch_deg := 4.0
var roll_deg := 0.0
var yaw_deg := 0.0
var aileron := 0.0
var rudder := 0.0
var elevator := 0.0
var flaps_deg := 0.0

var _particles: int = 0
var _multimesh: MultiMesh = null
var _material: Material = null
var _propeller: Node3D = null

@onready var _tunnel = $Physics
@onready var _drone: Node3D = $DroneView
@onready var _smoke: MultiMeshInstance3D = $Smoke
@onready var _camera: Camera3D = $Camera
@onready var _label: Label = _build_hud()

func _ready() -> void:
	_build_world()
	DroneFactory.build(_drone)
	_propeller = DroneFactory.build(_drone)["propeller"] as Node3D
	_tunnel.reset_trails()
	_build_smoke_mesh()
	_rebuild_smoke()
	_camera_orbit()
	print("[WIND TUNNEL] particles=%d instances=%d mesh_set=%s mat_set=%s aabb=%s" % [
		_particles,
		_multimesh.get_instance_count(),
		_smoke.multimesh != null,
		_smoke.material_override != null,
		_multimesh.custom_aabb,
	])

	# Add a collision shape so the drone physically interacts with the particle flow.
	var cs := CollisionShape3D.new()
	var bs := BoxShape3D.new()
	bs.size = Vector3(8, 2, 8)
	cs.shape = bs
	cs.position = Vector3.ZERO
	_drone.add_child(cs)

func _build_world() -> void:
	var we := WorldEnvironment.new()
	var env := Environment.new()
	env.background_mode = Environment.BG_COLOR
	env.background_color = Color(0.02, 0.03, 0.05)
	env.ambient_light_source = Environment.AMBIENT_SOURCE_COLOR
	env.ambient_light_color = Color(0.6, 0.75, 0.9)
	env.ambient_light_energy = 0.8
	we.environment = env
	add_child(we)

	var sun := DirectionalLight3D.new()
	sun.rotation_degrees = Vector3(-60, 40, 0)
	sun.light_energy = 1.5
	add_child(sun)

	var floor := MeshInstance3D.new()
	var fm := BoxMesh.new()
	fm.size = Vector3(200, 1, 200)
	var fm_mat := StandardMaterial3D.new()
	fm_mat.albedo_color = Color(0.05, 0.06, 0.08)
	floor.mesh = fm
	floor.material_override = fm_mat
	floor.position = Vector3(0, -12, 0)
	add_child(floor)

func _build_hud() -> Label:
	var hud := CanvasLayer.new()
	hud.name = "HUDCanvas"
	add_child(hud)
	var panel := PanelContainer.new()
	var style := StyleBoxFlat.new()
	style.bg_color = Color(0.02, 0.05, 0.09, 0.88)
	style.set_corner_radius_all(6)
	style.set_content_margin_all(12)
	panel.add_theme_stylebox_override("panel", style)
	panel.position = Vector2(12, 12)
	hud.add_child(panel)
	var label := Label.new()
	label.add_theme_font_size_override("font_size", 14)
	label.text = "Initializing flow field..."
	panel.add_child(label)
	return label

func _build_smoke_mesh() -> void:
	_particles = int(_tunnel.particle_count())
	var puff_count := _particles

	var mm := MultiMesh.new()
	mm.transform_format = MultiMesh.TRANSFORM_3D
	mm.use_custom_data = false
	mm.use_colors = true
	mm.instance_count = puff_count
	mm.custom_aabb = AABB(Vector3(-26, -8, -17), Vector3(42, 16, 34))
	var quad := QuadMesh.new()
	quad.size = Vector2.ONE
	mm.mesh = quad
	_smoke.multimesh = mm
	_multimesh = mm

	# ENGINE-NATIVE renderer (no custom GLSL, so it compiles on any driver):
	# a soft radial "molecule" dot, billboarded toward the camera, sized per
	# instance via the scaled basis. Per-instance colors carry age tint + fade.
	var dot := GradientTexture2D.new()
	dot.width = 64
	dot.height = 64
	dot.fill = GradientTexture2D.FILL_RADIAL
	dot.fill_from = Vector2(0.5, 0.5)
	dot.fill_to = Vector2(0.0, 0.5)
	var g := Gradient.new()
	g.offsets = PackedFloat32Array([0.0, 0.5, 1.0])
	g.colors = PackedColorArray([
		Color(1, 1, 1, 0.5),
		Color(1, 1, 1, 0.2),
		Color(1, 1, 1, 0.0),
	])
	dot.gradient = g

	var mat := StandardMaterial3D.new()
	mat.shading_mode = BaseMaterial3D.SHADING_MODE_UNSHADED
	mat.transparency = BaseMaterial3D.TRANSPARENCY_ALPHA
	mat.depth_draw_mode = BaseMaterial3D.DEPTH_DRAW_DISABLED
	mat.billboard_mode = BaseMaterial3D.BILLBOARD_ENABLED
	mat.albedo_color = Color(0.92, 0.98, 1.0)
	mat.albedo_texture = dot
	mat.vertex_color_use_as_albedo = true
	mat.emission_enabled = true
	mat.emission = Color(0.90, 0.97, 1.0)
	mat.emission_texture = dot
	mat.emission_energy_multiplier = 2.0
	_smoke.material_override = mat
	_smoke.cast_shadow = GeometryInstance3D.SHADOW_CASTING_SETTING_OFF
	_material = mat

## Gentle global brightness pulse so the mist visibly "breathes" with the flow
## even though each molecule's motion comes from the Rust advection.
func _update_flow_uniform() -> void:
	if _material is StandardMaterial3D:
		var t := Time.get_ticks_msec() * 0.001
		_material.emission_energy_multiplier = 3.6 + sin(t * 2.7) * 0.9

func _physics_process(delta: float) -> void:
	_apply_settings()
	_update_flow_uniform()
	_tunnel.step(delta)
	_drone.transform = _tunnel.get_drone_transform()
	if _propeller != null:
		var spin_speed = wind_speed * 0.5
		_propeller.rotate_y(spin_speed * delta)
	_rebuild_smoke(true)
	_camera_orbit()
	_update_hud()

func _apply_settings() -> void:
	_tunnel.set_wind_speed(wind_speed)
	_tunnel.set_wind_direction(wind_dir_deg)
	_tunnel.set_attitude(pitch_deg, roll_deg, yaw_deg)
	_tunnel.set_controls(aileron, rudder, elevator)
	_tunnel.set_flaps_deg(flaps_deg)

## Keep the MultiMesh in step with the Rust particle pool size.
func _ensure_smoke_size() -> void:
	var n := int(_tunnel.particle_count())
	if _multimesh == null or n != _particles:
		_build_smoke_mesh()

## Update every air molecule from the Rust pool. Each is a small billboard
## quad: radius (enlarging then thinning as it dissipates), with age + a
## per-particle tint/fade packed into the per-instance color so the molecules
## age from bright fresh mist into soft blue haze downstream.
func _rebuild_smoke(_enabled := true) -> void:
	_ensure_smoke_size()
	var d: PackedFloat32Array = _tunnel.get_particles()
	if d.size() < _particles * 6:
		return
	for i in _particles:
		var b := i * 6
		var p := Vector3(d[b], d[b + 1], d[b + 2])
		var r := d[b + 3]
		var age_norm := d[b + 4]
		_multimesh.set_instance_transform(i, Transform3D(Basis.from_scale(Vector3(r, r, 1.0)), p))
		var tint := Color(
			0.92 - 0.30 * age_norm,
			0.98 - 0.18 * age_norm,
			1.0 - 0.05 * age_norm,
			0.45 + 0.40 * (1.0 - age_norm * age_norm),
		)
		_multimesh.set_instance_color(i, tint)

func _camera_orbit() -> void:
	var sp := sin(cam_pitch)
	var cp := cos(cam_pitch)
	var pos := Vector3(cam_dist * cp * cos(cam_yaw), cam_dist * sp, cam_dist * cp * sin(cam_yaw))
	_camera.global_position = pos
	_camera.look_at(Vector3.ZERO, Vector3.UP)

func _unhandled_input(event: InputEvent) -> void:
	if event is InputEventMouseMotion and Input.is_mouse_button_pressed(MOUSE_BUTTON_LEFT):
		cam_yaw -= event.relative.x * 0.008
		cam_pitch = clampf(cam_pitch - event.relative.y * 0.008, -1.4, 1.4)
	elif event is InputEventMouseButton and event.pressed:
		if event.button_index == MOUSE_BUTTON_WHEEL_UP:
			cam_dist = clampf(cam_dist - 2.5, 8.0, 60.0)
		elif event.button_index == MOUSE_BUTTON_WHEEL_DOWN:
			cam_dist = clampf(cam_dist + 2.5, 8.0, 60.0)

func _process(delta: float) -> void:
	var rate := 45.0 * delta
	if Input.is_key_pressed(KEY_W):
		pitch_deg += rate
	if Input.is_key_pressed(KEY_S):
		pitch_deg -= rate
	if Input.is_key_pressed(KEY_A):
		yaw_deg += rate
	if Input.is_key_pressed(KEY_D):
		yaw_deg -= rate
	if Input.is_key_pressed(KEY_UP):
		roll_deg += rate
	if Input.is_key_pressed(KEY_DOWN):
		roll_deg -= rate
	if Input.is_key_pressed(KEY_Q):
		aileron = maxf(aileron - 45.0 * delta, -0.35)
	if Input.is_key_pressed(KEY_E):
		aileron = minf(aileron + 45.0 * delta, 0.35)
	if Input.is_key_pressed(KEY_Z):
		rudder = maxf(rudder - 45.0 * delta, -0.35)
	if Input.is_key_pressed(KEY_C):
		rudder = minf(rudder + 45.0 * delta, 0.35)
	if Input.is_key_pressed(KEY_SHIFT):
		wind_speed = minf(wind_speed + 25.0 * delta, 120.0)
	if Input.is_key_pressed(KEY_CTRL):
		wind_speed = maxf(wind_speed - 25.0 * delta, 1.0)
	if Input.is_key_pressed(KEY_R):
		wind_dir_deg += 30.0 * delta
	if Input.is_key_pressed(KEY_T):
		wind_dir_deg -= 30.0 * delta
	if _just_pressed(KEY_F):
		flaps_deg = 15.0 if flaps_deg < 5.0 else (30.0 if flaps_deg < 20.0 else 0.0)
	if _just_pressed(KEY_SPACE):
		wind_speed = 20.0
		wind_dir_deg = 0.0
		pitch_deg = 4.0
		roll_deg = 0.0
		yaw_deg = 0.0
		aileron = 0.0
		rudder = 0.0
		elevator = 0.0
		flaps_deg = 0.0
		_tunnel.reset_trails()
	if _just_pressed(KEY_ESCAPE):
		get_tree().change_scene_to_file("res://scenes/main_menu.tscn")

var _held := {}

func _just_pressed(key: Key) -> bool:
	return Input.is_key_pressed(key) and not _held.get(key, false)

func _input(event: InputEvent) -> void:
	if event is InputEventKey and event.pressed:
		_held[event.physical_keycode] = true
	elif event is InputEventKey and not event.pressed:
		_held[event.physical_keycode] = false

func _update_hud() -> void:
	var s: PackedFloat64Array = _tunnel.get_settings()
	var a: PackedFloat64Array = _tunnel.get_aero()
	var mag: PackedFloat64Array = _tunnel.get_aero_magnitudes()
	var has_aero: bool = _tunnel.has_aero()
	var aero_txt := "computing..."
	if has_aero:
		aero_txt = "Lift:  %7.0f N\nDrag:  %7.0f N\nSide:  %7.0f N\nRoll M:  %+6.0f Nm\nPitch M: %+6.0f Nm\nYaw M:   %+6.0f Nm\nCL:      %+5.2f" % [mag[0], mag[1], mag[2], a[3], a[4], a[5], a[6]]
	_label.text = """WIND TUNNEL
-----------------------------
Wind speed:  %5.1f m/s
Wind dir:    %5.0f deg
Pitch:       %5.1f deg
Roll:        %5.1f deg
Yaw (beta):  %5.1f deg
Aileron:     %+4.1f deg
Rudder:      %+4.1f deg
Flaps:       %3.0f deg
-----------------------------
%s

CONTROLS
-----------------------------
Pitch:      [W] / [S]
Yaw:        [A] / [D]
Roll:       [Up] / [Down]
Aileron:    [Q] / [E]
Rudder:     [Z] / [C]
Flaps:      [F] cycle
Wind speed: [Shift] / [Ctrl]
Wind dir:   [R] / [T]
Reset:      [Space]
Menu:       [Esc]
Camera:     [Left-drag] / [Scroll]""" % [
		s[0], s[1], s[2], s[3], s[4], s[5], s[6], s[7],
		aero_txt,
	]