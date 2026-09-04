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
var engine_out := 0  # 0 = both, 1 = left out, 2 = right out

var _particles: int = 25000
var _propellers: Array = []
var _flaps: Array = []
var _ailerons: Array = []
var _aircraft_index := 0
var _imported_name := ""
var smoke_density := 0.85
# Resolution for voxelizing an imported model into wind panels (edge cells,
# powers of two). Min/max matched to the Rust voxelizer (4..=24).
var _import_resolution := 12
const AIRCRAFT_NAMES := ["MQI", "TwinEngine"]
const MODEL_FILTERS := [
	"*.glb;GLTF Binary",
	"*.gltf;GLTF Text",
	"*.obj;Wavefront OBJ",
	"*.fbx;Autodesk FBX",
]
const _AircraftViewScript := preload("res://scripts/aircraft_view.gd")

var _label: Label
var _aircraft_menu: OptionButton
var _res_label: Label
var _material: ShaderMaterial = null
@onready var _tunnel = $Physics
@onready var _drone: Node3D = $DroneView
@onready var _smoke: GPUParticles3D = $Smoke
@onready var _camera: Camera3D = $Camera
@onready var _file_dialog: FileDialog = _build_file_dialog()

func _ready() -> void:
	_aircraft_menu = _build_hud()
	_build_world()
	# Use the imported GLB aircraft models rather than the procedural drone.
	var view: Node3D = _AircraftViewScript.new()
	view.name = "Model"
	_drone.add_child(view)
	_load_aircraft(AIRCRAFT_NAMES[_aircraft_index])
	_tunnel.reset_trails()
	# Particle advection happens on the GPU (GPUParticles3D); tell Rust to skip
	# its own CPU-side particle loop so we don't pay the cost twice.
	_tunnel.set_gpu_streaming(true)
	_build_smoke_mesh()
	_upload_flow_field()
	_camera_orbit()
	print("[WIND TUNNEL] particles=%d process_mat=%s pass_mesh=%s" % [
		_particles,
		_smoke.process_material != null,
		_smoke.draw_pass_1 != null,
	])

	# Add a collision shape so the drone physically interacts with the particle flow.
	var cs := CollisionShape3D.new()
	var bs := BoxShape3D.new()
	bs.size = Vector3(8, 2, 8)
	cs.shape = bs
	cs.position = Vector3.ZERO
	cs.name = "FlowCollision"
	_drone.add_child(cs)

## Switch the active aircraft: swaps the visual model and the Rust flow-grid
## aero config (including the collision-shape wind interaction).
func _load_aircraft(name: String) -> void:
	if _tunnel.switch_aircraft(name):
		_imported_name = ""
		var view := _drone.get_node_or_null("Model")
		if view:
			view.set_model(name)
			_propellers = view.propellers
			_ailerons = view.ailerons
			_flaps = view.flaps
		# Resize the Godot flow-collision box to roughly match the airframe.
		var cs := _drone.get_node_or_null("FlowCollision") as CollisionShape3D
		if cs and cs.shape is BoxShape3D:
			var len := 6.0 if name == "TwinEngine" else 4.0
			(cs.shape as BoxShape3D).size = Vector3(len, 1.6, len * 0.5)
		engine_out = 0
		_tunnel.reset_trails()

## Import an arbitrary 3D model into the wind tunnel: normalize it onto the
## tunnel's voxel grid, send the mesh to Rust to build its drag panels, and
## display it. An imported model replaces the coefficient aero (pure shape
## drag). Passing an empty/clears path reverts to the built-in aircraft.
func _import_model(path: String) -> void:
	var mesh_data := _extract_mesh_from_scene(path)
	if mesh_data.is_empty():
		push_error("WindTunnel: no triangle geometry found in '%s'" % path)
		return
	var verts: PackedVector3Array = mesh_data["vertices"]
	var idxs: PackedInt32Array = mesh_data["indices"]
	_normalize_mesh(verts)
	var npanels: int = _tunnel.set_imported_shape(verts, idxs, _import_resolution)
	if npanels <= 0:
		push_error("WindTunnel: mesh valid but voxelizer produced no panels")
		return

	# Show the model visually (import root wraps it, so its own pose is used).
	var view := _drone.get_node_or_null("Model")
	if view:
		view.show_imported(path)
		_propellers = []
		_ailerons = []
		_flaps = []
		_imported_name = path.get_file()
	# Freeze the object at the origin with a representative collision shape.
	var cs := _drone.get_node_or_null("FlowCollision") as CollisionShape3D
	if cs and cs.shape is BoxShape3D:
		(cs.shape as BoxShape3D).size = Vector3(7, 4, 7)
	engine_out = 0
	if _aircraft_menu and not _aircraft_menu.is_queued_for_deletion():
		_aircraft_menu.select(-1)
	_tunnel.reset_trails()
	print("[WIND TUNNEL] imported '%s' -> %d panels @ res %d" % [path.get_file(), npanels, _import_resolution])

## Flatten every MeshInstance3D under `scene_path` into a single triangle soup
## (world-space after applying all transforms), as vertex + index arrays.
## Returns an empty Dictionary if nothing usable is found.
func _extract_mesh_from_scene(path: String) -> Dictionary:
	var root: Node = null
	if ResourceLoader.exists(path):
		var packed := load(path)
		root = packed.instantiate() if packed is PackedScene else null
	if root == null:
		return {}
	var verts := PackedVector3Array()
	var idxs := PackedInt32Array()
	_collect_mesh_nodes(root, Transform3D.IDENTITY, verts, idxs)
	root.free()
	return {"vertices": verts, "indices": idxs}

## Recursive: append all triangle geometry under `node` (world-space after
## applying transforms) into `verts`/`idxs`.
func _collect_mesh_nodes(node: Node, xform: Transform3D, verts: PackedVector3Array, idxs: PackedInt32Array) -> void:
	if node is MeshInstance3D:
		var mesh := (node as MeshInstance3D).mesh
		if mesh is ArrayMesh:
			var am := mesh as ArrayMesh
			for s in am.get_surface_count():
				var arrs := am.surface_get_arrays(s)
				if arrs.is_empty():
					continue
				var v := arrs[Mesh.ARRAY_VERTEX] as PackedVector3Array
				if v.is_empty():
					continue
				var ni := arrs[Mesh.ARRAY_INDEX] as PackedInt32Array
				var base := verts.size()
				for p in v:
					verts.append(xform * p)
				if ni.is_empty():
					for q in v.size():
						idxs.append(base + q)
				else:
					for qi in ni:
						idxs.append(base + qi)
	for c in node.get_children():
		_collect_mesh_nodes(c, xform * c.transform, verts, idxs)

## Center the mesh's bounding box on the origin and scale it to fit the tunnel
## (~7 m across) so the voxelizer sees a consistent coordinate space.
func _normalize_mesh(verts: PackedVector3Array) -> void:
	if verts.is_empty():
		return
	var minv := verts[0]
	var maxv := verts[0]
	for i in verts.size():
		minv = minv.min(verts[i])
		maxv = maxv.max(verts[i])
	var center := (minv + maxv) * 0.5
	var ext := maxv - minv
	var size := maxf(maxf(ext.x, ext.y), ext.z)
	if size < 0.001:
		size = 1.0
	var scale := 7.0 / size
	for i in verts.size():
		verts[i] = (verts[i] - center) * scale

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

## Build the HUD: a top panel with the aircraft drop-down + import button and
## below it the telemetry label. Returns the aircraft `OptionButton`.
func _build_hud() -> OptionButton:
	var hud := CanvasLayer.new()
	hud.name = "HUDCanvas"
	add_child(hud)

	var box := VBoxContainer.new()
	box.position = Vector2(12, 12)
	box.add_theme_constant_override("separation", 10)
	hud.add_child(box)

	# --- Row: aircraft selector + import button -------------------------------
	var row := HBoxContainer.new()
	row.add_theme_constant_override("separation", 8)
	var panel := PanelContainer.new()
	var style := StyleBoxFlat.new()
	style.bg_color = Color(0.02, 0.05, 0.09, 0.9)
	style.set_corner_radius_all(6)
	style.set_content_margin_all(8)
	panel.add_theme_stylebox_override("panel", style)
	row.add_child(panel)
	box.add_child(row)

	var row_box := VBoxContainer.new()
	panel.add_child(row_box)
	var row_h := HBoxContainer.new()
	row_box.add_child(row_h)

	var menu := OptionButton.new()
	menu.add_theme_font_size_override("font_size", 14)
	for name in AIRCRAFT_NAMES:
		menu.add_item(name)
	menu.selected = _aircraft_index
	menu.custom_minimum_size = Vector2(140, 0)
	menu.item_selected.connect(_on_aircraft_selected)
	row_h.add_child(menu)

	var import_btn := Button.new()
	import_btn.text = "Import Model..."
	import_btn.add_theme_font_size_override("font_size", 14)
	import_btn.pressed.connect(_on_import_pressed)
	row_h.add_child(import_btn)

	var res_label := Label.new()
	res_label.text = "Res: %d" % _import_resolution
	res_label.add_theme_font_size_override("font_size", 13)
	row_h.add_child(res_label)
	_res_label = res_label
	var res_minus := Button.new()
	res_minus.text = "-"
	res_minus.add_theme_font_size_override("font_size", 13)
	res_minus.pressed.connect(_on_res_pressed.bind(false))
	row_h.add_child(res_minus)
	var res_plus := Button.new()
	res_plus.text = "+"
	res_plus.add_theme_font_size_override("font_size", 13)
	res_plus.pressed.connect(_on_res_pressed.bind(true))
	row_h.add_child(res_plus)

	# --- Telemetry panel ------------------------------------------------------
	var label := Label.new()
	label.add_theme_font_size_override("font_size", 14)
	label.text = "Initializing flow field..."
	label.add_theme_color_override("font_outline_color", Color(0, 0, 0, 1))
	label.add_theme_constant_override("outline_size", 4)
	box.add_child(label)
	_label = label
	return menu

func _build_file_dialog() -> FileDialog:
	var fd := FileDialog.new()
	fd.file_mode = FileDialog.FILE_MODE_OPEN_FILE
	fd.title = "Import 3D model"
	fd.access = FileDialog.ACCESS_FILESYSTEM
	fd.filters = MODEL_FILTERS
	fd.file_selected.connect(_on_import_file_selected)
	# Wrap in a CanvasLayer so it shows above the 3D viewport.
	var layer := CanvasLayer.new()
	layer.name = "FileDialogLayer"
	layer.layer = 20
	add_child(layer)
	layer.add_child(fd)
	return fd

func _on_aircraft_selected(index: int) -> void:
	_aircraft_index = index
	_load_aircraft(AIRCRAFT_NAMES[index])

func _on_import_pressed() -> void:
	_file_dialog.popup_centered_ratio(0.6)

func _on_import_file_selected(path: String) -> void:
	_import_model(path)

func _on_res_pressed(increase: bool) -> void:
	var step := 2
	_import_resolution = clampi(_import_resolution + step if increase else _import_resolution - step, 4, 24)
	if _res_label and is_instance_valid(_res_label):
		_res_label.text = "Res: %d" % _import_resolution

func _build_smoke_mesh() -> void:
	_particles = int(_tunnel.particle_count())

	# GPU particle system: the process stage advects every particle through the
	# Rust-computed flow field entirely on the GPU, and the draw pass renders
	# them as dense, soft overlapping puffs.  No CPU particle loop, no
	# MultiMesh buffer transfers -> thousands of particles at full frame rate.
	var pm := ShaderMaterial.new()
	pm.shader = load("res://shaders/wind_tunnel_flow.gdshader") as Shader
	_smoke.process_material = pm

	# Draw pass: a single unit quad scaled per-particle to a soft cloud puff.
	var quad := QuadMesh.new()
	quad.size = Vector2.ONE
	_smoke.draw_pass_1 = quad

	var mat := ShaderMaterial.new()
	mat.shader = load("res://shaders/dense_smoke.gdshader") as Shader
	mat.set_shader_parameter("flow_speed", wind_speed)
	mat.set_shader_parameter("density", smoke_density)
	_smoke.material_override = mat
	_smoke.amount = _particles
	_smoke.lifetime = 2.5
	_smoke.preprocess = _particles
	_smoke.explosiveness = 0.0
	_smoke.randomness = 1.0
	_smoke.fixed_fps = 60
	_smoke.cast_shadow = GeometryInstance3D.SHADOW_CASTING_SETTING_OFF
	_smoke.visibility_aabb = AABB(Vector3(-26, -8, -17), Vector3(42, 16, 34))
	_material = mat

## Upload the Rust flow field (Texture3D) + advection metadata into the GPU
## particle process shader once per frame.
func _upload_flow_field() -> void:
	var meta: PackedFloat32Array = _tunnel.get_flow_meta()
	if meta.size() < 8:
		return
	var origin := Vector3(meta[0], meta[1], meta[2])
	var cell := meta[3]
	var dims := Vector3i(int(meta[4]), int(meta[5]), int(meta[6]))
	var vmax := meta[7]

	# Rebuild the Texture3D only when the grid dims change (a rare event);
	# else just refresh the stored layer data from the live Rust field.
	if _flow_tex == null or _flow_dims != dims:
		_flow_dims = dims
		_make_flow_texture(dims)
	if _flow_tex != null:
		var data: PackedByteArray = _tunnel.get_flow_field()
		var slice_bytes := dims.x * dims.y * 4 * 4
		if data.size() >= slice_bytes * dims.z:
			for z in dims.z:
				var img := _flow_layers[z]
				img.set_data(
					dims.x, dims.y, false, Image.FORMAT_RGBAF,
					data.slice(z * slice_bytes, (z + 1) * slice_bytes),
				)
			_flow_tex.update(_flow_layers)
			_smoke.process_material.set_shader_parameter("flow_tex", _flow_tex)

	_smoke.process_material.set_shader_parameter("flow_origin", origin)
	_smoke.process_material.set_shader_parameter("flow_cell", Vector3.ONE * cell)
	_smoke.process_material.set_shader_parameter("flow_dims", Vector3(dims.x, dims.y, dims.z))
	_smoke.process_material.set_shader_parameter("vmax", vmax)
	_smoke.process_material.set_shader_parameter("particle_life", 2.5)
	_smoke.process_material.set_shader_parameter("puff_radius", 0.55)
	if _material is ShaderMaterial:
		_material.set_shader_parameter("flow_speed", wind_speed)
		_material.set_shader_parameter("density", smoke_density)
		_material.set_shader_parameter("core_alpha", 0.18)
		_material.set_shader_parameter("puff_soft", 0.65)
		_material.set_shader_parameter("puff_scale", 2.4)
		_material.set_shader_parameter("noise_scale", 0.45)
		_material.set_shader_parameter("glow", 0.9)
		_material.set_shader_parameter("age_fade", 0.35)

## Build an RGBAF Texture3D sized to the flow grid plus its persistent layer
## images, so we can refresh slice data without re-fetching from the texture.
func _make_flow_texture(dims: Vector3i) -> void:
	_flow_layers.clear()
	for z in dims.z:
		_flow_layers.append(Image.create_empty(dims.x, dims.y, false, Image.FORMAT_RGBAF))
	var tex := ImageTexture3D.new()
	tex.create(Image.FORMAT_RGBAF, dims.x, dims.y, dims.z, false, _flow_layers)
	_flow_tex = tex

var _flow_tex: ImageTexture3D = null
var _flow_layers: Array[Image] = []
var _flow_dims := Vector3i.ZERO

## Gentle global brightness pulse so the mist visibly "breathes" with the flow.
func _update_flow_uniform() -> void:
	if _material is ShaderMaterial:
		_material.set_shader_parameter("density", smoke_density)

func _physics_process(delta: float) -> void:
	_apply_settings()
	_update_flow_uniform()
	_tunnel.step(delta)
	_drone.transform = _tunnel.get_drone_transform()
	# Propellers spin with the wind; ailerons/flaps deflect with their controls.
	for prop in _propellers:
		if prop is Node3D:
			prop.rotate_x(wind_speed * 0.5 * delta)
	for ail in _ailerons:
		if ail is Node3D:
			ail.rotation.x = -aileron * 0.6
	for flap in _flaps:
		if flap is Node3D:
			flap.rotation.x = flaps_deg * PI / 180.0
	# Advect on the GPU through the live Rust flow field.
	_upload_flow_field()
	_camera_orbit()
	_update_hud()

func _apply_settings() -> void:
	_tunnel.set_wind_speed(wind_speed)
	_tunnel.set_wind_direction(wind_dir_deg)
	_tunnel.set_attitude(pitch_deg, roll_deg, yaw_deg)
	_tunnel.set_controls(aileron, rudder, elevator)
	_tunnel.set_flaps_deg(flaps_deg)
	_tunnel.set_throttle(engine_throttle())
	_tunnel.set_throttle_split(float(engine_out_side()))

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
	if Input.is_key_pressed(KEY_BRACKETLEFT):
		smoke_density = maxf(smoke_density - 0.6 * delta, 0.2)
	if Input.is_key_pressed(KEY_BRACKETRIGHT):
		smoke_density = minf(smoke_density + 0.6 * delta, 2.0)
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
		engine_out = 0
		smoke_density = 1.0
		_tunnel.reset_trails()
	if _just_pressed(KEY_M):
		_aircraft_index = (_aircraft_index + 1) % AIRCRAFT_NAMES.size()
		_load_aircraft(AIRCRAFT_NAMES[_aircraft_index])
	if _just_pressed(KEY_G):
		engine_out = (engine_out + 1) % 3
	if _just_pressed(KEY_ESCAPE):
		get_tree().change_scene_to_file("res://scenes/main_menu.tscn")

var _held := {}

func _just_pressed(key: Key) -> bool:
	return Input.is_key_pressed(key) and not _held.get(key, false)

## The tunnel holds the aircraft fixed in the flow, so the engines run at a
## fixed full throttle to make the thrust-line and asymmetric-engine moments
## (engine-out) visible on the aero HUD.
func engine_throttle() -> float:
	return 1.0

## Map the engine-out state to a throttle-split for the twin physics:
## -1 = left engine out, 0 = both running, +1 = right engine out.
## Has no effect on the single-engine aircraft (MQI).
func engine_out_side() -> int:
	return -1 if engine_out == 1 else (1 if engine_out == 2 else 0)

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
	# VLM diagnostics: spanwise circulation distribution
	var vlm_gamma: PackedFloat32Array = _tunnel.get_vlm_gamma()
	var vlm_str := ""
	if vlm_gamma.size() >= 4:
		# Report root circulation and total peak
		var peak := 0.0
		for i in range(1, vlm_gamma.size(), 2):
			peak = maxf(peak, absf(vlm_gamma[i]))
		vlm_str = "VLM spanwise circulation:\npeak Γ = %.1f m²/s\n%d panels" % [peak, vlm_gamma.size() / 2]
	var engine_str := "BOTH RUNNING"
	if engine_out == 1:
		engine_str = "LEFT ENGINE OUT [G]"
	elif engine_out == 2:
		engine_str = "RIGHT ENGINE OUT [G]"
	var aero_txt := "computing..."
	if has_aero:
		aero_txt = "Lift:  %7.0f N\nDrag:  %7.0f N\nSide:  %7.0f N\nRoll M:  %+6.0f Nm\nPitch M: %+6.0f Nm\nYaw M:   %+6.0f Nm\nCL:      %+5.2f" % [mag[0], mag[1], mag[2], a[3], a[4], a[5], a[6]]
		if _tunnel.is_imported_shape():
			var ia: PackedFloat64Array = _tunnel.get_imported_aero()
			aero_txt += "\n\nImported drag model (realistic):\nCd(ref frontal): %5.2f\nRe:             %8.1e\nFrontal area:   %5.2f m2\nWetted area:    %5.2f m2" % [ia[0], ia[1], ia[2], ia[3]]
	var aircraft_str := _imported_name if _imported_name != "" else ("%s (built-in)" % AIRCRAFT_NAMES[_aircraft_index])
	var vlm_txt := vlm_str if not _tunnel.is_imported_shape() else ""
	_label.text = """WIND TUNNEL
-----------------------------
Model:       %s
Import res:  %d (voxel)
Wind speed:  %5.1f m/s
Wind dir:    %5.0f deg
Pitch:       %5.1f deg
Roll:        %5.1f deg
Yaw (beta):  %5.1f deg
Aileron:     %+4.1f deg
Rudder:      %+4.1f deg
Flaps:       %3.0f deg
Engines:     %s
Smoke:       GPU advect, %d puffs
Smoke dens:  %4.2f
-----------------------------
%s
%s
METHODS
-----------------------------
Aircraft:    [M] or dropdown
Import:      "Import Model..." button
Resolution:  [-] / [+] buttons
Pitch:       [W] / [S]
Yaw:         [A] / [D]
Roll:        [Up] / [Down]
Aileron:     [Q] / [E]
Rudder:      [Z] / [C]
Flaps:       [F] cycle
Engine:      [G] both -> left out -> right out
Wind speed:  [Shift] / [Ctrl]
Wind dir:    [R] / [T]
Smoke dens:  [ / ]
Reset:       [Space]
Menu:        [Esc]
Camera:      [Left-drag] / [Scroll]""" % [
		aircraft_str,
		_import_resolution,
		s[0], s[1], s[2], s[3], s[4], s[5], s[6], s[7],
		engine_str,
		_particles,
		smoke_density,
		aero_txt,
		vlm_txt,
	]
