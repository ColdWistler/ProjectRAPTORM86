@tool
extends Node3D
## Interactive 6-DOF flight simulation. All physics live in `flight_core`
## (via the FlightSimNode GDExtension); this script only reads input, feeds
## controls into Rust, and renders the resulting transform / HUD in Godot.

const MAX_ELEVATOR := 0.35
const MAX_AILERON := 0.35
const MAX_RUDDER := 0.35
const MAX_TRIM := 0.15

var elevator := 0.0
var elevator_trim := 0.0
var aileron := 0.0
var rudder := 0.0
var flaps_deg := 0.0
var throttle := 0.0
var engine_out := 0  # 0 = both, 1 = left out, 2 = right out
var auto_level := false

const AIRCRAFT_NAMES := ["TwinEngine", "MQI", "Engine"]
const _AircraftViewScript := preload("res://scripts/aircraft_view.gd")
const _HudScript := preload("res://scripts/hud.gd")
const _TerrainGeneratorScript := preload("res://scripts/terrain/terrain_generator.gd")
const _TerrainSettingsMeta := "raptor_terrain_settings"
const _TerrainSettingKeys := [
	"noise_seed",
	"noise_scale",
	"octaves",
	"persistence",
	"lacunarity",
	"height_multiplier",
	"height_exponent",
	"view_chunks",
	"resolution",
	"chunk_size",
	"collision_enabled",
	"physics_grid_enabled",
	"wind_speed",
	"wind_direction",
]

## Fly over the generated FBM-Perlin world (Sebastian Lague style) instead of
## the imported GLB landscape. Toggle in flight with [B], re-roll the world
## with [N]. The generator feeds the Rust ground grid so collision, ground
## effect and orographic wind follow the generated mountains.
@export var procedural_terrain := true

## Trim (altitude m, airspeed m/s) each mode re-trims to on load/reset. ENGINE
## MODE is the fast long-range jet: it starts at high-altitude fast cruise so
## it behaves like a real turbine aircraft rather than a drone.
## The drone modes cruise at 800 m — clear of the generated mountains (default
## relief tops out near 500 m), so the aircraft always spawns above the terrain.
const MODE_TRIM := {
	"TwinEngine": Vector2(800.0, 60.0),
	"MQI": Vector2(800.0, 60.0),
	"Engine": Vector2(8000.0, 200.0),
}

var _propellers: Array = []
var _flaps: Array = []
var _ailerons: Array = []
var _aircraft_index := 0
var _telemetry := PackedFloat64Array()
var _hud_timer := 0.0
var _aircraft_btn: Button = null
var _terrain_generator = null
var _terrain_settings := {}

## Control path: true = fly through the avionics stack (attitude commands to
## the PID flight controller); false = manual surface passthrough. [L] toggles.
var avionics_mode := true
var _avionics_snap := PackedFloat64Array()
var _panel: PanelContainer = null
var _panel_values: RichTextLabel = null
var _fault_buttons := {}

## Fault names exposed on the visualization panel; the labels shown and the
## exact names Rust's `inject_fault` / `clear_fault` recognise.
const _FAULTS := [
	{"label": "IMU", "name": "imu"},
	{"label": "GPS", "name": "gps"},
	{"label": "BARO", "name": "baro"},
	{"label": "MAG", "name": "mag"},
	{"label": "AIRSPD", "name": "airspeed"},
	{"label": "ELEV", "name": "servo_elevator"},
	{"label": "AIL", "name": "servo_aileron"},
	{"label": "RUD", "name": "servo_rudder"},
	{"label": "ESC", "name": "esc"},
	{"label": "BATT", "name": "battery_depleted"},
]

var cam_orbit := false
var cam_yaw := 0.0
var cam_pitch := 0.3
var cam_dist := 40.0
var cam_center := Vector3.ZERO

@onready var _physics = $Physics
@onready var _drone: Node3D = $DroneView
@onready var _camera: Camera3D = $Camera
@onready var _hud: Control = _build_hud()

func _ready() -> void:
	var is_editor := Engine.is_editor_hint()

	# In the editor we only want the static visuals; the physics (Rust) node
	# and input/HUD handling are runtime-only.
	if not is_editor:
		var started: bool = _physics.start("TwinEngine.toml")
		if not started:
			started = _physics.start("aircraft.toml")
		if not started:
			push_error("FlightSimNode failed to load any aircraft config; drone will stay frozen at origin")
			set_physics_process(false)
			return
		var trm: Vector2 = MODE_TRIM.get(AIRCRAFT_NAMES[_aircraft_index], Vector2(800.0, 60.0))
		var tr: Vector2 = _physics.trim(trm.x, trm.y)
		elevator = 0.0
		elevator_trim = tr.x
		throttle = clampf(tr.y, 0.0, 1.0)
		_apply_terrain_settings()
		_apply_wind_settings()
		if procedural_terrain:
			_setup_procedural_terrain()
		else:
			_sample_terrain_grid.call_deferred()

	# Use the imported GLB aircraft models rather than the procedural drone.
	var view: Node3D = _AircraftViewScript.new()
	view.name = "Model"
	_drone.add_child(view)
	_load_aircraft(AIRCRAFT_NAMES[_aircraft_index])

	if not is_editor:
		cam_center = _drone.global_position
		_chase_camera()

## Switch the active aircraft (visual model + physics config) and re-trim.
## Load an aircraft model by name: switches the Rust physics config, swaps
## the visual model, grabs the control-surface nodes and re-trims to the
## mode's cruise altitude/speed.
func _load_aircraft(name: String) -> void:
	var is_editor := Engine.is_editor_hint()
	var ok := true
	if not is_editor:
		ok = _physics.switch_aircraft(name)
	var view := _drone.get_node_or_null("Model")
	if view:
		view.set_model(name)
		_propellers = view.propellers
		_ailerons = view.ailerons
		_flaps = view.flaps
	if ok and not is_editor:
		var trm: Vector2 = MODE_TRIM.get(name, Vector2(800.0, 60.0))
		var tr: Vector2 = _physics.trim(trm.x, trm.y)
		elevator = 0.0
		elevator_trim = tr.x
		aileron = 0.0
		rudder = 0.0
		flaps_deg = 0.0
		throttle = clampf(tr.y, 0.0, 1.0)
		engine_out = 0

## Read terrain choices saved by the main-menu Terrain tab from scene-root
## metadata. Root metadata survives scene changes, so this avoids writing a
## settings file while still carrying the selection into the flight scene.
func _apply_terrain_settings() -> void:
	var root := get_tree().root
	if root == null or not root.has_meta(_TerrainSettingsMeta):
		return
	var saved = root.get_meta(_TerrainSettingsMeta)
	if not (saved is Dictionary):
		return
	if saved.has("procedural_terrain"):
		procedural_terrain = bool(saved["procedural_terrain"])
	_terrain_settings.clear()
	for key in _TerrainSettingKeys:
		if saved.has(key):
			_terrain_settings[key] = saved[key]

## Push the steady wind chosen in the main-menu Terrain tab into Rust. The
## speed slider is 0 by default (still air, i.e. the env-configured wind), so
## this is a no-op until the user actually asks for wind.
func _apply_wind_settings() -> void:
	if _physics == null:
		return
	var wind_speed := float(_terrain_settings.get("wind_speed", 0.0))
	if wind_speed <= 0.0:
		return
	var wind_dir := float(_terrain_settings.get("wind_direction", 0.0))
	_physics.set_wind(wind_speed, wind_dir)
	print("wind configured: %0.1f m/s toward %0.0f°" % [wind_speed, wind_dir])

func _apply_terrain_settings_to_generator(gen) -> void:
	for key in _terrain_settings:
		if key in ["wind_speed", "wind_direction"]:
			continue
		gen.set(key, _terrain_settings[key])

func _save_terrain_settings() -> void:
	var root := get_tree().root
	if root == null:
		return
	var settings := {"procedural_terrain": procedural_terrain}
	for key in _TerrainSettingKeys:
		if _terrain_settings.has(key):
			settings[key] = _terrain_settings[key]
	if _terrain_generator != null:
		for key in _TerrainSettingKeys:
			settings[key] = _terrain_generator.get(key)
	root.set_meta(_TerrainSettingsMeta, settings)

## Create the chunked procedural-terrain generator, point it at the aircraft
## and hand it the Rust physics-grid sink. The imported GLB landscape is hid
## because it overlaps the generated world.
func _setup_procedural_terrain() -> void:
	if _terrain_generator != null:
		return
	var glb := get_node_or_null("Sketchfab_Scene")
	if glb:
		glb.visible = false
	var gen = _TerrainGeneratorScript.new()
	gen.name = "TerrainGenerator"
	add_child(gen)
	_apply_terrain_settings_to_generator(gen)
	gen.physics_grid_callback = Callable(self, "_push_physics_terrain")
	gen.set_target_node(_drone)
	_terrain_generator = gen
	_save_terrain_settings()
	_physics.set_terrain_enabled(false)
	print("procedural terrain: chunk world following the aircraft")

## Remove the generated world and go back to the imported GLB landscape, then
## re-sample its ground grid into the Rust physics.
func _teardown_procedural_terrain() -> void:
	if _terrain_generator:
		_terrain_generator.queue_free()
		_terrain_generator = null
	var glb := get_node_or_null("Sketchfab_Scene")
	if glb:
		glb.visible = true
	_physics.set_terrain_enabled(false)
	_sample_terrain_grid.call_deferred()

## The generator re-rolls the world seed and rebuilds chunks around the plane.
func _regenerate_procedural_terrain() -> void:
	if _terrain_generator:
		_terrain_generator.regenerate()
		_save_terrain_settings()
		print("procedural terrain: regenerated")

## Sink for the generator's coarse height grid; also (re)enables the physics
## terrain once a grid has actually arrived.
func _push_physics_terrain(
	north0: float,
	east0: float,
	spacing: float,
	nx: int,
	nz: int,
	heights: PackedFloat64Array,
) -> void:
	if _physics == null:
		return
	_physics.configure_terrain(north0, east0, spacing, nx, nz, heights)
	_physics.set_terrain_enabled(true)

## Sample the imported terrain mesh into a uniform height grid and hand it to
## the Rust physics so the mountains are the real ground (collision + ground
## effect + orographic wind). Godot world (x, z) maps to NED (north, east) and
## world y is the surface altitude, matching `FlightSimNode`.
func _sample_terrain_grid() -> void:
	var root := get_node_or_null("Sketchfab_Scene")
	if root == null:
		push_warning("terrain: no Sketchfab_Scene node to sample")
		return
	var meshes: Array = []
	_collect_meshes(root, meshes)
	if meshes.is_empty():
		push_warning("terrain: no mesh instances found")
		return

	# Pass 1: world-space bounding box of the (north=x, east=z) extent.
	var minx := INF
	var maxx := -INF
	var minz := INF
	var maxz := -INF
	for mi: MeshInstance3D in meshes:
		var tf: Transform3D = mi.global_transform
		var mesh: Mesh = mi.mesh
		for s in mesh.get_surface_count():
			var verts: PackedVector3Array = mesh.surface_get_arrays(s)[Mesh.ARRAY_VERTEX]
			for v in verts:
				var w := tf * v
				minx = minf(minx, w.x)
				maxx = maxf(maxx, w.x)
				minz = minf(minz, w.z)
				maxz = maxf(maxz, w.z)
	if not (minx < maxx and minz < maxz):
		push_warning("terrain: degenerate bounds")
		return

	# Grid: fixed 64 m cells; cap resolution so the grid stays cheap.
	var spacing := 64.0
	var nx := clampi(ceili((maxx - minx) / spacing), 8, 512)
	var nz := clampi(ceili((maxz - minz) / spacing), 8, 512)
	var heights := PackedFloat64Array()
	heights.resize(nx * nz)
	heights.fill(-INF)
	var miny := INF
	var maxy := -INF

	# Pass 2: accumulate the highest vertex per cell (keeps the aircraft from
	# "tunnelling" through any peak even for a coarse grid).
	for mi: MeshInstance3D in meshes:
		var tf: Transform3D = mi.global_transform
		var mesh: Mesh = mi.mesh
		for s in mesh.get_surface_count():
			var verts: PackedVector3Array = mesh.surface_get_arrays(s)[Mesh.ARRAY_VERTEX]
			for v in verts:
				var w := tf * v
				miny = minf(miny, w.y)
				maxy = maxf(maxy, w.y)
				var ix := int(floor((w.x - minx) / spacing))
				var iz := int(floor((w.z - minz) / spacing))
				ix = clampi(ix, 0, nx - 1)
				iz = clampi(iz, 0, nz - 1)
				var idx := iz * nx + ix
				if w.y > heights[idx]:
					heights[idx] = w.y

	# Fill any empty cells with the datum, then carve a flat apron around the
	# runway (the 1000 x 42 m strip at the world origin).
	for i in nx * nz:
		if heights[i] == -INF:
			heights[i] = 0.0
	for iz in nz:
		for ix in nx:
			var north := minx + ix * spacing
			var east := minz + iz * spacing
			if absf(north) < 650.0 and absf(east) < 90.0:
				heights[iz * nx + ix] = 0.0

	_physics.configure_terrain(minx, minz, spacing, nx, nz, heights)
	_physics.set_terrain_enabled(true)
	print("terrain grid configured: ", nx, "x", nz, " @ ", spacing, " m (", minx, "..", maxx, " , ", minz, "..", maxz, " , y ", miny, "..", maxy, " m)")

## Depth-first collect every `MeshInstance3D` under `n` into `out`.
func _collect_meshes(n: Node, out: Array) -> void:
	if n is MeshInstance3D:
		out.append(n)
	for c in n.get_children():
		_collect_meshes(c, out)

## Build the on-screen HUD: a full-screen pilot HUD overlay (instruments
## drawn by `FlightHUD`), the avionics component panel and the aircraft-swap
## button. Returns the overlay so the physics loop can feed it telemetry.
func _build_hud() -> Control:
	var hud := CanvasLayer.new()
	hud.name = "HUDCanvas"
	add_child(hud)

	# Full-screen instrument overlay (drawn first, so it sits behind the panels).
	var overlay := _HudScript.new()
	overlay.set_anchors_preset(Control.PRESET_FULL_RECT)
	overlay.mouse_filter = Control.MOUSE_FILTER_IGNORE
	hud.add_child(overlay)

	_aircraft_btn = Button.new()
	_aircraft_btn.position = Vector2(12, 620)
	_aircraft_btn.custom_minimum_size = Vector2(140, 28)
	_aircraft_btn.add_theme_font_size_override("font_size", 12)
	var btn_style := StyleBoxFlat.new()
	btn_style.bg_color = Color(0.04, 0.07, 0.11, 0.85)
	btn_style.set_corner_radius_all(6)
	btn_style.set_content_margin_all(6)
	_aircraft_btn.add_theme_stylebox_override("normal", btn_style)
	var btn_hover := btn_style.duplicate()
	btn_hover.bg_color = Color(0.10, 0.18, 0.28, 0.9)
	_aircraft_btn.add_theme_stylebox_override("hover", btn_hover)
	_aircraft_btn.pressed.connect(_on_aircraft_swap)
	hud.add_child(_aircraft_btn)
	_update_aircraft_btn_text()

	_build_avionics_panel(hud)

	return overlay

## Cycle to the next aircraft in the list and reload it.
func _on_aircraft_swap() -> void:
	_aircraft_index = (_aircraft_index + 1) % AIRCRAFT_NAMES.size()
	_load_aircraft(AIRCRAFT_NAMES[_aircraft_index])
	_update_aircraft_btn_text()

## Refresh the aircraft-swap button label and its "press to switch" tooltip.
func _update_aircraft_btn_text() -> void:
	if _aircraft_btn:
		var next: String = AIRCRAFT_NAMES[(_aircraft_index + 1) % AIRCRAFT_NAMES.size()]
		_aircraft_btn.text = "Aircraft: %s  [Swap]" % AIRCRAFT_NAMES[_aircraft_index]
		_aircraft_btn.tooltip_text = "Click or press [M] to switch to %s" % next

## Per-physics-frame tick: read input, push the control set and engine split
## into Rust, step the sim, then render the drone transform, control surfaces,
## chase/orbit camera and periodically refresh the HUD.
func _physics_process(delta: float) -> void:
	if Engine.is_editor_hint():
		return

	_handle_input(delta)

	if avionics_mode:
		_physics.set_avionics_command(aileron, elevator, rudder, throttle)
	else:
		_physics.set_controls(elevator, aileron, rudder, throttle, flaps_deg)
	_physics.set_elevator_trim(elevator_trim)
	_physics.set_throttle_split(float(engine_out_side()))
	_physics.step(delta)

	_drone.transform = _physics.get_drone_transform()
	_update_control_surfaces()
	_chase_camera()

	for prop in _propellers:
		if prop is Node3D:
			prop.rotate_x(delta * (throttle * 60.0 + 3.0))

	# Feed the instrument overlay every frame so the tapes/ladder feel live.
	_telemetry = _physics.telemetry()
	_hud.update_telemetry(_telemetry, _avionics_snap, avionics_mode, auto_level, engine_out)

	# The avionics panel only needs a lower refresh rate.
	_hud_timer += delta
	if _hud_timer >= 0.2:
		_hud_timer = 0.0
		_avionics_snap = _physics.avionics_snapshot()
		_update_avionics_panel()

## Poll keyboard/mouse inputs into the elevator/aileron/rudder/flap/trim/
## throttle state, plus camera and aircraft-swap handling. Control stick
## values auto-center when no pitch/roll input is held (unless autopilot on).
func _handle_input(delta: float) -> void:
	# Pitch: W/Up = push DOWN (dive), S/Down = pull UP (climb)
	var manual_pitch := false
	if Input.is_key_pressed(KEY_S) or Input.is_key_pressed(KEY_DOWN):
		elevator = maxf(elevator - 0.9 * delta, -MAX_ELEVATOR)
		manual_pitch = true
	elif Input.is_key_pressed(KEY_W) or Input.is_key_pressed(KEY_UP):
		elevator = minf(elevator + 0.9 * delta, MAX_ELEVATOR)
		manual_pitch = true
	elif not auto_level:
		elevator = _center_control(elevator, 0.9, 1.5, delta)

	# Roll: D/Right = RIGHT wing down, A/Left = LEFT wing down
	var manual_roll := false
	if Input.is_key_pressed(KEY_D) or Input.is_key_pressed(KEY_RIGHT):
		aileron = minf(aileron + 1.0 * delta, MAX_AILERON)
		manual_roll = true
	elif Input.is_key_pressed(KEY_A) or Input.is_key_pressed(KEY_LEFT):
		aileron = maxf(aileron - 1.0 * delta, -MAX_AILERON)
		manual_roll = true
	elif not auto_level:
		aileron = _center_control(aileron, 1.0, 1.5, delta)

	# Rudder: E/C = RIGHT, Q/Z = LEFT
	if Input.is_key_pressed(KEY_E) or Input.is_key_pressed(KEY_C):
		rudder = minf(rudder + 0.9 * delta, MAX_RUDDER)
	elif Input.is_key_pressed(KEY_Q) or Input.is_key_pressed(KEY_Z):
		rudder = maxf(rudder - 0.9 * delta, -MAX_RUDDER)
	else:
		rudder = _center_control(rudder, 0.9, 1.5, delta)

	# Flaps cycle 0 -> 15 -> 30
	if _just_pressed(KEY_F):
		flaps_deg = 15.0 if flaps_deg < 5.0 else (30.0 if flaps_deg < 20.0 else 0.0)

	# Engine out cycle: both -> left out -> right out (twin only)
	if _just_pressed(KEY_G):
		engine_out = (engine_out + 1) % 3

	# Elevator trim: [ = nose DOWN trim, ] = nose UP trim
	if Input.is_key_pressed(KEY_BRACKETRIGHT):
		elevator_trim = minf(elevator_trim + 0.05 * delta, MAX_TRIM)
	if Input.is_key_pressed(KEY_BRACKETLEFT):
		elevator_trim = maxf(elevator_trim - 0.05 * delta, -MAX_TRIM)

	# Throttle: Shift up / Ctrl down
	if Input.is_key_pressed(KEY_SHIFT):
		throttle = minf(throttle + 0.35 * delta, 1.0)
	if Input.is_key_pressed(KEY_CTRL):
		throttle = maxf(throttle - 0.35 * delta, 0.0)

	# Autopilot: H or T toggle
	if _just_pressed(KEY_H) or _just_pressed(KEY_T):
		auto_level = not auto_level
		_physics.set_auto_level(auto_level)

	# Avionics / manual path: L
	if _just_pressed(KEY_L):
		avionics_mode = not avionics_mode
		if avionics_mode:
			_physics.set_avionics_command(aileron, elevator, rudder, throttle)
		else:
			_physics.set_controls(elevator, aileron, rudder, throttle, flaps_deg)

	# Avionics component panel: P
	if _just_pressed(KEY_P) and _panel:
		_panel.visible = not _panel.visible

	# Camera view toggle: V
	if _just_pressed(KEY_V):
		cam_orbit = not cam_orbit
		if cam_orbit:
			cam_center = _drone.global_position
			var offset := _camera.global_position - cam_center
			cam_dist = maxf(offset.length(), 1.0)
			cam_yaw = atan2(offset.z, offset.x)
			cam_pitch = asin(clampf(offset.y / cam_dist, -1.0, 1.0))

	# Switch aircraft: M
	if _just_pressed(KEY_M):
		_aircraft_index = (_aircraft_index + 1) % AIRCRAFT_NAMES.size()
		_load_aircraft(AIRCRAFT_NAMES[_aircraft_index])
		_update_aircraft_btn_text()

	# Re-roll the procedural world: N
	if _just_pressed(KEY_N):
		_regenerate_procedural_terrain()

	# Switch procedural terrain / imported landscape: B
	if _just_pressed(KEY_B):
		procedural_terrain = not procedural_terrain
		if procedural_terrain:
			_setup_procedural_terrain()
		else:
			_teardown_procedural_terrain()
		_save_terrain_settings()

	# Reset: R (re-trims to the active mode's cruise altitude / speed)
	if _just_pressed(KEY_R):
		var trm: Vector2 = MODE_TRIM.get(AIRCRAFT_NAMES[_aircraft_index], Vector2(800.0, 60.0))
		var tr: Vector2 = _physics.trim(trm.x, trm.y)
		elevator = 0.0
		elevator_trim = tr.x
		aileron = 0.0
		rudder = 0.0
		flaps_deg = 0.0
		throttle = clampf(tr.y, 0.0, 1.0)
		engine_out = 0
		auto_level = false
		_physics.set_auto_level(false)

	# Esc: back to the main menu
	if _just_pressed(KEY_ESCAPE):
		get_tree().change_scene_to_file("res://scenes/main_menu.tscn")

## Move `value` toward zero at `rate * mult` per second (stick self-centering).
func _center_control(value: float, rate: float, mult: float, delta: float) -> float:
	if absf(value) < 0.01:
		return 0.0
	return value - signf(value) * rate * mult * delta

## Map the engine-out state to a throttle-split for the twin physics:
## +1 = left engine out, 0 = both running, -1 = right engine out
## (matching `flight_core` `throttle_split`).
func engine_out_side() -> int:
	return 1 if engine_out == 1 else (-1 if engine_out == 2 else 0)

## True only on the rising edge of `key` (physical keycode space, matching
## `_input` which records `physical_keycode`).
func _just_pressed(key: Key) -> bool:
	return Input.is_physical_key_pressed(key) and not _held.get(key, false)

var _held := {}

func _input(event: InputEvent) -> void:
	if event is InputEventKey and event.pressed:
		_held[event.physical_keycode] = true
	elif event is InputEventKey and not event.pressed:
		_held[event.physical_keycode] = false

	if event is InputEventMouseButton and event.pressed:
		if event.button_index == MOUSE_BUTTON_WHEEL_UP:
			cam_dist = clampf(cam_dist - 3.0, 8.0, 120.0)
		elif event.button_index == MOUSE_BUTTON_WHEEL_DOWN:
			cam_dist = clampf(cam_dist + 3.0, 8.0, 120.0)
	if event is InputEventMouseMotion and Input.is_mouse_button_pressed(MOUSE_BUTTON_RIGHT):
		if not cam_orbit:
			cam_orbit = true
			cam_center = _drone.global_position
		cam_yaw -= event.relative.x * 0.006
		cam_pitch = clampf(cam_pitch - event.relative.y * 0.006, -1.4, 1.4)

## Rotate the visual flap and aileron meshes to match the physics control
## deflections (hinge along span Z, deflection about local Z).
func _update_control_surfaces() -> void:
	for flap in _flaps:
		if flap is Node3D:
			flap.rotation.z = flaps_deg * PI / 180.0
	for ail in _ailerons:
		if ail is Node3D:
			ail.rotation.z = -aileron * 0.6

## Position the camera: either a fixed chase offset behind the drone, or a
## free orbit around `cam_center` driven by the yaw/pitch/dist parameters.
func _chase_camera() -> void:
	# Follow the drone from a fixed offset each frame in GLOBAL space. The
	# camera is NOT a child of the drone so it keeps a horizon-stable up while
	# tracking the aircraft (prevents the "locked to ground" pitching bug).
	var target := _drone.global_position
	if cam_orbit:
		var sp := sin(cam_pitch)
		var cp := cos(cam_pitch)
		var offset := Vector3(cam_dist * cp * cos(cam_yaw), cam_dist * sp, cam_dist * cp * sin(cam_yaw))
		cam_center = target
		_camera.global_position = cam_center + offset
		_camera.look_at(cam_center, Vector3.UP)
	else:
		var tf: Transform3D = _drone.global_transform
		_camera.global_position = tf * Vector3(-32, 7.0, 0)
		_camera.look_at(tf.origin, Vector3.UP)

## Build the avionics component-visualization panel (right side of the HUD
## canvas): a live readout of every component on the bus plus a fault-injection
## toggle for each. Updates come from `_physics.avionics_snapshot()`.
func _build_avionics_panel(hud: CanvasLayer) -> void:
	var panel := PanelContainer.new()
	panel.name = "AvionicsPanel"
	var style := StyleBoxFlat.new()
	style.bg_color = Color(0.02, 0.12, 0.09, 0.88)
	style.set_corner_radius_all(6)
	style.set_content_margin_all(10)
	panel.add_theme_stylebox_override("panel", style)
	panel.position = Vector2(12, 12)
	hud.add_child(panel)

	var vbox := VBoxContainer.new()
	panel.add_child(vbox)

	var values := RichTextLabel.new()
	values.name = "Values"
	values.bbcode_enabled = true
	values.fit_content = true
	values.scroll_active = false
	values.add_theme_font_size_override("normal_font_size", 12)
	vbox.add_child(values)

	var faults_header := Label.new()
	faults_header.text = "FAILURE INJECTION"
	faults_header.add_theme_font_size_override("font_size", 12)
	vbox.add_child(faults_header)

	var grid := GridContainer.new()
	grid.columns = 2
	vbox.add_child(grid)
	for fault in _FAULTS:
		var btn := CheckButton.new()
		btn.text = fault.label
		btn.add_theme_font_size_override("font_size", 11)
		btn.toggled.connect(_on_fault_toggled.bind(fault.name))
		grid.add_child(btn)
		_fault_buttons[fault.name] = btn

	_panel = panel
	_panel_values = values

	# Dock the panel on the right edge using its content minimum size.
	var vp_w := float(hud.get_viewport().get_visible_rect().size.x)
	var min_w := float(panel.get_combined_minimum_size().x)
	panel.position = Vector2(maxf(vp_w - min_w - 12.0, 12.0), 12.0)

## React to a fault CheckButton: inject or clear the named fault in Rust.
func _on_fault_toggled(on: bool, name: String) -> void:
	if on:
		_physics.inject_fault(name)
	else:
		_physics.clear_fault(name)

## A red "! FAIL" warning tag when the matching bus fault flag is set.
func _fault_note(failed: bool) -> String:
	return " [color=#ff5454][b]! FAIL[/b][/color]" if failed else ""

## Rewrite the avionics panel from the latest `avionics_snapshot()` array,
## and keep the fault toggle buttons in sync with the bus fault flags.
func _update_avionics_panel() -> void:
	if _panel == null or not _panel.visible:
		return
	var snap := _avionics_snap
	if snap.size() < 46:
		return
	var mode := "ATTITUDE"
	if snap[4] < 0.5:
		mode = "RATE"
	elif snap[4] >= 1.5:
		mode = "MANUAL"
	var path := "AVIONICS" if avionics_mode else "MANUAL BYPASS"

	var txt := "[b]AVIONICS STACK[/b]  fc=[color=#5ad7ff]%s[/color]  path=[color=#7cfc00]%s[/color]\n" % [mode, path]
	txt += "─────────────────────────\n"
	txt += "[b]AGENT → FC (commands)[/b]\n"
	txt += "  roll %+0.003f  pitch %+0.003f  yaw %+0.003f  thr %0.3f\n" % [snap[0], snap[1], snap[2], snap[3]]
	txt += "[b]FLIGHT CONTROL (PID → servos/ESC)[/b]\n"
	txt += "  elev %+0.1f°   ail %+0.1f°   rud %+0.1f°   esc %0.3f\n" % [snap[5], snap[6], snap[7], snap[8]]
	txt += "[b]IMU[/b]%s\n" % _fault_note(snap[35] > 0.5)
	txt += "  gyro  %+0.003f / %+0.003f / %+0.003f rad/s\n" % [snap[9], snap[10], snap[11]]
	txt += "  accel %+0.1f / %+0.1f / %+0.1f m/s²\n" % [snap[12], snap[13], snap[14]]
	txt += "[b]GPS[/b]%s\n" % _fault_note(snap[36] > 0.5)
	txt += "  N %+0.1f   E %+0.1f   alt %0.1f m\n" % [snap[15], snap[16], snap[17]]
	txt += "  N/E/D %+0.1f/%+0.1f/%+0.1f m/s  fix %d  HDOP %0.1f\n" % [snap[18], snap[19], snap[20], int(snap[21]), snap[22]]
	txt += "[b]BARO[/b]%s\n" % _fault_note(snap[37] > 0.5)
	txt += "  altitude %0.1f m\n" % snap[23]
	txt += "[b]MAG[/b]%s\n" % _fault_note(snap[38] > 0.5)
	txt += "  field %+0.0f / %+0.0f / %+0.0f nT\n" % [snap[24], snap[25], snap[26]]
	txt += "[b]AIRSPEED[/b]%s\n" % _fault_note(snap[39] > 0.5)
	txt += "  IAS %0.1f m/s\n" % snap[27]
	txt += "[b]SERVOS (actual feedback)[/b]\n"
	txt += "  elev %+0.1f°%s  ail %+0.1f°%s  rud %+0.1f°%s\n" % [snap[31], _fault_note(snap[40] > 0.5), snap[32], _fault_note(snap[41] > 0.5), snap[33], _fault_note(snap[42] > 0.5)]
	txt += "[b]ESC[/b]%s\n" % _fault_note(snap[43] > 0.5)
	txt += "  output %0.3f\n" % snap[34]
	txt += "[b]BATTERY[/b]%s\n" % _fault_note(snap[44] > 0.5)
	txt += "  %0.2f V   %0.2f A   %0.1f %% left\n" % [snap[28], snap[29], snap[30]]
	txt += "─────────────────────────\n"
	txt += "sim t = %0.1f s\n" % snap[45]
	_panel_values.set_text(txt)

	var names := ["imu", "gps", "baro", "mag", "airspeed", "servo_elevator", "servo_aileron", "servo_rudder", "esc", "battery_depleted"]
	for i in names.size():
		var btn: CheckButton = _fault_buttons[names[i]]
		btn.set_pressed_no_signal(snap[35 + i] > 0.5)

