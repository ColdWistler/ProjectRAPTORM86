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
var auto_level := false

const AIRCRAFT_NAMES := ["TwinEngine", "MQI"]
const _AircraftViewScript := preload("res://scripts/aircraft_view.gd")

var _propellers: Array = []
var _flaps: Array = []
var _ailerons: Array = []
var _aircraft_index := 0
var _telemetry := PackedFloat64Array()
var _hud_timer := 0.0
var _aircraft_btn: Button = null

var cam_orbit := false
var cam_yaw := 0.0
var cam_pitch := 0.3
var cam_dist := 40.0
var cam_center := Vector3.ZERO

@onready var _physics = $Physics
@onready var _drone: Node3D = $DroneView
@onready var _camera: Camera3D = $Camera
@onready var _label: Label = _build_hud()

func _ready() -> void:
	var is_editor := Engine.is_editor_hint()

	# In the editor we only want the static visuals; the physics (Rust) node
	# and input/HUD handling are runtime-only.
	if not is_editor:
		if not _physics.start("TwinEngine.toml"):
			if not _physics.start("aircraft.toml"):
				push_error("FlightSimNode failed to load any aircraft config")
			# Trim still works on the built-in defaults, so continue anyway.
		var tr: Vector2 = _physics.trim(50.0, 60.0)
		elevator = 0.0
		elevator_trim = tr.x
		throttle = clampf(tr.y, 0.0, 1.0)

	# Use the imported GLB aircraft models rather than the procedural drone.
	var view: Node3D = _AircraftViewScript.new()
	view.name = "Model"
	_drone.add_child(view)
	_load_aircraft(AIRCRAFT_NAMES[_aircraft_index])

	if not is_editor:
		cam_center = _drone.global_position
		_chase_camera()

## Switch the active aircraft (visual model + physics config) and re-trim.
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
		var tr: Vector2 = _physics.trim(50.0, 60.0)
		elevator = 0.0
		elevator_trim = tr.x
		aileron = 0.0
		rudder = 0.0
		flaps_deg = 0.0
		throttle = clampf(tr.y, 0.0, 1.0)

## Build the on-screen HUD (telemetry panel + aircraft-swap button).
func _build_hud() -> Label:
	var hud := CanvasLayer.new()
	hud.name = "HUDCanvas"
	add_child(hud)
	var panel := PanelContainer.new()
	var style := StyleBoxFlat.new()
	style.bg_color = Color(0.04, 0.07, 0.11, 0.85)
	style.set_corner_radius_all(6)
	style.set_content_margin_all(12)
	panel.add_theme_stylebox_override("panel", style)
	panel.position = Vector2(12, 12)
	hud.add_child(panel)
	var label := Label.new()
	label.add_theme_font_size_override("font_size", 13)
	label.text = "Initializing..."
	panel.add_child(label)

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

	return label

func _on_aircraft_swap() -> void:
	_aircraft_index = (_aircraft_index + 1) % AIRCRAFT_NAMES.size()
	_load_aircraft(AIRCRAFT_NAMES[_aircraft_index])
	_update_aircraft_btn_text()

func _update_aircraft_btn_text() -> void:
	if _aircraft_btn:
		var next: String = AIRCRAFT_NAMES[(_aircraft_index + 1) % AIRCRAFT_NAMES.size()]
		_aircraft_btn.text = "Aircraft: %s  [Swap]" % AIRCRAFT_NAMES[_aircraft_index]
		_aircraft_btn.tooltip_text = "Click or press [M] to switch to %s" % next

func _physics_process(delta: float) -> void:
	if Engine.is_editor_hint():
		return

	_handle_input(delta)

	_physics.set_controls(elevator, aileron, rudder, throttle, flaps_deg)
	_physics.set_elevator_trim(elevator_trim)
	_physics.step(delta)

	_drone.transform = _physics.get_drone_transform()
	_update_control_surfaces()
	_chase_camera()

	for prop in _propellers:
		if prop is Node3D:
			prop.rotate_x(delta * (throttle * 60.0 + 3.0))

	_hud_timer += delta
	if _hud_timer >= 0.2:
		_hud_timer = 0.0
		_telemetry = _physics.telemetry()
		_update_hud()

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

	# Reset: R
	if _just_pressed(KEY_R):
		var tr: Vector2 = _physics.reset()
		elevator = 0.0
		elevator_trim = tr.x
		aileron = 0.0
		rudder = 0.0
		flaps_deg = 0.0
		throttle = clampf(tr.y, 0.0, 1.0)
		auto_level = false
		_physics.set_auto_level(false)

	# Esc: back to the main menu
	if _just_pressed(KEY_ESCAPE):
		get_tree().change_scene_to_file("res://scenes/main_menu.tscn")

func _center_control(value: float, rate: float, mult: float, delta: float) -> float:
	if absf(value) < 0.01:
		return 0.0
	return value - signf(value) * rate * mult * delta

func _just_pressed(key: Key) -> bool:
	return Input.is_key_pressed(key) and not _held.get(key, false)

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

func _update_control_surfaces() -> void:
	for flap in _flaps:
		if flap is Node3D:
			flap.rotation.x = flaps_deg * PI / 180.0
	for ail in _ailerons:
		if ail is Node3D:
			ail.rotation.x = -aileron * 0.6

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

func _update_hud() -> void:
	if _telemetry.size() < 25:
		return
	var t := _telemetry
	var flap_str := "UP (0deg)"
	if flaps_deg < 20.0:
		flap_str = "TAKEOFF (15deg)"
	if flaps_deg >= 20.0:
		flap_str = "LANDING (30deg)"
	var ap := " [AUTOPILOT ON]" if auto_level else ""
	var stall := " [! STALL !]" if t[23] > 0.5 else ""
	var ac_name: String = AIRCRAFT_NAMES[_aircraft_index]
	_label.text = """TACTICAL UAV DRONE TELEMETRY%s%s
Aircraft:       %s (model: %s, press [M] to switch)
------------------------------------
Altitude:       %6.0f m (%6.0f ft)
Airspeed (TAS): %6.1f m/s (%5.0f kts)
Ground Speed:   %6.1f m/s
Wind:           %5.1f m/s from %3.0f deg
Airspeed (IAS): %6.0f kts  (Mach %4.2f)
Dyn. Pressure:  %6.0f Pa  (OAT: %+4.1f degC)
AoA / Slip:     %+5.1f deg / %+5.1f deg
Pitch / Roll:   %+5.1f deg / %+5.1f deg
Heading (Yaw):  %5.1f deg (Climb: %+4.1f deg)
Throttle:       %5.0f %%  (Flaps: %s)
Surfaces:       Ail: %+4.1f deg | Elev: %+4.1f deg (Trim %+4.1f) | Rud: %+4.1f deg

CONTROLS & DRONE SYSTEMS
------------------------------------
Pitch:     [W] Down / [S] Up
Roll:      [A] Left / [D] Right
Rudder:    [Q] Left / [E] Right (or [Z]/[C])
Flaps:     [F] 0 -> 15 -> 30
Trim:      [[] Down / []] Up
Throttle:  [Shift] Up / [Ctrl] Down
	Autopilot: [H]/[T] Hold    Reset: [R]
	Camera:   [V] Chase/Orbit  [RMB-drag] [Scroll]
Menu: [Esc]""" % [
		ap, stall,
		ac_name, ac_name,
		t[0], t[1],
		t[2], t[3],
		t[4],
		t[21], t[22],
		t[5], t[6],
		t[7], t[8],
		t[9], t[10],
		t[11], t[12],
		t[13], t[14],
		t[15], flap_str,
		t[17], t[18], t[19], t[20],
	]