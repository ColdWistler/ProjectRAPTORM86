extends Control
## Main menu with tab-based navigation: Home, Aircraft, Terrain, About.
## The entire UI is built procedurally so the scene file stays minimal.

# ── Colour palette ───────────────────────────────────────────────────
const BG_DARK      := Color(0.035, 0.05, 0.085)
const BG_PANEL     := Color(0.07, 0.09, 0.14)
const BG_CARD      := Color(0.09, 0.11, 0.17)
const BG_CARD_HVR  := Color(0.12, 0.15, 0.22)
const ACCENT       := Color(0.30, 0.65, 0.95)
const ACCENT_DIM   := Color(0.22, 0.45, 0.68)
const TEXT_LIGHT   := Color(0.88, 0.92, 0.97)
const TEXT_MID     := Color(0.55, 0.63, 0.75)
const TEXT_DIM     := Color(0.38, 0.46, 0.58)
const TEXT_SPEC    := Color(0.45, 0.75, 0.95)
const BORDER       := Color(0.15, 0.19, 0.28)
const BORDER_LIGHT := Color(0.20, 0.26, 0.36)

# ── Aircraft data (mirrors the TOML configs) ────────────────────────
const AIRCRAFT := {
	"MQI": {
		"role": "Light Tactical UAV",
		"desc": "Single-engine propeller-driven tactical reconnaissance drone. Optimised for low-speed loiter and short-range missions with a compact, lightweight airframe.",
		"mass": 620.0, "span": 7.8, "area": 7.5, "chord": 0.96,
		"propulsion": "Single piston engine",
		"thrust": 1100.0, "power": 78000.0,
		"cl0": 0.32, "cla": 5.4, "cd0": 0.026, "k": 0.05,
	},
	"TwinEngine": {
		"role": "Long-Endurance Twin-Prop UAV",
		"desc": "High-aspect-ratio twin-engine platform built for range and endurance. Counter-rotating propellers cancel net torque; twin-engine-out yaw training is built in.",
		"mass": 1700.0, "span": 26.0, "area": 34.0, "chord": 1.31,
		"propulsion": "Twin counter-rotating props",
		"thrust": 2000.0, "power": 160000.0,
		"cl0": 0.25, "cla": 5.3, "cd0": 0.020, "k": 0.0188,
	},
	"Engine": {
		"role": "Fast Twin-Turbofan Jet",
		"desc": "Swept-wing business-jet class aircraft with two turbofans. Jet thrust stays roughly constant with speed, enabling fast, long-range cruise past the propeller power ceiling.",
		"mass": 4400.0, "span": 16.5, "area": 26.0, "chord": 1.58,
		"propulsion": "Twin turbofan (jet)",
		"thrust": 18000.0, "power": 0.0,
		"cl0": 0.15, "cla": 4.9, "cd0": 0.0170, "k": 0.038,
		"mach_crit": 0.78,
	},
}

## Per-aircraft 3D model alignment (mirrors `aircraft_view.gd`). `rot` brings
## the raw GLB axis convention onto body axes (nose +X, up +Y, right +Z);
## `scale` normalizes it to a usable on-screen size. `"procedural": true`
## builds the jet from primitives via `EngineFactory`.
const MODEL_ALIGN := {
	"MQI": {
		"scene": "res://assets/aircraft/MQI.glb",
		"rot": Vector3(0, 90, 0),
		"scale": 0.35,
		"cam_dist": 5.5,
	},
	"TwinEngine": {
		"scene": "res://assets/aircraft/TwinEngine.glb",
		"rot": Vector3(0, 0, 0),
		"scale": 0.16,
		"cam_dist": 7.5,
	},
	"Engine": {
		"procedural": true,
		"cam_dist": 32.0,
	},
}

const _EngineFactoryScript := preload("res://scripts/engine_factory.gd")
const _TerrainGeneratorScript := preload("res://scripts/terrain/terrain_generator.gd")

const TABS := ["Home", "Aircraft", "Terrain", "About"]
const TERRAIN_SETTINGS_META := "raptor_terrain_settings"
const TERRAIN_SETTING_KEYS := [
	"procedural_terrain",
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
const TERRAIN_DEFAULTS := {
	"procedural_terrain": true,
	"noise_seed": 1337,
	"noise_scale": 900.0,
	"octaves": 5,
	"persistence": 0.4,
	"lacunarity": 2.0,
	"height_multiplier": 700.0,
	"height_exponent": 1.25,
	"view_chunks": 24,
	"resolution": 33,
	"chunk_size": 500.0,
	"collision_enabled": true,
	"physics_grid_enabled": true,
	"wind_speed": 0.0,
	"wind_direction": 0.0,
}
const TERRAIN_RESOLUTIONS := [17, 25, 33, 49]

## Live-preview recipe: the terrain tab renders a compact chunk ring (not the
## full in-flight world) so the sliders update in real time. It mirrors the
## noise + height + colour regions exactly; only the chop is coarser.
const PREVIEW_CHUNK_SIZE := 400.0
const PREVIEW_RESOLUTION := 17
const PREVIEW_VIEW_CHUNKS := 3
const PREVIEW_GEN_KEYS := [
	"noise_seed",
	"noise_scale",
	"octaves",
	"persistence",
	"lacunarity",
	"height_multiplier",
	"height_exponent",
]

const AVIONICS := {
	"IMU": "Inertial Measurement Unit — 6-axis accelerometer/gyroscope (MEMS ICM-42688-P class) providing body-frame angular rates and specific force.",
	"GPS": "Global Positioning System — lat/lon/alt at 5 Hz with selectable noise model; provides Earth-frame position and ground speed.",
	"Baro": "Barometric Altimeter — pressure-derived altitude with standard atmosphere conversion and drift modelling.",
	"Magnetometer": "3-axis magnetic compass — heading reference for yaw estimation; subject to hard/soft-iron interference.",
	"Airspeed": "Pitot-static airspeed sensor — measures dynamic pressure for true airspeed; includes lag, noise, and blockage models.",
	"Flight Controller": "PID attitude controller — roll/pitch/yaw-rate inner loops with integral windup protection; mixer converts rates to servo commands.",
	"Servos": "Actuator suite — elevator, aileron, rudder servo models with rate limits, deadband, and position quantisation.",
	"ESC": "Electronic Speed Controller — models throttle command-to-motor-response delay and efficiency.",
	"Battery": "LiPo battery — internal-resistance Thevenin model; tracks voltage sag under load and remaining capacity.",
}

# ── State ────────────────────────────────────────────────────────────
var _pages: Dictionary = {}
var _current_page := "Home"
var _tab_buttons: Dictionary = {}
var _tab_styles: Dictionary = {}
var _terrain_settings: Dictionary = {}
var _terrain_summary: Label = null
var _terrain_procedural_controls: Array = []

# ── Terrain-tab live preview state ──────────────────────────────────
var _terrain_viewport: SubViewport = null
var _terrain_viewport_cam: Camera3D = null
var _terrain_preview_gen = null
var _terrain_preview_arrow: Node3D = null
var _terrain_wind_label: Label = null
var _terrain_preview_yaw := -0.9

# ── 3D viewport state ────────────────────────────────────────────────
var _ac_containers: Dictionary = {}   # name → SubViewportContainer
var _ac_viewports: Dictionary = {}    # name → SubViewport
var _ac_models: Dictionary = {}       # name → Node3D (turntable root)
var _ac_spin_root: Dictionary = {}    # name → Node3D (model wrapper that rotates)

# ── Build ────────────────────────────────────────────────────────────
func _ready() -> void:
	_terrain_settings = _load_terrain_settings()
	_build_ui()
	_switch_page("Home")

## Drive the animated 3D previews: spin the aircraft turntables on the Aircraft
## page and gently orbit + refresh the terrain preview on the Terrain page.
func _process(delta: float) -> void:
	if _current_page == "Aircraft":
		var speed := 0.35 * delta
		for name in _ac_spin_root:
			var root: Node3D = _ac_spin_root[name]
			if root != null:
				root.rotation.y += speed
		return
	if _current_page == "Terrain":
		_update_terrain_preview_camera(delta)

func _build_ui() -> void:
	# Full-screen dark background
	var bg := ColorRect.new()
	bg.color = BG_DARK
	bg.set_anchors_preset(Control.PRESET_FULL_RECT)
	add_child(bg)

	# Root margin container
	var root := MarginContainer.new()
	root.set_anchors_preset(Control.PRESET_FULL_RECT)
	root.add_theme_constant_override("margin_left", 0)
	root.add_theme_constant_override("margin_right", 0)
	root.add_theme_constant_override("margin_top", 0)
	root.add_theme_constant_override("margin_bottom", 0)
	add_child(root)

	var vbox := VBoxContainer.new()
	vbox.add_theme_constant_override("separation", 0)
	root.add_child(vbox)

	# ── Navigation bar ──────────────────────────────────────────────
	var nav_bar := _build_nav_bar()
	vbox.add_child(nav_bar)

	# ── Page container ──────────────────────────────────────────────
	var page_holder := MarginContainer.new()
	page_holder.size_flags_vertical = Control.SIZE_EXPAND_FILL
	page_holder.add_theme_constant_override("margin_left", 40)
	page_holder.add_theme_constant_override("margin_right", 40)
	page_holder.add_theme_constant_override("margin_top", 24)
	page_holder.add_theme_constant_override("margin_bottom", 24)
	vbox.add_child(page_holder)

	_pages["Home"] = _build_home_page()
	_pages["Aircraft"] = _build_aircraft_page()
	_pages["Terrain"] = _build_terrain_page()
	_pages["About"] = _build_about_page()

	for key in _pages:
		_pages[key].visible = false
		page_holder.add_child(_pages[key])

# ── Navigation bar ───────────────────────────────────────────────────
func _build_nav_bar() -> PanelContainer:
	var bar := PanelContainer.new()
	bar.custom_minimum_size.y = 56

	var style := StyleBoxFlat.new()
	style.bg_color = Color(0.04, 0.06, 0.10)
	style.border_width_bottom = 2
	style.border_color = BORDER
	style.content_margin_left = 40
	style.content_margin_right = 40
	style.content_margin_top = 8
	style.content_margin_bottom = 8
	bar.add_theme_stylebox_override("panel", style)

	var hflow := HBoxContainer.new()
	hflow.add_theme_constant_override("separation", 8)
	hflow.alignment = BoxContainer.ALIGNMENT_CENTER
	bar.add_child(hflow)

	# Project title
	var title := Label.new()
	title.text = "PROJECT RAPTOR M86"
	title.add_theme_font_size_override("font_size", 20)
	title.add_theme_color_override("font_color", ACCENT)
	title.size_flags_vertical = Control.SIZE_SHRINK_CENTER
	hflow.add_child(title)

	var spacer := Control.new()
	spacer.size_flags_horizontal = Control.SIZE_EXPAND_FILL
	hflow.add_child(spacer)

	# Tab buttons
	for tab_name in TABS:
		var btn := Button.new()
		btn.text = tab_name
		btn.custom_minimum_size = Vector2(120, 36)
		btn.add_theme_font_size_override("font_size", 16)
		btn.add_theme_color_override("font_color", TEXT_MID)
		btn.add_theme_color_override("font_hover_color", TEXT_LIGHT)

		var btn_style := StyleBoxFlat.new()
		btn_style.bg_color = Color.TRANSPARENT
		btn_style.corner_radius_top_left = 6
		btn_style.corner_radius_top_right = 6
		btn_style.corner_radius_bottom_left = 6
		btn_style.corner_radius_bottom_right = 6
		btn_style.content_margin_left = 16
		btn_style.content_margin_right = 16
		btn_style.content_margin_top = 6
		btn_style.content_margin_bottom = 6
		btn.add_theme_stylebox_override("normal", btn_style)

		var btn_hover := btn_style.duplicate()
		btn_hover.bg_color = Color(0.12, 0.15, 0.22)
		btn.add_theme_stylebox_override("hover", btn_hover)

		var btn_active := btn_style.duplicate()
		btn_active.bg_color = Color(0.16, 0.22, 0.32)
		btn_active.border_width_bottom = 2
		btn_active.border_color = ACCENT

		_tab_styles[tab_name] = {
			"normal": btn_style,
			"hover": btn_hover,
			"active": btn_active,
		}

		btn.pressed.connect(_bind_tab(tab_name))
		_tab_buttons[tab_name] = btn
		hflow.add_child(btn)

	# Version label
	var ver := Label.new()
	ver.text = "v0.1  |  Rust + Godot 4"
	ver.add_theme_font_size_override("font_size", 12)
	ver.add_theme_color_override("font_color", TEXT_DIM)
	ver.size_flags_vertical = Control.SIZE_SHRINK_CENTER
	hflow.add_child(ver)

	return bar

func _bind_tab(tab_name: String) -> Callable:
	return func() -> void:
		_switch_page(tab_name)

func _switch_page(name: String) -> void:
	_current_page = name
	for key in _pages:
		_pages[key].visible = (key == name)
	# Pause 3D preview rendering unless its page is active.
	var aircraft_active := (name == "Aircraft")
	for key in _ac_viewports:
		var vp: SubViewport = _ac_viewports[key]
		if vp != null:
			vp.render_target_update_mode = (
				SubViewport.UPDATE_ALWAYS if aircraft_active
				else SubViewport.UPDATE_DISABLED)
	if _terrain_viewport != null:
		_terrain_viewport.render_target_update_mode = (
			SubViewport.UPDATE_ALWAYS if name == "Terrain"
			else SubViewport.UPDATE_DISABLED)
	# Update tab button styles
	for key in _tab_buttons:
		var styles: Dictionary = _tab_styles[key]
		if key == name:
			_tab_buttons[key].add_theme_stylebox_override("normal", styles["active"])
			_tab_buttons[key].add_theme_color_override("font_color", ACCENT)
			_tab_buttons[key].add_theme_color_override("font_hover_color", ACCENT)
		else:
			_tab_buttons[key].add_theme_stylebox_override("normal", styles["normal"])
			_tab_buttons[key].add_theme_color_override("font_color", TEXT_MID)
			_tab_buttons[key].add_theme_color_override("font_hover_color", TEXT_LIGHT)

# ── HOME PAGE ────────────────────────────────────────────────────────
func _build_home_page() -> Control:
	var page := MarginContainer.new()
	page.add_theme_constant_override("margin_top", 20)

	var vbox := VBoxContainer.new()
	vbox.add_theme_constant_override("separation", 32)
	vbox.alignment = BoxContainer.ALIGNMENT_CENTER
	vbox.size_flags_horizontal = Control.SIZE_EXPAND_FILL
	vbox.size_flags_vertical = Control.SIZE_EXPAND_FILL
	page.add_child(vbox)

	# Hero title
	var hero := Label.new()
	hero.text = "PROJECT RAPTOR M86"
	hero.horizontal_alignment = HORIZONTAL_ALIGNMENT_CENTER
	hero.add_theme_font_size_override("font_size", 48)
	hero.add_theme_color_override("font_color", TEXT_LIGHT)
	vbox.add_child(hero)

	var subtitle := Label.new()
	subtitle.text = "6-DOF fixed-wing flight dynamics engine — Rust physics, Godot visualization"
	subtitle.horizontal_alignment = HORIZONTAL_ALIGNMENT_CENTER
	subtitle.add_theme_font_size_override("font_size", 18)
	subtitle.add_theme_color_override("font_color", TEXT_MID)
	vbox.add_child(subtitle)

	vbox.add_child(HSeparator.new())

	# Mode cards in an HBox
	var card_row := HBoxContainer.new()
	card_row.add_theme_constant_override("separation", 24)
	card_row.alignment = BoxContainer.ALIGNMENT_CENTER
	card_row.size_flags_horizontal = Control.SIZE_EXPAND_FILL
	vbox.add_child(card_row)

	card_row.add_child(_make_mode_card(
		"Flight Simulator",
		"Take off at 800 m / 60 m/s trimmed cruise, clear of the generated\nmountains. Chase camera, full control surface authority, flaps, trim,\nautopilot hold, wind.",
		"res://scenes/flight_sim.tscn"
	))
	card_row.add_child(_make_mode_card(
		"Wind Tunnel",
		"Fix the aircraft in place and visualise the smoke flow field.\nFlow lines react to the real CL — upwash, downwash, tip vortices.",
		"res://scenes/wind_tunnel.tscn"
	))

	# Bottom row: Quit + hint
	var bottom := HBoxContainer.new()
	bottom.alignment = BoxContainer.ALIGNMENT_CENTER
	bottom.add_theme_constant_override("separation", 24)
	vbox.add_child(bottom)

	var btn_quit := _make_action_button("Quit", 140, 44)
	btn_quit.pressed.connect(get_tree().quit)
	bottom.add_child(btn_quit)

	var hint := Label.new()
	hint.text = "Esc returns to this menu from any mode"
	hint.add_theme_font_size_override("font_size", 14)
	hint.add_theme_color_override("font_color", TEXT_DIM)
	hint.size_flags_vertical = Control.SIZE_SHRINK_CENTER
	bottom.add_child(hint)

	return page

func _make_mode_card(card_title: String, card_desc: String, scene_path: String) -> PanelContainer:
	var panel := PanelContainer.new()
	panel.custom_minimum_size = Vector2(380, 200)
	panel.size_flags_horizontal = Control.SIZE_EXPAND_FILL

	var style := StyleBoxFlat.new()
	style.bg_color = BG_CARD
	style.border_width_left = 2
	style.border_width_right = 2
	style.border_width_top = 2
	style.border_width_bottom = 2
	style.border_color = BORDER
	style.corner_radius_top_left = 10
	style.corner_radius_top_right = 10
	style.corner_radius_bottom_left = 10
	style.corner_radius_bottom_right = 10
	style.content_margin_left = 24
	style.content_margin_right = 24
	style.content_margin_top = 24
	style.content_margin_bottom = 20
	panel.add_theme_stylebox_override("panel", style)

	var inner := VBoxContainer.new()
	inner.add_theme_constant_override("separation", 10)
	panel.add_child(inner)

	var heading := Label.new()
	heading.text = card_title
	heading.add_theme_font_size_override("font_size", 24)
	heading.add_theme_color_override("font_color", ACCENT)
	inner.add_child(heading)

	var desc_label := Label.new()
	desc_label.text = card_desc
	desc_label.autowrap_mode = TextServer.AUTOWRAP_WORD_SMART
	desc_label.add_theme_font_size_override("font_size", 14)
	desc_label.add_theme_color_override("font_color", TEXT_MID)
	inner.add_child(desc_label)

	var spacer := Control.new()
	spacer.size_flags_vertical = Control.SIZE_EXPAND_FILL
	inner.add_child(spacer)

	var launch := _make_action_button("Launch  →", 160, 40)
	launch.pressed.connect(func() -> void:
		get_tree().change_scene_to_file(scene_path))
	inner.add_child(launch)

	# Hover effect
	panel.mouse_entered.connect(func() -> void:
		style.bg_color = BG_CARD_HVR)
	panel.mouse_exited.connect(func() -> void:
		style.bg_color = BG_CARD)

	return panel

# ── AIRCRAFT PAGE ────────────────────────────────────────────────────
func _build_aircraft_page() -> Control:
	var scroll := ScrollContainer.new()
	scroll.size_flags_horizontal = Control.SIZE_EXPAND_FILL
	scroll.size_flags_vertical = Control.SIZE_EXPAND_FILL
	scroll.horizontal_scroll_mode = ScrollContainer.SCROLL_MODE_DISABLED

	var vbox := VBoxContainer.new()
	vbox.add_theme_constant_override("separation", 8)
	vbox.size_flags_horizontal = Control.SIZE_EXPAND_FILL
	scroll.add_child(vbox)

	var heading := Label.new()
	heading.text = "Aircraft Models"
	heading.add_theme_font_size_override("font_size", 32)
	heading.add_theme_color_override("font_color", TEXT_LIGHT)
	vbox.add_child(heading)

	var sub := Label.new()
	sub.text = "Three aircraft are included — two propeller-driven UAV drones and one turbofan jet. Each has a full TOML configuration driving the Rust physics engine."
	sub.autowrap_mode = TextServer.AUTOWRAP_WORD_SMART
	sub.add_theme_font_size_override("font_size", 15)
	sub.add_theme_color_override("font_color", TEXT_MID)
	sub.custom_minimum_size.y = 40
	vbox.add_child(sub)

	vbox.add_child(HSeparator.new())

	var gap := Control.new()
	gap.custom_minimum_size.y = 4
	vbox.add_child(gap)

	for ac_name in ["MQI", "TwinEngine", "Engine"]:
		vbox.add_child(_make_aircraft_card(ac_name, AIRCRAFT[ac_name]))

	return scroll

func _make_aircraft_card(ac_name: String, data: Dictionary) -> PanelContainer:
	var panel := PanelContainer.new()
	panel.size_flags_horizontal = Control.SIZE_EXPAND_FILL

	var style := StyleBoxFlat.new()
	style.bg_color = BG_CARD
	style.border_width_left = 3
	style.border_width_right = 1
	style.border_width_top = 1
	style.border_width_bottom = 1
	style.border_color = BORDER
	style.corner_radius_top_left = 8
	style.corner_radius_top_right = 8
	style.corner_radius_bottom_left = 8
	style.corner_radius_bottom_right = 8
	style.content_margin_left = 28
	style.content_margin_right = 28
	style.content_margin_top = 24
	style.content_margin_bottom = 24
	panel.add_theme_stylebox_override("panel", style)

	var outer_h := HBoxContainer.new()
	outer_h.add_theme_constant_override("separation", 32)
	panel.add_child(outer_h)

	# ── Left: live 3D preview (rotating turntable) ──
	var preview := _build_3d_viewport(ac_name)
	preview.size_flags_vertical = Control.SIZE_SHRINK_CENTER
	outer_h.add_child(preview)

	# ── Right: name, role, description, specs ──
	var right := VBoxContainer.new()
	right.size_flags_horizontal = Control.SIZE_EXPAND_FILL
	right.add_theme_constant_override("separation", 8)
	outer_h.add_child(right)

	var name_row := HBoxContainer.new()
	name_row.add_theme_constant_override("separation", 12)
	right.add_child(name_row)

	var ac_label := Label.new()
	ac_label.text = ac_name
	ac_label.add_theme_font_size_override("font_size", 28)
	ac_label.add_theme_color_override("font_color", ACCENT)
	name_row.add_child(ac_label)

	var role_label := Label.new()
	role_label.text = data["role"]
	role_label.add_theme_font_size_override("font_size", 16)
	role_label.add_theme_color_override("font_color", TEXT_MID)
	role_label.size_flags_vertical = Control.SIZE_SHRINK_CENTER
	name_row.add_child(role_label)

	var desc := Label.new()
	desc.text = data["desc"]
	desc.autowrap_mode = TextServer.AUTOWRAP_WORD_SMART
	desc.add_theme_font_size_override("font_size", 13)
	desc.add_theme_color_override("font_color", TEXT_DIM)
	right.add_child(desc)

	right.add_child(HSeparator.new())

	# Specs table
	var specs := GridContainer.new()
	specs.columns = 2
	specs.add_theme_constant_override("h_separation", 12)
	specs.add_theme_constant_override("v_separation", 6)
	specs.size_flags_horizontal = Control.SIZE_EXPAND_FILL
	right.add_child(specs)

	_add_spec_row(specs, "Mass", "%.0f kg" % data["mass"])
	_add_spec_row(specs, "Wingspan", "%.1f m" % data["span"])
	_add_spec_row(specs, "Wing area", "%.1f m²" % data["area"])
	_add_spec_row(specs, "MAC chord", "%.2f m" % data["chord"])
	_add_spec_row(specs, "Propulsion", data["propulsion"])
	_add_spec_row(specs, "Max thrust", "%.0f N" % data["thrust"])
	if data["power"] > 0:
		_add_spec_row(specs, "Max power", "%.0f kW" % (data["power"] / 1000.0))
	if data.has("mach_crit"):
		_add_spec_row(specs, "Mach crit", "%.2f" % data["mach_crit"])

	specs.add_child(_make_spacer_label())  # row gap

	_add_spec_row(specs, "CL₀", "%.2f" % data["cl0"])
	_add_spec_row(specs, "CLα", "%.1f /rad" % data["cla"])
	_add_spec_row(specs, "CD₀", "%.4f" % data["cd0"])
	_add_spec_row(specs, "K (induced)", "%.4f" % data["k"])

	return panel

## Build a framed live 3D viewport previewing the given aircraft. The model
## spins slowly (see `_process`) and renders into a SubViewport layered over a
## radial-gradient backdrop. Registering the viewport lets `_switch_page`
## pause rendering whenever the Aircraft page is hidden.
func _build_3d_viewport(ac_name: String) -> Control:
	var frame := PanelContainer.new()
	frame.custom_minimum_size = Vector2(430, 350)

	var fstyle := StyleBoxFlat.new()
	fstyle.bg_color = Color(0.045, 0.06, 0.10)
	fstyle.border_width_left = 2
	fstyle.border_width_right = 2
	fstyle.border_width_top = 2
	fstyle.border_width_bottom = 2
	fstyle.border_color = BORDER_LIGHT
	fstyle.corner_radius_top_left = 10
	fstyle.corner_radius_top_right = 10
	fstyle.corner_radius_bottom_left = 10
	fstyle.corner_radius_bottom_right = 10
	fstyle.content_margin_left = 6
	fstyle.content_margin_right = 6
	fstyle.content_margin_top = 6
	fstyle.content_margin_bottom = 6
	frame.add_theme_stylebox_override("panel", fstyle)

	# Radial-gradient backdrop drawn behind the transparent viewport.
	var grad := Gradient.new()
	grad.offsets = PackedFloat32Array([0.0, 1.0])
	grad.colors = PackedColorArray([Color(0.09, 0.13, 0.21), Color(0.025, 0.035, 0.06)])
	var tex := GradientTexture2D.new()
	tex.gradient = grad
	tex.fill = GradientTexture2D.FILL_RADIAL
	tex.fill_from = Vector2(0.5, 0.35)
	tex.fill_to = Vector2(0.82, 1.0)

	var bg_tex := TextureRect.new()
	bg_tex.texture = tex
	bg_tex.stretch_mode = TextureRect.STRETCH_SCALE
	bg_tex.expand_mode = TextureRect.EXPAND_IGNORE_SIZE
	bg_tex.mouse_filter = Control.MOUSE_FILTER_IGNORE
	frame.add_child(bg_tex)

	# 3D layer
	var container := SubViewportContainer.new()
	container.stretch = true
	container.mouse_filter = Control.MOUSE_FILTER_IGNORE
	frame.add_child(container)

	var vp := SubViewport.new()
	vp.size = Vector2i(418, 338)
	vp.transparent_bg = true
	vp.own_world_3d = true
	vp.world_3d = World3D.new()
	vp.render_target_update_mode = SubViewport.UPDATE_DISABLED
	container.add_child(vp)

	# Camera — fixed 3/4 perspective framing the origin.
	var cam := Camera3D.new()
	cam.fov = 45.0
	var d := float(MODEL_ALIGN[ac_name]["cam_dist"])
	cam.look_at_from_position(Vector3(0.25 * d, 0.35 * d, 0.9 * d), Vector3.ZERO, Vector3.UP)
	vp.add_child(cam)

	# Key light (warm, with shadows).
	var key := DirectionalLight3D.new()
	key.rotation_degrees = Vector3(-45, -35, 0)
	key.light_energy = 1.3
	key.shadow_enabled = true
	vp.add_child(key)

	# Fill light (cool, softens the shadow side).
	var fill := DirectionalLight3D.new()
	fill.rotation_degrees = Vector3(35, 25, 0)
	fill.light_energy = 0.4
	fill.light_color = Color(0.55, 0.65, 0.85)
	vp.add_child(fill)

	# Rim / back light (cool blue) to pop the silhouette.
	var rim := DirectionalLight3D.new()
	rim.rotation_degrees = Vector3(-25, 150, 0)
	rim.light_energy = 0.55
	rim.light_color = Color(0.45, 0.6, 1.0)
	vp.add_child(rim)

	# Turntable root that holds the model.
	var spin := Node3D.new()
	vp.add_child(spin)
	_load_3d_model(ac_name, spin)

	_ac_containers[ac_name] = container
	_ac_viewports[ac_name] = vp
	_ac_models[ac_name] = spin.get_child(0) if spin.get_child_count() > 0 else null
	_ac_spin_root[ac_name] = spin
	return frame

## Load `ac_name`'s model under `root`. GLB models are aligned with the same
## rotation/scale convention as `aircraft_view.gd`; the jet is built from
## primitives via `EngineFactory`.
func _load_3d_model(ac_name: String, root: Node3D) -> void:
	var align: Dictionary = MODEL_ALIGN[ac_name]
	if align.get("procedural", false):
		_EngineFactoryScript.build(root)
		return
	var scene_path: String = align["scene"]
	if not ResourceLoader.exists(scene_path):
		push_error("MainMenu: missing model '%s' (expected at '%s')" % [ac_name, scene_path])
		return
	var packed: PackedScene = load(scene_path)
	if packed == null:
		push_error("MainMenu: failed to load '%s'" % scene_path)
		return
	var inst := packed.instantiate()
	if inst == null:
		push_error("MainMenu: failed to instantiate '%s'" % scene_path)
		return
	inst.rotation_degrees = align["rot"]
	var s := float(align["scale"])
	inst.scale = Vector3(s, s, s)
	root.add_child(inst)

func _add_spec_row(parent: Container, label_text: String, value_text: String) -> void:
	var lbl := Label.new()
	lbl.text = label_text
	lbl.add_theme_font_size_override("font_size", 13)
	lbl.add_theme_color_override("font_color", TEXT_DIM)
	lbl.custom_minimum_size.x = 120
	parent.add_child(lbl)

	var val := Label.new()
	val.text = value_text
	val.add_theme_font_size_override("font_size", 13)
	val.add_theme_color_override("font_color", TEXT_SPEC)
	parent.add_child(val)

func _make_spacer_label() -> Label:
	var lbl := Label.new()
	lbl.text = ""
	lbl.custom_minimum_size.y = 4
	return lbl

# ── TERRAIN PAGE ─────────────────────────────────────────────────────
func _build_terrain_page() -> Control:
	var scroll := ScrollContainer.new()
	scroll.size_flags_horizontal = Control.SIZE_EXPAND_FILL
	scroll.size_flags_vertical = Control.SIZE_EXPAND_FILL
	scroll.horizontal_scroll_mode = ScrollContainer.SCROLL_MODE_DISABLED

	var vbox := VBoxContainer.new()
	vbox.add_theme_constant_override("separation", 16)
	vbox.size_flags_horizontal = Control.SIZE_EXPAND_FILL
	scroll.add_child(vbox)

	var heading := Label.new()
	heading.text = "Terrain Generator"
	heading.add_theme_font_size_override("font_size", 32)
	heading.add_theme_color_override("font_color", TEXT_LIGHT)
	vbox.add_child(heading)

	var sub := Label.new()
	sub.text = "Configure the procedural landmass used by the flight simulator, or keep the imported landscape. Launching applies these settings to the flight scene."
	sub.autowrap_mode = TextServer.AUTOWRAP_WORD_SMART
	sub.add_theme_font_size_override("font_size", 15)
	sub.add_theme_color_override("font_color", TEXT_MID)
	vbox.add_child(sub)
	vbox.add_child(HSeparator.new())

	vbox.add_child(_build_terrain_preview())

	var source_card := _make_settings_card()
	vbox.add_child(source_card[0])
	var source_inner: VBoxContainer = source_card[1]
	source_inner.add_child(_make_settings_heading("Terrain source"))

	var mode_row := HBoxContainer.new()
	mode_row.add_theme_constant_override("separation", 12)
	source_inner.add_child(mode_row)
	mode_row.add_child(_make_settings_label("Source"))
	var mode := OptionButton.new()
	mode.add_item("Procedural landmass")
	mode.add_item("Imported landscape")
	mode.custom_minimum_size.x = 260
	mode.selected = 0 if bool(_terrain_settings["procedural_terrain"]) else 1
	mode_row.add_child(mode)
	mode.item_selected.connect(func(index: int) -> void:
		_set_terrain_setting("procedural_terrain", index == 0)
		_set_procedural_controls_enabled(index == 0)
	)

	var shape_card := _make_settings_card()
	vbox.add_child(shape_card[0])
	var shape_inner: VBoxContainer = shape_card[1]
	shape_inner.add_child(_make_settings_heading("Landmass shape"))
	_add_slider_setting(shape_inner, "Seed", 0.0, 100000.0, 1.0, float(_terrain_settings["noise_seed"]), "%.0f", "noise_seed")
	_add_slider_setting(shape_inner, "Noise scale", 40.0, 2000.0, 10.0, float(_terrain_settings["noise_scale"]), "%.0f", "noise_scale")
	_add_slider_setting(shape_inner, "Mountain height", 100.0, 1800.0, 25.0, float(_terrain_settings["height_multiplier"]), "%.0f m", "height_multiplier")
	_add_slider_setting(shape_inner, "Ruggedness", 0.6, 1.6, 0.05, float(_terrain_settings["height_exponent"]), "%.2f", "height_exponent")

	var detail_card := _make_settings_card()
	vbox.add_child(detail_card[0])
	var detail_inner: VBoxContainer = detail_card[1]
	detail_inner.add_child(_make_settings_heading("Detail and physics"))
	_add_slider_setting(detail_inner, "Noise octaves", 1.0, 6.0, 1.0, float(_terrain_settings["octaves"]), "%.0f", "octaves")
	_add_slider_setting(detail_inner, "Persistence", 0.1, 0.9, 0.05, float(_terrain_settings["persistence"]), "%.2f", "persistence")
	_add_slider_setting(detail_inner, "Lacunarity", 1.5, 3.0, 0.1, float(_terrain_settings["lacunarity"]), "%.2f", "lacunarity")
	_add_slider_setting(detail_inner, "Loaded chunks", 3.0, 12.0, 1.0, float(_terrain_settings["view_chunks"]), "%.0f", "view_chunks")
	detail_inner.add_child(_make_resolution_row())
	detail_inner.add_child(_make_check_setting("Terrain collision", "collision_enabled"))
	detail_inner.add_child(_make_check_setting("Physics height grid", "physics_grid_enabled"))

	var actions := HBoxContainer.new()
	actions.add_theme_constant_override("separation", 12)
	vbox.add_child(actions)
	var randomize := _make_action_button("Randomize seed", 200, 44)
	randomize.pressed.connect(_randomize_terrain_seed)
	actions.add_child(randomize)
	_terrain_procedural_controls.append(randomize)
	var reset := _make_action_button("Reset defaults", 200, 44)
	reset.pressed.connect(_reset_terrain_settings)
	actions.add_child(reset)
	var launch := _make_action_button("Generate & launch flight", 300, 44)
	launch.pressed.connect(_launch_flight_with_terrain)
	actions.add_child(launch)

	_terrain_summary = Label.new()
	_terrain_summary.autowrap_mode = TextServer.AUTOWRAP_WORD_SMART
	_terrain_summary.add_theme_font_size_override("font_size", 14)
	_terrain_summary.add_theme_color_override("font_color", TEXT_MID)
	vbox.add_child(_terrain_summary)
	_set_procedural_controls_enabled(bool(_terrain_settings["procedural_terrain"]))
	_update_terrain_summary()
	return scroll

## The Terrain-tab centrepiece: a live 3D render of the procedural world plus
## a wind-direction arrow, with the wind controls parked beside it. The chunk
## ring stays coarse (see PREVIEW_*) so the shape sliders respond instantly.
func _build_terrain_preview() -> PanelContainer:
	var panel := PanelContainer.new()
	panel.size_flags_horizontal = Control.SIZE_EXPAND_FILL

	var style := StyleBoxFlat.new()
	style.bg_color = BG_CARD
	style.border_width_left = 2
	style.border_width_right = 1
	style.border_width_top = 1
	style.border_width_bottom = 1
	style.border_color = ACCENT_DIM
	style.corner_radius_top_left = 8
	style.corner_radius_top_right = 8
	style.corner_radius_bottom_left = 8
	style.corner_radius_bottom_right = 8
	style.content_margin_left = 24
	style.content_margin_right = 24
	style.content_margin_top = 20
	style.content_margin_bottom = 20
	panel.add_theme_stylebox_override("panel", style)

	var outer := HBoxContainer.new()
	outer.add_theme_constant_override("separation", 24)
	panel.add_child(outer)

	# ── Left: live 3D preview (slowly orbiting camera) ──
	outer.add_child(_build_preview_viewport())

	# ── Right: heading + wind controls ──
	var right := VBoxContainer.new()
	right.size_flags_horizontal = Control.SIZE_EXPAND_FILL
	right.add_theme_constant_override("separation", 10)
	outer.add_child(right)

	right.add_child(_make_settings_heading("Preview"))
	var pdesc := Label.new()
	pdesc.text = "The generated world at the current landmass settings, rendered before you commit. The amber arrow points the direction the steady wind blows toward."
	pdesc.autowrap_mode = TextServer.AUTOWRAP_WORD_SMART
	pdesc.add_theme_font_size_override("font_size", 13)
	pdesc.add_theme_color_override("font_color", TEXT_DIM)
	pdesc.custom_minimum_size.y = 58
	right.add_child(pdesc)

	right.add_child(HSeparator.new())

	right.add_child(_make_settings_heading("Wind"))
	right.add_child(_make_settings_label_note(
		"Wind speed and bearing are applied to the flight sim on launch. A bearing of 0° blows toward north; 90° toward east."
	))
	_add_slider_setting(right, "Wind speed", 0.0, 30.0, 0.5, float(_terrain_settings["wind_speed"]), "%.1f m/s", "wind_speed", false)
	_add_slider_setting(right, "Wind direction", 0.0, 360.0, 5.0, float(_terrain_settings["wind_direction"]), "%.0f°", "wind_direction", false)

	var spacer := Control.new()
	spacer.size_flags_vertical = Control.SIZE_EXPAND_FILL
	right.add_child(spacer)

	return panel

## Build the framed SubViewport that renders the preview terrain and the wind
## arrow, plus a 2D overlay label reporting the current wind.
func _build_preview_viewport() -> Control:
	var frame := Control.new()
	frame.custom_minimum_size = Vector2(720, 450)

	var bg := PanelContainer.new()
	bg.set_anchors_preset(Control.PRESET_FULL_RECT)
	var fstyle := StyleBoxFlat.new()
	fstyle.bg_color = Color(0.045, 0.06, 0.10)
	fstyle.border_width_left = 2
	fstyle.border_width_right = 2
	fstyle.border_width_top = 2
	fstyle.border_width_bottom = 2
	fstyle.border_color = BORDER_LIGHT
	fstyle.corner_radius_top_left = 10
	fstyle.corner_radius_top_right = 10
	fstyle.corner_radius_bottom_left = 10
	fstyle.corner_radius_bottom_right = 10
	fstyle.content_margin_left = 6
	fstyle.content_margin_right = 6
	fstyle.content_margin_top = 6
	fstyle.content_margin_bottom = 6
	bg.add_theme_stylebox_override("panel", fstyle)
	frame.add_child(bg)

	# 3D layer
	var container := SubViewportContainer.new()
	container.stretch = true
	container.set_anchors_preset(Control.PRESET_FULL_RECT)
	container.mouse_filter = Control.MOUSE_FILTER_IGNORE
	frame.add_child(container)

	var vp := SubViewport.new()
	vp.size = Vector2i(708, 438)
	vp.own_world_3d = true
	vp.world_3d = World3D.new()
	vp.render_target_update_mode = SubViewport.UPDATE_DISABLED
	container.add_child(vp)
	_terrain_viewport = vp

	# Ambient environment (dim sky + sun so the region colours read).
	var we := WorldEnvironment.new()
	var env := Environment.new()
	env.background_mode = Environment.BG_COLOR
	env.background_color = Color(0.30, 0.38, 0.48)
	env.ambient_light_source = Environment.AMBIENT_SOURCE_COLOR
	env.ambient_light_color = Color(0.62, 0.72, 0.85)
	env.ambient_light_energy = 0.85
	we.environment = env
	vp.add_child(we)

	var sun := DirectionalLight3D.new()
	sun.rotation_degrees = Vector3(-50, -35, 0)
	sun.light_energy = 1.25
	sun.shadow_enabled = true
	vp.add_child(sun)

	var fill := DirectionalLight3D.new()
	fill.rotation_degrees = Vector3(30, 25, 0)
	fill.light_energy = 0.4
	fill.light_color = Color(0.55, 0.65, 0.85)
	vp.add_child(fill)

	# Terrain generator — coarse, collisionless. The apron flatten stays on
	# (matching the flight world) so the preview shows the airport valley the
	# aircraft actually spawns into, not raw landmass over the runway.
	var gen = _TerrainGeneratorScript.new()
	gen.name = "PreviewTerrain"
	vp.add_child(gen)
	gen.collision_enabled = false
	gen.physics_grid_enabled = false
	gen.chunk_size = PREVIEW_CHUNK_SIZE
	gen.resolution = PREVIEW_RESOLUTION
	gen.view_chunks = PREVIEW_VIEW_CHUNKS
	# A fixed origin probe lets the chunk ring build around the world origin
	# (the generator's own _process will populate it once the tab is active).
	var anchor := Node3D.new()
	anchor.name = "PreviewAnchor"
	vp.add_child(anchor)
	gen.set_target_node(anchor)
	_terrain_preview_gen = gen

	# Camera — slow orbit around the origin, tilted down onto the relief.
	var cam := Camera3D.new()
	cam.fov = 60.0
	cam.current = true
	vp.add_child(cam)
	_terrain_viewport_cam = cam

	# Wind arrow (shaft + cone head, oriented toward the wind's destination).
	var arrow := _make_wind_arrow()
	vp.add_child(arrow)
	_terrain_preview_arrow = arrow

	# 2D overlay label (screen space above the viewport contents).
	var wind_lbl := Label.new()
	wind_lbl.position = Vector2(14, 10)
	wind_lbl.mouse_filter = Control.MOUSE_FILTER_IGNORE
	wind_lbl.add_theme_color_override("font_color", Color(1.0, 0.85, 0.2, 0.95))
	wind_lbl.add_theme_color_override("font_outline_color", Color(0, 0, 0, 0.9))
	wind_lbl.add_theme_constant_override("outline_size", 4)
	wind_lbl.add_theme_font_size_override("font_size", 15)
	frame.add_child(wind_lbl)
	_terrain_wind_label = wind_lbl

	_apply_terrain_preview_settings()
	_update_terrain_wind_arrow()

	return frame

func _make_settings_label_note(text: String) -> Label:
	var lbl := Label.new()
	lbl.text = text
	lbl.autowrap_mode = TextServer.AUTOWRAP_WORD_SMART
	lbl.add_theme_font_size_override("font_size", 12)
	lbl.add_theme_color_override("font_color", TEXT_DIM)
	return lbl

## Build an unlit amber arrow pointing along +X (world north), sized relative
## to the preview footprint. Its parent node is rotated about Y to steer it
## toward the configured wind bearing.
func _make_wind_arrow() -> Node3D:
	var root := Node3D.new()
	root.name = "WindArrow"
	var footprint := PREVIEW_CHUNK_SIZE * float(PREVIEW_VIEW_CHUNKS * 2 + 1)
	var shaft_len := footprint * 0.42
	var shaft_r := footprint * 0.015
	var head_len := shaft_len * 0.28
	var head_r := shaft_r * 2.8

	var mat := StandardMaterial3D.new()
	mat.shading_mode = BaseMaterial3D.SHADING_MODE_UNSHADED
	mat.albedo_color = Color(1.0, 0.72, 0.18)
	mat.emission_enabled = true
	mat.emission = Color(1.0, 0.72, 0.18)
	mat.emission_energy_multiplier = 1.6

	var shaft := MeshInstance3D.new()
	shaft.name = "Shaft"
	var sm := CylinderMesh.new()
	sm.top_radius = shaft_r
	sm.bottom_radius = shaft_r * 0.8
	sm.height = shaft_len
	sm.radial_segments = 10
	shaft.mesh = sm
	shaft.material_override = mat
	shaft.rotation_degrees.z = 90.0
	shaft.position.x = shaft_len * 0.5
	root.add_child(shaft)

	var head := MeshInstance3D.new()
	head.name = "Head"
	var hm := CylinderMesh.new()
	hm.top_radius = 0.0
	hm.bottom_radius = head_r
	hm.height = head_len
	hm.radial_segments = 10
	head.mesh = hm
	head.material_override = mat
	head.rotation_degrees.z = 90.0
	head.position.x = shaft_len + head_len * 0.5
	root.add_child(head)

	return root

## Push the current setting values into the live preview: regen the noise/height
## keys, steer the wind arrow and refresh the overlay readout.
func _apply_terrain_preview_settings() -> void:
	if _terrain_preview_gen == null:
		return
	for key: String in PREVIEW_GEN_KEYS:
		_terrain_preview_gen.set(key, _terrain_settings[key])
	_update_terrain_wind_arrow()

func _update_terrain_wind_arrow() -> void:
	if _terrain_preview_arrow != null:
		var dir := float(_terrain_settings["wind_direction"])
		_terrain_preview_arrow.rotation.y = -deg_to_rad(dir)
		if _terrain_preview_gen != null:
			var peak := _sample_preview_peak()
			_terrain_preview_arrow.position.y = peak + PREVIEW_CHUNK_SIZE * 0.08
	if _terrain_wind_label != null:
		var dir := float(_terrain_settings["wind_direction"])
		var speed := float(_terrain_settings["wind_speed"])
		_terrain_wind_label.text = "WIND  %5.1f m/s  %3d°" % [speed, roundi(dir)]

## Highest terrain in the preview footprint — the arrow hovers just above it so
## it stays visible no matter how the relief is sculpted.
func _sample_preview_peak() -> float:
	var half := PREVIEW_CHUNK_SIZE * float(PREVIEW_VIEW_CHUNKS) * 0.5
	var peak := 0.0
	var steps := 10
	for iz in steps + 1:
		for ix in steps + 1:
			var x := -half + 2.0 * half * ix / steps
			var z := -half + 2.0 * half * iz / steps
			peak = maxf(peak, _terrain_preview_gen.sample_height(x, z))
	return peak

## Slow turntable orbit for the preview camera while the Terrain tab is open.
func _update_terrain_preview_camera(delta: float) -> void:
	if _terrain_viewport_cam == null:
		return
	_terrain_preview_yaw += delta * 0.18
	var footprint := PREVIEW_CHUNK_SIZE * float(PREVIEW_VIEW_CHUNKS * 2 + 1)
	var dist := footprint * 1.15
	var pitch := 0.52
	var cp := cos(pitch)
	var sp := sin(pitch)
	var pos := Vector3(dist * cp * cos(_terrain_preview_yaw), dist * sp, dist * cp * sin(_terrain_preview_yaw))
	var peak := _sample_preview_peak()
	_terrain_viewport_cam.global_position = pos + Vector3(0.0, peak * 0.35, 0.0)
	_terrain_viewport_cam.look_at(Vector3(0.0, peak * 0.35, 0.0), Vector3.UP)

func _make_settings_card() -> Array:
	var panel := PanelContainer.new()
	panel.size_flags_horizontal = Control.SIZE_EXPAND_FILL
	var style := StyleBoxFlat.new()
	style.bg_color = BG_CARD
	style.border_width_left = 2
	style.border_width_right = 1
	style.border_width_top = 1
	style.border_width_bottom = 1
	style.border_color = BORDER
	style.corner_radius_top_left = 8
	style.corner_radius_top_right = 8
	style.corner_radius_bottom_left = 8
	style.corner_radius_bottom_right = 8
	style.content_margin_left = 24
	style.content_margin_right = 24
	style.content_margin_top = 20
	style.content_margin_bottom = 20
	panel.add_theme_stylebox_override("panel", style)
	var inner := VBoxContainer.new()
	inner.add_theme_constant_override("separation", 10)
	inner.size_flags_horizontal = Control.SIZE_EXPAND_FILL
	panel.add_child(inner)
	return [panel, inner]

func _make_settings_heading(text: String) -> Label:
	var heading := Label.new()
	heading.text = text
	heading.add_theme_font_size_override("font_size", 20)
	heading.add_theme_color_override("font_color", ACCENT)
	return heading

func _make_settings_label(text: String) -> Label:
	var label := Label.new()
	label.text = text
	label.custom_minimum_size.x = 180
	label.add_theme_font_size_override("font_size", 14)
	label.add_theme_color_override("font_color", TEXT_MID)
	label.size_flags_vertical = Control.SIZE_SHRINK_CENTER
	return label

func _add_slider_setting(parent: VBoxContainer, label_text: String, minimum: float, maximum: float, step: float, value: float, format: String, key: String, is_procedural: bool = true) -> HSlider:
	var row := HBoxContainer.new()
	row.add_theme_constant_override("separation", 12)
	parent.add_child(row)
	row.add_child(_make_settings_label(label_text))
	var slider := HSlider.new()
	slider.min_value = minimum
	slider.max_value = maximum
	slider.step = step
	slider.custom_minimum_size.x = 320
	slider.size_flags_horizontal = Control.SIZE_EXPAND_FILL
	slider.set_value_no_signal(clampf(value, minimum, maximum))
	row.add_child(slider)
	var readout := Label.new()
	readout.custom_minimum_size.x = 110
	readout.add_theme_font_size_override("font_size", 14)
	readout.add_theme_color_override("font_color", TEXT_SPEC)
	readout.size_flags_vertical = Control.SIZE_SHRINK_CENTER
	row.add_child(readout)
	readout.text = format % _terrain_settings[key]
	slider.value_changed.connect(func(new_value: float) -> void:
		_set_terrain_setting(key, new_value)
		readout.text = format % _terrain_settings[key]
	)
	if is_procedural:
		_terrain_procedural_controls.append(slider)
	return slider

func _make_resolution_row() -> HBoxContainer:
	var row := HBoxContainer.new()
	row.add_theme_constant_override("separation", 12)
	row.add_child(_make_settings_label("Chunk resolution"))
	var choices := OptionButton.new()
	for size in TERRAIN_RESOLUTIONS:
		choices.add_item("%d vertices" % size, size)
	choices.selected = TERRAIN_RESOLUTIONS.find(int(_terrain_settings["resolution"]))
	choices.custom_minimum_size.x = 260
	row.add_child(choices)
	choices.item_selected.connect(func(index: int) -> void:
		_set_terrain_setting("resolution", TERRAIN_RESOLUTIONS[index])
	)
	_terrain_procedural_controls.append(choices)
	return row

func _make_check_setting(label_text: String, key: String) -> CheckButton:
	var check := CheckButton.new()
	check.text = label_text
	check.add_theme_font_size_override("font_size", 14)
	check.set_pressed_no_signal(bool(_terrain_settings[key]))
	check.toggled.connect(func(pressed: bool) -> void:
		_set_terrain_setting(key, pressed)
	)
	_terrain_procedural_controls.append(check)
	return check

func _set_procedural_controls_enabled(enabled: bool) -> void:
	for control in _terrain_procedural_controls:
		if control is Range:
			(control as Range).editable = enabled
		elif control is BaseButton:
			(control as BaseButton).disabled = not enabled

func _randomize_terrain_seed() -> void:
	var rng := RandomNumberGenerator.new()
	rng.randomize()
	_set_terrain_setting("noise_seed", rng.randi_range(0, 100000))
	_refresh_terrain_page()

func _reset_terrain_settings() -> void:
	_terrain_settings = _load_terrain_defaults()
	_save_terrain_settings()
	_refresh_terrain_page()

func _refresh_terrain_page() -> void:
	var page: Control = _pages["Terrain"]
	var parent := page.get_parent()
	var position := parent.get_children().find(page)
	_terrain_procedural_controls.clear()
	_terrain_summary = null
	_terrain_viewport = null
	_terrain_viewport_cam = null
	_terrain_preview_gen = null
	_terrain_preview_arrow = null
	_terrain_wind_label = null
	parent.remove_child(page)
	page.queue_free()
	page = _build_terrain_page()
	parent.add_child(page)
	parent.move_child(page, position)
	_pages["Terrain"] = page
	page.visible = (_current_page == "Terrain")
	if _terrain_viewport != null:
		_terrain_viewport.render_target_update_mode = (
			SubViewport.UPDATE_ALWAYS if _current_page == "Terrain"
			else SubViewport.UPDATE_DISABLED)

func _launch_flight_with_terrain() -> void:
	_save_terrain_settings()
	get_tree().change_scene_to_file("res://scenes/flight_sim.tscn")

func _load_terrain_settings() -> Dictionary:
	var settings := _load_terrain_defaults()
	var root := get_tree().root
	if root != null and root.has_meta(TERRAIN_SETTINGS_META):
		var saved = root.get_meta(TERRAIN_SETTINGS_META)
		if saved is Dictionary:
			for key in TERRAIN_SETTING_KEYS:
				if saved.has(key):
					settings[key] = _sanitize_terrain_setting(key, saved[key])
	return settings

func _load_terrain_defaults() -> Dictionary:
	var settings := {}
	for key in TERRAIN_SETTING_KEYS:
		settings[key] = _sanitize_terrain_setting(key, TERRAIN_DEFAULTS[key])
	return settings

func _save_terrain_settings() -> void:
	var root := get_tree().root
	if root == null:
		return
	root.set_meta(TERRAIN_SETTINGS_META, _terrain_settings.duplicate())

func _set_terrain_setting(key: String, value: Variant) -> void:
	_terrain_settings[key] = _sanitize_terrain_setting(key, value)
	_save_terrain_settings()
	_update_terrain_summary()
	_apply_terrain_preview_settings()

func _sanitize_terrain_setting(key: String, value: Variant) -> Variant:
	match key:
		"procedural_terrain", "collision_enabled", "physics_grid_enabled":
			return _terrain_bool(value, bool(TERRAIN_DEFAULTS[key]))
		"noise_seed":
			return clampi(_terrain_int(value, int(TERRAIN_DEFAULTS[key])), 0, 100000)
		"noise_scale":
			return clampf(_terrain_float(value, float(TERRAIN_DEFAULTS[key])), 40.0, 2000.0)
		"octaves":
			return clampi(_terrain_int(value, int(TERRAIN_DEFAULTS[key])), 1, 6)
		"persistence":
			return clampf(_terrain_float(value, float(TERRAIN_DEFAULTS[key])), 0.1, 0.9)
		"lacunarity":
			return clampf(_terrain_float(value, float(TERRAIN_DEFAULTS[key])), 1.5, 3.0)
		"height_multiplier":
			return clampf(_terrain_float(value, float(TERRAIN_DEFAULTS[key])), 100.0, 1800.0)
		"height_exponent":
			return clampf(_terrain_float(value, float(TERRAIN_DEFAULTS[key])), 0.6, 1.6)
		"view_chunks":
			return clampi(_terrain_int(value, int(TERRAIN_DEFAULTS[key])), 3, 12)
		"resolution":
			return _nearest_terrain_resolution(_terrain_int(value, int(TERRAIN_DEFAULTS[key])))
		"chunk_size":
			return clampf(_terrain_float(value, float(TERRAIN_DEFAULTS[key])), 100.0, 2000.0)
		"wind_speed":
			return clampf(_terrain_float(value, float(TERRAIN_DEFAULTS[key])), 0.0, 30.0)
		"wind_direction":
			return fmod(_terrain_float(value, float(TERRAIN_DEFAULTS[key])), 360.0)
	return value

func _terrain_bool(value: Variant, fallback: bool) -> bool:
	if value is bool:
		return value
	if value is int:
		return value != 0
	if value is float:
		return value != 0.0
	if value is String:
		return value.to_lower() in ["1", "true", "yes", "on"]
	return fallback

func _terrain_int(value: Variant, fallback: int) -> int:
	if value is bool:
		return 1 if value else 0
	if value is int:
		return value
	if value is float:
		return int(round(value))
	if value is String and value.is_valid_int():
		return int(value)
	return fallback

func _terrain_float(value: Variant, fallback: float) -> float:
	if value is bool:
		return 1.0 if value else 0.0
	if value is int:
		return float(value)
	if value is float:
		return value
	if value is String and value.is_valid_float():
		return float(value)
	return fallback

func _nearest_terrain_resolution(candidate: int) -> int:
	var best: int = TERRAIN_RESOLUTIONS[0]
	for size in TERRAIN_RESOLUTIONS:
		if abs(size - candidate) < abs(best - candidate):
			best = size
	return best

func _update_terrain_summary() -> void:
	if _terrain_summary == null:
		return
	if not bool(_terrain_settings["procedural_terrain"]):
		_terrain_summary.text = "Imported landscape selected. Launching will use the packaged snowy-mountain terrain instead of generated land."
		return
	var seed: int = int(_terrain_settings["noise_seed"])
	var height: int = int(round(float(_terrain_settings["height_multiplier"])))
	var noise_scale: int = int(round(float(_terrain_settings["noise_scale"])))
	var octaves: int = int(_terrain_settings["octaves"])
	var resolution: int = int(_terrain_settings["resolution"])
	var chunks: int = int(_terrain_settings["view_chunks"])
	_terrain_summary.text = "Ready to generate: seed %d, %d m mountains, scale %d, %d octaves, %d-vertex chunks across %d chunks." % [seed, height, noise_scale, octaves, resolution, chunks]

# ── ABOUT PAGE ───────────────────────────────────────────────────────
func _build_about_page() -> Control:
	var scroll := ScrollContainer.new()
	scroll.size_flags_horizontal = Control.SIZE_EXPAND_FILL
	scroll.size_flags_vertical = Control.SIZE_EXPAND_FILL
	scroll.horizontal_scroll_mode = ScrollContainer.SCROLL_MODE_DISABLED

	var vbox := VBoxContainer.new()
	vbox.add_theme_constant_override("separation", 8)
	vbox.size_flags_horizontal = Control.SIZE_EXPAND_FILL
	scroll.add_child(vbox)

	# ── Section: Architecture ──────────────────────────────────────
	var arch_heading := Label.new()
	arch_heading.text = "Architecture"
	arch_heading.add_theme_font_size_override("font_size", 32)
	arch_heading.add_theme_color_override("font_color", TEXT_LIGHT)
	vbox.add_child(arch_heading)

	var arch_desc := RichTextLabel.new()
	arch_desc.bbcode_enabled = true
	arch_desc.text = (
		"[color=#5b8cbf]flight_core[/color]  — Pure-Rust physics engine with quaternion 6-DOF rigid-body dynamics,\n" +
		"RK4 integrator at 60 Hz, 1976 US Standard Atmosphere, nonlinear post-stall aero,\n" +
		"wind (steady / shear / Dryden turbulence), terrain collision, trimming, and autopilot.\n\n" +
		"[color=#5b8cbf]flight_gd[/color]  — godot-rust GDExtension exposing FlightSimNode and WindTunnelNode\n" +
		"as native Godot nodes. Rust computes all physics; Godot renders.\n\n" +
		"[color=#5b8cbf]godot/[/color]  — Godot 4.7 project with scenes, procedural models, chase/orbit camera,\n" +
		"HUD, sky, terrain, runway, and the MultiMesh smoke renderer."
	)
	arch_desc.add_theme_font_size_override("normal_font_size", 14)
	arch_desc.add_theme_color_override("default_color", TEXT_MID)
	arch_desc.custom_minimum_size.y = 140
	arch_desc.fit_content = true
	vbox.add_child(arch_desc)

	vbox.add_child(HSeparator.new())

	# ── Section: Avionics Module ───────────────────────────────────
	var av_heading := Label.new()
	av_heading.text = "Avionics Module"
	av_heading.add_theme_font_size_override("font_size", 32)
	av_heading.add_theme_color_override("font_color", TEXT_LIGHT)
	vbox.add_child(av_heading)

	var av_sub := Label.new()
	av_sub.text = "Simulated hardware-in-the-loop UAV avionics stack for reinforcement learning. Each component is feature-gated and communicates through a shared data bus."
	av_sub.autowrap_mode = TextServer.AUTOWRAP_WORD_SMART
	av_sub.add_theme_font_size_override("font_size", 14)
	av_sub.add_theme_color_override("font_color", TEXT_MID)
	av_sub.custom_minimum_size.y = 36
	vbox.add_child(av_sub)

	# Avionics grid — 2 columns
	var grid := GridContainer.new()
	grid.columns = 2
	grid.add_theme_constant_override("h_separation", 20)
	grid.add_theme_constant_override("v_separation", 12)
	grid.size_flags_horizontal = Control.SIZE_EXPAND_FILL
	vbox.add_child(grid)

	for mod_name in AVIONICS:
		grid.add_child(_make_avionics_tile(mod_name, AVIONICS[mod_name]))

	vbox.add_child(HSeparator.new())

	# ── Section: Controls Reference ────────────────────────────────
	var ctrl_heading := Label.new()
	ctrl_heading.text = "Controls Reference"
	ctrl_heading.add_theme_font_size_override("font_size", 32)
	ctrl_heading.add_theme_color_override("font_color", TEXT_LIGHT)
	vbox.add_child(ctrl_heading)

	# Two-column layout: Flight Sim | Wind Tunnel
	var ctrl_row := HBoxContainer.new()
	ctrl_row.add_theme_constant_override("separation", 32)
	ctrl_row.size_flags_horizontal = Control.SIZE_EXPAND_FILL
	vbox.add_child(ctrl_row)

	ctrl_row.add_child(_make_controls_card("Flight Simulator", {
		"Pitch": "W / S  (or ↑ / ↓)",
		"Roll": "A / D  (or ← / →)",
		"Rudder": "Q / E  (or Z / C)",
		"Flaps": "F  (0° → 15° → 30°)",
		"Trim": "[ / ]  (nose down / up)",
		"Throttle": "Shift / Ctrl",
		"Autopilot": "H or T  (level hold)",
		"Terrain": "N regenerate / B source",
		"Reset": "R",
		"Menu": "Esc",
	}))

	ctrl_row.add_child(_make_controls_card("Wind Tunnel", {
		"Orbit camera": "Click + drag",
		"Zoom": "Scroll wheel",
		"Pitch": "W / S",
		"Yaw": "A / D",
		"Roll": "↑ / ↓",
		"Aileron": "Q / E",
		"Rudder": "Z / C",
		"Flaps": "F",
		"Wind speed": "Shift / Ctrl",
		"Wind dir": "R / T",
		"Reset": "Space",
		"Menu": "Esc",
	}))

	vbox.add_child(HSeparator.new())

	# ── Section: Tech Stack ────────────────────────────────────────
	var tech_heading := Label.new()
	tech_heading.text = "Tech Stack"
	tech_heading.add_theme_font_size_override("font_size", 24)
	tech_heading.add_theme_color_override("font_color", TEXT_LIGHT)
	vbox.add_child(tech_heading)

	var tech_row := HBoxContainer.new()
	tech_row.add_theme_constant_override("separation", 16)
	vbox.add_child(tech_row)

	for item in [
		"Rust edition 2021",
		"nalgebra 0.33",
		"godot-rust 0.5 (API 4.7)",
		"Godot 4.7 — Forward+ renderer",
		"TOML aircraft configs",
	]:
		var chip := Label.new()
		chip.text = item
		chip.add_theme_font_size_override("font_size", 13)
		chip.add_theme_color_override("font_color", TEXT_SPEC)
		var chip_style := StyleBoxFlat.new()
		chip_style.bg_color = BG_CARD
		chip_style.border_width_left = 1
		chip_style.border_width_right = 1
		chip_style.border_width_top = 1
		chip_style.border_width_bottom = 1
		chip_style.border_color = BORDER_LIGHT
		chip_style.corner_radius_top_left = 4
		chip_style.corner_radius_top_right = 4
		chip_style.corner_radius_bottom_left = 4
		chip_style.corner_radius_bottom_right = 4
		chip_style.content_margin_left = 10
		chip_style.content_margin_right = 10
		chip_style.content_margin_top = 4
		chip_style.content_margin_bottom = 4
		chip.add_theme_stylebox_override("panel", chip_style)
		var wrap := PanelContainer.new()
		wrap.add_child(chip)
		tech_row.add_child(wrap)

	return scroll

func _make_avionics_tile(mod_name: String, description: String) -> PanelContainer:
	var panel := PanelContainer.new()
	panel.size_flags_horizontal = Control.SIZE_EXPAND_FILL

	var style := StyleBoxFlat.new()
	style.bg_color = BG_CARD
	style.border_width_left = 2
	style.border_width_top = 1
	style.border_width_right = 1
	style.border_width_bottom = 1
	style.border_color = BORDER
	style.corner_radius_top_left = 6
	style.corner_radius_top_right = 6
	style.corner_radius_bottom_left = 6
	style.corner_radius_bottom_right = 6
	style.content_margin_left = 16
	style.content_margin_right = 16
	style.content_margin_top = 14
	style.content_margin_bottom = 14
	panel.add_theme_stylebox_override("panel", style)

	var inner := VBoxContainer.new()
	inner.add_theme_constant_override("separation", 6)
	panel.add_child(inner)

	var lbl := Label.new()
	lbl.text = mod_name
	lbl.add_theme_font_size_override("font_size", 17)
	lbl.add_theme_color_override("font_color", ACCENT)
	inner.add_child(lbl)

	var desc := Label.new()
	desc.text = description
	desc.autowrap_mode = TextServer.AUTOWRAP_WORD_SMART
	desc.add_theme_font_size_override("font_size", 12)
	desc.add_theme_color_override("font_color", TEXT_DIM)
	inner.add_child(desc)

	# Hover effect
	panel.mouse_entered.connect(func() -> void:
		style.bg_color = BG_CARD_HVR
		style.border_color = ACCENT_DIM)
	panel.mouse_exited.connect(func() -> void:
		style.bg_color = BG_CARD
		style.border_color = BORDER)

	return panel

func _make_controls_card(section_title: String, controls: Dictionary) -> PanelContainer:
	var panel := PanelContainer.new()
	panel.size_flags_horizontal = Control.SIZE_EXPAND_FILL
	panel.custom_minimum_size.x = 300

	var style := StyleBoxFlat.new()
	style.bg_color = BG_CARD
	style.border_width_left = 2
	style.border_width_right = 1
	style.border_width_top = 1
	style.border_width_bottom = 1
	style.border_color = BORDER
	style.corner_radius_top_left = 8
	style.corner_radius_top_right = 8
	style.corner_radius_bottom_left = 8
	style.corner_radius_bottom_right = 8
	style.content_margin_left = 24
	style.content_margin_right = 24
	style.content_margin_top = 20
	style.content_margin_bottom = 20
	panel.add_theme_stylebox_override("panel", style)

	var inner := VBoxContainer.new()
	inner.add_theme_constant_override("separation", 6)
	panel.add_child(inner)

	var heading := Label.new()
	heading.text = section_title
	heading.add_theme_font_size_override("font_size", 20)
	heading.add_theme_color_override("font_color", ACCENT)
	inner.add_child(heading)

	inner.add_child(HSeparator.new())

	for action in controls:
		var row := HBoxContainer.new()
		row.add_theme_constant_override("separation", 12)
		inner.add_child(row)

		var albl := Label.new()
		albl.text = action
		albl.add_theme_font_size_override("font_size", 13)
		albl.add_theme_color_override("font_color", TEXT_MID)
		albl.custom_minimum_size.x = 120
		row.add_child(albl)

		var klbl := Label.new()
		klbl.text = controls[action]
		klbl.add_theme_font_size_override("font_size", 13)
		klbl.add_theme_color_override("font_color", TEXT_SPEC)
		row.add_child(klbl)

	return panel

# ── Shared helpers ───────────────────────────────────────────────────
func _make_action_button(label: String, min_w: int, min_h: int) -> Button:
	var btn := Button.new()
	btn.text = label
	btn.custom_minimum_size = Vector2(min_w, min_h)
	btn.add_theme_font_size_override("font_size", 17)

	var normal := StyleBoxFlat.new()
	normal.bg_color = ACCENT_DIM
	normal.corner_radius_top_left = 6
	normal.corner_radius_top_right = 6
	normal.corner_radius_bottom_left = 6
	normal.corner_radius_bottom_right = 6
	normal.content_margin_left = 18
	normal.content_margin_right = 18
	normal.content_margin_top = 8
	normal.content_margin_bottom = 8
	btn.add_theme_stylebox_override("normal", normal)

	var hover := normal.duplicate()
	hover.bg_color = ACCENT
	btn.add_theme_stylebox_override("hover", hover)

	var pressed := normal.duplicate()
	pressed.bg_color = Color(0.20, 0.50, 0.80)
	btn.add_theme_stylebox_override("pressed", pressed)

	btn.add_theme_color_override("font_color", TEXT_LIGHT)
	btn.add_theme_color_override("font_hover_color", Color.WHITE)
	return btn
