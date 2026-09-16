extends Control
## Main menu with tab-based navigation: Home, Aircraft, About.
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

# ── Build ────────────────────────────────────────────────────────────
func _ready() -> void:
	_build_ui()
	_switch_page("Home")

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
	for tab_name in ["Home", "Aircraft", "About"]:
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
		"Take off at 1 000 m / 60 m/s trimmed cruise. Chase camera, full\ncontrol surface authority, flaps, trim, autopilot hold, wind.",
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

	# ── Left column: name + role + description ──
	var left := VBoxContainer.new()
	left.size_flags_horizontal = Control.SIZE_EXPAND_FILL
	left.custom_minimum_size.x = 360
	left.add_theme_constant_override("separation", 8)
	outer_h.add_child(left)

	var name_row := HBoxContainer.new()
	name_row.add_theme_constant_override("separation", 12)
	left.add_child(name_row)

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
	left.add_child(desc)

	# ── Right column: specs table ──
	var right := GridContainer.new()
	right.columns = 2
	right.add_theme_constant_override("h_separation", 12)
	right.add_theme_constant_override("v_separation", 6)
	right.custom_minimum_size.x = 320
	outer_h.add_child(right)

	_add_spec_row(right, "Mass", "%.0f kg" % data["mass"])
	_add_spec_row(right, "Wingspan", "%.1f m" % data["span"])
	_add_spec_row(right, "Wing area", "%.1f m²" % data["area"])
	_add_spec_row(right, "MAC chord", "%.2f m" % data["chord"])
	_add_spec_row(right, "Propulsion", data["propulsion"])
	_add_spec_row(right, "Max thrust", "%.0f N" % data["thrust"])
	if data["power"] > 0:
		_add_spec_row(right, "Max power", "%.0f kW" % (data["power"] / 1000.0))
	if data.has("mach_crit"):
		_add_spec_row(right, "Mach crit", "%.2f" % data["mach_crit"])

	right.add_child(_make_spacer_label())  # row gap

	_add_spec_row(right, "CL₀", "%.2f" % data["cl0"])
	_add_spec_row(right, "CLα", "%.1f /rad" % data["cla"])
	_add_spec_row(right, "CD₀", "%.4f" % data["cd0"])
	_add_spec_row(right, "K (induced)", "%.4f" % data["k"])

	return panel

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
