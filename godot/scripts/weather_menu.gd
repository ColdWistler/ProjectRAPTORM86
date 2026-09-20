extends PanelContainer
## In-flight "Weather & Features" control panel.
##
## Opened with [F1] (or the "Weather & Features" button). Lets the pilot toggle
## the RL weather layer and common simulation features live, and shows a live
## weather readout. Created by `flight_sim.gd`, which calls `setup(weather, sim)`
## once; `flight_sim.gd` then calls `refresh()` every physics frame while the
## panel is visible so keyboard toggles and the readout stay in sync.
##
## All weather state lives in the `WeatherSystem` GDExtension node; this panel
## is only a control surface for it (preset / severity / seed / determinism /
## scene re-roll cadence / evasive maneuver).

## Preset names in the exact order the `WeatherSystem.configure()` Rust bridge
## recognises, alongside the human-readable labels shown in the dropdown.
const PRESETS := ["clear", "light_turb", "rain", "storm", "extreme_storm"]
const PRESET_LABELS := ["Clear", "Light turb", "Rain", "Storm", "Extreme"]
const SEVERITY_LABELS := ["Clear", "Light", "Moderate", "Severe"]
const TURB_LABELS := ["Light", "Moderate", "Severe", "Extreme"]
const PRECIP_LABELS := ["none", "rain", "snow", "hail"]
const AIRCRAFT_LABELS := ["TwinEngine", "MQI", "Engine"]
const FLAP_VALUES := [0.0, 15.0, 30.0]
const ENGINE_LABELS := ["Both", "Left out", "Right out"]

## Weather system node (the `WeatherSystem` child of the flight scene).
var weather: Node = null
## The FlightSim script driving the sim.
var sim: Node = null

## Current weather recipe selections (persisted across panel opens).
var seed_value := 12345
var deterministic := false
var auto_change := 0.0

# UI handles (typed so the calls below are static-checked).
var _cb_enabled: CheckButton = null
var _cb_avionics: CheckButton = null
var _cb_auto_level: CheckButton = null
var _cb_terrain: CheckButton = null
var _cb_deterministic: CheckButton = null
var _opt_preset: OptionButton = null
var _opt_severity: OptionButton = null
var _opt_aircraft: OptionButton = null
var _opt_flaps: OptionButton = null
var _opt_engine: OptionButton = null
var _spin_seed: SpinBox = null
var _slider_auto: HSlider = null
var _lbl_auto_value: Label = null
var _readout: Label = null

## Attach the weather/physics targets and build the panel once.
func setup(p_weather: Node, p_sim: Node) -> void:
	weather = p_weather
	sim = p_sim
	_build_ui()
	visible = false
	var vp := get_viewport().get_visible_rect().size
	var msize := get_combined_minimum_size()
	position = Vector2(16.0, maxf(16.0, vp.y - msize.y - 16.0))

func _build_ui() -> void:
	var style := StyleBoxFlat.new()
	style.bg_color = Color(0.03, 0.06, 0.10, 0.94)
	style.border_color = Color(0.2, 0.5, 0.7, 1.0)
	style.set_border_width_all(1)
	style.set_corner_radius_all(8)
	style.set_content_margin_all(10)
	add_theme_stylebox_override("panel", style)
	custom_minimum_size.x = 338

	var vbox := VBoxContainer.new()
	vbox.add_theme_constant_override("separation", 5)
	add_child(vbox)

	_section_header(vbox, "WEATHER & FEATURES  [F1]")
	_section_header(vbox, "WEATHER")

	_cb_enabled = CheckButton.new()
	_cb_enabled.text = "Weather enabled (fly through it)"
	_cb_enabled.toggled.connect(_on_enabled_toggled)
	vbox.add_child(_cb_enabled)

	var row_preset := _row(vbox, "Preset")
	_opt_preset = OptionButton.new()
	for label in PRESET_LABELS:
		_opt_preset.add_item(label)
	_opt_preset.item_selected.connect(_on_weather_configure)
	_opt_preset.size_flags_horizontal = Control.SIZE_EXPAND_FILL
	row_preset.add_child(_opt_preset)

	var row_sev := _row(vbox, "Severity")
	_opt_severity = OptionButton.new()
	for label in SEVERITY_LABELS:
		_opt_severity.add_item(label)
	_opt_severity.item_selected.connect(_on_severity_selected)
	_opt_severity.size_flags_horizontal = Control.SIZE_EXPAND_FILL
	row_sev.add_child(_opt_severity)

	var row_seed := _row(vbox, "Seed")
	_spin_seed = SpinBox.new()
	_spin_seed.min_value = 0
	_spin_seed.max_value = 1000000
	_spin_seed.step = 1
	_spin_seed.value = seed_value
	_spin_seed.value_changed.connect(_on_seed_changed)
	_spin_seed.size_flags_horizontal = Control.SIZE_EXPAND_FILL
	row_seed.add_child(_spin_seed)

	_cb_deterministic = CheckButton.new()
	_cb_deterministic.text = "Deterministic (fixed pattern)"
	_cb_deterministic.toggled.connect(_on_deterministic_toggled)
	vbox.add_child(_cb_deterministic)

	var row_auto := _row(vbox, "Scene re-roll (s)")
	_slider_auto = HSlider.new()
	_slider_auto.min_value = 0
	_slider_auto.max_value = 120
	_slider_auto.step = 5
	_slider_auto.value = auto_change
	_slider_auto.size_flags_horizontal = Control.SIZE_EXPAND_FILL
	_slider_auto.value_changed.connect(_on_auto_changed)
	row_auto.add_child(_slider_auto)
	_lbl_auto_value = Label.new()
	_lbl_auto_value.custom_minimum_size.x = 34
	_lbl_auto_value.text = "%0.0f s" % auto_change
	row_auto.add_child(_lbl_auto_value)

	var evasive := Button.new()
	evasive.text = "Evade gusts (evasive maneuver)"
	evasive.pressed.connect(_on_evasive_pressed)
	vbox.add_child(evasive)

	_section_header(vbox, "FLIGHT")

	_cb_avionics = CheckButton.new()
	_cb_avionics.text = "Avionics path (attitude) [L]"
	_cb_avionics.toggled.connect(_on_avionics_toggled)
	vbox.add_child(_cb_avionics)

	_cb_auto_level = CheckButton.new()
	_cb_auto_level.text = "Auto-level autopilot [H]"
	_cb_auto_level.toggled.connect(_on_auto_level_toggled)
	vbox.add_child(_cb_auto_level)

	var row_ac := _row(vbox, "Aircraft")
	_opt_aircraft = OptionButton.new()
	for label in AIRCRAFT_LABELS:
		_opt_aircraft.add_item(label)
	_opt_aircraft.item_selected.connect(_on_aircraft_selected)
	_opt_aircraft.size_flags_horizontal = Control.SIZE_EXPAND_FILL
	row_ac.add_child(_opt_aircraft)

	var row_flaps := _row(vbox, "Flaps")
	_opt_flaps = OptionButton.new()
	_opt_flaps.add_item("0 deg")
	_opt_flaps.add_item("15 deg")
	_opt_flaps.add_item("30 deg")
	_opt_flaps.item_selected.connect(_on_flaps_selected)
	_opt_flaps.size_flags_horizontal = Control.SIZE_EXPAND_FILL
	row_flaps.add_child(_opt_flaps)

	var row_eng := _row(vbox, "Engine")
	_opt_engine = OptionButton.new()
	for label in ENGINE_LABELS:
		_opt_engine.add_item(label)
	_opt_engine.item_selected.connect(_on_engine_selected)
	_opt_engine.size_flags_horizontal = Control.SIZE_EXPAND_FILL
	row_eng.add_child(_opt_engine)

	_cb_terrain = CheckButton.new()
	_cb_terrain.text = "Procedural terrain [B]"
	_cb_terrain.toggled.connect(_on_terrain_toggled)
	vbox.add_child(_cb_terrain)

	var regen := Button.new()
	regen.text = "Regenerate terrain [N]"
	regen.pressed.connect(_on_regenerate_pressed)
	vbox.add_child(regen)

	var reset := Button.new()
	reset.text = "Reset flight [R]"
	reset.pressed.connect(_on_reset_pressed)
	vbox.add_child(reset)

	_section_header(vbox, "LIVE WEATHER")
	_readout = Label.new()
	_readout.add_theme_font_size_override("font_size", 11)
	_readout.text = "weather not started"
	vbox.add_child(_readout)

## Refresh the panel from the live sim/weather state (called by flight_sim.gd
## every physics frame while visible so keyboard toggles stay in sync).
func refresh() -> void:
	if not visible:
		return
	if sim != null:
		_cb_enabled.set_pressed_no_signal(sim.weather_enabled)
		_cb_avionics.set_pressed_no_signal(sim.avionics_mode)
		_cb_auto_level.set_pressed_no_signal(sim.auto_level)
		_cb_terrain.set_pressed_no_signal(sim.procedural_terrain)
		var ac_idx: int = AIRCRAFT_LABELS.find(sim.current_aircraft())
		_opt_aircraft.select(maxi(ac_idx, 0))
		_opt_flaps.select(maxi(FLAP_VALUES.find(sim.flaps_deg), 0))
		_opt_engine.select(clampi(sim.engine_out, 0, 2))
	if weather == null:
		_readout.text = "no WeatherSystem node in scene"
		return
	if not weather.is_ready():
		_readout.text = "WeatherSystem not started"
		return
	_opt_severity.select(int(weather.get_severity()))
	var wind: Vector3 = weather.get_wind_vector()
	var txt := "wind  N %+5.1f  E %+5.1f  U %+5.1f m/s\n" % [wind.x, wind.z, wind.y]
	var turb: int = int(weather.get_turbulence_level())
	txt += "gust rms %4.1f m/s   turb %s\n" % [weather.get_gust_rms(), TURB_LABELS[turb]]
	txt += "rain %3.0f %%   vis %6.0f m\n" % [weather.get_rain_intensity() * 100.0, weather.get_visibility()]
	txt += "shear %5.3f (m/s)/m   updraft %+4.1f m/s\n" % [weather.get_shear_gradient(), weather.get_updraft_strength()]
	var precip: int = int(weather.get_precipitation_type())
	var sev: int = int(weather.get_severity())
	txt += "precip %s   severity %s\n" % [PRECIP_LABELS[precip], SEVERITY_LABELS[sev]]
	txt += "scene %d   sim t %6.1f s\n" % [weather.get_scene_index(), weather.get_sim_time()]
	txt += "weather penalty %+.2f   term %d\n" % [weather.get_reward_penalty_total(), weather.check_termination()]
	_readout.text = txt

# ---------------------------------------------------------------------------
# Handlers
# ---------------------------------------------------------------------------

## Push every weather recipe field into the WeatherSystem node at once
## (preset + seed + determinism + re-roll cadence).
func _apply_weather_recipe() -> void:
	if weather == null or not weather.is_ready():
		return
	var idx: int = _opt_preset.selected
	weather.configure(PRESETS[idx], seed_value, deterministic, auto_change)

func _on_enabled_toggled(on: bool) -> void:
	if sim != null:
		sim.set_weather_enabled(on)
		if on:
			# First power-on applies the currently selected recipe (preset +
			# seed + determinism + re-roll cadence) so the panel is instantly
			# meaningful instead of the plain env-default scene.
			_apply_weather_recipe()

func _on_weather_configure(_index: int) -> void:
	_apply_weather_recipe()

func _on_severity_selected(index: int) -> void:
	if weather != null and weather.is_ready():
		weather.set_severity(index)

func _on_seed_changed(value: float) -> void:
	seed_value = int(value)
	_apply_weather_recipe()

func _on_deterministic_toggled(on: bool) -> void:
	deterministic = on
	_apply_weather_recipe()

func _on_auto_changed(value: float) -> void:
	auto_change = value
	if _lbl_auto_value != null:
		_lbl_auto_value.text = "%0.0f s" % value
	if weather != null and weather.is_ready():
		weather.set_auto_change_interval(value)

func _on_evasive_pressed() -> void:
	if weather != null and weather.is_ready():
		weather.trigger_evasive_maneuver()

func _on_avionics_toggled(on: bool) -> void:
	if sim != null:
		sim.avionics_mode = on

func _on_auto_level_toggled(on: bool) -> void:
	if sim != null:
		sim.set_auto_level(on)

func _on_aircraft_selected(index: int) -> void:
	if sim != null:
		sim.swap_aircraft(AIRCRAFT_LABELS[index])

func _on_flaps_selected(index: int) -> void:
	if sim != null:
		sim.flaps_deg = FLAP_VALUES[index]

func _on_engine_selected(index: int) -> void:
	if sim != null:
		sim.engine_out = index

func _on_terrain_toggled(on: bool) -> void:
	if sim != null:
		sim.set_procedural_terrain(on)

func _on_regenerate_pressed() -> void:
	if sim != null:
		sim.regenerate_terrain()

func _on_reset_pressed() -> void:
	if sim != null:
		sim.reset_flight()

# ---------------------------------------------------------------------------
# Layout helpers
# ---------------------------------------------------------------------------

func _section_header(vbox: VBoxContainer, text: String) -> void:
	var lbl := Label.new()
	lbl.text = text
	lbl.add_theme_font_size_override("font_size", 12)
	lbl.add_theme_color_override("font_color", Color(0.55, 0.85, 1.0))
	lbl.mouse_filter = Control.MOUSE_FILTER_IGNORE
	vbox.add_child(lbl)

func _row(vbox: VBoxContainer, text: String) -> HBoxContainer:
	var row := HBoxContainer.new()
	row.add_theme_constant_override("separation", 6)
	var lbl := Label.new()
	lbl.text = text
	lbl.custom_minimum_size.x = 118
	lbl.add_theme_font_size_override("font_size", 11)
	row.add_child(lbl)
	vbox.add_child(row)
	return row