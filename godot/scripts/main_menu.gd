extends Control
## Main menu. Provides a choice between the flight simulator and the
## wind-tunnel visualization, plus a quit button. The entire UI is built
## procedurally so the scene file stays minimal.

func _ready() -> void:
	_build_ui()

func _build_ui() -> void:
	var bg := ColorRect.new()
	bg.color = Color(0.02, 0.03, 0.05)
	bg.set_anchors_preset(Control.PRESET_FULL_RECT)
	add_child(bg)

	var center := CenterContainer.new()
	center.set_anchors_preset(Control.PRESET_FULL_RECT)
	add_child(center)

	var col := VBoxContainer.new()
	col.add_theme_constant_override("separation", 22)
	center.add_child(col)

	var title := Label.new()
	title.text = "PROJECT RAPTOR M86"
	title.horizontal_alignment = HORIZONTAL_ALIGNMENT_CENTER
	title.add_theme_font_size_override("font_size", 56)
	title.add_theme_color_override("font_color", Color(0.75, 0.85, 0.98))
	col.add_child(title)

	var subtitle := Label.new()
	subtitle.text = "Rust 6-DOF flight dynamics visualized in Godot"
	subtitle.horizontal_alignment = HORIZONTAL_ALIGNMENT_CENTER
	subtitle.add_theme_font_size_override("font_size", 20)
	subtitle.add_theme_color_override("font_color", Color(0.45, 0.55, 0.7))
	col.add_child(subtitle)

	col.add_child(HSeparator.new())

	var btn_flight := _make_button("Flight Simulator", "Take off at 1000 m / 60 m/s trimmed cruise")
	var btn_tunnel := _make_button("Wind Tunnel", "Hold the aircraft fixed; visualize the smoke flow field")
	var btn_quit := _make_button("Quit", "")

	col.add_child(btn_flight)
	col.add_child(btn_tunnel)
	col.add_child(btn_quit)

	var hint := Label.new()
	hint.text = "Esc returns to this menu from any mode"
	hint.horizontal_alignment = HORIZONTAL_ALIGNMENT_CENTER
	hint.add_theme_font_size_override("font_size", 14)
	hint.add_theme_color_override("font_color", Color(0.35, 0.42, 0.55))
	col.add_child(hint)

	btn_flight.pressed.connect(func() -> void:
		get_tree().change_scene_to_file("res://scenes/flight_sim.tscn"))
	btn_tunnel.pressed.connect(func() -> void:
		get_tree().change_scene_to_file("res://scenes/wind_tunnel.tscn"))
	btn_quit.pressed.connect(get_tree().quit)

func _make_button(title_text: String, sub_text: String) -> Button:
	var btn := Button.new()
	btn.custom_minimum_size = Vector2(360, 62)
	btn.text = title_text
	btn.add_theme_font_size_override("font_size", 22)
	if sub_text != "":
		btn.tooltip_text = sub_text
	return btn
