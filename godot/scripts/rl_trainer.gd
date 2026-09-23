extends Control
## RL trainer control panel — embedded as the "RL" tab of the main menu.
##
## Launches the `rl_agent` Rust binary as a subprocess, streams its stdout
## into a scrolling log, and exposes algorithm / backend / hyper-parameter
## selection. The algorithm list below is the same extension seam as the
## Rust `rl_agent::algo::AlgoSpec` enum: to add a new model, add an entry to
## `ALGORITHMS` + `ALGO_FIELDS` here, implement `Algorithm<B>` in Rust, and
## register it in `algo::make_algorithm`.

const ALGORITHMS := [
	{ "name": "PPO — clipped surrogate", "id": "ppo" },
]

## Backends (display name -> `--backend` value).
const BACKENDS := [
	{ "name": "auto (probe GPU, fall back to CPU)", "id": "auto" },
	{ "name": "cpu (ndarray)", "id": "cpu" },
	{ "name": "gpu (wgpu / Vulkan)", "id": "gpu" },
]

## Per-algorithm hyper-parameters: each entry becomes a `--<key> <value>`
## flag passed to the binary. "int" and "float" map to SpinBoxes.
const ALGO_FIELDS := {
	"ppo": [
		{ "label": "Iterations",       "key": "iterations",    "value": 1000,    "min": 1,     "max": 100000, "step": 1,     "format": "%d",     "kind": "int" },
		{ "label": "Rollout steps",    "key": "rollout",       "value": 2048,    "min": 16,    "max": 100000, "step": 16,    "format": "%d",     "kind": "int" },
		{ "label": "Episode budget",   "key": "env-max-steps", "value": 2000,    "min": 10,    "max": 50000,  "step": 10,    "format": "%d",     "kind": "int" },
		{ "label": "Hidden width",     "key": "hidden",        "value": 128,     "min": 16,    "max": 4096,   "step": 16,    "format": "%d",     "kind": "int" },
		{ "label": "Minibatch",        "key": "minibatch",     "value": 64,      "min": 1,     "max": 8192,   "step": 1,     "format": "%d",     "kind": "int" },
		{ "label": "Epochs",           "key": "epochs",        "value": 10,      "min": 1,     "max": 100,    "step": 1,     "format": "%d",     "kind": "int" },
		{ "label": "Learning rate",    "key": "lr",            "value": 0.0003,  "min": 0.0,   "max": 1.0,    "step": 0.0001, "format": "%.5f", "kind": "float" },
		{ "label": "Gamma",            "key": "gamma",         "value": 0.99,    "min": 0.0,   "max": 1.0,    "step": 0.01,  "format": "%.2f",  "kind": "float" },
		{ "label": "Lambda (GAE)",     "key": "lambda",        "value": 0.95,    "min": 0.0,   "max": 1.0,    "step": 0.01,  "format": "%.2f",  "kind": "float" },
		{ "label": "Clip",             "key": "clip",          "value": 0.2,     "min": 0.01,  "max": 1.0,    "step": 0.01,  "format": "%.2f",  "kind": "float" },
		{ "label": "Seed",             "key": "seed",          "value": 42,      "min": 0,     "max": 999999, "step": 1,     "format": "%d",     "kind": "int" },
		{ "label": "Save every",       "key": "save-every",    "value": 100,     "min": 0,     "max": 100000, "step": 1,     "format": "%d",     "kind": "int" },
		{ "label": "Eval every",       "key": "eval-every",    "value": 100,     "min": 0,     "max": 100000, "step": 1,     "format": "%d",     "kind": "int" },
		{ "label": "Eval episodes",    "key": "eval-episodes", "value": 3,       "min": 0,     "max": 100,    "step": 1,     "format": "%d",     "kind": "int" },
	],
}

const LOG_PATH := "user://rl_training.log"
const DEFAULT_CHECKPOINT := "res://../checkpoints"
const DEFAULT_CONFIG := "res://../aircraft.toml"

const ChartScript := preload("res://scripts/rl_chart.gd")
# Runs directory (repo root's `runs/`) — always access via
# ProjectSettings.globalize_path(RUNS_DIR): res:// paths clamp ".." to the
# project root, so raw `res://../runs` access silently lists the project dir.
const RUNS_DIR := "res://../runs"

const BG_CARD := Color(0.09, 0.11, 0.17)
const BORDER := Color(0.15, 0.19, 0.28)
const BORDER_LIGHT := Color(0.20, 0.26, 0.36)
const ACCENT := Color(0.30, 0.65, 0.95)
const TEXT_LIGHT := Color(0.88, 0.92, 0.97)
const TEXT_MID := Color(0.55, 0.63, 0.75)
const TEXT_DIM := Color(0.38, 0.46, 0.58)
const OK_COLOR := Color(0.45, 0.85, 0.50)
const ERR_COLOR := Color(0.95, 0.45, 0.40)

var _algo_option: OptionButton
var _backend_option: OptionButton
var _fields_container: GridContainer
var _checkpoint_edit: LineEdit
var _run_name_edit: LineEdit
var _status: Label
var _log_view: RichTextLabel
var _binary_label: Label
var _start_btn: BaseButton
var _stop_btn: BaseButton
var _eval_btn: BaseButton
var _dash_btn: BaseButton
var _dash_auto: CheckButton

var _training_pid := 0
var _dash_pid := 0
var _log_offset := 0

# ── Live telemetry chart state ──────────────────────────────────────
var _chart: Control
var _metric_option: OptionButton
var _points := {}          # metric key -> Array of { "x": iteration, "y": value }
var _telemetry_path := ""  # absolute path of the active run's metrics.jsonl
var _telemetry_run_id := ""  # run dir currently feeding the chart ("" = none)
var _telemetry_offset := 0


func _ready() -> void:
	_build_ui()
	_update_status("Ready. Select an algorithm and backend, then Start training.", TEXT_LIGHT)


func _process(_delta: float) -> void:
	if _training_pid != 0:
		_poll_log()
		if not OS.is_process_running(_training_pid):
			var code := OS.get_process_exit_code(_training_pid)
			_training_pid = 0
			_set_buttons_running(false)
			var ok := code == 0
			_update_status(
				("Training finished successfully (exit %d).\nLatest weights + configs in: %s" % [code, ProjectSettings.globalize_path(DEFAULT_CHECKPOINT)])
				if ok else
				("Training process failed (exit %d). Check the log below." % code),
				OK_COLOR if ok else ERR_COLOR)
			_poll_log(true)
	# Always poll telemetry — even while idle — so the chart keeps showing the
	# newest run that has data (and switches to a new live run the moment it
	# writes its first record) instead of freezing on a dead/empty run.
	_poll_telemetry()


# ── UI construction ───────────────────────────────────────────────────
func _build_ui() -> void:
	var scroll := ScrollContainer.new()
	scroll.set_anchors_and_offsets_preset(Control.PRESET_FULL_RECT)
	scroll.size_flags_horizontal = Control.SIZE_EXPAND_FILL
	scroll.size_flags_vertical = Control.SIZE_EXPAND_FILL
	scroll.horizontal_scroll_mode = ScrollContainer.SCROLL_MODE_DISABLED
	add_child(scroll)

	var vbox := VBoxContainer.new()
	vbox.add_theme_constant_override("separation", 16)
	vbox.size_flags_horizontal = Control.SIZE_EXPAND_FILL
	scroll.add_child(vbox)

	var heading := Label.new()
	heading.text = "RL Trainer"
	heading.add_theme_font_size_override("font_size", 32)
	heading.add_theme_color_override("font_color", TEXT_LIGHT)
	vbox.add_child(heading)

	var sub := Label.new()
	sub.text = "Train the burn/PPO agent on the AvionicsEnvironment attitude task. Launches the rl_agent binary as a subprocess; the log below streams its output."
	sub.autowrap_mode = TextServer.AUTOWRAP_WORD_SMART
	sub.add_theme_font_size_override("font_size", 14)
	sub.add_theme_color_override("font_color", TEXT_MID)
	vbox.add_child(sub)

	vbox.add_child(HSeparator.new())

	# ── Algorithm & backend ─────────────────────────────────────────
	var select_card := _make_card()
	vbox.add_child(select_card)
	var select_inner: VBoxContainer = select_card.get_child(0)

	select_inner.add_child(_make_heading("Algorithm & backend"))

	var algo_row := HBoxContainer.new()
	algo_row.add_theme_constant_override("separation", 12)
	select_inner.add_child(algo_row)
	algo_row.add_child(_make_label("Algorithm"))
	_algo_option = OptionButton.new()
	for i in ALGORITHMS.size():
		_algo_option.add_item(str(ALGORITHMS[i]["name"]))
		_algo_option.set_item_metadata(i, ALGORITHMS[i]["id"])
	_algo_option.custom_minimum_size.x = 300
	algo_row.add_child(_algo_option)
	_algo_option.item_selected.connect(_on_algo_changed)

	var note := Label.new()
	note.text = "Add new models in rl_agent/src/algo.rs (AlgoSpec + make_algorithm) and in this list."
	note.autowrap_mode = TextServer.AUTOWRAP_WORD_SMART
	note.add_theme_font_size_override("font_size", 12)
	note.add_theme_color_override("font_color", TEXT_DIM)
	select_inner.add_child(note)

	var backend_row := HBoxContainer.new()
	backend_row.add_theme_constant_override("separation", 12)
	select_inner.add_child(backend_row)
	backend_row.add_child(_make_label("Backend"))
	_backend_option = OptionButton.new()
	for i in BACKENDS.size():
		_backend_option.add_item(str(BACKENDS[i]["name"]))
		_backend_option.set_item_metadata(i, BACKENDS[i]["id"])
	_backend_option.custom_minimum_size.x = 300
	backend_row.add_child(_backend_option)

	# ── Hyper-parameters (algorithm-specific) ───────────────────────
	var fields_card := _make_card()
	vbox.add_child(fields_card)
	var fields_inner: VBoxContainer = fields_card.get_child(0)
	fields_inner.add_child(_make_heading("Hyper-parameters"))
	_fields_container = GridContainer.new()
	_fields_container.columns = 2
	_fields_container.add_theme_constant_override("h_separation", 24)
	_fields_container.add_theme_constant_override("v_separation", 6)
	fields_inner.add_child(_fields_container)
	_rebuild_fields()

	var checkpoint_row := HBoxContainer.new()
	checkpoint_row.add_theme_constant_override("separation", 12)
	fields_inner.add_child(checkpoint_row)
	checkpoint_row.add_child(_make_label("Checkpoint dir"))
	_checkpoint_edit = LineEdit.new()
	_checkpoint_edit.text = ProjectSettings.globalize_path(DEFAULT_CHECKPOINT)
	_checkpoint_edit.custom_minimum_size.x = 460
	checkpoint_row.add_child(_checkpoint_edit)

	# ── Actions + status ────────────────────────────────────────────
	var actions_card := _make_card()
	vbox.add_child(actions_card)
	var actions_inner: VBoxContainer = actions_card.get_child(0)

	actions_inner.add_child(_make_heading("Run"))
	actions_inner.add_child(_make_label("Binary"))
	_binary_label = Label.new()
	_binary_label.autowrap_mode = TextServer.AUTOWRAP_WORD_SMART
	_binary_label.add_theme_font_size_override("font_size", 13)
	actions_inner.add_child(_binary_label)

	var run_name_row := HBoxContainer.new()
	run_name_row.add_theme_constant_override("separation", 12)
	actions_inner.add_child(run_name_row)
	run_name_row.add_child(_make_label("Run name"))
	_run_name_edit = LineEdit.new()
	_run_name_edit.placeholder_text = "optional label for the telemetry run id (e.g. first_attempt)"
	_run_name_edit.custom_minimum_size.x = 460
	run_name_row.add_child(_run_name_edit)

	var buttons := HBoxContainer.new()
	buttons.add_theme_constant_override("separation", 12)
	actions_inner.add_child(buttons)
	_start_btn = _make_button("Start training", 200)
	_start_btn.pressed.connect(_on_start_pressed)
	buttons.add_child(_start_btn)
	_stop_btn = _make_button("Stop", 120)
	_stop_btn.disabled = true
	_stop_btn.pressed.connect(_on_stop_pressed)
	buttons.add_child(_stop_btn)
	_eval_btn = _make_button("Evaluate checkpoint", 220)
	_eval_btn.pressed.connect(_on_eval_pressed)
	buttons.add_child(_eval_btn)
	var clear_btn := _make_button("Clear log", 140)
	clear_btn.pressed.connect(_clear_log)
	buttons.add_child(clear_btn)

	_status = Label.new()
	_status.autowrap_mode = TextServer.AUTOWRAP_WORD_SMART
	_status.add_theme_font_size_override("font_size", 14)
	actions_inner.add_child(_status)

	# ── Live metrics ────────────────────────────────────────────────
	var chart_card := _make_card()
	vbox.add_child(chart_card)
	var chart_inner: VBoxContainer = chart_card.get_child(0)
	chart_inner.add_child(_make_heading("Live metrics"))

	var chart_row := HBoxContainer.new()
	chart_row.add_theme_constant_override("separation", 12)
	chart_inner.add_child(chart_row)
	chart_row.add_child(_make_label("Plot"))
	_metric_option = OptionButton.new()
	for key in ChartScript.METRICS:
		_metric_option.add_item(str(ChartScript.METRICS[key]["label"]))
		_metric_option.set_item_metadata(_metric_option.item_count - 1, key)
	_metric_option.select(0)
	_metric_option.custom_minimum_size.x = 240
	_metric_option.item_selected.connect(_on_metric_changed)
	chart_row.add_child(_metric_option)

	var chart_panel := PanelContainer.new()
	var chart_style := StyleBoxFlat.new()
	chart_style.bg_color = Color(0.02, 0.03, 0.05)
	chart_style.border_width_left = 1
	chart_style.border_width_right = 1
	chart_style.border_width_top = 1
	chart_style.border_width_bottom = 1
	chart_style.border_color = BORDER_LIGHT
	chart_panel.add_theme_stylebox_override("panel", chart_style)
	chart_inner.add_child(chart_panel)

	_chart = ChartScript.new()
	_chart.custom_minimum_size = Vector2(0, 220)
	_chart.size_flags_horizontal = Control.SIZE_EXPAND_FILL
	_chart.size_flags_vertical = Control.SIZE_EXPAND_FILL
	chart_panel.add_child(_chart)

	var hint := Label.new()
	hint.text = "Streams runs/<run_id>/metrics.jsonl while training. For a desktop dashboard that live-follows and compares runs, open the native rl_dash app below."
	hint.autowrap_mode = TextServer.AUTOWRAP_WORD_SMART
	hint.add_theme_font_size_override("font_size", 12)
	hint.add_theme_color_override("font_color", TEXT_DIM)
	chart_inner.add_child(hint)

	var dash_row := HBoxContainer.new()
	dash_row.add_theme_constant_override("separation", 12)
	chart_inner.add_child(dash_row)
	dash_row.add_child(_make_label("Dashboard"))
	_dash_btn = _make_button("Open desktop dashboard (rl_dash)", 320)
	_dash_btn.pressed.connect(_on_dash_pressed)
	dash_row.add_child(_dash_btn)
	_dash_auto = CheckButton.new()
	_dash_auto.text = "Open on Start"
	_dash_auto.tooltip_text = "Launch the native rl_dash desktop dashboard (pointed at the same runs dir) whenever training starts."
	_dash_auto.add_theme_font_size_override("font_size", 13)
	dash_row.add_child(_dash_auto)

	# ── Log ─────────────────────────────────────────────────────────
	var log_card := _make_card()
	vbox.add_child(log_card)
	var log_inner: VBoxContainer = log_card.get_child(0)
	log_inner.add_child(_make_heading("Training log"))

	var log_panel := PanelContainer.new()
	log_panel.custom_minimum_size.y = 260
	var log_style := StyleBoxFlat.new()
	log_style.bg_color = Color(0.02, 0.03, 0.05)
	log_style.border_width_left = 1
	log_style.border_width_right = 1
	log_style.border_width_top = 1
	log_style.border_width_bottom = 1
	log_style.border_color = BORDER_LIGHT
	log_style.content_margin_left = 10
	log_style.content_margin_right = 10
	log_style.content_margin_top = 8
	log_style.content_margin_bottom = 8
	log_panel.add_theme_stylebox_override("panel", log_style)
	log_inner.add_child(log_panel)

	_log_view = RichTextLabel.new()
	_log_view.bbcode_enabled = true
	_log_view.scroll_following = true
	_log_view.fit_content = true
	_log_view.add_theme_font_size_override("normal_font_size", 13)
	log_panel.add_child(_log_view)

	_update_binary_label()


# ── Small UI builders (kept local so this panel is standalone) ───────
func _make_card() -> PanelContainer:
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
	return panel


func _make_heading(text: String) -> Label:
	var lbl := Label.new()
	lbl.text = text
	lbl.add_theme_font_size_override("font_size", 20)
	lbl.add_theme_color_override("font_color", ACCENT)
	return lbl


func _make_label(text: String) -> Label:
	var lbl := Label.new()
	lbl.text = text
	lbl.custom_minimum_size.x = 150
	lbl.add_theme_font_size_override("font_size", 14)
	lbl.add_theme_color_override("font_color", TEXT_MID)
	lbl.size_flags_vertical = Control.SIZE_SHRINK_CENTER
	return lbl


func _make_button(text: String, width: int) -> Button:
	var btn := Button.new()
	btn.text = text
	btn.custom_minimum_size = Vector2(width, 40)
	btn.add_theme_font_size_override("font_size", 15)
	return btn


func _on_algo_changed(_index: int) -> void:
	_rebuild_fields()


func _on_metric_changed(index: int) -> void:
	var key := str(_metric_option.get_item_metadata(index))
	_rebuild_chart(key)


## Repoint the chart at another series: rebuild from the stored points for
## the selected metric (cheap — runs only on dropdown changes).
func _rebuild_chart(key: String) -> void:
	_chart.clear()
	_chart.set_metric(key)
	var series: Array = _points.get(key, [])
	for p in series:
		_chart.push(int(p["x"]), float(p["y"]))
	_chart.update_title()


func _rebuild_fields() -> void:
	for child in _fields_container.get_children():
		_fields_container.remove_child(child)
		child.queue_free()
	var algo := _selected_algo_id()
	for field in ALGO_FIELDS[algo]:
		_fields_container.add_child(_make_label(field["label"]))
		var spin := SpinBox.new()
		spin.min_value = field["min"]
		spin.max_value = field["max"]
		spin.step = field["step"]
		spin.value = field["value"]
		spin.custom_minimum_size.x = 220
		if field["kind"] == "int":
			spin.allow_lesser = false
			spin.allow_greater = false
		_fields_container.add_child(spin)


# ── Process management ───────────────────────────────────────────────
func _on_start_pressed() -> void:
	var args := _build_args(false)
	_run_process(args, "Training started")


func _on_eval_pressed() -> void:
	var args := _build_args(true)
	_run_process(args, "Evaluation started")


func _run_process(args: Array, started_msg: String) -> void:
	if _training_pid != 0:
		_update_status("A run is already in progress.", ERR_COLOR)
		return
	var bin := _find_rl_binary()
	if bin.is_empty():
		_update_status(
			"rl_agent binary not found. Build it from the repo root with:  cargo build -r -p rl_agent",
			ERR_COLOR)
		return
	_clear_log()
	var quoted := []
	for arg in args:
		quoted.append(_shell_quote(str(arg)))
	var log_abs := ProjectSettings.globalize_path(LOG_PATH)
	var cmd := "exec %s %s > %s 2>&1" % [_shell_quote(bin), " ".join(quoted), _shell_quote(log_abs)]
	var pid := OS.create_process("sh", ["-c", cmd])
	if pid == 0:
		_update_status("Failed to launch rl_agent.", ERR_COLOR)
		return
	_training_pid = pid
	_log_offset = 0
	_set_buttons_running(true)
	if _dash_auto.button_pressed and not _launch_dash():
		started_msg += " — couldn't open the desktop dashboard (rl_dash not built?)"
	_update_status("%s (pid %d) — algo %s, backend %s" % [started_msg, pid, _selected_algo_id(), _selected_backend_id()], OK_COLOR)


func _on_stop_pressed() -> void:
	if _training_pid == 0:
		return
	OS.kill(_training_pid)
	_training_pid = 0
	_set_buttons_running(false)
	_update_status("Training stopped by user. Partial checkpoint (if any) remains on disk.", TEXT_MID)


func _on_dash_pressed() -> void:
	_launch_dash()


## Launch the native rl_dash desktop app pointed at the same telemetry root
## the training process writes to (`--runs` = the trainer's run dir, verbatim).
## rl_dash runs independently of Godot, so closing Godot leaves the dashboard up.
func _launch_dash() -> bool:
	if _dash_pid != 0 and OS.is_process_running(_dash_pid):
		_update_status("Desktop dashboard already running (pid %d)." % _dash_pid, TEXT_MID)
		return true
	var bin := _find_dash_binary()
	if bin.is_empty():
		_update_status(
			"rl_dash binary not found. Build it from the repo root with:  cargo build -p rl_dash (or -r for release)",
			ERR_COLOR)
		return false
	var args: PackedStringArray = ["--runs", ProjectSettings.globalize_path(RUNS_DIR)]
	var pid := OS.create_process(bin, args)
	if pid == 0:
		_update_status("Failed to launch rl_dash.", ERR_COLOR)
		return false
	_dash_pid = pid
	_update_status("Desktop dashboard launched (pid %d) — rl_dash is live-following this runs dir." % pid, OK_COLOR)
	return true


func _find_dash_binary() -> String:
	for rel in ["res://../target/release/rl_dash", "res://../target/debug/rl_dash"]:
		var p := ProjectSettings.globalize_path(rel)
		if FileAccess.file_exists(p):
			return p
	return ""


func _set_buttons_running(running: bool) -> void:
	_start_btn.disabled = running
	_stop_btn.disabled = not running
	_eval_btn.disabled = running


func _build_args(eval_only: bool) -> Array:
	var args := ["--backend", _selected_backend_id(), "--algo", _selected_algo_id()]
	args.append("--checkpoint")
	args.append(_checkpoint_edit.text.strip_edges())
	args.append("--config")
	args.append(ProjectSettings.globalize_path(DEFAULT_CONFIG))
	if eval_only:
		args.append("--mode")
		args.append("eval")
		args.append("--eval-episodes")
		args.append(_format_value("int", _field_value("eval-episodes")))
		return args
	args.append("--run-dir")
	args.append(ProjectSettings.globalize_path(RUNS_DIR))
	var run_name := _run_name_edit.text.strip_edges()
	if not run_name.is_empty():
		args.append("--name")
		args.append(run_name)
	for field in ALGO_FIELDS[_selected_algo_id()]:
		args.append("--" + field["key"])
		args.append(_format_value(field["kind"], _field_value(field["key"])))
	return args


## Emit CLI-parseable values: integers without a trailing ".0", floats with
## six decimals (Godot's `%` formatter supports f, not g) so values like the
## default lr of 0.0003 survive round-tripping to the Rust CLI.
func _format_value(kind: String, value: Variant) -> String:
	if kind == "int":
		return str(int(value))
	return "%.6f" % float(value)


func _field_value(key: String) -> Variant:
	var fields: Array = ALGO_FIELDS[_selected_algo_id()]
	var children := _fields_container.get_children()
	for i in fields.size():
		if fields[i]["key"] == key and i * 2 + 1 < children.size():
			return (children[i * 2 + 1] as SpinBox).value
	# Unknown key (e.g. an algo without that field): fall back to the default.
	for field in fields:
		if field["key"] == key:
			return field["value"]
	return 0


func _selected_algo_id() -> String:
	if _algo_option != null and _algo_option.selected >= 0:
		return _algo_option.get_item_metadata(_algo_option.selected)
	return str(ALGORITHMS[0]["id"])


func _selected_backend_id() -> String:
	if _backend_option != null and _backend_option.selected >= 0:
		return _backend_option.get_item_metadata(_backend_option.selected)
	return str(BACKENDS[0]["id"])


func _find_rl_binary() -> String:
	for rel in ["res://../target/release/rl_agent", "res://../target/debug/rl_agent"]:
		var p := ProjectSettings.globalize_path(rel)
		if FileAccess.file_exists(p):
			return p
	return ""


func _shell_quote(s: String) -> String:
	return "'" + s.replace("'", "'\\''") + "'"


# ── Telemetry streaming ─────────────────────────────────────────────
## Poll the active run's metrics.jsonl and feed the live chart. Runs every
## frame (also while idle) so the chart always follows the newest run that
## actually has records.
func _poll_telemetry() -> void:
	_resolve_telemetry_path()
	if _telemetry_path.is_empty() or not FileAccess.file_exists(_telemetry_path):
		return
	var f := FileAccess.open(_telemetry_path, FileAccess.READ)
	if f == null:
		return
	var size := f.get_length()
	if size < _telemetry_offset:
		_telemetry_offset = 0  # file was replaced
	var new_bytes := size - _telemetry_offset
	if new_bytes <= 0:
		f.close()
		return
	f.seek(_telemetry_offset)
	# get_as_text() ignores the cursor and returns the WHOLE file, so read
	# exactly the new bytes instead (re-parsing everything each frame would
	# append duplicate points).
	var chunk := f.get_buffer(new_bytes).get_string_from_utf8()
	f.close()
	_telemetry_offset = size
	if chunk.is_empty():
		return
	var parser := JSON.new()
	for line in chunk.split("\n"):
		if line.strip_edges().is_empty():
			continue
		# The writer appends with a flush, so the tail may catch a partially
		# written last line — skip it silently, it shows up next poll.
		var err := parser.parse(line)
		if err == OK and typeof(parser.data) == TYPE_DICTIONARY:
			_ingest_record(parser.data)


## The chart always reads the newest run dir that has records. A brand-new
## run starts with an empty `metrics.jsonl` (the file is created eagerly, but
## the first record only lands once iteration 0 finishes — a minute or two on
## the GPU/auto backend), and a run can die early leaving the file empty. In
## both cases we keep showing the newest run that has data instead of a blank
## "waiting for telemetry" chart; the moment the new run writes its first
## record we switch over and rebuild the series from it.
func _resolve_telemetry_path() -> void:
	# res:// paths clamp ".." to the project root, so every access to the
	# runs dir must go through the globalized (absolute) path — otherwise
	# the scan silently lists the project's own files and finds no runs.
	var root := ProjectSettings.globalize_path(RUNS_DIR)
	var dir := DirAccess.open(root)
	if dir == null:
		return
	dir.list_dir_begin()
	var newest_any := ""
	var newest_filled := ""
	var entry := dir.get_next()
	while not entry.is_empty():
		if dir.current_is_dir() and entry != "." and entry != "..":
			var p := "%s/%s/metrics.jsonl" % [root, entry]
			if FileAccess.file_exists(p):
				if entry > newest_any:
					newest_any = entry
				if _jsonl_filled(p) and entry > newest_filled:
					newest_filled = entry
		entry = dir.get_next()
	dir.list_dir_end()
	var target := newest_filled if not newest_filled.is_empty() else newest_any
	if target == _telemetry_run_id:
		return
	_telemetry_run_id = target
	_telemetry_path = "" if target.is_empty() else "%s/%s/metrics.jsonl" % [root, target]
	_telemetry_offset = 0
	_points.clear()
	_chart.clear()
	_chart.run_id = target
	_chart.set_metric(_chart.metric_key)
	_chart.update_title()


func _jsonl_filled(p: String) -> bool:
	var f := FileAccess.open(p, FileAccess.READ)
	if f == null:
		return false
	var filled := f.get_length() > 0
	f.close()
	return filled


func _ingest_record(rec: Dictionary) -> void:
	var it := int(rec.get("iteration", -1))
	if it < 0:
		return
	_push_point("mean_reward_per_step", it, rec.get("mean_reward_per_step"))
	_push_point("policy_loss", it, rec.get("policy_loss"))
	_push_point("value_loss", it, rec.get("value_loss"))
	_push_point("entropy", it, rec.get("entropy"))
	_push_point("advantage_mean", it, rec.get("advantage_mean"))
	var ev: Variant = rec.get("eval")
	if typeof(ev) == TYPE_DICTIONARY:
		_push_point("alt_err", it, ev.get("alt_err"))
		_push_point("spd_err", it, ev.get("spd_err"))
		_push_point("eval_reward", it, ev.get("total_reward"))


func _push_point(key: String, it: int, value: Variant) -> void:
	if value == null or typeof(value) != TYPE_FLOAT:
		return
	if not _points.has(key):
		_points[key] = []
	var arr: Array = _points[key]
	if arr.size() > 20000:
		arr = arr.slice(arr.size() - 20000)
	arr.append({ "x": it, "y": float(value) })
	_points[key] = arr
	if key == _chart.metric_key:
		_chart.push(it, float(value))


# ── Log streaming ────────────────────────────────────────────────────
func _poll_log(tail := false) -> void:
	if not FileAccess.file_exists(LOG_PATH):
		return
	var f := FileAccess.open(LOG_PATH, FileAccess.READ)
	if f == null:
		return
	var size := f.get_length()
	if size < _log_offset:
		_log_offset = 0  # log was replaced
	if not tail and size - _log_offset > (1 << 17):
		_log_offset = size - (1 << 17)  # cap the window on busy runs
		_log_view.text = ""
	var new_bytes := size - _log_offset
	if new_bytes <= 0:
		f.close()
		return
	f.seek(_log_offset)
	# get_as_text() ignores the cursor and returns the whole file; read only
	# the new bytes so appended output isn't re-printed every frame.
	var chunk := f.get_buffer(new_bytes).get_string_from_utf8()
	f.close()
	_log_offset = size
	if chunk.is_empty():
		return
	if tail:
		_log_view.text = ""
	_log_view.append_text(_escape_bbcode(chunk))


func _escape_bbcode(s: String) -> String:
	var out := s.replace("&", "&amp;")
	out = out.replace("[", "[lb]")
	out = out.replace("]", "[/lb]")
	return out


func _clear_log() -> void:
	if FileAccess.file_exists(LOG_PATH):
		DirAccess.remove_absolute(ProjectSettings.globalize_path(LOG_PATH))
	_log_view.text = ""
	_log_offset = 0
	# Reset run telemetry state so the next run streams a fresh chart.
	_telemetry_path = ""
	_telemetry_run_id = ""
	_telemetry_offset = 0
	_points.clear()
	_chart.set_metric("mean_reward_per_step")
	_chart.clear()


func _update_binary_label() -> void:
	var bin := _find_rl_binary()
	if bin.is_empty():
		_binary_label.text = "Binary not built — from the repo root run: cargo build -r -p rl_agent"
		_binary_label.add_theme_color_override("font_color", ERR_COLOR)
	else:
		_binary_label.text = bin
		_binary_label.add_theme_color_override("font_color", TEXT_MID)
	_update_dash_button()


func _update_dash_button() -> void:
	if _find_dash_binary().is_empty():
		_dash_btn.disabled = true
		_dash_btn.tooltip_text = "rl_dash not built — from the repo root run: cargo build -p rl_dash"
	else:
		_dash_btn.disabled = false
		_dash_btn.tooltip_text = "Open the native rl_dash desktop dashboard (live-follow + run comparison)."


func _update_status(text: String, color: Color) -> void:
	_status.text = text
	_status.add_theme_color_override("font_color", color)