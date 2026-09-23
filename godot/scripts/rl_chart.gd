extends Control
## Lightweight live training chart for the RL trainer panel.
##
## A dependency-free `_draw`-based line chart: given a sequence of
## (iteration, value) points it plots one metric against the iteration
## number with a grid, auto-scaled Y axis, min/max labels and a hover
## tooltip. Data is pushed in batches while a run streams `metrics.jsonl`
## and the panel re-issues `set_metric` without touching stored points.
##
## Extends nothing but Control, so it can be embedded anywhere.

const PAD_LEFT := 44.0
const PAD_RIGHT := 14.0
const PAD_TOP := 26.0
const PAD_BOTTOM := 24.0
const GRID_LINES := 5

const BG := Color(0.02, 0.03, 0.05)
const GRID_COLOR := Color(0.14, 0.18, 0.26, 0.6)
const AXIS_COLOR := Color(0.28, 0.34, 0.46)
const TEXT_COLOR := Color(0.55, 0.63, 0.75)
const TEXT_LIGHT := Color(0.88, 0.92, 0.97)
const HOVER_COLOR := Color(0.95, 0.80, 0.30, 0.9)

## Metric registry shared with the panel dropdown: key -> label + color.
const METRICS := {
	"mean_reward_per_step": { "label": "Reward / step", "color": Color(0.45, 0.85, 0.50) },
	"policy_loss":          { "label": "Policy loss",    "color": Color(0.95, 0.60, 0.20) },
	"value_loss":           { "label": "Value loss",     "color": Color(0.30, 0.65, 0.95) },
	"entropy":              { "label": "Entropy",        "color": Color(0.75, 0.45, 0.95) },
	"advantage_mean":       { "label": "Advantage mean", "color": Color(0.95, 0.80, 0.30) },
	"alt_err":              { "label": "Eval alt err (m)",   "color": Color(0.95, 0.45, 0.40) },
	"spd_err":              { "label": "Eval spd err (m/s)", "color": Color(0.30, 0.85, 0.85) },
	"eval_reward":          { "label": "Eval mean reward",   "color": Color(0.55, 0.80, 0.45) },
}

## Raw series: Array of {"x": int iteration, "y": float value}.
var points: Array = []
var metric_key := "mean_reward_per_step"
var title := ""
## Run id (dir name) whose records `points` holds; shown in the title so it is
## clear which run is plotted (the chart falls back to an older run while a
## fresh one is still writing its first record).
var run_id := ""

var _hover_index := -1

## Cap on stored points per series (keeps long runs bounded in memory).
const MAX_POINTS := 5000


func _ready() -> void:
	mouse_filter = Control.MOUSE_FILTER_STOP
	resized.connect(queue_redraw)
	mouse_exited.connect(_on_mouse_exited)
	_font_size(12)


func _on_mouse_exited() -> void:
	_hover_index = -1
	queue_redraw()


func set_metric(key: String) -> void:
	metric_key = key
	update_title()
	queue_redraw()


func clear() -> void:
	points.clear()
	_hover_index = -1
	queue_redraw()


## Append a point and drop the head if the cap is exceeded.
func push(iteration: int, value: float) -> void:
	points.append({ "x": iteration, "y": value })
	if points.size() > MAX_POINTS:
		points = points.slice(points.size() - MAX_POINTS)
	update_title()
	queue_redraw()


func point_count() -> int:
	return points.size()


func metric_label() -> String:
	return str(METRICS[metric_key]["label"])


func metric_color() -> Color:
	return METRICS[metric_key]["color"]


func update_title() -> void:
	var latest: Variant = latest_value()
	var run := "" if run_id.is_empty() else "  ·  run " + run_id
	if latest == null:
		title = metric_label() + "  —  waiting for telemetry…" + run
	else:
		title = "%s  ·  latest %s @ iter %d%s" % [metric_label(), _fmt(float(latest)), latest_iteration(), run]
	queue_redraw()


func latest_value() -> Variant:
	if points.is_empty():
		return null
	return points[-1]["y"]


func latest_iteration() -> int:
	if points.is_empty():
		return 0
	return int(points[-1]["x"])


func _gui_input(event: InputEvent) -> void:
	if event is InputEventMouseMotion:
		_hover_index = _hit_test(event.position)
		queue_redraw()
	elif event is InputEventMouseButton and event.pressed and event.button_index == MOUSE_BUTTON_WHEEL_UP:
		# Keep wheel scrolling from bubbling to the page ScrollContainer.
		accept_event()


func _draw() -> void:
	var w := size.x
	var h := size.y
	if w <= 0.0 or h <= 0.0:
		return

	draw_rect(Rect2(0, 0, w, h), BG)
	var plot := Rect2(PAD_LEFT, PAD_TOP, w - PAD_LEFT - PAD_RIGHT, h - PAD_TOP - PAD_BOTTOM)

	draw_string(get_theme_default_font(), Vector2(PAD_LEFT, 16), title,
		HORIZONTAL_ALIGNMENT_LEFT, -1, 13, TEXT_LIGHT)

	if points.is_empty():
		draw_string(get_theme_default_font(), plot.position + Vector2(0, plot.size.y * 0.5),
			"No data yet — start training to stream metrics.", HORIZONTAL_ALIGNMENT_LEFT, -1, 13, TEXT_COLOR)
		return

	var ys: Array = []
	for p in points:
		ys.append(float(p["y"]))
	var y_min := ys.min() as float
	var y_max := ys.max() as float
	if absf(y_max - y_min) < 1e-9:
		y_min -= 1.0
		y_max += 1.0
	var span := y_max - y_min

	# Horizontal grid + Y labels.
	for i in GRID_LINES + 1:
		var t := float(i) / float(GRID_LINES)
		var y := plot.position.y + plot.size.y * (1.0 - t)
		var value := y_min + t * span
		draw_line(Vector2(plot.position.x, y), Vector2(plot.end.x, y), GRID_COLOR, 1.0)
		draw_string(get_theme_default_font(), Vector2(0, y + 4), _fmt(value),
			HORIZONTAL_ALIGNMENT_LEFT, PAD_LEFT - 6, 11, TEXT_COLOR)

	# Axes frame.
	draw_rect(plot, AXIS_COLOR, false, 1.0)

	# Min/max labels.
	draw_string(get_theme_default_font(), Vector2(plot.position.x, plot.position.y - 6),
		"max %s" % _fmt(y_max), HORIZONTAL_ALIGNMENT_LEFT, -1, 11, TEXT_COLOR)
	draw_string(get_theme_default_font(), Vector2(plot.position.x, plot.end.y + 16),
		"min %s" % _fmt(y_min), HORIZONTAL_ALIGNMENT_LEFT, -1, 11, TEXT_COLOR)

	# X labels: first and last iteration.
	var x_first := int(points[0]["x"])
	var x_last := int(points[-1]["x"])
	draw_string(get_theme_default_font(), Vector2(plot.position.x, plot.end.y + 16),
		"iter %d" % x_first, HORIZONTAL_ALIGNMENT_LEFT, -1, 11, TEXT_COLOR)
	draw_string(get_theme_default_font(), Vector2(plot.end.x, plot.end.y + 16),
		"iter %d" % x_last, HORIZONTAL_ALIGNMENT_RIGHT, -1, 11, TEXT_COLOR)

	# Polyline.
	var color := metric_color()
	var prev: Vector2 = Vector2.ZERO
	for i in points.size():
		var px := _x_of(i, x_first, x_last, plot)
		var py := plot.position.y + plot.size.y * (1.0 - (float(points[i]["y"]) - y_min) / span)
		var pos := Vector2(px, py)
		if i > 0:
			draw_line(prev, pos, color, 2.0)
		draw_circle(pos, 1.6, color.lightened(0.2))
		prev = pos

	# Hover tooltip.
	if _hover_index >= 0 and _hover_index < points.size():
		var px := _x_of(_hover_index, x_first, x_last, plot)
		var py := plot.position.y + plot.size.y * (1.0 - (float(points[_hover_index]["y"]) - y_min) / span)
		draw_line(Vector2(px, plot.position.y), Vector2(px, plot.end.y), HOVER_COLOR, 1.0)
		draw_circle(Vector2(px, py), 4.0, HOVER_COLOR)
		draw_string(get_theme_default_font(), Vector2(px + 8, py - 8),
			"iter %d  %s" % [int(points[_hover_index]["x"]), _fmt(float(points[_hover_index]["y"]))],
			HORIZONTAL_ALIGNMENT_LEFT, -1, 12, TEXT_LIGHT)


func _x_of(index: int, x_first: int, x_last: int, plot: Rect2) -> float:
	if x_last <= x_first:
		return plot.position.x
	var t := (int(points[index]["x"]) - x_first) as float / (x_last - x_first) as float
	return plot.position.x + t * plot.size.x


func _hit_test(pos: Vector2) -> int:
	if points.is_empty():
		return -1
	var plot := Rect2(PAD_LEFT, PAD_TOP,
		size.x - PAD_LEFT - PAD_RIGHT, size.y - PAD_TOP - PAD_BOTTOM)
	if not plot.has_point(pos):
		return -1
	var x_first := int(points[0]["x"])
	var x_last := int(points[-1]["x"])
	var best := -1
	var best_dist := 1e9
	for i in points.size():
		var px := _x_of(i, x_first, x_last, plot)
		var d := absf(px - pos.x)
		if d < best_dist:
			best_dist = d
			best = i
	return best


## Compact numeric label: 2 decimals, 1 above 100, 0 above 1000.
func _fmt(v: float) -> String:
	var a := absf(v)
	if a >= 10000.0:
		return "%.0f" % v
	if a >= 100.0:
		return "%.1f" % v
	return "%.2f" % v


func _font_size(sz: int) -> void:
	add_theme_font_size_override("font_size", sz)