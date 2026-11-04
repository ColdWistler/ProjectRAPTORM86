extends Control
## Live flight view for a training run — watch the aircraft while it learns.
##
## Draws the newest frames the trainer has published: a following ground-track
## map with the flight path behind the aircraft, an attitude indicator, air data
## with the range the aircraft has actually flown, control-surface deflection,
## and stream health. Everything is `_draw`-based with no scene tree of its own,
## so it drops into the trainer panel as a single node.
##
## The reader ([rl_frame_reader.gd]) only ever decodes frames that are new, so
## this redraws at the display's frame rate without tracking the trainer's step
## rate. When the trainer steps faster than the display can show, frames are
## skipped rather than queued, and the count is reported on screen — a gapped
## trail that looks continuous would be a lie about what the aircraft did.

const ReaderScript := preload("res://scripts/rl_frame_reader.gd")

const BG := Color(0.02, 0.03, 0.05)
const PANEL := Color(0.07, 0.09, 0.14)
const GRID := Color(0.16, 0.21, 0.30, 0.7)
const AXIS := Color(0.28, 0.34, 0.46)
const TEXT := Color(0.62, 0.70, 0.82)
const TEXT_LIGHT := Color(0.88, 0.92, 0.97)
const TEXT_DIM := Color(0.42, 0.50, 0.62)
const ACCENT := Color(0.30, 0.65, 0.95)
const OK := Color(0.45, 0.85, 0.50)
const WARN := Color(0.95, 0.72, 0.30)
const ERR := Color(0.95, 0.45, 0.40)

const SKY := Color(0.20, 0.42, 0.68)
const GROUND := Color(0.34, 0.26, 0.16)

const PAD := 10.0
const HEADER_H := 22.0

## Metres of map shown per side at scale 1.0; the real scale is chosen per
## frame from the trail's extent so the aircraft always has context.
const MAP_SPAN := 200.0

## Metres per pixel per degree of pitch on the attitude indicator.
const PITCH_PX_PER_DEG := 1.6

## Untyped on purpose: the reader is a `RefCounted` with a wide, dynamically
## shaped surface, and a static type here would turn every field access into a
## compile error for no benefit.
var _reader = null
var _run_dir := ""
var _lag := 0
var _last_lag := -1

## Map scale in pixels per metre, eased toward its target so the view does not
## jitter every time a stray point widens the bounds.
var _scale := 1.0
var _scale_ready := false


func _ready() -> void:
	mouse_filter = Control.MOUSE_FILTER_IGNORE
	resized.connect(queue_redraw)
	set_process(true)


## Point the view at a run directory.
##
## Re-attaching resets the trail, so this is only worth calling when the run
## actually changes. Returns `false` when the run has no stream artifacts; the
## view then shows why instead of an empty map.
func set_run_dir(run_dir: String) -> bool:
	if run_dir == _run_dir and _reader != null and _reader.is_attached():
		return true
	_run_dir = run_dir
	_reader = ReaderScript.new()
	_scale_ready = false
	_lag = 0
	_last_lag = -1
	var ok: bool = _reader.attach(run_dir)
	if ok:
		_reader.poll()
	queue_redraw()
	return ok


func reader() -> Variant:
	return _reader


func _process(_delta: float) -> void:
	if _reader == null or not _reader.is_attached():
		return
	var got := int(_reader.poll())
	# How far the display is behind the trainer. Non-zero every frame during a
	# fast rollout, so it is shown as a rate rather than an error; the number
	# that matters is `skipped`, which counts frames genuinely lost.
	_lag = int(_reader.recorded()) - int(_reader.head)
	# Redrawing a few thousand trail points is not free, so skip it when
	# nothing moved — a paused trainer should not cost frames of render time.
	if got == 0 and _lag == _last_lag:
		return
	_last_lag = _lag
	queue_redraw()


# ── Layout helpers ───────────────────────────────────────────────────
func _rects() -> Dictionary:
	var w := size.x
	var h := size.y
	var body_y := HEADER_H + PAD
	var body_h := maxf(h - body_y - PAD, 40.0)
	var map_w := maxf(w * 0.56, 160.0)
	return {
		"map": Rect2(PAD, body_y, map_w, body_h),
		"instr": Rect2(PAD * 2.0 + map_w, body_y, maxf(w - map_w - PAD * 3.0, 140.0), body_h),
	}


func _draw() -> void:
	var w := size.x
	var h := size.y
	if w <= 1.0 or h <= 1.0:
		return
	draw_rect(Rect2(0, 0, w, h), BG)

	if _reader == null or not _reader.is_attached():
		_draw_idle()
		return

	var r := _rects()
	_draw_header()
	_draw_map(r["map"] as Rect2)
	_draw_instruments(r["instr"] as Rect2)


func _draw_idle() -> void:
	var font := get_theme_default_font()
	var msg := "Waiting for a run…"
	var detail := ""
	if _reader != null and not _reader.error.is_empty():
		msg = "No live flight data"
		detail = _reader.error
	elif _run_dir.is_empty():
		detail = "Start training to watch the aircraft fly."
	else:
		detail = _run_dir
	draw_string(font, Vector2(PAD, HEADER_H + 22.0), msg,
		HORIZONTAL_ALIGNMENT_LEFT, -1, 16, TEXT_LIGHT)
	if not detail.is_empty():
		draw_string(font, Vector2(PAD, HEADER_H + 44.0), detail,
			HORIZONTAL_ALIGNMENT_LEFT, -1, 12, TEXT_DIM)


func _draw_header() -> void:
	var font := get_theme_default_font()
	var f: Dictionary = _reader.latest
	var parts: PackedStringArray = []
	if not f.is_empty():
		parts.append("iter %d · step %d" % [int(f["iteration"]), int(f["step"])])
		parts.append("ep %d" % int(f["episode"]))
		parts.append("%d frames read" % int(_reader.head))
	draw_string(font, Vector2(PAD, 15.0), " · ".join(parts),
		HORIZONTAL_ALIGNMENT_LEFT, -1, 13, TEXT_LIGHT)

	# Stream health, right-aligned. `skipped` is the number that matters: it
	# counts frames that existed and were never shown, from either the
	# trainer's channel or this reader falling behind.
	var skipped := int(_reader.skipped)
	var health := "%d skipped" % skipped if skipped > 0 else "no gaps"
	var health_color := OK if skipped == 0 else WARN
	var text_w := font.get_string_size(health, HORIZONTAL_ALIGNMENT_LEFT, -1, 12).x
	draw_string(font, Vector2(size.x - PAD - text_w, 15.0), health,
		HORIZONTAL_ALIGNMENT_LEFT, -1, 12, health_color)


# ── Ground track ─────────────────────────────────────────────────────
func _draw_map(rect: Rect2) -> void:
	draw_rect(rect, PANEL)
	var font := get_theme_default_font()
	var trail: Array = _reader.trail

	# Centre on the aircraft and keep it there: a following map is the only
	# orientation that stays legible over a long flight.
	var latest: Dictionary = _reader.latest
	var cn := 0.0
	var ce := 0.0
	if latest.is_empty() and not trail.is_empty():
		cn = float(trail[trail.size() - 1]["n"])
		ce = float(trail[trail.size() - 1]["e"])

	var span := _map_scale(trail, cn, ce, rect)
	# NED to screen: east is right, north is up, so the north component is
	# negated. Written out rather than wrapped in a lambda because this runs
	# once per trail point, thousands of times a frame.
	var origin := rect.get_center() - Vector2(ce * span, -cn * span)
	var px := func(n: float, e: float) -> Vector2:
		return origin + Vector2(e * span, -n * span)

	_draw_grid(rect, span, cn, ce, px)

	if trail.size() < 2:
		draw_string(font, rect.get_center() + Vector2(-70, 0), "waiting for frames…",
			HORIZONTAL_ALIGNMENT_LEFT, -1, 13, TEXT_DIM)
		return

	# Decimate to a budget rather than drawing every retained point: a fast
	# rollout would otherwise ask for thousands of polyline segments per frame.
	var budget := int(clampf(rect.size.x / 2.0, 200.0, 1200.0))
	var stride := maxi(1, int(ceil(float(trail.size()) / float(budget))))
	var recent_from := trail.size() - 400

	var all := PackedVector2Array()
	var recent := PackedVector2Array()
	for i in range(0, trail.size(), stride):
		var p: Dictionary = trail[i]
		var pos: Vector2 = px.call(float(p["n"]), float(p["e"]))
		all.append(pos)
		if i >= recent_from:
			recent.append(pos)
	if all.size() >= 2:
		draw_polyline(all, ACCENT * Color(1, 1, 1, 0.30), 1.0)
	if recent.size() >= 2:
		draw_polyline(recent, ACCENT, 2.0)

	# Episode ends: red for a crash (terminated), amber for a step-budget
	# truncation. Watching where these cluster is the point of the view.
	for i in range(maxi(recent_from, 0), trail.size(), maxi(1, stride / 4)):
		var p: Dictionary = trail[i]
		var flags := int(p.get("flags") if p.has("flags") else 0)
		if flags == 0:
			continue
		var pos: Vector2 = px.call(float(p["n"]), float(p["e"]))
		draw_circle(pos, 3.0, ERR if flags == 1 else WARN)

	if not latest.is_empty():
		_draw_aircraft(rect, latest, px.call(float(latest["north"]), float(latest["east"])))

	_draw_scale_bar(rect, span, font)
	draw_string(font, rect.position + Vector2(8, rect.size.y - 8),
		"ground track · north up · %d pts" % trail.size(),
		HORIZONTAL_ALIGNMENT_LEFT, -1, 11, TEXT_DIM)


## Pixels per metre, fitted to the trail and eased so it does not flicker.
func _map_scale(trail: Array, cn: float, ce: float, rect: Rect2) -> float:
	if trail.is_empty():
		return rect.size.x / (2.0 * MAP_SPAN)
	# Fit the whole retained trail: a partial view would let the aircraft leave
	# the frame, which defeats the purpose of the map.
	var max_n := 1.0
	var max_e := 1.0
	for p in trail:
		max_n = maxf(max_n, absf(float(p["n"]) - cn))
		max_e = maxf(max_e, absf(float(p["e"]) - ce))
	var needed_n := max_n / maxf(rect.size.y * 0.42, 1.0)
	var needed_e := max_e / maxf(rect.size.x * 0.42, 1.0)
	var target := minf(needed_n, needed_e)
	if not _scale_ready:
		_scale = target
		_scale_ready = true
	else:
		# Ease toward the target: the trail grows monotonically at first, so
		# an instant fit would zoom out on every single frame.
		_scale = lerpf(_scale, target, 0.08)
	return maxf(_scale, 1e-6)


## Graticule at a round metre spacing, bounded by what the map can actually show.
func _draw_grid(rect: Rect2, span: float, cn: float, ce: float, px: Callable) -> void:
	var step := _nice_step(rect.size.x / (8.0 * span))
	if step <= 0.0 or not is_finite(step):
		return
	# Extent the map covers, walked as integer multiples of `step` so the lines
	# cannot drift by float accumulation and cannot run away.
	var half_e := rect.size.x * 0.5 / span
	var half_n := rect.size.y * 0.5 / span
	var i0 := int(floorf((ce - half_e) / step))
	var i1 := int(ceilf((ce + half_e) / step))
	var j0 := int(floorf((cn - half_n) / step))
	var j1 := int(ceilf((cn + half_n) / step))
	# A degenerate scale would produce an unbounded index range; the trail
	# normally bounds this, and this guard makes the bound explicit.
	i0 = maxi(i0, -1000000)
	i1 = mini(i1, 1000000)
	j0 = maxi(j0, -1000000)
	j1 = mini(j1, 1000000)

	var top := rect.position.y
	var bottom := rect.end.y
	for i in range(i0, i1 + 1):
		var x: float = rect.get_center().x + (float(i) * step - ce) * span
		draw_line(Vector2(x, top), Vector2(x, bottom), GRID, 1.0)
	for j in range(j0, j1 + 1):
		var y: float = rect.get_center().y - (float(j) * step - cn) * span
		draw_line(Vector2(rect.position.x, y), Vector2(rect.end.x, y), GRID, 1.0)
	draw_rect(rect, AXIS, false, 1.0)


## Round a metre spacing to 1, 2 or 5 times a power of ten.
func _nice_step(raw: float) -> float:
	if raw <= 0.0 or not is_finite(raw):
		return 0.0
	var mag := pow(10.0, floorf(log(raw) / log(10.0)))
	var norm := raw / mag
	if norm <= 1.0:
		return mag
	if norm <= 2.0:
		return 2.0 * mag
	if norm <= 5.0:
		return 5.0 * mag
	return 10.0 * mag


func _draw_aircraft(rect: Rect2, f: Dictionary, pos: Vector2) -> void:
	# Nose-up triangle oriented by heading. In NED, yaw is measured clockwise
	# from north, so the screen heading vector is (sin yaw, -cos yaw).
	var heading := float(f["yaw"])
	var dir := Vector2(sin(heading), -cos(heading))
	var right := Vector2(dir.y, -dir.x)
	var tip := pos + dir * 13.0
	var a := pos + dir * 6.0 + right * 8.0
	var b := pos + dir * 6.0 - right * 8.0
	var tail := pos - dir * 5.0
	draw_colored_polygon(PackedVector2Array([tip, a, tail]), ACCENT)
	draw_colored_polygon(PackedVector2Array([tip, tail, b]), ACCENT.darkened(0.25))
	draw_polyline(PackedVector2Array([a, tail, b]), ACCENT, 1.0)

	# Ground-velocity vector, which differs from heading whenever the aircraft
	# is crabbing — the most instructive thing on the whole map.
	var vn := float(f["vel_north"])
	var ve := float(f["vel_east"])
	var speed := sqrt(vn * vn + ve * ve)
	if speed > 0.5:
		var len_px := clampf(speed / maxf(_scale, 1e-6), 10.0, rect.size.y * 0.35)
		var tip2 := pos + Vector2(ve, -vn).normalized() * len_px
		draw_line(pos, tip2, OK, 1.5)


func _draw_scale_bar(rect: Rect2, span: float, font: Font) -> void:
	var target_px := rect.size.x * 0.22
	var metres := _nice_step(target_px / span)
	if metres <= 0.0:
		return
	var px := metres * span
	var base := rect.position + Vector2(rect.size.x - px - 14.0, rect.size.y - 20.0)
	draw_line(base, base + Vector2(px, 0), TEXT, 2.0)
	draw_line(base + Vector2(0, -4), base + Vector2(0, 4), TEXT, 2.0)
	draw_line(base + Vector2(px, -4), base + Vector2(px, 4), TEXT, 2.0)
	var label := "%g m" % metres
	draw_string(font, base + Vector2(px * 0.5 - 22.0, -8.0), label,
		HORIZONTAL_ALIGNMENT_LEFT, -1, 11, TEXT)


# ── Instruments ──────────────────────────────────────────────────────
func _draw_instruments(rect: Rect2) -> void:
	draw_rect(rect, PANEL)
	var f: Dictionary = _reader.latest
	var font := get_theme_default_font()
	if f.is_empty():
		draw_string(font, rect.get_center() + Vector2(-60, 0), "no frames yet",
			HORIZONTAL_ALIGNMENT_LEFT, -1, 13, TEXT_DIM)
		return

	var y := rect.position.y + 8.0
	var bottom := rect.end.y - 8.0
	var left := rect.position.x + 10.0
	var right := rect.end.x - 10.0

	# Attitude indicator occupies the top-left; air data stacks to its right.
	var horizon_d := minf(132.0, minf(rect.size.x * 0.44, rect.size.y * 0.42))
	y = _draw_attitude(Rect2(left, y, horizon_d, horizon_d), f, font)

	var air_x := left + horizon_d + 14.0
	var air_w := right - air_x
	y = _draw_air_data(Rect2(air_x, rect.position.y + 8.0, air_w, horizon_d), f, font)

	y += 10.0
	y = _draw_attitude_numbers(Rect2(left, y, rect.size.x - 20.0, 0.0), f, font, y)
	y = _draw_controls(Rect2(left, y + 6.0, rect.size.x - 20.0, 0.0), f, font, y, bottom)


## Artificial horizon: sky disc, ground segment, pitch ladder, fixed symbol.
func _draw_attitude(rect: Rect2, f: Dictionary, font: Font) -> float:
	var c := rect.get_center()
	var r := minf(rect.size.x, rect.size.y) * 0.5 - 6.0
	if r <= 4.0:
		return rect.end.y
	var roll := float(f["roll"])
	var pitch := float(f["pitch"])

	# Screen offset of the horizon line. Nose up moves the horizon down.
	var h_off := rad_to_deg(pitch) * PITCH_PX_PER_DEG
	var clamped := clampf(h_off, -r, r)

	draw_circle(c, r, SKY)

	# Ground is the circular segment below the horizon chord, so at high pitch
	# the sky correctly narrows to a sliver instead of overflowing.
	var x0 := sqrt(maxf(r * r - clamped * clamped, 0.0))
	var phi := atan2(clamped, x0)
	var seg := PackedVector2Array()
	seg.append(c + Vector2(-x0, clamped))
	var samples := 40
	for i in range(samples + 1):
		var a: float = lerpf(phi, PI - phi, float(i) / float(samples))
		seg.append(c + Vector2(cos(a) * r, sin(a) * r))
	draw_colored_polygon(seg, GROUND)
	draw_arc(c, r, 0.0, TAU, 96, AXIS, 2.0)

	# Pitch ladder, rotated with the aircraft.
	draw_set_transform(c, -roll, Vector2.ONE)
	draw_line(Vector2(-x0, clamped), Vector2(x0, clamped), TEXT_LIGHT, 1.5)
	for deg in [-30, -20, -10, 10, 20, 30]:
		var y: float = clamped - float(deg) * PITCH_PX_PER_DEG
		if absf(y) > r:
			continue
		var half := 34.0 if absi(deg) % 20 == 0 else 20.0
		draw_line(Vector2(-half, y), Vector2(half, y), TEXT, 1.0)
		var label := "%d" % absi(deg)
		draw_string(font, Vector2(half + 5.0, y + 4.0), label,
			HORIZONTAL_ALIGNMENT_LEFT, -1, 10, TEXT)
	# Bank scale at the top.
	for deg in [-60, -45, -30, -20, 0, 20, 30, 45, 60]:
		var a := deg_to_rad(deg) - PI * 0.5
		var d := Vector2(cos(a), sin(a)) * (r - 6.0)
		var tick := Vector2(cos(a), sin(a)) * (r - (14.0 if deg % 30 == 0 else 10.0))
		draw_line(c + d, c + tick, TEXT_DIM, 1.0)
	draw_set_transform(Vector2.ZERO, 0.0, Vector2.ONE)

	# Fixed aircraft symbol — the one element that must not move with attitude.
	draw_line(c + Vector2(-30, 0), c + Vector2(-10, 0), WARN, 2.0)
	draw_line(c + Vector2(10, 0), c + Vector2(30, 0), WARN, 2.0)
	draw_circle(c, 3.0, WARN)
	draw_arc(c, r, -PI * 0.5 - deg_to_rad(30.0), -PI * 0.5 - deg_to_rad(20.0), 8, WARN, 2.0)
	draw_arc(c, r, -PI * 0.5 + deg_to_rad(20.0), -PI * 0.5 + deg_to_rad(30.0), 8, WARN, 2.0)
	return rect.end.y


## Air data with the range the aircraft has actually flown over the trail.
func _draw_air_data(rect: Rect2, f: Dictionary, font: Font) -> float:
	var y := rect.position.y
	var w := rect.size.x
	var trail: Array = _reader.trail
	var ias_range := _range_of(trail, "ias")
	var alt_range := _range_of(trail, "alt")

	y = _air_row(rect, y, w, "IAS", "m/s", float(f["airspeed"]), ias_range, ACCENT, font)
	# Indicated vs true airspeed: the gap is the sensor noise the agent flies
	# through, so showing both is the point rather than a nicety.
	y = _air_row(rect, y, w, "ALT", "m", float(f["altitude"]), alt_range, OK, font)
	y = _bipolar_row(rect, y, w, "V/S", "m/s", -float(f["vel_down"]), 6.0, ACCENT, font)
	y = _bipolar_row(rect, y, w, "AoA", "deg", rad_to_deg(float(f["alpha"])), 15.0, WARN, font)
	y = _bipolar_row(rect, y, w, "AoS", "deg", rad_to_deg(float(f["beta"])), 15.0, WARN, font)
	y = _bipolar_row(rect, y, w, "REW", "", float(f["reward"]), _max_abs(1.0, float(f["reward"])), OK, font)

	var ias_true := float(f["airspeed"])
	var ias_ind := float(f["airspeed_indicated"])
	draw_string(font, Vector2(rect.position.x, y + 12.0),
		"IAS ind %.1f  ·  baro alt %.0f  ·  lag %d fr" % [ias_ind, float(f["baro_altitude"]), _lag],
		HORIZONTAL_ALIGNMENT_LEFT, -1, 11, TEXT_DIM)
	return y + 18.0


func _air_row(rect: Rect2, y: float, w: float, label: String, unit: String,
		value: float, value_range: Vector2, color: Color, font: Font) -> float:
	draw_string(font, Vector2(rect.position.x, y + 11.0), label,
		HORIZONTAL_ALIGNMENT_LEFT, -1, 12, TEXT_DIM)
	var text := "%.1f" % value
	if not unit.is_empty():
		text += " " + unit
	draw_string(font, Vector2(rect.position.x + w - 62.0, y + 11.0), text,
		HORIZONTAL_ALIGNMENT_LEFT, -1, 12, TEXT_LIGHT)
	var bar := Rect2(rect.position.x, y + 16.0, w, 6.0)
	_track(bar, ACCENT * Color(1, 1, 1, 0.15))
	if value_range.y > value_range.x:
		var lo := bar.position.x + (value_range.x - value_range.y) * 0.5
		var hi := bar.position.x + (value_range.y - value_range.x) * 0.5
		draw_rect(Rect2(lo, bar.position.y, hi - lo, bar.size.y), color * Color(1, 1, 1, 0.35))
	var span: float = maxf(value_range.y - value_range.x, 1e-3)
	var px := bar.position.x + (value - value_range.x) / span * bar.size.x
	draw_line(Vector2(px, bar.position.y - 2.0), Vector2(px, bar.end.y + 2.0), color, 2.0)
	return y + 30.0


func _bipolar_row(rect: Rect2, y: float, w: float, label: String, unit: String,
		value: float, span: float, color: Color, font: Font) -> float:
	draw_string(font, Vector2(rect.position.x, y + 11.0), label,
		HORIZONTAL_ALIGNMENT_LEFT, -1, 12, TEXT_DIM)
	var text := "%.2f" % value
	if not unit.is_empty():
		text += " " + unit
	draw_string(font, Vector2(rect.position.x + w - 62.0, y + 11.0), text,
		HORIZONTAL_ALIGNMENT_LEFT, -1, 12, TEXT_LIGHT)
	var bar := Rect2(rect.position.x, y + 16.0, w, 6.0)
	_track(bar, ACCENT * Color(1, 1, 1, 0.15))
	var mid := bar.get_center().x
	draw_line(Vector2(mid, bar.position.y - 1.0), Vector2(mid, bar.end.y + 1.0), AXIS, 1.0)
	var frac: float = clampf(value / maxf(span, 1e-6), -1.0, 1.0)
	var px := mid + frac * (bar.size.x * 0.5)
	draw_line(Vector2(px, bar.position.y - 2.0), Vector2(px, bar.end.y + 2.0), color, 2.0)
	return y + 28.0


func _track(rect: Rect2, fill: Color) -> void:
	draw_rect(rect, fill)
	draw_rect(rect, AXIS, false, 1.0)


func _range_of(trail: Array, key: String) -> Vector2:
	if trail.is_empty():
		return Vector2(0, 1)
	var lo := INF
	var hi := -INF
	# Sampling the whole trail every frame is what the raw data costs; stride it
	# because a range over a few thousand points does not need all of them.
	var stride := maxi(1, trail.size() / 400)
	for i in range(0, trail.size(), stride):
		var v := float(trail[i][key])
		lo = minf(lo, v)
		hi = maxf(hi, v)
	if hi - lo < 1e-6:
		hi = lo + 1.0
	return Vector2(lo, hi)


func _max_abs(a: float, b: float) -> float:
	return maxf(a, absf(b))


func _draw_attitude_numbers(rect: Rect2, f: Dictionary, font: Font, y: float) -> float:
	var cells := [
		["ROLL", "%.1f°" % rad_to_deg(float(f["roll"]))],
		["PITCH", "%.1f°" % rad_to_deg(float(f["pitch"]))],
		["YAW", "%.1f°" % rad_to_deg(wrapf(float(f["yaw"]), -PI, PI))],
		["p", "%.2f" % float(f["p"])],
		["q", "%.2f" % float(f["q"])],
		["r", "%.2f" % float(f["r"])],
	]
	var cw := rect.size.x / 3.0
	var rh := 34.0
	for i in cells.size():
		var cx := rect.position.x + cw * (i % 3)
		var cy := y + rh * (i / 3)
		draw_string(font, Vector2(cx, cy + 11.0), cells[i][0],
			HORIZONTAL_ALIGNMENT_LEFT, -1, 11, TEXT_DIM)
		draw_string(font, Vector2(cx, cy + 25.0), cells[i][1],
			HORIZONTAL_ALIGNMENT_LEFT, -1, 13, TEXT_LIGHT)
	return y + rh * 2


func _draw_controls(rect: Rect2, f: Dictionary, font: Font, y: float, bottom: float) -> float:
	var avail := bottom - y
	if avail < 70.0:
		return y
	draw_line(Vector2(rect.position.x, y), Vector2(rect.end.x, y), AXIS, 1.0)
	draw_string(font, Vector2(rect.position.x, y + 14.0), "CONTROLS",
		HORIZONTAL_ALIGNMENT_LEFT, -1, 11, TEXT_DIM)
	y += 22.0

	# Control deflection is what the policy is actually asking for, so it gets
	# the largest bars on the panel — these are the numbers being optimised.
	var rows := [
		["ELEV", float(f["elevator_deg"]), 20.0, ACCENT],
		["AIL", float(f["aileron_deg"]), 20.0, ACCENT],
		["RUD", float(f["rudder_deg"]), 20.0, ACCENT],
	]
	var bar_w := (rect.size.x - 90.0) / 3.0
	for i in rows.size():
		var x := rect.position.x + i * (bar_w + 45.0)
		_deflection_bar(Rect2(x, y, bar_w, 34.0), rows[i][0], rows[i][1], rows[i][2], rows[i][3], font)
	y += 40.0

	var thr := clampf(float(f["throttle"]), 0.0, 1.0)
	draw_string(font, Vector2(rect.position.x, y + 11.0), "THR",
		HORIZONTAL_ALIGNMENT_LEFT, -1, 12, TEXT_DIM)
	draw_string(font, Vector2(rect.position.x + rect.size.x - 46.0, y + 11.0), "%.2f" % thr,
		HORIZONTAL_ALIGNMENT_LEFT, -1, 12, TEXT_LIGHT)
	var tbar := Rect2(rect.position.x, y + 16.0, rect.size.x, 8.0)
	_track(tbar, ACCENT * Color(1, 1, 1, 0.15))
	draw_rect(Rect2(tbar.position.x, tbar.position.y, tbar.size.x * thr, tbar.size.y), OK)
	draw_line(Vector2(tbar.position.x, tbar.position.y - 2.0),
		Vector2(tbar.position.x, tbar.end.y + 2.0), TEXT_LIGHT, 1.0)
	return y + 30.0


func _deflection_bar(rect: Rect2, label: String, value: float, span: float,
		color: Color, font: Font) -> void:
	draw_string(font, Vector2(rect.position.x, rect.position.y - 3.0), label,
		HORIZONTAL_ALIGNMENT_LEFT, -1, 11, TEXT_DIM)
	var bar := Rect2(rect.position.x, rect.position.y, rect.size.x, 10.0)
	_track(bar, ACCENT * Color(1, 1, 1, 0.15))
	var mid := bar.get_center()
	var frac: float = clampf(value / maxf(span, 1e-6), -1.0, 1.0)
	draw_line(Vector2(mid.x, bar.position.y - 2.0), Vector2(mid.x, bar.end.y + 2.0), AXIS, 1.0)
	# A surface is only "commanded" on one side; fill from the centre so the
	# sign is visible at a glance.
	var end_x := mid.x + frac * (bar.size.x * 0.5)
	var fill := Rect2(minf(mid.x, end_x), bar.position.y, absf(end_x - mid.x), bar.size.y)
	draw_rect(fill, color)
	draw_string(font, Vector2(rect.position.x, rect.end.y + 12.0), "%.1f°" % value,
		HORIZONTAL_ALIGNMENT_LEFT, -1, 11, TEXT_LIGHT)
