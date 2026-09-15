class_name FlightHUD
extends Control
## Full-screen pilot-style HUD overlay. Drawn from the Rust telemetry array
## (indices 0..=24, see `FlightSimNode::telemetry`) via a single `_draw()`.
##
## Layout (PFD-inspired, no separate windows):
##   * Center — attitude indicator (artificial horizon): rotating pitch ladder,
##     bank scale and fixed aircraft reference.
##   * Left  — airspeed tape (IAS) with a moving bug box.
##   * Right — altitude tape (ft) plus a climb/descent (VSI) tape.
##   * Top   — heading tape with N/E/S/W cardinal dots.
##   * Corners — mode/autopilot/stall chips and engine status.
##   * Bottom — wind, AoA, TAS and a compact control legend.

const ACCENT := Color(0.00, 0.80, 1.00, 1.0)
const GOOD := Color(0.30, 1.00, 0.35, 1.0)
const WARN := Color(1.00, 0.70, 0.25, 1.0)
const DANGER := Color(1.00, 0.25, 0.20, 1.0)
const WHITE := Color(0.95, 0.96, 0.98, 1.0)
const DIM := Color(0.55, 0.62, 0.68, 1.0)
const FADE := Color(0.15, 0.20, 0.25, 0.45)
const BG := Color(0.02, 0.05, 0.08, 0.85)

var _t := PackedFloat64Array()          # latest telemetry
var _snap := PackedFloat64Array()       # avionics snapshot (46 ch) for battery
var _avionics := false
var _auto_level := false
var _engine_out := 0

## Push a fresh telemetry vector and request a redraw.
func update_telemetry(t: PackedFloat64Array, snap: PackedFloat64Array, avionics: bool, auto_level: bool, engine_out: int) -> void:
	_t = t
	_snap = snap
	_avionics = avionics
	_auto_level = auto_level
	_engine_out = engine_out
	queue_redraw()

## True after the first telemetry arrives.
func has_data() -> bool:
	return _t.size() >= 25

func _draw() -> void:
	if _t.size() < 25:
		return
	var w := size.x
	var h := size.y
	var cx := w * 0.5
	var cy := h * 0.42

	_draw_attitude(Vector2(w * 0.17, h * 0.80), minf(w, h))
	_draw_airspeed(w, h)
	_draw_altitude(w, h)
	_draw_heading(Vector2(cx, h * 0.10), w)
	_draw_status(w, h)
	_draw_engine_strip(w, h)
	_draw_legend(w, h)

# =========================================================================== #
# Attitude indicator
# =========================================================================== #
func _draw_attitude(c: Vector2, vp: float) -> void:
	var pitch := _t[11]
	var radius := vp * 0.050       # ladder / bank-scale radius
	var px := vp * 0.00105         # px per degree of pitch

	# Transparent background so the drone stays visible; only the instruments
	# themselves are drawn.
	draw_arc(c, radius * 1.20, 0.0, TAU, 72, Color(0.35, 0.45, 0.55, 0.28), 1.2, true)

	# Pitch ladder rotated so the horizon sits against the bank. With the nose
	# up (pitch > 0) the horizon line drops below the fixed wings, so each row
	# is offset by the angular difference from the current pitch.
	var rot := -deg_to_rad(_t[12])
	for deg in range(-90, 100, 5):
		var off := (pitch - deg) * px
		var center_y := c.y + off
		var p0 := _rot(c, Vector2(c.x - radius * 1.0, center_y), rot)
		var p1 := _rot(c, Vector2(c.x + radius * 1.0, center_y), rot)
		if deg % 10 == 0:
			draw_line(p0, p1, WHITE, 2.0, true)
			if absf(deg) <= 60 and deg != 0:
				var lbl := "%+d" % -deg
				var lp := _rot(c, Vector2(c.x + radius * 0.55, center_y - 4), rot)
				draw_string(ThemeDB.fallback_font, lp, lbl, HORIZONTAL_ALIGNMENT_CENTER, 80, 13, WHITE)
		else:
			# Dashed short reference on five-degree rows.
			draw_line(p0.lerp(p1, 0.5) + Vector2(-radius * 0.10, 0), p0.lerp(p1, 0.5) + Vector2(radius * 0.10, 0), DIM, 1.5, true)

	# Bank-angle scale around the top: ticks at fixed banks, pointer follows roll.
	for bank in [-60, -45, -30, -20, -10, 10, 20, 30, 45, 60]:
		var br := deg_to_rad(bank)
		var dir := Vector2(sin(br), -cos(br))
		var tip := 0.06 if bank % 30 == 0 else 0.035
		draw_line(c + dir * radius, c + dir * (radius * (1.0 + tip)), DIM, 1.5, true)
	var phead: Vector2 = c + Vector2(sin(deg_to_rad(_t[12])), -cos(deg_to_rad(_t[12]))) * radius * 0.94
	draw_colored_polygon(PackedVector2Array([
		phead + Vector2(-5, -3), phead + Vector2(5, -3), phead + Vector2(0, 4),
	]), WHITE)

	# Fixed aircraft reference: wings + centre dot.
	draw_line(c + Vector2(-radius * 0.82, 0), c + Vector2(radius * 0.82, 0), WHITE, 2.5, true)
	draw_circle(c, 3.0, WHITE)

# =========================================================================== #
# Airspeed tape (left edge), IAS in knots
# =========================================================================== #
func _draw_airspeed(w: float, h: float) -> void:
	var ias := _t[5]
	var x := w * 0.09
	var cy := h * 0.5
	var px := 1.6                       # px per knot
	var box := Rect2(x - 30, cy - 14, 62, 28)

	for k in range(int(ias) - 90, int(ias) + 90, 5):
		if k < 0:
			continue
		var y := cy + (ias - k) * px
		if y < 32 or y > h - 40:
			continue
		var major := k % 10 == 0
		var x0 := x - (52.0 if major else 32.0)
		draw_line(Vector2(x0, y), Vector2(x, y), WHITE if major else DIM, 2.0 if major else 1.0, true)
		if major:
			draw_string(ThemeDB.fallback_font, Vector2(x0 - 4, y + 4), str(k), HORIZONTAL_ALIGNMENT_RIGHT, 60, 13, WHITE)

	draw_rect(box, BG)
	draw_rect(box, ACCENT, false, 1.5)
	draw_string(ThemeDB.fallback_font, Vector2(box.position.x + 31, box.position.y + 19), "%d" % roundi(ias), HORIZONTAL_ALIGNMENT_CENTER, 0, 20, WHITE)
	draw_string(ThemeDB.fallback_font, Vector2(x - 30, cy + 26), "IAS", HORIZONTAL_ALIGNMENT_LEFT, 60, 10, DIM)

# =========================================================================== #
# Altitude tape (right edge) + VSI tape beside it
# =========================================================================== #
func _draw_altitude(w: float, h: float) -> void:
	var alt := _t[1]                     # feet
	var x := w * 0.91
	var cy := h * 0.5
	var px := 0.035                      # px per foot
	var box := Rect2(x - 32, cy - 14, 66, 28)

	for v in range(int(alt) - 1500, int(alt) + 1500, 100):
		if v < 0:
			continue
		var y := cy + (alt - v) * px
		if y < 32 or y > h - 40:
			continue
		var major := v % 500 == 0
		var x0 := x + (52.0 if major else 32.0)
		draw_line(Vector2(x, y), Vector2(x0, y), WHITE if major else DIM, 2.0 if major else 1.0, true)
		if major:
			draw_string(ThemeDB.fallback_font, Vector2(x0 + 4, y + 4), str(v), HORIZONTAL_ALIGNMENT_LEFT, 70, 12, WHITE)

	draw_rect(box, BG)
	draw_rect(box, ACCENT, false, 1.5)
	draw_string(ThemeDB.fallback_font, Vector2(box.position.x + 33, box.position.y + 19), "%d" % roundi(alt), HORIZONTAL_ALIGNMENT_CENTER, 0, 18, WHITE)

	# VSI column inboard of the altitude tape.
	var vsi_x := w * 0.97
	var vy := cy
	var fpm := _t[14] * (_t[2] * 196.85) * PI / 180.0    # climb angle -> ft/min
	var clamp_fpm := clampf(fpm, -2000.0, 2000.0)
	var vpx := 26.0
	draw_line(Vector2(vsi_x - 10, vy), Vector2(vsi_x + 10, vy), DIM, 1.0, true)
	for tic_fpm in [-1500, -1000, -500, 500, 1000, 1500]:
		var y: float = vy - float(tic_fpm) / 1000.0 * vpx
		draw_line(Vector2(vsi_x - 4, y), Vector2(vsi_x + 4, y), DIM, 1.0, true)
	var ycur := vy - clamp_fpm / 1000.0 * vpx
	draw_line(Vector2(vsi_x - 16, ycur), Vector2(vsi_x + 16, ycur), GOOD if clamp_fpm >= 0.0 else DANGER, 2.5, true)
	draw_rect(Rect2(vsi_x - 20, vy - 60, 40, 24), BG)
	draw_string(ThemeDB.fallback_font, Vector2(vsi_x - 12, vy - 43), "%+d" % roundi(fpm), HORIZONTAL_ALIGNMENT_LEFT, 60, 12, WHITE)

# =========================================================================== #
# Heading tape across the top
# =========================================================================== #
func _draw_heading(c: Vector2, w: float) -> void:
	var hdg := fposmod(_t[13], 360.0)
	var px := w / 240.0                  # px per degree
	var box := Rect2(c.x - 44, c.y - 14, 88, 28)
	var y := c.y + 2.0
	var cardinals := {0: "N", 90: "E", 180: "S", 270: "W"}

	for delta in range(-120, 121, 5):
		var deg := fposmod(hdg + delta, 360.0)
		var x := c.x + delta * px
		if x < 60 or x > w - 60:
			continue
		var major := int(roundf(deg)) % 30 == 0
		draw_line(Vector2(x, y - 8), Vector2(x, y - (16.0 if major else 11.0)), WHITE if major else DIM, 2.0 if major else 1.0, true)
		if major:
			var dv := int(roundf(deg))
			if cardinals.has(dv % 360):
				draw_string(ThemeDB.fallback_font, Vector2(x - 16, y - 20), cardinals[dv % 360], HORIZONTAL_ALIGNMENT_CENTER, 32, 15, ACCENT)
			else:
				draw_string(ThemeDB.fallback_font, Vector2(x - 16, y - 20), str(int(roundf(deg)) / 10), HORIZONTAL_ALIGNMENT_CENTER, 32, 13, WHITE)

	draw_rect(box, BG)
	draw_rect(box, ACCENT, false, 1.5)
	draw_string(ThemeDB.fallback_font, Vector2(box.position.x + 44, box.position.y + 19), "%03d" % int(roundf(hdg)), HORIZONTAL_ALIGNMENT_CENTER, 0, 20, WHITE)
	draw_line(Vector2(c.x, c.y - 16), Vector2(c.x, c.y - 8), ACCENT, 2.0, true)

# =========================================================================== #
# Corner chips: mode, autopilot, stall, engine, battery
# =========================================================================== #
func _draw_status(w: float, h: float) -> void:
	var chips: Array = []
	if _avionics:
		chips.append(["AVIONICS", ACCENT])
	else:
		chips.append(["MANUAL", WARN])
	if _auto_level:
		chips.append(["AP ON", GOOD])
	if _t[23] > 0.5:
		chips.append(["STALL", DANGER])
	if _engine_out != 0:
		chips.append(["ENG OUT", WARN])

	var x := 14.0
	var y := 12.0
	for chip: Array in chips:
		var name: String = chip[0]
		var color: Color = chip[1]
		var sw := ThemeDB.fallback_font.get_string_size(name, HORIZONTAL_ALIGNMENT_LEFT, -1, 13).x + 18.0
		var r := Rect2(x, y, sw, 20)
		draw_rect(r, BG)
		draw_rect(r, color, false, 1.0)
		draw_string(ThemeDB.fallback_font, Vector2(x + 9, y + 15), name, HORIZONTAL_ALIGNMENT_LEFT, -1, 13, color)
		x += sw + 6.0

	if _snap.size() >= 31:
		var vb := _snap[28]
		var pct := _snap[30]
		var bc := GOOD if pct > 50.0 else (WARN if pct > 20.0 else DANGER)
		var btxt := "BATT %.1fV %d%%" % [vb, roundi(pct)]
		var br := Rect2(14, y + 26, ThemeDB.fallback_font.get_string_size(btxt, HORIZONTAL_ALIGNMENT_LEFT, -1, 13).x + 14, 20)
		draw_rect(br, BG)
		draw_rect(br, bc, false, 1.0)
		draw_string(ThemeDB.fallback_font, Vector2(21, y + 41), btxt, HORIZONTAL_ALIGNMENT_LEFT, -1, 13, bc)

	# Wind + AoA + TAS cluster (bottom-left).
	var tas_text := "TAS %d" % roundi(_t[3])
	var aoa_text := "AoA %.1f" % _t[9]
	var wind_text := "WND %d°/%dkt" % [roundi(_t[22]), roundi(_t[21])]
	var bx := 14.0
	var by := h - 64.0
	for lbl: String in [tas_text, aoa_text, wind_text]:
		var lw := ThemeDB.fallback_font.get_string_size(lbl, HORIZONTAL_ALIGNMENT_LEFT, -1, 13).x + 14.0
		var rr := Rect2(bx, by, lw, 20)
		draw_rect(rr, FADE)
		draw_string(ThemeDB.fallback_font, Vector2(bx + 7, by + 15), lbl, HORIZONTAL_ALIGNMENT_LEFT, -1, 13, WHITE)
		bx += lw + 6.0

# =========================================================================== #
# Throttle / engine strip at the bottom right
# =========================================================================== #
func _draw_engine_strip(w: float, h: float) -> void:
	var thr := _t[15]
	var x := w - 260.0
	var y := h - 44.0
	var bar := Rect2(x + 40, y - 6, 150, 12)
	draw_string(ThemeDB.fallback_font, Vector2(x, y + 3), "THR", HORIZONTAL_ALIGNMENT_LEFT, 60, 12, DIM)
	draw_rect(bar, BG)
	draw_rect(Rect2(bar.position + Vector2(2, 2), Vector2((bar.size.x - 4) * clampf(thr / 100.0, 0.0, 1.0), bar.size.y - 4)), GOOD if thr > 25.0 else WARN)
	draw_rect(bar, WHITE, false, 1.0)
	draw_string(ThemeDB.fallback_font, Vector2(bar.end.x + 8, y + 3), "%d%%" % roundi(thr), HORIZONTAL_ALIGNMENT_LEFT, 60, 13, WHITE)

# =========================================================================== #
# Compact controls legend (bottom centre)
# =========================================================================== #
func _draw_legend(w: float, h: float) -> void:
	var hints := "Pitch W/S  ·  Roll A/D  ·  Rudder Q/E  ·  Flaps F  ·  Engine G  ·  Trim [ / ]  ·  Throttle Shift/Ctrl"
	var hint2 := "Avionics L  ·  Panel P  ·  Autopilot H/T  ·  Reset R  ·  Camera V"
	draw_string(ThemeDB.fallback_font, Vector2(w * 0.5, h - 20.0), hints, HORIZONTAL_ALIGNMENT_CENTER, w, 11, DIM)
	draw_string(ThemeDB.fallback_font, Vector2(w * 0.5, h - 6.0), hint2, HORIZONTAL_ALIGNMENT_CENTER, w, 11, DIM)

# Helper: rotate `p` by `rad` around `center` (screen coords, +y down).
func _rot(center: Vector2, p: Vector2, rad: float) -> Vector2:
	var d := p - center
	var c := cos(rad)
	var s := sin(rad)
	return center + Vector2(d.x * c - d.y * s, d.x * s + d.y * c)