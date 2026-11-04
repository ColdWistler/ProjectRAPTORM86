extends RefCounted
## Live reader for a training run's frame ring.
##
## The `rl_agent` trainer writes every environment step to `frames.bin`, a
## fixed-size ring of 124-byte `Frame` records, and publishes the write cursor
## to `stream_head.bin` (a little-endian u64). This class polls the cursor and
## pulls down only the frames that are new, so a viewer stays current without
## ever re-reading the whole ring.
##
## Nothing here blocks the trainer. The ring is written by a dedicated thread
## behind a lossy bounded channel, so a slow or absent viewer costs the trainer
## dropped frames and nothing else — and this class reports those drops rather
## than letting a frozen trail look like an aircraft that stopped moving.
##
## Layout note: the offsets below mirror `rl_agent::stream::Frame`. That
## mirror is checked against the writer's own published `stream.json`
## ([method attach]) instead of being trusted, so a layout change surfaces as
## an error rather than as an aircraft in a physically impossible attitude.

const FRAMES_FILE := "frames.bin"
const HEAD_FILE := "stream_head.bin"
const MANIFEST_FILE := "stream.json"

## Schema this reader understands. Must match `rl_agent::stream::STREAM_SCHEMA`.
const EXPECTED_SCHEMA := "rl_agent.stream.v1"

## Bytes per frame record.
const FRAME_BYTES := 124

## Number of slots in the ring. Must be a power of two, as the writer relies on
## masking rather than a modulo for the wrap.
const RING_SLOTS := 65536

## Frame field offsets, matching `FRAME_FIELD_OFFSETS` on the Rust side.
const O_ITERATION := 0
const O_STEP := 4
const O_EPISODE := 8
const O_WORKER := 12
const O_FLAGS := 14
const O_SIM_TIME := 16
const O_NORTH := 20
const O_EAST := 24
const O_ALT := 28
const O_VEL_N := 32
const O_VEL_E := 36
const O_VEL_D := 40
const O_ROLL := 44
const O_PITCH := 48
const O_YAW := 52
const O_P := 56
const O_Q := 60
const O_R := 64
const O_AIRSPEED := 68
const O_ALPHA := 72
const O_BETA := 76
const O_ELEVATOR := 96
const O_AILERON := 100
const O_RUDDER := 104
const O_THROTTLE := 108
const O_BARO_ALT := 112
const O_IAS := 116
const O_REWARD := 120

## Trail points retained for drawing. Each is a Dictionary of the handful of
## fields the view needs, so memory stays flat regardless of run length.
const MAX_TRAIL := 6000

## Ceiling on frames pulled in a single poll. A viewer attached mid-run, or one
## whose last poll was delayed, would otherwise try to catch up on the whole
## ring at once and stall the render loop. Anything older is skipped and
## counted — the honest outcome, since those frames are simply not shown.
const MAX_CATCHUP := 1500

## Ring file handle, opened once and seeked per poll.
var _ring: FileAccess = null
var _head_path := ""
var _ring_path := ""

## Write cursor of the last frame this reader has consumed. Frames before this
## are either already drawn or counted as skipped.
var head := 0

## Newest decoded frame, or `{}` before anything has been read.
var latest: Dictionary = {}

## Trailing flight path, oldest first. Each entry:
## `{n, e, alt, yaw, ias, t}` in metres, radians and m/s.
var trail: Array = []

## Frames the ring advanced past that this reader never decoded. Raised both by
## the trainer's own lossy channel and by this reader falling behind, so it is
## the single number that answers "did I miss anything".
var skipped := 0

## Why attach/poll stopped working, or `""` when healthy.
var error := ""

var _attached := false


## Point this reader at a run directory and validate its layout.
##
## Returns `false` when the directory has no stream artifacts, or when they
## describe a layout this reader cannot parse. Refusing here is deliberate: a
## mis-decoded frame produces plausible-looking numbers, so failing loudly at
## attach time is far cheaper than debugging an aircraft flying sideways.
func attach(run_dir: String) -> bool:
	detach()
	error = ""
	if run_dir.is_empty() or not DirAccess.dir_exists_absolute(run_dir):
		error = "no such run directory"
		return false

	var manifest_path := run_dir.path_join(MANIFEST_FILE)
	if not FileAccess.file_exists(manifest_path):
		error = "no %s — this run was started without frame streaming" % MANIFEST_FILE
		return false

	var file := FileAccess.open(manifest_path, FileAccess.READ)
	if file == null:
		error = "cannot read %s" % MANIFEST_FILE
		return false
	var parsed: Variant = JSON.parse_string(file.get_as_text())
	file.close()
	if typeof(parsed) != TYPE_DICTIONARY:
		error = "%s is not valid JSON" % MANIFEST_FILE
		return false

	var man: Dictionary = parsed
	var schema := str(man.get("schema") if man.has("schema") else "")
	if schema != EXPECTED_SCHEMA:
		error = "layout %s, this viewer reads %s" % [schema, EXPECTED_SCHEMA]
		return false

	var frame_bytes := int(man.get("frame_bytes") if man.has("frame_bytes") else 0)
	if frame_bytes != FRAME_BYTES:
		error = "stride %d, this viewer expects %d" % [frame_bytes, FRAME_BYTES]
		return false

	var ring_slots := int(man.get("ring_slots") if man.has("ring_slots") else 0)
	if ring_slots != RING_SLOTS:
		error = "ring has %d slots, this viewer expects %d" % [ring_slots, RING_SLOTS]
		return false

	# Cross-check the offset table itself. Cheap, and it turns a silent
	# mis-decode into a message naming the field that moved.
	var offsets: Variant = man.get("field_offsets") if man.has("field_offsets") else []
	if typeof(offsets) != TYPE_ARRAY or (offsets as Array).size() != EXPECTED_OFFSETS.size():
		error = "writer published %d field offsets, this viewer has %d" % [
			(offsets as Array).size() if typeof(offsets) == TYPE_ARRAY else -1,
			EXPECTED_OFFSETS.size(),
		]
		return false
	for i in EXPECTED_OFFSETS.size():
		if int((offsets as Array)[i]) != EXPECTED_OFFSETS[i]:
			error = "field %d is at byte %d here, byte %d in this viewer" % [
				i, int((offsets as Array)[i]), EXPECTED_OFFSETS[i]
			]
			return false

	_ring_path = run_dir.path_join(FRAMES_FILE)
	_head_path = run_dir.path_join(HEAD_FILE)
	if not FileAccess.file_exists(_ring_path):
		error = "no %s yet" % FRAMES_FILE
		return false
	_ring = FileAccess.open(_ring_path, FileAccess.READ)
	if _ring == null:
		error = "cannot open %s" % FRAMES_FILE
		return false

	# The ring is preallocated to its full budget, so a short file means the
	# writer is mid-creation or the run predates streaming.
	if _ring.get_length() < RING_SLOTS * FRAME_BYTES:
		error = "ring is %d bytes, expected %d" % [_ring.get_length(), RING_SLOTS * FRAME_BYTES]
		_ring = null
		return false

	_attached = true
	# Start at the current head: a viewer joining an hour-old run should show
	# recent flight, not replay it from the beginning. Those earlier frames are
	# not counted as skipped — nothing was promised, so nothing was lost.
	head = _read_head()
	return true


## Offset of every field, in declaration order. Mirrors
## `FRAME_FIELD_OFFSETS` in `rl_agent::stream`; validated against the writer's
## own copy on attach.
const EXPECTED_OFFSETS := [
	0, 4, 8, 12, 14, 16, 20, 24, 28, 32, 36, 40, 44, 48, 52, 56, 60, 64, 68, 72, 76, 80,
	84, 88, 92, 96, 100, 104, 108, 112, 116, 120,
]


func is_attached() -> bool:
	return _attached and _ring != null


func detach() -> void:
	if _ring != null:
		_ring.close()
	_ring = null
	_attached = false
	head = 0
	latest = {}
	trail.clear()
	skipped = 0
	error = ""


## Pull down whatever the trainer has written since the last poll.
##
## Returns the number of frames decoded. Safe to call every frame; it is a no-op
## when the cursor has not moved.
func poll() -> int:
	if not is_attached():
		return 0
	var cursor := _read_head()
	if cursor <= head:
		return 0

	# A cursor that leaps past the catch-up ceiling is not an error, just more
	# than one render frame can absorb. Skip to the tail and say so.
	var first := head
	if cursor - first > MAX_CATCHUP:
		first = cursor - MAX_CATCHUP
		skipped += first - head
		head = first

	var count := _read_range(first, cursor)
	head = cursor
	return count


## Frames recorded by the trainer, whether or not this reader has seen them.
## The gap against [member head] is how far behind the display is.
func recorded() -> int:
	if not is_attached():
		return 0
	return _read_head()


func _read_head() -> int:
	if _head_path.is_empty() or not FileAccess.file_exists(_head_path):
		return head
	var f := FileAccess.open(_head_path, FileAccess.READ)
	if f == null or f.get_length() < 8:
		if f != null:
			f.close()
		return head
	# The cursor is a little-endian u64 written after the ring bytes it
	# advertises are durable, so it never points at a slot still in a buffer.
	var b := f.get_buffer(8)
	f.close()
	return _decode_u64(b)


func _decode_u64(bytes: PackedByteArray) -> int:
	var v := 0
	# Little-endian, and assembled with shifts rather than decode_u64 so a
	# value above 2^53 degrades to "very large" instead of wrapping negative.
	for i in range(8):
		v += bytes[i] << (8 * i)
	return v


## Decode frames `[first, cursor)` out of the ring, oldest first.
##
## Walks the range in at most two segments because the ring wraps; a contiguous
## `get_buffer` across the wrap point would read the wrong slots entirely.
func _read_range(first: int, cursor: int) -> int:
	var decoded := 0
	var i := first
	while i < cursor:
		var slot := i % RING_SLOTS
		var run := mini(cursor - i, RING_SLOTS - slot)
		_ring.seek(slot * FRAME_BYTES)
		var chunk := _ring.get_buffer(run * FRAME_BYTES)
		if chunk.size() < run * FRAME_BYTES:
			# The file is preallocated, so this means it was truncated out from
			# under us (run directory deleted mid-poll).
			error = "ring shrank while reading"
			return decoded
		for k in range(run):
			var base := k * FRAME_BYTES
			_append(chunk, base)
			decoded += 1
		i += run
	return decoded


func _append(buf: PackedByteArray, base: int) -> void:
	var step := _u32(buf, base + O_STEP)
	var iteration := _u32(buf, base + O_ITERATION)
	var episode := _u32(buf, base + O_EPISODE)
	var flags := _u16(buf, base + O_FLAGS)

	# A jump in (iteration, step) means a frame existed that we did not see —
	# either the trainer's channel dropped it or this reader fell behind.
	# Recording it is what keeps a gapped trail from reading as a stationary
	# aircraft.
	if not trail.is_empty():
		var last: Dictionary = trail[trail.size() - 1]
		if int(last["it"]) == iteration and int(last["step"]) + 1 != step:
			skipped += 1
		elif int(last["it"]) != iteration:
			# A new iteration restarts `step` at 0, which is expected; an
			# episode reset resets sim time, which is also expected. Only a
			# backwards iteration is a genuine gap.
			if iteration < int(last["it"]):
				skipped += 1

	var point := {
		"n": _f32(buf, base + O_NORTH),
		"e": _f32(buf, base + O_EAST),
		"alt": _f32(buf, base + O_ALT),
		"yaw": _f32(buf, base + O_YAW),
		"ias": _f32(buf, base + O_AIRSPEED),
		"t": _f32(buf, base + O_SIM_TIME),
		"it": iteration,
		"step": step,
		"ep": episode,
		"flags": flags,
	}
	trail.append(point)
	if trail.size() > MAX_TRAIL:
		# Slide by slicing rather than pop_front: shifting an Array element by
		# element costs O(n) per frame, and this runs every poll.
		trail = trail.slice(trail.size() - MAX_TRAIL)

	latest = {
		"iteration": iteration,
		"step": step,
		"episode": episode,
		"worker": _u16(buf, base + O_WORKER),
		"terminated": (flags & 1) != 0,
		"truncated": (flags & 2) != 0,
		"sim_time": _f32(buf, base + O_SIM_TIME),
		"north": _f32(buf, base + O_NORTH),
		"east": _f32(buf, base + O_EAST),
		"altitude": _f32(buf, base + O_ALT),
		"vel_north": _f32(buf, base + O_VEL_N),
		"vel_east": _f32(buf, base + O_VEL_E),
		"vel_down": _f32(buf, base + O_VEL_D),
		"roll": _f32(buf, base + O_ROLL),
		"pitch": _f32(buf, base + O_PITCH),
		"yaw": _f32(buf, base + O_YAW),
		"p": _f32(buf, base + O_P),
		"q": _f32(buf, base + O_Q),
		"r": _f32(buf, base + O_R),
		"airspeed": _f32(buf, base + O_AIRSPEED),
		"alpha": _f32(buf, base + O_ALPHA),
		"beta": _f32(buf, base + O_BETA),
		"elevator_deg": _f32(buf, base + O_ELEVATOR),
		"aileron_deg": _f32(buf, base + O_AILERON),
		"rudder_deg": _f32(buf, base + O_RUDDER),
		"throttle": _f32(buf, base + O_THROTTLE),
		"baro_altitude": _f32(buf, base + O_BARO_ALT),
		"airspeed_indicated": _f32(buf, base + O_IAS),
		"reward": _f32(buf, base + O_REWARD),
	}


func _u16(buf: PackedByteArray, off: int) -> int:
	return buf.decode_u16(off)


func _u32(buf: PackedByteArray, off: int) -> int:
	return buf.decode_u32(off)


func _f32(buf: PackedByteArray, off: int) -> float:
	return buf.decode_float(off)
