class_name TerrainGenerator
extends Node3D
## Infinite chunked procedural terrain for the flight sim — the Godot twin of
## Sebastian Lague's `MapGenerator` + `EndlessTerrain` (E04/E08).
##
## A grid of square chunks follows the aircraft. Each chunk samples the same
## global FBM Perlin heightmap at exact world coordinates, colours the vertices
## by height region bands (deep water → shallow water → sand → grass → forest →
## rock → snow) and builds flat-shaded meshes at four LOD tiers by distance.
## The terrain also flattens a runway apron around the world origin, and a
## coarse height grid is pushed to the Rust `FlightSimNode` so collision, ground
## effect and orographic wind all follow the generated mountains.

const NoiseScript := preload("res://scripts/terrain/noise.gd")
const ChunkScript := preload("res://scripts/terrain/terrain_chunk.gd")

# ---- World size -------------------------------------------------------
@export_range(100.0, 2000.0, 10.0) var chunk_size := 500.0:
	set(v):
		chunk_size = maxf(v, 50.0)
		_schedule_rebuild()
@export_range(4, 64, 1) var resolution := 33:
	set(v):
		resolution = clampi(v, 4, 64)
		_schedule_rebuild()
@export_range(2, 16, 1) var view_chunks := 12:
	set(v):
		view_chunks = clampi(v, 2, 16)
		_schedule_rebuild()
@export var lod_rings := [2, 4, 7]:
	set(v):
		lod_rings = v
		_schedule_rebuild()

# ---- Height mapping ---------------------------------------------------
@export_range(50.0, 3000.0, 10.0) var height_multiplier := 700.0:
	set(v):
		height_multiplier = maxf(v, 1.0)
		_schedule_rebuild()
@export_range(0.0, 2.0, 0.05) var height_exponent := 1.25:
	set(v):
		height_exponent = maxf(v, 0.1)
		_schedule_rebuild()
@export var regions: Array = []:
	set(v):
		regions = v if not v.is_empty() else _default_regions()
		_schedule_rebuild()

# ---- Noise ------------------------------------------------------------
@export_range(0, 100000, 1) var noise_seed := 1337:
	set(v):
		noise_seed = v
		_recreate_noise()
@export_range(10.0, 3000.0, 5.0) var noise_scale := 900.0:
	set(v):
		noise_scale = maxf(v, 10.0)
		_recreate_noise()
@export_range(1, 8, 1) var octaves := 5:
	set(v):
		octaves = maxi(v, 1)
		_recreate_noise()
@export_range(0.0, 1.0, 0.05) var persistence := 0.4:
	set(v):
		persistence = v
		_recreate_noise()
@export_range(1.0, 4.0, 0.1) var lacunarity := 2.0:
	set(v):
		lacunarity = maxf(v, 1.0)
		_recreate_noise()
@export var noise_offset := Vector2.ZERO:
	set(v):
		noise_offset = v
		if _noise != null:
			_noise.offset = v
		_schedule_rebuild()

# ---- Airfield apron (keeps the spawn/runway area flat) ----------------
@export var runway_zone := Vector2(1800.0, 600.0):
	set(v):
		runway_zone = v
		_schedule_rebuild()
@export_range(50.0, 2000.0, 10.0) var flatten_fade := 350.0:
	set(v):
		flatten_fade = maxf(v, 10.0)
		_schedule_rebuild()

# ---- Collision / physics bridge ---------------------------------------
@export var collision_enabled := true
@export var physics_grid_enabled := true:
	set(v):
		physics_grid_enabled = v
		if not v:
			_grid_fill = -1
@export_range(512.0, 16384.0, 64.0) var physics_half_extent := 6144.0
@export_range(16.0, 256.0, 8.0) var physics_spacing := 48.0

## `Callable` expecting `(north0, east0, spacing, nx, nz, heights)` — wired to
## `FlightSimNode.configure_terrain` by the flight sim.
var physics_grid_callback: Callable

var target: Node3D = null

var _noise = null
var _material: StandardMaterial3D = null
var _chunks := {}
var _queued := {}
var _pending: Array = []
var _rebuild_pending := false
var _grid_center := Vector2.ZERO
var _grid_has_center := false
var _grid_n := 0
var _grid_fill := -1
var _grid_heights := PackedFloat64Array()

func _ready() -> void:
	if _material == null:
		_material = StandardMaterial3D.new()
		_material.vertex_color_use_as_albedo = true
		_material.roughness = 1.0
	if regions.is_empty():
		regions = _default_regions()
	if _noise == null:
		_recreate_noise()

func _process(_delta: float) -> void:
	if Engine.is_editor_hint():
		return
	if not is_inside_tree():
		return
	if target == null:
		return
	if _rebuild_pending:
		_rebuild_pending = false
		_rebuild_all()
	_update_world()
	_drain_pending()
	_update_physics_grid()

## Point the terrain world at an object (the aircraft) to follow.
func set_target_node(n: Node3D) -> void:
	target = n
	_grid_has_center = false

## Forget everything and rebuild from the current noise/height settings.
func regenerate(new_seed: int = -1) -> void:
	if new_seed >= 0:
		noise_seed = new_seed
	else:
		noise_seed += 1
	_rebuild_all()

func chunk_count() -> int:
	return _chunks.size()

func pending_count() -> int:
	return _pending.size()

# ----------------------------------------------------------------------- #
#  Sampling (single source of truth shared by chunks and the physics grid) #
# ----------------------------------------------------------------------- #

## Surface altitude (m) above the reference datum at world (x, z).
func sample_height(x: float, z: float) -> float:
	var y := sample_height_raw(x, z)
	var flat := _flatten_amount(x, z)
	return lerpf(y, 0.0, flat)

## Altitude before the runway apron is carved out.
func sample_height_raw(x: float, z: float) -> float:
	if _noise == null:
		return 0.0
	var t: float = _noise.get_height(x, z)
	if absf(height_exponent - 1.0) > 0.001:
		t = pow(t, height_exponent)
	return t * height_multiplier

## Height-region colour for a world position (matches E04's `colourMap`).
func sample_colour(x: float, z: float) -> Color:
	if _noise == null:
		return Color.WHITE
	var t: float = _noise.get_height(x, z)
	for region: Dictionary in regions:
		if t <= float(region.height):
			return region.colour
	if not regions.is_empty():
		return regions[-1].colour
	return Color.WHITE

## 0 outside the apron, 1 fully inside — used to blend heights to flat ground.
func _flatten_amount(x: float, z: float) -> float:
	var fx := clampf((runway_zone.x - absf(x)) / flatten_fade, 0.0, 1.0)
	var fz := clampf((runway_zone.y - absf(z)) / flatten_fade, 0.0, 1.0)
	return minf(fx, fz)

func _default_regions() -> Array:
	return [
		{"height": 0.40, "colour": Color(0.09, 0.35, 0.52)},
		{"height": 0.45, "colour": Color(0.20, 0.55, 0.70)},
		{"height": 0.51, "colour": Color(0.79, 0.71, 0.44)},
		{"height": 0.58, "colour": Color(0.26, 0.55, 0.24)},
		{"height": 0.72, "colour": Color(0.18, 0.38, 0.18)},
		{"height": 0.86, "colour": Color(0.46, 0.43, 0.39)},
		{"height": 1.00, "colour": Color(0.94, 0.94, 0.95)},
	]

# ----------------------------------------------------------------------- #
#  Chunk management                                                       #
# ----------------------------------------------------------------------- #

func _recreate_noise() -> void:
	_noise = NoiseScript.new(noise_seed, noise_scale, octaves, persistence, lacunarity)
	_noise.offset = noise_offset
	_schedule_rebuild()

func _schedule_rebuild() -> void:
	_rebuild_pending = true

func _rebuild_all() -> void:
	for key: String in _chunks:
		var c = _chunks[key]
		if is_instance_valid(c):
			c.queue_free()
	_chunks.clear()
	_queued.clear()
	_pending.clear()
	_grid_fill = -1
	_grid_has_center = false

func _target_position() -> Vector3:
	if target == null:
		return Vector3.ZERO
	if target.is_inside_tree():
		return target.global_position
	return target.position

func _update_world() -> void:
	if target == null:
		return
	var p := _target_position()
	var tcell := Vector2i(floori(p.x / chunk_size), floori(p.z / chunk_size))
	var desired := {}
	for dz in range(-view_chunks, view_chunks + 1):
		for dx in range(-view_chunks, view_chunks + 1):
			var cell := Vector2i(tcell.x + dx, tcell.y + dz)
			var dist := maxi(absi(dx), absi(dz))
			desired[_key(cell)] = _lod_for(dist)

	for key: String in _chunks.keys():
		if not desired.has(key):
			var c = _chunks[key]
			_chunks.erase(key)
			_queued.erase(key)
			if is_instance_valid(c):
				c.queue_free()

	var rebuilds: Array = []
	for key: String in desired:
		var want_lod: int = desired[key]
		if _chunks.has(key):
			var existing = _chunks[key]
			if existing.lod != want_lod:
				rebuilds.append({"node": existing, "lod": want_lod})
			elif collision_enabled and want_lod <= 1 and not is_instance_valid(existing.collision_body):
				# Near chunk still missing its body (e.g. spawned before
				# collision was toggled on) — rebuild to add it.
				rebuilds.append({"node": existing, "lod": want_lod})
			continue
		if not _queued.has(key):
			_queued[key] = true
			_pending.append({"key": key, "cell": _cell_from_key(key), "lod": want_lod})
	# Throttle synchronous LOD rebuilds: rebuilding is a full resample, so
	# doing the whole ring in one frame stalls streaming (visible as holes).
	# Nearest-first, bounded per frame; the rest follow in later frames.
	if not rebuilds.is_empty():
		var tpos := _target_position()
		rebuilds.sort_custom(func(a: Dictionary, b: Dictionary) -> bool:
			var na: Node3D = a.node
			var nb: Node3D = b.node
			return na.global_position.distance_squared_to(tpos) < nb.global_position.distance_squared_to(tpos))
		var n := mini(rebuilds.size(), 8)
		for i in n:
			var entry: Dictionary = rebuilds[i]
			var node = entry.node
			var lod: int = entry.lod
			node.lod = lod
			node.collision_enabled = collision_enabled and lod <= 1
			node.build()

func _drain_pending() -> void:
	if _pending.is_empty():
		return
	var p := _target_position()
	var tcell := Vector2i(floori(p.x / chunk_size), floori(p.z / chunk_size))
	# Pop the nearest pending cell first so the ground right around the
	# aircraft fills in immediately and far rings stream in behind it.
	# Budget by mesh cost: near LOD-0 chunks are 33×33, far LOD-3 chunks only
	# ~6×6, so far rings stream dozens of chunks per frame while near ones
	# still land smoothly.
	var vert_budget := 15000
	var max_chunks := 240
	var built := 0
	while not _pending.is_empty() and vert_budget > 0 and built < max_chunks:
		var best_i := 0
		var best_d := 0x7FFFFFFF
		for i in _pending.size():
			var c: Vector2i = _pending[i].cell
			var d := maxi(absi(c.x - tcell.x), absi(c.y - tcell.y))
			if d < best_d:
				best_d = d
				best_i = i
			if best_d == 0:
				break
		var task: Dictionary = _pending[best_i]
		_pending.remove_at(best_i)
		var key: String = task.key
		_queued.erase(key)
		var step_len := 1 if int(task.lod) <= 0 else 1 << mini(int(task.lod), 5)
		var side := (resolution - 1) / step_len + 1
		vert_budget -= side * side
		_spawn_chunk(task.cell, int(task.lod))
		built += 1

func _spawn_chunk(cell: Vector2i, lod: int) -> void:
	var key := _key(cell)
	if _chunks.has(key):
		return
	var chunk = ChunkScript.new()
	chunk.name = "Chunk_%d_%d" % [cell.x, cell.y]
	chunk.chunk_size = chunk_size
	chunk.resolution = resolution
	chunk.lod = lod
	chunk.material = _material
	# Only near chunks get concave collision bodies: cooking thousands of
	# far-field shapes stalls the streamer (holes) for geometry the aircraft
	# can never touch. LOD promotion rebuilds the chunk with collision.
	chunk.collision_enabled = collision_enabled and lod <= 1
	chunk.sample_fn = Callable(self, "sample_height")
	chunk.colour_fn = Callable(self, "sample_colour")
	chunk.world_cell = Vector2(cell.x * chunk_size, cell.y * chunk_size)
	add_child(chunk)
	chunk.global_position = Vector3(cell.x * chunk_size, 0.0, cell.y * chunk_size)
	chunk.build()
	_chunks[key] = chunk

func _lod_for(dist: int) -> int:
	if lod_rings.is_empty():
		return 0
	var lod := 0
	for ring: int in lod_rings:
		if dist >= int(ring):
			lod += 1
	return clampi(lod, 0, 3)

func _key(cell: Vector2i) -> String:
	return "%d:%d" % [cell.x, cell.y]

func _cell_from_key(key: String) -> Vector2i:
	var parts := key.split(":")
	return Vector2i(int(parts[0]), int(parts[1]))

# ----------------------------------------------------------------------- #
#  Physics bridge — coarse height grid pushed to the Rust FlightSimNode    #
# ----------------------------------------------------------------------- #

func _update_physics_grid() -> void:
	if not physics_grid_enabled:
		return
	if physics_grid_callback.is_null() or not physics_grid_callback.is_valid():
		return
	if target == null:
		return
	var p := _target_position()
	var c := Vector2(p.x, p.z)

	if _grid_fill == -1:
		# Hysteresis: only recenter once the aircraft has left the inner
		# quarter of the grid. The old threshold (2 cells) retriggered
		# almost continuously at flight speeds, spamming configure_terrain.
		var recenter_dist := maxf(physics_spacing * 2.0, physics_half_extent * 0.25)
		if _grid_has_center and c.distance_to(_grid_center) < recenter_dist:
			return
		_grid_has_center = true
		_grid_center = Vector2(
			floorf(c.x / physics_spacing) * physics_spacing,
			floorf(c.y / physics_spacing) * physics_spacing
		)
		_grid_n = int(physics_half_extent * 2.0 / physics_spacing) + 1
		_grid_heights = PackedFloat64Array()
		_grid_heights.resize(_grid_n * _grid_n)
		_grid_fill = 0
		return

	var north0 := _grid_center.x - physics_half_extent
	var east0 := _grid_center.y - physics_half_extent
	var budget := 2048
	while _grid_fill < _grid_n * _grid_n and budget > 0:
		var x := _grid_fill % _grid_n
		var z := _grid_fill / _grid_n
		_grid_heights[z * _grid_n + x] = sample_height(north0 + x * physics_spacing, east0 + z * physics_spacing)
		_grid_fill += 1
		budget -= 1
	if _grid_fill >= _grid_n * _grid_n:
		physics_grid_callback.call(north0, east0, physics_spacing, _grid_n, _grid_n, _grid_heights)
		_grid_fill = -1