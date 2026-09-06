@tool
extends Node3D
## Procedural volumetric cloud pockets. Scatters small box-shaped FogVolumes
## around the camera in a deterministic cell grid so the player flies between
## and through individual, realistic clouds. Requires the Forward+ renderer
## (volumetric fog is not supported by the Compatibility renderer).

# ---- Cloud layer --------------------------------------------------------
@export_range(0.0, 3000.0, 1.0) var cloud_base_alt := 150.0:
	set(v):
		cloud_base_alt = v
		_schedule_rebuild()
@export_range(0.0, 2500.0, 10.0) var alt_span := 550.0:
	set(v):
		alt_span = v
		_schedule_rebuild()
@export_range(0.001, 0.3, 0.001) var density := 0.12:
	set(v):
		density = v
		_schedule_rebuild()
@export var cloud_albedo := Color(0.94, 0.94, 0.97):
	set(v):
		cloud_albedo = v
		_schedule_rebuild()
@export_range(0.0, 5.0, 0.1) var edge_fade := 0.35:
	set(v):
		edge_fade = v
		_schedule_rebuild()

# ---- Pocket field --------------------------------------------------------
@export_range(50.0, 800.0, 5.0) var pocket_size := 380.0:
	set(v):
		pocket_size = v
		_schedule_rebuild()
@export_range(1, 20, 1) var pockets_per_cell := 6:
	set(v):
		pockets_per_cell = v
		_schedule_rebuild()
@export_range(300.0, 3000.0, 50.0) var cell_size := 1300.0:
	set(v):
		cell_size = v
		_schedule_rebuild()
@export_range(0, 3, 1) var view_cells := 1:
	set(v):
		view_cells = v
		_schedule_rebuild()
@export_range(4, 24, 1) var texture_pool_count := 12:
	set(v):
		texture_pool_count = v
		_pool.clear()
		_schedule_rebuild()

# ---- Shape noise --------------------------------------------------------
@export_range(0, 100000) var noise_seed := 1337:
	set(v):
		noise_seed = v
		_pool.clear()
		_schedule_rebuild()
@export_range(0.5, 40.0, 0.05) var noise_frequency := 4.0:
	set(v):
		noise_frequency = v
		_pool.clear()
		_schedule_rebuild()
@export_range(1, 8, 1) var fractal_octaves := 4:
	set(v):
		fractal_octaves = v
		_pool.clear()
		_schedule_rebuild()
@export_range(0.0, 1.0, 0.01) var fractal_gain := 0.55:
	set(v):
		fractal_gain = v
		_pool.clear()
		_schedule_rebuild()
@export_range(1.0, 4.0, 0.1) var fractal_lacunarity := 2.0:
	set(v):
		fractal_lacunarity = v
		_pool.clear()
		_schedule_rebuild()
@export_range(0.0, 1.0, 0.01) var weighted_strength := 0.35:
	set(v):
		weighted_strength = v
		_pool.clear()
		_schedule_rebuild()
@export_range(0.0, 1.0, 0.01) var density_floor := 0.38:
	set(v):
		density_floor = v
		_pool.clear()
		_schedule_rebuild()
@export_range(0.0, 1.0, 0.01) var density_peak := 0.76:
	set(v):
		density_peak = v
		_pool.clear()
		_schedule_rebuild()

# ---- Motion ------------------------------------------------------------
@export_range(0.0, 20.0, 0.1) var drift_speed := 2.0:
	set(v):
		drift_speed = v
@export var follow_player := true:
	set(v):
		follow_player = v
		_schedule_rebuild()

# ---- Lighting ----------------------------------------------------------
@export_range(0.0, 4.0, 0.05) var sun_volumetric_energy := 1.0:
	set(v):
		sun_volumetric_energy = v
		_reconfigure_sun()

var _wind_group: Node3D
var _pockets: Array[FogVolume] = []
var _pool: Array[Texture3D] = []
var _rebuild_pending := false
var _cam_cell := Vector3i(-1000000, 0, -1000000)
var _drift := 0.0

const NOISE_TEX_RES := 64
const FALL_EDGE0 := 0.62
const FALL_EDGE1 := 1.05

func _ready() -> void:
	if RenderingServer.get_current_rendering_driver_name() == "dummy":
		return
	if not is_node_ready():
		return
	_wind_group = Node3D.new()
	_wind_group.name = "Wind"
	add_child(_wind_group)
	if not Engine.is_editor_hint():
		_configure_environment()
		_configure_sun()
	_rebuild()

func _process(delta: float) -> void:
	if _rebuild_pending:
		_rebuild_pending = false
		_rebuild()
		return
	if Engine.is_editor_hint():
		return
	if not is_inside_tree():
		return
	if not follow_player:
		return
	var cam := _get_camera()
	if cam == null:
		return
	_drift += delta * drift_speed
	_wind_group.position = Vector3(_drift, 0.0, _drift * 0.35)
	var cell := Vector3i(
		floori(cam.global_position.x / cell_size),
		0,
		floori(cam.global_position.z / cell_size)
	)
	if cell != _cam_cell:
		_cam_cell = cell
		_rebuild()

func _schedule_rebuild() -> void:
	_rebuild_pending = true

func _rebuild() -> void:
	for p in _pockets:
		if is_instance_valid(p):
			p.queue_free()
	_pockets.clear()
	if _wind_group == null or not is_node_ready():
		return
	if _cam_cell.x == -1000000:
		var cam := _get_camera()
		if cam != null and follow_player:
			_cam_cell = Vector3i(
				floori(cam.global_position.x / cell_size),
				0,
				floori(cam.global_position.z / cell_size)
			)
		else:
			_cam_cell = Vector3i.ZERO
	_ensure_pool()
	var cx := _cam_cell.x
	var cz := _cam_cell.z
	for dx in range(-view_cells, view_cells + 1):
		for dz in range(-view_cells, view_cells + 1):
			_spawn_cell(cx + dx, cz + dz)
	_update_wind_group()

func _spawn_cell(cx: int, cz: int) -> void:
	for i in pockets_per_cell:
		var rng := RandomNumberGenerator.new()
		rng.seed = _hash_ints(noise_seed, _hash_ints(cx, cz, i), 0x1234567)
		var jx := (rng.randf() - 0.5) * cell_size * 0.85
		var jz := (rng.randf() - 0.5) * cell_size * 0.85
		var hy := cloud_base_alt + rng.randf() * alt_span
		var base := pocket_size * (0.6 + 0.8 * rng.randf())
		var w := base * (0.7 + 0.6 * rng.randf())
		var h := base * (0.3 + 0.45 * rng.randf())
		var d := base * (0.7 + 0.9 * rng.randf())
		var rot := rng.randf() * TAU
		var tilt_x := (rng.randf() - 0.5) * 0.5
		var tilt_z := (rng.randf() - 0.5) * 0.5
		var pool_idx := (rng.randi() % _pool.size()) if _pool.size() > 0 else 0
		var f := FogVolume.new()
		f.name = "Cloud_%d_%d_%d" % [cx, cz, i]
		f.shape = RenderingServer.FOG_VOLUME_SHAPE_BOX
		f.size = Vector3(w, h, d)
		f.rotation = Vector3(tilt_x, rot, tilt_z)
		f.position = Vector3(cx * cell_size + jx, hy, cz * cell_size + jz)
		f.material = _make_pocket_material(_pool[pool_idx], rng)
		_wind_group.add_child(f)
		_pockets.append(f)

func _hash_ints(a: int, b: int, c: int) -> int:
	var h := a & 0xFFFFFFFF
	h = (h * 0x9E3779B1) & 0xFFFFFFFF
	h ^= (b * 0x85EBCA6B) & 0xFFFFFFFF
	h = (((h ^ (h >> 15)) & 0xFFFFFFFF) + (c * 0xC2B2AE35)) & 0xFFFFFFFF
	h ^= h >> 13
	h = (h * 0x27D4EB2F) & 0xFFFFFFFF
	h ^= h >> 16
	return h & 0xFFFFFFFF

func _ensure_pool() -> void:
	if _pool.size() == texture_pool_count:
		return
	_pool.clear()
	for i in texture_pool_count:
		var noise := FastNoiseLite.new()
		noise.noise_type = FastNoiseLite.TYPE_PERLIN
		noise.fractal_type = FastNoiseLite.FRACTAL_FBM
		noise.seed = noise_seed + i * 104729
		noise.frequency = noise_frequency
		noise.fractal_octaves = fractal_octaves
		noise.fractal_gain = fractal_gain
		noise.fractal_lacunarity = fractal_lacunarity
		noise.fractal_weighted_strength = weighted_strength
		_pool.append(_make_density_texture(noise))

func _make_density_texture(noise: FastNoiseLite) -> Texture3D:
	var slices: Array[Image] = []
	slices.resize(NOISE_TEX_RES)
	var step := noise_frequency / float(NOISE_TEX_RES - 1)
	for z in NOISE_TEX_RES:
		var px := PackedByteArray()
		px.resize(NOISE_TEX_RES * NOISE_TEX_RES * 4)
		for y in NOISE_TEX_RES:
			for x in NOISE_TEX_RES:
				var n := noise.get_noise_3d(x * step, y * step, -z * step)
				var d0 := _noise_to_density(n)
				var nx := x / float(NOISE_TEX_RES - 1) * 2.0 - 1.0
				var ny := y / float(NOISE_TEX_RES - 1) * 2.0 - 1.0
				var nz := z / float(NOISE_TEX_RES - 1) * 2.0 - 1.0
				var r := sqrt(nx * nx + ny * ny + nz * nz)
				var fall := clampf(1.0 - smoothstep(FALL_EDGE0, FALL_EDGE1, r), 0.0, 1.0)
				var d := clampf(d0 * (0.25 + 1.6 * fall), 0.0, 1.0)
				var off := (y * NOISE_TEX_RES + x) * 4
				var v := int(clampf(d, 0.0, 1.0) * 255.0)
				px[off] = v
				px[off + 1] = v
				px[off + 2] = v
				px[off + 3] = 255
		slices[z] = Image.create_from_data(NOISE_TEX_RES, NOISE_TEX_RES, false, Image.FORMAT_RGBA8, px)
	var tex := ImageTexture3D.new()
	tex.create(Image.FORMAT_RGBA8, NOISE_TEX_RES, NOISE_TEX_RES, NOISE_TEX_RES, false, slices)
	return tex

func _noise_to_density(v: float) -> float:
	var t := clampf(v * 0.5 + 0.5, 0.0, 1.0)
	if t <= density_floor:
		return 0.0
	var span := maxf(1.0 - density_floor, 0.001)
	var a := clampf((t - density_floor) / span, 0.0, 1.0)
	var d := a * a * (3.0 - 2.0 * a)
	var top := clampf((t - density_peak) / maxf(1.0 - density_peak, 0.001), 0.0, 1.0)
	return clampf(d * (1.0 - 0.35 * top * top * (3.0 - 2.0 * top)), 0.0, 1.0)

func _make_pocket_material(tex: Texture3D, rng: RandomNumberGenerator) -> FogMaterial:
	var m := FogMaterial.new()
	m.density = density * (0.7 + 0.6 * rng.randf())
	m.albedo = cloud_albedo * (0.82 + 0.35 * rng.randf())
	m.emission = m.albedo * (0.22 + 0.18 * rng.randf())
	m.edge_fade = edge_fade
	m.density_texture = tex
	return m

func _update_wind_group() -> void:
	if _wind_group != null:
		_wind_group.position = Vector3(_drift, 0.0, _drift * 0.35)

func _get_camera() -> Camera3D:
	var viewport := get_viewport()
	if viewport == null:
		return null
	return viewport.get_camera_3d()

func _configure_environment() -> void:
	var we := _find_node(&"WorldEnvironment") as WorldEnvironment
	if we == null:
		return
	var env: Environment = we.environment
	env.volumetric_fog_enabled = true
	env.volumetric_fog_density = 0.0
	env.volumetric_fog_anisotropy = 0.45
	env.volumetric_fog_ambient_inject = maxf(env.volumetric_fog_ambient_inject, 0.8)
	env.volumetric_fog_length = maxf(env.volumetric_fog_length, 4000.0)
	env.volumetric_fog_sky_affect = 1.0

func _reconfigure_sun() -> void:
	if Engine.is_editor_hint():
		return
	var sun := _find_node(&"Sun") as DirectionalLight3D
	if sun != null:
		sun.light_volumetric_fog_energy = sun_volumetric_energy

func _configure_sun() -> void:
	_reconfigure_sun()

func _find_node(type_name: StringName) -> Node:
	var root := get_tree()
	if root == null:
		return null
	var scene := root.current_scene
	if scene == null:
		return null
	var stack: Array[Node] = [scene]
	while stack.size() > 0:
		var n: Node = stack.pop_back()
		if n.is_class(type_name):
			return n
		for child in n.get_children():
			stack.push_back(child)
	return null