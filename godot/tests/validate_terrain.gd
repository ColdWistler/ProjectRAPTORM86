extends SceneTree

const TerrainGeneratorScript := preload("res://scripts/terrain/terrain_generator.gd")

var _grid_calls := 0
var _grid_len := 0

func _init() -> void:
	_run.call_deferred()

func _run() -> void:
	var gen = TerrainGeneratorScript.new()
	root.add_child(gen)
	gen._ready()
	gen.collision_enabled = true
	gen.regions = gen._default_regions()
	gen.physics_grid_callback = Callable(self, "_on_grid")
	var drone := Node3D.new()
	root.add_child(drone)
	drone.position = Vector3(0.0, 50.0, 0.0)
	gen.set_target_node(drone)

	gen._update_world()
	var guard := 0
	while gen.pending_count() > 0 and guard < 5000:
		gen._drain_pending()
		gen._update_world()
		guard += 1
	var want: int = (gen.view_chunks * 2 + 1) * (gen.view_chunks * 2 + 1)
	print("WORLD chunks=", gen.chunk_count(), " pending=", gen.pending_count(), " want=", want)
	if gen.chunk_count() != want:
		_fail("chunk ring incomplete")
		return

	var max_h := -INF
	var min_h := INF
	for i in 512:
		var h: float = gen.sample_height(randf_range(-10000.0, 10000.0), randf_range(-10000.0, 10000.0))
		max_h = maxf(max_h, h)
		min_h = minf(min_h, h)
	print("SAMPLE min=", min_h, " max=", max_h)
	if max_h <= 0.0 or min_h >= gen.height_multiplier:
		_fail("height field flat or out of range")
		return
	if not is_equal_approx(gen.sample_height(0.0, 0.0), 0.0):
		_fail("runway apron not flattened at origin")
		return

	var near: MeshInstance3D = gen.get_node_or_null("Chunk_0_0/Mesh")
	if near == null or near.mesh == null or near.mesh.get_surface_count() == 0:
		_fail("central chunk mesh missing or empty")
		return
	var near_arrays := near.mesh.surface_get_arrays(0)
	var verts: PackedVector3Array = near_arrays[Mesh.ARRAY_VERTEX]
	var normals: PackedVector3Array = near_arrays[Mesh.ARRAY_NORMAL]
	var colors: PackedColorArray = near_arrays[Mesh.ARRAY_COLOR]
	var idxs: PackedInt32Array = near_arrays[Mesh.ARRAY_INDEX]
	var up := 0
	for i in mini(normals.size(), 200):
		if normals[i].y > 0.0:
			up += 1
	print("MESH verts=", verts.size(), " tris=", idxs.size() / 3, " colors=", colors.size(), " up=", up)
	if verts.size() == 0 or idxs.size() < 3 or colors.size() != verts.size() or up <= 0:
		_fail("central chunk mesh data inconsistent")
		return

	var near_tris := idxs.size() / 3
	var far_tris := 0
	for c in gen.get_children():
		if c is Node3D and c.name.begins_with("Chunk"):
			if absf(c.position.x) > 3000.0 and absf(c.position.z) > 3000.0:
				var far_mesh: MeshInstance3D = c.get_node_or_null("Mesh")
				if far_mesh != null and far_mesh.mesh != null:
					far_tris = far_mesh.mesh.surface_get_arrays(0)[Mesh.ARRAY_INDEX].size() / 3
					break
	print("LOD near_tris=", near_tris, " far_tris=", far_tris)
	if far_tris <= 0 or far_tris >= near_tris:
		_fail("LOD did not reduce far-chunk triangles")
		return

	drone.position = Vector3(48.0, 50.0, 48.0)
	gen._update_physics_grid()
	guard = 0
	while gen._grid_fill != -1 and guard < 200:
		gen._update_physics_grid()
		guard += 1
	print("GRID calls=", _grid_calls, " len=", _grid_len)
	if _grid_calls != 1 or _grid_len != gen._grid_n * gen._grid_n:
		_fail("initial physics grid was not populated correctly")
		return

	drone.position = Vector3(2000.0, 50.0, 2000.0)
	gen._update_physics_grid()
	guard = 0
	while gen._grid_fill != -1 and guard < 400:
		gen._update_physics_grid()
		guard += 1
	if _grid_calls != 2:
		_fail("physics grid did not re-centre after crossing threshold")
		return

	gen.regenerate()
	if gen.chunk_count() != 0:
		_fail("regenerate did not clear the world")
		return
	_pass()

func _on_grid(_north0: float, _east0: float, _spacing: float, _nx: int, _nz: int, heights: PackedFloat64Array) -> void:
	_grid_calls += 1
	_grid_len = heights.size()

func _fail(message: String) -> void:
	print("FAIL: ", message)
	quit(1)

func _pass() -> void:
	print("terrain validation PASSED")
	quit(0)
