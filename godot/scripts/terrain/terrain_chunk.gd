class_name TerrainChunk
extends Node3D

const MeshScript := preload("res://scripts/terrain/mesh_generator.gd")
## One square patch of procedural terrain: a height-field `ArrayMesh` plus a
## matching concave collision body. Identical to `TerrainGenerator` rendering
## one cell of Sebastian Lague's chunked landmass system (E07/E08), except the
## heights come from sampling a global function at exact world coordinates, so
## shared edges always line up with the neighbouring chunks.

## World-space (x, z) of this chunk's top-left corner (m).
var world_cell := Vector2.ZERO

var chunk_size := 500.0
var resolution := 33
var lod := 0
var collision_enabled := true
var material: Material = null

## Return the surface altitude (m) at world (x, z). Supplied by the generator.
var sample_fn: Callable
## Return the vertex colour for world (x, z). Supplied by the generator.
var colour_fn: Callable

var mesh_instance: MeshInstance3D = null
var collision_body: StaticBody3D = null
var heights := PackedFloat32Array()

func build() -> void:
	_clear()
	if sample_fn.is_null() or not sample_fn.is_valid():
		return
	if colour_fn.is_null() or not colour_fn.is_valid():
		return
	if material == null:
		var fallback := StandardMaterial3D.new()
		fallback.vertex_color_use_as_albedo = true
		fallback.roughness = 1.0
		material = fallback
	var spacing := chunk_size / float(resolution - 1)
	var step_len := 1 if lod <= 0 else 1 << mini(lod, 5)
	# Sample only the LOD-decimated grid (fast streaming: far chunks need
	# ~6x6 samples, not 33x33). LOD seams between neighbours are hidden by
	# the vertical skirt MeshScript.build_mesh appends around the border.
	var side := (resolution - 1) / step_len + 1
	var count := side * side
	var positions := PackedVector3Array()
	var colors := PackedColorArray()
	positions.resize(count)
	colors.resize(count)
	heights.resize(count)
	for iz in side:
		for ix in side:
			var wx := world_cell.x + ix * step_len * spacing
			var wz := world_cell.y + iz * step_len * spacing
			var y: float = sample_fn.call(wx, wz)
			positions[iz * side + ix] = Vector3(ix * step_len * spacing, y, iz * step_len * spacing)
			colors[iz * side + ix] = colour_fn.call(wx, wz)
			heights[iz * side + ix] = y

	mesh_instance = MeshInstance3D.new()
	mesh_instance.name = "Mesh"
	mesh_instance.material_override = material
	mesh_instance.mesh = MeshScript.build_mesh(positions, colors, side, 0, step_len * spacing)
	add_child(mesh_instance)

	if collision_enabled:
		var soup := MeshScript.triangle_soup(positions, side, 0)
		if not soup.is_empty():
			collision_body = StaticBody3D.new()
			collision_body.name = "Collision"
			var shape := ConcavePolygonShape3D.new()
			shape.data = soup
			var col := CollisionShape3D.new()
			col.shape = shape
			collision_body.add_child(col)
			add_child(collision_body)

## Drop the built nodes so `build()` can run again (LOD switch / regenerate).
func _clear() -> void:
	if mesh_instance != null:
		mesh_instance.queue_free()
		mesh_instance = null
	if collision_body != null:
		collision_body.queue_free()
		collision_body = null
	heights = PackedFloat32Array()