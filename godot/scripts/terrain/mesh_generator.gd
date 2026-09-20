class_name TerrainMeshGenerator
extends RefCounted
## Height-field to `ArrayMesh` conversion — the Godot analogue of
## `MeshGenerator.GenerateTerrainMesh` from the E05/E08 episodes.
##
## Chunks are sampled at their LOD-decimated density and a vertical skirt is
## hung around the rendered border: neighbouring chunks at different LODs
## otherwise leave T-junction cracks (the coarse edge is straight while the
## fine edge follows the terrain curve), and the skirt wall shows instead of
## a see-through hole.

## Turn `resolution * resolution` world vertices (row-major, +x then +z) into a
## shaded height-field mesh. Normals come from central-difference slopes of the
## vertex heights so flattish patches stay smooth. `step` mirrors Sebastian's
## `meshSimplificationIncrement` but as powers of two (0 → 1, else `1 << lod`)
## so the step always divides `resolution - 1` evenly.
static func build_mesh(
	positions: PackedVector3Array,
	colors: PackedColorArray,
	resolution: int,
	lod: int,
	vertex_spacing: float,
	skirt_depth: float = -1.0,
) -> ArrayMesh:
	var mesh := ArrayMesh.new()
	if positions.is_empty():
		return mesh
	# Normals must use the true full-res grid spacing, not the LOD step.
	var full_spacing := vertex_spacing / float(maxi(1 if lod <= 0 else 1 << mini(lod, 5), 1))
	var normals := _compute_normals(positions, resolution, maxf(full_spacing, 0.0001))
	var indices := _build_indices(resolution, lod)
	if skirt_depth < 0.0:
		# Just tall enough to cover the worst LOD seam step; capped so
		# distant chunks don't grow giant curtain walls.
		skirt_depth = clampf(maxf(vertex_spacing * 2.0, 30.0), 30.0, 120.0)
	_add_skirt(positions, colors, normals, indices, resolution, lod, skirt_depth)
	var arrays := []
	arrays.resize(Mesh.ARRAY_MAX)
	arrays[Mesh.ARRAY_VERTEX] = positions
	arrays[Mesh.ARRAY_NORMAL] = normals
	arrays[Mesh.ARRAY_COLOR] = colors
	arrays[Mesh.ARRAY_INDEX] = indices
	mesh.add_surface_from_arrays(Mesh.PRIMITIVE_TRIANGLES, arrays)
	return mesh

## Triangle soup for a `ConcavePolygonShape3D`, built from the exact same
## vertices/indices as the visual mesh so collision always matches the ground.
static func triangle_soup(
	positions: PackedVector3Array,
	resolution: int,
	lod: int,
) -> PackedVector3Array:
	var soup := PackedVector3Array()
	var idx := _build_indices(resolution, lod)
	for i in idx:
		soup.append(positions[i])
	return soup

static func _build_indices(resolution: int, lod: int) -> PackedInt32Array:
	var step := 1 if lod <= 0 else 1 << mini(lod, 5)
	var indices := PackedInt32Array()
	for iz in range(0, resolution - 1, step):
		var next_iz := mini(iz + step, resolution - 1)
		for ix in range(0, resolution - 1, step):
			var next_ix := mini(ix + step, resolution - 1)
			var p00 := iz * resolution + ix
			var p10 := iz * resolution + next_ix
			var p01 := next_iz * resolution + ix
			var p11 := next_iz * resolution + next_ix
			# Winding must match Godot's front-face convention (verified
			# against PlaneMesh, which faces +Y): reversed order is
			# backface-culled from above, leaving only skirts visible.
			indices.append(p11)
			indices.append(p01)
			indices.append(p10)
			indices.append(p01)
			indices.append(p00)
			indices.append(p10)
	return indices

## Append a vertical skirt around the LOD-decimated border. The coarse edge
## of one chunk is a straight line while a finer neighbour's shared edge
## follows the terrain curve (classic Lague E08 T-junction crack); the skirt
## hangs below the rendered border so any such seam shows skirt wall instead
## of a see-through hole. Must walk the *decimated* border (the vertices the
## index buffer actually references), not the full-res grid.
static func _add_skirt(
	positions: PackedVector3Array,
	colors: PackedColorArray,
	normals: PackedVector3Array,
	indices: PackedInt32Array,
	resolution: int,
	lod: int,
	depth: float,
) -> void:
	var step := 1 if lod <= 0 else 1 << mini(lod, 5)
	var border: PackedInt32Array = PackedInt32Array()
	for ix in range(0, resolution, step):
		border.append(ix)  # north edge (iz = 0)
		if ix + step >= resolution:
			break
	border[border.size() - 1] = resolution - 1
	for iz in range(step, resolution, step):
		border.append(mini(iz, resolution - 1) * resolution + (resolution - 1))  # east
	for ix in range(resolution - 1 - step, -1, -step):
		border.append((resolution - 1) * resolution + maxi(ix, 0))  # south
	for iz in range(resolution - 1 - step, 0, -step):
		border.append(maxi(iz, 0) * resolution)  # west
	if border.is_empty():
		return
	var base := positions.size()
	positions.resize(base + border.size() * 2)
	colors.resize(base + border.size() * 2)
	normals.resize(base + border.size() * 2)
	for b in border.size():
		var vi := border[b]
		var p: Vector3 = positions[vi]
		positions[base + b * 2] = p
		positions[base + b * 2 + 1] = Vector3(p.x, p.y - depth, p.z)
		var c: Color = colors[vi]
		colors[base + b * 2] = c
		colors[base + b * 2 + 1] = c
		var n: Vector3 = normals[vi]
		normals[base + b * 2] = n
		normals[base + b * 2 + 1] = n
	for b in border.size():
		var n2 := (b + 1) % border.size()
		var t0 := base + b * 2
		var b0 := base + b * 2 + 1
		var t1 := base + n2 * 2
		var b1 := base + n2 * 2 + 1
		indices.append_array(PackedInt32Array([t0, t1, b0, t1, b1, b0]))

## Smooth-shaded normals from central-difference slopes of the height field.
static func _compute_normals(positions: PackedVector3Array, resolution: int, vertex_spacing: float) -> PackedVector3Array:
	var normals := PackedVector3Array()
	normals.resize(positions.size())
	if resolution < 2:
		normals.fill(Vector3.UP)
		return normals
	var horizontal := maxf(vertex_spacing, 0.0001) * 2.0
	for iz in resolution:
		for ix in resolution:
			var h := positions[iz * resolution + ix].y
			var hl := _height_at(positions, resolution, ix - 1, iz, h)
			var hr := _height_at(positions, resolution, ix + 1, iz, h)
			var hu := _height_at(positions, resolution, ix, iz + 1, h)
			var hd := _height_at(positions, resolution, ix, iz - 1, h)
			var ddx := (hr - hl) / horizontal
			var ddz := (hu - hd) / horizontal
			normals[iz * resolution + ix] = Vector3(-ddx, 1.0, -ddz).normalized()
	return normals

static func _height_at(
	positions: PackedVector3Array,
	resolution: int,
	x: int,
	z: int,
	fallback: float,
) -> float:
	if x < 0 or x >= resolution or z < 0 or z >= resolution:
		return fallback
	return positions[z * resolution + x].y