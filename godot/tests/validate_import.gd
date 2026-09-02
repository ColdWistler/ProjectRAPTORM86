extends SceneTree

func _init() -> void:
	var node = WindTunnelNode.new()
	node.set_wind_speed(20.0)
	var glb := "res://assets/aircraft/TwinEngine.glb"
	if not ResourceLoader.exists(glb):
		print("GLB_MISSING")
		quit(1)
		return
	var packed := load(glb)
	var root: Node3D = packed.instantiate()
	# Traverse to collect triangle geometry (same logic as wind_tunnel.gd).
	var verts := PackedVector3Array()
	var idxs := PackedInt32Array()
	_collect(root, Transform3D.IDENTITY, verts, idxs)
	root.free()
	print("GLB_TRIS=", idxs.size() / 3, " verts=", verts.size())
	if verts.is_empty():
		print("GLB_NO_GEOM")
		quit(1)
		return
	_normalize(verts)
	var n: int = node.set_imported_shape(verts, idxs, 12)
	print("GLB_PANELS=", n)
	for res in [16, 20, 24]:
		var nn: int = node.set_imported_shape(verts, idxs, res)
		node.step(1.0 / 60.0)
		print("GLB_RES_", res, "_PANELS=", nn, " AERO=", node.get_aero(), " DIAG=", node.get_imported_aero())
	node.set_wind_direction(90.0)
	node.step(1.0 / 60.0)
	print("GLB_RES_24_WIND90_AERO=", node.get_aero())
	quit(0 if n > 0 else 1)

func _collect(node: Node, xform: Transform3D, verts: PackedVector3Array, idxs: PackedInt32Array) -> void:
	if node is MeshInstance3D:
		var mesh := (node as MeshInstance3D).mesh
		if mesh is ArrayMesh:
			var am := mesh as ArrayMesh
			for s in am.get_surface_count():
				var arrs := am.surface_get_arrays(s)
				if arrs.is_empty():
					continue
				var v := arrs[Mesh.ARRAY_VERTEX] as PackedVector3Array
				if v.is_empty():
					continue
				var ni := arrs[Mesh.ARRAY_INDEX] as PackedInt32Array
				var base := verts.size()
				for p in v:
					verts.append(xform * p)
				if ni.is_empty():
					for q in v.size():
						idxs.append(base + q)
				else:
					for qi in ni:
						idxs.append(base + qi)
	for c in node.get_children():
		_collect(c, xform * c.transform, verts, idxs)

func _normalize(verts: PackedVector3Array) -> void:
	if verts.is_empty():
		return
	var minv := verts[0]
	var maxv := verts[0]
	for i in verts.size():
		minv = minv.min(verts[i])
		maxv = maxv.max(verts[i])
	var center := (minv + maxv) * 0.5
	var ext := maxv - minv
	var size := maxf(maxf(ext.x, ext.y), ext.z)
	if size < 0.001:
		size = 1.0
	var scale := 7.0 / size
	for i in verts.size():
		verts[i] = (verts[i] - center) * scale
