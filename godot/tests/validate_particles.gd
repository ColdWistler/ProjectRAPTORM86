extends SceneTree

func _init() -> void:
	var s = load("res://shaders/smoke_puff.gdshader")
	if s == null:
		print("SHADER_NULL")
		quit(1)
		return
	print("SHADER_OK code_len=", s.code.length())

	var node = WindTunnelNode.new()
	node.set_wind_speed(20.0)
	node.set_attitude(4.0, 0.0, 0.0)
	node.reset_trails()
	print("COUNT=", node.particle_count())
	for f in 120:
		node.step(1.0 / 60.0)
	var d: PackedFloat32Array = node.get_particles()
	print("DATA_LEN=", d.size())
	var min_x := 1e9
	var max_x := -1e9
	var max_r := 0.0
	var sample_y := 0.0
	for i in node.particle_count():
		var b := i * 6
		var p := Vector3(d[b], d[b + 1], d[b + 2])
		min_x = minf(min_x, p.x)
		max_x = maxf(max_x, p.x)
		max_r = maxf(max_r, d[b + 3])
		if i == node.particle_count() / 2:
			sample_y = p.y
	print("SPREAD_X=", min_x, "..", max_x, " max_r=", max_r, " dist_sample=", sample_y)
	node.set_throttle(1.0)
	node.set_throttle_split(-1.0)
	node.switch_aircraft("TwinEngine")
	node.step(1.0 / 60.0)
	var a: PackedFloat64Array = node.get_aero()
	print("AERO=", a)
	# --- Imported-shape path: an axis-aligned unit cube (closed) ---------------
	var ishape := _cube_mesh()
	var npan: int = node.set_imported_shape(ishape[0], ishape[1], 8)
	print("IMPORTED_PANELS=", npan, " is_imported=", node.is_imported_shape())
	node.step(1.0 / 60.0)
	var ai: PackedFloat64Array = node.get_aero()
	print("IMPORTED_AERO=", ai)
	# Clearing should return to coefficient aero (not imported).
	node.set_imported_shape(PackedVector3Array(), PackedInt32Array(), 8)
	print("CLEARED_IS_IMPORTED=", node.is_imported_shape())
	quit(0)

## A closed unit cube ([-0.5,0.5]^3) as vertex+index triangle soup.
func _cube_mesh() -> Array:
	var verts := PackedVector3Array()
	var idxs := PackedInt32Array()
	var c := [
		Vector3(-0.5, -0.5, -0.5), Vector3(0.5, -0.5, -0.5), Vector3(0.5, 0.5, -0.5), Vector3(-0.5, 0.5, -0.5),
		Vector3(-0.5, -0.5, 0.5), Vector3(0.5, -0.5, 0.5), Vector3(0.5, 0.5, 0.5), Vector3(-0.5, 0.5, 0.5),
	]
	for v in c:
		verts.append(v)
	for quad in [[0, 1, 2, 3], [4, 6, 5, 7], [0, 5, 1, 4], [3, 2, 6, 7], [0, 4, 7, 3], [1, 5, 6, 2]]:
		idxs.append_array([quad[0], quad[1], quad[2], quad[0], quad[2], quad[3]])
	return [verts, idxs]