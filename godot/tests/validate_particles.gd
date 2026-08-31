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
	for i in 12000:
		var b := i * 6
		var p := Vector3(d[b], d[b + 1], d[b + 2])
		min_x = minf(min_x, p.x)
		max_x = maxf(max_x, p.x)
		max_r = maxf(max_r, d[b + 3])
		if i == 6000:
			sample_y = p.y
	print("SPREAD_X=", min_x, "..", max_x, " max_r=", max_r, " dist_sample=", sample_y)
	quit(0)