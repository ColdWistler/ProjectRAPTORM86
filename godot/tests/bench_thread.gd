extends SceneTree

var node: WindTunnelNode
var iters := 120

func _init() -> void:
	node = WindTunnelNode.new()
	node.set_wind_speed(20.0)
	node.set_attitude(4.0, 0.0, 0.0)
	node.switch_aircraft("MQI")
	node.reset_trails()

	# Measure step() in isolation
	var t0 := Time.get_ticks_usec()
	for f in 60:
		node.step(1.0 / 60.0)
	var step_us := (Time.get_ticks_usec() - t0) / 60.0

	# Measure the full texture-upload path as done in _upload_flow_volume
	var meta: PackedFloat32Array = node.get_flow_meta()
	var nx := int(meta[6]); var ny := int(meta[7]); var nz := int(meta[8])
	var vmax := maxf(meta[9], 1.0)
	var vol: PackedFloat32Array = node.get_flow_volume()
	var scale := 1.0 / vmax

	var images: Array = []
	for k in nz:
		images.append(Image.create(nx, ny, false, Image.FORMAT_RGBAF))
	var tex: ImageTexture3D = ImageTexture3D.new()
	tex.create(Image.FORMAT_RGBAF, nx, ny, nz, false, images)

	# warm
	for f in 10:
		_upload(vol, images, tex, nx, ny, nz, scale)

	t0 = Time.get_ticks_usec()
	for f in iters:
		_upload(vol, images, tex, nx, ny, nz, scale)
	var up_us := (Time.get_ticks_usec() - t0) / iters

	print("STEP_US=", step_us)
	print("UPLOAD_US=", up_us)
	print("STEP_PLUS_UPLOAD_US=", step_us + up_us)
	print("60FPS_BUDGET_US=", 16666)
	print("30FPS_BUDGET_US=", 33333)
	quit(0)

func _upload(vol: PackedFloat32Array, images: Array, tex: ImageTexture3D, nx: int, ny: int, nz: int, scale: float) -> void:
	var slice := PackedFloat32Array()
	slice.resize(nx * ny * 4)
	for k in nz:
		for j in ny:
			for i in nx:
				var src := ((k * ny) + j) * nx + i
				var vx: float = vol[src * 3]
				var vy: float = vol[src * 3 + 1]
				var vz: float = vol[src * 3 + 2]
				var dst := (j * nx + i) * 4
				slice[dst] = vx * scale
				slice[dst + 1] = vy * scale
				slice[dst + 2] = vz * scale
				var m := sqrt(vx * vx + vy * vy + vz * vz)
				slice[dst + 3] = m * scale
		var img: Image = images[k]
		img.set_data(nx, ny, false, Image.FORMAT_RGBAF, slice.to_byte_array())
	tex.update(images)
