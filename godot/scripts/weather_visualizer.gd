@tool
extends Node3D
## World-space weather visualization tied to the `WeatherSystem` GDExtension
## node (child of the flight scene, or a `/root` autoload). It renders:
##
##   • rain / snow / hail GPU particles — kind follows the precipitation
##     type, particle count follows rain intensity, drops are pushed by wind
##   • exponential depth fog matching the current visibility distance
##   • a world-space wind arrow + numeric label next to the camera
##   • an updraft / downdraft arrow (green up, red down)
##   • storm dimming (sky + sun energy) and cloud darkening, refreshed when
##     the weather scene rolls over
##
## The visualizer follows the active camera every frame and turns everything
## off (restoring the environment baseline) while weather is disabled from
## the [F1] menu, or when no WeatherSystem is present.

@export_range(0.0, 200.0, 1.0) var emitter_height := 45.0:
	set(v):
		emitter_height = v
		_apply_emitter_height()
@export_range(20.0, 250.0, 5.0) var precip_radius := 90.0
@export_range(500, 6000, 100) var max_particles := 3500
@export var hide_when_menu_off := true
## Camera-local placement (right, up, in front of the camera) for the wind HUD.
@export var wind_arrow_offset := Vector3(4.0, -0.6, -8.0)
@export var updraft_arrow_offset := Vector3(-4.0, -0.6, -8.0)
@export var wind_label_offset := Vector3(4.0, -2.0, -8.0)

var weather = null
var _env: Environment = null
var _sun: DirectionalLight3D = null
var _clouds: Node = null
var _clouds_dark := false

var _rain: GPUParticles3D = null
var _snow: GPUParticles3D = null
var _hail: GPUParticles3D = null
var _emitters: Array = []
var _wind_arrow: Node3D = null
var _updraft_arrow: Node3D = null
var _updraft_mat: StandardMaterial3D = null
var _wind_label: Label3D = null

var _baseline_bg_energy := 1.0
var _baseline_sun_energy := 1.0
var _baseline_fog_enabled := false
var _baseline_fog_density := 0.0
var _fog_density := 0.0
var _last_scene: int = -999
var _clock := 0.0

func _ready() -> void:
	if Engine.is_editor_hint():
		return
	weather = _find_weather_system()
	if weather == null:
		return
	var parent := get_parent()
	var we: WorldEnvironment = null
	if parent != null:
		we = parent.get_node_or_null("WorldEnvironment") as WorldEnvironment
	if we != null:
		# Work on a private copy so we never mutate the shared .tres resource.
		_env = we.environment
		_env = _env.duplicate()
		we.environment = _env
		_baseline_bg_energy = _env.background_energy_multiplier
		_baseline_fog_enabled = _env.fog_enabled
		_baseline_fog_density = _env.fog_density
	_sun = get_parent().get_node_or_null("Sun") as DirectionalLight3D if get_parent() != null else null
	if _sun != null:
		_baseline_sun_energy = _sun.light_energy
	_clouds = get_parent().get_node_or_null("Clouds") if get_parent() != null else null
	_build_precipitation()
	_build_arrows()
	_apply_emitter_height()

func _process(delta: float) -> void:
	if Engine.is_editor_hint():
		return
	_clock += delta
	var cam := get_viewport().get_camera_3d()
	if cam != null:
		global_position = cam.global_position
	if not _is_weather_active():
		_restore_baseline()
		return
	if not weather.is_ready():
		_restore_baseline()
		return
	if cam != null:
		_place_hud_children(cam)
	_update_precipitation()
	_update_fog()
	_update_arrows()
	_update_lighting()
	_update_clouds()

## Weather drives the visuals only while the [F1] "Weather enabled" switch is
## on (flight_sim exposes `weather_enabled` on the scene root).
func _is_weather_active() -> bool:
	if weather == null:
		return false
	if not hide_when_menu_off:
		return true
	var sc := get_tree()
	if sc == null or sc.current_scene == null:
		return true
	var scene: Node = sc.current_scene
	if "weather_enabled" in scene and not scene.weather_enabled:
		return false
	return true

func _restore_baseline() -> void:
	_set_precip_visible(false)
	if _wind_arrow != null:
		_wind_arrow.visible = false
	if _updraft_arrow != null:
		_updraft_arrow.visible = false
	if _wind_label != null:
		_wind_label.text = ""
	if _env != null:
		_env.fog_enabled = _baseline_fog_enabled
		_env.fog_density = _baseline_fog_density
		_env.background_energy_multiplier = _baseline_bg_energy
	if _sun != null:
		_sun.light_energy = _baseline_sun_energy
	if _clouds_dark and _clouds != null:
		_clouds.cloud_albedo = Color(0.94, 0.94, 0.97)
		_clouds.density = 0.12
		_clouds_dark = false
	_last_scene = -999

# ---------------------------------------------------------------------------
# Precipitation
# ---------------------------------------------------------------------------

func _build_precipitation() -> void:
	_rain = _make_emitter("Rain", _make_particle_mesh(_streak_texture()), {
		"gravity": Vector3(0, -4.0, 0),
		"speed_min": 16.0,
		"speed_max": 26.0,
		"scale_min": 0.035,
		"scale_max": 0.05,
		"lifetime": 1.4,
		"color": Color(0.55, 0.66, 0.82, 0.5),
	})
	_snow = _make_emitter("Snow", _make_particle_mesh(_dot_texture(32, 1.6)), {
		"gravity": Vector3(0, -0.6, 0),
		"speed_min": 0.8,
		"speed_max": 2.2,
		"spread": 14.0,
		"scale_min": 0.10,
		"scale_max": 0.17,
		"lifetime": 2.2,
		"color": Color(0.96, 0.97, 1.0, 0.92),
	})
	_hail = _make_emitter("Hail", _make_particle_mesh(_dot_texture(24, 2.2)), {
		"gravity": Vector3(0, -6.0, 0),
		"speed_min": 12.0,
		"speed_max": 22.0,
		"scale_min": 0.05,
		"scale_max": 0.09,
		"lifetime": 1.5,
		"color": Color(0.88, 0.92, 0.97, 0.95),
	})
	_emitters = [_rain, _snow, _hail]

func _make_emitter(name: String, mesh: Mesh, cfg: Dictionary) -> GPUParticles3D:
	var p := GPUParticles3D.new()
	p.name = name
	var pm := ParticleProcessMaterial.new()
	pm.emission_shape = ParticleProcessMaterial.EMISSION_SHAPE_BOX
	pm.emission_box_extents = Vector3(precip_radius, 2.0, precip_radius)
	pm.direction = Vector3(0, -1, 0)
	pm.gravity = cfg.gravity
	pm.initial_velocity_min = cfg.speed_min
	pm.initial_velocity_max = cfg.speed_max
	pm.spread = cfg.get("spread", 6.0)
	pm.scale_min = cfg.scale_min
	pm.scale_max = cfg.scale_max
	pm.color = cfg.color
	p.process_material = pm
	p.draw_passes = 1
	p.draw_pass_1 = mesh
	p.lifetime = cfg.lifetime
	p.amount = 1
	p.emitting = false
	p.visible = false
	p.visibility_aabb = AABB(
		Vector3(-precip_radius, -20.0, -precip_radius),
		Vector3(precip_radius * 2.0, 120.0, precip_radius * 2.0)
	)
	add_child(p)
	return p

func _update_precipitation() -> void:
	var ptype: int = int(weather.get_precipitation_type())
	var rain: float = float(weather.get_rain_intensity())
	var wind: Vector3 = weather.get_wind_vector()
	var target = null
	match ptype:
		1:
			target = _rain
		2:
			target = _snow
		3:
			target = _hail
	var target_amount := 0
	if target != null and rain > 0.02:
		target_amount = int(float(max_particles) * clampf(rain, 0.0, 1.0))
		var pm: ParticleProcessMaterial = target.process_material
		# Tilt the fall direction with the wind so drops visibly drift.
		pm.direction = (wind * 0.35 + Vector3(0, -1, 0)).normalized()
	for e in _emitters:
		var on: bool = e == target and target_amount > 0
		e.visible = on
		e.emitting = on
		if e == target:
			e.amount = maxi(target_amount, 1)

func _set_precip_visible(on: bool) -> void:
	for e in _emitters:
		if e != null:
			e.visible = on and e.emitting

func _apply_emitter_height() -> void:
	if _emitters.size() == 0:
		return
	for e in _emitters:
		if e != null:
			e.position = Vector3(0, emitter_height, 0)

func _make_particle_mesh(tex: ImageTexture) -> QuadMesh:
	var mat := StandardMaterial3D.new()
	mat.shading_mode = BaseMaterial3D.SHADING_MODE_UNSHADED
	mat.transparency = BaseMaterial3D.TRANSPARENCY_ALPHA
	mat.cull_mode = BaseMaterial3D.CULL_DISABLED
	mat.billboard_mode = BaseMaterial3D.BILLBOARD_ENABLED
	mat.albedo_texture = tex
	var q := QuadMesh.new()
	q.size = Vector2(1, 1)
	q.material = mat
	return q

## A thin vertical streak texture for rain drops.
func _streak_texture() -> ImageTexture:
	var w := 16
	var h := 128
	var img := Image.create(w, h, false, Image.FORMAT_RGBA8)
	for y in h:
		var a := 1.0 - float(y) / float(h)
		a *= 0.75 + 0.25 * sin(float(y) * 0.35)
		for x in w:
			var edge := absf(float(x) / float(w - 1) * 2.0 - 1.0)
			var edge_fall := clampf(1.0 - edge * edge * 3.0, 0.0, 1.0)
			img.set_pixel(x, y, Color(1, 1, 1, a * edge_fall))
	return ImageTexture.create_from_image(img)

## A soft radial dot texture (snow flakes / hail stones).
func _dot_texture(size: int, soft: float) -> ImageTexture:
	var img := Image.create(size, size, false, Image.FORMAT_RGBA8)
	var r := float(size) * 0.5
	for y in size:
		for x in size:
			var dx := float(x) + 0.5 - r
			var dy := float(y) + 0.5 - r
			var d := sqrt(dx * dx + dy * dy) / r
			var a := 1.0 - clampf(pow(d, soft), 0.0, 1.0)
			img.set_pixel(x, y, Color(1, 1, 1, a))
	return ImageTexture.create_from_image(img)

# ---------------------------------------------------------------------------
# Fog / visibility
# ---------------------------------------------------------------------------

func _update_fog() -> void:
	if _env == null:
		return
	var vis := float(weather.get_visibility())
	var target := 0.0
	if vis < 15000.0:
		# Exponential depth-fog density such that objects at `vis` are ~98% gone.
		target = clampf(4.0 / maxf(vis, 50.0), 0.0002, 0.01)
	_fog_density = lerpf(_fog_density, target, 0.15)
	_env.fog_enabled = _fog_density > 0.00025
	_env.fog_density = _fog_density

# ---------------------------------------------------------------------------
# Wind / updraft arrows
# ---------------------------------------------------------------------------

func _build_arrows() -> void:
	_wind_arrow = _make_arrow(Color(0.35, 0.9, 1.0))
	add_child(_wind_arrow)

	_updraft_arrow = _make_arrow(Color(0.25, 0.95, 0.4))
	_updraft_mat = _updraft_arrow.get_meta("mat")
	add_child(_updraft_arrow)

	_wind_label = Label3D.new()
	_wind_label.billboard = BaseMaterial3D.BILLBOARD_ENABLED
	_wind_label.fixed_size = true
	_wind_label.no_depth_test = true
	_wind_label.font_size = 22
	_wind_label.outline_size = 6
	_wind_label.modulate = Color(0.85, 0.95, 1.0, 0.95)
	add_child(_wind_label)

## Pin the wind HUD inside the view: offsets are camera-local and transformed
## by the camera basis each frame, so the arrows/readout always sit in-frame
## no matter which way the chase camera points.
func _place_hud_children(cam: Camera3D) -> void:
	var tf: Transform3D = cam.global_transform
	if _wind_arrow != null:
		_wind_arrow.global_position = tf * wind_arrow_offset
	if _updraft_arrow != null:
		_updraft_arrow.global_position = tf * updraft_arrow_offset
	if _wind_label != null:
		_wind_label.global_position = tf * wind_label_offset

func _make_arrow(color: Color) -> Node3D:
	var mat := StandardMaterial3D.new()
	mat.shading_mode = BaseMaterial3D.SHADING_MODE_UNSHADED
	mat.albedo_color = color
	var arrow := Node3D.new()
	arrow.set_meta("mat", mat)
	var shaft := MeshInstance3D.new()
	var cyl := CylinderMesh.new()
	cyl.top_radius = 0.09
	cyl.bottom_radius = 0.09
	cyl.height = 2.2
	cyl.radial_segments = 8
	shaft.mesh = cyl
	shaft.material_override = mat
	shaft.rotation_degrees = Vector3(90, 0, 0)
	arrow.add_child(shaft)
	var head := MeshInstance3D.new()
	var cone := CylinderMesh.new()
	cone.top_radius = 0.0
	cone.bottom_radius = 0.24
	cone.height = 0.6
	cone.radial_segments = 8
	head.mesh = cone
	head.material_override = mat
	head.rotation_degrees = Vector3(90, 0, 0)
	head.position = Vector3(0, 0, 1.4)
	arrow.add_child(head)
	arrow.visible = false
	return arrow

func _update_arrows() -> void:
	var wind: Vector3 = weather.get_wind_vector()
	var show := wind.length() > 0.3
	_wind_arrow.visible = show
	if show:
		_orient_arrow(_wind_arrow, wind.normalized())
		_wind_arrow.scale = Vector3(1.0, 1.0, clampf(wind.length() / 10.0, 0.4, 1.6))
	else:
		_wind_arrow.scale = Vector3.ONE
	var up := float(weather.get_updraft_strength())
	var show_up := absf(up) > 0.15
	_updraft_arrow.visible = show_up
	if show_up:
		_orient_arrow(_updraft_arrow, Vector3(0, signf(up), 0))
		_updraft_arrow.scale = Vector3(1.0, 1.0, clampf(absf(up) / 4.0, 0.4, 1.8))
		_updraft_mat.albedo_color = Color(0.25, 0.95, 0.4) if up > 0.0 else Color(0.95, 0.3, 0.3)
	else:
		_updraft_arrow.scale = Vector3.ONE
	_wind_label.text = "W %0.1f m/s   gust %0.1f" % [wind.length(), float(weather.get_gust_rms())]

## Align the arrow's local +Z axis with `dir` (robust to vertical directions).
func _orient_arrow(node: Node3D, dir: Vector3) -> void:
	var z := dir.normalized()
	var x := Vector3.RIGHT if absf(z.dot(Vector3.UP)) > 0.98 else Vector3.UP.cross(z).normalized()
	var y := z.cross(x)
	node.transform.basis = Basis(x, y, z)

# ---------------------------------------------------------------------------
# Lighting + clouds
# ---------------------------------------------------------------------------

func _update_lighting() -> void:
	var sev: int = int(weather.get_severity())
	var rain: float = float(weather.get_rain_intensity())
	var levels := [1.0, 0.94, 0.80, 0.62]
	var mult := float(levels[clampi(sev, 0, 3)]) * (1.0 - 0.22 * rain)
	mult = clampf(mult, 0.35, 1.0)
	if _env != null:
		_env.background_energy_multiplier = lerpf(
			_env.background_energy_multiplier, _baseline_bg_energy * mult, 0.12
		)
	if _sun != null:
		_sun.light_energy = lerpf(_sun.light_energy, _baseline_sun_energy * mult, 0.12)

## Darken / densify the cloud field once per weather scene roll-over (the
## cloud system rebuilds on these setters, so never poke them per-frame).
func _update_clouds() -> void:
	if _clouds == null:
		return
	var scene_idx: int = int(weather.get_scene_index())
	if scene_idx == _last_scene:
		return
	_last_scene = scene_idx
	var sev: int = int(weather.get_severity())
	if sev >= 1:
		var t := clampf(float(sev) / 3.0, 0.0, 1.0)
		_clouds.cloud_albedo = Color(0.94, 0.94, 0.97).lerp(Color(0.42, 0.44, 0.52), t)
		_clouds.density = 0.12 + 0.07 * t
		_clouds_dark = true
	else:
		_clouds.cloud_albedo = Color(0.94, 0.94, 0.97)
		_clouds.density = 0.12
		_clouds_dark = false

# ---------------------------------------------------------------------------
# Lookup
# ---------------------------------------------------------------------------

func _find_weather_system() -> Node:
	var parent := get_parent()
	if parent != null:
		var child := parent.get_node_or_null("WeatherSystem")
		if child != null:
			return child
	return get_node_or_null("/root/WeatherSystem")