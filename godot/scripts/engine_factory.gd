class_name EngineFactory
extends RefCounted
## Procedural "Engine MODE" jet: swept-wing twin-turbofan light jet
## Built nose-forward along +X, up +Y, right +Z — matching FlightSimNode convention
## Dimensions match Engine.toml: span 16.5m, area 26m², chord 1.58m, AR ~10.5

const BODY := Color(0.88, 0.90, 0.93)   # airliner white
const BELLY := Color(0.62, 0.66, 0.72)
const DARK := Color(0.15, 0.17, 0.20)
const GLASS := Color(0.08, 0.10, 0.13)
const NACELLE := Color(0.72, 0.74, 0.78)
const RED := Color(0.95, 0.12, 0.08)
const GREEN := Color(0.10, 0.90, 0.20)
const FAN_BLADE := Color(0.25, 0.27, 0.30)
const ENGINE_CORE := Color(0.12, 0.14, 0.18)

# TOML dimensions (metres)
const FUSELAGE_LEN := 14.5
const FUSELAGE_RADIUS := 0.85
const NOSE_LEN := 1.35
const TAIL_CONE_LEN := 2.1
const WING_HALFSPAN := 8.25       # 16.5 / 2
const WING_ROOT_CHORD := 2.2
const WING_TIP_CHORD := 0.95
const WING_THICKNESS := 0.12
const WING_SWEEP_DEG := 28.0      # ~0.5 rad sweep
const SWEEP_RAD := 0.4887         # 28 deg in rad
const H_STAB_HALFSPAN := 2.1
const H_STAB_CHORD := 1.2
const V_FIN_HEIGHT := 3.4
const V_FIN_CHORD := 0.65
const NACELLE_RADIUS := 0.50
const NACELLE_LEN := 3.4
const NACELLE_Z_OFFSET := 1.15    # lateral position on rear fuselage
const NACELLE_X_OFFSET := -2.6    # aft of wing

static func build(parent: Node3D) -> Dictionary:
	var result := {
		"propellers": [],   # turbofan fan disks (animated)
		"flaps": [],
		"ailerons": [],
	}

	# --- Continuous fuselage: nose -> cabin -> tail cone (no gaps) ---
	# Nose cone (ogive)
	_add(parent, _cone(FUSELAGE_RADIUS * 1.05, NOSE_LEN), BODY, Vector3(7.2, 0.40, 0), Vector3(0, 0, PI / 2))
	# Forward fuselage (cabin)
	_add(parent, _cyl(FUSELAGE_RADIUS, 10.5), BODY, Vector3(1.5, 0.40, 0), Vector3(0, 0, PI / 2))
	# Aft fuselage (tapering to tail)
	_add(parent, _cyl(FUSELAGE_RADIUS * 0.85, 3.0), BODY, Vector3(-4.0, 0.40, 0), Vector3(0, 0, PI / 2))
	# Tail cone
	_add(parent, _cone(FUSELAGE_RADIUS * 0.35, TAIL_CONE_LEN), BODY, Vector3(-6.9, 0.40, 0), Vector3(0, 0, PI / 2), Vector3(0.65, 1.0, 1.0))

	# --- Belly fairing (flatter bottom) ---
	_add(parent, _box(Vector3(11.0, 0.25, 1.4)), BELLY, Vector3(0.5, -0.15, 0))

	# --- Cockpit windshield (blended) ---
	_add(parent, _box(Vector3(1.4, 0.55, 1.15)), GLASS, Vector3(5.6, 0.60, 0))
	# Cockpit side windows
	_add(parent, _box(Vector3(0.8, 0.4, 0.04)), GLASS, Vector3(5.8, 0.55, 0.88))
	_add(parent, _box(Vector3(0.8, 0.4, 0.04)), GLASS, Vector3(5.8, 0.55, -0.88))
	# Cabin windows (3 per side)
	for wx in [4.6, 3.5, 2.4]:
		_add(parent, _box(Vector3(0.30, 0.22, 0.08)), GLASS, Vector3(wx, 0.52, 0.87))
		_add(parent, _box(Vector3(0.30, 0.22, 0.08)), GLASS, Vector3(wx, 0.52, -0.87))

	# --- Empennage: vertical fin + horizontal stabilizer ---
	# Vertical fin (swept)
	_add(parent, _box(Vector3(V_FIN_CHORD, V_FIN_HEIGHT, 0.10)), BODY, Vector3(-6.2, 2.0, 0), Vector3(0, 0, 0.15))
	_add(parent, _box(Vector3(0.10, 0.70, 0.30)), DARK, Vector3(-6.35, 3.4, 0), Vector3(0, 0, 0.15))  # fin tip
	# Horizontal stabilizer
	_add(parent, _box(Vector3(2.40, 0.10, 4.2)), BODY, Vector3(-5.4, 1.05, 0))

	# --- Rudder on vertical fin (trailing edge) ---
	var rudder_mesh := _box(Vector3(0.35, 2.8, 0.08))
	var rudder_pivot := _pivot(parent, Vector3(-6.2, 0.6, 0), rudder_mesh, "RUDDER")
	rudder_pivot.rotation = Vector3(0, 0, 0.15)  # match fin sweep
	result["ailerons"].append(rudder_pivot)  # reuse aileron array for rudder animation

	# --- Elevators on horizontal stabilizer ---
	var elev_mesh := _box(Vector3(0.60, 0.08, 1.9))
	var elev_l := _pivot(parent, Vector3(-5.4, 1.05, -1.55), elev_mesh, "ELEVATOR_L")
	var elev_r := _pivot(parent, Vector3(-5.4, 1.05, 1.55), elev_mesh.duplicate(), "ELEVATOR_R")
	result["flaps"] = [elev_l, elev_r]  # reuse flaps array for elevator animation

	# --- Swept main wings (right wing +Z, left wing -Z) ---
	# Wing box spans from root at X=0.4 to tip at X=0.4 - span*sin(sweep)
	var wing_root_x := 0.4
	var wing_tip_x := wing_root_x - WING_HALFSPAN * sin(SWEEP_RAD)  # ~0.4 - 8.25*0.47 = -3.5
	var wing_root_z := WING_HALFSPAN * cos(SWEEP_RAD)  # ~8.25*0.88 = 7.27 (but we use 4.0 for visual root)
	
	# Visual wing root at Z=±4.0, semi-span 7.0 swept back
	var vis_semi_span := 7.0
	var vis_sweep := SWEEP_RAD
	
	# Left wing (negative Z)
	_add(parent, _tapered_wing_box(vis_semi_span, WING_ROOT_CHORD, WING_TIP_CHORD, WING_THICKNESS), 
		BODY, Vector3(wing_root_x, 0.12, 4.0), Vector3(0, -vis_sweep, 0))
	# Right wing (positive Z)
	_add(parent, _tapered_wing_box(vis_semi_span, WING_ROOT_CHORD, WING_TIP_CHORD, WING_THICKNESS), 
		BODY, Vector3(wing_root_x, 0.12, -4.0), Vector3(0, vis_sweep, 0))

	# Wingtip navigation lights
	var tip_x := wing_root_x - vis_semi_span * sin(vis_sweep)
	var tip_z := 4.0 + vis_semi_span * cos(vis_sweep)
	_add(parent, _sphere(0.06), RED, Vector3(tip_x, 0.18, tip_z))
	_add(parent, _sphere(0.06), GREEN, Vector3(tip_x, 0.18, -tip_z))

	# --- Twin rear-fuselage turbofan nacelles (Citation-style pods) ---
	for zside in [NACELLE_Z_OFFSET, -NACELLE_Z_OFFSET]:
		var pod := _build_turbofan_nacelle()
		pod.position = Vector3(NACELLE_X_OFFSET, -0.55, zside)
		parent.add_child(pod)
		# Register fan disks for animation
		var fan_disk := pod.get_node_or_null("FanDisk")
		if fan_disk:
			result["propellers"].append(fan_disk)

	# --- Trailing-edge control surfaces: inboard flaps + outboard ailerons ---
	# Flaps: inboard, on wing root section
	var flap_mesh := _box(Vector3(0.60, 0.05, 2.2))
	# Right flap (negative Z side? Wait: right wing is -Z in our convention... Let me check)
	# Wing at Z=4.0 is right wing (positive Z is right in Godot? No: +Z is right, so wing at Z=4.0 is right)
	# Actually in Godot: +X forward, +Y up, +Z right. So right wing is +Z.
	var flap_r := _pivot(parent, Vector3(0.9, 0.10, 2.9), flap_mesh, "FLAP_R")
	flap_r.rotation = Vector3(0, -vis_sweep, 0)
	var flap_l := _pivot(parent, Vector3(0.9, 0.10, -2.9), flap_mesh.duplicate(), "FLAP_L")
	flap_l.rotation = Vector3(0, vis_sweep, 0)
	result["flaps"] = [flap_r, flap_l]

	# Ailerons: outboard, near wingtips
	var aileron_mesh := _box(Vector3(0.60, 0.05, 2.0))
	var ail_r := _pivot(parent, Vector3(-0.8, 0.10, 6.4), aileron_mesh, "AILERON_R")
	ail_r.rotation = Vector3(0, -vis_sweep, 0)
	var ail_l := _pivot(parent, Vector3(-0.8, 0.10, -6.4), aileron_mesh.duplicate(), "AILERON_L")
	ail_l.rotation = Vector3(0, vis_sweep, 0)
	result["ailerons"] = [ail_r, ail_l]

	return result

## Build a complete turbofan nacelle with animated fan disk
static func _build_turbofan_nacelle() -> Node3D:
	var pod := Node3D.new()
	pod.name = "Nacelle"

	# Nacelle body (cylinder along X)
	_add(pod, _cyl(NACELLE_RADIUS, NACELLE_LEN), NACELLE, Vector3.ZERO, Vector3(0, 0, PI / 2))

	# Intake lip (forward)
	_add(pod, _cyl(NACELLE_RADIUS * 1.08, 0.22), DARK, Vector3(-NACELLE_LEN/2 + 0.11, 0, 0), Vector3(0, 0, PI / 2))

	# Exhaust nozzle (aft)
	_add(pod, _cyl(NACELLE_RADIUS * 0.8, 0.30), DARK, Vector3(NACELLE_LEN/2 - 0.15, 0, 0), Vector3(0, 0, PI / 2))

	# --- Fan disk (animated) - at intake face ---
	var fan_disk := Node3D.new()
	fan_disk.name = "FanDisk"
	fan_disk.position = Vector3(-NACELLE_LEN/2 + 0.12, 0, 0)
	pod.add_child(fan_disk)

	# Fan spinner
	_add(fan_disk, _cone(0.12, 0.15), ENGINE_CORE, Vector3(0.08, 0, 0))

	# Fan blades (24 blades for high-bypass turbofan look)
	var num_blades := 24
	for i in num_blades:
		var angle := i * 2.0 * PI / num_blades
		var blade := _box(Vector3(0.008, 0.36, 0.025))
		var blade_mi := _add(fan_disk, blade, FAN_BLADE, Vector3.ZERO)
		blade_mi.rotation = Vector3(0, angle, 0)
		blade_mi.name = "FanBlade_%d" % i

	# Fan stators (stationary, behind fan)
	var stator := Node3D.new()
	stator.name = "Stator"
	stator.position = Vector3(0.15, 0, 0)
	fan_disk.add_child(stator)
	for i in 12:
		var angle := i * 2.0 * PI / 12
		var vane := _box(Vector3(0.006, 0.30, 0.02))
		var vane_mi := _add(stator, vane, DARK, Vector3.ZERO)
		vane_mi.rotation = Vector3(0, angle, 0)

	# Core compressor stages (visual only, inside nacelle)
	var core := Node3D.new()
	core.name = "Core"
	core.position = Vector3(0.5, 0, 0)
	pod.add_child(core)
	for stage in 3:
		var x_pos := 0.5 + stage * 0.4
		var r := NACELLE_RADIUS * 0.45 * (1.0 - stage * 0.1)
		_add(core, _cyl(r, 0.15), ENGINE_CORE, Vector3(x_pos, 0, 0), Vector3(0, 0, PI / 2))

	return pod

## Build a tapered wing box (single BoxMesh approximating trapezoidal planform)
static func _tapered_wing_box(semi_span: float, root_chord: float, tip_chord: float, thickness: float) -> BoxMesh:
	# Average chord for box approximation
	var avg_chord := (root_chord + tip_chord) * 0.5
	var b := BoxMesh.new()
	b.size = Vector3(avg_chord, thickness, semi_span * 2.0)
	return b

## Surface pivot with mesh child positioned aft so deflection moves trailing edge
static func _pivot(parent: Node3D, pos: Vector3, mesh: Mesh, label: String) -> Node3D:
	var pivot := Node3D.new()
	pivot.name = label
	pivot.position = pos
	parent.add_child(pivot)
	var mi := _add(pivot, mesh, DARK, Vector3(-0.30, 0, 0))
	mi.name = label + "-Mesh"
	return pivot

# --- Geometry helpers ---------------------------------------------------------

static func _add(
	parent: Node3D, mesh: Mesh, color: Color,
	pos := Vector3.ZERO, rot := Vector3.ZERO, scale := Vector3(1, 1, 1)
) -> MeshInstance3D:
	var mi := MeshInstance3D.new()
	mi.mesh = mesh
	mi.position = pos
	mi.rotation = rot
	mi.scale = scale
	var mat := StandardMaterial3D.new()
	mat.albedo_color = color
	mat.roughness = 0.35
	mat.metallic = 0.05
	mi.material_override = mat
	parent.add_child(mi)
	return mi

static func _box(size: Vector3) -> BoxMesh:
	var b := BoxMesh.new()
	b.size = size
	return b

static func _sphere(radius: float) -> SphereMesh:
	var s := SphereMesh.new()
	s.radius = radius
	s.height = radius * 2.0
	return s

static func _cyl(radius: float, height: float) -> CylinderMesh:
	var c := CylinderMesh.new()
	c.top_radius = radius
	c.bottom_radius = radius
	c.height = height
	return c

static func _cone(radius: float, height: float) -> CylinderMesh:
	var c := CylinderMesh.new()
	c.top_radius = 0.0
	c.bottom_radius = radius
	c.height = height
	return c