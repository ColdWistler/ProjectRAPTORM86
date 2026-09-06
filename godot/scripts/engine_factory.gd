class_name EngineFactory
extends RefCounted
## Procedural "normal aircraft" for ENGINE MODE: a swept-wing twin-turbofan
## light jet (airliner-inspired, NOT a UAV drone). Built nose-forward along +X,
## up +Y, right +Z — the same convention FlightSimNode / WindTunnelNode use for
## their transforms, matching the engine-airframe geometry in `Engine.toml`
## (span ~16.5 m, two rear-fuselage turbofan pods, ailerons + flaps).

const BODY := Color(0.88, 0.90, 0.93)   # airliner white
const BELLY := Color(0.62, 0.66, 0.72)
const DARK := Color(0.15, 0.17, 0.20)
const GLASS := Color(0.08, 0.10, 0.13)
const NACELLE := Color(0.72, 0.74, 0.78)
const RED := Color(0.95, 0.12, 0.08)
const GREEN := Color(0.10, 0.90, 0.20)

## Build the jet under `parent`. Returns a Dictionary:
##   "propellers": []               (turbofans do not spin)
##   "flaps":  [Node3D, Node3D]     pivots to deflect about local X
##   "ailerons":  [Node3D, Node3D]  pivots to deflect about local X
static func build(parent: Node3D) -> Dictionary:
	var result := {
		"propellers": [],
		"flaps": [],
		"ailerons": [],
	}

	# --- Fuselage: cabin tube + rounded nose + tail cone (axis along X) ---
	_add(parent, _cyl(0.85, 14.0), BODY, Vector3(0, 0.40, 0), Vector3(0, 0, PI / 2))
	_add(parent, _sphere(0.85), BODY, Vector3(7.3, 0.40, 0), Vector3.ZERO, Vector3(1.35, 1.0, 1.0))
	_add(parent, _sphere(0.80), BELLY, Vector3(-6.9, 0.40, 0), Vector3.ZERO, Vector3(0.65, 1.0, 1.0))

	# --- Cockpit windshield + cabin windows (thin dark slivers on the sides) ---
	_add(parent, _box(Vector3(1.4, 0.55, 1.15)), GLASS, Vector3(5.6, 0.60, 0))
	for wx in [4.6, 3.5, 2.4]:
		_add(parent, _box(Vector3(0.30, 0.22, 0.08)), GLASS, Vector3(wx, 0.52, 0.87))
		_add(parent, _box(Vector3(0.30, 0.22, 0.08)), GLASS, Vector3(wx, 0.52, -0.87))

	# --- Empennage: vertical fin + horizontal stabilizer ---
	_add(parent, _box(Vector3(0.65, 3.4, 0.10)), BODY, Vector3(-6.2, 2.0, 0), Vector3(0, 0, 0.15))
	_add(parent, _box(Vector3(0.10, 0.70, 0.30)), DARK, Vector3(-6.35, 3.4, 0), Vector3(0, 0, 0.15))
	_add(parent, _box(Vector3(2.40, 0.10, 4.2)), BODY, Vector3(-5.4, 1.05, 0))

	# --- Swept main wings (right wing +Z, left wing -Z) ---
	var sweep := 0.50
	_add(parent, _box(Vector3(2.9, 0.12, 7.0)), BODY, Vector3(0.4, 0.12, 4.0), Vector3(0, -sweep, 0))
	_add(parent, _box(Vector3(2.9, 0.12, 7.0)), BODY, Vector3(0.4, 0.12, -4.0), Vector3(0, sweep, 0))

	# Wingtip navigation lights: red right (+Z), green left (-Z).
	var tip_x := 0.4 - 7.0 * sin(sweep)
	var tip_z := 4.0 + 7.0 * cos(sweep)
	_add(parent, _sphere(0.06), RED, Vector3(tip_x, 0.18, tip_z))
	_add(parent, _sphere(0.06), GREEN, Vector3(tip_x, 0.18, -tip_z))

	# --- Twin rear-fuselage turbofan nacelles (Citation-style pods) ---
	for zside in [1.15, -1.15]:
		var pod := _pod()
		pod.position = Vector3(-2.6, -0.55, zside)
		parent.add_child(pod)

	# --- Trailing-edge control surfaces: inboard flaps + outboard ailerons ---
	var flap_mesh := _box(Vector3(0.60, 0.05, 2.2))
	var flap_r := _pivot(parent, Vector3(0.9, 0.10, 2.9), flap_mesh, "FLAP-R")
	var flap_l := _pivot(parent, Vector3(0.9, 0.10, -2.9), flap_mesh.duplicate(), "FLAP-L")
	result["flaps"] = [flap_r, flap_l]

	var aileron_mesh := _box(Vector3(0.60, 0.05, 2.0))
	var ail_r := _pivot(parent, Vector3(-0.8, 0.10, 6.4), aileron_mesh, "AILERON-R")
	var ail_l := _pivot(parent, Vector3(-0.8, 0.10, -6.4), aileron_mesh.duplicate(), "AILERON-L")
	result["ailerons"] = [ail_r, ail_l]

	return result

## Build one turbofan pod: a nacelle cylinder with a dark intake lip forward
## and a nozzle aft. Since it hangs off the rear fuselage, no pylon is drawn.
## Axis along local X.
static func _pod() -> Node3D:
	var pod := Node3D.new()
	_add(pod, _cyl(0.50, 3.4), NACELLE, Vector3.ZERO, Vector3(0, 0, PI / 2))
	_add(pod, _cyl(0.54, 0.22), DARK, Vector3(-1.7 + 0.11, 0, 0), Vector3(0, 0, PI / 2))
	_add(pod, _cyl(0.40, 0.30), DARK, Vector3(1.7 - 0.15, 0, 0), Vector3(0, 0, PI / 2))
	return pod

# --- Geometry helpers (mirroring drone.gd's) -------------------------------

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
	mi.material_override = mat
	parent.add_child(mi)
	return mi

## Surface pivot with a small mesh child positioned aft (-X) so deflecting the
## pivot about X moves the trailing edge down/up like a real control surface.
static func _pivot(parent: Node3D, pos: Vector3, mesh: Mesh, label: String) -> Node3D:
	var pivot := Node3D.new()
	pivot.name = label
	pivot.position = pos
	parent.add_child(pivot)
	var mi := _add(pivot, mesh, DARK, Vector3(-0.30, 0, 0))
	mi.name = label + "-Mesh"
	return pivot

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