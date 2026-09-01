class_name DroneFactory
extends RefCounted
## Procedural tactical UAV that matches the `flight_core` geometry assumed by
## the wind-tunnel flow field (fuselage, canopy, wing, V-tail, nacelle, prop).
## Built nose-forward along +X, up +Y, right +Z — the same convention
## `FlightSimNode`/`WindTunnelNode` use for their transforms.

const BODY := Color(0.62, 0.66, 0.72)
const DARK := Color(0.24, 0.26, 0.30)
const CONTROL := Color(0.45, 0.52, 0.46)
const PROP_COLOR := Color(0.78, 0.12, 0.10)
const GLASS := Color(0.12, 0.14, 0.16)
const RED := Color(0.95, 0.12, 0.08)
const GREEN := Color(0.10, 0.90, 0.20)

## Build the drone under `parent`. Returns a Dictionary:
##   "propeller": Node3D to spin about its local X
##   "flaps":     [Node3D, Node3D] pivots to deflect about local X
##   "ailerons":  [Node3D, Node3D] pivots to deflect about local X
static func build(parent: Node3D) -> Dictionary:
	var result := {
		"propeller": null,
		"flaps": [],
		"ailerons": [],
	}

	# --- Fuselage & nose ---
	_add(parent, _box(Vector3(4.2, 0.75, 0.75)), BODY)
	var dome := _add(parent, _sphere(0.48), GLASS, Vector3(2.1, 0.12, 0))
	dome.scale = Vector3(1.3, 1.05, 1.0)
	_add(parent, _box(Vector3(1.8, 0.42, 0.62)), GLASS, Vector3(0.9, 0.42, 0))
	_add(parent, _sphere(0.24), DARK, Vector3(1.7, -0.42, 0))
	_add(parent, _cyl(0.10, 0.08), Color(0.9, 0.9, 0.98), Vector3(1.86, -0.46, 0), Vector3(0, 0, PI / 2 - 0.2))
	_add(parent, _cyl(0.015, 0.60), DARK, Vector3(2.85, 0, 0), Vector3(0, 0, PI / 2))
	_add(parent, _box(Vector3(0.8, 0.25, 0.35)), DARK, Vector3(-1.1, 0.46, 0))

	# --- Main wing & leading-edge spar ---
	_add(parent, _box(Vector3(0.85, 0.09, 10.6)), BODY, Vector3(0.25, 0.15, 0))
	_add(parent, _box(Vector3(0.20, 0.07, 10.6)), DARK, Vector3(0.68, 0.15, 0))

	# --- Winglets (angled tips) + nave lights (red +Z, green -Z) ---
	var winglet := _box(Vector3(0.40, 0.35, 0.06))
	_add(parent, winglet.duplicate(), DARK, Vector3(0.25, 0.30, 5.3), Vector3(-0.25, 0, 0))
	_add(parent, winglet.duplicate(), DARK, Vector3(0.25, 0.30, -5.3), Vector3(0.25, 0, 0))
	_add(parent, _sphere(0.04), RED, Vector3(0.25, 0.18, 5.32))
	_add(parent, _sphere(0.04), GREEN, Vector3(0.25, 0.18, -5.32))

	# --- Trailing-edge movable surfaces ---
	# Left/right inboard flaps and outboard ailerons on pivots. The child mesh
	# sits slightly aft of the pivot so rotating the pivot bends the surface.
	var flap_mesh := _box(Vector3(0.32, 0.06, 2.0))
	var flap_left := _pivot(parent, Vector3(1.8, 0.14, 0), flap_mesh)
	var flap_right := _pivot(parent, Vector3(-1.8, 0.14, 0), flap_mesh.duplicate())
	result["flaps"] = [flap_left, flap_right]

	var aileron_mesh := _box(Vector3(0.32, 0.06, 2.4))
	var ail_left := _pivot(parent, Vector3(4.0, 0.14, 0), aileron_mesh)
	var ail_right := _pivot(parent, Vector3(-4.0, 0.14, 0), aileron_mesh.duplicate())
	result["ailerons"] = [ail_left, ail_right]

	# --- V-tail empennage (canted ~38°) ---
	var fin := _box(Vector3(0.75, 1.35, 0.08))
	_add(parent, fin.duplicate(), BODY, Vector3(-1.85, 0.65, 0.55), Vector3(0.62, 0, 0))
	_add(parent, fin.duplicate(), BODY, Vector3(-1.85, 0.65, -0.55), Vector3(-0.62, 0, 0))
	var rv := _box(Vector3(0.28, 1.25, 0.06))
	_add(parent, rv.duplicate(), CONTROL, Vector3(-2.39, 0.65, 0.55), Vector3(0.62, 0, 0))
	_add(parent, rv.duplicate(), CONTROL, Vector3(-2.39, 0.65, -0.55), Vector3(-0.62, 0, 0))

	# --- Rear pusher engine ---
	_add(parent, _cyl(0.20, 0.30), DARK, Vector3(-2.25, 0.08, 0), Vector3(0, 0, PI / 2))
	var prop := Node3D.new()
	prop.position = Vector3(-2.38, 0.08, 0)
	parent.add_child(prop)
	_add(prop, _box(Vector3(0.04, 1.45, 0.12)), PROP_COLOR)
	result["propeller"] = prop

	return result

# --- Geometry helpers --------------------------------------------------------

static func _add(
	parent: Node3D, mesh: Mesh, color: Color,
	pos := Vector3.ZERO, rot := Vector3.ZERO
) -> MeshInstance3D:
	var mi := MeshInstance3D.new()
	mi.mesh = mesh
	mi.position = pos
	mi.rotation = rot
	var mat := StandardMaterial3D.new()
	mat.albedo_color = color
	mi.material_override = mat
	parent.add_child(mi)
	return mi

## Surface pivot with a small mesh child positioned aft (-X) so deflecting the
## pivot about X moves the trailing edge down/up like a real control surface.
static func _pivot(parent: Node3D, pos: Vector3, mesh: Mesh) -> Node3D:
	var pivot := Node3D.new()
	pivot.position = pos
	parent.add_child(pivot)
	_add(pivot, mesh, CONTROL, Vector3(-0.16, 0, 0))
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
