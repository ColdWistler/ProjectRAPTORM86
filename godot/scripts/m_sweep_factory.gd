class_name MSweepFactory
extends RefCounted
## Procedural M-SWEEP Mini: variable-sweep canard/pusher with skyhook
## Built along +X forward, up +Y, right +Z to match physics frame
## Dimensions match m_sweep_mini.toml: span 2.7m, area 1.22m², chord 0.452m

const BODY := Color(0.20, 0.22, 0.26)
const WING := Color(0.18, 0.21, 0.25)
const INNER := Color(0.19, 0.22, 0.26)
const TAIL := Color(0.17, 0.19, 0.23)
const DARK := Color(0.10, 0.12, 0.15)
const GLASS := Color(0.12, 0.14, 0.18)
const METAL := Color(0.16, 0.18, 0.22)
const HOOK := Color(0.95, 0.65, 0.20)
const PROP_BLADE := Color(0.05, 0.06, 0.08)
const ACCENT := Color(0.60, 0.45, 0.20)
const NAV_RED := Color(1.0, 0.2, 0.2)
const NAV_GREEN := Color(0.2, 1.0, 0.3)

# TOML dimensions (metres)
const FUSELAGE_LEN := 2.15
const FUSELAGE_RADIUS := 0.155
const INNER_HALFSPAN := 0.38
const OUTER_HALFSPAN := 0.97
const WING_CHORD_ROOT := 0.48
const WING_CHORD_TIP := 0.32
const WING_THICKNESS := 0.038
const SWEEP_MIN_DEG := 0.0
const SWEEP_MAX_DEG := 24.0

# Pre-calculated fuselage station positions (continuous, no gaps)
const NOSE_TIP_X := 1.075
const NOSE_BASE_X := NOSE_TIP_X - 0.45
const NOSE_CENTER_X := (NOSE_TIP_X + NOSE_BASE_X) * 0.5
const FWD_CENTER_X := 0.125
const AFT_CENTER_X := -0.775
const TAIL_BOOM_CENTER_X := -1.65
const TAIL_BOOM2_CENTER_X := -2.475
const TAIL_BOOM2_END_X := -2.825

static func build(parent: Node3D) -> Dictionary:
	var result := {
		"propellers": [],
		"flaps": [],
		"ailerons": [],
		"wing_sweep_pivots": [],
	}

	# --- Fuselage: continuous nose-to-tail body (NO GAPS) ---
	# Nose cone pointing forward (+X) - fully attached
	var nose = _add(parent, _cone(FUSELAGE_RADIUS, 0.5), BODY, Vector3(NOSE_TIP_X - 0.25, 0.0, 0), Vector3(0, 0, -PI / 2))
	var nose_m = nose.mesh
	if nose_m is CylinderMesh:
		nose_m.top_radius = 0.0
		nose_m.bottom_radius = FUSELAGE_RADIUS
		nose_m.height = 0.5
	_add(parent, _cyl(FUSELAGE_RADIUS, 1.1), BODY, Vector3(0.075, 0.0, 0), Vector3(0, 0, PI / 2))
	_add(parent, _cyl(FUSELAGE_RADIUS * 0.95, 0.9), BODY, Vector3(-0.725, 0.0, 0), Vector3(0, 0, PI / 2))
	_add(parent, _cyl(0.048, 1.0), METAL, Vector3(TAIL_BOOM_CENTER_X, 0.0, 0), Vector3(0, 0, PI / 2))
	_add(parent, _cyl(0.042, 0.75), METAL, Vector3(TAIL_BOOM2_CENTER_X, 0.02, 0), Vector3(0, 0, PI / 2))

	# --- Cockpit canopy ---
	_add(parent, _box(Vector3(0.12, 0.08, 0.16)), GLASS, Vector3(0.5, 0.06, 0))
	# Chin bay / sensor housing
	_add(parent, _box(Vector3(0.18, 0.04, 0.14)), DARK, Vector3(0.35, -0.09, 0))

	# --- Single solid wing attached to fuselage, extending laterally ---
	var sweep_vis_deg := 0.0
	var sweep_vis := deg_to_rad(sweep_vis_deg)
	# Left wing (extends in -Z, attached to central fuselage body, horizontal)
	_add(parent, _tapered_wing(Vector3(0.48, WING_THICKNESS, 0.9), Vector3(0.32, WING_THICKNESS, 0.5)), WING, Vector3(0.05, 0.0, -0.7), Vector3(0, 0, 0))
	# Right wing (extends in +Z, attached to central fuselage body, horizontal)
	_add(parent, _tapered_wing(Vector3(0.48, WING_THICKNESS, 0.9), Vector3(0.32, WING_THICKNESS, 0.5)), WING, Vector3(0.05, 0.0, 0.7), Vector3(0, 0, 0))
	var wing_l := parent.get_child(parent.get_child_count() - 2)
	var wing_r := parent.get_child(parent.get_child_count() - 1)
	result["wing_sweep_pivots"].append({"pivot": wing_l, "side": -1, "base_sweep": sweep_vis})
	result["wing_sweep_pivots"].append({"pivot": wing_r, "side": 1, "base_sweep": sweep_vis})

	# Winglets with nav lights (like Skylark)
	var winglet_l := _add(wing_l, _box(Vector3(0.03, 0.18, 0.18)), ACCENT, Vector3(-0.15, 0.12, -0.65), Vector3(0, 0, 0.3))
	winglet_l.name = "Winglet_L"
	var winglet_r := _add(wing_r, _box(Vector3(0.03, 0.18, 0.18)), ACCENT, Vector3(-0.15, 0.12, 0.65), Vector3(0, 0, -0.3))
	winglet_r.name = "Winglet_R"

	# Nav lights on winglet tips
	_add(wing_l, _sphere(0.006), NAV_RED, Vector3(-0.05, 0.15, -0.55))
	_add(wing_r, _sphere(0.006), NAV_GREEN, Vector3(-0.05, 0.15, 0.55))

	# --- Ailerons on outer wing trailing edges (attached to wings) ---
	var ail_chord := 0.12
	var ail_span := 0.4
	var ail_mesh_l := _box(Vector3(ail_span, WING_THICKNESS * 0.9, ail_chord))
	var ail_pivot_l := _pivot(wing_l, Vector3(-0.15, 0.0, -0.32), ail_mesh_l, "AILERON_L")
	ail_pivot_l.rotation = Vector3(0, 0, 0)
	result["ailerons"].append(ail_pivot_l)

	var ail_mesh_r := _box(Vector3(ail_span, WING_THICKNESS * 0.9, ail_chord))
	var ail_pivot_r := _pivot(wing_r, Vector3(-0.15, 0.0, 0.32), ail_mesh_r, "AILERON_R")
	ail_pivot_r.rotation = Vector3(0, 0, 0)
	result["ailerons"].append(ail_pivot_r)

	# --- Twin tail booms (like Skylark) ---
	for side in [-1, 1]:
		var boom := _add(parent, _cyl(0.055, 2.4), DARK, Vector3(-0.8, 0.18, side * 0.85), Vector3(0, 0, PI/2))
		boom.name = "TailBoom_%s" % ("R" if side > 0 else "L")
		
		var fin := _add(parent, _box(Vector3(0.55, 0.75, 0.06)), WING, Vector3(-1.85, 0.55, side * 0.85))
		fin.name = "Fin_%s" % ("R" if side > 0 else "L")

	# --- Horizontal stabilizer connecting tail booms ---
	var stab := _add(parent, _box(Vector3(0.6, 0.06, 1.9)), WING, Vector3(-1.7, 0.18, 0))
	stab.name = "Stabilizer"

	# --- Elevators on stabilizer (move with elevator input) ---
	var elev_mesh := _box(Vector3(0.22, 0.06, 0.85))
	var elev_l := _pivot(parent, Vector3(-1.9, 0.18, -0.6), elev_mesh, "ELEVATOR_L")
	var elev_r := _pivot(parent, Vector3(-1.9, 0.18, 0.6), elev_mesh.duplicate(), "ELEVATOR_R")
	result["flaps"] = [elev_l, elev_r]

	# --- Pusher motor and propeller (2-blade like Skylark) ---
	var motor := _add(parent, _cyl(0.12, 0.4), DARK, Vector3(-1.5, 0.18, 0), Vector3(0, 0, PI/2))
	motor.name = "Motor"

	var prop_pivot := Node3D.new()
	prop_pivot.name = "PROP"
	prop_pivot.position = Vector3(-2.0, 0.18, 0)
	parent.add_child(prop_pivot)

	var hub := _add(prop_pivot, _sphere(0.05), BODY, Vector3.ZERO)
	hub.name = "PropHub"

	for blade_side in [-1, 1]:
		var blade := _add(prop_pivot, _box(Vector3(0.06, 0.5, 0.025)), PROP_BLADE, Vector3(0.0, blade_side * 0.25, 0), Vector3(0, 0, blade_side * 0.2))
		blade.name = "PropBlade_%s" % ("R" if blade_side > 0 else "L")
		blade.cast_shadow = GeometryInstance3D.SHADOW_CASTING_SETTING_ON

	result["propellers"] = [prop_pivot]

	# --- Underwing cargo pod (like Skylark) ---
	var pod := _add(parent, _capsule(0.22, 0.7), DARK, Vector3(0.1, -0.32, 0), Vector3(PI/2, 0, 0), Vector3(0.9, 1.0, 0.75))
	pod.name = "CargoPod"
	
	var pod_band := _add(parent, _box(Vector3(0.4, 0.04, 0.1)), ACCENT, Vector3(0.0, -0.12, 0.15))
	pod_band.name = "PodBand"

	# --- SkyHook on wing shoulder (like Skylark) ---
	var hook_pylon := _add(parent, _cyl(0.035, 0.22), DARK, Vector3(-0.05, 0.42, -0.25))
	hook_pylon.name = "HookPylon"

	var hook := _add(parent, _torus(0.12, 0.028), ACCENT, Vector3(-0.05, 0.62, -0.25), Vector3(0, 0, -PI * 0.4))
	hook.name = "Hook"

	for side in [-1, 1]:
		var horn := _add(parent, _cyl(0.014, 0.18), ACCENT, Vector3(side * 0.11, 0.78, -0.25), Vector3(0, 0, side * 0.35), Vector3(1,1,1))
		horn.name = "Horn_%s" % ("R" if side > 0 else "L")

	var strop := _add(parent, _cyl(0.008, 0.35), DARK, Vector3(-0.05, 0.38, -0.25), Vector3.ZERO, Vector3(1,1,1))
	strop.material_override.transparency = BaseMaterial3D.TRANSPARENCY_ALPHA
	strop.material_override.alpha_scissor_threshold = 0.01
	strop.name = "Strop"

	# --- Display sting mount (like Skylark) ---
	var sting := _add(parent, _cyl(0.025, 1.6), DARK, Vector3(0.0, -1.2, 0.15))
	sting.name = "Sting"
	
	var collar := _add(parent, _cyl(0.07, 0.12), BODY, Vector3(0.0, -0.5, 0.15))
	collar.name = "Collar"

	# --- Navigation lights ---
	_add(parent, _sphere(0.008), NAV_RED, Vector3(-0.28, 0.03, 1.28))
	_add(parent, _sphere(0.008), NAV_GREEN, Vector3(-0.28, 0.03, -1.28))
	_add(parent, _sphere(0.006), Color(0.9, 0.9, 0.9), Vector3(NOSE_TIP_X + 0.05, 0.0, 0))

	return result

static func _tapered_wing(root_size: Vector3, tip_size: Vector3) -> BoxMesh:
	var b := BoxMesh.new()
	b.size = root_size
	return b

static func _pivot(parent: Node3D, pos: Vector3, mesh: Mesh, label: String) -> Node3D:
	var pivot := Node3D.new()
	pivot.name = label
	pivot.position = pos
	parent.add_child(pivot)
	var mi := _add(pivot, mesh, DARK, Vector3(-0.12, 0, 0))
	mi.name = label + "-Mesh"
	return pivot

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
	mat.roughness = 0.65
	mat.metallic = 0.1
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

static func _capsule(radius: float, height: float) -> BoxMesh:
	var b := BoxMesh.new()
	b.size = Vector3(radius * 2.0, radius * 2.0, height)
	return b

static func _torus(major: float, minor: float) -> BoxMesh:
	var b := BoxMesh.new()
	b.size = Vector3(major * 2.0, minor * 2.0, minor * 2.0)
	return b