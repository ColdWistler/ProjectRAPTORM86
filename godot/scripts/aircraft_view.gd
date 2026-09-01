extends Node3D

## Loads a `.glb` aircraft under this node, aligned nose-forward along +X,
## up +Y and right +Z, scaled to a usable size. Each aircraft can carry a
## per-model alignment (rotation + scale) so the arbitrary Sketchfab axis
## conventions are corrected in one place. Swap models at runtime with
## `set_model(name)` (e.g. `"MQI"` or `"TwinEngine"`).

const MODEL_PATH := "res://assets/aircraft/"

## Per-aircraft visual alignment. `rotation_deg` is applied to the model to
## bring its nose onto +X; `scale` normalizes its size to the physics scale.
## These are best-guess from the raw model extents and may need tuning — tweak
## here rather than in code.
const ALIGN := {
	"MQI": {
		"scene": "MQI.glb",
		"rotation_deg": Vector3(0, 90, 0),
		"scale": Vector3(0.35, 0.35, 0.35),
		"propellers": ["PROPELLER"],
		"ailerons": ["AILERON"],
		"flaps": [],
	},
	"TwinEngine": {
		"scene": "TwinEngine.glb",
		"rotation_deg": Vector3(0, 0, 0),
		"scale": Vector3(0.16, 0.16, 0.16),
		"propellers": ["Cylinder", "Cone"],
		"ailerons": [],
		"flaps": [],
	},
}

var _current_name := ""
var _instance: Node3D = null
var _align_root: Node3D = null
var propellers: Array = []
var ailerons: Array = []
var flaps: Array = []

func _ready() -> void:
	pass

## Depth-first search for geometry (MeshInstance3D) nodes whose name matches a
## substring pattern (case-insensitive). Only leaf mesh nodes are returned so we
## animate the actual geometry rather than rotating a parent transform that
## already contains the child (which would double-rotate it).
func _find_nodes(root: Node, patterns: Array) -> Array:
	var out: Array = []
	if root == null:
		return out
	for child in root.get_children():
		if child is MeshInstance3D:
			var lower := child.name.to_lower()
			for p in patterns:
				if p.to_lower() in lower:
					out.append(child)
					break
		out.append_array(_find_nodes(child, patterns))
	return out

## Replace the currently displayed aircraft with `name`'s model. Returns `false`
## if the model could not be loaded.
func set_model(name: String) -> bool:
	if not ALIGN.has(name):
		push_error("AircraftView: unknown aircraft '%s'" % name)
		return false
	var align: Dictionary = ALIGN[name]

	# Already showing this model.
	if _current_name == name and _instance != null:
		return true

	if _align_root == null:
		_align_root = Node3D.new()
		add_child(_align_root)
	elif _current_name != "":
		# Clear the previous loaded model (keep the alignment root).
		for child in _align_root.get_children():
			_align_root.remove_child(child)
			child.queue_free()

	var scene_path := MODEL_PATH + str(align["scene"])
	if not ResourceLoader.exists(scene_path):
		push_error("AircraftView: missing model '%s' (expected at '%s')" % [align["scene"], scene_path])
		return false

	var packed: PackedScene = load(scene_path)
	_instance = packed.instantiate()
	if _instance == null:
		push_error("AircraftView: failed to instantiate '%s'" % scene_path)
		return false

	_instance.rotation_degrees = align["rotation_deg"]
	_instance.scale = align["scale"]
	_align_root.add_child(_instance)

	_current_name = name

	# Re-discover the animated sub-parts (propellers / ailerons / flaps).
	propellers = _find_nodes(_instance, align.get("propellers", []))
	ailerons = _find_nodes(_instance, align.get("ailerons", []))
	flaps = _find_nodes(_instance, align.get("flaps", []))
	return true

func current_name() -> String:
	return _current_name
