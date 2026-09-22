extends SceneTree
## Temporary headless e2e for the RL menu panel: instantiate rl_trainer.gd,
## shrink the hyper-parameters, press Start, wait for the spawned rl_agent
## binary to finish, then report the status + tail of the log.

var _trainer: Control = null
var _setup_done := false
var _started := false
var _elapsed := 0.0
const TIMEOUT := 300.0


func _setup() -> void:
	_trainer = load("res://scripts/rl_trainer.gd").new()
	root.add_child(_trainer)
	# Wait a frame so _ready() has built the UI.
	await process_frame
	var spins: Array = _trainer._fields_container.get_children()
	# Child order is [label, spin, label, spin, ...]; indexes 1/3/5/9 are the
	# first four spins (iterations, rollout, minibatch at 9).
	spins[1].value = 2     # iterations
	spins[3].value = 64    # rollout steps
	spins[9].value = 32    # minibatch
	# Keep the spawn deterministic and fast: CPU backend (the GPU JIT path is
	# exercised separately by the CLI smoke tests).
	_trainer._backend_option.select(1)  # cpu
	print("E2E: binary = ", _trainer._find_rl_binary())
	print("E2E: args   = ", _trainer._build_args(false))
	_trainer._on_start_pressed()


func _process(delta: float) -> bool:
	if not _setup_done:
		_setup_done = true
		_setup()
		return false
	if _started and _trainer._training_pid == 0:
		print("E2E: finished -> ", _trainer._status.text)
		var f := FileAccess.open("user://rl_training.log", FileAccess.READ)
		if f != null:
			var lines := f.get_as_text().split("\n")
			print("E2E: log tail:")
			for i in range(maxi(0, lines.size() - 8), lines.size()):
				print("E2E: | ", lines[i])
			f.close()
		quit(0)
		return true
	if _trainer._training_pid != 0:
		_started = true
	_elapsed += delta
	if _elapsed > TIMEOUT:
		print("E2E: TIMEOUT after %.0f s (pid=%d, started=%s)" % [_elapsed, _trainer._training_pid, str(_started)])
		quit(1)
		return true
	return false