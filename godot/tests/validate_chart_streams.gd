extends SceneTree
## Headless verification that the RL live chart actually streams telemetry.
## Spawns a short CPU training via the same code path as the RL tab, waits
## for it to finish, then asserts the in-engine chart ingested records from
## the run's metrics.jsonl. Also covers the re-resolution that keeps the
## chart tracking the newest run (and updates while idle, after the trainer
## process exits).

var _trainer: Control = null
var _setup_done := false
var _started := false
var _settle := 0.0
var _elapsed := 0.0
const TIMEOUT := 300.0


func _setup() -> void:
	_trainer = load("res://scripts/rl_trainer.gd").new()
	root.add_child(_trainer)
	# Wait a frame so _ready() has built the UI.
	await process_frame
	var spins: Array = _trainer._fields_container.get_children()
	spins[1].value = 2     # iterations
	spins[3].value = 64    # rollout steps
	spins[9].value = 32    # minibatch
	_trainer._backend_option.select(1)  # cpu (deterministic + fast, like the e2e)
	print("CHART: binary = ", _trainer._find_rl_binary())
	print("CHART: args   = ", _trainer._build_args(false))
	_trainer._on_start_pressed()


func _process(delta: float) -> bool:
	if not _setup_done:
		_setup_done = true
		_setup()
		return false
	_elapsed += delta
	if _elapsed > TIMEOUT:
		print("CHART: TIMEOUT after %.0f s (started=%s)" % [_elapsed, str(_started)])
		quit(1)
		return true
	if _trainer._training_pid != 0:
		_started = true
		return false
	if not _started:
		return false  # trainer subprocess hasn't spawned yet
	if _settle < 3.0:
		# Trainer exited; a few more frames so the always-on poll tail-ends
		# the final records into the chart.
		_settle += delta
		return false
	var points: Array = _trainer._points.get("mean_reward_per_step", [])
	var env_points: Array = _trainer._points.get("env_alt_mean", [])
	var chart_len: int = _trainer._chart.points.size()
	print("CHART: run_id     = ", _trainer._telemetry_run_id)
	print("CHART: chart title= ", _trainer._chart.title)
	print("CHART: stored pts = ", points.size(), " | chart pts = ", chart_len)
	print("CHART: env pts    = ", env_points.size())
	if points.size() >= 2 and chart_len >= 2:
		print("CHART: first/last = ", points[0], " / ", points[points.size() - 1])
		if env_points.size() >= 2:
			print("CHART: env first/last = ", env_points[0], " / ", env_points[env_points.size() - 1])
			print("CHART: PASS — telemetry streamed from metrics.jsonl into the live chart")
			quit(0)
			return true
		print("CHART: FAIL — env telemetry never reached the chart")
		quit(1)
		return true
	print("CHART: FAIL — telemetry never reached the chart")
	quit(1)
	return true