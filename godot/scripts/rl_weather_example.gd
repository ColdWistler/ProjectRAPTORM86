extends Node
## RL weather training-loop stub in GDScript.
##
## Attach this to any scene (or autoload) and add a `WeatherSystem` node as a
## child or autoload. It demonstrates the full RL surface the weather plugin
## exposes: observation vector, reward penalties, termination, curriculum and
## agent actions, plus optionally feeding the wind into a FlightSim node.
##
## The real training loop normally lives in Rust (see
## flight_core/examples/rl_weather_loop.rs); this script is the Godot-side
## wiring / instrumentation example.

const OBS_WEATHER_DIM := 12
const MAX_EPISODE_STEPS := 1200

## Child/autoload WeatherSystem node (resolved in _ready).
var weather: Node = null
## Optional FlightSimNode to fly through the weather wind.
var flight_sim: Node = null

var step_index := 0
var episode_reward := 0.0
var episode_weather_penalty := 0.0
var episodes_completed := 0
var total_steps := 0

func _ready() -> void:
	weather = get_node_or_null("WeatherSystem")
	if weather == null:
		weather = get_node_or_null("/root/WeatherSystem")
	flight_sim = get_node_or_null("FlightSim")

	if weather == null:
		push_warning("rl_weather_example.gd: no WeatherSystem node found; add one as autoload or a child.")
		set_process(false)
		return

	# --- curriculum: clear → light turbulence → rain → storm ---
	weather.set_curriculum(2024, 600)            # seed, steps per phase
	weather.weather_changed.connect(_on_weather_changed)
	weather.set_reference_altitude(800.0)
	weather.set_airspeed(60.0)

	print("rl_weather_example ready — phase ",
		weather.get_curriculum_phase(), "/3 (0 clear, 1 turb, 2 rain, 3 storm)")

func _physics_process(delta: float) -> void:
	if weather == null:
		return

	# 1. Step the weather scene (dry physics; the drone sim steps separately).
	weather.step(delta)

	# 2. Collect the 12-channel observation for the policy.
	var obs: PackedFloat64Array = weather.get_observation()
	if obs.size() != OBS_WEATHER_DIM:
		push_error("weather observation must be 12 channels")

	# 3. Shaped reward: survival bonus + weather penalties (already ≤ 0).
	var step_reward := delta * 1.0 + weather.get_reward_penalty_total()
	episode_reward += step_reward
	episode_weather_penalty += weather.get_reward_penalty_total()
	step_index += 1
	total_steps += 1

	# 4. Trivial heuristic policy (stand-in for a learned policy):
	#    evasive maneuver when turbulence reaches extreme; climb when rain.
	if weather.get_turbulence_level() >= 3:
		weather.trigger_evasive_maneuver()
	if weather.get_rain_intensity() > 0.5 and not weather.is_evasive():
		weather.set_severity(3) # storm band (severity modulation action)

	# 5. Termination: ground crash handled by the flight sim; extreme weather
	#    termination comes from the weather node itself (cfg-gated in Rust).
	if weather.is_terminated():
		_episode_done(weather.check_termination())

	# 6. Fly the drone through the weather wind (optional, requires FlightSim).
	if flight_sim != null and flight_sim.is_ready():
		var w: Vector3 = weather.get_wind_vector() # world (north, -down, east)
		flight_sim.set_external_wind(true, w.x, w.z, -w.y)

	if step_index >= MAX_EPISODE_STEPS:
		_episode_done(0)

func _episode_done(reason: int) -> void:
	episodes_completed += 1
	print("episode %d done (reason %d): reward %.2f, weather penalty %.2f, phase %d/%d (%.0f%%)" % [
		episodes_completed, reason, episode_reward, episode_weather_penalty,
		weather.get_curriculum_phase(), 3, weather.get_curriculum_progress() * 100.0,
	])
	# --- episode restart: same seed ⇒ same weather (reproducible episodes) ---
	weather.reset()
	weather.set_curriculum(2024 + episodes_completed, 600)
	step_index = 0
	episode_reward = 0.0
	episode_weather_penalty = 0.0
	total_steps = 0

func _on_weather_changed(severity: int) -> void:
	print("weather_changed → severity ", severity)

func _input(event: InputEvent) -> void:
	var k := event as InputEventKey
	if k == null or not k.pressed:
		return
	match k.keycode:
		KEY_W: # weather severity up
			var sev := clampi(weather.get_severity() + 1, 0, 3)
			weather.set_severity(sev)
		KEY_N: # next curriculum phase
			var phase: int = weather.next_curriculum_phase()
			print("curriculum phase → ", phase)
		KEY_M: # evasive maneuver
			weather.trigger_evasive_maneuver()
			print("evasive maneuver triggered")