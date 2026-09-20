# RL Weather Integration Guide

This document is the RL-facing manual for the weather system that lives in
`flight_core/src/weather.rs` (pure Rust) and its Godot bridge
`flight_gd/src/weather.rs` (a `WeatherSystem` `Node3D`).

The design goal is **library-agnostic**: the weather layer produces plain
arrays of `f64`/`f32` and a few helper structs. Feed them into tch-rs,
rust-rl, PyTorch via a Pipe, or your own PPO/A2C — nothing here is coupled to
an RL framework.

---

## 1. What the agent observes

Every environment step produces a flat **12-channel** vector
(`WEATHER_OBS_DIM`, channel table below). It is available as:

| Accessor | Type | Notes |
|---|---|---|
| `WeatherSystem::observation()` | `WeatherObservation` | structured, all units raw (e.g. visibility in metres) |
| `WeatherSystem::observation_array()` | `[f64; 12]` | compact vector ready for the policy |
| `WeatherSystem::observation_vec_f32()` | `Vec<f32>` | the usual tensor-crate input |
| `WeatherSystem::to_dvector()` | `nalgebra::DVector<f32>` | nalgebra-backed stacks |
| `Environment::weather_observation_array()` | `Option<[f64; 12]>` | `None` when no weather configured |
| `Environment::full_observation()` | `Vec<f64>` | flight (12) ‖ weather (12) concatenated |

Channel layout (`observation_array`):

```text
 0  wind_north (m/s, NED)
 1  wind_east  (m/s, NED)
 2  wind_down  (m/s, NED)
 3  turbulence_01   (0 = light … 1 = extreme, RMS σ / 9 m/s)
 4  rain_intensity  (0..1)
 5  visibility_01   (1 = clear … 0 = zero visibility; scale ref 20 km)
 6  shear_01        (|dW/dh| normalised, 1 = 0.05 (m/s)/m)
 7  updraft_strength (m/s, positive = up)
 8  precipitation rain (0/1)
 9  precipitation snow (0/1)
10  precipitation hail (0/1)
11  weather_severity_01 (0 = Clear … 1 = Severe)
```

### Normalisation

Two options:

* **Already-normalised**: channels 3–6, 8–11 are `[0,1]` by construction;
  winds (0–2) and updraft (7) are raw SI.
* **`ObservationNormalizer`** (`weather.rs`): running per-channel min/max with
  documented physical defaults (wind ±30 m/s, updraft ±20 m/s). Feed it every
  observation and feed the scaler into your trainer, e.g.:

```rust
let mut norm = ObservationNormalizer::from_default_bounds();
// in the loop:
let obs = env.weather_observation_array().unwrap();
let scaled = norm.normalize(&obs); // [0,1] per channel
```

The Godot node additionally exposes `get_observation_normalized()`.

---

## 2. How reward is shaped

All weather penalty functions are **≤ 0**. Add them to your per-step reward
(`Environment::step` already adds them through `reward_penalty_total()`):

| Method | Formula | Meaning |
|---|---|---|
| `weather_penalty()` | `-w_weather · severity_01` | weather-penetration cost |
| `turbulence_penalty()` | `-w_turbulence · (σ/9)²` | gust RMS exposure |
| `visibility_penalty()` | `-w_visibility · (1 − vis_01)²` | lost-visibility cost |
| `wind_penalty()` | `-w_wind · (W/20)²` | strong-wind loading |
| `reward_penalty_total()` | sum | the value the env adds per step |

* Evasive maneuvers halve the total penalty while active.
* On extreme-weather termination add `weather_crash_penalty()` (the weather
  analogue of `crash_penalty()`), mirroring how the existing env test loop
  adds the crash penalty.

Weights live in `WeatherConfig { w_weather, w_turbulence, w_visibility, w_wind }`.

---

## 3. How the agent acts

Three actions are wired into `WeatherSystem` (and exported on the Godot node):

```rust
weather.set_severity(WeatherSeverity::Severe);      // jump to a severity band
weather.modify_wind_vector(Vector3::new(dn, de, dd)); // additive NED wind bias (±30 m/s)
weather.trigger_evasive_maneuver();                  // reduce gust/penalty exposure for a few s
```

In GDScript:

```gdscript
$WeatherSystem.set_severity(3)      # 0 clear, 1 light, 2 moderate, 3 severe
$WeatherSystem.modify_wind_vector(n, e, -d)  # NED, pass NED quirkless
$WeatherSystem.trigger_evasive_maneuver()
```

---

## 4. Determinism / stochasticity (`seed`, `deterministic`)

`WeatherConfig`:

* `seed: u64` — the same seed produces **byte-identical weather** (scene
  re-rolls *and* the Dryden gust stream) for the whole run.
* `deterministic: true` — **fixed pattern** for the episode: one scene, no
  auto re-rolls, deterministic gust envelope → same seed = same weather on
  every episode restart. This is the A/B-comparison mode.
* `deterministic: false` (default) — weather re-rolls a new scene every
  `auto_change_interval` seconds; still fully reproducible for a given seed
  (the RNG stream is fixed), but the agent sees changing conditions.

Episode structure: call `weather.reset()` (or `env.reset()`) at the top of
each episode — it re-seeds and zeroes scene/exposure counters.

```rust
let cfg = WeatherConfig::from_preset(WeatherPreset::Rain, 7)
    .with_deterministic(true);
let mut env = Environment::with_config("aircraft.toml",
    EnvConfig { weather_config: Some(cfg), ..Default::default() })?;
env.reset();
// ... deterministic training comparison across agents
```

---

## 5. Curriculum learning

`WeatherCurriculum` climbs four phases:

| Phase | Preset | Conditions |
|---|---|---|
| `Phase1Clear` | `Clear` | VFR, light breeze |
| `Phase2LightTurb` | `LightTurb` | visible, rough air |
| `Phase3Rain` | `Rain` | rain, reduced visibility, shear |
| `Phase4Storm` | `Storm` | hail, strong gusts, updrafts, IFR |

```rust
let mut curriculum = WeatherCurriculum::new(1_000); // steps per phase
// ... in the training loop:
if let Some(new_phase) = curriculum.step() {
    // switch the environment to curriculum.config(seed) for the new phase
}
if curriculum.is_complete() { /* graduated the storm phase */ }
```

Or embed it in the weather system (it then drives scene progression
automatically, including in deterministic mode):

```rust
let sys = WeatherSystem::new(WeatherConfig::default())
    .with_curriculum(WeatherCurriculum::new(600), seed);
```

Godot node: `set_curriculum(seed, steps_per_phase)`, `next_curriculum_phase()`,
`is_curriculum_complete()`, `get_curriculum_progress()`.

Fast stress-testing presets (fixed conditions, no curriculum required):
`WeatherPreset::Clear / LightTurb / Rain / Storm / ExtremeStorm`.

---

## 6. Termination (`done`) from extreme weather

Enable in the config:

```rust
cfg.extreme_weather_termination = true;
cfg.gust_termination_limit = 35.0;   // m/s instantaneous 3-axis gust
cfg.min_visibility = 500.0;          // m below which exposure counts
cfg.visibility_timeout = 30.0;       // s of low visibility before done
cfg.severe_exposure_limit = 120.0;   // s in Severe weather before done
```

`WeatherSystem::termination()` returns:

```text
None (0)            benign
GustLimit (1)       gust magnitude exceeded  gust_termination_limit
VisibilityLimit (2) sub-min_visibility for longer than visibility_timeout
SevereExposure (3)  Severe weather for longer than severe_exposure_limit
```

`Environment::step` folds this into `EnvStep::terminated`; add
`weather_crash_penalty()` when it fires, exactly as you would for a ground
crash.

---

## 7. Godot (`WeatherSystem` node) quick start

```gdscript
# Add a child WeatherSystem (Node3D GDExtension) to your RL scene.
func _ready() -> void:
    $WeatherSystem.start()                    # env-var config
    $WeatherSystem.configure("rain", 42, true, 0.0)
    $WeatherSystem.weather_changed.connect(_on_weather_changed)

func _physics_process(delta: float) -> void:
    $WeatherSystem.step(delta)
    var obs: PackedFloat64Array = $WeatherSystem.get_observation()
    var penalty: float = $WeatherSystem.get_reward_penalty_total()
    # ... feed obs into your agent, sum penalty into the reward ...
    # Optionally fly the drone through this wind:
    var w: Vector3 = $WeatherSystem.get_wind_vector()
    $FlightSim.set_external_wind(true, w.x, w.z, -w.y)

func _on_weather_changed(severity: int) -> void:
    print("weather now severity ", severity)
```

Full runnable wiring also lives in `godot/scripts/rl_weather_example.gd`, and
`flight_sim.gd` automatically drives the drone through a node named
`WeatherSystem` when one is present.

---

## 8. Presets & configuration cheat-sheet

| Preset | Wind m/s | Rain | Visibility | Turbulence | Updraft | Precip |
|---|---|---|---|---|---|---|
| Clear | 2 | 0.0 | 20 km | Light | 0 | — |
| LightTurb | 5 | 0.05 | 12 km | Light | 1.0 | — |
| Rain | 10 | 0.5 | 5 km | Moderate | 2.5 | Rain |
| Storm | 18 | 0.8 | 2.5 km | Severe | 5.0 | Hail |
| ExtremeStorm | 25 | 1.0 | 1.2 km | Extreme | 8.0 | Hail |

`WeatherConfig` also exposes `length_scales` (longitudinal/lateral/vertical
integral scales, m), `gust_frequency` (envelope Hz), `gust_magnitude` (peak
m/s), `transition_time` (scene ramp s), `reference_altitude` and
`wind_shear` (power-law shear, p ≈ 0.2).

### Reference

* Dryden turbulence spectra — MIL-F-8785C / MIL-HDBK-1797 (implemented in
  `flight_core/src/wind.rs`).
* Boundary-layer power-law shear — ESDU micrometeorology profiles.
* See `flight_core/src/weather.rs` module docs for the full field/units table.