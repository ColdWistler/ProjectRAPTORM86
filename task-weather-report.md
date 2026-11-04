# Milestone 7 — Weather Exam (`weather_exam.rs`)

## Completed

- Created `rl_agent/src/weather_exam.rs` with:
  - `WeatherExamEntry` (label + `MissionScenario` + `WeatherPreset` + seed)
  - `weather_scenarios()` — exactly 10 entries spanning 5 presets (`clear`, `light_turb`, `rain`, `storm`, `extreme_storm`), 2 episodes each.
  - `WeatherConfig::Preset` reuse (`from_preset` + `with_seed` / `with_deterministic`).
  - `run_weather_episode()` — passes `weather_config` through `MissionEnvironment::new_with_weather()` (additive to `flight_core::mission.rs`).
  - `WeatherExamSummary` / `PresetStats` / `AggregateStats` with Wilson interval (`wilson_interval` at z=1.96).
- Updated `rl_agent/src/lib.rs` (`pub mod weather_exam;`).
- Updated `rl_agent/src/mission_env.rs` (`new_with_weather`).
- Updated `flight_core/src/mission.rs` (`new_with_weather`, `weather_config` field) — no physics changes.

## Verification

- `cargo test -p flight_core --lib`: 139 passed; 0 failed; weather tests still green (`deterministic_mode_fixed_pattern_per_episode`, `same_seed_same_wind_sequence`, etc.). No weather regression.
- `cargo test -p rl_agent --lib`: 98 passed; 0 failed; new `weather_exam` tests pass (`weather_scenarios_has_ten_entries_and_spans_all_presets`, `wilson_interval_computes_within_bounds`, `build_summary_aggregates_correctly`).
- `weather_scenarios()` produces 10 scenarios covering all 5 `WeatherPreset` values with 2 entries each.
- Wilson interval computes `[lo, hi]` within `[0, 1]` for all binary rates.

## Design Choices

- `WeatherConfig::Preset` mapped to `WeatherPreset` (existing enum in `flight_core::weather`).
- Geometry reuses default mission spread; weather is the only variable.
- `run_weather_episode()` runs a short scripted trajectory (neutral cruise action) to record termination/success/crash; no full RL policy required for exam framework.
- Report uses `BTreeMap` for preset-level breakdown + aggregate stats with Wilson intervals.

## Constraints Met

- Reuses `MissionConfig`, `MissionEnvironment`, `MissionScenario`.
- No new physics; only `EnvConfig.weather_config` threading.
- `rl_agent` tests pass; weather regression: zero.
