# RL Mission Foundation + PPO Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add the out-and-back fixed-wing mission environment, its shared observation/action contract, a heuristic baseline, the shared evaluation exam, and PPO training on that mission.

**Architecture:** `flight_core::mission` owns the mission state machine and wraps the existing `AvionicsEnvironment`, so physics, sensors, and the PID stack are unchanged. `rl_agent` adds a 32-channel contract adapter, a `Policy` abstraction with a deterministic heuristic, an explicit 30-scenario evaluator, and a generalized trainer that can run any `EnvSpec`. PPO is the only learned model in this pass; A2C, SAC, TD3, and the hybrid selector come in later plans.

**Tech Stack:** Rust 2021, `flight_core` (nalgebra, serde, rand), `rl_agent` (burn 0.21 ndarray/wgpu, serde_json, rand), `cargo test`.

**Spec:** `docs/superpowers/specs/2026-11-04-rl-mission-zoo-design.md`

## Global Constraints

- Fixed-wing only. No landing, taxi, or true hover. "Capture" is a slow-speed pass.
- The action contract is exactly 4 continuous channels: `[roll_cmd, pitch_cmd, yaw_rate_cmd, throttle]`, mapped as roll `±0.6 rad`, pitch `±0.4 rad`, yaw rate `±0.8 rad/s`, throttle `0.5·(tanh(a)+1)`.
- The observation contract is exactly 32 channels: the existing 19 noisy avionics channels unchanged, followed by the 13 mission features. Contract name is `mission.v1`.
- `flight_core` must not gain Godot, burn, or dashboard dependencies.
- `rl_agent` stays on burn 0.21 and must keep the ndarray CPU path working.
- Existing avionics-mode PPO training and legacy checkpoints (`ppo.json`, no `expert.json`) must keep working.
- Avionics mode remains the default for `--mission`; mission mode is opt-in.
- The working tree is dirty. Changes must be additive and must not refactor `weather.rs`, `rl_dash`, or Godot scripts.
- The mission airframe is `m_sweep_mini.toml`; `--config` can still override it.
- `capture_speed` defaults to `1.15 × V_stall`. The skyhook's `max_capture_speed = 5 m/s` is not used (it is below stall).
- Capture needs all four conditions held for `capture_dwell_steps`: horizontal radius, vertical tolerance, airspeed, and dwell.

## Review Focus

These are the input classes the spec implies but no happy-path test covers. Each gets a test in the task that owns the code.

1. An airframe with no `[skyhook]` block, or an arbitrary `--config` path: mission construction and capture-speed resolution must still work.
2. Airspeed at or near zero, or the aircraft exactly on the pickup/home point: bearing is undefined (`atan2(0,0)`), so features and rewards must stay finite.
3. Capture and return completing on the same step, or capture completing on the last budgeted step: bonuses must be paid once and termination precedence must be stable.
4. `max_steps = 0`, `max_steps = 1`, and very small `dt`: no panic, timeout path works.
5. Altitude outside `min_altitude..max_altitude` and horizontal range over `max_range`: out-of-bounds terminates and is not reported as a return success.

---

## Plan Decisions That Refine the Spec

These are builder decisions from the spec's §17. They are recorded here so the plan is implementable without re-reading the spec's open questions.

- **Default scenarios live in code.** `eval::default_scenarios()` generates the canonical 30. `--scenarios FILE` loads a TOML override. The spec's named `mission_eval.toml` becomes an optional override file rather than a required repository file.
- **`Policy` receives the raw contract observation**, not a pre-normalized one. Each learned policy owns its normalizer; the heuristic needs physical units. This keeps the selector and evaluator independent of any one expert's normalization.
- **Capture-alignment signal is a penalty**, `0.0` when perfectly aligned and positive when outside the zone or too fast. Reward profiles subtract it.
- **Crash penalty is inside `MissionEnv`'s reward.** `EnvSpec::crash_penalty()` returns `0.0` for the mission so the trainer cannot double-charge it.
- **Feature sign conventions:** bearing sin/cos are of `wrap_pi(target_bearing − heading)`, so positive sin means the target is to the right. Altitude error is `target_altitude − current_altitude`, so positive means climb.
- **`run_with_env` is the new training entry point.** `run_with_observer` becomes a thin avionics wrapper so existing callers and tests are untouched.

---

## Task 1: Mission Types, Defaults, Stall Speed, and Seeded Geometry

**Files:**
- Create: `flight_core/src/mission.rs`
- Modify: `flight_core/src/lib.rs:49-69`
- Test: `flight_core/src/mission.rs` (`#[cfg(test)] mod tests`)

**Interfaces:**
- Consumes: `crate::config::AircraftConfig`, `crate::atmosphere::Atmosphere`, `rand`.
- Produces: `MissionPhase`, `MissionTermination`, `MissionConfig`, `MissionScenario`, `MissionSignals`, `MissionFeatures`, `MISSION_FEATURE_DIM`, `stall_speed`, `MissionConfig::resolve_capture_speed`, `sample_pickup`.

- [ ] **Step 1: Write the failing tests**

Add this module at the bottom of `flight_core/src/mission.rs`:

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;
    use rand::rngs::StdRng;

    fn mini() -> AircraftConfig {
        let path = if std::path::Path::new("../m_sweep_mini.toml").exists() {
            "../m_sweep_mini.toml"
        } else {
            "m_sweep_mini.toml"
        };
        AircraftConfig::from_file(path).expect("m_sweep_mini.toml should load")
    }

    #[test]
    fn mission_config_defaults_are_sane() {
        let c = MissionConfig::default();
        assert_eq!(c.cruise_altitude, 1000.0);
        assert_eq!(c.cruise_airspeed, 55.0);
        assert_eq!(c.pickup_radius, 40.0);
        assert_eq!(c.capture_dwell_steps, 15);
        assert_eq!(c.max_steps, 4500);
        assert_eq!(c.dt, 1.0 / 30.0);
        assert!(c.pickup_distance_min < c.pickup_distance_max);
        assert!(c.pickup_altitude_min < c.pickup_altitude_max);
        assert!(c.min_altitude < c.max_altitude);
    }

    #[test]
    fn stall_speed_matches_m_sweep_expectation() {
        let v = stall_speed(&mini(), 1000.0);
        assert!((9.0..=11.0).contains(&v), "stall speed was {v}");
    }

    #[test]
    fn capture_speed_defaults_to_stall_margin() {
        let c = MissionConfig::default();
        let v = c.resolve_capture_speed(&mini());
        assert!((v - 1.15 * stall_speed(&mini(), c.cruise_altitude)).abs() < 1e-9);
    }

    #[test]
    fn explicit_capture_speed_overrides_stall() {
        let c = MissionConfig {
            capture_speed: Some(13.0),
            ..Default::default()
        };
        assert_eq!(c.resolve_capture_speed(&mini()), 13.0);
    }

    #[test]
    fn seeded_pickup_is_reproducible_and_in_range() {
        let c = MissionConfig::default();
        let mut a = StdRng::seed_from_u64(7);
        let mut b = StdRng::seed_from_u64(7);
        let mut different = StdRng::seed_from_u64(8);
        let pa = sample_pickup(&c, &mut a);
        let pb = sample_pickup(&c, &mut b);
        let pc = sample_pickup(&c, &mut different);
        assert_eq!(pa, pb);
        assert_ne!(pa, pc);
        let dist = (pa.0 * pa.0 + pa.1 * pa.1).sqrt();
        assert!((c.pickup_distance_min..=c.pickup_distance_max).contains(&dist));
        assert!((c.pickup_altitude_min..=c.pickup_altitude_max).contains(&pa.2));
    }
}
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `cargo test -p flight_core mission::tests -- --nocapture`
Expected: FAIL with `unresolved import` / `cannot find type MissionConfig` because the module does not exist yet.

- [ ] **Step 3: Implement the types and helpers**

In `flight_core/src/mission.rs`, add the module docs and these definitions. Use the exact defaults from the table:

```rust
pub const MISSION_FEATURE_DIM: usize = 13;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum MissionPhase {
    #[default]
    Outbound,
    Capture,
    Return,
    Done,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum MissionTermination {
    #[default]
    None,
    GroundImpact,
    OutOfBounds,
    Timeout,
    CapturedAndReturned,
}

#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize)]
pub struct MissionSignals {
    pub progress_outbound: f64,
    pub progress_inbound: f64,
    pub altitude_error: f64,
    pub speed_error: f64,
    /// 0.0 when perfectly aligned; positive when outside the zone or too fast.
    pub capture_alignment: f64,
    pub dwell_fraction: f64,
    pub time_penalty: f64,
    pub energy_penalty: f64,
    pub crash: bool,
    pub captured: bool,
    pub returned: bool,
}

#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize)]
pub struct MissionFeatures {
    pub pickup_distance_norm: f64,
    pub pickup_bearing_sin: f64,
    pub pickup_bearing_cos: f64,
    pub pickup_altitude_error_norm: f64,
    pub home_distance_norm: f64,
    pub home_bearing_sin: f64,
    pub home_bearing_cos: f64,
    pub phase: f64,
    pub speed_margin_norm: f64,
    pub time_remaining_fraction: f64,
    pub capture_dwell_fraction: f64,
    pub horizontal_offset_norm: f64,
    pub vertical_offset_norm: f64,
}

impl MissionFeatures {
    pub fn to_array(self) -> [f64; MISSION_FEATURE_DIM] { /* field order above */ }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MissionScenario {
    pub id: String,
    pub pickup_north: f64,
    pub pickup_east: f64,
    pub pickup_altitude: f64,
    #[serde(default)]
    pub initial_phase: MissionPhase,
    #[serde(default)]
    pub initial_north: Option<f64>,
    #[serde(default)]
    pub initial_east: Option<f64>,
    #[serde(default)]
    pub initial_altitude: Option<f64>,
    #[serde(default)]
    pub initial_airspeed: Option<f64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MissionConfig {
    pub cruise_altitude: f64,
    pub cruise_airspeed: f64,
    pub pickup_radius: f64,
    pub pickup_vertical_tolerance: f64,
    pub capture_speed: Option<f64>,
    pub capture_dwell_steps: usize,
    pub home_radius: f64,
    pub home_vertical_tolerance: f64,
    pub pickup_distance_min: f64,
    pub pickup_distance_max: f64,
    pub pickup_altitude_min: f64,
    pub pickup_altitude_max: f64,
    pub max_steps: usize,
    pub max_range: f64,
    pub min_altitude: f64,
    pub max_altitude: f64,
    pub ground_altitude: f64,
    pub dt: f64,
}

impl Default for MissionConfig { /* the §7.1 table exactly */ }

impl MissionConfig {
    /// `capture_speed` when set, otherwise `1.15 × stall_speed(aircraft, cruise_altitude)`.
    pub fn resolve_capture_speed(&self, aircraft: &AircraftConfig) -> f64 {
        self.capture_speed
            .unwrap_or_else(|| 1.15 * stall_speed(aircraft, self.cruise_altitude))
    }
}

/// Clean-config stall speed at `altitude`, using `cl0 + cla·alpha_stall_pos`.
pub fn stall_speed(aircraft: &AircraftConfig, altitude: f64) -> f64 {
    let cl_max = (aircraft.cl0 + aircraft.cla * aircraft.alpha_stall_pos).max(0.1);
    let density = Atmosphere::at_altitude(altitude).density;
    (2.0 * aircraft.mass * 9.80665 / (density * aircraft.wing_area * cl_max)).sqrt()
}

/// Uniform pickup sample in polar coordinates. Returns `(north, east, altitude)`.
pub fn sample_pickup(config: &MissionConfig, rng: &mut impl rand::Rng) -> (f64, f64, f64) {
    let bearing = rng.gen::<f64>() * std::f64::consts::TAU;
    let distance = rng.gen_range(config.pickup_distance_min..=config.pickup_distance_max);
    let altitude = rng.gen_range(config.pickup_altitude_min..=config.pickup_altitude_max);
    (distance * bearing.cos(), distance * bearing.sin(), altitude)
}
```

Add `use serde::{Deserialize, Serialize};` and the `AircraftConfig`/`Atmosphere` imports.

- [ ] **Step 4: Register the module and re-export it**

In `flight_core/src/lib.rs`, add the module under the same feature gate as `AvionicsEnvironment`, then re-export:

```rust
#[cfg(feature = "full-avionics")]
pub mod mission;

#[cfg(feature = "full-avionics")]
pub use mission::{
    sample_pickup, stall_speed, MissionConfig, MissionFeatures, MissionPhase, MissionScenario,
    MissionSignals, MissionTermination, MISSION_FEATURE_DIM,
};
```

- [ ] **Step 5: Run the tests**

Run: `cargo test -p flight_core mission::tests`
Expected: all five tests PASS.

- [ ] **Step 6: Commit**

```bash
git add flight_core/src/mission.rs flight_core/src/lib.rs
git commit -m "feat(flight_core): add mission types, defaults, and stall-speed geometry"
```

---

## Task 2: Mission Environment State Machine

**Files:**
- Modify: `flight_core/src/env.rs` (the `AvionicsEnvironment` impl)
- Modify: `flight_core/src/mission.rs`
- Modify: `flight_core/src/lib.rs`
- Test: `flight_core/src/mission.rs` (`mod tests`)

**Interfaces:**
- Consumes: `AvionicsEnvironment`, `AvionicsAction`, `AvionicsObservation`, `FlightFrame`, `AircraftState`, `MissionConfig`, `MissionScenario`, `MissionFeatures`.
- Produces: `MissionEnvironment`, `MissionReset`, `MissionStep`, `MissionContext`, `MissionFeatures` computation.

- [ ] **Step 1: Write the failing tests**

Add these tests inside `flight_core/src/mission.rs`'s test module:

```rust
fn mission_env() -> MissionEnvironment {
    MissionEnvironment::new(
        if std::path::Path::new("../m_sweep_mini.toml").exists() {
            "../m_sweep_mini.toml"
        } else {
            "m_sweep_mini.toml"
        },
        MissionConfig::default(),
        7,
    )
    .expect("mission env")
}

fn scenario_at_pickup(phase: MissionPhase, speed: f64) -> MissionScenario {
    MissionScenario {
        id: "test".into(),
        pickup_north: 0.0,
        pickup_east: 0.0,
        pickup_altitude: 1000.0,
        initial_phase: phase,
        initial_north: Some(0.0),
        initial_east: Some(0.0),
        initial_altitude: Some(1000.0),
        initial_airspeed: Some(speed),
    }
}

#[test]
fn reset_reproduces_seed_geometry() {
    let mut a = mission_env();
    let mut b = mission_env();
    let ra = a.reset();
    let rb = b.reset();
    assert_eq!(ra.phase, MissionPhase::Outbound);
    assert_eq!(a.scenario().pickup_north, b.scenario().pickup_north);
    assert_eq!(a.scenario().pickup_east, b.scenario().pickup_east);
    assert!(ra.features.pickup_distance_norm.is_finite());
}

#[test]
fn reset_with_scenario_pins_geometry() {
    let mut env = mission_env();
    let s = scenario_at_pickup(MissionPhase::Capture, 12.0);
    let r = env.reset_with_scenario(&s);
    assert_eq!(r.phase, MissionPhase::Capture);
    assert_eq!(env.scenario().pickup_north, 0.0);
    assert_eq!(env.capture_speed(), 1.15 * stall_speed(env.aircraft_config(), 1000.0));
}

#[test]
fn entering_the_pickup_zone_switches_to_capture() {
    let mut env = mission_env();
    env.reset_with_scenario(&scenario_at_pickup(MissionPhase::Outbound, 55.0));
    let out = env.step(AvionicsAction::level_cruise(0.5));
    assert_eq!(env.phase(), MissionPhase::Capture);
    assert_eq!(env.capture_attempts(), 1);
    assert!(!out.signals.captured);
}

#[test]
fn capture_requires_the_full_dwell() {
    let mut env = mission_env();
    let cfg = MissionConfig::default();
    let start = env.capture_speed() - 0.5;
    env.reset_with_scenario(&scenario_at_pickup(MissionPhase::Capture, start));
    for _ in 0..cfg.capture_dwell_steps - 1 {
        let out = env.step(AvionicsAction::level_cruise(0.0));
        assert!(!out.signals.captured);
    }
    let out = env.step(AvionicsAction::level_cruise(0.0));
    assert!(out.signals.captured);
    assert_eq!(out.signals.dwell_fraction, 1.0);
    assert_eq!(env.phase(), MissionPhase::Return);
}

#[test]
fn captured_aircraft_returns_home_and_finishes() {
    let mut env = mission_env();
    let mut s = scenario_at_pickup(MissionPhase::Capture, 11.0);
    s.initial_phase = MissionPhase::Return;
    s.initial_north = Some(0.0);
    s.initial_east = Some(0.0);
    env.reset_with_scenario(&s);
    // Force the captured latch for the return leg; reset_with_scenario does it.
    let out = env.step(AvionicsAction::level_cruise(0.0));
    assert_eq!(env.phase(), MissionPhase::Done);
    assert_eq!(out.termination, MissionTermination::CapturedAndReturned);
    assert!(out.terminated);
    assert!(!out.truncated);
    assert!(out.signals.returned);
}

#[test]
fn ground_impact_terminates() {
    let mut env = mission_env();
    let mut s = scenario_at_pickup(MissionPhase::Outbound, 55.0);
    s.initial_altitude = Some(0.0);
    env.reset_with_scenario(&s);
    let out = env.step(AvionicsAction::level_cruise(0.0));
    assert_eq!(out.termination, MissionTermination::GroundImpact);
    assert!(out.terminated);
    assert!(out.signals.crash);
}

#[test]
fn out_of_bounds_terminates() {
    let mut env = mission_env();
    let mut s = scenario_at_pickup(MissionPhase::Outbound, 55.0);
    s.initial_north = Some(6000.0);
    env.reset_with_scenario(&s);
    let out = env.step(AvionicsAction::level_cruise(0.5));
    assert_eq!(out.termination, MissionTermination::OutOfBounds);
    assert!(out.signals.crash);
}

#[test]
fn zero_budget_times_out_without_panicking() {
    let mut env = MissionEnvironment::new(
        if std::path::Path::new("../m_sweep_mini.toml").exists() {
            "../m_sweep_mini.toml"
        } else {
            "m_sweep_mini.toml"
        },
        MissionConfig { max_steps: 1, ..Default::default() },
        7,
    )
    .expect("mission env");
    env.reset();
    let out = env.step(AvionicsAction::level_cruise(0.5));
    assert_eq!(out.termination, MissionTermination::Timeout);
    assert!(out.truncated);
    assert!(!out.terminated);
}

#[test]
fn on_target_features_stay_finite_at_zero_airspeed() {
    let mut env = mission_env();
    env.reset_with_scenario(&scenario_at_pickup(MissionPhase::Capture, 12.0));
    {
        // Zero body velocity makes both bearing and airspeed degenerate.
        let s = env.state_mut();
        s.u = 0.0;
        s.v = 0.0;
        s.w = 0.0;
    }
    let features = env.features();
    for v in features.to_array() {
        assert!(v.is_finite(), "non-finite feature: {v}");
    }
    assert_eq!(features.pickup_distance_norm, 0.0);
}
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `cargo test -p flight_core mission::tests`
Expected: FAIL with `cannot find type MissionEnvironment`.

- [ ] **Step 3: Add the avionics accessors**

In `flight_core/src/env.rs`, inside `impl AvionicsEnvironment`:

```rust
/// Reset to trimmed level flight at an arbitrary altitude and airspeed.
pub fn reset_trimmed(
    &mut self,
    altitude: f64,
    airspeed: f64,
) -> (AvionicsObservation, usize) {
    let prev_steps = self.step_count;
    self.sim.reset();
    self.sim.trim_level_flight(altitude, airspeed);
    self.avionics.reset();
    self.step_count = 0;
    self.avionics.bus_mut().fc_fault_flags = FaultFlags::default();
    if let Some(wx) = &mut self.weather {
        wx.reset();
    }
    (self.observation(), prev_steps)
}

/// Mutable physics state, for scenario placement and tests.
pub fn state_mut(&mut self) -> &mut crate::state::AircraftState {
    &mut self.sim.state
}

/// Loaded airframe configuration.
pub fn aircraft_config(&self) -> &crate::config::AircraftConfig {
    &self.sim.config
}
```

Then change the existing `reset` body to call `self.reset_trimmed(1000.0, 60.0)` and return its result, so the two resets cannot drift.

- [ ] **Step 4: Implement `MissionEnvironment`**

In `flight_core/src/mission.rs` add:

```rust
#[derive(Debug, Clone)]
pub struct MissionReset {
    pub avionics: AvionicsObservation,
    pub features: MissionFeatures,
    pub phase: MissionPhase,
}

#[derive(Debug, Clone)]
pub struct MissionStep {
    pub avionics: AvionicsObservation,
    pub features: MissionFeatures,
    pub signals: MissionSignals,
    pub termination: MissionTermination,
    pub terminated: bool,
    pub truncated: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MissionContext {
    pub phase: MissionPhase,
    pub step: usize,
    pub captured: bool,
}

pub struct MissionEnvironment {
    inner: AvionicsEnvironment,
    config: MissionConfig,
    scenario: MissionScenario,
    phase: MissionPhase,
    captured: bool,
    dwell_steps: usize,
    step_count: usize,
    capture_attempts: u32,
    previous_pickup_distance: f64,
    previous_home_distance: f64,
    capture_speed: f64,
    v_stall: f64,
    rng: StdRng,
    last_signals: MissionSignals,
    termination: MissionTermination,
}
```

`MissionEnvironment::new(config_path, config, seed)`:
- Builds `AvionicsEnvironment::with_config` with
  `EnvConfig { target_altitude: config.cruise_altitude, target_airspeed: config.cruise_airspeed, ground_altitude: config.ground_altitude, max_steps: config.max_steps, dt: config.dt, ..Default::default() }`.
- Resolves `capture_speed` and `v_stall` from `inner.aircraft_config()`.
- Stores `StdRng::seed_from_u64(seed)`.
- Starts with a sampled scenario so `new` is usable without a preceding reset.

`reset()`: samples `(north, east, altitude)` with `sample_pickup`, builds an `Outbound` scenario with id `"random"`, delegates to `reset_with_scenario`.

`reset_with_scenario(&scenario)`:
- Clones the scenario; resolves start position as `(initial_north.unwrap_or(0.0), initial_east.unwrap_or(0.0))`.
- Calls `inner.reset_trimmed(start_alt, start_speed.max(1.0))` (trim divides by dynamic pressure, so a zero-speed start is not a valid trim), then writes `pos_x/pos_y/pos_z` through `state_mut()`.
- Sets `phase = scenario.initial_phase`, `captured = phase == Return`, dwell 0, step 0, attempts 0.
- Sets `previous_*_distance` from the placed state.
- Clears signals and termination to `None`.
- Returns `MissionReset` with the current avionics observation and features.

`step(action)`:
1. Save `phase_before = self.phase`.
2. Call `inner.step(action)`; ignore its `reward`.
3. `self.step_count += 1`.
4. Compute post-step position and distances.
5. Run the transition rules exactly:
   - Outbound → Capture when `pickup_distance <= pickup_radius`; increment `capture_attempts`.
   - Capture: if `pickup_distance > 2.0 * pickup_radius`, return to Outbound and reset dwell. Else if inside radius, inside vertical tolerance, and `airspeed <= capture_speed`, increment dwell; capture when dwell reaches `capture_dwell_steps`. Otherwise reset dwell to 0.
   - Return: if `home_distance <= home_radius` and `|altitude − cruise_altitude| <= home_vertical_tolerance`, set `Done` and `CapturedAndReturned`.
6. Termination precedence: ground impact, then out of bounds, then captured-and-returned, then timeout. (`inner.truncated` means timeout because `max_steps` is shared.)
7. Build `MissionSignals` using `phase_before` for progress and phase targets. `progress_*` is `(previous_distance − current_distance).max(0.0)`. `altitude_error` and `speed_error` are absolute errors against the phase-before targets. `capture_alignment` is `0.0` unless within `5 × pickup_radius`, otherwise `((dist − radius).max(0)/radius) + ((airspeed − capture_speed).max(0)/capture_speed)`. `dwell_fraction = dwell_steps as f64 / capture_dwell_steps as f64`. `time_penalty = 1.0`. `energy_penalty = action.throttle.clamp(0.0, 1.0)`. `crash` is true for ground impact and out of bounds. `captured` and `returned` are true only on the step they latch.
8. Build `MissionFeatures` with:
   - `pickup_distance_norm = dist / 2500.0`
   - bearing sin/cos from `wrap_pi(atan2(pickup_east − east, pickup_north − north) − heading)`
   - `pickup_altitude_error_norm = (pickup_altitude − altitude) / 500.0`
   - home distance and bearing the same way toward `(0,0)`
   - `phase` as `0.0/1.0/2.0/2.0` for Outbound/Capture/Return/Done
   - `speed_margin_norm = (airspeed − v_stall) / 10.0`
   - `time_remaining_fraction = 1.0 − step_count / max_steps` (clamped to `0.0..=1.0`; when `max_steps == 0`, use `0.0`)
   - `capture_dwell_fraction`
   - `horizontal_offset_norm = dist / pickup_radius.max(1e-6)`
   - `vertical_offset_norm = |altitude − pickup_altitude| / pickup_vertical_tolerance.max(1e-6)`
   - `heading` comes from `inner.state_mut()`'s `euler_angles().2`; if `atan2(0,0)` for bearing, it is defined as `0.0` and finite.
9. `terminated = matches!(term, GroundImpact | OutOfBounds | CapturedAndReturned)`, `truncated = term == Timeout`.
10. Store signals/termination and return `MissionStep`.

Add accessors: `phase()`, `capture_attempts()`, `capture_speed()`, `aircraft_config()`, `mission_config()`, `scenario()`, `pickup()`, `signals()`, `termination()`, `context()`, `features()`, `frame()`, `step_count()`, `state_mut()` (delegating for tests).

- [ ] **Step 5: Run the tests**

Run: `cargo test -p flight_core mission::tests`
Expected: all tests PASS.

- [ ] **Step 6: Run the existing environment tests**

Run: `cargo test -p flight_core`
Expected: all PASS, including `env::tests::reset_returns_level_flight_observation`.

- [ ] **Step 7: Commit**

```bash
git add flight_core/src/mission.rs flight_core/src/env.rs flight_core/src/lib.rs
git commit -m "feat(flight_core): add out-and-back mission state machine"
```

---

## Task 3: The 32-Channel Contract

**Files:**
- Create: `rl_agent/src/contracts.rs`
- Modify: `rl_agent/src/lib.rs:60-75`
- Modify: `rl_agent/src/trainer.rs:101-110` (replace the private action mapper)
- Test: `rl_agent/src/contracts.rs`

**Interfaces:**
- Consumes: `flight_core::{MissionFeatures, MISSION_FEATURE_DIM, AvionicsAction}`, `crate::AVIONICS_OBS_DIM`.
- Produces: `MISSION_CONTRACT`, `MISSION_OBS_DIM`, `AVIONICS_CHANNELS`, `MISSION_FEATURE_OFFSET`, index constants, `mission_observation`, `action_to_avionics`.

- [ ] **Step 1: Write the failing tests**

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use flight_core::{MissionFeatures, MISSION_FEATURE_DIM};

    #[test]
    fn contract_dimensions_add_up() {
        assert_eq!(AVIONICS_CHANNELS, 19);
        assert_eq!(MISSION_FEATURE_OFFSET, 19);
        assert_eq!(MISSION_FEATURE_DIM, 13);
        assert_eq!(MISSION_OBS_DIM, 32);
    }

    #[test]
    fn base_channels_are_preserved() {
        let avionics: [f64; AVIONICS_CHANNELS] = std::array::from_fn(|i| i as f64);
        let obs = mission_observation(&avionics, &MissionFeatures::default());
        for (i, expected) in avionics.iter().enumerate() {
            assert_eq!(obs[i], *expected);
        }
    }

    #[test]
    fn mission_features_fill_the_tail() {
        let features = MissionFeatures {
            pickup_distance_norm: 0.4,
            pickup_bearing_sin: 0.5,
            pickup_bearing_cos: 0.5,
            pickup_altitude_error_norm: -0.2,
            home_distance_norm: 0.1,
            home_bearing_sin: -0.5,
            home_bearing_cos: -0.5,
            phase: 1.0,
            speed_margin_norm: 0.5,
            time_remaining_fraction: 0.75,
            capture_dwell_fraction: 0.5,
            horizontal_offset_norm: 1.0,
            vertical_offset_norm: 0.0,
        };
        let obs = mission_observation(&[0.0; AVIONICS_CHANNELS], &features);
        assert_eq!(obs[OBS_PICKUP_DISTANCE], 0.4);
        assert_eq!(obs[OBS_PICKUP_BEARING_SIN], 0.5);
        assert_eq!(obs[OBS_PHASE], 1.0);
        assert_eq!(obs[OBS_VERTICAL_OFFSET], 0.0);
    }

    #[test]
    fn action_mapping_uses_the_documented_scales() {
        let a = action_to_avionics(&[0.0, 0.0, 0.0, 0.0]);
        assert!(a.roll_cmd.abs() < 1e-12);
        assert!((a.throttle - 0.5).abs() < 1e-12);
        let b = action_to_avionics(&[1.0, 1.0, 1.0, 1.0]);
        assert!((b.roll_cmd - 0.6 * 1.0f64.tanh()).abs() < 1e-12);
        assert!((b.pitch_cmd - 0.4 * 1.0f64.tanh()).abs() < 1e-12);
        assert!((b.yaw_rate_cmd - 0.8 * 1.0f64.tanh()).abs() < 1e-12);
        assert!((b.throttle - 0.5 * (1.0f64.tanh() + 1.0)).abs() < 1e-12);
    }
}
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `cargo test -p rl_agent contracts::tests`
Expected: FAIL with `cannot find module contracts`.

- [ ] **Step 3: Implement the contract**

Create `rl_agent/src/contracts.rs`:

```rust
pub const MISSION_CONTRACT: &str = "mission.v1";
pub const AVIONICS_CHANNELS: usize = flight_core::AVIONICS_OBS_DIM;
pub const MISSION_FEATURE_OFFSET: usize = AVIONICS_CHANNELS;
pub const MISSION_OBS_DIM: usize = MISSION_FEATURE_OFFSET + flight_core::MISSION_FEATURE_DIM;

pub const OBS_BARO_ALTITUDE: usize = 10;
pub const OBS_AIRSPEED_INDICATED: usize = 11;
pub const OBS_PICKUP_DISTANCE: usize = 19;
pub const OBS_PICKUP_BEARING_SIN: usize = 20;
pub const OBS_PICKUP_BEARING_COS: usize = 21;
pub const OBS_PICKUP_ALTITUDE_ERROR: usize = 22;
pub const OBS_HOME_DISTANCE: usize = 23;
pub const OBS_HOME_BEARING_SIN: usize = 24;
pub const OBS_HOME_BEARING_COS: usize = 25;
pub const OBS_PHASE: usize = 26;
pub const OBS_SPEED_MARGIN: usize = 27;
pub const OBS_TIME_REMAINING: usize = 28;
pub const OBS_CAPTURE_DWELL: usize = 29;
pub const OBS_HORIZONTAL_OFFSET: usize = 30;
pub const OBS_VERTICAL_OFFSET: usize = 31;

pub fn mission_observation(
    avionics: &[f64; AVIONICS_CHANNELS],
    features: &flight_core::MissionFeatures,
) -> [f64; MISSION_OBS_DIM] {
    let mut out = [0.0; MISSION_OBS_DIM];
    out[..AVIONICS_CHANNELS].copy_from_slice(avionics);
    out[MISSION_FEATURE_OFFSET..].copy_from_slice(&features.to_array());
    out
}

pub fn mission_observation_vec(
    avionics: &[f64; AVIONICS_CHANNELS],
    features: &flight_core::MissionFeatures,
) -> Vec<f64> {
    mission_observation(avionics, features).to_vec()
}

/// Map a tanh-space action to the avionics attitude command.
pub fn action_to_avionics(action: &[f32]) -> flight_core::AvionicsAction {
    flight_core::AvionicsAction {
        roll_cmd: f64::from(action[0].tanh()) * 0.6,
        pitch_cmd: f64::from(action[1].tanh()) * 0.4,
        yaw_rate_cmd: f64::from(action[2].tanh()) * 0.8,
        throttle: 0.5 * (f64::from(action[3].tanh()) + 1.0),
    }
}
```

- [ ] **Step 4: Wire it into the crate**

Add `pub mod contracts;` to `rl_agent/src/lib.rs`. In `rl_agent/src/trainer.rs`, remove the private `action_to_avionics` function and import `use crate::contracts::action_to_avionics;` so the mission adapter and the trainer share one mapping.

- [ ] **Step 5: Run the tests**

Run: `cargo test -p rl_agent contracts::tests`
Expected: PASS.

Run: `cargo test -p rl_agent`
Expected: all existing tests still PASS.

- [ ] **Step 6: Commit**

```bash
git add rl_agent/src/contracts.rs rl_agent/src/lib.rs rl_agent/src/trainer.rs
git commit -m "feat(rl_agent): add mission.v1 observation and action contract"
```

---

## Task 4: MissionEnv Adapter and Reward Profiles

**Files:**
- Create: `rl_agent/src/mission_env.rs`
- Modify: `rl_agent/src/lib.rs`
- Test: `rl_agent/src/mission_env.rs`

**Interfaces:**
- Consumes: `flight_core::{MissionEnvironment, MissionConfig, MissionSignals, MissionContext}`, `crate::contracts`, `crate::trainer::{EnvSpec, StepOutcome}`.
- Produces: `RewardProfile`, `MissionEnv` implementing `EnvSpec`.

- [ ] **Step 1: Write the failing tests**

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use crate::contracts::MISSION_OBS_DIM;

    fn env() -> MissionEnv {
        MissionEnv::new("../m_sweep_mini.toml", MissionConfig::default(), RewardProfile::balanced(), 7)
            .expect("mission env")
    }

    #[test]
    fn reset_and_step_use_32_channels() {
        let mut e = env();
        assert_eq!(e.obs_dim(), MISSION_OBS_DIM);
        assert_eq!(e.reset().len(), MISSION_OBS_DIM);
        let out = e.step(&[0.0, 0.0, 0.0, 0.0]);
        assert_eq!(out.obs.len(), MISSION_OBS_DIM);
        assert!(out.reward.is_finite());
        assert!(out.obs.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn balanced_profile_pays_progress_and_punishes_crash() {
        let p = RewardProfile::balanced();
        let good = MissionSignals {
            progress_outbound: 2.0,
            time_penalty: 1.0,
            energy_penalty: 0.5,
            ..Default::default()
        };
        let bad = MissionSignals {
            crash: true,
            time_penalty: 1.0,
            ..Default::default()
        };
        assert!(p.reward(&good) > 0.0);
        assert!(p.reward(&bad) < -100.0);
    }

    #[test]
    fn capture_and_return_bonuses_are_paid_once_in_the_signals() {
        let p = RewardProfile::balanced();
        let s = MissionSignals { captured: true, returned: true, ..Default::default() };
        assert_eq!(p.reward(&s), p.capture_bonus + p.return_bonus);
    }

    #[test]
    fn mission_env_does_not_double_charge_crashes() {
        let e = env();
        assert_eq!(e.crash_penalty(), 0.0);
    }

    #[test]
    fn named_profiles_resolve_and_unknown_names_fail() {
        assert_eq!(RewardProfile::from_name("balanced").unwrap().name, "balanced");
        assert!(RewardProfile::from_name("nope").is_err());
    }
}
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `cargo test -p rl_agent mission_env::tests`
Expected: FAIL with `cannot find module mission_env`.

- [ ] **Step 3: Implement `RewardProfile`**

```rust
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RewardProfile {
    pub name: String,
    pub progress_outbound: f64,
    pub progress_inbound: f64,
    pub altitude_error: f64,
    pub speed_error: f64,
    pub capture_alignment: f64,
    pub time_penalty: f64,
    pub energy_penalty: f64,
    pub crash_penalty: f64,
    pub capture_bonus: f64,
    pub return_bonus: f64,
}

impl RewardProfile {
    pub fn balanced() -> Self { /* values from the spec §7.1 */ }
    pub fn aggressive() -> Self { /* balanced with cruise-speed emphasis relaxed and progress weights raised */ }
    pub fn cautious() -> Self { /* balanced with altitude/speed weights raised */ }
    pub fn capture_focus() -> Self { /* balanced with capture_alignment raised */ }
    pub fn from_name(name: &str) -> Result<Self, String> { /* match the four names */ }
    pub fn from_toml_file(path: &str) -> Result<Self, Box<dyn std::error::Error>>;

    pub fn reward(&self, s: &flight_core::MissionSignals) -> f64 {
        let mut r = 0.0;
        r += self.progress_outbound * s.progress_outbound;
        r += self.progress_inbound * s.progress_inbound;
        r -= self.altitude_error * s.altitude_error;
        r -= self.speed_error * s.speed_error;
        r -= self.capture_alignment * s.capture_alignment;
        r -= self.time_penalty * s.time_penalty;
        r -= self.energy_penalty * s.energy_penalty;
        if s.crash { r -= self.crash_penalty; }
        if s.captured { r += self.capture_bonus; }
        if s.returned { r += self.return_bonus; }
        r
    }
}
```

`balanced` uses exactly: progress 1.0/1.0, altitude 0.02, speed 0.05, capture alignment 1.0, time 0.01, energy 0.05, crash 200.0, capture bonus 100.0, return bonus 200.0. `aggressive` = balanced with `speed_error = 0.02` and `progress_* = 1.5`. `cautious` = balanced with `altitude_error = 0.05`, `speed_error = 0.08`. `capture_focus` = balanced with `capture_alignment = 5.0`.

- [ ] **Step 4: Implement `MissionEnv`**

```rust
pub struct MissionEnv {
    inner: flight_core::MissionEnvironment,
    reward: RewardProfile,
    last_obs: [f64; crate::contracts::MISSION_OBS_DIM],
}

impl MissionEnv {
    pub fn new(
        config_path: &str,
        mission: flight_core::MissionConfig,
        reward: RewardProfile,
        seed: u64,
    ) -> Result<Self, Box<dyn std::error::Error>>;
}
```

`EnvSpec` implementation:
- `obs_dim() = MISSION_OBS_DIM`, `action_dim() = ACTION_DIM`.
- `reset()`: `let r = self.inner.reset();` store the composed 32-channel obs, return `Vec<f64>`.
- `step(action)`: `let out = self.inner.step(crate::contracts::action_to_avionics(action));` compose obs, compute `reward = self.reward.reward(&out.signals)`, return `StepOutcome { obs, reward, terminated: out.terminated, truncated: out.truncated }`.
- `crash_penalty() = 0.0` (already included).
- `target_altitude() = self.inner.mission_config().cruise_altitude` — expose `MissionEnvironment::mission_config(&self) -> &MissionConfig`.
- `target_airspeed() = self.inner.mission_config().cruise_airspeed`.
- `frame() = Some(self.inner.frame())`, `ground_altitude() = self.inner.mission_config().ground_altitude`.
- `mission_context() = Some(self.inner.context())` (the `EnvSpec` method added in Task 9; add the default now and override later, or add the method to `EnvSpec` in this task). **Do this now:** add `fn mission_context(&self) -> Option<flight_core::MissionContext> { None }` to `EnvSpec` in `trainer.rs`, then override it here.

- [ ] **Step 5: Run the tests**

Run: `cargo test -p rl_agent mission_env::tests`
Expected: PASS.

Run: `cargo test -p rl_agent`
Expected: all PASS.

- [ ] **Step 6: Commit**

```bash
git add rl_agent/src/mission_env.rs rl_agent/src/lib.rs rl_agent/src/trainer.rs
git commit -m "feat(rl_agent): add MissionEnv adapter and reward profiles"
```

---

## Task 5: Policy Abstraction, Expert Card, and Heuristic Baseline

**Files:**
- Create: `rl_agent/src/policy.rs`
- Modify: `rl_agent/src/lib.rs`
- Test: `rl_agent/src/policy.rs`

**Interfaces:**
- Consumes: `flight_core::{MissionPhase, MissionConfig}`, `crate::contracts`.
- Produces: `ModelKind`, `Specialization`, `ExpertCard`, `PolicyContext`, `Policy`, `HeuristicPolicy`, `HeuristicConfig`.

- [ ] **Step 1: Write the failing tests**

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use crate::contracts::{MISSION_OBS_DIM, OBS_AIRSPEED_INDICATED, OBS_PICKUP_BEARING_SIN};
    use flight_core::MissionPhase;

    fn obs_with(bearing_sin: f64, airspeed: f64) -> Vec<f32> {
        let mut o = vec![0.0f32; MISSION_OBS_DIM];
        o[OBS_PICKUP_BEARING_SIN] = bearing_sin as f32;
        o[OBS_AIRSPEED_INDICATED] = airspeed as f32;
        o
    }

    #[test]
    fn expert_card_round_trips() {
        let card = ExpertCard {
            id: "heuristic_v1".into(),
            model: ModelKind::Heuristic,
            specialization: Specialization::Full,
            contract: "mission.v1".into(),
            reward_profile: "balanced".into(),
            obs_mask: (0..32).collect(),
            trained_steps: 0,
            seed: 7,
            git_commit: "deadbeef".into(),
            notes: "baseline".into(),
        };
        let json = serde_json::to_string(&card).unwrap();
        let back: ExpertCard = serde_json::from_str(&json).unwrap();
        assert_eq!(card, back);
    }

    #[test]
    fn heuristic_actions_stay_in_bounds() {
        let mut p = HeuristicPolicy::new(55.0, 11.5, 10.0);
        let ctx = PolicyContext { phase: MissionPhase::Outbound, step: 0, captured: false };
        for bearing in [-3.0, -0.5, 0.0, 0.5, 3.0] {
            let a = p.act(&obs_with(bearing, 55.0), &ctx);
            assert_eq!(a.len(), 4);
            assert!(a[0].abs() <= 0.6 + 1e-6);
            assert!(a[1].abs() <= 0.4 + 1e-6);
            assert!(a[2].abs() <= 0.8 + 1e-6);
            assert!((0.0..=1.0).contains(&a[3]));
        }
    }

    #[test]
    fn heuristic_rolls_toward_a_target_on_the_right() {
        let mut p = HeuristicPolicy::new(55.0, 11.5, 10.0);
        let ctx = PolicyContext { phase: MissionPhase::Outbound, step: 0, captured: false };
        let a = p.act(&obs_with(0.5, 55.0), &ctx);
        assert!(a[0] > 0.0);
    }

    #[test]
    fn heuristic_bleeds_throttle_in_capture_when_fast() {
        let mut p = HeuristicPolicy::new(55.0, 11.5, 10.0);
        let fast = PolicyContext { phase: MissionPhase::Capture, step: 0, captured: false };
        let slow = PolicyContext { phase: MissionPhase::Capture, step: 0, captured: false };
        let a_fast = p.act(&obs_with(0.0, 40.0), &fast);
        let a_slow = p.act(&obs_with(0.0, 11.5), &slow);
        assert!(a_fast[3] < a_slow[3]);
    }

    #[test]
    fn heuristic_never_commands_nan_on_zero_airspeed() {
        let mut p = HeuristicPolicy::new(55.0, 11.5, 10.0);
        let ctx = PolicyContext { phase: MissionPhase::Capture, step: 0, captured: false };
        let a = p.act(&obs_with(0.0, 0.0), &ctx);
        assert!(a.iter().all(|v| v.is_finite()));
    }
}
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `cargo test -p rl_agent policy::tests`
Expected: FAIL with `cannot find module policy`.

- [ ] **Step 3: Implement the policy types and heuristic**

```rust
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ModelKind { Ppo, A2c, Sac, Td3, Heuristic }

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Specialization { Full, Outbound, Capture, Return }

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ExpertCard { /* the spec §9 fields exactly */ }

#[derive(Debug, Clone, Copy)]
pub struct PolicyContext {
    pub phase: flight_core::MissionPhase,
    pub step: usize,
    pub captured: bool,
}

/// A policy returns **physical** commands, not tanh-space network outputs:
/// `[roll_cmd (rad), pitch_cmd (rad), yaw_rate_cmd (rad/s), throttle (0..1)]`.
/// The evaluator feeds them straight into `AvionicsAction`. Learned policies
/// convert their network output with `contracts::action_to_avionics`; the
/// heuristic produces them directly.
pub trait Policy {
    fn act(&mut self, obs: &[f32], ctx: &PolicyContext) -> Vec<f32>;
    fn deterministic_action(&mut self, obs: &[f32], ctx: &PolicyContext) -> Vec<f32>;
    fn reset(&mut self);
    fn card(&self) -> &ExpertCard;
}

pub struct HeuristicConfig {
    pub cruise_airspeed: f64,
    pub capture_speed: f64,
    pub v_stall: f64,
    pub bank_gain: f64,
    pub pitch_gain: f64,
    pub speed_gain: f64,
}

pub struct HeuristicPolicy {
    cfg: HeuristicConfig,
    card: ExpertCard,
}
```

`HeuristicPolicy::new(cruise_airspeed, capture_speed, v_stall)` fills the default gains: `bank_gain = 1.2`, `pitch_gain = 0.0015`, `speed_gain = 0.03`, and builds a card with `model: Heuristic`, `specialization: Full`, `contract: MISSION_CONTRACT`, `reward_profile: "balanced"`, `obs_mask: (0..32).collect()`, `trained_steps: 0`.

`act`:
- Read `airspeed = obs[OBS_AIRSPEED_INDICATED] as f64`. If not finite, use `0.0`.
- Bearing channels: Outbound/Capture use `OBS_PICKUP_BEARING_SIN/COS`; Return uses `OBS_HOME_BEARING_SIN/COS`.
- `bearing_sin` finite fallback `0.0`.
- `roll = clamp(bank_gain * bearing_sin, -0.6, 0.6)`.
- Altitude error: Outbound/Capture `obs[OBS_PICKUP_ALTITUDE_ERROR] * 500.0`; Return `0.0` (the mission feature only carries pickup error). For return, hold altitude with `0.0` pitch command; the trimmed state holds it.
- `pitch = clamp(pitch_gain * alt_error, -0.4, 0.4)`.
- Reference speed: `capture_speed` in Capture, otherwise `cruise_airspeed`.
- `throttle = clamp(0.5 + speed_gain * (reference − airspeed), 0.0, 1.0)`. In Capture, add a stall guard: `if airspeed < v_stall + 1.0 { throttle = throttle.max(0.6); }`.
- `yaw_rate = 0.0`.
- Return `[roll, pitch, yaw_rate, throttle]` as `f32`.
- `deterministic_action` delegates to `act`.
- `reset` is a no-op.

- [ ] **Step 4: Register the module**

Add `pub mod policy;` to `rl_agent/src/lib.rs` and re-export the public policy types from the crate root if the rest of the crate uses them by short name.

- [ ] **Step 5: Run the tests**

Run: `cargo test -p rl_agent policy::tests`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add rl_agent/src/policy.rs rl_agent/src/lib.rs
git commit -m "feat(rl_agent): add Policy trait, ExpertCard, and heuristic baseline"
```

---

## Task 6: Scenario Exam and Scorecard

**Files:**
- Create: `rl_agent/src/eval.rs`
- Modify: `rl_agent/Cargo.toml` (add `toml = "0.8"`)
- Modify: `rl_agent/src/lib.rs`
- Test: `rl_agent/src/eval.rs`

**Interfaces:**
- Consumes: `Policy`, `PolicyContext`, `MissionEnvironment`, `MissionScenario`, `MissionConfig`, `RewardProfile`, `contracts`.
- Produces: `default_scenarios`, `load_scenarios`, `wilson_interval`, `EpisodeScore`, `EvalSummary`, `EvalSuite`, `run_eval_suite`, `write_eval_json`, `write_eval_csv`.

- [ ] **Step 1: Write the failing tests**

```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_exam_has_30_spread_scenarios() {
        let s = default_scenarios();
        assert_eq!(s.len(), 30);
        let mut bearings = std::collections::BTreeSet::new();
        for sc in &s {
            let b = sc.pickup_east.atan2(sc.pickup_north).to_degrees().round() as i32;
            bearings.insert(b.rem_euclid(360) / 36);
        }
        assert_eq!(bearings.len(), 10);
        assert!(s.iter().all(|sc| sc.id.starts_with("exam_")));
    }

    #[test]
    fn wilson_interval_is_sane_at_the_extremes() {
        let (lo, hi) = wilson_interval(10, 10, 1.96);
        assert!(lo > 0.7 && hi <= 1.0);
        let (lo, hi) = wilson_interval(0, 10, 1.96);
        assert!(lo >= 0.0 && hi < 0.3);
        let (lo, hi) = wilson_interval(0, 0, 1.96);
        assert_eq!((lo, hi), (0.0, 0.0));
    }

    #[test]
    fn summary_aggregates_rates_and_means() {
        let episodes = vec![
            EpisodeScore { scenario: "a".into(), success: true, captured: true, returned: true, time_to_capture_s: Some(10.0), time_to_home_s: Some(20.0), ..Default::default() },
            EpisodeScore { scenario: "b".into(), crash: true, ..Default::default() },
        ];
        let s = EvalSummary::from_episodes(&episodes);
        assert_eq!(s.n, 2);
        assert!((s.success_rate - 0.5).abs() < 1e-9);
        assert!((s.crash_rate - 0.5).abs() < 1e-9);
        assert_eq!(s.mean_time_to_capture_s, Some(10.0));
    }

    #[test]
    fn scenario_file_round_trips() {
        let dir = std::env::temp_dir().join("rl_agent_eval_test");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("scenarios.toml");
        let mut all = default_scenarios();
        let one = vec![all.remove(0)];
        write_scenarios(&path, &one).unwrap();
        let back = load_scenarios(path.to_str().unwrap()).unwrap();
        assert_eq!(back.len(), 1);
        assert_eq!(back[0].pickup_north, one[0].pickup_north);
    }
}
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `cargo test -p rl_agent eval::tests`
Expected: FAIL with `cannot find module eval`.

- [ ] **Step 3: Implement scenarios**

```rust
pub const EXAM_SCENARIOS: usize = 30;

pub fn default_scenarios() -> Vec<flight_core::MissionScenario> {
    // 10 bearings every 36°, cycling distances 1000/1500/2000 m and
    // altitudes 800/1000/1200 m. id = format!("exam_{i:03}"), i in 0..30.
    // pickup_north = d*cos(bearing), pickup_east = d*sin(bearing).
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScenarioFile {
    pub scenario: Vec<flight_core::MissionScenario>,
}

pub fn write_scenarios(path: &std::path::Path, scenarios: &[flight_core::MissionScenario])
    -> Result<(), Box<dyn std::error::Error>>;
pub fn load_scenarios(path: &str) -> Result<Vec<flight_core::MissionScenario>, Box<dyn std::error::Error>>;
```

- [ ] **Step 4: Implement scoring and the suite**

```rust
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct EpisodeScore {
    pub scenario: String,
    pub success: bool,
    pub captured: bool,
    pub returned: bool,
    pub crash: bool,
    pub timeout: bool,
    pub termination: String,
    pub time_to_capture_s: Option<f64>,
    pub time_to_home_s: Option<f64>,
    pub outbound_track_error: f64,
    pub inbound_track_error: f64,
    pub min_capture_speed: f64,
    pub capture_attempts: u32,
    pub energy: f64,
    pub reward: f64,
    pub steps: usize,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct EvalSummary {
    pub n: usize,
    pub success_rate: f64,
    pub success_ci_low: f64,
    pub success_ci_high: f64,
    pub capture_rate: f64,
    pub return_rate: f64,
    pub crash_rate: f64,
    pub timeout_rate: f64,
    pub mean_time_to_capture_s: Option<f64>,
    pub mean_time_to_home_s: Option<f64>,
    pub mean_outbound_track_error: f64,
    pub mean_inbound_track_error: f64,
    pub mean_min_capture_speed: f64,
    pub mean_energy: f64,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct EvalSuite {
    pub contract: String,
    pub expert: String,
    pub reward_profile: String,
    pub episodes: Vec<EpisodeScore>,
    pub summary: EvalSummary,
}

pub fn wilson_interval(successes: usize, n: usize, z: f64) -> (f64, f64);
pub fn run_eval_suite(
    policy: &mut dyn crate::policy::Policy,
    env: &mut flight_core::MissionEnvironment,
    scenarios: &[flight_core::MissionScenario],
    max_steps: usize,
    reward: &RewardProfile,
) -> EvalSuite;
pub fn write_eval_json(path: &std::path::Path, suite: &EvalSuite) -> Result<(), Box<dyn std::error::Error>>;
pub fn write_eval_csv(path: &std::path::Path, suite: &EvalSuite) -> Result<(), Box<dyn std::error::Error>>;
```

`write_eval_csv` writes exactly this header, then one row per episode:

```text
scenario,success,captured,returned,crash,timeout,time_to_capture_s,time_to_home_s,outbound_track_error,inbound_track_error,min_capture_speed,capture_attempts,energy,reward,steps
```

`run_eval_suite` sets `contract = contracts::MISSION_CONTRACT`, `expert = policy.card().id.clone()`, and `reward_profile = reward.name.clone()` on the returned suite.

`run_eval_suite` loop per scenario:
- `env.reset_with_scenario(scenario)`; get `r.features` and compose the obs via `contracts::mission_observation`.
- Reset the policy.
- Iterate up to `max_steps`:
  - `ctx = PolicyContext { phase: env.phase(), step, captured: env.phase() == Return || env.phase() == Done }`.
  - `action = policy.act(&obs, &ctx)`.
  - Build `flight_core::AvionicsAction { roll_cmd: action[0] as f64, pitch_cmd: action[1] as f64, yaw_rate_cmd: action[2] as f64, throttle: (action[3] as f64).clamp(0.0, 1.0) }` directly. `Policy` actions are already physical commands; do **not** apply `contracts::action_to_avionics` here. Call `env.step`.
  - Compose the next obs.
  - `reward_total += reward.reward(&out.signals)`; `energy += action[3].clamp(0,1)`.
  - Track `time_to_capture_s` when `out.signals.captured`, `time_to_home_s` when `out.signals.returned`.
  - Track `min_capture_speed` while the pre-step phase is Capture, using `env.frame().airspeed`.
  - Track cross-track error for Outbound and Return using `env.frame().north/east` and the pickup/home line (helper `cross_track`).
  - Stop on `out.terminated || out.truncated`.
- Fill `EpisodeScore`; `success = out.termination == CapturedAndReturned`; `crash = termination is GroundImpact or OutOfBounds`; `timeout = termination == Timeout`. `min_capture_speed` averages only over episodes that captured (leave `0.0` when none did), so a non-capturing episode cannot pull the mean down.
- `EvalSummary::from_episodes` computes rates and means; success interval uses `wilson_interval(success_count, n, 1.96)`.

- [ ] **Step 5: Add the `toml` dependency**

Add `toml = "0.8"` to `[dependencies]` in `rl_agent/Cargo.toml`.

- [ ] **Step 6: Run the tests**

Run: `cargo test -p rl_agent eval::tests`
Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add rl_agent/src/eval.rs rl_agent/src/lib.rs rl_agent/Cargo.toml
git commit -m "feat(rl_agent): add 30-scenario exam and per-phase scorecard"
```

---

## Task 7: Heuristic Feasibility Gate

**Files:**
- Create: `rl_agent/tests/mission_heuristic.rs`
- Test: the same file

**Interfaces:**
- Consumes: `HeuristicPolicy`, `MissionEnvironment`, `MissionConfig`, `default_scenarios`, `run_eval_suite`, `RewardProfile`.

- [ ] **Step 1: Write the failing test**

```rust
use flight_core::{MissionConfig, MissionEnvironment};
use rl_agent::eval::{default_scenarios, run_eval_suite};
use rl_agent::mission_env::RewardProfile;
use rl_agent::policy::HeuristicPolicy;

fn aircraft() -> &'static str {
    if std::path::Path::new("m_sweep_mini.toml").exists() {
        "m_sweep_mini.toml"
    } else {
        "../m_sweep_mini.toml"
    }
}

#[test]
fn heuristic_completes_at_least_one_full_mission() {
    let cfg = MissionConfig::default();
    let mut env = MissionEnvironment::new(aircraft(), cfg.clone(), 7).expect("mission env");
    let v_stall = flight_core::stall_speed(env.aircraft_config(), cfg.cruise_altitude);
    let capture = cfg.resolve_capture_speed(env.aircraft_config());
    let mut policy = HeuristicPolicy::new(cfg.cruise_airspeed, capture, v_stall);
    let scenarios = default_scenarios();
    let suite = run_eval_suite(
        &mut policy,
        &mut env,
        &scenarios[..3],
        cfg.max_steps,
        &RewardProfile::balanced(),
    );
    assert_eq!(suite.summary.n, 3);
    assert!(
        suite.summary.success_rate >= 1.0 / 3.0,
        "heuristic completed {}/3; episodes: {:?}",
        suite.episodes.iter().filter(|e| e.success).count(),
        suite.episodes,
    );
}

#[test]
fn heuristic_is_deterministic_and_bounded_on_the_first_scenario() {
    let cfg = MissionConfig::default();
    let mut env = MissionEnvironment::new(aircraft(), cfg.clone(), 7).expect("mission env");
    let v_stall = flight_core::stall_speed(env.aircraft_config(), cfg.cruise_altitude);
    let capture = cfg.resolve_capture_speed(env.aircraft_config());
    let mut policy = HeuristicPolicy::new(cfg.cruise_airspeed, capture, v_stall);
    let mut scenarios = default_scenarios();
    let scenario = scenarios.remove(0);
    let reset = env.reset_with_scenario(&scenario);
    let obs = rl_agent::contracts::mission_observation(&reset.avionics, &reset.features);
    let ctx = rl_agent::policy::PolicyContext {
        phase: env.phase(),
        step: 0,
        captured: false,
    };
    let a = policy.act(&obs, &ctx);
    let b = policy.act(&obs, &ctx);
    assert_eq!(a, b);
    assert!(a[0].abs() <= 0.6 + 1e-6);
    assert!(a[1].abs() <= 0.4 + 1e-6);
    assert!(a[2].abs() <= 0.8 + 1e-6);
    assert!((0.0..=1.0).contains(&a[3]));
}
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `cargo test -p rl_agent --test mission_heuristic`
Expected: FAIL if the heuristic cannot complete a mission yet, or fail to compile because `MissionEnvironment::features` or `contracts` are not public. Fix visibility first.

- [ ] **Step 3: Tune the heuristic until the gate passes**

The first full mission is the hard one: the aircraft must bleed from 55 m/s to about 11.5 m/s inside a 40 m zone. Adjust only the heuristic, in this order:
1. `speed_gain` (default 0.03) — raise on approach so it slows earlier.
2. Capture throttle floor — lower it while keeping the stall guard.
3. `bank_gain` (default 1.2) — raise to tighten the turn once slow.
4. Add an approach rule: when `OBS_PICKUP_DISTANCE * 2500.0 < 400.0` and phase is Outbound, target `capture_speed` rather than `cruise_airspeed`. This is the intended capture strategy and should be the first fix if the aircraft overflies the zone.

Record the final gains in `HeuristicConfig::default()` with a comment naming the scenario that motivated them.

- [ ] **Step 4: Re-run the gate**

Run: `cargo test -p rl_agent --test mission_heuristic -- --nocapture`
Expected: both tests PASS. The first must report at least one success out of three.

- [ ] **Step 5: Commit**

```bash
git add rl_agent/tests/mission_heuristic.rs rl_agent/src/policy.rs
git commit -m "test(rl_agent): add heuristic feasibility gate for the mission"
```

---

## Task 8: Mission Telemetry in the Run Manifest

**Files:**
- Modify: `rl_agent/src/contracts.rs` (`MissionSummary`)
- Modify: `rl_agent/src/telemetry.rs:85-121, 282-320`
- Modify: `rl_agent/src/trainer.rs:617-640` (construct the new `RunMeta` fields as `None` for now)
- Test: `rl_agent/src/telemetry.rs` (`mod tests`)

**Interfaces:**
- Consumes: `contracts::MISSION_CONTRACT`, `policy::ExpertCard`, `flight_core::MissionConfig`.
- Produces: `MissionSummary`, extended `RunMeta`/`Manifest`.

- [ ] **Step 1: Write the failing test**

Add to `telemetry.rs` tests:

```rust
#[test]
fn manifest_records_mission_metadata() {
    let dir = std::env::temp_dir().join(format!(
        "rl_agent_manifest_mission_{}",
        std::process::id()
    ));
    let _ = std::fs::remove_dir_all(&dir);
    let tcfg = TrainerConfig {
        iterations: 1,
        rollout_steps: 8,
        ..Default::default()
    };
    let ncfg = NetConfig { obs_dim: 32, ..Default::default() };
    let spec = AlgoSpec::from_id("ppo").unwrap();
    let summary = crate::contracts::MissionSummary {
        airframe: "m_sweep_mini.toml".into(),
        contract: crate::contracts::MISSION_CONTRACT.into(),
        reward_profile: "balanced".into(),
        cruise_altitude: 1000.0,
        cruise_airspeed: 55.0,
        pickup_radius: 40.0,
        pickup_vertical_tolerance: 15.0,
        capture_speed: 11.5,
        capture_dwell_steps: 15,
        max_steps: 4500,
    };
    let mut w = TelemetryWriter::start(
        &TelemetryConfig { run_root: dir.to_string_lossy().into(), name: Some("mt".into()) },
        &RunMeta {
            algo: "ppo",
            backend: "cpu",
            tcfg: &tcfg,
            ncfg: &ncfg,
            spec: &spec,
            mission: Some(&summary),
            reward_profile: Some("balanced"),
            specialization: Some("full"),
            contract: crate::contracts::MISSION_CONTRACT,
            expert: None,
        },
    )
    .unwrap()
    .unwrap();
    w.finish().unwrap();
    let json: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(w.run_dir().join("run.json")).unwrap()).unwrap();
    assert_eq!(json["mission"]["contract"], "mission.v1");
    assert_eq!(json["mission"]["capture_speed"], 11.5);
    assert_eq!(json["contract"], "mission.v1");
    let _ = std::fs::remove_dir_all(&dir);
}
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `cargo test -p rl_agent telemetry::tests::manifest_records_mission_metadata`
Expected: FAIL to compile because `RunMeta` has no `mission` field.

- [ ] **Step 3: Add `MissionSummary`**

In `contracts.rs`:

```rust
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MissionSummary {
    pub airframe: String,
    pub contract: String,
    pub reward_profile: String,
    pub cruise_altitude: f64,
    pub cruise_airspeed: f64,
    pub pickup_radius: f64,
    pub pickup_vertical_tolerance: f64,
    pub capture_speed: f64,
    pub capture_dwell_steps: usize,
    pub max_steps: usize,
}

impl MissionSummary {
    pub fn from_config(
        airframe: &str,
        config: &flight_core::MissionConfig,
        capture_speed: f64,
        reward_profile: &str,
    ) -> Self;
}
```

- [ ] **Step 4: Extend the manifest**

Add to `RunMeta` and `Manifest` (serde `skip_serializing_if = "Option::is_none"` for the options):

```rust
pub mission: Option<&'a MissionSummary>,
pub reward_profile: Option<&'a str>,
pub specialization: Option<&'a str>,
pub contract: &'a str,
pub expert: Option<&'a ExpertCard>,
```

Set `contract: "avionics"` for avionics runs. Populate `Manifest` from `meta` in `TelemetryWriter::start`.

- [ ] **Step 5: Update every `RunMeta` construction**

Search with `grep -rn "RunMeta {" rl_agent/src`. Add the five fields. For avionics callers use `mission: None, reward_profile: None, specialization: None, contract: "avionics", expert: None`. For tests, use the same unless the test is the new one.

- [ ] **Step 6: Run the tests**

Run: `cargo test -p rl_agent telemetry::tests`
Expected: PASS, including the existing manifest tests.

- [ ] **Step 7: Commit**

```bash
git add rl_agent/src/contracts.rs rl_agent/src/telemetry.rs rl_agent/src/trainer.rs
git commit -m "feat(rl_agent): record mission metadata in run.json"
```

---

## Task 9: Generalized Trainer and PPO on the Mission

**Files:**
- Modify: `rl_agent/src/trainer.rs:582-940` (`run` → `run_with_env`)
- Modify: `rl_agent/src/main.rs:20-105, 210-380`
- Modify: `rl_agent/src/lib.rs` (exports)
- Test: `rl_agent/src/trainer.rs` (`mod tests`), `rl_agent/src/main.rs` (module tests for pure helpers)

**Interfaces:**
- Consumes: `EnvSpec`, `MissionEnv`, `MissionSummary`, `ExpertCard`, `RewardProfile`.
- Produces: `run_with_env`, `TrainingOutcome` unchanged, CLI `--mission`, `--model`, `--reward-profile`.

- [ ] **Step 1: Write the failing tests**

In `trainer.rs` tests:

```rust
#[test]
fn tiny_ppo_run_on_the_mission_uses_32_channels() {
    let device = crate::backend::cpu_device();
    let tcfg = TrainerConfig {
        iterations: 1,
        rollout_steps: 32,
        env_max_steps: 64,
        save_every: 0,
        print_every: 0,
        eval_every: 0,
        eval_episodes: 0,
        seed: 3,
        ..Default::default()
    };
    let ncfg = NetConfig {
        obs_dim: crate::contracts::MISSION_OBS_DIM,
        action_dim: crate::ACTION_DIM,
        hidden: 16,
        log_std_init: -0.5,
    };
    let spec = AlgoSpec::from_id("ppo").unwrap();
    let mut env = crate::mission_env::MissionEnv::new(
        "../m_sweep_mini.toml",
        flight_core::MissionConfig { max_steps: 64, ..Default::default() },
        crate::mission_env::RewardProfile::balanced(),
        3,
    )
    .unwrap();
    let outcome = run_with_env::<Cpu>(
        &tcfg,
        &TelemetryConfig::disabled(),
        &ncfg,
        &spec,
        &device,
        &mut env,
        None,
        None,
        None,
        None,
        &mut TrainObserver::disabled(),
    )
    .expect("mission training run");
    assert_eq!(outcome.normalizer.dim, crate::contracts::MISSION_OBS_DIM);
    assert_eq!(outcome.algorithm.net_config().obs_dim, crate::contracts::MISSION_OBS_DIM);
}

#[test]
fn mission_checkpoint_round_trips_at_32_channels() {
    let device = crate::backend::cpu_device();
    let dir = std::env::temp_dir().join(format!("rl_agent_mission_ckpt_{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    let tcfg = TrainerConfig {
        iterations: 1,
        rollout_steps: 16,
        env_max_steps: 32,
        save_every: 0,
        print_every: 0,
        eval_every: 0,
        eval_episodes: 0,
        seed: 3,
        ..Default::default()
    };
    let ncfg = NetConfig {
        obs_dim: crate::contracts::MISSION_OBS_DIM,
        action_dim: crate::ACTION_DIM,
        hidden: 16,
        log_std_init: -0.5,
    };
    let spec = AlgoSpec::from_id("ppo").unwrap();
    let mut env = crate::mission_env::MissionEnv::new(
        "../m_sweep_mini.toml",
        flight_core::MissionConfig { max_steps: 32, ..Default::default() },
        crate::mission_env::RewardProfile::balanced(),
        3,
    )
    .unwrap();
    let outcome = run_with_env::<Cpu>(
        &tcfg,
        &TelemetryConfig::disabled(),
        &ncfg,
        &spec,
        &device,
        &mut env,
        None,
        None,
        None,
        None,
        &mut TrainObserver::disabled(),
    )
    .unwrap();
    save_checkpoint(
        outcome.algorithm.as_ref(),
        &outcome.normalizer,
        &tcfg,
        &ncfg,
        dir.to_str().unwrap(),
    )
    .unwrap();
    let loaded = load_checkpoint::<Cpu>(dir.to_str().unwrap(), &device, 3).unwrap();
    assert_eq!(loaded.normalizer.dim, crate::contracts::MISSION_OBS_DIM);
    let _ = std::fs::remove_dir_all(&dir);
}
```

In `main.rs`, add a new `#[cfg(test)] mod tests` at the bottom. `main.rs` is the
binary crate, so it must refer to the library through `rl_agent::`, not
`crate::`:

```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn observation_dim_follows_the_selected_mission() {
        let mut args = std::collections::HashMap::new();
        args.insert("mission".to_string(), "mission".to_string());
        assert_eq!(
            build_net_config(&args).unwrap().obs_dim,
            rl_agent::contracts::MISSION_OBS_DIM
        );
        args.insert("mission".to_string(), "avionics".to_string());
        assert_eq!(
            build_net_config(&args).unwrap().obs_dim,
            rl_agent::AVIONICS_OBS_DIM
        );
    }

    #[test]
    fn model_is_an_alias_for_algo() {
        let mut args = std::collections::HashMap::new();
        args.insert("model".to_string(), "ppo".to_string());
        assert_eq!(build_algo_config(&args).unwrap().id(), rl_agent::algo::ALGO_PPO);
    }

    #[test]
    fn mission_mode_defaults_to_the_skyhook_airframe() {
        let mut args = std::collections::HashMap::new();
        args.insert("mission".to_string(), "mission".to_string());
        assert_eq!(config_path(&args), "m_sweep_mini.toml");
        args.insert("mission".to_string(), "avionics".to_string());
        assert_eq!(config_path(&args), "aircraft.toml");
    }
}
```

`build_net_config`, `build_algo_config`, and `config_path` must therefore be
plain module-level functions in `main.rs`, not nested inside another function.

- [ ] **Step 2: Run the tests to verify they fail**

Run: `cargo test -p rl_agent tiny_ppo_run_on_the_mission -- --nocapture`
Expected: FAIL to compile because `run_with_env` does not exist.

- [ ] **Step 3: Extract `run_with_env`**

Replace the body of `run_with_observer` with an `AvionicsEnv` construction followed by a call to the new function. The new function contains the existing loop verbatim except it no longer constructs the env:

```rust
#[allow(clippy::too_many_arguments)]
pub fn run_with_env<B: AutodiffBackend<FloatElem = f32>>(
    tcfg: &TrainerConfig,
    telemetry: &TelemetryConfig,
    ncfg: &NetConfig,
    spec: &AlgoSpec,
    device: &B::Device,
    env: &mut dyn EnvSpec,
    mission: Option<&MissionSummary>,
    expert: Option<&ExpertCard>,
    reward_profile: Option<&str>,
    specialization: Option<&str>,
    obs: &mut TrainObserver,
) -> Result<TrainingOutcome<B>, Box<dyn std::error::Error>>;
```

Pass the mission/expert/profile/specialization through to `RunMeta` with `contract` set to `MISSION_CONTRACT` when `mission.is_some()`, else `"avionics"`.

`run_with_observer` stays public with its current signature and calls:

```rust
let mut env = AvionicsEnv::new(config_path, tcfg.env_dt, tcfg.env_max_steps)?;
run_with_env(tcfg, telemetry, ncfg, spec, device, &mut env, None, None, None, None, obs)
```

- [ ] **Step 4: Update the CLI helpers**

In `main.rs`:
- Add `fn mission_mode(args) -> &str` returning `"avionics"` or `"mission"`, rejecting anything else.
- Add `fn config_path(args) -> String`: explicit `--config` wins; otherwise `m_sweep_mini.toml` for mission, `aircraft.toml` for avionics.
- Change `build_net_config(args)` to choose `MISSION_OBS_DIM` or `AVIONICS_OBS_DIM` from `mission_mode`.
- `build_algo_config` reads `--model` first, then `--algo`.
- Add `fn build_reward_profile(args) -> Result<RewardProfile, _>` defaulting to `"balanced"`.
- Add `fn build_mission_config(args) -> Result<MissionConfig, _>` using `MissionConfig::default()` and overriding `max_steps` from `--env-max-steps`. In mission mode, if `--env-max-steps` was not supplied, also set `tcfg.env_max_steps` to that mission budget before building the env, so periodic evaluation uses the same episode length as training.

- [ ] **Step 5: Route train mode through the mission**

In `exec`'s `"train"` branch:
- If `--model heuristic`, build the heuristic policy and print that it is evaluation-only; run `eval::run_eval_suite` on the default scenarios and write `eval.json`/`eval.csv` into `tcfg.checkpoint_dir`. Then return.
- Otherwise build `let mut env: Box<dyn EnvSpec> = match mission_mode { "mission" => Box::new(MissionEnv::new(...)), _ => Box::new(AvionicsEnv::new(...)) }`.
- Build `MissionSummary` when in mission mode.
- Call `run_with_env` with `env.as_mut()`.

- [ ] **Step 6: Update the usage text**

Add to `USAGE`:
```
MISSION MODE:
  --mission avionics|mission    training target (default avionics)
  --model ppo|heuristic         alias for --algo; heuristic is eval-only
  --reward-profile NAME|PATH    balanced|aggressive|cautious|capture_focus|file
  --env-max-steps N             mission episode budget (default 4500 in mission mode)

  Mission mode defaults to m_sweep_mini.toml and the 32-channel mission.v1
  contract. Avionics mode defaults to aircraft.toml and the 19-channel bus.
```

- [ ] **Step 7: Run the tests**

Run: `cargo test -p rl_agent tiny_ppo_run_on_the_mission`
Expected: PASS.

Run: `cargo test -p rl_agent`
Expected: all PASS.

Run: `cargo test -p flight_core`
Expected: all PASS.

- [ ] **Step 8: Commit**

```bash
git add rl_agent/src/trainer.rs rl_agent/src/main.rs rl_agent/src/lib.rs
git commit -m "feat(rl_agent): train PPO on the mission with a generalized EnvSpec loop"
```

---

## Task 10: `eval-suite` Mode and Checkpoint-Backed Policies

**Files:**
- Modify: `rl_agent/src/policy.rs` (`RlPolicy`, checkpoint card fallback)
- Modify: `rl_agent/src/main.rs` (`exec` mode match, usage)
- Modify: `rl_agent/src/eval.rs` (no changes expected; used by the CLI)
- Test: `rl_agent/src/policy.rs`, `rl_agent/tests/mission_eval_cli.rs`

**Interfaces:**
- Consumes: `load_checkpoint`, `Algorithm`, `RunningNormalizer`, `run_eval_suite`, `HeuristicPolicy`.
- Produces: `RlPolicy`, `ExpertCard::from_checkpoint`, CLI `--mode eval-suite`.

- [ ] **Step 1: Write the failing tests**

In `policy.rs`:

```rust
#[test]
fn rl_policy_normalizes_and_returns_a_bounded_action() {
    let device = crate::backend::cpu_device();
    let ncfg = crate::net::NetConfig {
        obs_dim: crate::contracts::MISSION_OBS_DIM,
        action_dim: crate::ACTION_DIM,
        hidden: 16,
        log_std_init: 0.0,
    };
    let spec = crate::algo::AlgoSpec::from_id("ppo").unwrap();
    let algo = crate::algo::make_algorithm::<crate::backend::Cpu>(&spec, &ncfg, 1, &device);
    let normalizer = crate::normalize::RunningNormalizer::new(ncfg.obs_dim);
    let card = ExpertCard {
        id: "ppo_test".into(),
        model: ModelKind::Ppo,
        specialization: Specialization::Full,
        contract: crate::contracts::MISSION_CONTRACT.into(),
        reward_profile: "balanced".into(),
        obs_mask: (0..32).collect(),
        trained_steps: 0,
        seed: 1,
        git_commit: String::new(),
        notes: String::new(),
    };
    let mut p = RlPolicy::new(algo, normalizer, card);
    let ctx = PolicyContext { phase: flight_core::MissionPhase::Outbound, step: 0, captured: false };
    let a = p.act(&vec![0.0f32; crate::contracts::MISSION_OBS_DIM], &ctx);
    assert_eq!(a.len(), crate::ACTION_DIM);
    assert!(a.iter().all(|v| v.is_finite()));
}

#[test]
fn legacy_checkpoint_card_falls_back_to_full_specialization() {
    let dir = std::env::temp_dir().join(format!("rl_agent_legacy_card_{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    // No expert.json in this directory: the legacy checkpoint case.
    let card = ExpertCard::from_checkpoint(dir.to_str().unwrap(), crate::algo::ALGO_PPO);
    assert_eq!(card.model, ModelKind::Ppo);
    assert_eq!(card.specialization, Specialization::Full);
    assert_eq!(card.contract, crate::contracts::MISSION_CONTRACT);
    assert_eq!(card.obs_mask.len(), crate::contracts::MISSION_OBS_DIM);
    let _ = std::fs::remove_dir_all(&dir);
}
```

In `rl_agent/tests/mission_eval_cli.rs`:

```rust
use rl_agent::eval::{default_scenarios, run_eval_suite, write_eval_json, write_eval_csv};
use rl_agent::mission_env::RewardProfile;
use rl_agent::policy::HeuristicPolicy;
use flight_core::{MissionConfig, MissionEnvironment};

#[test]
fn eval_suite_writes_json_and_csv_for_the_heuristic() {
    let cfg = MissionConfig { max_steps: 400, ..Default::default() };
    let mut env = MissionEnvironment::new(
        if std::path::Path::new("m_sweep_mini.toml").exists() { "m_sweep_mini.toml" } else { "../m_sweep_mini.toml" },
        cfg.clone(),
        7,
    )
    .unwrap();
    let v_stall = flight_core::stall_speed(env.aircraft_config(), cfg.cruise_altitude);
    let mut policy = HeuristicPolicy::new(cfg.cruise_airspeed, cfg.resolve_capture_speed(env.aircraft_config()), v_stall);
    let suite = run_eval_suite(&mut policy, &mut env, &default_scenarios()[..1], cfg.max_steps, &RewardProfile::balanced());
    let dir = std::env::temp_dir().join(format!("rl_agent_eval_cli_{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    write_eval_json(&dir.join("eval.json"), &suite).unwrap();
    write_eval_csv(&dir.join("eval.csv"), &suite).unwrap();
    assert!(dir.join("eval.json").metadata().unwrap().len() > 0);
    let csv = std::fs::read_to_string(dir.join("eval.csv")).unwrap();
    assert!(csv.starts_with("scenario,success,captured"));
    let _ = std::fs::remove_dir_all(&dir);
}
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `cargo test -p rl_agent rl_policy_normalizes`
Expected: FAIL to compile because `RlPolicy` does not exist.

- [ ] **Step 3: Implement `RlPolicy`**

```rust
pub struct RlPolicy<B: burn::tensor::backend::AutodiffBackend<FloatElem = f32>> {
    algorithm: Box<dyn crate::algo::Algorithm<B>>,
    normalizer: crate::normalize::RunningNormalizer,
    card: ExpertCard,
}

impl<B> RlPolicy<B>
where
    B: burn::tensor::backend::AutodiffBackend<FloatElem = f32>,
{
    pub fn new(
        algorithm: Box<dyn crate::algo::Algorithm<B>>,
        normalizer: crate::normalize::RunningNormalizer,
        card: ExpertCard,
    ) -> Self;
}
```

`Policy::act`: normalize the raw obs with `self.normalizer.normalize_f32(obs)`, call `self.algorithm.act(&norm)` to get tanh-space output, convert it with `contracts::action_to_avionics`, then return the four physical numbers `[roll_cmd, pitch_cmd, yaw_rate_cmd, throttle]`. `deterministic_action` does the same with `algorithm.deterministic_action`. `reset` is a no-op. `card` returns the stored card.

Add `ExpertCard::from_checkpoint(dir: &str, algo_id: &str) -> ExpertCard`: if `expert.json` exists, parse it; otherwise return the legacy fallback with `model` mapped from `algo_id`, `specialization: Full`, `contract: MISSION_CONTRACT`, `reward_profile: "balanced"`, `obs_mask: (0..32).collect()`, `trained_steps: 0`, `seed: 0`, empty git commit/notes.

- [ ] **Step 4: Add the CLI mode**

In `exec`, add:

```rust
"eval-suite" => {
    let config_path = config_path(args);
    let mission_cfg = build_mission_config(args)?;
    let profile = build_reward_profile(args)?;
    let mut env = MissionEnvironment::new(&config_path, mission_cfg.clone(), tcfg.seed)?;
    let mut policy: Box<dyn Policy> = if model_id(args) == "heuristic" {
        let v_stall = flight_core::stall_speed(env.aircraft_config(), mission_cfg.cruise_altitude);
        Box::new(HeuristicPolicy::new(
            mission_cfg.cruise_airspeed,
            mission_cfg.resolve_capture_speed(env.aircraft_config()),
            v_stall,
        ))
    } else {
        let cp = load_checkpoint::<B>(&tcfg.checkpoint_dir, &device, tcfg.seed)?;
        let card = ExpertCard::from_checkpoint(&tcfg.checkpoint_dir, cp.algorithm.id());
        Box::new(RlPolicy::new(cp.algorithm, cp.normalizer, card))
    };
    let scenarios = match args.get("scenarios") {
        Some(path) => load_scenarios(path)?,
        None => default_scenarios(),
    };
    let suite = run_eval_suite(&mut *policy, &mut env, &scenarios, mission_cfg.max_steps, &profile);
    let out = args.get("out").cloned().unwrap_or_else(|| tcfg.checkpoint_dir.clone());
    let dir = std::path::Path::new(&out);
    std::fs::create_dir_all(dir)?;
    write_eval_json(&dir.join("eval.json"), &suite)?;
    write_eval_csv(&dir.join("eval.csv"), &suite)?;
    println!(
        "[rl_agent] eval-suite: {} scenarios, success {:.1}% [{:.1}, {:.1}], capture {:.1}%, crash {:.1}%",
        suite.summary.n,
        suite.summary.success_rate * 100.0,
        suite.summary.success_ci_low * 100.0,
        suite.summary.success_ci_high * 100.0,
        suite.summary.capture_rate * 100.0,
        suite.summary.crash_rate * 100.0,
    );
    Ok(())
}
```

Add `fn model_id(args) -> &str` as the shared `--model`/`--algo` lookup and use it in `build_algo_config` too. Add usage text for `eval-suite` and `--scenarios`/`--out`.

- [ ] **Step 5: Run the tests**

Run: `cargo test -p rl_agent rl_policy_normalizes`
Expected: PASS.

Run: `cargo test -p rl_agent --test mission_eval_cli`
Expected: PASS.

Run: `cargo test -p rl_agent`
Expected: all PASS.

- [ ] **Step 6: Commit**

```bash
git add rl_agent/src/policy.rs rl_agent/src/main.rs rl_agent/tests/mission_eval_cli.rs rl_agent/src/eval.rs
git commit -m "feat(rl_agent): add eval-suite mode and checkpoint-backed policies"
```

---

## Task 11: Mission Guide and Final Verification

**Files:**
- Create: `docs/rl_mission_guide.md`
- Modify: `docs/superpowers/specs/2026-11-04-rl-mission-zoo-design.md` (status → approved/implementing)
- Test: the commands below

**Interfaces:**
- Consumes: everything above.
- Produces: user-facing documentation.

- [ ] **Step 1: Write the guide**

Cover, with the exact commands from this plan:
- The mission: phases, defaults, capture conditions, why 5 m/s is not the gate.
- The `mission.v1` observation table (32 channels) and action mapping.
- Reward profiles and how to add one.
- Training commands:
  - `cargo run -p rl_agent -- --mission mission --model ppo --iterations 1000`
  - `cargo run -p rl_agent -- --mission mission --model ppo --config m_sweep_mini.toml --reward-profile capture_focus --name ppo_capture`
- Evaluation:
  - `cargo run -p rl_agent -- --mode eval-suite --mission mission --checkpoint checkpoints/ppo_x`
  - `cargo run -p rl_agent -- --mode eval-suite --mission mission --model heuristic`
  - `--scenarios PATH` and `--out DIR`.
- How to read `eval.json`, `eval.csv`, and the manifest fields.
- Compatibility: avionics mode is the default, legacy checkpoints load, `expert.json` is optional.
- What is intentionally not here yet: A2C/SAC/TD3, the selector, cable physics, landing, weather exam.

- [ ] **Step 2: Run the full test suites**

Run: `cargo test -p flight_core`
Expected: PASS.

Run: `cargo test -p rl_agent`
Expected: PASS.

- [ ] **Step 3: Run a manual mission training smoke test**

Run:
```bash
cargo run -p rl_agent -- --backend cpu --mission mission --model ppo \
  --iterations 2 --rollout 64 --env-max-steps 128 --hidden 16 \
  --telemetry off --stream off --control off --save-every 0
```
Expected: the run completes two iterations and prints `training complete`.

- [ ] **Step 4: Run a manual eval-suite smoke test**

Run:
```bash
cargo run -p rl_agent -- --backend cpu --mode eval-suite --mission mission \
  --model heuristic --env-max-steps 400 --out /tmp/rl_mission_eval
```
Expected: `eval.json` and `eval.csv` exist in `/tmp/rl_mission_eval`, and the printed success rate is a valid percentage.

- [ ] **Step 5: Confirm avionics compatibility**

Run:
```bash
cargo run -p rl_agent -- --backend cpu --iterations 1 --rollout 32 \
  --env-max-steps 64 --telemetry off --stream off --control off --save-every 0
```
Expected: completes with the 19-channel avionics environment.

- [ ] **Step 6: Commit**

```bash
git add docs/rl_mission_guide.md docs/superpowers/specs/2026-11-04-rl-mission-zoo-design.md
git commit -m "docs: add mission training and evaluation guide"
```

---

## Definition of Done for This Plan

- `cargo test -p flight_core` and `cargo test -p rl_agent` pass.
- A seeded reset reproduces pickup geometry; scenario resets pin it.
- Capture requires radius, altitude, speed, and dwell together.
- Ground impact, out of bounds, timeout, and captured-and-returned all terminate with the right reason.
- The heuristic completes at least one of the first three exam scenarios on the M-Sweep mini.
- PPO trains on the mission with `obs_dim = 32` and saves/loads a checkpoint.
- `eval-suite` writes `eval.json` and `eval.csv` with a success rate and Wilson interval.
- Avionics mode and the existing 19-channel PPO path still work.
- `docs/rl_mission_guide.md` explains the mission, contract, commands, and evaluation.

## Follow-Up Plans (not this one)

- Milestone 3: A2C.
- Milestone 4: SAC (replay buffer, twin critics, entropy tuning).
- Milestone 5: TD3.
- Milestone 6: rule-based hybrid selector and phase-specialized runs.
- Milestone 7: learned gating (optional).
- Cable-tension capture physics and a weather exam.
