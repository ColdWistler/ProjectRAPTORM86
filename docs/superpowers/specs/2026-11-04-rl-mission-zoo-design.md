# RL Mission Zoo and Hybrid Expert Selector — Design

**Date:** 2026-11-04  
**Status:** Proposed, awaiting approval  
**Scope:** `flight_core`, `rl_agent`, telemetry/dashboard integration  
**Related:** `docs/rl_agent_guide.md`, `docs/rl_weather_guide.md`, `ARCHITECTURE_REVIEW.md`

## 1. Intent and Why

Today `rl_agent` trains exactly one thing: PPO holding an attitude/altitude
setpoint in the `AvionicsEnvironment`. The algorithm layer is pluggable, but
only PPO is registered, and the trainer is hard-wired to one on-policy rollout
loop.

The goal is to make the simulator useful for **mission-level learning** and for
**comparing genuinely different learning systems**:

- Fly a fixed-wing aircraft out to a pickup point, perform a slow capture pass,
  and return home.
- Train several distinct model families on that mission, each with its own
  trainer, network, reward emphasis, and hyper-parameters.
- Score every finished model on the same fixed mission exam, so there is a
  shared yardstick even though training is not matched.
- Use those scores to drive a hybrid selector that picks the right expert for
  the current mission phase, with a heuristic expert as a predictable fallback.

The user explicitly does **not** require matched training budgets or identical
training conditions. Uniqueness belongs inside each model. The shared parts are
the mission, the observation/action contract, and the final exam.

## 2. Goals

1. A mission environment: outbound navigation → slow-speed capture at a pickup
   point → return home. No landing.
2. A canonical observation and action contract shared by every expert.
3. Per-expert training loops for PPO, A2C, SAC, and TD3, plus a deterministic
   heuristic expert.
4. A reward-parts interface so each expert can weight the mission differently
   without forking the environment.
5. A fixed evaluation suite that produces a per-phase scorecard for any expert.
6. A rule-based phase selector that dispatches to the best expert per phase,
   with a heuristic fallback.
7. Telemetry and dashboard support so runs, reward profiles, and scorecards are
   visible without reading JSON by hand.

## 3. Non-Goals

- No landing, taxi, or ground operations.
- No true hover. The aircraft is fixed-wing; "hover" is implemented as a
  slow-speed loiter/capture pass, not a stationary hold.
- No cable-tension physics in v1. Capture is a discrete geometry/speed event.
  The existing `skyhook.rs` cable model stays available but is not integrated
  into the first mission.
- No learned meta-policy in v1. The selector is rule-based first.
- No multi-agent training and no population-based training.
- No fairness guarantees across models: different budgets, nets, reward
  profiles, and exploration are expected.

## 4. Anti-Goals

- A single generic trainer that forces every algorithm through PPO's loop.
- A single scalar "score" that hides why a model won.
- Breaking existing `AvionicsEnvironment` training or existing checkpoints.
- Broad refactors of `weather.rs`, `rl_dash`, or Godot scripts that are not
  required for this work.
- Training that only looks good in calm air by flying in circles or avoiding
  the capture.

## 5. Constraints

- **Fixed-wing only.** Minimum speed is bounded by stall. The M-Sweep mini
  config already carries `[skyhook]` (`capture_radius = 0.35 m`,
  `max_capture_speed = 5 m/s`), which is a physical hook-in-ring capture, not a
  hover.
- **The mission airframe is `m_sweep_mini.toml`.** It is the small fixed-wing
  UAV that already carries the skyhook configuration and matches the "skyhook
  drone" intent. `--config` can still select `aircraft.toml`, `MQI.toml`,
  `Engine.toml`, or another file. Run metadata records the chosen airframe.
- **The skyhook `max_capture_speed = 5 m/s` is not used as the mission gate.**
  It is below the stall speed of every airframe here (M-Sweep ≈ 10 m/s,
  Cessna ≈ 24.5 m/s), so a capture at 5 m/s is physically impossible without
  stalling. Mission v1 computes `capture_speed` from stall by default
  (`1.15 × V_stall`, ≈ 11.5 m/s for M-Sweep) and exposes an explicit override.
  The 5 m/s value remains a follow-up for a different capture mechanism (for
  example a winched cable with a slow relative closing speed).
- **Existing action space is continuous 4-D:** roll command, pitch command,
  yaw-rate command, throttle. Those feed the PID flight controller, which
  drives servos and the ESC. Changing the action space would invalidate the
  current controller and the existing checkpoints.
- **Observations are the noisy avionics bus (19 channels).** Mission features
  must be appended; the agent should not get ground-truth physics channels it
  could not perceive.
- **`flight_core` must stay engine-independent** (no Godot or burn deps).
- **`rl_agent` uses burn 0.21** with ndarray and wgpu backends selected at
  runtime.
- **The working tree is dirty.** New work must be additive and must not churn
  unrelated weather, dashboard, or Godot changes.
- **CPU training is the default path.** Off-policy models must remain usable on
  CPU with modest nets and buffers.

## 6. Approaches Considered

### A. On-policy only (PPO + A2C)

Reuse the existing rollout → GAE → update loop. Smallest diff, but only compares
one algorithm family. PPO is usually the strongest on-policy method, so the
benchmark would be weak.

### B. Unique trainers + shared contract and exam (chosen)

Each model family owns its own training loop, network, reward weighting, and
hyper-parameters. All experts share the mission environment, observation/action
contract, checkpoint protocol, and final evaluation suite. The selector is a
thin dispatcher over the shared contract.

This is the only option that satisfies both requirements: genuine model
uniqueness and a meaningful answer to "which handles what better".

### C. Fully independent systems

Each model gets its own observations, actions, and mission definition. Maximum
freedom, but no shared job definition: a selector cannot swap experts safely,
and a scorecard cannot compare them. Rejected.

### D. Independent trainer per algorithm with no shared evaluator

Each model owns everything, judged on its own terms. Rejected for the same
reason: no shared yardstick, no selector.

**Decision:** B. Uniqueness is internal. The contract and the exam are shared.

## 7. System Architecture

```
flight_core
  mission.rs            MissionConfig, MissionPhase, MissionSignals,
                        MissionEnvironment, capture/termination logic
  env.rs                existing AvionicsEnvironment (kept; small accessors added)
  skyhook.rs            existing cable model (not used by mission v1)

rl_agent
  mission_env.rs        EnvSpec adapter: MissionEnvironment -> trainer/observer
  contracts.rs          obs/action layout constants + version
  policy.rs             Policy trait, ExpertCard, policy adapters
  eval.rs               scenario suite + scorecard (per-phase)
  selector.rs           rule-based phase selector + hysteresis
  experts/
    mod.rs              expert registry, ModelSpec, factory
    ppo.rs              existing PPO trainer (moved/wrapped, not rewritten)
    a2c.rs              on-policy A2C
    sac.rs              off-policy SAC (replay, twin Q, target nets, alpha)
    td3.rs              off-policy TD3 (replay, twin Q, target nets)
    heuristic.rs        pursuit + energy-management baseline
  training_common.rs    telemetry, checkpoints, observer, env factory
  trainer.rs            existing PPO loop kept compatible; shared helpers extracted

docs
  rl_mission_guide.md   mission, contracts, training, evaluation, selector
```

### 7.1 Mission Environment

`flight_core/src/mission.rs` owns the mission state machine and wraps the
existing avionics stack. It reuses `Simulator`, `standard_stack`, and the
existing integration step, so the physics and sensors stay identical.

**Phases:**

| Phase | Enter when | Target | Exit |
|---|---|---|---|
| `Outbound` | reset | pickup point | pickup captured, crash, timeout |
| `Capture` | within pickup zone | pickup point | captured, drifted out, crash, timeout |
| `Return` | captured | home | home reached, crash, timeout |
| `Done` | home reached | — | episode ends |

**Mission defaults (all in `MissionConfig`, all overridable):**

| Field | Default | Meaning |
|---|---|---|
| `cruise_altitude` | 1000 m | outbound/return target altitude |
| `cruise_airspeed` | 55 m/s | outbound/return target airspeed |
| `pickup_radius` | 40 m | horizontal capture zone |
| `pickup_vertical_tolerance` | 15 m | vertical capture zone |
| `capture_speed` | `auto` → 1.15 × V_stall | maximum airspeed for capture |
| `capture_dwell_steps` | 15 (0.5 s) | capture condition must hold this long |
| `home_radius` | 100 m | horizontal return gate |
| `home_vertical_tolerance` | 20 m | vertical return gate |
| `pickup_distance_range` | 1.0–2.0 km | reset randomization |
| `pickup_altitude_range` | 800–1200 m | reset randomization |
| `max_steps` | 4500 (150 s at 30 Hz) | total episode budget |
| `max_range` | 5 km | out-of-bounds radius |
| `min_altitude` | 100 m | lower out-of-bounds bound |
| `max_altitude` | 3000 m | upper out-of-bounds bound |

**Reset randomization (seeded):**

- Pickup point: 1.0–2.0 km from home, bearing uniform in 0–360°, altitude
  800–1200 m.
- Start: home position, trimmed level flight at `cruise_altitude` /
  `cruise_airspeed`.
- `max_steps` is a **total episode budget**, not a per-leg budget.
  `time_remaining_fraction` counts down from the total.
- Same seed reproduces the same pickup point and sensor noise.
- A scenario file can override the pickup geometry and start state explicitly
  (see §11), which is how the shared exam guarantees identical geometry for
  every expert.

**Pickup / capture condition:**

- Horizontal distance to pickup ≤ `pickup_radius` (default 40 m).
- Vertical distance to pickup altitude ≤ `pickup_vertical_tolerance` (default
  15 m).
- Airspeed ≤ `capture_speed` (default `1.15 × V_stall`, computed from the
  loaded airframe; ≈ 11.5 m/s for M-Sweep, ≈ 28 m/s for the Cessna).
- The condition must hold for `capture_dwell_steps` (default 15 steps / 0.5 s)
  to avoid a single lucky frame.
- On success: phase → `Return`, a `captured` flag is latched, and a capture
  bonus is emitted.

The 0.35 m physical ring radius is **not** the v1 success radius. It is too
tight for a first learnable task. The mission exposes `pickup_radius` and the
geometry check uses the hook offset for visualization future-proofing, but v1
scores the zone/dwell condition. A later "precision capture" mode can tighten
the radius toward the physical ring.

**Return condition:**

- Horizontal distance to home ≤ `home_radius` (default 100 m).
- Vertical distance to `cruise_altitude` ≤ `home_vertical_tolerance` (default
  20 m).
- No landing, no speed gate. Reaching the home gate ends the episode in
  success.

**Phase target altitude/speed:** outbound and return use
`cruise_altitude` / `cruise_airspeed`; capture uses the pickup altitude and
`capture_speed`. These values drive `altitude_error` and `speed_error` in
`MissionSignals`, the heuristic, and the reward profiles.

**Termination:**

- `ground_impact`: altitude ≤ `ground_altitude`.
- `out_of_bounds`: horizontal range from home > `max_range` or altitude
  outside `min_altitude..max_altitude`.
- `timeout`: step budget exhausted.
- `captured_and_returned`: success.

**Specialization and mid-mission starts:**

`Specialization` is both a reward emphasis and a reset distribution:

| Specialization | Reset | Episode ends |
|---|---|---|
| `Full` (default) | home, `Outbound` | capture + return, crash, timeout |
| `Outbound` | home, `Outbound` | pickup captured, crash, timeout |
| `Capture` | `Capture` phase, 200–600 m from pickup, heading within ±60° of the pickup bearing, `cruise_altitude`, `cruise_airspeed` | captured, drift out, crash, timeout |
| `Return` | pickup point, `Return` phase, captured flag latched | home reached, crash, timeout |

`Full` is the primary training and exam distribution. Phase specialists are
scored on both the full exam and a phase slice (the subset of scenarios that
starts in their phase), so the selector's competence table is meaningful. A
specialist is never judged only on its own slice.

**Reward parts instead of one reward:**

The environment returns `MissionSignals` each step:

```
progress_outbound   potential-based progress toward pickup
progress_inbound    potential-based progress toward home
altitude_error      |altitude − phase target altitude|
speed_error         |airspeed − phase target speed|
capture_alignment   zone distance + speed margin + dwell fraction
time_penalty        constant per step
energy_penalty      throttle integral / control effort
crash_penalty       large negative on termination
capture_bonus       large positive once
return_bonus        large positive once
```

Each training run supplies a `RewardProfile` (named or TOML file) that weights
these parts. The heuristic, PPO, A2C, SAC, and TD3 may each use a different
profile. The evaluator always uses the same scorecard, not the training reward.

Default `balanced` profile:

```
progress_outbound   +1.0 per metre of progress, clamped per step
progress_inbound    +1.0 per metre of progress, clamped per step
altitude_error      −0.02 per metre
speed_error         −0.05 per m/s
capture_alignment   −1.0 × normalized distance and speed excess
time_penalty        −0.01 per step
energy_penalty      −0.05 × throttle
crash_penalty       −200
capture_bonus       +100
return_bonus        +200
```

Profiles are data, not code. `aggressive` (more speed), `cautious` (more
altitude/speed tracking), and `capture_focus` (heavier capture alignment) are
provided as examples.

## 8. Observation and Action Contract

**Contract version:** `mission.v1`.

**Observation vector:** 32 channels.

```
 0..18  existing 19-channel noisy avionics bus
 19     distance to pickup / 2500 m
 20     bearing error to pickup, sin
 21     bearing error to pickup, cos
 22     altitude error to pickup / 500 m
 23     distance to home / 2500 m
 24     bearing error to home, sin
 25     bearing error to home, cos
 26     phase: outbound = 0.0, capture = 1.0, return = 2.0
 27     airspeed margin above stall / 10 m/s
 28     time remaining fraction (1 → 0)
 29     capture dwell fraction (0 → 1)
 30     horizontal hook/zone offset / pickup_radius
 31     vertical hook/zone offset / vertical tolerance
```

The base 19 channels are unchanged, so existing code and sensor docs remain
valid. Mission features come from the true state and target geometry; they are
navigation data a real autopilot would have (GPS target plus air data), not
privileged physics.

Each expert may select a subset of channels through its own `ObservationSpec`
mask. The env always emits the full 32-channel vector; masks are training-side.
This is how models stay unique without breaking the contract.

**Action vector:** 4 continuous channels, unchanged:

```
[roll_cmd, pitch_cmd, yaw_rate_cmd, throttle]
```

Mapped exactly as today: roll ±0.6 rad, pitch ±0.4 rad, yaw rate ±0.8 rad/s,
throttle `0.5·(tanh(a)+1)`. All are inside the flight controller's ±45°
setpoints. Every expert, including the heuristic, outputs this vector.

## 9. Expert Zoo

Each expert is a `Policy` plus an `ExpertCard`. The card travels with the
checkpoint and records what the model is, how it was trained, and what it is
good at.

```rust
struct ExpertCard {
    id: String,                 // e.g. "sac_capture_v1"
    model: ModelKind,           // Ppo | A2c | Sac | Td3 | Heuristic
    specialization: Phase,      // Full | Outbound | Capture | Return
    contract: String,           // "mission.v1"
    reward_profile: String,     // "balanced" / path
    obs_mask: Vec<usize>,
    trained_steps: u64,
    seed: u64,
    git_commit: String,
    notes: String,
}
```

### 9.1 PPO

Existing `PpoAgent` and clipped-surrogate update, moved behind the expert
registry. Unique choices: GAE-λ, multiple epochs, minibatch whitening, entropy
bonus. Training loop: existing rollout → GAE → update, adapted to the mission
env. Kept as the reference on-policy model.

### 9.2 A2C

On-policy advantage actor-critic. Unique choices: one gradient pass per
rollout, no clipping, larger learning rate, optional n-step returns, entropy
bonus. Shares the rollout helper but not PPO's epoch/minibatch machinery.

### 9.3 SAC

Off-policy soft actor-critic. Own network with a tanh-Gaussian actor and twin
Q-critics. Replay buffer (default 200k transitions), target critics, automatic
entropy temperature, one or more gradient steps per env step after a warmup.
Reward normalization inside the trainer. Unique choices: maximum-entropy
objective, stochastic policy, high replay ratio.

### 9.4 TD3

Off-policy twin-delayed DDPG. Deterministic actor, twin Q-critics, target
policy smoothing, delayed actor updates, Gaussian exploration noise. Own
network and replay path, separate from SAC. Unique choices: deterministic
policy, noise schedule, delayed updates.

### 9.5 Heuristic

Deterministic pursuit + energy management, no learning:

- Outbound/return: bearing error → roll command; altitude and climb error →
  pitch command; speed error → throttle.
- Capture: target the pickup zone, command a low throttle to bleed speed toward
  the capture band, hold altitude, and align heading; throttle up if the speed
  margin approaches stall.
- Phase transitions use the same mission state as the env.

This expert is a baseline, a selector fallback, and a feasibility proof that
the mission is flyable before any RL model is trained.

## 10. Training Orchestration

`rl_agent` gains a model dispatcher while keeping the current CLI working.

```
rl_agent --mode train --model ppo|a2c|sac|td3|heuristic
         --mission mission|avionics          (default: avionics for compatibility)
         --reward-profile balanced|<path>
         --specialization full|outbound|capture|return
         --name <label> ...
```

- `--algo` remains an alias for `--model` for PPO-era scripts.
- `--mission` switches the environment and the observation dimension.
- Each model's loop lives in its own module and is free to differ.
- `training_common.rs` extracts the shared parts: run directory/telemetry,
  checkpoint writing, frame/control observation, eval trigger, and env factory.
- The heuristic has no training loop; `--model heuristic` is evaluated
  directly.

**Checkpoints:** keep `net.json`, `algo.json`, `trainer.json`,
`normalizer.json`, and weights. Add `expert.json` (the card),
`mission.json` (mission config), and `reward.json` (profile). Existing
checkpoints without these files still load in avionics mode.

## 11. Evaluation Suite (the shared exam)

`rl_agent --mode eval-suite --checkpoint DIR --scenarios <file> --out <path>`
runs the same fixed scenario list for every expert.

**Scenario list (default `mission_eval.toml`):**

The exam is an **explicit scenario list**, not a Cartesian product of seeds.
Every scenario pins the pickup geometry and start state, so every expert sees
exactly the same mission. Default: 30 scenarios.

- 10 bearings spread around the compass (every 36°).
- Pickup distances cycling 1.0, 1.5, 2.0 km.
- Pickup altitudes cycling 800, 1000, 1200 m.
- Calm air in v1; a `wind` field is reserved for a later weather exam.
- Same `max_steps` for every scenario.

`MissionConfig` accepts a scenario override (`pickup_north`, `pickup_east`,
`pickup_altitude`, `initial_phase`, `initial_speed`) so `eval-suite` does not
depend on reset RNG for geometry. The seed still controls sensor noise and any
other stochasticity.

**Per-episode scorecard:**

```
success              captured and returned
captured             pickup condition met
returned             home radius reached
crash                ground impact or out of bounds
timeout              step budget exhausted
time_to_capture_s
time_to_home_s
outbound_track_error mean cross-track distance to pickup
inbound_track_error  mean cross-track distance to home
min_capture_speed    lowest airspeed during the capture phase
capture_attempts     number of times the aircraft entered the pickup zone
energy               integral of throttle
reward               raw env reward (not used for ranking)
```

**Aggregate:**

- Rates (success, capture, return, crash, timeout) with counts.
- Means and standard deviations for times, errors, and energy.
- A per-phase competence table: outbound, capture, return.
- A JSON artifact plus a flat CSV row per expert for dashboards.

**Ranking:** lexicographic, not a single opaque score:

1. mission success rate
2. capture rate
3. lower crash rate
4. lower mean time to capture + time home
5. lower track error

This preserves "which handles what better" instead of collapsing it.

## 12. Hybrid Selector

`selector.rs` implements the same `Policy` trait as an expert, so the evaluator
can score the hybrid exactly like a single model.

**Policy context.** The `Policy` trait takes the normalized observation plus a
read-only `PolicyContext` supplied by the environment adapter:

```rust
struct PolicyContext<'a> {
    phase: MissionPhase,          // Outbound | Capture | Return | Done
    step: usize,                  // step within the episode
    scenario: &'a MissionScenario,
}

trait Policy {
    fn act(&mut self, obs: &[f32], ctx: &PolicyContext) -> Vec<f32>;
    fn deterministic_action(&self, obs: &[f32], ctx: &PolicyContext) -> Vec<f32>;
    fn reset(&mut self, ctx: &PolicyContext);
    fn card(&self) -> &ExpertCard;
}
```

Experts ignore `ctx` unless they need it. The selector reads `ctx.phase`
directly, so it never has to infer the phase from a normalized observation
channel. `EnvSpec` gains `fn mission_context(&self) -> Option<&MissionContext>`
(default `None`) so the evaluator can build the context without knowing the
concrete environment type.

**v1: rule-based phase gating.**

- Read the mission phase from `PolicyContext`.
- Dispatch to the expert configured for that phase.
- If no expert is configured for a phase, use the heuristic expert.
- Hysteresis: switch only when the phase has been stable for a few steps; never
  switch during the capture dwell.
- Because all experts share the same action contract, switching is safe at the
  command level. A short bumpless-transfer filter (slew-limit the new command
  for a few frames) is applied on switch.

**Configuration:** `selector.toml` maps phase → expert checkpoint, e.g.:

```toml
[phases]
outbound = "checkpoints/ppo_outbound"
capture  = "checkpoints/sac_capture"
return   = "checkpoints/heuristic"
```

**Selector acceptance:** the selector is scored on the full 30-scenario exam
and reported with counts and a 95% Wilson interval. It is considered useful if
it matches or beats the best single expert on at least one phase slice without
reducing overall mission success by more than 5 percentage points. This is a
reported result, not a hard ship gate: with 30 scenarios a single episode is
about 3.3 percentage points, so small differences are noise. The rule-based
selector is still useful as a safety baseline and a debugging tool even when
it does not beat every specialist.

**Later (not v1):** a learned contextual bandit or small gating network uses
the competence matrix and online phase outcomes to choose experts. It is
trained from the same eval rollouts, not from a separate environment. The
rule-based selector remains the fallback and the safety baseline.

## 13. Telemetry and Dashboard

`run.json` gains:

```
mission: { airframe, pickup_radius, capture_speed, max_steps, ... }
reward_profile: "balanced"
specialization: "full"
contract: "mission.v1"
expert: { id, model, obs_mask, ... }
```

`metrics.jsonl` gains mission counters per iteration: captures, returns,
mission successes, crash reasons, mean phase times. Eval results are written as
`eval.json` in the run directory (and optionally a separate `evals/` root).

`rl_dash` grouping already keys on algo/backend/network/algorithm/trainer. Add
mission and reward profile to the group key so different mission experiments do
not collapse into one tab. Show success/capture/crash rates in the KPI strip
when the run has mission telemetry. This is additive; no redesign of the
dashboard.

Godot's RL tab can keep using the CLI; adding model and mission dropdown
entries is a follow-up after the CLI is stable.

## 14. Testing Strategy

**`flight_core` mission tests:**

- Reset with a fixed seed reproduces pickup position and phase.
- Outbound → capture → return → done transition on a scripted trajectory.
- Capture requires radius, altitude tolerance, speed, and dwell simultaneously.
- Ground impact, out-of-bounds, and timeout terminate with the right reason.
- Reward parts are finite and have the expected sign.
- Observation length is 32, finite, and phase encoding is correct.
- The heuristic can complete the mission in calm air (feasibility test).

**`rl_agent` tests:**

- `Policy`/`ExpertCard` serde round-trip.
- Each expert acts within the action bounds and produces finite values.
- SAC/TD3 warmup then update changes parameters and yields finite losses.
- Checkpoint save/load round-trip for each model.
- Selector dispatches by phase and falls back to the heuristic.
- Evaluator aggregates a synthetic scorecard correctly.
- A tiny end-to-end train+eval run for each model on CPU (few steps).

**Compatibility tests:**

- Existing avionics-mode PPO training still passes its current tests.
- Legacy checkpoints without `expert.json` still load.

Tests must stay fast enough for `cargo test` on CPU. GPU paths are optional
smoke tests.

## 15. Milestones

Each milestone is independently useful and reviewable.

1. **Mission foundation** — `mission.rs`, contracts, `MissionEnv` adapter,
   heuristic expert, evaluator, scorecard. No new RL algorithms.
2. **PPO on the mission** — adapt the existing PPO loop to the mission env and
   produce the first baseline scorecard.
3. **A2C** — second on-policy family with its own loop.
4. **SAC** — replay buffer, twin critics, entropy tuning, evaluation.
5. **TD3** — deterministic twin-Q path.
6. **Hybrid selector** — rule-based phase gating plus phase-specialized runs
   trained with different reward profiles.
7. **Learned gating (optional)** — contextual bandit over the competence
   matrix, with the rule-based selector as fallback.

The first implementation pass should land milestones 1–2. That proves the
mission, the contract, the exam, and the existing PPO path before adding three
new algorithms.

## 16. Risks and Mitigations

| Risk | Mitigation |
|---|---|
| Fixed-wing capture is too hard and all models fail | Start with a generous pickup radius and dwell; heuristic feasibility test; tighten later. |
| Default airframe or capture speed makes the mission unflyable | Mission defaults to M-Sweep mini; `capture_speed` is computed from stall, not the unusable skyhook 5 m/s placeholder; feasibility test runs before model training. |
| Reward hacking (circling, avoiding return) | Potential-based progress, hard timeout, out-of-bounds termination, capture/return bonuses. |
| Mission obs break existing models | Contract versioning; avionics mode remains the default; old checkpoints load. |
| Off-policy training is slow on CPU | Modest nets, 200k buffer, configurable replay ratio; GPU remains optional. |
| Selector switching causes transients | Phase hysteresis, no switching during dwell, short command slew limit. |
| Dirty tree causes conflicts | Additive modules; avoid refactoring weather, rl_dash, and Godot beyond additive fields. |
| Cable physics is expected but not present | v1 documents discrete capture; cable-force integration is a separate follow-up. |

## 17. Decisions Left to the Builder

These are implementation calls, not open product questions:

- Exact reward weights and capture defaults, tuned against the heuristic
  feasibility test.
- Whether to move `ppo.rs` into `experts/` or wrap it in place (wrapping in
  place is preferred to keep the diff small).
- SAC/TD3 network widths, replay ratio, and warmup length on CPU.
- Whether the evaluator lives in `rl_agent` only or is also exposed as a
  library function for tests.
- Whether `eval.json` lives per run, in a separate `evals/` root, or both.

## 18. Definition of Done

- `cargo test -p flight_core` and `cargo test -p rl_agent` pass.
- The heuristic expert completes the out-and-back mission in calm air on the
  M-Sweep mini airframe and the evaluator reports a scorecard.
- PPO, A2C, SAC, and TD3 each train on the mission and produce a checkpoint
  with an expert card.
- `eval-suite` scores all experts on the same 30 explicit scenarios and writes
  JSON/CSV, including per-phase results.
- The rule-based selector can be evaluated as a policy and its result is
  reported with a Wilson interval and the phase competence table. Matching or
  beating the best single expert is reported, not a hard gate.
- Existing avionics-mode PPO training and legacy checkpoints still work.
- `docs/rl_mission_guide.md` explains the mission, contracts, model zoo,
  training commands, evaluation, and selector configuration.
