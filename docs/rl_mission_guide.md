# RL Mission Guide (`docs/rl_mission_guide.md`)

**Status:** milestone 1 / foundation — mission environment, contract, heuristic, evaluator.
**Scope:** `flight_core::mission.rs`, `rl_agent::mission_env`, `policy`/`eval`, `selector` (rule-based).
**Related:** `docs/rl_agent_guide.md` (avionics trainer), `docs/rl_weather_guide.md` (weather layer), `ARCHITECTURE_REVIEW.md`, `docs/superpowers/specs/2026-11-04-rl-mission-zoo-design.md`.

---

## 1. What the mission is

Fixed-wing UAV flies a **three-phase out-and-back** with no landing, no taxi, no ground ops, no true hover. The airframe is `m_sweep_mini.toml` (small fixed-wing with skyhook config, but v1 uses a discrete geometry capture, not cable tension).

| Phase | Enter when | Target | Exit |
|---|---|---|---|
| `Outbound` | reset (home, trimmed cruise) | pickup point | captured / crash / timeout |
| `Capture` | inside pickup zone | pickup point, slow speed | captured (dwell) / drift out / crash / timeout |
| `Return` | `captured` latched | home gate | home reached / crash / timeout |
| `Done` | home reached | — | episode ends |

**Defaults (override via `MissionConfig` / scenario file):**

- Cruise altitude `1000 m`, cruise airspeed `55 m/s`
- Pickup radius `40 m`, vertical tolerance `15 m`
- Capture speed: `auto` = `1.15 × V_stall` (≈ 11.5 m/s for M-Sweep; NOT the skyhook 5 m/s placeholder)
- Capture dwell `15` steps (0.5 s)
- Home radius `100 m`, vertical tolerance `20 m`
- Pickup distance `1.0–2.0 km`, altitude `800–1200 m`
- Max steps `4500` (150 s at 30 Hz), max range `5 km`

**Reset randomization (seeded):** pickup point uniform bearing 0–360°, distance and altitude from ranges above. Start is always home at trimmed cruise. Same seed → same pickup + same sensor noise.

**Capture condition (must hold for full dwell):**
- horizontal distance ≤ pickup_radius
- vertical distance ≤ vertical_tolerance
- airspeed ≤ capture_speed
- all four simultaneously

**Termination reasons:** `ground_impact`, `out_of_bounds`, `timeout`, `captured_and_returned`.

---

## 2. Observation contract (`mission.v1`)

The env always emits **32 channels**. The base 19 are the noisy avionics bus (unchanged from `AvionicsEnvironment`); mission features (19–31) append navigation/phase data derived from true state + target geometry (not privileged physics — a real autopilot has GPS target + air data).

| Ch | Meaning | Unit / note |
|---|---|---|
| 0 | gyro_p | rad/s |
| 1 | gyro_q | rad/s |
| 2 | gyro_r | rad/s |
| 3 | accel_x | m/s² |
| 4 | accel_y | m/s² |
| 5 | accel_z | m/s² |
| 6 | gps_north | m (NED) |
| 7 | gps_east | m (NED) |
| 8 | gps_altitude | m |
| 9 | gps_fix_quality | 0..3 |
| 10 | baro_altitude | m (noisy) |
| 11 | airspeed_indicated | m/s (noisy) |
| 12 | battery_voltage | V |
| 13 | battery_capacity_remaining_pct | % |
| 14 | actual_elevator_deg | ° |
| 15 | actual_aileron_deg | ° |
| 16 | actual_rudder_deg | ° |
| 17 | actual_esc_output | 0..1 |
| 18 | sim_time | s |
| 19 | distance to pickup / 2500 m | normalized |
| 20 | bearing error to pickup, sin | — |
| 21 | bearing error to pickup, cos | — |
| 22 | altitude error to pickup / 500 m | normalized |
| 23 | distance to home / 2500 m | normalized |
| 24 | bearing error to home, sin | — |
| 25 | bearing error to home, cos | — |
| 26 | phase | 0.0=outbound, 1.0=capture, 2.0=return |
| 27 | airspeed margin above stall / 10 m/s | — |
| 28 | time remaining fraction (1→0) | — |
| 29 | capture dwell fraction (0→1) | — |
| 30 | horizontal hook/zone offset / pickup_radius | — |
| 31 | vertical hook/zone offset / vertical_tolerance | — |

Each expert may select a subset via `ObservationSpec` mask; the env always emits all 32.

---

## 3. Action contract (unchanged)

4-D continuous, tanh-squashed, mapped to attitude commands inside the flight controller's `±45°` clamps.

```
roll_cmd     = tanh(a0) * 0.6        (rad)
pitch_cmd    = tanh(a1) * 0.4        (rad)
yaw_rate_cmd = tanh(a2) * 0.8        (rad/s)
throttle     = 0.5 * (tanh(a3) + 1)  (0..1)
```

Every expert — PPO, A2C, SAC, TD3, heuristic — outputs exactly this vector.

---

## 4. Reward parts (`MissionSignals` — data, not hard-coded)

Instead of a single scalar, the env returns parts; each training run supplies a `RewardProfile` (named or TOML path) that weights them.

| Part | Default (`balanced`) | Meaning |
|---|---|---|
| `progress_outbound` | +1.0 / m (clamped) | potential-based toward pickup |
| `progress_inbound` | +1.0 / m (clamped) | potential-based toward home |
| `altitude_error` | −0.02 / m | |alt − phase target| |
| `speed_error` | −0.05 / m/s | |airspeed − phase target| |
| `capture_alignment` | −1.0 × normalized distance + speed + dwell | zone/speed/dwell penalty |
| `time_penalty` | −0.01 / step | constant |
| `energy_penalty` | −0.05 × throttle | throttle integral |
| `crash_penalty` | −200 | termination |
| `capture_bonus` | +100 | once on capture |
| `return_bonus` | +200 | once on return |

Profiles provided: `balanced`, `aggressive`, `cautious`, `capture_focus`. The evaluator always uses the same scorecard (not the training reward).

---

## 5. Commands — train / eval / eval-suite

All from repo root; `--mission mission` selects `m_sweep_mini.toml` + 32-ch contract. Default (`--mission avionics`) preserves existing behavior.

### Tiny PPO smoke (mission)
```bash
cargo run -p rl_agent --release -- --mode train --mission mission --algo ppo \
  --backend cpu --iterations 3 --rollout 128 --env-max-steps 4500 --seed 42
```

### Eval a checkpoint (deterministic episodes)
```bash
cargo run -p rl_agent --release -- --mode eval --mission mission --checkpoint checkpoints \
  --eval-episodes 3 --backend auto
```

### Eval-suite (shared exam) — heuristic (no checkpoint needed)
```bash
cargo run -p rl_agent --release -- --mode eval-suite --mission mission \
  --model heuristic --backend cpu --scenarios mission_eval.toml --out eval_results/
```

With a learned policy:
```bash
cargo run -p rl_agent --release -- --mode eval-suite --mission mission \
  --checkpoint checkpoints --scenarios mission_eval.toml --out eval_results/
```

The exam is **30 explicit scenarios** (10 bearings × 3 distances × 3 altitudes, calm air, same geometry pinned, not a random Cartesian product). It writes `eval.json` + `eval.csv`. Aggregate: success rate, capture rate, crash rate, time-to-capture/home, track errors, energy; ranked lexicographically (success → capture → crash → time → error).

---

## 6. Reading evaluation output

### `eval.json` (aggregate)
- Per-phase competence table (outbound / capture / return) with counts + 95% Wilson intervals.
- Rates: success, captured, returned, crash, timeout.
- Means / std for: time_to_capture_s, time_to_home_s, outbound_track_error, inbound_track_error, min_capture_speed, capture_attempts, energy, reward.
- Per-scenario rows (optional) with termination reason.

### `eval.csv`
- Flat row per expert / run: scenario id, success bool, capture bool, returned bool, crash bool, time_to_capture, time_to_home, track_errors, energy, reward.
- Designed for dashboards (`rl_dash`, HTML report) — load with any CSV tool.

### From telemetry (`runs/<run_id>/metrics.jsonl`)
Each line (per iteration) includes mission counters: captures, returns, mission successes, crash reasons, mean phase times, plus rollout-derived fields (`mean_episode_steps`, `crash_rate`, etc.) and the `env` block (true altitude/airspeed/climb/throttle, not just observed).

---

## 7. Compatibility notes

- **Avionics mode untouched.** `--mission avionics` (default) keeps `aircraft.toml`, 19-ch observation, existing `AvionicsEnvironment`. Existing checkpoints (`net.json` + `algo.json`) load unchanged; legacy v0.1 (`ppo.json`) upgraded automatically.
- **Mission mode is additive.** `mission_env.rs` adapts `MissionEnvironment` → trainer observer. `env.rs` keeps existing accessors; small additions for mission context (`mission_context()` → `Option<&MissionContext>`).
- **Algorithm layer unchanged.** Only `ppo.rs` is registered today (`ALGO_PPO`); new experts (A2C/SAC/TD3/heuristic) register via `algo.rs::make_algorithm`. The shared loop consumes only the `Algorithm<B>` trait.
- **Backends unchanged.** `cpu`/`gpu`/`auto` runtime-selected; `auto` probes GPU under 20 s / 0.5 ms budget and falls back. `RUST_MIN_STACK=16MiB` set before any backend thread spawn (cubecl DSD stack)
- **Telemetry format unchanged** in structure; `run.json` gains `mission`, `reward_profile`, `specialization`, `expert`, `contract: "mission.v1"`. `metrics.jsonl` gains mission counters per iteration.
- **Checkpoint extension:** add `expert.json` (card), `mission.json`, `reward.json`. Old checkpoints without them load in avionics mode; in mission mode they fall back to default config.
- **Godot RL tab:** keeps using CLI; dropdown additions for model / mission are a follow-up after CLI is stable.

---

## 8. What is intentionally NOT here (anti-goals / deferred)

Taken from design doc §3 / §4 / §9 / §11 / §15. Do not expect these in v1.

- **No landing, taxi, ground operations.** Episode ends at home gate, not on runway.
- **No true hover / station-keep.** "Capture" is a discrete slow-speed geometry event (40 m zone, 15-step dwell), not a stationary hold.
- **No cable-tension physics in v1.** `skyhook.rs` stays available; capture is geometry + speed, not winch force. The 0.35 m physical ring is not the v1 success radius (too tight for learnable v1; may tighten later).
- **No learned meta-policy / contextual gate in v1.** The `selector.rs` is rule-based phase gating with hysteresis + command slew. A learned bandit / gating network is a follow-up (after milestone 6 / 7).
- **No multi-agent, no population-based training, no fairness guarantees across budgets.** Each expert owns its loop, net, reward profile, budget.
- **No single opaque scalar score.** Ranking is lexicographic; per-phase scorecard is preserved.
- **No breaking change to `flight_core` independence.** No Godot / burn deps inside `flight_core`; `env.rs` additions are accessors, not engine rewrites.
- **No broad weather/dashboard/Godot refactors** beyond the additive mission fields already in telemetry.
- **No A2C / SAC / TD3 trainers in milestone 1.** Only heuristic + existing PPO are present; others lands at milestones 3 / 4 / 5.
- **No true "hover" capture at 5 m/s.** Capture speed is `1.15 × V_stall` (~11.5 m/s M-Sweep). 5 m/s is a future mechanism (winched cable / different capture).

---

## 9. Expert zoo (milestone 1 — heuristic + PPO only; rest follow)

| Expert | Model | Loop | Unique choices | Status |
|---|---|---|---|---|
| `heuristic` | `HeuristicPolicy` (deterministic) | none — baseline / fallback | pursuit + energy; no training; completes calm-air mission | **present** |
| `ppo` | `PpoAgent` | rollout → GAE → AdamW epochs/minibatch | GAE-λ, whitening, entropy, gradient norm clip | **present** |
| `a2c` | — | rollout → one pass, no clip, larger lr | — | milestone 3 |
| `sac` | — | replay 200k, twin Q, target nets, auto α | — | milestone 4 |
| `td3` | — | replay, twin Q, target smoothing, delayed actor | — | milestone 5 |

Every expert implements `Policy` (`act`, `deterministic_action`, `reset`, `card` with `ExpertCard`); `HeuristicPolicy` is eval-only (`--model heuristic`).

---

## 10. Selector (v1 — rule-based)

Reads `PolicyContext.phase` from env adapter (not from obs):

- Dispatch to expert configured for phase (`selector.toml`: `[phases] outbound = ...`)
- If none configured → heuristic fallback
- Hysteresis: switch only after stable phase for a few steps; never during capture dwell
- Bumpless-transfer: slew-limit command change on switch

Scored on same 30-scenario exam as single experts; considered useful if matches/beats best single expert on at least one phase slice without reducing overall success by >5 pp (reported, not a hard gate — 30 scenarios ≈ 3.3 pp per episode).

---

## 11. Quick-check commands

```bash
# 1) Tiny mission PPO (smoke)
cargo run -p rl_agent --release -- --mode train --mission mission --algo ppo --backend cpu --iterations 3 --rollout 128 --seed 42

# 2) Heuristic eval-suite (shared exam, no checkpoint)
cargo run -p rl_agent --release -- --mode eval-suite --mission mission --model heuristic --backend cpu

# 3) Avionics compatibility (existing path untouched)
cargo run -p rl_agent --release -- --mode train --mission avionics --algo ppo --backend cpu --iterations 2 --rollout 64 --seed 42

# 4) Tests (fast CPU path)
cargo test -p flight_core
cargo test -p rl_agent
```

---

*Written for milestone 1 (foundation). Additions at milestones 2–7: PPO on mission, A2C, SAC, TD3, hybrid selector, learned gating, tighter capture mode.*
