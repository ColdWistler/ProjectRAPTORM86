# Hybrid Selector — Task Report

**Milestone:** 6 (Hybrid Selector) per `.superpowers/sdd/2026-11-04-rl-mission-foundation/task-selector-brief.md`.
**Commit:** `feat(rl_agent): add hybrid selector`
**Files changed:** `rl_agent/src/selector.rs` (new), `rl_agent/src/lib.rs` (+ module wire).
**Constraints:** No weather (`weather.rs`) or Godot (`godot/`) changes.

## Components delivered
- `SelectorConfig` — rule-based phase gate (`HashMap<String, String>` mapping phase key → expert label) with hysteresis steps (`hysteresis_steps`), bumpless throttle slew (`bumpless_slew_max`), and heuristic fallback flag (`heuristic_fallback`).
- `PhaseSelector` — implements `Policy` trait (same as experts) so it can be evaluated with the same contract. Uses `PolicyContext` (Task 2 / `MissionContext`) for phase gating.
- `SelectorStatus` — `Stable`, `Transitioning`, `FallbackHeuristic`.
- `select_policy()` — standalone helper using `MissionContext` fields (`phase`, `step`, `captured`) via `PolicyContext`; applies hysteresis, switches only after `hysteresis_steps` stable, and falls back to heuristic when configured.

## Key behaviors
- **Rule-based phase gating:** reads `ctx.phase` (from `MissionPhase`) and selects expert label mapped in `SelectorConfig::phases`.
- **Hysteresis:** `select_policy()` increments `stable_cnt` only when phase stays the same; switches expert label only when `stable_cnt >= hysteresis_steps`.
- **Bumpless throttle transfer:** `apply_bumpless()` limits throttle delta per step to `bumpless_slew_max` (default `0.05`) to keep throttle continuous across phase switches.
- **Heuristic fallback on NaN / out-of-range:** `is_valid_action()` checks `is_finite()` and bounds (`abs <= 1.0`); if invalid, `status` is set to `FallbackHeuristic` and the heuristic expert is used.
- **Uses `MissionContext`:** `PolicyContext` embeds `MissionPhase`, `step`, `captured` — matching `MissionContext` (Task 2) fields.

## Tests (all passing)
- `selector_config_defaults`
- `phase_gate_uses_hysteresis`
- `hysteresis_prevents_rapid_flip`
- `bumpless_transfer_keeps_throttle_continuous`
- `heuristic_fallback_on_nan`
- `select_policy_standalone_uses_mission_context`
- `policy_trait_implemented`

## Verification
- `cargo test -p rl_agent --lib -- selector` → 7 passed.
- `cargo test -p rl_agent` → no new failures (pre-existing `ppo::tests::act_is_deterministic_under_same_seed` remains unchanged).
- No changes to `flight_core/src/weather.rs`, `godot/`, or any unrelated modules.
