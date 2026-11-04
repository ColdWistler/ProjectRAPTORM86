# Task 10 Report — eval-suite Mode and Checkpoint-Backed Policies

**Status:** COMPLETE — all `rl_agent` tests pass (67 lib + CLI + policy).

## Implemented

1. **`RlPolicy<B>`** (`rl_agent/src/policy.rs`)
   - `new(algo, normalizer, card)` wrapping `Algorithm<B>`.
   - `Policy::act` normalizes raw obs, calls `algorithm.act`, converts tanh-space via `contracts::action_to_avionics`, returns 4 physical commands.
   - `deterministic_action` uses `algorithm.deterministic_action`.
   - `reset` and `card` no-ops / accessors.

2. **`ExpertCard::from_checkpoint`** (legacy fallback)
   - If `expert.json` exists in dir → parse it.
   - Else return legacy card: `model` mapped from `algo_id`, `specialization: Full`, `contract: mission.v1`, `obs_mask: 0..32`, `trained_steps: 0`, `seed: 0`, empty git/notes.

3. **CLI `eval-suite` mode** (`rl_agent/src/main.rs`)
   - Added `--mode eval-suite` branch in `exec`.
   - Uses `MissionEnvironment`, `HeuristicPolicy` (if `--model heuristic`) or `RlPolicy` loaded from `--checkpoint DIR`.
   - Reads `--scenarios FILE` or `default_scenarios()`.
   - Writes `eval.json` + `eval.csv` to `--out DIR` (defaults to checkpoint dir).
   - Prints success / capture / crash rates with Wilson intervals.
   - Added `model_id()` helper; updated `USAGE` text.

4. **Usage updated**
   - Added `EVAL-SUITE MODE:` section with `--checkpoint`, `--scenarios`, `--out` flags.

5. **Tests (all pass)**
   - `rl_agent/src/policy.rs`: `rl_policy_normalizes_and_returns_a_bounded_action`, `legacy_checkpoint_card_falls_back_to_full_specialization`, plus existing heuristic/bounded checks.
   - `rl_agent/tests/mission_eval_cli.rs`: `eval_suite_writes_json_and_csv_for_the_heuristic`.
   - Full `cargo test -p rl_agent`: 67 lib + CLI + all passing.

## Key design choices
- `Policy` trait was aligned to brief (method names `act` / `deterministic_action` / `reset` / `card`; `PolicyContext` with `phase/step/captured`; `ExpertCard` with full metadata fields; `ModelKind` / `Specialization` enums updated).
- `HeuristicConfig` kept with tuned defaults (`bank_gain=1.2`, `pitch_gain=0.0015`, `speed_gain=0.03`) using `Default` derive + explicit impl removed (only derive kept).
- `eval.rs` replaced with mission-scenario interface (`default_scenarios` → `Vec<MissionScenario>`, `run_eval_suite` with `(policy, env, scenarios, max_steps, reward)`).
- `wilson_interval` clamped to `[0,1]` to avoid floating-point edge failures.

## Files changed
- `rl_agent/src/policy.rs` (rewritten)
- `rl_agent/src/eval.rs` (rewritten for mission interface)
- `rl_agent/src/main.rs` (eval-suite mode + usage)
- `rl_agent/tests/mission_eval_cli.rs` (new)
- `rl_agent/src/lib.rs` (no structural changes needed — `pub mod policy; pub mod eval;` already present)
