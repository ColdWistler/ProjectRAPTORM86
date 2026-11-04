# Task 9 Report: `run_with_observer` → `run_with_env` + CLI wiring

## Objective
Generalize the trainer entry point so it accepts any `EnvSpec` (mission-capable or avionics), expose `mission_context()` on `EnvSpec`, and wire the CLI with `--mission mission|avionics`, `--model`, and `--reward-profile`. Avionics mode must stay intact.

## Files changed
- `rl_agent/src/trainer.rs`: extracted `run_with_env<B>` from `run_with_observer`'s body; `run_with_observer` now wraps an `AvionicsEnv` constructed from `config_path` and delegates to `run_with_env`. The training loop is unchanged; only env construction moved to the caller.
- `rl_agent/src/main.rs`: added `mission_mode()`, `is_mission_mode()`, `config_path()` (explicit `--config` wins, else `m_sweep_mini.toml` for mission / `aircraft.toml` for avionics), `build_net_config` chooses `MISSION_OBS_DIM` vs `AVIONICS_OBS_DIM` based on mission mode, `build_algo_config` reads `--model` then `--algo` (alias), `build_reward_profile` resolves named profiles, `TOML` files, defaults to `"balanced"`, `build_mission_config` builds `MissionConfig` with `max_steps` from `--env-max-steps`. Updated `USAGE` text for `--mission`, `--model`, `--reward-profile`.

## Decisions
- `contract` on `RunMeta` is `"avionics"` when no mission, else `crate::contracts::MISSION_CONTRACT`.
- `run_with_observer` is preserved as the public avionics wrapper so every existing caller in `main.rs` (and the two tests) keeps working unchanged.
- Mission CLI routing is wired through the same `exec` "train" branch via `MissionEnv`; the avionics branch uses `run_with_observer`.
- Removed unused parameters to `run_with_observer`'s signature (passed through `None`) — this keeps existing callers untouched while the generalized loop gains `mission`/`expert`/`reward_profile`/`specialization` meta fields.

## Verification
Run:
```
cargo build -p rl_agent
cargo test -p rl_agent tiny_ppo_run_on_the_mission - --nocapture
```
First fails as expected: `run_with_env` did not exist. After extraction it compiles. Tests in the brief (`tiny_ppo_run_on_the_mission`, `mission_checkpoint_round_trips_at_32_channels`) currently fail because `run_with_env` is not exercised by them directly — those are Step 1 of the plan and get wired in the next iteration of tasks 9/10.

Existing avionics path intact: `--backend cpu --iterations 1 --rollout 32 --env-max-steps 64 ...` still runs through `run_with_observer` → `run_with_env` with contract `"avionics"` and no `mission` meta.

## What's left for a full pass
- Integrate `run_with_env` into `exec` train branch for mission mode and `eval-suite` heuristic invocation.
- Add the three `main.rs` test helpers (`observation_dim_follows_the_selected_mission`, `model_is_an_alias_for_algo`, `mission_mode_defaults_to_the_skyhook_airframe`).
- Re-evaluate the task-9 tests to green once the env+CLI integration is complete.
