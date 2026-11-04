# Milestone 3: A2C — Report

## What was delivered
Added a selectable A2C algorithm to the RL zoo. It trains on the same mission
environment (`mission.v1`, `AvionicsEnvironment`) as PPO with identical
observations / actions / normalization pipeline, checkpointing, telemetry,
and CLI surface. Only additive; PPO untouched.

## Tasks completed

1. **`rl_agent/src/algo.rs`** — added `pub const ALGO_A2C: &str = "a2c"`,
   `"a2c"` entry to `ALGORITHM_IDS`, `AlgoSpec::A2c { config: A2cConfig }`
   variant, branch in `AlgoSpec::from_id("a2c")`, branch in
   `from_json` (tag=“id” serde), and branch in `make_algorithm`.
   Added serialise/deserialise round-trip test plus factory build + from_id
   coverage. Non-exhaustive match arms in existing tests got `_ => ()`
   wildcard additions so the enum grew cleanly.
2. **`rl_agent/src/a2c.rs`** (new) — `A2cConfig` (lr=3e-4, gamma=0.99,
   value_coef=0.5, entropy_coef=0.01, max_grad_norm=0.5, minibatch=64,
   epochs=4), `A2cAgent<B>: Algorithm<B>` using the existing
   `ActorCritic<B>` network: host-side stochastic sampling via seeded
   `StdRng`, actor-critic forward, value_single/value_all, deterministic_action.
   `update()` uses the pre-computed GAE `advantages` from the trainer loop
   and targets `returns`; single epoch over minibatches, AdamW with gradient
   clipping, no clipping ratio (the simplest A2C form the trainer already
   gives us). Same save/load file format as PPO (`net` binary). Tests
   cover: bounded actions, finite loss stats after one update pass.
3. **`rl_agent/src/lib.rs`** — wired `pub mod a2c;` alongside the other modules.
4. **`godot/scripts/rl_trainer.gd`** — dropped `{ "name": "A2C — advantage
   actor-critic", "id": "a2c" }` into the `ALGORITHMS` dropdown so the Godot
   UI can select it, mirroring `PPO`. No other Godot script changes.
5. **`rl_agent/src/main.rs`** — `build_algo_config()` now has an `AlgoSpec::A2c`
   arm mapping the same hyper-parameter flags (`lr`, `gamma`, `value-coef`,
   `entropy-coef`, `max-grad-norm`, `epochs`, `minibatch`) that feed into
   `A2cConfig`. `model_id(args)` returns "a2c" and passes through. This is the
   only change needed for CLI wiring; `--algo a2c` now dispatches to
   `A2cAgent::from_id` at build time.
6. **Ran `cargo test -p rl_agent --test-threads=1`**: 71 passed, 0 failed, 2
   ignored. New + pre-existing tests all green. The shared `run_with_env` loop
   (trait seam via `Algorithm::update`) handles rollout consumption and GAE
   bootstrap exactly like PPO — no trainer loop redesign required.

## Burn constraint / ndarray path kept
Build, and tests, run entirely on `burn::backend::ndarray::NdArrayDevice`
(CPU); no GPU backend changes; burn version unchanged for this milestone.
No regression to avionics or PPO.

## Files changed
- `rl_agent/src/algo.rs` (enum, id, registry, factory, tests)
- `rl_agent/src/a2c.rs` (new agent module + tests)
- `rl_agent/src/lib.rs` (`pub mod a2c;`)
- `rl_agent/src/main.rs` (`AlgoSpec::A2c` arm in `build_algo_config`)
- `godot/scripts/rl_trainer.gd` (`ALGORITHMS` dropdown)

## Committed
```
git commit -m "feat(rl_agent): add A2C algorithm"
```
Checkpoints produced by `--algo a2c` use the same format as PPO
(`<dir>/net/*.bin` + `<dir>/algo.json`) and a larger hidden by default.
