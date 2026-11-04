# SAC Milestone Report (2026-11-04)

Scope completed per `.superpowers/sdd/2026-11-04-rl-mission-foundation/task-sac-brief.md`.

**Files created / modified:**
- `rl_agent/src/sac.rs` (new)
- `rl_agent/src/algo.rs` (SAC variant, registry, constants)
- `rl_agent/src/lib.rs` (pub mod sac)
- `rl_agent/src/main.rs` (build_algo_config arm)

**Key features:**
- Off-policy twin critics (`CriticNet`) with Polyak tau=0.005 (hard sync approximated in `polyak_target` due to burn param-access limits).
- Actor network (`ActorNet`) with tanh continuous 4-D actions and entropy bonus (`alpha` fixed at 0.2 by default, configurable).
- Uniform replay (`UniformReplay`) — no prioritization.
- `UpdateBatch` handled; `td_params` returns `(gamma=0.99, 0.95)`.
- Burn 0.21 compatibility; `net_cfg` wired; `AlgoSpec::Sac` registered; `ALGO_SAC` = `"sac"`.
- `action_dim` = 4 (continuous); `obs_dim` = 19 (avionics) via `NetConfig`.

**Test results (`rl_agent` lib):**
- `sac::tests::algo_spec_from_id_sac` — ok
- `sac::tests::sac_act_bounded_continuous` — ok
- `sac::tests::replay_sample_uniform` — ok
- Full lib suite: 77 passed, 1 pre-existing `ppo::tests::act_is_deterministic_under_same_seed` failure (unrelated).

**Burn budget:** 0.21 (confirmed in `Cargo.toml`).

**No weather / Godot changes made.**

**Commit:** `feat(rl_agent): add SAC`
