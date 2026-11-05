# Changelog

All notable changes to ProjectRAPTORM86 are recorded here.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and the project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

Versions before `0.2.0` were never tagged or published; `0.2.0` is the
first tagged build of the workspace.

## 0.2.0 — 2026-11-05

The first stable build. Everything below shipped together as one release:
a pure-Rust reinforcement-learning trainer with five working algorithms, the
telemetry/dashboard toolchain around it, and the mission/weather curricula
the models are trained against.

### Added

- **`rl_agent`** — a pure-Rust [burn](https://burn.dev) trainer for the
  `flight_core` avionics environment, with a CPU (`ndarray`) and GPU
  (`wgpu`) backend selected at runtime (`--backend cpu|gpu|auto`).
- **Pluggable algorithm layer** — one `Algorithm<B>` trait, one `AlgoSpec`
  registry, five models: PPO (reference), A2C, SAC, TD3, D3QN. The model is
  chosen at runtime with `--algo`.
- **On-policy / off-policy seam** — `is_off_policy`, `observe_transition`,
  `gradient_steps_per_env_step` and `set_normalizer` let the shared trainer
  drive both families from one loop.
- **Mission curriculum** — an out-and-back pickup mission in `flight_core`
  with a `mission.v1` observation/action contract, a `MissionEnv` adapter,
  and reward profiles.
- **Weather curriculum** — an RL-ready weather system (turbulence,
  precipitation, visibility, updraft/shear, curriculum stages) plus a
  weather exam suite.
- **Structured training telemetry** — per-iteration metrics, environment
  telemetry, run manifests and a live observation stream.
- **`rl_dash`** — a native desktop dashboard (egui) with a live-follow view,
  run comparison, a KDE-style live strip, and a CPU/RAM status panel.
- **Godot integration** — an RL trainer tab with per-algorithm
  hyper-parameter fields, a live RL chart, and a one-click launch of
  `rl_dash`.
- **Hybrid phase selector** and a **learned gate** for the mission phases.

### Fixed

- **SAC, TD3 and D3QN now actually train.** Target networks are exact
  copies of the online networks (burn's `Param::clone` re-initializes an
  unmaterialized parameter, so a cloned target used to be a *different*
  network), target updates are real Polyak averages, the actor's action
  stays attached to the autodiff graph, and the Bellman target is built
  from real rewards, next observations and done flags.
- **A2C no longer diverges** — one gradient pass per rollout, and the
  learnable policy `log_std` is clamped so an entropy bonus cannot turn
  the policy into pure noise.
- **The observation normalizer is frozen for the whole rollout** and
  refreshed afterwards, so the PPO importance ratio is exactly `1.0` on the
  first epoch and a replayed transition's `obs`/`next_obs` are scaled
  consistently.
- **Truncation bootstraps again** — only a real termination stops the
  bootstrap.
- **D3QN** — separate dueling critic and policy networks, double-Q
  targets, a working prioritized replay (sub-unit priorities were being
  truncated to zero), and a non-zero TD target.
- **TD3** — delayed actor updates, target-policy smoothing noise scaled by
  the action-noise standard deviation, and a configurable Polyak rate.
- **GPU auto-detection** now validates the device with a real training
  workload instead of a trivial probe.

### Known issues

- Mission mode and the mission eval suite still build their networks with
  the 19-channel avionics observation width instead of the 32-channel
  mission width, so they panic at start-up.
- The learned gate is still hand-tuned constants rather than a trained
  model, and the phase selector never dispatches an expert.
- CPU runs are not bitwise reproducible across processes (burn's `ndarray`
  backend enables `ndarray/rayon`); the in-process same-seed test is
  exact.
