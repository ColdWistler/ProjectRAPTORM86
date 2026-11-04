# Task 8 Report: Mission Telemetry in Run Manifest

## Completed

- Added `MissionSummary` to `rl_agent/src/contracts.rs` (`Serialize`/`Deserialize`, `from_config`).
- Added `AVIONICS_CONTRACT = "avionics"` for default run labeling.
- Extended `RunMeta` (`rl_agent/src/telemetry.rs`) with `mission`, `reward_profile`, `specialization`, `contract`, `expert`.
- Extended `Manifest` with same fields (with `skip_serializing_if`) and populated from `meta` in `TelemetryWriter::start`.
- Updated all `RunMeta { ... }` constructions in `telemetry.rs` (tests) and `trainer.rs` (`run_with_observer`; `mission: None`, `contract: "avionics"`).
- Added `manifest_records_mission_metadata` test to `telemetry.rs`; passes.

## Verification

- `cargo test -p rl_agent`: 74 passed, 0 failed.
- `cargo test -p rl_agent telemetry::tests`: 6 passed (includes new manifest test).
- `cargo test -p rl_agent --test mission_heuristic`: 2/2 passed.

## Key choices / notes

- `MissionSummary` uses `flight_core::MissionConfig` fields plus `airframe`/`reward_profile`; `from_config` computes from config + capture_speed.
- `Manifest` keeps `contract` always present ("avionics" or "mission.v1") and skips optional mission/reward/specialization/expert when `None`.
- `trainer.rs` references updated to supply the five new `RunMeta` fields so existing avionics training is unchanged.

## Files changed / created

- Modified: `rl_agent/src/contracts.rs`
- Modified: `rl_agent/src/telemetry.rs`
- Modified: `rl_agent/src/trainer.rs`
- Unchanged: `rl_agent/Cargo.toml`
- Report: `task-8-report.md`
