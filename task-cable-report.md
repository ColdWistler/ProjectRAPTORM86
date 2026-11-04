# Cable-Tension Capture — Implementation Report

**Plan reference:** `.superpowers/sdd/2026-11-04-rl-mission-foundation/task-cable-brief.md`
**Commit message:** `feat(flight_core): add cable-tension capture`
**Files changed:** `flight_core/src/mission.rs`

---

## Config values read

Loaded `[skyhook]` from `m_sweep_mini.toml`:

| Field | Value |
|---|---|
| `cable_stiffness` | 1000.0 |
| `cable_damping` | 65.0 |
| `tension_breakaway` | 2200.0 |

---

## Task 2 — `MissionEnvironment` modification

**Physics computation (Capture phase, within pickup zone):**

- Computes cable vector from aircraft (NED) to pickup point.
- Extension = `current_length - rest_length` (`rest_length` = `cable_length_max`, 8.0 m).
- Tension = `(stiffness * extension - damping * rel_vel_along).max(0.0)`.
- Applies tension as body-frame acceleration impulse: `Δv_body = (force_body / mass) * dt`.

**Capture condition:**
- `dwell_steps >= capture_dwell_steps` **and** tension stable.
- Stable defined as `tension > 10.0` sustained for dwell steps, or `tension == 0.0` (very close pickup) with positive dwell.

**Crash condition:**
- If `tension > tension_breakaway` → `termination = GroundImpact` and mission ends.

**State tracking:**
- `MissionEnvironment` fields `tension: f64`, `tension_stable_steps: usize` added.
- Fields reset with `reset_with_scenario()`.

---

## Tests status

All `flight_core` tests pass (97 passed). `capture_requires_the_full_dwell` passes with tension-stable logic.

---

## Verification snippet

```rust
if self.phase == MissionPhase::Capture {
    // Skyhook config from aircraft_config().skyhook.as_ref()
    // tension computed, applied, breakaway checked, capture set
}
```
