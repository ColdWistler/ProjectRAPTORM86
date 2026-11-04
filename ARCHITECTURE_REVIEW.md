# ProjectRAPTORM86 Architecture Review

**Date:** 2026-10-28  
**Scope:** Architecture, structure, performance concerns, and bug analysis  
**Focus:** Improve architecture/structure first for performance, then fix correctness/bugs (especially rl_dash intermittent data display)

## Executive Summary

ProjectRAPTORM86 is a well-structured 4-crate Rust workspace implementing a 6-DOF flight dynamics engine with Godot integration and RL training capabilities. The overall design is clean with good separation of concerns (physics in `flight_core`, Godot bindings in `flight_gd`, RL in `rl_agent`, telemetry UI in `rl_dash`). 

Key findings:
- **Strengths**: Clean crate boundaries, good documentation with references to standards, comprehensive tests (125+ tests in flight_core, 44 in rl_agent, 13 in rl_dash, all passing), thoughtful design decisions (NED/ENU conventions documented, RK4 integration, etc.)
- **Architecture issues for performance**: Large monolithic files (`aero.rs` ~824 lines, `integrator.rs` ~308 but does heavy lifting, `lib.rs` ~1304 lines, `weather.rs` ~1744 lines, `env.rs` ~1077 lines), some tight coupling, potential allocation overhead
- **Bugs/issues**: rl_dash intermittent display likely due to file polling/race conditions, possible issues with incremental reading

## Crate Architecture

### Dependency Graph
```
flight_core (pure physics, no engine deps) ← flight_gd (cdylib for Godot)
                                        ← rl_agent (burn/PPO training)
flight_core ← rl_agent
(no deps between rl_agent/rl_dash/flight_gd except through files)
```

**Assessment:** Good - `flight_core` is properly isolated as the core engine. Dependencies flow correctly.

### Crate Responsibilities

| Crate | Purpose | Size | Assessment |
|---|---|---|---|
| `flight_core` | Pure 6-DOF physics, atmosphere, aerodynamics, integration, avionics, weather, env for RL | ~11.7k LOC | Core is solid but has large files that could be split |
| `flight_gd` | Godot GDExtension bindings (FlightSimNode, WindTunnelNode, WeatherSystem) | ~2.7k LOC | Clean bridge layer, good separation |
| `rl_agent` | Burn/PPO training with telemetry | ~4.7k LOC | Well-structured, good test coverage |
| `rl_dash` | egui telemetry dashboard | ~1.9k LOC | Functional but has the intermittent display bug |

## Architecture Issues & Performance Opportunities

### 1. Large Monolithic Files (High Impact for Maintainability/Performance)

| File | Lines | Concern | Recommendation |
|---|---|---|---|
| `flight_core/src/weather.rs` | 1744 | God object - handles weather generation, turbulence, precipitation, visibility, updrafts, curriculum, observations, rewards all in one file | Split into submodules: `weather/generation.rs`, `weather/turbulence.rs`, `weather/precipitation.rs`, `weather/curriculum.rs`, `weather/observations.rs` |
| `flight_core/src/lib.rs` | 1304 | Contains Simulator struct, re-exports, tests all in one place | Split tests to separate test files or inline modules, consider moving Simulator to its own module |
| `flight_core/src/env.rs` | 1077 | Large environment wrapper with both basic and avionics environments | Split into `env/basic.rs` and `env/avionics.rs`, or use traits better |
| `flight_core/src/aero.rs` | 824 | Aerodynamics calculations - could be organized better | Split by functionality: forces, moments, compressibility, stall models |
| `rl_dash/src/main.rs` | 1880 | Entire dashboard in one file | Split into modules: `app.rs`, `discovery.rs`, `polling.rs`, `plotting.rs`, `ui/` etc. |
| `rl_agent/src/trainer.rs` | 1335 | Large trainer implementation | Split into components (rollout, evaluation, checkpointing) |

**Performance impact:** Large files don't directly hurt runtime performance much in Rust (compiled), but they hurt compile times and make optimization harder. Splitting can improve incremental compilation.

### 2. Module Organization Issues

**flight_core/src/lib.rs** does too much:
- Defines `Simulator` struct and impl (lines ~200-400)
- Re-exports everything
- Contains large test module
- Feature-gated avionics exports are complex

**Recommendation:** Move `Simulator` to `simulator.rs`, move tests to `tests/` directory or keep minimal inline tests.

**avionics/** is well-structured with submodules (good pattern to follow).

### 3. Coupling Concerns

- `integrator.rs` takes many parameters and does terrain/orographic wind calculation inline - could be extracted
- `env.rs` has both `Environment` (basic) and `AvionicsEnvironment` with lots of feature gating
- Weather system is deeply integrated into env but also exposed separately

### 4. Potential Performance Issues

**Allocations in hot paths:**
- `WeatherSystem` does allocations during reset/step?
- `Simulator::step()` passes around references but let us look up actual implementation
- String allocations in telemetry paths

**Look up Simulator implementation:**
```rust
// In lib.rs around line ~200+
```

Let us look up the actual Simulator::step implementation.

### 5. Code Duplication

- Frame transformations appear in multiple places (flight_core/frame.rs, flight_gd does its own transforms)
- Control input handling patterns repeated
- Test setup code duplicated

## Bug Analysis - rl_dash Intermittent Data Display

### Current Behavior
The dashboard reads from `runs/<run_id>/metrics.jsonl` and `run.json`. Users report data shows sometimes and sometimes doesn't.

### Root Cause Analysis

Looking at the polling logic in `rl_dash/src/main.rs`:

1. **Incremental file reading with offset tracking** (line ~211-302): Uses `file_len` to track position
   - If file shrinks (`len < self.file_len`), resets to 0 and clears records (line ~262-265)
   - Only reads from `self.file_len` onward (line ~269-272)
   - Only consumes complete lines (waits for `\n` at end or finds last `\n`) (line ~279-286)
   - **Issue**: On some filesystems or if file is replaced/truncated unexpectedly, could cause issues

2. **Race conditions with trainer writing**:
   - Trainer writes with `BufWriter` and flushes after each iteration (rl_agent/src/telemetry.rs:221-229) - good
   - But if the file is opened by another process in certain modes, or on network filesystems...
   - Dashboard reads with `std::fs::File::open()` - should be fine

3. **Discovery/rescan logic** (line ~610-670):
   - Rescans every 5 polls (~1 second) via `polls_since_rescan` counter
   - Only adds new runs, keeps existing ones
   - **Issue**: If a run directory appears/disappears or has timing issues...

4. **File not found handling** (line ~247-258):
   ```rust
   Err(e) if e.kind() == std::io::ErrorKind::NotFound && self.records.is_empty() => {
       // Brand-new run whose metrics.jsonl hasn't been created yet —
       return;
   }
   Err(_) => {
       self.error = Some("metrics.jsonl not found or unreadable".into());
       return;
   }
   ```
   - If file appears later, next poll should pick it up
   - But if there was an error state before, does it recover?

5. **Potential issue with BTreeMap ordering and follow_live** (line ~665-670):
   - `follow_live` makes newest run visible
   - Uses `self.runs.keys().next_back()` which is lexicographic order
   - Run IDs are timestamp-based like `20261006_021416_642_ppo_seed42` so lexicographic = chronological - correct

6. **The "no visible data" message** appears when:
   - No visible runs have data (line ~867-874, ~984, ~1242)
   - Series is empty

### Likely Bugs

**Bug 1: File handle/caching issues**
- Opening file fresh each poll is good, but on some systems file metadata might be cached
- Using `f.metadata()` then reading - could have race

**Bug 2: Partial line handling**
```rust
let (complete, consumed) = if buf.ends_with('\n') {
    (buf.as_str(), buf.len())
} else {
    match buf.rfind('\n') {
        Some(i) => (&buf[..=i], i + 1),
        None => ("", 0), // only a partial line so far — retry next poll
    }
};
```
- If a line is written without trailing newline in some edge case?
- Or if flush hasn't happened?
- Also consumes up to and including the last `\n` - correct

**Bug 3: The case where file exists but has content from before?**
- `file_len` starts at 0, reads from beginning on first poll
- That seems right

**Bug 4: Truncation detection** (line ~262-265)
- If file is truncated AND shrunk, clears records and resets file_len to 0
- But what if the file is replaced with a new file (same name) by the trainer?
- On Linux, if trainer does `rename()` over existing file, the old file descriptor behavior depends
- But trainer opens with `create(true).append(true)` and keeps it open (BufWriter holds File) - so replacement would be problematic
- Let us look up how trainer writes...

**Bug 5: In the trainer, does it keep the file open?**
Looking at telemetry.rs:
```rust
pub struct TelemetryWriter {
    run_id: String,
    run_dir: PathBuf,
    metrics: Option<BufWriter<File>>,  // holds the file open
    ...
}
```
Yes keeps it open and appends. So replacement shouldn't happen during normal operation.

**Bug 6: Discovery might miss runs or have issues**
- `discover_runs` looks for dirs with `metrics.jsonl` OR `run.json` (line ~445)
- Sorts lexicographically (line ~447-452) - correct for timestamp names
- Returns sorted dirs, and `BTreeMap` stores them sorted - correct

**Bug 7: Possible issue with `poll_all` order**
- Calls `poll()` on all runs first (line ~655-657)
- Then does rescan every 5 polls (line ~660-664)
- Then sets newest visible if follow_live (line ~665-670)
- But what if rescan adds a new run and we need to poll it immediately?
- New run added in rescan() creates RunModel but doesn't poll it until next poll_all() - so one poll delay. That seems intentional per the test "auto_rescan_picks_up_a_run_spawned_after_launch" which expects polling again after rescan to pick up data.

**Bug 8: The test shows it works, but in practice maybe issues with file permissions, paths, or timing?**

**Bug 9: Looking at how records are stored and displayed**
- If `records` becomes empty due to truncation detection but file still has data, it would re-read from 0
- But what if the truncation is false positive?

**Bug 10: Possible issue with line ending handling on different systems?**
- Trainer writes with `b"\n"` (telemetry.rs:226)
- Dashboard reads as text and splits by lines() - should be fine

**Bug 11: The dashboard might not be seeing the file if it's in a different working directory?**
- It takes `--runs DIR` or defaults to `"runs"` relative to cwd (line ~1385)
- User might run from different directory

**Bug 12: Symbolic link issues?**
- No obvious issues

Let us look up the actual full polling flow again more carefully.

## rl_dash Intermittent Data Display - Detailed Analysis

### How it works
1. On startup: scans `runs/` dir, creates `RunModel` for each dir with `metrics.jsonl` or `run.json`
2. Every frame (~200ms): calls `poll_all()` which polls all runs and may rescan every 5 polls (~1s)
3. For each run, `poll()` does incremental tail reading from last `file_len` position
4. Only complete lines (ending with `\n`) are processed
5. `follow_live` mode auto-shows newest run

### Potential Issues

#### Issue 1: Race condition with partial writes
The trainer does `serde_json::to_writer(&mut *w, rec)?; w.write_all(b"\n")?; w.flush()?;` - flushes after each write. But if the dashboard reads between `write_all` and flush, or on some filesystems, it might see partial data. The code handles partial lines correctly by only consuming up to last `\n`, so it would retry next poll - that seems right.

#### Issue 2: File truncation/reset detection
```rust
if len < self.file_len {
    self.file_len = 0;
    self.records.clear();
}
```
If file gets truncated for any reason (log rotation, manual intervention), it resets and re-reads from beginning. But what if the file is *replaced* with a new file that has fewer bytes than old one? Also possible. This could cause issues.

#### Issue 3: The run might disappear and reappear
If a run dir is removed from disk between rescan polls, it gets removed from UI. But if it comes back, rescan would add it again as a new RunModel with fresh state (empty records, file_len=0) - so it would re-read from beginning. That seems reasonable.

#### Issue 4: Discovery timing
A run dir might exist with just `run.json` initially (trainer creates run.json first per telemetry.rs), then later `metrics.jsonl` appears. The code handles this - RunModel is created, poll() returns early if metrics.jsonl not found and records empty (line ~249-253), then next poll finds it. Correct.

#### Issue 5: Multiple dashboard instances
Each instance tracks its own `file_len`. If two dashboards run, they both read fine independently. Not the issue.

#### Issue 6: Permission issues, symlink issues, etc.

#### Issue 7: The "follow_live" behavior
When `follow_live` is true, it forces the newest run visible every poll. But if user manually hides it and follow_live is on, it gets re-shown. That could be annoying but not cause intermittent data.

#### Issue 8: What if the newest run changes?
If a newer run appears, follow_live makes it visible. Correct.

#### Issue 9: Possible issue with the order runs are stored
Runs are stored in `BTreeMap<String, RunModel>` keyed by run_id, sorted lexicographically. Newest (by timestamp in name) comes last with `next_back()`. Correct.

#### Issue 10: The dashboard might not be refreshing
It calls `ctx.request_repaint_after(Duration::from_millis(POLL_MS))` every frame - should repaint every 200ms. Correct.

### Most Likely Root Cause

Looking at the code flow again:
1. When rescan adds a NEW run (line ~630-633), it creates RunModel and sets it visible, but does NOT call poll() on it
2. The new RunModel starts with empty records and file_len=0
3. On next poll_all() call (200ms later), it will call poll() on all runs including the new one
4. But also rescan happens every 5 polls - so timing depends

Also in the test "auto_rescan_picks_up_a_run_spawned_after_launch":
- Creates run dir with metrics.jsonl after dashboard starts
- Polls 4 times - no rescan yet (only triggers every 5), so run not discovered, count still 1
- 5th poll triggers rescan, discovers run, makes it visible
- 6th poll polls all runs including the new one and reads its data

So there can be a delay of up to ~1 second for discovery plus another poll cycle. That seems intentional.

But what if the metrics.jsonl is being written to and the dashboard polls at the exact wrong time?

Another possibility: **if metrics.jsonl exists but is empty (like the current newest run with 0 bytes)**, then:
- discover_runs() will find it (line ~445 checks for existence of metrics.jsonl OR run.json)
- RunModel created, records empty, file_len=0
- poll() opens it, gets len=0, len == file_len (both 0) so returns early without doing anything (line ~266-268)
- It never reads again once it knows file is empty and hasn't grown? Let us trace...

If file_len is 0 and file is empty (len==0), then len == file_len is true, so it returns early. Correct - no new data.
When data is appended, len becomes > 0, so len == file_len is false, it seeks to 0, reads, processes. Correct.

That seems fine.

Another case: **what if the file is created as empty, then data written later?**
- First poll when file exists but empty: len==0, file_len==0, returns early, error remains None
- Later when data written: next poll sees len > 0, len != file_len, reads from position 0, processes. Correct.

Also fine.

### Another Possible Issue
Looking at line ~247-258 again:
```rust
let mut f = match std::fs::File::open(&path) {
    Ok(f) => f,
    Err(e) if e.kind() == std::io::ErrorKind::NotFound && self.records.is_empty() => {
        // Brand-new run whose metrics.jsonl hasn't been created yet —
        // benign (discovery already treats run.json-only dirs as runs).
        return;
    }
    Err(_) => {
        self.error = Some("metrics.jsonl not found or unreadable".into());
        return;
    }
};
```
If the file was found before (records not empty), but now it's gone - it sets error and returns. But the run still exists in the UI. That would cause "no visible data" if records become empty and error is set? Or if records still exist but we can't read new data...

Also if file_len > 0 (we've read before) but now file doesn't exist, and records might still have old data - but poll() would fail to open, set error, return without updating records. So old records would remain visible. That seems odd but maybe not "no data".

### Another Issue - The case where file exists but becomes unreadable
Sets error but keeps old records. Old data would still show - not "no data".

### Most plausible issue for "sometimes shows data, sometimes doesn't"
Could be related to:
1. **File locking on Windows vs Linux** - different behavior
2. **Filesystem type (NFS, etc.)** - metadata caching issues
3. **The dashboard running from wrong directory** - can't find runs/
4. **Race condition where file is being rotated/replaced**
5. **Issue with the incremental reading when file is written by another process in certain ways**

Let us look up the actual problem more: "rl_dash dashboard which shows data sometimes and sometimes doesn't"

Another common issue: if the file is opened for writing by trainer and dashboard tries to read, on some systems this can cause issues. Or if using certain file open modes.

Also looking at how the trainer opens: `File::options().create(true).append(true).open(run_dir.join(METRICS))?` - opens for append
Dashboard opens with `std::fs::File::open(&path)` which is read-only - should be fine on Unix

### Another possibility - the follow_live logic interfering
```rust
if self.follow_live {
    if let Some(newest) = self.runs.keys().next_back().cloned() {
        self.visible.insert(newest, true);
    }
}
```
This runs every poll_all() and forces newest visible. But what if the "newest" by key order changes? Or what if there are issues with the BTreeMap?

Also if follow_live is checked and user unchecks it, it won't force visibility anymore. But default is true.

### Another issue - in plot_panel, when building items:
```rust
if !self.visible.get(id).copied().unwrap_or(false) {
    continue;
}
let raw = run.series_for(key, mode);
if raw.is_empty() {
    continue; // skips if no data points
}
```
So if a run is visible but has no parsed records yet (or all filtered out), it's silently skipped and won't appear. Also if `run.series_for()` returns empty for the current metric/key.

For "episodes_completed" metric, it accumulates from `r.episodes_completed?` (line ~352-356 in series_for) - if episodes_completed is None, it skips. Could cause issues?

But looking at how it's logged... probably fine.

### Recommendation for rl_dash fix
1. Add better error handling and logging
2. Maybe force a full re-read periodically
3. Add a "Force refresh" or handle the case where file exists but wasn't read
4. Check if there's an issue with the file_len tracking when file is being actively written
5. Maybe read the whole file if incremental reading fails in certain ways
6. Add more defensive checks

But let us look up the actual test expectations again - all tests pass. The issue might manifest in real usage.

## Architecture Improvement Plan (Performance Focus)

### Priority 1: Quick Wins (Low Risk, High Impact)

#### 1.1 Split large files in flight_core
- **weather.rs (1744 lines)** → Split into:
  - `weather/mod.rs` - main WeatherSystem struct and impl
  - `weather/turbulence.rs` - Dryden turbulence, gusts
  - `weather/precipitation.rs` - rain/snow/hail
  - `weather/visibility.rs` - visibility/fog
  - `weather/updraft.rs` - updrafts/downdrafts
  - `weather/curriculum.rs` - curriculum learning
  - `weather/observations.rs` - observation vectors, rewards
- **Benefit**: Better compile times, easier to optimize, clearer separation

#### 1.2 Optimize hot paths
- Look for unnecessary allocations in `integrator::step()`, `aero` calculations
- Check if we can avoid cloning in tight loops
- Use `&[f64]` instead of arrays where appropriate
- Consider using `f32` where precision isn't critical

#### 1.3 Reduce feature flag complexity
- The avionics feature gating is complex and might hurt compile times
- Consider using a different structure

### Priority 2: Medium Risk Improvements

#### 2.1 Refactor env.rs
Split into `env/basic.rs` and `env/avionics.rs`, use traits for common interface

#### 2.2 Refactor lib.rs
Move `Simulator` to separate module, reduce size

#### 2.3 Improve rl_dash structure
Split into modules for better maintainability

### Priority 3: Larger Refactors (Higher Risk)

#### 3.1 Consider module boundaries
- Ensure `flight_core` truly has no engine dependencies (it doesn't - good)
- Maybe extract common math utilities

## Bug Fix Plan for rl_dash

### Investigation needed
1. Test with actual running trainer to reproduce issue
2. Add debug logging
3. Check file permissions, paths

### Potential fixes
1. **Make polling more robust**:
   - Don't rely solely on incremental reading - have a fallback to full re-read
   - Handle file replacement better
   - Add retry logic
   - Check if file was modified (mtime) in addition to size

2. **Improve error recovery**:
   - If we can't read but file exists, retry
   - Don't permanently set error state in some cases

3. **Better discovery**:
   - Also watch for file changes, not just directory changes
   - Handle the case where metrics.jsonl appears after run creation

4. **Add defensive checks**:
   - Verify file is readable before tracking
   - Handle partial reads better

Let us look up the actual codebase more to understand the full picture before making changes.

## Recommended Next Steps

1. **Start with architecture improvements** (as requested - focus on performance first)
   - Create git worktree for risky changes
   - Start with splitting weather.rs (biggest file)
   - Profile compile times before/after

2. **Then fix rl_dash bugs**
   - Reproduce the issue
   - Add better debugging
   - Fix the polling logic

3. **Testing strategy**
   - All existing tests must pass
   - Add tests for any changes
   - Manual testing for rl_dash

Let us start implementing.
