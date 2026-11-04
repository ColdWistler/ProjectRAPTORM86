# rl_dash Dashboard & Visualization Improvements Plan

## Current State
- `rl_dash` is a native egui/eframe app (Glow/OpenGL) that tails `runs/*/metrics.jsonl` + `run.json` live.
- Features: Chart/Heatmap views, X-axis modes (iteration/env-steps/wall-clock), smoothing, best-so-far, log-y, eval/checkpoint markers, eval dots, seed bands (mean±std), run summary table, KPI strip, and switchable run tabs (group by model name, individual run, or architecture) that scope every view.
- Environment telemetry: each iteration's `env` block is flattened to `env_*` / `weather_*` series (altitude, airspeed, climb, attitude, wind, throttle, surface effort, sensor error, battery, weather), so ground-truth flight state can be plotted beside the training metrics.
- Integration: standalone CLI, launched from Godot RL tab with auto-launch option, shares schema with `rl_agent/telemetry.rs` and Godot chart.
- Tests: 22 passing unit tests covering polling, transforms, grouping, tab filtering, and rendering with real data.

## Goals
Make the dashboard more usable for analysis/comparison of training runs - better UX, clearer visualization, improved performance/robustness, useful features for debugging/tracking experiments.

## Improvement Areas

### 1) UI/UX Polish (High Impact, Low Risk)
- [ ] **Tighter sidebar layout**: Reduce padding, group controls into collapsible sections (View, Axes/Transforms, Filters). Make metric selector more compact.
- [ ] **Run list improvements**: Add search/filter by run name/algo/backend/status; show key stats inline (iters, latest reward) without expanding; sortable; "Hide completed"/"Hide old" filters.
- [x] **Run tabs**: Group runs by model (`--name`), individual run, or architecture. The active tab scopes the chart, KPIs, heatmap, table, bar, scatter, and summary; an All tab keeps the overlay comparison.
- [ ] **Better empty states**: When no runs found, show actionable message with path and hint to run training or use Rescan.
- [ ] **Keyboard shortcuts**: Space to pause/follow, arrows to switch runs/metrics, `h` for heatmap toggle, `r` to rescan.
- [ ] **Status bar/footer**: Show "last update Xs ago", polling rate, total runs/visible, file watching status.
- [ ] **Tooltips**: Add tooltips to toggles explaining what they do.

### 2) Visualization Enhancements (High Impact)
- [ ] **Crosshair/hover sync**: Show values for ALL visible series at cursor x-position (not just latest strip). Add vertical crosshair line.
- [ ] **Interactive legend**: Click legend items to toggle visibility/hide/show individual series.
- [ ] **Better downsampling**: For large series (>10k points), downsample for plotting (LTTB or uniform) to improve FPS. Keep full data for stats/summary.
- [ ] **Improved heatmap**: Add color scale legend with actual min/max values per row or global; option to normalize globally vs per-row; show iteration numbers on axis; better tooltips.
- [ ] **Comparison view improvements**: Align by step 0 or normalize to common range.
- [ ] **Export**: Add "Export CSV" for selected runs/metric and "Save plot as PNG".

### 3) Data/Robustness (High Impact)
- [ ] **Better file watching**: Track (inode/dev, size, mtime) for each run's metrics.jsonl to detect replacement/truncation more reliably.
- [ ] **Immediate pickup of new runs**: When rescan finds new run dirs, start polling them immediately.
- [ ] **Stale data indicator**: If a "live" run hasn't updated in >N seconds, show warning/badge.
- [ ] **Handle partial writes better**: More robust line reading.
- [ ] **Error surfacing**: Show parse errors more clearly.

### 4) Performance & Code Quality (Medium-High)
- [ ] **Modularize codebase**: Split 1887-line main.rs into modules.
- [ ] **Optimize allocations**: Cache transformed series (smoothed/best-so-far/log) when inputs unchanged.
- [ ] **More tests**: File replacement/truncation, crosshair logic, etc.

### 5) Features for Analysis (Nice to Have)
- [ ] **Run comparison table enhancements**: More columns, sparklines.
- [ ] **Multi-metric dashboard**: Grid of small charts.
- [ ] **Tags/notes**: Add notes to runs.

## Concrete Quick Wins (Implement First)
1. Crosshair showing all series values at cursor x
2. Interactive legend to toggle series
3. Better empty state + status bar with last update time
4. Stale live run indicator (>30s no updates)
5. Keyboard shortcuts (Space to toggle follow, R to rescan, H for heatmap)

## Verification
- All existing tests must pass
- Manual testing with real runs
- Check performance with large files
