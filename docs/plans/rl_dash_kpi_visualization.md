# rl_dash: KPI Visualization Plan (Main Screen + Drilldowns)

## Outcome
Make the main screen answer "is training healthy?" at a glance with KPI cards, while keeping the existing deep views (Chart/Heatmap/Table/Bar/Scatter) available for detailed analysis.

**Acceptance criteria:**
- Top KPI strip shows key metrics for visible runs (or selected run) with deltas/trends
- Clicking/expanding a KPI opens a detailed view (sparkline + history + stats)
- Layout preserves information density (product-UI track) without hero clutter
- All existing tests pass; new UI doesn't break polling/rendering
- Works in live-follow mode

## Current State
- `rl_dash/src/main.rs`: Chart/Heatmap/Table/Bar/Scatter, crosshair, sorting/filtering, stale/live indicators
- Data: `runs/*/run.json` + `metrics.jsonl` via `RunModel` (records, manifest, x-modes, transforms)
- Metrics available: reward/step, losses, entropy, advantages, SPS, crash rate, episode stats, eval metrics

## KPI Set (Proposed)
| KPI | Source | Purpose | Display |
|---|---|---|---|
| **Latest Reward/step** | `mean_reward_per_step` last | Primary objective | Value + delta vs previous iter + sparkline |
| **Best Reward/step** | running max | Peak performance | Value + iter achieved |
| **Iterations** | `records.len()` | Progress | Count + ETA if live (optional) |
| **Env Steps** | `total_env_steps()` | Sample efficiency | Formatted (M/K) |
| **Avg SPS** | `mean_sps()` | Throughput | Value + trend |
| **Crash Rate** | last `crash_rate` | Safety/stability | % with color (warn if >5%) |
| **Eval Reward** | last `eval.total_reward` | Held-out performance | Value + Δ vs last eval |
| **Entropy** | last `entropy` | Exploration | Value (useful for PPO) |
| **Wall Time** | `last_elapsed()` | Duration | Formatted (h/m/s) |
| **Losses (P/V)** | last policy/value loss | Convergence health | Compact pair |

## UI Design (ui-design: product-UI)
- **Hierarchy**: KPI strip first (scannable), main viz below, sidebar for run/control
- **Density**: Tight cards, monospace numbers, subtle dividers
- **Color**: Use existing palette; semantic (green/yellow/red) for thresholds
- **States**: Loading/empty, stale live, no data
- **Interaction**: Click KPI card → expand to detail panel/modal with full sparkline, stats table, recent values
- **Responsive**: Wrap cards, scroll horizontally if needed

## Layout Plan
```
┌─────────────────────────────────────────────────────────────────┐
│ KPI Strip (cards: Reward, Best, Iters, Steps, SPS, Crashes, ...)│
├─────────────┬───────────────────────────────────────────────────┤
│ Sidebar      │ Main Area: selected view (Chart/Heatmap/Table/...)│
│ - runs       │ + expandable detail on KPI click                  │
│ - controls   │                                                   │
└─────────────┴───────────────────────────────────────────────────┘
│ Status bar (runs/visible, last poll, shortcuts)                  │
└─────────────────────────────────────────────────────────────────┘
```

## Implementation Plan (Slices)

### Slice 1: KPI Data Model & Computation (Low risk)
- Add `KpiValue { value: Option<f64>, delta: Option<f64>, trend: Trend, label: &'static str, unit: &'static str }`
- Helper methods on `RunModel`: `kpi_latest()`, `kpi_best()`, `kpi_delta()`, `sparkline_for(key, n)`
- Aggregate across visible runs or focus on "primary" run (newest visible)

### Slice 2: KPI Strip UI (Main screen)
- Add `kpi_strip(ui: &mut egui::Ui)` in `plot_panel`/top of central area
- Card per KPI: label, large value, small delta, mini sparkline (inline)
- Click to select/focus KPI for detail
- Tooltips with definitions

### Slice 3: KPI Detail Expansion
- Expandable panel below strip or overlay: full sparkline, min/max/mean/std, last N values, raw stats
- "View in chart" button jumps to main chart with that metric
- Compare across visible runs

### Slice 4: Smart Defaults & Thresholds
- Color thresholds (crash rate >0.05 warn, >0.1 bad)
- Auto-select primary run (newest live or first visible)
- Show "stale" badge on KPIs if run stale

### Slice 5: Polish & Tests
- Keyboard: `K` toggle KPI strip, number keys to focus KPIs
- Persist collapsed/expanded state
- Unit tests for KPI computation
- Integration test rendering KPI strip

## Files to Modify
- `rl_dash/src/main.rs` - add KPI types, computation, UI, detail panels
- `rl_dash/Cargo.toml` - no new deps needed (egui/egui_plot sufficient)
- Tests in `rl_dash/src/main.rs` - add KPI tests

## Verification
```bash
cargo test -p rl_dash
cargo build -p rl_dash --release
# Manual: run with real runs, check live updates, click KPIs, test staleness
```

## Risks & Mitigations
- **Clutter**: Keep strip collapsible, default visible but compact
- **Performance**: Compute KPIs from cached series, avoid recomputing heavy transforms
- **Ambiguity**: When multiple runs visible, show "best/avg/latest" or require selecting primary run
- **API stability**: KPIs read existing fields only

## Next Steps
1. Implement Slice 1 (data model) - pure functions, easy to test
2. Slice 2 (strip UI) - visible improvement
3. Slice 3 (detail) - drilldown
4. Iterate based on usage
