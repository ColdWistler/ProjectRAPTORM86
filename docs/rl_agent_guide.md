# `rl_agent` Guide

`rl_agent` is a **pure-Rust, GPU-optimized PPO trainer** for the
`flight_core` flight-dynamics engine. It trains the **avionics
attitude-command environment** (`AvionicsEnvironment`): the agent emits
roll/pitch setpoints, a yaw-rate command, and a throttle command, and a
simulated PID flight controller inside `flight_core` turns those into
servo/ESC outputs. The agent sees the **noisy 19-channel sensor bus**
(gyro/accel/GPS/baro/airspeed/battery).

Everything is built on [burn](https://burn.dev) v0.21.0 — tensors and
autodiff — with both a CPU backend (`ndarray`) and a GPU backend
(`wgpu`/Vulkan) compiled into the same binary and selected at **runtime**.
There is no compile-time feature gating: `--backend auto` probes the GPU
and transparently falls back to CPU.

---

## 1. Quick start

```bash
# Train from the repo root (aircraft.toml is found automatically):
cargo run -p rl_agent --release -- --mode train --backend auto

# Or a quick CPU smoke run:
cargo run -p rl_agent -- --mode train --backend cpu --iterations 3 --rollout 128

# Evaluate a checkpoint with deterministic actions:
cargo run -p rl_agent --release -- --mode eval --backend auto --checkpoint checkpoints
```

`--help` prints the full CLI. There is also a **GUI launcher**: the
"RL" tab of the Godot main menu (`godot/scripts/rl_trainer.gd`) lets you pick
algorithm, backend and hyper-parameters and runs training with a live log
(see §6).

## 2. CLI reference (`cargo run -p rl_agent -- --help`)

| Flag | Default | Meaning |
|---|---|---|
| `--mode train\|eval\|eval-suite` | `train` | Train, evaluate one checkpoint, or score a policy over the mission scenarios |
| `--algo NAME` | `ppo` | Training algorithm: `ppo`, `a2c`, `d3qn`, `sac`, `td3`; `hybrid` is eval-suite only (see §5) |
| `--backend cpu\|gpu\|auto` | `auto` | Tensor backend at runtime (see §3) |
| `--iterations N` | `300` | Iterations (one rollout + update each) |
| `--rollout N` | `2048` | Env steps collected per iteration |
| `--env-max-steps N` | `2000` | Per-episode step budget (truncation) |
| `--env-dt SECONDS` | `0.0333` | Physics step size (1/30 s) |
| `--hidden N` | `128` | Width of both shared hidden layers |
| `--log-std-init F` | `-0.5` | Initial per-channel policy `log_std` |
| `--epochs N` | `10` | PPO epochs per update (A2C: `1`) |
| `--minibatch N` | `64` | Transitions per gradient step |
| `--lr F` | `3e-4` | AdamW learning rate |
| `--gamma F` | `0.99` | Discount factor |
| `--lambda F` | `0.95` | GAE trace-decay |
| `--clip F` | `0.2` | PPO ratio clip bound ε |
| `--value-coef F` | `0.5` | Value-loss coefficient |
| `--entropy-coef F` | `0.0` | Entropy bonus coefficient |
| `--max-grad-norm F` | `0.5` | Global gradient-norm clip |
| `--polyak-tau F` | `0.005` | Target-network soft-update rate (`sac`, `td3`, `d3qn`) |
| `--alpha F` | `0.2` | SAC entropy temperature |
| `--policy-delay N` | `2` | TD3 critic steps between actor updates |
| `--batch-size N` | `64` | Replay batch size (`sac`, `td3`, `d3qn`) |
| `--seed N` | `42` | Weights init + host sampler RNG |
| `--save-every N` | `100` | Checkpoint every N iterations (0 = never) |
| `--print-every N` | `1` | Log per-iteration stats every N (0 = silent) |
| `--eval-every N` | `100` | Deterministic eval every N iterations (0 = never) |
| `--eval-episodes N` | `3` | Episodes per eval run |
| `--config PATH` | `aircraft.toml` | Aircraft config for `flight_core` |
| `--checkpoint DIR` | `checkpoints` | Checkpoint directory |
| `--telemetry on\|off` | `on` | Write structured telemetry (see §12) |
| `--run-dir DIR` | `runs` | Root dir for per-run telemetry subdirectories |
| `--name LABEL` | none | Optional human label appended to the run id |
| `--hysteresis N` | `3` | Hybrid only: steps a phase must hold before the selector switches expert |
| `--learned-gate on\|off` | `off` | Hybrid only: use the trained phase-gating MLP instead of the phase→expert map |
| `--expert-map FILE` | none | Hybrid only: JSON object mapping `outbound\|capture\|return\|done` to checkpoint dirs |
| `--scenarios FILE` | built-in 30 | Eval-suite scenario file (JSON); defaults to the mission exam |
| `--out DIR` | checkpoint dir | Eval-suite output directory for `eval.json` / `eval.csv` |

## 3. Backends (`--backend cpu | gpu | auto`)

| Value | Behavior |
|---|---|
| `cpu` | `Autodiff<NdArray>` on `NdArrayDevice::Cpu`. No GPU required; fast to start and fully deterministic with a seed. |
| `gpu` | `Autodiff<Wgpu>` (Vulkan compute) on the first device that survives the functional probe. **Errors** if no working GPU is found or the GPU panics mid-run. |
| `auto` (default) | Selects the GPU only if it is genuinely usable for *this* workload, else falls back to CPU with a one-line warning. Selection: (1) pick the discrete GPU first (`DiscreteGpu(0)`, then `IntegratedGpu(0)`, then wgpu's `DefaultDevice` — override with `RL_GPU_DEVICE=discrete\|integrated\|default`); (2) a tiny matmul under `catch_unwind`; (3) `probe_gpu_train` — 16 real autodiff updates + 128 rollout-style per-step forwards that sync to the host every step, all under a 20 s wall-clock timeout *and* a 0.5 ms/synced-forward budget. A device that stalls (or panics — e.g. the old `into_scalar` bug) or whose per-call dispatch overhead exceeds the budget is rejected: on the reference laptop wgpu measures ≥1 ms/step and trains at ~48 sps vs ~2 000 sps on CPU. If GPU execution panics later, the **entire run** restarts on CPU. |

Both backends are compiled into one binary — switching is a flag, not a
rebuild. The only place concrete backend types are named is
`rl_agent/src/backend.rs`; every training routine is generic over
`B: AutodiffBackend`.

> **GPU worker-thread stack.** cubecl's data-stream-dispatch worker threads
> use the Rust default 2 MiB stack, which the fusion kernel code-gen path can
> overflow (an *abort*, so `catch_unwind` can't rescue it) — most visible in
> debug builds. `main.rs` therefore raises `RUST_MIN_STACK` to 16 MiB before
> any backend can spawn a thread. If you embed the crate as a library and hit
> `stack overflow` in a `DSD-*` thread, set the same env var before first use.

## 4. Training target

`flight_core::AvionicsEnvironment` with `EnvConfig { dt, max_steps, .. }`
(the file from `--config` supplies the rest). Observation layout
(19 channels, noisy):

```text
 0  gyro_p (rad/s)         10 baro_altitude (m)
 1  gyro_q (rad/s)         11 airspeed_indicated (m/s)
 2  gyro_r (rad/s)         12 battery_voltage (V)
 3  accel_x (m/s²)         13 battery_capacity_remaining_pct (%)
 4  accel_y (m/s²)         14 actual_elevator_deg (°)
 5  accel_z (m/s²)         15 actual_aileron_deg (°)
 6  gps_north (m, NED)     16 actual_rudder_deg (°)
 7  gps_east (m, NED)      17 actual_esc_output (0..1)
 8  gps_altitude (m)       18 sim_time (s)
 9  gps_fix_quality (0..3)
```

Actions are 4-dim **tanh-squashed** and mapped to attitude commands:

```rust
roll_cmd    = tanh(a0) * 0.6        // rad   (±45° clamp: 0.785 rad)
pitch_cmd   = tanh(a1) * 0.4        // rad
yaw_rate_cmd= tanh(a2) * 0.8        // rad/s
throttle    = 0.5 * (tanh(a3) + 1)  // 0..1
```

All components stay inside the flight controller's `±45°` setpoint clamps.
The service attempts attitude control toward a **target altitude**
(1000 m) and **target airspeed** (60 m/s) drawn from `aircraft.toml` /
`EnvConfig`; evaluation metrics compare the noisy `baro_altitude` and
`airspeed_indicated` channels (indices 10/11) against those targets.

Episodes end by **termination** (crash — the env's `crash_penalty()` is
added to the reward) or **truncation** (step budget exhausted).

## 5. The pluggable algorithm layer

The trainer is **algorithm-agnostic**. Everything a learning model needs from
the shared loop lives behind one trait, and *which* model runs is chosen at
runtime — on the CLI (`--algo`) or from the Godot menu dropdown.

| Piece | File | Notes |
|---|---|---|
| `Algorithm<B>` trait | `src/algo.rs` | `id`, `net_config`, `spec`, `td_params`, `act`/`deterministic_action`, `value_single`/`value_all`, `update`, `save_weights`/`load_weights`. The shared loop consumes only these. |
| Off-policy hooks | `src/algo.rs` | `is_off_policy`, `observe_transition`, `gradient_steps_per_env_step`, `set_normalizer` — see §5.1. |
| `AlgoSpec` enum | `src/algo.rs` | Serializable `(id, config)` selection; persisted into every checkpoint as `algo.json`. Internally `serde`-tagged so files are versioned by algorithm id. |
| Registry | `src/algo.rs::make_algorithm` | `Box<dyn Algorithm<B>>` factory — one match arm per algorithm. |
| `UpdateBatch` / `UpdateStats` | `src/algo.rs` | Algo-agnostic transition batch and averaged loss stats exchanged with `update`. On-policy algos consume the GAE fields; off-policy algos ignore them. |
| Trainer | `src/trainer.rs` | Generic `run<B>(&TrainerConfig, &NetConfig, &AlgoSpec, &device, &config_path)` — builds via the registry, rolls out, GAE, update, eval, checkpoint. |

The shared loop computes GAE with the discount/trace-decay from
`Algorithm::td_params()` (PPO returns its γ/λ); a value-free algorithm
returns `(0.99, 0.95)` and can override the bootstrap behavior in its own
`update`.

### 5.1 On-policy vs off-policy

Spec §10 lets each model's loop differ. The trait carries four optional
hooks so the shared trainer can drive both families:

| Hook | Default | Meaning |
|---|---|---|
| `is_off_policy()` | `false` | The model learns from its **own** replay buffer, not from the rollout batch. |
| `observe_transition(&Transition)` | no-op | Receives one raw (`obs`, `action`, `reward`, `terminated`, `truncated`, `next_obs`) transition per env step. |
| `gradient_steps_per_env_step()` | `0.0` | `0.0` = one update per rollout (on-policy). `1.0` = one gradient step per collected transition (off-policy). |
| `set_normalizer(RunningNormalizer)` | no-op | Hands the model the trainer's observation statistics. |

Why the off-policy models need them: `UpdateBatch` carries observations,
actions, log-probs, advantages and returns — **no reward, no next
observation and no done flag**. A Bellman target cannot be built from it.
So SAC / TD3 / D3QN keep their own replay buffer, receive the raw
transitions through `observe_transition`, and run their gradient steps
*inside* the rollout loop (one per env step, with a fractional `grad_debt`
so a ratio like `0.5` works). The trainer skips the host-side GAE for them.

**The normalizer is frozen for the whole rollout.** The statistics are
snapshotted before the rollout and refreshed from the rollout's observations
*after* the update:

- for PPO/A2C, the policy that sampled the data and the policy that trains
  on it must see identically scaled observations, otherwise the importance
  ratio is not `1.0` at the first epoch;
- for the off-policy models, `obs` and `next_obs` of one transition must be
  scaled with the *same* statistics, otherwise the Bellman target is
  inconsistent.

Off-policy replay buffers therefore store **raw** observations and scale the
sampled batch at update time with the injected snapshot.

### Adding a new model

1. Write a module, e.g. `rl_agent/src/dqn.rs`, owning the agent struct, its
   config (`Serialize`+`Deserialize`), its `update` math and its weight
   save/load bookkeeping.
2. `impl Algorithm<B> for YourAgent<B>` (with `B: AutodiffBackend<FloatElem =
   f32>`), implementing `save_weights`/`load_weights` via
   `BinFileRecorder` like `PpoAgent` does.
3. Add an `AlgoSpec::YourAlgo { config }` variant and one arm in
   `make_algorithm`.
4. Register the id in the CLI (`main.rs`: `AlgoSpec::from_id`) and in the menu
   (`godot/scripts/rl_trainer.gd`: the `ALGORITHMS` + `ALGO_FIELDS` tables).

That's the whole seam — no changes to the trainer loop are required on-policy
models that produce `(action, log_prob)` pairs and a value.

### Selecting an algorithm

```bash
cargo run -p rl_agent --release -- --algo ppo --backend auto
```

`--algo` accepts any id in `rl_agent::algo::ALGORITHM_IDS` (currently `ppo`,
`a2c`, `d3qn`, `sac`, `td3`). The algorithm's own flags (`--lr`, `--gamma`,
`--epochs`, `--polyak-tau`, ...) configure the selected model;
`AlgoSpec::from_id` rejects unknown ids with the registered list.

`hybrid` is a **policy composition**, not a gradient algorithm: it is
deliberately absent from `ALGORITHM_IDS` and only works with
`--mode eval-suite`. That path builds a `PhaseSelector` over the heuristic and
any experts named by `--expert-map`; `--hysteresis N` (default 3) controls how
long a phase must hold before switching, and `--learned-gate on|off` (default
`off`) replaces the phase→expert map with the trained gating MLP.
`--mode train --algo hybrid` exits 2 with a pointer to `--expert-map`.

```bash
# Score the heuristic-backed selector (no experts) over the 30-scenario exam.
cargo run -p rl_agent --release -- --mode eval-suite --algo hybrid \
  --mission mission --backend cpu --telemetry off --out /tmp/hybrid

# Bind trained per-phase experts.
cat > /tmp/experts.json <<'JSON'
{"outbound": "runs/outbound_expert", "capture": "runs/capture_expert"}
JSON
cargo run -p rl_agent --release -- --mode eval-suite --algo hybrid \
  --mission mission --expert-map /tmp/experts.json --out /tmp/hybrid
```

### The five models

| Id | Module | Family | What it does |
|---|---|---|---|
| `ppo` | `src/ppo.rs` | on-policy | PPO-Clip with per-minibatch advantage whitening. The reference model. |
| `a2c` | `src/a2c.rs` | on-policy | Advantage actor-critic, **one** gradient pass per rollout, no clipping. |
| `sac` | `src/sac.rs` | off-policy | Soft actor-critic: tanh-Gaussian actor, twin Q-critics, entropy bonus, Polyak targets. |
| `td3` | `src/td3.rs` | off-policy | TD3: deterministic tanh actor, twin Q-critics, delayed actor updates, target-policy smoothing. |
| `d3qn` | `src/d3qn.rs` | off-policy | Dueling critic + separate policy head, double-Q target, prioritized replay. |

Shared invariants (each has a behavioural test):

- the target networks start as an **exact copy** of the online networks and
  are then **Polyak-averaged** (`τ`), never hard-synced and never
  re-randomized;
- the actor's action stays attached to the autodiff graph — a host
  round-trip would sever the gradient and freeze the actor forever;
- a **truncation** (step budget) still bootstraps; only a **termination**
  stops the bootstrap;
- the learnable per-channel `log_std` is clamped to `[-5, 1]`, so an entropy
  bonus cannot turn the policy into pure noise;
- burn's `Linear` weights initialize **lazily** from a process-wide RNG, so
  every model seeds the backend and materializes its parameters under one
  lock (`net::init_lock` / `net::materialize`) before any clone. Cloning an
  unmaterialized parameter yields an *independently initialized* copy — that
  is how a target network silently becomes a different network.

## 6. The Godot menu

The main menu has an **RL** tab (`godot/scripts/rl_trainer.gd`, instantiated
by `main_menu.gd`). It:

- lists the registered algorithms (dropdown driven by `ALGORITHMS`) and the
  per-algorithm hyper-parameters (`ALGO_FIELDS` — the fields change when you
  switch algorithm);
- picks `auto` / `cpu` / `gpu`;
- launches the `rl_agent` binary as a subprocess (`sh -c 'exec …'` so the
  Stop button kills the actual trainer), streaming its stdout into a live,
  scrollable log **and** charting the run's `metrics.jsonl` in the **Live
  metrics** card (see §12);
- locates the binary automatically (`target/release/rl_agent`, then
  `target/debug/rl_agent` — build with `cargo build -r -p rl_agent` if the
  panel reports it missing);
- can run `--mode eval` against the latest checkpoint; Checkpoint/aircraft
  paths are derived from the repo (`res://../checkpoints`,
  `res://../aircraft.toml`).

`godot/tests/validate_rl_menu_e2e.gd` is a headless regression check that
instantiates the panel, dials the hyper-parameters down, presses Start, waits
for the subprocess, and asserts a successful exit + log output.

`godot/tests/validate_rl_layout.gd` is a second headless regression check for
the tab itself: it loads the real menu scene at the project window size,
switches to the RL tab, and asserts the page's `ScrollContainer` fills the
page rect (the panel root is a plain `Control`, so the scroll must use
full-rect anchors or the tab renders blank).

## 7. PPO algorithm and components

Training implements PPO-Clip (Schulman et al., 2017) with per-minibatch
advantage whitening:

```text
L = −E[ min(r·Â, clip(r, 1±ε)·Â) ] + c_v·E[(V−R)²] − c_e·E[H]
r = exp(log π_θ(a|s) − log π_old(a|s))
```

| Piece | File | Notes |
|---|---|---|
| Network | `src/net.rs` | 2-hidden-layer tanh MLP (`fc1`/`fc2`) → `mu_head` + `value_head`, plus a learnable per-channel `log_std` `Param`. Diagonal Gaussian, tanh-squashed. |
| Sampling | `src/ppo.rs` | Action sampled on the **host** with a seeded `StdRng` from the device-computed mean, so rollouts are reproducible. The tensor `log_prob` and the host `gaussian_log_prob_tanh` implement the **identical formula** (checked to 1e-5 in tests). |
| Buffer | `src/buffer.rs` | Flat `f32` arrays: raw obs, actions, log-probs, rewards, `terminated`/`truncated` flags. |
| GAE | `src/gae.rs` | Host-side generalized advantage estimation; `terminated` zeroes the next value **and** masks the accumulator. |
| Normalizer | `src/normalize.rs` | Welford running mean/variance; **frozen for the whole rollout** and refreshed from the rollout's observations after the update; clipped to ±5. See §5.1. |
| Optimizer | `src/ppo.rs` | AdamW with `GradientClippingConfig::Norm(max_grad_norm)` baked into the optimizer config. |
| Trainer | `src/trainer.rs` | Generic `run<B>` loop: rollout → values → GAE → update → eval/checkpoint. |

**Rollout flow** (per iteration):

1. The normalizer's statistics are snapshotted (`frozen`) and handed to the
   model (`set_normalizer`).
2. For `rollout_steps` transitions: normalize the current obs with the
   **frozen** stats, `agent.act()` produces `(action, log_prob)`, the env
   steps, reward += `crash_penalty()` on termination, the transition is
   pushed (with the **pre-reset** observation as `final_obs`), and on
   `terminated`/`truncated` the env resets. Off-policy models also receive
   the transition through `observe_transition` and run their own gradient
   steps here.
3. On-policy only: values are predicted over the whole normalized batch; the
   last value is `0` if the final rollout step terminated, else V(final obs).
4. On-policy only: GAE advantages + returns; one `agent.update` over
   shuffled minibatches.
5. The rollout's observations are folded into the running statistics (every
   step, not just the episode resets).
6. Periodic deterministic eval and checkpoint save.

## 8. Determinism

- Same `--seed` ⇒ same backend init (`B::seed`) and same host sampler
  stream ⇒ byte-identical normalizer stats and identical trained policies
  (asserted in `same_seed_reproduces_identical_training`).
- The sensor bus is deterministic for a fixed seed (`xorshift64*` with a
  fixed default), so two fresh environments reproduce identically — which is
  what makes the reproducibility test meaningful.
- `cpu` and `gpu` won't give *bit-identical* numbers to each other (different
  floating-point kernels), but each is reproducible within its backend.

## 9. Checkpoints

Saved to `--checkpoint DIR` every `save_every` iterations (and at the end):

| File | Contents |
|---|---|
| `net` (bin) | Policy + critic weights, `BinFileRecorder<FullPrecisionSettings>` (FP32, portable across backends). |
| `net.json` | `NetConfig` (obs/hidden/action dims, log_std init). |
| `algo.json` | `AlgoSpec` — the algorithm id + its hyper-parameters (e.g. `{"id":"ppo","config":{…}}`). **This is what makes any registered algorithm loadable by its own config.** |
| `trainer.json` | `TrainerConfig`. |
| `normalizer.json` | Running Welford stats + clip bound. |

Legacy v0.1 checkpoints (pre-algorithm-layer) stored `ppo.json` instead of
`algo.json`; `load_checkpoint` still reads those and upgrades them to an
`AlgoSpec::Ppo`. Covered by `loads_legacy_ppo_checkpoint_without_algo_json`.

Optimizer state is **not** persisted in v1 — on eval / resumption from a
checkpoint a fresh AdamW optimizer is built from the algorithm config.
Configure `save_every: 0` / `eval_every: 0` / `print_every: 0` to disable
those `.json` write paths entirely in tests/harnesses.

Eval mode loads the checkpoint and runs `--eval-episodes` deterministic
episodes, reporting mean steps, altitude error, airspeed error, mean reward,
and crash count. Training itself always restarts from a fresh bootstrap.

## 10. Using the crate programmatically

```rust
use rl_agent::algo::{Algorithm, AlgoSpec};
use rl_agent::backend::{cpu_device, Cpu};
use rl_agent::net::NetConfig;
use rl_agent::ppo::{PpoAgent, PpoConfig};
use rl_agent::telemetry::TelemetryConfig;
use rl_agent::trainer::{evaluate, load_checkpoint, run, TrainerConfig};

// Train — the spec selects the algorithm + its hyper-parameters:
let device = cpu_device();
let spec = AlgoSpec::Ppo { config: PpoConfig::default() };
let telemetry = TelemetryConfig::disabled(); // or TelemetryConfig::default() for runs/
let outcome = run::<Cpu>(
    &TrainerConfig::default(),
    &telemetry,
    &NetConfig::default(),
    &spec,
    &device,
    "aircraft.toml",
)?;
// outcome.algorithm     -> Box<dyn Algorithm<Cpu>> (trained)
// outcome.normalizer    -> RunningNormalizer (final stats)
// outcome.last_eval     -> Option<EvalMetrics>

// Algorithm-agnostic evaluation of a checkpoint:
let cp = load_checkpoint::<Cpu>("checkpoints", &device, 42)?;
let action = cp.algorithm.deterministic_action(&obs); // obs must be normalized

// Or skip the trainer and build a PPO agent directly:
let ncfg = NetConfig::default();
let agent = PpoAgent::<Cpu>::new(&device, &ncfg, &PpoConfig::default(), 42);
let (action, log_prob) = agent.act(&obs_norm);
let value = agent.value_single(&obs_norm);
```

## 11. Tests

```bash
# From the repo root (tests locate ../aircraft.toml):
cargo test -p rl_agent
```

The suite covers the network (host/tensor log-prob parity to 1e-5, the
`log_std` clamp), GAE math (terminated bootstrapping), the normalizer, the
rollout buffer, PPO / A2C updates, the algorithm layer (spec serde
round-trip, registry factory + `act`), and end-to-end trainer smoke tests —
including a real-environment two-iteration run, same-seed reproducibility,
FC-limit action mapping, new and legacy checkpoint round-trips to a temp
directory.

It also contains the **behavioural acceptance tests** for the off-policy
models (`rl_agent/src/{sac,td3,d3qn}.rs`) and the end-to-end CLI checks in
`rl_agent/tests/algo_training_cli.rs`: an algorithm whose weights never move
is not learning, so each test asserts that the actor/critic actually trains,
that the target networks are Polyak-averaged rather than hard-synced, that
the TD target is non-zero, and that a real training run reports non-zero
metrics.

---

## 12. Telemetry, log organization, and visualization

Training tends to produce a wall of per-iteration console lines. To keep the
console readable *and* the data useful, every run writes **structured
telemetry** alongside the console output. Turn it off with `--telemetry off`
(tests and library callers use `TelemetryConfig::disabled()`).

### Run layout

With default flags a training run creates one self-contained directory per
run, keyed by a UTC timestamp so runs sort chronologically and never collide:

```text
runs/
└── 20260923_143512_123_ppo_seed42_smoke/
    ├── run.json        # manifest: status, timestamps, git commit, argv, all configs
    ├── metrics.jsonl   # one JSON object per PPO iteration (flush per write)
    └── episodes.jsonl  # one JSON object per finished rollout episode
```

- `--name LABEL` appends a slugified label to the run id
  (`20260923_143512_123_ppo_seed42_smoke`); `--run-dir DIR` relocates the
  root.
- `run.json` starts as `"status": "running"` and is rewritten to
  `completed` on a normal finish (best-effort `interrupted` from `Drop` if
  the process dies), so it doubles as a liveness/heartbeat record.
- `metrics.jsonl` is append-only and flushed after **every iteration**, so
  live consumers can tail it while training runs.

### Metrics stream

Each line of `metrics.jsonl`:

```json
{"iteration":12,"elapsed_s":34.2,"steps_per_s":1536.4,
 "policy_loss":-0.02,"value_loss":0.51,"entropy":1.41,
 "advantage_mean":0.12,"advantage_std":0.94,
 "mean_reward_per_step":0.03,"mean_episode_steps":482.0,
 "crash_rate":0.25,"episodes_completed":4,
 "eval":{"alt_err":88.2,"spd_err":9.1,"total_reward":12.4,
         "crashes":0,"episodes":3,"mean_steps":1888.0},
 "env":{"samples":2048,"alt_mean":1200.5,"alt_min":1100.0,
        "alt_max":1300.0,"airspeed_mean":60.2,"climb_mean":0.4,
        "throttle_mean":0.55,"battery_pct_min":91.0},
 "checkpoint":true}
```

The rollout-derived fields (`mean_reward_per_step`, `mean_episode_steps`,
`crash_rate`, `episodes_completed`, `steps_per_s`) are new with this
stream — they were previously only visible as a console line. Flatten it to a
table with any JSONL tool, or press **Download CSV** in the dashboard.

### Environment telemetry

The `env` object (abridged in the example above) summarises the
**ground-truth** flight state over that rollout: the observation vector only
carries the noisy sensor bus the agent is allowed to see, so this is how you
find out what the aircraft actually did. It is written whenever telemetry is
on and the environment can report a `FlightFrame` — including with
`--stream off` — because it is training data, not a live-view side effect.

| Field | Meaning |
| --- | --- |
| `samples` | Rollout steps accumulated (0 means "no data") |
| `alt_mean/min/max` | True altitude (m) |
| `airspeed_mean/min/max` | True airspeed (m/s) |
| `climb_mean/min/max` | Climb rate (m/s, positive up) |
| `ground_speed_mean` | Ground-track speed (m/s) |
| `roll_abs_mean`, `pitch_abs_mean` | Mean absolute attitude (rad) |
| `aoa_mean`, `sideslip_abs_mean` | Angle of attack / sideslip (rad) |
| `wind_mean`, `wind_max` | Wind magnitude (m/s) |
| `throttle_mean` | Commanded throttle `[0, 1]` |
| `surface_effort_mean` | Mean absolute elevator/aileron/rudder deflection (deg) |
| `baro_err_abs_mean`, `airspeed_err_abs_mean` | Perceived vs true sensor error |
| `battery_pct_min` | Lowest battery charge seen (%) |
| `weather` | Optional block, only when the env has a weather system |

The optional `weather` block carries `wind_speed_mean`, `gust_rms_mean/max`,
`turbulence_mean`, `severity_mean`, `rain_mean`, `visibility_min`,
`updraft_abs_mean` and `shear_abs_mean`.

`episodes.jsonl` gets one row per episode that **ends during the rollout**
(eval episodes are deliberately excluded):

```json
{"episode":7,"iteration":12,"steps":482,"duration_s":24.1,"reward_sum":42.5,
 "terminated":false,"truncated":true,"termination":"step_budget",
 "env":{"samples":482,"alt_mean":1198.0,"...":"..."}}
```

`termination` is `ground_impact`, `gust_limit`, `visibility_limit`,
`severe_exposure` or `step_budget`. Episode length and reward total span
iteration boundaries, so a row's `steps` is the true episode length even when
it finished partway through an iteration.

Both dashboards expose the series as `env_*` / `weather_*` keys (for example
`env_alt_mean`, `env_throttle_mean`, `weather_gust_rms_max`), so altitude,
airspeed and control effort can be plotted against the training reward.

### Live chart (Godot RL tab)

The main-menu RL tab now shows a **Live metrics** card that streams the run's
`metrics.jsonl` while the subprocess trains: pick a series
(Reward/step, Policy/Value loss, Entropy, Advantage mean, the eval metrics, or
the `env_*` flight-state / `weather_*` series), hover for values, and read the
latest value in the chart title. The
raw console log stays below it (Clear log resets the chart for the next run).
Only the resolved run dir is read — the chart survives after the process
exits so you can inspect the finished run in-engine.

The same card also hosts the **Open desktop dashboard (rl_dash)** button,
which launches the native desktop app pointed at the *same* runs dir the
trainer writes to. Tick **Open on Start** and it pops up automatically
whenever training starts — so you get the in-engine chart *and* the
multi-run desktop dashboard for the same run.

### HTML dashboard (`rl_agent/dashboard.html`)

For post-run analysis open `rl_agent/dashboard.html` in any browser (no
dependencies, works offline from `file://`) and load a
`runs/<run_id>/metrics.jsonl` (drag & drop or picker):

- summary cards per metric (latest / range / best-in-class by direction),
- an interactive multi-series chart with smoothing + log-Y,
- a raw-rows table and a **Download CSV** export.

> The dashboard is a static file with no build step; keep it in sync if you
> add fields to `IterationRecord` in `rl_agent/src/telemetry.rs`.

### Native desktop app (`rl_dash`)

For a desktop monitor that keeps running while you work and can compare
*runs*, there is a pure-Rust native app (`egui`/`eframe` — no browser, no
system webview, no new native dependencies):

```bash
cargo run -p rl_dash --release          # watches ./runs
cargo run -p rl_dash -- --runs DIR      # explicit telemetry root
cargo run -p rl_dash -- --metric policy_loss
```

- Picks up runs automatically (new run dirs = new rows); **Follow live**
  keeps the newest run plotted and re-reads `metrics.jsonl` by byte offset,
  so a running trainer updates ~5×/s with no re-parsing of old lines. The
  runs dir is also re-scanned every second, so a run the trainer spawns
  *after* the dashboard opens (e.g. from the RL tab) appears on its own.
- **Chart view** — pick any metric (reward, policy/value loss, entropy,
  advantage mean/std, env steps/s throughput, crash rate, episode length,
  cumulative episodes, eval errors), overlay any run against the same axes,
  hover for cursor values, and read the latest values beneath the plot.
  - **X-axis** switches between iteration, cumulative **env steps**
    (sample-efficiency comparison), and wall clock.
  - **Best-so-far** draws each run's running max envelope (running min for
    loss metrics); **Log y** rescales for losses spanning decades.
  - **Evaluations / Checkpoints** draw vertical guides at eval/checkpoint
    iterations, plus eval-reward dots overlaid on the reward curve.
  - **Seed bands** groups runs that share a manifest config (seed excluded)
    and plots their mean ± std as filled bands across replications.
- **Heatmap view** — a run × iteration color matrix of any metric; each row
  normalizes over its own range so per-run trends stay readable.
- **Run summary** — a collapsible leaderboard per visible run: status, iters,
  wall time, cumulative env steps, average SPS, final/best reward, crash
  rate, backend.
- Smoothing (EMA window) applies to the chart.
- Launch it from the Godot RL tab's **Open desktop dashboard** button (or
  `--runs DIR` on the CLI for the Godot-spawned trainer's run dir); with the
  RL tab's **Open on Start** checkbox it opens automatically whenever
  training starts — tracking exactly the run the in-engine chart shows.
- A collapsible panel shows each run's `run.json` manifest (status, commit,
  algo, start/end times).
- Uses `egui_plot`; the JSONL schema is mirrored in
  `rl_dash/src/main.rs` — keep it in sync with `rl_agent/src/telemetry.rs`
  when adding fields.

Why a native app instead of another HTML page? The browser dashboard is
great for one run at a time; `rl_dash` stays open across whole training
sessions, tails files without a reload, and overlays many runs — including
the run you start next.

---

### Reference

- Schulman et al., *Proximal Policy Optimization Algorithms*, 2017
  (the clipped surrogate objective).
- Schulman et al., *High-Dimensional Continuous Control Using Generalized
  Advantage Estimation*, 2015 (GAE-λ).
- burn v0.21.0 docs: `burn::module`, `burn::optim`, `burn::record`,
  `burn::backend::{ndarray, wgpu}`.
- `flight_core::AvionicsEnvironment` / `EnvConfig` / `AVIONICS_OBS_DIM`
  (`flight_core/src/env.rs`).