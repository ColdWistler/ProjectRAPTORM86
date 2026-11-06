//! `rl_agent` command-line entry point.
//!
//! The executable selects a tensor backend at runtime (`--backend
//! cpu|gpu|auto`) and either trains (`train`) or rolls out a checkpointed
//! policy under deterministic actions (`eval`). In `auto` mode the GPU is
//! probed with the real training workload under a wall-clock timeout and any
//! panic or stall falls back to the CPU backend.

use std::collections::HashMap;
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::time::Duration;

use burn::tensor::backend::AutodiffBackend;

use rl_agent::algo::{AlgoSpec, ALGO_PPO};
use rl_agent::backend::{cpu_device, probe_gpu_train, try_gpu_device, Cpu, Gpu};
use rl_agent::net::NetConfig;
use rl_agent::telemetry::TelemetryConfig;
use rl_agent::trainer::{
    evaluate, load_checkpoint, run_mission, run_with_observer, AvionicsEnv, TrainObserver,
    TrainerConfig,
};
use rl_agent::{ACTION_DIM, AVIONICS_OBS_DIM};
use rl_agent::mission_env::{MissionEnv, RewardProfile};
use rl_agent::policy::{HeuristicPolicy, Policy};
use flight_core::{MissionConfig, MissionEnvironment, MissionPhase};

const USAGE: &str = "\
rl_agent — burn/PPO trainer for the flight_core AvionicsEnvironment

USAGE:
  rl_agent [--mode train|eval] [--algo ppo|a2c|d3qn|sac|td3] [--backend cpu|gpu|auto] [options]

ALGORITHM (--algo, default ppo):
  The training algorithm is pluggable: `--algo` selects one of the
  registered algorithms in `rl_agent::algo::ALGORITHM_IDS` and the flags
  below configure it. Add a new model by implementing `Algorithm<B>` and
  registering it in `algo::make_algorithm` (see docs/rl_agent_guide.md).

  ppo   on-policy  PPO-Clip with advantage whitening (the reference model)
  a2c   on-policy  advantage actor-critic, one gradient pass per rollout
  sac   off-policy soft actor-critic (twin critics, entropy bonus)
  td3   off-policy twin delayed DDPG (delayed actor, target smoothing)
  d3qn  off-policy dueling critic, double-Q target, prioritized replay

TRAIN MODE (default):
  --iterations N        update iterations                        (default 300)
  --rollout N           env steps collected per iteration        (default 2048)
  --env-max-steps N     episode step budget                      (default 2000)
  --env-dt SECONDS      physics step size                        (default 0.0333)
  --hidden N            shared hidden layer width                (default 128)
  --log-std-init F      initial log_std for the policy           (default -0.5)
  --epochs N            PPO epochs per update (A2C: 1)           (default 10)
  --minibatch N         transitions per gradient step            (default 64)
  --lr F                AdamW learning rate                      (default 3e-4)
  --gamma F             discount factor                          (default 0.99)
  --lambda F            GAE trace decay                          (default 0.95)
  --clip F              PPO ratio clip bound                     (default 0.2)
  --value-coef F        value-loss coefficient                   (default 0.5)
  --entropy-coef F      entropy bonus coefficient                (default 0.0)
  --max-grad-norm F     global gradient-norm clip                (default 0.5)
  --polyak-tau F        target-network soft-update rate          (default 0.005)
  --alpha F             SAC entropy temperature                  (default 0.2)
  --policy-delay N      TD3 critic steps between actor updates   (default 2)
  --batch-size N        replay batch size                        (default 64)
  --buffer-capacity N   replay capacity                          (default 8192)
  --seed N              RNG seed (weights + sampling)            (default 42)
  --save-every N        checkpoint every N iterations            (default 100)
  --print-every N       log stats every N iterations             (default 1)
  --eval-every N        deterministic eval every N iterations    (default 100)
  --eval-episodes N     episodes per eval run                    (default 3)
  --config PATH         aircraft.toml / m_sweep_mini.toml (default avionics)
  --mission avionics|mission  training mode (default avionics)
  --specialization full|outbound|capture|return  phase focus (default full)
  --model ppo|heuristic   alias for --algo; heuristic is eval-only
  --reward-profile NAME|PATH  balanced|aggressive|cautious|capture_focus|file
  --checkpoint DIR      checkpoint directory                     (default checkpoints)

TELEMETRY (train mode):
  Every iteration is appended to runs/<run_id>/metrics.jsonl, each finished
  rollout episode to runs/<run_id>/episodes.jsonl, and the run manifest lives
  in runs/<run_id>/run.json (see docs/rl_agent_guide.md §12).

  --telemetry on|off    write structured telemetry               (default on)
  --run-dir DIR         root dir for per-run subdirectories      (default runs)
  --name LABEL          optional human label for this run        (default none)
                        (embedded in the run id, e.g. ..._ppo_seed42_smoke)

LIVE OBSERVATION (train mode):
  While training, the run directory is also a live observation surface. Frames
  are published through a bounded, lossy channel to a writer thread, so
  training throughput never waits on a consumer (see rl_agent::stream).

  --stream on|off        per-step flight frames for a live viewer
                                                    (default on, needs telemetry)
  --control on|off       accept pause/resume/save/eval/stop from a GUI
                                                      (default on, needs telemetry)

  Artifacts written into runs/<run_id>/:
    frames.bin           fixed-size ring of recent steps (~7.8 MiB, overwritten)
    episodes/ep_N.f32    one file per completed episode, for replay
    control/inbox/       drop cmd_<seq>.json here to steer the run
    control/outbox/      trainer writes ack_<seq>.json for each command

  Control commands are JSON with a seq number and an op of pause, resume,
  save_checkpoint, evaluate_now or stop. They are applied at iteration
  boundaries, never inside the step loop.

EVAL MODE:
  rl_agent --mode eval --checkpoint DIR [--backend cpu|gpu] [--eval-episodes N]

  Loads the policy from DIR, then runs `eval-episodes` deterministic
  episodes and reports altitude/airspeed error and crash rate.

EVAL-SUITE MODE:
  rl_agent --mode eval-suite [--checkpoint DIR] [--scenarios FILE] [--out DIR]
      runs the 30-scenario mission exam through any policy loaded from DIR
      and writes `eval.json` + `eval.csv` into the chosen output directory.
  The heuristic policy (`--model heuristic`) skips the checkpoint entirely
  and evaluates that way; learned policies need `--checkpoint`.

MISSION MODE:
  --mission avionics|mission    training target (default avionics)
  --model ppo|heuristic         alias for --algo; heuristic is eval-only
  --reward-profile NAME|PATH    balanced|aggressive|cautious|capture_focus|file

  Mission mode defaults to m_sweep_mini.toml and the 32-channel mission.v1
  contract. Avionics mode defaults to aircraft.toml and the 19-channel bus.

BACKENDS:
  --backend cpu   ndarray backend (no GPU required)
  --backend gpu   wgpu/Vulkan backend (errors if unavailable)
  --backend auto  probe GPU with the real training workload under a
                  wall-clock timeout; fall back to CPU on failure (default)

  --help  print this message
";

fn main() {
    // cubecl's data-stream-dispatch worker threads are spawned with the Rust
    // default stack size (2 MiB). Under the fusion-enabled CUDA/Vulkan runtimes
    // the kernel code-gen path recurses deeply enough to overflow that stack in
    // debug builds (an abort, so it cannot be caught by the `auto` GPU probe).
    // Raise the default worker-thread stack before any backend can spawn a
    // thread; 16 MiB is virtual memory and only committed as used.
    std::env::set_var("RUST_MIN_STACK", (16 << 20).to_string());

    let args = parse_args();
    match run_main(&args) {
        Ok(()) => {}
        Err(e) => {
            eprintln!("rl_agent: {e}");
            std::process::exit(1);
        }
    }
}

fn parse_args() -> HashMap<String, String> {
    let mut out = HashMap::new();
    let mut it = std::env::args().skip(1);
    while let Some(arg) = it.next() {
        if arg == "--help" || arg == "-h" {
            print!("{USAGE}");
            std::process::exit(0);
        }
        match arg.strip_prefix("--") {
            Some(key) => match it.next() {
                Some(value) => {
                    out.insert(key.to_string(), value);
                }
                None => {
                    eprintln!("rl_agent: missing value for --{key}");
                    std::process::exit(2);
                }
            },
            None => {
                eprintln!("rl_agent: unexpected argument `{arg}` (see --help)");
                std::process::exit(2);
            }
        }
    }
    out
}

/// Wall-clock budget for the functional GPU probe ([`probe_gpu_train`]). A
/// device that cannot complete ~16 autodiff updates plus 128 synced
/// per-step forwards in this window is treated as unusable and `auto` falls
/// back to CPU. Healthy GPU boxes finish it in well under a second.
const GPU_PROBE_TIMEOUT: Duration = Duration::from_secs(20);

fn run_main(args: &HashMap<String, String>) -> Result<(), Box<dyn std::error::Error>> {
    let backend = args.get("backend").map(String::as_str).unwrap_or("auto");

    match backend {
        "auto" => match try_gpu_device() {
            Some(device) => {
                // A micro matmul passing (`try_gpu_device`) is not enough:
                // wgpu/Vulkan on some machines survives that but stalls in
                // autodiff/fusion codegen or per-call dispatch, leaving a
                // `running` run that never writes telemetry. Probe with the
                // real training workload under a wall-clock budget and fall
                // back to CPU when it does not complete.
                let (tx, rx) = std::sync::mpsc::channel();
                let probe_device = device.clone();
                std::thread::spawn(move || {
                    let _ = tx.send(probe_gpu_train(probe_device));
                });
                let probe_started = std::time::Instant::now();
                match rx.recv_timeout(GPU_PROBE_TIMEOUT) {
                    Ok(Ok(())) => exec::<Gpu>(args, device),
                    Ok(Err(e)) => {
                        eprintln!(
                            "rl_agent: warning: GPU probe failed after {:.1}s ({e}) — using CPU",
                            probe_started.elapsed().as_secs_f32()
                        );
                        exec::<Cpu>(args, cpu_device())
                    }
                    Err(_) => {
                        eprintln!(
                            "rl_agent: warning: GPU probe timed out after {:.1}s — using CPU",
                            probe_started.elapsed().as_secs_f32()
                        );
                        exec::<Cpu>(args, cpu_device())
                    }
                }
            }
            None => {
                eprintln!("rl_agent: warning: no usable Vulkan GPU found — using CPU");
                exec::<Cpu>(args, cpu_device())
            }
        },
        "gpu" => match try_gpu_device() {
            Some(device) => match catch_unwind(AssertUnwindSafe(|| exec::<Gpu>(args, device))) {
                Ok(res) => res,
                Err(_) => Err("GPU execution panicked (device failure); use --backend cpu".into()),
            },
            None => Err("--backend gpu requested but no usable Vulkan GPU was found".into()),
        },
        _ => exec::<Cpu>(args, cpu_device()),
    }
}

fn exec<B: AutodiffBackend<FloatElem = f32>>(
    args: &HashMap<String, String>,
    device: B::Device,
) -> Result<(), Box<dyn std::error::Error>> {
    let tcfg = build_trainer_config(args)?;
    let ncfg = build_net_config(args)?;
    let config_path = config_path(args);
    let mode = args.get("mode").map(String::as_str).unwrap_or("train");
    let spec = if mode == "eval-suite" && model_id(args) == "heuristic" {
        AlgoSpec::from_id(ALGO_PPO)? // dummy, never used in heuristic branch
    } else {
        build_algo_config(args)?
    };

    match mode {
        "eval" => {
            let cp = load_checkpoint::<B>(&tcfg.checkpoint_dir, &device, tcfg.seed)?;
            let mut env = AvionicsEnv::new(&config_path, tcfg.env_dt, tcfg.env_max_steps)?;
            let m = evaluate(
                cp.algorithm.as_ref(),
                &mut env,
                &cp.normalizer,
                tcfg.eval_episodes,
                tcfg.env_max_steps,
            );
            println!(
                "[rl_agent] eval: {} episodes (algo {}), mean steps {:.0}, alt_err {:.1} m, \
                 spd_err {:.1} m/s, reward {:.1}, crashes {}/{}",
                m.episodes,
                cp.algorithm.id(),
                m.mean_steps,
                m.alt_err,
                m.spd_err,
                m.total_reward,
                m.crashes,
                m.episodes,
            );
            Ok(())
        }
        "train" => {
            let telemetry = build_telemetry_config(args)?;
            let mut observer = build_observer(args, &telemetry)?;
            let outcome = if is_mission_mode(args) {
                let mission_cfg = build_mission_config(args, tcfg.env_max_steps)?;
                let reward = build_reward_profile(args)?;
                let specialization = build_specialization(args)?;
                run_mission::<B>(
                    &tcfg,
                    &telemetry,
                    &ncfg,
                    &spec,
                    &device,
                    &config_path,
                    &mission_cfg,
                    &reward,
                    specialization,
                    &mut observer,
                )?
            } else {
                run_with_observer::<B>(
                    &tcfg,
                    &telemetry,
                    &ncfg,
                    &spec,
                    &device,
                    &config_path,
                    &mut observer,
                )?
            };
            // The stream handle must outlive the run: dropping it signals the
            // writer thread and joins it, flushing the ring and any in-flight
            // episode. Report what it achieved so a silent stream is visible.
            if let Some(stream) = observer.stream {
                let (frames, dropped, episodes) = stream.counters();
                println!(
                    "[rl_agent] stream      {frames} frames, {dropped} dropped, {episodes} episodes"
                );
            }
            println!(
                "[rl_agent] training complete (algo {}) on bootstrap from {}",
                outcome.algorithm.id(),
                std::any::type_name::<B>(),
            );
            let _ = outcome;
            Ok(())
        }
        "eval-suite" => {
            let mission_cfg = build_mission_config(args, tcfg.env_max_steps)?;
            let profile = build_reward_profile(args)?;
            let mut env = MissionEnvironment::new(&config_path, mission_cfg.clone(), tcfg.seed)?;
            let mut policy: Box<dyn Policy> = if model_id(args) == "heuristic" {
                let v_stall = flight_core::stall_speed(env.aircraft_config(), mission_cfg.cruise_altitude);
                Box::new(HeuristicPolicy::new(
                    mission_cfg.cruise_airspeed,
                    mission_cfg.resolve_capture_speed(env.aircraft_config()),
                    v_stall,
                ))
            } else {
                let cp = load_checkpoint::<B>(&tcfg.checkpoint_dir, &device, tcfg.seed)?;
                if cp.normalizer.dim != ncfg.obs_dim {
                    return Err(format!(
                        "checkpoint `{}` has {}-channel observations but --mission {} needs {} \
                         (train a mission checkpoint with --mode train --mission mission)",
                        tcfg.checkpoint_dir,
                        cp.normalizer.dim,
                        mission_mode(args),
                        ncfg.obs_dim
                    )
                    .into());
                }
                let card = rl_agent::policy::ExpertCard::from_checkpoint(&tcfg.checkpoint_dir, cp.algorithm.id());
                Box::new(rl_agent::policy::RlPolicy::new(cp.algorithm, cp.normalizer, card))
            };
            let scenarios = match args.get("scenarios") {
                Some(path) => rl_agent::eval::load_scenarios(path)?,
                None => rl_agent::eval::default_scenarios(),
            };
            let suite = rl_agent::eval::run_eval_suite(
                &mut *policy,
                &mut env,
                &scenarios,
                mission_cfg.max_steps,
                &profile,
            );
            let out = args.get("out").cloned().unwrap_or_else(|| tcfg.checkpoint_dir.clone());
            let dir = std::path::Path::new(&out);
            std::fs::create_dir_all(dir)?;
            // Set expert / contract correctly on returned suite
            let mut final_suite = suite;
            final_suite.expert = policy.card().id.clone();
            final_suite.contract = rl_agent::contracts::MISSION_CONTRACT.into();
            rl_agent::eval::write_eval_json(&dir.join("eval.json"), &final_suite)?;
            rl_agent::eval::write_eval_csv(&dir.join("eval.csv"), &final_suite)?;
            println!(
                "[rl_agent] eval-suite: {} scenarios, success {:.1}% [{:.1}, {:.1}], capture {:.1}%, crash {:.1}%",
                final_suite.summary.n,
                final_suite.summary.success_rate * 100.0,
                final_suite.summary.success_ci_low * 100.0,
                final_suite.summary.success_ci_high * 100.0,
                final_suite.summary.capture_rate * 100.0,
                final_suite.summary.crash_rate * 100.0,
            );
            Ok(())
        }
        other => Err(format!("unknown --mode `{other}` (expected train, eval, or eval-suite)").into()),
    }
}

fn get_num<T>(args: &HashMap<String, String>, key: &str, default: T) -> Result<T, Box<dyn std::error::Error>>
where
    T: std::str::FromStr,
    <T as std::str::FromStr>::Err: std::fmt::Display,
{
    match args.get(key) {
        Some(v) => v
            .parse::<T>()
            .map_err(|e| format!("bad --{key} `{v}`: {e}").into()),
        None => Ok(default),
    }
}

fn build_trainer_config(args: &HashMap<String, String>) -> Result<TrainerConfig, Box<dyn std::error::Error>> {
    let mut cfg = TrainerConfig::default();
    cfg.iterations = get_num(args, "iterations", cfg.iterations)?;
    cfg.rollout_steps = get_num(args, "rollout", cfg.rollout_steps)?;
    cfg.env_max_steps = get_num(args, "env-max-steps", cfg.env_max_steps)?;
    cfg.env_dt = get_num(args, "env-dt", cfg.env_dt)?;
    cfg.seed = get_num(args, "seed", cfg.seed)?;
    cfg.save_every = get_num(args, "save-every", cfg.save_every)?;
    cfg.print_every = get_num(args, "print-every", cfg.print_every)?;
    cfg.eval_every = get_num(args, "eval-every", cfg.eval_every)?;
    cfg.eval_episodes = get_num(args, "eval-episodes", cfg.eval_episodes)?;
    if let Some(dir) = args.get("checkpoint") {
        cfg.checkpoint_dir = dir.clone();
    }
    Ok(cfg)
}

/// Build the live-observation observer (frame streaming + control channel).
///
/// The run directory is *not* resolved here: the trainer creates it when it
/// opens telemetry, and only then is the path known for certain. So this only
/// parses the two feature flags, and [`TrainObserver::bind`] does the wiring
/// once the trainer can supply the real directory.
fn build_observer(
    args: &HashMap<String, String>,
    telemetry: &TelemetryConfig,
) -> Result<TrainObserver, Box<dyn std::error::Error>> {
    if telemetry.run_root.is_empty() {
        // Telemetry off means no run dir exists to observe.
        return Ok(TrainObserver::disabled());
    }
    let flag = |key: &str| -> Result<bool, Box<dyn std::error::Error>> {
        match args.get(key).map(String::as_str) {
            None | Some("on") => Ok(true),
            Some("off") => Ok(false),
            Some(other) => Err(format!("--{key} must be `on` or `off`, got `{other}`").into()),
        }
    };
    Ok(TrainObserver::pending(flag("stream")?, flag("control")?))
}

fn build_telemetry_config(args: &HashMap<String, String>) -> Result<TelemetryConfig, Box<dyn std::error::Error>> {
    let enabled = args
        .get("telemetry")
        .map(String::as_str)
        .unwrap_or("on");
    if enabled == "off" {
        return Ok(TelemetryConfig::disabled());
    }
    if enabled != "on" {
        return Err(format!("--telemetry must be `on` or `off`, got `{enabled}`").into());
    }
    let run_root = args
        .get("run-dir")
        .cloned()
        .unwrap_or_else(|| "runs".to_string());
    let name = args.get("name").cloned();
    Ok(TelemetryConfig { run_root, name })
}

fn build_net_config(args: &HashMap<String, String>) -> Result<NetConfig, Box<dyn std::error::Error>> {
    Ok(NetConfig {
        obs_dim: if is_mission_mode(args) {
            rl_agent::contracts::MISSION_OBS_DIM
        } else {
            AVIONICS_OBS_DIM
        },
        action_dim: ACTION_DIM,
        hidden: get_num(args, "hidden", 128)?,
        log_std_init: get_num(args, "log-std-init", -0.5f32)?,
    })
}

fn mission_mode(args: &HashMap<String, String>) -> &str {
    match args.get("mission").map(String::as_str) {
        Some("mission") => "mission",
        Some("avionics") => "avionics",
        Some(other) => {
            eprintln!("rl_agent: --mission must be `avionics` or `mission`, got `{other}`");
            std::process::exit(2);
        }
        None => "avionics",
    }
}

fn is_mission_mode(args: &HashMap<String, String>) -> bool {
    mission_mode(args) == "mission"
}

fn config_path(args: &HashMap<String, String>) -> String {
    if let Some(p) = args.get("config") {
        return p.clone();
    }
    if is_mission_mode(args) {
        "m_sweep_mini.toml".into()
    } else {
        "aircraft.toml".into()
    }
}

fn model_id(args: &HashMap<String, String>) -> &str {
    args.get("model").map(String::as_str)
        .or_else(|| args.get("algo").map(String::as_str))
        .unwrap_or("ppo")
}

fn build_algo_config(args: &HashMap<String, String>) -> Result<AlgoSpec, Box<dyn std::error::Error>> {
    let id = model_id(args);
    let mut spec = AlgoSpec::from_id(id)?;
    match &mut spec {
        AlgoSpec::Ppo { config } => {
            config.lr = get_num(args, "lr", config.lr)?;
            config.gamma = get_num(args, "gamma", config.gamma)?;
            config.lambda = get_num(args, "lambda", config.lambda)?;
            config.clip_epsilon = get_num(args, "clip", config.clip_epsilon)?;
            config.value_coef = get_num(args, "value-coef", config.value_coef)?;
            config.entropy_coef = get_num(args, "entropy-coef", config.entropy_coef)?;
            config.max_grad_norm = get_num(args, "max-grad-norm", config.max_grad_norm)?;
            config.epochs = get_num(args, "epochs", config.epochs)?;
            config.minibatch_size = get_num(args, "minibatch", config.minibatch_size)?;
        }
        AlgoSpec::A2c { config } => {
            config.lr = get_num(args, "lr", config.lr)?;
            config.gamma = get_num(args, "gamma", config.gamma)?;
            config.value_coef = get_num(args, "value-coef", config.value_coef)?;
            config.entropy_coef = get_num(args, "entropy-coef", config.entropy_coef)?;
            config.max_grad_norm = get_num(args, "max-grad-norm", config.max_grad_norm)?;
            config.epochs = get_num(args, "epochs", config.epochs)?;
            config.minibatch_size = get_num(args, "minibatch", config.minibatch_size)?;
        }
        AlgoSpec::D3qn { config } => {
            config.lr = get_num(args, "lr", config.lr)?;
            config.gamma = get_num(args, "gamma", config.gamma)?;
            config.tau = get_num(args, "polyak-tau", config.tau)?;
            config.hidden = get_num(args, "hidden", config.hidden)?;
            config.batch_size = get_num(args, "batch-size", config.batch_size)?;
            config.buffer_capacity = get_num(args, "buffer-capacity", config.buffer_capacity)?;
        }
        AlgoSpec::Sac { config } => {
            config.lr = get_num(args, "lr", config.lr)?;
            config.gamma = get_num(args, "gamma", config.gamma)?;
            config.tau = get_num(args, "polyak-tau", config.tau)?;
            config.hidden = get_num(args, "hidden", config.hidden)?;
            config.batch_size = get_num(args, "batch-size", config.batch_size)?;
            config.buffer_capacity = get_num(args, "buffer-capacity", config.buffer_capacity)?;
            config.alpha = get_num(args, "alpha", config.alpha)?;
        }
        AlgoSpec::Td3 { config } => {
            config.lr = get_num(args, "lr", config.lr)?;
            config.gamma = get_num(args, "gamma", config.gamma)?;
            config.tau = get_num(args, "polyak-tau", config.tau)?;
            config.hidden = get_num(args, "hidden", config.hidden)?;
            config.batch_size = get_num(args, "batch-size", config.batch_size)?;
            config.buffer_capacity = get_num(args, "buffer-capacity", config.buffer_capacity)?;
            config.policy_delay = get_num(args, "policy-delay", config.policy_delay)?;
        }
    }
    Ok(spec)
}

fn build_specialization(
    args: &HashMap<String, String>,
) -> Result<Option<&str>, Box<dyn std::error::Error>> {
    match args.get("specialization").map(String::as_str) {
        None => Ok(None),
        Some(value @ ("full" | "outbound" | "capture" | "return")) => Ok(Some(value)),
        Some(other) => Err(format!(
            "--specialization must be `full`, `outbound`, `capture`, or `return`, got `{other}`"
        )
        .into()),
    }
}

fn build_reward_profile(args: &HashMap<String, String>) -> Result<rl_agent::mission_env::RewardProfile, Box<dyn std::error::Error>> {
    let name = args.get("reward-profile").map(String::as_str).unwrap_or("balanced");
    if name.contains("/") || name.ends_with(".toml") {
        Ok(rl_agent::mission_env::RewardProfile::from_toml_file(name)?)
    } else {
        rl_agent::mission_env::RewardProfile::from_name(name).map_err(|e| format!("{}", e).into())
    }
}

fn build_mission_config(args: &HashMap<String, String>, default_max_steps: usize) -> Result<flight_core::MissionConfig, Box<dyn std::error::Error>> {
    let mut cfg = flight_core::MissionConfig::default();
    if let Some(v) = args.get("env-max-steps") {
        cfg.max_steps = v.parse()?;
    } else {
        cfg.max_steps = default_max_steps;
    }
    Ok(cfg)
}
