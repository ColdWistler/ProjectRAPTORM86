//! `rl_agent` command-line entry point.
//!
//! The executable selects a tensor backend at runtime (`--backend
//! cpu|gpu|auto`) and either trains (`train`) or rolls out a checkpointed
//! policy under deterministic actions (`eval`). In `auto` mode the GPU is
//! probed with a tiny tensor op and any panic during GPU execution falls
//! back to the CPU backend.

use std::collections::HashMap;
use std::panic::{catch_unwind, AssertUnwindSafe};

use burn::tensor::backend::AutodiffBackend;

use rl_agent::algo::{AlgoSpec, ALGO_PPO};
use rl_agent::backend::{cpu_device, try_gpu_device, Cpu, Gpu};
use rl_agent::net::NetConfig;
use rl_agent::trainer::{
    evaluate, load_checkpoint, run, AvionicsEnv, TrainerConfig,
};
use rl_agent::{ACTION_DIM, AVIONICS_OBS_DIM};

const USAGE: &str = "\
rl_agent — burn/PPO trainer for the flight_core AvionicsEnvironment

USAGE:
  rl_agent [--mode train|eval] [--algo ppo] [--backend cpu|gpu|auto] [options]

ALGORITHM (--algo, default ppo):
  The training algorithm is pluggable: `--algo` selects one of the
  registered algorithms in `rl_agent::algo::ALGORITHM_IDS` and the flags
  below configure it. Add a new model by implementing `Algorithm<B>` and
  registering it in `algo::make_algorithm` (see docs/rl_agent_guide.md).

TRAIN MODE (default):
  --iterations N        PPO update iterations                    (default 1000)
  --rollout N           env steps collected per iteration        (default 2048)
  --env-max-steps N     episode step budget                      (default 2000)
  --env-dt SECONDS      physics step size                        (default 0.0333)
  --hidden N            shared hidden layer width                (default 128)
  --log-std-init F      initial log_std for the policy           (default -0.5)
  --epochs N            PPO epochs per update                    (default 10)
  --minibatch N         transitions per gradient step            (default 64)
  --lr F                AdamW learning rate                      (default 3e-4)
  --gamma F             discount factor                          (default 0.99)
  --lambda F            GAE trace decay                          (default 0.95)
  --clip F              PPO ratio clip bound                     (default 0.2)
  --value-coef F        value-loss coefficient                   (default 0.5)
  --entropy-coef F      entropy bonus coefficient                (default 0.0)
  --max-grad-norm F     global gradient-norm clip                (default 0.5)
  --seed N              RNG seed (weights + sampling)            (default 42)
  --save-every N        checkpoint every N iterations            (default 100)
  --print-every N       log stats every N iterations             (default 1)
  --eval-every N        deterministic eval every N iterations    (default 100)
  --eval-episodes N     episodes per eval run                    (default 3)
  --config PATH         aircraft.toml for flight_core            (default aircraft.toml)
  --checkpoint DIR      checkpoint directory                     (default checkpoints)

EVAL MODE:
  rl_agent --mode eval --checkpoint DIR [--backend cpu|gpu] [--eval-episodes N]

  Loads the policy from DIR, then runs `eval-episodes` deterministic
  episodes and reports altitude/airspeed error and crash rate.

BACKENDS:
  --backend cpu   ndarray backend (no GPU required)
  --backend gpu   wgpu/Vulkan backend (errors if unavailable)
  --backend auto  probe GPU, fall back to CPU on failure (default)

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

fn run_main(args: &HashMap<String, String>) -> Result<(), Box<dyn std::error::Error>> {
    let backend = args.get("backend").map(String::as_str).unwrap_or("auto");

    match backend {
        "auto" => match try_gpu_device() {
            Some(device) => match catch_unwind(AssertUnwindSafe(|| exec::<Gpu>(args, device))) {
                Ok(res) => res,
                Err(_) => {
                    eprintln!("rl_agent: warning: GPU execution panicked — falling back to CPU");
                    exec::<Cpu>(args, cpu_device())
                }
            },
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
    let spec = build_algo_config(args)?;
    let config_path = args
        .get("config")
        .cloned()
        .unwrap_or_else(|| "aircraft.toml".to_string());
    let mode = args.get("mode").map(String::as_str).unwrap_or("train");

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
            let outcome = run::<B>(&tcfg, &ncfg, &spec, &device, &config_path)?;
            println!(
                "[rl_agent] training complete (algo {}) on bootstrap from {}",
                outcome.algorithm.id(),
                std::any::type_name::<B>(),
            );
            let _ = outcome;
            Ok(())
        }
        other => Err(format!("unknown --mode `{other}` (expected train or eval)").into()),
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

fn build_net_config(args: &HashMap<String, String>) -> Result<NetConfig, Box<dyn std::error::Error>> {
    Ok(NetConfig {
        obs_dim: AVIONICS_OBS_DIM,
        action_dim: ACTION_DIM,
        hidden: get_num(args, "hidden", 128)?,
        log_std_init: get_num(args, "log-std-init", -0.5f32)?,
    })
}

fn build_algo_config(args: &HashMap<String, String>) -> Result<AlgoSpec, Box<dyn std::error::Error>> {
    let id = args.get("algo").map(String::as_str).unwrap_or(ALGO_PPO);
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
    }
    Ok(spec)
}