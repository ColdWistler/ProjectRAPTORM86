//! Training loop and environment plumbing.
//!
//! [`run`] is the generic (backend-agnostic) training loop over an
//! [`EnvSpec`] (by default the [`AvionicsEnv`] wrapper around
//! [`AvionicsEnvironment`]):
//!
//! 1. **Rollout**: collect `rollout_steps` transitions, observing the
//!    normalizer during resets, sampling stochastic actions host-side.
//! 2. **Values**: predict V(s) for the whole normalized batch and for the
//!    trailing observation, then compute GAE advantages host-side.
//! 3. **Update**: one [`Algorithm::update`] pass over shuffled minibatches.
//! 4. Periodically evaluate and save a checkpoint
//!    (weights + JSON configs; optimizer state is not persisted in v1).

use crate::algo::{make_algorithm, Algorithm, AlgoSpec, UpdateBatch};
use crate::buffer::{RolloutBuffer, Transition};
use crate::gae::compute_gae;
use crate::net::NetConfig;
use crate::normalize::RunningNormalizer;
use crate::ppo::PpoConfig;
use crate::telemetry::{
    EpisodeRecord, EnvSummary, IterationRecord, RunMeta, TelemetryConfig, TelemetryWriter,
    WeatherSummary,
};
use crate::{ACTION_DIM, AVIONICS_OBS_DIM};
use crate::contracts::action_to_avionics;
use burn::tensor::backend::AutodiffBackend;
use flight_core::{
    AvionicsAction, AvionicsEnvironment, EnvConfig, FlightFrame, TerminationReason,
    WeatherObservation,
};
use serde::{Deserialize, Serialize};
use std::path::Path;

/// Outcome of a single environment step (generic over the concrete env).
#[derive(Debug, Clone)]
pub struct StepOutcome {
    /// Next observation.
    pub obs: Vec<f64>,
    /// Scalar reward (before any crash penalty).
    pub reward: f64,
    /// Episode ended by termination (crash).
    pub terminated: bool,
    /// Episode ended by exhausting the step budget.
    pub truncated: bool,
}

/// Minimal environment interface the trainer needs.
pub trait EnvSpec {
    /// Observation dimension.
    fn obs_dim(&self) -> usize;
    /// Action dimension.
    fn action_dim(&self) -> usize;
    /// Reset to a fresh episode; returns the initial observation.
    fn reset(&mut self) -> Vec<f64>;
    /// Apply a continuous action (tanh-space), advancing the sim one `dt`.
    fn step(&mut self, action: &[f32]) -> StepOutcome;
    /// Penalty added when a step ends in termination (≤ 0).
    fn crash_penalty(&self) -> f64;
    /// Target cruise altitude (m) — used by the evaluator.
    fn target_altitude(&self) -> f64;
    /// Target cruise airspeed (m/s) — used by the evaluator.
    fn target_airspeed(&self) -> f64;

    /// Ground-truth flight state for visualising the current step.
    ///
    /// Defaults to `None` because the observation vector is all an RL
    /// algorithm is strictly entitled to: it carries the noisy sensor bus and
    /// nothing else, so an env that exposes only `obs` (a bandit, a toy grid,
    /// a wrapped third-party gym) needs no work to satisfy this trait.
    ///
    /// Environments that *can* report state — anything wrapping a physics
    /// simulator — should override it so a live viewer can draw what is
    /// happening. See [`flight_core::FlightFrame`].
    fn frame(&self) -> Option<flight_core::FlightFrame> {
        None
    }

    /// Terrain elevation treated as ground impact (m).
    ///
    /// Only used to name an episode's termination cause in telemetry; the env
    /// itself owns the actual crash test.
    fn ground_altitude(&self) -> f64 {
        0.0
    }

    /// Structured weather observation for the current step, if the env has a
    /// weather system.
    ///
    /// Defaults to `None`, matching [`Self::frame`]: a generic env has no
    /// weather, and the trainer simply omits the weather block.
    fn weather(&self) -> Option<WeatherObservation> {
        None
    }

    /// Active weather termination reason, if the weather system ended the
    /// episode. [`TerminationReason::None`] means benign weather.
    fn weather_termination(&self) -> Option<TerminationReason> {
        None
    }

    fn mission_context(&self) -> Option<flight_core::MissionContext> {
        None
    }
}

/// The default training target: `flight_core::AvionicsEnvironment`.
pub struct AvionicsEnv {
    inner: AvionicsEnvironment,
}

impl AvionicsEnv {
    /// Load the aircraft config and build an avionics environment with the
    /// given physics step and episode budget.
    pub fn new(config_path: &str, dt: f64, max_steps: usize) -> Result<Self, Box<dyn std::error::Error>> {
        let env = AvionicsEnvironment::with_config(
            config_path,
            EnvConfig {
                dt,
                max_steps,
                ..Default::default()
            },
        )?;
        Ok(Self { inner: env })
    }
}

impl EnvSpec for AvionicsEnv {
    fn obs_dim(&self) -> usize {
        AVIONICS_OBS_DIM
    }

    fn action_dim(&self) -> usize {
        ACTION_DIM
    }

    fn reset(&mut self) -> Vec<f64> {
        self.inner.reset().0.to_vec()
    }

    fn step(&mut self, action: &[f32]) -> StepOutcome {
        let out = self.inner.step(action_to_avionics(action));
        StepOutcome {
            obs: out.observation.to_vec(),
            reward: out.reward,
            terminated: out.terminated,
            truncated: out.truncated,
        }
    }

    fn crash_penalty(&self) -> f64 {
        self.inner.crash_penalty()
    }

    fn target_altitude(&self) -> f64 {
        self.inner.config.target_altitude
    }

    fn target_airspeed(&self) -> f64 {
        self.inner.config.target_airspeed
    }

    fn frame(&self) -> Option<flight_core::FlightFrame> {
        Some(self.inner.frame())
    }

    fn ground_altitude(&self) -> f64 {
        self.inner.config.ground_altitude
    }

    fn weather(&self) -> Option<WeatherObservation> {
        self.inner.weather.as_ref().map(|wx| wx.observation())
    }

    fn weather_termination(&self) -> Option<TerminationReason> {
        self.inner.weather.as_ref().map(|wx| wx.termination())
    }
}

/// Trainer-level hyper-parameters.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainerConfig {
    /// Number of PPO iterations (rollout + update).
    pub iterations: usize,
    /// Steps collected per iteration (episodes continue across the boundary).
    pub rollout_steps: usize,
    /// Per-episode step budget.
    pub env_max_steps: usize,
    /// Physics time step per env step (seconds).
    pub env_dt: f64,
    /// Seed for the agent RNG + deterministic net init.
    pub seed: u64,
    /// Save a checkpoint every N iterations (0 = never).
    pub save_every: usize,
    /// Print per-iteration stats every N iterations (0 = silent).
    pub print_every: usize,
    /// Run this many deterministic eval episodes every `eval_every`.
    pub eval_episodes: usize,
    /// Evaluate every N iterations (0 = never).
    pub eval_every: usize,
    /// Directory for checkpoints.
    pub checkpoint_dir: String,
}

impl Default for TrainerConfig {
    fn default() -> Self {
        Self {
            iterations: 1000,
            rollout_steps: 2048,
            env_max_steps: 2000,
            env_dt: 1.0 / 30.0,
            seed: 42,
            save_every: 100,
            print_every: 1,
            eval_episodes: 3,
            eval_every: 100,
            checkpoint_dir: "checkpoints".to_string(),
        }
    }
}

/// Aggregated evaluation metrics.
#[derive(Debug, Clone, Copy, Default, Serialize)]
pub struct EvalMetrics {
    /// Mean per-step |baro_altitude − target| over all eval steps.
    pub alt_err: f32,
    /// Mean per-step |airspeed_indicated − target| over all eval steps.
    pub spd_err: f32,
    /// Mean per-episode cumulative reward (crash penalties included).
    pub total_reward: f32,
    /// Number of episodes that ended by crashing.
    pub crashes: usize,
    /// Number of episodes evaluated.
    pub episodes: usize,
    /// Mean steps per episode.
    pub mean_steps: f32,
}

/// Aggregated per-rollout statistics (rewards, episodes, throughput).
#[derive(Debug, Clone, Copy, Default)]
pub struct RolloutStats {
    /// Total reward over the rollout (crash penalties included).
    pub reward_sum: f64,
    /// Episodes finished this rollout (crash or step-budget).
    pub completed_episodes: usize,
    /// Of the finished episodes, how many ended by crashing.
    pub crash_count: usize,
    /// Sum of steps across finished episodes, for the mean.
    pub episode_len_sum: usize,
}

impl RolloutStats {
    /// Mean per-step reward, crash penalties included.
    pub fn mean_reward_per_step(&self, steps: usize) -> f32 {
        (self.reward_sum / steps.max(1) as f64) as f32
    }

    /// Mean steps per finished episode (episodes may span iteration
    /// boundaries; the length is the whole episode, not the tail in this
    /// rollout).
    pub fn mean_episode_steps(&self) -> f32 {
        (self.episode_len_sum as f64 / self.completed_episodes.max(1) as f64) as f32
    }

    /// Crash rate over finished episodes (0..=1). `0` when nothing finished.
    pub fn crash_rate(&self) -> f32 {
        self.crash_count as f32 / self.completed_episodes.max(1) as f32
    }
}

/// Running count/sum/min/max for one scalar channel.
///
/// Non-finite samples are ignored rather than poisoning the aggregate, so one
/// bad physics step cannot turn a whole iteration's telemetry into `NaN`.
#[derive(Debug, Clone, Copy, Default)]
struct Series {
    count: u64,
    sum: f64,
    min: f64,
    max: f64,
}

impl Series {
    fn observe(&mut self, v: f64) {
        if !v.is_finite() {
            return;
        }
        if self.count == 0 {
            self.min = v;
            self.max = v;
        } else {
            self.min = self.min.min(v);
            self.max = self.max.max(v);
        }
        self.count += 1;
        self.sum += v;
    }

    fn mean(&self) -> f64 {
        if self.count == 0 {
            0.0
        } else {
            self.sum / self.count as f64
        }
    }

    fn min(&self) -> f64 {
        if self.count == 0 { 0.0 } else { self.min }
    }

    fn max(&self) -> f64 {
        if self.count == 0 { 0.0 } else { self.max }
    }
}

/// Accumulates the weather half of the environment telemetry.
#[derive(Debug, Clone, Copy, Default)]
struct WeatherStats {
    samples: u64,
    wind_speed: Series,
    gust_rms: Series,
    turbulence: Series,
    severity: Series,
    rain: Series,
    visibility: Series,
    updraft_abs: Series,
    shear_abs: Series,
}

impl WeatherStats {
    fn observe(&mut self, wx: &WeatherObservation) {
        self.samples += 1;
        self.wind_speed.observe(wx.wind_speed);
        self.gust_rms.observe(wx.gust_rms);
        self.turbulence.observe(wx.turbulence_level as f64);
        self.severity.observe(wx.severity.index() as f64);
        self.rain.observe(wx.rain_intensity);
        self.visibility.observe(wx.visibility);
        self.updraft_abs.observe(wx.updraft_strength.abs());
        self.shear_abs.observe(wx.shear_gradient.abs());
    }

    fn summary(&self) -> Option<WeatherSummary> {
        (self.samples > 0).then(|| WeatherSummary {
            samples: self.samples,
            wind_speed_mean: self.wind_speed.mean(),
            gust_rms_mean: self.gust_rms.mean(),
            gust_rms_max: self.gust_rms.max(),
            turbulence_mean: self.turbulence.mean(),
            severity_mean: self.severity.mean(),
            rain_mean: self.rain.mean(),
            visibility_min: self.visibility.min(),
            updraft_abs_mean: self.updraft_abs.mean(),
            shear_abs_mean: self.shear_abs.mean(),
        })
    }
}

/// Accumulates the flight half of the environment telemetry.
///
/// One instance spans a rollout (for the `env` block on an iteration record);
/// a second spans an episode (for `episodes.jsonl`). Episodes can straddle
/// iteration boundaries, so the episode accumulator lives outside the
/// per-iteration loop.
#[derive(Debug, Clone, Copy, Default)]
struct EnvStats {
    samples: u64,
    alt: Series,
    airspeed: Series,
    climb: Series,
    ground_speed: Series,
    roll_abs: Series,
    pitch_abs: Series,
    aoa: Series,
    sideslip_abs: Series,
    wind: Series,
    throttle: Series,
    surface_effort: Series,
    baro_err_abs: Series,
    airspeed_err_abs: Series,
    battery_pct: Series,
    weather: WeatherStats,
}

impl EnvStats {
    fn observe(&mut self, frame: &FlightFrame, weather: Option<&WeatherObservation>) {
        self.samples += 1;
        self.alt.observe(frame.altitude);
        self.airspeed.observe(frame.airspeed);
        self.climb.observe(frame.climb_rate());
        self.ground_speed.observe(frame.ground_speed());
        self.roll_abs.observe(frame.roll.abs());
        self.pitch_abs.observe(frame.pitch.abs());
        self.aoa.observe(frame.alpha);
        self.sideslip_abs.observe(frame.beta.abs());
        self.wind.observe(
            (frame.wind_north * frame.wind_north
                + frame.wind_east * frame.wind_east
                + frame.wind_down * frame.wind_down)
                .sqrt(),
        );
        self.throttle.observe(frame.cmd_throttle);
        self.surface_effort.observe(
            (frame.elevator_deg.abs() + frame.aileron_deg.abs() + frame.rudder_deg.abs()) / 3.0,
        );
        self.baro_err_abs.observe(frame.baro_altitude_error().abs());
        self.airspeed_err_abs.observe(frame.airspeed_error().abs());
        self.battery_pct.observe(frame.battery_pct);
        if let Some(wx) = weather {
            self.weather.observe(wx);
        }
    }

    fn summary(&self) -> Option<EnvSummary> {
        (self.samples > 0).then(|| EnvSummary {
            samples: self.samples,
            alt_mean: self.alt.mean(),
            alt_min: self.alt.min(),
            alt_max: self.alt.max(),
            airspeed_mean: self.airspeed.mean(),
            airspeed_min: self.airspeed.min(),
            airspeed_max: self.airspeed.max(),
            climb_mean: self.climb.mean(),
            climb_min: self.climb.min(),
            climb_max: self.climb.max(),
            ground_speed_mean: self.ground_speed.mean(),
            roll_abs_mean: self.roll_abs.mean(),
            pitch_abs_mean: self.pitch_abs.mean(),
            aoa_mean: self.aoa.mean(),
            sideslip_abs_mean: self.sideslip_abs.mean(),
            wind_mean: self.wind.mean(),
            wind_max: self.wind.max(),
            throttle_mean: self.throttle.mean(),
            surface_effort_mean: self.surface_effort.mean(),
            baro_err_abs_mean: self.baro_err_abs.mean(),
            airspeed_err_abs_mean: self.airspeed_err_abs.mean(),
            battery_pct_min: self.battery_pct.min(),
            weather: self.weather.summary(),
        })
    }
}

/// Human-readable cause for a finished episode, used in `episodes.jsonl`.
///
/// Weather wins over ground impact when both are true on the same step: the
/// extreme-weather limit is the more specific explanation, and the simulator
/// treats it as the terminating condition.
fn episode_termination(env: &dyn EnvSpec, terminated: bool, truncated: bool) -> &'static str {
    if truncated {
        return "step_budget";
    }
    if terminated {
        if let Some(reason) = env.weather_termination() {
            if reason != TerminationReason::None {
                return reason.name();
            }
        }
        return "ground_impact";
    }
    "unknown"
}

/// What `run` hands back for tests / the CLI.
pub struct TrainingOutcome<B: AutodiffBackend<FloatElem = f32>> {
    /// Trained algorithm (final params after all iterations).
    pub algorithm: Box<dyn Algorithm<B>>,
    /// Normalizer with the final running stats.
    pub normalizer: RunningNormalizer,
    /// Last completed eval, if any eval episode ran.
    pub last_eval: Option<EvalMetrics>,
    /// Directory this run's telemetry and stream artifacts live in.
    ///
    /// `None` when telemetry is disabled. Exposed because the run id embeds a
    /// millisecond timestamp and is therefore only knowable from inside the
    /// run — a caller that wants to watch or steer the run has to be told
    /// where it went.
    pub run_dir: Option<std::path::PathBuf>,
}

/// Optional live-observation hooks for a training run.
///
/// A single struct rather than four separate parameters, because they all
/// describe the same concern — "let something outside the training loop watch
/// and influence the run" — and a caller enabling any of them almost always
/// wants all of them.
///
/// Everything here is optional and defaults to off, so existing callers
/// (`main`, tests) keep their current behaviour without edits. The guarantee
/// the trainer makes is narrow and deliberate:
///
/// - [`TrainObserver::on_step`] is called inside the rollout's innermost loop.
///   It must be non-blocking; the default implementation does nothing.
/// - [`TrainObserver::on_iteration`] is called once per iteration, at a clean
///   boundary after the update, where state is consistent.
pub struct TrainObserver {
    /// Where per-step flight frames are published.
    pub frames: crate::stream::FrameSink,
    /// GUI→trainer command channel, polled once per iteration.
    pub control: crate::stream::ControlChannel,
    /// Writer thread handle. `None` when streaming is off or not yet bound.
    ///
    /// Field order matters for the drop sequence: `stream` is declared last so
    /// it is dropped *first*, and its `Drop` sets the stop flag and joins the
    /// writer. The sink's channel then disconnects and the writer drains. Both
    /// orders work, but this one means the writer is already gone by the time
    /// the sink drops, so no frame can be lost in the gap.
    pub stream: Option<crate::stream::FrameStream>,
    /// Feature flags retained until [`Self::bind`] supplies the run directory.
    ///
    /// The run directory is only known once the trainer has created it, so
    /// observation is configured in two steps: parse the flags (`pending`),
    /// then attach the real paths (`bind`). This avoids predicting the run id —
    /// which embeds a millisecond timestamp — and guessing wrong.
    wants_frames: bool,
    wants_control: bool,
}

impl TrainObserver {
    /// No streaming, no control. Equivalent to [`Default`].
    pub fn disabled() -> Self {
        Self {
            frames: crate::stream::FrameSink::disabled(),
            control: crate::stream::ControlChannel::disabled(),
            stream: None,
            wants_frames: false,
            wants_control: false,
        }
    }

    /// An observer with feature flags set but no run directory yet.
    pub fn pending(stream_frames: bool, accept_control: bool) -> Self {
        Self {
            wants_frames: stream_frames,
            wants_control: accept_control,
            ..Self::disabled()
        }
    }

    /// Attach the run directory, starting the writer thread if frames are on.
    ///
    /// Called by the trainer immediately after telemetry opens the run, since
    /// that is the first moment the directory exists. Binding twice is a no-op,
    /// so an already-live observer is safe to pass in.
    pub fn bind(&mut self, run_dir: &std::path::Path) -> std::io::Result<()> {
        if self.frames.is_enabled() || run_dir.as_os_str().is_empty() {
            return Ok(());
        }
        if self.wants_frames {
            // Degrade to no streaming rather than failing the run: observation
            // must never be the reason training cannot start.
            if let Some((sink, stream)) = crate::stream::FrameStream::start(run_dir) {
                self.frames = sink;
                self.stream = Some(stream);
            }
        }
        if self.wants_control {
            self.control = crate::stream::ControlChannel::new(run_dir);
        }
        Ok(())
    }
}

impl Default for TrainObserver {
    fn default() -> Self {
        Self::disabled()
    }
}

/// Run `tcfg.iterations` training iterations on the avionics environment.
///
/// The algorithm is built from `spec` through the [`make_algorithm`]
/// registry, so the loop itself stays algorithm-agnostic. When `telemetry`
/// is enabled, every iteration is appended (flushed) to the run's
/// `metrics.jsonl` and the run manifest (`run.json`) is finished on exit.
#[allow(clippy::too_many_arguments)]
pub fn run<B: AutodiffBackend<FloatElem = f32>>(
    tcfg: &TrainerConfig,
    telemetry: &TelemetryConfig,
    ncfg: &NetConfig,
    spec: &AlgoSpec,
    device: &B::Device,
    config_path: &str,
) -> Result<TrainingOutcome<B>, Box<dyn std::error::Error>> {
    run_with_observer(
        tcfg,
        telemetry,
        ncfg,
        spec,
        device,
        config_path,
        &mut TrainObserver::disabled(),
    )
}

/// [`run`], with live frame streaming and GUI control enabled.
///
/// The two differ only in what they pass down; the training loop itself is
/// shared so a streamed run trains identically to an unstreamed one.
#[allow(clippy::too_many_arguments)]
pub fn run_with_observer<B: AutodiffBackend<FloatElem = f32>>(
    tcfg: &TrainerConfig,
    telemetry: &TelemetryConfig,
    ncfg: &NetConfig,
    spec: &AlgoSpec,
    device: &B::Device,
    config_path: &str,
    obs: &mut TrainObserver,
) -> Result<TrainingOutcome<B>, Box<dyn std::error::Error>> {
    assert!(tcfg.rollout_steps > 0, "rollout_steps must be > 0");

    let mut env = AvionicsEnv::new(config_path, tcfg.env_dt, tcfg.env_max_steps)?;
    let mut agent: Box<dyn Algorithm<B>> = make_algorithm(spec, ncfg, tcfg.seed, device);
    let mut normalizer = RunningNormalizer::new(ncfg.obs_dim);
    let (gamma, lambda) = agent.td_params();

    let meta = RunMeta {
        algo: agent.id(),
        backend: std::any::type_name::<B::Device>(),
        tcfg,
        ncfg,
        spec,
    };
    let mut tele = TelemetryWriter::start(telemetry, &meta)?;
    if let Some(w) = tele.as_ref() {
        println!("[rl_agent] telemetry  run: {}", w.run_id());
    }
    // The run directory now exists, so live observation can be attached. Doing
    // it here rather than before the run means the stream always lands in the
    // same directory the manifest describes — no predicted run id.
    if let Some(w) = tele.as_ref() {
        obs.bind(w.run_dir())?;
    }

    let mut obs_vec = env.reset();
    normalizer.observe(&obs_vec);

    // The env block is the ground-truth record of how the aircraft flew, so it
    // is sampled whenever telemetry is on — not only when a live viewer also
    // wants frames (`--stream off` must still produce it).
    let telemetry_on = tele.is_some();
    // Episodes can straddle iteration boundaries, so their length, reward total
    // and accumulator all live outside the per-iteration loop. Keeping
    // `ep_steps` here also makes the episode's reported length its true length,
    // not just the part collected since the last iteration boundary.
    let mut episode_stats = EnvStats::default();
    let mut episode_reward = 0.0f64;
    let mut ep_steps = 0usize;

    let mut last_eval: Option<EvalMetrics> = None;
    let run_start = std::time::Instant::now();
    let mut last_iter_at = run_start;
    // Monotonic across the whole run; frames carry it so a viewer can segment
    // the ring by episode rather than inferring boundaries from flags alone.
    let mut episode_id: u32 = 0;
    let mut stop_requested = false;

    for it in 0..tcfg.iterations {
        // ---- Control gate: iteration boundary ----
        // Polling here rather than inside the rollout means a pause can never
        // strand a half-collected rollout, and the inbox costs one directory
        // scan per iteration instead of per step.
        let mut force_eval = false;
        let mut force_save = false;
        for action in obs.control.poll() {
            match action {
                crate::stream::ControlAction::EvaluateNow => force_eval = true,
                crate::stream::ControlAction::SaveCheckpoint => force_save = true,
                crate::stream::ControlAction::Stop => stop_requested = true,
                // Pause/Resume mutate the channel's own state; nothing to do.
                crate::stream::ControlAction::Pause | crate::stream::ControlAction::Resume => {}
            }
        }
        // Park here until resumed. Backs off so a long pause does not spin.
        let mut paused_for = std::time::Duration::ZERO;
        while obs.control.is_paused() && !stop_requested {
            crate::stream::pause_sleep(paused_for);
            paused_for += std::time::Duration::from_millis(100);
            // Keep honouring commands while parked, so a Resume (or Stop)
            // queued during the pause is picked up without the GUI having to
            // signal out-of-band.
            for action in obs.control.poll() {
                if matches!(action, crate::stream::ControlAction::Stop) {
                    stop_requested = true;
                }
            }
        }
        if stop_requested {
            println!("[rl_agent] stop requested via control channel");
            break;
        }

        let mut buffer = RolloutBuffer::new(ncfg.obs_dim, ncfg.action_dim);
        let mut final_obs: Option<Vec<f32>> = None;
        let mut roll = RolloutStats::default();

        // `streaming` hoisted out of the loop: it is a plain bool, so the inner
        // loop reads a register instead of chasing the observer's Option.
        let streaming = obs.frames.is_enabled();

        // ---- 1. Rollout ----
        let mut env_stats = EnvStats::default();
        for step in 0..tcfg.rollout_steps {
            let obs_f32: Vec<f32> = obs_vec.iter().map(|&x| x as f32).collect();
            let norm = normalizer.normalize_f32(&obs_f32);
            let (action, log_prob) = agent.act(&norm);
            let outcome = env.step(&action);

            let mut reward = outcome.reward;
            if outcome.terminated {
                reward += env.crash_penalty();
            }
            // Sample the true state once per step when either consumer needs
            // it. `env.step` has already run, so the frame reports the state
            // this action produced, and `action` is still ours to borrow — no
            // clone, no extra allocation. The telemetry accumulators and the
            // live ring get the same frame, so the two views can never disagree
            // about what happened.
            //
            // `try_send` is non-blocking, so the streaming cost is one atomic
            // increment in the common case, or one dropped frame when the
            // writer falls behind. Training throughput never waits on the GUI.
            if streaming || telemetry_on {
                if let Some(ff) = env.frame() {
                    if telemetry_on {
                        let wx = env.weather();
                        env_stats.observe(&ff, wx.as_ref());
                        episode_stats.observe(&ff, wx.as_ref());
                    }
                    if streaming {
                        obs.frames.push(crate::stream::Frame::from_flight(
                            &ff,
                            &action,
                            reward as f32,
                            it as u32,
                            step as u32,
                            episode_id,
                            0, // worker_id: the rollout is serial today.
                            outcome.terminated,
                            outcome.truncated,
                        ));
                    }
                }
            }

            buffer.push(Transition {
                obs: obs_f32,
                action,
                log_prob,
                reward: reward as f32,
                terminated: outcome.terminated,
                truncated: outcome.truncated,
            });

            roll.reward_sum += reward;
            episode_reward += reward;
            ep_steps += 1;
            let next_f32: Vec<f32> = outcome.obs.iter().map(|&x| x as f32).collect();
            final_obs = Some(next_f32);

            if outcome.terminated || outcome.truncated {
                roll.completed_episodes += 1;
                roll.crash_count += usize::from(outcome.terminated);
                roll.episode_len_sum += ep_steps;

                // Flush the episode row before resetting: the accumulator and
                // reward total still describe the episode that just ended.
                if telemetry_on {
                    if let Some(summary) = episode_stats.summary() {
                        if let Some(w) = tele.as_mut() {
                            w.record_episode(&EpisodeRecord {
                                episode: episode_id,
                                iteration: it,
                                steps: ep_steps,
                                duration_s: ep_steps as f64 * tcfg.env_dt,
                                reward_sum: episode_reward,
                                terminated: outcome.terminated,
                                truncated: outcome.truncated,
                                termination: episode_termination(
                                    &env,
                                    outcome.terminated,
                                    outcome.truncated,
                                ),
                                env: summary,
                            })?;
                        }
                    }
                }

                ep_steps = 0;
                episode_reward = 0.0;
                episode_stats = EnvStats::default();
                episode_id += 1;
                obs_vec = env.reset();
                normalizer.observe(&obs_vec);
            } else {
                obs_vec = outcome.obs;
            }
        }

        let final_obs = final_obs.expect("rollout_steps > 0 guarantees one transition");

        // ---- 2. Values + GAE ----
        let norm_batch = normalizer.normalize_batch(&buffer.obs_flat());
        let values = agent.value_all(&norm_batch);
        let last_value = match buffer.last() {
            Some(last) if last.terminated => 0.0,
            Some(_) => agent.value_single(&normalizer.normalize_f32(&final_obs)),
            None => 0.0,
        };
        let (advantages, returns) = compute_gae(
            &buffer.rewards(),
            &values,
            &buffer.terminated(),
            last_value,
            gamma,
            lambda,
        );

        // ---- 3. Update ----
        let batch = UpdateBatch {
            obs: norm_batch,
            actions: buffer.actions_flat(),
            old_log_probs: buffer.log_probs(),
            advantages,
            returns,
        };
        let stats = agent.update(&batch);

        // ---- 4. Eval + checkpoint + telemetry ----
        let iter_elapsed = run_start.elapsed().as_secs_f64();
        let since_last = last_iter_at.elapsed().as_secs_f64().max(1e-9);
        last_iter_at = std::time::Instant::now();
        let steps_per_s = tcfg.rollout_steps as f64 / since_last;

        let mut iter_eval: Option<&EvalMetrics> = None;
        if force_eval || (tcfg.eval_every > 0 && it % tcfg.eval_every == 0) {
            last_eval = Some(evaluate(
                agent.as_ref(),
                &mut env,
                &normalizer,
                tcfg.eval_episodes,
                tcfg.env_max_steps,
            ));
            if let Some(ev) = last_eval.as_ref() {
                println!(
                    "[rl_agent] eval       alt_err {:.1} m  spd_err {:.1} m/s  reward {:.1}  crashes {}/{}",
                    ev.alt_err, ev.spd_err, ev.total_reward, ev.crashes, ev.episodes,
                );
            }
            iter_eval = last_eval.as_ref();
        }

        let mut saved = false;
        if force_save
            || (tcfg.save_every > 0 && (it % tcfg.save_every == 0 || it == tcfg.iterations - 1))
        {
            save_checkpoint(agent.as_ref(), &normalizer, tcfg, ncfg, &tcfg.checkpoint_dir)?;
            saved = true;
        }

        if let Some(w) = tele.as_mut() {
            // Report stream loss alongside the metrics rather than on a
            // separate channel: a viewer reading `metrics.jsonl` already has
            // everything it needs to know whether it missed something.
            let stream = obs.frames.stats().map(|s| {
                let (frames, dropped, episodes) = s.load();
                crate::telemetry::StreamRecord {
                    frames,
                    dropped,
                    episodes,
                }
            });
            w.record(&IterationRecord {
                iteration: it,
                elapsed_s: iter_elapsed,
                steps_per_s,
                policy_loss: stats.policy_loss,
                value_loss: stats.value_loss,
                entropy: stats.entropy,
                advantage_mean: stats.advantage_mean,
                advantage_std: stats.advantage_std,
                mean_reward_per_step: roll.mean_reward_per_step(tcfg.rollout_steps),
                mean_episode_steps: roll.mean_episode_steps(),
                crash_rate: roll.crash_rate(),
                episodes_completed: roll.completed_episodes,
                eval: iter_eval,
                env: env_stats.summary(),
                checkpoint: saved,
                stream,
            })?;
        }

        if tcfg.print_every > 0 && (it % tcfg.print_every == 0 || it == tcfg.iterations - 1) {
            println!(
                "[rl_agent] iter {:>4}/{}  policy {:.4}  value {:.4}  entropy {:.4}  \
                 adv {:.3}±{:.3}  rew/step {:.4}  crash {:>3.0}%  eps {:>2}  {:>7.1} sps",
                it + 1,
                tcfg.iterations,
                stats.policy_loss,
                stats.value_loss,
                stats.entropy,
                stats.advantage_mean,
                stats.advantage_std,
                roll.mean_reward_per_step(tcfg.rollout_steps),
                roll.crash_rate() * 100.0,
                roll.completed_episodes,
                steps_per_s,
            );
        }
    }

    if let Some(w) = tele.as_mut() {
        w.finish()?;
    }

    Ok(TrainingOutcome {
        algorithm: agent,
        normalizer,
        last_eval,
        run_dir: tele.as_ref().map(|w| w.run_dir().to_path_buf()),
    })
}

/// Run `episodes` deterministic episodes and aggregate metrics.
pub fn evaluate<B: AutodiffBackend<FloatElem = f32>>(
    agent: &dyn Algorithm<B>,
    env: &mut dyn EnvSpec,
    normalizer: &RunningNormalizer,
    episodes: usize,
    max_steps: usize,
) -> EvalMetrics {
    let mut m = EvalMetrics {
        episodes,
        ..Default::default()
    };
    if episodes == 0 {
        return m;
    }
    let target_alt = env.target_altitude();
    let target_spd = env.target_airspeed();

    for _ in 0..episodes {
        let mut obs = env.reset();
        let mut total_reward = 0.0f64;
        let mut steps = 0usize;
        let mut crashed = false;
        let mut alt_err = 0.0f64;
        let mut spd_err = 0.0f64;
        let mut done = false;

        while !done && steps < max_steps {
            let obs_f32: Vec<f32> = obs.iter().map(|&x| x as f32).collect();
            let action = agent.deterministic_action(&normalizer.normalize_f32(&obs_f32));
            let out = env.step(&action);
            total_reward += out.reward;
            if out.terminated {
                total_reward += env.crash_penalty();
                crashed = true;
            }
            alt_err += (out.obs[10] - target_alt).abs();
            spd_err += (out.obs[11] - target_spd).abs();
            done = out.terminated || out.truncated;
            obs = out.obs;
            steps += 1;
        }

        m.alt_err += (alt_err / steps.max(1) as f64) as f32;
        m.spd_err += (spd_err / steps.max(1) as f64) as f32;
        m.total_reward += (total_reward / steps.max(1) as f64) as f32;
        m.mean_steps += steps as f32;
        m.crashes += usize::from(crashed);
    }

    let n = episodes as f32;
    m.alt_err /= n;
    m.spd_err /= n;
    m.total_reward /= n;
    m.mean_steps /= n;
    m
}

/// Save the model weights (bin) plus JSON configs and normalizer.
///
/// The algorithm id + hyper-parameters are persisted as `algo.json`; the
/// network architecture as `net.json`; weights as `net.bin` (via the
/// algorithm's own recorder).
pub fn save_checkpoint<B: AutodiffBackend<FloatElem = f32>>(
    agent: &dyn Algorithm<B>,
    normalizer: &RunningNormalizer,
    tcfg: &TrainerConfig,
    ncfg: &NetConfig,
    dir: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    std::fs::create_dir_all(dir)?;
    agent.save_weights(Path::new(dir))?;
    write_json(Path::new(dir).join("net.json"), &serde_json::to_string_pretty(ncfg)?)?;
    write_json(Path::new(dir).join("algo.json"), &agent.spec().to_json()?)?;
    write_json(
        Path::new(dir).join("trainer.json"),
        &serde_json::to_string_pretty(tcfg)?,
    )?;
    write_json(
        Path::new(dir).join("normalizer.json"),
        &serde_json::to_string_pretty(normalizer)?,
    )?;
    Ok(())
}

fn write_json(path: std::path::PathBuf, contents: &str) -> Result<(), Box<dyn std::error::Error>> {
    std::fs::write(path, contents)?;
    Ok(())
}

/// Loaded checkpoint: the trained algorithm plus the stats needed to run it.
pub struct Checkpoint<B: AutodiffBackend<FloatElem = f32>> {
    /// Trained algorithm (weights on `device`), algorithm-agnostic.
    pub algorithm: Box<dyn Algorithm<B>>,
    /// Normalizer stats to apply before feeding observations to the net.
    pub normalizer: RunningNormalizer,
}

/// Load a checkpoint from `dir`. Optimizer state is not stored in v1 and is
/// rebuilt fresh from the algorithm config by the factory.
///
/// Checkpoints written by v0.1 trainers (before the algorithm layer) used
/// `ppo.json` instead of `algo.json`; those are still loadable.
pub fn load_checkpoint<B: AutodiffBackend<FloatElem = f32>>(
    dir: &str,
    device: &B::Device,
    seed: u64,
) -> Result<Checkpoint<B>, Box<dyn std::error::Error>> {
    let net_cfg: NetConfig = serde_json::from_str(&std::fs::read_to_string(
        Path::new(dir).join("net.json"),
    )?)?;
    let spec: AlgoSpec = {
        let algo_path = Path::new(dir).join("algo.json");
        if algo_path.exists() {
            AlgoSpec::from_json(&std::fs::read_to_string(algo_path)?)?
        } else {
            // Legacy < 0.2 checkpoints: PPO only.
            let ppo_cfg: PpoConfig = serde_json::from_str(&std::fs::read_to_string(
                Path::new(dir).join("ppo.json"),
            )?)?;
            AlgoSpec::Ppo { config: ppo_cfg }
        }
    };
    let normalizer: RunningNormalizer = serde_json::from_str(
        &std::fs::read_to_string(Path::new(dir).join("normalizer.json"))?,
    )?;
    let mut algorithm = make_algorithm(&spec, &net_cfg, seed, device);
    algorithm.load_weights(Path::new(dir), device)?;
    Ok(Checkpoint {
        algorithm,
        normalizer,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::Cpu;
    use std::sync::atomic::{AtomicBool, Ordering};
    use std::sync::Arc;

    fn config_path() -> String {
        if Path::new("../aircraft.toml").exists() {
            "../aircraft.toml".to_string()
        } else {
            "aircraft.toml".to_string()
        }
    }

    fn trained() -> TrainingOutcome<Cpu> {
        let device = crate::backend::cpu_device();
        let tcfg = TrainerConfig {
            iterations: 2,
            rollout_steps: 64,
            env_max_steps: 300,
            env_dt: 0.05,
            seed: 1234,
            save_every: 0,
            print_every: 0,
            eval_episodes: 1,
            eval_every: 1,
            checkpoint_dir: "checkpoints".to_string(),
        };
        let ncfg = NetConfig {
            obs_dim: AVIONICS_OBS_DIM,
            hidden: 32,
            action_dim: ACTION_DIM,
            log_std_init: -0.5,
        };
        let spec = AlgoSpec::Ppo {
            config: PpoConfig {
                lr: 1e-3,
                gamma: 0.99,
                lambda: 0.95,
                clip_epsilon: 0.2,
                value_coef: 0.5,
                entropy_coef: 0.0,
                max_grad_norm: 5.0,
                epochs: 1,
                minibatch_size: 32,
            },
        };
        run::<Cpu>(&tcfg, &crate::telemetry::TelemetryConfig::disabled(), &ncfg, &spec, &device, &config_path()).expect("run should succeed")
    }

    #[test]
    fn training_run_produces_finite_results_and_eval() {
        let outcome = trained();
        assert!(outcome.normalizer.count > 0);
        // Rollouts + an eval episode must move the running stats.
        assert!(outcome.normalizer.mean.iter().all(|m| m.is_finite()));
        let ev = outcome.last_eval.expect("eval_every=1 must produce metrics");
        assert!(ev.alt_err.is_finite());
        assert!(ev.spd_err.is_finite());
        assert!(ev.total_reward.is_finite());
        assert_eq!(ev.episodes, 1);
    }

    #[test]
    fn same_seed_reproduces_identical_training() {
        let a = trained();
        let b = trained();
        assert_eq!(a.normalizer.mean, b.normalizer.mean, "normalizer must match run-to-run");
        let obs = vec![0.0f32; AVIONICS_OBS_DIM];
        // Deterministic actions on a fixed input must match exactly.
        let act_a = a.algorithm.deterministic_action(&obs);
        let act_b = b.algorithm.deterministic_action(&obs);
        assert_eq!(act_a, act_b, "same seed must reproduce the trained policy");
    }

    /// Read a frame out of a ring file at a logical index.
///
/// The ring only wraps once it is full, so for a short test run the logical
/// index is still the physical slot.
    fn ring_frame(ring: &[u8], index: usize) -> crate::stream::Frame {
        let stride = crate::stream::FRAME_BYTES;
        let slot = index % crate::stream::RING_SLOTS;
        let off = slot * stride;
        crate::stream::Frame::decode(&ring[off..off + stride]).expect("frame decodes")
    }

    /// Unique temp dir per test.
///
/// Tests run concurrently in one process, so a dir keyed only on the pid would
/// be shared and every run would see the others' artifacts.
    fn scratch(tag: &str) -> std::path::PathBuf {
        static NEXT: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
        let n = NEXT.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let p = std::env::temp_dir().join(format!(
            "rl_obs_{tag}_{}_{n}",
            std::process::id()
        ));
        let _ = std::fs::remove_dir_all(&p);
        p
    }

    /// Train a tiny run with live observation enabled, returning the run dir.
    fn trained_with_observer(observer: &mut TrainObserver, tcfg: TrainerConfig) -> std::path::PathBuf {
        let device = crate::backend::cpu_device();
        let ncfg = NetConfig {
            obs_dim: AVIONICS_OBS_DIM,
            hidden: 16,
            action_dim: ACTION_DIM,
            log_std_init: -0.5,
        };
        let spec = AlgoSpec::Ppo {
            config: PpoConfig {
                lr: 1e-3,
                gamma: 0.99,
                lambda: 0.95,
                clip_epsilon: 0.2,
                value_coef: 0.5,
                entropy_coef: 0.0,
                max_grad_norm: 5.0,
                epochs: 1,
                minibatch_size: 32,
            },
        };
        let root = scratch("ring");
        let _ = std::fs::remove_dir_all(&root);
        let tele = crate::telemetry::TelemetryConfig {
            run_root: root.to_str().unwrap().to_string(),
            name: Some("obs".into()),
        };
        let outcome = run_with_observer::<Cpu>(
            &tcfg,
            &tele,
            &ncfg,
            &spec,
            &device,
            &config_path(),
            observer,
        )
        .expect("run_with_observer should succeed");

        // Returned live: the caller inspects these artifacts, so the directory
        // must outlive this helper. The writer thread is still running, since
        // the observer is only borrowed — callers drop it before reading the
        // files so the buffered tail is flushed.
        outcome
            .run_dir
            .expect("telemetry was on, so a run dir exists")
    }

    fn tiny_tcfg() -> TrainerConfig {
        TrainerConfig {
            iterations: 2,
            rollout_steps: 48,
            env_max_steps: 60,
            env_dt: 0.05,
            seed: 7,
            save_every: 0,
            print_every: 0,
            eval_episodes: 0,
            eval_every: 0,
            checkpoint_dir: std::env::temp_dir()
                .join("rl_obs_ckpt")
                .to_str()
                .unwrap()
                .to_string(),
        }
    }

    #[test]
    fn observer_writes_a_frame_ring_the_trainer_actually_filled() {
        let mut obs = TrainObserver::pending(true, false);
        let run_dir = trained_with_observer(&mut obs, tiny_tcfg());

        // Drop the observer so the writer thread is joined and the ring's
        // buffered tail is flushed before anything reads the file.
        let (written, dropped, episodes) = obs.stream.as_ref().unwrap().counters();
        drop(obs);

        let ring_path = run_dir.join(crate::stream::FRAMES_FILE);
        assert!(ring_path.exists(), "no ring written to {}", run_dir.display());

        // The ring is a fixed-size file regardless of how much was recorded.
        let ring = std::fs::read(&ring_path).unwrap();
        assert_eq!(
            ring.len(),
            crate::stream::RING_SLOTS * crate::stream::FRAME_BYTES,
            "ring should be preallocated to its full budget"
        );

        // The sink counted real frames — this is the assertion that would fail
        // if `bind` were never called or `streaming` stayed false in the loop.
        let steps = 2 * 48usize as u64;
        assert_eq!(
            written, steps,
            "trainer should publish one frame per env step"
        );
        assert_eq!(
            dropped, 0,
            "a 96-step run should never saturate an 8192-slot channel"
        );
        // Episodes span iteration boundaries, so 96 steps at a 60-step budget
        // completes exactly one: the first ends truncated at step 60, and the
        // second needs another 60 steps it does not have. This is the
        // documented behaviour ("episodes continue across the boundary").
        assert_eq!(
            episodes, 1,
            "96 steps at a 60-step budget ends one episode"
        );

        // Decode the newest frame and check it carries real flight state.
        let f = ring_frame(&ring, written as usize - 1);
        assert_eq!(
            (f.iteration, f.step),
            (1, 47),
            "newest frame should be the last step of the last iteration"
        );
        assert!(f.altitude.is_finite() && f.altitude > 0.0, "alt {}", f.altitude);
        assert!(f.airspeed.is_finite() && f.airspeed > 10.0, "ias {}", f.airspeed);
        assert!(f.roll.is_finite() && f.pitch.is_finite());
        assert!((-1.5..1.5).contains(&f.pitch), "pitch {}", f.pitch);
        assert!(f.throttle >= 0.0 && f.throttle <= 1.0);
        assert_eq!(f.worker_id, 0, "serial rollout");

        // Within one episode, sim time must advance by exactly `env_dt` per step —
        // the check that would catch frames landing out of order, duplicated,
        // or written from the wrong step. It resets at each `reset()`, so the
        // assertion is per-episode rather than across the whole ring.
        let dt = tiny_tcfg().env_dt as f32;
        for i in 1..written as usize {
            let cur = ring_frame(&ring, i);
            let prev = ring_frame(&ring, i - 1);
            if cur.episode == prev.episode {
                assert!(
                    (cur.sim_time - prev.sim_time - dt).abs() < 1e-3,
                    "frame {i}: sim_time jumped {} -> {} (expected +{dt})",
                    prev.sim_time,
                    cur.sim_time
                );
            } else {
                assert!(
                    cur.episode == prev.episode + 1,
                    "frame {i}: episode went {} -> {}",
                    prev.episode,
                    cur.episode
                );
                assert!(
                    cur.sim_time <= dt * 2.0,
                    "frame {i}: new episode starts at sim_time {}",
                    cur.sim_time
                );
            }
        }
    }

    #[test]
    fn observer_episodes_directory_holds_replay_files() {
        // A budget of 40 over 2x48 steps ends exactly three episodes, all truncated,
        // so the replay files are fully deterministic to assert on.
        let mut tcfg = tiny_tcfg();
        tcfg.env_max_steps = 40;
        let mut obs = TrainObserver::pending(true, false);
        let run_dir = trained_with_observer(&mut obs, tcfg);
        drop(obs);

        let eps = run_dir.join(crate::stream::EPISODES_DIR);
        assert!(eps.is_dir(), "no episodes dir in {}", run_dir.display());
        let mut files: Vec<_> = std::fs::read_dir(&eps)
            .unwrap()
            .flatten()
            .map(|e| e.path())
            .filter(|p| p.extension().is_some_and(|e| e == "f32"))
            .collect();
        assert_eq!(
            files.len(),
            3,
            "one replay file per completed episode in {}",
            run_dir.display()
        );
        files.sort();

        // Each file is a whole number of frames, numbered from 0, and holds one
        // episode. 96 steps at a 40-step budget gives 2 complete episodes plus
        // a 16-step tail that is still in flight when the run ends — the writer
        // flushes that partial one on shutdown rather than discarding it.
        let stride = crate::stream::FRAME_BYTES;
        let mut lengths = Vec::new();
        for (i, f) in files.iter().enumerate() {
            let bytes = std::fs::read(f).unwrap();
            assert_eq!(
                bytes.len() % stride,
                0,
                "{} is not a whole number of frames",
                f.display()
            );
            let n = bytes.len() / stride;
            lengths.push(n);
            let first = crate::stream::Frame::decode(&bytes[..stride]).unwrap();
            assert_eq!(first.episode as usize, i, "episode numbering in {}", f.display());
            assert!(n <= 40, "{} holds {} frames, past the 40-step budget", f.display(), n);
            // Complete episodes end on the frame the writer keys on; only the
            // final in-flight clip may lack it.
            let last = crate::stream::Frame::decode(&bytes[bytes.len() - stride..]).unwrap();
            assert_eq!(
                last.is_episode_end(),
                i < 2,
                "{} has the wrong episode-end flag",
                f.display()
            );
        }
        assert_eq!(lengths, vec![40, 40, 16], "unexpected episode lengths");
    }

    #[test]
    fn series_ignores_non_finite_samples() {
        let mut s = Series::default();
        s.observe(f64::NAN);
        s.observe(f64::INFINITY);
        assert_eq!(s.count, 0, "non-finite samples must not be counted");
        assert_eq!(s.mean(), 0.0);

        s.observe(2.0);
        s.observe(4.0);
        assert_eq!(s.count, 2);
        assert!((s.mean() - 3.0).abs() < 1e-12);
        assert!((s.min() - 2.0).abs() < 1e-12);
        assert!((s.max() - 4.0).abs() < 1e-12);
    }

    #[test]
    fn weather_stats_summarise_absolute_quantities() {
        let wx = WeatherObservation {
            wind: flight_core::nalgebra::Vector3::new(1.0, 2.0, 0.0),
            turbulence_level: 2,
            gust_rms: 3.0,
            rain_intensity: 0.5,
            visibility: 1000.0,
            shear_gradient: -0.02,
            updraft_strength: -1.5,
            precipitation: flight_core::PrecipitationType::Rain,
            severity: flight_core::WeatherSeverity::Moderate,
            wind_speed: 7.0,
            timestamp: 1.0,
        };
        let mut stats = WeatherStats::default();
        stats.observe(&wx);
        let s = stats.summary().expect("one sample is a summary");
        assert_eq!(s.samples, 1);
        assert!((s.wind_speed_mean - 7.0).abs() < 1e-12);
        assert!((s.gust_rms_max - 3.0).abs() < 1e-12);
        assert!((s.turbulence_mean - 2.0).abs() < 1e-12);
        assert!((s.severity_mean - 2.0).abs() < 1e-12);
        assert!((s.visibility_min - 1000.0).abs() < 1e-12);
        // Magnitudes, not signs: a downdraft and a negative shear both cost
        // control authority, so the summary reports |updraft| and |shear|.
        assert!((s.updraft_abs_mean - 1.5).abs() < 1e-12);
        assert!((s.shear_abs_mean - 0.02).abs() < 1e-12);
    }

    #[test]
    fn env_telemetry_is_written_even_with_streaming_off() {
        // `--stream off` must not suppress the env block: it is training data,
        // not a live-view side effect.
        let mut obs = TrainObserver::pending(false, false);
        let run_dir = trained_with_observer(&mut obs, tiny_tcfg());
        drop(obs);

        let metrics = std::fs::read_to_string(run_dir.join("metrics.jsonl")).expect("metrics");
        let rows: Vec<serde_json::Value> = metrics
            .lines()
            .filter(|l| !l.trim().is_empty())
            .map(|l| serde_json::from_str(l).expect("iteration json"))
            .collect();
        assert_eq!(rows.len(), 2, "two iterations were requested");
        for row in &rows {
            let env = row.get("env").expect("iteration carries an env block");
            assert_eq!(env["samples"], 48, "one sample per rollout step");
            assert!(env["alt_mean"].as_f64().unwrap() > 0.0);
            assert!(env["alt_min"].as_f64().unwrap() <= env["alt_max"].as_f64().unwrap());
            assert!(env["airspeed_mean"].as_f64().unwrap().is_finite());
            assert!(env["climb_mean"].as_f64().unwrap().is_finite());
            assert!(env["surface_effort_mean"].as_f64().unwrap() >= 0.0);
            // Default EnvConfig has no weather system, so the block is absent.
            assert!(env.get("weather").is_none());
        }

        // Episodes end on a crash or the step budget; which one is seed- and
        // policy-dependent, so assert the invariants rather than a fixed count.
        let episodes =
            std::fs::read_to_string(run_dir.join("episodes.jsonl")).expect("episodes");
        let rows: Vec<serde_json::Value> = episodes
            .lines()
            .filter(|l| !l.trim().is_empty())
            .map(|l| serde_json::from_str(l).expect("episode json"))
            .collect();
        assert!(!rows.is_empty(), "at least one episode ended inside the run");
        for (i, ep) in rows.iter().enumerate() {
            assert_eq!(ep["episode"], i as u64, "episodes are numbered in order");
            let steps = ep["steps"].as_u64().expect("steps");
            assert!((1..=60).contains(&steps), "steps {steps} outside budget");
            assert_eq!(ep["env"]["samples"], steps, "one env sample per episode step");
            assert!(ep["reward_sum"].as_f64().unwrap().is_finite());
            let termination = ep["termination"].as_str().expect("termination");
            assert!(
                matches!(
                    termination,
                    "ground_impact"
                        | "gust_limit"
                        | "visibility_limit"
                        | "severe_exposure"
                        | "step_budget"
                ),
                "unexpected termination {termination}"
            );
            assert_eq!(
                ep["truncated"].as_bool().unwrap(),
                termination == "step_budget",
                "truncation and termination cause must agree"
            );
        }
    }

    #[test]
    fn stop_command_ends_training_early() {
        // The run asks for 50 iterations but a `Stop` is queued before it
        // starts, so it must exit after the first boundary. Without this the
        // trainer would ignore the control channel entirely.
        let mut obs = TrainObserver::pending(false, true);
        let root = scratch("stop");

        let device = crate::backend::cpu_device();
        let tcfg = TrainerConfig {
            iterations: 50,
            // A long first rollout, so the watcher thread's `stop` is
            // guaranteed to be in the inbox before the first boundary is
            // polled. With a tiny rollout the run could finish iteration 0
            // before the thread ever wrote the file, which would make this
            // test a race rather than an assertion about the control channel.
            rollout_steps: 512,
            env_max_steps: 512,
            env_dt: 0.05,
            seed: 3,
            save_every: 0,
            print_every: 0,
            eval_episodes: 0,
            eval_every: 0,
            checkpoint_dir: std::env::temp_dir()
                .join("rl_stop_ckpt")
                .to_str()
                .unwrap()
                .to_string(),
        };
        let ncfg = NetConfig {
            obs_dim: AVIONICS_OBS_DIM,
            hidden: 8,
            action_dim: ACTION_DIM,
            log_std_init: -0.5,
        };
        let spec = AlgoSpec::Ppo {
            config: PpoConfig {
                lr: 1e-3,
                gamma: 0.99,
                lambda: 0.95,
                clip_epsilon: 0.2,
                value_coef: 0.5,
                entropy_coef: 0.0,
                max_grad_norm: 5.0,
                epochs: 1,
                minibatch_size: 8,
            },
        };
        let tele = crate::telemetry::TelemetryConfig {
            run_root: root.to_str().unwrap().to_string(),
            name: Some("stop".into()),
        };
        // The run directory is created *inside* the trainer, and it is where the
        // inbox lives, so a client cannot queue a command until that directory
        // exists. That is the real protocol: a GUI discovers the new run dir
        // and drops commands into it. Spawn a thread that does exactly that,
        // standing in for the GUI.
        let watcher_root = root.clone();
        let queued = Arc::new(AtomicBool::new(false));
        let queued_flag = Arc::clone(&queued);
        let watcher = std::thread::spawn(move || {
            for _ in 0..40_000 {
                if let Ok(entries) = std::fs::read_dir(&watcher_root) {
                    for e in entries.flatten() {
                        let p = e.path();
                        if p.is_dir() {
                            // Wait until iteration 0 has actually been
                            // recorded before stopping. A stop that lands
                            // before the first write would exit with zero
                            // records, which proves the channel works but not
                            // that it can interrupt a run *in progress* — the
                            // thing a GUI's stop button needs to do.
                            let metrics = p.join("metrics.jsonl");
                            let started = std::fs::metadata(&metrics)
                                .map(|m| m.len() > 0)
                                .unwrap_or(false);
                            if !started {
                                continue;
                            }
                            let inbox = p.join(crate::stream::CONTROL_INBOX);
                            if std::fs::create_dir_all(&inbox).is_ok() {
                                let cmd = inbox.join("cmd_000000000001.json");
                                std::fs::write(&cmd, r#"{"seq":1,"op":"stop"}"#).unwrap();
                                queued_flag.store(true, Ordering::SeqCst);
                                return p;
                            }
                        }
                    }
                }
                std::thread::sleep(std::time::Duration::from_millis(1));
            }
            panic!("run never started, so the stop command was never queued");
        });

        let run_dir = run_with_observer::<Cpu>(
            &tcfg,
            &tele,
            &ncfg,
            &spec,
            &device,
            &config_path(),
            &mut obs,
        )
        .expect("stopped run still returns Ok")
        .run_dir
        .expect("telemetry was on, so a run dir exists");
        watcher.join().expect("watcher thread");
        assert!(
            queued.load(Ordering::SeqCst),
            "the stop command was never written"
        );

        // The run must have stopped well short of 50 iterations. Exactly one
        // is not asserted: the boundary polls once per iteration, and whether
        // it sees the command on the first poll is a scheduling detail. What
        // matters is that it stopped at all, and did not run to completion.
        let metrics = std::fs::read_to_string(run_dir.join("metrics.jsonl")).unwrap();
        let lines = metrics
            .lines()
            .filter(|l| !l.trim().is_empty())
            .count();
        assert!(
            (1..50).contains(&lines),
            "expected an early stop, got {lines} iteration records"
        );

        // And the trainer must acknowledge what it acted on.
        let ack_dir = run_dir.join(crate::stream::CONTROL_OUTBOX);
        let ack: crate::stream::ControlAck = std::fs::read_to_string(
            ack_dir.join("ack_000000000001.json"),
        )
        .map(|s| serde_json::from_str(&s).expect("ack is valid json"))
        .expect("a stop command must be acknowledged");
        assert_eq!(ack.seq, 1);
        assert!(ack.ok, "stop should be accepted: {ack:?}");
        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn action_mapping_stays_inside_fc_limits() {
        let action = [0.999f32, -0.999, 0.999, -0.999];
        let av = action_to_avionics(&action);
        assert!(av.roll_cmd.abs() <= 0.6 + 1e-6);
        assert!(av.pitch_cmd.abs() <= 0.4 + 1e-6);
        assert!(av.yaw_rate_cmd.abs() <= 0.8 + 1e-6);
        assert!((0.0..=1.0).contains(&av.throttle));
    }

    #[test]
    fn checkpoint_round_trip() {
        use crate::net::NetConfig;
        let device = crate::backend::cpu_device();
        let tcfg = TrainerConfig {
            iterations: 2,
            rollout_steps: 16,
            env_max_steps: 100,
            env_dt: 0.05,
            seed: 99,
            save_every: 1,
            print_every: 0,
            eval_episodes: 0,
            eval_every: 0,
            checkpoint_dir: std::env::temp_dir().join("rl_agent_cp").to_str().unwrap().to_string(),
        };
        let ncfg = NetConfig {
            obs_dim: AVIONICS_OBS_DIM,
            hidden: 16,
            action_dim: ACTION_DIM,
            log_std_init: -0.5,
        };
        let spec = AlgoSpec::Ppo {
            config: PpoConfig {
                minibatch_size: 16,
                epochs: 1,
                ..Default::default()
            },
        };
        run::<Cpu>(&tcfg, &crate::telemetry::TelemetryConfig::disabled(), &ncfg, &spec, &device, &config_path()).expect("run");

        let cp = load_checkpoint::<Cpu>(&tcfg.checkpoint_dir, &device, tcfg.seed).expect("load");
        assert_eq!(cp.algorithm.net_config().hidden, 16);
        assert_eq!(cp.algorithm.id(), crate::algo::ALGO_PPO);
        match cp.algorithm.spec() {
            AlgoSpec::Ppo { config } => assert_eq!(config.minibatch_size, 16),
        }
        assert_eq!(cp.normalizer.dim, AVIONICS_OBS_DIM);

        let _ = std::fs::remove_dir_all(&tcfg.checkpoint_dir);
    }

    #[test]
    fn loads_legacy_ppo_checkpoint_without_algo_json() {
        // Checkpoints written before the algorithm layer stored PPO config as
        // `ppo.json` and had no `algo.json`; they must still load.
        use crate::net::NetConfig;
        let device = crate::backend::cpu_device();
        let dir = std::env::temp_dir()
            .join("rl_agent_cp_legacy")
            .to_str()
            .unwrap()
            .to_string();
        let tcfg = TrainerConfig {
            iterations: 1,
            rollout_steps: 16,
            env_max_steps: 100,
            env_dt: 0.05,
            seed: 7,
            save_every: 1,
            print_every: 0,
            eval_episodes: 0,
            eval_every: 0,
            checkpoint_dir: dir.clone(),
        };
        let ncfg = NetConfig {
            obs_dim: AVIONICS_OBS_DIM,
            hidden: 16,
            action_dim: ACTION_DIM,
            log_std_init: -0.5,
        };
        let spec = AlgoSpec::Ppo {
            config: PpoConfig::default(),
        };
        run::<Cpu>(&tcfg, &crate::telemetry::TelemetryConfig::disabled(), &ncfg, &spec, &device, &config_path()).expect("run");

        // Rewrite the checkpoint as a legacy one: drop algo.json, add ppo.json.
        std::fs::remove_file(std::path::Path::new(&dir).join("algo.json")).expect("remove algo.json");
        let ppo_cfg = match spec {
            AlgoSpec::Ppo { config } => config,
        };
        std::fs::write(
            std::path::Path::new(&dir).join("ppo.json"),
            serde_json::to_string_pretty(&ppo_cfg).expect("serialize ppo.json"),
        )
        .expect("write ppo.json");

        let cp = load_checkpoint::<Cpu>(&dir, &device, tcfg.seed).expect("legacy load");
        assert_eq!(cp.algorithm.id(), crate::algo::ALGO_PPO);
        match cp.algorithm.spec() {
            AlgoSpec::Ppo { config } => assert_eq!(config.minibatch_size, PpoConfig::default().minibatch_size),
        }
        let _ = std::fs::remove_dir_all(&dir);
    }
}