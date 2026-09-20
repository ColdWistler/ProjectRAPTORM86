//! # RL-Ready Weather System
//!
//! A stochastic, seed-reproducible **weather and turbulence layer** for the
//! fixed-wing flight dynamics engine, designed as an observation/reward source
//! for Rust reinforcement-learning agents (tch-rs, rust-rl, custom PPO/A2C, …).
//!
//! The weather system stacks on top of the Dryden turbulence model in
//! [`crate::wind`]:
//!
//! * **Steady wind** with power-law altitude shear (boundary layer).
//! * **Dryden-style time-correlated gusts** (AR-1 spectral shaping), whose
//!   RMS envelope is driven by a slow Ornstein–Uhlenbeck random walk at a
//!   configurable `gust_frequency` so "gusts come in waves".
//! * **Precipitation** (rain / snow / hail), reduced **visibility**, and
//!   **updraft/downdraft** vertical wind that observably shear the flow.
//!
//! The whole field is driven by a [`rand::SeedableRng`] stream so any two runs
//! with the same seed produce byte-identical wind — the reproducibility RL
//! experiment comparison needs. Two recipe modes are provided:
//!
//! * `deterministic = true` — a **fixed pattern** for the whole episode:
//!   same seed = the exact same weather time-series every episode restart
//!   (perfect A/B agent comparison).
//! * `deterministic = false` — weather re-rolls to a new scene every
//!   `auto_change_interval` seconds; still fully reproducible (same RNG
//!   stream), but exercises the agent across varying conditions.
//!
//! For curriculum learning a [`WeatherCurriculum`] climbs through
//! clear → light turbulence → rain → storm phases, and `WeatherConfig::from_preset`
//! provides fixed stress-test profiles (e.g. [`WeatherPreset::ExtremeStorm`]).
//!
//! # Frames & units
//! All winds are in the Earth **NED** frame `(north, east, down)` in m/s,
//! matching [`crate::wind::WindEnvironment`] and [`crate::AircraftState`].
//! Updraft strength is reported as *positive = up* (NED down component is
//! negated for the vertical). Visibility is in metres, `rain_intensity` and all
//! `*_01` normalized observations in `[0, 1]`.
//!
//! # Observation (RL)
//! [`WeatherSystem::observation_array`] packages a fixed 12-channel vector
//! (see [`WEATHER_OBS_DIM`] and the channel table on
//! [`WeatherObservation`]); [`WeatherSystem::to_dvector`] yields an
//! `nalgebra::DVector<f32>` for tensor-backed RL libraries, and
//! [`ObservationNormalizer`] tracks running min/max for input scaling.
//!
//! # Standards references
//! - Turbulence spectra — MIL-F-8785C / MIL-HDBK-1797 (Dryden), implemented in
//!   [`crate::wind`]; severe/extreme bands extend the standard categories to
//!   UAV-stress-test magnitudes.
//! - Boundary-layer shear — power-law `V(h) = V_ref·(h/h_ref)^p` profile, p≈0.2
//!   over open terrain (ESDU micrometeorology).
//! - Reward shaping follows the standard RL flight-env structure: negative
//!   penalties for weather penetration (see the `*_penalty` methods).

use crate::wind::{WindConfig, WindEnvironment};
use nalgebra::{DVector, Vector3};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use rand_distr::{Distribution, StandardNormal};

// ---------------------------------------------------------------------------
// Public constants
// ---------------------------------------------------------------------------

/// Dimension of the compact agent observation vector produced by
/// [`WeatherSystem::observation_array`]. Channel layout (SI unless noted):
/// ```text
///   0  wind_north (m/s, NED)
///   1  wind_east  (m/s, NED)
///   2  wind_down  (m/s, NED)
///   3  turbulence_01  (0 = light … 1 = extreme, RMS σ/9.0)
///   4  rain_intensity (0..1)
///   5  visibility_01  (1 = clear … 0 = zero visibility)
///   6  shear_01       (|dW/dh| normalised, 1 = 0.05 (m/s)/m)
///   7  updraft_strength (m/s, positive = up)
///   8  precipitation: rain (0/1)
///   9  precipitation: snow (0/1)
///  10  precipitation: hail (0/1)
///  11  weather_severity_01 (0 = Clear … 1 = Severe)
/// ```
pub const WEATHER_OBS_DIM: usize = 12;

/// Reference visibility (m) treated as "full / unlimited" for normalization.
/// A GA-type clear day is 20 km.
pub const REFERENCE_VISIBILITY: f64 = 20_000.0;

/// Reference turbulence RMS (m/s) (the `Extreme` band σ_u) used to normalise
/// `turbulence_01` to `[0, 1]`.
pub const REFERENCE_TURBULENCE_SIGMA: f64 = 9.0;

/// Reference shear gradient `(m/s)/m` treated as "severe" for `shear_01`.
pub const REFERENCE_SHEAR_GRADIENT: f64 = 0.05;

/// Default auto-change interval (s) between weather scenes.
pub const DEFAULT_AUTO_CHANGE_INTERVAL: f64 = 30.0;

/// Gust `σ_u` (m/s) of each [`TurbulenceLevel`] band.
pub fn turbulence_sigma_u(level: TurbulenceLevel) -> f64 {
    match level {
        TurbulenceLevel::Light => 1.5,
        TurbulenceLevel::Moderate => 3.0,
        TurbulenceLevel::Severe => 5.5,
        TurbulenceLevel::Extreme => REFERENCE_TURBULENCE_SIGMA,
    }
}

// ---------------------------------------------------------------------------
// Enums
// ---------------------------------------------------------------------------

/// Turbulence severity band for the agent observation
/// (`Light = 0` … `Extreme = 3`). Extends the flight-core categories with the
/// `Extreme` band used for RL stress testing.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
#[repr(i32)]
pub enum TurbulenceLevel {
    Light = 0,
    Moderate = 1,
    Severe = 2,
    Extreme = 3,
}

impl TurbulenceLevel {
    /// Index in the enum declaration order (`Light = 0`).
    pub fn index(self) -> i32 {
        self as i32
    }

    pub fn from_index(i: i32) -> Self {
        match i {
            0 => TurbulenceLevel::Light,
            1 => TurbulenceLevel::Moderate,
            2 => TurbulenceLevel::Severe,
            _ => TurbulenceLevel::Extreme,
        }
    }

    pub fn name(self) -> &'static str {
        match self {
            TurbulenceLevel::Light => "light",
            TurbulenceLevel::Moderate => "moderate",
            TurbulenceLevel::Severe => "severe",
            TurbulenceLevel::Extreme => "extreme",
        }
    }

    /// RMS `σ_u` (m/s) of this band.
    pub fn sigma_u(self) -> f64 {
        turbulence_sigma_u(self)
    }

    /// Heuristic band for a measured RMS gust `σ` (m/s).
    pub fn from_sigma(sigma: f64) -> Self {
        let s = sigma.abs();
        if s <= 1.65 {
            TurbulenceLevel::Light
        } else if s <= 3.3 {
            TurbulenceLevel::Moderate
        } else if s <= 6.05 {
            TurbulenceLevel::Severe
        } else {
            TurbulenceLevel::Extreme
        }
    }
}

/// Overall weather severity. `Clear = 0` … `Severe = 3` (matches the exported
/// `WeatherSeverity` enum used by the Godot bridge; `turbulence_01` /
/// `severity_01` in the observation are normalized from these ranks).
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
#[repr(i32)]
pub enum WeatherSeverity {
    Clear = 0,
    Light = 1,
    Moderate = 2,
    Severe = 3,
}

impl WeatherSeverity {
    pub fn index(self) -> i32 {
        self as i32
    }

    pub fn from_index(i: i32) -> Self {
        match i {
            0 => WeatherSeverity::Clear,
            1 => WeatherSeverity::Light,
            2 => WeatherSeverity::Moderate,
            _ => WeatherSeverity::Severe,
        }
    }

    pub fn name(self) -> &'static str {
        match self {
            WeatherSeverity::Clear => "clear",
            WeatherSeverity::Light => "light",
            WeatherSeverity::Moderate => "moderate",
            WeatherSeverity::Severe => "severe",
        }
    }

    /// Normalized `[0, 1]` rank suitable for reward shaping / observation.
    pub fn normalized(self) -> f64 {
        self.index() as f64 / 3.0
    }
}

/// Precipitation the agent can condition its policy on. Three one-hot channels
/// (rain / snow / hail) are packed into the observation vector.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(i32)]
pub enum PrecipitationType {
    None = 0,
    Rain = 1,
    Snow = 2,
    Hail = 3,
}

impl PrecipitationType {
    pub fn index(self) -> i32 {
        self as i32
    }

    pub fn from_index(i: i32) -> Self {
        match i {
            1 => PrecipitationType::Rain,
            2 => PrecipitationType::Snow,
            3 => PrecipitationType::Hail,
            _ => PrecipitationType::None,
        }
    }

    /// One-hot `[rain, snow, hail]` (all zeros for `None`).
    pub fn one_hot(self) -> [f64; 3] {
        match self {
            PrecipitationType::Rain => [1.0, 0.0, 0.0],
            PrecipitationType::Snow => [0.0, 1.0, 0.0],
            PrecipitationType::Hail => [0.0, 0.0, 1.0],
            PrecipitationType::None => [0.0, 0.0, 0.0],
        }
    }

    pub fn name(self) -> &'static str {
        match self {
            PrecipitationType::None => "none",
            PrecipitationType::Rain => "rain",
            PrecipitationType::Snow => "snow",
            PrecipitationType::Hail => "hail",
        }
    }
}

/// Ready-made weather profiles. `from_preset` produces a full
/// [`WeatherConfig`]; also used by [`WeatherCurriculum`] phases.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WeatherPreset {
    /// VFR, light breeze.
    Clear,
    /// Still visual but noticeably rough air.
    LightTurb,
    /// Moderate rain, reduced visibility, bumps.
    Rain,
    /// Storm: heavy rain/hail, strong gusts, updrafts, sheared flow.
    Storm,
    /// Worst-case stress test: extreme gusts, IFR visibility, hail.
    ExtremeStorm,
}

impl WeatherPreset {
    pub fn name(self) -> &'static str {
        match self {
            WeatherPreset::Clear => "clear",
            WeatherPreset::LightTurb => "light_turb",
            WeatherPreset::Rain => "rain",
            WeatherPreset::Storm => "storm",
            WeatherPreset::ExtremeStorm => "extreme_storm",
        }
    }

    pub fn severity(self) -> WeatherSeverity {
        match self {
            WeatherPreset::Clear => WeatherSeverity::Clear,
            WeatherPreset::LightTurb => WeatherSeverity::Light,
            WeatherPreset::Rain => WeatherSeverity::Moderate,
            WeatherPreset::Storm | WeatherPreset::ExtremeStorm => WeatherSeverity::Severe,
        }
    }
}

/// Curriculum phase. `PHASE_1_CLEAR` … `PHASE_4_STORM`, climbing severity.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
#[repr(i32)]
pub enum CurriculumPhase {
    Phase1Clear = 0,
    Phase2LightTurb = 1,
    Phase3Rain = 2,
    Phase4Storm = 3,
}

impl CurriculumPhase {
    pub fn index(self) -> i32 {
        self as i32
    }

    pub fn from_index(i: i32) -> Self {
        match i {
            1 => CurriculumPhase::Phase2LightTurb,
            2 => CurriculumPhase::Phase3Rain,
            3 => CurriculumPhase::Phase4Storm,
            _ => CurriculumPhase::Phase1Clear,
        }
    }

    pub fn name(self) -> &'static str {
        match self {
            CurriculumPhase::Phase1Clear => "phase_1_clear",
            CurriculumPhase::Phase2LightTurb => "phase_2_light_turb",
            CurriculumPhase::Phase3Rain => "phase_3_rain",
            CurriculumPhase::Phase4Storm => "phase_4_storm",
        }
    }

    /// The [`WeatherPreset`] this phase flies.
    pub fn preset(self) -> WeatherPreset {
        match self {
            CurriculumPhase::Phase1Clear => WeatherPreset::Clear,
            CurriculumPhase::Phase2LightTurb => WeatherPreset::LightTurb,
            CurriculumPhase::Phase3Rain => WeatherPreset::Rain,
            CurriculumPhase::Phase4Storm => WeatherPreset::Storm,
        }
    }
}

/// Why the episode terminated on weather grounds. `None` = weather is benign.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(i32)]
pub enum TerminationReason {
    None = 0,
    /// Instantaneous combined gust exceeded `gust_termination_limit` m/s.
    GustLimit = 1,
    /// Visibility sat below `min_visibility` for longer than `visibility_timeout`.
    VisibilityLimit = 2,
    /// The agent remained in Severe weather longer than `severe_exposure_limit`.
    SevereExposure = 3,
}

impl TerminationReason {
    pub fn index(self) -> i32 {
        self as i32
    }

    pub fn name(self) -> &'static str {
        match self {
            TerminationReason::None => "none",
            TerminationReason::GustLimit => "gust_limit",
            TerminationReason::VisibilityLimit => "visibility_limit",
            TerminationReason::SevereExposure => "severe_exposure",
        }
    }
}

// ---------------------------------------------------------------------------
// Config
// ---------------------------------------------------------------------------

/// Configuration of the weather layer. Every parameter the agent can observe
/// (or the trainer wants to control) is here; all numeric fields are SI.
#[derive(Debug, Clone)]
pub struct WeatherConfig {
    /// PRNG seed — same seed ⇒ byte-identical weather (RL reproducibility).
    /// Changing it re-rolls the scene sequence and gust stream.
    pub seed: u64,
    /// `true`: weather follows a **fixed pattern** for the whole episode
    /// (no auto scene changes); same seed = same weather every restart.
    /// `false`: re-rolls a new scene every `auto_change_interval` seconds.
    pub deterministic: bool,
    /// Seconds between weather scene changes in stochastic mode.
    pub auto_change_interval: f64,
    /// Overall severity band the scenes sample within.
    pub severity: WeatherSeverity,
    /// Steady wind speed (m/s) at `reference_altitude`.
    pub mean_wind_speed: f64,
    /// True bearing (deg, north=0, clockwise) the wind blows **toward**.
    pub wind_direction_deg: f64,
    /// Enable boundary-layer power-law shear (wind grows near the ground).
    pub wind_shear: bool,
    /// Altitude (m) at which `mean_wind_speed` is exact.
    pub reference_altitude: f64,
    /// Turbulence band; the Dryden gust RMS is scaled from this band and
    /// modulated by `gust_magnitude` / `gust_frequency`.
    pub turbulence: TurbulenceLevel,
    /// Integral length scales `(longitudinal, lateral, vertical)` in metres —
    /// the two-axis tuple the agent observes for gust decorrelation.
    pub length_scales: Vector3<f64>,
    /// Gust envelope frequency (Hz): how fast the gust RMS walks (0.2 ≈ one
    /// gust every 5 s). Drives the Ornstein–Uhlenbeck envelope.
    pub gust_frequency: f64,
    /// Peak gust magnitude (m/s) the envelope modulates to (~3·σ_u).
    pub gust_magnitude: f64,
    /// Target rain intensity `[0, 1]` for the active scene.
    pub rain_intensity: f64,
    /// Target visibility (m) for the active scene.
    pub visibility: f64,
    /// Target updraft (m/s, positive = up; may be negative = downdraft).
    pub updraft_strength: f64,
    /// Target precipitation for the active scene.
    pub precipitation: PrecipitationType,
    /// Seconds taken to ramp rain/visibility/updraft toward the scene target.
    pub transition_time: f64,

    // -- Reward shaping weights (all ≥ 0; each penalty ≤ 0) --
    /// Coefficient of the severity/procipitation weather-penetration penalty.
    pub w_weather: f64,
    /// Coefficient of the gust-RMS turbulence penalty.
    pub w_turbulence: f64,
    /// Coefficient of the reduced-visibility penalty.
    pub w_visibility: f64,
    /// Coefficient of the steady-wind (crosswind/headwind loading) penalty.
    pub w_wind: f64,

    // -- Extreme-weather termination --
    /// Master switch for the weather-based episode `done` signals.
    pub extreme_weather_termination: bool,
    /// Instantaneous 3-axis gust magnitude (m/s) above which the episode ends.
    pub gust_termination_limit: f64,
    /// Visibility (m) below which exposure begins counting toward termination.
    pub min_visibility: f64,
    /// Seconds of continuous sub-`min_visibility` flight before termination.
    pub visibility_timeout: f64,
    /// Seconds of continuous Severe weather before termination.
    pub severe_exposure_limit: f64,

    // -- Evasive maneuver (agent action) --
    /// How long (s) a `trigger_evasive_maneuver()` command mitigates exposure.
    pub evasive_duration: f64,
    /// Gust-RMS multiplier while an evasive maneuver is active (e.g. 0.4 =
    /// the maneuver finds 60 % smoother air).
    pub evasive_sigma_factor: f64,
}

impl Default for WeatherConfig {
    fn default() -> Self {
        WeatherConfig::from_preset(WeatherPreset::Clear, 0)
    }
}

impl WeatherConfig {
    /// Build a config from a named profile (severity, wind, rain, visibility,
    /// turbulence, updraft and precipitation are all consistent).
    pub fn from_preset(preset: WeatherPreset, seed: u64) -> Self {
        let (severity, wind, rain, visibility, turb, gust, updraft, precip, shear) = match preset {
            WeatherPreset::Clear => (
                WeatherSeverity::Clear,
                2.0,
                0.0,
                20_000.0,
                TurbulenceLevel::Light,
                4.5,
                0.0,
                PrecipitationType::None,
                false,
            ),
            WeatherPreset::LightTurb => (
                WeatherSeverity::Light,
                5.0,
                0.05,
                12_000.0,
                TurbulenceLevel::Light,
                7.5,
                1.0,
                PrecipitationType::None,
                false,
            ),
            WeatherPreset::Rain => (
                WeatherSeverity::Moderate,
                10.0,
                0.5,
                5_000.0,
                TurbulenceLevel::Moderate,
                13.5,
                2.5,
                PrecipitationType::Rain,
                true,
            ),
            WeatherPreset::Storm => (
                WeatherSeverity::Severe,
                18.0,
                0.8,
                2_500.0,
                TurbulenceLevel::Severe,
                21.0,
                5.0,
                PrecipitationType::Hail,
                true,
            ),
            WeatherPreset::ExtremeStorm => (
                WeatherSeverity::Severe,
                25.0,
                1.0,
                1_200.0,
                TurbulenceLevel::Extreme,
                30.0,
                8.0,
                PrecipitationType::Hail,
                true,
            ),
        };
        Self {
            seed,
            deterministic: false,
            auto_change_interval: DEFAULT_AUTO_CHANGE_INTERVAL,
            severity,
            mean_wind_speed: wind,
            wind_direction_deg: 0.0,
            wind_shear: shear,
            reference_altitude: 800.0,
            turbulence: turb,
            length_scales: Vector3::new(533.0, 533.0, 266.5),
            gust_frequency: 0.2,
            gust_magnitude: gust,
            rain_intensity: rain,
            visibility,
            updraft_strength: updraft,
            precipitation: precip,
            transition_time: 10.0,
            w_weather: 1.0,
            w_turbulence: 2.0,
            w_visibility: 1.5,
            w_wind: 0.5,
            extreme_weather_termination: false,
            gust_termination_limit: 35.0,
            min_visibility: 500.0,
            visibility_timeout: 30.0,
            severe_exposure_limit: 120.0,
            evasive_duration: 6.0,
            evasive_sigma_factor: 0.4,
        }
    }

    /// Builder: override the seed (re-rolls everything).
    pub fn with_seed(mut self, seed: u64) -> Self {
        self.seed = seed;
        self
    }

    /// Builder: switch deterministic / stochastic recipe.
    pub fn with_deterministic(mut self, deterministic: bool) -> Self {
        self.deterministic = deterministic;
        self
    }

    /// Builder: set the auto-change interval (seconds).
    pub fn with_auto_change(mut self, seconds: f64) -> Self {
        self.auto_change_interval = seconds.max(0.0);
        self
    }

    /// Builder: override the overall severity band.
    pub fn with_severity(mut self, severity: WeatherSeverity) -> Self {
        self.severity = severity;
        self
    }
}

// ---------------------------------------------------------------------------
// Observation
// ---------------------------------------------------------------------------

/// Snapshot of the weather the RL agent observes on one step. All vectors in
/// the Earth NED frame; `visibility` in metres; `updraft_strength` positive =
/// up. The identical data is packed into the flat
/// [`WEATHER_OBS_DIM`]-channel array [`WeatherSystem::observation_array`].
#[derive(Debug, Clone)]
pub struct WeatherObservation {
    /// Total wind (steady + gusts + agent offset) in the NED frame (m/s).
    pub wind: Vector3<f64>,
    /// Current turbulence band index (`TurbulenceLevel::Light = 0` … `Extreme = 3`).
    pub turbulence_level: i32,
    /// Effective gust RMS σ_u (m/s).
    pub gust_rms: f64,
    /// Rain intensity `[0, 1]`.
    pub rain_intensity: f64,
    /// Visibility distance (m).
    pub visibility: f64,
    /// Wind-shear gradient magnitude `|dW/dh|` in `(m/s)/m`.
    pub shear_gradient: f64,
    /// Updraft strength (m/s, positive = up, negative = downdraft).
    pub updraft_strength: f64,
    /// Current precipitation type.
    pub precipitation: PrecipitationType,
    /// Current overall severity.
    pub severity: WeatherSeverity,
    /// Steady wind speed of the active scene (m/s).
    pub wind_speed: f64,
    /// Simulation time elapsed in this episode (s).
    pub timestamp: f64,
}

/// Running min/max normaliser for the weather observation vector. Keeps
/// per-channel `[min, max]` so the agent input lands in `[0, 1]`; seeded with
/// documented physical bounds and updated online from observed values.
#[derive(Debug, Clone)]
pub struct ObservationNormalizer {
    min: [f64; WEATHER_OBS_DIM],
    max: [f64; WEATHER_OBS_DIM],
}

/// Physical default observation bounds (wind ±30 m/s, updraft ±20 m/s).
pub const DEFAULT_NORMALIZER_MIN: [f64; WEATHER_OBS_DIM] = [
    -30.0, -30.0, -30.0, 0.0, 0.0, 0.0, 0.0, -20.0, 0.0, 0.0, 0.0, 0.0,
];
/// Physical default observation bounds (upper).
pub const DEFAULT_NORMALIZER_MAX: [f64; WEATHER_OBS_DIM] = [
    30.0, 30.0, 30.0, 1.0, 1.0, 1.0, 1.0, 20.0, 1.0, 1.0, 1.0, 1.0,
];

impl ObservationNormalizer {
    /// Start from the documented physical bounds in
    /// [`DEFAULT_NORMALIZER_MIN`] / [`DEFAULT_NORMALIZER_MAX`].
    pub fn from_default_bounds() -> Self {
        Self {
            min: DEFAULT_NORMALIZER_MIN,
            max: DEFAULT_NORMALIZER_MAX,
        }
    }

    /// Start from a compact `[[min0, max0], …]` bound table.
    pub fn from_bounds(min: [f64; WEATHER_OBS_DIM], max: [f64; WEATHER_OBS_DIM]) -> Self {
        Self { min, max }
    }

    /// Fold one observation into the running min/max.
    pub fn update(&mut self, obs: &[f64; WEATHER_OBS_DIM]) {
        for i in 0..WEATHER_OBS_DIM {
            self.min[i] = self.min[i].min(obs[i]);
            self.max[i] = self.max[i].max(obs[i]);
        }
    }

    /// Normalise one observation to `[0, 1]` per channel (one-hot / already
    /// unit channels pass through unchanged).
    pub fn normalize(&self, obs: &[f64; WEATHER_OBS_DIM]) -> [f64; WEATHER_OBS_DIM] {
        let mut out = [0.0; WEATHER_OBS_DIM];
        for i in 0..WEATHER_OBS_DIM {
            let lo = self.min[i];
            let hi = self.max[i];
            out[i] = if (hi - lo).abs() < 1e-9 {
                0.0
            } else {
                ((obs[i] - lo) / (hi - lo)).clamp(0.0, 1.0)
            };
        }
        out
    }

    pub fn min(&self) -> &[f64; WEATHER_OBS_DIM] {
        &self.min
    }

    pub fn max(&self) -> &[f64; WEATHER_OBS_DIM] {
        &self.max
    }

    pub fn reset(&mut self) {
        *self = Self::from_default_bounds();
    }
}

// ---------------------------------------------------------------------------
// Curriculum
// ---------------------------------------------------------------------------

/// Curriculum-learning schedule: climb clear → light turbulence → rain →
/// storm, switching phase automatically every `steps_per_phase` environment
/// steps. `config(seed)` gives the [`WeatherConfig`] for the current phase.
#[derive(Debug, Clone)]
pub struct WeatherCurriculum {
    /// Environment steps per phase.
    pub steps_per_phase: usize,
    /// Total steps consumed by the curriculum so far.
    pub step_count: usize,
    /// Active phase.
    pub current: CurriculumPhase,
    /// True once the final (storm) phase has been entered-and-completed.
    pub done: bool,
}

impl WeatherCurriculum {
    pub fn new(steps_per_phase: usize) -> Self {
        Self {
            steps_per_phase: steps_per_phase.max(1),
            step_count: 0,
            current: CurriculumPhase::Phase1Clear,
            done: false,
        }
    }

    /// Increment the step counter; auto-advance the phase when the per-phase
    /// budget is exhausted. Returns `Some(phase)` when the phase changed.
    pub fn step(&mut self) -> Option<CurriculumPhase> {
        self.step_count += 1;
        self.advance_auto()
    }

    /// Advance `n` steps at once (e.g. at the end of an episode).
    pub fn advance(&mut self, n: usize) -> Option<CurriculumPhase> {
        self.step_count = self.step_count.saturating_add(n);
        self.advance_auto()
    }

    fn advance_auto(&mut self) -> Option<CurriculumPhase> {
        if self.done {
            return None;
        }
        let phase_idx = (self.step_count / self.steps_per_phase) as i32;
        if phase_idx <= self.current.index() {
            return None;
        }
        let next = CurriculumPhase::from_index(phase_idx);
        self.current = next;
        if next == CurriculumPhase::Phase4Storm {
            self.done = true;
        }
        Some(next)
    }

    /// Explicitly move to the next phase (used for manual curricula or the
    /// Godot bridge `next_phase()`). Returns the new phase.
    pub fn next_phase(&mut self) -> CurriculumPhase {
        let idx = (self.current.index() + 1).min(CurriculumPhase::Phase4Storm.index());
        self.current = CurriculumPhase::from_index(idx);
        if self.current == CurriculumPhase::Phase4Storm {
            self.done = true;
        }
        self.current
    }

    pub fn phase(&self) -> CurriculumPhase {
        self.current
    }

    pub fn is_complete(&self) -> bool {
        self.done && self.current == CurriculumPhase::Phase4Storm
    }

    /// Progress through the curriculum as a fraction `[0, 1]`.
    pub fn progress(&self) -> f64 {
        let total = 4.0 * self.steps_per_phase as f64;
        (self.step_count as f64 / total.max(1.0)).clamp(0.0, 1.0)
    }

    /// A fresh [`WeatherConfig`] for the currently active phase (seeded for
    /// reproducibility of each phase's episodes).
    pub fn config(&self, seed: u64) -> WeatherConfig {
        WeatherConfig::from_preset(self.current.preset(), seed)
    }
}

// ---------------------------------------------------------------------------
// Weather system
// ---------------------------------------------------------------------------

/// Internal target of the active weather scene (what the smoothed fields ramp
/// toward).
#[derive(Debug, Clone)]
struct SceneParams {
    wind_speed: f64,
    wind_dir_deg: f64,
    rain: f64,
    visibility: f64,
    updraft: f64,
    turbulence: TurbulenceLevel,
    gust_magnitude: f64,
    precipitation: PrecipitationType,
    severity: WeatherSeverity,
}

impl SceneParams {
    fn from_preset(preset: WeatherPreset) -> Self {
        let cfg = WeatherConfig::from_preset(preset, 0);
        Self {
            wind_speed: cfg.mean_wind_speed,
            wind_dir_deg: cfg.wind_direction_deg,
            rain: cfg.rain_intensity,
            visibility: cfg.visibility,
            updraft: cfg.updraft_strength,
            turbulence: cfg.turbulence,
            gust_magnitude: cfg.gust_magnitude,
            precipitation: cfg.precipitation,
            severity: cfg.severity,
        }
    }

    /// Nominal (mid-band) scene for the configured severity — used by the
    /// deterministic recipe and as the base of stochastic re-rolls.
    fn from_severity(severity: WeatherSeverity) -> Self {
        let preset = match severity {
            WeatherSeverity::Clear => WeatherPreset::Clear,
            WeatherSeverity::Light => WeatherPreset::LightTurb,
            WeatherSeverity::Moderate => WeatherPreset::Rain,
            WeatherSeverity::Severe => WeatherPreset::Storm,
        };
        Self::from_preset(preset)
    }
}

/// Climbs clear → light → moderate → storm. The stochastic scene re-roll path:
/// keeps the severity band fixed but perturbs each parameter with a seeded
/// draw, so a seed reproduces the exact sequence of scenes.
fn randomise_scene(base: &SceneParams, rng: &mut StdRng) -> SceneParams {
    let mut jitter = |spread: f64| -> f64 { spread * (2.0 * rng.gen::<f64>() - 1.0) };
    let mut s = base.clone();
    s.wind_speed = (s.wind_speed + jitter(2.0)).max(0.0);
    s.wind_dir_deg = (s.wind_dir_deg + jitter(40.0)).rem_euclid(360.0);
    s.rain = (s.rain + jitter(0.15)).clamp(0.0, 1.0);
    s.visibility = (s.visibility + jitter(1500.0)).max(200.0);
    s.updraft = s.updraft + jitter(1.5);
    s.precipitation = match base.precipitation {
        PrecipitationType::None => PrecipitationType::None,
        _ => {
            let roll = rng.gen::<f64>();
            if roll < 0.15 {
                PrecipitationType::Snow
            } else if roll < 0.25 {
                PrecipitationType::Hail
            } else {
                PrecipitationType::Rain
            }
        }
    };
    s
}

/// The weather state machine + wind/penalty source for RL.
#[derive(Debug)]
pub struct WeatherSystem {
    config: WeatherConfig,
    /// Embedded Dryden turbulence + steady wind (seeded, see [`crate::wind`]).
    wind: WindEnvironment,
    /// Seedable PRNG driving scene re-rolls and the gust envelope.
    rng: StdRng,
    /// Active scene targets.
    scene: SceneParams,
    scene_index: u64,
    time_in_scene: f64,
    sim_time: f64,

    // Smoothed / current values.
    rain_current: f64,
    visibility_current: f64,
    updraft_current: f64,
    envelope: f64,
    sigma_effective: f64,
    turb_level: TurbulenceLevel,
    severity: WeatherSeverity,
    precipitation: PrecipitationType,

    // Wind state.
    last_wind: Vector3<f64>,
    last_gust: Vector3<f64>,
    last_altitude: f64,
    /// Additive wind bias the agent can command via `modify_wind_vector`.
    wind_offset: Vector3<f64>,

    // Exposure / termination.
    severe_elapsed: f64,
    visibility_violation_elapsed: f64,
    terminated: TerminationReason,

    // Evasive maneuver state.
    evasive_timer: f64,

    // Curriculum (optional).
    curriculum: Option<WeatherCurriculum>,
    curriculum_seed: u64,
}

impl WeatherSystem {
    /// Build a weather system from a config. The scene, PRNG stream and
    /// embedded Dryden gusts are all seeded — two systems with the same
    /// [`WeatherConfig`] behave identically step-for-step.
    pub fn new(config: WeatherConfig) -> Self {
        let base = SceneParams::from_severity(config.severity);
        let scene = Self::initial_scene(&config, &base);
        let scene_copy = scene.clone();
        let (sigma0, level0) = Self::scene_sigma(&config, &scene, 1.0);
        Self {
            config: config.clone(),
            wind: Self::build_wind(&config, &scene, config.seed),
            rng: StdRng::seed_from_u64(config.seed),
            scene,
            scene_index: 0,
            time_in_scene: 0.0,
            sim_time: 0.0,
            rain_current: scene_copy.rain,
            visibility_current: scene_copy.visibility,
            updraft_current: scene_copy.updraft,
            envelope: 1.0,
            sigma_effective: sigma0,
            turb_level: level0,
            severity: scene_copy.severity,
            precipitation: scene_copy.precipitation,
            last_wind: Vector3::zeros(),
            last_gust: Vector3::zeros(),
            last_altitude: config.reference_altitude,
            wind_offset: Vector3::zeros(),
            severe_elapsed: 0.0,
            visibility_violation_elapsed: 0.0,
            terminated: TerminationReason::None,
            evasive_timer: 0.0,
            curriculum: None,
            curriculum_seed: config.seed,
        }
    }

    /// The episode-start scene: the severity band's nominal profile, or its
    /// seeded randomisation in stochastic mode (fixed draw order so `new()` and
    /// `reset()` coincide).
    fn initial_scene(config: &WeatherConfig, base: &SceneParams) -> SceneParams {
        if config.deterministic {
            return base.clone();
        }
        let mut rng = StdRng::seed_from_u64(config.seed);
        let mut s = randomise_scene(base, &mut rng);
        // Keep a precipitation band for Moderate/Severe nominal scenes.
        if s.precipitation == PrecipitationType::None
            && matches!(
                config.severity,
                WeatherSeverity::Moderate | WeatherSeverity::Severe
            )
        {
            s.precipitation = base.precipitation;
        }
        s
    }

    fn build_wind(config: &WeatherConfig, scene: &SceneParams, seed: u64) -> WindEnvironment {
        let wc = WindConfig {
            wind_speed: scene.wind_speed,
            wind_direction: scene.wind_dir_deg.to_radians(),
            reference_altitude: config.reference_altitude,
            wind_shear: config.wind_shear,
            turbulence: crate::wind::TurbulenceIntensity::Custom(0.1),
            turbulence_scale: config.length_scales.x,
            turbulence_scale_lat: config.length_scales.y,
            turbulence_scale_vert: config.length_scales.z,
            seed,
        };
        WindEnvironment::new(wc)
    }

    /// Gust RMS target for a scene and envelope factor. The **band** RMS is the
    /// primary driver; the configured `gust_magnitude` (peak ≈ 3·σ) acts as a
    /// ceiling so `gust_magnitude` limits are honoured without overruling the
    /// severity band.
    fn scene_sigma(
        config: &WeatherConfig,
        scene: &SceneParams,
        envelope: f64,
    ) -> (f64, TurbulenceLevel) {
        let base = scene.turbulence.sigma_u();
        let floor = 0.3;
        let sigma = (base * envelope).max(floor);
        if config.gust_magnitude > 0.0 && scene.gust_magnitude > 0.0 {
            let ceiling = config.gust_magnitude.max(scene.gust_magnitude) / 3.0;
            let sigma = sigma.min(ceiling).max(floor);
            (sigma, TurbulenceLevel::from_sigma(sigma))
        } else {
            (sigma, TurbulenceLevel::from_sigma(sigma))
        }
    }

    // -----------------------------------------------------------------
    // Lifecycle
    // -----------------------------------------------------------------

    /// Reset to the episode start: re-seed everything, rebuild the wind field,
    /// re-initialise the scene and zero all exposure timers. Same seed ⇒ same
    /// episode weather (both recipes).
    pub fn reset(&mut self) {
        self.rng = StdRng::seed_from_u64(self.config.seed);
        let base = SceneParams::from_severity(self.config.severity);
        let scene = Self::initial_scene(&self.config, &base);
        self.scene = scene;
        self.scene_index = 0;
        self.time_in_scene = 0.0;
        self.sim_time = 0.0;
        self.wind = Self::build_wind(&self.config, &self.scene, self.config.seed);
        self.rain_current = self.scene.rain;
        self.visibility_current = self.scene.visibility;
        self.updraft_current = self.scene.updraft;
        self.envelope = 1.0;
        self.severity = self.scene.severity;
        self.config.severity = self.scene.severity;
        self.precipitation = self.scene.precipitation;
        let (sigma0, level0) = Self::scene_sigma(&self.config, &self.scene, 1.0);
        self.sigma_effective = sigma0;
        self.turb_level = level0;
        self.last_wind = Vector3::zeros();
        self.last_gust = Vector3::zeros();
        self.last_altitude = self.config.reference_altitude;
        self.wind_offset = Vector3::zeros();
        self.severe_elapsed = 0.0;
        self.visibility_violation_elapsed = 0.0;
        self.terminated = TerminationReason::None;
        self.evasive_timer = 0.0;
        if let Some(c) = &mut self.curriculum {
            c.step_count = 0;
            c.current = CurriculumPhase::Phase1Clear;
            c.done = false;
        }
    }

    /// Install a curriculum; its phases then drive the scene progression and
    /// override both the stochastic auto-change and deterministic recipes.
    pub fn with_curriculum(mut self, curriculum: WeatherCurriculum, seed: u64) -> Self {
        self.curriculum_seed = seed;
        self.curriculum = Some(curriculum);
        self.apply_preset(self.curriculum.as_ref().unwrap().phase().preset());
        self
    }

    fn apply_preset(&mut self, preset: WeatherPreset) {
        self.scene = SceneParams::from_preset(preset);
        self.severity = self.scene.severity;
        self.config.severity = self.scene.severity;
        self.precipitation = self.scene.precipitation;
        self.rain_current = self.scene.rain;
        self.visibility_current = self.scene.visibility;
        self.updraft_current = self.scene.updraft;
        self.time_in_scene = 0.0;
    }

    // -----------------------------------------------------------------
    // Core step
    // -----------------------------------------------------------------

    /// Advance the weather by `dt` seconds at the given geometric altitude and
    /// true airspeed, returning the total wind in the **NED** frame (m/s)
    /// exactly as the flight dynamics expects it. Call once per physics step
    /// *before* integrating the aircraft.
    pub fn step(&mut self, altitude_m: f64, vt_air: f64, dt: f64) -> Vector3<f64> {
        let dt = dt.max(1e-4);
        self.sim_time += dt;
        self.time_in_scene += dt;
        self.last_altitude = altitude_m;

        // Curriculum drives the scene even in deterministic mode.
        if let Some(c) = &mut self.curriculum {
            if let Some(new_phase) = c.step() {
                self.apply_preset(new_phase.preset());
                self.scene_index += 1;
            }
        } else if !self.config.deterministic
            && self.config.auto_change_interval > 0.0
            && self.time_in_scene >= self.config.auto_change_interval
        {
            self.scene_index += 1;
            self.time_in_scene = 0.0;
            self.scene = randomise_scene(&self.scene, &mut self.rng);
            if self.scene.precipitation == PrecipitationType::None
                && matches!(
                    self.scene.severity,
                    WeatherSeverity::Moderate | WeatherSeverity::Severe
                )
            {
                self.scene.precipitation = PrecipitationType::Rain;
            }
            self.severity = self.scene.severity;
        }

        // First-order ramp toward the scene targets (perceptual smoothness).
        let tau = self.config.transition_time.max(0.5);
        let a = 1.0 - (-dt / tau).exp();
        self.rain_current += a * (self.scene.rain - self.rain_current);
        self.visibility_current += a * (self.scene.visibility - self.visibility_current);
        self.updraft_current += a * (self.scene.updraft - self.updraft_current);
        self.precipitation = self.scene.precipitation;

        // Gust envelope: OU random walk at gust_frequency, or a fixed-time
        // deterministic oscillation in the deterministic recipe.
        self.envelope = self.update_envelope(dt);

        // Effective gust RMS → Dryden filter target.
        let (sigma, level) = Self::scene_sigma(&self.config, &self.scene, self.envelope);
        let evasive = self.evasive_timer > 0.0;
        let sigma = if evasive {
            sigma * self.config.evasive_sigma_factor
        } else {
            sigma
        };
        self.sigma_effective = sigma;
        self.turb_level = if evasive { self.turb_level } else { level };
        self.wind.set_sigma(sigma);

        // Push the scene into the embedded wind field.
        self.wind.config.wind_speed = self.scene.wind_speed;
        self.wind.config.wind_direction = self.scene.wind_dir_deg.to_radians();
        self.wind.config.wind_shear = self.config.wind_shear;
        self.wind.config.reference_altitude = self.config.reference_altitude;
        self.wind.config.turbulence_scale = self.config.length_scales.x;
        self.wind.config.turbulence_scale_lat = self.config.length_scales.y;
        self.wind.config.turbulence_scale_vert = self.config.length_scales.z;

        // Total wind: steady shear wind + Dryden gusts + agent offset +
        // scene updraft (upward ≡ negative NED down).
        let steady = self.wind.steady_wind(altitude_m);
        let gust = self.wind.turbulence(vt_air, dt);
        self.last_gust = gust;
        let mut total = steady + gust + self.wind_offset;
        total.z -= self.updraft_current;
        self.last_wind = total;

        // Exposure tracking.
        if self.severity >= WeatherSeverity::Severe {
            self.severe_elapsed += dt;
        } else {
            self.severe_elapsed = 0.0;
        }
        if self.visibility_current < self.config.min_visibility {
            self.visibility_violation_elapsed += dt;
        } else {
            self.visibility_violation_elapsed = 0.0;
        }
        if self.evasive_timer > 0.0 {
            self.evasive_timer -= dt;
        }

        self.terminated = self.evaluate_termination();
        total
    }

    /// Advance the gust envelope one step, returning the new envelope. The
    /// deterministic recipe uses a closed-form sine (no RNG draws, so the same
    /// seed reproduces the exact envelope); the stochastic recipe uses an
    /// Ornstein–Uhlenbeck walk driven by the seeded PRNG.
    fn update_envelope(&mut self, dt: f64) -> f64 {
        let freq = self.config.gust_frequency.max(1e-3);
        if self.config.deterministic {
            let phase = self.config.seed as f64 * 0.000_123_456_789;
            let env = 1.0 + 0.35 * (std::f64::consts::TAU * freq * self.sim_time + phase).sin();
            env.clamp(0.5, 1.35)
        } else {
            let mean = 1.0;
            let rate = freq;
            let sig = 0.12;
            let noise: f64 = StandardNormal.sample(&mut self.rng);
            let next = self.envelope + rate * (mean - self.envelope) * dt + sig * dt.sqrt() * noise;
            next.clamp(0.5, 1.5)
        }
    }

    /// Evaluate extreme-weather termination against the configured limits.
    fn evaluate_termination(&self) -> TerminationReason {
        if !self.config.extreme_weather_termination {
            return TerminationReason::None;
        }
        if self.last_gust.norm() > self.config.gust_termination_limit {
            return TerminationReason::GustLimit;
        }
        if self.visibility_current < self.config.min_visibility
            && self.visibility_violation_elapsed > self.config.visibility_timeout
        {
            return TerminationReason::VisibilityLimit;
        }
        if self.severe_elapsed > self.config.severe_exposure_limit {
            return TerminationReason::SevereExposure;
        }
        TerminationReason::None
    }

    // -----------------------------------------------------------------
    // RL interface: observations
    // -----------------------------------------------------------------

    /// Full structured observation snapshot (see [`WeatherObservation`]).
    pub fn observation(&self) -> WeatherObservation {
        let shear = self.shear_gradient(self.last_altitude);
        WeatherObservation {
            wind: self.last_wind,
            turbulence_level: self.turb_level.index(),
            gust_rms: self.sigma_effective,
            rain_intensity: self.rain_current,
            visibility: self.visibility_current,
            shear_gradient: shear,
            updraft_strength: self.updraft_current,
            precipitation: self.precipitation,
            severity: self.severity,
            wind_speed: self.scene.wind_speed,
            timestamp: self.sim_time,
        }
    }

    /// Flat `[WEATHER_OBS_DIM]` observation vector `f64` (channel layout on
    /// [`WEATHER_OBS_DIM`]). This is the array an RL library consumes directly.
    pub fn observation_array(&self) -> [f64; WEATHER_OBS_DIM] {
        let o = self.observation();
        let [r, s, h] = o.precipitation.one_hot();
        [
            o.wind.x,
            o.wind.y,
            o.wind.z,
            (o.gust_rms / REFERENCE_TURBULENCE_SIGMA).clamp(0.0, 1.0),
            o.rain_intensity.clamp(0.0, 1.0),
            (o.visibility / REFERENCE_VISIBILITY).clamp(0.0, 1.0),
            (o.shear_gradient.abs() / REFERENCE_SHEAR_GRADIENT).clamp(0.0, 1.0),
            o.updraft_strength,
            r,
            s,
            h,
            o.severity.normalized(),
        ]
    }

    /// Observation as `Vec<f32>` — the typical RL-library input format.
    pub fn observation_vec_f32(&self) -> Vec<f32> {
        self.observation_array().iter().map(|&v| v as f32).collect()
    }

    /// Observation as an `nalgebra::DVector<f32>` (tch-rs / custom PPO-A2C).
    pub fn to_dvector(&self) -> DVector<f32> {
        DVector::from(self.observation_vec_f32())
    }

    // -----------------------------------------------------------------
    // RL interface: rewards
    // -----------------------------------------------------------------

    /// Negative penalty for penetrating the current weather severity band
    /// (`-w_weather · severity_01`).
    pub fn weather_penalty(&self) -> f64 {
        -self.config.w_weather * self.severity.normalized()
    }

    /// Negative penalty proportional to the squared gust RMS (turbulence).
    pub fn turbulence_penalty(&self) -> f64 {
        let r = (self.sigma_effective / REFERENCE_TURBULENCE_SIGMA).clamp(0.0, 1.0);
        -self.config.w_turbulence * r * r
    }

    /// Negative penalty growing with lost visibility
    /// (`-w_visibility · (1 − visibility_01)²`).
    pub fn visibility_penalty(&self) -> f64 {
        let v = (self.visibility_current / REFERENCE_VISIBILITY).clamp(0.0, 1.0);
        -self.config.w_visibility * (1.0 - v) * (1.0 - v)
    }

    /// Negative penalty for strong steady wind loading (crosswind/headwind).
    pub fn wind_penalty(&self) -> f64 {
        let ref_speed = 20.0;
        let r = (self.scene.wind_speed / ref_speed).clamp(0.0, 1.0);
        -self.config.w_wind * r * r
    }

    /// Sum of all weather penalties for this step (≤ 0). Add to the flight
    /// reward. Halved while an evasive maneuver is active.
    pub fn reward_penalty_total(&self) -> f64 {
        let mut p = self.weather_penalty();
        p -= self.config.w_weather * 0.5 * self.rain_current; // precipitation exposure
        p += self.turbulence_penalty() + self.visibility_penalty() + self.wind_penalty();
        if self.evasive_timer > 0.0 {
            p *= 0.5;
        }
        p
    }

    /// Large negative bonus applied when the episode terminates on extreme
    /// weather (weather analogue of the crash penalty).
    pub fn weather_crash_penalty(&self) -> f64 {
        -4.0 * self.config.w_weather
    }

    // -----------------------------------------------------------------
    // RL interface: actions
    // -----------------------------------------------------------------

    /// Force the weather to the given severity band (scene targets snap to the
    /// band's nominal profile). Used by agents/environments that modulate
    /// weather severity (adversarial scenarios, robustness sweeps).
    pub fn set_severity(&mut self, severity: WeatherSeverity) {
        self.apply_preset(match severity {
            WeatherSeverity::Clear => WeatherPreset::Clear,
            WeatherSeverity::Light => WeatherPreset::LightTurb,
            WeatherSeverity::Moderate => WeatherPreset::Rain,
            WeatherSeverity::Severe => WeatherPreset::Storm,
        });
    }

    /// Apply an additive wind perturbation (m/s, NED) — the agent's
    /// `modify_wind_vector(delta)` action. Clamped to ±30 m/s per axis.
    pub fn modify_wind_vector(&mut self, delta: Vector3<f64>) {
        const MAX_OFFSET: f64 = 30.0;
        let clamp = |v: f64| v.clamp(-MAX_OFFSET, MAX_OFFSET);
        self.wind_offset = Vector3::new(
            clamp(self.wind_offset.x + delta.x),
            clamp(self.wind_offset.y + delta.y),
            clamp(self.wind_offset.z + delta.z),
        );
    }

    /// Reset the agent wind bias to zero.
    pub fn clear_wind_offset(&mut self) {
        self.wind_offset = Vector3::zeros();
    }

    /// The agent's evasive-maneuver action: temporarily (for
    /// [`WeatherConfig::evasive_duration`]) find smoother air — gust RMS and
    /// the weather penalty are reduced by the configured factor.
    pub fn trigger_evasive_maneuver(&mut self) {
        self.evasive_timer = self.config.evasive_duration;
    }

    pub fn is_evasive(&self) -> bool {
        self.evasive_timer > 0.0
    }

    // -----------------------------------------------------------------
    // RL interface: termination & curriculum
    // -----------------------------------------------------------------

    /// Current weather termination reason ([`TerminationReason::None`] when the
    /// weather is benign).
    pub fn termination(&self) -> TerminationReason {
        self.terminated
    }

    /// True once a termination reason is active.
    pub fn is_terminated(&self) -> bool {
        self.terminated != TerminationReason::None
    }

    // -----------------------------------------------------------------
    // Getters (bridge / diagnostics)
    // -----------------------------------------------------------------

    pub fn config(&self) -> &WeatherConfig {
        &self.config
    }

    pub fn severity(&self) -> WeatherSeverity {
        self.severity
    }

    pub fn turbulence_level(&self) -> TurbulenceLevel {
        self.turb_level
    }

    pub fn rain_intensity(&self) -> f64 {
        self.rain_current
    }

    pub fn visibility(&self) -> f64 {
        self.visibility_current
    }

    pub fn updraft_strength(&self) -> f64 {
        self.updraft_current
    }

    pub fn precipitation(&self) -> PrecipitationType {
        self.precipitation
    }

    pub fn gust_rms(&self) -> f64 {
        self.sigma_effective
    }

    pub fn total_wind(&self) -> Vector3<f64> {
        self.last_wind
    }

    pub fn gust_vector(&self) -> Vector3<f64> {
        self.last_gust
    }

    pub fn wind_offset(&self) -> Vector3<f64> {
        self.wind_offset
    }

    pub fn sim_time(&self) -> f64 {
        self.sim_time
    }

    pub fn scene_index(&self) -> u64 {
        self.scene_index
    }

    /// Wind-shear gradient magnitude `|dW/dh|` in `(m/s)/m` at `altitude_m`
    /// (finite-difference of the embedded steady field). Zero when shearing
    /// is disabled.
    pub fn shear_gradient(&self, altitude_m: f64) -> f64 {
        self.shear_gradient_vector(altitude_m).norm()
    }

    /// Wind-shear gradient as a `Vector3` — the rate of change of each NED
    /// wind component with altitude `(m/s)/m`.
    pub fn shear_gradient_vector(&self, altitude_m: f64) -> Vector3<f64> {
        if !self.config.wind_shear {
            return Vector3::zeros();
        }
        let dh = 5.0;
        let lo = self.wind.steady_wind(altitude_m - dh);
        let hi = self.wind.steady_wind(altitude_m + dh);
        (hi - lo) / (2.0 * dh)
    }

    // -----------------------------------------------------------------
    // Seed / config control (bridge)
    // -----------------------------------------------------------------

    /// Change the seed and restart the episode. Same seed ⇒ same weather.
    pub fn set_seed(&mut self, seed: u64) {
        self.config.seed = seed;
        self.reset();
    }

    /// Switch deterministic/stochastic recipe and restart the episode.
    /// Stochastic keeps the same seed stream for reproducibility.
    pub fn set_deterministic(&mut self, deterministic: bool) {
        self.config.deterministic = deterministic;
        self.reset();
    }

    pub fn set_auto_change_interval(&mut self, seconds: f64) {
        self.config.auto_change_interval = seconds.max(0.0);
    }

    // -----------------------------------------------------------------
    // Curriculum bridge
    // -----------------------------------------------------------------

    /// Explicitly advance to the next curriculum phase; returns the new phase
    /// index (0-based, 0 = Phase 1 · Clear).
    pub fn next_curriculum_phase(&mut self) -> i32 {
        if let Some(c) = &mut self.curriculum {
            let next = c.next_phase();
            self.apply_preset(next.preset());
            self.scene_index += 1;
            next.index()
        } else {
            self.severity.index()
        }
    }

    pub fn curriculum_phase(&self) -> i32 {
        self.curriculum
            .as_ref()
            .map(|c| c.phase().index())
            .unwrap_or(0)
    }

    pub fn is_curriculum_complete(&self) -> bool {
        self.curriculum
            .as_ref()
            .map(|c| c.is_complete())
            .unwrap_or(true)
    }

    pub fn curriculum_progress(&self) -> f64 {
        self.curriculum
            .as_ref()
            .map(|c| c.progress())
            .unwrap_or(1.0)
    }
}

// ---------------------------------------------------------------------------
// Conversions for RL libraries
// ---------------------------------------------------------------------------

impl From<&WeatherObservation> for Vec<f32> {
    fn from(o: &WeatherObservation) -> Self {
        let [r, s, h] = o.precipitation.one_hot();
        vec![
            o.wind.x as f32,
            o.wind.y as f32,
            o.wind.z as f32,
            (o.gust_rms / REFERENCE_TURBULENCE_SIGMA).clamp(0.0, 1.0) as f32,
            o.rain_intensity.clamp(0.0, 1.0) as f32,
            (o.visibility / REFERENCE_VISIBILITY).clamp(0.0, 1.0) as f32,
            (o.shear_gradient.abs() / REFERENCE_SHEAR_GRADIENT).clamp(0.0, 1.0) as f32,
            o.updraft_strength as f32,
            r as f32,
            s as f32,
            h as f32,
            o.severity.normalized() as f32,
        ]
    }
}

impl From<&WeatherObservation> for DVector<f32> {
    fn from(o: &WeatherObservation) -> Self {
        DVector::from(Vec::<f32>::from(o))
    }
}

// ---------------------------------------------------------------------------
// V&V
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn sys(seed: u64) -> WeatherSystem {
        WeatherSystem::new(WeatherConfig::from_preset(WeatherPreset::Storm, seed))
    }

    fn run(seed: u64, deterministic: bool, steps: usize, dt: f64) -> Vec<Vector3<f64>> {
        let cfg = WeatherConfig::from_preset(WeatherPreset::Rain, seed)
            .with_deterministic(deterministic)
            .with_auto_change(5.0);
        let mut s = WeatherSystem::new(cfg);
        s.reset();
        let mut winds = Vec::with_capacity(steps);
        for i in 0..steps {
            let h = 1000.0 + (i as f64 % 50.0);
            winds.push(s.step(h, 60.0, dt));
        }
        winds
    }

    #[test]
    fn same_seed_same_wind_sequence() {
        // Stochastic recipe: identical seeds give identical NED wind traces.
        let a = run(1234, false, 600, 1.0 / 30.0);
        let b = run(1234, false, 600, 1.0 / 30.0);
        for (wa, wb) in a.iter().zip(b.iter()) {
            assert!((wa - wb).norm() < 1e-9, "seed reproducibility broken");
        }
        let c = run(999, false, 600, 1.0 / 30.0);
        assert!(
            (a.last().unwrap() - c.last().unwrap()).norm() > 1e-6,
            "different seeds should diverge"
        );
    }

    #[test]
    fn deterministic_mode_fixed_pattern_per_episode() {
        // Deterministic + same seed ⇒ byte-identical weather across resets.
        let mut s = sys(7);
        s.config.deterministic = true;
        let mut traces = Vec::new();
        for _ in 0..2 {
            s.reset();
            let mut t = Vec::new();
            for _i in 0..300 {
                t.push(s.step(900.0, 60.0, 1.0 / 30.0));
                assert_eq!(s.scene_index, 0, "deterministic must not re-scene");
            }
            traces.push(t);
        }
        for (wa, wb) in traces[0].iter().zip(traces[1].iter()) {
            assert!((wa - wb).norm() < 1e-9);
        }
    }

    #[test]
    fn stochastic_recipe_changes_scene_over_time() {
        let cfg = WeatherConfig::from_preset(WeatherPreset::Rain, 42).with_auto_change(3.0);
        let mut s = WeatherSystem::new(cfg);
        // 30 s @ 10 Hz → ~10 scene re-rolls.
        for _ in 0..300 {
            s.step(1000.0, 60.0, 0.1);
        }
        assert!(
            s.scene_index >= 8,
            "expected frequent scene changes, got {}",
            s.scene_index
        );
    }

    #[test]
    fn observation_layout_and_bounds() {
        let mut s = sys(5);
        s.reset();
        for _i in 0..60 {
            s.step(1000.0, 60.0, 1.0 / 30.0);
        }
        let arr = s.observation_array();
        assert_eq!(arr.len(), WEATHER_OBS_DIM);
        // wind channels wired.
        assert!((arr[0] - s.last_wind.x).abs() < 1e-9);
        assert!((arr[1] - s.last_wind.y).abs() < 1e-9);
        assert!((arr[2] - s.last_wind.z).abs() < 1e-9);
        // Range checks on normalised channels.
        for i in [3usize, 4, 5, 6, 8, 9, 10, 11] {
            assert!(arr[i] >= 0.0 && arr[i] <= 1.0, "channel {i} out of [0,1]");
        }
        // Storm scene: rain, precip one-hot, severity top band.
        assert!(arr[4] > 0.4, "storm scene should rain");
        assert!(arr[11] > 0.99, "storm severity should be max");
        // Structured observation matches.
        let o = s.observation();
        assert_eq!(o.timestamp, s.sim_time);
        assert!(o.visibility > 0.0);
    }

    #[test]
    fn penalties_are_negative_and_monotonic_in_severity() {
        // Same base config, stepped through all four severities.
        let mut prev = 0.0f64;
        for severity in [
            WeatherSeverity::Clear,
            WeatherSeverity::Light,
            WeatherSeverity::Moderate,
            WeatherSeverity::Severe,
        ] {
            let mut s = WeatherSystem::new(
                WeatherConfig::from_preset(WeatherPreset::Clear, 1).with_severity(severity),
            );
            s.set_severity(severity);
            for _ in 0..30 {
                s.step(1000.0, 60.0, 1.0 / 30.0);
            }
            let p = s.reward_penalty_total();
            assert!(p <= 0.0, "penalty must be ≤ 0, got {p}");
            assert!(
                p <= prev + 1e-12,
                "penalty should get more negative with severity: {p} vs {prev}"
            );
            prev = p;
        }
    }

    #[test]
    fn evasive_maneuver_cuts_gust_rms() {
        // While an evasive maneuver is active the gust RMS is multiplied by
        // evasive_sigma_factor, so the turbulence exposure must drop.
        let mut s = sys(3);
        s.config.evasive_duration = 5.0;
        s.reset();
        for _ in 0..30 {
            s.step(1000.0, 60.0, 1.0 / 30.0);
        }
        let rms_before = s.sigma_effective;
        assert!(!s.is_evasive());
        s.trigger_evasive_maneuver();
        assert!(s.is_evasive());
        for _ in 0..10 {
            s.step(1000.0, 60.0, 1.0 / 30.0);
        }
        assert!(
            s.sigma_effective < rms_before,
            "evasive maneuver must cut gust RMS ({:.2} ≥ {rms_before:.2})",
            s.sigma_effective
        );
    }

    #[test]
    fn gust_limit_termination_fires() {
        let mut s = WeatherSystem::new(WeatherConfig::from_preset(WeatherPreset::ExtremeStorm, 9));
        s.config.extreme_weather_termination = true;
        s.config.gust_termination_limit = 20.0;
        // Deterministic envelope peaks at σ_u ≈ 9·1.35 = 12.2 m/s; short
        // integral scales decorrelate the gusts quickly so the 3-axis gust
        // magnitude reliably touches the 20 m/s cap.
        s.config.gust_magnitude = 45.0;
        s.config.deterministic = true;
        s.config.length_scales = Vector3::new(80.0, 80.0, 40.0);
        s.reset();
        let mut fired = false;
        for _ in 0..3000 {
            s.step(1000.0, 60.0, 1.0 / 30.0);
            if s.termination() != TerminationReason::None {
                fired = true;
                break;
            }
        }
        assert!(
            fired,
            "gust-limit termination never fired under a 20 m/s cap"
        );
    }

    #[test]
    fn curriculum_progresses_to_storm() {
        let mut c = WeatherCurriculum::new(100);
        let mut phases = Vec::new();
        for _ in 0..400 {
            if let Some(p) = c.step() {
                phases.push(p);
            }
        }
        assert_eq!(c.current, CurriculumPhase::Phase4Storm);
        assert!(c.is_complete());
        assert!(!phases.is_empty());
        // Phase configs map to the expected presets.
        assert_eq!(
            c.config(1).severity,
            WeatherSeverity::Severe,
            "final phase must be storm severity"
        );
        assert_eq!(c.progress(), 1.0);
    }

    #[test]
    fn normalizer_scales_to_unit() {
        let mut n = ObservationNormalizer::from_default_bounds();
        let obs = [
            15.0, -15.0, 5.0, 0.5, 0.5, 0.5, 0.5, 0.0, 1.0, 0.0, 0.0, 0.5,
        ];
        n.update(&obs);
        let out = n.normalize(&obs);
        assert!(out.iter().all(|&v| (-1e-9..=1.0 + 1e-9).contains(&v)));
        assert!((out[0] - 0.75).abs() < 1e-9); // (15 - -30)/(30 - -30)
    }

    #[test]
    fn agent_actions_mutate_wind_and_reseed() {
        let mut s = sys(11);
        s.reset();
        s.modify_wind_vector(Vector3::new(3.0, 0.0, 0.0));
        let w1 = s.step(1000.0, 60.0, 1.0 / 30.0);
        // The agent bias is baked into total wind (compare without offset).
        let mut plain = sys(11);
        plain.reset();
        let w2 = plain.step(1000.0, 60.0, 1.0 / 30.0);
        assert!(
            (w1.x - w2.x).abs() > 2.9,
            "wind offset must shift the NED wind"
        );
        // Reseeding restarts the episode deterministically.
        s.set_seed(11);
        assert_eq!(s.sim_time(), 0.0, "set_seed must reset the episode clock");
    }

    #[test]
    fn wind_offset_clamped() {
        let mut s = sys(1);
        s.modify_wind_vector(Vector3::new(1e5, -1e5, 1e5));
        let o = s.wind_offset();
        assert!(o.x <= 30.0 && o.y >= -30.0 && o.z <= 30.0);
    }

    #[test]
    fn shear_vector_zero_when_disabled() {
        let mut s = sys(2);
        s.config.wind_shear = false;
        s.reset();
        let v = s.shear_gradient_vector(500.0);
        assert_eq!(v, Vector3::zeros());
    }
}
