//! # WeatherSystem — RL weather bridge for Godot
//!
//! Exposes `flight_core::weather::WeatherSystem` to GDScript as a
//! `WeatherSystem` `Node3D`. This is the Godot half of the RL weather plugin:
// close the node in any scene (or `add_child` it), call [`start`](WeatherSystem::start),
// then feed it with [`step`](WeatherSystem::step) every physics frame. The agent
// reads the flat observation with [`get_observation`](WeatherSystem::get_observation),
// collects penalties in [`get_reward_penalty_total`](WeatherSystem::get_reward_penalty_total)
// and can act via [`set_severity`](WeatherSystem::set_severity),
// [`modify_wind_vector`](WeatherSystem::modify_wind_vector) and
// [`trigger_evasive_maneuver`](WeatherSystem::trigger_evasive_maneuver).
// A curriculum can be enabled with [`set_curriculum`](WeatherSystem::set_curriculum).
//
// # Conventions
// Winds returned to Godot are **world (Y-up) vectors** `(north, -down, east)`.
// The flat observation array and all scalar getters stay in the documented
// `flight_core` units (NED m/s, metres, `[0,1]`).
//
// # Environment variables (defaults when called without arguments)
//   RAPTOR_WEATHER_SEED             u64   (default 1)
//   RAPTOR_WEATHER_DETERMINISTIC    "1"    fixed-pattern episode (default off)
//   RAPTOR_WEATHER_PRESET           clear|light_turb|rain|storm|extreme_storm
//   RAPTOR_WEATHER_SEVERITY         clear|light|moderate|severe (default clear)
//   RAPTOR_WEATHER_AUTO_CHANGE      seconds between stochastic scene re-rolls

use flight_core::nalgebra::Vector3 as NVec3;
use flight_core::weather::WeatherSystem as CoreWeather;
use flight_core::weather::{
    ObservationNormalizer, WeatherConfig, WeatherCurriculum, WeatherPreset, WeatherSeverity,
    DEFAULT_AUTO_CHANGE_INTERVAL,
};
use godot::builtin::{PackedFloat64Array, Vector3};
use godot::classes::Node3D;
use godot::prelude::*;

/// Build a [`WeatherConfig`] from the `RAPTOR_WEATHER_*` environment variables
/// (see the module docs), so the RL harness can be tuned without recompiling.
fn weather_config_from_env() -> WeatherConfig {
    let env = |k: &str| std::env::var(k).ok();
    let seed = env("RAPTOR_WEATHER_SEED")
        .and_then(|v| v.parse::<u64>().ok())
        .unwrap_or(1);
    let deterministic = env("RAPTOR_WEATHER_DETERMINISTIC")
        .map(|v| v == "1")
        .unwrap_or(false);
    let preset = match env("RAPTOR_WEATHER_PRESET").as_deref() {
        Some("light_turb" | "light") => WeatherPreset::LightTurb,
        Some("rain") => WeatherPreset::Rain,
        Some("storm") => WeatherPreset::Storm,
        Some("extreme_storm" | "extreme") => WeatherPreset::ExtremeStorm,
        _ => WeatherPreset::Clear,
    };
    let severity = match env("RAPTOR_WEATHER_SEVERITY").as_deref() {
        Some("light") => WeatherSeverity::Light,
        Some("moderate") => WeatherSeverity::Moderate,
        Some("severe") => WeatherSeverity::Severe,
        _ => WeatherSeverity::Clear,
    };
    let auto_change = env("RAPTOR_WEATHER_AUTO_CHANGE")
        .and_then(|v| v.parse::<f64>().ok())
        .unwrap_or(DEFAULT_AUTO_CHANGE_INTERVAL);
    WeatherConfig::from_preset(preset, seed)
        .with_auto_change(auto_change)
        .with_deterministic(deterministic)
        .with_severity(severity)
}

/// Convert an NED wind vector to the Godot world (Y-up) frame `(north, -down,
/// east)` — the same convention `FlightSimNode` uses for transforms.
fn world_from_ned(v: NVec3<f64>) -> Vector3 {
    Vector3::new(v.x as f32, -v.z as f32, v.y as f32)
}

/// # WeatherSystem
///
/// RL weather node. Create it in a scene (or `add_child`), call
/// [`start`](Self::start) (or [`configure`](Self::configure)), then advance
/// every physics frame with [`step`](Self::step). The embedded seed means two
/// runs with the same seed produce byte-identical weather — the reproducibility
/// RL experiment comparison needs.
#[derive(GodotClass)]
#[class(base = Node3D, init)]
pub(crate) struct WeatherSystem {
    /// The core weather state machine (or `None` before `start`).
    weather: Option<CoreWeather>,
    /// Geometric altitude (m) the wind field is evaluated at by [`step`](Self::step).
    altitude: f64,
    /// True airspeed (m/s) used for the Dryden gust decorrelation.
    airspeed: f64,
    /// Severity index last emitted on `weather_changed` (duplicate suppression).
    emitted_severity: i32,
    /// Scene index last emitted on `weather_changed`.
    emitted_scene: u64,
    /// Stable curriculum seed across resets.
    curriculum_seed: u64,
    base: Base<Node3D>,
}

#[godot_api]
impl WeatherSystem {
    /// Emitted every time the weather re-scenes: on severity changes and on
    /// each stochastic scene re-roll. Argument is the `WeatherSeverity` index
    /// (0 clear, 1 light, 2 moderate, 3 severe).
    #[signal]
    fn weather_changed(severity: i64);

    /// Initialise the weather system from the `RAPTOR_WEATHER_*` environment
    /// variables. Returns `false` if the core config failed to initialise
    /// (should not happen with defaults).
    #[func]
    fn start(&mut self) -> bool {
        self.configure_from(&weather_config_from_env())
    }

    /// Configure the scene from explicit parameters:
    ///   preset: "clear" | "light_turb" | "rain" | "storm" | "extreme_storm"
    ///   seed: reproducible re-roll seed (Godot int, ≥ 0)
    ///   deterministic: fixed-pattern episode when true
    ///   auto_change: seconds between stochastic scene re-rolls
    /// Returns `true` on success.
    #[func]
    fn configure(
        &mut self,
        preset: GString,
        seed: i64,
        deterministic: bool,
        auto_change: f64,
    ) -> bool {
        let preset = match preset.to_string().as_str() {
            "light_turb" | "light" => WeatherPreset::LightTurb,
            "rain" => WeatherPreset::Rain,
            "storm" => WeatherPreset::Storm,
            "extreme_storm" | "extreme" => WeatherPreset::ExtremeStorm,
            _ => WeatherPreset::Clear,
        };
        let cfg = WeatherConfig::from_preset(preset, seed.max(0) as u64)
            .with_auto_change(auto_change)
            .with_deterministic(deterministic);
        self.configure_from(&cfg)
    }

    fn configure_from(&mut self, cfg: &WeatherConfig) -> bool {
        self.weather = Some(CoreWeather::new(cfg.clone()));
        self.emitted_severity = -1;
        self.emitted_scene = 0;
        self.curriculum_seed = cfg.seed;
        true
    }

    /// True once [`start`](Self::start) / [`configure`](Self::configure) has
    /// initialised the weather.
    #[func]
    fn is_ready(&self) -> bool {
        self.weather.is_some()
    }

    /// Re-seed the episode (same seed ⇒ same weather). Call at the top of an
    /// RL episode for reproducible training runs.
    #[func]
    fn set_seed(&mut self, seed: i64) {
        let seed = seed.max(0) as u64;
        if let Some(w) = &mut self.weather {
            w.set_seed(seed);
        }
        self.emitted_severity = -1;
        self.emitted_scene = u64::MAX;
    }

    /// Force the weather to a severity band: 0 clear, 1 light, 2 moderate,
    /// 3 severe (agent `set_severity` action / robustness sweep).
    #[func]
    fn set_severity(&mut self, index: i64) {
        if let Some(w) = &mut self.weather {
            w.set_severity(WeatherSeverity::from_index(index as i32));
            self.emit_weather_changed();
        }
    }

    /// Switch deterministic (fixed pattern) / stochastic scene-rolling recipe.
    #[func]
    fn set_deterministic(&mut self, on: bool) {
        if let Some(w) = &mut self.weather {
            w.set_deterministic(on);
        }
    }

    /// Seconds between stochastic scene re-rolls (0 = never).
    #[func]
    fn set_auto_change_interval(&mut self, seconds: f64) {
        if let Some(w) = &mut self.weather {
            w.set_auto_change_interval(seconds);
        }
    }

    /// Altitude (m) the wind field / shear gradient is evaluated at.
    #[func]
    fn set_reference_altitude(&mut self, meters: f64) {
        self.altitude = meters.max(0.0);
    }

    /// True airspeed (m/s) driving the Dryden gust decorrelation.
    #[func]
    fn set_airspeed(&mut self, mps: f64) {
        self.airspeed = mps.max(0.5);
    }

    /// Advance the weather by `delta` seconds. Emits `weather_changed` when the
    /// scene re-rolls or the severity changes.
    #[func]
    fn step(&mut self, delta: f64) {
        let Some(w) = &mut self.weather else {
            return;
        };
        w.step(
            self.altitude,
            self.airspeed,
            delta.clamp(0.0, 0.1).max(1e-4),
        );
        self.emit_weather_changed();
    }

    /// Reset the episode (same seed ⇒ identical weather trace).
    #[func]
    fn reset(&mut self) {
        if let Some(w) = &mut self.weather {
            w.reset();
        }
        self.emitted_severity = -1;
        self.emitted_scene = u64::MAX;
    }

    fn emit_weather_changed(&mut self) {
        let Some(w) = &self.weather else {
            return;
        };
        let sev = w.severity().index();
        let scene = w.scene_index();
        if sev != self.emitted_severity || scene != self.emitted_scene {
            self.emitted_severity = sev;
            self.emitted_scene = scene;
            self.signals().weather_changed().emit(sev as i64);
        }
    }

    // -------------------------------------------------------------
    // RL observations
    // -------------------------------------------------------------

    /// The flat 12-channel agent observation (see `WeatherSystem` layout):
    /// wind_north/east/down, turbulence_01, rain, visibility_01, shear_01,
    /// updraft, precipitation one-hot (rain/snow/hail), severity_01.
    /// Returns an empty array before `start`.
    #[func]
    fn get_observation(&self) -> PackedFloat64Array {
        let Some(w) = &self.weather else {
            return PackedFloat64Array::new();
        };
        PackedFloat64Array::from(w.observation_array().to_vec())
    }

    /// The observation normalised per-channel to `[0, 1]` using the documented
    /// physical bounds (wind ±30 m/s, updraft ±20 m/s) for network-friendly
    /// inputs.
    #[func]
    fn get_observation_normalized(&self) -> PackedFloat64Array {
        let Some(w) = &self.weather else {
            return PackedFloat64Array::new();
        };
        let n = ObservationNormalizer::from_default_bounds();
        PackedFloat64Array::from(n.normalize(&w.observation_array()).to_vec())
    }

    /// Total wind (steady + gusts + agent offset + updraft) as a world vector
    /// `(north, -down, east)` m/s.
    #[func]
    fn get_wind_vector(&self) -> Vector3 {
        self.weather
            .as_ref()
            .map(|w| world_from_ned(w.total_wind()))
            .unwrap_or(Vector3::ZERO)
    }

    /// Last Dryden gust component as a world vector `(north, -down, east)`.
    #[func]
    fn get_last_gust(&self) -> Vector3 {
        self.weather
            .as_ref()
            .map(|w| world_from_ned(w.gust_vector()))
            .unwrap_or(Vector3::ZERO)
    }

    /// Current agent wind offset (world frame) commanded via
    /// [`modify_wind_vector`](Self::modify_wind_vector).
    #[func]
    fn get_wind_offset(&self) -> Vector3 {
        self.weather
            .as_ref()
            .map(|w| world_from_ned(w.wind_offset()))
            .unwrap_or(Vector3::ZERO)
    }

    #[func]
    fn get_rain_intensity(&self) -> f64 {
        self.weather
            .as_ref()
            .map(|w| w.rain_intensity())
            .unwrap_or(0.0)
    }

    /// Visibility distance (m).
    #[func]
    fn get_visibility(&self) -> f64 {
        self.weather.as_ref().map(|w| w.visibility()).unwrap_or(0.0)
    }

    /// Visibility normalised `[0,1]` (1 = clear, 0 = zero visibility).
    #[func]
    fn get_visibility_01(&self) -> f64 {
        let v = self.get_visibility();
        (v / flight_core::weather::REFERENCE_VISIBILITY).clamp(0.0, 1.0)
    }

    /// Steady wind speed of the active scene (m/s).
    #[func]
    fn get_wind_speed(&self) -> f64 {
        self.weather
            .as_ref()
            .map(|w| w.observation().wind_speed)
            .unwrap_or(0.0)
    }

    /// Turbulence level index: 0 light, 1 moderate, 2 severe, 3 extreme.
    #[func]
    fn get_turbulence_level(&self) -> i64 {
        self.weather
            .as_ref()
            .map(|w| w.turbulence_level().index() as i64)
            .unwrap_or(0)
    }

    /// Effective gust RMS σ_u (m/s).
    #[func]
    fn get_gust_rms(&self) -> f64 {
        self.weather.as_ref().map(|w| w.gust_rms()).unwrap_or(0.0)
    }

    /// Wind-shear gradient magnitude `|dW/dh|` in `(m/s)/m`.
    #[func]
    fn get_shear_gradient(&self) -> f64 {
        self.weather
            .as_ref()
            .map(|w| w.shear_gradient(self.altitude))
            .unwrap_or(0.0)
    }

    /// Updraft strength (m/s, positive = up).
    #[func]
    fn get_updraft_strength(&self) -> f64 {
        self.weather
            .as_ref()
            .map(|w| w.updraft_strength())
            .unwrap_or(0.0)
    }

    /// Precipitation type index: 0 none, 1 rain, 2 snow, 3 hail.
    #[func]
    fn get_precipitation_type(&self) -> i64 {
        self.weather
            .as_ref()
            .map(|w| w.precipitation().index() as i64)
            .unwrap_or(0)
    }

    /// Severity index: 0 clear … 3 severe.
    #[func]
    fn get_severity(&self) -> i64 {
        self.weather
            .as_ref()
            .map(|w| w.severity().index() as i64)
            .unwrap_or(0)
    }

    #[func]
    fn get_sim_time(&self) -> f64 {
        self.weather.as_ref().map(|w| w.sim_time()).unwrap_or(0.0)
    }

    #[func]
    fn get_scene_index(&self) -> i64 {
        self.weather
            .as_ref()
            .map(|w| w.scene_index() as i64)
            .unwrap_or(0)
    }

    // -------------------------------------------------------------
    // RL rewards
    // -------------------------------------------------------------

    #[func]
    fn get_weather_penalty(&self) -> f64 {
        self.weather
            .as_ref()
            .map(|w| w.weather_penalty())
            .unwrap_or(0.0)
    }

    #[func]
    fn get_turbulence_penalty(&self) -> f64 {
        self.weather
            .as_ref()
            .map(|w| w.turbulence_penalty())
            .unwrap_or(0.0)
    }

    #[func]
    fn get_visibility_penalty(&self) -> f64 {
        self.weather
            .as_ref()
            .map(|w| w.visibility_penalty())
            .unwrap_or(0.0)
    }

    #[func]
    fn get_wind_penalty(&self) -> f64 {
        self.weather
            .as_ref()
            .map(|w| w.wind_penalty())
            .unwrap_or(0.0)
    }

    /// Sum of all weather penalties for the current step (≤ 0). Add to the
    /// episode reward.
    #[func]
    fn get_reward_penalty_total(&self) -> f64 {
        self.weather
            .as_ref()
            .map(|w| w.reward_penalty_total())
            .unwrap_or(0.0)
    }

    // -------------------------------------------------------------
    // RL termination
    // -------------------------------------------------------------

    /// 0 = benign, 1 = gust limit, 2 = visibility loss, 3 = severe exposure.
    /// `extreme_weather_termination` must be enabled in the config for this to
    /// trigger (see [`configure`](Self::configure) config, or override via the
    /// Rust API).
    #[func]
    fn check_termination(&self) -> i64 {
        self.weather
            .as_ref()
            .map(|w| w.termination().index() as i64)
            .unwrap_or(0)
    }

    #[func]
    fn is_terminated(&self) -> bool {
        self.weather
            .as_ref()
            .map(|w| w.is_terminated())
            .unwrap_or(false)
    }

    // -------------------------------------------------------------
    // RL actions
    // -------------------------------------------------------------

    /// Agent action: add a wind perturbation in the world frame
    /// `(north, -down, east)` (m/s), clamped to ±30 m/s per axis.
    #[func]
    fn modify_wind_vector(&mut self, north: f64, east: f64, down_world: f64) {
        if let Some(w) = &mut self.weather {
            w.modify_wind_vector(NVec3::new(north, -down_world, east));
        }
    }

    #[func]
    fn clear_wind_offset(&mut self) {
        if let Some(w) = &mut self.weather {
            w.clear_wind_offset();
        }
    }

    /// Agent action: temporarily find smoother air — gust RMS and the weather
    /// penalty are both reduced while the maneuver lasts.
    #[func]
    fn trigger_evasive_maneuver(&mut self) {
        if let Some(w) = &mut self.weather {
            w.trigger_evasive_maneuver();
        }
    }

    #[func]
    fn is_evasive(&self) -> bool {
        self.weather
            .as_ref()
            .map(|w| w.is_evasive())
            .unwrap_or(false)
    }

    // -------------------------------------------------------------
    // Curriculum
    // -------------------------------------------------------------

    /// Enable the curriculum (clear → light turbulence → rain → storm), with
    /// `steps_per_phase` environment steps per phase. The seed makes each
    /// phase's episodes reproducible.
    #[func]
    fn set_curriculum(&mut self, seed: i64, steps_per_phase: i64) {
        let Some(w) = &mut self.weather else {
            return;
        };
        let curriculum = WeatherCurriculum::new(steps_per_phase.max(1) as usize);
        self.curriculum_seed = seed.max(0) as u64;
        let sys = std::mem::replace(w, CoreWeather::new(WeatherConfig::default()));
        *w = sys.with_curriculum(curriculum, self.curriculum_seed);
        self.emitted_severity = -1;
        self.emitted_scene = u64::MAX;
    }

    /// Explicitly advance to the next curriculum phase. Returns the new phase
    /// index (0 = clear … 3 = storm).
    #[func]
    fn next_curriculum_phase(&mut self) -> i64 {
        let Some(w) = &mut self.weather else {
            return 0;
        };
        let idx = w.next_curriculum_phase();
        self.emit_weather_changed();
        idx as i64
    }

    #[func]
    fn get_curriculum_phase(&self) -> i64 {
        self.weather
            .as_ref()
            .map(|w| w.curriculum_phase() as i64)
            .unwrap_or(0)
    }

    #[func]
    fn is_curriculum_complete(&self) -> bool {
        self.weather
            .as_ref()
            .map(|w| w.is_curriculum_complete())
            .unwrap_or(false)
    }

    #[func]
    fn get_curriculum_progress(&self) -> f64 {
        self.weather
            .as_ref()
            .map(|w| w.curriculum_progress())
            .unwrap_or(1.0)
    }
}
