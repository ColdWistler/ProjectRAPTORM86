//! RL weather training-loop demo: fly a fixed-wing drone through a
//! seed-reproducible weather layer while collecting observations, shaping the
//! reward with weather penalties, and pacing a curriculum. Run with the repo
//! root as CWD:
//!
//! ```text
//! cargo run -p flight_core --example rl_weather_loop
//! ```
//!
//! This is a CLI demonstration of the *interface* (and a reproducibility
//! sanity check) — plug the observation vector / penalties into any Rust RL
//! stack (tch-rs, rust-rl, custom PPO/A2C). See the module docs in
//! `flight_core/src/weather.rs` and `docs/rl_weather_guide.md`.

use flight_core::{
    ControlAction, EnvConfig, Environment, TurbulenceLevel, WeatherConfig, WeatherCurriculum,
    WeatherPreset, WeatherSystem, WEATHER_OBS_DIM,
};
use nalgebra::Vector3;

/// Search a few relative paths for the aircraft TOML (matches `wind_probe`).
fn config_path() -> &'static str {
    if std::path::Path::new("aircraft.toml").exists() {
        "aircraft.toml"
    } else {
        "../aircraft.toml"
    }
}

fn main() {
    // ------------------------------------------------------------------
    // 1. Standalone WeatherSystem: observation packaging & reproducibility
    // ------------------------------------------------------------------
    println!("== 1. standalone weather system ==");
    let mut weather = WeatherSystem::new(
        WeatherConfig::from_preset(WeatherPreset::Rain, 42).with_deterministic(true),
    );
    for _ in 0..(10 * 30) {
        weather.step(1_000.0, 60.0, 1.0 / 30.0);
    }
    let obs = weather.observation();
    println!(
        "rain[{}] vis[{} m] updraft[{:+.1}] shear[{:.4} (m/s)/m] severity[{}] turb[{:?}]",
        obs.rain_intensity,
        obs.visibility as u32,
        obs.updraft_strength,
        obs.shear_gradient,
        obs.severity.name(),
        TurbulenceLevel::from_index(obs.turbulence_level),
    );
    let flat: [f64; WEATHER_OBS_DIM] = weather.observation_array();
    let vec32: Vec<f32> = weather.observation_vec_f32();
    let dvec = weather.to_dvector();
    println!(
        "flat[{}]  = {:6.3} {:6.3} {:6.3} | turb {:.2} rain {:.2} vis {:.2} shear {:.2} up {:.2} | precip {} {} {} | sev {:.2}",
        WEATHER_OBS_DIM, flat[0], flat[1], flat[2], flat[3], flat[4], flat[5], flat[6], flat[7],
        flat[8], flat[9], flat[10], flat[11],
    );
    assert_eq!(vec32.len(), WEATHER_OBS_DIM);
    assert_eq!(dvec.len(), WEATHER_OBS_DIM);
    println!(
        "penalties: weather {:.3}, turb {:.3}, vis {:.3}, wind {:.3} → total {:.3}",
        weather.weather_penalty(),
        weather.turbulence_penalty(),
        weather.visibility_penalty(),
        weather.wind_penalty(),
        weather.reward_penalty_total(),
    );

    // Same seed, restarted episode ⇒ identical weather trace (RL comparison).
    let mut a = WeatherSystem::new(
        WeatherConfig::from_preset(WeatherPreset::Storm, 7).with_deterministic(true),
    );
    let mut b = WeatherSystem::new(
        WeatherConfig::from_preset(WeatherPreset::Storm, 7).with_deterministic(true),
    );
    let mut winds_a = Vec::new();
    let mut winds_b = Vec::new();
    for _ in 0..(30 * 30) {
        winds_a.push(a.step(800.0, 60.0, 1.0 / 30.0));
        winds_b.push(b.step(800.0, 60.0, 1.0 / 30.0));
    }
    let same = winds_a
        .iter()
        .zip(&winds_b)
        .all(|(x, y)| (x - y).norm() < 1e-9);
    println!("same seed, two runs → identical NED wind trace: {same}");

    // ------------------------------------------------------------------
    // 2. Environment training loop with weather penalties folded into reward
    // ------------------------------------------------------------------
    println!("\n== 2. environment loop (rain, auto-change 15 s) ==");
    let mut env = Environment::with_config(
        config_path(),
        EnvConfig {
            dt: 1.0 / 30.0,
            max_steps: 1_200,
            weather_config: Some(
                WeatherConfig::from_preset(WeatherPreset::Rain, 2024).with_auto_change(15.0),
            ),
            ..Default::default()
        },
    )
    .expect("aircraft.toml should load");

    // Trivial policy: hold trim, occasionally trigger an evasive maneuver when
    // turbulence gets extreme (a stand-in for a learned policy).
    let mut total_reward = 0.0;
    let mut total_weather_penalty = 0.0;
    let mut steps = 0;
    env.reset();
    let action = ControlAction::neutral();
    loop {
        if let Some(w) = env.weather.as_ref() {
            if w.turbulence_level() == TurbulenceLevel::Extreme {
                env.weather.as_mut().unwrap().trigger_evasive_maneuver();
            }
        }
        let r = env.step(action);
        total_reward += r.reward;
        total_weather_penalty += env.weather_penalty();
        steps += 1;
        if r.terminated || r.truncated {
            break;
        }
    }
    println!("episode: {steps} steps, reward {total_reward:+.2}, weather penalty {total_weather_penalty:+.2}");
    let full = env.full_observation();
    println!(
        "full observation (flight 12 ‖ weather {}): {full:?}",
        WEATHER_OBS_DIM
    );
    assert_eq!(full.len(), 12 + WEATHER_OBS_DIM);

    // ------------------------------------------------------------------
    // 3. Curriculum: clear → light turbulence → rain → storm
    // ------------------------------------------------------------------
    println!("\n== 3. curriculum ==");
    let mut curriculum = WeatherCurriculum::new(1_000);
    while !curriculum.is_complete() {
        curriculum.advance(1_000);
    }
    println!(
        "curriculum reached {}, complete = {}, progress = {:.0}%",
        curriculum.phase().name(),
        curriculum.is_complete(),
        curriculum.progress() * 100.0,
    );
    let storm_cfg = curriculum.config(99);
    println!(
        "phase config: severity {} preset {}",
        storm_cfg.severity.name(),
        WeatherPreset::Storm.name()
    );

    // Curriculum embedded in the weather system: phases drive the scenes.
    let mut sys = WeatherSystem::new(WeatherConfig::default())
        .with_curriculum(WeatherCurriculum::new(600), 404);
    for _ in 0..(4 * 600 + 60) {
        sys.step(800.0, 60.0, 1.0 / 30.0);
    }
    println!(
        "embedded curriculum finished at phase {} (complete {}), scene index {}",
        sys.curriculum_phase(),
        sys.is_curriculum_complete(),
        sys.scene_index(),
    );

    // ------------------------------------------------------------------
    // 4. Agent actions
    // ------------------------------------------------------------------
    println!("\n== 4. agent actions ==");
    let mut w2 = WeatherSystem::new(WeatherConfig::from_preset(WeatherPreset::Clear, 9));
    w2.modify_wind_vector(Vector3::new(5.0, 0.0, 0.0));
    w2.set_severity(flight_core::WeatherSeverity::Severe);
    w2.trigger_evasive_maneuver();
    let o = w2.observation();
    println!(
        "after actions: severity {} (evasive {}) wind {:+.2} {:+.2} {:+.2} penalty {:.3}",
        o.severity.name(),
        w2.is_evasive(),
        o.wind.x,
        o.wind.y,
        o.wind.z,
        w2.reward_penalty_total(),
    );

    println!("\nRL interface ready — observations, penalties, curriculum and actions are wired.");
}
