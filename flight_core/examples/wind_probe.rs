//! Probe: verify wind affects groundspeed (not airspeed) and doesn't destabilize
//! trimmed flight. Run with the repo root as CWD: `cargo run -p flight_core --example wind_probe`.

use flight_core::{Simulator, WindConfig, WindEnvironment};
use nalgebra::Vector3;

fn main() {
    let path = if std::path::Path::new("aircraft.toml").exists() {
        "aircraft.toml"
    } else {
        "../aircraft.toml"
    };
    let mut sim = Simulator::new(path);
    let (elev, thr) = sim.trim_level_flight(1000.0, 60.0);
    println!("trim elev={:.4} rad thr={:.3}", elev, thr);

    for (name, wind_vec) in [
        ("still", Vector3::zeros()),
        ("tailwind 10", Vector3::new(10.0, 0.0, 0.0)),   // blowing north
        ("crosswind 10", Vector3::new(0.0, 10.0, 0.0)), // blowing east
        ("headwind 10", Vector3::new(-10.0, 0.0, 0.0)), // blowing south
    ] {
        sim.state = sim.state.clone();
        sim.state.trim_level_flight(&sim.config, 1000.0, 60.0);
        // Re-trim is not wind-aware; use the still-air trim to keep it simple.
        let _ = (elev, thr);
        // Step forward 20 s at 60 Hz under the given constant wind.
        for _ in 0..(20 * 60) {
            sim.step_6dof(elev, 0.0, 0.0, thr, 0.0, Some(&wind_vec), 1.0 / 60.0);
        }
        let st = &sim.state;
        let tas = st.true_airspeed(&wind_vec);
        let gs = st.airspeed();
        println!(
            "{name:14} TAS={tas:6.2} m/s  GS={gs:6.2} m/s  alt={:.1} m  pitch={:.1}°",
            st.altitude(),
            st.euler_angles().1.to_degrees()
        );
    }

    // Turbulence run: moderate turbulence for 30 s.
    sim.state = sim.state.clone();
    sim.state.trim_level_flight(&sim.config, 1000.0, 60.0);
    let mut wcfg = WindConfig::default();
    wcfg.turbulence = flight_core::TurbulenceIntensity::Moderate;
    let mut wind_env = WindEnvironment::new(wcfg);
    let mut spare = 0.0f64;
    for _ in 0..(30 * 60) {
        let vt = sim.state.airspeed();
        let w = wind_env.total_wind(&sim.state, vt, 1.0 / 60.0);
        spare = spare.max(w.norm());
        sim.step_6dof(elev, 0.0, 0.0, thr, 0.0, Some(&w), 1.0 / 60.0);
    }
    println!(
        "turbulent 30s: max_wind={spare:.2} m/s  alt={:.1} m  TAS={:.2} m/s",
        sim.state.altitude(),
        sim.state.true_airspeed(&Vector3::zeros())
    );
}
