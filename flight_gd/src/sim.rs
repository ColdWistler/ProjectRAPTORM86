//! Interactive 6-DOF flight simulation bridge.
//!
//! Kinematics/forces run inside Rust (`flight_core`); Godot only reads the
//! resulting transform + telemetry and writes control inputs. All coordinates
//! handed to Godot are in Godot's Y-up world frame: the NED Earth frame
//! (North, East, Down) maps to world `(north, -down, east)`, and the aircraft
//! model is built with its nose along local **+X** (up = +Y, right = +Z).
//!
//! # Conventions
//! - Angles between GDScript and Rust are radians except where noted (°).
//! - Telemetry layout is index-fixed and documented on [`FlightSimNode::telemetry`].
//! - Environment variables (`RAPTOR_*`) tune the wind without recompiling;
//!   see [`wind_config_from_env`].

use flight_core::avionics::{standard_stack, AvionicsSystem, FaultFlag};
use flight_core::{
    nalgebra::Vector3 as NVec3, AircraftConfig, AircraftState, Atmosphere, Simulator, Terrain,
    TurbulenceIntensity, WindConfig, WindEnvironment,
};
use godot::builtin::{PackedFloat64Array, Transform3D, Vector2, Vector3};
use godot::classes::Node3D;
use godot::prelude::*;

/// Candidate directories (relative to the Godot working directory) that may
/// contain aircraft config TOML files. When launched via `godot --path godot`
/// the CWD is `godot/`, so the workspace root is two levels up. When run from
/// the editor the CWD is the project root, so `..` and `../..` cover that too.
const CONFIG_PATHS: [&str; 6] = ["", "..", "../..", "../../..", "../../...", "../../../.."];
/// Max elevator/aileron/rudder deflection (radians).
const MAX_ELEVATOR: f64 = 0.35;
const MAX_AILERON: f64 = 0.35;
const MAX_RUDDER: f64 = 0.35;

/// Search the known candidate directories (relative to the Godot working
/// directory) for a config file called `file_name`, returning its path if
/// found.
fn resolve_config_in(file_name: &str) -> Option<String> {
    for p in CONFIG_PATHS {
        let candidate = if p.is_empty() {
            file_name.to_string()
        } else {
            format!("{p}/{file_name}")
        };
        if std::path::Path::new(&candidate).exists() {
            return Some(candidate);
        }
    }
    None
}

/// Build a [`WindConfig`] from environment variables so the air model can be
/// tweaked without recompiling:
///   RAPTOR_WIND_SPEED    m/s (default 0 = still air)
///   RAPTOR_WIND_DIR_DEG  true bearing the wind blows TOWARD (default 0)
///   RAPTOR_WIND_SHEAR    "1" enables altitude shear (default off)
///   RAPTOR_TURBULENCE    light | moderate | severe (default light)
fn wind_config_from_env() -> WindConfig {
    let env = |k: &str| std::env::var(k).ok();
    let speed = env("RAPTOR_WIND_SPEED")
        .and_then(|v| v.parse::<f64>().ok())
        .unwrap_or(0.0);
    let dir_deg = env("RAPTOR_WIND_DIR_DEG")
        .and_then(|v| v.parse::<f64>().ok())
        .unwrap_or(0.0);
    let shear = env("RAPTOR_WIND_SHEAR").map(|v| v == "1").unwrap_or(false);
    let turbulence = match env("RAPTOR_TURBULENCE").as_deref() {
        Some("moderate") => TurbulenceIntensity::Moderate,
        Some("severe") => TurbulenceIntensity::Severe,
        _ => TurbulenceIntensity::Light,
    };
    WindConfig {
        wind_speed: speed,
        wind_direction: dir_deg.to_radians(),
        reference_altitude: 1000.0,
        wind_shear: shear,
        turbulence,
        turbulence_scale: 533.0,
        seed: 0x9E37_79B9_7F4A_7C15,
    }
}

/// A 6-DOF flight dynamics node. Create it in a scene (or `add_child`), then
/// call [`FlightSimNode::start`] with a path to `aircraft.toml`.
#[derive(GodotClass)]
#[class(base = Node3D, init)]
struct FlightSimNode {
    sim: Option<(Simulator, WindEnvironment)>,
    last_wind: NVec3<f64>,
    /// Avionics hardware stack (sensors → PID flight controller → servos/ESC
    /// → battery). Built in [`FlightSimNode::start`]; the main control mode.
    avionics: Option<AvionicsSystem>,
    /// True while `step` flies through the avionics stack. Calling
    /// [`FlightSimNode::set_avionics_command`] enables it;
    /// [`FlightSimNode::set_controls`] bypasses it for direct manual control.
    avionics_active: bool,
    /// Manual stick input (radians).
    elevator: f64,
    /// Baseline pitch trim tab (radians).
    elevator_trim: f64,
    aileron: f64,
    rudder: f64,
    /// Trailing-edge flap deployment (0, 15, 30 degrees).
    flaps_deg: f64,
    /// Engine thrust setting (0..=1).
    throttle: f64,
    /// Asymmetric engine throttle split (-1 = left engine out, +1 = right out).
    throttle_split: f64,
    /// Wing-leveler / altitude-hold assist.
    auto_level: bool,
    target_alt: f64,
    /// Physical ground: sampled grid / analytic hills. `flat()` disables
    /// terrain collision, ground effect and orographic wind.
    terrain: Terrain,
    /// Master switch; the airport runway stays flat while the rest bumps.
    terrain_enabled: bool,
    base: Base<Node3D>,
}

#[godot_api]
impl FlightSimNode {
    /// Load the aircraft config and initialize a trimmed level-flight state at
    /// 1000 m, 60 m/s. Returns `false` if the config file could not be loaded.
    #[func]
    fn start(&mut self, config_path: GString) -> bool {
        let file_name = if !config_path.is_empty() {
            config_path.to_string()
        } else {
            "aircraft.toml".to_string()
        };
        godot_warn!("FlightSimNode: searching for '{file_name}' in {:?}", CONFIG_PATHS);
        let path = match resolve_config_in(&file_name) {
            Some(p) => {
                godot_warn!("FlightSimNode: found '{file_name}' at '{p}'");
                p
            }
            None => {
                godot_error!("FlightSimNode: '{file_name}' not found in search paths");
                return false;
            }
        };
        let Ok(config) = AircraftConfig::from_file(&path) else {
            godot_error!("FlightSimNode: failed to parse config {path}");
            return false;
        };
        let mut state = AircraftState::default();
        let (trim_elev, trim_throttle) = state.trim_level_flight(&config, 50.0, 60.0);
        self.sim = Some((Simulator { config, state }, self.wind_environment()));
        self.avionics = Some(Self::build_avionics());
        self.avionics_active = true;
        if let Some(av) = self.avionics.as_mut() {
            av.bus_mut().cmd_throttle = trim_throttle;
        }
        self.elevator = 0.0;
        self.elevator_trim = trim_elev;
        self.aileron = 0.0;
        self.rudder = 0.0;
        self.flaps_deg = 0.0;
        self.throttle = trim_throttle;
        self.auto_level = false;
        self.target_alt = 50.0;
        self.last_wind = NVec3::zeros();
        true
    }

    /// True once [`start`](Self::start) has initialized the simulation.
    #[func]
    fn is_ready(&self) -> bool {
        self.sim.is_some()
    }

    /// Advance the 6-DOF simulation by `dt` seconds (0 => 1/60 s). By default
    /// the aircraft flies through the avionics stack (sensors → PID flight
    /// controller → servos/ESC → battery); direct manual surface control is
    /// used instead only while [`set_controls`](Self::set_controls) bypasses it.
    #[func]
    fn step(&mut self, dt: f64) {
        let Some((sim, wind)) = self.sim.as_mut() else {
            return;
        };
        let dt = dt.clamp(0.0, 0.1).max(1e-4);

        let avionics_active = self.avionics_active && self.avionics.is_some();

        // Autopilot: wing leveler + altitude/pitch hold (manual path only;
        // the flight controller already stabilizes attitude in avionics mode).
        if self.auto_level && !avionics_active {
            let (roll, pitch, _) = sim.state.euler_angles();
            let state = &sim.state;

            let roll_cmd = (-0.75 * roll - 0.35 * state.p).clamp(-MAX_AILERON, MAX_AILERON);
            self.aileron = roll_cmd;

            let alt_error = self.target_alt - state.altitude();
            let target_climb_pitch = (alt_error * 0.003).clamp(-0.12, 0.12);
            let pitch_error = target_climb_pitch - pitch;
            let pitch_cmd = (0.6 * pitch_error - 0.45 * state.q).clamp(-MAX_ELEVATOR * 0.6, MAX_ELEVATOR * 0.6);
            self.elevator = pitch_cmd;
        }

        // Total elevator = stick + trim, clamped like the legacy visualizer.
        let total_elevator = (self.elevator + self.elevator_trim).clamp(-MAX_ELEVATOR, MAX_ELEVATOR);

        let vt_air = sim.state.airspeed();
        let wind_earth = wind.total_wind(&sim.state, vt_air, dt);
        self.last_wind = wind_earth;
        // Apply the asymmetric engine split (engine-out) to the active config.
        sim.config.throttle_split = self.throttle_split;

        let terrain = if self.terrain_enabled {
            Some(&self.terrain)
        } else {
            None
        };

        // Avionics path (main mode): the FC reads the attitude commands, the
        // sensor→controller→actuator→battery chain runs, and the resulting
        // surface deflections + ESC throttle drive the 6-DOF physics.
        if avionics_active {
            if let Some(av) = self.avionics.as_mut() {
                let tas = sim.state.true_airspeed(&wind_earth);
                let q_dynamic = Atmosphere::at_altitude(sim.state.altitude()).dynamic_pressure(tas);
                av.bus_mut().write_true_state(&sim.state, &wind_earth, q_dynamic);
                av.step();
                let (elev, ail, rud, thr) = av.bus().read_actuator_outputs();
                sim.step_6dof(
                    elev,
                    ail,
                    rud,
                    thr,
                    self.flaps_deg.to_radians(),
                    Some(&wind_earth),
                    dt,
                    terrain,
                );
                return;
            }
        }

        sim.step_6dof(
            total_elevator,
            self.aileron,
            self.rudder,
            self.throttle,
            self.flaps_deg.to_radians(),
            Some(&wind_earth),
            dt,
            terrain,
        );
    }

    /// Replace the physical ground with a sampled height grid from an imported
    /// terrain mesh. `north0`/`east0` are the NED (north, east) coordinates of
    /// cell (0,0), `spacing` the cell size (m), `nx`/`nz` the dimensions, and
    /// `heights` the row-major surface altitudes in m above datum.
    ///
    /// The grid maps Godot world `(x, z)` to NED `(north, east)` and the world
    /// `y` of the sampled mesh to the surface altitude.
    #[func]
    fn configure_terrain(
        &mut self,
        north0: f64,
        east0: f64,
        spacing: f64,
        nx: i64,
        nz: i64,
        heights: PackedFloat64Array,
    ) {
        let nx = nx.max(2) as usize;
        let nz = nz.max(2) as usize;
        if heights.len() < nx * nz {
            godot_error!(
                "configure_terrain: got {} heights, need {} ({}x{})",
                heights.len(),
                nx * nz,
                nx,
                nz
            );
            return;
        }
        let mut h = Vec::with_capacity(nx * nz);
        for i in 0..nx * nz {
            h.push(heights.get(i).unwrap_or(0.0));
        }
        self.terrain = Terrain::from_grid(north0, east0, spacing, nx, nz, h);
        godot_warn!("FlightSimNode: terrain grid {nx}x{nz} @ {spacing:.1} m set");
    }

    /// Enable/disable the physical terrain (collision + ground effect +
    /// orographic wind). Disabled = flat infinite floor, the historical mode.
    #[func]
    fn set_terrain_enabled(&mut self, enabled: bool) {
        self.terrain_enabled = enabled;
    }

    /// Height of the physical terrain (m above datum) at a NED (north, east)
    /// position. Returns 0 when terrain is flat/disabled, which also lets
    /// callers compute AGL: `altitude - terrain_height_at(...)`.
    #[func]
    fn terrain_height_at(&self, north: f64, east: f64) -> f64 {
        if !self.terrain_enabled {
            return 0.0;
        }
        self.terrain.height(north, east)
    }

    /// Replace all control inputs in one call (angles in radians, flaps in °).
    /// This bypasses the avionics stack and drives the surfaces directly.
    #[func]
    fn set_controls(&mut self, elevator: f64, aileron: f64, rudder: f64, throttle: f64, flaps_deg: f64) {
        self.elevator = elevator.clamp(-MAX_ELEVATOR, MAX_ELEVATOR);
        self.aileron = aileron.clamp(-MAX_AILERON, MAX_AILERON);
        self.rudder = rudder.clamp(-MAX_RUDDER, MAX_RUDDER);
        self.throttle = throttle.clamp(0.0, 1.0);
        self.flaps_deg = flaps_deg;
        self.avionics_active = false;
    }

    /// Command the flight controller's attitude setpoints and switch to the
    /// avionics path (the main mode): `roll`/`pitch` in radians, `yaw_rate`
    /// in rad/s, `throttle` 0..=1. `step` then flies through the sensors,
    /// PID flight controller, servos/ESC and battery.
    #[func]
    fn set_avionics_command(&mut self, roll: f64, pitch: f64, yaw_rate: f64, throttle: f64) {
        if let Some(av) = self.avionics.as_mut() {
            let bus = av.bus_mut();
            bus.cmd_roll = roll;
            bus.cmd_pitch = pitch;
            bus.cmd_yaw_rate = yaw_rate;
            bus.cmd_throttle = throttle.clamp(0.0, 1.0);
            self.avionics_active = true;
        }
    }

    /// True while `step` flies through the avionics stack (main mode).
    #[func]
    fn is_avionics_active(&self) -> bool {
        self.avionics_active && self.avionics.is_some()
    }

    /// Inject a named fault into the avionics stack, e.g. `"imu"`, `"gps"`,
    /// `"baro"`, `"mag"`, `"airspeed"`, `"servo_elevator"`,
    /// `"servo_aileron"`, `"servo_rudder"`, `"esc"` or
    /// `"battery_depleted"`. Returns `false` if the name is not recognised.
    #[func]
    fn inject_fault(&mut self, name: GString) -> bool {
        let Some(flag) = fault_flag_from_str(&name.to_string()) else {
            return false;
        };
        if let Some(av) = self.avionics.as_mut() {
            av.inject_fault(flag, true);
            return true;
        }
        false
    }

    /// Clear a previously injected fault. Returns `false` if the name is not
    /// recognised.
    #[func]
    fn clear_fault(&mut self, name: GString) -> bool {
        let Some(flag) = fault_flag_from_str(&name.to_string()) else {
            return false;
        };
        if let Some(av) = self.avionics.as_mut() {
            av.inject_fault(flag, false);
            return true;
        }
        false
    }

    /// The 19-channel noisy sensor observation from the avionics bus. See the
    /// layout documented on `flight_core::avionics::AvionicsBus::sensor_observation`.
    /// Returns an empty array before [`start`](Self::start).
    #[func]
    fn avionics_observation(&self) -> PackedFloat64Array {
        let Some(av) = &self.avionics else {
            return PackedFloat64Array::new();
        };
        PackedFloat64Array::from(av.bus().sensor_observation().to_vec())
    }

    /// Set the asymmetric engine throttle split (`-1..=1`). `0` runs both
    /// engines together; `-1` shuts the left engine down, `+1` the right.
    #[func]
    fn set_throttle_split(&mut self, split: f64) {
        self.throttle_split = split.clamp(-1.0, 1.0);
    }

    /// Switch the active aircraft configuration by name (e.g. `"MQI"` or
    /// `"TwinEngine"`) and re-trim to level flight. Returns `true` on success.
    /// The name is mapped to a `<name>.toml` in the same search locations as
    /// the default `aircraft.toml`.
    #[func]
    fn switch_aircraft(&mut self, name: GString) -> bool {
        let name = name.to_string();
        let file_name = format!("{name}.toml");
        let Some(path) = resolve_config_in(&file_name) else {
            godot_error!("FlightSimNode: no config for aircraft '{name}'");
            return false;
        };
        let Ok(config) = AircraftConfig::from_file(&path) else {
            godot_error!("FlightSimNode: failed to parse config {path}");
            return false;
        };
        let mut state = AircraftState::default();
        let (trim_elev, trim_throttle) = state.trim_level_flight(&config, 50.0, 60.0);
        self.sim = Some((Simulator { config, state }, self.wind_environment()));
        self.avionics = Some(Self::build_avionics());
        self.avionics_active = true;
        if let Some(av) = self.avionics.as_mut() {
            av.bus_mut().cmd_throttle = trim_throttle;
        }
        self.elevator = 0.0;
        self.elevator_trim = trim_elev;
        self.aileron = 0.0;
        self.rudder = 0.0;
        self.flaps_deg = 0.0;
        self.throttle = trim_throttle;
        self.throttle_split = 0.0;
        self.auto_level = false;
        self.target_alt = 50.0;
        true
    }

    #[func]
    fn set_elevator_trim(&mut self, radians: f64) {
        self.elevator_trim = radians.clamp(-0.15, 0.15);
    }

    /// Enable/disable the auto-level altitude/pitch hold. Enabling snapshots the
    /// current altitude as the hold target.
    #[func]
    fn set_auto_level(&mut self, on: bool) {
        self.auto_level = on;
        if on {
            if let Some((sim, _)) = &self.sim {
                self.target_alt = sim.state.altitude();
            }
        }
    }

    #[func]
    fn is_auto_level(&self) -> bool {
        self.auto_level
    }

    #[func]
    fn set_target_altitude(&mut self, altitude: f64) {
        self.target_alt = altitude;
    }

    /// Re-trim to straight and level flight at `altitude` / `speed` (m/s).
    /// Returns the required `(elevator_trim_rad, throttle_trim)`.
    #[func]
    fn trim(&mut self, altitude: f64, speed: f64) -> Vector2 {
        let Some((sim, _)) = self.sim.as_mut() else {
            return Vector2::ZERO;
        };
        let (e, t) = sim.trim_level_flight(altitude, speed);
        self.elevator = 0.0;
        self.elevator_trim = e;
        self.throttle = t;
        Vector2::new(e as f32, t as f32)
    }

    /// Reset to trimmed cruise flight. Returns `(elevator_trim_rad, throttle_trim)`.
    #[func]
    fn reset(&mut self) -> Vector2 {
        let Some((sim, _)) = self.sim.as_mut() else {
            return Vector2::ZERO;
        };
        let (e, t) = sim.trim_level_flight(50.0, 60.0);
        self.avionics = Some(Self::build_avionics());
        self.avionics_active = true;
        if let Some(av) = self.avionics.as_mut() {
            av.bus_mut().cmd_throttle = t;
        }
        self.elevator = 0.0;
        self.elevator_trim = e;
        self.aileron = 0.0;
        self.rudder = 0.0;
        self.flaps_deg = 0.0;
        self.throttle = t;
        self.throttle_split = 0.0;
        self.auto_level = false;
        self.target_alt = 50.0;
        Vector2::new(e as f32, t as f32)
    }

    /// Override the wind without restarting the sim. `dir_deg` is the true
    /// bearing the wind blows **toward** (north = 0, clockwise).
    #[func]
    fn set_wind(&mut self, speed: f64, dir_deg: f64) {
        let Some((_, wind)) = self.sim.as_mut() else {
            return;
        };
        let mut cfg = wind_config_from_env();
        cfg.wind_speed = speed;
        cfg.wind_direction = dir_deg.to_radians();
        *wind = WindEnvironment::new(cfg);
    }

    /// Aircraft transform in Godot world space (Y-up). The model must be built
    /// with its nose toward local **+X**, top toward +Y, right wing toward +Z.
    #[func]
    fn get_drone_transform(&self) -> Transform3D {
        let Some((sim, _)) = &self.sim else {
            return Transform3D::IDENTITY;
        };
        let state = &sim.state;
        origin_and_basis(state)
    }

    /// Full telemetry vector for the HUD. Layout (all SI unless noted):
    ///   0  altitude m,         1  altitude ft,
    ///   2  TAS m/s,            3  TAS kts,
    ///   4  ground speed m/s,   5  IAS kts,
    ///   6  Mach,               7  dynamic pressure Pa,
    ///   8  OAT °C,             9  AoA °,
    ///   10 sideslip °,         11 pitch °,
    ///   12 roll °,             13 heading °,
    ///   14 climb angle °,      15 throttle %,
    ///   16 flaps °,            17 aileron °,
    ///   18 elevator (trimmed) °, 19 trim tab °,
    ///   20 rudder °,           21 wind speed m/s,
    ///   22 wind direction °,   23 stall flag (0/1),
    ///   24 autopilot flag (0/1).
    ///
    /// When the avionics stack is active, channels 25..=43 are appended (the
    /// noisy sensor observation, see
    /// `flight_core::avionics::AvionicsBus::sensor_observation`); HUD indices
    /// 0..=24 never change.
    #[func]
    fn telemetry(&self) -> PackedFloat64Array {
        let Some((sim, _)) = &self.sim else {
            return PackedFloat64Array::new();
        };
        let state = &sim.state;

        let alt_m = state.altitude();
        let tas_ms = state.true_airspeed(&self.last_wind);
        let gs_ms = state.airspeed();
        let atm = Atmosphere::at_altitude(alt_m);
        let ias_kts = atm.calibrated_airspeed(tas_ms) * 1.94384;
        let mach = atm.mach_number(tas_ms);
        let q_dyn = atm.dynamic_pressure(tas_ms);
        let oat = atm.temperature_c;

        let alpha = state.air_angle_of_attack(&self.last_wind).to_degrees();
        let beta = state.air_sideslip_angle(&self.last_wind).to_degrees();
        let (roll, pitch, yaw) = state.euler_angles();
        let climb = state.flight_path_angle().to_degrees();
        let wind_ms = self.last_wind.norm();
        let wind_dir = (self.last_wind.y.atan2(self.last_wind.x).to_degrees() + 360.0) % 360.0;

        let total_elev = (self.elevator + self.elevator_trim).clamp(-MAX_ELEVATOR, MAX_ELEVATOR);

        let stall = if alpha > 14.5 { 1.0 } else { 0.0 };
        let ap = if self.auto_level { 1.0 } else { 0.0 };

        // Append the noisy avionics sensor observation (25..=43) when the
        // stack is active; base HUD indices 0..=24 stay stable.
        let mut t = vec![
            alt_m,
            alt_m * 3.28084,
            tas_ms,
            tas_ms * 1.94384,
            gs_ms,
            ias_kts,
            mach,
            q_dyn,
            oat,
            alpha,
            beta,
            pitch.to_degrees(),
            roll.to_degrees(),
            (yaw.to_degrees() + 360.0) % 360.0,
            climb,
            self.throttle * 100.0,
            self.flaps_deg,
            self.aileron.to_degrees(),
            total_elev.to_degrees(),
            self.elevator_trim.to_degrees(),
            self.rudder.to_degrees(),
            wind_ms,
            wind_dir,
            stall,
            ap,
        ];
        if let Some(av) = &self.avionics {
            t.extend_from_slice(&av.bus().sensor_observation());
        }

        PackedFloat64Array::from(t)
    }
}

impl FlightSimNode {
    fn wind_environment(&self) -> WindEnvironment {
        WindEnvironment::new(wind_config_from_env())
    }

    /// Build the standard avionics hardware stack at the default 60 Hz step,
    /// initialised, mirroring the RL environment's exact component order.
    fn build_avionics() -> AvionicsSystem {
        let mut av = standard_stack(1.0 / 60.0);
        av.init();
        av
    }
}

/// Map a fault name (`"imu"`, `"gps"`, `"esc"`, …) to a [`FaultFlag`], used
/// by [`FlightSimNode::inject_fault`] / [`FlightSimNode::clear_fault`].
fn fault_flag_from_str(name: &str) -> Option<FaultFlag> {
    Some(match name {
        "imu" => FaultFlag::Imu,
        "gps" => FaultFlag::Gps,
        "baro" => FaultFlag::Baro,
        "mag" | "magnetometer" => FaultFlag::Mag,
        "airspeed" => FaultFlag::Airspeed,
        "servo_elevator" | "servo_elev" => FaultFlag::ServoElevator,
        "servo_aileron" | "servo_ail" => FaultFlag::ServoAileron,
        "servo_rudder" | "servo_rud" => FaultFlag::ServoRudder,
        "esc" => FaultFlag::Esc,
        "battery_depleted" | "battery" => FaultFlag::BatteryDepleted,
        _ => return None,
    })
}

/// Map an NED `AircraftState` to a Godot `Transform3D`. The NED→Godot frame
/// conversion is `(north, -down, east)`, and the three body axes are rotated
/// into Godot's world frame so a +X-nose model matches the physics exactly.
pub(crate) fn origin_and_basis(state: &AircraftState) -> Transform3D {
    let world = |v: NVec3<f64>| Vector3::new(v.x as f32, -v.z as f32, v.y as f32);

    let (fwd, right, down) = state.body_axes_in_earth();
    let fwd = world(fwd).normalized();
    let up = world(-down).normalized();
    let right = world(right).normalized();

    let origin = world(NVec3::new(state.pos_x, state.pos_y, state.pos_z));
    Transform3D::from_cols(fwd, up, right, origin)
}