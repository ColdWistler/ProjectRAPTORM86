//! # flight_gd
//!
//! Godot 4 (GDExtension) bridge for the `flight_core` 6-DOF flight dynamics
//! engine. Rust owns the physics; Godot handles all assets and visualization.
//!
//! Exposed engine classes (each a `Node3D`):
//! * `FlightSimNode` — interactive 6-DOF flight simulation (manual controls,
//!   autopilot hold, full HUD telemetry, NED→Godot transform).
//! * `WindTunnelNode` — fixed-aircraft wind-tunnel flow field: physics-driven
//!   smoke-streak advection plus the aero forces/moments for the HUD.
//! * `WeatherSystem` — RL weather plugin: seeded turbulence/precipitation/
//!   visibility/updraft scene, agent observation vector, reward penalties,
//!   extreme-weather termination, curriculum learning and the `weather_changed`
//!   signal. See the module docs in `weather.rs` for the GDScript surface.
//!
//! # Conventions
//! Godot world is **Y-up**; NED Earth frame maps to world
//! `(north, -down, east)`. `flight_core` body frame is nose +X / right +Y /
//! down +Z; the Godot visual models use nose +X / up +Y / right +Z, converted
//! by `origin_and_basis`.
//!
//! # Standards & Traceability
//! Physics semantics (units, frames, derivative conventions) are inherited
//! verbatim from `flight_core`; see the crate documentation there for the
//! underlying standards (NASA SP-747, MIL-F-8785C, Viterna-Corrigan, RK4).

mod sim;
mod tunnel;
mod voxel;
mod weather;

use godot::prelude::*;

/// GDExtension entry point. Registers every `#[godot_class]` in this crate.
struct FlightGdExtension;

#[gdextension]
unsafe impl ExtensionLibrary for FlightGdExtension {}
