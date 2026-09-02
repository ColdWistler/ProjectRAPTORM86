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

mod sim;
mod tunnel;
mod voxel;

use godot::prelude::*;

/// GDExtension entry point. Registers every `#[godot_class]` in this crate.
struct FlightGdExtension;

#[gdextension]
unsafe impl ExtensionLibrary for FlightGdExtension {}