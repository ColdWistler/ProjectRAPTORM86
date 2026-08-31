# ProjectRAPTORM86

A Rust workspace implementing a **6-DOF fixed-wing aircraft flight dynamics
engine**, exposed to **Godot 4** through a GDExtension (`godot-rust`) so the
physics run in Rust while Godot handles all assets and visualization.

## Architecture

```
┌────────────────────────────────────────────────────────────┐
│  flight_core  (pure Rust physics, no engine deps)          │
│   • quaternion 6-DOF rigid-body dynamics + RK4 integrator  │
│   • 1976 US Standard Atmosphere                            │
│   • nonlinear post-stall aerodynamics                      │
│   • steady wind / wind shear / Dryden turbulence           │
│   • level-flight trimming + autopilot assist               │
└──────────────────────────┬─────────────────────────────────┘
                           │ GDExtension (flight_gd, godot-rust)
┌──────────────────────────┴─────────────────────────────────┐
│  flight_gd  (Rust GDExtension dylib)                       │
│   • FlightSimNode  — 6-DOF simulation node (step, trims,   │
│     controls, NED→Godot transform, full HUD telemetry)     │
│   • WindTunnelNode — fixed-aircraft flow-field smoke       │
│     (solid-body deflection, wing circulation driven by the │
│     physics CL, tip vortices, turbulent wake, top rake,    │
│     pusher-prop slipstream) + aero forces for the HUD      │
└──────────────────────────┬─────────────────────────────────┘
                           │ godot project (godot/)
┌──────────────────────────┴─────────────────────────────────┐
│  Godot 4.7  (assets & visualization only)                  │
│   • scenes: flight_sim.tscn, wind_tunnel.tscn              │
│   • procedural drone model (DroneFactory)                  │
│   • chase camera / orbit camera, sky, terrain, runway, HUD │
│   • single-MultiMesh smoke renderer (one draw call)        │
└────────────────────────────────────────────────────────────┘
```

## Workspace Layout

| Path            | Description                                                                                                            |
| --------------- | ----------------------------------------------------------------------------------------------------------------------- |
| `flight_core/`  | Pure-Rust physics engine: quaternion 6-DOF dynamics, 1976 US Standard Atmosphere, nonlinear post-stall aerodynamics, RK4 integrator, trim/autopilot logic. |
| `flight_gd/`    | `godot-rust` GDExtension crate exposing the physics as native Godot nodes (`FlightSimNode`, `WindTunnelNode`).        |
| `godot/`        | Godot 4.7 project — scenes, the procedural drone model, camera, HUD, sky/terrain, and the smoke MultiMesh renderer.     |
| `aircraft.toml` | Aircraft geometry / mass / aero coefficients consumed by `flight_core`.                                                |

## Features

- Full 6-DOF rigid-body simulation (elevator, aileron, rudder, throttle, flaps)
- Trailing-edge flap aerodynamics: lift, drag, and nose-down pitching-moment increments with stall-angle reduction
- Atmospheric wind simulation: steady wind, altitude wind shear, and Dryden-style turbulence — aerodynamics act on the true airspeed (air-relative velocity), while the trajectory integrates ground speed
- Quaternion-based attitude representation
- 1976 US Standard Atmosphere (Mach, calibrated airspeed, dynamic pressure)
- Nonlinear post-stall aerodynamics with configurable stall angles
- RK4 numerical integration at 60 Hz
- Level-flight trimming and altitude/flight-path autopilot assist
- Programmable aircraft geometry exposed via `aircraft.toml`
- Wind-tunnel smoke visualization whose flow field is driven by the *same*
  lift coefficient the physics computes (upwash/downwash & tip vortices react
  to actual AoA)
- Interactive wind-tunnel: pitch/roll/yaw the fixed aircraft, tune wind,
  orbit camera, read the computed forces

## Requirements

- Rust (edition 2021) — install via [rustup](https://rustup.rs)
- A matching Rust toolchain supports `godot` 0.5.x (MSRV 1.94)
- [Godot 4.7](https://godotengine.org/download/linux/) (4.2+ minimum)
- A Vulkan-capable GPU

## Building the Extension

```bash
cargo build -p flight_gd --release
mkdir -p godot/bin
cp target/release/libflight_gd.so godot/bin/libflight_gd.release.so
```

For a debug extension (used by the official editor build):

```bash
cargo build -p flight_gd
mkdir -p godot/bin
cp target/debug/libflight_gd.so godot/bin/libflight_gd.debug.so
```

## Running the Simulator

1. Build and copy the extension as above.
2. Open the project in Godot (import `godot/project.godot`), or run directly:

```bash
godot --path godot
```

The project boots into a main menu (`main_menu.tscn`) offering the Flight
Simulator and Wind Tunnel modes. Flight Sim starts a trimmed cruise flight at
1000 m / 60 m/s. The `aircraft.toml` is located automatically relative to the
project directory. `Esc` in either mode returns to the main menu.

## Controls — Flight Sim

| Action      | Control                        |
| ----------- | ------------------------------ |
| Pitch       | `W` nose down / `S` nose up (or `↑` / `↓`) |
| Roll        | `A` left / `D` right (or `←` / `→`)       |
| Rudder      | `Q` left / `E` right (or `Z` / `C`)        |
| Flaps       | `F` cycle 0° → 15° → 30°                   |
| Trim        | `[` nose down / `]` nose up                |
| Throttle    | `Shift` up / `Ctrl` down                   |
| Autopilot   | `H` or `T` toggle level-flight hold        |
| Reset       | `R` reset to trimmed cruise flight         |
| Wind tunnel | `Esc`                                      |

## Controls — Wind Tunnel

Pick **Wind Tunnel** from the main menu, or open `scenes/wind_tunnel.tscn`
directly. It drops into an empty test chamber. The aircraft is held fixed at
the origin while a field of volumetric smoke puffs flows over/around the airframe;
the streamlines react to the actual `flight_core` lift coefficient (pitch up →
stronger upwash/downwash and wing-tip vortices). The HUD reports computed
lift, drag, side force, moments and CL.

| Action         | Control               |
| -------------- | --------------------- |
| Orbit camera   | Click + drag mouse    |
| Zoom camera    | Mouse scroll wheel    |
| Pitch          | `W` up / `S` down     |
| Yaw (sideslip) | `A` / `D`             |
| Roll           | `↑` / `↓`             |
| Aileron        | `Q` left / `E` right  |
| Rudder         | `Z` left / `C` right  |
| Flaps          | `F` cycle             |
| Wind speed     | `Shift` up / `Ctrl` down |
| Wind direction | `R` / `T`             |
| Reset          | `Space`               |
| Menu           | `Esc`                 |

## Flight Tests

```bash
cargo test -p flight_core
```

The suite verifies trimmed level flight, flap aerodynamics, control-derivative
sign conventions, autopilot/trim math, and roll/yaw coupling behaviour.

## Aircraft Configuration

Mass properties, moments of inertia, and aerodynamic coefficients live in
`aircraft.toml`:

```toml
mass = 1100.0       # kg
wing_area = 16.2    # m^2
cla = 5.5           # lift curve slope (per radian)
cmq = -20.0         # pitch damping derivative
thrust_max = 1800.0 # static (low-speed) thrust, N
power_max = 119000  # engine shaft power, W (thrust ~ P/V above corner speed)
cl_flap = 1.10      # flap lift increment (per radian of deflection)
cd_flap = 0.14      # flap drag increment (per radian of deflection)
cm_flap = -0.20     # flap pitching moment (per radian, nose-down)
```

Throttle is modeled as a constant-power propeller: above the corner speed the
available thrust falls off as `P_max / V`, which caps the airspeed and gives
the phugoid its characteristic damped, energy-exchanging climb.

## Using `flight_core` as a Library

```rust
use flight_core::Simulator;

let mut sim = Simulator::new("aircraft.toml");
let (elev_trim, throttle_trim) = sim.trim_level_flight(1000.0, 60.0);

// One 60 Hz physics step at current trim (still air: wind = None)
let obs = sim.step_6dof(elev_trim, 0.0, 0.0, throttle_trim, 0.0, None, 1.0 / 60.0);
```

The returned observation array contains position, velocity, attitude,
angular rates, and more — suitable for reinforcement-learning training loops.
`step_6dof` takes `(elevator, aileron, rudder, throttle, flaps, wind, dt)` where
`flaps` is the trailing-edge flap deflection in radians and `wind` is an
optional `&Vector3<f64>` giving the wind in the Earth NED frame (m/s);
pass `None` for still air. Aerodynamics act on the air-relative velocity.

## Wind simulation

The RL `Environment` (`env.rs`) exposes wind through
[`EnvConfig::wind_config`](flight_core/src/env.rs). The Godot simulator reads
wind from environment variables so it can be tuned without recompiling:

| Variable               | Meaning                                        | Default    |
|------------------------|------------------------------------------------|------------|
| `RAPTOR_WIND_SPEED`    | Steady wind speed (m/s)                        | `0` (still)|
| `RAPTOR_WIND_DIR_DEG`  | True bearing the wind blows **toward** (deg)   | `0`        |
| `RAPTOR_WIND_SHEAR`    | `1` enables altitude (boundary-layer) shear    | off        |
| `RAPTOR_TURBULENCE`    | `light` / `moderate` / `severe`                | `light`    |

Example — a 12 m/s crosswind with moderate turbulence:

```bash
RAPTOR_WIND_SPEED=12 RAPTOR_WIND_DIR_DEG=90 RAPTOR_TURBULENCE=moderate godot --path godot
```