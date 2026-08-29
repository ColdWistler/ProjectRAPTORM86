# ProjectRAPTORM86

A Rust workspace implementing a **6-DOF fixed-wing aircraft flight dynamics
engine** and a real-time **Bevy 3D flight simulator** for a tactical UAV drone.

## Workspace Layout

| Crate        | Description                                                                                                            |
| ------------ | ---------------------------------------------------------------------------------------------------------------------- |
| `flight_core`| Pure-Rust physics engine: quaternion rigid-body dynamics, 1976 US Standard Atmosphere, nonlinear post-stall aerodynamics, and an RK4 integrator. |
| `flight_vis` | Interactive Bevy visualizer: procedural terrain, airport, city, clouds, animated control surfaces, chase camera, and a live telemetry HUD. |

## Features

- Full 6-DOF rigid-body simulation (elevator, aileron, rudder, throttle, flaps)
- Trailing-edge flap aerodynamics: lift, drag, and nose-down pitching-moment increments with stall-angle reduction
- Quaternion-based attitude representation
- 1976 US Standard Atmosphere (Mach, calibrated airspeed, dynamic pressure)
- Nonlinear post-stall aerodynamics with configurable stall angles
- RK4 numerical integration at 60 Hz
- Level-flight trimming and altitude/flight-path autopilot assist
- Programmable aircraft geometry exposed via `aircraft.toml`

## Requirements

- Rust (edition 2021) — install via [rustup](https://rustup.rs)
- Bevy 0.15 requires the usual OS graphics/audio development libraries
  (on Linux: `libasound2-dev`, `libudev-dev`, and a Vulkan-capable GPU)

## Getting Started

Run the visual simulator:

```bash
cargo run -p flight_vis
```

Run the physics test suite:

```bash
cargo test -p flight_core
```

The simulator loads its aircraft configuration from `aircraft.toml`
(falls back to `../aircraft.toml` if run from inside a subcrate).

## Controls

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

// One 60 Hz physics step at current trim
let obs = sim.step_6dof(elev_trim, 0.0, 0.0, throttle_trim, 0.0, 1.0 / 60.0);
```

The returned observation array contains position, velocity, attitude,
angular rates, and more — suitable for reinforcement-learning training loops.
`step_6dof` takes `(elevator, aileron, rudder, throttle, flaps, dt)` where
`flaps` is the trailing-edge flap deflection in radians.