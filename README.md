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
- Atmospheric wind simulation: steady wind, altitude wind shear, and Dryden-style turbulence — aerodynamics act on the true airspeed (air-relative velocity), while the trajectory integrates ground speed
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

## Wind Tunnel Mode

Pick **Wind Tunnel Simulator** from the main menu. The surrounding world is
hidden and the aircraft is held fixed at the center of an empty test chamber
while a field of streak particles flows over and around it, visualizing how
the airstream deflects around the airframe — like a real wind tunnel. The HUD
reports the aerodynamic lift, drag, side force, and moments computed by
`flight_core` for the current attitude.

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
| Exit           | `Esc`                 |

The airstream direction/speed shown by the particles and the resulting forces
react to the aircraft's orientation relative to the fixed flow.

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

### Wind simulation

The RL `Environment` (`env.rs`) exposes wind through
[`EnvConfig::wind_config`](flight_core/src/env.rs). The visual simulator reads
wind from environment variables so it can be tuned without recompiling:

| Variable               | Meaning                                        | Default    |
|------------------------|------------------------------------------------|------------|
| `RAPTOR_WIND_SPEED`    | Steady wind speed (m/s)                        | `0` (still)|
| `RAPTOR_WIND_DIR_DEG`  | True bearing the wind blows **toward** (deg)   | `0`        |
| `RAPTOR_WIND_SHEAR`    | `1` enables altitude (boundary-layer) shear    | off        |
| `RAPTOR_TURBULENCE`    | `light` / `moderate` / `severe`                | `light`    |

Example — a 12 m/s crosswind with moderate turbulence:

```bash
RAPTOR_WIND_SPEED=12 RAPTOR_WIND_DIR_DEG=90 RAPTOR_TURBULENCE=moderate cargo run -p flight_vis
```