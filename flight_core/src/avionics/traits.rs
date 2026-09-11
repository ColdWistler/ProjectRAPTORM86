//! Traits defining the avionics component contracts.
//!
//! Every avionics component (sensor, actuator, controller) implements
//! one of these traits. Components communicate only through
//! [`AvionicsBus`]; they never import or call each other.
//!
//! This design ensures each component can be developed, tested, and
//! compiled independently behind its own Cargo feature flag.

use super::bus::AvionicsBus;

/// Base trait for all avionics components.
///
/// Every component must implement `name()` (for logging/debugging),
/// `init()` (one-time setup with the physics time step), `step()` (per-frame
/// update reading from and writing to the bus), and `reset()` (return to
/// initial state).
pub trait AvionicsComponent {
    /// Human-readable name for logging (e.g. "IMU", "GPS", "Servo-elevator").
    fn name(&self) -> &str;

    /// One-time initialization with the physics time step `dt` (seconds).
    /// Called once before the simulation loop begins.
    fn init(&mut self, dt: f64);

    /// Per-frame update. Read the fields you need from `bus`, compute your
    /// output, and write it back. `dt` is the physics time step (seconds).
    fn step(&mut self, bus: &mut AvionicsBus, dt: f64);

    /// Reset internal state (integrators, drift accumulators, lag buffers)
    /// back to power-on defaults.
    fn reset(&mut self);
}

/// Trait for sensors: devices that measure the true aircraft state and
/// produce noisy/degraded readings on the bus.
///
/// Sensors are called by the orchestrator at their own update rate,
/// not every physics frame. The orchestrator calls `should_sample()`
/// to decide whether to invoke `step()` this frame.
pub trait Sensor: AvionicsComponent {
    /// Sensor update rate in Hz (e.g. 400 for IMU, 10 for GPS).
    fn update_rate_hz(&self) -> f64;

    /// Returns `true` if the sensor should produce a new sample at the
    /// given simulation time. Default implementation compares elapsed
    /// time since last sample against the sample period.
    fn should_sample(&self, sim_time: f64, last_sample_time: f64) -> bool {
        let period = 1.0 / self.update_rate_hz();
        (sim_time - last_sample_time) >= period - 1e-9
    }
}

/// Trait for actuators: devices that receive a commanded position and
/// produce an actual (rate-limited, quantized) position.
pub trait Actuator: AvionicsComponent {
    /// Set the commanded target position (degrees for servos, 0–1 for ESC).
    fn command(&mut self, target: f64);

    /// Read the actual output position after rate limiting and quantization.
    fn actual_output(&self) -> f64;
}

/// Trait for flight controllers: compute servo/ESC commands from
/// sensor readings and agent commands.
pub trait Controller: AvionicsComponent {
    /// Current flight controller mode.
    fn mode(&self) -> super::bus::FcMode;

    /// Switch the flight controller mode.
    fn set_mode(&mut self, mode: super::bus::FcMode);
}
