//! Avionics system orchestrator.
//!
//! Wires together all registered avionics components (sensors, controllers,
//! actuators) and drives them in the correct order each simulation step.
//! Components are added via feature-gated constructors so each one compiles
//! independently.
//!
//! # Execution order (per physics step)
//! 1. Write true aircraft state from physics into the bus
//! 2. Each sensor samples at its own rate (may skip frames)
//! 3. Flight controller reads sensors, writes servo/ESC commands
//! 4. Actuators/device components (servos, ESC, battery) write actual outputs
//! 5. Caller reads actuator outputs and feeds them to physics
//!
//! # Failure injection
//! Components self-check the shared `fc_fault_flags` register (see
//! [`FaultFlag`]), so injecting a fault is a single flag write and the
//! failing device drives its own failure behaviour. Use
//! [`AvionicsSystem::inject_fault`] to inject faults from outside (e.g. the
//! RL training environment).
//!
//! # Usage
//! ```rust,ignore
//! let mut avionics = AvionicsSystem::new(1.0 / 60.0);
//! // Register components behind feature flags
//! #[cfg(feature = "imu")]
//! avionics.add_sensor(Box::new(ImuSensor::new(imu_config)));
//! // ...
//!
//! // Each physics frame:
//! avionics.write_true_state(&aircraft_state, &wind_earth, q_dynamic);
//! avionics.step();  // dt fixed at construction
//! let (elev, ail, rud, thr) = avionics.bus().read_actuator_outputs();
//! ```

use super::bus::{AvionicsBus, FaultFlag, FaultFlags, FcMode};
use super::traits::{Actuator, AvionicsComponent, Controller, Sensor};

/// The avionics system: owns the bus and all registered components.
pub struct AvionicsSystem {
    bus: AvionicsBus,
    sensors: Vec<Box<dyn Sensor>>,
    controllers: Vec<Box<dyn Controller>>,
    actuators: Vec<Box<dyn Actuator>>,
    devices: Vec<Box<dyn AvionicsComponent>>,
    dt: f64,
}

impl AvionicsSystem {
    /// Create a new avionics system with the given physics time step.
    pub fn new(dt: f64) -> Self {
        Self {
            bus: AvionicsBus::default(),
            sensors: Vec::new(),
            controllers: Vec::new(),
            actuators: Vec::new(),
            devices: Vec::new(),
            dt,
        }
    }

    /// Borrow the bus for reading (e.g. to extract observations for the RL agent).
    pub fn bus(&self) -> &AvionicsBus {
        &self.bus
    }

    /// Mutably borrow the bus (for direct agent command injection).
    pub fn bus_mut(&mut self) -> &mut AvionicsBus {
        &mut self.bus
    }

    /// Register a sensor component.
    pub fn add_sensor(&mut self, sensor: Box<dyn Sensor>) {
        tracing_log(&format!("[Avionics] registered sensor: {}", sensor.name()));
        self.sensors.push(sensor);
    }

    /// Register a flight controller component.
    pub fn add_controller(&mut self, controller: Box<dyn Controller>) {
        tracing_log(&format!(
            "[Avionics] registered controller: {}",
            controller.name()
        ));
        self.controllers.push(controller);
    }

    /// Register an actuator component.
    pub fn add_actuator(&mut self, actuator: Box<dyn Actuator>) {
        tracing_log(&format!(
            "[Avionics] registered actuator: {}",
            actuator.name()
        ));
        self.actuators.push(actuator);
    }

    /// Register a generic component that is neither a sensor, controller nor
    /// actuator (e.g. the ESC and battery). Devices run after actuators each
    /// step. `add_sensor`/`add_controller`/`add_actuator` remain available for
    /// the trait-specific stages.
    pub fn add_component(&mut self, device: Box<dyn AvionicsComponent>) {
        tracing_log(&format!("[Avionics] registered component: {}", device.name()));
        self.devices.push(device);
    }

    /// Set (`on = true`) or clear (`on = false`) a fault in the shared
    /// register. Components self-check the register and apply their own
    /// failure behaviour.
    pub fn inject_fault(&mut self, flag: FaultFlag, on: bool) {
        self.bus.fc_fault_flags.set(flag, on);
    }

    /// Snapshot of the currently injected faults.
    pub fn faults(&self) -> FaultFlags {
        self.bus.fc_fault_flags
    }

    /// Initialize all registered components.
    pub fn init(&mut self) {
        for sensor in &mut self.sensors {
            sensor.init(self.dt);
        }
        for ctrl in &mut self.controllers {
            ctrl.init(self.dt);
        }
        for act in &mut self.actuators {
            act.init(self.dt);
        }
        for dev in &mut self.devices {
            dev.init(self.dt);
        }
        tracing_log("[Avionics] system initialized");
    }

    /// Run one avionics step. Call this after the physics step and before
    /// reading actuator outputs.
    ///
    /// # Execution order
    /// 1. Sensors sample at their own rates (skip frames as needed)
    /// 2. Controllers compute servo/ESC commands from sensor data
    /// 3. Actuators process commands into actual positions
    /// 4. Devices (ESC, battery) finalize power/output state
    pub fn step(&mut self) {
        // 1. Sensors: each samples at its own rate.
        // Sensors handle their own internal timing; we call step() every
        // frame and the sensor decides whether to actually sample.
        for i in 0..self.sensors.len() {
            self.sensors[i].step(&mut self.bus, self.dt);
        }

        // 2. Controllers
        for ctrl in &mut self.controllers {
            ctrl.step(&mut self.bus, self.dt);
        }

        // 3. Actuators
        for act in &mut self.actuators {
            act.step(&mut self.bus, self.dt);
        }

        // 4. Devices
        for dev in &mut self.devices {
            dev.step(&mut self.bus, self.dt);
        }

        self.bus.sim_time += self.dt;
    }

    /// Reset all components to initial state.
    pub fn reset(&mut self) {
        for sensor in &mut self.sensors {
            sensor.reset();
        }
        for ctrl in &mut self.controllers {
            ctrl.reset();
        }
        for act in &mut self.actuators {
            act.reset();
        }
        for dev in &mut self.devices {
            dev.reset();
        }
        self.bus.clear_sensor_outputs();
        self.bus.sim_time = 0.0;
        tracing_log("[Avionics] system reset");
    }

    /// Get the current flight controller mode.
    pub fn fc_mode(&self) -> FcMode {
        self.bus.fc_mode
    }

    /// Number of registered sensors.
    pub fn sensor_count(&self) -> usize {
        self.sensors.len()
    }
}

/// Minimal logging without pulling in the `tracing` crate.
fn tracing_log(msg: &str) {
    // Use eprintln! for now; the Godot GDExtension bridge captures stderr.
    // A future iteration can wire this to godot_warn! or tracing.
    eprintln!("{msg}");
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::avionics::AvionicsComponent;
    use nalgebra::Vector3;

    /// A passthrough sensor for testing: copies true angular rate to gyro output.
    struct TestGyro {
        last_sample: f64,
    }

    impl TestGyro {
        fn new() -> Self {
            Self { last_sample: 0.0 }
        }
    }

    impl AvionicsComponent for TestGyro {
        fn name(&self) -> &str {
            "TestGyro"
        }
        fn init(&mut self, _dt: f64) {}
        fn step(&mut self, bus: &mut AvionicsBus, _dt: f64) {
            bus.gyro = bus.true_angular_rates;
        }
        fn reset(&mut self) {
            self.last_sample = 0.0;
        }
    }

    impl Sensor for TestGyro {
        fn update_rate_hz(&self) -> f64 {
            400.0
        }
    }

    /// A passthrough actuator for testing: copies its servo command to output.
    struct TestServo {
        command: f64,
    }

    impl TestServo {
        fn new() -> Self {
            Self { command: 0.0 }
        }
    }

    impl AvionicsComponent for TestServo {
        fn name(&self) -> &str {
            "TestServo"
        }
        fn init(&mut self, _dt: f64) {}
        fn step(&mut self, bus: &mut AvionicsBus, _dt: f64) {
            // Real servos read their commanded position from the FC outputs.
            self.command = bus.fc_servo_elevator;
            bus.actual_elevator_deg = self.command;
        }
        fn reset(&mut self) {
            self.command = 0.0;
        }
    }

    impl Actuator for TestServo {
        fn command(&mut self, target: f64) {
            self.command = target;
        }
        fn actual_output(&self) -> f64 {
            self.command
        }
    }

    #[test]
    fn system_initializes_with_no_components() {
        let mut sys = AvionicsSystem::new(1.0 / 60.0);
        sys.init();
        assert_eq!(sys.sensor_count(), 0);
        assert_eq!(sys.bus().sim_time, 0.0);
    }

    #[test]
    fn sensor_writes_to_bus() {
        let mut sys = AvionicsSystem::new(1.0 / 60.0);
        sys.add_sensor(Box::new(TestGyro::new()));
        sys.init();

        // Set true angular rates on the bus
        sys.bus_mut().true_angular_rates = Vector3::new(0.1, 0.2, 0.3);

        // Step should copy true rates to gyro output
        sys.step();
        let gyro = sys.bus().gyro;
        assert!((gyro.x - 0.1).abs() < 1e-9);
        assert!((gyro.y - 0.2).abs() < 1e-9);
        assert!((gyro.z - 0.3).abs() < 1e-9);
    }

    #[test]
    fn actuator_writes_to_bus() {
        let mut sys = AvionicsSystem::new(1.0 / 60.0);
        sys.add_actuator(Box::new(TestServo::new()));
        sys.init();

        sys.bus_mut().fc_servo_elevator = 5.0;
        sys.step();
        assert!((sys.bus().actual_elevator_deg - 5.0).abs() < 1e-9);
    }

    #[test]
    fn read_actuator_outputs_converts_to_radians() {
        let mut sys = AvionicsSystem::new(1.0 / 60.0);
        sys.add_actuator(Box::new(TestServo::new()));
        sys.init();

        sys.bus_mut().fc_servo_elevator = 10.0;
        sys.step();
        let (elev, _, _, _) = sys.bus().read_actuator_outputs();
        assert!((elev - 10.0_f64.to_radians()).abs() < 1e-9);
    }

    #[test]
    fn reset_clears_state() {
        let mut sys = AvionicsSystem::new(1.0 / 60.0);
        sys.add_sensor(Box::new(TestGyro::new()));
        sys.init();

        sys.bus_mut().true_angular_rates = Vector3::new(1.0, 2.0, 3.0);
        sys.step();
        assert!(sys.bus().gyro.norm() > 0.0);

        sys.reset();
        assert_eq!(sys.bus().sim_time, 0.0);
    }
}
