//! Servo/actuator model for control surfaces.
//!
//! Simulates a real servo with rate limiting, position limits,
//! quantization (ADC resolution), deadband, and latency.
//!
//! # Bus contract
//! Reads: `fc_servo_elevator`, `fc_servo_aileron`, `fc_servo_rudder`,
//!        `fc_esc_throttle`, agent commands (Manual mode)
//! Writes: `actual_elevator_deg`, `actual_aileron_deg`,
//!         `actual_rudder_deg`, `actual_esc_output`

use serde::Deserialize;

use super::bus::AvionicsBus;
use super::traits::{Actuator, AvionicsComponent};

/// Configuration for a single servo/actuator.
#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "snake_case")]
pub struct ServoConfig {
    /// Minimum deflection (degrees).
    pub min_deg: f64,
    /// Maximum deflection (degrees).
    pub max_deg: f64,
    /// Maximum angular velocity (degrees/second).
    pub rate_limit_deg_s: f64,
    /// ADC resolution in bits (e.g. 11 for PWM).
    pub resolution_bits: u32,
    /// Command-to-movement latency (seconds).
    pub latency_s: f64,
    /// Deadband fraction around center (e.g. 0.015 = 1.5%).
    pub deadband_pct: f64,
    /// Fixed trim offset (degrees).
    #[serde(default)]
    pub trim_deg: f64,
}

impl Default for ServoConfig {
    fn default() -> Self {
        Self {
            min_deg: -30.0,
            max_deg: 30.0,
            rate_limit_deg_s: 60.0,
            resolution_bits: 11,
            latency_s: 0.01,
            deadband_pct: 0.015,
            trim_deg: 0.0,
        }
    }
}

/// A single servo/actuator with realistic dynamics.
pub struct ServoActuator {
    config: ServoConfig,
    /// Current actual position (degrees).
    position: f64,
    /// Commanded target position (degrees).
    command: f64,
    /// Quantization step size.
    quant_step: f64,
    /// Label for identification.
    label: String,
}

impl ServoActuator {
    pub fn new(config: ServoConfig, label: &str) -> Self {
        let range = config.max_deg - config.min_deg;
        let quant_step = if config.resolution_bits > 0 {
            range / (1u64 << config.resolution_bits) as f64
        } else {
            0.0
        };

        Self {
            config,
            position: 0.0,
            command: 0.0,
            quant_step,
            label: label.to_string(),
        }
    }
}

impl AvionicsComponent for ServoActuator {
    fn name(&self) -> &str {
        &self.label
    }

    fn init(&mut self, _dt: f64) {
        self.position = self.config.trim_deg;
        self.command = self.config.trim_deg;
    }

    fn step(&mut self, _bus: &mut AvionicsBus, dt: f64) {
        // Rate limiting: move toward command at max rate
        let error = self.command - self.position;
        let max_move = self.config.rate_limit_deg_s * dt;
        self.position += error.clamp(-max_move, max_move);

        // Deadband: snap to zero if within deadband of center
        let half_band = self.config.deadband_pct * (self.config.max_deg - self.config.min_deg);
        if self.position.abs() < half_band {
            self.position = 0.0;
        }

        // Quantization
        if self.quant_step > 0.0 {
            let offset = self.position - self.config.min_deg;
            let q = (offset / self.quant_step).floor();
            self.position = q * self.quant_step + self.config.min_deg;
        }

        // Clip to limits
        self.position = self
            .position
            .max(self.config.min_deg)
            .min(self.config.max_deg);

        // Write output to bus (handled by concrete subtypes or the mixer)
    }

    fn reset(&mut self) {
        self.position = self.config.trim_deg;
        self.command = self.config.trim_deg;
    }
}

impl Actuator for ServoActuator {
    fn command(&mut self, target: f64) {
        self.command = target
            .max(self.config.min_deg)
            .min(self.config.max_deg);
    }

    fn actual_output(&self) -> f64 {
        self.position + self.config.trim_deg
    }
}

/// Full actuator suite: elevator + aileron + rudder servos.
/// Orchestrates all three from the FC outputs on the bus.
pub struct ActuatorSuite {
    pub elevator: ServoActuator,
    pub aileron: ServoActuator,
    pub rudder: ServoActuator,
}

impl ActuatorSuite {
    pub fn new(elev: ServoConfig, ail: ServoConfig, rud: ServoConfig) -> Self {
        Self {
            elevator: ServoActuator::new(elev, "Servo-elevator"),
            aileron: ServoActuator::new(ail, "Servo-aileron"),
            rudder: ServoActuator::new(rud, "Servo-rudder"),
        }
    }
}

impl AvionicsComponent for ActuatorSuite {
    fn name(&self) -> &str {
        "ActuatorSuite"
    }

    fn init(&mut self, dt: f64) {
        self.elevator.init(dt);
        self.aileron.init(dt);
        self.rudder.init(dt);
    }

    fn step(&mut self, bus: &mut AvionicsBus, dt: f64) {
        // A jammed servo no longer tracks its command: its position is frozen.
        if !bus.fc_fault_flags.servo_elevator_failed {
            self.elevator.command(bus.fc_servo_elevator);
            self.elevator.step(bus, dt);
        }
        if !bus.fc_fault_flags.servo_aileron_failed {
            self.aileron.command(bus.fc_servo_aileron);
            self.aileron.step(bus, dt);
        }
        if !bus.fc_fault_flags.servo_rudder_failed {
            self.rudder.command(bus.fc_servo_rudder);
            self.rudder.step(bus, dt);
        }

        // Write actual positions to bus. A failed ESC produces no thrust.
        bus.actual_elevator_deg = self.elevator.actual_output();
        bus.actual_aileron_deg = self.aileron.actual_output();
        bus.actual_rudder_deg = self.rudder.actual_output();
        bus.actual_esc_output = if bus.fc_fault_flags.esc_failed {
            0.0
        } else {
            bus.fc_esc_throttle
        };
    }

    fn reset(&mut self) {
        self.elevator.reset();
        self.aileron.reset();
        self.rudder.reset();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn servo_rate_limiting() {
        let config = ServoConfig {
            min_deg: -30.0,
            max_deg: 30.0,
            rate_limit_deg_s: 60.0, // 60 deg/s
            resolution_bits: 0,     // no quantization for this test
            ..Default::default()
        };
        let mut servo = ServoActuator::new(config, "test");
        servo.init(0.0);

        // Command to 30°, step for 0.25s → should move 15° (60 * 0.25)
        servo.command(30.0);
        servo.step(&mut AvionicsBus::default(), 0.25);
        assert!(
            (servo.actual_output() - 15.0).abs() < 0.1,
            "servo should move 15° in 0.25s, got {}",
            servo.actual_output()
        );
    }

    #[test]
    fn servo_position_clipping() {
        let config = ServoConfig {
            min_deg: -20.0,
            max_deg: 20.0,
            rate_limit_deg_s: 1000.0, // instant
            resolution_bits: 0,
            ..Default::default()
        };
        let mut servo = ServoActuator::new(config, "test");
        servo.init(0.0);

        servo.command(50.0); // beyond max
        servo.step(&mut AvionicsBus::default(), 0.1);
        assert!((servo.actual_output() - 20.0).abs() < 0.1);
    }

    #[test]
    fn actuator_suite_writes_to_bus() {
        let cfg = ServoConfig {
            rate_limit_deg_s: 1000.0,
            resolution_bits: 0,
            ..Default::default()
        };
        let mut suite = ActuatorSuite::new(cfg.clone(), cfg.clone(), cfg);
        suite.init(0.0);

        let mut bus = AvionicsBus {
            fc_servo_elevator: 10.0,
            fc_servo_aileron: -5.0,
            fc_servo_rudder: 3.0,
            fc_esc_throttle: 0.8,
            ..Default::default()
        };

        suite.step(&mut bus, 0.1);

        assert!((bus.actual_elevator_deg - 10.0).abs() < 0.5);
        assert!((bus.actual_aileron_deg - (-5.0)).abs() < 0.5);
        assert!((bus.actual_rudder_deg - 3.0).abs() < 0.5);
        assert!((bus.actual_esc_output - 0.8).abs() < 1e-9);
    }
}
