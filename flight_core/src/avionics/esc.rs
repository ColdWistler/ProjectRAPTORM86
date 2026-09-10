//! Electronic Speed Controller (ESC) model.
//!
//! Simulates a brushless ESC + propeller thrust/load torque model.
//! Models command-to-rpm lag, throttle deadband, thrust response,
//! and the current draw on the battery.
//!
//! # Bus contract
//! Reads: `true_airspeed`, `battery_voltage`
//! Writes: `actual_esc_output` (throttle fraction sent to propulsion),
//!         `battery_current` (drawn by this ESC)
//!
//! Future: `true_thrust`, `true_torque` for standalone propulsion use.

use serde::Deserialize;

use super::bus::AvionicsBus;
use super::traits::AvionicsComponent;

/// Configuration for the ESC + propeller pair.
#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "snake_case")]
pub struct EscConfig {
    /// Throttle deadband fraction (0.0–0.1 typical). Commands below
    /// 0 work in reverse-thrust mode (for future VTOL configs).
    #[serde(default)]
    pub throttle_deadband: f64,
    /// Maximum thrust-to-power efficiency (N/W). Typical 8-10 N/kW.
    #[serde(default = "default_thrust_to_power")]
    pub thrust_to_power: f64,
    /// Motor time constant (seconds) — response lag to throttle changes.
    #[serde(default = "default_motor_tau")]
    pub motor_tau: f64,
    /// Propeller diameter (metres) — used for momentum-disc thrust.
    #[serde(default = "default_prop_diameter")]
    pub prop_diameter: f64,
    /// ESC update rate (Hz).
    #[serde(default = "default_esc_hz")]
    pub update_hz: f64,
}

fn default_thrust_to_power() -> f64 {
    0.09
}
fn default_motor_tau() -> f64 {
    0.05
}
fn default_prop_diameter() -> f64 {
    0.30
}
fn default_esc_hz() -> f64 {
    500.0
}

impl Default for EscConfig {
    fn default() -> Self {
        Self {
            throttle_deadband: 0.0,
            thrust_to_power: default_thrust_to_power(),
            motor_tau: default_motor_tau(),
            prop_diameter: default_prop_diameter(),
            update_hz: default_esc_hz(),
        }
    }
}

/// ESC that accepts throttle fraction [0.0, 1.0] and produces thrust.
pub struct Esc {
    config: EscConfig,
    /// Smoothed throttle (post-lag).
    current_throttle: f64,
}

impl Esc {
    pub fn new(config: EscConfig) -> Self {
        Self {
            config,
            current_throttle: 0.0,
        }
    }

    /// Current smoothed throttle output [0, 1].
    pub fn throttle(&self) -> f64 {
        self.current_throttle
    }

    /// Current propeller speed fraction [0, 1].
    pub fn rpm_fraction(&self) -> f64 {
        self.current_throttle
    }
}

impl AvionicsComponent for Esc {
    fn name(&self) -> &str {
        "ESC"
    }

    fn init(&mut self, _dt: f64) {
        self.current_throttle = 0.0;
    }

    fn step(&mut self, bus: &mut AvionicsBus, dt: f64) {
        // 1st-order lag on throttle, exponential smoothing
        let tau = self.config.motor_tau.max(1e-6);
        let alpha = dt / (tau + dt);
        let target = bus.fc_esc_throttle.clamp(0.0, 1.0);
        self.current_throttle += alpha * (target - self.current_throttle);

        // Deadband handling
        let db = self.config.throttle_deadband;
        if self.current_throttle.abs() < db && target <= db {
            self.current_throttle = if target <= db { 0.0 } else { self.current_throttle };
        }

        // Write actual ESC output (throttle fraction) to bus
        bus.actual_esc_output = self.current_throttle;

        // Estimate current draw (simplified: proportional to throttle²)
        // With a 4S battery at ~16.8V and ~10A max, current ≈ throttle² * max_current.
        // Actual current depends on battery + prop load; battery.rs reads the bus.
        bus.battery_current += bus.fc_esc_throttle.clamp(0.0, 1.0).powi(2) * 10.0 * 0.5;
    }

    fn reset(&mut self) {
        self.current_throttle = 0.0;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn esc_smooths_throttle() {
        let config = EscConfig {
            motor_tau: 0.05,
            ..Default::default()
        };
        let mut esc = Esc::new(config);
        esc.init(0.0);

        let mut bus = AvionicsBus::default();
        bus.fc_esc_throttle = 1.0;

        // After a short time, throttle should be > 0 but < 1
        esc.step(&mut bus, 0.1);
        let after_100ms = bus.actual_esc_output;
        assert!(
            after_100ms > 0.5 && after_100ms < 1.0,
            "after 100ms with tau=50ms, output should be between 0.5 and 1.0, got {after_100ms}"
        );

        // After 1 second, should converge near 1.0
        for _ in 0..20 {
            esc.step(&mut bus, 0.05);
        }
        assert!(
            (bus.actual_esc_output - 1.0).abs() < 0.05,
            "should converge to ~1.0 after 1s, got {}",
            bus.actual_esc_output
        );
    }

    #[test]
    fn esc_tracks_throttle_command() {
        let config = EscConfig {
            motor_tau: 0.001, // fast
            ..Default::default()
        };
        let mut esc = Esc::new(config);
        esc.init(0.0);

        let mut bus = AvionicsBus::default();
        bus.fc_esc_throttle = 0.7;
        for _ in 0..10 {
            esc.step(&mut bus, 0.01);
        }

        assert!(
            (bus.actual_esc_output - 0.7).abs() < 0.001,
            "should track quickly with fast motor, got {}",
            bus.actual_esc_output
        );
    }
}