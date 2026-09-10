//! Battery / power supply model.
//!
//! Simulates the onboard LiPo battery with internal resistance, state of
//! charge (SoC), voltage sag under load, and configurable capacity.
//!
//! # Bus contract
//! Reads: `battery_current` (accumulated by ESC/avionics consumers)
//! Writes: `battery_voltage`, `battery_capacity_remaining_pct`
//!
//! # Model
//! Simple Thevenin equivalent: `V_terminal = V_oc(SOC) − I × R_internal`.
//! Open-circuit voltage is a linear function of state of charge.
//!
//! # Standards
//! Typical 4S LiPo: 4×4.2V = 16.8V full, 4×3.7V = 14.8V nominal,
//! ~3.3V/cell = 13.2V empty (near 0% SoC).

use serde::Deserialize;

use super::bus::AvionicsBus;
use super::traits::AvionicsComponent;

/// Configuration for the battery model.
#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "snake_case")]
pub struct BatteryConfig {
    /// Full-charge terminal voltage (V). 4S LiPo = 16.8V.
    #[serde(default = "default_full_voltage")]
    pub full_voltage: f64,
    /// Empty/discharged terminal voltage (V). 4S LiPo ≈ 13.2V.
    #[serde(default = "default_empty_voltage")]
    pub empty_voltage: f64,
    /// Battery capacity in Amp-hours (Ah).
    #[serde(default = "default_capacity_ah")]
    pub capacity_ah: f64,
    /// Internal resistance (Ohms). 4S LiPo ~ 0.02-0.05 Ω.
    #[serde(default = "default_internal_resistance")]
    pub internal_resistance: f64,
    /// Nominal continuous current draw of the avionics bus (Amps).
    /// Consumed even at zero throttle.
    #[serde(default = "default_bus_current")]
    pub bus_current: f64,
}

fn default_full_voltage() -> f64 {
    16.8
}
fn default_empty_voltage() -> f64 {
    13.2
}
fn default_capacity_ah() -> f64 {
    5.2
}
fn default_internal_resistance() -> f64 {
    0.03
}
fn default_bus_current() -> f64 {
    0.5
}

impl Default for BatteryConfig {
    fn default() -> Self {
        Self {
            full_voltage: default_full_voltage(),
            empty_voltage: default_empty_voltage(),
            capacity_ah: default_capacity_ah(),
            internal_resistance: default_internal_resistance(),
            bus_current: default_bus_current(),
        }
    }
}

/// Lithium-polymer battery with internal resistance.
pub struct Battery {
    config: BatteryConfig,
    /// Current state of charge as a fraction [0.0, 1.0].
    soc: f64,
    /// Accumulated step time (seconds).
    sim_time: f64,
}

impl Battery {
    pub fn new(config: BatteryConfig) -> Self {
        Self {
            config,
            soc: 1.0,
            sim_time: 0.0,
        }
    }

    /// Set the state of charge as a fraction [0.0, 1.0].
    pub fn set_soc(&mut self, soc: f64) {
        self.soc = soc.clamp(0.0, 1.0);
    }

    /// Current state of charge as a fraction [0.0, 1.0].
    pub fn soc(&self) -> f64 {
        self.soc
    }

    /// Open-circuit voltage (V) at the current SoC.
    fn ocv(&self) -> f64 {
        let range = self.config.full_voltage - self.config.empty_voltage;
        self.config.empty_voltage + range * self.soc
    }
}

impl AvionicsComponent for Battery {
    fn name(&self) -> &str {
        "Battery"
    }

    fn init(&mut self, _dt: f64) {
        self.soc = 1.0;
        self.sim_time = 0.0;
    }

    fn step(&mut self, bus: &mut AvionicsBus, dt: f64) {
        // Total current = load current (written by ESC etc.) + bus current
        let total_current = bus.battery_current + self.config.bus_current;

        // Terminal voltage: OCV − I×R (voltage sag under load)
        let v_terminal = self.ocv() - total_current * self.config.internal_resistance;
        bus.battery_voltage = v_terminal.max(0.0);

        // State of charge depletion: Coulomb count
        let capacity_coulombs = self.config.capacity_ah * 3600.0;
        self.soc -= total_current * dt / capacity_coulombs;
        self.soc = self.soc.clamp(0.0, 1.0);

        bus.battery_capacity_remaining_pct = self.soc * 100.0;
        bus.battery_current = total_current;

        // Depletion flag for failure injection
        bus.fc_fault_flags.battery_depleted = self.soc <= 0.0;

        self.sim_time += dt;
    }

    fn reset(&mut self) {
        self.soc = 1.0;
        self.sim_time = 0.0;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn battery_full_voltage_at_init() {
        let config = BatteryConfig::default();
        let mut batt = Battery::new(config);
        let mut bus = AvionicsBus::default();
        batt.init(0.0);
        batt.step(&mut bus, 0.01);

        assert!(
            (bus.battery_voltage - 16.8).abs() < 0.2,
            "full battery should read ~16.8V, got {}",
            bus.battery_voltage
        );
    }

    #[test]
    fn battery_sags_under_load() {
        let config = BatteryConfig {
            internal_resistance: 0.1,
            ..Default::default()
        };
        let mut batt = Battery::new(config);
        let mut bus = AvionicsBus::default();
        batt.init(0.0);

        bus.battery_current = 10.0; // heavy load
        batt.step(&mut bus, 0.01);

        // V = 16.8 - 10.0 * 0.1 = 15.8
        assert!(
            (bus.battery_voltage - 15.8).abs() < 0.2,
            "expected ~15.8V under 10A load, got {}",
            bus.battery_voltage
        );
    }

    #[test]
    fn battery_depletes_over_time() {
        let config = BatteryConfig {
            capacity_ah: 1.0, // 3600 Coulombs
            internal_resistance: 0.0,
            bus_current: 0.0,
            ..Default::default()
        };
        let mut batt = Battery::new(config);
        let mut bus = AvionicsBus::default();
        batt.init(0.0);

        // Draw 10A for 5 minutes = 0.833Ah → SoC drops to ~17%
        bus.battery_current = 10.0;
        for _ in 0..(5 * 60 * 100) {
            batt.step(&mut bus, 0.01);
        }

        assert!(
            (bus.battery_capacity_remaining_pct - 16.7).abs() < 3.0,
            "after 5min at 10A, ~17% remaining, got {}",
            bus.battery_capacity_remaining_pct
        );
    }

    #[test]
    fn battery_empty_trigger_flags() {
        let config = BatteryConfig {
            capacity_ah: 0.1, // tiny battery
            ..Default::default()
        };
        let mut batt = Battery::new(config);
        batt.set_soc(0.01);
        let mut bus = AvionicsBus::default();
        bus.battery_current = 20.0;
        for _ in 0..1000 {
            batt.step(&mut bus, 0.01);
        }
        assert!(
            bus.fc_fault_flags.battery_depleted,
            "battery should be flagged depleted"
        );
    }
}