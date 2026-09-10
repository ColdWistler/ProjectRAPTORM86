//! Pitot-static airspeed sensor.
//!
//! Simulates a differential pressure / pitot-static tube airspeed
//! sensor (e.g., MS4525DO) with configurable noise, bias, position
//! error coefficient, and update rate.
//!
//! # Bus contract
//! Reads: `true_airspeed`, `true_q_dynamic`, `true_alpha`, `true_beta`
//! Writes: `airspeed_indicated`, `airspeed_sample_time`
//!
//! # Physics
//! Indicated airspeed (IAS) relates to true airspeed (TAS) via:
//!   `IAS = TAS × √(ρ / ρ₀)` where ρ₀ = 1.225 kg/m³ at sea level.
//! The pitot tube measures dynamic pressure directly: `q = 0.5 ρ V²`.
//! Position error coefficient models static port placement error.

use serde::Deserialize;

use super::bus::AvionicsBus;
use super::sensor_model::{NoiseDistribution, NoiseType, SensorConfig, SensorModel};
use super::traits::{AvionicsComponent, Sensor};

/// Configuration for the airspeed sensor.
#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "snake_case")]
pub struct AirspeedConfig {
    /// Noise standard deviation (m/s).
    #[serde(default = "default_airspeed_noise")]
    pub noise: f64,
    /// Update rate (Hz).
    #[serde(default = "default_airspeed_hz")]
    pub update_hz: f64,
    /// Fixed bias (m/s).
    #[serde(default)]
    pub bias: f64,
    /// Position error coefficient (fractional). Models systematic
    /// error from pitot tube placement: 0.0 = perfect, ±0.05 = typical.
    #[serde(default)]
    pub position_error: f64,
}

fn default_airspeed_noise() -> f64 {
    0.3
}
fn default_airspeed_hz() -> f64 {
    50.0
}

impl Default for AirspeedConfig {
    fn default() -> Self {
        Self {
            noise: default_airspeed_noise(),
            update_hz: default_airspeed_hz(),
            bias: 0.0,
            position_error: 0.0,
        }
    }
}

/// Pitot-static airspeed sensor.
pub struct AirspeedSensor {
    config: AirspeedConfig,
    pipeline: SensorModel,
    last_sample_time: f64,
}

impl AirspeedSensor {
    pub fn new(config: AirspeedConfig) -> Self {
        let pipeline = SensorModel::new(SensorConfig {
            noise: config.noise,
            noise_type: NoiseType::Absolute,
            noise_distribution: NoiseDistribution::Gaussian,
            bias: config.bias,
            ..Default::default()
        });

        Self {
            config,
            pipeline,
            last_sample_time: 0.0,
        }
    }

    pub fn with_seed(config: AirspeedConfig, seed: u64) -> Self {
        let mut as_ = Self::new(config);
        as_.pipeline.set_seed(seed);
        as_
    }
}

impl AvionicsComponent for AirspeedSensor {
    fn name(&self) -> &str {
        "Airspeed"
    }

    fn init(&mut self, _dt: f64) {
        // Negative offset so the first sample fires at t=0.
        self.last_sample_time = -1.0 / self.config.update_hz;
    }

    fn step(&mut self, bus: &mut AvionicsBus, dt: f64) {
        let period = 1.0 / self.config.update_hz;
        if (bus.sim_time - self.last_sample_time) < period - 1e-9 {
            return;
        }

        // Indicated airspeed ≈ TAS × (1 + position_error) + noise + bias
        let ias = bus.true_airspeed * (1.0 + self.config.position_error);
        bus.airspeed_indicated = self.pipeline.process(ias, dt);

        self.last_sample_time = bus.sim_time;
        bus.airspeed_sample_time = bus.sim_time;
    }

    fn reset(&mut self) {
        self.pipeline.reset();
        self.last_sample_time = 0.0;
    }
}

impl Sensor for AirspeedSensor {
    fn update_rate_hz(&self) -> f64 {
        self.config.update_hz
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn airspeed_perfect_passthrough() {
        let config = AirspeedConfig {
            noise: 0.0,
            bias: 0.0,
            position_error: 0.0,
            ..Default::default()
        };
        let mut as_ = AirspeedSensor::with_seed(config, 42);
        let mut bus = AvionicsBus {
            true_airspeed: 25.0,
            ..Default::default()
        };
        as_.init(0.002);

        bus.sim_time = 0.03;
        as_.step(&mut bus, 0.002);

        assert!((bus.airspeed_indicated - 25.0).abs() < 1e-9);
    }

    #[test]
    fn airspeed_position_error() {
        let config = AirspeedConfig {
            noise: 0.0,
            position_error: 0.05, // +5% error
            ..Default::default()
        };
        let mut as_ = AirspeedSensor::with_seed(config, 42);
        let mut bus = AvionicsBus {
            true_airspeed: 20.0,
            ..Default::default()
        };
        as_.init(0.002);

        bus.sim_time = 0.03;
        as_.step(&mut bus, 0.002);

        assert!((bus.airspeed_indicated - 21.0).abs() < 1e-9);
    }

    #[test]
    fn airspeed_bias() {
        let config = AirspeedConfig {
            noise: 0.0,
            bias: 1.5,
            ..Default::default()
        };
        let mut as_ = AirspeedSensor::with_seed(config, 42);
        let mut bus = AvionicsBus {
            true_airspeed: 10.0,
            ..Default::default()
        };
        as_.init(0.002);

        bus.sim_time = 0.03;
        as_.step(&mut bus, 0.002);

        assert!((bus.airspeed_indicated - 11.5).abs() < 1e-9);
    }

    #[test]
    fn airspeed_update_rate() {
        let config = AirspeedConfig {
            noise: 0.0,
            update_hz: 50.0, // 20ms period
            ..Default::default()
        };
        let mut as_ = AirspeedSensor::with_seed(config, 42);
        let mut bus = AvionicsBus {
            true_airspeed: 15.0,
            ..Default::default()
        };
        as_.init(0.001);

        bus.sim_time = 0.0;
        as_.step(&mut bus, 0.001);
        assert!((bus.airspeed_indicated - 15.0).abs() < 1e-9);

        // 10ms: too soon
        bus.true_airspeed = 25.0;
        bus.sim_time = 0.010;
        as_.step(&mut bus, 0.001);
        assert!((bus.airspeed_indicated - 15.0).abs() < 1e-9);

        // 21ms: should update
        bus.sim_time = 0.021;
        as_.step(&mut bus, 0.001);
        assert!((bus.airspeed_indicated - 25.0).abs() < 1e-9);
    }
}
