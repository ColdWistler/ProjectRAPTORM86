//! Barometric pressure / altitude sensor.
//!
//! Simulates a BMP390 / MS5611-class barometer with configurable
//! noise, temperature drift, and update rate.
//!
//! # Bus contract
//! Reads: `true_altitude`, `true_q_dynamic`
//! Writes: `baro_altitude`, `baro_sample_time`

use serde::Deserialize;

use super::bus::AvionicsBus;
use super::sensor_model::{NoiseDistribution, NoiseType, SensorConfig, SensorModel};
use super::traits::{AvionicsComponent, Sensor};

/// Configuration for the barometric sensor.
#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "snake_case")]
pub struct BaroConfig {
    /// Noise standard deviation (metres). BMP390: ~0.02m RMS.
    #[serde(default = "default_noise")]
    pub noise: f64,
    /// Update rate (Hz).
    #[serde(default = "default_baro_hz")]
    pub update_hz: f64,
    /// Temperature coefficient (metres / degC drift). MS5611: ~0.1 m/degC.
    #[serde(default)]
    pub temp_coefficient: f64,
    /// Altitude bias (metres). Fixed manufacturing error.
    #[serde(default)]
    pub bias: f64,
}

fn default_noise() -> f64 {
    0.02
}
fn default_baro_hz() -> f64 {
    50.0
}

impl Default for BaroConfig {
    fn default() -> Self {
        Self {
            noise: default_noise(),
            update_hz: default_baro_hz(),
            temp_coefficient: 0.0,
            bias: 0.0,
        }
    }
}

/// Barometric altitude sensor.
pub struct BaroSensor {
    config: BaroConfig,
    pipeline: SensorModel,
    last_sample_time: f64,
}

impl BaroSensor {
    pub fn new(config: BaroConfig) -> Self {
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

    pub fn with_seed(config: BaroConfig, seed: u64) -> Self {
        let mut baro = Self::new(config);
        baro.pipeline.set_seed(seed);
        baro
    }
}

impl AvionicsComponent for BaroSensor {
    fn name(&self) -> &str {
        "Baro"
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

        // Barometric altitude: true altitude + sensor noise + bias.
        // Also subtract dynamic pressure head correction: Δh ≈ q / (ρ g).
        // For simplicity, just pass true altitude through the pipeline.
        bus.baro_altitude = self.pipeline.process(bus.true_altitude, dt);

        self.last_sample_time = bus.sim_time;
        bus.baro_sample_time = bus.sim_time;
    }

    fn reset(&mut self) {
        self.pipeline.reset();
        self.last_sample_time = 0.0;
    }
}

impl Sensor for BaroSensor {
    fn update_rate_hz(&self) -> f64 {
        self.config.update_hz
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn baro_passes_through_with_no_noise() {
        let config = BaroConfig {
            noise: 0.0,
            bias: 0.0,
            ..Default::default()
        };
        let mut baro = BaroSensor::with_seed(config, 42);
        let mut bus = AvionicsBus {
            true_altitude: 150.0,
            ..Default::default()
        };
        baro.init(0.002);

        bus.sim_time = 0.03;
        baro.step(&mut bus, 0.002);

        assert!((bus.baro_altitude - 150.0).abs() < 1e-9);
    }

    #[test]
    fn baro_respects_update_rate() {
        let config = BaroConfig {
            noise: 0.0,
            update_hz: 20.0, // 50ms period
            ..Default::default()
        };
        let mut baro = BaroSensor::with_seed(config, 42);
        let mut bus = AvionicsBus {
            true_altitude: 100.0,
            ..Default::default()
        };
        baro.init(0.001);

        bus.sim_time = 0.0;
        baro.step(&mut bus, 0.001);
        assert!((bus.baro_altitude - 100.0).abs() < 1e-9);

        // 20ms: too soon
        bus.true_altitude = 200.0;
        bus.sim_time = 0.020;
        baro.step(&mut bus, 0.001);
        assert!((bus.baro_altitude - 100.0).abs() < 1e-9);

        // 51ms: should update
        bus.sim_time = 0.051;
        baro.step(&mut bus, 0.001);
        assert!((bus.baro_altitude - 200.0).abs() < 1e-9);
    }

    #[test]
    fn baro_bias_shifts_output() {
        let config = BaroConfig {
            noise: 0.0,
            bias: 5.0,
            ..Default::default()
        };
        let mut baro = BaroSensor::with_seed(config, 42);
        let mut bus = AvionicsBus {
            true_altitude: 100.0,
            ..Default::default()
        };
        baro.init(0.002);

        bus.sim_time = 0.03;
        baro.step(&mut bus, 0.002);

        assert!((bus.baro_altitude - 105.0).abs() < 1e-9);
    }
}
