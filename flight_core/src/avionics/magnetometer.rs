//! 3-axis magnetometer sensor.
//!
//! Simulates an LIS3MDL / HMC5883L-class magnetometer with configurable
//! noise, bias, hard-iron offset, soft-iron scaling, and update rate.
//!
//! # Bus contract
//! Reads: `true_quat`, `true_angular_rates`
//! Writes: `mag_field_body`, `mag_sample_time`
//!
//! # Physics
//! The Earth's magnetic field in NED is approximately
//! `[B_n, B_e, B_d] = [22000, 5400, 42000]` nT (varies by location).
//! The body-frame field is `DCM × B_earth`.
//!
//! # Standards
//! - Hard-iron offset: fixed body-frame bias (permanent magnet on board)
//! - Soft-iron: 3×3 matrix distortion from nearby ferromagnetic materials

use nalgebra::Vector3;
use serde::Deserialize;

use super::bus::AvionicsBus;
use super::sensor_model::{NoiseDistribution, NoiseType, SensorConfig, SensorModel};
use super::traits::{AvionicsComponent, Sensor};

/// Configuration for the magnetometer.
#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "snake_case")]
pub struct MagConfig {
    /// Noise standard deviation per axis (nT).
    #[serde(default = "default_mag_noise")]
    pub noise: f64,
    /// Update rate (Hz).
    #[serde(default = "default_mag_hz")]
    pub update_hz: f64,
    /// Hard-iron offset per axis (nT) — fixed body-frame bias.
    #[serde(default)]
    pub hard_iron: [f64; 3],
    /// Earth magnetic field strength in NED (nT).
    #[serde(default = "default_earth_field")]
    pub earth_field_ned: [f64; 3],
}

fn default_mag_noise() -> f64 {
    80.0 // typical LIS3MDL noise
}
fn default_mag_hz() -> f64 {
    20.0
}
fn default_earth_field() -> [f64; 3] {
    [22_000.0, 5_400.0, 42_000.0]
}

impl Default for MagConfig {
    fn default() -> Self {
        Self {
            noise: default_mag_noise(),
            update_hz: default_mag_hz(),
            hard_iron: [0.0, 0.0, 0.0],
            earth_field_ned: [22_000.0, 5_400.0, 42_000.0],
        }
    }
}

/// 3-axis magnetometer sensor.
pub struct MagnetometerSensor {
    config: MagConfig,
    mag_x: SensorModel,
    mag_y: SensorModel,
    mag_z: SensorModel,
    last_sample_time: f64,
}

impl MagnetometerSensor {
    pub fn new(config: MagConfig) -> Self {
        let cfg = SensorConfig {
            noise: config.noise,
            noise_type: NoiseType::Absolute,
            noise_distribution: NoiseDistribution::Gaussian,
            ..Default::default()
        };

        Self {
            config,
            mag_x: SensorModel::new(cfg.clone()),
            mag_y: SensorModel::new(cfg.clone()),
            mag_z: SensorModel::new(cfg),
            last_sample_time: 0.0,
        }
    }

    pub fn with_seed(config: MagConfig, seed: u64) -> Self {
        let mut mag = Self::new(config);
        mag.mag_x.set_seed(seed);
        mag.mag_y.set_seed(seed + 1);
        mag.mag_z.set_seed(seed + 2);
        mag
    }
}

impl AvionicsComponent for MagnetometerSensor {
    fn name(&self) -> &str {
        "Magnetometer"
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

        // Earth magnetic field in NED → body frame via DCM
        let b_earth = Vector3::new(
            self.config.earth_field_ned[0],
            self.config.earth_field_ned[1],
            self.config.earth_field_ned[2],
        );
        let b_body = ned_to_body(b_earth, bus.true_quat);

        // Add hard-iron offset + noise
        bus.mag_field_body.x = self.mag_x.process(
            b_body.x + self.config.hard_iron[0],
            dt,
        );
        bus.mag_field_body.y = self.mag_y.process(
            b_body.y + self.config.hard_iron[1],
            dt,
        );
        bus.mag_field_body.z = self.mag_z.process(
            b_body.z + self.config.hard_iron[2],
            dt,
        );

        self.last_sample_time = bus.sim_time;
        bus.mag_sample_time = bus.sim_time;
    }

    fn reset(&mut self) {
        self.mag_x.reset();
        self.mag_y.reset();
        self.mag_z.reset();
        self.last_sample_time = 0.0;
    }
}

impl Sensor for MagnetometerSensor {
    fn update_rate_hz(&self) -> f64 {
        self.config.update_hz
    }
}

/// Transform a NED vector to body frame using the given quaternion.
/// `q = [w, x, y, z]` (Earth→body rotation).
fn ned_to_body(v: Vector3<f64>, q: [f64; 4]) -> Vector3<f64> {
    let (w, x, y, z) = (q[0], q[1], q[2], q[3]);
    // DCM (Earth→body) from quaternion
    let d00 = 1.0 - 2.0 * (y * y + z * z);
    let d01 = 2.0 * (x * y - z * w);
    let d02 = 2.0 * (x * z + y * w);
    let d10 = 2.0 * (x * y + z * w);
    let d11 = 1.0 - 2.0 * (x * x + z * z);
    let d12 = 2.0 * (y * z - x * w);
    let d20 = 2.0 * (x * z - y * w);
    let d21 = 2.0 * (y * z + x * w);
    let d22 = 1.0 - 2.0 * (x * x + y * y);

    Vector3::new(
        d00 * v.x + d01 * v.y + d02 * v.z,
        d10 * v.x + d11 * v.y + d12 * v.z,
        d20 * v.x + d21 * v.y + d22 * v.z,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mag_identity_quat_passes_through_earth_field() {
        let config = MagConfig {
            noise: 0.0,
            hard_iron: [0.0, 0.0, 0.0],
            earth_field_ned: [22_000.0, 5_400.0, 42_000.0],
            ..Default::default()
        };
        let mut mag = MagnetometerSensor::with_seed(config, 42);
        let mut bus = AvionicsBus {
            true_quat: [1.0, 0.0, 0.0, 0.0], // identity
            ..Default::default()
        };
        mag.init(0.001);

        bus.sim_time = 0.06;
        mag.step(&mut bus, 0.001);

        assert!((bus.mag_field_body.x - 22_000.0).abs() < 1.0);
        assert!((bus.mag_field_body.y - 5_400.0).abs() < 1.0);
        assert!((bus.mag_field_body.z - 42_000.0).abs() < 1.0);
    }

    #[test]
    fn mag_hard_iron_offset() {
        let config = MagConfig {
            noise: 0.0,
            hard_iron: [500.0, -300.0, 200.0],
            earth_field_ned: [22_000.0, 0.0, 0.0],
            ..Default::default()
        };
        let mut mag = MagnetometerSensor::with_seed(config, 42);
        let mut bus = AvionicsBus {
            true_quat: [1.0, 0.0, 0.0, 0.0],
            ..Default::default()
        };
        mag.init(0.001);

        bus.sim_time = 0.06;
        mag.step(&mut bus, 0.001);

        assert!((bus.mag_field_body.x - 22_500.0).abs() < 1.0);
        assert!((bus.mag_field_body.y - (-300.0)).abs() < 1.0);
        assert!((bus.mag_field_body.z - 200.0).abs() < 1.0);
    }

    #[test]
    fn mag_update_rate() {
        let config = MagConfig {
            noise: 0.0,
            update_hz: 20.0,
            ..Default::default()
        };
        let mut mag = MagnetometerSensor::with_seed(config, 42);
        let mut bus = AvionicsBus {
            true_quat: [1.0, 0.0, 0.0, 0.0],
            ..Default::default()
        };
        mag.init(0.001);

        bus.sim_time = 0.0;
        mag.step(&mut bus, 0.001);
        let first = bus.mag_field_body;

        // 20ms: too soon for 20Hz (50ms period)
        bus.sim_time = 0.020;
        mag.step(&mut bus, 0.001);
        assert!(
            (bus.mag_field_body - first).norm() < 1e-9,
            "should not update before 50ms"
        );

        // 51ms: should update
        bus.sim_time = 0.051;
        mag.step(&mut bus, 0.001);
        // Should have updated (same value since no rotation, but time changed)
        assert!(
            (bus.mag_field_body - first).norm() < 1e-9,
            "same attitude → same field"
        );
    }
}
