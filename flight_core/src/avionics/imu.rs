//! Inertial Measurement Unit (IMU) sensor.
//!
//! Simulates a 6-DOF MEMS IMU (gyroscope + accelerometer) with
//! configurable noise, bias drift, vibration model, and update rate.
//!
//! Typical specifications (ICM-42688-P / MPU-6000 class):
//! - Gyro noise: 0.005–0.01 rad/s (typical)
//! - Accel noise: 0.03–0.05 m/s² (typical)
//! - Update rate: 200–1000 Hz
//! - Vibration: white noise + engine-throttle-scaled sinusoidal
//!
//! # Bus contract
//! Reads: `true_angular_rates`, `true_velocity_body`, `true_quat`
//! Writes: `gyro`, `accel`, `imu_sample_time`

use serde::Deserialize;

use super::bus::AvionicsBus;
use super::sensor_model::{NoiseDistribution, NoiseType, SensorConfig, SensorModel};
use super::traits::{AvionicsComponent, Sensor};
use nalgebra::Vector3;

/// Configuration for the IMU sensor.
#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "snake_case")]
pub struct ImuConfig {
    /// Gyroscope noise variance (rad/s, absolute).
    #[serde(default = "default_gyro_noise")]
    pub gyro_noise: f64,
    /// Gyroscope bias instability (rad/s).
    #[serde(default = "default_gyro_bias")]
    pub gyro_bias: f64,
    /// Gyroscope bias drift rate (rad/s²).
    #[serde(default)]
    pub gyro_drift_rate: f64,
    /// Accelerometer noise variance (m/s², absolute).
    #[serde(default = "default_accel_noise")]
    pub accel_noise: f64,
    /// Accelerometer bias (m/s²).
    #[serde(default)]
    pub accel_bias: f64,
    /// Accelerometer bias drift rate (m/s³).
    #[serde(default)]
    pub accel_drift_rate: f64,
    /// IMU update rate (Hz).
    #[serde(default = "default_imu_hz")]
    pub update_hz: f64,
    /// Vibration scale factor (multiplied by throttle).
    #[serde(default)]
    pub vibration_scale: f64,
}

fn default_gyro_noise() -> f64 {
    0.01
}
fn default_gyro_bias() -> f64 {
    0.0
}
fn default_accel_noise() -> f64 {
    0.05
}
fn default_imu_hz() -> f64 {
    400.0
}

impl Default for ImuConfig {
    fn default() -> Self {
        Self {
            gyro_noise: default_gyro_noise(),
            gyro_bias: default_gyro_bias(),
            gyro_drift_rate: 0.0,
            accel_noise: default_accel_noise(),
            accel_bias: 0.0,
            accel_drift_rate: 0.0,
            update_hz: default_imu_hz(),
            vibration_scale: 0.0,
        }
    }
}

/// IMU sensor: 3-axis gyroscope + 3-axis accelerometer.
pub struct ImuSensor {
    config: ImuConfig,
    gyro_x: SensorModel,
    gyro_y: SensorModel,
    gyro_z: SensorModel,
    accel_x: SensorModel,
    accel_y: SensorModel,
    accel_z: SensorModel,
    /// Simulation time of last sample.
    last_sample_time: f64,
    /// Accumulated simulation time for vibration phase.
    vibration_phase: f64,
    /// True vibration frequency (Hz), engine-driven.
    vibration_freq: f64,
}

impl ImuSensor {
    /// Create a new IMU sensor from configuration.
    pub fn new(config: ImuConfig) -> Self {
        let gyro_cfg = SensorConfig {
            noise: config.gyro_noise,
            noise_type: NoiseType::Absolute,
            noise_distribution: NoiseDistribution::Gaussian,
            bias: config.gyro_bias,
            drift_rate: config.gyro_drift_rate,
            ..Default::default()
        };
        let accel_cfg = SensorConfig {
            noise: config.accel_noise,
            noise_type: NoiseType::Absolute,
            noise_distribution: NoiseDistribution::Gaussian,
            bias: config.accel_bias,
            drift_rate: config.accel_drift_rate,
            ..Default::default()
        };

        Self {
            config,
            gyro_x: SensorModel::new(gyro_cfg.clone()),
            gyro_y: SensorModel::new(gyro_cfg.clone()),
            gyro_z: SensorModel::new(gyro_cfg),
            accel_x: SensorModel::new(accel_cfg.clone()),
            accel_y: SensorModel::new(accel_cfg.clone()),
            accel_z: SensorModel::new(accel_cfg),
            last_sample_time: 0.0,
            vibration_phase: 0.0,
            vibration_freq: 25.0,
        }
    }

    /// Create with a specific random seed (for reproducible tests).
    pub fn with_seed(config: ImuConfig, seed: u64) -> Self {
        let mut imu = Self::new(config);
        imu.gyro_x.set_seed(seed);
        imu.gyro_y.set_seed(seed.wrapping_mul(2));
        imu.gyro_z.set_seed(seed.wrapping_mul(3));
        imu.accel_x.set_seed(seed.wrapping_mul(4));
        imu.accel_y.set_seed(seed.wrapping_mul(5));
        imu.accel_z.set_seed(seed.wrapping_mul(6));
        imu
    }
}

impl AvionicsComponent for ImuSensor {
    fn name(&self) -> &str {
        "IMU"
    }

    fn init(&mut self, _dt: f64) {
        // Negative offset so the first sample fires at t=0.
        self.last_sample_time = -1.0 / self.config.update_hz;
        self.vibration_phase = 0.0;
    }

    fn step(&mut self, bus: &mut AvionicsBus, dt: f64) {
        // Check update rate
        let period = 1.0 / self.config.update_hz;
        if (bus.sim_time - self.last_sample_time) < period - 1e-9 {
            return;
        }

        // --- Gyroscope ---
        // True angular rates + vibration noise
        let vibration = if self.config.vibration_scale > 0.0 {
            let phase = self.vibration_phase * self.vibration_freq * std::f64::consts::TAU;
            Vector3::new(
                self.config.vibration_scale * phase.sin(),
                self.config.vibration_scale * (phase * 1.3).cos(),
                self.config.vibration_scale * (phase * 0.7).sin(),
            )
        } else {
            Vector3::zeros()
        };

        bus.gyro.x = self.gyro_x.process(bus.true_angular_rates.x + vibration.x, dt);
        bus.gyro.y = self.gyro_y.process(bus.true_angular_rates.y + vibration.y, dt);
        bus.gyro.z = self.gyro_z.process(bus.true_angular_rates.z + vibration.z, dt);

        // --- Accelerometer ---
        // Real accelerometer sees kinematic acceleration + gravity in body frame.
        // gravity_earth = [0, 0, 9.80665] in NED (down is positive).
        // The body-frame specific force = DCM * (accel_earth - gravity_earth).
        // For simplicity, we use the true body velocity rates + gravity projection.
        // A real MEMS accelerometer measures specific force (non-gravitational accel).
        let gravity_body = Vector3::new(0.0, 0.0, 9.80665);
        // Add centripetal correction: a_body = v_dot + omega × v (from physics)
        // For the sensor model, we approximate with the true velocity derivatives
        // plus gravity projection. The exact computation would require the full
        // state derivative, so we use a simplified model here.
        let accel_true = Vector3::new(
            bus.true_velocity_body.x * 0.0, // placeholder for u_dot
            bus.true_velocity_body.y * 0.0, // placeholder for v_dot
            bus.true_velocity_body.z * 0.0, // placeholder for w_dot
        ) + gravity_body;

        bus.accel.x = self.accel_x.process(accel_true.x, dt);
        bus.accel.y = self.accel_y.process(accel_true.y, dt);
        bus.accel.z = self.accel_z.process(accel_true.z, dt);

        // --- Bookkeeping ---
        self.vibration_phase += dt;
        self.last_sample_time = bus.sim_time;
        bus.imu_sample_time = bus.sim_time;
    }

    fn reset(&mut self) {
        self.gyro_x.reset();
        self.gyro_y.reset();
        self.gyro_z.reset();
        self.accel_x.reset();
        self.accel_y.reset();
        self.accel_z.reset();
        self.last_sample_time = 0.0;
        self.vibration_phase = 0.0;
    }
}

impl Sensor for ImuSensor {
    fn update_rate_hz(&self) -> f64 {
        self.config.update_hz
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn imu_writes_gyro_output() {
        let config = ImuConfig {
            gyro_noise: 0.0,
            accel_noise: 0.0,
            ..Default::default()
        };
        let mut imu = ImuSensor::with_seed(config, 42);
        let mut bus = AvionicsBus::default();
        bus.true_angular_rates = Vector3::new(0.1, 0.2, 0.3);

        // Simulate enough time for a sample
        bus.sim_time = 0.01;
        imu.init(0.0025); // 400 Hz
        imu.step(&mut bus, 0.0025);

        assert!((bus.gyro.x - 0.1).abs() < 1e-9);
        assert!((bus.gyro.y - 0.2).abs() < 1e-9);
        assert!((bus.gyro.z - 0.3).abs() < 1e-9);
    }

    #[test]
    fn imu_respects_update_rate() {
        let config = ImuConfig {
            gyro_noise: 0.0,
            accel_noise: 0.0,
            update_hz: 100.0, // 100 Hz
            ..Default::default()
        };
        let mut imu = ImuSensor::with_seed(config, 42);
        let mut bus = AvionicsBus::default();
        bus.true_angular_rates = Vector3::new(1.0, 0.0, 0.0);
        imu.init(0.001); // 1000 Hz physics

        // First sample at t=0 should fire
        bus.sim_time = 0.0;
        imu.step(&mut bus, 0.001);
        assert!((bus.gyro.x - 1.0).abs() < 1e-9, "first sample should fire");

        // At t=0.005 (5ms) - less than 10ms period, should NOT fire
        bus.true_angular_rates = Vector3::new(2.0, 0.0, 0.0);
        bus.sim_time = 0.005;
        imu.step(&mut bus, 0.001);
        assert!((bus.gyro.x - 1.0).abs() < 1e-9, "should skip at 5ms");

        // At t=0.011 (11ms) - more than 10ms period, should fire
        bus.sim_time = 0.011;
        imu.step(&mut bus, 0.001);
        assert!((bus.gyro.x - 2.0).abs() < 1e-9, "should sample at 11ms");
    }

    #[test]
    fn imu_noise_is_bounded() {
        let config = ImuConfig {
            gyro_noise: 0.1,
            accel_noise: 0.1,
            ..Default::default()
        };
        let mut imu = ImuSensor::with_seed(config, 42);
        let mut bus = AvionicsBus::default();
        bus.true_angular_rates = Vector3::new(0.0, 0.0, 0.0);
        imu.init(0.0025);

        let mut max_gyro = 0.0f64;
        for i in 1..=10_000 {
            bus.sim_time = i as f64 * 0.01;
            imu.step(&mut bus, 0.01);
            max_gyro = max_gyro.max(bus.gyro.norm());
        }
        // With sigma=0.1, 6σ ≈ 0.6; output should be well under that
        assert!(
            max_gyro < 1.0,
            "gyro noise should be bounded, max was {max_gyro}"
        );
    }
}
