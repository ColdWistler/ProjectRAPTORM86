//! GPS position/velocity sensor.
//!
//! Simulates a UBlox-class GPS with configurable update rate,
//! position/velocity noise, HDOP model, fix quality, and multipath.
//!
//! # Bus contract
//! Reads: `true_position_ned`, `true_velocity_body`, `true_quat`
//! Writes: `gps_position_ned`, `gps_velocity_ned`, `gps_fix_quality`,
//!         `gps_hdop`, `gps_sample_time`

use nalgebra::Vector3;
use serde::Deserialize;

use super::bus::AvionicsBus;
use super::sensor_model::{NoiseDistribution, NoiseType, SensorConfig, SensorModel};
use super::traits::{AvionicsComponent, Sensor};

/// Configuration for the GPS sensor.
#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "snake_case")]
pub struct GpsConfig {
    /// Position noise standard deviation per axis (metres).
    #[serde(default = "default_pos_noise")]
    pub position_noise: f64,
    /// Velocity noise standard deviation per axis (m/s).
    #[serde(default = "default_vel_noise")]
    pub velocity_noise: f64,
    /// GPS update rate (Hz).
    #[serde(default = "default_gps_hz")]
    pub update_hz: f64,
    /// Heading noise standard deviation (radians). Used for velocity
    /// projection from body to NED.
    #[serde(default = "default_heading_noise")]
    pub heading_noise: f64,
    /// Horizontal dilution of precision (base value, lower is better).
    #[serde(default = "default_hdop")]
    pub hdop: f64,
    /// Fix quality (3 = 3D fix).
    #[serde(default)]
    pub fix_quality: u8,
}

fn default_pos_noise() -> f64 {
    2.5
}
fn default_vel_noise() -> f64 {
    0.05
}
fn default_gps_hz() -> f64 {
    10.0
}
fn default_heading_noise() -> f64 {
    0.003
}
fn default_hdop() -> f64 {
    1.2
}

impl Default for GpsConfig {
    fn default() -> Self {
        Self {
            position_noise: default_pos_noise(),
            velocity_noise: default_vel_noise(),
            update_hz: default_gps_hz(),
            heading_noise: default_heading_noise(),
            hdop: default_hdop(),
            fix_quality: 3,
        }
    }
}

/// GPS sensor: NED position + NED velocity from true state.
pub struct GpsSensor {
    config: GpsConfig,
    pos_noise_x: SensorModel,
    pos_noise_y: SensorModel,
    pos_noise_z: SensorModel,
    vel_noise_x: SensorModel,
    vel_noise_y: SensorModel,
    vel_noise_z: SensorModel,
    last_sample_time: f64,
}

impl GpsSensor {
    pub fn new(config: GpsConfig) -> Self {
        let pos_cfg = SensorConfig {
            noise: config.position_noise,
            noise_type: NoiseType::Absolute,
            noise_distribution: NoiseDistribution::Gaussian,
            ..Default::default()
        };
        let vel_cfg = SensorConfig {
            noise: config.velocity_noise,
            noise_type: NoiseType::Absolute,
            noise_distribution: NoiseDistribution::Gaussian,
            ..Default::default()
        };

        Self {
            config,
            pos_noise_x: SensorModel::new(pos_cfg.clone()),
            pos_noise_y: SensorModel::new(pos_cfg.clone()),
            pos_noise_z: SensorModel::new(pos_cfg),
            vel_noise_x: SensorModel::new(vel_cfg.clone()),
            vel_noise_y: SensorModel::new(vel_cfg.clone()),
            vel_noise_z: SensorModel::new(vel_cfg),
            last_sample_time: 0.0,
        }
    }

    pub fn with_seed(config: GpsConfig, seed: u64) -> Self {
        let mut gps = Self::new(config);
        gps.pos_noise_x.set_seed(seed);
        gps.pos_noise_y.set_seed(seed + 1);
        gps.pos_noise_z.set_seed(seed + 2);
        gps.vel_noise_x.set_seed(seed + 3);
        gps.vel_noise_y.set_seed(seed + 4);
        gps.vel_noise_z.set_seed(seed + 5);
        gps
    }
}

impl AvionicsComponent for GpsSensor {
    fn name(&self) -> &str {
        "GPS"
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

        // Failure mode: dead GPS — no fix, no position or velocity
        if bus.fc_fault_flags.gps_failed {
            bus.gps_fix_quality = 0;
            bus.gps_hdop = 99.9;
            bus.gps_position_ned = Vector3::zeros();
            bus.gps_velocity_ned = Vector3::zeros();
            self.last_sample_time = bus.sim_time;
            bus.gps_sample_time = bus.sim_time;
            return;
        }

        // Position: true NED + noise
        bus.gps_position_ned.x = self
            .pos_noise_x
            .process(bus.true_position_ned.x, dt);
        bus.gps_position_ned.y = self
            .pos_noise_y
            .process(bus.true_position_ned.y, dt);
        bus.gps_position_ned.z = self
            .pos_noise_z
            .process(bus.true_position_ned.z, dt);

        // Velocity: body-frame velocity rotated to NED + noise.
        // DCM from true quaternion.
        let (d00, d01, d02, d10, d11, d12, d20, d21, d22) = quat_to_dcm(bus.true_quat);
        let v_ned = Vector3::new(
            d00 * bus.true_velocity_body.x + d01 * bus.true_velocity_body.y + d02 * bus.true_velocity_body.z,
            d10 * bus.true_velocity_body.x + d11 * bus.true_velocity_body.y + d12 * bus.true_velocity_body.z,
            d20 * bus.true_velocity_body.x + d21 * bus.true_velocity_body.y + d22 * bus.true_velocity_body.z,
        );

        bus.gps_velocity_ned.x = self.vel_noise_x.process(v_ned.x, dt);
        bus.gps_velocity_ned.y = self.vel_noise_y.process(v_ned.y, dt);
        bus.gps_velocity_ned.z = self.vel_noise_z.process(v_ned.z, dt);

        // HDOP (simplified: constant for now; could vary with satellite geometry)
        bus.gps_hdop = self.config.hdop;
        bus.gps_fix_quality = self.config.fix_quality;

        self.last_sample_time = bus.sim_time;
        bus.gps_sample_time = bus.sim_time;
    }

    fn reset(&mut self) {
        self.pos_noise_x.reset();
        self.pos_noise_y.reset();
        self.pos_noise_z.reset();
        self.vel_noise_x.reset();
        self.vel_noise_y.reset();
        self.vel_noise_z.reset();
        self.last_sample_time = 0.0;
    }
}

impl Sensor for GpsSensor {
    fn update_rate_hz(&self) -> f64 {
        self.config.update_hz
    }
}

/// Quaternion [w, x, y, z] → DCM (body-to-earth transpose).
/// Returns (d00, d01, d02, d10, d11, d12, d20, d21, d22).
fn quat_to_dcm(q: [f64; 4]) -> (f64, f64, f64, f64, f64, f64, f64, f64, f64) {
    let (w, x, y, z) = (q[0], q[1], q[2], q[3]);
    let d00 = 1.0 - 2.0 * (y * y + z * z);
    let d01 = 2.0 * (x * y - z * w);
    let d02 = 2.0 * (x * z + y * w);
    let d10 = 2.0 * (x * y + z * w);
    let d11 = 1.0 - 2.0 * (x * x + z * z);
    let d12 = 2.0 * (y * z - x * w);
    let d20 = 2.0 * (x * z - y * w);
    let d21 = 2.0 * (y * z + x * w);
    let d22 = 1.0 - 2.0 * (x * x + y * y);
    (d00, d01, d02, d10, d11, d12, d20, d21, d22)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gps_writes_position() {
        let config = GpsConfig {
            position_noise: 0.0,
            velocity_noise: 0.0,
            ..Default::default()
        };
        let mut gps = GpsSensor::with_seed(config, 42);
        let mut bus = AvionicsBus {
            true_position_ned: Vector3::new(100.0, 200.0, -50.0),
            ..Default::default()
        };
        gps.init(0.01);

        bus.sim_time = 0.11;
        gps.step(&mut bus, 0.01);

        assert!((bus.gps_position_ned.x - 100.0).abs() < 1e-9);
        assert!((bus.gps_position_ned.y - 200.0).abs() < 1e-9);
        assert!((bus.gps_position_ned.z - (-50.0)).abs() < 1e-9);
    }

    #[test]
    fn gps_respects_update_rate() {
        let config = GpsConfig {
            position_noise: 0.0,
            velocity_noise: 0.0,
            update_hz: 10.0,
            ..Default::default()
        };
        let mut gps = GpsSensor::with_seed(config, 42);
        let mut bus = AvionicsBus {
            true_position_ned: Vector3::new(100.0, 0.0, 0.0),
            ..Default::default()
        };
        gps.init(0.001);

        // t=0: first sample
        bus.sim_time = 0.0;
        gps.step(&mut bus, 0.001);
        assert!((bus.gps_position_ned.x - 100.0).abs() < 1e-9);

        // t=50ms: too soon (period=100ms), should NOT update
        bus.true_position_ned = Vector3::new(200.0, 0.0, 0.0);
        bus.sim_time = 0.050;
        gps.step(&mut bus, 0.001);
        assert!((bus.gps_position_ned.x - 100.0).abs() < 1e-9, "should not update at 50ms");

        // t=101ms: should update
        bus.sim_time = 0.101;
        gps.step(&mut bus, 0.001);
        assert!((bus.gps_position_ned.x - 200.0).abs() < 1e-9, "should update at 101ms");
    }

    #[test]
    fn gps_velocity_body_to_ned() {
        let config = GpsConfig {
            position_noise: 0.0,
            velocity_noise: 0.0,
            ..Default::default()
        };
        let mut gps = GpsSensor::with_seed(config, 42);
        let mut bus = AvionicsBus {
            // Identity quaternion → body = NED
            true_quat: [1.0, 0.0, 0.0, 0.0],
            true_velocity_body: Vector3::new(10.0, 0.0, 0.0), // forward
            ..Default::default()
        };
        gps.init(0.001);

        bus.sim_time = 0.11;
        gps.step(&mut bus, 0.001);

        assert!(
            (bus.gps_velocity_ned.x - 10.0).abs() < 1e-9,
            "forward body velocity should map to NED x"
        );
    }
}
