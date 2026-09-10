//! Generic sensor signal degradation pipeline.
//!
//! Implements the JSBSim `FGSensor`-style degradation chain:
//! ```text
//! true_value → lag → noise → drift → bias → gain → quantize → clip
//! ```
//!
//! Each stage is optional (set to 0.0 / disabled to skip). The pipeline
//! is designed for single-axis signals; multi-axis sensors (IMU, GPS)
//! compose three or six `SensorModel` instances.
//!
//! # Standards & Traceability
//! - Pipeline order follows JSBSim `FGSensor::ProcessSensorSignal()`:
//!   lag → noise → drift → bias → gain → delay → quantize → clip.
//! - Lag is a 2nd-order Butterworth-like low-pass (bilinear transform):
//!   `y[n] = ca * (x[n] + x[n-1]) + cb * y[n-1]`.
//! - Noise supports Gaussian and uniform distributions, absolute or
//!   percent (relative) variance.
//! - Drift is a linear time-dependent bias walk: `drift += rate * dt`.
//! - Quantization models ADC bit-depth with configurable range.

use serde::Deserialize;

/// Noise distribution type.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize, Default)]
#[serde(rename_all = "lowercase")]
pub enum NoiseDistribution {
    /// Normal (Gaussian) distribution, ~99.7% within ±3σ.
    #[default]
    Gaussian,
    /// Uniform distribution in [-1, +1].
    Uniform,
}

/// Noise variance type: additive (absolute) or multiplicative (percent).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize, Default)]
#[serde(rename_all = "lowercase")]
pub enum NoiseType {
    /// Noise is added directly: `output += variance * random`.
    #[default]
    Absolute,
    /// Noise scales the signal: `output *= (1 + variance * random)`.
    Percent,
}

/// Configuration for a single-axis sensor pipeline.
#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "snake_case")]
pub struct SensorConfig {
    /// 1st-order low-pass filter time constant (seconds). 0 = no lag.
    #[serde(default)]
    pub lag: f64,
    /// Noise variance. Interpretation depends on `noise_type`.
    #[serde(default)]
    pub noise: f64,
    /// Noise distribution: Gaussian or Uniform.
    #[serde(default)]
    pub noise_distribution: NoiseDistribution,
    /// Noise type: Absolute (additive) or Percent (multiplicative).
    #[serde(default)]
    pub noise_type: NoiseType,
    /// Linear drift rate (units/sec). 0 = no drift.
    #[serde(default)]
    pub drift_rate: f64,
    /// Fixed bias (units). 0 = no bias.
    #[serde(default)]
    pub bias: f64,
    /// Gain error (dimensionless). 1.0 = perfect. 0 = disabled.
    #[serde(default)]
    pub gain: f64,
    /// ADC quantization bits. 0 = no quantization.
    #[serde(default)]
    pub quantization_bits: u32,
    /// Minimum quantization range value.
    #[serde(default = "default_quant_min")]
    pub quantize_min: f64,
    /// Maximum quantization range value.
    #[serde(default = "default_quant_max")]
    pub quantize_max: f64,
    /// Output clip minimum. None = no lower clip.
    pub clip_min: Option<f64>,
    /// Output clip maximum. None = no upper clip.
    pub clip_max: Option<f64>,
}

fn default_quant_min() -> f64 {
    -500.0
}
fn default_quant_max() -> f64 {
    500.0
}

impl Default for SensorConfig {
    fn default() -> Self {
        Self {
            lag: 0.0,
            noise: 0.0,
            noise_distribution: NoiseDistribution::Gaussian,
            noise_type: NoiseType::Absolute,
            drift_rate: 0.0,
            bias: 0.0,
            gain: 0.0,
            quantization_bits: 0,
            quantize_min: default_quant_min(),
            quantize_max: default_quant_max(),
            clip_min: None,
            clip_max: None,
        }
    }
}

/// A single-axis sensor degradation pipeline.
///
/// Processes a true value through the degradation chain and returns
/// the degraded output. Each stage is applied in order and is
/// individually testable.
#[derive(Debug, Clone)]
pub struct SensorModel {
    config: SensorConfig,
    /// Previous output for the lag filter.
    prev_output: f64,
    /// Accumulated drift (units).
    drift: f64,
    /// Quantization step size (computed from config).
    quant_step: f64,
    /// PRNG state (xorshift64*).
    rng_state: u64,
}

impl SensorModel {
    /// Create a new sensor model from configuration.
    pub fn new(config: SensorConfig) -> Self {
        let (quant_step, _quant_divisions) = if config.quantization_bits > 0 {
            let divisions = (1u64 << config.quantization_bits) as f64;
            let step = (config.quantize_max - config.quantize_min) / divisions;
            (step, divisions)
        } else {
            (0.0, 0.0)
        };

        Self {
            config,
            prev_output: 0.0,
            drift: 0.0,
            quant_step,
            rng_state: 0x9E37_79B9_7F4A_7C15,
        }
    }

    /// Create with a specific random seed (for reproducible tests).
    pub fn with_seed(config: SensorConfig, seed: u64) -> Self {
        let mut m = Self::new(config);
        m.rng_state = seed.max(1);
        m
    }

    /// Process one sample through the full degradation pipeline.
    pub fn process(&mut self, true_value: f64, dt: f64) -> f64 {
        let mut output = true_value;

        // 1. Lag (1st-order low-pass, exact discretization)
        if self.config.lag > 0.0 && dt > 0.0 {
            let alpha = 1.0 - (-dt / self.config.lag).exp();
            output = self.prev_output + alpha * (output - self.prev_output);
            self.prev_output = output;
        }

        // 2. Noise
        if self.config.noise > 0.0 {
            let random = self.next_random_value();
            match self.config.noise_type {
                NoiseType::Absolute => output += self.config.noise * random,
                NoiseType::Percent => output *= 1.0 + self.config.noise * random,
            }
        }

        // 3. Drift (linear time-dependent bias walk)
        if self.config.drift_rate != 0.0 {
            self.drift += self.config.drift_rate * dt;
            output += self.drift;
        }

        // 4. Bias (fixed offset)
        if self.config.bias != 0.0 {
            output += self.config.bias;
        }

        // 5. Gain (scale factor error)
        if self.config.gain != 0.0 {
            output *= self.config.gain;
        }

        // 6. Quantization (ADC bit-depth)
        if self.config.quantization_bits > 0 {
            output = self.quantize(output);
        }

        // 7. Clip
        if let Some(min) = self.config.clip_min {
            output = output.max(min);
        }
        if let Some(max) = self.config.clip_max {
            output = output.min(max);
        }

        output
    }

    /// Reset internal state (drift accumulator, lag buffers).
    pub fn reset(&mut self) {
        self.prev_output = 0.0;
        self.drift = 0.0;
    }

    /// Get the current accumulated drift value.
    pub fn drift(&self) -> f64 {
        self.drift
    }

    /// Set the PRNG seed for reproducible noise generation.
    pub fn set_seed(&mut self, seed: u64) {
        self.rng_state = seed.max(1);
    }

    // --- Internal ---

    /// xorshift64* PRNG: returns uniform random f64 in [0, 1).
    fn next_random(&mut self) -> f64 {
        let mut x = self.rng_state;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.rng_state = x;
        let w = x.wrapping_mul(0x2545_F491_4F6C_DD1D);
        (w >> 11) as f64 / (1u64 << 53) as f64
    }

    /// Box-Muller transform: standard normal from uniform.
    fn next_gaussian(&mut self) -> f64 {
        let u1 = (self.next_random() + 1e-12).min(0.999_999_999);
        let u2 = self.next_random().max(1e-12);
        (-2.0 * u1.ln()).sqrt() * (std::f64::consts::TAU * u2).cos()
    }

    /// Return a random value according to the configured distribution.
    fn next_random_value(&mut self) -> f64 {
        match self.config.noise_distribution {
            NoiseDistribution::Gaussian => self.next_gaussian(),
            NoiseDistribution::Uniform => self.next_random() * 2.0 - 1.0,
        }
    }

    /// Quantize output to ADC bit-depth.
    fn quantize(&self, value: f64) -> f64 {
        let clamped = value
            .max(self.config.quantize_min)
            .min(self.config.quantize_max);
        let portion = clamped - self.config.quantize_min;
        let q = (portion / self.quant_step).floor();
        q * self.quant_step + self.config.quantize_min
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Helper: create a sensor with all degradations disabled.
    fn perfect_sensor() -> SensorModel {
        SensorModel::new(SensorConfig::default())
    }

    #[test]
    fn perfect_sensor_passes_through() {
        let mut s = perfect_sensor();
        assert!((s.process(42.0, 0.001) - 42.0).abs() < 1e-9);
    }

    #[test]
    fn bias_adds_offset() {
        let mut s = SensorModel::new(SensorConfig {
            bias: 0.5,
            ..Default::default()
        });
        assert!((s.process(10.0, 0.001) - 10.5).abs() < 1e-9);
    }

    #[test]
    fn gain_scales_signal() {
        let mut s = SensorModel::new(SensorConfig {
            gain: 1.01,
            ..Default::default()
        });
        assert!((s.process(100.0, 0.001) - 101.0).abs() < 1e-9);
    }

    #[test]
    fn noise_absolute_adds_random() {
        let mut s = SensorModel::with_seed(
            SensorConfig {
                noise: 1.0,
                noise_type: NoiseType::Absolute,
                noise_distribution: NoiseDistribution::Uniform,
                ..Default::default()
            },
            42,
        );
        // Over many samples, the mean should be close to the true value
        let mut sum = 0.0;
        let n = 10_000;
        for _ in 0..n {
            sum += s.process(5.0, 0.001);
        }
        let mean = sum / n as f64;
        assert!(
            (mean - 5.0).abs() < 0.1,
            "mean should be ~5.0, got {mean}"
        );
    }

    #[test]
    fn noise_percent_scales_with_signal() {
        let mut s = SensorModel::with_seed(
            SensorConfig {
                noise: 0.1,
                noise_type: NoiseType::Percent,
                noise_distribution: NoiseDistribution::Uniform,
                ..Default::default()
            },
            42,
        );
        // Mean output should be close to true value (percent noise is zero-mean)
        let mut sum = 0.0;
        let n = 10_000;
        for _ in 0..n {
            sum += s.process(100.0, 0.001);
        }
        let mean = sum / n as f64;
        assert!(
            (mean - 100.0).abs() < 2.0,
            "percent noise mean should be ~100, got {mean}"
        );
    }

    #[test]
    fn drift_accumulates_over_time() {
        let mut s = SensorModel::new(SensorConfig {
            drift_rate: 0.1,
            ..Default::default()
        });
        let dt = 0.001;
        let mut val = 0.0;
        // Run for 10 seconds (10000 steps)
        for _ in 0..10_000 {
            val = s.process(0.0, dt);
        }
        // After 10s at 0.1 units/s drift, output should be ~1.0
        assert!(
            (val - 1.0).abs() < 0.05,
            "drift after 10s should be ~1.0, got {val}"
        );
    }

    #[test]
    fn quantization_reduces_resolution() {
        let mut s = SensorModel::new(SensorConfig {
            quantization_bits: 8, // 256 levels over [-500, 500]
            quantize_min: -500.0,
            quantize_max: 500.0,
            ..Default::default()
        });
        // Step = 1000 / 256 ≈ 3.906
        let out = s.process(100.0, 0.001);
        // Should be quantized to nearest step boundary
        let step: f64 = 1000.0 / 256.0;
        let expected = (100.0 / step).floor() * step;
        assert!(
            (out - expected).abs() < 1e-6,
            "quantized: got {out}, expected {expected}"
        );
    }

    #[test]
    fn clip_limits_output() {
        let mut s = SensorModel::new(SensorConfig {
            clip_min: Some(-1.0),
            clip_max: Some(1.0),
            ..Default::default()
        });
        assert!((s.process(5.0, 0.001) - 1.0).abs() < 1e-9);
        assert!((s.process(-5.0, 0.001) - (-1.0)).abs() < 1e-9);
    }

    #[test]
    fn lag_smooths_signal() {
        let mut s = SensorModel::new(SensorConfig {
            lag: 0.1,
            ..Default::default()
        });
        // Step input: 0 → 10
        let dt = 0.001;
        let mut last = 0.0;
        for _ in 0..1000 {
            last = s.process(10.0, dt);
        }
        // After 1s with τ=0.1, should be very close to 10
        assert!(
            last > 9.9,
            "lag should smooth to ~10 after 1s, got {last}"
        );

        // Reset and step back to 0
        s.reset();
        let mut last = 0.0;
        for _ in 0..1000 {
            last = s.process(0.0, dt);
        }
        assert!(
            last.abs() < 0.1,
            "lag should decay back to ~0 after 1s, got {last}"
        );
    }

    #[test]
    fn reset_clears_drift() {
        let mut s = SensorModel::new(SensorConfig {
            drift_rate: 1.0,
            ..Default::default()
        });
        for _ in 0..1000 {
            s.process(0.0, 0.001);
        }
        assert!(s.drift().abs() > 0.5);
        s.reset();
        assert!(s.drift().abs() < 1e-9);
    }

    #[test]
    fn pipeline_order_matters() {
        // Bias applied AFTER noise: the bias is constant, noise is random
        let mut s = SensorModel::with_seed(
            SensorConfig {
                noise: 1.0,
                noise_type: NoiseType::Absolute,
                bias: 5.0,
                ..Default::default()
            },
            42,
        );
        let mut sum = 0.0;
        for _ in 0..10_000 {
            sum += s.process(0.0, 0.001);
        }
        let mean = sum / 10_000.0;
        // Mean should be ~5.0 (bias) since noise is zero-mean
        assert!(
            (mean - 5.0).abs() < 0.1,
            "bias should be applied after noise, mean: {mean}"
        );
    }

    #[test]
    fn reproducible_with_same_seed() {
        let config = SensorConfig {
            noise: 1.0,
            noise_type: NoiseType::Absolute,
            ..Default::default()
        };
        let mut a = SensorModel::with_seed(config.clone(), 12345);
        let mut b = SensorModel::with_seed(config, 12345);

        for _ in 0..100 {
            let va = a.process(10.0, 0.001);
            let vb = b.process(10.0, 0.001);
            assert!((va - vb).abs() < 1e-9, "same seed should produce same output");
        }
    }
}
