//! Welford online observation normalizer.
//!
//! Raw sensor observations have wildly different scales (baro altitude in
//! hundreds of meters, gyro rates in rad/s, GPS fix quality 0..3), which
//! hurts MLP training. During rollout collection we keep a running
//! mean/variance per channel (Welford's algorithm — numerically stable,
//! one pass), then normalize with the *final* stats at update time and
//! again for evaluation. Values are clipped to `[-clip, +clip]` so a
//! single noisy channel cannot dominate the loss.

use serde::{Deserialize, Serialize};

/// Default clip bound for normalized observations.
pub const DEFAULT_CLIP: f64 = 5.0;

/// A serializable running normalizer (Welford's online mean / M2 variance).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RunningNormalizer {
    /// Number of observations seen.
    pub count: usize,
    /// Running per-channel mean.
    pub mean: Vec<f64>,
    /// Running sum of squared differences from the current mean.
    pub m2: Vec<f64>,
    /// Dimensionality of a single observation.
    pub dim: usize,
    /// Symmetric clip bound applied on normalize.
    pub clip: f64,
}

impl RunningNormalizer {
    /// Create a fresh normalizer for `dim`-dimensional observations.
    pub fn new(dim: usize) -> Self {
        Self {
            count: 0,
            mean: vec![0.0; dim],
            m2: vec![0.0; dim],
            dim,
            clip: DEFAULT_CLIP,
        }
    }

    /// Fold one observation into the running statistics.
    pub fn observe(&mut self, obs: &[f64]) {
        debug_assert_eq!(obs.len(), self.dim, "observation dimension mismatch");
        self.count += 1;
        let count = self.count as f64;
        for (i, &x) in obs.iter().enumerate() {
            let delta = x - self.mean[i];
            self.mean[i] += delta / count;
            self.m2[i] += delta * (x - self.mean[i]);
        }
    }

    /// Per-channel population standard deviation (floor 1e-6 to stay finite).
    pub fn std(&self) -> Vec<f64> {
        self.mean
            .iter()
            .zip(self.m2.iter())
            .map(|(&_m, &m2)| {
                let variance = if self.count > 1 { m2 / self.count as f64 } else { 1.0 };
                variance.max(1e-6).sqrt()
            })
            .collect()
    }

    /// Normalize one observation (f64) to `(x - mean)/std`, clipped.
    pub fn normalize(&self, obs: &[f64]) -> Vec<f64> {
        debug_assert_eq!(obs.len(), self.dim, "observation dimension mismatch");
        let std = self.std();
        let mut out = vec![0.0; self.dim];
        for i in 0..self.dim {
            let z = if std[i] > 1e-9 {
                (obs[i] - self.mean[i]) / std[i]
            } else {
                // Unexercised channel: leave as-is (still subtract mean).
                obs[i] - self.mean[i]
            };
            out[i] = z.clamp(-self.clip, self.clip);
        }
        out
    }

    /// Normalize a single observation carried as `f32`.
    pub fn normalize_f32(&self, obs: &[f32]) -> Vec<f32> {
        let f64_in: Vec<f64> = obs.iter().map(|&x| x as f64).collect();
        self.normalize(&f64_in)
            .into_iter()
            .map(|x| x as f32)
            .collect()
    }

    /// Normalize a flattened batch `obs` of `n*dim` floats into a flattened
    /// `n*dim` vector. Used at update time with the final running stats.
    pub fn normalize_batch(&self, obs: &[f32]) -> Vec<f32> {
        debug_assert_eq!(
            obs.len() % self.dim,
            0,
            "batch length must be a multiple of the observation dimension"
        );
        let std = self.std();
        let mut out = vec![0.0f32; obs.len()];
        for (chunk, out_chunk) in obs.chunks_exact(self.dim).zip(out.chunks_exact_mut(self.dim)) {
            for i in 0..self.dim {
                let z = if std[i] > 1e-9 {
                    (chunk[i] as f64 - self.mean[i]) / std[i]
                } else {
                    chunk[i] as f64 - self.mean[i]
                };
                out_chunk[i] = z.clamp(-self.clip, self.clip) as f32;
            }
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn welford_matches_direct_mean_std() {
        let dim = 3;
        let mut norm = RunningNormalizer::new(dim);
        let data = [
            [1.0, 10.0, -5.0],
            [2.0, 12.0, -3.0],
            [3.0, 11.0, -8.0],
            [4.0, 9.0, -2.0],
        ];
        for obs in &data {
            norm.observe(obs);
        }
        for i in 0..dim {
            let mean: f64 = data.iter().map(|o| o[i]).sum::<f64>() / 4.0;
            let var: f64 = data.iter().map(|o| (o[i] - mean).powi(2)).sum::<f64>() / 4.0;
            let std = var.sqrt();
            assert!(
                (norm.mean[i] - mean).abs() < 1e-12,
                "mean[{i}] {:.12} != {:.12}",
                norm.mean[i],
                mean
            );
            assert!(
                (norm.std()[i] - std).abs() < 1e-12,
                "std[{i}] {:.12} != {:.12}",
                norm.std()[i],
                std
            );
        }
    }

    #[test]
    fn normalize_produces_unit_stats_on_seen_data() {
        let mut norm = RunningNormalizer::new(2);
        for i in 0..100 {
            norm.observe(&[i as f64, (i as f64).sin()]);
        }
        let obs = norm.normalize(&[norm.mean[0], norm.mean[1]]);
        assert!(obs[0].abs() < 1e-6);
        assert!(obs[1].abs() < 1e-6);
        // A far-out point gets clipped.
        let far = norm.normalize(&[norm.mean[0] + 100.0 * norm.std()[0], 0.0]);
        assert_eq!(far[0], norm.clip);
    }

    #[test]
    fn serde_round_trip_preserves_stats() {
        let mut norm = RunningNormalizer::new(3);
        for i in 0..50 {
            norm.observe(&[i as f64, -(i as f64), i as f64 * 0.5]);
        }
        let json = serde_json::to_string(&norm).unwrap();
        let back: RunningNormalizer = serde_json::from_str(&json).unwrap();
        assert_eq!(back.count, norm.count);
        assert_eq!(back.mean, norm.mean);
        assert_eq!(back.m2, norm.m2);
        assert_eq!(back.dim, norm.dim);
    }

    #[test]
    fn batch_normalize_matches_per_sample() {
        let mut norm = RunningNormalizer::new(2);
        for i in 0..20 {
            norm.observe(&[i as f64, 2.0 * i as f64]);
        }
        let batch = vec![1.0f32, 50.0, 3.0, -7.0];
        let flat = norm.normalize_batch(&batch);
        let per_a = norm.normalize_f32(&batch[0..2]);
        let per_b = norm.normalize_f32(&batch[2..4]);
        assert_eq!(flat, [per_a, per_b].concat());
    }
}