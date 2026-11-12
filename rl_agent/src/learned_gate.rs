//! Learned phase-gating MLP: mission features → 4 phase logits.
//!
//! Input: 13-dim `MissionFeatures::to_array()` (float32).
//! Hidden: 32 units, tanh.
//! Output: 4 logits (Outbound, Capture, Return, Done).
//!
//! Trained with full-batch SGD on softmax cross-entropy over a seeded
//! synthetic exam, so the gate and its accuracy are reproducible.

use flight_core::{MissionFeatures, MissionPhase, MISSION_FEATURE_DIM};
use rand::{SeedableRng, Rng};
use rand::rngs::StdRng;

/// Seed for the synthetic exam. Fixed so every run sees the same samples.
pub const EXAM_SEED: u64 = 2026;

/// Seed for the gate's random initialisation.
pub const GATE_INIT_SEED: u64 = 2026;

/// Full-batch gradient-descent epochs used by [`train_gate`].
pub const TRAIN_EPOCHS: usize = 2000;

/// Learning rate used by [`train_gate`].
pub const TRAIN_LR: f32 = 0.5;

/// Label mapping: 0=Outbound,1=Capture,2=Return,3=Done
pub const PHASE_LABELS: [&str; 4] = ["outbound", "capture", "return", "done"];

pub fn phase_from_idx(i: usize) -> MissionPhase {
    match i {
        0 => MissionPhase::Outbound,
        1 => MissionPhase::Capture,
        2 => MissionPhase::Return,
        _ => MissionPhase::Done,
    }
}

pub fn idx_from_phase(p: MissionPhase) -> usize {
    match p {
        MissionPhase::Outbound => 0,
        MissionPhase::Capture => 1,
        MissionPhase::Return => 2,
        MissionPhase::Done => 3,
    }
}

/// Small MLP gate.
pub struct GateMLP {
    w1: Vec<Vec<f32>>, // [13][32]
    b1: Vec<f32>,       // [32]
    w2: Vec<Vec<f32>>, // [32][4]
    b2: Vec<f32>,       // [4]
}

impl GateMLP {
    /// Randomly initialise the gate from `seed` (Glorot-uniform weights).
    pub fn new(seed: u64) -> Self {
        let mut rng = StdRng::seed_from_u64(seed);
        let limit1 = (6.0 / (MISSION_FEATURE_DIM + 32) as f32).sqrt();
        let limit2 = (6.0 / (32 + 4) as f32).sqrt();

        let mut w1 = vec![vec![0.0f32; 32]; MISSION_FEATURE_DIM];
        for row in &mut w1 {
            for v in row {
                *v = rng.gen_range(-limit1..limit1);
            }
        }
        let b1 = vec![0.0f32; 32];

        let mut w2 = vec![vec![0.0f32; 4]; 32];
        for row in &mut w2 {
            for v in row {
                *v = rng.gen_range(-limit2..limit2);
            }
        }
        let b2 = vec![0.0f32; 4];

        Self { w1, b1, w2, b2 }
    }

    /// Forward pass: input array (13) -> 4 logits.
    pub fn forward(&self, x: &[f32]) -> [f32; 4] {
        assert!(x.len() == MISSION_FEATURE_DIM);
        let mut h = vec![0.0f32; 32];
        for j in 0..32 {
            let mut s = self.b1[j];
            for i in 0..MISSION_FEATURE_DIM {
                s += x[i] * self.w1[i][j];
            }
            h[j] = s.tanh();
        }
        let mut out = [0.0f32; 4];
        for j in 0..4 {
            let mut s = self.b2[j];
            for i in 0..32 {
                s += h[i] * self.w2[i][j];
            }
            out[j] = s;
        }
        out
    }

    /// Predict phase index via argmax.
    pub fn predict_idx(&self, x: &[f32]) -> usize {
        let logits = self.forward(x);
        let mut best = 0;
        for j in 1..4 {
            if logits[j] > logits[best] {
                best = j;
            }
        }
        best
    }

    /// Predict expert label.
    pub fn predict_label(&self, x: &[f32]) -> String {
        PHASE_LABELS[self.predict_idx(x)].into()
    }

    /// Helper to rebuild hidden for numeric update.
    fn hidden_for(&self, x: &[f32]) -> Vec<f32> {
        let mut h = vec![0.0f32; 32];
        for j in 0..32 {
            let mut s = self.b1[j];
            for i in 0..MISSION_FEATURE_DIM {
                s += x[i] * self.w1[i][j];
            }
            h[j] = s.tanh();
        }
        h
    }

    /// Train with full-batch SGD on softmax cross-entropy.
    ///
    /// Every sample contributes to the gradient each epoch; the update is the
    /// mean gradient over the batch. The loop is deterministic for a fixed
    /// initial gate and sample order.
    pub fn train_sgd(&mut self, samples: &[(Vec<f32>, usize)], epochs: usize, lr: f32) {
        if samples.is_empty() {
            return;
        }
        let n = samples.len() as f32;
        let mut grad_w1 = vec![vec![0.0f32; 32]; MISSION_FEATURE_DIM];
        let mut grad_b1 = vec![0.0f32; 32];
        let mut grad_w2 = vec![vec![0.0f32; 4]; 32];
        let mut grad_b2 = vec![0.0f32; 4];

        for _ in 0..epochs {
            for g in &mut grad_w1 {
                g.fill(0.0);
            }
            grad_b1.fill(0.0);
            for g in &mut grad_w2 {
                g.fill(0.0);
            }
            grad_b2.fill(0.0);

            for (x, label) in samples {
                let h = self.hidden_for(x);
                let logits = self.forward(x);

                // Softmax cross-entropy gradient wrt logits: p - onehot.
                let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                let mut exp = [0.0f32; 4];
                let mut denom = 0.0f32;
                for j in 0..4 {
                    exp[j] = (logits[j] - max).exp();
                    denom += exp[j];
                }
                let mut dz2 = [0.0f32; 4];
                for j in 0..4 {
                    let p = exp[j] / denom;
                    let target = if j == *label { 1.0 } else { 0.0 };
                    dz2[j] = (p - target) / n;
                }

                // Output layer.
                for i in 0..32 {
                    for j in 0..4 {
                        grad_w2[i][j] += h[i] * dz2[j];
                    }
                }
                for j in 0..4 {
                    grad_b2[j] += dz2[j];
                }

                // Backprop through tanh hidden layer.
                let mut dz1 = [0.0f32; 32];
                for i in 0..32 {
                    let mut dh = 0.0f32;
                    for j in 0..4 {
                        dh += self.w2[i][j] * dz2[j];
                    }
                    dz1[i] = dh * (1.0 - h[i] * h[i]);
                }
                for i in 0..MISSION_FEATURE_DIM {
                    for j in 0..32 {
                        grad_w1[i][j] += x[i] * dz1[j];
                    }
                }
                for j in 0..32 {
                    grad_b1[j] += dz1[j];
                }
            }

            for i in 0..MISSION_FEATURE_DIM {
                for j in 0..32 {
                    self.w1[i][j] -= lr * grad_w1[i][j];
                }
            }
            for j in 0..32 {
                self.b1[j] -= lr * grad_b1[j];
            }
            for i in 0..32 {
                for j in 0..4 {
                    self.w2[i][j] -= lr * grad_w2[i][j];
                }
            }
            for j in 0..4 {
                self.b2[j] -= lr * grad_b2[j];
            }
        }
    }
}

impl Clone for GateMLP {
    fn clone(&self) -> Self {
        Self {
            w1: self.w1.clone(),
            b1: self.b1.clone(),
            w2: self.w2.clone(),
            b2: self.b2.clone(),
        }
    }
}

/// Build the seeded synthetic 30-scenario exam.
///
/// The sample set is identical on every call. Return and Done deliberately
/// share `f[7] = 2.0` and differ only at `f[9]` (0.0 vs 0.05), so the gate has
/// to learn the time-remaining split rather than the phase one-hot alone.
pub fn exam_scenarios() -> Vec<(Vec<f32>, usize)> {
    let mut rng = StdRng::seed_from_u64(EXAM_SEED);
    let mut samples = Vec::with_capacity(30);
    // Outbound (0): phase=0.0, others random
    for _ in 0..7 {
        let mut f = [0.0f32; MISSION_FEATURE_DIM];
        f[7] = 0.0;
        f[0] = rng.gen_range(0.0..0.8);
        f[10] = rng.gen_range(0.0..1.0);
        samples.push((f.to_vec(), 0));
    }
    // Capture (1): phase=1.0
    for _ in 0..8 {
        let mut f = [0.0f32; MISSION_FEATURE_DIM];
        f[7] = 1.0;
        f[0] = rng.gen_range(0.1..0.4);
        f[11] = rng.gen_range(0.0..1.0);
        samples.push((f.to_vec(), 1));
    }
    // Return (2): phase=2.0, no time-remaining signal
    for _ in 0..7 {
        let mut f = [0.0f32; MISSION_FEATURE_DIM];
        f[7] = 2.0;
        f[5] = rng.gen_range(0.0..0.5);
        samples.push((f.to_vec(), 2));
    }
    // Done (3): same phase encoding as Return, low time remaining
    for _ in 0..8 {
        let mut f = [0.0f32; MISSION_FEATURE_DIM];
        f[7] = 2.0;
        f[9] = 0.05;
        samples.push((f.to_vec(), 3));
    }
    samples
}

/// Fraction of `samples` the gate classifies correctly.
pub fn evaluate_gate(gate: &GateMLP, samples: &[(Vec<f32>, usize)]) -> f64 {
    if samples.is_empty() {
        return 0.0;
    }
    let correct = samples
        .iter()
        .filter(|(x, label)| gate.predict_idx(x) == *label)
        .count();
    correct as f64 / samples.len() as f64
}

/// Train a gate from a random initialisation to the exam.
pub fn train_gate() -> GateMLP {
    let mut gate = GateMLP::new(GATE_INIT_SEED);
    let exam = exam_scenarios();
    gate.train_sgd(&exam, TRAIN_EPOCHS, TRAIN_LR);
    gate
}

/// Convert MissionFeatures array (f64) to f32 input.
pub fn features_to_input(f: &MissionFeatures) -> [f32; MISSION_FEATURE_DIM] {
    let arr = f.to_array();
    let mut out = [0.0f32; MISSION_FEATURE_DIM];
    for i in 0..MISSION_FEATURE_DIM {
        out[i] = arr[i] as f32;
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn trained_gate_beats_the_chance_baseline() {
        let gate = train_gate();
        let exam = exam_scenarios();
        let accuracy = evaluate_gate(&gate, &exam);
        let correct = (accuracy * exam.len() as f64).round() as usize;
        println!(
            "learned gate: {correct}/{} correct ({:.1}%) after {TRAIN_EPOCHS} epochs at lr {TRAIN_LR}",
            exam.len(),
            accuracy * 100.0
        );
        // Four balanced classes => chance is 25%.
        assert!(
            accuracy >= 0.90,
            "gate accuracy {accuracy:.3} is below 0.90 (chance = 0.25)"
        );
    }

    #[test]
    fn trained_gate_is_seeded_and_reproducible() {
        let first = train_gate();
        let second = train_gate();
        for (x, _) in exam_scenarios() {
            assert_eq!(
                first.forward(&x),
                second.forward(&x),
                "training must be deterministic for a fixed seed"
            );
            assert_eq!(first.predict_idx(&x), second.predict_idx(&x));
        }
    }

    #[test]
    fn exam_scenarios_are_deterministic() {
        let first = exam_scenarios();
        let second = exam_scenarios();
        assert_eq!(first.len(), 30);
        assert_eq!(first, second, "the exam must not change between calls");
    }
}
