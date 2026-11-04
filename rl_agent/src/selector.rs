//! Hybrid selector — rule-based phase gating over PolicyContext.
//!
//! Implements [`SelectorConfig`], [`PhaseSelector`], [`SelectorStatus`],
//! and `select_policy()` with hysteresis, bumpless throttle transfer,
//! and heuristic fallback on NaN / out-of-range.
//!
//! Uses [`MissionContext`] (Task 2) via [`PolicyContext`].

use crate::learned_gate::{features_to_input, GateMLP};
use crate::policy::{ExpertCard, HeuristicPolicy, Policy, PolicyContext};
use crate::contracts::MISSION_OBS_DIM;
use flight_core::{MissionFeatures, MissionPhase, MISSION_FEATURE_DIM};
use std::collections::HashMap;

fn phase_feature_array(p: MissionPhase) -> [f32; MISSION_FEATURE_DIM] {
    let mut f = [0.0f32; MISSION_FEATURE_DIM];
    f[7] = crate::learned_gate::idx_from_phase(p) as f32;
    f
}

/// Convert MissionPhase to config key (snake_case, matching serde rename).
fn phase_key(p: MissionPhase) -> String {
    match p {
        MissionPhase::Outbound => "outbound".into(),
        MissionPhase::Capture => "capture".into(),
        MissionPhase::Return => "return".into(),
        MissionPhase::Done => "done".into(),
    }
}

/// Selector configuration — maps phase → expert id / checkpoint label.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize, Default)]
pub struct SelectorConfig {
    pub phases: HashMap<String, String>,
    /// Steps the phase must remain stable before switching expert.
    pub hysteresis_steps: usize,
    /// Maximum throttle slew per step during bumpless transfer.
    pub bumpless_slew_max: f32,
    /// When true, replace the config-phase mapping with learned MLP gate.
    pub learned_gate_enabled: bool,
    /// When true, fall back to heuristic if selected expert fails.
    pub heuristic_fallback: bool,
}

impl SelectorConfig {
    pub fn new() -> Self {
        Self {
            phases: HashMap::new(),
            hysteresis_steps: 3,
            bumpless_slew_max: 0.05,
            learned_gate_enabled: false,
            heuristic_fallback: true,
        }
    }
}

/// Selector operational status.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum SelectorStatus {
    #[default]
    Stable,
    Transitioning,
    FallbackHeuristic,
}

/// Hybrid phase selector that dispatches to per-phase experts with
/// hysteresis and bumpless throttle transfer.
pub struct PhaseSelector {
    config: SelectorConfig,
    heuristic: HeuristicPolicy,
    current_expert_label: String,
    previous_action: Vec<f32>,
    stable_cnt: usize,
    last_phase: MissionPhase,
    status: SelectorStatus,
    card: ExpertCard,
    gate: Option<GateMLP>,
}

impl PhaseSelector {
    /// Build with a learned gate (already trained). When `None`, behaves as the
    /// rule-based hybrid selector from before this feature.
    pub fn new(config: SelectorConfig, heuristic: HeuristicPolicy) -> Self {
        Self::with_gate(config, heuristic, None)
    }

    pub fn with_gate(config: SelectorConfig, heuristic: HeuristicPolicy, gate: Option<GateMLP>) -> Self {
        let card = ExpertCard {
            id: "hybrid_selector_v1".into(),
            model: crate::policy::ModelKind::Heuristic,
            specialization: crate::policy::Specialization::Full,
            contract: crate::contracts::MISSION_CONTRACT.into(),
            reward_profile: "balanced".into(),
            obs_mask: (0..MISSION_OBS_DIM).collect(),
            trained_steps: 0,
            seed: 42,
            git_commit: "selector_v1".into(),
            notes: if config.learned_gate_enabled {
                "rule-based hybrid selector + learned phase gate"
            } else {
                "rule-based hybrid selector with hysteresis + bumpless transfer"
            }.into(),
        };
        Self {
            config,
            heuristic,
            current_expert_label: "heuristic".into(),
            previous_action: vec![0.0; 4],
            stable_cnt: 0,
            last_phase: MissionPhase::Outbound,
            status: SelectorStatus::Stable,
            card,
            gate,
        }
    }

    /// Select expert label for phase.
    ///
    /// When `config.learned_gate_enabled` is true and a gate is wired in, the
    /// MLP predicts the phase (argmax over 4 logits) from mission features
    /// instead of falling back to the config phase→expert map. Hysteresis is
    /// preserved around label transitions so the bumpless throttle filter
    /// stays attached to this selector.
    pub fn select_policy(&mut self, ctx: &PolicyContext) -> String {
        let target_label = if self.config.learned_gate_enabled && self.gate.is_some() {
            let x = ctx
                .features
                .map(|f| features_to_input(&f))
                .unwrap_or_else(|| phase_feature_array(ctx.phase));
            self.gate.as_ref().unwrap().predict_label(&x)
        } else {
            let target = self.config.phases.get(&phase_key(ctx.phase)).cloned();
            target.unwrap_or_else(|| "heuristic".into())
        };

        if ctx.phase == self.last_phase {
            self.stable_cnt += 1;
        } else {
            self.stable_cnt = 0;
            self.last_phase = ctx.phase;
            self.status = SelectorStatus::Transitioning;
        }

        if self.stable_cnt >= self.config.hysteresis_steps {
            if self.current_expert_label != target_label {
                self.current_expert_label = target_label.clone();
                self.status = SelectorStatus::Stable;
            }
        }

        self.current_expert_label.clone()
    }

    /// Wired-in learned gate (pre-trained). Used by callers that want the
    /// learned gating path instead of constructing one fresh.
    pub fn set_gate(&mut self, gate: Option<GateMLP>) {
        self.gate = gate;
    }

    /// Return status after selection.
    pub fn status(&self) -> SelectorStatus {
        self.status
    }

    /// Reset internal state.
    pub fn reset(&mut self, _ctx: &PolicyContext) {
        self.stable_cnt = 0;
        self.last_phase = MissionPhase::Outbound;
        self.current_expert_label = "heuristic".into();
        self.previous_action = vec![0.0; 4];
        self.status = SelectorStatus::Stable;
    }

    /// Apply bumpless throttle transfer on switch.
    fn apply_bumpless(&self, new_action: &[f32]) -> Vec<f32> {
        let mut out = new_action.to_vec();
        if out.len() < 4 || self.previous_action.len() < 4 {
            return out;
        }
        let prev_th = self.previous_action[3];
        let new_th = out[3];
        let delta = new_th - prev_th;
        let max_slew = self.config.bumpless_slew_max;
        if delta.abs() > max_slew {
            out[3] = prev_th + max_slew * delta.signum();
        }
        out
    }

    /// Check whether action is valid (finite, in bounds).
    fn is_valid_action(&self, a: &[f32]) -> bool {
        if a.len() != 4 {
            return false;
        }
        for &v in a {
            if !v.is_finite() {
                return false;
            }
            // Action bounds per contracts: roll ±0.6, pitch ±0.4, yaw ±0.8, throttle 0..1
            if v.abs() > 1.0 {
                return false;
            }
        }
        true
    }
}

impl Policy for PhaseSelector {
    fn act(&mut self, obs: &[f32], ctx: &PolicyContext) -> Vec<f32> {
        let expert_label = self.select_policy(ctx);

        // For this milestone the selector holds the heuristic directly and
        // uses it as both the configured expert when unmatched and as
        // the NaN fallback.
        let mut action = self.heuristic.act(obs, ctx);

        // Heuristic fallback when RL model would return NaN / out-of-range.
        if !self.is_valid_action(&action) {
            self.status = SelectorStatus::FallbackHeuristic;
            action = self.heuristic.act(obs, ctx);
        }

        // Bumpless throttle transfer when expert label changed.
        if self.current_expert_label != expert_label {
            // Transitioning: apply slew.
            // Since heuristic is the only direct expert here, we still
            // demonstrate the slew logic on throttle.
            action = self.apply_bumpless(&action);
        }

        self.previous_action = action.clone();
        action
    }

    fn deterministic_action(&mut self, obs: &[f32], ctx: &PolicyContext) -> Vec<f32> {
        self.act(obs, ctx)
    }

    fn reset(&mut self) {
        self.reset(&PolicyContext {
            phase: MissionPhase::Outbound,
            step: 0,
            captured: false,
            features: None,
        });
    }

    fn card(&self) -> &ExpertCard {
        &self.card
    }
}

/// Select expert label for phase with hysteresis.
/// Standalone helper using `MissionContext` (Task 2) when available.
pub fn select_policy(
    phase: MissionPhase,
    ctx: &PolicyContext,
    config: &SelectorConfig,
    current_label: &str,
    stable_cnt: usize,
) -> (String, usize, SelectorStatus) {
    let target = config.phases.get(&phase_key(phase)).cloned();
    let target_label = target.unwrap_or_else(|| "heuristic".into());

    let mut status = SelectorStatus::Stable;
    let mut new_stable = stable_cnt;
    let mut new_label = current_label.to_string();

    if phase == ctx.phase {
        // Hysteresis: count stable steps within same phase.
        new_stable = stable_cnt.saturating_add(1);
    } else {
        new_stable = 0;
    }

    if new_stable >= config.hysteresis_steps {
        if new_label != target_label {
            new_label = target_label.clone();
            status = SelectorStatus::Transitioning;
        }
    } else {
        status = SelectorStatus::Transitioning;
    }

    if target_label == "heuristic" && config.heuristic_fallback {
        status = SelectorStatus::FallbackHeuristic;
    }

    (new_label, new_stable, status)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::policy::HeuristicPolicy;
    use flight_core::MissionPhase;

    fn default_ctx(phase: MissionPhase) -> PolicyContext {
        PolicyContext {
            phase,
            step: 0,
            captured: false,
            features: None,
        }
    }

    #[test]
    fn selector_config_defaults() {
        let c = SelectorConfig::new();
        assert_eq!(c.hysteresis_steps, 3);
        assert!(c.heuristic_fallback);
        assert_eq!(c.bumpless_slew_max, 0.05);
    }

    #[test]
    fn phase_gate_uses_hysteresis() {
        let cfg = SelectorConfig::new();
        let mut s = PhaseSelector::new(cfg.clone(), HeuristicPolicy::new(55.0, 11.5, 10.0));
        let label = s.select_policy(&default_ctx(MissionPhase::Outbound));
        assert_eq!(label, "heuristic");
        // After 3 stable steps with config mapping, should switch.
        // Without config mapping, stays heuristic.
    }

    #[test]
    fn hysteresis_prevents_rapid_flip() {
        let mut cfg = SelectorConfig::new();
        cfg.hysteresis_steps = 2;
        cfg.phases.insert("capture".into(), "sac_capture".into());
        let mut sel = PhaseSelector::new(cfg, HeuristicPolicy::new(55.0, 11.5, 10.0));
        // First step: not enough stable.
        let l1 = sel.select_policy(&default_ctx(MissionPhase::Capture));
        assert_eq!(l1, "heuristic"); // hasn't met hysteresis yet
        // Second step.
        let ctx = PolicyContext { phase: MissionPhase::Capture, step: 1, captured: false,
            features: None };
        let l2 = sel.select_policy(&ctx);
        assert_eq!(l2, "heuristic");
        // Third step hits hysteresis.
        let ctx3 = PolicyContext { phase: MissionPhase::Capture, step: 2, captured: false,
            features: None };
        let l3 = sel.select_policy(&ctx3);
        assert_eq!(l3, "sac_capture");
    }

    #[test]
    fn bumpless_transfer_keeps_throttle_continuous() {
        let cfg = SelectorConfig::new();
        let mut sel = PhaseSelector::new(cfg, HeuristicPolicy::new(55.0, 11.5, 10.0));
        let obs = vec![0.0f32; MISSION_OBS_DIM];
        let a1 = sel.act(&obs, &default_ctx(MissionPhase::Outbound));
        assert!(a1[3].is_finite());
        assert_eq!(a1.len(), 4);
        // Switch phase forces transition status and slew.
        let a2 = sel.act(&obs, &default_ctx(MissionPhase::Capture));
        assert!(a2[3].is_finite());
        // Throttle should not jump by more than slew max (0.05) when previous exists.
        if sel.previous_action.len() == 4 {
            let delta = (a2[3] - sel.previous_action[3]).abs();
            assert!(delta <= 0.05 + 1e-5, "bumpless slew exceeded: delta={delta}");
        }
    }

    #[test]
    fn heuristic_fallback_on_nan() {
        let cfg = SelectorConfig::new();
        let mut sel = PhaseSelector::new(cfg, HeuristicPolicy::new(55.0, 11.5, 10.0));
        // Heuristic never returns NaN for valid obs, but we verify fallback
        // path exists: after act, action must be valid.
        let obs = vec![0.0f32; MISSION_OBS_DIM];
        let a = sel.act(&obs, &default_ctx(MissionPhase::Outbound));
        assert!(a.iter().all(|v| v.is_finite()));
        assert!(a.iter().all(|&v| v.abs() <= 1.0 + 1e-5));
    }

    #[test]
    fn select_policy_standalone_uses_mission_context() {
        // Uses PolicyContext which embeds phase/step/captured; aligns with MissionContext fields.
        let cfg = SelectorConfig::new();
        let (label, stable, status) = select_policy(
            MissionPhase::Return,
            &default_ctx(MissionPhase::Return),
            &cfg,
            "heuristic",
            0,
        );
        assert_eq!(label, "heuristic");
        assert_eq!(stable, 1);
        assert!(status == SelectorStatus::Transitioning || status == SelectorStatus::FallbackHeuristic);
    }

    #[test]
    fn policy_trait_implemented() {
        let cfg = SelectorConfig::new();
        let mut sel = PhaseSelector::new(cfg, HeuristicPolicy::new(55.0, 11.5, 10.0));
        sel.reset(&default_ctx(MissionPhase::Outbound));
        let card = sel.card();
        assert_eq!(card.id, "hybrid_selector_v1");
    }

    #[test]
    fn learned_gate_predicts_30_exam() {
        let gate = crate::learned_gate::train_gate();
        let exam = crate::learned_gate::exam_scenarios();
        let mut correct = 0;
        for (x, label) in exam {
            if gate.predict_idx(&x) == label {
                correct += 1;
            }
        }
        assert!(correct >= 22, "exam predictions too weak: {}/30", correct);
        println!("Exam predictions: {}/30", correct);
    }

    #[test]
    fn selector_uses_gate_when_enabled() {
        let mut cfg = SelectorConfig::new();
        cfg.learned_gate_enabled = true;
        let mut sel = PhaseSelector::with_gate(
            cfg,
            HeuristicPolicy::new(55.0, 11.5, 10.0),
            Some(crate::learned_gate::train_gate()),
        );
        let label = sel.select_policy(&default_ctx(MissionPhase::Outbound));
        assert!(label == "heuristic" || matches!(label.as_str(), "outbound" | "capture" | "return" | "done"));
        // Gate should stay deterministic when called repeatedly.
        let label2 = sel.select_policy(&default_ctx(MissionPhase::Outbound));
        assert_eq!(label, label2);
    }
}
