//! Policy abstractions for RL agent (Task 10).
//! Defines [`Policy`] trait, [`RlPolicy`], [`HeuristicPolicy`], and metadata.

use crate::contracts::{action_to_avionics, AVIONICS_CONTRACT, MISSION_OBS_DIM};
use crate::{algo::Algorithm, normalize::RunningNormalizer};
use burn::tensor::backend::AutodiffBackend;
use flight_core::{MissionFeatures, MissionPhase, MissionContext};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ModelKind {
    Ppo,
    A2c,
    Sac,
    Td3,
    D3qn,
    Heuristic,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Specialization {
    Full,
    Outbound,
    Capture,
    Return,
}

#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct ExpertCard {
    pub id: String,
    pub model: ModelKind,
    pub specialization: Specialization,
    pub contract: String,
    pub reward_profile: String,
    pub obs_mask: Vec<usize>,
    pub trained_steps: usize,
    pub seed: u64,
    pub git_commit: String,
    pub notes: String,
}

impl ExpertCard {
    /// Build the card for a run that is training right now (there is no
    /// `expert.json` on disk yet).
    pub fn for_training(
        algo_id: &str,
        obs_dim: usize,
        reward_profile: &str,
        specialization: &str,
        seed: u64,
    ) -> Self {
        let model = match algo_id {
            "ppo" => ModelKind::Ppo,
            "a2c" => ModelKind::A2c,
            "sac" => ModelKind::Sac,
            "td3" => ModelKind::Td3,
            "d3qn" => ModelKind::D3qn,
            _ => ModelKind::Ppo,
        };
        let specialization = match specialization {
            "outbound" => Specialization::Outbound,
            "capture" => Specialization::Capture,
            "return" => Specialization::Return,
            _ => Specialization::Full,
        };
        let contract = if obs_dim == MISSION_OBS_DIM {
            crate::contracts::MISSION_CONTRACT
        } else {
            AVIONICS_CONTRACT
        };
        Self {
            id: format!("{algo_id}-{seed}"),
            model,
            specialization,
            contract: contract.into(),
            reward_profile: reward_profile.into(),
            obs_mask: (0..obs_dim).collect(),
            trained_steps: 0,
            seed,
            git_commit: String::new(),
            notes: String::new(),
        }
    }

    /// Load the card persisted beside a checkpoint, or reconstruct an honest
    /// one from `net.json` when no `expert.json` exists.
    ///
    /// Legacy avionics checkpoints used to be reported as 32-channel mission
    /// experts. The observation width in `net.json` is the ground truth, so
    /// the contract and mask come from there. A malformed `expert.json` is an
    /// error rather than a panic.
    pub fn from_checkpoint(
        dir: &str,
        algo_id: &str,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        let expert_path = std::path::Path::new(dir).join("expert.json");
        if expert_path.exists() {
            let content = std::fs::read_to_string(&expert_path)?;
            return Ok(serde_json::from_str(&content)?);
        }
        let net: crate::net::NetConfig = serde_json::from_str(&std::fs::read_to_string(
            std::path::Path::new(dir).join("net.json"),
        )?)?;
        let model = match algo_id {
            "ppo" => ModelKind::Ppo,
            "a2c" => ModelKind::A2c,
            "sac" => ModelKind::Sac,
            "td3" => ModelKind::Td3,
            "d3qn" => ModelKind::D3qn,
            _ => ModelKind::Ppo,
        };
        let (contract, obs_mask) = if net.obs_dim == MISSION_OBS_DIM {
            (
                crate::contracts::MISSION_CONTRACT,
                (0..MISSION_OBS_DIM).collect(),
            )
        } else {
            (AVIONICS_CONTRACT, (0..net.obs_dim).collect())
        };
        Ok(Self {
            id: "legacy".into(),
            model,
            specialization: Specialization::Full,
            contract: contract.into(),
            reward_profile: "balanced".into(),
            obs_mask,
            trained_steps: 0,
            seed: 0,
            git_commit: String::new(),
            notes: String::new(),
        })
    }
}

#[derive(Debug, Clone, Copy, Default)]
pub struct PolicyContext {
    pub phase: MissionPhase,
    pub step: usize,
    pub captured: bool,
    /// Optional mission features (13-dim). Ignored when `None`; the selector
    /// falls back to phase-only gating in that case.
    pub features: Option<MissionFeatures>,
}

/// A policy returns **physical** commands (not tanh-space).
pub trait Policy {
    fn act(&mut self, obs: &[f32], ctx: &PolicyContext) -> Vec<f32>;
    fn deterministic_action(&mut self, obs: &[f32], ctx: &PolicyContext) -> Vec<f32>;
    fn reset(&mut self);
    fn card(&self) -> &ExpertCard;
}

#[derive(Debug, Clone, Copy)]
pub struct HeuristicConfig {
    pub cruise_airspeed: f64,
    pub capture_speed: f64,
    pub v_stall: f64,
    pub bank_gain: f64,
    pub pitch_gain: f64,
    pub speed_gain: f64,
}

impl Default for HeuristicConfig {
    fn default() -> Self {
        Self {
            cruise_airspeed: 55.0,
            capture_speed: 11.5,
            v_stall: 10.0,
            bank_gain: 1.2,
            pitch_gain: 0.0015,
            speed_gain: 0.03,
        }
    }
}

#[derive(Debug, Clone)]
pub struct HeuristicPolicy {
    cfg: HeuristicConfig,
    card: ExpertCard,
}

impl HeuristicPolicy {
    pub fn new(cruise_airspeed: f64, capture_speed: f64, v_stall: f64) -> Self {
        let mut cfg = HeuristicConfig::default();
        cfg.cruise_airspeed = cruise_airspeed;
        cfg.capture_speed = capture_speed;
        cfg.v_stall = v_stall;
        Self {
            cfg,
            card: ExpertCard {
                id: "heuristic_v1".into(),
                model: ModelKind::Heuristic,
                specialization: Specialization::Full,
                contract: crate::contracts::MISSION_CONTRACT.into(),
                reward_profile: "balanced".into(),
                obs_mask: (0..MISSION_OBS_DIM).collect(),
                trained_steps: 0,
                seed: 7,
                git_commit: "deadbeef".into(),
                notes: "baseline".into(),
            },
        }
    }
}

impl Policy for HeuristicPolicy {
    fn act(&mut self, obs: &[f32], ctx: &PolicyContext) -> Vec<f32> {
        let airspeed = obs.get(crate::contracts::OBS_AIRSPEED_INDICATED).copied().unwrap_or(0.0) as f64;
        let airspeed = if airspeed.is_finite() { airspeed } else { 0.0 };
        let phase = ctx.phase;
        let bearing_sin = if phase == MissionPhase::Return {
            obs.get(crate::contracts::OBS_HOME_BEARING_SIN).copied().unwrap_or(0.0) as f64
        } else {
            obs.get(crate::contracts::OBS_PICKUP_BEARING_SIN).copied().unwrap_or(0.0) as f64
        };
        let bearing_sin = if bearing_sin.is_finite() { bearing_sin } else { 0.0 };
        let roll = (self.cfg.bank_gain * bearing_sin).clamp(-0.6, 0.6);
        let alt_err = if phase == MissionPhase::Return {
            0.0
        } else {
            obs.get(crate::contracts::OBS_PICKUP_ALTITUDE_ERROR).copied().unwrap_or(0.0) as f64 * 500.0
        };
        let pitch = (self.cfg.pitch_gain * alt_err).clamp(-0.4, 0.4);
        let reference = if phase == MissionPhase::Capture { self.cfg.capture_speed } else { self.cfg.cruise_airspeed };
        let throttle = (0.5 + self.cfg.speed_gain * (reference - airspeed)).clamp(0.0, 1.0);
        let throttle = if phase == MissionPhase::Capture && airspeed < self.cfg.v_stall + 1.0 {
            throttle.max(0.6)
        } else {
            throttle
        };
        vec![roll as f32, pitch as f32, 0.0_f32, throttle as f32]
    }

    fn deterministic_action(&mut self, obs: &[f32], ctx: &PolicyContext) -> Vec<f32> {
        self.act(obs, ctx)
    }

    fn reset(&mut self) {}

    fn card(&self) -> &ExpertCard {
        &self.card
    }
}

pub struct RlPolicy<B: AutodiffBackend<FloatElem = f32>> {
    algorithm: Box<dyn Algorithm<B>>,
    normalizer: RunningNormalizer,
    card: ExpertCard,
}

impl<B> RlPolicy<B>
where
    B: AutodiffBackend<FloatElem = f32>,
{
    pub fn new(
        algorithm: Box<dyn Algorithm<B>>,
        normalizer: RunningNormalizer,
        card: ExpertCard,
    ) -> Self {
        Self { algorithm, normalizer, card }
    }
}

impl<B> Policy for RlPolicy<B>
where
    B: AutodiffBackend<FloatElem = f32>,
{
    fn act(&mut self, obs: &[f32], _ctx: &PolicyContext) -> Vec<f32> {
        let norm = self.normalizer.normalize_f32(obs);
        let (tanh_action, _) = self.algorithm.act(&norm);
        let physical = action_to_avionics(&tanh_action);
        vec![
            physical.roll_cmd as f32,
            physical.pitch_cmd as f32,
            physical.yaw_rate_cmd as f32,
            physical.throttle as f32,
        ]
    }

    fn deterministic_action(&mut self, obs: &[f32], _ctx: &PolicyContext) -> Vec<f32> {
        let norm = self.normalizer.normalize_f32(obs);
        let tanh_action = self.algorithm.deterministic_action(&norm);
        let physical = action_to_avionics(&tanh_action);
        vec![
            physical.roll_cmd as f32,
            physical.pitch_cmd as f32,
            physical.yaw_rate_cmd as f32,
            physical.throttle as f32,
        ]
    }

    fn reset(&mut self) {}

    fn card(&self) -> &ExpertCard {
        &self.card
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::contracts::{MISSION_OBS_DIM, OBS_AIRSPEED_INDICATED, OBS_PICKUP_BEARING_SIN};
    use flight_core::MissionPhase;

    fn obs_with(bearing_sin: f64, airspeed: f64) -> Vec<f32> {
        let mut o = vec![0.0f32; MISSION_OBS_DIM];
        o[OBS_PICKUP_BEARING_SIN] = bearing_sin as f32;
        o[OBS_AIRSPEED_INDICATED] = airspeed as f32;
        o
    }

    #[test]
    fn expert_card_round_trips() {
        let card = ExpertCard {
            id: "heuristic_v1".into(),
            model: ModelKind::Heuristic,
            specialization: Specialization::Full,
            contract: "mission.v1".into(),
            reward_profile: "balanced".into(),
            obs_mask: (0..32).collect(),
            trained_steps: 0,
            seed: 7,
            git_commit: "deadbeef".into(),
            notes: "baseline".into(),
        };
        let json = serde_json::to_string(&card).unwrap();
        let back: ExpertCard = serde_json::from_str(&json).unwrap();
        assert_eq!(card, back);
    }

    #[test]
    fn heuristic_actions_stay_in_bounds() {
        let mut p = HeuristicPolicy::new(55.0, 11.5, 10.0);
        let ctx = PolicyContext { phase: MissionPhase::Outbound, step: 0, captured: false,
            features: None };
        for bearing in [-3.0, -0.5, 0.0, 0.5, 3.0] {
            let a = p.act(&obs_with(bearing, 55.0), &ctx);
            assert_eq!(a.len(), 4);
            assert!(a[0].abs() <= 0.6 + 1e-6);
            assert!(a[1].abs() <= 0.4 + 1e-6);
            assert!(a[2].abs() <= 0.8 + 1e-6);
            assert!((0.0..=1.0).contains(&a[3]));
        }
    }

    #[test]
    fn heuristic_rolls_toward_a_target_on_the_right() {
        let mut p = HeuristicPolicy::new(55.0, 11.5, 10.0);
        let ctx = PolicyContext { phase: MissionPhase::Outbound, step: 0, captured: false,
            features: None };
        let a = p.act(&obs_with(0.5, 55.0), &ctx);
        assert!(a[0] > 0.0);
    }

    #[test]
    fn heuristic_bleeds_throttle_in_capture_when_fast() {
        let mut p = HeuristicPolicy::new(55.0, 11.5, 10.0);
        let fast = PolicyContext { phase: MissionPhase::Capture, step: 0, captured: false,
            features: None };
        let slow = PolicyContext { phase: MissionPhase::Capture, step: 0, captured: false,
            features: None };
        let a_fast = p.act(&obs_with(0.0, 40.0), &fast);
        let a_slow = p.act(&obs_with(0.0, 11.5), &slow);
        assert!(a_fast[3] < a_slow[3]);
    }

    #[test]
    fn heuristic_never_commands_nan_on_zero_airspeed() {
        let mut p = HeuristicPolicy::new(55.0, 11.5, 10.0);
        let ctx = PolicyContext { phase: MissionPhase::Capture, step: 0, captured: false,
            features: None };
        let a = p.act(&obs_with(0.0, 0.0), &ctx);
        assert!(a.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn rl_policy_normalizes_and_returns_a_bounded_action() {
        let device = crate::backend::cpu_device();
        let ncfg = crate::net::NetConfig {
            obs_dim: MISSION_OBS_DIM,
            action_dim: crate::ACTION_DIM,
            hidden: 16,
            log_std_init: 0.0,
        };
        let spec = crate::algo::AlgoSpec::from_id("ppo").unwrap();
        let algo = crate::algo::make_algorithm::<crate::backend::Cpu>(&spec, &ncfg, 1, &device);
        let normalizer = crate::normalize::RunningNormalizer::new(ncfg.obs_dim);
        let card = ExpertCard {
            id: "ppo_test".into(),
            model: ModelKind::Ppo,
            specialization: Specialization::Full,
            contract: crate::contracts::MISSION_CONTRACT.into(),
            reward_profile: "balanced".into(),
            obs_mask: (0..32).collect(),
            trained_steps: 0,
            seed: 1,
            git_commit: String::new(),
            notes: String::new(),
        };
        let mut p = RlPolicy::new(algo, normalizer, card);
        let ctx = PolicyContext { phase: MissionPhase::Outbound, step: 0, captured: false,
            features: None };
        let a = p.act(&vec![0.0f32; MISSION_OBS_DIM], &ctx);
        assert_eq!(a.len(), crate::ACTION_DIM);
        assert!(a.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn legacy_checkpoint_card_falls_back_to_full_specialization() {
        let dir = std::env::temp_dir().join(format!("rl_agent_legacy_card_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(
            dir.join("net.json"),
            r#"{"obs_dim":32,"hidden":128,"action_dim":4,"log_std_init":-0.5}"#,
        )
        .unwrap();
        let card = ExpertCard::from_checkpoint(dir.to_str().unwrap(), crate::algo::ALGO_PPO).unwrap();
        assert_eq!(card.model, ModelKind::Ppo);
        assert_eq!(card.specialization, Specialization::Full);
        assert_eq!(card.contract, crate::contracts::MISSION_CONTRACT);
        assert_eq!(card.obs_mask.len(), MISSION_OBS_DIM);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn legacy_checkpoint_card_reports_its_real_contract_and_mask() {
        let dir = std::env::temp_dir().join(format!("raptor-card-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(
            dir.join("net.json"),
            r#"{"obs_dim":19,"hidden":128,"action_dim":4,"log_std_init":-0.5}"#,
        )
        .unwrap();
        let card = ExpertCard::from_checkpoint(dir.to_str().unwrap(), "d3qn").unwrap();
        assert_eq!(card.contract, crate::contracts::AVIONICS_CONTRACT);
        assert_eq!(card.obs_mask, (0..19).collect::<Vec<_>>());
        assert_eq!(card.model, ModelKind::D3qn);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn malformed_expert_json_is_an_error_not_a_panic() {
        let dir = std::env::temp_dir().join(format!("raptor-card-bad-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(dir.join("expert.json"), "{not json").unwrap();
        assert!(ExpertCard::from_checkpoint(dir.to_str().unwrap(), "ppo").is_err());
        let _ = std::fs::remove_dir_all(&dir);
    }
}
