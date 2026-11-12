//! Policy abstractions for RL agent (Task 10).
//! Defines [`Policy`] trait, [`RlPolicy`], [`HeuristicPolicy`], and metadata.

use crate::contracts::{action_to_avionics, AVIONICS_CONTRACT, MISSION_OBS_DIM};
use crate::{algo::Algorithm, normalize::RunningNormalizer};
use burn::tensor::backend::AutodiffBackend;
use flight_core::{MissionFeatures, MissionPhase};

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
    /// Cruise speed on the outbound and return legs (m/s). The mission asks
    /// for `cruise_airspeed` (55 m/s), which this airframe cannot hold, so
    /// the heuristic cruises at a speed the wing can sustain.
    pub cruise_speed: f64,
    /// The capture speed (m/s): the fastest the aircraft may fly and still
    /// latch the hook.
    pub capture_speed: f64,
    /// The stall speed (m/s).
    pub v_stall: f64,
    /// The horizontal capture radius (m) — `MissionConfig::pickup_radius`.
    pub capture_radius_m: f64,
    /// The vertical tolerance around the pickup altitude (m) —
    /// `MissionConfig::pickup_vertical_tolerance`.
    pub pickup_vertical_tolerance_m: f64,
    /// The maximum bank command (rad). The airframe spirals above ~35 deg of
    /// bank, so the bank stays well inside that.
    pub bank_cap: f64,
    /// The bank command per radian of bearing error.
    pub bank_gain: f64,
    /// The bank command per rad/s of bearing rate. This damps the turn so the
    /// aircraft settles on the target bearing instead of overshooting it.
    pub bank_damp: f64,
    /// The speed flown through the turn-back (m/s). A slow turn has a small
    /// radius, which keeps the return leg short.
    pub turn_speed: f64,
    /// The distance (m) at which the approach speed ramp starts.
    pub approach_distance_m: f64,
    /// The altitude error → the demanded climb rate (m/s per m).
    pub climb_rate_gain: f64,
    /// The maximum climb-rate demand (m/s).
    pub max_climb_rate: f64,
    /// The climb-rate error → the pitch command (rad per m/s).
    pub climb_gain: f64,
    /// The nose-up pitch per m/s of excess airspeed (rad). Raising the nose
    /// bleeds speed on the approach.
    pub speed_pitch_gain: f64,
    /// The airspeed error → the throttle.
    pub throttle_speed_gain: f64,
    /// The base throttle (fraction of the throttle range).
    pub throttle_base: f64,
    /// The vertical clearance kept from the pickup altitude while capturing
    /// (m). The skyhook cable is only 8 m long but it is loaded hard inside
    /// the capture radius, so the aircraft captures clear of it.
    pub capture_altitude_margin: f64,
}

impl Default for HeuristicConfig {
    fn default() -> Self {
        Self {
            cruise_speed: 18.0,
            capture_speed: 11.5,
            v_stall: 10.0,
            capture_radius_m: 40.0,
            pickup_vertical_tolerance_m: 15.0,
            bank_cap: 0.22,
            bank_gain: 0.9,
            bank_damp: 0.9,
            turn_speed: 18.0,
            approach_distance_m: 350.0,
            climb_rate_gain: 0.05,
            max_climb_rate: 2.0,
            climb_gain: 0.04,
            speed_pitch_gain: 0.03,
            throttle_speed_gain: 0.06,
            throttle_base: 0.05,
            capture_altitude_margin: 25.0,
        }
    }
}

/// The pitch command (rad, nose-up positive) that holds level flight at a
/// given speed.
///
/// This is *not* just the trim attitude. The flight controller's outer pitch
/// loop is proportional-only, so it needs a rate command to hold an attitude
/// against the trim elevator. At a steady attitude that leaves a fixed error
/// of `-elevator_trim / 90` rad, so the command that actually produces the
/// level-flight attitude is `trim_attitude - elevator_trim / 90`.
///
/// Both terms are measured with the airframe's trim solver
/// (`Simulator::trim_level_flight`) for `m_sweep_mini.toml`: the wing needs a
/// high angle of attack (and a lot of nose-up elevator) when slow and flies
/// nearly flat when fast.
fn level_flight_pitch_cmd(speed: f64) -> f64 {
    /// (airspeed m/s, nose-up pitch command rad)
    const TABLE: [(f64, f64); 13] = [
        (9.00, 0.7329),
        (10.00, 0.5701),
        (11.00, 0.4491),
        (11.43, 0.4066),
        (12.50, 0.3188),
        (14.00, 0.2281),
        (16.00, 0.1442),
        (19.00, 0.0644),
        (22.00, 0.0093),
        (26.00, -0.0260),
        (30.00, -0.0518),
        (35.00, -0.0723),
        (40.00, -0.0856),
    ];
    let v = speed.clamp(TABLE[0].0, TABLE[TABLE.len() - 1].0);
    for pair in TABLE.windows(2) {
        let (v0, p0) = pair[0];
        let (v1, p1) = pair[1];
        if v <= v1 {
            let t = (v - v0) / (v1 - v0).max(1e-9);
            return p0 + t * (p1 - p0);
        }
    }
    TABLE[TABLE.len() - 1].1
}

/// Wrap an angle into (-pi, pi].
fn wrap_pi(a: f64) -> f64 {
    let two_pi = 2.0 * std::f64::consts::PI;
    let mut w = a % two_pi;
    if w > std::f64::consts::PI {
        w -= two_pi;
    } else if w <= -std::f64::consts::PI {
        w += two_pi;
    }
    w
}

#[derive(Debug, Clone)]
pub struct HeuristicPolicy {
    cfg: HeuristicConfig,
    card: ExpertCard,
    /// The launch altitude. The mission launches at the cruise altitude and
    /// the home leg has to come back to it, so it is the altitude reference
    /// for the whole flight.
    home_altitude: Option<f64>,
    /// The previous bearing error, used to damp the turn.
    prev_bearing: Option<f64>,
    /// The phase the previous bearing error was measured in.
    prev_phase: Option<MissionPhase>,
    /// The previous altitude, used to estimate the climb rate.
    prev_altitude: Option<f64>,
    /// The filtered climb rate (m/s).
    climb_rate: f64,
}

impl HeuristicPolicy {
    pub fn new(cruise_airspeed: f64, capture_speed: f64, v_stall: f64) -> Self {
        let mut cfg = HeuristicConfig::default();
        // The airframe cannot sustain the mission's requested cruise speed, so
        // the heuristic cruises at a speed the wing can hold level.
        cfg.cruise_speed = if cruise_airspeed > 0.0 && cruise_airspeed < 40.0 {
            cruise_airspeed
        } else {
            cfg.cruise_speed
        };
        cfg.capture_speed = capture_speed;
        cfg.v_stall = v_stall;
        Self {
            cfg,
            home_altitude: None,
            prev_bearing: None,
            prev_phase: None,
            prev_altitude: None,
            climb_rate: 0.0,
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

    /// Read one observation channel, mapping non-finite values to zero.
    fn read(obs: &[f32], index: usize) -> f64 {
        let v = obs.get(index).copied().unwrap_or(0.0) as f64;
        if v.is_finite() { v } else { 0.0 }
    }
}

impl Policy for HeuristicPolicy {
    fn act(&mut self, obs: &[f32], ctx: &PolicyContext) -> Vec<f32> {
        let dt = 1.0 / 30.0;
        let airspeed = Self::read(obs, crate::contracts::OBS_AIRSPEED_INDICATED);
        let altitude = Self::read(obs, crate::contracts::OBS_BARO_ALTITUDE);
        let phase = ctx.phase;
        let return_leg = matches!(phase, MissionPhase::Return | MissionPhase::Done);

        // The mission launches at the cruise altitude and the home leg has to
        // come back to it, so the launch altitude is the altitude reference.
        if self.home_altitude.is_none() && altitude > 0.0 {
            self.home_altitude = Some(altitude);
        }
        let home_altitude = self.home_altitude.unwrap_or(altitude);

        // The pickup altitude, recovered from the observation: the mission
        // reports the altitude error against it.
        let pickup_altitude =
            altitude + Self::read(obs, crate::contracts::OBS_PICKUP_ALTITUDE_ERROR) * 500.0;

        // Phase-appropriate steering: chase the pickup outbound, then home.
        let (bearing_sin, bearing_cos) = if return_leg {
            (
                Self::read(obs, crate::contracts::OBS_HOME_BEARING_SIN),
                Self::read(obs, crate::contracts::OBS_HOME_BEARING_COS),
            )
        } else {
            (
                Self::read(obs, crate::contracts::OBS_PICKUP_BEARING_SIN),
                Self::read(obs, crate::contracts::OBS_PICKUP_BEARING_COS),
            )
        };
        let bearing_err = bearing_sin.atan2(bearing_cos);
        let distance_m = if return_leg {
            Self::read(obs, crate::contracts::OBS_HOME_DISTANCE) * 100.0
        } else {
            Self::read(obs, crate::contracts::OBS_PICKUP_DISTANCE) * 2500.0
        };

        // --- Speed target ------------------------------------------------
        // The capture only latches inside the pickup radius at or below the
        // capture speed, so the speed has to be on its way down *before* the
        // pickup. The ramp starts far enough out that the aircraft can bleed
        // the speed in the last few hundred metres.
        let speed_target = match phase {
            MissionPhase::Capture => self.cfg.capture_speed - 0.3,
            MissionPhase::Return => {
                // The return leg has no speed limit, so hold cruise: the
                // turn-back is flown fast enough to stay clear of the stall
                // and the pitch/roll coupling that comes with it.
                self.cfg.cruise_speed
            }
            _ => {
                if distance_m < self.cfg.approach_distance_m {
                    let span = (self.cfg.approach_distance_m - self.cfg.capture_radius_m).max(1.0);
                    let ramp = ((distance_m - self.cfg.capture_radius_m) / span).clamp(0.0, 1.0);
                    self.cfg.capture_speed
                        + (self.cfg.cruise_speed - self.cfg.capture_speed) * ramp
                } else {
                    self.cfg.cruise_speed
                }
            }
        };

        // --- Altitude target ---------------------------------------------
        // The skyhook cable is loaded hard anywhere inside the capture radius
        // at the pickup altitude, so the aircraft captures clear of it. When
        // the home altitude is already clear of the cable the capture happens
        // at the home altitude and the return leg needs no altitude change.
        let capture_altitude = if (home_altitude - pickup_altitude).abs()
            <= self.cfg.pickup_vertical_tolerance_m
        {
            pickup_altitude + self.cfg.capture_altitude_margin
        } else {
            home_altitude
        };
        let altitude_target = if return_leg {
            home_altitude
        } else {
            capture_altitude
        };

        // --- Bank --------------------------------------------------------
        // The bank turns the aircraft toward the target. The bearing rate
        // damps the turn so the aircraft settles on the bearing instead of
        // overshooting and spiralling.
        let mut bank = self.cfg.bank_gain * bearing_err;
        if self.prev_phase != Some(phase) {
            self.prev_bearing = None;
            self.prev_phase = Some(phase);
        }
        if let Some(prev) = self.prev_bearing {
            let rate = wrap_pi(bearing_err - prev) / dt;
            bank += self.cfg.bank_damp * rate;
        }
        self.prev_bearing = Some(bearing_err);
        let bank = bank.clamp(-self.cfg.bank_cap, self.cfg.bank_cap);

        // --- Pitch -------------------------------------------------------
        // The altitude error demands a climb rate; the climb-rate loop turns
        // that into a nose-up attitude around the level-flight trim.
        let climb = match self.prev_altitude {
            Some(prev) => {
                let raw = (altitude - prev) / dt;
                // The altitude only moves centimetres per step, so the raw
                // difference is noisy: filter it before feeding it back.
                self.climb_rate += 0.05 * (raw - self.climb_rate);
                self.climb_rate
            }
            None => {
                self.climb_rate = 0.0;
                0.0
            }
        };
        self.prev_altitude = Some(altitude);

        let alt_error = altitude_target - altitude;
        // The feedforward is the level-flight command for the *target* speed,
        // not the current one. Following the current speed's trim is a
        // positive feedback loop — fast asks for nose-down, which dives and
        // speeds up further — and it made the phugoid diverge. Commanding the
        // target trim instead makes the pitch speed-stabilizing: a fast
        // aircraft sees a nose-up command and bleeds the excess.
        let mut pitch_up = level_flight_pitch_cmd(speed_target);
        // Explicit speed damping, active on every leg: fast → nose up (bleed
        // speed), slow → nose down (recover speed).
        pitch_up += self.cfg.speed_pitch_gain * (airspeed - speed_target).clamp(-8.0, 8.0);
        // Altitude: demand a climb rate from the altitude error, then track it
        // with the pitch. The climb-rate term damps the phugoid.
        let desired_climb = (self.cfg.climb_rate_gain * alt_error)
            .clamp(-self.cfg.max_climb_rate, self.cfg.max_climb_rate);
        pitch_up += self.cfg.climb_gain * (desired_climb - climb);
        let pitch_up = pitch_up.clamp(-0.4, 0.4);

        // --- Throttle ----------------------------------------------------
        // The throttle carries the speed; the pitch carries the altitude.
        let throttle = (self.cfg.throttle_base
            + self.cfg.throttle_speed_gain * (speed_target - airspeed))
            .clamp(0.0, 1.0);

        vec![bank as f32, (-pitch_up) as f32, 0.0, throttle as f32]
    }

    fn deterministic_action(&mut self, obs: &[f32], ctx: &PolicyContext) -> Vec<f32> {
        self.act(obs, ctx)
    }

    fn reset(&mut self) {
        self.home_altitude = None;
        self.prev_bearing = None;
        self.prev_phase = None;
        self.prev_altitude = None;
        self.climb_rate = 0.0;
    }

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
