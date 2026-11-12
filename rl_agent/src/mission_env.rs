use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RewardProfile {
    pub name: String,
    pub progress_outbound: f64,
    pub progress_inbound: f64,
    pub altitude_error: f64,
    pub speed_error: f64,
    pub capture_alignment: f64,
    pub time_penalty: f64,
    pub energy_penalty: f64,
    pub crash_penalty: f64,
    pub capture_bonus: f64,
    pub return_bonus: f64,
}

impl RewardProfile {
    pub fn balanced() -> Self {
        Self {
            name: "balanced".into(),
            progress_outbound: 1.0,
            progress_inbound: 1.0,
            altitude_error: 0.02,
            speed_error: 0.05,
            capture_alignment: 1.0,
            time_penalty: 0.01,
            energy_penalty: 0.05,
            crash_penalty: 200.0,
            capture_bonus: 100.0,
            return_bonus: 200.0,
        }
    }

    pub fn aggressive() -> Self {
        Self {
            name: "aggressive".into(),
            progress_outbound: 1.5,
            progress_inbound: 1.5,
            altitude_error: 0.02,
            speed_error: 0.02,
            capture_alignment: 1.0,
            time_penalty: 0.01,
            energy_penalty: 0.05,
            crash_penalty: 200.0,
            capture_bonus: 100.0,
            return_bonus: 200.0,
        }
    }

    pub fn cautious() -> Self {
        Self {
            name: "cautious".into(),
            progress_outbound: 1.0,
            progress_inbound: 1.0,
            altitude_error: 0.05,
            speed_error: 0.08,
            capture_alignment: 1.0,
            time_penalty: 0.01,
            energy_penalty: 0.05,
            crash_penalty: 200.0,
            capture_bonus: 100.0,
            return_bonus: 200.0,
        }
    }

    pub fn capture_focus() -> Self {
        Self {
            name: "capture_focus".into(),
            progress_outbound: 1.0,
            progress_inbound: 1.0,
            altitude_error: 0.02,
            speed_error: 0.05,
            capture_alignment: 5.0,
            time_penalty: 0.01,
            energy_penalty: 0.05,
            crash_penalty: 200.0,
            capture_bonus: 100.0,
            return_bonus: 200.0,
        }
    }

    pub fn from_name(name: &str) -> Result<Self, String> {
        match name {
            "balanced" => Ok(Self::balanced()),
            "aggressive" => Ok(Self::aggressive()),
            "cautious" => Ok(Self::cautious()),
            "capture_focus" => Ok(Self::capture_focus()),
            _ => Err(format!("unknown reward profile: {}", name)),
        }
    }

    pub fn from_toml_file(path: &str) -> Result<Self, Box<dyn std::error::Error>> {
        let content = std::fs::read_to_string(path)?;
        let profile: Self = toml::from_str(&content)?;
        Ok(profile)
    }

    pub fn reward(&self, s: &flight_core::MissionSignals) -> f64 {
        let mut r = 0.0;
        r += self.progress_outbound * s.progress_outbound;
        r += self.progress_inbound * s.progress_inbound;
        r -= self.altitude_error * s.altitude_error;
        r -= self.speed_error * s.speed_error;
        r -= self.capture_alignment * s.capture_alignment;
        r -= self.time_penalty * s.time_penalty;
        r -= self.energy_penalty * s.energy_penalty;
        if s.crash {
            r -= self.crash_penalty;
        }
        if s.captured {
            r += self.capture_bonus;
        }
        if s.returned {
            r += self.return_bonus;
        }
        r
    }
}

use flight_core::{WeatherConfig, MissionConfig, MissionEnvironment};
use crate::contracts::{MISSION_OBS_DIM, mission_observation, action_to_avionics};
use crate::trainer::{EnvSpec, StepOutcome};

pub struct MissionEnv {
    inner: MissionEnvironment,
    reward: RewardProfile,
    last_obs: [f64; MISSION_OBS_DIM],
}

impl MissionEnv {
    pub fn new(
        config_path: &str,
        mission: MissionConfig,
        reward: RewardProfile,
        seed: u64,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        Self::new_with_weather(config_path, mission, reward, seed, None)
    }

    pub fn new_with_weather(
        config_path: &str,
        mission: MissionConfig,
        reward: RewardProfile,
        seed: u64,
        weather_config: Option<WeatherConfig>,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        let inner = MissionEnvironment::new_with_weather(config_path, mission, seed, weather_config)?;
        Ok(Self {
            inner,
            reward,
            last_obs: [0.0; MISSION_OBS_DIM],
        })
    }

    /// Airframe loaded from `config_path` (used to resolve capture speed and
    /// report the mission summary without re-reading the file).
    pub fn aircraft_config(&self) -> &flight_core::AircraftConfig {
        self.inner.aircraft_config()
    }

    /// Mission configuration this environment was built with.
    pub fn mission_config(&self) -> &flight_core::MissionConfig {
        self.inner.mission_config()
    }

    /// Reset onto a named exam scenario, returning the 32-channel observation.
    pub fn reset_with_scenario_pub(&mut self, s: &flight_core::MissionScenario) -> Vec<f32> {
        let r = self.inner.reset_with_scenario(s);
        let obs = mission_observation(&r.avionics, &r.features);
        self.last_obs = obs;
        obs.iter().map(|&x| x as f32).collect()
    }

    /// Current mission phase, for tests that drive the policy directly.
    pub fn phase_public(&self) -> flight_core::MissionPhase {
        self.inner.phase()
    }

    /// Whether the episode ended by capturing the pickup and returning.
    pub fn terminated_success(&self) -> bool {
        self.inner.termination() == flight_core::MissionTermination::CapturedAndReturned
    }

    /// Step the mission with **physical** control commands.
    ///
    /// The [`EnvSpec`](crate::trainer::EnvSpec) contract is tanh-space (the
    /// learning algorithms emit tanh actions), while
    /// [`Policy`](crate::policy::Policy) implementations return physical
    /// commands. Driving a policy through this environment therefore has to
    /// bypass the tanh mapping, exactly like
    /// [`run_eval_suite`](crate::eval::run_eval_suite) does.
    pub fn step_physical(&mut self, action: &[f32]) -> StepOutcome {
        let out = self.inner.step(flight_core::AvionicsAction {
            roll_cmd: action[0] as f64,
            pitch_cmd: action[1] as f64,
            yaw_rate_cmd: action[2] as f64,
            throttle: (action[3] as f64).clamp(0.0, 1.0),
        });
        let obs = mission_observation(&out.avionics, &out.features);
        self.last_obs = obs;
        let reward = self.reward.reward(&out.signals);
        StepOutcome {
            obs: obs.to_vec(),
            reward,
            terminated: out.terminated,
            truncated: out.truncated,
        }
    }
}

impl EnvSpec for MissionEnv {
    fn obs_dim(&self) -> usize {
        MISSION_OBS_DIM
    }

    fn action_dim(&self) -> usize {
        crate::ACTION_DIM
    }

    fn reset(&mut self) -> Vec<f64> {
        let r = self.inner.reset();
        let obs = mission_observation(&r.avionics, &r.features);
        self.last_obs = obs;
        obs.to_vec()
    }

    fn step(&mut self, action: &[f32]) -> StepOutcome {
        let out = self.inner.step(action_to_avionics(action));
        let obs = mission_observation(&out.avionics, &out.features);
        self.last_obs = obs;
        let reward = self.reward.reward(&out.signals);
        StepOutcome {
            obs: obs.to_vec(),
            reward,
            terminated: out.terminated,
            truncated: out.truncated,
        }
    }

    fn crash_penalty(&self) -> f64 {
        0.0
    }

    fn target_altitude(&self) -> f64 {
        self.inner.mission_config().cruise_altitude
    }

    fn target_airspeed(&self) -> f64 {
        self.inner.mission_config().cruise_airspeed
    }

    fn frame(&self) -> Option<flight_core::FlightFrame> {
        None
    }

    fn ground_altitude(&self) -> f64 {
        self.inner.mission_config().ground_altitude
    }

    fn mission_context(&self) -> Option<flight_core::MissionContext> {
        Some(self.inner.context())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::contracts::MISSION_OBS_DIM;
    use flight_core::MissionSignals;

    fn env() -> MissionEnv {
        MissionEnv::new("../m_sweep_mini.toml", MissionConfig::default(), RewardProfile::balanced(), 7)
            .expect("mission env")
    }

    #[test]
    fn reset_and_step_use_32_channels() {
        let mut e = env();
        assert_eq!(e.obs_dim(), MISSION_OBS_DIM);
        assert_eq!(e.reset().len(), MISSION_OBS_DIM);
        let out = e.step(&[0.0, 0.0, 0.0, 0.0]);
        assert_eq!(out.obs.len(), MISSION_OBS_DIM);
        assert!(out.reward.is_finite());
        assert!(out.obs.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn balanced_profile_pays_progress_and_punishes_crash() {
        let p = RewardProfile::balanced();
        let good = MissionSignals {
            progress_outbound: 2.0,
            time_penalty: 1.0,
            energy_penalty: 0.5,
            ..Default::default()
        };
        let bad = MissionSignals {
            crash: true,
            time_penalty: 1.0,
            ..Default::default()
        };
        assert!(p.reward(&good) > 0.0);
        assert!(p.reward(&bad) < -100.0);
    }

    #[test]
    fn capture_and_return_bonuses_are_paid_once_in_the_signals() {
        let p = RewardProfile::balanced();
        let s = MissionSignals { captured: true, returned: true, ..Default::default() };
        assert_eq!(p.reward(&s), p.capture_bonus + p.return_bonus);
    }

    #[test]
    fn mission_env_does_not_double_charge_crashes() {
        let e = env();
        assert_eq!(e.crash_penalty(), 0.0);
    }

    #[test]
    fn named_profiles_resolve_and_unknown_names_fail() {
        assert_eq!(RewardProfile::from_name("balanced").unwrap().name, "balanced");
        assert!(RewardProfile::from_name("nope").is_err());
    }
}
