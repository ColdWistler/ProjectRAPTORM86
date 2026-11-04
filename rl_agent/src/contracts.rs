pub const MISSION_CONTRACT: &str = "mission.v1";
pub const AVIONICS_CHANNELS: usize = flight_core::AVIONICS_OBS_DIM;
pub const MISSION_FEATURE_OFFSET: usize = AVIONICS_CHANNELS;
pub const MISSION_OBS_DIM: usize = MISSION_FEATURE_OFFSET + flight_core::MISSION_FEATURE_DIM;

pub const OBS_BARO_ALTITUDE: usize = 10;
pub const OBS_AIRSPEED_INDICATED: usize = 11;
pub const OBS_PICKUP_DISTANCE: usize = 19;
pub const OBS_PICKUP_BEARING_SIN: usize = 20;
pub const OBS_PICKUP_BEARING_COS: usize = 21;
pub const OBS_PICKUP_ALTITUDE_ERROR: usize = 22;
pub const OBS_HOME_DISTANCE: usize = 23;
pub const OBS_HOME_BEARING_SIN: usize = 24;
pub const OBS_HOME_BEARING_COS: usize = 25;
pub const OBS_PHASE: usize = 26;
pub const OBS_SPEED_MARGIN: usize = 27;
pub const OBS_TIME_REMAINING: usize = 28;
pub const OBS_CAPTURE_DWELL: usize = 29;
pub const OBS_HORIZONTAL_OFFSET: usize = 30;
pub const OBS_VERTICAL_OFFSET: usize = 31;

pub fn mission_observation(
    avionics: &[f64; AVIONICS_CHANNELS],
    features: &flight_core::MissionFeatures,
) -> [f64; MISSION_OBS_DIM] {
    let mut out = [0.0; MISSION_OBS_DIM];
    out[..AVIONICS_CHANNELS].copy_from_slice(avionics);
    out[MISSION_FEATURE_OFFSET..].copy_from_slice(&features.to_array());
    out
}

pub fn mission_observation_vec(
    avionics: &[f64; AVIONICS_CHANNELS],
    features: &flight_core::MissionFeatures,
) -> Vec<f64> {
    mission_observation(avionics, features).to_vec()
}

/// Map a tanh-space action to the avionics attitude command.
/// Values are widened to f64 *before* tanh so the comparison-tolerance
/// matches the documented scalar formula `scale * tanh(a)` directly.
pub fn action_to_avionics(action: &[f32]) -> flight_core::AvionicsAction {
    flight_core::AvionicsAction {
        roll_cmd: (action[0] as f64).tanh() * 0.6,
        pitch_cmd: (action[1] as f64).tanh() * 0.4,
        yaw_rate_cmd: (action[2] as f64).tanh() * 0.8,
        throttle: 0.5 * ((action[3] as f64).tanh() + 1.0),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use flight_core::{MissionFeatures, MISSION_FEATURE_DIM};

    #[test]
    fn contract_dimensions_add_up() {
        assert_eq!(AVIONICS_CHANNELS, 19);
        assert_eq!(MISSION_FEATURE_OFFSET, 19);
        assert_eq!(MISSION_FEATURE_DIM, 13);
        assert_eq!(MISSION_OBS_DIM, 32);
    }

    #[test]
    fn base_channels_are_preserved() {
        let avionics: [f64; AVIONICS_CHANNELS] = std::array::from_fn(|i| i as f64);
        let obs = mission_observation(&avionics, &MissionFeatures::default());
        for (i, expected) in avionics.iter().enumerate() {
            assert_eq!(obs[i], *expected);
        }
    }

    #[test]
    fn mission_features_fill_the_tail() {
        let features = MissionFeatures {
            pickup_distance_norm: 0.4,
            pickup_bearing_sin: 0.5,
            pickup_bearing_cos: 0.5,
            pickup_altitude_error_norm: -0.2,
            home_distance_norm: 0.1,
            home_bearing_sin: -0.5,
            home_bearing_cos: -0.5,
            phase: 1.0,
            speed_margin_norm: 0.5,
            time_remaining_fraction: 0.75,
            capture_dwell_fraction: 0.5,
            horizontal_offset_norm: 1.0,
            vertical_offset_norm: 0.0,
        };
        let obs = mission_observation(&[0.0; AVIONICS_CHANNELS], &features);
        assert_eq!(obs[OBS_PICKUP_DISTANCE], 0.4);
        assert_eq!(obs[OBS_PICKUP_BEARING_SIN], 0.5);
        assert_eq!(obs[OBS_PHASE], 1.0);
        assert_eq!(obs[OBS_VERTICAL_OFFSET], 0.0);
    }

    #[test]
    fn action_mapping_uses_the_documented_scales() {
        let a = action_to_avionics(&[0.0, 0.0, 0.0, 0.0]);
        assert!(a.roll_cmd.abs() < 1e-12);
        assert!((a.throttle - 0.5).abs() < 1e-12);
        let b = action_to_avionics(&[1.0, 1.0, 1.0, 1.0]);
        assert!((b.roll_cmd - 0.6 * 1.0f64.tanh()).abs() < 1e-12);
        assert!((b.pitch_cmd - 0.4 * 1.0f64.tanh()).abs() < 1e-12);
        assert!((b.yaw_rate_cmd - 0.8 * 1.0f64.tanh()).abs() < 1e-12);
        assert!((b.throttle - 0.5 * (1.0f64.tanh() + 1.0)).abs() < 1e-12);
    }
}
