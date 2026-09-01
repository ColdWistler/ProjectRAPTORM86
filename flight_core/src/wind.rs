//! Atmospheric wind field for the flight dynamics engine.
//!
//! Models the air the aircraft flies through as the superposition of a
//! **steady wind** (with optional **wind shear** as a function of altitude),
//! and a stochastic **turbulence/gust** component using a Dryden-style
//! time-correlated model.
//!
//! Wind is expressed in the Earth-fixed **NED** frame
//! `(north, east, down)` in metres per second. The flight dynamics uses the
//! air-relative velocity `V_air = V_ground - W` to compute true airspeed,
//! angle of attack, sideslip, dynamic pressure and Mach, while the trajectory
//! is still integrated from the ground velocity.

use crate::state::AircraftState;
use nalgebra::Vector3;

/// Turbulence intensity settings. Roughly follows the light/moderate/severe
/// qualitative categories used in PILOT FRIENDLY pilot handbooks / MIL-STD
/// gust magnitude bands at typical UAV cruise altitudes.
#[derive(Debug, Clone, Copy, PartialEq)]
#[allow(clippy::derivable_impls)]
pub enum TurbulenceIntensity {
    /// `sigma_u = 1.5 m/s` — negligible bumps, roughly clear-air.
    Light,
    /// `sigma_u = 3.0 m/s` — occasional gusts, airframe stays comfortable.
    Moderate,
    /// `sigma_u = 5.5 m/s` — sustained rough air, noticeable control load.
    Severe,
    /// Use a caller-supplied RMS gust velocity.
    Custom(f64),
}

impl TurbulenceIntensity {
    /// Gust RMS magnitude `sigma_u` of the longitudinal turbulent velocity
    /// component (m/s). Per Dryden models, the lateral/vertical components
    /// scale from this.
    pub fn sigma_u(self) -> f64 {
        match self {
            TurbulenceIntensity::Light => 1.5,
            TurbulenceIntensity::Moderate => 3.0,
            TurbulenceIntensity::Severe => 5.5,
            TurbulenceIntensity::Custom(s) => s.abs(),
        }
    }
}

/// Configuration for the [`WindEnvironment`].
#[derive(Debug, Clone)]
pub struct WindConfig {
    /// Magnitude of the steady wind at the reference altitude (m/s).
    pub wind_speed: f64,
    /// True heading (radians, north = 0, clockwise) that the wind **blows
    /// toward**. A wind blowing from the north toward the south is heading PI.
    pub wind_direction: f64,
    /// Reference altitude (m) at which `wind_speed` applies.
    pub reference_altitude: f64,
    /// Enable logarithmic wind shear: wind grows with altitude below the
    /// reference altitude (surface boundary layer) and stays roughly constant
    /// above it. If false, the steady wind is uniform.
    pub wind_shear: bool,
    /// Turbulence intensity model.
    pub turbulence: TurbulenceIntensity,
    /// Turbulence integral scale (wavelength) in metres. Governs how quickly
    /// gust velocity decorrelates with distance travelled through the air.
    pub turbulence_scale: f64,
    /// Seed for the reproducible turbulence PRNG.
    pub seed: u64,
}

impl Default for WindConfig {
    fn default() -> Self {
        Self {
            wind_speed: 0.0,
            wind_direction: 0.0,
            reference_altitude: 1000.0,
            wind_shear: false,
            turbulence: TurbulenceIntensity::Light,
            turbulence_scale: 533.0,
            seed: 0x9E37_79B9_7F4A_7C15,
        }
    }
}

/// A self-contained deterministic PRNG (xoshiro-style mixing) so turbulence
/// is reproducible for a given seed — important for reinforcement-learning
/// training loops.
#[derive(Debug, Clone)]
struct Prng {
    state: u64,
}

impl Prng {
    fn new(seed: u64) -> Self {
        Self { state: seed.max(1) }
    }

    /// Return a uniform random `f64` in `[0, 1)`.
    fn next_f64(&mut self) -> f64 {
        // xorshift64*
        let mut x = self.state;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.state = x;
        // 0x2545_F491_4F6C_DD1D
        let w = x.wrapping_mul(0x2545_F491_4F6C_DD1D);
        (w >> 11) as f64 / (1u64 << 53) as f64
    }

    /// Standard normal via Box–Muller.
    fn next_unit_gaussian(&mut self) -> f64 {
        let u1 = (self.next_f64() + 1e-12).min(0.999_999_999);
        let u2 = self.next_f64().max(1e-12);
        (-2.0 * u1.ln()).sqrt() * (std::f64::consts::TAU * u2).cos()
    }
}

/// Time-correlated gust state for one turbulence axis.
#[derive(Debug, Clone)]
struct GustState {
    velocity: f64,
    prng: Prng,
}

impl GustState {
    /// Advance the Dryden-style gust with a first-order (low-pass) spectral
    /// shaping of white noise. The time constant is derived from the integral
    /// length scale `L` and the true airspeed `V_air` such that the gust
    /// spatial wavelength decorrelation matches the Dryden spectrum.
    fn step(&mut self, vt_air: f64, sigma: f64, l: f64, dt: f64) -> f64 {
        // Dryden time constant for the u-component: tau = L / V.
        // For a target integral scale and airspeed, low-pass with tau = L / V
        // produces the characteristic "long-wavelength" gust build.
        let v = vt_air.abs().max(15.0); // guard near hover/stall
        let tau = (l / v).max(0.05);
        // Discrete first-order low-pass. For an AR(1) filter
        //   x[n] = a*white + (1-a)*x[n-1]
        // the steady-state variance is a²/(2a−a²) * Var(white). Scaling the
        // white-noise input so the output converges to std `sigma`:
        let alpha = dt / (tau + dt);
        let white_std = sigma * ((2.0 - alpha) / alpha.max(1e-9)).sqrt();
        let white = self.prng.next_unit_gaussian() * white_std;
        self.velocity += alpha * (white - self.velocity);
        self.velocity
    }
}

/// The atmospheric wind field evaluated at a point in space and time.
pub struct WindEnvironment {
    pub config: WindConfig,
    gust_u: GustState,
    gust_v: GustState,
    gust_w: GustState,
}

impl WindEnvironment {
    /// Construct with the given configuration.
    pub fn new(config: WindConfig) -> Self {
        let base = config.seed;
        Self {
            config,
            gust_u: GustState {
                velocity: 0.0,
                prng: Prng::new(base.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1),
            },
            gust_v: GustState {
                velocity: 0.0,
                prng: Prng::new(base.wrapping_mul(0xD1B5_4A32_D192_ED03) | 1),
            },
            gust_w: GustState {
                velocity: 0.0,
                prng: Prng::new(base.wrapping_mul(0x243F_6A88_85A3_08D3) | 1),
            },
        }
    }

    /// Steady (non-turbulent) wind in the Earth NED frame at the specified
    /// geometric altitude, including logarithmic shear if enabled.
    pub fn steady_wind(&self, altitude: f64) -> Vector3<f64> {
        let mut mag = self.config.wind_speed;

        // Logarithmic boundary-layer shear: wind grows from ~0 at the surface
        // up to the reference wind at the reference altitude.
        if self.config.wind_shear && self.config.wind_speed > 0.0 {
            let alt = altitude.max(1.0);
            let z_ref = self.config.reference_altitude.max(1.0);
            if alt < z_ref {
                // Below reference: scale down as ~sqrt or log; use a physically
                // credible power-law boundary layer profile.
                mag *= (alt / z_ref).powf(0.2);
            }
            // Above reference: keep the reference wind.
        }

        // Wind direction is the true heading the wind blows *toward*.
        let dir = self.config.wind_direction;
        let north = mag * dir.cos();
        let east = mag * dir.sin();
        // No steady vertical (down) component for a clean steady geostrophic wind.
        Vector3::new(north, east, 0.0)
    }

    /// Turbulent gust component in the Earth NED frame at the current true
    /// airspeed, advanced by `dt` seconds.
    pub fn turbulence(&mut self, vt_air: f64, dt: f64) -> Vector3<f64> {
        let sigma_u = self.config.turbulence.sigma_u();
        let l = self.config.turbulence_scale.max(10.0);
        // Dryden scaling of the lateral/vertical RMS from the longitudinal one.
        let sigma_v = sigma_u;
        let sigma_w = sigma_u.mul_add(0.7, 0.0);
        let u_g = self.gust_u.step(vt_air, sigma_u, l, dt);
        let v_g = self.gust_v.step(vt_air, sigma_v, l, dt);
        let w_g = self.gust_w.step(vt_air, sigma_w, l * 0.5, dt);
        // Wind is expressed in Earth NED; the gust axes were generated in the
        // NED basis (an approximation of the true body-aligned Dryden frame,
        // which for a near-level cruise is a small-angle difference).
        Vector3::new(u_g, v_g, w_g)
    }

    /// Total wind (steady + turbulence) in the Earth NED frame.
    pub fn total_wind(&mut self, state: &AircraftState, vt_air: f64, dt: f64) -> Vector3<f64> {
        let steady = self.steady_wind(state.altitude());
        let turb = self.turbulence(vt_air, dt);
        steady + turb
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::AircraftState;

    #[test]
    fn steady_wind_direction_is_bowed_toward_given_bearing() {
        // Wind blowing toward due east => north component 0, east = speed.
        let mut cfg = WindConfig::default();
        cfg.wind_speed = 10.0;
        cfg.wind_direction = std::f64::consts::FRAC_PI_2; // east (pi/2)
        cfg.wind_shear = false;
        let w = WindEnvironment::new(cfg);
        let v = w.steady_wind(500.0);
        assert!((v.x).abs() < 1e-9, "north should be ~0, got {}", v.x);
        assert!((v.y - 10.0).abs() < 1e-9, "east should be 10, got {}", v.y);
    }

    #[test]
    fn shear_reduces_wind_near_ground() {
        let mut cfg = WindConfig::default();
        cfg.wind_speed = 20.0;
        cfg.reference_altitude = 1000.0;
        cfg.wind_shear = true;
        cfg.wind_direction = 0.0; // blows north
        let w = WindEnvironment::new(cfg);
        let low = w.steady_wind(50.0);
        let high = w.steady_wind(2000.0);
        assert!(
            low.x < high.x,
            "shear should reduce wind near the ground: low {} vs high {}",
            low.x,
            high.x
        );
        assert!((high.x - 20.0).abs() < 1e-9);
    }

    #[test]
    fn turbulence_is_reproducible_and_bounded() {
        let mut cfg = WindConfig::default();
        cfg.turbulence = TurbulenceIntensity::Severe; // sigma_u = 5.5
        let mut a = WindEnvironment::new(cfg.clone());
        let mut b = WindEnvironment::new(cfg.clone());

        // Identical seeds => identical gust sequences (reproducibility).
        let mut max_a = 0.0f64;
        let mut sum_sq = 0.0f64;
        let n = 20_000;
        for _ in 0..n {
            let wa = a.turbulence(60.0, 1.0 / 30.0);
            let wb = b.turbulence(60.0, 1.0 / 30.0);
            assert!((wa - wb).norm() < 1e-9, "turbulence should be reproducible");
            max_a = max_a.max(wa.norm());
            sum_sq += wa.x * wa.x;
        }
        // The longitudinal gust RMS should converge to sigma_u (5.5 m/s), not
        // vanish — guards against a degenerate low-pass silent near-zero bug.
        // The estimate carries sampling noise from temporal gust correlation,
        // so allow a generous tolerance.
        let rms_u = (sum_sq / n as f64).sqrt();
        assert!(
            (rms_u - 5.5).abs() < 2.0,
            "u-gust RMS should be ~sigma_u, got {:.2} m/s",
            rms_u
        );
        // Bound: over many samples a combined 3-axis gust (RMS ~8.7 m/s for
        // severe) should essentially never exceed ~4 RMS (~35 m/s).
        assert!(
            max_a < 50.0,
            "severe turbulence should stay bounded, got max {:.1} m/s",
            max_a
        );
    }

    #[test]
    fn wind_changes_ground_speed_not_air_speed() {
        // A tailwind should raise the trimmed ground speed for the same
        // aerodynamic state, i.e. true airspeed is unaffected by translation.
        let state = AircraftState::default();
        // Still air: airspeed == groundspeed.
        let air0 = state.true_airspeed(&Vector3::zeros());
        assert!((air0 - state.airspeed()).abs() < 1e-9);

        // 20 m/s blowing north, aircraft heading north.
        let wind = Vector3::new(20.0, 0.0, 0.0);
        // Heading north means body x-axis aligns with north (Earth x).
        // If the aircraft pointed north, ground-north velocity u equals
        // airspeed + tailwind. The air-relative north velocity must shrink.
        let v_air = state.air_velocity(&wind);
        assert!(
            v_air.x < state.u,
            "tailwind should reduce air-relative north velocity"
        );
    }
}
