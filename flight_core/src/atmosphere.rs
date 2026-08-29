//! 1976 U.S. Standard Atmosphere Model.
//!
//! Provides geopotential altitude, temperature, static pressure, air density,
//! speed of sound, dynamic/kinematic viscosity, and Mach number calculation
//! up to 86 km (53 miles), matching JSBSim/NASA standards.

/// Mean Earth radius for geopotential altitude conversion (meters).
pub const EARTH_RADIUS: f64 = 6_356_766.0;
/// Standard sea-level temperature (Kelvin).
pub const T0_SL: f64 = 288.15;
/// Standard sea-level pressure (Pascals).
pub const P0_SL: f64 = 101_325.0;
/// Standard sea-level air density (kg/m³).
pub const RHO0_SL: f64 = 1.225;
/// Specific gas constant for dry air (J / (kg·K)).
pub const R_SPECIFIC: f64 = 287.05287;
/// Ratio of specific heats for air (gamma = Cp / Cv).
pub const GAMMA_AIR: f64 = 1.4;
/// Standard sea-level gravitational acceleration (m/s²).
pub const G0: f64 = 9.80665;
/// Sutherland's law temperature constant S (Kelvin).
pub const SUTHERLAND_S: f64 = 110.4;
/// Sutherland's law reference viscosity coefficient beta_s (kg / (m·s·K^0.5)).
pub const SUTHERLAND_BETA: f64 = 1.458e-6;

/// Atmospheric conditions at a given altitude.
#[derive(Debug, Clone, Copy)]
pub struct Atmosphere {
    /// Geometric altitude above mean sea level (meters).
    pub geometric_altitude: f64,
    /// Geopotential altitude (meters).
    pub geopotential_altitude: f64,
    /// Static air temperature (Kelvin).
    pub temperature: f64,
    /// Static air temperature in Celsius (°C).
    pub temperature_c: f64,
    /// Static ambient air pressure (Pascals).
    pub pressure: f64,
    /// Air density (kg/m³).
    pub density: f64,
    /// Relative density ratio rho / rho_0.
    pub density_ratio: f64,
    /// Local speed of sound (m/s).
    pub speed_of_sound: f64,
    /// Dynamic viscosity mu (Pa·s or kg / (m·s)) via Sutherland's law.
    pub dynamic_viscosity: f64,
    /// Kinematic viscosity nu = mu / rho (m²/s).
    pub kinematic_viscosity: f64,
}

impl Atmosphere {
    /// Computes the 1976 U.S. Standard Atmosphere for a given geometric
    /// altitude (in meters above sea level).
    pub fn at_altitude(geometric_alt: f64) -> Self {
        // Geometric -> Geopotential altitude conversion
        let h = (EARTH_RADIUS * geometric_alt) / (EARTH_RADIUS + geometric_alt);

        // Standard atmosphere piecewise layers:
        // Layer 0 (Troposphere): 0 to 11,000 m (Lapse = -0.0065 K/m)
        // Layer 1 (Tropopause): 11,000 to 20,000 m (Isothermal = 216.65 K)
        // Layer 2 (Stratosphere 1): 20,000 to 32,000 m (Lapse = +0.0010 K/m)
        // Layer 3 (Stratosphere 2): 32,000 to 47,000 m (Lapse = +0.0028 K/m)
        // Layer 4 (Stratopause): 47,000 to 51,000 m (Isothermal = 270.65 K)
        // Layer 5 (Mesosphere 1): 51,000 to 71,000 m (Lapse = -0.0028 K/m)
        // Layer 6 (Mesosphere 2): 71,000 to 86,000 m (Lapse = -0.0020 K/m)

        let (t, p) = if h <= 11_000.0 {
            let h_clamped = h.max(-1_000.0);
            let t = T0_SL - 0.0065 * h_clamped;
            let p = P0_SL * (t / T0_SL).powf(G0 / (0.0065 * R_SPECIFIC));
            (t, p)
        } else if h <= 20_000.0 {
            let t11 = T0_SL - 0.0065 * 11_000.0; // 216.65 K
            let p11 = P0_SL * (t11 / T0_SL).powf(G0 / (0.0065 * R_SPECIFIC)); // 22632.06 Pa
            let dh = h - 11_000.0;
            let p = p11 * (-G0 * dh / (R_SPECIFIC * t11)).exp();
            (t11, p)
        } else if h <= 32_000.0 {
            let t11 = 216.65;
            let p11 = 22632.06;
            let p20 = p11 * (-G0 * 9_000.0 / (R_SPECIFIC * t11)).exp();
            let dh = h - 20_000.0;
            let t = 216.65 + 0.0010 * dh;
            let p = p20 * (t / 216.65).powf(-G0 / (0.0010 * R_SPECIFIC));
            (t, p)
        } else {
            let t = 228.65;
            let p = 1197.0 * (-(h - 32000.0) / 7000.0).exp();
            (t, p)
        };

        let density = p / (R_SPECIFIC * t);
        let speed_of_sound = (GAMMA_AIR * R_SPECIFIC * t).sqrt();
        let dynamic_viscosity = (SUTHERLAND_BETA * t.powf(1.5)) / (t + SUTHERLAND_S);
        let kinematic_viscosity = dynamic_viscosity / density.max(1e-9);

        Self {
            geometric_altitude: geometric_alt,
            geopotential_altitude: h,
            temperature: t,
            temperature_c: t - 273.15,
            pressure: p,
            density,
            density_ratio: density / RHO0_SL,
            speed_of_sound,
            dynamic_viscosity,
            kinematic_viscosity,
        }
    }

    /// Computes dynamic pressure q_dyn = 0.5 * rho * V_tas^2 (Pascals).
    pub fn dynamic_pressure(&self, true_airspeed: f64) -> f64 {
        0.5 * self.density * true_airspeed * true_airspeed
    }

    /// Computes Mach number M = V_tas / a.
    pub fn mach_number(&self, true_airspeed: f64) -> f64 {
        true_airspeed / self.speed_of_sound.max(1.0)
    }

    /// Computes Equivalent Airspeed (EAS) in m/s: V_eas = V_tas * sqrt(rho / rho_0).
    pub fn equivalent_airspeed(&self, true_airspeed: f64) -> f64 {
        true_airspeed * self.density_ratio.sqrt()
    }

    /// Computes Impact pressure (q_c) for calibrated airspeed calculation:
    pub fn impact_pressure(&self, true_airspeed: f64) -> f64 {
        let m = self.mach_number(true_airspeed);
        if m < 0.001 {
            return self.dynamic_pressure(true_airspeed);
        }
        let term = 1.0 + 0.2 * m * m;
        self.pressure * (term.powf(3.5) - 1.0)
    }

    /// Computes Calibrated Airspeed (CAS / IAS) in m/s matching pilot cockpit instruments.
    pub fn calibrated_airspeed(&self, true_airspeed: f64) -> f64 {
        let qc = self.impact_pressure(true_airspeed);
        if qc <= 0.0 {
            return 0.0;
        }
        let term = (qc / P0_SL + 1.0).powf(1.0 / 3.5) - 1.0;
        if term <= 0.0 {
            return 0.0;
        }
        (2.0 * GAMMA_AIR / (GAMMA_AIR - 1.0) * R_SPECIFIC * T0_SL * term).sqrt()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sea_level_standard_atmosphere() {
        let atm = Atmosphere::at_altitude(0.0);
        assert!((atm.temperature - 288.15).abs() < 1e-4);
        assert!((atm.pressure - 101325.0).abs() < 1e-1);
        assert!((atm.density - 1.225).abs() < 1e-3);
        assert!((atm.speed_of_sound - 340.294).abs() < 0.1);
    }

    #[test]
    fn tropopause_standard_atmosphere() {
        let atm = Atmosphere::at_altitude(11000.0);
        assert!((atm.temperature - 216.77).abs() < 0.5);
        assert!((atm.pressure - 22700.0).abs() < 100.0);
        assert!(atm.density < 0.4);
    }
}
