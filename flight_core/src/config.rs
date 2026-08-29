//! Aircraft / airframe configuration with TOML deserialization.

use serde::Deserialize;
use std::fs;

fn default_oswald_e() -> f64 { 0.80 }
fn default_alpha_stall_pos() -> f64 { 16.0_f64.to_radians() } // ~0.279 rad
fn default_alpha_stall_neg() -> f64 { (-12.0_f64).to_radians() } // ~-0.209 rad
fn default_cd_max() -> f64 { 1.95 }
fn default_mach_crit() -> f64 { 0.65 }
fn default_cm_adot() -> f64 { -3.0 }
fn default_thrust_arm() -> f64 { 0.0 }
fn default_power_max() -> f64 { 119_000.0 }

fn default_cy_beta() -> f64 { -0.31 }
fn default_cy_dr() -> f64 { -0.15 }  // Rudder side force (positive rudder yaws right, tail pushed left)

fn default_cl_beta() -> f64 { -0.09 } // Dihedral effect
fn default_cl_p() -> f64 { -0.45 }    // Roll damping
fn default_cl_r() -> f64 { 0.10 }     // Roll due to yaw rate
fn default_cl_da() -> f64 { 0.07 }    // Aileron roll authority
fn default_cl_dr() -> f64 { 0.01 }

fn default_cn_beta() -> f64 { 0.12 }  // Directional weathercock stability
fn default_cn_p() -> f64 { -0.02 }    // Yaw due to roll rate (adverse yaw)
fn default_cn_r() -> f64 { -0.16 }    // Yaw damping
fn default_cn_da() -> f64 { -0.004 }  // Aileron-to-yaw coupling (near-neutral)
fn default_cn_dr() -> f64 { 0.05 }    // Rudder yaw authority (positive = yaw right)

/// A fully-parameterized description of the aircraft that the flight
/// dynamics and aerodynamic models depend on.
///
/// All physical quantities use SI units throughout.
#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "snake_case")]
pub struct AircraftConfig {
    /// Total aircraft mass in kilograms.
    pub mass: f64,
    /// Wing planform area in square meters (m²).
    pub wing_area: f64,
    /// Wing span in meters (m).
    pub wing_span: f64,
    /// Mean aerodynamic chord in meters (m).
    pub chord: f64,

    /// Moment of inertia about the body roll (X) axis, kg·m².
    pub ixx: f64,
    /// Moment of inertia about the body pitch (Y) axis, kg·m².
    pub iyy: f64,
    /// Moment of inertia about the body yaw (Z) axis, kg·m².
    pub izz: f64,

    /// Lift coefficient at zero angle-of-attack.
    pub cl0: f64,
    /// Lift curve slope in per radian (dCL/dα).
    pub cla: f64,
    /// Parasite (zero-lift) drag coefficient.
    pub cd0: f64,
    /// Induced drag factor (k in CD = cd0 + k·CL²).
    pub k_drag: f64,

    /// Pitching moment coefficient at zero angle-of-attack.
    pub cm0: f64,
    /// Pitching moment stability derivative per radian (dCm/dα).
    pub cma: f64,
    /// Pitch damping derivative per rad/s (dCm/d(q·c/2V)).
    pub cmq: f64,
    /// Elevator effectiveness per radian (dCm/dδ_e).
    pub cme: f64,

    /// Maximum engine thrust in Newtons (N).
    pub thrust_max: f64,

    /// Maximum engine shaft power in Watts (W). A fixed-pitch propeller
    /// delivers roughly constant power, so the available thrust falls off as
    /// `P_max / V` once airspeed rises above the corner speed; this caps the
    /// airspeed and damps the phugoid.
    #[serde(default = "default_power_max")]
    pub power_max: f64,

    // --- JSBSim-grade High-AoA & Nonlinear Aero Extensions ---
    /// Oswald wing efficiency span factor e (0.75 - 0.85).
    #[serde(default = "default_oswald_e")]
    pub oswald_e: f64,
    /// Positive stall angle of attack in radians.
    #[serde(default = "default_alpha_stall_pos")]
    pub alpha_stall_pos: f64,
    /// Negative stall angle of attack in radians.
    #[serde(default = "default_alpha_stall_neg")]
    pub alpha_stall_neg: f64,
    /// Peak post-stall flat-plate drag coefficient Cd_max.
    #[serde(default = "default_cd_max")]
    pub cd_max: f64,
    /// Critical drag-divergence Mach number.
    #[serde(default = "default_mach_crit")]
    pub mach_crit: f64,
    /// Pitch damping due to (alpha-dot) / downwash lag
    /// (dCm/d(alpha_dot * c / 2V)). Negative = stabilizing.
    #[serde(default = "default_cm_adot")]
    pub cm_adot: f64,
    /// Vertical offset of the thrust line from the CG along the body Z
    /// axis (positive = thrust line below CG, so throttle pulls the nose up).
    #[serde(default = "default_thrust_arm")]
    pub thrust_arm: f64,

    // --- Lateral-directional aerodynamic stability derivatives ---
    #[serde(default = "default_cy_beta")]
    pub cy_beta: f64,
    #[serde(default = "default_cy_dr")]
    pub cy_dr: f64,

    #[serde(default = "default_cl_beta")]
    pub cl_beta: f64,
    #[serde(default = "default_cl_p")]
    pub cl_p: f64,
    #[serde(default = "default_cl_r")]
    pub cl_r: f64,
    #[serde(default = "default_cl_da")]
    pub cl_da: f64,
    #[serde(default = "default_cl_dr")]
    pub cl_dr: f64,

    #[serde(default = "default_cn_beta")]
    pub cn_beta: f64,
    #[serde(default = "default_cn_p")]
    pub cn_p: f64,
    #[serde(default = "default_cn_r")]
    pub cn_r: f64,
    #[serde(default = "default_cn_da")]
    pub cn_da: f64,
    #[serde(default = "default_cn_dr")]
    pub cn_dr: f64,
}

impl AircraftConfig {
    /// Wing aspect ratio AR = b² / S.
    pub fn aspect_ratio(&self) -> f64 {
        (self.wing_span * self.wing_span) / self.wing_area.max(1e-3)
    }

    /// Theoretical induced drag factor k = 1 / (pi * AR * e).
    pub fn induced_drag_k(&self) -> f64 {
        if self.k_drag > 0.0 {
            self.k_drag
        } else {
            1.0 / (std::f64::consts::PI * self.aspect_ratio() * self.oswald_e)
        }
    }

    /// Load an `AircraftConfig` from a TOML file on disk.
    pub fn from_file(path: &str) -> Result<Self, Box<dyn std::error::Error>> {
        let contents = fs::read_to_string(path)?;
        let config: AircraftConfig = toml::from_str(&contents)?;
        Ok(config)
    }
}

/// Convenience helper that loads a config from `path` and initializes a
/// fresh flight state.
pub fn load_config(path: &str) -> AircraftConfig {
    AircraftConfig::from_file(path)
        .unwrap_or_else(|e| panic!("failed to load aircraft config from `{}`: {}", path, e))
}
