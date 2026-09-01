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

fn default_engine_count() -> u32 { 1 }
fn default_engine_lateral_arm() -> f64 { 0.0 }
fn default_prop_torque_coeff() -> f64 { 0.0 }
fn default_p_factor_coeff() -> f64 { 0.0 }
fn default_gyro_coeff() -> f64 { 0.0 }
fn default_throttle_split() -> f64 { 0.0 }

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

// Plain flap increments (per radian of trailing-edge deflection). Values are
// tuned for a light/tactical UAV such that full flap (~30 deg) adds roughly
// Delta_CL ~ 0.7, Delta_CD ~ 0.08, and a nose-down Delta_Cm ~ -0.25.
fn default_cl_flap() -> f64 { 1.10 }
fn default_cd_flap() -> f64 { 0.14 }
fn default_cm_flap() -> f64 { -0.20 }
fn default_flap_stall_shift() -> f64 { 6.0_f64.to_radians() }
// Nose-down spiral-divergence authority at steep bank. Small: it only needs
// to overcome the residual nose-up from level-trim coupling (~0.4 kN*m), not
// dominate normal flight.
fn default_spiral_nose_drop_cm() -> f64 { 0.10 }

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

    // --- Multi-engine (twin) propeller model --------------------------------
    /// Number of propulsion units. 1 models a single centreline thrust line;
    /// 2 splits the total thrust into left/right engines separated by
    /// [`engine_lateral_arm`](Self::engine_lateral_arm) either side of the CG,
    /// unlocking asymmetric-thrust (engine-out), P-factor, prop-torque and
    /// gyroscopic precession behaviour.
    #[serde(default = "default_engine_count")]
    pub engine_count: u32,
    /// Lateral distance (m) of each engine thrust line from the CG along the
    /// body Y axis. With `engine_count == 2` the left engine sits at
    /// `-engine_lateral_arm` and the right at `+engine_lateral_arm`. Governs
    /// the arm of the asymmetric-thrust yawing moment (Vmc).
    #[serde(default = "default_engine_lateral_arm")]
    pub engine_lateral_arm: f64,
    /// Dimensionless propeller-torque rolling-moment constant. Positive values
    /// roll the aircraft opposite the prop rotation when power is applied; in
    /// a counter-rotating twin the two torque couples nearly cancel, so this
    /// is typically small.
    #[serde(default = "default_prop_torque_coeff")]
    pub prop_torque_coeff: f64,
    /// Dimensionless P-factor yawing-moment constant. At high power and high
    /// angle of attack the descending prop blade produces more thrust than the
    /// ascending one, yawing the nose. Negative = convention for standard
    /// clockwise-rotating (right-hand) props when viewed from behind.
    #[serde(default = "default_p_factor_coeff")]
    pub p_factor_coeff: f64,
    /// Dimensionless gyroscopic-precession coupling constant. Scales the pitch
    /// and yaw couples produced when the spinning propeller mass is pitched or
    /// yawed (proportional to the propeller angular momentum).
    #[serde(default = "default_gyro_coeff")]
    pub gyro_coeff: f64,
    /// Asymmetric engine throttle split, `-1..=1`, applied on top of the master
    /// throttle. `0` = both engines at the master setting; `-1` = left engine
    /// shut down / right at full; `+1` = right engine shut down / left at full.
    /// Adjustable at runtime (e.g. to inject an engine-out for the Vmc check).
    #[serde(default = "default_throttle_split")]
    pub throttle_split: f64,

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

    // --- Flap (trailing-edge) aerodynamic increments ---
    /// Lift increment per radian of flap deflection, sqrt-free linear
    /// `dCL = cl_flap * delta_flap`. Fixed/plain flaps on a light aircraft.
    #[serde(default = "default_cl_flap")]
    pub cl_flap: f64,
    /// Drag increment per radian of flap deflection, `dCD = cd_flap * delta_flap`.
    #[serde(default = "default_cd_flap")]
    pub cd_flap: f64,
    /// Pitching-moment increment per radian of flap deflection,
    /// `dCm = cm_flap * delta_flap`. Negative = nose-down (flaps push nose down).
    #[serde(default = "default_cm_flap")]
    pub cm_flap: f64,
    /// Reduction in the positive stall angle (radians) at full flap; flaps
    /// lower the stall AoA. `alpha_stall_pos = alpha_stall_pos - flap_stall_shift * delta_flap`.
    #[serde(default = "default_flap_stall_shift")]
    pub flap_stall_shift: f64,

    // --- Steep-bank spiral nose-drop ---
    /// Additional nose-down pitching-moment coefficient that engages at steep
    /// bank angle, so a hand-off aircraft in a high bank falls into a dive
    /// (spiral divergence) instead of being able to climb with its wings
    /// near-vertical. Zero at shallow bank; ramps to full magnitude beyond
    /// ~45 deg of bank. Negative = nose-down, acting about body Y.
    #[serde(default = "default_spiral_nose_drop_cm")]
    pub spiral_nose_drop_cm: f64,

    /// Flat-plate panels that make up the aircraft's collision shape. Each
    /// panel is a surface with a centre of pressure, body-frame normal and
    /// area; the imposed wind pours onto every panel and produces a
    /// shape-dependent force + moment (a flat-plate pressure model). This is
    /// what lets the two aircraft models feel different in the wind.
    #[serde(default)]
    pub collision_panels: Vec<CollisionPanel>,
}

/// One flat-plate surface of the aircraft's collision shape, defined in the
/// **body** frame (nose +X, up +Y, right +Z) — the same convention as the
/// aerodynamics and the visual model.
#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "snake_case")]
pub struct CollisionPanel {
    /// Surface area in square metres (m²).
    pub area: f64,
    /// Unit normal of the panel in the body frame (components `[nx, ny, nz]`).
    pub normal: [f64; 3],
    /// Centre of pressure relative to the CG in the body frame, `[x, y, z]` m.
    pub cp: [f64; 3],
    /// Flat-plate drag coefficient for this panel (typical 1.2 – 2.0).
    #[serde(default = "default_panel_cd")]
    pub cd: f64,
}

fn default_panel_cd() -> f64 { 1.5 }

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
