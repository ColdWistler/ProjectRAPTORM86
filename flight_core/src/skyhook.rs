//! Skyhook system for aerial recovery and payload delivery.
//!
//! Implements cable dynamics, capture logic, winching, and payload attachment.

use crate::config::SkyhookConfig;
use crate::state::AircraftState;
use nalgebra::Vector3;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SkyhookMode {
    Idle,
    Deployed,
    Armed,
    Attached,
    Winching,
    Released,
}

#[derive(Debug, Clone)]
pub struct SkyhookState {
    pub mode: SkyhookMode,
    pub deployed: bool,
    pub attached: bool,
    pub cable_length: f64,
    pub cable_rest_length: f64,
    pub tension: f64,
    pub winch_velocity: f64, // +1 extend, 0 hold, -1 retract
    pub target_pos_ned: Option<Vector3<f64>>,
    pub target_ring_radius: f64,
    pub last_capture_attempt: f64,
}

impl Default for SkyhookState {
    fn default() -> Self {
        Self {
            mode: SkyhookMode::Idle,
            deployed: false,
            attached: false,
            cable_length: 5.0,
            cable_rest_length: 5.0,
            tension: 0.0,
            winch_velocity: 0.0,
            target_pos_ned: None,
            target_ring_radius: 0.35,
            last_capture_attempt: 0.0,
        }
    }
}

impl SkyhookState {
    pub fn new(config: &SkyhookConfig) -> Self {
        Self {
            cable_rest_length: config.cable_length_max.min(8.0).max(2.0),
            cable_length: config.cable_length_max.min(8.0).max(2.0),
            target_ring_radius: config.capture_radius,
            ..Default::default()
        }
    }
}

/// Compute hook tip position in NED frame
pub fn hook_tip_position_ned(state: &AircraftState, hook_offset_body: &[f64; 3]) -> Vector3<f64> {
    let rot = state.rotation_earth_to_body();
    let rot_inv = rot.inverse();
    let hook_body = Vector3::new(hook_offset_body[0], hook_offset_body[1], hook_offset_body[2]);
    let hook_ned = rot_inv.transform_vector(&hook_body);
    Vector3::new(state.pos_x, state.pos_y, state.pos_z) + hook_ned
}

/// Apply skyhook forces and update state
///
/// `_current_time` is accepted for API stability; the winch model is
/// step-driven and does not use it yet.
pub fn update(
    state: &mut AircraftState,
    skyhook: &mut SkyhookState,
    config: &SkyhookConfig,
    dt: f64,
    _current_time: f64,
) -> (Vector3<f64>, Vector3<f64>) {
    let mut force = Vector3::zeros();
    let mut moment = Vector3::zeros();
    
    // Update winch
    if skyhook.deployed && !skyhook.attached {
        skyhook.cable_rest_length += skyhook.winch_velocity * config.winch_rate * dt;
        skyhook.cable_rest_length = skyhook.cable_rest_length.clamp(2.0, config.cable_length_max);
        skyhook.cable_length = skyhook.cable_rest_length;
    }
    
    // Check for capture if armed and have target
    if skyhook.mode == SkyhookMode::Armed && !skyhook.attached {
        if let Some(target_pos) = skyhook.target_pos_ned {
            let hook_pos = hook_tip_position_ned(state, &config.hook_offset_body);
            let dist = (hook_pos - target_pos).norm();
            let hook_vel_ned = state.velocity_ned();
            let rel_vel = hook_vel_ned.norm();
            
            if dist < skyhook.target_ring_radius && rel_vel < config.max_capture_speed {
                // Attempt capture
                skyhook.attached = true;
                skyhook.mode = SkyhookMode::Attached;
                skyhook.cable_length = dist;
            }
        }
    }
    
    // Apply cable forces if attached
    if skyhook.attached {
        if let Some(target_pos) = skyhook.target_pos_ned {
            let hook_pos = hook_tip_position_ned(state, &config.hook_offset_body);
            let cable_vec = target_pos - hook_pos;
            let current_length = cable_vec.norm();
            skyhook.cable_length = current_length;
            
            if current_length > 1e-6 {
                let cable_dir = cable_vec / current_length;
                let extension = current_length - skyhook.cable_rest_length;
                let hook_vel = state.velocity_ned();
                // Simple relative velocity along cable (approximate)
                let vel_along = hook_vel.dot(&cable_dir);
                let tension = (config.cable_stiffness * extension - config.cable_damping * vel_along)
                    .max(0.0);
                skyhook.tension = tension.min(3500.0);
                
                if skyhook.tension > config.tension_breakaway {
                    // Breakaway
                    skyhook.attached = false;
                    skyhook.mode = SkyhookMode::Deployed;
                    skyhook.tension = 0.0;
                } else {
                    // Apply tension toward target
                    force = cable_dir * skyhook.tension;
                    
                    // Apply moment about CG
                    let rot_inv = state.rotation_earth_to_body().inverse();
                    let hook_body = Vector3::new(config.hook_offset_body[0], 
                                                config.hook_offset_body[1], 
                                                config.hook_offset_body[2]);
                    let force_body = rot_inv.transform_vector(&force);
                    moment = hook_body.cross(&force_body);
                }
            }
        }
    }
    
    // Winch when attached
    if skyhook.mode == SkyhookMode::Attached && skyhook.winch_velocity < 0.0 {
        skyhook.mode = SkyhookMode::Winching;
    }
    if skyhook.mode == SkyhookMode::Winching {
        skyhook.cable_rest_length += skyhook.winch_velocity * config.winch_rate * dt;
        skyhook.cable_rest_length = skyhook.cable_rest_length.max(1.0);
        if skyhook.cable_rest_length < 1.5 && skyhook.winch_velocity < 0.0 {
            // Close to winched in
        }
    }
    
    (force, moment)
}

pub fn deploy(skyhook: &mut SkyhookState) {
    if skyhook.mode == SkyhookMode::Idle {
        skyhook.deployed = true;
        skyhook.mode = SkyhookMode::Deployed;
    }
}

pub fn arm(skyhook: &mut SkyhookState) {
    if skyhook.mode == SkyhookMode::Deployed {
        skyhook.mode = SkyhookMode::Armed;
    }
}

pub fn release(skyhook: &mut SkyhookState) {
    skyhook.attached = false;
    skyhook.mode = SkyhookMode::Deployed;
    skyhook.tension = 0.0;
}
