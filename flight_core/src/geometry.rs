//! Geometry utilities for morphing airframes.
//!
//! Handles rotation of pivoted wing panels for variable sweep and
//! provides helpers for computing positions of moving points like the hook.

use crate::config::{CollisionPanel, MorphingWingConfig};
use nalgebra::{Rotation3, Vector3};

/// Build the effective collision panel list for the current sweep angle.
/// Fixed panels remain unchanged; panels on pivots are rotated about
/// their pivot axis in the body frame.
pub fn build_effective_panels(
    morphing: &MorphingWingConfig,
    sweep_rad: f64,
) -> Vec<CollisionPanel> {
    let mut panels: Vec<CollisionPanel> = Vec::new();
    
    // Add fixed panels
    panels.extend_from_slice(&morphing.fixed_panels);
    
    if !morphing.enabled {
        for pivot in &morphing.pivots {
            panels.extend_from_slice(&pivot.panels);
        }
        return panels;
    }
    
    for pivot in &morphing.pivots {
        let pivot_pt = Vector3::new(pivot.pivot[0], pivot.pivot[1], pivot.pivot[2]);
        let p_axis = Vector3::new(pivot.axis[0], pivot.axis[1], pivot.axis[2]);
        let p_axis_unit = if p_axis.norm() < 1e-12 { nalgebra::Unit::new_normalize(Vector3::z()) } else { nalgebra::Unit::new_normalize(p_axis) };
        let p_rot = Rotation3::from_axis_angle(&p_axis_unit, sweep_rad);
        
        for panel in &pivot.panels {
            let cp = Vector3::new(panel.cp[0], panel.cp[1], panel.cp[2]);
            let normal = Vector3::new(panel.normal[0], panel.normal[1], panel.normal[2]);
            
            let cp_rel = cp - pivot_pt;
            let cp_rot = p_rot * cp_rel + pivot_pt;
            let normal_rot = p_rot * normal;
            
            panels.push(CollisionPanel {
                area: panel.area,
                normal: [normal_rot.x, normal_rot.y, normal_rot.z],
                cp: [cp_rot.x, cp_rot.y, cp_rot.z],
                cd: panel.cd,
                id: panel.id.clone(),
            });
        }
    }
    
    panels
}

/// Get hook tip position in body frame
pub fn hook_tip_body_position(skyhook: &crate::config::SkyhookConfig) -> Vector3<f64> {
    Vector3::new(skyhook.hook_offset_body[0], 
                 skyhook.hook_offset_body[1], 
                 skyhook.hook_offset_body[2])
}
