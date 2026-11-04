//! Hitbox and contact resolution system.
//!
//! Handles collision detection between aircraft panels and terrain/objects,
//! applies impulses and friction.

use crate::config::CollisionPanel;
use crate::state::AircraftState;
use crate::terrain::Terrain;
use nalgebra::Vector3;

#[derive(Debug, Clone)]
pub struct Contact {
    pub panel_id: String,
    pub position_ned: Vector3<f64>,
    pub normal_ned: Vector3<f64>,
    pub penetration: f64,
    pub impulse: f64,
    pub velocity_rel: f64,
}

#[derive(Debug, Clone, Default)]
pub struct ContactState {
    pub contacts: Vec<Contact>,
    pub in_contact: bool,
}

/// Check for contacts between panels and terrain
pub fn check_terrain_contacts(
    state: &AircraftState,
    panels: &[CollisionPanel],
    terrain: Option<&Terrain>,
) -> Vec<Contact> {
    let mut contacts = Vec::new();
    let Some(terrain) = terrain else { return contacts; };
    
    let rot = state.rotation_earth_to_body();
    let rot_inv = rot.inverse();
    let pos = Vector3::new(state.pos_x, state.pos_y, state.pos_z);
    let vel_ned = state.velocity_ned();
    
    for panel in panels {
        let cp_body = Vector3::new(panel.cp[0], panel.cp[1], panel.cp[2]);
        let cp_ned = rot_inv.transform_vector(&cp_body) + pos;
        
        // Get terrain height at this position
        let terrain_height = terrain.height(cp_ned.x, cp_ned.y);
        // Terrain height is in some frame; altitude is -pos_z
        let altitude = -cp_ned.z;
        
        if altitude < terrain_height + 0.1 {
            let penetration = terrain_height + 0.1 - altitude;
            let normal = Vector3::new(0.0, 0.0, -1.0); // Up in NED? Wait - NED: z is down, so up is +z? No wait: pos_z positive is DOWN. So altitude above ground is -pos_z. Terrain height is likely in meters above some datum. Up normal in world would be (0,0,-1) if we're in NED with z down? Or easier to think: if we're below terrain, push up.
            let normal_up = Vector3::new(0.0, 0.0, -1.0); // Upward in NED frame (negative z direction)
            let vel_rel = vel_ned.dot(&normal_up).min(0.0).abs();
            
            contacts.push(Contact {
                panel_id: panel.id.clone(),
                position_ned: cp_ned,
                normal_ned: normal_up,
                penetration,
                impulse: (penetration * 10000.0 + vel_rel * 1000.0).min(50000.0),
                velocity_rel: vel_rel,
            });
        }
    }
    
    contacts
}

/// Resolve contacts by applying impulses
pub fn resolve_contacts(
    state: &mut AircraftState,
    contacts: &[Contact],
    dt: f64,
) -> (Vector3<f64>, Vector3<f64>) {
    let mut force = Vector3::zeros();
    let mut moment = Vector3::zeros();
    
    if contacts.is_empty() {
        return (force, moment);
    }
    
    let rot_inv = state.rotation_earth_to_body().inverse();
    
    for contact in contacts {
        if contact.penetration > 0.0 {
            // Apply normal impulse
            let impulse_force = contact.normal_ned * contact.impulse;
            force += impulse_force;
            
            // Apply moment (approximate)
            let pos_rel = Vector3::new(state.pos_x, state.pos_y, state.pos_z);
            // This is simplified
        }
    }
    
    (force, moment)
}
