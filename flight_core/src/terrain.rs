//! Terrain height model for the flight dynamics engine.
//!
//! Models the ground as a smooth, differentiable elevation surface
//! `h(x, y)` in the Earth **NED** frame (`x` = north, `y` = east), in metres
//! above the reference (sea-level) datum. The surface is built from a
//! superposition of smooth Gaussian mountains / valleys, so it has well-defined
//! gradients that orographic wind and ground effect can use without the
//! normal discontinuities of a raw heightmap.
//!
//! A flat (`Terrain::flat()`) surface is the default and disables terrain
//! effects, so aircraft configs and the RL environment that don't opt in are
//! unaffected.

use nalgebra::{Vector2, Vector3};

/// One smooth Gaussian mountain or valley centred on `(cx, cy)`.
///
/// Positive `amplitude` is a mountain peak rising `amplitude` metres above the
/// surrounding datum; negative is a valley. `sigma` controls the horizontal
/// half-width (m) of the feature (larger = wider, gentler).
#[derive(Debug, Clone, Copy)]
pub struct TerrainHill {
    /// Centroid (north, east) in metres.
    pub centre: [f64; 2],
    /// Peak (or trough) height in metres above the datum.
    pub amplitude: f64,
    /// Horizontal width scale (m); the hill is ~3·sigma across.
    pub sigma: f64,
}

/// A smooth analytic elevation surface.
#[derive(Debug, Clone)]
pub struct Terrain {
    hills: Vec<TerrainHill>,
}

impl Default for Terrain {
    fn default() -> Self {
        Self::flat()
    }
}

impl Terrain {
    /// A perfectly flat surface at height 0 — no terrain effects.
    pub fn flat() -> Self {
        Self { hills: Vec::new() }
    }

    /// Build a terrain from a list of smooth hills / valleys.
    pub fn from_hills(hills: Vec<TerrainHill>) -> Self {
        Self { hills }
    }

    /// Ground elevation `h(x, y)` at a NED (north, east) position, in metres
    /// above the reference datum. Always >= 0 for the supplied hill set (a
    /// layered-cake of positive hills over a zero datum).
    pub fn height(&self, north: f64, east: f64) -> f64 {
        let mut h = 0.0;
        for hill in &self.hills {
            let dx = north - hill.centre[0];
            let dy = east - hill.centre[1];
            let r2 = (dx * dx + dy * dy) / (2.0 * hill.sigma * hill.sigma);
            h += hill.amplitude * (-r2).exp();
        }
        h
    }

    /// Local surface gradient `(dh/dnorth, dh/deast)` at a NED position.
    /// Used to deflect the steady wind into orographic up/downdrafts.
    pub fn gradient(&self, north: f64, east: f64) -> Vector2<f64> {
        let mut g = Vector2::zeros();
        for hill in &self.hills {
            let dx = north - hill.centre[0];
            let dy = east - hill.centre[1];
            let r2 = (dx * dx + dy * dy) / (2.0 * hill.sigma * hill.sigma);
            let gauss = (-r2).exp();
            let scale = hill.amplitude * gauss / hill.sigma;
            g.x -= dx * scale / hill.sigma; // d/dx (exp(-(dx²+dy²)/2σ²)) = -(dx/σ²)·exp
            g.y -= dy * scale / hill.sigma;
        }
        g
    }

    /// Altitude of a point **above the terrain surface** at a NED position and
    /// geometric altitude (`alt`), in metres. Negative means below ground.
    pub fn altitude_above_ground(&self, north: f64, east: f64, altitude: f64) -> f64 {
        altitude - self.height(north, east)
    }

    /// Orographic vertical wind (m/s, **NED**: positive = downward air) induced
    /// by the steady horizontal wind blowing over the terrain slope.
    ///
    /// Where the wind drives air up a windward slope the air is forced to rise,
    /// producing an updraft (negative NED vertical wind); over a leeward slope
    /// it produces a downdraft. The strength follows the projection of the
    /// horizontal wind onto the local surface gradient and decays with altitude
    /// above ground (`agl_decay_scale` m), so the effect is confined to the
    /// terrain boundary layer.
    ///
    /// `wind_earth` is the steady mean wind in NED `(north, east, down)` m/s.
    pub fn orographic_wind(
        &self,
        wind_earth: &Vector3<f64>,
        north: f64,
        east: f64,
        altitude: f64,
        agl_decay_scale: f64,
    ) -> Vector3<f64> {
        let agl = self.altitude_above_ground(north, east, altitude);
        if agl <= 0.0 {
            return Vector3::zeros();
        }
        let g = self.gradient(north, east);
        // Vertical air velocity from advection up the slope. Wind toward an
        // upslope rises: w_z(NED, positive down) = -(wind·∇h).
        let w_z = -(wind_earth.x * g.x + wind_earth.y * g.y);
        let decay = (-agl / agl_decay_scale.max(1e-6)).exp();
        Vector3::new(0.0, 0.0, w_z * decay)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn flat_terrain_is_always_zero_height_and_no_gradient() {
        let t = Terrain::flat();
        assert_eq!(t.height(100.0, -50.0), 0.0);
        assert_eq!(t.altitude_above_ground(100.0, -50.0, 60.0), 60.0);
        assert_eq!(t.gradient(100.0, -50.0).norm(), 0.0);
    }

    #[test]
    fn mountain_peak_is_amplitude_and_increases_altitude_over_ground() {
        let t = Terrain::from_hills(vec![TerrainHill {
            centre: [0.0, 0.0],
            amplitude: 400.0,
            sigma: 500.0,
        }]);
        // At the peak the ground is 400 m.
        assert!((t.height(0.0, 0.0) - 400.0).abs() < 1e-9);
        // Far away it returns to the flat datum.
        assert!(t.height(5000.0, 5000.0) < 1.0);
        // Altitude above ground drops near the peak.
        let agl = t.altitude_above_ground(0.0, 0.0, 500.0);
        assert!((agl - 100.0).abs() < 1e-9);
    }

    #[test]
    fn gradient_points_downhill_away_from_peak() {
        let t = Terrain::from_hills(vec![TerrainHill {
            centre: [0.0, 0.0],
            amplitude: 400.0,
            sigma: 500.0,
        }]);
        // East of the peak (north=0, east>0) the height drops as east grows,
        // so dh/deast should be negative; dh/dnorth should be ~0 on the east
        // axis.
        let g = t.gradient(0.0, 100.0);
        assert!(
            g.y < 0.0,
            "gradient east of peak should be negative, got {}",
            g.y
        );
        assert!(g.x.abs() < 1e-6, "gradient north on east-axis should be ~0");
    }
}
