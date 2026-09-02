//! Terrain height model for the flight dynamics engine.
//!
//! Models the ground as a smooth elevation surface `h(x, y)` in the Earth
//! **NED** frame (`x` = north, `y` = east), in metres above the reference
//! (sea-level) datum. Two representations are supported, used interchangeably
//! through a single `Terrain` handle:
//!
//! * **Analytic hills** — a superposition of smooth Gaussian mountains / valleys
//!   with well-defined gradients (for orographic wind) and no sharp edges.
//! * **Sampled grid** — a uniform heightfield `north0/east0 + (i·s, j·s)` built
//!   from (e.g.) an imported terrain mesh, queried with bilinear interpolation
//!   and central-difference gradients.
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

/// A uniform rectangular grid of ground-elevation samples, `(north, east)` in
/// metres above the datum. Cell `(i, j)` covers
/// `north = north0 + i·spacing`, `east = east0 + j·spacing`.
///
/// Heights are stored **row-major** (`index = j·nx + i`): `j` indexes east
/// (columns), `i` indexes north (rows).
#[derive(Debug, Clone)]
pub struct TerrainGrid {
    north0: f64,
    east0: f64,
    spacing: f64,
    nx: usize,
    nz: usize,
    heights: Vec<f64>,
}

impl TerrainGrid {
    /// Build a grid. `heights` must be exactly `nx * nz` (row-major, `j·nx + i`).
    pub fn new(north0: f64, east0: f64, spacing: f64, nx: usize, nz: usize, heights: Vec<f64>) -> Self {
        assert_eq!(
            heights.len(),
            nx * nz,
            "TerrainGrid height count {} != {}x{}",
            heights.len(),
            nx,
            nz
        );
        debug_assert!(spacing > 0.0);
        TerrainGrid {
            north0,
            east0,
            spacing,
            nx,
            nz,
            heights,
        }
    }

    #[inline]
    fn cell(&self, i: usize, j: usize) -> f64 {
        self.heights[j * self.nx + i]
    }

    /// Elevation at an arbitrary (north, east) via bilinear interpolation over
    /// the four surrounding cells. Out-of-range queries clamp to the grid edge.
    pub fn height(&self, north: f64, east: f64) -> f64 {
        let fx = (north - self.north0) / self.spacing.max(1e-9);
        let fz = (east - self.east0) / self.spacing.max(1e-9);
        let i = fx.floor().clamp(0.0, (self.nx as f64) - 1.0) as usize;
        let j = fz.floor().clamp(0.0, (self.nz as f64) - 1.0) as usize;
        let i1 = (i + 1).min(self.nx - 1);
        let j1 = (j + 1).min(self.nz - 1);

        let h00 = self.cell(i, j);
        let h10 = self.cell(i1, j);
        let h01 = self.cell(i, j1);
        let h11 = self.cell(i1, j1);

        let ti = (fx - i as f64).clamp(0.0, 1.0);
        let tj = (fz - j as f64).clamp(0.0, 1.0);

        let top = h00 + (h10 - h00) * ti;
        let bottom = h01 + (h11 - h01) * ti;
        top + (bottom - top) * tj
    }

    /// Slope vector `(dh/dnorth, dh/deast)` by central differences (m/m).
    pub fn gradient(&self, north: f64, east: f64) -> Vector2<f64> {
        let fx = (north - self.north0) / self.spacing.max(1e-9);
        let fz = (east - self.east0) / self.spacing.max(1e-9);
        let i = fx.clamp(1.0, (self.nx as f64) - 2.0) as usize;
        let j = fz.clamp(1.0, (self.nz as f64) - 2.0) as usize;

        let dh_dn = (self.cell(i + 1, j) - self.cell(i - 1, j)) / (2.0 * self.spacing);
        let dh_de = (self.cell(i, j + 1) - self.cell(i, j - 1)) / (2.0 * self.spacing);
        Vector2::new(dh_dn, dh_de)
    }
}

/// A smooth elevation surface, from analytic hills and/or a sampled grid.
///
/// If a grid is present it takes precedence over the analytic hills (the two
/// are not combined). The default is flat.
#[derive(Debug, Clone)]
pub struct Terrain {
    hills: Vec<TerrainHill>,
    grid: Option<TerrainGrid>,
}

impl Default for Terrain {
    fn default() -> Self {
        Self::flat()
    }
}

impl Terrain {
    /// A perfectly flat surface at height 0 — no terrain effects.
    pub fn flat() -> Self {
        Self {
            hills: Vec::new(),
            grid: None,
        }
    }

    /// Build a terrain from a list of smooth hills / valleys.
    pub fn from_hills(hills: Vec<TerrainHill>) -> Self {
        Self { hills, grid: None }
    }

    /// Build a terrain from a sampled grid heightfield.
    pub fn from_grid(north0: f64, east0: f64, spacing: f64, nx: usize, nz: usize, heights: Vec<f64>) -> Self {
        Self {
            hills: Vec::new(),
            grid: Some(TerrainGrid::new(north0, east0, spacing, nx, nz, heights)),
        }
    }

    /// Ground elevation `h(north, east)` at a NED position, in metres above the
    /// reference datum.
    pub fn height(&self, north: f64, east: f64) -> f64 {
        match &self.grid {
            Some(g) => g.height(north, east),
            None => {
                let mut h = 0.0;
                for hill in &self.hills {
                    let dx = north - hill.centre[0];
                    let dy = east - hill.centre[1];
                    let r2 = (dx * dx + dy * dy) / (2.0 * hill.sigma * hill.sigma);
                    h += hill.amplitude * (-r2).exp();
                }
                h
            }
        }
    }

    /// Local surface gradient `(dh/dnorth, dh/deast)` at a NED position.
    /// Used to deflect the steady wind into orographic up/downdrafts.
    pub fn gradient(&self, north: f64, east: f64) -> Vector2<f64> {
        match &self.grid {
            Some(g) => g.gradient(north, east),
            None => {
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
        }
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

    #[test]
    fn grid_height_interpolates_bilinearly_between_cells() {
        // 3x3 grid, spacing 100 m, all zero except cell (i=1, j=1) at
        // north=100, east=100 which is 200 m. A query halfway between that cell
        // and the (0,0) cell must interpolate to 50 m.
        let mut heights = vec![0.0f64; 9];
        heights[1 + 1 * 3] = 200.0;
        let t = Terrain::from_grid(0.0, 0.0, 100.0, 3, 3, heights);

        assert!((t.height(100.0, 100.0) - 200.0).abs() < 1e-9, "high cell");
        assert!((t.height(0.0, 0.0) - 0.0).abs() < 1e-9, "corner cell");
        // Between them: x=50, y=50 → 0.25·200 = 50.
        assert!((t.height(50.0, 50.0) - 50.0).abs() < 1e-9, "bilinear midpoint");
        // Out-of-range clamps to the edge value.
        assert!((t.height(5000.0, 5000.0) - 0.0).abs() < 1e-9);
    }

    #[test]
    fn grid_gradient_is_slope_of_the_surface() {
        let mut heights = vec![0.0f64; 9];
        // Cells with i=2 (north = 200m) are 20 m higher than cells with i=0
        // (north = 0m), i.e. a linear 0.1 m/m climb in north.
        for j in 0..3 {
            heights[0 + j * 3] = 0.0;
            heights[2 + j * 3] = 20.0;
        }
        let t = Terrain::from_grid(0.0, 0.0, 100.0, 3, 3, heights);
        let g = t.gradient(100.0, 100.0);
        assert!((g.x - 0.1).abs() < 1e-9, "dh/dnorth = 20m/200m, got {}", g.x);
        assert!(g.y.abs() < 1e-9, "dh/deast should be 0");
    }
}