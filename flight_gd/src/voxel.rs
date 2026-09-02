//! Voxel wind-body builder for *arbitrary* imported 3D models.
//!
//! The wind tunnel normally uses the coefficient-based `flight_core` aero model
//! tuned per aircraft (`MQI` / `TwinEngine`). When the user imports a random
//! 3D model there are no coefficients, so this module conservatively voxelizes
//! the triangle soup into a coarse occupancy grid, then extracts the *exposed*
//! cell faces as flat-plate [`CollisionPanel`]s. `compute_shape_wind` then
//! turns those panels into a real pressure-drag force and moment — drag against
//! the wind, side force from cross-flow, and a centre-of-pressure moment about
//! the model's origin — without ever needing aerodynamic constants.
//!
//! Geometry is expected in the body frame used everywhere else: nose +X, up +Y,
//! right +Z. The model should already be centred at the origin (CG) and scaled
//! to a sensible size; the caller normalizes it before handing the triangles in.

use flight_core::config::CollisionPanel;

/// Flat-plate drag coefficient applied to every exposed voxel face. 1.2 is the
/// classic bluff-body / flat-plate value; a needle would be lower, a flat plate
/// normal to the wind closer to 2.0. Single value keeps the model simple.
const PANEL_CD: f64 = 1.2;

/// Minimum grid resolution allowed.
const MIN_RES: usize = 4;
/// Maximum grid resolution. Above this the panel count explodes for no benefit.
const MAX_RES: usize = 24;

// --- Triangle / AABB helpers -------------------------------------------------

fn sub(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

fn cross(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

fn dot(a: [f64; 3], b: [f64; 3]) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

/// Resolution-independent geometry metrics of an imported triangle mesh,
/// computed directly from the source triangles (not the coarse voxel shell).
/// This lets the realistic drag model use the *true* wetted area and frontal
/// silhouette regardless of the voxelization grid size.
#[derive(Debug, Clone, Copy, Default)]
pub struct MeshMetrics {
    /// True summed triangle area (m²) — the wetted surface area.
    pub wetted_area: f64,
    /// True projected frontal area normal to the stream axis (m²).
    pub frontal_area: f64,
    /// Axis-aligned bounding-box extent (max minus min) per axis (m).
    pub size_x: f64,
    pub size_y: f64,
    pub size_z: f64,
}

/// Compute mesh metrics from the source triangle soup. `stream_axis` (default
/// +X, the tunnel's free-stream nose direction) selects the frontal-projection
/// axis. All quantities are independent of any voxelization resolution.
pub fn mesh_metrics(vertices: &[[f64; 3]], triangles: &[[usize; 3]]) -> MeshMetrics {
    let mut m = MeshMetrics::default();
    let mut minb = [f64::INFINITY; 3];
    let mut maxb = [f64::NEG_INFINITY; 3];

    let mut frontal = 0.0; // accumulated |projected area| on the X-normal plane
    for t in triangles {
        let a = vertices[t[0]];
        let b = vertices[t[1]];
        let c = vertices[t[2]];
        // Triangle normal (length = 2x area). Use the X component for the
        // frontal projection (stream axis = +X).
        let n = cross(sub(b, a), sub(c, a));
        let tri_area = 0.5 * (dot(n, n)).sqrt();
        m.wetted_area += tri_area;
        // Signed projected area onto the YZ plane (normal to +X).
        frontal += n[0].abs() * 0.5;
        for v in [a, b, c] {
            for axis in 0..3 {
                minb[axis] = minb[axis].min(v[axis]);
                maxb[axis] = maxb[axis].max(v[axis]);
            }
        }
    }
    // A closed shell's upwind+downwind faces both project; half the sum is the
    // true frontal silhouette.
    m.frontal_area = 0.5 * frontal;
    if m.frontal_area < 1e-9 {
        m.frontal_area = m.wetted_area.max(1e-9);
    }
    m.size_x = (maxb[0] - minb[0]).max(1e-9);
    m.size_y = (maxb[1] - minb[1]).max(1e-9);
    m.size_z = (maxb[2] - minb[2]).max(1e-9);
    m
}

/// Conservative voxelization of a triangle soup into flat-plate panels.
///
/// Returns the exposed-face panels. `vertices` are `[x, y, z]` triples in body
/// frame; `triangles` are `[a, b, c]` indices into `vertices`. `resolution` is
/// the number of grid cells across the longest bounding-box edge (clamped).
///
/// Algorithm: each cell is classified solid by a ray-cast *inside/outside*
/// parity test (standard for arbitrary triangle soups — no manifold
/// requirement). Only cells whose centre is inside the model are solid, so a
/// closed shell reliably produces a solid body whose *outer* faces become the
/// pressure panels.
pub fn voxelize_panels(
    vertices: &[[f64; 3]],
    triangles: &[[usize; 3]],
    resolution: usize,
) -> Vec<CollisionPanel> {
    if vertices.is_empty() || triangles.is_empty() {
        return Vec::new();
    }

    // Bounding box of the input mesh.
    let mut minb = [f64::INFINITY; 3];
    let mut maxb = [f64::NEG_INFINITY; 3];
    for v in vertices {
        for a in 0..3 {
            minb[a] = minb[a].min(v[a]);
            maxb[a] = maxb[a].max(v[a]);
        }
    }
    let ext = |a: usize| (maxb[a] - minb[a]).max(1e-9);
    let max_ext = ext(0).max(ext(1)).max(ext(2));

    let res = resolution.clamp(MIN_RES, MAX_RES);
    let cell = max_ext / res as f64;

    let nx = ((ext(0) / cell).ceil() as usize).max(1);
    let ny = ((ext(1) / cell).ceil() as usize).max(1);
    let nz = ((ext(2) / cell).ceil() as usize).max(1);

    let cell_center = |i: usize, j: usize, k: usize| -> [f64; 3] {
        [
            minb[0] + (i as f64 + 0.5) * cell,
            minb[1] + (j as f64 + 0.5) * cell,
            minb[2] + (k as f64 + 0.5) * cell,
        ]
    };
    let idx = |i: usize, j: usize, k: usize| -> usize { k * (ny * nx) + j * nx + i };

    // Bucket every triangle by its *yz footprint*. A ray cast along +X from a
    // cell centre at (y, z) can only cross triangles whose yz-projection covers
    // that (y, z), so the per-cell check inspects only nearby geometry instead
    // of the whole triangle soup.
    //
    // Precompute each triangle's plane normal + offset so the crossing height
    // is a single division later.
    let plane: Vec<([f64; 3], f64)> = triangles
        .iter()
        .map(|t| {
            let v0 = vertices[t[0]];
            let e1 = sub(vertices[t[1]], v0);
            let e2 = sub(vertices[t[2]], v0);
            let n = cross(e1, e2);
            (n, dot(n, v0))
        })
        .collect();

    let mut buckets: Vec<Vec<usize>> = vec![Vec::new(); ny * nz];
    for (ti, t) in triangles.iter().enumerate() {
        let tri = [vertices[t[0]], vertices[t[1]], vertices[t[2]]];
        let ymin = tri[0][1].min(tri[1][1]).min(tri[2][1]);
        let ymax = tri[0][1].max(tri[1][1]).max(tri[2][1]);
        let zmin = tri[0][2].min(tri[1][2]).min(tri[2][2]);
        let zmax = tri[0][2].max(tri[1][2]).max(tri[2][2]);
        let j0 = (((ymin - minb[1]) / cell - 1e-9).floor() as isize).max(0);
        let k0 = (((zmin - minb[2]) / cell - 1e-9).floor() as isize).max(0);
        let j1 = (((ymax - minb[1]) / cell + 1e-9).ceil() as isize).min(ny as isize);
        let k1 = (((zmax - minb[2]) / cell + 1e-9).ceil() as isize).min(nz as isize);

        // Half-open range [j0, j1) × [k0, k1). The epsilon keeps triangles that
        // lie exactly on a cell boundary from vanishing out of the buckets.
        for k in k0..k1 {
            for j in j0..j1 {
                buckets[(k.min(nz as isize - 1) as usize * ny)
                    + j.min(ny as isize - 1) as usize]
                    .push(ti);
            }
        }
    }

    // 2D point-in-triangle test in the (y, z) plane (used to confirm a plane
    // crossing actually lands inside the triangle).
    let inside_yz = |py: f64, pz: f64, a: &[f64; 3], b: &[f64; 3], c: &[f64; 3]| -> bool {
        let s1 = (b[1] - a[1]) * (pz - a[2]) - (b[2] - a[2]) * (py - a[1]);
        let s2 = (c[1] - b[1]) * (pz - b[2]) - (c[2] - b[2]) * (py - b[1]);
        let s3 = (a[1] - c[1]) * (pz - c[2]) - (a[2] - c[2]) * (py - c[1]);
        !((s1 < 0.0 || s2 < 0.0 || s3 < 0.0) && (s1 > 0.0 || s2 > 0.0 || s3 > 0.0))
    };

    // Classify cells: a cell is solid when a ray from its centre cast along +X
    // crosses the model an ODD number of times (inside/outside parity). For
    // each candidate triangle in the cell's yz bucket, solve the plane equation
    // for the ray crossing height and count it when it lies ahead and inside.
    //
    // The ray is de-symmetrized with a tiny per-cell jitter: grid centres can
    // otherwise land exactly on mesh edges, where an *inclusive* point-in-
    // triangle test double-counts the shared edge of two adjacent triangles
    // (2 crossings → wrongly "outside"). A sub-cell offset keeps that from
    // ever happening while the geometry model is unaffected.
    let jitter = |a: usize, b: usize, c: usize| -> f64 {
        let x = a as f64 * 0.618033988749895
            + b as f64 * 0.381966011250105
            + c as f64 * 0.316624790355;
        let x = (x * 12.9898 + 78.233).sin() * 43758.5453;
        x - x.floor()
    };
    let mut solid = vec![false; nx * ny * nz];
    for k in 0..nz {
        for j in 0..ny {
            for i in 0..nx {
                let c = cell_center(i, j, k);
                let off = cell * 0.002;
                // Distinct seeds for y and z so the offset is never along a
                // symmetry that could keep the point glued to a mesh edge.
                let cy = c[1] + (jitter(i, j + 1, k + 7) - 0.5) * off;
                let cz = c[2] + (jitter(i + 5, j + 3, k + 1) - 0.5) * off;
                let mut crossings = 0usize;
                for &ti in &buckets[k * ny + j] {
                    let t = triangles[ti];
                    let (n, d) = plane[ti];
                    let n_len2 = dot(n, n);
                    if n_len2 < 1e-18 || n[0].abs() < 1e-12 {
                        continue; // Degenerate, or the ray is parallel to the plane.
                    }
                    // Ray p(t) = (cx + t, cy, cz); solve n·p = d for t.
                    let t_hit = (d - n[1] * cy - n[2] * cz) / n[0] - c[0];
                    if t_hit > 1e-9
                        && inside_yz(cy, cz, &vertices[t[0]], &vertices[t[1]], &vertices[t[2]])
                    {
                        crossings += 1;
                    }
                }
                solid[idx(i, j, k)] = crossings % 2 == 1;
            }
        }
    }

    // Extract the exposed faces as flat-plate panels. Each face is a unit
    // square of area `cell²`, normal pointing outward, centred on the face.
    const FACES: [([i64; 3], [f64; 3]); 6] = [
        ([1, 0, 0], [1.0, 0.0, 0.0]),
        ([-1, 0, 0], [-1.0, 0.0, 0.0]),
        ([0, 1, 0], [0.0, 1.0, 0.0]),
        ([0, -1, 0], [0.0, -1.0, 0.0]),
        ([0, 0, 1], [0.0, 0.0, 1.0]),
        ([0, 0, -1], [0.0, 0.0, -1.0]),
    ];

    let mut panels = Vec::new();
    for k in 0..nz {
        for j in 0..ny {
            for i in 0..nx {
                if !solid[idx(i, j, k)] {
                    continue;
                }
                let c = cell_center(i, j, k);
                for (dir, normal) in FACES {
                    let ni = i as i64 + dir[0];
                    let nj = j as i64 + dir[1];
                    let nk = k as i64 + dir[2];
                    let exposed = ni < 0
                        || nj < 0
                        || nk < 0
                        || ni >= nx as i64
                        || nj >= ny as i64
                        || nk >= nz as i64
                        || !solid[idx(ni as usize, nj as usize, nk as usize)];
                    if !exposed {
                        continue;
                    }
                    // Centre of pressure = face centre, offset by half a cell
                    // along the outward normal from the solid cell's centre.
                    let cp = [
                        c[0] + dir[0] as f64 * 0.5 * cell,
                        c[1] + dir[1] as f64 * 0.5 * cell,
                        c[2] + dir[2] as f64 * 0.5 * cell,
                    ];
                    panels.push(CollisionPanel {
                        area: cell * cell,
                        normal,
                        cp,
                        cd: PANEL_CD,
                    });
                }
            }
        }
    }

    panels
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A unit box centred at the origin, solid, should yield roughly six panels
    /// (one per exposed face), with the +X and -X faces showing the two frontal
    /// plates.
    #[test]
    fn box_yields_shell_of_panels() {
        // Build a watertight box: 12 triangles for the six faces.
        let mut vertices = Vec::new();
        let mut triangles = Vec::new();
        let (x0, x1) = (-0.5, 0.5);
        let (y0, y1) = (-0.5, 0.5);
        let (z0, z1) = (-0.5, 0.5);
        let corners = [
            [x0, y0, z0], [x1, y0, z0], [x1, y1, z0], [x0, y1, z0],
            [x0, y0, z1], [x1, y0, z1], [x1, y1, z1], [x0, y1, z1],
        ];
        let faces: [[usize; 4]; 6] = [
            [0, 1, 2, 3], // -Z
            [4, 6, 5, 7], // +Z
            [0, 5, 1, 4],  // -Y ... (ordering only matters for area, not +/-
            [3, 2, 6, 7], // +Y
            [0, 4, 7, 3], // -X
            [1, 5, 6, 2], // +X
        ];
        for f in faces {
            vertices.push(corners[f[0]]);
            vertices.push(corners[f[1]]);
            vertices.push(corners[f[2]]);
            let base = (triangles.len() * 3) as usize;
            triangles.push([base, base + 1, base + 2]);
            vertices.push(corners[f[0]]);
            vertices.push(corners[f[2]]);
            vertices.push(corners[f[3]]);
            let base = (triangles.len() * 3) as usize;
            triangles.push([base, base + 1, base + 2]);
        }

        let panels = voxelize_panels(&vertices, &triangles, 8);
        assert!(!panels.is_empty(), "should have produced panels");
        // Volume ≈ 1 m³ at res 8 => ~8³=512 cells, exposed faces should be on
        // the order of a few hundred (each 0.125² = 0.0156 m²).
        let total_area: f64 = panels.iter().map(|p| p.area).sum();
        let shell = 6.0 * (8.0 * 8.0) * (0.125 * 0.125);
        assert!(total_area > shell * 0.5 && total_area < shell * 1.2, "area {total_area} vs ~{shell}");

        // Every panel must lie on the box shell (|coord| ≈ 0.5 ± cell).
        for p in &panels {
            let maxcoord = p.cp[0].abs().max(p.cp[1].abs()).max(p.cp[2].abs());
            assert!(maxcoord >= 0.4 && maxcoord <= 0.7, "cp {p:?} not on the hull");
        }
    }

    #[test]
    fn empty_mesh_yields_no_panels() {
        assert!(voxelize_panels(&[], &[], 8).is_empty());
    }

}