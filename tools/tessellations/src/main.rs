use std::f64::consts::PI;

/// 3D point / vector
#[derive(Debug, Clone, Copy)]
struct Vec3 {
    x: f64,
    y: f64,
    z: f64,
}

impl Vec3 {
    fn new(x: f64, y: f64, z: f64) -> Self {
        Self { x, y, z }
    }

    fn dot(self, other: Vec3) -> f64 {
        self.x * other.x + self.y * other.y + self.z * other.z
    }

    fn cross(self, other: Vec3) -> Vec3 {
        Vec3::new(
            self.y * other.z - self.z * other.y,
            self.z * other.x - self.x * other.z,
            self.x * other.y - self.y * other.x,
        )
    }

    fn norm(self) -> f64 {
        self.dot(self).sqrt()
    }

    fn normalize(self) -> Vec3 {
        let n = self.norm();
        if n == 0.0 {
            self
        } else {
            Vec3::new(self.x / n, self.y / n, self.z / n)
        }
    }

    fn scale(self, s: f64) -> Vec3 {
        Vec3::new(self.x * s, self.y * s, self.z * s)
    }

    fn add(self, other: Vec3) -> Vec3 {
        Vec3::new(self.x + other.x, self.y + other.y, self.z + other.z)
    }

    fn sub(self, other: Vec3) -> Vec3 {
        Vec3::new(self.x - other.x, self.y - other.y, self.z - other.z)
    }
}

/// Triangle represented by indices into vertex list
#[derive(Debug, Clone)]
struct Tri {
    a: usize,
    b: usize,
    c: usize,
}

/// Build the 12 vertices and 20 faces of a regular icosahedron (unit sphere)
fn build_icosahedron() -> (Vec<Vec3>, Vec<Tri>) {
    // golden ratio
    let phi = (1.0 + 5.0f64.sqrt()) / 2.0;
    let inv = 1.0 / (1.0 + phi * phi).sqrt(); // normalize scale later

    let mut verts = vec![
        Vec3::new(-1.0,  phi, 0.0),
        Vec3::new( 1.0,  phi, 0.0),
        Vec3::new(-1.0, -phi, 0.0),
        Vec3::new( 1.0, -phi, 0.0),

        Vec3::new(0.0, -1.0,  phi),
        Vec3::new(0.0,  1.0,  phi),
        Vec3::new(0.0, -1.0, -phi),
        Vec3::new(0.0,  1.0, -phi),

        Vec3::new( phi, 0.0, -1.0),
        Vec3::new( phi, 0.0,  1.0),
        Vec3::new(-phi, 0.0, -1.0),
        Vec3::new(-phi, 0.0,  1.0),
    ];

    // normalize all vertices onto unit sphere
    for v in &mut verts {
        *v = v.normalize();
    }

    let faces = vec![
        Tri { a: 0, b: 11, c: 5 },
        Tri { a: 0, b: 5, c: 1 },
        Tri { a: 0, b: 1, c: 7 },
        Tri { a: 0, b: 7, c: 10 },
        Tri { a: 0, b: 10, c: 11 },
        Tri { a: 1, b: 5, c: 9 },
        Tri { a: 5, b: 11, c: 4 },
        Tri { a: 11, b: 10, c: 2 },
        Tri { a: 10, b: 7, c: 6 },
        Tri { a: 7, b: 1, c: 8 },
        Tri { a: 3, b: 9, c: 4 },
        Tri { a: 3, b: 4, c: 2 },
        Tri { a: 3, b: 2, c: 6 },
        Tri { a: 3, b: 6, c: 8 },
        Tri { a: 3, b: 8, c: 9 },
        Tri { a: 4, b: 9, c: 5 },
        Tri { a: 2, b: 4, c: 11 },
        Tri { a: 6, b: 2, c: 10 },
        Tri { a: 8, b: 6, c: 7 },
        Tri { a: 9, b: 8, c: 1 },
    ];

    (verts, faces)
}

/// Linearly interpolate between two points and project onto unit sphere
fn midpoint_on_sphere(a: Vec3, b: Vec3) -> Vec3 {
    a.add(b).normalize()
}

/// Subdivide each triangle into 4 smaller triangles (frequency 1 subdivision).
/// Repeat recursion times to get higher frequency.
fn subdivide(
    verts: &mut Vec<Vec3>,
    faces: &mut Vec<Tri>,
    recursion: usize,
) {
    // We'll use a cache to avoid duplicating midpoints for shared edges.
    use std::collections::HashMap;
    for _level in 0..recursion {
        let mut new_faces = Vec::new();
        let mut edge_cache: HashMap<(usize, usize), usize> = HashMap::new();

        let get_mid = |edge_cache: &mut HashMap<(usize, usize), usize>,
                       verts: &mut Vec<Vec3>,
                       i: usize,
                       j: usize| -> usize {
            let key = if i < j { (i, j) } else { (j, i) };
            if let Some(&idx) = edge_cache.get(&key) {
                return idx;
            }
            let v = midpoint_on_sphere(verts[key.0], verts[key.1]);
            verts.push(v);
            let idx = verts.len() - 1;
            edge_cache.insert(key, idx);
            idx
        };

        for f in faces.iter() {
            let a = f.a;
            let b = f.b;
            let c = f.c;

            let ab = get_mid(&mut edge_cache, verts, a, b);
            let bc = get_mid(&mut edge_cache, verts, b, c);
            let ca = get_mid(&mut edge_cache, verts, c, a);

            new_faces.push(Tri { a, b: ab, c: ca });
            new_faces.push(Tri { a: b, b: bc, c: ab });
            new_faces.push(Tri { a: c, b: ca, c: bc });
            new_faces.push(Tri { a: ab, b: bc, c: ca });
        }

        *faces = new_faces;
    }
}

/// Convert lat/lon (degrees) to unit sphere Vec3 (x,y,z)
fn latlon_to_xyz(lat_deg: f64, lon_deg: f64) -> Vec3 {
    let lat = lat_deg.to_radians();
    let lon = lon_deg.to_radians();
    let clat = lat.cos();
    Vec3::new(lon.cos() * clat, lon.sin() * clat, lat.sin()).normalize()
}

/// Compute barycentric coordinates of point p w.r.t triangle (on plane)
/// We will compute barycentric using vector math in 3D with triangle projected onto tangent plane.
/// Returns (u, v, w) corresponding to weights for (a, b, c). If triangle is degenerate, returns None.
fn barycentric_on_sphere(p: Vec3, a: Vec3, b: Vec3, c: Vec3) -> Option<(f64, f64, f64)> {
    // Project to plane using one vertex as origin
    let v0 = b.sub(a);
    let v1 = c.sub(a);
    let v2 = p.sub(a);

    let d00 = v0.dot(v0);
    let d01 = v0.dot(v1);
    let d11 = v1.dot(v1);
    let d20 = v2.dot(v0);
    let d21 = v2.dot(v1);

    let denom = d00 * d11 - d01 * d01;
    if denom.abs() < 1e-12 {
        return None;
    }

    let v = (d11 * d20 - d01 * d21) / denom;
    let w = (d00 * d21 - d01 * d20) / denom;
    let u = 1.0 - v - w;
    Some((u, v, w))
}

/// Find the closest triangle to a point on the sphere by searching all triangles.
/// Returns (triangle_index, barycentric coords (u,v,w))
fn find_triangle_for_point(
    p: Vec3,
    verts: &Vec<Vec3>,
    faces: &Vec<Tri>,
) -> Option<(usize, (f64, f64, f64))> {
    // We'll choose the triangle whose spherical distance from p to triangle plane is smallest
    // but more robust is using barycentric and checking if p projects inside triangle.
    // We'll check projection onto triangle plane; if inside, accept; otherwise pick the triangle
    // with minimal angular distance to triangle centroid.
    let mut best: Option<(usize, f64, (f64, f64, f64))> = None;

    for (i, tri) in faces.iter().enumerate() {
        let a = verts[tri.a];
        let b = verts[tri.b];
        let c = verts[tri.c];

        // compute normal and signed distance (angle) from p to triangle plane (through origin)
        let normal = b.sub(a).cross(c.sub(a)).normalize();
        // For plane through vertices (not necessarily through origin), but since vertices are on unit sphere
        // and plane doesn't pass through origin, we'll instead check barycentric after projecting p
        // onto plane tangent at a-> use barycentric_on_sphere with 3D vectors (works approx).
        if let Some((u, v, w)) = barycentric_on_sphere(p, a, b, c) {
            // If point is inside triangle in plane projection (allow some epsilon)
            if u >= -1e-6 && v >= -1e-6 && w >= -1e-6 {
                // closer is better; choose by absolute angular deviation between p and triangle centroid
                let centroid = Vec3::new((a.x + b.x + c.x) / 3.0, (a.y + b.y + c.y) / 3.0, (a.z + b.z + c.z) / 3.0).normalize();
                let ang = p.dot(centroid).acos(); // angular separation
                match &best {
                    Some((_, best_ang, _)) if ang >= *best_ang => {}
                    _ => best = Some((i, ang, (u, v, w))),
                }
            } else {
                // not inside; compute distance to triangle centroid as fallback
                let centroid = Vec3::new((a.x + b.x + c.x) / 3.0, (a.y + b.y + c.y) / 3.0, (a.z + b.z + c.z) / 3.0).normalize();
                let ang = p.dot(centroid).acos();
                match &best {
                    Some((_, best_ang, _)) if ang >= *best_ang => {}
                    _ => best = Some((i, ang, (u, v, w))),
                }
            }
        }
    }

    best.map(|(i, _ang, bary)| (i, bary))
}

/// Convenience: build tessellation with given recursion (frequency).
/// recursion = 0 -> 20 faces (icosahedron). recursion = 1 -> 80 faces, etc.
fn build_tessellation(recursion: usize) -> (Vec<Vec3>, Vec<Tri>) {
    let (mut verts, mut faces) = build_icosahedron();
    subdivide(&mut verts, &mut faces, recursion);
    (verts, faces)
}

fn triangle_vertices(verts: &Vec<Vec3>, tri: &Tri) -> (Vec3, Vec3, Vec3) {
    (verts[tri.a], verts[tri.b], verts[tri.c])
}

/// Example usage
fn main() {
    // Choose subdivision level: 0..5 reasonable. Level 0 = 20 faces, level 1 = 80, level 2 = 320, level 3 = 1280, ...
    let recursion = 2;
    let (verts, faces) = build_tessellation(recursion);
    println!("Generated tessellation: {} vertices, {} triangles", verts.len(), faces.len());

    // Example coordinate: New York City ~ (lat, lon)
    let lat = 40.7128;
    let lon = -74.0060;
    let p = latlon_to_xyz(lat, lon);

    if let Some((tri_idx, (u, v, w))) = find_triangle_for_point(p, &verts, &faces) {
        println!("Point mapped to triangle index {}", tri_idx);
        println!("Barycentric coordinates (u,v,w) = ({:.6}, {:.6}, {:.6})", u, v, w);
        let tri = &faces[tri_idx];
        let (a, b, c) = triangle_vertices(&verts, tri);
        println!("Triangle vertex unit coordinates:");
        println!("A: ({:.6}, {:.6}, {:.6})", a.x, a.y, a.z);
        println!("B: ({:.6}, {:.6}, {:.6})", b.x, b.y, b.z);
        println!("C: ({:.6}, {:.6}, {:.6})", c.x, c.y, c.z);
    } else {
        println!("No triangle found (unexpected).");
    }
}