# Tesselations

Self-contained Rust implementation that builds an icosahedron-based tessellation (geodesic-like subdivision) of the sphere and provides functions to:

- Generate the base icosahedron vertices and triangular faces.
- Subdivide each triangular face to a given recursion level (frequency) to produce a nearly-uniform triangular mesh on the unit sphere.
- Map geographic coordinates (latitude, longitude) to the nearest triangle (cell) in the tessellation and return the triangle index and barycentric coordinates.
- Convert triangle indices back to triangle vertex coordinates on the sphere.

For now this code focuses on clarity and correctness rather than extreme performance. It uses spherical normalization so all vertices lie on the unit sphere. For lookups it uses a simple linear search (O(N)) over triangles; in the future will be replaced with spatial indexing for production use with:
- k-d tree, 
- bounding volume hierarchy, or 
- HEALPix-like scheme

Notes and possible improvements

- Subdivision method: this code uses midpoint subdivision (each triangle → 4 triangles) and reprojects midpoints onto the unit sphere. This creates a class-I geodesic-like triangulation; it is not exactly equal-area but is commonly used and straightforward.
- Performance: face lookup is linear. For large subdivisions (recursion >= 5 produces tens or hundreds of thousands of triangles) build an acceleration structure:
- Map each triangle to a bounding cap (centroid + angular radius), store in a sphere-aware spatial index.
- Use an R-tree on triangle centroids, or a precomputed hierarchical mesh (quadtrees on subdivided faces).
- Alternative tessellations: HEALPix or S2 geometry provide well-tested indexing and equal-area cells and fast lookups.
- Mapping result: I return the triangle index and barycentric coordinates. You can compute e.g. the exact closest point on the triangle by combining vertices with barycentrics and re-normalizing to unit sphere.
- Edge cases: Points exactly on triangle boundaries may map arbitrarily depending on iteration order. You can break ties consistently by checking edges.
