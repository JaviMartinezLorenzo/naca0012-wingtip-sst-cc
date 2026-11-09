"""
Build the wing STL for snappyHexMesh directly from the analytic geometry.

Same geometry as generate_wing.py (see notes.md):
  - NACA 0012 (same thickness equation), chord 48 in
  - blunt trailing edge 0.100 in thick (profile truncated where the half
    thickness drops to 0.05 in, then flat to x/c = 1)
  - constant-chord section from z = 0 (root, on the symmetry plane) to
    z = 33.12 in
  - rounded tip: upper profile revolved 180 deg about the chord line
  - 10 deg angle of attack (nose up) about the quarter chord, quarter chord
    moved to the origin

Why not use the gmsh STL directly
---------------------------------
The gmsh export (lc = 1 in on the body) is too coarse for a wall-resolved
mesh:
  - on the rounded tip, neighbouring facets meet at up to 38 deg (median 6,
    p90 14 deg). snappy follows these kinks, the extrusion directions of
    25 thin prism layers cross, and snappy drops the layers. In a test mesh,
    64 % of tip faces got no layers, exactly where the vortex forms;
  - the facets miss the true surface by up to 0.7 mm on the main wing and
    1.4 mm on the tip, which is large next to 1.6 mm cells and a 0.01 mm
    first layer;
  - the tip trailing edge has an open 2.5 mm half-disc, and the tip/wing
    seam is non-conforming.

This script tessellates the same surfaces finely and conformingly:
  - chordwise: cosine spacing (fine at LE and TE), max ~6 mm mid-chord
  - spanwise (straight section): uniform, ~28 mm (the surface is exactly
    straight in z, so this costs no accuracy)
  - tip: 1.5 deg circumferential steps (facet kink <= 1.5 deg)
Deviation from the analytic surface is < 0.02 mm everywhere.
Written as binary STL (~5 MB; OpenFOAM reads it directly).
  - blunt TE face and tip TE half-disc closed and conforming
The root (z = 0) is left open, as in the original; the symmetry plane closes
it.

Usage
-----
    python geometry/build_wing_stl.py OpenFOAM/constant/triSurface/wing.stl
"""

import math
import sys

import numpy as np

INCH = 0.0254
CHORD = 48.0 * INCH
TE_HALF = 0.05 * INCH            # half of the 0.100 in blunt trailing edge
Z_STRAIGHT = 33.12 * INCH        # end of the constant-chord section
AOA = math.radians(10.0)
QC = 0.25 * CHORD

N_CHORD = 301                    # chordwise points per surface (cosine spaced)
N_SPAN = 30                      # spanwise intervals on the straight section (~28 mm)
N_THETA = 120                    # circumferential intervals on the tip (1.5 deg)


def half_thickness(xc):
    """NACA 0012 half thickness / c, same coefficients as generate_wing.py."""
    return 0.594689181 * (
        0.298222773 * np.sqrt(xc)
        - 0.127125232 * xc
        - 0.357907906 * xc**2
        + 0.291984971 * xc**3
        - 0.105174606 * xc**4
    )


def profile():
    """Chordwise stations (x, y_half) in metres, unrotated frame, LE at x=0.

    Half thickness follows the NACA equation until it drops to TE_HALF, then
    stays flat to x = c (blunt TE), as in generate_wing.py.
    """
    te = TE_HALF / CHORD
    # x/c where the NACA thickness reaches the TE half thickness
    lo, hi = 0.9, 1.0
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        if half_thickness(mid) > te:
            lo = mid
        else:
            hi = mid
    x_cut = 0.5 * (lo + hi)

    t = np.linspace(0.0, np.pi, N_CHORD)
    xc = 0.5 * (1.0 - np.cos(t))
    xc = xc[xc < x_cut - 1e-6]
    xc = np.concatenate([xc, [x_cut, 1.0]])
    yc = np.concatenate([half_thickness(xc[:-2]), [te, te]])
    return xc * CHORD, yc * CHORD


def to_wing_frame(p):
    """Rotate nose-up by AOA about the quarter chord, then put QC at origin."""
    x, y, z = p[..., 0] - QC, p[..., 1], p[..., 2]
    c, s = math.cos(-AOA), math.sin(-AOA)
    return np.stack([x * c - y * s, x * s + y * c, z], axis=-1)


class Mesh:
    def __init__(self):
        self.pts, self.idx, self.tris = [], {}, []

    def node(self, p):
        key = tuple(np.round(p, 12))
        if key not in self.idx:
            self.idx[key] = len(self.pts)
            self.pts.append(np.asarray(p, float))
        return self.idx[key]

    def tri(self, a, b, c):
        if len({a, b, c}) == 3:
            self.tris.append((a, b, c))

    def quad(self, a, b, c, d):
        self.tri(a, b, c)
        self.tri(a, c, d)


def build():
    x, yh = profile()
    nx = len(x)
    zs = np.linspace(0.0, Z_STRAIGHT, N_SPAN + 1)
    th = np.linspace(0.0, np.pi, N_THETA + 1)
    m = Mesh()

    def P(i, sign, k):          # straight-section point: station i, upper(+1)/lower(-1), span k
        return m.node((x[i], sign * yh[i], zs[k]))

    def T(i, j):                # tip point: station i, angle j (0 = upper, pi = lower)
        r = yh[i]
        return m.node((x[i], r * math.cos(th[j]), Z_STRAIGHT + r * math.sin(th[j])))

    # straight section, upper and lower surfaces
    for k in range(N_SPAN):
        for i in range(nx - 1):
            m.quad(P(i, 1, k), P(i, 1, k + 1), P(i + 1, 1, k + 1), P(i + 1, 1, k))
            m.quad(P(i, -1, k), P(i + 1, -1, k), P(i + 1, -1, k + 1), P(i, -1, k + 1))

    # blunt TE face of the straight section (mid node at y = 0 so the tip
    # half-disc fan below joins it conformingly)
    xe = x[-1]
    for k in range(N_SPAN):
        top = [m.node((xe, s, zs[k])) for s in (TE_HALF, 0.0, -TE_HALF)]
        bot = [m.node((xe, s, zs[k + 1])) for s in (TE_HALF, 0.0, -TE_HALF)]
        for a in range(2):
            m.quad(top[a], top[a + 1], bot[a + 1], bot[a])

    # rounded tip (j = 0 and j = N_THETA coincide with the straight-section
    # upper and lower edges at z = Z_STRAIGHT, so the seam is conforming)
    for j in range(N_THETA):
        for i in range(nx - 1):
            m.quad(T(i, j), T(i, j + 1), T(i + 1, j + 1), T(i + 1, j))

    # tip TE half-disc: fan from the TE mid point at z = Z_STRAIGHT
    c = m.node((xe, 0.0, Z_STRAIGHT))
    for j in range(N_THETA):
        m.tri(c, T(nx - 1, j + 1), T(nx - 1, j))

    pts = to_wing_frame(np.array(m.pts))
    tris = np.array(m.tris)

    # orient all normals outward: away from the chord line (straight section)
    # or from the tip axis (tip); TE faces point downstream.
    tri_pts = pts[tris]
    n = np.cross(tri_pts[:, 1] - tri_pts[:, 0], tri_pts[:, 2] - tri_pts[:, 0])
    cen = tri_pts.mean(axis=1)
    chord_dir = np.array([math.cos(-AOA), math.sin(-AOA), 0.0])
    axis_pt = cen.copy()
    s = cen @ chord_dir
    axis_pt = np.outer(s, chord_dir)
    axis_pt[:, 2] = np.minimum(cen[:, 2], Z_STRAIGHT)
    out = cen - axis_pt
    te_face = (s > (CHORD - QC) - 1e-6)
    out[te_face] = chord_dir
    flip = np.einsum("ij,ij->i", n, out) < 0
    tris[flip] = tris[flip][:, [0, 2, 1]]
    return pts, tris


def write_stl(path, pts, tris):
    """Binary STL (80-byte header, uint32 count, 50 bytes per triangle)."""
    v = pts[tris]
    n = np.cross(v[:, 1] - v[:, 0], v[:, 2] - v[:, 0])
    n /= np.linalg.norm(n, axis=1)[:, None]
    rec = np.zeros(len(tris), dtype=[("n", "<f4", 3), ("v", "<f4", (3, 3)), ("attr", "<u2")])
    rec["n"], rec["v"] = n, v
    with open(path, "wb") as f:
        f.write(b"NACA 0012 wing, built by geometry/build_wing_stl.py".ljust(80, b" "))
        f.write(np.uint32(len(tris)).tobytes())
        f.write(rec.tobytes())


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    pts, tris = build()
    write_stl(sys.argv[1], pts, tris)
    lo, hi = pts.min(axis=0), pts.max(axis=0)
    print(f"wrote {sys.argv[1]}: {len(tris)} triangles, {len(pts)} points")
    print(f"bounding box ({lo[0]:.6f} {lo[1]:.6f} {lo[2]:.6f}) ({hi[0]:.6f} {hi[1]:.6f} {hi[2]:.6f})")
