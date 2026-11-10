"""
pv_vortex.py: images and vortex-core metrics for one run (ParaView pvpython).

Run through pv_vortex.sh (finds pvpython, converts WSL paths), or directly:
    pvpython --force-offscreen-rendering pv_vortex.py <case dir or .foam> <label>

Writes scripts/results/<label>/:
    images/        wing Cp and y+, Q-criterion isosurface, cross-flow planes at
                   the measured stations, streamlines, mesh through the core
    core_metrics.csv     per measured station: core position, Cp_min, axial
                         velocity, peak swirl, core radius, cells across the
                         core, k and nu_t at the core
    swirl_profiles.csv   azimuthally averaged swirl velocity vs radius
    yplus_summary.csv    area-weighted y+ and % of area above 1/3/5/10, per
                         wing region
    run_info.txt         case, time, settings

All colour ranges and cameras are fixed, so images from different runs
(nutk vs Spalding, SST vs SST-CC, coarse/medium/fine) compare directly.
Compare runs with compare_runs.py.

Core detection per station (works over the wing and in the wake):
  1. Core = minimum Cp_static among cells more than WALL_CLEAR from the wing
     where Q > 0 (rotation dominates strain). Q > 0 excludes the suction
     regions of the potential flow (Q < 0 there), so the low-pressure peak on
     the surface is not mistaken for the vortex. Using min Cp (rather than
     max Q) also avoids locking onto small secondary vortices at the tip, and
     matches how the experimental core is defined (min Cp_static per plane).
  2. Swirl v_theta(r) azimuthally averaged around the core. The first peak
     moving outward from the centre gives v_theta,max and the core radius r_c
     (over the wing, the flow around the tip makes v_theta grow again further
     out, so the global maximum would be wrong).

Only numpy and the ParaView/VTK modules are used (pvpython has no pandas).
"""

import argparse
import datetime
import math
import os
import sys

import numpy as np
from paraview import servermanager as sm
from paraview.simple import (Calculator, CellCenters, CellSize, Clip,
                             ColorBy, Contour, CreateView, ExtractBlock,
                             GetColorTransferFunction, GetSources, Glyph,
                             Gradient, Hide, MergeBlocks, OpenFOAMReader,
                             SaveScreenshot, Show, Slice, StreamTracer,
                             Tube, XMLPolyDataReader)
from vtk.numpy_interface import dataset_adapter as dsa
from vtkmodules.util.numpy_support import numpy_to_vtk, vtk_to_numpy
from vtkmodules.vtkFiltersCore import vtkImplicitPolyDataDistance, vtkThreshold
from vtkmodules.vtkFiltersGeometry import vtkGeometryFilter
from vtkmodules.vtkIOXML import vtkXMLPolyDataWriter

# ---------------------------------------------------------------------------
# Case constants (wing-fixed frame, metres)
# ---------------------------------------------------------------------------
U_INF = 51.816
NU = 1.37335e-5
CHORD = 1.2192
AOA = math.radians(10.0)
Z_TIP_SEAM = 33.12 * 0.0254        # start of the rounded tip
WALL_CLEAR = 0.005                 # ignore cells closer than this to the wing

# Search window for the vortex in each cross-flow plane
WIN_Y = (-0.25, 0.10)
WIN_Z = (0.60, 1.00)

# Fixed colour ranges
CP_RANGE = (-3.0, 0.5)
UX_RANGE = (0.6, 1.8)              # u / U_inf
YPLUS_RANGE = (0.0, 5.0)

# Measured stations (fallback if the experimental file cannot be read)
FALLBACK_STATIONS = [0.1801, 0.4202, 0.5403, 0.6604, 0.7624, 0.8885, 0.9069,
                     1.0538, 1.2007, 1.4524, 1.7272]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def log(msg):
    print(f"[pv_vortex {datetime.datetime.now():%H:%M:%S}] {msg}", flush=True)


def measured_stations(repo_root):
    """x of the experimental cross-flow planes (RETAKALL.DAT), CFD frame."""
    path = os.path.join(repo_root, "Wingtip_expdata", "RETAKALL.DAT")
    try:
        x_raw = np.genfromtxt(path, usecols=(0,), invalid_raise=False)
        x_raw = np.unique(x_raw[np.isfinite(x_raw)])
        xtrans = 0.75 * 48 * math.cos(AOA) + 0.25
        x = (x_raw + xtrans) * 0.0254
        return [float(v) for v in x if v > 0.0]       # skip the inflow plane
    except Exception as exc:                            # pragma: no cover
        log(f"could not read {path} ({exc}); using built-in station list")
        return FALLBACK_STATIONS


def fetch_np(source):
    """Fetch a source (merged to one dataset) as a numpy-wrapped object."""
    return dsa.WrapDataObject(sm.Fetch(MergeBlocks(Input=source)))


def wall_distance(points, wing_poly):
    """Unsigned distance from points (N,3) to the wing surface."""
    imp = vtkImplicitPolyDataDistance()
    imp.SetInput(wing_poly)
    pts = numpy_to_vtk(np.ascontiguousarray(points, dtype=np.float64), deep=True)
    out = numpy_to_vtk(np.zeros(len(points)), deep=True)
    imp.FunctionValue(pts, out)
    return np.abs(vtk_to_numpy(out).copy())


def first_peak(r, v):
    """First local maximum of |v(r)| moving outward (3-bin smoothing)."""
    if len(v) < 3:
        return np.nan, np.nan
    a = np.abs(np.asarray(v))
    s = np.convolve(a, np.ones(3) / 3.0, mode="same")
    s[0], s[-1] = a[0], a[-1]
    for j in range(1, len(s) - 1):
        if s[j] >= s[j - 1] and s[j] > s[j + 1] and s[j] >= 0.5 * s.max():
            return float(a[j]), float(r[j])
    j = int(np.argmax(a))
    return float(a[j]), float(r[j])


def wing_regions(centres):
    """Label wing faces: upper / lower / tip, and chordwise x/c."""
    x, y, z = centres.T
    xc = (x * math.cos(AOA) - y * math.sin(AOA) + 0.25 * CHORD) / CHORD
    yn = x * math.sin(AOA) + y * math.cos(AOA)
    side = np.where(z > Z_TIP_SEAM, "tip", np.where(yn >= 0.0, "upper", "lower"))
    return side, xc


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------

class Renderer:
    def __init__(self, outdir, size=(1600, 1000)):
        self.outdir = outdir
        self.view = CreateView("RenderView")
        self.view.ViewSize = list(size)
        self.view.OrientationAxesVisibility = 1
        try:
            self.view.UseColorPaletteForBackground = 0
        except AttributeError:
            pass
        self.view.Background = [1.0, 1.0, 1.0]
        os.makedirs(outdir, exist_ok=True)

    def lut(self, array, rng, preset="Cool to Warm", above=None):
        lut = GetColorTransferFunction(array)
        lut.ApplyPreset(preset, True)
        lut.AutomaticRescaleRangeMode = "Never"
        lut.RescaleTransferFunction(*rng)
        if above is not None:
            lut.UseAboveRangeColor = 1
            lut.AboveRangeColor = above
        return lut

    def show(self, src, array=None, assoc="POINTS", rng=None, preset="Cool to Warm",
             above=None, rep="Surface", solid=(0.75, 0.75, 0.75), opacity=1.0,
             bar_title=None):
        d = Show(src, self.view)
        d.Representation = rep
        d.Opacity = opacity
        if array:
            ColorBy(d, (assoc, array))
            self.lut(array, rng, preset, above)
            d.SetScalarBarVisibility(self.view, True)
            from paraview.simple import GetScalarBar
            sb = GetScalarBar(GetColorTransferFunction(array), self.view)
            sb.Title = bar_title or array
            sb.ComponentTitle = ""
            sb.TitleColor = [0, 0, 0]
            sb.LabelColor = [0, 0, 0]
            sb.WindowLocation = "Any Location"
            sb.Position = [0.89, 0.08]
            sb.ScalarBarLength = 0.5
        else:
            try:
                ColorBy(d, None)
            except RuntimeError:          # empty dataset (nothing to colour)
                pass
            d.AmbientColor = list(solid)
            d.DiffuseColor = list(solid)
        return d

    def clear(self):
        for src in GetSources().values():
            Hide(src, self.view)

    def camera(self, focal, direction, up, parallel=None, distance=4.0):
        f = np.array(focal, float)
        dvec = np.array(direction, float)
        dvec /= np.linalg.norm(dvec)
        self.view.CameraFocalPoint = list(f)
        self.view.CameraPosition = list(f - distance * dvec)
        self.view.CameraViewUp = list(up)
        if parallel:
            self.view.CameraParallelProjection = 1
            self.view.CameraParallelScale = parallel
        else:
            self.view.CameraParallelProjection = 0
            self.view.CameraViewAngle = 30

    def title(self, text):
        from paraview.simple import Text
        t = Text(Text=text)
        d = Show(t, self.view)
        d.WindowLocation = "Upper Left Corner"
        d.FontSize = 26
        d.Color = [0, 0, 0]
        return t

    def save(self, name, title=None):
        if title:
            self.title(title)
        path = os.path.join(self.outdir, name)
        self.view.StillRender()
        SaveScreenshot(path, self.view, ImageResolution=list(self.view.ViewSize),
                       TransparentBackground=0)
        log(f"  image {name}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("case", help="case directory or .foam file")
    ap.add_argument("label", help="run label, e.g. nutk_medium_500")
    ap.add_argument("--time", type=float, default=None, help="time to load (default: latest)")
    ap.add_argument("--decomposed", action="store_true", help="read processor* directories")
    ap.add_argument("--out", default=None, help="output directory (default: scripts/results/<label>)")
    ap.add_argument("--q-star", type=float, default=50.0,
                    help="Q isosurface level as Q* = Q c^2 / U_inf^2 (default 50)")
    ap.add_argument("--no-images", action="store_true", help="metrics only")
    args = ap.parse_args()

    here = os.path.dirname(os.path.abspath(__file__))
    repo = os.path.abspath(os.path.join(here, "..", ".."))
    outdir = args.out or os.path.join(repo, "scripts", "results", args.label)
    imgdir = os.path.join(outdir, "images")
    os.makedirs(imgdir, exist_ok=True)

    case = args.case
    if os.path.isdir(case):
        foam = os.path.join(case, "case.foam")
        if not os.path.exists(foam):
            open(foam, "w").close()
        case = foam

    # ---- read --------------------------------------------------------------
    log(f"reading {case}")
    reader = OpenFOAMReader(FileName=case)
    reader.CaseType = "Decomposed Case" if args.decomposed else "Reconstructed Case"
    reader.MeshRegions = ["internalMesh", "patch/wing"]
    reader.Createcelltopointfiltereddata = 1
    reader.UpdatePipelineInformation()
    times = list(reader.TimestepValues) if reader.TimestepValues else [0.0]
    t = args.time if args.time is not None else max(times)
    available = list(reader.CellArrays.Available)
    reader.CellArrays = available
    log(f"time {t:g}; fields: {', '.join(available)}")

    internal = ExtractBlock(Input=reader, Selectors=["/Root/internalMesh"])
    wing = ExtractBlock(Input=reader, Selectors=["/Root/boundary/wing"])

    # Derived fields: Cp from p if missing, u/U_inf, Q from grad(U) if missing
    flow = internal
    if "CpStatic" not in available:
        flow = Calculator(Input=flow, ResultArrayName="CpStatic",
                          Function=f"p/(0.5*{U_INF}^2)")
    flow = Calculator(Input=flow, ResultArrayName="ux", Function=f"U_X/{U_INF}")
    if "Q" not in available:
        log("Q not in the case: computing it from grad(U)")
        flow = Gradient(Input=flow, ScalarArray=["POINTS", "U"])
        flow.ComputeQCriterion = 1
        flow.QCriterionArrayName = "Q"
    flow.UpdatePipeline(t)
    wing.UpdatePipeline(t)

    wing_poly = sm.Fetch(MergeBlocks(Input=wing))
    geo = vtkGeometryFilter()
    geo.SetInputData(wing_poly)
    geo.Update()
    wing_poly = geo.GetOutput()

    stations = measured_stations(repo)
    log(f"{len(stations)} measured stations: {', '.join(f'{s:.3f}' for s in stations)}")

    # ---- wing y+ summary ------------------------------------------------------
    rows_y = []
    if "yPlus" in available:
        ws = CellSize(Input=wing, ComputeArea=1, ComputeLength=0, ComputeVolume=0,
                      ComputeVertexCount=0)
        wc = CellCenters(Input=ws)
        d = fetch_np(wc)
        yp = np.asarray(d.PointData["yPlus"])
        area = np.asarray(d.PointData["Area"])
        side, xc = wing_regions(np.asarray(d.Points))
        regions = [("wing", np.ones_like(yp, bool))]
        for s in ("upper", "lower", "tip"):
            regions.append((s, side == s))
            for lo, hi, nm in ((0.0, 0.05, "LE"), (0.05, 0.95, "mid"), (0.95, 1.01, "TE")):
                if s != "tip":
                    regions.append((f"{s}_{nm}", (side == s) & (xc >= lo) & (xc < hi)))
        for name, m in regions:
            if not m.any():
                continue
            a = area[m]
            row = [name, int(m.sum()), float(np.sum(yp[m] * a) / a.sum())]
            row += [float(100 * a[yp[m] > lim].sum() / a.sum()) for lim in (1, 3, 5, 10)]
            row.append(float(yp[m].max()))
            rows_y.append(row)
        with open(os.path.join(outdir, "yplus_summary.csv"), "w") as f:
            f.write("region,faces,yplus_area_avg,pct_area_gt1,pct_area_gt3,pct_area_gt5,pct_area_gt10,yplus_max\n")
            for r in rows_y:
                f.write(f"{r[0]},{r[1]},{r[2]:.3f},{r[3]:.2f},{r[4]:.2f},{r[5]:.2f},{r[6]:.2f},{r[7]:.1f}\n")
        log(f"wing y+: area-weighted {rows_y[0][2]:.2f}, {rows_y[0][5]:.2f} % of area above 5, max {rows_y[0][7]:.0f}")

    # ---- core metrics per station ---------------------------------------------
    metrics, profiles = [], []
    for x in stations:
        sl = Slice(Input=flow)
        sl.SliceType = "Plane"
        sl.SliceType.Origin = [x, 0.0, 0.0]
        sl.SliceType.Normal = [1.0, 0.0, 0.0]
        sl.Triangulatetheslice = 0
        cs = CellSize(Input=sl, ComputeArea=1, ComputeLength=0, ComputeVolume=0,
                      ComputeVertexCount=0)
        d = fetch_np(CellCenters(Input=cs))
        P = np.asarray(d.Points)
        if len(P) == 0:
            continue
        y, z = P[:, 1], P[:, 2]
        win = (y > WIN_Y[0]) & (y < WIN_Y[1]) & (z > WIN_Z[0]) & (z < WIN_Z[1])
        idx = np.where(win)[0]
        P = P[idx]
        cp = np.asarray(d.PointData["CpStatic"])[idx]
        U = np.asarray(d.PointData["U"])[idx]
        Q = np.asarray(d.PointData["Q"])[idx]
        area = np.asarray(d.PointData["Area"])[idx]
        k = np.asarray(d.PointData["k"])[idx] if "k" in d.PointData.keys() else np.full(len(idx), np.nan)
        nut = np.asarray(d.PointData["nut"])[idx] if "nut" in d.PointData.keys() else np.full(len(idx), np.nan)
        wd = wall_distance(P, wing_poly)

        cand = (wd > WALL_CLEAR) & (Q > 0.0)
        if not cand.any():
            continue
        ic = np.where(cand)[0][np.argmin(cp[cand])]
        yc, zc = P[ic, 1], P[ic, 2]

        # swirl profile: v_theta = ((y-yc) w - (z-zc) v) / r
        dy, dz = P[:, 1] - yc, P[:, 2] - zc
        r = np.hypot(dy, dz)
        ring = (r > 1e-9) & (r < 0.12) & (wd > 0.003)
        vt = (dy[ring] * U[ring, 2] - dz[ring] * U[ring, 1]) / r[ring]
        edges = np.arange(0.0, 0.1201, 0.002)
        which = np.digitize(r[ring], edges) - 1
        rb, vb = [], []
        for b in range(len(edges) - 1):
            m = which == b
            if m.sum() >= 3:
                rb.append(0.5 * (edges[b] + edges[b + 1]))
                vb.append(float(np.mean(vt[m])))
                profiles.append((x, rb[-1], vb[-1] / U_INF, int(m.sum())))
        vmax, rc = first_peak(rb, vb)
        h = math.sqrt(area[ic])
        metrics.append(dict(
            x=x, x_c=x / CHORD, over_wing=int(x < 0.9),
            y_core=yc, z_core=zc, y_c=yc / CHORD, z_c=zc / CHORD,
            cp_min=float(cp[ic]), u_core=float(U[ic, 0] / U_INF),
            vtheta_max=vmax / U_INF, r_c=rc, r_c_c=rc / CHORD,
            cell_size_core=h, cells_across_core=2 * rc / h if h > 0 else np.nan,
            k_core=float(k[ic] / U_INF**2), nut_core_over_nu=float(nut[ic] / NU),
            q_star_core=float(Q[ic] * CHORD**2 / U_INF**2),
            wall_dist_core=float(wd[ic])))
        log(f"  x={x:.3f}: core y={yc:+.4f} z={zc:.4f}  Cp_min={cp[ic]:+.2f}  "
            f"u={U[ic,0]/U_INF:.2f}  vtheta_max={vmax/U_INF:.2f}  r_c={rc*1e3:.0f} mm  "
            f"cells across={2*rc/h:.0f}")

    keys = ["x", "x_c", "over_wing", "y_core", "z_core", "y_c", "z_c", "cp_min", "u_core",
            "vtheta_max", "r_c", "r_c_c", "cell_size_core", "cells_across_core",
            "k_core", "nut_core_over_nu", "q_star_core", "wall_dist_core"]
    with open(os.path.join(outdir, "core_metrics.csv"), "w") as f:
        f.write(",".join(keys) + "\n")
        for m in metrics:
            f.write(",".join(f"{m[k]:.6g}" for k in keys) + "\n")
    with open(os.path.join(outdir, "swirl_profiles.csv"), "w") as f:
        f.write("x,r,vtheta_over_Uinf,n_cells\n")
        for p in profiles:
            f.write(f"{p[0]:.4f},{p[1]:.4f},{p[2]:.5f},{p[3]}\n")

    with open(os.path.join(outdir, "run_info.txt"), "w") as f:
        f.write(f"label      {args.label}\ncase       {args.case}\ntime       {t:g}\n"
                f"created    {datetime.datetime.now():%Y-%m-%d %H:%M}\n"
                f"fields     {', '.join(available)}\nQ* level   {args.q_star}\n"
                f"wall clear {WALL_CLEAR} m\n")

    if args.no_images:
        log("done (metrics only)")
        return

    # ---- images -------------------------------------------------------------
    R = Renderer(imgdir)
    up_n = np.array([math.sin(AOA), math.cos(AOA), 0.0])     # upper-surface normal
    wing_focal = [0.30, -0.035, 0.46]
    wing_views = {
        "upper": dict(focal=wing_focal, direction=-up_n, up=[0, 0, -1], parallel=0.56),
        "lower": dict(focal=wing_focal, direction=up_n, up=[0, 0, 1], parallel=0.56),
        "tip":   dict(focal=[0.30, -0.04, 0.88], direction=[0, 0, -1], up=[0, 1, 0], parallel=0.30),
    }

    for name, cam in wing_views.items():
        R.clear()
        R.show(wing, "CpStatic", "CELLS", (-3.0, 1.0), bar_title="Cp static")
        R.camera(**cam)
        R.save(f"wing_cp_{name}.png", title=f"{args.label}: Cp on the wing, {name} view")
    if "yPlus" in available:
        for name, cam in wing_views.items():
            R.clear()
            R.show(wing, "yPlus", "CELLS", YPLUS_RANGE, preset="Viridis (matplotlib)",
                   above=[1.0, 0.0, 1.0], bar_title="y+ (magenta > 5)")
            R.camera(**cam)
            R.save(f"wing_yplus_{name}.png", title=f"{args.label}: y+ on the wing, {name} view")

    # Q isosurface, with the near-wall skin (noise from the 10 um cells) removed
    q_level = args.q_star * U_INF**2 / CHORD**2
    iso = Contour(Input=flow, ContourBy=["POINTS", "Q"], Isosurfaces=[q_level],
                  ComputeScalars=1)
    iso_poly = sm.Fetch(MergeBlocks(Input=iso))
    pts = iso_poly.GetPoints()
    if pts is not None and pts.GetNumberOfPoints() > 0:
        P = vtk_to_numpy(pts.GetData())
        wd = wall_distance(P, wing_poly)
        arr = numpy_to_vtk(wd, deep=True)
        arr.SetName("wallDist")
        iso_poly.GetPointData().AddArray(arr)
        th = vtkThreshold()
        th.SetInputData(iso_poly)
        th.SetInputArrayToProcess(0, 0, 0, 0, "wallDist")
        th.SetLowerThreshold(WALL_CLEAR)
        th.SetUpperThreshold(1e9)
        th.SetThresholdFunction(vtkThreshold.THRESHOLD_BETWEEN)
        th.Update()
        g = vtkGeometryFilter()
        g.SetInputData(th.GetOutput())
        g.Update()
        vtp = os.path.join(outdir, "q_iso_clean.vtp")
        w = vtkXMLPolyDataWriter()
        w.SetFileName(vtp)
        w.SetInputData(g.GetOutput())
        w.Write()
        iso_clean = XMLPolyDataReader(FileName=[vtp])
        iso_clean.UpdatePipeline()
        iso_views = {
            "persp": dict(focal=[1.0, -0.08, 0.78], direction=[1.4, -0.63, -0.97], up=[0, 1, 0],
                          distance=1.9),
            "side":  dict(focal=[1.05, -0.06, 0.80], direction=[0, 0, -1], up=[0, 1, 0], parallel=0.45),
            "top":   dict(focal=[1.05, -0.07, 0.70], direction=[0, -1, 0], up=[0, 0, -1],
                          parallel=0.45),
        }
        for name, cam in iso_views.items():
            R.clear()
            R.show(wing, solid=(0.8, 0.8, 0.8))
            R.show(iso_clean, "CpStatic", "POINTS", CP_RANGE, bar_title=f"Cp static (Q* = {args.q_star:g})")
            R.camera(**cam)
            R.save(f"q_iso_{name}.png",
                   title=f"{args.label}: Q-criterion isosurface (Q* = {args.q_star:g}), {name} view")

    # Cross-flow planes at the measured stations
    core_at = {m["x"]: (m["y_core"], m["z_core"]) for m in metrics}
    for x in stations:
        sl = Slice(Input=flow)
        sl.SliceType = "Plane"
        sl.SliceType.Origin = [x, 0.0, 0.0]
        sl.SliceType.Normal = [1.0, 0.0, 0.0]
        sl.Triangulatetheslice = 0
        yc, zc = core_at.get(x, (-0.08, 0.82))
        box = Clip(Input=sl)
        box.ClipType = "Box"
        try:
            box.ClipType.UseReferenceBounds = 0
        except AttributeError:
            pass
        box.ClipType.Position = [x - 0.01, yc - 0.12, zc - 0.19]
        box.ClipType.Length = [0.02, 0.24, 0.38]
        box.Invert = 1
        cross = Calculator(Input=box, ResultArrayName="Ucross", Function="U_Y*jHat+U_Z*kHat")
        arrows = Glyph(Input=cross, GlyphType="Arrow")
        arrows.OrientationArray = ["POINTS", "Ucross"]
        arrows.ScaleArray = ["POINTS", "Ucross"]
        arrows.ScaleFactor = 0.0004
        arrows.GlyphMode = "Uniform Spatial Distribution (Bounds Based)"
        arrows.MaximumNumberOfSamplePoints = 700
        outline = Slice(Input=wing)                       # wing section in this plane
        outline.SliceType = "Plane"
        outline.SliceType.Origin = [x - 1e-4, 0.0, 0.0]   # just upstream, drawn on top
        outline.SliceType.Normal = [1.0, 0.0, 0.0]
        cam = dict(focal=[x, yc, zc], direction=[1, 0, 0], up=[0, 1, 0], parallel=0.10)
        for field, rng, label in (("CpStatic", CP_RANGE, "Cp static"), ("ux", UX_RANGE, "u / U_inf")):
            R.clear()
            R.show(box, field, "POINTS", rng, bar_title=label)
            R.show(arrows, solid=(0.0, 0.0, 0.0))
            if x < 0.905:                                  # wing present in this plane
                od = R.show(outline, solid=(0.0, 0.0, 0.0))
                od.LineWidth = 4
            R.camera(**cam)
            R.save(f"plane_x{x:.3f}_{field}.png",
                   title=f"{args.label}: x = {x:.3f} m (x/c = {x/CHORD:.2f}), looking downstream, z to the right")

    # Mesh through the core (resolution check), over the wing and in the wake
    for x in (0.6604, 1.2007):
        xs = min(stations, key=lambda s: abs(s - x))
        yc, zc = core_at.get(xs, (-0.08, 0.82))
        sl = Slice(Input=flow)
        sl.SliceType = "Plane"
        sl.SliceType.Origin = [xs, 0.0, 0.0]
        sl.SliceType.Normal = [1.0, 0.0, 0.0]
        sl.Triangulatetheslice = 0
        R.clear()
        R.show(sl, "CpStatic", "CELLS", CP_RANGE, rep="Surface With Edges",
               bar_title=f"Cp static, mesh at x = {xs:.3f} m")
        R.camera(focal=[xs, yc, zc], direction=[1, 0, 0], up=[0, 1, 0], parallel=0.05)
        R.save(f"mesh_core_x{xs:.3f}.png", title=f"{args.label}: mesh through the core, x = {xs:.3f} m")

    # Streamlines: one seed cloud at the tip, one line across the outer span
    for name, setup in (
        ("tip", dict(SeedType="PointCloud", center=[0.15, 0.02, 0.90], radius=0.025, n=250)),
        ("span", dict(SeedType="Line", p1=[0.88, -0.13, 0.55], p2=[0.88, -0.13, 0.93], n=120)),
    ):
        st = StreamTracer(Input=flow, SeedType=setup["SeedType"])
        st.Vectors = ["POINTS", "U"]
        st.MaximumStreamlineLength = 3.0
        if setup["SeedType"] == "PointCloud":
            st.SeedType.Center = setup["center"]
            st.SeedType.Radius = setup["radius"]
            st.SeedType.NumberOfPoints = setup["n"]
        else:
            st.SeedType.Point1 = setup["p1"]
            st.SeedType.Point2 = setup["p2"]
            st.SeedType.Resolution = setup["n"]
        tube = Tube(Input=st, Radius=0.0012)
        R.clear()
        R.show(wing, solid=(0.8, 0.8, 0.8))
        R.show(tube, "ux", "POINTS", UX_RANGE, bar_title="u / U_inf")
        R.camera(focal=[1.0, -0.08, 0.78], direction=[1.4, -0.63, -0.97], up=[0, 1, 0],
                 distance=1.9)
        R.save(f"streamlines_{name}.png", title=f"{args.label}: streamlines ({name} seed)")

    log(f"done: {outdir}")


if __name__ == "__main__":
    main()
