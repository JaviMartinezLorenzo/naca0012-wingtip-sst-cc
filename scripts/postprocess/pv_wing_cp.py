"""
pv_wing_cp.py: wing surface Cp at the experimental pressure-tap rows (pvpython).

    pvpython pv_wing_cp.py <case dir or .foam> <label> [--time T] [--decomposed]

The experiment (CHORDQPL.DAT) has chordwise tap rows at
    span stations  Z = 6, 12, 24, 28, 30, 31, 32 in   (constant-chord part)
    tip angles     theta = 10, 22.5, 45, 67.5, 90 deg (rounded tip, z > 33.12 in)
on the upper and lower surface. The rounded tip is the NACA 0012 section
revolved about its chord line, so a tap row at angle theta lies in the
half-plane that contains the chord line at z = 33.12 in and makes the angle
theta with the section's 'up' direction (theta = 0: edge of the constant-chord
part, theta = 90: the very tip).

The wing patch is sliced with those planes and CpStatic (interpolated to the
surface points) is written with x/c measured from the leading edge, as in
the experiment.

Writes scripts/results/runs/<label>/wing_cp.csv:
    kind (span|tip), station (Z in inches or theta in deg), side (upper|lower),
    x_c, cp
"""

import argparse
import math
import os

import numpy as np
from paraview import servermanager as sm
from paraview.simple import ExtractBlock, MergeBlocks, OpenFOAMReader, Slice
from vtk.numpy_interface import dataset_adapter as dsa

CHORD = 1.2192
U_INF = 51.816
AOA = math.radians(10.0)
INCH = 0.0254
Z_SEAM = 33.12 * INCH
SPAN_STATIONS = [6, 12, 24, 28, 30, 31, 32]           # inches from the root
TIP_STATIONS = [10.0, 22.5, 45.0, 67.5, 90.0]          # degrees

# Wing-section axes in the CFD frame (wing pitched 10 deg nose up about the
# root quarter chord, which is the origin).
EX = np.array([math.cos(AOA), -math.sin(AOA), 0.0])    # chordwise, LE -> TE
EY = np.array([math.sin(AOA), math.cos(AOA), 0.0])     # section 'up'
EZ = np.array([0.0, 0.0, 1.0])                         # spanwise, to the tip


def cut(wing, origin, normal, t):
    """Points and CpStatic of the wing patch cut by a plane."""
    s = Slice(Input=wing)
    s.SliceType = "Plane"
    s.SliceType.Origin = list(map(float, origin))
    s.SliceType.Normal = list(map(float, normal))
    s.UpdatePipeline(t)
    d = dsa.WrapDataObject(sm.Fetch(MergeBlocks(Input=s)))
    if d.GetNumberOfPoints() == 0:
        return np.zeros((0, 3)), np.zeros(0)
    p = np.asarray(d.Points)
    if "CpStatic" in d.PointData.keys():
        cp = np.asarray(d.PointData["CpStatic"])
    else:
        cp = np.asarray(d.PointData["p"]) / (0.5 * U_INF ** 2)
    return p, cp


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("case")
    ap.add_argument("label")
    ap.add_argument("--time", type=float, default=None)
    ap.add_argument("--decomposed", action="store_true")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    here = os.path.dirname(os.path.abspath(__file__))
    repo = os.path.abspath(os.path.join(here, "..", ".."))
    out = a.out or os.path.join(repo, "scripts", "results", "runs", a.label)
    os.makedirs(out, exist_ok=True)

    case = a.case
    if os.path.isdir(case):
        case = os.path.join(case, "case.foam")
        if not os.path.exists(case):
            open(case, "w").close()

    r = OpenFOAMReader(FileName=case)
    r.CaseType = "Decomposed Case" if a.decomposed else "Reconstructed Case"
    r.MeshRegions = ["patch/wing"]
    r.Createcelltopointfiltereddata = 1
    r.UpdatePipelineInformation()
    times = list(r.TimestepValues) if r.TimestepValues else [0.0]
    t = a.time if a.time is not None else max(times)
    avail = list(r.CellArrays.Available)
    r.CellArrays = [f for f in ("CpStatic", "p") if f in avail]
    wing = ExtractBlock(Input=r, Selectors=["/Root/boundary/wing"])
    wing.UpdatePipeline(t)
    print(f"[pv_wing_cp] {case} time {t:g}", flush=True)

    rows = []
    for zin in SPAN_STATIONS:
        p, cp = cut(wing, (0, 0, zin * INCH), EZ, t)
        xc = (p @ EX + 0.25 * CHORD) / CHORD
        yn = p @ EY
        for side, m in (("upper", yn >= 0), ("lower", yn < 0)):
            rows += [("span", zin, side, x, c) for x, c in zip(xc[m], cp[m])]
        print(f"[pv_wing_cp] Z = {zin} in: {len(p)} points", flush=True)

    p0 = np.array([0.0, 0.0, Z_SEAM])
    for th in TIP_STATIONS:
        tr = math.radians(th)
        for side, sgn in (("upper", 1.0), ("lower", -1.0)):
            d = sgn * math.cos(tr) * EY + math.sin(tr) * EZ   # in-plane, away from axis
            n = np.cross(EX, d)
            p, cp = cut(wing, p0, n, t)
            m = ((p - p0) @ d > 1e-6) & (p[:, 2] > Z_SEAM - 1e-6)
            xc = (p[m] @ EX + 0.25 * CHORD) / CHORD
            rows += [("tip", th, side, x, c) for x, c in zip(xc, cp[m])]
            print(f"[pv_wing_cp] theta = {th} {side}: {int(m.sum())} points", flush=True)

    path = os.path.join(out, "wing_cp.csv")
    with open(path, "w") as f:
        f.write("kind,station,side,x_c,cp\n")
        for k, s, side, x, c in sorted(rows, key=lambda q: (q[0], q[1], q[2], q[3])):
            f.write(f"{k},{s:g},{side},{x:.5f},{c:.5f}\n")
    print(f"[pv_wing_cp] written {path} ({len(rows)} points)", flush=True)


if __name__ == "__main__":
    main()
