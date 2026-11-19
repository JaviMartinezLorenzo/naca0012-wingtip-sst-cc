"""
pv_limiter.py: sample the SST eddy-viscosity limiter on cross-flow planes (pvpython).

    pvpython pv_limiter.py <case dir or .foam> <label> [--time T] [--decomposed]

k-omega SST sets  nu_t = a1 k / max(a1 omega, F2 S).  The ratio

    r = nu_t omega / k

is 1 where the limiter is off (nu_t = k / omega) and below 1 where it cuts
nu_t (there r = a1 omega / (F2 S)). The script samples r, nu_t / nu, k and
omega on regular grids (resampled from the cells, so plots are smooth
images instead of scattered cell centres):

    zoom_x<x>.npz   cross-flow plane x = const around the tip vortex
                    (y -0.30..0.15, z 0.55..1.00 m, 1.5 mm grid)
    span_x0.89.npz  whole semispan just upstream of the trailing edge
                    (y -0.35..0.15, z 0..1.05 m, 2.5 mm grid)

Each npz has y, z (1-D), r, nut_nu, k, omega (2-D, [z, y]) and mask
(1 inside the fluid, 0 inside the wing / outside the mesh), plus x.
--extra NAME ... also samples those cell fields (e.g. fr1 for SST-RC runs)
and stores them under the same names.
Plot with plot_limiter.py.
"""

import argparse
import os

import numpy as np
from paraview import servermanager as sm
from paraview.simple import ExtractBlock, OpenFOAMReader, ResampleToImage
from vtk.numpy_interface import dataset_adapter as dsa

NU = 1.37335e-5
ZOOM_X = [0.30, 0.66, 0.89, 1.05]
ZOOM_BOX = (-0.30, 0.15, 0.55, 1.00, 0.0015)
SPAN_X = 0.89
SPAN_BOX = (-0.35, 0.15, 0.0, 1.05, 0.0025)


def sample(src, x, box, t, extra=()):
    y0, y1, z0, z1, h = box
    ny = int(round((y1 - y0) / h)) + 1
    nz = int(round((z1 - z0) / h)) + 1
    rs = ResampleToImage(Input=src)
    rs.UseInputBounds = 0
    rs.SamplingBounds = [x, x, y0, y1, z0, z1]
    rs.SamplingDimensions = [1, ny, nz]
    rs.UpdatePipeline(t)
    d = dsa.WrapDataObject(sm.Fetch(rs))
    pd = d.PointData
    shape = (nz, ny)                       # VTK order: y fastest, then z

    def g(name):
        return np.asarray(pd[name]).reshape(shape)

    k, om, nut = g("k"), g("omega"), g("nut")
    mask = g("vtkValidPointMask").astype(np.int8)
    with np.errstate(divide="ignore", invalid="ignore"):
        r = np.where(mask > 0, nut * om / np.maximum(k, 1e-30), np.nan)
    return dict(x=x, y=np.linspace(y0, y1, ny), z=np.linspace(z0, z1, nz),
                r=r, nut_nu=np.where(mask > 0, nut / NU, np.nan),
                k=np.where(mask > 0, k, np.nan), omega=np.where(mask > 0, om, np.nan),
                mask=mask,
                **{n: np.where(mask > 0, g(n), np.nan) for n in extra})


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("case")
    ap.add_argument("label")
    ap.add_argument("--time", type=float, default=None)
    ap.add_argument("--decomposed", action="store_true")
    ap.add_argument("--out", default=None)
    ap.add_argument("--extra", nargs="*", default=[],
                    help="more cell fields to sample, e.g. fr1")
    a = ap.parse_args()

    here = os.path.dirname(os.path.abspath(__file__))
    repo = os.path.abspath(os.path.join(here, "..", ".."))
    out = a.out or os.path.join(repo, "scripts", "results", "runs", a.label, "limiter")
    os.makedirs(out, exist_ok=True)

    case = a.case
    if os.path.isdir(case):
        case = os.path.join(case, "case.foam")
        if not os.path.exists(case):
            open(case, "w").close()

    r = OpenFOAMReader(FileName=case)
    r.CaseType = "Decomposed Case" if a.decomposed else "Reconstructed Case"
    r.MeshRegions = ["internalMesh"]
    r.Createcelltopointfiltereddata = 0          # sample raw cell values
    r.UpdatePipelineInformation()
    times = list(r.TimestepValues) if r.TimestepValues else [0.0]
    t = a.time if a.time is not None else max(times)
    r.CellArrays = ["k", "omega", "nut"] + a.extra
    internal = ExtractBlock(Input=r, Selectors=["/Root/internalMesh"])
    internal.UpdatePipeline(t)
    print(f"[pv_limiter] {case} time {t:g}", flush=True)

    jobs = [(f"zoom_x{x:.2f}", x, ZOOM_BOX) for x in ZOOM_X]
    jobs.append((f"span_x{SPAN_X:.2f}", SPAN_X, SPAN_BOX))
    for name, x, box in jobs:
        d = sample(internal, x, box, t, a.extra)
        np.savez_compressed(os.path.join(out, name + ".npz"), time=t, **d)
        lim = np.nanmean(d["r"] < 0.99) * 100
        print(f"[pv_limiter] {name}: {d['r'].shape}, limiter active in {lim:.1f} % of the plane",
              flush=True)
    print(f"[pv_limiter] written {out}", flush=True)


if __name__ == "__main__":
    main()
