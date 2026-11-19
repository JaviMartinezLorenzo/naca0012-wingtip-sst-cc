#!/usr/bin/env python3
"""
plot_report.py: compact figures for reports.

    python3 scripts/postprocess/plot_report.py core  <label> ... --names N ... --out FILE.png
    python3 scripts/postprocess/plot_report.py nutr  <label> ... --names N ... --out FILE.png

core   core Cp_min, axial velocity and core radius against x/c, runs and
       experiment (same core definition as compare_runs.py).
nutr   swirl velocity v_theta(r) and azimuthally averaged nu_t(r)/nu around
       the vortex core, at the wake planes x = 0.89 m and 1.05 m.
       Reads runs/<label>/swirl_profiles.csv, limiter/zoom_x*.npz and
       core_metrics.csv.
"""

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from compare_runs import experiment            # noqa: E402
from plot_limiter import CHORD, RUNS, core_at  # noqa: E402

COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"]


def fig_core(labels, names, out):
    exp, _ = experiment()
    keys = [("cp_min", "core $C_{p,\\min}$", True),
            ("u_core", "axial velocity $u_{core}/U_\\infty$", False),
            ("r_c_c", "core radius $r_c/c$", False)]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.4), constrained_layout=True)
    for ax, (k, title, inv) in zip(axes, keys):
        ax.plot(exp["x_c"], exp[k], "ko-", lw=1.8, ms=5, label="Experiment")
        for i, (lab, nm) in enumerate(zip(labels, names)):
            c = pd.read_csv(RUNS / lab / "core_metrics.csv")
            ax.plot(c["x_c"], c[k], "s--", color=COLORS[i], ms=4, lw=1.4, label=nm)
        ax.axvline(0.739, color="0.5", ls=":", lw=1)
        ax.text(0.75, 0.04, "TE", transform=ax.get_xaxis_transform(), color="0.4")
        ax.set_title(title)
        ax.set_xlabel("x / c")
        ax.grid(alpha=0.3)
        if inv:
            ax.invert_yaxis()
    axes[0].legend(fontsize=9)
    fig.savefig(out, dpi=150)
    plt.close(fig)


def radial_nut(lab, plane, rmax, nbin=40):
    d = dict(np.load(RUNS / lab / "limiter" / f"{plane}.npz"))
    yc, zc = core_at(lab, float(d["x"]))
    Z, Y = np.meshgrid(d["z"], d["y"], indexing="ij")     # npz arrays are [z, y]
    r = np.hypot(Y - yc, Z - zc)
    edges = np.linspace(0, rmax, nbin + 1)
    idx = np.digitize(r.ravel(), edges) - 1
    v = d["nut_nu"].ravel()
    prof = np.array([np.nanmean(v[idx == i]) if np.any(idx == i) else np.nan
                     for i in range(nbin)])
    return 0.5 * (edges[1:] + edges[:-1]), prof


def fig_nutr(labels, names, out):
    planes = [("zoom_x0.89", 0.8885), ("zoom_x1.05", 1.0538)]
    rmax = 0.10
    fig, axes = plt.subplots(2, 2, figsize=(12, 7.5), constrained_layout=True, sharex=True)
    for j, (plane, xs) in enumerate(planes):
        for i, (lab, nm) in enumerate(zip(labels, names)):
            s = pd.read_csv(RUNS / lab / "swirl_profiles.csv")
            s = s[np.isclose(s["x"], xs, atol=2e-3)]
            axes[0, j].plot(s["r"] / CHORD, s["vtheta_over_Uinf"].abs(), color=COLORS[i], lw=1.6,
                            label=nm)
            rr, nut = radial_nut(lab, plane, rmax)
            axes[1, j].semilogy(rr / CHORD, nut, color=COLORS[i], lw=1.6, label=nm)
        axes[0, j].set_title(f"x = {xs:.2f} m (x/c = {xs / CHORD:.2f})")
        axes[0, j].set_ylabel("$|v_\\theta| / U_\\infty$")
        axes[1, j].set_ylabel("$\\nu_t / \\nu$ (azimuthal average)")
        axes[1, j].set_xlabel("distance from the core centre r / c")
        for ax in axes[:, j]:
            ax.grid(alpha=0.3, which="both")
            ax.set_xlim(0, rmax / CHORD)
    axes[0, 0].legend(fontsize=9)
    fig.savefig(out, dpi=150)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("kind", choices=["core", "nutr"])
    ap.add_argument("labels", nargs="+")
    ap.add_argument("--names", nargs="+", default=None)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    names = a.names or a.labels
    a.out.parent.mkdir(parents=True, exist_ok=True)
    {"core": fig_core, "nutr": fig_nutr}[a.kind](a.labels, names, a.out)
    print(f"written: {a.out}")


if __name__ == "__main__":
    main()
