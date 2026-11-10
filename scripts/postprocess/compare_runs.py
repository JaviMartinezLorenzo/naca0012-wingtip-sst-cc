#!/usr/bin/env python3
"""
compare_runs.py: overlay vortex-core metrics of several runs and the experiment.

    python3 scripts/postprocess/compare_runs.py <label> [<label> ...] [--out NAME]

Each <label> is a folder scripts/results/<label>/ written by pv_vortex.py.
Writes scripts/results/<NAME>/ (default compare_<label1>_vs_<label2>...):
    core_comparison.png   Cp_min, axial velocity, peak swirl, core radius and
                          core position vs x/c, runs + experiment
    swirl_profiles.png    v_theta(r) at the trailing edge and two wake stations
    core_comparison.csv   all metrics side by side
    yplus_comparison.csv  wing y+ summary side by side

The experimental core is found exactly as in pv_vortex.py: minimum Cp_static
per plane (RETAKALL.DAT), swirl v_theta(r) azimuthally averaged around it
from TAKALL.DAT, core radius at the first swirl peak.
"""

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
RESULTS = ROOT / "scripts" / "results"
sys.path.insert(0, str(HERE))
from import_exp_data import exp_data          # noqa: E402

CHORD = 1.2192
X_TE = 0.9007                                  # trailing edge, wing frame [m]


def first_peak(r, v):
    """Same definition as pv_vortex.py: first local max of |v| outward."""
    a = np.abs(np.asarray(v, float))
    if len(a) < 3:
        return np.nan, np.nan
    s = np.convolve(a, np.ones(3) / 3.0, mode="same")
    s[0], s[-1] = a[0], a[-1]
    for j in range(1, len(s) - 1):
        if s[j] >= s[j - 1] and s[j] > s[j + 1] and s[j] >= 0.5 * s.max():
            return a[j], r[j]
    j = int(np.argmax(a))
    return a[j], r[j]


def experiment():
    """Core metrics and swirl profiles from the measured planes."""
    _, exp_p, exp_v = exp_data()
    rows, prof = [], []
    for x, g in exp_p.groupby(exp_p["x_of"].round(4)):
        if x <= 0:                               # inflow plane
            continue
        i = g["cp_stat"].idxmin()
        yc, zc, cp = g.at[i, "y_of"], g.at[i, "z_of"], g.at[i, "cp_stat"]
        v = exp_v[exp_v["x_of"].round(4) == x]
        if v.empty:
            continue
        dy, dz = v["y_of"].to_numpy() - yc, v["z_of"].to_numpy() - zc
        r = np.hypot(dy, dz)
        u_core = v["u"].to_numpy()[np.argmin(r)]
        m = (r > 1e-9) & (r < 0.12)
        vt = (dy[m] * v["w"].to_numpy()[m] - dz[m] * v["v"].to_numpy()[m]) / r[m]
        edges = np.arange(0.0, 0.1201, 0.004)    # measurement grid is coarser than CFD
        which = np.digitize(r[m], edges) - 1
        rb, vb = [], []
        for b in range(len(edges) - 1):
            k = which == b
            if k.sum() >= 2:
                rb.append(0.5 * (edges[b] + edges[b + 1]))
                vb.append(vt[k].mean())
                prof.append((x, rb[-1], vb[-1]))
        vmax, rc = first_peak(rb, vb)
        rows.append(dict(x=x, x_c=x / CHORD, y_core=yc, z_core=zc, y_c=yc / CHORD,
                         z_c=zc / CHORD, cp_min=cp, u_core=u_core, vtheta_max=vmax,
                         r_c=rc, r_c_c=rc / CHORD))
    return (pd.DataFrame(rows).sort_values("x"),
            pd.DataFrame(prof, columns=["x", "r", "vtheta_over_Uinf"]))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("labels", nargs="+", help="run labels (folders in scripts/results)")
    ap.add_argument("--out", default=None, help="output folder name in scripts/results")
    args = ap.parse_args()

    runs = {}
    for lab in args.labels:
        f = RESULTS / lab / "core_metrics.csv"
        if not f.exists():
            sys.exit(f"missing {f}: run pv_vortex.sh for '{lab}' first")
        runs[lab] = (pd.read_csv(f).sort_values("x"),
                     pd.read_csv(RESULTS / lab / "swirl_profiles.csv"))
    exp, exp_prof = experiment()

    out = RESULTS / (args.out or "compare_" + "_vs_".join(args.labels))
    out.mkdir(parents=True, exist_ok=True)

    # ---- metrics vs x/c -------------------------------------------------------
    panels = [("cp_min", r"$C_{p,\min}$ at core", True),
              ("u_core", r"$u_{core}/U_\infty$", False),
              ("vtheta_max", r"$v_{\theta,\max}/U_\infty$", False),
              ("r_c_c", r"$r_c / c$", False),
              ("y_c", r"$y_{core}/c$", False),
              ("z_c", r"$z_{core}/c$", False)]
    fig, axes = plt.subplots(2, 3, figsize=(15, 8.5))
    for ax, (key, title, invert) in zip(axes.flat, panels):
        ax.plot(exp["x_c"], exp[key], "ko-", lw=1.5, ms=5, label="Experiment")
        for lab, (m, _) in runs.items():
            ax.plot(m["x_c"], m[key], "s--", ms=4, lw=1.3, label=lab)
        ax.axvline(X_TE / CHORD, color="0.6", lw=1, ls=":")
        ax.set_title(title)
        ax.set_xlabel("x / c")
        ax.grid(alpha=0.3)
        if invert:
            ax.invert_yaxis()
    for ax in axes.flat:
        ax.text(X_TE / CHORD, 0.02, " TE", color="0.4", transform=ax.get_xaxis_transform())
    axes.flat[0].legend(fontsize=9)
    fig.suptitle("Vortex core: " + ", ".join(runs) + " vs experiment (Chow et al. 1997)")
    fig.tight_layout()
    fig.savefig(out / "core_comparison.png", dpi=150)
    plt.close(fig)

    # ---- swirl profiles at TE and two wake stations ----------------------------
    targets = [0.889, 1.201, 1.727]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), sharey=True)
    for ax, xt in zip(axes, targets):
        xe = exp_prof["x"].unique()
        xe = xe[np.argmin(np.abs(xe - xt))]
        e = exp_prof[exp_prof["x"] == xe]
        ax.plot(e["r"] * 1e3, e["vtheta_over_Uinf"].abs(), "ko-", ms=4, label="Experiment")
        for lab, (_, p) in runs.items():
            xs = p["x"].unique()
            xs = xs[np.argmin(np.abs(xs - xt))]
            q = p[p["x"] == xs]
            ax.plot(q["r"] * 1e3, q["vtheta_over_Uinf"].abs(), "-", lw=1.5, label=lab)
        ax.set_title(f"x = {xe:.3f} m (x/c = {xe / CHORD:.2f})")
        ax.set_xlabel("r [mm]")
        ax.grid(alpha=0.3)
    axes[0].set_ylabel(r"$|v_\theta| / U_\infty$")
    axes[0].legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(out / "swirl_profiles.png", dpi=150)
    plt.close(fig)

    # ---- tables ------------------------------------------------------------------
    keys = ["y_c", "z_c", "cp_min", "u_core", "vtheta_max", "r_c_c"]
    table = exp[["x", "x_c"] + keys].rename(columns={k: f"{k}_exp" for k in keys})
    for lab, (m, _) in runs.items():
        cols = keys + [c for c in ("cells_across_core", "nut_core_over_nu") if c in m]
        mm = m[["x"] + cols].rename(columns={k: f"{k}_{lab}" for k in cols})
        table = pd.merge_asof(table.sort_values("x"), mm.sort_values("x"), on="x",
                              direction="nearest", tolerance=0.01)
    table.to_csv(out / "core_comparison.csv", index=False, float_format="%.5g")

    ys = []
    for lab in runs:
        f = RESULTS / lab / "yplus_summary.csv"
        if f.exists():
            d = pd.read_csv(f)
            d.insert(0, "run", lab)
            ys.append(d)
    if ys:
        ytab = pd.concat(ys)
        ytab.to_csv(out / "yplus_comparison.csv", index=False)
        print("\nWing y+ (whole wing):")
        print(ytab[ytab["region"] == "wing"].to_string(index=False))

    pd.set_option("display.width", 200)
    print("\nCore Cp_min and core radius vs experiment:")
    show = ["x_c", "cp_min_exp"] + [f"cp_min_{l}" for l in runs] + \
           ["r_c_c_exp"] + [f"r_c_c_{l}" for l in runs]
    print(table[show].to_string(index=False, float_format=lambda v: f"{v:7.3f}"))
    print(f"\nwritten to {out}")


if __name__ == "__main__":
    main()
