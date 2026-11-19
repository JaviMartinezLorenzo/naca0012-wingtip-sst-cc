#!/usr/bin/env python3
"""
plot_rc.py: where the SST-RC rotation/curvature correction acts.

    python3 scripts/postprocess/plot_rc.py <label SST-RC> <label SST> [--out DIR]

Reads runs/<RC label>/fr1/*.npz (pv_limiter.py --extra fr1 ...) and
runs/<label>/limiter/*.npz of both runs.

Writes DIR/fr1_and_nut.png, four cross-flow planes:
    row 1  fr1, the factor on the turbulence production
           (1 = no change, < 1 = production cut, > 1 = boosted)
    row 2  nu_t(SST-RC) / nu_t(SST)
"""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm, TwoSlopeNorm

from plot_limiter import NUT_MIN, RESULTS, RUNS, core_at, draw_wing, where

NAMES = ["zoom_x0.30", "zoom_x0.66", "zoom_x0.89", "zoom_x1.05"]


def load(path):
    d = dict(np.load(path))
    return {k: (v.T if np.ndim(v) == 2 else v) for k, v in d.items()}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("rc")
    ap.add_argument("sst")
    ap.add_argument("--out", type=Path, default=RESULTS / "comparisons" / "rc_figures")
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 4, figsize=(20, 10.5), constrained_layout=True)
    fr_norm = TwoSlopeNorm(vmin=0.0, vcenter=1.0, vmax=1.25)
    rat_norm = LogNorm(0.1, 10)
    for j, n in enumerate(NAMES):
        f = load(RUNS / a.rc / "fr1" / f"{n}.npz")
        r = load(RUNS / a.rc / "limiter" / f"{n}.npz")
        s = load(RUNS / a.sst / "limiter" / f"{n}.npz")
        ext = [f["z"][0], f["z"][-1], f["y"][0], f["y"][-1]]
        quiet = (r["nut_nu"] < NUT_MIN) & (s["nut_nu"] < NUT_MIN)
        im1 = axes[0, j].imshow(np.where(quiet, np.nan, f["fr1"]), origin="lower", extent=ext,
                                cmap="RdBu", norm=fr_norm, interpolation="nearest")
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = r["nut_nu"] / s["nut_nu"]
        im2 = axes[1, j].imshow(np.where(quiet, np.nan, ratio), origin="lower", extent=ext,
                                cmap="PuOr_r", norm=rat_norm, interpolation="nearest")
        for i, lab in ((0, a.rc), (1, a.rc)):
            ax = axes[i, j]
            draw_wing(ax, f)
            c = core_at(lab, float(f["x"]))
            if c is not None:
                ax.plot(c[1], c[0], "+", color="k", ms=14, mew=2, zorder=5)
            ax.set_aspect("equal")
            ax.set_xlabel("z (span) [m]")
            ax.set_ylabel("y (up) [m]")
        axes[0, j].set_title(where(float(f["x"])))
    for i, txt in enumerate(("fr1", "ν_t ratio RC / SST")):
        axes[i, 0].annotate(txt, (-0.32, 0.5), xycoords="axes fraction", rotation=90,
                            fontsize=14, fontweight="bold", va="center")
    cb = fig.colorbar(im1, ax=axes[0, :], shrink=0.9, pad=0.01)
    cb.set_label("fr1   (red: production cut;  white: 1, no change;  blue: boosted)")
    cb = fig.colorbar(im2, ax=axes[1, :], shrink=0.9, pad=0.01)
    cb.set_label("ν_t(SST-RC) / ν_t(SST)   (purple: less ν_t with RC)")
    fig.suptitle("SST-RC (Smirnov-Menter), iteration 1000.  + = vortex core.  "
                 "Seen from upstream, tip to the right.  Grey: ν_t < 10 ν in both runs.",
                 fontsize=14)
    for ax in axes.flat:
        ax.set_facecolor("0.85")
    fig.savefig(a.out / "fr1_and_nut.png", dpi=130)
    plt.close(fig)
    print(f"written: {a.out}")


if __name__ == "__main__":
    main()
