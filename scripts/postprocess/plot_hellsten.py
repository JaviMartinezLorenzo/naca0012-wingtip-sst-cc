#!/usr/bin/env python3
"""
plot_hellsten.py: where the Hellsten correction acts, for one or more runs.

    python3 scripts/postprocess/plot_hellsten.py <SST label> <Hellsten label> [...]
            [--names N1 N2 ...] [--out DIR]

Reads runs/<label>/F4/*.npz (pv_limiter.py --extra F4) and
runs/<label>/limiter/*.npz.

Writes DIR/F4_and_nut.png, four cross-flow planes; for each Hellsten run:
    row  F4, the factor on the omega destruction
         (< 1: less destruction, higher omega, less nu_t;
          > 1: more destruction, lower omega, more nu_t)
    row  nu_t(Hellsten) / nu_t(SST)
"""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm

from plot_limiter import NUT_MIN, RESULTS, RUNS, core_at, draw_wing, where

NAMES = ["zoom_x0.30", "zoom_x0.66", "zoom_x0.89", "zoom_x1.05"]


def load(path):
    d = dict(np.load(path))
    return {k: (v.T if np.ndim(v) == 2 else v) for k, v in d.items()}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("sst")
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--names", nargs="+", default=None)
    ap.add_argument("--out", type=Path, default=RESULTS / "comparisons" / "hellsten_figures")
    a = ap.parse_args()
    names = a.names or a.runs
    a.out.mkdir(parents=True, exist_ok=True)

    nrow = 2 * len(a.runs)
    fig, axes = plt.subplots(nrow, 4, figsize=(20, 5.2 * nrow), constrained_layout=True,
                             squeeze=False)
    norm = LogNorm(0.1, 10)
    for k, (lab, nm) in enumerate(zip(a.runs, names)):
        for j, n in enumerate(NAMES):
            f = load(RUNS / lab / "F4" / f"{n}.npz")
            r = load(RUNS / lab / "limiter" / f"{n}.npz")
            s = load(RUNS / a.sst / "limiter" / f"{n}.npz")
            ext = [f["z"][0], f["z"][-1], f["y"][0], f["y"][-1]]
            quiet = (r["nut_nu"] < NUT_MIN) & (s["nut_nu"] < NUT_MIN)
            with np.errstate(divide="ignore", invalid="ignore"):
                ratio = r["nut_nu"] / s["nut_nu"]
            for i, (field, cmap) in enumerate(((f["F4"], "RdBu_r"), (ratio, "PuOr_r"))):
                ax = axes[2 * k + i, j]
                im = ax.imshow(np.where(quiet, np.nan, field), origin="lower", extent=ext,
                               cmap=cmap, norm=norm, interpolation="nearest")
                draw_wing(ax, f)
                c = core_at(lab, float(f["x"]))
                if c is not None:
                    ax.plot(c[1], c[0], "+", color="k", ms=14, mew=2, zorder=5)
                ax.set_aspect("equal")
                ax.set_facecolor("0.85")
                ax.set_xlabel("z (span) [m]")
                ax.set_ylabel("y (up) [m]")
                if i == 0:
                    imF = im
                else:
                    imR = im
            if k == 0:
                axes[0, j].set_title(where(float(f["x"])))
        axes[2 * k, 0].annotate(f"F4, {nm}", (-0.32, 0.5), xycoords="axes fraction",
                                rotation=90, fontsize=14, fontweight="bold", va="center")
        axes[2 * k + 1, 0].annotate(f"ν_t ratio {nm} / SST", (-0.32, 0.5),
                                    xycoords="axes fraction", rotation=90, fontsize=14,
                                    fontweight="bold", va="center")
    for k in range(len(a.runs)):
        cb = fig.colorbar(imF, ax=axes[2 * k, :], shrink=0.9, pad=0.01)
        cb.set_label("F4  (red > 1: more ω destruction → more ν_t;\nblue < 1: less → less ν_t)")
        cb = fig.colorbar(imR, ax=axes[2 * k + 1, :], shrink=0.9, pad=0.01)
        cb.set_label("ν_t(Hellsten) / ν_t(SST)\n(purple: less ν_t;  orange: more)")
    fig.suptitle("Hellsten correction, iteration 1000.  + = vortex core.  Seen from upstream, "
                 "tip to the right.  Grey: ν_t < 10 ν in both runs.", fontsize=14)
    fig.savefig(a.out / "F4_and_nut.png", dpi=110)
    plt.close(fig)
    print(f"written: {a.out}")


if __name__ == "__main__":
    main()
