#!/usr/bin/env python3
"""
plot_limiter.py: figures that explain the k-omega SST eddy-viscosity limiter.

    python3 scripts/postprocess/plot_limiter.py <label a1=0.31> [<label a1=1>]
            [--out DIR]

Reads scripts/results/runs/<label>/limiter/*.npz (pv_limiter.py) and
core_metrics.csv (pv_vortex.py, for the vortex-core marker).

Writes to DIR (default scripts/results/comparisons/limiter_figures/):
    1_limiter_theory.png    what the limiter does: nu_t / (k/omega) against the
                            strain S/omega, for a1 = 0.31 and a1 = 1
    2_limiter_map.png       where it is active in the a1 = 0.31 solution
                            (r = nu_t omega / k), four cross-flow planes
    3_limiter_span.png      the same just upstream of the trailing edge, whole
                            semispan (boundary layers on both surfaces)
    4_nut_a031_vs_a1.png    nu_t / nu with a1 = 0.31 and a1 = 1, and their
                            ratio (only with the second label)

Planes are seen from upstream: span z to the right (tip on the right),
y up (lift direction). The wing is dark grey.
"""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import BoundaryNorm, ListedColormap, LogNorm, TwoSlopeNorm

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "scripts" / "results"
RUNS = RESULTS / "runs"                 # one folder per run (pv_*.py output)
CHORD = 1.2192
X_TE = 0.9007
BETA_STAR = 0.09
WING_GREY = "#3a3a3a"
OFF_BLUE = "#bcd7ec"               # r = 1: limiter off
QUIET = "#ffffff"                  # nu_t < NUT_MIN nu: freestream level, limiter irrelevant
NUT_MIN = 10.0

plt.rcParams.update({"font.size": 11, "axes.titlesize": 12})


def load(label, name):
    """npz arrays are [z, y]; transpose to [y, z] so imshow rows are y (up)."""
    d = dict(np.load(RUNS / label / "limiter" / f"{name}.npz"))
    for k, v in d.items():
        if np.ndim(v) == 2:
            d[k] = v.T
    return d


def core_at(label, x):
    """Vortex-core (y, z) at x from core_metrics.csv, or None."""
    f = RUNS / label / "core_metrics.csv"
    if not f.exists():
        return None
    c = pd.read_csv(f)
    xs = c.iloc[:, 0].to_numpy()
    ycol = [k for k in c.columns if k.startswith("y_c")][0]
    zcol = [k for k in c.columns if k.startswith("z_c")][0]
    if x < xs.min() - 0.15:
        return None
    return float(np.interp(x, xs, c[ycol])), float(np.interp(x, xs, c[zcol]))


def where(x):
    s = "over the wing" if x < X_TE else "in the wake"
    return f"x = {x:.2f} m  (x/c = {x / CHORD:.2f}, {s})"


def draw_wing(ax, d):
    ext = [d["z"][0], d["z"][-1], d["y"][0], d["y"][-1]]
    solid = np.ma.masked_where(d["mask"] > 0, np.ones_like(d["mask"], float))
    ax.imshow(solid, origin="lower", extent=ext, cmap=ListedColormap([WING_GREY]),
              interpolation="nearest", zorder=3)
    return ext


# ---------------------------------------------------------------------------
def fig_theory(out):
    s = np.linspace(0.0, 4.0, 800)                   # S / omega
    fig, ax = plt.subplots(1, 2, figsize=(14, 5.2), constrained_layout=True)

    for a1, col in ((0.31, "#1f6fb4"), (1.0, "#d1495b")):
        r = np.minimum(1.0, a1 / np.maximum(s, 1e-9))
        ax[0].plot(s, r, color=col, lw=2.4, label=f"a1 = {a1:g}")
        ax[0].axvline(a1, color=col, ls=":", lw=1.2)
        pk = r * s ** 2 / BETA_STAR                   # P_k / epsilon with that nu_t
        ax[1].plot(s, pk, color=col, lw=2.4, label=f"a1 = {a1:g}")
        ax[1].axhline(a1 ** 2 / BETA_STAR, color=col, ls=":", lw=1.2)
    s_free = s
    ax[1].plot(s_free, s_free ** 2 / BETA_STAR, color="0.5", lw=1.4, ls="--",
               label="no limiter (k-ω / BSL)")

    ax[0].fill_between([0, 0.31], 0, 1.08, color="#1f6fb4", alpha=0.06)
    ax[0].text(0.155, 0.45, "both\nunlimited", ha="center", fontsize=10)
    ax[0].fill_between([0.31, 1.0], 0, 1.08, color="#d1495b", alpha=0.06)
    ax[0].text(0.72, 0.9, "only a1 = 0.31\ncuts ν_t", ha="center", fontsize=10)
    ax[0].text(2.6, 0.62, "both cut ν_t\n(a1 = 1 keeps 3.2× more ν_t)", ha="center", fontsize=10)
    ax[0].set_xlim(0, 4)
    ax[0].set_ylim(0, 1.08)
    ax[0].set_xlabel("strain rate / turbulence frequency   S / ω   (with F2 = 1)")
    ax[0].set_ylabel("r = ν_t / (k/ω)   (1 = limiter off)")
    ax[0].set_title("Eddy viscosity:  ν_t = a1 k / max(a1 ω, F2 S)")
    ax[0].legend(loc="upper right")
    ax[0].grid(alpha=0.3)

    ax[1].set_xlim(0, 4)
    ax[1].set_ylim(0, 25)
    ax[1].set_xlabel("S / ω")
    ax[1].set_ylabel("production / dissipation of k   P_k / ε")
    ax[1].set_title("What the cap means: P_k/ε stops growing like (S/ω)²")
    ax[1].text(1.5, 0.3, "limiter starts at P_k/ε = 1.07 (a1 = 0.31)",
               ha="left", color="#1f6fb4", fontsize=10)
    ax[1].text(1.15, 11.6, "limiter starts at P_k/ε = 11 (a1 = 1)",
               ha="left", color="#d1495b", fontsize=10)
    ax[1].legend(loc="upper left")
    ax[1].grid(alpha=0.3)

    fig.suptitle("The SST limiter: once the strain is large compared with ω, ν_t stops "
                 "following k/ω and the shear stress is capped at a1·k (Bradshaw)",
                 fontsize=12)
    fig.savefig(out / "1_limiter_theory.png", dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
R_BOUNDS = [0.0, 0.1, 0.2, 0.4, 0.6, 0.8, 0.99, 1.01]
R_CMAP = ListedColormap(["#5e0b15", "#9e2a2b", "#d1495b", "#ee8959", "#f4c095",
                         "#fbe6c8", OFF_BLUE])
R_NORM = BoundaryNorm(R_BOUNDS, R_CMAP.N)


def r_panel(ax, d, label):
    ext = [d["z"][0], d["z"][-1], d["y"][0], d["y"][-1]]
    im = ax.imshow(np.clip(d["r"], 0, 1.0), origin="lower", extent=ext, cmap=R_CMAP,
                   norm=R_NORM, interpolation="nearest")
    quiet = np.ma.masked_where(~(d["nut_nu"] < NUT_MIN), np.ones_like(d["r"]))
    ax.imshow(quiet, origin="lower", extent=ext, cmap=ListedColormap([QUIET]),
              interpolation="nearest", zorder=2)
    yy, zz = np.meshgrid(d["y"], d["z"], indexing="ij")
    ax.contour(zz, yy, np.nan_to_num(d["nut_nu"]), levels=[NUT_MIN], colors="0.55",
               linewidths=0.6, zorder=2)
    draw_wing(ax, d)
    c = core_at(label, float(d["x"]))
    if c is not None:
        ax.plot(c[1], c[0], "+", color="k", ms=14, mew=2, zorder=5)
    ax.set_xlabel("z (span) [m]")
    ax.set_ylabel("y (up) [m]")
    ax.set_aspect("equal")
    return im


def fig_map(label, out, a1="0.31", fname="2_limiter_map.png", subtitle=None):
    names = ["zoom_x0.30", "zoom_x0.66", "zoom_x0.89", "zoom_x1.05"]
    fig, axes = plt.subplots(1, 4, figsize=(20, 6.2), constrained_layout=True)
    for ax, n in zip(axes, names):
        d = load(label, n)
        im = r_panel(ax, d, label)
        ax.set_title(where(float(d["x"])))
    # (text position z, y), text, (arrow tip z, y) or None
    notes = {
        0: [((0.68, -0.24), "flow round the tip:\nlimited at the corner", (0.90, -0.085)),
            ((0.66, 0.125), "outer edge of the\nboundary layer", (0.72, 0.01))],
        1: [((0.66, 0.02), "vortex starts at the tip;\nits core is blue (r = 1)", (0.82, -0.065)),
            ((0.70, -0.26), "tip shear layer leaving\nthe pressure side: limited", (0.865, -0.12))],
        2: [((0.66, -0.26), "boundary layer near the\ntrailing edge: limited", (0.62, -0.155)),
            ((0.97, 0.12), "feeding\nshear layer", (0.91, 0.0))],
        3: [((0.70, 0.04), "wake sheet rolling in:\nlimited", (0.86, 0.0)),
            ((0.80, -0.27), "edge of the turbulent\nregion: limited", (0.965, -0.13))],
    }
    if a1 != "0.31":
        notes = {}                     # labels are placed for the a1 = 0.31 solution
    for i, items in notes.items():
        for (z, y), txt, tip in items:
            kw = dict(arrowprops=dict(arrowstyle="->", lw=1.3, color="k")) if tip else {}
            axes[i].annotate(txt, tip or (z, y), xytext=(z, y), fontsize=10, ha="center",
                             va="center", zorder=6,
                             bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="0.3",
                                       alpha=0.9), **kw)
    cb = fig.colorbar(im, ax=axes, shrink=0.85, pad=0.01, ticks=R_BOUNDS[:-1] + [1.0])
    cb.set_label("r = ν_t ω / k      (blue: r = 1, limiter off;  red: ν_t cut to r × k/ω)")
    fig.suptitle(f"Where the SST limiter is active (a1 = {a1}, iteration 1000).  + = vortex core (r = 1 at every station).  "
                 "Seen from upstream, tip to the right.\n"
                 + (subtitle or f"White: ν_t < {NUT_MIN:g} ν (freestream level, grey line), the limiter changes nothing there."),
                 fontsize=13)
    fig.savefig(out / fname, dpi=150)
    plt.close(fig)


def fig_span(label, out):
    d = load(label, "span_x0.89")
    fig, ax = plt.subplots(figsize=(17, 6.0), constrained_layout=True)
    im = r_panel(ax, d, label)
    ax.set_ylim(-0.28, 0.12)
    ax.set_xlabel("z (span) [m]      (z = 0: root at the tunnel wall;  tip at z = 0.91)")
    ax.set_title(f"Whole semispan, just upstream of the trailing edge: {where(float(d['x']))}")
    cb = fig.colorbar(im, ax=ax, shrink=0.9, pad=0.01, ticks=R_BOUNDS[:-1] + [1.0])
    cb.set_label("r = ν_t ω / k")
    ax.annotate("boundary layers on both surfaces near the trailing edge: limited all along the span\n"
                "(adverse pressure gradient: P_k > ε).  The wing itself is the thin dark line.",
                (0.30, -0.157), xytext=(0.36, -0.255), fontsize=11, ha="center", va="center",
                arrowprops=dict(arrowstyle="->", lw=1.3),
                bbox=dict(boxstyle="round", fc="white", ec="0.3"), zorder=6)
    ax.annotate("vortex core: limiter off", (0.82, -0.10), xytext=(0.62, -0.06), fontsize=11,
                ha="center", arrowprops=dict(arrowstyle="->", lw=1.3),
                bbox=dict(boxstyle="round", fc="white", ec="0.3"), zorder=6)
    fig.savefig(out / "3_limiter_span.png", dpi=150)
    plt.close(fig)


def fig_compare(lab1, lab2, out):
    names = ["zoom_x0.30", "zoom_x0.66", "zoom_x0.89", "zoom_x1.05"]
    fig, axes = plt.subplots(3, 4, figsize=(20, 15.5), constrained_layout=True)
    nut_norm = LogNorm(1, 3000)
    rat_norm = TwoSlopeNorm(vmin=0.5, vcenter=1.0, vmax=4.0)
    for j, n in enumerate(names):
        d1, d2 = load(lab1, n), load(lab2, n)
        ext = [d1["z"][0], d1["z"][-1], d1["y"][0], d1["y"][-1]]
        for i, (d, lab) in enumerate(((d1, lab1), (d2, lab2))):
            ax = axes[i, j]
            imn = ax.imshow(np.clip(d["nut_nu"], 1, None), origin="lower", extent=ext,
                            cmap="viridis", norm=nut_norm, interpolation="nearest")
            draw_wing(ax, d)
            c = core_at(lab, float(d["x"]))
            if c is not None:
                ax.plot(c[1], c[0], "+", color="w", ms=14, mew=2, zorder=5)
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = d2["nut_nu"] / d1["nut_nu"]
        ax = axes[2, j]
        quiet = (d1["nut_nu"] < NUT_MIN) & (d2["nut_nu"] < NUT_MIN)
        imr = ax.imshow(np.where(quiet, np.nan, ratio), origin="lower", extent=ext,
                        cmap="RdBu_r", norm=rat_norm, interpolation="nearest")
        draw_wing(ax, d1)
        for (lab, dd) in ((lab1, d1),):
            c = core_at(lab, float(dd["x"]))
            if c is not None:
                ax.plot(c[1], c[0], "+", color="k", ms=14, mew=2, zorder=5)
        axes[0, j].set_title(where(float(d1["x"])))
        for ax in axes[:, j]:
            ax.set_aspect("equal")
            ax.set_xlabel("z (span) [m]")
            ax.set_ylabel("y (up) [m]")
    box = dict(boxstyle="round,pad=0.25", fc="white", ec="0.3", alpha=0.9)
    arrow = dict(arrowstyle="->", lw=1.3)
    for j in (2,):
        axes[2, j].annotate("core: about the same ν_t\n(limiter off in both runs)", (0.82, -0.10),
                            xytext=(0.80, -0.26), ha="center", fontsize=10, bbox=box,
                            arrowprops=arrow, zorder=6)
        axes[2, j].annotate("shear layer feeding the core:\n2-4× more ν_t with a1 = 1",
                            (0.74, -0.05), xytext=(0.70, 0.11), ha="center", fontsize=10,
                            bbox=box, arrowprops=arrow, zorder=6)
    for i, txt in enumerate(("a1 = 0.31", "a1 = 1", "ratio a1=1 / a1=0.31")):
        axes[i, 0].annotate(txt, (-0.32, 0.5), xycoords="axes fraction", rotation=90,
                            fontsize=14, fontweight="bold", va="center")
    cb = fig.colorbar(imn, ax=axes[:2, :], shrink=0.8, pad=0.01)
    cb.set_label("ν_t / ν  (log scale)")
    cb = fig.colorbar(imr, ax=axes[2, :], shrink=0.9, pad=0.01)
    cb.set_label("ν_t(a1 = 1) / ν_t(a1 = 0.31)   (red: more with a1 = 1; white area: ν_t < 10 ν in both)")
    fig.suptitle("Eddy viscosity with the default a1 = 0.31 and the team's a1 = 1 "
                 "(both converged, iteration 1000)\nBlue ring in the ratio maps: edge of the turbulent region, "
                 "where ν_t is close to the freestream value in both runs (not important).",
                 fontsize=14)
    fig.savefig(out / "4_nut_a031_vs_a1.png", dpi=130)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("label")
    ap.add_argument("label_a1", nargs="?")
    ap.add_argument("--out", type=Path, default=RESULTS / "comparisons" / "limiter_figures")
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    fig_theory(a.out)
    fig_map(a.label, a.out)
    fig_span(a.label, a.out)
    if a.label_a1:
        fig_map(a.label_a1, a.out, a1="1", fname="2b_limiter_map_a1.png",
                subtitle="Same colours as for a1 = 0.31. Where ν_t is large (tip shear layer, trailing-edge boundary layers) the limiter is now off.\n"
                         "Dark red outside: nearly freestream fluid (ν_t ≈ 10-15 ν, tiny ω), limited in both runs; with a1 = 0.31 its ν_t is < 10 ν, so it shows white there.")
        fig_compare(a.label, a.label_a1, a.out)
    print(f"written: {a.out}")


if __name__ == "__main__":
    main()
