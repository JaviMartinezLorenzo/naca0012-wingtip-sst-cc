#!/usr/bin/env python3
"""
compare_wing_cp.py: wing surface Cp of several runs against the experiment.

    python3 scripts/postprocess/compare_wing_cp.py <label> [<label> ...]
            [--names "SST a1=0.31" "SST a1=1"] [--out DIR]

Each <label> is a folder scripts/results/runs/<label>/ with wing_cp.csv from
pv_wing_cp.py. The experiment is CHORDQPL.DAT (24 tap rows: Z = 6..32 in on
the constant-chord part, theta = 10..90 deg on the rounded tip; upper and
lower surface).

Writes to DIR (default scripts/results/comparisons/compare_wing_<labels>/):
    wing_cp_span.png   Cp vs x/c at the 7 span stations
    wing_cp_tip.png    Cp vs x/c at the 5 tip angles
    wing_cp_cn.png     section normal-force coefficient c_n vs span
    wing_cp_error.csv  per tap row and run: mean |Cp_CFD - Cp_exp| (upper,
                       lower), c_n from the experiment and the CFD
"""

import argparse
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "scripts" / "results"
RUNS = RESULTS / "runs"                 # one folder per run (pv_*.py output)
EXP = ROOT / "Wingtip_expdata" / "CHORDQPL.DAT"

# Block order in CHORDQPL.DAT (README3.DAT)
BLOCKS = ([("span", z, s) for z in (6, 12, 24, 28, 30, 31, 32) for s in ("lower", "upper")]
          + [("tip", t, s) for t in (10, 22.5, 45, 67.5, 90) for s in ("lower", "upper")])
SPAN_IN = 36.0                 # semispan (tip at 33.12 + 0.06 c ~ 36 in)
RUN_COLORS = ["#1f6fb4", "#d1495b", "#2a9d8f", "#e9a03b"]


def read_exp():
    text = EXP.read_text()
    chunks = text.split("$OPTIONS")[1:]
    rows = []
    num = re.compile(r"^\s*(\d+)\s+([-\d.]+)\s+([-\d.]+)\s+([-\d.]+)")
    for (kind, st, side), ch in zip(BLOCKS, chunks):
        body = ch.split("$END", 1)[1]
        for line in body.splitlines():
            m = num.match(line)
            if m:
                rows.append((kind, float(st), side, float(m.group(2)), float(m.group(4))))
    return pd.DataFrame(rows, columns=["kind", "station", "side", "x_c", "cp"])


def cn(df_side_upper, df_side_lower):
    """Section normal-force coefficient: integral of (Cp_lower - Cp_upper) d(x/c)."""
    xs = np.linspace(0.0, 1.0, 401)
    up = np.interp(xs, df_side_upper.x_c, df_side_upper.cp)
    lo = np.interp(xs, df_side_lower.x_c, df_side_lower.cp)
    return np.trapezoid(lo - up, xs)


def station_label(kind, st):
    return f"Z = {st:g} in  (z/b = {st / SPAN_IN:.2f})" if kind == "span" else f"tip, θ = {st:g}°"


def plot_set(exp, runs, names, kind, path, title):
    stations = sorted(exp.loc[exp.kind == kind, "station"].unique())
    ncol = 4 if kind == "span" else 3
    nrow = int(np.ceil(len(stations) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.4 * ncol, 3.7 * nrow),
                             constrained_layout=True, squeeze=False)
    for ax, st in zip(axes.flat, stations):
        for side, mk in (("upper", "o"), ("lower", "s")):
            e = exp[(exp.kind == kind) & (exp.station == st) & (exp.side == side)]
            ax.plot(e.x_c, e.cp, mk, ms=5, mfc="k" if side == "upper" else "white",
                    mec="k", mew=1.0, ls="none", zorder=5,
                    label=f"experiment, {side}")
        for (lab, df), name, col in zip(runs.items(), names, RUN_COLORS):
            for side, ls in (("upper", "-"), ("lower", "--")):
                c = df[(df.kind == kind) & (df.station == st) & (df.side == side)].sort_values("x_c")
                ax.plot(c.x_c, c.cp, ls, color=col, lw=1.6, label=f"{name}, {side}")
        ax.invert_yaxis()
        ax.set_xlim(-0.02, 1.02)
        ax.set_title(station_label(kind, st), fontsize=11)
        ax.set_xlabel("x/c")
        ax.set_ylabel("Cp")
        ax.grid(alpha=0.3)
        ax.axhline(0, color="0.4", lw=0.6)
    used = len(stations)
    for ax in axes.flat[used:]:
        ax.axis("off")
    handles, labels = axes.flat[0].get_legend_handles_labels()
    if used < nrow * ncol:
        axes.flat[used].legend(handles, labels, loc="center", fontsize=10, frameon=False)
    else:
        fig.legend(handles, labels, loc="lower center", ncol=len(labels), fontsize=9)
    fig.suptitle(title, fontsize=13)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("labels", nargs="+")
    ap.add_argument("--names", nargs="+", default=None, help="legend names, one per label")
    ap.add_argument("--out", type=Path, default=None)
    a = ap.parse_args()
    names = a.names or a.labels
    out = a.out or RESULTS / "comparisons" / ("compare_wing_" + "_vs_".join(a.labels))
    out.mkdir(parents=True, exist_ok=True)

    exp = read_exp()
    runs = {lab: pd.read_csv(RUNS / lab / "wing_cp.csv") for lab in a.labels}

    plot_set(exp, runs, names, "span", out / "wing_cp_span.png",
             "Wing surface Cp at the span stations (constant-chord part)")
    plot_set(exp, runs, names, "tip", out / "wing_cp_tip.png",
             "Wing surface Cp on the rounded tip (θ = 0° at its start, 90° at the very tip)")

    # ---- error table and c_n --------------------------------------------------
    rows = []
    for (kind, st), _ in exp.groupby(["kind", "station"]):
        e = exp[(exp.kind == kind) & (exp.station == st)]
        eu, el = (e[e.side == s].sort_values("x_c") for s in ("upper", "lower"))
        row = dict(kind=kind, station=st, cn_exp=cn(eu, el))
        for lab, name in zip(a.labels, names):
            c = runs[lab]
            c = c[(c.kind == kind) & (c.station == st)]
            cu, cl = (c[c.side == s].sort_values("x_c") for s in ("upper", "lower"))
            for side, es, cs in (("upper", eu, cu), ("lower", el, cl)):
                m = es.x_c > 0.0                      # skip the shared LE tap
                row[f"mae_{side}_{lab}"] = np.mean(np.abs(
                    np.interp(es.x_c[m], cs.x_c, cs.cp) - es.cp[m]))
            row[f"cn_{lab}"] = cn(cu, cl)
        rows.append(row)
    tab = pd.DataFrame(rows).sort_values(["kind", "station"], ascending=[False, True])
    tab.to_csv(out / "wing_cp_error.csv", index=False, float_format="%.4f")

    span = tab[tab.kind == "span"]
    fig, ax = plt.subplots(figsize=(7, 4.2), constrained_layout=True)
    ax.plot(span.station / SPAN_IN, span.cn_exp, "ko", ms=7, label="experiment")
    for lab, name, col in zip(a.labels, names, RUN_COLORS):
        ax.plot(span.station / SPAN_IN, span[f"cn_{lab}"], "-o", color=col, ms=4, label=name)
    ax.set_xlabel("z / b  (span station / semispan)")
    ax.set_ylabel("section normal force  c_n = ∫(Cp,lower − Cp,upper) d(x/c)")
    ax.set_title("Spanwise loading from the surface pressure")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.savefig(out / "wing_cp_cn.png", dpi=150)
    plt.close(fig)

    pd.set_option("display.width", 200)
    print(tab.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    print(f"written: {out}")


if __name__ == "__main__":
    main()
