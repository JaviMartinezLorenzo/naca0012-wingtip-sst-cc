#!/usr/bin/env python3
"""
wake_convergence.py: when do the forces and the wake vortex stop changing?

    python3 scripts/postprocess/wake_convergence.py <case dir> <label>
            [--window 200] [--tol-cp 0.005] [--tol-force 0.001]

Reads, from <case dir>/postProcessing (all restart folders, joined):
    wake_x*/<t>/volFieldValue.dat   min p in the slabs at the measured wake
                                    stations (controlDict, system/topoSetDict)
    forceCoeffs/<t>/coefficient.dat Cl, Cd

A signal counts as settled at iteration i when, over the window of the last
`window` iterations, its spread (max - min) is below tol * |mean|. The
"settled from" iteration is the first one after which it stays settled until
the end of the data. The case is settled when every signal is.

Writes scripts/results/<label>/:
    wake_convergence.png   Cp_min per station and Cl, Cd vs iteration, and
                           the change relative to the last value
    wake_convergence.csv   per signal: last value, spread over the last
                           window, settled-from iteration
"""

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "scripts" / "results"

U_INF = 51.816
Q_INF = 0.5 * U_INF ** 2          # kinematic dynamic pressure
CHORD = 1.2192


def read_dat(files):
    """Join OpenFOAM .dat files of several restarts; later files win."""
    frames = []
    for f in files:
        header = None
        with open(f) as fh:
            for line in fh:
                if line.startswith("#"):
                    header = line[1:].split()
                else:
                    break
        df = pd.read_csv(f, comment="#", sep=r"\s+", header=None)
        if header and len(header) == df.shape[1]:
            df.columns = header
        frames.append(df)
    if not frames:
        return None
    df = pd.concat(frames)
    df = df.rename(columns={df.columns[0]: "Time"})
    return df.drop_duplicates("Time", keep="last").sort_values("Time") \
             .set_index("Time")


def sorted_files(case, name, fname):
    d = case / "postProcessing" / name
    return sorted(d.glob(f"*/{fname}"), key=lambda p: float(p.parent.name))


def settled_from(s, window, tol):
    """First iteration after which the windowed spread stays below tol*|mean|."""
    v = s.to_numpy()
    if len(v) <= window:
        return np.nan, np.nan
    ser = pd.Series(v)
    spread = (ser.rolling(window + 1).max() - ser.rolling(window + 1).min())
    ok = spread <= tol * ser.rolling(window + 1).mean().abs()
    last_spread = spread.iloc[-1] / abs(ser.iloc[-window - 1:].mean())
    if not ok.iloc[-1]:
        return np.nan, last_spread
    bad = np.where(~ok.to_numpy()[window:])[0]
    first = window + (bad[-1] + 1 if len(bad) else 0)
    return s.index[first], last_spread


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("case", type=Path, help="OpenFOAM case dir (has postProcessing/)")
    ap.add_argument("label", help="output folder name under scripts/results/")
    ap.add_argument("--window", type=int, default=200, help="iterations (default 200)")
    ap.add_argument("--tol-cp", type=float, default=0.005,
                    help="relative spread allowed for Cp_min (default 0.005)")
    ap.add_argument("--tol-force", type=float, default=0.001,
                    help="relative spread allowed for Cl, Cd (default 0.001)")
    a = ap.parse_args()

    signals = {}                                     # name -> (series, tol)
    stations = sorted(p.name for p in (a.case / "postProcessing").glob("wake_x*"))
    for st in stations:
        df = read_dat(sorted_files(a.case, st, "volFieldValue.dat"))
        if df is None:
            continue
        col = [c for c in df.columns if "p" in c][0]
        x = int(st.split("_x")[1]) / 1000.0
        signals[f"Cp_min x/c={x / CHORD:.3f}"] = (df[col] / Q_INF, a.tol_cp)
    fc = read_dat(sorted_files(a.case, "forceCoeffs", "coefficient.dat"))
    if fc is not None:
        signals["Cl"] = (fc["Cl"], a.tol_force)
        signals["Cd"] = (fc["Cd"], a.tol_force)
    if not signals:
        sys.exit(f"no monitor data in {a.case / 'postProcessing'}")

    out = RESULTS / a.label
    out.mkdir(parents=True, exist_ok=True)

    rows = []
    for name, (s, tol) in signals.items():
        it, spread = settled_from(s, a.window, tol)
        rows.append(dict(signal=name, first_iter=int(s.index[0]),
                         last_iter=int(s.index[-1]), last_value=s.iloc[-1],
                         spread_last_window=spread, tol=tol, settled_from=it))
    table = pd.DataFrame(rows)
    table.to_csv(out / "wake_convergence.csv", index=False)

    # ---- plot ---------------------------------------------------------------
    fig, ax = plt.subplots(2, 2, figsize=(13, 8), constrained_layout=True)
    for name, (s, _) in signals.items():
        if name.startswith("Cp"):
            ax[0, 0].plot(s.index, s, lw=1.2, label=name)
            ax[1, 0].plot(s.index, 100 * (s / s.iloc[-1] - 1), lw=1.2, label=name)
    for name in ("Cl", "Cd"):
        if name in signals:
            s = signals[name][0]
            axr = ax[0, 1] if name == "Cl" else ax[0, 1].twinx()
            axr.plot(s.index, s, lw=1.2, color="C0" if name == "Cl" else "C3",
                     label=name)
            axr.set_ylabel(name)
            ax[1, 1].plot(s.index, 100 * (s / s.iloc[-1] - 1), lw=1.2,
                          color="C0" if name == "Cl" else "C3", label=name)
    ax[0, 0].set_ylabel("core Cp_min")
    ax[0, 0].set_title("Vortex core pressure at the wake stations")
    ax[0, 1].set_title("Forces")
    for a_ in (ax[1, 0], ax[1, 1]):
        a_.set_ylabel("change vs last value [%]")
        a_.set_ylim(-5, 5)
        a_.axhline(0, color="k", lw=0.6)
    for a_ in ax.flat:
        a_.set_xlabel("iteration")
        a_.grid(alpha=0.3)
    ax[0, 0].legend(fontsize=8)
    ax[1, 0].legend(fontsize=8)
    ax[1, 1].legend(fontsize=8)
    fig.suptitle(f"{a.label}: window {a.window} it, tol Cp {a.tol_cp:.1%}, "
                 f"forces {a.tol_force:.1%}")
    fig.savefig(out / "wake_convergence.png", dpi=130)

    # ---- report -------------------------------------------------------------
    pd.set_option("display.width", 120)
    print(table.to_string(index=False, float_format=lambda v: f"{v:.5g}"))
    if table["settled_from"].notna().all():
        print(f"\nALL SETTLED from iteration {int(table['settled_from'].max())}")
    else:
        left = ", ".join(table.loc[table["settled_from"].isna(), "signal"])
        print(f"\nNOT SETTLED yet: {left}")
    print(f"written: {out}")


if __name__ == "__main__":
    main()
