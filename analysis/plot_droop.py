"""Current and light against the number of lit pixels, from `--what droop`.

    python analysis/plot_droop.py "<..._PIXELS_Droop>"

droop.csv holds, per bias and lighting order (random: spread over the ROI; patch: a
disc growing from its centre), |I_B| and P with N = 1, 2, 3, ... pixels lit at once.
Writes droop.png beside it, one row per bias:

* total |I_B| and P against N, with N x the N = 1 value dashed: where the curve
  falls below the line, adding a pixel adds less than one pixel's worth;
* per pixel, I/N and P/N over their N = 1 value: 1.0 while every pixel gets its full
  drive, falling as the shared path (the n-GaN sheet resistance) takes the voltage;
* P/I over its N = 1 value: whether the light falls with the current or faster.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd


OLD_NAMES = {"v": "NVLED_V", "order": "ORDER", "n": "N_LIT", "i_b": "NVLED_CURR_MEAN",
             "i_a": "VLED_CURR_MEAN", "p_w": "PM400_POWER_MEAN", "x_added": "X_ADDED",
             "y_added": "Y_ADDED", "quadrant": "QUADRANT"}


def main(run: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # the RUN_FORMAT names, or the lowercase ones of the first runs (2026-10-28)
    d = pd.read_csv(run / "droop.csv").rename(columns=OLD_NAMES)
    cur = "NVLED_CURR_MEDIAN" if "NVLED_CURR_MEDIAN" in d else "NVLED_CURR_MEAN"
    d["I"], d["P"], d["n"] = d[cur].abs(), d["PM400_POWER_MEAN"].abs(), d["N_LIT"]
    volts = sorted(d["NVLED_V"].unique())
    style = {"random": "-", "patch": "--"}
    fig, ax = plt.subplots(len(volts), 4, figsize=(19, 4.2 * len(volts)), squeeze=False)
    for row, v in zip(ax, volts):
        for o, g in d[d["NVLED_V"] == v].groupby("ORDER"):
            g = g.sort_values("n")
            i1, p1 = g["I"].iloc[0] / g["n"].iloc[0], g["P"].iloc[0] / g["n"].iloc[0]
            ls = style.get(o, ":")
            row[0].plot(g["n"], g["I"] * 1e3, ls, label=o)
            row[0].plot(g["n"], g["n"] * i1 * 1e3, ls, color="0.6", lw=0.8)
            row[1].plot(g["n"], g["P"] * 1e6, ls, label=o)
            row[1].plot(g["n"], g["n"] * p1 * 1e6, ls, color="0.6", lw=0.8)
            row[2].plot(g["n"], g["I"] / g["n"] / i1, ls, label=f"I/N, {o}")
            row[2].plot(g["n"], g["P"] / g["n"] / p1, ls, label=f"P/N, {o}")
            row[3].plot(g["n"], (g["P"] / g["I"]) / (p1 / i1), ls, label=o)
        row[0].set_ylabel(f"{v:+.2f} V\n|I_B| total (mA)")
        row[1].set_ylabel("P total (uW)")
        row[2].set_ylabel("per pixel, / its N=1 value")
        row[3].set_ylabel("P/I, / its N=1 value")
        row[0].set_title("current (grey: N x one pixel)", fontsize=9)
        row[1].set_title("light (grey: N x one pixel)", fontsize=9)
        row[2].set_title("per-pixel current and light", fontsize=9)
        row[3].set_title("efficiency", fontsize=9)
        for a in row:
            a.set_xlabel("N lit pixels")
            a.grid(alpha=0.3)
            a.legend(fontsize=8)
    fig.suptitle(f"Droop with the number of lit pixels ({run.name})", fontsize=11)
    fig.tight_layout()
    fig.savefig(run / "droop.png", dpi=150)
    plt.close(fig)
    print(f"wrote {run / 'droop.png'}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    main(Path(sys.argv[1]))
