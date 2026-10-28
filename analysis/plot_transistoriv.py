"""Drive-transistor IV from a `characterize_capacitance --what transistoriv` run.

    python analysis/plot_transistoriv.py "<..._PIXELS_Transistoriv>"

transistoriv.csv holds |I_B|(V) of shorted pixels at each bitval: 1, 2, 4, 8 are the
four drive transistors alone, 15 all four, 0 none (the leakage the others sit on).
Writes beside it:

* transistoriv.png -- top: every pixel's current per bitval, the median bold. Middle:
  the same for the optical power: whether a "short" emits at some bias. Bottom: the
  medians together; additivity (I15 against I1 + I2 + I4 + I8 - 3 I0, per pixel and
  bias: the leakage is in each single sweep once); the spread across pixels
  (1.4826 MAD / median) against bias; and each bitval's current over bitval 1's at the
  most negative bias, against the binary 1:2:4:8:15.
* transistoriv_summary.csv -- per bitval and bias: n, median, MAD, p10, p90 of |I|,
  and the median optical power.

Reads the RUN_FORMAT names (X, Y, BITVAL, NVLED_V, NVLED_CURR_*, PM400_POWER_*) and the
lowercase ones of the first runs (pixel_x, bitval, v, i_b, p_w; 2026-10-28).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd


OLD_NAMES = {"pixel_x": "X", "pixel_y": "Y", "quadrant": "QUADRANT", "bitval": "BITVAL",
             "v": "NVLED_V", "i_b": "NVLED_CURR_MEAN", "i_a": "VLED_CURR_MEAN",
             "p_w": "PM400_POWER_MEAN"}


def load(run: Path) -> pd.DataFrame:
    d = pd.read_csv(run / "transistoriv.csv").rename(columns=OLD_NAMES)
    cur = "NVLED_CURR_MEDIAN" if "NVLED_CURR_MEDIAN" in d else "NVLED_CURR_MEAN"
    d["I"] = d[cur].abs()
    d["P"] = d["PM400_POWER_MEAN"]
    d["V"] = -d["NVLED_V"]                             # positive: the drive voltage
    return d


def main(run: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    d = load(run)
    bits = sorted(d["BITVAL"].unique())
    key = ["BITVAL", "NVLED_V"]
    g = d.groupby(key)["I"]
    summ = pd.DataFrame({"n": g.size(), "median_A": g.median(),
                         "mad_A": g.apply(lambda s: 1.4826 * (s - s.median()).abs().median()),
                         "p10_A": g.quantile(0.1), "p90_A": g.quantile(0.9),
                         "P_median_W": d.groupby(key)["P"].median()}).reset_index()
    summ.to_csv(run / "transistoriv_summary.csv", index=False)
    colors = dict(zip(bits, plt.cm.viridis(np.linspace(0, 0.9, len(bits)))))
    n_px = d.groupby(["X", "Y"]).ngroups

    fig = plt.figure(figsize=(3.4 * max(len(bits), 4), 12.6))
    gs = fig.add_gridspec(3, max(len(bits), 4))
    for row, (col, scale, label, med) in enumerate((
            ("I", 1e6, "|I_B| (uA)", "median_A"),
            ("P", 1e9, "optical power (nW)", "P_median_W"))):
        for k, b in enumerate(bits):
            a = fig.add_subplot(gs[row, k])
            for _, c in d[d["BITVAL"] == b].groupby(["X", "Y"]):
                a.plot(c["V"], c[col] * scale, color=colors[b], lw=0.5, alpha=0.3)
            s = summ[summ["BITVAL"] == b]
            a.plot(-s["NVLED_V"], s[med] * scale, color="k", lw=2)
            a.set_title(f"bitval {b}", fontsize=10)
            a.set_xlabel("|V_NVLED| (V)")
            if k == 0:
                a.set_ylabel(label)
            a.grid(alpha=0.3)

    a = fig.add_subplot(gs[2, 0])                      # medians together
    for b in bits:
        s = summ[summ["BITVAL"] == b]
        a.plot(-s["NVLED_V"], s["median_A"] * 1e6, color=colors[b], label=f"{b}")
        a.fill_between(-s["NVLED_V"], s["p10_A"] * 1e6, s["p90_A"] * 1e6, color=colors[b],
                       alpha=0.15, lw=0)
    a.set_xlabel("|V_NVLED| (V)")
    a.set_ylabel("|I_B| (uA), median, p10-p90")
    a.legend(title="bitval", fontsize=8)
    a.grid(alpha=0.3)

    a = fig.add_subplot(gs[2, 1])                      # additivity
    w = d.pivot_table(index=["X", "Y", "NVLED_V"], columns="BITVAL", values="I")
    if {0, 1, 2, 4, 8, 15} <= set(w.columns):
        s1 = w[1] + w[2] + w[4] + w[8] - 3 * w[0]
        a.scatter(s1 * 1e6, w[15] * 1e6, s=4, alpha=0.4)
        hi = float(np.nanmax([s1.max(), w[15].max()])) * 1e6
        a.plot([0, hi], [0, hi], "k--", lw=0.8, label="1:1")
        ok = s1 > 0.2 * s1.max()
        r = float(np.median(w[15][ok] / s1[ok])) if ok.any() else np.nan
        a.set_title(f"additivity: median I15 / sum = {r:.3f}\n(points above 20 % of max)",
                    fontsize=9)
        a.legend(fontsize=8)
    else:
        a.set_title("additivity needs bitvals 0, 1, 2, 4, 8, 15", fontsize=9)
    a.set_xlabel("I1 + I2 + I4 + I8 - 3 I0 (uA)")
    a.set_ylabel("I15 (uA)")
    a.grid(alpha=0.3)

    a = fig.add_subplot(gs[2, 2])                      # spread across pixels
    for b in bits:
        s = summ[summ["BITVAL"] == b]
        a.plot(-s["NVLED_V"], 100 * s["mad_A"] / s["median_A"], color=colors[b], label=f"{b}")
    a.set_xlabel("|V_NVLED| (V)")
    a.set_ylabel("spread across pixels: 1.4826 MAD / median (%)")
    a.legend(title="bitval", fontsize=8)
    a.grid(alpha=0.3)

    a = fig.add_subplot(gs[2, 3])                      # weighting at the deepest bias
    vmin = d["NVLED_V"].min()
    top = d[d["NVLED_V"] == vmin].pivot_table(index=["X", "Y"], columns="BITVAL",
                                              values="I")
    if 1 in top.columns:
        # plain I / I1: the leakage under it is ~1 % of bitval 1, and taking it off
        # divides by a difference that the simulator (I0 = I1) makes zero
        rat = {b: top[b] / top[1] for b in bits if b != 0}
        a.boxplot([rat[b].dropna() for b in rat], positions=range(len(rat)),
                  tick_labels=[str(b) for b in rat], showfliers=False)
        a.plot(range(len(rat)), [b for b in rat], "r_", ms=20, mew=2,
               label="binary: = bitval")
        a.set_yscale("log")
        a.legend(fontsize=8)
    a.set_xlabel("bitval")
    a.set_ylabel(f"I / I(bitval 1) at {vmin:g} V")
    a.grid(alpha=0.3, which="both")

    fig.suptitle(f"Drive-transistor IV through shorted pixels -- {n_px} pixels "
                 f"({run.name})", fontsize=11)
    fig.tight_layout()
    fig.savefig(run / "transistoriv.png", dpi=150)
    plt.close(fig)
    print(f"wrote {run / 'transistoriv.png'} and transistoriv_summary.csv")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    main(Path(sys.argv[1]))
