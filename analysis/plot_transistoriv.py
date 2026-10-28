"""Drive-transistor IV from a `characterize_capacitance --what transistoriv` run.

    python analysis/plot_transistoriv.py "<..._PIXELS_Transistoriv>"

transistoriv.csv holds |I_B|(V) of shorted pixels at each bitval: 1, 2, 4, 8 are the
four drive transistors alone, 15 all four, 0 none (the leakage the others sit on).
Writes beside it:

* transistoriv.png -- top: every pixel's curve per bitval, the median bold. Bottom: the
  medians together; additivity (I15 against I1 + I2 + I4 + I8 - 3 I0, per pixel and
  bias: the leakage is in each single sweep once); the spread across pixels
  (1.4826 MAD / median) against bias; and each bitval's current over bitval 1's at the
  most negative bias, against the binary 1:2:4:8:15.
* transistoriv_summary.csv -- per bitval and bias: n, median, MAD, p10, p90 of |I|.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd


def main(run: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    d = pd.read_csv(run / "transistoriv.csv")
    d["I"] = d["i_b"].abs()
    d["V"] = -d["v"]                                   # positive: the drive voltage
    bits = sorted(d["bitval"].unique())
    g = d.groupby(["bitval", "v"])["I"]
    summ = pd.DataFrame({"n": g.size(), "median_A": g.median(),
                         "mad_A": g.apply(lambda s: 1.4826 * (s - s.median()).abs().median()),
                         "p10_A": g.quantile(0.1), "p90_A": g.quantile(0.9)}).reset_index()
    summ.to_csv(run / "transistoriv_summary.csv", index=False)
    colors = dict(zip(bits, plt.cm.viridis(np.linspace(0, 0.9, len(bits)))))
    n_px = d.groupby(["pixel_x", "pixel_y"]).ngroups

    fig = plt.figure(figsize=(3.4 * max(len(bits), 4), 8.6))
    gs = fig.add_gridspec(2, max(len(bits), 4))
    for k, b in enumerate(bits):
        a = fig.add_subplot(gs[0, k])
        db = d[d["bitval"] == b]
        for _, c in db.groupby(["pixel_x", "pixel_y"]):
            a.plot(c["V"], c["I"] * 1e6, color=colors[b], lw=0.5, alpha=0.3)
        s = summ[summ["bitval"] == b]
        a.plot(-s["v"], s["median_A"] * 1e6, color="k", lw=2)
        a.set_title(f"bitval {b}", fontsize=10)
        a.set_xlabel("|V_NVLED| (V)")
        if k == 0:
            a.set_ylabel("|I_B| (uA)")
        a.grid(alpha=0.3)

    a = fig.add_subplot(gs[1, 0])                      # medians together
    for b in bits:
        s = summ[summ["bitval"] == b]
        a.plot(-s["v"], s["median_A"] * 1e6, color=colors[b], label=f"{b}")
        a.fill_between(-s["v"], s["p10_A"] * 1e6, s["p90_A"] * 1e6, color=colors[b],
                       alpha=0.15, lw=0)
    a.set_xlabel("|V_NVLED| (V)")
    a.set_ylabel("|I_B| (uA), median, p10-p90")
    a.legend(title="bitval", fontsize=8)
    a.grid(alpha=0.3)

    a = fig.add_subplot(gs[1, 1])                      # additivity
    w = d.pivot_table(index=["pixel_x", "pixel_y", "v"], columns="bitval", values="I")
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

    a = fig.add_subplot(gs[1, 2])                      # spread across pixels
    for b in bits:
        s = summ[summ["bitval"] == b]
        a.plot(-s["v"], 100 * s["mad_A"] / s["median_A"], color=colors[b], label=f"{b}")
    a.set_xlabel("|V_NVLED| (V)")
    a.set_ylabel("spread across pixels: 1.4826 MAD / median (%)")
    a.legend(title="bitval", fontsize=8)
    a.grid(alpha=0.3)

    a = fig.add_subplot(gs[1, 3])                      # weighting at the deepest bias
    vmin = d["v"].min()
    top = d[d["v"] == vmin].pivot_table(index=["pixel_x", "pixel_y"], columns="bitval",
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
