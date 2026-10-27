"""The leak matrix of a characterize_capacitance `leakmatrix` run, as one figure.

    python analysis/plot_leakmatrix.py "<..._CAPACITANCE_Leakmatrix>"

leakmatrix.csv holds the cathode current with the LED off (0 V to -0.4 V) for every
bias x frame state x quadrant: the pixel at bitval 15 and 0, and a blank frame, each
with the pixel's own quadrant enabled ("own"), all four, or none. Writes
leakmatrix.png beside it: one panel per bias, the median |I| per state, a bar per
quadrant, the burst SD as the error bar, log scale.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

QUAD_COLORS = {"TL": "tab:blue", "TR": "tab:orange", "BL": "tab:green", "BR": "tab:red"}


def main(run: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    d = pd.read_csv(run / "leakmatrix.csv")
    states = list(dict.fromkeys(d["state"]))          # run order
    quads = [q for q in QUAD_COLORS if q in set(d["quadrant"])]
    volts = sorted(d["v"].unique(), reverse=True)
    w = 0.8 / len(quads)
    fig, ax = plt.subplots(1, len(volts), figsize=(5.2 * len(volts), 4.6), sharey=True,
                           squeeze=False)
    for a, v in zip(ax[0], volts):
        dv = d[d["v"] == v]
        for k, q in enumerate(quads):
            r = dv[dv["quadrant"] == q].set_index("state").reindex(states)
            i = r["NVLED_CURR_MEDIAN"].abs() * 1e9
            a.bar(np.arange(len(states)) + (k - (len(quads) - 1) / 2) * w, i, w,
                  yerr=r["NVLED_CURR_SAMPSTD"] * 1e9, color=QUAD_COLORS[q], capsize=2,
                  label=f"{q} ({int(r['x'].iloc[0])},{int(r['y'].iloc[0])})")
        a.set_xticks(np.arange(len(states)), states, rotation=45, ha="right", fontsize=8)
        a.set_yscale("log")
        a.set_title(f"{v:+.2f} V", fontsize=10)
        a.grid(axis="y", which="both", alpha=0.3)
    ax[0][0].set_ylabel("|I_NVLED| median (nA)")
    ax[0][0].legend(title="pixel's quadrant", fontsize=8)
    fig.suptitle(f"Leak matrix -- LED off: cathode current per frame state and quadrant "
                 f"enables ({run.name})", fontsize=10)
    fig.tight_layout()
    fig.savefig(run / "leakmatrix.png", dpi=200)
    plt.close(fig)
    print(f"wrote {run / 'leakmatrix.png'}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    main(Path(sys.argv[1]))
