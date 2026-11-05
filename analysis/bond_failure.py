"""Does the bond resistance predict which pixels die?

    python analysis/bond_failure.py

Pairs of full-array fast scans of one array, earlier -> later. Every pixel alive in the
earlier scan (in bond_resistance.py's selection, with its R_bond) is followed into the
later scan's class: still lit, or dark (open, short, dark_conducting). The death rate is
binned by the earlier R_bond, and by the distance to the nearest pixel that was already
dark in the earlier scan (Chebyshev, so the 8 neighbours are 1).

Writes results/comparisons/bond_resistance/failure/: failure.png, failure_by_R.csv,
failure_by_distance.csv, died.csv (the pixels that died, with their earlier R_bond).
"""
from __future__ import annotations

import sys

import numpy as np
import pandas as pd
from scipy.ndimage import distance_transform_cdt

import analyze_array as aa
import bond_resistance as br

PAIRS = [("Array38", "260706-1114", "261022-0037"),     # July (caps) -> October (no caps)
         ("Array38", "261022-0037", "261026-0813"),
         ("Array42", "261020-1454", "261102-1703")]     # caps -> no caps
DARK = ("open", "short", "dark_conducting")


def classes(run, V):
    fs = aa._load_aggregated(br.run_dir(run) / "FastScan")
    fs = fs[fs.NVLED_V.astype(float).round(3) == round(V, 3)]
    return aa.classify_pixels(fs).set_index(["X", "Y"]).primary


def main() -> int:
    out = br.OUT / "failure"
    out.mkdir(parents=True, exist_ok=True)
    plt = aa._plt()
    runs = pd.read_csv(br.OUT / "fastscan_runs.csv")
    fig, ax = plt.subplots(2, len(PAIRS), figsize=(5.5 * len(PAIRS), 9), squeeze=False)
    rows_r, rows_d, died_all = [], [], []
    for j, (arr, a, b) in enumerate(PAIRS):
        ra = runs[runs.run.str.startswith(a) & (runs.V.abs() <= 1.7)].iloc[0]
        rb_ = runs[runs.run.str.startswith(b) & (runs.V.abs() <= 1.7)].iloc[0]
        pa = pd.read_csv(br.OUT / "pixels" / f"{a}_{aa._bias_tag(ra.V)}.csv").set_index(["X", "Y"])
        ca, cb = classes(ra.run, ra.V), classes(rb_.run, rb_.V)
        later = cb.reindex(pa.index)
        pa["died"] = later.isin(DARK)
        pa = pa[later.notna()]
        # distance to the nearest pixel dark in the EARLIER scan
        dark = np.zeros((512, 512), bool)
        idx = ca[ca.isin(DARK)].index
        dark[idx.get_level_values("Y"), idx.get_level_values("X")] = True
        dist = distance_transform_cdt(~dark, metric="chessboard")
        pa["dist_dark"] = dist[pa.index.get_level_values("Y"), pa.index.get_level_values("X")]
        n, nd = len(pa), int(pa.died.sum())
        base = nd / n
        print(f"{arr} {a} -> {b}: {nd} of {n} pixels died ({100 * base:.3f} %)")
        # by R_bond decile
        q = pd.qcut(pa.R_best_kOhm, 10, duplicates="drop")
        g = pa.groupby(q, observed=True)
        t = pd.DataFrame({"R_median_kOhm": g.R_best_kOhm.median(), "n": g.size(),
                          "died": g.died.sum()})
        t["rate_pct"] = 100 * t.died / t.n
        t = t.reset_index(drop=True).assign(array=arr, earlier=a, later=b)
        rows_r.append(t)
        x = ax[0, j]
        x.plot(t.R_median_kOhm, t.rate_pct, "o-")
        x.axhline(100 * base, color="0.5", ls=":", lw=1, label=f"all: {100 * base:.3f} %")
        x.set_xscale("symlog", linthresh=1)
        x.set_xlabel(f"bond resistance above the best pixels, in {a} (kOhm), deciles")
        x.set_ylabel(f"died by {b} (%)")
        x.set_title(f"{arr}: {a} -> {b}, {nd} of {n} died", fontsize=10)
        x.grid(alpha=0.3, which="both")
        x.legend(fontsize=8)
        # by distance to the dark region
        bins = [0, 1, 2, 3, 5, 10, 20, 50, 1000]
        c = pd.cut(pa.dist_dark, bins, right=True)
        g = pa.groupby(c, observed=True)
        u = pd.DataFrame({"n": g.size(), "died": g.died.sum(),
                          "R_median_kOhm": g.R_best_kOhm.median()})
        u["rate_pct"] = 100 * u.died / u.n
        u.index = [f"{iv.left + 1}-{iv.right}" if iv.right - iv.left > 1 else f"{iv.right}"
                   for iv in u.index]
        rows_d.append(u.reset_index(names="dist_px").assign(array=arr, earlier=a, later=b))
        x = ax[1, j]
        x.bar(range(len(u)), u.rate_pct, color="tab:red", alpha=0.7)
        x.set_xticks(range(len(u)), u.index, rotation=30)
        x.set_xlabel(f"distance to the nearest pixel dark in {a} (px)")
        x.set_ylabel(f"died by {b} (%)")
        x2 = x.twinx()
        x2.plot(range(len(u)), u.R_median_kOhm, "ko-", ms=3)
        x2.set_ylabel(f"median R_bond in {a} (kOhm)")
        x.set_title(f"{arr}: by distance to the dark region", fontsize=10)
        x.grid(alpha=0.3)
        died_all.append(pa[pa.died].reset_index()[["X", "Y", "quadrant", "R_best_kOhm",
                                                   "dist_dark"]].assign(array=arr, earlier=a,
                                                                        later=b))
    fig.suptitle("Does the bond resistance predict which pixels die? Death rate between two "
                 "full-array fast scans", fontsize=11)
    fig.tight_layout()
    fig.savefig(out / "failure.png", dpi=150)
    plt.close(fig)
    pd.concat(rows_r).to_csv(out / "failure_by_R.csv", index=False, float_format="%.4g")
    pd.concat(rows_d).to_csv(out / "failure_by_distance.csv", index=False, float_format="%.4g")
    pd.concat(died_all).to_csv(out / "died.csv", index=False, float_format="%.4g")
    for t in rows_r:
        print(t[["R_median_kOhm", "n", "died", "rate_pct"]].round(3).to_string(index=False))
    for u in rows_d:
        print(u[["dist_px", "n", "died", "rate_pct", "R_median_kOhm"]].round(3).to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
