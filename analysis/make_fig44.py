"""Figure 44: Array38's emission map in July (capacitors on) and October (off).

    python analysis/make_fig44.py

Fast scans at -1.35 V over all 262 144 pixels: results_georg 260706-1114 and
results 261022-0037. Marked: targets that were lit in the fast scan but drew only
leakage current at -2.60 V, the first bias of each sweep, and so kept one bias.
"""
import collections
import csv
import glob
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

ROOT = Path(__file__).resolve().parent.parent
OUT = Path(__file__).resolve().parent / "investigation_report"   # as make_capacitance_figures.py: never CWD-relative


def _f(s):
    try:
        return float(s)
    except (TypeError, ValueError):
        return float("nan")


def load(d):
    fs = [p for p in glob.glob(str(d / "FastScan" / "*aggregated*.csv")) if "screen" not in p][0]
    img = np.full((512, 512), np.nan)
    for r in csv.DictReader(open(fs, encoding="utf-8")):
        img[int(_f(r["Y"])), int(_f(r["X"]))] = abs(_f(r["PM400_POWER_MEAN"]))
    sw = (glob.glob(str(d / "NVLEDSweep" / "*" / "*aggregated*mean_std.csv"))
          + glob.glob(str(d / "NVLEDSweep" / "*aggregated*mean_std.csv")))[0]
    n = collections.Counter((int(_f(r["X"])), int(_f(r["Y"])))
                            for r in csv.DictReader(open(sw, encoding="utf-8")))
    return img, [p for p, k in n.items() if k == 1], len(n)


runs = [(sorted(ROOT.glob("results_georg/results/260706-1114_Array38_IQE_ROI_(0, 0, 511, 511)"))[0],
         "6 July 2026, capacitors on (260706-1114)"),
        (sorted(ROOT.glob("results/261022-0037_*"))[0],
         "22 October 2026, capacitors off (261022-0037)")]
fig, ax = plt.subplots(1, 2, figsize=(13.5, 6.6))
for a, (d, title) in zip(ax, runs):
    img, drop, ntgt = load(d)
    p90 = np.nanpercentile(img, 90)
    alive = np.nanmean(img >= 0.2 * p90) * 100
    im = a.imshow(np.clip(img, 1e-11, None) * 1e9, norm=LogNorm(vmin=0.01, vmax=p90 * 1.5e9),
                  cmap="magma", origin="upper", interpolation="nearest")
    if drop:
        xs, ys = zip(*drop)
        a.scatter(xs, ys, s=14, facecolors="none", edgecolors="cyan", linewidths=0.9,
                  label=f"lit at -1.35 V, dark at -2.60 V: {len(drop)} of {ntgt} targets")
    a.axhline(255.5, color="w", lw=0.5, alpha=0.5); a.axvline(255.5, color="w", lw=0.5, alpha=0.5)
    a.set_title(f"{title}\n{alive:.1f} % alive at -1.35 V (>= 20 % of the 90th percentile)",
                fontsize=9.5)
    a.set_xlabel("X"); a.set_ylabel("Y")
    a.legend(fontsize=7.5, loc="lower left")
    fig.colorbar(im, ax=a, fraction=0.046, pad=0.02, label="fast-scan power at -1.35 V  [nW]")
fig.suptitle("Array38: the dark region and the -2.60 V dropouts are the same before and after "
             "the capacitor removal", fontsize=10.5)
fig.tight_layout()
OUT.mkdir(exist_ok=True)
fig.savefig(OUT / "fig44_array38_yield.png", dpi=300)
print("wrote", OUT / "fig44_array38_yield.png")
