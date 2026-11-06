"""Is the displayed image (smile_s.jpg) burned into Array42, and in what: EQE, current, R?

    python analysis/image_burnin.py        (after bond_resistance.py)

The image goes through send_image's own processing (256x256, 16 levels) and is placed on
each quadrant as the hardware mirrors it: TL as sent, TR left-right, BL top-bottom, BR both
(the TL, TR, BR fits to the 2 Nov EQE pick exactly these; BL is too weak to choose).

Test, per fast scan: in every 32x32 block, the median of the pixels driven bright (level
>= 12) minus those driven dark (<= 3); the median over blocks. Block-matched, so the wafer
pattern, the edge gradients and any uniform shift (bias, capacitors, offsets) cancel.
Quantities: EQE (P/I), current at the scan bias, R_best (bond_resistance.py), and the LED
anchor (V at 1.5 uA in the reference sweep; lower = more current at low injection).

Writes results/comparisons/image_burnin/: summary.csv, maps.png; caches the per-scan maps
in cache/ (loading a full scan takes minutes).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.ndimage import median_filter, zoom

import analyze_array as aa
import bond_resistance as br

sys.path.insert(0, str(br.REPO))
from smile.utils.image_processor import ImageProcessor  # noqa: E402

OUT = br.REPO / "results" / "comparisons" / "image_burnin"
IMAGE = br.REPO / "smile_s.jpg"
RUNS = {"16 Oct": ("261016-2252_Array42_IQE_ROI_(0, 0, 511, 511)", -1.35, "261016-2252_m1p35V"),
        "19 Oct": ("261019-2342_Array42_IQE_ROI_(0, 0, 511, 511)", -1.35, "261019-2342_m1p35V"),
        "20 Oct": ("261020-1454_Array42_IQE_ROI_(0, 0, 511, 511)", -1.35, "261020-1454_m1p35V"),
        "2 Nov": ("261102-1703_Array42_ROI_(0, 0, 511, 511)", -1.4, "261102-1703_m1p40V")}
BRIGHT, DARK = 12, 3


def img(s):
    m = np.full((512, 512), np.nan)
    m[s.index.get_level_values("Y"), s.index.get_level_values("X")] = s.to_numpy()
    return m


def scan_maps(k):
    f = OUT / "cache" / f"{RUNS[k][2]}.npz"
    if f.exists():
        return dict(np.load(f))
    run, V, tag = RUNS[k]
    fs = aa._load_aggregated(br.run_dir(run) / "FastScan")
    fs = fs[fs.NVLED_V.astype(float).round(3) == V]
    d = aa.classify_pixels(fs).set_index(["X", "Y"])
    d["I_LED"] = d.I_A - d.I_leak_A
    p = pd.read_csv(br.OUT / "pixels" / f"{tag}.csv").set_index(["X", "Y"])
    d = d.loc[p.index].join(p.R_best_kOhm)
    out = {"eqe": img(d.P_W / d.I_LED), "I": img(d.I_LED), "R": img(d.R_best_kOhm)}
    f.parent.mkdir(parents=True, exist_ok=True)
    np.savez(f, **out)
    return out


def image_on_array():
    p = ImageProcessor(str(IMAGE))
    p.process(grayscale_levels=16, canvas_width=256, canvas_height=256)
    g = np.array(p._img, dtype=float).reshape(256, 256)
    return np.block([[g, np.fliplr(g)], [np.flipud(g), np.rot90(g, 2)]])


def bright_minus_dark(m, level):
    Y, X = np.mgrid[0:512, 0:512]
    df = pd.DataFrame({"b": (Y // 32 * 16 + X // 32).ravel(), "g": level.ravel(),
                       "v": m.ravel()}).dropna()
    iqr = df.v.quantile(.75) - df.v.quantile(.25)
    df = df[np.abs(df.v - df.v.median()) < 5 * iqr]
    d = (df[df.g >= BRIGHT].groupby("b").v.median()
         - df[df.g <= DARK].groupby("b").v.median()).dropna()
    # standard error of the median over blocks
    return d.median(), 1.25 * d.std() / np.sqrt(len(d)), len(d)


def detrend(m, rel):
    """For the maps only: 32x32 block medians, smoothly upsampled, removed."""
    b = np.nanmedian(m.reshape(16, 32, 16, 32).transpose(0, 2, 1, 3).reshape(16, 16, -1), axis=2)
    bg = zoom(np.where(np.isfinite(b), b, np.nanmedian(b)), 32, order=1)
    return m / bg - 1 if rel else m - bg


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    plt = aa._plt()
    level = image_on_array()
    M = {k: scan_maps(k) for k in RUNS}
    fa = OUT / "cache" / "anchor_Array42.npy"
    if not fa.exists():
        np.save(fa, img(br.best_reference("Array42")[0] * 1e3))
    M["2 Nov"]["anchor"] = np.load(fa)
    rows = []
    for k, m in M.items():
        for q, rel, unit in (("eqe", True, "rel."), ("I", True, "rel."), ("R", False, "kOhm"),
                             ("anchor", False, "mV")):
            if q in m:
                v = m[q] / np.nanmedian(m[q]) - 1 if rel else m[q]
                med, se, n = bright_minus_dark(v, level)
                rows.append(dict(scan=k, quantity=q, unit=unit, bright_minus_dark=med, se=se,
                                 blocks=n))
    t = pd.DataFrame(rows)
    t.to_csv(OUT / "summary.csv", index=False, float_format="%.5g")
    print(t.to_string(index=False, float_format=lambda x: f"{x:+.4f}"))
    fig, ax = plt.subplots(3, 5, figsize=(30, 18))
    for i, (q, rel, lim, lab) in enumerate((("eqe", True, .06, "EQE, rel."),
                                            ("I", True, .01, "current, rel."),
                                            ("R", False, .15, "R_best, kOhm"))):
        for j, k in enumerate(RUNS):
            x = ax[i, j]
            im = x.imshow(median_filter(np.nan_to_num(detrend(M[k][q], rel)), size=5),
                          cmap="RdBu_r", vmin=-lim, vmax=lim, interpolation="nearest")
            r = t[(t.scan == k) & (t.quantity == q)].iloc[0]
            x.set_title(f"{k}: {lab} (32 px background removed, 5 px median)\nbright - dark "
                        f"{r.bright_minus_dark:+.4f} +- {r.se:.4f}", fontsize=10)
            fig.colorbar(im, ax=x, shrink=0.8)
        ax[i, 4].imshow(level, cmap="gray")
        ax[i, 4].set_title("smile_s.jpg as displayed: drive level 0-15 per pixel", fontsize=10)
    fig.suptitle("Array42: the displayed image in EQE, current and series resistance "
                 "(block-matched bright minus dark)", fontsize=13)
    fig.tight_layout()
    fig.savefig(OUT / "maps.png", dpi=80)
    plt.close(fig)
    return 0


if __name__ == "__main__":
    sys.exit(main())
