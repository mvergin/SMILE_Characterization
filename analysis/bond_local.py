"""Local reference: each pixel against the best pixels of its own 32x32 block.

    python analysis/bond_local.py        (after bond_resistance.py)

R_best (bond_resistance.py) is measured against the array's best 2 % of pixels. Those sit
mostly 4-60 px in from the array edge, so R_best carries the position. Per fast scan:
  position  = the 32x32 block's median R_best
  R_local   = R_best - the block's 5th percentile (its own best pixels); blocks >= 100 pixels
  residual  = R_best - block median: the pixel-to-pixel part

Writes results/comparisons/bond_resistance/local/:
  maps_<array>.png    reference scan: R_best, block median, block best, residual
  best_location.png   where the best pixels sit; R_best and R_local against the edge distance
  summary.md          per scan: position spread against pixel-to-pixel spread
"""
from __future__ import annotations

import sys

import numpy as np
import pandas as pd

import analyze_array as aa
import bond_resistance as br

B, Q, N_MIN = 32, 0.05, 100
EDGE_BINS = [-1, 3, 7, 15, 31, 63, 127, 255]


def add_local(d):
    """Block median, block best (5th percentile), R_local, residual and edge distance."""
    d = d.copy()
    g = d.groupby([d.X // B, d.Y // B]).R_best_kOhm
    big = g.transform("count") >= N_MIN
    d["block_median_kOhm"] = g.transform("median").where(big)
    d["block_best_kOhm"] = g.transform(lambda s: s.quantile(Q)).where(big)
    d["R_local_kOhm"] = d.R_best_kOhm - d.block_best_kOhm
    d["residual_kOhm"] = d.R_best_kOhm - d.block_median_kOhm
    d["edge_px"] = np.minimum.reduce([d.X, 511 - d.X, d.Y, 511 - d.Y])
    return d


def block_img(d, col):
    img = np.full((512 // B, 512 // B), np.nan)
    b = d.groupby([d.Y // B, d.X // B])[col].first().dropna()
    img[b.index.get_level_values(0), b.index.get_level_values(1)] = b.to_numpy()
    return img


def pixel_img(d, col):
    img = np.full((512, 512), np.nan)
    img[d.Y.to_numpy(), d.X.to_numpy()] = d[col].to_numpy()
    return img


def edge_profile(d, col):
    g = d.groupby(pd.cut(d.edge_px, EDGE_BINS), observed=True)[col]
    return g.median(), g.quantile(.25), g.quantile(.75)


def clim(d):
    """Colour range from the block medians: Array38's tail (90 % at 18 kOhm) and its
    20-50 kOhm edge blocks would wash out the rest."""
    bm = d.groupby([d.X // B, d.Y // B]).block_median_kOhm.first().dropna()
    return float(bm.quantile(.02)), float(bm.quantile(.9))


def bin_labels():
    return [f"{a + 1}-{b}" for a, b in zip(EDGE_BINS, EDGE_BINS[1:])]


def main() -> int:
    out = br.OUT / "local"
    out.mkdir(parents=True, exist_ok=True)
    plt = aa._plt()
    runs = pd.read_csv(br.OUT / "runs.csv")
    L = ["# Local reference: each pixel against its 32x32 block's best (bond_local.py)", "",
         "All in kOhm. Position: spread of the 32x32 block medians. Pixel-to-pixel: R_best minus "
         "its block median. Block best: the block's 5th percentile, against the array's best.", "",
         "| scan | array | pixels | R_best median | block best median | R_local median | "
         "position 10-90 % | pixel-to-pixel 25-75 % | pixel-to-pixel 10-90 % |", "|" + "---|" * 9]
    ref = {}
    for _, r in runs.iterrows():
        d = add_local(pd.read_csv(br.OUT / "pixels" / f"{r.tag}.csv"))
        d.to_csv(out / f"{r.tag}.csv", index=False, float_format="%.5g")
        bm = d.groupby([d.X // B, d.Y // B]).block_median_kOhm.first().dropna()
        bb = d.groupby([d.X // B, d.Y // B]).block_best_kOhm.first().dropna()
        rs = d.residual_kOhm.dropna()
        L.append(f"| {r.tag} | {r['array']} | {len(d)} | {d.R_best_kOhm.median():.2f} | "
                 f"{bb.median():.2f} | {d.R_local_kOhm.median():.2f} | "
                 f"{bm.quantile(.1):.2f} to {bm.quantile(.9):.2f} | "
                 f"{rs.quantile(.25):+.2f} to {rs.quantile(.75):+.2f} | "
                 f"{rs.quantile(.1):+.2f} to {rs.quantile(.9):+.2f} |")
        if r.run.startswith(br.REFERENCE_RUN[r["array"]]):
            ref[r["array"]] = (r, d)
    (out / "summary.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))

    # maps of the reference scans: what is position, what is pixel-to-pixel
    cm = plt.get_cmap("magma").copy()
    cm.set_bad("0.75")
    dv = plt.get_cmap("RdBu_r").copy()
    dv.set_bad("0.75")
    for a, (r, d) in ref.items():
        lo, hi = clim(d)
        rr = float(d.residual_kOhm.abs().quantile(.75))
        fig, ax = plt.subplots(2, 2, figsize=(13, 12))
        panels = ((pixel_img(d, "R_best_kOhm"), cm, lo, hi,
                   "R_best: above the array's best 2 %, every pixel"),
                  (block_img(d, "block_median_kOhm"), cm, lo, hi,
                   "position: 32x32 block median of R_best"),
                  (block_img(d, "block_best_kOhm"), cm, lo, hi,
                   "local reference: the block's best (5th percentile of R_best)"),
                  (pixel_img(d, "residual_kOhm"), dv, -rr, rr,
                   "pixel-to-pixel: R_best - block median"))
        for x, (img, c, vmin, vmax, ttl) in zip(ax.flat, panels):
            im = x.imshow(img, cmap=c, vmin=vmin, vmax=vmax, extent=(-0.5, 511.5, 511.5, -0.5),
                          interpolation="nearest")
            x.set_title(ttl, fontsize=10)
            fig.colorbar(im, ax=x, label="kOhm", shrink=0.85)
        fig.suptitle(f"{a}, {r.tag}: R_best median {d.R_best_kOhm.median():.2f} kOhm, block best "
                     f"median {d.groupby([d.X // B, d.Y // B]).block_best_kOhm.first().median():.2f}, "
                     f"pixel-to-pixel 25-75 % {d.residual_kOhm.quantile(.25):+.2f} to "
                     f"{d.residual_kOhm.quantile(.75):+.2f} (grey: not used or block < {N_MIN} px)",
                     fontsize=10)
        fig.tight_layout()
        fig.savefig(out / f"maps_{a}.png", dpi=130)
        plt.close(fig)

    # where the best pixels sit
    fig, ax = plt.subplots(len(ref), 3, figsize=(19, 6 * len(ref)), squeeze=False)
    for j, (a, (r, d)) in enumerate(ref.items()):
        anchor, _, best = br.best_reference(a, d.set_index(["X", "Y"]).I_LED)
        bx, by = best.get_level_values("X"), best.get_level_values("Y")
        x = ax[j, 0]
        lo, hi = clim(d)
        im = x.imshow(block_img(d, "block_median_kOhm"), cmap=cm, vmin=lo, vmax=hi,
                      extent=(-0.5, 511.5, 511.5, -0.5), interpolation="nearest")
        x.plot(bx, by, ".", ms=2.5, color="tab:cyan")
        for e in (32, 479.5):
            x.axhline(e, color="w", lw=0.6, ls=":")
            x.axvline(e, color="w", lw=0.6, ls=":")
        fig.colorbar(im, ax=x, label=f"32x32 block median R_best, {r.tag} (kOhm)", shrink=0.85)
        x.set_title(f"{a}: the {len(best)} best pixels (cyan) on the block medians; dotted: 32 px "
                    f"from the edge", fontsize=9)
        x = ax[j, 1]
        e_all = pd.cut(np.minimum.reduce([anchor.index.get_level_values("X"),
                                          511 - anchor.index.get_level_values("X"),
                                          anchor.index.get_level_values("Y"),
                                          511 - anchor.index.get_level_values("Y")]), EDGE_BINS)
        e_best = pd.cut(np.minimum.reduce([bx, 511 - bx, by, 511 - by]), EDGE_BINS)
        f_all = pd.Series(e_all).value_counts(normalize=True, sort=False)
        f_best = pd.Series(e_best).value_counts(normalize=True, sort=False)
        k = np.arange(len(EDGE_BINS) - 1)
        x.bar(k - 0.2, 100 * f_all.to_numpy(), 0.4, color="0.6", label="all swept pixels")
        x.bar(k + 0.2, 100 * f_best.to_numpy(), 0.4, color="tab:cyan", label="the best 2 %")
        x.set_xticks(k, bin_labels())
        x.set_xlabel("distance to the array edge (px)")
        x.set_ylabel("% of the pixels")
        x.set_title(f"{a}: where the best pixels sit", fontsize=10)
        x.legend(fontsize=8)
        x.grid(alpha=0.3, axis="y")
        x = ax[j, 2]
        for col, c, lab in (("R_best_kOhm", "tab:red", "R_best: against the array's best"),
                            ("R_local_kOhm", "tab:blue", "R_local: against the block's best")):
            m, q1, q3 = edge_profile(d, col)
            x.errorbar(k[:len(m)], m, yerr=[m - q1, q3 - m], fmt="o-", capsize=3, color=c, label=lab)
        x.axhline(0, color="k", lw=0.8)
        x.set_xticks(k, bin_labels())
        x.set_xlabel("distance to the array edge (px)")
        x.set_ylabel("kOhm (median, 25-75 %)")
        x.set_title(f"{a}, {r.tag}: R against the edge distance", fontsize=10)
        x.legend(fontsize=8)
        x.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out / "best_location.png", dpi=130)
    plt.close(fig)
    return 0


if __name__ == "__main__":
    sys.exit(main())
