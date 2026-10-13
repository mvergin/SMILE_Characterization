"""subsample_convergence.py — how many measured pixels are needed to recover the SPATIAL
STRUCTURE (not just the median) of a non-uniform array?

For a device that is genuinely non-uniform (brightness/yield gradients, per-pixel scatter), the
median of a channel converges with a handful of pixels, but the *map* needs far more. This tool
sub-samples from 1% → 100% of the available pixels and measures, per channel, how well the
sub-sample reproduces the full spatial conclusion:

  - median            — the easy target (∝ 1/√N),
  - gradient          — large-scale plane (magnitude + direction) vs the full-set gradient,
  - MAP FIDELITY      — nearest-neighbour reconstruction RMS vs the reference map (the real
                        question: "if I'd only measured N pixels, how wrong is my map?").

Two ground truths are used:
  • BRIGHTNESS uses the FULL fast-scan map (all pixels) as the reference — a true full-coverage
    ground truth, so reconstruction error is measured against reality, not a subset.
  • PEAK-IQE (and J_peak) use the full set of *swept* pixels as the reference (no full-coverage
    truth exists for a fit-based quantity).

Usage:
    python subsample_convergence.py <run_dir> [--boot 80] [--blockpx 32]

Writes  <run_dir>/REPORT_assets/subsample_convergence.png  and prints a recommended minimum
pixel count (where map fidelity reaches within 1.3× of its full-coverage floor).
"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import analyze_array as A                                   # reuse fit_abc / median_eqe_curve / EQE

FRACS = [0.01, 0.02, 0.03, 0.05, 0.08, 0.12, 0.20, 0.30, 0.50, 0.75, 1.0]
PEAK_J_LO, PEAK_J_HI = 6.0, 25.0                            # EQE-plateau window for the peak proxy


def _load(run_dir, name):
    p = list((run_dir / name).rglob("aggregated_bitval=*_mean_std.csv"))
    return pd.read_csv(p[0]) if p else None


def _per_pixel_channels(run_dir):
    """Return dict of channel -> DataFrame[X, Y, val], plus the fast-scan brightness reference."""
    out = {}
    sw = _load(run_dir, "NVLEDSweep")
    if sw is not None:
        sw = sw.copy()
        sw["I"] = sw["NVLED_CURR_MEAN"].abs()
        sw["P"] = sw["PM400_POWER_MEAN"].clip(lower=0)
        sw = sw[(sw["I"] > 1e-6) & (sw["P"] > 3e-11)]      # 1 µA offset floor + optical floor
        sw["J"] = sw["I"] / (A.PIXEL_AREA_UM2 * 1e-8)
        sw["EQE"] = sw["P"] / sw["I"]
        win = sw[(sw["J"] >= PEAK_J_LO) & (sw["J"] <= PEAK_J_HI)]
        g = win.groupby(["X", "Y"]).agg(eqe=("EQE", "median"), n=("EQE", "size")).reset_index()
        g = g[g["n"] >= 3]
        # peak-IQE proxy, calibrated to the pooled fit
        med = A.median_eqe_curve(sw["J"].values, sw["EQE"].values)
        fit = A.fit_abc(med["J"].values, med["EQE"].values)
        ref = fit["peak_IQE"] if fit else np.nan
        if np.isfinite(ref):
            out["peak IQE"] = pd.DataFrame({"X": g["X"], "Y": g["Y"],
                                            "val": g["eqe"] / g["eqe"].median() * ref * 100})
        # per-pixel J at EQE-max (SRH balance / peak location)
        jm = sw.loc[sw.groupby(["X", "Y"])["EQE"].idxmax(), ["X", "Y", "J"]]
        jm = jm.merge(g[["X", "Y"]], on=["X", "Y"])
        out["J_peak"] = jm.rename(columns={"J": "val"})[["X", "Y", "val"]]
    fs = _load(run_dir, "FastScan")
    fs_ref = None
    if fs is not None:
        fs = fs.copy(); fs["val"] = fs["PM400_POWER_MEAN"].abs() * 1e9
        thr = A.DEAD_THRESHOLD_FRAC * np.nanpercentile(fs["val"].values, A.FASTSCAN_ALIVE_PCTL)
        fs_ref = fs[fs["val"] >= thr][["X", "Y", "val"]].reset_index(drop=True)   # full-coverage truth
    return out, fs_ref


def _nn_recon_rms(sample, ref):
    """RMS of the nearest-neighbour reconstruction error: predict every REF pixel from the nearest
    SAMPLE pixel, relative to the reference median (%). This is the map-fidelity metric."""
    from scipy.spatial import cKDTree
    st = cKDTree(sample[["X", "Y"]].values)
    _, idx = st.query(ref[["X", "Y"]].values, k=1)
    pred = sample["val"].values[idx]
    return float(np.sqrt(np.mean((pred - ref["val"].values) ** 2)) / np.median(ref["val"].values) * 100)


def _gradient(df):
    A_ = np.c_[df["X"].values, df["Y"].values, np.ones(len(df))]
    c, *_ = np.linalg.lstsq(A_, df["val"].values, rcond=None)
    return np.array([c[0], c[1]])                          # per-pixel gradient vector


def _convergence(channel_df, boot, rng):
    """HOLD-OUT cross-validation: a fixed independent test set (20% of pixels) is reconstructed from
    N pixels sampled from the remaining 80%. Predicting *held-out* pixels means the error plateaus at
    the real spatial-interpolation floor (local roughness + noise) instead of the degenerate 0 you
    get when a set predicts itself. Returns per-fraction metrics + the full-set gradient."""
    n_all = len(channel_df)
    n_test = min(1500, n_all // 5)                         # 20% held out (independent evaluation)
    perm = rng.permutation(n_all)
    test = channel_df.iloc[perm[:n_test]].reset_index(drop=True)
    pool = channel_df.iloc[perm[n_test:]].reset_index(drop=True)
    med_full = np.median(channel_df["val"].values)
    g_full = _gradient(channel_df)
    span = max(np.ptp(channel_df["X"].values), np.ptp(channel_df["Y"].values), 1)
    grad_full_pct = np.hypot(*g_full) * span / med_full * 100
    rows = []
    for f in FRACS:
        n = min(max(6, int(round(f * n_all))), len(pool))  # N as a fraction of the FULL available set
        recon, gerr, merr, gang = [], [], [], []
        for _ in range(boot):
            s = pool.iloc[rng.choice(len(pool), n, replace=False)]
            recon.append(_nn_recon_rms(s, test))           # predict the held-out test pixels
            gs = _gradient(s)
            gerr.append(np.hypot(*(gs - g_full)) * span / med_full * 100)     # gradient error %/edge
            if np.hypot(*g_full) > 0 and np.hypot(*gs) > 0:
                cosang = np.dot(gs, g_full) / (np.hypot(*gs) * np.hypot(*g_full))
                gang.append(np.degrees(np.arccos(np.clip(cosang, -1, 1))))
            merr.append(abs(np.median(s["val"].values) - med_full) / med_full * 100)
        rows.append(dict(frac=n / n_all, n=n,
                         recon=np.mean(recon), recon_sd=np.std(recon),
                         grad_err=np.mean(gerr), med_err=np.mean(merr),
                         dir_err=np.mean(gang) if gang else np.nan))
    return pd.DataFrame(rows), grad_full_pct


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_dir")
    ap.add_argument("--boot", type=int, default=80, help="bootstrap repeats per fraction")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    run_dir = Path(args.run_dir)
    rng = np.random.default_rng(args.seed)

    channels, fs_ref = _per_pixel_channels(run_dir)
    if not channels:
        print("no swept-pixel channels found"); sys.exit(1)

    plt = A._plt()
    # BRIGHTNESS uses the full fast-scan map (real full-coverage ground truth); the sweep channels
    # use their own full set. Each task is (label, per-pixel dataframe).
    tasks = []
    if fs_ref is not None and len(fs_ref) > 50:
        tasks.append(("brightness (full fast-scan truth)", fs_ref))
    for name, df in channels.items():
        tasks.append((name + " (swept)", df))

    fig, axes = plt.subplots(1, len(tasks), figsize=(5.2 * len(tasks), 4.6), squeeze=False)
    print(f"\n=== sub-sampling convergence: {run_dir.name} ===")
    for ax, (label, ch) in zip(axes[0], tasks):
        conv, grad_full = _convergence(ch, args.boot, rng)
        perpix_floor = conv["recon"].iloc[-1]               # per-pixel scatter (NOT reconstructible)
        # Recommendation keys off the RECOVERABLE structure — the large-scale gradient — reaching a
        # useful accuracy: within 2%/edge OR 1/5 of the full gradient, whichever is looser.
        gtarget = max(2.0, 0.2 * grad_full)
        okg = conv[conv["grad_err"] <= gtarget]
        rec_n = int(okg["n"].iloc[0]) if len(okg) else int(conv["n"].iloc[-1])
        rec_f = rec_n / len(ch)
        print(f"\n[{label}]  (available={len(ch)}, full gradient {grad_full:.0f}%/edge, "
              f"per-pixel floor {perpix_floor:.0f}%)")
        print(f"  {'N':>6} {'%avail':>7} {'map-RMS%':>9} {'grad_err%':>10} {'dir_err°':>9} {'med_err%':>9}")
        for _, r in conv.iterrows():
            print(f"  {int(r.n):>6} {100*r.frac:>6.0f}% {r.recon:>8.1f} {r.grad_err:>10.1f} "
                  f"{r.dir_err:>8.0f} {r.med_err:>8.2f}")
        print(f"  → LARGE-SCALE structure (gradient→{gtarget:.0f}%/edge) captured at N≈{rec_n} "
              f"({100*rec_f:.0f}% of available); per-pixel map needs full coverage (floor {perpix_floor:.0f}%).")
        ax.plot(conv["n"], conv["grad_err"], "o-", ms=4, color="tab:red", label="gradient error (structure)")
        ax.plot(conv["n"], conv["med_err"], "^-", ms=3, color="tab:green", label="median error")
        ax.plot(conv["n"], conv["recon"], "s:", ms=3, color="tab:blue",
                label=f"per-pixel floor ({perpix_floor:.0f}%, flat)")
        ax.axhline(gtarget, color="0.5", ls=":", lw=1)
        ax.axvline(rec_n, color="tab:red", ls="--", lw=1, label=f"rec. N≈{rec_n}")
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_xlabel("# measured pixels"); ax.set_ylabel("error (% of median)")
        ax.set_title(label, fontsize=10); ax.grid(alpha=0.3, which="both"); ax.legend(fontsize=7)
    fig.suptitle(f"Spatial-structure convergence — {run_dir.name}\n"
                 "map recon RMS = how wrong the reconstructed map is with N pixels (vs full coverage)")
    fig.tight_layout()
    out = run_dir / "REPORT_assets"; out.mkdir(exist_ok=True)
    fig.savefig(out / "subsample_convergence.png", dpi=300); plt.close(fig)
    print(f"\nwrote {out/'subsample_convergence.png'}")


if __name__ == "__main__":
    main()
