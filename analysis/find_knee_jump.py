"""find_knee_jump.py — find pixels whose EQE(J) has the NPLC-switch discontinuity at the knee.

The per-segment sweep switches NPLC 2↔0.5 at the knee (default V≈-1.3). A stale SMU zero across that
switch gives a small current-offset step → an EQE=P/I jump at the knee. This scans a run's swept
pixels, fits a local log-EQE trend on EACH side of the knee (NPLC-2 rising side and NPLC-0.5 droop
side), extrapolates both to the knee voltage, and reports the fractional step between them per pixel.

It then prints the worst `--n` jumpers and the `--n` cleanest (near-zero) pixels — ready to paste into
`autozero_bench.py --pixels ...` — and writes a histogram + example curves.

Usage:  python find_knee_jump.py <run_dir> [--n 3] [--knee -1.3]
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def _load(run_dir):
    ag = list((Path(run_dir) / "NVLEDSweep").rglob("*aggregated_bitval=*_mean_std.csv"))
    if not ag:
        raise SystemExit(f"no sweep aggregate under {run_dir}")
    df = pd.read_csv(ag[0])
    val = lambda base: df[f"{base}_MEDIAN"] if f"{base}_MEDIAN" in df.columns else df[f"{base}_MEAN"]
    df["I"] = val("NVLED_CURR").abs()
    df["P"] = val("PM400_POWER").clip(lower=0)
    df["EQE"] = np.where((df["I"] > 1e-6) & (df["P"] > 3e-11), df["P"] / df["I"], np.nan)
    cfgs = list((Path(run_dir) / "NVLEDSweep").rglob("*_config.json"))
    knee = -1.3
    if cfgs:
        knee = json.load(open(cfgs[0]))["settings"].get("vramp_knee_v_used") \
            or json.load(open(cfgs[0]))["settings"].get("vramp_knee_v", -1.3)
    return df, float(knee)


def knee_jump_pct(g, knee, k=4):
    """Fractional EQE step across the knee for one pixel: fit log(EQE) vs V on the k points nearest
    the knee on each side, extrapolate both to `knee`, return 100*(lo_side/hi_side - 1). NaN if too
    few points. lo_side = NPLC-2 rising side (V>knee); hi_side = NPLC-0.5 droop side (V<=knee)."""
    g = g[np.isfinite(g["EQE"]) & (g["EQE"] > 0)]
    lo = g[g["NVLED_V"] > knee]; hi = g[g["NVLED_V"] <= knee]
    if len(lo) < 2 or len(hi) < 2:
        return np.nan
    lo = lo.nsmallest(min(k, len(lo)), "NVLED_V")     # V closest to the knee from above
    hi = hi.nlargest(min(k, len(hi)), "NVLED_V")      # V closest to the knee from below
    def extrap(d):
        c = np.polyfit(d["NVLED_V"].values, np.log(d["EQE"].values), 1)
        return np.exp(np.polyval(c, knee))
    try:
        return float(100.0 * (extrap(lo) / extrap(hi) - 1.0))
    except Exception:
        return np.nan


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_dir")
    ap.add_argument("--n", type=int, default=3, help="how many jumpers / clean pixels to list")
    ap.add_argument("--knee", type=float, default=None, help="override knee voltage")
    args = ap.parse_args()
    df, knee = _load(args.run_dir)
    if args.knee is not None:
        knee = args.knee

    rows = []
    for (x, y), g in df.groupby(["X", "Y"]):
        j = knee_jump_pct(g, knee)
        # brightness at the knee (current there), to show the jump vs signal trend
        Ik = g[np.isfinite(g["EQE"])]["I"].median()
        if np.isfinite(j):
            rows.append((int(x), int(y), j, float(Ik) * 1e6))
    r = pd.DataFrame(rows, columns=["X", "Y", "jump", "I_uA"]).dropna()
    print(f"knee={knee:.2f} V, {len(r)} pixels analysed")
    print(f"knee-jump distribution: median {r.jump.median():+.1f}%, "
          f"|jump| median {r.jump.abs().median():.1f}%, 90th pct {r.jump.abs().quantile(0.9):.1f}%")

    jumpers = r.reindex(r.jump.abs().sort_values(ascending=False).index).head(args.n)
    clean = r.reindex(r.jump.abs().sort_values().index).head(args.n)
    print(f"\nWORST {args.n} JUMPERS:")
    for _, p in jumpers.iterrows():
        print(f"  ({int(p.X)},{int(p.Y)}): {p.jump:+.1f}%  (I@knee {p.I_uA:.0f} µA)")
    print(f"CLEANEST {args.n} (near-zero jump):")
    for _, p in clean.iterrows():
        print(f"  ({int(p.X)},{int(p.Y)}): {p.jump:+.1f}%  (I@knee {p.I_uA:.0f} µA)")
    sel = pd.concat([jumpers, clean])
    print("\n--pixels argument for autozero_bench.py:")
    print("  " + " ".join(f"{int(p.X)},{int(p.Y)}" for _, p in sel.iterrows()))

    # is the jump correlated with signal? (offset is ~fixed current → bigger relative jump when dim)
    if r["I_uA"].std() > 0:
        cc = np.corrcoef(np.log(r["I_uA"].clip(lower=1)), r["jump"].abs())[0, 1]
        print(f"\ncorr(|jump|, -log I@knee) = {-cc:+.2f}  "
              f"({'dimmer pixels jump more → consistent with a fixed current offset' if -cc > 0.2 else 'no clear signal dependence'})")

    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
        ax[0].hist(r.jump, bins=60, color="tab:purple", alpha=0.8)
        ax[0].axvline(0, color="k", lw=0.8); ax[0].set_xlabel("knee EQE jump (%)")
        ax[0].set_ylabel("pixels"); ax[0].set_title(f"knee-jump distribution (knee {knee:.2f} V)")
        for _, p in pd.concat([jumpers.head(3), clean.head(3)]).iterrows():
            g = df[(df.X == p.X) & (df.Y == p.Y) & np.isfinite(df.EQE)].sort_values("NVLED_V")
            ax[1].plot(g.I / 1e-6, g.EQE, ".-", ms=3, lw=0.7,
                       label=f"({int(p.X)},{int(p.Y)}) {p.jump:+.0f}%")
        ax[1].axvline((df[df.NVLED_V.round(2) == round(knee, 2)].I.median() / 1e-6), color="0.6", ls=":", lw=1)
        ax[1].set_xscale("log"); ax[1].set_xlabel("J (A/cm²)"); ax[1].set_ylabel("EQE")
        ax[1].set_title("example jumpers (steep) vs clean pixels"); ax[1].legend(fontsize=7)
        out = Path(args.run_dir) / "REPORT_assets"; out.mkdir(exist_ok=True)
        fig.tight_layout(); fig.savefig(out / "knee_jump.png", dpi=300)
        print(f"\nwrote {out/'knee_jump.png'}")
    except Exception as e:
        print(f"(plot skipped: {e})")


if __name__ == "__main__":
    main()
