"""The full 512x512 fast-scan map, and the proof that its 10 ms points are quantitative.

    python plot_full_array.py RUN [--out DIR]

RUN is a run directory whose FastScan covers the whole array and whose NVLEDSweep is a
`sweep_order=bias_major` voltage sweep over a subset of it (run 261015-0015).

Two figures:

    full_array_map.png    power map, EQE map, and the two histograms side by side
    fastscan_valid.png    the fast scan against the slow sweep, per pixel

The point of the second figure. A fast-scan point is a 10 ms dwell on an array whose
cathode carries 42.8 uF, so it reads an UNSETTLED charge state, and the expectation was
that it could only ever be a qualitative picture. It is not. Against the sweep's 54.47 s
settle, at matched bias, the bulk of pixels agree to **1.0 % robust spread** on power
with a -2.6 % median offset, and the deviations in the two channels correlate at
r = 0.99 -- whatever moves a pixel moves its light and its current together. So the
offset cancels in the ratio: EQE agrees to +0.2 % median, 1.5 % robust.

The std of the power ratio is 6.3 %, not 1.0 %, and the gap between the two is the
result. It is three pixels of ninety-seven, all low, 23-42 %. Three more were dropped
before the comparison by REF_CONTAMINATED -- their own leakage reference carried ~58 %
of their light at every bias. EQE survives all six at <= 13 %, because whatever is wrong
with them removes current and light together.

So the whole-array scan is quantitative, and "a good grasp over the whole array" goes
from 364 days to 2.2 h. Quote the robust spread with the std beside it, never alone.

The comparison is made at matched bias by log-interpolating each swept pixel's own
P(V) and I(V) to the fast scan's bias. Comparing -1.35 V against the nearest swept
point instead divides the diode's own I-V by itself and reports a 15 % disagreement
that is pure bias mismatch -- the mistake section 13 of the report records.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

DEAD_FRAC = 0.01     # power below this fraction of the array median = dead
LIVE_FRAC = 0.5      # and below this, not trustworthy as an efficiency
REF_CARRIED = 0.1    # reference carried this much of the light...
REF_SWALLOWED = 0.2  # ...and the net fell below this fraction of its bias cohort


def load_fastscan(run: Path) -> pd.DataFrame:
    """One row per pixel: X, Y, P (W), I (A, magnitude), E = P/I."""
    raw = run / "FastScan" / "raw_data" / "FastScan.csv"
    if not raw.is_file():
        sys.exit(f"{run} has no FastScan/raw_data/FastScan.csv")
    d = pd.read_csv(raw)
    f = d.pivot_table(index=["X", "Y"], columns="TYPE", values="MEAS_VALUE").reset_index()
    f = f.rename(columns={"PM400": "P", "NVLED": "I"})
    f["I"] = f["I"].abs()
    f["E"] = np.where(f["I"] > 0, f["P"] / f["I"], np.nan)
    f.attrs["bias"] = float(d["NVLED_V"].iloc[0])
    return f


def load_sweep(run: Path) -> pd.DataFrame:
    """Aggregated sweep with the reference subtracted and REF_SWALLOWED points dropped."""
    agg = sorted((run / "NVLEDSweep").glob("*/aggregated_bitval=*_mean_std.csv"))
    if not agg:
        sys.exit(f"{run} has no aggregated NVLEDSweep CSV")
    a = pd.read_csv(agg[0])
    a["P"] = a["PM400_POWER_MEAN"] - a["PM400_POWER_REF"]
    a["I"] = (a["NVLED_CURR_MEAN"] - a["NVLED_CURR_REF"]).abs()
    cohort = a.groupby("NVLED_V")["P"].transform("median")
    carried = a["PM400_POWER_REF"] / a["PM400_POWER_MEAN"]
    bad = (carried > REF_CARRIED) & (a["P"] < REF_SWALLOWED * cohort)
    # and the whole pixel, if its reference carries light at every bias -- see
    # analyze_array.REF_CONTAMINATED. The net-power test above only fires where the
    # damage is severe, and those pixels are 42-58 % low long before it does.
    bad |= carried.groupby([a["X"], a["Y"]]).transform("median") > REF_CARRIED
    if bad.any():
        print(f"  dropped {bad.sum()} contaminated points on "
              f"{a[bad].groupby(['X', 'Y']).ngroups} pixel(s)")
    return a[~bad]


def at_bias(a: pd.DataFrame, v: float) -> pd.DataFrame:
    """Log-interpolate each pixel's own P(V) and I(V) to v. See the module docstring."""
    out = []
    for (x, y), g in a.groupby(["X", "Y"]):
        g = g.sort_values("NVLED_V")
        m = (g["P"] > 0) & (g["I"] > 0)
        if m.sum() < 2 or not (g["NVLED_V"][m].min() <= v <= g["NVLED_V"][m].max()):
            continue
        vv = g["NVLED_V"][m].values
        out.append((x, y,
                    float(np.exp(np.interp(v, vv, np.log(g["P"][m].values)))),
                    float(np.exp(np.interp(v, vv, np.log(g["I"][m].values))))))
    d = pd.DataFrame(out, columns=["X", "Y", "P", "I"])
    d["E"] = d["P"] / d["I"]
    return d


def rsd(x) -> float:
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    return 100.0 * x.std() / x.mean()


def rrsd(x) -> float:
    """Outlier-resistant RSD. Three pixels of ninety-seven take the plain one from 1 %
    to 6.3 %, so the two are only honest together -- see the module docstring."""
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    lo, hi = np.percentile(x, [25, 75])
    return 100.0 * (hi - lo) / 1.349 / np.median(x)


def raster(f: pd.DataFrame, col: str, live: np.ndarray) -> np.ndarray:
    img = np.full((512, 512), np.nan)
    img[f["Y"].values, f["X"].values] = np.where(live, f[col].values, np.nan)
    return img


def boundary_rows(f: pd.DataFrame, dead: np.ndarray) -> pd.DataFrame:
    """Dead count per row. Rows 127/255/256/384/511 carry 30 % of every dead pixel."""
    img = np.zeros((512, 512), bool)
    img[f["Y"].values, f["X"].values] = dead
    n = img.sum(axis=1)
    return pd.DataFrame({"Y": np.arange(512), "n_dead": n, "boundary": n >= 10})


def main() -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize

    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run", type=Path)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()
    out = args.out or (args.run / "REPORT_assets")
    out.mkdir(parents=True, exist_ok=True)

    f = load_fastscan(args.run)
    v_fs = f.attrs["bias"]
    pmed = float(np.median(f["P"]))
    dead = (f["P"] < DEAD_FRAC * pmed).values
    live = ((f["P"] > LIVE_FRAC * pmed) & (f["I"] > LIVE_FRAC * np.median(f["I"]))).values
    print(f"fast scan at {v_fs:+.3f} V: {len(f)} pixels, {dead.sum()} dead "
          f"({100 * dead.mean():.2f} %), {live.sum()} usable")
    print(f"  power RSD {rsd(f['P'][live]):.1f} %   EQE RSD {rsd(f['E'][live]):.1f} %")

    a = load_sweep(args.run)
    sw = at_bias(a, v_fs).merge(f[["X", "Y", "P", "I", "E"]], on=["X", "Y"],
                                suffixes=("_sw", "_fs"))
    ratios = {k: (sw[k + "_sw"] / sw[k + "_fs"]).values for k in ("P", "I", "E")}
    for k, lbl in (("P", "power"), ("I", "current"), ("E", "EQE")):
        r = ratios[k]
        print(f"  sweep/fastscan {lbl:>7}: {100 * (np.median(r) - 1):+5.1f} %  "
              f"spread {rrsd(r):4.1f} % robust / {rsd(r):4.1f} % std")
    print(f"  corr(power deviation, current deviation) r = "
          f"{np.corrcoef(ratios['P'], ratios['I'])[0, 1]:.3f}")

    # ---------------- figure 1: the map ----------------
    fig, ax = plt.subplots(1, 4, figsize=(21, 5.0),
                           gridspec_kw=dict(width_ratios=[1, 1, 0.75, 0.75]))
    for k, (col, lbl, scale, unit) in enumerate((("P", "optical power", 1e9, "nW"),
                                                 ("E", "relative EQE", 1.0, ""))):
        img = raster(f, col, live) * scale
        m = float(np.nanmedian(img))
        h = ax[k].imshow(img / m, origin="lower", cmap=plt.get_cmap("viridis"),
                         norm=Normalize(0.7, 1.3), interpolation="nearest")
        ax[k].set(title=f"{lbl} / median  ({m:.3g} {unit})  RSD {rsd(img):.1f} %",
                  xlabel="X", ylabel="Y")
        fig.colorbar(h, ax=ax[k], fraction=0.046)

    bins = np.linspace(0.6, 1.4, 90)
    for col, lbl in (("P", "optical power"), ("E", "relative EQE")):
        x = f[col].values[live]
        ax[2].hist(x / np.median(x), bins=bins, histtype="step", lw=1.6,
                   label=f"{lbl}, RSD {rsd(x):.1f} %")
    ax[2].set(xlabel="value / array median", ylabel="pixels", yscale="log",
              title="the 16 % power spread is 8 % efficiency")
    ax[2].legend(fontsize=8, loc="upper left")
    ax[2].grid(alpha=.3)

    b = boundary_rows(f, dead)
    ax[3].bar(b["Y"], b["n_dead"], width=2.0,
              color=np.where(b["boundary"], "crimson", "0.45"))
    ax[3].set(xlabel="row Y", ylabel="dead pixels in the row", yscale="symlog",
              title=f"{dead.sum()} dead; {b.loc[b.boundary, 'n_dead'].sum()} of them "
                    f"on {int(b.boundary.sum())} rows")
    for y in b.loc[b["boundary"], "Y"]:
        ax[3].annotate(str(y), (y, b.loc[y, "n_dead"]), fontsize=7, ha="center",
                       va="bottom", color="crimson")
    ax[3].grid(alpha=.3)
    fig.suptitle(f"Full 512x512 array at {v_fs:+.2f} V, 262 144 pixels in 2.2 h "
                 f"(run {args.run.name.split('_')[0]})", y=1.0)
    fig.tight_layout()
    fig.savefig(out / "full_array_map.png", dpi=130, bbox_inches="tight")
    plt.close(fig)

    # ---------------- figure 2: the scan is quantitative ----------------
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.8))
    for k, lbl, c in (("P", "power", "tab:blue"), ("I", "current", "tab:orange"),
                      ("E", "EQE = P/I", "tab:green")):
        r = ratios[k]
        ax[0].hist(r, bins=np.linspace(0.3, 1.2, 95), histtype="step", lw=1.8, color=c,
                   label=f"{lbl}: {100 * (np.median(r) - 1):+.1f} %, "
                         f"{rrsd(r):.1f} % robust / {rsd(r):.1f} % std")
    ax[0].axvline(1.0, color="k", lw=.8, ls=":")
    ax[0].set(xlabel="sweep / fast scan, same pixel, same bias", ylabel="pixels",
              yscale="log",
              title=f"deviations track across channels: r = "
                    f"{np.corrcoef(ratios['P'], ratios['I'])[0, 1]:.3f}")
    ax[0].legend(fontsize=8)
    ax[0].grid(alpha=.3)

    ax[1].scatter(sw["E_fs"], sw["E_sw"], s=16, alpha=.75, edgecolor="none")
    lim = [0.9 * min(sw["E_fs"].min(), sw["E_sw"].min()),
           1.1 * max(sw["E_fs"].max(), sw["E_sw"].max())]
    ax[1].plot(lim, lim, "k:", lw=.8)
    r = float(np.corrcoef(sw["E_fs"], sw["E_sw"])[0, 1])
    ax[1].set(xlabel="fast scan EQE, 10 ms dwell", ylabel="sweep EQE, 54.47 s settle",
              xlim=lim, ylim=lim, title=f"per pixel, r = {r:.3f}")
    ax[1].grid(alpha=.3)

    sp = a.assign(E=a["P"] / a["I"]).groupby("NVLED_V")["E"].apply(rsd)
    ax[2].plot(-sp.index, sp.values, "o-", ms=4, label="sweep, pixel-to-pixel EQE RSD")
    ax[2].axhline(rsd(f["E"][live]), color="tab:red", ls="--",
                  label=f"whole array, fast scan: {rsd(f['E'][live]):.1f} %")
    ax[2].set(xlabel="|bias| (V)", ylabel="EQE RSD over pixels (%)", ylim=(0, None),
              title="7.6 % over 1.4 V, then the optical floor")
    ax[2].legend(fontsize=8)
    ax[2].grid(alpha=.3)
    fig.suptitle("A 10 ms fast-scan point reproduces a 54 s-settled sweep point, "
                 "in the ratio", y=1.02)
    fig.tight_layout()
    fig.savefig(out / "fastscan_valid.png", dpi=130, bbox_inches="tight")
    plt.close(fig)

    pd.DataFrame({"what": ["power", "current", "EQE"],
                  "median_dev_pct": [100 * (np.median(ratios[k]) - 1) for k in "PIE"],
                  "robust_spread_pct": [rrsd(ratios[k]) for k in "PIE"],
                  "std_spread_pct": [rsd(ratios[k]) for k in "PIE"]}
                 ).to_csv(out / "fastscan_valid.csv", index=False)
    print("\nwrote " + str(out / "full_array_map.png") + ", "
          + str(out / "fastscan_valid.png"))


def demo() -> None:
    """Self-check on at_bias(), the step that the section-13 mistake lives in."""
    # a pixel whose P and I are exact exponentials in V: log-interpolation must be exact
    v = np.linspace(-2.6, -0.8, 30)
    a = pd.DataFrame({"X": 1, "Y": 2, "NVLED_V": v,
                      "P": 1e-7 * np.exp(3.0 * (v + 1.35)),
                      "I": 1e-4 * np.exp(3.0 * (v + 1.35))})
    g = at_bias(a, -1.35)
    assert len(g) == 1, g
    assert abs(g["P"].iloc[0] / 1e-7 - 1) < 1e-9, g["P"].iloc[0]
    assert abs(g["E"].iloc[0] / 1e-3 - 1) < 1e-9, g["E"].iloc[0]

    # and it must REFUSE to extrapolate rather than quietly return an edge value
    assert at_bias(a, -3.0).empty, "extrapolated past the swept range"

    # the mistake it replaces: taking the nearest swept point instead. With 30 points
    # over 1.8 V the nearest bias is up to 31 mV away, which on this slope is ~10 %.
    worst = 0.5 * (v[0] + v[1])                      # exactly between two grid points
    want = float(at_bias(a, worst)["P"].iloc[0])
    near = float(a.iloc[int(np.argmin(np.abs(v - worst)))]["P"])
    assert abs(near / want - 1) > 0.03, (near, want)

    assert abs(rsd([1.0, 1.0, 1.0])) < 1e-12

    # the robust spread must survive the outliers that take the plain one apart --
    # that split IS the finding, so a regression here would bury it
    clean = np.random.default_rng(0).normal(1.0, 0.01, 94)
    mixed = np.concatenate([clean, [0.41, 0.42, 0.44, 0.58, 0.58, 0.77]])
    assert rsd(mixed) > 8.0, rsd(mixed)
    assert rrsd(mixed) < 2.0, rrsd(mixed)
    print("ok  plot_full_array: at_bias interpolates in log and refuses to extrapolate")


if __name__ == "__main__":
    if "--demo" in sys.argv:
        demo()
    else:
        main()
