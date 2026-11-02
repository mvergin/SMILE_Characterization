"""One array before and after its carrier capacitors were removed.

    python analysis/compare_cap_removal.py            # Array42, the runs below
    python analysis/compare_cap_removal.py --before <step run> --after <step run> \
        --fs-before <fast-scan run> --fs-after <fast-scan run> --out <folder>

Writes into --out (default results/comparisons/Array42_cap_removal):
  step_table.csv     per stage, High-C, range and bias: median settled current, its
                     share of the range, t to 1 %, light settling; before and after
  step_long.png      the long captures, |I - I_inf| / I_inf against time
  fastscan.png       the fast scans pixel by pixel, each normalised to its median
  cap_table.csv      the cap method's capacitance per setpoint
  summary.md         the numbers

The cap and step figures (capacitance, power-law tail, range artifact, settling) come
from ../analyze_cap_comparison.py run on the same two step runs into the same folder.

Absolute light is not compared: the optical coupling changed ~100x between the runs
(Array42 fast scan 55 nW at -1.35 V on 261019, 6031 nW at -1.4 V on 261102; Array38 shows
the same jump between 261022 and 261026), with the capacitors on the board unchanged in
between for Array38. Ratios within a run are compared instead.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import analyze_array as aa
from plot_transients import clean

R = Path(__file__).resolve().parent.parent / "results"
DEFAULT = dict(before="261021-1237_Array42_CAPACITANCE_Cap_Step",
               after="261102-1346_Array42_CAPACITANCE_Cap_Step",
               fs_before="261019-1537_Array42_IQE_ROI_(0, 0, 63, 63)",
               fs_after="261102-1333_Array42_ROI_(0, 0, 63, 63)_FASTSCAN",
               out="comparisons/Array42_cap_removal")


def step_table(run: Path) -> pd.DataFrame:
    d = pd.read_csv(run / "step.csv")
    d["range_A"] = pd.to_numeric(d["range_i"], errors="coerce").fillna(d["compliance_a"])
    d["settled_A"] = d["settled"].abs()
    d["frac_of_range"] = d["settled_A"] / d["range_A"]
    return (d.groupby(["stage", "highc", "range_A", "v_to"])
            [["settled_A", "frac_of_range", "t_1pct", "tail_drift_frac", "p_t_1pct"]]
            .median())


def long_traces(run: Path):
    """[(key, t, |I - I_inf| / I_inf, I_inf)] of the long captures."""
    out = []
    with np.load(run / "traces.npz") as z:
        for k in sorted(z.files):
            if "_step_long_" not in k or z[k].shape[0] != 3:   # 2 rows: the P(t) trace
                continue
            t, _ia, ib = z[k]
            tc, ic = clean(np.asarray(t, float), np.abs(np.asarray(ib, float)))
            if tc.size < 10:
                continue
            inf = float(np.median(ic[tc >= tc[-1] - 0.1 * (tc[-1] - tc[0])]))
            out.append((k, tc - tc[0], np.abs(ic - inf) / inf, inf))
    return out


def fastscan(run: Path) -> pd.DataFrame:
    fs = aa._load_aggregated(run / "FastScan")
    fs, _ = aa.split_fastscan_biases(fs, aa._load_config(run / "FastScan"))
    d = pd.DataFrame({"X": fs["X"].astype(int), "Y": fs["Y"].astype(int),
                      "I": fs["NVLED_CURR_MEAN"].abs(), "P": fs["PM400_POWER_MEAN"].abs()})
    d.attrs["v"] = float(fs["NVLED_V"].iloc[0])
    return d.set_index(["X", "Y"])


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    for k, v in DEFAULT.items():
        ap.add_argument(f"--{k.replace('_', '-')}", default=v,
                        help=f"run folder under results (default {v})")
    a = ap.parse_args(argv)
    out = R / a.out
    out.mkdir(parents=True, exist_ok=True)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    rb, ra = R / a.before, R / a.after
    L = [f"# Cap removal: {a.before} (caps) against {a.after} (no caps)", ""]

    # ---- step: settled reading on each range ----------------------------------------
    tb, ta = step_table(rb), step_table(ra)
    t = tb.join(ta, lsuffix="_caps", rsuffix="_nocaps", how="outer")
    t.to_csv(out / "step_table.csv")
    fast = t.xs("fast", level="stage")
    L += ["## Step, fast stage (median of 4 pixels, one per quadrant)", "",
          "| High-C | range (A) | v_to (V) | settled caps (uA) | % of range | "
          "settled no caps (uA) | % of range | t 1 % caps (ms) | t 1 % no caps (ms) |",
          "|" + "---|" * 9]
    for (hc, rng, v), r in fast.iterrows():
        f = lambda x, s=1, d=3: "overflow" if not np.isfinite(x) else f"{x * s:.{d}g}"
        L.append(f"| {hc} | {rng:.0e} | {v:+.2f} | {f(r['settled_A_caps'], 1e6)} | "
                 f"{f(r['frac_of_range_caps'], 100)} | {f(r['settled_A_nocaps'], 1e6)} | "
                 f"{f(r['frac_of_range_nocaps'], 100)} | {f(r['t_1pct_caps'], 1e3)} | "
                 f"{f(r['t_1pct_nocaps'], 1e3)} |")
    lg = t.xs("long", level="stage")
    L += ["", "## Step, long stage (-1.0 -> -0.8 V, NPLC 1)", "",
          "| High-C | range (A) | settled caps (nA) | settled no caps (nA) | t 1 % caps (s) | "
          "t 1 % no caps (s) |", "|" + "---|" * 6]
    for (hc, rng, v), r in lg.iterrows():
        L.append(f"| {hc} | {rng:.0e} | {r['settled_A_caps'] * 1e9:.0f} | "
                 f"{r['settled_A_nocaps'] * 1e9:.0f} | {r['t_1pct_caps']:.3g} | "
                 f"{r['t_1pct_nocaps']:.3g} |")

    # ---- long captures ----------------------------------------------------------------
    fig, ax = plt.subplots(1, 2, figsize=(15, 5.5), sharey=True)
    for x, hc in zip(ax, ("hc0", "hc1")):
        for run, col, name in ((rb, "tab:blue", "caps"), (ra, "tab:orange", "no caps")):
            first = True
            for k, tt, dev, inf in long_traces(run):
                if f"_{hc}_" not in k:
                    continue
                x.plot(tt[1:], 100 * dev[1:], lw=0.8, color=col, alpha=0.8,
                       label=f"{name} (I_inf {inf * 1e9:.0f} nA)" if first else None)
                first = False
        x.set_xscale("log")
        x.set_yscale("log")
        x.set_ylim(1e-2, 1e3)
        x.axhline(1, color="k", ls="--", lw=0.8)
        x.set_xlabel("time after the step (s)")
        x.set_title(f"Long captures, High-C {'on' if hc == 'hc1' else 'off (1 uA range)'}, "
                    f"4 pixels each", fontsize=9)
        x.grid(alpha=0.3, which="both")
        x.legend(fontsize=8)
    ax[0].set_ylabel("|I - I_inf| / I_inf (%)")
    fig.tight_layout()
    fig.savefig(out / "step_long.png", dpi=150)
    plt.close(fig)

    # ---- cap method ---------------------------------------------------------------------
    cb, ca = pd.read_csv(rb / "cap.csv"), pd.read_csv(ra / "cap.csv")
    keep = ["lit", "i_set", "quadrant", "capacitance_F", "v_start", "v_end", "leakage_suspect"]
    cap = pd.concat([cb.assign(run="caps"), ca.assign(run="no caps")])
    cap[["run"] + [k for k in keep if k in cap] +
        [k for k in ("park_ok", "park_off") if k in cap]].to_csv(out / "cap_table.csv",
                                                                 index=False)
    cm_ = cap.groupby(["run", "lit", "i_set"])["capacitance_F"].median().unstack("run") * 1e6
    L += ["", "## Cap method (median C over the pixels, uF)", "",
          "```", cm_.to_string(float_format=lambda x: f"{x:.3g}"), "```"]
    if "park_ok" in ca:
        L.append(f"\nNo caps: {int((ca['park_ok'] == 0).sum())} of {len(ca)} ramps did not "
                 f"start at the park voltage (park_ok 0, median offset "
                 f"{ca['park_off'].abs().median():.2f} V): no capacitance from them.")

    # ---- fast scans, pixel by pixel -------------------------------------------------------
    fb, fa = fastscan(R / a.fs_before), fastscan(R / a.fs_after)
    m = fb.join(fa, lsuffix="_caps", rsuffix="_nocaps", how="inner")
    for s in ("_caps", "_nocaps"):
        m[f"PI{s}"] = m[f"P{s}"] / m[f"I{s}"]
        lit = m[f"P{s}"] > 0.2 * m[f"P{s}"].quantile(0.9)
        m[f"lit{s}"] = lit
        for c in ("I", "P", "PI"):
            m[f"{c}n{s}"] = m[f"{c}{s}"] / m.loc[lit, f"{c}{s}"].median()
    both = m["lit_caps"] & m["lit_nocaps"]
    m.reset_index().to_csv(out / "fastscan_pixels.csv", index=False)
    fig, ax = plt.subplots(2, 3, figsize=(16, 9.5))
    for j, (c, name) in enumerate((("I", "current"), ("P", "light"), ("PI", "P/I"))):
        x, y = m.loc[both, f"{c}n_caps"], m.loc[both, f"{c}n_nocaps"]
        r = np.corrcoef(x, y)[0, 1]
        ax[0, j].scatter(x, y, s=3, alpha=0.4)
        lo, hi = np.nanpercentile(np.r_[x, y], [0.5, 99.5])
        ax[0, j].plot([lo, hi], [lo, hi], "k--", lw=0.8)
        ax[0, j].set_xlim(lo, hi)
        ax[0, j].set_ylim(lo, hi)
        ax[0, j].set_xlabel(f"{name} / median, caps ({fb.attrs['v']:+.2f} V)")
        ax[0, j].set_ylabel(f"{name} / median, no caps ({fa.attrs['v']:+.2f} V)")
        ax[0, j].set_title(f"{name}: r = {r:.3f} over {int(both.sum())} pixels", fontsize=9)
        ax[0, j].grid(alpha=0.3)
        rat = (m[f"{c}n_nocaps"] / m[f"{c}n_caps"]).where(both).unstack("Y").T
        im = ax[1, j].imshow(rat.values, origin="upper", cmap="RdBu_r", vmin=0.8, vmax=1.2,
                             extent=(rat.columns.min() - .5, rat.columns.max() + .5,
                                     rat.index.max() + .5, rat.index.min() - .5))
        ax[1, j].set_title(f"{name}: no caps / caps, each over its median", fontsize=9)
        fig.colorbar(im, ax=ax[1, j], shrink=0.8)
        if j == 0:
            L += ["## Fast scans, pixel by pixel", ""]
        L.append(f"Fast scan {name}: pixel-by-pixel r = {r:.3f}; no caps / caps "
                 f"(normalised) median {float((y / x).median()):.3f}, 10-90 % "
                 f"{float((y / x).quantile(.1)):.3f}-{float((y / x).quantile(.9)):.3f}.")
    fig.suptitle(f"Fast scans: {a.fs_before} (caps) against {a.fs_after} (no caps)",
                 fontsize=10)
    fig.tight_layout()
    fig.savefig(out / "fastscan.png", dpi=150)
    plt.close(fig)
    dead_b, dead_a = ~m["lit_caps"], ~m["lit_nocaps"]
    L.append(f"Dark pixels: {int(dead_b.sum())} with caps, {int(dead_a.sum())} without, "
             f"{int((dead_b & dead_a).sum())} the same.")
    L.append(f"Median current {m.loc[m['lit_caps'], 'I_caps'].median() * 1e6:.1f} uA at "
             f"{fb.attrs['v']:+.2f} V (caps), {m.loc[m['lit_nocaps'], 'I_nocaps'].median() * 1e6:.1f}"
             f" uA at {fa.attrs['v']:+.2f} V (no caps): different biases, not comparable.")
    (out / "summary.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))
    return 0


if __name__ == "__main__":
    sys.exit(main())
