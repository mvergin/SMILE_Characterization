"""Figures 40-42 for INVESTIGATION_REPORT.md section 27 (Array42 against de-capped Array38).

    python analysis/make_fig27.py [path-to-a-run-tree]

Reads the run directories directly and prints the numbers each figure was drawn from,
so the report text can quote them. Reuses the run reader from
SMILE_Characterization/analyze_cap_comparison.py, so trace-to-row pairing is the same
code that produced the comparison figures.

  fig40_capacitance_compare.png  the cap ramps: Array42 clean; Array38 before and after
                                 the 0 A window was closed
  fig41_settling_compare.png     long captures per quadrant, both arrays; and the
                                 Array38 calibration traces, current AND light
  fig42_array42_nohighc.png      what the SMU reports without High-C, and the light
                                 that proves the pixel did not change
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = (Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else Path(__file__).resolve().parent.parent)
sys.path.insert(0, str(ROOT))
from analyze_cap_comparison import Run, clean, log_bin  # noqa: E402

OUT = Path(__file__).resolve().parent / "investigation_report"   # as make_capacitance_figures.py: never CWD-relative
DPI = 300
QUAD = ("TL", "TR", "BL", "BR")
C42, C38 = "#1f77b4", "#d62728"


def run(stamp: str) -> Run:
    hits = sorted((ROOT / "results").glob(f"{stamp}_*"))
    if not hits:
        raise SystemExit(f"run {stamp} not found under {ROOT / 'results'}")
    return Run(hits[0])


def _num(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return float("nan")


# ------------------------------------------------------------------ fig 40
def fig40():
    a42, a38b, a38a = run("261021-1237"), run("261021-2129"), run("261021-2236")
    fig, ax = plt.subplots(1, 3, figsize=(15.5, 4.6))
    panels = ((ax[0], a42, "TL", "Array42, 261021-1237, TL (124,135)\nsetpoints 1 / 10 / 100 uA"),
              (ax[1], a38b, "TL", "Array38, 261021-2129, TL\n10 / 30 / 100 nA, via a 0 A source"),
              (ax[2], a38a, "TL", "Array38, 261021-2236, TL\n10 / 30 / 100 nA, straight to the setpoint"))
    report = {}
    for a, r, q, title in panels:
        seen_px = None
        cols = plt.get_cmap("viridis")(np.linspace(0.05, 0.85, 3))
        isets = sorted({_num(x["i_set"]) for x in r.cap})
        for row, tv in r.cap_traces():
            if row["quadrant"] != q:
                continue
            px = (row["pixel_x"], row["pixel_y"])
            if seen_px is None:
                seen_px = px
            if px != seen_px:
                continue                       # one pixel per panel
            t, v = clean(tv[0], tv[1])
            k = isets.index(_num(row["i_set"]))
            lit = row["lit"] == "True"
            a.plot(t, v, color=cols[k], ls="-" if lit else "--", lw=1.2,
                   label=f"{_num(row['i_set'])*1e9:g} nA {'lit' if lit else 'blanked'}")
            report.setdefault(r.path.name, []).append(
                (row["lit"], _num(row["i_set"]), _num(row.get("v_parked")), v[0] if v.size else np.nan,
                 _num(row["capacitance_F"])))
        a.axhline(-0.40, color="0.3", lw=1.0, ls=":")
        a.text(0.02, -0.40, " park −0.400 V", va="bottom", fontsize=7.5, color="0.3",
               transform=a.get_yaxis_transform())
        a.set_title(title, fontsize=9.5)
        a.set_xlabel("t after the source switches to current  [s]")
        a.grid(alpha=0.3)
        h, l = a.get_legend_handles_labels()
        a.legend(dict(zip(l, h)).values(), dict(zip(l, h)).keys(), fontsize=6.8, ncol=2)
    ax[0].set_ylabel("V on ch B  [V]")
    fig.suptitle("Capacitance ramps. Array42 starts at the park and ramps cleanly; Array38 "
                 "does not start at the park, and after the fix its start depends on the "
                 "setpoint", fontsize=10.5)
    fig.tight_layout()
    fig.savefig(OUT / "fig40_capacitance_compare.png", dpi=DPI)
    plt.close(fig)
    print("\nfig40 -- ramp start against park (lit, i_set, v_parked, v_start, C):")
    for k, rows in report.items():
        print(f"  {k}")
        for lit, i, vp, v0, c in rows:
            print(f"     lit={lit:5} I={i*1e9:8.1f} nA  v_parked={vp:+.4f}  v_start={v0:+.4f}  "
                  f"C={c*1e6:10.3f} uF")


# ------------------------------------------------------------------ fig 41
def fig41():
    a42, a38 = run("261021-1237"), run("261021-1820")
    fig = plt.figure(figsize=(15.5, 8.6))
    gs = fig.add_gridspec(2, 4, height_ratios=[1, 1.05])
    print("\nfig41 -- long captures, High-C on, -0.80 V: t_1pct / t_01pct / settled")
    for qi, q in enumerate(QUAD):
        a = fig.add_subplot(gs[0, qi])
        for r, col, name in ((a42, C42, "Array42"), (a38, C38, "Array38")):
            for row, it, pt in r.step_traces():
                if row["stage"] != "long" or row["highc"] != "True" or row["quadrant"] != q:
                    continue
                t, i = clean(it[0], it[2])
                fin = np.median(np.abs(i[int(0.8 * i.size):]))
                tb, yb = log_bin(t, np.abs(i) / fin, nbin=200)
                a.plot(tb, yb, color=col, lw=1.3,
                       label=f"{name} ({row['pixel_x']},{row['pixel_y']})")
                print(f"   {name} {q} ({row['pixel_x']},{row['pixel_y']})  "
                      f"t1%={_num(row['t_1pct']):7.2f} s  t0.1%={_num(row['t_01pct']):6.2f}  "
                      f"I={_num(row['settled'])*1e9:6.1f} nA  drift={_num(row['tail_drift_frac'])*100:.2f}%")
        a.axhspan(0.99, 1.01, color="0.85", zorder=0)
        a.set_xscale("log")
        a.set_ylim(0.55, 1.12)
        a.set_xlim(left=0.3)
        a.set_title(q, fontsize=10)
        a.set_xlabel("t after the step  [s, log]")
        a.grid(which="both", alpha=0.25)
        a.legend(fontsize=7, loc="lower right")
        if qi == 0:
            a.set_ylabel("current / its own final value")

    z = np.load(sorted((ROOT / "results").glob("261021-2349_*"))[0] / "calibration_traces.npz")
    print("\nfig41 -- calibration 261021-2349, % of final, median per window "
          "(0-1 / 1-5 / 5-20 / 20-60 / 60-150 s) and share of record outside 1 %:")
    for pi, v in enumerate(("-2.6000", "-0.8000")):
        a = fig.add_subplot(gs[1, 2 * pi:2 * pi + 2])
        cmap = plt.get_cmap("tab10")
        keys = sorted(k for k in z.files if k.endswith(f"_v{v}"))
        for ki, k in enumerate(keys):
            t, i, p = z[k]
            # At -0.80 V the light is a few hundred pW and its scatter is 15-40 %: it says
            # nothing about settling and buried the current traces, so it is left out there.
            pairs = ((i, "-", "current"), (p, ":", "light")) if pi == 0 else ((i, "-", "current"),)
            for y, ls, nm in pairs:
                fin = np.median(np.abs(y[int(0.8 * y.size):]))
                d = np.abs(np.abs(y) - fin) / fin * 100
                tb, db = log_bin(t, d, nbin=90)
                keep = tb > 0
                a.plot(tb[keep], db[keep], color=cmap(ki), ls=ls, lw=1.2 if ls == "-" else 1.0,
                       label=f"({k.split('_')[0]},{k.split('_')[1]}) {nm}")
                w = [np.median(d[(t >= lo) & (t < hi)]) for lo, hi in
                     ((0, 1), (1, 5), (5, 20), (20, 60), (60, 150))]
                print(f"   V={v} {k.split('_v')[0]:8} {nm:7} "
                      + " ".join(f"{x:5.2f}" for x in w) + f"   >1%: {100*np.mean(d > 1):4.1f}%")
        for lev, lab in ((1.0, "1 %"), (0.1, "0.1 %")):
            a.axhline(lev, color="0.35", ls="--", lw=0.9)
            a.text(0.995, lev, lab, transform=a.get_yaxis_transform(), ha="right",
                   va="bottom", fontsize=7.5, color="0.35")
        a.axvline(0.91, color="0.5", ls=":", lw=1.0)
        a.text(0.91, 0.02, " 0.91 s = metric floor\n (half a smoothing window)", fontsize=7,
               color="0.4", transform=a.get_xaxis_transform())
        a.set_xscale("log"); a.set_yscale("log")
        a.set_ylim(0.005, 20)
        a.set_xlim(0.1, 200)
        a.set_title(f"Array38 calibration 261021-2349, {float(v):+.2f} V, four quadrants\n"
                    + ("solid = current, dotted = light" if pi == 0 else
                       "current only: the light here is a few hundred pW with 15-40 % scatter"),
                    fontsize=9.5)
        a.set_xlabel("t after the step  [s, log]")
        a.set_ylabel("|value - final| / final  [%]")
        a.grid(which="both", alpha=0.25)
        a.legend(fontsize=6.3, ncol=2, loc="upper right")
    fig.suptitle("Top: long captures, High-C on, step -1.00 -> -0.80 V, each over its own final "
                 "value. Bottom: distance from final in the Array38 calibration traces.",
                 fontsize=10.5)
    fig.tight_layout()
    fig.savefig(OUT / "fig41_settling_compare.png", dpi=DPI)
    plt.close(fig)


# ------------------------------------------------------------------ fig 42
def fig42():
    a42, a38 = run("261021-1237"), run("261021-1820")
    fig, ax = plt.subplots(1, 2, figsize=(15.0, 5.0))
    a = ax[0]
    cmap = plt.get_cmap("plasma")
    print("\nfig42a -- Array42 TL (124,135), fast capture at -0.80 V, High-C OFF:")
    rngs = sorted({_num(r["range_i"]) for r in a42.step if r["highc"] == "False"
                   and r["stage"] == "fast"}, reverse=True)
    for row, it, _ in a42.step_traces():
        if (row["stage"] != "fast" or row["quadrant"] != "TL"
                or abs(_num(row["v_to"]) + 0.80) > 1e-9):
            continue
        t, ib = it[0], np.abs(it[2])
        if row["highc"] == "True":
            # Not drawn: in High-C the 1 mA limit selects the 1 mA range, whose per-sample
            # noise at NPLC 0.01 is ~1 uA, and the trace hid the 1e-6 band entirely. The
            # High-C reading is on the right-hand panel, from the long capture.
            continue
        rg = _num(row["range_i"])
        k = rngs.index(rg)
        ok = ib < 1e36
        a.plot(t[ok] * 1e3, ib[ok], color=cmap(k / max(1, len(rngs))), lw=0.6, alpha=0.75,
               label=f"range {rg:g} A: settled {_num(row['settled'])*1e9:,.0f} nA, "
                     f"{row['n_overflow']}/{row['n']} over-range")
        a.axhline(rg, color=cmap(k / max(1, len(rngs))), ls="--", lw=0.8)
        print(f"   range {rg:7g}  settled {_num(row['settled'])*1e9:12.1f} nA  "
              f"= {100*_num(row['settled'])/rg:5.1f} % of full scale   "
              f"over-range {row['n_overflow']}/{row['n']}   ripple {100*_num(row['ripple']):.0f} %")
    a.set_yscale("log")
    a.set_ylim(1e-8, 2e-3)
    a.set_xlabel("t after the step  [ms]")
    a.set_ylabel("|I| on ch B, raw samples  [A]")
    a.set_title("Array42, TL (124,135), −0.80 V, High-C OFF, one capture per range\n"
                "dashed = that range's full scale. Over-range samples are not drawn.",
                fontsize=9.5)
    a.grid(which="both", alpha=0.25)
    a.legend(fontsize=6.5, loc="lower right")

    a = ax[1]
    b = a.twinx()
    print("\nfig42b -- long capture at -0.80 V, TL, High-C on against off:")
    for r, col, name in ((a42, C42, "Array42"), (a38, C38, "Array38")):
        for row, it, pt in r.step_traces():
            if row["stage"] != "long" or row["quadrant"] != "TL":
                continue
            hc = row["highc"] == "True"
            t, i = clean(it[0], it[2])
            tb, yb = log_bin(t, np.abs(i) * 1e9, nbin=160)
            a.plot(tb, yb, color=col, lw=1.4 if hc else 1.0, ls="-" if hc else "--",
                   label=f"{name} current, High-C {'on' if hc else 'off'}")
            if pt is not None:
                tp, p = clean(pt[0], pt[1])
                keep = tp >= 1.0      # before ~1 s a bin holds a handful of samples
                tpb, pb = log_bin(tp[keep], np.abs(p[keep]) * 1e12, nbin=45)
                b.plot(tpb, pb, color=col, lw=0.8, ls=":" if hc else (0, (1, 3)), alpha=0.8)
            print(f"   {name} TL High-C {'on ' if hc else 'off'}  I={_num(row['settled'])*1e9:7.1f} nA  "
                  f"P={_num(row['power_W'])*1e12:7.1f} pW  ripple={100*_num(row['ripple']):5.2f} %")
    a.set_xscale("log"); a.set_yscale("log")
    a.set_xlabel("t after the step  [s, log]")
    a.set_ylabel("|I| on ch B, log-binned  [nA]")
    b.set_ylabel("optical power, log-binned  [pW]  (dotted)")
    b.set_ylim(0, 400)
    a.set_title("TL, long capture at −0.80 V: current (solid/dashed, left) and light "
                "(dotted, right)\nArray42's two current traces sit 13x apart; its two light "
                "traces lie on top of each other", fontsize=9.5)
    a.grid(which="both", alpha=0.25)
    a.legend(fontsize=7, loc="center right")
    fig.tight_layout()
    fig.savefig(OUT / "fig42_array42_nohighc.png", dpi=DPI)
    plt.close(fig)


def not_settled_reasons():
    """Why a row is flagged NOT SETTLED, by array, stage and High-C state."""
    print("\nNOT SETTLED rows by reason (t_1pct NaN = never inside the band / tail drift >= 1 %):")
    for name, stamp in (("Array42", "261021-1237"), ("Array38", "261021-1820")):
        r = run(stamp)
        tally = {}
        for row in r.step:
            key = (row["stage"], "High-C" if row["highc"] == "True" else "no High-C")
            n, bad, nan_t, drift = tally.get(key, (0, 0, 0, 0))
            ok = row.get("settled_ok") == "True"
            t1 = _num(row["t_1pct"]); dr = _num(row["tail_drift_frac"])
            tally[key] = (n + 1, bad + (not ok), nan_t + (not ok and not np.isfinite(t1)),
                          drift + (not ok and np.isfinite(t1) and np.isfinite(dr) and dr >= 0.01))
        for (stage, hc), (n, bad, nt, dr) in sorted(tally.items()):
            print(f"   {name} {stage:4} {hc:9}  {bad:2}/{n:2} flagged   t1% NaN: {nt:2}   drift>=1%: {dr:2}")


if __name__ == "__main__":
    OUT.mkdir(exist_ok=True)
    fig40()
    fig41()
    fig42()
    not_settled_reasons()
