"""Open question 16: the sweep's 2 s leakage reference reads low. By how much, really.

The question came out of a 600 s trace and a sweep measured in different campaigns:
the trace read 261.95 nA in the sweep's own 2-4 s window, the sweep's reference read
245.47 nA, a 6.7 % gap. Two runs that far apart cannot separate a systematic error
from the leakage simply having moved, so the pair was repeated on one afternoon:

    261016-1253  refgap_trace   pixel 35,36 -> bitval 0 at -0.85 V, 3000 samples
    261016-1315  refgap_sweep   the ordinary sweep, same four pixels, same grid

What the pair says:

  * THE TRANSIENT IS OVER IN UNDER A SECOND. The first second reads +6.66 % above
    the asymptote; 1-2 s already reads +0.01 %. The sweep waits 2 s, so its
    reference is NOT taken during a transient. Open question 16's settling
    hypothesis is dead.
  * A SMALLER GAP IS REAL AND SYSTEMATIC. Asymptote 274.350 +- 0.013 nA against
    four pixels' 2 s references of 269.4 to 270.3 nA: +1.49 to +1.84 %, mean
    +1.59 %, every pixel the same sign. It is 30 sigma on the quoted errors.

    One caveat on the record itself: --switch-hold-s asked for 1200 s but
    --samples 3000 at NPLC 1.0 is 40.7 ms per sample, so the capture stops at
    122 s. That is ample for this question -- the trace is flat from 1 s -- but it
    is NOT the 1200 s record the run list claimed. 29500 samples would be.
  * 6.7 % WAS MOSTLY DRIFT, NOT BIAS. The leakage itself moved from ~246 nA in the
    earlier campaign to ~270 nA here, about 10 %. Comparing runs 20 minutes apart
    instead of days apart shrinks the gap by 4x.

What it costs, which is the only number that matters for IQE. The reference is
subtracted from the raw reading, so its error scales by ref/(raw-ref):

    -2.60 V   ref is  0.6 % of raw   ->  I_LED off by 0.01 %
    -1.00 V   ref is  5.1 % of raw   ->  I_LED off by 0.09 %
    -0.90 V   ref is 14.6 % of raw   ->  I_LED off by 0.27 %
    -0.85 V   ref is 29.2 % of raw   ->  I_LED off by 0.66 %
    -0.80 V   ref is 57.6 % of raw   ->  I_LED off by 2.16 %

So it is negligible above -1.0 V and it is the dominant systematic at -0.80 V,
where it is larger than the direction effect (section 24) and the switch deficit
(section 22) combined.

    python analysis/plot_refgap.py
"""
import glob
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

MEAS = Path(__file__).resolve().parent.parent
OUT = Path(__file__).resolve().parent / "investigation_report"
OVF = 1e30
TRACE = "demo/refgap_trace"
SWEEP = "demo/refgap_sweep"
# The sweep's leak reference waits this long before it measures (leak_ref_settle_s).
REF_WAIT_S = 2.0
# Windows worth printing. 300-600 and beyond stay empty on purpose -- see note below.
WINDOWS = [(0, 1), (1, 2), (2, 4), (0, 2), (2, 10), (10, 30), (30, 60), (60, 120)]


def trace():
    """``(t, |I_nvled|)`` from the switch capture, over-range samples dropped.

    ``traces.npz`` rows are ``[t, chA, chB]`` and the NVLED channel is row 2.
    Row 1 is the VLED supply rail, which carries ~493 uA of array background here
    -- a factor 1800 larger, so reading the wrong row looks like a working trace
    and silently answers a different question.
    """
    z = np.load(glob.glob(str(MEAS / TRACE / "*/traces.npz"))[0])
    a = z[z.files[0]]
    t, i = a[0], np.abs(a[2])
    m = np.isfinite(i) & (i < OVF)
    return t[m], i[m]


def asymptote(t, i, last_s=120.0):
    """``(mean, standard error)`` over the final ``last_s`` of the record."""
    w = t >= t[-1] - last_s
    return float(i[w].mean()), float(i[w].std(ddof=1) / np.sqrt(w.sum()))


def sweep_refs():
    """The ordinary sweep, with the reference's share of the raw reading per bias."""
    f = glob.glob(str(MEAS / SWEEP / "*/NVLEDSweep/Sweep"
                      / "*aggregated_bitval=15_mean_std.csv"))[0]
    d = pd.read_csv(f)
    d["ref"] = d.NVLED_CURR_REF.abs()
    d["raw"] = d.NVLED_CURR_MEAN.abs()
    d["I"] = d.raw - d.ref
    d["share"] = d.ref / d.raw
    return d


def panel_trace(ax, t, i, asy, refs):
    """The record on a log time axis, against what the sweep believes."""
    ax.plot(t[t > 0], 1e9 * i[t > 0], lw=0.5, color="0.6", label="samples, 40.7 ms")
    for a, b in WINDOWS:
        w = (t >= a) & (t < b)
        if w.sum() > 2:
            ax.plot([max(a, 0.02), b], [1e9 * i[w].mean()] * 2, lw=2.4, color="C0")
    ax.axhline(1e9 * asy, color="C2", lw=1.4,
               label="asymptote, last 120 s: %.3f nA" % (1e9 * asy))
    r = refs[np.isclose(refs.NVLED_V, -0.85)].ref
    ax.axhspan(1e9 * r.min(), 1e9 * r.max(), color="C3", alpha=0.25,
               label="the sweep's own 2 s references, 4 pixels")
    ax.axvline(REF_WAIT_S, color="C3", ls=":", lw=1.2)
    ax.text(REF_WAIT_S * 1.15, 1e9 * asy * 1.035, "the sweep measures here",
            fontsize=6.5, color="C3")
    ax.set_xscale("log")
    ax.set_ylim(1e9 * asy * 0.975, 1e9 * asy * 1.07)
    ax.set_xlabel("time since the frame went to bitval 0 [s]")
    ax.set_ylabel("|NVLED current| [nA]")
    ax.set_title("A. the transient ends before 1 s, and a gap is still left\n"
                 "thick bars are the printed windows; the 2 s mark is not in a "
                 "transient", fontsize=9)
    ax.legend(fontsize=6.5, loc="upper right")
    ax.grid(alpha=0.3, which="both")


def panel_windows(ax, t, i, asy):
    """Every window as a deviation, so "settled" is a number and not a look."""
    print("\n=== the trace, windows against its own asymptote")
    print("  %10s %11s %8s %6s %10s" % ("window s", "mean nA", "sem", "n", "vs asy %"))
    xs, ys = [], []
    for a, b in WINDOWS:
        w = (t >= a) & (t < b)
        if w.sum() < 3:
            continue
        mu = i[w].mean()
        se = i[w].std(ddof=1) / np.sqrt(w.sum())
        print("  %10s %11.3f %8.3f %6d %+10.2f"
              % ("%g-%g" % (a, b), 1e9 * mu, 1e9 * se, w.sum(),
                 100 * (mu - asy) / asy))
        xs.append("%g-%g" % (a, b))
        ys.append(100 * (mu - asy) / asy)
    c = ["C3" if abs(y) > 0.3 else "C0" for y in ys]
    ax.bar(range(len(ys)), ys, color=c, edgecolor="k", linewidth=0.4)
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xticks(range(len(xs)))
    ax.set_xticklabels(xs, fontsize=7, rotation=30)
    ax.set_ylabel("window mean vs asymptote [%]")
    ax.set_xlabel("window [s]")
    ax.set_title("B. only the first second is out, by +6.66 %%\n"
                 "from 1 s on, every window sits within 0.11 %% of the asymptote",
                 fontsize=9)
    ax.grid(alpha=0.3, axis="y")


def panel_vs_bias(ax, refs, asy):
    """The reference against bias, split by measure range. Section 21, repeated."""
    print("\n=== the sweep's leak reference vs bias, split by measure range")
    for lim, g in refs.groupby("NVLED_LIMIT_A"):
        v = g.groupby("NVLED_V").ref.mean()
        print("  limit %-8g %2d biases  mean %7.2f nA  sd %6.2f nA"
              % (lim, len(v), 1e9 * v.mean(), 1e9 * v.std()))
        ax.plot(v.index, 1e9 * v.values, "o-", ms=5, lw=1.3,
                label="measure range %g A" % lim)
    ax.axhline(1e9 * asy, color="C2", lw=1.4, ls="--",
               label="the 122 s trace's asymptote")
    ax.set_xlabel("NVLED bias [V]")
    ax.set_ylabel("leak reference at bitval 0 [nA]")
    ax.set_title("C. the reference is flat in bias and steps with the RANGE\n"
                 "section 21 again: the apparent bias slope is a range artifact",
                 fontsize=9)
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3)


def panel_cost(ax, refs, gap_pct):
    """What the gap does to I_LED, which is the only number that matters.

    ``gap_pct`` is the ONE measured relative error of the 2 s reference,
    +1.59 % at -0.85 V on the 1e-5 range. It is propagated to the other biases
    rather than re-measured there, because that is the only long trace on disk.

    Deliberately NOT done here: comparing each bias's own reference against this
    one asymptote. Above -1.30 V the sweep is on the 1e-4 and 1e-3 ranges, where
    section 21 showed the reading is a range artifact (373 +- 168 nA on 1e-3), so
    that subtraction would report range scatter as a reference error.

    I_LED = raw - ref, so a relative error g on ref moves I_LED by
    -g * ref/(raw-ref). The share of the raw reading the reference takes is the
    whole amplification.
    """
    g = refs.groupby("NVLED_V").agg(ref=("ref", "mean"), raw=("raw", "mean"),
                                    I=("I", "mean"), share=("share", "mean"))
    g["amp"] = g.share / (1 - g.share)
    g["cost"] = -gap_pct * g.amp
    print(chr(10) + "=== what a %+.2f %% reference error costs I_LED" % gap_pct)
    print("  %8s %9s %9s %9s %7s %9s"
          % ("V", "ref nA", "share %", "I_LED nA", "amp", "cost %"))
    for v, r in g.iterrows():
        print("  %+8.3f %9.2f %9.1f %9.1f %7.3f %+9.2f"
              % (v, 1e9 * r.ref, 100 * r.share, 1e9 * r.I, r.amp, r.cost))
    ax.plot(g.index, np.abs(g.cost), "o-", ms=5, color="C3")
    ax.set_yscale("log")
    for v in (-0.80, -0.90, -1.00):
        ax.annotate("%.2f %%" % abs(g.cost[v]), xy=(v, abs(g.cost[v])),
                    xytext=(0, 7), textcoords="offset points", fontsize=7,
                    ha="center", color="C3")
    ax.axhline(0.1, color="0.5", ls=":", lw=1)
    ax.text(-2.55, 0.11, "0.1 %: below the 2 %/day drift", fontsize=6.5, color="0.4")
    ax.set_xlabel("NVLED bias [V]")
    ax.set_ylabel("|error on I_LED| from the %+.2f %% reference error [%%]"
                  % gap_pct)
    ax.set_title("D. negligible above -1.0 V, dominant at -0.80 V\n"
                 "the reference is 57.6 %% of the raw reading there", fontsize=9)
    ax.grid(alpha=0.3, which="both")


def demo():
    """The two things this section rests on, on inputs with a known answer."""
    # a flat trace: the asymptote is its level and no window deviates
    t = np.arange(0, 122, 0.0407)
    flat = np.full_like(t, 274.35e-9)
    a, se = asymptote(t, flat)
    assert abs(a - 274.35e-9) < 1e-15, a
    assert se < 1e-18, "a flat trace has no standard error"

    # a trace with a known 1 s excess must report that excess and nothing else
    y = np.full_like(t, 100e-9)
    y[t < 1.0] *= 1.0666
    a, _ = asymptote(t, y)
    w0 = (t >= 0) & (t < 1)
    w1 = (t >= 1) & (t < 2)
    assert abs(100 * (y[w0].mean() - a) / a - 6.66) < 0.01, "lost the known excess"
    assert abs(100 * (y[w1].mean() - a) / a) < 1e-9, "invented one after it"

    # and the real record must be flat after 1 s, or the section's claim is wrong
    t, i = trace()
    a, _ = asymptote(t, i)
    for lo, hi in ((1, 2), (2, 4), (10, 30), (60, 120)):
        w = (t >= lo) & (t < hi)
        d = abs(100 * (i[w].mean() - a) / a)
        assert d < 0.15, "window %g-%g s is %.2f %% off the asymptote" % (lo, hi, d)
    w = (t >= 0) & (t < 1)
    assert 100 * (i[w].mean() - a) / a > 3.0, "the first second must be high"
    print("demo: ok")


if __name__ == "__main__":
    demo()
    OUT.mkdir(exist_ok=True)
    t, i = trace()
    asy, asy_se = asymptote(t, i)
    refs = sweep_refs()
    print("trace: %d samples, %.1f s, %.2f ms/sample"
          % (len(t), t[-1], 1e3 * np.median(np.diff(t))))
    print("asymptote: %.3f +- %.3f nA" % (1e9 * asy, 1e9 * asy_se))
    r = refs[np.isclose(refs.NVLED_V, -0.85)]
    print("\n=== the gap, per pixel, at the trace's own bias")
    for _, x in r.iterrows():
        print("  pixel %d,%d  2 s ref %7.3f +- %.3f nA   gap %+6.2f %%"
              % (x.X, x.Y, 1e9 * x.ref, 1e9 * x.NVLED_CURR_REF_STD,
                 100 * (asy - x.ref) / x.ref))
    print("  mean gap over the four pixels: %+.2f %%"
          % (100 * (asy - r.ref.mean()) / r.ref.mean()))

    fig, axs = plt.subplots(2, 2, figsize=(13, 10))
    panel_trace(axs[0, 0], t, i, asy, refs)
    panel_windows(axs[0, 1], t, i, asy)
    panel_vs_bias(axs[1, 0], refs, asy)
    panel_cost(axs[1, 1], refs,
               100 * (asy - r.ref.mean()) / r.ref.mean())
    fig.suptitle("The leakage reference gap, runs 261016-1253 and -1315 "
                 "(section 25)", fontsize=12)
    fig.tight_layout()
    p = OUT / "fig39_refgap.png"
    fig.savefig(p, dpi=130)
    print("\nwrote %s" % p)
