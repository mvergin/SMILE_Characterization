"""The four follow-up runs of 16 October 2026, and section 21.

Three things the five bench runs of section 20 could not settle, settled here:

  Q4b  the frame-switch deficit, at two more biases, with compliance headroom.
       Section 20 had ONE bias, and its second was voided by the compliance defect.
  Q5b  the leakage at the bright end, which no record had ever reached.
  Q1   the bitval-0 frame-write failure rate, counted over 128 writes, which is
       what open question 9 asks and what the bitval-1 attempt could not test.

It also prints the two things nobody asked for and the runs answered anyway: the
16-minute repeatability of the whole pipeline, and the fact that `block_seed` does
not pin the pixel set.

    python plot_followup.py
"""
import glob
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

MEAS = Path(r"C:\_D\programs\SMILE_Characterization")
OUT = Path("investigation_report")
OVF = 1e30           # the SMU writes 9.91e37 for an over-range reading
TAIL_FRAC = 0.1      # the last tenth of a record is its reference
PROD_SWITCH_S = 2.0  # config.pixel_switch_settle_s, the value in production
GATE = 0.25          # config.leak_ref_max_frac, the retry gate


def traces(tag):
    """{key: (t, |I_cathode|)} with the over-range marker removed from both in step."""
    z = np.load(glob.glob(str(MEAS / "demo" / tag / "*" / "traces.npz"))[0])
    out = {}
    for k in z.files:
        a = z[k]
        t, y = a[0], np.abs(a[2])
        m = np.isfinite(y) & (y < OVF)
        out[k] = (t[m], y[m])
    return out


def sweep(tag):
    f = glob.glob(str(MEAS / "demo" / tag / "*" / "NVLEDSweep" / "Sweep"
                      / "*aggregated_bitval=15_mean_std.csv"))[0]
    return pd.read_csv(f)


def tail(y, frac=TAIL_FRAC):
    n = max(4, int(len(y) * frac))
    a = y[-n:]
    return float(a.mean()), float(a.std(ddof=1) / np.sqrt(n))


def deficit(t, y, edges):
    """Per-window (centre, % vs tail, % SEM, n). The tail is the reference because
    the transient is still alive at 60 s, so any wider reference divides it out."""
    ref, _ = tail(y)
    out = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        w = (t >= lo) & (t < hi)
        if w.sum() < 2:
            continue
        a = y[w]
        out.append((np.sqrt(lo * hi) if lo > 0 else hi / 2,
                    100 * (a.mean() / ref - 1),
                    100 * a.std(ddof=1) / np.sqrt(w.sum()) / ref, int(w.sum())))
    return out


def bias_of(key):
    return float(key.split("_v")[1][:6])


EDGES = [0, 0.5, 1, 2, 5, 10, 20, 40, 60, 90, 122]


def q4(ax):
    """Three biases, two runs, one shape: ~-0.5 % at 2 s decaying over ~60 s."""
    print("=== Q4b: the frame-switch deficit, now at three biases")
    series = [("q4_switch120", "-0.850", "s20"), ("q4b_switch", "-0.800", "s21"),
              ("q4b_switch", "-1.050", "s21")]
    for tag, v, src in series:
        tr = traces(tag)
        k = next(k for k in tr if "_v" + v in k)
        t, y = tr[k]
        ref, sem = tail(y)
        d = deficit(t, y, EDGES)
        ax.errorbar([a[0] for a in d], [a[1] for a in d],
                    yerr=[3 * a[2] for a in d], marker="o", ms=4, lw=1.4, capsize=2,
                    label="%+.2f V  (%.0f nA)" % (float(v), 1e9 * ref))
        print("  %+.2f V  settled %8.2f nA  tail SEM %.3f %%  [%s]"
              % (float(v), 1e9 * ref, 100 * sem / ref, src))
        for c0, dev, s, n in d:
            print("      t~%7.2f s  %+7.3f %% +- %.3f  (n=%d)" % (c0, dev, s, n))
    ax.axvline(PROD_SWITCH_S, color="k", ls=":", lw=1)
    ax.text(PROD_SWITCH_S * 1.15, -0.95,
            "production\npixel_switch_settle_s = %g s" % PROD_SWITCH_S,
            fontsize=7, va="bottom")
    ax.axhline(0, color="k", lw=0.6)
    ax.axhspan(-0.1, 0.1, color="0.85", zorder=0)
    ax.set_xscale("log")
    # the 0-0.5 s window is 13 samples with a +-0.9 % error bar. Keeping it on
    # scale hides the -0.5 % plateau the panel exists to show, so clip and say so.
    ax.set_ylim(-1.3, 0.45)
    ax.text(0.22, -1.25, "first 13 samples\nrun off-scale", fontsize=6,
            color="0.4", va="bottom")
    ax.set_xlabel("time after the frame switch [s]")
    ax.set_ylabel("current vs the record's tail [%]")
    ax.set_title("A. the switch deficit: -0.5 % at 2 s, 60 s to clear\n"
                 "grey band = +-0.1 %, the level section 17 cared about", fontsize=9)
    ax.legend(fontsize=7, title="NPLC 1.0, 1200 s hold", title_fontsize=7)
    ax.grid(alpha=0.3, which="both")


def q5(ax):
    """The leakage is a ~250 nA offset that the bias barely moves below -2 V."""
    print("\n=== Q5b: the leakage against bias, from three independent sources")
    pts = []
    for tag in ("q5_leaktrace", "q5b_leakbright"):
        for k, (t, y) in traces(tag).items():
            v = bias_of(k)
            ref, sem = tail(y)
            # the -1.300 V record in q5_leaktrace held its LIT state at the 1e-5
            # default, so the pre-switch state was clamped: section 20 voided it
            ok = not (tag == "q5_leaktrace" and v < -1.0)
            pts.append((v, 1e9 * ref, 1e9 * sem, tag, ok))
            print("  %+.2f V  %7.2f +- %.2f nA  %s%s"
                  % (v, 1e9 * ref, 1e9 * sem, tag,
                     "" if ok else "   (VOID: lit state clamped at 1e-5)"))
    good = [p for p in pts if p[4]]
    ax.errorbar([p[0] for p in good], [p[1] for p in good],
                yerr=[3 * p[2] for p in good], fmt="o", ms=7, capsize=3, lw=1.5,
                color="C3", label="600 s dedicated trace, 3 SEM", zorder=5)
    for p in pts:
        if not p[4]:
            ax.plot(p[0], p[1], "x", ms=8, color="0.6",
                    label="voided by the compliance defect", zorder=4)

    # The sweep's per-point reference, split BY MEASURE RANGE. Pooling the ranges
    # invents a bias dependence: the reference is 224-260 nA across the 1e-4 and
    # 1e-3 points and 244-247 nA across the 1e-5 ones, and the difference is the
    # range's own noise, not the pixel's.
    d = pd.concat([sweep(t) for t in ("tries_a", "tries_b")])
    g = d.groupby(d.NVLED_V.round(3)).agg(
        ref=("NVLED_CURR_REF", "median"), sd=("NVLED_CURR_REF", "std"),
        own=("NVLED_CURR_REF_STD", "median"), lim=("NVLED_LIMIT_A", "median"),
        n=("NVLED_CURR_REF", "size"))
    print("\n  the sweep's own 2 s reference, both runs pooled, BY RANGE:")
    print("  %8s %9s %9s %8s %10s" % ("V", "limit A", "ref nA", "sd nA", "own sd nA"))
    for v, r in g.iterrows():
        print("  %+8.3f %9.0e %9.2f %8.2f %10.2f"
              % (v, r.lim, -1e9 * r.ref, 1e9 * r.sd, 1e9 * r.own))
    for lim, mk, c in ((1e-5, "s", "C0"), (1e-4, "^", "C1"), (1e-3, "v", "C4")):
        q = g[g.lim == lim]
        ax.errorbar(q.index, -1e9 * q.ref, yerr=3e9 * q.sd / np.sqrt(q.n),
                    fmt=mk + "-", ms=5, lw=1.2, capsize=2, color=c,
                    label="2 s reference, limit %.0e (n=8)" % lim)
    f = g[g.lim == 1e-5]
    print("\n  on the 1e-5 range, where it can be measured (%d biases, %+.2f..%+.2f V):"
          % (len(f), f.index.max(), f.index.min()))
    print("    %.1f to %.1f nA, a %.1f %% spread -- flat"
          % (-1e9 * f.ref.max(), -1e9 * f.ref.min(),
             100 * (f.ref.min() / f.ref.max() - 1)))

    # The gap this run opened and cannot close: the 600 s trace and the sweep's 2 s
    # read disagree at the SAME bias on the SAME range, and it is not settling.
    tr = traces("q5_leaktrace")
    t, y = tr[next(k for k in tr if "-0.850" in k)]
    w = (t >= 2) & (t < 4)                       # the sweep's own reference window
    trace_2_4 = 1e9 * y[w].mean()
    sweep_2s = -1e9 * g.loc[-0.85, "ref"]
    print("\n  !! the two ways of measuring the same leakage DISAGREE at -0.85 V:")
    print("     600 s trace, its 2-4 s window: %.2f nA" % trace_2_4)
    print("     sweep's 2 s reference:         %.2f nA" % sweep_2s)
    print("     gap %.1f %%, and NOT settling: the trace is flat to 0.13 %% from 1 s"
          % (100 * (trace_2_4 / sweep_2s - 1)))
    ax.set_yscale("log")
    ax.set_yticks([200, 250, 300, 400, 600])
    ax.get_yaxis().set_major_formatter(plt.ScalarFormatter())
    ax.set_xlabel("NVLED bias [V]")
    ax.set_ylabel("leakage at bitval 0 [nA]")
    ax.set_title("B. the leakage is a ~250 nA offset, not a diode\n"
                 "the apparent slope above -1 V is the measure range, and the\n"
                 "two methods disagree by 6.7 % at -0.85 V", fontsize=9)
    ax.legend(fontsize=6.5)
    ax.grid(alpha=0.3)


def q1(ax):
    """128 bitval-0 writes, zero retries. The gate's margin is a bias effect."""
    print("\n=== Q1: the bitval-0 frame-write failure rate, over 128 writes")
    d = pd.concat([sweep(t).assign(run=t) for t in ("tries_a", "tries_b")])
    n = len(d)
    retried = int((d.LEAK_REF_TRIES > 1).sum())
    print("  LEAK_REF_TRIES: %s"
          % d.LEAK_REF_TRIES.value_counts().sort_index().to_dict())
    print("  %d retries in %d writes.  95 %% upper bound (rule of three): %.1f %% "
          "per write" % (retried, n, 300 / n))
    g = d.groupby(d.NVLED_V.round(3)).LEAK_REF_FRAC.agg(["median", "max"])
    ax.plot(g.index, 100 * g["max"], "o-", ms=4, color="C2",
            label="worst of 8 writes per bias")
    ax.plot(g.index, 100 * g["median"], "s--", ms=3, lw=1, color="C2", alpha=0.5,
            label="median of 8")
    ax.axhline(100 * GATE, color="C3", lw=1.5,
               label="leak_ref_max_frac = %g: the retry gate" % GATE)
    ax.set_yscale("log")
    ax.set_xlabel("NVLED bias [V]")
    ax.set_ylabel("P_REF / P_lit  [%]")
    ax.set_title("C. %d retries in %d writes\nthe gate's margin shrinks because "
                 "P_lit falls, not because\nthe reference brightens" % (retried, n),
                 fontsize=9)
    ax.legend(fontsize=7, loc="upper left")
    ax.grid(alpha=0.3, which="both")
    print("  P_REF/P_lit per bias:")
    for v, r in g.iterrows():
        print("    %+7.3f V  median %7.4f %%  worst %7.4f %%"
              % (v, 100 * r["median"], 100 * r["max"]))
    worst = d.LEAK_REF_FRAC.max()
    print("  worst over all %d points: %.2f %%, a factor %.1f under the gate"
          % (n, 100 * worst, GATE / worst))


def repeat(ax):
    """tries_a vs tries_b: identical commands, 16 minutes apart."""
    print("\n=== the 16-minute repeatability of the whole pipeline")
    a, b = sweep("tries_a"), sweep("tries_b")
    pa = set(map(tuple, a[["X", "Y"]].drop_duplicates().values))
    pb = set(map(tuple, b[["X", "Y"]].drop_duplicates().values))
    print("  tries_a pixels %s" % sorted(pa))
    print("  tries_b pixels %s" % sorted(pb))
    print("  !! block_seed=0 and seed=0 gave %d of 4 the same pixels. The FPS start "
          "depends on the ALIVE MASK, which was 1013 then 1014." % len(pa & pb))
    m = a.merge(b, on=["X", "Y", "NVLED_V"], suffixes=("_a", "_b"))
    m["dI"] = 100 * (((m.NVLED_CURR_MEAN_b - m.NVLED_CURR_REF_b)
                      / (m.NVLED_CURR_MEAN_a - m.NVLED_CURR_REF_a)) - 1)
    m["dP"] = 100 * (((m.PM400_POWER_MEAN_b - m.PM400_POWER_REF_b)
                      / (m.PM400_POWER_MEAN_a - m.PM400_POWER_REF_a)) - 1)
    print("  %d matched points over %d shared pixels" % (len(m), len(pa & pb)))
    g = m.groupby(m.NVLED_V.round(3))
    for col, c, lab in (("dI", "C0", "I_LED"), ("dP", "C1", "P_LED")):
        ax.errorbar(g.NVLED_V.median(), g[col].median(), yerr=g[col].std(),
                    fmt="o-", ms=4, lw=1.2, capsize=2, color=c,
                    label="%s: median, bars = sd over 3 pixels" % lab)
    ax.axhline(0, color="k", lw=0.6)
    ax.axhspan(-0.5, 0.5, color="0.85", zorder=0)
    # P_LED at -0.80 V is -7.9 +- 8.7 %, on a 0.21 nW reading. On scale it buries
    # the 0.1-0.4 % the other fifteen biases repeat to.
    ax.set_ylim(-3.2, 1.6)
    ax.annotate("P_LED -7.9 +- 8.7 %\n(P_LED = 0.21 nW)", xy=(-0.80, -3.0),
                xytext=(-1.35, -2.6), fontsize=6.5, color="C1",
                arrowprops=dict(arrowstyle="->", color="C1", lw=0.8))
    ax.set_xlabel("NVLED bias [V]")
    ax.set_ylabel("run b / run a - 1  [%]")
    ax.set_title("D. 16 minutes apart, same command: current repeats to 0.4 %\n"
                 "light does not, below -0.9 V, where P_LED < 2 nW", fontsize=9)
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3)
    print("  %8s %8s %7s %8s %7s %9s"
          % ("V", "dI med", "dI sd", "dP med", "dP sd", "P_lit nW"))
    for v, q in g:
        print("  %+8.3f %+8.2f %7.2f %+8.2f %7.2f %9.3f"
              % (v, q.dI.median(), q.dI.std(), q.dP.median(), q.dP.std(),
                 1e9 * q.PM400_POWER_MEAN_a.median()))
    print("  overall: dI sd %.2f %%   dP sd %.2f %%" % (m.dI.std(), m.dP.std()))
    bright = m[m.NVLED_V <= -1.3]
    print("  at -2.60..-1.30 V only: dI sd %.2f %%   dP sd %.2f %%"
          % (bright.dI.std(), bright.dP.std()))


def demo():
    """What the helpers promise, on inputs whose answers are known by hand."""
    y = np.array([1.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 2.0])
    assert tail(y, 0.5) == (2.0, 0.0), tail(y, 0.5)
    # tail() must never read fewer than 4 samples, however small frac is
    assert tail(y, 0.01)[0] == 2.0

    # deficit() measures against the TAIL, so a step that ends high reads negative
    t = np.linspace(0, 100, 1000)
    s = np.where(t < 10, 99.0, 100.0)
    d = [dev for _, dev, _, _ in deficit(t, s, [0, 5, 20, 100])]
    assert abs(min(d) + 1.0) < 1e-9, d       # the early window is -1.00 %
    assert abs(max(d)) < 1e-9, d             # the late one is exactly 0

    # the over-range marker must leave t and y the same length
    z = np.array([[0.0, 1.0, 2.0], [0.0, 0.0, 0.0], [1e-9, 9.91e37, 2e-9]])
    m = np.isfinite(np.abs(z[2])) & (np.abs(z[2]) < OVF)
    assert z[0][m].size == np.abs(z[2])[m].size == 2

    assert bias_of("000_switch_hc1_v-0.800_39-40to35-36_b15_1") == -0.8
    assert bias_of("001_switch_hc1_v-1.800_35-36to35-36_b0_1") == -1.8
    print("demo: ok")


if __name__ == "__main__":
    demo()
    OUT.mkdir(exist_ok=True)
    fig, axs = plt.subplots(2, 2, figsize=(13, 10))
    q4(axs[0, 0])
    q5(axs[0, 1])
    q1(axs[1, 0])
    repeat(axs[1, 1])
    fig.suptitle("The four follow-up runs, 16 October 2026 (section 21)", fontsize=12)
    fig.tight_layout()
    p = OUT / "fig35_followup.png"
    fig.savefig(p, dpi=130)
    print("\nwrote %s" % p)
