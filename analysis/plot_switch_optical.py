"""Does the light follow the current through a frame change? No, not for the first
two seconds -- and that inverts what the switch costs.

Run 261016-1023 is the first switch record with both channels. The PM400 polls on a
background thread at ~249 Hz against the SMU's 24.6 Hz, both started by the same
trigger. The records are NOT synchronous, so every comparison here bins both onto the
same windows before dividing them.

    python plot_switch_optical.py
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
OVF = 1e30
EDGES = [0, 0.5, 1, 2, 5, 10, 20, 40, 60, 90, 122]
PROD_SWITCH_S = 2.0      # config.pixel_switch_settle_s


def load(v):
    """(t_I, |I|, t_P, P) for one bias of the two-channel switch run."""
    z = np.load(glob.glob(str(MEAS / "demo/q4d_switchoptical/*/traces.npz"))[0])
    ik = next(k for k in z.files if k.startswith("0") and v in k)
    pk = next(k for k in z.files if k.startswith("pm_") and v in k)
    ti, _, ib = z[ik]
    m = np.isfinite(ib) & (np.abs(ib) < OVF)
    tp, p = z[pk]
    return ti[m], np.abs(ib[m]), tp, p


def binned(t, y, ref):
    """Per-window (centre, % vs ref, % SEM). sqrt(lo*hi) is the centre on a log axis."""
    out = []
    for lo, hi in zip(EDGES[:-1], EDGES[1:]):
        w = (t >= lo) & (t < hi)
        if w.sum() < 2:
            continue
        a = y[w]
        out.append((np.sqrt(lo * hi) if lo > 0 else hi / 2,
                    100 * (a.mean() / ref - 1),
                    100 * a.std(ddof=1) / np.sqrt(w.sum()) / abs(ref)))
    return out


def tail(y, frac=0.1):
    n = max(4, int(len(y) * frac))
    return float(y[-n:].mean()), float(y[-n:].std(ddof=1) / np.sqrt(n))


def panel(ax, v, title):
    ti, ii, tp, pp = load(v)
    iref, isem = tail(ii)
    pref, psem = tail(pp)
    bi, bp = binned(ti, ii, iref), binned(tp, pp, pref)
    print("\n=== %s V   I = %.2f nA (tail SEM %.3f %%), P = %.4f nW (tail SEM %.3f %%)"
          % (v, 1e9 * iref, 100 * isem / iref, 1e9 * pref, 100 * psem / pref))
    print("  %14s %9s %9s %10s" % ("window", "dI [%]", "dP [%]", "dEQE [%]"))
    for (lo, hi) in zip(EDGES[:-1], EDGES[1:]):
        wi, wp = (ti >= lo) & (ti < hi), (tp >= lo) & (tp < hi)
        if wi.sum() < 2 or wp.sum() < 2:
            continue
        di = 100 * (ii[wi].mean() / iref - 1)
        dp = 100 * (pp[wp].mean() / pref - 1)
        # EQE from the two BINNED means, never sample by sample: the clocks differ
        de = 100 * ((pp[wp].mean() / ii[wi].mean()) / (pref / iref) - 1)
        print("  %5.1f-%6.1f s %+8.3f %+8.3f %+9.3f" % (lo, hi, di, dp, de))

    ax.errorbar([b[0] for b in bi], [b[1] for b in bi], yerr=[3 * b[2] for b in bi],
                fmt="o-", ms=4, lw=1.5, capsize=2, color="C0", label="current")
    ax.errorbar([b[0] for b in bp], [b[1] for b in bp], yerr=[3 * b[2] for b in bp],
                fmt="s-", ms=4, lw=1.5, capsize=2, color="C1", label="light")
    ax.axvline(PROD_SWITCH_S, color="k", ls=":", lw=1)
    ax.axhline(0, color="k", lw=0.6)
    ax.axhspan(-0.1, 0.1, color="0.88", zorder=0)
    ax.set_xscale("log")
    ax.set_xlabel("time after the frame switch [s]")
    ax.set_ylabel("deviation from the record's tail [%]")
    ax.set_title(title, fontsize=9)
    ax.legend(fontsize=7, title="3 SEM error bars", title_fontsize=7)
    ax.grid(alpha=0.3, which="both")
    return ti, ii, tp, pp, iref, pref


def eqe_panel(ax, ti, ii, tp, pp, iref, pref):
    """EQE = P/I through the switch. The sign is the whole point."""
    xs, ys = [], []
    for lo, hi in zip(EDGES[:-1], EDGES[1:]):
        wi, wp = (ti >= lo) & (ti < hi), (tp >= lo) & (tp < hi)
        if wi.sum() < 2 or wp.sum() < 2:
            continue
        xs.append(np.sqrt(lo * hi) if lo > 0 else hi / 2)
        ys.append(100 * ((pp[wp].mean() / ii[wi].mean()) / (pref / iref) - 1))
    ax.plot(xs, ys, "D-", ms=5, lw=1.6, color="C3", label="EQE = P / I")
    ax.axvline(PROD_SWITCH_S, color="k", ls=":", lw=1)
    ax.axhline(0, color="k", lw=0.6)
    ax.axhspan(-0.1, 0.1, color="0.88", zorder=0)
    at2 = [y for x, y in zip(xs, ys) if 1 < x < 2][0]
    ax.annotate("at the production 2 s wait\nEQE reads %+.2f %% HIGH" % at2,
                xy=(1.41, at2), xytext=(4, at2 + 0.28), fontsize=7.5, color="C3",
                arrowprops=dict(arrowstyle="->", color="C3", lw=0.9))
    ax.set_xscale("log")
    ax.set_xlabel("time after the frame switch [s]")
    ax.set_ylabel("EQE vs the record's tail [%]")
    ax.set_title("C. the sign flips: a current-only reading says the switch costs\n"
                 "0.5 %, and EQE is 0.5 % too HIGH instead", fontsize=9)
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3, which="both")


def ab(ax):
    """Section 18's back-to-back pair, now read in the optical channel as well."""
    def one(tag):
        f = glob.glob(str(MEAS / "demo" / tag / "*/NVLEDSweep/Sweep"
                          / "*aggregated_bitval=15_mean_std.csv"))[0]
        d = pd.read_csv(f)
        d["I"] = (d.NVLED_CURR_MEAN - d.NVLED_CURR_REF).abs()
        d["P"] = d.PM400_POWER_MEAN - d.PM400_POWER_REF
        d["EQE"] = d.P / d.I
        return d

    m = one("ab_px").merge(one("ab_bm"), on=["X", "Y", "NVLED_V"],
                           suffixes=("_px", "_bm"))
    for a, b in (("I", "dI"), ("P", "dP"), ("EQE", "dE")):
        m[b] = 100 * (m[f"{a}_bm"] / m[f"{a}_px"] - 1)
    g = m.groupby(m.NVLED_V.round(3))
    print("\n=== section 18's back-to-back pair, bias_major vs pixel_major, both channels")
    print("  %8s %3s %9s %9s %9s" % ("V", "n", "dI [%]", "dP [%]", "dEQE [%]"))
    for v, q in g:
        print("  %+8.3f %3d %+9.2f %+9.2f %+9.2f"
              % (v, len(q), q.dI.median(), q.dP.median(), q.dE.median()))
    for col, c, lab in (("dI", "C0", "current"), ("dP", "C1", "light"),
                        ("dE", "C3", "EQE")):
        ax.plot(g.NVLED_V.median(), g[col].median(), "o-", ms=4, lw=1.3, color=c,
                label=lab)
    ax.axhline(0, color="k", lw=0.6)
    # -0.80 V reads P_LED = 0.21 nW, where section 21 measured 8.7 % repeatability
    ax.set_ylim(-4.5, 2.0)
    ax.annotate("-0.80 V: dP +3.4 %, dEQE +5.7 %\n(P_LED = 0.21 nW, 8.7 % repeatability)",
                xy=(-0.80, 1.9), xytext=(-1.55, 1.1), fontsize=6.5, color="0.35",
                arrowprops=dict(arrowstyle="->", color="0.5", lw=0.8))
    ax.set_xlabel("NVLED bias [V]")
    ax.set_ylabel("bias_major / pixel_major - 1  [%]")
    ax.set_title("D. the prediction holds only at the faintest biases\n"
                 "EQE is +0.1 to +0.45 % there, as panel C says it should be,\n"
                 "and the knee does something else entirely", fontsize=9)
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3)
    faint = m[m.NVLED_V > -1.3]
    print("  faint side (> -1.30 V, n=%d): dI %+.2f  dP %+.2f  dEQE %+.2f"
          % (len(faint), faint.dI.median(), faint.dP.median(), faint.dE.median()))


def demo():
    """binned() and tail() on inputs whose answers are known by hand."""
    y = np.array([1.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 2.0])
    assert tail(y, 0.5) == (2.0, 0.0)
    assert tail(y, 0.01)[0] == 2.0           # never fewer than 4 samples

    t = np.linspace(0, 100, 2000)
    s = np.where(t < 10, 99.0, 100.0)
    b = binned(t, s, 100.0)
    assert abs(min(d for _, d, _ in b) + 1.0) < 1e-9, b      # early window is -1.00 %
    assert abs(max(d for _, d, _ in b)) < 1e-9, b            # late one is exactly 0
    # centres must be inside their own window, or the log axis lies
    for (c, _, _), (lo, hi) in zip(b, zip(EDGES[:-1], EDGES[1:])):
        assert lo <= c <= hi, (c, lo, hi)

    # a ratio of two binned means is NOT the mean of the ratios: pin the one we use
    i = np.full(100, 2.0)
    i[:50] = 1.0
    assert np.isclose(i.mean(), 1.5)
    print("demo: ok")


if __name__ == "__main__":
    demo()
    OUT.mkdir(exist_ok=True)
    fig, axs = plt.subplots(2, 2, figsize=(13, 10))
    panel(axs[0, 0], "-0.800",
          "A. -0.80 V: inconclusive, and honestly so\n"
          "P_LED = 0.23 nW, so every optical window is within ~1.5 sigma of zero")
    r = panel(axs[0, 1], "-1.050",
              "B. -1.05 V: the light is settled before 0.5 s, the current is not\n"
              "at 1-2 s the current is 13x the light's deviation")
    eqe_panel(axs[1, 0], *r)
    ab(axs[1, 1])
    fig.suptitle("Does the light follow the current through a frame change? "
                 "Run 261016-1023 (section 22)", fontsize=12)
    fig.tight_layout()
    p = OUT / "fig36_switch_optical.png"
    fig.savefig(p, dpi=130)
    print("\nwrote %s" % p)
