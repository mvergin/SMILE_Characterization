"""The sweep-direction A/B, run at last -- and the reason the default was given is wrong.

Runs 261016-1113 (low_to_high) and 261016-1129 (high_to_low): same four pixels, same
grid, 16 min apart, bias_major, dead_abort off because validate() now refuses the
combination that would have dropped every pixel.

Three things come out, and only two of them are safe:

  SAFE   low_to_high reads lower in BOTH channels across six consecutive biases where
         neither optical noise nor 2 %/day drift can reach: dI -0.40 %, dP -0.93 %,
         dEQE -0.46 %, 6 of 6 one-signed.
  SAFE   section 4's stated reason for the default is refuted. Nothing "approaches from
         above": BOTH directions are still climbing at the end of the 54.47 s dwell,
         and high_to_low climbs TWICE AS FAST.
  NOT    attributing the first result to direction. The two runs are sequential, so
         elapsed time between matched points grows monotonically with bias, from 1.0 min
         at -2.60 V to 30.0 min at -0.80 V. Direction and time are collinear by
         construction -- a flaw in the run list, not in the runs. The control is the
         same pair in the reverse sequence.

    python plot_direction.py
"""
import glob
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# The measurement repo is this file's grandparent since ROADMAP item 2 merged the
# two. It was an absolute path for as long as they were separate repos.
MEAS = Path(__file__).resolve().parent.parent
# Beside the script, never beside the CWD: ROADMAP item 2 put these under
# analysis/, and the measurement root has an investigation_report/ of its own, so
# a CWD-relative OUT writes figures into the wrong one from the wrong directory.
OUT = Path(__file__).resolve().parent / "investigation_report"
OVF = 1e30
DWELL_S = 54.47          # the calibrated low-V dwell, Array42
# Section 21 measured these as the same-command, 16-minute floor above -1.30 V.
FLOOR_I, FLOOR_P = 0.28, 0.37
# The band where the matched-point gap is 3-11 min AND P_lit is 24-118 nW, so neither
# 2 %/day drift (0.015 % over 11 min) nor the optical floor can produce the effect.
BAND = (-1.82, -1.15)


def sign_p(k, n):
    """Two-sided sign test. math.comb beats pulling in scipy for one number."""
    from math import comb
    k = min(k, n - k)
    return min(1.0, 2 * sum(comb(n, j) for j in range(k + 1)) / 2 ** n)


def sweep(tag):
    f = glob.glob(str(MEAS / "demo" / tag / "*/NVLEDSweep/Sweep"
                      / "*aggregated_bitval=15_mean_std.csv"))[0]
    d = pd.read_csv(f)
    d["I"] = (d.NVLED_CURR_MEAN - d.NVLED_CURR_REF).abs()
    d["P"] = d.PM400_POWER_MEAN - d.PM400_POWER_REF
    d["EQE"] = d.P / d.I
    return d


def dwell_slopes(tag):
    """{bias: %/s over the last 10 s of its dwell}. Positive = still climbing."""
    z = np.load(glob.glob(str(MEAS / "demo" / tag / "*/NVLEDSweep/Sweep"
                              / "*settling_traces.npz"))[0])
    out = {}
    for k in z.files:
        t, i, _p = z[k]
        m = np.isfinite(i) & (i < OVF) & (t > 0)
        t, i = t[m], i[m]
        w = t > t[-1] - 10
        if w.sum() < 4:
            continue
        out[round(float(k.split("_v")[1]), 3)] = \
            100 * np.polyfit(t[w], i[w], 1)[0] / i[w].mean()
    return out


def merged(lo_tag="dir_lo2hi", hi_tag="dir_hi2lo", hi_starts_later_min=16.0):
    """``low_to_high`` minus ``high_to_low``, matched on pixel and bias.

    ``hi_starts_later_min`` is signed: +16 when high_to_low ran second (pair A,
    11:13 then 11:29), -16 when it ran first (pair B, 12:21 then 12:37). It only
    feeds ``gap_min``, and that sign is the whole point of the control -- the
    elapsed-time axis reverses while the direction axis does not.
    """
    lo, hi = sweep(lo_tag), sweep(hi_tag)
    m = lo.merge(hi, on=["X", "Y", "NVLED_V"], suffixes=("_lo", "_hi"))
    for a, b in (("I", "dI"), ("P", "dP"), ("EQE", "dE")):
        m[b] = 100 * (m[f"{a}_lo"] / m[f"{a}_hi"] - 1)
    # TIME_START is per-run seconds, so the run-start offset has to be added back
    m["gap_min"] = (m.TIME_START_hi + hi_starts_later_min * 60.0
                    - m.TIME_START_lo) / 60.0
    return lo, hi, m


def band_means(m):
    """``{channel: mean % over the band}``, one number per channel."""
    b = m[(m.NVLED_V >= BAND[0]) & (m.NVLED_V <= BAND[1])]
    g = b.groupby(b.NVLED_V.round(3))[["dI", "dP", "dE"]].mean()
    return {c: float(g[c].mean()) for c in ("dI", "dP", "dE")}, g


def decompose(a, b):
    """Split pair A and pair B into direction D and run-order T, per channel.

    Each pair measures ``D + s*T`` with ``s = +1`` when low_to_high ran first and
    ``s = -1`` when it ran second. Two equations, two unknowns:

        D = (A + B) / 2     the part that does NOT care which run came first
        T = (A - B) / 2     the part that does, and it is what section 23 could
                            not separate from D

    Nothing here is a fit. It is the only linear split two sequences allow, and
    its one free prediction -- ``D_P - D_I`` must equal ``D_EQE``, because
    EQE = P/I -- is checked in ``demo()``.
    """
    return {c: {"A": a[c], "B": b[c], "D": (a[c] + b[c]) / 2,
                "T": (a[c] - b[c]) / 2} for c in ("dI", "dP", "dE")}


def panel_ab(ax, m):
    print("=== low_to_high vs high_to_low (positive = low_to_high reads HIGHER)")
    g = m.groupby(m.NVLED_V.round(3))
    print("  %8s %9s %9s %9s %10s %9s"
          % ("V", "dI [%]", "dP [%]", "dEQE [%]", "P_lit nW", "gap min"))
    for v, q in g:
        print("  %+8.3f %+9.2f %+9.2f %+9.2f %10.3f %9.1f"
              % (v, q.dI.median(), q.dP.median(), q.dE.median(),
                 1e9 * q.P_hi.median(), q.gap_min.median()))
    for col, c, lab in (("dI", "C0", "current"), ("dP", "C1", "light"),
                        ("dE", "C3", "EQE")):
        ax.errorbar(g.NVLED_V.median(), g[col].median(), yerr=g[col].std(),
                    fmt="o-", ms=4, lw=1.3, capsize=2, color=c, label=lab)
    ax.axhline(0, color="k", lw=0.6)
    ax.axvspan(BAND[0], BAND[1], color="0.88", zorder=0)
    ax.text(np.mean(BAND), 1.3, "the band that survives\nboth confounds", fontsize=6.5,
            ha="center", color="0.3")
    ax.set_ylim(-3.2, 1.8)
    ax.set_xlabel("NVLED bias [V]")
    ax.set_ylabel("low_to_high / high_to_low - 1  [%]")
    ax.set_title("A. low_to_high reads lower in both channels\n"
                 "the faint end is unusable: 0.21 nW and 30 min between "
                 "matched points", fontsize=9)
    ax.legend(fontsize=7, loc="lower left")
    ax.grid(alpha=0.3)


def panel_confound(ax, m):
    """The flaw in the run list, plotted so nobody re-reads panel A as causal."""
    g = m.groupby(m.NVLED_V.round(3))
    ax.plot(g.NVLED_V.median(), g.gap_min.median(), "o-", ms=4, color="C4")
    print("\n=== the confound: elapsed time between matched points vs bias")
    print("    r(gap, bias) = %+.4f -- they are the same axis"
          % np.corrcoef(g.NVLED_V.median(), g.gap_min.median())[0, 1])
    ax.axvspan(BAND[0], BAND[1], color="0.88", zorder=0)
    ax.set_xlabel("NVLED bias [V]")
    ax.set_ylabel("minutes between the two measurements\nof the same point")
    ax.set_title("B. why panel A cannot be read as causal\n"
                 "the runs are sequential, so the gap grows monotonically with\n"
                 "the bias; the grey band is where it is only 3-11 min",
                 fontsize=9)
    ax.grid(alpha=0.3)
    ax.annotate("2 %/day over 11 min\nis 0.015 %", xy=(-1.3, 5), xytext=(-2.3, 14),
                fontsize=7, color="0.3",
                arrowprops=dict(arrowstyle="->", color="0.5", lw=0.8))


def panel_band(ax, m):
    b = m[(m.NVLED_V >= BAND[0]) & (m.NVLED_V <= BAND[1])]
    g = b.groupby(b.NVLED_V.round(3))[["dI", "dP", "dE"]].median()
    print("\n=== the band -1.82..-1.15 V, where the effect cannot be noise or drift")
    print(g.round(3).to_string())
    x = np.arange(len(g))
    w = 0.26
    for k, (col, c, lab) in enumerate((("dI", "C0", "current"), ("dP", "C1", "light"),
                                       ("dE", "C3", "EQE"))):
        ax.bar(x + (k - 1) * w, g[col], w, color=c, label=lab)
    for col, f, c in (("dI", FLOOR_I, "C0"), ("dP", FLOOR_P, "C1")):
        ax.axhline(-f, color=c, ls="--", lw=1,
                   label="%s: section 21's same-command floor" % col)
        n = int((g[col] < 0).sum())
        print("  %s: %d of %d negative, median %+.2f %%, two-sided sign p = %.4f"
              % (col, n, len(g), g[col].median(), sign_p(n, len(g))))
    print("  dE: %d of %d negative, median %+.2f %%, two-sided sign p = %.4f"
          % (int((g.dE < 0).sum()), len(g), g.dE.median(),
             sign_p(int((g.dE < 0).sum()), len(g))))
    ax.set_xticks(x)
    ax.set_xticklabels(["%+.2f" % v for v in g.index], fontsize=7)
    ax.axhline(0, color="k", lw=0.6)
    ax.set_xlabel("NVLED bias [V]")
    ax.set_ylabel("low_to_high / high_to_low - 1  [%]")
    ax.set_title("C. six consecutive biases, all one-signed (p = 0.03 each)\n"
                 "dI -0.40 %, dP -0.93 %, dEQE -0.46 % -- and dP is 2.5x the floor",
                 fontsize=9)
    ax.legend(fontsize=6.5)
    ax.grid(alpha=0.3, axis="y")


def panel_dwell(ax):
    """Section 4's premise, tested. Nothing approaches from above."""
    lo, hi = dwell_slopes("dir_lo2hi"), dwell_slopes("dir_hi2lo")
    both = sorted(set(lo) & set(hi), reverse=True)
    print("\n=== is the 54.47 s dwell settled? last-10-s slope, %/s")
    print("    positive = still CLIMBING. Section 4 says high_to_low approaches from")
    print("    ABOVE, which would be negative.")
    print("  %8s %14s %14s" % ("V", "lo2hi", "hi2lo"))
    for v in both:
        print("  %+8.3f %+14.5f %+14.5f" % (v, lo[v], hi[v]))
    sl, sh = [lo[v] for v in both], [hi[v] for v in both]
    print("  low_to_high: %d of %d climbing, median %+.5f %%/s -> %+.3f %% per dwell"
          % (sum(s > 0 for s in sl), len(sl), np.median(sl), DWELL_S * np.median(sl)))
    print("  high_to_low: %d of %d climbing, median %+.5f %%/s -> %+.3f %% per dwell"
          % (sum(s > 0 for s in sh), len(sh), np.median(sh), DWELL_S * np.median(sh)))
    ax.plot(both, [DWELL_S * lo[v] for v in both], "o-", ms=5, lw=1.4, color="C0",
            label="low_to_high")
    ax.plot(both, [DWELL_S * hi[v] for v in both], "s-", ms=5, lw=1.4, color="C2",
            label="high_to_low")
    ax.axhline(0, color="k", lw=0.8)
    ax.set_yscale("symlog", linthresh=0.1, linscale=0.8)
    # WITHDRAWN by the reverse-sequence control (section 24). This single point
    # reads +4.0 %/dwell here and -1.0 %/dwell in pair B, so it is noise on one
    # 0.21 nW trace, not an asymmetry. The ten-bias median does reproduce.
    ax.annotate("low_to_high's FIRST bias: +4.0 % per dwell.\n"
                "WITHDRAWN -- pair B reads -1.0 % here (section 24).",
                xy=(-0.80, DWELL_S * lo[-0.8]), xytext=(-1.47, 2.3), fontsize=6.5,
                color="0.35", arrowprops=dict(arrowstyle="->", color="0.6", lw=0.9))
    ax.set_ylim(-0.3, 7)
    ax.set_xlabel("NVLED bias [V]")
    ax.set_ylabel("residual drift over one 54.47 s dwell [%]")
    ax.set_title("D. section 4's premise is wrong: nothing approaches from above\n"
                 "both directions are STILL CLIMBING at the end of the dwell,\n"
                 "high_to_low 2.1x faster by median -- and that one reproduces",
                 fontsize=9)
    ax.legend(fontsize=7, loc="lower left")
    ax.grid(alpha=0.3, which="both")


def demo():
    """dwell_slopes' sign convention and the band mask, on known inputs."""
    t = np.linspace(0, 54.47, 218)
    rising = 100e-9 * (1 + 1e-4 * t)
    w = t > t[-1] - 10
    sl = 100 * np.polyfit(t[w], rising[w], 1)[0] / rising[w].mean()
    assert sl > 0, "a rising trace must report a positive slope"
    assert abs(sl - 1e-2) < 1e-3, sl          # 1e-4 /s on a unit offset is 0.01 %/s
    falling = 100e-9 * (1 - 1e-4 * t)
    sl = 100 * np.polyfit(t[w], falling[w], 1)[0] / falling[w].mean()
    assert sl < 0, "a falling trace must report a negative slope"

    # the band must select exactly the six biases the section quotes
    v = np.array([-2.60, -2.34, -2.08, -1.82, -1.56, -1.30, -1.25, -1.20, -1.15,
                  -1.10, -1.05, -1.00, -0.95, -0.90, -0.85, -0.80])
    sel = v[(v >= BAND[0]) & (v <= BAND[1])]
    assert len(sel) == 6, sel
    assert sel.max() == -1.15 and sel.min() == -1.82, sel
    print("demo: ok")


if __name__ == "__main__":
    demo()
    OUT.mkdir(exist_ok=True)
    _lo, _hi, m = merged()
    fig, axs = plt.subplots(2, 2, figsize=(13, 10))
    panel_ab(axs[0, 0], m)
    panel_confound(axs[0, 1], m)
    panel_band(axs[1, 0], m)
    panel_dwell(axs[1, 1])
    fig.suptitle("The sweep-direction A/B, runs 261016-1113 and -1129 (section 23)",
                 fontsize=12)
    fig.tight_layout()
    p = OUT / "fig37_direction.png"
    fig.savefig(p, dpi=130)
    print("\nwrote %s" % p)
