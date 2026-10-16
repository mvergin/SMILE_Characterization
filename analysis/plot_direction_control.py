"""The reverse-sequence control for section 23, and what it leaves standing.

Section 23 measured four pixels twice, `low_to_high` then `high_to_low`, and found
`low_to_high` reading low in both channels. It could not say why. The two runs are
sequential, so "which direction" and "how long ago" are the same axis -- open
question 18. The control is the identical pair in the opposite order:

    pair A   261016-1113 low_to_high  ->  261016-1129 high_to_low
    pair B   261016-1221 high_to_low  ->  261016-1237 low_to_high

Each pair measures `D + s*T`, with `s = +1` when `low_to_high` ran first. Two
sequences, two unknowns, so the split is arithmetic rather than a fit:

    channel      pair A    pair B    D (direction)   T (run order)
    current      -0.390    +0.336        -0.027          -0.363
    light        -0.916    -0.181        -0.548          -0.367
    EQE          -0.528    -0.516        -0.522          -0.006

Three results, and the first one overturns section 23's headline:

  * `dI` FLIPS SIGN with the order. The current difference section 23 reported was
    the run-order term, not direction. Direction moves the current by -0.027 %,
    which is nothing.
  * `dEQE` DOES NOT FLIP: -0.528 then -0.516, and per bias it repeats to 0.05 pp
    across the band (at -1.30 V, -0.739 % then -0.730 %, a 0.009 pp spread).
    12 of 12 one-signed, sign p = 0.0005.
    The direction effect is real and it is purely optical.
  * T is common mode: -0.363 % on the current, -0.367 % on the light, so it cancels
    to -0.006 % in EQE. Whichever run goes second reads ~0.37 % higher in BOTH
    channels. It does not scale with the gap -- pair A averages 6 min between
    matched points and pair B 26 min, for the same T -- so it is a saturating
    warm-up, not a linear drift.

The one free prediction, checked in `demo()`: EQE = P/I forces `D_P - D_I = D_EQE`.
Measured, -0.521 against -0.522.

    python analysis/plot_direction_control.py
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from plot_direction import (BAND, DWELL_S, OUT, band_means, decompose,
                            dwell_slopes, merged, sign_p)

# pair A ran low_to_high first, pair B ran it second. The sign is the control.
PAIR_A = ("dir_lo2hi", "dir_hi2lo", +16.0)
PAIR_B = ("dir_lo2hi_b", "dir_hi2lo_b", -16.0)
CH = (("dI", "C0", "current"), ("dP", "C1", "light"), ("dE", "C3", "EQE"))


def per_bias(m, col):
    """``(biases, values)`` sorted by bias, mean over the four pixels."""
    g = m.groupby(m.NVLED_V.round(3))[col].mean()
    return g.index.values, g.values


def panel_eqe(ax, ma, mb):
    """The result that survives: EQE does not care which run came first."""
    print("\n=== EQE: does the effect survive the order swap?")
    print("  %8s %10s %10s %10s" % ("V", "pair A", "pair B", "spread pp"))
    va, ya = per_bias(ma, "dE")
    vb, yb = per_bias(mb, "dE")
    for v, a, b in zip(va, ya, yb):
        mark = "  <-- band" if BAND[0] <= v <= BAND[1] else ""
        print("  %+8.3f %+10.3f %+10.3f %10.3f%s" % (v, a, b, abs(a - b), mark))
    ax.plot(va, ya, "o-", ms=4, lw=1.4, color="C3", label="pair A: low_to_high first")
    ax.plot(vb, yb, "s--", ms=4, lw=1.4, color="C2", label="pair B: high_to_low first")
    ax.axhline(0, color="k", lw=0.6)
    ax.axvspan(BAND[0], BAND[1], color="0.88", zorder=0)
    ax.annotate("-0.739 % and -0.730 %\nsame bias, opposite sequence",
                xy=(-1.30, ya[list(va).index(-1.30)]), xytext=(-2.45, -1.9),
                fontsize=6.5, color="0.25",
                arrowprops=dict(arrowstyle="->", color="0.5", lw=0.8))
    ax.set_ylim(-3.0, 1.6)
    ax.set_xlabel("NVLED bias [V]")
    ax.set_ylabel("EQE:  low_to_high / high_to_low - 1  [%]")
    ax.set_title("A. the EQE deficit reproduces with the order reversed\n"
                 "both sequences negative across the band, to 0.03 pp per bias",
                 fontsize=9)
    ax.legend(fontsize=7, loc="lower right")
    ax.grid(alpha=0.3)


def panel_current(ax, ma, mb):
    """The result that does not: the current difference was the clock."""
    print("\n=== current: the same test, and it fails it")
    va, ya = per_bias(ma, "dI")
    vb, yb = per_bias(mb, "dI")
    ax.plot(va, ya, "o-", ms=4, lw=1.4, color="C0", label="pair A: low_to_high first")
    ax.plot(vb, yb, "s--", ms=4, lw=1.4, color="C4", label="pair B: high_to_low first")
    ax.plot(va, (ya + yb) / 2, "k-", lw=2, label="mean = the direction term")
    ax.axhline(0, color="k", lw=0.6)
    ax.axvspan(BAND[0], BAND[1], color="0.88", zorder=0)
    b = (va >= BAND[0]) & (va <= BAND[1])
    print("  band mean: pair A %+.3f %%, pair B %+.3f %%, mean of the two %+.3f %%"
          % (ya[b].mean(), yb[b].mean(), ((ya + yb) / 2)[b].mean()))
    print("  the two sequences are mirror images about zero -- that is a clock,")
    print("  not a direction.")
    ax.set_ylim(-1.6, 1.0)
    ax.set_xlabel("NVLED bias [V]")
    ax.set_ylabel("current:  low_to_high / high_to_low - 1  [%]")
    ax.set_title("B. the current difference FLIPS with the order\n"
                 "section 23's -0.40 %% was elapsed time; direction gives %+.3f %%"
                 % ((ya + yb) / 2)[b].mean(), fontsize=9)
    ax.legend(fontsize=7, loc="lower right")
    ax.grid(alpha=0.3)


def panel_split(ax, dec):
    """D and T side by side, per channel, with the EQE identity on the plot."""
    print("\n=== the split, band mean over six biases")
    print("  %8s %9s %9s %12s %12s" % ("channel", "pair A", "pair B",
                                       "D direction", "T run order"))
    x = np.arange(3)
    w = 0.2
    for k, (key, lab) in enumerate((("A", "pair A"), ("B", "pair B"),
                                    ("D", "D: direction"), ("T", "T: run order"))):
        vals = [dec[c][key] for c, _c, _l in CH]
        ax.bar(x + (k - 1.5) * w, vals, w, label=lab,
               color=["0.75", "0.55", "C3", "C5"][k],
               edgecolor="k", linewidth=0.4)
    for c, _col, lab in CH:
        print("  %8s %+9.3f %+9.3f %+12.3f %+12.3f"
              % (lab, dec[c]["A"], dec[c]["B"], dec[c]["D"], dec[c]["T"]))
    ident = dec["dP"]["D"] - dec["dI"]["D"]
    print("  identity check: D_P - D_I = %+.3f %%  vs  D_EQE = %+.3f %%  (EQE = P/I)"
          % (ident, dec["dE"]["D"]))
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(["current", "light", "EQE"])
    ax.set_ylabel("low_to_high / high_to_low - 1  [%]")
    ax.text(0.03, 0.04,
            "EQE = P/I forces  D_P - D_I = D_EQE\n"
            "measured  %+.3f  vs  %+.3f  (%.3f pp)"
            % (ident, dec["dE"]["D"], abs(ident - dec["dE"]["D"])),
            transform=ax.transAxes, fontsize=6.8, color="0.25",
            bbox=dict(fc="w", ec="0.7", lw=0.5, pad=2.5))
    ax.set_title("C. direction is optical only, run order is common mode\n"
                 "D: current %+.3f, light %+.3f, EQE %+.3f  |  "
                 "T: %+.3f and %+.3f cancel to %+.3f"
                 % (dec["dI"]["D"], dec["dP"]["D"], dec["dE"]["D"],
                    dec["dI"]["T"], dec["dP"]["T"], dec["dE"]["T"]), fontsize=9)
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3, axis="y")


def panel_dwell_both(ax):
    """Section 23's dwell claim, in both sequences. This one holds."""
    print("\n=== dwell residual, both sequences (positive = still climbing)")
    out = {}
    for nm, (lo_tag, hi_tag, _s) in (("A", PAIR_A), ("B", PAIR_B)):
        lo, hi = dwell_slopes(lo_tag), dwell_slopes(hi_tag)
        k = sorted(set(lo) & set(hi), reverse=True)
        sl = [lo[v] for v in k]
        sh = [hi[v] for v in k]
        out[nm] = (k, sl, sh)
        print("  pair %s  low_to_high median %+.5f %%/s (%+.3f %%/dwell), "
              "%d of %d climbing" % (nm, np.median(sl), DWELL_S * np.median(sl),
                                     sum(s > 0 for s in sl), len(sl)))
        print("          high_to_low median %+.5f %%/s (%+.3f %%/dwell), "
              "%d of %d climbing" % (np.median(sh), DWELL_S * np.median(sh),
                                     sum(s > 0 for s in sh), len(sh)))
        print("          high_to_low is %.1fx further from settled"
              % (np.median(sh) / np.median(sl)))
    for nm, (ls, col) in (("A", ("-", "C0")), ("B", ("--", "C2"))):
        k, sl, sh = out[nm]
        ax.plot(k, [DWELL_S * v for v in sl], "o" + ls, ms=4, lw=1.3, color=col,
                label="low_to_high, pair %s" % nm)
        ax.plot(k, [DWELL_S * v for v in sh], "s" + ls, ms=4, lw=1.3,
                color="C3" if nm == "A" else "C1",
                label="high_to_low, pair %s" % nm)
    ax.axhline(0, color="k", lw=0.8)
    ax.set_yscale("symlog", linthresh=0.1, linscale=0.8)
    ax.set_xlabel("NVLED bias [V]")
    ax.set_ylabel("residual drift over one 54.47 s dwell [%]")
    ax.set_title("D. high_to_low is further from settled in BOTH sequences\n"
                 "19 of 20 bias-runs still climbing; the -0.80 V outlier that\n"
                 "section 23 quoted is the one thing that does not reproduce",
                 fontsize=9)
    ax.legend(fontsize=6.5, loc="upper right", ncol=2)
    ax.grid(alpha=0.3, which="both")


def demo():
    """The split, and its one free prediction, on inputs with a known answer."""
    # a pure direction effect: both sequences read the same, T must vanish
    d = decompose({"dI": -0.5, "dP": -0.5, "dE": -0.5},
                  {"dI": -0.5, "dP": -0.5, "dE": -0.5})
    assert abs(d["dI"]["D"] + 0.5) < 1e-12, d["dI"]
    assert abs(d["dI"]["T"]) < 1e-12, "no order term when the pairs agree"

    # a pure order effect: the sequences mirror, D must vanish
    d = decompose({"dI": +0.4, "dP": +0.4, "dE": +0.4},
                  {"dI": -0.4, "dP": -0.4, "dE": -0.4})
    assert abs(d["dI"]["D"]) < 1e-12, "no direction term when the pairs mirror"
    assert abs(d["dI"]["T"] - 0.4) < 1e-12, d["dI"]

    # and the real data must satisfy EQE = P/I to better than 0.05 pp, or the
    # decomposition is measuring arithmetic noise rather than two components
    a, _ = band_means(merged(*PAIR_A)[2])
    b, _ = band_means(merged(*PAIR_B)[2])
    dec = decompose(a, b)
    gap = abs((dec["dP"]["D"] - dec["dI"]["D"]) - dec["dE"]["D"])
    assert gap < 0.05, "D_P - D_I = %+.3f but D_EQE = %+.3f" % (
        dec["dP"]["D"] - dec["dI"]["D"], dec["dE"]["D"])

    # the headline: EQE keeps its sign, the current does not
    assert a["dE"] < 0 and b["dE"] < 0, "EQE deficit must survive the order swap"
    assert a["dI"] * b["dI"] < 0, "the current difference must flip sign"
    print("demo: ok (D_P - D_I matches D_EQE to %.3f pp)" % gap)
    return dec


if __name__ == "__main__":
    dec = demo()
    OUT.mkdir(exist_ok=True)
    _la, _ha, ma = merged(*PAIR_A)
    _lb, _hb, mb = merged(*PAIR_B)
    fig, axs = plt.subplots(2, 2, figsize=(13, 10))
    panel_eqe(axs[0, 0], ma, mb)
    panel_current(axs[0, 1], ma, mb)
    panel_split(axs[1, 0], dec)
    panel_dwell_both(axs[1, 1])
    fig.suptitle("The reverse-sequence control: direction is optical, run order is "
                 "common mode (section 24)", fontsize=12)
    fig.tight_layout()
    p = OUT / "fig38_direction_control.png"
    fig.savefig(p, dpi=130)
    print("\nwrote %s" % p)
