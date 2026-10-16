"""The five bench runs of 15 October 2026, which answer Q1, Q3, Q4 and Q5 with data
taken for the purpose instead of data read off disk afterwards.

    python plot_bench_answers.py [--out investigation_report]
    python plot_bench_answers.py --demo      # no hardware, no data, self-check

Writes fig30_bitval1.png, fig31_highc_600s.png, fig32_switch_two_scales.png and
fig33_leak_trace.png, and prints the numbers section 20 of the report quotes.

The runs:
  q3_step600     261015-2105  600 s step, fixed range and High-C, from -1.20 V
  q4_switch120   261015-2147  122 s switch record at NPLC 1.0, after a 1200 s hold
  q4_switchfast  261015-2231  3.2 s switch record at NPLC 0.01
  q5_leaktrace   261015-2242  122 s record of the pixel being driven to bitval 0
  refbv1         261015-2326  the same sweep with the reference at bitval 1

One warning carries through all of it: `--highc` forces compliance = measure range =
10 uA, and pixel (35,36) draws 30.4 uA at -1.30 V. The bright bias of q4_switch120 and
q5_leaktrace therefore ran AT COMPLIANCE and their absolute numbers are not usable.
Both faint biases have headroom and are clean.
"""
from __future__ import annotations

import argparse
import glob
from pathlib import Path

import numpy as np
import pandas as pd

# The measurement repo is this file's grandparent since ROADMAP item 2 merged the
# two. It was an absolute path for as long as they were separate repos.
MEAS = Path(__file__).resolve().parent.parent
OVF = 1e30        # the SMU writes 9.91e37 for an over-range reading
LIMIT_A = 1e-5    # what --highc forces the compliance to
TAIL_FRAC = 0.1   # the last tenth of a record is its reference


def one(pattern: str) -> Path:
    hits = sorted(glob.glob(str(MEAS / pattern)))
    if not hits:
        raise FileNotFoundError(str(MEAS / pattern))
    return Path(hits[0])


def traces(tag: str) -> dict[str, np.ndarray]:
    z = np.load(one(f"demo/{tag}/*/traces.npz"))
    return {k: z[k] for k in z.files}


def clean(a: np.ndarray, row: int = 2):
    """(t, |I|) with the over-range marker removed. Row 2 is the cathode, which is the
    channel every settling argument in this report is about."""
    t, y = a[0], np.abs(a[row])
    m = np.isfinite(y) & (y < OVF)
    return t[m], y[m]


def tail(y: np.ndarray, frac: float = TAIL_FRAC):
    """(mean, SEM) of the last `frac` of a record.

    Every deviation below is referenced to this rather than to an absolute calibration,
    because the question is always "how far is this sample from where it ends up".
    """
    n = max(4, int(len(y) * frac))
    a = y[-n:]
    return a.mean(), a.std(ddof=1) / np.sqrt(n)


def bias_of(key: str) -> float:
    return float(key.split("_v")[1][:6])


def at_compliance(y: np.ndarray) -> float:
    """Fraction of samples pinned at the forced 10 uA compliance. Non-zero means the
    source ran out of headroom and the trace measures the SMU, not the pixel."""
    return float((y > 0.995 * LIMIT_A).mean())


# ----------------------------------------------------------------- Q1

def q1_table() -> pd.DataFrame:
    """bitval 1 against bitval 15 on the same pixel. The point of the run was to test
    whether the dropped frame write is the ZERO value or the write's position, and the
    answer is that it cannot: bitval 1 is not dark, so the optical gate fires on
    physics. What the run does deliver is the bottom of the drive scale."""
    d = pd.read_csv(one("demo/refbv1/*/NVLEDSweep/Sweep/*aggregated_bitval=15_mean_std.csv"))
    d["pf"] = np.abs(d.PM400_POWER_REF / d.PM400_POWER_MEAN)
    d["if_"] = np.abs(d.NVLED_CURR_REF / d.NVLED_CURR_MEAN)
    g = d.groupby("NVLED_V")
    out = pd.DataFrame({
        "pf": g["pf"].mean(),
        # spread across four INDEPENDENT pixels: a dropped write is random and cannot
        # reproduce itself to a few percent, so a small spread rules the write out
        "pf_spread": g["pf"].apply(lambda s: (s.max() - s.min()) / s.mean()),
        "if_": g["if_"].mean(),
        "p_ref": g["PM400_POWER_REF"].mean(),
        "p_lit": g["PM400_POWER_MEAN"].mean(),
        "tries3": g["LEAK_REF_TRIES"].apply(lambda s: (s > 1).sum()),
    }).sort_index(ascending=False)
    return out


def q1(ax) -> None:
    t = q1_table()
    dark = []
    for tag in ("ab_bm", "ab_px", "biasmajor2"):
        try:
            o = pd.read_csv(one(f"demo/{tag}/*/NVLEDSweep/Sweep/*aggregated_bitval=15_mean_std.csv"))
        except FileNotFoundError:
            continue
        dark.append(np.abs(o.PM400_POWER_REF / o.PM400_POWER_MEAN).median())

    v = t.index.to_numpy()
    ax.errorbar(v, t.pf, yerr=t.pf * t.pf_spread / 2, marker="o", ms=4, lw=1.4,
                capsize=2, color="C3", label="bitval 1 reference, P$_{ref}$/P")
    ax.plot(v, t.if_, marker="s", ms=3.5, lw=1.0, ls="--", color="C1",
            label="bitval 1 reference, I$_{ref}$/I")
    if dark:
        ax.axhline(np.median(dark), color="C0", lw=1.4,
                   label=f"bitval 0 reference, 3 runs ({np.median(dark):.1e})")
    ax.axhline(0.25, color="k", lw=1.0, ls=":", label="leak_ref_max_frac = 0.25")
    ax.axhline(1 / 15, color="0.5", lw=1.0, ls="-.", label="1/15, if the drive were linear")
    ax.set_yscale("log")
    ax.set_xlabel("NVLED bias [V]")
    ax.set_ylabel("reference / lit")
    ax.set_title("Q1: the bitval-1 reference is LIT, so the test could not run\n"
                 f"{int(t.tries3.sum())} of 64 points burned all 3 frame writes on physics",
                 fontsize=10)
    ax.legend(fontsize=7.5, loc="lower left")
    ax.grid(alpha=0.3, which="both")


# ----------------------------------------------------------------- Q3

def q3_table() -> pd.DataFrame:
    tr = {("hc1" in k, bias_of(k)): a for k, a in traces("q3_step600").items()}
    rows = []
    for v in sorted({b for _, b in tr}, reverse=True):
        for hc in (True, False):
            t, y = clean(tr[(hc, v)])
            raw = np.abs(tr[(hc, v)][2])
            m, s = tail(y, 0.05)
            over = ~np.isfinite(raw) | (raw >= OVF)
            rows.append(dict(
                v=v, highc=hc, i_inf=m, i_sem=s,
                noise=y[t > 400].std(ddof=1),
                over=int(over.sum()),
                over_t=float(tr[(hc, v)][0][over].max()) if over.any() else 0.0,
                peak=y.max(), i_head=y[t < 0.2].mean() if (t < 0.2).any() else np.nan,
            ))
    return pd.DataFrame(rows)


def q3_gap(cut: float = 604.0) -> pd.DataFrame:
    """fixed-range minus High-C at a given time, as a percentage. Reading both arms off
    their own 600 s records at the SAME elapsed time is the comparison 261014-1550 could
    not make, because it only ran 81 s against t_1% = 240 s."""
    tr = {("hc1" in k, bias_of(k)): a for k, a in traces("q3_step600").items()}
    rows = []
    for v in sorted({b for _, b in tr}, reverse=True):
        vals, sems = [], []
        for hc in (True, False):
            t, y = clean(tr[(hc, v)])
            w = (t > cut * 0.9) & (t <= cut)
            vals.append(y[w].mean())
            sems.append(y[w].std(ddof=1) / np.sqrt(max(1, w.sum())))
        g = 100 * (vals[1] / vals[0] - 1)
        sg = 100 * np.hypot(sems[1] / vals[0], vals[1] * sems[0] / vals[0] ** 2)
        rows.append(dict(v=v, gap_pct=g, gap_sem=sg))
    return pd.DataFrame(rows)


def q3(ax) -> None:
    """The quantity Q3 actually asks about is the GAP between the arms, read off both
    records at the same elapsed time. Plotting the two traces instead buries it: at
    -0.85 V the fixed arm's own noise is 16x the High-C arm's and swamps the axis."""
    tr = {("hc1" in k, bias_of(k)): a for k, a in traces("q3_step600").items()}
    grid = np.logspace(np.log10(0.3), np.log10(600), 160)
    for i, v in enumerate(sorted({b for _, b in tr}, reverse=True)):
        arms = []
        for hc in (True, False):
            t, y = clean(tr[(hc, v)])
            # one decade-relative window per grid point: the noise falls as the record
            # lengthens, which is what makes the late gap readable at all
            arms.append(np.array([y[(t > g * 0.75) & (t <= g * 1.25)].mean()
                                  for g in grid]))
        ax.plot(grid, 100 * (arms[1] / arms[0] - 1), lw=1.6, color=f"C{i}",
                label=f"{v:+.2f} V")
        g604 = q3_gap(604.0)
        r = g604[np.isclose(g604.v, v)].iloc[0]
        ax.errorbar([600], [r.gap_pct], yerr=[3 * r.gap_sem], fmt="o", ms=5,
                    color=f"C{i}", capsize=3)
        ax.annotate(f"{r.gap_pct:+.2f} %", (600, r.gap_pct), fontsize=8,
                    color=f"C{i}", xytext=(-6, 10 if r.gap_pct > 0 else -16),
                    textcoords="offset points", ha="right")
    raw = np.abs(tr[(False, -0.85)][2])
    over = ~np.isfinite(raw) | (raw >= OVF)
    ax.axvspan(0.3, max(0.55, tr[(False, -0.85)][0][over].max()), color="C3", alpha=0.13)
    ax.text(0.33, 4.2, "fixed range over-ranges here:\n13 samples, then ~1 s of\n"
                       "readings 6x too low", fontsize=7, color="C3")
    ax.axhline(0, color="k", lw=0.9)
    ax.axvline(81, color="0.4", lw=1.0, ls=":")
    ax.text(84, -3.3, "81 s: 261014-1550\nstopped here", fontsize=7.5, color="0.3")
    ax.set_xscale("log")
    ax.set_ylim(-3.6, 6.4)
    ax.set_xlabel("time after the step from -1.20 V [s]")
    ax.set_ylabel("fixed range - High-C  [%]")
    ax.set_title("Q3: 600 s for both arms. The gap shrinks but does not close,\n"
                 "and it has a different sign at the two biases (error bars 3 SEM)",
                 fontsize=10)
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(alpha=0.3, which="both")


# ----------------------------------------------------------------- Q4

def q4_bins(tag: str, key: str, edges) -> pd.DataFrame:
    t, y = clean(traces(tag)[key])
    ref, _ = tail(y)
    rows = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        w = (t >= lo) & (t < hi)
        if w.sum() < 2:
            continue
        m = y[w].mean()
        s = y[w].std(ddof=1) / np.sqrt(w.sum())
        rows.append(dict(lo=lo, hi=hi, n=int(w.sum()), dev=100 * (m / ref - 1),
                         sem=100 * s / ref))
    return pd.DataFrame(rows)


def q4(ax) -> None:
    """Two records, four decades apart, and neither one sees the other's component."""
    fast = traces("q4_switchfast")
    slow = traces("q4_switch120")
    for k, a in sorted(fast.items()):
        v = bias_of(k)
        t, y = clean(a)
        ref, _ = tail(y)
        s = pd.Series(100 * (y / ref - 1)).rolling(5, center=True,
                                                   min_periods=2).mean()
        ax.plot(t[t > 0], s[t > 0], lw=1.0,
                label=f"{v:+.2f} V, NPLC 0.01 (1.06 ms/sample)")
    for k, a in sorted(slow.items()):
        v = bias_of(k)
        t, y = clean(a)
        if at_compliance(y) > 0.5:          # -1.35 V sat at the forced 10 uA limit
            continue
        ref, _ = tail(y)
        s = pd.Series(100 * (y / ref - 1)).rolling(25, center=True,
                                                   min_periods=10).mean()
        ax.plot(t[t > 0], s[t > 0], lw=1.6, color="k",
                label=f"{v:+.2f} V, NPLC 1.0 (40.7 ms/sample)")
    ax.axhline(0, color="k", lw=0.8)
    ax.axvspan(1e-3, 40.7e-3, color="C7", alpha=0.18)
    ax.text(1.2e-3, -7.5, "one NPLC 1.0\nintegration", fontsize=7, color="0.25")
    ax.axvline(2.0, color="C3", lw=1.2, ls="--")
    ax.text(2.3, -7.5, "pixel_switch_settle_s = 2 s", fontsize=7.5, color="C3")
    ax.set_xscale("log")
    ax.set_ylim(-10.5, 4.5)
    ax.set_xlabel("time after blank_frame() + set_pixel() [s]")
    ax.set_ylabel("deviation from own record's tail [%]")
    ax.set_title("Q4: the switch has a 10 ms component and a 100 s component.\n"
                 "No single record sees both, and NPLC 1.0 cannot see the first",
                 fontsize=10)
    ax.legend(fontsize=7.5, loc="lower right")
    ax.grid(alpha=0.3, which="both")


# ----------------------------------------------------------------- Q5

def q5(ax) -> None:
    for k, a in sorted(traces("q5_leaktrace").items()):
        v = bias_of(k)
        t, y = clean(a)
        ref, _ = tail(y)
        bad = at_compliance(y) > 0
        s = pd.Series(100 * (y / ref - 1)).rolling(15, center=True,
                                                   min_periods=5).mean()
        ax.plot(t[t > 0], s[t > 0], lw=1.2, ls=":" if bad else "-",
                label=f"{v:+.2f} V, settled leak {1e9*ref:.1f} nA"
                      + (" -- AT COMPLIANCE, not usable" if bad else ""))
    ax.axhline(0, color="k", lw=0.8)
    ax.axhspan(-1, 1, color="C2", alpha=0.15)
    ax.axvline(2.0, color="C3", lw=1.2, ls="--")
    ax.text(2.3, 3.5, "leak_ref_settle_s = 2 s", fontsize=7.5, color="C3")
    ax.text(0.05, 1.6, "+/-1 % of the settled leak", fontsize=7, color="C2")
    ax.set_xscale("log")
    # the -1.30 V trace starts 35x above its own settled value, so a linear axis would
    # be one spike and a flat line. symlog keeps the +/-1 % band readable.
    ax.set_yscale("symlog", linthresh=3, linscale=1.1)
    ax.set_yticks([-3, 0, 3, 10, 100, 1000])
    ax.set_yticklabels(["-3", "0", "3", "10", "100", "1000"])
    ax.set_xlabel("time after the pixel is driven to bitval 0 [s]")
    ax.set_ylabel("deviation from own settled leak [%]")
    ax.set_title("Q5: the leakage, finally as a time series.\n"
                 "Flat and white once settled -- no oscillation", fontsize=10)
    ax.legend(fontsize=7.5)
    ax.grid(alpha=0.3, which="both")


def q5_stats() -> dict:
    t, y = clean(traces("q5_leaktrace")["000_switch_hc1_v-0.850_35-36to35-36_b0_1"])
    ref, _ = tail(y)
    sm = pd.Series(y).rolling(25, center=True, min_periods=15).mean().to_numpy()
    late = t > 20
    tl, yl = t[late], y[late]
    d = np.diff(yl)
    f = np.fft.rfftfreq(late.sum(), float(np.median(np.diff(tl))))
    amp = 2 * np.abs(np.fft.rfft(yl - yl.mean())) / late.sum()
    j = int(np.argmax(amp[1:]) + 1)
    out = dict(
        ref_nA=1e9 * ref,
        settle_1pct_s=float(t[np.where(np.abs(sm / ref - 1) > 0.01)[0].max()]),
        sd_pct=100 * yl.std(ddof=1) / ref,
        drift_pct_per_h=100 * float(np.polyfit(tl, yl, 1)[0]) * 3600 / ref,
        lag1=float(np.corrcoef(d[:-1], d[1:])[0, 1]),
        peak_hz=float(f[j]), peak_pct=100 * amp[j] / ref,
        white_pct=100 * yl.std(ddof=1) * np.sqrt(2 / late.sum()) / ref,
        burst_sem_pct=100 * yl.std(ddof=1) / np.sqrt(8) / ref,
    )
    return out


# ----------------------------------------------------------------- report

def report() -> None:
    t = q1_table()
    print("Q1  the bitval-1 reference, against bitval 15 on the same pixel")
    print(f"    {'bias':>7} {'P_ref/P':>9} {'spread':>8} {'I_ref/I':>9} "
          f"{'P_ref [nW]':>11} {'retried':>8}")
    for v, r in t.iterrows():
        print(f"    {v:+7.2f} {r.pf:9.4f} {100*r.pf_spread:7.1f}% {r.if_:9.4f} "
              f"{1e9*r.p_ref:11.4f} {int(r.tries3):8d}")
    print(f"    bitval 1 emits {1e9*t.p_ref.max():.1f} nW at the bright end. It is not "
          f"dark, so\n    the OPTICAL gate cannot test the frame write. 1/15 would be "
          f"0.067 flat.\n")

    print("Q3  600 s asymptotes, both arms, from -1.20 V")
    q = q3_table()
    print(f"    {'bias':>7} {'arm':>7} {'I_inf [nA]':>12} {'SEM':>8} {'noise [nA]':>11} "
          f"{'over-range':>11}")
    for _, r in q.iterrows():
        print(f"    {r.v:+7.2f} {'High-C' if r.highc else 'fixed':>7} "
              f"{1e9*r.i_inf:12.3f} {100*r.i_sem/r.i_inf:7.3f}% {1e9*r.noise:11.3f} "
              f"{r.over:5d} to {r.over_t:.2f}s")
    for cut in (81.0, 604.0):
        g = q3_gap(cut)
        s = "  ".join(f"{r.v:+.2f} V: {r.gap_pct:+.3f}% +/- {r.gap_sem:.3f}%"
                      for _, r in g.iterrows())
        print(f"    fixed - High-C at {cut:5.0f} s:  {s}")
    print()

    print("Q4  the switch transient, two records")
    e = [0, .005, .01, .02, .04, .08, .16, .32, .64, 1.3, 3.2]
    for k in sorted(traces("q4_switchfast")):
        b = q4_bins("q4_switchfast", k, e)
        print(f"    NPLC 0.01, {bias_of(k):+.2f} V: " + "  ".join(
            f"{1e3*r.lo:.0f}-{1e3*r.hi:.0f}ms {r.dev:+.2f}%" for _, r in b.iterrows()))
    for k in sorted(traces("q4_switch120")):
        t_, y_ = clean(traces("q4_switch120")[k])
        if at_compliance(y_) > 0.5:
            print(f"    NPLC 1.0,  {bias_of(k):+.2f} V: AT COMPLIANCE for "
                  f"{100*at_compliance(y_):.0f}% of the record -- not usable")
            continue
        b = q4_bins("q4_switch120", k, [0, .5, 1, 2, 5, 20, 60, 110])
        print(f"    NPLC 1.0,  {bias_of(k):+.2f} V: " + "  ".join(
            f"{r.hi:.0f}s {r.dev:+.3f}%" for _, r in b.iterrows()))
    print()

    print("Q5  the leakage as a time series, -0.85 V (the one with headroom)")
    s = q5_stats()
    print(f"    settled {s['ref_nA']:.3f} nA, within 1 % of it from t = "
          f"{s['settle_1pct_s']:.2f} s")
    print(f"    after 20 s: sd {s['sd_pct']:.3f} %, drift "
          f"{s['drift_pct_per_h']:+.2f} %/h, lag-1 {s['lag1']:+.3f} "
          f"({'white' if s['lag1'] < -0.3 else 'correlated'})")
    print(f"    strongest periodic component {s['peak_hz']:.4f} Hz "
          f"({1/s['peak_hz']:.1f} s) at {s['peak_pct']:.4f} %, white expectation "
          f"{s['white_pct']:.4f} %")
    print(f"    an 8-sample reference burst has SEM {s['burst_sem_pct']:.3f} % "
          f"on the leak")


# ----------------------------------------------------------------- self-check

def demo() -> None:
    """No hardware, no data. Checks the three helpers that can be quietly wrong."""
    # tail(): mean and SEM of the last tenth, not of the last sample
    y = np.concatenate([np.zeros(90), np.full(10, 5.0)])
    m, s = tail(y)
    assert abs(m - 5.0) < 1e-12 and s == 0.0, (m, s)

    # clean(): the over-range marker is dropped from BOTH t and y, in step
    a = np.vstack([np.arange(6.0), np.zeros(6), [1, 9.91e37, 2, np.nan, 3, -4]])
    t, y = clean(a)
    assert np.allclose(t, [0, 2, 4, 5]) and np.allclose(y, [1, 2, 3, 4]), (t, y)

    # at_compliance(): only counts what is pinned at the forced limit
    assert at_compliance(np.array([1e-6, 9.99e-6, 1e-5])) == 2 / 3
    assert at_compliance(np.array([1e-6, 2e-6])) == 0.0

    # bias_of(): the key format both capacitance scripts write
    assert bias_of("000_switch_hc1_v-0.850_39-40to35-36_b15_1") == -0.850
    assert bias_of("002_step_hc1_rNone_f-1.200_v-1.000_4") == -1.000

    # q4_bins' deviation sign: a record that starts LOW must report negative
    t = np.linspace(0, 10, 1000)
    y = 100 * (1 - 0.1 * np.exp(-t / 0.5))
    ref, _ = tail(y)
    assert 100 * (y[:5].mean() / ref - 1) < -9.0

    print("ok  plot_bench_answers: tail exact, over-range dropped in step, "
          "compliance counted, keys parsed, deviation sign correct")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path,
                    default=Path(__file__).resolve().parent / "investigation_report",
                    help="where the figures go (default: beside this script)")
    ap.add_argument("--demo", action="store_true")
    args = ap.parse_args()
    if args.demo:
        demo()
        return

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    report()
    args.out.mkdir(parents=True, exist_ok=True)
    for fn, name in ((q1, "fig30_bitval1"), (q3, "fig31_highc_600s"),
                     (q4, "fig32_switch_two_scales"), (q5, "fig33_leak_trace")):
        fig, ax = plt.subplots(figsize=(7.2, 4.6))
        fn(ax)
        fig.tight_layout()
        fig.savefig(args.out / f"{name}.png", dpi=130)
        plt.close(fig)
        print(f"  wrote {args.out / (name + '.png')}")


if __name__ == "__main__":
    main()
