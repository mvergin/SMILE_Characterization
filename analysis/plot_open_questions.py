"""The four questions a colleague asked, answered from data already on disk.

    python plot_open_questions.py [--out investigation_report]
    python plot_open_questions.py --demo        # no hardware, no data, self-check

Writes fig26_bitval0.png, fig27_dwell_optical.png, fig28_highc_converge.png and
fig29_pixel_switch.png, and prints the numbers section 19 of the report quotes.

Q1  do some pixels fail to go dark at bitval 0, and is it a timing race?
Q2  is the optical power recorded during the dwell, and does it relax like the current?
Q3  does non-High-C converge to the same value as High-C?
Q4  what does the current do during a pixel switch?
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
OVF = 1e30          # the SMU writes 9.91e37 for an over-range reading
LEAK_GATE = 0.25    # P_REF/P above this is "did not go dark"


def one(pattern: str) -> Path:
    hits = sorted(glob.glob(str(MEAS / pattern)))
    if not hits:
        raise FileNotFoundError(str(MEAS / pattern))
    return Path(hits[0])


def finite(t, y):
    """Drop the SMU's over-range marker. 9.91e37 is not a current, and leaving it in
    turns a mean into nonsense and a log plot into an empty axis."""
    m = np.isfinite(y) & (np.abs(y) < OVF)
    return t[m], y[m]


def decile(y):
    """(first-tenth mean, its SEM, last-tenth mean, its SEM).

    A single first sample is what the obvious version uses, and on the optical channel
    its noise is 0.3 % -- the same size as the drift being measured. The decile mean
    brings that to 0.03 % and makes the question answerable at all.
    """
    n = max(4, len(y) // 10)
    a, b = y[:n], y[-n:]
    return a.mean(), a.std(ddof=1) / np.sqrt(n), b.mean(), b.std(ddof=1) / np.sqrt(n)


def rsd(y):
    y = np.asarray(y, float)
    return (np.percentile(y, 75) - np.percentile(y, 25)) / 1.349 if len(y) >= 4 else np.nan


def faintness_corr(ratio, volts):
    """Rank correlation of a ratio with how faint the bias is.

    This is the Q1 discriminator. A frame write that never arrived leaves the pixel at
    its previous drive level, so P_REF/P = 1 at EVERY bias and this is 0. A device that
    leaks a bias-dependent fraction gives a strong positive value.
    """
    rk = pd.Series(np.asarray(ratio, float)).rank().to_numpy()
    vk = pd.Series(-np.abs(np.asarray(volts, float))).rank().to_numpy()
    if np.ptp(rk) == 0 or np.ptp(vk) == 0:
        return 0.0
    return float(np.corrcoef(rk, vk)[0, 1])


# ----------------------------------------------------------------------------- Q1
def q1(ax=None):
    d = pd.read_csv(one("demo/full/*/NVLEDSweep/Sweep/*aggregated_bitval=*_mean_std.csv"))
    d["frac"] = d["PM400_POWER_REF"] / d["PM400_POWER_MEAN"]
    d["ifrac"] = d["NVLED_CURR_REF"] / d["NVLED_CURR_MEAN"]
    med = d.groupby(["X", "Y"])["frac"].median()
    bad = med[med > LEAK_GATE].sort_values(ascending=False)
    clean = med[med <= LEAK_GATE]
    print(f"Q1  {len(bad)} of {len(med)} swept pixels carry >{LEAK_GATE:.0%} of their "
          f"own light at bitval 0")
    print(f"    the other {len(clean)}: median P_REF/P = {clean.median():.5f}, "
          f"worst {clean.max():.5f}")
    print("    contaminated: "
          + ", ".join(f"({x},{y}) {v:.2f}" for (x, y), v in bad.items()))
    isbad = d.set_index(["X", "Y"]).index.isin(bad.index)
    print(f"    I_REF/I  contaminated {d[isbad]['ifrac'].median():.3f}  "
          f"clean {d[~isbad]['ifrac'].median():.3f}")
    sub = d[isbad & (d.PM400_POWER_REF > 0)]
    r = (sub["PM400_POWER_REF"] / sub["PM400_POWER_MEAN"]).to_numpy()
    print(f"    a stale frame predicts P_REF/P = 1.00 flat. Measured {r.min():.2f}-"
          f"{r.max():.2f}, rank corr with faintness "
          f"r = {faintness_corr(r, sub['NVLED_V']):+.3f}")
    if ax is None:
        return
    for (x, y) in bad.index:
        s = d[(d.X == x) & (d.Y == y)].sort_values("NVLED_V")
        ax.plot(-s.NVLED_V, s.frac, "o-", ms=3, label=f"({x},{y})")
    s = d[~isbad].groupby("NVLED_V")["frac"].median()
    ax.plot(-s.index, s.values, "k.-", ms=3, lw=1, label=f"the other {len(clean)} pixels")
    ax.axhline(1.0, color="r", ls="--", lw=1)
    ax.text(1.5, 1.15, "a stale frame write would sit here, flat", color="r", fontsize=7)
    ax.set_yscale("log")
    ax.set_xlabel("|cathode bias| (V)      <-- fainter          brighter -->")
    ax.set_ylabel("$P_{REF}/P$  at bitval 0")
    ax.set_title("Q1  bitval 0 is dark to 2e-4, except on 3 pixels in 100 --\n"
                 "and those 3 leak a bias-dependent fraction, not a stale frame",
                 fontsize=9)
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3)


# ----------------------------------------------------------------------------- Q2
def q2(ax=None):
    runs = {"261014-1616 pixel_major": "demo/new", "261015-1421 ab_px": "demo/ab_px",
            "261015-1529 ab_bm": "demo/ab_bm", "261014-1912 repeat": "demo/repeat/new_1",
            "261015-0015 full": "demo/full"}
    print("\nQ2  optical power IS recorded in every dwell (settling_traces.npz row 2)")
    print(f"    {'run':<26}{'n':>4}{'dI':>9}{'dP':>9}{'dEQE':>9}")
    pooled, shown = [], None
    for label, d in runs.items():
        try:
            z = np.load(one(d + "/*/NVLEDSweep/Sweep/settling_traces.npz"),
                        allow_pickle=True)
        except FileNotFoundError:
            continue
        acc = []
        for k in sorted(z.keys()):
            a = z[k]
            if a.shape[0] < 3 or a.shape[1] < 40:
                continue
            i0, i0e, i1, i1e = decile(a[1])
            p0, p0e, p1, p1e = decile(a[2])
            si = 100 * np.hypot(i0e / i0, i1e / i1)
            sp = 100 * np.hypot(p0e / p0, p1e / p1)
            if si > 1.0 or sp > 1.0:        # the trace cannot resolve 1 %: skip it
                continue
            acc.append((100 * (i1 / i0 - 1), 100 * (p1 / p0 - 1),
                        float(k.rsplit("_v", 1)[1]), abs(i1)))
            if shown is None and "v-1.0000" in k:
                shown = (label, k, a)
        if not acc:
            continue
        a = np.array(acc)
        pooled.append(a)
        print(f"    {label:<26}{len(a):4d}{np.median(a[:, 0]):+8.2f}%"
              f"{np.median(a[:, 1]):+8.2f}%{np.median(a[:, 1] - a[:, 0]):+8.2f}%")
    A = np.vstack(pooled)
    de = A[:, 1] - A[:, 0]
    print(f"    pooled {len(A)} traces: current {np.median(A[:, 0]):+.2f} %, "
          f"power {np.median(A[:, 1]):+.2f} %, EQE {np.median(de):+.2f} %")
    print(f"    current rises AND power falls in "
          f"{int(((A[:, 0] > 0) & (A[:, 1] < 0)).sum())}/{len(A)} traces")
    pw = A[:, 3] * np.abs(A[:, 2])
    o = np.argsort(pw)
    k = max(1, len(o) // 4)
    print("    self-heating predicts the droop GROWS with dissipated power. It does the "
          "opposite:")
    print(f"      lowest  I*V quartile ({1e6 * pw[o[:k]].min():.1f}-"
          f"{1e6 * pw[o[:k]].max():.1f} uW)  dEQE {np.median(de[o[:k]]):+.2f} %")
    print(f"      highest I*V quartile ({1e6 * pw[o[-k:]].min():.1f}-"
          f"{1e6 * pw[o[-k:]].max():.1f} uW)  dEQE {np.median(de[o[-k:]]):+.2f} %")
    if ax is None or shown is None:
        return
    label, key, a = shown
    t, i, p = a
    ax.plot(t, 100 * (i / np.mean(i[-len(i) // 10:]) - 1), lw=1,
            label="|cathode current|")
    ax.plot(t, 100 * (p / np.mean(p[-len(p) // 10:]) - 1), lw=1, label="optical power")
    ax.axhline(0, color="k", lw=0.5)
    ax.set_xlabel("time into the 54.4 s dwell (s)")
    ax.set_ylabel("deviation from the dwell's last tenth (%)")
    ax.set_title("Q2  the current relaxes upward, the light does not follow it\n"
                 f"{label}, {key}", fontsize=9)
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3)


# ----------------------------------------------------------------------------- Q3
def q3(ax=None):
    z = np.load(one("demo/transients/261014-1550_CAPACITANCE*/traces.npz"),
                allow_pickle=True)
    by: dict[str, dict] = {}
    for k in z.keys():
        t, _, ib = z[k]
        v = k.rsplit("_v", 1)[1].rsplit("_", 1)[0]
        rng = k.split("_r", 1)[1].split("_f")[0]
        hc = "hc1" if "_hc1_" in k else "hc0"
        tt, ii = finite(t, ib)
        by.setdefault(v, {})[(hc, rng)] = (tt, ii, len(ib) - len(ii))
    print("\nQ3  non-High-C vs High-C, 261014-1550, 81 s records")
    print(f"    {'bias':>8}  {'mode':<20}{'final |I|':>12}{'vs High-C':>11}"
          f"{'tail noise':>12}{'over-range':>12}")
    for v in sorted(by, key=float, reverse=True):
        ref = np.median(by[v][("hc1", "None")][1][-50:])
        for (hc, rng), (t, i, nbad) in sorted(by[v].items(), key=lambda kv: kv[0][0]):
            lbl = "High-C" if hc == "hc1" else f"fixed range {rng}"
            if len(i) == 0:
                print(f"    {v:>8}  {lbl:<20}{'--':>12}{'--':>11}{'--':>12}"
                      f"{nbad:12d}  ALL OVER-RANGE")
                continue
            fin = np.median(i[-50:])
            print(f"    {v:>8}  {lbl:<20}{abs(fin):12.4e}{100 * (fin / ref - 1):+10.2f}%"
                  f"{100 * abs(rsd(i[-50:]) / fin):11.2f}%{nbad:12d}")
    print("    !! the record is 81 s and t_1% at the faint end is 240 s (section 3), so "
          "the\n       -0.80 V row compares two traces that are both still relaxing.")
    if ax is None:
        return
    v = "-0.850"
    for (hc, rng), (t, i, _) in sorted(by[v].items(), key=lambda kv: kv[0][0]):
        if len(i) == 0:
            continue
        lbl = "High-C" if hc == "hc1" else f"range {rng}"
        ax.plot(t, 1e9 * np.abs(i), lw=1.4 if hc == "hc1" else 0.7, label=lbl)
    ax.set_xlabel("time after the voltage step (s)")
    ax.set_ylabel("|cathode current| (nA)")
    ax.set_title(f"Q3  non-High-C lands on the same value at {v} V,\n"
                 "to within its own noise -- which is 20x larger", fontsize=9)
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3)


# ----------------------------------------------------------------------------- Q4
def q4(ax=None):
    print("\nQ4  the current during a pixel switch (frame change at a held bias)")
    print(f"    {'run':<12}{'bias':>7}{'I_inf':>12}"
          + "".join(f"{f'@{x}s':>9}" for x in (0.1, 0.5, 1, 2, 5, 10)) + f"{'noise':>8}")
    traces = {}
    for run, hc in (("261014-2223", "High-C"), ("261014-2151", "no High-C")):
        z = np.load(one(f"demo/switchtest/{run}_CAPACITANCE*/traces.npz"),
                    allow_pickle=True)
        for k in sorted(z.keys()):
            t, _, ib = z[k]
            t, ib = finite(t, ib)
            v = float(k.split("_v")[1].split("_")[0])
            inf = np.median(ib[-100:])
            traces[(hc, v)] = (t, ib, inf)
            cells = "".join(
                f"{100.0 * (ib[min(np.searchsorted(t, x), len(ib) - 1)] / inf - 1.0):+8.2f}%"
                for x in (0.1, 0.5, 1, 2, 5, 10))
            print(f"    {hc:<12}{v:+7.2f}{abs(inf):12.4e}{cells}"
                  f"{100 * abs(rsd(ib[-100:]) / inf):7.2f}%")
    print("    !! this is the TOTAL cathode current against its own 20 s tail, so a "
          "component\n       slower than 20 s is invisible to it -- see section 15.")
    if ax is None:
        return
    for (hc, v), (t, ib, inf) in sorted(traces.items()):
        if abs(v + 0.85) > 1e-6:
            continue
        ax.plot(t, 100 * (ib / inf - 1), lw=1, label=f"{hc}, {v:+.2f} V")
    ax.axvline(2.0, color="r", ls="--", lw=1)
    ax.axhline(0, color="k", lw=0.5)
    ax.text(2.3, ax.get_ylim()[1] * 0.55, "pixel_switch_settle_s", color="r", fontsize=7)
    ax.set_xlabel("time after the frame change (s)")
    ax.set_ylabel("deviation from the 20 s tail (%)")
    ax.set_title("Q4  a frame change costs <1 % and is over in ~0.5 s\n"
                 "in the total cathode current, within a 20 s record", fontsize=9)
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3)


def demo() -> None:
    """Self-check with no hardware and no run data: the reductions, not the plots."""
    # decile() on a pure ramp: last tenth over first tenth, computed by hand
    y = np.linspace(1.0, 1.1, 200)
    a, _, b, _ = decile(y)
    assert abs(b / a - np.mean(y[-20:]) / np.mean(y[:20])) < 1e-12
    assert abs(b / a - 1.0 - 0.0900) < 1e-3, b / a

    rng = np.random.default_rng(0)
    _, ae, _, _ = decile(1.0 + 0.01 * rng.standard_normal(1000))
    assert 0.0005 < ae < 0.002, ae          # SEM of 100 samples at 1 % sigma

    # finite() must drop the over-range marker and the NaN, and keep the rest aligned
    t = np.arange(5.0)
    tt, yy = finite(t, np.array([1.0, 9.91e37, 2.0, np.nan, 3.0]))
    assert list(tt) == [0.0, 2.0, 4.0] and list(yy) == [1.0, 2.0, 3.0], (tt, yy)

    assert abs(rsd(rng.standard_normal(20000)) - 1.0) < 0.05

    # the Q1 discriminator. This is the whole argument of Q1, so it gets a test:
    # a stale frame write gives a FLAT ratio of 1 and must score 0; the measured
    # bias-dependent leak must score near +1.
    v = -np.linspace(0.8, 2.6, 30)
    assert faintness_corr(np.ones(30), v) == 0.0
    assert faintness_corr(np.linspace(1.0, 0.38, 30), v) > 0.99
    print("ok  plot_open_questions: decile exact on a ramp, over-range dropped, robust "
          "sd calibrated, the stale-frame discriminator separates flat from rising")


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

    args.out.mkdir(parents=True, exist_ok=True)
    for fn, name in ((q1, "fig26_bitval0"), (q2, "fig27_dwell_optical"),
                     (q3, "fig28_highc_converge"), (q4, "fig29_pixel_switch")):
        fig, ax = plt.subplots(figsize=(7.2, 4.6))
        fn(ax)
        fig.tight_layout()
        fig.savefig(args.out / f"{name}.png", dpi=130)
        plt.close(fig)
        print(f"  wrote {args.out / (name + '.png')}")


if __name__ == "__main__":
    main()
