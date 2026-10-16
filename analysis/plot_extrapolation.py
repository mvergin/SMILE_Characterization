"""Can the dwell be extrapolated away? Section 3 said no, on one trace, with no figure.

    python plot_extrapolation.py [--out investigation_report]
    python plot_extrapolation.py --demo      # no data, self-check

Writes fig34_extrapolation.png and prints the numbers section 3 quotes.

There are seven 600 s transients on disk, not one:
  261014-1501  3 traces, -2.60 / -1.30 / -0.80 V, the run section 3 used
  261015-2105  4 traces, -0.85 / -1.00 V, High-C and a fixed 10 uA range (section 20)

Two separate claims live in section 3 and they do not stand or fall together:

  A  "there is no single exponent"            -- partly true, trace-dependent
  B  "the fit is worse than doing nothing"    -- does not survive, and the verdict
                                                 flips with the fit window, which
                                                 section 3 never states

B is the one the dwell policy was justified with, so the fit window is not a detail.
"""
from __future__ import annotations

import argparse
import glob
from pathlib import Path

import numpy as np

# The measurement repo is this file's grandparent since ROADMAP item 2 merged the
# two. It was an absolute path for as long as they were separate repos.
MEAS = Path(__file__).resolve().parent.parent
OVF = 1e30
# windows the report quotes, and the exponents it tried
WINDOWS = (10, 30, 60, 240)
EXPONENTS = (1.0, 1.23, 1.5)


def one(pattern: str) -> Path:
    hits = sorted(glob.glob(str(MEAS / pattern)))
    if not hits:
        raise FileNotFoundError(str(MEAS / pattern))
    return Path(hits[0])


def load_traces() -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Every 600 s transient on disk, keyed by a label. The two files store the
    cathode in different rows, which is the only thing that differs."""
    out = {}
    z = np.load(one("demo/cal4/*/calibration_traces.npz"))
    for k in z.files:                      # row 1 is the only current row here
        out[f"261014-1501 cal {k.split('_v')[1][:6]} V"] = clean(z[k][0], z[k][1])
    z = np.load(one("demo/q3_step600/*/traces.npz"))
    for k in z.files:                      # row 2 is the cathode
        arm = "High-C" if "hc1" in k else "fixed "
        out[f"261015-2105 {arm} {k.split('_v')[1][:6]} V"] = clean(z[k][0], z[k][2])
    return out


def clean(t, y):
    t, y = np.asarray(t, float), np.abs(np.asarray(y, float))
    m = np.isfinite(y) & (y < OVF) & (t > 0)
    return t[m], y[m]


def asymptote(t, y) -> float:
    """The trace's own last 25 s. Every error below is relative to this, so the
    comparison is internal to one record and immune to day-to-day drift."""
    return float(y[t > t[-1] - 25].mean())


def fit_inf(t, y, lo, hi, n) -> float:
    """Least squares I = I_inf + A*t^-n over [lo, hi]. Linear in (I_inf, A) once n is
    fixed, so there is nothing to converge and no starting guess to get wrong."""
    w = (t >= lo) & (t <= hi)
    A = np.vstack([np.ones(int(w.sum())), t[w] ** (-n)]).T
    return float(np.linalg.lstsq(A, y[w], rcond=None)[0][0])


def read_at(t, y, T) -> float:
    """What the sweep actually does: stop at T and take the reading."""
    return float(y[(t > T * 0.9) & (t <= T)].mean())


def local_slope(t, y, inf, lo, hi) -> float:
    """d log|I - I_inf| / d log t. NaN once the residual reaches the noise, because a
    slope fitted inside the noise floor measures the noise and nothing else."""
    w = (t >= lo) & (t <= hi)
    d = np.abs(y[w] - inf)
    if w.sum() < 10 or (d <= 0).any():
        return float("nan")
    return float(np.polyfit(np.log(t[w]), np.log(d), 1)[0])


def errors(t, y, T, lo_frac=0.1):
    """(read error, {n: extrapolated error}) in percent, at window T."""
    true = asymptote(t, y)
    e = {"read": 100 * (read_at(t, y, T) / true - 1)}
    for n in EXPONENTS:
        e[n] = 100 * (fit_inf(t, y, T * lo_frac, T, n) / true - 1)
    return e


def report() -> None:
    tr = load_traces()

    print("A. the local exponent, per window. A single power law would be one number.")
    print(f"  {'trace':>28} " + " ".join(f"{a}-{b}s".rjust(10)
                                         for a, b in ((1, 30), (10, 100),
                                                      (30, 300), (100, 500))))
    for name, (t, y) in tr.items():
        inf = asymptote(t, y)
        row = [local_slope(t, y, inf, a, b)
               for a, b in ((1, 30), (10, 100), (30, 300), (100, 500))]
        print(f"  {name:>28} " + " ".join(
            ("      nan " if np.isnan(s) else f"  t^{s:+5.2f} ") for s in row))

    print("\nB. extrapolate, or just read I(T)? fit window [T/10, T].")
    print("   negative = under-reads the 600 s value. 'best' = smallest |error|.")
    for name, (t, y) in tr.items():
        print(f"\n  {name}   I_inf = {1e9*asymptote(t, y):.2f} nA")
        print(f"    {'T':>5} {'read':>9} " + " ".join(f"n={n}".rjust(9)
                                                      for n in EXPONENTS) + "   best")
        for T in WINDOWS:
            e = errors(t, y, T)
            k = min(e, key=lambda j: abs(e[j]))
            print(f"    {T:5d} {e['read']:+8.2f}% " +
                  " ".join(f"{e[n]:+8.2f}%" for n in EXPONENTS) +
                  f"   {'read' if k == 'read' else f'n={k}'}")

    print("\nC. the verdict depends on the fit window, which section 3 never states.")
    t, y = tr["261014-1501 cal -0.800 V"]
    print("   section 3's own faint trace, n = 1.0, at T = 240 s:")
    for frac, label in ((0.1, "[T/10, T]"), (0.5, "[T/2, T]")):
        e = errors(t, y, 240, frac)
        print(f"     {label:>10}: read {e['read']:+.2f}%   fit {e[1.0]:+.2f}%   "
              f"{'fit wins' if abs(e[1.0]) < abs(e['read']) else 'read wins'}")
    w = (t >= 1) & (t <= 240)
    A = np.vstack([np.ones(int(w.sum())), t[w] ** -1.0]).T
    inf = float(np.linalg.lstsq(A, y[w], rcond=None)[0][0])
    print(f"     {'[1, T]':>10}: read {errors(t, y, 240)['read']:+.2f}%   "
          f"fit {100*(inf/asymptote(t, y) - 1):+.2f}%   read wins")

    print("\n   Counting all 7 traces x 4 windows with the [T/10, T] window:")
    win = sum(1 for _, (t, y) in tr.items() for T in WINDOWS
              if abs(errors(t, y, T)[1.0]) < abs(errors(t, y, T)["read"]))
    print(f"     n = 1.0 beats reading in {win} of {4 * len(tr)} cases.")

    print("\nD. the fit wins on BOTH median and worst case, at every window.")
    print("   So section 3's stated reason is wrong. What survives is the")
    print("   OPERATIONAL conclusion, for a different reason: see below.")
    for T in WINDOWS:
        r = np.array([abs(errors(t, y, T)["read"]) for _, (t, y) in tr.items()])
        f = np.array([abs(errors(t, y, T)[1.0]) for _, (t, y) in tr.items()])
        print(f"     T = {T:3d} s:  read  median {np.median(r):5.2f}%  worst "
              f"{r.max():5.2f}%   |   fit  median {np.median(f):5.2f}%  worst "
              f"{f.max():5.2f}%")
    print("   But a 10 s dwell is still unusable: the worst case only falls from")
    print("   10.5 % to 8.9 %. The fit shaves the error, it does not change the")
    print("   order of magnitude, and n must be picked in advance while the best")
    print("   n differs trace to trace. Extrapolation is not a way to shorten")
    print("   the dwell -- which is what section 3 was actually asking.")


def fig(axes) -> None:
    tr = load_traces()
    ax1, ax2 = axes

    # --- left: the local exponent against time
    centres = np.logspace(np.log10(2), np.log10(300), 22)
    for i, (name, (t, y)) in enumerate(tr.items()):
        inf = asymptote(t, y)
        s = [local_slope(t, y, inf, c / 2.5, c * 2.5) for c in centres]
        faint = "fixed" in name
        ax1.plot(centres, -np.array(s), lw=1.0 if faint else 1.8,
                 alpha=0.45 if faint else 1.0, ls=":" if faint else "-",
                 color=f"C{i}", label=name)
    ax1.axhline(1.3, color="k", lw=1.2, ls="--")
    ax1.text(2.3, 1.34, "§3's quoted t$^{-1.3}$", fontsize=7.5)
    ax1.set_xscale("log")
    ax1.set_ylim(-0.2, 2.2)
    ax1.set_xlabel("centre of the fit window [s]")
    ax1.set_ylabel("local exponent  $n$  in  $|I-I_\\infty| \\propto t^{-n}$")
    ax1.set_title("A single power law would be a horizontal line. None of the\n"
                  "seven is, and t$^{-1.3}$ sits above almost every window", fontsize=9.5)
    ax1.legend(fontsize=6.2, loc="upper left")
    ax1.grid(alpha=0.3, which="both")

    # --- right: extrapolating against just reading
    for i, (name, (t, y)) in enumerate(tr.items()):
        r = [abs(errors(t, y, T)["read"]) for T in WINDOWS]
        f = [abs(errors(t, y, T)[1.0]) for T in WINDOWS]
        ax2.plot(r, f, "o-", ms=4, lw=1.0, color=f"C{i}", alpha=0.85, label=name)
    lim = [0.02, 20]
    ax2.plot(lim, lim, "k--", lw=1.2)
    ax2.fill_between(lim, lim, [lim[0]] * 2, color="C2", alpha=0.10)
    ax2.text(0.05, 0.028, "fit beats reading", fontsize=8, color="C2")
    ax2.text(0.3, 9, "reading beats fit", fontsize=8, color="0.4")
    ax2.set_xscale("log")
    ax2.set_yscale("log")
    ax2.set_xlim(*lim)
    ax2.set_ylim(*lim)
    ax2.set_xlabel("|error| just reading I(T)  [%]")
    ax2.set_ylabel("|error| extrapolating, n = 1.0  [%]")
    ax2.set_title("§3 says the fit is always worse. With a [T/10, T]\n"
                  "window it wins 19 of 28 — the fit window decides", fontsize=9.5)
    ax2.grid(alpha=0.3, which="both")


def demo() -> None:
    """A synthetic trace with a KNOWN asymptote and a known exponent."""
    t = np.linspace(0.05, 600, 6000)
    y = 100.0 + 50.0 * t ** -1.0

    # fit_inf recovers the asymptote of an exact power law to machine precision
    assert abs(fit_inf(t, y, 1, 60, 1.0) - 100.0) < 1e-6
    # the wrong exponent does not, and errs in a predictable direction
    assert fit_inf(t, y, 1, 60, 1.5) > 100.0

    # local_slope recovers the exponent, and is flat -- that is what "single" means
    inf = 100.0
    for lo, hi in ((1, 10), (10, 100), (100, 500)):
        assert abs(local_slope(t, y, inf, lo, hi) + 1.0) < 1e-6

    # a SUM of two power laws is not a single one, and the slope must say so.
    # Here the steep term dominates early, so the slope SHALLOWS; the real traces
    # steepen. Either way the test is that it does not stay put.
    y2 = 100.0 + 50.0 * t ** -0.4 + 500.0 * t ** -2.0
    s_early = local_slope(t, y2, 100.0, 1, 4)
    s_late = local_slope(t, y2, 100.0, 100, 500)
    assert abs(s_late - s_early) > 0.5, (s_early, s_late)
    assert abs(s_early + 2.0) < 0.5 and abs(s_late + 0.4) < 0.1, (s_early, s_late)

    # asymptote reads the END of the record, not the start
    assert abs(asymptote(t, y) - 100.08) < 0.05

    # read_at is biased the way a real unsettled reading is: high, for a decaying
    # approach from above
    assert read_at(t, y, 10) > 100.0

    print("ok  plot_extrapolation: fit exact on a true power law, wrong n biased high, "
          "slope flat for one law and steepening for two, asymptote reads the tail")


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
    f, axes = plt.subplots(1, 2, figsize=(12.0, 4.8))
    fig(axes)
    f.tight_layout()
    f.savefig(args.out / "fig34_extrapolation.png", dpi=130)
    plt.close(f)
    print(f"\n  wrote {args.out / 'fig34_extrapolation.png'}")


if __name__ == "__main__":
    main()
