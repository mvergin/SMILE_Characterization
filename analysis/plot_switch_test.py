"""What a pixel switch at a held bias costs, and what that buys on run time.

    python plot_switch_test.py RUN [RUN ...] [--label L L] [--out DIR]

RUN is a `characterize_capacitance.py --what switch` directory (switch.csv + traces.npz).
Give two to compare High-C against no High-C.

The question behind it: the sweep pays its settle once per (pixel, bias) because it
loops pixel-major. The array carries one global capacitance charged by the common
cathode, so if a frame change costs nothing the loops can be inverted and the settle is
paid once per bias for every pixel.

    switch_cost.png   left  : deviation from the settled value against how long you wait
                      right : run time against pixel count, both loop orders

The left panel is deliberately NOT read from `t_1pct`. A short capture of a quiet signal
returns the settling criterion's "nothing was ever outside the band" sentinel, and with
High-C off that band is ~10 % of the reading -- which would announce success from a
measurement that cannot see 10 %. The reference here is the mean of the record's second
half, and the error bar is the spread of 1 s windows within it, which contains the
correlated low-frequency noise that an N**-0.5 estimate throws away.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

WAITS = (0.0, 0.2, 0.5, 1.0, 2.0, 5.0)


def load(run: Path) -> dict[float, dict]:
    """Per-bias: the wait-vs-deviation curve and the measured 1 s resolution."""
    npz = run / "traces.npz"
    if not npz.is_file():
        sys.exit(f"{run} has no traces.npz - was it run with --what switch?")
    z = np.load(npz)
    out = {}
    for key in sorted(z.files):
        if "switch" not in key:
            continue
        a = np.asarray(z[key], float)
        if a.ndim != 2 or a.shape[0] < 3 or a.shape[1] < 20:
            continue
        t, i = a[0], np.abs(a[2])
        dt = float(np.median(np.diff(t))) or 1e-3
        w = max(2, int(round(1.0 / dt)))
        if i.size < 3 * w:
            continue
        ref = i[i.size // 2:]
        mu = float(ref.mean())
        wins = [i[s:s + w].mean() for s in range(2 * w, i.size - w, w)]
        res = float(np.std(wins, ddof=1)) if len(wins) > 2 else float("nan")
        dev = {}
        for d in WAITS:
            j = int(d / dt)
            seg = i[j:j + w]
            dev[d] = 100.0 * (seg.mean() / mu - 1.0) if seg.size >= w else float("nan")
        v = float(key.split("_v")[1][:6])
        out[v] = dict(dev=dev, settled_nA=1e9 * mu, res_pct=100.0 * res / mu,
                      noise_nA=1e9 * float(ref.std(ddof=1)), record_s=float(t[-1]))
    if not out:
        sys.exit(f"{run} has no switch traces")
    return out


def runtime_hours(n_px: int, n_pts: int, lowv_s: float, highv_s: float,
                  low_frac: float, ref_s: float, switch_s: float, order: str) -> float:
    """Settle time alone, the term that dominates. Mirrors acquire.py's estimate."""
    n_low = int(round(n_pts * low_frac))
    n_high = n_pts - n_low
    if order == "bias_major":
        # the dwell once for all pixels; one reference settle per point, not two,
        # because the frame is blanked next anyway
        return (n_low * lowv_s + n_high * highv_s
                + n_px * n_pts * (switch_s + ref_s)) / 3600.0
    return n_px * (n_low * (lowv_s + 2 * ref_s)
                   + n_high * (highv_s + 2 * ref_s)) / 3600.0


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("runs", nargs="+", type=Path)
    ap.add_argument("--label", nargs="+", default=None)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--n-points", type=int, default=30)
    ap.add_argument("--lowv-s", type=float, default=54.47)
    ap.add_argument("--highv-s", type=float, default=1.36)
    ap.add_argument("--low-frac", type=float, default=0.6)
    ap.add_argument("--ref-s", type=float, default=2.0)
    ap.add_argument("--switch-s", type=float, default=2.0)
    args = ap.parse_args()

    out = args.out or args.runs[-1]
    out.mkdir(parents=True, exist_ok=True)
    labels = args.label or [r.name for r in args.runs]
    if len(labels) != len(args.runs):
        sys.exit("--label needs one label per run")

    data = [load(r) for r in args.runs]
    rows = []
    for lbl, d in zip(labels, data):
        for v, rec in sorted(d.items()):
            rows.append(dict(run=lbl, bias_V=v, settled_nA=rec["settled_nA"],
                             noise_nA=rec["noise_nA"], resolution_pct=rec["res_pct"],
                             **{f"dev_{w:g}s_pct": rec["dev"][w] for w in WAITS}))
    tab = pd.DataFrame(rows)
    pd.set_option("display.width", 220)
    print(tab.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    tab.to_csv(out / "switch_cost.csv", index=False)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (a, b) = plt.subplots(1, 2, figsize=(12.6, 5.0))

    styles = ["o-", "s--", "^:"]
    colors = {}
    for k, (lbl, d) in enumerate(zip(labels, data)):
        for v, rec in sorted(d.items()):
            c = colors.setdefault(v, f"C{len(colors)}")
            xs = [w for w in WAITS if np.isfinite(rec["dev"][w])]
            ys = [abs(rec["dev"][w]) for w in xs]
            a.plot(xs, ys, styles[k % len(styles)], ms=5, color=c,
                   label=f"{v:+.2f} V, {lbl}")
            a.fill_between(xs, 0, rec["res_pct"], color=c, alpha=0.08)
    a.axhline(3.8, color="k", lw=1.2, ls="-",
              label="what a 54 s pixel-major dwell leaves")
    a.set_yscale("log")
    a.set_xlabel("wait after the frame change, before measuring (s)")
    a.set_ylabel("|deviation| from the settled value (%)")
    a.set_title("A pixel switch at a held bias costs well under a percent\n"
                "shaded = that capture's own 1 s resolution", fontsize=10)
    a.legend(fontsize=7, ncol=2)
    a.grid(alpha=0.3, which="both")

    n = np.arange(4, 201)
    for order, col, name in (("pixel_major", "tab:red", "pixel-major (for pixel: for bias)"),
                             ("bias_major", "tab:blue", "bias-major (for bias: for pixel)")):
        h = [runtime_hours(int(k), args.n_points, args.lowv_s, args.highv_s,
                           args.low_frac, args.ref_s, args.switch_s, order) for k in n]
        b.plot(n, h, color=col, lw=2, label=name)
    for k in (16, 100):
        hp = runtime_hours(k, args.n_points, args.lowv_s, args.highv_s, args.low_frac,
                           args.ref_s, args.switch_s, "pixel_major")
        hb = runtime_hours(k, args.n_points, args.lowv_s, args.highv_s, args.low_frac,
                           args.ref_s, args.switch_s, "bias_major")
        b.annotate(f"{k} px: {hp:.1f} h -> {hb:.1f} h  ({hp / hb:.0f}x)",
                   xy=(k, hb), xytext=(k * 0.75, hb * 3.2), fontsize=8,
                   arrowprops=dict(arrowstyle="->", lw=0.8))
    b.axhline(24, color="k", lw=0.8, ls=":")
    b.text(5, 26, "one day", fontsize=8)
    b.set_yscale("log")
    b.set_xlabel("pixels swept")
    b.set_ylabel("settle time alone (hours)")
    b.set_title(f"What the inversion buys ({args.n_points} points, "
                f"{args.lowv_s:g}/{args.highv_s:g} s calibrated dwell)", fontsize=10)
    b.legend(fontsize=8, loc="lower right")
    b.grid(alpha=0.3, which="both")

    fig.suptitle("The dwell belongs to the bias, not to the pixel", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(out / "switch_cost.png", dpi=300)
    plt.close(fig)
    print(f"\nwrote {out / 'switch_cost.png'} and switch_cost.csv")


def demo() -> None:
    """Self-check on the run-time model, which is the panel that drives a decision."""
    kw = dict(n_pts=30, lowv_s=54.47, highv_s=1.36, low_frac=0.6, ref_s=2.0, switch_s=2.0)
    px1 = runtime_hours(1, order="pixel_major", **kw)
    px2 = runtime_hours(2, order="pixel_major", **kw)
    assert abs(px2 - 2 * px1) < 1e-9, "pixel_major must be linear in pixels"

    bm1 = runtime_hours(1, order="bias_major", **kw)
    bm2 = runtime_hours(2, order="bias_major", **kw)
    # bias_major has a fixed dwell term, so doubling the pixels less than doubles it
    assert bm2 < 2 * bm1, (bm1, bm2)
    dwell = (18 * 54.47 + 12 * 1.36) / 3600.0
    assert abs((bm2 - bm1) - 30 * 4.0 / 3600.0) < 1e-9, "per-pixel term is n_pts*(2+2)s"
    assert abs(bm1 - (dwell + 30 * 4.0 / 3600.0)) < 1e-9, bm1

    # and the headline the report quotes
    for n, want in ((16, 6.1), (100, 8.6)):
        r = runtime_hours(n, order="pixel_major", **kw) / runtime_hours(n, order="bias_major", **kw)
        assert abs(r - want) < 0.15, f"{n} px: speedup {r:.2f}, expected ~{want}"
    print("ok  plot_switch_test: the run-time model matches the figures it draws")


if __name__ == "__main__":
    if "--demo" in sys.argv:
        demo()
    else:
        main()
