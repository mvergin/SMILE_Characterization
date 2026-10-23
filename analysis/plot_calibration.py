"""Plot a `measure_iqe --calibrate` record: every capture against its own final value.

    python analysis/plot_calibration.py <run>_CALIBRATION

One panel per bias. Each pixel's current (solid) and light (dashed) as a deviation from
its own settled value, on a log time axis; its t_target (the time it entered the target
band for good) marked; the target band shaded; and the dwell the calibration recommends
for that bias drawn as a vertical line. Makes the recommended dwell checkable at a
glance: the line should sit right of every pixel's t_target, with the margin to spare.

Writes calibration_traces.png into the run folder and prints the table behind it.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


def load(run_dir: Path):
    import yaml
    recs = sorted(run_dir.glob("*_calibration.yaml"))
    if not recs:
        sys.exit(f"{run_dir}: no *_calibration.yaml -- not a --calibrate run folder")
    cal = yaml.safe_load(recs[0].read_text(encoding="utf-8"))
    npz = run_dir / cal.get("traces_file", "calibration_traces.npz")
    traces = {}
    if npz.exists():
        with np.load(npz) as z:
            traces = {k: z[k] for k in z.files}
    return cal, traces


def _rolling_median(y, n):
    """Centred rolling median, edges padded: smooths a capture without shifting it."""
    pad = np.pad(y, n // 2, mode="edge")
    return np.median(np.lib.stride_tricks.sliding_window_view(pad, n), axis=1)


def knee_of(run_dir: Path) -> float:
    """The knee that splits rising from droop dwell, from the run's own config."""
    import json
    for cfg in run_dir.rglob("*_config.json"):
        try:
            return float(json.loads(cfg.read_text(encoding="utf-8"))["settings"]["vramp_knee_v"])
        except Exception:
            continue
    return -1.3


def main(run_dir: Path) -> Path:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cal, traces = load(run_dir)
    meas = cal.get("measurements") or []
    rec = cal.get("recommended") or {}
    target = 100 * float(cal.get("target", 0.01))
    knee = knee_of(run_dir)
    biases = sorted({round(float(m["v_to"]), 4) for m in meas})
    if not biases:
        sys.exit(f"{run_dir}: the calibration record has no measurements")
    pixels = sorted({(int(m["pixel_x"]), int(m["pixel_y"])) for m in meas})
    colors = {p: plt.cm.tab10(i % 10) for i, p in enumerate(pixels)}

    fig, ax = plt.subplots(1, len(biases), figsize=(5.4 * len(biases), 4.6), squeeze=False)
    print(f"{run_dir.name}: target {target:g} %, recommended {rec}")
    print(f"  {'pixel':<10} {'bias':>7} {'t_target':>9} {'t_0.1%':>8} {'settled':>11}")
    for a, v in zip(ax[0], biases):
        dwell = rec.get("vramp_lowv_settle_s" if v > knee else "vramp_highv_settle_s")
        for m in [m for m in meas if round(float(m["v_to"]), 4) == v]:
            px = (int(m["pixel_x"]), int(m["pixel_y"]))
            key = f"{px[0]}_{px[1]}_v{v:+.4f}"
            tr = traces.get(key)
            if tr is not None and tr.ndim == 2 and tr.shape[0] >= 2:
                t = tr[0].astype(float)
                ok = t >= 0.1                      # the first sample sits at ~1 us
                for row, ls in ((1, "-"), (2, "--")):
                    if row < tr.shape[0]:
                        y = np.abs(tr[row].astype(float))
                        fin = np.median(y[t > 0.8 * t.max()])
                        if fin <= 0:
                            continue
                        dev = 100 * (_rolling_median(y, 9) / fin - 1)
                        # Light only where it can be read against the target: at -0.80 V
                        # it is ~0.2 nW and its noise would cover the panel.
                        tail = 100 * (y[t > 0.8 * t.max()] / fin - 1)
                        if row == 2 and np.std(tail) > target:
                            continue
                        a.plot(t[ok], dev[ok], ls, color=colors[px], lw=1,
                               label=f"({px[0]},{px[1]})" if row == 1 else None)
            tt = m.get("t_target")
            if tt is not None and np.isfinite(float(tt)):
                a.axvline(float(tt), color=colors[px], ls=":", lw=1)
            print(f"  {f'({px[0]},{px[1]})':<10} {v:+7.3f} "
                  f"{float(tt) if tt is not None else float('nan'):9.2f} "
                  f"{float(m.get('t_01pct') or np.nan):8.2f} "
                  f"{float(m.get('settled_a', np.nan)):11.4g}")
        a.axhspan(-target, target, color="0.9", zorder=0)
        if dwell:
            a.axvline(float(dwell), color="k", lw=1.5, label=f"recommended dwell {float(dwell):g} s")
        a.set_xscale("log")
        a.set_xlim(left=0.1)
        a.set_ylim(-10, 5)
        a.set_xlabel("time after the step (s)")
        a.set_title(f"{v:+.2f} V ({'rising' if v > knee else 'droop'} side)", fontsize=10)
        a.legend(fontsize=7, loc="lower right")
    ax[0][0].set_ylabel("deviation from the final value (%)\nsolid: current, dashed: light")
    fig.suptitle(f"{run_dir.name}: each capture against its own final value (9-sample median); dotted = its "
                 f"t_target; grey = ±{target:g} % target", fontsize=10)
    fig.tight_layout()
    out = run_dir / "calibration_traces.png"
    fig.savefig(out, dpi=200)
    plt.close(fig)
    print(f"wrote {out}")
    return out


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    main(Path(sys.argv[1]))
