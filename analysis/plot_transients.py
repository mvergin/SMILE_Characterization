"""Plot the settling transients and their spectra from a capacitance characterisation run.

    python plot_transients.py <run_dir>

Reads `traces.npz` plus the summary CSVs that `characterize_capacitance.py` writes
(SMILE_Characterization) and produces, per run:

    transients.png    I(t) after each voltage step, log time axis, with the settled
                      value and the 1% band marked
    spectra.png       amplitude spectrum of each trace, mains line marked
    switch_sync/<tag>.png
                      one per switch capture with a baseline: I(t) and P(t) across
                      the frame change, the switch's sent-to-ack window shaded

The spectra exist to answer one specific question: **is a sine riding on the
transient?** At NPLC 0.01 the SMU integrates 1/100 of a mains cycle, so it has no
mains rejection at all — only an integer NPLC does. The measurement script already
records `spec_peak_hz`, `spec_peak_amp` and `spec_mains_amp` per capture; this draws
the spectrum behind those numbers so a line can be identified rather than guessed.

`traces.npz` holds one `(3, N)` array per capture — time, channel A, channel B —
thinned uniformly, so the frequency content is preserved and the effective sample
rate is the trailing `_k` stride in each key.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

OVERFLOW = 1e36          # the 2600's over-range sentinel, same as the measurement side


def load_traces(run_dir: Path) -> dict[str, np.ndarray]:
    npz = run_dir / "traces.npz"
    if not npz.exists():
        sys.exit(f"{npz} not found. Re-run with --trace-points > 0 (default 4000).")
    with np.load(npz) as d:
        # Current traces only: (t, I_A, I_B), shape (3, N). Since 16 October the same
        # file also holds P(t) as pm_* and V(t) as v_*, both (2, N), and unpacking those
        # as three rows crashed on every switch and cap-step run from then on.
        keep = {k: d[k] for k in d.files
                if d[k].ndim == 2 and d[k].shape[0] == 3 and not k.startswith(("pm_", "v_"))}
        other = len(d.files) - len(keep)
    if other:
        print(f"  ({other} P(t)/V(t) trace(s) in {npz.name} not plotted here)")
    return keep


def si(x, unit="A"):
    """3.2e-07 -> "320 nA". Plot labels, not data."""
    if x is None or not np.isfinite(x) or x == 0:
        return f"{x}"
    for scale, p in ((1, ""), (1e-3, "m"), (1e-6, "u"), (1e-9, "n"), (1e-12, "p")):
        if abs(x) >= scale:
            return f"{x/scale:g} {p}{unit}"
    return f"{x:g} {unit}"


def run_settings(run_dir: Path) -> str:
    """The run-wide settings, for the figure subtitle.

    These do not vary between captures, so they are not in the trace key -- but
    without them a plot is unreadable. The compliance limit above all: in High-C the
    limit also selects the measure range, so it is the single most important number
    on the figure, and it appears nowhere in the key or the trace.
    """
    import json
    p = run_dir / "run_config.json"
    if not p.exists():
        return ""
    try:
        a = json.load(p.open(encoding="utf-8")).get("args", {})
    except (ValueError, OSError):
        return ""
    bits = []
    if a.get("v_start") is not None:
        bits.append(f"stepped from {a['v_start']:+.3f} V")
    if a.get("compliance"):
        bits.append(f"limit {si(a['compliance'])}")
    if a.get("charge_limit"):
        bits.append(f"charged at {si(a['charge_limit'])} first")
    if a.get("nplc") is not None:
        bits.append(f"NPLC {a['nplc']:g}")
    if a.get("highc"):
        bits.append("High-C: the limit sets the range")
    if a.get("pixels"):
        bits.append(f"pixel {','.join(map(str, a['pixels']))}")
    return "   ".join(bits)


def parse_key(key: str) -> dict:
    """Decode a traces.npz key into its settings.

    The key is built by the measurement script and is a filename, not a label:

        000_step_hc1_rNone_v-0.950_5
         |   |    |     |       |  stride: every 5th sample was kept
         |   |    |     |       voltage stepped TO
         |   |    |     measure range, None = not forced (High-C picks it)
         |   |    High-C: 1 on, 0 off
         |   experiment
         capture index, in run order

    `leak` keys carry a frame state (lit / off / blank) where `step` carries hc/range.
    Anything unrecognised falls through into `rest`, so a new tag never crashes a plot.
    """
    parts = key.split("_")
    out = {"idx": parts[0], "what": parts[1] if len(parts) > 1 else "?",
           "stride": parts[-1], "v": None, "v_from": None, "highc": None,
           "range": None, "rest": []}
    for p in parts[2:-1]:
        if p.startswith("v-") or p.startswith("v+"):
            out["v"] = float(p[1:])
        elif p.startswith("f-") or p.startswith("f+"):
            out["v_from"] = float(p[1:])
        elif p.startswith("hc") and p[2:].isdigit():
            out["highc"] = bool(int(p[2:]))
        elif p.startswith("r"):
            out["range"] = None if p == "rNone" else p[1:]
        else:
            out["rest"].append(p)
    return out


def label_of(key: str) -> str:
    """Trace key -> a title a human can read, e.g. "-0.950 V, High-C, range auto"."""
    k = parse_key(key)
    bits = [f"{k['v']:+.3f} V" if k["v"] is not None else k["what"]]
    if k["v_from"] is not None:
        bits[0] += f" (from {k['v_from']:+.3f})"      # the step size sets the settle time
    bits += k["rest"]
    if k["highc"] is not None:
        bits.append("High-C" if k["highc"] else "no High-C")
    if k["range"] is not None:
        bits.append(f"range {k['range']}")
    elif k["highc"]:
        bits.append("range auto")
    return ", ".join(bits)


def clean(t, i):
    """Drop over-range samples; they are a sentinel, not a measurement."""
    ok = np.isfinite(i) & (np.abs(i) < OVERFLOW)
    return t[ok], i[ok]


def amplitude_spectrum(t, i, tail_frac=0.5):
    """(freqs, amplitudes) over the settled tail, matching `spectrum_summary`.

    The tail only: the settling is a power law, so a quadratic cannot detrend it and
    over the whole trace the residual transient swamps every low-frequency bin.
    """
    if t.size < 64:
        return np.array([]), np.array([])
    cut = int((1.0 - tail_frac) * t.size)
    if t.size - cut >= 64:
        t, i = t[cut:], i[cut:]
    dt = float(np.median(np.diff(t)))
    if not dt > 0:
        return np.array([]), np.array([])
    resid = i - np.polyval(np.polyfit(t, i, 2), t)
    w = np.hanning(resid.size)
    return (np.fft.rfftfreq(resid.size, dt),
            np.abs(np.fft.rfft(resid * w)) * (4.0 / resid.size))


def smooth(i, n):
    """Moving average over 1% of the record. Returns the window used."""
    w = max(5, i.size // 100)
    return np.convolve(i, np.ones(w) / w, mode="valid"), w


def settle_time(t, i, settled, frac=0.01):
    """Last time the smoothed trace was outside the band around `settled`.

    The band is `frac` of the settled value, or 5 sigma of the averaged noise if that
    is wider -- the same tolerance `characterize_capacitance.py` writes to `t_1pct`,
    so the marker on the plot matches the number in the CSV. A plain 1% band would
    report a settling time set by noise at low currents, not by the transient.

    NaN when the trace never gets inside the band. That is the honest answer to "when
    did it settle?"; do not report 0 for it.
    """
    sm, w = smooth(i, i.size)
    noise = float(np.std(np.diff(i[int(0.8 * i.size):]))) / np.sqrt(2) if i.size > 10 else 0.0
    tol = max(frac * abs(settled), 5.0 * noise / np.sqrt(w))
    bad = np.nonzero(np.abs(sm - settled) > tol)[0]
    if bad.size == 0:
        return 0.0
    if bad[-1] + 1 >= sm.size:
        return float("nan")
    return float(t[w - 1:][bad[-1]])


def plot(run_dir: Path, traces: dict[str, np.ndarray], mains_hz: float) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    settings = run_settings(run_dir)
    keys = sorted(traces)
    n = len(keys)
    ncol = min(3, n)
    nrow = int(np.ceil(n / ncol))

    fig, ax = plt.subplots(nrow, ncol, figsize=(5.2 * ncol, 3.6 * nrow), squeeze=False)
    fig2, bx = plt.subplots(nrow, ncol, figsize=(5.2 * ncol, 3.6 * nrow), squeeze=False)
    fig3, cx = plt.subplots(figsize=(7.2, 5.0))
    for idx, key in enumerate(keys):
        a, b = ax[idx // ncol][idx % ncol], bx[idx // ncol][idx % ncol]
        t, _ia, ib = traces[key]
        tc, ic = clean(np.asarray(t, float), np.abs(np.asarray(ib, float)))
        label = label_of(key)
        if tc.size < 10:
            a.set_title(f"{label}\n(all over-range)", fontsize=8)
            continue

        settled = float(np.median(ic[int(0.8 * ic.size):]))
        t1, t01 = settle_time(tc, ic, settled, 0.01), settle_time(tc, ic, settled, 0.001)

        # Linear time axis: the question is "how many seconds", not "how many decades".
        # The first samples overshoot the settled value by 10-20x, so clip the y axis
        # to the approach instead of letting the spike set the scale.
        a.plot(tc, ic * 1e9, lw=0.7, color="tab:blue")
        a.axhline(settled * 1e9, color="tab:red", ls="--", lw=1.1,
                  label=f"settled {settled*1e9:,.1f} nA")
        a.axhspan(settled * 0.99 * 1e9, settled * 1.01 * 1e9, color="tab:red", alpha=0.15,
                  label="±1% band")
        for tt, c, nm in ((t1, "tab:green", "1%"), (t01, "tab:olive", "0.1%")):
            if np.isfinite(tt) and tt > 0:
                a.axvline(tt, color=c, ls=":", lw=1.2, label=f"settled to {nm}: {tt:.2f} s")
        a.set_xlim(0, tc[-1])
        a.set_ylim(0, settled * 1.6 * 1e9)
        a.set_xlabel("time after the voltage step (s)")
        a.set_ylabel("|I| through the LED (nA)")
        a.set_title(label, fontsize=9)
        a.legend(fontsize=7, loc="lower right")
        a.grid(alpha=0.3)

        # Same data as a deviation, log-log: this is where the decay law is visible.
        # Smoothed, or the noise floor buries the straight section.
        sm, w = smooth(ic, ic.size)
        dev = np.abs(sm - settled) / abs(settled)
        ts = tc[w - 1:]
        m = (ts > 0) & (dev > 0)
        line, = cx.plot(ts[m], dev[m] * 100, lw=1.0,
                        label=f"{label}   (settled {settled*1e9:,.0f} nA)")
        fit = m & (ts > 0.8) & (ts < 10) & (dev > 2e-3)
        if fit.sum() > 20:
            slope = np.polyfit(np.log(ts[fit]), np.log(dev[fit]), 1)[0]
            k = int(np.argmax(ts[fit] > 2.0))
            cx.annotate(f"t^{slope:.1f}", (ts[fit][k], dev[fit][k] * 100), fontsize=8,
                        color=line.get_color(), textcoords="offset points", xytext=(6, 6))

        f, amp = amplitude_spectrum(tc, ic)
        if f.size and np.any(amp[f > 2.0] > 0):
            sel = f > 2.0
            b.semilogy(f[sel], amp[sel] * 1e9, lw=0.6, color="tab:purple")
            for h, style in ((mains_hz, "--"), (2 * mains_hz, ":")):
                if h < f[-1]:
                    b.axvline(h, color="red", ls=style, lw=1,
                              label=f"{h:.0f} Hz")
            k = int(np.argmax(amp[sel]))
            b.annotate(f"{f[sel][k]:.1f} Hz\n{amp[sel][k]*1e9:.2f} nA",
                       (f[sel][k], amp[sel][k] * 1e9), fontsize=6,
                       textcoords="offset points", xytext=(4, 4))
        b.set_xlabel("frequency (Hz)")
        b.set_ylabel("amplitude (nA)")
        b.set_title(label, fontsize=8)
        if b.get_legend_handles_labels()[1]:
            b.legend(fontsize=6)
        b.grid(alpha=0.3, which="both")

    cx.set_xscale("log")
    cx.set_yscale("log")
    cx.set_ylim(2e-2, None)      # below the 0.1% line is all noise floor
    cx.axhline(1.0, color="tab:red", ls="--", lw=1, label="1% of settled")
    cx.axhline(0.1, color="tab:red", ls=":", lw=1, label="0.1% of settled")
    cx.set_xlabel("time after the voltage step (s)")
    cx.set_ylabel("|I − I∞| / I∞  (%)")
    cx.set_title(f"how the current approaches its final value — {run_dir.name}\n"
                 f"{settings}\n"
                 "a straight line here is a power law, not an exponential", fontsize=9)
    cx.legend(fontsize=8)
    cx.grid(alpha=0.3, which="both")

    for f_, a_ in ((fig, ax), (fig2, bx)):
        for j in range(n, nrow * ncol):
            a_[j // ncol][j % ncol].axis("off")
        f_.tight_layout(rect=(0, 0, 1, 0.90))
    fig3.tight_layout()
    fig.suptitle(f"settling transients — {run_dir.name}\n{settings}", fontsize=11)
    fig2.suptitle(f"transient spectra — {run_dir.name}  (is a sine riding on it?)\n"
                  f"{settings}", fontsize=11)
    for f_, name in ((fig, "transients.png"), (fig2, "spectra.png"),
                     (fig3, "settling.png")):
        out = run_dir / name
        f_.savefig(out, dpi=300, bbox_inches="tight")
        print(f"wrote {out}")
        plt.close(f_)


def plot_switch_sync(run_dir: Path) -> None:
    """One figure per switch capture: I(t) and P(t) across the frame change.

    Only for captures with a baseline (switch.csv has ``sync_switch_ack_s``, written
    since 2026-10-25). The orange band runs from the host sending the frame to the
    FPGA's ack: the switch happened somewhere inside it. The grey band at t = 0 is the
    uncertainty of the first sample's host time, which every other time on the axis
    inherits. P(t) is on the right axis when it was recorded on the same axis.
    """
    import csv
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    path = run_dir / "switch.csv"
    npz = run_dir / "traces.npz"
    if not (path.exists() and npz.exists()):
        return
    rows = [r for r in csv.DictReader(path.open())
            if np.isfinite(float(r.get("sync_switch_ack_s") or "nan"))]
    if not rows:
        return
    out_dir = run_dir / "switch_sync"
    out_dir.mkdir(exist_ok=True)
    with np.load(npz) as d:
        keys = list(d.files)
        used: set[str] = set()
        for r in rows:
            tag = r["trace_tag"]
            cur = next((k for k in keys if k not in used and not k.startswith(("pm_", "v_"))
                        and f"_{tag}_" in k), None)
            if cur is None:
                continue
            used.add(cur)
            pmk = next((k for k in keys if k not in used and k.startswith("pm_")
                        and k.endswith(f"_{tag}")), None)
            if pmk is not None:
                used.add(pmk)
            t, _ia, ib = d[cur]
            t, ib = clean(t, np.abs(ib))
            lo, hi = float(r["sync_switch_sent_s"]), float(r["sync_switch_ack_s"])
            unc = float(r["sync_first_unc_s"])
            fig, ax = plt.subplots(figsize=(10.0, 4.8))
            ax.plot(t, ib * 1e9, lw=0.8, color="tab:blue")
            ax.set_ylabel("|I_B| (nA)", color="tab:blue")
            ax.tick_params(axis="y", colors="tab:blue")
            ax.axvspan(lo, hi, color="tab:orange", alpha=0.35,
                       label=f"switch sent -> acked ({(hi - lo) * 1e3:.1f} ms)")
            ax.axvspan(-unc, unc, color="0.5", alpha=0.25,
                       label=f"first-sample time ±{unc * 1e3:.1f} ms")
            if pmk is not None and r.get("pm_on_trace_axis") == "1":
                tp, pp = d[pmk]
                ax2 = ax.twinx()
                ax2.plot(tp, pp * 1e9, lw=0.8, color="tab:red")
                ax2.set_ylabel("P (nW)", color="tab:red")
                ax2.tick_params(axis="y", colors="tab:red")
            ax.set_xlabel("time from the first SMU sample (s)")
            ax.grid(alpha=0.3)
            ax.legend(fontsize=8, loc="best")
            late = " -- START ack late, switch position NOT trustworthy" \
                if r.get("sync_start_ack_late") == "True" else ""
            ax.set_title(
                f"switch ({r['from_x']},{r['from_y']}) {r.get('from_quadrant', '')} -> "
                f"({r['to_x']},{r['to_y']}) {r.get('to_quadrant', '')}, bitval "
                f"{r['to_bitval']}, {float(r['v']):+.3f} V, hold {float(r['hold_s']):g} s, "
                f"baseline {float(r['pre_s']):g} s{late}\n{run_settings(run_dir)}",
                fontsize=9)
            fig.tight_layout()
            out = out_dir / f"{tag}.png"
            fig.savefig(out, dpi=200)
            plt.close(fig)
            print(f"wrote {out}")


def plot_isrc(run_dir: Path) -> None:
    """V(t) and I(t) with channel B held as a CURRENT source, from isrc.csv.

    A different transient to the voltage-mode ones: there the current settles, here
    the current is forced and the *voltage* settles. `isrc` polls at ~2 Hz for minutes
    instead of filling the SMU buffer, so these are slow curves by construction and
    there is no traces.npz for them.

    The plot answers whether current-sourcing the cathode can hold a bias at all. It
    cannot below the backplane leakage: the source has no way to supply the leak as
    well as the LED, so the node walks away.
    """
    import csv
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    path = run_dir / "isrc.csv"
    if not path.exists():
        return
    rows = list(csv.DictReader(path.open()))
    if not rows:
        return
    sets = sorted({float(r["i_set"]) for r in rows})
    fig, (a, b) = plt.subplots(1, 2, figsize=(12.0, 4.6))
    print(f"\ncurrent-source mode ({len(sets)} setpoints, {path.name})")
    print(f"  {'I_set':>9} {'V final':>9} {'drift, last 60 s':>18} {'I measured':>12} {'held?':>7}")
    for i_set in sets:
        rs = [r for r in rows if float(r["i_set"]) == i_set]
        t = np.array([float(r["t_s"]) for r in rs])
        v = np.array([float(r["v_b"]) for r in rs])
        ib = np.abs(np.array([float(r["i_b"]) for r in rs]))
        lbl = f"{i_set*1e9:,.0f} nA"
        a.plot(t, v, lw=1.2, label=lbl)
        b.plot(t, ib * 1e9, lw=1.2, label=lbl)
        m = t > t[-1] - 60
        drift = np.polyfit(t[m], v[m], 1)[0] if m.sum() > 2 else float("nan")
        held = abs(drift) < 1e-4                     # < 100 uV/s is settled
        print(f"  {i_set*1e9:>8.0f}n {np.mean(v[-10:]):>+9.4f} {drift*1e6:>+15.1f} uV/s "
              f"{np.mean(ib[-5:])*1e9:>10.1f}n {'yes' if held else 'NO':>7}")
    for ax_, yl, ti in ((a, "V at the cathode (V)", "voltage, with the current forced"),
                        (b, "|I| through the LED (nA)", "delivered current")):
        ax_.set_xlabel("time at this setpoint (s)")
        ax_.set_ylabel(yl)
        ax_.set_title(ti, fontsize=9)
        ax_.legend(fontsize=8, title="I set")
        ax_.grid(alpha=0.3)
    b.set_yscale("log")
    fig.suptitle(f"channel B as a current source — {run_dir.name}\n{run_settings(run_dir)}",
                 fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    out = run_dir / "isrc.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    print(f"wrote {out}")
    plt.close(fig)


def summarise(traces: dict[str, np.ndarray], mains_hz: float) -> None:
    """Periodic content per capture, computed from the traces themselves.

    Not read from the summary CSV: a CSV written before the tail-only fix reports an
    `rms_ac` ~30x too high, because the quadratic detrend cannot remove a power-law
    transient. Recomputing keeps this table consistent with the plot beside it.
    """
    print("periodic content per capture (settled tail only)")
    print(f"  {'capture':<30} {'fs':>9} {'peak':>10} {'amp':>9} "
          f"{str(int(mains_hz))+'Hz':>9} {'2x':>8} {'rms_ac':>9}")
    for key in sorted(traces):
        t, _ia, ib = traces[key]
        tc, ic = clean(np.asarray(t, float), np.abs(np.asarray(ib, float)))
        f, amp = amplitude_spectrum(tc, ic)
        if not f.size:
            continue
        fs = 1.0 / float(np.median(np.diff(tc)))
        sel = f > 2.0
        k = int(np.argmax(amp[sel])) if sel.any() else 0
        def band(target):
            if target >= fs / 2:
                return float("nan")
            b = np.abs(f - target) < max(2.0, 3 * fs / tc.size)
            return float(np.max(amp[b])) if b.any() else float("nan")
        def fmt(target, width=9):
            # Above Nyquist the line is not absent, it is unrepresentable. Saying so
            # beats printing "nannA", which reads as a broken measurement.
            if target >= fs / 2:
                return f"{'>fN':>{width}}"
            return f"{band(target)*1e9:>{width-2}.2f}nA"

        cut = int(0.5 * tc.size)
        print(f"  {label_of(key):<30} {fs:>7.1f}Hz {f[sel][k]:>8.1f}Hz "
              f"{amp[sel][k]*1e9:>7.2f}nA {fmt(mains_hz)} "
              f"{fmt(2 * mains_hz, 8)} {np.std(ic[cut:])*1e9:>7.2f}nA")
    print("\n  A peak at the mains line means the readings carry mains pickup, which an")
    print("  integer NPLC would null entirely. NPLC 0.01 rejects none of it.")
    print('  ">fN" means the sample rate cannot represent that frequency at all, so the')
    print("  capture says nothing about it either way (NPLC 1 at ~25 Hz: fN = 12 Hz).")
    print("  rms_ac is the per-sample noise in the tail. It falls as sqrt(NPLC).\n")


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_dir", help="a *_CAPACITANCE directory")
    ap.add_argument("--mains-hz", type=float, default=50.0)
    args = ap.parse_args()
    run_dir = Path(args.run_dir)
    if not run_dir.is_dir():
        sys.exit(f"{run_dir} is not a directory")
    plot_isrc(run_dir)                       # no-op unless the run has an isrc.csv
    plot_switch_sync(run_dir)                # no-op unless switch.csv has sync columns
    if (run_dir / "traces.npz").exists():
        traces = load_traces(run_dir)
        summarise(traces, args.mains_hz)
        plot(run_dir, traces, args.mains_hz)
    elif not (run_dir / "isrc.csv").exists():
        sys.exit(f"{run_dir} has neither traces.npz nor isrc.csv. For traces, re-run "
                 "the measurement with --trace-points > 0 (default 4000).")


if __name__ == "__main__":
    main()
