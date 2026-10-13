"""Plot the settling transients and their spectra from a capacitance characterisation run.

    python plot_transients.py <run_dir>

Reads `traces.npz` plus the summary CSVs that `characterize_capacitance.py` writes
(SMILE_Characterization) and produces, per run:

    transients.png    I(t) after each voltage step, log time axis, with the settled
                      value and the 1% band marked
    spectra.png       amplitude spectrum of each trace, mains line marked

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
        return {k: d[k] for k in d.files}


def label_of(key: str) -> str:
    """Trace key -> the readable part, e.g. "step_hc1_rNone_v-0.800"."""
    return key.split("_", 1)[1].rsplit("_", 1)[0]


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
                        label=f"{label.split('_v')[-1]} V   (settled {settled*1e9:,.0f} nA)")
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
                 "a straight line here is a power law, not an exponential", fontsize=9)
    cx.legend(fontsize=8)
    cx.grid(alpha=0.3, which="both")

    for f_, a_ in ((fig, ax), (fig2, bx)):
        for j in range(n, nrow * ncol):
            a_[j // ncol][j % ncol].axis("off")
        f_.tight_layout(rect=(0, 0, 1, 0.94))
    fig3.tight_layout()
    fig.suptitle(f"settling transients — {run_dir.name}", fontsize=11)
    fig2.suptitle(f"transient spectra — {run_dir.name}  (is a sine riding on it?)",
                  fontsize=11)
    for f_, name in ((fig, "transients.png"), (fig2, "spectra.png"),
                     (fig3, "settling.png")):
        out = run_dir / name
        f_.savefig(out, dpi=140, bbox_inches="tight")
        print(f"wrote {out}")
        plt.close(f_)


def summarise(traces: dict[str, np.ndarray], mains_hz: float) -> None:
    """Periodic content per capture, computed from the traces themselves.

    Not read from the summary CSV: a CSV written before the tail-only fix reports an
    `rms_ac` ~30x too high, because the quadratic detrend cannot remove a power-law
    transient. Recomputing keeps this table consistent with the plot beside it.
    """
    print("periodic content per capture (settled tail only)")
    print(f"  {'capture':<22} {'fs':>9} {'peak':>10} {'amp':>9} "
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
        cut = int(0.5 * tc.size)
        print(f"  {label_of(key):<22} {fs:>7.1f}Hz {f[sel][k]:>8.1f}Hz "
              f"{amp[sel][k]*1e9:>7.2f}nA {band(mains_hz)*1e9:>7.2f}nA "
              f"{band(2*mains_hz)*1e9:>6.2f}nA {np.std(ic[cut:])*1e9:>7.2f}nA")
    print("\n  A peak at the mains line means the readings carry mains pickup, which an")
    print("  integer NPLC would null entirely. NPLC 0.01 rejects none of it.")
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
    traces = load_traces(run_dir)
    summarise(traces, args.mains_hz)
    plot(run_dir, traces, args.mains_hz)


if __name__ == "__main__":
    main()
