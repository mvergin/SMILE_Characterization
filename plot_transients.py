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


def plot(run_dir: Path, traces: dict[str, np.ndarray], mains_hz: float) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    keys = sorted(traces)
    n = len(keys)
    ncol = min(3, n)
    nrow = int(np.ceil(n / ncol))

    fig, ax = plt.subplots(nrow, ncol, figsize=(5.2 * ncol, 3.4 * nrow), squeeze=False)
    fig2, bx = plt.subplots(nrow, ncol, figsize=(5.2 * ncol, 3.4 * nrow), squeeze=False)
    for idx, key in enumerate(keys):
        a, b = ax[idx // ncol][idx % ncol], bx[idx // ncol][idx % ncol]
        t, _ia, ib = traces[key]
        tc, ic = clean(np.asarray(t, float), np.abs(np.asarray(ib, float)))
        label = label_of(key)
        if tc.size < 10:
            a.set_title(f"{label}\n(all over-range)", fontsize=8)
            continue

        settled = float(np.median(ic[int(0.8 * ic.size):]))
        a.plot(tc, ic * 1e9, lw=0.6, color="tab:blue")
        a.axhline(settled * 1e9, color="tab:red", ls="--", lw=1,
                  label=f"settled {settled*1e9:.1f} nA")
        a.axhspan(settled * 0.99 * 1e9, settled * 1.01 * 1e9, color="tab:red", alpha=0.12,
                  label="1% band")
        a.set_xscale("symlog", linthresh=max(tc[1] if tc.size > 1 else 1e-3, 1e-3))
        a.set_xlabel("t after step (s)")
        a.set_ylabel("|I| (nA)")
        a.set_title(label, fontsize=8)
        a.legend(fontsize=6)
        a.grid(alpha=0.3, which="both")

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

    for f_, a_ in ((fig, ax), (fig2, bx)):
        for j in range(n, nrow * ncol):
            a_[j // ncol][j % ncol].axis("off")
        f_.tight_layout()
    fig.suptitle(f"settling transients — {run_dir.name}", y=1.0, fontsize=10)
    fig2.suptitle(f"transient spectra — {run_dir.name} "
                  f"(is a sine riding on it?)", y=1.0, fontsize=10)
    for f_, name in ((fig, "transients.png"), (fig2, "spectra.png")):
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
