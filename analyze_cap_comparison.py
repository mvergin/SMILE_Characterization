"""Plot and compare what `characterize_capacitance.py --what cap step` measured.

    uv run python analyze_cap_comparison.py results/<run_A> [results/<run_B>] [-o DIR]

Built for the Array42-vs-Array38 carrier-capacitor comparison, but one run works
too -- useful before the swap. Every figure answers one question:

  cap_per_quadrant.png   is the carrier capacitance smaller, and is it uniform?
  cap_ramps.png          the raw V(t) the capacitance came from, lit against blanked
  step_fast.png          the charge edge and the first seconds of relaxation
  step_long_loglog.png   the relaxation as a power law -- the decay EXPONENT
  range_artifact.png     can High-C be dropped? does a reading track bias or range?
  settling_bars.png      electrical against optical settling, both High-C states

WHAT IS AND IS NOT COMPARABLE between two arrays. Array42 and Array38 are different
arrays on (hopefully) identical carriers, so board-level quantities -- capacitance,
settling, whether High-C is needed -- compare directly. Array properties -- leakage,
EQE, yield -- do not: a difference there may be the array, not the capacitors.

Reads only the run directory: cap.csv, step.csv, traces.npz. Nothing touches hardware.
"""
from __future__ import annotations

import argparse
import csv
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OVERFLOW = 1e36          # the 2600's over-range sentinel, 9.91e37. Not a number.
DPI = 300
QUADRANTS = ("TL", "TR", "BL", "BR")
# Two arrays, two colours, everywhere. Index 0 is the first run on the command line.
COLOURS = ("#1f77b4", "#d62728", "#2ca02c", "#9467bd")


# --------------------------------------------------------------------------- loading

def _num(s):
    try:
        x = float(s)
    except (TypeError, ValueError):
        return float("nan")
    return float("nan") if abs(x) > OVERFLOW else x


def _quadrant_of(x, y):
    """Which 256x256 quadrant a logical pixel sits in. Mirrors characterize_capacitance."""
    return QUADRANTS[(1 if x >= 256 else 0) + 2 * (1 if y >= 256 else 0)]


# Columns this script reads that older runs do not have, and what to assume instead.
# A run from before the two-stage split has no `stage`, and everything in it was what
# is now called a fast capture; one from before the paired leakage test has no
# `leakage_suspect`, and nothing in it was ever flagged. Backfilling beats crashing,
# but the assumption is printed so a missing column is never mistaken for a measurement.
BACKFILL = {"step.csv": {"stage": "fast", "p_t_1pct": "", "p_settled_W": "",
                         "channels": "2"},
            "cap.csv": {"leakage_suspect": "False", "lit_blank_ratio": ""}}


def _rows(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as f:
        out = list(csv.DictReader(f))
    fill = BACKFILL.get(path.name, {})
    missing = sorted(k for k in fill if out and k not in out[0])
    for r in out:
        for k in missing:
            r[k] = fill[k]
        # --pixels records quadrant="given", so derive it rather than plotting a
        # single bar labelled "given" for four different quadrants.
        if r.get("quadrant") in (None, "", "given"):
            r["quadrant"] = _quadrant_of(_num(r.get("pixel_x")), _num(r.get("pixel_y")))
    if missing:
        print(f"  ! {path.parent.name}/{path.name} predates "
              f"{', '.join(missing)} -- assuming "
              + ", ".join(f"{k}={fill[k]!r}" for k in missing), file=sys.stderr)
    return out


class Run:
    """One characterisation run directory."""

    def __init__(self, path: Path):
        self.path = path
        self.cap = _rows(path / "cap.csv")
        self.step = _rows(path / "step.csv")
        self.npz = (np.load(path / "traces.npz") if (path / "traces.npz").exists()
                    else None)
        # The run directory is named <stamp>_<sample>_CAPACITANCE. Anything written
        # before --sample-name became mandatory has no trustworthy name.
        m = re.match(r"\d{6}-\d{4}_(.+)_CAPACITANCE$", path.name)
        self.name = m.group(1) if m else path.name
        self.stamp = path.name[:11]
        self._st = None
        self._ct = None

    def __repr__(self):
        return f"<{self.name} {len(self.cap)} cap, {len(self.step)} step>"

    @property
    def keys(self) -> list[str]:
        return list(self.npz.files) if self.npz is not None else []

    def traces(self, pattern: str) -> list[tuple[str, np.ndarray]]:
        """Trace arrays in run order. The index prefix IS the order they were taken.

        Matched by regex, not by a string prefix: the I(t) keys are NNN_step_... with
        no letter prefix, and selecting them with startswith("0") worked only while a
        run had fewer than 100 traces. The real comparison run takes 26 captures on
        each of 8 pixels, so keys 100 upwards would have been dropped in silence.
        """
        rx = re.compile(pattern)
        return [(k, self.npz[k]) for k in sorted(self.keys) if rx.match(k)]

    def step_traces(self) -> list[tuple[dict, np.ndarray, np.ndarray | None]]:
        """Pair every step.csv row with its I(t) and P(t), by order within a tag group.

        Trace tags carry no pixel identity -- `step_long_hc1_rNone_f-1.000_v-0.800` is
        the same string for all four pixels -- so the n-th trace of a tag belongs to
        the n-th row of that tag. Grouping first means one missing trace (keep_trace
        skips an empty capture) cannot shift every later pairing onto the wrong pixel.
        """
        if self._st is not None:
            return self._st
        if self.npz is None:
            # A run in progress flushes its CSVs per pixel; before this script's
            # companion change it only wrote traces.npz at the very end. Say that
            # once, rather than one mismatch warning for each of 26 tag groups.
            print(f"  ! {self.name}: no traces.npz yet -- CSV figures only",
                  file=sys.stderr)
            self._st = []
            return self._st
        itr, ptr = defaultdict(list), defaultdict(list)
        for k, arr in self.traces(r"\d{3}_step_"):        # NNN_step_..._<stride> : t, i_a, i_b
            itr[re.sub(r"^\d{3}_", "", re.sub(r"_\d+$", "", k))].append(arr)
        for k, arr in self.traces(r"pm_\d{3}_"):      # pm_NNN_step_... : t, P
            ptr[re.sub(r"^pm_\d{3}_", "", k)].append(arr)
        grouped = defaultdict(list)
        for r in self.step:
            rng = r["range_i"]
            grouped[f"step_{r['stage']}_hc{int(r['highc'] == 'True')}"
                    f"_r{'None' if rng == 'limit' else rng}"
                    f"_f{_num(r['v_from']):+.3f}_v{_num(r['v_to']):+.3f}"].append(r)
        out = []
        for tag, rs in grouped.items():
            its, pts = itr.get(tag, []), ptr.get(tag, [])
            if len(its) != len(rs):
                print(f"  ! {self.name}: {len(rs)} rows but {len(its)} traces for "
                      f"{tag} -- skipping its traces", file=sys.stderr)
                continue
            for i, r in enumerate(rs):
                out.append((r, its[i], pts[i] if i < len(pts) else None))
        self._st = out
        return out

    def cap_traces(self) -> list[tuple[dict, np.ndarray]]:
        """Pair every cap.csv row with its V(t) ramp, the same way."""
        if self._ct is not None:
            return self._ct
        if self.npz is None:
            self._ct = []
            return self._ct
        vtr = defaultdict(list)
        for k, arr in self.traces(r"v_\d{3}_"):
            vtr[re.sub(r"^v_\d{3}_", "", k)].append(arr)
        grouped = defaultdict(list)
        for r in self.cap:
            grouped[f"cap_lit{int(r['lit'] == 'True')}"
                    f"_i{_num(r['i_set']):.0e}"].append(r)
        out = []
        for tag, rs in grouped.items():
            vs = vtr.get(tag, [])
            if len(vs) != len(rs):
                print(f"  ! {self.name}: {len(rs)} rows but {len(vs)} ramps for {tag}",
                      file=sys.stderr)
                continue
            out.extend(zip(rs, vs))
        self._ct = out
        return out


def clean(t, y):
    """Drop over-range samples. Left in, 9.91e37 poisons every median and axis limit."""
    t, y = np.asarray(t, float), np.asarray(y, float)
    ok = np.isfinite(t) & np.isfinite(y) & (np.abs(y) < OVERFLOW)
    return t[ok], y[ok]


def log_axes(ax, x=True, y=True):
    """Log-scale only the axes that actually have positive data to show.

    A panel can legitimately be empty -- "High-C off, long" has no data in a run taken
    before --step-long-ranges existed -- and a log axis over nothing raises "Data has
    no positive values" out of tight_layout, which loses the WHOLE figure rather than
    one panel. So check first, and label the empty panel instead of crashing on it.
    """
    xs = np.concatenate([np.asarray(l.get_xdata(), float).ravel()
                         for l in ax.get_lines()] or [np.array([])])
    ys = np.concatenate([np.asarray(l.get_ydata(), float).ravel()
                         for l in ax.get_lines()] or [np.array([])])
    has_x = xs.size and np.any(np.isfinite(xs) & (xs > 0))
    has_y = ys.size and np.any(np.isfinite(ys) & (ys > 0))
    if x and has_x:
        ax.set_xscale("log")
    if y and has_y:
        ax.set_yscale("log")
    if not (has_x or has_y):
        ax.text(0.5, 0.5, "no data in these runs", ha="center", va="center",
                transform=ax.transAxes, fontsize=9, color="0.5")
    return bool(has_x or has_y)


# ------------------------------------------------------------------------- power law

def noise_sigma(y):
    """Per-sample noise from consecutive differences, immune to the trend underneath.

    A std over the record measures the relaxation, not the noise. Successive samples
    differ by the noise plus a negligible slice of a slow decay, so the robust scale
    of diff(y) over sqrt(2) is the per-sample sigma.
    """
    d = np.diff(np.asarray(y, float))
    d = d[np.isfinite(d)]
    if d.size < 8:
        return 0.0
    return float(1.4826 * np.median(np.abs(d - np.median(d))) / np.sqrt(2.0))


def fit_power_law(t, i, t_min=1.0, t_max=None, nbin=120, min_bins=20, snr=3.0):
    """Fit |I(t) - I_inf| = A * t**-alpha with I_inf a FREE parameter.

    I_inf and alpha are strongly correlated, so taking the tail median as I_inf -- the
    `settled` column -- biases alpha. In the Array42 reference record `t_01pct` is NaN:
    the trace was still drifting at 370 s, so `settled` still contains relaxation.

    No scipy (it is an optional extra here), and none needed: for a FIXED I_inf the fit
    is linear in log-log, so scan I_inf on a grid and keep the value whose log-log fit
    is most linear. Returns (alpha, I_inf, r2, n_bins_used).

    TWO THINGS THAT MADE AN EARLIER VERSION OF THIS WRONG, both caught by fitting a
    synthetic t^-1.2 whose answer was known:

      * Fitting the raw samples. They are uniform in t, so a 411 s record puts ~95 % of
        its points in the last decade, where the excess is smallest. An unweighted
        log-log fit is then dominated by the flat tail and drags alpha toward 0: with
        NO noise at all, a true 1.6 came back as 0.584. Fitting the log-binned trace
        gives every decade of time equal weight, which is what a power law needs.
      * Keeping bins whose excess had sunk into the noise. Those contribute pure
        scatter near zero, which again flattens the slope. Bins are kept only while
        the binned excess clears `snr` sigma, sigma shrinking as sqrt(n) per bin.

    `t_min` skips the charge-limited edge. At a 10 uA limit a 0.2 V step across 47 uF
    takes 0.94 s pinned at the limit, and those samples are clipped, not relaxing.
    """
    t, i = clean(t, i)
    win = (t >= t_min) & (t <= (t_max if t_max else np.inf)) & (t > 0)
    t, i = t[win], np.abs(i[win])
    nan4 = (float("nan"), float("nan"), float("nan"), 0)
    if t.size < 200:
        return nan4
    sigma = noise_sigma(i)
    # log bins: equal weight per decade of time, and each bin's median averages the
    # noise down by sqrt(count).
    edges = np.geomspace(t.min(), t.max(), nbin + 1)
    idx = np.clip(np.digitize(t, edges) - 1, 0, nbin - 1)
    tb, ib, sb = [], [], []
    for b in range(nbin):
        sel = idx == b
        n = int(np.count_nonzero(sel))
        if n:
            tb.append(float(np.exp(np.mean(np.log(t[sel])))))
            ib.append(float(np.median(i[sel])))
            sb.append(sigma / np.sqrt(n) if sigma else 0.0)
    tb, ib, sb = np.array(tb), np.array(ib), np.array(sb)
    if tb.size < min_bins:
        return nan4
    def scan(cands):
        out = (float("nan"), float("nan"), -np.inf, 0)
        for cand in cands:
            # abs(), because the approach can come from EITHER side. An earlier
            # version used ib - cand with cand below the data, which only ever
            # described a decay from above -- and the Array42 record does the
            # opposite: 312 -> 419 -> 474 -> 481 nA, rising to its asymptote. The
            # fit then returned alpha = -0.04 at r2 = 0.60, a straight line through
            # the wrong model. I_inf simply sits ABOVE the data for a rise.
            d = np.abs(ib - cand)
            keep = d > snr * sb if sigma else d > 0
            if np.count_nonzero(keep) < min_bins:
                continue
            x, y = np.log10(tb[keep]), np.log10(d[keep])
            var = float(np.var(y))
            if var <= 0:
                continue
            slope, const = np.polyfit(x, y, 1)
            r2 = 1.0 - float(np.var(y - (slope * x + const))) / var
            if r2 > out[2]:
                out = (-float(slope), float(cand), r2, int(np.count_nonzero(keep)))
        return out

    # Candidates on BOTH sides of the data: below it describes a decay toward I_inf,
    # above it an approach from below. Array42 does the second. The band inside the
    # data is skipped -- there d crosses zero and log10 of it is meaningless.
    lo, hi = float(ib.min()), float(ib.max())
    span = max(hi - lo, 1e-15)
    below = np.linspace(max(0.0, lo - 4 * span), lo, 400, endpoint=False)
    above = np.linspace(hi + span * 1e-6, hi + 4 * span, 400)
    # Two passes. A single grid has a step of ~1 nA here, and an I_inf wrong by 1 nA
    # leaves a constant offset in the tail that flattens the fit: a true 1.6 came back
    # as 1.12 on synthetic data purely from grid resolution.
    best = scan(np.concatenate([below, above]))
    if np.isfinite(best[1]):
        step = 4 * span / 400.0
        best = max(best, scan(np.linspace(best[1] - 2 * step, best[1] + 2 * step,
                                          400)), key=lambda b: b[2])
    return best if np.isfinite(best[0]) else nan4


def log_bin(t, y, nbin=140):
    """Geometric bins, so a log-log plot of 200k samples is still a readable line."""
    t, y = clean(t, y)
    ok = t > 0
    t, y = t[ok], y[ok]
    if t.size < 2:
        return t, y
    edges = np.geomspace(t.min(), t.max(), nbin + 1)
    idx = np.clip(np.digitize(t, edges) - 1, 0, nbin - 1)
    tb = np.full(nbin, np.nan)
    yb = np.full(nbin, np.nan)
    for b in range(nbin):
        sel = idx == b
        if np.any(sel):
            tb[b] = t[sel].mean()
            yb[b] = np.median(y[sel])
    ok = np.isfinite(tb) & np.isfinite(yb)
    return tb[ok], yb[ok]


# --------------------------------------------------------------------------- figures

def _q_axes(n, w=3.4, h=3.2, **kw):
    """One panel per quadrant, in a row. Returns (fig, [ax...])."""
    fig, axes = plt.subplots(1, max(1, n), figsize=(w * max(1, n), h),
                             squeeze=False, **kw)
    return fig, list(axes[0])


def _quadrants_in(runs):
    qs = {r["quadrant"] for run in runs for r in run.step}
    return [q for q in QUADRANTS if q in qs] or ["?"]


def fig_capacitance(runs, out: Path):
    """C per quadrant, every setpoint shown, lit against blanked.

    Bars hid the whole story. What matters is that the usable setpoints AGREE -- with
    each other, across quadrants, and between lit and blanked -- while the smallest one
    does not. On Array42 the 1 uA setpoint reads 186-308 uF blanked against 42-46 uF
    lit, a 4.4-7.0x disagreement, because at -0.4 V the diode's own leakage takes a
    large share of a 1 uA ramp. The capacitor cannot care whether a sub-turn-on pixel
    is addressed, so that disagreement is the test, and BOTH values fail it.
    """
    fig, ax = plt.subplots(figsize=(8.6, 4.6))
    summary, xs = {}, []
    qs = _quadrants_in(runs)
    for ri, run in enumerate(runs):
        good = []
        for r in run.cap:
            q = r["quadrant"]
            if q not in qs:
                continue
            c, lit = _num(r["capacitance_F"]) * 1e6, r["lit"] == "True"
            bad = r["leakage_suspect"] == "True"
            if not np.isfinite(c):
                continue
            x = qs.index(q) + (ri - (len(runs) - 1) / 2) * 0.3
            xs.append(x)
            ax.plot(x, c, marker="o" if lit else "s", ms=7 if lit else 5,
                    mfc=COLOURS[ri] if not bad else "none", mec=COLOURS[ri],
                    mew=1.4, ls="none", alpha=0.9)
            if not bad and lit:
                good.append(c)
        if good:
            summary[run.name] = (float(np.median(good)), float(np.min(good)),
                                 float(np.max(good)))
            ax.axhline(np.median(good), color=COLOURS[ri], ls="--", lw=0.9, alpha=0.6)
        ax.plot([], [], "o", color=COLOURS[ri], ls="none", label=f"{run.name} lit")
        ax.plot([], [], "s", color=COLOURS[ri], ls="none", label=f"{run.name} blanked")
    ax.plot([], [], "o", mfc="none", mec="0.3", ls="none",
            label="hollow = lit/blanked disagreed, NOT usable")
    ax.set_yscale("log")
    ax.set_xticks(range(len(qs)), qs)
    ax.set_xlabel("quadrant")
    ax.set_ylabel("fitted capacitance  [uF]")
    ax.set_title("Carrier capacitance: every setpoint, lit and blanked\n"
                 "the usable ones agree; the smallest is leakage-dominated and both "
                 "of its values are wrong", fontsize=10)
    ax.legend(fontsize=7.5, loc="best")
    ax.grid(axis="y", which="both", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out / "capacitance.png", dpi=DPI)
    plt.close(fig)
    return summary


def fig_cap_ramps(runs, out: Path):
    """The raw V(t) behind each C. A current source charges the node: dV/dt = I/C.

    Lit solid, blanked dashed. Where the two separate, the diode is taking part of the
    ramp current and the slope under-reads.
    """
    sets = sorted({_num(r["i_set"]) for run in runs for r in run.cap})
    sets = [s for s in sets if np.isfinite(s)]
    if not sets:
        return
    fig, axes = plt.subplots(1, len(sets), figsize=(3.8 * len(sets), 3.8),
                             squeeze=False, sharey=True)
    for si, iset in enumerate(sets):
        ax = axes[0][si]
        for ri, run in enumerate(runs):
            for r, tv in run.cap_traces():
                if abs(_num(r["i_set"]) - iset) > 1e-15:
                    continue
                t, v = clean(tv[0], tv[1])
                lit = r["lit"] == "True"
                ax.plot(t, v, color=COLOURS[ri], lw=1.0 if lit else 0.9,
                        ls="-" if lit else "--", alpha=0.8,
                        label=f"{run.name} {'lit' if lit else 'blanked'}")
        flagged = any(r["leakage_suspect"] == "True" for run in runs for r in run.cap
                      if abs(_num(r["i_set"]) - iset) < 1e-15)
        ax.set_title(f"I = {iset * 1e9:.0f} nA"
                     + ("\nLEAKAGE-DOMINATED, rejected" if flagged else ""),
                     fontsize=9, color="#b22222" if flagged else "black")
        ax.set_xlabel("t  [s]")
        ax.grid(alpha=0.3)
        if si == 0:
            ax.set_ylabel("V on ch B  [V]")
        h, l = ax.get_legend_handles_labels()
        seen = dict(zip(l, h))
        ax.legend(seen.values(), seen.keys(), fontsize=7)
    fig.suptitle("Capacitance ramps. Steeper = smaller C. Lit and blanked must "
                 "overlap, or the diode is stealing ramp current.", fontsize=10)
    fig.tight_layout()
    fig.savefig(out / "cap_ramps.png", dpi=DPI)
    plt.close(fig)


def fig_charge_edge(runs, out: Path, window_ms=120.0):
    """The fast capture's reason to exist: the charge edge, High-C on, at the faintest
    bias. Linear in milliseconds.

    This was a log axis before, which spread the first few milliseconds over most of
    the panel and left the reader deciphering "10^-1 s". The edge is a few ms wide, so
    milliseconds is what the axis should say.
    """
    biases = sorted({_num(r["v_to"]) for run in runs for r in run.step
                     if r["stage"] == "fast" and r["highc"] == "True"})
    if not biases:
        return
    v_want = max(biases)                       # the faintest, where the long one runs
    fig, ax = plt.subplots(figsize=(7.6, 4.2))
    for ri, run in enumerate(runs):
        for r, it, _ in run.step_traces():
            if (r["stage"] != "fast" or r["highc"] != "True"
                    or abs(_num(r["v_to"]) - v_want) > 1e-9):
                continue
            t, i = clean(it[0], it[2])
            sel = t * 1e3 <= window_ms
            ax.plot(t[sel] * 1e3, i[sel] * 1e9, color=COLOURS[ri], lw=0.9, alpha=0.75)
        ax.plot([], [], color=COLOURS[ri], lw=1.4, label=run.name)
    ax.set_xlabel("t after the step  [ms]")
    ax.set_ylabel("I on ch B  [nA]")
    ax.set_title(f"Charge edge at V = {v_want:+.2f} V, High-C on (one line per pixel)\n"
                 "0.2 V across ~47 uF at a 1 mA limit is ~9 ms", fontsize=10)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "charge_edge.png", dpi=DPI)
    plt.close(fig)


def fig_settling(runs, out: Path):
    """THE figure. Current and light on one time axis, each as a fraction of its final.

    Measured on Array42: the light is flat by ~2 s while the current is still climbing
    at 400 s and never reaches 0.1 % inside the record. So the slow electrical tail
    carries NO light -- it is parasitic, not LED current. What the sweep's dwell waits
    for is the denominator of the EQE, not the numerator.

    Normalising both to their own final value is the only way to put a nanoamp and a
    picowatt on one axis, and it is also the honest comparison: the question is when
    each one STOPS CHANGING, not what either is worth.
    """
    qs = _quadrants_in(runs)
    fig, axes = _q_axes(len(qs), w=3.6, h=3.6, sharey=True)
    rows = {}
    for qi, q in enumerate(qs):
        ax = axes[qi]
        for ri, run in enumerate(runs):
            for r, it, pt in run.step_traces():
                if (r["stage"] != "long" or r["highc"] != "True"
                        or r["quadrant"] != q):
                    continue
                t, i = clean(it[0], it[2])
                if t.size < 50:
                    continue
                # Log-bin BOTH for display. The optical record is 200k samples at
                # 500 Hz: drawn raw it is a solid block of ink that hides the current
                # curve underneath it. The median per bin is the same trace, readable.
                fin = np.median(np.abs(i[int(0.8 * i.size):]))
                tb, yb = log_bin(t, np.abs(i) / fin, nbin=220)
                ax.plot(tb, yb, color=COLOURS[ri], lw=1.3,
                        label=f"{run.name} current")
                if pt is not None:
                    tp, p = clean(pt[0], pt[1])
                    if tp.size > 50:
                        pf = np.median(np.abs(p[int(0.8 * p.size):]))
                        tpb, pb = log_bin(tp, np.abs(p) / pf, nbin=220)
                        ax.plot(tpb, pb, color="#ff7f0e", lw=1.3, ls="--",
                                alpha=0.85, label=f"{run.name} light")
                e, o = _num(r["t_1pct"]), _num(r["p_t_1pct"])
                rows[(run.name, q)] = (e, o, _num(r["t_01pct"]))
                for val, col in ((o, "#ff7f0e"), (e, COLOURS[ri])):
                    if np.isfinite(val) and val > 0:
                        ax.axvline(val, color=col, ls=":", lw=1.1, alpha=0.8)
        ax.axhspan(0.99, 1.01, color="0.85", zorder=0)
        ax.set_xscale("log")
        ax.set_ylim(0.0, 1.35)
        ax.set_xlabel("t after the step  [s]")
        ax.set_title(q, fontsize=10)
        ax.grid(which="both", alpha=0.25)
        if qi == 0:
            ax.set_ylabel("fraction of its own final value")
            h, l = ax.get_legend_handles_labels()
            seen = dict(zip(l, h))
            ax.legend(seen.values(), seen.keys(), fontsize=7, loc="lower right")
    fig.suptitle("The light settles. The current does not.  solid = I(t), dashed = "
                 "P(t), both over their own final value\n"
                 "grey band = within 1 %, dotted verticals = when each last left it",
                 fontsize=10)
    fig.tight_layout()
    fig.savefig(out / "settling.png", dpi=DPI)
    plt.close(fig)
    return rows


def fig_power_law(runs, out: Path, t_min=1.0):
    """The relaxation on log-log, where a power law is a straight line.

    Only High-C. Without it the reading is not a current at all (see highc_needed.png),
    so fitting an exponent to it would be fitting the meter.

    I_inf is FITTED and may sit on either side of the data: Array42 approaches from
    BELOW, 312 -> 419 -> 474 -> 481 nA, so the asymptote is above every sample. An
    earlier version searched below only and reported alpha = -0.04 at r2 = 0.60.
    """
    fig, ax = plt.subplots(figsize=(7.8, 5.0))
    fits = {}
    for ri, run in enumerate(runs):
        for r, it, _ in run.step_traces():
            if r["stage"] != "long" or r["highc"] != "True":
                continue
            t, i = clean(it[0], it[2])
            alpha, iinf, r2, nb = fit_power_law(t, i, t_min=t_min)
            q = r["quadrant"]
            fits[(run.name, q)] = (alpha, iinf, r2, nb, int(t.size))
            if not np.isfinite(alpha):
                continue
            # Colour is the QUADRANT and dash is the RUN. Four lines of one colour
            # told the reader nothing about which was which, and with two arrays the
            # pairs that matter are same-colour: the same quadrant before and after.
            col = COLOURS[QUADRANTS.index(q) % len(COLOURS)] if q in QUADRANTS \
                else COLOURS[0]
            dash = [(0, ()), (0, (6, 2)), (0, (1, 1.5))][ri % 3]
            tb, yb = log_bin(t, np.abs(np.abs(i) - iinf))
            keep = (tb >= t_min) & (yb > 0)
            ax.plot(tb[keep], yb[keep] * 1e9, color=col, lw=1.3, alpha=0.9, ls=dash,
                    label=f"{run.name} {q}: a={alpha:.2f}, r2={r2:.3f}")
            tf = np.geomspace(tb[keep][0], tb[keep][-1], 40)
            ax.plot(tf, yb[keep][0] * (tf / tf[0]) ** -alpha * 1e9,
                    color=col, ls=":", lw=1.0, alpha=0.5)
    log_axes(ax)
    ax.set_xlabel("t after the step  [s]")
    ax.set_ylabel("distance from the fitted asymptote  |I - I_inf|  [nA]")
    ax.set_title("Relaxation is a power law, not an RC exponential\n"
                 "straight on log-log = t^-a. Dotted = the fit. High-C only.",
                 fontsize=10)
    ax.legend(fontsize=7.5)
    ax.grid(which="both", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out / "power_law.png", dpi=DPI)
    plt.close(fig)
    return fits


def fig_highc_needed(runs, out: Path):
    """Can High-C be dropped? Two independent answers, side by side.

    LEFT: at a fixed bias the non-High-C reading tracks the selected RANGE, sitting at
    about half of whatever was chosen, across five decades. A line of slope 1 is an
    artifact; flat would be a device.

    RIGHT: the decisive one, because the power meter is an independent witness. The
    same pixel at the same bias, measured long with High-C and without: on Array42 the
    CURRENT differs by 13x (424-488 nA against 35-37 nA) while the LIGHT agrees to
    about 1 %. The device was in the same state both times, so the disagreement is the
    meter, not the pixel.
    """
    fig, axes = plt.subplots(1, 2, figsize=(12.4, 4.8), squeeze=False)
    ax = axes[0][0]
    verdict = {}
    for ri, run in enumerate(runs):
        all_x = []
        biases = sorted({_num(r["v_to"]) for r in run.step}, reverse=True)
        for bi, v in enumerate(biases):
            pts = sorted((_num(r["range_i"]), abs(_num(r["settled"])))
                         for r in run.step
                         if (r["stage"] == "fast" and r["highc"] == "False"
                             and abs(_num(r["v_to"]) - v) < 1e-9
                             and np.isfinite(_num(r["range_i"]))
                             and np.isfinite(_num(r["settled"]))))
            if len(pts) < 2:
                continue
            x = np.array([p[0] for p in pts])
            y = np.array([p[1] for p in pts])
            all_x.append(x)
            ax.plot(x, y, "o-", lw=1.0, ms=4, color=COLOURS[bi % len(COLOURS)],
                    alpha=0.9, label=f"{run.name} V={v:+.2f}" if ri == 0 or True else None)
            verdict.setdefault(run.name, []).append(
                (v, float(np.median(y / x)), float(np.ptp(np.log10(y)))))
        drawn = np.concatenate(all_x) if all_x else np.array([])
        if drawn.size:
            xs = np.geomspace(drawn.min(), drawn.max())
            ax.plot(xs, 0.5 * xs, color="0.55", ls=":", lw=1.4,
                    label="50 % of range = pure artifact")
    log_axes(ax)
    ax.set_xlabel("selected measure range  [A]")
    ax.set_ylabel("|settled| reading  [A]")
    ax.set_title("High-C OFF: the reading follows the RANGE\n"
                 "on the guide line = artifact, flat = a real current", fontsize=9.5)
    ax.grid(which="both", alpha=0.25)
    ax.legend(fontsize=7)

    ax2 = axes[0][1]
    qs = _quadrants_in(runs)
    w, labels = 0.35, []
    for ri, run in enumerate(runs):
        cur, pwr = {}, {}
        for r in run.step:
            if r["stage"] != "long":
                continue
            key = (r["quadrant"], r["highc"] == "True")
            cur[key] = abs(_num(r["settled"])) * 1e9
            pwr[key] = abs(_num(r["power_W"])) * 1e12
        x = np.arange(len(qs))
        on = [cur.get((q, True), np.nan) for q in qs]
        off = [cur.get((q, False), np.nan) for q in qs]
        ax2.bar(x - w / 2, on, w, color=COLOURS[ri], label=f"{run.name} I, High-C on")
        ax2.bar(x + w / 2, off, w, color=COLOURS[ri], alpha=0.4, hatch="//",
                label=f"{run.name} I, High-C off")
        for xi, q in enumerate(qs):
            a, b = cur.get((q, True)), cur.get((q, False))
            pa, pb = pwr.get((q, True)), pwr.get((q, False))
            if a and b and np.isfinite(a) and np.isfinite(b) and b:
                ax2.annotate(f"{a / b:.0f}x", (xi, max(a, b) * 1.25), ha="center",
                             fontsize=9, color="#b22222")
            if pa and pb and np.isfinite(pa) and np.isfinite(pb) and pb:
                ax2.annotate(f"light {pa:.0f} vs {pb:.0f} pW\n({100 * (pa / pb - 1):+.1f} %)",
                             (xi, max(a or 0, b or 0) * 0.45), ha="center", fontsize=7,
                             color="0.25")
        labels = qs
    # Headroom for the ratio labels, which sit above the taller bar and were being
    # clipped off the top of the axis entirely.
    top = max([v for v in ax2.get_ylim()] + [0])
    ax2.set_ylim(0, top * 1.35)
    ax2.set_xticks(np.arange(len(labels)), labels)
    ax2.set_xlabel("quadrant")
    ax2.set_ylabel("long-capture settled current  [nA]")
    ax2.set_title("Same pixel, same bias, same light -- 13x different current\n"
                  "the power meter is the witness: the device did not change",
                  fontsize=9.5)
    ax2.grid(axis="y", alpha=0.3)
    ax2.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out / "highc_needed.png", dpi=DPI)
    plt.close(fig)
    return verdict


def fig_bias_response(runs, out: Path):
    """Current and light against bias, High-C on. They do not fall together.

    On Array42 between -0.95 and -0.80 V the current falls 5.4x while the light falls
    22x. So as the bias drops, a growing share of the current stops producing photons
    -- which is the same parasitic path the settling figure sees, measured a second way.
    """
    fig, axes = plt.subplots(1, 2, figsize=(10.6, 4.3), squeeze=False, sharex=True)
    ratios = {}
    for ri, run in enumerate(runs):
        per_q = defaultdict(list)
        for r in run.step:
            if r["stage"] != "fast" or r["highc"] != "True":
                continue
            per_q[r["quadrant"]].append((_num(r["v_to"]), abs(_num(r["settled"])),
                                         abs(_num(r["power_W"]))))
        for q, pts in sorted(per_q.items()):
            pts.sort()
            v = np.array([p[0] for p in pts])
            i = np.array([p[1] for p in pts]) * 1e9
            p = np.array([p[2] for p in pts]) * 1e12
            axes[0][0].plot(v, i, "o-", ms=4, lw=1.0, color=COLOURS[ri], alpha=0.8)
            axes[0][1].plot(v, p, "o-", ms=4, lw=1.0, color=COLOURS[ri], alpha=0.8)
            if i.size > 1 and i[0] > 0 and p[0] > 0:
                ratios.setdefault(run.name, []).append(
                    (q, float(i[0] / i[-1]), float(p[0] / p[-1])))
        axes[0][0].plot([], [], color=COLOURS[ri], lw=1.4, label=run.name)
    for ax, lab, ttl in ((axes[0][0], "I on ch B  [nA]", "current"),
                         (axes[0][1], "optical power  [pW]", "light")):
        ax.set_yscale("log")
        ax.set_xlabel("bias  [V]")
        ax.set_ylabel(lab)
        ax.set_title(ttl, fontsize=10)
        ax.grid(which="both", alpha=0.25)
    axes[0][0].legend(fontsize=8)
    fig.suptitle("Current and light against bias, High-C on. They do not fall together:"
                 " the gap is current that makes no photons.", fontsize=10)
    fig.tight_layout()
    fig.savefig(out / "bias_response.png", dpi=DPI)
    plt.close(fig)
    return ratios


# ------------------------------------------------------------------------------ main

def build_parser():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("runs", nargs="+", type=Path,
                    help="run directories written by characterize_capacitance.py")
    ap.add_argument("-o", "--out", type=Path, default=None,
                    help="figure directory (default: analysis/<stamp>_<name>[_vs_...])")
    ap.add_argument("--fit-from", type=float, default=1.0, metavar="S",
                    help="skip this many seconds before fitting the power law. The "
                         "charge-limited edge is clipped, not relaxing: a 0.2 V step "
                         "across 47 uF at a 10 uA limit takes 0.94 s (default 1.0)")
    return ap


def main(argv=None):
    a = build_parser().parse_args(argv)
    runs = []
    for p in a.runs:
        if not p.exists():
            raise SystemExit(f"no such run directory: {p}")
        r = Run(p)
        if not r.cap and not r.step:
            raise SystemExit(f"{p} has neither cap.csv nor step.csv")
        runs.append(r)
    out = a.out or Path("analysis") / "_vs_".join(f"{r.stamp}_{r.name}" for r in runs)
    # Two runs of the same sample -- a repeat, or a baseline against itself -- would
    # otherwise draw two bars with one legend entry and look like a single run. After
    # `out`, so the directory name does not grow the stamp twice.
    if len({r.name for r in runs}) < len(runs):
        for r in runs:
            r.name = f"{r.name} {r.stamp}"
    out.mkdir(parents=True, exist_ok=True)

    print(f"\n{'=' * 78}\n{len(runs)} run(s) -> {out}\n{'=' * 78}")
    for r in runs:
        print(f"  {r.name:22} {r.stamp}  {len(r.cap):3} cap rows, "
              f"{len(r.step):4} step rows, {len(r.keys):4} traces")
    if len(runs) > 1:
        print("\n  Board-level quantities (C, settling, whether High-C is needed) "
              "compare directly.\n  Array properties (leakage, EQE, yield) do NOT: "
              "these are different arrays.")

    caps = fig_capacitance(runs, out)
    fig_cap_ramps(runs, out)
    fig_charge_edge(runs, out)
    settle = fig_settling(runs, out)
    fits = fig_power_law(runs, out, t_min=a.fit_from)
    verdict = fig_highc_needed(runs, out)
    ratios = fig_bias_response(runs, out)

    print(f"\n-- capacitance, lit, leakage-free setpoints {'-' * 34}")
    for name, (med, lo, hi) in caps.items():
        print(f"  {name:22} median {med:7.2f} uF    per-quadrant {lo:6.2f} - {hi:6.2f}")

    print(f"\n-- power law |I-I_inf| ~ t^-alpha, fitted for t > {a.fit_from:g} s {'-' * 16}")
    if fits:
        for (name, where), (alpha, iinf, r2, n, nsamp) in sorted(fits.items()):
            if not np.isfinite(alpha):
                print(f"  {name:14} {where:5}  record of {nsamp} samples is too short "
                      f"to fit -- raise --step-long-samples")
                continue
            # Synthetic records with a known exponent all fit at r2 > 0.99, so
            # anything below that is a record without a resolvable power law in it,
            # not a measured exponent. Quoting it would be quoting noise.
            # A clean synthetic power law fits at r2 > 0.998; the real High-C
            # records sit at 0.97-0.99, because a device is not a pure power law.
            # Below 0.95 there is no resolvable power law to quote.
            flag = "" if r2 > 0.95 else "   << WEAK FIT, do not quote this exponent"
            print(f"  {name:14} {where:5}  alpha={alpha:5.2f}  "
                  f"I_inf={iinf * 1e9:8.1f} nA  r2={r2:5.3f}  bins={n}{flag}")
    elif any(r["stage"] == "long" for run in runs for r in run.step):
        # The rows are there, so the captures happened -- what is missing is the raw
        # record the exponent is fitted from. A run in progress is the usual reason.
        print("  long captures exist in step.csv but their traces do not. The "
              "exponent is")
        print("  fitted from traces.npz, which a run only has once it has flushed it.")
    else:
        print("  no long captures found. Without --highc-both there is one state only,")
        print("  and before --step-long-ranges the non-High-C long capture never ran.")

    print(f"\n-- does a non-High-C reading track bias or range? {'-' * 28}")
    for name, items in verdict.items():
        for v, frac, decades in sorted(items, reverse=True):
            tag = ("ARTIFACT, tracks the range" if decades > 1.0
                   else "tracks bias -- High-C may be droppable")
            print(f"  {name:14} V={v:+.2f}  reading/range {frac:5.2f}  "
                  f"spans {decades:4.2f} decades  -> {tag}")

    print(f"\n-- settling, High-C long: electrical against optical {'-' * 24}")
    for (name, q), (e, o, e01) in sorted(settle.items()):
        ratio = (f"{e / o:6.1f}x" if (np.isfinite(e) and np.isfinite(o) and o > 0)
                 else "   n/a")
        tail = "" if np.isfinite(e01) else "   (never within 0.1 % inside the record)"
        print(f"  {name:14} {q:3}  current {e:8.2f} s   light {o:7.3f} s  "
              f"{ratio}{tail}")

    print(f"\n-- current against light across bias, High-C on {'-' * 29}")
    for name, items in sorted(ratios.items()):
        for q, fi, fp in sorted(items):
            print(f"  {name:14} {q:3}  across the bias range the current falls "
                  f"{fi:5.1f}x, the light {fp:5.1f}x  -> {fp / fi:4.1f}x more of the "
                  f"current makes no photons")

    print(f"\nwrote {len(list(out.glob('*.png')))} figures to {out}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
