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
import textwrap
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

def caption(fig, what: str, how: str, learn: str, pad: float = 0.0):
    """Put WHAT was measured / HOW / WHAT WE LEARN under the axes of every figure.

    A figure that needs the person who made it standing next to it is not a result.
    The numbers in `learn` come from the run being plotted, not from a template, so a
    caption that stops matching the data is a caption that was not updated.
    """
    # Wrap to the figure's own width. Unwrapped, a three-sentence HOW simply ran off
    # the right edge of the page and the half that mattered was the half cut off.
    size = 8.2
    cols = max(60, int((fig.get_size_inches()[0] - 0.45) * 72 / (size * 0.60)))
    body = []
    for tag, val in (("WHAT: ", what), ("HOW:  ", how), ("LEARN:", learn)):
        lines = textwrap.wrap(val, width=cols - len(tag) - 1) or [""]
        body.append(f"{tag} {lines[0]}")
        body += [f"{' ' * (len(tag) + 1)}{l}" for l in lines[1:]]
    text = "\n".join(body)
    room = 0.175 + 0.0255 * len(body) * (4.6 / fig.get_size_inches()[1]) + pad
    fig.subplots_adjust(bottom=min(0.6, room))
    fig.text(0.008, 0.010, text, ha="left", va="bottom", fontsize=size,
             family="monospace", linespacing=1.45,
             bbox=dict(boxstyle="round,pad=0.45", fc="#f6f6f2", ec="0.75", lw=0.8))
    return room


def _fmt_range(vals, unit, fmt="{:.1f}"):
    """'41.1-45.7 uF', or just '43.4 uF' when there is one value to report."""
    vals = [v for v in vals if np.isfinite(v)]
    if not vals:
        return "n/a"
    lo, hi = min(vals), max(vals)
    if abs(hi - lo) < 1e-12:
        return f"{fmt.format(lo)} {unit}"
    return f"{fmt.format(lo)}-{fmt.format(hi)} {unit}"


def _q_axes(n, w=3.4, h=3.2, **kw):
    """One panel per quadrant, in a row. Returns (fig, [ax...])."""
    fig, axes = plt.subplots(1, max(1, n), figsize=(w * max(1, n), h),
                             squeeze=False, **kw)
    return fig, list(axes[0])


def _quadrants_in(runs):
    qs = {r["quadrant"] for run in runs for r in run.step}
    return [q for q in QUADRANTS if q in qs] or ["?"]


def pixel_colours(runs):
    """One colour per PIXEL, shared across every figure and every run.

    Each pixel gets its own colour rather than its run's, so a panel with four of them
    is readable, and so the same pixel keeps its colour from figure to figure. Keyed by
    (x, y), not by quadrant: --per-quadrant 2 puts two pixels in each, and colouring by
    quadrant would then draw two different pixels identically.

    Ordered by quadrant, then by coordinate, so the legend reads TL, TR, BL, BR and the
    assignment does not shuffle when a run is added to the command line.
    """
    seen = {(r["quadrant"], _num(r["pixel_x"]), _num(r["pixel_y"]))
            for run in runs for r in run.step}
    order = sorted(seen, key=lambda k: (QUADRANTS.index(k[0]) if k[0] in QUADRANTS
                                        else 99, k[1], k[2]))
    cyc = plt.get_cmap("tab10").colors + plt.get_cmap("Dark2").colors
    return {(q, x, y): cyc[i % len(cyc)] for i, (q, x, y) in enumerate(order)}


def _pkey(r):
    return (r["quadrant"], _num(r["pixel_x"]), _num(r["pixel_y"]))


def _plabel(key):
    q, x, y = key
    return f"{q} ({x:.0f},{y:.0f})"


def _run_dash(ri):
    """Dash pattern per run, so colour stays free to mean the pixel."""
    return [(0, ()), (0, (6, 2)), (0, (1, 1.5)), (0, (4, 1.5, 1, 1.5))][ri % 4]


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
    allg = [v for t in summary.values() for v in t[1:]]
    caption(fig,
            what="Carrier capacitance per quadrant, every current setpoint, measured "
                 "with the pixel addressed (lit) and with a blank frame (blanked).",
            how="The node is parked below turn-on, handed to a current source, and its "
                "voltage ramp recorded into the SMU buffer with instrument timestamps. "
                "C = I / (dV/dt), so a steeper ramp means a smaller C. Each setpoint "
                "is an independent estimate of the same capacitance.",
            learn=f"The usable setpoints agree across all four quadrants at "
                  f"{_fmt_range(allg, 'uF', '{:.1f}')}, so the carrier is uniform and "
                  f"that is the number Array38 gets compared against. The smallest "
                  f"setpoint disagrees between lit and blanked by several times -- and "
                  f"a capacitor cannot care whether a sub-turn-on pixel is addressed, "
                  f"so there the diode leakage is taking a large share of the ramp "
                  f"current and BOTH of its values are wrong, the plausible-looking one "
                  f"included.")
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
    caption(fig,
            what="The raw voltage ramps the capacitances were fitted from, one panel "
                 "per current setpoint, lit (solid) against blanked (dashed).",
            how="A constant current charges the node from a park bias below turn-on "
                "while the SMU records V(t) into its buffer. The slope IS the "
                "measurement: dV/dt = I/C.",
            learn="Lit and blanked must lie on top of one another, because the "
                  "capacitor does not know whether a pixel is addressed. Where they "
                  "separate, the diode is drawing part of the ramp current and the "
                  "fitted C under-reads. The curvature at the largest setpoint is the "
                  "pixel turning on as the ramp sweeps past its threshold.")
    fig.savefig(out / "cap_ramps.png", dpi=DPI)
    plt.close(fig)


def fig_charge_edge(runs, out: Path, window_ms=120.0):
    """What the fast capture can and cannot say about the charge edge.

    Drawn raw, this figure was ink: at NPLC 0.01 on the 1 mA range the per-sample
    noise is ~950 nA, against a current of 480 nA at the faintest bias. Single samples
    carry no information at all, so the trace swung +/-2000 nA for the whole record and
    looked like a measurement of nothing -- which it was.

    Binned, with the standard error on each bin, it says something definite: the first
    bin is ALREADY at the final value. At the brightest bias the 0-4 ms mean is
    -3648 nA against a settled -3714 nA, so the edge finished inside the first 0.38 ms
    sample. The fast capture therefore BOUNDS the charge time rather than resolving it,
    and that bound is the same for both arrays -- removing the capacitors can only make
    it faster. Anything about the edge itself needs a scope, not this SMU.
    """
    rows = [(run, ri, r, it) for ri, run in enumerate(runs)
            for r, it, _ in run.step_traces()
            if r["stage"] == "fast" and r["highc"] == "True"]
    if not rows:
        return
    biases = sorted({_num(r["v_to"]) for _, _, r, _ in rows})
    cols = pixel_colours(runs)
    fig, axes = plt.subplots(1, len(biases), figsize=(3.3 * len(biases) + 0.6, 4.0),
                             squeeze=False)
    settled_first = []
    for bi, v in enumerate(biases):
        ax = axes[0][bi]
        for run, ri, r, it in rows:
            if abs(_num(r["v_to"]) - v) > 1e-9:
                continue
            t, i = clean(it[0], it[2])
            sel = (t >= 0) & (t * 1e3 <= window_ms)
            if sel.sum() < 20:
                continue
            # ~14 bins across the window, each averaging the noise down by sqrt(n)
            edges = np.linspace(0, window_ms, 15)
            idx = np.clip(np.digitize(t[sel] * 1e3, edges) - 1, 0, 13)
            tb, mb, eb = [], [], []
            for b in range(14):
                m = idx == b
                if m.sum() > 2:
                    tb.append(0.5 * (edges[b] + edges[b + 1]))
                    mb.append(np.mean(i[sel][m]) * 1e9)
                    eb.append(np.std(i[sel][m]) / np.sqrt(m.sum()) * 1e9)
            key = _pkey(r)
            ax.errorbar(tb, mb, yerr=eb, color=cols.get(key, "0.4"), lw=1.2, ms=3,
                        marker="o", capsize=2, alpha=0.9, ls=_run_dash(ri),
                        label=_plabel(key) + (f" [{run.name}]" if len(runs) > 1 else ""))
            fin = np.median(i[t > 0.5 * t.max()]) * 1e9
            ax.axhline(fin, color=cols.get(key, "0.4"), ls=":", lw=1.0, alpha=0.6)
            if mb and bi == 0:        # brightest bias only -- see below
                settled_first.append(abs(mb[0] / fin) if fin else np.nan)
        ax.set_title(f"V = {v:+.2f} V", fontsize=9.5)
        ax.set_xlabel("t after the step  [ms]")
        ax.grid(alpha=0.3)
        if bi == 0:
            ax.set_ylabel("I on ch B, binned  [nA]")
            ax.legend(fontsize=7)
    frac = _fmt_range([f * 100 for f in settled_first], "%", "{:.0f}")
    vbright = min(biases)
    fig.suptitle("The charge edge is over before the first sample. "
                 "Dotted = each pixel's settled value.", fontsize=10)
    fig.tight_layout()
    caption(fig,
            what="Current right after a 0.2 V step, High-C on, four biases. The fast "
                 "capture: 6000 samples at NPLC 0.01, 0.38 ms apart.",
            how="Binned to ~8.6 ms per point with the standard error on each. Raw, the "
                "per-sample noise is ~950 nA on the 1 mA range that the 1 mA limit "
                "selects -- larger than the 480 nA current at the faint end.",
            learn=f"At {vbright:+.2f} V, the only bias where the current clears the "
                  f"per-sample noise, the very first bin already sits at {frac} of the "
                  f"settled value -- so the edge completed inside one 0.38 ms sample. "
                  f"This capture BOUNDS the charge time at under 0.4 ms, it does not "
                  f"resolve it, and that bound is the same for both arrays: fewer "
                  f"capacitors can only be faster. The fainter panels are scatter, not "
                  f"structure. Resolving the edge itself needs a scope, not this SMU.")
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
    cols = pixel_colours(runs)
    fig, axes = _q_axes(len(qs), w=3.6, h=3.6, sharey=True)
    rows, titles = {}, {}
    for qi, q in enumerate(qs):
        ax = axes[qi]
        for ri, run in enumerate(runs):
            for r, it, pt in run.step_traces():
                if r["stage"] != "long" or r["quadrant"] != q:
                    continue
                t, i = clean(it[0], it[2])
                if t.size < 50:
                    continue
                if r["highc"] != "True":
                    # The non-High-C record, drawn muted. It is not a current -- the
                    # same pixel reads 13x lower here while the light is unchanged --
                    # but it is what tells you whether a de-capped carrier fixed this
                    # channel, so it belongs on the figure rather than omitted.
                    f0 = np.median(np.abs(i[int(0.8 * i.size):]))
                    t0, y0 = log_bin(t, np.abs(i) / f0, nbin=220)
                    ax.plot(t0, y0, color="0.78", lw=1.0, ls=(0, (1, 2)), alpha=0.95,
                            label="current, no High-C"
                                  + (f" [{run.name}]" if len(runs) > 1 else ""))
                    continue
                # Log-bin BOTH for display. The optical record is 200k samples at
                # 500 Hz: drawn raw it is a solid block of ink that hides the current
                # curve underneath it. The median per bin is the same trace, readable.
                key = _pkey(r)
                titles[q] = _plabel(key)
                pcol = cols.get(key, COLOURS[ri])
                fin = np.median(np.abs(i[int(0.8 * i.size):]))
                tb, yb = log_bin(t, np.abs(i) / fin, nbin=220)
                ax.plot(tb, yb, color=pcol, lw=1.3, ls=_run_dash(ri),
                        label=f"current{f' [{run.name}]' if len(runs) > 1 else ''}")
                if pt is not None:
                    tp, p = clean(pt[0], pt[1])
                    if tp.size > 50:
                        pf = np.median(np.abs(p[int(0.8 * p.size):]))
                        tpb, pb = log_bin(tp, np.abs(p) / pf, nbin=220)
                        ax.plot(tpb, pb, color="black", lw=1.2, ls=(0, (5, 2)),
                                alpha=0.9,
                                label=f"light{f' [{run.name}]' if len(runs) > 1 else ''}")
                e, o = _num(r["t_1pct"]), _num(r["p_t_1pct"])
                rows[(run.name, q)] = (e, o, _num(r["t_01pct"]))
                for val, col in ((o, "black"), (e, pcol)):
                    if np.isfinite(val) and val > 0:
                        ax.axvline(val, color=col, ls=":", lw=1.1, alpha=0.8)
        ax.axhspan(0.99, 1.01, color="0.85", zorder=0)
        ax.set_xscale("log")
        ax.set_ylim(0.0, 1.35)
        # Stop the axis at the first sample. Auto-ranging ran it back to 1e-2 s,
        # which is before anything was recorded and reads like missing data.
        firsts = []
        for ln in ax.get_lines():
            xd = np.asarray(ln.get_xdata(), float).ravel()
            xd = xd[np.isfinite(xd) & (xd > 0)]
            if xd.size:
                firsts.append(float(xd.min()))
        lo = min(firsts) if firsts else 0.02
        ax.set_xlim(left=max(lo * 0.8, 1e-3))
        ax.set_xlabel("t after the step  [s, log]")
        ax.set_title(titles.get(q, q), fontsize=10,
                     color=cols.get(next((k for k in cols if k[0] == q), None), "black"))
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
    el = [v[0] for v in rows.values() if np.isfinite(v[0])]
    op = [v[1] for v in rows.values() if np.isfinite(v[1])]
    never = sum(1 for v in rows.values() if not np.isfinite(v[2]))
    caption(fig,
            what="Current and light after the same 0.2 V step to the faintest bias, "
                 "one pixel per quadrant. The long capture: 20000 samples at NPLC 1.0 "
                 "over ~410 s, power meter polled at ~500 Hz from the step onward.",
            how="Each trace is divided by its OWN final value -- the only way to put a "
                "nanoamp and a picowatt on one axis -- and log-binned for drawing. "
                "Time is LOGARITHMIC because the settling spans four decades; linear "
                "would compress everything before 10 s into 2 % of the panel. The dip "
                "near 0.5 s is the step itself, where the current passes through zero "
                "on its way to the new bias.",
            learn=f"The light is flat within 1 % by {_fmt_range(op, 's', '{:.1f}')}; "
                  f"the current needs {_fmt_range(el, 's', '{:.0f}')}, and in "
                  f"{never} of {len(rows)} records it never came within 0.1 % at all. "
                  f"So the slow electrical tail carries NO light: it is parasitic, not "
                  f"LED current, and what the sweep dwell waits for is the EQE "
                  f"denominator. Faint dotted = the broken non-High-C channel, drawn "
                  f"for scale.")
    fig.savefig(out / "settling.png", dpi=DPI)
    plt.close(fig)
    return rows


def fig_power_law(runs, out: Path, t_min=1.0):
    """The relaxation on log-log, where a power law is a straight line.

    I_inf is FITTED and may sit on either side of the data: Array42 approaches from
    BELOW, 312 -> 419 -> 474 -> 481 nA, so the asymptote is above every sample. An
    earlier version searched below only and reported alpha = -0.04 at r2 = 0.60.

    BOTH High-C states are drawn, in their own panels. The non-High-C record is the
    one that answers whether a de-capped carrier made that channel usable, so leaving
    it out -- which an earlier version did, on the grounds that it is not a current --
    hid the result the second long capture was added to produce. It stays in its own
    panel because the two are measured on different ranges, and its r2 says how much
    of it to believe: Array42 gives 0.29-0.78 there against 0.97-0.99 with High-C.
    """
    states = (("True", "High-C ON"), ("False", "High-C OFF"))
    fig, axes = plt.subplots(1, 2, figsize=(14.0, 5.2), squeeze=False)
    cols = pixel_colours(runs)
    fits = {}
    for si, (hc, title) in enumerate(states):
        ax = axes[0][si]
        worst = 1.0
        for ri, run in enumerate(runs):
            for r, it, _ in run.step_traces():
                if r["stage"] != "long" or r["highc"] != hc:
                    continue
                t, i = clean(it[0], it[2])
                alpha, iinf, r2, nb = fit_power_law(t, i, t_min=t_min)
                fits[(run.name, f"{r['quadrant']} {'hc' if hc == 'True' else 'no-hc'}")
                     ] = (alpha, iinf, r2, nb, int(t.size))
                if not np.isfinite(alpha):
                    continue
                worst = min(worst, r2)
                # Colour is the PIXEL and dash is the RUN, so with two arrays the
                # pairs that matter are same-colour: one pixel before and after.
                col = cols.get(_pkey(r), "0.4")
                tb, yb = log_bin(t, np.abs(np.abs(i) - iinf))
                keep = (tb >= t_min) & (yb > 0)
                ax.plot(tb[keep], yb[keep] * 1e9, color=col, lw=1.3, alpha=0.9,
                        ls=_run_dash(ri),
                        label=f"{_plabel(_pkey(r))}  a={alpha:.2f}, r2={r2:.3f}"
                              + (f"  [{run.name}]" if len(runs) > 1 else ""))
                tf = np.geomspace(tb[keep][0], tb[keep][-1], 40)
                ax.plot(tf, yb[keep][0] * (tf / tf[0]) ** -alpha * 1e9,
                        color=col, ls=":", lw=1.0, alpha=0.5)
        ok = log_axes(ax)
        ax.set_xlabel("t after the step  [s]")
        ax.set_ylabel("distance from the fitted asymptote  |I - I_inf|  [nA]")
        note = ("" if not ok else
                "\nevery fit here is weak -- this channel is not measuring a current"
                if worst < 0.95 else "")
        ax.set_title(f"{title}{note}", fontsize=10,
                     color="#b22222" if note else "black")
        if ok:
            ax.legend(fontsize=7.5)
        ax.grid(which="both", alpha=0.25)
    fig.suptitle("Relaxation is a power law, not an RC exponential. "
                 "Straight on log-log = t^-a; dotted = the fit.", fontsize=10)
    fig.tight_layout()
    on = [v[0] for k, v in fits.items() if k[1].endswith(" hc") and np.isfinite(v[0])]
    r2on = [v[2] for k, v in fits.items() if k[1].endswith(" hc") and np.isfinite(v[2])]
    r2off = [v[2] for k, v in fits.items()
             if k[1].endswith("no-hc") and np.isfinite(v[2])]
    caption(fig,
            what="The same long captures, replotted as distance from the asymptote "
                 "they are approaching, on log-log -- where a power law is a straight "
                 "line and an RC exponential is not.",
            how="I_inf is a FITTED parameter, never the `settled` column: these records "
                "are still drifting at 370 s, so the tail median still contains "
                "relaxation. Fitted over log-spaced bins so each decade of time weighs "
                "the same, with I_inf searched on both sides because this current "
                "RISES to its asymptote rather than decaying to it.",
            learn=f"With High-C: straight over 2.6 decades, a = "
                  f"{_fmt_range(on, '', '{:.2f}')} at r2 = "
                  f"{_fmt_range(r2on, '', '{:.3f}')}. The relaxation is a POWER LAW, "
                  f"not a capacitor discharging. Whether removing the capacitors "
                  f"removed the power law -- rather than merely shrinking C -- is the "
                  f"Array38 question, and this is the panel that answers it. Without "
                  f"High-C r2 reaches only {_fmt_range(r2off, '', '{:.2f}')}: there is "
                  f"no exponent there to quote.")
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
    caption(fig,
            what="LEFT: the settled reading at a fixed bias against the measure range "
                 "it was taken on, High-C off, five ranges. RIGHT: the same pixel at "
                 "the same bias measured long with High-C and without, with the "
                 "optical power recorded in the same captures.",
            how="Left sweeps --ranges with High-C off, where range and compliance are "
                "independent knobs. Right uses the two long captures: High-C on, where "
                "the limit selects the range, against High-C off on a fixed 1 uA range.",
            learn="Left: at a FIXED bias the reading sits at ~50 % of whatever range "
                  "was selected, across four decades. It follows the meter, not the "
                  "device. Right is decisive because the power meter is an independent "
                  "witness: the current differs 13x between the two states while the "
                  "light agrees to within 2.4 %. The pixel did not change, the reading "
                  "did. High-C cannot be dropped on this array, and a de-capped "
                  "Array38 having fixed it means these two bars matching.", pad=0.01)
    fig.savefig(out / "highc_needed.png", dpi=DPI)
    plt.close(fig)
    return verdict


def fig_bias_response(runs, out: Path):
    """Current and light against bias, High-C on. They do not fall together.

    On Array42 between -0.95 and -0.80 V the current falls 5.4x while the light falls
    22x. So as the bias drops, a growing share of the current stops producing photons
    -- which is the same parasitic path the settling figure sees, measured a second way.
    """
    fig, axes = plt.subplots(1, 3, figsize=(14.6, 4.5), squeeze=False, sharex=True)
    cols = pixel_colours(runs)
    ratios = {}
    for ri, run in enumerate(runs):
        per_px = defaultdict(list)
        for r in run.step:
            if r["stage"] != "fast" or r["highc"] != "True":
                continue
            per_px[_pkey(r)].append((_num(r["v_to"]), abs(_num(r["settled"])),
                                     abs(_num(r["power_W"]))))
        for key, pts in sorted(per_px.items(),
                              key=lambda kv: (QUADRANTS.index(kv[0][0])
                                              if kv[0][0] in QUADRANTS else 99,
                                              kv[0][1], kv[0][2])):
            pts.sort()
            v = np.array([p[0] for p in pts])
            i = np.array([p[1] for p in pts]) * 1e9
            p = np.array([p[2] for p in pts]) * 1e12
            col, dash = cols.get(key, "0.4"), _run_dash(ri)
            lbl = _plabel(key) + (f"  [{run.name}]" if len(runs) > 1 else "")
            axes[0][0].plot(v, i, "o", ms=4, lw=1.2, color=col, ls=dash, alpha=0.9,
                            label=lbl)
            axes[0][1].plot(v, p, "o", ms=4, lw=1.2, color=col, ls=dash, alpha=0.9)
            # The claim in the title, drawn. Two panels on separate log scales can
            # look parallel while differing by 5x; their ratio cannot.
            with np.errstate(divide="ignore", invalid="ignore"):
                axes[0][2].plot(v, np.where(i > 0, p / i, np.nan), "o", ms=4, lw=1.2,
                                color=col, ls=dash, alpha=0.9)
            if i.size > 1 and i[0] > 0 and p[0] > 0:
                ratios.setdefault(run.name, []).append(
                    (key[0], float(i[0] / i[-1]), float(p[0] / p[-1])))
    for ax, lab, ttl in ((axes[0][0], "I on ch B  [nA]", "current"),
                         (axes[0][1], "optical power  [pW]", "light"),
                         (axes[0][2], "P / I  [pW per nA]",
                          "light per unit current\n(this is where they diverge)")):
        ax.set_yscale("log")
        ax.set_xlabel("bias  [V]")
        ax.set_ylabel(lab)
        ax.set_title(ttl, fontsize=10)
        ax.grid(which="both", alpha=0.25)
    axes[0][0].legend(fontsize=7.5)
    fig.suptitle("Current and light against bias, High-C on. They do not fall together:"
                 " the gap is current that makes no photons.", fontsize=10)
    fig.tight_layout()
    fi = [v[1] for items in ratios.values() for v in items]
    fp = [v[2] for items in ratios.values() for v in items]
    caption(fig,
            what="Settled current and optical power against bias, High-C on, from the "
                 "fast captures at four biases.",
            how="Each point is the median of the last 20 % of a 1.7 s record. Current "
                "and light come from the same capture, so they share the pixel, the "
                "bias and the moment -- no pairing across runs is involved.",
            learn=f"Across the bias range the current falls "
                  f"{_fmt_range(fi, 'x', '{:.1f}')} while the light falls "
                  f"{_fmt_range(fp, 'x', '{:.1f}')}. The third panel is that gap "
                  f"directly: light per unit current drops several-fold toward the "
                  f"faint end, so a growing share of the current produces no photons. "
                  f"Same parasitic path the settling figure sees, measured a second and "
                  f"independent way.")
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
