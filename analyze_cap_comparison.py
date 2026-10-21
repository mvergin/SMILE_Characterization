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
DPI = 150
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
BACKFILL = {"stage": "fast", "leakage_suspect": "False", "lit_blank_ratio": "",
            "p_t_1pct": "", "p_settled_W": "", "channels": "2"}


def _rows(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as f:
        out = list(csv.DictReader(f))
    missing = sorted(k for k in BACKFILL if out and k not in out[0])
    for r in out:
        for k in missing:
            r[k] = BACKFILL[k]
        # --pixels records quadrant="given", so derive it rather than plotting a
        # single bar labelled "given" for four different quadrants.
        if r.get("quadrant") in (None, "", "given"):
            r["quadrant"] = _quadrant_of(_num(r.get("pixel_x")), _num(r.get("pixel_y")))
    if missing:
        print(f"  ! {path.parent.name}/{path.name} predates "
              f"{', '.join(missing)} -- assuming "
              + ", ".join(f"{k}={BACKFILL[k]!r}" for k in missing), file=sys.stderr)
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

    def traces(self, prefix: str) -> list[tuple[str, np.ndarray]]:
        """Trace arrays in run order. The index prefix IS the order they were taken."""
        return [(k, self.npz[k]) for k in sorted(self.keys) if k.startswith(prefix)]

    def step_traces(self) -> list[tuple[dict, np.ndarray, np.ndarray | None]]:
        """Pair every step.csv row with its I(t) and P(t), by order within a tag group.

        Trace tags carry no pixel identity -- `step_long_hc1_rNone_f-1.000_v-0.800` is
        the same string for all four pixels -- so the n-th trace of a tag belongs to
        the n-th row of that tag. Grouping first means one missing trace (keep_trace
        skips an empty capture) cannot shift every later pairing onto the wrong pixel.
        """
        if self._st is not None:
            return self._st
        itr, ptr = defaultdict(list), defaultdict(list)
        for k, arr in self.traces("0"):        # NNN_step_..._<stride> : t, i_a, i_b
            itr[re.sub(r"^\d{3}_", "", re.sub(r"_\d+$", "", k))].append(arr)
        for k, arr in self.traces("pm_"):      # pm_NNN_step_... : t, P
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
        vtr = defaultdict(list)
        for k, arr in self.traces("v_"):
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
            d = ib - cand
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

    # Two passes. A single 400-point grid over [0, min] has a step of ~1 nA, and an
    # I_inf wrong by 1 nA leaves a constant offset in the tail that flattens the fit:
    # a true 1.6 came back as 1.12 on synthetic data purely from grid resolution.
    top = float(ib.min())
    best = scan(np.linspace(0.0, top, 400, endpoint=False))
    if np.isfinite(best[1]):
        step = top / 400.0
        best = max(best, scan(np.linspace(max(0.0, best[1] - 2 * step),
                                          min(top * (1 - 1e-12), best[1] + 2 * step),
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

def fig_cap_per_quadrant(runs, out: Path):
    """C per quadrant, lit, usable setpoints only.

    A setpoint is usable when the lit and blanked ramps agree: the capacitor cannot
    care whether a sub-turn-on pixel is addressed, so a disagreement means the diode's
    leakage is taking a share of the ramp current. Both values are then unusable --
    including the one that looks plausible (39.41 uF against 213.56 uF on Array42).
    """
    fig, ax = plt.subplots(figsize=(9.5, 4.8))
    w = 0.8 / max(1, len(runs))
    summary = {}
    for ri, run in enumerate(runs):
        per_q = defaultdict(list)
        for r in run.cap:
            if r["lit"] != "True":
                continue
            c = _num(r["capacitance_F"])
            if np.isfinite(c):
                per_q[r["quadrant"]].append((c, r["leakage_suspect"] == "True"))
        good, flagged = [], []
        for q in QUADRANTS:
            vals = per_q.get(q, [])
            keep = [c for c, bad in vals if not bad]
            drop = [c for c, bad in vals if bad]
            good.append(np.median(keep) * 1e6 if keep else np.nan)
            flagged.append(np.median(drop) * 1e6 if drop else np.nan)
        xs = np.arange(len(QUADRANTS))
        off = (ri - (len(runs) - 1) / 2) * w
        ax.bar(xs + off, good, w, color=COLOURS[ri], label=f"{run.name} usable")
        ax.bar(xs + off, flagged, w, facecolor="none", edgecolor=COLOURS[ri],
               hatch="///", linewidth=1.0,
               label=f"{run.name} leakage-flagged (NOT usable)")
        for x, g in zip(xs, good):
            if np.isfinite(g):
                ax.annotate(f"{g:.1f}", (x + off, g), ha="center", va="bottom",
                            fontsize=7.5, rotation=90)
        ok = [g for g in good if np.isfinite(g)]
        summary[run.name] = (float(np.median(ok)) if ok else float("nan"),
                             float(np.min(ok)) if ok else float("nan"),
                             float(np.max(ok)) if ok else float("nan"))
    ax.set_xticks(np.arange(len(QUADRANTS)), QUADRANTS)
    ax.set_xlabel("quadrant")
    ax.set_ylabel("carrier capacitance  [uF]")
    ax.set_title("Capacitance per quadrant (lit, leakage-free setpoints)\n"
                 "hatched = lit and blanked disagreed, so neither value is usable",
                 fontsize=10)
    ax.legend(fontsize=7.5, ncol=2)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out / "cap_per_quadrant.png", dpi=DPI)
    plt.close(fig)
    return summary


def fig_cap_ramps(runs, out: Path):
    """The raw V(t) the capacitance was fitted from. C = I / (dV/dt), so slope is 1/C.

    Lit solid, blanked dashed. Where the two diverge, the diode is drawing part of the
    ramp current and the slope under-reads -- the whole leakage test, made visible.
    """
    sets = sorted({_num(r["i_set"]) for run in runs for r in run.cap})
    sets = [s for s in sets if np.isfinite(s)]
    if not sets:
        return
    fig, axes = plt.subplots(1, len(sets), figsize=(4.0 * len(sets), 3.9),
                             squeeze=False, sharey=True)
    for si, iset in enumerate(sets):
        ax = axes[0][si]
        for ri, run in enumerate(runs):
            for r, tv in run.cap_traces():
                if abs(_num(r["i_set"]) - iset) > 1e-15:
                    continue
                t, v = clean(tv[0], tv[1])
                lit = r["lit"] == "True"
                ax.plot(t, v, color=COLOURS[ri], lw=1.1 if lit else 0.9,
                        ls="-" if lit else "--", alpha=0.85,
                        label=f"{run.name} {'lit' if lit else 'blanked'}")
        ax.set_title(f"I = {iset * 1e9:.0f} nA", fontsize=9)
        ax.set_xlabel("t  [s]")
        ax.grid(alpha=0.3)
        if si == 0:
            ax.set_ylabel("V on ch B  [V]")
        handles, labels = ax.get_legend_handles_labels()   # one entry per (run, lit)
        seen = dict(zip(labels, handles))
        ax.legend(seen.values(), seen.keys(), fontsize=7)
    fig.suptitle("Capacitance ramps: a current source charges the node, dV/dt = I/C "
                 "(lit solid, blanked dashed)", fontsize=10)
    fig.tight_layout()
    fig.savefig(out / "cap_ramps.png", dpi=DPI)
    plt.close(fig)


def fig_step_fast(runs, out: Path):
    """The fast capture: the charge edge and the first seconds, High-C on, per bias.

    High-C only, because without it a sub-10 uA reading is a range artifact rather
    than a current -- see range_artifact.png.
    """
    biases = sorted({_num(r["v_to"]) for run in runs for r in run.step
                     if r["stage"] == "fast" and r["highc"] == "True"}, reverse=True)
    if not biases:
        return
    fig, axes = plt.subplots(1, len(biases), figsize=(3.4 * len(biases), 3.7),
                             squeeze=False)
    for bi, v in enumerate(biases):
        ax = axes[0][bi]
        for ri, run in enumerate(runs):
            for r, it, _ in run.step_traces():
                if (r["stage"] != "fast" or r["highc"] != "True"
                        or abs(_num(r["v_to"]) - v) > 1e-9):
                    continue
                t, i = clean(it[0], it[2])          # row 2 is ch B
                ok = t > 0
                ax.plot(t[ok], i[ok] * 1e9, color=COLOURS[ri], lw=0.8, alpha=0.7)
            ax.plot([], [], color=COLOURS[ri], lw=1.2, label=run.name)
        ax.set_title(f"V = {v:+.2f} V", fontsize=9)
        ax.set_xlabel("t after the step  [s]")
        log_axes(ax, x=True, y=False)
        ax.grid(which="both", alpha=0.3)
        if bi == 0:
            ax.set_ylabel("I on ch B  [nA]")
            ax.legend(fontsize=7.5)
    fig.suptitle("Fast capture, High-C on: charge edge and early relaxation "
                 "(one line per pixel)", fontsize=10)
    fig.tight_layout()
    fig.savefig(out / "step_fast.png", dpi=DPI)
    plt.close(fig)


def fig_step_long(runs, out: Path, t_min=1.0):
    """The long capture on log-log, with the fitted exponent. The headline figure.

    Settling here is a POWER LAW, not an RC exponential: the amplitude scales with the
    step size and the decay goes as t^-1.2, which is displaced charge / dielectric
    absorption. Whether removing the capacitors removed the power law -- not merely
    made C smaller -- is the question this plot answers.

    One panel per High-C state, because the two are measured on different ranges, so
    their absolute currents are not comparable to each other. Across runs, inside one
    panel, they are.
    """
    states = (("True", "High-C on"), ("False", "High-C off"))
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.6), squeeze=False)
    fits = {}
    for si, (hc, title) in enumerate(states):
        ax = axes[0][si]
        found = False
        for ri, run in enumerate(runs):
            for r, it, _ in run.step_traces():
                if r["stage"] != "long" or r["highc"] != hc:
                    continue
                t, i = clean(it[0], it[2])
                if t.size < 50:
                    continue
                found = True
                alpha, iinf, r2, n = fit_power_law(t, i, t_min=t_min)
                key = (run.name, title, f"{r['quadrant']} {_num(r['v_to']):+.2f}V")
                fits[key] = (alpha, iinf, r2, n)
                tb, yb = log_bin(t, np.abs(i) - (iinf if np.isfinite(iinf) else 0.0))
                keep = (tb >= t_min) & (yb > 0)
                ax.plot(tb[keep], yb[keep] * 1e9, color=COLOURS[ri], lw=1.0, alpha=0.8)
                if np.isfinite(alpha) and keep.sum() > 2:
                    tf = np.geomspace(tb[keep][0], tb[keep][-1], 50)
                    ref = yb[keep][0]
                    ax.plot(tf, ref * (tf / tf[0]) ** (-alpha) * 1e9,
                            color=COLOURS[ri], ls=":", lw=1.5)
            ax.plot([], [], color=COLOURS[ri], lw=1.2, label=run.name)
        ax.plot([], [], color="0.4", ls=":", lw=1.5, label="fitted power law")
        log_axes(ax)
        ax.set_xlabel("t after the step  [s]")
        ax.set_ylabel("|I(t) - I_inf|  [nA]")
        ax.set_title(title + ("" if found else "  (no long capture in these runs)"),
                     fontsize=10)
        ax.grid(which="both", alpha=0.25)
        ax.legend(fontsize=7.5)
    fig.suptitle("Long capture on log-log: a straight line IS a power law.  "
                 f"I_inf is fitted, not the `settled` column.  t > {t_min:g} s",
                 fontsize=10)
    fig.tight_layout()
    fig.savefig(out / "step_long_loglog.png", dpi=DPI)
    plt.close(fig)
    return fits


def fig_range_artifact(runs, out: Path):
    """Can High-C be dropped? The one plot that answers it.

    Without High-C on Array42, a sub-10 uA reading sits at ~50 % of whatever measure
    range was selected, at a FIXED bias -- so `settled` tracks the RANGE and not the
    device. On this plot that is a line of slope 1. A reading that tracks bias instead
    is flat, at the High-C value. Flat is the bar for Array38.
    """
    fig, axes = plt.subplots(1, len(runs), figsize=(4.9 * len(runs), 4.2),
                             squeeze=False, sharey=True)
    verdict = {}
    for ri, run in enumerate(runs):
        ax = axes[0][ri]
        biases = sorted({_num(r["v_to"]) for r in run.step}, reverse=True)
        all_x = []
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
            col = COLOURS[bi % len(COLOURS)]
            all_x.append(x)
            ax.plot(x, y, "o-", lw=1.0, ms=4, color=col, label=f"V = {v:+.2f} V")
            truth = [abs(_num(r["settled"])) for r in run.step
                     if (r["highc"] == "True" and r["stage"] == "long"
                         and abs(_num(r["v_to"]) - v) < 1e-9)]
            truth = truth or [abs(_num(r["settled"])) for r in run.step
                              if (r["highc"] == "True"
                                  and abs(_num(r["v_to"]) - v) < 1e-9)]
            truth = [a for a in truth if np.isfinite(a) and a > 0]
            if truth:
                ax.axhline(float(np.median(truth)), color=col, ls="--", lw=0.9,
                           alpha=0.7)
            verdict.setdefault(run.name, []).append(
                (v, float(np.median(y / x)), float(np.ptp(np.log10(y)))))
        # The ranges actually plotted. Scraping get_lines() instead picked up the
        # axhline, whose xdata is [0, 1] in axes coordinates, so the guide line ran
        # out to 1 A and stretched the axis past any measurement.
        drawn = np.concatenate(all_x) if all_x else np.array([])
        if drawn.size:
            xs = np.geomspace(drawn.min(), drawn.max())
            ax.plot(xs, 0.5 * xs, color="0.6", ls=":", lw=1.2,
                    label="50 % of range (pure artifact)")
        log_axes(ax)
        ax.set_xlabel("selected measure range  [A]")
        ax.set_title(f"{run.name} -- High-C OFF\n"
                     "on the guide line = artifact, flat = usable\n"
                     "dashed = the High-C truth at that bias", fontsize=8.5)
        ax.grid(which="both", alpha=0.25)
        if ri == 0:
            ax.set_ylabel("|settled| reading  [A]")
        ax.legend(fontsize=7)
    fig.suptitle("Does a non-High-C reading track the DEVICE or the RANGE?",
                 fontsize=10)
    fig.tight_layout()
    fig.savefig(out / "range_artifact.png", dpi=DPI)
    plt.close(fig)
    return verdict


def fig_settling(runs, out: Path):
    """Electrical against optical settling, per High-C state and stage.

    On Array42 the light settles about 45x faster than the current (2.24 s against
    101.15 s at -0.80 V), so the slow electrical tail carries NO light and is
    parasitic. If removing the capacitors collapses the electrical time toward the
    optical one, the relaxation lived in their dielectric.
    """
    groups = (("long", "True", "long, High-C on"),
              ("long", "False", "long, High-C off"),
              ("fast", "True", "fast, High-C on"))
    labels, el, op = [], [], []
    for stage, hc, title in groups:
        for run in runs:
            sel = [r for r in run.step if r["stage"] == stage and r["highc"] == hc]
            e = [x for x in (_num(r["t_1pct"]) for r in sel) if np.isfinite(x)]
            p = [x for x in (_num(r["p_t_1pct"]) for r in sel) if np.isfinite(x)]
            if not sel:
                continue
            labels.append(f"{run.name}\n{title}")
            el.append(float(np.median(e)) if e else np.nan)
            op.append(float(np.median(p)) if p else np.nan)
    if not labels:
        return {}
    fig, ax = plt.subplots(figsize=(1.6 * len(labels) + 3, 4.6))
    x = np.arange(len(labels))
    ax.bar(x - 0.2, el, 0.4, color="#1f77b4", label="electrical t(1%)")
    ax.bar(x + 0.2, op, 0.4, color="#ff7f0e", label="optical t(1%)")
    for xi, (a, b) in enumerate(zip(el, op)):
        for dx, val in ((-0.2, a), (0.2, b)):
            if np.isfinite(val) and val > 0:
                ax.annotate(f"{val:.3g}", (xi + dx, val), ha="center", va="bottom",
                            fontsize=7)
        if np.isfinite(a) and np.isfinite(b) and b > 0:
            ax.annotate(f"{a / b:.0f}x", (xi, max(a, b) * 1.7), ha="center",
                        fontsize=8.5, color="0.3")
    if np.any(np.isfinite(el + op) & (np.array(el + op) > 0)):
        ax.set_yscale("log")
    ax.set_xticks(x, labels, fontsize=7.5)
    ax.set_ylabel("time to 1 % of final  [s]")
    ax.set_title("Current settles long after the light does. A missing electrical bar "
                 "means it never reached 1 % inside the record.", fontsize=9.5)
    ax.legend(fontsize=8)
    ax.grid(axis="y", which="both", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out / "settling_bars.png", dpi=DPI)
    plt.close(fig)
    return dict(zip(labels, zip(el, op)))


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

    caps = fig_cap_per_quadrant(runs, out)
    fig_cap_ramps(runs, out)
    fig_step_fast(runs, out)
    fits = fig_step_long(runs, out, t_min=a.fit_from)
    verdict = fig_range_artifact(runs, out)
    settle = fig_settling(runs, out)

    print(f"\n-- capacitance, lit, leakage-free setpoints {'-' * 34}")
    for name, (med, lo, hi) in caps.items():
        print(f"  {name:22} median {med:7.2f} uF    per-quadrant {lo:6.2f} - {hi:6.2f}")

    print(f"\n-- power law |I-I_inf| ~ t^-alpha, fitted for t > {a.fit_from:g} s {'-' * 16}")
    if fits:
        for (name, state, where), (alpha, iinf, r2, n) in sorted(fits.items()):
            # Synthetic records with a known exponent all fit at r2 > 0.99, so
            # anything below that is a record without a resolvable power law in it,
            # not a measured exponent. Quoting it would be quoting noise.
            flag = "" if r2 > 0.99 else "   << WEAK FIT, do not quote this exponent"
            print(f"  {name:14} {state:12} {where:12}  alpha={alpha:5.2f}  "
                  f"I_inf={iinf * 1e9:8.1f} nA  r2={r2:5.3f}  bins={n}{flag}")
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

    print(f"\n-- settling, electrical against optical {'-' * 37}")
    for label, (e, p) in settle.items():
        one = label.replace("\n", " / ")
        ratio = (f"{e / p:7.1f}x" if (np.isfinite(e) and np.isfinite(p) and p > 0)
                 else "    n/a")
        print(f"  {one:42} {e:9.3f} s {p:8.3f} s {ratio}")

    print(f"\nwrote {len(list(out.glob('*.png')))} figures to {out}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
