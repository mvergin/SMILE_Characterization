"""IQE acquisition for the SMILE micro-LED array — the measurement core.

This module acquires data and writes CSVs.  It does NOT plot or fit — that lives
in the separate SMILE_IQE_Analysis repo::

    python analyze_array.py "<run_dir>"        # SMILE_IQE_Analysis

Design (informed by auditing the results_georg data and the 10mA/1mA bench runs):

  * **Voltage-source adaptive sweep (default, proven).** ch B sources voltage; a
    quick per-pixel coarse pre-sweep maps I(V), then the fine sweep places points
    at log-spaced LED currents across the IQE peak. On Array42 this captured the
    peak (J≈6, ~59%). A ``source_mode="current"`` path exists (ch B sinks a set
    current) but does NOT work on this device — the +1.8 V anode bypasses ch B
    through a leakage path, so the cathode current setpoint never becomes the LED
    current (see ``IQEConfig.source_mode``). Voltage mode is the one to use.
  * **NVLED (ch B) current is the LED current.** EQE/J use ch B; ch A (VLED) is
    recorded as a leakage cross-check.
  * **PM400 and Keithley overlap.** The SMU integrates its burst in the background
    (fire) while the PM400 is read, then we join — roughly halves per-point time.
  * **Incremental disk writes.** Each pixel's rows are appended + flushed as soon
    as it finishes, so a crash mid-run keeps all completed pixels.
  * **Settle-then-average, adaptive effort.** Apply the setpoint, wait a few ms,
    average. Per point the effort follows the bottleneck: high current -> fast
    (low NPLC, few PM reads); low current -> many PM reads (optical-limited). The
    stored ``*_STD`` columns are the standard error of the mean, so heavier
    averaging genuinely lowers the gated uncertainty.

Output (specified in RUN_FORMAT.md, consumed by SMILE_IQE_Analysis)::

    <run_dir>/FastScan/{aggregated_bitval=<b>_mean_std.csv, raw_data/...}
    <run_dir>/NVLEDSweep/Sweep/{aggregated_bitval=<b>_mean_std.csv, raw_data/...}

Every parameter lives on :class:`~measure_iqe.config.IQEConfig`; nothing here
reads module-level settings.  Pass ``simulate=True`` to
:func:`run_iqe_measurement` to dry-run the whole flow (both source modes) against
a built-in ABC-model rig, with no hardware attached.
"""

from __future__ import annotations

import csv
import dataclasses
import datetime
import json
import math
import queue
import sys
import threading
import time
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator, NamedTuple

import numpy as np

from .config import Q_E, SMU_I_RANGES, IQEConfig
from .calibrate import calibrate, write_calibration
from .safeexit import parked, safe_exit

Logger = Callable[[str], None]
Row = dict[str, Any]          # one CSV row: column name -> value


def make_logger(stream=None) -> Logger:
    """A timestamped print-style logger that survives a cp1252 console.

    A unicode log line must never crash a running measurement, so the fallback
    re-encodes with replacement characters instead of raising.
    """
    out = stream if stream is not None else sys.stdout
    try:
        out.reconfigure(encoding="utf-8", errors="replace")   # type: ignore[union-attr]
    except Exception:
        pass

    def log(message: str) -> None:
        line = f"[{datetime.datetime.now():%H:%M:%S}] {message}"
        try:
            print(line, file=out)
        except UnicodeEncodeError:
            print(line.encode("ascii", "replace").decode("ascii"), file=out)

    return log



def roi_pixels(x1: int, y1: int, x2: int, y2: int) -> list[tuple[int, int]]:
    return [(x, y) for y in range(min(y1, y2), max(y1, y2) + 1)
            for x in range(min(x1, x2), max(x1, x2) + 1)]


def spacing_summary(rows, min_sep: int = 0) -> str:
    """How spread out a pixel selection actually is, as a one-line log string.

    The nearest-neighbour distance is the number that distinguishes a genuine
    spatial sample from one that only looks stratified: if picks touch, the
    per-block statistics blur into their neighbours.
    """
    if len(rows) < 2:
        return f"spacing: {len(rows)} pixel(s), nothing to compare"
    # ponytail: O(n^2) pairwise, skipped above 2000 picks (a >9 h run); chunk or use a
    # KD-tree if a sweep ever gets that big.
    if len(rows) > 2000:
        return f"spacing: {len(rows)} pixels (pairwise spacing check skipped)"
    p = np.array([[r["X"], r["Y"]] for r in rows], float)
    d = np.abs(p[:, None, :] - p[None, :, :]).max(-1)
    np.fill_diagonal(d, np.inf)
    nn = d.min(axis=1)
    return (f"spacing: nearest-neighbour min {nn.min():.0f} px, median {np.median(nn):.0f} px"
            f" (target ≥ {min_sep}); {int((nn <= 1).sum())} touching pixel(s)")


def _fps_pick(rows, k, start_idx, avoid=None, min_sep=0):
    """Up to k rows by farthest-point sampling, seeded at start_idx.

    ``avoid`` holds pixels already selected in OTHER blocks. They enter the distance
    objective, so each pick is pushed away from the neighbouring blocks' picks and
    towards this block's interior. Without it the walk only ever maximised distance
    *within* the block, whose extreme points are its own border — and the block next
    door did the same, so the two border picks landed side by side across the shared
    edge. The seeded first pick keeps ``block_seed`` varying runs.

    ``min_sep`` is a hard Chebyshev floor against everything already chosen, in this
    block and in ``avoid`` alike. A candidate closer than that is never taken, so a
    block too sparse to supply k separated pixels returns FEWER rather than two
    touching LEDs whose readings would be redundant anyway. Pass 0 or 1 to allow
    neighbours.
    """
    pts = np.array([[r["X"], r["Y"]] for r in rows], float)
    far = np.full(len(rows), np.inf)      # squared Euclidean to nearest chosen — the spread objective
    sep = np.full(len(rows), np.inf)      # Chebyshev to nearest chosen — the hard floor

    def mark(px, py):
        nonlocal far, sep
        far = np.minimum(far, (pts[:, 0] - px) ** 2 + (pts[:, 1] - py) ** 2)
        sep = np.minimum(sep, np.maximum(np.abs(pts[:, 0] - px), np.abs(pts[:, 1] - py)))

    for px, py in (avoid or ()):
        mark(px, py)

    chosen: list[int] = []
    while len(chosen) < k:
        ok = sep >= min_sep
        ok[chosen] = False                             # never pick the same pixel twice
        if not ok.any():
            break                                      # block exhausted at this separation
        if not chosen and ok[int(start_idx)]:
            i = int(start_idx)                         # seeded start → block_seed varies runs
        else:
            i = int(np.argmax(np.where(ok, far, -np.inf)))
        chosen.append(i)
        mark(pts[i, 0], pts[i, 1])
    return [rows[i] for i in chosen]


def select_sweep_pixels(alive: list[Row], cfg: IQEConfig, log: Logger) -> list[Row]:
    """Pick which alive pixels get the full IQE sweep — see SWEEP_SELECT."""
    if not alive:
        return []
    if cfg.sweep_select == "stride":
        sel = alive[::cfg.nth_pixel_sweep][:cfg.max_sweep_pixels]
        log(f"  selected {len(sel)}/{len(alive)} alive pixels (stride, every {cfg.nth_pixel_sweep}th)")
        return sel

    if cfg.sweep_select == "blocks":
        x1, y1, x2, y2 = cfg.resolve_roi()
        xlo, xhi, ylo, yhi = min(x1, x2), max(x1, x2), min(y1, y2), max(y1, y2)
        nbx = max(1, int(round((xhi - xlo + 1) / cfg.block_size)))
        nby = max(1, int(round((yhi - ylo + 1) / cfg.block_size)))
        xe = np.linspace(xlo, xhi + 1, nbx + 1)
        ye = np.linspace(ylo, yhi + 1, nby + 1)
        rng = np.random.default_rng(cfg.block_seed)
        picks = []                                     # one list of K pixels per non-empty block
        taken: list[tuple[int, int]] = []              # every pixel chosen so far, ALL blocks
        bs = cfg.block_size
        n_short = 0
        for bx in range(nbx):
            for by in range(nby):
                block = [r for r in alive if xe[bx] <= r["X"] < xe[bx + 1]
                         and ye[by] <= r["Y"] < ye[by + 1]]
                if not block:
                    continue
                # Only picks within one block of this one can end up touching it, so the
                # avoid-set stays small however big the ROI is.
                near = [p for p in taken if xe[bx] - bs <= p[0] < xe[bx + 1] + bs
                        and ye[by] - bs <= p[1] < ye[by + 1] + bs]
                # RANDOM start per block → each block's FPS pattern differs → sampled
                # rows/columns diversified across blocks (no repeating sub-lattice).
                got = _fps_pick(block, cfg.pixels_per_block, int(rng.integers(len(block))),
                                avoid=near, min_sep=cfg.min_pixel_separation)
                if not got:
                    continue
                if len(got) < cfg.pixels_per_block:
                    n_short += 1          # too few well-separated alive pixels in this block
                picks.append(got)
                taken.extend((r["X"], r["Y"]) for r in got)
        # Round-robin MEASUREMENT ORDER: pass 0 takes the 1st pixel of every block, pass 1 the
        # 2nd, etc. So consecutive measurements hop between blocks and each block's K pixels are
        # spread across the whole run → slow thermal drift is decorrelated from spatial position
        # (no fake gradient) and averages out in each block's median. (Block-by-block order would
        # alias the drift onto a raster pattern.)
        sel = []
        for p in range(max((len(bp) for bp in picks), default=0)):
            for bp in picks:
                if p < len(bp):
                    sel.append(bp[p])
        log(f"  selected {len(sel)} pixels: {nbx}×{nby} blocks (~{cfg.block_size}px), "
            f"{cfg.pixels_per_block}/block, {len(picks)} non-empty blocks, seed {cfg.block_seed} "
            f"(randomized-start FPS, round-robin order → drift-decorrelated)")
        # Report the realised spacing — this is the number that says whether the sample is
        # spatially spread or secretly clustered, so it belongs in the run log.
        log("  " + spacing_summary(sel, cfg.min_pixel_separation)
            + (f"; {n_short} block(s) gave fewer than {cfg.pixels_per_block} "
               f"well-separated alive pixels" if n_short else ""))
        return sel

    # "spatial": global farthest-point sampling of a fraction of the alive pixels
    n = len(alive)
    k = min(max(cfg.min_sweep_pixels, int(round(cfg.sweep_fraction * n))), cfg.max_sweep_pixels, n)
    pts = np.array([[r["X"], r["Y"]] for r in alive], float)
    start = int(np.argmin(((pts - pts.mean(axis=0)) ** 2).sum(axis=1)))   # nearest centroid
    sel = _fps_pick(alive, k, start)
    log(f"  selected {len(sel)}/{n} alive pixels ({100*len(sel)/n:.0f}%, farthest-point "
        f"sampling for uniform spatial coverage)")
    return sel


# =====================================================================
#  Acquisition primitive: SMU burst and PM reads OVERLAP
# =====================================================================

def _stats(vals: Iterable[float | None], robust: bool = False) -> tuple[float, float]:
    """Return (centre, standard-error-of-the-mean) of the samples.

    The stored *_STD columns hold the SEM (σ/√N), i.e. the uncertainty of the
    MEAN — not the raw per-sample scatter. That is the right quantity for the
    analysis reliability gates: averaging more samples (e.g. many SMU reads at low
    J) genuinely lowers it, so the well-averaged low-J points stop being rejected.

    robust=True uses a median + MAD-based σ instead of mean/std. Used for the SMU
    CURRENT, which autorange can spike with the occasional range-hunting outlier
    (seen in warmup-diag 260629-0759/0847): the median ignores those so a single
    bad sample in the burst can't corrupt the point. The clean PM channel stays mean.
    """
    a = np.asarray([v for v in vals if v is not None and np.isfinite(v)], float)
    if a.size == 0:
        return float("nan"), float("nan")
    if a.size == 1:
        return float(a[0]), 0.0
    if robust:
        med = float(np.median(a))
        mad = float(np.median(np.abs(a - med)))
        sigma = 1.4826 * mad if mad > 0 else float(a.std(ddof=1))
        return med, float(sigma / math.sqrt(a.size))
    return float(a.mean()), float(a.std(ddof=1) / math.sqrt(a.size))


def read_voltage_b(smu, n: int = 3) -> tuple[float, float]:
    """Average n quick ch B voltage reads (diode voltage in current-source mode)."""
    if hasattr(smu, "inst"):
        vals = []
        for _ in range(int(n)):
            try:
                vals.append(float(smu.inst.query("print(smub.measure.v())")))
            except Exception:
                pass
        return _stats(vals)
    return smu.measure_voltage_b(), 0.0


def acquire_point(pm, smu, n_smu: int, n_pm: int, measure_vb: bool = False,
                  return_raw: bool = False):
    """Average n_smu SMU current samples + n_pm PM readings at the current state.

    The SMU current burst is *fired* (non-blocking) so it integrates on the
    instrument while we read the PM400, then we *join* — the two instruments
    overlap. Returns (P,Pσ, Iv,Ivσ, Inv,Invσ, Vb,Vbσ); currents signed, σ ≥ 0.
    """
    smu.clear_buffers_only()
    smu.measure_burst_fire(int(n_smu))         # SMU starts integrating in background
    pvals = [pm.measure() for _ in range(int(n_pm))]   # PM reads concurrently
    smu.measure_burst_join()                   # wait for the SMU burst to finish
    ia, ib = smu.read_buffers(int(n_smu))
    vb_m, vb_s = read_voltage_b(smu) if measure_vb else (float("nan"), float("nan"))
    iv_m, iv_s = _stats(ia, robust=True)     # currents: median (autorange-glitch proof)
    inv_m, inv_s = _stats(ib, robust=True)
    p_m, p_s = _stats(pvals)                 # light: clean, plain mean
    out = (p_m, p_s, iv_m, iv_s, inv_m, inv_s, vb_m, vb_s)
    if return_raw:
        # the individual burst samples, for the raw-samples file and full mean/median/std
        return out, dict(pm=list(pvals), vled=list(ia), nvled=list(ib))
    return out


def burst_stats(vals: Iterable[float | None]) -> dict[str, float]:
    """(mean, median, std, sem, n) of a burst — for the aggregated file's honest columns."""
    a = np.asarray([v for v in vals if v is not None and np.isfinite(v)], float)
    if a.size == 0:
        return dict(mean=float("nan"), median=float("nan"), std=float("nan"), sem=float("nan"), n=0)
    std = float(a.std(ddof=1)) if a.size > 1 else 0.0
    return dict(mean=float(a.mean()), median=float(np.median(a)), std=std,
                sem=std / math.sqrt(a.size) if a.size > 1 else 0.0, n=int(a.size))


# =====================================================================
#  Current setpoints (current-source mode) / adaptive voltages (legacy)
# =====================================================================

def adaptive_params(i_target: float | None, cfg: IQEConfig) -> tuple[float, int, int, float]:
    """(NPLC, N_PM, N_SMU, SETTLE_MS) for a point given its target LED current. Heavy
    CURRENT averaging + longer settle at low current (noisy, fine range settles slowly);
    fast at high current. None target → low-J (conservative)."""
    if i_target is None or abs(i_target) < cfg.adaptive_threshold_a:
        return cfg.lowj_nplc, cfg.lowj_n_pm, cfg.lowj_n_smu, cfg.lowj_settle_ms
    return cfg.highj_nplc, cfg.highj_n_pm, cfg.highj_n_smu, cfg.highj_settle_ms


def build_current_setpoints(cfg: IQEConfig) -> list[float]:
    """Log-spaced LED current setpoints (negative — ch B sinks the current)."""
    mags = np.logspace(math.log10(cfg.current_sweep_low_a), math.log10(cfg.current_sweep_high_a),
                       int(cfg.current_sweep_n))
    return [-float(m) for m in mags]           # ch B sinks → negative setpoint


def build_fine_voltages(pm, smu, cfg: IQEConfig, log: Logger,
                        tag: str = "") -> list[tuple[float, float | None]]:
    """Coarse pre-sweep → fine (voltage, target-current) pairs.

    Returns a list of (voltage, target_current_A); target_current is None for the
    plain-ramp fallback. The current targets are log-spaced but biased toward LOW
    current (LOWJ_WEIGHT) so the SRH-rising side that pins peak IQE is sampled
    densely — replacing the old multi-bitval trick with smarter spacing.
    """
    if cfg.fine_sweep_mode == "voltage":
        # Voltage ramp — no coarse map. Densify the rising-side/peak window (split at the
        # knee); targets None so the points are taken as-is.
        if cfg.vramp_low_frac and 0.0 < cfg.vramp_low_frac < 1.0 and cfg.vramp_knee_v is not None:
            n_low = max(2, int(round(cfg.n_fine * cfg.vramp_low_frac)))
            n_high = max(2, cfg.n_fine - n_low)
            V = np.concatenate([
                np.linspace(cfg.nvled_v_coarse_start, cfg.vramp_knee_v, n_low, endpoint=False),
                np.linspace(cfg.vramp_knee_v, cfg.nvled_v_coarse_stop, n_high)])
            dens = (f"split: {n_low} pts {cfg.nvled_v_coarse_start}→{cfg.vramp_knee_v} V (rising), "
                    f"{n_high} pts {cfg.vramp_knee_v}→{cfg.nvled_v_coarse_stop} V (droop)")
        else:
            V = np.linspace(cfg.nvled_v_coarse_start, cfg.nvled_v_coarse_stop, cfg.n_fine)
            dens = f"uniform {cfg.n_fine} pts {cfg.nvled_v_coarse_start}→{cfg.nvled_v_coarse_stop} V"
        ramp = [(float(v), None) for v in V]
        out = sorted(ramp, reverse=(cfg.sweep_direction == "low_to_high"))
        log(f"    {tag}voltage ramp ({dens}), "
            f"{'range from the limit (High-C)' if cfg.highc_b else f'fixed {cfg.nvled_range_i:g} A range'}, "
            f"{cfg.sweep_direction}")
        return out
    set_nplc(smu, cfg.coarse_nplc)
    # Sweep the coarse map in the SAME direction as the fine sweep. The device is
    # hysteretic (conditioning/parasitic/afterglow), so a fixed low→high coarse map
    # mispredicts a high→low fine sweep — which mis-places points (and, before the
    # probe-then-measure fix, mis-set the ranges → the high-bias overflow). Matching the
    # direction makes the map reflect the same I–V state the fine sweep will see.
    Vc = np.linspace(cfg.nvled_v_coarse_start, cfg.nvled_v_coarse_stop, cfg.n_coarse)
    if cfg.sweep_direction == "high_to_low":
        Vc = Vc[::-1]                          # start at high bias, ramp down — matches fine
    Ic = np.full(len(Vc), np.nan)
    for k, v in enumerate(Vc):
        smu.set_voltage("b", float(v))
        time.sleep(cfg.coarse_settle_ms / 1000.0)
        _, _, _, _, inv_m, _, _, _ = acquire_point(pm, smu, cfg.coarse_n_smu, 1)
        Ic[k] = abs(inv_m) if np.isfinite(inv_m) else np.nan
    ok = np.isfinite(Ic)
    ramp = [(float(v), None) for v in np.linspace(cfg.nvled_v_coarse_start, cfg.nvled_v_coarse_stop, cfg.n_fine)]
    if cfg.sweep_direction == "high_to_low":
        ramp = ramp[::-1]                       # fallback ramp also follows the direction
    if ok.sum() < 5:
        log("    coarse pre-sweep failed; using plain voltage ramp")
        return ramp
    V, I = Vc[ok], Ic[ok]
    order = np.argsort(I)
    I_sorted, V_sorted = I[order], V[order]
    I_mono = np.maximum.accumulate(I_sorted)
    I_max = float(I_mono[-1])
    I_lo = max(cfg.low_current_target_a, float(I_mono[0]))
    I_hi = cfg.high_current_frac * I_max
    if not (I_hi > I_lo > 0):
        log("    current span too small; using plain voltage ramp")
        return ramp
    # log-current targets, biased toward I_lo: u in [0,1], u**LOWJ_WEIGHT clusters
    # near 0 (low current); map to log10(I).
    u = np.linspace(0.0, 1.0, cfg.n_fine) ** float(cfg.lowj_weight)
    targets = 10.0 ** (math.log10(I_lo) + u * (math.log10(I_hi) - math.log10(I_lo)))
    Vf = np.clip(np.interp(targets, I_mono, V_sorted), cfg.nvled_v_coarse_stop, cfg.nvled_v_coarse_start)
    # pair (rounded voltage -> target); dedup collapsed voltages.
    pairs = {}
    for v, it in zip(Vf, targets):
        pairs[round(float(v), 4)] = float(it)
    # Order by SWEEP_DIRECTION. "low_to_high" (descending V, least-negative first)
    # measures the rising side first — right after turn-on at minimal power — so the
    # peak-IQE-determining points are taken when the pixel is COOLEST. "high_to_low"
    # starts hot (for the self-heating A/B test).
    out = sorted(pairs.items(), reverse=(cfg.sweep_direction == "low_to_high"))
    n_low = sum(1 for _, it in out if it < cfg.adaptive_threshold_a)
    log(f"    {tag}coarse I(V): {I_mono[0]*1e6:.3f} µA @ start → {I_max*1e6:.1f} µA max; "
        f"{len(out)} fine steps {I_lo*1e6:.3f}–{I_hi*1e6:.1f} µA "
        f"({n_low} low-J at NPLC {cfg.lowj_nplc}/{cfg.lowj_n_smu} SMU/{cfg.lowj_n_pm} PM, "
        f"rest {cfg.highj_nplc}/{cfg.highj_n_smu} SMU/{cfg.highj_n_pm} PM)")
    return out


# =====================================================================
#  CSV writers — aggregated (per bitval) + incremental raw
# =====================================================================

# Back-compat columns: *_MEAN and *_STD are what analyze_array reads (historically MEAN = the robust
# median, STD = the SEM σ/√N). When SAVE_RAW_SAMPLES, we additionally write honest *_MEDIAN and
# *_SAMPSTD (population std) columns; analyze_array prefers MEDIAN/SEM when present (see its loader).
_AGG_COLS = ["X", "Y", "BITVAL", "NVLED_V", "TIME_START", "TIME_END",
             "NVLED_CURR_MEAN", "NVLED_CURR_STD", "VLED_CURR_MEAN", "VLED_CURR_STD",
             "PM400_POWER_MEAN", "PM400_POWER_STD"]
_AGG_COLS_FULL = _AGG_COLS + [
    "NVLED_CURR_MEDIAN", "NVLED_CURR_SAMPSTD", "NVLED_CURR_N",
    "VLED_CURR_MEDIAN", "VLED_CURR_SAMPSTD",
    "PM400_POWER_MEDIAN", "PM400_POWER_SAMPSTD"]
# The leakage reference: the same pixel at LEAK_REF_BITVAL, measured back to back with the
# lit point. I_LED = NVLED_CURR_MEAN - NVLED_CURR_REF, and that subtraction is what makes
# the rising side meaningful at all -- at -0.80 V the reference is 58% of the reading.
# Appended, never inserted, so a reader that does not know these columns still works.
_AGG_COLS_REF = ["NVLED_CURR_REF", "NVLED_CURR_REF_STD", "VLED_CURR_REF",
                 "PM400_POWER_REF", "PM400_POWER_REF_STD", "LEAK_REF_BITVAL"]
# The compliance actually in force at each point, and whether the point hit it. Written
# for the voltage ramp, where the limit tracks the measured current (see pick_limit).
_AGG_COLS_LIMIT = ["NVLED_LIMIT_A", "NVLED_AT_COMPLIANCE"]


def agg_cols(cfg: IQEConfig) -> list[str]:
    """The aggregated-CSV columns this configuration writes."""
    cols = list(_AGG_COLS_FULL if cfg.save_raw_samples else _AGG_COLS)
    if cfg.leak_reference:
        cols += _AGG_COLS_REF
    if cfg.fine_sweep_mode == "voltage":
        cols += _AGG_COLS_LIMIT
    return cols


def write_aggregated_one(path: str | Path, rows: list[Row]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        f.write(",".join(_AGG_COLS) + "\n")
        for r in rows:
            f.write(",".join(repr(r[c]) for c in _AGG_COLS) + "\n")


def write_aggregated_per_bitval(folder: str | Path, rows: list[Row]) -> None:
    folder = Path(folder)
    for b in sorted({r["BITVAL"] for r in rows}):
        write_aggregated_one(folder / f"aggregated_bitval={b}_mean_std.csv",
                             [r for r in rows if r["BITVAL"] == b])


RAW_HEADER = "X,Y,BITVAL,NVLED_V,TIME,TYPE,MEAS_VALUE,MEAS_STD\n"


def raw_lines(rows: list[Row], dark: Row | None = None) -> Iterator[str]:
    """The raw-CSV lines for one pixel's point rows (+ optional dark reading).

    Both writers emit this exact format, so the formatting lives here once —
    ``analyze_array`` matches on the TYPE column, and a drift between the two
    writers would silently produce two different raw files.
    """
    for r in rows:
        for typ, mean, std in [("PM400", r["PM400_POWER_MEAN"], r["PM400_POWER_STD"]),
                               ("VLED", r["VLED_CURR_MEAN"], r["VLED_CURR_STD"]),
                               ("NVLED", r["NVLED_CURR_MEAN"], r["NVLED_CURR_STD"])]:
            yield (f"{r['X']},{r['Y']},{r['BITVAL']},{r['NVLED_V']},"
                   f"{r['TIME_START']},{typ},{mean!r},{std!r}\n")
    if dark is not None:
        for typ, val in [("DARK_PM400", dark["P"]), ("DARK_VLED", dark["Iv"]),
                         ("DARK_NVLED", dark["Inv"])]:
            yield (f"{dark['X']},{dark['Y']},{dark['BITVAL']},{dark['NVLED_V']},"
                   f"{dark['TIME']},{typ},{val!r},nan\n")


class IncrementalRaw:
    """Append-and-flush raw CSV writer — survives a crash mid-run."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.f = open(self.path, "w")
        self.f.write(RAW_HEADER)
        self.f.flush()

    def append(self, rows, dark=None):
        self.f.writelines(raw_lines(rows, dark))
        self.f.flush()

    def close(self):
        try:
            self.f.close()
        except Exception:
            pass


class BackgroundWriter:
    """Writes the raw + per-bitval aggregate CSVs from a background thread, so the per-pixel
    flush never blocks the measurement loop. This matters because the save dir is often a
    cloud-synced / network folder (Nextcloud) where a flush can take 50–200 ms; with a raw
    and an aggregate stream that is ~0.1–0.4 s/pixel of blocking, which adds up over hundreds
    to thousands of pixels. Each pixel's batch is enqueued by submit(); the worker appends
    BOTH streams and flushes. close() drains the queue and joins, so a clean stop (incl. a
    caught interrupt via the caller's finally) leaves complete, directly-analysable files.
    The aggregate is what the analysis reads — an interrupted run needs no reconstruction."""

    def __init__(self, raw_path: str | Path, agg_folder: str | Path,
                 cfg: IQEConfig, log: Logger, append: bool = False) -> None:
        self._agg_folder = Path(agg_folder)
        Path(raw_path).parent.mkdir(parents=True, exist_ok=True)
        self._agg_folder.mkdir(parents=True, exist_ok=True)
        self._log = log
        self._q: queue.Queue = queue.Queue()
        self._append = append          # resuming: keep what the killed run already wrote
        mode = "a" if append else "w"
        self._raw_f = open(raw_path, mode)
        if not append:
            self._raw_f.write(RAW_HEADER)
            self._raw_f.flush()
        self._cols = agg_cols(cfg)
        # Per-individual-sample file (every burst reading), when enabled.
        self._samp_f = None
        if cfg.save_raw_samples:
            self._samp_f = open(Path(raw_path).parent / "Sweep_samples.csv", mode)
            if not append:
                self._samp_f.write("X,Y,BITVAL,NVLED_V,TIME,TYPE,SAMPLE_IDX,VALUE\n")
                self._samp_f.flush()
        self._agg_f = {}
        self._thread = threading.Thread(target=self._worker, daemon=True)
        self._thread.start()

    def _agg_fh(self, bitval):
        if bitval not in self._agg_f:
            path = self._agg_folder / f"aggregated_bitval={bitval}_mean_std.csv"
            existing = self._append and path.exists() and path.stat().st_size > 0
            f = open(path, "a" if existing else "w")
            if not existing:
                f.write(",".join(self._cols) + "\n")
            self._agg_f[bitval] = f
        return self._agg_f[bitval]

    def _write_one(self, prows, dark, samples=None):
        self._raw_f.writelines(raw_lines(prows, dark))
        self._raw_f.flush()
        # every individual burst sample → Sweep_samples.csv
        if self._samp_f is not None and samples:
            for x, y, b, v, t, raw in samples:
                for typ, arr in [("PM400", raw["pm"]), ("VLED", raw["vled"]), ("NVLED", raw["nvled"])]:
                    for k, val in enumerate(arr):
                        self._samp_f.write(f"{x},{y},{b},{v},{t},{typ},{k},{val!r}\n")
            self._samp_f.flush()
        for r in prows:
            self._agg_fh(r["BITVAL"]).write(
                ",".join(repr(r.get(c, float("nan"))) for c in self._cols) + "\n")
        for f in self._agg_f.values():
            f.flush()

    def _worker(self):
        while True:
            item = self._q.get()
            if item is None:
                self._q.task_done()
                break
            try:
                self._write_one(*item)
            except Exception as e:
                try:
                    self._log(f"WRITER error: {e}")
                except Exception:
                    pass
            self._q.task_done()

    def submit(self, prows, dark, samples=None):
        """Enqueue one pixel's point rows (+ dark, + per-sample bursts) — never blocks on I/O."""
        self._q.put((prows, dark, samples))

    def close(self):
        self._q.put(None)
        self._thread.join(timeout=30.0)
        for f in [self._raw_f, self._samp_f, *self._agg_f.values()]:
            try:
                if f is not None:
                    f.close()
            except Exception:
                pass


# =====================================================================
#  Instrument configuration (real)
# =====================================================================

def connect_real(cfg: IQEConfig, log: Logger):
    from instruments.pm400 import PM400
    from instruments.keithley2602b import Keithley2602B
    log(f"PM400 @ {cfg.pm400_addr}")
    pm = PM400(cfg.pm400_addr)
    try:
        log(f"Keithley 2602B @ {cfg.smu_addr}")
        smu = Keithley2602B(cfg.smu_addr)
    except Exception:
        pm.close()          # don't leave the PM400 session open on a failed SMU connect
        raise
    try:
        from smile.core import Smile
        from instruments.smile_fpga import SmileFPGA
        smile = SmileFPGA(Smile(timeout=5000, debug_lvl=0))
        log("SMILE FPGA connected.")
    except Exception as e:
        log(f"SMILE FPGA unavailable ({e}); blanking will be a no-op.")
        from instruments.smile_fpga import SmileFPGA
        smile = SmileFPGA(None)
    return pm, smu, smile


def configure_common(pm, smu, cfg: IQEConfig, log: Logger) -> None:
    """PM400 + ch A (voltage anode) + buffers. ch B is configured per source mode."""
    pm.set_wavelength(cfg.pm_wavelength)
    pm.set_power_unit("W")
    pm.set_auto_range(False)
    pm.set_range(cfg.pm_range)
    pm.set_averaging(cfg.pm_averaging)
    if hasattr(pm, "inst"):
        pm.inst.write("INP:FILT 0")            # fast response (no low-pass settling)
    smu.configure_channel("a", cfg.vled_compliance, cfg.sweep_nplc, False, zero_delays=True,
                          range_i=cfg.vled_range_i)
    # ch B starts as a voltage source — the fast scan voltage-drives it. The
    # current-source sweep reconfigures it later (configure_current_source_b).
    configure_voltage_source_b(smu, cfg, log)
    smu.setup_buffers(timestamps=False)
    if hasattr(smu, "inst"):
        smu.inst.write("smub.nvbuffer2.appendmode = 1")
    if cfg.smu_display_off:
        try:
            smu.display_off()
        except Exception:
            pass


def pick_limit(i_last: float | None, cfg: IQEConfig) -> float:
    """The compliance limit for the next sweep point, from the last measured current.

    The limit CANNOT come from the density knee. That was the first attempt and run
    261014-0007 shows what it costs: the knee is at -1.3 V but the device still draws
    30 uA at -1.27 V, so a 10 uA limit clamped eight consecutive points at exactly
    10000.4 nA. A clamped point is not a measurement, and the EQE computed from it is
    meaningless.

    The sweep runs high current to low current, so the previous point is an upper bound
    on the next one: take the smallest standard range that still clears it with 50%
    headroom. That self-tunes per pixel instead of assuming one array's I(V).

    Under High-C the limit also picks the measure range, so this is what sets the noise
    floor too -- hence the floor at ``vramp_lowv_limit_a`` rather than going lower still.
    """
    ladder = [r for r in SMU_I_RANGES
              if cfg.vramp_lowv_limit_a <= r <= cfg.vramp_highv_limit_a]
    if not ladder:
        return cfg.vramp_highv_limit_a
    if i_last is None or not math.isfinite(i_last):
        return ladder[-1]                      # first point: widest, nothing is known yet
    need = 1.5 * abs(i_last)
    return next((r for r in ladder if r >= need), ladder[-1])


def configure_voltage_source_b(smu, cfg: IQEConfig, log: Logger | None = None) -> None:
    """ch B = voltage source / current measure (the adaptive sweep).

    Without High-C: a fixed measure range, which the fine sweep may narrow per point
    (``set_measure_range_b``).

    With ``highc_b``: High-C mode is enabled via raw TSP, because the driver refuses
    ``high_c=True`` on channel B. In High-C the 2602B derives the current range from
    the *limit* and **locks the range settings** — error 5069, "SMU B: Autorange
    locked for HighC mode". So the order is: leave High-C, set the range, re-enable
    High-C, then never touch the range again (``set_measure_range_b`` and
    ``perpoint_range`` are both suppressed). AUTOZERO_ONCE is what the driver's own
    notes say High-C needs. The error queue is drained afterwards, so a refused
    setting is reported instead of silently misreading current.

    Both transitions matter and both were wrong once: writing the range *after*
    enabling High-C is refused, and so is re-entering this function while High-C is
    still on from a previous call.
    """
    if hasattr(smu, "inst"):
        # Leave High-C FIRST, unconditionally. While it is on, the 2602B locks the
        # range settings ("5069 SMU B: Autorange locked for HighC mode"), and
        # configure_channel's next act is to write measure.autorangei and
        # measure.rangei. Disabling first makes this function idempotent: it works
        # whatever state the channel was left in by a previous call.
        smu.inst.write("smub.source.highc = smub.DISABLE")
    smu.configure_channel("b", cfg.nvled_compliance, cfg.sweep_nplc, False,
                          zero_delays=not cfg.highc_b, range_i=cfg.nvled_range_i)
    if hasattr(smu, "inst"):
        if cfg.highc_b:
            # order matters: range first (above), then High-C, then nothing else.
            smu.inst.write("smub.source.highc = smub.ENABLE")
            smu.inst.write("smub.source.delay = smub.DELAY_AUTO")
            smu.inst.write("smub.measure.delay = smub.DELAY_AUTO")
            smu.inst.write("smub.measure.autozero = smub.AUTOZERO_ONCE")
        else:
            smu.inst.write("smub.source.highc = smub.DISABLE")
            # Explicitly manage autozero (was previously left in whatever state the SMU happened
            # to be in) — this is what fixes the NPLC-switch current step at the knee.
            smu.inst.write(f"smub.measure.autozero = smub.AUTOZERO_{cfg.sweep_autozero}")
        errs = smu.check_error_queue() if hasattr(smu, "check_error_queue") else []
        if errs and log is not None:
            log(f"  SMU rejected a ch B setting: {errs}")
    else:
        smu._mode = "voltage"



def set_measure_range_b(smu, rng: float) -> None:
    """Fix ch B current-measure range (autorange OFF) to a chosen value."""
    if hasattr(smu, "inst"):
        smu.inst.write("smub.measure.autorangei = smub.AUTORANGE_OFF")
        smu.inst.write(f"smub.measure.rangei = {rng}")


def acquire_autoranged(pm, smu, n_smu: int, n_pm: int, cfg: IQEConfig):
    """Software auto-ranging by stepping DOWN one decade at a time from the safe wide range.

    A single probe read on the wide range is fooled by range zero-offsets: the 1 mA range
    reads ~+37 µA for a 38 nA current (measured in run 1130), so probing once on 1 mA reads
    the offset, picks 1 mA again, and confirms it (the stuck ~45 µA in run 1123). Instead we
    start wide (covers the device, can't overflow) and narrow while the reading still fits a
    finer range, re-reading each time — landing on the finest range that truly holds the
    current, where the offset is small relative to the signal. Reverts on overflow."""
    ranges = [r for r in SMU_I_RANGES if r <= cfg.probe_range_i]   # ascending up to the probe range
    rng = ranges[-1]
    set_measure_range_b(smu, rng)
    time.sleep(cfg.range_settle_ms / 1000.0)
    _, _, _, _, i, _, _, _ = acquire_point(pm, smu, cfg.probe_n_smu, 1)
    for finer in reversed(ranges[:-1]):
        if not (np.isfinite(i) and abs(i) < 0.85 * finer):
            break                                              # won't fit the finer range
        set_measure_range_b(smu, finer)
        time.sleep(cfg.range_settle_ms / 1000.0)
        _, _, _, _, i2, _, _, _ = acquire_point(pm, smu, cfg.probe_n_smu, 1)
        if not np.isfinite(i2) or abs(i2) >= 0.95 * finer:     # overflowed the finer range
            set_measure_range_b(smu, rng)                      # revert, keep the coarser one
            time.sleep(cfg.range_settle_ms / 1000.0)
            break
        rng, i = finer, i2
    return acquire_point(pm, smu, n_smu, n_pm)


def configure_current_source_b(smu, cfg: IQEConfig) -> None:
    """ch B = current source / voltage measure, with a voltage compliance."""
    if hasattr(smu, "inst"):
        smu.inst.write("smub.source.func = smub.OUTPUT_DCAMPS")
        smu.inst.write(f"smub.source.limitv = {abs(cfg.nvled_v_limit)}")
        smu.inst.write("smub.source.leveli = 0")
        smu.inst.write("smub.source.autorangei = smub.AUTORANGE_OFF")
        smu.inst.write(f"smub.source.rangei = {cfg.current_source_range_i}")
        smu.inst.write("smub.measure.autorangev = smub.AUTORANGE_OFF")
        smu.inst.write(f"smub.measure.rangev = {cfg.measure_v_range}")
        smu.inst.write(f"smub.measure.nplc = {cfg.sweep_nplc}")
        smu.inst.write("smub.source.highc = smub.DISABLE")
        smu.inst.write("smub.source.delay = 0 smub.measure.delay = 0 "
                       "smub.measure.autozero = smub.AUTOZERO_OFF")
    else:
        smu._mode = "current"


def set_nplc(smu, nplc: float) -> None:
    if hasattr(smu, "inst"):
        smu.inst.write(f"smua.measure.nplc = {nplc} smub.measure.nplc = {nplc}")


def set_current_b(smu, i: float) -> None:
    if hasattr(smu, "inst"):
        smu.inst.write(f"smub.source.leveli = {i}")
    else:
        smu.set_current("b", i)


# =====================================================================
#  Simulation rig (built-in ABC model; supports both source modes)
# =====================================================================

class _SimSMU:
    A, B, C, ETA = 1.36e6, 1.0e-11, 8.8e-31, 5e-4
    IMAX, VHALF, WIDTH = 1.4e-4, -2.3, 0.35

    def __init__(self, cfg):
        self._cfg = cfg
        self._vb = 0.0
        self._iset = 0.0
        self._mode = "voltage"
        self._va = self._cfg.vled_voltage
        self._pixel_gain = 1.0
        self._bit_scale = 1.0
        self._on = True
        self._rng = np.random.default_rng(0)

    def configure_channel(self, *a, **k): pass
    def setup_buffers(self, *a, **k): pass
    def clear_buffers_only(self): pass
    def display_off(self): pass
    def display_on(self): pass
    def enable_output(self, *a, **k): pass
    def check_error_queue(self): return []
    def close(self): pass
    def measure_burst_fire(self, n): self._n = int(n)
    def measure_burst_join(self): pass

    def set_voltage(self, ch, v):
        if ch.lower() == "b":
            self._vb = float(v)
        else:
            self._va = float(v)

    def set_current(self, ch, i):
        if ch.lower() == "b":
            self._iset = float(i)

    def _max_current(self):
        return self.IMAX * self._pixel_gain * self._bit_scale

    def _led_current(self):
        if not self._on:
            return 0.0
        if self._mode == "current":
            return min(abs(self._iset), self._max_current())     # compliance clamp
        return self._max_current() / (1.0 + math.exp((self._vb - self.VHALF) / self.WIDTH))

    def read_buffers(self, n):
        mag = self._led_current()
        relnoise = 0.02 + 3e-8 / (mag + 1e-9)
        ib = -mag * (1 + self._rng.normal(0, relnoise, n))
        ia = mag * (1 + self._rng.normal(0, relnoise, n)) * 1.03
        return list(ia), list(ib)

    def measure_voltage_b(self):
        mx = self._max_current()
        I = self._led_current()
        if I >= 0.999 * mx or I <= 0:
            return -abs(self._cfg.nvled_v_limit)
        g = self.IMAX * self._pixel_gain * self._bit_scale
        v = self.VHALF + self.WIDTH * math.log(max(g / I - 1.0, 1e-9))
        return float(max(v, -abs(self._cfg.nvled_v_limit)))


class _SimPM:
    def __init__(self, smu, cfg):
        self._smu = smu
        self._cfg = cfg
        self._rng = np.random.default_rng(1)

    def set_wavelength(self, *a): pass
    def set_power_unit(self, *a): pass
    def set_auto_range(self, *a): pass
    def set_range(self, *a): pass
    def set_averaging(self, *a): pass
    def display_off(self): pass
    def display_on(self): pass
    def close(self): pass

    def measure(self):
        s = self._smu
        I = s._led_current()
        if not s._on or I <= 0:
            return abs(self._rng.normal(0, 2e-12))
        J = I / (self._cfg.pixel_area_um2 * 1e-8)
        roots = np.roots([s.C, s.B, s.A, -J / (Q_E * 10e-7)])
        real = [r.real for r in roots if abs(r.imag) < 1e-6 * (abs(r.real) + 1e-30) and r.real > 0]
        if not real:
            return abs(self._rng.normal(0, 2e-12))
        n = min(real)
        R = s.A * n + s.B * n**2 + s.C * n**3
        eqe = s.ETA * (s.B * n**2 / R if R > 0 else 0.0)
        return max(0.0, eqe * (I / Q_E) * self._cfg.photon_energy_j * (1 + self._rng.normal(0, 0.02)))


class _SimSmile:
    def __init__(self, smu):
        self._smu = smu

    def set_pixel(self, x, y, bit):
        self._smu._on = True
        h = (1103515245 * (x * 1009 + y) + 12345) & 0x7fffffff
        self._smu._pixel_gain = 0.0 if (h % 13 == 0) else 0.6 + 0.7 * ((h >> 8) % 1000) / 1000.0
        self._smu._bit_scale = max(1, int(bit)) / 15.0
        return time.perf_counter()

    def blank_frame(self):
        self._smu._on = False
        return time.perf_counter()

    def close(self): pass


def connect_sim(cfg: IQEConfig, log: Logger):
    log("SIMULATION rig (built-in ABC model) — no hardware.")
    smu = _SimSMU(cfg)
    return _SimPM(smu, cfg), smu, _SimSmile(smu)


# =====================================================================
#  Phases
# =====================================================================

def run_fast_scan(pm, smu, smile, x1: int, y1: int, x2: int, y2: int,
                  run_dir: Path, cfg: IQEConfig, log: Logger) -> list[Row]:
    log("=" * 56)
    log(f"FAST SCAN @ NVLED={cfg.nvled_v_fastscan} V over {(x2-x1+1)*(y2-y1+1)} pixels")
    # The fast scan is a DIFFERENT operating point from the sweep and needs its own
    # compliance. The sweep's rising-side limit clamped every pixel in run 261014-0048.
    configure_voltage_source_b(
        smu, dataclasses.replace(cfg, nvled_compliance=cfg.fastscan_limit_a), log)
    set_nplc(smu, cfg.fastscan_nplc)
    smu.set_voltage("b", cfg.nvled_v_fastscan)     # fast scan always voltage-drives ch B
    fs_rows = []
    order = list(roi_pixels(x1, y1, x2, y2))
    if cfg.fastscan_random_order:
        np.random.default_rng(cfg.fastscan_order_seed).shuffle(order)
        log(f"  visiting {len(order)} pixels in randomized order (seed {cfg.fastscan_order_seed})")
    t0 = time.perf_counter()
    for (x, y) in order:
        if cfg.fastscan_blank_between:
            smile.blank_frame()             # black frame → guarantee only this pixel emits
        smile.set_pixel(x, y, cfg.fastscan_bitval)
        if cfg.fastscan_settle_ms:
            time.sleep(cfg.fastscan_settle_ms / 1000.0)
        p, ps, iv, ivs, inv, invs, _, _ = acquire_point(pm, smu, cfg.fastscan_n_smu, 3)
        fs_rows.append(dict(X=x, Y=y, BITVAL=cfg.fastscan_bitval, NVLED_V=cfg.nvled_v_fastscan,
                            TIME_START=time.perf_counter() - t0, TIME_END=time.perf_counter() - t0,
                            NVLED_CURR_MEAN=inv, NVLED_CURR_STD=invs,
                            VLED_CURR_MEAN=iv, VLED_CURR_STD=ivs,
                            PM400_POWER_MEAN=p, PM400_POWER_STD=ps))
    write_aggregated_one(run_dir / "FastScan" / f"aggregated_bitval={cfg.fastscan_bitval}_mean_std.csv",
                         fs_rows)
    raw = IncrementalRaw(run_dir / "FastScan" / "raw_data" / "FastScan.csv")
    raw.append(fs_rows); raw.close()
    (run_dir / "FastScan" / "raw_data" / "FastScan_config.json").write_text(
        json.dumps({"settings": cfg.to_dict()}, indent=2), encoding="utf-8")

    P = np.array([abs(r["PM400_POWER_MEAN"]) for r in fs_rows])
    med = np.nanmedian(P)
    # "Typical lit" reference = a HIGH percentile, so the threshold survives a majority-dead ROI
    # (the median falls to the noise floor there → 20%×median ≈ noise → nearly everything passes).
    bright = float(np.nanpercentile(P, cfg.fastscan_alive_pctl))
    thr = cfg.dead_threshold_frac * bright
    alive = [r for r, p in zip(fs_rows, P) if p >= thr]
    log(f"  alive {len(alive)}/{len(fs_rows)} (lit~{bright*1e9:.1f} nW @ p{cfg.fastscan_alive_pctl}, "
        f"thr {thr*1e9:.1f} nW, median {med*1e9:.2f} nW)")
    if med < thr:
        log(f"  NOTE: ROI is majority-dark (median {med*1e9:.2f} nW < threshold {thr*1e9:.1f} nW) — "
            f"most of this region is dead; block selection draws from the lit minority.")
    smile.blank_frame()      # never hand the next stage a lit pixel
    return select_sweep_pixels(alive, cfg, log)


def run_sweep_current(pm, smu, smile, selected: list[Row], run_dir: Path,
                      cfg: IQEConfig, log: Logger) -> int:
    """Current-source sweep: set log-spaced currents at bitval 15, measure V + P."""
    setpoints = build_current_setpoints(cfg)
    log("=" * 56)
    log(f"CURRENT-SOURCE SWEEP — {len(selected)} pixels, bitval {cfg.current_sweep_bitval}, "
        f"{len(setpoints)} setpoints {cfg.current_sweep_low_a*1e6:.2f}–{cfg.current_sweep_high_a*1e6:.0f} µA "
        f"(|V| compliance {cfg.nvled_v_limit} V)")
    configure_current_source_b(smu, cfg)
    set_nplc(smu, cfg.sweep_nplc)
    sweep_dir = run_dir / "NVLEDSweep" / "Sweep"
    raw = IncrementalRaw(sweep_dir / "raw_data" / "Sweep.csv")
    sw_rows = []
    t0 = time.perf_counter()
    # try/finally so an interrupt mid-run still closes the raw file with every
    # completed pixel flushed — same guarantee the voltage sweep gives.
    try:
        for i, r in enumerate(selected, 1):
            x, y = r["X"], r["Y"]
            log(f"  [{i}/{len(selected)}] pixel ({x},{y})")
            smile.set_pixel(x, y, cfg.current_sweep_bitval)
            prows = []
            for iset in setpoints:
                set_current_b(smu, iset)
                time.sleep(cfg.settle_ms / 1000.0)
                p, ps, iv, ivs, inv, invs, vb, vbs = acquire_point(
                    pm, smu, cfg.n_smu_samples, cfg.n_pm_samples, measure_vb=True)
                row = dict(X=x, Y=y, BITVAL=cfg.current_sweep_bitval, NVLED_V=round(vb, 4),
                           TIME_START=time.perf_counter() - t0, TIME_END=time.perf_counter() - t0,
                           NVLED_CURR_MEAN=inv, NVLED_CURR_STD=invs,
                           VLED_CURR_MEAN=iv, VLED_CURR_STD=ivs,
                           PM400_POWER_MEAN=p, PM400_POWER_STD=ps)
                prows.append(row); sw_rows.append(row)
            # dark: lowest current, pixel blanked
            set_current_b(smu, setpoints[0])
            smile.blank_frame()
            time.sleep(max(cfg.settle_ms, 5.0) / 1000.0)
            dp, _, div, _, dinv, _, dvb, _ = acquire_point(pm, smu, cfg.coarse_n_smu, 3,
                                                           measure_vb=True)
            dark = dict(X=x, Y=y, BITVAL=cfg.current_sweep_bitval, NVLED_V=round(dvb, 4),
                        TIME=time.perf_counter() - t0, P=dp, Iv=div, Inv=dinv)
            raw.append(prows, dark)                # incremental flush per pixel
    finally:
        raw.close()
        write_aggregated_per_bitval(sweep_dir, sw_rows)
    (sweep_dir / "raw_data" / "Sweep_config.json").write_text(
        json.dumps({"settings": cfg.to_dict()}, indent=2), encoding="utf-8")
    return len(sw_rows)


def measure_point(pm, smu, smile, x, y, bitval, v, limit, n_pm, n_smu,
                  cfg: IQEConfig, t0: float, log: Logger,
                  restore: bool = True) -> tuple[dict, dict | None]:
    """Acquire one (pixel, bias) point and build its row.

    The caller has already driven the bias, chosen the limit/range and the NPLC, and
    waited whatever dwell that point needs. This is only the part that is identical
    whichever order the grid is walked -- the acquisition, the leakage reference, the
    row and the compliance flag -- so `pixel_major` and `bias_major` cannot drift apart
    in what a "point" means.

    `restore=False` says the caller is about to blank the frame anyway, so the pixel
    does not need putting back to `bitval` and settling after the reference.

    Returns (row, raw_samples_or_None); the caller stashes raw_samples if it wants them.
    """
    raw_s = None
    if cfg.perpoint_range:
        # range from the ACTUAL current (probe-then-measure), not the coarse map
        p, ps, iv, ivs, inv, invs, _, _ = acquire_autoranged(pm, smu, n_smu, n_pm, cfg)
    elif cfg.save_raw_samples:
        (p, ps, iv, ivs, inv, invs, _, _), raw_s = acquire_point(
            pm, smu, n_smu, n_pm, return_raw=True)
    else:
        p, ps, iv, ivs, inv, invs, _, _ = acquire_point(pm, smu, n_smu, n_pm)

    # Leakage reference: the SAME pixel at bitval 0. Measured back to back with the lit
    # point so both carry the same relaxation history and the subtraction cancels most of
    # it. At -0.80 V this is 58% of the reading.
    #
    # It stays per point even in bias_major, where one reference per bias looks tempting.
    # The backplane leak varies ~10% between pixels (run 261014-1616 at -1.30 V: 159-192
    # nA across four), and at -0.80 V the reference IS 58% of the reading, so 10% of it
    # is a 6% error on I_LED. bias_major amortises the DWELL; the reference is not a dwell.
    ref = None
    if cfg.leak_reference:
        smile.set_pixel(x, y, cfg.leak_ref_bitval)
        time.sleep(cfg.leak_ref_settle_s * cfg.settle_scale)
        # FULL n_pm, not a quarter of it. The reference current is stable to ~0.2 nA and
        # needs almost no averaging, but the reference POWER is the dark level being
        # subtracted from a ~250 pW signal, so at the dim end it is the dominant error.
        # Run 261014-0035 with n_pm//4: the reference SEM was 30 pW against the lit
        # reading's 14 pW -- the term that mattered most was averaged the least.
        ref = acquire_point(pm, smu, n_smu, n_pm)
        if restore:
            smile.set_pixel(x, y, bitval)
            time.sleep(cfg.leak_ref_settle_s * cfg.settle_scale)
        # else: the caller blanks the frame next, so lighting this pixel again and
        # waiting for it to settle would be two seconds spent on a state nothing reads.
        # In bias_major that is half the per-point cost, on every point.

    vr = round(float(v), 4)
    row = dict(X=x, Y=y, BITVAL=bitval, NVLED_V=vr,
               TIME_START=time.perf_counter() - t0, TIME_END=time.perf_counter() - t0,
               NVLED_CURR_MEAN=inv, NVLED_CURR_STD=invs,
               VLED_CURR_MEAN=iv, VLED_CURR_STD=ivs,
               PM400_POWER_MEAN=p, PM400_POWER_STD=ps)
    if ref is not None:
        rp, rps, riv, _, rinv, rinvs, _, _ = ref
        row.update(NVLED_CURR_REF=rinv, NVLED_CURR_REF_STD=rinvs,
                   VLED_CURR_REF=riv, PM400_POWER_REF=rp,
                   PM400_POWER_REF_STD=rps, LEAK_REF_BITVAL=cfg.leak_ref_bitval)
    if raw_s is not None:
        # honest per-burst stats. _MEAN = true mean, _MEDIAN = median, _SAMPSTD =
        # population std, _STD = SEM (sigma/sqrt(N)) -- kept as _STD because that is the
        # column analyze_array reads for its reliability gate; analyze prefers _MEDIAN.
        sn = burst_stats(raw_s["nvled"]); sv = burst_stats(raw_s["vled"]); sp = burst_stats(raw_s["pm"])
        row.update(NVLED_CURR_MEAN=sn["mean"], NVLED_CURR_STD=sn["sem"],
                   NVLED_CURR_MEDIAN=sn["median"], NVLED_CURR_SAMPSTD=sn["std"], NVLED_CURR_N=sn["n"],
                   VLED_CURR_MEAN=sv["mean"], VLED_CURR_STD=sv["sem"],
                   VLED_CURR_MEDIAN=sv["median"], VLED_CURR_SAMPSTD=sv["std"],
                   PM400_POWER_MEAN=sp["mean"], PM400_POWER_STD=sp["sem"],
                   PM400_POWER_MEDIAN=sp["median"], PM400_POWER_SAMPSTD=sp["std"])
    # A point sitting on the limit is not a measurement. Flag it in the file and in the
    # log rather than letting it reach the fit as a real current.
    inv_m = row["NVLED_CURR_MEAN"]
    row["NVLED_LIMIT_A"] = limit
    at_limit = bool(limit) and abs(inv_m) >= 0.995 * limit
    row["NVLED_AT_COMPLIANCE"] = at_limit
    if at_limit:
        log(f"      AT COMPLIANCE at {v:+.3f} V: |I|={abs(inv_m):.6g} A is the "
            f"{limit:g} A limit, not the device. Point is flagged, not usable.")
    return row, raw_s


def completed_pixels(run_dir: str | Path) -> set[tuple[int, int]]:
    """Pixels already written to a run's aggregated sweep CSV.

    The writer flushes per pixel, so a run killed mid-sweep -- by Windows Update, say --
    has complete rows for every pixel it finished. Those are the ones to skip on resume.
    A pixel that was interrupted PART WAY through its own sweep is not complete, so it
    is re-measured from scratch rather than stitched.
    """
    done: set[tuple[int, int]] = set()
    sweep = Path(run_dir) / "NVLEDSweep" / "Sweep"
    counts: dict[tuple[int, int], int] = {}
    for agg in sweep.glob("aggregated_bitval=*_mean_std.csv"):
        try:
            with agg.open() as f:
                rdr = csv.DictReader(f)
                for row in rdr:
                    try:
                        key = (int(float(row["X"])), int(float(row["Y"])))
                    except (KeyError, TypeError, ValueError):
                        continue
                    counts[key] = counts.get(key, 0) + 1
        except OSError:
            continue
    if not counts:
        return done
    # A finished pixel has the full point count. Use the mode rather than the max, so one
    # over-long pixel cannot mark every other one unfinished.
    full = max(set(counts.values()), key=list(counts.values()).count)
    return {k for k, v in counts.items() if v >= full}


def trace_dwell(pm, smu, cfg: IQEConfig, seconds: float):
    """Poll |I| and P for `seconds`, instead of sleeping. Returns (t, I, P).

    Same total wait as the sleep it replaces, so the sweep is not slowed and the value
    it reports is still the burst taken afterwards. The point is only to keep the
    transient that was being thrown away -- 99% of the dwell at a 65 s settle.

    Stops early if the hardware has been parked, and never runs past `seconds`: a poll
    that would overrun the dwell is not started.
    """
    t0 = time.perf_counter()
    ts: list[float] = []
    iv: list[float] = []
    pv: list[float] = []
    while True:
        dt = time.perf_counter() - t0
        if dt >= seconds or parked():
            break
        p, _, _, _, inv, _, _, _ = acquire_point(
            pm, smu, cfg.trace_settling_n_smu, cfg.trace_settling_n_pm)
        ts.append(dt)
        iv.append(abs(inv))
        pv.append(p)
        rest = cfg.trace_settling_period_s - (time.perf_counter() - t0 - dt)
        if rest > 0:
            time.sleep(min(rest, max(0.0, seconds - (time.perf_counter() - t0))))
    left = seconds - (time.perf_counter() - t0)
    if left > 0:
        time.sleep(left)                       # keep the dwell exactly as long as asked
    return (np.asarray(ts, np.float32), np.asarray(iv, np.float32),
            np.asarray(pv, np.float32))


def completed_biases(run_dir: str | Path) -> set[float]:
    """Biases already measured on EVERY pixel, for --resume in bias_major order.

    `completed_pixels` is the wrong question in this order: a run killed mid-sweep has
    every pixel partially swept and no pixel complete, so it would re-measure the lot.
    What IS complete is each bias the writer flushed, and the writer flushes per bias.
    """
    rows: dict[float, set[tuple[int, int]]] = {}
    sweep = Path(run_dir) / "NVLEDSweep" / "Sweep"
    for agg in sweep.glob("aggregated_bitval=*_mean_std.csv"):
        try:
            with agg.open() as f:
                for row in csv.DictReader(f):
                    try:
                        v = round(float(row["NVLED_V"]), 4)
                        key = (int(float(row["X"])), int(float(row["Y"])))
                    except (KeyError, TypeError, ValueError):
                        continue
                    rows.setdefault(v, set()).add(key)
        except OSError:
            continue
    if not rows:
        return set()
    # A complete bias covers as many pixels as the fullest one does. Using the max
    # rather than a configured count keeps this correct when pixels were dead-skipped.
    full = max(len(px) for px in rows.values())
    return {v for v, px in rows.items() if len(px) == full}


def _sweep_bias_major(pm, smu, smile, selected: list[Row], writer: "BackgroundWriter",
                      settle_traces: list, cfg: IQEConfig, log: Logger,
                      t0: float) -> int:
    """Walk the biases on the outside and the pixels on the inside.

    The dwell belongs to the bias, not to the pixel (see `IQEConfig.sweep_order`), so
    paying it once per bias and then stepping the frame measures every pixel for the
    price of one. The per-point cost becomes `pixel_switch_settle_s` plus acquisition.

    The pixel loop deliberately mirrors the fast scan: blank the frame, then light one
    pixel. That is the sequence measured in run 261014-2151, and the blank is also what
    stops a previous pixel emitting at bitval 0 -- the stuck-on-neighbour of the
    pixel-major loop, for the same reason.
    """
    n_rows = 0
    # One fine grid for all pixels. The biases are a property of the ramp, not of the
    # pixel, so building it per pixel would only add 16 coarse pre-sweeps -- and they
    # cannot differ anyway, or there would be no shared bias to hold.
    first = selected[0]
    smile.blank_frame()
    smile.set_pixel(first["X"], first["Y"], cfg.bitvals[0])
    fine = build_fine_voltages(pm, smu, cfg, log,
                               tag=f"grid from ({first['X']},{first['Y']}): ")
    if not fine:
        log("  no fine voltages built - nothing to sweep")
        return 0

    done_v = completed_biases(writer._agg_folder.parent.parent.parent) if cfg.resume else set()
    if done_v:
        log(f"  resuming: {len(done_v)} bias(es) already complete on every pixel")

    live = [(int(r["X"]), int(r["Y"])) for r in selected]
    n_dead = 0
    cur_limit = cur_nplc = cur_range = None
    i_last = None
    for bitval in cfg.bitvals:
        for jpt, (v, _i_target) in enumerate(fine):
            if parked():
                break
            if round(float(v), 4) in done_v:
                continue
            if cfg.vramp_perseg_effort and v > cfg.vramp_knee_v:
                nplc, n_pm, n_smu, settle_s = (cfg.vramp_lowv_nplc, cfg.vramp_lowv_n_pm,
                                               cfg.vramp_lowv_n_smu, cfg.vramp_lowv_settle_s)
            elif cfg.vramp_perseg_effort:
                nplc, n_pm, n_smu, settle_s = (cfg.vramp_highv_nplc, cfg.vramp_highv_n_pm,
                                               cfg.vramp_highv_n_smu, cfg.vramp_highv_settle_s)
            else:
                nplc, n_pm, n_smu, settle_s = (cfg.sweep_nplc, cfg.n_pm_samples,
                                               cfg.n_smu_samples, cfg.settle_ms / 1000.0)
            limit = pick_limit(i_last, cfg)
            if limit != cur_limit:
                configure_voltage_source_b(
                    smu, dataclasses.replace(cfg, nvled_compliance=limit), log)
                cur_limit, cur_nplc, cur_range = limit, None, None
            if nplc != cur_nplc:
                set_nplc(smu, nplc); cur_nplc = nplc
            if not cfg.perpoint_range and not cfg.highc_b and cfg.nvled_range_i != cur_range:
                set_measure_range_b(smu, cfg.nvled_range_i); cur_range = cfg.nvled_range_i

            # --- the one dwell this bias costs, for all the pixels below it ---
            #
            # Light a pixel FIRST. The previous bias ended with a blanked frame for its
            # dark reading, and relaxing at the new bias in that state is not the state
            # the measurements are taken in: a blanked frame conducts ~4x the bitval-0
            # level (section 6), so the array settles to a different charge. Run
            # 261014-2308 paid its dwell blanked and read 2.2-2.9% LOW at every bias
            # from -1.25 to -0.90 V against the pixel-major run on the same 16 pixels --
            # with the reference and the raw lit current both confirming the offset was
            # in the lit reading, not the subtraction.
            #
            # The switch test (261014-2223) held the bias with ANOTHER PIXEL LIT, which
            # is why it measured 0.3% and this loop lost 2.5%: it was not reproducing
            # the condition the test validated.
            smile.blank_frame()
            smile.set_pixel(live[0][0], live[0][1], bitval)
            smu.set_voltage("b", float(v))
            dwell = settle_s * cfg.settle_scale
            if cfg.trace_settling and dwell >= cfg.trace_settling_min_s:
                tr = trace_dwell(pm, smu, cfg, dwell)
                if tr[0].size:
                    settle_traces.append((f"bias_b{bitval}_v{v:+.4f}", *tr))
            else:
                time.sleep(dwell)
            log(f"  [{jpt + 1}/{len(fine)}] {v:+.3f} V  settled {dwell:g}s once "
                f"with ({live[0][0]},{live[0][1]}) lit, now {len(live)} pixel(s) at "
                f"{cfg.pixel_switch_settle_s * cfg.settle_scale:g}s each")

            rows, samps = [], []
            dead_here = []
            for (x, y) in live:
                if parked():
                    break
                smile.blank_frame()
                smile.set_pixel(x, y, bitval)
                time.sleep(cfg.pixel_switch_settle_s * cfg.settle_scale)
                row, raw_s = measure_point(pm, smu, smile, x, y, bitval, v, limit,
                                           n_pm, n_smu, cfg, t0, log, restore=False)
                if raw_s is not None:
                    samps.append((x, y, bitval, row["NVLED_V"], row["TIME_START"], raw_s))
                rows.append(row)
                i_last = abs(row["NVLED_CURR_MEAN"])
                # Dead-pixel abort, judged at the FIRST bias only -- which under
                # high_to_low is the brightest, exactly as in pixel_major. Here it
                # drops the pixel from every remaining bias, so one judgement saves
                # n_points measurements instead of n_points-1.
                if cfg.dead_abort and jpt == 0 and abs(row["PM400_POWER_MEAN"]) < \
                        max(cfg.dead_abort_nw * 1e-9,
                            cfg.dead_abort_frac * float(np.median(
                                [abs(r["PM400_POWER_MEAN"]) for r in rows]))):
                    dead_here.append((x, y))
            for px in dead_here:
                live.remove(px)
                n_dead += 1
            if dead_here:
                log(f"      dead-skip: {len(dead_here)} pixel(s) below the live floor at "
                    f"the brightest bias - dropped from the remaining {len(fine) - 1} points")

            # One dark reading per bias, with the frame blanked. pixel_major takes one
            # per pixel at whatever bias it stopped on; per bias is both cheaper and a
            # better match to what the analysis subtracts. It leaves the frame blanked,
            # which is why the next bias re-lights before it steps -- see above.
            smile.blank_frame()
            time.sleep(max(cfg.settle_ms, 5.0) / 1000.0)
            dp, _, div, _, dinv, _, _, _ = acquire_point(pm, smu, cfg.coarse_n_smu, 3)
            dark = dict(X=live[0][0] if live else first["X"],
                        Y=live[0][1] if live else first["Y"],
                        BITVAL=bitval, NVLED_V=round(float(v), 4),
                        TIME=time.perf_counter() - t0, P=dp, Iv=div, Inv=dinv)
            writer.submit(rows, dark, samps)
            n_rows += len(rows)
    smile.blank_frame()
    if n_dead:
        log(f"  dead-skipped {n_dead}/{len(selected)} selected pixels (fast-scan false-alive)")
    return n_rows


def run_sweep_voltage(pm, smu, smile, selected: list[Row], run_dir: Path,
                      cfg: IQEConfig, log: Logger) -> int:
    """Voltage-source adaptive sweep. Single bitval, low-J-weighted spacing, and
    longer integration on the low-current rising side (per-point NPLC)."""
    log("=" * 56)
    if cfg.fine_sweep_mode == "voltage":
        eff = (f"low-V NPLC {cfg.vramp_lowv_nplc}/{cfg.vramp_lowv_n_pm} PM/"
               f"{cfg.vramp_lowv_settle_s:g}s/{cfg.vramp_lowv_limit_a:g}A, "
               f"high-V NPLC {cfg.vramp_highv_nplc}/{cfg.vramp_highv_n_pm} PM/"
               f"{cfg.vramp_highv_settle_s:g}s/{cfg.vramp_highv_limit_a:g}A"
               if cfg.vramp_perseg_effort else
               f"uniform NPLC {cfg.sweep_nplc}/{cfg.n_pm_samples} PM")
        # Under High-C the limit, not nvled_range_i, selects the measure range -- saying
        # "fixed 10 mA range" there would be a lie, and it was.
        rng = ("range from the limit (High-C)" if cfg.highc_b
               else f"fixed {cfg.nvled_range_i:g} A range")
        log(f"VOLTAGE SWEEP — {len(selected)} pixels, {cfg.sweep_direction}, V ramp "
            f"{cfg.nvled_v_coarse_start}→{cfg.nvled_v_coarse_stop} V ({cfg.n_fine} pts, low-J-dense @ knee "
            f"{cfg.vramp_knee_v} V), {rng}, {eff}")
        if cfg.leak_reference:
            log(f"  leakage reference: every point re-measured at bitval {cfg.leak_ref_bitval} "
                f"(+{cfg.leak_ref_settle_s:g}s each) → NVLED_CURR_REF. I_LED = MEAN - REF.")
        # Weight by the ACTUAL split, not the average of the two settles: with 18 slow
        # points and 12 fast ones the average under-reported by 16% (run 261014-1306).
        n_low = max(2, int(round(cfg.n_fine * cfg.vramp_low_frac))) if (
            cfg.vramp_low_frac and cfg.vramp_knee_v is not None) else cfg.n_fine // 2
        n_high = max(0, cfg.n_fine - n_low)
        ref_s = 2 * cfg.leak_ref_settle_s if cfg.leak_reference else 0
        npx = max(1, len(selected))
        dwell_all = n_low * cfg.vramp_lowv_settle_s + n_high * cfg.vramp_highv_settle_s
        if cfg.sweep_order == "bias_major":
            # The dwell is paid ONCE per bias for every pixel; what scales with pixels
            # is the frame switch and the reference. Reporting the pixel_major budget
            # here would overstate a 100-pixel run by more than an order of magnitude.
            # one reference settle per point here, not two: see measure_point(restore=)
            ref_bm = cfg.leak_ref_settle_s if cfg.leak_reference else 0
            est = (dwell_all + npx * cfg.n_fine * (cfg.pixel_switch_settle_s + ref_bm)) \
                * cfg.settle_scale
            log(f"  settle time alone: ~{est/3600:.1f} h for {len(selected)} pixels "
                f"(BIAS-MAJOR: {dwell_all:g}s of dwell ONCE for all of them "
                f"[{n_low}x{cfg.vramp_lowv_settle_s:g}s rising + "
                f"{n_high}x{cfg.vramp_highv_settle_s:g}s droop], then "
                f"{cfg.pixel_switch_settle_s:g}s switch + {ref_bm:g}s reference per point).")
            px_major = npx * (n_low * (cfg.vramp_lowv_settle_s + ref_s)
                              + n_high * (cfg.vramp_highv_settle_s + ref_s)) * cfg.settle_scale
            if px_major > est > 0:
                log(f"  pixel-major would be ~{px_major/3600:.1f} h "
                    f"({px_major/est:.0f}x) for the same points.")
        else:
            est = npx * (n_low * (cfg.vramp_lowv_settle_s + ref_s)
                         + n_high * (cfg.vramp_highv_settle_s + ref_s)) * cfg.settle_scale
            per = est / npx
            log(f"  settle time alone: ~{est/3600:.1f} h for {len(selected)} pixels "
                f"({per/60:.1f} min/pixel = {n_low}x{cfg.vramp_lowv_settle_s:g}s rising + "
                f"{n_high}x{cfg.vramp_highv_settle_s:g}s droop + {ref_s:g}s reference each).")
        if est > 6 * 3600:
            log(f"  !! that is {est/3600:.0f} hours. To cut it: "
                + ("fewer pixels (--set max_sweep_pixels=N), fewer points "
                   "(--set n_fine=N), or a looser settle target "
                   "(--set calibrate_target=0.03) re-measured with --calibrate."
                   if cfg.sweep_order == "bias_major" else
                   "--set sweep_order=bias_major, which pays each dwell once for all "
                   "pixels instead of once per pixel, or fewer pixels "
                   "(--set max_sweep_pixels=N), or fewer points (--set n_fine=N)."))
    else:
        log(f"VOLTAGE SWEEP — {len(selected)} pixels × bitvals {list(cfg.bitvals)}, {cfg.sweep_direction}, "
            f"current-targeted (w={cfg.lowj_weight}); adaptive low-J NPLC {cfg.lowj_nplc}, high-J {cfg.highj_nplc}; "
            f"per-point range={cfg.perpoint_range}")
    configure_voltage_source_b(smu, cfg, log)
    sweep_dir = run_dir / "NVLEDSweep" / "Sweep"
    (sweep_dir / "raw_data").mkdir(parents=True, exist_ok=True)
    # Config up front so it exists even if the run is interrupted.
    (sweep_dir / "raw_data" / "Sweep_config.json").write_text(
        json.dumps({"settings": cfg.to_dict()}, indent=2), encoding="utf-8")
    # Self-contained background writer (no external deps): raw + aggregate are flushed off the
    # measurement thread, so slow (Nextcloud/network) flushes don't block the sweep, and the
    # aggregate is written per pixel → an interrupted run is directly analysable. try/finally
    # ensures close() drains the queue and joins on a clean stop or caught interrupt.
    settle_traces: list = []          # (tag, t, I, P) per point, kept for offline work
    skip = completed_pixels(run_dir) if cfg.resume else set()
    if skip:
        log(f"  resuming: {len(skip)} pixel(s) already complete, "
            f"{sum(1 for r in selected if (int(r['X']), int(r['Y'])) not in skip)} to go")
    writer = BackgroundWriter(sweep_dir / "raw_data" / "Sweep.csv", sweep_dir, cfg, log,
                              append=bool(skip))
    sw_rows = []
    live_peaks = []                 # brightest-point power (W) of confirmed-live pixels
    n_dead = 0
    t0 = time.perf_counter()
    try:
      if cfg.sweep_order == "bias_major":
        # Shares the writer, the trace list and the finally below; only the walk differs.
        return _sweep_bias_major(pm, smu, smile, selected, writer, settle_traces,
                                 cfg, log, t0)
      for i, r in enumerate(selected, 1):
        x, y = r["X"], r["Y"]
        if (int(x), int(y)) in skip:
            log(f"  [{i}/{len(selected)}] pixel ({x},{y}) - already complete, skipping")
            continue
        log(f"  [{i}/{len(selected)}] pixel ({x},{y})")
        prows = []
        psamps = []                              # per-sample bursts for this pixel (raw file)
        last_v = cfg.nvled_v_fastscan
        dead_skip = False
        # First point is the brightest (high→low). Dead below an absolute floor OR below a
        # fraction of the running live-pixel median (bootstraps from the absolute floor).
        live_med = float(np.median(live_peaks)) if live_peaks else 0.0
        dead_thr = max(cfg.dead_abort_nw * 1e-9, cfg.dead_abort_frac * live_med)
        first_p = None
        for bitval in cfg.bitvals:
            # Blank FIRST, every pixel, exactly as the fast scan does. Without this the
            # first swept pixel inherits whatever the fast scan left lit -- run_fast_scan
            # returns without blanking -- so two pixels emit while "lit" and the leftover
            # one still emits at bitval 0. That is the "stuck-on neighbour" seen on the
            # first pixel of every scan, old script included: run 261014-0048 pixel 33,32
            # had REF/lit = 0.5018 against 0.0065 for all 23 others.
            smile.blank_frame()
            smile.set_pixel(x, y, bitval)
            fine = build_fine_voltages(pm, smu, cfg, log, tag=f"bit{bitval}: ")
            # Conditioning dwell: hold at high current to clear the slow non-radiative
            # parasitic current that otherwise contaminates the low-J denominator on a
            # fresh/cold pixel (see --warmup-diag 260629-0858). Makes low-J reproducible.
            if cfg.warmup_ms:
                smu.set_voltage("b", float(cfg.warmup_nvled_v))
                time.sleep(cfg.warmup_ms / 1000.0)
            # Then settle at the first sweep point (short vs the seconds-scale
            # conditioning, so the conditioned state persists into the low-J points).
            if fine and cfg.thermal_settle_ms:
                smu.set_voltage("b", float(fine[0][0]))
                time.sleep(cfg.thermal_settle_ms / 1000.0)
            cur_nplc = None
            cur_range = None
            cur_limit = None
            i_last = None          # drives the compliance ladder; see pick_limit
            for jpt, (v, i_target) in enumerate(fine):
                if parked():      # Ctrl-C / shutdown: stop before touching a dead session
                    break
                if cfg.fine_sweep_mode == "voltage":
                    if cfg.vramp_perseg_effort and v > cfg.vramp_knee_v:
                        # low-V / rising side (faint light) → heavy PM averaging + NPLC
                        nplc, n_pm, n_smu, settle_s = (
                            cfg.vramp_lowv_nplc, cfg.vramp_lowv_n_pm,
                            cfg.vramp_lowv_n_smu, cfg.vramp_lowv_settle_s)
                    elif cfg.vramp_perseg_effort:
                        # high-V droop (bright) → fast
                        nplc, n_pm, n_smu, settle_s = (
                            cfg.vramp_highv_nplc, cfg.vramp_highv_n_pm,
                            cfg.vramp_highv_n_smu, cfg.vramp_highv_settle_s)
                    else:
                        nplc, n_pm, n_smu, settle_s = (
                            cfg.sweep_nplc, cfg.n_pm_samples, cfg.n_smu_samples,
                            cfg.settle_ms / 1000.0)
                    # The LIMIT tracks the measured current, not the density knee.
                    limit = pick_limit(i_last, cfg)
                else:
                    # heavy current averaging + longer settle at low current; fast at high
                    nplc, n_pm, n_smu, settle_ms = adaptive_params(i_target, cfg)
                    settle_s, limit = settle_ms / 1000.0, cfg.nvled_compliance
                if limit != cur_limit:
                    # No single limit spans the sweep: -2.6 V draws ~175 uA while the rising
                    # side needs a 10 uA limit to resolve at all. Under High-C the limit also
                    # selects the measure range, so this is a range change mid-sweep -- which
                    # was verified against single-limit ground truth and agreed to 1.1%
                    # (runs 261013-2140 / -2155). Re-entering configure is safe: it leaves
                    # High-C first, so the range write is never refused.
                    configure_voltage_source_b(
                        smu, dataclasses.replace(cfg, nvled_compliance=limit), log)
                    cur_limit, cur_nplc = limit, None
                if nplc != cur_nplc:
                    set_nplc(smu, nplc); cur_nplc = nplc
                smu.set_voltage("b", float(v))
                dwell = settle_s * cfg.settle_scale
                if cfg.trace_settling and dwell >= cfg.trace_settling_min_s:
                    # Poll THROUGH the dwell rather than sleeping it. Same total wait,
                    # same final burst, same reported value -- the transient is simply
                    # kept instead of discarded.
                    tr = trace_dwell(pm, smu, cfg, dwell)
                    if tr[0].size:
                        settle_traces.append(
                            (f"{x}_{y}_b{bitval}_v{v:+.4f}", *tr))
                else:
                    time.sleep(dwell)
                if not cfg.perpoint_range and not cfg.highc_b and cfg.nvled_range_i != cur_range:
                    # In High-C the limit governs the range — writing it here is the
                    # conflict that made earlier High-C runs fail. Leave it alone.
                    set_measure_range_b(smu, cfg.nvled_range_i); cur_range = cfg.nvled_range_i
                row, raw_s = measure_point(pm, smu, smile, x, y, bitval, v, limit,
                                           n_pm, n_smu, cfg, t0, log)
                p, inv = row["PM400_POWER_MEAN"], row["NVLED_CURR_MEAN"]
                if raw_s is not None:
                    psamps.append((x, y, bitval, row["NVLED_V"], row["TIME_START"], raw_s))
                i_last = abs(inv)
                prows.append(row); sw_rows.append(row)
                last_v = v
                # Early dead-pixel abort: judge on the first (brightest) point only.
                if cfg.dead_abort and jpt == 0:
                    first_p = abs(p)
                    if first_p < dead_thr:
                        log(f"      dead-skip: brightest point P={first_p*1e9:.2f} nW "
                            f"< {dead_thr*1e9:.2f} nW → skipping remaining {len(fine)-1} points")
                        dead_skip = True
                        break
            if dead_skip:
                break
        smile.blank_frame()
        if dead_skip:
            n_dead += 1
            writer.submit(prows, None, psamps)  # keep the one measured point; no dark on a dead pixel
            continue
        if first_p is not None:
            live_peaks.append(first_p)  # feed the running live-median for the next pixels
        time.sleep(max(cfg.settle_ms, 5.0) / 1000.0)
        dp, _, div, _, dinv, _, _, _ = acquire_point(pm, smu, cfg.coarse_n_smu, 3)
        dark = dict(X=x, Y=y, BITVAL=cfg.bitvals[-1], NVLED_V=round(float(last_v), 4),
                    TIME=time.perf_counter() - t0, P=dp, Iv=div, Inv=dinv)
        writer.submit(prows, dark, psamps)  # raw + samples + aggregate → background thread
    finally:
        writer.close()                  # drain queue + join → files complete on clean stop
        if settle_traces:
            npz = sweep_dir / "settling_traces.npz"
            np.savez_compressed(npz, **{
                tag: np.vstack([t, i, p]) for tag, t, i, p in settle_traces})
            log(f"  wrote {npz} ({len(settle_traces)} dwell traces, "
                f"{npz.stat().st_size/1e6:.1f} MB) - the transient at every point, "
                f"for the extrapolation and EQE-settling questions")
    if n_dead:
        log(f"  dead-skipped {n_dead}/{len(selected)} selected pixels (fast-scan false-alive)")
    return len(sw_rows)


def run_warmup_diag(pm, smu, smile, pixels: list[tuple[int, int]], run_dir: Path,
                    cfg: IQEConfig, log: Logger) -> None:
    """Characterise time-stability and per-point noise at a fixed operating point on
    several pixels, in two phases each:
      (A) right after a FRESH turn-on, and
      (B) after a high-current conditioning dwell.
    For each phase it reports the LIGHT drift/scatter, the CURRENT scatter, and their
    correlation — which is what tells you whether the irreproducibility is optical
    (a real turn-on transient) or just current-readout noise. Writes warmup_diag.csv
    (pixel_x, pixel_y, phase, t_s, P_W, I_A, EQE)."""
    log("=" * 56)
    log(f"WARM-UP DIAGNOSTIC on {len(pixels)} pixel(s) at NVLED={cfg.warmup_diag_v} V "
        f"({cfg.warmup_diag_seconds:.0f}s fresh + {cfg.warmup_diag_seconds:.0f}s after conditioning each)")
    configure_voltage_source_b(smu, cfg, log)
    set_nplc(smu, cfg.warmup_diag_nplc)
    rows = []

    def watch(x, y, phase, secs):
        t0 = time.perf_counter()
        while time.perf_counter() - t0 < secs:
            p, ps, iv, ivs, inv, invs, _, _ = acquire_point(
                pm, smu, cfg.warmup_diag_n_smu, cfg.warmup_diag_n_pm)
            t = time.perf_counter() - t0
            eqe = ((abs(p) / cfg.photon_energy_j) / (abs(inv) / Q_E)
                   if (inv and np.isfinite(inv) and inv != 0 and np.isfinite(p)) else float("nan"))
            rows.append(dict(pixel_x=x, pixel_y=y, phase=phase, t_s=round(t, 4),
                             P_W=p, I_A=inv, EQE=eqe))

    def report(x, y, phase):
        g = [r for r in rows if r["pixel_x"] == x and r["pixel_y"] == y and r["phase"] == phase]
        P = np.array([r["P_W"] for r in g], float)
        I = np.array([r["I_A"] for r in g], float)
        m = np.isfinite(P) & np.isfinite(I)
        P, I = P[m], I[m]
        if P.size < 5:
            log(f"    ({x},{y}) {phase}: too few points"); return
        relP = 100 * P.std() / abs(P.mean()) if P.mean() else float("nan")
        relI = 100 * I.std() / abs(I.mean()) if I.mean() else float("nan")
        driftP = 100 * np.polyfit(np.arange(P.size), P, 1)[0] * P.size / abs(P.mean())
        corr = np.corrcoef(P, np.abs(I))[0, 1] if P.size > 2 else float("nan")
        log(f"    ({x},{y}) {phase}: P {P.mean():.3e} W (scatter {relP:.1f}%, drift "
            f"{driftP:+.1f}%/run)  |  I {I.mean():.3e} A (scatter {relI:.0f}%)  |  "
            f"corr(P,|I|) {corr:+.2f}")

    for x, y in pixels:
        log(f"  pixel ({x},{y})")
        # A: fresh turn-on at the low-current operating point
        smile.blank_frame(); time.sleep(0.5)
        smu.set_voltage("b", float(cfg.warmup_diag_v))
        smile.set_pixel(x, y, cfg.fastscan_bitval)
        watch(x, y, "fresh_turnon", cfg.warmup_diag_seconds)
        report(x, y, "fresh_turnon")
        # B: condition at high current, then drop to the low point and watch relaxation
        smu.set_voltage("b", float(cfg.warmup_nvled_v))
        time.sleep(max(cfg.warmup_ms, 2000.0) / 1000.0)
        smu.set_voltage("b", float(cfg.warmup_diag_v))
        watch(x, y, "after_conditioning", cfg.warmup_diag_seconds)
        report(x, y, "after_conditioning")

    out = run_dir / "warmup_diag.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        f.write("pixel_x,pixel_y,phase,t_s,P_W,I_A,EQE\n")
        for r in rows:
            f.write(f"{r['pixel_x']},{r['pixel_y']},{r['phase']},{r['t_s']},"
                    f"{r['P_W']!r},{r['I_A']!r},{r['EQE']!r}\n")
    (run_dir / "warmup_diag_config.json").write_text(
        json.dumps({"settings": cfg.to_dict(), "pixels": [list(p) for p in pixels]}, indent=2),
        encoding="utf-8")
    log(f"  saved {out}")


# =====================================================================
#  Main

# =====================================================================
#  Public entry point
# =====================================================================

class IQEResult(NamedTuple):
    """What a finished run produced."""

    run_dir: Path           # directory holding FastScan/ and NVLEDSweep/
    n_pixels: int           # pixels selected for the full sweep
    n_points: int           # sweep points written (0 for a warm-up diagnostic)
    config: IQEConfig       # the configuration actually used


def make_run_dir(cfg: IQEConfig, base_dir: str | Path | None = None,
                 suffix: str | None = None) -> Path:
    """Create and return ``<base_dir>/<stamp>_<sample>_ROI_(x1, y1, x2, y2)``."""
    x1, y1, x2, y2 = cfg.resolve_roi()
    stamp = datetime.datetime.now().strftime("%y%m%d-%H%M")
    root = Path(base_dir) if base_dir is not None else Path(cfg.base_dir)
    tail = f"_{suffix}" if suffix else ""
    run_dir = root / f"{stamp}_{cfg.sample_name}_ROI_({x1}, {y1}, {x2}, {y2}){tail}"
    # The stamp has minute resolution, so two runs started in the same minute with the
    # same sample and ROI would write into one directory and silently merge. Only the
    # resume path may reuse an existing directory, and it passes run_dir explicitly.
    if run_dir.exists() and any(run_dir.iterdir()):
        stamp2 = datetime.datetime.now().strftime("%y%m%d-%H%M%S")
        run_dir = root / f"{stamp2}_{cfg.sample_name}_ROI_({x1}, {y1}, {x2}, {y2}){tail}"
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def connect(cfg: IQEConfig, log: Logger, simulate: bool = False):
    """Open the instruments - the built-in simulation rig when ``simulate``."""
    return connect_sim(cfg, log) if simulate else connect_real(cfg, log)


def shutdown(pm, smu, smile, cfg: IQEConfig) -> None:
    """Park the hardware in a safe state and close every handle.

    Each step is independent: one failing instrument must not leave the others
    powered, so every call is guarded.
    """
    if cfg.source_mode == "current" and hasattr(smu, "inst"):
        try:
            smu.inst.write("smub.source.leveli = 0")
        except Exception:
            pass
    for ch in ("a", "b"):
        try:
            smu.enable_output(ch, False)
        except Exception:
            pass
    try:
        smile.blank_frame()
    except Exception:
        pass
    if cfg.smu_display_off:
        try:
            smu.display_on()
        except Exception:
            pass
    for dev in (pm, smu, smile):
        try:
            dev.close()
        except Exception:
            pass


def run_calibration(config: IQEConfig | None = None, *, simulate: bool = False,
                    base_dir: str | Path | None = None,
                    log: Logger | None = None) -> Path:
    """Characterise ONE array's settling and write a reusable calibration file.

    Run once per array. The sweep then takes `--calibration <file>` instead of a
    handful of hand-typed settle overrides, and the numbers are measured on that
    array rather than inherited from another.
    """
    cfg = (config or IQEConfig()).validate()
    log = log or make_logger()
    out_dir = make_run_dir(cfg, base_dir, suffix="CALIBRATION")
    out_dir.mkdir(parents=True, exist_ok=True)
    log(f"Output: {out_dir}")
    pm, smu, smile = connect(cfg, log, simulate=simulate)
    parked = safe_exit(lambda: shutdown(pm, smu, smile, cfg), log)
    parked.__enter__()
    try:
        configure_common(pm, smu, cfg, log)
        smu.set_voltage("a", cfg.vled_voltage)
        smu.enable_output("a", True)
        smu.enable_output("b", True)
        x1, y1, x2, y2 = cfg.resolve_roi()
        selected = run_fast_scan(pm, smu, smile, x1, y1, x2, y2, out_dir, cfg, log)
        pixels = [(int(r["X"]), int(r["Y"])) for r in selected[:cfg.calibrate_pixels]]
        if not pixels:
            raise RuntimeError("no alive pixels found to calibrate on")
        data = calibrate(pm, smu, smile, pixels, cfg, log, out_dir=out_dir)
    finally:
        parked.__exit__(None, None, None)
    record, stable = write_calibration(out_dir, data)
    log("=" * 56)
    log(f"CALIBRATION for {cfg.sample_name}")
    log(f"  record  : {record}")
    log(f"  per-array: {stable}")
    log(f"Use it with:  python -m measure_iqe --array {cfg.sample_name}")
    return stable


def run_iqe_measurement(config: IQEConfig | None = None, *, simulate: bool = False,
                        warmup_diag: bool = False, base_dir: str | Path | None = None,
                        run_dir: str | Path | None = None,
                        log: Logger | None = None) -> IQEResult:
    """Run a full acquisition: fast scan -> pixel selection -> sweep.

    Args:
        config: the measurement parameters. ``None`` uses :class:`IQEConfig`
            defaults. Validated before any hardware is touched.
        simulate: drive the built-in ABC-model rig instead of real instruments.
        warmup_diag: run the warm-up/conditioning diagnostic instead of the sweep.
        base_dir: output root, overriding ``config.base_dir``.
        run_dir: write straight into this directory instead of creating a
            timestamped one under ``base_dir``.
        log: line logger; defaults to a timestamped stdout logger.

    Returns:
        An :class:`IQEResult` with the run directory and what was written.

    Example:
        >>> from measure_iqe import IQEConfig, run_iqe_measurement
        >>> result = run_iqe_measurement(IQEConfig(roi=(0, 0, 3, 3)), simulate=True)
        >>> result.run_dir.is_dir()
        True
    """
    cfg = (config or IQEConfig()).validate()
    log = log or make_logger()
    if run_dir is not None and Path(run_dir).exists():
        # Resuming into an existing directory. Turn it on in the config rather than as a
        # separate argument, so --print-config and the saved settings both show it.
        cfg = dataclasses.replace(cfg, resume=True)
    x1, y1, x2, y2 = cfg.resolve_roi()
    out_dir = Path(run_dir) if run_dir is not None else make_run_dir(cfg, base_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    log(f"Output: {out_dir}   (source_mode={cfg.source_mode})")

    if simulate and cfg.settle_scale:
        # The CLI zeroes settle_scale under --simulate unless the user set it explicitly;
        # it is the only place that knows the difference between "the default" and "the
        # user asked for 1.0". A library caller that wants the fast path passes
        # settle_scale=0 itself, as the tests do.
        log(f"  simulate: settle_scale={cfg.settle_scale:g}, so the real waits apply "
            f"(~{cfg.vramp_lowv_settle_s:g}s per rising point)")
    pm, smu, smile = connect(cfg, log, simulate=simulate)
    n_points = 0
    # try/finally does not cover the case that actually bit us: Windows Update
    # restarted the machine mid-run and the SMU output stayed on. safe_exit also
    # catches the console-close / logoff / shutdown events, which Python turns into
    # no signal at all and which atexit never sees.
    parked = safe_exit(lambda: shutdown(pm, smu, smile, cfg), log)
    parked.__enter__()
    selected: list[dict[str, Any]] = []
    try:
        configure_common(pm, smu, cfg, log)
        smu.set_voltage("a", cfg.vled_voltage)
        smu.set_voltage("b", cfg.nvled_v_fastscan)   # voltage mode for the fast scan
        smu.enable_output("a", True)
        smu.enable_output("b", True)
        time.sleep(0.05)

        selected = run_fast_scan(pm, smu, smile, x1, y1, x2, y2, out_dir, cfg, log)
        if not selected:
            log("No alive pixels selected - nothing to sweep.")
        elif warmup_diag:
            pixels = [(r["X"], r["Y"]) for r in selected[:cfg.warmup_diag_n_pixels]]
            run_warmup_diag(pm, smu, smile, pixels, out_dir, cfg, log)
        elif cfg.source_mode == "current":
            n_points = run_sweep_current(pm, smu, smile, selected, out_dir, cfg, log)
        else:
            n_points = run_sweep_voltage(pm, smu, smile, selected, out_dir, cfg, log)
        log("=" * 56)
        if selected and not warmup_diag:
            log(f"DONE. {n_points} sweep points over {len(selected)} pixels.")
            log('Analyze separately (SMILE_IQE_Analysis):\n'
                f'    python analyze_array.py "{out_dir}"')
    finally:
        parked.__exit__(None, None, None)      # idempotent: shutdown() runs exactly once
    return IQEResult(run_dir=out_dir, n_pixels=len(selected), n_points=n_points, config=cfg)
