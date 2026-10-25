"""IQE acquisition for the SMILE micro-LED array — the measurement core.

This module acquires data and writes CSVs.  It does NOT plot or fit — that lives
in ``analysis/``, merged in from the SMILE_IQE_Analysis repo by ROADMAP item 2::

    python analysis/analyze_array.py "<run_dir>"

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
    stored ``*_SEM`` columns are the standard error of the mean, so heavier
    averaging genuinely lowers the gated uncertainty.

Output (specified in RUN_FORMAT.md, consumed by ``analysis/``)::

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

from .config import NAMED_ROIS, Q_E, SMU_I_RANGES, IQEConfig
from .provenance import provenance, run_slug, write_provenance
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
    # Every line is kept so the run can save it. Nothing used to: the Console_LOG.txt
    # in one old run directory was the operator redirecting output by hand, so warnings,
    # "<< NOT SETTLED" notices and the pre-settle messages lived only in a terminal and
    # were gone as soon as it scrolled. A 30 h run deserves better. ~100 bytes a line,
    # so even a very chatty run is a megabyte.
    lines: list[str] = []

    def log(message: str) -> None:
        line = f"[{datetime.datetime.now():%H:%M:%S}] {message}"
        lines.append(line)
        try:
            print(line, file=out)
        except UnicodeEncodeError:
            print(line.encode("ascii", "replace").decode("ascii"), file=out)

    def save(path) -> None:
        """Write everything logged so far. Safe to call from a finally."""
        try:
            p = Path(path)
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text("\n".join(lines) + "\n", encoding="utf-8")
        except Exception:
            pass          # a log that cannot be saved must not end the run

    log.lines = lines          # type: ignore[attr-defined]
    log.save = save            # type: ignore[attr-defined]
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


QUADRANT_EDGE = 256          # the 512x512 array is four mirrored 256x256 quadrants


def block_edges(lo: int, hi: int, bs: int) -> np.ndarray:
    """Block edges over [lo, hi] that never straddle the quadrant boundary.

    ``linspace`` over the whole ROI does straddle it. At the full array with
    ``block_size=102`` the edges land on 0, 102.4, 204.8, 307.2, 409.6, 512, so the
    middle block spans 204.8-307.2 and crosses 256 on both axes -- 9 of 25 blocks mix
    two quadrants. That was wrong before pooling existed, for two reasons:

      * the ``tl_pixen``/``tr_pixen``/... enables live in the FRAME instruction, not in
        the pixel data, so a pooled group drawn from such a block lights the near
        quadrant's mirror image instead of the far member;
      * even for single pixels, a block median over two quadrants averages two
        different sets of row drivers, which is the structure section 16 is measuring.

    Tiling each side of the boundary separately removes the case and keeps
    ``block_size`` a free knob. The block COUNT changes when ``block_size`` does not
    divide 256 evenly -- 102 px gives 6 blocks per axis, not 5 -- which is the
    correction, not a regression.
    """
    parts = ([(lo, hi)] if not (lo < QUADRANT_EDGE <= hi)
             else [(lo, QUADRANT_EDGE - 1), (QUADRANT_EDGE, hi)])
    out = [np.linspace(a, b + 1, max(1, int(round((b - a + 1) / bs))) + 1)
           for a, b in parts]
    return np.unique(np.concatenate(out))


def split_block(got: list[Row], cfg: IQEConfig) -> list[Row]:
    """One block's draw -> its single pixels, then one head row per pooled group.

    Groups are formed INSIDE the block, from the block's own well-separated draw, so a
    pooled row measures its block and not the whole quadrant. The earlier array-wide
    draw put 291 of 300 member pairs in different blocks, a median of 152 px apart,
    which averaged away the +3.7 % spatial gradient section 16 measures.

    The head row is the group's first FRESH member and never the shared single, so X,Y
    stays unique across the selection and every ``groupby(["X","Y"])`` downstream keeps
    working. With ``pool_include_singles`` the group also carries one of the block's own
    single pixels, which makes the comparison paired: P(group) - P(single) is the rest
    of the group measured on the same silicon, so an additivity loss is separable from
    pixel-to-pixel variation.

    Blocks never straddle the quadrant boundary (``block_edges``), so every member of a
    group is in one quadrant by construction. The assert is cheap and says so.
    """
    if not cfg.pool_groups:
        return got
    singles = got[:cfg.pixels_per_block]
    spare = list(got[cfg.pixels_per_block:])
    out = list(singles)
    for gi, k in enumerate(int(k) for k in cfg.pool_groups):
        share = singles[gi % len(singles)] if (cfg.pool_include_singles and singles) else None
        need = k - (share is not None)
        if len(spare) < need:
            break                 # this block had too few well-separated alive pixels
        members = [spare.pop(0) for _ in range(need)] + ([share] if share else [])
        coords = tuple((int(r["X"]), int(r["Y"])) for r in members)
        assert len({(x >= 256, y >= 256) for x, y in coords}) == 1, (
            f"group {coords} crosses a quadrant boundary")
        head = dict(members[0])
        head["POOL_MEMBERS"] = coords
        out.append(head)
    return out


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
        xe = block_edges(xlo, xhi, cfg.block_size)
        ye = block_edges(ylo, yhi, cfg.block_size)
        nbx, nby = len(xe) - 1, len(ye) - 1
        rng = np.random.default_rng(cfg.block_seed)
        picks = []                                     # one list of K pixels per non-empty block
        taken: list[tuple[int, int]] = []              # every pixel chosen so far, ALL blocks
        bs = cfg.block_size
        n_short = 0
        # Pooled members come out of the SAME per-block draw as the singles, so they
        # inherit min_pixel_separation and the avoid-set for free. With
        # pool_include_singles a group of k needs only k-1 fresh pixels.
        n_extra = sum(int(k) - bool(cfg.pool_include_singles and cfg.pixels_per_block)
                      for k in cfg.pool_groups)
        want = cfg.pixels_per_block + n_extra
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
                got = _fps_pick(block, want, int(rng.integers(len(block))),
                                avoid=near, min_sep=cfg.min_pixel_separation)
                if not got:
                    continue
                if len(got) < want:
                    n_short += 1          # too few well-separated alive pixels in this block
                picks.append(split_block(got, cfg))
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
        log(f"  selected {len(sel)} targets: {nbx}×{nby} blocks (~{cfg.block_size}px), "
            f"{cfg.pixels_per_block} single(s)/block, {len(picks)} non-empty blocks, "
            f"seed {cfg.block_seed} (randomized-start FPS, round-robin order → "
            f"drift-decorrelated)")
        register_pools(sel, alive, cfg, log)
        # Report the realised spacing — this is the number that says whether the sample is
        # spatially spread or secretly clustered, so it belongs in the run log.
        log("  " + spacing_summary(sel, cfg.min_pixel_separation)
            + (f"; {n_short} block(s) gave fewer than {want} "
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

    The fast scan stores this pair as *_MEAN / *_SEM. The SEM (σ/√N) is the
    uncertainty of the centre — not the raw per-sample scatter. That is the right
    quantity for the analysis reliability gates: averaging more samples (e.g. many SMU
    reads at low J) genuinely lowers it, so the well-averaged low-J points stop being
    rejected. The sweep does not use this pair: it writes burst_stats() of the raw
    samples instead, so its *_MEAN is a true mean (see measure_point).

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
    """(mean, median, std, madstd, sem, n) of a burst — for the aggregated file's columns.

    std is the SAMPLE std (ddof=1). madstd is 1.4826·MAD, the σ a Gaussian burst would
    have, and it ignores the odd autorange spike. sem = std/√n, the uncertainty of the
    mean, which is what the analysis gates and error bars need.
    """
    a = np.asarray([v for v in vals if v is not None and np.isfinite(v)], float)
    if a.size == 0:
        nan = float("nan")
        return dict(mean=nan, median=nan, std=nan, madstd=nan, sem=nan, n=0)
    std = float(a.std(ddof=1)) if a.size > 1 else 0.0
    med = float(np.median(a))
    return dict(mean=float(a.mean()), median=med, std=std,
                madstd=1.4826 * float(np.median(np.abs(a - med))),
                sem=std / math.sqrt(a.size), n=int(a.size))


# burst_stats() key -> column suffix. One block of these per channel in the sweep file.
_STAT_SUFFIX = dict(mean="MEAN", median="MEDIAN", std="SAMPSTD", madstd="MADSTD",
                    sem="SEM", n="N")


def stat_columns(base: str, vals: Iterable[float | None]) -> dict[str, float]:
    """``{base}_MEAN … {base}_N`` for one channel's burst."""
    return {f"{base}_{_STAT_SUFFIX[k]}": v for k, v in burst_stats(vals).items()}


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
    if cfg.fine_sweep_mode == "voltage" and cfg.sweep_voltages:
        out = sorted(((float(v), None) for v in dict.fromkeys(cfg.sweep_voltages)),
                     reverse=(cfg.sweep_direction == "low_to_high"))
        log(f"    {tag}voltage list: {len(out)} biases "
            f"{', '.join(f'{v:g}' for v, _ in out)} V, {cfg.sweep_direction}")
        return out
    if cfg.fine_sweep_mode == "voltage":
        # Voltage ramp — no coarse map. Densify the rising-side/peak window (split at the
        # knee); targets None so the points are taken as-is.
        # The knee only splits a ramp it is INSIDE. With nvled_v_coarse_stop=-1.1 V and
        # the default knee at -1.3 V the first leg overshot the stop and the second came
        # back up: run 261019-1537 asked for -0.8..-1.1 V and measured -0.8..-1.3 V,
        # 8 points over 6 distinct biases with -1.1 and -1.2 duplicated. A knee outside
        # the ramp carries no information about where to spend points, so fall back to
        # uniform and say which happened.
        lo, hi = sorted((cfg.nvled_v_coarse_start, cfg.nvled_v_coarse_stop))
        knee_inside = (cfg.vramp_knee_v is not None and lo < cfg.vramp_knee_v < hi)
        if cfg.vramp_low_frac and 0.0 < cfg.vramp_low_frac < 1.0 and knee_inside:
            n_low = max(2, int(round(cfg.n_fine * cfg.vramp_low_frac)))
            n_high = max(2, cfg.n_fine - n_low)
            V = np.concatenate([
                np.linspace(cfg.nvled_v_coarse_start, cfg.vramp_knee_v, n_low, endpoint=False),
                np.linspace(cfg.vramp_knee_v, cfg.nvled_v_coarse_stop, n_high)])
            dens = (f"split: {n_low} pts {cfg.nvled_v_coarse_start}→{cfg.vramp_knee_v} V (rising), "
                    f"{n_high} pts {cfg.vramp_knee_v}→{cfg.nvled_v_coarse_stop} V (droop)")
        elif cfg.vramp_low_frac and cfg.vramp_knee_v is not None and not knee_inside:
            V = np.linspace(cfg.nvled_v_coarse_start, cfg.nvled_v_coarse_stop, cfg.n_fine)
            dens = (f"uniform {cfg.n_fine} pts {cfg.nvled_v_coarse_start}→"
                    f"{cfg.nvled_v_coarse_stop} V -- vramp_knee_v={cfg.vramp_knee_v} is "
                    f"OUTSIDE this ramp, so it cannot split it")
        else:
            V = np.linspace(cfg.nvled_v_coarse_start, cfg.nvled_v_coarse_stop, cfg.n_fine)
            dens = f"uniform {cfg.n_fine} pts {cfg.nvled_v_coarse_start}→{cfg.nvled_v_coarse_stop} V"
        ramp = [(float(v), None) for v in V]
        out = sorted(ramp, reverse=(cfg.sweep_direction == "low_to_high"))
        log(f"    {tag}voltage ramp ({dens}), "
            f"{'range from the limit (High-C)' if cfg.highc_b else f'fixed {cfg.nvled_range_i:g} A range' if cfg.nvled_range_i else 'range from the limit (High-C off)'}, "
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

# Fast-scan / screen columns. *_SEM is the standard error of the centre (σ/√N), not the
# per-sample spread. Before 2026-10-24 it was written as *_STD; analyze_array renames
# that on load. Here *_MEAN is _stats()'s centre: the MEDIAN for the two currents.
_AGG_KEYS = ["X", "Y", "BITVAL", "NVLED_V", "TIME_START", "TIME_END"]
# QUADRANT (TL/TR/BL/BR, from X and Y) is LAST in the fast-scan file, so a reader that
# counts columns there still finds the old ones where they were. In the sweep, which
# is read by name, it follows the keys.
_AGG_COLS = _AGG_KEYS + ["NVLED_CURR_MEAN", "NVLED_CURR_SEM", "VLED_CURR_MEAN",
                         "VLED_CURR_SEM", "PM400_POWER_MEAN", "PM400_POWER_SEM",
                         "QUADRANT"]


def quadrant_of(x, y) -> str:
    """Which 256x256 quadrant a logical pixel sits in: TL, TR, BL or BR."""
    return ("TL", "TR", "BL", "BR")[(int(x) >= 256) + 2 * (int(y) >= 256)]


def _with_quadrant(r: Row) -> Row:
    """The row with QUADRANT filled in from X and Y, if it does not carry one."""
    return r if "QUADRANT" in r else {**r, "QUADRANT": quadrant_of(r["X"], r["Y"])}
_CHANNELS = ["NVLED_CURR", "VLED_CURR", "PM400_POWER"]
_STATS = list(_STAT_SUFFIX.values())


def _sweep_stat_cols(ref: bool) -> list[str]:
    """The sweep's per-channel blocks, grouped: each channel, then its reference.

    Every block is burst_stats() of the raw samples, so *_MEAN is a true mean here.
    ``{ch}_REF`` is the value the analysis subtracts -- _stats()'s centre, so the median
    for the currents and the mean for the power. It duplicates one of the stats on
    purpose: every reader of the leakage subtraction keys on it.
    """
    cols = []
    for ch in _CHANNELS:
        cols += [f"{ch}_{s}" for s in _STATS]
        if ref:
            cols += [f"{ch}_REF"] + [f"{ch}_REF_{s}" for s in _STATS]
    return cols


# The leakage reference: the same pixel at LEAK_REF_BITVAL, measured back to back with the
# lit point. I_LED = NVLED_CURR_MEDIAN - NVLED_CURR_REF, and that subtraction is what makes
# the rising side meaningful at all -- at -0.80 V the reference is 58% of the reading.
# Its value and stats sit next to each channel (_sweep_stat_cols); these are the rest.
_AGG_COLS_REF = ["LEAK_REF_BITVAL",
                 # how many frame writes it took, and the optical ratio that was
                 # accepted. TRIES > 1 means a write failed and was re-sent; FRAC
                 # above leak_ref_max_frac means it never landed.
                 "LEAK_REF_TRIES", "LEAK_REF_FRAC",
                 # 1 = this point reused the per-pixel reference from the pre-pass
                 # instead of measuring its own. TRIES is 0 on those points, because
                 # no frame write happened. See config.leak_ref_per_point_above_v.
                 "LEAK_REF_CACHED"]
# The compliance actually in force at each point, and whether the point hit it. Written
# for the voltage ramp, where the limit tracks the measured current (see pick_limit).
_AGG_COLS_LIMIT = ["NVLED_LIMIT_A", "NVLED_AT_COMPLIANCE"]
# Only written when pooling is on, so an ordinary run's files do not change at all.
_AGG_COLS_POOL = ["POOL_K", "POOL_MEMBERS"]
# Written only when the range actually moves during a run, so an ordinary file is
# unchanged. Without it a reader cannot tell which scale a point was taken on, and the
# floor differs 5.45x between them.
_AGG_COLS_PMRANGE = ["PM_RANGE_W"]


def agg_cols(cfg: IQEConfig) -> list[str]:
    """The sweep's aggregated-CSV columns for this configuration.

    Grouped by channel and found by NAME: a reader that indexes columns by position
    breaks on this file.
    """
    cols = _AGG_KEYS + ["QUADRANT"] + _sweep_stat_cols(cfg.leak_reference)
    if cfg.leak_reference:
        cols += _AGG_COLS_REF
    if cfg.fine_sweep_mode == "voltage":
        cols += _AGG_COLS_LIMIT
    if cfg.pool_groups:
        cols += _AGG_COLS_POOL
    if cfg.pool_pm_range_below_v is not None or cfg.pm_autorange:
        cols += _AGG_COLS_PMRANGE
    return cols


def agg_field(v) -> str:
    """One aggregated-CSV field.

    ``repr`` for everything numeric, because round-tripping a float through ``str``
    loses digits. A string is written bare: ``repr`` would wrap it in quotes that are
    not CSV quoting, and every reader would carry them into the value. The only string
    column is POOL_MEMBERS, and it uses ":" between coordinates precisely so it carries
    no comma of its own -- this writer joins fields with a comma and does not quote.
    """
    return v if isinstance(v, str) else repr(v)


# The screen's own file: the fast-scan columns plus what makes a group a group.
_SCREEN_COLS = _AGG_COLS + ["POOL_K", "POOL_MEMBERS", "SCREEN_FLAGGED"]


def write_aggregated_one(path: str | Path, rows: list[Row],
                         cols: list[str] | None = None) -> None:
    cols = list(cols or _AGG_COLS)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        f.write(",".join(cols) + "\n")
        for r in rows:
            r = _with_quadrant(r)
            f.write(",".join(agg_field(r[c]) for c in cols) + "\n")


def write_aggregated_per_bitval(folder: str | Path, rows: list[Row],
                                prefix: str = "") -> None:
    folder = Path(folder)
    for b in sorted({r["BITVAL"] for r in rows}):
        write_aggregated_one(folder / f"{prefix}aggregated_bitval={b}_mean_std.csv",
                             [r for r in rows if r["BITVAL"] == b])


RAW_HEADER = "X,Y,BITVAL,NVLED_V,TIME,TYPE,MEAS_VALUE,MEAS_STD\n"


def raw_lines(rows: list[Row], dark: Row | None = None) -> Iterator[str]:
    """The raw-CSV lines for one pixel's point rows (+ optional dark reading).

    Both writers emit this exact format, so the formatting lives here once —
    ``analyze_array`` matches on the TYPE column, and a drift between the two
    writers would silently produce two different raw files.
    """
    for r in rows:
        for typ, mean, std in [("PM400", r["PM400_POWER_MEAN"], r["PM400_POWER_SEM"]),
                               ("VLED", r["VLED_CURR_MEAN"], r["VLED_CURR_SEM"]),
                               ("NVLED", r["NVLED_CURR_MEAN"], r["NVLED_CURR_SEM"])]:
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
                 cfg: IQEConfig, log: Logger, append: bool = False,
                 prefix: str = "") -> None:
        self._prefix = prefix
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
            self._samp_f = open(
                Path(raw_path).parent / f"{prefix}Sweep_samples.csv", mode)
            if not append:
                self._samp_f.write("X,Y,BITVAL,NVLED_V,TIME,TYPE,SAMPLE_IDX,VALUE\n")
                self._samp_f.flush()
        self._agg_f = {}
        self._thread = threading.Thread(target=self._worker, daemon=True)
        self._thread.start()

    def _agg_fh(self, bitval):
        if bitval not in self._agg_f:
            path = self._agg_folder / \
                f"{self._prefix}aggregated_bitval={bitval}_mean_std.csv"
            existing = self._append and path.exists() and path.stat().st_size > 0
            if existing:
                # Appending rows under another header shifts every column silently.
                # A run started by older code must finish on that code.
                with open(path) as old:
                    head = old.readline().strip().split(",")
                if head != self._cols:
                    raise RuntimeError(
                        f"cannot resume {path.name}: its columns differ from what this "
                        f"version writes. Resume it with the version that started it.")
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
                        self._samp_f.write(f"{x},{y},{b},{v},{t},{typ},{k},{float('nan' if val is None else val)!r}\n")
            self._samp_f.flush()
        for r in prows:
            r = _with_quadrant(r)
            self._agg_fh(r["BITVAL"]).write(
                ",".join(agg_field(r.get(c, float("nan"))) for c in self._cols) + "\n")
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
        for f in self._agg_f.values():
            try:
                order_rows_by_pixel(f.name)
            except Exception as e:      # the bias-ordered file is still a valid file
                self._log(f"  could not reorder {Path(f.name).name} by pixel: {e}")


def order_rows_by_pixel(path: str | Path) -> None:
    """Rewrite an aggregated CSV so each pixel's rows sit together, pixels in run order.

    bias_major writes one bias at a time, so the file comes out ordered by bias. It is
    written that way on purpose -- every finished bias is on disk if the run dies --
    and reordered here, once, at the end. The sort is stable: inside a pixel the rows
    keep the order they were measured in. A pixel's place is where it FIRST appears,
    which is the run order. Text in, text out, so no float is re-formatted.
    """
    path = Path(path)
    lines = path.read_text().splitlines(keepends=True)
    if len(lines) < 3:
        return
    rows = lines[1:]
    rank: dict[tuple[str, str], int] = {}
    key = lambda ln: tuple(ln.split(",", 2)[:2])
    for ln in rows:
        rank.setdefault(key(ln), len(rank))
    rows.sort(key=lambda ln: rank[key(ln)])
    tmp = path.with_name(path.name + ".part")
    tmp.write_text(lines[0] + "".join(rows))
    tmp.replace(path)


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


# What the meter reports after configure_common, or 0 if it was never asked. The
# pooled over-range check needs the REALISED scale: measured on run 261019-1031,
# SENS:POW:RANG 1e-7 lands the S171C on 470 nW, so comparing against the request was
# comparing against a number 4.7x too small.
_pm_range_realised = [0.0]

PM_RANGE_REALISED_DEFAULT = 470e-9      # S171C, from a pm_range=1e-7 request


def configure_common(pm, smu, cfg: IQEConfig, log: Logger) -> None:
    """PM400 + ch A (voltage anode) + buffers. ch B is configured per source mode."""
    pm.set_wavelength(cfg.pm_wavelength)
    pm.set_power_unit("W")
    pm.set_auto_range(False)
    pm.set_range(cfg.pm_range)
    pm.set_averaging(cfg.pm_averaging)
    if hasattr(pm, "inst"):
        pm.inst.write("INP:FILT 0")            # fast response (no low-pass settling)
        # pm_range is a REQUEST: SENS:POW:RANG picks the smallest hardware range whose
        # full scale covers it, so the realised scale is not cfg.pm_range. Measured on
        # 261016-2252: single pixels read up to 257 nW with pm_range=1e-7 and no sign of
        # clipping. Nothing recorded which range that was, so a pooled target had no way
        # to know its headroom. Log it.
        try:
            got = float(pm.inst.query("SENS:POW:RANG?"))
            _pm_range_realised[0] = got
            log(f"  PM400 range {got*1e9:.0f} nW (autorange off, requested "
                f"{cfg.pm_range*1e9:.0f} nW)")
        except Exception as e:                 # a query failure must not end a 30 h run
            log(f"  PM400 range query failed ({e}); headroom unknown")
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


def range_for_limit(limit: float) -> float:
    """The smallest standard current range that holds ``limit``."""
    return next((r for r in sorted(SMU_I_RANGES) if r >= abs(limit) * (1 - 1e-9)),
                max(SMU_I_RANGES))


def measure_range_b(cfg: IQEConfig) -> float:
    """ch B's measure range for the current limit: the limit's own range, as High-C sets
    it, unless ``nvled_range_i`` fixes one (and High-C is off). One rule for both modes."""
    if cfg.nvled_range_i and not cfg.highc_b:
        return cfg.nvled_range_i
    return range_for_limit(cfg.nvled_compliance)


def pick_limit(i_last: float | None, cfg: IQEConfig,
               v_from: float | None = None, v_to: float | None = None) -> float:
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
    Without High-C the range now follows the limit as well (``measure_range_b``).

    With ``v_from``/``v_to`` it PREDICTS: the last current is scaled by the median I(V)
    ratio between the two biases (``bias_scale``, measured over 1200 pixels) before the
    headroom. Going down in bias the range then steps down at the bias that needs it,
    not one bias late; going up (``low_to_high``) the rise is covered instead of left to
    the 1.5x headroom, which a 0.03 V step on the rising side (x1.4) nearly uses up.
    """
    # Between vramp_leak_above_v and 0 V the LED is off and only the leak flows, so the
    # floor drops a decade there -- without High-C only, where the limit is not also
    # the current that charges the array.
    floor = (cfg.vramp_leak_limit_a
             if (not cfg.highc_b and v_to is not None and v_to >= cfg.vramp_leak_above_v)
             else cfg.vramp_lowv_limit_a)
    ladder = [r for r in SMU_I_RANGES if floor <= r <= cfg.vramp_highv_limit_a]
    if not ladder:
        return cfg.vramp_highv_limit_a
    if i_last is None or not math.isfinite(i_last):
        return ladder[-1]                      # first point: widest, nothing is known yet
    scale = 1.0
    if v_from is not None and v_to is not None and v_from != v_to:
        scale = bias_scale(v_to, v_from)[1]
    need = 1.5 * abs(i_last) * scale
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
                          zero_delays=not cfg.highc_b, range_i=measure_range_b(cfg))
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


def acquire_autoranged(pm, smu, n_smu: int, n_pm: int, cfg: IQEConfig,
                       return_raw: bool = False):
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
    return acquire_point(pm, smu, n_smu, n_pm, return_raw=return_raw)


def configure_current_source_b(smu, cfg: IQEConfig, level_i: float = 0.0) -> None:
    """ch B = current source / voltage measure, with a voltage compliance.

    ``level_i`` is what the source is set to AS the mode changes. It used to be a
    hard 0, which hands the node to a 0 A source: it then floats to wherever its own
    junction and leakage put it, and how far it gets before the caller sets the real
    setpoint depends entirely on the capacitance. On the 43 uF carrier that drift was
    6 mV and nobody noticed for months. On a de-capped one, 130x less charge is needed
    and the node arrives at its equilibrium immediately -- Array38 parked at -0.400 V
    and began its ramp at +0.326 V, and parked at -1.050 V began at +0.299 V. Same
    landing from parks 650 mV apart, which is the signature of a float, not a kick.

    Callers that want the old behaviour get it by leaving level_i at 0.
    """
    if hasattr(smu, "inst"):
        smu.inst.write("smub.source.func = smub.OUTPUT_DCAMPS")
        smu.inst.write(f"smub.source.limitv = {abs(cfg.nvled_v_limit)}")
        smu.inst.write(f"smub.source.leveli = {float(level_i)}")
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
        self._n_lit = 1           # pixels lit by the current frame -- see _SimPM.measure
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
    def set_range(self, r, *a): self._req_range = float(r)
    def get_range(self): return getattr(self, "_req_range", 1e-7)
    def set_averaging(self, *a): pass
    def display_off(self): pass
    def display_on(self): pass
    def close(self): pass

    def measure(self):
        s = self._smu
        I = s._led_current()
        if not s._on or I <= 0:
            return abs(self._rng.normal(0, 2e-12))
        # Over the LIT area, not one pixel's. K pixels at the same bias each sit at
        # the same J; dividing K x the current by one pixel's area put them K x up the
        # ABC droop curve and made a pooled target emit less than its members' sum.
        # The new additivity_p column in --what cluster is what caught it.
        J = I / (getattr(s, "_n_lit", 1) * self._cfg.pixel_area_um2 * 1e-8)
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

    @staticmethod
    def _gain(x, y):
        h = (1103515245 * (x * 1009 + y) + 12345) & 0x7fffffff
        return 0.0 if (h % 13 == 0) else 0.6 + 0.7 * ((h >> 8) % 1000) / 1000.0

    def set_pixel(self, x, y, bit, enables=None):
        self._smu._on = enables != "none"      # a disabled quadrant does not light
        self._smu._pixel_gain = self._gain(x, y)
        self._smu._n_lit = 1
        self._smu._bit_scale = max(1, int(bit)) / 15.0
        return time.perf_counter()

    def set_pixels(self, coords, bit):
        """K pixels in parallel: the gains add, which is the whole point of a cluster.

        Pooled EQE = sum(P)/sum(I), so a correct sim only has to make both sums
        scale with K. The sim has no cluster-specific physics beyond that.
        """
        coords = [(int(x), int(y)) for x, y in coords]
        if not coords:
            raise ValueError("coords is empty")
        self._smu._on = True
        self._smu._pixel_gain = sum(self._gain(x, y) for x, y in coords)
        self._smu._n_lit = len(coords)
        self._smu._bit_scale = max(1, int(bit)) / 15.0
        return time.perf_counter()

    def blank_frame(self, enables=None):
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

def local_median(a: np.ndarray, w: int) -> np.ndarray:
    """Median over a w x w neighbourhood of ``a``, edges padded by repetition.

    numpy only -- scipy is an optional extra for the analysis side and the measurement
    side must run without it.
    """
    r = w // 2
    win = np.lib.stride_tricks.sliding_window_view(np.pad(a, r, mode="edge"), (w, w))
    return np.median(win, axis=(2, 3))


def pool_tiles(x1: int, y1: int, x2: int, y2: int, k: int):
    """Tile the ROI into k-pixel groups, each inside ONE quadrant.

    Square tiles when k is a perfect square, 1 x k strips otherwise -- that is the
    shape the replay in ``fastscan_pool`` measured. Returns (tiles, leftover) where
    ``tiles`` is a list of (quadrant_index, iy, ix, members) and ``leftover`` is every
    pixel the tiling could not cover, which the caller scans one by one.

    One quadrant per tile, because the tl_pixen/tr_pixen/... enables live in the frame
    instruction and not in the pixel data: a tile across 256 would light the near
    quadrant's mirror image instead of the far member.
    """
    side = int(round(k ** 0.5))
    h, w = (side, side) if side * side == k else (1, k)
    tiles, leftover = [], []
    qi = 0
    for qx0, qx1 in ((x1, min(x2, 255)), (max(x1, 256), x2)):
        for qy0, qy1 in ((y1, min(y2, 255)), (max(y1, 256), y2)):
            if qx0 > qx1 or qy0 > qy1:
                continue
            nx, ny = (qx1 - qx0 + 1) // w, (qy1 - qy0 + 1) // h
            for iy in range(ny):
                for ix in range(nx):
                    tiles.append((qi, iy, ix, tuple(
                        (qx0 + ix * w + dx, qy0 + iy * h + dy)
                        for dy in range(h) for dx in range(w))))
            # The far edge of each quadrant, where a whole tile does not fit: two
            # strips, derived from the tile counts rather than by differencing sets.
            leftover += [(px, py) for px in range(qx0 + nx * w, qx1 + 1)
                         for py in range(qy0, qy1 + 1)]
            leftover += [(px, py) for px in range(qx0, qx0 + nx * w)
                         for py in range(qy0 + ny * h, qy1 + 1)]
            qi += 1
    return tiles, leftover


def screen_fast_scan(pm, smu, smile, x1: int, y1: int, x2: int, y2: int,
                     cfg: IQEConfig, log: Logger, t0: float):
    """Group-test screen: light K pixels at once, rescan only the suspicious groups.

    Returns (order, rows): the pixels still to be scanned one by one, and one row per
    group. A group row carries POOL_K, POOL_MEMBERS and SCREEN_FLAGGED, and its power
    and current are the SUM over its members.

    The flag is taken against a LOCAL median (fastscan_pool_detrend), because the array
    has a real +3.7 % gradient and a row-driver signature: a global median flags one
    corner wholesale. Replayed on run 261016-2252's 262 144 pixels, detrending lifts
    K=4 recall from 85.5 % to 91.8 % at the same bench cost.

    This is a SCREEN. The groups it does not flag are never measured per pixel, so the
    missed fraction is real -- 8.2 % of the dead pixels at K=4 on that run -- and the
    log says so every time.
    """
    k = int(cfg.fastscan_pool)
    tiles, leftover = pool_tiles(x1, y1, x2, y2, k)
    if not tiles:
        log(f"  screen: no {k}-pixel tile fits in the ROI -- scanning every pixel")
        return list(roi_pixels(x1, y1, x2, y2)), []
    # The screen runs BEFORE the bias loop, so it has to set its own bias and pay its
    # own settle. Reading at whatever ch B happened to hold would flag groups by where
    # the source was left, not by how bright they are.
    v = cfg.fastscan_v_list()[0]
    smu.set_voltage("b", float(v))
    time.sleep(max(cfg.fastscan_settle_ms, cfg.thermal_settle_ms) / 1000.0)
    log(f"  screen @ {v:g} V: {len(tiles)} group(s) of {k}"
        + (f" plus {len(leftover)} edge pixel(s) scanned singly" if leftover else "")
        + f", rescanning the dimmest {cfg.fastscan_pool_rescan_frac*100:g}% against a "
          f"{cfg.fastscan_pool_detrend}-group local median")
    rows = []
    for (_qi, _iy, _ix, members) in tiles:
        if cfg.fastscan_blank_between:
            smile.blank_frame()
        smile.set_pixels(members, cfg.fastscan_bitval)
        if cfg.fastscan_settle_ms:
            time.sleep(cfg.fastscan_settle_ms / 1000.0)
        ts = time.perf_counter() - t0
        p, ps, iv, ivs, inv, invs, _, _ = acquire_point(pm, smu, cfg.fastscan_n_smu, 3)
        rows.append(dict(X=members[0][0], Y=members[0][1], BITVAL=cfg.fastscan_bitval,
                         NVLED_V=v, TIME_START=ts, TIME_END=time.perf_counter() - t0,
                         NVLED_CURR_MEAN=inv, NVLED_CURR_SEM=invs,
                         VLED_CURR_MEAN=iv, VLED_CURR_SEM=ivs,
                         PM400_POWER_MEAN=p, PM400_POWER_SEM=ps,
                         POOL_K=len(members),
                         POOL_MEMBERS=" ".join(f"{mx}:{my}" for mx, my in members)))
    # Detrend inside each quadrant's own tile grid, then flag on one global quantile so
    # the rescan budget is the budget the operator asked for.
    rel = np.ones(len(rows))
    for qi in {t[0] for t in tiles}:
        idx = [i for i, t in enumerate(tiles) if t[0] == qi]
        ny = max(tiles[i][1] for i in idx) + 1
        nx = max(tiles[i][2] for i in idx) + 1
        grid = np.full((ny, nx), np.nan)
        for i in idx:
            grid[tiles[i][1], tiles[i][2]] = abs(rows[i]["PM400_POWER_MEAN"])
        ref = local_median(grid, min(cfg.fastscan_pool_detrend, max(3, 2 * (min(ny, nx) // 2) - 1)))
        with np.errstate(invalid="ignore", divide="ignore"):
            r = np.where(ref > 0, grid / ref, 1.0)
        for i in idx:
            rel[i] = r[tiles[i][1], tiles[i][2]]
    # The rule is physical, and the budget is only a CAP on it. A dead member is a
    # 1/K deficit on the group sum, so anything at or below half a member's worth of
    # deficit is worth a closer look. A FIXED quantile was the first version and it is
    # wrong: on an array with a high dead rate it rescans the budget and silently
    # ignores the rest, which is the one failure a screen must not have quietly.
    cut = 1.0 - 0.5 / k
    want = rel <= cut
    cap = max(1, int(np.ceil(cfg.fastscan_pool_rescan_frac * len(rows))))
    capped = int(want.sum()) > cap
    if capped:
        thr = float(np.partition(rel, cap - 1)[cap - 1])
        flag = rel <= thr
    else:
        thr, flag = cut, want
    order = list(leftover)
    for i, t in enumerate(tiles):
        rows[i]["SCREEN_FLAGGED"] = int(flag[i])
        if flag[i]:
            order += list(t[3])
    n_flag = int(flag.sum())
    npx = (x2 - x1 + 1) * (y2 - y1 + 1)
    log(f"  screen: {n_flag}/{len(tiles)} group(s) flagged at rel <= {thr:.3f} "
        f"(physical cut {cut:.3f} = half a member; group spread "
        f"{np.nanstd(rel)*100:.1f}% after detrending) -> {len(order)} of {npx} "
        f"pixel(s) rescanned singly, {len(tiles) + len(order)} test(s) instead of "
        f"{npx} ({npx / max(1, len(tiles) + len(order)):.2f}x)")
    if capped:
        log(f"  !! THE RESCAN CAP IS BINDING: {int(want.sum())} group(s) are below the "
            f"physical cut but only {cap} fit in "
            f"fastscan_pool_rescan_frac={cfg.fastscan_pool_rescan_frac:g}. The screen "
            f"is looking at an array far dimmer or far deader than the 99.57 % yield "
            f"it was sized for, so its recall here is LOW and unknown. Raise "
            f"fastscan_pool_rescan_frac to "
            f"{min(1.0, round(1.3 * want.sum() / len(rows), 2)):g}, lower "
            f"fastscan_pool, or set fastscan_pool=0 and scan every pixel.")
    log(f"  !! the {len(tiles) - n_flag} unflagged group(s) are NOT measured per pixel. "
        f"Their members enter the sweep selection with the group's per-member estimate "
        f"and SCREENED=1. Replayed on run 261016-2252 at 99.57 % yield this shape "
        f"missed 8.2 % of the dead pixels at K=4. Set fastscan_pool=0 for a census.")
    return order, rows


_DEAD_IV_COLS = ["X", "Y", "STATE", "NVLED_V", "NVLED_CURR_MEAN", "NVLED_CURR_SEM",
                 "PM400_POWER_MEAN", "PM400_POWER_SEM", "SCAN_POWER_W",
                 "R_OHM", "R_FIT_R2", "I_OFFSET_A", "AT_COMPLIANCE"]


def fit_resistance(volts, currents) -> tuple[float, float, float]:
    """Least-squares R from a sub-turn-on IV, as (R_ohm, r2, I_offset_A).

    The offset is fitted rather than forced through the origin: the SMU has a range
    zero-offset of a few nA, and forcing the line through (0, 0) folds that straight
    into the slope. At 1 nA of offset over a 0.5 V sweep that is a 500 Mohm error --
    irrelevant against a short, fatal against a healthy pixel.

    Returns R = inf when the slope is zero or negative, which is what a pixel that is
    genuinely not conducting looks like. inf is the honest answer there, not a huge
    number that invites being averaged.
    """
    v = np.asarray(volts, float)
    i = np.asarray(currents, float)
    ok = np.isfinite(v) & np.isfinite(i)
    if ok.sum() < 3:
        return float("nan"), float("nan"), float("nan")
    v, i = v[ok], i[ok]
    slope, offset = np.polyfit(v, i, 1)
    var = float(np.var(i))
    r2 = 1.0 - float(np.var(i - (slope * v + offset))) / var if var > 0 else float("nan")
    # "Not conducting" has to be decided against the noise, not against zero. A healthy
    # pixel reads a flat line whose least-squares slope is some denormal positive
    # number, and 1/that is ~1e24 ohm -- a value that looks like a measurement and
    # survives being averaged. The current has to swing by more than the scatter it was
    # measured through before a slope means anything.
    swing = abs(float(slope) * float(np.ptp(v)))
    noise = float(np.std(i - (slope * v + offset)))
    if slope <= 0 or swing <= 3.0 * noise or np.ptp(i) == 0:
        return float("inf"), r2, float(offset)
    return 1.0 / float(slope), r2, float(offset)


def measure_dead_iv(pm, smu, smile, dead: list[Row], alive: list[Row],
                    run_dir: Path, cfg: IQEConfig, log: Logger) -> list[Row]:
    """Sweep a few dead pixels sub-turn-on and fit a resistance. And the same alive.

    A dead pixel is normally shorted, so below turn-on it is a RESISTOR while a healthy
    one is a reverse-biased diode drawing almost nothing. That difference shows up in a
    handful of points and needs none of the sweep's machinery -- no High-C, no settling
    ladder, no power meter range dance. The meter is still read, because a pixel that
    conducts AND emits is a third thing worth seeing.

    An equal number of alive pixels get the identical sweep. A resistance on its own
    says very little; the same number measured the same way on pixels known to work is
    what turns it into a comparison.

    The dead pixels are the DARKEST ones, not merely those under the alive threshold.
    A pixel sitting just below the threshold is still emitting, so it still has a
    working junction and would return a diode curve. `dead_iv_dark_sigma` keeps only
    pixels whose fast-scan power is within that many sigma of dark, sigma being the
    median per-pixel burst std of the scan -- the meter's own noise, measured in the
    same run rather than assumed.
    """
    n = cfg.dead_iv_pixels
    sigma = float(np.nanmedian([abs(r.get("PM400_POWER_SEM") or np.nan)
                                for r in dead] or [np.nan]))
    cut = cfg.dead_iv_dark_sigma * sigma
    cands = [r for r in dead if abs(r["PM400_POWER_MEAN"]) <= cut] if np.isfinite(cut) \
        else []
    if len(cands) < n:
        # Say what was relaxed and by how much, rather than quietly sweeping pixels
        # that are dim instead of dead.
        darkest = sorted(dead, key=lambda r: abs(r["PM400_POWER_MEAN"]))[:n]
        log(f"  dead IV: only {len(cands)} pixel(s) within {cfg.dead_iv_dark_sigma:g} "
            f"sigma of dark ({cut * 1e9:.3f} nW); taking the {len(darkest)} darkest "
            f"instead, up to {abs(darkest[-1]['PM400_POWER_MEAN']) * 1e9:.3f} nW"
            if darkest else "  dead IV: no dead pixels at all")
        cands = darkest
    # Spread them out. Neighbouring dead pixels are usually one defect, so sweeping
    # four of them measures that defect four times.
    picked = _fps_pick(cands, n, 0, min_sep=cfg.min_pixel_separation) if cands else []
    refs = _fps_pick(alive, min(n, len(alive)), 0,
                     min_sep=cfg.min_pixel_separation) if alive else []
    if not picked and not refs:
        log("  dead IV: nothing to measure")
        return []

    log(f"  dead IV: {len(picked)} dead + {len(refs)} alive over "
        f"{', '.join(f'{v:g}' for v in cfg.dead_iv_volts)} V "
        f"(limit {cfg.dead_iv_limit_a:g} A, NPLC {cfg.dead_iv_nplc:g}, no High-C)")
    # Its own limit and no High-C: this is a resistance measurement, and High-C exists
    # to settle a capacitive node that a short does not have.
    configure_voltage_source_b(smu, dataclasses.replace(
        cfg, nvled_compliance=cfg.dead_iv_limit_a, highc_b=False), log)
    set_nplc(smu, cfg.dead_iv_nplc)
    rows: list[Row] = []
    try:
        for state, group in (("dead", picked), ("alive", refs)):
            for r in group:
                x, y = r["X"], r["Y"]
                smile.blank_frame()
                smile.set_pixel(x, y, cfg.fastscan_bitval)
                vs, is_ = [], []
                here: list[Row] = []
                for v in cfg.dead_iv_volts:
                    smu.set_voltage("b", float(v))
                    time.sleep(max(cfg.fastscan_settle_ms, 20) / 1000.0)
                    p, ps, _, _, inv, invs, _, _ = acquire_point(
                        pm, smu, cfg.fastscan_n_smu, 3)
                    vs.append(v)
                    is_.append(inv)
                    here.append(dict(X=x, Y=y, STATE=state, NVLED_V=v,
                                     NVLED_CURR_MEAN=inv, NVLED_CURR_SEM=invs,
                                     PM400_POWER_MEAN=p, PM400_POWER_SEM=ps,
                                     SCAN_POWER_W=r["PM400_POWER_MEAN"]))
                res, r2, off = fit_resistance(vs, is_)
                at_lim = any(abs(i) >= 0.95 * cfg.dead_iv_limit_a
                             for i in is_ if i is not None and np.isfinite(i))
                for row in here:
                    row.update(R_OHM=res, R_FIT_R2=r2, I_OFFSET_A=off,
                               AT_COMPLIANCE=int(at_lim))
                rows.extend(here)
                shown = ("inf (not conducting)" if res == float("inf")
                         else f"{res:.4g} ohm" if np.isfinite(res) else "n/a")
                log(f"    {state:5} ({x},{y})  R={shown}  r2={r2:.3f}"
                    f"  scan P={abs(r['PM400_POWER_MEAN']) * 1e9:7.3f} nW"
                    + ("  << AT COMPLIANCE, R is an upper bound" if at_lim else ""))
    finally:
        smile.blank_frame()
        smu.set_voltage("b", float(cfg.fastscan_v_list()[0]))
        configure_voltage_source_b(smu, dataclasses.replace(
            cfg, nvled_compliance=cfg.fastscan_limit_a), log)
        set_nplc(smu, cfg.fastscan_nplc)
    if rows:
        pfx = run_slug(cfg.sample_name, run_dir) + "_"
        write_aggregated_one(run_dir / "DeadIV" / f"{pfx}dead_iv.csv", rows,
                             cols=_DEAD_IV_COLS)
        med = {s: np.nanmedian([r["R_OHM"] for r in rows if r["STATE"] == s] or [np.nan])
               for s in ("dead", "alive")}
        log(f"  dead IV: median R  dead {med['dead']:.4g} ohm   "
            f"alive {med['alive']:.4g} ohm")
    return rows


def run_fast_scan(pm, smu, smile, x1: int, y1: int, x2: int, y2: int,
                  run_dir: Path, cfg: IQEConfig, log: Logger) -> list[Row]:
    volts = cfg.fastscan_v_list()
    _pool_members.clear()      # a fresh selection owns the pooled groups
    npx = (x2 - x1 + 1) * (y2 - y1 + 1)
    log("=" * 56)
    log(f"FAST SCAN @ NVLED={', '.join(f'{v:g}' for v in volts)} V over {npx} pixels"
        + (f" ({len(volts)} biases x {npx} = {len(volts) * npx} points)"
           if len(volts) > 1 else ""))
    # The fast scan is a DIFFERENT operating point from the sweep and needs its own
    # compliance. The sweep's rising-side limit clamped every pixel in run 261014-0048.
    # One limit for the whole list: 1e-3 A spans -0.8 to -2.6 V, so the biases share a
    # range and nothing is reconfigured between them.
    configure_voltage_source_b(
        smu, dataclasses.replace(cfg, nvled_compliance=cfg.fastscan_limit_a), log)
    set_nplc(smu, cfg.fastscan_nplc)
    fs_rows = []
    re_rows: list[Row] = []         # the fastscan_remeasure_n repeats, kept apart
    t0 = time.perf_counter()
    screen_rows: list[Row] = []
    if cfg.fastscan_pool:
        if cfg.pm_autorange:
            # K pixels lit at once emit ~K x one pixel: range the screen for that.
            smu.set_voltage("b", float(volts[0]))
            time.sleep(max(cfg.fastscan_settle_ms, cfg.thermal_settle_ms) / 1000.0)
            probe_pm_range(pm, smu, smile, list(roi_pixels(x1, y1, x2, y2)), cfg, log,
                           cfg.fastscan_bitval, cfg.fastscan_pm_probe_n,
                           scale=cfg.fastscan_pool, tag=" for the group screen")
        # The screen narrows `order` once, at volts[0]. Extra biases then cover the
        # same narrowed set, which is the only coherent choice: a bias pass over
        # pixels the screen never kept would have nothing to compare against.
        order, screen_rows = screen_fast_scan(pm, smu, smile, x1, y1, x2, y2,
                                              cfg, log, t0)
    else:
        order = list(roi_pixels(x1, y1, x2, y2))
    if cfg.fastscan_random_order:
        np.random.default_rng(cfg.fastscan_order_seed).shuffle(order)
        log(f"  visiting {len(order)} pixels in randomized order (seed {cfg.fastscan_order_seed})")
    # Bias on the outside, pixels on the inside: the bias is set once per pass instead
    # of once per pixel, and each pass keeps the single-bias scan's exact sequence --
    # same randomized order, same blank-then-light, same settle. volts[0] runs first,
    # so a one-bias run is byte-for-byte what it was before this loop existed.
    for v_fs in volts:
        smu.set_voltage("b", float(v_fs))   # fast scan always voltage-drives ch B
        if len(volts) > 1:
            # The pass change moves the bias on a blanked array; give it the same
            # settle a pixel switch gets before the first pixel is judged.
            time.sleep(max(cfg.fastscan_settle_ms, cfg.thermal_settle_ms) / 1000.0)
            log(f"  pass {volts.index(v_fs) + 1}/{len(volts)} @ {v_fs:g} V")
        # Every bias pass picks its own PM400 range: from -2.6 V to -0.8 V the light
        # spans three decades, and no one range is both on scale at the bright end and
        # quiet at the faint one.
        pm_rng = None
        if cfg.pm_autorange:
            time.sleep(max(cfg.fastscan_settle_ms, cfg.thermal_settle_ms) / 1000.0)
            pm_rng = probe_pm_range(pm, smu, smile, order, cfg, log, cfg.fastscan_bitval,
                                    cfg.fastscan_pm_probe_n, tag=f" @ {v_fs:+.3f} V")
        # The first fastscan_remeasure_n pixels come round again at the end of the
        # SAME loop -- no pause, no second scan -- so the pair differs only in time.
        n_re = min(max(0, int(cfg.fastscan_remeasure_n)), len(order))
        for i, (x, y) in enumerate(order + order[:n_re]):
            if cfg.fastscan_blank_between:
                smile.blank_frame()         # black frame → guarantee only this pixel emits
            smile.set_pixel(x, y, cfg.fastscan_bitval)
            if cfg.fastscan_settle_ms:
                time.sleep(cfg.fastscan_settle_ms / 1000.0)
            ts = time.perf_counter() - t0
            p, ps, iv, ivs, inv, invs, _, _ = acquire_point(pm, smu, cfg.fastscan_n_smu, 3)
            # Brighter than the probe saw: up a range, re-read, and stay up for the
            # rest of the pass rather than flap back and forth.
            while pm_rng is not None and pm_over(p, pm_rng, cfg):
                up = pm_range_up(pm, smu, pm_rng, cfg.fastscan_n_smu, 3, cfg, log)
                if up == pm_rng:
                    break
                pm_rng = up
                ts = time.perf_counter() - t0
                p, ps, iv, ivs, inv, invs, _, _ = acquire_point(
                    pm, smu, cfg.fastscan_n_smu, 3)
            (fs_rows if i < len(order) else re_rows).append(
                dict(X=x, Y=y, BITVAL=cfg.fastscan_bitval, NVLED_V=v_fs,
                     TIME_START=ts, TIME_END=time.perf_counter() - t0,
                     NVLED_CURR_MEAN=inv, NVLED_CURR_SEM=invs,
                     VLED_CURR_MEAN=iv, VLED_CURR_SEM=ivs,
                     PM400_POWER_MEAN=p, PM400_POWER_SEM=ps,
                     **({"PM_RANGE_W": pm_rng} if pm_rng is not None else {})))
    # Every file a run writes carries <sample>_<stamp>_, so a CSV that is copied out of
    # its directory can still be traced back to the run that made it.
    pfx = run_slug(cfg.sample_name, run_dir) + "_"
    # One aggregated file per bias. A single-bias scan keeps the name it always had;
    # with several, each file says its bias, so no reader gets two rows per pixel
    # from one file. raw_data/FastScan.csv stays one long-format file with NVLED_V.
    fs_cols = _AGG_COLS + (["PM_RANGE_W"] if cfg.pm_autorange else [])
    for v_fs in volts:
        tag = "" if len(volts) == 1 else f"_v={v_fs:+.3f}"
        write_aggregated_one(
            run_dir / "FastScan"
            / f"{pfx}aggregated_bitval={cfg.fastscan_bitval}{tag}_mean_std.csv",
            [r for r in fs_rows if r["NVLED_V"] == v_fs], cols=fs_cols)
    raw = IncrementalRaw(run_dir / "FastScan" / "raw_data" / f"{pfx}FastScan.csv")
    raw.append(fs_rows); raw.close()
    if re_rows:
        # "remeasure_", not "aggregated_": the fast-scan globs must not pick it up
        for v_fs in volts:
            tag = "" if len(volts) == 1 else f"_v={v_fs:+.3f}"
            write_aggregated_one(
                run_dir / "FastScan"
                / f"{pfx}remeasure_bitval={cfg.fastscan_bitval}{tag}_mean_std.csv",
                [r for r in re_rows if r["NVLED_V"] == v_fs], cols=fs_cols)
        raw = IncrementalRaw(run_dir / "FastScan" / "raw_data"
                             / f"{pfx}FastScan_remeasure.csv")
        raw.append(re_rows); raw.close()
        log(f"  re-measured the first {len(re_rows) // len(volts)} pixel(s) of each "
            f"bias at the end of its pass -> remeasure_bitval=*.csv")
    if screen_rows:
        write_aggregated_one(
            run_dir / "FastScan"
            / f"{pfx}screen_bitval={cfg.fastscan_bitval}_mean_std.csv", screen_rows,
            cols=_SCREEN_COLS)
    (run_dir / "FastScan" / "raw_data" / f"{pfx}FastScan_config.json").write_text(
        json.dumps({"settings": cfg.to_dict(), "provenance": provenance()}, indent=2),
        encoding="utf-8")

    # Classification and selection read volts[0] ONLY. Mixing passes would compare a
    # pixel at -1.05 V against the -1.35 V population and call half the array dead, and
    # it would make the swept pixel set depend on which extra biases were requested.
    sel_rows = [r for r in fs_rows if r["NVLED_V"] == volts[0]]
    # A screened group that was NOT flagged is never measured per pixel, so its members
    # enter the population with the group's PER-MEMBER estimate and SCREENED=1 saying
    # so. They have to be here: the measured rows are the flagged, dim ones, and a
    # percentile over that biased sample would put the alive threshold in the wrong
    # place entirely.
    for r in screen_rows:
        if r["SCREEN_FLAGGED"]:
            continue
        k = int(r["POOL_K"])
        for m in r["POOL_MEMBERS"].split():
            mx, my = (int(c) for c in m.split(":"))
            sel_rows.append(dict(X=mx, Y=my, BITVAL=cfg.fastscan_bitval,
                                 NVLED_V=volts[0],
                                 NVLED_CURR_MEAN=r["NVLED_CURR_MEAN"] / k,
                                 VLED_CURR_MEAN=r["VLED_CURR_MEAN"] / k,
                                 PM400_POWER_MEAN=r["PM400_POWER_MEAN"] / k,
                                 SCREENED=1))
    P = np.array([abs(r["PM400_POWER_MEAN"]) for r in sel_rows])
    med = np.nanmedian(P)
    # "Typical lit" reference = a HIGH percentile, so the threshold survives a majority-dead ROI
    # (the median falls to the noise floor there → 20%×median ≈ noise → nearly everything passes).
    bright = float(np.nanpercentile(P, cfg.fastscan_alive_pctl))
    thr = cfg.dead_threshold_frac * bright
    alive = [r for r, p in zip(sel_rows, P) if p >= thr]
    if screen_rows:
        n_est = sum(1 for r in sel_rows if r.get("SCREENED"))
        log(f"  classification population: {len(sel_rows) - n_est} measured pixel(s) "
            f"plus {n_est} screened estimate(s)")
    log(f"  alive {len(alive)}/{len(sel_rows)} @ {volts[0]:g} V "
        f"(lit~{bright*1e9:.1f} nW @ p{cfg.fastscan_alive_pctl}, "
        f"thr {thr*1e9:.1f} nW, median {med*1e9:.2f} nW)")
    if med < thr:
        log(f"  NOTE: ROI is majority-dark (median {med*1e9:.2f} nW < threshold {thr*1e9:.1f} nW) — "
            f"most of this region is dead; block selection draws from the lit minority.")
    smile.blank_frame()      # never hand the next stage a lit pixel
    if cfg.dead_iv_pixels:
        # Here, not later: this is the only point where the dead population exists.
        # select_sweep_pixels keeps the alive ones and the dead are never named again.
        dead = [r for r, p in zip(sel_rows, P) if p < thr]
        measure_dead_iv(pm, smu, smile, dead, alive, run_dir, cfg, log)
    if cfg.fastscan_only:
        return alive
    sel = select_sweep_pixels(alive, cfg, log)
    # Pooled groups come AFTER the rotation, deliberately: sweep_rotate exists to move a
    # different single pixel into slot 0 (section 17, open question 13), and a pooled
    # target in that slot would answer a different question.
    if cfg.sweep_rotate and sel:
        k = cfg.sweep_rotate % len(sel)
        sel = sel[k:] + sel[:k]
        log(f"  sweep_rotate={cfg.sweep_rotate}: same {len(sel)} pixels, order rotated by "
            f"{k}. Slot 0 is now ({sel[0]['X']},{sel[0]['Y']}).")
    return sel


# The median pixel's power and current against bias, over the 1200 swept pixels of run
# 261016-2252. The selection has a fast-scan reading at nvled_v_fastscan and needs the
# reading at the sweep's STOP, which it does not have, so it scales by the ratio of these
# two. A single fixed -2.60 V ratio was the first version and it is wrong for any sweep
# that stops short: run 261019-1537 stopped at -1.1 V and was told its groups would emit
# 757 nW when they measured 188 nW.
#   (bias V, median |P| nW, median |I| uA)
BIAS_CURVE = [
    (-2.6000, 147.7, 117.5), (-2.4818, 146.5, 116.2), (-2.3636, 144.9, 114.8),
    (-2.2455, 142.7, 112.5), (-2.1273, 138.7, 108.7), (-2.0091, 132.0, 102.0),
    (-1.8909, 121.7, 92.24), (-1.7727, 108.7, 80.28), (-1.6545, 93.82, 67.10),
    (-1.5364, 77.52, 53.56), (-1.4182, 60.58, 40.25), (-1.3000, 43.51, 27.88),
    (-1.2722, 39.53, 25.18), (-1.2444, 35.69, 22.54), (-1.2167, 31.95, 20.03),
    (-1.1889, 28.28, 17.63), (-1.1611, 24.73, 15.32), (-1.1333, 21.37, 13.16),
    (-1.1056, 18.16, 11.17), (-1.0778, 15.16, 9.332), (-1.0500, 12.40, 7.665),
    (-1.0222, 9.870, 6.155), (-0.9944, 7.642, 4.849), (-0.9667, 5.704, 3.727),
    (-0.9389, 4.064, 2.782), (-0.9111, 2.721, 2.006), (-0.8833, 1.682, 1.392),
    (-0.8556, 0.9446, 0.9385), (-0.8278, 0.4632, 0.6326), (-0.8000, 0.2031, 0.4523),
]


def bias_scale(v_to: float, v_from: float) -> tuple[float, float]:
    """(power, current) scale factors from bias ``v_from`` to bias ``v_to``.

    Log-interpolated on BIAS_CURVE, because both quantities span three decades over the
    sweep and a linear interpolation between -0.9 V and -1.3 V would be badly wrong in
    the middle. Clamped at the ends: outside the measured span the nearest point is the
    honest answer, and the curve covers the whole sweep range.
    """
    # BIAS_CURVE runs -2.6 -> -0.8 V, which is already ASCENDING in V, and np.interp
    # silently returns nonsense on a descending x. Reversing it was a bug that made
    # every scale factor 1.000.
    xs = [c[0] for c in BIAS_CURVE]

    def at(col):
        ys = [np.log(c[col]) for c in BIAS_CURVE]
        return (float(np.exp(np.interp(v_to, xs, ys))),
                float(np.exp(np.interp(v_from, xs, ys))))
    p_to, p_from = at(1)
    i_to, i_from = at(2)
    return (p_to / p_from if p_from else 1.0), (i_to / i_from if i_from else 1.0)


def register_pools(sel: list[Row], alive: list[Row], cfg: IQEConfig,
                   log: Logger) -> int:
    """Register every pooled head row so ``light()`` lights its members, and say so.

    Returns the number of groups. Also runs the compliance check, because selection is
    where a doomed run can still be cancelled -- 30 hours later NVLED_AT_COMPLIANCE
    says the same thing, far too late.
    """
    _pool_members.clear()
    groups = [r for r in sel if "POOL_MEMBERS" in r]
    for g in groups:
        _pool_members[(int(g["X"]), int(g["Y"]))] = g["POOL_MEMBERS"]
    if not groups:
        if cfg.pool_groups:
            log(f"  pooling: pool_groups={tuple(cfg.pool_groups)} but no block gave "
                f"enough well-separated alive pixels -- nothing pooled, the single "
                f"pixels are unaffected")
        return 0
    sizes = sorted({len(g["POOL_MEMBERS"]) for g in groups})
    n_px = sum(len(g["POOL_MEMBERS"]) for g in groups)
    shared = sum(1 for g in groups
                 if any((int(s_["X"]), int(s_["Y"])) in g["POOL_MEMBERS"]
                        for s_ in sel if "POOL_MEMBERS" not in s_))
    log(f"  pooling: {len(groups)} group(s), size(s) {sizes}, {n_px} pixel-slot(s) lit "
        f"across {len(sel) - len(groups)} single(s). Each group is formed inside ONE "
        f"block, so it measures that block and not the quadrant. {shared} group(s) "
        f"share a member with a single pixel "
        f"(pool_include_singles={cfg.pool_include_singles}) -- a shared member makes "
        f"the comparison paired. A pooled row carries POOL_K and the SUM of its "
        f"members: EQE is unaffected, per-pixel current is not.")

    # The harder cap on K. Members draw K x the current and pick_limit's ladder stops
    # at vramp_highv_limit_a, so at the measured 117.7 uA per pixel at -2.60 V a 1 mA
    # ceiling clamps at K = 8. .get() because a caller may hand in bare {X, Y} rows --
    # RUN_FORMAT.md's rule for readers, and here it means the check quietly does not
    # run rather than taking the selection down with it.
    # To the sweep's ACTUAL stop, not a fixed -2.60 V. A truncated sweep was being
    # warned about currents and powers it never reaches.
    p_scale, i_scale = bias_scale(cfg.nvled_v_coarse_stop, cfg.nvled_v_fastscan)
    i_fs = {(int(r["X"]), int(r["Y"])): abs(float(r.get("NVLED_CURR_MEAN") or 0.0))
            for r in alive}
    worst_g = max(groups, key=lambda g: sum(i_fs.get(m, 0.0) for m in g["POOL_MEMBERS"]),
                  default=None)
    worst = sum(i_fs.get(m, 0.0) for m in worst_g["POOL_MEMBERS"]) if worst_g else 0.0
    i_bright = worst * i_scale
    k_worst = len(worst_g["POOL_MEMBERS"]) if worst_g else 1
    if i_bright > cfg.vramp_highv_limit_a:
        per_px = max(i_bright / max(k_worst, 1), 1e-12)
        log(f"  !! POOLED CLAMPING EXPECTED: the brightest group (K={k_worst}) should "
            f"draw ~{i_bright*1e6:.0f} uA at {cfg.nvled_v_coarse_stop:g} V, over the "
            f"{cfg.vramp_highv_limit_a*1e6:.0f} uA ceiling of pick_limit's ladder. "
            f"Those points sit on the compliance limit and are NOT measurements -- "
            f"they are flagged NVLED_AT_COMPLIANCE and dropped by the analysis. Keep "
            f"every pool_groups entry <= {max(1, int(cfg.vramp_highv_limit_a / per_px))}, "
            f"or raise vramp_highv_limit_a, or stop the sweep short of the bright end.")
    elif i_bright > 0.7 * cfg.vramp_highv_limit_a:
        log(f"  pooled bright-end estimate ~{i_bright*1e6:.0f} uA against a "
            f"{cfg.vramp_highv_limit_a*1e6:.0f} uA ceiling -- close, watch "
            f"NVLED_AT_COMPLIANCE.")

    # The OTHER ceiling, and the one nothing used to check: the PM400 range is FIXED
    # (autorange off), so a group K x brighter than a single can run out of scale at
    # the bright end. pm_range is only a request -- the realised range is logged at
    # startup -- so this compares against what was asked for and says what to do.
    p_fs = {(int(r["X"]), int(r["Y"])): abs(float(r.get("PM400_POWER_MEAN") or 0.0))
            for r in alive}
    p_worst = max((sum(p_fs.get(m, 0.0) for m in g["POOL_MEMBERS"]) for g in groups),
                  default=0.0) * p_scale
    # The realised scale if this run asked the meter, else the measured default for a
    # 1e-7 request, else the request itself. pm_range is only a request: SENS:POW:RANG
    # picks the smallest hardware range that covers it.
    rng_w = (cfg.pool_pm_range or _pm_range_realised[0]
             or (PM_RANGE_REALISED_DEFAULT if cfg.pm_range == 1e-7 else cfg.pm_range))
    how = ("set by pool_pm_range" if cfg.pool_pm_range else
           "realised, queried at startup" if _pm_range_realised[0] else
           "the measured S171C scale for a 1e-7 request")
    if p_worst > rng_w:
        log(f"  !! POOLED OVER-RANGE LIKELY: the brightest group should emit "
            f"~{p_worst*1e9:.0f} nW at {cfg.nvled_v_coarse_stop:g} V against a PM400 "
            f"range of {rng_w*1e9:.0f} nW ({how}). The range is FIXED (autorange is "
            f"off on purpose, so a mid-sweep switch cannot put a step in the data). "
            f"Raise pool_pm_range, or use smaller groups, or stop short of the bright "
            f"end.")
    elif p_worst > 0.8 * rng_w:
        log(f"  pooled bright-end estimate ~{p_worst*1e9:.0f} nW against a "
            f"{rng_w*1e9:.0f} nW PM400 range ({how}) -- close.")
    return len(groups)


def run_sweep_current(pm, smu, smile, selected: list[Row], run_dir: Path,
                      cfg: IQEConfig, log: Logger) -> int:
    """Current-source sweep: set log-spaced currents at bitval 15, measure V + P."""
    _ref_warned.clear()
    _ref_cache.clear()
    setpoints = build_current_setpoints(cfg)
    log("=" * 56)
    log(f"CURRENT-SOURCE SWEEP — {len(selected)} pixels, bitval {cfg.current_sweep_bitval}, "
        f"{len(setpoints)} setpoints {cfg.current_sweep_low_a*1e6:.2f}–{cfg.current_sweep_high_a*1e6:.0f} µA "
        f"(|V| compliance {cfg.nvled_v_limit} V)")
    configure_current_source_b(smu, cfg)
    set_nplc(smu, cfg.sweep_nplc)
    sweep_dir = run_dir / "NVLEDSweep" / "Sweep"
    pfx = run_slug(cfg.sample_name, run_dir) + "_"
    raw = IncrementalRaw(sweep_dir / "raw_data" / f"{pfx}Sweep.csv")
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
                ts = time.perf_counter() - t0
                p, ps, iv, ivs, inv, invs, vb, vbs = acquire_point(
                    pm, smu, cfg.n_smu_samples, cfg.n_pm_samples, measure_vb=True)
                row = dict(X=x, Y=y, BITVAL=cfg.current_sweep_bitval, NVLED_V=round(vb, 4),
                           TIME_START=ts, TIME_END=time.perf_counter() - t0,
                           NVLED_CURR_MEAN=inv, NVLED_CURR_SEM=invs,
                           VLED_CURR_MEAN=iv, VLED_CURR_SEM=ivs,
                           PM400_POWER_MEAN=p, PM400_POWER_SEM=ps)
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
        write_aggregated_per_bitval(sweep_dir, sw_rows, prefix=pfx)
    (sweep_dir / "raw_data" / f"{pfx}Sweep_config.json").write_text(
        json.dumps({"settings": cfg.to_dict(), "provenance": provenance()}, indent=2),
        encoding="utf-8")
    return len(sw_rows)


_ref_warned: set[tuple[int, int]] = set()
# ponytail: one set per process, cleared by the sweep entry points. A process runs one
# sweep, so this only matters to the tests, which is why they clear it.

_ref_cache: dict[tuple[int, int], tuple] = {}

# (x, y) -> every member of the pooled group this target stands for. A target absent
# from this map is an ordinary single pixel, which is the default and the common case.
# Kept module-level beside _ref_cache for the same reason: measure_point is handed
# coordinates, not the selection row, and threading a members argument through the two
# sweep walks, the reference read and the pre-pass would be five signatures changed to
# carry one fact that never varies within a run.
_pool_members: dict[tuple[int, int], tuple[tuple[int, int], ...]] = {}


def per_member_power(row) -> float:
    """|P| divided by the number of members -- the quantity a dead test can compare.

    Both dead-pixel tests judge a target against the median of the others. A pooled
    target emits K x as much, so mixing the two unscaled would raise the median by
    roughly K and condemn perfectly live SINGLE pixels. Per member, the two are
    comparable again.
    """
    return abs(row["PM400_POWER_MEAN"]) / max(1, int(row.get("POOL_K", 1) or 1))


def pool_members(x, y) -> tuple[tuple[int, int], ...]:
    """The members this target lights. ``((x, y),)`` for an ordinary single pixel."""
    return _pool_members.get((int(x), int(y)), ((int(x), int(y)),))


def fs_power_anchor(r) -> float:
    """A selected target's fast-scan power x its member count: the first PM-range guess."""
    try:
        return abs(float(r.get("PM400_POWER_MEAN") or 0.0)) * len(pool_members(r["X"], r["Y"]))
    except (TypeError, ValueError):
        return 0.0


def light(smile, x, y, bitval):
    """Drive the target keyed by ``(x, y)`` at ``bitval``.

    A single pixel goes through ``set_pixel``; a pooled group goes through
    ``set_pixels``, which writes all K members in ONE frame instruction. That is why a
    pooled point costs what a single point costs: the dwell, the settle and the burst
    are per FRAME, not per pixel.
    """
    members = _pool_members.get((int(x), int(y)))
    if members is None:
        return smile.set_pixel(x, y, bitval)
    return smile.set_pixels(members, bitval)
# Per-pixel leakage reference, filled by prefill_reference_cache() and read by
# measure_point() at every bias at or below cfg.leak_ref_per_point_above_v. Same
# lifetime and the same reason as _ref_warned: one sweep per process.
#
# It is keyed on the PIXEL only. The reference is taken at leak_ref_bitval whatever the
# lit bitval is, and the whole point of the cache is that it does not depend on the
# bias, so (x, y) is the entire key.
#
# It must never outlive the run. The same 16 pixels at -0.80 V read 258.20, 267.20,
# 277.67 and 280.32 nA across the four direction runs of 261016 -- 9.0 nA between two
# runs in the same thermal position, which is 4.7% of I_LED at that bias. A reference
# inherited from an earlier run is worse than no reference at all.


def read_reference(pm, smu, smile, x, y, n_pm, n_smu, cfg: IQEConfig,
                   p_lit: float | None, log: Logger, v: float | None = None):
    """One leakage reference for ``(x, y)``: the same pixel at ``leak_ref_bitval``.

    Returns ``(ref, tries, frac)``. ``ref`` is the acquire_point tuple with the raw
    burst samples appended as ``ref[8]``, so the row can carry the reference's own
    burst_stats. ``frac`` is the optical gate's ratio against ``p_lit`` and is NaN when
    the caller has no lit reading to gate on.

    The caller owns the frame: this leaves the pixel at ``leak_ref_bitval`` and does
    not restore it.
    """
    ref = None
    tries = 0
    frac = float("nan")
    # The frame write sometimes does not land, so the reference is VERIFIED rather
    # than assumed, and re-written when it fails. Section 19.
    #
    # The gate is the OPTICAL ratio, and it has to be: a healthy backplane
    # legitimately conducts 58% of the reading at -0.80 V (section 6), so the
    # current cannot discriminate -- but it does not EMIT. Across the 100 pixels of
    # run 261015-0015 a healthy pixel references at 2e-4 of its own light and the
    # failures sit at 0.38-1.03, with no clean point above the gate at ANY bias.
    for tries in range(1, max(1, cfg.leak_ref_tries) + 1):
        # The reference pools too. Section 6: an all-off frame conducts ~4x the
        # bitval-0 level, and for a pooled target the same-members-at-bitval-0 frame is
        # the only reference with the same frame structure and select state.
        light(smile, x, y, cfg.leak_ref_bitval)
        time.sleep(cfg.leak_ref_settle_s * cfg.settle_scale)
        # FULL n_pm, not a quarter of it. The reference current is stable to ~0.2 nA
        # and needs almost no averaging, but the reference POWER is the dark level
        # being subtracted from a ~250 pW signal, so at the dim end it is the
        # dominant error. Run 261014-0035 with n_pm//4: the reference SEM was 30 pW
        # against the lit reading's 14 pW -- the term that mattered most was
        # averaged the least.
        pt, raw = acquire_point(pm, smu, n_smu, n_pm, return_raw=True)
        ref = pt + (raw,)
        frac = abs(ref[0] / p_lit) if p_lit else float("nan")
        if not (frac > cfg.leak_ref_max_frac):   # NaN must not trigger a retry
            break
        if tries < max(1, cfg.leak_ref_tries):
            log(f"      ({x},{y})" + (f" at {v:+.2f} V" if v is not None else "")
                + f": bitval {cfg.leak_ref_bitval} reference carries "
                f"{100 * frac:.0f}% of the lit light -- the frame write did not land. "
                f"Re-writing it (try {tries + 1} of {cfg.leak_ref_tries}).")
    return ref, tries, frac


def prefill_reference_cache(pm, smu, smile, pixels, cfg: IQEConfig, log: Logger) -> int:
    """Measure one leakage reference per pixel, for every bias that reuses it.

    Runs once, before the bias walk, at ``leak_ref_prepass_v``. Costs one pass over the
    pixels (~3 s each) and removes the reference from every point at or below
    ``leak_ref_per_point_above_v`` -- 2200 of 3000 points at the default grid, which is
    1.69 h of a 4.82 h sweep. See ``config.leak_ref_per_point_above_v`` for the measured
    cost, which is under 0.14% everywhere except -2.60 V.

    There is no lit reading here, so the optical gate cannot run. The estimator is the
    MINIMUM of ``leak_ref_tries`` reads instead, which is the right one on this failure
    mode: a dropped frame write leaves the pixel emitting, and that only ever ADDS
    current. Measured contamination runs 10 700-28 200 nA against a clean 256-280 nA, a
    factor 40-110, so the minimum is clean unless every read failed.
    """
    if not cfg.leak_reference or cfg.leak_ref_per_point_above_v is None:
        return 0
    v = float(cfg.leak_ref_prepass_v)
    log(f"  leakage reference pre-pass: {len(pixels)} pixel(s) at {v:+.2f} V, "
        f"min of {max(1, cfg.leak_ref_tries)} read(s) each. Reused at every bias "
        f"<= {cfg.leak_ref_per_point_above_v:+.2f} V, re-measured above it.")
    # The pre-pass bias is on the faint side, so it takes the faint side's limit and
    # NPLC. That is also where the reference is most precise -- burst sem 0.13 nA on
    # the 1e-5 range against 9.06 nA on 1e-3 -- and that precision carries to every
    # bias that reuses the value.
    configure_voltage_source_b(
        smu, dataclasses.replace(cfg, nvled_compliance=cfg.vramp_lowv_limit_a), log)
    set_nplc(smu, cfg.vramp_lowv_nplc)
    smu.set_voltage("b", v)
    t_pre = time.perf_counter()
    suspect = []
    for (x, y) in pixels:
        smile.blank_frame()
        reads = []
        for _ in range(max(1, cfg.leak_ref_tries)):
            r, _t, _f = read_reference(pm, smu, smile, x, y, cfg.vramp_lowv_n_pm,
                                       cfg.vramp_lowv_n_smu, cfg, None, log, v)
            if r is not None:
                reads.append(r)
        if not reads:
            continue
        best = min(reads, key=lambda r: abs(r[4]))    # r[4] = NVLED current mean
        # keyed by the range it was read on when the range moves (see measure_point)
        _ref_cache[(int(x), int(y))
                   + ((_pm_range_now[0],) if cfg.pm_autorange else ())] = best
        hi = max(abs(r[4]) for r in reads)
        if hi > 2.0 * abs(best[4]):
            # Not a rejection -- the minimum already rejected it. This says the pixel
            # drops frame writes, which is what section 19 is about and what makes a
            # pixel worth watching in the analysis.
            suspect.append((int(x), int(y), abs(best[4]) * 1e9, hi * 1e9))
    smile.blank_frame()
    for (x, y, lo, hi) in suspect:
        log(f"    !! ({x},{y}): reference reads spanned {lo:.1f}-{hi:.0f} nA over "
            f"{max(1, cfg.leak_ref_tries)} writes. The minimum is used and is clean, "
            f"but this pixel drops frame writes -- section 19.")
    log(f"  pre-pass done: {len(_ref_cache)} reference(s) cached in "
        f"{time.perf_counter() - t_pre:.0f}s"
        + (f", {len(suspect)} pixel(s) dropped a write" if suspect else ""))
    return len(_ref_cache)


# What the meter is set to right now, so the switch writes only when it has to.
_pm_range_now = [0.0]


def pm_range_for(x: int, y: int, v: float, cfg: IQEConfig) -> float:
    """The PM400 range this (target, bias) should be read on.

    Pooled targets only, and only at or below ``pool_pm_range_below_v``: a single pixel
    reaches 257 nW at -2.6 V and never leaves the fine range, so switching for one would
    cost 5.45x in floor (section 26.13) and buy nothing.
    """
    if (cfg.pool_pm_range_below_v is None or not cfg.pool_pm_range
            or v > cfg.pool_pm_range_below_v):
        return cfg.pm_range
    return cfg.pool_pm_range if len(pool_members(x, y)) > 1 else cfg.pm_range


def set_pm_range(pm, want: float, n_pm: int = 4, smu=None, n_smu: int = 2,
                 log: Logger | None = None) -> tuple[float, bool]:
    """Set the meter's range if it is not already there. Returns (range, changed).

    The REFERENCE must be read on the same range as the lit point, or the offset it
    subtracts is the other range's -- 1154 pW apart, measured. measure_point() sets the
    range once at its top and nothing inside it changes the range.

    **On a change, one burst is read and discarded.** Run 261020-0842 produced three
    corrupt points in 2048 and every one was the first point after a range change: the
    meter's internal averaging (pm_averaging=3) mixes samples from across the gain
    switch, so the first burst carries pre-switch samples at the post-switch gain --
    which came back NEGATIVE twice in the cross-check, with a burst std 300x normal. A
    throwaway burst flushes the averager whatever its time constant is, where
    pm_range_settle_s was a guess that turned out too short.
    """
    if _pm_range_now[0] == want:
        return want, False
    pm.set_range(want)
    prev, _pm_range_now[0] = _pm_range_now[0], want
    if log is not None and prev:
        log(f"      PM400 range {prev*1e9:.0f} -> {want*1e9:.0f} nW (requested); "
            f"flushing one burst")
    return want, True


def flush_pm_after_range(pm, smu, n_smu: int, n_pm: int, cfg: IQEConfig) -> None:
    """Settle, then read and discard one burst. See set_pm_range."""
    if cfg.pm_range_settle_s:
        time.sleep(cfg.pm_range_settle_s)
    try:
        acquire_point(pm, smu, max(1, n_smu), max(1, n_pm))
    except Exception:
        pass            # a throwaway read must never be what ends a 30 h run


# The PM400's hardware ranges as the meter realises them, learned once per session:
# a request picks the smallest range that covers it (1e-7 -> 470 nW on the S171C), so
# the ladder is asked for, not assumed.
_pm_ladder: list[float] = []


def pm_ladder(pm, cfg: IQEConfig, log: Logger | None = None) -> list[float]:
    """The realised full scales, ascending. Requests every decade 1e-9..1e-2 W once."""
    if _pm_ladder:
        return _pm_ladder
    got = set()
    if hasattr(pm, "get_range"):
        for e in range(-9, -1):
            try:
                pm.set_range(10.0 ** e)
                got.add(float(pm.get_range()))
            except Exception:
                pass
        try:
            pm.set_range(_pm_range_now[0] or cfg.pm_range)   # back where it was
        except Exception:
            pass
    _pm_ladder.extend(sorted(r for r in got if r > 0) or [cfg.pm_range])
    if log is not None:
        log("  PM400 ranges: " + ", ".join(f"{r:.3g} W" for r in _pm_ladder))
    return _pm_ladder


def pick_pm_range(p_pred: float | None, cfg: IQEConfig, ladder: list[float]) -> float:
    """The smallest range >= pm_range_headroom x the predicted power. With nothing
    predicted, the widest: a clipped point is lost, a coarse one only noisier."""
    if p_pred is None or not math.isfinite(p_pred) or p_pred <= 0:
        return ladder[-1]
    need = cfg.pm_range_headroom * abs(p_pred)
    return next((r for r in ladder if r >= need), ladder[-1])


def pm_over(p, rng: float, cfg: IQEConfig) -> bool:
    """An over-range (or unreadable) PM400 reading on full scale ``rng``."""
    return p is None or not math.isfinite(p) or abs(p) >= cfg.pm_overrange_frac * rng


def pm_range_up(pm, smu, rng: float, n_smu: int, n_pm: int, cfg: IQEConfig,
                log: Logger) -> float:
    """One range up the ladder, with the discarded burst. ``rng`` at the top."""
    up = next((r for r in (_pm_ladder or [rng]) if r > rng), None)
    if up is None:
        return rng
    log(f"      PM400 over-range on {rng:.3g} W: up to {up:.3g} W and read again")
    set_pm_range(pm, up)
    flush_pm_after_range(pm, smu, n_smu, n_pm, cfg)
    return up


def probe_pm_range(pm, smu, smile, pixels, cfg: IQEConfig, log: Logger, bitval: int,
                   n: int, scale: float = 1.0, tag: str = "") -> float:
    """Set the range from the brightest of up to ``n`` pixels read on the widest range.

    For a pass whose power nothing predicts yet: a fast-scan bias, a pixel search. The
    brightest x ``scale`` (K for groups) x headroom picks the range; a pixel that is
    brighter still is caught by the per-point over-range retry."""
    ladder = pm_ladder(pm, cfg, log)
    set_pm_range(pm, ladder[-1])
    flush_pm_after_range(pm, smu, 1, 1, cfg)
    pick = list(pixels)
    if len(pick) > n:
        idx = np.random.default_rng(0).choice(len(pick), n, replace=False)
        pick = [pick[i] for i in sorted(idx)]
    p_max = 0.0
    for (x, y) in pick:
        smile.blank_frame()
        smile.set_pixel(x, y, bitval)
        time.sleep(max(cfg.fastscan_settle_ms, 5.0) / 1000.0)
        p = pm.measure()
        if p is not None and math.isfinite(p):
            p_max = max(p_max, abs(p))
    smile.blank_frame()
    rng = pick_pm_range(p_max * scale, cfg, ladder)
    _, changed = set_pm_range(pm, rng)
    if changed:
        flush_pm_after_range(pm, smu, 1, 1, cfg)
    log(f"  PM400 range {rng:.3g} W{tag}: brightest of {len(pick)} probed pixel(s) "
        f"{p_max:.3g} W" + (f" x{scale:g}" if scale != 1 else ""))
    return rng


# A lit reading that draws no more than this times its own leakage reference did not
# light the pixel: the frame write was dropped, as the bitval-0 write sometimes is
# (section 19). A working pixel is >= ~1.9x even at -0.80 V, where the reference is at
# most 52 % of the raw reading (report 33.1). 261016-2252, (489,181): 300 nA, about its
# reference, and no light at -1.418 and -1.189 V; nothing caught it at the bench.
LIT_FAIL_RATIO = 1.10


def measure_point(pm, smu, smile, x, y, bitval, v, limit, n_pm, n_smu,
                  cfg: IQEConfig, t0: float, log: Logger,
                  restore: bool = True,
                  pm_rng: float | None = None) -> tuple[dict, dict | None]:
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
    # Before ANYTHING is read, including the reference: both must be on one range.
    # Only when the switch is configured -- an ordinary run must not gain an SCPI write
    # per point for a range that never moves.
    rng = cfg.pm_range
    if cfg.pool_pm_range_below_v is not None:
        rng, changed = set_pm_range(pm, pm_range_for(x, y, v, cfg), log=log)
        if changed:
            flush_pm_after_range(pm, smu, n_smu, n_pm, cfg)
    if pm_rng is not None:                      # the caller's per-bias pick
        rng, changed = set_pm_range(pm, pm_rng)
        if changed:
            flush_pm_after_range(pm, smu, n_smu, n_pm, cfg)
    # TIME_START/TIME_END bracket the lit burst that the row reports -- the re-read's,
    # when the first lit frame did not land.
    t_lit = [0.0, 0.0]

    def read_lit():
        # The raw samples are taken whatever save_raw_samples says: the row's stats are
        # computed from them, so the columns mean the same thing in every run. The
        # setting decides only whether they also go to Sweep_samples.csv.
        t_lit[0] = time.perf_counter() - t0
        if cfg.perpoint_range:
            # range from the ACTUAL current (probe-then-measure), not the coarse map
            pt, raw = acquire_autoranged(pm, smu, n_smu, n_pm, cfg, return_raw=True)
        else:
            pt, raw = acquire_point(pm, smu, n_smu, n_pm, return_raw=True)
        t_lit[1] = time.perf_counter() - t0
        return pt[:6], raw

    (p, ps, iv, ivs, inv, invs), raw_s = read_lit()
    # Over range: one range up and read again, as often as the ladder allows. The
    # reference below is then read on the same, final range.
    while pm_rng is not None and pm_over(p, rng, cfg):
        up = pm_range_up(pm, smu, rng, n_smu, n_pm, cfg, log)
        if up == rng:
            break
        rng = up
        (p, ps, iv, ivs, inv, invs), raw_s = read_lit()

    # Leakage reference: the SAME pixel at bitval 0. Measured back to back with the lit
    # point so both carry the same relaxation history and the subtraction cancels most of
    # it. At -0.80 V this is 58% of the reading.
    #
    # One reference per BIAS, shared across pixels, is wrong and stays wrong: the
    # backplane leak varies ~10% between pixels (run 261014-1616 at -1.30 V: 159-192 nA
    # across four), and at -0.80 V the reference IS 58% of the reading, so 10% of it is
    # a 6% error on I_LED. bias_major amortises the DWELL; the reference is not a dwell.
    #
    # One reference per PIXEL, shared across biases, is the other slice and it holds --
    # below the faint end. Over all 3000 points of run 261015-0015 one fixed per-pixel
    # value costs 0.03-0.13% from -0.83 V down and 0.20% at -2.60 V. So the reference is
    # cached by pixel below leak_ref_per_point_above_v and measured per point above it.
    ref = None
    tries = 0
    frac = float("nan")
    cached = False
    if cfg.leak_reference:
        # Cached or measured? Cached at or below leak_ref_per_point_above_v, where the
        # reference is a few percent of the reading and a fixed per-pixel value costs
        # under 0.14%; measured per point above it, where it is 5-62% of the reading
        # and the subtraction decides the answer. See config.leak_ref_per_point_above_v.
        # With the PM range moving, a cached reference is only valid on the range it
        # was read on: the ranges' dark offsets differ (1154 pW, 470 nW vs 4.7 uW).
        # So the cache is per (pixel, range), and a pixel reaching a new range below
        # the threshold measures its reference once there and caches it.
        ck = (int(x), int(y)) + ((rng,) if pm_rng is not None else ())
        below = (cfg.leak_ref_per_point_above_v is not None
                 and v <= cfg.leak_ref_per_point_above_v)
        use_cache = below and ck in _ref_cache
        if use_cache:
            ref = _ref_cache[ck]
            cached = True
            # The frame was never touched, so there is nothing to restore and no
            # settle to pay -- the saving is the 2 s reference settle, its burst, and
            # in pixel_major the 2 s restore settle on top.
            #
            # The gate still runs, against the CACHED dark power and THIS point's lit
            # power. That is a real test: the cache holds a clean minimum, so a large
            # frac here means the LIT reading collapsed, not that the reference failed.
            frac = abs(ref[0] / p) if p else float("nan")
        else:
            ref, tries, frac = read_reference(pm, smu, smile, x, y, n_pm, n_smu,
                                              cfg, p, log, v)
            if pm_rng is not None and below and not (frac > cfg.leak_ref_max_frac):
                _ref_cache[ck] = ref
            if restore:
                light(smile, x, y, bitval)
                time.sleep(cfg.leak_ref_settle_s * cfg.settle_scale)
            # else: the caller blanks the frame next, so lighting this pixel again and
            # waiting for it to settle would be two seconds spent on a state nothing
            # reads. In bias_major that is half the per-point cost, on every point.

    if ref is not None and abs(inv) <= LIT_FAIL_RATIO * abs(ref[4]):
        # The lit frame did not land: the pixel drew only its leakage. Write it again,
        # give it the pixel-switch settle, and read once more. One retry -- a pixel that
        # is dark twice is dead or dropping out, and the analysis gates remove it.
        log(f"      ({x},{y}) at {v:+.3f} V: lit read {abs(inv) * 1e9:.0f} nA against a "
            f"{abs(ref[4]) * 1e9:.0f} nA reference -- the lit frame did not land. "
            f"Re-writing it.")
        light(smile, x, y, bitval)
        time.sleep(cfg.pixel_switch_settle_s * cfg.settle_scale)
        (p, ps, iv, ivs, inv, invs), raw_s = read_lit()
        if abs(inv) <= LIT_FAIL_RATIO * abs(ref[4]):
            log(f"      ({x},{y}): still {abs(inv) * 1e9:.0f} nA after the re-write -- the "
                f"pixel is dark here, not the frame. Point kept; the analysis drops it.")
        frac = abs(ref[0] / p) if p else float("nan")   # the gate, against the new read

    vr = round(float(v), 4)
    # A pooled row's X,Y is its FIRST member, so the row still has a unique key and the
    # analysis groups by it unchanged. POOL_K says how many pixels the numbers are the
    # sum of. A reader that ignores POOL_K gets K x the power AND K x the current, so
    # EQE comes out right and only per-pixel current density is wrong -- see
    # RUN_FORMAT.md, which says to treat an absent column as 1.
    members = pool_members(x, y)
    # Every channel's stats come from its raw samples, so *_MEAN is a true mean and
    # *_SEM is SAMPSTD/sqrt(N) in every sweep file. The centre the analysis uses is
    # *_MEDIAN. acquire_point's own (median, MAD) pair is not written.
    row = dict(X=x, Y=y, BITVAL=bitval, NVLED_V=vr,
               TIME_START=t_lit[0], TIME_END=t_lit[1],
               **stat_columns("NVLED_CURR", raw_s["nvled"]),
               **stat_columns("VLED_CURR", raw_s["vled"]),
               **stat_columns("PM400_POWER", raw_s["pm"]))
    if len(members) > 1:
        # "x:y x:y", not "x,y": the aggregated writer joins fields with a comma and
        # does not quote, so a comma inside a value silently shifts every later column.
        row.update(POOL_K=len(members),
                   POOL_MEMBERS=" ".join(f"{mx}:{my}" for mx, my in members))
    if cfg.pool_pm_range_below_v is not None or pm_rng is not None:
        row["PM_RANGE_W"] = rng
    if ref is not None:
        row.update(NVLED_CURR_REF=ref[4], VLED_CURR_REF=ref[2], PM400_POWER_REF=ref[0],
                   LEAK_REF_BITVAL=cfg.leak_ref_bitval)
        if len(ref) > 8:        # the raw reference samples (read_reference)
            rraw = ref[8]
            row.update(**stat_columns("NVLED_CURR_REF", rraw["nvled"]),
                       **stat_columns("VLED_CURR_REF", rraw["vled"]),
                       **stat_columns("PM400_POWER_REF", rraw["pm"]))
        row.update(LEAK_REF_TRIES=tries, LEAK_REF_FRAC=frac,
                   LEAK_REF_CACHED=int(cached))
        # Say it at the bench, not three days later in the analysis. Only once per
        # pixel, because a pixel whose writes keep failing would otherwise print on
        # every one of its 30 biases.
        if frac > cfg.leak_ref_max_frac and (x, y) not in _ref_warned:
            _ref_warned.add((x, y))
            how = ("the CACHED reference carries" if cached else
                   f"the bitval {cfg.leak_ref_bitval} reference still carries")
            why = ("The cache holds a clean minimum from the pre-pass, so this is the "
                   "LIT reading collapsing, not a failed write."
                   if cached else
                   f"The write did not land in {cfg.leak_ref_tries} tries, so the "
                   f"subtraction removes most of the signal.")
            log(f"  !! ({x},{y}): {how} {100 * frac:.0f}% of the lit light at "
                f"{v:+.2f} V. {why} These points are not usable. Intermittent in both "
                f"directions -- see section 19. The analysis flags it REF_CONTAMINATED.")
    # A point sitting on the limit is not a measurement. Flag it in the file and in the
    # log rather than letting it reach the fit as a real current.
    inv_m = row["NVLED_CURR_MEAN"]
    row["NVLED_LIMIT_A"] = limit
    at_limit = bool(limit) and abs(inv_m) >= 0.995 * limit
    row["NVLED_AT_COMPLIANCE"] = at_limit
    if at_limit:
        log(f"      AT COMPLIANCE at {v:+.3f} V: |I|={abs(inv_m):.6g} A is the "
            f"{limit:g} A limit, not the device. Point is flagged, not usable.")
    return row, (raw_s if cfg.save_raw_samples else None)


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
    for agg in sweep.glob("*aggregated_bitval=*_mean_std.csv"):
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
    for agg in sweep.glob("*aggregated_bitval=*_mean_std.csv"):
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


def range_boundary_bias(fine, cfg: IQEConfig) -> float | None:
    """The sweep bias to cross-check the PM400 range at, or None.

    The LEAST negative bias past the threshold, not the deepest. Two reasons, and the
    second is the one that bites: it is adjacent to the last fine-range point, so the
    comparison is tightest; and the fine range has to still hold a pooled target there
    or the re-read is over-range and measures nothing. At K=4 on the measured array a
    group reads 444 nW at -1.82 V against the 470 nW the fine range realises, and
    586 nW at -2.6 V, which does not fit.
    """
    if cfg.pool_pm_range_below_v is None:
        return None
    past = [v for v, _ in fine if v <= cfg.pool_pm_range_below_v]
    return max(past) if past else None


def pm_range_dual_pass(pm, smu, smile, live, bitval, v, limit, n_pm, n_smu,
                       rows: list[Row], cfg: IQEConfig, t0: float,
                       log: Logger) -> list[Row]:
    """Re-measure this bias's pooled targets on the OTHER PM400 range.

    Called once, at the first sweep bias past ``pool_pm_range_below_v``, from inside
    bias_major -- so the bias is already settled and the dwell is already paid. Each
    target costs a frame switch, a reference and a burst, the same as its first reading.

    Returns rows tagged with the range they were taken on. They are NOT added to the
    sweep, because two rows for one (target, bias) would break every groupby in the
    analysis; they go to pm_range_check.csv beside the pre-flight pair.
    """
    pooled = [(x, y) for (x, y) in live if len(pool_members(x, y)) > 1]
    if not pooled:
        return []
    other = cfg.pm_range          # the pooled targets just used pool_pm_range here
    first = {(int(r["X"]), int(r["Y"])): r for r in rows}
    # The whole point of the switch is that the fine range cannot hold a pooled target
    # at the bright end -- so re-reading it there would be over-range, and comparing an
    # over-range reading against an on-scale one measures nothing. The brightest target
    # just measured at THIS bias is the exact test, no estimate needed.
    realised = (_pm_range_realised[0]
                or (PM_RANGE_REALISED_DEFAULT if other == 1e-7 else other))
    worst = max((abs(first[k]["PM400_POWER_MEAN"]) for k in first
                 if len(pool_members(*k)) > 1), default=0.0)
    if worst > 0.95 * realised:
        log(f"      range cross-check SKIPPED at {v:+.3f} V: the brightest pooled "
            f"target reads {worst*1e9:.0f} nW, over 95 % of the {realised*1e9:.0f} nW "
            f"the fine range realises, so re-reading it there would be over-range and "
            f"the comparison would measure nothing. Move pool_pm_range_below_v to a "
            f"bias where a pooled target still fits the fine range -- at K=4 that is "
            f"about -1.6 V (section 26.18).")
        return []
    log(f"      range cross-check at {v:+.3f} V: re-measuring {len(pooled)} pooled "
        f"target(s) on {other*1e9:.0f} nW (they were on "
        f"{cfg.pool_pm_range*1e9:.0f} nW). Same dwell, same reference.")
    out: list[Row] = []
    for (x, y) in pooled:
        if parked():
            break
        smile.blank_frame()
        light(smile, x, y, bitval)
        time.sleep(cfg.pixel_switch_settle_s * cfg.settle_scale)
        # pm_range_for() would hand back the coarse range here, so force the other one
        # and let measure_point's own flush deal with the change.
        _, ch = set_pm_range(pm, other, log=log)
        if ch:
            flush_pm_after_range(pm, smu, n_smu, n_pm, cfg)
        row, _ = measure_point(pm, smu, smile, x, y, bitval, v, limit, n_pm, n_smu,
                               dataclasses.replace(cfg, pool_pm_range_below_v=None),
                               t0, log, restore=False)
        row["PM_RANGE_W"] = other
        out.append({k: row.get(k, "") for k in PM_CHECK_COLS})
    # paired, same target, same bias, same everything but the range
    d = []
    for r in out:
        f = first.get((int(r["X"]), int(r["Y"])))
        if f is None or not f["PM400_POWER_MEAN"] or not r["PM400_POWER_MEAN"]:
            continue
        if (r["PM400_POWER_MEAN"] <= 0 or f["PM400_POWER_MEAN"] <= 0
                or abs(r["PM400_POWER_SEM"]) > 0.02 * abs(r["PM400_POWER_MEAN"])
                or abs(f["PM400_POWER_SEM"]) > 0.02 * abs(f["PM400_POWER_MEAN"])):
            continue            # a bad read, not a range step -- see 26.20
        d.append(abs(f["PM400_POWER_MEAN"]) / abs(r["PM400_POWER_MEAN"]) - 1.0)
    if not d:
        log(f"      !! every pair was unusable, so this bias did NOT verify the switch.")
        return out
    m = float(np.median(d))
    sd = float(np.std(d, ddof=1)) if len(d) > 1 else float("nan")
    sem = sd / math.sqrt(len(d)) if len(d) > 1 else float("nan")
    sig = abs(m) / sem if sem and sem > 0 else float("nan")
    log(f"      coarse/fine = {1+m:.4f} ({100*m:+.3f} %) over {len(d)} target(s) at a "
        f"REAL sweep point, spread {100*sd:.3f} %"
        + (f", {sig:.1f} sigma on the mean" if np.isfinite(sig) else ""))
    if np.isfinite(sd) and sd > 0.02:
        log(f"      !! spread {100*sd:.1f} % -- this cannot resolve a step, so the "
            f"small mean means nothing. Something is wrong with the readings.")
    elif abs(m) > 0.01:
        log(f"      !! THE RANGES DISAGREE BY {100*m:+.2f} % at a real sweep point, "
            f"over the 1 % this switch assumes. Every pooled point at or below "
            f"{v:+.3f} V carries it as a step the analysis cannot see.")
    elif np.isfinite(sig) and sig >= 3.0:
        log(f"      a real but small step, {100*m:+.3f} % at {sig:.1f} sigma. Under "
            f"1 %, so the switch stands -- put it in any budget spanning the two "
            f"segments.")
    else:
        log(f"      consistent with no step, measured on the sweep's own grid.")
    return out


def _sweep_bias_major(pm, smu, smile, selected: list[Row], writer: "BackgroundWriter",
                      settle_traces: list, cfg: IQEConfig, log: Logger,
                      t0: float, run_dir: Path | None = None) -> int:
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
    light(smile, first["X"], first["Y"], cfg.bitvals[0])
    fine = build_fine_voltages(pm, smu, cfg, log,
                               tag=f"grid from ({first['X']},{first['Y']}): ")
    if not fine:
        log("  no fine voltages built - nothing to sweep")
        return 0

    # run_dir, not writer._agg_folder.parent.parent.parent: that is one level ABOVE the
    # run (the folder is <run>/NVLEDSweep/Sweep), found nothing, and so every bias_major
    # resume re-measured the whole sweep and doubled its rows.
    done_v = completed_biases(run_dir) if cfg.resume else set()
    if done_v:
        log(f"  resuming: {len(done_v)} bias(es) already complete on every pixel")

    live = [(int(r["X"]), int(r["Y"])) for r in selected]

    # One conditioning pass over every pixel, before the first bias. See
    # config.bias_major_precondition for why this is once-per-run and not per point.
    if cfg.bias_major_precondition and cfg.warmup_ms:
        # Explicit limit: the soak is only a soak if the pixel can draw its current.
        configure_voltage_source_b(
            smu, dataclasses.replace(cfg, nvled_compliance=cfg.vramp_highv_limit_a), log)
        cur_limit = cfg.vramp_highv_limit_a
        smu.set_voltage("b", float(cfg.warmup_nvled_v))
        t_pre = time.perf_counter()
        for (x, y) in live:
            smile.blank_frame()
            light(smile, x, y, cfg.bitvals[0])
            time.sleep(cfg.warmup_ms / 1000.0)
        smile.blank_frame()
        log(f"  preconditioned {len(live)} pixel(s) at {cfg.warmup_nvled_v:+.2f} V, "
            f"{cfg.warmup_ms / 1000.0:g}s each, limit {cfg.vramp_highv_limit_a:g} A "
            f"({time.perf_counter() - t_pre:.0f}s). Tests whether the conditioning "
            f"persists -- section 17.")
    else:
        cur_limit = None

    # Say it every run, because an operator who sees a deficit will reach for this
    # knob, and 261015-0958 already paid 2 hours to prove it is the wrong one.
    sw = cfg.pixel_switch_settle_s * cfg.settle_scale
    log(f"  !! bias_major reads ~1% low in current on the FAINT side vs pixel_major, and "
        f"nothing above -1.8V (back-to-back runs 261015-1421/-1529, 1.05mV). It reads ~3% "
        f"against a reference from the day BEFORE: two thirds of that is CROSS-RUN DRIFT, "
        f"~2%/day, which is now the project's largest error. Anchor any absolute number "
        f"with a SAME-DAY pixel_major run. Report against J: at matched CURRENT the "
        f"orders agree to 0.84%. Leave pixel_switch_settle_s={sw:g}s alone -- see "
        f"config.pixel_switch_settle_s.")
    if sw > 5.0:
        log(f"  !! pixel_switch_settle_s={sw:g}s costs n_px x n_points x {sw:g}s and buys "
            f"nothing measurable. 2s is the measured-correct value.")
    n_dead = 0
    cur_nplc = cur_range = None      # cur_limit is set by the preconditioning block
    i_last = None
    v_last = None
    # The PM400 range, chosen per bias from the brightest target of the previous bias
    # scaled by the measured P(V), exactly as the SMU limit is. The first bias has no
    # prediction and starts on the widest range.
    p_last = pv_last = None
    pm_lad = pm_ladder(pm, cfg, log) if cfg.pm_autorange else None
    if pm_lad:
        # the first bias, from the brightest target's own fast-scan power
        p0 = max((fs_power_anchor(r) for r in selected), default=0.0)
        if p0 > 0:
            p_last, pv_last = p0, float(cfg.nvled_v_fastscan)
    # The first sweep bias at or below the threshold -- the one whose pooled targets
    # switched range. None when the switch is off or no grid point is past it.
    dual_rows: list[Row] = []
    boundary_v = range_boundary_bias(fine, cfg)
    if cfg.pool_pm_range_below_v is not None:
        if boundary_v is None:
            log(f"  no sweep bias reaches {cfg.pool_pm_range_below_v:+.2f} V, so the "
                f"range never switches and there is nothing to cross-check.")
        else:
            log(f"  range boundary: {boundary_v:+.3f} V is the first bias past "
                f"{cfg.pool_pm_range_below_v:+.2f} V. Every pooled target will be "
                f"measured there on BOTH ranges.")

    for bitval in cfg.bitvals:
        for jpt, (v, _i_target) in enumerate(fine):
            if parked():
                break
            if not live:
                # dead_abort dropped every target at the first bias. Nothing is left to
                # light, and serving 15 more dwells to no pixel would only burn time.
                log(f"  every target was dropped as dead; the remaining "
                    f"{len(fine) - jpt} bias(es) are skipped.")
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
            # From the BRIGHTEST target of the previous bias, not the last one measured:
            # with pooled groups a single pixel set a 100 uA limit and the next group,
            # 4x the current, clamped (261022-0037; 261016-2252 had one more).
            limit = pick_limit(i_last, cfg, v_last, v)
            if limit != cur_limit:
                configure_voltage_source_b(
                    smu, dataclasses.replace(cfg, nvled_compliance=limit), log)
                cur_limit, cur_nplc, cur_range = limit, None, None
            if nplc != cur_nplc:
                set_nplc(smu, nplc); cur_nplc = nplc
            if (not cfg.perpoint_range and not cfg.highc_b and cfg.nvled_range_i
                    and cfg.nvled_range_i != cur_range):
                set_measure_range_b(smu, cfg.nvled_range_i); cur_range = cfg.nvled_range_i
            i_bias = 0.0
            p_bias = 0.0
            pm_rng = None
            if pm_lad:
                pm_rng = pick_pm_range(
                    p_last * bias_scale(v, pv_last)[0] if p_last else None, cfg, pm_lad)
                # set before the dwell, so the dwell's own P(t) trace is on scale too
                if set_pm_range(pm, pm_rng)[1]:
                    flush_pm_after_range(pm, smu, n_smu, n_pm, cfg)

            # --- the one dwell this bias costs, for all the pixels below it ---
            #
            # Light a pixel FIRST, so the array relaxes at the new bias in the state the
            # measurements are taken in rather than blanked -- a blanked frame conducts
            # ~4x the bitval-0 level (section 6).
            #
            # This works, and it reaches exactly ONE pixel per bias -- which is why the
            # run-level number barely moved when it landed. Splitting 261014-2308 and
            # 261015-0822 by loop position, against pixel_major on the same 16 pixels,
            # median over -1.25..-0.80 V:
            #
            #                        before this fix      after
            #     order 0  (here)    -2.53% I            -0.70% I     <- the fix
            #     order >0 (switch)  -2.42% I            -2.99% I     <- untouched
            #
            # Lighting a pixel before the step is still right -- a blanked frame
            # conducts ~4x the bitval-0 level (section 6) -- but it did NOT cause the
            # order-0 improvement once credited to it. Two later runs killed that:
            #
            #   261015-0958  the same loop at pixel_switch_settle_s=20. Order >0 went
            #                -2.99% -> -3.25%. Ten times the wait, no reduction.
            #   261015-1225  the same loop with sweep_rotate=1. The clean pixel followed
            #                the PIXEL to slot 15 (-0.22%), and the pixel moved into
            #                slot 0 came out worst in the run (-3.84%).
            #
            # So there is no privileged slot and nothing here to wait out. The
            # conditioning candidate died too (bias_major_precondition, run 261015-1321).
            #
            # And then the back-to-back A/B (261015-1421 / -1529, 68 min apart) showed
            # two thirds of the deficit was never this loop: -3.0% against a day-old
            # reference, -1.04% against a same-day one. What this loop really costs is
            # ~1% on the faint side and nothing above -1.8V.
            #
            # See config.pixel_switch_settle_s and sections 17-18.
            smile.blank_frame()
            light(smile, live[0][0], live[0][1], bitval)
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
                light(smile, x, y, bitval)
                time.sleep(cfg.pixel_switch_settle_s * cfg.settle_scale)
                row, raw_s = measure_point(pm, smu, smile, x, y, bitval, v, limit,
                                           n_pm, n_smu, cfg, t0, log, restore=False,
                                           pm_rng=pm_rng)
                if raw_s is not None:
                    samps.append((x, y, bitval, row["NVLED_V"], row["TIME_START"], raw_s))
                rows.append(row)
                i_bias = max(i_bias, abs(row["NVLED_CURR_MEAN"]))
                p_bias = max(p_bias, abs(row["PM400_POWER_MEAN"]))
                if pm_rng is not None:      # an over-range moved it up: keep it up
                    pm_rng = max(pm_rng, row.get("PM_RANGE_W", pm_rng))
                # Dead-pixel abort, judged at the FIRST bias only -- which under
                # high_to_low is the brightest, exactly as in pixel_major. Here it
                # drops the pixel from every remaining bias, so one judgement saves
                # n_points measurements instead of n_points-1.
                if cfg.dead_abort and jpt == 0 and per_member_power(row) < \
                        max(cfg.dead_abort_nw * 1e-9,
                            cfg.dead_abort_frac * float(np.median(
                                [per_member_power(r) for r in rows]))):
                    dead_here.append((x, y))
            if i_bias > 0:
                i_last, v_last = i_bias, float(v)
            if p_bias > 0:
                p_last, pv_last = p_bias, float(v)
            for px in dead_here:
                live.remove(px)
                n_dead += 1
            if dead_here:
                log(f"      dead-skip: {len(dead_here)} pixel(s) below the live floor at "
                    f"the brightest bias - dropped from the remaining {len(fine) - 1} points")

            # The boundary point on BOTH ranges, once per run, every pooled target.
            # This is the verification the pre-flight check only approximates: a real
            # sweep point, the sweep's own dwell and reference, and all the targets
            # rather than eight. See pm_range_dual_pass.
            if v == boundary_v:
                dual_rows += pm_range_dual_pass(pm, smu, smile, live, bitval, v, limit,
                                                n_pm, n_smu, rows, cfg, t0, log)

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
    if dual_rows and run_dir is not None:
        # Appended to the pre-flight pair's file, with PM_RANGE_W and NVLED_V telling
        # them apart: the pre-flight rows sit at pool_pm_range_below_v, these at the
        # sweep grid point past it.
        pfx = run_slug(cfg.sample_name, run_dir) + "_"
        path = run_dir / "NVLEDSweep" / "Sweep" / f"{pfx}pm_range_check.csv"
        prev = []
        if path.exists():
            import csv as _csv
            with path.open() as f:
                prev = [{k: (float(v) if v not in ("", None) else "")
                         for k, v in r.items()} for r in _csv.DictReader(f)]
        write_aggregated_one(path, prev + dual_rows, cols=PM_CHECK_COLS)
        log(f"  wrote {len(dual_rows)} dual-range row(s) to {path.name}")
    return n_rows


PM_CHECK_COLS = ["X", "Y", "POOL_K", "NVLED_V", "PM_RANGE_W",
                 "PM400_POWER_MEAN", "PM400_POWER_SEM", "NVLED_CURR_MEAN",
                 "PM400_POWER_REF", "NVLED_CURR_REF"]


def pm_range_boundary_check(pm, smu, smile, selected: list[Row], run_dir: Path,
                            cfg: IQEConfig, log: Logger, t0: float) -> list[Row]:
    """Read the switch bias on BOTH ranges and say whether the gain agrees.

    Section 26.17 measured +0.150 % at 0.56 sigma across the boundary, on four pixels
    at one bias on one evening. A sweep that trusts that number for thirty hours should
    verify it on its own array at its own boundary, which costs two points per checked
    target. That is what a calibration knob on real hardware is for.

    The reading is lit-minus-reference on each range in turn, so the 1154 pW offset
    between the ranges cancels exactly as it must during the sweep itself. What is left
    is the gain.
    """
    if cfg.pool_pm_range_below_v is None or not cfg.pm_range_check_targets:
        return []
    pooled = [r for r in selected if len(pool_members(int(r["X"]), int(r["Y"]))) > 1]
    pooled = pooled[:cfg.pm_range_check_targets]
    if not pooled:
        return []
    v = float(cfg.pool_pm_range_below_v)
    log(f"  PM400 range cross-check: {len(pooled)} pooled target(s) at the boundary "
        f"{v:+.2f} V on both {cfg.pm_range*1e9:.0f} and {cfg.pool_pm_range*1e9:.0f} nW "
        f"(requested). Verifying the 0.150 % of section 26.17 on THIS array.")
    limit = pick_limit(None, cfg)        # no prior current here: the ladder's default
    smu.set_voltage("b", v)
    rows: list[Row] = []
    for r in pooled:
        x, y = int(r["X"]), int(r["Y"])
        smile.blank_frame()
        light(smile, x, y, cfg.bitvals[0])
        time.sleep(cfg.vramp_highv_settle_s * cfg.settle_scale)
        for rng in (cfg.pm_range, cfg.pool_pm_range):
            # measure_point sets the range itself and flushes on a change, so the
            # reading here goes through exactly the path the sweep uses.
            row, _ = measure_point(pm, smu, smile, x, y, cfg.bitvals[0], v, limit,
                                   cfg.vramp_highv_n_pm, cfg.vramp_highv_n_smu,
                                   cfg, t0, log)
            row["PM_RANGE_W"] = rng
            rows.append({k: row.get(k, "") for k in PM_CHECK_COLS})
    smile.blank_frame()
    # fine vs coarse, paired per target
    fine = [r for r in rows if r["PM_RANGE_W"] == cfg.pm_range]
    coarse = [r for r in rows if r["PM_RANGE_W"] == cfg.pool_pm_range]
    # Reject a reading before averaging it in. Run 261020-0842's cross-check averaged
    # two NEGATIVE powers with a burst std 300x normal and reported "consistent with no
    # step" off a 28.7 % spread, because it only ever tested the MEAN's significance.
    pairs, junk = [], []
    for f, c in zip(fine, coarse):
        ok = all(abs(r["PM400_POWER_MEAN"]) > 0 and r["PM400_POWER_MEAN"] > 0
                 and abs(r["PM400_POWER_SEM"]) < 0.02 * abs(r["PM400_POWER_MEAN"])
                 for r in (f, c))
        (pairs if ok else junk).append((f, c))
    if junk:
        log(f"    !! {len(junk)} of {len(fine)} pair(s) REJECTED: a power at or below "
            f"zero, or a burst std over 2 % of the reading. That is a bad read, not a "
            f"range step, and averaging it in is how the first version of this check "
            f"called a 28.7 % spread 'consistent with no step'.")
    d = [abs(c["PM400_POWER_MEAN"]) / abs(f["PM400_POWER_MEAN"]) - 1.0
         for f, c in pairs]
    if not d:
        log(f"    !! EVERY pair was rejected, so this run has NOT verified the switch. "
            f"Treat every pooled point below {v:+.2f} V as unverified, or set "
            f"pool_pm_range_below_v=null.")
    else:
        m = float(np.median(d))
        sd = float(np.std(d, ddof=1)) if len(d) > 1 else float("nan")
        sem = sd / math.sqrt(len(d)) if len(d) > 1 else float("nan")
        sig = abs(m) / sem if sem and sem > 0 else float("nan")
        log(f"    coarse/fine = {1+m:.4f} ({100*m:+.3f} %) over {len(d)} usable "
            f"target(s), spread {100*sd:.3f} %"
            + (f", {sig:.1f} sigma on the mean" if np.isfinite(sig) else ""))
        # A wide spread VETOES the verdict. It is not evidence of no step; it is
        # evidence the measurement cannot see one.
        if np.isfinite(sd) and sd > 0.02:
            log(f"    !! the spread is {100*sd:.1f} %, far over the 1 % this switch "
                f"assumes, so this check cannot resolve a step at all -- a small mean "
                f"here means nothing. Something is wrong with the readings themselves. "
                f"Raise pm_range_check_targets, or set pool_pm_range_below_v=null.")
        elif abs(m) > 0.01:
            log(f"    !! THE TWO RANGES DISAGREE BY {100*m:+.2f} %, over the 1 % this "
                f"switch assumes. Every pooled point below {v:+.2f} V carries that as "
                f"a step the analysis cannot see. Set pool_pm_range_below_v=null and "
                f"stop the sweep short of the bright end instead.")
        elif np.isfinite(sig) and sig >= 3.0:
            log(f"    a real but small step: {100*m:+.3f} % at {sig:.1f} sigma. Under "
                f"1 %, so the switch stands, but it is a systematic between the two "
                f"segments and belongs in any error budget that spans them.")
        else:
            log(f"    consistent with no step -- the switch is free, as section 26.17 "
                f"measured.")
    pfx = run_slug(cfg.sample_name, run_dir) + "_"
    write_aggregated_one(run_dir / "NVLEDSweep" / "Sweep" / f"{pfx}pm_range_check.csv",
                         rows, cols=PM_CHECK_COLS)
    return rows


def run_sweep_voltage(pm, smu, smile, selected: list[Row], run_dir: Path,
                      cfg: IQEConfig, log: Logger) -> int:
    """Voltage-source adaptive sweep. Single bitval, low-J-weighted spacing, and
    longer integration on the low-current rising side (per-point NPLC)."""
    _ref_warned.clear()
    _ref_cache.clear()
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
               else f"fixed {cfg.nvled_range_i:g} A range" if cfg.nvled_range_i
               else "range from the limit (High-C off, same ladder)")
        log(f"VOLTAGE SWEEP — {len(selected)} pixels, {cfg.sweep_direction}, V ramp "
            f"{cfg.nvled_v_coarse_start}→{cfg.nvled_v_coarse_stop} V ({cfg.n_fine} pts, low-J-dense @ knee "
            f"{cfg.vramp_knee_v} V), {rng}, {eff}")
        if cfg.leak_reference and cfg.leak_ref_per_point_above_v is None:
            log(f"  leakage reference: every point re-measured at bitval {cfg.leak_ref_bitval} "
                f"(+{cfg.leak_ref_settle_s:g}s each) → NVLED_CURR_REF. I_LED = MEAN - REF.")
        elif cfg.leak_reference:
            log(f"  leakage reference: per point at bitval {cfg.leak_ref_bitval} above "
                f"{cfg.leak_ref_per_point_above_v:+.2f} V (+{cfg.leak_ref_settle_s:g}s each), "
                f"cached per pixel at or below it from one pre-pass at "
                f"{cfg.leak_ref_prepass_v:+.2f} V. I_LED = MEAN - REF, and "
                f"LEAK_REF_CACHED says which. The settle estimate below counts the "
                f"reference on EVERY point, so it is an upper bound -- on the default "
                f"30-point grid the cache removes it from 22 of them.")
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
    # Before either walk, and after the fast scan has lit every pixel once -- a pixel
    # referenced stone cold reads 10-22 nA high (runs 261016-1113/-1221, where
    # low_to_high anchors on a cold -0.80 V), and that is the one error this pass can
    # avoid for free. It leaves the SMU on the faint-side limit and NPLC, which both
    # walks reconfigure on their first point anyway.
    prefill_reference_cache(pm, smu, smile,
                            [(int(r["X"]), int(r["Y"])) for r in selected], cfg, log)
    configure_voltage_source_b(smu, cfg, log)
    pm_range_boundary_check(pm, smu, smile, selected, run_dir, cfg, log,
                            time.perf_counter())
    if cfg.pool_pm_range_below_v is not None:
        # leave it where an unpooled point expects it, and flush so the sweep's first
        # point is not the throwaway
        _, ch = set_pm_range(pm, cfg.pm_range)
        if ch:
            flush_pm_after_range(pm, smu, cfg.vramp_highv_n_smu,
                                 cfg.vramp_highv_n_pm, cfg)
    sweep_dir = run_dir / "NVLEDSweep" / "Sweep"
    pfx = run_slug(cfg.sample_name, run_dir) + "_"
    (sweep_dir / "raw_data").mkdir(parents=True, exist_ok=True)
    # Config up front so it exists even if the run is interrupted, and with it the
    # commit that produced the run -- see measure_iqe/provenance.py.
    (sweep_dir / "raw_data" / f"{pfx}Sweep_config.json").write_text(
        json.dumps({"settings": cfg.to_dict(), "provenance": provenance()}, indent=2),
        encoding="utf-8")
    # Self-contained background writer (no external deps): raw + aggregate are flushed off the
    # measurement thread, so slow (Nextcloud/network) flushes don't block the sweep, and the
    # aggregate is written per pixel → an interrupted run is directly analysable. try/finally
    # ensures close() drains the queue and joins on a clean stop or caught interrupt.
    settle_traces: list = []          # (tag, t, I, P) per point, kept for offline work
    skip = completed_pixels(run_dir) if cfg.resume else set()
    if skip:
        log(f"  resuming: {len(skip)} pixel(s) already complete, "
            f"{sum(1 for r in selected if (int(r['X']), int(r['Y'])) not in skip)} to go")
    writer = BackgroundWriter(sweep_dir / "raw_data" / f"{pfx}Sweep.csv", sweep_dir,
                              cfg, log, append=bool(skip), prefix=pfx)
    sw_rows = []
    live_peaks = []                 # brightest-point power (W) of confirmed-live pixels
    p_first_prev = None             # first-point power of the previous pixel (PM range)
    n_dead = 0
    t0 = time.perf_counter()
    try:
      if cfg.sweep_order == "bias_major":
        # Shares the writer, the trace list and the finally below; only the walk differs.
        return _sweep_bias_major(pm, smu, smile, selected, writer, settle_traces,
                                 cfg, log, t0, run_dir)
      for i, r in enumerate(selected, 1):
        x, y = r["X"], r["Y"]
        if (int(x), int(y)) in skip:
            log(f"  [{i}/{len(selected)}] pixel ({x},{y}) - already complete, skipping")
            continue
        log(f"  [{i}/{len(selected)}] pixel ({x},{y})")
        prows = []
        psamps = []                              # per-sample bursts for this pixel (raw file)
        last_v = cfg.fastscan_v_list()[0]
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
            light(smile, x, y, bitval)
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
            # The PM400 range follows the same rule along the pixel; its first point
            # starts from the previous pixel's first-point power.
            pm_lad = pm_ladder(pm, cfg, log) if cfg.pm_autorange else None
            p_px = pv_px = None
            if pm_lad:
                p_px, pv_px = fs_power_anchor(r), float(cfg.nvled_v_fastscan)
                if not p_px:          # no fast-scan reading: the previous pixel's first
                    p_px, pv_px = p_first_prev, (fine[0][0] if fine else None)
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
                    # The LIMIT tracks the measured current, not the density knee,
                    # scaled to this bias by the median I(V).
                    limit = pick_limit(i_last, cfg, last_v if i_last is not None else None, v)
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
                if (not cfg.perpoint_range and not cfg.highc_b and cfg.nvled_range_i
                        and cfg.nvled_range_i != cur_range):
                    # In High-C the limit governs the range — writing it here is the
                    # conflict that made earlier High-C runs fail. Leave it alone.
                    set_measure_range_b(smu, cfg.nvled_range_i); cur_range = cfg.nvled_range_i
                pm_rng = None
                if pm_lad:
                    pm_rng = pick_pm_range(
                        p_px * bias_scale(v, pv_px)[0] if p_px else None, cfg, pm_lad)
                row, raw_s = measure_point(pm, smu, smile, x, y, bitval, v, limit,
                                           n_pm, n_smu, cfg, t0, log, pm_rng=pm_rng)
                p, inv = row["PM400_POWER_MEAN"], row["NVLED_CURR_MEAN"]
                if pm_lad:
                    p_px, pv_px = abs(p), v
                    if jpt == 0:
                        p_first_prev = abs(p)
                if raw_s is not None:
                    psamps.append((x, y, bitval, row["NVLED_V"], row["TIME_START"], raw_s))
                i_last = abs(inv)
                prows.append(row); sw_rows.append(row)
                last_v = v
                # Early dead-pixel abort: judge on the first (brightest) point only.
                if cfg.dead_abort and jpt == 0:
                    # Per member, so a pooled target is judged on the same scale as a
                    # single pixel and cannot poison the running live median.
                    first_p = per_member_power(row)
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
            npz = sweep_dir / f"{pfx}settling_traces.npz"
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

# ----------------------------------------------------------------------
#  Wall-clock estimate, before anything is measured
# ----------------------------------------------------------------------
# Measured on the bench from the CSV timestamps of runs 261020-0842 (Array42) and
# 261022-0037 (Array38), identical code, bias_major:
#   * one read = the SMU burst, n_smu samples x 2 channels x (nplc/50 s + 0.66 ms). The
#     PM400 reads run while the SMU integrates (acquire_point fires the burst first), so
#     they only count if they take longer. That reproduces every bias_major run since
#     14 October to the millisecond: 0.976 s rising (12 x 2 x 40.66 ms), 0.325 s droop.
#     PM400 at pm_averaging 3: 32 reads fit inside 0.976 s, so <= 30 ms each; 25 ms used.
#   * one bias change = its dwell + ~4 s (77.8 s at a 73.14 s dwell, 112.2 s at 107.59).
#   * one fast-scan pixel = fastscan_settle_ms + 20.7 ms (262 144 px in 2 h 14 min).
#   * the reference pre-pass = leak_ref_tries x (leak_ref_settle_s + one rising read)
#     per target (385 targets in 3442 s); the PM-range check = one droop dwell plus two
#     points per checked group (8 groups in 603 s).
#   * NOT modelled: pixel_major runs that take the reference at every point ran ~1.2 s
#     per point slower than this (261014-0048, -1616, 261015-1421), bias_major runs of
#     the same days and code did not. The cause is in that loop and not yet found.
# Re-measure these if the instruments or the acquisition code change.
EST_SMU_SAMPLE_OVERHEAD_S = 0.00066
EST_PM_READ_S = 0.025
EST_BIAS_OVERHEAD_S = 4.0
EST_FASTSCAN_PX_S = 0.0207


def _acq_s(nplc: float, n_smu: int, n_pm: int) -> float:
    smu = n_smu * 2 * (nplc / 50.0 + EST_SMU_SAMPLE_OVERHEAD_S)
    return max(smu, n_pm * EST_PM_READ_S)


def estimate_targets(cfg: IQEConfig) -> int:
    """Targets the selection would pick if every pixel in the ROI were alive.

    Counted from the block grid instead of running the selection, which takes ~40 s on
    the full array. A dark block gives no target, so a mostly dark array gets fewer.
    """
    x1, y1, x2, y2 = cfg.resolve_roi()
    npx = (abs(x2 - x1) + 1) * (abs(y2 - y1) + 1)
    if cfg.sweep_select == "blocks":
        nbx = len(block_edges(min(x1, x2), max(x1, x2), cfg.block_size)) - 1
        nby = len(block_edges(min(y1, y2), max(y1, y2), cfg.block_size)) - 1
        return nbx * nby * (cfg.pixels_per_block + len(cfg.pool_groups))
    if cfg.sweep_select == "stride":
        return min(-(-npx // max(1, cfg.nth_pixel_sweep)), cfg.max_sweep_pixels)
    return min(max(cfg.min_sweep_pixels, int(round(cfg.sweep_fraction * npx))),
               cfg.max_sweep_pixels, npx)


def estimate_run(cfg: IQEConfig, n_targets: int | None = None) -> dict[str, float]:
    """Seconds per stage of a run with this config, from the config alone.

    An UPPER bound on the sweep: every target is assumed alive and swept at every
    bias, and ``dead_abort`` drops some after the first one (44 of 385 in 261022-0037,
    which made that run 0.6 h shorter than this says). Voltage mode only is exact;
    ``fine_sweep_mode='current'`` places its grid from a pre-sweep and is approximated
    by ``n_fine`` points on the rising-side effort.
    """
    x1, y1, x2, y2 = cfg.resolve_roi()
    npx = (abs(x2 - x1) + 1) * (abs(y2 - y1) + 1)
    k = cfg.settle_scale
    frac = (1.0 / cfg.fastscan_pool + cfg.fastscan_pool_rescan_frac) if cfg.fastscan_pool else 1.0
    out: dict[str, float] = {
        "fast scan": npx * frac * len(cfg.fastscan_v_list())
        * (cfg.fastscan_settle_ms / 1000.0 + EST_FASTSCAN_PX_S)}
    if cfg.fastscan_only:
        out["total"] = out["fast scan"]
        return out

    T = estimate_targets(cfg) if n_targets is None else n_targets
    rising = (cfg.vramp_lowv_nplc, cfg.vramp_lowv_n_smu, cfg.vramp_lowv_n_pm,
              cfg.vramp_lowv_settle_s)
    droop = (cfg.vramp_highv_nplc, cfg.vramp_highv_n_smu, cfg.vramp_highv_n_pm,
             cfg.vramp_highv_settle_s)
    flat = (cfg.sweep_nplc, cfg.n_smu_samples, cfg.n_pm_samples, cfg.settle_ms / 1000.0)
    if cfg.fine_sweep_mode == "voltage":
        grid = [v for v, _ in build_fine_voltages(None, None, cfg, lambda _m: None)]
    else:
        grid = [cfg.nvled_v_coarse_start] * cfg.n_fine
    above = cfg.leak_ref_per_point_above_v

    def seg(v):
        if not cfg.vramp_perseg_effort:
            return flat
        return rising if (cfg.vramp_knee_v is None or v > cfg.vramp_knee_v) else droop

    def ref(v):
        return cfg.leak_reference and (above is None or v > above)

    if cfg.leak_reference and above is not None:
        out["reference pre-pass"] = T * max(1, cfg.leak_ref_tries) * (
            cfg.leak_ref_settle_s * k + _acq_s(*rising[:3]))
    n_groups = T * len(cfg.pool_groups) / max(1, cfg.pixels_per_block + len(cfg.pool_groups))
    if cfg.pool_groups and cfg.pool_pm_range_below_v is not None and cfg.pm_range_check_targets:
        out["PM-range check"] = min(cfg.pm_range_check_targets, n_groups) * (
            droop[3] * k + 2 * (cfg.pixel_switch_settle_s * k + _acq_s(*droop[:3])
                                + cfg.pm_range_settle_s))
    dwell = points = 0.0
    if cfg.sweep_order == "bias_major":
        if cfg.bias_major_precondition:
            out["precondition"] = T * cfg.warmup_ms / 1000.0
        for v in grid:
            nplc, n_smu, n_pm, settle = seg(v)
            a = _acq_s(nplc, n_smu, n_pm)
            dwell += settle * k + EST_BIAS_OVERHEAD_S
            points += T * (cfg.pixel_switch_settle_s * k + a
                           + (cfg.leak_ref_settle_s * k + a if ref(v) else 0.0))
    else:
        for v in grid:
            nplc, n_smu, n_pm, settle = seg(v)
            a = _acq_s(nplc, n_smu, n_pm)
            dwell += T * settle * k
            points += T * (a + (2 * cfg.leak_ref_settle_s * k + a if ref(v) else 0.0))
        points += T * cfg.warmup_ms / 1000.0
    out["bias dwell"] = dwell
    out["sweep points"] = points
    out["total"] = sum(out.values())
    out["targets"] = T
    out["points"] = T * len(grid)
    return out


def format_estimate(est: dict[str, float]) -> str:
    """One line per stage, as the log and the console show it."""
    lines = []
    for name, s in est.items():
        if name in ("targets", "points"):
            continue
        lines.append(f"  {name:<20} {s / 3600:6.2f} h"
                     + (f"   ({100 * s / est['total']:.0f} %)" if name != "total" else ""))
    if "targets" in est:
        lines.append(f"  {int(est['targets'])} target(s) x {int(est['points'] / max(1, est['targets']))}"
                     f" bias(es), every target assumed alive: an upper bound")
    return "\n".join(lines)


class IQEResult(NamedTuple):
    """What a finished run produced."""

    run_dir: Path           # directory holding FastScan/ and NVLEDSweep/
    n_pixels: int           # pixels selected for the full sweep
    n_points: int           # sweep points written (0 for a warm-up diagnostic)
    config: IQEConfig       # the configuration actually used


def stamp_run(out_dir: Path, cfg: IQEConfig, log: Logger) -> None:
    """Record what produced this run: the commit, and the calibration that was applied.

    Both are written before any measurement, so an interrupted run is still identified.
    The calibration is COPIED rather than referenced: `calibrations/` is gitignored and
    gets overwritten by the next --calibrate, so a path alone would rot.
    """
    prov = provenance()
    write_provenance(out_dir, {"sample_name": cfg.sample_name,
                               "calibration_source": cfg.calibration_source})
    # Name the files. "1 file(s)" sent the reader to git status weeks later to find out
    # whether it was a source file or a scratch run.bat, and the answer changes whether
    # the run is reproducible. Untracked files are listed separately and are harmless.
    edits = prov.get("changed") or []
    stray = prov.get("untracked") or []
    note = ""
    if prov["dirty"]:
        note = (f"  !! UNCOMMITTED EDITS to {', '.join(edits[:4])}"
                + (f" (+{len(edits) - 4} more)" if len(edits) > 4 else "")
                + ": this run cannot be reproduced from the commit alone")
    elif stray:
        note = (f"  (untracked, does not affect the code that ran: "
                + ", ".join(stray[:3]) + (f" +{len(stray) - 3}" if len(stray) > 3 else "")
                + ")")
    log(f"  code: {prov['describe']} on {prov['branch']}{note}")
    src = Path(cfg.calibration_source) if cfg.calibration_source else None
    if src and src.is_file():
        dst = out_dir / f"{run_slug(cfg.sample_name, out_dir)}_calibration_applied{src.suffix}"
        try:
            dst.write_bytes(src.read_bytes())
            log(f"  calibration: {src}  (copied to the run as {dst.name})")
        except OSError as e:                      # a share went away; not worth aborting
            log(f"  calibration: {src}  !! could not copy into the run: {e}")
    elif src:
        log(f"  calibration: {src}  !! NO LONGER ON DISK - not copied")


def make_run_dir(cfg: IQEConfig, base_dir: str | Path | None = None,
                 suffix: str | None = None, simulated: bool = False) -> Path:
    """Create and return ``<base_dir>/<stamp>_<sample>_ROI_(x1, y1, x2, y2)``.

    A simulated run gets ``SIMULATED_`` in front of the sample name. Without it a
    software-rig run lands in results/ looking exactly like a hardware one: 24 did,
    among them eight calibrations named Array42_IQE written while Array38 was on the
    stage (report, Appendix C.7).
    """
    x1, y1, x2, y2 = cfg.resolve_roi()
    stamp = datetime.datetime.now().strftime("%y%m%d-%H%M")
    root = Path(base_dir) if base_dir is not None else Path(cfg.base_dir)
    tail = f"_{suffix}" if suffix else ""
    name = ("SIMULATED_" if simulated else "") + cfg.sample_name
    run_dir = root / f"{stamp}_{name}_ROI_({x1}, {y1}, {x2}, {y2}){tail}"
    # The stamp has minute resolution, so two runs started in the same minute with the
    # same sample and ROI would write into one directory and silently merge. Only the
    # resume path may reuse an existing directory, and it passes run_dir explicitly.
    if run_dir.exists() and any(run_dir.iterdir()):
        stamp2 = datetime.datetime.now().strftime("%y%m%d-%H%M%S")
        run_dir = root / f"{stamp2}_{name}_ROI_({x1}, {y1}, {x2}, {y2}){tail}"
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


def calibration_pixels(pm, smu, smile, out_dir: Path, cfg: IQEConfig,
                       log: Logger) -> list[tuple[int, int]]:
    """Which pixels the dwell is measured on. Every quadrant, from a small window each.

    The dwell this produces governs the whole sweep, so a calibration that samples one
    corner sets the array's dwell from that corner. Array38's first one ran on
    roi=(32,32,63,63) and picked (35,34), (32,46) and (35,54) -- three neighbours
    inside TL, the one quadrant the capacitance run had measured as slow (108 s against
    ~1 s in TR, BL and BR). It recommended 15.23 s on that evidence, and the three
    pixels disagreed among themselves by 3.9x (2.62 / 4.58 / 10.15 s).

    So: a centred window per quadrant, which is what characterize_capacitance has always
    done. Four 24x24 windows is ~2300 pixels, about 70 s, against the two hours a
    full-ROI fast scan costs before the first capture even starts.

    ``calibrate_per_quadrant = 0`` restores the old single-ROI behaviour.
    """
    if not cfg.calibrate_per_quadrant:
        x1, y1, x2, y2 = cfg.resolve_roi()
        sel = run_fast_scan(pm, smu, smile, x1, y1, x2, y2, out_dir, cfg, log)
        return [(int(r["X"]), int(r["Y"])) for r in sel[:cfg.calibrate_pixels]]

    win = max(2, int(cfg.calibrate_quadrant_window))
    out: list[tuple[int, int]] = []
    for quad in ("TL", "TR", "BL", "BR"):
        qx1, qy1, qx2, qy2 = NAMED_ROIS[quad]
        cx, cy = (qx1 + qx2) // 2, (qy1 + qy2) // 2
        wx1, wy1 = max(qx1, cx - win // 2), max(qy1, cy - win // 2)
        wx2, wy2 = min(qx2, wx1 + win - 1), min(qy2, wy1 + win - 1)
        log(f"  {quad}: calibration window ({wx1},{wy1})-({wx2},{wy2})")
        # The sweep's sampler cannot be handed a 24 px window with the sweep's own
        # 64 px blocks: it reported "0 non-empty blocks" out of 532 alive pixels.
        # One block = the whole window, as characterize_capacitance does. Pooling is
        # off because a pooled target is K pixels and the dwell is about one.
        wcfg = dataclasses.replace(
            cfg, roi=(wx1, wy1, wx2, wy2), sweep_select="blocks", block_size=win,
            pixels_per_block=cfg.calibrate_per_quadrant,
            max_sweep_pixels=cfg.calibrate_per_quadrant,
            pool_groups=(), fastscan_pool=0, sweep_rotate=0, fastscan_remeasure_n=0)
        # Its own directory per quadrant: run_fast_scan writes a fixed set of file
        # names, so four calls into one directory would overwrite each other.
        sel = run_fast_scan(pm, smu, smile, wx1, wy1, wx2, wy2,
                            out_dir / f"FastScan_{quad}", wcfg, log)
        got = [(int(r["X"]), int(r["Y"])) for r in sel[:cfg.calibrate_per_quadrant]]
        if not got:
            log(f"    {quad}: no alive pixel in the window -- this quadrant is not "
                f"represented in the dwell")
        out += got
    return out


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
    out_dir = make_run_dir(cfg, base_dir, suffix="CALIBRATION", simulated=simulate)
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
        pixels = calibration_pixels(pm, smu, smile, out_dir, cfg, log)
        if not pixels:
            raise RuntimeError("no alive pixels found to calibrate on")
        data = calibrate(pm, smu, smile, pixels, cfg, log, out_dir=out_dir)
    finally:
        parked.__exit__(None, None, None)
    record, stable = write_calibration(out_dir, data, simulated=simulate)
    log("=" * 56)
    log(f"CALIBRATION for {cfg.sample_name}")
    log(f"  record  : {record}")
    if stable is None:
        log(f"  per-array: NOT written -- this was --simulate. "
            f"calibrations/{cfg.sample_name}.yaml is a measured fact about a physical "
            f"array and a simulated run must not replace it.")
    else:
        log(f"  per-array: {stable}")
        log(f"Use it with:  python -m measure_iqe --array {cfg.sample_name}")
    # Always a usable path. Under --simulate there is no per-array copy by design, but
    # the run-directory record is a complete calibration file and --calibration <path>
    # loads it, so returning None would make a simulated calibration unusable rather
    # than merely unpublished.
    return stable or record


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
    out_dir = Path(run_dir) if run_dir is not None else make_run_dir(
        cfg, base_dir, suffix="FASTSCAN" if cfg.fastscan_only else None, simulated=simulate)
    out_dir.mkdir(parents=True, exist_ok=True)
    log(f"Output: {out_dir}   (source_mode={cfg.source_mode})")
    stamp_run(out_dir, cfg, log)
    log("estimated wall time, from the config:\n" + format_estimate(estimate_run(cfg)))

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
        smu.set_voltage("b", cfg.fastscan_v_list()[0])  # voltage mode for the fast scan
        smu.enable_output("a", True)
        smu.enable_output("b", True)
        time.sleep(0.05)

        selected = run_fast_scan(pm, smu, smile, x1, y1, x2, y2, out_dir, cfg, log)
        if cfg.fastscan_only:
            n_points = (x2 - x1 + 1) * (y2 - y1 + 1) * len(cfg.fastscan_v_list())
            log(f"fastscan_only: {n_points} fast-scan points written, no sweep.")
        elif not selected:
            log("No alive pixels selected - nothing to sweep.")
        elif warmup_diag:
            # Singles only. The diagnostic watches one pixel turn on and relax; a
            # pooled target would be driven as its first member alone here, which is a
            # different measurement reported under the same coordinates.
            pixels = [(r["X"], r["Y"]) for r in selected
                      if "POOL_MEMBERS" not in r][:cfg.warmup_diag_n_pixels]
            run_warmup_diag(pm, smu, smile, pixels, out_dir, cfg, log)
        elif cfg.source_mode == "current":
            n_points = run_sweep_current(pm, smu, smile, selected, out_dir, cfg, log)
        else:
            n_points = run_sweep_voltage(pm, smu, smile, selected, out_dir, cfg, log)
        log("=" * 56)
        if selected and not warmup_diag and not cfg.fastscan_only:
            log(f"DONE. {n_points} sweep points over {len(selected)} pixels.")
            log('Analyse it again any time with:\n'
                f'    python analysis/analyze_array.py "{out_dir}"')
    finally:
        parked.__exit__(None, None, None)      # idempotent: shutdown() runs exactly once
    return IQEResult(run_dir=out_dir, n_pixels=len(selected), n_points=n_points, config=cfg)
