"""Characterise the array's capacitance, per pixel, and how low in J you can actually go.

The array carries a large capacitor in parallel with the LED. Moving the cathode
bias across the rising-side span (-0.8 -> -1.3 V) moves ``C*dV ~ 25 uC`` of charge
at C = 50 uF. At LED currents (1-10 uA) that is seconds per pixel; at the SMU's
compliance it is microseconds. Which of those supplies the charge decides whether a
low-J current reading is right, and it plausibly explains the direction hysteresis,
the fine ranges reading low J wrong, the oscillation seen in the buffer, and the
"slow parasitic current" the warm-up diagnostic found.

**That C is the same everywhere is an assumption**, so every pixel-dependent
experiment runs on several pixels per quadrant. Each pixel is checked for light
first, so a long run is never spent on a dead one.

Current density on a 10x10 um pixel: ``J [A/cm2] = I [A] / 1e-6``, so

    1 uA -> J = 1        300 nA -> J = 0.3        100 nA -> J = 0.1

Getting below J = 1 means measuring at and under a microamp. See ``optfloor``
before planning that: the optical signal, not the current, is the binding limit.

Experiments (``--what``):

  step      I(t) after a voltage step at the lowest NPLC, per voltage x range x
            High-C state. Settled value, time to 1% and 0.1%, overshoot, ripple.
  leak      Settled current and light at each bias, pixel LIT and BLANKED. Both
            source modes read the TOTAL cathode current, so neither separates LED
            injection from leakage; blanking at the same bias removes the LED and
            leaves the leakage, and lit minus blanked is the true LED current. The
            cap ramps implied a few hundred nA of parallel current near -0.35 V,
            comparable to the 249 nA measured at -0.70 V, so this decides whether
            the apparent EQE collapse below J ~ 0.4 is physics or a contaminated
            denominator.

  cap       C per pixel from a known current and the voltage slope it produces:
            C = I/(dV/dt). Park below turn-on, hand the node to a current source,
            time the ramp. Needs only a voltage slope, so it is immune to the
            current-range problem that defeats every other approach here. Run
            blanked and lit; a lit-vs-blanked difference is a per-pixel
            contribution.
  isrc      ch B sinks a set current and HOLDS FOR MINUTES, logging chA, chB, V
            and dV/dt. The earlier rejection of current-source mode was measured
            with a 15 ms settle, which at 0.3 uA into 50 uF moves the node by
            0.09 mV -- the operating point never moved. If chA converges toward
            chB's setpoint as the voltage settles, there is no bypass path.
  stepdir   One voltage reached from a small and from a large step. A parasitic
            depends on time since turn-on; a capacitor depends on charge moved.
            If settling scales with |dV|, it is the capacitor.
  decay     Charge-decay. Charge the cap, then source 0 A so the node floats and
            discharges through the LED, logging V(t) and light. Then
            ``I = C*dV/dt``, which at 100 nA into 50 uF is 2 mV/s -- a far easier
            measurement than 100 nA of current. Run lit and blanked; the
            difference is the LED current. This is the route to J < 1.
  optfloor  The PM400 noise floor vs averaging, array blanked. Decides how low in
            J you can go at all, since the light scales with the current.
  cluster   K pixels lit at once (K = 1, 4, 16), pooled P and I at one faint bias.
            optfloor says how big the meter noise is; this says whether it
            AVERAGES DOWN. If the relative scatter falls as 1/sqrt(K) the floor is
            the meter and a cluster buys a cleaner faint end; if it stays at 8.5 %
            the scatter is the array. The reference pools too -- the same K pixels
            at bitval 0, never a blanked frame. Needs no --pixels.

Usage:
    python characterize_capacitance.py --what all
    python characterize_capacitance.py --what step cap --per-quadrant 3
    python characterize_capacitance.py --what isrc --pixels 54,33 177,63 --isrc-seconds 300
    python characterize_capacitance.py --what optfloor          # 2 minutes, no pixels needed
    python characterize_capacitance.py --what cluster --highc --compliance 1e-5

Writes one CSV per experiment, every row tagged with its pixel. It fits and plots
nothing; point ``analysis/`` at the CSVs for that.

NOTE: ``--simulate`` exercises the plumbing only. The built-in rig has no
capacitance model, so it reports instant settling. The physics needs the bench.
"""
from __future__ import annotations

import argparse
import csv
import dataclasses
import datetime
import json
import math
import re
import threading
import time
from pathlib import Path

import numpy as np

import measure_iqe as m
from measure_iqe.calibrate import RECORD_LIMITED_FRAC, SMOOTH_MAX_S
from measure_iqe import IQEConfig
from measure_iqe.safeexit import safe_exit

# Voltages worth probing: the sub-turn-on region that reaches J < 1, the rising side
# that pins peak IQE, the effort-split knee, and the droop.
DEFAULT_VOLTAGES = [-0.6, -0.7, -0.8, -0.9, -1.0, -1.1, -1.3, -1.8, -2.6]
# The 10 mA range is what the clean runs used; the fine ones read low J "not just
# noisily but WRONG" before, which is the thing to explain.
DEFAULT_RANGES = [1e-3, 1e-4, 1e-5, 1e-6, 1e-7]
# step's own default: four 0.05-0.20 V steps from -1.00 V. This is the configuration
# behind the README's "the settling is not RC" -- excess ~ t^-1.2 with amplitude
# proportional to the step -- which the old -2.6 V start and nine-bias span did not
# reproduce. Varying the STEP SIZE from a fixed start is what established the power law.
STEP_VOLTAGES = [-0.95, -0.90, -0.85, -0.80]
QUADRANTS = ("TL", "TR", "BL", "BR")
# The 2600 returns 9.91e37 for an over-range reading. It is a sentinel, not a number:
# left in, it poisons every median and max it touches.
OVERFLOW = 1e36


def to_j(i_a, cfg):
    """Current density in A/cm2 for this pixel geometry."""
    return i_a / (cfg.pixel_area_um2 * 1e-8)


# ----------------------------------------------------------------------
#  pixel selection + liveness
# ----------------------------------------------------------------------

def read_power(pm, n):
    vals = [pm.measure() for _ in range(int(n))]
    a = np.asarray([v for v in vals if v is not None and np.isfinite(v)], float)
    if a.size == 0:
        return float("nan"), float("nan")
    return float(a.mean()), float(a.std(ddof=1) / np.sqrt(a.size)) if a.size > 1 else 0.0


def charge_time_s(capacitance, dv, limit_a):
    """How long the SMU needs to move the array capacitance through dv at its limit.

    The limit is the only current available to charge C, so this is a hard floor on
    any settle time. It is what made a 10 uA limit look like a dead array: 50 uF
    through 1.35 V at 10 uA is 6.75 s, and the liveness check waited 10 ms.
    """
    if not limit_a:
        return 0.0
    return abs(capacitance * dv / limit_a)


def check_alive(pm, smu, smile, cfg, x, y, args, log):
    """Fast turn-on check. Returns (power_W, dark_W, alive).

    Deliberately uses its own channel-B setup rather than the measurement one. This
    is a go/no-go light check, not a measurement, so it wants a healthy current
    limit to charge the array fast. Inheriting a low measurement compliance made
    every pixel read dead, because the node had not reached the bias yet. Every
    experiment reconfigures ch B itself, so changing it here is safe.
    """
    probe = dataclasses.replace(cfg, highc_b=False,
                                nvled_compliance=args.alive_compliance)
    m.configure_voltage_source_b(smu, probe, log)
    settle = max(cfg.fastscan_settle_ms / 1000.0, 0.01,
                 3 * charge_time_s(args.capacitance, cfg.nvled_v_fastscan,
                                   args.alive_compliance))
    smile.blank_frame()
    smu.set_voltage("b", cfg.nvled_v_fastscan)
    time.sleep(settle)
    dark, _ = read_power(pm, args.alive_n_pm)
    smile.set_pixel(x, y, cfg.fastscan_bitval)
    time.sleep(settle)
    p, _ = read_power(pm, args.alive_n_pm)
    smile.blank_frame()
    alive = (p - dark) >= args.alive_nw * 1e-9
    log(f"    ({x},{y}) turn-on {(p-dark)*1e9:+8.2f} nW over dark "
        f"{dark*1e9:+.2f} nW  (probe limit {args.alive_compliance:g} A, "
        f"settle {settle*1e3:.0f} ms) -> {'ALIVE' if alive else 'DEAD, skipping'}")
    return p, dark, alive


def select_pixels(pm, smu, smile, cfg, args, log):
    """Explicit ``--pixels``, else N alive pixels per quadrant from a scan window.

    The auto path scans a window at the centre of each quadrant and hands the alive
    pixels to ``select_sweep_pixels`` in blocks mode, so the picks inherit the
    minimum-separation guarantee instead of landing next to each other.
    """
    if args.pixels:
        out = [tuple(int(v) for v in p.split(",")) for p in args.pixels]
        log(f"  {len(out)} pixel(s) given explicitly")
        return [dict(X=x, Y=y, quadrant="given") for x, y in out]

    picked = []
    # Same reasoning as check_alive, and the same bug if omitted: this is a brightness
    # scan, not a measurement, so it needs a healthy limit. Inheriting a low
    # measurement compliance cannot charge the array to the fast-scan bias (45.7 uF
    # through 1.35 V at 10 uA is 6.2 s per pixel), so every pixel would read dark and
    # the run would find nothing to characterise.
    probe = dataclasses.replace(cfg, highc_b=False,
                                nvled_compliance=args.alive_compliance)
    m.configure_voltage_source_b(smu, probe, log)
    settle = max(cfg.fastscan_settle_ms / 1000.0, 0.005,
                 3 * charge_time_s(args.capacitance, cfg.nvled_v_fastscan,
                                   args.alive_compliance))
    smu.set_voltage("b", cfg.nvled_v_fastscan)
    time.sleep(max(settle, 0.2))
    log(f"  scan probe: limit {args.alive_compliance:g} A, {settle*1e3:.0f} ms/pixel settle")
    for quad in args.quadrants:
        x1, y1, x2, y2 = m.NAMED_ROIS[quad]
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        half = args.quadrant_window // 2
        wx1, wy1 = max(x1, cx - half), max(y1, cy - half)
        wx2, wy2 = min(x2, wx1 + args.quadrant_window - 1), min(y2, wy1 + args.quadrant_window - 1)
        log(f"  {quad}: scanning window ({wx1},{wy1})-({wx2},{wy2})")
        rows = []
        smile.blank_frame()
        time.sleep(0.02)
        dark, _ = read_power(pm, args.alive_n_pm)
        for y in range(wy1, wy2 + 1):
            for x in range(wx1, wx2 + 1):
                smile.blank_frame()
                smile.set_pixel(x, y, cfg.fastscan_bitval)
                time.sleep(settle)
                p, _ = read_power(pm, 2)
                rows.append(dict(X=x, Y=y, PM400_POWER_MEAN=p - dark))
        smile.blank_frame()
        alive = [r for r in rows if r["PM400_POWER_MEAN"] >= args.alive_nw * 1e-9]
        log(f"    alive {len(alive)}/{len(rows)}")
        if not alive:
            continue
        # reuse the blocks sampler for its separation guarantee: one block = the window
        wcfg = dataclasses.replace(cfg, roi=(wx1, wy1, wx2, wy2),
                                   block_size=max(wx2 - wx1 + 1, wy2 - wy1 + 1),
                                   pixels_per_block=args.per_quadrant)
        for r in m.select_sweep_pixels(alive, wcfg, lambda _s: None):
            picked.append(dict(X=r["X"], Y=r["Y"], quadrant=quad))
        log(f"    picked {[(p['X'], p['Y']) for p in picked if p['quadrant'] == quad]}")
    return picked


# ----------------------------------------------------------------------
#  capture primitive
# ----------------------------------------------------------------------

# MEASURED on the instrument, four points over two channel counts and two NPLCs:
#   2 channels: charatests/261013-2140 and -2155 (NPLC 0.01, 1060 us),
#               -2201 and -2337        (NPLC 1.0, 40670 us)  -> 2.001*nplc/50 + 660 us
#   1 channel : results/261021-1051    (NPLC 0.01, 291 us; NPLC 1.0, 20145 us)
#                                                            -> 1.003*nplc/50 +  91 us
# The integration scales with the channel count, as it must -- the burst issues one
# measure call per channel. The OVERHEAD does not: it is per measure-CALL, not per
# sample, so one channel costs 91 us and two cost 660 us rather than 182. A single
# constant cannot describe both, which is why this is a table and not a formula.
SAMPLE_OVERHEAD_S = {1: 91e-6, 2: 660e-6}


def sample_interval_s(nplc: float, channels: int = 2) -> float:
    """Seconds per buffered sample at this NPLC and channel count. Measured.

    Matches all four bench points to 0.3 %. An unmeasured channel count falls back to
    the two-channel overhead, which is the conservative direction for a VISA timeout.
    """
    return channels * nplc / 50.0 + SAMPLE_OVERHEAD_S.get(int(channels), 660e-6)


def capture_step(smu, v_from, v_to, n_samples, nplc=0.01, settle_first_s=2.0,
                 chunk=2000, log=None, limit_a=None, capacitance=45.7e-6,
                 max_presettle_s=60.0, cfg=None, charge_limit=None,
                 switch_settle_s=0.5, trigger=None, channels=2,
                 pre_s=None, sync=None, on_fire=None, hold_tick=None, tick_s=5.0):
    """Step ch B from v_from to v_to, then burst-capture I(t) with timestamps.

    `trigger` replaces the voltage step with any other perturbation -- `exp_switch`
    passes a frame change, to time a pixel switch at a FIXED bias. Everything before
    it (the two-phase charge, the pre-settle) and after it (the chunked burst, the
    instrument timestamps) is the same measurement, so it is the same function.

    Returns (t, i_a, i_b), t in seconds from the first sample. Instrument
    timestamps are used, not host time, so the spacing is the SMU's own.

    ``pre_s=None`` (the default, and every capture before 2026-10-25) perturbs FIRST
    and starts the burst after it, so the record begins after the event and holds no
    baseline. With ``pre_s`` set, the burst starts first, ``pre_s`` of baseline is
    recorded, and only then does the perturbation run, while the SMU is integrating.
    ``sync`` (a dict) then receives the host-clock bracket of each event, relative to
    the estimated first sample:

        first_unc_s           half-width of the first-sample bracket: the burst was
                              sent at -first_unc_s and its START print arrived at
                              +first_unc_s, so the first sample lies in between
        trigger_lo_s/_hi_s    the perturbation was sent at lo and acknowledged at hi
                              (the FPGA ACKs every frame, so hi is the switch's ack)
        host_t0               perf_counter() of the first-sample estimate, to put
                              another host-clock record (the PM400) on this axis

    ``on_fire`` runs just before the burst is sent, to start a host-clock record that
    should include the baseline.

    ``hold_tick`` runs every ``tick_s`` during the pre-settle. exp_switch re-writes the
    lit pixel with it, because the FPGA's USB link sleeps after 10-30 s idle and the
    next write then waits ~1 s for it to wake (fpga_timing.py, 261025).

    The burst is issued in chunks. One ``measure_burst(20000)`` blocks the VISA
    read for the whole burst — about 8 s at NPLC 0.01 — which overruns the
    driver's 10 s timeout once overhead is added and then kills the run. The
    buffer is in append mode, so chunks accumulate into one record, and the
    instrument timestamps make the small gap between chunks visible rather than
    silently distorting the time axis. A chunk that still times out is caught: the
    interface is cleared so the stale 'DONE' cannot desync later queries, and
    whatever was captured so far is returned.
    """
    # Two-phase settle. The limit is the only current available to charge C, so a low
    # measurement limit makes the node take 9-40 s to arrive. Charging at a high limit
    # first takes ~0.09 s instead, then the limit drops for the measurement. The catch:
    # in High-C the limit SELECTS the range, so this switches the range mid-point, and
    # a range change is what corrupted the rising side in run 260630. Verify against a
    # single-limit run at the same bias before trusting it.
    if charge_limit and cfg is not None and hasattr(smu, "inst"):
        m.configure_voltage_source_b(
            smu, dataclasses.replace(cfg, nvled_compliance=charge_limit), log)
        smu.set_voltage("b", float(v_from))
        time.sleep(max(0.05, 3 * charge_time_s(capacitance, v_from, charge_limit)))
        smu.set_voltage("b", float(v_to))
        time.sleep(max(0.05, 3 * charge_time_s(capacitance, v_to - v_from, charge_limit)))
        m.configure_voltage_source_b(smu, cfg, log)       # back to the measure limit
        time.sleep(switch_settle_s)
        if log:
            log(f"      charged at {charge_limit:g} A, measuring at {limit_a:g} A "
                f"(switch settle {switch_settle_s:.2f}s)")
    m.set_nplc(smu, nplc)
    smu.setup_buffers(timestamps=True)
    smu.set_voltage("b", float(v_from))
    # The pre-settle has to clear the charge time at the configured limit, or the
    # capture starts before the node has reached v_from and measures the wrong state.
    # the charge phase already brought the node there, so the slow pre-settle is moot
    want = (settle_first_s if charge_limit
            else max(settle_first_s, 3 * charge_time_s(capacitance, v_from, limit_a)))
    pre = min(want, max_presettle_s)
    if want > max_presettle_s and log:
        log(f"      pre-settle CAPPED at {pre:.0f}s, charge needs {want:.0f}s at "
            f"{limit_a:g} A — the node may not have reached {v_from:+.2f} V. "
            f"Raise --max-presettle-s or the limit.")
    elif pre > settle_first_s and log:
        log(f"      pre-settle {pre:.1f}s (charge-limited at {limit_a:g} A)")
    t_end = time.perf_counter() + pre          # start from a known, settled state
    while True:
        left = t_end - time.perf_counter()
        if left <= 0:
            break
        time.sleep(min(left, tick_s) if hold_tick is not None else left)
        if hold_tick is not None and time.perf_counter() < t_end:
            hold_tick()
    smu.clear_buffers_only()
    if pre_s is not None:
        return _capture_with_baseline(smu, v_to, n_samples, nplc, chunk, log, trigger,
                                      channels, pre_s, sync, on_fire)
    if trigger is not None:
        trigger()                              # e.g. blank the frame and light a pixel
    else:
        smu.set_voltage("b", float(v_to))      # the step
    if not hasattr(smu, "inst"):               # sim rig: no TSP, no instrument clock
        t0 = time.perf_counter()
        smu.measure_burst_fire(int(n_samples))
        smu.measure_burst_join()
        ia, ib = smu.read_buffers(int(n_samples))
        return (np.linspace(0.0, time.perf_counter() - t0, len(ib)),
                np.asarray(ia, float), np.asarray(ib, float))

    # MEASURED, not modelled: the instrument timestamps of four bench runs at NPLC
    # 0.01 and 1.0 fit dt = 2.001 * nplc/50 + 660 us (1060 us and 40670 us). The 2x is
    # the burst reading BOTH channels sequentially -- smua.measure.i then smub.measure.i
    # -- and the 660 us is per-sample TSP and interface overhead. The old
    # max(2*nplc/50, 200e-6) underestimated 0.01 by 2.65x, so every record-length claim
    # built on it was short.
    per_sample_s = sample_interval_s(nplc, channels)
    old_timeout = smu.inst.timeout
    smu.inst.timeout = max(10_000, int(chunk * per_sample_s * 4_000))
    done = 0
    try:
        while done < int(n_samples):
            k = min(int(chunk), int(n_samples) - done)
            if channels == 1:
                smu.measure_burst_i_b(k)       # ch B only: half the integration
            else:
                smu.measure_burst(k)
            done += k
    except Exception as exc:
        try:
            smu.inst.clear()
        except Exception:
            pass
        if log:
            log(f"      burst stopped after {done}/{n_samples} samples: "
                f"{type(exc).__name__} — keeping what was captured")
    finally:
        # the buffer read is a single large ASCII transfer; give it room too
        smu.inst.timeout = max(old_timeout, 30_000)
        try:
            if channels == 1:
                ta, ib = smu.read_buffer_b()
                ia = [float("nan")] * len(ib)
            else:
                ia, ib, ta, _tb = smu.read_buffer_with_timestamps()
        finally:
            smu.inst.timeout = old_timeout
    if not ib:
        return np.array([]), np.array([]), np.array([])
    n = min(len(ia), len(ib), len(ta))
    return (np.asarray(ta[:n], float) - ta[0],
            np.asarray(ia[:n], float), np.asarray(ib[:n], float))


def _capture_with_baseline(smu, v_to, n_samples, nplc, chunk, log, trigger, channels,
                           pre_s, sync, on_fire):
    """capture_step's ``pre_s`` path: the trigger model first, then the perturbation.

    The SMU's trigger model takes the readings, not a TSP ``for`` loop: the firmware
    paces them, so the timestamps are steadier and the per-call TSP overhead is gone.
    ``initiate()`` returns at once, so the host is free while the buffer fills and
    sends the frame change into the running record. One record, no chunks -- the
    buffer fills once (FILL_ONCE) and ``waitcomplete()`` returns when it is full.

    Only for a perturbation OUTSIDE the SMU. Whether a source-level write is safe
    while the trigger model runs is not verified on the bench, so a voltage step is
    refused here rather than guessed at. ``chunk`` is unused on this path.
    """
    if trigger is None:
        raise ValueError("pre_s needs an external trigger: a source step during a "
                         "running trigger model is not verified on the bench")
    sync = {} if sync is None else sync
    n_samples = int(n_samples)
    # The TSP-loop interval. The trigger model is expected to be no slower, so this
    # bounds the record from above: it sizes the timeout and the late-START check.
    per_sample_s = sample_interval_s(nplc, channels)
    record_s = n_samples * per_sample_s
    real = hasattr(smu, "inst")
    if real:
        smu.arm_trigger_burst(n_samples, channels)
    if on_fire is not None:
        on_fire()
    t_sent = time.perf_counter()
    if real:
        t_ack = smu.start_trigger_burst_marked()
    else:                                       # sim rig: the burst fills at once
        smu.measure_burst_fire(n_samples)
        t_ack = time.perf_counter()
    time.sleep(max(0.0, pre_s))
    t_trig = time.perf_counter()
    trigger()
    t_trig_done = time.perf_counter()
    host_t0 = 0.5 * (t_sent + t_ack)
    sync.update(first_unc_s=0.5 * (t_ack - t_sent), host_t0=host_t0,
                trigger_lo_s=t_trig - host_t0, trigger_hi_s=t_trig_done - host_t0,
                start_ack_late=False)
    if real and t_ack - t_sent > 0.5 * record_s:
        # START came back with the end of the record, not its beginning: the host
        # could not tell when the readings began, so the events are not placed.
        sync["start_ack_late"] = True
        if log:
            log(f"      !! START took {t_ack - t_sent:.2f}s against a ~{record_s:.2f}s "
                f"record: the switch cannot be placed on the trace. Event times are "
                f"recorded but not trustworthy.")
    if not real:
        smu.measure_burst_join()
        ia, ib = smu.read_buffers(n_samples)
        n = min(len(ia), len(ib))
        t = np.linspace(0.0, max(time.perf_counter() - host_t0, pre_s), n)
        return t, np.asarray(ia[:n], float), np.asarray(ib[:n], float)
    old_timeout = smu.inst.timeout
    try:
        smu.inst.timeout = max(10_000, int((4 * record_s + 10) * 1000))
        smu.wait_trigger_burst()
    except Exception as exc:
        try:
            smu.inst.clear()
        except Exception:
            pass
        if log:
            log(f"      trigger record did not complete: {type(exc).__name__} — "
                f"keeping what was captured")
    finally:
        smu.abort_trigger()                     # measure action off, whatever happened
        smu.inst.timeout = max(old_timeout, 30_000)
        try:
            if channels == 1:
                ta, ib = smu.read_buffer_b()
                ia = [float("nan")] * len(ib)
            else:
                ia, ib, ta, _tb = smu.read_buffer_with_timestamps()
        finally:
            smu.inst.timeout = old_timeout
    if not ib:
        return np.array([]), np.array([]), np.array([])
    n = min(len(ia), len(ib), len(ta))
    if log and n >= 2:
        log(f"      trigger model: {n} readings, "
            f"{1e6 * (ta[n - 1] - ta[0]) / (n - 1):.0f} us apart "
            f"(TSP loop: {1e6 * per_sample_s:.0f} us)")
    return (np.asarray(ta[:n], float) - ta[0],
            np.asarray(ia[:n], float), np.asarray(ib[:n], float))


class PowerTrace:
    """``P(t)`` on a background thread, started at the perturbation instant.

    The SMU burst is one blocking VISA read — ``capture_step`` cannot poll anything
    while it runs — so a second thread on the power meter is the only way to get
    ``P(t)`` beside ``I(t)``. The two instruments are separate VISA resources, which is
    the case PyVISA is thread-safe for.

    Host time, not the instrument clock. The two clocks cannot be aligned better than
    the poll period in any case, and both records start at the same event: ``start()``
    is called from inside the trigger, so t = 0 is the frame change for both.

    Expect ~30-80 Hz against the SMU's 24.6 Hz at NPLC 1.0, so the optical record is
    the denser of the two. It is not synchronous with it: do not divide them
    sample-by-sample, bin both onto a common grid first.
    """

    def __init__(self, pm, n=1, averaging=None):
        self.pm, self.n, self.rows = pm, int(n), []
        self.averaging = averaging        # set on start(), restored on stop()
        self._prev_avg = None
        self._stop = threading.Event()
        self._th = None
        self.t0 = None

    def start(self):
        # Drop the meter's internal averaging for the trace. pm_averaging=3 is three
        # averages plus a USB round-trip per sample, which caps the record at 30-80 Hz
        # where the PM400 can do roughly 1 kHz. A settling trace wants resolution; the
        # settled reading afterwards is taken at the configured averaging, so the EQE
        # point stays comparable with every other run in the project.
        if self.averaging is not None:
            try:
                self._prev_avg = self.averaging
                self.pm.set_averaging(int(self.averaging))
            except Exception:
                self._prev_avg = None
        # kept, so a caller can move the record onto another host-clock axis
        self.t0 = t0 = time.perf_counter()

        def run():
            while not self._stop.is_set():
                try:
                    power, _ = read_power(self.pm, self.n)
                except Exception:
                    return          # a dead meter must not take the current trace too
                self.rows.append((time.perf_counter() - t0, power))

        self._th = threading.Thread(target=run, daemon=True)
        self._th.start()

    def stop(self, restore_averaging=None):
        """``(t, P)``. Empty arrays if it never started or the meter failed."""
        self._stop.set()
        if self._th is not None:
            self._th.join(timeout=2.0)
        if self._prev_avg is not None and restore_averaging is not None:
            try:
                self.pm.set_averaging(int(restore_averaging))
            except Exception:
                pass
        return (np.array([r[0] for r in self.rows], float),
                np.array([r[1] for r in self.rows], float))


# How far the 1 % settling band may be widened by noise before the row stops being a
# measurement. A band at 6 % still says something -- "settled to within 6 %" -- and
# firing there buries the real case in chatter. The real case, measured on Array42
# without High-C at the 1 uA range, was 38 %: 35.6 nA against 13.6 nA of band.
NOISE_LIMITED_FRAC = 0.10


def flush_traces(out_dir: Path, log=None) -> Path | None:
    """Write every trace collected so far to traces.npz. Safe to call repeatedly.

    Called after each pixel as well as at the end, for the same reason write_csv is:
    a run is ~75 minutes and used to hold all its raw data in memory until the last
    line. The CSVs were already flushed per pixel, so a half-finished run gave you
    `settled` and `t_1pct` but none of the traces the exponent is fitted from.

    savez_compressed rewrites the whole archive, so this is O(n^2) over the run -- at
    eight pixels and a few MB each that is seconds in total, against 75 minutes of
    measurement. Writing to a temporary name first means an interrupted flush cannot
    leave a truncated archive where a complete one used to be.
    """
    if not (_TRACES or _PTRACES or _VTRACES):
        return None
    npz = out_dir / "traces.npz"
    # The suffix MUST stay .npz: savez_compressed appends ".npz" to any name that
    # lacks it, so "traces.npz.part" became "traces.npz.part.npz" on disk and the
    # rename below then failed on a file that was never written.
    tmp = out_dir / "traces.part.npz"
    try:
        # Current traces are (t, I_A, I_B) on the instrument clock. Power traces are
        # (t, P) on the host clock at their own rate, so they cannot share a key:
        # the "pm_" prefix keeps one file per run without pretending they align.
        np.savez_compressed(tmp, **{
            f"{i:03d}_{tag}_{k}": np.vstack([t, ia, ib])
            for i, (tag, t, ia, ib, k) in enumerate(_TRACES)
        }, **{
            f"pm_{i:03d}_{tag}": np.vstack([t, p])
            for i, (tag, t, p) in enumerate(_PTRACES)
        }, **{
            f"v_{i:03d}_{tag}": np.vstack([t, v])
            for i, (tag, t, v) in enumerate(_VTRACES)})
        tmp.replace(npz)
    except Exception as exc:                     # a flush must never kill a measurement
        if log is not None:
            log(f"    could not flush traces: {type(exc).__name__}: {exc}")
        return None
    return npz


def capture_verdict(met, stage, highc, rng):
    """Say in words when a row's numbers are not measurements.

    A long capture is ~7 minutes, and it used to end in a line of plausible-looking
    figures whatever happened. Two ways a row can be worthless while still printing:

      * OVER-RANGE. `settled` is the median of the surviving samples, so at
        5695/6000 over-range it is the median of the 305 that happened to fit --
        a biased minority, not the current.
      * NOISE-LIMITED. settling_metrics widens the 1 % band to 5 sigma when the
        noise is larger than 1 % (`t_1pct_tol` records what it used), because a
        trace with 3 % scatter can never enter a 1 % band and a pure-fraction
        criterion would return NaN forever. When that happens the trace never
        leaves its own band, and t_1pct falls back to half a smoothing window --
        a sentinel. Printed bare it reads like a fast settling time, when it
        means the opposite: the current is too small to resolve on this range.

    Measured example, Array42 at -0.80 V on the 1 uA range without High-C: 35.6 nA
    against 27.1 nA of scatter, so the band is 13.6 nA = 38 % of the reading, and
    "t1%=1.017" is exactly half the 100-sample window. Both of these are the ANSWER
    for a non-High-C capture -- they are only misleading when left unsaid.
    """
    out = []
    n, over = met.get("n", 0), met.get("n_overflow", 0)
    settled, tol = abs(met.get("settled", float("nan"))), met.get("t_1pct_tol")
    state = f"{'High-C' if highc else 'no High-C'}, range {rng}"
    if n and over > 0.5 * n:
        out.append(f"{stage}: OVER-RANGE, {over}/{n} samples ({100 * over / n:.0f} %). "
                   f"`settled` is the median of the {n - over} that fit, which is a "
                   f"biased minority and not the current. ({state})")
    if np.isfinite(settled) and settled > 0 and tol and np.isfinite(tol):
        if tol > NOISE_LIMITED_FRAC * settled:
            out.append(
                f"{stage}: NOISE-LIMITED. The settling band had to be widened to "
                f"{tol * 1e9:.2f} nA ({100 * tol / settled:.0f} % of the "
                f"{settled * 1e9:.1f} nA reading) because the scatter is larger than "
                f"1 %. t1% is then half a smoothing window, not a settling time: this "
                f"current is not resolvable here. ({state})")
    return out


def keep_trace(tag, t, ia, ib, n_max):
    """Stash one trace for the run's traces.npz, and return its spectrum summary.

    ``n_max`` follows decimate_trace: 0 or negative means keep every sample. None means
    saving is off (--no-traces). The two used to disagree -- decimate_trace read 0 as
    "no thinning" while this read it as "do not save" -- so 0 did the opposite of what
    the helper documented.
    """
    spec = spectrum_summary(t, np.abs(ib))
    if n_max is not None and t.size:
        td, ibd, k = decimate_trace(t, ib, n_max)
        _, iad, _ = decimate_trace(t, ia, n_max)
        _TRACES.append((tag, td.astype(np.float32), iad.astype(np.float32),
                        ibd.astype(np.float32), k))
    return spec


# Raw traces, collected by capture_step and written per experiment by main(). A
# module-level sink keeps capture_step's signature and the six experiment signatures
# unchanged; this is a script, not a library.
# ponytail: global collector, pass a sink explicitly if this ever grows a second caller.
_TRACES: list = []
# P(t) records, kept apart from _TRACES because they are on the host clock at their own
# rate. They land in the same traces.npz under a "pm_" prefix so one file still holds
# everything a run measured.
_PTRACES: list = []
# V(t) ramps from exp_cap. Separate because they are a voltage against a current, on
# their own clock, and the "v_" prefix in traces.npz says so.
_VTRACES: list = []

# Extra CSVs an experiment wants beside its own, keyed by file stem. The settling scan
# needs one row per (K, probe time), which does not fit cluster.csv's one row per K.
_EXTRA_ROWS: dict = {}


def keep_trace_v(tag, t, v):
    """Stash one ``V(t)`` ramp. exp_cap's raw data, which used to be discarded.

    Lands in traces.npz under a "v_" prefix, so one file still holds everything a run
    measured and a reader can tell a voltage ramp from a current transient by the key.
    """
    if t.size:
        _VTRACES.append((tag, t.astype(np.float32), v.astype(np.float32)))


def keep_power_trace(tag, t, p):
    """Stash one ``P(t)`` beside the current trace of the same tag."""
    if t.size:
        _PTRACES.append((tag, t.astype(np.float32), p.astype(np.float32)))


def spectrum_summary(t, i, mains_hz=50.0, tail_frac=0.5):
    """Dominant periodic component of a trace, and the amplitude at the mains line.

    A sine riding on the transient is the hypothesis this answers. At NPLC 0.01 the
    SMU integrates 1/100 of a mains cycle, so it has NO mains rejection -- only an
    integer NPLC does -- and the ~940 Hz sample rate resolves 50 Hz easily. Returns
    the peak frequency and amplitude outside the DC/drift region, plus the amplitude
    in a narrow band at the mains line and its first harmonic.

    Only the last `tail_frac` of the trace is used. The settling is a power law, not
    an exponential, so a quadratic cannot detrend it: over the whole trace the
    residual transient dominates every low bin and inflates `rms_ac` by ~30x over the
    real noise. The tail is also the only part that matters, because that is where
    the sweep averages.
    """
    out = dict(fs_hz=float("nan"), nyquist_hz=float("nan"), peak_hz=float("nan"),
               peak_amp=float("nan"), mains_amp=float("nan"), mains2_amp=float("nan"),
               rms_ac=float("nan"))
    ok = np.isfinite(i) & (np.abs(i) < OVERFLOW)
    if ok.sum() < 64 or t.size < 2:
        return out
    tv, iv = t[ok], i[ok]
    cut = int((1.0 - tail_frac) * tv.size)
    if tv.size - cut >= 64:
        tv, iv = tv[cut:], iv[cut:]
    dt = float(np.median(np.diff(tv)))
    if not dt > 0:
        return out
    fs = 1.0 / dt
    # detrend: the transient itself is a huge low-frequency component
    iv = iv - np.polyval(np.polyfit(tv, iv, 2), tv)
    w = np.hanning(iv.size)
    amp = np.abs(np.fft.rfft(iv * w)) * (4.0 / iv.size)      # ~peak amplitude
    f = np.fft.rfftfreq(iv.size, dt)
    lo = f > 2.0                                              # skip residual drift
    out.update(fs_hz=fs, nyquist_hz=fs / 2, rms_ac=float(np.std(iv)))
    if lo.any():
        k = int(np.argmax(amp[lo]))
        out.update(peak_hz=float(f[lo][k]), peak_amp=float(amp[lo][k]))
    for key, target in (("mains_amp", mains_hz), ("mains2_amp", 2 * mains_hz)):
        if target < fs / 2:
            band = np.abs(f - target) < max(2.0, 3 * fs / iv.size)
            if band.any():
                out[key] = float(np.max(amp[band]))
    return out


def decimate_trace(t, i, n_max):
    """Uniformly thin a trace to at most n_max points, preserving the sample interval.

    Uniform, not log-spaced: log spacing would destroy the frequency content that the
    sine-wave question depends on. Returns the stride used so a reader knows the
    effective sample rate.
    """
    if n_max <= 0 or t.size <= n_max:
        return t, i, 1
    k = int(np.ceil(t.size / n_max))
    return t[::k], i[::k], k


def settling_metrics(t, i):
    """Settled value plus how long it took to get there.

    Settled = median of the last 20% of the VALID samples. "Time to x%" is the LAST
    time the trace is outside x% of it, because a trace that re-enters the band and
    leaves again has not settled.

    Three columns to read carefully:
      * ``t_*pct`` is NaN when the trace is STILL outside the band at the end of the
        record. It did not settle in the time captured, so raise --samples. 0.0 means
        it was inside the band from the first valid sample.
      * ``settled_ok`` is False exactly when t_1pct is NaN.
      * ``n_overflow`` counts over-range samples. Anything above zero means the
        measure range was too small for the transient, even if the settled value fits.
    """
    valid = np.isfinite(i) & (np.abs(i) < OVERFLOW)
    base = dict(n=int(t.size), n_overflow=int((~valid).sum()), settled=float("nan"),
                t_1pct=float("nan"), t_01pct=float("nan"), overshoot=float("nan"),
                ripple=float("nan"), settled_ok=False, tail_noise=float("nan"),
                t_1pct_tol=float("nan"), t_01pct_tol=float("nan"),
                tail_drift_frac=float("nan"), smooth_window=0)
    if valid.sum() < 10:
        return base
    tv, iv = t[valid], i[valid]
    tail = iv[int(0.8 * iv.size):]
    settled = float(np.median(tail))
    base.update(settled=settled,
                ripple=float(np.std(tail) / abs(settled)) if settled else float("nan"),
                overshoot=float(np.max(np.abs(iv)) / abs(settled)) if settled else float("nan"))
    # The band has to respect the measurement noise. A trace with 3% ripple can never
    # enter a 1% band, so a pure-fraction criterion reports NaN forever and tells you
    # nothing about settling. "Settled" therefore means indistinguishable from the
    # final value GIVEN the noise: within frac, or within 3 sigma of the tail.
    noise = float(np.std(tail))
    base["tail_noise"] = noise
    # Smooth before the band test, and use 5 sigma rather than 3. With 3 sigma about
    # 0.3% of samples sit outside the band by chance, so on a noisy trace the LAST such
    # sample is near the end almost always and t_1pct just reports the record length.
    # Smoothing removes single-sample excursions; 5 sigma makes a chance excursion
    # negligible even over thousands of samples. When the noise is larger than the
    # nominal band the tolerance is noise-dominated, which is honest -- 2% per-sample
    # noise cannot resolve 1% settling. t_*pct_tol records the tolerance actually used.
    # Cap the smoothing window in TIME, not as a fraction of the record, and reject an
    # answer past half the record: both from measure_iqe.calibrate, which is where this
    # criterion is defined. A fraction-of-record window makes the resolution degrade as
    # the record grows (a 600 s capture smoothed over 30 s), and without the record
    # limit the answer scales with how long you happened to watch -- 63%, 67%, 62% of
    # the record on three captures of the same device.
    dtm = float(np.median(np.diff(tv))) or 1e-3
    w = int(np.clip(iv.size // 100, 5, max(5, round(SMOOTH_MAX_S / dtm))))
    smooth = np.convolve(iv, np.ones(w) / w, mode="valid")
    ts = tv[w - 1:]
    base["smooth_window"] = int(w)
    for frac, key in ((0.01, "t_1pct"), (0.001, "t_01pct")):
        tol = max(frac * abs(settled), 5.0 * noise / np.sqrt(w))
        base[key + "_tol"] = tol
        bad = np.nonzero(np.abs(smooth - settled) > tol)[0]
        if bad.size == 0:
            base[key] = float(ts[0]) / 2.0      # no better than half a smoothing window
        elif bad[-1] + 1 >= smooth.size:
            base[key] = float("nan")            # never settled inside the record
        elif float(ts[bad[-1]]) > RECORD_LIMITED_FRAC * float(tv[-1]):
            base[key] = float("nan")            # measures the window, not the device
        else:
            base[key] = float(ts[bad[-1]])
    # A short record can LOOK settled, because "settled" is measured against the end
    # of that same record: a slow exponential still climbing looks flat over its last
    # 20%. So test the tail for a trend as well. A tail that is still drifting means
    # the capture was too short, whatever t_1pct says.
    tail_t = tv[int(0.8 * tv.size):]
    drift = float("nan")
    if tail_t.size > 2 and settled:
        slope = float(np.polyfit(tail_t, tail, 1)[0])            # A/s
        # extrapolate over the window the slope was fitted on, not the whole record:
        # multiplying by the full record over-states the drift of an exponential tail
        drift = abs(slope * (tail_t[-1] - tail_t[0])) / abs(settled)
    base["tail_drift_frac"] = drift
    base["settled_ok"] = bool(np.isfinite(base["t_1pct"])
                              and (not np.isfinite(drift) or drift < 0.01))
    return base


# ----------------------------------------------------------------------
#  per-pixel experiments
# ----------------------------------------------------------------------

def switch_deviation(t, i):
    """``(worst_early_1s, tail, resolution)`` in amps, for the frame-switch question.

    Both choices here were forced by run 261016-0726, and the old version got both
    wrong (section 21):

      * The reference is the record's TAIL, not the mean of every window after 2 s.
        That trace still sits -0.11 % low at 60-90 s, so a "later" spanning the whole
        record contains most of the transient and divides it out.
      * The early number is the WORST one-second window of the first five, never their
        average. At -0.80 V the trace spikes +0.49 % for half a second and then sits
        at -0.56 % for seconds. Averaging across that sign change gives -0.02 % for
        second one, and the old metric called a 16-sigma deficit FREE.

    Returns ``(nan, nan, nan)`` when the record is too short to hold five early
    seconds, a gap, and a tail that do not overlap.
    """
    nan = float("nan")
    y = np.abs(np.asarray(i, float))
    y = y[np.isfinite(y) & (y < OVERFLOW)]
    if t.size < 9 or y.size < 9:
        return nan, nan, nan
    dt = float(np.median(np.diff(t))) or 1e-3
    w = max(2, int(round(1.0 / dt)))
    if y.size <= 12 * w:
        return nan, nan, nan
    n_tail = min(max(2 * w, y.size // 10), y.size - 6 * w)
    tw = [y[k:k + w].mean() for k in range(y.size - n_tail, y.size - w + 1, w)]
    if len(tw) < 3:
        return nan, nan, nan
    later = float(np.mean(tw))
    res = float(np.std(tw, ddof=1))              # 1 s resolution, measured
    early = [y[k:k + w].mean() for k in range(0, 5 * w, w)]
    return float(max(early, key=lambda a: abs(a - later))), later, res



def drain_errors(smu):
    """The SMU error queue as one string, empty when clean.

    Called after every range change and every capture, and recorded in the CSV.
    Checking only once at configure time misses a range the SMU refused later and a
    measurement that over-ranged during the transient.
    """
    if not hasattr(smu, "check_error_queue"):
        return ""
    try:
        return " | ".join(smu.check_error_queue())
    except Exception as exc:
        return f"error-queue read failed: {exc}"


def step_stages(v: float, highc: bool, rng, args):
    """The captures to take at one (voltage, range, High-C) point.

    Always the fast one. The long one only where slow settling lives, because every
    combination at 400 s would be 72 hours: the faintest bias, High-C on (the state the
    sweep actually uses), and -- when --highc-both sweeps ranges with High-C off -- not
    repeated per range, since the long stage is about the device and not the range.
    """
    out = [("fast", args.nplc, args.samples, None)]
    if not args.step_long_nplc or args.step_long_samples <= 0:
        return out
    # The FAINTEST bias is the LEAST negative, i.e. max(): the sweep runs -0.8 V (faint,
    # 240 s to settle) to -2.6 V (bright, 1.8 s). Picking min() would put the long
    # capture at the bright end, where there is nothing slow to see.
    want = (args.step_long_voltages if args.step_long_voltages is not None
            else [max(args.step_voltages)])
    if v not in want:
        return out
    # Both High-C states get a long record: whether the node settles WITHOUT High-C
    # is the question a de-capped carrier exists to answer, and only a long record
    # answers it. But non-High-C sweeps ranges, and five long captures is 33 min a
    # pixel, so it runs at --step-long-ranges only.
    if not highc and rng is not None and rng not in args.step_long_ranges:
        return out
    # The limit is the measure range ONLY in High-C, and there it has to drop so the
    # tail resolves (~1 nA/sample instead of ~100). With High-C off the range comes
    # from --ranges, so the run limit stays and the charge edge stays 9 ms.
    out.append(("long", args.step_long_nplc, args.step_long_samples,
                (args.step_long_compliance or None) if highc else None))
    return out


def exp_step(pm, smu, smile, pixel, cfg, args, log):
    """I(t) after a voltage step, per voltage x range x High-C state."""
    rows = []
    smile.set_pixel(pixel[0], pixel[1], cfg.bitvals[0])
    for highc in ([False, True] if args.highc_both else [args.highc]):
        m.configure_voltage_source_b(smu, dataclasses.replace(cfg, highc_b=highc), log)
        # In High-C the limit governs the range, so sweeping ranges is meaningless.
        for rng in ([None] if highc else args.ranges):
            if rng is not None:
                m.set_measure_range_b(smu, rng)
                time.sleep(0.05)
            err_range = drain_errors(smu)
            if err_range:
                log(f"    SMU rejected range {rng}: {err_range}")
            for v in args.step_voltages:
              # Two captures per point, because one cannot be both fast and long. The
              # long stage is narrow on purpose -- see step_long_points().
              for stage, nplc_s, nsamp, comp_s in step_stages(v, highc, rng, args):
                # The stage's own limit. In High-C this is also a range change, so it
                # goes through configure_voltage_source_b -- which leaves High-C, writes
                # the range and re-enables it, the only order the 2602B accepts.
                scfg = cfg if comp_s is None else dataclasses.replace(
                    cfg, nvled_compliance=comp_s)
                if comp_s is not None:
                    m.configure_voltage_source_b(
                        smu, dataclasses.replace(scfg, highc_b=highc), log)
                    if rng is not None:
                        # configure_channel writes cfg.nvled_range_i, which silently
                        # replaces the range --ranges asked for. In High-C the limit
                        # IS the range so there is nothing to restore, but with High-C
                        # off the capture would land on the wrong range and nothing in
                        # the row would show it: range_i records what we asked for.
                        m.set_measure_range_b(smu, rng)
                    log(f"      {stage}: ch B limit {comp_s:g} A"
                        + (" (also the High-C measure range)" if highc
                           else f" at range {rng:g}" if rng is not None else ""))
                elif stage != "fast":
                    # Announce it even when nothing was reconfigured. A long capture is
                    # ~7 min of silence otherwise, with no way to tell it from a hang.
                    log(f"      {stage}: ch B limit {scfg.nvled_compliance:g} A"
                        + (f" at range {rng:g}" if rng is not None else ""))
                # P(t) beside I(t). capture_step's `trigger` hook exists for exactly
                # this -- "replaces the voltage step with any other perturbation" -- so
                # the step and the optical record start on the same statement and share
                # t = 0. exp_switch has used it since the switch-deviation work.
                ptrace = (PowerTrace(pm, args.step_pm_trace,
                                     averaging=args.step_pm_averaging)
                          if args.step_pm_trace else None)

                def _step(_v=v):
                    smu.set_voltage("b", float(_v))
                    if ptrace is not None:
                        ptrace.start()

                t, ia, ib = capture_step(smu, args.v_start, v, nsamp, nplc_s,
                                     chunk=args.chunk, log=log,
                                     limit_a=scfg.nvled_compliance,
                                     capacitance=args.capacitance,
                                     max_presettle_s=args.max_presettle_s,
                                      cfg=scfg, charge_limit=args.charge_limit,
                                      switch_settle_s=args.switch_settle_s,
                                      trigger=_step, channels=args.step_channels)
                p_t = p_p = np.array([])
                if ptrace is not None:
                    # restore the configured averaging before the settled reading
                    p_t, p_p = ptrace.stop(restore_averaging=cfg.pm_averaging)
                met = settling_metrics(t, np.abs(ib))
                # Say it when the record cannot have seen the asymptote. settled is the
                # median of the last 20 % of the window, which is a number whatever the
                # window was, so nothing else in the row reveals this.
                span = float(t[-1] - t[0]) if t.size > 1 else 0.0
                truncated = bool(np.isfinite(met.get("t_1pct", float("nan")))
                                 and met["t_1pct"] >= 0.9 * span)
                if truncated or (span and not np.isfinite(met.get("t_1pct", float("nan")))):
                    log(f"      {stage}: the {span:.1f} s record did not reach 1 % "
                        f"settling, so `settled` is the end of the window and not the "
                        f"asymptote. Raise --step-long-samples or --step-long-nplc.")
                tag = (f"step_{stage}_hc{int(highc)}_r{rng}"
                       f"_f{args.v_start:+.3f}_v{v:+.3f}")
                spec = keep_trace(tag, t, ia, ib, args.trace_points)
                # Optical settling, measured the same way the electrical is: the LAST
                # time the trace leaves the band around its own settled value.
                p_met = {}
                if p_t.size > 10:
                    keep_power_trace(tag, p_t, p_p)
                    pm_met = settling_metrics(p_t, np.abs(p_p))
                    p_met = dict(p_settled_W=pm_met["settled"],
                                 p_t_1pct=pm_met["t_1pct"],
                                 p_settled_ok=pm_met["settled_ok"],
                                 p_n=int(p_t.size),
                                 p_rate_hz=float(p_t.size / max(p_t[-1], 1e-9)))
                    if log:
                        el = met.get("t_1pct", float("nan"))
                        log(f"      {stage}: P(t) {p_t.size} samples at "
                            f"{p_met['p_rate_hz']:.0f} Hz, optical t_1pct "
                            f"{pm_met['t_1pct']:.2f}s vs electrical {el:.2f}s")
                ia_ok = ia[np.isfinite(ia) & (np.abs(ia) < OVERFLOW)] if ia.size else ia
                # The node is settled by now, so read the light too: at these currents
                # it is well above the measured optical floor, which turns a settling
                # characterisation into an EQE point for free.
                pw_lit, pw_sem = read_power(pm, args.step_n_pm)
                # At these currents the emission reaches single-digit pW, where the
                # PM400's own offset (+/-7 pW, sign-random) is the same size as the
                # signal. So measure the dark at THIS bias, not once per run: blank the
                # pixel, read, relight. The next capture has its own pre-settle.
                pw_dark = args.dark_w
                if args.step_dark:
                    smile.blank_frame()
                    time.sleep(0.05)
                    pw_dark, _ = read_power(pm, args.step_n_pm)
                    smile.set_pixel(pixel[0], pixel[1], cfg.bitvals[0])
                pw = pw_lit - pw_dark
                eqe = float("nan")
                if np.isfinite(met["settled"]) and met["settled"]:
                    eqe = (pw / cfg.photon_energy_j) / (
                        abs(met["settled"]) / 1.602176634e-19)
                err = (err_range + " " + drain_errors(smu)).strip()
                # A limit below the current the LED wants at this bias means the SMU
                # is in compliance: it is current-sourcing and the node never reached
                # the requested voltage, so v_to is not the operating point.
                at_lim = bool(np.isfinite(met["settled"])
                              and abs(met["settled"]) >= 0.95 * scfg.nvled_compliance)
                if at_lim:
                    log(f"      at compliance ({scfg.nvled_compliance:g} A): V={v:+.2f} is "
                        f"not reachable at this limit, the node is current-limited")
                if comp_s is not None:      # put the run's limit back for the next point
                    m.configure_voltage_source_b(
                        smu, dataclasses.replace(cfg, highc_b=highc), log)
                    if rng is not None:
                        m.set_measure_range_b(smu, rng)
                rows.append(dict(stage=stage, highc=highc, **p_met,
                                 range_i=rng if rng is not None else "limit",
                                 compliance_a=scfg.nvled_compliance, nplc=nplc_s,
                                 n_requested=nsamp, record_span_s=span,
                                 record_truncated=truncated,
                                 at_compliance=at_lim,
                                 v_from=args.v_start, v_to=v, **met,
                                 j_settled=to_j(met["settled"], cfg),
                                 power_W=pw, power_lit_W=pw_lit, power_dark_W=pw_dark,
                                 power_sem=pw_sem, eqe_ext=eqe,
                                 power_sigma=abs(pw) / pw_sem if pw_sem else float("nan"),
                                 # NaN on --step-channels 1: ch A is not read at
                                 # all there, so this is absent rather than zero.
                                 i_a_settled=float(np.median(ia_ok[int(0.8 * ia_ok.size):]))
                                 if ia_ok.size else float("nan"),
                                 channels=args.step_channels,
                                 record_s=float(t[-1]) if t.size else 0.0,
                                 charge_limit=args.charge_limit or "",
                                 **{f"spec_{k}": v2 for k, v2 in spec.items()},
                                 smu_errors=err))
                log(f"    highc={highc!s:5} range={str(rng):>7} V={v:+.2f}  "
                    f"I={met['settled']*1e9:+10.1f} nA (J={to_j(met['settled'], cfg):+8.2f})  "
                    f"t1%={met['t_1pct']:.3f} t0.1%={met['t_01pct']:.3f}  "
                    f"P={pw*1e12:+9.1f}pW({abs(pw)/pw_sem if pw_sem else 0:4.0f}sig) "
                    f"EQE={100*eqe:6.3f}%  "
                    f"ripple={100*met['ripple']:.0f}%  over={met['n_overflow']}/{met['n']}  "
                    f"rec={float(t[-1]) if t.size else 0:.2f}s  "
                    f"sine {spec['peak_hz']:.1f}Hz@{spec['peak_amp']*1e9:.1f}nA "
                    # NaN here is not a failure: it means 50 Hz is above this capture's
                    # Nyquist frequency and therefore cannot be measured. At NPLC 1.0 the
                    # sample rate is ~25 Hz, so Nyquist is ~12 Hz. Say that instead of
                    # printing "nannA", which reads like something broke.
                    + (f"(50Hz {spec['mains_amp']*1e9:.1f}nA)"
                       if np.isfinite(spec['mains_amp'])
                       else f"(50Hz n/a: above Nyquist {spec['fs_hz']/2:.0f}Hz)")
                    + ("  << NOT SETTLED" if not met["settled_ok"] else "")
                    + (f"  ERR: {err}" if err else ""))
                for line in capture_verdict(met, stage, highc, rng):
                    log(f"      {line}")
    smile.blank_frame()
    return rows


def exp_cap(pm, smu, smile, pixel, cfg, args, log):
    """C from a known current and the voltage slope it produces: ``C = I / (dV/dt)``.

    The original method integrated the current transient of a voltage step to get
    Q, then C = Q/dV. It failed on the bench for two reasons that are not fixable
    by tuning: it needs an accurate current *through* a fast transient, which is
    exactly what the capacitor prevents, and a single over-range sample at 9.91e37
    poisons the whole integral (observed: C = 1e40 F).

    Sourcing a known current and timing the voltage ramp needs only a voltage
    slope, which the SMU measures well, and it sidesteps the current-range problem
    entirely because the current is *set* rather than measured.

    Run below turn-on, where the LED draws a small fraction of the setpoint, so the
    slope is the capacitor's. Blanked and lit, so a per-pixel contribution shows.
    """
    rows = []
    park = args.cap_park
    for lit in (False, True):
        if lit:
            smile.set_pixel(pixel[0], pixel[1], cfg.bitvals[0])
        else:
            smile.blank_frame()
        for i_set in args.cap_currents:
            # park in voltage mode at a bias below turn-on, then hand the node to a
            # current source and watch it ramp
            m.configure_voltage_source_b(smu, cfg, log)
            smu.set_voltage("b", float(park))
            time.sleep(max(0.5, 3 * charge_time_s(args.capacitance, park,
                                                  cfg.nvled_compliance)))
            # Read the node BEFORE handing it to the current source. Without this the
            # only voltage on record is the ramp's first sample, which is after the
            # mode switch -- so a park that worked and a park destroyed by the switch
            # look identical. On Array38 every ramp began 0.41-0.74 V from the park
            # while Array42 began within 14 mV, and nothing could say which half of
            # the sequence was at fault.
            v_parked = float("nan")
            try:
                v_parked = float(m.read_voltage_b(smu, 3)[0])
            except Exception:                 # a diagnostic must never fail the run
                pass
            src = dataclasses.replace(cfg, current_source_range_i=cap_source_range(i_set))
            # Hand the node straight to the real setpoint. Going via 0 A lets it float
            # to its own equilibrium first, and on a small capacitance it gets all the
            # way there before the ramp begins -- which is what destroyed every
            # Array38 capture.
            # Same sign as the set_current_b below: a SINK, driving the node negative.
            m.configure_current_source_b(smu, src, level_i=-abs(i_set))
            m.set_nplc(smu, max(args.nplc, 0.1))
            # Buffered V(t) on ch B, with the INSTRUMENT's timestamps. Polling this
            # from Python gave host-clock times and, worse, threw the raw ramp away --
            # only the fitted slope reached the CSV, so C could never be examined per
            # voltage segment. A junction's C is voltage-dependent, and a single slope
            # across -0.4 -> -0.65 V cannot show that.
            #
            # One channel, so no 2x: sample_interval_s(nplc, channels=1). The whole
            # cap_seconds is captured and the stop voltage is applied in ANALYSIS, which
            # is what makes segmenting possible -- the live break could only ever give
            # one span.
            nplc_cap = max(args.nplc, 0.1)
            n_cap = max(16, int(args.cap_seconds / sample_interval_s(nplc_cap, 1)))
            m.set_nplc(smu, nplc_cap)
            smu.setup_buffers(timestamps=True)
            smu.clear_buffers_only()
            m.set_current_b(smu, -abs(i_set))        # sink: drives the node negative
            if hasattr(smu, "inst"):
                smu.measure_burst_v_b(n_cap)
                ts, vs = smu.read_buffer_b(n_cap)
            else:                                     # sim rig: no TSP
                t0 = time.perf_counter()
                ts, vs = [], []
                while (dt := time.perf_counter() - t0) < min(args.cap_seconds, 0.2):
                    vb, _ = m.read_voltage_b(smu, n=2)
                    ts.append(dt); vs.append(vb)
            m.set_current_b(smu, 0.0)
            tsa, vsa = np.asarray(ts, float), np.asarray(vs, float)
            if tsa.size:
                tsa = tsa - tsa[0]
            # The stop is now a MASK, not a break: keep everything, fit what is below
            # turn-on. Samples past cap_v_stop are still saved -- they are the only
            # record of what the node does once the diode starts conducting.
            n_all = int(tsa.size)
            use = np.abs(vsa) <= abs(args.cap_v_stop) if tsa.size else np.zeros(0, bool)
            if use.sum() >= 6:
                tsa, vsa = tsa[use], vsa[use]
            keep_trace_v(f"cap_lit{int(lit)}_i{i_set:.0e}", np.asarray(ts, float),
                         np.asarray(vs, float))
            slope, cap = ramp_capacitance(tsa, vsa, i_set, cfg.nvled_v_limit)
            # Is the setpoint large enough to dominate the diode's own leakage? At
            # -0.4 V on Array42 the 1 uA point gave C = 213.56 uF blanked against
            # 39.41 uF lit -- a factor of 5 -- while 10 and 100 uA agreed at 45-52 uF.
            # The leakage steals part of the ramp current, the slope under-reads, and C
            # comes out too high. There is nothing to tune: the setpoint has to be well
            # above the leakage. Flagged, not silently averaged with the good points.
            i_leak_frac = (abs(cap * slope) / abs(i_set)) if (slope and i_set) else float("nan")
            suspect = bool(np.isfinite(cap) and cap > 2.0 * args.capacitance)
            if suspect:
                log(f"      !! I={i_set*1e9:.0f} nA gives C={cap*1e6:.1f} uF, over 2x "
                    f"the {args.capacitance*1e6:.1f} uF expected. At this bias the "
                    f"diode's leakage is a large fraction of the setpoint, so it steals "
                    f"ramp current and the slope under-reads. Trust the larger "
                    f"setpoints; this one is leakage-dominated.")
            # Did the node actually get to the park bias? C = I/(dV/dt) assumes the
            # ramp starts from a known sub-turn-on voltage. Array38 parked at -0.400 V
            # and every ramp began at +0.334 V, which nothing noticed.
            v0 = float(vsa[0]) if vsa.size else float("nan")
            park_off = abs(v0 - park) if np.isfinite(v0) else float("nan")
            park_bad = bool(np.isfinite(park_off) and park_off > args.cap_park_tol)
            if park_bad:
                suspect = True
                blame = "unknown (no park reading)"
                if np.isfinite(v_parked):
                    blame = (f"the park itself settled to {v_parked:+.3f} V"
                             if abs(v_parked - park) > args.cap_park_tol
                             else f"the park reached {v_parked:+.3f} V and the "
                                  f"voltage-to-current MODE SWITCH moved it "
                                  f"{abs(v0 - v_parked)*1e3:.0f} mV")
                log(f"    !! park MISSED by {park_off*1e3:.0f} mV: asked for "
                    f"{park:+.3f} V, the ramp began at {v0:+.3f} V -- {blame}. "
                    f"C = I/(dV/dt) assumes a known start, so this capture is "
                    f"rejected. Note the settle wait is sized from --capacitance "
                    f"({args.capacitance*1e6:.1f} uF): on a carrier with far less "
                    f"than that, the same mode-switch charge moves the node much "
                    f"further.")
            rows.append(dict(lit=lit, i_set=i_set, park_v=park, nplc=nplc_cap,
                             n_samples=int(tsa.size), n_captured=n_all,
                             n_requested=n_cap,
                             ramp_s=float(tsa[-1]) if tsa.size else 0.0,
                             v_start=v0,
                             v_end=float(vsa[-1]) if vsa.size else float("nan"),
                             v_parked=v_parked, park_off=park_off,
                             park_kick=(abs(v0 - v_parked)
                                        if np.isfinite(v_parked) and np.isfinite(v0)
                                        else float("nan")),
                             park_ok=int(not park_bad),
                             dvdt=slope, capacitance_F=cap,
                             leakage_suspect=suspect,
                             smu_errors=drain_errors(smu)))
            log(f"    lit={lit!s:5} I={i_set*1e9:>8.0f} nA  "
                f"V {vsa[0] if vsa.size else float('nan'):+.3f} -> "
                f"{vsa[-1] if vsa.size else float('nan'):+.3f} V in "
                f"{tsa[-1] if tsa.size else 0:.1f}s  dV/dt={slope*1e3:+8.2f} mV/s  "
                f"-> C={cap*1e6:8.2f} uF")
    m.set_current_b(smu, 0.0)
    smile.blank_frame()
    flag_cap_leakage(rows, log)
    return rows


def flag_cap_leakage(rows, log, tol=0.30):
    """Mark setpoints where lit and blanked disagree. Mutates ``rows``.

    The capacitor cannot care whether a sub-turn-on pixel is addressed, so a large
    lit/blanked difference at one setpoint means something other than the capacitor is
    taking the ramp current -- the diode's own leakage, which is a large fraction of a
    small setpoint at -0.4 V.

    This is the test an absolute threshold cannot do. Measured on Array42 at 1 uA:
    213.56 uF blanked against 39.41 uF lit, 5.4x apart, and the lit value is the one
    that looks right. Both are wrong; only the pair shows it.
    """
    by_i: dict = {}
    for r in rows:
        by_i.setdefault(r["i_set"], {})[bool(r["lit"])] = r
    for i_set, pair in sorted(by_i.items()):
        a, b = pair.get(False), pair.get(True)
        if a is None or b is None:
            continue
        ca, cb = a["capacitance_F"], b["capacitance_F"]
        if not (np.isfinite(ca) and np.isfinite(cb) and ca > 0 and cb > 0):
            continue
        ratio = max(ca, cb) / min(ca, cb)
        bad = ratio > 1.0 + tol
        a["lit_blank_ratio"] = b["lit_blank_ratio"] = float(ratio)
        if bad:
            a["leakage_suspect"] = b["leakage_suspect"] = True
            log(f"    !! I={i_set*1e9:.0f} nA: lit and blanked disagree by "
                f"{ratio:.2f}x ({cb*1e6:.2f} vs {ca*1e6:.2f} uF). The capacitor cannot "
                f"care whether a sub-turn-on pixel is addressed, so at this setpoint "
                f"the diode's leakage is taking a large share of the ramp current. "
                f"NEITHER value is usable -- not even the one that looks plausible. "
                f"Use a larger --cap-currents entry.")
    good = [r for r in rows if not r.get("leakage_suspect")
            and r.get("park_ok", 1) and np.isfinite(r["capacitance_F"])]
    if good:
        cs = sorted(r["capacitance_F"] for r in good)
        med = cs[len(cs) // 2]
        # A median over readings that disagree by 100x is not a measurement of
        # anything. BR printed "median 39.33 uF (spread 0.33-42.51)" and that is the
        # line a reader quotes, so it has to refuse rather than average.
        span = cs[-1] / cs[0] if cs[0] > 0 else float("inf")
        if span > 3.0:
            log(f"    !! the {len(good)} surviving capture(s) span {span:.0f}x "
                f"({cs[0]*1e6:.2f} to {cs[-1]*1e6:.2f} uF). These cannot all be the "
                f"same capacitance, so there is no median to report. Look at which "
                f"setpoints they came from: a ramp too fast for the sample rate and "
                f"one too slow to leave the noise both survive the paired test.")
        else:
            log(f"    C from {len(good)} usable capture(s): median {med*1e6:.2f} uF "
                f"(spread {cs[0]*1e6:.2f}-{cs[-1]*1e6:.2f})")
    else:
        log(f"    !! no usable capture: every setpoint is leakage-dominated or failed "
            f"to fit. Raise --cap-currents, or park further below turn-on with "
            f"--cap-park.")


def ramp_capacitance(ts, vs, i_set, v_limit):
    """(dV/dt, C) from a constant-current voltage ramp. C = I / (dV/dt).

    Drops samples pinned at the voltage compliance and skips the first 10% of the
    ramp, which carries the source-mode switch transient. Returns (nan, nan) when
    too few usable samples remain, rather than fitting noise.
    """
    ts, vs = np.asarray(ts, float), np.asarray(vs, float)
    if ts.size < 6:
        return float("nan"), float("nan")
    ok = np.isfinite(vs) & (np.abs(vs) < abs(v_limit) * 0.98)
    if ok.sum() < 5:
        return float("nan"), float("nan")
    t_ok = ts[ok]
    mid = ok & (ts > t_ok[0] + 0.1 * (t_ok[-1] - t_ok[0]))
    if mid.sum() < 5:
        return float("nan"), float("nan")
    slope = float(np.polyfit(ts[mid], vs[mid], 1)[0])            # V/s
    return slope, (abs(i_set / slope) if slope else float("nan"))


def cap_source_range(i_set):
    """Smallest 2602B source range that holds i_set, so the setpoint is accurate."""
    for r in m.SMU_I_RANGES:
        if r >= abs(i_set):
            return r
    return m.SMU_I_RANGES[-1]


def exp_leak(pm, smu, smile, pixel, cfg, args, log):
    """Settled current and light at each bias, in THREE frame states.

    Both source modes read the TOTAL cathode current, so neither separates LED
    injection from a parallel leakage path. The reference state has to remove the
    injection WITHOUT changing anything else.

    Blanking the frame does not qualify, and the first attempt at this experiment
    proved it: the blanked current came out HIGHER than the lit current at every
    bias (934 nA vs 250 nA at -0.70 V), giving a negative LED current. It was also
    nearly bias-independent, and the cap ramps had independently inferred the same
    ~900 nA. So an all-off frame leaves the backplane conducting ~5x more than a
    frame with one pixel addressed, and lit-minus-blanked subtracts the wrong number.

    The valid reference is the SAME pixel addressed at bitval 0: identical frame
    structure and select state, zero drive. All three states are recorded so the
    difference between them stays visible:

        i_led   = lit - off      the LED current, from the valid reference
        i_off   = bitval 0, addressed
        i_blank = no frame, for comparison with the earlier runs
    """
    rows = []
    m.configure_voltage_source_b(smu, cfg, log)
    for v in args.voltages:
        meas = {}
        for state in ("lit", "off", "blank"):
            if state == "lit":
                smile.set_pixel(pixel[0], pixel[1], cfg.bitvals[0])
            elif state == "off":
                smile.set_pixel(pixel[0], pixel[1], 0)   # addressed, zero drive
            else:
                smile.blank_frame()
            time.sleep(0.05)
            t, _ia, ib = capture_step(smu, args.v_start, v, args.samples, args.nplc,
                                      chunk=args.chunk, log=log,
                                      limit_a=cfg.nvled_compliance,
                                      capacitance=args.capacitance,
                                      max_presettle_s=args.max_presettle_s,
                                      cfg=cfg, charge_limit=args.charge_limit,
                                      switch_settle_s=args.switch_settle_s)
            met = settling_metrics(t, np.abs(ib))
            spec = keep_trace(f"leak_{state}_f{args.v_start:+.3f}_v{v:+.3f}", t, _ia, ib,
                              args.trace_points)
            pw, pw_sem = read_power(pm, args.step_n_pm)
            meas[state] = (met, pw, pw_sem, spec)
        lit, off, blank = (meas[k] for k in ("lit", "off", "blank"))
        spec = lit[3]
        i_led = lit[0]["settled"] - off[0]["settled"]
        # SEM of each settled value, then added in quadrature. This is the number
        # that decides how far down in J the subtraction stays meaningful: at 95%
        # leakage a 1% error on each state is a 37% error on i_led.
        sems = []
        for st in (lit[0], off[0]):
            n_tail = max(int(0.2 * (st["n"] - st["n_overflow"])), 1)
            sems.append(st["tail_noise"] / np.sqrt(n_tail)
                        if np.isfinite(st["tail_noise"]) else float("nan"))
        i_led_sem = float(np.hypot(*sems))
        p_led = lit[1] - off[1]
        # A non-physical LED current means the reference is still wrong for this
        # device. Say so rather than publishing an EQE computed from it.
        ok = bool(np.isfinite(i_led) and i_led > 0)
        eqe = ((p_led / cfg.photon_energy_j) / (i_led / 1.602176634e-19)) if ok else float("nan")
        rows.append(dict(v_to=v,
                         i_lit=lit[0]["settled"], i_off=off[0]["settled"],
                         i_blank=blank[0]["settled"], i_led=i_led,
                         j_led=to_j(i_led, cfg) if ok else float("nan"),
                         led_current_physical=ok,
                         i_led_sem=i_led_sem,
                         i_led_sigma=abs(i_led) / i_led_sem if i_led_sem else float("nan"),
                         leak_frac=(abs(off[0]["settled"] / lit[0]["settled"])
                                    if lit[0]["settled"] else float("nan")),
                         blank_vs_off=(abs(blank[0]["settled"] / off[0]["settled"])
                                       if off[0]["settled"] else float("nan")),
                         p_lit=lit[1], p_off=off[1], p_blank=blank[1], p_led=p_led,
                         p_sem=lit[2], p_sigma=abs(p_led) / lit[2] if lit[2] else float("nan"),
                         eqe_led=eqe,
                         settled_lit=lit[0]["settled_ok"], settled_off=off[0]["settled_ok"],
                         over_lit=lit[0]["n_overflow"], over_off=off[0]["n_overflow"],
                         charge_limit=args.charge_limit or "",
                         **{f"spec_{k}": v2 for k, v2 in spec.items()},
                         smu_errors=drain_errors(smu)))
        log(f"    V={v:+.2f}  lit {lit[0]['settled']*1e9:+8.1f}  off "
            f"{off[0]['settled']*1e9:+8.1f}  blank {blank[0]['settled']*1e9:+8.1f} nA"
            f"  -> LED {i_led*1e9:+8.1f}+-{i_led_sem*1e9:.1f} nA "
            f"({abs(i_led)/i_led_sem if i_led_sem else 0:.0f}sig)"
            + ("" if ok else " << NON-PHYSICAL")
            + f"  P {p_led*1e12:+7.1f} pW ({abs(p_led)/lit[2] if lit[2] else 0:.0f}sig)"
            + (f"  EQE {100*eqe:6.3f}%" if ok else ""))
    smile.blank_frame()
    return rows


def exp_isrc(pm, smu, smile, pixel, cfg, args, log):
    """Current-source mode, held long enough to actually equilibrate."""
    rows = []
    smile.set_pixel(pixel[0], pixel[1], cfg.bitvals[0])
    m.set_nplc(smu, max(args.nplc, 0.1))
    for i_set in args.isrc_currents:
        # Source range PER setpoint. On the 1 mA default a 100 nA setpoint is 0.01% of
        # range and the SMU delivered 60 nA instead of 100 - measured 261013.
        src_range = cap_source_range(i_set)
        m.configure_current_source_b(
            smu, dataclasses.replace(cfg, current_source_range_i=src_range))
        m.set_nplc(smu, max(args.nplc, 0.1))
        m.set_current_b(smu, -abs(i_set))      # ch B sinks the LED current
        t0 = time.perf_counter()
        last = None
        while (dt := time.perf_counter() - t0) < args.isrc_seconds:
            if hasattr(smu, "measure_instant"):
                ia, ib = smu.measure_instant()
            else:                               # sim rig
                a, b = smu.read_buffers(1)
                ia, ib = (a[0] if a else float("nan")), (b[0] if b else float("nan"))
            vb, _ = m.read_voltage_b(smu, n=2)
            p, _ = read_power(pm, 2)
            dvdt = ((vb - last[1]) / (dt - last[0])) if last else float("nan")
            rows.append(dict(i_set=i_set, j_set=to_j(i_set, cfg), t_s=dt, i_a=ia, i_b=ib,
                             v_b=vb, dvdt=dvdt, power_W=p,
                             src_range=src_range,
                             leak_ratio=abs(ia / ib) if ib else float("nan")))
            last = (dt, vb)
            time.sleep(args.isrc_period)
        tail = [r for r in rows if r["i_set"] == i_set][-5:]
        log(f"    I_set={i_set*1e9:>7.1f} nA (J={to_j(i_set, cfg):.3f}) after "
            f"{args.isrc_seconds:.0f}s: chA={np.mean([r['i_a'] for r in tail])*1e9:+8.1f} nA  "
            f"chB={np.mean([r['i_b'] for r in tail])*1e9:+8.1f} nA  "
            f"V={np.mean([r['v_b'] for r in tail]):+.4f}  "
            f"dV/dt={np.nanmean([r['dvdt'] for r in tail])*1e3:+.3f} mV/s")
    m.set_current_b(smu, 0.0)
    smile.blank_frame()
    return rows


def exp_stepdir(pm, smu, smile, pixel, cfg, args, log):
    """One voltage reached from a small and from a large step."""
    rows = []
    smile.set_pixel(pixel[0], pixel[1], cfg.bitvals[0])
    m.configure_voltage_source_b(smu, cfg, log)
    if not cfg.highc_b:
        # In High-C the range comes from the limit and the setting is locked
        # (error 5069), so writing it would be refused.
        m.set_measure_range_b(smu, args.ranges[0])
        time.sleep(0.05)
    for v_from in args.stepdir_from:
        t, _ia, ib = capture_step(smu, v_from, args.stepdir_target, args.samples, args.nplc,
                                     chunk=args.chunk, log=log,
                                     limit_a=cfg.nvled_compliance,
                                     capacitance=args.capacitance,
                                     max_presettle_s=args.max_presettle_s,
                                      cfg=cfg, charge_limit=args.charge_limit,
                                      switch_settle_s=args.switch_settle_s)
        met = settling_metrics(t, np.abs(ib))
        # Keep the trace: stepdir is the experiment that measures settle time against
        # step size, which is what sets the sweep's budget, so it wants the same plots.
        spec = keep_trace(f"stepdir_hc{int(cfg.highc_b)}_f{v_from:+.3f}"
                          f"_v{args.stepdir_target:+.3f}", t, _ia, ib, args.trace_points)
        rows.append(dict(v_from=v_from, v_to=args.stepdir_target,
                         dv=args.stepdir_target - v_from, **met,
                         **{f"spec_{k}": v for k, v in spec.items()}))
        log(f"    {v_from:+.2f} -> {args.stepdir_target:+.2f} V "
            f"(dV={args.stepdir_target-v_from:+.2f}): I={met['settled']*1e9:+8.1f} nA  "
            f"t1%={met['t_1pct']:.3f}s t0.1%={met['t_01pct']:.3f}s")
    smile.blank_frame()
    return rows


def exp_switch(pm, smu, smile, pixel, cfg, args, log):
    """Does a PIXEL switch at a fixed bias cost anything?

    The sweep pays its 54 s dwell once per bias per pixel, because it runs
    pixel-major: for each pixel, walk the biases. If a pixel switch at a fixed bias
    settles in milliseconds, the loops can be swapped -- walk the pixels inside each
    bias -- and the dwell is paid once per bias for the whole set. For 100 pixels that
    is 18 h against 45 min.

    The fast scan already suggests it: 30 ms per pixel, blanking the frame between
    each, and it reproduces the 54 s-dwell sweep to 11 % at -1.35 V. But that is
    35 uA. This test asks the same question at the faint end, where 450 nA flows and
    a blanked frame's own leakage (section 6: ~4x the bitval-0 level) is no longer
    negligible.

    Procedure, per bias:

      1. light the `--switch-from` pixel and hold the bias until the node is settled
      2. start the burst, record `--switch-pre-s` of baseline, then light the target
         pixel with ONE set_pixel (no blank frame first) while the SMU integrates. Host times bracket the
         burst start and the frame write, so switch.csv places the switch on the
         trace (sync_switch_sent_s .. sync_switch_ack_s). Before 2026-10-25 the
         switch came FIRST and the burst started after it, so no record before then
         contains the switch itself.
      3. compare the first second after the switch's ack against the spread of every
         later second

    Run it WITH --highc. Without it the per-sample noise is ~20 nA against a 480 nA
    reading, which cannot resolve the faint end below about 1 %; with it the noise is
    0.8 nA (section 13) and the bound is ~0.1 %:

        --what switch --highc --pixels 35,36 --switch-from 39 40
        --voltages -0.80 -0.85 -1.00 --switch-hold-s 300 --samples 500 --nplc 1.0

    Measured on Array42, 261014-2151, High-C off: the first second sits 0.3-0.9 %
    below the later ones at -0.85 and -1.00 V, and is lost in the noise at -0.80 V.
    261014-2223 with High-C put the bound at 0.5 % with 0.13 % resolution.

    !! THAT ANSWER IS WRONG, and the way it is wrong is the point of this note.

    The sweep settled it directly. Run 261015-0822 (bias_major, 16 pixels) against the
    pixel-major run on the same 16: the pixel that holds the frame through the 54.47 s
    bias dwell lands at -0.70 % in current, and every pixel that gets only the 2 s
    switch wait lands at -3.0 % in current and -5.1 % in optical power. Six times this
    experiment's bound.

    -- and then run 261015-0958 repeated that at a 20 s switch wait, and the deficit did
    NOT shrink (-2.99 % -> -3.25 % in current). So the 3 % is real but it is not a
    switch transient at all, and this experiment's "a switch is free" verdict is, by
    accident, closer to the truth than the 3 % reading of it was. What the 3 % is
    instead: a 4-6 mV offset on the voltage axis that does not decay with waiting.
    At matched CURRENT the two loop orders agree to 1.9 %. See section 17.

    Two reasons it could not see that, and both are design, not noise:

      * it watches the TOTAL channel-B current. At the faint end 58 % of that is
        backplane leakage (section 6), so a 3.5 % error in the LED's own current
        arrives here as 1.5 %, and the optical channel -- where the deficit is
        largest at 5.1 % -- is not recorded at all.
      * it references the early samples to its OWN TAIL. With `--samples 500 --nplc 1.0`
        the record is 20 s. A transient still running at 20 s moves the reference with
        the signal, and the comparison returns the curvature inside the record instead
        of the offset. This is the record-limited artifact of section 3, in a test
        written to avoid the tolerance-band version of the same mistake.

    Both lessons stand even though the conclusion they were written to support fell
    over: this experiment cannot resolve what the sweep depends on, and a record read
    against its own tail cannot see a transient longer than itself.

    If this is ever run again, give it a record that flattens inside itself. At nplc 1.0
    one sample is 40 ms, so 3000 samples is 120 s:

        --what switch --highc --pixels 35,36 --switch-from 39 40
        --voltages -0.80 -1.05 -1.35 --switch-hold-s 300 --samples 3000 --nplc 1.0

    ``switch_dev_pct`` is the worst of the first five seconds against the tail, which
    is what run 261016-0726 showed the old "first second vs later" could not see.
    Still confirm ``tail_drift_frac`` is small before believing it. It is NOT the next
    experiment to run -- section 17 names two sweep runs that are, and both measure the
    LED's own current and light rather than the total cathode current.
    """
    rows = []
    x, y = pixel
    # "X,Y" like --pixels, or the older "X Y"
    fx, fy = ([int(v) for v in ",".join(args.switch_from).replace(" ", ",").split(",") if v]
              if args.switch_from else (x, y))
    m.configure_voltage_source_b(smu, cfg, log)
    if not cfg.highc_b:
        m.set_measure_range_b(smu, args.ranges[0])
        time.sleep(0.05)
    rec_s = args.samples * max(2 * args.nplc / 50.0, 200e-6)
    if args.switch_hold_s < 10 * rec_s:
        log(f"    !! hold {args.switch_hold_s:.0f}s is short against a {rec_s:.0f}s "
            f"record: residual BIAS drift will show up as switch settling. "
            f"Use --switch-hold-s >= {10 * rec_s:.0f} or fewer --samples.")
    for v in args.voltages:
        # Hold the bias with the OTHER pixel lit, so the capture measures a frame
        # change and not a turn-on from a blank array.
        smile.blank_frame()
        smile.set_pixel(fx, fy, cfg.bitvals[0])

        to_bv = cfg.bitvals[0] if args.switch_to_bitval is None \
            else args.switch_to_bitval

        # The optical channel, if asked for. §19's Q4 could not answer "does the light
        # follow the current through a frame change" because only the dwell had P(t).
        # With a baseline it starts just before the burst and is moved onto the SMU's
        # axis afterwards. Without one (legacy) it starts inside the trigger, so both
        # records' t = 0 is the frame change.
        ptrace = PowerTrace(pm, args.switch_n_pm) if args.switch_n_pm else None
        pre_s = args.switch_pre_s if args.switch_pre_s >= 0 else None

        def _switch(_x=x, _y=y, _b=to_bv):
            # ONE frame write, no blank_frame() first. set_pixel opens with RESET_DIS,
            # which clears the display itself, so the old pixel goes off and the new
            # one comes on in a single acknowledged instruction. The blank used to be
            # a second write -- a full 2048-word image -- inside the switch window.
            smile.set_pixel(_x, _y, _b)
            if ptrace is not None and pre_s is None:
                ptrace.start()

        # v_from == v_to: the bias never moves, so the pre-settle is the only wait and
        # the trigger is the frame change.
        sync = {}
        wake = {}

        def _relight(_x=fx, _y=fy, _b=cfg.bitvals[0]):
            # Re-write the SAME lit pixel during the hold. The FPGA's USB link sleeps
            # after 10-30 s without traffic, and the next write then waits ~1 s for it
            # to wake (fpga_timing.py, 261025: 982 ms after 30 s, 1058 ms after 60 s).
            # Without this the switch window was ~1 s on every run with a 60 s hold.
            smile.set_pixel(_x, _y, _b)

        def _relight_then_record():
            # once more just before the record, timed: ~1000 ms here would mean the
            # link slept anyway and the refresh period is too long
            t = time.perf_counter()
            _relight()
            wake["ms"] = 1e3 * (time.perf_counter() - t)
            if ptrace is not None:
                ptrace.start()

        t, ia, ib = capture_step(smu, v, v, args.samples, args.nplc,
                                 settle_first_s=args.switch_hold_s,
                                 chunk=args.chunk, log=log,
                                 limit_a=cfg.nvled_compliance,
                                 capacitance=args.capacitance,
                                 max_presettle_s=max(args.max_presettle_s,
                                                     args.switch_hold_s),
                                 cfg=cfg, charge_limit=args.charge_limit,
                                 switch_settle_s=args.switch_settle_s,
                                 trigger=_switch, pre_s=pre_s, sync=sync,
                                 channels=args.switch_channels,
                                 on_fire=_relight_then_record if pre_s is not None else None,
                                 hold_tick=_relight, tick_s=args.switch_refresh_s)
        # Every metric below asks what happens AFTER the switch, so it reads the record
        # from the switch's ack on, with t = 0 there. The baseline before it is kept in
        # the trace for the plot.
        t_sw = sync.get("trigger_hi_s", 0.0) if not sync.get("start_ack_late") else 0.0
        post = t >= t_sw
        t_post, ib_post = t[post] - t_sw, ib[post]
        met = settling_metrics(t_post, np.abs(ib_post))
        tag = (f"switch_hc{int(cfg.highc_b)}_v{v:+.3f}"
               f"_{fx}-{fy}to{x}-{y}_b{to_bv}")
        spec = keep_trace(tag, t, ia, ib, args.trace_points)
        p_t = p_p = np.array([])
        if ptrace is not None:
            p_t, p_p = ptrace.stop()
            if pre_s is not None and ptrace.t0 is not None and "host_t0" in sync:
                p_t = p_t + (ptrace.t0 - sync["host_t0"])     # onto the SMU's axis
            keep_power_trace(tag, p_t, p_p)
            if p_t.size > 2 and log:
                span = p_t[-1] - p_t[0]
                log(f"      P(t): {p_t.size} samples over {span:.1f}s "
                    f"({p_t.size / max(span, 1e-9):.1f} Hz vs the SMU's "
                    f"{t.size / max(t[-1], 1e-9) if t.size else 0:.1f} Hz)")
        nan = float("nan")
        p_lit, p_std = read_power(pm, args.step_n_pm) if args.step_n_pm else (nan, nan)

        # The honest bound, and the reason this does not read t_1pct. A short capture
        # of a quiet signal returns t_1pct = half a smoothing window -- a sentinel
        # meaning "nothing was ever outside the band", not a measured settling time.
        # Without High-C that band is ~10 % of the reading, so "t1%=0.1s" would
        # announce success from a measurement that cannot see 10 %. Instead: compare
        # the FIRST second against the spread of every later second. That spread
        # already contains the correlated low-frequency noise an N**-0.5 estimate
        # throws away, so it is the error bar this question has to clear.
        first, later, res = switch_deviation(t_post, ib_post)
        dev = 100.0 * (first / later - 1.0) if later else nan
        res_pct = 100.0 * res / later if later else nan
        sigma = (first - later) / res if res else nan

        rows.append(dict(v=v, from_x=fx, from_y=fy, to_x=x, to_y=y,
                         to_bitval=to_bv,
                         hold_s=args.switch_hold_s, **met, power_W=p_lit,
                         power_std=p_std, first_s_a=first, later_s_a=later,
                         switch_dev_pct=dev, resolution_pct=res_pct,
                         switch_sigma=sigma,
                         from_quadrant=_quadrant_of(fx, fy), to_quadrant=_quadrant_of(x, y),
                         # Where the switch sits on the stored trace, in its own
                         # seconds-from-first-sample. NaN for a legacy capture
                         # (--switch-pre-s < 0), which starts after the switch.
                         trace_tag=tag, pre_s=nan if pre_s is None else pre_s,
                         sync_first_unc_s=sync.get("first_unc_s", nan),
                         sync_switch_sent_s=sync.get("trigger_lo_s", nan),
                         sync_switch_ack_s=sync.get("trigger_hi_s", nan),
                         sync_start_ack_late=sync.get("start_ack_late", nan),
                         # the last re-write of the hold pixel, just before the
                         # record: ~1000 ms means the link slept during the hold
                         fpga_refresh_ms=wake.get("ms", nan),
                         pm_on_trace_axis=int(ptrace is not None and pre_s is not None),
                         **{f"spec_{k}": val for k, val in spec.items()}))

        if not np.isfinite(sigma):
            verdict = "no verdict: the record is too short to form an error bar"
        elif abs(sigma) < 2:
            verdict = (f"FREE to within what this can see "
                       f"(<{max(abs(dev), 2 * res_pct):.1f}%)")
        elif abs(dev) < 1.0:
            verdict = f"real but small: {dev:+.2f}% in the worst early second"
        else:
            verdict = f"COSTS {abs(dev):.1f}% -- the pixel owns part of the dwell"
        log(f"    {v:+.3f} V  ({fx},{fy}) -> ({x},{y})  "
            f"I={met['settled']*1e9:+9.1f} nA  "
            f"worst early 1s {dev:+.2f}% vs tail ({res_pct:.2f}% resolution, "
            f"{sigma:+.1f} sigma)  rec={t[-1] if t.size else 0:.1f}s")
        log(f"            << {verdict}")
        if not cfg.highc_b:
            log(f"            (High-C OFF: the noise floor is ~20 nA, so the faint end "
                f"cannot be resolved below ~1%. Re-run with --highc.)")
    smile.blank_frame()
    return rows


def exp_decay(pm, smu, smile, pixel, cfg, args, log):
    """Charge-decay: let the capacitor discharge through the LED and watch V(t).

    ``I = C*dV/dt``. At 100 nA into 50 uF that is 2 mV/s, which the SMU's voltmeter
    resolves far better than its 100 nA current range resolves 100 nA. The decay
    sweeps the operating point continuously, so one run yields a dense I(V) curve
    instead of discrete points. Run blanked too: leakage discharges the cap as
    well, and the lit-minus-blanked difference is the LED current.
    """
    rows = []
    for lit in (False, True):
        m.configure_voltage_source_b(smu, cfg, log)
        if lit:
            smile.set_pixel(pixel[0], pixel[1], cfg.bitvals[0])
        else:
            smile.blank_frame()
        smu.set_voltage("b", float(args.decay_from))
        time.sleep(args.decay_charge_s)        # charge the cap at the SMU's compliance
        m.configure_current_source_b(
            smu, dataclasses.replace(cfg, current_source_range_i=args.decay_source_range))
        m.set_current_b(smu, 0.0)              # float the node: it discharges via the LED
        t0 = time.perf_counter()
        last = None
        while (dt := time.perf_counter() - t0) < args.decay_seconds:
            vb, vbs = m.read_voltage_b(smu, n=args.decay_n_v)
            p, ps = read_power(pm, args.decay_n_pm)
            dvdt = ((vb - last[1]) / (dt - last[0])) if last else float("nan")
            i_est = args.capacitance * dvdt    # sign follows dV/dt
            rows.append(dict(lit=lit, t_s=dt, v_b=vb, v_b_sem=vbs, dvdt=dvdt,
                             i_from_dvdt=i_est, j_from_dvdt=to_j(i_est, cfg),
                             power_W=p, power_sem=ps))
            last = (dt, vb)
            time.sleep(args.decay_period)
        tail = rows[-5:]
        log(f"    lit={lit!s:5} end V={tail[-1]['v_b']:+.4f}  "
            f"dV/dt={np.nanmean([r['dvdt'] for r in tail])*1e3:+.4f} mV/s  "
            f"-> I={np.nanmean([r['i_from_dvdt'] for r in tail])*1e9:+8.2f} nA  "
            f"P={np.nanmean([r['power_W'] for r in tail])*1e12:+8.1f} pW")
    m.set_current_b(smu, 0.0)
    smile.blank_frame()
    return rows


# ----------------------------------------------------------------------
#  array-wide experiment
# ----------------------------------------------------------------------

def exp_optfloor(pm, smu, smile, cfg, args, log):
    """PM400 noise floor vs averaging, array blanked.

    This is the binding limit on how low in J you can go, because the light scales
    with the current. At J = 0.1 (100 nA) the emission is of order 50-90 pW, so the
    floor has to be well under that. Measured, not assumed.
    """
    rows = []
    smile.blank_frame()
    smu.set_voltage("b", 0.0)
    time.sleep(0.2)
    for n_pm in args.optfloor_n:
        reps = []
        t0 = time.perf_counter()
        for _ in range(args.optfloor_reps):
            p, sem = read_power(pm, n_pm)
            reps.append(p)
        # per-point cost of this averaging, which is what a sweep has to pay
        per_point_s = (time.perf_counter() - t0) / max(len(reps), 1)
        a = np.asarray(reps, float)
        spread = float(a.std(ddof=1)) if a.size > 1 else float("nan")
        rows.append(dict(n_pm=n_pm, pm_range=cfg.pm_range, pm_averaging=cfg.pm_averaging,
                         reps=len(reps), mean_W=float(a.mean()),
                         repeat_std_W=spread,
                         detectable_pW=spread * 3e12,
                         per_point_s=per_point_s,
                         min_j_3sigma=to_j(spread * 3 / args.responsivity_w_per_a, cfg)
                         if args.responsivity_w_per_a else float("nan")))
        log(f"    N_PM={n_pm:>4} x{args.optfloor_reps}: offset {a.mean()*1e12:+8.1f} pW, "
            f"repeatability {spread*1e12:8.2f} pW  -> 3-sigma detectable "
            f"{spread*3e12:8.1f} pW  ({per_point_s*1e3:.0f} ms/point)")
    return rows


def _quadrant_of(x, y):
    """Which 256x256 quadrant a logical pixel sits in."""
    return ("TL", "TR", "BL", "BR")[(1 if x >= 256 else 0) + 2 * (1 if y >= 256 else 0)]


def settling_verdict(srows: list[dict], ks: set, settle_s: float) -> list[str]:
    """Read a probe scan and say what dwell each K needs.

    Significance, not percentage. A probe is ONE reading, so its uncertainty is the
    per-rep scatter -- and at the faint end that scatter IS the ~30 pW optical floor. A
    single pixel emitting 455 pW cannot resolve a 7 % deviation, while a K=16 group
    emitting 5863 pW resolves 2.7 % at 4.5 sigma. Ranking by percentage inverts the
    answer, which is what the first version of this did (run 261019-1057).

    Three outcomes per K, and the middle one is the only useful one:

      * nothing above 3 sigma -> an UPPER BOUND, not a zero. Said as such.
      * the last significant probe has a later, insignificant one after it -> that
        later time is the dwell this K needs.
      * every probe is still moving -> the dwell is past the end of the scan, and the
        scan has to be extended. Reporting a number here would be a guess.
    """
    out = []
    for k in sorted(ks):
        scan = [r for r in srows if r["k"] == k and np.isfinite(r["dev_sigma"])]
        if not scan:
            continue
        scan.sort(key=lambda r: r["probe_s"])
        out.append(
            f"    settling K={k}, probe minus settled against the per-rep scatter: "
            + ", ".join(f"{r['probe_s']*1e3:.0f}ms:{100*r['dev_frac']:+.2f}%"
                        f"({r['dev_sigma']:.1f}s)" for r in scan))
        # Drift first, because it decides whether a settling time can be read at all.
        # A transient decays; a persistent offset does not, and no dwell fixes it. Test
        # the tail's MEAN against its own spread, not each point against the per-rep
        # scatter. Run 261019-1147: every probe from 2 s to 20 s reads high, mean
        # +0.77 %, 6.4 sigma on the mean.
        tail = scan[len(scan) // 2:]
        drift = None
        if len(tail) >= 4:
            fr = np.array([r["dev_frac"] for r in tail], float)
            sem = fr.std(ddof=1) / np.sqrt(len(fr))
            if sem > 0 and abs(fr.mean()) / sem >= 3.0:
                drift = (fr.mean(), abs(fr.mean()) / sem)

        sig = [r for r in scan if r["dev_sigma"] >= 3.0]
        if not sig:
            res = [abs(r["dev_W"]) / r["dev_sigma"] / r["p_settled_W"]
                   for r in scan if r["dev_sigma"] > 0]
            out.append(
                f"      -> nothing resolved at K={k}. A probe is one reading and its "
                f"noise is the floor, so the deviation is under "
                f"~{100 * max(res, default=float('nan')):.1f}% and NOT shown to be "
                f"zero. Pool harder or raise --cluster-n-pm.")
        else:
            # The dwell is the first time whose WHOLE TAIL is quiet, and the tail needs
            # at least two probes -- one quiet point at the end of the scan has nothing
            # after it to confirm. Taking the last significant probe and the next one
            # after it reads noise around a floor as a settling time: run 261019-1147
            # has 500 ms at 2.0 sigma and 1000 ms at 3.2 sigma.
            dwell = next((r["probe_s"] for i, r in enumerate(scan)
                          if len(scan) - i >= 2
                          and all(q["dev_sigma"] < 3.0 for q in scan[i:])), None)
            worst = max(sig, key=lambda r: r["probe_s"])
            out.append(
                f"      -> NOT settled at {worst['probe_s']*1e3:.0f} ms "
                f"({100*worst['dev_frac']:+.2f}%). "
                + (f"Drift dominates past that (see below), so this scan cannot give a "
                   f"settling time: a transient under a {100*abs(drift[0]):.2f}% offset "
                   f"is not resolvable. Shorten the scan and raise --cluster-reps to "
                   f"separate them."
                   if drift is not None else
                   f"Quiet from {dwell*1e3:.0f} ms on, so that is the dwell K={k} "
                   f"needs -- not the single-pixel one."
                   if dwell is not None else
                   f"No probe time has two quiet probes after it, so this scan does "
                   f"not reach the dwell. Extend --cluster-probe-s past "
                   f"{settle_s*1e3:.0f} ms."))
        if drift is not None:
            out.append(
                f"      -> and it does NOT decay: the last {len(tail)} probe(s), "
                f"{tail[0]['probe_s']*1e3:.0f}-{tail[-1]['probe_s']*1e3:.0f} ms, sit "
                f"{100*drift[0]:+.2f}% from settled ({drift[1]:.1f} sigma on the mean). "
                f"That is DRIFT, not a transient -- the reference was taken at "
                f"{tail[-1]['settle_ref_s']:.0f} s and every probe before it is on the "
                f"same side. No dwell removes it; it is what warmup_ms and the "
                f"round-robin measurement order are for.")
    return out


def exp_cluster(pm, smu, smile, cfg, args, log):
    """Pooled K-pixel measurement. Does the faint-end scatter fall as 1/sqrt(K)?

    ROADMAP item 5, and the only live attack on open question 1. The PM400 dark
    reading wanders 75 pW under a 210 pW signal, and faint-end power scatters
    8.5-8.9 % under BOTH sweep methods -- which is why the suspect is the meter and
    not the method. Lighting K pixels multiplies the light by K and leaves the drift
    where it is.

    MEASURED, run 261019-0952 at -0.82 V: the scatter is the meter, and it is
    ADDITIVE. The number that says so is the ABSOLUTE scatter, ``p_led_sd_W``, which
    was flat at 30.3 pW while the signal grew 12.7x from K=1 to K=16 (exponent
    K^-0.10). Relative scatter therefore falls as K^-1.02 -- a factor of K, not
    sqrt(K): 8.52 % at one pixel, 1.76 % at K=4, 0.52 % at K=8.

    So read the columns in this order:

      * ``p_led_sd_W`` FLAT across K   -> additive noise in the meter. Pooling buys a
        factor of K. This is what the array does.
      * ``p_led_sd_W`` rising as sqrt(K) -> independent per-member noise. Pooling buys
        sqrt(K), which is what averaging already buys more cheaply.
      * ``p_led_sd_W`` rising as K     -> the scatter is proportional to the signal and
        pooling buys nothing.

    ``rel_scatter_x_sqrt_k`` is kept for continuity and because it separates the middle
    case from the other two: it is flat only for independent per-member noise. An
    earlier version of this docstring called that flatness the meter-noise signature,
    which was the sqrt(K) hypothesis and is wrong. On the real array the column FALLS
    (0.0852, 0.0485, 0.0352, 0.0147, 0.0253), and falling is the stronger result.

    ``scatter_vs_pred`` is the same statement from the other side: measured scatter
    over the quadrature sum of the members own scatter. It fell as 1/sqrt(K)
    (1.55, 0.88, 0.56, 0.20, 0.32), which is one SHARED additive term averaging down
    rather than K independent ones.

    Two things this must not get wrong:

      * **the reference pools too.** Section 6: an all-off frame conducts ~4x the
        bitval-0 level, so lit-minus-blanked subtracts the wrong number. The
        reference is the SAME K pixels addressed at ``leak_ref_bitval``.
      * **the members are separated, not adjacent.** ``select_pixels`` hands back
        picks at least ``min_pixel_separation`` apart. Adjacent members would
        confound K-scaling with thermal and optical crosstalk, and the question
        here is only whether the meter noise averages down.

    K is nested: K=4 is the first four members of the K=16 set, so the comparison
    across K is the same pixels plus more, never a different sample.

    Every member is ALSO measured alone first. Without that, run 261019-0914's 9 %
    drop in per-pixel power from K=1 to K=4 is either an IR drop on the shared
    supply or just different pixels, and nothing in the data separates the two.
    The singles give three things the groups cannot:

      * ``additivity_p`` = pooled P / sum of the members' own P. 1.0 means the
        pool is the sum of its parts. Below 1 means lighting K together costs
        something -- supply droop is the suspect.
      * ``p_sd_pred_W`` = the pooled scatter you would see if each member's noise
        were independent, quadrature-summed from the members' own scatter.
        ``scatter_vs_pred`` near 1 means independent noise, below 1 means a
        shared term averaging down, above 1 means a shared term adding up.
      * ``singles_rel_spread`` = the member-to-member brightness spread, measured
        on these pixels rather than assumed from the array median.
    """
    members = [(p["X"], p["Y"]) for p in
               select_pixels(pm, smu, smile, cfg,
                             argparse.Namespace(**{**vars(args),
                                                   "quadrants": [args.quadrants[0]],
                                                   "per_quadrant": max(args.cluster_k)}),
                             log)]
    quads = sorted({_quadrant_of(x, y) for x, y in members})
    if len(quads) > 1:
        raise ValueError(f"a cluster must stay inside one quadrant, got {quads}: the "
                         f"pixel-enable bits are per frame, not per pixel")
    ks = [k for k in sorted(args.cluster_k) if k <= len(members)]
    if not ks:
        log(f"    only {len(members)} alive pixel(s), need at least {min(args.cluster_k)}")
        return []
    if max(args.cluster_k) > len(members):
        log(f"    only {len(members)} alive pixel(s) in {quads[0]}, K capped at {max(ks)}")

    # The settled reference must sit PAST the last probe, or settled_frac divides by a
    # reading taken at the same moment and reads 1.0 by construction. 1.5x the last
    # probe, and never less than what was asked for.
    probe_ts = sorted(t for t in args.cluster_probe_s if t > 0)
    if probe_ts:
        args.cluster_settle_s = max(args.cluster_settle_s, 1.5 * probe_ts[-1])
        acq_ms = args.cluster_n_pm * 4.0
        if probe_ts[0] * 1e3 < acq_ms:
            log(f"    note: the first probe is at {probe_ts[0]*1e3:.0f} ms but one "
                f"acquisition takes about {acq_ms:.0f} ms at cluster_n_pm="
                f"{args.cluster_n_pm}, so it averages over the transient rather than "
                f"sampling it. Lower --cluster-n-pm or drop that point.")

    m.configure_voltage_source_b(smu, cfg, log)
    smu.set_voltage("b", args.cluster_v)
    # The bias is set once and never moves, so the charge-limited wait is paid once.
    # Every later wait is a frame change at a held bias, which is the cheap one.
    charge = min(3 * charge_time_s(args.capacitance, args.cluster_v,
                                   cfg.nvled_compliance), args.max_presettle_s)
    log(f"    {len(members)} member(s) in {quads[0]} at {args.cluster_v:+.2f} V, "
        f"K = {ks}, {args.cluster_reps} reps; charging {charge:.1f} s")
    smile.set_pixels(members[:1], cfg.bitvals[0])
    time.sleep(max(charge, args.cluster_settle_s))

    def measure(group):
        """Lit minus the SAME pixels at leak_ref_bitval -- section 6, never blanked.

        Returns (p, i, states, probe) where ``probe`` is ONE early reading taken
        cluster_probe_s after the frame changed, before it has settled. K members are
        K x the switched capacitance on the same row driver and the frame instruction
        spans more rows, so there is no reason to assume a group settles like a single
        pixel. One extra reading per K is what it costs to find out.
        """
        states, probes = {}, []
        for state, bv in (("lit", cfg.bitvals[0]), ("off", cfg.leak_ref_bitval)):
            smile.set_pixels(group, bv)
            if state == "lit" and probe_ts:
                # Walk the probe times in order, sleeping only the gap to the next, so
                # the whole scan costs the one settle it was always going to pay plus
                # one acquisition per time. t is the moment the acquisition STARTS: it
                # lasts about cluster_n_pm x 4 ms, so a probe much under that smears
                # the transient it is trying to sample and the log says so.
                t0_frame = time.perf_counter()
                for t in probe_ts:
                    time.sleep(max(0.0, t - (time.perf_counter() - t0_frame)))
                    probes.append((t, m.acquire_point(pm, smu, args.cluster_n_smu,
                                                      args.cluster_n_pm)))
                time.sleep(max(0.0, args.cluster_settle_s
                               - (time.perf_counter() - t0_frame)))
            else:
                time.sleep(args.cluster_settle_s)
            states[state] = [m.acquire_point(pm, smu, args.cluster_n_smu,
                                             args.cluster_n_pm)
                             for _ in range(args.cluster_reps)]
        p = np.array([r[0] for r in states["lit"]]) - np.array([r[0] for r in states["off"]])
        i = np.array([r[4] for r in states["lit"]]) - np.array([r[4] for r in states["off"]])
        return p, i, states, probes

    # Each member alone, before any group. Same protocol and same rep count, so the
    # single scatter and the pooled scatter are the same quantity.
    singles = {}
    for t in members[:max(ks)]:
        p1, i1, _, _ = measure([t])
        singles[t] = (float(p1.mean()), float(abs(i1).mean()),
                      float(p1.std(ddof=1)) if p1.size > 1 else float("nan"))
    sp = [v[0] for v in singles.values()]
    spread = float(np.std(sp, ddof=1) / np.mean(sp)) if len(sp) > 1 and np.mean(sp) else float("nan")
    log(f"    {len(singles)} member(s) alone: P {np.mean(sp)*1e12:.1f} pW "
        f"+-{100*spread:.1f}% member-to-member")

    rows: list[dict] = []
    srows: list[dict] = []          # one row per (K, probe time) -> cluster_settling.csv
    base = None
    for k in ks:
        grp = members[:k]
        p, i, states, probes = measure(grp)
        off_mean = float(np.mean([r[0] for r in states["off"]]))
        p_sing = sum(singles[t][0] for t in grp)
        i_sing = sum(singles[t][1] for t in grp)
        sd_pred = math.sqrt(sum(singles[t][2] ** 2 for t in grp))
        p_m, i_m = float(p.mean()), float(abs(i).mean())
        p_sd = float(p.std(ddof=1)) if p.size > 1 else float("nan")
        rel = abs(p_sd / p_m) if p_m else float("nan")
        ok = bool(np.isfinite(p_m) and p_m > 0 and i_m > 0)
        rows.append(dict(k=k, quadrant=quads[0], v=args.cluster_v,
                         members=" ".join(f"{x},{y}" for x, y in grp),
                         reps=args.cluster_reps,
                         p_led_W=p_m, p_led_sd_W=p_sd, rel_scatter=rel,
                         # the verdict column: flat across K means meter noise
                         rel_scatter_x_sqrt_k=rel * math.sqrt(k),
                         p_per_pixel_W=p_m / k,
                         # the additivity block: pooled against the members' own singles
                         p_sum_singles_W=p_sing, i_sum_singles_A=i_sing,
                         additivity_p=p_m / p_sing if p_sing else float("nan"),
                         additivity_i=i_m / i_sing if i_sing else float("nan"),
                         p_sd_pred_W=sd_pred,
                         # settling: the FIRST probe, kept here so cluster.csv keeps
                         # its shape. The whole scan is in cluster_settling.csv, one
                         # row per (K, probe time), which is what a dwell needs.
                         p_probe_W=(float(probes[0][1][0] - off_mean)
                                    if probes else float("nan")),
                         probe_s=(probes[0][0] if probes else float("nan")),
                         scatter_vs_pred=p_sd / sd_pred if sd_pred else float("nan"),
                         singles_rel_spread=spread,
                         i_led_A=i_m, i_led_sd_A=float(abs(i).std(ddof=1)) if i.size > 1 else float("nan"),
                         i_per_pixel_A=i_m / k, j_per_pixel=to_j(i_m / k, cfg),
                         p_lit_W=float(np.mean([r[0] for r in states["lit"]])),
                         p_off_W=float(np.mean([r[0] for r in states["off"]])),
                         i_lit_A=float(np.mean([abs(r[4]) for r in states["lit"]])),
                         i_off_A=float(np.mean([abs(r[4]) for r in states["off"]])),
                         pooled_eqe=((p_m / cfg.photon_energy_j)
                                     / (i_m / 1.602176634e-19)) if ok else float("nan"),
                         # K x the current against the ch B limit. A clamped point is the
                         # INSTRUMENT's current, not the device's, so every verdict below
                         # drops it. Measured at -0.90 V on 1e-5: K=8 and K=16 both pinned
                         # at exactly 10.000 uA, which is what made the first version
                         # report 39 % additivity and call it supply droop.
                         at_compliance=int(abs(float(np.mean(
                             [abs(r[4]) for r in states["lit"]])))
                             >= 0.95 * cfg.nvled_compliance),
                         led_current_physical=ok,
                         smu_errors=drain_errors(smu)))
        rows[-1]["settled_frac"] = (rows[-1]["p_probe_W"] / p_m if p_m else float("nan"))
        for t, pr in probes:
            dev = float(pr[0] - off_mean) - p_m
            srows.append(dict(k=k, quadrant=quads[0], v=args.cluster_v, probe_s=t,
                              p_probe_W=float(pr[0] - off_mean), p_settled_W=p_m,
                              settled_frac=((pr[0] - off_mean) / p_m if p_m
                                            else float("nan")),
                              dev_W=dev, dev_frac=dev / p_m if p_m else float("nan"),
                              # the per-rep scatter IS this one reading's uncertainty,
                              # and at the faint end it is the ~30 pW optical floor
                              dev_sigma=abs(dev) / p_sd if p_sd else float("nan"),
                              settle_ref_s=args.cluster_settle_s))
        if base is None:
            base = rel
        log(f"    K={k:>3}: P {p_m*1e12:9.1f} pW +-{p_sd*1e12:7.1f}  "
            f"({100*rel:5.2f}% scatter, x sqrt(K) = {100*rel*math.sqrt(k):5.2f}%)  "
            f"I {i_m*1e9:8.1f} nA  J/px {to_j(i_m/k, cfg):6.3f}"
            + (f"  EQE {100*rows[-1]['pooled_eqe']:6.3f}%" if ok else "  << NON-PHYSICAL")
            + f"  add {rows[-1]['additivity_p']:5.3f}"
            + f"  sd/pred {rows[-1]['scatter_vs_pred']:5.2f}"
            + (f"  settled@{rows[-1]['probe_s']*1e3:.0f}ms "
               f"{100*rows[-1]['settled_frac']:5.1f}%"
               if np.isfinite(rows[-1]["settled_frac"]) else "")
            + ("  << AT COMPLIANCE, NOT A MEASUREMENT"
               if rows[-1]["at_compliance"] else ""))
    smile.blank_frame()
    if srows:
        _EXTRA_ROWS.setdefault("cluster_settling", []).extend(srows)

    clamped = [r for r in rows if r["at_compliance"]]
    if clamped:
        ks = ", ".join(str(r["k"]) for r in clamped)
        need = max(abs(r["i_lit_A"]) for r in clamped) / max(r["k"] for r in clamped)
        log(f"    !! K = {ks} sat ON the {cfg.nvled_compliance*1e6:.0f} uA ch B limit. "
            f"Those rows are the INSTRUMENT's current, not the device's, so every "
            f"verdict below ignores them -- and so should any reading of additivity or "
            f"settling for those K. At {args.cluster_v:+.2f} V a member draws "
            f"~{need*1e6:.2f} uA, so K={max(r['k'] for r in clamped)} needs "
            f"--compliance {10 ** np.ceil(np.log10(need * max(r['k'] for r in clamped) * 1.5)):.0e} "
            f"or a fainter bias. In High-C the compliance also sets the measure range, "
            f"so raising it costs current resolution: prefer the fainter bias.")
    rows_ok = [r for r in rows if not r["at_compliance"]]
    # The settling scan is PER K, so it must not sit behind the K-to-K comparison
    # below. --cluster-k 16 wrote the scan and printed no conclusion until this moved.
    for line in settling_verdict(srows, {r["k"] for r in rows_ok},
                                 args.cluster_settle_s):
        log(line)
    if len(rows_ok) > 1:
        first, last = rows_ok[0], rows_ok[-1]
        gain = first["rel_scatter"] / last["rel_scatter"] if last["rel_scatter"] else float("nan")
        ratio = last["k"] / first["k"]
        # The verdict is the slope of the ABSOLUTE scatter, because that is what tells
        # additive noise from proportional noise. 0 = additive (pooling buys K),
        # +0.5 = independent per member (buys sqrt(K)), +1 = proportional (buys none).
        ks = np.array([r["k"] for r in rows_ok], float)
        sds = np.array([r["p_led_sd_W"] for r in rows_ok], float)
        ok_sd = np.isfinite(sds) & (sds > 0)
        slope = (float(np.polyfit(np.log(ks[ok_sd]), np.log(sds[ok_sd]), 1)[0])
                 if ok_sd.sum() > 1 else float("nan"))
        kind = ("ADDITIVE -- the noise is in the METER, and pooling buys a factor of K"
                if slope < 0.25 else
                "INDEPENDENT PER MEMBER -- pooling buys sqrt(K), which averaging buys "
                "more cheaply" if slope < 0.75 else
                "PROPORTIONAL TO THE SIGNAL -- pooling buys nothing")
        log(f"    absolute scatter goes as K^{slope:+.2f} over K="
            + ",".join(str(r["k"]) for r in rows_ok) + f" -> {kind}")
        log(f"    absolute scatter {' '.join(f'{v*1e12:.1f}' for v in sds)} pW, "
            f"mean {sds.mean()*1e12:.1f} pW -- flat is the whole result, because it is "
            f"what tells additive noise from proportional")
        log(f"    K={first['k']} -> K={last['k']}: relative scatter fell {gain:.2f}x "
            f"(1/sqrt(K) predicts {math.sqrt(ratio):.2f}x, 1/K predicts {ratio:.2f}x)")
        # Significance, not percentage. A probe is ONE reading, so its uncertainty is
        # the per-rep scatter -- and at the faint end that scatter IS the ~30 pW optical
        # floor. A single pixel emitting 455 pW cannot resolve a 7 % deviation, while a
        # K=16 group emitting 5863 pW resolves 2.7 % at 4.5 sigma. Ranking by percentage
        # inverts the answer, which is what the first version of this did.
        add = last["additivity_p"]
        if np.isfinite(add) and abs(add - 1) > 0.03:
            log(f"    !! K={last['k']} emits {100*add:.1f}% of its members' own sum, "
                f"and it is NOT on the current limit. The pool is not the sum of its "
                f"parts, so pooled EQE is not the members' EQE. Supply droop is the "
                f"suspect at {args.cluster_v:+.2f} V. (Check at_compliance first: a "
                f"clamped K reads low here for a reason that has nothing to do with "
                f"the device.)")
        if args.simulate:
            log("    (the sim rig has no additive optical floor -- its PM noise is "
                "purely relative, so it CANNOT average down. Ignore this verdict, and "
                "the additivity one: both need real hardware.)")
    return rows


PER_PIXEL = {"step": exp_step, "cap": exp_cap, "leak": exp_leak, "isrc": exp_isrc,
             "stepdir": exp_stepdir, "switch": exp_switch, "decay": exp_decay}
ARRAY_WIDE = {"optfloor": exp_optfloor, "cluster": exp_cluster}
ALL = [*PER_PIXEL, *ARRAY_WIDE]


# ----------------------------------------------------------------------

def write_csv(out_dir: Path, name: str, rows: list[dict]) -> Path | None:
    if not rows:
        return None
    cols: list[str] = []
    for r in rows:                             # union of keys, first-seen order
        cols += [k for k in r if k not in cols]
    path = out_dir / f"{name}.csv"
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows({c: r.get(c, "") for c in cols} for r in rows)
    return path


def estimate_seconds(what, cfg, args, n_pixels):
    """Rough wall-clock estimate, so a 12-pixel run is a decision and not a surprise.

    Calibrated against the bench: 20000 samples at NPLC 0.01 took 21 s, and the PM
    reads cost ~4 ms each. The pre-settle is the charge time at the configured limit,
    which dominates at low compliance.
    """
    pre = min(max(2.0, 3 * charge_time_s(args.capacitance, args.v_start,
                                         cfg.nvled_compliance)), args.max_presettle_s)
    ch = getattr(args, "step_channels", 2)
    rec = args.samples * sample_interval_s(args.nplc, ch)
    pm_s = args.step_n_pm * 4e-3
    per_capture = pre + rec
    total = 0.0
    n_hc = 2 if args.highc_both else 1
    if "step" in what:
        # With --highc-both the two halves differ: High-C OFF sweeps the ranges,
        # High-C ON uses one (the limit owns the range). Counting len(ranges) for both
        # overstated a 12-pixel run by 1.7x, and the operator budgets from this number.
        n_combo = (len(args.ranges) + 1 if args.highc_both
                   else 1 if args.highc else len(args.ranges))
        total += n_pixels * n_combo * len(args.step_voltages) * (
            per_capture + pm_s * (2 if args.step_dark else 1))
        # the long stage: one capture per pixel, at the faintest bias, High-C on, and
        # it dominates -- 400 s against the fast stage's 2.4 s
        if args.step_long_nplc and args.step_long_samples > 0:
            n_long = len(args.step_long_voltages or [1])
            total += n_pixels * n_long * (
                args.step_long_samples * sample_interval_s(args.step_long_nplc, ch)
                + pm_s * (2 if args.step_dark else 1))
    if "leak" in what:
        # three frame states per bias: lit, addressed-at-bitval-0, blanked
        total += n_pixels * len(args.voltages) * 3 * (per_capture + pm_s)
    if "stepdir" in what:
        total += n_pixels * len(args.stepdir_from) * per_capture
    if "switch" in what:
        # the hold replaces the charge-limited pre-settle: it is the whole point
        total += n_pixels * len(args.voltages) * (args.switch_hold_s + rec + pm_s)
    if "cap" in what:
        total += n_pixels * 2 * len(args.cap_currents) * (4.0 + args.cap_seconds)
    if "isrc" in what:
        total += n_pixels * len(args.isrc_currents) * args.isrc_seconds
    if "decay" in what:
        total += n_pixels * 2 * (args.decay_charge_s + args.decay_seconds)
    if "optfloor" in what:
        total += sum(n * 4e-3 * args.optfloor_reps for n in args.optfloor_n)
    if "cluster" in what:
        # two frame states per K, each one settle plus reps x (PM reads || SMU burst)
        per_rep = max(args.cluster_n_pm * 4e-3, args.cluster_n_smu * 2 / 50.0)
        # the per-member singles run the same protocol, and each lit state pays one
        # extra probe reading
        n_states = (len(args.cluster_k) + max(args.cluster_k)) * 2
        # the settled reference is pushed out to 1.5x the last probe
        settle = max(args.cluster_settle_s,
                     1.5 * max([t for t in args.cluster_probe_s if t > 0], default=0.0))
        total += n_states * (settle + args.cluster_reps * per_rep)
        # plus one acquisition per probe time, for every lit state
        total += ((len(args.cluster_k) + max(args.cluster_k))
                  * max(1, len([t for t in args.cluster_probe_s if t > 0])) * per_rep)
    if not args.pixels:
        total += len(args.quadrants) * args.quadrant_window ** 2 * 0.21
    return total


def build_parser():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--what", nargs="+", default=["step", "cap"],
                    choices=[*ALL, "all"], help="which experiments to run")
    ap.add_argument("--pixels", nargs="+", default=None, metavar="X,Y",
                    help="explicit pixels; otherwise --per-quadrant are found by scan")
    ap.add_argument("--per-quadrant", type=int, default=3,
                    help="alive pixels to characterise per quadrant (default 3)")
    ap.add_argument("--quadrants", nargs="+", default=list(QUADRANTS), choices=QUADRANTS,
                    help="quadrants to find --per-quadrant pixels in (ignored with --pixels)")
    ap.add_argument("--quadrant-window", type=int, default=24,
                    help="edge of the scan window at each quadrant centre")
    ap.add_argument("--alive-nw", type=float, default=2.0,
                    help="turn-on power over dark (nW) for a pixel to count as alive")
    ap.add_argument("--alive-n-pm", type=int, default=8, help="PM reads for the liveness check")
    ap.add_argument("--alive-compliance", type=float, default=1e-3,
                    help="ch B limit used for the liveness check only. Kept healthy on "
                         "purpose: the limit is the charging current, so a low measurement "
                         "compliance makes every pixel read dead")
    ap.add_argument("--sample-name", default=None,
                    help="which array this is, e.g. Array38. It goes in the output "
                         "directory name AND in run_config.json. Without it the "
                         "recorded name is the config default, which is a different "
                         "array -- so a cross-array comparison would label both runs "
                         "the same. Required unless --simulate.")
    ap.add_argument("--out", default=None, help="output root (default: config base_dir)")
    ap.add_argument("--simulate", action="store_true",
                    help="plumbing only - the sim rig has no capacitance model")
    ap.add_argument("--voltages", nargs="+", type=float, default=DEFAULT_VOLTAGES,
                    help="biases (V) for leak and switch")
    ap.add_argument("--step-voltages", nargs="+", type=float, default=STEP_VOLTAGES,
                    help="biases for step/stepdir. Defaults to the four that produced "
                         "the 'settling is not RC' result -- 0.05 to 0.20 V steps from "
                         "--v-start -1.0, where the excess current decays as t^-1.2 "
                         "with amplitude proportional to the step. --voltages keeps its "
                         "wide span because leak and switch want the whole sweep range, "
                         "not four neighbouring points.")
    ap.add_argument("--ranges", nargs="+", type=float, default=DEFAULT_RANGES,
                    help="ch B measure ranges (A) without High-C: step captures at every "
                         "one, stepdir and switch use the first")
    ap.add_argument("--v-start", type=float, default=-1.0,
                    help="voltage to step FROM. -1.0, with --voltages -0.95..-0.80, is "
                         "the configuration that produced the 'settling is not RC' "
                         "result in the README: excess current decaying as t^-1.2 with "
                         "amplitude proportional to the step size. The old -2.6 V "
                         "default did not reproduce that figure.")
    ap.add_argument("--samples", type=int, default=6000,
                    help="buffer samples per step. record_s in the CSV says how much time "
                         "that bought; raise it until the record outlasts the settling")
    ap.add_argument("--trace-points", type=int, default=0,
                    help="cap on raw I(t) samples kept per capture, written to "
                         "traces.npz for plotting and spectra. 0 (the default) keeps "
                         "EVERY sample; a positive value thins UNIFORMLY so the "
                         "frequency content survives. The old 4000 default silently "
                         "dropped a third of each 6000-sample capture to save about "
                         "5 MB on a worst-case run, which is not a trade worth making "
                         "when a decay exponent is the measurement. Use --no-traces to "
                         "turn saving off.")
    ap.add_argument("--no-traces", action="store_true",
                    help="do not write traces.npz at all. The summary CSVs are "
                         "unaffected.")
    ap.add_argument("--chunk", type=int, default=2000,
                    help="samples per TSP burst. The buffer appends, so this only bounds "
                         "how long a single blocking VISA read lasts; lower it if a burst "
                         "times out")
    ap.add_argument("--nplc", type=float, default=0.01,
                    help="integration time for I(t). 0.001 is the SMU minimum but gave "
                         "60-99%% ripple and unusable magnitudes on the bench; 0.01 trades "
                         "5x time resolution for a reading you can believe")
    ap.add_argument("--compliance", type=float, default=None,
                    help="ch B current limit (A). In High-C mode the LIMIT selects the "
                         "measure range, so the 0.5 A default gives mA-level resolution and "
                         "the reading is pure noise. Use ~1e-3: still charges 25 uC in 25 ms, "
                         "but resolves microamps")
    ap.add_argument("--highc", action="store_true", help="run with High-C enabled")
    ap.add_argument("--highc-both", action="store_true",
                    help="run the step experiment with High-C both off and on")
    ap.add_argument("--isrc-currents", nargs="+", type=float, default=[1e-7, 3e-7, 1e-6, 1e-5],
                    help="setpoints for the current-source test (A); 1e-7 is J=0.1")
    ap.add_argument("--isrc-seconds", type=float, default=120.0,
                    help="hold time per setpoint - needs to be minutes, not ms")
    ap.add_argument("--isrc-period", type=float, default=0.5, help="log period (s)")
    ap.add_argument("--switch-from", nargs="+", default=None, metavar="X,Y",
                    help="switch: the pixel lit while the bias settles, before the "
                         "frame changes to the target (default: the target itself, "
                         "which tests re-addressing only)")
    ap.add_argument("--switch-to-bitval", type=int, default=None,
                    help="drive level the frame change writes (default: bitvals[0], "
                         "i.e. lit). Pass 0 to record a LEAKAGE time trace: the pixel "
                         "is held lit and settled, then driven to bitval 0 and the "
                         "cathode current recorded. Nothing else measures whether the "
                         "backplane leak is steady or wanders.")
    ap.add_argument("--switch-hold-s", type=float, default=300.0,
                    help="switch: how long to hold the bias before the frame change. "
                         "Default 300 s because the bias itself needs ~240 s to reach "
                         "1%% at -0.80 V: hold less and the capture times the BIAS "
                         "settling, which is the thing being controlled for. At 300 s "
                         "the residual bias drift is ~0.1%% over a 20 s record.")
    ap.add_argument("--switch-pre-s", type=float, default=0.5,
                    help="switch: seconds of baseline recorded BEFORE the frame change. "
                         "The burst starts first and the switch is sent while it runs, "
                         "so the trace shows before and after, and switch.csv records "
                         "when the switch was sent and acknowledged on the trace's own "
                         "time axis. Negative restores the pre-2026-10-25 order, where "
                         "the switch came first and the record started after it.")
    ap.add_argument("--switch-refresh-s", type=float, default=5.0,
                    help="switch: re-write the hold pixel this often during the hold, "
                         "so the FPGA's USB link never sleeps. It sleeps after 10-30 s "
                         "idle and the next write then takes ~1 s (fpga_timing.py), "
                         "which put a 1 s delay inside every switch window.")
    ap.add_argument("--switch-channels", type=int, choices=(1, 2), default=1,
                    help="switch: 1 (default) reads ch B only, 2 reads ch A as well. ch A "
                         "costs one integration plus 569 us per reading (sample_interval_s) "
                         "and yields only a supply-rail sanity number. Before 2026-10-25 "
                         "the switch test always read both.")
    ap.add_argument("--switch-n-pm", type=int, default=0,
                    help="switch: PM reads per optical sample, ON A BACKGROUND THREAD, "
                         "starting at the frame change. 0 (default) records current "
                         "only, which is every switch run before 16 October 2026. Pass "
                         "1 for the fastest P(t) (~30-80 Hz, denser than the SMU at "
                         "NPLC 1.0) or 4 to trade rate for a lower optical floor. The "
                         "records are NOT synchronous: bin both onto a common grid "
                         "before dividing them. Answers whether the light follows the "
                         "current through a frame change (section 19, Q4).")
    ap.add_argument("--stepdir-target", type=float, default=-0.85,
                    help="stepdir: the bias every step goes TO (V)")
    ap.add_argument("--stepdir-from", nargs="+", type=float, default=[-0.8, -1.0, -1.3, -2.6],
                    help="stepdir: the biases to step FROM (V), one capture each, so the "
                         "approach from above and below can be compared")
    ap.add_argument("--decay-from", type=float, default=-1.3,
                    help="charge to this voltage, then let it decay through the LED")
    ap.add_argument("--decay-charge-s", type=float, default=3.0,
                    help="decay: how long to charge at --decay-from before releasing (s)")
    ap.add_argument("--decay-seconds", type=float, default=120.0,
                    help="decay: how long to log the free decay (s)")
    ap.add_argument("--decay-period", type=float, default=0.5,
                    help="decay: log period (s)")
    ap.add_argument("--decay-n-v", type=int, default=8, help="voltage reads per decay sample")
    ap.add_argument("--decay-n-pm", type=int, default=16, help="PM reads per decay sample")
    ap.add_argument("--decay-source-range", type=float, default=1e-7,
                    help="ch B source current range while holding 0 A. The finest range "
                         "minimises the regulation error that would add to the decay "
                         "current; the blanked run cancels what is left")
    ap.add_argument("--cap-park-tol", type=float, default=0.05, metavar="V",
                    help="how far the ramp may begin from the park bias before the "
                         "capture is rejected (default 0.05 V). C = I/(dV/dt) assumes "
                         "the ramp starts from a known sub-turn-on voltage. Array42 "
                         "parked at -0.400 V and began at -0.401; Array38 began at "
                         "+0.334 V, 0.73 V away and on the wrong side of zero, and "
                         "reported capacitances anyway.")
    ap.add_argument("--cap-currents", nargs="+", type=float, default=[1e-6, 1e-5, 1e-4],
                    help="known currents for the C = I/(dV/dt) ramp (A)")
    ap.add_argument("--cap-park", type=float, default=-0.40,
                    help="bias to park at before the ramp. Below turn-on, so the LED's "
                         "own current is a small fraction of the setpoint")
    ap.add_argument("--cap-v-stop", type=float, default=-0.65,
                    help="stop the ramp here, before the LED turns on and starts taking "
                         "a share of the setpoint")
    ap.add_argument("--cap-seconds", type=float, default=5.0, help="max ramp duration")
    ap.add_argument("--step-channels", type=int, default=1, choices=(1, 2),
                    help="SMU channels measured per buffered sample. 1 (the default) "
                         "reads only ch B, which is the one being stepped. 2 also reads "
                         "ch A, the VLED anode drive, whose record yields one sanity "
                         "number (i_a_settled) that nothing else uses -- and doubles "
                         "the integration per sample. Measured: NPLC 1.0 costs 20.66 ms "
                         "per sample on one channel against 40.67 ms on two, so the "
                         "long stage runs in 207 s instead of 407 s, or gets twice the "
                         "resolution for the same duration. At NPLC 0.01 the 660 us "
                         "fixed overhead dominates and the gain is only 1.23x.")
    ap.add_argument("--step-pm-averaging", type=int, default=1,
                    help="PM400 internal averaging DURING a step trace. The config "
                         "default of 3 means three averages plus a USB round-trip per "
                         "sample, which is why the optical record runs at 30-80 Hz "
                         "where the meter can do roughly 1 kHz. For a settling trace "
                         "that trade is backwards: resolution matters and the asymptote "
                         "is found by averaging the tail. The SETTLED reading still "
                         "uses pm_averaging, so the EQE point stays comparable with "
                         "every other run.")
    ap.add_argument("--step-pm-trace", type=int, default=1,
                    help="PM reads per optical sample during a step capture, on a "
                         "background thread started AT the step. 0 disables. step used "
                         "to take one settled reading and nothing else, so optical "
                         "settling after a bias change was measured nowhere -- and the "
                         "sweep's dwell is ultimately about when the EQE is "
                         "trustworthy, which needs both channels. Lands in traces.npz "
                         "under pm_*, on host time, with t=0 at the step for both "
                         "records. Do not divide the two sample-by-sample: bin them "
                         "onto a common grid first, they are not synchronous.")
    ap.add_argument("--step-long-nplc", type=float, default=1.0,
                    help="NPLC for the SECOND, long step capture. At the fast stage's "
                         "--nplc 0.01 a 6000-sample record is 2.4 s, against a 240 s "
                         "faint-end settle -- so the fast capture cannot see the "
                         "asymptote and its `settled` is the median of a window that "
                         "never got there. 1.0 gives 40 ms per sample, so 10000 "
                         "samples is a 400 s record. 0 disables the long stage.")
    ap.add_argument("--step-long-compliance", type=float, default=1e-5,
                    help="ch B limit for the LONG step capture IN HIGH-C, where the "
                         "limit selects the measure range. Ignored with High-C off, "
                         "because there the range comes from --ranges and the run "
                         "limit gives a 9 ms charge edge instead of 940 ms. Separate "
                         "from "
                         "--compliance because in High-C the limit selects the measure "
                         "range and the two stages want opposite things. At 1 mA the "
                         "charge edge is clean (9 ms) but a 1 mA range leaves ~100 "
                         "nA/sample of noise, and the relaxation this stage exists to "
                         "fit is 38 nA at 4 s -- buried. At 10 uA the range gives ~1 "
                         "nA/sample and the tail resolves past 8 s, at the price of a "
                         "939 ms charge that only matters to the fast stage. The long "
                         "stage runs at the faintest bias, where the current is well "
                         "under 10 uA. 0 uses --compliance for both.")
    ap.add_argument("--step-long-ranges", nargs="+", type=float, default=[1e-6],
                    metavar="A",
                    help="ranges at which the NON-High-C long capture runs. The "
                         "non-High-C pass sweeps --ranges, and a long capture at each "
                         "is 33 min a pixel, so it runs only at these. Default 1e-6, "
                         "the range that fits the ~0.5 uA a faint pixel really draws. "
                         "A value not in --ranges produces no non-High-C long record. "
                         "In High-C there is no range to choose: the limit selects it.")
    ap.add_argument("--step-long-samples", type=int, default=20000,
                    help="buffer samples for the long step capture. 20000 at NPLC 1.0 "
                         "on one channel is 413 s, which clears the 240 s faint-end "
                         "settle measured on Array42 with 1.7x margin. 10000 would be "
                         "207 s -- SHORT of it, which is the mistake --step-channels 1 "
                         "quietly introduced by halving the per-sample time.")
    ap.add_argument("--step-long-voltages", nargs="+", type=float, default=None,
                    help="biases to run the long capture at. Default: the faintest of "
                         "--voltages, which is where slow settling lives. Every "
                         "combination would be 648 captures x 400 s = 72 h, so this is "
                         "deliberately narrow.")
    ap.add_argument("--step-n-pm", type=int, default=512,
                    help="PM reads after each step capture. 512 gives a ~7.6 pW 3-sigma "
                         "floor (measured), against ~219 pW at J=0.25")
    ap.add_argument("--step-dark", action=argparse.BooleanOptionalAction, default=True,
                    help="measure the optical dark at each bias by blanking the pixel. "
                         "Required below ~100 pW, where the PM offset rivals the signal")
    ap.add_argument("--pm-range", type=float, default=None,
                    help="override the PM400 range (W). The floor was measured at the "
                         "1e-7 default; a lower range may lower it, which is the cheapest "
                         "route to usable light below J~0.4")
    ap.add_argument("--dark-w", type=float, default=0.0,
                    help="optical dark level to subtract (W). The measured offset is "
                         "under 7 pW and sign-random, so 0 is usually right")
    ap.add_argument("--charge-limit", type=float, default=None,
                    help="two-phase settle: charge the array at THIS limit, then drop to "
                         "--compliance to measure. 1e-3 moves the 95 uC in 0.09 s against "
                         "8.7 s at a 10 uA limit. In High-C the limit also selects the "
                         "range, so verify against a single-limit run before trusting it")
    ap.add_argument("--switch-settle-s", type=float, default=0.5,
                    help="wait after the charge-to-measure limit switch")
    ap.add_argument("--max-presettle-s", type=float, default=60.0,
                    help="cap on the charge-derived pre-settle before each capture. The "
                         "charge time is C*dV/limit, which at a 1 uA limit is minutes per "
                         "point; exceeding the cap is logged as a warning")
    ap.add_argument("--capacitance", type=float, default=45.7e-6,
                    help="C used to turn dV/dt into a current. Default is the MEASURED "
                         "value: 45.7 +/- 3.8 uF, from the 10 uA and 100 uA cap ramps "
                         "on pixel (54,46), 261013")
    ap.add_argument("--optfloor-n", nargs="+", type=int, default=[4, 8, 32, 128, 512],
                    help="optfloor: PM400 readings per point to test; the floor should "
                         "fall as N^-0.5")
    ap.add_argument("--optfloor-reps", type=int, default=12,
                    help="repeats per N_PM, for the repeatability spread")
    ap.add_argument("--responsivity-w-per-a", type=float, default=22e-9 / 25e-6,
                    help="W of optical power per A of LED current, for the min-J estimate")
    ap.add_argument("--cluster-k", nargs="+", type=int, default=[1, 4, 16],
                    help="how many pixels to light at once, nested: K=4 is the first "
                         "four members of the K=16 set")
    ap.add_argument("--cluster-v", type=float, default=-0.90,
                    help="the faint-end bias the scatter is measured at")
    ap.add_argument("--cluster-reps", type=int, default=12,
                    help="repeats per K, for the scatter. This is the whole statistic, "
                         "so it is the knob that costs time")
    ap.add_argument("--cluster-settle-s", type=float, default=2.0,
                    help="wait after a frame change at the held bias. The bias itself "
                         "is charged once, before the first K")
    ap.add_argument("--cluster-probe-s", nargs="+", type=float, default=[0.05],
                    help="probe time(s) in seconds after the frame changes, before it "
                         "has settled. One value says whether K pixels settle like one; "
                         "a LIST gives the dwell curve a pooled sweep needs, e.g. "
                         "--cluster-probe-s 0.05 0.1 0.2 0.5 1 2 3. The settled "
                         "reference is pushed out to 1.5x the last probe, because "
                         "dividing by a reading taken at the same moment reads 1.0 by "
                         "construction. One acquisition takes about cluster_n_pm x 4 ms, "
                         "so probes much under that average over the transient. 0 "
                         "disables the probe entirely")
    ap.add_argument("--cluster-n-pm", type=int, default=8,
                    help="cluster: PM400 readings per acquisition")
    ap.add_argument("--cluster-n-smu", type=int, default=16,
                    help="cluster: SMU readings per acquisition")
    return ap


def main(argv=None):
    args = build_parser().parse_args(argv)
    what = list(ALL) if "all" in args.what else args.what
    log = m.make_logger()
    cfg = IQEConfig(highc_b=args.highc)
    if args.pm_range is not None:
        cfg = dataclasses.replace(cfg, pm_range=args.pm_range)
        log(f"PM400 range set to {args.pm_range:g} W")
    if args.compliance is not None:
        cfg = dataclasses.replace(cfg, nvled_compliance=args.compliance)
        log(f"ch B current limit set to {args.compliance:g} A"
            + (" (this also selects the High-C measure range)" if args.highc or args.highc_both
               else ""))

    if args.sample_name:
        cfg = dataclasses.replace(cfg, sample_name=args.sample_name)
    elif not args.simulate:
        # Refuse rather than mislabel. Every bench run before 20 October 2026 carries
        # the default name whatever array it was on, and there is no way to tell from
        # the file which ones are right.
        ap_err = (f"--sample-name is required: without it this run would be recorded as "
                  f"{cfg.sample_name!r}, which is the config default and may not be the "
                  f"array on the stage. Pass --sample-name Array38 (or whichever), or "
                  f"--simulate if there is no hardware.")
        log(f"config error: {ap_err}")
        return 2

    root = Path(args.out) if args.out else Path(cfg.base_dir)
    slug = re.sub(r"[^A-Za-z0-9_.-]+", "-", cfg.sample_name).strip("-")
    # SIMULATED_ in the name, so a software-rig run can never pass for a hardware one in
    # results/ -- 24 did before this (report, Appendix C.7).
    if args.simulate:
        slug = "SIMULATED_" + slug
    # The experiments that ran, after CAPACITANCE: <stamp>_<sample>_CAPACITANCE_Switch,
    # ..._CAPACITANCE_Cap_Step. The folder says what is inside without opening it.
    out_dir = root / (f"{datetime.datetime.now():%y%m%d-%H%M}_{slug}_CAPACITANCE_"
                      + "_".join(w.capitalize() for w in what))
    out_dir.mkdir(parents=True, exist_ok=True)
    log(f"Output: {out_dir}")
    (out_dir / "run_config.json").write_text(
        json.dumps({"config": cfg.to_dict(),
                    "args": {k: v for k, v in sorted(vars(args).items())}},
                   indent=2, default=str), encoding="utf-8")
    if args.no_traces:
        args.trace_points = None          # keep_trace skips; the CSVs are unaffected
        log("--no-traces: traces.npz will not be written")
    elif args.trace_points:
        log(f"--trace-points {args.trace_points}: captures longer than that are thinned "
            f"uniformly. The default of 0 keeps every sample.")
    if args.simulate:
        log("SIMULATION - plumbing only, the sim rig has no capacitance model.")

    if args.samples > 50000:
        log(f"WARNING: --samples {args.samples} may exceed the SMU buffer; "
            f"check record_s and n in the CSV against what you asked for")
    # Measured 261013: a voltage-mode current reading without High-C is a range
    # artifact below ~10 uA, not a noisy measurement - it lands at 42-58% of
    # whatever range is selected regardless of bias. step and stepdir read current
    # in voltage mode, so they need it. cap does not: it measures a voltage slope.
    needs_highc = [n for n in what if n in ("step", "stepdir", "leak")]
    if needs_highc and not (args.highc or args.highc_both):
        log(f"WARNING: {', '.join(needs_highc)} measure current in voltage mode without "
            f"High-C. Below ~10 uA that reads ~50% of the selected range regardless of "
            f"bias. Pass --highc (and --compliance, e.g. 1e-5) unless you are "
            f"deliberately reproducing the artifact.")
    n_px = len(args.pixels) if args.pixels else args.per_quadrant * len(args.quadrants)
    est = estimate_seconds(what, cfg, args, n_px)
    log(f"estimated {est/60:.0f} min for {', '.join(what)} on "
        f"{n_px} pixel(s){'' if args.pixels else ' (plus the selection scan)'}")
    pm, smu, smile = m.connect(cfg, log, simulate=args.simulate)
    collected: dict[str, list[dict]] = {name: [] for name in what}
    # try/finally alone does not park on Ctrl-Break (the console's Stop button), a closed
    # console or a Windows shutdown: Python's default handlers end the process without
    # unwinding. safe_exit catches all three and parks exactly once.
    parked = safe_exit(lambda: m.shutdown(pm, smu, smile, cfg), log)
    parked.__enter__()
    try:
        m.configure_common(pm, smu, cfg, log)
        smu.set_voltage("a", cfg.vled_voltage)
        smu.enable_output("a", True)
        smu.enable_output("b", True)
        time.sleep(0.05)

        for name in [n for n in what if n in ARRAY_WIDE]:
            log("=" * 56)
            log(name.upper())
            collected[name] = ARRAY_WIDE[name](pm, smu, smile, cfg, args, log)

        per_pixel = [n for n in what if n in PER_PIXEL]
        if per_pixel:
            log("=" * 56)
            log("PIXEL SELECTION")
            pixels = select_pixels(pm, smu, smile, cfg, args, log)
            if not pixels:
                log("no pixels to characterise")
            for k, px in enumerate(pixels, 1):
                log("=" * 56)
                log(f"PIXEL {k}/{len(pixels)} ({px['X']},{px['Y']}) [{px['quadrant']}]")
                p_on, dark, alive = check_alive(pm, smu, smile, cfg, px["X"], px["Y"], args, log)
                if not alive:
                    continue
                for name in per_pixel:
                    log(f"  {name}")
                    # One failing experiment must not cost the pixels already done.
                    try:
                        rows = PER_PIXEL[name](pm, smu, smile, (px["X"], px["Y"]),
                                               cfg, args, log)
                    except Exception as exc:
                        log(f"    {name} FAILED on ({px['X']},{px['Y']}): "
                            f"{type(exc).__name__}: {exc}")
                        log("    continuing with the next experiment; "
                            "data so far is already on disk")
                        continue
                    for r in rows:
                        r.update(pixel_x=px["X"], pixel_y=px["Y"], quadrant=px["quadrant"],
                                 turnon_W=p_on, dark_W=dark)
                    collected[name].extend(rows)
                    # flush after every experiment, so a later crash keeps this
                    write_csv(out_dir, name, collected[name])
                    # and the raw traces with them. The CSVs alone give `settled` and
                    # `t_1pct` but not the records the exponent is fitted from, and a
                    # run is over an hour -- too long to see nothing until the end.
                    flush_traces(out_dir, log)
    finally:
        parked.__exit__(None, None, None)     # idempotent: shutdown() runs exactly once
        # Writing in the finally is the point: previously an exception here skipped
        # the writes entirely and a run that died partway lost every completed row.
        log("=" * 56)
        npz = flush_traces(out_dir, log)
        if npz:
            log(f"wrote {npz} ({len(_TRACES)} I(t)"
                + (f" + {len(_PTRACES)} P(t)" if _PTRACES else "")
                + (f" + {len(_VTRACES)} V(t)" if _VTRACES else "")
                + f", {npz.stat().st_size / 1e6:.1f} MB)")
        for name, rs in _EXTRA_ROWS.items():
            collected.setdefault(name, []).extend(rs)
        wrote = [p for p in (write_csv(out_dir, n, r) for n, r in collected.items()) if p]
        for p in wrote:
            log(f"wrote {p}")
        if not wrote:
            log("no data written")
        # Last, so it contains everything above -- including the lines that only ever
        # existed in a terminal: the High-C warning, every "<< NOT SETTLED", the
        # pre-settle notices, the optical-vs-electrical comparison. In the finally, so a
        # run that dies partway still leaves its log.
        log(f"wrote {out_dir / 'Console_LOG.txt'}")
        if hasattr(log, "save"):
            log.save(out_dir / "Console_LOG.txt")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
