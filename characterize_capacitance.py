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

Usage:
    python characterize_capacitance.py --what all
    python characterize_capacitance.py --what step cap --per-quadrant 3
    python characterize_capacitance.py --what isrc --pixels 54,33 177,63 --isrc-seconds 300
    python characterize_capacitance.py --what optfloor          # 2 minutes, no pixels needed

Writes one CSV per experiment, every row tagged with its pixel. It fits and plots
nothing; point SMILE_IQE_Analysis at the CSVs for that.

NOTE: ``--simulate`` exercises the plumbing only. The built-in rig has no
capacitance model, so it reports instant settling. The physics needs the bench.
"""
from __future__ import annotations

import argparse
import csv
import dataclasses
import datetime
import json
import time
from pathlib import Path

import numpy as np

import measure_iqe as m
from measure_iqe import IQEConfig

# Voltages worth probing: the sub-turn-on region that reaches J < 1, the rising side
# that pins peak IQE, the effort-split knee, and the droop.
DEFAULT_VOLTAGES = [-0.6, -0.7, -0.8, -0.9, -1.0, -1.1, -1.3, -1.8, -2.6]
# The 10 mA range is what the clean runs used; the fine ones read low J "not just
# noisily but WRONG" before, which is the thing to explain.
DEFAULT_RANGES = [1e-3, 1e-4, 1e-5, 1e-6, 1e-7]
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
    smu.set_voltage("b", cfg.nvled_v_fastscan)
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
                time.sleep(max(cfg.fastscan_settle_ms, 5.0) / 1000.0)
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

def capture_step(smu, v_from, v_to, n_samples, nplc=0.01, settle_first_s=2.0,
                 chunk=2000, log=None, limit_a=None, capacitance=50e-6,
                 max_presettle_s=60.0):
    """Step ch B from v_from to v_to, then burst-capture I(t) with timestamps.

    Returns (t, i_a, i_b), t in seconds from the first sample. Instrument
    timestamps are used, not host time, so the spacing is the SMU's own.

    The burst is issued in chunks. One ``measure_burst(20000)`` blocks the VISA
    read for the whole burst — about 8 s at NPLC 0.01 — which overruns the
    driver's 10 s timeout once overhead is added and then kills the run. The
    buffer is in append mode, so chunks accumulate into one record, and the
    instrument timestamps make the small gap between chunks visible rather than
    silently distorting the time axis. A chunk that still times out is caught: the
    interface is cleared so the stale 'DONE' cannot desync later queries, and
    whatever was captured so far is returned.
    """
    m.set_nplc(smu, nplc)
    smu.setup_buffers(timestamps=True)
    smu.set_voltage("b", float(v_from))
    # The pre-settle has to clear the charge time at the configured limit, or the
    # capture starts before the node has reached v_from and measures the wrong state.
    want = max(settle_first_s, 3 * charge_time_s(capacitance, v_from, limit_a))
    pre = min(want, max_presettle_s)
    if want > max_presettle_s and log:
        log(f"      pre-settle CAPPED at {pre:.0f}s, charge needs {want:.0f}s at "
            f"{limit_a:g} A — the node may not have reached {v_from:+.2f} V. "
            f"Raise --max-presettle-s or the limit.")
    elif pre > settle_first_s and log:
        log(f"      pre-settle {pre:.1f}s (charge-limited at {limit_a:g} A)")
    time.sleep(pre)                            # start from a known, settled state
    smu.clear_buffers_only()
    smu.set_voltage("b", float(v_to))          # the step
    if not hasattr(smu, "inst"):               # sim rig: no TSP, no instrument clock
        t0 = time.perf_counter()
        smu.measure_burst_fire(int(n_samples))
        smu.measure_burst_join()
        ia, ib = smu.read_buffers(int(n_samples))
        return (np.linspace(0.0, time.perf_counter() - t0, len(ib)),
                np.asarray(ia, float), np.asarray(ib, float))

    # two channels per iteration, integration plus overhead, then 4x head-room
    per_sample_s = max(2.0 * nplc / 50.0, 200e-6)
    old_timeout = smu.inst.timeout
    smu.inst.timeout = max(10_000, int(chunk * per_sample_s * 4_000))
    done = 0
    try:
        while done < int(n_samples):
            k = min(int(chunk), int(n_samples) - done)
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
            ia, ib, ta, _tb = smu.read_buffer_with_timestamps()
        finally:
            smu.inst.timeout = old_timeout
    if not ib:
        return np.array([]), np.array([]), np.array([])
    n = min(len(ia), len(ib), len(ta))
    return (np.asarray(ta[:n], float) - ta[0],
            np.asarray(ia[:n], float), np.asarray(ib[:n], float))


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
    w = max(5, iv.size // 100)
    smooth = np.convolve(iv, np.ones(w) / w, mode="valid")
    ts = tv[w - 1:]
    base["smooth_window"] = int(w)
    for frac, key in ((0.01, "t_1pct"), (0.001, "t_01pct")):
        tol = max(frac * abs(settled), 5.0 * noise / np.sqrt(w))
        base[key + "_tol"] = tol
        bad = np.nonzero(np.abs(smooth - settled) > tol)[0]
        if bad.size == 0:
            base[key] = 0.0                     # in band from the first valid sample
        elif bad[-1] + 1 >= smooth.size:
            base[key] = float("nan")            # never settled inside the record
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
            for v in args.voltages:
                t, ia, ib = capture_step(smu, args.v_start, v, args.samples, args.nplc,
                                     chunk=args.chunk, log=log,
                                     limit_a=cfg.nvled_compliance,
                                     capacitance=args.capacitance,
                                     max_presettle_s=args.max_presettle_s)
                met = settling_metrics(t, np.abs(ib))
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
                              and abs(met["settled"]) >= 0.95 * cfg.nvled_compliance)
                if at_lim:
                    log(f"      at compliance ({cfg.nvled_compliance:g} A): V={v:+.2f} is "
                        f"not reachable at this limit, the node is current-limited")
                rows.append(dict(highc=highc, range_i=rng if rng is not None else "limit",
                                 compliance_a=cfg.nvled_compliance, nplc=args.nplc,
                                 at_compliance=at_lim,
                                 v_from=args.v_start, v_to=v, **met,
                                 j_settled=to_j(met["settled"], cfg),
                                 power_W=pw, power_lit_W=pw_lit, power_dark_W=pw_dark,
                                 power_sem=pw_sem, eqe_ext=eqe,
                                 power_sigma=abs(pw) / pw_sem if pw_sem else float("nan"),
                                 i_a_settled=float(np.median(ia_ok[int(0.8 * ia_ok.size):]))
                                 if ia_ok.size else float("nan"),
                                 record_s=float(t[-1]) if t.size else 0.0,
                                 smu_errors=err))
                log(f"    highc={highc!s:5} range={str(rng):>7} V={v:+.2f}  "
                    f"I={met['settled']*1e9:+10.1f} nA (J={to_j(met['settled'], cfg):+8.2f})  "
                    f"t1%={met['t_1pct']:.3f} t0.1%={met['t_01pct']:.3f}  "
                    f"P={pw*1e12:+9.1f}pW({abs(pw)/pw_sem if pw_sem else 0:4.0f}sig) "
                    f"EQE={100*eqe:6.3f}%  "
                    f"ripple={100*met['ripple']:.0f}%  over={met['n_overflow']}/{met['n']}  "
                    f"rec={float(t[-1]) if t.size else 0:.2f}s"
                    + ("  << NOT SETTLED" if not met["settled_ok"] else "")
                    + (f"  ERR: {err}" if err else ""))
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
            src = dataclasses.replace(cfg, current_source_range_i=cap_source_range(i_set))
            m.configure_current_source_b(smu, src)
            m.set_nplc(smu, max(args.nplc, 0.1))
            m.set_current_b(smu, -abs(i_set))        # sink: drives the node negative
            t0 = time.perf_counter()
            ts, vs = [], []
            while (dt := time.perf_counter() - t0) < args.cap_seconds:
                vb, _ = m.read_voltage_b(smu, n=2)
                ts.append(dt)
                vs.append(vb)
                # honour the stop only once the ramp is genuinely under way: a glitch
                # on the first read after the mode switch must not end it immediately
                if len(ts) >= 5 and abs(vb) > abs(args.cap_v_stop):
                    break                            # before turn-on or compliance
            m.set_current_b(smu, 0.0)
            tsa, vsa = np.asarray(ts), np.asarray(vs)
            slope, cap = ramp_capacitance(tsa, vsa, i_set, cfg.nvled_v_limit)
            rows.append(dict(lit=lit, i_set=i_set, park_v=park, n_samples=int(tsa.size),
                             ramp_s=float(tsa[-1]) if tsa.size else 0.0,
                             v_start=float(vsa[0]) if vsa.size else float("nan"),
                             v_end=float(vsa[-1]) if vsa.size else float("nan"),
                             dvdt=slope, capacitance_F=cap,
                             smu_errors=drain_errors(smu)))
            log(f"    lit={lit!s:5} I={i_set*1e9:>8.0f} nA  "
                f"V {vsa[0] if vsa.size else float('nan'):+.3f} -> "
                f"{vsa[-1] if vsa.size else float('nan'):+.3f} V in "
                f"{tsa[-1] if tsa.size else 0:.1f}s  dV/dt={slope*1e3:+8.2f} mV/s  "
                f"-> C={cap*1e6:8.2f} uF")
    m.set_current_b(smu, 0.0)
    smile.blank_frame()
    return rows


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


def exp_isrc(pm, smu, smile, pixel, cfg, args, log):
    """Current-source mode, held long enough to actually equilibrate."""
    rows = []
    smile.set_pixel(pixel[0], pixel[1], cfg.bitvals[0])
    m.configure_current_source_b(smu, cfg)
    m.set_nplc(smu, max(args.nplc, 0.1))
    for i_set in args.isrc_currents:
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
                                     max_presettle_s=args.max_presettle_s)
        met = settling_metrics(t, np.abs(ib))
        rows.append(dict(v_from=v_from, v_to=args.stepdir_target,
                         dv=args.stepdir_target - v_from, **met))
        log(f"    {v_from:+.2f} -> {args.stepdir_target:+.2f} V "
            f"(dV={args.stepdir_target-v_from:+.2f}): I={met['settled']*1e9:+8.1f} nA  "
            f"t1%={met['t_1pct']:.3f}s t0.1%={met['t_01pct']:.3f}s")
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


PER_PIXEL = {"step": exp_step, "cap": exp_cap, "isrc": exp_isrc,
             "stepdir": exp_stepdir, "decay": exp_decay}
ARRAY_WIDE = {"optfloor": exp_optfloor}
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


def build_parser():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--what", nargs="+", default=["step", "cap"],
                    choices=[*ALL, "all"], help="which experiments to run")
    ap.add_argument("--pixels", nargs="+", default=None, metavar="X,Y",
                    help="explicit pixels; otherwise --per-quadrant are found by scan")
    ap.add_argument("--per-quadrant", type=int, default=3,
                    help="alive pixels to characterise per quadrant (default 3)")
    ap.add_argument("--quadrants", nargs="+", default=list(QUADRANTS), choices=QUADRANTS)
    ap.add_argument("--quadrant-window", type=int, default=24,
                    help="edge of the scan window at each quadrant centre")
    ap.add_argument("--alive-nw", type=float, default=2.0,
                    help="turn-on power over dark (nW) for a pixel to count as alive")
    ap.add_argument("--alive-n-pm", type=int, default=8, help="PM reads for the liveness check")
    ap.add_argument("--alive-compliance", type=float, default=1e-3,
                    help="ch B limit used for the liveness check only. Kept healthy on "
                         "purpose: the limit is the charging current, so a low measurement "
                         "compliance makes every pixel read dead")
    ap.add_argument("--out", default=None, help="output root (default: config base_dir)")
    ap.add_argument("--simulate", action="store_true",
                    help="plumbing only - the sim rig has no capacitance model")
    ap.add_argument("--voltages", nargs="+", type=float, default=DEFAULT_VOLTAGES)
    ap.add_argument("--ranges", nargs="+", type=float, default=DEFAULT_RANGES)
    ap.add_argument("--v-start", type=float, default=-2.6, help="voltage to step FROM")
    ap.add_argument("--samples", type=int, default=6000,
                    help="buffer samples per step. record_s in the CSV says how much time "
                         "that bought; raise it until the record outlasts the settling")
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
    ap.add_argument("--stepdir-target", type=float, default=-0.85)
    ap.add_argument("--stepdir-from", nargs="+", type=float, default=[-0.8, -1.0, -1.3, -2.6])
    ap.add_argument("--decay-from", type=float, default=-1.3,
                    help="charge to this voltage, then let it decay through the LED")
    ap.add_argument("--decay-charge-s", type=float, default=3.0)
    ap.add_argument("--decay-seconds", type=float, default=120.0)
    ap.add_argument("--decay-period", type=float, default=0.5)
    ap.add_argument("--decay-n-v", type=int, default=8, help="voltage reads per decay sample")
    ap.add_argument("--decay-n-pm", type=int, default=16, help="PM reads per decay sample")
    ap.add_argument("--decay-source-range", type=float, default=1e-7,
                    help="ch B source current range while holding 0 A. The finest range "
                         "minimises the regulation error that would add to the decay "
                         "current; the blanked run cancels what is left")
    ap.add_argument("--cap-currents", nargs="+", type=float, default=[1e-6, 1e-5, 1e-4],
                    help="known currents for the C = I/(dV/dt) ramp (A)")
    ap.add_argument("--cap-park", type=float, default=-0.40,
                    help="bias to park at before the ramp. Below turn-on, so the LED's "
                         "own current is a small fraction of the setpoint")
    ap.add_argument("--cap-v-stop", type=float, default=-0.65,
                    help="stop the ramp here, before the LED turns on and starts taking "
                         "a share of the setpoint")
    ap.add_argument("--cap-seconds", type=float, default=5.0, help="max ramp duration")
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
    ap.add_argument("--max-presettle-s", type=float, default=60.0,
                    help="cap on the charge-derived pre-settle before each capture. The "
                         "charge time is C*dV/limit, which at a 1 uA limit is minutes per "
                         "point; exceeding the cap is logged as a warning")
    ap.add_argument("--capacitance", type=float, default=50e-6,
                    help="C used to turn dV/dt into a current; refine it with --what cap")
    ap.add_argument("--optfloor-n", nargs="+", type=int, default=[4, 8, 32, 128, 512])
    ap.add_argument("--optfloor-reps", type=int, default=12,
                    help="repeats per N_PM, for the repeatability spread")
    ap.add_argument("--responsivity-w-per-a", type=float, default=22e-9 / 25e-6,
                    help="W of optical power per A of LED current, for the min-J estimate")
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

    root = Path(args.out) if args.out else Path(cfg.base_dir)
    out_dir = root / f"{datetime.datetime.now():%y%m%d-%H%M}_CAPACITANCE"
    out_dir.mkdir(parents=True, exist_ok=True)
    log(f"Output: {out_dir}")
    (out_dir / "run_config.json").write_text(
        json.dumps({"config": cfg.to_dict(),
                    "args": {k: v for k, v in sorted(vars(args).items())}},
                   indent=2, default=str), encoding="utf-8")
    if args.simulate:
        log("SIMULATION - plumbing only, the sim rig has no capacitance model.")

    if args.samples > 50000:
        log(f"WARNING: --samples {args.samples} may exceed the SMU buffer; "
            f"check record_s and n in the CSV against what you asked for")
    # Measured 261013: a voltage-mode current reading without High-C is a range
    # artifact below ~10 uA, not a noisy measurement - it lands at 42-58% of
    # whatever range is selected regardless of bias. step and stepdir read current
    # in voltage mode, so they need it. cap does not: it measures a voltage slope.
    needs_highc = [n for n in what if n in ("step", "stepdir")]
    if needs_highc and not (args.highc or args.highc_both):
        log(f"WARNING: {', '.join(needs_highc)} measure current in voltage mode without "
            f"High-C. Below ~10 uA that reads ~50% of the selected range regardless of "
            f"bias. Pass --highc (and --compliance, e.g. 1e-5) unless you are "
            f"deliberately reproducing the artifact.")
    pm, smu, smile = m.connect(cfg, log, simulate=args.simulate)
    collected: dict[str, list[dict]] = {name: [] for name in what}
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
    finally:
        m.shutdown(pm, smu, smile, cfg)
        # Writing in the finally is the point: previously an exception here skipped
        # the writes entirely and a run that died partway lost every completed row.
        log("=" * 56)
        wrote = [p for p in (write_csv(out_dir, n, r) for n, r in collected.items()) if p]
        for p in wrote:
            log(f"wrote {p}")
        if not wrote:
            log("no data written")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
