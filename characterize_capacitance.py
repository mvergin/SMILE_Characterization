"""Characterise the array's ~50 uF capacitance and what it does to a current reading.

The array carries a large capacitor in parallel with the LED. Moving the cathode
bias across the rising-side span (-0.8 -> -1.3 V) moves ``C*dV ~ 25 uC`` of
charge. At LED currents (1-10 uA) that is seconds per pixel; at the SMU's
compliance it is microseconds. Which of those you get decides whether a low-J
current reading is right, and it explains the direction hysteresis, the useless
low ranges, the oscillation seen in the buffer, and probably the "slow parasitic
current" the warm-up diagnostic found.

This script measures that instead of reasoning about it. Four experiments:

  step        After a voltage step, capture I(t) at the lowest NPLC into the SMU
              buffer with instrument timestamps. Per voltage, per measure range,
              High-C on and off. Reports the settled value, the time to reach 1%
              and 0.1% of it, and an oscillation metric.

  cap         With the pixel blanked (no LED current), step the voltage and
              integrate the current: C = integral(I dt) / dV. Gives the actual
              capacitance and the dielectric-absorption tail.

  isrc        The decisive current-source test. Channel B sinks a small current
              and holds for minutes while chA current, chB current and chB
              voltage are logged. If chA converges towards chB's setpoint as the
              voltage settles, there is no bypass path and the earlier
              "current-source mode does not work" conclusion was the capacitor.
              If chA sits at tens of uA with dV/dt already ~0, the bypass is
              real.

  stepdir     The same voltage reached from a small step and from a large step.
              A parasitic conduction path depends on time since turn-on; a
              capacitor depends on how much charge had to move. If the settling
              time scales with the step size, it is the capacitor.

Usage:
    python characterize_capacitance.py --pixel 54 33 --what step cap
    python characterize_capacitance.py --pixel 54 33 --what isrc --isrc-seconds 300
    python characterize_capacitance.py --what all --out D:/runs

Writes one CSV per experiment plus a printed summary. It fits and plots nothing;
point SMILE_IQE_Analysis at the CSVs for that.

NOTE: ``--simulate`` exercises the plumbing only. The built-in sim rig has no
capacitance model, so it will report instant settling. The physics needs the bench.
"""
from __future__ import annotations

import argparse
import csv
import datetime
import time
from pathlib import Path

import numpy as np

import measure_iqe as m
from measure_iqe import IQEConfig

# Voltages worth probing: below turn-on, the rising side that pins peak IQE, the
# effort-split knee, and the droop. Override with --voltages.
DEFAULT_VOLTAGES = [-0.8, -0.9, -1.0, -1.1, -1.3, -1.8, -2.6]
# Ranges to compare. The 10 mA range is what the clean runs used; the fine ones are
# the ones that read low J "not just noisily but WRONG" before.
DEFAULT_RANGES = [1e-3, 1e-4, 1e-5, 1e-6]


# ----------------------------------------------------------------------
#  capture primitive
# ----------------------------------------------------------------------

def capture_step(smu, v_from, v_to, n_samples, nplc=0.001, settle_first_s=2.0):
    """Step ch B from v_from to v_to, then burst-capture I(t) with timestamps.

    Returns (t, i_a, i_b) as numpy arrays, t in seconds from the first sample.
    The instrument timestamps are used, not host time, so the sample spacing is
    the SMU's own and is not polluted by USB latency.
    """
    m.set_nplc(smu, nplc)
    smu.setup_buffers(timestamps=True)
    smu.set_voltage("b", float(v_from))
    time.sleep(settle_first_s)                 # start from a known, settled state
    smu.clear_buffers_only()
    smu.set_voltage("b", float(v_to))          # the step
    if not hasattr(smu, "inst"):               # sim rig: no TSP, no instrument clock
        t0 = time.perf_counter()
        smu.measure_burst_fire(int(n_samples))
        smu.measure_burst_join()
        ia, ib = smu.read_buffers(int(n_samples))
        t = np.linspace(0.0, time.perf_counter() - t0, len(ib))
        return t, np.asarray(ia, float), np.asarray(ib, float)
    smu.measure_burst(int(n_samples))          # fills as fast as the ADC allows
    ia, ib, ta, _tb = smu.read_buffer_with_timestamps()
    if not ib:
        return np.array([]), np.array([]), np.array([])
    t = np.asarray(ta, float) - ta[0]
    return t, np.asarray(ia, float), np.asarray(ib, float)


def settling_metrics(t, i):
    """Settled value plus how long it took to get there.

    The settled value is the median of the last 20% of the record. "Time to x%"
    is the last time the trace is still outside x% of it, which is the honest
    definition: a trace that re-enters the band and leaves again has not settled.
    """
    if t.size < 10:
        return dict(settled=float("nan"), t_1pct=float("nan"), t_01pct=float("nan"),
                    overshoot=float("nan"), ripple=float("nan"), n=int(t.size))
    tail = i[int(0.8 * i.size):]
    settled = float(np.median(tail))
    out = dict(settled=settled, n=int(t.size),
               ripple=float(np.std(tail) / abs(settled)) if settled else float("nan"),
               overshoot=float(np.max(np.abs(i)) / abs(settled)) if settled else float("nan"))
    for frac, key in ((0.01, "t_1pct"), (0.001, "t_01pct")):
        bad = np.nonzero(np.abs(i - settled) > frac * abs(settled))[0]
        out[key] = float(t[bad[-1]]) if bad.size and bad[-1] + 1 < t.size else 0.0
    return out


# ----------------------------------------------------------------------
#  experiments
# ----------------------------------------------------------------------

def exp_step(smu, smile, pixel, cfg, args, log):
    """I(t) after a voltage step, per voltage x range x High-C state."""
    rows = []
    smile.set_pixel(pixel[0], pixel[1], cfg.bitvals[0])
    for highc in ([False, True] if args.highc_both else [args.highc]):
        m.configure_voltage_source_b(smu, IQEConfig(**{**cfg.to_dict(), "highc_b": highc}), log)
        # In High-C the limit governs the range, so sweeping ranges is meaningless there.
        ranges = [None] if highc else args.ranges
        for rng in ranges:
            if rng is not None:
                m.set_measure_range_b(smu, rng)
                time.sleep(0.05)
            for v in args.voltages:
                t, ia, ib = capture_step(smu, args.v_start, v, args.samples, args.nplc)
                met = settling_metrics(t, np.abs(ib))
                rows.append(dict(highc=highc, range_i=rng if rng is not None else "limit",
                                 v_from=args.v_start, v_to=v, **met,
                                 i_a_settled=float(np.median(ia[int(0.8 * ia.size):]))
                                 if ia.size else float("nan"),
                                 record_s=float(t[-1]) if t.size else 0.0))
                log(f"  highc={highc!s:5} range={str(rng):>7} V={v:+.2f}  "
                    f"I={met['settled']*1e6:+9.3f} uA  t1%={met['t_1pct']:.3f}s  "
                    f"t0.1%={met['t_01pct']:.3f}s  ripple={100*met['ripple']:.1f}%")
    smile.blank_frame()
    return rows


def exp_cap(smu, smile, pixel, cfg, args, log):
    """C = integral(I dt) / dV with the LED dark, so only the capacitor conducts."""
    rows = []
    smile.blank_frame()                        # no LED current: all charge is the cap
    m.configure_voltage_source_b(smu, cfg, log)
    m.set_measure_range_b(smu, args.ranges[0])
    time.sleep(0.05)
    for v in args.voltages:
        dv = v - args.v_start
        if abs(dv) < 1e-6:
            continue
        t, _ia, ib = capture_step(smu, args.v_start, v, args.samples, args.nplc)
        if t.size < 10:
            continue
        # subtract the settled (leakage) level, then integrate what is left: that is
        # the charge the capacitor took, so C = Q / dV.
        base = float(np.median(ib[int(0.8 * ib.size):]))
        q = float(np.trapezoid(ib - base, t))
        rows.append(dict(v_from=args.v_start, v_to=v, dv=dv, charge_C=q,
                         capacitance_F=q / dv if dv else float("nan"),
                         leakage_A=base, record_s=float(t[-1])))
        log(f"  dV={dv:+.2f} V  Q={q*1e6:+.3f} uC  ->  C={q/dv*1e6:+.1f} uF   "
            f"(settled leak {base*1e9:+.2f} nA)")
    return rows


def exp_isrc(smu, smile, pixel, cfg, args, log):
    """Current-source mode, held long enough to actually equilibrate.

    The earlier rejection of current-source mode ("the anode bypasses ch B") was
    measured with a 15 ms settle. At 0.3 uA into 50 uF, 15 ms moves the node by
    0.09 mV — the operating point never moved. This holds for minutes and logs the
    approach, so the bypass hypothesis and the capacitor hypothesis separate.
    """
    rows = []
    smile.set_pixel(pixel[0], pixel[1], cfg.bitvals[0])
    m.configure_current_source_b(smu, cfg)
    m.set_nplc(smu, max(args.nplc, 0.1))
    for i_set in args.isrc_currents:
        m.set_current_b(smu, -abs(i_set))      # ch B sinks the LED current
        t0 = time.perf_counter()
        last_v = None
        while (dt := time.perf_counter() - t0) < args.isrc_seconds:
            if hasattr(smu, "measure_instant"):
                ia, ib = smu.measure_instant()
            else:                               # sim rig
                a, b = smu.read_buffers(1)
                ia, ib = (a[0] if a else float("nan")), (b[0] if b else float("nan"))
            vb, _ = m.read_voltage_b(smu, n=2)
            dvdt = ((vb - last_v[1]) / (dt - last_v[0])) if last_v else float("nan")
            rows.append(dict(i_set=i_set, t_s=dt, i_a=ia, i_b=ib, v_b=vb, dvdt=dvdt,
                             leak_ratio=abs(ia / ib) if ib else float("nan")))
            last_v = (dt, vb)
            time.sleep(args.isrc_period)
        tail = [r for r in rows if r["i_set"] == i_set][-5:]
        log(f"  I_set={i_set*1e6:.3f} uA after {args.isrc_seconds:.0f}s: "
            f"chA={np.mean([r['i_a'] for r in tail])*1e6:+.3f} uA  "
            f"chB={np.mean([r['i_b'] for r in tail])*1e6:+.3f} uA  "
            f"V={np.mean([r['v_b'] for r in tail]):+.4f} V  "
            f"dV/dt={np.nanmean([r['dvdt'] for r in tail])*1e3:+.3f} mV/s")
    m.set_current_b(smu, 0.0)
    smile.blank_frame()
    return rows


def exp_stepdir(smu, smile, pixel, cfg, args, log):
    """Reach one voltage from a small step and from a large step.

    Parasitic conduction depends on time since turn-on. A capacitor depends on the
    charge that had to move. If t_settle scales with |dV|, it is the capacitor.
    """
    rows = []
    smile.set_pixel(pixel[0], pixel[1], cfg.bitvals[0])
    m.configure_voltage_source_b(smu, cfg, log)
    m.set_measure_range_b(smu, args.ranges[0])
    time.sleep(0.05)
    target = args.stepdir_target
    for v_from in args.stepdir_from:
        t, _ia, ib = capture_step(smu, v_from, target, args.samples, args.nplc)
        met = settling_metrics(t, np.abs(ib))
        rows.append(dict(v_from=v_from, v_to=target, dv=target - v_from, **met))
        log(f"  {v_from:+.2f} -> {target:+.2f} V (dV={target-v_from:+.2f}): "
            f"I={met['settled']*1e6:+.3f} uA  t1%={met['t_1pct']:.3f}s  "
            f"t0.1%={met['t_01pct']:.3f}s")
    smile.blank_frame()
    return rows


EXPERIMENTS = {"step": exp_step, "cap": exp_cap, "isrc": exp_isrc, "stepdir": exp_stepdir}


# ----------------------------------------------------------------------

def write_csv(out_dir: Path, name: str, rows: list[dict]) -> Path | None:
    if not rows:
        return None
    cols = list(rows[0])
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
                    choices=[*EXPERIMENTS, "all"], help="which experiments to run")
    ap.add_argument("--pixel", nargs=2, type=int, default=[128, 128], metavar=("X", "Y"))
    ap.add_argument("--out", default=None, help="output root (default: config base_dir)")
    ap.add_argument("--simulate", action="store_true",
                    help="plumbing only — the sim rig has no capacitance model")
    ap.add_argument("--voltages", nargs="+", type=float, default=DEFAULT_VOLTAGES)
    ap.add_argument("--ranges", nargs="+", type=float, default=DEFAULT_RANGES)
    ap.add_argument("--v-start", type=float, default=-2.6,
                    help="voltage to step FROM (default: the conditioned high-bias point)")
    ap.add_argument("--samples", type=int, default=2000, help="buffer samples per step")
    ap.add_argument("--nplc", type=float, default=0.001, help="integration time for I(t)")
    ap.add_argument("--highc", action="store_true", help="run with High-C enabled")
    ap.add_argument("--highc-both", action="store_true",
                    help="run the step experiment with High-C both off and on")
    ap.add_argument("--isrc-currents", nargs="+", type=float, default=[3e-7, 1e-6, 1e-5],
                    help="setpoints for the current-source test (A)")
    ap.add_argument("--isrc-seconds", type=float, default=120.0,
                    help="hold time per setpoint — needs to be minutes, not ms")
    ap.add_argument("--isrc-period", type=float, default=0.5, help="log period (s)")
    ap.add_argument("--stepdir-target", type=float, default=-0.85,
                    help="the voltage to arrive at, from different distances")
    ap.add_argument("--stepdir-from", nargs="+", type=float, default=[-0.8, -1.0, -1.3, -2.6])
    return ap


def main(argv=None):
    args = build_parser().parse_args(argv)
    what = list(EXPERIMENTS) if "all" in args.what else args.what
    log = m.make_logger()
    cfg = IQEConfig(highc_b=args.highc)

    root = Path(args.out) if args.out else Path(cfg.base_dir)
    out_dir = root / f"{datetime.datetime.now():%y%m%d-%H%M}_CAPACITANCE"
    out_dir.mkdir(parents=True, exist_ok=True)
    log(f"Output: {out_dir}")
    if args.simulate:
        log("SIMULATION — plumbing only, the sim rig has no capacitance model.")

    pm, smu, smile = m.connect(cfg, log, simulate=args.simulate)
    written = []
    try:
        m.configure_common(pm, smu, cfg, log)
        smu.set_voltage("a", cfg.vled_voltage)
        smu.enable_output("a", True)
        smu.enable_output("b", True)
        time.sleep(0.05)
        for name in what:
            log("=" * 56)
            log(f"{name.upper()}")
            rows = EXPERIMENTS[name](smu, smile, tuple(args.pixel), cfg, args, log)
            path = write_csv(out_dir, name, rows)
            if path:
                written.append(path)
                log(f"  -> {path.name} ({len(rows)} rows)")
    finally:
        m.shutdown(pm, smu, smile, cfg)
    log("=" * 56)
    for p in written:
        log(f"wrote {p}")
    if not written:
        log("no data written")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
