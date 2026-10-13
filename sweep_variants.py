"""Measurement-variant comparison harness for the SMILE IQE setup.

Purpose: stop guessing which acquisition settings give a clean, *settled* NVLED
current and just try them on the bench, side by side, on the SAME few pixels.

It does two independent things:

  1. SETTLING PROBE (answers "do the low current ranges actually settle?").
     At a low-J bias and a high-J bias, for every standard measure range
     (1 µA … 10 mA), with High-C OFF and ON, it logs the ch B current vs time
     and reports: final value, time to settle, residual scatter, and drift.
     This is the direct test of the thing that made you stick to the 10 mA range.

  2. VARIANT SWEEPS. Runs a full per-pixel fine sweep for each named variant
     (fixed 10 mA vs adaptive per-point range; low→high vs high→low; warm-up
     on/off; High-C on ch B) into its own folder, then prints, per variant,
     whether the NVLED current spans the full range and whether the light tracks
     NVLED (corr of log P vs log I) — i.e. whether that variant gives a physical
     current you can compute IQE from. Each folder is a normal run you can pass
     straight to analyze_array.py (in the SMILE_IQE_Analysis repo).

Reuses measure_iqe.py's primitives by overriding its module globals per variant.

    python sweep_variants.py                 # real hardware
    python sweep_variants.py --simulate       # dry-run the whole flow
    python sweep_variants.py --settle-only     # just the settling probe
    python sweep_variants.py --no-settle        # just the variant sweeps
"""

from __future__ import annotations

import argparse
import datetime
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

import measure_iqe as m


# =====================================================================
#  What to try — edit freely
# =====================================================================

N_PIXELS = 3                      # how many alive pixels to run every variant on
VARIANT_N_FINE = 45               # fine points per sweep (fewer than a real run: this is
                                  # a comparison of measurement quality, not final IQE)

# Keys overridden per variant (everything not listed keeps measure_iqe's default).
_OVERRIDE_KEYS = ["PERPOINT_RANGE", "NVLED_RANGE_I", "SWEEP_DIRECTION", "WARMUP_MS", "HIGHC_B"]

VARIANTS = [
    dict(name="fixed10mA_low2high",
         PERPOINT_RANGE=False, NVLED_RANGE_I=10e-3, SWEEP_DIRECTION="low_to_high",
         WARMUP_MS=0.0, HIGHC_B=False),
    dict(name="fixed10mA_high2low",
         PERPOINT_RANGE=False, NVLED_RANGE_I=10e-3, SWEEP_DIRECTION="high_to_low",
         WARMUP_MS=0.0, HIGHC_B=False),
    dict(name="adaptive_low2high",
         PERPOINT_RANGE=True, SWEEP_DIRECTION="low_to_high", WARMUP_MS=0.0, HIGHC_B=False),
    dict(name="adaptive_high2low",
         PERPOINT_RANGE=True, SWEEP_DIRECTION="high_to_low", WARMUP_MS=0.0, HIGHC_B=False),
    dict(name="adaptive_low2high_warmup",
         PERPOINT_RANGE=True, SWEEP_DIRECTION="low_to_high", WARMUP_MS=1500.0, HIGHC_B=False),
    # High-C on ch B is disabled: the bench returned Keithley error 5069 (High-C locks
    # the current measurement to autorange, so manual per-point ranging is rejected) and
    # the readings it did give were noisy and never settled. Re-enable only if you also
    # set PERPOINT_RANGE=False — but expect autorange (which is broken on the burst read).
    # dict(name="highcB_adaptive_low2high",
    #      PERPOINT_RANGE=False, SWEEP_DIRECTION="low_to_high", WARMUP_MS=0.0, HIGHC_B=True),
]

# ── Settling probe ──────────────────────────────────────────────────
# Uses the sweep's REAL integration (NPLC + per-timepoint averaging), not a single short
# read — otherwise the measurement is dominated by aperture noise (a fixed fraction of the
# range) and everything at low current looks like it "never settles". With clean reads,
# drift/scatter over the window reflect actual settling.
SETTLE_BIASES = [(-0.85, "lowJ"), (-2.6, "highJ")]   # (NVLED_V, label)
SETTLE_RANGES = [1e-6, 1e-5, 1e-4, 1e-3, 1e-2]       # ch B measure ranges to compare
SETTLE_HIGHC = [False]                               # High-C is a dead end (err 5069) — off
SETTLE_SECONDS = 8.0                                 # capture duration per cell
SETTLE_NPLC = 1.0                                    # realistic integration (match the sweep)
SETTLE_N_SMU = 8                                     # samples averaged per timepoint (median)
_OVERFLOW = 1e36                                     # 2602B over-range sentinel (~9.91e37)


# =====================================================================
#  Helpers
# =====================================================================

def snapshot(keys):
    missing = [k for k in keys if not hasattr(m, k)]
    if missing:
        raise SystemExit(
            f"measure_iqe.py is out of date — missing {missing}.\n"
            f"  importing: {getattr(m, '__file__', '?')}\n"
            f"This harness needs the updated measure_iqe.py (HIGHC_B, PERPOINT_RANGE, "
            f"per-point ranging). Update it (or remove any older copy earlier on sys.path).")
    return {k: getattr(m, k) for k in keys}


def apply_overrides(base, ov):
    for k in _OVERRIDE_KEYS:                          # reset to base, then apply variant
        setattr(m, k, base[k])
    for k, v in ov.items():
        if k in _OVERRIDE_KEYS:
            setattr(m, k, v)
    m.N_FINE = VARIANT_N_FINE


def settle_metrics(t, i):
    """(I_final, t_settle_s, scatter_%, drift_%) for one current-vs-time capture.
    Over-range (2602B ~9.91e37) points are dropped; if the cell is mostly over-range it
    returns inf I_final so the summary shows it as unusable on that range."""
    t = np.asarray(t, float); i = np.asarray(i, float)
    ok = np.isfinite(i) & (np.abs(i) < _OVERFLOW)
    if ok.sum() < 0.5 * max(i.size, 1):
        return float("inf"), float("nan"), float("nan"), float("nan")   # over-range
    t, i = t[ok], i[ok]
    if i.size < 8:
        return float("nan"), float("nan"), float("nan"), float("nan")
    n4 = max(2, i.size // 4)
    last, first = i[-n4:], i[:n4]
    i_fin = float(np.median(last))
    denom = abs(i_fin) if i_fin else float("nan")
    scatter = 100 * float(np.std(last)) / denom if denom else float("nan")
    drift = 100 * (float(np.median(last)) - float(np.median(first))) / denom if denom else float("nan")
    # time to settle: last instant the trace is still >10% off the final value
    off = np.abs(i - i_fin) / denom > 0.10 if denom else np.zeros(i.size, bool)
    t_settle = float(t[np.where(off)[0][-1]]) if off.any() else 0.0
    return i_fin, t_settle, scatter, drift


def run_settling_probe(pm, smu, smile, pixel, out_dir, log):
    x, y = pixel
    log("=" * 60)
    log(f"SETTLING PROBE on pixel ({x},{y}) — {len(SETTLE_BIASES)} biases × "
        f"{len(SETTLE_RANGES)} ranges × High-C {SETTLE_HIGHC}, {SETTLE_SECONDS:.0f}s each")
    smile.set_pixel(x, y, m.FASTSCAN_BITVAL)
    rows, summary = [], []
    for highc in SETTLE_HIGHC:
        m.HIGHC_B = highc
        m.configure_voltage_source_b(smu)             # applies / clears High-C on ch B
        m.set_nplc(smu, SETTLE_NPLC)
        for bias, label in SETTLE_BIASES:
            for rng in SETTLE_RANGES:
                try:
                    m.set_measure_range_b(smu, rng)
                except Exception as e:
                    log(f"  range {rng:g} rejected ({e})"); continue
                smu.set_voltage("b", float(bias))
                t0 = time.perf_counter()
                ts, iv_t = [], []
                while time.perf_counter() - t0 < SETTLE_SECONDS:
                    _, _, _, _, inv, _, _, _ = m.acquire_point(pm, smu, SETTLE_N_SMU, 1)
                    tt = time.perf_counter() - t0
                    ts.append(tt); iv_t.append(inv)
                    rows.append((int(highc), label, bias, rng, round(tt, 4), inv))
                i_fin, t_set, sc, dr = settle_metrics(ts, iv_t)
                summary.append((highc, label, bias, rng, len(ts), i_fin, t_set, sc, dr))
                log(f"  hiC={int(highc)} {label:5s} V={bias:+.2f} range={rng:.0e}: "
                    f"I_fin={i_fin: .3e}  t_settle={t_set:4.2f}s  scatter={sc:5.1f}%  drift={dr:+6.1f}%")
    m.HIGHC_B = False
    m.configure_voltage_source_b(smu)                 # leave ch B in the plain state
    # write raw + summary
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "settling_probe.csv", "w") as f:
        f.write("highc,bias_label,bias_v,range_i,t_s,I_NVLED\n")
        for r in rows:
            f.write(",".join(repr(v) for v in r) + "\n")
    with open(out_dir / "settling_summary.csv", "w") as f:
        f.write("highc,bias_label,bias_v,range_i,n,I_final,t_settle_s,scatter_pct,drift_pct\n")
        for r in summary:
            f.write(",".join(repr(v) for v in r) + "\n")
    log(f"  saved {out_dir/'settling_probe.csv'} and settling_summary.csv")
    return summary


def variant_quality(sweep_csv):
    """Read a variant's aggregated sweep and report whether NVLED is physical:
    its span, and how well the light tracks NVLED vs VLED (log-log correlation)."""
    import pandas as pd
    df = pd.read_csv(sweep_csv)
    P = df["PM400_POWER_MEAN"].astype(float).values
    Inv = np.abs(df["NVLED_CURR_MEAN"].astype(float).values)
    Iv = np.abs(df["VLED_CURR_MEAN"].astype(float).values)
    msk = (P > 2e-11) & (Inv > 1e-9) & (Iv > 1e-9)
    def corr(a, b):
        return float(np.corrcoef(np.log(a[msk]), np.log(b[msk]))[0, 1]) if msk.sum() > 3 else float("nan")
    return dict(n=len(df), nvled_min=float(Inv.min()), nvled_max=float(Inv.max()),
                vled_max=float(Iv.max()), p_max=float(P.max()),
                corr_logP_logNVLED=corr(P, Inv), corr_logP_logVLED=corr(P, Iv))


# =====================================================================
#  Main
# =====================================================================

def main():
    ap = argparse.ArgumentParser(description="IQE measurement-variant comparison harness.")
    ap.add_argument("--simulate", action="store_true")
    ap.add_argument("--out", default=None)
    ap.add_argument("--pixels", type=int, default=N_PIXELS)
    ap.add_argument("--settle-only", action="store_true", help="only the settling probe")
    ap.add_argument("--no-settle", action="store_true", help="skip the settling probe")
    args = ap.parse_args()

    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

    def log(msg):
        try:
            print(f"[{datetime.datetime.now():%H:%M:%S}] {msg}")
        except UnicodeEncodeError:
            print(f"[{datetime.datetime.now():%H:%M:%S}] "
                  + str(msg).encode("ascii", "replace").decode("ascii"))

    base_dir = Path(args.out) if args.out else Path(m.BASE_DIR)
    x1, y1, x2, y2 = m.parse_roi(m.ROI)
    stamp = datetime.datetime.now().strftime("%y%m%d-%H%M")
    root = base_dir / f"{stamp}_{m.SAMPLE_NAME}_VARIANTS"
    root.mkdir(parents=True, exist_ok=True)
    log(f"Variant harness output: {root}")

    pm, smu, smile = m.connect_sim(log) if args.simulate else m.connect_real(log)
    m.configure_common(pm, smu, log)
    smu.set_voltage("a", m.VLED_VOLTAGE)
    smu.set_voltage("b", m.NVLED_V_FASTSCAN)
    smu.enable_output("a", True)
    smu.enable_output("b", True)
    time.sleep(0.05)

    base = snapshot(_OVERRIDE_KEYS)
    fs_cfg = dict(sample_name=m.SAMPLE_NAME, roi=[x1, y1, x2, y2], purpose="variant_harness")

    try:
        selected = m.run_fast_scan(pm, smu, smile, x1, y1, x2, y2, root, fs_cfg, log)
        pixels = [(r["X"], r["Y"]) for r in selected[:max(1, args.pixels)]]
        if not pixels:
            log("No alive pixels — nothing to do."); return
        log(f"Using pixels: {pixels}")

        settle_sum = None
        if not args.no_settle:
            settle_sum = run_settling_probe(pm, smu, smile, pixels[0], root / "settling", log)
        if args.settle_only:
            log("settle-only: done."); return

        results = []
        for ov in VARIANTS:
            name = ov["name"]
            apply_overrides(base, ov)
            vdir = root / name
            log("=" * 60)
            log(f"VARIANT {name}: PERPOINT_RANGE={m.PERPOINT_RANGE} range={m.NVLED_RANGE_I:g} "
                f"dir={m.SWEEP_DIRECTION} warmup={m.WARMUP_MS:g} highc={m.HIGHC_B}")
            cfg = dict(variant=name, sample_name=m.SAMPLE_NAME, roi=[x1, y1, x2, y2],
                       perpoint_range=m.PERPOINT_RANGE, nvled_range_i=m.NVLED_RANGE_I,
                       sweep_direction=m.SWEEP_DIRECTION, warmup_ms=m.WARMUP_MS,
                       highc_b=m.HIGHC_B, n_fine=m.N_FINE,
                       timestamp=datetime.datetime.now().isoformat())
            sel = [dict(X=x, Y=y) for x, y in pixels]
            try:
                n = m.run_sweep_voltage(pm, smu, smile, sel, vdir, cfg, log)
                csv = vdir / "NVLEDSweep" / "Sweep" / "aggregated_bitval=15_mean_std.csv"
                q = variant_quality(csv) if csv.exists() else {}
                results.append((name, q))
                if q:
                    log(f"  -> NVLED {q['nvled_min']:.2e}…{q['nvled_max']:.2e} A | "
                        f"VLED max {q['vled_max']:.2e} | corr(logP,logNVLED)={q['corr_logP_logNVLED']:+.3f} "
                        f"corr(logP,logVLED)={q['corr_logP_logVLED']:+.3f}")
            except Exception as e:
                log(f"  variant {name} FAILED: {e}")
                results.append((name, {"error": str(e)}))

        write_summary(root, settle_sum, results, log)
    finally:
        for ch in ("a", "b"):
            try: smu.enable_output(ch, False)
            except Exception: pass
        try: smile.blank_frame()
        except Exception: pass
        for dev in (pm, smu, smile):
            try: dev.close()
            except Exception: pass


def write_summary(root, settle_sum, results, log):
    lines = ["# Variant harness summary", ""]
    if settle_sum:
        lines += ["## Settling probe (does the range settle?)", "",
                  "| HighC | bias | range | I_final (A) | t_settle (s) | scatter % | drift % |",
                  "|---|---|---|---|---|---|---|"]
        for highc, label, bias, rng, n, i_fin, t_set, sc, dr in settle_sum:
            lines.append(f"| {int(highc)} | {label} ({bias:+.2f}V) | {rng:.0e} | "
                         f"{i_fin:.3e} | {t_set:.2f} | {sc:.1f} | {dr:+.1f} |")
        lines += ["", "Read: a usable range has small t_settle, low scatter, and ~0 drift. "
                  "The point that determines peak IQE is the lowJ bias on a fine range.", ""]
    lines += ["## Variant sweeps (is NVLED a physical LED current?)", "",
              "| variant | NVLED min..max (A) | VLED max (A) | corr(logP,logNVLED) | corr(logP,logVLED) |",
              "|---|---|---|---|---|"]
    for name, q in results:
        if not q or "error" in q:
            lines.append(f"| {name} | — | — | {q.get('error','n/a') if q else 'n/a'} | |")
            continue
        lines.append(f"| {name} | {q['nvled_min']:.2e}..{q['nvled_max']:.2e} | {q['vled_max']:.2e} | "
                     f"{q['corr_logP_logNVLED']:+.3f} | {q['corr_logP_logVLED']:+.3f} |")
    lines += ["", "Read: NVLED should span ~2 decades up to ~10^-4 A and corr(logP,logNVLED) "
              "should be ≈ corr(logP,logVLED) ≈ 1. If NVLED is compressed (small max) or its "
              "correlation is much worse than VLED's, that variant misreads the current.", "",
              "Analyze any variant (SMILE_IQE_Analysis): "
              "`python analyze_array.py \"<variant folder>\"`"]
    (root / "SUMMARY.md").write_text("\n".join(lines), encoding="utf-8")
    log("=" * 60)
    log(f"Wrote {root/'SUMMARY.md'}")


if __name__ == "__main__":
    main()
