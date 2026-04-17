"""Experimental hardware-paced NVLED sweep mode.

One PM400 1 s array capture per pixel; inside that capture a TSP script
on the Keithley walks the full voltage list and, per voltage, rapid-fires
K chA + chB measurement pairs. All three buffers (chA current, chB
current, chB voltage) come back with per-sample instrument timestamps,
which we align to the PM400's perf_counter frame to splice per-voltage
averages and save both waveforms.

Timing budget (see :func:`compute_sweep_timing`):

  per_voltage = 0.8 s / N

If the operator leaves ``SWEEP_NVLED_SETTLE_MS > 0`` (default "auto"
path), we allocate ``settle_frac`` (60 %) of per_voltage to source
settling and the remaining 40 % to the K rapid-fire measurement burst.
If the operator sets ``SWEEP_NVLED_SETTLE_MS == 0`` ("zero-settle"
path), we use the entire per_voltage for measurement and trim the first
10 % of both K and PM samples per voltage to discard the settling
transient.

Output format matches the existing FastScan / Full Transient modes:
  * Overview CSV — one PM400 / VLED / NVLED row per voltage step with
    mean + std, plus an ``NVLED_V_MEAS`` row carrying the measured ch B
    voltage and the per-voltage sample count.
  * Secondary storage — one per-pixel transient file with the PM400
    waveform and the full N·K chA/chB current + chB voltage waveforms.
"""

from __future__ import annotations

import time

import numpy as np

from .context import MeasurementContext
from .modes import MeasurementMode


# Time available to the scripted sweep inside the 1 s PM400 buffer.
# 200 ms of the 1 s PM400 window is reserved as margin (arm jitter, K
# loop overhead, instrument clock skew).
SWEEP_BUDGET_S = 0.8

# Auto-allocated settle fraction of per-voltage time. The remainder
# (1 - settle_frac) is spent rapid-firing K measurements.
DEFAULT_SETTLE_FRAC = 0.6

# Trim applied to the per-voltage K-sample and PM400 windows when
# running zero-settle (lets the leading edge's settling transient be
# discarded from the mean/std).
ZERO_SETTLE_TRIM_PCT = 10.0


def compute_sweep_timing(
    n_voltages,
    user_settle_ms,
    nplc,
    line_freq_hz=50.0,
    budget_s=SWEEP_BUDGET_S,
    settle_frac=DEFAULT_SETTLE_FRAC,
):
    """Return (settle_s, measure_s, K, trim_pct, per_voltage_s, total_s).

    Two paths:
      * ``user_settle_ms == 0`` → zero-settle: settle=0, full per_voltage
        used for measurement, trim 10 % of the leading edge post-hoc.
      * ``user_settle_ms > 0``  → auto 60/40: settle = ``settle_frac``·per_v,
        measure = (1-``settle_frac``)·per_v; user setting is ignored.

    ``K`` is computed from ``measure_s`` and the per-sample integration
    time ``nplc / line_freq_hz``. Minimum is 1 to keep the script valid.
    """
    n = max(1, int(n_voltages))
    per_v_s = budget_s / n
    nplc_s = max(nplc / line_freq_hz, 1e-6)

    if float(user_settle_ms) == 0.0:
        settle_s = 0.0
        measure_s = per_v_s
        trim_pct = ZERO_SETTLE_TRIM_PCT
    else:
        settle_s = per_v_s * settle_frac
        measure_s = per_v_s * (1.0 - settle_frac)
        trim_pct = 0.0

    # K is the number of chA+chB pairs rapid-fired during ``measure_s``.
    K = max(1, int(measure_s / nplc_s))
    total_s = n * (settle_s + measure_s)
    return settle_s, measure_s, K, trim_pct, per_v_s, total_s


def _trimmed_stats(arr, trim_pct):
    """Return (mean, std, n_used) after trimming the leading ``trim_pct``%."""
    n = len(arr)
    if n == 0:
        return float("nan"), float("nan"), 0
    start = int(n * trim_pct / 100.0)
    core = np.asarray(arr[start:], dtype=np.float64)
    if core.size == 0:
        core = np.asarray(arr, dtype=np.float64)
    return float(core.mean()), float(core.std()), int(core.size)


def _splice_pm_per_voltage(
    pm_times_abs,
    pm_arr,
    k_seg_starts_abs,
    k_seg_ends_abs,
    trim_pct,
):
    """For each voltage j, select PM samples in
    ``[k_seg_starts_abs[j], k_seg_ends_abs[j]]`` (both in the PM400
    perf_counter frame) and trim ``trim_pct`` % from the leading edge.

    Returns a list of (mean, std, n_used) tuples, one per voltage.
    """
    pm_t = np.asarray(pm_times_abs, dtype=np.float64)
    pm_v = np.asarray(pm_arr, dtype=np.float64)
    out = []
    for t_lo, t_hi in zip(k_seg_starts_abs, k_seg_ends_abs):
        mask = (pm_t >= t_lo) & (pm_t <= t_hi)
        seg = pm_v[mask]
        n_seg = seg.size
        if n_seg == 0:
            out.append((float("nan"), float("nan"), 0))
            continue
        start = int(n_seg * trim_pct / 100.0)
        core = seg[start:] if start < n_seg else seg
        out.append((float(core.mean()), float(core.std()), int(core.size)))
    return out


class ExperimentalSweepMode(MeasurementMode):
    """Hardware-paced NVLED sweep with co-timed PM400 array capture."""

    is_chunk_mode = False

    def configure_instruments(self, pm, smu, ctx: MeasurementContext) -> dict:
        cfg = ctx.cfg
        delta_t_us = 100  # always max PM400 rate (100 µs = 10 kHz)
        window_ms = 1000.0
        n_pm_samples, _ = pm.configure_array_mode(window_ms, delta_t_us)

        # Experimental mode hard-codes low NPLC + zero delays so the K
        # rapid-fire burst fits inside the 40 % measure window per voltage.
        # Re-do channel setup explicitly (previous phases leave different
        # NPLC / range / autozero state that would stall the TSP script).
        k_nplc = 0.001
        smu.configure_channel(
            "a", compliance_current=cfg.vled_compliance,
            nplc=k_nplc, high_c=False, zero_delays=True,
            range_i=cfg.vled_range_i,
        )
        smu.configure_channel(
            "b", compliance_current=cfg.nvled_compliance,
            nplc=k_nplc, high_c=False, zero_delays=True,
            range_i=cfg.nvled_range_i,
        )
        return {
            "delta_t_us": delta_t_us,
            "window_ms": window_ms,
            "n_pm_samples": n_pm_samples,
            "k_nplc": k_nplc,
        }

    def cleanup_instruments(self, pm, smu, ctx: MeasurementContext) -> None:
        try:
            smu.cleanup_experimental_sweep()
        except Exception as e:
            ctx.log(f"Experimental cleanup error: {e}")
        try:
            pm.abort()
        except Exception:
            pass

    def measure_step(
        self, pm, smu, smile_dev, log_x, log_y, bit_val,
        nvled_voltages, data_buffer, instr, ctx,
    ) -> None:
        cfg = ctx.cfg
        profiling = ctx.profiling
        start_time = ctx.start_time
        delta_t_us = instr["delta_t_us"]
        n_pm_samples = instr["n_pm_samples"]
        k_nplc = instr["k_nplc"]

        v_list = list(nvled_voltages)
        if not v_list:
            return

        settle_s, measure_s, K, trim_pct, per_v_s, total_s = compute_sweep_timing(
            n_voltages=len(v_list),
            user_settle_ms=cfg.nvled_settle_ms,
            nplc=k_nplc,
        )

        profiling["img_gen"].append(0.0)

        pm.abort()
        t0 = time.perf_counter()
        smu.configure_experimental_sweep(
            v_list=v_list,
            vled_voltage=cfg.vled_voltage,
            settle_time_s=settle_s,
            points_per_voltage=K,
        )
        profiling["k_setup"].append(time.perf_counter() - t0)

        t0 = time.perf_counter()
        if smile_dev is not None:
            smile_dev.set_pixel(log_x, log_y, bit_val)
        profiling["img_send"].append(time.perf_counter() - t0)

        # Arm PM400, then fire the K script. The offset between the two
        # arm timestamps is small (<< 1 ms) — used to map K timestamps
        # (referenced to K's first sample) into the PM400 perf_counter
        # frame below.
        T_pm_start = time.perf_counter()
        pm.start_array()
        T_k_fire_start = time.perf_counter()
        smu.initiate_experimental_sweep()
        T_k_fire_end = time.perf_counter()

        # Wait for the script to finish + a small margin, then tear down.
        sleep_s = total_s + 0.05
        elapsed = time.perf_counter() - T_pm_start
        remaining = sleep_s - elapsed
        if remaining > 0:
            time.sleep(remaining)

        sweep_ok = True
        try:
            smu.join_experimental_sweep(timeout_s=total_s + 0.5)
        except Exception as e:
            sweep_ok = False
            ctx.log(f"ERROR: experimental sweep join failed: {e}")

        total_elapsed = time.perf_counter() - T_pm_start
        profiling["remaining"].append(max(0.0, 1.0 - total_elapsed))

        n_captured = min(
            n_pm_samples,
            max(1, int(total_elapsed * 1e6 / delta_t_us)),
        )
        pm.abort()

        t0 = time.perf_counter()
        pm_arr = pm.fetch_array(n_captured)
        profiling["pm_arm_to_fetch"].append(time.perf_counter() - t0)

        t0 = time.perf_counter()
        ia, ib, vb, ta_inst, tb_inst = smu.read_experimental_sweep_buffers()
        profiling["k_arm_to_fetch"].append(time.perf_counter() - t0)

        try:
            smu.cleanup_experimental_sweep()
        except Exception:
            pass

        if not sweep_ok or not tb_inst or not pm_arr or not ia:
            ctx.log(
                f"WARNING: experimental sweep returned empty buffers at "
                f"pixel ({log_x},{log_y}) — n_pm={len(pm_arr)}, "
                f"n_ka={len(ta_inst)}, n_kb={len(tb_inst)}"
            )
            self._maybe_turnoff_dis(smile_dev, ctx)
            return

        # Map K instrument timestamps into the PM400 perf_counter frame.
        # The K script's first sample corresponds (modulo settle_s) to
        # T_k_fire (perf_counter at initiate). We take the midpoint of
        # the initiate() call as t=0 reference.
        T_k_ref = (T_k_fire_start + T_k_fire_end) / 2.0
        ta_abs = [T_k_ref + t for t in ta_inst]
        tb_abs = [T_k_ref + t for t in tb_inst]

        # Slice the flat N·K buffer into per-voltage chunks. If the
        # instrument returned fewer than expected, clamp to K_got.
        n_total = len(ib)
        K_got = n_total // len(v_list) if len(v_list) > 0 else 0
        if K_got < 1:
            ctx.log(
                f"WARNING: experimental sweep got {n_total} K samples "
                f"for {len(v_list)} voltages — skipping pixel."
            )
            self._maybe_turnoff_dis(smile_dev, ctx)
            return

        # PM400 sample times in the same absolute (perf_counter) frame.
        pm_times_abs = [
            T_pm_start + i * delta_t_us * 1e-6 for i in range(len(pm_arr))
        ]

        # Per-voltage window boundaries for PM splicing — use the K
        # burst's first and last timestamps for voltage j.
        k_seg_starts = []
        k_seg_ends = []
        for j in range(len(v_list)):
            i0 = j * K_got
            i1 = min((j + 1) * K_got - 1, len(tb_abs) - 1)
            k_seg_starts.append(tb_abs[i0])
            k_seg_ends.append(tb_abs[i1])

        pm_per_v = _splice_pm_per_voltage(
            pm_times_abs, pm_arr, k_seg_starts, k_seg_ends, trim_pct,
        )

        # Emit one row-set per voltage (PM400 / VLED / NVLED / NVLED_V_MEAS),
        # with the anchor time on the K burst start for that voltage.
        for j, nv_nom in enumerate(v_list):
            i0 = j * K_got
            i1 = (j + 1) * K_got
            vl_seg = ia[i0:i1]
            nv_seg = ib[i0:i1]
            vv_seg = vb[i0:i1]

            vl_m, vl_s, _ = _trimmed_stats(vl_seg, trim_pct)
            nv_m, nv_s, _ = _trimmed_stats(nv_seg, trim_pct)
            vv_m, vv_s, n_used = _trimmed_stats(vv_seg, trim_pct)
            pm_m, pm_s, n_pm_used = pm_per_v[j]

            t_anchor = round(k_seg_starts[j] - start_time, 6)
            data_buffer.extend(
                [
                    [log_x, log_y, bit_val, nv_nom, t_anchor, "PM400",
                     f"{pm_m:.6e}", f"{pm_s:.6e}"],
                    [log_x, log_y, bit_val, nv_nom, t_anchor, "VLED",
                     f"{vl_m:.6e}", f"{vl_s:.6e}"],
                    [log_x, log_y, bit_val, nv_nom, t_anchor, "NVLED",
                     f"{nv_m:.6e}", f"{nv_s:.6e}"],
                    [log_x, log_y, bit_val, nv_nom, t_anchor, "NVLED_V_MEAS",
                     f"{vv_m:.6e}", f"{n_used}"],
                ]
            )

        # Full waveform dump (non-blocking via DataWriter thread).
        if cfg.secondary_storage_enabled and ctx.sec_dir is not None:
            pm_times_rel = [t - T_pm_start for t in pm_times_abs]
            k_times_rel = [t - T_pm_start for t in tb_abs]
            ctx.data_writer.write_transient(
                x=log_x, y=log_y, bv=bit_val, nv_vol=float(v_list[0]),
                pm_times=pm_times_rel, pm_arr=list(pm_arr),
                k_times=k_times_rel, vled_arr=list(ia), nvled_arr=list(ib),
                T0=T_pm_start, mode="EXP",
                t_ack_s=(T_k_ref - T_pm_start), t_turnoff_s=None,
            )

        self._maybe_turnoff_dis(smile_dev, ctx)

        if cfg.dark_acq:
            self._dark_scalar(pm, smu, log_x, log_y, bit_val,
                              float(v_list[-1]), data_buffer, ctx)
            pm.configure_array_mode(instr["window_ms"], delta_t_us)

    def _dark_scalar(
        self, pm, smu, log_x, log_y, bit_val, nv_vol, data_buffer, ctx,
    ) -> None:
        cfg = ctx.cfg
        dark_settle = cfg.dark_settle_ms / 1000.0
        if dark_settle > 0:
            time.sleep(dark_settle)
        t0 = time.perf_counter()
        pm.configure_scalar()
        pm_dark = pm.measure()
        vled_dark, nvled_dark = smu.measure_instant()
        ctx.profiling["dark_acq"].append(time.perf_counter() - t0)
        t_anchor = round(time.perf_counter() - ctx.start_time, 6)
        data_buffer.extend(
            [
                [log_x, log_y, bit_val, nv_vol, t_anchor, "DARK_PM400",
                 f"{pm_dark:.6e}", "nan"],
                [log_x, log_y, bit_val, nv_vol, t_anchor, "DARK_VLED",
                 f"{vled_dark:.6e}", "nan"],
                [log_x, log_y, bit_val, nv_vol, t_anchor, "DARK_NVLED",
                 f"{nvled_dark:.6e}", "nan"],
            ]
        )
