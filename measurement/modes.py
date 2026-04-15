"""Measurement strategies.

Each concrete ``MeasurementMode`` subclass encapsulates one acquisition
technique's instrument configuration, per-pixel capture, and teardown.
The ``ScanCoordinator`` (see ``coordinator.py``) drives the outer pixel
loop and dispatches into these strategies, so adding a new technique
(e.g. Ultra-Fast mode) only requires a new subclass — not a branch in a
monolithic worker.

Design notes
------------
* ``configure_instruments`` runs once at the start of a scan and returns
  an opaque ``instr`` dict that the coordinator echoes back to
  ``measure_step``. Anything that belongs in that dict (PM400 sample
  counts, Keithley trigger count, fast-scan n_pts, ...) is owned by the
  mode.
* ``measure_step`` performs a single ``(pixel, bit_value)`` measurement
  and appends overview CSV rows to ``data_buffer``. It may itself
  iterate over NVLED voltages (Mode B transient, swept Fast Scan) — the
  coordinator does not know about voltage sweeps.
* ``cleanup_instruments`` is called unconditionally after the coordinator
  finishes (even on early abort), so it must be safe to call after a
  partial capture.
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod

import numpy as np

from .context import MeasurementContext


def _steady_state_stats(arr, tail_pct=20.0):
    """Return (mean_str, std_str) of the last ``tail_pct%`` of ``arr``.

    Duplicated from the pre-refactor module-level helper in
    ``smile_automatic_gui_2602b.py`` so this package has no back-reference
    to the GUI module.
    """
    if not arr:
        return "nan", "nan"
    n = max(1, int(len(arr) * tail_pct / 100.0))
    tail = np.array(arr[-n:], dtype=np.float64)
    return f"{float(tail.mean()):.6e}", f"{float(tail.std()):.6e}"


class MeasurementMode(ABC):
    """Strategy interface for one acquisition technique.

    Two dispatch styles are supported:

    * **Per-pixel** (default, ``is_chunk_mode = False``) — the coordinator
      iterates ``pixel × bit_val`` and calls :meth:`measure_step` for each
      combination. This is what :class:`TransientMode` and
      :class:`FastScanMode` use.
    * **Chunked** (``is_chunk_mode = True``) — the coordinator passes all
      remaining pixels to :meth:`measure_chunk`, which consumes as many
      as fit in one capture buffer and returns the count.  Used by
      :class:`~measurement.ultra_fast.UltraFastMode`, which captures
      many pixels inside a single PM400 array acquisition and slices
      them post-hoc by FPGA-ACK timestamp.
    """

    is_chunk_mode: bool = False

    @abstractmethod
    def configure_instruments(self, pm, smu, ctx: MeasurementContext) -> dict:
        """One-shot setup at the start of a scan.

        Returns an ``instr`` dict that the coordinator threads back into
        each ``measure_step`` / ``measure_chunk`` call. May be empty for
        modes that keep no per-run state.
        """

    def measure_step(
        self,
        pm,
        smu,
        smile_dev,
        log_x: int,
        log_y: int,
        bit_val: int,
        nvled_voltages: list,
        data_buffer: list,
        instr: dict,
        ctx: MeasurementContext,
    ) -> None:
        """Acquire one ``(pixel, bit_val)`` combination.

        Rows are appended to ``data_buffer`` (flushed by the coordinator
        at pixel boundaries). Secondary-storage writes go straight to
        ``ctx.data_writer``. Per-pixel modes MUST override this;
        chunk-mode implementations may leave it as the default raise.
        """
        raise NotImplementedError(
            "measure_step() not implemented — is this a chunk-mode strategy?"
        )

    def measure_chunk(
        self,
        pm,
        smu,
        smile_dev,
        pixels: list,
        bit_val: int,
        nvled_voltages: list,
        data_buffer: list,
        instr: dict,
        ctx: MeasurementContext,
    ) -> int:
        """Acquire as many pixels as fit in one capture buffer.

        ``pixels`` is the full list of remaining pixels for this bit
        value.  The method fires pixels until the capture window is
        about to expire and returns the number actually consumed.

        Appends rows to ``data_buffer`` in the same schema as
        :meth:`measure_step` (``[x, y, bitval, nv_v, t, type, mean, std]``).
        Chunk-mode strategies must override this.
        """
        raise NotImplementedError

    def cleanup_instruments(self, pm, smu, ctx: MeasurementContext) -> None:
        """Per-run teardown. Default is a no-op."""
        return None

    # ------------------------------------------------------------------
    # Shared helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _maybe_turnoff_dis(smile_dev, ctx: MeasurementContext) -> None:
        """Blank the FPGA frame between pixels if the operator asked for it."""
        if not ctx.cfg.turnoff_dis:
            return
        t0 = time.perf_counter()
        if smile_dev is not None:
            smile_dev.blank_frame()
        ctx.profiling["turnoff_dis"].append(time.perf_counter() - t0)


# ======================================================================
# Full Transient
# ======================================================================


class TransientMode(MeasurementMode):
    """Hardware-triggered PM400 array + Keithley buffer per pixel.

    Mode A (fixed NVLED): one transient per ``(pixel, bit_val)``.
    Mode B (swept NVLED): one transient per voltage step inside a
    ``(pixel, bit_val)`` — selected by ``cfg.nvled_sweep``.
    """

    def configure_instruments(self, pm, smu, ctx: MeasurementContext) -> dict:
        cfg = ctx.cfg
        window_ms = cfg.window_ms
        min_remaining_ms = cfg.min_remaining_ms
        delta_t_us = 100
        dark_tail_ms = cfg.dark_tail_ms if cfg.dark_acq else 0
        extended_window_ms = window_ms + min_remaining_ms + dark_tail_ms
        n_pm_samples, _ = pm.configure_array_mode(extended_window_ms, delta_t_us)
        k_nplc = max(cfg.vled_nplc, cfg.nvled_nplc)
        expected_k_count = max(1, int(extended_window_ms / (k_nplc * 20.0)))
        k_trigger_count = min(int(expected_k_count * 1.5) + 10, 60000)

        t0 = time.perf_counter()
        smu.setup_buffers(timestamps=True)
        smu.configure_hardware_trigger(k_trigger_count)
        ctx.profiling["k_setup"].append(time.perf_counter() - t0)

        return {
            "window_ms": window_ms,
            "min_remaining_ms": min_remaining_ms,
            "dark_tail_ms": dark_tail_ms,
            "delta_t_us": delta_t_us,
            "extended_window_ms": extended_window_ms,
            "n_pm_samples": n_pm_samples,
            "k_trigger_count": k_trigger_count,
        }

    def measure_step(
        self,
        pm,
        smu,
        smile_dev,
        log_x,
        log_y,
        bit_val,
        nvled_voltages,
        data_buffer,
        instr,
        ctx,
    ) -> None:
        if ctx.cfg.nvled_sweep:
            self._mode_b(
                pm, smu, smile_dev, log_x, log_y, bit_val,
                nvled_voltages, data_buffer, instr, ctx,
            )
        else:
            self._mode_a(
                pm, smu, smile_dev, log_x, log_y, bit_val,
                data_buffer, instr, ctx,
            )

    # ------------------------------------------------------------------
    # Mode A: fixed NVLED — single turn-on transient per bit value.
    # ------------------------------------------------------------------
    def _mode_a(
        self, pm, smu, smile_dev, log_x, log_y, bit_val,
        data_buffer, instr, ctx,
    ) -> None:
        cfg = ctx.cfg
        profiling = ctx.profiling
        start_time = ctx.start_time
        n_pm_samples = instr["n_pm_samples"]
        delta_t_us = instr["delta_t_us"]
        window_ms = instr["window_ms"]
        min_remaining_ms = instr["min_remaining_ms"]
        k_trigger_count = instr["k_trigger_count"]

        profiling["img_gen"].append(0.0)

        # Reset PM400 state from previous pixel (ABOR clears FETC:STAT).
        # CONF:ARR parameters remain valid after ABOR.
        t0 = time.perf_counter()
        pm.abort()
        smu.clear_buffers_only()
        smu.configure_hardware_trigger(k_trigger_count)
        profiling["k_buffer_clear"].append(time.perf_counter() - t0)

        T_arm_start = time.perf_counter()
        pm.start_array()
        smu.start_hardware_trigger()
        T_arm_end = time.perf_counter()
        T0 = (T_arm_start + T_arm_end) / 2.0

        t0 = time.perf_counter()
        if smile_dev is not None:
            smile_dev.set_pixel(log_x, log_y, bit_val)
        profiling["img_send"].append(time.perf_counter() - t0)
        t_ack_s = time.perf_counter() - T0

        elapsed = time.perf_counter() - T_arm_start
        remaining_normal = (window_ms / 1000.0) - elapsed
        dark_tail_ms = instr.get("dark_tail_ms", 0)
        t_turnoff_s = None
        if dark_tail_ms > 0 and cfg.dark_acq:
            # Sleep until end of illumination window, then blank.
            if remaining_normal > 0:
                time.sleep(remaining_normal)
            self._maybe_turnoff_dis(smile_dev, ctx)
            t_turnoff_s = time.perf_counter() - T0
            time.sleep(dark_tail_ms / 1000.0)
        else:
            actual_sleep = max(remaining_normal, min_remaining_ms / 1000.0)
            if actual_sleep > 0:
                time.sleep(actual_sleep)
        profiling["remaining"].append(remaining_normal)

        n_captured = min(
            n_pm_samples,
            max(1, int((time.perf_counter() - T_arm_start) * 1e6 / delta_t_us)),
        )
        smu.abort_trigger()
        pm.abort()

        tail_pct = cfg.steady_tail_pct
        need_full = cfg.secondary_storage_enabled
        t0 = time.perf_counter()
        if need_full:
            pm_arr = pm.fetch_array(n_captured)
            pm_fetch_offset = 0
        else:
            n_tail = max(1, int(n_captured * tail_pct / 100))
            pm_fetch_offset = n_captured - n_tail
            pm_arr = pm.fetch_array(n_tail, start_offset=pm_fetch_offset)
        profiling["pm_arm_to_fetch"].append(time.perf_counter() - t0)

        t0 = time.perf_counter()
        vled_arr, nvled_arr, ta_arr, _ = smu.read_buffer_with_timestamps()
        profiling["k_arm_to_fetch"].append(time.perf_counter() - t0)

        pm_times = [
            T0 + (pm_fetch_offset + i) * delta_t_us * 1e-6
            for i in range(len(pm_arr))
        ]
        k_times = [t - (ta_arr[0] - T0) for t in ta_arr] if ta_arr else []

        nv_vol = cfg.nvled_voltage

        # Extract dark samples from tail if in-window dark was captured.
        if dark_tail_ms > 0 and cfg.dark_acq:
            n_dark = max(1, int(dark_tail_ms * 1000 / delta_t_us))
            dark_samples = (
                pm_arr[-n_dark:] if len(pm_arr) >= n_dark else pm_arr
            )
            pm_dark_val = (
                float(np.mean(dark_samples))
                if dark_samples
                else float("nan")
            )
            t_anchor = round(time.perf_counter() - start_time, 6)
            data_buffer.extend(
                [
                    [
                        log_x, log_y, bit_val, nv_vol, t_anchor,
                        "DARK_PM400",
                        f"{pm_dark_val:.6e}",
                        f"{float(np.std(dark_samples)):.6e}",
                    ],
                ]
            )

        t_anchor = round(T0 - start_time, 6)
        eff_tail = 100.0 if not need_full else tail_pct
        pm_mean, pm_std = _steady_state_stats(pm_arr, eff_tail)
        vl_mean, vl_std = _steady_state_stats(vled_arr, tail_pct)
        nv_mean, nv_std = _steady_state_stats(nvled_arr, tail_pct)
        data_buffer.extend(
            [
                [log_x, log_y, bit_val, nv_vol, t_anchor, "PM400", pm_mean, pm_std],
                [log_x, log_y, bit_val, nv_vol, t_anchor, "VLED", vl_mean, vl_std],
                [log_x, log_y, bit_val, nv_vol, t_anchor, "NVLED", nv_mean, nv_std],
            ]
        )

        if cfg.secondary_storage_enabled and ctx.sec_dir is not None:
            ctx.data_writer.write_transient(
                x=log_x, y=log_y, bv=bit_val, nv_vol=nv_vol,
                pm_times=pm_times, pm_arr=pm_arr,
                k_times=k_times, vled_arr=vled_arr, nvled_arr=nvled_arr,
                T0=T0, mode="A",
                t_ack_s=t_ack_s, t_turnoff_s=t_turnoff_s,
            )

        # Skip redundant turnoff when dark_tail_ms > 0 — the pixel was
        # already blanked mid-window to create the dark tail.
        if not (dark_tail_ms > 0 and cfg.dark_acq):
            self._maybe_turnoff_dis(smile_dev, ctx)
        self._dark_acq_transient(
            pm, smu, log_x, log_y, bit_val, nv_vol,
            data_buffer, instr, ctx,
        )

    # ------------------------------------------------------------------
    # Mode B: swept NVLED — one transient per voltage step.
    # ------------------------------------------------------------------
    def _mode_b(
        self, pm, smu, smile_dev, log_x, log_y, bit_val,
        nvled_voltages, data_buffer, instr, ctx,
    ) -> None:
        cfg = ctx.cfg
        profiling = ctx.profiling
        start_time = ctx.start_time
        n_pm_samples = instr["n_pm_samples"]
        delta_t_us = instr["delta_t_us"]
        window_ms = instr["window_ms"]
        min_remaining_ms = instr["min_remaining_ms"]
        k_trigger_count = instr["k_trigger_count"]

        profiling["img_gen"].append(0.0)

        t0 = time.perf_counter()
        if smile_dev is not None:
            smile_dev.set_pixel(log_x, log_y, bit_val)
        profiling["img_send"].append(time.perf_counter() - t0)

        dark_every = cfg.dark_every_n_sweep if cfg.dark_acq else 0

        # Pre-sweep dark (pixel not yet displayed for transient — blank is
        # the initial state, so just measure)
        if cfg.dark_acq:
            self._dark_acq_transient(
                pm, smu, log_x, log_y, bit_val, cfg.nvled_voltage,
                data_buffer, instr, ctx,
            )
            # Re-display pixel for the sweep
            if smile_dev is not None:
                smile_dev.set_pixel(log_x, log_y, bit_val)

        nv_vol = cfg.nvled_voltage  # fallback value if sweep list is empty
        for step_idx, nv_vol in enumerate(nvled_voltages):
            if not ctx.is_running():
                break

            # Interleaved dark every N voltage steps
            if dark_every > 0 and step_idx > 0 and step_idx % dark_every == 0:
                self._dark_acq_transient(
                    pm, smu, log_x, log_y, bit_val, nv_vol,
                    data_buffer, instr, ctx,
                )
                # Re-display pixel
                if smile_dev is not None:
                    smile_dev.set_pixel(log_x, log_y, bit_val)

            pm.abort()
            smu.clear_buffers_only()
            smu.configure_hardware_trigger(k_trigger_count)

            smu.set_voltage("b", nv_vol)
            time.sleep(cfg.nvled_settle_ms / 1000.0)

            T_arm_start = time.perf_counter()
            pm.start_array()
            smu.start_hardware_trigger()
            T_arm_end = time.perf_counter()
            T0 = (T_arm_start + T_arm_end) / 2.0

            elapsed = time.perf_counter() - T_arm_start
            remaining_normal = (window_ms / 1000.0) - elapsed
            dark_tail_ms = instr.get("dark_tail_ms", 0)
            t_turnoff_s = None
            if dark_tail_ms > 0 and cfg.dark_acq:
                if remaining_normal > 0:
                    time.sleep(remaining_normal)
                self._maybe_turnoff_dis(smile_dev, ctx)
                t_turnoff_s = time.perf_counter() - T0
                time.sleep(dark_tail_ms / 1000.0)
            else:
                actual_sleep = max(remaining_normal, min_remaining_ms / 1000.0)
                if actual_sleep > 0:
                    time.sleep(actual_sleep)
            profiling["remaining"].append(remaining_normal)

            n_captured = min(
                n_pm_samples,
                max(
                    1,
                    int((time.perf_counter() - T_arm_start) * 1e6 / delta_t_us),
                ),
            )
            smu.abort_trigger()
            pm.abort()

            tail_pct = cfg.steady_tail_pct
            need_full = cfg.secondary_storage_enabled
            t0 = time.perf_counter()
            if need_full:
                pm_arr = pm.fetch_array(n_captured)
                pm_fetch_offset = 0
            else:
                n_tail = max(1, int(n_captured * tail_pct / 100))
                pm_fetch_offset = n_captured - n_tail
                pm_arr = pm.fetch_array(n_tail, start_offset=pm_fetch_offset)
            profiling["pm_arm_to_fetch"].append(time.perf_counter() - t0)

            t0 = time.perf_counter()
            vled_arr, nvled_arr, ta_arr, _ = smu.read_buffer_with_timestamps()
            profiling["k_arm_to_fetch"].append(time.perf_counter() - t0)

            pm_times = [
                T0 + (pm_fetch_offset + i) * delta_t_us * 1e-6
                for i in range(len(pm_arr))
            ]
            k_times = [t - (ta_arr[0] - T0) for t in ta_arr] if ta_arr else []

            if dark_tail_ms > 0 and cfg.dark_acq:
                n_dark = max(1, int(dark_tail_ms * 1000 / delta_t_us))
                dark_samples = (
                    pm_arr[-n_dark:] if len(pm_arr) >= n_dark else pm_arr
                )
                pm_dark_val = (
                    float(np.mean(dark_samples))
                    if dark_samples
                    else float("nan")
                )
                t_anchor_dark = round(time.perf_counter() - start_time, 6)
                data_buffer.extend(
                    [
                        [
                            log_x, log_y, bit_val, nv_vol, t_anchor_dark,
                            "DARK_PM400",
                            f"{pm_dark_val:.6e}",
                            f"{float(np.std(dark_samples)):.6e}",
                        ],
                    ]
                )

            t_anchor = round(T0 - start_time, 6)
            eff_tail = 100.0 if not need_full else tail_pct
            pm_mean, pm_std = _steady_state_stats(pm_arr, eff_tail)
            vl_mean, vl_std = _steady_state_stats(vled_arr, tail_pct)
            nv_mean, nv_std = _steady_state_stats(nvled_arr, tail_pct)
            data_buffer.extend(
                [
                    [log_x, log_y, bit_val, nv_vol, t_anchor, "PM400", pm_mean, pm_std],
                    [log_x, log_y, bit_val, nv_vol, t_anchor, "VLED", vl_mean, vl_std],
                    [log_x, log_y, bit_val, nv_vol, t_anchor, "NVLED", nv_mean, nv_std],
                ]
            )

            if cfg.secondary_storage_enabled and ctx.sec_dir is not None:
                ctx.data_writer.write_transient(
                    x=log_x, y=log_y, bv=bit_val, nv_vol=nv_vol,
                    pm_times=pm_times, pm_arr=pm_arr,
                    k_times=k_times, vled_arr=vled_arr, nvled_arr=nvled_arr,
                    T0=T0, mode="B",
                    t_ack_s=None, t_turnoff_s=t_turnoff_s,
                )

        # After all voltages swept — skip redundant turnoff when
        # dark_tail_ms > 0 (pixel was blanked inside each voltage step's
        # window already).
        if not (instr["dark_tail_ms"] > 0 and cfg.dark_acq):
            self._maybe_turnoff_dis(smile_dev, ctx)
        self._dark_acq_transient(
            pm, smu, log_x, log_y, bit_val, nv_vol,
            data_buffer, instr, ctx,
        )

    # ------------------------------------------------------------------
    # Dark acquisition (array-mode safe)
    # ------------------------------------------------------------------
    def _dark_acq_transient(
        self, pm, smu, log_x, log_y, bit_val, nv_vol,
        data_buffer, instr, ctx,
    ) -> None:
        """Bug 2 fix — take a scalar dark reading without clobbering array mode.

        * ``dark_tail_ms > 0``: no-op, the illumination window was extended
          and the dark samples are already in the PM400 tail.
        * ``dark_tail_ms == 0`` + ``dark_acq``: drop PM400 into scalar mode
          (``CONF:POW``), take one measurement, then re-arm array mode for
          the next pixel.
        """
        cfg = ctx.cfg
        if not cfg.dark_acq:
            return
        if cfg.dark_tail_ms > 0:
            return

        dark_settle = cfg.dark_settle_ms / 1000.0
        if dark_settle > 0:
            time.sleep(dark_settle)
        t0 = time.perf_counter()
        pm.configure_scalar()
        pm_dark = pm.measure()
        pm.configure_array_mode(instr["extended_window_ms"], instr["delta_t_us"])
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


# ======================================================================
# Fast Scan
# ======================================================================


class FastScanMode(MeasurementMode):
    """Continuous PM400 fetch + Keithley burst-to-buffer per pixel.

    Much lower per-pixel overhead than :class:`TransientMode` at the cost
    of no transient waveform — just ``n_pts`` samples per voltage step.
    """

    def configure_instruments(self, pm, smu, ctx: MeasurementContext) -> dict:
        cfg = ctx.cfg
        # Mirror the Part-7 benchmark sequence that confirmed FETC? works:
        #   ABOR → CONF:POW → re-apply settings → INIT:CONT.
        # Skipping CONF:POW leaves the instrument in an undefined mode;
        # FETC? then returns the same stale measurement on every call.
        # ``configure_scalar`` encapsulates the ABOR+CONF:POW half of the
        # sequence so this path also works against PM400Sim, which has no
        # raw ``inst`` handle.
        pm.configure_scalar()
        pm.set_wavelength(cfg.pm_wavelength)
        pm.set_power_unit("W")
        pm.set_auto_range(False)
        pm.set_range(cfg.pm_range)
        pm.set_averaging(1)
        time.sleep(0.05)
        pm.start_continuous()

        t0 = time.perf_counter()
        smu.setup_buffers(timestamps=False)  # appendmode=1 for burst
        ctx.profiling["k_setup"].append(time.perf_counter() - t0)

        return {
            "settle_s": cfg.fast_scan_settle_ms / 1000.0,
            "n_pts": cfg.fast_scan_n_pts,
            "is_swept": cfg.nvled_sweep,
        }

    def cleanup_instruments(self, pm, smu, ctx: MeasurementContext) -> None:
        try:
            pm.stop_continuous()
        except Exception:
            pass

    def measure_step(
        self, pm, smu, smile_dev, log_x, log_y, bit_val,
        nvled_voltages, data_buffer, instr, ctx,
    ) -> None:
        cfg = ctx.cfg
        profiling = ctx.profiling
        start_time = ctx.start_time
        settle_s = instr["settle_s"]
        n_pts = instr["n_pts"]
        is_swept = instr["is_swept"]

        profiling["img_gen"].append(0.0)

        t0 = time.perf_counter()
        if smile_dev is not None:
            smile_dev.set_pixel(log_x, log_y, bit_val)
        profiling["img_send"].append(time.perf_counter() - t0)

        if settle_s > 0:
            time.sleep(settle_s)

        # Clear SMU buffer once; burst appends for each voltage.
        smu.clear_buffers_only()
        pm_vals_by_vol = []  # list[list[(value, t)]] indexed by [vol_idx][pt_idx]
        dark_every = cfg.dark_every_n_sweep if (cfg.dark_acq and is_swept) else 0

        # Pre-sweep dark: blank frame + dark reading before first voltage
        if cfg.dark_acq and is_swept:
            self._take_dark_reading(
                pm, smu, smile_dev, log_x, log_y, bit_val,
                cfg.nvled_voltage, data_buffer, ctx,
            )
            # Re-display the pixel for the sweep
            if smile_dev is not None:
                smile_dev.set_pixel(log_x, log_y, bit_val)
            if settle_s > 0:
                time.sleep(settle_s)

        nv_vol = cfg.nvled_voltage  # fallback for turnoff / dark acq paths
        for step_idx, nv_vol in enumerate(nvled_voltages):
            if not ctx.is_running():
                break

            # Interleaved dark every N voltage steps
            if dark_every > 0 and step_idx > 0 and step_idx % dark_every == 0:
                self._take_dark_reading(
                    pm, smu, smile_dev, log_x, log_y, bit_val,
                    nv_vol, data_buffer, ctx,
                )
                # Re-display the pixel and re-settle
                if smile_dev is not None:
                    smile_dev.set_pixel(log_x, log_y, bit_val)
                if settle_s > 0:
                    time.sleep(settle_s)

            if is_swept:
                smu.set_voltage("b", nv_vol)
                time.sleep(cfg.nvled_settle_ms / 1000.0)

            # Fire SMU burst (non-blocking write), then immediately start
            # PM400 fetches. The Keithley executes the TSP loop autonomously
            # while Python is busy with FETC? calls.
            t0_k = time.perf_counter()
            smu.measure_burst_fire(n_pts)

            t0 = time.perf_counter()
            blk = [
                (pm.fetch_latest(), time.perf_counter()) for _ in range(n_pts)
            ]
            profiling["pm_arm_to_fetch"].append(time.perf_counter() - t0)

            smu.measure_burst_join()
            profiling["k_arm_to_fetch"].append(time.perf_counter() - t0_k)
            pm_vals_by_vol.append(blk)

        # Single SMU buffer read for all n_vols × n_pts entries.
        vled_all, nvled_all = smu.read_buffers()

        for j, nv_vol_j in enumerate(nvled_voltages):
            pm_blk = pm_vals_by_vol[j] if j < len(pm_vals_by_vol) else []
            for k in range(n_pts):
                idx = j * n_pts + k
                pm_val, t_pm = (
                    pm_blk[k]
                    if k < len(pm_blk)
                    else (float("nan"), time.perf_counter())
                )
                vled_i = vled_all[idx] if idx < len(vled_all) else float("nan")
                nvled_i = (
                    nvled_all[idx] if idx < len(nvled_all) else float("nan")
                )
                t_anchor = round(t_pm - start_time, 6)
                data_buffer.extend(
                    [
                        [log_x, log_y, bit_val, nv_vol_j, t_anchor, "PM400",
                         f"{pm_val:.6e}", "nan"],
                        [log_x, log_y, bit_val, nv_vol_j, t_anchor, "VLED",
                         f"{vled_i:.6e}", "nan"],
                        [log_x, log_y, bit_val, nv_vol_j, t_anchor, "NVLED",
                         f"{nvled_i:.6e}", "nan"],
                    ]
                )

        self._maybe_turnoff_dis(smile_dev, ctx)
        # fetch_latest() is safe here — FETC:POW? does not abort INIT:CONT.
        self._maybe_dark_acq_fast(
            pm, smu, log_x, log_y, bit_val, nv_vol, data_buffer, ctx,
        )

    def _take_dark_reading(
        self, pm, smu, smile_dev, log_x, log_y, bit_val, nv_vol,
        data_buffer, ctx,
    ) -> None:
        """Blank the pixel, settle, take one dark PM400 + SMU reading.

        Used for interleaved darks during sweep and the final post-pixel dark.
        The caller is responsible for re-displaying the pixel afterwards if
        the measurement continues.
        """
        cfg = ctx.cfg
        if smile_dev is not None:
            smile_dev.blank_frame()
        dark_settle = cfg.dark_settle_ms / 1000.0
        if dark_settle > 0:
            time.sleep(dark_settle)
        t0 = time.perf_counter()
        pm_dark = pm.fetch_latest()
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

    def _maybe_dark_acq_fast(
        self, pm, smu, log_x, log_y, bit_val, nv_vol, data_buffer, ctx,
    ) -> None:
        """Final dark acquisition after all voltage steps complete.

        The pixel should already be blanked by ``_maybe_turnoff_dis``
        before this is called.
        """
        cfg = ctx.cfg
        if not cfg.dark_acq:
            return
        dark_settle = cfg.dark_settle_ms / 1000.0
        if dark_settle > 0:
            time.sleep(dark_settle)
        t0 = time.perf_counter()
        pm_dark = pm.fetch_latest()
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
