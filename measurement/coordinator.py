"""Outer scan driver.

Owns the pixel iteration loop, ETA estimation, and progress signalling.
Per-pixel work is delegated to a :class:`~measurement.modes.MeasurementMode`
strategy, so adding a new technique never touches this file.

The coordinator is deliberately thin — it knows nothing about PM400
modes, NVLED sweeps, dark acquisition, or secondary storage. Those all
live inside the mode implementations.

Two dispatch paths
------------------
* **Per-pixel** (default) — outer loop is ``pixel × bit_val``; each
  combination hits ``mode.measure_step``. Used by Full Transient and
  Fast Scan, which cannot batch pixels.
* **Chunked** (``mode.is_chunk_mode``) — outer loop is
  ``bit_val × chunks-of-pixels``; each chunk hits ``mode.measure_chunk``.
  Used by Ultra-Fast mode, which collapses per-pixel arm/fetch overhead
  by capturing many pixels inside one PM400 array acquisition.
"""

from __future__ import annotations

import datetime
import time
from typing import List, Sequence

from .context import MeasurementContext
from .modes import MeasurementMode


class ScanCoordinator:
    """Drives one full scan across ``pixel_list × bit_values``.

    Parameters
    ----------
    mode : MeasurementMode
        The acquisition strategy. Owns instrument configuration and
        per-step capture.
    ctx : MeasurementContext
        Runtime services — config, data writer, profiling dict, cancel
        flag, and progress callbacks.
    """

    def __init__(self, mode: MeasurementMode, ctx: MeasurementContext):
        self.mode = mode
        self.ctx = ctx

    def run(
        self,
        pm,
        smu,
        smile_dev,
        pixel_list: Sequence,
        bit_values: Sequence[int],
        nvled_voltages: Sequence[float],
    ) -> int:
        """Execute the scan. Returns the number of ``(pixel, bit_val)``
        steps that completed — used by the caller to gate post-processing.

        The mode's ``cleanup_instruments`` hook runs unconditionally
        (even on early abort / exception) so that e.g. PM400 continuous
        mode always gets stopped.
        """
        ctx = self.ctx
        instr = self.mode.configure_instruments(pm, smu, ctx)
        try:
            if self.mode.is_chunk_mode:
                return self._run_chunked(
                    pm, smu, smile_dev, pixel_list, bit_values,
                    nvled_voltages, instr,
                )
            return self._run_per_pixel(
                pm, smu, smile_dev, pixel_list, bit_values,
                nvled_voltages, instr,
            )
        finally:
            try:
                self.mode.cleanup_instruments(pm, smu, ctx)
            except Exception as e:
                ctx.log(f"Mode cleanup error: {e}")

    # ------------------------------------------------------------------
    # Per-pixel dispatch
    # ------------------------------------------------------------------
    def _run_per_pixel(
        self, pm, smu, smile_dev, pixel_list, bit_values, nvled_voltages, instr,
    ) -> int:
        ctx = self.ctx
        step_count = 0
        total_steps = max(1, len(pixel_list) * len(bit_values))
        moving_avg_time = 0.0

        for log_x, log_y in pixel_list:
            if not ctx.is_running():
                break
            ctx.pixel_active(log_x, log_y)
            data_buffer: List[list] = []

            for bit_val in bit_values:
                if not ctx.is_running():
                    break
                t_step_start = time.perf_counter()
                self.mode.measure_step(
                    pm, smu, smile_dev,
                    log_x, log_y, bit_val,
                    nvled_voltages, data_buffer,
                    instr, ctx,
                )
                step_count += 1
                step_duration = time.perf_counter() - t_step_start
                moving_avg_time = (
                    step_duration
                    if moving_avg_time == 0
                    else 0.95 * moving_avg_time + 0.05 * step_duration
                )
                rem_seconds = int((total_steps - step_count) * moving_avg_time)
                ctx.set_eta(
                    f"Pixel: {log_x},{log_y} | "
                    f"Progress: {int((step_count / total_steps) * 100)}% | "
                    f"ETA: {datetime.timedelta(seconds=rem_seconds)}"
                )

            if data_buffer:
                ctx.data_writer.write_rows(data_buffer)
            ctx.pixel_done(log_x, log_y)

        return step_count

    # ------------------------------------------------------------------
    # Chunked dispatch
    # ------------------------------------------------------------------
    def _run_chunked(
        self, pm, smu, smile_dev, pixel_list, bit_values, nvled_voltages, instr,
    ) -> int:
        ctx = self.ctx
        pixel_list = list(pixel_list)
        n_pixels = len(pixel_list)
        total_steps = max(1, n_pixels * len(bit_values))
        step_count = 0
        moving_avg_time = 0.0

        for bit_val in bit_values:
            if not ctx.is_running():
                break

            cursor = 0
            while cursor < n_pixels and ctx.is_running():
                remaining = pixel_list[cursor:]

                t_chunk_start = time.perf_counter()
                data_buffer: List[list] = []
                n_consumed = self.mode.measure_chunk(
                    pm, smu, smile_dev,
                    remaining, bit_val, nvled_voltages,
                    data_buffer, instr, ctx,
                )
                if n_consumed <= 0:
                    ctx.log("WARNING: chunk consumed 0 pixels — aborting")
                    break

                chunk = pixel_list[cursor : cursor + n_consumed]
                if data_buffer:
                    ctx.data_writer.write_rows(data_buffer)
                for lx, ly in chunk:
                    ctx.pixel_done(lx, ly)

                chunk_duration = time.perf_counter() - t_chunk_start
                per_step = chunk_duration / n_consumed
                moving_avg_time = (
                    per_step
                    if moving_avg_time == 0
                    else 0.95 * moving_avg_time + 0.05 * per_step
                )
                step_count += n_consumed
                cursor += n_consumed
                rem_seconds = int((total_steps - step_count) * moving_avg_time)
                last_px = chunk[-1]
                ctx.set_eta(
                    f"Chunk @ bv={bit_val} | "
                    f"Pixel: {last_px[0]},{last_px[1]} | "
                    f"Progress: {int((step_count / total_steps) * 100)}% | "
                    f"ETA: {datetime.timedelta(seconds=rem_seconds)}"
                )

        return step_count
