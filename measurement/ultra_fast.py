"""Ultra-Fast measurement mode.

Strategy
--------
Instead of arming/fetching the PM400 once per pixel (transient mode) or
relying on one FETC? call per pixel (fast-scan mode), Ultra-Fast fills a
single continuous PM400 array capture with many pixels at once.

The outer per-pixel overhead collapses to just the FPGA's ACK latency
(~5 ms for ``set_pixel``). The PM400 samples continuously at the
operator-chosen ``delta_t_us`` into its 10 000-sample on-board buffer.
After the chunk ends we slice the array by ACK timestamps to recover
per-pixel mean / std. Between pixels we insert blank frames every
``dark_every_n`` steps so we can simultaneously track ambient /
dark-current drift and apply a linearly-interpolated dark correction.

Windowing
---------
Each inter-ACK interval ``[ACK[i], ACK[i+1])`` is the time span during
which the LED shows ``pixel_i``. The first and last ``trim_pct%`` of
that interval are discarded to avoid the turn-on / turn-off transitions
that would otherwise bias the mean. A typical value of 10 % on a 5 ms
interval gives a clean 4 ms average window per pixel.

Chunk sizing
------------
Two buffers compete for the limiting constraint:

* **PM400** — 10 000 samples at ``delta_t_us``.
* **Keithley 2602B** — 60 000 samples per channel, at a rate set by
  the auto-computed NPLC (roughly matched to ``delta_t_us``).

``compute_chunk_geometry`` picks whichever fills first, derives the
chunk duration from that, then divides by the nominal ~6 ms per pixel
(5 ms ACK + 1 ms Python overhead) with a 15 % safety margin.

NPLC selection
--------------
The GUI NPLC spinboxes are greyed out in Ultra-Fast mode. Instead,
NPLC is auto-computed as ``delta_t_us / 20 000`` (at 50 Hz mains),
clamped to ``[0.001, 0.05]``. This roughly matches one Keithley
integration per PM400 sample, giving adequate current readings per
pixel window without overflowing the 60 k buffer.

Output
------
Rows are appended to ``data_buffer`` in the same schema as
:class:`~measurement.modes.TransientMode` — three rows per
``(pixel, bit_val)`` (PM400, VLED, NVLED) plus optional ``DARK_PM400``
rows from the interleaved blank frames, so the existing post-processing
pipeline consumes Ultra-Fast data unchanged.

Diagnostics
-----------
Each chunk saves a ``.npz`` file and a ``.png`` plot under a
``ultra_fast_diag/`` directory next to the measurement CSV. The plot
shows the raw PM400 trace with ACK timestamps overlaid, making it easy
to verify that the slicing windows line up with the actual pixel
intervals.
"""

from __future__ import annotations

import threading
import time
from pathlib import Path

import numpy as np

from .context import MeasurementContext
from .modes import MeasurementMode


# PM400 internal buffer: always 10 000 samples at 100 µs = 1.000 s.
# The delta_t parameter to CONF:ARR is a decimation factor, NOT a
# true sample-rate change.  delta_t=200 means every output sample
# averages 2 internal 100 µs samples, so max samples = 10000/(dt/100).
# The capture duration is always 1 s regardless of delta_t.
_PM_INTERNAL_SAMPLES = 10000
_PM_INTERNAL_DT_US = 100
_PM_CAPTURE_MS = 1000.0  # always 1 second
# Keithley 2602B per-channel buffer depth.
_K_BUFFER_MAX = 60000
# Fixed delay after each FPGA ACK before sending the next command.
# The ACK marks when the display update begins; this wait ensures
# the update is complete before the next pixel fires.
_FPGA_SETTLE_MS = 10.0
# Safety margin — leave 15 % of the chunk budget unused so ACK jitter
# cannot push the last pixel past the array-capture deadline.
_CHUNK_FILL_FRACTION = 0.85
# Assumed overhead per Keithley measurement on top of the NPLC
# integration time (fixed-range, autozero-off).
_K_OVERHEAD_S = 0.0001  # 100 µs


def auto_nplc(delta_t_us: int) -> float:
    """Pick an NPLC that roughly matches the PM400 output sample rate.

    One NPLC at 50 Hz = 20 ms, so ``nplc = delta_t_us / 20 000``
    gives approximately one Keithley reading per PM400 output sample.
    Clamped to [0.001, 0.05] to stay within practical limits.
    """
    return max(0.001, min(0.05, delta_t_us / 20_000.0))


def _k_sample_period_s(nplc: float) -> float:
    """Estimated wall-clock time for one Keithley measurement."""
    return nplc * 0.02 + _K_OVERHEAD_S  # 50 Hz mains assumed


def compute_chunk_geometry(
    delta_t_us: int,
    dark_every_n: int = 1,
) -> tuple[int, float, int, float]:
    """Return ``(n_pm_samples, chunk_duration_ms, pixels_per_chunk, nplc)``.

    Used both at configure time and by the GUI's geometry label.

    The PM400 always captures for 1 second (10 000 internal samples at
    100 µs).  ``delta_t_us`` controls decimation: higher values give
    fewer but averaged output samples (max = 10000 / (delta_t/100)).

    The chunk duration is therefore always 1 s (PM-limited), unless the
    Keithley buffer would overflow first (unlikely at NPLC ≤ 0.05).
    """
    delta_t_us = max(100, int(round(delta_t_us / 100.0)) * 100)
    nplc = auto_nplc(delta_t_us)
    decimation = delta_t_us // _PM_INTERNAL_DT_US
    n_pm_samples = _PM_INTERNAL_SAMPLES // decimation

    # Chunk duration is always 1 s from the PM400 side.
    # Check Keithley doesn't overflow.
    chunk_duration_ms = _PM_CAPTURE_MS
    k_period_s = _k_sample_period_s(nplc)
    k_chunk_ms = _K_BUFFER_MAX * k_period_s * 1000.0
    chunk_duration_ms = min(chunk_duration_ms, k_chunk_ms)

    # Average FPGA events per pixel: 1 (the pixel itself) plus dark
    # frames.  Leading + trailing blanks add 2 events total but are
    # amortised over many pixels, so the dominant term is the
    # intermediate darks.
    if dark_every_n > 0:
        events_per_pixel = 1.0 + 1.0 / dark_every_n
    else:
        events_per_pixel = 1.0
    # Per-event cost: ~2.5 ms ACK round-trip + fixed post-ACK settle.
    est_event_ms = 3.0 + _FPGA_SETTLE_MS
    avg_pixel_ms = events_per_pixel * est_event_ms

    pixels_per_chunk = max(
        1,
        int(chunk_duration_ms * _CHUNK_FILL_FRACTION / avg_pixel_ms),
    )

    return n_pm_samples, chunk_duration_ms, pixels_per_chunk, nplc


class UltraFastMode(MeasurementMode):
    """Chunked PM400 array capture with post-hoc pixel slicing."""

    is_chunk_mode = True

    def __init__(self):
        self._chunk_idx = 0
        self._diag_threads: list[threading.Thread] = []

    # ------------------------------------------------------------------
    # Setup / teardown
    # ------------------------------------------------------------------
    def configure_instruments(self, pm, smu, ctx: MeasurementContext) -> dict:
        cfg = ctx.cfg
        delta_t_us_req = int(cfg.ultra_fast_delta_t_us)
        dark_every_n = max(0, int(cfg.ultra_fast_dark_every_n))
        n_samples, chunk_duration_ms, pixels_per_chunk, nplc = compute_chunk_geometry(
            delta_t_us_req, dark_every_n
        )

        # Apply PM400 acquisition settings that remain constant across
        # chunks. configure_array_mode() is re-called per chunk because
        # fetch_array() leaves the meter idle.
        pm.abort()
        pm.set_wavelength(cfg.pm_wavelength)
        pm.set_power_unit("W")
        pm.set_auto_range(False)
        pm.set_range(cfg.pm_range)
        pm.set_averaging(1)
        n_samples_real, delta_t_us = pm.configure_array_mode(
            chunk_duration_ms, delta_t_us_req
        )

        # SMU setup — auto-computed NPLC overrides the GUI value.
        # Both channels get the same NPLC for simplicity. zero_delays
        # is used for maximum throughput; high_c is incompatible.
        smu.configure_channel(
            "a",
            compliance_current=cfg.vled_compliance,
            nplc=nplc,
            high_c=False,
            zero_delays=True,
            range_i=cfg.vled_range_i,
        )
        smu.configure_channel(
            "b",
            compliance_current=cfg.nvled_compliance,
            nplc=nplc,
            high_c=False,
            zero_delays=True,
            range_i=cfg.nvled_range_i,
        )
        t0 = time.perf_counter()
        smu.setup_buffers(timestamps=True)
        ctx.profiling["k_setup"].append(time.perf_counter() - t0)

        # Keithley trigger count — sized to fill the chunk. Clamp to
        # 60 000 (the 2602B per-channel buffer ceiling).
        k_period_s = _k_sample_period_s(nplc)
        expected_k_count = max(1, int((chunk_duration_ms / 1000.0) / k_period_s))
        k_trigger_count = min(int(expected_k_count * 1.1) + 10, _K_BUFFER_MAX)

        config_msg = (
            f"Ultra-Fast: delta_t={delta_t_us} us, nplc={nplc:.4f}, "
            f"chunk={chunk_duration_ms:.0f} ms, "
            f"pixels/chunk={pixels_per_chunk}, "
            f"dark_every={cfg.ultra_fast_dark_every_n}, "
            f"trim={cfg.ultra_fast_trim_pct:.0f}%"
        )
        ctx.log(config_msg)
        self._append_log(ctx, config_msg)

        return {
            "delta_t_us": delta_t_us,
            "delta_t_s": delta_t_us / 1e6,
            "chunk_duration_ms": chunk_duration_ms,
            "n_samples": n_samples_real,
            "dark_every_n": int(cfg.ultra_fast_dark_every_n),
            "trim_pct": float(cfg.ultra_fast_trim_pct) / 100.0,
            "k_trigger_count": k_trigger_count,
            "nplc": nplc,
        }

    def cleanup_instruments(self, pm, smu, ctx: MeasurementContext) -> None:
        try:
            pm.abort()
        except Exception:
            pass
        # Wait for any background diagnostic threads to finish.
        for t in self._diag_threads:
            t.join(timeout=10.0)
        self._diag_threads.clear()

    # ------------------------------------------------------------------
    # Chunk acquisition
    # ------------------------------------------------------------------
    def measure_chunk(
        self,
        pm,
        smu,
        smile_dev,
        pixels,
        bit_val,
        nvled_voltages,  # unused — Ultra-Fast does not sweep NVLED
        data_buffer,
        instr,
        ctx,
    ) -> int:
        """Fire pixels dynamically until the PM buffer window expires.

        Returns the number of pixels actually consumed from ``pixels``.
        """
        cfg = ctx.cfg
        profiling = ctx.profiling
        start_time = ctx.start_time

        delta_t_us = instr["delta_t_us"]
        delta_t_s = instr["delta_t_s"]
        chunk_duration_ms = instr["chunk_duration_ms"]
        chunk_duration_s = chunk_duration_ms / 1000.0
        n_samples = instr["n_samples"]
        dark_every_n = max(0, int(instr["dark_every_n"]))
        trim_pct = max(0.0, min(0.45, float(instr["trim_pct"])))
        k_trigger_count = instr["k_trigger_count"]

        # --- Arm instruments ------------------------------------------
        pm.abort()
        smu.clear_buffers_only()
        pm.configure_array_mode(chunk_duration_ms, delta_t_us)
        smu.configure_hardware_trigger(k_trigger_count)

        # --- Start capture --------------------------------------------
        T_ref = time.perf_counter()
        pm.start_array()
        T_after_pm = time.perf_counter()
        smu.start_hardware_trigger()
        T_after_smu = time.perf_counter()

        pm_start_latency_ms = (T_after_pm - T_ref) * 1000.0
        smu_start_latency_ms = (T_after_smu - T_after_pm) * 1000.0
        smu_offset_s = (T_after_pm + T_after_smu) / 2.0 - T_ref

        # --- Fire pixels dynamically until deadline --------------------
        # After each FPGA ACK we wait a fixed _FPGA_SETTLE_MS before
        # sending the next command.  The ACK marks when the display
        # update begins; the settle ensures it finishes before we
        # change the pixel again.  A fixed post-ACK delay is more
        # robust than a minimum-loop-period approach because USB
        # jitter in the ACK time doesn't eat into the settle window.
        settle_s = _FPGA_SETTLE_MS / 1000.0
        # Estimate per-event wall-clock cost for deadline calculation:
        # ACK round-trip (~2.5 ms typical) + settle.
        est_event_s = 0.003 + settle_s
        # Budget needed for: (optionally) 1 dark + 1 pixel + trailing blank
        events_for_one_more = 2 + (1 if dark_every_n > 0 else 0)
        deadline_s = chunk_duration_s - events_for_one_more * est_event_s

        def _fire_event(kind, lx, ly):
            """Send one FPGA command, record ACK, wait for settle."""
            if kind == "dark":
                t_ack = (
                    smile_dev.blank_frame()
                    if smile_dev is not None
                    else time.perf_counter()
                )
            else:
                t_ack = (
                    smile_dev.set_pixel(lx, ly, bit_val)
                    if smile_dev is not None
                    else time.perf_counter()
                )
            ack_entries.append((kind, lx, ly, t_ack - T_ref))
            time.sleep(settle_s)

        ack_entries: list[tuple[str, int, int, float]] = []

        # Leading blank — gives a clean dark reference at the start.
        _fire_event("dark", -1, -1)

        n_pixels_fired = 0
        for i, (lx, ly) in enumerate(pixels):
            if time.perf_counter() - T_ref >= deadline_s:
                break
            if not ctx.is_running():
                break
            # Interleave dark before this pixel if needed
            if (
                dark_every_n > 0
                and n_pixels_fired > 0
                and n_pixels_fired % dark_every_n == 0
            ):
                _fire_event("dark", -1, -1)
                if time.perf_counter() - T_ref >= deadline_s:
                    break
            _fire_event("pixel", int(lx), int(ly))
            n_pixels_fired += 1

        # Trailing blank — bounds the last pixel's window.
        _fire_event("dark", -1, -1)

        t_fire_end = time.perf_counter()
        profiling["img_send"].append(t_fire_end - T_ref)
        profiling["img_gen"].append(0.0)

        # --- Wait for PM400 capture to complete -----------------------
        # Sleep for the bulk of the remaining window, then use
        # poll_array_complete() to confirm the PM400 buffer is full.
        # This is critical at higher delta_t (long chunks) where a
        # blind sleep + abort would race against the capture and
        # produce empty/partial buffers.
        elapsed = time.perf_counter() - T_ref
        remaining = chunk_duration_ms / 1000.0 - elapsed
        if remaining > 0.01:
            time.sleep(remaining - 0.01)  # sleep most of it
        profiling["remaining"].append(remaining)

        # Poll until the capture truly finishes. Generous timeout: the
        # capture should complete within a few hundred ms of the nominal
        # window. If it doesn't, something is wrong but we still fetch
        # whatever we can.
        poll_ok = pm.poll_array_complete(timeout_s=max(2.0, remaining + 1.0))
        if not poll_ok:
            warn_msg = (
                f"WARNING: PM400 array capture did not complete within "
                f"timeout (chunk #{self._chunk_idx}). Data may be partial."
            )
            ctx.log(warn_msg)
            self._append_log(ctx, warn_msg)

        smu.abort_trigger()
        pm.abort()

        # --- Fetch data -----------------------------------------------
        t0 = time.perf_counter()
        pm_arr = pm.fetch_array(n_samples)
        profiling["pm_arm_to_fetch"].append(time.perf_counter() - t0)

        t0 = time.perf_counter()
        vled_arr, nvled_arr, ta_arr, _ = smu.read_buffer_with_timestamps()
        profiling["k_arm_to_fetch"].append(time.perf_counter() - t0)

        pm_arr_np = np.asarray(pm_arr, dtype=np.float64)
        vled_np = np.asarray(vled_arr, dtype=np.float64)
        nvled_np = np.asarray(nvled_arr, dtype=np.float64)

        log_msg = (
            f"Chunk #{self._chunk_idx} bv={bit_val}: "
            f"pixels={n_pixels_fired}, "
            f"PM samples={len(pm_arr)}/{n_samples}, "
            f"K samples={len(vled_arr)}, "
            f"events={len(ack_entries)}, "
            f"poll_ok={poll_ok}, "
            f"pm_start={pm_start_latency_ms:.1f}ms, "
            f"smu_start={smu_start_latency_ms:.1f}ms"
        )
        ctx.log(log_msg)
        self._append_log(ctx, log_msg)

        # Align SMU timestamps to PM-array time base. Keithley
        # internal timestamps start at 0 when trigger.initiate()
        # fires, which happened smu_offset_s after T_ref.
        if len(ta_arr) > 0:
            ta_np = np.asarray(ta_arr, dtype=np.float64) + smu_offset_s
        else:
            ta_np = np.zeros(0, dtype=np.float64)

        chunk_end_s = time.perf_counter() - T_ref  # actual elapsed

        # --- Diagnostics: save raw buffers + plot (background) ---------
        if cfg.ultra_fast_save_diag:
            t = threading.Thread(
                target=self._save_diagnostics,
                args=(
                    ctx, bit_val, delta_t_s, n_samples,
                    pm_arr_np.copy(), vled_np.copy(), nvled_np.copy(),
                    ta_np.copy(), list(ack_entries), chunk_end_s, trim_pct,
                ),
                daemon=True,
            )
            t.start()
            self._diag_threads.append(t)

        # --- First pass: compute dark means for interpolation --------
        dark_means: list[tuple[float, float]] = []  # (t_center_s, mean)
        for i, (kind, _lx, _ly, t_ack_s) in enumerate(ack_entries):
            if kind != "dark":
                continue
            t_next_s = (
                ack_entries[i + 1][3] if i + 1 < len(ack_entries) else chunk_end_s
            )
            lo, hi = self._trimmed_window(t_ack_s, t_next_s, trim_pct)
            if hi <= lo:
                continue
            j_lo, j_hi = self._pm_index_slice(lo, hi, delta_t_s, len(pm_arr_np))
            if j_hi <= j_lo:
                continue
            dark_slice = pm_arr_np[j_lo:j_hi]
            if dark_slice.size == 0:
                continue
            dark_means.append(((t_ack_s + t_next_s) / 2.0, float(dark_slice.mean())))

        def interp_dark(t_s: float) -> float:
            if not dark_means:
                return 0.0
            if len(dark_means) == 1:
                return dark_means[0][1]
            if t_s <= dark_means[0][0]:
                return dark_means[0][1]
            if t_s >= dark_means[-1][0]:
                return dark_means[-1][1]
            for k in range(len(dark_means) - 1):
                t_a, v_a = dark_means[k]
                t_b, v_b = dark_means[k + 1]
                if t_a <= t_s <= t_b:
                    if t_b > t_a:
                        alpha = (t_s - t_a) / (t_b - t_a)
                    else:
                        alpha = 0.0
                    return v_a + alpha * (v_b - v_a)
            return dark_means[-1][1]

        # --- Second pass: per-pixel rows ------------------------------
        nv_vol = cfg.nvled_voltage
        apply_dark = bool(cfg.dark_acq) and bool(dark_means)

        for i, (kind, lx, ly, t_ack_s) in enumerate(ack_entries):
            if kind != "pixel":
                continue
            t_next_s = (
                ack_entries[i + 1][3] if i + 1 < len(ack_entries) else chunk_end_s
            )
            lo, hi = self._trimmed_window(t_ack_s, t_next_s, trim_pct)
            if hi <= lo:
                pm_mean_raw = float("nan")
                pm_std = float("nan")
                vl_mean = vl_std = float("nan")
                nv_mean = nv_std = float("nan")
            else:
                j_lo, j_hi = self._pm_index_slice(lo, hi, delta_t_s, len(pm_arr_np))
                if j_hi > j_lo:
                    pm_slice = pm_arr_np[j_lo:j_hi]
                    pm_mean_raw = float(pm_slice.mean())
                    pm_std = float(pm_slice.std())
                else:
                    pm_mean_raw = float("nan")
                    pm_std = float("nan")

                if ta_np.size > 0 and vled_np.size > 0:
                    k_mask = (ta_np >= lo) & (ta_np <= hi)
                    vled_slice = vled_np[k_mask[: vled_np.size]]
                    nvled_slice = nvled_np[k_mask[: nvled_np.size]]
                else:
                    vled_slice = np.zeros(0)
                    nvled_slice = np.zeros(0)

                if vled_slice.size > 0:
                    vl_mean = float(vled_slice.mean())
                    vl_std = float(vled_slice.std())
                else:
                    vl_mean = vl_std = float("nan")
                if nvled_slice.size > 0:
                    nv_mean = float(nvled_slice.mean())
                    nv_std = float(nvled_slice.std())
                else:
                    nv_mean = nv_std = float("nan")

            t_center_s = (t_ack_s + t_next_s) / 2.0
            if apply_dark:
                pm_mean = pm_mean_raw - interp_dark(t_center_s)
            else:
                pm_mean = pm_mean_raw

            t_anchor = round((T_ref + t_ack_s) - start_time, 6)
            data_buffer.extend(
                [
                    [
                        lx,
                        ly,
                        bit_val,
                        nv_vol,
                        t_anchor,
                        "PM400",
                        f"{pm_mean:.6e}",
                        f"{pm_std:.6e}",
                    ],
                    [
                        lx,
                        ly,
                        bit_val,
                        nv_vol,
                        t_anchor,
                        "VLED",
                        f"{vl_mean:.6e}",
                        f"{vl_std:.6e}",
                    ],
                    [
                        lx,
                        ly,
                        bit_val,
                        nv_vol,
                        t_anchor,
                        "NVLED",
                        f"{nv_mean:.6e}",
                        f"{nv_std:.6e}",
                    ],
                ]
            )

        # --- Emit DARK_PM400 reference rows (if requested) ------------
        if cfg.dark_acq and dark_means:
            for t_center_s, dark_val in dark_means:
                t_anchor = round((T_ref + t_center_s) - start_time, 6)
                data_buffer.append(
                    [
                        -1,
                        -1,
                        bit_val,
                        nv_vol,
                        t_anchor,
                        "DARK_PM400",
                        f"{dark_val:.6e}",
                        "nan",
                    ]
                )

        self._chunk_idx += 1
        return n_pixels_fired

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------
    def _append_log(self, ctx: MeasurementContext, msg: str) -> None:
        """Append a line to the per-run diagnostic log file."""
        if ctx.raw_data_dir is None:
            return
        try:
            log_path = Path(ctx.raw_data_dir) / "ultra_fast_diag" / "chunk_log.txt"
            log_path.parent.mkdir(parents=True, exist_ok=True)
            with open(log_path, "a", encoding="utf-8") as f:
                f.write(f"[{time.strftime('%H:%M:%S')}] {msg}\n")
        except Exception:
            pass

    def _save_diagnostics(
        self,
        ctx,
        bit_val,
        delta_t_s,
        n_samples_expected,
        pm_arr_np,
        vled_np,
        nvled_np,
        ta_np,
        ack_entries,
        chunk_end_s,
        trim_pct,
    ):
        """Save raw chunk buffers as .npz and generate a diagnostic plot."""
        if ctx.raw_data_dir is None:
            return
        diag_dir = Path(ctx.raw_data_dir) / "ultra_fast_diag"
        try:
            diag_dir.mkdir(parents=True, exist_ok=True)
        except Exception:
            return

        tag = f"chunk{self._chunk_idx:04d}_bv{bit_val}"

        # Save raw arrays
        try:
            ack_kinds = [e[0] for e in ack_entries]
            ack_x = [e[1] for e in ack_entries]
            ack_y = [e[2] for e in ack_entries]
            ack_t = [e[3] for e in ack_entries]
            np.savez_compressed(
                diag_dir / f"{tag}.npz",
                pm_arr=pm_arr_np,
                vled_arr=vled_np,
                nvled_arr=nvled_np,
                k_timestamps=ta_np,
                ack_kinds=ack_kinds,
                ack_x=ack_x,
                ack_y=ack_y,
                ack_t=np.array(ack_t, dtype=np.float64),
                delta_t_s=delta_t_s,
                n_samples_expected=n_samples_expected,
                chunk_end_s=chunk_end_s,
                trim_pct=trim_pct,
            )
        except Exception as e:
            ctx.log(f"Diag save error ({tag}): {e}")

        # Generate plot
        try:
            self._plot_chunk(
                diag_dir / f"{tag}.png",
                pm_arr_np,
                delta_t_s,
                ack_entries,
                chunk_end_s,
                trim_pct,
                tag,
            )
        except Exception as e:
            ctx.log(f"Diag plot error ({tag}): {e}")

    @staticmethod
    def _plot_chunk(
        out_path,
        pm_arr_np,
        delta_t_s,
        ack_entries,
        chunk_end_s,
        trim_pct,
        title,
    ):
        """Render one diagnostic PNG for a chunk."""
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches

        n = len(pm_arr_np)
        if n == 0:
            return
        t_pm = np.arange(n) * delta_t_s

        fig, ax = plt.subplots(figsize=(14, 5))
        ax.plot(t_pm, pm_arr_np, linewidth=0.4, color="steelblue", label="PM400")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Power (W)")
        ax.set_title(title)

        # Draw ACK lines and trimmed windows
        for i, (kind, lx, ly, t_ack_s) in enumerate(ack_entries):
            t_next_s = (
                ack_entries[i + 1][3] if i + 1 < len(ack_entries) else chunk_end_s
            )
            color = "red" if kind == "dark" else "limegreen"
            ax.axvline(t_ack_s, color=color, linewidth=0.6, alpha=0.7)

            # Label pixel ACKs
            if kind == "pixel":
                ax.text(
                    t_ack_s,
                    ax.get_ylim()[1],
                    f"{lx},{ly}",
                    fontsize=5,
                    rotation=90,
                    va="top",
                    ha="right",
                    color="green",
                )

        # Legend
        handles = [
            mpatches.Patch(color="limegreen", label="Pixel ACK"),
            mpatches.Patch(color="red", label="Dark ACK"),
        ]
        ax.legend(handles=handles, loc="upper right", fontsize=7)
        fig.tight_layout()
        fig.savefig(str(out_path), dpi=150)
        plt.close(fig)

    # ------------------------------------------------------------------
    # Slicing helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _trimmed_window(
        t_start_s: float, t_end_s: float, trim_pct: float
    ) -> tuple[float, float]:
        """Return the ``[lo, hi)`` sub-window after trimming each edge."""
        if t_end_s <= t_start_s:
            return (t_start_s, t_start_s)
        dur = t_end_s - t_start_s
        lo = t_start_s + trim_pct * dur
        hi = t_end_s - trim_pct * dur
        if hi <= lo:
            mid = (t_start_s + t_end_s) / 2.0
            return (mid, mid)
        return (lo, hi)

    @staticmethod
    def _pm_index_slice(
        t_lo_s: float,
        t_hi_s: float,
        delta_t_s: float,
        n_samples: int,
    ) -> tuple[int, int]:
        """Map a time window in chunk-local seconds to PM400 array indices."""
        if delta_t_s <= 0 or n_samples <= 0:
            return (0, 0)
        j_lo = max(0, int(t_lo_s / delta_t_s))
        j_hi = min(n_samples, int(t_hi_s / delta_t_s))
        if j_hi <= j_lo:
            # Fall back to one sample so a pixel never silently disappears.
            j_hi = min(n_samples, j_lo + 1)
        return (j_lo, j_hi)
