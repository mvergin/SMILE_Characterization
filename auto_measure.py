"""Automated SMILE measurement script.

Runs a full characterisation sequence without the GUI:

  1. Fast Scan on ROI  -> identify alive / dead pixels, yield
  2. NVLED sweep on every Nth alive pixel (within-1-sigma group)
  3. NVLED sweep on every Nth alive pixel (outside-1-sigma group)

All data is saved in a structured folder hierarchy and post-processed
automatically.  Edit the CONFIGURATION block below, then run:

    uv run python auto_measure.py
"""

from __future__ import annotations

import datetime
import json
import random
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

# =====================================================================
#  CONFIGURATION — edit these before running
# =====================================================================

# ── Sample & output ──────────────────────────────────────────────────
SAMPLE_NAME = "AutoTest"
BASE_DIR = r"C:\Users\Georg Schötter\Nextcloud\Promotion\12 SMILE Env\smile_hssi_tubs_24112025_LabView_PC\results"

# ── ROI ──────────────────────────────────────────────────────────────
# Named shortcuts or explicit coordinates.
#   "FULL" = (0,0,511,511)   "TL" = (0,0,255,255)    "TR" = (256,0,511,255)
#   "BL"   = (0,256,255,511) "BR" = (256,256,511,511)
#   or pass a tuple:  ROI = (100, 100, 200, 200)
ROI = (50,50,55,55)

SAMPLE_NAME = f"{SAMPLE_NAME}_ROI_{str(ROI)}"
# ── Pixel selection ──────────────────────────────────────────────────
BITVALS = [15]  # list of bit-values to measure
PIXEL_ORDER = "normal"  # "normal" = row by row left-to-right
# "snake"  = alternating row direction
# "random" = shuffled
NTH_PIXEL_FASTSCAN = 1  # measure every Nth pixel in the fast scan
# (1 = all pixels, 2 = every other, etc.)
NTH_PIXEL_SWEEP = 5  # every Nth *alive* pixel is selected for
# the NVLED sweep (applied after classification)

# ── Simulation (no hardware) ────────────────────────────────────────
SIMULATE = False  # True = use simulated instruments (no VISA)

# ── Instrument VISA addresses ───────────────────────────────────────
PM400_ADDR = "USB0::0x1313::0x8075::P5006526::INSTR"  # None = auto-detect on VISA bus
SMU_ADDR = "USB0::0x05E6::0x2602::4649721::INSTR"  # None = auto-detect on VISA bus

# ── PM400 (optical power meter) ─────────────────────────────────────
PM_WAVELENGTH = 440.0  # nm — detector sensitivity correction
PM_RANGE = 1e-7  # W  — measurement range upper limit

# ── SMU — VLED (Keithley ch A, drives the micro-LED anode) ──────────
VLED_VOLTAGE = 1.8  # V  — forward bias applied to LED
#      constant throughout all phases
VLED_COMPLIANCE = 1  # A  — current compliance limit
VLED_NPLC = 0.001  # integration time in power-line cycles
#      (1 NPLC = 20 ms at 50 Hz)
VLED_RANGE_I = 10e-3  # A  — current measurement range
VLED_HIGHC = False  # high-capacitance mode (ch A only)

# ── SMU — NVLED (Keithley ch B, non-radiative recombination path) ───
NVLED_VOLTAGE_FASTSCAN = -3.2  # V  — NVLED bias during Phase 1 (fast scan).
#      The fast scan measures all pixels at
#      this single NVLED voltage to determine
#      which pixels are alive.
NVLED_COMPLIANCE = 0.5  # A  — current compliance limit
NVLED_NPLC = 0.001  # integration time in power-line cycles
NVLED_RANGE_I = 10e-3  # A  — current measurement range
NVLED_HIGHC = False  # high-capacitance mode (not recommended for ch B)

# ── NVLED sweep voltages (Phase 2 & 3) ─────────────────────────────
# Power-law spacing concentrates points near TARGET:
#   u = linspace(0, 1, N)
#   t = 1 - (1 - u) ** POWER
#   V = START + (TARGET - START) * t
# POWER = 1.0  → linear
# POWER > 1.0  → finer steps near TARGET (e.g. 2.5 hugs -3.2)
# POWER < 1.0  → finer steps near START
NVLED_SWEEP_START = -1.0  # V  — first sweep voltage
NVLED_SWEEP_TARGET = -3.2  # V  — last sweep voltage
NVLED_SWEEP_N = 30  # number of voltage steps (incl. endpoints)
NVLED_SWEEP_POWER = 2.5  # spacing exponent (see above)

# ── Sweep measurement mode ──────────────────────────────────────────
# Which acquisition mode to use during the NVLED sweep phases:
#   "Fast Scan"      — continuous PM400 + burst SMU, faster but no waveform
#   "Full Transient" — array-mode PM400 + triggered SMU, captures full
#                      turn-on transient per voltage step
SWEEP_MODE_NVLED = "Full Transient"

# Experimental hardware-paced NVLED sweep. Overrides SWEEP_MODE_NVLED
# when True: phases 2/3 use ``ExperimentalSweepMode`` — the Keithley
# walks through the voltage list via ``smub.trigger.source.listv`` with
# ``smub.source.delay`` per-step, co-timed to a single 1-second PM400
# array capture. The whole sweep must fit in the 10 000-sample PM400
# buffer (``n_voltages × (settle + nplc/50) < ~0.8 s``). Post-hoc
# splicing uses the K hardware timestamps as per-voltage boundaries.
EXPERIMENTAL_NVLED_SWEEP = True

# Save full PM400 + SMU waveforms for every Full-Transient capture.
# Off → only steady-state mean/std rows go into the CSV.
# On  → also writes per-(pixel, bitval, voltage) waveform files to
#       <run_dir>/transient_data/  (HDF5 if h5py is available, else CSV).
SAVE_TRANSIENTS = True

# ── Phase 1: Fast Scan settings ─────────────────────────────────────
# Used only for the initial alive/dead classification scan. Optimised
# for speed — short settle, few averaging points, low NPLC.
FAST_SCAN_SETTLE_MS = 5.0  # ms — wait after pixel turn-on before
#      first PM400/SMU reading
FAST_SCAN_N_PTS = 10  # PM400 + SMU sample pairs per voltage step

# ── Phase 2/3: NVLED sweep settings ────────────────────────────────
# Used only for the NVLED voltage sweeps. Generally slower / more
# averaging than the fast scan to get better per-voltage statistics.

# Common to both Fast Scan and Full Transient sweep modes:
SWEEP_VLED_NPLC = 0.1  # Keithley ch A integration time (PLCs)
SWEEP_NVLED_NPLC = 0.1  # Keithley ch B integration time (PLCs)
SWEEP_NVLED_SETTLE_MS = 50.0  # ms — wait after each NVLED voltage change

# Used when SWEEP_MODE_NVLED == "Fast Scan":
SWEEP_FAST_SCAN_SETTLE_MS = 50.0  # ms — wait after pixel turn-on
SWEEP_FAST_SCAN_N_PTS = 50  # samples per voltage step

# Used when SWEEP_MODE_NVLED == "Full Transient":
SWEEP_PRE_SETTLE_MS = 200.0  # ms — wait after SMU outputs on (thermal)
SWEEP_WINDOW_MS = 200.0  # ms — PM400 array capture window length
MIN_REMAINING_MS = 0.0  # ms — minimum time to keep capturing after
#      the nominal window (safety margin)
STEADY_TAIL_PCT = 20.0  # %  — use last N% of captured samples for
#      mean/std (steady-state extraction)

# ── Dark acquisition ────────────────────────────────────────────────
# Dark = measurement with pixel blanked (FPGA shows black frame).
# Used to subtract detector background / ambient light.
#
# Behaviour per mode:
#
# Fast Scan (with or without sweep):
#   DARK_ACQ=True  → after all voltage steps on a pixel, the pixel is
#                    blanked (TURNOFF_DIS), then one PM400 + SMU dark
#                    reading is taken.
#   DARK_SETTLE_MS → wait time (ms) between blanking the pixel and
#                    taking the dark reading. Increase if you see
#                    residual glow in dark values.
#   DARK_TAIL_MS   → not used in Fast Scan (ignored).
#   DARK_EVERY_N_SWEEP → if >0, insert additional dark measurements
#                    every N voltage steps during the sweep (in addition
#                    to the final dark). 0 = only one dark after all
#                    voltages complete.
#
# Full Transient:
#   DARK_ACQ=True + DARK_TAIL_MS=0 → after pixel blanked, PM400 drops
#                    to scalar mode for one dark reading, then re-arms
#                    array mode. DARK_SETTLE_MS applies.
#   DARK_ACQ=True + DARK_TAIL_MS>0 → PM400 capture window is extended
#                    by DARK_TAIL_MS; the extra tail samples (pixel is
#                    blanked mid-window) become the dark reference.
#                    DARK_SETTLE_MS is ignored in this case.
#   DARK_EVERY_N_SWEEP → same as Fast Scan: interleave dark readings
#                    every N voltage steps.
#
DARK_ACQ = True  # enable dark acquisition
DARK_SETTLE_MS = 5.0  # ms — wait after blank before dark reading
DARK_TAIL_MS = 0.0  # ms — dark tail extension (Transient only)
DARK_EVERY_N_SWEEP = 0  # 0 = dark only at end of pixel
# >0 = also take dark every N voltage steps
TURNOFF_DIS = True  # blank the pixel between measurement steps
# (recommended: prevents thermal buildup)

# ── Per-phase time limits (minutes) ─────────────────────────────────
# Upper bound on wall-clock duration of each phase. When the limit is
# reached the scan stops cleanly at the next pixel/bit-value boundary
# (no data loss — rows already written are kept and post-processing
# still runs). Set to 0 or None to disable the limit for that phase.
FASTSCAN_TIME_LIMIT_MIN = 60.0        # Phase 1
SWEEP_WITHIN_TIME_LIMIT_MIN = 240.0   # Phase 2a
SWEEP_OUTSIDE_TIME_LIMIT_MIN = 240.0  # Phase 2b

# ── Dead-pixel classification ───────────────────────────────────────
# After the fast scan, pixels with mean PM400 power below this fraction
# of the alive-pixel median are classified as dead.
DEAD_THRESHOLD_FRAC = 0.20  # 0.10 = less than 10% of median → dead

# ── SMU display ─────────────────────────────────────────────────────
SMU_DISPLAY_OFF = True  # disable Keithley front-panel display
# during measurement (reduces electrical noise)


# =====================================================================
#  Implementation — no edits needed below
# =====================================================================


def _parse_roi(roi):
    """Return (x1, y1, x2, y2) from the ROI constant."""
    if isinstance(roi, (tuple, list)) and len(roi) == 4:
        return tuple(int(v) for v in roi)
    named = {
        "FULL": (0, 0, 511, 511),
        "TL": (0, 0, 255, 255),
        "TR": (256, 0, 511, 255),
        "BL": (0, 256, 255, 511),
        "BR": (256, 256, 511, 511),
    }
    key = str(roi).strip().upper()
    if key not in named:
        print(f"ERROR: Unknown ROI '{roi}'. Use FULL/TL/TR/BL/BR or (x1,y1,x2,y2).")
        sys.exit(1)
    return named[key]


def _build_pixel_list(x1, y1, x2, y2, order, nth):
    """Generate pixel list with ordering and subsampling."""
    pixels = []
    for y in range(min(y1, y2), max(y1, y2) + 1):
        for x in range(min(x1, x2), max(x1, x2) + 1):
            pixels.append((x, y))

    if order == "snake":
        y_rows = defaultdict(list)
        for x, y in pixels:
            y_rows[y].append(x)
        snaked = []
        for row_idx, y_val in enumerate(sorted(y_rows)):
            row_x = sorted(y_rows[y_val])
            if row_idx % 2 == 1:
                row_x = row_x[::-1]
            snaked.extend((x, y_val) for x in row_x)
        pixels = snaked
    elif order == "random":
        random.shuffle(pixels)

    if nth > 1:
        pixels = pixels[::nth]
    return pixels


def _build_nvled_voltages(start, target, n, power):
    """Power-law NVLED sweep list — denser near TARGET when power > 1.

        u = linspace(0, 1, n)
        t = 1 - (1 - u) ** power
        V = start + (target - start) * t

    Voltages are rounded to 1 mV. Duplicate steps that collapse to the
    same rounded value are removed (keeps order, preserves endpoints).
    """
    if n < 2:
        return [round(float(start), 3)]
    u = np.linspace(0.0, 1.0, int(n))
    t = 1.0 - (1.0 - u) ** float(power)
    raw = start + (target - start) * t
    out = []
    seen = set()
    for v in raw:
        v_r = round(float(v), 3)
        if v_r not in seen:
            seen.add(v_r)
            out.append(v_r)
    return out


def _make_config(measurement_mode, nvled_sweep, nvled_voltage, **overrides):
    """Build a MeasurementConfig from the script constants.

    Phase-specific knobs (settle / n_pts / NPLC / window) are passed via
    ``**overrides`` so the fast-scan and sweep phases can use different
    values without diverging this builder.
    """
    from measurement.config import MeasurementConfig

    base = dict(
        save_dir=BASE_DIR,
        sample_name=SAMPLE_NAME,
        post_process_enabled=True,
        pixel_source="Area",
        roi_enabled=True,
        bit_values=",".join(str(b) for b in BITVALS),
        nth_pixel=1,
        sim_pm400=SIMULATE,
        sim_smu=SIMULATE,
        sim_smile=SIMULATE,
        pm_wavelength=PM_WAVELENGTH,
        pm_range=PM_RANGE,
        vled_voltage=VLED_VOLTAGE,
        vled_compliance=VLED_COMPLIANCE,
        vled_nplc=VLED_NPLC,
        vled_range_i=VLED_RANGE_I,
        vled_highc=VLED_HIGHC,
        nvled_voltage=nvled_voltage,
        nvled_compliance=NVLED_COMPLIANCE,
        nvled_nplc=NVLED_NPLC,
        nvled_range_i=NVLED_RANGE_I,
        nvled_highc=NVLED_HIGHC,
        smu_display_off=SMU_DISPLAY_OFF,
        nvled_sweep=nvled_sweep,
        nvled_sweep_target=NVLED_SWEEP_TARGET,
        nvled_sweep_step=0.0,  # unused — sweep voltages built explicitly
        nvled_settle_ms=0.0,
        measurement_mode=measurement_mode,
        pre_settle_ms=0.0,
        window_ms=50.0,
        min_remaining_ms=MIN_REMAINING_MS,
        steady_tail_pct=STEADY_TAIL_PCT,
        fast_scan_settle_ms=FAST_SCAN_SETTLE_MS,
        fast_scan_n_pts=FAST_SCAN_N_PTS,
        dark_acq=DARK_ACQ,
        dark_settle_ms=DARK_SETTLE_MS,
        dark_tail_ms=DARK_TAIL_MS,
        dark_every_n_sweep=DARK_EVERY_N_SWEEP,
        turnoff_dis=TURNOFF_DIS,
        secondary_storage_enabled=bool(SAVE_TRANSIENTS),
        secondary_storage_format="CSV",
        secondary_storage_dir="",
    )
    base.update(overrides)
    return MeasurementConfig.from_gui_dict(base)


def _auto_detect_visa(instrument_hint):
    """Auto-detect VISA address for a given instrument hint ('PM400' or '2602')."""
    try:
        import pyvisa

        rm = pyvisa.ResourceManager()
        for addr in rm.list_resources():
            try:
                dev = rm.open_resource(addr)
                dev.timeout = 3000
                dev.read_termination = "\n"
                dev.write_termination = "\n"
                idn = dev.query("*IDN?").strip()
                dev.close()
                if instrument_hint.upper() in idn.upper():
                    rm.close()
                    return addr
            except Exception:
                pass
        rm.close()
    except Exception:
        pass
    return None


def _connect_instruments(log):
    """Connect to PM400, Keithley 2602B, and SMILE FPGA (or sims)."""
    if SIMULATE:
        from instruments.sim import PM400Sim, Keithley2602BSim, SmileFPGASim

        pm = PM400Sim("SIM")
        smu = Keithley2602BSim("SIM")
        smile_dev = SmileFPGASim()
        log("All instruments in SIMULATION mode.")
    else:
        # PM400
        pm_addr = PM400_ADDR
        if pm_addr is None:
            log("Auto-detecting PM400...")
            pm_addr = _auto_detect_visa("PM400")
            if pm_addr is None:
                print("ERROR: PM400 not found. Set PM400_ADDR or enable SIMULATE.")
                sys.exit(1)
        log(f"PM400 @ {pm_addr}")
        from instruments.pm400 import PM400

        pm = PM400(pm_addr)

        # SMU
        smu_addr = SMU_ADDR
        if smu_addr is None:
            log("Auto-detecting Keithley 2602B...")
            smu_addr = _auto_detect_visa("2602")
            if smu_addr is None:
                print(
                    "ERROR: Keithley 2602B not found. Set SMU_ADDR or enable SIMULATE."
                )
                sys.exit(1)
        log(f"Keithley 2602B @ {smu_addr}")
        from instruments.keithley2602b import Keithley2602B

        smu = Keithley2602B(smu_addr)

        # SMILE FPGA
        try:
            from smile.core import Smile
            from instruments.smile_fpga import SmileFPGA

            raw_smile = Smile(timeout=5000, debug_lvl=0)
            smile_dev = SmileFPGA(raw_smile)
            log("SMILE FPGA connected.")
        except ImportError:
            from instruments.sim import SmileFPGASim

            smile_dev = SmileFPGASim()
            log("SMILE FPGA not available — using simulation.")

    # Configure PM400
    pm.set_wavelength(PM_WAVELENGTH)
    pm.set_power_unit("W")
    pm.set_auto_range(False)
    pm.set_range(PM_RANGE)
    pm.set_averaging(1)

    # Configure SMU channels
    smu.configure_channel(
        "a",
        VLED_COMPLIANCE,
        VLED_NPLC,
        VLED_HIGHC,
        zero_delays=True,
        range_i=VLED_RANGE_I,
    )
    smu.configure_channel(
        "b",
        NVLED_COMPLIANCE,
        NVLED_NPLC,
        NVLED_HIGHC,
        zero_delays=True,
        range_i=NVLED_RANGE_I,
    )
    if SMU_DISPLAY_OFF:
        smu.display_off()

    return pm, smu, smile_dev


def _run_measurement(
    pm,
    smu,
    smile_dev,
    cfg,
    pixel_list,
    run_dir,
    label,
    log,
    nvled_voltages=None,
    time_limit_s=None,
):
    """Execute one complete measurement run (fast scan or sweep).

    ``nvled_voltages`` overrides any cfg-derived sweep list. Required for
    Phase 2/3 because we use a power-law spacing built outside the cfg.

    Returns (csv_path, run_dir) on success.
    """
    from data.writer import DataWriter
    from measurement.config import MeasurementConfig
    from measurement.context import MeasurementContext
    from measurement.coordinator import ScanCoordinator
    from measurement.experimental import ExperimentalSweepMode
    from measurement.modes import FastScanMode, TransientMode

    raw_data_dir = run_dir / "raw_data"
    raw_data_dir.mkdir(parents=True, exist_ok=True)

    sec_dir = None
    if cfg.secondary_storage_enabled:
        sec_dir = run_dir / "transient_data"
        sec_dir.mkdir(parents=True, exist_ok=True)

    bit_values = [
        int(x.strip())
        for x in cfg.bit_values.split(",")
        if x.strip().isdigit() and int(x.strip()) < 16
    ]

    if nvled_voltages is None:
        nvled_voltages = [cfg.nvled_voltage]
    else:
        nvled_voltages = list(nvled_voltages)

    fname_base = label
    csv_path = raw_data_dir / f"{fname_base}.csv"

    # Save config JSON
    config_path = raw_data_dir / f"{fname_base}_config.json"
    with open(config_path, "w") as f:
        json.dump(
            {
                "settings": cfg.to_dict(),
                "script": "auto_measure.py",
                "timestamp": datetime.datetime.now().isoformat(),
            },
            f,
            indent=4,
        )

    data_writer = DataWriter(
        csv_path,
        sec_dir=sec_dir,
        hdf5_file=None,
        error_callback=lambda msg: log(f"WRITER: {msg}"),
    )
    data_writer.start()

    # Outputs on
    smu.set_voltage("a", cfg.vled_voltage)
    smu.set_voltage("b", cfg.nvled_voltage)
    smu.enable_output("a", True)
    smu.enable_output("b", True)
    log(f"Pre-settling {cfg.pre_settle_ms} ms...")
    time.sleep(cfg.pre_settle_ms / 1000.0)

    profiling = {
        "img_gen": [],
        "img_send": [],
        "remaining": [],
        "pm_arm_to_fetch": [],
        "k_arm_to_fetch": [],
        "k_setup": [],
        "k_buffer_clear": [],
        "turnoff_dis": [],
        "dark_acq": [],
    }

    t_phase_start = time.perf_counter()
    timeout_hit = {"flag": False}

    def _is_running():
        if time_limit_s is None or time_limit_s <= 0:
            return True
        elapsed = time.perf_counter() - t_phase_start
        if elapsed >= time_limit_s:
            if not timeout_hit["flag"]:
                timeout_hit["flag"] = True
                log(
                    f"TIME LIMIT reached for '{label}' "
                    f"({time_limit_s/60:.1f} min) — stopping at next pixel boundary."
                )
            return False
        return True

    ctx = MeasurementContext(
        cfg=cfg,
        data_writer=data_writer,
        profiling=profiling,
        start_time=time.perf_counter(),
        sec_dir=sec_dir,
        raw_data_dir=raw_data_dir,
        is_running=_is_running,
        log=log,
        set_eta=lambda msg: print(f"  {msg}", end="\r"),
    )

    if cfg.measurement_mode == "Fast Scan":
        mode = FastScanMode()
    elif cfg.measurement_mode == "Experimental Sweep":
        mode = ExperimentalSweepMode()
    else:
        mode = TransientMode()

    coordinator = ScanCoordinator(mode, ctx)
    budget_str = (
        f", time limit {time_limit_s/60:.1f} min"
        if time_limit_s and time_limit_s > 0
        else ""
    )
    log(
        f"Measuring {len(pixel_list)} pixels x {len(bit_values)} bitvals "
        f"x {len(nvled_voltages)} voltages ({cfg.measurement_mode}){budget_str}..."
    )
    step_count = coordinator.run(
        pm, smu, smile_dev, pixel_list, bit_values, nvled_voltages
    )

    data_writer.close()

    # Outputs off
    smu.enable_output("a", False)
    smu.enable_output("b", False)

    phase_elapsed = time.perf_counter() - t_phase_start
    status = " (TIME LIMIT)" if timeout_hit["flag"] else ""
    log(
        f"Completed {step_count} steps in "
        f"{datetime.timedelta(seconds=int(phase_elapsed))}{status}. Data: {csv_path}"
    )
    print()  # clear the \r ETA line
    return csv_path, run_dir


def _postprocess(csv_path, run_dir, log):
    """Run post-processing and return the yield report DataFrame (or None)."""
    import smile_postprocess as pp

    pp.run_postprocess(
        run_dir, do_heatmaps=True, do_ontimes=False, do_plots=False, log_fn=log
    )
    report_path = run_dir / "pm400_optical_yield_report.csv"
    if report_path.exists():
        import pandas as pd

        return pd.read_csv(report_path)
    return None


def _classify_pixels(csv_path, log):
    """Read the raw CSV, compute per-pixel mean PM400, classify into
    alive-within-1sigma, alive-outside-1sigma, and dead.

    Returns (within_1std, outside_1std) as lists of (x, y) tuples.
    """
    import pandas as pd

    df = pd.read_csv(csv_path)
    pm = df[df["TYPE"] == "PM400"].copy()
    if pm.empty:
        log("WARNING: No PM400 data found in fast scan CSV.")
        return [], []

    px_mean = pm.groupby(["X", "Y"])["MEAS_VALUE"].mean().reset_index()
    px_mean.rename(columns={"MEAS_VALUE": "POWER"}, inplace=True)
    median_power = px_mean["POWER"].median()

    # Dead: <10% of median
    alive_mask = px_mean["POWER"] >= DEAD_THRESHOLD_FRAC * median_power
    alive = px_mean[alive_mask].copy()
    dead = px_mean[~alive_mask]

    n_dead = len(dead)
    n_alive = len(alive)
    log(
        f"Classification: {n_alive} alive, {n_dead} dead "
        f"(threshold: {DEAD_THRESHOLD_FRAC * 100:.0f}% of median={median_power:.3e} W)"
    )

    if n_alive == 0:
        return [], []

    alive_mean = alive["POWER"].mean()
    alive_std = alive["POWER"].std() if n_alive > 1 else 0.0

    within_mask = (alive["POWER"] >= alive_mean - alive_std) & (
        alive["POWER"] <= alive_mean + alive_std
    )
    within = alive[within_mask]
    outside = alive[~within_mask]

    within_list = list(zip(within["X"].astype(int), within["Y"].astype(int)))
    outside_list = list(zip(outside["X"].astype(int), outside["Y"].astype(int)))

    log(
        f"  Within 1-sigma: {len(within_list)} pixels "
        f"(mean={alive_mean:.3e}, std={alive_std:.3e})"
    )
    log(f"  Outside 1-sigma: {len(outside_list)} pixels")

    return within_list, outside_list


def _subsample_pixels(pixels, nth):
    """Take every Nth pixel, preserving order."""
    if nth <= 1:
        return list(pixels)
    return list(pixels)[::nth]


# =====================================================================
#  Main
# =====================================================================


def main():
    t_start_total = time.perf_counter()

    def log(msg):
        ts = datetime.datetime.now().strftime("%H:%M:%S")
        print(f"[{ts}] {msg}")

    x1, y1, x2, y2 = _parse_roi(ROI)
    log(f"Sample: {SAMPLE_NAME}")
    log(f"ROI: ({x1},{y1}) -> ({x2},{y2})  order={PIXEL_ORDER}")
    log(f"BITVALS: {BITVALS}")

    # Top-level output directory
    timestamp = datetime.datetime.now().strftime("%y%m%d-%H%M")
    sample_dir = Path(BASE_DIR) / f"{timestamp}_{SAMPLE_NAME}"
    sample_dir.mkdir(parents=True, exist_ok=True)
    log(f"Output: {sample_dir}")

    # ── Connect instruments ──────────────────────────────────────────
    pm, smu, smile_dev = _connect_instruments(log)

    try:
        # ==============================================================
        #  PHASE 1: Fast Scan
        # ==============================================================
        log("=" * 60)
        log("PHASE 1: Fast Scan on full ROI")
        log("=" * 60)

        fastscan_dir = sample_dir / "FastScan"
        fastscan_pixels = _build_pixel_list(
            x1, y1, x2, y2, PIXEL_ORDER, NTH_PIXEL_FASTSCAN
        )
        fastscan_cfg = _make_config(
            measurement_mode="Fast Scan",
            nvled_sweep=False,
            nvled_voltage=NVLED_VOLTAGE_FASTSCAN,
            # Fast-scan Phase 1 has no transient output regardless of flag.
            secondary_storage_enabled=False,
        )

        csv_fs, _ = _run_measurement(
            pm,
            smu,
            smile_dev,
            fastscan_cfg,
            fastscan_pixels,
            fastscan_dir,
            label="FastScan",
            log=log,
            time_limit_s=(FASTSCAN_TIME_LIMIT_MIN or 0) * 60.0,
        )

        log("Post-processing fast scan...")
        _postprocess(csv_fs, fastscan_dir, log)

        # ── Classify pixels ──────────────────────────────────────────
        within_1std, outside_1std = _classify_pixels(csv_fs, log)

        # Save pixel lists for reference
        _save_pixel_list(sample_dir / "pixels_within_1std.csv", within_1std)
        _save_pixel_list(sample_dir / "pixels_outside_1std.csv", outside_1std)

        # Subsample
        sweep_within = _subsample_pixels(within_1std, NTH_PIXEL_SWEEP)
        sweep_outside = _subsample_pixels(outside_1std, NTH_PIXEL_SWEEP)
        log(
            f"NVLED sweep selection: {len(sweep_within)} within-1std, "
            f"{len(sweep_outside)} outside-1std (every {NTH_PIXEL_SWEEP}th)"
        )

        # Build NVLED sweep voltages once (power-law, denser near TARGET).
        sweep_voltages = _build_nvled_voltages(
            NVLED_SWEEP_START, NVLED_SWEEP_TARGET,
            NVLED_SWEEP_N, NVLED_SWEEP_POWER,
        )

        # Resolve which mode the sweep phases should use.
        effective_sweep_mode = (
            "Experimental Sweep" if EXPERIMENTAL_NVLED_SWEEP else SWEEP_MODE_NVLED
        )
        if EXPERIMENTAL_NVLED_SWEEP:
            from measurement.experimental import compute_sweep_timing

            k_nplc = max(SWEEP_VLED_NPLC, SWEEP_NVLED_NPLC)
            settle_s, measure_s, K, trim_pct, per_v_s, total_s = (
                compute_sweep_timing(
                    n_voltages=len(sweep_voltages),
                    user_settle_ms=SWEEP_NVLED_SETTLE_MS,
                    nplc=k_nplc,
                )
            )
            mode_tag = "zero-settle (trim 10%)" if settle_s == 0 else "auto 60/40"
            log(
                f"Experimental sweep enabled — {len(sweep_voltages)} voltages, "
                f"per-voltage {per_v_s*1000:.1f} ms [{mode_tag}]: "
                f"settle={settle_s*1000:.1f} ms, measure={measure_s*1000:.1f} ms, "
                f"K={K} samples/voltage (nplc={k_nplc}) — total ~{total_s*1000:.1f} ms "
                f"of PM400 1 s buffer."
            )
        log(
            f"NVLED voltages ({len(sweep_voltages)} pts, power={NVLED_SWEEP_POWER}): "
            f"{sweep_voltages[0]:.3f} V → {sweep_voltages[-1]:.3f} V "
            f"(first step {sweep_voltages[1]-sweep_voltages[0]:+.3f} V, "
            f"last step {sweep_voltages[-1]-sweep_voltages[-2]:+.3f} V)"
        )

        # Phase-2/3-only overrides: slower settle, more averaging,
        # higher NPLC, longer transient window than the fast-scan phase.
        sweep_overrides = dict(
            vled_nplc=SWEEP_VLED_NPLC,
            nvled_nplc=SWEEP_NVLED_NPLC,
            nvled_settle_ms=SWEEP_NVLED_SETTLE_MS,
            fast_scan_settle_ms=SWEEP_FAST_SCAN_SETTLE_MS,
            fast_scan_n_pts=SWEEP_FAST_SCAN_N_PTS,
            pre_settle_ms=SWEEP_PRE_SETTLE_MS,
            window_ms=SWEEP_WINDOW_MS,
        )

        # ==============================================================
        #  PHASE 2: NVLED Sweep — within 1-sigma
        # ==============================================================
        if sweep_within:
            log("=" * 60)
            log(f"PHASE 2a: NVLED Sweep on {len(sweep_within)} within-1-sigma pixels")
            log("=" * 60)

            sweep_within_dir = sample_dir / "NVLEDSweep" / "Within1STD"
            sweep_cfg = _make_config(
                measurement_mode=effective_sweep_mode,
                nvled_sweep=True,
                nvled_voltage=NVLED_SWEEP_START,
                **sweep_overrides,
            )
            csv_sw, _ = _run_measurement(
                pm,
                smu,
                smile_dev,
                sweep_cfg,
                sweep_within,
                sweep_within_dir,
                label="Sweep_Within1STD",
                log=log,
                nvled_voltages=sweep_voltages,
                time_limit_s=(SWEEP_WITHIN_TIME_LIMIT_MIN or 0) * 60.0,
            )
            log("Post-processing within-1std sweep...")
            _postprocess(csv_sw, sweep_within_dir, log)
        else:
            log("Skipping within-1std sweep — no pixels.")

        # ==============================================================
        #  PHASE 3: NVLED Sweep — outside 1-sigma
        # ==============================================================
        if sweep_outside:
            log("=" * 60)
            log(f"PHASE 2b: NVLED Sweep on {len(sweep_outside)} outside-1-sigma pixels")
            log("=" * 60)

            sweep_outside_dir = sample_dir / "NVLEDSweep" / "Outside1STD"
            sweep_cfg = _make_config(
                measurement_mode=effective_sweep_mode,
                nvled_sweep=True,
                nvled_voltage=NVLED_SWEEP_START,
                **sweep_overrides,
            )
            csv_so, _ = _run_measurement(
                pm,
                smu,
                smile_dev,
                sweep_cfg,
                sweep_outside,
                sweep_outside_dir,
                label="Sweep_Outside1STD",
                log=log,
                nvled_voltages=sweep_voltages,
                time_limit_s=(SWEEP_OUTSIDE_TIME_LIMIT_MIN or 0) * 60.0,
            )
            log("Post-processing outside-1std sweep...")
            _postprocess(csv_so, sweep_outside_dir, log)
        else:
            log("Skipping outside-1std sweep — no pixels.")

        # ==============================================================
        #  Summary
        # ==============================================================
        elapsed = time.perf_counter() - t_start_total
        log("=" * 60)
        log(f"ALL DONE in {datetime.timedelta(seconds=int(elapsed))}")
        log(f"Results: {sample_dir}")
        log("=" * 60)

    finally:
        # Always turn off and close
        try:
            smu.enable_output("a", False)
            smu.enable_output("b", False)
        except Exception:
            pass
        try:
            if SMU_DISPLAY_OFF:
                smu.display_on()
        except Exception:
            pass
        try:
            pm.close()
        except Exception:
            pass
        try:
            smu.close()
        except Exception:
            pass
        try:
            smile_dev.close()
        except Exception:
            pass


def _save_pixel_list(path, pixels):
    """Save a pixel list as a simple CSV."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        f.write("X,Y\n")
        for x, y in pixels:
            f.write(f"{x},{y}\n")


if __name__ == "__main__":
    main()
