# 16.03.2026
# v9 - Layout: sample info + hardware setup side by side, strategy + device config side by side,
#      ROI spinboxes vertical; buffer acq + secondary storage moved into sample info box;
#      profiling: add remaining sleep + k_setup; hoist smu setup outside pixel loop

GUI_VERSION = "20260415_v1"

import sys
import time
import os
import csv
import json
import datetime
import random
import struct
import numpy as np
from pathlib import Path
from PyQt6.QtWidgets import (
    QApplication,
    QMainWindow,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QPushButton,
    QLabel,
    QSpinBox,
    QComboBox,
    QGroupBox,
    QMessageBox,
    QFileDialog,
    QLineEdit,
    QCheckBox,
    QFormLayout,
    QStatusBar,
    QDoubleSpinBox,
    QTabWidget,
    QScrollArea,
    QGridLayout,
)
from PyQt6.QtCore import Qt, QThread, QTimer, pyqtSignal

try:
    import h5py

    HDF5_AVAILABLE = True
except ImportError:
    HDF5_AVAILABLE = False

# --- HARDWARE IMPORTS ---
try:
    from smile.core import Smile
    from smile.utils.display import send_image
    from smile.utils.display import set_pixel
    from smile.utils.image_processor import ImageProcessor

    SMILE_AVAILABLE = True
except ImportError:
    SMILE_AVAILABLE = False

try:
    from instrumentlib import PM400 as _PM400Real, Keithley2602B as _Keithley2602BReal

    DRIVERS_AVAILABLE = True
except ImportError:
    _PM400Real = None
    _Keithley2602BReal = None
    DRIVERS_AVAILABLE = False


# =============================================================================
# No-scroll spinbox / combobox subclasses
# Prevents accidental value changes when scrolling over the control panel.
# =============================================================================


class NoWheelSpinBox(QSpinBox):
    def wheelEvent(self, e):
        e.ignore()


class NoWheelDoubleSpinBox(QDoubleSpinBox):
    def wheelEvent(self, e):
        e.ignore()


class NoWheelComboBox(QComboBox):
    def wheelEvent(self, e):
        e.ignore()


# =============================================================================
# Simulation mock classes (always available regardless of import success)
# =============================================================================


from instruments.sim import PM400Sim as _PM400Sim
from instruments.sim import Keithley2602BSim as _Keithley2602BSim
from instruments.sim import SmileFPGASim as _SmileFPGASim


# =============================================================================
# Helper Functions
# =============================================================================


def steady_state_mean(arr, tail_pct=20.0):
    """Returns mean of the last tail_pct% of arr as a formatted string."""
    if not arr:
        return float("nan")
    n = max(1, int(len(arr) * tail_pct / 100.0))
    v = float(np.mean(arr[-n:]))
    return f"{v:.6e}"


def generate_timing_ratio_matrix(timing_json_path, output_txt_path=None):
    if not os.path.exists(timing_json_path):
        print(f"Error: {timing_json_path} not found.")
        return
    with open(timing_json_path, "r") as f:
        timing_data = json.load(f)
    keys = [
        k for k in timing_data if timing_data[k] is not None and "avg" in timing_data[k]
    ]
    if not keys:
        print("No valid timing data to compare.")
        return
    col_width = max([len(k) for k in keys] + [8]) + 2
    lines = [
        "Pairwise Timing Ratio Matrix",
        "Formula: [Column Avg Time] / [Row Avg Time]",
        "Reading Example: 'Row takes X times less/more than Column'",
        "-" * (col_width * (len(keys) + 1)),
        "".ljust(col_width) + "".join([k.rjust(col_width) for k in keys]),
    ]
    for row_key in keys:
        row_str = row_key.ljust(col_width)
        row_avg = timing_data[row_key]["avg"]
        for col_key in keys:
            col_avg = timing_data[col_key]["avg"]
            if row_avg == 0:
                row_str += "INF".rjust(col_width)
            else:
                row_str += f"{col_avg / row_avg:.2f}".rjust(col_width)
        lines.append(row_str)
    if output_txt_path is None:
        output_txt_path = os.path.join(
            os.path.dirname(timing_json_path), "timing_ratio_matrix.txt"
        )
    with open(output_txt_path, "w") as f:
        f.write("\n".join(lines))
    print(f"Timing ratio matrix saved to: {output_txt_path}")


# =============================================================================
# Array Map Widget
# =============================================================================


class ArrayMapWidget(QLabel):
    def __init__(self):
        super().__init__()
        self.setFixedSize(512, 512)
        self.setStyleSheet("background-color: black;")
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.pixel_status = np.zeros((512, 512), dtype=np.uint8)
        self.image = None
        self._dirty = False
        self._refresh_timer = QTimer(self)
        self._refresh_timer.setInterval(33)  # ~30 fps cap
        self._refresh_timer.timeout.connect(self._flush_if_dirty)
        self._refresh_timer.start()
        self.update_view()

    def set_pixel_status(self, x, y, status):
        if 0 <= x < 512 and 0 <= y < 512:
            self.pixel_status[y, x] = status
            self._dirty = True

    def reset_map(self):
        self.pixel_status.fill(0)
        self.update_view()

    def _flush_if_dirty(self):
        if self._dirty:
            self.update_view()
            self._dirty = False

    def update_view(self):
        from PyQt6.QtGui import QPixmap, QImage

        buffer = np.zeros((512, 512, 4), dtype=np.uint8)
        buffer[:, :, 3] = 255
        buffer[self.pixel_status == 1, 0] = 255
        buffer[self.pixel_status == 2, 1] = 255
        qim = QImage(buffer.data, 512, 512, QImage.Format.Format_RGBA8888)
        self.setPixmap(QPixmap.fromImage(qim))


import smile_postprocess as _postprocess

ACCEPTABLE_VERSIONS = {
    "instrumentlib": {"20260415_v1"},
    "smile_postprocess": {"20260415_v1"},
    "gui": {"20260415_v1"},
}


def _write_readme(run_dir, cfg, versions):
    """Write a concise human-readable README.txt to the run root folder."""
    lines = []
    now_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
    lines += [
        "SMILE Measurement Run Summary",
        "=" * 40,
        f"Date/time   : {now_str}",
        f"Sample name : {cfg.get('sample_name', '(unnamed)')}",
        "",
        "--- Software versions ---",
        f"  GUI            : {versions.get('gui', 'unknown')}",
        f"  instrumentlib  : {versions.get('instrumentlib', 'unknown')}",
        f"  smile_postprocess : {versions.get('smile_postprocess', 'unknown')}",
        "",
        "--- Measurement mode ---",
        f"  Mode         : {cfg.get('measurement_mode', '?')}",
    ]

    mode = cfg.get("measurement_mode", "")
    if mode == "Full Transient":
        lines += [
            f"  Window       : {cfg.get('window_ms', '?')} ms",
            f"  Min remain   : {cfg.get('min_remaining_ms', '?')} ms",
        ]
        if cfg.get("dark_acq"):
            lines += [
                f"  Dark acq     : yes  (tail = {cfg.get('dark_tail_ms', 0)} ms, "
                f"settle = {cfg.get('dark_settle_ms', 0)} ms)",
            ]
        else:
            lines.append("  Dark acq     : no")
    elif mode == "Fast Scan":
        lines.append(f"  Scan settle  : {cfg.get('scan_settle_ms', '?')} ms")

    lines += [
        "",
        "--- Pixel / scan region ---",
    ]
    pixel_src = cfg.get("pixel_source", "Grid")
    if pixel_src == "CSV":
        lines.append(f"  Pixel source : CSV ({cfg.get('csv_file_path', '?')})")
    else:
        lines.append(f"  Quadrant     : {cfg.get('quadrant', '?')}")
        if cfg.get("roi_enabled"):
            lines.append(
                f"  ROI          : ({cfg['roi_x1']},{cfg['roi_y1']}) → "
                f"({cfg['roi_x2']},{cfg['roi_y2']})  "
                f"step ({cfg.get('roi_step_x', 1)},{cfg.get('roi_step_y', 1)})"
            )
        else:
            lines.append(
                f"  Pixel step   : ({cfg.get('step_x', 1)},{cfg.get('step_y', 1)})"
            )

    bit_values = cfg.get("bit_values", "?")
    lines += [
        "",
        "--- Bias conditions ---",
        f"  Bit values   : {bit_values}",
        f"  VLED         : {cfg.get('vled_voltage', '?')} V  "
        f"(compliance {cfg.get('vled_compliance', '?')} A)",
    ]
    if cfg.get("nvled_sweep"):
        lines.append(
            f"  NVLED sweep  : {cfg.get('nvled_voltage', '?')} V → "
            f"{cfg.get('nvled_sweep_target', '?')} V  "
            f"step {cfg.get('nvled_sweep_step', '?')} V  "
            f"settle {cfg.get('nvled_settle_ms', '?')} ms"
        )
    else:
        lines.append(f"  NVLED        : {cfg.get('nvled_voltage', '?')} V (fixed)")

    lines += [
        "",
        "--- Data storage ---",
        f"  Repeats      : {cfg.get('n_repeats', 1)}",
    ]
    if cfg.get("secondary_storage_enabled"):
        fmt = cfg.get("secondary_storage_format", "CSV")
        lines.append(f"  Transient data : yes ({fmt})")
        if cfg.get("plot_transients"):
            lines.append("  Transient plots: yes")
        else:
            lines.append("  Transient plots: no")
    else:
        lines.append("  Transient data : no")

    if cfg.get("turnoff_dis"):
        lines.append("  Turnoff between pixels : yes (blank frame)")
    else:
        lines.append("  Turnoff between pixels : no")

    lines += ["", "(Full settings in raw_data/*_config.json)"]

    readme_path = Path(run_dir) / "README.txt"
    with open(readme_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def _check_versions():
    import instrumentlib as _il

    warnings = []
    if getattr(_il, "VERSION", None) not in ACCEPTABLE_VERSIONS["instrumentlib"]:
        warnings.append(
            f"instrumentlib version '{getattr(_il, 'VERSION', 'MISSING')}' not in accepted set {ACCEPTABLE_VERSIONS['instrumentlib']}"
        )
    if (
        getattr(_postprocess, "VERSION", None)
        not in ACCEPTABLE_VERSIONS["smile_postprocess"]
    ):
        warnings.append(
            f"smile_postprocess version '{getattr(_postprocess, 'VERSION', 'MISSING')}' not in accepted set {ACCEPTABLE_VERSIONS['smile_postprocess']}"
        )
    return warnings


# =============================================================================
# Measurement Worker
# =============================================================================


class MeasurementWorker(QThread):
    log_msg = pyqtSignal(str)
    pixel_update = pyqtSignal(int, int, int)
    eta_update = pyqtSignal(str)
    finished = pyqtSignal()
    error = pyqtSignal(str)

    def __init__(self, setup_config, device_addresses):
        super().__init__()
        from measurement.config import MeasurementConfig

        if isinstance(setup_config, dict):
            self.cfg = MeasurementConfig.from_gui_dict(setup_config)
        else:
            self.cfg = setup_config
        self.addrs = device_addresses
        self.is_running = True
        self._last_pixel_emit_t = 0.0
        self._last_eta_emit_t = 0.0

    # Emit pixel-in-progress update at most ~20 fps; completions always pass through.
    def _emit_pixel_start(self, x, y):
        now = time.perf_counter()
        if now - self._last_pixel_emit_t >= 0.05:
            self.pixel_update.emit(x, y, 1)
            self._last_pixel_emit_t = now

    # Emit ETA/status bar update at most ~10 fps.
    def _emit_eta(self, msg):
        now = time.perf_counter()
        if now - self._last_eta_emit_t >= 0.1:
            self.eta_update.emit(msg)
            self._last_eta_emit_t = now

    def load_pixels_from_csv(self, path):
        import pandas as pd

        df = pd.read_csv(path)
        pixels = df[["X", "Y"]].drop_duplicates().values.tolist()
        return [(int(p[0]), int(p[1])) for p in pixels]

    def generate_pixel_sequence(self):
        if self.cfg.roi_enabled:
            x1, y1 = self.cfg.roi_x1, self.cfg.roi_y1
            x2, y2 = self.cfg.roi_x2, self.cfg.roi_y2
            ranges = [((min(x1, x2), max(x1, x2) + 1), (min(y1, y2), max(y1, y2) + 1))]
        else:
            quad = self.cfg.quadrant
            ranges = []
            if quad == "TL":
                ranges.append(((0, 256), (0, 256)))
            elif quad == "TR":
                ranges.append(((256, 512), (0, 256)))
            elif quad == "BL":
                ranges.append(((0, 256), (256, 512)))
            elif quad == "BR":
                ranges.append(((256, 512), (256, 512)))
            elif quad == "Full":
                ranges.append(((0, 512), (0, 512)))

        pixels = []
        for x_range, y_range in ranges:
            for y in range(y_range[0], y_range[1]):
                for x in range(x_range[0], x_range[1]):
                    pixels.append((x, y))

        if self.cfg.snake_scan:
            y_rows = {}
            for x, y in pixels:
                y_rows.setdefault(y, []).append(x)
            snaked = []
            for row_idx, y_val in enumerate(sorted(y_rows)):
                row_x = sorted(y_rows[y_val])
                if row_idx % 2 == 1:
                    row_x = row_x[::-1]
                snaked.extend((x, y_val) for x in row_x)
            pixels = snaked

        if self.cfg.random_mode:
            random.shuffle(pixels)
        nth = self.cfg.nth_pixel
        if nth > 1:
            pixels = pixels[::nth]
        return pixels

    def save_profiling(self, profiling_data, path):
        stats = {}
        path = Path(path)
        base_dir = path.parent
        for k, v in profiling_data.items():
            if len(v) > 0:
                stats[k] = {
                    "min": float(np.min(v)),
                    "max": float(np.max(v)),
                    "avg": float(np.mean(v)),
                    "std": float(np.std(v)),
                    "count": len(v),
                }
                csv_path = base_dir / f"timing_{k}.csv"
                with open(csv_path, "w", newline="") as csv_file:
                    writer = csv.writer(csv_file)
                    writer.writerow([k])
                    writer.writerows([[val] for val in v])
            else:
                stats[k] = None
        with open(path, "w") as f:
            json.dump(stats, f, indent=4)

    # ------------------------------------------------------------------
    # Instrument connection & setup helpers
    # ------------------------------------------------------------------

    def _connect_devices(self):
        self.log_msg.emit("Connecting to devices...")
        PMClass = (
            _PM400Sim if (self.cfg.sim_pm400 or not DRIVERS_AVAILABLE) else _PM400Real
        )
        SMUClass = (
            _Keithley2602BSim
            if (self.cfg.sim_smu or not DRIVERS_AVAILABLE)
            else _Keithley2602BReal
        )
        use_smile = SMILE_AVAILABLE and not self.cfg.sim_smile

        pm = PMClass(self.addrs["pm400"])
        pm.set_wavelength(self.cfg.pm_wavelength)
        pm.set_power_unit("W")
        pm.set_auto_range(False)
        pm.set_range(self.cfg.pm_range)
        pm.set_averaging(1)

        smu = SMUClass(self.addrs["smu"])
        smu.configure_channel(
            "a",
            self.cfg.vled_compliance,
            self.cfg.vled_nplc,
            self.cfg.vled_highc,
            zero_delays=True,
            range_i=self.cfg.vled_range_i,
        )
        smu.configure_channel(
            "b",
            self.cfg.nvled_compliance,
            self.cfg.nvled_nplc,
            self.cfg.nvled_highc,
            zero_delays=True,
            range_i=self.cfg.nvled_range_i,
        )

        from instruments.smile_fpga import SmileFPGA

        if use_smile:
            raw_smile = Smile(timeout=5000, debug_lvl=0)
            smile_dev = SmileFPGA(raw_smile)
        else:
            self.log_msg.emit("[SIM] SMILE platform simulated.")
            smile_dev = _SmileFPGASim()

        if self.cfg.smu_display_off:
            self.log_msg.emit("Disabling Keithley display...")
            smu.display_off()
        return pm, smu, smile_dev

    def _setup_run_dir(self, pm, smu):
        timestamp = datetime.datetime.now().strftime("%y%m%d-%H%M")
        safe_name = "".join(
            c for c in self.cfg.sample_name if c.isalnum() or c in (" ", "_", "-")
        ).strip()
        folder_name = (
            f"{timestamp}_{safe_name}" if safe_name else f"{timestamp}_Measurement"
        )
        base_dir = Path(self.cfg.save_dir)
        run_dir = base_dir / folder_name
        raw_data_dir = run_dir / "raw_data"
        raw_data_dir.mkdir(parents=True, exist_ok=True)

        sec_dir = None
        hdf5_file = None
        if self.cfg.secondary_storage_enabled:
            sec_dir = (
                Path(self.cfg.secondary_storage_dir) / folder_name
                if self.cfg.secondary_storage_dir.strip()
                else run_dir / "transient_data"
            )
            sec_dir.mkdir(parents=True, exist_ok=True)
            if self.cfg.secondary_storage_format == "HDF5" and HDF5_AVAILABLE:
                hdf5_file = h5py.File(str(sec_dir / "transients.h5"), "w")

        if self.cfg.pixel_source == "CSV":
            pixel_list = self.load_pixels_from_csv(self.cfg.csv_file_path)
            fname_base = "meas_CSV_loaded"
        else:
            pixel_list = self.generate_pixel_sequence()
            if self.cfg.roi_enabled:
                fname_base = (
                    f"meas_{safe_name}_ROI_"
                    f"{self.cfg.roi_x1},{self.cfg.roi_y1}_"
                    f"{self.cfg.roi_x2},{self.cfg.roi_y2}"
                )
            else:
                fname_base = f"meas_{safe_name}_{self.cfg.quadrant}"

        import instrumentlib as _il

        versions = {
            "gui": GUI_VERSION,
            "instrumentlib": getattr(_il, "VERSION", "unknown"),
            "smile_postprocess": getattr(_postprocess, "VERSION", "unknown"),
        }
        cfg_dict = self.cfg.to_dict()
        config_path = raw_data_dir / f"{fname_base}_config.json"
        with open(config_path, "w") as f:
            json.dump(
                {
                    "settings": cfg_dict,
                    "addresses": self.addrs,
                    "pm400_config": pm.get_config_dict(),
                    "smu_config": smu.get_config_dict(),
                    "versions": versions,
                },
                f,
                indent=4,
            )
        _write_readme(run_dir, cfg_dict, versions)
        return run_dir, raw_data_dir, sec_dir, hdf5_file, fname_base, pixel_list

    def _build_nvled_voltages(self):
        if self.cfg.nvled_sweep:
            start_v = self.cfg.nvled_voltage
            target_v = self.cfg.nvled_sweep_target
            step_v = abs(self.cfg.nvled_sweep_step) * (1 if target_v > start_v else -1)
            return np.arange(start_v, target_v + step_v * 0.5, step_v).tolist()
        return [self.cfg.nvled_voltage]

    def _empty_profiling(self):
        return {
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

    # ------------------------------------------------------------------
    # Post-processing & cleanup
    # ------------------------------------------------------------------

    @staticmethod
    def _read_pm400_waveform(fpath):
        """Return sorted (time_s, power_W) list of PM400 rows from a secondary CSV."""
        rows = []
        with open(fpath, newline="") as f:
            reader = csv.reader(f)
            header = next(reader)
            t_idx = header.index("TIME_s")
            type_idx = header.index("TYPE")
            val_idx = header.index("VALUE")
            for row in reader:
                if row[type_idx] == "PM400":
                    rows.append((float(row[t_idx]), float(row[val_idx])))
        rows.sort(key=lambda r: r[0])
        return rows

    def _plot_transient_arrays(self, sec_dir, run_dir):
        """Save one PNG per secondary CSV file showing the PM400 waveform."""
        try:
            import matplotlib

            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except ImportError as e:
            msg = (
                f"Plot transients skipped — matplotlib not installed: {e}\n"
                f"  Install with: pip install matplotlib"
            )
            print(msg)
            self.log_msg.emit(msg)
            return

        import glob

        pattern = str(sec_dir / "x*_y*_b*.csv")
        files = sorted(glob.glob(pattern))
        if not files:
            self.log_msg.emit("Plot transients: no secondary CSV files found.")
            return

        n_ok = 0
        for fpath in files:
            try:
                rows = self._read_pm400_waveform(fpath)
                if not rows:
                    continue
                ts = np.array([r[0] for r in rows])
                vs = np.array([r[1] for r in rows])
                ts_ms = (ts - ts[0]) * 1000.0

                stem = Path(fpath).stem
                fig, ax = plt.subplots(figsize=(8, 4))
                ax.plot(ts_ms, vs, lw=0.8, color="tab:blue")
                ax.set_xlabel("Time (ms)")
                ax.set_ylabel("Power (W)")
                ax.set_title(f"PM400 transient — {stem.replace('_', '  ')}")
                ax.grid(True, alpha=0.3)
                fig.tight_layout()
                fig.savefig(str(sec_dir / f"{stem}.png"), dpi=120)
                plt.close(fig)
                n_ok += 1
            except Exception as e:
                self.log_msg.emit(f"Plot transients: skipping {Path(fpath).name}: {e}")

        self.log_msg.emit(f"Transient plots saved ({n_ok} PNGs) in secondary folder.")

    def _analyze_on_times(self, sec_dir, run_dir):
        """Gradient-based on-time detection across all secondary CSVs.

        For each waveform: finds the time after which dP/dt stays below 1% of
        its peak — i.e. the LED has reached steady state.  Results are saved to
        ontime_summary.csv and a recommended Scan Settle (mean + 3σ) is printed.

        Only meaningful when Turn Off Display is enabled so waveforms start from
        the noise floor (clean monotonic rise).  Without it the gradient sees the
        tail of the previous pixel and a mid-window dip during image send.
        """
        import glob

        pattern = str(sec_dir / "x*_y*_b*.csv")
        files = sorted(glob.glob(pattern))
        if not files:
            return

        if not self.cfg.turnoff_dis:
            msg = (
                "On-time analysis: results may be unreliable — "
                "'Turn Off Display After Each Pixel' is not enabled. "
                "Waveforms do not start from the noise floor."
            )
            print(msg)
            self.log_msg.emit(msg)

        results = []
        for fpath in files:
            try:
                rows = self._read_pm400_waveform(fpath)
                if len(rows) < 10:
                    continue
                ts = np.array([r[0] for r in rows])
                vs = np.array([r[1] for r in rows])
                ts_ms = (ts - ts[0]) * 1000.0

                grad = np.gradient(vs, ts_ms)
                peak_grad = grad.max()
                if peak_grad <= 0:
                    t_steady_ms = float("nan")
                else:
                    threshold = 0.01 * peak_grad
                    peak_idx = int(np.argmax(grad))
                    below = np.where(grad[peak_idx:] < threshold)[0]
                    t_steady_ms = (
                        float(ts_ms[peak_idx + below[0]])
                        if len(below)
                        else float("nan")
                    )

                results.append(
                    {"file": Path(fpath).stem, "t_steady_ms": round(t_steady_ms, 3)}
                )
            except Exception as e:
                self.log_msg.emit(f"On-time analysis: skipping {Path(fpath).name}: {e}")

        if not results:
            return

        summary_path = run_dir / "ontime_summary.csv"
        with open(summary_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["file", "t_steady_ms"])
            w.writeheader()
            w.writerows(results)

        valid = np.array(
            [r["t_steady_ms"] for r in results if not np.isnan(r["t_steady_ms"])]
        )
        if len(valid) > 0:
            mean_ms = float(valid.mean())
            std_ms = float(valid.std())
            rec = mean_ms + 3.0 * std_ms
            msg = (
                f"On-time analysis: {len(valid)}/{len(results)} pixels detected. "
                f"mean={mean_ms:.1f} ms  std={std_ms:.1f} ms  "
                f"→ recommended Scan Settle ≥ {rec:.1f} ms (mean + 3σ). "
                f"Saved: {summary_path.name}"
            )
            print(msg)
            self.log_msg.emit(msg)

    def _post_process(self, profiling, raw_data_dir, csv_path, run_dir, sec_dir=None):
        self.log_msg.emit("Saving profiling data...")
        timing_path = raw_data_dir / "timing.json"
        self.save_profiling(profiling, timing_path)
        generate_timing_ratio_matrix(str(timing_path))
        _postprocess.run_postprocess(
            run_dir,
            do_aggregate=True,
            do_heatmaps=True,
            do_ontimes=(sec_dir is not None),
            do_plots=(self.cfg.plot_transients and sec_dir is not None),
            log_fn=self.log_msg.emit,
        )

    def _cleanup(self, pm, smu, smile_dev, hdf5_file):
        self.log_msg.emit("Shutting down...")
        if hdf5_file is not None:
            try:
                hdf5_file.close()
            except Exception:
                pass
        if smu:
            try:
                smu.abort_trigger()
            except Exception:
                pass
            try:
                smu.display_on()
            except Exception:
                pass
            smu.close()
        if pm:
            try:
                pm.stop_continuous()
            except Exception:
                pass
            try:
                pm.display_on()
            except Exception:
                pass
            pm.close()
        if smile_dev:
            smile_dev.close()
        self.finished.emit()

    # ------------------------------------------------------------------
    # Main run coordinator
    # ------------------------------------------------------------------

    def run(self):
        pm = smu = smile_dev = None
        data_writer = None
        hdf5_file = None
        step_count = 0

        try:
            pm, smu, smile_dev = self._connect_devices()
            run_dir, raw_data_dir, sec_dir, hdf5_file, fname_base, pixel_list = (
                self._setup_run_dir(pm, smu)
            )

            bit_values = [
                int(x.strip())
                for x in self.cfg.bit_values.split(",")
                if x.strip().isdigit() and int(x.strip()) < 16
            ]
            nvled_voltages = self._build_nvled_voltages()
            profiling = self._empty_profiling()

            csv_path = raw_data_dir / f"{fname_base}.csv"
            from data.writer import DataWriter

            data_writer = DataWriter(
                csv_path,
                sec_dir=sec_dir,
                hdf5_file=hdf5_file,
                error_callback=self.log_msg.emit,
            )
            data_writer.start()

            self.log_msg.emit("Turning outputs ON...")
            smu.set_voltage("a", self.cfg.vled_voltage)
            smu.set_voltage("b", self.cfg.nvled_voltage)
            smu.enable_output("a", True)
            smu.enable_output("b", True)
            self.log_msg.emit(f"Pre-settling for {self.cfg.pre_settle_ms} ms...")
            time.sleep(self.cfg.pre_settle_ms / 1000.0)
            self.log_msg.emit("Measuring ...")
            start_time = time.perf_counter()

            from measurement.context import MeasurementContext
            from measurement.coordinator import ScanCoordinator
            from measurement.modes import FastScanMode, TransientMode
            from measurement.ultra_fast import UltraFastMode

            ctx = MeasurementContext(
                cfg=self.cfg,
                data_writer=data_writer,
                profiling=profiling,
                start_time=start_time,
                sec_dir=sec_dir,
                raw_data_dir=raw_data_dir,
                is_running=lambda: self.is_running,
                log=self.log_msg.emit,
                pixel_active=self._emit_pixel_start,
                pixel_done=lambda x, y: self.pixel_update.emit(x, y, 2),
                set_eta=self._emit_eta,
            )

            mode_name = self.cfg.measurement_mode
            if mode_name == "Ultra Fast":
                mode = UltraFastMode()
            elif mode_name == "Fast Scan":
                mode = FastScanMode()
            else:
                mode = TransientMode()
            coordinator = ScanCoordinator(mode, ctx)
            step_count = coordinator.run(
                pm,
                smu,
                smile_dev,
                pixel_list,
                bit_values,
                nvled_voltages,
            )

            data_writer.close()

            self.log_msg.emit("Turning outputs OFF...")
            try:
                smu.enable_output("a", False)
                smu.enable_output("b", False)
            except Exception:
                pass
            if step_count > 0 and self.cfg.post_process_enabled:
                self._post_process(profiling, raw_data_dir, csv_path, run_dir, sec_dir)

            self.log_msg.emit("Measurement Finished.")

        except Exception as e:
            self.error.emit(str(e))
        finally:
            if data_writer is not None:
                try:
                    data_writer.close()
                except Exception:
                    pass
            self._cleanup(pm, smu, smile_dev, hdf5_file)

    def stop(self):
        self.is_running = False


# =============================================================================
# GUI
# =============================================================================


class MicroLEDCharGUI(QMainWindow):
    CONFIG_FILE = "gui_config_v9.json"

    def __init__(self):
        super().__init__()
        self.setWindowTitle("SMILE MicroLED Characterization")
        self.resize(1500, 1000)
        self.worker = None
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.init_ui()
        self.refresh_resources()
        self.load_gui_state()

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------
    def init_ui(self):
        main_widget = QWidget()
        main_layout = QHBoxLayout()

        left_layout = QVBoxLayout()

        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setStyleSheet("QScrollArea { border: none; }")

        scroll_widget = QWidget()
        scroll_layout = QVBoxLayout(scroll_widget)
        scroll_layout.setContentsMargins(0, 0, 10, 0)

        # ── Top row: Sample Info | Hardware Setup ──────────────────────
        top_row = QHBoxLayout()

        # Sample Info (name, save path, buffer acquisition, secondary storage)
        grp_sample = QGroupBox("Sample Info")
        sample_form = QFormLayout()

        self.txt_sample_name = QLineEdit("Sample001")
        sample_form.addRow("Sample Name:", self.txt_sample_name)

        self.txt_dir = QLineEdit(os.getcwd())
        btn_dir = QPushButton("...")
        btn_dir.clicked.connect(
            lambda: self.txt_dir.setText(
                QFileDialog.getExistingDirectory(self, "Save Directory")
                or self.txt_dir.text()
            )
        )
        dir_layout = QHBoxLayout()
        dir_layout.addWidget(self.txt_dir)
        dir_layout.addWidget(btn_dir)
        dir_widget = QWidget()
        dir_widget.setLayout(dir_layout)
        sample_form.addRow("Save Path:", dir_widget)

        sample_form.addRow(QLabel("<b>Buffer Acquisition</b>"))
        self.sb_window_ms = NoWheelSpinBox()
        self.sb_window_ms.setRange(10, 1000)
        self.sb_window_ms.setValue(200)
        self.sb_window_ms.setSuffix(" ms")
        self.sb_window_ms.setToolTip(
            "Total PM400 array-capture window (transient mode only).\n"
            "The PM400 is armed (ABOR → INIT) BEFORE the image is sent to the\n"
            "SMILE controller, so the window starts slightly before the pixel turns on.\n"
            "Must be long enough to capture the full LED turn-on transient.\n"
            "Typical: 100–200 ms. Longer = more data, slower scan."
        )
        sample_form.addRow("Capture Window:", self.sb_window_ms)

        self.sb_min_remaining_ms = NoWheelSpinBox()
        self.sb_min_remaining_ms.setRange(0, 500)
        self.sb_min_remaining_ms.setValue(30)
        self.sb_min_remaining_ms.setSuffix(" ms")
        self.sb_min_remaining_ms.setToolTip(
            "If remaining window after image send is less than this, "
            "re-arm the PM400 for a fresh capture with the correct pixel on."
        )
        sample_form.addRow("Min. Remaining:", self.sb_min_remaining_ms)

        self.dsb_steady_tail = NoWheelDoubleSpinBox()
        self.dsb_steady_tail.setRange(1.0, 50.0)
        self.dsb_steady_tail.setValue(20.0)
        self.dsb_steady_tail.setSuffix(" %")
        self.dsb_steady_tail.setDecimals(1)
        self.dsb_steady_tail.setToolTip(
            "The last N % of the capture window used to compute the steady-state\n"
            "mean and std (stored as MEAS_VALUE / MEAS_STD in the CSV).\n"
            "Example: 20 % of a 200 ms window → last 40 ms averaged.\n"
            "A larger tail reduces noise but requires the LED to have fully settled\n"
            "within the first (100 − N) % of the window.\n"
            "Tip: measure a few transients first to see how long turn-on takes,\n"
            "then set the window and tail accordingly."
        )
        sample_form.addRow("Steady-State Tail:", self.dsb_steady_tail)

        self.sb_pre_settle = NoWheelSpinBox()
        self.sb_pre_settle.setRange(0, 30000)
        self.sb_pre_settle.setValue(500)
        self.sb_pre_settle.setSuffix(" ms")
        self.sb_pre_settle.setToolTip(
            "Wait time after the SMU outputs are turned on, before the first pixel\n"
            "measurement starts. Allows the VLED/NVLED bias to reach thermal and\n"
            "electrical steady state. 500 ms is a safe default; can be reduced\n"
            "once the supply is known to settle quickly."
        )
        sample_form.addRow("Output Pre-Settle:", self.sb_pre_settle)

        sample_form.addRow(QLabel("<b>Secondary Storage</b>"))
        self.chk_secondary = QCheckBox("Save Full Transient Arrays")
        self.chk_secondary.toggled.connect(self._on_secondary_toggled)
        self.chk_secondary.setToolTip(
            "Save the raw PM400 time-series (every sample in the capture window)\n"
            "for each pixel, in addition to the summary CSV.\n"
            "Required for rise-time analysis and minimum on-time determination.\n"
            "Storage cost: ~4 kB per pixel per bit-value at 100 ms / 10 kHz."
        )
        sample_form.addRow(self.chk_secondary)

        self.cb_sec_format = NoWheelComboBox()
        self.cb_sec_format.addItems(
            ["HDF5", "CSV Folder"] if HDF5_AVAILABLE else ["CSV Folder"]
        )
        self.cb_sec_format.setToolTip(
            "HDF5: single compressed file, fast random access, requires h5py.\n"
            "CSV Folder: one .csv file per (x, y, bitval), no extra dependencies\n"
            "but much larger total disk footprint."
        )
        sample_form.addRow("Format:", self.cb_sec_format)

        self.txt_sec_dir = QLineEdit()
        self.txt_sec_dir.setPlaceholderText("(Same as save path)")
        self.btn_sec_dir = QPushButton("...")
        self.btn_sec_dir.clicked.connect(
            lambda: self.txt_sec_dir.setText(
                QFileDialog.getExistingDirectory(self, "Transient Storage Directory")
                or self.txt_sec_dir.text()
            )
        )
        sec_dir_layout = QHBoxLayout()
        sec_dir_layout.addWidget(self.txt_sec_dir)
        sec_dir_layout.addWidget(self.btn_sec_dir)
        sec_dir_w = QWidget()
        sec_dir_w.setLayout(sec_dir_layout)
        sample_form.addRow("Directory:", sec_dir_w)

        self.chk_plot_transients = QCheckBox("Plot PM400 Transients After Measurement")
        self.chk_plot_transients.setEnabled(False)  # enabled only when secondary is on
        self.chk_plot_transients.setToolTip(
            "After the measurement, saves one PNG per secondary CSV file showing\n"
            "the PM400 waveform (time in ms vs power in W). PNGs are written to\n"
            "the same folder as the CSVs with the same filename.\n"
            "Also generates 'ontime_summary.csv' with gradient-detected on-times\n"
            "for each waveform (requires Turn Off Display to be enabled for clean data).\n"
            "Requires 'Save Full Transient Arrays' to be enabled.\n"
            "Requires matplotlib: pip install matplotlib"
        )
        sample_form.addRow(self.chk_plot_transients)

        grp_sample.setLayout(sample_form)
        top_row.addWidget(grp_sample, 3)

        # Hardware Setup (VISA connections + simulation checkboxes)
        grp_conn = QGroupBox("Hardware Setup")
        grid_conn = QGridLayout()
        grid_conn.setColumnStretch(1, 1)

        self.cb_pm400 = NoWheelComboBox()
        self.chk_sim_pm400 = QCheckBox("Simulate")
        self.cb_2602b = NoWheelComboBox()
        self.chk_sim_smu = QCheckBox("Simulate")
        self.chk_sim_smile = QCheckBox("Simulate SMILE")
        btn_refresh = QPushButton("Refresh VISA")
        btn_refresh.clicked.connect(self.refresh_resources)

        row = 0
        grid_conn.addWidget(QLabel("PM400:"), row, 0)
        grid_conn.addWidget(self.cb_pm400, row, 1)
        grid_conn.addWidget(self.chk_sim_pm400, row, 2)
        row += 1
        grid_conn.addWidget(QLabel("Dual SMU (2602B):"), row, 0)
        grid_conn.addWidget(self.cb_2602b, row, 1)
        grid_conn.addWidget(self.chk_sim_smu, row, 2)
        row += 1
        grid_conn.addWidget(self.chk_sim_smile, row, 1, 1, 2)
        row += 1
        self.chk_turnoff_dis = QCheckBox("Turn Off Display After Each Pixel")
        self.chk_turnoff_dis.setToolTip(
            "Sends a blank (all-zero) frame after each pixel's measurement window.\n"
            "Prevents light from the current pixel bleeding into the next measurement.\n"
            "The next set_pixel() call turns the chosen pixel back on."
        )
        grid_conn.addWidget(self.chk_turnoff_dis, row, 1, 1, 2)
        row += 1
        self.chk_dark_acq = QCheckBox("Dark Acquisition After Pixel")
        self.chk_dark_acq.setToolTip(
            "After the blank frame is sent, extends the capture window by 'dark tail' ms\n"
            "and averages those PM400 samples as the dark level for background subtraction.\n"
            "Requires 'Turn Off Display After Each Pixel' to be enabled for meaningful dark data."
        )
        grid_conn.addWidget(self.chk_dark_acq, row, 1, 1, 2)
        row += 1
        self.sb_dark_settle_ms = NoWheelSpinBox()
        self.sb_dark_settle_ms.setRange(0, 2000)
        self.sb_dark_settle_ms.setValue(0)
        self.sb_dark_settle_ms.setSuffix(" ms")
        self.sb_dark_settle_ms.setToolTip(
            "Wait time after the blank frame ACK before taking the dark measurement.\n"
            "The blank frame ACK arrives quickly but the display may not be fully dark\n"
            "until the current frame has finished refreshing. Set to 0 and increase\n"
            "if dark measurements show residual signal from the previous pixel."
        )
        grid_conn.addWidget(QLabel("Dark Settle:"), row, 0)
        grid_conn.addWidget(self.sb_dark_settle_ms, row, 1)
        row += 1
        self.sb_dark_tail_ms = NoWheelSpinBox()
        self.sb_dark_tail_ms.setRange(0, 500)
        self.sb_dark_tail_ms.setValue(0)
        self.sb_dark_tail_ms.setSuffix(" ms")
        self.sb_dark_tail_ms.setToolTip(
            "Transient mode only: extend the capture window by this many ms and\n"
            "send a blank frame at the original window end. The final X ms of\n"
            "the PM400 buffer will be dark — averaged automatically as dark data.\n"
            "Set to 0 to disable in-window dark capture."
        )
        grid_conn.addWidget(QLabel("Dark Tail (Transient):"), row, 0)
        grid_conn.addWidget(self.sb_dark_tail_ms, row, 1)
        row += 1
        self.chk_smu_display_off = QCheckBox(
            "Disable Keithley Display During Measurement"
        )
        self.chk_smu_display_off.setToolTip(
            "Turns off the Keithley front-panel display during measurement\n"
            "to reduce CPU load on the instrument. Restored on cleanup."
        )
        grid_conn.addWidget(self.chk_smu_display_off, row, 1, 1, 2)
        row += 1
        self.chk_post_process = QCheckBox("Run Post-Processing After Measurement")
        self.chk_post_process.setChecked(True)
        self.chk_post_process.setToolTip(
            "Runs data aggregation, heatmaps, on-time analysis, and transient plots\n"
            "after the measurement finishes. Uncheck to skip (e.g. for quick scans)."
        )
        grid_conn.addWidget(self.chk_post_process, row, 1, 1, 2)
        row += 1
        grid_conn.addWidget(btn_refresh, row, 1)

        grp_conn.setLayout(grid_conn)
        top_row.addWidget(grp_conn, 2)

        scroll_layout.addLayout(top_row)

        # ── Bottom row: Measurement Strategy | Device Config ───────────
        bottom_row = QHBoxLayout()

        # Measurement Strategy
        grp_strat = QGroupBox("Measurement Strategy")
        strat_layout = QVBoxLayout()

        mode_row = QHBoxLayout()
        mode_row.addWidget(QLabel("Mode:"))
        self.cb_meas_mode = NoWheelComboBox()
        self.cb_meas_mode.addItems(["Full Transient", "Fast Scan", "Ultra Fast"])
        self.cb_meas_mode.setToolTip(
            "Full Transient: arms PM400 array capture (ABOR → INIT) per pixel.\n"
            "Captures the complete LED turn-on waveform at 10 kHz.\n"
            "Slower (~capture window per pixel) but yields rise-time, on-time,\n"
            "and steady-state data. Supports NVLED sweep (Mode B).\n\n"
            "Fast Scan: uses PM400 INIT:CONT continuous mode + Keithley TSP burst.\n"
            "No per-pixel arm/fetch overhead — ~1–5 ms per pixel typical.\n"
            "Returns only mean ± std snapshots (no waveform). Best for\n"
            "quick yield maps across the full array.\n\n"
            "Ultra Fast: batches many pixels into one continuous PM400 array\n"
            "capture. Unsorts post-hoc by FPGA ACK timestamps with smart\n"
            "window trimming. Fastest mode by far; does not support NVLED sweep\n"
            "or secondary storage."
        )
        mode_row.addWidget(self.cb_meas_mode, 1)
        strat_layout.addLayout(mode_row)

        settle_row = QHBoxLayout()
        settle_row.addWidget(QLabel("Scan Settle:"))
        self.sb_fast_scan_settle = NoWheelSpinBox()
        self.sb_fast_scan_settle.setRange(0, 10000)
        self.sb_fast_scan_settle.setValue(20)
        self.sb_fast_scan_settle.setSuffix(" ms")
        self.sb_fast_scan_settle.setToolTip(
            "Dwell time after send_image before measuring (Fast Scan only)."
        )
        settle_row.addWidget(self.sb_fast_scan_settle, 1)
        strat_layout.addLayout(settle_row)

        npts_row = QHBoxLayout()
        npts_row.addWidget(QLabel("Meas./Pixel:"))
        self.sb_fast_scan_n_pts = NoWheelSpinBox()
        self.sb_fast_scan_n_pts.setRange(1, 1000)
        self.sb_fast_scan_n_pts.setValue(10)
        self.sb_fast_scan_n_pts.setToolTip(
            "Number of PM400 + SMU measurements per pixel (Fast Scan only)."
        )
        npts_row.addWidget(self.sb_fast_scan_n_pts, 1)
        strat_layout.addLayout(npts_row)

        # ── Ultra-Fast controls ─────────────────────────────────────
        ultra_dt_row = QHBoxLayout()
        ultra_dt_row.addWidget(QLabel("PM Δt:"))
        self.sb_ultra_delta_t = NoWheelSpinBox()
        self.sb_ultra_delta_t.setRange(100, 1000)
        self.sb_ultra_delta_t.setSingleStep(100)
        self.sb_ultra_delta_t.setValue(200)
        self.sb_ultra_delta_t.setSuffix(" μs")
        self.sb_ultra_delta_t.setToolTip(
            "PM400 sample interval (Ultra Fast only). Rounded to the\n"
            "nearest 100 μs. Smaller Δt → higher temporal resolution\n"
            "but shorter chunk duration (10 000-sample PM400 buffer cap).\n"
            "  100 μs → 1 s chunk  (~141 px/chunk)\n"
            "  200 μs → 2 s chunk  (~283 px/chunk)\n"
            "  500 μs → 5 s chunk  (~708 px/chunk)\n"
            "  1000 μs → 10 s chunk (~1416 px/chunk)"
        )
        ultra_dt_row.addWidget(self.sb_ultra_delta_t, 1)
        strat_layout.addLayout(ultra_dt_row)

        ultra_dark_row = QHBoxLayout()
        ultra_dark_row.addWidget(QLabel("Dark every N:"))
        self.sb_ultra_dark_every_n = NoWheelSpinBox()
        self.sb_ultra_dark_every_n.setRange(0, 100)
        self.sb_ultra_dark_every_n.setValue(10)
        self.sb_ultra_dark_every_n.setToolTip(
            "Insert a blank-frame dark reference every N pixels inside\n"
            "each chunk (Ultra Fast only). Used for drift-corrected\n"
            "dark subtraction via linear interpolation. 0 disables\n"
            "intermediate darks (still a chunk head + tail blank)."
        )
        ultra_dark_row.addWidget(self.sb_ultra_dark_every_n, 1)
        strat_layout.addLayout(ultra_dark_row)

        ultra_trim_row = QHBoxLayout()
        ultra_trim_row.addWidget(QLabel("Window Trim:"))
        self.dsb_ultra_trim_pct = NoWheelDoubleSpinBox()
        self.dsb_ultra_trim_pct.setRange(0.0, 40.0)
        self.dsb_ultra_trim_pct.setValue(10.0)
        self.dsb_ultra_trim_pct.setSuffix(" %")
        self.dsb_ultra_trim_pct.setDecimals(1)
        self.dsb_ultra_trim_pct.setToolTip(
            "Drop the first/last N% of each inter-ACK interval before\n"
            "computing per-pixel statistics (Ultra Fast only).\n"
            "Avoids turn-on / turn-off transitions that would bias the\n"
            "mean. Typical: 10 % (4 ms usable window on a 5 ms pixel)."
        )
        ultra_trim_row.addWidget(self.dsb_ultra_trim_pct, 1)
        strat_layout.addLayout(ultra_trim_row)

        self.chk_ultra_save_diag = QCheckBox("Save chunk diagnostics (.npz + .png)")
        self.chk_ultra_save_diag.setChecked(False)
        self.chk_ultra_save_diag.setToolTip(
            "Save per-chunk raw buffers and diagnostic plots to the "
            "raw_data/ultra_fast_diag/ folder. Runs in a background "
            "thread so it does not slow down the measurement."
        )
        strat_layout.addWidget(self.chk_ultra_save_diag)

        self.lbl_ultra_geom = QLabel("")
        self.lbl_ultra_geom.setStyleSheet("color: #666; font-size: 10px;")
        strat_layout.addWidget(self.lbl_ultra_geom)

        self.sb_ultra_delta_t.valueChanged.connect(self._update_ultra_geom_label)
        self.sb_ultra_dark_every_n.valueChanged.connect(self._update_ultra_geom_label)
        self._update_ultra_geom_label()

        self.cb_meas_mode.currentIndexChanged.connect(self._on_mode_changed)
        self._on_mode_changed(0)  # initialise visibility

        self.tabs_strat = QTabWidget()

        # Tab 1: Area Selection
        tab_area = QWidget()
        form_area = QFormLayout()
        self.cb_quadrant = NoWheelComboBox()
        self.cb_quadrant.addItems(["TL", "TR", "BL", "BR", "Full"])
        self.chk_roi = QCheckBox("Enable Rectangle Mode (ROI)")
        self.chk_roi.setToolTip(
            "Define a custom rectangular region instead of a full quadrant.\n"
            "X1/Y1 = top-left corner, X2/Y2 = bottom-right (inclusive, 0-indexed).\n"
            "Useful for targeting a sub-region of the 256×256 pixel array."
        )

        self.sb_roi_x1 = NoWheelSpinBox()
        self.sb_roi_x1.setRange(0, 511)
        self.sb_roi_y1 = NoWheelSpinBox()
        self.sb_roi_y1.setRange(0, 511)
        self.sb_roi_x2 = NoWheelSpinBox()
        self.sb_roi_x2.setRange(0, 511)
        self.sb_roi_y2 = NoWheelSpinBox()
        self.sb_roi_y2.setRange(0, 511)

        roi_form = QFormLayout()
        roi_form.addRow("X1:", self.sb_roi_x1)
        roi_form.addRow("Y1:", self.sb_roi_y1)
        roi_form.addRow("X2:", self.sb_roi_x2)
        roi_form.addRow("Y2:", self.sb_roi_y2)
        self.widget_roi = QWidget()
        self.widget_roi.setLayout(roi_form)
        self.widget_roi.setEnabled(False)
        self.chk_roi.toggled.connect(
            lambda c: (
                self.widget_roi.setEnabled(c),
                self.cb_quadrant.setEnabled(not c),
            )
        )

        self.sb_nth = NoWheelSpinBox()
        self.sb_nth.setRange(1, 10000)
        self.sb_nth.setValue(1)
        self.sb_nth.setToolTip(
            "Measure every N-th pixel from the selected area/CSV.\n"
            "N=1: all pixels. N=4: every 4th pixel (16× fewer measurements).\n"
            "Useful for fast yield surveys before committing to a full scan."
        )
        self.chk_random = QCheckBox("Random Order")
        self.chk_random.setToolTip(
            "Randomise the pixel visit order within the selected area.\n"
            "Reduces systematic spatial artefacts from slow instrument drift\n"
            "or thermal gradients across the measurement session."
        )
        self.chk_snake_scan = QCheckBox("Boustrophedon (Snake) Order")
        self.chk_snake_scan.setToolTip(
            "Alternates scan direction each row: left→right, then right→left, etc.\n"
            "Reduces display update overhead for large area scans."
        )

        form_area.addRow("Quadrant:", self.cb_quadrant)
        form_area.addRow(self.chk_roi)
        form_area.addRow(self.widget_roi)
        form_area.addRow("Every N-th Pixel:", self.sb_nth)
        form_area.addRow("", self.chk_random)
        form_area.addRow("", self.chk_snake_scan)
        tab_area.setLayout(form_area)
        self.tabs_strat.addTab(tab_area, "Area Selection")

        # Tab 2: CSV Load
        tab_csv = QWidget()
        form_csv = QFormLayout()
        self.txt_csv_path = QLineEdit()
        self.txt_csv_path.setReadOnly(True)
        btn_csv = QPushButton("Load CSV...")
        btn_csv.clicked.connect(self.browse_csv)
        csv_hl = QHBoxLayout()
        csv_hl.addWidget(self.txt_csv_path)
        csv_hl.addWidget(btn_csv)
        csv_hl_w = QWidget()
        csv_hl_w.setLayout(csv_hl)
        self.lbl_csv_info = QLabel("Loaded: 0 pixels")
        form_csv.addRow("Pixel CSV:", csv_hl_w)
        form_csv.addRow("", self.lbl_csv_info)
        tab_csv.setLayout(form_csv)
        self.tabs_strat.addTab(tab_csv, "From CSV")

        # Common strategy params
        form_strat_common = QFormLayout()
        self.txt_bitvals = QLineEdit(",".join(str(i) for i in range(16)))
        self.txt_bitvals.setToolTip(
            "Comma-separated list of grayscale drive levels to measure at each pixel (0–15).\n"
            "0 = off / minimum drive, 15 = maximum drive.\n"
            "Measuring all 16 levels gives the full EL-vs-drive curve.\n"
            "For a quick yield scan, use a single value, e.g. '15'."
        )
        form_strat_common.addRow("Bit Values:", self.txt_bitvals)

        strat_layout.addWidget(self.tabs_strat)
        strat_layout.addLayout(form_strat_common)
        grp_strat.setLayout(strat_layout)
        bottom_row.addWidget(grp_strat, 1)

        # Device Config
        grp_dev = QGroupBox("Device Config")
        dev_outer = QVBoxLayout()

        # PM400 — full width within device config
        grp_pm = QGroupBox("PM400")
        pm_grid = QGridLayout()
        self.sb_wavelength = NoWheelSpinBox()
        self.sb_wavelength.setRange(200, 2000)
        self.sb_wavelength.setValue(440)
        self.sb_pm_range = NoWheelComboBox()
        self.sb_pm_range.addItems(
            ["1e-3", "1e-4", "1e-5", "1e-6", "1e-7", "1e-8", "1e-9"]
        )
        self.sb_pm_range.setToolTip(
            "Fixed power measurement range (W). Auto-range is disabled to avoid\n"
            "range-switching latency mid-scan.\n"
            "Choose the smallest range that keeps your signal below full scale:\n"
            "  1e-3 W = 1 mW full scale\n"
            "  1e-6 W = 1 µW full scale  (best resolution for weak emitters)\n"
            "Too small → clipping; too large → poor ADC resolution."
        )
        pm_grid.addWidget(QLabel("Wavelength (nm):"), 0, 0)
        pm_grid.addWidget(self.sb_wavelength, 0, 1)
        pm_grid.addWidget(QLabel("Fixed Range (W):"), 0, 2)
        pm_grid.addWidget(self.sb_pm_range, 0, 3)
        grp_pm.setLayout(pm_grid)
        dev_outer.addWidget(grp_pm)

        # VLED + NVLED side by side
        smu_row = QHBoxLayout()

        grp_vled = QGroupBox("VLED — Ch A")
        form_vled = QFormLayout()
        self.dsb_vled_v = NoWheelDoubleSpinBox()
        self.dsb_vled_v.setRange(0, 3)
        self.dsb_vled_v.setValue(1.8)
        self.dsb_vled_v.setDecimals(3)
        self.dsb_vled_comp = NoWheelDoubleSpinBox()
        self.dsb_vled_comp.setRange(0.001, 1.0)
        self.dsb_vled_comp.setValue(1.0)
        self.dsb_vled_comp.setDecimals(3)
        self.dsb_vled_comp.setToolTip(
            "Current compliance limit (A). The SMU clamps the source current at this\n"
            "value to protect the device. Set just above the expected LED current."
        )
        self.dsb_vled_nplc = NoWheelDoubleSpinBox()
        self.dsb_vled_nplc.setRange(0.001, 25)
        self.dsb_vled_nplc.setValue(1.0)
        self.dsb_vled_nplc.setDecimals(3)
        self.dsb_vled_nplc.setToolTip(
            "Integration time in Power Line Cycles (50 Hz mains):\n"
            "  1     NPLC = 20.0 ms  (lowest noise)\n"
            "  0.1   NPLC =  2.0 ms\n"
            "  0.01  NPLC =  0.2 ms\n"
            "  0.001 NPLC =  0.02 ms  (fastest, highest noise)\n"
            "Use 0.001 for fast scan; 0.1–1.0 for precise transient steady-state."
        )
        self.chk_vled_highc = QCheckBox("High Capacitance Mode")
        self.chk_vled_highc.setToolTip(
            "Enables the SMU's internal high-capacitance compensation.\n"
            "Use when driving large pixel arrays or capacitive loads to prevent\n"
            "oscillation. Slightly increases source settling time."
        )
        self.cb_vled_range = NoWheelComboBox()
        self.cb_vled_range.addItems(
            ["1e-7", "1e-6", "1e-5", "1e-4", "1e-3", "1e-2", "1e-1", "1.0", "3.0"]
        )
        self.cb_vled_range.setCurrentText("1e-3")
        self.cb_vled_range.setToolTip(
            "Current measurement range (A). Use the smallest range that keeps the\n"
            "measured LED current below full scale for best resolution.\n"
            "Too small → overflow/clipping; too large → poor noise floor."
        )
        form_vled.addRow("Voltage (V):", self.dsb_vled_v)
        form_vled.addRow("Compl. (A):", self.dsb_vled_comp)
        form_vled.addRow("NPLC:", self.dsb_vled_nplc)
        form_vled.addRow("Range (A):", self.cb_vled_range)
        form_vled.addRow("", self.chk_vled_highc)
        grp_vled.setLayout(form_vled)

        grp_nvled = QGroupBox("NVLED — Ch B")
        form_nvled = QFormLayout()
        self.dsb_nvled_v = NoWheelDoubleSpinBox()
        self.dsb_nvled_v.setRange(-10, 10)
        self.dsb_nvled_v.setValue(-3.2)
        self.dsb_nvled_v.setDecimals(3)
        self.dsb_nvled_comp = NoWheelDoubleSpinBox()
        self.dsb_nvled_comp.setRange(0.001, 1.0)
        self.dsb_nvled_comp.setValue(0.5)
        self.dsb_nvled_comp.setDecimals(3)
        self.dsb_nvled_comp.setToolTip(
            "Current compliance limit (A) for the NVLED bias supply (Ch B).\n"
            "Set just above the expected maximum NVLED leakage/current."
        )
        self.dsb_nvled_nplc = NoWheelDoubleSpinBox()
        self.dsb_nvled_nplc.setRange(0.001, 25)
        self.dsb_nvled_nplc.setValue(1.0)
        self.dsb_nvled_nplc.setDecimals(3)
        self.dsb_nvled_nplc.setToolTip(
            "Integration time in Power Line Cycles (50 Hz mains):\n"
            "  1     NPLC = 20.0 ms  (lowest noise)\n"
            "  0.1   NPLC =  2.0 ms\n"
            "  0.01  NPLC =  0.2 ms\n"
            "  0.001 NPLC =  0.02 ms  (fastest, highest noise)\n"
            "Use 0.001 for fast scan; 0.1–1.0 for precise transient steady-state."
        )
        self.chk_nvled_highc = QCheckBox("High Capacitance Mode")
        self.chk_nvled_highc.setToolTip(
            "Enables the SMU's internal high-capacitance compensation for Ch B.\n"
            "Use when the NVLED bias drives a large capacitive pad structure."
        )
        self.cb_nvled_range = NoWheelComboBox()
        self.cb_nvled_range.addItems(
            ["1e-7", "1e-6", "1e-5", "1e-4", "1e-3", "1e-2", "1e-1", "1.0", "3.0"]
        )
        self.cb_nvled_range.setCurrentText("1e-3")
        self.cb_nvled_range.setToolTip(
            "Current measurement range (A) for the NVLED channel. Use the smallest\n"
            "range that keeps the measured current below full scale."
        )
        form_nvled.addRow("Start V (V):", self.dsb_nvled_v)
        form_nvled.addRow("Compl. (A):", self.dsb_nvled_comp)
        form_nvled.addRow("NPLC:", self.dsb_nvled_nplc)
        form_nvled.addRow("Range (A):", self.cb_nvled_range)
        form_nvled.addRow("", self.chk_nvled_highc)

        # NVLED sweep (inside NVLED group)
        self.chk_nvled_sweep = QCheckBox("Enable NVLED Sweep (Mode B)")
        self.chk_nvled_sweep.setToolTip(
            "Mode A (unchecked): fixed NVLED voltage for all pixels.\n"
            "Mode B (checked): sweeps NVLED from Start V to Target V in steps,\n"
            "measuring PM400 + SMU at each voltage. Gives EL vs NVLED bias curves.\n"
            "Each pixel takes (number of steps) × (capture window) time."
        )
        self.chk_nvled_sweep.toggled.connect(
            lambda c: self.widget_nvled_sweep.setEnabled(c)
        )
        sweep_form = QFormLayout()
        self.dsb_nvled_target = NoWheelDoubleSpinBox()
        self.dsb_nvled_target.setRange(-10, 10)
        self.dsb_nvled_target.setValue(-1.0)
        self.dsb_nvled_target.setDecimals(3)
        self.dsb_nvled_step = NoWheelDoubleSpinBox()
        self.dsb_nvled_step.setRange(0.001, 2)
        self.dsb_nvled_step.setValue(0.05)
        self.dsb_nvled_step.setDecimals(3)
        self.sb_nvled_settle = NoWheelSpinBox()
        self.sb_nvled_settle.setRange(0, 10000)
        self.sb_nvled_settle.setValue(100)
        self.sb_nvled_settle.setSuffix(" ms")
        self.sb_nvled_settle.setToolTip(
            "Wait time after each NVLED voltage step before arming the PM400.\n"
            "Allows the NVLED bias and any RC parasitics to settle to the new level.\n"
            "Increase if EL data shows voltage-step artefacts at the start of transients."
        )
        sweep_form.addRow("Target (V):", self.dsb_nvled_target)
        sweep_form.addRow("Step Size (V):", self.dsb_nvled_step)
        sweep_form.addRow("Pre-Sweep Settle:", self.sb_nvled_settle)
        self.widget_nvled_sweep = QWidget()
        self.widget_nvled_sweep.setLayout(sweep_form)
        self.widget_nvled_sweep.setEnabled(False)

        form_nvled.addRow(self.chk_nvled_sweep)
        form_nvled.addRow(self.widget_nvled_sweep)
        grp_nvled.setLayout(form_nvled)

        smu_row.addWidget(grp_vled)
        smu_row.addWidget(grp_nvled)
        dev_outer.addLayout(smu_row)
        grp_dev.setLayout(dev_outer)
        bottom_row.addWidget(grp_dev, 1)

        scroll_layout.addLayout(bottom_row)
        scroll_layout.addStretch()
        self.scroll_area.setWidget(scroll_widget)
        left_layout.addWidget(self.scroll_area)

        # ── Controls (fixed outside scroll) ───────────────────────────
        btn_layout = QHBoxLayout()
        self.btn_start = QPushButton("START")
        self.btn_start.setFixedHeight(50)
        self.btn_start.setStyleSheet(
            "background-color: #4CAF50; color: white; font-weight: bold;"
        )
        self.btn_start.clicked.connect(self.start_measurement)
        self.btn_stop = QPushButton("STOP")
        self.btn_stop.setFixedHeight(50)
        self.btn_stop.setStyleSheet(
            "background-color: #F44336; color: white; font-weight: bold;"
        )
        self.btn_stop.setEnabled(False)
        self.btn_stop.clicked.connect(self.stop_measurement)
        btn_layout.addWidget(self.btn_start)
        btn_layout.addWidget(self.btn_stop)

        self.log_box = QLabel("Ready")
        self.log_box.setWordWrap(True)
        self.log_box.setStyleSheet(
            "border: 1px solid #ccc; padding: 5px; background: white;"
        )
        left_layout.addLayout(btn_layout)
        left_layout.addWidget(self.log_box)

        # ── Right: Pixel map ───────────────────────────────────────────
        right_layout = QVBoxLayout()
        self.map_widget = ArrayMapWidget()
        right_layout.addWidget(QLabel("<b>Pixel Map (512×512)</b>"))
        right_layout.addWidget(self.map_widget)
        right_layout.addStretch()

        main_layout.addLayout(left_layout, 1)
        main_layout.addLayout(right_layout, 0)
        main_widget.setLayout(main_layout)

        # --- Post-Processing tab ---
        pp_widget = QWidget()
        pp_layout = QVBoxLayout()

        pp_form = QFormLayout()
        self.txt_pp_dir = QLineEdit()
        self.txt_pp_dir.setPlaceholderText("Select a measurement run folder...")
        btn_pp_dir = QPushButton("Browse...")
        btn_pp_dir.clicked.connect(
            lambda: self.txt_pp_dir.setText(
                QFileDialog.getExistingDirectory(self, "Select Run Directory")
                or self.txt_pp_dir.text()
            )
        )
        dir_row = QHBoxLayout()
        dir_row.addWidget(self.txt_pp_dir)
        dir_row.addWidget(btn_pp_dir)
        pp_form.addRow("Run Directory:", dir_row)

        self.chk_pp_aggregate = QCheckBox("Aggregate + yield stats")
        self.chk_pp_aggregate.setChecked(True)
        self.chk_pp_heatmaps = QCheckBox("Generate heatmaps")
        self.chk_pp_heatmaps.setChecked(True)
        self.chk_pp_ontimes = QCheckBox("On-time analysis (transient CSVs)")
        self.chk_pp_ontimes.setChecked(True)
        self.chk_pp_plots = QCheckBox("Plot PM400 transients (requires matplotlib)")
        self.chk_pp_plots.setChecked(False)
        pp_form.addRow("Steps:", self.chk_pp_aggregate)
        pp_form.addRow("", self.chk_pp_heatmaps)
        pp_form.addRow("", self.chk_pp_ontimes)
        pp_form.addRow("", self.chk_pp_plots)

        btn_run_pp = QPushButton("Run Post-Processing")
        btn_run_pp.clicked.connect(self._run_postprocess_tab)
        pp_form.addRow(btn_run_pp)

        self.lbl_pp_log = QLabel("(no output yet)")
        self.lbl_pp_log.setWordWrap(True)
        self.lbl_pp_log.setAlignment(Qt.AlignmentFlag.AlignTop)
        pp_scroll = QScrollArea()
        pp_scroll.setWidgetResizable(True)
        pp_log_widget = QWidget()
        pp_log_layout = QVBoxLayout()
        pp_log_layout.addWidget(self.lbl_pp_log)
        pp_log_layout.addStretch()
        pp_log_widget.setLayout(pp_log_layout)
        pp_scroll.setWidget(pp_log_widget)

        pp_layout.addLayout(pp_form)
        pp_layout.addWidget(QLabel("Log:"))
        pp_layout.addWidget(pp_scroll, 1)
        pp_widget.setLayout(pp_layout)

        # Top-level tab widget
        top_tabs = QTabWidget()
        top_tabs.addTab(main_widget, "Measurement")
        top_tabs.addTab(pp_widget, "Post-Processing")

        outer_widget = QWidget()
        outer_layout = QVBoxLayout()
        outer_layout.setContentsMargins(0, 0, 0, 0)
        outer_layout.addWidget(top_tabs)
        outer_widget.setLayout(outer_layout)
        self.setCentralWidget(outer_widget)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def browse_csv(self):
        file, _ = QFileDialog.getOpenFileName(
            self, "Select Pixel CSV", "", "CSV Files (*.csv)"
        )
        if file:
            self.txt_csv_path.setText(file)
            try:
                import pandas as pd

                df = pd.read_csv(file)
                if "X" in df.columns and "Y" in df.columns:
                    self.lbl_csv_info.setText(
                        f"Loaded: {len(df[['X', 'Y']].drop_duplicates())} pixels"
                    )
                else:
                    self.lbl_csv_info.setText(
                        "Error: CSV must have 'X' and 'Y' columns"
                    )
            except Exception as e:
                self.lbl_csv_info.setText(f"Error: {e}")

    def _on_mode_changed(self, _index=0):
        mode = self.cb_meas_mode.currentText()
        is_fast = mode == "Fast Scan"
        is_ultra = mode == "Ultra Fast"
        is_transient = mode == "Full Transient"
        # Fast-scan-only controls
        for w in (self.sb_fast_scan_settle, self.sb_fast_scan_n_pts):
            w.setEnabled(is_fast)
        # Ultra-fast-only controls
        for w in (
            self.sb_ultra_delta_t,
            self.sb_ultra_dark_every_n,
            self.dsb_ultra_trim_pct,
            self.chk_ultra_save_diag,
            self.lbl_ultra_geom,
        ):
            w.setEnabled(is_ultra)
        # Transient-only controls (buffer acquisition + secondary storage)
        for w in (
            self.sb_window_ms,
            self.sb_min_remaining_ms,
            self.dsb_steady_tail,
            self.chk_secondary,
            self.cb_sec_format,
            self.txt_sec_dir,
            self.btn_sec_dir,
            self.chk_plot_transients,
        ):
            w.setEnabled(is_transient)
        # NPLC + high-C are auto-managed in Ultra Fast mode (NPLC is
        # derived from delta_t, high-C is incompatible with zero-delay
        # burst sampling). Guard with hasattr because _on_mode_changed
        # can fire during init_ui before the Device Config widgets exist.
        if hasattr(self, "dsb_vled_nplc"):
            for w in (
                self.dsb_vled_nplc,
                self.dsb_nvled_nplc,
                self.chk_vled_highc,
                self.chk_nvled_highc,
            ):
                w.setEnabled(not is_ultra)
        # Re-apply secondary-toggle state only while in transient mode, so
        # chk_plot_transients respects chk_secondary when returning here.
        if is_transient:
            self._on_secondary_toggled(self.chk_secondary.isChecked())

    def _update_ultra_geom_label(self):
        try:
            from measurement.ultra_fast import compute_chunk_geometry

            delta_t = int(self.sb_ultra_delta_t.value())
            dark_n = int(self.sb_ultra_dark_every_n.value())
            n_pm, dur_ms, ppc, nplc = compute_chunk_geometry(delta_t, dark_n)
            self.lbl_ultra_geom.setText(
                f"~{ppc} px/chunk, {dur_ms / 1000:.1f} s capture, "
                f"NPLC={nplc:.3f} "
                f"({n_pm} PM samples @ {delta_t} μs, "
                f"{delta_t // 100}x avg)"
            )
        except Exception:
            self.lbl_ultra_geom.setText("")

    def _on_secondary_toggled(self, checked):
        self.chk_plot_transients.setEnabled(checked)
        if not checked:
            self.chk_plot_transients.setChecked(False)

    def refresh_resources(self):
        from instruments.resources import list_visa_resources

        resources = list_visa_resources()
        if not resources:
            resources = ["Sim_PM400", "Sim_K2602B"]
        for cb in (self.cb_pm400, self.cb_2602b):
            cb.clear()
            cb.addItems(resources)
        for res in resources:
            r = res.upper()
            if "USB" in r:
                self.cb_pm400.setCurrentText(res)
            elif "GPIB" in r or "26" in r:
                self.cb_2602b.setCurrentText(res)

    def closeEvent(self, event):
        self.save_gui_state()
        event.accept()

    # ------------------------------------------------------------------
    # State persistence
    # ------------------------------------------------------------------
    def save_gui_state(self):
        state = {
            "save_dir": self.txt_dir.text(),
            "sample_name": self.txt_sample_name.text(),
            "pixel_source_tab": self.tabs_strat.currentIndex(),
            "quadrant": self.cb_quadrant.currentIndex(),
            "roi_enabled": self.chk_roi.isChecked(),
            "roi": [
                self.sb_roi_x1.value(),
                self.sb_roi_y1.value(),
                self.sb_roi_x2.value(),
                self.sb_roi_y2.value(),
            ],
            "csv_path": self.txt_csv_path.text(),
            "bit_values": self.txt_bitvals.text(),
            "nth_pixel": self.sb_nth.value(),
            "random_mode": self.chk_random.isChecked(),
            "sim_pm400": self.chk_sim_pm400.isChecked(),
            "sim_smu": self.chk_sim_smu.isChecked(),
            "sim_smile": self.chk_sim_smile.isChecked(),
            "snake_scan": self.chk_snake_scan.isChecked(),
            "dark_acq": self.chk_dark_acq.isChecked(),
            "pm_range_idx": self.sb_pm_range.currentIndex(),
            "pm_wave": self.sb_wavelength.value(),
            "vled_v": self.dsb_vled_v.value(),
            "vled_comp": self.dsb_vled_comp.value(),
            "vled_nplc": self.dsb_vled_nplc.value(),
            "vled_range_idx": self.cb_vled_range.currentIndex(),
            "vled_highc": self.chk_vled_highc.isChecked(),
            "nvled_v": self.dsb_nvled_v.value(),
            "nvled_comp": self.dsb_nvled_comp.value(),
            "nvled_nplc": self.dsb_nvled_nplc.value(),
            "nvled_range_idx": self.cb_nvled_range.currentIndex(),
            "nvled_highc": self.chk_nvled_highc.isChecked(),
            "nvled_sweep": self.chk_nvled_sweep.isChecked(),
            "nvled_target": self.dsb_nvled_target.value(),
            "nvled_step": self.dsb_nvled_step.value(),
            "nvled_settle": self.sb_nvled_settle.value(),
            "measurement_mode": self.cb_meas_mode.currentText(),
            "fast_scan_settle": self.sb_fast_scan_settle.value(),
            "fast_scan_n_pts": self.sb_fast_scan_n_pts.value(),
            "ultra_fast_delta_t_us": self.sb_ultra_delta_t.value(),
            "ultra_fast_dark_every_n": self.sb_ultra_dark_every_n.value(),
            "ultra_fast_trim_pct": self.dsb_ultra_trim_pct.value(),
            "ultra_fast_save_diag": self.chk_ultra_save_diag.isChecked(),
            "turnoff_dis": self.chk_turnoff_dis.isChecked(),
            "dark_settle_ms": self.sb_dark_settle_ms.value(),
            "window_ms": self.sb_window_ms.value(),
            "min_remaining_ms": self.sb_min_remaining_ms.value(),
            "steady_tail": self.dsb_steady_tail.value(),
            "pre_settle": self.sb_pre_settle.value(),
            "secondary_enabled": self.chk_secondary.isChecked(),
            "secondary_format": self.cb_sec_format.currentText(),
            "secondary_dir": self.txt_sec_dir.text(),
            "plot_transients": self.chk_plot_transients.isChecked(),
            "post_process_enabled": self.chk_post_process.isChecked(),
            "smu_display_off": self.chk_smu_display_off.isChecked(),
            "dark_tail_ms": self.sb_dark_tail_ms.value(),
        }
        try:
            with open(self.CONFIG_FILE, "w") as f:
                json.dump(state, f, indent=4)
        except Exception:
            pass

    def load_gui_state(self):
        if not os.path.exists(self.CONFIG_FILE):
            return
        try:
            with open(self.CONFIG_FILE, "r") as f:
                s = json.load(f)

            def get(key, widget_setter):
                if key in s:
                    widget_setter(s[key])

            get("save_dir", self.txt_dir.setText)
            get("sample_name", self.txt_sample_name.setText)
            get("pixel_source_tab", self.tabs_strat.setCurrentIndex)
            get("quadrant", self.cb_quadrant.setCurrentIndex)
            get("roi_enabled", self.chk_roi.setChecked)
            if "roi" in s and len(s["roi"]) == 4:
                self.sb_roi_x1.setValue(s["roi"][0])
                self.sb_roi_y1.setValue(s["roi"][1])
                self.sb_roi_x2.setValue(s["roi"][2])
                self.sb_roi_y2.setValue(s["roi"][3])
            get("bit_values", self.txt_bitvals.setText)
            get("nth_pixel", self.sb_nth.setValue)
            get("random_mode", self.chk_random.setChecked)
            get("sim_pm400", self.chk_sim_pm400.setChecked)
            get("sim_smu", self.chk_sim_smu.setChecked)
            get("sim_smile", self.chk_sim_smile.setChecked)
            get("snake_scan", self.chk_snake_scan.setChecked)
            get("dark_acq", self.chk_dark_acq.setChecked)
            get("pm_wave", self.sb_wavelength.setValue)
            get("pm_range_idx", self.sb_pm_range.setCurrentIndex)
            get("vled_v", self.dsb_vled_v.setValue)
            get("vled_comp", self.dsb_vled_comp.setValue)
            get("vled_nplc", self.dsb_vled_nplc.setValue)
            get("vled_range_idx", self.cb_vled_range.setCurrentIndex)
            get("vled_highc", self.chk_vled_highc.setChecked)
            get("nvled_v", self.dsb_nvled_v.setValue)
            get("nvled_comp", self.dsb_nvled_comp.setValue)
            get("nvled_nplc", self.dsb_nvled_nplc.setValue)
            get("nvled_range_idx", self.cb_nvled_range.setCurrentIndex)
            get("nvled_highc", self.chk_nvled_highc.setChecked)
            get("nvled_sweep", self.chk_nvled_sweep.setChecked)
            get("nvled_target", self.dsb_nvled_target.setValue)
            get("nvled_step", self.dsb_nvled_step.setValue)
            get("nvled_settle", self.sb_nvled_settle.setValue)
            if "measurement_mode" in s:
                idx = self.cb_meas_mode.findText(s["measurement_mode"])
                if idx >= 0:
                    self.cb_meas_mode.setCurrentIndex(idx)
            get("fast_scan_settle", self.sb_fast_scan_settle.setValue)
            get("fast_scan_n_pts", self.sb_fast_scan_n_pts.setValue)
            get("ultra_fast_delta_t_us", self.sb_ultra_delta_t.setValue)
            get("ultra_fast_dark_every_n", self.sb_ultra_dark_every_n.setValue)
            get("ultra_fast_trim_pct", self.dsb_ultra_trim_pct.setValue)
            get("ultra_fast_save_diag", self.chk_ultra_save_diag.setChecked)
            self._update_ultra_geom_label()
            get("turnoff_dis", self.chk_turnoff_dis.setChecked)
            get("dark_settle_ms", self.sb_dark_settle_ms.setValue)
            get("window_ms", self.sb_window_ms.setValue)
            get("min_remaining_ms", self.sb_min_remaining_ms.setValue)
            get("steady_tail", self.dsb_steady_tail.setValue)
            get("pre_settle", self.sb_pre_settle.setValue)
            get("secondary_enabled", self.chk_secondary.setChecked)
            if "secondary_format" in s:
                idx = self.cb_sec_format.findText(s["secondary_format"])
                if idx >= 0:
                    self.cb_sec_format.setCurrentIndex(idx)
            get("secondary_dir", self.txt_sec_dir.setText)
            get("plot_transients", self.chk_plot_transients.setChecked)
            get("post_process_enabled", self.chk_post_process.setChecked)
            get("smu_display_off", self.chk_smu_display_off.setChecked)
            get("dark_tail_ms", self.sb_dark_tail_ms.setValue)
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Measurement control
    # ------------------------------------------------------------------
    def start_measurement(self):
        cfg = {
            "save_dir": self.txt_dir.text(),
            "sample_name": self.txt_sample_name.text(),
            "pixel_source": "CSV" if self.tabs_strat.currentIndex() == 1 else "Area",
            "csv_file_path": self.txt_csv_path.text(),
            "quadrant": self.cb_quadrant.currentText(),
            "roi_enabled": self.chk_roi.isChecked(),
            "roi_x1": self.sb_roi_x1.value(),
            "roi_y1": self.sb_roi_y1.value(),
            "roi_x2": self.sb_roi_x2.value(),
            "roi_y2": self.sb_roi_y2.value(),
            "bit_values": self.txt_bitvals.text(),
            "nth_pixel": self.sb_nth.value(),
            "random_mode": self.chk_random.isChecked(),
            "sim_pm400": self.chk_sim_pm400.isChecked(),
            "sim_smu": self.chk_sim_smu.isChecked(),
            "sim_smile": self.chk_sim_smile.isChecked(),
            "snake_scan": self.chk_snake_scan.isChecked(),
            "dark_acq": self.chk_dark_acq.isChecked(),
            "pm_wavelength": self.sb_wavelength.value(),
            "pm_range": float(self.sb_pm_range.currentText()),
            "vled_voltage": self.dsb_vled_v.value(),
            "vled_compliance": self.dsb_vled_comp.value(),
            "vled_nplc": self.dsb_vled_nplc.value(),
            "vled_range_i": float(self.cb_vled_range.currentText()),
            "vled_highc": self.chk_vled_highc.isChecked(),
            "nvled_voltage": self.dsb_nvled_v.value(),
            "nvled_compliance": self.dsb_nvled_comp.value(),
            "nvled_nplc": self.dsb_nvled_nplc.value(),
            "nvled_range_i": float(self.cb_nvled_range.currentText()),
            "nvled_highc": self.chk_nvled_highc.isChecked(),
            "nvled_sweep": self.chk_nvled_sweep.isChecked(),
            "nvled_sweep_target": self.dsb_nvled_target.value(),
            "nvled_sweep_step": self.dsb_nvled_step.value(),
            "nvled_settle_ms": self.sb_nvled_settle.value(),
            "measurement_mode": self.cb_meas_mode.currentText(),
            "fast_scan_settle_ms": self.sb_fast_scan_settle.value(),
            "fast_scan_n_pts": self.sb_fast_scan_n_pts.value(),
            "ultra_fast_delta_t_us": self.sb_ultra_delta_t.value(),
            "ultra_fast_dark_every_n": self.sb_ultra_dark_every_n.value(),
            "ultra_fast_trim_pct": self.dsb_ultra_trim_pct.value(),
            "ultra_fast_save_diag": self.chk_ultra_save_diag.isChecked(),
            "turnoff_dis": self.chk_turnoff_dis.isChecked(),
            "dark_settle_ms": self.sb_dark_settle_ms.value(),
            "window_ms": self.sb_window_ms.value(),
            "min_remaining_ms": self.sb_min_remaining_ms.value(),
            "steady_tail_pct": self.dsb_steady_tail.value(),
            "pre_settle_ms": self.sb_pre_settle.value(),
            # Secondary storage + plot_transients + transient-only SMU
            # display are only valid in Full Transient mode — Fast Scan
            # and Ultra Fast have no per-pixel arrays to save.
            "secondary_storage_enabled": (
                self.chk_secondary.isChecked()
                and self.cb_meas_mode.currentText() == "Full Transient"
            ),
            "secondary_storage_format": self.cb_sec_format.currentText(),
            "secondary_storage_dir": self.txt_sec_dir.text(),
            "plot_transients": (
                self.chk_plot_transients.isChecked()
                and self.cb_meas_mode.currentText() == "Full Transient"
            ),
            "post_process_enabled": self.chk_post_process.isChecked(),
            "smu_display_off": (
                self.chk_smu_display_off.isChecked()
                and self.cb_meas_mode.currentText() == "Full Transient"
            ),
            "dark_tail_ms": self.sb_dark_tail_ms.value(),
        }
        addrs = {
            "pm400": self.cb_pm400.currentText(),
            "smu": self.cb_2602b.currentText(),
        }
        self.map_widget.reset_map()
        self.btn_start.setEnabled(False)
        self.btn_stop.setEnabled(True)
        self.status_bar.showMessage("Starting...")

        self.worker = MeasurementWorker(cfg, addrs)
        self.worker.log_msg.connect(self.log_box.setText)
        self.worker.pixel_update.connect(self.map_widget.set_pixel_status)
        self.worker.eta_update.connect(self.status_bar.showMessage)
        self.worker.finished.connect(self.on_worker_finished)
        self.worker.error.connect(lambda e: QMessageBox.critical(self, "Error", e))
        self.worker.start()

    def stop_measurement(self):
        if self.worker:
            self.worker.stop()
            self.log_box.setText("Stopping...")
            self.btn_stop.setEnabled(False)

    def on_worker_finished(self):
        self.btn_start.setEnabled(True)
        self.btn_stop.setEnabled(False)
        self.log_box.setText("Idle")
        self.status_bar.showMessage("Done")

    def _run_postprocess_tab(self):
        run_dir = self.txt_pp_dir.text().strip()
        if not run_dir:
            self.lbl_pp_log.setText("Error: no run directory selected.")
            return
        run_dir = Path(run_dir)
        if not run_dir.exists():
            self.lbl_pp_log.setText(f"Error: directory does not exist:\n{run_dir}")
            return
        log_lines = []

        def _log(msg):
            log_lines.append(str(msg))
            self.lbl_pp_log.setText("\n".join(log_lines[-50:]))
            QApplication.processEvents()

        try:
            _postprocess.run_postprocess(
                run_dir,
                do_aggregate=self.chk_pp_aggregate.isChecked(),
                do_heatmaps=self.chk_pp_heatmaps.isChecked(),
                do_ontimes=self.chk_pp_ontimes.isChecked(),
                do_plots=self.chk_pp_plots.isChecked(),
                log_fn=_log,
            )
            _log("Post-processing complete.")
        except Exception as e:
            _log(f"Error: {e}")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MicroLEDCharGUI()
    window.show()
    sys.exit(app.exec())
