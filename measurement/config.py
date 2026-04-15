"""Typed measurement configuration.

Replaces the ~45-key `cfg` dict previously marshalled from the GUI with
a single frozen dataclass. Consumers use attribute access (`cfg.window_ms`)
instead of string-keyed dict lookups, which gives IDE autocomplete,
typo detection at startup, and a single place to document every knob.

The GUI still builds a plain dict from widget states; call
`MeasurementConfig.from_gui_dict(d)` to convert.
"""

from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Any, Dict


@dataclass(frozen=True)
class MeasurementConfig:
    """All settings needed to run one measurement job.

    Fields mirror the historic GUI config dict 1:1 so state files
    (`gui_config_v9.json`) and code that still thinks in dict-keys can
    round-trip through `from_gui_dict` / `to_dict`.
    """

    # ---------------- Output ----------------
    save_dir: str = ""
    sample_name: str = ""
    post_process_enabled: bool = True
    plot_transients: bool = False

    # ---------------- Pixel selection ----------------
    pixel_source: str = "Area"  # "Area" or "CSV"
    csv_file_path: str = ""
    quadrant: str = "Full"
    roi_enabled: bool = False
    roi_x1: int = 0
    roi_y1: int = 0
    roi_x2: int = 511
    roi_y2: int = 511
    bit_values: str = "15"
    nth_pixel: int = 1
    random_mode: bool = False
    snake_scan: bool = False

    # ---------------- Simulation flags ----------------
    sim_pm400: bool = False
    sim_smu: bool = False
    sim_smile: bool = False

    # ---------------- PM400 ----------------
    pm_wavelength: float = 550.0
    pm_range: float = 1.0e-3

    # ---------------- Keithley 2602B - VLED (ch A) ----------------
    vled_voltage: float = 0.0
    vled_compliance: float = 0.1
    vled_nplc: float = 1.0
    vled_range_i: float = 1.0e-3
    vled_highc: bool = False

    # ---------------- Keithley 2602B - NVLED (ch B) ----------------
    nvled_voltage: float = 0.0
    nvled_compliance: float = 0.1
    nvled_nplc: float = 1.0
    nvled_range_i: float = 1.0e-3
    nvled_highc: bool = False
    smu_display_off: bool = False

    # ---------------- NVLED sweep ----------------
    nvled_sweep: bool = False
    nvled_sweep_target: float = 0.0
    nvled_sweep_step: float = 0.1
    nvled_settle_ms: float = 0.0

    # ---------------- Measurement mode ----------------
    measurement_mode: str = "Full Transient"
    pre_settle_ms: float = 100.0
    window_ms: float = 50.0
    min_remaining_ms: float = 0.0
    steady_tail_pct: float = 20.0
    fast_scan_settle_ms: float = 2.0
    fast_scan_n_pts: int = 10

    # ---------------- Dark / turnoff ----------------
    dark_acq: bool = False
    dark_settle_ms: float = 0.0
    dark_tail_ms: float = 0.0
    dark_every_n_sweep: int = 0  # interleave dark every N voltage steps (0=off)
    turnoff_dis: bool = False

    # ---------------- Secondary storage ----------------
    secondary_storage_enabled: bool = False
    secondary_storage_format: str = "HDF5"
    secondary_storage_dir: str = ""

    # ---------------- Ultra-Fast mode ----------------
    # Operator-selected PM400 sample interval. The chunk length and
    # pixels-per-chunk are derived from this + the 10k-sample PM400
    # buffer limit.
    ultra_fast_delta_t_us: int = 200
    # Interleave a blank-frame dark reference every N pixels for drift
    # compensation.
    ultra_fast_dark_every_n: int = 10
    # Drop the first/last N% of each inter-ACK interval before computing
    # per-pixel statistics. This avoids turn-on / turn-off transitions
    # that the PM400 would otherwise include in the average.
    ultra_fast_trim_pct: float = 10.0
    # Save per-chunk .npz and .png diagnostics (runs in background thread).
    ultra_fast_save_diag: bool = False

    # ------------------------------------------------------------------
    @classmethod
    def from_gui_dict(cls, d: Dict[str, Any]) -> "MeasurementConfig":
        """Build a MeasurementConfig from the GUI's dict marshalling.

        Unknown keys are silently dropped (keeps older state files
        loading cleanly). Missing keys fall back to dataclass defaults.
        """
        known = {f.name for f in fields(cls)}
        filtered = {k: v for k, v in d.items() if k in known}
        return cls(**filtered)

    def to_dict(self) -> Dict[str, Any]:
        """Return a plain dict representation (for JSON state saves)."""
        return {f.name: getattr(self, f.name) for f in fields(self)}
