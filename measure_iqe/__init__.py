"""IQE / EQE acquisition for the SMILE micro-LED array.

Use it as a library::

    from measure_iqe import IQEConfig, run_iqe_measurement

    cfg = IQEConfig(roi=(32, 32, 63, 63), sweep_direction="low_to_high")
    result = run_iqe_measurement(cfg, simulate=True)
    print(result.run_dir, result.n_points)

or from the command line::

    python -m measure_iqe --simulate --set roi='[0,0,7,7]'

The package writes CSVs only. Fit and plot the result with the separate
SMILE_IQE_Analysis repo; RUN_FORMAT.md specifies the handoff.
"""

from .acquire import (
    IQEResult,
    acquire_point,
    build_fine_voltages,
    burst_stats,
    configure_common,
    configure_voltage_source_b,
    connect,
    make_logger,
    make_run_dir,
    run_fast_scan,
    run_iqe_measurement,
    run_sweep_current,
    run_sweep_voltage,
    run_warmup_diag,
    select_sweep_pixels,
    set_measure_range_b,
    set_nplc,
    shutdown,
)
from .config import NAMED_ROIS, SMU_I_RANGES, IQEConfig

__version__ = "1.0.0"

__all__ = [
    "IQEConfig",
    "IQEResult",
    "NAMED_ROIS",
    "SMU_I_RANGES",
    "acquire_point",
    "build_fine_voltages",
    "burst_stats",
    "configure_common",
    "configure_voltage_source_b",
    "connect",
    "make_logger",
    "make_run_dir",
    "run_fast_scan",
    "run_iqe_measurement",
    "run_sweep_current",
    "run_sweep_voltage",
    "run_warmup_diag",
    "select_sweep_pixels",
    "set_measure_range_b",
    "set_nplc",
    "shutdown",
]
