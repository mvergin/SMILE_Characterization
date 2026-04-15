"""Runtime services handed to measurement modes + coordinator.

The context bundles everything a mode needs from the outside world —
config, data sink, profiling dict, cancel flag, progress callbacks — so
mode implementations never need a back-reference to `MeasurementWorker`.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

from .config import MeasurementConfig


@dataclass
class MeasurementContext:
    """Everything a measurement mode needs from the outer run scope."""

    cfg: MeasurementConfig
    data_writer: object  # data.writer.DataWriter (avoid hard import)
    profiling: dict
    start_time: float
    sec_dir: Optional[Path]
    raw_data_dir: Optional[Path] = None
    is_running: Callable[[], bool] = lambda: True
    log: Callable[[str], None] = lambda _: None
    pixel_active: Callable[[int, int], None] = lambda _x, _y: None
    pixel_done: Callable[[int, int], None] = lambda _x, _y: None
    set_eta: Callable[[str], None] = lambda _: None
