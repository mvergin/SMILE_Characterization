"""Background CSV / HDF5 writer for measurement output.

The measurement loop produces two data streams:
  1. A single "overview" CSV row per pixel measurement (mean/std values).
  2. Optional per-pixel transient data — either dumped into an HDF5 file
     or into individual CSV files in a secondary directory.

Both streams are drained by a single background thread so the measurement
loop never blocks on disk I/O. `DataWriter` owns the queue, the worker
thread, and the open file handles; callers push work via `write_rows()`
and `write_transient()`, then call `close()` to flush and join.
"""

from __future__ import annotations

import csv
import queue
import threading
from pathlib import Path
from typing import Callable, Optional

import numpy as np

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    MPL_AVAILABLE = True
except Exception:
    MPL_AVAILABLE = False

try:
    import h5py

    HDF5_AVAILABLE = True
except Exception:
    HDF5_AVAILABLE = False


CSV_HEADER = [
    "X",
    "Y",
    "BITVAL",
    "NVLED_V",
    "TIME",
    "TYPE",
    "MEAS_VALUE",
    "MEAS_STD",
]


def _save_transient_hdf5(
    hdf5_file,
    x,
    y,
    bv,
    nv_vol,
    pm_times,
    pm_arr,
    k_times,
    vled_arr,
    nvled_arr,
    T0,
    mode="A",
    t_ack_s=None,
    t_turnoff_s=None,
):
    grp = hdf5_file.require_group(f"x{x:03d}_y{y:03d}/bv{bv:02d}/nv{nv_vol:.4f}")
    grp.attrs["T0_perf_counter_s"] = T0
    grp.attrs["mode"] = mode
    grp.attrs["x"] = x
    grp.attrs["y"] = y
    grp.attrs["bitval"] = bv
    grp.attrs["nvled_v"] = nv_vol
    if t_ack_s is not None:
        grp.attrs["t_ack_s"] = t_ack_s
    if t_turnoff_s is not None:
        grp.attrs["t_turnoff_s"] = t_turnoff_s
    if pm_arr:
        grp.create_dataset(
            "pm400_time_s",
            data=np.array(pm_times, dtype=np.float64),
            compression="gzip",
        )
        grp.create_dataset(
            "pm400_power_W",
            data=np.array(pm_arr, dtype=np.float32),
            compression="gzip",
        )
    if vled_arr:
        grp.create_dataset(
            "k_time_s",
            data=np.array(k_times, dtype=np.float64),
            compression="gzip",
        )
        grp.create_dataset(
            "vled_current_A",
            data=np.array(vled_arr, dtype=np.float32),
            compression="gzip",
        )
        grp.create_dataset(
            "nvled_current_A",
            data=np.array(nvled_arr, dtype=np.float32),
            compression="gzip",
        )


def _save_transient_csv(
    sec_dir,
    x,
    y,
    bv,
    nv_vol,
    pm_times,
    pm_arr,
    k_times,
    vled_arr,
    nvled_arr,
    T0,
    mode="A",
    t_ack_s=None,
    t_turnoff_s=None,
):
    rows = []
    for t, v in zip(pm_times, pm_arr):
        rows.append([round(t, 9), "PM400", v])
    for t, iv, in_ in zip(k_times, vled_arr, nvled_arr):
        rows.append([round(t, 9), "VLED", iv])
        rows.append([round(t, 9), "NVLED", in_])
    if t_ack_s is not None:
        rows.append([round(T0 + t_ack_s, 9), "ACK", 1.0])
    if t_turnoff_s is not None:
        rows.append([round(T0 + t_turnoff_s, 9), "TURNOFF_ACK", 1.0])
    rows.sort(key=lambda r: r[0])
    fpath = sec_dir / f"x{x:03d}_y{y:03d}_b{bv:02d}_nv{nv_vol:.4f}.csv"
    with open(fpath, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["TIME_s", "TYPE", "VALUE"])
        w.writerows(rows)


def _plot_experimental_pixel(
    sec_dir,
    x,
    y,
    bv,
    pm_times,
    pm_arr,
    k_times,
    nvled_arr,
    v_list,
    seg_starts,
    seg_ends,
):
    """Render PM400 power + NVLED current with per-voltage chunk shading.

    One PNG per (x, y, bitval). Coloured spans mark the K-sample window
    that contributed to each voltage's mean/std row.
    """
    if not MPL_AVAILABLE or not pm_arr or not k_times:
        return
    fig, (ax_pm, ax_nv) = plt.subplots(
        2, 1, figsize=(11, 6), sharex=True
    )
    ax_pm.plot(pm_times, np.asarray(pm_arr) * 1e6, lw=0.8, color="tab:blue")
    ax_pm.set_ylabel("PM400 power (µW)")
    ax_pm.set_title(
        f"Experimental sweep — pixel ({x},{y}) bv={bv}, {len(v_list)} voltages"
    )
    ax_nv.plot(
        k_times, np.asarray(nvled_arr) * 1e3, lw=0.8, color="tab:red"
    )
    ax_nv.set_ylabel("NVLED current (mA)")
    ax_nv.set_xlabel("Time (s, rel. PM arm)")

    cmap = plt.get_cmap("viridis")
    n = max(1, len(v_list))
    for j, (t0, t1, v) in enumerate(zip(seg_starts, seg_ends, v_list)):
        c = cmap(j / n)
        for ax in (ax_pm, ax_nv):
            ax.axvspan(t0, t1, color=c, alpha=0.18, lw=0)
        ax_nv.text(
            (t0 + t1) / 2.0,
            ax_nv.get_ylim()[1],
            f"{v:+.2f}V",
            ha="center",
            va="top",
            fontsize=7,
            rotation=90,
            color=c,
        )
    ax_pm.grid(alpha=0.3)
    ax_nv.grid(alpha=0.3)
    fig.tight_layout()
    out = sec_dir / f"x{x:03d}_y{y:03d}_b{bv:02d}_exp.png"
    fig.savefig(out, dpi=110)
    plt.close(fig)


class DataWriter:
    """Owns the overview CSV + background writer thread.

    Parameters
    ----------
    csv_path : Path
        Path to the overview CSV file. Will be opened for writing and
        the header row emitted immediately.
    sec_dir : Path or None
        Secondary storage directory for per-pixel transient CSVs. Only
        used when HDF5 is not selected.
    hdf5_file : h5py.File or None
        Open HDF5 file handle for transient data. If provided, takes
        precedence over `sec_dir` for transient storage.
    error_callback : callable, optional
        Called with an error message string when a background write
        fails. Defaults to a no-op.
    """

    def __init__(
        self,
        csv_path: Path,
        sec_dir: Optional[Path] = None,
        hdf5_file=None,
        error_callback: Optional[Callable[[str], None]] = None,
    ):
        self._csv_path = Path(csv_path)
        self._sec_dir = sec_dir
        self._hdf5_file = hdf5_file
        self._error_cb = error_callback or (lambda msg: None)
        self._queue: queue.Queue = queue.Queue()
        self._csv_fh = None
        self._csv_writer = None
        self._thread: Optional[threading.Thread] = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------
    def __enter__(self) -> "DataWriter":
        self.start()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def start(self) -> None:
        """Open the CSV, write the header, and launch the worker thread."""
        self._csv_fh = open(self._csv_path, "w", newline="")
        self._csv_writer = csv.writer(self._csv_fh)
        self._csv_writer.writerow(CSV_HEADER)
        self._csv_fh.flush()
        self._thread = threading.Thread(target=self._worker, daemon=True)
        self._thread.start()

    def close(self) -> None:
        """Flush the queue, join the worker, close the CSV file."""
        if self._thread is not None and self._thread.is_alive():
            self._queue.put(None)
            self._thread.join(timeout=30.0)
        self._thread = None
        if self._csv_fh is not None:
            try:
                self._csv_fh.close()
            except Exception:
                pass
            self._csv_fh = None
            self._csv_writer = None

    # ------------------------------------------------------------------
    # Producer API
    # ------------------------------------------------------------------
    def write_rows(self, rows) -> None:
        """Enqueue a batch of overview CSV rows."""
        self._queue.put(("csv", rows))

    def write_transient(self, **kwargs) -> None:
        """Enqueue a per-pixel transient record for secondary storage.

        Keyword arguments are passed through to the underlying HDF5 or
        CSV writer: x, y, bv, nv_vol, pm_times, pm_arr, k_times,
        vled_arr, nvled_arr, T0, mode, t_ack_s, t_turnoff_s.

        Callers may also pass `hdf5_file` / `sec_dir` for compatibility
        with the pre-refactor API; those keys are stripped here since
        the DataWriter owns the real handles.
        """
        if self._hdf5_file is None and self._sec_dir is None:
            return  # No secondary storage configured — silently drop.
        kwargs.pop("hdf5_file", None)
        kwargs.pop("sec_dir", None)
        self._queue.put(("secondary", kwargs))

    def plot_experimental(self, **kwargs) -> None:
        """Enqueue a per-pixel experimental-sweep plot.

        Payload keys: x, y, bv, pm_times, pm_arr, k_times, nvled_arr,
        v_list, seg_starts, seg_ends. Silently dropped if no sec_dir.
        """
        if self._sec_dir is None or not MPL_AVAILABLE:
            return
        self._queue.put(("plot_experimental", kwargs))

    # ------------------------------------------------------------------
    # Worker
    # ------------------------------------------------------------------
    def _worker(self) -> None:
        while True:
            item = self._queue.get()
            if item is None:
                self._queue.task_done()
                break
            try:
                kind, payload = item
                if kind == "csv":
                    self._csv_writer.writerows(payload)
                    self._csv_fh.flush()
                elif kind == "secondary":
                    if self._hdf5_file is not None and HDF5_AVAILABLE:
                        _save_transient_hdf5(self._hdf5_file, **payload)
                    elif self._sec_dir is not None:
                        _save_transient_csv(self._sec_dir, **payload)
                elif kind == "plot_experimental":
                    if self._sec_dir is not None:
                        _plot_experimental_pixel(self._sec_dir, **payload)
            except Exception as e:
                try:
                    self._error_cb(f"Secondary save error: {e}")
                except Exception:
                    pass
            self._queue.task_done()
