"""
smile_postprocess.py — Standalone post-processing module for SMILE measurements.
Can be imported by the GUI or run as a CLI tool.
"""

VERSION = "20260415_v1"

import csv
import glob
import argparse
import numpy as np
from pathlib import Path


# ---------------------------------------------------------------------------
# Helpers (copied from GUI)
# ---------------------------------------------------------------------------


def steady_state_stats(arr, tail_pct=20.0):
    """Returns (mean_str, std_str) of the last tail_pct% of arr."""
    if not arr:
        return "nan", "nan"
    n = max(1, int(len(arr) * tail_pct / 100.0))
    tail = np.array(arr[-n:], dtype=np.float64)
    return f"{float(tail.mean()):.6e}", f"{float(tail.std()):.6e}"


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


def _read_all_channels(fpath):
    """Return dict of TYPE → sorted [(time_s, value)] from a secondary CSV."""
    channels = {}
    with open(fpath, newline="") as f:
        reader = csv.reader(f)
        header = next(reader)
        t_idx = header.index("TIME_s")
        type_idx = header.index("TYPE")
        val_idx = header.index("VALUE")
        for row in reader:
            tp = row[type_idx]
            channels.setdefault(tp, []).append((float(row[t_idx]), float(row[val_idx])))
    for tp in channels:
        channels[tp].sort(key=lambda r: r[0])
    return channels


_NOISE_WINDOW_MS = 3.0  # ms of leading dark samples used for noise floor


def _noise_floor(ts_ms, vs, t_ack_ms=None, log_fn=None):
    """
    Estimate noise floor from the first 3 ms of PM400 samples (dark period
    before any light arrives).

    If ``t_ack_ms`` is provided and is shorter than the noise window, a
    warning is emitted because the ACK already returned before 3 ms elapsed —
    meaning the pixel may be at full brightness inside the noise window and
    the estimate will be unreliable.

    Returns (noise_mean, noise_std) — both in the same units as vs.
    """
    mask = ts_ms < _NOISE_WINDOW_MS
    if mask.sum() >= 2:
        dark = vs[mask]
    else:
        dark = vs[: max(2, 1)]  # degenerate: fewer than 2 samples in 3 ms

    if t_ack_ms is not None and t_ack_ms < _NOISE_WINDOW_MS:
        msg = (
            f"_noise_floor: ACK returned at {t_ack_ms:.2f} ms < {_NOISE_WINDOW_MS} ms — "
            f"noise window may include illuminated samples; floor estimate may be too high."
        )
        if log_fn:
            log_fn(msg)

    return float(np.mean(dark)), float(np.std(dark))


def _estimate_on_time_ma(ts_ms, vs, t_turnoff_ms=None, ma_window=5, threshold=0.9):
    """
    Moving-average on-time estimator.

    Scans forward from the start and returns the time (ms) at which the
    moving average of PM400 power first exceeds ``threshold`` × the mean of
    the last 10 % of the on-portion.  Dark-tail samples (after
    ``t_turnoff_ms``) are excluded from the reference and the scan target.

    Returns float("nan") if estimation fails.
    """
    if len(vs) < 10:
        return float("nan")
    # Restrict to on-portion (exclude dark tail)
    if t_turnoff_ms is not None:
        mask = ts_ms < t_turnoff_ms
        on_ts = ts_ms[mask]
        on_vs = vs[mask]
    else:
        on_ts = ts_ms
        on_vs = vs
    if len(on_vs) < 10:
        return float("nan")
    # Reference: mean of last 10 % of on-portion
    n_ref = max(1, int(len(on_vs) * 0.10))
    reference = float(np.mean(on_vs[-n_ref:]))
    if reference <= 0:
        return float("nan")
    thresh = threshold * reference
    half_w = max(1, ma_window // 2)
    for i in range(len(on_vs)):
        lo = max(0, i - half_w)
        hi = min(len(on_vs), i + half_w + 1)
        ma = float(np.mean(on_vs[lo:hi]))
        if ma >= thresh:
            return float(on_ts[i])
    return float("nan")


def _estimate_on_time_validated(
    ts_ms, vs, t_turnoff_ms, on_threshold, stay_on_frac=0.5
):
    """
    Wrapper around ``_estimate_on_time_ma`` that rejects spike detections.

    After finding ``t_on_ms``, checks that the mean signal from ``t_on_ms``
    to the end of the on-portion (before ``t_turnoff_ms``) is at least
    ``on_threshold``.  If the signal collapses back to noise immediately
    after the detected edge (a spike), returns ``float("nan")`` so the pixel
    is treated as dead.

    ``stay_on_frac`` is unused currently but kept as a hook for future
    fractional-threshold tuning relative to on_threshold.
    """
    t_on_ms = _estimate_on_time_ma(ts_ms, vs, t_turnoff_ms=t_turnoff_ms)
    if not np.isfinite(t_on_ms):
        return float("nan")

    # Samples after the detected on-time, still within the on-portion
    if t_turnoff_ms is not None:
        stay_mask = (ts_ms >= t_on_ms) & (ts_ms < t_turnoff_ms)
    else:
        stay_mask = ts_ms >= t_on_ms

    if stay_mask.sum() < 2:
        # Too few samples to verify — accept the detection
        return t_on_ms

    mean_after = float(np.mean(vs[stay_mask]))
    if mean_after < on_threshold:
        return float("nan")  # signal didn't stay on → spike
    return t_on_ms


# ---------------------------------------------------------------------------
# Core post-processing functions
# ---------------------------------------------------------------------------


def post_process_data(raw_csv_path, out_dir, log_fn=print):
    """
    Aggregate and compute yield statistics from the raw measurement CSV.

    Args:
        raw_csv_path: Path to the raw CSV file.
        out_dir: Path to the run output directory (stats/ sub-dir will be created).
        log_fn: Callable for logging messages (default: print).
    """
    raw_csv_path = Path(raw_csv_path)
    out_dir = Path(out_dir)
    try:
        import pandas as pd

        filtered_out_dir = out_dir / "stats"
        filtered_out_dir.mkdir(parents=True, exist_ok=True)
        df = pd.read_csv(raw_csv_path)
        if df.empty:
            return

        grouped = df.groupby(["X", "Y", "BITVAL", "NVLED_V"])
        has_std_col = "MEAS_STD" in df.columns

        def _grp_stats(sub_df):
            vals = sub_df["MEAS_VALUE"]
            if vals.empty:
                return np.nan, np.nan
            mean = float(vals.mean())
            if len(vals) > 1:
                std = float(vals.std())
            elif has_std_col:
                try:
                    std = float(sub_df["MEAS_STD"].iloc[0])
                    if np.isnan(std):
                        std = np.nan
                except Exception:
                    std = np.nan
            else:
                std = np.nan
            return mean, std

        agg_rows = []
        for name, group in grouped:
            x, y, bitval, nvled_v = name
            pm_mean, pm_std = _grp_stats(group[group["TYPE"] == "PM400"])
            vl_mean, vl_std = _grp_stats(group[group["TYPE"] == "VLED"])
            nv_mean, nv_std = _grp_stats(group[group["TYPE"] == "NVLED"])
            agg_rows.append(
                {
                    "X": int(x),
                    "Y": int(y),
                    "BITVAL": int(bitval),
                    "NVLED_V": round(nvled_v, 3),
                    "TIME_START": round(group["TIME"].min(), 6),
                    "TIME_END": round(group["TIME"].max(), 6),
                    "NVLED_CURR_MEAN": nv_mean,
                    "NVLED_CURR_STD": nv_std,
                    "VLED_CURR_MEAN": vl_mean,
                    "VLED_CURR_STD": vl_std,
                    "PM400_POWER_MEAN": pm_mean,
                    "PM400_POWER_STD": pm_std,
                }
            )
        df_agg = pd.DataFrame(agg_rows)
        if not df_agg.empty:
            for b_val, group_df in df_agg.groupby("BITVAL"):
                group_df.to_csv(
                    out_dir / f"aggregated_bitval={int(b_val)}_mean_std.csv",
                    index=False,
                )

        # -- Scatter plots: optical power vs NVLED / VLED current ----------
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            if not df_agg.empty:
                plots_dir = filtered_out_dir  # stats/
                bitvals = sorted(df_agg["BITVAL"].unique())
                cmap = plt.cm.get_cmap("tab10", max(len(bitvals), 1))

                for x_col, x_label, fname_suffix in [
                    ("NVLED_CURR_MEAN", "Average NVLED current (A)", "nvled"),
                    ("VLED_CURR_MEAN", "Average VLED current (A)", "vled"),
                ]:
                    fig, ax = plt.subplots(figsize=(8, 6))
                    for ci, bv in enumerate(bitvals):
                        sub = df_agg[df_agg["BITVAL"] == bv]
                        ax.scatter(
                            sub[x_col], sub["PM400_POWER_MEAN"],
                            s=8, alpha=0.6, color=cmap(ci),
                            label=f"BITVAL {int(bv)}",
                        )
                    ax.set_xlabel(x_label)
                    ax.set_ylabel("Average optical power (W)")
                    ax.set_title(f"Optical power vs {x_label.split('(')[0].strip()}")
                    ax.legend(fontsize=7, markerscale=2)
                    ax.grid(True, alpha=0.3)
                    fig.tight_layout()
                    fig.savefig(str(plots_dir / f"scatter_power_vs_{fname_suffix}.png"), dpi=200)
                    plt.close(fig)
                log_fn("Scatter plots saved (power vs NVLED, power vs VLED).")
        except ImportError:
            log_fn("Scatter plots skipped — matplotlib not installed.")
        except Exception as e:
            log_fn(f"Scatter plot error: {e}")

        yield_stats = []
        for (bitval, nvled_v, mtype), group in df.groupby(
            ["BITVAL", "NVLED_V", "TYPE"]
        ):
            px_avg = group.groupby(["X", "Y"])["MEAS_VALUE"].mean().reset_index()
            px_avg["BITVAL"] = int(bitval)
            px_avg["NVLED_V"] = nvled_v
            px_avg["TYPE"] = mtype
            px_avg = px_avg[["X", "Y", "BITVAL", "NVLED_V", "TYPE", "MEAS_VALUE"]]
            vals = px_avg["MEAS_VALUE"]
            if vals.empty:
                continue
            mean_val, std_val = vals.mean(), vals.std()
            median_val = vals.median()
            q1, q3 = vals.quantile(0.25), vals.quantile(0.75)
            iqr = q3 - q1
            total_pixels = len(vals)

            def save_cond(cond, suffix):
                filtered = px_avg[cond]
                if not filtered.empty:
                    filtered.to_csv(
                        filtered_out_dir
                        / f"pivot_bitval={int(bitval)}_nvled={nvled_v:.3f}_{mtype}_{suffix}.csv",
                        index=False,
                    )
                return len(filtered)

            n_low_out = save_cond(vals < (q1 - 1.5 * iqr), "low_outliers")
            n_high_out = save_cond(vals > (q3 + 1.5 * iqr), "high_outliers")
            n_inliers = save_cond(
                (vals >= (q1 - 1.5 * iqr)) & (vals <= (q3 + 1.5 * iqr)),
                "inliers_iqr",
            )
            n_1std = save_cond(
                (vals >= mean_val - std_val) & (vals <= mean_val + std_val),
                "inside_1std",
            )
            n_2std = save_cond(
                (vals >= mean_val - 2 * std_val) & (vals <= mean_val + 2 * std_val),
                "inside_2std",
            )
            n_3std = save_cond(
                (vals >= mean_val - 3 * std_val) & (vals <= mean_val + 3 * std_val),
                "inside_3std",
            )
            save_cond(vals < (mean_val - 3 * std_val), "outside_3std_low")
            save_cond(vals > (mean_val + 3 * std_val), "outside_3std_high")

            if mtype == "PM400" and total_pixels > 0:
                dead_mask = (vals < (0.10 * median_val)) if median_val > 0 else pd.Series(False, index=vals.index)
                dead_pixels = int(dead_mask.sum())
                alive_vals = vals[~dead_mask]
                n_alive = len(alive_vals)

                if n_alive > 0:
                    alive_mean = alive_vals.mean()
                    alive_std = alive_vals.std() if n_alive > 1 else 0.0
                    cv = (alive_std / alive_mean) if alive_mean != 0 else np.nan
                    n_1std_alive = int(((alive_vals >= alive_mean - alive_std) & (alive_vals <= alive_mean + alive_std)).sum())
                    n_2std_alive = int(((alive_vals >= alive_mean - 2 * alive_std) & (alive_vals <= alive_mean + 2 * alive_std)).sum())
                    n_3std_alive = int(((alive_vals >= alive_mean - 3 * alive_std) & (alive_vals <= alive_mean + 3 * alive_std)).sum())
                else:
                    alive_mean = np.nan
                    alive_std = np.nan
                    cv = np.nan
                    n_1std_alive = 0
                    n_2std_alive = 0
                    n_3std_alive = 0

                yield_stats.append(
                    {
                        "BITVAL": int(bitval),
                        "NVLED_V": round(nvled_v, 3),
                        "TOTAL_PIXELS": total_pixels,
                        "ALIVE_PIXELS": n_alive,
                        "DEAD_PIXELS_COUNT": dead_pixels,
                        "YIELD_ALIVE_%": round((n_alive / total_pixels) * 100, 2),
                        "MEAN_POWER": float(f"{alive_mean:.4e}") if np.isfinite(alive_mean) else np.nan,
                        "STD_POWER": float(f"{alive_std:.4e}") if np.isfinite(alive_std) else np.nan,
                        "COEF_OF_VARIATION_CV": round(cv, 4) if np.isfinite(cv) else np.nan,
                        "YIELD_1STD_%": round((n_1std_alive / n_alive) * 100, 2) if n_alive > 0 else np.nan,
                        "YIELD_2STD_%": round((n_2std_alive / n_alive) * 100, 2) if n_alive > 0 else np.nan,
                        "YIELD_3STD_%": round((n_3std_alive / n_alive) * 100, 2) if n_alive > 0 else np.nan,
                        "COUNT_1STD": n_1std_alive,
                        "COUNT_2STD": n_2std_alive,
                        "COUNT_3STD": n_3std_alive,
                    }
                )
        if yield_stats:
            pd.DataFrame(yield_stats).to_csv(
                out_dir / "pm400_optical_yield_report.csv", index=False
            )
    except Exception as e:
        log_fn(f"Post-processing error: {e}")


def _analyze_on_times(sec_dir, run_dir, log_fn=print):
    """
    Moving-average on-time detection across all secondary CSVs.

    For each waveform: finds the first time at which the moving average
    reaches 90% of the mean of the last 10% of the on-portion (dark tail
    excluded).  Results are saved to ontime_summary.csv and a recommended
    Scan Settle (mean + 3σ) is printed.
    """
    sec_dir = Path(sec_dir)
    run_dir = Path(run_dir)
    pattern = str(sec_dir / "x*_y*_b*.csv")
    files = sorted(glob.glob(pattern))
    if not files:
        return

    # ------------------------------------------------------------------
    # Pass 1 — collect per-pixel peaks and noise floors
    # ------------------------------------------------------------------
    pixel_data = []
    for fpath in files:
        try:
            ch = _read_all_channels(fpath)
            if "PM400" not in ch or len(ch["PM400"]) < 10:
                continue
            ts = np.array([t for t, _ in ch["PM400"]])
            vs = np.array([v for _, v in ch["PM400"]])
            ts_ms = (ts - ts[0]) * 1000.0

            t_ack_ms = None
            if "ACK" in ch:
                t_ack_ms = (ch["ACK"][0][0] - ts[0]) * 1000.0
            t_turnoff_ms = None
            if "TURNOFF_ACK" in ch:
                t_turnoff_ms = (ch["TURNOFF_ACK"][0][0] - ts[0]) * 1000.0

            noise_mean, _ = _noise_floor(ts_ms, vs, t_ack_ms=t_ack_ms, log_fn=log_fn)

            if t_turnoff_ms is not None:
                on_mask = ts_ms < t_turnoff_ms
                on_vs = vs[on_mask] if on_mask.sum() > 0 else vs
            else:
                on_vs = vs
            peak_power = float(np.max(on_vs))

            pixel_data.append(
                {
                    "fpath": fpath,
                    "ts_ms": ts_ms,
                    "vs": vs,
                    "t_turnoff_ms": t_turnoff_ms,
                    "peak_power": peak_power,
                    "noise_mean": noise_mean,
                }
            )
        except Exception as e:
            log_fn(f"On-time analysis: skipping {Path(fpath).name}: {e}")

    if not pixel_data:
        return

    # ------------------------------------------------------------------
    # Global "on" threshold: 10 % of the run-wide maximum peak.
    # Guard: if the global max itself is not meaningfully above the average
    # noise floor, nothing in this run turned on.
    # ------------------------------------------------------------------
    global_peak_max = max(d["peak_power"] for d in pixel_data)
    global_noise_mean = float(np.mean([d["noise_mean"] for d in pixel_data]))
    if global_peak_max > 3.0 * max(abs(global_noise_mean), 1e-14):
        on_threshold = global_peak_max * 0.20
    else:
        on_threshold = float("inf")  # everything is dark

    log_fn(
        f"On-time analysis: global peak={global_peak_max:.3e} W, "
        f"noise floor≈{global_noise_mean:.3e} W, "
        f"on-threshold={on_threshold:.3e} W (20% of peak)."
    )

    # ------------------------------------------------------------------
    # Pass 2 — classify and estimate on-times
    # ------------------------------------------------------------------
    results = []
    for d in pixel_data:
        turned_on = d["peak_power"] > on_threshold
        if turned_on:
            t_on_ms = _estimate_on_time_validated(
                d["ts_ms"],
                d["vs"],
                t_turnoff_ms=d["t_turnoff_ms"],
                on_threshold=on_threshold,
            )
            if not np.isfinite(t_on_ms):
                turned_on = False  # peak passed but signal didn't stay on → spike
        else:
            t_on_ms = float("nan")
        results.append(
            {
                "file": Path(d["fpath"]).stem,
                "t_on_ms": round(t_on_ms, 3) if np.isfinite(t_on_ms) else float("nan"),
                "turned_on": turned_on,
                "peak_power": round(d["peak_power"], 12),
                "noise_mean": round(d["noise_mean"], 12),
            }
        )

    summary_path = run_dir / "ontime_summary.csv"
    with open(summary_path, "w", newline="") as f:
        w = csv.DictWriter(
            f, fieldnames=["file", "t_on_ms", "turned_on", "peak_power", "noise_mean"]
        )
        w.writeheader()
        w.writerows(results)

    on_results = [r for r in results if r["turned_on"]]
    valid = np.array([r["t_on_ms"] for r in on_results if np.isfinite(r["t_on_ms"])])
    n_dead = len(results) - len(on_results)
    if len(valid) > 0:
        mean_ms = float(valid.mean())
        std_ms = float(valid.std())
        rec = mean_ms + 3.0 * std_ms
        msg = (
            f"On-time analysis: {len(valid)}/{len(results)} pixels turned on "
            f"({n_dead} filtered as dark/dead). "
            f"mean={mean_ms:.1f} ms  std={std_ms:.1f} ms  "
            f"→ recommended Scan Settle ≥ {rec:.1f} ms (mean + 3σ). "
            f"Saved: {summary_path.name}"
        )
    else:
        msg = (
            f"On-time analysis: no pixels detected as turned on "
            f"({n_dead}/{len(results)} filtered as dark/dead). "
            f"Global peak={global_peak_max:.3e} W ≤ threshold={on_threshold:.3e} W. "
            f"Saved: {summary_path.name}"
        )
    print(msg)
    log_fn(msg)


def analyze_on_times(sec_dir, run_dir, log_fn=print):
    """Public wrapper for _analyze_on_times."""
    return _analyze_on_times(sec_dir, run_dir, log_fn=log_fn)


def _plot_transient_arrays(sec_dir, run_dir, log_fn=print):
    """
    Save one PNG per secondary CSV showing PM400 (left axis) and
    VLED + negated-NVLED (right axis).  Vertical lines mark set_pixel ACK,
    blank-frame TURNOFF_ACK, and the moving-average estimated on-time.
    Y-axis ranges are fixed to the global min/max across all files in the run.
    """
    sec_dir = Path(sec_dir)
    run_dir = Path(run_dir)
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
        log_fn(msg)
        return

    pattern = str(sec_dir / "x*_y*_b*.csv")
    files = sorted(glob.glob(pattern))
    if not files:
        log_fn("Plot transients: no secondary CSV files found.")
        return

    # ------------------------------------------------------------------
    # Load SMU settings from run config JSON (NPLC + range for legend)
    # ------------------------------------------------------------------
    vled_label = "VLED (A)"
    nvled_label = "NVLED (A, negated)"
    try:
        import json as _json

        config_files = list((run_dir / "raw_data").glob("*_config.json"))
        if config_files:
            with open(config_files[0]) as _f:
                _cfg = _json.load(_f).get("settings", {})
            vled_nplc = _cfg.get("vled_nplc", "?")
            vled_range = _cfg.get("vled_range_i", "?")
            nvled_nplc = _cfg.get("nvled_nplc", "?")
            nvled_range = _cfg.get("nvled_range_i", "?")
            vled_label = f"VLED  (NPLC={vled_nplc}, range={vled_range} A)"
            nvled_label = f"NVLED negated  (NPLC={nvled_nplc}, range={nvled_range} A)"
    except Exception:
        pass

    # ------------------------------------------------------------------
    # Pass 1 — collect global y-axis ranges and per-file PM400 peaks
    # ------------------------------------------------------------------
    pm_min, pm_max = np.inf, -np.inf
    iv_min, iv_max = np.inf, -np.inf  # covers VLED and (−NVLED)
    file_data = {}  # fpath → channels dict
    file_peaks = {}  # fpath → peak PM400 power in on-portion
    file_noise = {}  # fpath → noise_mean
    for fpath in files:
        try:
            ch = _read_all_channels(fpath)
            file_data[fpath] = ch
            if "PM400" in ch:
                vals = [v for _, v in ch["PM400"]]
                pm_min = min(pm_min, min(vals))
                pm_max = max(pm_max, max(vals))

                # Peak in on-portion (before TURNOFF_ACK if present)
                ts_pm = np.array([t for t, _ in ch["PM400"]])
                vs_pm = np.array(vals)
                ts_ms_pm = (ts_pm - ts_pm[0]) * 1000.0
                t_ack_ms_p = (
                    (ch["ACK"][0][0] - ts_pm[0]) * 1000.0 if "ACK" in ch else None
                )
                t_to_ms_p = (
                    (ch["TURNOFF_ACK"][0][0] - ts_pm[0]) * 1000.0
                    if "TURNOFF_ACK" in ch
                    else None
                )
                n_mean, _ = _noise_floor(ts_ms_pm, vs_pm, t_ack_ms=t_ack_ms_p)
                file_noise[fpath] = n_mean
                if t_to_ms_p is not None:
                    on_mask = ts_ms_pm < t_to_ms_p
                    on_vs = vs_pm[on_mask] if on_mask.sum() > 0 else vs_pm
                else:
                    on_vs = vs_pm
                file_peaks[fpath] = float(np.max(on_vs))

            if "VLED" in ch:
                vals = [v for _, v in ch["VLED"]]
                iv_min = min(iv_min, min(vals))
                iv_max = max(iv_max, max(vals))
            if "NVLED" in ch:
                vals = [-v for _, v in ch["NVLED"]]  # negated
                iv_min = min(iv_min, min(vals))
                iv_max = max(iv_max, max(vals))
        except Exception as e:
            log_fn(f"Plot transients (pass 1): {Path(fpath).name}: {e}")

    # Global on-threshold (same logic as _analyze_on_times)
    if file_peaks:
        _global_peak = max(file_peaks.values())
        _global_noise = float(np.mean(list(file_noise.values()))) if file_noise else 0.0
        if _global_peak > 3.0 * max(abs(_global_noise), 1e-14):
            _on_threshold = _global_peak * 0.10
        else:
            _on_threshold = float("inf")
    else:
        _on_threshold = float("inf")

    def _ylim(lo, hi, margin_frac=0.05):
        if not (np.isfinite(lo) and np.isfinite(hi)):
            return None
        if hi <= lo:
            return (lo - abs(lo) * 0.1, hi + abs(hi) * 0.1 + 1e-30)
        m = (hi - lo) * margin_frac
        return (lo - m, hi + m)

    pm_ylim = _ylim(pm_min, pm_max)
    iv_ylim = _ylim(iv_min, iv_max)

    # ------------------------------------------------------------------
    # Pass 2 — plot each file
    # ------------------------------------------------------------------
    n_ok = 0
    for fpath, ch in file_data.items():
        try:
            stem = Path(fpath).stem
            fig, ax1 = plt.subplots(figsize=(10, 5))
            ax2 = ax1.twinx()

            # Reference time = first PM400 sample (≈ T0)
            t0_ref = ch["PM400"][0][0] if "PM400" in ch else 0.0

            lines = []

            # PM400 — left axis
            t_turnoff_ms = None
            pm_ms = pm_vs = None
            if "PM400" in ch:
                pm_ts = np.array([t for t, _ in ch["PM400"]])
                pm_vs = np.array([v for _, v in ch["PM400"]])
                pm_ms = (pm_ts - t0_ref) * 1000.0
                (ln,) = ax1.plot(
                    pm_ms, pm_vs, lw=0.8, color="tab:blue", label="PM400 (W)"
                )
                lines.append(ln)

            # VLED — right axis
            if "VLED" in ch:
                vled_ts = np.array([t for t, _ in ch["VLED"]])
                vled_vs = np.array([v for _, v in ch["VLED"]])
                vled_ms = (vled_ts - t0_ref) * 1000.0
                (ln,) = ax2.plot(
                    vled_ms, vled_vs, lw=0.8, color="tab:orange", label=vled_label
                )
                lines.append(ln)

            # NVLED (negated) — right axis
            if "NVLED" in ch:
                nvled_ts = np.array([t for t, _ in ch["NVLED"]])
                nvled_vs = -np.array([v for _, v in ch["NVLED"]])
                nvled_ms = (nvled_ts - t0_ref) * 1000.0
                (ln,) = ax2.plot(
                    nvled_ms, nvled_vs, lw=0.8, color="tab:green", label=nvled_label
                )
                lines.append(ln)

            # ACK vertical line (set_pixel returned)
            if "ACK" in ch:
                t_ack_ms = (ch["ACK"][0][0] - t0_ref) * 1000.0
                vl = ax1.axvline(
                    t_ack_ms,
                    color="tab:purple",
                    ls="--",
                    lw=1.0,
                    label=f"set_pixel ACK ({t_ack_ms:.2f} ms)",
                )
                lines.append(vl)

            # TURNOFF_ACK vertical line (blank frame returned)
            if "TURNOFF_ACK" in ch:
                t_turnoff_ms = (ch["TURNOFF_ACK"][0][0] - t0_ref) * 1000.0
                vl = ax1.axvline(
                    t_turnoff_ms,
                    color="tab:red",
                    ls=":",
                    lw=1.0,
                    label=f"TURNOFF_ACK ({t_turnoff_ms:.2f} ms)",
                )
                lines.append(vl)

            # Moving-average on-time estimate — only for pixels that turned on
            pixel_peak = file_peaks.get(fpath, 0.0)
            if pm_ms is not None and pm_vs is not None and pixel_peak > _on_threshold:
                t_on_ms = _estimate_on_time_validated(
                    pm_ms, pm_vs, t_turnoff_ms=t_turnoff_ms, on_threshold=_on_threshold
                )
                if np.isfinite(t_on_ms):
                    vl = ax1.axvline(
                        t_on_ms,
                        color="tab:cyan",
                        ls="-.",
                        lw=1.2,
                        label=f"t_on ≈ {t_on_ms:.2f} ms",
                    )
                    lines.append(vl)

            # Apply fixed y-axis ranges
            if pm_ylim:
                ax1.set_ylim(pm_ylim)
            if iv_ylim:
                ax2.set_ylim(iv_ylim)

            ax1.set_xlabel("Time (ms)")
            ax1.set_ylabel("Optical power (W)", color="tab:blue")
            ax2.set_ylabel("Current (A)")
            ax1.tick_params(axis="y", labelcolor="tab:blue")
            ax1.set_title(f"Transient — {stem.replace('_', '  ')}")
            ax1.grid(True, alpha=0.3)
            ax2.axhline(0, color="gray", lw=0.6, ls="--", alpha=0.4, zorder=0)

            labels = [ln.get_label() for ln in lines]
            ax1.legend(lines, labels, loc="upper right", fontsize=7)

            fig.tight_layout()
            fig.savefig(str(sec_dir / f"{stem}.png"), dpi=300)
            plt.close(fig)
            n_ok += 1
        except Exception as e:
            log_fn(f"Plot transients: skipping {Path(fpath).name}: {e}")

    log_fn(f"Transient plots saved ({n_ok} PNGs) in secondary folder.")


def plot_transient_arrays(sec_dir, run_dir, log_fn=print):
    """Public wrapper for _plot_transient_arrays."""
    return _plot_transient_arrays(sec_dir, run_dir, log_fn=log_fn)


def generate_heatmaps(csv_path, run_dir, log_fn=print):
    """
    Generate heatmap images from the raw measurement CSV.

    Args:
        csv_path: Path to the raw CSV file.
        run_dir: Path to the run output directory (heatmaps/ sub-dir will be created).
        log_fn: Callable for logging messages (default: print).
    """
    csv_path = Path(csv_path)
    run_dir = Path(run_dir)
    log_fn("Generating heatmaps...")
    try:
        import smile_heatmap

        heatmap_out_dir = run_dir / "heatmaps"
        heatmap_out_dir.mkdir(parents=True, exist_ok=True)
        smile_heatmap.generate_images(str(csv_path), str(heatmap_out_dir))
    except Exception as e:
        log_fn(f"Heatmap error: {e}")


def _stamp_config_json(run_dir, log_fn=print):
    """Update the run's config JSON with the current postprocess VERSION."""
    run_dir = Path(run_dir)
    raw_data_dir = run_dir / "raw_data"
    if not raw_data_dir.exists():
        return
    config_files = list(raw_data_dir.glob("*_config.json"))
    if not config_files:
        return
    import json

    config_path = config_files[0]
    try:
        with open(config_path) as f:
            data = json.load(f)
        versions = data.get("versions", {})
        versions["smile_postprocess_last_run"] = VERSION
        data["versions"] = versions
        with open(config_path, "w") as f:
            json.dump(data, f, indent=4)
    except Exception as e:
        log_fn(f"Config JSON stamp error: {e}")


def run_postprocess(
    run_dir,
    do_aggregate=True,
    do_heatmaps=True,
    do_ontimes=True,
    do_plots=False,
    log_fn=print,
):
    """
    Orchestrate all post-processing steps for a measurement run directory.

    Args:
        run_dir: Path to the run directory.
        do_aggregate: Whether to run aggregation + yield stats.
        do_heatmaps: Whether to generate heatmaps.
        do_ontimes: Whether to run on-time analysis (requires transient CSVs).
        do_plots: Whether to plot PM400 transients (requires matplotlib).
        log_fn: Callable for logging messages (default: print).
    """
    run_dir = Path(run_dir)
    _stamp_config_json(run_dir, log_fn=log_fn)

    # Find raw CSV
    csv_files = (
        list((run_dir / "raw_data").glob("*.csv"))
        if (run_dir / "raw_data").exists()
        else []
    )
    # exclude config JSONs — find the main data CSV (not timing or stats)
    csv_files = [
        f for f in csv_files if not f.name.startswith("timing") and f.suffix == ".csv"
    ]
    csv_path = csv_files[0] if csv_files else None

    # Find sec_dir (transient data)
    sec_dir = run_dir / "transient_data"
    if not sec_dir.exists():
        sec_dir = None

    if do_aggregate and csv_path is not None:
        log_fn("Post-processing data (aggregation + yield)...")
        post_process_data(csv_path, run_dir, log_fn=log_fn)
    elif do_aggregate:
        log_fn("Aggregation skipped: no raw CSV found in raw_data/")

    if do_heatmaps and csv_path is not None:
        generate_heatmaps(csv_path, run_dir, log_fn=log_fn)
    elif do_heatmaps:
        log_fn("Heatmaps skipped: no raw CSV found in raw_data/")

    if do_ontimes and sec_dir is not None:
        try:
            _analyze_on_times(sec_dir, run_dir, log_fn=log_fn)
        except Exception as e:
            log_fn(f"On-time analysis error: {e}")
    elif do_ontimes:
        log_fn("On-time analysis skipped: no transient_data directory found.")

    if do_plots and sec_dir is not None:
        try:
            _plot_transient_arrays(sec_dir, run_dir, log_fn=log_fn)
        except Exception as e:
            log_fn(f"Transient plot error: {e}")
    elif do_plots:
        log_fn("Transient plots skipped: no transient_data directory found.")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="SMILE post-processing: aggregate, heatmaps, on-times, transient plots."
    )
    parser.add_argument("run_dir", help="Path to a measurement run folder.")
    parser.add_argument(
        "--no-aggregate", action="store_true", help="Skip aggregation + yield stats."
    )
    parser.add_argument(
        "--no-heatmaps", action="store_true", help="Skip heatmap generation."
    )
    parser.add_argument(
        "--no-ontimes", action="store_true", help="Skip on-time analysis."
    )
    parser.add_argument(
        "--plots",
        action="store_true",
        help="Generate PM400 transient plots (requires matplotlib).",
    )
    args = parser.parse_args()

    run_postprocess(
        run_dir=args.run_dir,
        do_aggregate=not args.no_aggregate,
        do_heatmaps=not args.no_heatmaps,
        do_ontimes=not args.no_ontimes,
        do_plots=args.plots,
        log_fn=print,
    )
