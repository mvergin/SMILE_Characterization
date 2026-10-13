"""Report a sweep-parameter benchmark — the ANALYSIS half.

    python benchmark_report.py <BENCHMARK_root>

Reads the ``benchmark_metrics.csv`` and per-variation run directories that
``benchmark_sweep.py`` (SMILE_Characterization) measured, then does the parts
that need fitting:

  * the ABC fit per sweep, for the peak-IQE cross-check;
  * DETRENDING against the interleaved baselines — the device's EQE drifts ~10%
    over tens of minutes of back-to-back sweeps (cumulative self-heating), so a
    raw peak-IQE difference between two variations is mostly drift. A baseline
    sweep was run before every variation, so the drift is measured and
    subtracted, and the spread among baselines gives the reproducibility band.

The precision metric (low-J relative SEM) needs no fit and is already in
``benchmark_metrics.csv``; it is carried through here for the table.

Writes ``<root>/benchmark_report/{table.md, benchmark.png}``.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import analyze_array as aa


def peak_iqe(run_dir: Path) -> tuple[float, float]:
    """(peak IQE %, R²) from one sweep's aggregated CSV, or (nan, nan).

    Uses the documented run format: concatenate the aggregated bitval CSVs, enrich,
    keep the reliable points, fit. nan whenever the fit is absent or the peak was
    not captured — a variation that loses the peak is exactly what we want to see.
    """
    sweep = Path(run_dir) / "NVLEDSweep" / "Sweep"
    files = sorted(sweep.glob("aggregated_bitval=*_mean_std.csv"))
    if not files:
        return float("nan"), float("nan")
    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    enriched = aa.enrich(df, dark_W=aa._dark_power(sweep))
    rel = enriched[enriched.RELIABLE]
    if len(rel) < 6:
        return float("nan"), float("nan")
    fit = aa.fit_abc(rel.J_LED.values, rel.EQE_LED.values)
    if not fit:
        return float("nan"), float("nan")
    peak = 100 * float(fit["peak_IQE"]) if fit.get("peak_captured") else float("nan")
    return peak, float(fit["R_squared"])


def load(root: Path) -> pd.DataFrame:
    """benchmark_metrics.csv + the fitted peak for each sweep."""
    metrics = Path(root) / "benchmark_metrics.csv"
    if not metrics.exists():
        sys.exit(f"{metrics} not found — run benchmark_sweep.py first (SMILE_Characterization)")
    df = pd.read_csv(metrics)
    fits = [peak_iqe(Path(root) / d) for d in df["run_dir"]]
    df["peak"] = [f[0] for f in fits]
    df["r2"] = [f[1] for f in fits]
    df["t_rel"] = df["t_center_unix"] - df["t_center_unix"].min()
    return df


def drift_model(base: pd.DataFrame) -> tuple[float, float, float]:
    """(slope %/s, intercept, reproducibility band) from the baseline peaks over time."""
    ok = base["peak"].notna()
    if ok.sum() >= 2:
        slope, intercept = np.polyfit(base["t_rel"][ok], base["peak"][ok], 1)
        resid = base["peak"][ok] - (slope * base["t_rel"][ok] + intercept)
        return float(slope), float(intercept), max(float(np.std(resid)), 1.0)
    return 0.0, float(base["peak"].median()) if ok.any() else float("nan"), 2.0


def report(root: Path) -> None:
    df = load(root)
    base = df[df["kind"] == "baseline"]
    var = df[df["kind"] == "variation"]
    if base.empty or var.empty:
        sys.exit("need at least one baseline and one variation")
    slope, intercept, band = drift_model(base)
    b_cur = float(np.nanmedian(base["cur_rsem"]))
    b_light = float(np.nanmedian(base["light_rsem"]))
    b_t = float(np.nanmedian(base["t_wall_s"]))
    bp = base["peak"].dropna()

    lines = [
        "# Sweep-parameter benchmark (drift-robust)", "",
        f"**Measured device drift:** baseline peak IQE trends **{slope*60:+.1f}%/min** over the "
        + (f"run (first {bp.iloc[0]:.0f}% → last {bp.iloc[-1]:.0f}%)" if len(bp) >= 2 else "run")
        + f"; reproducibility band ±{band:.0f}%. This is why raw single-sweep peak IQE is "
          "unreliable — it is DETRENDED below.", "",
        f"Baseline low-J precision: current relSEM **{100*b_cur:.0f}%**, light relSEM "
        f"**{100*b_light:.0f}%**, {b_t:.0f} s/pixel.", "",
        "Verdict = SAFE if the cut keeps low-J current relSEM <50% (the gate) with margin, "
        "light relSEM not much worse, and the DRIFT-CORRECTED peak within the ±band.", "",
        "| variation | cur relSEM | light relSEM | Δpeak (detrended) | s/pixel | speedup | verdict |",
        "|---|---|---|---|---|---|---|",
    ]
    print("\n".join(lines[2:3] + lines[6:]))
    for _, v in var.iterrows():
        expected = slope * v["t_rel"] + intercept     # baseline peak expected at this time
        dpk = v["peak"] - expected if np.isfinite(v["peak"]) else np.nan
        speed = b_t / v["t_wall_s"] if v["t_wall_s"] else float("nan")
        cr, lr = v["cur_rsem"], v["light_rsem"]
        safe = (np.isfinite(cr) and cr < 0.5 and cr <= max(0.30, 1.6 * b_cur)
                and (not np.isfinite(lr) or lr <= max(0.08, 1.6 * b_light))
                and (not np.isfinite(dpk) or abs(dpk) <= 2 * band))
        lines.append(f"| {v['name']} | {100*cr:.0f}% | {100*lr:.0f}% | {dpk:+.0f}% | "
                     f"{v['t_wall_s']:.0f} | {speed:.2f}× | {'✅ safe' if safe else '⚠ check'} |")
        print(f"  {v['name']:16s} cur {100*cr:.0f}%  light {100*lr:.0f}%  Δpk{dpk:+.0f}  "
              f"{v['t_wall_s']:.0f}s  {speed:.2f}×  {'safe' if safe else 'check'}")

    out = Path(root) / "benchmark_report"
    out.mkdir(exist_ok=True)
    (out / "table.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"\nwrote {out / 'table.md'}")
    try:
        _plot(base, var, slope, intercept, b_cur, b_light, out / "benchmark.png")
        print(f"wrote {out / 'benchmark.png'}")
    except Exception as exc:
        print(f"(plot skipped: {exc})")


def _plot(base, var, slope, intercept, b_cur, b_light, out_png) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 5))
    # left: primary metric — low-J relSEM per variation vs baseline
    x = np.arange(len(var))
    a1.bar(x - 0.2, 100 * var["cur_rsem"], 0.4, label="current relSEM", color="tab:purple")
    a1.bar(x + 0.2, 100 * var["light_rsem"], 0.4, label="light relSEM", color="tab:green")
    a1.axhline(100 * b_cur, color="tab:purple", ls="--", lw=1, label="baseline cur")
    a1.axhline(100 * b_light, color="tab:green", ls="--", lw=1, label="baseline light")
    a1.axhline(50, color="red", ls=":", lw=1, label="current gate 50%")
    a1.set_xticks(x)
    a1.set_xticklabels(var["name"], rotation=60, ha="right", fontsize=7)
    a1.set_ylabel("low-J relSEM (%)")
    a1.set_title("Per-point precision (drift-immune) — the averaging metric")
    a1.legend(fontsize=7)
    a1.grid(True, axis="y", alpha=0.3)
    # right: baseline drift + the variation peaks against it
    a2.scatter(base["t_rel"], base["peak"], color="tab:red", label="baselines (drift)", zorder=3)
    tt = np.linspace(base["t_rel"].min(), base["t_rel"].max(), 50)
    a2.plot(tt, slope * tt + intercept, "r--", lw=1, label=f"drift {slope*60:+.1f}%/min")
    for _, v in var.iterrows():
        a2.scatter([v["t_rel"]], [v["peak"]], color="tab:blue", s=20)
        a2.annotate(v["name"], (v["t_rel"], v["peak"]), fontsize=6,
                    textcoords="offset points", xytext=(4, 3))
    a2.set_xlabel("time (s)")
    a2.set_ylabel("peak IQE (%)")
    a2.set_title("Device drift over the run (baselines) + variations\n"
                 "(judge variations vs the red drift line)")
    a2.legend(fontsize=8)
    a2.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_png, dpi=140)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("root", help="the *_BENCHMARK directory written by benchmark_sweep.py")
    report(Path(ap.parse_args().root))


if __name__ == "__main__":
    main()
