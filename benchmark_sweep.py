"""Per-pixel sweep-parameter benchmark — the MEASUREMENT half.

Sweeps one pixel repeatedly, varying one acquisition knob at a time, and records
what each variation cost and how precise it was. It fits nothing and plots
nothing: it writes ``benchmark_metrics.csv`` and the per-variation run
directories, then stops.

Turn that into a report (drift detrending, peak-IQE cross-check, plots) with the
analysis repo:

    python benchmark_report.py <root>        # SMILE_IQE_Analysis

The split exists because judging a variation needs the ABC fit, which belongs to
the analysis repo; measuring it needs the hardware, which belongs here. See
RUN_FORMAT.md.

Two design points carried over from the original:

  (a) PRIMARY METRIC = per-point PRECISION at low J (the current & light relative SEM in the
      rising-side band). That is exactly what the averaging knobs (NPLC, N_PM, N_SMU) control,
      and it is DRIFT-IMMUNE — it is the within-point scatter, not a slow trend. Cut a knob and
      read directly whether the low-J noise stays under the analysis gates. It needs no fit,
      so it is computed here and printed at the end of the run.

  (b) INTERLEAVED BASELINES — a baseline sweep is run before every variation, so the slow
      device drift (~10%/tens of minutes, cumulative self-heating) can be measured and the
      peak-IQE cross-check DETRENDED. The spread among baselines gives the reproducibility
      band. The detrending itself is the analysis repo's job; this script just guarantees the
      baselines are there and timestamped.

    python benchmark_sweep.py --pixel 54 33 --repeat 1
    python benchmark_sweep.py --simulate
"""
from __future__ import annotations

import argparse
import csv
import dataclasses
import datetime
import time
from pathlib import Path

import numpy as np
import pandas as pd

import measure_iqe as m
from measure_iqe import IQEConfig

LOWJ = (0.5, 2.0)                 # rising-side band where averaging is stressed & light is faint

BASELINE = dict(n_fine=90, vramp_lowv_nplc=2.0, vramp_lowv_n_pm=32,
                vramp_lowv_n_smu=16, vramp_lowv_settle_ms=200.0, warmup_ms=1500.0)

VARIATIONS = [
    ("N_FINE=60", dict(n_fine=60)),
    ("N_FINE=45", dict(n_fine=45)),
    ("N_FINE=30", dict(n_fine=30)),
    ("lowV_NPLC=1", dict(vramp_lowv_nplc=1.0)),
    ("lowV_NPLC=0.5", dict(vramp_lowv_nplc=0.5)),
    ("lowV_NPM=16", dict(vramp_lowv_n_pm=16)),
    ("lowV_NPM=8", dict(vramp_lowv_n_pm=8)),
    ("lowV_NSMU=8", dict(vramp_lowv_n_smu=8)),
    ("lowV_settle=100", dict(vramp_lowv_settle_ms=100.0)),
    ("lowV_settle=50", dict(vramp_lowv_settle_ms=50.0)),
    ("warmup=500", dict(warmup_ms=500.0)),
    ("LEAN combo", dict(n_fine=60, vramp_lowv_nplc=1.0, vramp_lowv_n_pm=16,
                        vramp_lowv_n_smu=8, vramp_lowv_settle_ms=100.0)),
]

METRIC_COLS = ["kind", "name", "repeat", "t_wall_s", "t_center_unix",
               "cur_rsem", "light_rsem", "run_dir"]


def variant_config(base: IQEConfig, overrides: dict) -> IQEConfig:
    """Baseline config with one variation applied — immutable, no module globals touched."""
    return dataclasses.replace(base, **{**BASELINE, **overrides}).validate()


def low_j_precision(sweep_dir: Path) -> tuple[float, float]:
    """(current relSEM, light relSEM) median over the low-J band of one sweep.

    Reads only the columns RUN_FORMAT.md guarantees, so this needs no fitting and
    no analysis code. Returns (nan, nan) when the sweep produced no low-J points.
    """
    files = sorted(Path(sweep_dir).glob("aggregated_bitval=*_mean_std.csv"))
    if not files:
        return float("nan"), float("nan")
    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    cur = "NVLED_CURR_MEDIAN" if "NVLED_CURR_MEDIAN" in df else "NVLED_CURR_MEAN"
    pwr = "PM400_POWER_MEDIAN" if "PM400_POWER_MEDIAN" in df else "PM400_POWER_MEAN"
    i = df[cur].astype(float).abs()
    p = df[pwr].astype(float)
    j = i / 1e-6                                  # 10x10 um pixel: 1 uA ~ 1 A/cm^2
    sel = (j >= LOWJ[0]) & (j < LOWJ[1]) & (i > 2e-8)
    if not sel.any():
        return float("nan"), float("nan")
    return (float(np.median(df.NVLED_CURR_STD.astype(float)[sel] / i[sel])),
            float(np.median(df.PM400_POWER_STD.astype(float)[sel] / p[sel].clip(lower=1e-30))))


def sweep_once(pm, smu, smile, pixel, run_dir: Path, cfg: IQEConfig) -> dict:
    """One real sweep of one pixel. Returns its timing + drift-immune precision."""
    t0 = time.perf_counter()
    m.run_sweep_voltage(pm, smu, smile, [dict(X=pixel[0], Y=pixel[1])], run_dir,
                        cfg, lambda *_: None)
    t_wall = time.perf_counter() - t0
    cur_rsem, light_rsem = low_j_precision(run_dir / "NVLEDSweep" / "Sweep")
    return dict(t_wall_s=t_wall, t_center_unix=time.time(),
                cur_rsem=cur_rsem, light_rsem=light_rsem, run_dir=run_dir.name)


def write_metrics(root: Path, rows: list[dict]) -> Path:
    """One row per sweep → benchmark_metrics.csv, the handoff to the analysis repo."""
    out = root / "benchmark_metrics.csv"
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=METRIC_COLS)
        w.writeheader()
        w.writerows({c: r.get(c, "") for c in METRIC_COLS} for r in rows)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--simulate", action="store_true")
    ap.add_argument("--out", default=None)
    ap.add_argument("--pixel", nargs=2, type=int, default=None)
    ap.add_argument("--repeat", type=int, default=1,
                    help="repeats per variation (baselines interleaved regardless)")
    args = ap.parse_args()
    log = m.make_logger()      # encode-safe timestamped logger

    base_cfg = IQEConfig()
    base_dir = Path(args.out) if args.out else Path(base_cfg.base_dir)
    root = base_dir / f"{datetime.datetime.now():%y%m%d-%H%M}_BENCHMARK"
    root.mkdir(parents=True, exist_ok=True)
    pm, smu, smile = m.connect(base_cfg, log, simulate=args.simulate)
    m.configure_common(pm, smu, base_cfg, log)
    smu.set_voltage("a", base_cfg.vled_voltage); smu.set_voltage("b", base_cfg.nvled_v_fastscan)
    smu.enable_output("a", True); smu.enable_output("b", True)
    time.sleep(0.05)

    rows: list[dict] = []
    try:
        if args.pixel:
            pixel = tuple(args.pixel)
        else:
            x1, y1, x2, y2 = base_cfg.resolve_roi()
            sel = m.run_fast_scan(pm, smu, smile, x1, y1, x2, y2, root, base_cfg, log)
            pixel = (sel[len(sel) // 2]["X"], sel[len(sel) // 2]["Y"]) if sel else (x1, y1)
        log(f"pixel {pixel}: {len(VARIATIONS)} variations, baseline interleaved before each "
            f"(+ one final) → {len(VARIATIONS)+1} baselines")

        def run(kind, name, ov):
            cfg = variant_config(base_cfg, ov)
            for rep in range(args.repeat if kind == "variation" else 1):
                n = sum(1 for r in rows if r["name"] == name)
                mm = sweep_once(pm, smu, smile, pixel,
                                root / f"{name}_{n}".replace(" ", "_"), cfg)
                mm.update(kind=kind, name=name, repeat=rep)
                rows.append(mm)
                write_metrics(root, rows)        # flush after every sweep — survives an abort
                log(f"  {name:16s} cur_rSEM {100*mm['cur_rsem']:.0f}% "
                    f"light_rSEM {100*mm['light_rsem']:.0f}% {mm['t_wall_s']:.0f}s")

        run("baseline", "baseline", {})                # bracket start
        for name, ov in VARIATIONS:
            run("variation", name, ov)
            run("baseline", "baseline", {})            # baseline before/after each variation
    finally:
        for ch in ("a", "b"):
            try: smu.enable_output(ch, False)
            except Exception: pass
        for d in (pm, smu, smile):
            try: d.close()
            except Exception: pass
        if rows:
            log(f"wrote {write_metrics(root, rows)}")

    summarize(rows)
    log(f"report it with:  python benchmark_report.py \"{root}\"     (SMILE_IQE_Analysis)")


def summarize(rows: list[dict]) -> None:
    """The fit-free primary metric, printed so a run is useful on its own."""
    base = [r for r in rows if r["kind"] == "baseline"]
    var = [r for r in rows if r["kind"] == "variation"]
    if not base or not var:
        print("(no variations measured)")
        return
    b_cur = float(np.nanmedian([r["cur_rsem"] for r in base]))
    b_light = float(np.nanmedian([r["light_rsem"] for r in base]))
    b_t = float(np.nanmedian([r["t_wall_s"] for r in base]))
    print(f"\nbaseline: low-J current relSEM {100*b_cur:.0f}%, light relSEM "
          f"{100*b_light:.0f}%, {b_t:.0f} s/pixel")
    print(f"{'variation':<16} {'cur relSEM':>11} {'light relSEM':>13} {'s/pixel':>8} {'speedup':>8}")
    for r in var:
        sp = b_t / r["t_wall_s"] if r["t_wall_s"] else float("nan")
        print(f"{r['name']:<16} {100*r['cur_rsem']:>10.0f}% {100*r['light_rsem']:>12.0f}% "
              f"{r['t_wall_s']:>8.0f} {sp:>7.2f}x")
    print("\nPeak-IQE cross-check and drift detrending: run benchmark_report.py "
          "in SMILE_IQE_Analysis.")


if __name__ == "__main__":
    main()
