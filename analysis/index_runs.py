"""One CSV row per run: what it is, how it was taken, and its headline numbers.

    python analysis/index_runs.py [ROOT ...] [-o runs_index.csv]

ROOT defaults to the run trees of this repository: results/, demo/, charatests/,
caprun/ and results_georg/results/, where they exist. Every folder named like a run
(yymmdd-hhmm_...) is one row; a run folder inside another run folder is not.

Reads files only and fits nothing, so a few hundred runs take a minute. The IQE columns
come from what analyze_array.py left in the run folder -- abc_fit_per_pixel.csv for the
per-pixel numbers, REPORT.md for the joint fit -- so a run that was never analysed shows
has_report = False and empty IQE columns. A completed measure_iqe run analyses itself.

Nothing in a run folder is written.
"""
from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent
ROOTS = ("results", "demo", "charatests", "caprun", "results_georg/results")
RUN = re.compile(r"^(\d{6}-\d{4,6})_")
DEAD_FRAC, ALIVE_PCTL = 0.20, 90          # the alive rule of analyze_array / measure_iqe


def _json(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def kind_of(d: Path) -> str:
    if any(d.glob("*_calibration.y*ml")):
        return "calibration"
    if (d / "NVLEDSweep").is_dir() and any((d / "NVLEDSweep").rglob("*aggregated*.csv")):
        return "sweep"
    if (d / "FastScan").is_dir():
        return "fastscan"
    if d.name.endswith("_CAPACITANCE"):
        return "bench"
    return "other"


def settings_of(d: Path) -> dict:
    for pat in ("NVLEDSweep/**/raw_data/*config*.json", "FastScan/raw_data/*config*.json",
                "**/*config*.json", "run_config.json"):
        for f in d.glob(pat):
            j = _json(f)
            s = j.get("settings") or j.get("config")
            if s:
                return dict(s, **({"_args": j["args"]} if "args" in j else {}))
    return {}


def fast_scan(d: Path, row: dict) -> None:
    files = [f for f in (d / "FastScan").glob("*aggregated_bitval=*.csv") if "screen" not in f.name]
    if not files:
        return
    fs = pd.read_csv(files[0], usecols=lambda c: c in ("X", "Y", "NVLED_V",
                                                         "PM400_POWER_MEAN", "NVLED_CURR_MEAN"))
    if "NVLED_V" not in fs:
        return
    biases = list(dict.fromkeys(fs["NVLED_V"].round(4)))
    row["fastscan_biases"] = " ".join(f"{b:g}" for b in biases)
    parts = []
    for b in biases:
        g = fs[fs["NVLED_V"].round(4) == b]
        P = g["PM400_POWER_MEAN"].abs()
        alive = P >= DEAD_FRAC * np.nanpercentile(P, ALIVE_PCTL)
        parts.append(f"{b:g}V:{int(alive.sum())}/{len(g)}")
        if b == biases[0]:
            row["fastscan_pixels"] = len(g)
            row["alive_first_bias"] = int(alive.sum())
            row["alive_frac"] = round(float(alive.mean()), 4)
            row["fastscan_P_median_nW"] = round(float(P[alive].median()) * 1e9, 3) if alive.any() else ""
            row["fastscan_I_median_uA"] = (round(float(g["NVLED_CURR_MEAN"].abs()[alive].median()) * 1e6, 3)
                                           if alive.any() else "")
    row["alive_per_bias"] = " ".join(parts)


def sweep(d: Path, row: dict) -> None:
    aggs = sorted((d / "NVLEDSweep").rglob("*aggregated_bitval=*mean_std.csv"))
    if not aggs:
        return
    a = pd.read_csv(aggs[0])
    k = a["POOL_K"].fillna(1) if "POOL_K" in a else pd.Series(1, index=a.index)
    tg = a.assign(K=k).groupby(["X", "Y"])["K"].first()
    row.update(n_targets=len(tg), n_pooled_targets=int((tg > 1).sum()), n_points=len(a),
               n_biases=a["NVLED_V"].round(4).nunique() if "NVLED_V" in a else "",
               has_reference="NVLED_CURR_REF" in a.columns,
               has_cached_reference="LEAK_REF_CACHED" in a.columns)
    fits = sorted((d / "NVLEDSweep").rglob("IQE_EQE/abc_fit_per_pixel.csv"))
    if fits:
        f = pd.read_csv(fits[0])
        if {"clean_fit", "peak_IQE"} <= set(f.columns):
            c = f[(f["clean_fit"] == True) & f["peak_IQE"].notna()]
            row["n_clean_fits"] = len(c)
            if len(c):
                p = 100 * c["peak_IQE"]
                row.update(iqe_pixel_median=round(float(p.median()), 2),
                           iqe_p10=round(float(p.quantile(0.1)), 2),
                           iqe_p90=round(float(p.quantile(0.9)), 2))
                if "POOL_K" in c.columns and (c["POOL_K"] > 1).any():
                    row["iqe_singles_median"] = round(float(p[c["POOL_K"] <= 1].median()), 2)
                    row["iqe_groups_median"] = round(float(p[c["POOL_K"] > 1].median()), 2)


def report(d: Path, row: dict) -> None:
    rep = d / "REPORT.md"
    row["has_report"] = rep.exists()
    if not rep.exists():
        return
    t = rep.read_text(encoding="utf-8", errors="replace")
    m = (re.search(r"Joint fit[^\n]*?peak IQE \*\*([\d.]+) %", t)
         or re.search(r"Peak IQE ≈ ([\d.]+)%\*\* \(pooled", t))     # REPORT.md before 23 Oct
    if m:
        row["iqe_joint_fit"] = float(m.group(1))
    m = re.search(r"_Generated (\d{4}-\d{2}-\d{2} \d{2}:\d{2})", t)
    if m:
        row["report_generated"] = m.group(1)


def calibration(d: Path, row: dict) -> None:
    import yaml
    f = next(d.glob("*_calibration.y*ml"))
    c = yaml.safe_load(f.read_text(encoding="utf-8")) or {}
    rec = c.get("recommended") or {}
    row.update(cal_dwell_rising_s=rec.get("vramp_lowv_settle_s", ""),
               cal_dwell_droop_s=rec.get("vramp_highv_settle_s", ""),
               cal_target=c.get("target", ""), cal_window_s=c.get("seconds_per_capture", ""),
               cal_pixels=" ".join(f"{x},{y}" for x, y in (c.get("pixels") or [])))


def bench(d: Path, row: dict) -> None:
    exps = {f.stem: sum(1 for _ in f.open(encoding="utf-8", errors="replace")) - 1
            for f in d.glob("*.csv")}
    row["experiments"] = " ".join(f"{k}:{v}" for k, v in sorted(exps.items()))
    row["has_traces"] = (d / "traces.npz").exists()


def index_run(d: Path, base: Path) -> dict:
    rid = RUN.match(d.name).group(1)
    s = settings_of(d)
    prov = _json(d / "RUN_PROVENANCE.json")
    kind = kind_of(d)
    row = dict(run=rid, kind=kind, folder=str(d.relative_to(base)),
               sample=s.get("sample_name") or prov.get("sample_name", ""),
               simulated="SIMULATED" in d.name, roi=s.get("roi", ""),
               commit=prov.get("commit_short", ""), dirty=prov.get("dirty", ""),
               sweep_order=s.get("sweep_order", "") or ("pixel_major" if kind == "sweep" else ""),
               direction=s.get("sweep_direction", ""), n_fine=s.get("n_fine", ""),
               v_start=s.get("nvled_v_coarse_start", ""), v_stop=s.get("nvled_v_coarse_stop", ""),
               dwell_rising_s=s.get("vramp_lowv_settle_s", ""),
               dwell_droop_s=s.get("vramp_highv_settle_s", ""),
               highc_b=s.get("highc_b", ""), leak_reference=s.get("leak_reference", ""),
               pool_groups=" ".join(map(str, s.get("pool_groups") or [])),
               size_mb=round(sum(f.stat().st_size for f in d.rglob("*") if f.is_file()) / 1e6, 1))
    for fn, when in ((fast_scan, kind in ("sweep", "fastscan")), (sweep, kind == "sweep"),
                     (report, kind in ("sweep", "fastscan")), (calibration, kind == "calibration"),
                     (bench, kind == "bench")):
        if when:
            try:
                fn(d, row)
            except Exception as e:                   # one broken run must not stop the index
                row.setdefault("errors", "")
                row["errors"] += f"{fn.__name__}: {type(e).__name__}: {e}; "
    return row


def run_dirs(root: Path):
    """Run folders under root, not descending into a run folder once found."""
    for p in sorted(root.iterdir()) if root.is_dir() else []:
        if not p.is_dir():
            continue
        if RUN.match(p.name):
            yield p
        else:
            yield from run_dirs(p)


def main(argv=None) -> Path:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("roots", nargs="*", help="run trees (default: this repo's)")
    ap.add_argument("-o", "--out", default=str(REPO / "runs_index.csv"))
    args = ap.parse_args(argv)
    roots = [Path(r) for r in args.roots] or [REPO / r for r in ROOTS if (REPO / r).is_dir()]
    rows = []
    for root in roots:
        base = REPO if root.resolve().is_relative_to(REPO) else root
        for d in run_dirs(root):
            rows.append(index_run(d, base))
            print(f"  {rows[-1]['run']} {rows[-1]['kind']:<11} {d.name[:60]}")
    cols = list(dict.fromkeys(k for r in rows for k in r))
    out = Path(args.out)
    with out.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {out}: {len(rows)} runs")
    return out


if __name__ == "__main__":
    main()
