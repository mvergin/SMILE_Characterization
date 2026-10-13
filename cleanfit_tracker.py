"""Clean-fit tracker: one table of clean per-pixel-fit % per run, to watch progress
toward the 90% goal. Re-run after each new measurement.

A fit is "clean" (trustworthy) iff the IQE peak is captured in the data, R² ≥ 0.70, and
0.05 ≤ β ≤ 20 (not a degenerate runaway). Same definition as analyze_array.classify_fit.

    python cleanfit_tracker.py            # print table + write investigation_report/cleanfit_tracker.md
"""
from __future__ import annotations
import glob
import re
from pathlib import Path

import numpy as np
import pandas as pd

R2MIN, BETA_MAX, BETA_MIN = 0.70, 20.0, 0.05
# Full-population fits (both sweep directions, 16–20 clean pixels) put the peak at J≈11–14 — this
# is a sidewall-SRH-limited 10 µm µLED, so the peak sits at high current density. Earlier "J≈5"
# came from 1–5-pixel small samples; "J≈2/77%" (run 1107) was a 4-pixel low-J-offset outlier. So
# the reference is J≈12, and we only warn on fits well beyond that (J > ~30) as genuinely runaway.
PEAK_J_REF = 12.0             # this device's population peak location (A/cm²)
PEAK_J_WARN = 2.5             # warn if median J_peak > PEAK_J_WARN×PEAK_J_REF (=30) → runaway fit
RESULTS = Path("results_georg/results")
OUT = Path("investigation_report")


def clean_stats(pf):
    df = pd.read_csv(pf)
    if not {"beta", "peak_captured", "R2"}.issubset(df.columns):
        return None
    m = ((df.peak_captured == True) & (df.R2 >= R2MIN)
         & (df.beta >= BETA_MIN) & (df.beta <= BETA_MAX))
    raw = df["peak_IQE_raw"] if "peak_IQE_raw" in df else df.get("peak_IQE")
    jpk = df["J_at_peak"] if "J_at_peak" in df else None
    return dict(n=len(df), n_clean=int(m.sum()),
                med_peak=float(raw[m].median()) if m.any() else float("nan"),
                med_jpeak=float(jpk[m].median()) if (jpk is not None and m.any()) else float("nan"))


def main():
    rows = []
    for folder in sorted(glob.glob(str(RESULTS / "26*"))):
        pf = glob.glob(folder + "/**/abc_fit_per_pixel.csv", recursive=True)
        if not pf:
            continue
        st = clean_stats(pf[0])
        if st is None:
            continue
        name = Path(folder).name
        stamp = name[:11]
        rows.append((stamp, st["n_clean"], st["n"],
                     100 * st["n_clean"] / st["n"] if st["n"] else 0,
                     st["med_peak"], st["med_jpeak"]))

    # Reference peak-J is fixed at the device's established value (§11) — using the runs'
    # own median would be contaminated by the droop-artifact runs pulling it up.
    ref = PEAK_J_REF
    lines = ["# Clean-fit tracker", "",
             "Clean per-pixel ABC fits per run (peak captured in-data, R²≥0.70, 0.05≤β≤20).",
             "Goal: **90%+ clean**. **Read clean % together with J_peak** — a high clean % with a",
             "J_peak far above the device norm (≈5) means the fit ran into the droop (clean but wrong).",
             "See abc_fit_GUIDE.md for β and the flags.", "",
             "| run | clean / total | clean % | median peak IQE | median J_peak | note |",
             "|---|---|---|---|---|---|"]
    print("\n".join(lines[6:]))
    for stamp, nc, n, pct, med, jp in rows:
        bar = "█" * int(round(pct / 10)) + "░" * (10 - int(round(pct / 10)))
        meds = f"{100*med:.0f}%" if med == med else "—"
        jps = f"{jp:.1f}" if jp == jp else "—"
        warn = "⚠ droop artifact?" if (jp == jp and jp > PEAK_J_WARN * ref and nc >= 3) else ""
        lines.append(f"| {stamp} | {nc}/{n} | {bar} {pct:.0f}% | {meds} | {jps} | {warn} |")
        print(f"{stamp}  {nc:>3}/{n:<3}  {bar} {pct:3.0f}%   peak {meds}   J_peak {jps}  {warn}")
    lines += ["", f"_β band: {BETA_MIN}–{BETA_MAX}; R²≥{R2MIN}; J_peak-ref≈{ref:.1f}. "
              f"Regenerate with `python cleanfit_tracker.py`._"]
    OUT.mkdir(exist_ok=True)
    (OUT / "cleanfit_tracker.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"\nwrote {OUT/'cleanfit_tracker.md'}")


if __name__ == "__main__":
    main()
