"""Per-pixel sweep-parameter benchmark — drift-robust edition.

The naive version (fit a single pixel's peak IQE per sweep) failed because (1) the device
EQE drifts ~10% over tens of minutes of back-to-back sweeps (cumulative self-heating →
thermal droop of the near-peak EQE), and (2) single-pixel peak IQE is hypersensitive to
that, so it can't resolve parameter effects. This version fixes both:

  (a) PRIMARY METRIC = per-point PRECISION at low J (the current & light relative SEM in the
      rising-side band). That is exactly what the averaging knobs (NPLC, N_PM, N_SMU) control,
      and it is DRIFT-IMMUNE — it is the within-point scatter, not a slow trend. Cut a knob and
      read directly whether the low-J noise stays under the analysis gates.

  (b) INTERLEAVED BASELINES — a baseline sweep is run before every variation, so the slow drift
      is measured and the peak-IQE cross-check is DETRENDED (variation minus the time-interpolated
      baseline). The spread among baselines also gives the true reproducibility band.

    python benchmark_sweep.py --pixel 54 33 --repeat 1
    python benchmark_sweep.py --simulate

Writes benchmark_report/ (table.md + benchmark.png).
"""
from __future__ import annotations

import argparse
import dataclasses
import datetime
import time
from pathlib import Path

import numpy as np
import pandas as pd

import measure_iqe as m
from measure_iqe import IQEConfig
import analyze_array as aa

OUT = Path("benchmark_report")
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


def variant_config(base: IQEConfig, overrides: dict) -> IQEConfig:
    """Baseline config with one variation applied — immutable, no module globals touched."""
    return dataclasses.replace(base, **{**BASELINE, **overrides}).validate()


def sweep_once(pm, smu, smile, pixel, tmp, cfg):
    """One real sweep; returns metrics incl. the DRIFT-IMMUNE low-J relSEM."""
    t0 = time.perf_counter()
    m.run_sweep_voltage(pm, smu, smile, [dict(X=pixel[0], Y=pixel[1])], tmp,
                        cfg, lambda *_: None)
    dt = time.perf_counter() - t0
    tc = time.time()
    f = list((tmp / "NVLEDSweep" / "Sweep").glob("aggregated_bitval=*_mean_std.csv"))
    out = dict(t_wall=dt, t_center=tc, peak=np.nan, r2=np.nan,
               cur_rsem=np.nan, light_rsem=np.nan)
    if not f:
        return out
    df = pd.read_csv(f[0])
    I = df.NVLED_CURR_MEAN.astype(float).abs()
    P = df.PM400_POWER_MEAN.astype(float)
    J = I / 1e-6
    s = (J >= LOWJ[0]) & (J < LOWJ[1]) & (I > 2e-8)
    if s.sum():
        out["cur_rsem"] = float(np.median(df.NVLED_CURR_STD.astype(float)[s] / I[s]))
        out["light_rsem"] = float(np.median(df.PM400_POWER_STD.astype(float)[s] / P[s].clip(lower=1e-30)))
    e = aa.enrich(df, dark_W=0.0)
    rel = e[e.RELIABLE]
    fit = aa.fit_abc(rel.J_LED.values, rel.EQE_LED.values) if len(rel) >= 6 else None
    if fit:
        out["r2"] = float(fit["R_squared"])
        if fit.get("peak_captured"):
            out["peak"] = 100 * float(fit["peak_IQE"])
    return out


def main():
    ap = argparse.ArgumentParser(description="Drift-robust sweep-parameter benchmark.")
    ap.add_argument("--simulate", action="store_true")
    ap.add_argument("--out", default=None)
    ap.add_argument("--pixel", nargs=2, type=int, default=None)
    ap.add_argument("--repeat", type=int, default=1, help="repeats per variation (baselines interleaved regardless)")
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

    baselines, variations = [], []
    try:
        if args.pixel:
            pixel = tuple(args.pixel)
        else:
            x1, y1, x2, y2 = base_cfg.resolve_roi()
            sel = m.run_fast_scan(pm, smu, smile, x1, y1, x2, y2, root, base_cfg, log)
            pixel = (sel[len(sel) // 2]["X"], sel[len(sel) // 2]["Y"]) if sel else (x1, y1)
        log(f"pixel {pixel}: {len(VARIATIONS)} variations, baseline interleaved before each "
            f"(+ one final) → {len(VARIATIONS)+1} baselines")

        def run(name, ov, store):
            cfg = variant_config(base_cfg, ov)
            for r in range(args.repeat if store is variations else 1):
                mm = sweep_once(pm, smu, smile, pixel,
                                root / f"{name}_{len(store)}".replace(" ", "_"), cfg)
                mm["name"] = name
                store.append(mm)
                log(f"  {name:16s} cur_rSEM {100*mm['cur_rsem']:.0f}% light_rSEM "
                    f"{100*mm['light_rsem']:.0f}% peak {mm['peak']:.0f}% {mm['t_wall']:.0f}s")

        run("baseline", {}, baselines)                 # bracket start
        for name, ov in VARIATIONS:
            run(name, ov, variations)
            run("baseline", {}, baselines)             # baseline before/after each variation
    finally:
        for ch in ("a", "b"):
            try: smu.enable_output(ch, False)
            except Exception: pass
        for d in (pm, smu, smile):
            try: d.close()
            except Exception: pass

    report(baselines, variations)


def report(baselines, variations):
    OUT.mkdir(exist_ok=True)
    bt = np.array([b["t_center"] for b in baselines], float)
    bp = np.array([b["peak"] for b in baselines], float)
    t0 = bt.min() if len(bt) else 0.0
    bt -= t0
    ok = np.isfinite(bp)
    # drift model: linear baseline-peak vs time; residual std = reproducibility band
    if ok.sum() >= 2:
        sl, ic = np.polyfit(bt[ok], bp[ok], 1)
        band = max(float(np.std(bp[ok] - (sl * bt[ok] + ic))), 1.0)
    else:
        sl, ic, band = 0.0, float(np.nanmedian(bp)) if ok.any() else np.nan, 2.0
    base_cur = float(np.nanmedian([b["cur_rsem"] for b in baselines]))
    base_light = float(np.nanmedian([b["light_rsem"] for b in baselines]))
    base_t = float(np.nanmedian([b["t_wall"] for b in baselines]))

    lines = ["# Sweep-parameter benchmark (drift-robust)", "",
             f"**Measured device drift:** baseline peak IQE trends **{sl*60:+.1f}%/min** over the run "
             f"(first {bp[ok][0]:.0f}% → last {bp[ok][-1]:.0f}%); reproducibility band ±{band:.0f}%. "
             f"This is why raw single-sweep peak IQE is unreliable — it is DETRENDED below.", "",
             f"Baseline low-J precision: current relSEM **{100*base_cur:.0f}%**, light relSEM "
             f"**{100*base_light:.0f}%**, {base_t:.0f} s/pixel.", "",
             "Verdict = SAFE if the cut keeps low-J current relSEM <50% (the gate) with margin, "
             "light relSEM not much worse, and the DRIFT-CORRECTED peak within the ±band.", "",
             "| variation | cur relSEM | light relSEM | Δpeak (detrended) | s/pixel | speedup | verdict |",
             "|---|---|---|---|---|---|---|"]
    print("\n".join(lines[2:3] + lines[6:]))
    for v in variations:
        pred = sl * (v["t_center"] - t0) + ic          # baseline peak expected at this time
        dpk = v["peak"] - pred if np.isfinite(v["peak"]) else np.nan
        speed = base_t / v["t_wall"] if v["t_wall"] else 1
        cr, lr = v["cur_rsem"], v["light_rsem"]
        safe = (np.isfinite(cr) and cr < 0.5 and cr <= max(0.30, 1.6 * base_cur)
                and (not np.isfinite(lr) or lr <= max(0.08, 1.6 * base_light))
                and (not np.isfinite(dpk) or abs(dpk) <= 2 * band))
        v_str = f"| {v['name']} | {100*cr:.0f}% | {100*lr:.0f}% | {dpk:+.0f}% | {v['t_wall']:.0f} | {speed:.2f}× | {'✅ safe' if safe else '⚠ check'} |"
        lines.append(v_str)
        print(f"  {v['name']:16s} cur {100*cr:.0f}%  light {100*lr:.0f}%  Δpk{dpk:+.0f}  "
              f"{v['t_wall']:.0f}s  {speed:.2f}×  {'safe' if safe else 'check'}")
    (OUT / "table.md").write_text("\n".join(lines), encoding="utf-8")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 5))
        # left: primary metric — low-J relSEM per variation vs baseline
        names = [v["name"] for v in variations]
        x = np.arange(len(names))
        a1.bar(x - 0.2, [100 * v["cur_rsem"] for v in variations], 0.4, label="current relSEM", color="tab:purple")
        a1.bar(x + 0.2, [100 * v["light_rsem"] for v in variations], 0.4, label="light relSEM", color="tab:green")
        a1.axhline(100 * base_cur, color="tab:purple", ls="--", lw=1, label="baseline cur")
        a1.axhline(100 * base_light, color="tab:green", ls="--", lw=1, label="baseline light")
        a1.axhline(50, color="red", ls=":", lw=1, label="current gate 50%")
        a1.set_xticks(x); a1.set_xticklabels(names, rotation=60, ha="right", fontsize=7)
        a1.set_ylabel("low-J relSEM (%)"); a1.set_title("Per-point precision (drift-immune) — the averaging metric")
        a1.legend(fontsize=7); a1.grid(True, axis="y", alpha=0.3)
        # right: baseline drift + detrended variation peaks
        a2.scatter((bt), bp, color="tab:red", label="baselines (drift)", zorder=3)
        tt = np.linspace(bt.min(), bt.max(), 50)
        a2.plot(tt, sl * tt + ic, "r--", lw=1, label=f"drift {sl*60:+.1f}%/min")
        for v in variations:
            a2.scatter([v["t_center"] - t0], [v["peak"]], color="tab:blue", s=20)
            a2.annotate(v["name"], (v["t_center"] - t0, v["peak"]), fontsize=6,
                        textcoords="offset points", xytext=(4, 3))
        a2.set_xlabel("time (s)"); a2.set_ylabel("peak IQE (%)")
        a2.set_title("Device drift over the run (baselines) + variations\n(judge variations vs the red drift line)")
        a2.legend(fontsize=8); a2.grid(True, alpha=0.3)
        fig.tight_layout(); fig.savefig(OUT / "benchmark.png", dpi=140)
        print(f"\nwrote {OUT/'table.md'} and {OUT/'benchmark.png'}")
    except Exception as e:
        print(f"(plot skipped: {e})")


if __name__ == "__main__":
    main()
