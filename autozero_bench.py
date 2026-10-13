"""autozero_bench.py — A/B test the SMU autozero / NPLC handling on ONE known pixel.

The per-segment sweep switches NPLC (2 ↔ 0.5) at the knee (V≈-1.3). If the SMU's zero reference is
stale across that switch, the CURRENT reads with a small NPLC-dependent offset → a ~1-3% step in
EQE=P/I at the knee, on every pixel. This bench sweeps the SAME pixel under each autozero/NPLC policy,
overlays the EQE(J) curves (full + a zoom on the knee), and times each, so you can see which policy
removes the step and what it costs.

Conditions:
  single_nplc     — no per-segment split, one NPLC for the whole sweep (no discontinuity by design)
  az_auto         — per-segment, AUTOZERO_AUTO (re-zeros as needed at each NPLC)
  az_once         — per-segment, AUTOZERO_ONCE (one zero at the start, then off)
  az_off          — per-segment, AUTOZERO_OFF (no autozero)
  az_per_pixel    — per-segment, AUTOZERO_OFF but an explicit AUTOZERO_ONCE issued at each pixel start

Usage:  python autozero_bench.py --pixel 128 128 [--repeats 2] [--nplc 2.0] [--simulate]
Writes autozero_bench/<stamp>/  with autozero_bench.png + table.md.
"""
from __future__ import annotations
import argparse
import time
from dataclasses import replace
from pathlib import Path

import numpy as np

import measure_iqe as m
from measure_iqe import IQEConfig


def _sweep_pixel(pm, smu, smile, x, y, cfg, per_pixel_az=False, log=print):
    """One voltage sweep of a single pixel with the given IQEConfig.
    Returns (J_arr, EQE_arr, wall_s, points, samples) where `points` is a list of per-point dicts
    (full mean/median/std/SEM for I and P) and `samples` is a list of (V, NPLC, raw-burst-dict)."""
    area = cfg.pixel_area_um2 * 1e-8
    smile.set_pixel(x, y, cfg.bitvals[0])
    if cfg.warmup_ms:
        smu.set_voltage("b", float(cfg.warmup_nvled_v)); time.sleep(cfg.warmup_ms / 1000.0)
    if per_pixel_az and hasattr(smu, "inst"):
        smu.inst.write("smub.measure.autozero = smub.AUTOZERO_ONCE")   # fresh zero for THIS pixel
    fine = m.build_fine_voltages(pm, smu, cfg, lambda *a, **k: None)
    if fine and cfg.thermal_settle_ms:
        smu.set_voltage("b", float(fine[0][0])); time.sleep(cfg.thermal_settle_ms / 1000.0)
    m.set_measure_range_b(smu, cfg.nvled_range_i)
    pts, samples = [], []
    cur_nplc = None
    t0 = time.perf_counter()
    for v, _it in fine:
        if cfg.vramp_perseg_effort and v > cfg.vramp_knee_v:
            nplc, n_pm, n_smu, settle = (cfg.vramp_lowv_nplc, cfg.vramp_lowv_n_pm,
                                         cfg.vramp_lowv_n_smu, cfg.vramp_lowv_settle_ms)
        elif cfg.vramp_perseg_effort:
            nplc, n_pm, n_smu, settle = (cfg.vramp_highv_nplc, cfg.vramp_highv_n_pm,
                                         cfg.vramp_highv_n_smu, cfg.vramp_highv_settle_ms)
        else:
            nplc, n_pm, n_smu, settle = (cfg.sweep_nplc, cfg.vramp_lowv_n_pm,
                                         cfg.vramp_lowv_n_smu, cfg.vramp_lowv_settle_ms)
        if nplc != cur_nplc:
            m.set_nplc(smu, nplc); cur_nplc = nplc
        smu.set_voltage("b", float(v)); time.sleep(settle / 1000.0)
        (p, ps, iv, ivs, inv, invs, _, _), raw = m.acquire_point(pm, smu, n_smu, n_pm, return_raw=True)
        sn = m.burst_stats(raw["nvled"]); sp = m.burst_stats(raw["pm"])
        I = abs(sn["median"]); P = max(sp["mean"], 0.0)
        eqe = (P / cfg.photon_energy_j) / (I / 1.602176634e-19) if (I > 1e-6 and P > 3e-11) else float("nan")
        pts.append(dict(V=round(float(v), 4), NPLC=nplc,
                        I_mean=sn["mean"], I_median=sn["median"], I_std=sn["std"], I_sem=sn["sem"], I_n=sn["n"],
                        P_mean=sp["mean"], P_median=sp["median"], P_std=sp["std"], P_sem=sp["sem"], P_n=sp["n"],
                        J=I / area, EQE=eqe))
        samples.append((round(float(v), 4), nplc, raw))
    smile.blank_frame()
    Js = np.array([pt["J"] for pt in pts if np.isfinite(pt["EQE"])])
    Es = np.array([pt["EQE"] for pt in pts if np.isfinite(pt["EQE"])])
    o = np.argsort(Js)                       # ASCENDING J (the sweep runs high→low) — np.interp needs it
    return Js[o], Es[o], time.perf_counter() - t0, pts, samples


CONDITIONS = [
    dict(name="single_nplc", perseg=False, az="AUTO",  per_pixel=False),
    dict(name="az_auto",     perseg=True,  az="AUTO",  per_pixel=False),
    dict(name="az_once",     perseg=True,  az="ONCE",  per_pixel=False),
    dict(name="az_off",      perseg=True,  az="OFF",   per_pixel=False),
    dict(name="az_per_pixel", perseg=True, az="OFF",   per_pixel=True),
]


def _knee_step(J, E, knee_J):
    """max local point-to-point EQE step within a factor ~2 of the knee current density."""
    o = np.argsort(J); J, E = J[o], E[o]; sel = (J >= 0.6 * knee_J) & (J <= 1.8 * knee_J)
    if sel.sum() < 3:
        return np.nan
    e = E[sel]; return float(np.nanmax(np.abs(np.diff(e)) / e[:-1]) * 100)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pixels", nargs="+", default=None, metavar="X,Y",
                    help='pixels to test, e.g. --pixels 177,63 155,463 80,15 (jumpers + clean; see find_knee_jump.py)')
    ap.add_argument("--pixel", type=int, nargs=2, default=[128, 128], metavar=("X", "Y"),
                    help="single pixel (used only if --pixels is not given)")
    ap.add_argument("--repeats", type=int, default=2, help="sweeps per condition (for timing + scatter)")
    ap.add_argument("--settle", type=float, default=30.0,
                    help="dead time (s) before each strategy AND each repeat, so temperature settles")
    ap.add_argument("--nplc", type=float, default=2.0, help="the single-NPLC value")
    ap.add_argument("--simulate", action="store_true")
    args = ap.parse_args()
    pixels = [tuple(int(v) for v in p.split(",")) for p in args.pixels] if args.pixels else [tuple(args.pixel)]

    log = m.make_logger()      # encode-safe timestamped logger

    base_cfg = IQEConfig(sweep_nplc=args.nplc)
    pm, smu, smile = m.connect(base_cfg, log, simulate=args.simulate)
    m.configure_common(pm, smu, base_cfg, log)
    smu.set_voltage("a", base_cfg.vled_voltage); smu.enable_output("a", True); smu.enable_output("b", True)
    time.sleep(0.05)
    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    except Exception:
        plt = None

    # results[cond][pixel] = list of (J, E, t) over repeats
    results = {c["name"]: {p: [] for p in pixels} for c in CONDITIONS}
    times = {c["name"]: [] for c in CONDITIONS}
    data_rows, samp_rows = [], []                # every point / every burst sample → CSV
    log(f"testing {len(pixels)} pixel(s) × {len(CONDITIONS)} strategies × {args.repeats} repeats, "
        f"{args.settle:.0f}s dead time before each — est. "
        f"{len(pixels)*len(CONDITIONS)*args.repeats*(args.settle+3)/60:.0f} min")
    for c in CONDITIONS:
        # one immutable config per condition — no module-global mutation
        cfg = replace(base_cfg, vramp_perseg_effort=c["perseg"], sweep_autozero=c["az"])
        m.configure_voltage_source_b(smu, cfg)            # applies autozero + perseg config
        for rep in range(args.repeats):
            # dead time before each strategy/repeat — let the die temperature relax so a slow
            # thermal drift can't be mistaken for a strategy difference.
            if args.settle > 0:
                smile.blank_frame()
                log(f"  {c['name']} rep{rep+1}: settling {args.settle:.0f}s (blanked)…")
                time.sleep(args.settle if not args.simulate else 0.05)
            for (x, y) in pixels:
                J, E, t, pts, samples = _sweep_pixel(pm, smu, smile, x, y, cfg,
                                                     per_pixel_az=c["per_pixel"], log=log)
                results[c["name"]][(x, y)].append((J, E)); times[c["name"]].append(t)
                for pt in pts:
                    data_rows.append(dict(strategy=c["name"], X=x, Y=y, repeat=rep, **pt))
                for vv, nplc, raw in samples:
                    for typ, arr in [("PM400", raw["pm"]), ("VLED", raw["vled"]), ("NVLED", raw["nvled"])]:
                        for k, val in enumerate(arr):
                            samp_rows.append((c["name"], x, y, rep, vv, nplc, typ, k, val))
        log(f"  {c['name']:12s}: {np.median(times[c['name']]):5.1f} s/sweep")

    # peak / knee current density (from the first clean-ish curve) for the step metric + zoom
    allJ = [cv[0] for cn in results for p in results[cn] for cv in results[cn][p] if len(cv[0])]
    knee_J = float(np.median([J[np.argmin(np.abs(J - 20))] for J in allJ])) if allJ else 20.0

    print("\nstrategy      | s/sweep | median knee EQE step % (per pixel)")
    lines = ["| strategy | s/sweep | " + " | ".join(f"({x},{y})" for x, y in pixels) + " |",
             "|" + "---|" * (len(pixels) + 2)]
    for c in CONDITIONS:
        steps = []
        for p in pixels:
            ss = [_knee_step(J, E, knee_J) for J, E in results[c["name"]][p]]
            steps.append(float(np.nanmedian(ss)) if ss else np.nan)
        print(f"  {c['name']:12s} | {np.median(times[c['name']]):6.1f}  | " +
              "  ".join(f"{s:4.1f}" for s in steps))
        lines.append(f"| {c['name']} | {np.median(times[c['name']]):.1f} | " +
                     " | ".join(f"{s:.1f}" for s in steps) + " |")

    out = Path("autozero_bench") / time.strftime("%y%m%d-%H%M")
    out.mkdir(parents=True, exist_ok=True)
    (out / "table.md").write_text("\n".join(lines), encoding="utf-8")
    # the actual measurement data: every point (mean/median/std/SEM of I and P) + every burst sample
    import pandas as pd
    pd.DataFrame(data_rows).to_csv(out / "autozero_bench_points.csv", index=False)
    if samp_rows:
        pd.DataFrame(samp_rows, columns=["strategy", "X", "Y", "repeat", "V", "NPLC",
                                         "TYPE", "SAMPLE_IDX", "VALUE"]).to_csv(
            out / "autozero_bench_samples.csv", index=False)
    log(f"wrote {len(data_rows)} points + {len(samp_rows)} samples to CSV")
    if plt is not None:
        _plot_bench(pixels, lambda name, pix: results[name][pix], knee_J, out / "autozero_bench.png", plt)
    print(f"\nwrote {out}/autozero_bench.png + table.md")


def _plot_bench(pixels, curves_of, knee_J, out_png, plt):
    """One row per pixel: full EQE(J) + a knee zoom, overlaying strategies. `curves_of(name, pixel)`
    returns a list of (J_arr, E_arr) over repeats. Robust to unsorted J and empty curves."""
    import warnings as _w
    fig, ax = plt.subplots(len(pixels), 2, figsize=(13, 4.2 * len(pixels)), squeeze=False)
    for r, p in enumerate(pixels):
        for name in [c["name"] for c in CONDITIONS]:
            cv = [(np.asarray(J, float), np.asarray(E, float)) for J, E in curves_of(name, p) if len(J) > 1]
            if not cv:
                continue
            lo = min(J.min() for J, _ in cv); hi = max(J.max() for J, _ in cv)
            Jref = np.logspace(np.log10(max(1e-3, lo)), np.log10(hi), 60)
            interps = []
            for J, E in cv:
                o = np.argsort(J)                         # np.interp needs increasing x
                interps.append(np.interp(Jref, J[o], E[o], left=np.nan, right=np.nan))
            with _w.catch_warnings():
                _w.simplefilter("ignore")
                Em = np.nanmedian(interps, axis=0)
            ax[r][0].plot(Jref, Em, "-", lw=1.2, label=name)
            ax[r][1].plot(Jref, Em, "-o", ms=3, lw=1.0, label=name)
        for a in ax[r]:
            a.set_xscale("log"); a.set_ylabel(f"({p[0]},{p[1]})\nEQE=P/I"); a.grid(alpha=0.3, which="both")
        ax[r][0].legend(fontsize=7)
        ax[r][1].set_xlim(0.6 * knee_J, 2.2 * knee_J); ax[r][1].axvline(knee_J, color="0.6", ls=":", lw=1)
    ax[0][0].set_title("full EQE(J)"); ax[0][1].set_title(f"zoom on the knee (J≈{knee_J:.0f}, NPLC switch)")
    for a in ax[-1]:
        a.set_xlabel("J (A/cm²)")
    fig.suptitle("Autozero / NPLC A-B — does the knee step vanish, at what time cost? (jumpers + clean pixels)")
    fig.tight_layout(); fig.savefig(out_png, dpi=150); plt.close(fig)


def replot(folder):
    """Regenerate autozero_bench.png from a run's saved autozero_bench_points.csv (no hardware)."""
    import pandas as pd, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    folder = Path(folder)
    p = pd.read_csv(folder / "autozero_bench_points.csv")
    p = p[p["EQE"].notna()]
    pixels = sorted(set(zip(p["X"], p["Y"])))
    knee_J = float(p[p["V"].round(2) == -1.3]["J"].median()) if (p["V"].round(2) == -1.3).any() else 20.0

    def curves_of(name, pix):
        g = p[(p["strategy"] == name) & (p["X"] == pix[0]) & (p["Y"] == pix[1])]
        return [(gg["J"].values, gg["EQE"].values) for _, gg in g.groupby("repeat")]
    _plot_bench(pixels, curves_of, knee_J, folder / "autozero_bench.png", plt)
    print(f"re-plotted {folder/'autozero_bench.png'}")


if __name__ == "__main__":
    import sys
    if len(sys.argv) >= 3 and sys.argv[1] == "--replot":
        replot(sys.argv[2])
    else:
        main()
