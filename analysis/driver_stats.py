"""Pixel-driver statistics from transistoriv runs, and what they mean under voltage drive.

    python analysis/driver_stats.py                      # the Array38 and Array42 shorts
    python analysis/driver_stats.py <tiv run> [<tiv run> ...] [-o <folder under results>]

Each transistoriv run is one population of drivers: the shorts of an array, or (the
check of 2026-11-04) ordinary pixels measured the same way. The sweep that goes with a
run is the run its --pixel-list came from.

Writes into results/<out> (default comparisons/drivers):
  driver_stats.csv   per run: quadrant medians and spreads, spread per bit, code
                     linearity, additivity, output slope, saturation headroom, gradient
  spread_vs_bias.csv per run and bias: the drivers' current spread against the typical
                     LED pixels' in the matching sweep
  drivers.png        a: bitval-15 current per quadrant at -3.2 V; b: spread against bit
                     weight, with the 1/sqrt(weight) of independent unit cells;
                     c: code linearity; d: current spread against bias, LEDs and drivers
  best_shorts.png    the shorts only: a: mean of the N best (by bitval-15 current at
                     -3.2 V) against N; b: the I(V) of the 10 best at bitval 15;
                     c: every short's current, sorted, against its rank in percent
  best_shorts.csv    the 10 best shorts of each run
  emitting_shorts.csv shorts excluded because they gave light (see LIGHT_SIGMA)
  shorts_isolation.png the shorts only: a: current against the distance to the nearest
                     other short; b: the current of the shorts with no other short
                     within r, against r. Distance = max(|dX|, |dY|), so all 8
                     neighbours are 1, and every short of the list counts, measured,
                     lit or not working: another short draws current without light
  shorts_isolation.csv per short: the nearest other short and the current
  summary.md         the numbers

Spreads are robust CVs: 1.4826 x MAD / median. A bit's current is I(bit) - I(bitval 0).
In an LED pixel the LED is in series: at -3.2 V it passes ~90 % of a short's current at
bitval 15, so there the high codes are partly LED-limited; bits 1-4 (<= 40 uA) are not.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import analyze_array as aa
import pixel_classes as pc

R = Path(__file__).resolve().parent.parent / "results"
DEFAULT = ["261029-0836_Array38_PIXELS_Transistoriv", "261103-1623_Array42_PIXELS_Transistoriv"]
BITS = (1, 2, 4, 8, 15)
QUADS = ("TL", "TR", "BL", "BR")
V_REF = -3.2               # the bias of the per-bit and per-quadrant numbers
I_MIN = 50e-6              # bitval 15 below this at V_REF: not a working driver
# A short has no LED in the path, so it gives no light. One that does, at any point, is
# not a clean driver measurement: its own LED conducts part of the current, or another
# pixel was lit with it (a latch), whose current the reading then includes. "Light" is
# above the run's own floor: the median of all its readings + LIGHT_SIGMA robust sigma,
# the rule transistoriv itself applies since 2026-11-05 (TIV_LIGHT_SIGMA). The floor
# moves between sessions: +0.04 +- 0.37 nW in 261029-0836, -1.18 +- 0.26 nW in
# 261103-1623, so a fixed watt value would be too strict on one and too lax on the other.
# A run with LIGHT_OK columns (the check ran) also drops every short with a flagged point.
LIGHT_SIGMA = 10.0
N_BEST = 10


def rcv(s):
    s = pd.Series(s).dropna()
    return 100 * 1.4826 * (s - s.median()).abs().median() / s.median() if len(s) else np.nan


def load(run: Path):
    t = pd.read_csv(run / "transistoriv.csv")
    t["I"] = t["NVLED_CURR_MEAN"].abs()
    t["V"] = t["NVLED_V"].round(2)
    p = t.pivot_table(index=["X", "Y", "QUADRANT"], columns=["BITVAL", "V"], values="I")
    ok = p[(15, V_REF)] > I_MIN
    rc = json.loads((run / "run_config.json").read_text(encoding="utf-8"))
    args = rc.get("args", {})
    vled = float(rc.get("config", {}).get("vled_voltage", 1.8))
    src = Path(str(args.get("pixel_list") or "").replace("\\", "/"))
    sweep = next((d for d in R.iterdir() if d.name in str(src)), None)
    emit = pd.DataFrame()
    if src.name == "short.csv":          # only a short must be dark; an LED pixel is lit
        pm = t.pivot_table(index=["X", "Y", "QUADRANT"], columns=["BITVAL", "V"],
                           values="PM400_POWER_MEAN")
        mx = pm.max(axis=1)
        at = pm.idxmax(axis=1)
        P = t["PM400_POWER_MEAN"]
        off = P.median()
        thr = off + LIGHT_SIGMA * 1.4826 * (P - off).abs().median()
        lit = mx > thr
        if "LIGHT_OK" in t:
            lit |= t.groupby(["X", "Y", "QUADRANT"])["LIGHT_OK"].min() == 0
        lit = lit.reindex(p.index, fill_value=False)
        emit = pd.DataFrame({"max_light_W": mx[lit[lit].index],
                             "at_bitval": [a[0] for a in at[lit[lit].index]],
                             "at_V": [a[1] for a in at[lit[lit].index]],
                             "I15_ref_uA": p.loc[lit, (15, V_REF)] * 1e6,
                             "floor_W": thr})
        ok &= ~lit
    return p[ok], int((p[(15, V_REF)] <= I_MIN).sum()), emit, src, sweep, vled


def stats(name, p, n_bad, n_light, src):
    row = dict(run=name, pixel_list=src.name, n=len(p), n_not_working=n_bad,
               n_emitting_excluded=n_light,
               **{f"n_{q}": int((p.index.get_level_values("QUADRANT") == q).sum())
                  for q in QUADS})
    volts = sorted({v for b, v in p.columns if b == 15})
    for v in (-0.5, -1.4, V_REF):
        if (15, v) not in p:
            continue
        s = p[(15, v)]
        q = s.groupby(level="QUADRANT")
        tag = f"{v:+.1f}V"
        row[f"I15_{tag}_uA"] = s.median() * 1e6
        row[f"cv_{tag}_pct"] = rcv(s)
        row[f"between_quadrant_{tag}_pct"] = 100 * q.median().std() / s.median()
        row[f"within_quadrant_{tag}_pct"] = float(np.median([rcv(g) for _, g in q]))
        for k, m in q.median().items():
            row[f"I15_{tag}_{k}_uA"] = m * 1e6
    net = pd.DataFrame({b: p[(b, V_REF)] - p[(0, V_REF)] for b in BITS})
    for b in BITS:
        row[f"bit{b}_uA"] = net[b].median() * 1e6
        row[f"bit{b}_cv_pct"] = rcv(net[b])
        row[f"code{b}_over_code1"] = net[b].median() / net[1].median()
    row["additivity_15_over_sum"] = float((net[15] / net[[1, 2, 4, 8]].sum(axis=1)).median())
    g = net[[1, 2, 4, 8]].div(net[[1, 2, 4, 8]].median())
    ev = np.linalg.eigvalsh(np.cov(g.T.values))
    row["common_gain_share_pct"] = 100 * ev[-1] / ev.sum()
    row["corr_bit1_bit8"] = float(net[1].corr(net[8]))
    sat = [v for v in volts if -3.2 <= v <= -1.5]
    if len(sat) >= 5:
        sl = np.polyfit(np.array(sat), p[15][sat].values.T, 1)[0]
        row["slope_pct_per_V"] = float(np.median(np.abs(sl) / p[(15, -1.4)].values) * 100)
        row["r_out_MOhm"] = float(np.median(1 / np.abs(sl)) / 1e6)
        ratio = p[15].div(p[(15, V_REF)], axis=0)
        cols = sorted(ratio.columns, reverse=True)
        v95 = ratio[cols].apply(lambda r: next((c for c in cols if r[c] >= 0.95), np.nan),
                                axis=1)
        row["nvled_95pct_V"] = float(v95.median())
    s = p[(15, V_REF)].reset_index()
    A = np.c_[np.ones(len(s)), s["X"], s["Y"]]
    c = np.linalg.lstsq(A, s[(15, V_REF)], rcond=None)[0]
    row["gradient_X_pct"] = 100 * c[1] * 511 / s[(15, V_REF)].median()
    row["gradient_Y_pct"] = 100 * c[2] * 511 / s[(15, V_REF)].median()
    return row, net


def led_spread(sweep: Path):
    """Robust CV of I_LED per bias over the sweep's typical pixels."""
    cf = sweep / "REPORT_assets/pixelclassification/sweep/sweep_classes.csv"
    if not cf.is_file():
        return None
    sw = pc.load_sweep(sweep)
    cls = pd.read_csv(cf).set_index(["X", "Y"]).filter(like="fastscan_").iloc[:, 0]
    sw = sw.join(cls.rename("cls"), on=["X", "Y"])
    sw = sw[(sw["cls"] == "typical") & sw["I_ref"].notna()]
    g = sw.groupby("V")["I_LED"]
    t = pd.DataFrame({"led_I_uA": g.median() * 1e6, "led_cv_pct": g.agg(rcv)})
    return t[t["led_I_uA"] >= 0.1]         # below ~100 nA the spread is the meter's noise


def isolation(pops: dict, out: Path, plt):
    """shorts_isolation.png/.csv: does a short's current depend on how close the next
    short is? A neighbouring short draws current and gives no light, so the light check
    cannot see it; if it added to the reading, clustered shorts would read high."""
    pops = {n: v for n, v in pops.items() if v[1].is_file()}
    if not pops:
        return
    fig, ax = plt.subplots(1, 2, figsize=(15, 5.6))
    cm = plt.get_cmap("tab10")
    rows, radii = [], np.arange(0, 31)
    for i, (name, (p, lst)) in enumerate(pops.items()):
        c = cm(i)
        allp = pd.read_csv(lst)[["X", "Y"]].drop_duplicates().to_numpy()
        xy = np.array([(k[0], k[1]) for k in p.index])
        d = np.abs(allp[None, :, :] - xy[:, None, :]).max(axis=2)
        d[d == 0] = 10 ** 6                          # the short itself
        nn = d.min(axis=1)
        I = p[(15, V_REF)].to_numpy() * 1e6
        lab = f"{name.split('_')[1]} {name.split('_')[0]} ({len(p)} measured, {len(allp)} in the list)"
        jit = np.random.default_rng(i).uniform(0.9, 1.1, len(nn))
        ax[0].plot(nn * jit, I, ".", ms=4, color=c, alpha=0.7, label=lab)
        med = [np.median(I[nn > r]) if (nn > r).sum() >= 3 else np.nan for r in radii]
        q1 = [np.percentile(I[nn > r], 25) if (nn > r).sum() >= 3 else np.nan for r in radii]
        q3 = [np.percentile(I[nn > r], 75) if (nn > r).sum() >= 3 else np.nan for r in radii]
        cnt = [(nn > r).sum() for r in radii]
        ax[1].plot(radii, med, "o-", ms=3, color=c, label=lab)
        ax[1].fill_between(radii, q1, q3, color=c, alpha=0.15, lw=0)
        for r_ in (0, 1, 5, 10, 20):
            if cnt[r_] >= 3:
                ax[1].annotate(f"n={cnt[r_]}", (r_, med[r_]), fontsize=7, color=c,
                               textcoords="offset points", xytext=(2, 6))
        rows += [dict(run=name, X=k[0], Y=k[1], quadrant=k[2], nearest_short_px=int(n_),
                      I15_uA=v) for k, n_, v in zip(p.index, nn, I)]
    pd.DataFrame(rows).to_csv(out / "shorts_isolation.csv", index=False, float_format="%.3f")
    ax[0].set_xscale("log")
    ax[0].set_xlabel("distance to the nearest other short, max(|dX|, |dY|) (px; 1 = touching)")
    ax[0].set_ylabel(f"I at bitval 15, {V_REF:+.1f} V (uA)")
    ax[0].set_title("Each short against its nearest neighbouring short", fontsize=10)
    ax[1].set_xlabel("r: only shorts with no other short within r (px)")
    ax[1].set_ylabel(f"median I at bitval 15, {V_REF:+.1f} V (uA), 25-75 % shaded")
    ax[1].set_title("Current of ever more isolated shorts", fontsize=10)
    for a in ax:
        a.grid(alpha=0.3, which="both")
        a.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out / "shorts_isolation.png", dpi=150)
    plt.close(fig)


def best_shorts(pops: dict, out: Path, plt):
    """best_shorts.png and .csv, for the shorts runs only. If the shorts differ because
    the short damaged some of them, the undamaged ones are the best: the top of each
    array's distribution is then the transistor itself, and two arrays with the same
    transistors meet there even when their medians differ."""
    if not pops:
        return None
    fig, ax = plt.subplots(1, 3, figsize=(19, 5.6))
    cm = plt.get_cmap("tab10")
    res, rows = {}, []
    for i, (name, (p, vled)) in enumerate(pops.items()):
        c = cm(i)
        lab = f"{name.split('_')[1]} {name.split('_')[0]} ({len(p)} shorts)"
        s = p[(15, V_REF)].sort_values(ascending=False) * 1e6
        n = min(N_BEST, len(s))
        top = s.iloc[:n].expanding().mean().values
        res[name] = dict(top=top, median=float(s.median()), p90=float(s.quantile(0.9)))
        ax[0].plot(range(1, n + 1), top, "o-", color=c, label=lab)
        ax[0].axhline(s.median(), color=c, ls=":", lw=1)
        volts = sorted(v for b, v in p.columns if b == 15)
        vt = vled - np.array(volts)            # across the transistor: VLED - NVLED
        for k, idx in enumerate(s.index[:n]):
            ax[1].plot(vt, p.loc[idx, [(15, v) for v in volts]].values * 1e6, lw=0.8,
                       color=c, alpha=0.7, label=lab if k == 0 else None)
        ax[1].plot(vt, p[[(15, v) for v in volts]].median().values * 1e6, "--", lw=2,
                   color=c, label=f"{lab.split(' (')[0]}: median of all")
        rank = 100 * np.arange(1, len(s) + 1) / len(s)
        ax[2].plot(rank, s.values, ".-", ms=3, lw=0.8, color=c, label=lab)
        for k, (idx, v) in enumerate(s.iloc[:n].items()):
            rows.append(dict(run=name, rank=k + 1, X=idx[0], Y=idx[1], quadrant=idx[2],
                             I15_uA=v, mean_of_best_uA=top[k]))
    pd.DataFrame(rows).to_csv(out / "best_shorts.csv", index=False, float_format="%.3f")
    ax[0].set_xlabel("N, the best shorts")
    ax[0].set_ylabel(f"mean I at bitval 15, {V_REF:+.1f} V (uA)")
    ax[0].set_title("Mean of the N best shorts (dotted: median of all)", fontsize=10)
    ax[0].set_xticks(range(1, N_BEST + 1))
    ax[1].set_xlabel("VLED - NVLED, the voltage across the short (V)")
    ax[1].set_ylabel("I at bitval 15 (uA)")
    ax[1].set_title(f"The {N_BEST} best shorts of each array (dashed: median of all)",
                    fontsize=10)
    ax[2].set_xlabel("rank, % of the array's shorts (0 = best)")
    ax[2].set_ylabel(f"I at bitval 15, {V_REF:+.1f} V (uA)")
    ax[2].set_title("Every short, sorted: arrays with the same transistors meet at the "
                    "top", fontsize=10)
    for a in ax:
        a.grid(alpha=0.3)
        a.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out / "best_shorts.png", dpi=150)
    plt.close(fig)
    return res


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("runs", nargs="*", default=DEFAULT, help="transistoriv run folders")
    ap.add_argument("-o", "--out", default="comparisons/drivers",
                    help="output folder under results")
    a = ap.parse_args(argv)
    out = R / a.out
    out.mkdir(parents=True, exist_ok=True)
    plt = aa._plt()
    rows, nets, spreads, leds, emits = [], {}, [], {}, []
    for r in a.runs:
        run = R / Path(r).name
        p, bad, emit, src, sweep, vled = load(run)
        if len(emit):
            emits.append(emit.assign(run=run.name))
        row, net = stats(run.name, p, bad, len(emit), src)
        rows.append(row)
        nets[run.name] = (p, net, sweep.name if sweep else None, vled,
                          R.parent / src if not src.is_absolute() else src)
        print(f"{run.name}: {len(p)} drivers from {src.name}"
              + (f", sweep {sweep.name}" if sweep else ""))
        if sweep is not None and sweep.name not in leds:
            leds[sweep.name] = led_spread(sweep)
        led = leds.get(sweep.name) if sweep is not None else None
        for v in sorted({v for b, v in p.columns if b == 15}):
            d = dict(run=run.name, V=v, drv_I_uA=p[(15, v)].median() * 1e6,
                     drv_cv_pct=rcv(p[(15, v)]))
            if led is not None and v in led.index:
                d.update(led.loc[v].to_dict())
            spreads.append(d)
    st = pd.DataFrame(rows)
    st.to_csv(out / "driver_stats.csv", index=False, float_format="%.4g")
    sp = pd.DataFrame(spreads)
    sp.to_csv(out / "spread_vs_bias.csv", index=False, float_format="%.4g")

    lab = lambda n: f"{n.split('_')[1]} {n.split('_')[0]} ({src_of(n)})"
    src_of = lambda n: st.set_index("run").loc[n, "pixel_list"].replace(".csv", "")
    fig, ax = plt.subplots(2, 2, figsize=(14, 10))
    cm = plt.get_cmap("tab10")
    shown = set()                          # one LED curve per sweep
    for i, (name, (p, net, swn, _vled, _src)) in enumerate(nets.items()):
        c = cm(i)
        s = p[(15, V_REF)]
        for j, q in enumerate(QUADS):
            v = s[s.index.get_level_values("QUADRANT") == q] * 1e6
            x = j + (i - (len(nets) - 1) / 2) * 0.8 / len(nets)
            ax[0, 0].boxplot([v], positions=[x], widths=0.7 / len(nets), showfliers=False,
                             patch_artist=True, boxprops=dict(facecolor=c, alpha=0.4),
                             medianprops=dict(color=c))
            ax[0, 0].plot(np.full(len(v), x) + np.random.default_rng(j).uniform(
                -0.08, 0.08, len(v)) / len(nets), v, ".", ms=2, color=c,
                label=lab(name) if j == 0 else None)
        cv = [rcv(net[b]) for b in BITS]
        ax[0, 1].plot(BITS, cv, "o-", color=c, label=lab(name))
        ax[0, 1].plot(BITS, cv[0] / np.sqrt(BITS), ":", color=c)
        lin = [net[b].median() / net[1].median() for b in BITS]
        ax[1, 0].plot(BITS, lin, "o-", color=c, label=lab(name))
        ss = sp[sp["run"] == name]
        ax[1, 1].plot(ss["V"], ss["drv_cv_pct"], "s--", color=c, label=f"{lab(name)}: drivers")
        if "led_cv_pct" in ss and ss["led_cv_pct"].notna().any() and swn not in shown:
            shown.add(swn)
            m = ss["led_cv_pct"].notna()
            ax[1, 1].plot(ss.loc[m, "V"], ss.loc[m, "led_cv_pct"], "o-", color=c,
                          label=f"{name.split('_')[1]}: typical LED pixels ({swn[:11]} sweep)")
    ax[0, 0].set_xticks(range(4), QUADS)
    ax[0, 0].set_ylabel(f"I at bitval 15, {V_REF:+.1f} V (uA)")
    ax[0, 0].set_title("Driver current per quadrant", fontsize=10)
    ax[0, 0].legend(fontsize=7, markerscale=4)
    ax[0, 1].set_xscale("log", base=2)
    ax[0, 1].set_xticks(BITS, [str(b) for b in BITS])
    ax[0, 1].set_xlabel("code (bit weight)")
    ax[0, 1].set_ylabel("spread across drivers, robust CV (%)")
    ax[0, 1].set_title(f"Spread per code at {V_REF:+.1f} V; dotted: independent unit cells "
                       f"(CV ~ 1/sqrt(weight))", fontsize=10)
    ax[0, 1].legend(fontsize=7)
    ax[1, 0].plot(BITS, BITS, "k--", lw=0.8, label="ideal")
    ax[1, 0].set_xlabel("code")
    ax[1, 0].set_ylabel("I(code) / I(code 1), bitval 0 subtracted")
    ax[1, 0].set_title(f"Code linearity at {V_REF:+.1f} V", fontsize=10)
    ax[1, 0].legend(fontsize=7)
    ax[1, 1].set_yscale("log")
    ax[1, 1].set_xlim(-3.3, -0.45)          # above: the drivers leave saturation
    ax[1, 1].set_xlabel("NVLED (V)")
    ax[1, 1].set_ylabel("current spread, robust CV (%)")
    ax[1, 1].set_title("Voltage drive: current spread of LED pixels and of their drivers",
                       fontsize=10)
    ax[1, 1].legend(fontsize=7)
    for x in ax.flat:
        x.grid(alpha=0.3, which="both")
    fig.tight_layout()
    fig.savefig(out / "drivers.png", dpi=150)
    plt.close(fig)
    emits_thr = {e["run"].iloc[0]: float(e["floor_W"].iloc[0]) for e in emits}
    if emits:
        pd.concat(emits).reset_index().to_csv(out / "emitting_shorts.csv", index=False,
                                              float_format="%.4g")
    isolation({n: (v[0], v[4]) for n, v in nets.items()
              if st.set_index("run").loc[n, "pixel_list"] == "short.csv"}, out, plt)
    best = best_shorts({n: (v[0], v[3]) for n, v in nets.items()
                        if st.set_index("run").loc[n, "pixel_list"] == "short.csv"},
                       out, plt)

    f = lambda x, d=2: "" if pd.isna(x) else f"{x:.{d}f}"
    L = ["# Pixel drivers from transistoriv", "",
         "Robust CV = 1.4826 x MAD / median. Bit current = I(bit) - I(bitval 0). "
         f"Per-bit and per-quadrant numbers at {V_REF:+.1f} V.", ""]
    for r in rows:
        L += [f"## {r['run']} ({r['pixel_list']}, {r['n']} drivers, "
              f"{r['n_not_working']} not working, {r['n_emitting_excluded']} excluded "
              f"for light above the run's floor (offset + {LIGHT_SIGMA:g} sigma"
              + (f", {emits_thr[r['run']] * 1e9:.2f} nW" if r["run"] in emits_thr else "")
              + "); " + ", ".join(
                  f"{q} {r[f'n_{q}']}" for q in QUADS) + ")", ""]
        for v in ("-0.5V", "-1.4V", "-3.2V"):
            if f"I15_{v}_uA" in r:
                L.append(f"- bitval 15 at {v}: median {f(r[f'I15_{v}_uA'], 1)} uA, CV "
                         f"{f(r[f'cv_{v}_pct'])} %; between quadrant medians "
                         f"{f(r[f'between_quadrant_{v}_pct'])} %, within a quadrant "
                         f"{f(r[f'within_quadrant_{v}_pct'])} %; quadrants " + " ".join(
                             f"{q} {f(r.get(f'I15_{v}_{q}_uA', np.nan), 1)}" for q in QUADS))
        L.append("- per code, median uA / CV %: " + "; ".join(
            f"{b}: {f(r[f'bit{b}_uA'])} / {f(r[f'bit{b}_cv_pct'])}" for b in BITS))
        L.append("- code linearity I(code)/I(1), ideal 2 4 8 15: " + " ".join(
            f(r[f"code{b}_over_code1"]) for b in BITS[1:]) + f"; I15 / (I1+I2+I4+I8) "
            f"{f(r['additivity_15_over_sum'], 3)}")
        L.append(f"- one common per-driver gain explains {f(r['common_gain_share_pct'], 0)} % "
                 f"of the variance of bits 1-8; corr(bit 1, bit 8) "
                 f"{f(r['corr_bit1_bit8'])}")
        if "r_out_MOhm" in r and not pd.isna(r.get("r_out_MOhm")):
            L.append(f"- saturation: +{f(r['slope_pct_per_V'])} %/V from -1.5 to -3.2 V "
                     f"(r_out {f(r['r_out_MOhm'])} MOhm); 95 % of the -3.2 V current at "
                     f"NVLED {f(r['nvled_95pct_V'])} V")
        L.append(f"- gradient across the die: {f(r['gradient_X_pct'])} % in X, "
                 f"{f(r['gradient_Y_pct'])} % in Y")
        ss = sp[(sp["run"] == r["run"])]
        if "led_cv_pct" in ss and ss["led_cv_pct"].notna().any():
            L += ["", "| NVLED (V) | LED pixels I (uA) | LED pixels CV (%) | drivers I (uA) | "
                      "drivers CV (%) |", "|---|---|---|---|---|"]
            for _, s in ss[ss["led_cv_pct"].notna()].iterrows():
                L.append(f"| {s['V']:+.2f} | {f(s['led_I_uA'], 1)} | {f(s['led_cv_pct'])} | "
                         f"{f(s['drv_I_uA'], 1)} | {f(s['drv_cv_pct'])} |")
        L.append("")
    if best is not None:
        L += ["## The best shorts (bitval 15, -3.2 V; best_shorts.png)", "",
              "| run | best | mean of best 3 | best 5 | best 10 | median of all | "
              "top 10 % (percentile 90) |", "|---|---|---|---|---|---|---|"]
        for n, b in best.items():
            L.append(f"| {n} | {b['top'][0]:.1f} | {b['top'][2]:.1f} | {b['top'][4]:.1f} | "
                     f"{b['top'][-1]:.1f} | {b['median']:.1f} | {b['p90']:.1f} |")
        L.append("")
    (out / "summary.md").write_text("\n".join(L), encoding="utf-8")
    print("\n".join(L))
    return 0


if __name__ == "__main__":
    sys.exit(main())
