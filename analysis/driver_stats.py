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


def rcv(s):
    s = pd.Series(s).dropna()
    return 100 * 1.4826 * (s - s.median()).abs().median() / s.median() if len(s) else np.nan


def load(run: Path):
    t = pd.read_csv(run / "transistoriv.csv")
    t["I"] = t["NVLED_CURR_MEAN"].abs()
    t["V"] = t["NVLED_V"].round(2)
    p = t.pivot_table(index=["X", "Y", "QUADRANT"], columns=["BITVAL", "V"], values="I")
    ok = p[(15, V_REF)] > I_MIN
    args = json.loads((run / "run_config.json").read_text(encoding="utf-8")).get("args", {})
    src = Path(args.get("pixel_list") or "")
    sweep = next((d for d in R.iterdir() if d.name in str(src)), None)
    return p[ok], int((~ok).sum()), src, sweep


def stats(name, p, n_bad, src):
    row = dict(run=name, pixel_list=src.name, n=len(p), n_not_working=n_bad,
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


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("runs", nargs="*", default=DEFAULT, help="transistoriv run folders")
    ap.add_argument("-o", "--out", default="comparisons/drivers",
                    help="output folder under results")
    a = ap.parse_args(argv)
    out = R / a.out
    out.mkdir(parents=True, exist_ok=True)
    plt = aa._plt()
    rows, nets, spreads, leds = [], {}, [], {}
    for r in a.runs:
        run = R / Path(r).name
        p, bad, src, sweep = load(run)
        row, net = stats(run.name, p, bad, src)
        rows.append(row)
        nets[run.name] = (p, net, sweep.name if sweep else None)
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
    for i, (name, (p, net, swn)) in enumerate(nets.items()):
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

    f = lambda x, d=2: "" if pd.isna(x) else f"{x:.{d}f}"
    L = ["# Pixel drivers from transistoriv", "",
         "Robust CV = 1.4826 x MAD / median. Bit current = I(bit) - I(bitval 0). "
         f"Per-bit and per-quadrant numbers at {V_REF:+.1f} V.", ""]
    for r in rows:
        L += [f"## {r['run']} ({r['pixel_list']}, {r['n']} drivers, "
              f"{r['n_not_working']} not working; " + ", ".join(
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
    (out / "summary.md").write_text("\n".join(L), encoding="utf-8")
    print("\n".join(L))
    return 0


if __name__ == "__main__":
    sys.exit(main())
