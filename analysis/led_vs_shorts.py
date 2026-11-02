"""The LED itself, from a sweep and the transistor IV of the same array's shorts.

    python analysis/led_vs_shorts.py "results/<sweep run>"

A short is a pixel whose LED is bypassed: its current at bitval b is what that pixel's
driver gives at the bias (transistoriv runs, found by pixel_classes.shorts_iv). A working
pixel puts the LED in series with the same driver. Where the pixel draws far less than
the short at the same NVLED, the driver is not what limits the current and nearly all of
VLED - NVLED sits across the LED. Bound used: the driver's drop is at most
I_LED x (VLED - NVLED) / I_short, the short's chord resistance at that bias (an upper
bound on its on-resistance while its I-V is concave). So
    V_LED in [V_tot (1 - I_LED / I_short), V_tot],  V_tot = VLED - NVLED.

Writes REPORT_assets/led_vs_shorts/:
  bias_table.csv     per bias: shorts (bitval 15 and 0), sweep total, reference, I_LED,
                     the V_LED bounds, and V_LED if the driver depended on V_ds alone
  pixel_subthreshold.csv   per pixel: I_LED at the lowest biases with a reference, its
                     noise, the ideality between two of them, the excess of the total
                     current over its quadrant's median where no reference was taken
  leakage.png        bitval-0 current: shorts (driver off) against the sweep's reference
  led_iv.png         median LED I-V with the V_LED bounds, next to the shorts' driver IV
  ideality.png       per-pixel ideality and sub-threshold current, by fast-scan class
  summary.md         the numbers, for the report
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import analyze_array as aa
import pixel_classes as pc

VT = 0.025852          # kT/q at 300 K (V)
SNR = 5.0              # an I_LED counts as measured above this many SEMs


def _vled(run: Path) -> float:
    cfg = aa._load_config(run / "NVLEDSweep" / "Sweep")
    return float(cfg.get("vled_voltage", 1.8))


def bias_table(sw, tiv, vled):
    tiv = pc.main_run(tiv)
    t15 = tiv[tiv["BITVAL"] == 15]
    t0 = tiv[tiv["BITVAL"] == 0]
    sh = pd.DataFrame({
        "I_short_b15_A": t15.groupby(t15["NVLED_V"].round(4))["NVLED_CURR_MEAN"].apply(
            lambda s: s.abs().median()),
        "I_short_b0_A": t0.groupby(t0["NVLED_V"].round(4))["NVLED_CURR_MEAN"].apply(
            lambda s: s.abs().median())})
    g = sw.groupby("V")
    t = pd.DataFrame({"n": g.size(), "I_tot_A": g["I_tot"].median(),
                      "I_ref_A": g["I_ref"].median(), "I_LED_A": g["I_LED"].median(),
                      "I_LED_p10_A": g["I_LED"].quantile(.1),
                      "I_LED_p90_A": g["I_LED"].quantile(.9),
                      "I_sem_A": g["I_sem"].median(), "P_W": g["P"].median()})
    t = t.join(sh, how="outer")
    t.index.name = "NVLED_V"
    t["V_tot"] = vled - t.index.values
    r = t["I_LED_A"] / t["I_short_b15_A"]
    t["I_LED_over_I_short"] = r
    t["V_LED_min"] = t["V_tot"] * (1 - r.clip(0, 1))
    t["V_LED_max"] = t["V_tot"]
    # If the driver's current depended on V_ds alone, the short's I(V_tot) would be the
    # driver's I(V_ds): the V_ds at which the short draws I_LED is the drop.
    s = t["I_short_b15_A"].dropna().sort_index(ascending=False)     # V_tot ascending
    vt_s, i_s = vled - s.index.values, s.values
    mono = np.maximum.accumulate(i_s)
    vds = np.interp(t["I_LED_A"].clip(lower=0), mono, vt_s, left=np.nan, right=np.nan)
    t["V_LED_if_Vds_only"] = t["V_tot"] - vds
    return t.sort_index(ascending=False)


def pixel_table(sw, cls):
    """Per pixel: I_LED and its SEM at the lowest-|V| biases with a reference, the
    ideality between the two above SNR, and the excess total current over the quadrant
    median at each bias without a reference."""
    ref_v = sorted(v for v in sw["V"].unique() if sw.loc[sw["V"] == v, "I_ref"].notna().any())
    low = sorted(ref_v, reverse=True)[:4]          # e.g. -0.5, -0.6, -0.7, -0.8
    noref = sorted(v for v in sw["V"].unique() if v not in ref_v)
    p = sw[sw["V"].isin(low)].pivot_table(index=["X", "Y"], columns="V",
                                          values=["I_LED", "I_sem"])
    p.columns = [f"{a}_{b:+.2f}V" for a, b in p.columns]
    # ideality between the pair of adjacent biases where the median pixel is above SNR
    # at both, the lowest such pair: the exponential part, below any series drop
    med = sw[sw["V"].isin(low)].groupby("V")[["I_LED", "I_sem"]].median()
    ok = [v for v in sorted(low, reverse=True) if med.loc[v, "I_LED"] > SNR * med.loc[v, "I_sem"]]
    pair = (ok[0], ok[1]) if len(ok) >= 2 else None
    if pair:
        i1, i2 = p[f"I_LED_{pair[0]:+.2f}V"], p[f"I_LED_{pair[1]:+.2f}V"]
        with np.errstate(divide="ignore", invalid="ignore"):
            p["ideality"] = abs(pair[1] - pair[0]) / (VT * np.log(i2 / i1))
        p.loc[(i1 <= 0) | (i2 <= i1), "ideality"] = np.nan
    if noref:
        tot = sw[sw["V"].isin(noref)]
        exc = tot["I_tot"] - tot.groupby(["V", "QUADRANT"])["I_tot"].transform("median")
        e = tot.assign(exc=exc).pivot_table(index=["X", "Y"], columns="V", values="exc")
        e.columns = [f"I_excess_{v:+.2f}V" for v in e.columns]
        p = p.join(e)
    if cls is not None:
        p = p.join(cls)
    return p, low, pair, noref


def main(argv=None) -> int:
    run = Path((argv or sys.argv[1:])[0]).resolve()
    tiv = pc.shorts_iv(run)
    if tiv is None or not (run / "NVLEDSweep" / "Sweep").is_dir():
        print("no sweep or no transistoriv run on this run's shorts: nothing to do")
        return 0
    out = run / "REPORT_assets" / "led_vs_shorts"
    out.mkdir(parents=True, exist_ok=True)
    plt = aa._plt()
    vled = _vled(run)
    sw = pc.load_sweep(run)
    n_sh = tiv[["X", "Y"]].drop_duplicates().shape[0]
    print(f"  {sw[['X', 'Y']].drop_duplicates().shape[0]} sweep pixels, {n_sh} shorts "
          f"({', '.join(tiv['RUN'].unique())}), VLED {vled} V")
    t = bias_table(sw, tiv, vled)
    t.to_csv(out / "bias_table.csv")
    cf = run / "REPORT_assets" / "pixelclassification" / "sweep" / "sweep_classes.csv"
    cls = None
    if cf.is_file():
        c = pd.read_csv(cf)
        cls = c.set_index(["X", "Y"])[[k for k in c.columns if k.startswith("fastscan_")]
                                      ].iloc[:, 0].rename("fastscan_class")
    px, low, pair, noref = pixel_table(sw, cls)
    px.to_csv(out / "pixel_subthreshold.csv")

    # ---- leakage: bitval 0 in both ------------------------------------------------
    fig, ax = plt.subplots(1, 2, figsize=(15, 5.5))
    t0 = pc.main_run(tiv)
    t0 = t0[t0["BITVAL"] == 0]
    for q, col in aa.QUAD_COLORS.items():
        a = t0[t0["QUADRANT"] == q]
        if len(a):
            m = a.groupby("NVLED_V")["NVLED_CURR_MEAN"].apply(lambda s: s.abs().median())
            ax[0].plot(vled - m.index, m * 1e9, "--", color=col, lw=1,
                       label=f"{q} shorts, bitval 0 ({a[['X', 'Y']].drop_duplicates().shape[0]})")
        b = sw[sw["QUADRANT"] == q]
        mr = b.groupby("V")["I_ref"].median().dropna()
        mt = b.groupby("V")["I_tot"].median()
        ax[0].plot(vled - mr.index, mr * 1e9, "o-", ms=3, color=col, label=f"{q} sweep reference")
        ax[0].plot(vled - mt.index, mt * 1e9, ":", color=col, lw=1)
    lk = pd.concat([t["I_ref_A"], t["I_short_b0_A"]]).max()
    ax[0].set_ylim(0, 1.3 * lk * 1e9)        # the lit total runs off the top: it is lit
    ax[0].set_xlabel("VLED - NVLED (V)")
    ax[0].set_ylabel("|I_B| (nA)")
    ax[0].set_title("Bitval 0: shorts (dashed, driver off, other day) and the sweep's "
                    "reference (o); dotted: sweep lit total", fontsize=9)
    ax[0].legend(fontsize=7, ncol=2)
    ax[0].grid(alpha=0.3)
    ok = t["I_LED_A"].notna()
    ax[1].errorbar(t.loc[ok, "V_tot"], t.loc[ok, "I_LED_A"].abs() * 1e9,
                   yerr=t.loc[ok, "I_sem_A"] * 1e9, fmt="o-", ms=3, color="k",
                   label="sweep median |I_LED| = lit - reference")
    ax[1].plot(t.loc[ok, "V_tot"], t.loc[ok, "I_sem_A"] * 1e9, ":", color="0.4",
               label="median SEM of one reading")
    ax[1].plot(t["V_tot"], t["I_ref_A"] * 1e9, "s-", ms=3, color="tab:blue",
               label="sweep median reference (bitval 0)")
    ax[1].set_yscale("log")
    ax[1].set_xlim(vled + 0.3, vled + 1.3)
    ax[1].set_xlabel("VLED - NVLED (V)")
    ax[1].set_ylabel("nA")
    ax[1].set_title("The LED's own current below threshold against the leakage the "
                    "reference removes", fontsize=9)
    ax[1].grid(alpha=0.3, which="both")
    ax[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "leakage.png", dpi=150)
    plt.close(fig)

    # ---- LED I-V, median pixel, with the driver-drop bound --------------------------
    fig, ax = plt.subplots(figsize=(9, 6))
    # the band only where the driver passes at most half its current: beyond that the
    # bound allows any V_LED and says nothing
    tt = t[(t["I_LED_A"] > SNR * t["I_sem_A"]) & (t["I_LED_over_I_short"] < 0.5)]
    ax.fill_betweenx(tt["I_LED_A"] * 1e6, tt["V_LED_min"], tt["V_LED_max"],
                     color="tab:green", alpha=0.25, label="V_LED, bounds")
    tl = t[t["I_LED_A"] > SNR * t["I_sem_A"]]
    ax.plot(tl["V_LED_max"], tl["I_LED_A"] * 1e6, "o-", ms=3, color="tab:green",
            label="median pixel against V_tot (= V_LED where the driver drop is 0)")
    tv = t["I_short_b15_A"].dropna()
    ax.plot(vled - tv.index, tv * 1e6, "s-", ms=3, color="tab:red",
            label="shorts, bitval 15: the driver alone, against V_tot")
    w = t.dropna(subset=["V_LED_if_Vds_only"])
    w = w[w["I_LED_A"] > SNR * w["I_sem_A"]]
    ax.plot(w["V_LED_if_Vds_only"], w["I_LED_A"] * 1e6, "x--", color="0.4",
            label="V_LED if the driver depended on V_ds only")
    ax.set_yscale("log")
    ax.set_xlabel("V (V)")
    ax.set_ylabel("I (uA)")
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=8)
    ax.set_title("LED I-V of the median pixel, de-embedded from the shorts' driver IV",
                 fontsize=10)
    fig.tight_layout()
    fig.savefig(out / "led_iv.png", dpi=150)
    plt.close(fig)

    # ---- per pixel: ideality and sub-threshold current by class ---------------------
    lo = f"I_LED_{low[0]:+.2f}V" if low else None
    fig, ax = plt.subplots(1, 2, figsize=(15, 5.5))
    groups = (px.groupby("fastscan_class") if "fastscan_class" in px else [("all", px)])
    for name, gdf in groups:
        col = aa.PIXEL_CLASSES.get(name, ("k",))[0]
        if "ideality" in gdf and gdf["ideality"].notna().sum() > 5:
            ax[0].hist(gdf["ideality"].dropna(), bins=np.linspace(0.5, 4, 141), density=True,
                       histtype="step", color=col,
                       label=f"{name}: median {gdf['ideality'].median():.2f} "
                             f"({gdf['ideality'].notna().sum()})")
        if lo and gdf[lo].notna().sum() > 5:
            ax[1].hist(gdf[lo].dropna() * 1e9, bins=np.linspace(-5, 15, 161), density=True,
                       histtype="step", color=col,
                       label=f"{name}: median {gdf[lo].median() * 1e9:.2f} nA")
    if pair:
        ax[0].set_xlabel(f"ideality n from I_LED at {pair[0]:+.2f} and {pair[1]:+.2f} V")
    ax[0].set_ylabel("density")
    ax[0].legend(fontsize=7)
    ax[0].grid(alpha=0.3)
    if lo:
        ax[1].set_xlabel(f"I_LED at {low[0]:+.2f} V (V_tot {vled - low[0]:.2f} V), nA")
        ax[1].set_yscale("log")
        ax[1].legend(fontsize=7)
        ax[1].grid(alpha=0.3)
    fig.suptitle("Per pixel, by fast-scan class", fontsize=11)
    fig.tight_layout()
    fig.savefig(out / "ideality.png", dpi=150)
    plt.close(fig)

    # ---- summary ---------------------------------------------------------------------
    L = [f"# LED vs shorts: {run.name}", "",
         f"VLED {vled} V. {len(px)} sweep pixels. Driver curve: the "
         f"{pc.main_run(tiv)[['X', 'Y']].drop_duplicates().shape[0]} shorts of "
         f"{pc.main_run(tiv)['RUN'].iloc[0]} (median per bias).", "",
         "| NVLED (V) | V_tot (V) | I_short b15 (uA) | I_short b0 (nA) | sweep ref (nA) | "
         "I_LED median (nA) | SEM (nA) | I_LED/I_short | V_LED bounds (V) | "
         "V_LED if V_ds only (V) |", "|" + "---|" * 10]
    for v, r in t.iterrows():
        f = lambda x, s=1, d=3: "" if not np.isfinite(x) else f"{x * s:.{d}f}"
        L.append(f"| {v:+.2f} | {r['V_tot']:.2f} | {f(r['I_short_b15_A'], 1e6, 1)} | "
                 f"{f(r['I_short_b0_A'], 1e9, 0)} | {f(r['I_ref_A'], 1e9, 0)} | "
                 f"{f(r['I_LED_A'], 1e9, 2)} | {f(r['I_sem_A'], 1e9, 2)} | "
                 f"{f(r['I_LED_over_I_short'], 1, 4)} | "
                 + (f"{r['V_LED_min']:.3f}-{r['V_LED_max']:.2f} | "
                    if np.isfinite(r["V_LED_min"]) else " | ")
                 + f"{f(r['V_LED_if_Vds_only'], 1, 3)} |")
    if pair:
        L += ["", f"Ideality from {pair[0]:+.2f} / {pair[1]:+.2f} V: median "
                  f"{px['ideality'].median():.2f}, 10-90 % {px['ideality'].quantile(.1):.2f}-"
                  f"{px['ideality'].quantile(.9):.2f} ({px['ideality'].notna().sum()} pixels)."]
    if lo:
        i0, s0 = px[lo], px[f"I_sem_{low[0]:+.2f}V"]
        lk = (i0 > SNR * s0) & (i0 > 5 * i0.median())
        L += [f"I_LED at {low[0]:+.2f} V (V_tot {vled - low[0]:.2f} V): median "
              f"{i0.median() * 1e9:.2f} nA, so the median shunt resistance is above "
              f"{(vled - low[0]) / max(i0.median(), 1e-15) / 1e9:.1f} GOhm. "
              f"{int(lk.sum())} pixels draw more than 5x the median at more than "
              f"{SNR:g} SEM (pixel_subthreshold.csv)."]
    for v in noref:
        k = f"I_excess_{v:+.2f}V"
        if k in px:
            e = px[k]
            mad = 1.4826 * (e - e.median()).abs().median()
            L.append(f"No reference at {v:+.2f} V: total minus quadrant median, MAD "
                     f"{mad * 1e9:.2f} nA; {int((e > 10 * mad).sum())} pixels above 10 MAD.")
    (out / "summary.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))
    return 0


if __name__ == "__main__":
    sys.exit(main())
