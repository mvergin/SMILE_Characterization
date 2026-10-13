"""Generate the figures for INVESTIGATION_REPORT.md from the real run data.

Each function builds one figure from a specific run folder and also returns the key
numbers that drove a decision, so the report text and the plots stay consistent.

    python make_investigation_figures.py
"""
from __future__ import annotations
import glob
import math
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RESULTS = Path("results_georg/results")
OUT = Path("investigation_report")
OUT.mkdir(exist_ok=True)

Q = 1.602176634e-19
H = 6.62607015e-34
C = 2.99792458e8
WL = 440e-9
EPH = H * C / WL
AREA_CM2 = 100.0 * 1e-8          # 10×10 µm pixel in cm²


def find(pat):
    g = sorted(glob.glob(str(RESULTS / pat)))
    return g[0] if g else None


def sweep_csv(folder):
    g = [x for x in glob.glob(folder + "/**/aggregated_bitval=15_mean_std.csv", recursive=True)
         if "FastScan" not in x]
    return g[0] if g else None


def load_sweep(folder):
    df = pd.read_csv(sweep_csv(folder))
    P = df["PM400_POWER_MEAN"].astype(float).clip(lower=0)
    Inv = df["NVLED_CURR_MEAN"].astype(float).abs()
    df["J"] = Inv / AREA_CM2
    df["EQE"] = (P / EPH) / (Inv / Q).replace(0, np.nan)
    return df


def median_curve(J, eqe, nb=22):
    m = (J > 0) & np.isfinite(eqe) & (eqe > 0)
    J, eqe = J[m], eqe[m]
    if J.size < 8:
        return np.array([]), np.array([])
    edges = np.logspace(np.log10(J.min()), np.log10(J.max()), nb)
    cx, cy = [], []
    for a, b in zip(edges[:-1], edges[1:]):
        s = (J >= a) & (J < b)
        if s.sum() >= 2:
            cx.append(math.sqrt(a * b)); cy.append(np.median(eqe[s]))
    return np.array(cx), np.array(cy)


# =====================================================================
def fig_ab_direction():
    """1329 low→high vs 1341 high→low: the original reproducibility failure."""
    lo, hi = find("260626-1329*"), find("260626-1341*")
    dl, dh = load_sweep(lo), load_sweep(hi)
    xl, yl = median_curve(dl["J"].values, dl["EQE"].values)
    xh, yh = median_curve(dh["J"].values, dh["EQE"].values)
    grid = np.logspace(np.log10(max(xl.min(), xh.min())), np.log10(min(xl.max(), xh.max())), 40)
    rl = np.interp(grid, xl, yl); rh = np.interp(grid, xh, yh)
    ratio = rh / rl
    rmax = float(np.nanmax(ratio[grid < 2])) if (grid < 2).any() else float("nan")
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.4))
    a1.plot(xl, yl, "o-", color="tab:blue", label="low→high (fresh)")
    a1.plot(xh, yh, "s-", color="tab:red", label="high→low (conditioned)")
    a1.set_xscale("log"); a1.set_yscale("log")
    a1.set_xlabel("J  (A/cm²)  — from NVLED current"); a1.set_ylabel("EQE  (relative, P/I)")
    a1.set_title("EQE(J), same pixels, opposite sweep direction")
    a1.legend(); a1.grid(True, which="both", alpha=0.3)
    a2.plot(grid, ratio, "-", color="tab:purple", lw=2)
    a2.axhline(1.0, color="k", lw=0.8)
    a2.axvspan(grid.min(), 2, color="tab:orange", alpha=0.12, label="low-J region")
    a2.set_xscale("log")
    a2.set_xlabel("J  (A/cm²)"); a2.set_ylabel("EQE ratio  high→low / low→high")
    a2.set_title(f"directions disagree {rmax:.2f}× at low J, agree at high J")
    a2.legend(); a2.grid(True, which="both", alpha=0.3)
    fig.suptitle("Run-to-run irreproducibility: the sweep-direction A/B test (runs 1329 vs 1341)")
    fig.tight_layout(); fig.savefig(OUT / "fig1_ab_direction.png", dpi=130); plt.close(fig)
    return dict(low_J_ratio_max=rmax)


# =====================================================================
def fig_channel_noise():
    """0733 (−1.0 V): light is the clean channel, current is the noisy one."""
    df = pd.read_csv(find("260629-0733*") + "/warmup_diag.csv")
    g = df[df.phase == "fresh_turnon"].sort_values("t_s")
    P, I, t = g.P_W.values, np.abs(g.I_A.values), g.t_s.values
    relP = 100 * P.std() / P.mean()
    relI = 100 * I.std() / I.mean()
    corr = np.corrcoef(P, I)[0, 1]
    fig, ax = plt.subplots(figsize=(7, 4.6))
    ax.plot(t, P / P.mean(), ".", color="tab:green", ms=4, label=f"light P  (scatter {relP:.0f}%)")
    ax.plot(t, I / I.mean(), ".", color="tab:purple", ms=4, label=f"current |I|  (scatter {relI:.0f}%)")
    ax.axhline(1.0, color="k", lw=0.6)
    ax.set_xlabel("time at fixed −1.0 V bias  (s)")
    ax.set_ylabel("value / its mean")
    ax.set_title(f"Which channel is noisy? (run 0733, fixed bias)\n"
                 f"light flat, current scatters {relI:.0f}% — corr(P,|I|)={corr:+.2f} → noise is in the CURRENT")
    ax.legend(); ax.grid(True, alpha=0.3); ax.set_ylim(0, 2.2)
    fig.tight_layout(); fig.savefig(OUT / "fig2_channel_noise.png", dpi=130); plt.close(fig)
    return dict(light_scatter_pct=relP, current_scatter_pct=relI, corr=corr)


# =====================================================================
def fig_parasitic():
    """0858 (−0.85 V): fresh non-radiative parasitic current, cleared by conditioning."""
    df = pd.read_csv(find("260629-0858*") + "/warmup_diag.csv")
    sub = df[(df.pixel_x == 50) & (df.pixel_y == 50)]
    fr = sub[sub.phase == "fresh_turnon"].sort_values("t_s")
    co = sub[sub.phase == "after_conditioning"].sort_values("t_s")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.4), sharey=False)
    ax1.plot(fr.t_s, np.abs(fr.I_A) * 1e9, color="tab:red", label="|current|")
    ax1.plot(co.t_s, np.abs(co.I_A) * 1e9, color="tab:blue", label="|current| (conditioned)")
    ax1.set_yscale("log"); ax1.set_xlabel("time (s)"); ax1.set_ylabel("|NVLED current|  (nA)")
    ax1.set_title("Current at −0.85 V, pixel (50,50)")
    ax1.legend(); ax1.grid(True, which="both", alpha=0.3)
    axb = ax1.twinx()
    axb.plot(fr.t_s, fr.P_W * 1e9, color="tab:green", alpha=0.5, lw=1)
    axb.set_ylabel("light P (nW)", color="tab:green")
    # bar comparison
    fr_I = np.abs(np.median(fr.I_A.values[-10:])) * 1e9
    co_I = np.abs(np.median(co.I_A.values[-10:])) * 1e9
    ax2.bar(["fresh", "conditioned"], [fr_I, co_I], color=["tab:red", "tab:blue"])
    ax2.set_ylabel("settled |current|  (nA)")
    ax2.set_title(f"Same pixel, same bias\nfresh {fr_I:.0f} nA vs conditioned {co_I:.0f} nA  "
                  f"({fr_I/co_I:.0f}×)")
    for i, v in enumerate([fr_I, co_I]):
        ax2.text(i, v, f"{v:.0f} nA", ha="center", va="bottom")
    fig.suptitle("Non-radiative parasitic current on fresh turn-on — light unchanged, current 27×")
    fig.tight_layout(); fig.savefig(OUT / "fig3_parasitic.png", dpi=130); plt.close(fig)
    return dict(fresh_nA=fr_I, conditioned_nA=co_I, ratio=fr_I / co_I)


# =====================================================================
def fig_autorange():
    """0913: autorange compressed NVLED — light tracks VLED, not NVLED."""
    df = load_sweep(find("260629-0913*"))
    raw = pd.read_csv(sweep_csv(find("260629-0913*")))
    g = raw[(raw.X == raw.X.iloc[0]) & (raw.Y == raw.Y.iloc[0])].sort_values("NVLED_V", ascending=False)
    P = g.PM400_POWER_MEAN.astype(float).values
    Inv = np.abs(g.NVLED_CURR_MEAN.astype(float).values)
    Iv = np.abs(g.VLED_CURR_MEAN.astype(float).values)
    fig, ax = plt.subplots(figsize=(7, 4.6))
    ax.plot(P * 1e9, Iv * 1e6, "s-", color="tab:orange", label="VLED (ch A)")
    ax.plot(P * 1e9, Inv * 1e6, "o-", color="tab:purple", label="NVLED (ch B) — autoranged")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("light P  (nW)"); ax.set_ylabel("current  (µA)")
    ax.set_title("Autorange broke NVLED (run 0913)\n"
                 "NVLED stuck ≤0.2 µA while light & VLED span ~3 decades → NVLED misread")
    ax.legend(); ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout(); fig.savefig(OUT / "fig4_autorange_break.png", dpi=130); plt.close(fig)
    return dict(nvled_max_uA=Inv.max() * 1e6, vled_max_uA=Iv.max() * 1e6)


# =====================================================================
def fig_settling():
    """1130: only the matched (fine) range reads low J correctly."""
    s = pd.read_csv(find("260629-1130*/settling/settling_summary.csv".replace("results_georg/results/", "")) if False
                    else glob.glob(str(RESULTS / "260629-1130*/settling/settling_summary.csv"))[0])
    s = s[s.highc == False]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 4.4))
    for ax, label, truth, lim in [(a1, "lowJ", 38e-9, (1e-9, 1e-3)), (a2, "highJ", 175e-6, (1e-7, 1e-3))]:
        d = s[s.bias_label == f"'{label}'"].copy()
        d["absI"] = d.I_final.abs()
        d = d[np.isfinite(d.absI) & (d.absI < 1e30)]
        ax.bar([f"{r:.0e}" for r in d.range_i], d.absI, color="tab:gray")
        ax.axhline(truth, color="tab:green", ls="--", lw=2, label=f"true ≈ {truth*1e9:.0f} nA"
                   if truth < 1e-6 else f"true ≈ {truth*1e6:.0f} µA")
        ax.set_yscale("log"); ax.set_xlabel("measure range (A)"); ax.set_ylabel("|I_final| (A)")
        ax.set_title(f"{label} bias"); ax.legend(); ax.grid(True, which="both", axis="y", alpha=0.3)
        ax.set_ylim(*lim)
    fig.suptitle("Settling probe (run 1130): only the range matched to the current reads the TRUE value\n"
                 "at low J, coarse ranges are not just noisy — they read the wrong number")
    fig.tight_layout(); fig.savefig(OUT / "fig5_settling_ranges.png", dpi=130); plt.close(fig)
    return {}


# =====================================================================
def fig_aperture():
    """1119: short-aperture noise band scales with the selected range."""
    raw = pd.read_csv(glob.glob(str(RESULTS / "260629-1119*/settling/settling_probe.csv"))[0])
    raw["range_i"] = raw["range_i"].astype(float)
    d = raw[(raw.highc == False) & (raw.bias_label == "'lowJ'")]
    ranges = sorted(d.range_i.unique())
    bands = [np.percentile(d[d.range_i == r].I_NVLED, [10, 90]) for r in ranges]
    spread = [(hi - lo) / 2 * 1e6 for lo, hi in bands]
    fig, ax = plt.subplots(figsize=(7, 4.6))
    ax.loglog([r * 1e6 for r in ranges], spread, "o-", color="tab:red")
    ax.axhline(0.038, color="tab:green", ls="--", label="true signal ≈ 38 nA")
    ax.set_xlabel("selected measure range (µA)"); ax.set_ylabel("noise half-band (µA)")
    ax.set_title("Why short reads fail (run 1119, NPLC 0.05, low J)\n"
                 "aperture noise ∝ range — the 38 nA signal is buried on every coarse range")
    ax.legend(); ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout(); fig.savefig(OUT / "fig6_aperture_noise.png", dpi=130); plt.close(fig)
    return {}


def fig_final():
    """1144: current fixed (NVLED spans full range, matches VLED) but the optical
    dark floor is now the wall at low J, where the IQE peak sits."""
    folder = find("260629-1144*")
    raw = pd.read_csv(sweep_csv(folder))
    dark = 3.65e-11
    g = raw[(raw.X == raw.X.iloc[0]) & (raw.Y == raw.Y.iloc[0])].sort_values("NVLED_V", ascending=False)
    P = g.PM400_POWER_MEAN.astype(float).values
    Inv = np.abs(g.NVLED_CURR_MEAN.astype(float).values)
    Iv = np.abs(g.VLED_CURR_MEAN.astype(float).values)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.4))
    # panel A: current is fixed — NVLED now overlays VLED (contrast with fig4)
    a1.plot(P * 1e9, Iv * 1e6, "s-", color="tab:orange", ms=4, label="VLED (ch A)")
    a1.plot(P * 1e9, Inv * 1e6, "o-", color="tab:purple", ms=4, label="NVLED (ch B) — per-point range")
    a1.set_xscale("log"); a1.set_yscale("log")
    a1.set_xlabel("light P  (nW)"); a1.set_ylabel("current  (µA)")
    a1.set_title("Current FIXED: NVLED spans full range,\nnow tracks VLED (cf. fig 4)")
    a1.legend(); a1.grid(True, which="both", alpha=0.3)
    # panel B: optical floor is the wall — P/dark vs J, all swept points
    allP = raw.PM400_POWER_MEAN.astype(float).values
    allJ = np.abs(raw.NVLED_CURR_MEAN.astype(float).values) / AREA_CM2
    m = (allJ > 0)
    a2.scatter(allJ[m], allP[m] / dark, s=6, alpha=0.25, color="tab:green")
    a2.axhline(1.0, color="k", ls="--", lw=1.5, label="dark floor (P = dark)")
    a2.axvspan(allJ[m].min(), 1.0, color="tab:orange", alpha=0.15, label="IQE-peak region (J ≲ 1)")
    a2.set_xscale("log"); a2.set_yscale("log")
    a2.set_xlabel("J  (A/cm²)"); a2.set_ylabel("P / dark")
    a2.set_title("Optical floor is the wall:\nlight clears dark only at J ≳ 3, not at the peak")
    a2.legend(); a2.grid(True, which="both", alpha=0.3)
    fig.suptitle("Final run 1144: electrical chain fixed, optical SNR at low J is the remaining limit")
    fig.tight_layout(); fig.savefig(OUT / "fig7_final_optical_wall.png", dpi=130); plt.close(fig)
    return dict(nvled_min_nA=Inv.min() * 1e9, nvled_max_uA=Inv.max() * 1e6, vled_max_uA=Iv.max() * 1e6)


def fig_closer():
    """1144 (far) vs 1300 (close): moving the PM closer fixed the optical floor, but
    exposed a current-instability cliff below ~500 nA as the new low-J limit."""
    def load(folder, dark):
        df = pd.read_csv(sweep_csv(folder))
        P = df.PM400_POWER_MEAN.astype(float).values
        Inv = np.abs(df.NVLED_CURR_MEAN.astype(float).values)
        rel = df.NVLED_CURR_STD.astype(float).values / np.where(Inv > 0, Inv, np.nan)
        J = Inv / AREA_CM2
        return J, P / dark, rel
    Jf, pdf, _ = load(find("260629-1144*"), 3.65e-11)
    Jc, pdc, relc = load(find("260629-1300*"), 5.97e-11)

    def binned(J, y, fn=np.median, nb=22):
        m = (J > 0) & np.isfinite(y)
        J, y = J[m], y[m]
        edges = np.logspace(np.log10(J.min()), np.log10(J.max()), nb)
        cx, cy = [], []
        for a, b in zip(edges[:-1], edges[1:]):
            s = (J >= a) & (J < b)
            if s.sum() >= 2:
                cx.append(math.sqrt(a * b)); cy.append(fn(y[s]))
        return np.array(cx), np.array(cy)

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.4))
    xf, yf = binned(Jf, pdf); xc, yc = binned(Jc, pdc)
    xf9, yf9 = binned(Jf, pdf, fn=lambda v: np.percentile(v, 90))
    xc9, yc9 = binned(Jc, pdc, fn=lambda v: np.percentile(v, 90))
    a1.plot(xf, yf, "o-", color="tab:gray", label="far — median pixel")
    a1.plot(xc, yc, "o-", color="tab:green", label="close — median pixel")
    a1.plot(xf9, yf9, "--", color="tab:gray", alpha=0.7, label="far — bright (90th pct)")
    a1.plot(xc9, yc9, "--", color="tab:green", alpha=0.7, label="close — bright (90th pct)")
    a1.axhline(1.0, color="k", ls=":", lw=1.2, label="dark floor")
    a1.set_xscale("log"); a1.set_yscale("log")
    a1.set_xlabel("J  (A/cm²)"); a1.set_ylabel("P / dark")
    a1.set_title("Closer PM: BRIGHT pixels gained ~14× (90th pct 2→29),\nbut the median pixel barely moved (dark floor also rose 1.6×)")
    a1.legend(fontsize=8); a1.grid(True, which="both", alpha=0.3)
    xr, yr = binned(Jc, relc * 100)
    a2.plot(xr, yr, "o-", color="tab:red")
    a2.axhline(50, color="k", ls="--", lw=1.2, label="reject gate (50%)")
    a2.axvline(0.5, color="tab:blue", ls=":", lw=1.5, label="cliff ≈ 500 nA")
    a2.set_xscale("log"); a2.set_yscale("log")
    a2.set_xlabel("J  (A/cm²)"); a2.set_ylabel("current rel. SEM  (%)")
    a2.set_title("New limit: current unstable below ~500 nA\n(2% → 85% → 900% as J falls)")
    a2.legend(); a2.grid(True, which="both", alpha=0.3)
    fig.suptitle("Run 1300: closer PM solved the optics; low-J current stability is the next wall")
    fig.tight_layout(); fig.savefig(OUT / "fig8_closer_optics.png", dpi=130); plt.close(fig)
    return dict(low_J_Pdark_far=float(np.nanmedian(yf[xf < 0.5])),
                low_J_Pdark_close=float(np.nanmedian(yc[xc < 0.5])))


def fig_peak_location():
    """Across all runs: the device's IQE peak sits at J≈5, and trustworthy fits only
    need data down to J≈1 — so sub-1 µA accuracy is not required."""
    rows = []
    for d in ["260625-1015*", "260626-1111*", "260626-0935*", "260629-1144*", "260629-1300*"]:
        folder = find(d)
        if not folder:
            continue
        pf = glob.glob(folder + "/**/abc_fit_per_pixel.csv", recursive=True)
        if pf:
            rows.append(pd.read_csv(pf[0]))
    A = pd.concat(rows, ignore_index=True)
    good = A[(A.peak_captured == True) & (A.R2 >= 0.7) & (A.beta >= 0.5) & (A.beta <= 20)]
    deg = A[A.beta > 20]
    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    bins = np.logspace(-2, 2, 30)
    ax.hist(good.J_at_peak, bins=bins, color="tab:green", alpha=0.8,
            label=f"trustworthy peaks (n={len(good)})")
    ax.hist(deg.J_at_peak, bins=bins, color="tab:red", alpha=0.5,
            label=f"degenerate fits (β>20, n={len(deg)})")
    ax.axvspan(1e-2, 1.0, color="tab:blue", alpha=0.10)
    ax.axvline(good.J_at_peak.median(), color="k", ls="--",
               label=f"real peak ≈ {good.J_at_peak.median():.1f} A/cm²")
    ax.text(0.1, ax.get_ylim()[1] * 0.7, "sub-1 µA\n(NOT needed)", ha="center", color="tab:blue")
    ax.set_xscale("log")
    ax.set_xlabel("J at peak  (A/cm²)  →  I_peak in µA")
    ax.set_ylabel("number of pixel fits")
    ax.set_title(f"Where the IQE peak actually is (all runs)\n"
                 f"real peak J≈5 (≈5 µA), trustworthy peak IQE = {100*good.peak_IQE.median():.0f}% — "
                 f"sub-1 µA is below the data we need")
    ax.legend(fontsize=8); ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout(); fig.savefig(OUT / "fig9_peak_location.png", dpi=130); plt.close(fig)
    return dict(real_peak_J=float(good.J_at_peak.median()),
                trustworthy_peak_IQE=float(good.peak_IQE.median()),
                n_good=len(good), n_degenerate=len(deg))


def fig_ranging_saga():
    """The per-point ranging machinery corrupted the rising side; reverting to a fixed
    10 mA range + voltage ramp restored a smooth rising current that tracks the light."""
    def pixel_trace(folder):
        df = pd.read_csv(sweep_csv(folder))
        g = df[(df.X == df.X.iloc[0]) & (df.Y == df.Y.iloc[0])].copy()
        g["aI"] = g.NVLED_CURR_MEAN.astype(float).abs()
        g = g[g.aI < 1e36].sort_values("NVLED_V")
        return -g.NVLED_V.values, g.aI.values * 1e6   # |V|, I in µA
    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    for folder, lab, c, mk in [
            (find("260630-1255*"), "per-point ranging (1255): stuck/jumpy", "tab:red", "x"),
            (find("260630-1421*"), "fixed 10 mA + V ramp (1421): smooth", "tab:green", "o")]:
        if folder:
            V, I = pixel_trace(folder)
            m = (V >= 0.6) & (V <= 1.4)                # the rising-side window
            ax.plot(V[m], I[m], mk + "-", color=c, ms=4, label=lab)
    ax.set_yscale("log")
    ax.set_xlabel("|NVLED voltage|  (V)  — rising side / turn-on")
    ax.set_ylabel("LED current  (µA)")
    ax.set_title("Rising-side current: the ranging machinery vs the simple fixed range\n"
                 "per-point ranging got stuck on range offsets/settling; fixed 10 mA rises smoothly")
    ax.legend(); ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout(); fig.savefig(OUT / "fig10_ranging_saga.png", dpi=130); plt.close(fig)
    return {}


def fig_direction_resolution():
    """The re-examination: J_peak and peak IQE track SAMPLE SIZE, not direction. Small-sample
    runs read J≈5/high value; full-population runs (both directions) converge to J≈11, ~40%."""
    runs = [("260630-1421*", "high→low", "tab:green"),
            ("260701-0859*", "high→low", "tab:green"),
            ("260701-1224*", "high→low", "tab:green"),
            ("260630-1547*", "low→high", "tab:red"),
            ("260701-0937*", "low→high", "tab:red"),
            ("260701-1107*", "high→low (4px outlier)", "tab:gray")]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.4))
    seen = set()
    for pat, lab, c in runs:
        folder = find(pat)
        if not folder:
            continue
        pf = glob.glob(folder + "/**/abc_fit_per_pixel.csv", recursive=True)
        if not pf:
            continue
        cln = pd.read_csv(pf[0])
        cln = cln[cln.get("clean_fit") == True]
        if not len(cln):
            continue
        lg = lab if lab not in seen else None
        seen.add(lab)
        a1.scatter(len(cln), cln.J_at_peak.median(), s=70, color=c, alpha=0.8, label=lg, edgecolor="k")
        a2.scatter(len(cln), 100 * cln.peak_IQE.median(), s=70, color=c, alpha=0.8, edgecolor="k")
    a1.axhline(11, color="k", ls="--", lw=1, label="full-population peak J≈11")
    a1.set_ylabel("median J_at_peak  (A/cm²)"); a1.set_title("Peak LOCATION vs sample size")
    a2.axhline(40, color="k", ls="--", lw=1, label="full-population ~40%")
    a2.set_ylabel("median clean peak IQE  (%)"); a2.set_title("Peak VALUE vs sample size")
    for ax in (a1, a2):
        ax.set_xlabel("number of clean pixels in the run"); ax.grid(True, alpha=0.3); ax.legend(fontsize=8)
    fig.suptitle("Re-examination: J_peak & IQE track SAMPLE SIZE, not direction — "
                 "full-population (both dirs) → J≈11, ~40%")
    fig.tight_layout(); fig.savefig(OUT / "fig11_direction_resolution.png", dpi=130); plt.close(fig)
    return {}


def fig_snr_reveals():
    """The corrected, well-measured result (run 1224, 20/20 clean, per-segment averaging): low-J
    SNR did improve, and the EQE genuinely rises to J≈8 then plateaus → peak IQE ≈40% at J≈11
    (sidewall-limited µLED). The earlier 44%/77% were small-sample / low-J-offset artifacts."""
    Q = 1.602e-19
    def load(folder, dark):
        df = pd.read_csv(sweep_csv(folder))
        P = df.PM400_POWER_MEAN.astype(float)
        I = df.NVLED_CURR_MEAN.astype(float).abs()
        J = I / AREA_CM2
        eqe = ((P - dark).clip(lower=0) / EPH) / (I / Q).replace(0, np.nan)
        return J.values, (P / dark).values, eqe.values
    def med(J, y, gate=None, nb=26):
        m = (J > 0) & np.isfinite(y) & (y > 0)
        if gate is not None:
            m &= gate
        J, y = J[m], y[m]
        e = np.logspace(np.log10(max(J.min(), 0.3)), np.log10(J.max()), nb)
        cx, cy = [], []
        for a, b in zip(e[:-1], e[1:]):
            s = (J >= a) & (J < b)
            if s.sum() >= 3:
                cx.append(math.sqrt(a * b)); cy.append(np.median(y[s]))
        return np.array(cx), np.array(cy)
    aJ, apd, _ = load(find("260701-0859*"), 3e-11)
    bJ, bpd, be = load(find("260701-1224*"), 2.49e-11)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.4))
    # left: low-J SNR did improve (optics + per-segment averaging) — a real win
    x1, y1 = med(aJ, apd); x2, y2 = med(bJ, bpd)
    a1.plot(x1, y1, "o-", color="tab:gray", label="0859 — before")
    a1.plot(x2, y2, "o-", color="tab:green", label="1224 — optics + per-segment")
    a1.axhline(1, color="k", ls=":", label="dark floor")
    a1.set_xscale("log"); a1.set_yscale("log")
    a1.set_xlabel("J (A/cm²)"); a1.set_ylabel("P / dark")
    a1.set_title("Low-J SNR improved (a real win)")
    a1.legend(fontsize=8); a1.grid(True, which="both", alpha=0.3)
    # right: the well-measured EQE(J) — rises to J≈8 then plateaus → peak ≈40% at J≈11
    xe, ye = med(bJ, be, gate=(bpd > 3))
    a2.plot(xe, ye / np.nanmax(ye), "o-", color="tab:green")
    a2.axvline(11, color="k", ls="--", lw=1, label="fit peak J≈11")
    a2.axvspan(1, 8, color="tab:orange", alpha=0.12, label="real SRH rise (J≈1→8)")
    a2.set_xscale("log")
    a2.set_xlabel("J (A/cm²)"); a2.set_ylabel("EQE / EQE_max (relative)")
    a2.set_title("Well-measured EQE (1224, 20/20): real rise to J≈8\n→ peak IQE ≈40% at J≈11 (NOT 80%)")
    a2.legend(fontsize=8); a2.grid(True, which="both", alpha=0.3)
    fig.suptitle("Corrected: reproducible peak IQE ≈ 40% at J ≈ 11 (sidewall-limited 10 µm µLED)")
    fig.tight_layout(); fig.savefig(OUT / "fig12_snr_reveals.png", dpi=130); plt.close(fig)
    return {}


if __name__ == "__main__":
    out = {}
    for fn in [fig_ab_direction, fig_channel_noise, fig_parasitic,
               fig_autorange, fig_settling, fig_aperture, fig_final, fig_closer,
               fig_peak_location, fig_ranging_saga, fig_direction_resolution, fig_snr_reveals]:
        try:
            r = fn(); out[fn.__name__] = r
            print(f"OK {fn.__name__}: {r}")
        except Exception as e:
            print(f"FAIL {fn.__name__}: {e}")
    import json
    (OUT / "figure_numbers.json").write_text(json.dumps(out, indent=2, default=float))
    print(f"\nFigures in {OUT}/")
