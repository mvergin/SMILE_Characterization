"""Where a pixel's voltage goes, and what the LED part looks like: plots for discussion.

    python analysis/led_voltage_budget.py

A pixel is driver + LED (+ bump and interconnect) in series across V_tot = VLED - NVLED. The
driver's drop comes from its model (bond_resistance.fit_driver), so V_LED = V_tot - V_driver(I)
is everything else. The LED's own light then splits V_LED in two:

    V_j  = V_LED(I_REF) + N_R kT/q ln(P / P(I_REF))     junction: emission ~ exp(q V_j / N_R kT)
    V_ex = V_LED - V_j                                     excess: barriers and series resistance

N_R = 1.07 is the optical ideality the pixels show at nA, where nothing else drops a voltage.
The excess is counted from I_REF = 0.6 uA, where the die's light clears its photodiode floor
(the pixels' excess is < 10 mV there). The same split is done for the SGA_15 10 um die from
its own IVL measurement (2023-08-18: LED current and photodiode current, 2.20-3.60 V in 10 mV
steps; `SGA_15 10um/SGA_15_10um_IVL.csv`, converted from the .xls, kept local).

Inputs: the typical pixels' median sweep curves (class_curves.csv of 261102-1703 and
261026-0813), every swept pixel (per-pixel excess), the fast scans' medians, R_best per pixel
(bond_resistance.py), the SGA_15 die. Writes results/comparisons/led_voltage/: the figures
(NN_*.png), median_curves.csv, fits.csv, per_pixel_<array>.csv.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import least_squares

import analyze_array as aa
import bond_resistance as br
import driver_stats as ds
import pixel_classes as pc

OUT = br.REPO / "results" / "comparisons" / "led_voltage"
DIE_IVL = br.REPO / "SGA_15 10um" / "SGA_15_10um_IVL.csv"
SHARE = Path(r"X:\public\Schoettler\von Max\SMILE_IQE_Results\runs\results")
VT = 0.025852                # kT/q at 300 K
N_R = 1.07                   # optical ideality at nA (Array42 1.06-1.08, Array38 1.07)
I_REF = 0.6e-6               # the excess is counted from here
CAP_FRAC = 0.7               # the driver's drop is trusted below 70 % of what it can carry
TARGETS = (1.5e-6, 5e-6, 20e-6)
ARRAYS = ("Array42", "Array38")
COL = {"Array42": "tab:blue", "Array38": "tab:orange", "die": "k"}
DIE_LABEL = "SGA_15 10 um die (unbonded, probed 2023)"
FASTSCANS = {"Array42": [("261108-1227_Array42_ROI_(0, 0, 511, 511)_FASTSCAN", -1.2),
                         ("261102-1703_Array42_ROI_(0, 0, 511, 511)", -1.4),
                         ("261108-0025_Array42_ROI_(0, 0, 511, 511)_FASTSCAN", -1.6),
                         ("261108-0316_Array42_ROI_(0, 0, 511, 511)_FASTSCAN", -2.2),
                         ("261107-1918_Array42_ROI_(0, 0, 511, 511)_FASTSCAN", -2.2),
                         ("261107-1918_Array42_ROI_(0, 0, 511, 511)_FASTSCAN", -3.2)],
             "Array38": [("261026-0813_Array38_ROI_(0, 0, 511, 511)", -1.4),
                         ("261029-1028_Array38_ROI_(0, 0, 511, 511)_FASTSCAN", -2.2),
                         ("261029-1028_Array38_ROI_(0, 0, 511, 511)_FASTSCAN", -3.2)]}


def tag(i):
    return f"{i * 1e6:g}uA".replace(".", "p")


def run_path(run):
    try:
        return br.run_dir(run)
    except FileNotFoundError:
        return SHARE / run


def capacity(vtot, P):
    """What the driver can carry with all of V_tot across it (the gate term is 0: PMOS)."""
    return br.f15(np.asarray(vtot, float), 0.0, P)


def interp_log(i, x, y):
    """y at currents x, interpolated in ln I over (i, y), i ascending; NaN outside."""
    return np.interp(np.log(x), np.log(i), y, left=np.nan, right=np.nan)


def mid(a):
    return np.sqrt(a[1:] * a[:-1])


# ---- data -----------------------------------------------------------------------------------
def light_split(i, v, light, ok):
    """Junction and excess voltage, the excess counted from I_REF (log-interpolated)."""
    m = ok & np.isfinite(v) & (light > 0)
    v0 = np.interp(np.log(I_REF), np.log(i[m]), v[m])
    l0 = np.interp(np.log(I_REF), np.log(i[m]), np.log(light[m]))
    vj = v0 + N_R * VT * (np.log(np.where(light > 0, light, np.nan)) - l0)
    return vj, v - vj


def load_die():
    d = pd.read_csv(DIE_IVL).sort_values("Voltage_V")
    d = d[d.Current_A > 0].reset_index(drop=True)
    d["L"] = -d.Photodiode_A                       # the photodiode reads negative
    floor = float(d.L[d.Voltage_V < 2.5].std())
    d["light_ok"] = d.L > 20 * floor
    i, v, light = d.Current_A.to_numpy(), d.Voltage_V.to_numpy(), d.L.to_numpy()
    d["V_j"], d["V_ex"] = light_split(i, v, light, d.light_ok.to_numpy() & (i >= 0.3e-6))
    d.loc[~d.light_ok, ["V_j", "V_ex"]] = np.nan
    d.attrs["floor"] = floor
    return d


def median_curve(arr, P):
    cc = pd.read_csv(ds.R / br.SWEEP[arr] / "REPORT_assets/pixelclassification/sweep/class_curves.csv")
    w = cc[(cc.cls == "typical") & cc.has_ref & (cc.I_LED_med > 0)].sort_values("V", ascending=False)
    d = pd.DataFrame({"V_tot": (br.VLED - w.V).to_numpy(), "I": w.I_LED_med.to_numpy(),
                      "I_tot": w.I_tot_med.to_numpy(), "I_p25": w.I_LED_p25.to_numpy(),
                      "I_p75": w.I_LED_p75.to_numpy(), "P": w.P_med.to_numpy()})
    d["leak"] = d.I_tot - d.I
    d["cap"] = capacity(d.V_tot, P)
    d["ok"] = d.I <= CAP_FRAC * d.cap
    d["V_drv"] = br.v_driver(d.I.to_numpy(), 0.0, P)
    d["V_LED"] = d.V_tot - d.V_drv
    i, v, p = d.I.to_numpy(), d.V_LED.to_numpy(), d.P.to_numpy()
    d["V_j"], d["V_ex"] = light_split(i, v, p, d.ok.to_numpy() & (p > 2e-9))
    return d


def fs_median(arr, run, V):
    """Median current and light of a fast scan's typical pixels at bias V, leakage removed
    per quadrant (the median current of its dark pixels)."""
    fs = aa._load_aggregated(run_path(run) / "FastScan")
    fs = fs[fs.NVLED_V.astype(float).round(2) == round(V, 2)]
    P = fs.PM400_POWER_MEAN.astype(float).abs()
    I = fs.NVLED_CURR_MEAN.astype(float).abs()
    p90 = P.quantile(0.9)
    alive = P > 0.1 * p90
    # open pixels: dark and drawing almost nothing. Shorts are dark too, but carry the
    # driver's full current (on Array42 they outnumber the opens)
    opn = (P < 0.01 * p90) & (I < 0.05 * I[alive].median())
    q = fs.QUADRANT.astype(str)
    leak = I[opn].groupby(q[opn]).median()
    i_led = I - q.map(leak).fillna(float(I[opn].median())).to_numpy()
    mi, mp = i_led[alive].median(), P[alive].median()
    typ = alive & ((i_led / mi - 1).abs() < 0.15) & ((P / mp - 1).abs() < 0.5)
    return dict(array=arr, run=run[:11], V=V, V_tot=br.VLED - V, I=float(i_led[typ].median()),
                P=float(P[typ].median()), n=int(typ.sum()), n_open=int(opn.sum()),
                leak=float(I[opn].median()))


def per_pixel(arr, P, die):
    """Every swept pixel: V_LED and light at I_REF and at TARGETS (log-interpolated between its
    own sweep points), its excess voltage there, and its V_LED against the die's."""
    sw = pc.load_sweep(ds.R / br.SWEEP[arr])
    # a clean reference reads the array's leakage, 0.15-0.18 uA, alike at every pixel. One
    # taken with the pixel still partly lit reads more and over-subtracts the current: at
    # -0.8..-1.0 V, ~10 % of Array42's swept pixels read 0.3-3.5 uA, and a 1 uA cut lets
    # most of them through (they fake a low excess). So: within 1.3 x the bias's median.
    ref_med = sw.groupby("V").I_ref.transform("median")
    sw = sw[sw.I_ref.notna() & (sw.I_ref <= 1.3 * ref_med) & (sw.I_LED > 0)
            & (sw.P > 5 * sw.P_noise)].copy()                 # light above noise
    sw["V_tot"] = br.VLED - sw.V
    sw = sw[sw.I_LED <= CAP_FRAC * capacity(sw.V_tot.to_numpy(), P)]
    sw["V_LED"] = sw.V_tot - br.v_driver(sw.I_LED.to_numpy(), 0.0, P)
    sw = sw[np.isfinite(sw.V_LED)]
    piv = sw.pivot_table(index=["X", "Y"], columns="V_tot", values=["I_LED", "V_LED", "P"])
    lni = np.log(piv["I_LED"].to_numpy(float))
    vl = piv["V_LED"].to_numpy(float)
    lnp = np.log(piv["P"].to_numpy(float))
    cols, rows = np.arange(lni.shape[1]), np.arange(lni.shape[0])

    def at(t):
        lt = np.log(t)
        lo = np.where(lni <= lt, cols, -1).max(1)
        hi = np.where(lni >= lt, cols, 10 ** 6).min(1)
        ok = (lo >= 0) & (hi < 10 ** 6) & (hi >= lo)
        lo, hi = np.where(ok, lo, 0), np.where(ok, hi, 0)
        x0, x1 = lni[rows, lo], lni[rows, hi]
        w = np.where(hi > lo, (lt - x0) / np.where(hi > lo, x1 - x0, 1), 0.0)

        def g(a):
            return np.where(ok, a[rows, lo] + w * (a[rows, hi] - a[rows, lo]), np.nan)
        return g(vl), g(lnp)

    v0, l0 = at(I_REF)
    res = pd.DataFrame(index=piv.index)
    die_v = lambda t: float(interp_log(die.Current_A.to_numpy(), np.array([t]),
                                       die.Voltage_V.to_numpy())[0])
    for t in TARGETS:
        v, lp = at(t)
        res[f"V_ex_{tag(t)}"] = (v - v0) - N_R * VT * (lp - l0)
        res[f"V_LED_minus_die_{tag(t)}"] = v - die_v(t)
    rb = pd.read_csv(br.OUT / "pixels" / f"{br.REFERENCE_RUN[arr]}_m1p40V.csv").set_index(["X", "Y"])
    res = res.join(rb[["R_best_kOhm", "quadrant", "I_LED"]].rename(columns={"I_LED": "I_fastscan"}),
                   how="left")
    # a second, latched driver adds current in the sweep (not in the same run's fast scan,
    # where such pixels are classified out): one-sided, a latch only adds
    vc = piv["I_LED"].columns.to_numpy(float)
    k = int(np.argmin(np.abs(vc - (br.VLED + 1.4))))
    res["I_sweep_over_fastscan"] = piv["I_LED"].to_numpy(float)[:, k] / res.I_fastscan.to_numpy()
    res["inflated"] = res.I_sweep_over_fastscan > 1.05
    res.loc[res.inflated, [c for c in res if c.startswith("V_")]] = np.nan
    return res.reset_index()


# ---- fits -------------------------------------------------------------------------------------
MODELS = {
    "resistor only": (lambda p, i: (i - I_REF) * p[0], [3e3], ([-1e5], [1e6])),
    "barrier only": (lambda p, i: p[0] * (np.log1p(i / 10 ** p[1]) - np.log1p(I_REF / 10 ** p[1])),
                     [0.1, -5.5], ([0, -9], [2, -2])),
    "barrier + resistor": (lambda p, i: p[0] * (np.log1p(i / 10 ** p[1]) - np.log1p(I_REF / 10 ** p[1]))
                           + (i - I_REF) * p[2], [0.1, -5.5, 500], ([0, -9, -2e4], [2, -2, 1e5])),
}


def fit(i, vex, model):
    f, p0, bounds = MODELS[model]
    r = least_squares(lambda p: f(p, i) - vex, p0, bounds=bounds)
    cov = np.linalg.pinv(r.jac.T @ r.jac) * (r.fun @ r.fun) / max(1, len(i) - len(p0))
    return dict(p=r.x, se=np.sqrt(np.abs(np.diag(cov))), rms=float(np.sqrt(np.mean(r.fun ** 2))),
                f=lambda x, p=r.x: f(p, x), n=len(i))


def describe(model, r):
    p, se = r["p"], r["se"]
    if model == "resistor only":
        s = f"R {p[0] / 1e3:.2f} kOhm"
    else:
        s = f"a {1e3 * p[0]:.0f} mV, I_b {10 ** p[1] * 1e6:.2g} uA"
        if model == "barrier + resistor":
            s += f", R {p[2] / 1e3:.2f} +- {se[2] / 1e3:.2f} kOhm"
    return f"{model}: {s}, rms {1e3 * r['rms']:.1f} mV"


def excess_points(d, i_col, ok):
    m = ok & np.isfinite(d.V_ex) & (d[i_col] >= I_REF)
    return d[i_col][m].to_numpy(), d.V_ex[m].to_numpy()


# ---- figures ----------------------------------------------------------------------------------
def save(fig, name, plt):
    fig.tight_layout()
    fig.savefig(OUT / name, dpi=150)
    plt.close(fig)


def fig_driver(drv, mc, plt):
    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    vv = np.linspace(0, 5, 501)
    for arr in ARRAYS:
        P = drv[arr]
        vt, i = P["data"]
        ax.plot(vt - i * br.R_SHORT, i * 1e6, "o", ms=3, color=COL[arr], alpha=.5,
                label=f"{arr}: 10 best shorts, their own path removed")
        ax.plot(vv, br.f15(vv, 0.0, P) * 1e6, "-", color=COL[arr],
                label=f"{arr}: model" + (" (Array42's shape, current x "
                                          f"{P['scale']:.3f})" if "shape_from" in P else ""))
        k = mc[arr][mc[arr].ok]
        ax.plot(k.V_drv, k.I * 1e6, "s", mfc="none", ms=7, color=COL[arr],
                label=f"{arr}: typical pixels' operating points")
    ax.set(xlabel="voltage across the driver (V)", ylabel="current (uA)", xlim=(0, 5), ylim=(0, 150),
           title="The pixel transistor (all four on, code 15), and where the pixels sit on it")
    ax.grid(alpha=.3)
    ax.legend(fontsize=7, loc="lower right")
    save(fig, "00_driver.png", plt)


def fig_iv(mc, die, fsm, plt, log):
    fig, ax = plt.subplots(figsize=(7.5, 5.8))
    for arr in ARRAYS:
        d = mc[arr]
        ax.plot(d.V_tot, d.I * 1e6, "-o", ms=3, color=COL[arr], label=f"{arr}: whole pixel (V_tot)")
        k = d[d.ok]
        ax.plot(k.V_LED, k.I * 1e6, "--s", ms=3, color=COL[arr],
                label=f"{arr}: driver subtracted (LED + bump + lines)")
        if not log:
            ax.fill_between(d.V_tot, d.I_p25 * 1e6, d.I_p75 * 1e6, color=COL[arr], alpha=.15, lw=0)
        f = fsm[fsm.array == arr]
        ax.plot(f.V_tot, f.I * 1e6, "D", mfc="none", ms=8, color=COL[arr],
                label=f"{arr}: full-array fast scans, median")
    ax.plot(die.Voltage_V, die.Current_A * 1e6, "-", color="k", lw=1.3, label=DIE_LABEL)
    if log:
        ax.set(yscale="log", ylim=(1e-3, 300), xlim=(2.4, 5.1))
    else:
        ax.set(ylim=(0, 135), xlim=(2.6, 5.1))
    ax.set(xlabel="voltage (V)", ylabel="LED current, leakage removed (uA)",
           title="I-V of the typical pixel: whole, and with the transistor subtracted"
           + ("" if log else "\n(bands: 25-75 % of the pixels)"))
    ax.grid(alpha=.3, which="both")
    ax.legend(fontsize=7, loc="lower right")
    save(fig, "01_iv_log.png" if log else "02_iv_linear.png", plt)


def fig_budget(arr, d, fits_, plt):
    d = d[d.V_tot >= 2.55].copy()
    r = fits_[(arr, "barrier + resistor")]
    v_led = np.where(d.ok, d.V_LED, d.V_j + r["f"](d.I.to_numpy()))   # driver-limited: the rest
    vj = d.V_j.to_numpy()
    vex = np.clip(v_led - vj, 0, None)
    x = d.V_tot.to_numpy()
    fig, ax = plt.subplots(figsize=(8, 5.8))
    base = 2.4
    ax.fill_between(x, base, vj, color="tab:green", alpha=.55,
                    label="LED junction (from its light)")
    ax.fill_between(x, vj, vj + vex, color="tab:red", alpha=.5,
                    label="excess: barrier + series resistance (LED, bump, lines)")
    ax.fill_between(x, vj + vex, x, color="tab:gray", alpha=.5, label="transistor (driver)")
    sat = ~d.ok.to_numpy()
    if sat.any():
        ax.fill_between(x, vj + vex, x, where=np.r_[sat[1:] | sat[:-1], sat[-1]], facecolor="none",
                        hatch="//", edgecolor="0.3", lw=0,
                        label="driver-limited: driver = the rest (LED extrapolated by the fit)")
    ax.plot(x, x, "k-", lw=1)
    lines = ["NVLED   V_tot      I    junction   excess         driver"]
    for vt in (3.0, 3.2, 3.4, 4.0, 5.0):
        k = np.argmin(np.abs(x - vt))
        if abs(x[k] - vt) > 0.01:
            continue
        drv = x[k] - vj[k] - vex[k]
        lines.append(f"{br.VLED - x[k]:+.1f} V  {x[k]:.1f} V  {d.I.iloc[k] * 1e6:5.1f} uA  "
                     f"{vj[k]:.2f} V  {1e3 * vex[k]:4.0f} mV {100 * vex[k] / x[k]:3.0f} %  "
                     f"{1e3 * drv:5.0f} mV {100 * drv / x[k]:3.0f} %")
    ax.text(0.02, 0.62, "\n".join(lines), transform=ax.transAxes, fontsize=6.6,
            family="monospace", va="top", bbox=dict(boxstyle="round", fc="w", alpha=.9))
    ax.set(xlabel="V_tot = VLED - NVLED (V)", ylabel=f"voltage (V); the axis starts at {base} V",
           ylim=(base, 5.3), xlim=(x.min(), x.max() + 0.05),
           title=f"{arr}, typical pixel: where the voltage goes")
    ax.grid(alpha=.3)
    ax.legend(fontsize=7, loc="upper left")
    save(fig, f"03_voltage_budget_{arr}.png", plt)


def fig_ideality(mc, die, plt):
    fig, ax = plt.subplots(figsize=(7.8, 5.6))
    for arr in ARRAYS:
        k = mc[arr][mc[arr].ok & (mc[arr].P > 2e-9)].sort_values("I")
        i, v, p = k.I.to_numpy(), k.V_LED.to_numpy(), k.P.to_numpy()
        ax.plot(mid(i) * 1e6, np.diff(v) / (VT * np.diff(np.log(p))), "-o", ms=4, color=COL[arr],
                label=f"{arr}: from the light")
        ax.plot(mid(i) * 1e6, np.diff(v) / (VT * np.diff(np.log(i))), "--s", ms=3, color=COL[arr],
                alpha=.7, label=f"{arr}: from the current")
    k = die[die.light_ok & (die.Current_A > 0.2e-6)].iloc[::3]
    i, v, light = k.Current_A.to_numpy(), k.Voltage_V.to_numpy(), k.L.to_numpy()
    ax.plot(mid(i) * 1e6, np.diff(v) / (VT * np.diff(np.log(light))), "-", color="k", lw=1.5,
            label="die: from the light")
    k = die[die.Current_A > 0.02e-6].iloc[::3]
    i, v = k.Current_A.to_numpy(), k.Voltage_V.to_numpy()
    ax.plot(mid(i) * 1e6, np.diff(v) / (VT * np.diff(np.log(i))), "--", color="k", lw=1, alpha=.7,
            label="die: from the current")
    ax.axvspan(1e-3, 0.6, color="0.85", zorder=0,
               label="pixels: leakage > 25 % of the current (current-based value unreliable)")
    ax.axhline(1, color="k", lw=.6, ls=":")
    ax.set(xscale="log", xlabel="current (uA)", ylabel="local ideality  dV / (kT/q d ln X)",
           ylim=(0, 16), xlim=(0.02, 500),
           title="Ideality from the light and from the current\n"
                 "1 = an ideal junction; everything above it is extra voltage")
    ax.grid(alpha=.3, which="both")
    ax.legend(fontsize=7, loc="upper left")
    save(fig, "04_ideality.png", plt)


def fig_dvdi(mc, die, plt, drv):
    fig, ax = plt.subplots(figsize=(7.8, 5.8))
    ii = np.logspace(-8, -3, 50)
    for n in (1, 2, 4, 8, 16):
        ax.plot(ii * 1e6, n * VT / ii / 1e3, color="0.75", lw=.7)
        ax.annotate(f"n = {n}", (0.035, n * VT / 0.035e-6 / 1e3), fontsize=6, color="0.5")
    for arr in ARRAYS:
        k = mc[arr][mc[arr].ok & (mc[arr].I > 0.1e-6)].sort_values("I")
        i = k.I.to_numpy()
        ax.plot(mid(i) * 1e6, np.diff(k.V_LED) / np.diff(i) / 1e3, "-o", ms=4, color=COL[arr],
                label=f"{arr}: driver subtracted")
        e = k[k.I >= I_REF]
        ax.plot(mid(e.I.to_numpy()) * 1e6, np.diff(e.V_ex) / np.diff(e.I) / 1e3, ":s", ms=4,
                color=COL[arr], label=f"{arr}: excess only")
    k = die[die.Current_A > 0.05e-6].iloc[::3]
    i = k.Current_A.to_numpy()
    ax.plot(mid(i) * 1e6, np.diff(k.Voltage_V) / np.diff(i) / 1e3, "-", color="k", lw=1.4,
            label="die: whole")
    e = die[die.light_ok & (die.Current_A >= I_REF)].iloc[::3]
    ax.plot(mid(e.Current_A.to_numpy()) * 1e6, np.diff(e.V_ex) / np.diff(e.Current_A) / 1e3, ":",
            color="k", lw=1.4, label="die: excess only")
    ax.axhline(0.27, color="tab:purple", ls="--", lw=1, label="die's series R from its Cheung fit (270 Ohm)")
    ron = drv["Array42"]["vov"] / (drv["Array42"]["I15"] * (1 - 5 * drv["Array42"]["lam"])) / 1e3
    ax.axhline(ron, color="tab:gray", ls="--", lw=1, label=f"transistor on-resistance ({ron:.1f} kOhm)")
    ax.set(xscale="log", yscale="log", xlabel="current (uA)", ylabel="dV/dI (kOhm)",
           xlim=(0.03, 1000), ylim=(0.05, 3000),
           title="Differential resistance: a resistor would be flat\n"
                 "grey lines: an ideal diode n kT/(qI)")
    ax.grid(alpha=.3, which="both")
    ax.legend(fontsize=7, loc="upper right")
    save(fig, "05_differential_resistance.png", plt)


def fig_r_vs_v(arr, d, drv, plt):
    k = d[d.ok & (d.I > 0.1e-6)].sort_values("I")
    i, x = k.I.to_numpy(), mid(k.V_tot.to_numpy())
    fig, ax = plt.subplots(figsize=(7.8, 5.6))
    ax.plot(x, np.diff(k.V_tot) / np.diff(i) / 1e3, "-o", ms=4, color="k", label="whole pixel")
    ax.plot(x, np.diff(k.V_LED) / np.diff(i) / 1e3, "-s", ms=4, color=COL[arr],
            label="transistor subtracted (LED + bump + lines)")
    e = k.I.to_numpy() >= I_REF
    ax.plot(mid(k.V_tot.to_numpy()[e]), np.diff(k.V_ex.to_numpy()[e]) / np.diff(i[e]) / 1e3, "-^",
            ms=5, color="tab:red", label="excess over the junction (barrier + series R)")
    ron = drv[arr]["vov"] / (drv[arr]["I15"] * (1 - 5 * drv[arr]["lam"])) / 1e3
    ax.axhline(ron, color="tab:gray", ls="--", lw=1, label=f"transistor on-resistance ({ron:.1f} kOhm)")
    ax.axhline(0.27, color="tab:purple", ls=":", lw=1, label="die's own series R (270 Ohm)")
    sec = ax.secondary_xaxis("top", functions=(lambda v: br.VLED - v, lambda n: br.VLED - n))
    sec.set_xlabel("NVLED (V)")
    ax.set(xlabel="V_tot (V)", ylabel="dV/dI between neighbouring sweep points (kOhm)",
           yscale="log", ylim=(0.1, 300),
           title=f"{arr}: the 'series resistance' at each voltage\n"
                 "falling with voltage = not a resistor")
    ax.grid(alpha=.3, which="both")
    ax.legend(fontsize=7, loc="upper right")
    save(fig, f"06_resistance_vs_voltage_{arr}.png", plt)


def iqr_bars(ax, pp, prefix, color, scale=1e3, dx=1.0):
    for t in TARGETS:
        c = f"{prefix}_{tag(t)}"
        if pp is None or c not in pp:
            continue
        s = pp[c].dropna() * scale
        if len(s) < 50:
            continue
        q1, q2, q3 = s.quantile([.25, .5, .75])
        ax.errorbar(t * 1e6 * dx, q2, yerr=[[q2 - q1], [q3 - q2]], fmt="o", ms=7, mfc="w",
                    color=color, capsize=4, zorder=5)


def fig_minus_die(mc, die, pp, plt):
    fig, ax = plt.subplots(figsize=(7.8, 5.6))
    for arr, dx in zip(ARRAYS, (0.95, 1.05)):
        k = mc[arr][mc[arr].ok & (mc[arr].I > 0.3e-6)].sort_values("I")
        dv = k.V_LED.to_numpy() - interp_log(die.Current_A.to_numpy(), k.I.to_numpy(),
                                             die.Voltage_V.to_numpy())
        ax.plot(k.I * 1e6, 1e3 * dv, "-o", ms=4, color=COL[arr],
                label=f"{arr}: typical pixels' median curve")
        iqr_bars(ax, pp.get(arr), "V_LED_minus_die", COL[arr], dx=dx)
    ax.axhline(0, color="k", lw=.8)
    ax.set(xscale="log", xlabel="current (uA)",
           ylabel="V_LED(pixel) - V(die) at the same current (mV)",
           title="Pixel against the unbonded SGA_15 die\n"
                 "offset = the pixels' junction turns on earlier; rise = what the bond adds\n"
                 "(open circles: all swept pixels, median and 25-75 %)")
    ax.grid(alpha=.3, which="both")
    ax.legend(fontsize=8)
    save(fig, "07_pixel_minus_die.png", plt)


def fig_excess(mc, die, pp, plt):
    fig, ax = plt.subplots(figsize=(7.8, 5.6))
    e = die[die.light_ok & (die.Current_A >= I_REF)]
    ax.plot(e.Current_A * 1e6, 1e3 * e.V_ex, "-", color="k", lw=2, label=DIE_LABEL)
    for arr, dx in zip(ARRAYS, (0.95, 1.05)):
        i, v = excess_points(mc[arr], "I", mc[arr].ok)
        ax.plot(i * 1e6, 1e3 * v, "-o", ms=5, color=COL[arr], label=f"{arr}: typical pixels (bonded)")
        iqr_bars(ax, pp.get(arr), "V_ex", COL[arr], dx=dx)
    lines = ["current     die  Array42  Array38  (mV)"]
    for t in (1.5e-6, 5e-6, 20e-6, 40e-6, 60e-6):
        vals = [1e3 * interp_log(e.Current_A.to_numpy(), np.array([t]), e.V_ex.to_numpy())[0]]
        for arr in ARRAYS:
            i, v = excess_points(mc[arr], "I", mc[arr].ok)
            vals.append(1e3 * interp_log(i, np.array([t]), v)[0])
        lines.append(f"{t * 1e6:5.1f} uA " + "".join(f"{x:7.0f}  " if np.isfinite(x) else "      -  "
                                                     for x in vals))
    ax.text(0.03, 0.70, "\n".join(lines), transform=ax.transAxes, fontsize=7.5,
            family="monospace", va="top", bbox=dict(boxstyle="round", fc="w", alpha=.9))
    ax.set(xscale="log", xlabel="current (uA)", ylabel=f"excess voltage over the junction (mV), "
                                                       f"counted from {I_REF * 1e6:g} uA",
           xlim=(0.5, 500),
           title="The excess voltage is already in the unbonded die\n"
                 "junction from each device's own light; circles: all swept pixels (median, 25-75 %)")
    ax.grid(alpha=.3, which="both")
    ax.legend(fontsize=8, loc="upper left")
    save(fig, "08_excess_die_vs_pixels.png", plt)


def fig_fits(name, i, v, fits_, plt, i_max=None):
    fig, (ax, rx) = plt.subplots(2, 1, figsize=(7.8, 7), sharex=True,
                                 gridspec_kw=dict(height_ratios=[3, 1.3]))
    ax.plot(i * 1e6, 1e3 * v, "ko", ms=4, label="data")
    xx = np.logspace(np.log10(I_REF), np.log10(i.max()), 200)
    for (model, r), c in zip(fits_.items(), ("tab:gray", "tab:green", "tab:red")):
        ax.plot(xx * 1e6, 1e3 * r["f"](xx), "-", color=c, label=describe(model, r))
        rx.plot(i * 1e6, 1e3 * (v - r["f"](i)), "o-", ms=3, color=c)
    rx.axhline(0, color="k", lw=.6)
    ax.set(xscale="log", ylabel="excess voltage (mV)",
           title=f"{name}: what model describes the excess voltage?")
    rx.set(xlabel="current (uA)", ylabel="data - fit (mV)")
    for a in (ax, rx):
        a.grid(alpha=.3, which="both")
    ax.legend(fontsize=7, loc="upper left")
    save(fig, f"09_excess_fits_{name.split()[0].replace('SGA_15', 'die')}.png", plt)


def fig_window(rows, plt):
    fig, ax = plt.subplots(figsize=(7.8, 5.4))
    t = pd.DataFrame(rows)
    for name, c in (("SGA_15 die", "k"), ("Array42", COL["Array42"]), ("Array38", COL["Array38"])):
        s = t[t.curve == name]
        ax.errorbar(s.i_max_uA, s.R_kOhm, yerr=s.R_se_kOhm, fmt="o-", capsize=3, color=c, label=name)
    ax.axhline(0.27, color="tab:purple", ls="--", label="die's series R from its Cheung fit (270 Ohm)")
    ax.axhline(0, color="k", lw=.6)
    ax.set(xscale="log", xlabel=f"upper end of the fit window (uA); lower end {I_REF * 1e6:g} uA",
           ylabel="series R of the barrier + resistor fit (kOhm)",
           title="The fitted series R depends on how far the data reach\n"
                 "the die: still falling at 0.42 mA; the pixels stop at ~70 uA (their driver)")
    ax.grid(alpha=.3, which="both")
    ax.legend(fontsize=8)
    save(fig, "10_fit_window.png", plt)


def fig_cheung(mc, plt):
    lv, li = br.LV, br.LI                          # the die's IV sweep, to 0.84 mA
    m = li > 1e-6
    i, v = li[m], lv[m]
    h = mid(i) * np.diff(v) / np.diff(i)
    im = mid(i)
    fig, ax = plt.subplots(figsize=(7.8, 5.6))
    ax.plot(im * 1e3, h, "ko-", ms=4, label="die: I dV/dI")
    sel = (im > 0.35e-3) & (im < 0.9e-3)
    b, a = np.polyfit(im[sel], h[sel], 1)
    xx = np.linspace(0, 0.9e-3, 10)
    ax.plot(xx * 1e3, a + b * xx, "--", color="tab:purple",
            label=f"die, 0.35-0.9 mA: intercept {a:.3f} V (n = {a / VT:.1f}), slope {b:.0f} Ohm")
    for arr in ARRAYS:
        k = mc[arr][mc[arr].ok & (mc[arr].I > 0.5e-6)].sort_values("I")
        i2 = k.I.to_numpy()
        ax.plot(mid(i2) * 1e3, mid(i2) * np.diff(k.V_LED) / np.diff(i2), "o-", ms=4, color=COL[arr],
                label=f"{arr}: driver subtracted")
    ax.axvspan(0, 0.07, color="0.9", zorder=0, label="the pixels' range (driver-limited above)")
    ax.set(xlabel="current (mA)", ylabel="I dV/dI (V)", xlim=(0, 0.9), ylim=(0, 0.5),
           title="Cheung plot: I dV/dI = n kT/q + R I\n"
                 "the die's series R is the slope above 0.35 mA (its own analysis: 270 Ohm, n = 10.4)")
    ax.grid(alpha=.3)
    ax.legend(fontsize=7, loc="lower right")
    save(fig, "11_cheung.png", plt)
    return a, b


def fig_hist(pp, die, plt):
    fig, ax = plt.subplots(figsize=(7.8, 5.2))
    e = die[die.light_ok & (die.Current_A >= I_REF)]
    t = 20e-6
    dv = 1e3 * interp_log(e.Current_A.to_numpy(), np.array([t]), e.V_ex.to_numpy())[0]
    for arr in ARRAYS:
        s = 1e3 * pp[arr][f"V_ex_{tag(t)}"].dropna()
        ax.hist(s.clip(0, 600), bins=150, histtype="step", lw=1.5, color=COL[arr],
                label=f"{arr}: {len(s)} pixels, median {s.median():.0f} mV, "
                      f"25-75 % {s.quantile(.25):.0f}-{s.quantile(.75):.0f}")
    ax.axvline(dv, color="k", lw=2, label=f"SGA_15 die: {dv:.0f} mV")
    ax.set(xlabel=f"excess voltage at {t * 1e6:g} uA (mV), clipped to 0-600", ylabel="pixels",
           yscale="log", title="Excess voltage per swept pixel at 20 uA, against the die")
    ax.grid(alpha=.3)
    ax.legend(fontsize=8)
    save(fig, "12_excess_per_pixel.png", plt)


def fig_map(arr, p, plt, B=16):
    col = f"V_ex_{tag(20e-6)}"
    s = p.dropna(subset=[col])
    g = s.groupby([s.Y // B, s.X // B])[col]
    blk = g.median()[g.size() >= 5]
    img = np.full((512 // B, 512 // B), np.nan)
    img[blk.index.get_level_values(0), blk.index.get_level_values(1)] = 1e3 * blk.to_numpy()
    resid = 1e3 * (s[col] - s.set_index([s.Y // B, s.X // B]).index.map(blk).to_numpy())
    cm = plt.get_cmap("magma").copy()
    cm.set_bad("0.8")
    lo, hi = np.nanpercentile(img, [2, 98])
    fig, ax = plt.subplots(figsize=(7.2, 6.2))
    im = ax.imshow(img, cmap=cm, vmin=lo, vmax=hi, extent=(-.5, 511.5, 511.5, -.5),
                   interpolation="nearest")
    fig.colorbar(im, ax=ax, label=f"median excess at 20 uA in {B}x{B} blocks (mV)", fraction=.046)
    bq = np.nanpercentile(img, [10, 90])
    rq = np.nanpercentile(resid, [25, 75, 10, 90])
    ax.set(xlabel="X", ylabel="Y",
           title=f"{arr}: excess voltage at 20 uA, {len(s)} swept pixels (grey: none)\n"
                 f"blocks 10-90 % {bq[0]:.0f}-{bq[1]:.0f} mV; pixel minus its block: "
                 f"25-75 % {rq[0]:+.0f} to {rq[1]:+.0f}, 10-90 % {rq[2]:+.0f} to {rq[3]:+.0f} mV")
    ax.title.set_fontsize(9)
    save(fig, f"13_excess_map_{arr}.png", plt)


def fig_vs_rbest(pp, plt):
    fig, ax = plt.subplots(figsize=(7.8, 5.4))
    col = f"V_ex_{tag(20e-6)}"
    for arr in ARRAYS:
        s = pp[arr].dropna(subset=[col, "R_best_kOhm"])
        q = pd.qcut(s.R_best_kOhm, 20, duplicates="drop")
        g = s.groupby(q, observed=True)
        x = g.R_best_kOhm.median()
        y = 1e3 * g[col].median()
        lo, hi = 1e3 * g[col].quantile(.25), 1e3 * g[col].quantile(.75)
        r = s.R_best_kOhm.corr(s[col], method="spearman")
        ax.errorbar(x, y, yerr=[y - lo, hi - y], fmt="o-", ms=4, capsize=2, color=COL[arr],
                    label=f"{arr}: {len(s)} pixels, Spearman {r:.2f}")
    ax.set(xscale="symlog", xlabel="R_best: series resistance above the array's best pixels, "
                                   "fast scan at -1.40 V (kOhm, 20 bins)",
           ylabel="excess voltage at 20 uA (mV; median, 25-75 %)",
           title="Does the excess voltage follow the per-pixel resistance R_best?")
    ax.grid(alpha=.3, which="both")
    ax.legend(fontsize=8)
    save(fig, "14_excess_vs_Rbest.png", plt)


# ---- main ---------------------------------------------------------------------------------
def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    plt = aa._plt()
    drv = {a: br.fit_driver(a) for a in ARRAYS}
    die = load_die()
    print(f"die: photodiode floor {die.attrs['floor']:.2e} A, light usable from "
          f"{die.Current_A[die.light_ok].min() * 1e6:.2f} uA")
    mc = {a: median_curve(a, drv[a]) for a in ARRAYS}
    rows = []
    for a, d in mc.items():
        rows.append(d.assign(curve=a))
    pd.concat(rows + [die.assign(curve="SGA_15 die")], ignore_index=True).to_csv(
        OUT / "median_curves.csv", index=False, float_format="%.6g")
    print("fast-scan medians ...", flush=True)
    fsm = pd.DataFrame([fs_median(a, r, v) for a in ARRAYS for r, v in FASTSCANS[a]])
    fsm.to_csv(OUT / "fastscan_medians.csv", index=False, float_format="%.6g")
    print(fsm.to_string(index=False))
    print("per-pixel ...", flush=True)
    pp = {}
    for a in ARRAYS:
        pp[a] = per_pixel(a, drv[a], die)
        pp[a].to_csv(OUT / f"per_pixel_{a}.csv", index=False, float_format="%.5g")
        print(f"  {a}: {len(pp[a])} pixels, V_ex(20 uA) median "
              f"{1e3 * pp[a][f'V_ex_{tag(20e-6)}'].median():.0f} mV", flush=True)

    # fits: every curve, every model; and the window test
    fits_, frows, wrows = {}, [], []
    e = die[die.light_ok & (die.Current_A >= I_REF)]
    curves = {"SGA_15 die": (e.Current_A.to_numpy(), e.V_ex.to_numpy())}
    for a in ARRAYS:
        curves[a] = excess_points(mc[a], "I", mc[a].ok)
    for name, (i, v) in curves.items():
        for model in MODELS:
            r = fit(i, v, model)
            fits_[(name, model)] = r
            frows.append(dict(curve=name, model=model, n=r["n"], rms_mV=1e3 * r["rms"],
                              **{f"p{k}": x for k, x in enumerate(r["p"])},
                              **{f"se{k}": x for k, x in enumerate(r["se"])}))
            print(f"  {name}: {describe(model, r)}")
        for top in (20e-6, 40e-6, 70e-6, 120e-6, 200e-6, 300e-6, 420e-6):
            m = i <= top * 1.02
            if m.sum() >= 5 and i[m].max() >= 0.8 * top:
                r = fit(i[m], v[m], "barrier + resistor")
                wrows.append(dict(curve=name, i_max_uA=top * 1e6, R_kOhm=r["p"][2] / 1e3,
                                  R_se_kOhm=r["se"][2] / 1e3, a_mV=1e3 * r["p"][0], n=int(m.sum())))
    pd.DataFrame(frows).to_csv(OUT / "fits.csv", index=False, float_format="%.5g")
    pd.DataFrame(wrows).to_csv(OUT / "fit_window.csv", index=False, float_format="%.4g")
    print(pd.DataFrame(wrows).round(3).to_string(index=False))

    fig_driver(drv, mc, plt)
    fig_iv(mc, die, fsm, plt, log=True)
    fig_iv(mc, die, fsm, plt, log=False)
    for a in ARRAYS:
        fig_budget(a, mc[a], fits_, plt)
        fig_r_vs_v(a, mc[a], drv, plt)
        fig_map(a, pp[a], plt)
    fig_ideality(mc, die, plt)
    fig_dvdi(mc, die, plt, drv)
    fig_minus_die(mc, die, pp, plt)
    fig_excess(mc, die, pp, plt)
    for name, (i, v) in curves.items():
        fig_fits(name, i, v, {m: fits_[(name, m)] for m in MODELS}, plt)
    fig_window(wrows, plt)
    a, b = fig_cheung(mc, plt)
    print(f"die Cheung fit 0.35-0.9 mA: intercept {a:.3f} V (n {a / VT:.1f}), slope {b:.0f} Ohm")
    fig_hist(pp, die, plt)
    fig_vs_rbest(pp, plt)
    for a in ARRAYS:
        d = mc[a]
        print(f"\n{a} median curve:")
        print(d[["V_tot", "I", "leak", "V_drv", "V_LED", "V_j", "V_ex", "ok"]].assign(
            I=lambda t: t.I * 1e6, leak=lambda t: t.leak * 1e6).round(4).to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
