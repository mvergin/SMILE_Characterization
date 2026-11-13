"""Where a pixel's voltage goes, and what the LED part looks like: plots for discussion.

    python analysis/led_voltage_budget.py

Data: the NVLED sweeps, every swept pixel (Array42 261102-1703, Array38 261026-0813; NVLED 0 to
-3.2 V, with a leakage reference from -0.7 / -0.5 V on), and the SGA_15 10 um die (its own IVL
measurement of 2023-08-18: LED current and photodiode current, 2.20-3.60 V in 10 mV steps;
`SGA_15 10um/SGA_15_10um_IVL.csv`, converted from the .xls, kept local). The fast scans enter
only as validation markers (01, 02) and through R_best (14).

A pixel is transistor + LED (+ bump and interconnect) in series across V_tot = VLED - NVLED.
The transistor's drop comes from its model (bond_resistance.fit_driver), so
V_LED = V_tot - V_driver(I) is everything else. The LED's own light splits V_LED in two:

    V_j  = V_LED(I_REF) + N_R kT/q ln(P / P(I_REF))     junction: emission ~ exp(q V_j / N_R kT)
    V_ex = V_LED - V_j                                     excess: barriers and series resistance

N_R = 1.07 is the optical ideality the pixels show at nA, where nothing else drops a voltage.
The excess is counted from I_REF = 0.6 uA, where the die's light clears its photodiode floor.

Where the transistor limits the current (I > 70 % of what it can carry), its model no longer
gives its drop: a 1 % current error there is worth 0.1-0.8 V. There the LED's voltage is its
junction (from its light) plus the fitted excess, and the transistor takes the rest.

Pixels: every swept pixel, without two artefacts. A leakage reference above 1.3 x its bias's
median was taken with the pixel still partly lit (that point is dropped); a pixel whose sweep
current at -1.4 V is above 1.05 x the same run's fast scan had a second driver latched on (the
pixel is dropped). "Best 2 %": R_best's reference pixels (bond_resistance.best_reference).

Writes results/comparisons/led_voltage/: the figures (NN_*.png, 300 dpi), curves.csv,
grid_stats.csv, per_pixel_<array>.csv, fits.csv, fit_window.csv, fastscan_medians.csv.
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import least_squares

import analyze_array as aa
import bond_resistance as br
import pixel_classes as pc

OUT = br.REPO / "results" / "comparisons" / "led_voltage"
DIE_IVL = br.REPO / "SGA_15 10um" / "SGA_15_10um_IVL.csv"
SHARE = Path(r"X:\public\Schoettler\von Max\SMILE_IQE_Results\runs\results")
VT = 0.025852                # kT/q at 300 K
N_R = 1.07                   # optical ideality at nA (Array42 1.06-1.08, Array38 1.07)
I_REF = 0.6e-6               # the excess is counted from here
CAP_FRAC = 0.7               # the transistor's model drop is trusted below 70 % of its capacity
REF_TOL = 1.3                # leakage reference within 1.3 x its bias's median
LATCH = 1.05                 # sweep / fast-scan current at -1.4 V above this: a latched driver
TARGETS = (1.5e-6, 5e-6, 20e-6)
GRID = np.geomspace(0.8e-6, 100e-6, 19)
DPI = 300
ARRAYS = ("Array42", "Array38")
# NPG (ggsci) palette: Publications/ColorPalettes/NPG.json
NPG = dict(red="#E64B35", cyan="#4DBBD5", teal="#00A087", navy="#3C5488", salmon="#F39B7F",
           lilac="#8491B4", mint="#91D1C2", darkred="#DC0000", brown="#7E6148", tan="#B09C85")
WHOLE = {"Array42": NPG["navy"], "Array38": NPG["red"]}       # whole pixel; an array's own colour
SUB = {"Array42": NPG["cyan"], "Array38": NPG["salmon"]}       # transistor subtracted
DIE = "#000000"
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
# Leakage tests: a few pixels lit and off (own quadrant enabled) at NVLED 0, -0.2, -0.4 V,
# where the sweeps took no leakage reference (leak_ref_skip_above_v -0.7 / -0.45)
LEAKTEST = {"Array42": "261102-1240_Array42_CAPACITANCE_Leakmatrix",
            "Array38": "261025-1950_Array38_CAPACITANCE_Leakmatrix"}
SOURCE = {}                  # per array: the data line printed under its figures


def leak_test(arr):
    """Per bias: the LED's current as lit minus off of the same pixel (mean over the tested
    pixels, standard error from their spread), and the leakage itself (the pixel off)."""
    d = pd.read_csv(run_path(LEAKTEST[arr]) / "leakmatrix.csv")
    p = d.pivot_table(index=["v", "x", "y"], columns="state", values="NVLED_CURR_MEAN",
                      aggfunc="first").abs()
    g = (p.lit_own - p.off_own).groupby(level="v")
    return pd.DataFrame({"V_tot": br.VLED - g.mean().index.to_numpy(), "I_led": g.mean().to_numpy(),
                         "se": (g.std() / np.sqrt(g.count())).to_numpy(), "n": g.count().to_numpy(),
                         "I_off": p.off_own.groupby(level="v").mean().to_numpy()})


def tag(i):
    return f"{i * 1e6:g}uA".replace(".", "p")


def run_path(run):
    try:
        return br.run_dir(run)
    except FileNotFoundError:
        return SHARE / run


def capacity(vtot, P):
    """What the transistor can carry with all of V_tot across it (no gate term: PMOS)."""
    return br.f15(np.asarray(vtot, float), 0.0, P)


def interp_log(i, x, y):
    """y at currents x, interpolated in ln I over (i, y), i ascending; NaN outside."""
    return np.interp(np.log(x), np.log(i), y, left=np.nan, right=np.nan)


def mid(a):
    return np.sqrt(a[1:] * a[:-1])


# ---- data -----------------------------------------------------------------------------------
def light_split(i, v, light, ok):
    """Junction and excess voltage, the excess counted from I_REF (log-interpolated over the
    points in ok, i ascending). Returns (V_j, V_ex, (V_LED, ln P) at I_REF)."""
    m = ok & np.isfinite(v) & (light > 0) & (i > 0)
    v0 = np.interp(np.log(I_REF), np.log(i[m]), v[m])
    l0 = np.interp(np.log(I_REF), np.log(i[m]), np.log(light[m]))
    vj = v0 + N_R * VT * (np.log(np.where(light > 0, light, np.nan)) - l0)
    return vj, v - vj, (v0, l0)


def load_die():
    d = pd.read_csv(DIE_IVL).sort_values("Voltage_V")
    d = d[d.Current_A > 0].reset_index(drop=True)
    d["L"] = -d.Photodiode_A                       # the photodiode reads negative
    floor = float(d.L[d.Voltage_V < 2.5].std())
    d["light"] = d.L > 20 * floor
    i, v, light = d.Current_A.to_numpy(), d.Voltage_V.to_numpy(), d.L.to_numpy()
    vj, vex, _ = light_split(i, v, light, d.light.to_numpy() & (i >= 0.3e-6))
    d["V_j"] = np.where(d.light, vj, np.nan)
    d["V_ex"] = np.where(d.light, vex, np.nan)
    d.attrs["floor"] = floor
    return d


def load_sweep_points(arr, P, rb):
    """Every swept pixel and bias, without the two artefacts, transistor subtracted. Returns
    the points with a leakage reference, and the median total current at every bias."""
    sw = pc.load_sweep(br.REPO / "results" / br.SWEEP[arr])
    s14 = sw[sw.V.round(3) == -1.4].set_index(["X", "Y"]).I_LED
    ratio = s14 / rb.I_LED.reindex(s14.index)
    latched = ratio.index[ratio > LATCH]
    sw = sw[~pd.MultiIndex.from_arrays([sw.X, sw.Y]).isin(latched)]
    tot = sw.groupby("V").I_tot.median()                    # every bias, reference or not
    ref_med = sw.groupby("V").I_ref.transform("median")
    n_lit_ref = int((sw.I_ref > REF_TOL * ref_med).sum())
    sw = sw[sw.I_ref.notna() & (sw.I_ref <= REF_TOL * ref_med)].copy()
    sw["V_tot"] = br.VLED - sw.V
    sw["ok"] = sw.I_LED <= CAP_FRAC * capacity(sw.V_tot.to_numpy(), P)
    sw["V_LED"] = sw.V_tot - br.v_driver(sw.I_LED.clip(lower=1e-13).to_numpy(), 0.0, P)
    sw["light"] = sw.P > 5 * sw.P_noise
    sw.attrs.update(n_latched=len(latched), n_lit_ref=n_lit_ref,
                    n_pixels=int(pd.MultiIndex.from_arrays([sw.X, sw.Y]).nunique()))
    return sw, tot


def curve(sw, P, idx=None):
    """The median pixel at every sweep bias (of all pixels, or of those in idx)."""
    s = sw if idx is None else sw[pd.MultiIndex.from_arrays([sw.X, sw.Y]).isin(idx)]
    g = s.groupby("V")
    d = pd.DataFrame({"n": g.size(), "I": g.I_LED.median(), "I_q1": g.I_LED.quantile(.25),
                      "I_q3": g.I_LED.quantile(.75), "I_tot": g.I_tot.median(),
                      "P": g.P.median(), "P_noise": g.P_noise.median()}).reset_index()
    d["V_tot"] = br.VLED - d.V
    d = d.sort_values("V_tot").reset_index(drop=True)
    d["ok"] = d.I <= CAP_FRAC * capacity(d.V_tot, P)
    d["V_drv"] = br.v_driver(d.I.clip(lower=1e-13).to_numpy(), 0.0, P)
    d["V_LED"] = d.V_tot - d.V_drv
    d["light"] = (d.P > 5 * d.P_noise) & (d.I > 0)
    vj, vex, ref = light_split(d.I.to_numpy(), d.V_LED.to_numpy(), d.P.to_numpy(),
                               (d.ok & d.light).to_numpy())
    d["V_j"] = np.where(d.light, vj, np.nan)
    d["V_ex"] = np.where(d.light & d.ok, vex, np.nan)
    d.attrs["ref"] = ref
    return d


def led_model(d, r):
    """V_LED(I): the junction from the light (ln P interpolated in ln I) plus the fitted excess."""
    m = d.light.to_numpy()
    o = np.argsort(d.I[m].to_numpy())
    li, lp = np.log(d.I[m].to_numpy()[o]), np.log(d.P[m].to_numpy()[o])
    v0, l0 = d.attrs["ref"]
    return lambda i: v0 + N_R * VT * (np.interp(np.log(i), li, lp) - l0) + r["f"](i)


def add_estimate(d, r):
    """Where the transistor limits the current: LED = junction (light) + fitted excess."""
    est = d.V_j.to_numpy() + r["f"](d.I.clip(lower=1e-13).to_numpy())
    d["V_LED_est"] = np.where(d.ok, d.V_LED, est)
    d["V_drv_est"] = d.V_tot - d.V_LED_est
    return d


def per_pixel(sw, die, best, rb, i_star, D):
    """Every pixel: V_LED and light at I_REF, TARGETS, GRID and i_star (the fast scan's median
    current, for the comparison with R_best), log-interpolated between its own sweep points;
    its excess there, and its V_LED against the die's at the same current.
    Returns the per-pixel table and the distribution over pixels at each GRID current."""
    s = sw[sw.ok & (sw.I_LED > 0)]
    s = s.assign(P=s.P.where(s.light))
    piv = s.pivot_table(index=["X", "Y"], columns="V_tot", values=["I_LED", "V_LED", "P"])
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
    di, dv = die.Current_A.to_numpy(), die.Voltage_V.to_numpy()
    isbest = piv.index.isin(best)

    def quant(a, q):
        return float(np.nanquantile(a, q)) if np.isfinite(a).sum() >= 20 else np.nan

    res = pd.DataFrame(index=piv.index)
    for t in TARGETS:
        v, lp = at(t)
        res[f"V_ex_{tag(t)}"] = (v - v0) - N_R * VT * (lp - l0)
        res[f"V_LED_minus_die_{tag(t)}"] = v - interp_log(di, np.array([t]), dv)[0]
    v, lp = at(i_star)
    res["V_ex_Istar"] = (v - v0) - N_R * VT * (lp - l0)
    # per pixel, over its own points from I_REF to the transistor limit: the die-anchored fit
    # V_ex = k D(I) + R (I - I_REF). Linear in k and R, so solved exactly (normal equations).
    # k: the barrier against the die's (1 = the same); R: the extra ohmic part
    ii = np.exp(lni)
    vex = (vl - v0[:, None]) - N_R * VT * (lnp - l0[:, None])
    m = np.isfinite(vex) & np.isfinite(ii) & (ii >= I_REF)
    dd_ = np.where(m, D(np.where(m, ii, I_REF)), 0.0)
    xx = np.where(m, ii - I_REF, 0.0)
    yy = np.where(m, vex, 0.0)
    sdd, sdx, sxx = (dd_ * dd_).sum(1), (dd_ * xx).sum(1), (xx * xx).sum(1)
    sdy, sxy = (dd_ * yy).sum(1), (xx * yy).sum(1)
    det = sdd * sxx - sdx ** 2
    nfit = m.sum(1)
    good = (nfit >= 4) & (det > 0)
    with np.errstate(all="ignore"):
        k = np.where(good, (sdy * sxx - sxy * sdx) / det, np.nan)
        rr = np.where(good, (sdd * sxy - sdx * sdy) / det, np.nan)
        fit_res = np.where(m, yy - (k[:, None] * dd_ + rr[:, None] * xx), np.nan)
        rms = np.sqrt(np.nanmean(fit_res ** 2, axis=1))
    res["k_barrier"] = k
    res["R_extra_kOhm"] = rr / 1e3
    res["fit_rms_mV"] = 1e3 * rms
    res["n_fit"] = nfit
    rows_ = []
    for t in GRID:
        v, lp = at(t)
        ex = (v - v0) - N_R * VT * (lp - l0)
        dd = v - interp_log(di, np.array([t]), dv)[0]
        rows_.append(dict(I=t, n=int(np.isfinite(ex).sum()), n_best=int(np.isfinite(ex[isbest]).sum()),
                          ex_q1=quant(ex, .25), ex_med=quant(ex, .5), ex_q3=quant(ex, .75),
                          ex_best=quant(ex[isbest], .5), dd_q1=quant(dd, .25),
                          dd_med=quant(dd, .5), dd_q3=quant(dd, .75), dd_best=quant(dd[isbest], .5)))
    res = res.join(rb[["R_best_kOhm", "quadrant"]], how="left")
    res["best"] = isbest
    return res.reset_index(), pd.DataFrame(rows_)


def fs_median(arr, run, V):
    """Median current and light of a fast scan's typical pixels at bias V, leakage removed
    per quadrant (the median current of its open pixels)."""
    fs = aa._load_aggregated(run_path(run) / "FastScan")
    fs = fs[fs.NVLED_V.astype(float).round(2) == round(V, 2)]
    P = fs.PM400_POWER_MEAN.astype(float).abs()
    I = fs.NVLED_CURR_MEAN.astype(float).abs()
    p90 = P.quantile(0.9)
    alive = P > 0.1 * p90
    # open pixels: dark and drawing almost nothing. Shorts are dark too, but carry the
    # transistor's full current (on Array42 they outnumber the opens)
    opn = (P < 0.01 * p90) & (I < 0.05 * I[alive].median())
    q = fs.QUADRANT.astype(str)
    leak = I[opn].groupby(q[opn]).median()
    i_led = I - q.map(leak).fillna(float(I[opn].median())).to_numpy()
    mi, mp = i_led[alive].median(), P[alive].median()
    typ = alive & ((i_led / mi - 1).abs() < 0.15) & ((P / mp - 1).abs() < 0.5)
    return dict(array=arr, run=run[:11], V=V, V_tot=br.VLED - V, I=float(i_led[typ].median()),
                P=float(P[typ].median()), n=int(typ.sum()), n_open=int(opn.sum()),
                leak=float(I[opn].median()))


# ---- fits -------------------------------------------------------------------------------------
MODELS = {
    "resistor only": (lambda p, i: (i - I_REF) * p[0], [3e3], ([-1e5], [1e6])),
    "barrier only": (lambda p, i: p[0] * (np.log1p(i / 10 ** p[1]) - np.log1p(I_REF / 10 ** p[1])),
                     [0.1, -5.5], ([0, -9], [2, -2])),
    "barrier + resistor": (lambda p, i: p[0] * (np.log1p(i / 10 ** p[1]) - np.log1p(I_REF / 10 ** p[1]))
                           + (i - I_REF) * p[2], [0.1, -5.5, 500], ([0, -9, -2e4], [2, -2, 1e5])),
}
FIT_COL = {"resistor only": NPG["tan"], "barrier only": NPG["lilac"], "barrier + resistor": NPG["teal"]}


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


def die_excess(die):
    """The die's excess as measured (no fit), interpolated in ln I: its LED barrier and its
    own resistance, fixed for the bond fits."""
    e = die[die.light & (die.Current_A >= 0.3e-6)].sort_values("Current_A")
    li, ve = np.log(e.Current_A.to_numpy()), e.V_ex.to_numpy()
    return lambda i: np.interp(np.log(i), li, ve)


def _barrier(a, lb, i):
    return a * (np.log1p(i / 10 ** lb) - np.log1p(I_REF / 10 ** lb))


# the pixel as the die's LED plus a bond term; D is the die's measured excess
BOND = {
    "die + R": (lambda D, p, i: D(i) + p[0] * (i - I_REF), [500.], ([-2e4], [5e4])),
    "die + 2nd barrier": (lambda D, p, i: D(i) + _barrier(p[0], p[1], i), [0.05, -5.],
                          ([0, -8], [1, -3])),
    "k x die + R": (lambda D, p, i: p[0] * D(i) + p[1] * (i - I_REF), [1., 500.],
                    ([0.2, -2e4], [5, 5e4])),
}
BOND_COL = {"die + R": NPG["tan"], "die + 2nd barrier": NPG["lilac"], "k x die + R": NPG["teal"]}
BOND_TEXT = {"die + R": "the die's LED + a resistor (bond)",
             "die + 2nd barrier": "the die's LED + a second barrier (oxide, Schottky)",
             "k x die + R": "the die's barrier scaled by k + a resistor"}


def bond_fit(D, i, v, model):
    f, p0, bounds = BOND[model]
    r = least_squares(lambda p: f(D, p, i) - v, p0, bounds=bounds)
    cov = np.linalg.pinv(r.jac.T @ r.jac) * (r.fun @ r.fun) / max(1, len(i) - len(p0))
    return dict(p=r.x, se=np.sqrt(np.abs(np.diag(cov))), rms=float(np.sqrt(np.mean(r.fun ** 2))),
                f=lambda x, p=r.x: f(D, p, x), n=len(i))


def describe_bond(model, r):
    p, se = r["p"], r["se"]
    if model == "die + R":
        s = f"R {p[0] / 1e3:+.2f} +- {se[0] / 1e3:.2f} kOhm"
    elif model == "die + 2nd barrier":
        s = f"a {1e3 * p[0]:.0f} mV, I_b {10 ** p[1] * 1e6:.2g} uA"
    else:
        s = f"k {p[0]:.2f} +- {se[0]:.2f}, R {p[1] / 1e3:+.2f} +- {se[1] / 1e3:.2f} kOhm"
    return f"{model}: {s}, rms {1e3 * r['rms']:.1f} mV"


def excess_points(d):
    m = d.ok & d.light & np.isfinite(d.V_ex) & (d.I >= I_REF)
    return d.I[m].to_numpy(), d.V_ex[m].to_numpy()


# ---- figures ----------------------------------------------------------------------------------
def footer(fig, text):
    fig.text(0.01, 0.005, text, fontsize=6, color="0.35", va="bottom", ha="left", wrap=True)


def save(fig, name, plt, text):
    fig.tight_layout(rect=(0, 0.035, 1, 1))
    footer(fig, text)
    fig.savefig(OUT / name, dpi=DPI)
    plt.close(fig)


def die_source(die):
    return (f"SGA_15 10 um die: IVL of 2023-08-18, light from {die.Current_A[die.light].min() * 1e6:.2f} uA "
            f"(photodiode floor {die.attrs['floor'] * 1e12:.0f} pA).")


def fig_driver(arr, P, d, model, plt):
    fig, ax = plt.subplots(figsize=(7.8, 5.8))
    vv = np.linspace(0, 5, 501)
    vt, i = P["data"]
    ax.plot(vt - i * br.R_SHORT, i * 1e6, "o", ms=2.5, color="0.6",
            label=f"{arr}'s 10 best shorts, their own path removed")
    ax.plot(vv, br.f15(vv, 0.0, P) * 1e6, "-", color="k", lw=1.8,
            label="transistor model, all four on (code 15)"
            + (f": Array42's shape, current x {P['scale']:.3f}" if "shape_from" in P else ""))
    ii = np.linspace(1e-6, 124e-6, 400)
    for vt_, c in zip((3.2, 4.0, 5.0), (NPG["teal"], NPG["lilac"], NPG["brown"])):
        ax.plot(vt_ - model(ii), ii * 1e6, "--", color=c, lw=1.3,
                label=f"what the LED leaves the transistor at NVLED {br.VLED - vt_:+.1f} V")
    k = d[d.I > 0.5e-6]
    ax.plot(k.V_drv_est, k.I * 1e6, "o", ms=6, mfc="w", mew=1.5, color=WHOLE[arr],
            label="operating point of the median pixel at each sweep bias")
    for _, r in k.iterrows():
        if any(abs(r.V_tot - x) < 0.01 for x in (3.2, 4.0, 5.0)):
            ax.annotate(f"{br.VLED - r.V_tot:+.1f} V", (r.V_drv_est, r.I * 1e6), xytext=(4, -11),
                        textcoords="offset points", fontsize=7, color=WHOLE[arr])
    # where the transistor limits the current: the model inversion (the lowest voltage at
    # which the model carries that current) against the operating point from the light
    s = d[~d.ok & np.isfinite(d.V_drv) & np.isfinite(d.V_drv_est)].sort_values("I")
    if len(s):
        ax.plot(s.V_drv, s.I * 1e6, "s", ms=5.5, color=NPG["red"],
                label="model inversion (lowest voltage at which the model carries the current): "
                + ", ".join(f"{v:.2f}" for v in s.V_drv) + " V")
        for _, r in s.iterrows():
            ax.plot([r.V_drv, r.V_drv_est], [r.I * 1e6] * 2, ":", color=NPG["red"], lw=1.2)
        r = s.iloc[-1]
        ax.text(2.0, r.I * 1e6 - 1.0,
                f"model inversion {r.V_drv:.2f} V, from the LED's light {r.V_drv_est:.2f} V",
                ha="left", va="center", fontsize=7, color=NPG["red"])
    ron = P["vov"] / (P["I15"] * (1 - 5 * P["lam"]))
    rout = 1 / (P["lam"] * P["I15"])
    ax.annotate(f"linear region:\nacts like a\n{ron / 1e3:.1f} kOhm resistor", (0.22, 45), xytext=(0.6, 3),
                fontsize=8, bbox=dict(boxstyle="round", fc="w", ec="0.7"),
                arrowprops=dict(arrowstyle="->", color="0.4"))
    ax.annotate(f"saturation: a current source.\nOnly {100 * P['lam']:.1f} %/V more current\n"
                f"(output resistance {rout / 1e6:.2f} MOhm),\nso any further voltage stays here",
                (2.25, 97), fontsize=8, va="bottom", bbox=dict(boxstyle="round", fc="w", ec="0.7"))
    fit_note = ([f"$V_{{ov}}$, m, $\\lambda$: Array42's (no data below the knee)",
                 f"$I_{{15}}$: Array42's x {P['scale']:.3f}, fitted to the shorts, rms {P['rms_pct']:.2f} %"]
                if "shape_from" in P else [f"fitted to the 10 best shorts: rms {P['rms_pct']:.2f} %"])
    ax.text(2.25, 36, "\n".join(
        [r"$I = I_{15}\,\left(1+\lambda\,(V_{ds}-5\,\mathrm{V})\right)\,\frac{x}{\left(1+x^{m}\right)^{1/m}},"
         r"\quad x = \frac{V_{ds}}{V_{ov}}$",
         f"$I_{{15}}$ = {P['I15'] * 1e6:.1f} uA,   $\\lambda$ = {100 * P['lam']:.2f} %/V",
         f"$V_{{ov}}$ = {P['vov']:.3f} V,   m = {P['m']:.2f}"] + fit_note
        + [f"on-resistance $V_{{ov}}\\,/\\,(I_{{15}}(1-5\\,\\mathrm{{V}}\\,\\lambda))$ = {ron / 1e3:.1f} kOhm,",
           f"output resistance $1\\,/\\,(\\lambda I_{{15}})$ = {rout / 1e6:.2f} MOhm"]),
        fontsize=8, va="bottom", ha="left", linespacing=1.5,
        bbox=dict(boxstyle="round", fc="w", ec="0.7"))
    ax.set(xlabel="voltage across the transistor (V)", ylabel="current (uA)", xlim=(0, 5),
           ylim=(0, 150), title=f"{arr}: the pixel transistor, and where the pixel operates on it\n"
                                "operating point = where the transistor's curve meets what the LED leaves it")
    ax.grid(alpha=.3)
    ax.legend(fontsize=6.5, loc="lower right")
    save(fig, f"00_transistor_{arr}.png", plt,
         f"Data: transistor IV of {arr}'s shorts ({br.DRIVER[arr]}); " + SOURCE[arr] +
         " Above ~100 uA the LED's voltage is its junction (from its light) plus the fitted excess.")


def fig_subtraction(arr, d, P, plt):
    """How the transistor is subtracted: at every sweep point of the median pixel an arrow
    from V_tot to V_tot - V_transistor(I), its length the transistor's voltage."""
    k = d[(d.I > 1e-6) & np.isfinite(d.V_LED_est)].sort_values("V_tot")
    ok = k.ok.to_numpy()
    fig, ax = plt.subplots(figsize=(8.4, 6.2))
    ax.plot(k.V_tot, k.I * 1e6, "-o", ms=6, color=WHOLE[arr], label="whole pixel: V_tot (median pixel)")
    j = int(np.flatnonzero(ok)[-1])
    ax.plot(k.V_LED_est[:j + 1], k.I[:j + 1] * 1e6, "-s", ms=6, color=SUB[arr],
            label="transistor subtracted: V_tot - V_transistor(I), model (00)")
    ax.plot(k.V_LED_est[j:], k.I[j:] * 1e6, ":s", ms=6, mfc="w", color=SUB[arr],
            label="where the transistor limits the current: LED = junction (light) + fitted excess,\n"
                  "transistor = V_tot - LED")
    for _, r in k.iterrows():
        ax.annotate("", xy=(r.V_LED_est, r.I * 1e6), xytext=(r.V_tot, r.I * 1e6),
                    arrowprops=dict(arrowstyle="->", color=NPG["lilac"], lw=1.6, shrinkA=4, shrinkB=4))
        if r.V_drv_est > 0.06:
            on = r.V_drv_est > 0.16                       # long arrow: the label sits on it
            ax.text((r.V_tot + r.V_LED_est) / 2, r.I * 1e6 + (0 if on else 2.2),
                    f"{1e3 * r.V_drv_est:.0f} mV", ha="center", va="center" if on else "bottom",
                    fontsize=7.5, color=NPG["navy"], zorder=5,
                    bbox=dict(boxstyle="round,pad=0.15", fc="w", ec="none", alpha=.9))
    ron = P["vov"] / (P["I15"] * (1 - 5 * P["lam"]))
    top = k.iloc[-1]
    ax.text(0.585, 0.66,
            "arrows: the transistor's voltage\n"
            "at that current\n\n"
            f"below ~100 uA: like a {ron / 1e3:.1f} kOhm\n"
            "resistor, the arrows grow with I\n\n"
            "above ~110 uA: a current source.\n"
            f"The current stops near {top.I * 1e6:.0f} uA, and\n"
            "every further volt of V_tot goes\n"
            f"into it ({top.V_drv_est:.2f} V of {top.V_tot:.1f} V at {br.VLED - top.V_tot:+.1f} V)\n\n"
            "The subtracted curve is the LED's\n"
            "own I-V: it ends where the current\n"
            f"ends, {top.V_LED_est:.2f} V at {top.I * 1e6:.0f} uA",
            transform=ax.transAxes, fontsize=7, family="monospace", va="top",
            bbox=dict(boxstyle="round", fc="w", alpha=.92))
    ax.set(xlabel="voltage (V)", ylabel="current (uA)", xlim=(2.55, 5.1), ylim=(0, 135),
           title=f"{arr}: how the transistor is subtracted")
    ax.grid(alpha=.3)
    ax.legend(fontsize=7, loc="lower right")
    save(fig, f"00b_subtraction_{arr}.png", plt,
         "Data: " + SOURCE[arr] + " Transistor model as in 00.")


def fig_light_inversion(arr, d, plt):
    """How the junction voltage is read off the light: log P against the LED's voltage, the
    ideal-junction line through the 0.6 uA point, and the excess as the horizontal gap."""
    k = d[d.light & (d.I > 0)].sort_values("I")
    v0, l0 = d.attrs["ref"]
    fig, ax = plt.subplots(figsize=(8.2, 6.2))
    pp = np.geomspace(k.P.min() * 0.5, k.P.max() * 2, 200)
    ax.plot(v0 + N_R * VT * (np.log(pp) - l0), pp * 1e9, "-", color=NPG["teal"], lw=2,
            label=f"ideal junction: P ~ exp(q V / {N_R} kT), {1e3 * N_R * VT * np.log(10):.1f} mV per "
                  "decade of light, through the 0.6 uA point")
    m = k.ok.to_numpy()
    ax.plot(k.V_LED[m], k.P[m] * 1e9, "o", ms=6, color=SUB[arr],
            label="measured: light against the LED's voltage (transistor subtracted)")
    e = k[~k.ok & np.isfinite(k.V_LED_est)]
    if len(e):
        ax.plot(e.V_LED_est, e.P * 1e9, "o", ms=6, mfc="w", color=SUB[arr],
                label="transistor-limited: light measured, LED voltage = junction + fitted excess")
    for _, r in k[k.I >= 1.4e-6].iterrows():
        vl = r.V_LED if r.ok else r.V_LED_est
        ax.annotate("", xy=(vl, r.P * 1e9), xytext=(r.V_j, r.P * 1e9),
                    arrowprops=dict(arrowstyle="->", color=NPG["red"], lw=1.4,
                                    ls="-" if r.ok else ":", shrinkA=0, shrinkB=5))
        if r.ok:
            ax.text(vl + 0.012, r.P * 1e9, f"{1e3 * (vl - r.V_j):.0f} mV at {r.I * 1e6:.0f} uA",
                    va="center", fontsize=7, color=NPG["red"])
    if len(e):                                       # nearly the same light: one label for all
        ex = 1e3 * (e.V_LED_est - e.V_j)
        ax.text((float(e.V_j.min()) + float(e.V_LED_est.max())) / 2, float(e.P.max()) * 1e9 * 1.15,
                f"{ex.min():.0f}-{ex.max():.0f} mV at {e.I.min() * 1e6:.0f}-{e.I.max() * 1e6:.0f} uA "
                "(estimate)", ha="center", va="bottom", fontsize=7, color=NPG["red"])
    ax.text(0.56, 0.60,
            "Reading the junction voltage off the light:\n"
            "the light depends only on the carriers in the well,\n"
            f"  P ~ exp(q V_j / n kT), n = {N_R} (measured at nA).\n"
            "For each point, go left at the same light to the\n"
            "teal line: that is V_j. The LED's measured voltage\n"
            "lies further right. The gap (red) is the excess:\n"
            "voltage that does not raise the light.\n\n"
            "The current is not used here, so the droop\n"
            "does not enter (02c). Only light ratios enter:\n"
            "the fraction of the light collected cancels,\n"
            "as long as it does not change with current.\n"
            "Line and data meet at 0.6 uA by construction.",
            transform=ax.transAxes, fontsize=7.2, family="monospace", va="top",
            bbox=dict(boxstyle="round", fc="w", alpha=.92))
    xmax = float(np.nanmax(np.r_[k.V_LED[m], e.V_LED_est])) if len(e) else float(k.V_LED[m].max())
    ax.set(yscale="log", xlabel="voltage (V)", ylabel="light (nW)", xlim=(2.45, xmax + 0.2),
           ylim=(pp.min() * 1e9, pp.max() * 1e9),
           title=f"{arr}, median pixel: how the light gives the junction voltage")
    ax.grid(alpha=.3, which="both")
    ax.legend(fontsize=6.8, loc="lower right")
    save(fig, f"02b_light_inversion_{arr}.png", plt, "Data: " + SOURCE[arr])


def fig_droop(curves, die, plt):
    """The droop: light per current falls on the high-current side. Top: light per current
    relative to its peak. Bottom: what that is worth in junction voltage, i.e. the junction
    from the light minus the junction from the current (if the light followed the current)."""
    fig, (ax, bx) = plt.subplots(2, 1, figsize=(8.2, 8.2), sharex=True,
                                 gridspec_kw=dict(height_ratios=[1.15, 1]))
    sets = [(a, curves[a].loc[lambda t: t.light & (t.I >= 0.1e-6)].sort_values("I"), "I", "P", WHOLE[a])
            for a in ARRAYS]
    sets.append(("SGA_15 die", die[die.light & (die.Current_A >= 0.1e-6)], "Current_A", "L", DIE))
    rows = []
    for name, k, ic, pc_, col in sets:
        i, eff = k[ic].to_numpy(), (k[pc_] / k[ic]).to_numpy()
        e0 = np.exp(np.interp(np.log(I_REF), np.log(i), np.log(eff)))
        dv = 1e3 * N_R * VT * np.log(eff / e0)
        sty = dict(color=col, lw=1.8) if name.startswith("SGA") else dict(color=col, marker="o", ms=4)
        ax.plot(i * 1e6, eff / eff.max(), "-", label=f"{name}: falls to {eff[-1] / eff.max():.0%} of "
                f"its peak at {i[-1] * 1e6:.0f} uA", **sty)
        bx.plot(i * 1e6, dv, "-", label=f"{name}: {dv.min():+.0f} to {dv.max():+.0f} mV", **sty)
        rows.append((name, i, dv))
    ax.set(xscale="log", ylabel="light per current, relative to its peak", ylim=(0, 1.08),
           title="The droop: on the high-current side less of each extra uA becomes light")
    ax.legend(fontsize=7.5, loc="lower left", bbox_to_anchor=(0.0, 0.27))
    ax.text(0.02, 0.05,
            "ABC picture: current ~ A n + B n^2 + C n^3 (+ leakage), light ~ B n^2,\n"
            "and the carriers set the junction: n^2 ~ exp(q V_j / kT).\n"
            "Droop: the C n^3 and leakage share of the current grows.\n"
            "That lowers the light per CURRENT, not the light per JUNCTION VOLTAGE:\n"
            "the light still measures V_j (02b).",
            transform=ax.transAxes, fontsize=7, family="monospace", va="bottom",
            bbox=dict(boxstyle="round", fc="w", alpha=.92))
    bx.axhline(0, color="k", lw=.7)
    bx.set(xscale="log", xlabel="current (uA)",
           ylabel="junction from the light minus\njunction from the current (mV)",
           title="What the droop is worth in junction voltage: a few mV (the excess is 100-500 mV)")
    bx.text(0.27, 0.03,
            f"= {N_R} kT/q ln(light per current,\n"
            "  relative to 0.6 uA): the shift of the\n"
            "  junction estimate if the light simply\n"
            "  followed the current.\n"
            "Limit: the light per carrier pair (B)\n"
            "changes itself at the highest densities\n"
            "(phase-space filling, field screening):\n"
            "tens of %, i.e. 5-10 mV.",
            transform=bx.transAxes, fontsize=7, family="monospace", va="bottom",
            bbox=dict(boxstyle="round", fc="w", alpha=.92))
    bx.legend(fontsize=7.5, loc="upper left")
    for a in (ax, bx):
        a.grid(alpha=.3, which="both")
    save(fig, "02c_droop.png", plt, "Data: " + SOURCE["Array42"] + " " + SOURCE["Array38"] + " "
         + die_source(die) + " Median pixels, every sweep bias with light.")
    return rows


def fig_iv(arr, d, db, tot, die, fsm, plt, log):
    fig, ax = plt.subplots(figsize=(7.8, 6))
    if log:
        ax.plot(br.VLED - tot.index.to_numpy(), tot.to_numpy() * 1e6, "-", color="0.7", lw=2,
                label="total current, CMOS leakage included (every bias)")
        m = br.LV < 2.25
        ax.plot(br.LV[m], br.LI[m] * 1e6, "--", color=DIE, lw=.9)
    ax.plot(d.V_tot, d.I * 1e6, "-o", ms=3.5, color=WHOLE[arr], label="whole pixel (V_tot): median")
    if not log:
        ax.fill_between(d.V_tot, d.I_q1 * 1e6, d.I_q3 * 1e6, color=WHOLE[arr], alpha=.15, lw=0,
                        label="whole pixel: 25-75 % of the pixels")
    ax.plot(db.V_tot, db.I * 1e6, "--", color=WHOLE[arr], lw=1.3, label="whole pixel: the best 2 %")
    k = d[d.ok & (d.I > 0)]
    ax.plot(k.V_LED, k.I * 1e6, "-s", ms=3.5, color=SUB[arr], label="transistor subtracted: median")
    e = d[~d.ok & np.isfinite(d.V_LED_est)]
    if len(e):
        ax.plot(np.r_[k.V_LED.iloc[-1], e.V_LED_est], np.r_[k.I.iloc[-1], e.I] * 1e6, ":s", ms=4,
                mfc="w", color=SUB[arr],
                label="transistor subtracted, where it limits the current\n(LED = junction from its light + fitted excess)")
    kb = db[db.ok & (db.I > 0)]
    eb = db[~db.ok & np.isfinite(db.V_LED_est)]
    ax.plot(np.r_[kb.V_LED, eb.V_LED_est], np.r_[kb.I, eb.I] * 1e6, "--", color=SUB[arr], lw=1.3,
            label="transistor subtracted: the best 2 %")
    f = fsm[fsm["array"] == arr]
    ax.plot(f.V_tot, f.I * 1e6, "D", mfc="none", ms=8, color=WHOLE[arr],
            label="whole pixel, full-array fast scans: median")
    ax.plot(die.Voltage_V, die.Current_A * 1e6, "-", color=DIE, lw=1.4, label=DIE_LABEL)
    if log:
        lt = leak_test(arr)
        ax.plot(lt.V_tot, lt.I_off * 1e6, "x", ms=8, mew=1.6, color="0.45",
                label=f"leakage test: the pixel off = CMOS leakage ({lt.n.iloc[0]} pixels)")
        sig = lt.I_led > 2 * lt.se
        if sig.any():
            ax.errorbar(lt.V_tot[sig], lt.I_led[sig] * 1e6, yerr=lt.se[sig] * 1e6, fmt="*", ms=11,
                        color=WHOLE[arr], capsize=3,
                        label="leakage test: lit - off, same pixel (LED current)")
        if (~sig).any():
            ul = (lt.I_led.clip(lower=0) + 2 * lt.se)[~sig]
            ax.plot(lt.V_tot[~sig], ul * 1e6, "v", ms=8, mfc="w", color=WHOLE[arr],
                    label="leakage test: upper limit (lit - off not resolved, 2 sigma)")
        ax.set(yscale="log", ylim=(1e-5, 400), xlim=(1.75, 5.1))
        ax.annotate(f"the sweep took its leakage reference from NVLED {br.VLED - d.V_tot.min():+.1f} V on;\n"
                    f"below that the leakage test resolves ~{2e9 * lt.se.median():.1f} nA (2 sigma)",
                    (1.8, 2.5e-3), fontsize=7, color="0.4")
    else:
        ax.set(ylim=(0, 135), xlim=(2.6, 5.1))
    ax.set(xlabel="voltage (V)", ylabel="LED current, leakage subtracted (uA)",
           title=f"{arr}: I-V of the pixel, whole and with the transistor subtracted")
    ax.grid(alpha=.3, which="both")
    ax.legend(fontsize=6.5, loc="lower right")
    save(fig, f"{'01_iv_log' if log else '02_iv_linear'}_{arr}.png", plt,
         "Data: " + SOURCE[arr] + " Fast scans: " + ", ".join(sorted(set(f.run))) + ". " + die_source(die))


def fig_budget(arr, d, plt):
    k = d[d.light & np.isfinite(d.V_LED_est)].sort_values("V_tot")
    x0 = k.V_tot.iloc[0]
    xl = np.linspace(1.8, x0, 30)[:-1]                   # no measurable current: all on the junction
    x = np.r_[xl, k.V_tot.to_numpy()]
    vj = np.r_[xl, np.minimum(k.V_j.to_numpy(), k.V_LED_est.to_numpy())]
    led = np.r_[xl, k.V_LED_est.to_numpy()]
    sat = np.r_[np.zeros(len(xl), bool), ~k.ok.to_numpy()]
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.fill_between(x, 0, vj, color=NPG["teal"], alpha=.75, lw=0, label="LED junction (from its own light)")
    ax.fill_between(x, vj, led, color=NPG["red"], alpha=.75, lw=0,
                    label="excess: barrier + series resistance (LED, bump, lines)")
    ax.fill_between(x, led, x, color=NPG["lilac"], alpha=.75, lw=0, label="transistor")
    if sat.any():
        j = np.flatnonzero(sat)[0] - 1
        ax.fill_between(x[j:], led[j:], x[j:], facecolor="none", hatch="//", edgecolor="0.25", lw=0,
                        label="transistor saturated: it limits the current and takes the rest")
    ax.plot(x, x, "k-", lw=1)
    lines = ["NVLED   V_tot      I      junction    excess        transistor"]
    for vt in (3.0, 3.2, 3.4, 4.0, 5.0):
        i = np.argmin(np.abs(k.V_tot.to_numpy() - vt))
        r = k.iloc[i]
        if abs(r.V_tot - vt) > 0.01:
            continue
        ex, drv = r.V_LED_est - r.V_j, r.V_tot - r.V_LED_est
        lines.append(f"{br.VLED - r.V_tot:+.1f} V  {r.V_tot:.1f} V  {r.I * 1e6:5.1f} uA  "
                     f"{r.V_j:.2f} V {100 * r.V_j / r.V_tot:3.0f} %  {1e3 * ex:4.0f} mV {100 * ex / r.V_tot:3.0f} %  "
                     f"{1e3 * drv:5.0f} mV {100 * drv / r.V_tot:3.0f} %")
    ax.text(0.02, 0.97, "\n".join(lines), transform=ax.transAxes, fontsize=6.6, family="monospace",
            va="top", bbox=dict(boxstyle="round", fc="w", alpha=.92))
    ax.annotate("below ~2.5 V almost no current flows:\nall of V_tot is across the LED junction",
                (1.85, 0.25), fontsize=7.5, color="w")
    sec = ax.secondary_xaxis("top", functions=(lambda v: br.VLED - v, lambda n: br.VLED - n))
    sec.set_xlabel("NVLED (V)")
    ax.set(xlabel="V_tot = VLED - NVLED (V)", ylabel="voltage (V)", xlim=(1.8, x.max()),
           ylim=(0, 5.2), title=f"{arr}, median pixel: where the voltage goes")
    ax.legend(fontsize=7, loc="lower right")
    save(fig, f"03_voltage_budget_{arr}.png", plt,
         "Data: " + SOURCE[arr] + " Junction: V_LED(0.6 uA) + 1.07 kT/q ln(P/P(0.6 uA)). Hatched: the "
         "transistor's model cannot give its drop; the LED is its junction plus the fitted excess (09).")


def fig_ideality(curves, die, plt):
    fig, ax = plt.subplots(figsize=(8.2, 6.2))
    peaks = {}
    for arr in ARRAYS:
        d = curves[arr]
        k = d[d.ok & d.light].sort_values("I")
        i, v, p = k.I.to_numpy(), k.V_LED.to_numpy(), k.P.to_numpy()
        ax.plot(mid(i) * 1e6, np.diff(v) / (VT * np.diff(np.log(p))), "-o", ms=4, color=WHOLE[arr],
                label=f"{arr}: from the light")
        ax.plot(mid(i) * 1e6, np.diff(v) / (VT * np.diff(np.log(i))), "--s", ms=3, color=SUB[arr],
                label=f"{arr}: from the current")
        e = k[k.I > 0.1e-6]
        ip = float(e.I.iloc[np.argmax((e.P / e.I).to_numpy())])
        peaks[arr] = ip
        ax.axvline(ip * 1e6, color=WHOLE[arr], ls=":", lw=1.2)
    k = die[die.light & (die.Current_A > 0.2e-6)].iloc[::3]
    i, v, light = k.Current_A.to_numpy(), k.Voltage_V.to_numpy(), k.L.to_numpy()
    ax.plot(mid(i) * 1e6, np.diff(v) / (VT * np.diff(np.log(light))), "-", color=DIE, lw=1.6,
            label="die: from the light")
    k = die[die.Current_A > 0.02e-6].iloc[::3]
    i, v = k.Current_A.to_numpy(), k.Voltage_V.to_numpy()
    ax.plot(mid(i) * 1e6, np.diff(v) / (VT * np.diff(np.log(i))), "--", color=DIE, lw=1, alpha=.7,
            label="die: from the current")
    ax.axvspan(1e-3, 0.6, color="0.88", zorder=0)
    ax.annotate("pixels: leakage > 25 % of the\ncurrent, current-based value\nunreliable", (0.025, 3.2),
                fontsize=6.5, color="0.4")
    ax.axhline(1, color="k", lw=.6, ls=":")
    ax.annotate("dotted: highest light/current (IQE peak)\n"
                + ", ".join(f"{a} {i * 1e6:.1f} uA" for a, i in peaks.items()),
                (min(peaks.values()) * 1e6 * 0.9, 0.25), fontsize=6.5, color="0.3", ha="right")
    ax.text(0.02, 0.97,
            "local ideality: how much voltage the LED needs per e-fold\n"
            "  from the light:    n_L = (q/kT) dV / d ln P\n"
            "  from the current:  n_I = (q/kT) dV / d ln I\n"
            "an ideal junction: n_L = 1 (all voltage raises the light)\n\n"
            "tie to the IQE (light/current):\n"
            "  d ln(IQE)/dV = (q/kT) (1/n_L - 1/n_I)\n"
            "  n_L < n_I: IQE rising     n_L > n_I: IQE falling (droop)\n"
            "  they cross at the IQE peak (dotted lines)\n\n"
            "n_L >> 1 at high current: voltage that raises no light,\n"
            "the excess of 08 (barriers + series resistance)",
            transform=ax.transAxes, fontsize=6.6, family="monospace", ha="left", va="top",
            bbox=dict(boxstyle="round", fc="w", alpha=.92))
    sec = ax.secondary_xaxis("top", functions=(lambda u: u * 1e-6 / (aa.PIXEL_AREA_UM2 * 1e-8),
                                               lambda j: j * aa.PIXEL_AREA_UM2 * 1e-8 * 1e6))
    sec.set_xlabel("current density (A/cm2)")
    ax.set(xscale="log", xlabel="current (uA)", ylabel="local ideality", ylim=(0, 16), xlim=(0.02, 500),
           title="Ideality from the light and from the current (median pixel)")
    ax.grid(alpha=.3, which="both")
    ax.legend(fontsize=7, loc="lower right")
    save(fig, "04_ideality.png", plt,
         "Data: " + SOURCE["Array42"] + " " + SOURCE["Array38"] + " " + die_source(die))


def fig_dvdi(curves, die, drv, plt):
    fig, ax = plt.subplots(figsize=(8.2, 6.2))
    ii = np.logspace(-8, -3, 50)
    for n in (1, 2, 4, 8, 16):
        ax.plot(ii * 1e6, n * VT / ii / 1e3, color="0.8", lw=.8)
        ax.annotate(f"n = {n}", (700, n * VT / 700e-6 / 1e3), fontsize=6.5, color="0.5", va="bottom")
    for arr in ARRAYS:
        k = curves[arr]
        k = k[k.ok & (k.I > 0.1e-6)].sort_values("I")
        i = k.I.to_numpy()
        ax.plot(mid(i) * 1e6, np.diff(k.V_LED) / np.diff(i) / 1e3, "-", lw=1.2, color=SUB[arr],
                label=f"{arr}: transistor subtracted (LED junction still in)")
        e = k[k.light & (k.I >= I_REF)]
        ax.plot(mid(e.I.to_numpy()) * 1e6, np.diff(e.V_ex) / np.diff(e.I) / 1e3, "-o", ms=4, lw=2,
                color=WHOLE[arr], label=f"{arr}: excess only (junction removed via its light)")
    k = die[die.Current_A > 0.05e-6].iloc[::3]
    i = k.Current_A.to_numpy()
    ax.plot(mid(i) * 1e6, np.diff(k.Voltage_V) / np.diff(i) / 1e3, "-", color=DIE, lw=1.2,
            label="die: whole (LED junction in)")
    e = die[die.light & (die.Current_A >= I_REF)].iloc[::3]
    ax.plot(mid(e.Current_A.to_numpy()) * 1e6, np.diff(e.V_ex) / np.diff(e.Current_A) / 1e3, "-",
            color=DIE, lw=2.4, label="die: excess only (0.6-420 uA)")
    ax.axhline(0.27, color=NPG["brown"], ls="--", lw=1.2)
    ax.annotate("a resistor is a horizontal line, e.g. the die's own 270 Ohm (its Cheung fit, see 11)",
                (0.04, 0.29), fontsize=7, color=NPG["brown"])
    ax.set(xscale="log", yscale="log", xlabel="current (uA)", ylabel="dV/dI (kOhm)",
           xlim=(0.03, 1500), ylim=(0.05, 3000),
           title="Differential resistance: even with the junction removed it falls with current\n"
                 "grey: an ideal diode, n kT/(qI); a resistor would be flat")
    ax.grid(alpha=.3, which="both")
    ax.legend(fontsize=7, loc="upper right")
    save(fig, "05_differential_resistance.png", plt,
         "Data: " + SOURCE["Array42"] + " " + SOURCE["Array38"] + " " + die_source(die)
         + " Median pixel; transistor-limited points left out.")


def fig_r_vs_v(arr, d, P, plt):
    k = d[d.ok & (d.I > 0.1e-6)].sort_values("I")
    i, x = k.I.to_numpy(), mid(k.V_tot.to_numpy())
    fig, ax = plt.subplots(figsize=(7.8, 5.8))
    ax.plot(x, np.diff(k.V_tot) / np.diff(i) / 1e3, "-o", ms=4, color="0.5", lw=1, label="whole pixel")
    ax.plot(x, np.diff(k.V_LED) / np.diff(i) / 1e3, "-s", ms=4, color=SUB[arr], lw=1.2,
            label="transistor subtracted: LED (junction still in) + bump + lines")
    e = (k.I.to_numpy() >= I_REF) & k.light.to_numpy()
    ax.plot(mid(k.V_tot.to_numpy()[e]), np.diff(k.V_ex.to_numpy()[e]) / np.diff(i[e]) / 1e3, "-^",
            ms=6, lw=2.4, color=WHOLE[arr],
            label="excess: the LED junction removed via its light (barrier + series R)")
    ron = P["vov"] / (P["I15"] * (1 - 5 * P["lam"])) / 1e3
    ax.axhline(ron, color=NPG["lilac"], ls="--", lw=1.2, label=f"for scale: the transistor's on-resistance ({ron:.1f} kOhm)")
    ax.axhline(0.27, color=NPG["brown"], ls=":", lw=1.2,
               label="for scale: the die's own series R (270 Ohm, its Cheung fit)")
    sec = ax.secondary_xaxis("top", functions=(lambda v: br.VLED - v, lambda n: br.VLED - n))
    sec.set_xlabel("NVLED (V)")
    ax.set(xlabel="V_tot (V)", ylabel="dV/dI between neighbouring sweep points (kOhm)",
           yscale="log", ylim=(0.1, 300),
           title=f"{arr}: the 'series resistance' at each voltage\n"
                 "the junction-free excess still falls 5x: not a resistor")
    ax.grid(alpha=.3, which="both")
    ax.legend(fontsize=7, loc="upper right")
    save(fig, f"06_resistance_vs_voltage_{arr}.png", plt,
         "Data: " + SOURCE[arr] + " Median pixel; transistor-limited points left out.")


def full(g, best=False):
    """Grid points that most pixels reach below the transistor limit. Near it, only the
    pixels with the least current still count (on Array42 at 76 uA: 2121 of 45 465, the
    high-resistance ones), which biases the median."""
    n = g.n_best if best else g.n
    return (n >= 0.7 * n.max()).to_numpy()


def band(ax, g, lo, mid_, hi, color, label):
    ok = np.isfinite(g[mid_]).to_numpy() & full(g)
    ax.fill_between(g.I[ok] * 1e6, 1e3 * g[lo][ok], 1e3 * g[hi][ok], color=color, alpha=.18, lw=0)
    ax.plot(g.I[ok] * 1e6, 1e3 * g[mid_][ok], "-o", ms=3.5, color=color, label=label)


def fig_minus_die(gs, die, plt):
    fig, ax = plt.subplots(figsize=(7.8, 5.8))
    for arr in ARRAYS:
        g = gs[arr]
        band(ax, g, "dd_q1", "dd_med", "dd_q3", WHOLE[arr], f"{arr}: median, band 25-75 % of the pixels")
        ok = np.isfinite(g.dd_best).to_numpy() & full(g, best=True)
        ax.plot(g.I[ok] * 1e6, 1e3 * g.dd_best[ok], "--", lw=1.6, color=WHOLE[arr], label=f"{arr}: the best 2 %")
    ax.axhline(0, color="k", lw=.8)
    ax.set(xscale="log", xlabel="current (uA)",
           ylabel="V_LED(pixel) - V(die) at the same current (mV)",
           title="Pixel (transistor subtracted) against the unbonded SGA_15 die\n"
                 "offset: the pixels' junction turns on earlier; rise: what the bond adds")
    ax.grid(alpha=.3, which="both")
    ax.legend(fontsize=7.5)
    save(fig, "07_pixel_minus_die.png", plt,
         "Data: every swept pixel, interpolated to each current: " + SOURCE["Array42"] + " "
         + SOURCE["Array38"] + " " + die_source(die))


def fig_excess(gs, die, plt):
    fig, ax = plt.subplots(figsize=(7.8, 5.8))
    e = die[die.light & (die.Current_A >= I_REF)]
    ax.plot(e.Current_A * 1e6, 1e3 * e.V_ex, "-", color=DIE, lw=2.2, label=DIE_LABEL)
    for arr in ARRAYS:
        g = gs[arr]
        band(ax, g, "ex_q1", "ex_med", "ex_q3", WHOLE[arr], f"{arr} (bonded): median, band 25-75 %")
        ok = np.isfinite(g.ex_best).to_numpy() & full(g, best=True)
        ax.plot(g.I[ok] * 1e6, 1e3 * g.ex_best[ok], "--", lw=1.6, color=WHOLE[arr], label=f"{arr}: the best 2 %")
    lines = ["current     die  A42 med  best  A38 med  best  (mV)"]
    for t in (1.5e-6, 5e-6, 20e-6, 40e-6, 55e-6):
        vals = [1e3 * interp_log(e.Current_A.to_numpy(), np.array([t]), e.V_ex.to_numpy())[0]]
        for arr in ARRAYS:
            g = gs[arr]
            for c in ("ex_med", "ex_best"):
                ok = np.isfinite(g[c]).to_numpy() & full(g, best=c == "ex_best")
                vals.append(1e3 * interp_log(g.I[ok].to_numpy(), np.array([t]), g[c][ok].to_numpy())[0])
        lines.append(f"{t * 1e6:5.1f} uA " + "".join(f"{x:7.0f}" if np.isfinite(x) else "      -"
                                                     for x in vals))
    ax.text(0.03, 0.72, "\n".join(lines), transform=ax.transAxes, fontsize=7, family="monospace",
            va="top", bbox=dict(boxstyle="round", fc="w", alpha=.92))
    ax.set(xscale="log", xlabel="current (uA)",
           ylabel=f"excess voltage over the junction (mV), counted from {I_REF * 1e6:g} uA",
           xlim=(0.5, 500),
           title="The excess voltage is already in the unbonded die\n"
                 "junction from each device's own light")
    ax.grid(alpha=.3, which="both")
    ax.legend(fontsize=7.5, loc="upper left")
    save(fig, "08_excess_die_vs_pixels.png", plt,
         "Data: every swept pixel, interpolated to each current: " + SOURCE["Array42"] + " "
         + SOURCE["Array38"] + " " + die_source(die))


def fig_fits(name, i, v, fits_, plt, source, note=""):
    fig, (ax, rx) = plt.subplots(2, 1, figsize=(7.8, 7.2), sharex=True,
                                 gridspec_kw=dict(height_ratios=[3, 1.3]))
    if note:
        ax.text(0.03, 0.62, note, transform=ax.transAxes, fontsize=7, family="monospace", va="top",
                bbox=dict(boxstyle="round", fc="w", alpha=.92))
    ax.plot(i * 1e6, 1e3 * v, "ko", ms=4, label="data")
    xx = np.logspace(np.log10(I_REF), np.log10(i.max()), 200)
    for model, r in fits_.items():
        ax.plot(xx * 1e6, 1e3 * r["f"](xx), "-", lw=1.8, color=FIT_COL[model], label=describe(model, r))
        rx.plot(i * 1e6, 1e3 * (v - r["f"](i)), "o-", ms=3, color=FIT_COL[model])
    rx.axhline(0, color="k", lw=.6)
    ax.set(xscale="log", ylabel="excess voltage (mV)",
           title=f"{name}: what describes the excess voltage?\n"
                 "a resistor alone would be a straight line on a linear axis")
    rx.set(xlabel="current (uA)", ylabel="data - fit (mV)")
    for a in (ax, rx):
        a.grid(alpha=.3, which="both")
    ax.legend(fontsize=7, loc="upper left")
    save(fig, f"09_excess_fits_{'die' if name.startswith('SGA') else name}.png", plt, "Data: " + source)


def fig_bond_fits(arr, d, D, plt):
    """Is the pixel the die's LED plus a bond term? The die's excess is fixed as measured;
    only the bond term is fitted (to the median pixel's excess)."""
    i, v = excess_points(d)
    fits = {m: bond_fit(D, i, v, m) for m in BOND}
    fig, (ax, rx) = plt.subplots(2, 1, figsize=(7.8, 7.6), sharex=True,
                                 gridspec_kw=dict(height_ratios=[3, 1.4]))
    xx = np.geomspace(I_REF, i.max(), 200)
    ax.plot(xx * 1e6, 1e3 * D(xx), "--", color=DIE, lw=1.8, zorder=5,
            label="SGA_15 die as measured: its LED barrier, fixed")
    ax.plot(i * 1e6, 1e3 * v, "o", ms=7, color=WHOLE[arr], label=f"{arr}: median pixel")
    rx.plot(i * 1e6, 1e3 * (v - D(i)), "o-", ms=4, color=DIE, lw=1.2, label="data - die alone")
    for m, r in fits.items():
        ax.plot(xx * 1e6, 1e3 * r["f"](xx), "-", lw=1.6, color=BOND_COL[m], label=describe_bond(m, r))
        rx.plot(i * 1e6, 1e3 * (v - r["f"](i)), "o-", ms=3, color=BOND_COL[m])
    rx.axhline(0, color="k", lw=.6)
    ax.text(0.03, 0.62,
            "Fixed: the die's excess as measured, i.e. its LED barrier\n"
            "and its own resistance (same wafer, unbonded).\n"
            "Fitted: only what the pixel adds or lacks:\n"
            "  die + R:            a resistor (bump, lines, n-path)\n"
            "  die + 2nd barrier:  a second barrier (oxide, Schottky)\n"
            "  k x die + R:        the die's barrier scaled by k\n"
            "                      (die to die), plus a resistor",
            transform=ax.transAxes, fontsize=7, family="monospace", va="top",
            bbox=dict(boxstyle="round", fc="w", alpha=.92))
    ax.set(xscale="log", ylabel="excess voltage (mV), counted from 0.6 uA",
           title=f"{arr}: is the pixel the die's LED plus a bond term?")
    rx.set(xlabel="current (uA)", ylabel="data - fit (mV)")
    for a in (ax, rx):
        a.grid(alpha=.3, which="both")
    ax.legend(fontsize=6.8, loc="upper left")
    rx.legend(fontsize=7, loc="lower right")
    save(fig, f"09_excess_fits_{arr}.png", plt,
         "Data: " + SOURCE[arr] + " Median pixel, the points below the transistor limit. "
         "Die: its IVL of 2023-08-18, interpolated.")
    return fits


def fig_window(rows, plt, die):
    fig, ax = plt.subplots(figsize=(7.8, 5.6))
    t = pd.DataFrame(rows)
    for name, c in (("SGA_15 die", DIE), ("Array42", WHOLE["Array42"]), ("Array38", WHOLE["Array38"])):
        s = t[t.curve == name]
        ax.errorbar(s.i_max_uA, s.R_kOhm, yerr=s.R_se_kOhm, fmt="o-", capsize=3, color=c, label=name)
    ax.axhline(0.27, color=NPG["brown"], ls="--", label="die's series R from its Cheung fit (270 Ohm)")
    ax.axhline(0, color="k", lw=.6)
    ax.set(xscale="log", xlabel=f"upper end of the fit window (uA); lower end {I_REF * 1e6:g} uA",
           ylabel="series R of the barrier + resistor fit (kOhm)",
           title="The fitted series R depends on how far the data reach\n"
                 "the die: still falling at 0.42 mA; the pixels stop at ~70-90 uA (transistor)")
    ax.grid(alpha=.3, which="both")
    ax.legend(fontsize=8)
    save(fig, "10_fit_window.png", plt, "Data: median pixel of " + SOURCE["Array42"] + " "
         + SOURCE["Array38"] + " " + die_source(die))


def fig_cheung(curves, plt):
    lv, li = br.LV, br.LI                          # the die's IV sweep, to 0.84 mA
    m = li > 1e-6
    i, v = li[m], lv[m]
    h = mid(i) * np.diff(v) / np.diff(i)
    im = mid(i)
    fig, ax = plt.subplots(figsize=(7.8, 5.8))
    ax.plot(im * 1e3, h, "ko-", ms=4, label="die: I dV/dI")
    sel = (im > 0.35e-3) & (im < 0.9e-3)
    b, a = np.polyfit(im[sel], h[sel], 1)
    xx = np.linspace(0, 0.9e-3, 10)
    ax.plot(xx * 1e3, a + b * xx, "--", color=NPG["brown"], lw=1.4,
            label=f"die, line through 0.35-0.9 mA: intercept {a:.3f} V (n = {a / VT:.1f}), slope {b:.0f} Ohm")
    for arr in ARRAYS:
        k = curves[arr]
        k = k[k.ok & (k.I > 0.5e-6)].sort_values("I")
        i2 = k.I.to_numpy()
        ax.plot(mid(i2) * 1e3, mid(i2) * np.diff(k.V_LED) / np.diff(i2), "o-", ms=4, color=WHOLE[arr],
                label=f"{arr}: transistor subtracted, median pixel")
    ax.axvspan(0, 0.07, color="0.9", zorder=0)
    ax.annotate("pixels' range\n(transistor-\nlimited above)", (0.004, 0.02), fontsize=7, color="0.4")
    ax.text(0.97, 0.05,
            "Cheung's method: for V = n kT/q ln(I/I0) + R I\n"
            "  dV/dI = n kT/(qI) + R   =>   I dV/dI = n kT/q + R I\n"
            "a straight line in I: intercept n kT/q, slope R.\n"
            "It holds only where n is constant: here above ~0.35 mA.\n"
            "Below, the curve bends: the barrier term changes\n"
            "with current, and no single n and R describe it.",
            transform=ax.transAxes, fontsize=7, family="monospace", ha="right", va="bottom",
            bbox=dict(boxstyle="round", fc="w", alpha=.92))
    ax.set(xlabel="current (mA)", ylabel="I dV/dI (V)", xlim=(0, 0.9), ylim=(0, 0.5),
           title="Cheung plot: where the die's 270 Ohm comes from\n"
                 "its own analysis: 270 Ohm, n = 10.4 (fit over 0.38-0.85 mA)")
    ax.grid(alpha=.3)
    ax.legend(fontsize=7, loc="upper left")
    save(fig, "11_cheung.png", plt, "Data: the die's IV sweep of 2023-08-18 (0.1 V steps, to 0.84 mA); "
         + SOURCE["Array42"] + " " + SOURCE["Array38"])
    return a, b


def die_at(die, t):
    """The die's excess at current t (mV)."""
    e = die[die.light & (die.Current_A >= I_REF)]
    return 1e3 * interp_log(e.Current_A.to_numpy(), np.array([t]), e.V_ex.to_numpy())[0]


def fig_hist(pp, die, plt):
    """Per pixel: how much more (or less) voltage than the unbonded die, at 20 uA."""
    fig, ax = plt.subplots(figsize=(7.8, 5.6))
    t = 20e-6
    dv = die_at(die, t)
    for arr in ARRAYS:
        s = 1e3 * pp[arr][f"V_ex_{tag(t)}"].dropna() - dv
        sb = 1e3 * pp[arr].loc[pp[arr].best, f"V_ex_{tag(t)}"].dropna() - dv
        ax.hist(s.clip(-150, 450), bins=150, histtype="step", lw=1.6, color=WHOLE[arr],
                label=f"{arr}: {len(s)} pixels, median {s.median():+.0f} mV "
                      f"(25-75 %: {s.quantile(.25):+.0f} to {s.quantile(.75):+.0f})")
        ax.axvline(sb.median(), color=WHOLE[arr], ls="--", lw=1.2,
                   label=f"{arr}: median of its best 2 %, {sb.median():+.0f} mV")
    ax.axvline(0, color=DIE, lw=2, label=f"0 = the unbonded SGA_15 die (its excess: {dv:.0f} mV)")
    ax.set(xlabel=f"excess voltage at {t * 1e6:g} uA minus the die's (mV), clipped to -150..450",
           ylabel="pixels", yscale="log",
           title="Per swept pixel: how much more voltage than the unbonded die, at 20 uA\n"
                 "excess = voltage that does not raise the light (02b); < 0: less than the die, "
                 "> 0: bonding adds")
    ax.title.set_fontsize(10)
    ax.grid(alpha=.3)
    ax.legend(fontsize=7.2, loc="upper right")
    save(fig, "12_excess_per_pixel.png", plt, "Data: " + SOURCE["Array42"] + " " + SOURCE["Array38"]
         + " " + die_source(die))


def fig_map(arr, p, dv, plt, B=16):
    """Where the pixels need more (red) or less (blue) than the unbonded die, at 20 uA."""
    col = f"V_ex_{tag(20e-6)}"
    s = p.dropna(subset=[col])
    g = s.groupby([s.Y // B, s.X // B])[col]
    blk = g.median()[g.size() >= 5]
    img = np.full((512 // B, 512 // B), np.nan)
    img[blk.index.get_level_values(0), blk.index.get_level_values(1)] = 1e3 * blk.to_numpy() - dv
    resid = 1e3 * (s[col] - s.set_index([s.Y // B, s.X // B]).index.map(blk).to_numpy())
    cm = plt.get_cmap("RdBu_r").copy()
    cm.set_bad("0.8")
    lim = float(np.nanpercentile(np.abs(img), 90))    # the edge blocks saturate; their tail is in 12
    fig, ax = plt.subplots(figsize=(7.4, 6.8))
    im = ax.imshow(img, cmap=cm, vmin=-lim, vmax=lim, extent=(-.5, 511.5, 511.5, -.5),
                   interpolation="nearest")
    fig.colorbar(im, ax=ax, extend="both", fraction=.046,
                 label=f"pixel minus die at 20 uA: median in {B}x{B} blocks (mV)")
    bq = np.nanpercentile(img, [10, 90])
    rq = np.nanpercentile(resid, [25, 75])
    ax.set(xlabel="X", ylabel="Y",
           title=f"{arr}: excess at 20 uA minus the unbonded die's ({dv:.0f} mV)\n"
                 f"blue: less than the die, red: more. Blocks 10-90 %: {bq[0]:+.0f} to {bq[1]:+.0f} mV;\n"
                 f"a pixel against its block: {rq[0]:+.0f} to {rq[1]:+.0f} mV (25-75 %). "
                 f"{len(s)} swept pixels, grey: none")
    ax.title.set_fontsize(9)
    save(fig, f"13_excess_map_{arr}.png", plt, "Data: " + SOURCE[arr] + " " + die_source_short())


def fig_barrier_maps(arr, p, plt, B=16):
    """Where the barrier (k, against the die's) and the extra resistor (R) sit: the per-pixel
    die-anchored fits as 16x16 block medians, side by side."""
    from matplotlib.colors import Normalize, TwoSlopeNorm
    s = p.dropna(subset=["k_barrier", "R_extra_kOhm"])
    fig, ax = plt.subplots(1, 2, figsize=(13.8, 6.4))
    for x, col, lab, norm, cmap in (
            (ax[0], "k_barrier", "barrier k (1 = the die's)", TwoSlopeNorm(1.0, 0.3, 1.7), "RdBu_r"),
            (ax[1], "R_extra_kOhm", "extra resistor R (kOhm)", Normalize(0, 5), "magma")):
        g = s.groupby([s.Y // B, s.X // B])[col]
        blk = g.median()[g.size() >= 5]
        img = np.full((512 // B, 512 // B), np.nan)
        img[blk.index.get_level_values(0), blk.index.get_level_values(1)] = blk.to_numpy()
        cm = plt.get_cmap(cmap).copy()
        cm.set_bad("0.8")
        im = x.imshow(img, cmap=cm, norm=norm, extent=(-.5, 511.5, 511.5, -.5), interpolation="nearest")
        fig.colorbar(im, ax=x, fraction=.046, extend="both", label=f"{lab}, median in {B}x{B} blocks")
        q = s[col].quantile([.1, .5, .9])
        x.set(xlabel="X", ylabel="Y",
              title=f"{lab}: median {q[.5]:.2f}, 10-90 % of the pixels {q[.1]:.2f} to {q[.9]:.2f}")
        x.title.set_fontsize(9.5)
    fig.suptitle(f"{arr}: each pixel fitted as the die's barrier x k plus a resistor R\n"
                 "(its own sweep points from 0.6 uA to the transistor limit; grey: none)", fontsize=11)
    save(fig, f"15_barrier_and_resistor_map_{arr}.png", plt,
         "Data: " + SOURCE[arr] + f" {len(s)} pixels; per-pixel fit: median {s.n_fit.median():.0f} "
         f"points, rms {s.fit_rms_mV.median():.1f} mV. R carries +-0.5 kOhm from the transistor model.")


DEAD_WIN = 33                # px: the neighbourhood whose share of dead pixels is counted
DEAD_GROUPS = [0, 10, 20, 30, 50, 100]   # % dead neighbours: the groups of figure 17


def dead_map(arr):
    """Share of dark pixels (open, short, dark_conducting in the same run's fast scan at
    -1.40 V) in the DEAD_WIN x DEAD_WIN px around each pixel, in %, as a 512 x 512 map."""
    import bond_failure as bf
    from scipy.ndimage import uniform_filter
    cls = bf.classes(br.SWEEP[arr], -1.4)
    dark = np.zeros((512, 512), bool)
    idx = cls[cls.isin(bf.DARK)].index
    dark[idx.get_level_values("Y"), idx.get_level_values("X")] = True
    return 100 * uniform_filter(dark.astype(float), size=DEAD_WIN, mode="nearest")


def dead_groups(sw, P, dm):
    """The median pixel of each DEAD_GROUPS group of swept pixels: {label: (n, curve)}."""
    px = pd.MultiIndex.from_arrays([sw.X, sw.Y]).unique()
    fr = dm[px.get_level_values(1).to_numpy(), px.get_level_values(0).to_numpy()]
    codes = np.digitize(fr, DEAD_GROUPS[1:-1])
    out = {}
    for k, (lo, hi) in enumerate(zip(DEAD_GROUPS[:-1], DEAD_GROUPS[1:])):
        sel = px[codes == k]
        if len(sel) >= 200:
            out[f"{lo}-{hi} %"] = (len(sel), curve(sw, P, sel))
    return out


def fig_vs_dark(arr, p, ref, dm, plt):
    """Does the barrier or the resistor grow where more pixels have died? Per-pixel k and R
    against the share of dark pixels (open, short, dark_conducting in the same run's fast scan)
    in a DEAD_WIN x DEAD_WIN window around the pixel. The dead pixels are scattered through the
    live area (~20 % in Array38's best corner), so the distance to the nearest one is 1 px for
    nearly every pixel and says nothing; the local share does."""
    s = p.dropna(subset=["k_barrier", "R_extra_kOhm"]).copy()
    s["dead"] = dm[s.Y.to_numpy(), s.X.to_numpy()]
    bins = np.arange(0, 101, 10)
    g = s.groupby(pd.cut(s.dead, bins, include_lowest=True), observed=True)
    labels = [f"{max(iv.left, 0):.0f}-{iv.right:.0f}" for iv in g.size().index]
    fig, (ax, bx) = plt.subplots(2, 1, figsize=(7.8, 7.8), sharex=True)
    for x, col, lab, refv, die_v in ((ax, "k_barrier", "barrier k (1 = the die's)", ref["k"], 1.0),
                                     (bx, "R_extra_kOhm", "extra resistor R (kOhm)", ref["R"], None)):
        med, q1, q3 = g[col].median(), g[col].quantile(.25), g[col].quantile(.75)
        xs = np.arange(len(med))
        x.errorbar(xs, med, yerr=[med - q1, q3 - med], fmt="o-", ms=5, capsize=3, color=WHOLE[arr],
                   label=f"{arr}: median, 25-75 % of the pixels")
        x.axhline(refv, color=WHOLE["Array42"], ls=":", lw=1.4, label=f"Array42, all pixels: {refv:.2f}")
        if die_v is not None:
            x.axhline(die_v, color=DIE, lw=1, label="the unbonded die")
        x.set(ylabel=lab)
        x.grid(alpha=.3)
        x.legend(fontsize=7.5, loc="upper right")
    for xi, n in zip(np.arange(len(g.size())), g.size()):
        ax.annotate(f"{n}", (xi, ax.get_ylim()[0]), xytext=(0, 3), textcoords="offset points",
                    ha="center", fontsize=6.5, color="0.4")
    bx.set_xticks(np.arange(len(labels)), labels)
    bx.set(xlabel=f"dead pixels in the {DEAD_WIN} x {DEAD_WIN} px around the pixel (%); "
                  "grey numbers: pixels per bin")
    ax.set_title(f"{arr}: does the barrier or the resistor grow where more pixels have died?",
                 fontsize=10.5)
    save(fig, f"16_barrier_and_resistor_vs_dark_{arr}.png", plt,
         "Data: " + SOURCE[arr] + " Dark: open, short or dark_conducting in the same run's fast scan "
         "at -1.40 V. R carries +-0.5 kOhm from the transistor model.")


def _g(i, lb):
    """The barrier term, counted from I_REF: ln(1 + I/I_b) - ln(1 + I_REF/I_b), I_b = 10**lb."""
    return np.log1p(i / 10 ** lb) - np.log1p(I_REF / 10 ** lb)


# what degradation adds to a reference pixel's excess r(I): delta(I) = model(p, I, r(I))
SECOND = {
    "+ resistor": (lambda p, i, r: p[0] * (i - I_REF), [1e3], ([-5e4], [5e4])),
    "+ 2nd barrier": (lambda p, i, r: p[0] * _g(i, p[1]), [0.1, -5.0], ([0, -8], [2, -2])),
    "+ 2nd barrier + resistor": (lambda p, i, r: p[0] * _g(i, p[1]) + p[2] * (i - I_REF),
                                 [0.1, -5.0, 1e3], ([0, -8, -5e4], [2, -2, 5e4])),
    "reference barrier x k + resistor": (lambda p, i, r: (p[0] - 1) * r + p[1] * (i - I_REF),
                                         [1.2, 1e3], ([0, -5e4], [5, 5e4])),
}
SECOND_COL = {"+ resistor": NPG["tan"], "+ 2nd barrier": NPG["red"],
              "+ 2nd barrier + resistor": NPG["navy"], "reference barrier x k + resistor": NPG["teal"]}


def fit_second(model, i, d, r):
    f, p0, bounds = SECOND[model]
    res = least_squares(lambda p: f(p, i, r) - d, p0, bounds=bounds)
    cov = np.linalg.pinv(res.jac.T @ res.jac) * (res.fun @ res.fun) / max(1, len(i) - len(p0))
    return dict(p=res.x, se=np.sqrt(np.abs(np.diag(cov))), n=len(i),
                rms=float(np.sqrt(np.mean(res.fun ** 2))), f=lambda x, rr, p=res.x: f(p, x, rr))


def describe_second(model, r):
    p, se = r["p"], r["se"]
    if model == "+ resistor":
        s = f"R {p[0] / 1e3:.2f} kOhm"
    elif model == "+ 2nd barrier":
        s = f"a {1e3 * p[0]:.0f} mV, I_b {10 ** p[1] * 1e6:.2g} uA"
    elif model == "+ 2nd barrier + resistor":
        s = (f"a {1e3 * p[0]:.0f} +- {1e3 * se[0]:.0f} mV, I_b {10 ** p[1] * 1e6:.2g} uA "
             f"(x/ {10 ** se[1]:.2g}), R {p[2] / 1e3:.2f} +- {se[2] / 1e3:.2f} kOhm")
    else:
        s = f"k {p[0]:.2f}, R {p[1] / 1e3:.2f} kOhm"
    return f"{model}: {s}; rms {1e3 * r['rms']:.1f} mV"


def fig_second_barrier(groups, c42, plt):
    """Figure 17: what the 'second barrier' is, which equation is assumed and fitted, and what
    Array38's data give. Reference: its healthiest pixels (< 10 % dead neighbours), which look
    like Array42 (k 0.72 against 0.70), so delta(I) is what degradation added."""
    from matplotlib.patches import FancyBboxPatch
    names = list(groups)
    n0, g0 = groups[names[0]]
    i0, v0 = excess_points(g0)
    i0, v0 = np.r_[I_REF, i0], np.r_[0.0, v0]          # by definition 0 at I_REF
    ref = lambda i: np.interp(np.log(i), np.log(i0), v0, left=np.nan, right=np.nan)
    fig, ax = plt.subplots(2, 2, figsize=(15.5, 12.5))
    # (a) the model
    x = ax[0, 0]
    x.axis("off")
    boxes = [("junction", "V_j from the light (02b)\nmakes the light", NPG["teal"]),
             ("reference barrier", "V_ref(I): measured, FIXED\n(die, or healthy pixels)", "0.45"),
             ("2nd barrier", "a ln(1 + I/I_b)\nFITTED: a, I_b", NPG["red"]),
             ("resistor", "R I\nFITTED: R", NPG["tan"])]
    for k, (title, text, col) in enumerate(boxes):
        x0 = 0.02 + k * 0.245
        x.add_patch(FancyBboxPatch((x0, 0.70), 0.2, 0.2, boxstyle="round,pad=0.01", fc="w", ec=col,
                                   lw=2.2, transform=x.transAxes))
        x.text(x0 + 0.1, 0.86, title, ha="center", va="center", fontsize=9.5, weight="bold",
               color=col, transform=x.transAxes)
        x.text(x0 + 0.1, 0.76, text, ha="center", va="center", fontsize=7.5, transform=x.transAxes)
        if k < len(boxes) - 1:
            x.annotate("", xy=(x0 + 0.245, 0.80), xytext=(x0 + 0.205, 0.80), xycoords="axes fraction",
                       arrowprops=dict(arrowstyle="-", lw=2, color="0.3"))
    x.text(0.5, 0.97, "in series, the same current I through all four", ha="center", fontsize=9,
           transform=x.transAxes, color="0.3")
    x.text(0.0, 0.60,
           r"$V_{LED}(I) = V_j(I) + V_{ref}(I) + a\,\left[\ln(1+I/I_b) - \ln(1+I_0/I_b)\right]"
           r" + R\,(I - I_0)$", fontsize=12, transform=x.transAxes)
    x.text(0.0, 0.52,
           "I_0 = 0.6 uA: every excess term is counted from there (the excess is 0 at I_0).\n\n"
           "The 2nd barrier: an element that conducts I = I_b (exp(V/a) - 1), like a contact or\n"
           "an interface that carriers cross by thermionic emission or tunnelling:\n"
           "  - below I_b it acts like a resistor, R_0 = a / I_b\n"
           "  - above I_b it adds only a per e-fold of current (log-like, as the LED's own barrier)\n"
           "  - a: its log slope (26 mV for an ideal thermionic barrier, more for tunnelling)\n\n"
           "The alternative 'k' model has no new element: V_ex = k V_ref(I) + R (I - I_0),\n"
           "i.e. more (k > 1) or less (k < 1) of the reference's own barrier.\n\n"
           "Fitted per curve: a, I_b and/or R (and k), least squares on the excess.",
           fontsize=8.6, family="monospace", va="top", transform=x.transAxes)
    x.set_title("(a) The assumed model", fontsize=11, loc="left")
    # (b) the shape of a barrier against a resistor
    x = ax[0, 1]
    ii = np.linspace(0, 100e-6, 400)
    a_ = 0.1
    for ib, col in zip((1e-6, 5e-6, 20e-6, 100e-6), (NPG["teal"], NPG["lilac"], NPG["red"], NPG["brown"])):
        x.plot(ii * 1e6, 1e3 * a_ * np.log1p(ii / ib), "-", lw=2, color=col,
               label=f"barrier, a 100 mV, I_b {ib * 1e6:g} uA (R_0 = a/I_b = {a_ / ib / 1e3:.0f} kOhm)")
    x.plot(ii * 1e6, 1e3 * 1e3 * ii, "--", lw=1.6, color="k", label="resistor, 1 kOhm")
    x.axvspan(I_REF * 1e6, 90, color="0.92", zorder=0, label="the pixels' usable range, 0.6-90 uA")
    x.text(97, 20, "I_b >> range:\nstraight,\nlike R", ha="right", fontsize=8, color=NPG["brown"])
    x.text(14, 300, "I_b << range: bends early,\nthe LED barrier's own shape", fontsize=8,
           color=NPG["teal"])
    x.set(xlabel="current (uA)", ylabel="voltage across the element (mV)", xlim=(0, 100), ylim=(0, 500))
    x.set_title("(b) A barrier is only distinguishable from a resistor\n"
                "if it bends inside the measured range", fontsize=11, loc="left")
    x.legend(fontsize=7.2, loc="upper left")
    x.grid(alpha=.3)
    # (c) Array38's data, by dead neighbours
    x = ax[1, 0]
    cols = [NPG["navy"], NPG["teal"], NPG["salmon"], NPG["red"], NPG["darkred"]]
    rows = []
    fits_c = {}
    for name, col in zip(names[1:], cols[1:]):
        n, g = groups[name]
        i, v = excess_points(g)
        r = ref(i)
        m = np.isfinite(r)
        i, v, r = i[m], v[m], r[m]
        d = v - r
        f2 = fit_second("+ 2nd barrier", i, d, r)
        fits_c[name] = (i, d, r, n)
        x.plot(i * 1e6, 1e3 * d, "o", ms=6, color=col,
               label=f"{name} dead neighbours ({n} px): 2nd barrier a {1e3 * f2['p'][0]:.0f} mV, "
                     f"I_b {10 ** f2['p'][1] * 1e6:.2g} uA")
        xx = np.linspace(I_REF, i.max(), 200)
        x.plot(xx * 1e6, 1e3 * f2["f"](xx, ref(xx)), "-", color=col, lw=1.3)
        for model in SECOND:
            rr = fit_second(model, i, d, r)
            rows.append(dict(group=name, n=n, model=model, rms_mV=1e3 * rr["rms"],
                             **{f"p{k}": p for k, p in enumerate(rr["p"])},
                             **{f"se{k}": s for k, s in enumerate(rr["se"])}))
    i42, v42 = excess_points(c42)
    r42 = ref(i42)
    m = np.isfinite(r42)
    x.plot(i42[m] * 1e6, 1e3 * (v42[m] - r42[m]), "s--", ms=5, color="0.5", lw=1,
           label="Array42, median pixel (for comparison)")
    x.axhline(0, color="k", lw=.8)
    x.set(xlabel="current (uA)", ylabel="excess minus the reference's (mV)", xlim=(0, 92))
    x.set_title(f"(c) Array38: what degradation added, by the share of dead pixels nearby\n"
                f"reference: its healthiest pixels (< 10 % dead within {DEAD_WIN} px, {n0} px); "
                "lines: the 2nd-barrier fit", fontsize=10, loc="left")
    x.legend(fontsize=7, loc="upper left")
    x.grid(alpha=.3)
    # (d) all four models on the most degraded group
    x = ax[1, 1]
    name = names[-1]
    i, d, r, n = fits_c[name]
    x.plot(i * 1e6, 1e3 * d, "o", ms=8, color=cols[len(names) - 1], label=f"{name} dead neighbours, {n} px")
    xx = np.linspace(I_REF, i.max(), 200)
    three = None
    for model in SECOND:
        rr = fit_second(model, i, d, r)
        if model == "+ 2nd barrier + resistor":
            three = rr
        x.plot(xx * 1e6, 1e3 * rr["f"](xx, ref(xx)), "-", lw=1.8, color=SECOND_COL[model],
               label=describe_second(model, rr))
    x.axhline(0, color="k", lw=.8)
    rs = fit_second("+ resistor", i, d, r)
    x.text(0.98, 0.04,
           f"{len(i)} measured points. A resistor alone misses by {1e3 * rs['rms']:.0f} mV rms:\n"
           "the added voltage bends inside the range, so it is barrier-like.\n"
           f"With barrier AND resistor free, R comes out NEGATIVE\n"
           f"({three['p'][2] / 1e3:.2f} +- {three['se'][2] / 1e3:.2f} kOhm), which is unphysical: no resistor\n"
           "is added, and the extra bends slightly more than one log term.\n"
           "A new barrier and more of the reference's own barrier (k)\n"
           "fit equally well: 'barrier, not resistor' is clear, which\n"
           "barrier it is, is not.",
           transform=x.transAxes, fontsize=7.3, family="monospace", ha="right", va="bottom",
           bbox=dict(boxstyle="round", fc="w", alpha=.92))
    x.set(xlabel="current (uA)", ylabel="excess minus the reference's (mV)", xlim=(0, 92))
    x.set_title("(d) The four models fitted to the most degraded group", fontsize=11, loc="left")
    x.legend(fontsize=6.8, loc="upper left")
    x.grid(alpha=.3)
    save(fig, "17_second_barrier_explained.png", plt,
         "Data: " + SOURCE["Array38"] + f" Groups: share of dark pixels (open, short, dark_conducting "
         f"in the same run's fast scan) within {DEAD_WIN} x {DEAD_WIN} px; each group's median pixel per "
         "sweep bias, below the transistor limit. Array42: " + SOURCE["Array42"])
    t = pd.DataFrame(rows)
    t.to_csv(OUT / "fits_second_barrier.csv", index=False, float_format="%.5g")
    return t


def die_source_short():
    return "Die: SGA_15 10 um, IVL of 2023-08-18."


def fig_vs_rbest(arr, p, i_star, plt):
    """Two methods, one quantity: R_best from the fast scan (LED anchored at 1.5 uA, the array's
    best pixels as reference) against the sweep's excess above the best pixels, per current."""
    s = p.dropna(subset=["V_ex_Istar", "R_best_kOhm"])
    ref = float(s.loc[s.best, "V_ex_Istar"].median())
    y = ((s.V_ex_Istar - ref) / i_star / 1e3).to_numpy()
    x = s.R_best_kOhm.to_numpy()
    fig, ax = plt.subplots(figsize=(7.4, 6.8))
    ax.scatter(x, y, s=2, alpha=.08, color=WHOLE[arr], lw=0, rasterized=True)
    ax.plot([], [], "o", ms=4, alpha=.6, color=WHOLE[arr], label=f"{len(s)} swept pixels")
    q = pd.qcut(pd.Series(x), 25, duplicates="drop")
    g = pd.DataFrame({"x": x, "y": y}).groupby(q.values, observed=True)
    ax.plot(g.x.median(), g.y.median(), "o-", ms=4, color="k", lw=1.2,
            label="median in 25 bins of R_best")
    lo = float(min(np.nanquantile(x, .002), np.nanquantile(y, .002)))
    hi = float(max(np.nanquantile(x, .998), np.nanquantile(y, .998)))
    ax.plot([lo, hi], [lo, hi], "--", color=NPG["red"], lw=1.4, label="1:1")
    rho = pd.Series(x).corr(pd.Series(y), method="spearman")
    body = (x > np.nanquantile(x, .05)) & (x < np.nanquantile(x, .95))
    ax.text(0.03, 0.97,
            f"Spearman {rho:.2f}\n"
            f"sweep minus fast scan, 5-95 % of R_best: median {np.nanmedian((y - x)[body]):+.2f} kOhm\n\n"
            "x: fast scan at -1.40 V, LED anchored at 1.5 uA (bond_resistance.py)\n"
            f"y: sweep, junction from the light, at {i_star * 1e6:.0f} uA\n"
            "both: above the array's best 2 %, per current",
            transform=ax.transAxes, fontsize=7.2, family="monospace", va="top",
            bbox=dict(boxstyle="round", fc="w", alpha=.92))
    ax.set_xscale("symlog", linthresh=1)
    ax.set_yscale("symlog", linthresh=1)
    ax.set(xlim=(lo, hi), ylim=(lo, hi),
           xlabel="R_best from the fast scan (kOhm)",
           ylabel=f"(excess - best pixels' excess) / {i_star * 1e6:.0f} uA from the sweep (kOhm)",
           title=f"{arr}: two methods, one quantity\n"
                 "the fast scan's R_best against the sweep's light-based excess")
    ax.grid(alpha=.3, which="both")
    ax.legend(fontsize=7.5, loc="lower right")
    save(fig, f"14_Rbest_vs_sweep_{arr}.png", plt,
         "Data: " + SOURCE[arr] + f" R_best: its fast scan at -1.40 V (bond_resistance.py).")


# ---- main ---------------------------------------------------------------------------------
def main() -> int:
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    OUT.mkdir(parents=True, exist_ok=True)
    for f in OUT.glob("*.png"):                      # names change between versions
        f.unlink()
    plt = aa._plt()
    drv = {a: br.fit_driver(a) for a in ARRAYS}
    die = load_die()
    curves, bests, tots, pp, gs, istar = {}, {}, {}, {}, {}, {}
    D = die_excess(die)                              # the die's excess as measured
    for a in ARRAYS:
        print(f"{a}: loading the sweep ...", flush=True)
        rb = pd.read_csv(br.OUT / "pixels" / f"{br.REFERENCE_RUN[a]}_m1p40V.csv").set_index(["X", "Y"])
        sw, tots[a] = load_sweep_points(a, drv[a], rb)
        best = br.best_reference(a, rb.I_LED)[2]
        curves[a] = curve(sw, drv[a])
        bests[a] = curve(sw, drv[a], best)
        istar[a] = float(rb.I_LED.median())          # the fast scan's median current at -1.40 V
        pp[a], gs[a] = per_pixel(sw, die, best, rb, istar[a], D)
        if a == "Array38":                           # figure 17: groups by dead neighbours
            dm38 = dead_map(a)
            groups38 = dead_groups(sw, drv[a], dm38)
        SOURCE[a] = (f"{a}: NVLED sweep {br.REFERENCE_RUN[a]}, {sw.attrs['n_pixels']} pixels "
                     f"(best 2 %: {len(best)}; {sw.attrs['n_latched']} latched pixels and "
                     f"{sw.attrs['n_lit_ref']} partly lit references left out).")
        print("  " + SOURCE[a], flush=True)
        del sw
    print("fast-scan medians ...", flush=True)
    fsm = pd.DataFrame([fs_median(a, r, v) for a in ARRAYS for r, v in FASTSCANS[a]])
    fsm.to_csv(OUT / "fastscan_medians.csv", index=False, float_format="%.6g")

    # fits: every curve, every model; the window test
    fits_, frows, wrows = {}, [], []
    e = die[die.light & (die.Current_A >= I_REF)]
    xy = {"SGA_15 die": (e.Current_A.to_numpy(), e.V_ex.to_numpy())}
    for a in ARRAYS:
        xy[a] = excess_points(curves[a])
        xy[f"{a} best 2 %"] = excess_points(bests[a])
    for name, (i, v) in xy.items():
        for model in MODELS:
            r = fit(i, v, model)
            fits_[(name, model)] = r
            frows.append(dict(curve=name, model=model, n=r["n"], rms_mV=1e3 * r["rms"],
                              **{f"p{k}": x for k, x in enumerate(r["p"])},
                              **{f"se{k}": x for k, x in enumerate(r["se"])}))
            print(f"  {name}: {describe(model, r)}")
        if "best" in name:
            continue
        for top in (20e-6, 40e-6, 70e-6, 120e-6, 200e-6, 300e-6, 420e-6):
            m = i <= top * 1.02
            if m.sum() >= 5 and i[m].max() >= 0.8 * top:
                r = fit(i[m], v[m], "barrier + resistor")
                wrows.append(dict(curve=name, i_max_uA=top * 1e6, R_kOhm=r["p"][2] / 1e3,
                                  R_se_kOhm=r["se"][2] / 1e3, a_mV=1e3 * r["p"][0], n=int(m.sum())))
    pd.DataFrame(frows).to_csv(OUT / "fits.csv", index=False, float_format="%.5g")
    pd.DataFrame(wrows).to_csv(OUT / "fit_window.csv", index=False, float_format="%.4g")
    models = {}
    for a in ARRAYS:
        for d, name in ((curves[a], a), (bests[a], f"{a} best 2 %")):
            add_estimate(d, fits_[(name, "barrier + resistor")])
        models[a] = led_model(curves[a], fits_[(a, "barrier + resistor")])
    pd.concat([d.assign(curve=k, pixels="all") for k, d in curves.items()]
              + [d.assign(curve=k, pixels="best 2 %") for k, d in bests.items()]
              + [die.assign(curve="SGA_15 die")], ignore_index=True).to_csv(
        OUT / "curves.csv", index=False, float_format="%.6g")
    pd.concat([g.assign(array=a) for a, g in gs.items()]).to_csv(OUT / "grid_stats.csv", index=False,
                                                               float_format="%.5g")
    for a in ARRAYS:
        pp[a].to_csv(OUT / f"per_pixel_{a}.csv", index=False, float_format="%.5g")

    # the pixel as the die's LED plus a bond term (die's excess fixed as measured)
    brows = []
    for a in ARRAYS:
        for d, px in ((curves[a], "median"), (bests[a], "best 2 %")):
            i, v = excess_points(d)
            for m in BOND:
                r = bond_fit(D, i, v, m)
                brows.append(dict(array=a, pixels=px, model=m, n=r["n"], rms_mV=1e3 * r["rms"],
                                  **{f"p{k}": x for k, x in enumerate(r["p"])},
                                  **{f"se{k}": x for k, x in enumerate(r["se"])}))
                print(f"  {a} {px}: {describe_bond(m, r)}")
    pd.DataFrame(brows).to_csv(OUT / "fits_bond.csv", index=False, float_format="%.5g")

    dv20 = die_at(die, 20e-6)
    for a in ARRAYS:
        fig_driver(a, drv[a], curves[a], models[a], plt)
        fig_subtraction(a, curves[a], drv[a], plt)
        fig_light_inversion(a, curves[a], plt)
        fig_iv(a, curves[a], bests[a], tots[a], die, fsm, plt, log=True)
        fig_iv(a, curves[a], bests[a], tots[a], die, fsm, plt, log=False)
        fig_budget(a, curves[a], plt)
        fig_r_vs_v(a, curves[a], drv[a], plt)
        fig_bond_fits(a, curves[a], D, plt)
        fig_map(a, pp[a], dv20, plt)
        fig_vs_rbest(a, pp[a], istar[a], plt)
        fig_barrier_maps(a, pp[a], plt)
    ref42 = {"k": float(pp["Array42"].k_barrier.median()), "R": float(pp["Array42"].R_extra_kOhm.median())}
    fig_vs_dark("Array38", pp["Array38"], ref42, dm38, plt)
    t17 = fig_second_barrier(groups38, curves["Array42"], plt)
    print("  second barrier (Array38, against its healthiest pixels):")
    for _, r in t17.iterrows():
        print(f"    {r.group:>9s} {r.model:34s} rms {r.rms_mV:5.1f} mV  p0 {r.p0:.4g}  p1 {r.get('p1', np.nan):.4g}"
              f"  p2 {r.get('p2', np.nan):.4g}")
    for a in ARRAYS:
        q = pp[a][["k_barrier", "R_extra_kOhm", "fit_rms_mV", "n_fit"]].describe(percentiles=[.1, .5, .9])
        print(f"  {a} per-pixel fits:\n{q.round(3).to_string()}")
    for name, i_, dvv in fig_droop(curves, die, plt):
        print(f"  droop, {name}: junction from the light minus from the current "
              f"{dvv.min():+.1f} to {dvv.max():+.1f} mV over {i_.min() * 1e6:.1f}-{i_.max() * 1e6:.0f} uA")
    fig_ideality(curves, die, plt)
    fig_dvdi(curves, die, drv, plt)
    fig_minus_die(gs, die, plt)
    fig_excess(gs, die, plt)
    fig_window(wrows, plt, die)
    a, b = fig_cheung(curves, plt)
    print(f"die Cheung fit 0.35-0.9 mA: intercept {a:.3f} V (n {a / VT:.1f}), slope {b:.0f} Ohm")
    i, v = xy["SGA_15 die"]
    rb_ = fits_[("SGA_15 die", "barrier + resistor")]
    fig_fits("SGA_15 die", i, v, {m: fits_[("SGA_15 die", m)] for m in MODELS}, plt, die_source(die),
             note=("The resistor here: the die's excess (junction removed via\n"
                   f"the light) over 0.6-{i.max() * 1e6:.0f} uA gives R {rb_['p'][2] / 1e3:.2f} kOhm.\n"
                   f"Cheung on its IV sweep at 0.35-0.85 mA (11): {b:.0f} Ohm, n {a / VT:.1f};\n"
                   "its own analysis: 270 Ohm, n 10.4. The light ends at 3.6 V\n"
                   "(0.42 mA), so this fit cannot reach that range. Below it\n"
                   "the barrier still curves, and R depends on the range (10)."))
    fig_hist(pp, die, plt)
    for a in ARRAYS:
        for d, name in ((curves[a], "all pixels"), (bests[a], "best 2 %")):
            print(f"\n{a}, {name}:")
            print(d[["V_tot", "n", "I", "V_drv", "V_LED", "V_j", "V_ex", "ok", "V_LED_est", "V_drv_est"]]
                  .assign(I=lambda t: t.I * 1e6).round(4).to_string(index=False))
    for a in ARRAYS:
        print(f"\n{a} grid:")
        print(gs[a].assign(I=lambda t: t.I * 1e6).round(4).to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
