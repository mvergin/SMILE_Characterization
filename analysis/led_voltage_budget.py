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
SOURCE = {}                  # per array: the data line printed under its figures


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


def per_pixel(sw, die, best, rb):
    """Every pixel: V_LED and light at I_REF, TARGETS and GRID (log-interpolated between its
    own sweep points), its excess there, and its V_LED against the die's at the same current.
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
        ax.set(yscale="log", ylim=(1e-5, 400), xlim=(1.8, 5.1))
        ax.annotate("below ~2.45 V the LED's own current is\nfar under the CMOS leakage: the leakage\n"
                    "reference was measured from NVLED -0.7 V on", (1.85, 5e-3), fontsize=7, color="0.4")
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


def fig_fits(name, i, v, fits_, plt, source):
    fig, (ax, rx) = plt.subplots(2, 1, figsize=(7.8, 7.2), sharex=True,
                                 gridspec_kw=dict(height_ratios=[3, 1.3]))
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


def fig_hist(pp, die, plt):
    fig, ax = plt.subplots(figsize=(7.8, 5.4))
    e = die[die.light & (die.Current_A >= I_REF)]
    t = 20e-6
    dv = 1e3 * interp_log(e.Current_A.to_numpy(), np.array([t]), e.V_ex.to_numpy())[0]
    for arr in ARRAYS:
        s = 1e3 * pp[arr][f"V_ex_{tag(t)}"].dropna()
        sb = 1e3 * pp[arr].loc[pp[arr].best, f"V_ex_{tag(t)}"].dropna()
        ax.hist(s.clip(0, 600), bins=150, histtype="step", lw=1.6, color=WHOLE[arr],
                label=f"{arr}: {len(s)} pixels, median {s.median():.0f} mV, "
                      f"25-75 % {s.quantile(.25):.0f}-{s.quantile(.75):.0f}")
        ax.axvline(sb.median(), color=WHOLE[arr], ls="--", lw=1.2,
                   label=f"{arr}: median of the best 2 %, {sb.median():.0f} mV")
    ax.axvline(dv, color=DIE, lw=2, label=f"SGA_15 die: {dv:.0f} mV")
    ax.set(xlabel=f"excess voltage at {t * 1e6:g} uA (mV), clipped to 0-600", ylabel="pixels",
           yscale="log", title="Excess voltage per swept pixel at 20 uA, against the die")
    ax.grid(alpha=.3)
    ax.legend(fontsize=7.5)
    save(fig, "12_excess_per_pixel.png", plt, "Data: " + SOURCE["Array42"] + " " + SOURCE["Array38"]
         + " " + die_source(die))


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
    fig, ax = plt.subplots(figsize=(7.4, 6.6))
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
    save(fig, f"13_excess_map_{arr}.png", plt, "Data: " + SOURCE[arr])


def fig_vs_rbest(pp, plt):
    fig, ax = plt.subplots(figsize=(7.8, 5.6))
    col = f"V_ex_{tag(20e-6)}"
    for arr in ARRAYS:
        s = pp[arr].dropna(subset=[col, "R_best_kOhm"])
        q = pd.qcut(s.R_best_kOhm, 20, duplicates="drop")
        g = s.groupby(q, observed=True)
        x = g.R_best_kOhm.median()
        y = 1e3 * g[col].median()
        lo, hi = 1e3 * g[col].quantile(.25), 1e3 * g[col].quantile(.75)
        r = s.R_best_kOhm.corr(s[col], method="spearman")
        ax.errorbar(x, y, yerr=[y - lo, hi - y], fmt="o-", ms=4, capsize=2, color=WHOLE[arr],
                    label=f"{arr}: {len(s)} pixels, Spearman {r:.2f}")
    ax.set(xscale="symlog", xlabel="R_best: series resistance above the array's best pixels (kOhm, 20 bins)",
           ylabel="excess voltage at 20 uA (mV; median, 25-75 %)",
           title="Does the excess voltage follow the per-pixel resistance R_best?")
    ax.grid(alpha=.3, which="both")
    ax.legend(fontsize=8)
    save(fig, "14_excess_vs_Rbest.png", plt,
         "Data: excess from " + SOURCE["Array42"] + " " + SOURCE["Array38"]
         + " R_best: the same runs' fast scans at -1.40 V (bond_resistance.py).")


# ---- main ---------------------------------------------------------------------------------
def main() -> int:
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    OUT.mkdir(parents=True, exist_ok=True)
    for f in OUT.glob("*.png"):                      # names change between versions
        f.unlink()
    plt = aa._plt()
    drv = {a: br.fit_driver(a) for a in ARRAYS}
    die = load_die()
    curves, bests, tots, pp, gs = {}, {}, {}, {}, {}
    for a in ARRAYS:
        print(f"{a}: loading the sweep ...", flush=True)
        rb = pd.read_csv(br.OUT / "pixels" / f"{br.REFERENCE_RUN[a]}_m1p40V.csv").set_index(["X", "Y"])
        sw, tots[a] = load_sweep_points(a, drv[a], rb)
        best = br.best_reference(a, rb.I_LED)[2]
        curves[a] = curve(sw, drv[a])
        bests[a] = curve(sw, drv[a], best)
        pp[a], gs[a] = per_pixel(sw, die, best, rb)
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

    for a in ARRAYS:
        fig_driver(a, drv[a], curves[a], models[a], plt)
        fig_iv(a, curves[a], bests[a], tots[a], die, fsm, plt, log=True)
        fig_iv(a, curves[a], bests[a], tots[a], die, fsm, plt, log=False)
        fig_budget(a, curves[a], plt)
        fig_r_vs_v(a, curves[a], drv[a], plt)
        fig_map(a, pp[a], plt)
    fig_ideality(curves, die, plt)
    fig_dvdi(curves, die, drv, plt)
    fig_minus_die(gs, die, plt)
    fig_excess(gs, die, plt)
    for name in ("SGA_15 die",) + ARRAYS:
        i, v = xy[name]
        src = die_source(die) if name.startswith("SGA") else SOURCE[name] + " Median pixel."
        fig_fits(name, i, v, {m: fits_[(name, m)] for m in MODELS}, plt, src)
    fig_window(wrows, plt, die)
    a, b = fig_cheung(curves, plt)
    print(f"die Cheung fit 0.35-0.9 mA: intercept {a:.3f} V (n {a / VT:.1f}), slope {b:.0f} Ohm")
    fig_hist(pp, die, plt)
    fig_vs_rbest(pp, plt)
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
