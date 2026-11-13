"""The dense Array42 run (261113-1356): 400 pixels of the 1703 sweep, 0.05 V steps from
-0.85 to -1.9 V (configs/runs/run5_array42_dense.yaml).

    python analysis/led_voltage_dense.py        (after led_voltage_budget.py)

Same method as led_voltage_budget.py (transistor model, junction from the light, excess over
it), on ~20 points per pixel between 0.6 uA and the transistor limit instead of 6:
  - the data against 1703 for the same pixels (current, light, excess);
  - which model describes the excess of the median pixel, now with enough points for the
    three- and four-parameter ones;
  - the fit-window test (10) for pixels;
  - per pixel: barrier k and resistor R against the die, from 20 points against 1703's 6.

Writes results/comparisons/led_voltage/: 18_dense_median.png, 18b_dense_terms.png,
19_dense_per_pixel.png,
per_pixel_dense.csv, fits_dense.csv, fit_window_dense.csv, checks_dense.csv.
"""
from __future__ import annotations

import sys
import warnings

import numpy as np
import pandas as pd
from scipy.optimize import least_squares

import analyze_array as aa
import bond_resistance as br
import led_voltage_budget as lv

RUN = "261113-1356_Array42_LIST-a42_dense"
LIST = br.REPO / "configs" / "runs" / "a42_dense.csv"
OUT = lv.OUT
IB_GRID = np.geomspace(0.3e-6, 300e-6, 61)
GROUP_COL = {"best 2 %": lv.NPG["teal"], "random": lv.NPG["navy"],
             "top 1 % excess at 20 uA": lv.NPG["red"]}


def fastscan_rb(run):
    """The run's own fast scan at -1.40 V as (X, Y) -> I_LED, for the latch check. Its CMOS
    leakage (~0.19 uA) is 0.5 % of the 37 uA there: left in."""
    fs = aa._load_aggregated(br.run_dir(run) / "FastScan")
    fs = fs[fs.NVLED_V.astype(float).round(2) == -1.4]
    return pd.DataFrame({"I_LED": fs.NVLED_CURR_MEAN.astype(float).abs().to_numpy()},
                        index=pd.MultiIndex.from_arrays([fs.X, fs.Y]))


# models for the excess V_ex(I); D is the die's measured excess, g the barrier term
def g(i, lb):
    return lv._g(i, lb)


MODELS = {
    "resistor": (lambda D, p, i: p[0] * (i - lv.I_REF), [3e3], ([-1e5], [1e6])),
    "barrier": (lambda D, p, i: p[0] * g(i, p[1]), [0.1, -5.5], ([0, -9], [2, -2])),
    "barrier + resistor": (lambda D, p, i: p[0] * g(i, p[1]) + p[2] * (i - lv.I_REF),
                           [0.1, -5.5, 500], ([0, -9, -2e4], [2, -2, 1e5])),
    "die x k + resistor": (lambda D, p, i: p[0] * D(i) + p[1] * (i - lv.I_REF), [1.0, 500.],
                           ([0.0, -2e4], [5, 5e4])),
    "die + 2nd barrier + resistor": (lambda D, p, i: D(i) + p[0] * g(i, p[1]) + p[2] * (i - lv.I_REF),
                                     [0.05, -5.0, 500.], ([0, -9, -2e4], [2, -2, 5e4])),
    "die x k + 2nd barrier + resistor": (lambda D, p, i: p[0] * D(i) + p[1] * g(i, p[2])
                                         + p[3] * (i - lv.I_REF),
                                         [0.8, 0.02, -5.0, 500.], ([0, 0, -9, -2e4], [5, 2, -2, 5e4])),
}
MODEL_COL = {"resistor": lv.NPG["tan"], "barrier": lv.NPG["lilac"],
             "barrier + resistor": lv.NPG["brown"], "die x k + resistor": lv.NPG["teal"],
             "die + 2nd barrier + resistor": lv.NPG["red"],
             "die x k + 2nd barrier + resistor": lv.NPG["navy"]}


def fit(D, i, v, model):
    f, p0, bounds = MODELS[model]
    r = least_squares(lambda p: f(D, p, i) - v, p0, bounds=bounds)
    cov = np.linalg.pinv(r.jac.T @ r.jac) * (r.fun @ r.fun) / max(1, len(i) - len(p0))
    return dict(p=r.x, se=np.sqrt(np.abs(np.diag(cov))), rms=float(np.sqrt(np.mean(r.fun ** 2))),
                n=len(i), f=lambda x, p=r.x: f(D, p, x))


def describe(model, r):
    p, se = r["p"], r["se"]
    kohm = lambda k: f"R {p[k] / 1e3:.2f} +- {se[k] / 1e3:.2f} kOhm"
    bar = lambda ka, kb: f"a {1e3 * p[ka]:.0f} +- {1e3 * se[ka]:.0f} mV, I_b {10 ** p[kb] * 1e6:.2g} uA"
    s = {"resistor": lambda: kohm(0),
         "barrier": lambda: bar(0, 1),
         "barrier + resistor": lambda: bar(0, 1) + ", " + kohm(2),
         "die x k + resistor": lambda: f"k {p[0]:.3f} +- {se[0]:.3f}, " + kohm(1),
         "die + 2nd barrier + resistor": lambda: bar(0, 1) + ", " + kohm(2),
         "die x k + 2nd barrier + resistor": lambda: f"k {p[0]:.2f} +- {se[0]:.2f}, " + bar(1, 2)
         + ", " + kohm(3)}[model]()
    return f"{model}: {s}; rms {1e3 * r['rms']:.2f} mV"


def pixel_fits(sw, D):
    """Per pixel, on its own points from I_REF to the transistor limit (anchored at I_REF):
    die x k + R (linear) and die + 2nd barrier + R (I_b on a grid, a >= 0, linear in a, R)."""
    rows = []
    for (x, y), s in sw[sw.ok & sw.light & (sw.I_LED > 0)].groupby(["X", "Y"]):
        s = s.sort_values("I_LED")
        i, vl, lp = s.I_LED.to_numpy(), s.V_LED.to_numpy(), np.log(s.P.to_numpy())
        if (i < lv.I_REF).sum() < 1 or (i >= lv.I_REF).sum() < 6:
            continue
        v0 = np.interp(np.log(lv.I_REF), np.log(i), vl)
        l0 = np.interp(np.log(lv.I_REF), np.log(i), lp)
        m = i >= lv.I_REF
        i, vex = i[m], (vl[m] - v0) - lv.N_R * lv.VT * (lp[m] - l0)
        d, xr = D(i), i - lv.I_REF
        A = np.c_[d, xr]
        (k, R), *_ = np.linalg.lstsq(A, vex, rcond=None)
        rms_k = float(np.sqrt(np.mean((vex - A @ [k, R]) ** 2)))
        def grid(target):
            """a ln-barrier (I_b on the grid, a >= 0) + R fitted to target; (rms, a, I_b, R)."""
            best = (np.inf, np.nan, np.nan, np.nan)
            for ib in IB_GRID:
                gg = np.log1p(i / ib) - np.log1p(lv.I_REF / ib)
                (a, r2), *_ = np.linalg.lstsq(np.c_[gg, xr], target, rcond=None)
                if a < 0:                             # a barrier only adds voltage
                    a, r2 = 0.0, float(np.dot(xr, target) / np.dot(xr, xr))
                rms = float(np.sqrt(np.mean((target - a * gg - r2 * xr) ** 2)))
                if rms < best[0]:
                    best = (rms, a, ib, r2)
            return best
        fb, db = grid(vex), grid(vex - d)             # free barrier + R; die + 2nd barrier + R
        rows.append(dict(X=x, Y=y, n=len(i), I_max_uA=1e6 * i.max(), V_ex_20uA=float(np.interp(
            np.log(20e-6), np.log(i), vex, left=np.nan, right=np.nan)),
            k=k, R_kOhm=R / 1e3, rms_k_mV=1e3 * rms_k,
            a_free_mV=1e3 * fb[1], Ib_free_uA=1e6 * fb[2], R_free_kOhm=fb[3] / 1e3, rms_free_mV=1e3 * fb[0],
            a_2b_mV=1e3 * db[1], Ib_2b_uA=1e6 * db[2], R_2b_kOhm=db[3] / 1e3, rms_2b_mV=1e3 * db[0]))
    return pd.DataFrame(rows).set_index(["X", "Y"])


def main() -> int:
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    OUT.mkdir(parents=True, exist_ok=True)
    plt = aa._plt()
    P = br.fit_driver("Array42")
    die = lv.load_die()
    D = lv.die_excess(die)
    lst = pd.read_csv(LIST).set_index(["X", "Y"])
    rb03 = pd.read_csv(br.OUT / "pixels" / f"{br.REFERENCE_RUN['Array42']}_m1p40V.csv").set_index(["X", "Y"])
    sw, _ = lv.load_sweep_points("Array42", P, fastscan_rb(RUN), run=RUN)
    old, _ = lv.load_sweep_points("Array42", P, rb03)
    old = old[pd.MultiIndex.from_arrays([old.X, old.Y]).isin(lst.index)]
    src = (f"Dense run {RUN[:11]}: {sw.attrs['n_pixels']} pixels ({sw.attrs['n_latched']} latched and "
           f"{sw.attrs['n_lit_ref']} partly lit references left out); the same pixels in 1703.")
    print(src)

    # ---- checks against 1703 -------------------------------------------------------------
    j = sw.merge(old, on=["X", "Y", "V"], suffixes=("", "_o"))
    chk = []
    for v, gq in j.groupby("V"):
        ok = (gq.I_LED > 0) & (gq.I_LED_o > 0)
        okp = (gq.P > 5 * gq.P_noise) & (gq.P_o > 5 * gq.P_noise_o)
        chk.append(dict(V=v, n=len(gq), I_new_uA=1e6 * gq.I_LED.median(),
                        I_ratio=(gq.I_LED / gq.I_LED_o)[ok].median(),
                        P_ratio=(gq.P / gq.P_o)[okp].median() if okp.any() else np.nan,
                        EQE_ratio=((gq.P / gq.I_LED) / (gq.P_o / gq.I_LED_o))[ok & okp].median()
                        if (ok & okp).any() else np.nan,
                        Iref_new_nA=1e9 * gq.I_ref.median(), Iref_old_nA=1e9 * gq.I_ref_o.median()))
    chk = pd.DataFrame(chk)
    chk.to_csv(OUT / "checks_dense.csv", index=False, float_format="%.5g")
    print(chk.round(4).to_string(index=False))

    # ---- median pixel: excess and models ------------------------------------------------------
    cd = lv.curve(sw, P)
    co = lv.curve(old, P)
    i, v = lv.excess_points(cd)
    io, vo = lv.excess_points(co)
    fits = {m: fit(D, i, v, m) for m in MODELS}
    frows = [dict(curve="dense median", model=m, n=r["n"], rms_mV=1e3 * r["rms"],
                  **{f"p{k}": x for k, x in enumerate(r["p"])},
                  **{f"se{k}": x for k, x in enumerate(r["se"])}) for m, r in fits.items()]
    for m, r in fits.items():
        print("  " + describe(m, r))
    # the fit-window test, now with enough points
    wrows = []
    for top in (10e-6, 20e-6, 30e-6, 40e-6, 50e-6, 60e-6, 70e-6, 80e-6, 90e-6):
        m = i <= top * 1.02
        if m.sum() >= 5 and i[m].max() >= 0.8 * top:
            r = fit(D, i[m], v[m], "barrier + resistor")
            wrows.append(dict(curve="Array42 dense", i_max_uA=top * 1e6, n=int(m.sum()),
                              R_kOhm=r["p"][2] / 1e3, R_se_kOhm=r["se"][2] / 1e3, a_mV=1e3 * r["p"][0]))
            rk = fit(D, i[m], v[m], "die x k + resistor")
            wrows[-1].update(k=rk["p"][0], R_k_kOhm=rk["p"][1] / 1e3, R_k_se_kOhm=rk["se"][1] / 1e3)
    win = pd.DataFrame(wrows)
    win_die = pd.read_csv(lv.OUT / "fit_window.csv")
    print(win.round(3).to_string(index=False))
    groups = {}
    for name, idx in lst.groupby("group").groups.items():
        cg = lv.curve(sw, P, idx)
        gi, gv = lv.excess_points(cg)
        groups[name] = (len(idx), gi, gv, fit(D, gi, gv, "die x k + resistor"))
        frows.append(dict(curve=f"dense {name}", model="die x k + resistor", n=len(gi),
                          rms_mV=1e3 * groups[name][3]["rms"],
                          p0=groups[name][3]["p"][0], p1=groups[name][3]["p"][1],
                          se0=groups[name][3]["se"][0], se1=groups[name][3]["se"][1]))
    pd.DataFrame(frows).to_csv(OUT / "fits_dense.csv", index=False, float_format="%.5g")
    win.to_csv(OUT / "fit_window_dense.csv", index=False, float_format="%.4g")

    fig, ax = plt.subplots(2, 2, figsize=(15.5, 12.5))
    x = ax[0, 0]
    e = die[die.light & (die.Current_A >= lv.I_REF)]
    x.plot(e.Current_A * 1e6, 1e3 * e.V_ex, "-", color=lv.DIE, lw=1.8, label=lv.DIE_LABEL)
    x.plot(i * 1e6, 1e3 * v, "o", ms=6, color=lv.NPG["navy"],
           label=f"dense run, median of {sw.attrs['n_pixels']} pixels ({len(i)} points)")
    x.plot(io * 1e6, 1e3 * vo, "s", ms=8, mfc="none", mew=1.6, color=lv.NPG["red"],
           label=f"1703, the same pixels ({len(io)} points)")
    for name, (n, gi, gv, r) in groups.items():
        x.plot(gi * 1e6, 1e3 * gv, "-", lw=1, color=GROUP_COL.get(name, "0.5"), alpha=.8,
               label=f"dense, {name} ({n} px): k {r['p'][0]:.2f}, R {r['p'][1] / 1e3:.2f} kOhm")
    x.set(xscale="log", xlabel="current (uA)", ylabel="excess voltage over the junction (mV)",
          xlim=(0.5, 500), title="(a) The excess, densely sampled, against 1703 and the die")
    x.legend(fontsize=7, loc="upper left")
    x.grid(alpha=.3, which="both")
    x = ax[0, 1]
    for m, r in fits.items():
        x.plot(i * 1e6, 1e3 * (v - r["f"](i)), "o-", ms=3.5, lw=1.2, color=MODEL_COL[m],
               label=describe(m, r))
    x.axhline(0, color="k", lw=.7)
    x.set(xscale="log", xlabel="current (uA)", ylabel="data - fit (mV)", ylim=(-12, 12),
          title=f"(b) Which model describes the median pixel's excess? ({len(i)} points)")
    x.legend(fontsize=6.4, loc="lower left")
    x.grid(alpha=.3, which="both")
    x = ax[1, 0]
    s = win_die[win_die.curve == "SGA_15 die"]
    x.errorbar(s.i_max_uA, s.R_kOhm, yerr=s.R_se_kOhm, fmt="o-", capsize=3, color=lv.DIE,
               label="die: barrier + resistor, free")
    x.errorbar(win.i_max_uA, win.R_kOhm, yerr=win.R_se_kOhm, fmt="s-", capsize=3, color=lv.NPG["brown"],
               label="Array42 dense: barrier + resistor, free")
    x.errorbar(win.i_max_uA, win.R_k_kOhm, yerr=win.R_k_se_kOhm, fmt="D-", capsize=3, color=lv.NPG["teal"],
               label="Array42 dense: die x k + resistor (R on top of the die's own)")
    s42 = win_die[win_die.curve == "Array42"]
    x.plot(s42.i_max_uA, s42.R_kOhm, "x", ms=10, mew=2, color=lv.NPG["red"], label="Array42, 1703 (6 points)")
    x.axhline(0.27, color=lv.NPG["brown"], ls=":", label="die's series R, Cheung (270 Ohm)")
    x.set(xscale="log", xlabel="upper end of the fit window (uA); lower end 0.6 uA",
          ylabel="fitted series resistance (kOhm)",
          title="(c) The fit-window test with enough points")
    x.legend(fontsize=7.5)
    x.grid(alpha=.3, which="both")
    x = ax[1, 1]
    x.plot(win.i_max_uA, win.k, "D-", color=lv.NPG["teal"], label="Array42 dense: k of die x k + resistor")
    x.axhline(1, color=lv.DIE, lw=1, label="the die (k = 1)")
    x.set(xscale="log", xlabel="upper end of the fit window (uA)", ylabel="barrier k (1 = the die's)",
          title="(d) The barrier ratio k against the fit window")
    x.legend(fontsize=8)
    x.grid(alpha=.3, which="both")
    lv.save(fig, "18_dense_median.png", plt, src + " Transistor model as in 00; junction from the light.")
    fig, ax = plt.subplots(2, 2, figsize=(14, 10.5))
    for x, m in zip(ax.flat, list(MODELS)[2:]):          # the models with more than one term
        lv.plot_parts(x, i, v, m, fits[m], D)
    fig.suptitle(f"Array42 dense run, median pixel ({len(i)} points): the models of 18(b) split into "
                 "their terms\nV_die: the SGA_15 die's excess, measured and fixed", fontsize=11)
    lv.save(fig, "18b_dense_terms.png", plt, src + " Transistor model as in 00; junction from the light.")

    # ---- per pixel: 20 points against 1703's 6 ------------------------------------------------
    pp = pixel_fits(sw, D)
    p03 = pd.read_csv(lv.OUT / "per_pixel_Array42.csv").set_index(["X", "Y"])
    pp = pp.join(p03[["k_barrier", "R_extra_kOhm", "n_fit"]].rename(
        columns={"k_barrier": "k_1703", "R_extra_kOhm": "R_1703_kOhm", "n_fit": "n_1703"})).join(lst[["group"]])
    pp.reset_index().to_csv(OUT / "per_pixel_dense.csv", index=False, float_format="%.5g")
    q = pp[["n", "k", "R_kOhm", "rms_k_mV", "a_free_mV", "Ib_free_uA", "R_free_kOhm", "rms_free_mV",
            "rms_2b_mV"]].describe(percentiles=[.1, .5, .9]).round(3)
    print(q.to_string())
    fig, ax = plt.subplots(1, 3, figsize=(19, 6))
    for x, (col, c03, lab) in zip(ax[:2], (("k", "k_1703", "barrier k (1 = the die's)"),
                                           ("R_kOhm", "R_1703_kOhm", "extra resistor R (kOhm)"))):
        s = pp.dropna(subset=[col, c03])
        for name, gq in s.groupby("group"):
            x.plot(gq[c03], gq[col], "o", ms=3.5, alpha=.7, color=GROUP_COL.get(name, "0.5"),
                   label=f"{name} ({len(gq)})")
        lo, hi = np.nanpercentile(np.r_[s[col], s[c03]], [1, 99])
        x.plot([lo, hi], [lo, hi], "k--", lw=1, label="1:1")
        rho = s[col].corr(s[c03], method="spearman")
        d = s[col] - s[c03]
        x.set(xlim=(lo, hi), ylim=(lo, hi), xlabel=f"{lab}, 1703 ({s.n_1703.median():.0f} points)",
              ylabel=f"{lab}, dense run ({s.n.median():.0f} points)",
              title=f"{lab}: Spearman {rho:.2f}; dense - 1703: median {d.median():+.3f}, "
                    f"25-75 % {d.quantile(.25):+.3f} to {d.quantile(.75):+.3f}")
        x.title.set_fontsize(9)
        x.legend(fontsize=7.5)
        x.grid(alpha=.3)
    x = ax[2]
    s = pp.dropna(subset=["rms_k_mV", "rms_free_mV", "rms_2b_mV"])
    for col, c, lab in (("rms_k_mV", lv.NPG["teal"], "die x k + R (2 parameters)"),
                        ("rms_free_mV", lv.NPG["brown"], "barrier + R, free (3 parameters)"),
                        ("rms_2b_mV", lv.NPG["red"], "die + 2nd barrier + R (3 parameters)")):
        x.hist(s[col].clip(0, 10), bins=np.linspace(0, 10, 61), histtype="step", lw=1.6, color=c,
               label=f"{lab}: median {s[col].median():.2f} mV")
    x.text(0.80, 0.55, "die + 2nd barrier + R cannot fit:\nArray42's pixels have LESS barrier\n"
           "than the die (k < 1), and a barrier\nonly adds voltage. Piled up at\nthe 10 mV clip.",
           transform=x.transAxes, fontsize=7.5, ha="right", va="top", color=lv.NPG["red"])
    x.set(xlabel="per-pixel fit rms (mV), clipped at 10", ylabel="pixels",
          title=f"Per pixel ({len(s)} pixels, ~{s.n.median():.0f} points each): fit quality")
    x.legend(fontsize=7.5, loc="upper right")
    x.grid(alpha=.3)
    lv.save(fig, "19_dense_per_pixel.png", plt, src + " Per pixel: its own points from 0.6 uA to the "
            "transistor limit; 1703 values from per_pixel_Array42.csv.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
