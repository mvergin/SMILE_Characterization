"""Bond resistance of every pixel from the fast scans, tracked over time.

    python analysis/bond_resistance.py

Per pixel, at a fast-scan bias NVLED (V_tot = VLED - NVLED across the pixel):

    V_tot = V_driver(I) + V_LED(I) + I R_bond
    R_bond = (V_tot - V_LED(I) - V_driver(I)) / I                         (model)
    R_rel  = (V_tot - V_typical(I)) / I                                   (relative)

- V_LED: the measured I(V) of a 10 um SGA_15 micro-LED (Array38 and Array42 are from wafer
  SGA_15), `SGA_15 10um/SGA_15_10um_IV.csv`, interpolated in log I. It carries the LED's own
  series resistance (270 Ohm, the device's rser plot), so R_bond is what remains.
- V_driver: the drive transistor at code 15, a smooth-knee model fitted to each array's
  10 best shorts (their own series path, 2.7 kOhm, removed: the code-1/code-15 split of
  261105-1024). Array42's shorts reach 0 V across them (261105-1024). Array38's start at
  0.8 V (261029-0836), above the knee: free, the fit invented a near-vertical rise below it
  (V_ov 0.07 V, 0.02 V at 27 uA). So Array38 takes Array42's measured shape (V_ov, m, lam),
  scaled by one factor I15 fitted to its own shorts (SHAPE_FROM).
- I: the fast-scan current minus its quadrant's leakage (classify_pixels' I_leak_A).
- V_typical: the typical class's median I(V) from the array's no-caps sweep (261026-0813,
  261102-1703).

Pixels used: lit, not short/open/dark_conducting, not overcurrent or neighbour_on (a
latched second pixel adds current), I_LED >= 1 uA; in runs with carrier capacitors only
pixels measured >= 100 s after the scan began. Runs: the "usable" rows of
results/comparisons/bond_resistance/fastscan_runs.csv at -1.35 to -1.65 V; at -2.2 V and
below the driver limits the current and 1 % of it is worth 0.25-0.4 V.

Writes results/comparisons/bond_resistance/:
  pixels/<run>_<bias>.csv   per pixel: current, R_bond, R_rel, class
  runs.csv                  per run: median, quartiles, n, and the subtracted voltages
  median_over_time.png      task 5: median R_bond per array against date (quartiles)
  cdf.png                   task 6: cumulative R_bond distribution, one curve per run
  maps/<run>_<bias>.png     task 7: R_bond map of each run
  subtraction.png, subtraction.md   task 8: what is subtracted, per array
  model_vs_relative.png     the two methods, pixel by pixel, on the reference runs
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import least_squares

import analyze_array as aa
import driver_stats as ds

REPO = Path(__file__).resolve().parent.parent
OUT = REPO / "results" / "comparisons" / "bond_resistance"
LED_CSV = REPO / "SGA_15 10um" / "SGA_15_10um_IV.csv"
VLED = 1.8
# The shorts' own series path. At low voltage the shorts' resistance per code is
# R_on(code 1)/weight + 2.72 kOhm (261105-1024), but the code-independent 2.72 kOhm holds
# two things: the short path, and the driver's source resistance R_src shared by its four
# PMOS, which every LED pixel has too. The design widths (6.8, 2.59, 1.2, 0.54 um, L 2 um)
# fit the shorts' code currents only with both a narrow-width offset (dW 0.18 um) and a
# shared source resistance, R_src / V_ov = 885 /A (Array42); V_ov = 2 I_sat R_on = 1.13 V
# (code 1 and code 15 agree), so R_src = 1.0 kOhm. Only the rest is the short's own path;
# R_src stays in the driver. Array38 has no low-voltage shorts: same path assumed.
R_SRC = 1.0e3
R_SHORT = 2.72e3 - R_SRC
I_MIN = 1e-6
DRIVER = {"Array42": "261105-1024_Array42_PIXELS_Transistoriv",
          "Array38": "261029-0836_Array38_PIXELS_Transistoriv"}
SWEEP = {"Array42": "261102-1703_Array42_ROI_(0, 0, 511, 511)",
         "Array38": "261026-0813_Array38_ROI_(0, 0, 511, 511)"}
REFERENCE_RUN = {"Array42": "261102-1703", "Array38": "261026-0813"}
# arrays whose shorts have no data below the knee borrow the knee of an array that has;
# same design, drivers within ~3 %
SHAPE_FROM = {"Array38": "Array42"}


# ---- LED, driver, typical pixel ---------------------------------------------------------
def led_curve():
    d = pd.read_csv(LED_CSV)
    d = d[(d.Voltage_V > 0) & (d.Current_A > 0) & (d.at_compliance == 0)]
    return d.Voltage_V.to_numpy(), np.maximum.accumulate(d.Current_A.to_numpy())


LV, LI = led_curve()


def v_led(i):
    return np.interp(np.log(np.clip(i, 1e-15, None)), np.log(LI), LV)


def f15(vds, vn, P):
    x = np.clip(vds, 0, None) / P["vov"]
    return (P["I15"] * (1 + P["g"] * (vn - 3.2)) * (1 + P["lam"] * (vds - 5.0))
            * x / (1 + x ** P["m"]) ** (1 / P["m"]))


def fit_driver(arr):
    """Code-15 driver of the array's 10 best shorts, the short path removed: each point
    (V_tot, I) becomes (V_ds = V_tot - I R_SHORT, I)."""
    p = ds.load(ds.R / DRIVER[arr])[0]
    q = p.loc[p[(15, -3.2)].nlargest(10).index]
    volts = sorted(v for b, v in p.columns if b == 15)
    vt = np.array([VLED - v for v in volts])
    i = np.array([float((q[(15, v)] - q[(0, v)]).median()) for v in volts])
    keep = vt > 0.02
    vt, i = vt[keep], i[keep]
    vds, vn = vt - i * R_SHORT, vt - VLED
    # The drivers are PMOS with the source at VLED (1.8 V in every run): the gate drive does
    # not see NVLED, so the gate term g is 0, not a fit parameter. Free, it took +3.2 %/V
    # (Array42) and -3.3 %/V (Array38), compensating the saturation slope.
    if arr in SHAPE_FROM:
        S = fit_driver(SHAPE_FROM[arr])
        shape = {k: S[k] for k in ("lam", "vov", "m")}

        def res(x):
            return f15(vds, vn, dict(shape, I15=x[0], g=0.0)) / i - 1
        r = least_squares(res, [S["I15"]])
        P = dict(shape, I15=float(r.x[0]), g=0.0, shape_from=SHAPE_FROM[arr],
                 scale=float(r.x[0] / S["I15"]))
    else:
        names = ["I15", "lam", "vov", "m"]

        def res(x):
            return f15(vds, vn, dict(zip(names, x), g=0.0)) / i - 1
        r = least_squares(res, [i.max(), 0.02, 0.5, 2.0],
                          bounds=([1e-5, 0.0, 0.02, 0.5], [1e-3, 0.1, 6, 20]))
        P = dict(zip(names, r.x), g=0.0)
    P["rms_pct"] = 100 * float(np.sqrt(np.mean(r.fun ** 2)))
    P["vt_min"] = float(vt.min())
    P["data"] = (vt, i)
    return P


def v_driver(i, vn, P):
    """V_ds at which the driver carries i at gate drive vn; NaN above what it can."""
    grid = np.linspace(0, 6, 6001)
    fi = np.maximum.accumulate(f15(grid, vn, P))
    v = np.interp(i, fi, grid, right=np.nan)
    return np.where(i < fi[-1] * 0.999, v, np.nan)


def typical_curve(arr):
    cc = pd.read_csv(ds.R / SWEEP[arr] / "REPORT_assets/pixelclassification/sweep/class_curves.csv")
    w = cc[(cc.cls == "typical") & cc.has_ref & (cc.I_LED_med > 0)].sort_values("V", ascending=False)
    return (VLED - w.V).to_numpy(), np.maximum.accumulate(w.I_LED_med.to_numpy())


BEST_FRAC = 0.02          # the best pixels: top 2 % of current at -1.40 V in the sweep


AGREE = 0.03             # the best pixels' sweep current must match the fast scan within 3 %


def best_reference(arr, fs_i=None):
    """The array's best pixels as the reference: no driver model, no LED curve.

    Two pixels of one array that carry the same current have the same driver drop (PMOS,
    identical to 0.5 %, depending only on the voltage across it), and the same LED up to its
    turn-on offset. So with each pixel's own LED anchor dV (its V_tot - V_SGA(I) at the sweep
    point nearest 1.5 uA, where driver and bond drop a few mV), U = V_tot - dV is the voltage
    on driver + LED shape + bond, and a pixel's bond above the best ones is
    (U_i - U_best(I_i)) / I_i. U_best(I): per sweep bias, the best pixels' median current
    against V_tot minus their median anchor; interpolated in log I.
    The best pixels are chosen by the same run's fast scan current at -1.40 V (fs_i): random
    order, retest Spearman 0.97, latched pixels removed. Chosen by the sweep's own -1.40 V
    point (before 2026-11-06), half of Array42's set drew >5 % more in the sweep than in the
    fast scan: a second driver adding current (neighbour_on/other), not a better bond.
    Returns (anchor per pixel, the curve as (I, U), the best pixels' index)."""
    import pixel_classes as pc
    sw = pc.load_sweep(ds.R / SWEEP[arr])
    sw = sw[sw.I_ref.notna() & (sw.I_ref < 1e-6) & (sw.I_LED > 0)].copy()   # no lit references
    sw["vtot"] = VLED - sw.V
    a = sw[(sw.I_LED >= 0.5e-6) & (sw.I_LED <= 3e-6)].copy()
    a["dist"] = np.abs(np.log(a.I_LED / 1.5e-6))
    a = a.loc[a.groupby(["X", "Y"]).dist.idxmin()].set_index(["X", "Y"])
    anchor = (a.vtot - v_led(a.I_LED.to_numpy())).rename("dV")
    cls = pd.read_csv(ds.R / SWEEP[arr] / "REPORT_assets/pixelclassification/sweep/sweep_classes.csv"
                      ).set_index(["X", "Y"])
    ok = cls.index[cls.get("m1p40V", pd.Series(dtype=str)) == "typical"]
    at = sw[sw.V.round(3) == -1.4].set_index(["X", "Y"]).I_LED
    at = at[at.index.isin(ok) & at.index.isin(anchor.index)]
    if fs_i is not None:
        f = fs_i.reindex(at.index)
        at = f[(at / f - 1).abs() <= AGREE]
    best = at.nlargest(max(20, int(BEST_FRAC * len(at)))).index
    b = sw.set_index(["X", "Y"]).loc[lambda q: q.index.isin(best)].reset_index()
    g = b.groupby("V")
    curve = pd.DataFrame({"I": g.I_LED.median(),
                          "U": (VLED - g.V.first()) - float(anchor.loc[best].median())})
    curve = curve[curve.I > 1e-6].sort_values("I")
    return anchor, (curve.I.to_numpy(), curve.U.to_numpy()), best


# ---- one fast scan ----------------------------------------------------------------------
def run_dir(run):
    for base in (REPO / "results", REPO / "results_georg" / "results"):
        if (base / run).is_dir():
            return base / run
    raise FileNotFoundError(run)


def pixels(row, P, typ, P_other, ref=None):
    fs = aa._load_aggregated(run_dir(row.run) / "FastScan")
    fs = fs[fs.NVLED_V.astype(float).round(3) == round(row.V, 3)].copy()
    d = aa.classify_pixels(fs)
    t = fs.groupby(["X", "Y"]).TIME_START.first()
    d["t_s"] = d.set_index(["X", "Y"]).index.map(t) - t.min()
    d["I_LED"] = (d.I_A - d.I_leak_A).astype(float)
    ok = (~d.primary.isin(["short", "open", "dark_conducting", "overcurrent", "neighbour_on"])
          & (d.I_LED >= I_MIN))
    if row.caps:
        ok &= d.t_s >= 100
    d = d[ok].copy()
    vtot, vn = VLED - row.V, -row.V
    d["V_LED"] = v_led(d.I_LED.to_numpy())
    d["V_driver"] = v_driver(d.I_LED.to_numpy(), vn, P)
    d["R_bond_kOhm"] = (vtot - d.V_LED - d.V_driver) / d.I_LED / 1e3
    # the bracket: the same pixels with the OTHER array's driver. Array38's shorts start at
    # 0.8 V, so its knee is borrowed from Array42 (SHAPE_FROM); Array42's is measured to 0 V.
    d["R_bond_other_driver_kOhm"] = ((vtot - d.V_LED - v_driver(d.I_LED.to_numpy(), vn, P_other))
                                     / d.I_LED / 1e3)
    tv, ti = typ
    d["V_typical"] = np.interp(np.log(d.I_LED), np.log(ti), tv)
    d["R_rel_kOhm"] = (vtot - d.V_typical) / d.I_LED / 1e3
    if ref is not None:
        anchor, (ci, cu), _ = ref
        dv = d.set_index(["X", "Y"]).index.map(anchor).to_numpy(dtype=float)
        d["anchored"] = np.isfinite(dv)
        dv = np.where(np.isfinite(dv), dv, float(anchor.median()))
        ub = np.interp(np.log(d.I_LED), np.log(ci), cu)
        d["R_best_kOhm"] = (vtot - dv - ub) / d.I_LED / 1e3
    return d


# ---- plots --------------------------------------------------------------------------------
def map_png(d, row, rng, out, plt):
    x0, x1, y0, y1 = int(row.roi_x0), int(row.roi_x1), int(row.roi_y0), int(row.roi_y1)
    img = np.full((y1 - y0 + 1, x1 - x0 + 1), np.nan)
    img[d.Y.to_numpy() - y0, d.X.to_numpy() - x0] = d.R_bond_kOhm.to_numpy()
    fig, axs = plt.subplots(1, 2, figsize=(15, 6.5))
    cm = plt.get_cmap("magma").copy()
    cm.set_bad("0.75")
    B = 8 if img.shape[0] >= 128 else 2               # block medians: the trend under the noise
    h, w = img.shape[0] // B * B, img.shape[1] // B * B
    with np.errstate(all="ignore"):
        blk = np.nanmedian(img[:h, :w].reshape(h // B, B, w // B, B).transpose(0, 2, 1, 3)
                           .reshape(h // B, w // B, B * B), axis=2)
    for a, im_, ttl in ((axs[0], img, "every pixel"), (axs[1], blk, f"{B} x {B} block medians")):
        im = a.imshow(im_, cmap=cm, vmin=rng[0], vmax=rng[1], origin="upper",
                      extent=(x0 - 0.5, x0 - 0.5 + (w if im_ is blk else img.shape[1]),
                              y0 - 0.5 + (h if im_ is blk else img.shape[0]), y0 - 0.5),
                      interpolation="nearest")
        a.set_title(ttl, fontsize=9)
        fig.colorbar(im, ax=a, label="R_bond (kOhm)", shrink=0.85)
    fig.suptitle(f"{row["array"]} {row.date}, {row.V:+.2f} V, {'caps' if row.caps else 'no caps'}: "
                 f"R_bond, median {d.R_bond_kOhm.median():.2f} kOhm ({len(d)} pixels; grey: "
                 f"not used)", fontsize=10)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def main() -> int:
    plt = aa._plt()
    (OUT / "pixels").mkdir(parents=True, exist_ok=True)
    (OUT / "maps").mkdir(parents=True, exist_ok=True)
    runs = pd.read_csv(OUT / "fastscan_runs.csv")
    runs = runs[runs.usable.str.startswith(("yes", "partly")) & (runs.V.abs() <= 1.7)].copy()
    xy = runs.roi.str.extract(r"\((\d+),(\d+)\)-\((\d+),(\d+)\)").astype(int)
    runs[["roi_x0", "roi_y0", "roi_x1", "roi_y1"]] = xy.to_numpy()
    drv = {a: fit_driver(a) for a in DRIVER}
    typ = {a: typical_curve(a) for a in DRIVER}
    # the best pixels are chosen by the reference run's own fast scan (best_reference)
    best = {}
    for a in DRIVER:
        r0 = runs[runs.run.str.startswith(REFERENCE_RUN[a])].iloc[0]
        f0 = pixels(r0, drv[a], typ[a], drv[a]).set_index(["X", "Y"]).I_LED
        best[a] = best_reference(a, f0)
    for a, (anc, (ci, cu), bi) in best.items():
        print(f"{a} best-pixel reference: {len(bi)} pixels, {len(anc)} anchored; curve "
              + ", ".join(f"{u:.3f} V at {i * 1e6:.1f} uA" for i, u in zip(ci, cu)))
    for a, P in drv.items():
        print(f"{a} driver (10 best shorts, code 15, short path {R_SHORT / 1e3:.2f} kOhm removed): "
              f"rms {P['rms_pct']:.2f} %, lam {100 * P['lam']:.2f} %/V, g {100 * P['g']:+.2f} %/V, "
              f"data from {P['vt_min']:.2f} V across the short")
    rows, data = [], {}
    for _, r in runs.iterrows():
        other = next(k for k in DRIVER if k != r["array"])
        d = pixels(r, drv[r["array"]], typ[r["array"]], drv[other], best[r["array"]])
        tag = f"{r.run[:11]}_{aa._bias_tag(r.V)}"
        d[["X", "Y", "quadrant", "primary", "t_s", "I_LED", "V_LED", "V_driver", "R_bond_kOhm",
           "R_bond_other_driver_kOhm", "R_rel_kOhm", "R_best_kOhm", "anchored"]].to_csv(OUT / "pixels" / f"{tag}.csv", index=False, float_format="%.5g")
        data[tag] = (r, d)
        q = d.R_bond_kOhm.quantile([.1, .25, .5, .75, .9])
        im = d.I_LED.median()
        rows.append(dict(tag=tag, run=r.run, date=r.date, array=r["array"], V=r.V, caps=r.caps,
                         range=r.range, n=len(d), I_LED_median_uA=im * 1e6,
                         V_tot=VLED - r.V, V_LED_at_median=float(v_led(im)),
                         V_driver_at_median=float(v_driver(np.array([im]), -r.V, drv[r["array"]])[0]),
                         R_p10=q[.1], R_p25=q[.25], R_median=q[.5], R_p75=q[.75], R_p90=q[.9],
                         R_rel_median=d.R_rel_kOhm.median(),
                         R_median_other_driver=d.R_bond_other_driver_kOhm.median(),
                         Rbest_p10=d.R_best_kOhm.quantile(.1), Rbest_p25=d.R_best_kOhm.quantile(.25),
                         Rbest_median=d.R_best_kOhm.median(), Rbest_p75=d.R_best_kOhm.quantile(.75),
                         Rbest_p90=d.R_best_kOhm.quantile(.9), anchored_frac=d.anchored.mean()))
        print(f"  {tag} {r["array"]} {r.V:+.2f} V: {len(d)} px, I {im * 1e6:.1f} uA, R_bond median "
              f"{q[.5]:.2f} kOhm (25-75 % {q[.25]:.2f}..{q[.75]:.2f}), R_rel median "
              f"{d.R_rel_kOhm.median():+.2f}")
    t = pd.DataFrame(rows)
    # the same pixels against the array's reference run: a change per pixel, free of the
    # ROI mix and of the model's common offset (it still carries a bias-dependent one:
    # 261016-2252 reads -0.68 kOhm at -1.35 V and +0.61 at -1.65 V on the same pixels)
    for a in DRIVER:
        ref = next((k for k, (r, _) in data.items() if r.run.startswith(REFERENCE_RUN[a])), None)
        if ref is None:
            continue
        rr = data[ref][1].set_index(["X", "Y"]).R_bond_kOhm
        rq = data[ref][1].set_index(["X", "Y"]).R_rel_kOhm
        rbst = data[ref][1].set_index(["X", "Y"]).R_best_kOhm
        for k, (r, d) in data.items():
            if r["array"] != a:
                continue
            j = d.set_index(["X", "Y"]).R_bond_kOhm.to_frame("R").join(rr.rename("ref"),
                                                                        how="inner").dropna()
            dif = j.R - j.ref
            jr = d.set_index(["X", "Y"]).R_rel_kOhm.to_frame("R").join(rq.rename("ref"),
                                                                       how="inner").dropna()
            jb = d.set_index(["X", "Y"]).R_best_kOhm.to_frame("R").join(rbst.rename("ref"),
                                                                        how="inner").dropna()
            i = t.index[t.tag == k][0]
            t.loc[i, ["ref", "n_common", "dR_median", "dR_p25", "dR_p75", "dR_rel_median",
                      "dR_best_median"]] = [
                ref, len(j), dif.median(), dif.quantile(.25), dif.quantile(.75),
                (jr.R - jr.ref).median(), (jb.R - jb.ref).median()]
    t.to_csv(OUT / "runs.csv", index=False, float_format="%.5g")

    # the best-pixel reference: distributions, over time, and the two arrays' best curves
    fig, ax = plt.subplots(1, 3, figsize=(19, 5.6))
    cmap = plt.get_cmap("viridis")
    for a, ls in zip(DRIVER, ("-", "--")):
        sel = [k for k, (r, _) in data.items() if r["array"] == a]
        for j, k in enumerate(sel):
            v = np.sort(data[k][1].R_best_kOhm.dropna().to_numpy())
            ax[0].plot(v, np.arange(1, v.size + 1) / v.size, ls=ls, lw=1,
                       color=cmap(j / max(1, len(sel) - 1)), label=f"{a} {k}")
        s_ = t[t["array"] == a].copy()
        s_["when"] = pd.to_datetime(s_.date)
        ax[1].errorbar(s_.when, s_.Rbest_median, yerr=[s_.Rbest_median - s_.Rbest_p25,
                                                       s_.Rbest_p75 - s_.Rbest_median],
                       fmt="o" if a == "Array42" else "s", capsize=3, label=f"{a}: median (25-75 %)")
        ci, cu = best[a][1]
        ax[2].plot(cu, ci * 1e6, "o-", ms=3, label=f"{a}: best {BEST_FRAC:.0%} of pixels")
    ax[0].set_xscale("symlog", linthresh=1)
    ax[0].set_xlabel("bond resistance above the array's best pixels (kOhm)")
    ax[0].set_ylabel("fraction of pixels")
    ax[0].set_title("Per scan (solid Array42, dashed Array38)", fontsize=10)
    ax[0].legend(fontsize=5, ncol=2)
    ax[1].set_ylabel("kOhm above the array's best pixels")
    ax[1].set_title("Over time", fontsize=10)
    ax[1].tick_params(axis="x", labelrotation=30, labelsize=7)
    ax[1].legend(fontsize=8)
    ax[2].set_xlabel("V_tot - the best pixels' median LED anchor (V)")
    ax[2].set_ylabel("current (uA)")
    ax[2].set_xlim(2.7, 4.2)
    ax[2].set_title("The reference curves: each array's best pixels", fontsize=10)
    ax[2].legend(fontsize=8)
    for x in ax:
        x.grid(alpha=0.3, which="both")
    fig.suptitle("Bond resistance against the array's own best pixels (no driver model, no LED "
                 "curve; each pixel's LED anchored at ~1.5 uA)", fontsize=11)
    fig.tight_layout()
    fig.savefig(OUT / "best_reference.png", dpi=150)
    plt.close(fig)

    # task 7: maps, one colour range per array
    for a in DRIVER:
        sel = [k for k, (r, _) in data.items() if r["array"] == a]
        if not sel:
            continue
        allr = pd.concat([data[k][1].R_bond_kOhm for k in sel])
        rng = (float(allr.quantile(0.05)), float(allr.quantile(0.95)))   # the tail runs to 100s of kOhm
        for k in sel:
            map_png(data[k][1], data[k][0], rng, OUT / "maps" / f"{k}.png", plt)

    # task 5: median over time (top), and the same pixels' change against the reference
    fig, ax = plt.subplots(2, 2, figsize=(15, 10))
    for a, col in zip(DRIVER, range(2)):
        s = t[t["array"] == a].copy()
        s["when"] = pd.to_datetime(s.date)
        for row, (y, lo, hi, what) in enumerate((
                ("R_median", "R_p25", "R_p75", "median R_bond per fast scan"),
                ("dR_median", "dR_p25", "dR_p75",
                 f"per-pixel change against {REFERENCE_RUN[a]}, same pixels"))):
            x = ax[row, col]
            for caps, mk in ((True, "s"), (False, "o")):
                c = s[(s.caps == caps) & s[y].notna()]
                x.errorbar(c.when, c[y], yerr=[c[y] - c[lo], c[hi] - c[y]], fmt=mk, capsize=3,
                           label="with capacitors" if caps else "no capacitors")
                for _, q in c.iterrows():
                    x.annotate(f"{q.V:+.2f} V, n={q.n_common if row else q.n:.0f}"
                               + (", 10 mA range" if "10 mA" in q["range"] else ""),
                               (q.when, q[y]), fontsize=6, rotation=30, xytext=(4, 4),
                               textcoords="offset points")
            if row:
                x.axhline(0, color="k", lw=0.8)
            x.set_title(f"{a}: {what} (bars: 25-75 %)", fontsize=10)
            x.set_ylabel("R_bond (kOhm)" if not row else "R_bond - R_bond(reference) (kOhm)")
            x.grid(alpha=0.3)
            x.legend(fontsize=8)
            x.tick_params(axis="x", labelrotation=30, labelsize=7)
    fig.tight_layout()
    fig.savefig(OUT / "median_over_time.png", dpi=150)
    plt.close(fig)

    # task 6: cumulative distributions
    fig, ax = plt.subplots(1, 2, figsize=(15, 5.5))
    cm = plt.get_cmap("viridis")
    for a, x in zip(DRIVER, ax):
        sel = [k for k, (r, _) in data.items() if r["array"] == a]
        for j, k in enumerate(sel):
            r, d = data[k]
            v = np.sort(d.R_bond_kOhm.dropna().to_numpy())
            x.plot(v, np.arange(1, v.size + 1) / v.size, lw=1, color=cm(j / max(1, len(sel) - 1)),
                   ls="--" if r.caps else "-", label=f"{r.date[:10]} {r.V:+.2f} V (n={v.size})")
        x.set_xscale("symlog", linthresh=1)
        x.set_xlabel("R_bond (kOhm), symlog: linear within +-1")
        x.set_ylabel("fraction of pixels")
        x.set_title(f"{a}: cumulative R_bond per fast scan (dashed: with capacitors)", fontsize=10)
        x.grid(alpha=0.3, which="both")
        x.legend(fontsize=6)
    fig.tight_layout()
    fig.savefig(OUT / "cdf.png", dpi=150)
    plt.close(fig)

    # task 8: what is subtracted
    fig, ax = plt.subplots(1, 2, figsize=(15, 5.8))
    for a, x in zip(DRIVER, ax):
        ii = np.logspace(-6, np.log10(1.3e-4), 200)
        x.plot(v_led(ii), ii * 1e6, color="tab:blue", label="LED, SGA_15 10 um (measured)")
        for vb, ls in ((-1.35, "-"), (-1.4, "--")):
            x.plot(v_driver(ii, -vb, drv[a]), ii * 1e6, ls=ls, color="tab:red",
                   label=f"driver code 15 at NVLED {vb:+.2f} V (model, short path removed)")
        vt_d, i_d = drv[a]["data"]
        x.plot(vt_d - i_d * R_SHORT, i_d * 1e6, ".", ms=3, color="tab:red", alpha=0.5,
               label=f"10 best shorts, V_tot - I x {R_SHORT / 1e3:.2f} kOhm")
        tv, ti = typ[a]
        x.plot(tv, ti * 1e6, "o-", ms=3, color="k", label="typical pixel, whole (sweep median)")
        x.set_xlim(0, 4.0)
        x.set_ylim(0, 130)
        x.set_xlabel("V (V)")
        x.set_ylabel("I (uA)")
        x.grid(alpha=0.3)
        x.legend(fontsize=7)
        x.set_title(f"{a}: the parts subtracted from V_tot", fontsize=10)
    fig.tight_layout()
    fig.savefig(OUT / "subtraction.png", dpi=150)
    plt.close(fig)
    L = ["# Bond resistance: what is subtracted", "",
         "R_bond = (V_tot - V_LED(I) - V_driver(I)) / I, per pixel, at its own current. "
         "Values at each run's median pixel:", "",
         "| run | array | NVLED | I median (uA) | V_tot (V) | V_LED (V) | V_driver (V) | "
         "left: I x R_bond (V) | R_bond median (kOhm) |", "|" + "---|" * 9]
    for _, q in t.iterrows():
        left = q.V_tot - q.V_LED_at_median - q.V_driver_at_median
        L.append(f"| {q.tag} | {q["array"]} | {q.V:+.2f} | {q.I_LED_median_uA:.1f} | {q.V_tot:.2f} | "
                 f"{q.V_LED_at_median:.3f} | {q.V_driver_at_median:.3f} | {left:+.3f} | "
                 f"{q.R_median:.2f} |")
    L += ["", "## The parts", "",
          "- **LED:** measured I(V) of a 10 um micro-LED from wafer SGA_15 (`SGA_15 10um/SGA_15_10um_IV.csv`, "
          "2023-08-18), the wafer of both arrays. It includes the LED's own series resistance, "
          "270 Ohm (`SGA_15 10um rser.bmp`). Not the same die: at 1.5 uA, where driver and bond "
          "drops are negligible, it needs ~50 mV more than our bonded typical pixel (sweep "
          "261102-1703). That sets the zero of R_bond to ~+-1.3 kOhm at 38 uA.",
          f"- **Driver:** code 15, the 10 best shorts of each array with their own series path "
          f"({R_SHORT / 1e3:.2f} kOhm, from the code dependence of 261105-1024) removed, fitted "
          f"with a smooth-knee model (I15 (1 + g (V_n - 3.2)) (1 + lam (V_ds - 5)) x / (1 + x^m)^(1/m)):"]
    for a, P in drv.items():
        L.append(f"  - {a} ({DRIVER[a]}): rms {P['rms_pct']:.2f} %, saturation slope "
                 f"{100 * P['lam']:.2f} %/V, gate term {100 * P['g']:+.2f} %/V; shorts measured "
                 f"from {P['vt_min']:.2f} V across them"
                 + (f" -- no data below the knee: {P['shape_from']}'s shape (V_ov, m, lam), "
                    f"current x {P['scale']:.3f}" if "shape_from" in P else ""))
    L += ["- **Leakage:** each quadrant's open-pixel current, subtracted from the fast-scan current.",
          "- **Relative method:** R_rel = (V_tot - V_typical(I)) / I with the typical class's sweep "
          "median as V_typical (261026-0813, 261102-1703).",
          "- **Errors to quote with the results:** runs on the 10 mA range (Array38, July) have "
          "an offset of several % of their 25-30 uA; 1 % of current is ~0.1 kOhm at -1.35 V. "
          "Runs with capacitors use only pixels measured after 100 s."]
    (OUT / "subtraction.md").write_text("\n".join(L) + "\n", encoding="utf-8")

    # hypothesis: do high bond resistances change more? Pairs of scans of one array at the
    # same bias and capacitor state, earlier -> later, same pixels, binned by the earlier R.
    pairs = []
    for a in DRIVER:
        ks = [k for k, (r, _) in data.items() if r["array"] == a]
        for k1, k2 in zip(ks, ks[1:]):
            r1, d1 = data[k1]
            r2, d2 = data[k2]
            if r1.V != r2.V or r1.caps != r2.caps:
                continue
            j = (d1.set_index(["X", "Y"]).R_bond_kOhm.to_frame("R1")
                 .join(d2.set_index(["X", "Y"]).R_bond_kOhm.rename("R2"), how="inner").dropna())
            if len(j) >= 1000:
                pairs.append((a, k1, k2, j))
    if pairs:
        fig, ax = plt.subplots(1, len(pairs), figsize=(5.2 * len(pairs), 4.8), squeeze=False)
        rows_p = []
        for x, (a, k1, k2, j) in zip(ax[0], pairs):
            j["dR"] = j.R2 - j.R1
            # binned by the MEAN of both scans: binning by the earlier one alone makes the
            # top decile fall and the bottom rise from noise (regression to the mean)
            j["Rm"] = (j.R1 + j.R2) / 2
            edges = np.unique(np.nanpercentile(j.Rm, np.linspace(0, 100, 11)))
            b = pd.cut(j.Rm, edges, include_lowest=True)
            g = j.groupby(b, observed=True).dR
            mid = j.groupby(b, observed=True).Rm.median()
            x.errorbar(mid, g.median(), yerr=[g.median() - g.quantile(.25),
                                              g.quantile(.75) - g.median()], fmt="o-", capsize=3)
            x.axhline(0, color="k", lw=0.8)
            x.set_xscale("symlog", linthresh=1)
            x.set_yscale("symlog", linthresh=0.1)
            x.set_xlabel("R_bond, mean of both scans (kOhm), deciles")
            x.set_ylabel(f"R_bond({k2[:11]}) - R_bond({k1[:11]}) (kOhm)")
            x.set_title(f"{a}: {k1[:11]} -> {k2[:11]} ({len(j)} pixels)", fontsize=9)
            x.grid(alpha=0.3)
            for (iv, dm), m_ in zip(g.median().items(), mid):
                rows_p.append(dict(array=a, earlier=k1, later=k2, R_mean_median=m_,
                                   dR_median=dm, n=int(g.size()[iv])))
        fig.suptitle("Do high bond resistances change more? Change per pixel against its "
                     "earlier value (median, 25-75 %)", fontsize=11)
        fig.tight_layout()
        fig.savefig(OUT / "change_vs_R.png", dpi=150)
        plt.close(fig)
        pd.DataFrame(rows_p).to_csv(OUT / "change_vs_R.csv", index=False, float_format="%.4g")

    # the two methods. Within one scan every pixel sees the same V_tot, so both are
    # functions of the current alone: compare the extra resistance each gives a pixel
    # against its current (top), and both medians over time (bottom).
    fig, ax = plt.subplots(2, 2, figsize=(14, 10))
    for a, col in zip(DRIVER, range(2)):
        x = ax[0, col]
        for k, (r, d) in data.items():
            if r["array"] != a or not r.run.startswith(REFERENCE_RUN[a]):
                continue
            vt = VLED - r.V
            ii = np.logspace(np.log10(d.I_LED.quantile(0.01)), np.log10(d.I_LED.quantile(0.99)), 200)
            im = d.I_LED.median()

            def rb(i, vt=vt, r=r, a=a):
                return (vt - v_led(i) - v_driver(i, -r.V, drv[a])) / i / 1e3
            tv, ti = typ[a]
            rr = (vt - np.interp(np.log(ii), np.log(ti), tv)) / ii / 1e3
            x.plot(ii * 1e6, rb(ii) - rb(np.array([im]))[0], color="tab:blue",
                   label="model: R_bond(I) - R_bond(median pixel)")
            x.plot(ii * 1e6, rr, "--", color="tab:orange",
                   label="relative: (V_tot - V_typical(I)) / I")
            x.axvline(im * 1e6, color="0.5", lw=0.8)
            x.set_xscale("log")
            x.set_yscale("symlog", linthresh=1)
            x.set_xlabel("pixel current I_LED (uA); grey: the median pixel")
            x.set_ylabel("extra resistance against the median pixel (kOhm)")
            x.set_title(f"{a}, {k}: the two methods for a pixel drawing I", fontsize=10)
            x.grid(alpha=0.3, which="both")
            x.legend(fontsize=8)
        x = ax[1, col]
        s_ = t[t["array"] == a].copy()
        s_["when"] = pd.to_datetime(s_.date)
        x.plot(s_.when, s_.dR_median, "o",
               label="model: median per-pixel change against the reference")
        x.plot(s_.when, s_.dR_rel_median, "x", ms=8,
               label="relative: median per-pixel change against the reference")
        x.axhline(0, color="k", lw=0.8)
        x.set_ylabel("kOhm")
        x.set_title(f"{a}: both methods over time (reference {REFERENCE_RUN[a]})", fontsize=10)
        x.grid(alpha=0.3)
        x.legend(fontsize=8)
        x.tick_params(axis="x", labelrotation=30, labelsize=7)
    fig.tight_layout()
    fig.savefig(OUT / "model_vs_relative.png", dpi=150)
    plt.close(fig)
    print("\n".join(L))
    return 0


if __name__ == "__main__":
    sys.exit(main())
