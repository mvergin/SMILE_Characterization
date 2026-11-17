"""The method notebook on whole arrays: the transistor it uses, and its section-9 models on every swept pixel.

    python analysis/method_maps.py            # with joblib: 3 worker processes (~45 min), without: serial (~1.5 h)
    python analysis/method_maps.py --replot   # only the figures and summary.md, from the CSVs of the last run

The method lives in results/comparisons/led_voltage/led_voltage_method.ipynb. This script runs all of the
notebook's code cells (plots off) and uses what they define: the transistor (section 3), n_R and I0
(section 6), the die's excess (section 8), the models in EXTRA, their limits and RMS_FACTOR (section 9).
Change the method there, then rerun this. Runs are read from the local results/ folder where they exist
(the same files as on the share, read faster).

Writes
  results/comparisons/drivers/: transistor.csv and transistor.json (the transistor of both arrays, the
      file every analysis reads), transistor.png, README.md
  results/comparisons/bond_resistance/models/: the models of MODELS (A, B, G2) fitted to every pixel of each
      array's full sweep (SWEEPS), exactly as the notebook fits its six pixels, and the winner among them.
      <array>_models.csv (one row per pixel); models_<array>.png (the winner: A grey, G2 by its I_b, B by its
      second barrier, and where B wins); R2_<array>.png (R2 of each winner, the hue its model); G2_<array>.png
      (G2's I_b and slope on every pixel, the fit quality); absolute_vs_relative.png (the absolute extra against
      R_best and R_local); summary.md. Every map names the models' equations at its foot.
"""
from __future__ import annotations

import gc
import json
import sys
import time
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

import pixel_classes as pc

REPO = Path(__file__).resolve().parent.parent
LOCAL = REPO / "results"
SHARE = Path(r"X:\public\Schoettler\von Max\SMILE_IQE_Results\runs\results")
NB = LOCAL / "comparisons" / "led_voltage" / "led_voltage_method.ipynb"
DRIVERS = LOCAL / "comparisons" / "drivers"
BOND = LOCAL / "comparisons" / "bond_resistance"
OUT = BOND / "models"
SWEEPS = {"Array42": "261102-1703_Array42_ROI_(0, 0, 511, 511)",     # 49 150 pixels swept, 9 biases with light
          "Array38": "261026-0813_Array38_ROI_(0, 0, 511, 511)"}     # 113 568 pixels, 8 to 10 biases with light
RELATIVE = {"Array42": "261102-1703_m1p40V", "Array38": "261026-0813_m1p40V"}   # the same runs' fast scans
ANCHOR_MAX = 1.5     # a pixel's lowest light must lie within this factor of I0. Eq. (9) anchors V_j at I0, and
#                      np.interp holds a pixel's lowest point below it: at 1.5 x I0 that misses <= 3 mV of the die
N_MIN = 5            # fit points per pixel at least: two more than the most parameters a model has (B, G2)
I_MAP = 30e-6        # B's second barrier is shown as the voltage it adds at this current (all pixels reach it)
V_REL = -1.4         # the bias of the fast scans in bond_resistance/, for the comparison with R_best and R_local
BLOCK = 16           # px: blocks for the share of pixels where B wins
N = 512              # pixels per side
JOBS, CHUNK = 3, 1000    # worker processes and pixels per task: the fits take ~30 ms per pixel. 3, not 8: memory
MODELS = ("A", "B", "G2")    # the models of the notebook's EXTRA fitted here, by their letter: a resistor, a second
#                              barrier, the die's barrier with its own turn-on (G2: and its slope within A_BAND).
#                              The winner is chosen among these with the notebook's rule. G instead of G2 also works.
SHIFT = next(m for m in MODELS if m.startswith("G"))    # the die's-barrier model: blue in the maps

# colours (dataviz reference palette): blue = the die's barrier (G2), orange = B, grey = A
BLUE = ["#cde2fb", "#86b6ef", "#3987e5", "#256abf", "#104281", "#0d366b"]
ORANGE = ["#fbe0d3", "#f4a77f", "#eb6834", "#c9501f", "#8f3412"]
GREY = ["#e1e0d9", "#c3c2b7", "#898781", "#52514e", "#0b0b0b"]
VIOLET = ["#e3defa", "#b3a8f0", "#8576e4", "#6250d6", "#3d2f9c"]
SKIPPED, NOT_SWEPT = "#e1e0d9", "#ffffff"
DIVERGING = ["#184f95", "#5598e7", "#f0efec", "#e66767", "#a8302f"]
RAMP = {"A": GREY[1:], "B": ORANGE[1:], SHIFT: BLUE[1:]}           # per model: its colours from light to dark


def notebook():
    """Run every code cell of the notebook, plots and tables off; return its namespace."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    try:
        import IPython.display as ipd
    except ImportError:                       # the notebook imports display and Markdown from it, nothing else
        import types
        ipd = sys.modules["IPython.display"] = types.ModuleType("IPython.display")
        sys.modules["IPython"] = types.ModuleType("IPython")
        ipd.Markdown = str
    plt.show = lambda *a, **k: plt.close("all")
    ipd.display = lambda *a, **k: None
    ns = {}
    for cell in json.loads(NB.read_text(encoding="utf-8"))["cells"]:
        if cell["cell_type"] == "code":
            exec(compile("".join(cell["source"]), str(NB), "exec"), ns)
            runs = [*ns.get("RUNS", {}).values(), *ns.get("SHORTS", {}).values(), *ns.get("SURVEY", {}).values()]
            if "BASEDIR" in ns and runs and all((LOCAL / r).is_dir() for r in runs):
                ns["BASEDIR"] = LOCAL                                # the same runs, read locally
    plt.close("all")
    return ns, plt


def run_dir(run):
    return LOCAL / run if (LOCAL / run).is_dir() else SHARE / run


# ---- 1. the transistor ---------------------------------------------------------------------------------
def export_transistor(nb, plt):
    """drivers/: the transistor of section 3 for both arrays, as the files every analysis reads, a figure and
    the README with every number of the derivation."""
    DRIVERS.mkdir(parents=True, exist_ok=True)
    vds, i42, tr, knee = nb["VDS42"], nb["I42"], nb["tr"], float(nb["V_KNEE"])
    s42, s38, best = nb["shorts"]["Array42"], nb["shorts"]["Array38"], nb["BEST"]
    pd.DataFrame({"V_ds_V": vds, "I_Array42_A": i42, "I_Array38_A": tr["Array38"] * i42,
                  "below_knee": vds < knee}).to_csv(DRIVERS / "transistor.csv", index=False, float_format="%.6g")
    tab, expected = nb["ohmic_table"](s42, nb["V_LOW"], nb["V_HIGH"])
    model = np.array([nb["stack_current"](g, nb["V_OV"], nb["BETA5"], nb["V_ON5"], nb["VLED42"]) for g in nb["GC"]])
    at = {i: float(nb["v_transistor"](i, 1.0)) for i in (20e-6, 50e-6, 100e-6)}
    r_short = []                                     # check 3f (2): R_short with I_sat read at 2-5 V instead of V_SAT
    for v_sat in (2.0, 3.0, 4.0, 5.0):
        rc, isat = nb["measure"](nb["BEST42"], v_sat)
        b5, von5 = nb["fit_m5"](rc, isat, nb["R_S"], nb["VLED42"])[1:3]
        r_short.append(nb["R_S"] - 1 / (b5 * (nb["VLED42"] - von5)))
    P = {"updated": str(date.today()), "made_by": "analysis/method_maps.py from " + NB.name + ", section 3",
         "runs": {"Array42": s42["run"], "Array38": s38["run"], "Array42_survey": nb["all_shorts"]["Array42"]["run"]},
         "best_short": {a: [int(v) for v in best[a]] for a in best},
         "VLED_V": float(nb["VLED42"]), "codes": [int(c) for c in nb["codes"]],
         "R_c_Ohm": [float(r) for r in nb["RC"]], "I_sat_A": [float(i) for i in nb["ISAT"]], "V_sat_V": float(nb["V_SAT"]),
         "R_s_Ohm": float(nb["R_S"]), "R_M5_Ohm": float(nb["R_M5"]), "R_short_Ohm": float(nb["R_SHORT"]),
         "V_ov_V": float(nb["V_OV"]), "beta5_A_per_V2": float(nb["BETA5"]), "V_on5_V": float(nb["V_ON5"]),
         "f": {a: float(f) for a, f in tr.items()}, "f_read_at_V_tot_V": float(nb["vtoteval"]),
         "V_knee_V": knee, "knee_current_A": {a: float(nb["i_transistor"](knee, f)) for a, f in tr.items()}}
    (DRIVERS / "transistor.json").write_text(json.dumps(P, indent=1), encoding="utf-8")

    fig, (ax, bx) = plt.subplots(1, 2, figsize=(13, 5.2))
    for arr, s, col in (("Array42", s42, BLUE[3]), ("Array38", s38, ORANGE[2])):
        for xy in s["best"]:
            ax.plot(s["V_tot"], s["cur"][xy][15] * 1e6, "-", lw=0.5, color=col, alpha=0.3)
        ax.plot(s["V_tot"], s["cur"][best[arr]][15] * 1e6, "--", lw=1.2, color=col,
                label=f"{arr}: best short {best[arr]}, as measured (V_tot); thin: its 10 best")
        ax.plot(vds, tr[arr] * i42 * 1e6, "-", lw=2.2, color=col,
                label=f"{arr}: transistor, " + ("V_tot - I R_short" if arr == "Array42" else f"Array42's x {tr[arr]:.3f}"))
        ii = np.linspace(0.5e-6, 0.995 * tr[arr] * i42.max(), 300)
        bx.plot(ii * 1e6, nb["v_transistor"](ii, tr[arr]), "-", lw=2, color=col, label=f"{arr}: V_ds(I), eq. (5)")
        kc = P["knee_current_A"][arr]
        bx.plot(kc * 1e6, knee, "o", color=col, label=f"{arr}: knee at {kc * 1e6:.0f} uA")
    ax.axvline(knee, color="#52514e", ls=":", lw=1, label=f"knee: V_ds = {knee:.2f} V (s = 1/2)")
    bx.axhline(knee, color="#52514e", ls=":", lw=1)
    bx.axhspan(knee, 5.2, color="#e1e0d9", alpha=0.5, label="above the knee: not used to split V_tot")
    ax.set(xlabel="voltage (V)", ylabel="current, code 15 - code 0 (uA)", xlim=(0, 5.0), ylim=(0, 150),
           title="The transistor stack (M1-M4 + M5), from the best short")
    bx.set(xlabel="current (uA)", ylabel="V_ds (V)", xlim=(0, 145), ylim=(0, 5.0),
           title="What is subtracted: V_ds at the pixel's current")
    for a in (ax, bx):
        a.grid(alpha=0.3)
        a.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(DRIVERS / "transistor.png", dpi=150)
    plt.close(fig)

    codes = P["codes"]
    L = [f"# The pixel transistor (updated {P['updated']})", "",
         "Made by `analysis/method_maps.py`, which runs the method notebook "
         "(`../led_voltage/led_voltage_method.ipynb`, section 3) and writes its transistor here. "
         "Older files in this folder: see the end.", "",
         "## What every analysis uses now: the measured curve, no model", "",
         "- **The stack:** VLED -> M1-M4 (four PMOS in parallel, switched by the code) -> M5 (thick-oxide PMOS, "
         "15/0.55 um, the on/off switch) -> LED -> NVLED. The curve is all of M1-M4 and M5 at code 15.",
         f"- **Array42:** the best short of {s42['run']} (X={best['Array42'][0]}, Y={best['Array42'][1]}), measured "
         f"from 0 V. Its own path R_short = {P['R_short_Ohm'] / 1e3:.2f} kOhm is removed (V_ds = V_tot - I R_short), "
         "then the curve is interpolated linearly. No fit function.",
         f"- **Array38 (assumed):** its shorts ({s38['run']}) start at 0.8 V, above the knee. So it takes Array42's curve, "
         f"scaled by f_38 = {tr['Array38']:.3f}: Array38's best short ({best['Array38'][0]}, {best['Array38'][1]}) over "
         f"Array42's at V_tot = {P['f_read_at_V_tot_V']} V, code 15, deep in saturation where the paths hardly matter.",
         "- **Why the best short:** shorts are random defects, each with its own path. The best one has the least, so "
         "it is the closest to a bare transistor: the transistors on the array are at least this good.",
         f"- **The knee:** V_ds = {knee:.3f} V, where d ln I / d ln V_ds = 1/2. Array42 reaches it at "
         f"{P['knee_current_A']['Array42'] * 1e6:.0f} uA, Array38 at {P['knee_current_A']['Array38'] * 1e6:.0f} uA. "
         "Above it a 1 % current error moves V_ds by far more than below, so only points below the knee split V_tot.",
         f"- **V_ds of Array42's transistor:** " + ", ".join(f"{1e3 * v:.0f} mV at {i * 1e6:.0f} uA" for i, v in at.items()) + ".",
         "", "## The files", "",
         "- `transistor.csv`: V_ds_V, I_Array42_A, I_Array38_A (= f_38 x Array42), below_knee. "
         "Use `np.interp(I, I_Array42_A, V_ds_V)` for V_ds at a current, `np.interp(V_ds, V_ds_V, I_Array42_A)` the other way.",
         "- `transistor.json`: every number below, and the runs it comes from.",
         "- `transistor.png`: left, the shorts as measured and the transistor; right, V_ds against the current.",
         "", "## How the curve is made (notebook section 3)", "",
         f"1. **The best short.** Of Array42's 10 best shorts (the top 10 of its survey {P['runs']['Array42_survey']}), "
         f"{int(tab['ohmic'].sum())} pass the ohmic test (the current ratio between {nb['V_LOW']} and {nb['V_HIGH']} V "
         f"within 10 % of {expected:.2f}). Of those, the one with the most current at {nb['V_HIGH']} V wins: "
         f"({best['Array42'][0]}, {best['Array42'][1]}).",
         f"2. **Per code:** the resistance at 0 V (R_c, from a fit below 0.2 V) and the saturation current at "
         f"{P['V_sat_V']:.0f} V:", "",
         "   | code | R_c (kOhm) | I_sat (uA) | the stack model (uA) | V across M1-M4 (V) |", "   |---|---|---|---|---|"]
    L += [f"   | {c} | {r / 1e3:.2f} | {i * 1e6:.1f} | {m[0] * 1e6:.1f} | {m[1]:.3f} |"
          for c, r, i, m in zip(codes, P["R_c_Ohm"], P["I_sat_A"], model)]
    L += ["",
          f"3. **The series resistance, from the parallel rule:** code 15 must be codes 1, 2, 4 and 8 in parallel once "
          f"the common series part is removed: 1/(R_15 - R_s) = sum 1/(R_c - R_s). That holds for R_s = "
          f"{P['R_s_Ohm'] / 1e3:.2f} kOhm = R_M5 + R_short.",
          f"4. **M5, from the saturation currents:** M5 saturated in series with M1-M4 in the linear region fits the five "
          f"I_sat with V_ov = {P['V_ov_V']:.3f} V (M1-M4), beta_5 = {P['beta5_A_per_V2'] * 1e3:.3f} mA/V2 and "
          f"V_on5 = {P['V_on5_V']:.3f} V. M1-M4 never saturate, M5 caps the current. Fully on, M5 is "
          f"R_M5 = {P['R_M5_Ohm'] / 1e3:.2f} kOhm, so the short's own path is R_short = R_s - R_M5 = "
          f"{P['R_short_Ohm'] / 1e3:.2f} kOhm.",
          "5. **Checks (notebook 3f):** the measured beta of codes 1-8 follow the design widths (0.54, 1.2, 2.59, 6.8 um, "
          f"L 2 um) up to one width offset of {nb['dW']:+.2f} um. Reading I_sat at 2-5 V instead of {P['V_sat_V']:.0f} V "
          f"gives R_short = {min(r_short) / 1e3:.2f} to {max(r_short) / 1e3:.2f} kOhm, which moves V_ds by at most "
          f"{(max(r_short) - min(r_short)) * 100e-6 * 1e3:.0f} mV at 100 uA.",
          "6. **Remove the path:** V_ds = V_tot - I R_short on the best short at code 15, the current made monotonic.",
          "", "## Superseded", "",
          "- **The smooth-knee model** I = I15 (1 + lam (V_ds - 5)) x / (1 + x^m)^(1/m), x = V_ds / V_ov, fitted to the "
          "median of the 10 best shorts with a short path of 1.72 kOhm and a source resistance R_src = 1.0 kOhm kept "
          "in the driver (`bond_resistance.py` before 2026-11-17, `led_voltage_budget.py`). It fitted to 1 %, but its "
          "parameters carry no physics here. What it called R_src is mostly M5 "
          f"(R_M5 = {P['R_M5_Ohm'] / 1e3:.2f} kOhm), which was not in the picture then.",
          "- `bond_resistance.py`, `bond_sweep.py` and the scripts that import them now read `transistor.csv`.",
          "", "## Still valid: the driver statistics (`summary.md`, `analysis/driver_stats.py`, 2026-11-05)", "",
          "Spread of the drivers across the array (CV per code and bias, per quadrant), code linearity, the best shorts "
          "per run, shorts that emit light (`emitting_shorts.csv`), isolation of the shorts (`shorts_isolation.*`) and "
          "the Array42 drift (`a42_drift_shorts.csv`). These are statistics of the measured shorts and need no model. "
          "Its 'saturation' and 'r_out' lines describe the shorts as measured (their path included)."]
    (DRIVERS / "README.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    return P


# ---- 2. every pixel -------------------------------------------------------------------------------------
def key(model):
    return model.split(":")[0]                  # "G2: die barrier, ..." -> "G2"


def subset(nb):
    """The models of the notebook's EXTRA that are fitted here (MODELS), as the notebook defines them."""
    return {m: s for m, s in nb["EXTRA"].items() if key(m) in MODELS}


def equations(nb):
    """The fitted models as the notebook writes them, for the figures."""
    band = f", a within +-{100 * nb['A_BAND']:.0f} % of a_die" if SHIFT == "G2" else ""
    return ("extra = V_ex - V_die: what a pixel needs on top of the die.   g(I; I_b) = ln(1 + I/I_b) - ln(1 + I0/I_b)\n"
            + "     ".join(f"{key(m)}: {s['eq']}" + (band if key(m) == "G2" else "") for m, s in subset(nb).items())
            + f"\nthe winner: the simplest model (fewest parameters) within {nb['RMS_FACTOR']:g} x the lowest rms")


def fit_rows(f, arr, keys, I, P, sP, V, ok, v_led, light, anchor):
    """Sections 6, 7 and 9 of the notebook for a block of pixels (rows sorted by pixel), one row each.
    f: the notebook's I0 and functions."""
    I0, V_DIE = f["I0"], f["V_DIE"]
    cut = np.flatnonzero((np.diff(keys, axis=0) != 0).any(axis=1)) + 1
    rows = []
    for a, b in zip(np.r_[0, cut], np.r_[cut, len(keys)]):
        row = dict(X=int(keys[a, 0]), Y=int(keys[a, 1]))
        rows.append(row)
        lo = I[a:b][anchor[a:b]]
        row["I_low_uA"] = 1e6 * lo.min() if lo.size else np.nan
        if not lo.size or lo.min() > ANCHOR_MAX * I0:
            row["skip"] = "no light near I0"
            continue
        c = dict(name=f"{arr} ({row['X']}, {row['Y']})", arr=arr, I=I[a:b], P=P[a:b], sP=sP[a:b], ok=ok[a:b],
                 V_LED=v_led[a:b], light=light[a:b])
        c["V_j"] = f["junction_from_light"](c)                                # eq. (9)
        c["V_ex"] = np.where(c["ok"], c["V_LED"] - c["V_j"], np.nan)          # eq. (12)
        c["fit_I"], c["fit_V"] = f["points"](c)
        row["n"] = len(c["fit_I"])
        if row["n"] < N_MIN:
            row["skip"] = f"fewer than {N_MIN} points"
            continue
        c["extra"] = c["fit_V"] - V_DIE(c["fit_I"])                           # on top of the die
        k = np.argmax(c["fit_I"])
        row.update(I_max_uA=1e6 * c["fit_I"][k], V_ex_mV=1e3 * c["fit_V"][k], extra_mV=1e3 * c["extra"][k])
        r14 = np.flatnonzero(np.isclose(V[a:b], V_REL) & np.isfinite(c["V_ex"]) & c["light"])
        if r14.size:                                                          # the measured extra at the fast scans' bias
            row.update(I14_uA=1e6 * c["I"][r14[0]], extra14_mV=1e3 * (c["V_ex"][r14[0]] - V_DIE(c["I"][r14[0]])))
        add = {m: f["fit_extra"](c, spec) for m, spec in f["EXTRA"].items()}
        best = f["pick_best"](add)
        row.update(best=key(best), best_rms_mV=1e3 * add[best]["rms"], best_limit=add[best]["limit"],
                   best_R2=add[best]["p"].get("R2", np.nan))
        for m, r in add.items():
            row[f"{key(m)}_rms_mV"] = 1e3 * r["rms"]
            row.update({f"{key(m)}_{p}": v for p, v in r["p"].items()})
            row[f"{key(m)}_limit"] = r["limit"]
    return pd.DataFrame(rows)                    # a frame, not dicts: 100 000 rows of dicts take ~0.5 GB


def pixel_fits(nb, arr, sample=None, seed=0):
    """Every pixel of SWEEPS[arr] through the notebook: the points used and the latch check (section 1), the
    transistor (5), the junction (6), the excess (7), every model of EXTRA and the best one (9)."""
    run = run_dir(SWEEPS[arr])
    sw = pc.load_sweep(run)
    vled = nb["read_json"](next((run / "NVLEDSweep/Sweep/raw_data").glob("*_Sweep_config.json")))["settings"]["vled_voltage"]
    # section 1, latched: the sweep at -1.4 V draws > 5 % and >= 5 uA more than the fast scan
    fs = pd.read_csv(next((run / "FastScan").glob("*_aggregated_bitval=*.csv")),
                     usecols=["X", "Y", "NVLED_V", "NVLED_CURR_MEAN"])
    f14 = (fs[fs.NVLED_V.round(2) == -1.4].drop_duplicates(["X", "Y"], keep="last")
           .set_index(["X", "Y"]).NVLED_CURR_MEAN.abs())
    i14 = sw[sw.V.round(3) == -1.4].groupby(["X", "Y"]).I_tot.max()
    f14 = f14.reindex(i14.index)
    latched = i14.index[(i14 > 1.05 * f14) & (i14 - f14 >= 5e-6)]
    # section 1, the points used: a reference taken, and not partly lit (above 1.3 x the bias's median)
    xy = pd.MultiIndex.from_frame(sw[["X", "Y"]])
    swept = int(xy.unique().size)
    sw = sw[sw.I_ref.notna() & ~(sw.I_ref > 1.3 * sw.groupby("V").I_ref.transform("median")) & ~xy.isin(latched)]
    del xy, fs
    if sample is not None:                       # N random pixels, or a list of (X, Y)
        pick = (sw[["X", "Y"]].drop_duplicates().sample(sample, random_state=seed) if isinstance(sample, int)
                else pd.DataFrame(sample, columns=["X", "Y"]))
        sw = sw[pd.MultiIndex.from_frame(sw[["X", "Y"]]).isin(pd.MultiIndex.from_frame(pick))]
    sw = sw.sort_values(["X", "Y", "V"], ascending=[True, True, False])
    I, P, sP, V = (sw[k].to_numpy(float) for k in ("I_LED", "P", "P_noise", "V"))
    vds = nb["v_transistor"](np.clip(I, 1e-13, None), nb["tr"][arr])         # eq. (5)
    ok = vds < nb["V_KNEE"]                                                   # below the knee (section 4)
    cols = dict(I=I, P=P, sP=sP, V=V, ok=ok, v_led=vled - V - vds,           # eqs. (2), (7)
                light=(P > 5 * sP) & (I > 0))                                 # section 5
    cols["anchor"] = cols["light"] & ok & (P >= nb["ANCHOR_SIGMA"] * sP)      # eq. (11)
    # the notebook's functions travel to the worker processes with what they use (joblib pickles them)
    f = {k: nb[k] for k in ("I0", "junction_from_light", "points", "V_DIE", "fit_extra", "pick_best")}
    f["EXTRA"] = subset(nb)                                                   # only the models of MODELS
    keys = sw[["X", "Y"]].to_numpy()
    del sw                                                                    # the arrays hold all that is needed
    gc.collect()
    starts = np.r_[0, np.flatnonzero((np.diff(keys, axis=0) != 0).any(axis=1)) + 1]
    edges = np.r_[starts[::CHUNK], len(keys)]                                 # blocks of whole pixels
    t0 = time.time()
    tasks = [(f, arr, keys[a:b], *(v[a:b] for v in cols.values())) for a, b in zip(edges, edges[1:])]
    try:
        from joblib import Parallel, delayed     # pickles the notebook's lambdas for the workers
        parts = Parallel(n_jobs=JOBS if len(tasks) > 1 else 1, verbose=5)(delayed(fit_rows)(*t) for t in tasks)
    except ImportError:                          # ponytail: serial without joblib, ~1.5 h for both arrays
        parts = [fit_rows(*t) for t in tasks]
    d = pd.concat(parts, ignore_index=True)
    # readable columns from the notebook's own definitions (section 9 show, eq. 21)
    VT, A_D, LGIB_D, g = nb["VT"], nb["A_D"], nb["LGIB_D"], nb["g"]
    for m in ("G", "G2"):
        if f"{m}_lgIb" in d:
            d[f"{m}_Ib_ratio"] = 10 ** (d[f"{m}_lgIb"] - LGIB_D)                 # I_b / I_b,die
            d[f"{m}_dPhi_meV"] = -1e3 * VT * np.log(d[f"{m}_Ib_ratio"])          # if thermal
    if "G2_a" in d:
        d["G2_a_ratio"] = d["G2_a"] / A_D
    if "B_a2" in d:
        d["B_Ib2_uA"] = 1e6 * 10 ** d["B_lgIb2"]
        d["B_n_b"] = d["B_a2"] / VT
        d["B_V_map_mV"] = 1e3 * d["B_a2"] * g(I_MAP, 10 ** d["B_lgIb2"])          # what it adds at I_MAP
        astar = np.mean(nb.get("ASTAR", (24.0, 120.0)))
        d["B_Phi_eV"] = VT * np.log(nb["AREA"] * astar * nb["T"] ** 2 / 10 ** d["B_lgIb2"])   # eq. (21)
    return d, dict(run=run.name, swept=swept, latched=len(latched), seconds=time.time() - t0)


# ---- 3. maps --------------------------------------------------------------------------------------------
def cmap(colors, name):
    from matplotlib.colors import LinearSegmentedColormap
    return LinearSegmentedColormap.from_list(name, colors)


def image(d, values, gap=0):
    """The values (one per row of d, or one RGBA per row) on the pixel grid, NaN where there is no pixel. gap > 0: an
    empty pixel shows its nearest pixel within gap px, so the map of a sampled sweep reads as an area."""
    img = np.full((N, N) + np.shape(values)[1:], np.nan)
    img[d.Y.to_numpy(), d.X.to_numpy()] = np.asarray(values, float)
    if gap:
        from scipy.ndimage import distance_transform_edt
        dist, idx = distance_transform_edt(np.isnan(img if img.ndim == 2 else img[..., 0]), return_indices=True)
        img = img[tuple(idx)]
        img[dist > gap] = np.nan
    return img


def look(d):
    """How to draw an array: a sampled sweep (Array42's 1703 took about one pixel in five) gets its gaps filled up to
    3 px and 16 px blocks; a sweep of every pixel of an area (Array38) gets neither and 8 px blocks."""
    occ = np.zeros((N, N))
    occ[d.Y.to_numpy(), d.X.to_numpy()] = 1
    cover = occ.reshape(N // 16, 16, N // 16, 16).mean(axis=(1, 3))
    if np.median(cover[cover > 0]) >= 0.5:
        return dict(gap=0, B=8, note="")
    return dict(gap=3, B=16, note=f"; the sweep took {100 * occ.mean():.0f} % of the pixels, gaps up to 3 px show the nearest one")


def blocks(img, B, stat=np.nanmedian, min_n=10):
    h = N // B
    with np.errstate(all="ignore"):
        t = img.reshape(h, B, h, B).transpose(0, 2, 1, 3).reshape(h, h, B * B)
        out = stat(t, axis=2)
    out[np.isfinite(t).sum(axis=2) < min_n] = np.nan
    return out


def show(ax, img, title, **kw):
    im = ax.imshow(img, origin="upper", interpolation="nearest", extent=(-0.5, N - 0.5, N - 0.5, -0.5), **kw)
    ax.set(title=title, xlabel="X (px)", ylabel="Y (px)")
    ax.title.set_fontsize(9)
    ax.grid(False)
    return im


def ratio_norm(v):
    """A log scale for I_b / I_b,die over the 5-95 % range of v that always holds 1, the die's own turn-on."""
    from matplotlib.colors import LogNorm
    lo, hi = np.nanpercentile(v, [5, 95])
    return LogNorm(min(lo, 0.95), max(hi, 1.05))


def plain_ticks(cb, norm, n=6):
    """Decimal ticks on a log colour bar (1.4, 1.7, 2 instead of 1.4 x 10^0), with 1, the die, among them."""
    t = np.unique(np.r_[np.round(np.geomspace(norm.vmin, norm.vmax, n), 2), 1.0])
    t = t[(t >= norm.vmin) & (t <= norm.vmax)]
    cb.set_ticks(t)
    cb.set_ticklabels([f"{v:g}" + (" (the die)" if v == 1 else "") for v in t])
    cb.ax.minorticks_off()


def finish(fig, info, name, plt):
    """The models' equations at the foot of the figure, then save."""
    fig.text(0.01, 0.01, info["eqs"], fontsize=7.5, ha="left", va="bottom", family="monospace", color="#52514e")
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    fig.savefig(OUT / name, dpi=150)
    plt.close(fig)


def fig_models(arr, d, info, plt):
    """Left: every pixel in the colour of its best model: A grey, B by its second barrier, the die's-barrier model
    (G2) by its turn-on I_b. Right: the share of pixels per block where B wins."""
    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import Normalize, to_rgba
    from matplotlib.patches import Patch
    lk, fit = look(d), d[d.best.notna()]
    S, B, A = (d.best.eq(m).to_numpy() for m in (SHIFT, "B", "A"))
    cs, cb = cmap(RAMP[SHIFT], SHIFT), cmap(RAMP["B"], "B")
    ns = ratio_norm(d[f"{SHIFT}_Ib_ratio"][S])
    nb_ = Normalize(0, max(1.0, float(np.nanpercentile(d.B_V_map_mV[B], 95))) if B.any() else 50)
    col = np.tile(to_rgba(SKIPPED), (len(d), 1))                     # one RGBA per pixel
    col[A] = to_rgba(GREY[2])
    col[S] = cs(ns(d[f"{SHIFT}_Ib_ratio"][S].clip(ns.vmin, ns.vmax)))
    col[B] = cb(nb_(d.B_V_map_mV[B].clip(0, nb_.vmax)))
    ok = d.best.notna().to_numpy()
    rgba = image(d[ok], col[ok], lk["gap"])                          # gaps take the nearest fitted pixel
    sk = d[~ok]
    rgba[sk.Y.to_numpy(), sk.X.to_numpy()] = to_rgba(SKIPPED)         # the unfitted ones only where they are
    rgba[np.isnan(rgba[..., 0])] = to_rgba(NOT_SWEPT)
    pct = lambda s: f"{s.sum()}, {100 * s.sum() / len(fit):.0f} %"
    handles = [Patch(color=GREY[2], label=f"A best ({pct(A)}): the die's barrier as it is, plus R2"),
               Patch(color=SKIPPED, label=f"not fitted: {(~ok).sum()} (no light near I0, or < {N_MIN} points)")]
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(16, 8.4), gridspec_kw=dict(width_ratios=[1.25, 1]))
    show(ax, rgba, f"{arr}, sweep {info['run'][:11]}: the best of {', '.join(MODELS)} per pixel"
         + lk["note"].replace("; ", "\n"))
    cbs = fig.colorbar(ScalarMappable(ns, cs), ax=ax, shrink=0.8, pad=0.01, extend="both")
    cbs.set_label(f"{SHIFT} best ({pct(S)}): I_b / I_b,die, the die's barrier turning on later (> 1) or earlier")
    plain_ticks(cbs, ns)
    cbb = fig.colorbar(ScalarMappable(nb_, cb), ax=ax, shrink=0.8, pad=0.04, extend="max")
    cbb.set_label(f"B best ({pct(B)}): its second barrier at {I_MAP * 1e6:.0f} uA (mV)")
    ax.legend(handles=handles, fontsize=7.5, loc="upper center", bbox_to_anchor=(0.5, -0.07), ncol=1, frameon=False)
    share = blocks(image(fit, (fit.best == "B").astype(float)), BLOCK, np.nanmean)
    im = show(bx, 100 * share, f"share of pixels where B (a second barrier) wins, {BLOCK} x {BLOCK} px blocks",
              cmap=cmap(ORANGE, "Bshare"), vmin=0, vmax=100)
    fig.colorbar(im, ax=bx, shrink=0.8, label="pixels with B best (%)")
    finish(fig, info, f"models_{arr}.png", plt)


def fig_r2(arr, d, info, plt):
    """R2 of each pixel's best model. Left: every pixel, its hue the model (blue the die's-barrier model, grey A,
    orange B), its shade R2, one scale for all. Right: the model-free resistance, the measured extra at -1.4 V over
    the current, block medians: R2 of B is not comparable (it sits on -R_die where B's barrier takes the voltage)."""
    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import Normalize, to_rgba
    lk, fit = look(d), d[d.best.notna()]
    r2 = fit.best_R2 / 1e3
    norm = Normalize(*np.nanpercentile(r2, [5, 95]))
    col = np.zeros((len(fit), 4))
    for m in MODELS:
        s = fit.best.eq(m).to_numpy()
        col[s] = cmap(RAMP[m], m)(norm(r2[s].clip(norm.vmin, norm.vmax)))
    rgba = image(fit, col, lk["gap"])
    rgba[np.isnan(rgba[..., 0])] = to_rgba(NOT_SWEPT)
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(17, 7.6), gridspec_kw=dict(width_ratios=[1.3, 1]))
    show(ax, rgba, f"{arr}: R2 of each pixel's best model; the hue marks the model")
    for m in MODELS[::-1]:
        n = fit.best.eq(m).sum()
        cb = fig.colorbar(ScalarMappable(norm, cmap(RAMP[m], m)), ax=ax, shrink=0.8, pad=0.015, aspect=35, extend="both")
        cb.set_label(f"{m} best ({n}, {100 * n / len(fit):.0f} %): R2 (kOhm)", fontsize=8)
    r14 = fit.extra14_mV / fit.I14_uA                                 # mV / uA = kOhm, whichever model
    im = show(bx, blocks(image(fit, r14), lk["B"]), f"{arr}: the measured extra at -1.4 V over the current, (V_ex - V_die) / I,\n"
              f"{lk['B']} x {lk['B']} px block medians: model-free, so B pixels compare with the rest",
              cmap=cmap(VIOLET, "R14"), norm=Normalize(*np.nanpercentile(r14, [5, 95])))
    fig.colorbar(im, ax=bx, shrink=0.8, extend="both", label="(V_ex - V_die) / I at -1.4 V (kOhm)")
    fig.suptitle(f"{arr}, sweep {info['run'][:11]}: R2, the resistance on top of the die's {info['R_D'] / 1e3:.2f} kOhm, "
                 f"median {r2.median():.2f} kOhm\nWhere B wins, R2 often sits on its lower limit (-R_die): its barrier "
                 "then carries the resistance" + lk["note"].replace("; ", "\n"), fontsize=9.5)
    finish(fig, info, f"R2_{arr}.png", plt)


def fig_shift(arr, d, info, a_band, plt):
    """The die's-barrier model (G2) fitted to every pixel, whichever model wins there: its turn-on I_b, its slope,
    and the rms of the best model."""
    from matplotlib.colors import Normalize, TwoSlopeNorm
    lk, fit = look(d), d[d.best.notna()]
    ratio = fit[f"{SHIFT}_Ib_ratio"]
    ns = ratio_norm(ratio)
    panels = [(ratio, f"{SHIFT} (every pixel): I_b / I_b,die, the turn-on of the die's barrier", ns, BLUE,
               "I_b / I_b,die", "both")]
    if SHIFT == "G2":
        lim = fit.G2_limit.fillna("")
        panels.append((fit.G2_a_ratio, f"G2 (every pixel): slope a / a_die, free within +-{100 * a_band:.0f} %\n"
                       f"on the low edge: {100 * lim.str.contains('a low').mean():.0f} %, on the high edge: "
                       f"{100 * lim.str.contains('a high').mean():.0f} % of the pixels",
                       TwoSlopeNorm(1.0, 1 - a_band, 1 + a_band), DIVERGING, "a / a_die", "neither"))
    panels.append((fit.best_rms_mV, "fit quality: rms of the best model",
                   Normalize(0, float(np.nanpercentile(fit.best_rms_mV, 95))), ["#f9f9f7"] + GREY[1:], "rms (mV)", "max"))
    fig, axs = plt.subplots(1, len(panels), figsize=(6.6 * len(panels), 7.2), squeeze=False)
    for ax, (v, t, norm, cols, lab, ext) in zip(axs.flat, panels):
        im = show(ax, image(fit, v, lk["gap"]), t, cmap=cmap(cols, "m"), norm=norm)
        cb = fig.colorbar(im, ax=ax, shrink=0.8, label=lab, extend=ext)
        if norm is ns:
            plain_ticks(cb, ns)
    fig.suptitle(f"{arr}, sweep {info['run'][:11]}: the die's barrier in every pixel ({len(fit)} pixels{lk['note']})",
                 fontsize=11)
    finish(fig, info, f"{SHIFT}_{arr}.png", plt)


def relative(arr):
    """The same run's fast-scan R_best (bond_resistance/pixels) and R_local (bond_resistance/local), per pixel."""
    out = []
    for sub, col in (("pixels", "R_best_kOhm"), ("local", "R_local_kOhm")):
        f = BOND / sub / f"{RELATIVE[arr]}.csv"
        if f.exists():
            out.append(pd.read_csv(f, usecols=["X", "Y", col]).set_index(["X", "Y"])[col])
    return pd.concat(out, axis=1) if out else pd.DataFrame()


def fig_absolute_vs_relative(dd, plt):
    """The extra on top of the die at -1.4 V over the current (absolute), against R_best and R_local of the same
    pixels in the same run's fast scan at -1.4 V (relative to the array's best pixels, or to the block's best)."""
    from scipy.stats import spearmanr
    rows = []
    fig, axs = plt.subplots(2, 2, figsize=(13, 11), squeeze=False)
    for j, (arr, d) in enumerate(dd.items()):
        r = relative(arr)
        x = d.set_index(["X", "Y"])
        x = x[x.best.notna() & x.extra14_mV.notna()].join(r, how="inner")
        x["R_abs"] = x.extra14_mV / x.I14_uA                                   # mV / uA = kOhm
        for k, col in enumerate(("R_best_kOhm", "R_local_kOhm")):
            ax = axs[j, k]
            if col not in x:
                ax.set_visible(False)
                continue
            q = x[[col, "R_abs"]].dropna()
            lim = np.nanpercentile(np.r_[q[col], q.R_abs], [1, 98])
            ax.hexbin(q[col], q.R_abs, gridsize=80, extent=(*lim, *lim), mincnt=1, bins="log", cmap=cmap(BLUE, "h"))
            ax.plot(lim, lim, "-", color="#52514e", lw=0.8, label="1:1")
            rho = spearmanr(q[col], q.R_abs)[0]
            zero = q[q[col].abs() < 0.05].R_abs.median()
            ax.axhline(zero, color=ORANGE[2], lw=1, ls="--",
                       label=f"pixels at {col.split('_')[1]} ~ 0: {zero:.2f} kOhm above the die")
            ax.set(xlim=lim, ylim=lim, xlabel=f"{col.replace('_kOhm', '')} (kOhm): fast scan {RELATIVE[arr]}, relative",
                   ylabel="(V_ex - V_die) / I at -1.4 V (kOhm): sweep, absolute (vs the die)",
                   title=f"{arr}: {len(q)} pixels, Spearman {rho:.2f}")
            ax.legend(fontsize=7, loc="upper left")
            rows.append(dict(array=arr, against=col.replace("_kOhm", ""), n=len(q), spearman=rho, zero_point_kOhm=zero,
                             median_relative=q[col].median(), median_absolute=q.R_abs.median()))
    fig.tight_layout()
    fig.savefig(OUT / "absolute_vs_relative.png", dpi=150)
    plt.close(fig)
    return pd.DataFrame(rows)


# ---- 4. the summary -------------------------------------------------------------------------------------
def q(s, f="{:.2f}", ps=(10, 50, 90)):
    s = pd.Series(s).dropna()
    return " / ".join(f.format(v) for v in np.percentile(s, ps)) if len(s) else "-"


def summary(nb, dd, infos, comp):
    S = SHIFT
    L = [f"# Every pixel through the method notebook (method_maps.py, {date.today()})", "",
         f"The notebook's section 9 on every swept pixel, with the models {', '.join(MODELS)} of its EXTRA: the die's "
         "excess comes off, each model is fitted to what is left, and the simplest model within RMS_FACTOR of the "
         "lowest rms wins. Same code, same settings:",
         f"n_R = {nb['N_R']:.3f}, I0 = {nb['I0'] * 1e6:.3f} uA, the die {1e3 * nb['A_D']:.1f} mV / "
         f"{1e6 * 10 ** nb['LGIB_D']:.2f} uA / {nb['R_D'] / 1e3:.2f} kOhm (a / I_b / R, IMAX {nb['IMAX'] * 1e6:.0f} uA), "
         f"RMS_FACTOR {nb['RMS_FACTOR']}, A_BAND {nb['A_BAND']}. Models: "
         + "; ".join(f"{m} = {s['eq']}" for m, s in subset(nb).items()) + ".", "",
         f"A pixel whose I_b equals the die's (and, in G2, whose slope does too) has no barrier term left: it is model A, "
         f"and A wins it with fewer parameters. So {S} wins only where the turn-on really moved.", "",
         f"Pixels used: a pixel needs light within {ANCHOR_MAX} x I0 (its V_j is anchored at I0) and at least {N_MIN} "
         "points with light between I0 and the knee. Latched pixels and partly lit references are left out as in the notebook.",
         "", "| | " + " | ".join(dd) + " |", "|---|" + "---|" * len(dd)]
    def line(label, f):
        L.append(f"| {label} | " + " | ".join(f(a, d) for a, d in dd.items()) + " |")
    line("sweep", lambda a, d: infos[a]["run"])
    line("pixels swept / latched", lambda a, d: f"{infos[a]['swept']} / {infos[a]['latched']}")
    line("not fitted: no light near I0 / too few points",
         lambda a, d: f"{(d.skip == 'no light near I0').sum()} / {d.skip.str.startswith('fewer').sum()}")
    line("fitted", lambda a, d: f"{d.best.notna().sum()}")
    line("points per pixel (10/50/90 %)", lambda a, d: q(d.n[d.best.notna()], "{:.0f}"))
    for m in MODELS:
        line(f"**{m} best**", lambda a, d, m=m: f"{(d.best == m).sum()} ({100 * (d.best == m).mean() / d.best.notna().mean():.1f} %)")
    line("rms of the best model (mV)", lambda a, d: q(d.best_rms_mV))
    line("best model on a limit", lambda a, d: f"{100 * (d.best_limit.fillna('') != '')[d.best.notna()].mean():.1f} %")
    line("R2 of the best model (kOhm)", lambda a, d: q(d.best_R2 / 1e3))
    line("where A wins: R2 (kOhm)", lambda a, d: q(d.A_R2[d.best == "A"] / 1e3))
    line(f"where A wins: {S}'s I_b / I_b,die", lambda a, d: q(d[f"{S}_Ib_ratio"][d.best == "A"]))
    line(f"where {S} wins: I_b / I_b,die", lambda a, d: q(d[f"{S}_Ib_ratio"][d.best == S]))
    line(f"where {S} wins: barrier height vs the die if thermal (meV)", lambda a, d: q(d[f"{S}_dPhi_meV"][d.best == S], "{:+.0f}"))
    line(f"where {S} wins: R2 (kOhm)", lambda a, d: q(d[f"{S}_R2"][d.best == S] / 1e3))
    if S == "G2":
        line("G2 (every pixel): a / a_die", lambda a, d: q(d.G2_a_ratio, "{:.3f}"))
        line("G2 (every pixel): slope on the band's low / high edge",
             lambda a, d: f"{100 * d.G2_limit.fillna('').str.contains('a low')[d.best.notna()].mean():.0f} % / "
                          f"{100 * d.G2_limit.fillna('').str.contains('a high')[d.best.notna()].mean():.0f} %")
    line(f"{S} (every pixel): I_b / I_b,die", lambda a, d: q(d[f"{S}_Ib_ratio"]))
    line("where B wins: a2 (mV)", lambda a, d: q(1e3 * d.B_a2[d.best == "B"], "{:.0f}"))
    line("where B wins: n_b = a2/(kT/q)", lambda a, d: q(d.B_n_b[d.best == "B"], "{:.1f}"))
    line("where B wins: I_b2 (uA)", lambda a, d: q(d.B_Ib2_uA[d.best == "B"], "{:.2g}"))
    line(f"where B wins: second barrier at {I_MAP * 1e6:.0f} uA (mV)", lambda a, d: q(d.B_V_map_mV[d.best == "B"], "{:.0f}"))
    line("where B wins: Phi_B2 if thermal, eq. (21) (eV)", lambda a, d: q(d.B_Phi_eV[d.best == "B"], "{:.2f}"))
    line("where B wins: R2 (kOhm)", lambda a, d: q(d.B_R2[d.best == "B"] / 1e3))
    L += ["", "Values: 10 / 50 / 90 % of the fitted pixels.", ""]
    if len(comp):
        L += ["## Absolute against relative (`absolute_vs_relative.png`)", "",
              "Absolute: the measured excess on top of the die at -1.4 V, over the current, (V_ex - V_die)/I, from the sweep. "
              "Relative: R_best (against the array's best 2 %) and R_local (against the 32 x 32 block's best) of the same "
              "pixels, from the same run's fast scan at -1.40 V (`../pixels/`, `../local/`).", "",
              "| array | against | pixels | Spearman | absolute where the relative is ~0 (kOhm) | median relative | median absolute |",
              "|---|---|---|---|---|---|---|"]
        L += [f"| {r.array} | {r.against} | {r.n} | {r.spearman:.2f} | {r.zero_point_kOhm:.2f} | {r.median_relative:.2f} | "
              f"{r.median_absolute:.2f} |" for r in comp.itertuples()]
        L += [""]
    L += ["## The winner by R_best decile", "",
          "| array | R_best decile (kOhm) | pixels | " + " | ".join(f"{m} best (%)" for m in MODELS) + " |",
          "|---|---|---|" + "---|" * len(MODELS)]
    for arr, d in dd.items():
        x = d.set_index(["X", "Y"]).join(relative(arr), how="inner")
        x = x[x.best.notna()]
        if "R_best_kOhm" not in x or not len(x):
            continue
        b = pd.qcut(x.R_best_kOhm, 10, duplicates="drop")
        for iv, g in x.groupby(b, observed=True):
            L.append(f"| {arr} | {iv.left:.2f} to {iv.right:.2f} | {len(g)} | "
                     + " | ".join(f"{100 * (g.best == m).mean():.0f}" for m in MODELS) + " |")
    L += ["", "## Files", "",
          "- `<array>_models.csv`: one row per swept pixel. X, Y; I_low_uA (its lowest light), skip (why not fitted); "
          "n (fit points), I_max_uA, V_ex_mV and extra_mV at its highest fitted point; I14_uA and extra14_mV at -1.4 V; "
          "best (the winning model), best_rms_mV, best_limit, best_R2 (Ohm); per model <M>_rms_mV, its parameters in SI "
          "units (a, a2 in V; lgIb, lgIb2 = log10 of the current in A; R2 in Ohm) and <M>_limit; "
          f"{S}_Ib_ratio, {S}_dPhi_meV, " + ("G2_a_ratio, " if S == "G2" else "") + "B_Ib2_uA, B_n_b, B_V_map_mV, B_Phi_eV.",
          f"- `models_<array>.png`: the best model per pixel: A grey, {S} blue by I_b / I_b,die, B orange by its second "
          f"barrier at {I_MAP * 1e6:.0f} uA; right, the share of B per {BLOCK} x {BLOCK} block.",
          f"- `R2_<array>.png`: R2 of each pixel's best model, the hue its model ({S} blue, A grey, B orange), and the "
          "model-free (V_ex - V_die) / I at -1.4 V in blocks. Where B wins, R2 often sits on its lower limit (-R_die): its barrier carries the resistance.",
          f"- `{S}_<array>.png`: {S} fitted to every pixel: its I_b, " + ("its slope, " if S == "G2" else "")
          + "and the rms of the best model.",
          "- `absolute_vs_relative.png`: the absolute excess against R_best and R_local.",
          "- Every map names the models' equations at its foot."]
    (OUT / "summary.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    return L


def check_against_notebook(nb):
    """The notebook's own pixels from the same sweep, through pixel_fits: the rms of every model of MODELS and the
    winner among them must be the notebook's (section 9). Fails loudly if the two ever drift apart. pixel_classes
    reads the sweep as float32, the notebook as float64: the rms agree to ~1e-6, not to the last digit."""
    for arr, run in SWEEPS.items():
        mine = [c for c in nb["pcs"] if c["arr"] == arr and nb["RUNS"][arr] == run]
        if not mine:
            continue
        d = pixel_fits(nb, arr, sample=[c["xy"] for c in mine])[0].set_index(["X", "Y"])
        for c in mine:
            row, add = d.loc[c["xy"]], {m: r for m, r in c["add"].items() if key(m) in MODELS}
            for m, r in add.items():
                assert np.isclose(row[f"{key(m)}_rms_mV"], 1e3 * r["rms"], rtol=1e-3), (c["name"], m, row[f"{key(m)}_rms_mV"], 1e3 * r["rms"])
            assert row.best == key(nb["pick_best"](add)), (c["name"], row.best, nb["pick_best"](add))
            print(f"  check {c['name']}: as in the notebook, {row.best} best of {', '.join(MODELS)} at {row.best_rms_mV:.2f} mV")


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")      # the model names hold Greek letters
    replot = "--replot" in sys.argv[1:]          # figures and summary only, from the CSVs of the last fit run
    nb, plt = notebook()
    print(f"notebook: {NB.name}, I0 {nb['I0'] * 1e6:.3f} uA, n_R {nb['N_R']:.3f}, models fitted {list(subset(nb))}")
    if not replot:
        check_against_notebook(nb)
        P = export_transistor(nb, plt)
        print(f"drivers: R_short {P['R_short_Ohm'] / 1e3:.2f} kOhm, f_38 {P['f']['Array38']:.3f}, knee {P['V_knee_V']:.3f} V")
    OUT.mkdir(parents=True, exist_ok=True)
    dd, infos = {}, (json.loads((OUT / "fits_info.json").read_text()) if replot else {})
    for arr in SWEEPS:
        if replot:
            d = pd.read_csv(OUT / f"{arr}_models.csv")
        else:
            d, info = pixel_fits(nb, arr)
            infos[arr] = dict(info, rms_factor=nb["RMS_FACTOR"])
            d.to_csv(OUT / f"{arr}_models.csv", index=False, float_format="%.5g")
            (OUT / "fits_info.json").write_text(json.dumps(infos, indent=1), encoding="utf-8")
        print(f"{arr}: {len(d)} pixels, {d.best.notna().sum()} fitted; best: "
              + ", ".join(f"{m} {n}" for m, n in d.best.value_counts().items()))
        dd[arr], info = d, dict(infos[arr], eqs=equations(nb), R_D=nb["R_D"])
        fig_models(arr, d, info, plt)
        fig_r2(arr, d, info, plt)
        fig_shift(arr, d, info, nb["A_BAND"], plt)
    comp = fig_absolute_vs_relative(dd, plt)
    print("\n".join(summary(nb, dd, infos, comp)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
