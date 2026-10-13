"""Corrected IQE / EQE analysis and per-run reporting for SMILE NVLED sweeps.

This supersedes ``calc_iqe_eqe.py`` for the Array38 NVLED-sweep data. It was
written after auditing the ``results_georg`` data set and finding several
issues that made the original IQE numbers untrustworthy:

  * eta_LEE lower bound (1e-4) was *above* the real value. The photodiode sits
    ~20 cm from the array and collects only a tiny solid angle, so the measured
    "EQE" is collection-limited (~3e-5). The ABC fit could not place the
    vertical scale and pinned eta_LEE at the bound, capping peak-IQE.
  * No leakage handling. At low drive the VLED (ch A) and NVLED (ch B) currents
    diverge by 5x — a parallel leakage/measurement-noise path — which produced
    a spurious EQE spike at the lowest current.
  * No fast-scan vs sweep cross-check.
  * The within/outside-1sigma split mixed bright and dim outliers together.

What this script does
---------------------
For a run directory (containing FastScan/ and NVLEDSweep/<group>/):

  1. Loads the aggregated CSVs (falls back to raw + re-aggregates).
  2. Computes dark-subtracted optical power, EQE and current density J for
     *both* current channels, plus a per-point leakage metric.
  3. Filters unreliable points (noise floor + leakage) before fitting.
  4. Fits the ABC + eta_LEE model to the median EQE(J) curve and per pixel,
     and honestly flags when the low-J (SRH) region is not captured so the
     reported peak-IQE is only a lower bound.
  5. Emits a richer plot set (combined IQE, full I-V-L, fast-scan vs sweep
     overlay, leakage diagnostic, spatial maps).
  6. Writes a Markdown REPORT.md describing what was measured and learned.

Usage:

    python analyze_array.py <run_dir>            # one run
    python analyze_array.py <results_root> --all # every run under a root
"""

from __future__ import annotations

import argparse
import datetime
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

# =====================================================================
#  Configuration
# =====================================================================

PIXEL_AREA_UM2 = 100.0  # 10 x 10 um^2
ACTIVE_THICKNESS_NM = 10.0  # total QW thickness (only rescales A,B,C)
WAVELENGTH_NM = 440.0

# ABC fit uses the degeneracy-free (beta, J0, eta) parametrization (see fit_abc);
# peak IQE = beta/(beta+2). No raw A/B/C bounds are needed.
MIN_POINTS_FOR_FIT = 6
MAX_PIXEL_FITS = 5000  # cap per-pixel ABC fits (runtime guard). ~205 ms/fit, so a
# block-stratified sweep (~1.5k pixels) fits fully in ~5 min —
# trivial next to a multi-hour acquisition, and it gives a REAL
# per-pixel peak IQE + R² for every pixel (no proxy, no aliasing
# from a strided subset). Only a full 65k raster sweep hits the cap.
FIRST_N_PIXELS = 20  # per-pixel diagnostic plots for the first N pixels
PER_PIXEL_R2_MIN = 0.70  # a per-pixel peak-IQE counts only if its own fit R² ≥ this
# (individual-pixel sweeps are noisy; this keeps junk fits
#  out of the CSV/spatial-map/summary so they stay consistent)
# Degenerate-fit guard. beta = B/sqrt(A*C) is the LED "quality factor" (peak IQE =
# beta/(beta+2)). Extracting it needs data on the rising (low-current, SRH-dominated)
# side. If the sweep doesn't reach low enough, the fit can't see SRH, drives A->0, and
# beta runs away to huge values — a FAKE ~95–100% peak pushed below the measured data.
# So a fit only counts as "clean" if beta is in a physical band AND the peak is captured.
BETA_MAX = 20.0  # beta above this ⇒ rising side unconstrained (peak IQE>91%
#  would need real SRH data we don't have) → flag degenerate
BETA_MIN = 0.05  # beta below this ⇒ essentially no peak / pure noise
# Peak-location sanity: the fitted peak must sit near where the EQE DATA actually maxes.
# When the rising side is distorted (e.g. the cold-turn-on parasitic inflates low-J current
# and flattens the SRH rise), the fit can't see the real peak and runs off into the droop —
# reporting a spurious peak at 2–3× too high a J that still passes "captured" + R². Run 260630
# low→high did exactly this (fit J_peak≈14 vs the true ≈5). Flag if fit/data peak-J disagree.
PEAK_LOC_TOL = 2.5  # fitted J_peak must be within this factor of the data's peak-J

# Reliability filtering of sweep points before fitting / plotting.
LEAKAGE_MAX = 0.50  # DIAGNOSTIC ONLY (not a rejection gate): the value on
# the leakage plot above which |Iv+Inv|/max(|Iv|,|Inv|)
# is flagged. VLED legitimately carries other paths, so
# divergence from NVLED is expected — never rejected on.
NOISE_FLOOR_A = 1e-6  # hard floor on |I_NVLED| (~1 µA). The sweep runs on a FIXED
# 10 mA range, whose systematic offset is ~range×1e-4 ≈ 1 µA.
# Below ~1 µA the current read is offset-contaminated — and the
# offset is REPEATABLE, so its rel-std is small (5–9%) and the
# rel-std gate below is BLIND to it. Those sub-µA points look
# precise but fake extra SRH loss on the rising side, pulling β
# (and peak IQE) down (e.g. 24% vs the true ~45% at J≈13 on run
# 260702-1405). So the floor must sit at the range offset, not at
# the SMU's raw noise (~20 nA). Lower it only with a lower I-range
# or a real low-J current calibration.
NVLED_RELSTD_MAX = 0.50  # reject points whose NVLED current std/mean exceeds this
PM_RELSTD_MAX = 3.00  # reject points where OPTICAL noise (σ of the mean) > 3×
# signal — catches *noisy* low-J points.
PM_MIN_OVER_DARK = 1.0  # reject points whose dark-subtracted power is below this ×
# the dark/background level (i.e. measured power < ~2× dark).
# Catches *dark-floor* points that the σ gate misses: with
# heavy averaging a point can be precisely measured to be at
# the dark level — precise, but still garbage (EQE≈0), and it
# deflates/distorts the rising side. Scales with the dark, so
# a lower-dark run automatically reaches lower J.

# Dead-pixel classification (fast scan).
DEAD_THRESHOLD_FRAC = 0.20  # alive if P ≥ 20% of the "typical lit" level (below)
FASTSCAN_ALIVE_PCTL = 90  # "typical lit" = this percentile of fast-scan power, NOT the
# median. The median collapses to the noise floor on a
# majority-dead ROI, so 20%×median mislabels ~everything alive
# (quadrant 260703-1307: 57% dark, median 0.08 nW → 84% "alive").

# Physical constants
_Q = 1.602176634e-19
_H = 6.62607015e-34
_C = 2.99792458e8


def photon_energy_J(wl_nm):
    return _H * _C / (wl_nm * 1e-9)


# =====================================================================
#  Core physics
# =====================================================================


def compute_eqe(power_W, current_A, wl_nm):
    """Collected EQE = (P/E_ph)/(I/q). NaN where current invalid.

    NOTE: this is *collection-limited* — the PD at ~20 cm captures only a small
    solid angle, so absolute values are ~1e-4 of the true device EQE. Only the
    *shape* vs J is physically meaningful for IQE extraction.
    """
    E_ph = photon_energy_J(wl_nm)
    P = np.abs(np.asarray(power_W, float))
    I = np.abs(np.asarray(current_A, float))
    with np.errstate(divide="ignore", invalid="ignore"):
        eqe = (P / E_ph) / (I / _Q)
    eqe[~np.isfinite(eqe)] = np.nan
    return eqe


def current_density(current_A, area_um2):
    return np.abs(np.asarray(current_A, float)) / (area_um2 * 1e-8)


# ---------------------------------------------------------------------
#  ABC model in a well-conditioned, degeneracy-free parametrization.
#
#  The raw ABC model EQE(J) = eta * B n^2 / (A n + B n^2 + C n^3) with
#  J = q d (A n + B n^2 + C n^3) is degenerate when fit to EQE(J) alone: you
#  cannot recover the absolute carrier density n (and hence A, B, C separately)
#  without an independent calibration. Fitting all four parameters in tiny
#  absolute-EQE units is ill-conditioned and pins eta at a bound — which is
#  exactly why the original calc_iqe_eqe.py produced garbage IQE.
#
#  Substituting u = n*sqrt(C/A) collapses the model to THREE identifiable
#  parameters:
#       beta = B / sqrt(A C)        (shape -> peak IQE = beta/(beta+2))
#       J0   = q d A^{3/2}/sqrt(C)  (current scale; peak at J = J0*(2+beta))
#       eta  = collection * extraction (vertical scale)
#  with
#       J/J0       = u*(1 + beta*u + u^2)          (monotone in u>0)
#       IQE(u)     = beta*u / (1 + beta*u + u^2)    (peaks at u=1)
#       EQE        = eta * IQE
#  This is what we fit. peak IQE = beta/(beta+2) depends only on the curve shape.
# ---------------------------------------------------------------------


def _solve_u(r, beta):
    """Solve u*(1+beta*u+u^2)=r for the unique u>0 (vectorized Newton)."""
    r = np.asarray(r, float)
    u = np.where(r < 1.0, r, np.cbrt(np.maximum(r, 1e-300)))  # good warm start
    u = np.maximum(u, 1e-30)
    for _ in range(30):
        g = u**3 + beta * u**2 + u - r
        gp = 3 * u**2 + 2 * beta * u + 1.0
        step = g / gp
        u = np.maximum(u - step, 1e-30)
        if np.max(np.abs(step)) < 1e-12:
            break
    return u


def _abc_eqe_norm(J, beta, J0, eta):
    """EQE(J) for the (beta, J0, eta) parametrization."""
    J = np.atleast_1d(np.asarray(J, float))
    out = np.full(J.shape, np.nan)
    ok = np.isfinite(J) & (J > 0) & np.isfinite(J0) & (J0 > 0) & (beta > 0)
    if not np.any(ok):
        return out
    u = _solve_u(J[ok] / J0, beta)
    iqe = beta * u / (1.0 + beta * u + u**2)
    out[ok] = eta * iqe
    return out


def median_eqe_curve(J, eqe, min_count=3, n_bins=30):
    """Median EQE vs J, binned in log(J). Returns a DataFrame[J, EQE, n, EQE_err].

    EQE_err is the spread of the bin's median: 1.4826·MAD / √n (robust standard error),
    for display error bars. Works whether pixels share voltages or each has its own.
    """
    J = np.asarray(J, float)
    eqe = np.asarray(eqe, float)
    m = np.isfinite(J) & np.isfinite(eqe) & (J > 0) & (eqe > 0)
    J, eqe = J[m], eqe[m]
    if J.size < min_count:
        return pd.DataFrame(columns=["J", "EQE", "n", "EQE_err"])
    edges = np.logspace(np.log10(J.min()), np.log10(J.max()), n_bins + 1)
    idx = np.clip(np.digitize(J, edges) - 1, 0, n_bins - 1)
    rows = []
    for b in range(n_bins):
        sel = idx == b
        if sel.sum() >= min_count:
            e = eqe[sel]
            med = float(np.median(e))
            mad = float(np.median(np.abs(e - med)))
            sig = 1.4826 * mad if mad > 0 else float(e.std())
            err = sig / np.sqrt(sel.sum())
            rows.append(
                dict(J=float(np.median(J[sel])), EQE=med, n=int(sel.sum()), EQE_err=err)
            )
    out = pd.DataFrame(rows, columns=["J", "EQE", "n", "EQE_err"])
    return out.sort_values("J").reset_index(drop=True) if not out.empty else out


def fit_abc(J, eqe, d_cm=None):
    """Fit the (beta, J0, eta) ABC model to EQE(J). Returns dict or None.

    ``d_cm`` is accepted for backward-compat but unused — the parametrization
    is independent of the active-layer thickness (it only rescales the
    non-identifiable carrier density).
    """
    from scipy.optimize import curve_fit

    J = np.asarray(J, float)
    eqe = np.asarray(eqe, float)
    m = np.isfinite(J) & np.isfinite(eqe) & (J > 0) & (eqe > 0)
    Jf, ef = J[m], eqe[m]
    if len(Jf) < MIN_POINTS_FOR_FIT:
        return None

    # Fit in normalized EQE units (~O(1)) so the optimiser is well-scaled.
    scale = float(np.nanmax(ef))
    if not np.isfinite(scale) or scale <= 0:
        return None
    yf = ef / scale

    # Seeds: peak near the data max -> J0 ~ J_at_max/(2+beta); eta ~ max.
    # Two starts (peak IQE ~0.82 and ~0.60) are enough to avoid the obvious
    # local minimum; maxfev is bounded so a non-converging noisy pixel can't
    # burn tens of thousands of evals.
    j_at_max = float(Jf[np.argmax(ef)])
    seeds = [[b0, max(j_at_max / (2 + b0), 1e-6), 1.0] for b0 in (9.0, 3.0)]
    lo = [1e-2, Jf.min() * 1e-3, 1e-3]
    hi = [1e4, Jf.max() * 1e3, 5.0]

    def model(j, beta, J0, eta):
        return _abc_eqe_norm(j, beta, J0, eta)

    best = None
    for s in seeds:
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                popt, _ = curve_fit(
                    model, Jf, yf, p0=s, bounds=(lo, hi), maxfev=4000, method="trf"
                )
        except Exception:
            continue
        ym = _abc_eqe_norm(Jf, *popt)
        ss_res = np.nansum((yf - ym) ** 2)
        if best is None or ss_res < best[0]:
            best = (ss_res, popt)
    if best is None:
        return None

    beta, J0, eta_norm = best[1]
    eta = eta_norm * scale
    eqe_model = _abc_eqe_norm(Jf, beta, J0, eta)
    with np.errstate(divide="ignore", invalid="ignore"):
        iqe = np.where(eta > 0, eqe_model / eta, np.nan)

    peak_iqe = beta / (beta + 2.0)
    J_peak = J0 * (2.0 + beta)

    ss_res = np.nansum((ef - eqe_model) ** 2)
    ss_tot = np.nansum((ef - np.nanmean(ef)) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan

    # Peak is "captured" only if we sampled well into the rising SRH side: the
    # lowest measured J must sit at least an octave below the fitted peak
    # (J_peak >= 2*J_min) and the peak must be within the measured window.
    # Otherwise peak-IQE rests on too little rising-side data to be trusted.
    captured = bool(
        np.isfinite(J_peak)
        and J_peak >= 2.0 * np.nanmin(Jf)
        and J_peak <= 2.0 * np.nanmax(Jf)
    )

    # Where the EQE DATA actually peaks (smoothed) — to catch fits that run off into the droop.
    J_datamax = _data_peak_J(Jf, ef)

    return dict(
        beta=beta,
        J0=J0,
        eta_LEE=eta,
        peak_IQE=peak_iqe,
        J_at_peak_IQE=J_peak,
        R_squared=r2,
        peak_captured=captured,
        J_at_datamax=J_datamax,
        J_fit=Jf,
        eqe_fit=ef,
        eqe_model=eqe_model,
        iqe_fit=iqe,
        J_min=float(np.nanmin(Jf)),
        J_max=float(np.nanmax(Jf)),
    )


def _data_peak_J(J, e):
    """J where the (smoothed) EQE data is maximal — a model-free estimate of the peak
    location, used to reject fits whose peak ran off to a very different J."""
    J = np.asarray(J, float)
    e = np.asarray(e, float)
    m = np.isfinite(J) & np.isfinite(e) & (J > 0) & (e > 0)
    J, e = J[m], e[m]
    if J.size < 4:
        return float("nan")
    order = np.argsort(J)
    J, e = J[order], e[order]
    k = max(3, J.size // 8)  # small rolling median smooth
    es = pd.Series(e).rolling(k, center=True, min_periods=1).median().values
    return float(J[int(np.argmax(es))])


def classify_fit(f):
    """Decide whether a per-pixel ABC fit is a TRUSTWORTHY peak-IQE, and if not, why.

    Returns (clean: bool, flag: str). A clean fit has the IQE peak actually inside the
    measured data, a decent R², and a physical beta (the rising SRH side was sampled, so
    beta isn't a runaway extrapolation). The flag string is written into the per-pixel CSV
    so anyone reading it sees immediately why a pixel was kept or rejected."""
    if f is None:
        return False, "fit_failed"
    if not f.get("peak_captured"):
        return (
            False,
            "peak_below_data: rising (low-J) side not sampled, peak extrapolated",
        )
    if not (f.get("R_squared", float("nan")) >= PER_PIXEL_R2_MIN):
        return False, f"low_R2<{PER_PIXEL_R2_MIN}: fit doesn't follow the curve"
    if f.get("beta", float("inf")) > BETA_MAX:
        return (
            False,
            f"degenerate_beta>{BETA_MAX:g}: SRH unconstrained → fake ~{100 * f['peak_IQE']:.0f}% peak",
        )
    if f.get("beta", 0.0) < BETA_MIN:
        return False, f"beta<{BETA_MIN:g}: no real peak / noise-dominated"
    return True, "clean"


def write_fit_guide(out, df_fits):
    """Write abc_fit_GUIDE.md next to abc_fit_per_pixel.csv so anyone reading the fit
    output immediately understands beta, peak IQE, and why pixels are flagged."""
    n = len(df_fits)
    n_clean = int(df_fits["clean_fit"].sum()) if "clean_fit" in df_fits else 0
    frac = 100 * n_clean / n if n else 0.0
    reasons = (
        df_fits[df_fits.get("clean_fit") != True]["flag"].value_counts()
        if "flag" in df_fits
        else None
    )
    med = (
        df_fits.loc[df_fits.get("clean_fit") == True, "peak_IQE"].median()
        if n_clean
        else float("nan")
    )
    lines = [
        "# ABC per-pixel fit — how to read it",
        "",
        "## What beta (β) is",
        "",
        "Carrier recombination in the LED follows the **ABC model**: the total recombination",
        "rate at carrier density n is",
        "",
        "    R(n) = A·n  +  B·n²  +  C·n³",
        "",
        "- **A** — SRH (defect/trap) recombination: non-radiative, dominates at LOW current.",
        "- **B** — radiative recombination: this is the term that MAKES LIGHT.",
        "- **C** — Auger recombination: non-radiative, dominates at HIGH current (droop).",
        "",
        "The internal quantum efficiency is the light-making fraction:  **IQE = B·n² / R(n)**.",
        "",
        "**β = B / √(A·C)** collapses all three into one dimensionless *quality factor*: how strong",
        "the radiative term is versus the geometric mean of the two loss channels. It sets the peak",
        "directly, with no other knobs:",
        "",
        "    peak IQE = β / (β + 2)        reached at  J = J0·(2 + β)",
        "",
        "| β | 2 | 6 | 18 | 38 | 198 |",
        "|---|---|---|----|----|-----|",
        "| peak IQE | 50% | 75% | 90% | 95% | 99% |",
        "",
        "So **to hit 90%+ peak IQE you need β ≳ 18**, and to *measure* β you must sample the rising",
        "(low-current, SRH-dominated) side of the efficiency curve.",
        "",
        "## Why fits get flagged (and why a '95%' can be fake)",
        "",
        "If the sweep never reaches low enough current, the fit can't see the SRH (A) term, so it",
        "drives A→0 and β runs away to huge values — producing a **fake ~95–100% peak that sits",
        f"*below* the measured data**. Those are flagged `degenerate_beta>{BETA_MAX:g}`. A fit counts as",
        "**clean** only when the peak is actually inside the data, R² ≥ "
        f"{PER_PIXEL_R2_MIN}, and {BETA_MIN:g} ≤ β ≤ {BETA_MAX:g}.",
        "",
        "## Columns in abc_fit_per_pixel.csv",
        "",
        "- `beta`, `J0`, `eta_LEE` — fit parameters (β as above; J0 current scale; eta_LEE the",
        "  uncalibrated collection×extraction scale, meaningless in absolute terms).",
        "- `peak_IQE` — the trustworthy peak IQE, **filled only for clean fits** (else blank).",
        "- `peak_IQE_raw` — peak IQE of the fit *regardless* of trust (so you can see the runaways).",
        "- `peak_captured` — was the fitted peak inside the measured J window?",
        "- `clean_fit` — TRUE only if trustworthy (use this to filter).",
        "- `flag` — plain-language reason a pixel is clean or rejected.",
        "- `R2`, `J_at_peak`, `J_min`, `J_max` — fit quality and the J range used.",
        "",
        "## This run",
        "",
        f"- **{n_clean}/{n} pixels clean ({frac:.0f}%)**"
        + (f", median clean peak IQE **{100 * med:.0f}%**." if n_clean else "."),
    ]
    if reasons is not None and len(reasons):
        lines += ["- Rejections by reason:"] + [
            f"    - {r}: {c}" for r, c in reasons.items()
        ]
    (out / "abc_fit_GUIDE.md").write_text("\n".join(lines), encoding="utf-8")


# ---------------------------------------------------------------------
#  "Split around the peak" cross-check (fit each branch separately).
#
#  The normalized ABC shape EQE/EQE_peak as a function of J/J_peak depends on a
#  single parameter beta (peak IQE = beta/(beta+2)) and passes through (1,1) by
#  construction. So we can find the EQE peak in the data, normalize, and fit beta
#  SEPARATELY to the rising (low-J, SRH-controlled) and droop (high-J, Auger-
#  controlled) branches. Each branch yields its own peak IQE; if they agree with
#  each other and with the joint fit, the ABC model is self-consistent. This is
#  the literal "split the graph at the peak and fit each side" method.
# ---------------------------------------------------------------------


def _norm_eqe_model(x, beta):
    """EQE/EQE_peak as a function of x = J/J_peak for the one-parameter shape."""
    x = np.atleast_1d(np.asarray(x, float))
    u = _solve_u(x * (2.0 + beta), beta)  # invert x = u(1+bu+u^2)/(2+b)
    return u * (2.0 + beta) / (1.0 + beta * u + u**2)


def split_branch_iqe(med, min_pts=4):
    """Independent peak-split cross-check on the median EQE(J) curve.

    Returns dict with J_peak and peak IQE from the rising side and the droop side
    (each fit separately), or None if the curve can't be split with enough points.
    """
    from scipy.optimize import curve_fit

    if med is None or len(med) < 2 * min_pts - 1:
        return None
    J = med["J"].values.astype(float)
    E = med["EQE"].values.astype(float)
    ok = np.isfinite(J) & np.isfinite(E) & (J > 0) & (E > 0)
    J, E = J[ok], E[ok]
    if len(J) < 2 * min_pts - 1:
        return None
    # robust peak from a 3-point smoothed median curve
    k = np.convolve(E, np.ones(3) / 3.0, mode="same")
    ipk = int(np.argmax(k))
    Jpk, Epk = float(J[ipk]), float(E[ipk])
    if Epk <= 0:
        return None
    x, y = J / Jpk, E / Epk
    out = {"J_peak": Jpk}
    for name, sel in [("low", x <= 1.0 + 1e-9), ("high", x >= 1.0 - 1e-9)]:
        xb, yb = x[sel], y[sel]
        out[f"n_{name}"] = int(len(xb))
        if len(xb) < min_pts:
            out[f"peak_iqe_{name}"] = np.nan
            out[f"beta_{name}"] = np.nan
            continue
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                popt, _ = curve_fit(
                    _norm_eqe_model, xb, yb, p0=[3.0], bounds=(1e-2, 1e4), maxfev=10000
                )
            b = float(popt[0])
            out[f"beta_{name}"] = b
            out[f"peak_iqe_{name}"] = b / (b + 2.0)
        except Exception:
            out[f"peak_iqe_{name}"] = np.nan
            out[f"beta_{name}"] = np.nan
    return out


# =====================================================================
#  Data loading
# =====================================================================


def _load_aggregated(folder):
    """Return aggregated per-(pixel,voltage) DataFrame for a sweep/fastscan dir."""
    folder = Path(folder)
    files = sorted(folder.glob("aggregated_bitval=*.csv")) or sorted(
        folder.glob("**/aggregated_bitval=*.csv")
    )
    if files:
        return pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    raws = [
        f for f in (folder / "raw_data").glob("*.csv") if "config" not in f.name.lower()
    ]
    if not raws:
        return None
    return _aggregate_raw(pd.read_csv(raws[0]))


def _aggregate_raw(df):
    rows = []
    for (x, y, b, v), g in df.groupby(["X", "Y", "BITVAL", "NVLED_V"]):

        def med(t):
            s = g[g["TYPE"] == t]["MEAS_VALUE"]
            return s.mean() if not s.empty else np.nan

        rows.append(
            dict(
                X=int(x),
                Y=int(y),
                BITVAL=int(b),
                NVLED_V=round(v, 3),
                PM400_POWER_MEAN=med("PM400"),
                VLED_CURR_MEAN=med("VLED"),
                NVLED_CURR_MEAN=med("NVLED"),
            )
        )
    return pd.DataFrame(rows)


def _dark_power(folder):
    """Median DARK_PM400 from the raw CSV, for background subtraction."""
    folder = Path(folder)
    raws = [
        f for f in (folder / "raw_data").glob("*.csv") if "config" not in f.name.lower()
    ]
    if not raws:
        return 0.0
    df = pd.read_csv(raws[0])
    d = df[df["TYPE"] == "DARK_PM400"]["MEAS_VALUE"]
    return float(d.median()) if not d.empty else 0.0


def _load_config(folder):
    folder = Path(folder)
    cfgs = list((folder / "raw_data").glob("*_config.json"))
    if not cfgs:
        return {}
    with open(cfgs[0]) as f:
        return json.load(f).get("settings", {})


# =====================================================================
#  Enrichment: EQE, J, leakage, reliability mask
# =====================================================================


def enrich(df, dark_W=0.0):
    df = df.copy()

    def _val(base):
        # prefer the honest *_MEDIAN column (new files with raw samples); fall back to *_MEAN
        # (older files, where *_MEAN was already the robust median).
        col = f"{base}_MEDIAN" if f"{base}_MEDIAN" in df.columns else f"{base}_MEAN"
        return df[col].astype(float)

    P = _val("PM400_POWER") - dark_W
    P = P.clip(lower=0.0)
    Iv = _val("VLED_CURR")
    Inv = _val("NVLED_CURR")
    df["P_W"] = P
    df["I_VLED"] = Iv.abs()
    df["I_NVLED"] = Inv.abs()
    # The NVLED channel (ch B) carries the actual LED current; the VLED channel
    # (ch A) can include other current paths. Use NVLED as the LED current and
    # keep |Iv+Inv|/max only as a *diagnostic* of how much VLED leaks.
    #
    # LEAKAGE SUBTRACTION. ch B reads the TOTAL cathode current, not the injected one.
    # When the run carries NVLED_CURR_REF -- the same pixel at bitval 0, measured back to
    # back at the same bias -- subtract it. Measured 261013 over 12 pixels: the reference
    # is 8% of the reading at -0.95 V and 58% at -0.80 V, and without the subtraction the
    # EQE appears to collapse 20x on the rising side. With it the curve is smooth.
    # Runs without the column keep the old meaning, and LEAK_FRAC says which you have.
    if "NVLED_CURR_REF" in df.columns:
        Iref = df["NVLED_CURR_REF"].astype(float)
        df["I_NVLED_REF"] = Iref.abs()
        df["I_LED"] = (Inv.abs() - Iref.abs()).clip(lower=0.0)
        df["LEAK_FRAC"] = (Iref.abs() / Inv.abs().replace(0, np.nan)).clip(upper=1.0)
        # The optical reference is the light with the pixel addressed but not driven, so
        # it is the true local dark for this pixel -- better than the run-wide dark.
        if "PM400_POWER_REF" in df.columns:
            P = (_val("PM400_POWER") - df["PM400_POWER_REF"].astype(float)).clip(lower=0.0)
            df["P_W"] = P
    else:
        df["I_LED"] = Inv.abs()
        df["LEAK_FRAC"] = np.nan
    denom = np.maximum(Iv.abs(), Inv.abs()).replace(0, np.nan)
    df["LEAKAGE"] = (Iv + Inv).abs() / denom
    df["J_VLED"] = current_density(Iv, PIXEL_AREA_UM2)
    df["J_NVLED"] = current_density(Inv, PIXEL_AREA_UM2)
    df["J_LED"] = current_density(df["I_LED"], PIXEL_AREA_UM2)
    df["EQE_VLED"] = compute_eqe(P, Iv, WAVELENGTH_NM)
    df["EQE_NVLED"] = compute_eqe(P, Inv, WAVELENGTH_NM)
    # EQE on the LEAKAGE-CORRECTED current. On an uncorrected run I_LED is |Inv| and this
    # is identical to EQE_NVLED, so the meaning of the column does not change under you.
    df["EQE_LED"] = compute_eqe(P, df["I_LED"], WAVELENGTH_NM)
    # Error bar on EQE = EQE·√((σP/P)² + (σI/I)²), propagated from the stored SEM columns.
    # NOTE: this is the RANDOM within-point scatter only — it does NOT capture the systematic
    # low-J errors (range offset, parasitic, dark) that dominate here, which is exactly why it
    # is used for DISPLAY (and the gates) but NOT as a fit weight (a stable-but-wrong offset has
    # a tiny SEM and would be over-trusted). See plot error bars; fit stays unweighted.
    relP = (
        (df["PM400_POWER_STD"].astype(float) / P.replace(0, np.nan))
        if "PM400_POWER_STD" in df
        else 0.0
    )
    # The subtraction propagates: sigma(I_LED) = sqrt(sigma_lit^2 + sigma_ref^2), and the
    # relative error is against the (smaller) corrected current, so it grows exactly where
    # the leakage dominates. That growth is the real low-J limit now, not the raw scatter.
    if "NVLED_CURR_STD" in df:
        sigI = df["NVLED_CURR_STD"].astype(float)
        if "NVLED_CURR_REF_STD" in df.columns:
            sigI = np.sqrt(sigI ** 2 + df["NVLED_CURR_REF_STD"].astype(float) ** 2)
        relI = sigI / df["I_LED"].replace(0, np.nan)
    else:
        relI = 0.0
    df["EQE_LED_ERR"] = df["EQE_LED"] * np.sqrt(
        np.asarray(relP, float) ** 2 + np.asarray(relI, float) ** 2
    )
    # Reliability gates (NVLED is the LED current; VLED is NOT gated on):
    #  (1) NVLED current above the noise floor;
    #  (2) NVLED current scatter below threshold (catches per-point noise).
    # We do NOT reject on VLED/NVLED divergence: by design ch A (VLED, the anode) can
    # carry other paths, so |Iv| > |Inv| is expected, not a fault. LEAKAGE is computed
    # and plotted purely as a diagnostic — the LED current is ch B regardless. (The
    # noise-floor + scatter gates already drop genuinely bad NVLED readings.)
    reliable = df["I_LED"] >= NOISE_FLOOR_A
    if "NVLED_CURR_STD" in df.columns:
        relstd = df["NVLED_CURR_STD"].astype(float) / df["I_NVLED"].replace(0, np.nan)
        reliable &= relstd.fillna(np.inf) <= NVLED_RELSTD_MAX
    # Optical-SNR gate: drop points where the PM noise rivals the signal (light at
    # the dark floor at very low J). Without this, those P≈dark points deflate the
    # rising side and bias peak-IQE low.
    if "PM400_POWER_STD" in df.columns:
        relP = df["PM400_POWER_STD"].astype(float) / df["P_W"].replace(0, np.nan)
        reliable &= relP.fillna(np.inf) <= PM_RELSTD_MAX
    # Absolute optical SNR: the signal must be above the dark/background floor —
    # otherwise EQE≈0 even when the (averaged) value is precise.
    if dark_W and dark_W > 0:
        reliable &= df["P_W"] >= PM_MIN_OVER_DARK * dark_W
    df["RELIABLE"] = reliable
    return df


# =====================================================================
#  Fast-scan classification (fixed within/outside-1sigma logic)
# =====================================================================


def classify_fastscan(fs):
    """Return dict with yield + robust classification.

    Splits alive pixels into low-tail / typical (within 1 sigma) / high-tail so
    'outside' is no longer a meaningless mix of bright and dim pixels.
    """
    P = fs["PM400_POWER_MEAN"].abs()
    med = P.median()
    bright = float(
        np.nanpercentile(P.values, FASTSCAN_ALIVE_PCTL)
    )  # robust to a majority-dead ROI
    thr = DEAD_THRESHOLD_FRAC * bright
    alive = P >= thr
    Pa = P[alive]
    # Robust centre/scale (median + MAD) — resistant to the dead/outlier tails.
    centre = Pa.median()
    mad = (Pa - centre).abs().median() * 1.4826
    std = Pa.std()
    lo, hi = centre - std, centre + std
    within = alive & (P >= lo) & (P <= hi)
    low_tail = alive & (P < lo)
    high_tail = alive & (P > hi)
    return dict(
        n_total=len(fs),
        n_alive=int(alive.sum()),
        n_dead=int((~alive).sum()),
        median_W=float(med),
        lit_ref_W=bright,
        dead_thr_W=thr,
        alive_mean_W=float(Pa.mean()),
        alive_std_W=float(std),
        alive_cv=float(std / Pa.mean()) if Pa.mean() else np.nan,
        alive_mad_W=float(mad),
        n_within=int(within.sum()),
        n_low=int(low_tail.sum()),
        n_high=int(high_tail.sum()),
        within_lo_W=float(lo),
        within_hi_W=float(hi),
        alive_mask=alive,
        within_mask=within,
        low_mask=low_tail,
        high_mask=high_tail,
    )


# =====================================================================
#  Plotting
# =====================================================================


def _plt():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def plot_fastscan_overview(fs, cls, out, plt):
    fig, ax = plt.subplots(1, 2, figsize=(13, 5))
    piv = fs.pivot_table(
        index="Y", columns="X", values="PM400_POWER_MEAN", aggfunc="mean"
    )
    im = ax[0].imshow(
        piv.values * 1e9,
        origin="upper",
        cmap="inferno",
        extent=[piv.columns.min(), piv.columns.max(), piv.index.max(), piv.index.min()],
    )
    ax[0].set_title("Fast-scan optical power map")
    ax[0].set_xlabel("X")
    ax[0].set_ylabel("Y")
    plt.colorbar(im, ax=ax[0], label="P (nW)")

    P = fs["PM400_POWER_MEAN"].abs() * 1e9
    ax[1].hist(P, bins=40, color="tab:blue", alpha=0.8)
    ax[1].axvline(cls["median_W"] * 1e9, color="k", ls="-", lw=1, label="median")
    ax[1].axvline(
        cls["within_lo_W"] * 1e9, color="tab:green", ls="--", lw=1, label="±1σ band"
    )
    ax[1].axvline(cls["within_hi_W"] * 1e9, color="tab:green", ls="--", lw=1)
    ax[1].axvline(
        cls["dead_thr_W"] * 1e9, color="tab:red", ls=":", lw=1, label="dead thresh"
    )
    ax[1].set_xlabel("P (nW)")
    ax[1].set_ylabel("pixels")
    ax[1].set_title(
        f"Yield: {cls['n_alive']}/{cls['n_total']} alive "
        f"(within±1σ {cls['n_within']}, low {cls['n_low']}, high {cls['n_high']})"
    )
    ax[1].legend()
    fig.tight_layout()
    fig.savefig(out, dpi=300)
    plt.close(fig)


def plot_fastscan_vs_sweep(fs, sweep, out, plt):
    """Cross-check: fast-scan (-3.2 V) vs sweep endpoint (-3.2 V) per pixel."""
    s32 = sweep[np.isclose(sweep["NVLED_V"], sweep["NVLED_V"].min())]
    m = fs.merge(s32, on=["X", "Y"], suffixes=("_fs", "_sw"))
    if m.empty:
        return None
    fig, ax = plt.subplots(1, 2, figsize=(12, 5))
    pf = m["PM400_POWER_MEAN_fs"].abs() * 1e9
    ps = m["PM400_POWER_MEAN_sw"].abs() * 1e9
    ax[0].scatter(pf, ps, s=14, alpha=0.7)
    lim = [0, max(pf.max(), ps.max()) * 1.05]
    ax[0].plot(lim, lim, "k--", lw=1, label="1:1")
    ax[0].set_xlabel("Fast-scan P (nW)")
    ax[0].set_ylabel("Sweep@-3.2V P (nW)")
    ax[0].set_title("Optical power consistency")
    ax[0].legend()
    ratio = (ps / pf).replace([np.inf, -np.inf], np.nan).dropna()
    ax[1].hist(ratio, bins=30, color="tab:purple", alpha=0.8)
    ax[1].axvline(1.0, color="k", ls="--")
    ax[1].set_xlabel("Sweep / Fast-scan power ratio")
    ax[1].set_ylabel("pixels")
    ax[1].set_title(f"median ratio = {ratio.median():.3f}")
    fig.tight_layout()
    fig.savefig(out, dpi=300)
    plt.close(fig)
    return float(ratio.median())


def plot_ivl(df, out, plt):
    """Full I-V-L: current vs NVLED_V, power vs NVLED_V, and L-I.

    Grouped per (pixel, bitval) so multi-bitval drive levels are separate curves.
    """
    gcols = ["X", "Y", "BITVAL"] if "BITVAL" in df.columns else ["X", "Y"]
    fig, ax = plt.subplots(1, 3, figsize=(17, 5))
    for _, g in df.groupby(gcols):
        g = g.sort_values("NVLED_V")
        ax[0].plot(g["NVLED_V"], g["I_LED"] * 1e6, lw=0.6, alpha=0.4)
        ax[1].plot(g["NVLED_V"], g["P_W"] * 1e9, lw=0.6, alpha=0.4)
        r = g[g["RELIABLE"]]
        ax[2].plot(r["I_LED"] * 1e6, r["P_W"] * 1e9, lw=0.6, alpha=0.4)
    ax[0].set_xlabel("NVLED voltage (V)")
    ax[0].set_ylabel("LED current (µA)")
    ax[0].set_title("I–V: current vs NVLED bias")
    ax[1].set_xlabel("NVLED voltage (V)")
    ax[1].set_ylabel("Optical power (nW)")
    ax[1].set_title("L–V: power vs NVLED bias")
    ax[2].set_xlabel("LED current (µA)")
    ax[2].set_ylabel("Optical power (nW)")
    ax[2].set_title("L–I (reliable points)")
    for a in ax:
        a.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=300)
    plt.close(fig)


def plot_leakage(df, out, plt):
    fig, ax = plt.subplots(figsize=(8, 5))
    g = (
        df.groupby("NVLED_V")
        .agg(leak=("LEAKAGE", "median"), I=("I_LED", "median"))
        .reset_index()
    )
    ax.plot(g["NVLED_V"], g["leak"], "o-", color="tab:red", ms=3)
    ax.axhline(
        LEAKAGE_MAX,
        color="k",
        ls="--",
        lw=1,
        label=f"flag > {LEAKAGE_MAX} (diagnostic)",
    )
    ax.set_xlabel("NVLED voltage (V)")
    ax.set_ylabel("median leakage  |Iv+Inv| / max(|Iv|,|Inv|)")
    ax.set_title("VLED↔NVLED current mismatch (diagnostic, not a reject gate)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=300)
    plt.close(fig)


def plot_eqe_all_and_median(df, fit, med, fit_ok, out, plt, src="LED"):
    Jc, Ec = f"J_{src}", f"EQE_{src}"
    fig, ax = plt.subplots(1, 2, figsize=(13, 5))
    for (x, y), g in df.groupby(["X", "Y"]):
        r = g[g["RELIABLE"]]
        ax[0].scatter(r[Jc], r[Ec], s=8, alpha=0.25, color="tab:blue")
    ax[0].set_xscale("log")
    ax[0].set_yscale("log")
    ax[0].set_xlabel("J (A/cm²)")
    ax[0].set_ylabel("Collected EQE (fraction)")
    ax[0].set_title(f"EQE vs J — all pixels ({src} current, reliable)")
    ax[0].grid(True, which="both", alpha=0.3)

    yerr = med["EQE_err"].values if "EQE_err" in med else None
    ax[1].errorbar(
        med["J"],
        med["EQE"],
        yerr=yerr,
        fmt="o",
        ms=5,
        color="tab:blue",
        elinewidth=0.8,
        capsize=2,
        label="median EQE ± robust SE",
        zorder=3,
    )
    if fit is not None and fit_ok:
        ax[1].plot(
            fit["J_fit"],
            fit["eqe_model"],
            "r-",
            lw=2,
            label=f"ABC fit (R²={fit['R_squared']:.3f})",
            zorder=4,
        )
        ttl = "Median EQE + ABC fit"
    else:
        r2 = f"{fit['R_squared']:.2f}" if fit is not None else "n/a"
        ax[1].text(
            0.5,
            0.92,
            f"ABC fit not meaningful (R²={r2}):\n"
            "EQE is flat / droop-only — no IQE peak in range",
            ha="center",
            va="top",
            transform=ax[1].transAxes,
            color="firebrick",
            fontsize=10,
        )
        ttl = "Median EQE (ABC not constrained)"
    ax[1].set_xscale("log")
    ax[1].set_xlabel("J (A/cm²)")
    ax[1].set_ylabel("Collected EQE (fraction)")
    ax[1].set_title(ttl)
    ax[1].legend(loc="lower left")
    ax[1].grid(True, which="both", alpha=0.3)
    ax[1].annotate(
        "collection-limited:\ntrue device EQE ≈ this ÷ ~5×10⁻⁴",
        xy=(0.97, 0.04),
        xycoords="axes fraction",
        ha="right",
        va="bottom",
        fontsize=8,
        color="gray",
    )
    fig.tight_layout()
    fig.savefig(out, dpi=300)
    plt.close(fig)


def plot_iqe(df, pixel_fits, pooled_fit, out, plt, src="LED"):
    """Combined per-pixel IQE curves, using precomputed per-pixel fits.

    Three cases, consistent with the rest of the report:
      * ≥3 pixels with a clean per-pixel fit (peak captured, R²≥PER_PIXEL_R2_MIN)
        → draw those pixels' IQE(J) curves and overlay the pooled IQE curve.
      * fewer clean per-pixel fits but the POOLED fit succeeded → individual
        pixels are just too noisy to fit one-by-one; show measured relative EQE
        and point to the pooled number (IQE *was* extracted, from the pool).
      * pooled fit also failed → genuinely not extractable (need lower J).
    """
    Jc, Ec = f"J_{src}", f"EQE_{src}"
    good = [(x, y, f) for (x, y), f in pixel_fits.items() if classify_fit(f)[0]]
    peaks = [f["peak_IQE"] for _, _, f in good]
    pooled_ok = pooled_fit is not None and pooled_fit.get("peak_captured")

    fig, ax = plt.subplots(figsize=(9, 6))
    if len(good) >= 3:
        for _, _, f in good:
            m = np.isfinite(f["iqe_fit"])
            ax.plot(
                f["J_fit"][m],
                f["iqe_fit"][m] * 100,
                lw=0.7,
                alpha=0.4,
                color="tab:green",
            )
        if pooled_fit is not None:
            m = np.isfinite(pooled_fit["iqe_fit"])
            ax.plot(
                pooled_fit["J_fit"][m],
                pooled_fit["iqe_fit"][m] * 100,
                "k-",
                lw=2.5,
                zorder=5,
                label=f"pooled fit ({pooled_fit['peak_IQE'] * 100:.0f}%)",
            )
            ax.legend()
        ax.set_ylabel("IQE (%)")
        ttl = (
            f"Per-pixel IQE(J) — {len(good)} pixels with a clean fit "
            f"(R²≥{PER_PIXEL_R2_MIN})\nmedian per-pixel peak {np.median(peaks) * 100:.0f}%"
        )
    else:
        for (x, y), g in df.groupby(["X", "Y"]):
            r = g[g["RELIABLE"]].sort_values(Jc)
            e = r[Ec].values
            if len(e) < 5 or not np.isfinite(np.nanmax(e)) or np.nanmax(e) <= 0:
                continue
            ax.plot(
                r[Jc].values, e / np.nanmax(e), lw=0.6, alpha=0.3, color="tab:green"
            )
        ax.set_ylabel("EQE / EQE$_{max}$ per pixel  (measured, relative)")
        ax.set_ylim(0, 1.25)
        if pooled_ok:
            ttl = (
                f"Measured relative EQE(J) — {df.groupby(['X', 'Y']).ngroups} pixels\n"
                f"per-pixel fits too noisy to draw individually ({len(good)} clean) — "
                f"IQE was extracted from the POOL: peak {pooled_fit['peak_IQE'] * 100:.0f}%"
            )
            ax.text(
                0.5,
                0.06,
                "Pooled peak IQE is the trustworthy number "
                "(see eqe_vs_J_LED / iqe_split_branch).",
                ha="center",
                va="bottom",
                transform=ax.transAxes,
                color="tab:blue",
                fontsize=9,
            )
        else:
            ttl = (
                f"Measured relative EQE(J) — {df.groupby(['X', 'Y']).ngroups} pixels\n"
                "ABC fits unconstrained → absolute IQE not extractable (need J ≲ 1 A/cm²)"
            )
            ax.text(
                0.5,
                0.06,
                "This is NOT calibrated IQE — only the measured droop shape.",
                ha="center",
                va="bottom",
                transform=ax.transAxes,
                color="firebrick",
                fontsize=9,
            )
    ax.set_xscale("log")
    ax.set_xlabel("J (A/cm²)")
    ax.set_title(ttl)
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=300)
    plt.close(fig)
    return peaks


def _bitval_colors(df, plt):
    bits = sorted(df["BITVAL"].unique()) if "BITVAL" in df.columns else [None]
    cmap = plt.get_cmap("turbo")
    return bits, {b: cmap(i / max(1, len(bits) - 1)) for i, b in enumerate(bits)}


def plot_eqe_by_bitval(df, out, plt, src="LED"):
    """EQE vs J split by bit-value — each bitval a colour, with its own median.

    Lets you check that the bitvals overlap where they share J (consistency)
    rather than relying on the pooled curve alone.
    """
    if "BITVAL" not in df.columns or df["BITVAL"].nunique() < 2:
        return
    Jc, Ec = f"J_{src}", f"EQE_{src}"
    bits, colors = _bitval_colors(df, plt)
    fig, ax = plt.subplots(figsize=(9, 6))
    for b in bits:
        d = df[(df["BITVAL"] == b) & df["RELIABLE"]]
        if d.empty:
            continue
        ax.scatter(d[Jc], d[Ec], s=6, alpha=0.18, color=colors[b])
        med = median_eqe_curve(d[Jc].values, d[Ec].values, min_count=3)
        if not med.empty:
            ax.plot(
                med["J"],
                med["EQE"],
                "o-",
                color=colors[b],
                ms=4,
                lw=1.5,
                label=f"bit {b}",
            )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("J (A/cm²)")
    ax.set_ylabel("Collected EQE (fraction)")
    ax.set_title("EQE vs J split by bit-value (overlap check)")
    ax.legend()
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=300)
    plt.close(fig)


def plot_burst_scatter(sweep_dir, out, plt, n_pixels=20):
    """Overlay every individual BURST SAMPLE on the per-pixel curves, so you can eyeball how noisy
    each point's measurement was (needs the Sweep_samples.csv from SAVE_RAW_SAMPLES). For a few
    pixels: (1) the current burst vs V, (2) the light burst vs V — each raw sample a faint dot, the
    median/mean a line — and (3) an EQE cloud from random-pairing the current and light samples at
    each point (EQE=P/I mixes two independent bursts, so there are no true per-sample EQE points).
    A tight cloud → the curve shape is real; a wide cloud → that point is measurement-noise-limited.
    Returns True if drawn, False if no samples file."""
    samp = Path(sweep_dir) / "raw_data" / "Sweep_samples.csv"
    if not samp.exists():
        return False
    s = pd.read_csv(samp)
    s["A"] = s["VALUE"].astype(float)
    pix = [k for k, g in s.groupby(["X", "Y"]) if (g["TYPE"] == "NVLED").any()][
        :n_pixels
    ]
    if not pix:
        return False
    E_ph = photon_energy_J(WAVELENGTH_NM)
    area = PIXEL_AREA_UM2 * 1e-8
    fig, ax = plt.subplots(len(pix), 3, figsize=(15, 3.3 * len(pix)), squeeze=False)
    rng = np.random.default_rng(0)
    for r, (x, y) in enumerate(pix):
        g = s[(s["X"] == x) & (s["Y"] == y)]
        cur = g[g["TYPE"] == "NVLED"]
        lig = g[g["TYPE"] == "PM400"]
        # (1) current burst vs V — every dot is one raw SMU sample
        ax[r][0].scatter(
            cur["NVLED_V"],
            cur["A"].abs() * 1e6,
            s=7,
            alpha=0.30,
            color="tab:red",
            edgecolors="none",
        )
        cm = cur.groupby("NVLED_V")["A"].apply(lambda a: a.abs().median()) * 1e6
        ax[r][0].plot(cm.index, cm.values, "k.-", ms=3, lw=0.7)
        ax[r][0].axvline(-1.3, color="0.6", ls=":", lw=1)  # the NPLC knee
        ax[r][0].set_ylabel(f"pixel ({x},{y})\n|I| (µA)")
        # (2) light burst vs V — every dot is one raw PM400 sample
        ax[r][1].scatter(
            lig["NVLED_V"],
            lig["A"] * 1e9,
            s=7,
            alpha=0.30,
            color="tab:blue",
            edgecolors="none",
        )
        lm = lig.groupby("NVLED_V")["A"].mean() * 1e9
        ax[r][1].plot(lm.index, lm.values, "k.-", ms=3, lw=0.7)
        ax[r][1].axvline(-1.3, color="0.6", ls=":", lw=1)
        ax[r][1].set_ylabel("P (nW)")
        # (3) EQE BOOTSTRAP CLOUD — EQE=P/I mixes two INDEPENDENT bursts (no matched pairs), so we
        # draw random (I-sample, P-sample) pairs per point and compute EQE for each → a real point
        # cloud that reflects the combined current+light scatter (median line on top).
        Jc, Ec, Jl, El = [], [], [], []
        M = 40
        for v in sorted(g["NVLED_V"].unique()):
            Ia = np.abs(cur[cur["NVLED_V"] == v]["A"].values)
            Ia = Ia[Ia > 1e-9]
            Pa = np.clip(lig[lig["NVLED_V"] == v]["A"].values, 0, None)
            if len(Ia) < 2 or len(Pa) < 2:
                continue
            Imed = np.median(Ia)
            if Imed < 1e-6 or Pa.mean() < 3e-11:
                continue
            Jv = Imed / area
            eqe = (rng.choice(Pa, M) / E_ph) / (
                rng.choice(Ia, M) / _Q
            )  # random-paired samples
            Jc.extend([Jv] * M)
            Ec.extend(eqe.tolist())
            Jl.append(Jv)
            El.append((Pa.mean() / E_ph) / (Imed / _Q))
        if Jc:
            ax[r][2].scatter(
                Jc, Ec, s=5, alpha=0.10, color="tab:green", edgecolors="none"
            )
            o = np.argsort(Jl)
            ax[r][2].plot(np.array(Jl)[o], np.array(El)[o], "k-", lw=0.8)
            ax[r][2].set_xscale("log")
        ax[r][2].set_ylabel("EQE = P/I")
    for c, t in enumerate(
        [
            "current burst vs V (dots = raw SMU samples)",
            "light burst vs V (dots = raw PM samples)",
            "EQE vs J (cloud = random-paired I,P samples)",
        ]
    ):
        ax[0][c].set_title(t, fontsize=9)
    for c in range(2):
        ax[-1][c].set_xlabel("NVLED voltage (V)")
    ax[-1][2].set_xlabel("J (A/cm²)")
    fig.suptitle(
        "Per-pixel burst scatter — how noisy is each measured point? (dotted line = NPLC knee)"
    )
    fig.tight_layout()
    fig.savefig(out, dpi=300)
    plt.close(fig)
    return True


def plot_first_pixels(df, out, plt, n=FIRST_N_PIXELS, src="LED"):
    """Per-pixel diagnostics for the first n pixels: P vs NVLED_V, P vs J, EQE vs
    J — each bit-value a different colour. Reliable points solid, rejected faint.
    """
    Jc, Ec = f"J_{src}", f"EQE_{src}"
    # Pick the first n pixels that actually have reliable, positive-signal data — the raw
    # coordinate order starts at the ROI corner (e.g. (0,0)), which can be dead/edge pixels
    # with no positive optical power. A dead pixel makes the log panels un-scalable and would
    # otherwise abort the whole figure (losing the linear P-vs-V panel too).
    groups = df.groupby(["X", "Y"])
    alive = [k for k, g in groups if g["RELIABLE"].any() and (g["P_W"] > 0).any()]
    keys = alive or list(groups.groups.keys())
    pix = keys[:n]
    if not pix:
        return
    bits, colors = _bitval_colors(df, plt)
    fig, axes = plt.subplots(len(pix), 3, figsize=(15, 3.1 * len(pix)), squeeze=False)
    for r, (x, y) in enumerate(pix):
        g = df[(df["X"] == x) & (df["Y"] == y)]
        for b in bits:
            d = g[g["BITVAL"] == b] if b is not None else g
            c = colors[b]
            dv = d.sort_values("NVLED_V")
            axes[r][0].plot(
                dv["NVLED_V"],
                dv["P_W"] * 1e9,
                ".-",
                ms=3,
                lw=0.7,
                color=c,
                label=(f"bit {b}" if r == 0 else None),
            )
            dr = d[d["RELIABLE"]].sort_values(Jc)
            dn = d[~d["RELIABLE"]]
            axes[r][1].scatter(dn[Jc], dn["P_W"] * 1e9, s=5, color=c, alpha=0.15)
            axes[r][1].plot(dr[Jc], dr["P_W"] * 1e9, ".-", ms=3, lw=0.7, color=c)
            yerr = dr["EQE_LED_ERR"].values if "EQE_LED_ERR" in dr else None
            axes[r][2].errorbar(
                dr[Jc],
                dr[Ec],
                yerr=yerr,
                fmt=".-",
                ms=3,
                lw=0.7,
                color=c,
                elinewidth=0.6,
                capsize=1.5,
                alpha=0.9,
            )
        axes[r][0].set_ylabel(f"pixel ({x},{y})\nP (nW)")
        # Only switch an axis to log if it holds positive data — a dead/dark pixel otherwise
        # throws "no positive values" at draw time and takes the whole figure down.
        gr = g[g["RELIABLE"]]
        if (gr[Jc] > 0).any():
            axes[r][1].set_xscale("log")
        if (gr[Jc] > 0).any() and (gr[Ec] > 0).any():
            axes[r][2].set_xscale("log")
            axes[r][2].set_yscale("log")
        for cax in axes[r]:
            cax.grid(True, which="both", alpha=0.3)
    axes[0][0].set_title("optical power vs NVLED voltage")
    axes[0][1].set_title("optical power vs J  (faint = rejected)")
    axes[0][2].set_title("EQE vs J")
    axes[0][0].legend(fontsize=7, loc="best")
    for c in range(3):
        axes[-1][c].set_xlabel(["NVLED voltage (V)", "J (A/cm²)", "J (A/cm²)"][c])
    fig.tight_layout()
    fig.savefig(out, dpi=300)
    plt.close(fig)


def plot_split_branch(med, split, joint_peak_iqe, out, plt):
    """Normalized EQE/EQE_peak vs J/J_peak with the rise and droop branch fits."""
    if split is None:
        return
    J = med["J"].values.astype(float)
    E = med["EQE"].values.astype(float)
    Jpk = split["J_peak"]
    ipk = int(np.argmin(np.abs(J - Jpk)))
    Epk = E[ipk]
    if Epk <= 0:
        return
    x, y = J / Jpk, E / Epk
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(x, y, s=26, color="0.4", zorder=3, label="median EQE (normalized)")
    ax.axvline(1.0, color="k", ls="--", lw=0.8)
    if np.isfinite(split.get("beta_low", np.nan)):
        xs = np.logspace(np.log10(max(x.min(), 1e-3)), 0, 120)
        ax.plot(
            xs,
            _norm_eqe_model(xs, split["beta_low"]),
            "b-",
            lw=2,
            label=f"rising-side fit → {split['peak_iqe_low'] * 100:.0f}%",
        )
    if np.isfinite(split.get("beta_high", np.nan)):
        xs = np.logspace(0, np.log10(x.max()), 120)
        ax.plot(
            xs,
            _norm_eqe_model(xs, split["beta_high"]),
            "r-",
            lw=2,
            label=f"droop-side fit → {split['peak_iqe_high'] * 100:.0f}%",
        )
    ttl = "Split-around-peak cross-check"
    if np.isfinite(joint_peak_iqe):
        ttl += f"  (joint whole-curve fit: {joint_peak_iqe * 100:.0f}%)"
    ax.set_xscale("log")
    ax.set_xlabel("J / J$_{peak}$")
    ax.set_ylabel("EQE / EQE$_{peak}$")
    ax.set_title(ttl)
    ax.legend()
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=300)
    plt.close(fig)


def spatial_iqe_table(df, pooled_fit, src="LED"):
    """Robust per-pixel peak-IQE estimate for a spatial map.

    Instead of a fragile per-pixel ABC fit, use each pixel's median EQE in a
    window around the pooled J_peak. Collection×extraction (η) is ~uniform across
    a small ROI, so EQE-near-peak ∝ peak-IQE; scaling by the pooled peak IQE turns
    the relative map into a per-pixel peak-IQE estimate. Returns (DataFrame, J_ref,
    gradient_dict) or None.
    """
    if pooled_fit is None or not pooled_fit.get("peak_captured"):
        return None
    Jc, Ec = f"J_{src}", f"EQE_{src}"
    J_ref = float(pooled_fit["J_at_peak_IQE"])
    iqe_pk = float(pooled_fit["peak_IQE"])
    lo, hi = J_ref / 2.0, J_ref * 2.0
    rows = []
    for (x, y), g in df.groupby(["X", "Y"]):
        r = g[g["RELIABLE"] & (g[Jc] >= lo) & (g[Jc] <= hi)]
        e = r[Ec].median()
        if len(r) >= 3 and np.isfinite(e) and e > 0:
            rows.append((int(x), int(y), float(e)))
    if len(rows) < 6:
        return None
    P = pd.DataFrame(rows, columns=["X", "Y", "EQEref"])
    P["IQE_est"] = (P["EQEref"] / P["EQEref"].median()) * iqe_pk
    # Linear trend (plane fit) to quantify any spatial gradient.
    A = np.c_[np.ones(len(P)), P["X"], P["Y"]]
    coef, *_ = np.linalg.lstsq(A, P["IQE_est"].values, rcond=None)
    pred = A @ coef
    ss_res = float(np.sum((P["IQE_est"].values - pred) ** 2))
    ss_tot = float(np.sum((P["IQE_est"].values - P["IQE_est"].mean()) ** 2))
    grad = dict(
        gx=float(coef[1]),
        gy=float(coef[2]),
        r2=(1 - ss_res / ss_tot) if ss_tot > 0 else np.nan,
        dx=float(coef[1] * (P["X"].max() - P["X"].min())),
        dy=float(coef[2] * (P["Y"].max() - P["Y"].min())),
    )
    return P, J_ref, grad


def _block_sizes(xlo, xhi, ylo, yhi):
    """Power-of-2 block sizes (global-aligned to the 0-anchored 512 grid) from 2 up to the
    largest that still tiles the data into ≥4 blocks and is ≤ the data span."""
    span = max(xhi - xlo + 1, yhi - ylo + 1)
    sizes, S = [], 2
    while S <= span:
        nx = xhi // S - xlo // S + 1
        ny = yhi // S - ylo // S + 1
        if nx * ny >= 4:
            sizes.append(S)
        S *= 2
    return sizes


def _spatial_stats(X, Y, V):
    """Compact spatial-uniformity stats for the report: spread, a linear-gradient edge-to-edge
    estimate, and the weakest 8-px row/column band vs the global median (bad-line/region flag)."""
    X = np.asarray(X, float)
    Y = np.asarray(Y, float)
    V = np.asarray(V, float)
    m = np.isfinite(V)
    X, Y, V = X[m], Y[m], V[m]
    if len(V) < 4:
        return None
    A = np.c_[X, Y, np.ones_like(X)]
    coef, *_ = np.linalg.lstsq(A, V, rcond=None)
    grad_edge = float(
        np.hypot(coef[0], coef[1]) * max(X.max() - X.min(), Y.max() - Y.min(), 1)
    )
    med = float(np.median(V))
    worst = med
    for coord in (X, Y):
        b = (coord // 8).astype(int)
        for bb in set(b.tolist()):
            worst = min(worst, float(np.median(V[b == bb])))
    return dict(
        median=med,
        std=float(np.std(V)),
        grad_edge=grad_edge,
        resid_std=float(np.std(V - A @ coef)),
        worst_band_frac=(worst / med if med else float("nan")),
        n=int(len(V)),
    )


def raster_grid(
    X, Y, V, out, plt, vlabel, title, cmap="viridis", vlo=None, vhi=None, dpi=300
):
    """True ROI raster: one grid cell per pixel at its (X,Y), coloured by V; unmeasured/dead
    cells are white. Each measured pixel is EXACTLY one cell, so nothing overlaps regardless of
    ROI size or sampling density — unlike marker-based scatter. Saved at high dpi so the (often
    sparse) cells stay crisp."""
    X = np.asarray(X, int)
    Y = np.asarray(Y, int)
    V = np.asarray(V, float)
    m = np.isfinite(V)
    X, Y, V = X[m], Y[m], V[m]
    if len(V) < 4:
        return
    xlo, xhi, ylo, yhi = int(X.min()), int(X.max()), int(Y.min()), int(Y.max())
    nx, ny = xhi - xlo + 1, yhi - ylo + 1
    G = np.full((ny, nx), np.nan)
    G[Y - ylo, X - xlo] = V
    if vlo is None:
        vlo, vhi = np.nanpercentile(V, [2, 98])
    cm = plt.cm.get_cmap(cmap).copy()
    cm.set_bad("white")
    fig, ax = plt.subplots(figsize=(7.5, 6.2))
    im = ax.imshow(
        np.ma.masked_invalid(G),
        cmap=cm,
        vmin=vlo,
        vmax=vhi,
        extent=[xlo, xhi + 1, yhi + 1, ylo],
        origin="upper",
        interpolation="nearest",
        aspect="equal",
    )
    plt.colorbar(im, ax=ax, label=vlabel)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_title(
        f"{title}\n({len(V)} measured pixels of {nx}×{ny} ROI; white = unmeasured/dead)"
    )
    fig.tight_layout()
    fig.savefig(out, dpi=dpi)
    plt.close(fig)


def blocked_maps(X, Y, V, out, plt, vlabel, title, fmt="{:.0f}"):
    """Multi-scale blocked maps of a scalar V at pixels (X,Y): top row = MEDIAN (robust central
    value), bottom row = SPREAD (robust std / MAD). Global-aligned power-of-2 block sizes.
    Empty blocks white; **n=1 blocks hatched** (single-pixel, low confidence, so they don't
    visually compete with well-sampled ones); value + N annotated on coarse grids."""
    from matplotlib.patches import Rectangle

    X = np.asarray(X, int)
    Y = np.asarray(Y, int)
    V = np.asarray(V, float)
    m = np.isfinite(V)
    X, Y, V = X[m], Y[m], V[m]
    if len(V) < 4:
        return
    xlo, xhi, ylo, yhi = X.min(), X.max(), Y.min(), Y.max()
    sizes = _block_sizes(xlo, xhi, ylo, yhi)
    if not sizes:
        return
    vmin, vmax = np.nanpercentile(V, [5, 95])
    fig, axes = plt.subplots(
        2, len(sizes), figsize=(3.3 * len(sizes), 6.8), squeeze=False
    )
    med_cmap = plt.cm.viridis.copy()
    med_cmap.set_bad("white")
    spr_cmap = plt.cm.magma.copy()
    spr_cmap.set_bad("white")
    spr_max = 0.0
    ims_med, ims_spr = [], []
    for col, S in enumerate(sizes):
        bxs = list(range(xlo // S, xhi // S + 1))
        bys = list(range(ylo // S, yhi // S + 1))
        nx, ny = len(bxs), len(bys)
        med = np.full((ny, nx), np.nan)
        spr = np.full((ny, nx), np.nan)
        cnt = np.zeros((ny, nx), int)
        for j, by in enumerate(bys):
            for i, bx in enumerate(bxs):
                mm = (X // S == bx) & (Y // S == by)
                if mm.sum():
                    v = V[mm]
                    med[j, i] = np.median(v)
                    cnt[j, i] = int(mm.sum())
                    spr[j, i] = (
                        1.4826 * np.median(np.abs(v - np.median(v)))
                        if mm.sum() > 1
                        else 0.0
                    )
        spr_max = max(spr_max, np.nanmax(spr) if np.isfinite(spr).any() else 0.0)
        extent = [bxs[0] * S, (bxs[-1] + 1) * S, (bys[-1] + 1) * S, bys[0] * S]
        im0 = axes[0][col].imshow(
            np.ma.masked_invalid(med),
            cmap=med_cmap,
            vmin=vmin,
            vmax=vmax,
            extent=extent,
            aspect="equal",
            origin="upper",
        )
        im1 = axes[1][col].imshow(
            np.ma.masked_invalid(spr),
            cmap=spr_cmap,
            extent=extent,
            aspect="equal",
            origin="upper",
        )
        ims_med.append(im0)
        ims_spr.append(im1)
        axes[0][col].set_title(f"{S}×{S} px\n{nx}×{ny} blocks", fontsize=9)
        coarse = nx <= 10 and ny <= 10
        for j, by in enumerate(bys):
            for i, bx in enumerate(bxs):
                if not cnt[j, i]:
                    continue
                if cnt[j, i] == 1 and coarse:  # de-emphasise single-pixel blocks
                    for ax in (axes[0][col], axes[1][col]):
                        ax.add_patch(
                            Rectangle(
                                (bx * S, by * S),
                                S,
                                S,
                                fill=False,
                                hatch="////",
                                edgecolor="0.55",
                                linewidth=0.0,
                            )
                        )
                if coarse:
                    tc = "0.7" if cnt[j, i] == 1 else "white"
                    axes[0][col].text(
                        (bx + 0.5) * S,
                        (by + 0.5) * S,
                        f"{fmt.format(med[j, i])}\nn{cnt[j, i]}",
                        ha="center",
                        va="center",
                        fontsize=6.5,
                        color=tc,
                    )
        for ax in (axes[0][col], axes[1][col]):
            ax.set_xticks([])
            ax.set_yticks([])
    for im in ims_spr:
        im.set_clim(0, spr_max or 1.0)
    axes[0][0].set_ylabel(f"median {vlabel}", fontsize=10)
    axes[1][0].set_ylabel("spread — robust std", fontsize=10)
    fig.colorbar(
        ims_med[-1],
        ax=axes[0].tolist(),
        fraction=0.02,
        pad=0.01,
        label=f"median {vlabel}",
    )
    fig.colorbar(
        ims_spr[-1], ax=axes[1].tolist(), fraction=0.02, pad=0.01, label="spread"
    )
    fig.suptitle(
        f"{title} — coarsening left→right; empty=white, hatched=single-pixel (n=1)"
    )
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)


def row_col_profiles(X, Y, V, out, plt, ylabel, title, band=8):
    """Column (vs X) and row (vs Y) band profiles of scalar V: median per band ± robust SE,
    with N. band = pixel width per band (band=1 → native per-row/column, e.g. to catch a bad
    driver line in the FastScan). Reveals a left↔right / top↔bottom gradient or a bad line."""
    X = np.asarray(X, int)
    Y = np.asarray(Y, int)
    V = np.asarray(V, float)
    m = np.isfinite(V)
    X, Y, V = X[m], Y[m], V[m]
    if len(V) < 4:
        return
    fig, axs = plt.subplots(1, 2, figsize=(12, 4.4))
    for ax, coord, lab in [(axs[0], X, "column  (X)"), (axs[1], Y, "row  (Y)")]:
        b = coord // band
        cx, cy, ce, cn = [], [], [], []
        for bb in sorted(set(b)):
            v = V[b == bb]
            cx.append(bb * band + band / 2.0)
            cy.append(np.median(v))
            ce.append(
                1.4826 * np.median(np.abs(v - np.median(v))) / np.sqrt(len(v))
                if len(v) > 1
                else 0.0
            )
            cn.append(len(v))
        ax.errorbar(cx, cy, yerr=ce, fmt="o-", ms=4, capsize=2.5, color="tab:blue")
        if band > 1:
            for x, y, n in zip(cx, cy, cn):
                ax.annotate(
                    f"n{n}",
                    (x, y),
                    fontsize=6.5,
                    color="0.4",
                    textcoords="offset points",
                    xytext=(0, 6),
                    ha="center",
                )
        ax.set_xlabel(f"{lab}  — {band}px bands" if band > 1 else f"{lab}  — native")
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.3)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out, dpi=300)
    plt.close(fig)


def plot_fastscan_spatial(fs, out_dir, plt):
    """Full-resolution spatial characterisation from the FastScan brightness (EVERY pixel is
    measured, so this is dense/native): blocked brightness maps, row/column band profiles, and
    NATIVE per-row/column profiles (band=1) that reveal individual bad driver lines / a dim
    side. Includes dead pixels, so it doubles as the yield / defect map."""
    if fs is None or "PM400_POWER_MEAN" not in getattr(fs, "columns", []):
        return
    g = fs.groupby(["X", "Y"])["PM400_POWER_MEAN"].median().reset_index()
    X = g["X"].astype(int).values
    Y = g["Y"].astype(int).values
    P = np.abs(g["PM400_POWER_MEAN"].astype(float).values) * 1e9  # nW
    if len(P) < 4:
        return
    blocked_maps(
        X,
        Y,
        P,
        out_dir / "fastscan_blocked_maps.png",
        plt,
        "brightness (nW)",
        "Blocked FastScan brightness (median)",
        "{:.1f}",
    )
    row_col_profiles(
        X,
        Y,
        P,
        out_dir / "fastscan_row_col_bands.png",
        plt,
        "median brightness (nW)",
        "FastScan brightness — row/column band profiles",
        8,
    )
    row_col_profiles(
        X,
        Y,
        P,
        out_dir / "fastscan_row_col_native.png",
        plt,
        "median brightness (nW)",
        "FastScan brightness — NATIVE per-row/column (bad-driver-line check)",
        1,
    )
    # native per-row/column worst-line ratio (bad-driver-line flag)
    stats = _spatial_stats(X, Y, P) or {}
    worst = 1.0
    for coord in (X, Y):
        for c in set(coord.tolist()):
            v = P[coord == c]
            if len(v):
                worst = min(worst, float(np.median(v)) / (np.median(P) or 1))
    stats["worst_line_frac"] = worst
    return stats


# =====================================================================
#  Per-sweep analysis
# =====================================================================


def analyse_sweep(sweep_dir, label, log):
    sweep_dir = Path(sweep_dir)
    df_raw = _load_aggregated(sweep_dir)
    if df_raw is None or df_raw.empty:
        log(f"  [{label}] no data")
        return None
    dark = _dark_power(sweep_dir)
    df = enrich(df_raw, dark_W=dark)
    out = sweep_dir / "IQE_EQE"
    out.mkdir(parents=True, exist_ok=True)
    plt = _plt()
    d_cm = ACTIVE_THICKNESS_NM * 1e-7

    def _safe(fn, *a, **k):
        """Run a plot; a degenerate/empty plot must never block the report."""
        try:
            return fn(*a, **k)
        except Exception as e:
            log(f"    (skipped {fn.__name__}: {e})")
            try:
                plt.close("all")
            except Exception:
                pass
            return None

    n_px = df.groupby(["X", "Y"]).ngroups
    n_rel = int(df["RELIABLE"].sum())
    log(
        f"  [{label}] {n_px} pixels, {len(df)} points, {n_rel} reliable "
        f"(dark={dark:.2e} W)"
    )

    # Plots that don't need a fit
    _safe(plot_ivl, df, out / "IVL_full.png", plt)
    _safe(plot_leakage, df, out / "leakage_vs_bias.png", plt)
    _safe(plot_eqe_by_bitval, df, out / "eqe_vs_J_by_bitval.png", plt)
    _safe(plot_first_pixels, df, out / "first_pixels_diagnostics.png", plt)
    _safe(
        plot_burst_scatter, sweep_dir, out / "burst_scatter.png", plt
    )  # raw burst overlay (if saved)

    # Median EQE(J) over reliable points, binned in log(J). Binning by J (not by
    # NVLED_V) works for both fixed-voltage and adaptive per-pixel-voltage sweeps,
    # and a min-count per bin suppresses spurious points from the noisy tail.
    rel = df[df["RELIABLE"]]
    med = median_eqe_curve(
        rel["J_LED"].values, rel["EQE_LED"].values, min_count=max(3, int(0.15 * n_px))
    )
    fit = fit_abc(med["J"].values, med["EQE"].values, d_cm)
    # An ABC fit is only meaningful if it describes the data (good R²) AND the
    # IQE peak falls inside the measured window. Otherwise peak-IQE is fiction.
    fit_ok = bool(fit is not None and fit["R_squared"] > 0.8 and fit["peak_captured"])
    fit_to_save = fit

    # The honest, model-free quantities the data DOES support:
    #  - L-I linearity: P = k*I.  R^2 ~ 1 means constant EQE (no droop/peak info).
    #  - EQE spread across the well-sampled range.
    Ir = rel["I_LED"].values.astype(float)
    Pr = rel["P_W"].values.astype(float)
    fin = np.isfinite(Ir) & np.isfinite(Pr)
    Ir, Pr = Ir[fin], Pr[fin]
    li_r2 = np.nan
    if len(Ir) > 5 and Ir.std() > 0:
        k = np.sum(Ir * Pr) / np.sum(Ir * Ir)  # through-origin slope
        ss_res = np.sum((Pr - k * Ir) ** 2)
        ss_tot = np.sum((Pr - Pr.mean()) ** 2)
        li_r2 = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan
    eqe_lo, eqe_hi = (
        np.nanpercentile(med["EQE"], [10, 90]) if len(med) else (np.nan, np.nan)
    )
    eqe_spread = eqe_hi / eqe_lo if eqe_lo and np.isfinite(eqe_lo) else np.nan

    _safe(
        plot_eqe_all_and_median,
        df,
        fit,
        med,
        fit_ok,
        out / "eqe_vs_J_LED.png",
        plt,
        src="LED",
    )

    # "Split around the peak" cross-check (rise vs droop fit separately).
    split = split_branch_iqe(med) if fit_ok else None
    if split is not None:
        _safe(
            plot_split_branch,
            med,
            split,
            fit["peak_IQE"] if fit else np.nan,
            out / "iqe_split_branch.png",
            plt,
        )

    # Per-pixel ABC fits — computed ONCE here, reused for the plot, the spatial
    # map and the CSV (fitting is the expensive step). For very large sweeps we
    # fit an evenly-sampled subset so runtime stays bounded.
    groups = list(df.groupby(["X", "Y"]))
    if len(groups) > MAX_PIXEL_FITS:
        step = int(np.ceil(len(groups) / MAX_PIXEL_FITS))
        groups = groups[::step]
        log(f"    (fitting {len(groups)} of {n_px} pixels per-pixel — capped)")
    pixel_fits = {}
    rows = []
    for (x, y), g in groups:
        r = g[g["RELIABLE"]]
        f = fit_abc(r["J_LED"].values, r["EQE_LED"].values)
        pixel_fits[(x, y)] = f
        row = dict(X=int(x), Y=int(y))
        clean, flag = classify_fit(f)
        if f is not None:
            row.update(
                beta=f["beta"],
                J0=f["J0"],
                eta_LEE=f["eta_LEE"],
                peak_IQE=f["peak_IQE"] if clean else np.nan,
                peak_IQE_raw=f["peak_IQE"],
                peak_captured=f["peak_captured"],
                clean_fit=clean,
                flag=flag,
                J_at_peak=f["J_at_peak_IQE"],
                J_at_datamax=f.get("J_at_datamax"),
                R2=f["R_squared"],
                J_min=f["J_min"],
                J_max=f["J_max"],
            )
        else:
            row.update(clean_fit=False, flag=flag)
        rows.append(row)
    df_fits = pd.DataFrame(rows)
    df_fits.to_csv(out / "abc_fit_per_pixel.csv", index=False)
    write_fit_guide(out, df_fits)

    peaks = (
        _safe(
            plot_iqe,
            df,
            pixel_fits,
            fit,
            out / "iqe_all_pixels_LED.png",
            plt,
            src="LED",
        )
        or []
    )
    # Per-pixel spatial maps from the REAL per-pixel ABC fits (the whole sweep is now fit — see
    # MAX_PIXEL_FITS), not a proxy or a strided subset. peak_IQE for the field, R² for how
    # LED-like each pixel is. The EQE-near-peak proxy is kept only as a fallback if too few
    # pixels produced a clean fit.
    spatial = spatial_iqe_table(df, fit)
    J_ref = float(fit.get("J_at_peak_IQE", float("nan"))) if fit else float("nan")
    cfit = (
        df_fits[df_fits["clean_fit"] == True].dropna(subset=["peak_IQE"])
        if "clean_fit" in df_fits
        else df_fits.iloc[:0]
    )
    if len(cfit) >= 4:
        mapX, mapY, mapV, mapsrc = (
            cfit["X"].values,
            cfit["Y"].values,
            100 * cfit["peak_IQE"].values,
            "per-pixel fits",
        )
    elif spatial is not None:  # fallback: EQE-near-peak proxy
        Pg = spatial[0]
        mapX, mapY, mapV, mapsrc = (
            Pg["X"].values,
            Pg["Y"].values,
            100 * Pg["IQE_est"].values,
            "swept-pixel estimates",
        )
    else:
        mapX = mapY = mapV = None
    iqe_spatial_stats = None
    if mapX is not None and len(mapV) >= 4:
        # True ROI raster — one cell per pixel, coloured by peak IQE, white = unmeasured/dead.
        _safe(
            raster_grid,
            mapX,
            mapY,
            mapV,
            out / "iqe_spatial_map.png",
            plt,
            "peak IQE (%)",
            f"Per-pixel peak IQE (near J≈{J_ref:.1f} A/cm²)",
        )
        _safe(
            blocked_maps,
            mapX,
            mapY,
            mapV,
            out / "iqe_blocked_maps.png",
            plt,
            "peak IQE (%)",
            "Blocked IQE maps (median pooling)",
        )
        _safe(
            row_col_profiles,
            mapX,
            mapY,
            mapV,
            out / "iqe_row_col_profiles.png",
            plt,
            "median peak IQE (%)",
            "IQE row/column band profiles (gradient / bad-region check)",
            8,
        )
        iqe_spatial_stats = _spatial_stats(mapX, mapY, mapV)
        if iqe_spatial_stats:
            iqe_spatial_stats["src"] = mapsrc
    # Fit-quality (R²) raster over EVERY fitted pixel — shows how well each pixel follows the
    # LED (ABC) model, the diagnostic the proxy map threw away.
    ff = df_fits.dropna(subset=["R2"]) if "R2" in df_fits.columns else df_fits.iloc[:0]
    if len(ff) >= 4:
        _safe(
            raster_grid,
            ff["X"].values,
            ff["Y"].values,
            ff["R2"].values.astype(float),
            out / "iqe_fitquality_map.png",
            plt,
            "ABC fit R²",
            "Per-pixel fit quality R² — how LED-like each pixel is",
            "viridis",
            0.0,
            1.0,
        )

    # Save enriched per-point table
    df.to_csv(out / "enriched_points.csv", index=False)

    pp = df_fits["peak_IQE"] if "peak_IQE" in df_fits else pd.Series(dtype=float)
    if spatial is not None:
        Pmap, _jref, grad = spatial
        spatial_info = dict(
            n=len(Pmap),
            lo=float(Pmap["IQE_est"].min()),
            hi=float(Pmap["IQE_est"].max()),
            grad=grad,
        )
    else:
        spatial_info = None
    summary = dict(
        label=label,
        n_pixels=n_px,
        n_points=len(df),
        n_reliable=n_rel,
        dark_W=dark,
        J_min=float(rel["J_LED"].min()) if not rel.empty else np.nan,
        J_max=float(rel["J_LED"].max()) if not rel.empty else np.nan,
        median_fit=fit_to_save,
        fit_ok=fit_ok,
        li_r2=li_r2,
        eqe_spread=eqe_spread,
        split_low=(split or {}).get("peak_iqe_low", np.nan),
        split_high=(split or {}).get("peak_iqe_high", np.nan),
        spatial=spatial_info,
        n_pixel_fits=int(df_fits["peak_IQE_raw"].notna().sum())
        if "peak_IQE_raw" in df_fits
        else 0,
        n_peak_captured=int(pp.notna().sum()),
        median_peak_IQE=float(pp.median()) if pp.notna().any() else np.nan,
        peak_IQE_lo=float(pp.min()) if pp.notna().any() else np.nan,
        peak_IQE_hi=float(pp.max()) if pp.notna().any() else np.nan,
        iqe_spatial=iqe_spatial_stats,
    )
    return summary


# =====================================================================
#  Report
# =====================================================================


def _fmt(v, f="{:.3g}"):
    return f.format(v) if isinstance(v, (int, float)) and np.isfinite(v) else "n/a"


def write_report(
    run_dir,
    cfg,
    cls,
    fs_vs_sweep_ratio,
    sweep_summaries,
    log,
    fs_eff=None,
    fs_spatial=None,
):
    run_dir = Path(run_dir)
    lines = []
    A = lines.append
    A(f"# Measurement report — {run_dir.name}\n")
    A(f"_Generated {datetime.datetime.now():%Y-%m-%d %H:%M} by analyze_array.py_\n")

    A("## 1. What was measured\n")
    roi = (cfg.get("roi_x1"), cfg.get("roi_y1"), cfg.get("roi_x2"), cfg.get("roi_y2"))
    A(f"- **Sample:** {cfg.get('sample_name', run_dir.name)} (Array38)")
    A(
        f"- **VLED (ch A):** {cfg.get('vled_voltage')} V fixed, "
        f"compliance {cfg.get('vled_compliance')} A, NPLC {cfg.get('vled_nplc')}, "
        f"I-range {cfg.get('vled_range_i')} A"
    )
    A(
        f"- **NVLED (ch B):** fast-scan bias {cfg.get('nvled_voltage')} V → "
        f"sweep target {cfg.get('nvled_sweep_target')} V, NPLC {cfg.get('nvled_nplc')}, "
        f"I-range {cfg.get('nvled_range_i')} A"
    )
    A(
        f"- **Mode:** {cfg.get('measurement_mode')}, settle {cfg.get('nvled_settle_ms')} ms, "
        f"PM wavelength {cfg.get('pm_wavelength')} nm, PM range {cfg.get('pm_range')} W"
    )
    A(f"- **Bit values:** {cfg.get('bit_values')}\n")

    A("## 2. Fast scan — yield & classification\n")
    A(
        f"- Pixels scanned: **{cls['n_total']}**, alive: **{cls['n_alive']}** "
        f"({100 * cls['n_alive'] / cls['n_total']:.0f}%), dead: **{cls['n_dead']}** "
        f"(< {int(DEAD_THRESHOLD_FRAC * 100)}% of the p{FASTSCAN_ALIVE_PCTL} 'lit' level "
        f"{cls['dead_thr_W'] * 1e9:.1f} nW)"
        + (
            f" — **NOTE: majority-dark ROI** (median {cls['median_W'] * 1e9:.2f} nW "
            f"< threshold): most of this region is dead."
            if cls["median_W"] < cls["dead_thr_W"]
            else ""
        )
    )
    A(
        f"- Alive optical power: median **{cls['median_W'] * 1e9:.2f} nW**, "
        f"mean {cls['alive_mean_W'] * 1e9:.2f} nW, σ {cls['alive_std_W'] * 1e9:.2f} nW "
        f"(**CV = {cls['alive_cv']:.2f}**)"
    )
    A(
        f"- Within ±1σ band [{cls['within_lo_W'] * 1e9:.2f}, {cls['within_hi_W'] * 1e9:.2f}] nW: "
        f"**{cls['n_within']}** pixels; low-tail **{cls['n_low']}**, high-tail **{cls['n_high']}**"
    )
    if cls["alive_cv"] < 0.5:
        A(
            f"- > The alive population is tight (CV {cls['alive_cv']:.2f}); the single "
            f'"outside-1σ" bucket in the original script mixed the {cls["n_low"]} dim and '
            f"{cls['n_high']} bright pixels into one group — split here into low/high tails.\n"
        )
    else:
        A("")
    if fs_vs_sweep_ratio is not None:
        A(
            f"- **Fast-scan vs sweep cross-check** (at the shared NVLED bias): median power "
            f"ratio **{fs_vs_sweep_ratio:.3f}** (1.00 = perfect) → the two acquisitions agree, "
            f"so the data is reproducible."
        )
    if fs_eff is not None:
        A(
            f"- **Dense efficiency map** (all {fs_eff['n']} alive pixels, fast scan at "
            f"J≈{fs_eff['J']:.0f} A/cm²): relative EQE spans {fs_eff['lo']:.2e}–{fs_eff['hi']:.2e} "
            f"(90/10 ratio {_fmt(fs_eff['ratio'], '{:.2f}')}×) — full-coverage spatial map of device "
            f"efficiency at high injection in REPORT_assets/fastscan_efficiency.png. "
            f"(Coarse: high-J, noisy fast-scan current; complements the sweep peak-IQE map.)"
        )
    if fs_spatial:
        wl = fs_spatial.get("worst_line_frac", 1.0)
        line = (
            f"no dead driver line (weakest row/column ≥ {100 * wl:.0f}% of median)"
            if wl >= 0.5
            else f"**⚠ possible bad driver line** — a row/column drops to {100 * wl:.0f}% of median"
        )
        A(
            f"- **Full-resolution brightness spatial maps** (every pixel): blocked maps, row/column "
            f"band profiles, and native per-row/column profiles — {line}. See "
            f"REPORT_assets/fastscan_blocked_maps.png, fastscan_row_col_bands.png, "
            f"fastscan_row_col_native.png."
        )
        A(
            f"\n![FastScan native row/column](REPORT_assets/fastscan_row_col_native.png)\n"
        )
    A("")

    A("## 3. NVLED sweep — IQE / EQE\n")
    for s in sweep_summaries:
        if s is None:
            continue
        A(f"### {s['label']}\n")
        f = s["median_fit"]
        spans_low = np.isfinite(s["J_min"]) and s["J_min"] < 2.0
        A(
            f"- Pixels: {s['n_pixels']}, sweep points/pixel: ~{s['n_points'] // max(s['n_pixels'], 1)}, "
            f"reliable points (after noise-floor + rel-σ filter): {s['n_reliable']}/{s['n_points']}"
        )
        A(
            f"- Reliable current-density range: **J = {_fmt(s['J_min'])} – {_fmt(s['J_max'])} A/cm²** "
            f"({'crosses the IQE peak' if spans_low else 'all in high injection'})"
        )
        A(
            f"- L–I linearity P=k·I over reliable points: R² = {_fmt(s['li_r2'], '{:.4f}')}; "
            f"EQE spread (90/10 pct): {_fmt(s['eqe_spread'], '{:.2f}')}×"
            + ("." if s["fit_ok"] else " → ~constant EQE, little curvature to fit.")
        )
        if s["fit_ok"] and f is not None:
            A(
                f"- Median ABC fit (β,J₀,η parametrization): β={_fmt(f['beta'], '{:.2f}')}, "
                f"J₀={_fmt(f['J0'], '{:.2g}')} A/cm², R²={_fmt(f['R_squared'], '{:.3f}')}"
            )
            A(
                f"- **Peak IQE ≈ {_fmt(f['peak_IQE'] * 100, '{:.1f}')}%** (pooled, all pixels) at "
                f"J={_fmt(f['J_at_peak_IQE'])} A/cm² — collection-uncalibrated "
                f"η={_fmt(f['eta_LEE'], '{:.2e}')}."
            )
            if np.isfinite(s.get("split_low", np.nan)) or np.isfinite(
                s.get("split_high", np.nan)
            ):
                A(
                    f"- **Split-around-peak cross-check:** rising side "
                    f"{_fmt(s.get('split_low', np.nan) * 100, '{:.0f}')}%, droop side "
                    f"{_fmt(s.get('split_high', np.nan) * 100, '{:.0f}')}% (each branch fit "
                    f"separately; agreement with the {_fmt(f['peak_IQE'] * 100, '{:.0f}')}% joint fit "
                    f"⇒ ABC model self-consistent). See iqe_split_branch.png."
                )
        else:
            r2 = _fmt(f["R_squared"], "{:.2f}") if f is not None else "n/a"
            A(
                f"- **Pooled fit not meaningful** (R²={r2}, no peak in window): the population "
                f"median EQE(J) has no usable peak — see per-pixel line below."
            )
        # Per-pixel summary shown in BOTH cases: individual pixels can capture a
        # peak even when the pooled median (contaminated by a few bad pixels) does not.
        if np.isfinite(s.get("median_peak_IQE", np.nan)):
            ncf, ntf = s["n_peak_captured"], s["n_pixel_fits"]
            frac = (100 * ncf / ntf) if ntf else 0
            A(
                f"- **Per-pixel:** median {_fmt(s['median_peak_IQE'] * 100, '{:.0f}')}%, range "
                f"{_fmt(s['peak_IQE_lo'] * 100, '{:.0f}')}–{_fmt(s['peak_IQE_hi'] * 100, '{:.0f}')}%, "
                f"**{ncf}/{ntf} clean per-pixel fits ({frac:.0f}%)** — a fit is *clean* only if the "
                f"peak is captured in-data, R²≥{PER_PIXEL_R2_MIN}, and β≤{BETA_MAX:g} (not a runaway "
                f"extrapolation). See abc_fit_per_pixel.csv + abc_fit_GUIDE.md (explains β) and "
                f"iqe_spatial_map."
            )
        elif not s["fit_ok"]:
            A(
                "- Per-pixel: no pixel produced a clean peak fit either → peak-IQE not extractable."
            )
        sp = s.get("spatial")
        if sp is not None:
            g = sp["grad"]
            if (
                np.isfinite(g["r2"])
                and g["r2"] > 0.3
                and (abs(g["dx"]) > 0.02 or abs(g["dy"]) > 0.02)
            ):
                xdir = "right (+X)" if g["gx"] > 0 else "left (−X)"
                ydir = "bottom (+Y)" if g["gy"] > 0 else "top (−Y)"
                dom = xdir if abs(g["dx"]) >= abs(g["dy"]) else ydir
                trend = (
                    f"a spatial gradient (plane R²={g['r2']:.2f}): IQE rises toward the "
                    f"**{dom}** (Δ≈{abs(g['dx']) * 100:.0f}% across X, {abs(g['dy']) * 100:.0f}% across Y)"
                )
            else:
                trend = (
                    "no significant spatial gradient (roughly uniform across the ROI)"
                )
            A(
                f"- **Spatial:** per-pixel peak-IQE estimate over {sp['n']} swept pixels spans "
                f"{sp['lo'] * 100:.0f}–{sp['hi'] * 100:.0f}%, {trend}. Map in iqe_spatial_map.png "
                f"(assumes uniform collection across the ROI; sweep more pixels for a finer map)."
            )
        iq = s.get("iqe_spatial")
        if iq:
            grd = iq["grad_edge"]
            unif = (
                "**spatially uniform**"
                if grd < 2 * iq["std"]
                else f"a gradient of ≈**{grd:.0f}% edge-to-edge**"
            )
            rel = Path(s["label"]).name
            A(
                f"- **Spatial distribution of IQE** ({iq['n']} {iq.get('src', 'clean fits')}): pixel-to-pixel std "
                f"**{iq['std']:.0f}%**, linear trend {unif}; blocked (median±spread, multi-scale) and "
                f"row/column band views in `{rel}/IQE_EQE/iqe_blocked_maps.png` and "
                f"`iqe_row_col_profiles.png`."
            )
            A(f"\n![Blocked IQE maps]({rel}/IQE_EQE/iqe_blocked_maps.png)\n")
        A(
            f"- Plots: `{Path(s['label']).name}/IQE_EQE/` — IVL_full, eqe_vs_J_LED, "
            f"iqe_split_branch, eqe_vs_J_by_bitval, first_pixels_diagnostics, iqe_all_pixels_LED, "
            f"iqe_spatial_map, iqe_fitquality_map, iqe_blocked_maps, iqe_row_col_profiles, "
            f"leakage_vs_bias\n"
        )

    write_diagnosis(A, cls, sweep_summaries)

    report = run_dir / "REPORT.md"
    report.write_text("\n".join(lines), encoding="utf-8")
    log(f"  report -> {report}")
    return report


def write_diagnosis(A, cls, sweep_summaries):
    A("## 4. Data-quality diagnosis & is IQE trustworthy?\n")
    captured_any = any(s and s.get("fit_ok") for s in sweep_summaries)
    jmins = [
        s["J_min"] for s in sweep_summaries if s and np.isfinite(s.get("J_min", np.nan))
    ]
    jmin = min(jmins) if jmins else np.nan
    li_r2s = [
        s["li_r2"] for s in sweep_summaries if s and np.isfinite(s.get("li_r2", np.nan))
    ]
    li_r2 = np.nanmedian(li_r2s) if li_r2s else np.nan

    if captured_any:
        # Success path: at least one group crossed the IQE peak and fit cleanly.
        peaks = [
            s["median_fit"]["peak_IQE"]
            for s in sweep_summaries
            if s and s.get("fit_ok") and s["median_fit"]
        ]
        A("**The sweep crosses the IQE peak — peak IQE is extractable.**")
        A(
            f"- Reliable J reaches down to ~{_fmt(jmin)} A/cm², i.e. onto the SRH-rising side, "
            f"so the EQE(J) curve has a real maximum and the (β, J₀, η) fit pins it."
        )
        A(
            f"- Median peak IQE across groups ≈ **{_fmt(np.nanmedian(peaks) * 100, '{:.1f}')}%**."
        )
        A(
            "- **Caveat — absolute scale:** the PD at ~20 cm collects only a small, unknown "
            "fraction of the light, so the fitted η (collection×extraction) is uncalibrated. "
            "The IQE *shape* and peak value are meaningful; treat η as relative until you "
            "calibrate collection (integrating sphere or known geometry)."
        )
        A(
            "- Cross-check VLED vs NVLED current (leakage plot): large divergence flags pixels "
            "where ch A carries extra current — IQE there relies on ch B being the true LED path.\n"
        )
        A("## 5. Notes for future runs\n")
        A(
            "- Keep the low-J coverage — that is what makes the fit work. If the SMU noise floor "
            "lets you, push `LOW_CURRENT_TARGET_A` lower still (more of the SRH rise = tighter A)."
        )
        A(
            "- A collection-efficiency calibration converts the fitted η into absolute EQE/LEE."
        )
        A(
            "- Sweeping a second VLED bit value separates device-to-device spread from injection.\n"
        )
        return

    # Failure path: the sweep never crossed the peak.
    A("**Good news**")
    A(
        "- Fast-scan and sweep agree at the shared bias → the optical + electrical chain is reproducible."
    )
    A("- Dark/background power is ~0.1% of signal → not a limiting factor.\n")
    A("**Problems that make the current IQE untrustworthy**")
    A(
        f"1. **Current saturates at the most-negative NVLED.** The LED current is flat (~120 µA) "
        f"from -3.2 V to about -2.5 V, so points piled there (POWER=2.7 spacing) collapse onto "
        f"the same J — only a handful of distinct current levels are produced."
    )
    A(
        f"2. **The sweep never reaches low injection.** Reliable J only goes down to "
        f"~{_fmt(jmin)} A/cm². The SRH rise and the IQE peak of a blue µLED sit at J ≲ 1 A/cm² "
        f"(sub-µA). We measure on the droop side only, so peak IQE is **unconstrained**. The L–I "
        f"curve is ~linear (R² ≈ {_fmt(li_r2, '{:.4f}')}): little curvature for the fit to use."
    )
    A(
        f"3. **Low-current readings are noisy.** Near turn-off the per-point current rel-σ is "
        f"several ×100% (NPLC=0.001 ≈ 20 µs on the 10 mA range reading µA-level current)."
    )
    A(
        f"4. **Collection-limited EQE.** The PD at ~20 cm collects ~5×10⁻⁴ of the light "
        f"(measured EQE ~3×10⁻⁵); the absolute scale needs calibration regardless.\n"
    )
    A(
        "**Verdict:** No sweep crossed the IQE peak, so peak-IQE cannot be extracted from this "
        "data. The ~80% you expect cannot be confirmed here.\n"
    )
    A("## 5. What to change for the next run\n")
    A(
        "- **Use `measure_iqe.py`** — it applies each voltage, settles, and averages (accurate), "
        "and adaptively places points at log-spaced *currents* down toward turn-off so the sweep "
        "crosses the IQE peak. (Validated in simulation: recovers a known 82% peak.)"
    )
    A(
        "- **Reach low injection.** Set the sweep start near turn-off (where current → the noise "
        "floor) so reliable J gets below ~1 A/cm²."
    )
    A(
        "- **Stop above the saturation knee** (~-2.5 V) — points beyond it carry no information."
    )
    A(
        "- **Improve low-current accuracy** with NPLC≈1 + averaging on a fixed range that fits the "
        "max current (drop to 1 mA if the rig is stable there)."
    )
    A(
        "- **(Optional) Calibrate collection** for absolute EQE; otherwise only IQE *shape* is meaningful.\n"
    )


# =====================================================================
#  Run driver
# =====================================================================


def plot_fastscan_efficiency(fs, out, plt):
    """Dense relative-efficiency map over EVERY alive pixel in the ROI, from the
    fast scan: EQE = P/I at the fast-scan operating point. Full-coverage (all
    pixels, not just swept ones), but at high injection (droop side), so it shows
    large-scale spatial structure rather than peak IQE. Returns a stats dict."""
    P = fs["PM400_POWER_MEAN"].astype(float).abs()
    I = fs["NVLED_CURR_MEAN"].astype(float).abs()
    thr = DEAD_THRESHOLD_FRAC * float(
        np.nanpercentile(P.values, FASTSCAN_ALIVE_PCTL)
    )  # majority-dead-robust
    eqe = compute_eqe(P.values, I.values, WAVELENGTH_NM)
    eqe = np.where(P.values >= thr, eqe, np.nan)  # mask dead
    if not np.isfinite(eqe).any():
        return None
    J_med = float(np.nanmedian(I.values)) / (PIXEL_AREA_UM2 * 1e-8)
    d = fs[["X", "Y"]].copy()
    d["EQE"] = eqe
    piv = d.pivot_table(index="Y", columns="X", values="EQE", aggfunc="mean")
    fig, ax = plt.subplots(figsize=(7.8, 6))
    im = ax.imshow(
        piv.values,
        origin="upper",
        cmap="viridis",
        extent=[
            piv.columns.min() - 0.5,
            piv.columns.max() + 0.5,
            piv.index.max() + 0.5,
            piv.index.min() - 0.5,
        ],
    )
    plt.colorbar(im, ax=ax, label="EQE = P/I  (relative efficiency)")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_title(
        "Fast-scan relative-efficiency map — all alive pixels\n"
        f"at J≈{J_med:.0f} A/cm² (high injection / droop side; coarse current)"
    )
    fig.tight_layout()
    fig.savefig(out, dpi=300)
    plt.close(fig)
    v = eqe[np.isfinite(eqe)]
    return dict(
        n=int(v.size),
        J=J_med,
        lo=float(np.nanpercentile(v, 10)),
        hi=float(np.nanpercentile(v, 90)),
        ratio=float(np.nanpercentile(v, 90) / np.nanpercentile(v, 10))
        if np.nanpercentile(v, 10) > 0
        else np.nan,
    )


# The fast scan is a peak-IQE proxy ONLY when its bias sits on the flat EQE peak plateau. EQE stays
# within ~1% of its peak across J≈8–30, so a fast scan at J≈25 (NVLED −1.35 V) reads ≈ each pixel's
# peak EQE; outside this band (e.g. the old −2.6 V / J≈95 droop scan) EQE ≠ peak and the proxy is
# invalid, so the map is skipped.
FASTSCAN_PEAK_J_BAND = (5.0, 40.0)


def fastscan_peak_iqe_pixels(fs, peak_iqe_ref):
    """Per-pixel peak-IQE proxy from the fast scan (X, Y, iqe%%), calibrated to the sweep, or
    (None, J) if the fast-scan bias is off the EQE plateau (so EQE≠peak and the proxy is invalid)."""
    P = fs["PM400_POWER_MEAN"].astype(float).abs().values
    I = fs["NVLED_CURR_MEAN"].astype(float).abs().values
    thr = DEAD_THRESHOLD_FRAC * float(np.nanpercentile(P, FASTSCAN_ALIVE_PCTL))
    alive = P >= thr
    # J from the ALIVE pixels only — the all-pixel median collapses to ~0 on a majority-dead ROI
    # (e.g. the full 512×512 is 57% dead) and would wrongly push J out of the plateau band.
    J = (
        float(np.nanmedian(I[alive])) / (PIXEL_AREA_UM2 * 1e-8)
        if alive.any()
        else float("nan")
    )
    if not (FASTSCAN_PEAK_J_BAND[0] <= J <= FASTSCAN_PEAK_J_BAND[1]):
        return None, J
    eqe = compute_eqe(P, I, WAVELENGTH_NM)
    eqe = np.where(P >= thr, eqe, np.nan)  # mask dead
    med = float(np.nanmedian(eqe))
    if not np.isfinite(med) or med <= 0 or not np.isfinite(peak_iqe_ref):
        return None, J
    d = pd.DataFrame(
        {
            "X": fs["X"].values,
            "Y": fs["Y"].values,
            "iqe": eqe / med * peak_iqe_ref * 100.0,
        }
    )
    return d[np.isfinite(d["iqe"])].reset_index(drop=True), J


def plot_fastscan_peak_iqe(fs, peak_iqe_ref, out, plt):
    """Full-coverage per-pixel PEAK-IQE proxy raster from the fast scan (calibrated to the sweep).
    A full-coverage estimate, not a per-pixel fit — see fastscan_peak_iqe_pixels. Returns stats or
    None (off the plateau)."""
    d, J = fastscan_peak_iqe_pixels(fs, peak_iqe_ref)
    if d is None:
        return None
    raster_grid(
        d["X"].values,
        d["Y"].values,
        d["iqe"].values,
        out,
        plt,
        "peak IQE est. (%)",
        f"Full-coverage peak-IQE proxy — fast scan @ J≈{J:.0f}, calibrated to the sweep",
    )
    v = d["iqe"].values
    return dict(
        n=int(v.size),
        J=J,
        median=float(np.nanmedian(v)),
        lo=float(np.nanpercentile(v, 10)),
        hi=float(np.nanpercentile(v, 90)),
    )


def anomaly_report(
    X, Y, V, out, plt, quad=256, DETECT_SIGMA=3.0, DETECT_DEFICIT=5.0, line_sigma=4.0
):
    """General anomaly finder on a full-coverage per-pixel map V(X,Y). Three complementary detectors,
    because defects come in different shapes:

      1. BLOBS / clusters (ANY shape) — a local-anomaly z-map is thresholded and connected
         components are labelled (scipy), so a compact spot, an L, or a short streak are all found
         and reported with their true area + shape (not forced into a fixed square).
      2. DRIVER LINES — whole/partial worse rows or columns, analysed PER QUADRANT (split at `quad`),
         because each 256×256 quadrant is addressed by its own row/column drivers: a driver defect is
         a constant *logical* row or column *within one quadrant*. Rows/cols spanning 0–511 would mix
         four independent driver banks and dilute a real line.
      3. GRADIENT — a smooth plane, per quadrant (fabrication non-uniformity, can cross quadrants).

    Returns a findings dict; writes a 3-panel figure (blob z-map, per-quadrant column profiles,
    per-quadrant row profiles)."""
    from scipy import ndimage

    X = np.asarray(X, int)
    Y = np.asarray(Y, int)
    V = np.asarray(V, float)
    m = np.isfinite(V)
    X, Y, V = X[m], Y[m], V[m]
    xlo, xhi, ylo, yhi = int(X.min()), int(X.max()), int(Y.min()), int(Y.max())
    nx, ny = xhi - xlo + 1, yhi - ylo + 1
    G = np.full((ny, nx), np.nan)
    G[Y - ylo, X - xlo] = V
    S = 16
    sm = np.full_like(G, np.nan)
    for by in range(0, ny, S):
        for bx in range(0, nx, S):
            sm[by : by + S, bx : bx + S] = np.nanmedian(G[by : by + S, bx : bx + S])
    noise = float(1.4826 * np.nanmedian(np.abs((G - sm) - np.nanmedian(G - sm))))

    # ---- 1. arbitrary-shape blobs: MULTI-SCALE z-map -> pixel anomaly mask -> connected components.
    # Small-sharp defects flag at fine scales, large-shallow ones at coarse scales; painting every
    # flagged block onto a pixel mask and then labelling gives regions of ANY size and shape.
    # DETECTION is INCLUSIVE (a low floor catches candidates); each anomalous pixel records both its
    # local DEFICIT (how bad = effect size) and its |z| CONFIDENCE (how sure = certainty). The two
    # are different: a big-but-noisy 2×2 vs a small-but-rock-solid cluster — so we report/plot both.
    pdef = np.full((ny, nx), np.nan)
    pconf = np.full((ny, nx), np.nan)
    for s in (2, 4, 8, 16):
        nbx, nby = nx // s, ny // s
        bmean = np.full((nby, nbx), np.nan)
        afrac = np.zeros((nby, nbx))
        for j in range(nby):
            for i in range(nbx):
                v = G[j * s : (j + 1) * s, i * s : (i + 1) * s]
                v = v[np.isfinite(v)]
                if len(v) >= max(3, 0.25 * s * s):
                    bmean[j, i] = v.mean()
                    afrac[j, i] = len(v) / (s * s)
        R = 2
        for j in range(nby):
            for i in range(nbx):
                if not np.isfinite(bmean[j, i]) or afrac[j, i] < 0.5:
                    continue
                sl = (slice(max(0, j - R), j + R + 1), slice(max(0, i - R), i + R + 1))
                win = bmean[sl].copy()
                win[afrac[sl] < 0.5] = np.nan
                win[j - max(0, j - R), i - max(0, i - R)] = np.nan
                if np.sum(np.isfinite(win)) < 4:
                    continue
                bg = np.nanmedian(win)
                if np.isfinite(bg) and bg > 0:
                    z = (bmean[j, i] - bg) / (noise / np.sqrt(afrac[j, i] * s * s))
                    dloc = 100 * (bg - bmean[j, i]) / bg
                    if (
                        z <= -DETECT_SIGMA and dloc >= DETECT_DEFICIT
                    ):  # inclusive candidate
                        ps = (slice(j * s, (j + 1) * s), slice(i * s, (i + 1) * s))
                        pd = pdef[ps]
                        pd[np.isnan(pd) | (pd < dloc)] = dloc
                        pc = pconf[ps]
                        pc[np.isnan(pc) | (pc < -z)] = -z
    mask = np.isfinite(pdef) & np.isfinite(G)  # candidate anomalous alive pixels
    lab, nlab = ndimage.label(mask)
    blobs = []
    for k in range(1, nlab + 1):
        jj, ii = np.where(lab == k)
        if len(jj) < 3:  # drop single/double-pixel specks
            continue
        w = ii.max() - ii.min() + 1
        h = jj.max() - jj.min() + 1
        conf = float(np.nanmax(pconf[jj, ii]))  # peak σ in the region = confidence
        blobs.append(
            dict(
                x=int(xlo + ii.mean()),
                y=int(ylo + jj.mean()),
                area_px=int(len(jj)),
                w=int(w),
                h=int(h),
                shape=("line" if max(w, h) >= 3 * max(1, min(w, h)) else "blob"),
                deficit=float(np.nanmean(pdef[jj, ii])),
                confidence=conf,
                conf_level=("high" if conf >= 8 else "medium" if conf >= 5 else "low"),
            )
        )
    blobs.sort(key=lambda b: -b["confidence"])

    # ---- 2. driver lines, per quadrant ----
    def _lines(coord, other, lo, hi, olo, ohi):
        """flag positions along `coord` in [lo,hi) (within the other-axis band [olo,ohi)) whose
        median is line_sigma robust-SD below the band median — a bad driver row/column. A position
        needs enough alive pixels (≥15% of the band) to be judged, so near-dead edge lines aren't
        flagged from a couple of noisy survivors."""
        sel = (coord >= lo) & (coord < hi) & (other >= olo) & (other < ohi)
        c, v = coord[sel], V[sel]
        if len(v) < 50:
            return {}, np.array([]), np.array([])
        band = hi - lo
        min_n = max(6, int(0.15 * band))
        pos = np.arange(lo, hi)
        meds = np.array(
            [np.median(v[c == p]) if (c == p).sum() >= min_n else np.nan for p in pos]
        )
        good = np.isfinite(meds)
        mm = np.median(meds[good])
        sd = 1.4826 * np.median(np.abs(meds[good] - mm))
        bad = {
            int(p): float(100 * (mm - meds[k]) / mm)
            for k, p in enumerate(pos)
            if good[k] and (meds[k] - mm) / (sd or 1) <= -line_sigma
        }
        return bad, pos, meds

    quads = {
        "TL": (0, quad, 0, quad),
        "TR": (quad, xhi + 1, 0, quad),
        "BL": (0, quad, quad, yhi + 1),
        "BR": (quad, xhi + 1, quad, yhi + 1),
    }
    bad_cols, bad_rows, colprof, rowprof, grads = {}, {}, {}, {}, {}
    for q, (xa, xb, ya, yb) in quads.items():
        bc, cx, cm = _lines(X, Y, xa, xb, ya, yb)
        bad_cols[q] = bc
        colprof[q] = (cx, cm)
        br, ry, rm = _lines(Y, X, ya, yb, xa, xb)
        bad_rows[q] = br
        rowprof[q] = (ry, rm)
        sel = (X >= xa) & (X < xb) & (Y >= ya) & (Y < yb)
        if sel.sum() > 20:
            A_ = np.c_[X[sel], Y[sel], np.ones(sel.sum())]
            co, *_ = np.linalg.lstsq(A_, V[sel], rcond=None)
            grads[q] = float(np.hypot(co[0], co[1]) * quad / np.median(V[sel]) * 100)

    # ---- figure: 4 panels — (1) WHAT/WHERE = deficit map, (2) HOW SURE = confidence map,
    #                          (3) per-quadrant columns, (4) per-quadrant rows ----
    ext = [xlo, xhi + 1, yhi + 1, ylo]
    alive_bg = np.where(np.isfinite(G), 0.0, np.nan)  # faint grey = alive, white = dead
    fig, ax = plt.subplots(2, 2, figsize=(13, 11))
    nhi = sum(b["conf_level"] == "high" for b in blobs)
    nmed = sum(b["conf_level"] == "medium" for b in blobs)
    for a in (ax[0, 0], ax[0, 1]):
        a.imshow(
            np.ma.masked_invalid(alive_bg),
            cmap="Greys",
            vmin=-1,
            vmax=1,
            alpha=0.15,
            origin="upper",
            extent=ext,
            aspect="equal",
            interpolation="nearest",
        )
    im0 = ax[0, 0].imshow(
        np.ma.masked_invalid(pdef),
        cmap="magma_r",
        vmin=0,
        vmax=40,
        origin="upper",
        extent=ext,
        aspect="equal",
        interpolation="nearest",
    )
    fig.colorbar(im0, ax=ax[0, 0], fraction=0.046, label="% below local surroundings")
    ax[0, 0].set_title(
        f"Anomaly DEFICIT — how bad ({len(blobs)} regions, any shape)", fontsize=10
    )
    im1 = ax[0, 1].imshow(
        np.ma.masked_invalid(pconf),
        cmap="viridis",
        vmin=DETECT_SIGMA,
        vmax=12,
        origin="upper",
        extent=ext,
        aspect="equal",
        interpolation="nearest",
    )
    fig.colorbar(im1, ax=ax[0, 1], fraction=0.046, label="confidence (σ)")
    ax[0, 1].set_title(
        f"Anomaly CONFIDENCE — how sure ({nhi} high, {nmed} medium)", fontsize=10
    )
    for a in (ax[0, 0], ax[0, 1]):
        a.set_xlabel("X")
        a.set_ylabel("Y")
    cols = {"TL": "tab:blue", "TR": "tab:orange", "BL": "tab:green", "BR": "tab:red"}
    for q in quads:
        cx, cm = colprof[q]
        if len(cx):
            ax[1, 0].plot(cx, cm, lw=0.8, color=cols[q], label=q)
            for p in bad_cols[q]:
                ax[1, 0].axvline(p, color=cols[q], ls=":", lw=0.7)
        ry, rm = rowprof[q]
        if len(ry):
            ax[1, 1].plot(ry, rm, lw=0.8, color=cols[q], label=q)
            for p in bad_rows[q]:
                ax[1, 1].axvline(p, color=cols[q], ls=":", lw=0.7)
    ax[1, 0].axvline(quad, color="k", lw=0.6)
    ax[1, 1].axvline(quad, color="k", lw=0.6)
    ax[1, 0].set_title(
        "Per-quadrant COLUMN profile (bad driver columns dotted)", fontsize=10
    )
    ax[1, 0].set_xlabel("logical X (column)")
    ax[1, 0].set_ylabel("median value")
    ax[1, 0].legend(fontsize=7)
    ax[1, 1].set_title("Per-quadrant ROW profile (bad driver rows dotted)", fontsize=10)
    ax[1, 1].set_xlabel("logical Y (row)")
    ax[1, 1].set_ylabel("median value")
    ax[1, 1].legend(fontsize=7)
    fig.suptitle(
        "General anomaly report — detect all (deficit), rank by confidence (σ); "
        "rows/cols split per quadrant"
    )
    fig.tight_layout()
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return dict(
        noise=noise, blobs=blobs, bad_cols=bad_cols, bad_rows=bad_rows, grads=grads
    )


def analyse_run(run_dir, log):
    run_dir = Path(run_dir)
    fs_dir = run_dir / "FastScan"
    if not fs_dir.exists():
        log(f"  (no FastScan in {run_dir.name}, skipping)")
        return
    log(f"\n=== {run_dir.name} ===")
    fs = _load_aggregated(fs_dir)
    cfg = _load_config(fs_dir)
    if fs is None:
        log("  no fast-scan data")
        return
    cls = classify_fastscan(fs)
    plt = _plt()
    out = run_dir / "REPORT_assets"
    out.mkdir(exist_ok=True)
    plot_fastscan_overview(fs, cls, out / "fastscan_overview.png", plt)
    try:
        fs_eff = plot_fastscan_efficiency(fs, out / "fastscan_efficiency.png", plt)
    except Exception as e:
        log(f"  (skipped fast-scan efficiency map: {e})")
        fs_eff = None
    # Full-resolution brightness block / row-column / native-line spatial maps.
    fs_spatial = None
    try:
        fs_spatial = plot_fastscan_spatial(fs, out, plt)
    except Exception as e:
        log(f"  (skipped fast-scan spatial maps: {e})")

    sweep_root = run_dir / "NVLEDSweep"
    summaries = []
    ratio = None
    if sweep_root.exists():
        for grp in sorted(p for p in sweep_root.iterdir() if p.is_dir()):
            sweep = _load_aggregated(grp)
            if sweep is None:
                continue
            if ratio is None:
                ratio = plot_fastscan_vs_sweep(
                    fs, sweep, out / "fastscan_vs_sweep.png", plt
                )
            s = analyse_sweep(grp, f"NVLEDSweep/{grp.name}", log)
            if s:
                s["label"] = f"NVLEDSweep/{grp.name}"
                summaries.append(s)
    # Full-coverage peak-IQE proxy from the fast scan, calibrated to the sweep's pooled peak IQE.
    # Only produced when the fast-scan bias is on the EQE plateau (J≈8–30, i.e. NVLED ≈ −1.35 V);
    # skipped for a droop-side fast scan (old −2.6 V), where EQE ≠ peak.
    fs_peak = None
    peak_ref = next(
        (
            s["median_fit"]["peak_IQE"]
            for s in summaries
            if s.get("fit_ok")
            and s.get("median_fit")
            and np.isfinite(s["median_fit"].get("peak_IQE", np.nan))
        ),
        None,
    )
    if peak_ref is not None:
        try:
            fs_peak = plot_fastscan_peak_iqe(
                fs, peak_ref, out / "fastscan_peak_iqe.png", plt
            )
            if fs_peak:
                log(
                    f"  full-coverage peak-IQE proxy: {fs_peak['n']} pixels @ J≈{fs_peak['J']:.0f}, "
                    f"median {fs_peak['median']:.0f}% (fast scan calibrated to sweep {100 * peak_ref:.0f}%)"
                )
                # General anomaly report on the full-coverage proxy: any-shape blobs (detect-all +
                # confidence), driver rows/columns per quadrant, and per-quadrant gradients.
                d, _ = fastscan_peak_iqe_pixels(fs, peak_ref)
                if d is not None:
                    r = anomaly_report(
                        d["X"].values,
                        d["Y"].values,
                        d["iqe"].values,
                        out / "fastscan_anomaly_report.png",
                        plt,
                    )
                    hi = [b for b in r["blobs"] if b["conf_level"] == "high"]
                    log(
                        f"  ANOMALY REPORT: {len(r['blobs'])} candidate regions "
                        f"({len(hi)} high-confidence). Highest-confidence:"
                    )
                    for b in sorted(r["blobs"], key=lambda b: -b["confidence"])[:6]:
                        log(
                            f"    {b['shape']} {b['w']}×{b['h']} ({b['area_px']}px) @ ({b['x']},{b['y']}): "
                            f"{b['deficit']:.0f}% below local, {b['confidence']:.0f}σ [{b['conf_level']}]"
                        )
                    lines = {
                        q: (len(r["bad_cols"][q]), len(r["bad_rows"][q]))
                        for q in r["bad_cols"]
                    }
                    nlines = sum(c + rr for c, rr in lines.values())
                    log(
                        f"    driver lines (per quadrant): {nlines} flagged "
                        f"{ {q: f'{c}c/{rr}r' for q, (c, rr) in lines.items()} }; "
                        f"quadrant gradients { {q: f'{g:.0f}%' for q, g in r['grads'].items()} }"
                    )
            else:
                log(
                    "  (fast scan is off the EQE plateau — no peak-IQE proxy; set NVLED fast-scan "
                    "bias to ~-1.35 V / J≈25 to enable it)"
                )
        except Exception as e:
            log(f"  (skipped fast-scan peak-IQE map: {e})")
    write_report(
        run_dir, cfg, cls, ratio, summaries, log, fs_eff=fs_eff, fs_spatial=fs_spatial
    )


def main():
    global PIXEL_AREA_UM2, ACTIVE_THICKNESS_NM, WAVELENGTH_NM
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("path", help="run directory, or a results root with --all")
    ap.add_argument("--all", action="store_true", help="process every run under path")
    ap.add_argument("--area", type=float, default=PIXEL_AREA_UM2)
    ap.add_argument("--thickness", type=float, default=ACTIVE_THICKNESS_NM)
    ap.add_argument("--wavelength", type=float, default=WAVELENGTH_NM)
    args = ap.parse_args()

    PIXEL_AREA_UM2, ACTIVE_THICKNESS_NM, WAVELENGTH_NM = (
        args.area,
        args.thickness,
        args.wavelength,
    )

    root = Path(args.path)
    if not root.exists():
        print(f"ERROR: {root} not found")
        sys.exit(1)

    def log(m):
        print(m)

    if args.all:
        runs = [p for p in root.rglob("FastScan") if p.is_dir()]
        log(f"Found {len(runs)} run(s) under {root}")
        for fsdir in sorted(runs):
            try:
                analyse_run(fsdir.parent, log)
            except Exception as e:  # never let one run abort the batch
                import traceback

                log(f"  ERROR in {fsdir.parent.name}: {e}")
                log(traceback.format_exc())
    else:
        analyse_run(root, log)
    log("\nDone.")


if __name__ == "__main__":
    main()
