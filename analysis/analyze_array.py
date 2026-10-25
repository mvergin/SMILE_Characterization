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
# A reference that carries most of the LIGHT means something was still emitting while
# it was taken. Two very different things look like that, and only one is harmless:
#
#   * a stuck-on NEIGHBOUR -- run 261014-0048 pixel (33,32): P_REF/P = 0.46, and the
#     subtraction works exactly as intended, recovering a net 160.7 nW against a cohort
#     median of ~155 nW. The contaminant is removed and the pixel survives.
#   * the measured pixel ITSELF still lit -- run 261014-2308 pixel (56,51): P_REF/P =
#     0.88, and the subtraction removes the signal. Net 17 nW against the same ~150 nW
#     cohort, and I_LED collapses to 1.5 nA from a true ~185 nA.
#
# The raw fraction cannot separate 0.46 from 0.88 without a threshold fitted to two
# anecdotes. The NET power can: compared with the other pixels at the same bias, the
# neighbour case is normal and the self-lit case is an order of magnitude low.
REF_SWALLOWED_FRAC = 0.2   # net power below this fraction of the per-bias median...
REF_POWER_SUSPECT = 0.1    # ...while the reference carried at least this much light
SPLIT_AGREE_PTS = 5.0  # rising and droop branch peaks count as agreeing with the joint fit
                       # within this many IQE points: ~3.5x the fit's own run-to-run
                       # reproducibility of 1.4 points (report 28.4)
FS_AGREE = 0.05        # fast scan and sweep agree when their power ratio, at the SAME
                       # bias, is within 5 %. Section 16 measured -2.7 % at -1.35 V
REF_SUSPECT_FACTOR = 3.0  # a leakage reference this far above its NEIGHBOURS in the
                          # same sweep was not taken at bitval 0; see enrich()
NOISE_FLOOR_REF_A = 1e-8  # floor for a run that carries the leakage reference. With High-C
                          # the limit picks the range, so there is no fixed-range offset to
                          # hide under, and the measured SEM at 166 nA is 0.16% (run
                          # 261014-0048). The rel-std and optical-SNR gates below do the
                          # real work; this only rejects the obviously absent.
NOISE_FLOOR_A = 1e-6  # hard floor on |I_NVLED| (~1 µA) for runs WITHOUT the reference.
# The old sweep ran on a FIXED
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


def shared_bias_grid(df):
    """True when the targets share one bias grid (every voltage-mode sweep), so the
    median curve can be taken per bias. Current mode gives each pixel its own biases."""
    if "NVLED_V" not in df.columns or df.empty:
        return False
    per = df.groupby(["X", "Y"]).size().max()
    return df["NVLED_V"].round(4).nunique() <= 2 * per


def median_eqe_curve(J, eqe, min_count=3, n_bins=30, V=None):
    """Median EQE vs J. Returns a DataFrame[J, EQE, n, EQE_err].

    With V, the bias of every point, the curve is the median PER BIAS: one point per bias,
    placed at that bias's median J. Use it whenever the targets share a bias grid. Equal
    log-J bins put a voltage grid's six brightest biases into two bins, so the droop lost
    most of its weight in the joint fit: 261016-2252 read 80.3 % binned against a 75.9 %
    per-pixel median. Without V the points are binned in log(J), which still works when
    every pixel has its own biases (current mode).

    EQE_err is the spread of the median: 1.4826·MAD / √n (robust standard error), for
    display error bars.
    """
    J = np.asarray(J, float)
    eqe = np.asarray(eqe, float)
    m = np.isfinite(J) & np.isfinite(eqe) & (J > 0) & (eqe > 0)
    J, eqe = J[m], eqe[m]
    if J.size < min_count:
        return pd.DataFrame(columns=["J", "EQE", "n", "EQE_err"])
    if V is not None:
        Vr = np.round(np.asarray(V, float)[m], 4)
        groups = [Vr == v for v in np.unique(Vr)]
    else:
        edges = np.logspace(np.log10(J.min()), np.log10(J.max()), n_bins + 1)
        idx = np.clip(np.digitize(J, edges) - 1, 0, n_bins - 1)
        groups = [idx == b for b in range(n_bins)]
    rows = []
    for sel in groups:
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
    files = sorted(folder.glob("*aggregated_bitval=*.csv")) or sorted(
        folder.glob("**/*aggregated_bitval=*.csv")
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


def legacy_sem(df):
    """Rename the SEM columns of a run written before 2026-10-24 to their current names.

    Those runs stored the standard error of the mean under ``*_STD``. Newer runs call
    it ``*_SEM`` and write no ``*_STD`` at all, so the rename is safe in both directions.
    ``*_SAMPSTD`` and ``*_MADSTD`` do not end in ``_STD`` and are left alone.
    """
    return df.rename(columns={c: c[:-4] + "_SEM" for c in df.columns
                              if c.endswith("_STD") and c[:-4] + "_SEM" not in df.columns})


def enrich(df, dark_W=0.0):
    df = legacy_sem(df)
    # How many pixels each row is the SUM of. 1 for every ordinary row, and for every
    # run written before pooling existed -- RUN_FORMAT.md says an absent column means 1,
    # not unknown, so a pre-pooling run reads exactly as it always did.
    #
    # This divides the CURRENT DENSITY and nothing else. Power and current stay as
    # measured, because they are what the instrument read and halving them into a
    # per-member average would invent a number. EQE is a ratio of the two and is
    # unaffected either way -- sum(P)/sum(I) is the current-weighted mean of the
    # members' EQEs, which is the figure a pooled row is for. J is the one quantity
    # that must be per member, because every EQE(J) curve compares a pooled target
    # against single pixels on the same axis.
    pool_k = (df["POOL_K"].astype(float).fillna(1.0).clip(lower=1.0)
              if "POOL_K" in df.columns else 1.0)
    df["POOL_K_EFF"] = pool_k

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
    df["J_VLED"] = current_density(Iv / pool_k, PIXEL_AREA_UM2)
    df["J_NVLED"] = current_density(Inv / pool_k, PIXEL_AREA_UM2)
    df["J_LED"] = current_density(df["I_LED"] / pool_k, PIXEL_AREA_UM2)
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
        (df["PM400_POWER_SEM"].astype(float) / P.replace(0, np.nan))
        if "PM400_POWER_SEM" in df
        else 0.0
    )
    # The subtraction propagates: sigma(I_LED) = sqrt(sigma_lit^2 + sigma_ref^2), and the
    # relative error is against the (smaller) corrected current, so it grows exactly where
    # the leakage dominates. That growth is the real low-J limit now, not the raw scatter.
    if "NVLED_CURR_SEM" in df:
        sigI = df["NVLED_CURR_SEM"].astype(float)
        if "NVLED_CURR_REF_SEM" in df.columns:
            sigI = np.sqrt(sigI ** 2 + df["NVLED_CURR_REF_SEM"].astype(float) ** 2)
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
    # The 1 uA floor exists because the OLD sweep ran on a fixed 10 mA range whose
    # repeatable ~1 uA offset the rel-std gate is blind to. A run with the leakage
    # reference has neither problem -- it runs in High-C, where the limit picks the
    # range -- and that floor would throw away the entire J < 1 region this whole
    # investigation was about (run 261014-0048 reaches J = 0.17).
    floor = NOISE_FLOOR_REF_A if "NVLED_CURR_REF" in df.columns else NOISE_FLOOR_A
    reliable = pd.Series(True, index=df.index)

    def gate(name, fails):
        """Apply one reliability gate and keep what it fails as FAIL_<name>, so the
        report can count each gate in order (gate_table) instead of only the total."""
        nonlocal reliable
        fails = pd.Series(fails, index=df.index).fillna(False).astype(bool)
        df[f"FAIL_{name}"] = fails
        reliable &= ~fails

    gate("CURRENT_FLOOR", ~(df["I_LED"] >= floor))
    # A point that sat on the compliance limit is not a measurement of the device -- the
    # current is the instrument's. Run 261014-0007 had 8 such points in a row reading
    # exactly 10000.4 nA, which would otherwise have entered the fit as a plateau.
    if "NVLED_AT_COMPLIANCE" in df.columns:
        clamped = df["NVLED_AT_COMPLIANCE"].astype(str).str.lower().isin(("true", "1"))
        gate("COMPLIANCE", clamped)
    # A reference reading that jumps far above the rest of that pixel's sweep did not
    # measure the pixel at bitval 0 -- the drive had not actually come off yet, or the
    # frame write did not take. Run 261014-0048 had exactly one such point in 720:
    # pixel 56,51 at -1.300 V read 18535 nA where its other 29 points read ~227 nA, so
    # I_LED came out 7x low. Compared against the pixel's OWN median rather than the
    # per-bias median, because a pixel with a stuck-on neighbour has a legitimately large
    # reference at every bias and must not be thrown away -- the subtraction handles it.
    if "I_NVLED_REF" in df.columns and {"X", "Y", "NVLED_V"}.issubset(df.columns):
        # Compare against the NEIGHBOURING biases of the same pixel, not against the
        # pixel's median and not against the other pixels:
        #  * the reference drifts smoothly with bias (219 -> 235 nA), so neighbours are
        #    the right baseline and a real outlier stands out by ~90x;
        #  * the pixel's own median would flag the first point of every sweep, whose
        #    reference is elevated by the warm-up relaxation but is still only 0.6% of
        #    the reading;
        #  * the per-bias median across pixels would flag every point of a pixel with a
        #    stuck-on neighbour, whose reference is legitimately ~50% of lit at every
        #    bias -- and there the subtraction works, recovering the single-pixel
        #    current to 0.4% (pixel 33,32 in run 261014-0048).
        o = df.sort_values(["X", "Y", "NVLED_V"])
        local = (o.groupby(["X", "Y"])["I_NVLED_REF"]
                  .transform(lambda v: v.rolling(5, center=True, min_periods=2).median()))
        df["REF_SUSPECT"] = (o["I_NVLED_REF"] > REF_SUSPECT_FACTOR * local).reindex(df.index)
        gate("REF_SUSPECT", df["REF_SUSPECT"])

        # The reference swallowed the signal: see REF_SWALLOWED_FRAC above. Needs the
        # optical reference, because the CURRENT alone cannot tell "the neighbour was
        # subtracted off" from "the pixel was subtracted off" -- both leave I_REF a
        # large fraction of I.
        if "PM400_POWER_REF" in df.columns and "P_W" in df.columns:
            lit = _val("PM400_POWER").astype(float)
            refp = df["PM400_POWER_REF"].astype(float)
            carried = (refp / lit.replace(0, np.nan)).fillna(0.0)
            cohort = df.groupby(["BITVAL", "NVLED_V"])["P_W"].transform("median")
            swallowed = ((carried > REF_POWER_SUSPECT)
                         & (df["P_W"] < REF_SWALLOWED_FRAC * cohort))
            # ...but the net-power half only fires where the damage is already severe.
            # In run 261015-0015 four pixels whose reference carried ~58 % of their
            # light passed this gate at -1.35 V -- net still above a fifth of the
            # cohort -- and came out 42-58 % below the fast scan. The contamination is
            # a property of the PIXEL, not of one bias: its median carried fraction
            # over the whole sweep is 0.55-0.61 while every other pixel sits at 0.000
            # to 0.001. That is a 500x gap with nothing in it, so flagging the pixel
            # needs no threshold fitted to anecdotes, unlike the 0.46-vs-0.88 single
            # point the net-power test was introduced for.
            # Median over the points that MEASURED their own reference. Since
            # 17 October 2026 the sweep caches one reference per pixel below
            # leak_ref_per_point_above_v (LEAK_REF_CACHED = 1), and the cache holds a
            # min-of-k clean value by construction. Those points carry ~0 by
            # definition, so including them would dilute the median with 22 of 30
            # rows and stop this gate firing on a pixel contaminated at every bias it
            # actually read. Older runs have no column and are all measured.
            if "LEAK_REF_CACHED" in df.columns:
                carried_meas = carried.where(
                    df["LEAK_REF_CACHED"].astype(float).fillna(0.0) == 0)
            else:
                carried_meas = carried
            per_px = carried_meas.groupby([df["X"], df["Y"]]).transform("median")
            # A pixel with no measured reference at all yields NaN here, which fails
            # the comparison and leaves it unflagged -- the same fail-open the
            # .fillna(False) below gives every other missing input.
            df["REF_CONTAMINATED"] = per_px > REF_POWER_SUSPECT
            df["REF_SWALLOWED"] = swallowed
            gate("REF_SWALLOWED", swallowed)
            gate("REF_CONTAMINATED", df["REF_CONTAMINATED"])
            # ...and the median is blind to a pixel lit at SOME biases. Run 261015-0015,
            # (342,147): the reference carried 0.39-1.00 of the light at 9 of 30 biases,
            # its median stayed clean, three of the nine were adjacent so REF_SUSPECT's
            # rolling median was contaminated too, and four stayed RELIABLE. So judge each
            # point as well: reference light above the per-bias median reference (which
            # removes the dark offset) by more than REF_POWER_SUSPECT of the lit light,
            # and by more than 5 sigma of the reference's own burst, so the faint end's
            # noise does not trip it. Pooled groups with one member still lit at bitval 0
            # trip it too (14-23 % of their light): the same failure, one member wide.
            keys = [c for c in ("BITVAL", "NVLED_V") if c in df.columns]
            refx = refp - refp.groupby([df[c].round(4) for c in keys]).transform("median")
            sig = (refx > 5 * df["PM400_POWER_REF_SEM"].astype(float).abs()
                   if "PM400_POWER_REF_SEM" in df.columns else True)
            df["REF_LIT_POINT"] = ((refx / lit.abs().replace(0, np.nan) > REF_POWER_SUSPECT)
                                   & sig & (per_px <= REF_POWER_SUSPECT))
            gate("REF_LIT_POINT", df["REF_LIT_POINT"])
    if "NVLED_CURR_SEM" in df.columns:
        relstd = df["NVLED_CURR_SEM"].astype(float) / df["I_NVLED"].replace(0, np.nan)
        gate("CURRENT_NOISE", ~(relstd.fillna(np.inf) <= NVLED_RELSTD_MAX))
    # Optical-SNR gate: drop points where the PM noise rivals the signal (light at
    # the dark floor at very low J). Without this, those P≈dark points deflate the
    # rising side and bias peak-IQE low.
    if "PM400_POWER_SEM" in df.columns:
        relP = df["PM400_POWER_SEM"].astype(float) / df["P_W"].replace(0, np.nan)
        gate("OPTICAL_NOISE", ~(relP.fillna(np.inf) <= PM_RELSTD_MAX))
    # Absolute optical SNR: the signal must be above the dark/background floor —
    # otherwise EQE≈0 even when the (averaged) value is precise.
    if dark_W and dark_W > 0:
        gate("DARK_FLOOR", ~(df["P_W"] >= PM_MIN_OVER_DARK * dark_W))
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


def split_fastscan_biases(fs, cfg):
    """(rows at the classifying bias, {other bias: its rows}).

    measure_iqe's ``fastscan_voltages`` scans the whole ROI once per bias into ONE CSV,
    told apart by NVLED_V. Everything downstream -- the alive threshold, the yield, the
    spatial and efficiency maps, the sweep cross-check -- assumes one row per pixel, so
    it gets the bias the measurement itself classified on: ``nvled_v_fastscan`` from the
    run's config, else the bias scanned first. A single-bias run comes back unchanged.
    """
    if "NVLED_V" not in fs.columns:
        return fs, {}
    v = fs["NVLED_V"].astype(float).round(4)
    biases = list(dict.fromkeys(v))
    if len(biases) < 2:
        return fs, {}
    want = cfg.get("nvled_v_fastscan")
    if want is not None and round(float(want), 4) in biases:
        first = round(float(want), 4)
    elif "TIME_START" in fs.columns:
        first = float(v[fs["TIME_START"].astype(float).idxmin()])
    else:
        first = biases[0]
    return (fs[v == first].copy(),
            {b: fs[v == b].copy() for b in sorted(biases) if b != first})


def _alive_mask(df):
    """Alive per pixel at one bias, by the same rule as classify_fastscan."""
    P = df.set_index(["X", "Y"])["PM400_POWER_MEAN"].abs()
    return P, P >= DEAD_THRESHOLD_FRAC * np.nanpercentile(P.values, FASTSCAN_ALIVE_PCTL)


def fastscan_other_biases(fs, extra, out, plt):
    """Each extra fast-scan bias against the classifying one: yield, the pixels that
    change state between them, and the light ratio of the pixels alive at both.

    The state changes are the point. Array38 has pixels lit at -1.35 V that draw only
    leakage at -2.60 V (report 29.3); a second fast-scan bias finds every one of them
    across the whole ROI instead of the few the sweep happens to visit.
    """
    if not extra:
        return []
    v0 = float(fs["NVLED_V"].astype(float).round(4).iloc[0])
    P0, a0 = _alive_mask(fs)
    rows = []
    fig, ax = plt.subplots(1, len(extra) + 1, figsize=(5.2 * (len(extra) + 1), 4.8),
                           squeeze=False)
    ax = ax[0]

    def show(a, P, title, mark=None):
        x = P.index.get_level_values(0); y = P.index.get_level_values(1)
        img = np.full((y.max() - y.min() + 1, x.max() - x.min() + 1), np.nan)
        img[y - y.min(), x - x.min()] = np.clip(P.values, 1e-12, None) * 1e9
        from matplotlib.colors import LogNorm
        im = a.imshow(img, origin="upper", cmap="magma", interpolation="nearest",
                      extent=(x.min() - .5, x.max() + .5, y.max() + .5, y.min() - .5),
                      norm=LogNorm(vmin=max(1e-3, np.nanpercentile(img, 1)),
                                   vmax=np.nanpercentile(img, 99.5)))
        if mark is not None and len(mark):
            a.scatter(*zip(*mark), s=4, c="cyan", marker="s", lw=0,
                      label=f"lit at {v0:+.2f} V, dark here: {len(mark)}")
            a.legend(loc="lower left", fontsize=7)
        a.set_title(title, fontsize=9)
        plt.colorbar(im, ax=a, fraction=0.046, label="nW")

    show(ax[0], P0, f"{v0:+.3f} V (classifies): {int(a0.sum())}/{len(a0)} alive")
    for i, (v, df) in enumerate(extra.items(), 1):
        P, a = _alive_mask(df)
        P, a = P.reindex(P0.index), a.reindex(P0.index).fillna(False).astype(bool)
        lost = a0 & ~a
        gained = ~a0 & a
        both = a0 & a
        ratio = (P[both] / P0[both]).replace([np.inf, -np.inf], np.nan).dropna()
        rows.append(dict(v=v, n=int(P.notna().sum()), alive=int(a.sum()),
                         lost=int(lost.sum()), gained=int(gained.sum()),
                         ratio=float(ratio.median()) if len(ratio) else np.nan,
                         ratio_iqr=(float(ratio.quantile(.25)), float(ratio.quantile(.75)))
                         if len(ratio) else (np.nan, np.nan)))
        show(ax[i], P, f"{v:+.3f} V: {int(a.sum())}/{int(P.notna().sum())} alive",
             mark=list(lost[lost].index))
    fig.suptitle("Fast scan at every bias; alive = >= 20 % of that bias's 90th percentile",
                 fontsize=10)
    fig.tight_layout()
    fig.savefig(out, dpi=200)
    plt.close(fig)
    return rows


def _bias_tag(v):
    """-1.35 -> 'm1p35V': a file name without signs or dots."""
    return f"{v:+.2f}V".replace("+", "p").replace("-", "m").replace(".", "p")


def roi_of(cfg, X, Y):
    """(x1, y1, x2, y2) to draw a sparse map on: the run's ROI when the config gives it
    as four numbers, else the box around the points."""
    r = (cfg or {}).get("roi")
    if isinstance(r, (list, tuple)) and len(r) == 4:
        return tuple(int(v) for v in r)
    return int(np.min(X)), int(np.min(Y)), int(np.max(X)), int(np.max(Y))


def nearest_grid(X, Y, V, roi):
    """A full raster over ``roi`` where every cell takes the value of the NEAREST measured
    point. Sparse measurements (swept pixels) become a block map: each block is the area
    closest to one measured pixel, so the whole ROI is coloured and nothing is white."""
    from scipy.spatial import cKDTree
    X, Y, V = (np.asarray(a, float) for a in (X, Y, V))
    ok = np.isfinite(X) & np.isfinite(Y) & np.isfinite(V)
    x1, y1, x2, y2 = roi
    if not ok.any():
        return np.full((y2 - y1 + 1, x2 - x1 + 1), np.nan)
    gy, gx = np.mgrid[y1:y2 + 1, x1:x2 + 1]
    _, idx = cKDTree(np.c_[X[ok], Y[ok]]).query(np.c_[gx.ravel(), gy.ravel()])
    return V[ok][idx].reshape(gx.shape)


def draw_nearest(ax, fig, X, Y, V, roi, label, cmap="viridis", log=False, vlo=None,
                 vhi=None, centred=False):
    """nearest_grid on ``ax`` with the measured pixels as dots, and its colour bar."""
    from matplotlib.colors import LogNorm, TwoSlopeNorm
    G = nearest_grid(X, Y, V, roi)
    fin = np.asarray(V, float)
    fin = fin[np.isfinite(fin) & ((fin > 0) if log else True)]
    if fin.size == 0:
        ax.set_axis_off()
        return
    if vlo is None:
        vlo, vhi = np.percentile(fin, [2, 98])
    if centred:
        m = max(abs(vlo), abs(vhi), 1e-12)
        norm = TwoSlopeNorm(0.0, -m, m)
    else:
        norm = LogNorm(max(vlo, vhi * 1e-4), vhi) if log else None
    x1, y1, x2, y2 = roi
    im = ax.imshow(G, origin="upper", cmap=cmap, interpolation="nearest",
                   extent=(x1 - .5, x2 + .5, y2 + .5, y1 - .5), norm=norm,
                   **({} if (log or centred) else dict(vmin=vlo, vmax=vhi)))
    ax.plot(X, Y, ".", ms=1.5, color="k", alpha=0.6)
    fig.colorbar(im, ax=ax, fraction=0.046, label=label)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")


def nearest_raster(X, Y, V, out, plt, vlabel, title, cmap="viridis", vlo=None, vhi=None,
                   roi=None):
    """raster_grid for sparse data: the nearest-pixel block map, measured pixels dotted."""
    X, Y, V = (np.asarray(a, float) for a in (X, Y, V))
    if np.isfinite(V).sum() < 4:
        return
    roi = roi or (int(X.min()), int(Y.min()), int(X.max()), int(Y.max()))
    fig, ax = plt.subplots(figsize=(7.5, 6.2))
    draw_nearest(ax, fig, X, Y, V, roi, vlabel, cmap, vlo=vlo, vhi=vhi)
    ax.set_title(f"{title}\n({np.isfinite(V).sum()} measured pixels, dotted; every cell "
                 f"shows the nearest one)", fontsize=9)
    fig.tight_layout()
    fig.savefig(out, dpi=200)
    plt.close(fig)


def pool_members(row):
    """The pixels a target lit: POOL_MEMBERS "x:y x:y" for a pooled target, else (X, Y)."""
    s = row.get("POOL_MEMBERS")
    if isinstance(s, str) and s.strip():
        return [tuple(int(v) for v in m.split(":")) for m in s.split()]
    return [(int(row["X"]), int(row["Y"]))]


def fs_target_values(fs, targets):
    """For each swept target: its centre and its fast-scan P, I and EQE. A pooled target's
    fast-scan values are pooled after the fact: P and I summed over its members, EQE =
    sum P / sum I, the same as the sweep measured. DataFrame indexed like ``targets``."""
    fsi = (fs.groupby(["X", "Y"])[["PM400_POWER_MEAN", "NVLED_CURR_MEAN"]].mean().abs())
    rows = []
    for i, r in targets.iterrows():
        mem = pool_members(r)
        sub = fsi.reindex(mem)
        ok = sub.notna().all(axis=1).all()
        P = float(sub["PM400_POWER_MEAN"].sum()) if ok else np.nan
        I = float(sub["NVLED_CURR_MEAN"].sum()) if ok else np.nan
        rows.append(dict(cx=float(np.mean([m[0] for m in mem])),
                         cy=float(np.mean([m[1] for m in mem])),
                         FS_P_W=P, FS_I_A=I,
                         FS_EQE=float(compute_eqe(np.array([P]), np.array([I]), WAVELENGTH_NM)[0])
                         if ok and I > 0
                         else np.nan, K=len(mem)))
    return pd.DataFrame(rows, index=targets.index)


def nearest_scale(fs, targets, value_col, n_nearest=1):
    """Per fast-scan pixel: value / fast-scan EQE of the nearest swept target, single or
    pooled. Returns (scale per fs row, number of targets used), or (None, 0).

    n_nearest=1: the one nearest target. n_nearest=k: the median over the k nearest,
    which keeps a regional correction but no longer copies one target's own scatter
    onto its whole neighbourhood. n_nearest=None: one median over every target, the
    same number for every pixel (the calibration before 2026-10-25)."""
    from scipy.spatial import cKDTree
    t = targets.join(fs_target_values(fs, targets))
    r = (t[value_col] / t["FS_EQE"]).replace([np.inf, -np.inf], np.nan)
    ok = r.notna() & (r > 0)
    if not ok.any():
        return None, 0
    rv = r[ok].to_numpy(float)
    if n_nearest is None:
        return np.full(len(fs), float(np.median(rv))), int(ok.sum())
    k = min(int(n_nearest), len(rv))
    _, idx = cKDTree(np.c_[t.loc[ok, "cx"], t.loc[ok, "cy"]]).query(
        np.c_[fs["X"].to_numpy(float), fs["Y"].to_numpy(float)], k=k)
    return (rv[idx] if k == 1 else np.median(rv[idx], axis=1)), int(ok.sum())


def sweep_targets(sw_raw):
    """One row per (target, bias) of a sweep, RAW: |I| and P as measured (burst median
    where the file has it), no leakage or dark subtracted, EQE = P/I of those."""
    d = sw_raw.copy()
    val = lambda b: d[f"{b}_MEDIAN"] if f"{b}_MEDIAN" in d.columns else d[f"{b}_MEAN"]
    d["SW_I_A"] = val("NVLED_CURR").astype(float).abs()
    d["SW_P_W"] = val("PM400_POWER").astype(float).abs()
    d["SW_EQE"] = compute_eqe(d["SW_P_W"], d["SW_I_A"], WAVELENGTH_NM)
    d["SW_NVLED_V"] = d["NVLED_V"].astype(float)
    for c, src in (("SW_I_REF_A", "NVLED_CURR_REF"), ("SW_P_REF_W", "PM400_POWER_REF")):
        d[c] = d[src].astype(float).abs() if src in d.columns else np.nan
    d["POOL_K"] = (d["POOL_K"].astype(float).fillna(1.0) if "POOL_K" in d.columns else 1.0)
    if "POOL_MEMBERS" not in d.columns:
        d["POOL_MEMBERS"] = ""
    d["POOL_MEMBERS"] = d["POOL_MEMBERS"].fillna("").astype(str)
    return d


def fastscan_vs_sweep_compare(by_bias, sw_raw, eta, roi, out_dir, plt, log):
    """Fast scan against sweep, target by target, at the sweep bias nearest each fast-scan
    bias. RAW on both sides: no leakage, dark or calibration applied, so the difference
    is what the 10 ms fast-scan point and the settled sweep point really differ by.

    Writes fastscan_vs_sweep_<single|pooled>.csv (fast-scan columns first, then the
    sweep's), a scatter figure per kind, and per kind and bias a 3x4 block map:
    rows fast scan / sweep / difference, columns P, I, EQE, IQE = EQE / eta.
    """
    sw = sweep_targets(sw_raw)
    tkey = sw.groupby(["X", "Y"], as_index=False).first()[["X", "Y", "POOL_K", "POOL_MEMBERS"]]
    tables = []
    for v, fs in by_bias.items():
        fv = fs_target_values(fs, tkey)
        t = tkey.join(fv)
        biases = np.sort(sw["SW_NVLED_V"].unique())
        vs = biases[np.argmin(np.abs(biases - v))]
        s = sw[sw["SW_NVLED_V"] == vs][["X", "Y", "SW_NVLED_V", "SW_P_W", "SW_I_A",
                                        "SW_EQE", "SW_I_REF_A", "SW_P_REF_W"]]
        t = t.merge(s, on=["X", "Y"], how="left")
        t.insert(5, "FS_NVLED_V", v)
        t["DV_V"] = t["SW_NVLED_V"] - v
        tables.append(t)
    if not tables:
        return None
    T = pd.concat(tables, ignore_index=True)
    T.insert(2, "QUADRANT", np.where(T["X"] >= 256, np.where(T["Y"] >= 256, "BR", "TR"),
                                     np.where(T["Y"] >= 256, "BL", "TL")))
    cols = ["X", "Y", "QUADRANT", "POOL_K", "POOL_MEMBERS", "FS_NVLED_V", "FS_P_W", "FS_I_A",
            "FS_EQE", "SW_NVLED_V", "DV_V", "SW_P_W", "SW_I_A", "SW_EQE", "SW_I_REF_A",
            "SW_P_REF_W"]
    res = []
    for kind, sel in (("single", T["POOL_K"] <= 1), ("pooled", T["POOL_K"] > 1)):
        K = T[sel]
        if K.empty:
            continue
        K[cols].to_csv(out_dir / f"fastscan_vs_sweep_{kind}.csv", index=False)
        bl = sorted(K["FS_NVLED_V"].unique())
        # scatter: sweep against fast scan, P, I, EQE, one row per fast-scan bias
        fig, ax = plt.subplots(len(bl), 3, figsize=(15, 4.3 * len(bl)), squeeze=False)
        for row, v in zip(ax, bl):
            k = K[K["FS_NVLED_V"] == v]
            dv = float(k["DV_V"].median())
            for a, (fc, sc, lab, sca) in zip(row, (("FS_P_W", "SW_P_W", "P (nW)", 1e9),
                                                   ("FS_I_A", "SW_I_A", "|I| (uA)", 1e6),
                                                   ("FS_EQE", "SW_EQE", "EQE (x1e-4)", 1e4))):
                x, y = k[fc] * sca, k[sc] * sca
                a.scatter(x, y, s=8, alpha=0.5)
                lim = [np.nanmin([x.min(), y.min()]), np.nanmax([x.max(), y.max()])]
                a.plot(lim, lim, "k--", lw=0.8, label="1:1")
                rr = float(np.nanmedian(y / x))
                a.set_xlabel(f"fast scan {lab}")
                a.set_ylabel(f"sweep {lab}")
                a.set_title(f"FS {v:+.3f} V vs sweep {v + dv:+.3f} V (dV {dv * 1e3:+.0f} mV): "
                            f"median sweep/FS = {rr:.3f}", fontsize=8)
                a.grid(alpha=0.3)
        fig.suptitle(f"Fast scan vs sweep, {kind} targets, RAW (no reference, dark or "
                     f"calibration)", fontsize=10)
        fig.tight_layout()
        fig.savefig(out_dir / f"fastscan_vs_sweep_{kind}.png", dpi=200)
        plt.close(fig)
        # block maps: rows fast scan / sweep / difference, cols P, I, EQE, IQE
        for v in bl:
            k = K[K["FS_NVLED_V"] == v]
            dv = k["DV_V"]
            X, Y = k["cx"].to_numpy(), k["cy"].to_numpy()
            e = eta if eta and np.isfinite(eta) and eta > 0 else np.nan
            quant = (("P (nW)", "FS_P_W", "SW_P_W", 1e9, True, "magma"),
                     ("|I| (uA)", "FS_I_A", "SW_I_A", 1e6, True, "cividis"),
                     ("EQE (x1e-4)", "FS_EQE", "SW_EQE", 1e4, False, "viridis"),
                     ("IQE = EQE/eta (%)", "FS_EQE", "SW_EQE", 100 / e, False, "viridis"))
            fig, ax = plt.subplots(3, 4, figsize=(21, 15))
            for c, (lab, fc, sc, sca, lg, cm) in enumerate(quant):
                f, s = k[fc].to_numpy(float) * sca, k[sc].to_numpy(float) * sca
                both = np.r_[f, s][np.isfinite(np.r_[f, s])]
                lo, hi = (np.percentile(both, [2, 98]) if both.size else (None, None))
                for r_, vals, name in ((0, f, "fast scan"), (1, s, "sweep")):
                    draw_nearest(ax[r_, c], fig, X, Y, vals, roi, lab, cm, log=lg,
                                 vlo=lo, vhi=hi)
                    ax[r_, c].set_title(f"{name}: {lab}", fontsize=9)
                d = 100 * (s / f - 1)
                q = np.nanpercentile(np.abs(d[np.isfinite(d)]), 95) if np.isfinite(d).any() else 1
                draw_nearest(ax[2, c], fig, X, Y, d, roi, "sweep / fast scan - 1 (%)",
                             "RdBu_r", vlo=-q, vhi=q, centred=True)
                ax[2, c].set_title(f"difference: {lab}, median {np.nanmedian(d):+.1f} %",
                                   fontsize=9)
            fig.suptitle(
                f"{kind} targets: fast scan at {v:+.3f} V against the nearest sweep bias "
                f"{v + float(dv.median()):+.3f} V (dV {float(dv.min()) * 1e3:+.0f} to "
                f"{float(dv.max()) * 1e3:+.0f} mV, NOT corrected). RAW values; each block is "
                f"the area nearest one swept target (dotted).", fontsize=11)
            fig.tight_layout()
            fig.savefig(out_dir / f"fastscan_vs_sweep_maps_{kind}_{_bias_tag(v)}.png", dpi=150)
            plt.close(fig)
            res.append(dict(kind=kind, v=v, n=len(k), dv=float(dv.median())))
    return res


def leakage_map(df, roi, out, plt):
    """The leakage reference of every swept single pixel as a nearest-pixel block map, at
    the faintest, a middle and the brightest bias that measured its own reference."""
    if "NVLED_CURR_REF" not in df.columns:
        return None
    d = df.copy()
    if "POOL_K_EFF" in d.columns:
        d = d[d["POOL_K_EFF"] <= 1]
    if "LEAK_REF_CACHED" in d.columns:
        d = d[d["LEAK_REF_CACHED"].astype(float) != 1]       # its own read, not the cache
    vs = sorted(d["NVLED_V"].astype(float).round(4).unique(), reverse=True)
    if not vs:
        return None
    pick = sorted({vs[0], vs[len(vs) // 2], vs[-1]}, reverse=True)
    fig, ax = plt.subplots(1, len(pick), figsize=(6.4 * len(pick), 5.4), squeeze=False)
    for a, v in zip(ax[0], pick):
        g = d[d["NVLED_V"].astype(float).round(4) == v]
        vals = g["NVLED_CURR_REF"].astype(float).abs().to_numpy() * 1e9
        draw_nearest(a, fig, g["X"].to_numpy(), g["Y"].to_numpy(), vals, roi,
                     "|I_ref| (nA)", "plasma")
        a.set_title(f"{v:+.3f} V: {len(g)} pixels, median {np.nanmedian(vals):.0f} nA",
                    fontsize=9)
    fig.suptitle("Leakage reference (pixel addressed at bitval 0) of each swept pixel; each "
                 "block is the area nearest one pixel (dotted)", fontsize=10)
    fig.tight_layout()
    fig.savefig(out, dpi=200)
    plt.close(fig)
    return pick


def _sweep_eqe_at(sw, v):
    """Each swept pixel's EQE_LED interpolated (linear in log EQE) to bias v, or NaN where
    the pixel's sweep does not span v."""
    out = {}
    for (x, y), g in sw.groupby(["X", "Y"]):
        g = g.sort_values("NVLED_V")
        V = g["NVLED_V"].to_numpy(float)
        E = g["EQE_LED"].to_numpy(float)
        ok = np.isfinite(E) & (E > 0)
        if ok.sum() >= 2 and V[ok].min() <= v <= V[ok].max():
            out[(int(x), int(y))] = float(np.exp(np.interp(v, V[ok], np.log(E[ok]))))
    return pd.Series(out, dtype=float)


def fastscan_bias_maps(by_bias, eta, enr, out_dir, plt):
    """One figure per fast-scan bias: optical power, current, and IQE, every pixel.

    IQE = EQE / eta, the ABC model's own relation, with eta from the sweep's fit. The fast
    scan's EQE is first scaled by k(v): the median, over the swept single pixels alive at
    that bias, of the sweep's EQE interpolated to v over the fast scan's EQE. k absorbs
    what the fast scan does differently -- a ~10 ms dwell and no leakage reference -- and
    is reported; k = 1 where the sweep does not span the bias. This is IQE AT that bias,
    not each pixel's peak IQE: for that see the plateau-only peak-IQE proxy.

    Power and current are drawn for every pixel, dead ones included: a dead pixel that
    draws current is a short, one that draws none is open, and only the current map can
    tell them apart. IQE is drawn for the alive pixels only.
    """
    sw, members_of = None, {}
    if enr is not None and {"EQE_LED", "RELIABLE", "NVLED_V"} <= set(enr.columns):
        keep = enr["RELIABLE"].astype(bool)
        sw = enr[keep][["X", "Y", "NVLED_V", "EQE_LED"]]
        if "POOL_MEMBERS" in enr.columns:       # pooled targets scale their members' area
            members_of = enr.groupby(["X", "Y"])["POOL_MEMBERS"].first().to_dict()
    from matplotlib.colors import LogNorm
    stats, rows_for_grid = [], []
    for v, df in by_bias.items():
        df = df.groupby(["X", "Y"], as_index=False).mean(numeric_only=True)
        P = df["PM400_POWER_MEAN"].abs().to_numpy(float)
        I = df["NVLED_CURR_MEAN"].abs().to_numpy(float)
        alive = P >= DEAD_THRESHOLD_FRAC * np.nanpercentile(P, FASTSCAN_ALIVE_PCTL)
        eqe = np.where(alive, compute_eqe(P, I, WAVELENGTH_NM), np.nan)
        # k per pixel: the sweep's EQE at v over the fast scan's EQE, of the NEAREST
        # swept target (single or pooled). One array-wide median used to stand in for
        # every pixel; a region that reads differently is now corrected locally.
        k, n_k, scale = np.nan, 0, None
        if sw is not None:
            ref = _sweep_eqe_at(sw, v)
            if len(ref):
                tg = pd.DataFrame({"X": [i[0] for i in ref.index],
                                   "Y": [i[1] for i in ref.index], "val": ref.values})
                tg["POOL_MEMBERS"] = [members_of.get((x, y), "")
                                      for x, y in zip(tg["X"], tg["Y"])]
                scale, n_k = nearest_scale(df, tg, "val")
                if scale is not None:
                    k = float(np.nanmedian(scale))
        iqe = (eqe * (scale if scale is not None else 1.0) / eta * 100.0
               if eta and np.isfinite(eta) and eta > 0 else None)
        X = df["X"].to_numpy(int); Y = df["Y"].to_numpy(int)
        ext = (X.min() - .5, X.max() + .5, Y.max() + .5, Y.min() - .5)

        def grid(vals):
            g = np.full((Y.max() - Y.min() + 1, X.max() - X.min() + 1), np.nan)
            g[Y - Y.min(), X - X.min()] = vals
            return g

        panels = [("optical power (nW)", grid(np.clip(P, 1e-13, None) * 1e9), "magma", True),
                  ("current |I_NVLED| (µA)", grid(np.clip(I, 1e-12, None) * 1e6), "cividis", True),
                  ("EQE = P/I (x1e-4, collected)", grid(eqe * 1e4), "viridis", False)]
        if iqe is not None:
            panels.append((f"IQE at this bias (%), k from the nearest of {n_k} swept "
                           f"targets (median {k:.3f})" if np.isfinite(k)
                           else "IQE at this bias (%), uncalibrated (k = 1)",
                           grid(iqe), "viridis", False))
        rows_for_grid.append((v, panels, ext))
        fig, ax = plt.subplots(1, len(panels), figsize=(5.4 * len(panels), 4.9), squeeze=False)
        for a, (lab, g, cmap, log) in zip(ax[0], panels):
            fin = g[np.isfinite(g) & (g > 0)] if log else g[np.isfinite(g)]
            if fin.size == 0:
                a.set_axis_off(); continue
            lo, hi = np.nanpercentile(fin, 1), np.nanpercentile(fin, 99.5)
            norm = LogNorm(vmin=max(lo, hi * 1e-4), vmax=hi) if log else None
            im = a.imshow(g, origin="upper", cmap=cmap, interpolation="nearest", extent=ext,
                          norm=norm, **({} if log else dict(vmin=np.nanpercentile(fin, 2),
                                                             vmax=np.nanpercentile(fin, 98))))
            plt.colorbar(im, ax=a, fraction=0.046, label=lab)
            a.set_title(lab, fontsize=9); a.set_xlabel("X"); a.set_ylabel("Y")
        J_alive = float(np.nanmedian(I[alive])) / (PIXEL_AREA_UM2 * 1e-8) if alive.any() else np.nan
        fig.suptitle(f"Fast scan at {v:+.3f} V: {int(alive.sum())}/{alive.size} alive, "
                     f"median J of the alive {J_alive:.1f} A/cm²", fontsize=10)
        fig.tight_layout()
        name = f"fastscan_maps_{_bias_tag(v)}.png"
        fig.savefig(out_dir / name, dpi=200)
        plt.close(fig)
        iv = iqe[np.isfinite(iqe)] if iqe is not None else np.array([])
        stats.append(dict(v=v, alive=int(alive.sum()), n=int(alive.size),
                          P=float(np.nanmedian(P[alive])) if alive.any() else np.nan,
                          J=J_alive, k=k, n_k=n_k, file=name,
                          iqe=float(np.median(iv)) if iv.size else np.nan,
                          iqe_lo=float(np.percentile(iv, 10)) if iv.size else np.nan,
                          iqe_hi=float(np.percentile(iv, 90)) if iv.size else np.nan))
    # Every bias in one figure: a row per bias, columns P, I, EQE, IQE.
    if rows_for_grid:
        ncol = max(len(p) for _, p, _ in rows_for_grid)
        fig, ax = plt.subplots(len(rows_for_grid), ncol,
                               figsize=(5.2 * ncol, 4.6 * len(rows_for_grid)), squeeze=False)
        for row, (v, panels, ext) in zip(ax, rows_for_grid):
            for a, (lab, g, cmap, log) in zip(row, panels):
                fin = g[np.isfinite(g) & (g > 0)] if log else g[np.isfinite(g)]
                if fin.size == 0:
                    a.set_axis_off(); continue
                lo, hi = np.nanpercentile(fin, 1), np.nanpercentile(fin, 99.5)
                norm = LogNorm(vmin=max(lo, hi * 1e-4), vmax=hi) if log else None
                im = a.imshow(g, origin="upper", cmap=cmap, interpolation="nearest",
                              extent=ext, norm=norm,
                              **({} if log else dict(vmin=np.nanpercentile(fin, 2),
                                                     vmax=np.nanpercentile(fin, 98))))
                plt.colorbar(im, ax=a, fraction=0.046)
                a.set_title(f"{v:+.3f} V: {lab}", fontsize=8)
            for a in row[len(panels):]:
                a.set_axis_off()
        fig.suptitle("Fast scan at every bias: optical power, current, EQE and IQE "
                     "(IQE scaled pixel by pixel by the nearest swept target)", fontsize=11)
        fig.tight_layout()
        fig.savefig(out_dir / "fastscan_grid.png", dpi=150)
        plt.close(fig)
    return stats


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
    """Cross-check: fast-scan against the sweep point at the SAME bias, per pixel.

    This used to take `NVLED_V.min()`, the sweep's most negative point, on the stale
    assumption that the sweep ends where the fast scan sits. It no longer does: the
    fast scan runs at -1.35 V and the sweep ends at -2.60 V, so the ratio was
    P(-2.60 V) / P(-1.35 V) = 2.8 -- the diode's own I-V, reported to the reader as
    "the fast scan and the sweep do NOT agree". They agree to 11 %.

    Returns (ratio, bias_used, bias_gap) so the caller can say which bias it used, or
    None when the sweep has no point near the fast scan's bias.
    """
    if "NVLED_V" not in fs or fs["NVLED_V"].isna().all():
        return None
    v_fs = float(fs["NVLED_V"].median())
    if "POOL_K" in sweep.columns:
        # Single pixels only. A pooled row holds the SUM of its K members' light under
        # its first member's coordinates, and the fast scan holds that one pixel: the
        # ratio came out 2.0-2.8 on the pooled full-array runs for that reason alone.
        sweep = sweep[sweep["POOL_K"].astype(float).fillna(1.0) <= 1.0]
    biases = np.asarray(sorted(sweep["NVLED_V"].unique()), float)
    v_sw = float(biases[int(np.argmin(np.abs(biases - v_fs)))])
    gap = abs(v_sw - v_fs)
    if biases.min() <= v_fs <= biases.max() and gap > 1e-4:
        # Interpolate each pixel's sweep power TO the fast-scan bias (linear in log P
        # between the two bracketing biases) instead of comparing at the nearest one.
        # The nearest was 50 mV away in every full run (-1.30 vs -1.35 V), and on this
        # I-V 50 mV is worth ~15 % by itself: run 261016-2252 reported "agree" at 0.847
        # while the band was wide enough to call nearly anything agreement.
        def _at(g):
            g = g.sort_values("NVLED_V")
            pw = g["PM400_POWER_MEAN"].abs().to_numpy(float)
            ok = pw > 0
            if ok.sum() < 2:
                return np.nan
            return float(np.exp(np.interp(v_fs, g["NVLED_V"].to_numpy(float)[ok],
                                          np.log(pw[ok]))))
        s32 = (sweep.groupby(["X", "Y"])[["NVLED_V", "PM400_POWER_MEAN"]].apply(_at)
               .rename("PM400_POWER_MEAN").reset_index())
        v_sw, gap = v_fs, 0.0
    else:
        s32 = sweep[np.isclose(sweep["NVLED_V"], v_sw)]
    m = fs.merge(s32, on=["X", "Y"], suffixes=("_fs", "_sw"))
    if m.empty:
        return None
    fig, ax = plt.subplots(1, 2, figsize=(12, 5))
    pf = m["PM400_POWER_MEAN_fs"].abs() * 1e9
    ps = m["PM400_POWER_MEAN_sw"].abs() * 1e9
    ax[0].scatter(pf, ps, s=14, alpha=0.7)
    lim = [0, max(pf.max(), ps.max()) * 1.05]
    ax[0].plot(lim, lim, "k--", lw=1, label="1:1")
    ax[0].set_xlabel(f"Fast-scan P at {v_fs:+.3f} V (nW)")
    ax[0].set_ylabel(f"Sweep P at {v_sw:+.3f} V (nW)")
    ax[0].set_title("Optical power consistency")
    ax[0].legend()
    ratio = (ps / pf).replace([np.inf, -np.inf], np.nan).dropna()
    if ratio.empty:
        plt.close(fig)
        return None
    ax[1].hist(ratio, bins=30, color="tab:purple", alpha=0.8)
    ax[1].axvline(1.0, color="k", ls="--")
    ax[1].set_xlabel("Sweep / Fast-scan power ratio")
    ax[1].set_ylabel("pixels")
    ax[1].set_title(f"median ratio = {ratio.median():.3f} "
                    f"({v_sw:+.3f} V vs {v_fs:+.3f} V, gap {1000 * gap:.0f} mV)")
    fig.tight_layout()
    fig.savefig(out, dpi=300)
    plt.close(fig)
    return float(ratio.median()), v_fs, v_sw, gap


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


def eqe_median_sd(rel, src="LED"):
    """Per bias (or per log-J bin when the targets share no bias grid): median EQE over
    targets, its sample SD (ddof=1) and count, at the median J. DataFrame[J, EQE, SD, n]."""
    Jc, Ec = f"J_{src}", f"EQE_{src}"
    r = rel[np.isfinite(rel[Jc]) & np.isfinite(rel[Ec]) & (rel[Jc] > 0) & (rel[Ec] > 0)]
    if r.empty:
        return pd.DataFrame(columns=["J", "EQE", "SD", "n"])
    if shared_bias_grid(r):
        key = r["NVLED_V"].astype(float).round(4)
    else:                           # current-source sweeps: no common bias, bin in log J
        edges = np.logspace(np.log10(r[Jc].min()), np.log10(r[Jc].max()), 31)
        key = np.clip(np.digitize(r[Jc], edges) - 1, 0, 29)
    g = r.groupby(key)
    out = pd.DataFrame({"J": g[Jc].median(), "EQE": g[Ec].median(),
                        "SD": g[Ec].std(ddof=1), "n": g[Ec].size()})
    return out[out["n"] >= 3].sort_values("J").reset_index(drop=True)


def plot_eqe_by_kind(df, out_dir, plt, src="LED"):
    """EQE(J) for single pixels and for pooled targets, in separate figures.

    Each figure: left, every target's reliable points as a line; right, the median
    over targets at each bias with +-1 sample SD across targets. Writes
    eqe_vs_J_LED_single.png and, when the run pooled, eqe_vs_J_LED_pooled.png.
    """
    Jc, Ec = f"J_{src}", f"EQE_{src}"
    k = df["POOL_K_EFF"] if "POOL_K_EFF" in df.columns else pd.Series(1.0, index=df.index)
    files = []
    for kind, sel in (("single", k <= 1), ("pooled", k > 1)):
        d = df[sel]
        if d.empty:
            continue
        rel = d[d["RELIABLE"]]
        stats = eqe_median_sd(rel, src)
        n_t = d.groupby(["X", "Y"]).ngroups
        fig, ax = plt.subplots(1, 2, figsize=(13, 5.4), sharey=True)
        for _, g in rel.groupby(["X", "Y"]):
            g = g.sort_values(Jc)
            ax[0].plot(g[Jc], g[Ec], lw=0.6, alpha=0.3, color="tab:blue")
        ax[1].errorbar(stats["J"], stats["EQE"], yerr=stats["SD"], fmt="o-", ms=4,
                       capsize=2, color="tab:blue", label="median over targets, ±1 SD")
        for a in ax:
            a.set_xscale("log")
            a.set_yscale("log")
            a.set_xlabel("J (A/cm²)")
            a.grid(True, which="both", alpha=0.3)
        ax[0].set_ylabel("Collected EQE (fraction)")
        ax[0].set_title(f"every {kind} target ({n_t}), reliable points", fontsize=10)
        ax[1].set_title(f"median and SD over {kind} targets", fontsize=10)
        ax[1].legend(fontsize=8, loc="lower left")
        what = ("one pixel" if kind == "single" else
                "K pixels lit together: EQE = sum P / sum I, J per member")
        fig.text(0.5, 0.005,
                 f"Each point: one target ({what}) at one bias. EQE from the burst MEDIAN "
                 f"current and power, leakage reference and dark subtracted. Right: the "
                 f"MEDIAN over targets per bias, bars ±1 sample SD (ddof=1) across targets.",
                 ha="center", va="bottom", fontsize=7.5, color="0.3", wrap=True)
        fig.suptitle(f"EQE vs J -- {kind} targets ({src} current)", fontsize=11)
        fig.tight_layout(rect=(0, 0.04, 1, 1))
        files.append(out_dir / f"eqe_vs_J_{src}_{kind}.png")
        fig.savefig(files[-1], dpi=200)
        plt.close(fig)
    return files


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
                label=f"joint fit ({pooled_fit['peak_IQE'] * 100:.0f}%)",
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
                f"IQE comes from the joint fit only: peak {pooled_fit['peak_IQE'] * 100:.0f}%"
            )
            ax.text(
                0.5,
                0.06,
                "Only the joint fit to the median curve is available "
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
    samp = next(iter(sorted((Path(sweep_dir) / "raw_data").glob("*Sweep_samples.csv"))),
                Path(sweep_dir) / "raw_data" / "Sweep_samples.csv")
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
    cm = plt.get_cmap(cmap).copy()
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


GATES = [
    ("CURRENT_FLOOR", "I_LED below the noise floor ({floor} A)"),
    ("COMPLIANCE", "the point sat on its compliance limit"),
    ("REF_SUSPECT", f"reference > {REF_SUSPECT_FACTOR:g}x the rolling median of its 5 neighbours"),
    ("REF_SWALLOWED", f"reference carried > {REF_POWER_SUSPECT:g} of the light and net power "
                      f"< {REF_SWALLOWED_FRAC:g}x the per-bias median"),
    ("REF_CONTAMINATED", f"the pixel's median reference light (measured references) "
                         f"> {REF_POWER_SUSPECT:g} of its lit light"),
    ("REF_LIT_POINT", f"this point's reference light, above the per-bias median, "
                      f"> {REF_POWER_SUSPECT:g} of the lit light and > 5 sigma"),
    ("CURRENT_NOISE", f"current SEM / current > {NVLED_RELSTD_MAX:g}"),
    ("OPTICAL_NOISE", f"power SEM / power > {PM_RELSTD_MAX:g}"),
    ("DARK_FLOOR", f"power < {PM_MIN_OVER_DARK:g}x the dark level"),
]


def gate_table(df):
    """Per gate, in enrich()'s order: points it fails, and points it REMOVES (fails while
    every earlier gate passed), plus the limits in force and the reference retries."""
    floor = NOISE_FLOOR_REF_A if "NVLED_CURR_REF" in df.columns else NOISE_FLOOR_A
    left = pd.Series(True, index=df.index)
    rows = []
    for name, rule in GATES:
        col = f"FAIL_{name}"
        if col not in df.columns:
            rows.append(dict(gate=name, rule=rule.format(floor=f"{floor:g}"), fails=None,
                             removed=None))
            continue
        f = df[col].astype(bool)
        rows.append(dict(gate=name, rule=rule.format(floor=f"{floor:g}"),
                         fails=int(f.sum()), removed=int((f & left).sum())))
        left &= ~f
    limits = (df["NVLED_LIMIT_A"].value_counts().sort_index().to_dict()
              if "NVLED_LIMIT_A" in df.columns else {})
    tries = {}
    if "LEAK_REF_TRIES" in df.columns:
        t = df["LEAK_REF_TRIES"]
        if "LEAK_REF_CACHED" in df.columns:
            t = t[df["LEAK_REF_CACHED"].astype(float).fillna(0) == 0]
        tries = t.dropna().astype(int).value_counts().sort_index().to_dict()
    return dict(rows=rows, limits=limits, tries=tries, n=len(df))


# Both checks refit every sampled pixel many times and are OFF by default: the
# truncation check took most of an analysis and changed no conclusion. --truncation
# and --decimation turn them on.
RUN_TRUNCATION = False
RUN_DECIMATION = False
TRUNC_MAX_PIXELS = 100   # ponytail: per-pixel truncation on an even sample; ~0.2 s a fit
TRUNC_N = 5              # cut 0..5 points from each end
TRUNC_WARN_PTS = 2.0     # warn when one or two cuts move the peak IQE more than this
TRUNC_MIN_POINTS = 8     # never fit fewer points than this


def _trunc_peaks(J, E):
    """Peak IQE (%) after cutting n = 0..TRUNC_N points from the low-J end, the high-J
    end, or both; NaN where the fit is not clean or too few points remain."""
    o = np.argsort(J)
    J, E = np.asarray(J, float)[o], np.asarray(E, float)[o]
    out = {}
    for mode in ("low", "high", "both"):
        for c in range(TRUNC_N + 1):
            lo, hi = (c, len(J)) if mode == "low" else (0, len(J) - c) if mode == "high" \
                else (c, len(J) - c)
            if hi - lo < TRUNC_MIN_POINTS:
                out[(mode, c)] = np.nan
                continue
            f = fit_abc(J[lo:hi], E[lo:hi])
            ok, _ = classify_fit(f)
            out[(mode, c)] = 100 * f["peak_IQE"] if ok else np.nan
    return out


DECIM_STEPS = (1, 2, 3, 4, 5, 10)   # use every k-th bias
DECIM_MAX_PIXELS = 300               # ponytail: an even sample; ~0.2 s a fit, 6 fits a pixel
DECIM_MIN_POINTS = 4                 # an ABC fit needs at least this many biases


def decimated_biases(vs, k):
    """Every k-th bias from the first, and always the last: both ends stay in."""
    keep = list(vs[::k])
    if keep[-1] != vs[-1]:
        keep.append(vs[-1])
    return keep


def decimation_check(df, out, plt, log):
    """Could fewer biases have given the same peak IQE?

    Keeps the lowest and highest bias and uses every k-th one in between, k in
    DECIM_STEPS, with no interpolation. Refits the joint per-bias median curve and the
    clean single pixels (an even sample of up to DECIM_MAX_PIXELS) on each subset, and
    reports how far the peak IQE moves from the fit on every bias. Writes
    decimation.csv and decimation.png. Off by default (RUN_DECIMATION).
    """
    rel = df[df["RELIABLE"]].copy()
    rel["_v"] = rel["NVLED_V"].astype(float).round(4)
    vs = sorted(rel["_v"].unique())
    if len(vs) < 2 * DECIM_MIN_POINTS:
        return None
    groups = [g for _, g in rel.groupby(["X", "Y"])
              if not ("POOL_K_EFF" in g and g["POOL_K_EFF"].iloc[0] > 1)]
    if len(groups) > DECIM_MAX_PIXELS:
        groups = groups[:: int(np.ceil(len(groups) / DECIM_MAX_PIXELS))]
    n_px = rel.groupby(["X", "Y"]).ngroups

    def peak(J, E):
        if len(J) < DECIM_MIN_POINTS:
            return np.nan
        f = fit_abc(np.asarray(J, float), np.asarray(E, float))
        ok, _ = classify_fit(f)
        return 100 * f["peak_IQE"] if ok else np.nan

    joint, pix = {}, {}
    for k in DECIM_STEPS:
        keep = decimated_biases(vs, k)
        sub = rel[rel["_v"].isin(keep)]
        med = median_eqe_curve(sub["J_LED"].values, sub["EQE_LED"].values,
                               min_count=max(3, int(0.15 * n_px)), V=sub["_v"].values)
        joint[k] = (peak(med["J"], med["EQE"]) if len(med) else np.nan, len(keep))
        pix[k] = []
        for g in groups:
            gg = g[g["_v"].isin(keep)]
            pix[k].append(peak(gg["J_LED"], gg["EQE_LED"]))
    base_px = np.asarray(pix[1], float)
    rows = []
    for k in DECIM_STEPS:
        d = np.asarray(pix[k], float) - base_px
        d = d[np.isfinite(d)]
        rows.append(dict(k=k, n_biases=joint[k][1], joint_peak_IQE=joint[k][0],
                         joint_shift=joint[k][0] - joint[1][0],
                         pixel_median_shift=float(np.median(d)) if d.size else np.nan,
                         pixel_p10=float(np.percentile(d, 10)) if d.size else np.nan,
                         pixel_p90=float(np.percentile(d, 90)) if d.size else np.nan,
                         pixel_median_abs_shift=float(np.median(np.abs(d))) if d.size else np.nan,
                         n_pixels=int(d.size), n_pixels_sampled=len(groups)))
    t = pd.DataFrame(rows)
    t.to_csv(out / "decimation.csv", index=False)
    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    ax.plot(t.n_biases, t.joint_shift, "ko-", label="joint per-bias median fit")
    ax.plot(t.n_biases, t.pixel_median_shift, "o-", color="tab:green",
            label=f"per-pixel median shift ({int(t.n_pixels.max())} pixels)")
    ax.fill_between(t.n_biases, t.pixel_p10, t.pixel_p90, color="tab:green", alpha=0.2,
                    label="per-pixel 10-90 %")
    for r in t.itertuples():
        ax.annotate(f"every {r.k}", (r.n_biases, r.pixel_median_shift), fontsize=7,
                    textcoords="offset points", xytext=(3, 5))
    ax.axhline(0, color="0.5", lw=0.8)
    ax.invert_xaxis()
    ax.set_xlabel("biases used (both ends always kept)")
    ax.set_ylabel("peak IQE change vs every bias (points)")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.suptitle(f"Decimation: the peak IQE from every k-th of {len(vs)} biases, "
                 f"no interpolation", fontsize=10)
    fig.tight_layout()
    fig.savefig(out / "decimation.png", dpi=200)
    plt.close(fig)
    for r in t.itertuples():
        log(f"    decimation every {r.k} ({r.n_biases} biases): joint {r.joint_shift:+.2f}, "
            f"per-pixel median {r.pixel_median_shift:+.2f} [{r.pixel_p10:+.2f}, "
            f"{r.pixel_p90:+.2f}] points, {r.n_pixels} pixels")
    return t


def truncation_check(df, med, per_bias, out, plt, log):
    """How far the peak IQE moves when the ends of the curve are cut.

    The joint fit is refitted on its median curve with n points cut, and the per-pixel
    fits on an even sample of up to TRUNC_MAX_PIXELS clean single pixels. A peak that
    moves by more than TRUNC_WARN_PTS within the first two cuts rests on the end points
    rather than on the curve, and the report says so. Writes truncation.csv and
    truncation.png.
    """
    rel = df[df["RELIABLE"]]
    rows = []
    joint = _trunc_peaks(med["J"].values, med["EQE"].values) if len(med) else {}
    for (mode, c), v in joint.items():
        rows.append(dict(scope="joint", mode=mode, n_cut=c, peak_IQE=v))
    groups = []
    for (x, y), g in rel.groupby(["X", "Y"]):
        if "POOL_K_EFF" in g and g["POOL_K_EFF"].iloc[0] > 1:
            continue
        groups.append(g)
    full = max((len(g) for g in groups), default=0)
    groups = [g for g in groups if len(g) == full]
    if len(groups) > TRUNC_MAX_PIXELS:
        groups = groups[:: int(np.ceil(len(groups) / TRUNC_MAX_PIXELS))]
    shifts = {}
    for g in groups:
        tp = _trunc_peaks(g["J_LED"].values, g["EQE_LED"].values)
        base = tp.get(("low", 0), np.nan)
        if not np.isfinite(base):
            continue
        for k, v in tp.items():
            shifts.setdefault(k, []).append(v - base)
    for (mode, c), v in sorted(shifts.items()):
        a = np.asarray(v, float)
        a = a[np.isfinite(a)]
        rows.append(dict(scope="pixels", mode=mode, n_cut=c,
                         median_shift=float(np.median(a)) if a.size else np.nan,
                         p10=float(np.percentile(a, 10)) if a.size else np.nan,
                         p90=float(np.percentile(a, 90)) if a.size else np.nan,
                         n_pixels=int(a.size)))
    t = pd.DataFrame(rows)
    t.to_csv(out / "truncation.csv", index=False)
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.4), sharey=True)
    j0 = joint.get(("low", 0), np.nan)
    warn = []
    for a, mode in zip(ax, ("low", "high", "both")):
        jj = t[(t.scope == "joint") & (t["mode"] == mode)]
        a.plot(jj.n_cut, jj.peak_IQE - j0, "ko-", label="joint fit")
        pp = t[(t.scope == "pixels") & (t["mode"] == mode)]
        if len(pp):
            a.plot(pp.n_cut, pp.median_shift, "o-", color="tab:green",
                   label=f"per-pixel median shift ({int(pp.n_pixels.max())} px)")
            a.fill_between(pp.n_cut, pp.p10, pp.p90, color="tab:green", alpha=0.2,
                           label="per-pixel 10-90 %")
        a.axhspan(-TRUNC_WARN_PTS, TRUNC_WARN_PTS, color="0.9", zorder=0)
        a.axhline(0, color="0.5", lw=0.8)
        a.set_title({"low": "cut from the low-J end", "high": "cut from the high-J end",
                     "both": "cut from both ends"}[mode], fontsize=10)
        a.set_xlabel("points cut")
        for scope, col, sel in (("joint fit", "peak_IQE", jj), ("per-pixel median", "median_shift", pp)):
            vals = (sel[sel.n_cut.between(1, 2)][col] - (j0 if col == "peak_IQE" else 0)).abs()
            if len(vals) and vals.max() > TRUNC_WARN_PTS:
                warn.append(f"{scope}, {mode}: {vals.max():.1f} points")
    ax[0].set_ylabel("peak IQE change (points)")
    ax[0].legend(fontsize=8)
    fig.suptitle(f"Truncation check: does the peak IQE rest on the end points? "
                 f"(grey band ±{TRUNC_WARN_PTS:g} points)", fontsize=10)
    fig.tight_layout()
    fig.savefig(out / "truncation.png", dpi=200)
    plt.close(fig)
    if warn:
        log("    !! truncation: the peak IQE moves > "
            f"{TRUNC_WARN_PTS:g} points within two cuts -- " + "; ".join(warn))
    return dict(warn=warn, n_pixels=int(t[t.scope == "pixels"].n_pixels.max())
                if (t.scope == "pixels").any() else 0)


def iqe_dir(sweep_dir):
    """Where a sweep group's IQE/EQE outputs go: <run>/REPORT_assets/IQE_EQE, or
    IQE_EQE_<group> for a group not named Sweep. Every plot and table of a run lives
    under REPORT_assets; NVLEDSweep holds only what the instrument wrote. Runs analysed
    before 2026-10-25 had this under NVLEDSweep/<group>/IQE_EQE."""
    sweep_dir = Path(sweep_dir)
    name = "IQE_EQE" if sweep_dir.name == "Sweep" else f"IQE_EQE_{sweep_dir.name}"
    return sweep_dir.parent.parent / "REPORT_assets" / name


def analyse_sweep(sweep_dir, label, log):
    sweep_dir = Path(sweep_dir)
    df_raw = _load_aggregated(sweep_dir)
    if df_raw is None or df_raw.empty:
        log(f"  [{label}] no data")
        return None
    dark = _dark_power(sweep_dir)
    df = enrich(df_raw, dark_W=dark)
    out = iqe_dir(sweep_dir)
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
    # Say it here or it is invisible: a pooled target looks like one pixel in every
    # groupby, and its EQE is a current-weighted mean of K members, not one pixel's.
    pooled = (df[df["POOL_K_EFF"] > 1].groupby(["X", "Y"])
              if "POOL_K_EFF" in df.columns else None)
    n_pool = pooled.ngroups if pooled is not None else 0
    log(
        f"  [{label}] {n_px} pixels, {len(df)} points, {n_rel} reliable "
        f"(dark={dark:.2e} W)"
        + (f" -- {n_pool} of the {n_px} are POOLED targets covering "
           f"{int(df[df['POOL_K_EFF'] > 1].groupby(['X', 'Y'])['POOL_K_EFF'].first().sum())} "
           f"pixels; their EQE is a current-weighted mean of their members and their J "
           f"is per member" if n_pool else "")
    )

    # Plots that don't need a fit
    _safe(plot_ivl, df, out / "IVL_full.png", plt)
    _safe(plot_leakage, df, out / "leakage_vs_bias.png", plt)
    _safe(plot_eqe_by_bitval, df, out / "eqe_vs_J_by_bitval.png", plt)
    _safe(plot_first_pixels, df, out / "first_pixels_diagnostics.png", plt)
    _safe(
        plot_burst_scatter, sweep_dir, out / "burst_scatter.png", plt
    )  # raw burst overlay (if saved)
    _safe(plot_settling_traces, sweep_dir, out / "settling_traces.png", plt)
    _safe(leakage_map, df, roi_of(_load_config(sweep_dir), df["X"], df["Y"]),
          out / "leakage_map.png", plt)

    # Median EQE(J) over reliable points, binned in log(J). Binning by J (not by
    # NVLED_V) works for both fixed-voltage and adaptive per-pixel-voltage sweeps,
    # and a min-count per bin suppresses spurious points from the noisy tail.
    rel = df[df["RELIABLE"]]
    per_bias = shared_bias_grid(rel)
    med = median_eqe_curve(
        rel["J_LED"].values, rel["EQE_LED"].values, min_count=max(3, int(0.15 * n_px)),
        V=rel["NVLED_V"].values if per_bias else None,
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
        row = dict(X=int(x), Y=int(y),
                   POOL_K=int(g["POOL_K_EFF"].iloc[0]) if "POOL_K_EFF" in g else 1)
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
    # Single pixels and pooled targets apart: a pooled target is K pixels' summed light
    # and current, so mixing the two puts different things in one median.
    _safe(plot_eqe_by_kind, df, out, plt)
    if "POOL_K_EFF" in df.columns and (df["POOL_K_EFF"] > 1).any():
        kk = df.groupby(["X", "Y"])["POOL_K_EFF"].first()
        for kind, keep in (("single", kk <= 1), ("pooled", kk > 1)):
            sub = set(kk[keep].index)
            _safe(plot_iqe, df[[xy in sub for xy in zip(df["X"], df["Y"])]],
                  {xy: f for xy, f in pixel_fits.items() if xy in sub}, fit,
                  out / f"iqe_all_pixels_LED_{kind}.png", plt, src="LED")
    # Per-pixel spatial maps from the REAL per-pixel ABC fits (the whole sweep is now fit — see
    # MAX_PIXEL_FITS), not a proxy or a strided subset. peak_IQE for the field, R² for how
    # LED-like each pixel is. The EQE-near-peak proxy is kept only as a fallback if too few
    # pixels produced a clean fit.
    spatial = spatial_iqe_table(df, fit)
    J_ref = float(fit.get("J_at_peak_IQE", float("nan"))) if fit else float("nan")
    cfit = (
        df_fits[df_fits["clean_fit"] == True].dropna(subset=["peak_IQE"])
        # peak_IQE is absent when no pixel produced a fit at all (a 4-pixel, 6-point
        # sweep), and dropna then raised instead of skipping the map.
        if "clean_fit" in df_fits and "peak_IQE" in df_fits
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
            nearest_raster,
            mapX,
            mapY,
            mapV,
            out / "iqe_spatial_map.png",
            plt,
            "peak IQE (%)",
            f"Per-pixel peak IQE (near J≈{J_ref:.1f} A/cm²)",
            roi=roi_of(_load_config(sweep_dir), mapX, mapY),
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
            nearest_raster,
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
            roi=roi_of(_load_config(sweep_dir), ff["X"].values, ff["Y"].values),
        )

    # Save enriched per-point table
    df.to_csv(out / "enriched_points.csv", index=False)

    pp = df_fits["peak_IQE"] if "peak_IQE" in df_fits else pd.Series(dtype=float)
    single_clean = (df_fits[(df_fits["clean_fit"] == True) & (df_fits["POOL_K"] <= 1)]
                    if {"clean_fit", "POOL_K", "eta_LEE"} <= set(df_fits.columns)
                    else df_fits.iloc[:0])
    gates = gate_table(df)
    trunc = (_safe(truncation_check, df, med, per_bias, out, plt, log)
             if RUN_TRUNCATION else None)
    if RUN_DECIMATION:
        _safe(decimation_check, df, out, plt, log)
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
        peak_IQE_p10=float(pp.quantile(0.1)) if pp.notna().any() else np.nan,
        peak_IQE_p90=float(pp.quantile(0.9)) if pp.notna().any() else np.nan,
        eta_pixels=float(single_clean["eta_LEE"].median()) if len(single_clean) else np.nan,
        curve="median per bias" if per_bias else "median in log-J bins",
        gates=gates,
        trunc=trunc,
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
    fs_biases=(),
    fs_v0=None,
    fs_maps=(),
):
    run_dir = Path(run_dir)
    lines = []
    A = lines.append
    A(f"# Measurement report — {run_dir.name}\n")
    A(f"_Generated {datetime.datetime.now():%Y-%m-%d %H:%M} by analyze_array.py_\n")

    A("## 1. What was measured\n")
    roi = (cfg.get("roi_x1"), cfg.get("roi_y1"), cfg.get("roi_x2"), cfg.get("roi_y2"))
    # Key names follow IQEConfig. The previous block read pre-refactor names
    # (vled_nplc, nvled_voltage, measurement_mode, bit_values, ...) and so printed
    # "None" for almost every setting -- a report that silently misstates what was run.
    def g(key, default="?"):
        v = cfg.get(key, default)
        return default if v is None else v

    A(f"- **Sample:** {g('sample_name', run_dir.name)}, ROI {g('roi')}")
    A(f"- **VLED (ch A):** {g('vled_voltage')} V fixed, "
      f"compliance {g('vled_compliance')} A, I-range {g('vled_range_i')} A")
    A(f"- **NVLED (ch B):** fast scan at {g('nvled_v_fastscan')} V; sweep "
      f"{g('nvled_v_coarse_start')} to {g('nvled_v_coarse_stop')} V in "
      f"{g('n_fine')} points, {g('sweep_direction')}")
    if cfg.get("highc_b"):
        A(f"- **High-C on**, so the compliance picks the measure range: "
          f"{g('vramp_lowv_limit_a')} A on the rising side, "
          f"{g('vramp_highv_limit_a')} A on the droop side")
    else:
        A(f"- **High-C OFF**, "
          + (f"fixed I-range {g('nvled_range_i')} A" if cfg.get("nvled_range_i")
             else "the measure range follows the limit (same ladder as High-C)")
          + " -- on a carrier with its capacitors (Array42) readings below ~10 uA are "
            "range artifacts, not currents (INVESTIGATION_REPORT.md section 1)")
    A(f"- **Effort:** NPLC {g('vramp_lowv_nplc')}/{g('vramp_highv_nplc')} "
      f"(rising/droop), settle {g('vramp_lowv_settle_s')}/{g('vramp_highv_settle_s')} s, "
      f"PM averaging {g('vramp_lowv_n_pm')}/{g('vramp_highv_n_pm')}")
    if cfg.get("leak_reference"):
        A(f"- **Leakage reference on:** every point re-measured at bitval "
          f"{g('leak_ref_bitval')}; I_LED = ch B minus reference")
    else:
        A("- **Leakage reference OFF** -- at low bias most of ch B is backplane leakage, "
          "so the rising side of this run is not trustworthy")
    A(f"- **Mode:** {g('fine_sweep_mode')} ramp, PM wavelength {g('pm_wavelength')} nm, "
      f"PM range {g('pm_range')} W, bit values {g('bitvals')}\n")

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
        r, v_fs, v_sw, gap = fs_vs_sweep_ratio
        # The two are only comparable if the sweep actually has a point near the fast
        # scan's bias. The diode's I-V is steep here, so a 100 mV gap is already worth
        # tens of percent and must not be read as a disagreement.
        where = (f"sweep interpolated to {v_fs:+.3f} V" if gap == 0 else
                 f"fast scan {v_fs:+.3f} V vs the nearest sweep point {v_sw:+.3f} V, "
                 f"gap **{1000 * gap:.0f} mV**")
        A(
            f"- **Fast-scan vs sweep cross-check** ({where}): median power ratio "
            f"**{r:.3f}** (1.00 = perfect) → "
            + (f"**inconclusive** — a {1000 * gap:.0f} mV gap on this I-V explains a "
               f"ratio of this size on its own, and the fast-scan bias lies outside the "
               f"sweep, so it cannot be interpolated. Sweep a point at {v_fs:+.3f} V."
               if gap > 0.01 else
               f"they agree within {100 * FS_AGREE:.0f} % at the same bias."
               if abs(r - 1) <= FS_AGREE else
               f"**they differ by {100 * (r - 1):+.1f} %** at the same bias. The fast "
               f"scan reads after a ~10 ms dwell, the sweep after its full one; treat "
               f"the fast-scan map as relative only, or rescale it by this ratio.")
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
    if fs_biases:
        A(f"- **Fast scan at {1 + len(fs_biases)} biases.** Everything above uses the "
          f"classifying bias, {fs_v0:+.3f} V. Each other bias, against it, with alive judged "
          f"at each bias by the same rule:\n")
        A(f"| bias | alive | lit at {fs_v0:+.2f} V, dark here | dark at {fs_v0:+.2f} V, lit here "
          f"| light ratio, pixels alive at both (IQR) |")
        A("|---|---|---|---|---|")
        for r in fs_biases:
            A(f"| {r['v']:+.3f} V | {r['alive']}/{r['n']} ({100 * r['alive'] / max(1, r['n']):.1f} %) "
              f"| {r['lost']} | {r['gained']} | {r['ratio']:.3f} "
              f"({r['ratio_iqr'][0]:.3f}–{r['ratio_iqr'][1]:.3f}) |")
        A("\n![Fast scan at every bias](REPORT_assets/fastscan_biases.png)\n")
    if fs_maps:
        A("- **Optical power, current and IQE at every fast-scan bias.** IQE = EQE / η with η "
          "from the sweep's ABC fit, after scaling the fast scan's EQE by k: the median, over the "
          "swept single pixels, of the sweep's EQE at that bias over the fast scan's. k absorbs "
          "the fast scan's ~10 ms dwell and missing leakage reference. This is IQE *at* the "
          "bias, not each pixel's peak IQE. The current map shows dead pixels too: one that "
          "draws current is a short, one that draws none is open.\n")
        A("| bias | alive | median P, alive | median J, alive | k (pixels) | IQE median (10–90 %) |")
        A("|---|---|---|---|---|---|")
        for m in fs_maps:
            A(f"| {m['v']:+.3f} V | {m['alive']}/{m['n']} | {m['P'] * 1e9:.2f} nW | "
              f"{m['J']:.1f} A/cm² | "
              + (f"{m['k']:.3f} ({m['n_k']}) | " if np.isfinite(m['k']) else "— | ")
              + (f"{m['iqe']:.0f} % ({m['iqe_lo']:.0f}–{m['iqe_hi']:.0f} %) |"
                 if np.isfinite(m['iqe']) else "no sweep fit |"))
        for m in fs_maps:
            A(f"\n![Fast scan at {m['v']:+.3f} V](REPORT_assets/{m['file']})")
        A("")
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
        # The per-pixel median leads: it is the number the investigation quotes, and it
        # does not depend on how the median curve weights the biases.
        if np.isfinite(s.get("median_peak_IQE", np.nan)):
            ncf, ntf = s["n_peak_captured"], s["n_pixel_fits"]
            A(
                f"- **Peak IQE, per-pixel median: {_fmt(s['median_peak_IQE'] * 100, '{:.1f}')} %** "
                f"(10–90 %: {_fmt(s['peak_IQE_p10'] * 100, '{:.0f}')}–"
                f"{_fmt(s['peak_IQE_p90'] * 100, '{:.0f}')} %, full range "
                f"{_fmt(s['peak_IQE_lo'] * 100, '{:.0f}')}–{_fmt(s['peak_IQE_hi'] * 100, '{:.0f}')} %), "
                f"from **{ncf} clean fits of {ntf} targets**. A fit is *clean* only if the peak "
                f"is captured in-data, R²≥{PER_PIXEL_R2_MIN} and β≤{BETA_MAX:g}. See "
                f"abc_fit_per_pixel.csv, abc_fit_GUIDE.md and iqe_spatial_map."
            )
        elif not s["fit_ok"]:
            A("- Per-pixel: no pixel produced a clean peak fit → peak IQE not extractable.")
        if s["fit_ok"] and f is not None:
            A(
                f"- Joint fit — one ABC fit (β, J₀, η) to the {s.get('curve', 'median curve')} "
                f"of all targets' reliable points: peak IQE "
                f"**{_fmt(f['peak_IQE'] * 100, '{:.1f}')} %** at J={_fmt(f['J_at_peak_IQE'])} A/cm², "
                f"β={_fmt(f['beta'], '{:.2f}')}, J₀={_fmt(f['J0'], '{:.2g}')} A/cm², "
                f"R²={_fmt(f['R_squared'], '{:.3f}')}, collection-uncalibrated "
                f"η={_fmt(f['eta_LEE'], '{:.2e}')}."
            )
            if np.isfinite(s.get("split_low", np.nan)) or np.isfinite(
                s.get("split_high", np.nan)
            ):
                lo, hi = s.get("split_low", np.nan), s.get("split_high", np.nan)
                joint = f["peak_IQE"]
                # The verdict has to come from the numbers. This sentence used to say
                # "agreement => self-consistent" whatever they were: run 261020-1454 printed
                # it under rising 92 % and droop 67 %.
                agree = all(np.isfinite(b) and abs(b - joint) <= SPLIT_AGREE_PTS / 100
                            for b in (lo, hi))
                A(
                    f"- **Split-around-peak cross-check:** rising side "
                    f"{_fmt(lo * 100, '{:.0f}')}%, droop side {_fmt(hi * 100, '{:.0f}')}% "
                    f"(each branch fit separately) against "
                    f"{_fmt(joint * 100, '{:.0f}')}% joint → "
                    + ("they agree within "
                       f"{SPLIT_AGREE_PTS:g} points, so one ABC β describes both sides."
                       if agree else
                       f"**they disagree** by more than {SPLIT_AGREE_PTS:g} points: one ABC β "
                       f"does not describe both sides, so the joint peak IQE depends on how "
                       f"many points each side contributes. Quote the peak with that spread.")
                    + " See iqe_split_branch.png."
                )
        else:
            r2 = _fmt(f["R_squared"], "{:.2f}") if f is not None else "n/a"
            A(
                f"- **Joint fit not meaningful** (R²={r2}, no peak in window): the median "
                f"EQE(J) of all targets has no usable peak — see the per-pixel line above."
            )
        tr = s.get("trunc")
        if tr:
            A("- **Truncation check** (truncation.png, truncation.csv): "
              + ("**the peak IQE moves more than "
                 f"{TRUNC_WARN_PTS:g} points within two cuts** — " + "; ".join(tr["warn"])
                 + ". It rests on the end points; quote it with that spread."
                 if tr["warn"] else
                 f"cutting one or two points from either end moves the peak IQE by less than "
                 f"{TRUNC_WARN_PTS:g} points, joint fit and per-pixel median "
                 f"({tr['n_pixels']} pixels) alike."))
        gt = s.get("gates")
        if gt:
            A(f"\n**Reliability gates**, in the order `enrich()` applies them, over {gt['n']} "
              f"points. *Removes* counts what the gate drops that every earlier gate kept:\n")
            A("| gate | rule | fails | removes |")
            A("|---|---|---|---|")
            for r in gt["rows"]:
                A(f"| `{r['gate']}` | {r['rule']} | "
                  + (f"{r['fails']} | {r['removed']} |" if r["fails"] is not None
                     else "— | not applied (input column absent) |"))
            if gt["limits"]:
                A("\nCompliance limits in force (`NVLED_LIMIT_A`): "
                  + ", ".join(f"{float(k):g} A × {v}" for k, v in gt["limits"].items()) + ".")
            if gt["tries"]:
                A("Frame writes per measured leakage reference (`LEAK_REF_TRIES`): "
                  + ", ".join(f"{k} × {v}" for k, v in gt["tries"].items()) + ".")
            A("")
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
        # relative to REPORT.md, which sits in the run directory beside REPORT_assets
        rel = "REPORT_assets/" + iqe_dir(Path(s["label"])).name
        iq = s.get("iqe_spatial")
        if iq:
            grd = iq["grad_edge"]
            unif = (
                "**spatially uniform**"
                if grd < 2 * iq["std"]
                else f"a gradient of ≈**{grd:.0f}% edge-to-edge**"
            )
            A(
                f"- **Spatial distribution of IQE** ({iq['n']} {iq.get('src', 'clean fits')}): pixel-to-pixel std "
                f"**{iq['std']:.0f}%**, linear trend {unif}; blocked (median±spread, multi-scale) and "
                f"row/column band views in `{rel}/iqe_blocked_maps.png` and "
                f"`iqe_row_col_profiles.png`."
            )
            A(f"\n![Blocked IQE maps]({rel}/iqe_blocked_maps.png)\n")
        A(
            f"- Plots: `{rel}/` — IVL_full, eqe_vs_J_LED, "
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


# The fast scan is a peak-IQE proxy only near the EQE peak, and the band below is generous.
# MEASURED over the 100 swept pixels of 261015-0015: median EQE runs 93.7% to 99.7% of its peak
# across J≈8–30 -- a 6.3% spread, not the ~1% this comment used to claim -- and the peak sits at
# −1.05 V, J=7.2. A fast scan at the default −1.35 V (J≈33) therefore reads ~92% of peak, so the
# map it produces is a RANK map calibrated to the sweep, not each pixel's peak EQE. Outside the
# band (e.g. the old −2.6 V / J≈95 droop scan, 74% of peak) it is invalid and the map is skipped.
# Withdrawn claim and the numbers behind it: investigation report 26.2.
FASTSCAN_PEAK_J_BAND = (5.0, 40.0)


QUAD_COLORS = {"TL": "tab:blue", "TR": "tab:orange", "BL": "tab:green", "BR": "tab:red"}
IP_DENSITY_BINS = 200     # square bins across the larger of the two spans


def _quadrant_col(df):
    """QUADRANT from the file when it has one (since 2026-10-25), else from X and Y."""
    if "QUADRANT" in df.columns:
        return df["QUADRANT"].astype(str)
    q = (df["X"].astype(int) >= 256).astype(int) + 2 * (df["Y"].astype(int) >= 256).astype(int)
    return q.map({0: "TL", 1: "TR", 2: "BL", 3: "BR"})


def plot_settling_traces(sweep_dir, out, plt):
    """The dwell transients the sweep records in settling_traces.npz, one curve a bias.

    bias_major records ONE trace per bias: |I| and P polled through the bias dwell on
    the first pixel lit, at whatever rate the meter allows (~4 Hz), and only where the
    dwell is at least trace_settling_min_s. Each curve is divided by its own last 10 %,
    so all biases share one axis and the panel shows how far from settled each was.
    Panels: current, optical power, and their ratio P/I (EQE up to a constant).
    """
    npz = next(iter(sorted(Path(sweep_dir).glob("*settling_traces.npz"))), None)
    if npz is None:
        return None
    with np.load(npz) as d:
        tr = {k: d[k] for k in d.files if d[k].ndim == 2 and d[k].shape[0] == 3}
    if not tr:
        return None

    def bias(k):
        try:
            return float(k.split("_v")[-1])
        except ValueError:
            return np.nan

    keys = sorted(tr, key=bias)
    cmap = plt.get_cmap("viridis")
    fig, ax = plt.subplots(1, 3, figsize=(17, 4.8), sharex=True)
    for i, k in enumerate(keys):
        t, I, P = tr[k]
        I, P = np.abs(I), np.abs(P)
        tail = t >= t[-1] - 0.1 * (t[-1] - t[0])
        c = cmap(i / max(1, len(keys) - 1))
        for a, y in zip(ax, (I, P, P / np.where(I > 0, I, np.nan))):
            ref = np.nanmedian(y[tail])
            if np.isfinite(ref) and ref:
                a.plot(t, y / ref, lw=0.9, color=c, label=f"{bias(k):+.3f} V")
    for a, name in zip(ax, ("|I|", "P", "P / I")):
        a.axhline(1, color="0.5", lw=0.8)
        a.set_xlabel("time in the bias dwell (s)")
        a.set_ylabel(f"{name} / its last 10 %")
        a.set_title(name, fontsize=10)
        a.grid(alpha=0.3)
    ax[0].legend(fontsize=7, ncol=2, title="bias")
    fig.suptitle(f"Settling through each bias dwell, first pixel lit "
                 f"({len(keys)} traces from {npz.name})", fontsize=10)
    fig.tight_layout()
    fig.savefig(out, dpi=200)
    plt.close(fig)
    return len(keys)


def plot_fastscan_ip(fs, out_dir, plt, tag=""):
    """Fast scan: current on x, optical power on y, three ways. One set per bias.

    fastscan_IP_quadrant: coloured by quadrant, all pixels and zoomed on the dim ones,
        where each quadrant's own leakage separates the clouds.
    fastscan_IP_density: coloured by how many pixels share a square cell, the cell
        being IP_DENSITY_BINS-th of the larger span, with 1 uA on x the same length
        as 1 nW on y. A distance in those units weighs current and light equally, so
        the colours do not change when an axis is rescaled.
    fastscan_IP_position: coloured by where the pixel sits: a 2D colour for (X, Y),
        then X alone and Y alone.
    """
    from matplotlib.colors import LogNorm
    I = fs["NVLED_CURR_MEAN"].astype(float).abs().values * 1e6     # uA
    P = fs["PM400_POWER_MEAN"].astype(float).abs().values * 1e9    # nW
    quad = _quadrant_col(fs).values
    v = float(fs["NVLED_V"].iloc[0]) if "NVLED_V" in fs.columns else float("nan")
    ok = np.isfinite(I) & np.isfinite(P)
    I, P, quad = I[ok], P[ok], quad[ok]
    X, Y = fs["X"].astype(int).values[ok], fs["Y"].astype(int).values[ok]
    lbl = f"fast scan @ {v:+.3f} V, {ok.sum()} pixels"
    files = []

    # 1. by quadrant, full and zoomed on the dim end
    # The near-origin corner: pixels that give almost no light, so the current is the
    # quadrant's leakage. Below 10 % of the dead threshold in light, and below 10 % of
    # the lit pixels' median current, which leaves out the shorted ones near the limit.
    p_lit = float(np.nanpercentile(P, FASTSCAN_ALIVE_PCTL))
    p_thr = 0.1 * DEAD_THRESHOLD_FRAC * p_lit
    lit = P >= DEAD_THRESHOLD_FRAC * p_lit
    i_thr = 0.1 * float(np.nanmedian(I[lit])) if lit.any() else np.inf
    dim = (P < p_thr) & (I < i_thr)
    fig, ax = plt.subplots(1, 2, figsize=(13, 5.2))
    for a, sel, ttl in ((ax[0], np.ones_like(dim), "all pixels"),
                        (ax[1], dim, f"near the origin: P < {p_thr:.2g} nW, I < {i_thr:.2g} uA: the leakage")):
        for q, c in QUAD_COLORS.items():
            m = sel & (quad == q)
            a.scatter(I[m], P[m], s=1, alpha=0.25, color=c, label=f"{q} ({m.sum()})",
                      rasterized=True)
        a.set_xlabel("|I_NVLED| (uA)")
        a.set_ylabel("P (nW)")
        a.set_title(ttl, fontsize=9)
        a.grid(alpha=0.3)
        a.legend(fontsize=8, markerscale=8)
    fig.suptitle(f"Current vs optical power by quadrant -- {lbl}", fontsize=10)
    fig.tight_layout()
    files.append(out_dir / f"fastscan_IP_quadrant{tag}.png")
    fig.savefig(files[-1], dpi=200)
    plt.close(fig)

    # 2. density, equal weight for 1 uA and 1 nW
    span = max(np.ptp(I), np.ptp(P), 1e-12)
    cell = span / IP_DENSITY_BINS
    ix = np.floor((I - I.min()) / cell).astype(np.int64)
    iy = np.floor((P - P.min()) / cell).astype(np.int64)
    key = ix * (iy.max() + 1) + iy
    _, inv, cnt = np.unique(key, return_inverse=True, return_counts=True)
    dens = cnt[inv]
    o = np.argsort(dens)                       # densest on top
    fig, ax = plt.subplots(figsize=(7.5, 5.8))
    sc = ax.scatter(I[o], P[o], c=dens[o], s=2, cmap="viridis",
                    norm=LogNorm(), rasterized=True)
    fig.colorbar(sc, ax=ax, label="pixels in the same cell")
    ax.set_xlabel("|I_NVLED| (uA)")
    ax.set_ylabel("P (nW)")
    ax.grid(alpha=0.3)
    ax.set_title(f"Density -- {lbl}\ncell {cell:.3g} uA x {cell:.3g} nW: 1 uA and 1 nW "
                 f"weigh the same", fontsize=9)
    fig.tight_layout()
    files.append(out_dir / f"fastscan_IP_density{tag}.png")
    fig.savefig(files[-1], dpi=200)
    plt.close(fig)

    # 3. by position: 2D colour, X, Y
    rgb = np.c_[X / 511.0, np.full(len(X), 0.35), Y / 511.0]
    fig, ax = plt.subplots(1, 3, figsize=(18, 5.2))
    ax[0].scatter(I, P, c=rgb, s=1, alpha=0.4, rasterized=True)
    ax[0].set_title("colour = position: red grows with X, blue with Y", fontsize=9)
    for a, val, name in ((ax[1], X, "X"), (ax[2], Y, "Y")):
        sc = a.scatter(I, P, c=val, s=1, alpha=0.4, cmap="turbo", vmin=0, vmax=511,
                       rasterized=True)
        fig.colorbar(sc, ax=a, label=f"{name} (px)")
        a.set_title(f"colour = {name}", fontsize=9)
    inset = ax[0].inset_axes([0.72, 0.05, 0.25, 0.25])
    g = np.linspace(0, 1, 64)
    inset.imshow(np.dstack([np.tile(g, (64, 1)), np.full((64, 64), 0.35),
                            np.tile(g[:, None], (1, 64))]), origin="upper",
                 extent=(0, 511, 511, 0))
    inset.set_xlabel("X", fontsize=6)
    inset.set_ylabel("Y", fontsize=6)
    inset.tick_params(labelsize=5)
    for a in ax:
        a.set_xlabel("|I_NVLED| (uA)")
        a.set_ylabel("P (nW)")
        a.grid(alpha=0.3)
    fig.suptitle(f"Current vs optical power by position on the array -- {lbl}",
                 fontsize=10)
    fig.tight_layout()
    files.append(out_dir / f"fastscan_IP_position{tag}.png")
    fig.savefig(files[-1], dpi=200)
    plt.close(fig)
    return files


def plot_fastscan_remeasure(fs_dir, out, plt):
    """EQE of the first N fast-scan pixels, measured at the start and again at the end.

    measure_iqe repeats the first ``fastscan_remeasure_n`` pixels at the end of each
    bias pass (remeasure_bitval=*.csv). Per bias: the two point clouds at their median
    times, t = 0 at the first, and one least-squares line through both clouds. No
    other analysis uses these numbers yet. Returns the slopes, or None without data.
    """
    files = sorted(Path(fs_dir).glob("*remeasure_bitval=*.csv"))
    if not files:
        return None
    rem = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    fs = _load_aggregated(fs_dir)
    keys = ["X", "Y", "NVLED_V"]
    both = rem.merge(fs, on=keys, suffixes=("_re", "_fs"))
    biases = sorted(both["NVLED_V"].round(4).unique())
    fig, axes = plt.subplots(1, len(biases), figsize=(5.5 * len(biases), 4.6),
                             squeeze=False)
    res = []
    for ax, v in zip(axes[0], biases):
        b = both[both["NVLED_V"].round(4) == v]
        p_all = fs.loc[fs["NVLED_V"].round(4) == v, "PM400_POWER_MEAN"].abs()
        thr = DEAD_THRESHOLD_FRAC * float(np.nanpercentile(p_all, FASTSCAN_ALIVE_PCTL))
        b = b[b["PM400_POWER_MEAN_fs"].abs() >= thr]          # alive on the first pass
        if b.empty:
            continue
        t1 = float(b["TIME_START_fs"].median())
        t2 = (float(b["TIME_START_re"].median()) - t1) / 60.0
        e1 = 100 * compute_eqe(b["PM400_POWER_MEAN_fs"].abs(), b["NVLED_CURR_MEAN_fs"].abs(),
                               WAVELENGTH_NM)
        e2 = 100 * compute_eqe(b["PM400_POWER_MEAN_re"].abs(), b["NVLED_CURR_MEAN_re"].abs(),
                               WAVELENGTH_NM)
        x = np.r_[np.zeros(len(e1)), np.full(len(e2), t2)]
        y = np.r_[np.asarray(e1, float), np.asarray(e2, float)]
        ok = np.isfinite(y)
        slope, icpt = np.polyfit(x[ok], y[ok], 1)
        ax.scatter(x[:len(e1)], e1, s=10, alpha=0.4, color="tab:blue", label="first pass")
        ax.scatter(x[len(e1):], e2, s=10, alpha=0.4, color="tab:orange", label="re-measured")
        xx = np.array([0.0, t2])
        rel = 100 * slope / icpt if icpt else float("nan")
        ax.plot(xx, icpt + slope * xx, color="k", lw=1.5,
                label=f"fit: {slope:+.3g} %-pt/min ({rel:+.2f} % of start / min)")
        ax.set_xlabel("time since the first pass (min)")
        ax.set_ylabel("EQE (%)")
        ax.set_title(f"{v:+.3f} V: {len(e1)} pixels, re-measured {t2:.1f} min later",
                     fontsize=9)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
        res.append(dict(v=v, n=len(e1), dt_min=t2, slope_pctpt_per_min=slope,
                        slope_rel_pct_per_min=rel))
    fig.suptitle("Fast-scan drift: the first pixels measured again at the end of the pass\n"
                 "EQE = P/I of one fast-scan burst; pixels alive on the first pass only",
                 fontsize=10)
    fig.tight_layout()
    fig.savefig(out, dpi=200)
    plt.close(fig)
    return res


# How the peak-IQE proxy turns a fast-scan EQE into a peak IQE. The factor is a swept
# pixel's fitted peak IQE over its own fast-scan EQE; the three differ in WHICH swept
# pixels set it for a given fast-scan pixel. All three are drawn side by side.
PEAK_PROXY_MODES = {
    "median": (None, "one median over all swept pixels"),
    "nearest": (1, "the nearest swept pixel"),
    "nearest8": (8, "median of the 8 nearest swept pixels"),
}
PEAK_PROXY_DEFAULT = "nearest8"     # the headline map and its stats
# The anomaly report reads the ONE-median map: there the factor is a constant, so every
# region it flags is in the fast-scan data. On a locally scaled map it also flags the
# calibration -- on 261016-2252 regions 1-4 % low in raw EQE read 6-17 points low.
PEAK_PROXY_ANOMALY = "median"


def fastscan_peak_iqe_pixels(fs, peak_iqe_ref, mode=PEAK_PROXY_DEFAULT):
    """Per-pixel peak-IQE proxy from the fast scan (X, Y, iqe%%), or (None, J) if the
    fast-scan bias is off the EQE plateau (so EQE≠peak and the proxy is invalid).

    ``peak_iqe_ref`` is either a number -- the old calibration, the map's median set to it
    -- or a pandas Series of clean per-pixel peak IQE indexed by (X, Y), the swept single
    pixels. Then the map is scaled by the median over those pixels of peak IQE / fast-scan
    EQE, so it agrees with the per-pixel fits rather than with the joint fit (261016-2252:
    the old scaling put the map's median at the joint 80.3 % against a 75.9 % per-pixel
    median).
    """
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
    if isinstance(peak_iqe_ref, pd.Series):
        # Scaled by the NEAREST swept pixel's peak IQE / fast-scan EQE, pixel by pixel,
        # not by one array-wide median of that ratio.
        tg = pd.DataFrame({"X": [i[0] for i in peak_iqe_ref.index],
                           "Y": [i[1] for i in peak_iqe_ref.index],
                           "val": peak_iqe_ref.values})
        scale, n = nearest_scale(fs, tg, "val", PEAK_PROXY_MODES[mode][0])
        if scale is None:
            return None, J
    else:
        med = float(np.nanmedian(eqe))
        if not np.isfinite(med) or med <= 0 or not np.isfinite(peak_iqe_ref):
            return None, J
        scale = peak_iqe_ref / med
    d = pd.DataFrame(
        {
            "X": fs["X"].values,
            "Y": fs["Y"].values,
            "iqe": eqe * scale * 100.0,
        }
    )
    return d[np.isfinite(d["iqe"])].reset_index(drop=True), J


def plot_fastscan_peak_iqe(fs, peak_iqe_ref, out, plt):
    """Full-coverage per-pixel PEAK-IQE proxy raster from the fast scan (calibrated to the sweep).
    A full-coverage estimate, not a per-pixel fit — see fastscan_peak_iqe_pixels. Returns stats or
    None (off the plateau)."""
    modes = (list(PEAK_PROXY_MODES) if isinstance(peak_iqe_ref, pd.Series)
             else [PEAK_PROXY_DEFAULT])          # a single number has no neighbours
    maps = {m: fastscan_peak_iqe_pixels(fs, peak_iqe_ref, m) for m in modes}
    d, J = maps[PEAK_PROXY_DEFAULT]
    if d is None:
        return None
    allv = np.concatenate([dm["iqe"].values for dm, _ in maps.values() if dm is not None])
    lo, hi = np.nanpercentile(allv, [2, 98])
    fig, ax = plt.subplots(1, len(modes), figsize=(7 * len(modes), 6.2), squeeze=False)
    for a, m in zip(ax[0], modes):
        dm, _ = maps[m]
        if dm is None:
            a.set_axis_off(); continue
        X, Y, V = dm["X"].to_numpy(int), dm["Y"].to_numpy(int), dm["iqe"].to_numpy(float)
        G = np.full((Y.max() - Y.min() + 1, X.max() - X.min() + 1), np.nan)
        G[Y - Y.min(), X - X.min()] = V
        cm = plt.get_cmap("viridis").copy(); cm.set_bad("white")
        im = a.imshow(np.ma.masked_invalid(G), cmap=cm, vmin=lo, vmax=hi, origin="upper",
                      interpolation="nearest",
                      extent=(X.min() - .5, X.max() + .5, Y.max() + .5, Y.min() - .5))
        plt.colorbar(im, ax=a, fraction=0.046, label="peak IQE est. (%)")
        a.set_title(f"factor = {PEAK_PROXY_MODES[m][1]}\nmedian {np.median(V):.1f} %, "
                    f"{100 * np.mean(V > 90):.1f} % of pixels above 90 %", fontsize=9)
        a.set_xlabel("X"); a.set_ylabel("Y")
    fig.suptitle(f"Peak-IQE proxy from the fast scan @ J≈{J:.0f} A/cm²: fast-scan EQE x "
                 f"(fitted peak IQE / fast-scan EQE) of swept pixels. White = dead.",
                 fontsize=10)
    fig.tight_layout()
    fig.savefig(out, dpi=200)
    plt.close(fig)
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

    # ---- 2b. lines that are not dim but DEAD-HEAVY ----
    # _lines() judges a line by its median, so it cannot see a row that is 9 % dead and
    # otherwise healthy. Run 261015-0015 had five: rows 127/255/256/384/511 held 211 of
    # the array's 715 dead pixels at 42x the rate of any other row, with normal medians
    # and clean immediate neighbours. The median test flagged one line in the whole
    # array and none of these. Counting the holes is the orthogonal test, and it is the
    # one that sees a row driver.
    def _dead_lines(axis: int) -> dict[int, int]:
        n = np.isnan(G).sum(axis=1 - axis)
        typical = max(1.0, float(np.median(n)))
        return {int(p): int(n[p]) for p in np.where(n >= max(10, 10 * typical))[0]}

    dead_rows = {k + ylo: v for k, v in _dead_lines(0).items()}
    dead_cols = {k + xlo: v for k, v in _dead_lines(1).items()}

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
        noise=noise, blobs=blobs, bad_cols=bad_cols, bad_rows=bad_rows, grads=grads,
        dead_rows=dead_rows, dead_cols=dead_cols
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
    fs, fs_extra = split_fastscan_biases(fs, cfg)
    plt = _plt()
    out = run_dir / "REPORT_assets"
    out.mkdir(exist_ok=True)
    fs_biases = []
    if fs_extra:
        log(f"  fast scan at {1 + len(fs_extra)} biases: classifying on "
            f"{float(fs['NVLED_V'].iloc[0]):+.3f} V, the others compared against it")
        try:
            fs_biases = fastscan_other_biases(fs, fs_extra, out / "fastscan_biases.png", plt)
        except Exception as e:
            log(f"  (skipped the per-bias fast-scan comparison: {e})")
    cls = classify_fastscan(fs)
    plot_fastscan_overview(fs, cls, out / "fastscan_overview.png", plt)
    for fsv in [fs, *fs_extra.values()]:
        try:
            tag = "" if not fs_extra else f"_v{float(fsv['NVLED_V'].iloc[0]):+.3f}"
            plot_fastscan_ip(fsv, out, plt, tag)
        except Exception as e:
            log(f"  (skipped the current-vs-power plots: {e})")
    try:
        drift = plot_fastscan_remeasure(fs_dir, out / "fastscan_remeasure.png", plt)
        for d in drift or []:
            log(f"  fast-scan re-measure @ {d['v']:+.3f} V: {d['n']} pixels, "
                f"{d['dt_min']:.1f} min apart, {d['slope_rel_pct_per_min']:+.3f} %/min")
    except Exception as e:
        log(f"  (skipped the fast-scan re-measure plot: {e})")
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
    first_sweep = None
    if sweep_root.exists():
        for grp in sorted(p for p in sweep_root.iterdir() if p.is_dir()):
            sweep = _load_aggregated(grp)
            if sweep is None:
                continue
            if ratio is None:
                ratio = plot_fastscan_vs_sweep(
                    fs, sweep, out / "fastscan_vs_sweep.png", plt
                )
                first_sweep = sweep
            s = analyse_sweep(grp, f"NVLEDSweep/{grp.name}", log)
            if s:
                s["label"] = f"NVLEDSweep/{grp.name}"
                summaries.append(s)
    # Power, current and IQE maps at EVERY fast-scan bias. eta and the sweep's EQE come
    # from the first sweep whose fit succeeded; without one the IQE panel is left out.
    fit_s = next((s for s in summaries if s.get("fit_ok") and s.get("median_fit")), None)
    # eta from the clean single-pixel fits, not the joint fit, so the IQE maps agree with
    # the per-pixel numbers the report leads with. The joint eta is the fallback.
    eta = None
    if fit_s:
        eta = fit_s.get("eta_pixels")
        if eta is None or not np.isfinite(eta):
            eta = fit_s["median_fit"].get("eta_LEE")
    enr = None
    if fit_s:
        ep = iqe_dir(run_dir / fit_s["label"]) / "enriched_points.csv"
        enr = pd.read_csv(ep) if ep.exists() else None
    fs_maps = []
    try:
        v0 = float(fs["NVLED_V"].astype(float).round(4).iloc[0]) if "NVLED_V" in fs else 0.0
        fs_maps = fastscan_bias_maps({v0: fs, **fs_extra}, eta, enr, out, plt)
        for m in fs_maps:
            log(f"  fast-scan maps {m['v']:+.3f} V: {m['alive']}/{m['n']} alive, "
                f"J~{m['J']:.1f} A/cm2, IQE median {m['iqe']:.0f}% (k={m['k']:.3f})")
    except Exception as e:
        log(f"  (skipped the per-bias fast-scan maps: {e})")
    if first_sweep is not None:
        try:
            v0 = float(fs["NVLED_V"].astype(float).round(4).iloc[0]) if "NVLED_V" in fs else 0.0
            cmp_ = fastscan_vs_sweep_compare(
                {v0: fs, **fs_extra}, first_sweep, eta,
                roi_of(cfg, fs["X"], fs["Y"]), out, plt, log)
            for c in cmp_ or []:
                log(f"  fast scan vs sweep, {c['kind']} @ {c['v']:+.3f} V: {c['n']} targets, "
                    f"nearest sweep bias {c['dv'] * 1e3:+.0f} mV away")
        except Exception as e:
            log(f"  (skipped the fast-scan vs sweep comparison: {e})")
    # Full-coverage peak-IQE proxy from the fast scan, calibrated to the sweep's pooled peak IQE.
    # Only produced when the fast-scan bias is on the EQE plateau (J≈8–30, i.e. NVLED ≈ −1.35 V);
    # skipped for a droop-side fast scan (old −2.6 V), where EQE ≠ peak.
    fs_peak = None
    # Calibrate against the swept single pixels' own clean fits, not the joint fit: the
    # map then agrees with the per-pixel median the report leads with. The joint fit's
    # peak IQE is only the fallback when no per-pixel fit is clean.
    peak_ref = None
    if fit_s:
        fp = iqe_dir(run_dir / fit_s["label"]) / "abc_fit_per_pixel.csv"
        if fp.exists():
            pf = pd.read_csv(fp)
            if {"clean_fit", "peak_IQE"} <= set(pf.columns):
                pf = pf[(pf["clean_fit"] == True) & pf["peak_IQE"].notna()]
                if "POOL_K" in pf.columns:
                    pf = pf[pf["POOL_K"] <= 1]
                if len(pf) >= 4:
                    peak_ref = pd.Series(pf["peak_IQE"].values,
                                         index=pd.MultiIndex.from_arrays([pf["X"], pf["Y"]]))
    if peak_ref is None:
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
                    f"median {fs_peak['median']:.0f}% (calibrated to "
                    + (f"{len(peak_ref)} swept pixels' own fits)" if isinstance(peak_ref, pd.Series)
                       else f"the joint fit, {100 * peak_ref:.0f}%)")
                )
                # General anomaly report on the full-coverage proxy: any-shape blobs (detect-all +
                # confidence), driver rows/columns per quadrant, and per-quadrant gradients.
                d, _ = fastscan_peak_iqe_pixels(fs, peak_ref, PEAK_PROXY_ANOMALY)
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
                    for lbl, dd in (("rows", r.get("dead_rows", {})),
                                    ("cols", r.get("dead_cols", {}))):
                        if dd:
                            log(
                                f"    dead-heavy {lbl}: {len(dd)} holding "
                                f"{sum(dd.values())} dead pixels — "
                                + ", ".join(f"{p}:{n}" for p, n in
                                            sorted(dd.items(), key=lambda kv: -kv[1])[:8])
                                + " (normal medians; a driver signature, not dimness)"
                            )
            else:
                log(
                    "  (fast scan is off the EQE plateau — no peak-IQE proxy; set NVLED fast-scan "
                    "bias to ~-1.35 V / J≈25 to enable it)"
                )
        except Exception as e:
            log(f"  (skipped fast-scan peak-IQE map: {e})")
    write_report(
        run_dir, cfg, cls, ratio, summaries, log, fs_eff=fs_eff, fs_spatial=fs_spatial,
        fs_biases=fs_biases, fs_v0=float(fs["NVLED_V"].iloc[0]) if "NVLED_V" in fs else None,
        fs_maps=fs_maps,
    )


def main():
    global PIXEL_AREA_UM2, ACTIVE_THICKNESS_NM, WAVELENGTH_NM, RUN_TRUNCATION, RUN_DECIMATION
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("path", help="run directory, or a results root with --all")
    ap.add_argument("--all", action="store_true", help="process every run under path")
    ap.add_argument("--area", type=float, default=PIXEL_AREA_UM2)
    ap.add_argument("--thickness", type=float, default=ACTIVE_THICKNESS_NM)
    ap.add_argument("--wavelength", type=float, default=WAVELENGTH_NM)
    ap.add_argument("--truncation", action="store_true",
                    help="run the end-point truncation check (slow, off by default)")
    ap.add_argument("--decimation", action="store_true",
                    help="refit on every k-th bias, ends kept: could fewer biases have "
                         "given the same peak IQE? (slow, off by default)")
    args = ap.parse_args()
    RUN_TRUNCATION, RUN_DECIMATION = args.truncation, args.decimation

    PIXEL_AREA_UM2, ACTIVE_THICKNESS_NM, WAVELENGTH_NM = (
        args.area,
        args.thickness,
        args.wavelength,
    )

    root = Path(args.path)
    if not root.exists():
        print(f"ERROR: {root} not found")
        sys.exit(1)

    # The Windows console is cp1252 and raises UnicodeEncodeError on the maths symbols
    # this script prints. That crash was being caught by a surrounding try/except and
    # reported as "skipped fast-scan peak-IQE map" -- the map had in fact been written,
    # and it was the log line that failed. Make output lossy instead of fatal.
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass

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
