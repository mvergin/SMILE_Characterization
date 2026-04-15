"""IQE / EQE calculation from NVLED sweep data.

Reads the aggregated CSV(s) produced by auto_measure.py (or the GUI)
after an NVLED voltage sweep, computes EQE and fits the ABC
recombination model to extract IQE, light-extraction efficiency, and
additional LED performance metrics.

Two current sources are analysed independently:
  - VLED current  (channel A)
  - NVLED current (channel B)
For good pixels they should agree; deviations flag leakage or shunting.

Usage:

    uv run python calc_iqe_eqe.py <sample_dir>

where <sample_dir> is the top-level folder produced by auto_measure.py
(contains NVLEDSweep/Within1STD/ and NVLEDSweep/Outside1STD/).

Or point directly at one sweep folder:

    uv run python calc_iqe_eqe.py <sweep_dir>
"""

from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

import numpy as np

# =====================================================================
#  CONFIGURATION
# =====================================================================

# ── LED geometry ─────────────────────────────────────────────────────
PIXEL_AREA_UM2 = 100.0             # 10 x 10 um^2
ACTIVE_THICKNESS_NM = 10.0         # total active region thickness (sum of QWs)

# ── Emission wavelength ─────────────────────────────────────────────
WAVELENGTH_NM = 450.0              # for photon energy calculation

# ── ABC fit ─────────────────────────────────────────────────────────
# Initial guesses for [A, B, C, eta_LEE]
ABC_INIT = [1e7, 1e-11, 1e-30, 0.05]
# Bounds: ([A_lo, B_lo, C_lo, eta_lo], [A_hi, B_hi, C_hi, eta_hi])
ABC_BOUNDS = ([1e4, 1e-15, 1e-40, 1e-4], [1e10, 1e-7, 1e-20, 1.0])

# ── Minimum data points per pixel for fitting ───────────────────────
MIN_POINTS_FOR_FIT = 5


# =====================================================================
#  Physical constants
# =====================================================================

_Q = 1.602176634e-19        # elementary charge (C)
_H = 6.62607015e-34         # Planck constant (J s)
_C = 2.99792458e8           # speed of light (m/s)
_K_B = 1.380649e-23         # Boltzmann constant (J/K)


def photon_energy_J(wavelength_nm):
    """Photon energy in Joules."""
    return _H * _C / (wavelength_nm * 1e-9)


# =====================================================================
#  Core calculations
# =====================================================================

def compute_eqe(optical_power_W, current_A, wavelength_nm):
    """EQE = (P / E_photon) / (I / q).

    Returns NaN where current is zero or negative.
    """
    E_ph = photon_energy_J(wavelength_nm)
    with np.errstate(divide="ignore", invalid="ignore"):
        eqe = (np.abs(optical_power_W) / E_ph) / (np.abs(current_A) / _Q)
    eqe[~np.isfinite(eqe)] = np.nan
    # Cap at 1.0 — EQE > 100% is unphysical (measurement noise)
    eqe = np.clip(eqe, 0, 1.0)
    return eqe


def compute_current_density(current_A, area_um2):
    """J in A/cm^2."""
    area_cm2 = area_um2 * 1e-8  # um^2 -> cm^2
    return np.abs(current_A) / area_cm2


def _abc_eqe_model(J, A, B, C, eta_LEE, d_cm):
    """Predicted EQE from ABC model at current density J (A/cm^2).

    For each J, solve  J = q * d * (A*n + B*n^2 + C*n^3)  for n,
    then  EQE = eta_LEE * B*n^2 / (A*n + B*n^2 + C*n^3).
    """
    eqe = np.zeros_like(J, dtype=float)
    for i, j_val in enumerate(J):
        if j_val <= 0 or not np.isfinite(j_val):
            eqe[i] = np.nan
            continue
        # Solve cubic: C*n^3 + B*n^2 + A*n - J/(q*d) = 0
        rhs = j_val / (_Q * d_cm)
        coeffs = [C, B, A, -rhs]
        roots = np.roots(coeffs)
        # Take the positive real root
        real_pos = [r.real for r in roots if abs(r.imag) < 1e-6 * abs(r.real + 1e-30) and r.real > 0]
        if not real_pos:
            eqe[i] = np.nan
            continue
        n = min(real_pos)  # smallest positive root
        total_R = A * n + B * n**2 + C * n**3
        if total_R <= 0:
            eqe[i] = np.nan
            continue
        eqe[i] = eta_LEE * B * n**2 / total_R
    return eqe


def fit_abc_model(J, eqe_measured, d_cm, init=None, bounds=None):
    """Fit the ABC + eta_LEE model to measured EQE(J) data.

    Returns dict with fitted parameters, IQE curve, etc., or None on failure.
    """
    from scipy.optimize import curve_fit

    mask = np.isfinite(J) & np.isfinite(eqe_measured) & (J > 0) & (eqe_measured > 0)
    J_fit = J[mask]
    eqe_fit = eqe_measured[mask]

    if len(J_fit) < MIN_POINTS_FOR_FIT:
        return None

    if init is None:
        init = list(ABC_INIT)
    if bounds is None:
        bounds = ABC_BOUNDS

    def model(j, A, B, C, eta_LEE):
        return _abc_eqe_model(j, A, B, C, eta_LEE, d_cm)

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            popt, pcov = curve_fit(
                model, J_fit, eqe_fit,
                p0=init, bounds=bounds,
                maxfev=20000,
                method="trf",
            )
    except Exception:
        return None

    A, B, C, eta_LEE = popt
    perr = np.sqrt(np.diag(pcov)) if pcov is not None else [np.nan] * 4

    # Compute IQE = B*n^2 / (A*n + B*n^2 + C*n^3) at each J
    eqe_model = _abc_eqe_model(J_fit, A, B, C, eta_LEE, d_cm)
    with np.errstate(divide="ignore", invalid="ignore"):
        iqe_fit = np.where(eta_LEE > 0, eqe_model / eta_LEE, np.nan)

    # Peak IQE and corresponding J
    valid = np.isfinite(iqe_fit)
    if valid.any():
        peak_idx = np.argmax(iqe_fit[valid])
        peak_iqe = iqe_fit[valid][peak_idx]
        j_at_peak = J_fit[valid][peak_idx]
    else:
        peak_iqe = np.nan
        j_at_peak = np.nan

    # R-squared
    ss_res = np.nansum((eqe_fit - eqe_model) ** 2)
    ss_tot = np.nansum((eqe_fit - np.nanmean(eqe_fit)) ** 2)
    r_squared = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan

    return {
        "A": A, "B": B, "C": C, "eta_LEE": eta_LEE,
        "A_err": perr[0], "B_err": perr[1], "C_err": perr[2], "eta_LEE_err": perr[3],
        "peak_IQE": peak_iqe,
        "J_at_peak_IQE": j_at_peak,
        "R_squared": r_squared,
        "J_fit": J_fit,
        "eqe_fit": eqe_fit,
        "eqe_model": eqe_model,
        "iqe_fit": iqe_fit,
    }


# =====================================================================
#  Additional LED metrics
# =====================================================================

def compute_wall_plug_efficiency(optical_power_W, current_A, voltage_V):
    """WPE = P_optical / P_electrical = P_optical / (I * V)."""
    with np.errstate(divide="ignore", invalid="ignore"):
        wpe = np.abs(optical_power_W) / (np.abs(current_A) * np.abs(voltage_V))
    wpe[~np.isfinite(wpe)] = np.nan
    return np.clip(wpe, 0, 1.0)


def compute_responsivity(optical_power_W, current_A):
    """Slope efficiency / responsivity = dP/dI (W/A).

    Computed as finite differences.
    """
    sort_idx = np.argsort(np.abs(current_A))
    I_sorted = np.abs(current_A[sort_idx])
    P_sorted = np.abs(optical_power_W[sort_idx])
    dP = np.diff(P_sorted)
    dI = np.diff(I_sorted)
    with np.errstate(divide="ignore", invalid="ignore"):
        resp = dP / dI
    resp[~np.isfinite(resp)] = np.nan
    # Map back: assign each diff value to the midpoint
    I_mid = (I_sorted[:-1] + I_sorted[1:]) / 2
    return I_mid, resp


def estimate_ideality_factor(voltage_V, current_A, T_K=300.0):
    """Estimate diode ideality factor n from I-V data.

    Uses linear regression on ln(I) vs V in the exponential region.
    Returns (ideality_factor, series_resistance_estimate_ohm) or (NaN, NaN).
    """
    mask = (np.abs(current_A) > 0) & np.isfinite(voltage_V) & np.isfinite(current_A)
    V = np.abs(voltage_V[mask])
    I = np.abs(current_A[mask])
    if len(V) < 3:
        return np.nan, np.nan

    ln_I = np.log(I)
    # Use middle portion (avoid very low / very high current)
    q10 = np.percentile(ln_I, 10)
    q90 = np.percentile(ln_I, 90)
    mid_mask = (ln_I >= q10) & (ln_I <= q90)
    if mid_mask.sum() < 3:
        mid_mask = np.ones_like(ln_I, dtype=bool)

    V_mid = V[mid_mask]
    ln_I_mid = ln_I[mid_mask]

    try:
        coeffs = np.polyfit(V_mid, ln_I_mid, 1)
        slope = coeffs[0]  # d(ln I)/dV = q / (n * k_B * T)
        if slope > 0:
            n_ideality = _Q / (slope * _K_B * T_K)
        else:
            n_ideality = np.nan
    except Exception:
        n_ideality = np.nan

    # Series resistance: at high current, V_diode saturates, excess V = I*R_s
    # Simple estimate from the two highest current points
    try:
        top2_idx = np.argsort(I)[-2:]
        if len(top2_idx) == 2:
            dV = V[top2_idx[1]] - V[top2_idx[0]]
            dI = I[top2_idx[1]] - I[top2_idx[0]]
            R_s = dV / dI if dI > 0 else np.nan
        else:
            R_s = np.nan
    except Exception:
        R_s = np.nan

    return n_ideality, R_s


def compute_droop_onset(J, eqe):
    """Find the current density at which EQE peaks (droop onset).

    Returns (J_droop, EQE_peak) or (NaN, NaN).
    """
    mask = np.isfinite(J) & np.isfinite(eqe) & (J > 0)
    if mask.sum() < 3:
        return np.nan, np.nan
    J_v = J[mask]
    eqe_v = eqe[mask]
    idx = np.argmax(eqe_v)
    return float(J_v[idx]), float(eqe_v[idx])


# =====================================================================
#  Plotting
# =====================================================================

def _setup_matplotlib():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def plot_eqe_scatter(df, J_col, eqe_col, title, out_path, plt):
    """Scatter plot of EQE vs J for all pixels, colored by pixel."""
    fig, ax = plt.subplots(figsize=(8, 6))
    pixels = df.groupby(["X", "Y"])
    for (x, y), grp in pixels:
        ax.scatter(grp[J_col], grp[eqe_col] * 100, s=10, alpha=0.5)
    ax.set_xlabel("Current density J (A/cm$^2$)")
    ax.set_ylabel("EQE (%)")
    ax.set_title(title)
    ax.set_xscale("log")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(str(out_path), dpi=200)
    plt.close(fig)


def plot_eqe_median_with_fit(J_median, eqe_median, fit_result, title, out_path, plt):
    """Median EQE(J) with ABC fit overlay."""
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(J_median, eqe_median * 100, s=30, color="tab:blue", label="Median EQE", zorder=3)
    if fit_result is not None:
        ax.plot(fit_result["J_fit"], fit_result["eqe_model"] * 100,
                "r-", lw=2, label=f"ABC fit (R²={fit_result['R_squared']:.3f})", zorder=4)
    ax.set_xlabel("Current density J (A/cm$^2$)")
    ax.set_ylabel("EQE (%)")
    ax.set_title(title)
    ax.set_xscale("log")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(str(out_path), dpi=200)
    plt.close(fig)


def plot_iqe_vs_J(fit_result, title, out_path, plt):
    """IQE vs J from ABC fit."""
    if fit_result is None:
        return
    fig, ax = plt.subplots(figsize=(8, 6))
    mask = np.isfinite(fit_result["iqe_fit"])
    ax.plot(fit_result["J_fit"][mask], fit_result["iqe_fit"][mask] * 100,
            "g-o", lw=2, ms=4, label="IQE (ABC model)")
    ax.axhline(fit_result["peak_IQE"] * 100, color="gray", ls="--", lw=0.8,
               label=f"Peak IQE = {fit_result['peak_IQE']*100:.1f}%")
    ax.set_xlabel("Current density J (A/cm$^2$)")
    ax.set_ylabel("IQE (%)")
    ax.set_title(title)
    ax.set_xscale("log")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(str(out_path), dpi=200)
    plt.close(fig)


def plot_eqe_comparison(vled_eqe, nvled_eqe, J_vled, J_nvled, title, out_path, plt):
    """Side-by-side comparison of EQE computed from VLED vs NVLED current."""
    fig, ax = plt.subplots(figsize=(8, 6))
    mask_v = np.isfinite(vled_eqe) & np.isfinite(J_vled) & (J_vled > 0)
    mask_n = np.isfinite(nvled_eqe) & np.isfinite(J_nvled) & (J_nvled > 0)
    ax.scatter(J_vled[mask_v], vled_eqe[mask_v] * 100, s=8, alpha=0.4,
               color="tab:orange", label="EQE (VLED current)")
    ax.scatter(J_nvled[mask_n], nvled_eqe[mask_n] * 100, s=8, alpha=0.4,
               color="tab:green", label="EQE (NVLED current)")
    ax.set_xlabel("Current density J (A/cm$^2$)")
    ax.set_ylabel("EQE (%)")
    ax.set_title(title)
    ax.set_xscale("log")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(str(out_path), dpi=200)
    plt.close(fig)


def plot_wpe(J, wpe, title, out_path, plt):
    """Wall-plug efficiency vs J."""
    fig, ax = plt.subplots(figsize=(8, 6))
    mask = np.isfinite(J) & np.isfinite(wpe) & (J > 0)
    ax.scatter(J[mask], wpe[mask] * 100, s=8, alpha=0.5, color="tab:purple")
    ax.set_xlabel("Current density J (A/cm$^2$)")
    ax.set_ylabel("WPE (%)")
    ax.set_title(title)
    ax.set_xscale("log")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(str(out_path), dpi=200)
    plt.close(fig)


def plot_LI(current_A, power_W, title, out_path, plt):
    """L-I curve (optical power vs current)."""
    fig, ax = plt.subplots(figsize=(8, 6))
    mask = np.isfinite(current_A) & np.isfinite(power_W)
    ax.scatter(np.abs(current_A[mask]) * 1e3, np.abs(power_W[mask]) * 1e6,
               s=8, alpha=0.5, color="tab:blue")
    ax.set_xlabel("Current (mA)")
    ax.set_ylabel("Optical power (uW)")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(str(out_path), dpi=200)
    plt.close(fig)


# =====================================================================
#  Per-pixel and median analysis
# =====================================================================

def analyse_sweep_data(sweep_dir, label, log):
    """Full IQE/EQE analysis on one sweep sub-run.

    Looks for aggregated_bitval=*.csv in sweep_dir or sweep_dir/stats/.
    Falls back to raw CSV if aggregated files don't exist.
    """
    import pandas as pd

    sweep_dir = Path(sweep_dir)
    out_dir = sweep_dir / "IQE_EQE"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Find aggregated CSVs
    agg_files = sorted(sweep_dir.glob("aggregated_bitval=*.csv"))
    if not agg_files:
        # Maybe post-processing put them in the root
        agg_files = sorted(sweep_dir.glob("**/aggregated_bitval=*.csv"))
    if not agg_files:
        # Fall back: read raw CSV and aggregate ourselves
        raw_csvs = list((sweep_dir / "raw_data").glob("*.csv"))
        raw_csvs = [f for f in raw_csvs if "config" not in f.name.lower()]
        if not raw_csvs:
            log(f"  No data found in {sweep_dir}")
            return
        log(f"  No aggregated CSV found; reading raw CSV: {raw_csvs[0].name}")
        df_raw = pd.read_csv(raw_csvs[0])
        df_agg = _aggregate_raw(df_raw)
    else:
        frames = [pd.read_csv(f) for f in agg_files]
        df_agg = pd.concat(frames, ignore_index=True)
        log(f"  Loaded {len(agg_files)} aggregated file(s), {len(df_agg)} rows.")

    if df_agg.empty:
        log(f"  Empty data for {label}")
        return

    # Ensure required columns exist
    required = {"X", "Y", "BITVAL", "NVLED_V", "PM400_POWER_MEAN",
                "VLED_CURR_MEAN", "NVLED_CURR_MEAN"}
    missing = required - set(df_agg.columns)
    if missing:
        log(f"  Missing columns: {missing}")
        return

    area_um2 = PIXEL_AREA_UM2
    d_cm = ACTIVE_THICKNESS_NM * 1e-7  # nm -> cm
    wl = WAVELENGTH_NM

    plt = _setup_matplotlib()

    # ── Compute EQE and J for both current sources ───────────────────
    P = df_agg["PM400_POWER_MEAN"].values.astype(float)
    I_vled = df_agg["VLED_CURR_MEAN"].values.astype(float)
    I_nvled = df_agg["NVLED_CURR_MEAN"].values.astype(float)
    V_nvled = df_agg["NVLED_V"].values.astype(float)

    eqe_vled = compute_eqe(P, I_vled, wl)
    eqe_nvled = compute_eqe(P, I_nvled, wl)
    J_vled = compute_current_density(I_vled, area_um2)
    J_nvled = compute_current_density(I_nvled, area_um2)

    df_agg = df_agg.copy()
    df_agg["EQE_VLED"] = eqe_vled
    df_agg["EQE_NVLED"] = eqe_nvled
    df_agg["J_VLED_Acm2"] = J_vled
    df_agg["J_NVLED_Acm2"] = J_nvled

    # WPE (using VLED voltage = applied voltage from NVLED_V isn't right;
    # we'd need the forward voltage. Use VLED_V as proxy if available.)
    # For now approximate: WPE uses the LED forward voltage.
    # The VLED voltage is constant; the varying parameter is NVLED_V.
    # WPE = P_opt / (I * V_forward). We'll use VLED as the current source.
    # V_forward isn't directly available in aggregated data, so skip WPE
    # from aggregated. If raw data has it, it could be added.

    # ── Save per-pixel EQE table ─────────────────────────────────────
    eqe_path = out_dir / "eqe_per_pixel.csv"
    df_agg.to_csv(eqe_path, index=False)
    log(f"  Saved: {eqe_path.name}")

    # ── Scatter plots ────────────────────────────────────────────────
    plot_eqe_scatter(df_agg, "J_VLED_Acm2", "EQE_VLED",
                     f"EQE vs J (VLED) — {label}", out_dir / "eqe_vs_J_vled_scatter.png", plt)
    plot_eqe_scatter(df_agg, "J_NVLED_Acm2", "EQE_NVLED",
                     f"EQE vs J (NVLED) — {label}", out_dir / "eqe_vs_J_nvled_scatter.png", plt)

    # ── Comparison plot ──────────────────────────────────────────────
    plot_eqe_comparison(eqe_vled, eqe_nvled, J_vled, J_nvled,
                        f"EQE comparison VLED vs NVLED — {label}",
                        out_dir / "eqe_vled_vs_nvled_comparison.png", plt)

    # ── L-I curve ────────────────────────────────────────────────────
    plot_LI(I_vled, P, f"L-I curve (VLED) — {label}", out_dir / "LI_curve_vled.png", plt)
    plot_LI(I_nvled, P, f"L-I curve (NVLED) — {label}", out_dir / "LI_curve_nvled.png", plt)

    # ── Per-pixel ABC fits ───────────────────────────────────────────
    fit_rows = []
    for current_label, J_col, eqe_col in [
        ("VLED", "J_VLED_Acm2", "EQE_VLED"),
        ("NVLED", "J_NVLED_Acm2", "EQE_NVLED"),
    ]:
        pixel_fits = []
        for (x, y), grp in df_agg.groupby(["X", "Y"]):
            J_px = grp[J_col].values
            eqe_px = grp[eqe_col].values
            result = fit_abc_model(J_px, eqe_px, d_cm)

            # Ideality factor (using NVLED_V as the bias and current)
            if current_label == "NVLED":
                n_ideal, R_s = estimate_ideality_factor(
                    grp["NVLED_V"].values, grp["NVLED_CURR_MEAN"].values)
            else:
                # For VLED, voltage is constant — ideality not meaningful
                n_ideal, R_s = np.nan, np.nan

            # Droop onset
            J_droop, eqe_peak = compute_droop_onset(J_px, eqe_px)

            row = {
                "X": int(x), "Y": int(y),
                "CURRENT_SOURCE": current_label,
                "J_DROOP_Acm2": J_droop,
                "EQE_PEAK": eqe_peak,
                "IDEALITY_FACTOR": n_ideal,
                "R_SERIES_OHM": R_s,
            }
            if result is not None:
                row.update({
                    "A": result["A"], "B": result["B"], "C": result["C"],
                    "eta_LEE": result["eta_LEE"],
                    "peak_IQE": result["peak_IQE"],
                    "J_at_peak_IQE": result["J_at_peak_IQE"],
                    "R_squared": result["R_squared"],
                })
                pixel_fits.append(result)
            else:
                row.update({
                    "A": np.nan, "B": np.nan, "C": np.nan,
                    "eta_LEE": np.nan, "peak_IQE": np.nan,
                    "J_at_peak_IQE": np.nan, "R_squared": np.nan,
                })
            fit_rows.append(row)

        log(f"  ABC fits ({current_label}): {len(pixel_fits)}/{len(df_agg.groupby(['X','Y']))} pixels converged")

    # Save fit summary
    df_fits = pd.DataFrame(fit_rows)
    fits_path = out_dir / "abc_fit_summary.csv"
    df_fits.to_csv(fits_path, index=False)
    log(f"  Saved: {fits_path.name}")

    # ── Median EQE curve + fit ───────────────────────────────────────
    for current_label, J_col, eqe_col in [
        ("VLED", "J_VLED_Acm2", "EQE_VLED"),
        ("NVLED", "J_NVLED_Acm2", "EQE_NVLED"),
    ]:
        # Group by NVLED_V (same voltage step across pixels) -> median
        median_data = df_agg.groupby("NVLED_V").agg(
            J_median=(J_col, "median"),
            EQE_median=(eqe_col, "median"),
        ).dropna().sort_values("J_median").reset_index()

        J_med = median_data["J_median"].values
        eqe_med = median_data["EQE_median"].values

        fit_median = fit_abc_model(J_med, eqe_med, d_cm)

        suffix = current_label.lower()
        plot_eqe_median_with_fit(
            J_med, eqe_med, fit_median,
            f"Median EQE + ABC fit ({current_label}) — {label}",
            out_dir / f"eqe_median_abc_fit_{suffix}.png", plt,
        )
        plot_iqe_vs_J(
            fit_median,
            f"IQE vs J ({current_label}) — {label}",
            out_dir / f"iqe_vs_J_{suffix}.png", plt,
        )

        if fit_median is not None:
            log(f"  Median ABC fit ({current_label}): "
                f"A={fit_median['A']:.2e}, B={fit_median['B']:.2e}, "
                f"C={fit_median['C']:.2e}, eta_LEE={fit_median['eta_LEE']:.3f}, "
                f"peak_IQE={fit_median['peak_IQE']*100:.1f}%, "
                f"R²={fit_median['R_squared']:.4f}")

            # Save median fit params
            median_params_path = out_dir / f"abc_median_params_{suffix}.json"
            import json
            with open(median_params_path, "w") as f:
                json.dump({
                    "current_source": current_label,
                    "A": fit_median["A"],
                    "B": fit_median["B"],
                    "C": fit_median["C"],
                    "eta_LEE": fit_median["eta_LEE"],
                    "peak_IQE": fit_median["peak_IQE"],
                    "J_at_peak_IQE": fit_median["J_at_peak_IQE"],
                    "R_squared": fit_median["R_squared"],
                    "pixel_area_um2": area_um2,
                    "active_thickness_nm": ACTIVE_THICKNESS_NM,
                    "wavelength_nm": wl,
                }, f, indent=4)

    # ── IQE scatter: all pixels ──────────────────────────────────────
    _plot_iqe_scatter(df_fits, label, out_dir, plt)

    log(f"  Analysis complete for {label}. Output: {out_dir}")


def _plot_iqe_scatter(df_fits, label, out_dir, plt):
    """Scatter plot of peak IQE for all pixels, split by current source."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    for ax, src in zip(axes, ["VLED", "NVLED"]):
        sub = df_fits[df_fits["CURRENT_SOURCE"] == src].dropna(subset=["peak_IQE"])
        if sub.empty:
            ax.set_title(f"No IQE data ({src})")
            continue
        sc = ax.scatter(sub["X"], sub["Y"], c=sub["peak_IQE"] * 100,
                        s=12, cmap="viridis", alpha=0.7)
        plt.colorbar(sc, ax=ax, label="Peak IQE (%)")
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_title(f"Peak IQE map ({src}) — {label}")
        ax.set_aspect("equal")
    fig.tight_layout()
    fig.savefig(str(out_dir / "iqe_spatial_map.png"), dpi=200)
    plt.close(fig)


def _aggregate_raw(df_raw):
    """Aggregate raw CSV into per-(pixel, voltage) means.
    Mirrors smile_postprocess.post_process_data aggregation.
    """
    import pandas as pd

    grouped = df_raw.groupby(["X", "Y", "BITVAL", "NVLED_V"])
    rows = []
    for name, group in grouped:
        x, y, bitval, nvled_v = name
        pm = group[group["TYPE"] == "PM400"]["MEAS_VALUE"]
        vl = group[group["TYPE"] == "VLED"]["MEAS_VALUE"]
        nv = group[group["TYPE"] == "NVLED"]["MEAS_VALUE"]
        rows.append({
            "X": int(x), "Y": int(y), "BITVAL": int(bitval),
            "NVLED_V": round(nvled_v, 3),
            "PM400_POWER_MEAN": pm.mean() if not pm.empty else np.nan,
            "VLED_CURR_MEAN": vl.mean() if not vl.empty else np.nan,
            "NVLED_CURR_MEAN": nv.mean() if not nv.empty else np.nan,
        })
    return pd.DataFrame(rows)


# =====================================================================
#  Main
# =====================================================================

def main():
    global PIXEL_AREA_UM2, ACTIVE_THICKNESS_NM, WAVELENGTH_NM

    parser = argparse.ArgumentParser(
        description="Compute IQE/EQE from NVLED sweep measurement data."
    )
    parser.add_argument(
        "data_dir",
        help="Path to sample dir (with NVLEDSweep/) or a single sweep folder.",
    )
    parser.add_argument(
        "--area", type=float, default=PIXEL_AREA_UM2,
        help=f"Pixel area in um^2 (default: {PIXEL_AREA_UM2})",
    )
    parser.add_argument(
        "--thickness", type=float, default=ACTIVE_THICKNESS_NM,
        help=f"Active region thickness in nm (default: {ACTIVE_THICKNESS_NM})",
    )
    parser.add_argument(
        "--wavelength", type=float, default=WAVELENGTH_NM,
        help=f"Emission wavelength in nm (default: {WAVELENGTH_NM})",
    )
    args = parser.parse_args()

    PIXEL_AREA_UM2 = args.area
    ACTIVE_THICKNESS_NM = args.thickness
    WAVELENGTH_NM = args.wavelength

    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        print(f"ERROR: {data_dir} does not exist.")
        sys.exit(1)

    def log(msg):
        print(msg)

    # Detect directory structure
    sweep_root = data_dir / "NVLEDSweep"
    if sweep_root.exists():
        # Top-level sample dir from auto_measure.py
        for sub_name in ["Within1STD", "Outside1STD"]:
            sub_dir = sweep_root / sub_name
            if sub_dir.exists():
                log(f"\n{'='*60}")
                log(f"Analysing: {sub_name}")
                log(f"{'='*60}")
                analyse_sweep_data(sub_dir, sub_name, log)
    else:
        # Direct sweep folder
        log(f"\n{'='*60}")
        log(f"Analysing: {data_dir.name}")
        log(f"{'='*60}")
        analyse_sweep_data(data_dir, data_dir.name, log)

    log("\nDone.")


if __name__ == "__main__":
    main()
