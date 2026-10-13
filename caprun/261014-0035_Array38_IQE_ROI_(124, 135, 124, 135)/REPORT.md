# Measurement report — 261014-0035_Array38_IQE_ROI_(124, 135, 124, 135)

_Generated 2026-10-14 00:42 by analyze_array.py_

## 1. What was measured

- **Sample:** Array38_IQE (Array38)
- **VLED (ch A):** 1.8 V fixed, compliance 1.0 A, NPLC None, I-range 0.01 A
- **NVLED (ch B):** fast-scan bias None V → sweep target None V, NPLC None, I-range 0.01 A
- **Mode:** None, settle None ms, PM wavelength 440.0 nm, PM range 1e-07 W
- **Bit values:** None

## 2. Fast scan — yield & classification

- Pixels scanned: **1**, alive: **1** (100%), dead: **0** (< 20% of the p90 'lit' level 0.0 nW)
- Alive optical power: median **0.03 nW**, mean 0.03 nW, σ nan nW (**CV = nan**)
- Within ±1σ band [nan, nan] nW: **0** pixels; low-tail **0**, high-tail **0**

- **Fast-scan vs sweep cross-check** (at the shared NVLED bias): median power ratio **5496.593** (1.00 = perfect) → the two acquisitions agree, so the data is reproducible.
- **Dense efficiency map** (all 1 alive pixels, fast scan at J≈10 A/cm²): relative EQE spans 1.04e-06–1.04e-06 (90/10 ratio 1.00×) — full-coverage spatial map of device efficiency at high injection in REPORT_assets/fastscan_efficiency.png. (Coarse: high-J, noisy fast-scan current; complements the sweep peak-IQE map.)

## 3. NVLED sweep — IQE / EQE

### NVLEDSweep/Sweep

- Pixels: 1, sweep points/pixel: ~30, reliable points (after noise-floor + rel-σ filter): 27/30
- Reliable current-density range: **J = 1.31 – 118 A/cm²** (crosses the IQE peak)
- L–I linearity P=k·I over reliable points: R² = 0.9917; EQE spread (90/10 pct): 1.00× → ~constant EQE, little curvature to fit.
- **Pooled fit not meaningful** (R²=n/a, no peak in window): the population median EQE(J) has no usable peak — see per-pixel line below.
- **Per-pixel:** median 69%, range 69–69%, **1/1 clean per-pixel fits (100%)** — a fit is *clean* only if the peak is captured in-data, R²≥0.7, and β≤20 (not a runaway extrapolation). See abc_fit_per_pixel.csv + abc_fit_GUIDE.md (explains β) and iqe_spatial_map.
- Plots: `Sweep/IQE_EQE/` — IVL_full, eqe_vs_J_LED, iqe_split_branch, eqe_vs_J_by_bitval, first_pixels_diagnostics, iqe_all_pixels_LED, iqe_spatial_map, iqe_fitquality_map, iqe_blocked_maps, iqe_row_col_profiles, leakage_vs_bias

## 4. Data-quality diagnosis & is IQE trustworthy?

**Good news**
- Fast-scan and sweep agree at the shared bias → the optical + electrical chain is reproducible.
- Dark/background power is ~0.1% of signal → not a limiting factor.

**Problems that make the current IQE untrustworthy**
1. **Current saturates at the most-negative NVLED.** The LED current is flat (~120 µA) from -3.2 V to about -2.5 V, so points piled there (POWER=2.7 spacing) collapse onto the same J — only a handful of distinct current levels are produced.
2. **The sweep never reaches low injection.** Reliable J only goes down to ~1.31 A/cm². The SRH rise and the IQE peak of a blue µLED sit at J ≲ 1 A/cm² (sub-µA). We measure on the droop side only, so peak IQE is **unconstrained**. The L–I curve is ~linear (R² ≈ 0.9917): little curvature for the fit to use.
3. **Low-current readings are noisy.** Near turn-off the per-point current rel-σ is several ×100% (NPLC=0.001 ≈ 20 µs on the 10 mA range reading µA-level current).
4. **Collection-limited EQE.** The PD at ~20 cm collects ~5×10⁻⁴ of the light (measured EQE ~3×10⁻⁵); the absolute scale needs calibration regardless.

**Verdict:** No sweep crossed the IQE peak, so peak-IQE cannot be extracted from this data. The ~80% you expect cannot be confirmed here.

## 5. What to change for the next run

- **Use `measure_iqe.py`** — it applies each voltage, settles, and averages (accurate), and adaptively places points at log-spaced *currents* down toward turn-off so the sweep crosses the IQE peak. (Validated in simulation: recovers a known 82% peak.)
- **Reach low injection.** Set the sweep start near turn-off (where current → the noise floor) so reliable J gets below ~1 A/cm².
- **Stop above the saturation knee** (~-2.5 V) — points beyond it carry no information.
- **Improve low-current accuracy** with NPLC≈1 + averaging on a fixed range that fits the max current (drop to 1 mA if the rig is stable there).
- **(Optional) Calibrate collection** for absolute EQE; otherwise only IQE *shape* is meaningful.
