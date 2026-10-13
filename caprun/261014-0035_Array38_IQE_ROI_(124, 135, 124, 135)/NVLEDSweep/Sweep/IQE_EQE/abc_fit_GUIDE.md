# ABC per-pixel fit — how to read it

## What beta (β) is

Carrier recombination in the LED follows the **ABC model**: the total recombination
rate at carrier density n is

    R(n) = A·n  +  B·n²  +  C·n³

- **A** — SRH (defect/trap) recombination: non-radiative, dominates at LOW current.
- **B** — radiative recombination: this is the term that MAKES LIGHT.
- **C** — Auger recombination: non-radiative, dominates at HIGH current (droop).

The internal quantum efficiency is the light-making fraction:  **IQE = B·n² / R(n)**.

**β = B / √(A·C)** collapses all three into one dimensionless *quality factor*: how strong
the radiative term is versus the geometric mean of the two loss channels. It sets the peak
directly, with no other knobs:

    peak IQE = β / (β + 2)        reached at  J = J0·(2 + β)

| β | 2 | 6 | 18 | 38 | 198 |
|---|---|---|----|----|-----|
| peak IQE | 50% | 75% | 90% | 95% | 99% |

So **to hit 90%+ peak IQE you need β ≳ 18**, and to *measure* β you must sample the rising
(low-current, SRH-dominated) side of the efficiency curve.

## Why fits get flagged (and why a '95%' can be fake)

If the sweep never reaches low enough current, the fit can't see the SRH (A) term, so it
drives A→0 and β runs away to huge values — producing a **fake ~95–100% peak that sits
*below* the measured data**. Those are flagged `degenerate_beta>20`. A fit counts as
**clean** only when the peak is actually inside the data, R² ≥ 0.7, and 0.05 ≤ β ≤ 20.

## Columns in abc_fit_per_pixel.csv

- `beta`, `J0`, `eta_LEE` — fit parameters (β as above; J0 current scale; eta_LEE the
  uncalibrated collection×extraction scale, meaningless in absolute terms).
- `peak_IQE` — the trustworthy peak IQE, **filled only for clean fits** (else blank).
- `peak_IQE_raw` — peak IQE of the fit *regardless* of trust (so you can see the runaways).
- `peak_captured` — was the fitted peak inside the measured J window?
- `clean_fit` — TRUE only if trustworthy (use this to filter).
- `flag` — plain-language reason a pixel is clean or rejected.
- `R2`, `J_at_peak`, `J_min`, `J_max` — fit quality and the J range used.

## This run

- **1/1 pixels clean (100%)**, median clean peak IQE **69%**.