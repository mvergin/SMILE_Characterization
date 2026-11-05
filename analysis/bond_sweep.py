"""Series resistance of every swept pixel from its own I(V): no LED zero point needed.

    python analysis/bond_sweep.py

Per pixel of a sweep (Array38 261026-0813, Array42 261102-1703):
1. Anchor its LED: at its point nearest 1.5 uA (0.5-3 uA), driver and bond drop a few mV,
   so dV_i = V_tot - V_LED_SGA(I) is this pixel's own LED offset against the SGA_15 die.
   It removes the die-to-die zero point and the LED-to-LED turn-on spread.
2. At every point from 5 uA up to 70 % of what the driver can carry at that bias:
   V_rest = V_tot - V_driver(I) - (V_LED_SGA(I) + dV_i), the voltage on everything else.
3. Fit V_rest = c + R_s I. R_s is the series resistance; c should be ~0 after the anchor.
   With >= 4 points also V_rest = c + R I + k I^2: the curvature, as the change of dV/dI
   across the pixel's current range over R_s (> 0 the resistance grows with current).

Compared with the same day's fast-scan R_bond of the same pixels (bond_resistance.py).
Writes results/comparisons/bond_resistance/sweep/.
"""
from __future__ import annotations

import sys

import numpy as np
import pandas as pd

import analyze_array as aa
import bond_resistance as br
import pixel_classes as pc

SWEEPS = {"Array38": ("261026-0813_Array38_ROI_(0, 0, 511, 511)", "261026-0813_m1p40V"),
          "Array42": ("261102-1703_Array42_ROI_(0, 0, 511, 511)", "261102-1703_m1p40V")}
I_LO, I_HI_FRAC = 5e-6, 0.7


def per_pixel(arr, P):
    sw = pc.load_sweep(br.REPO / "results" / SWEEPS[arr][0])
    # a reference above 1 uA was taken with the pixel still lit (clean: 125-175 nA); it
    # shrinks I_LED at that bias, and with 3-5 points per pixel one such point fakes a
    # second population at 15-20 kOhm (Array42, 261102-1703: 41 216 REF_CONTAMINATED points)
    sw = sw[sw.I_ref.notna() & (sw.I_ref < 1e-6) & (sw.I_LED > 0)].copy()
    sw["vtot"] = br.VLED - sw.V
    sw["vled"] = br.v_led(sw.I_LED.to_numpy())
    # anchor: the point nearest 1.5 uA, within 0.5-3 uA
    a = sw[(sw.I_LED >= 0.5e-6) & (sw.I_LED <= 3e-6)].copy()
    a["dist"] = np.abs(np.log(a.I_LED / 1.5e-6))
    a = a.loc[a.groupby(["X", "Y"]).dist.idxmin()].set_index(["X", "Y"])
    dv = (a.vtot - a.vled).rename("dV_led")
    # the fit points
    cap = {v: float(br.f15(np.array([br.VLED - v]), -v, P)[0]) for v in sw.V.unique()}
    m = sw[(sw.I_LED >= I_LO) & (sw.I_LED <= I_HI_FRAC * sw.V.map(cap))].copy()
    m = m.join(dv, on=["X", "Y"], how="inner")
    vdrv = np.full(len(m), np.nan)
    for v, idx in m.groupby("V").groups.items():
        vdrv[m.index.get_indexer(idx)] = br.v_driver(m.loc[idx, "I_LED"].to_numpy(), -v, P)
    m["vrest"] = m.vtot - vdrv - m.vled - m.dV_led
    m = m[np.isfinite(m.vrest)]
    rows = []
    for (x, y), g in m.groupby(["X", "Y"]):
        if len(g) < 3:
            continue
        i, v = g.I_LED.to_numpy(), g.vrest.to_numpy()
        R, c = np.polyfit(i, v, 1)
        res = v - (c + R * i)
        row = dict(X=x, Y=y, quadrant=g.QUADRANT.iloc[0], n=len(g), I_min_uA=i.min() * 1e6,
                   I_max_uA=i.max() * 1e6, R_s_kOhm=R / 1e3, offset_mV=c * 1e3,
                   rms_mV=1e3 * float(np.sqrt(np.mean(res ** 2))), dV_led_mV=dv[(x, y)] * 1e3)
        if len(g) >= 4:
            k2, k1, _ = np.polyfit(i, v, 2)
            row["curvature"] = 2 * k2 * (i.max() - i.min()) / R if R else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def main() -> int:
    out = br.OUT / "sweep"
    out.mkdir(parents=True, exist_ok=True)
    plt = aa._plt()
    drv = {a: br.fit_driver(a) for a in SWEEPS}
    fig, ax = plt.subplots(2, 3, figsize=(18, 10))
    L = ["# Series resistance per swept pixel (bond_sweep.py)", ""]
    for j, arr in enumerate(SWEEPS):
        d = per_pixel(arr, drv[arr])
        fsr = pd.read_csv(br.OUT / "pixels" / f"{SWEEPS[arr][1]}.csv").set_index(["X", "Y"])
        d = d.join(fsr[["R_bond_kOhm", "primary"]], on=["X", "Y"])
        d.to_csv(out / f"{arr}_series_resistance.csv", index=False, float_format="%.5g")
        q = d.R_s_kOhm.quantile([.1, .25, .5, .75, .9])
        cur = d.curvature.dropna()
        L += [f"## {arr} ({SWEEPS[arr][0]})", "",
              f"- {len(d)} pixels with >= 3 points between 5 uA and 70 % of the driver's current "
              f"(median {d.n.median():.0f} points, {d.I_min_uA.median():.0f}-"
              f"{d.I_max_uA.median():.0f} uA)",
              f"- **R_s median {q[.5]:.2f} kOhm** (10-90 %: {q[.1]:.2f}-{q[.9]:.2f}, quartiles "
              f"{q[.25]:.2f}-{q[.75]:.2f})",
              f"- fit offset after the LED anchor: median {d.offset_mV.median():+.1f} mV "
              f"(10-90 % {d.offset_mV.quantile(.1):+.1f} to {d.offset_mV.quantile(.9):+.1f}); "
              f"residual rms median {d.rms_mV.median():.1f} mV",
              f"- the pixel's own LED offset against the SGA_15 die: median "
              f"{d.dV_led_mV.median():+.1f} mV (10-90 % {d.dV_led_mV.quantile(.1):+.1f} to "
              f"{d.dV_led_mV.quantile(.9):+.1f})",
              f"- curvature ({len(cur)} pixels with >= 4 points): median {cur.median():+.2f}, "
              f"{100 * (cur > 0).mean():.0f} % superlinear (dV/dI grows with current)",
              f"- against the same day's fast-scan R_bond: median of R_s - R_bond "
              f"{(d.R_s_kOhm - d.R_bond_kOhm).median():+.2f} kOhm, rank correlation "
              f"{d.R_s_kOhm.corr(d.R_bond_kOhm, method='spearman'):.3f}", ""]
        x = ax[j, 0]
        v = d.R_s_kOhm.clip(-5, 60)
        x.hist(v, bins=200, color="tab:blue")
        x.set_xlabel("R_s from the sweep (kOhm), clipped to -5..60")
        x.set_ylabel("pixels")
        x.set_yscale("log")
        x.set_title(f"{arr}: series resistance per swept pixel, median {q[.5]:.2f} kOhm",
                    fontsize=10)
        x.grid(alpha=0.3)
        x = ax[j, 1]
        ok = d.R_bond_kOhm.notna()
        lo, hi = np.nanpercentile(np.r_[d.R_s_kOhm[ok], d.R_bond_kOhm[ok]], [1, 99])
        aa.point_image(x, d.R_bond_kOhm[ok].clip(lo, hi).to_numpy(),
                       d.R_s_kOhm[ok].clip(lo, hi).to_numpy(),
                       np.tile([0.12, 0.47, 0.71], (int(ok.sum()), 1)), alpha=0.4)
        x.plot([lo, hi], [lo, hi], "k--", lw=0.8)
        x.set_xlim(lo, hi)
        x.set_ylim(lo, hi)
        x.set_xlabel("fast-scan R_bond, one point (kOhm)")
        x.set_ylabel("sweep R_s, slope, LED anchored per pixel (kOhm)")
        x.set_title(f"{arr}: the same pixels, same day (Spearman "
                    f"{d.R_s_kOhm.corr(d.R_bond_kOhm, method='spearman'):.3f})", fontsize=10)
        x.grid(alpha=0.3)
        x = ax[j, 2]
        x.hist(cur.clip(-2, 2), bins=120, color="tab:green")
        x.axvline(0, color="k", lw=0.8)
        x.set_xlabel("curvature: change of dV/dI across the pixel's range / R_s (clipped +-2)")
        x.set_ylabel("pixels")
        x.set_title(f"{arr}: linearity, {100 * (cur > 0).mean():.0f} % superlinear", fontsize=10)
        x.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out / "series_resistance.png", dpi=150)
    plt.close(fig)
    (out / "summary.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))
    return 0


if __name__ == "__main__":
    sys.exit(main())
