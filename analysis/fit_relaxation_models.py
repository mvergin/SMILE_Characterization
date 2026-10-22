"""What shape is the relaxation? Fit five candidate laws to every long Array42 trace.

    python analysis/fit_relaxation_models.py [path-to-a-run-tree]

Section 3 called the settling a power law, |I - I_inf| ~ t^-alpha. The operator's reading
of the plots -- faster at first, then slower, then flat -- is what EVERY decay looks like
on linear axes, so the shape alone does not decide it. What decides it is curvature on
log-log, and that differs between the candidates:

  PL     I_inf + A t^-a                    straight on log-log, constant local slope
  EXP    I_inf + A exp(-t/tau)             one time constant (RC-like, any tau)
  2EXP   I_inf + A1 e^-t/t1 + A2 e^-t/t2   two independent processes
  STR    I_inf + A exp(-(t/tau)^b)         stretched: a broad distribution of time constants
  PLC    I_inf + A t^-a exp(-t/tau)        a power law that is cut off at long times

Every model has I_inf free. Fitting is variable projection: the non-linear parameters
(exponent, time constants, stretch) are scanned on a grid, and for each grid point the
linear ones (I_inf and the amplitudes) come from weighted least squares, so no optimiser
can wander into a local minimum and no scipy is needed. Data are log-binned (equal weight
per decade, as in analyze_cap_comparison) with each bin weighted by its standard error,
estimated from the sample-to-sample noise. Models are ranked by BIC = chi2 + k ln N.

Prints a table and writes investigation_report/fig43_relaxation_models.png.
"""
from __future__ import annotations

import itertools
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = (Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else Path(__file__).resolve().parent.parent)
sys.path.insert(0, str(ROOT))
from analyze_cap_comparison import Run, clean, noise_sigma  # noqa: E402

OUT = Path(__file__).resolve().parent / "investigation_report"   # as make_capacitance_figures.py: never CWD-relative


# ------------------------------------------------------------------ data
def binned(t, i, t_min, nbin=90):
    """Geometric bins: mean t, mean |I|, standard error from the trace's own noise."""
    t, i = clean(t, i)
    i = np.abs(i)
    sig = noise_sigma(i)
    m = t >= t_min
    t, i = t[m], i[m]
    edges = np.geomspace(t.min(), t.max(), nbin + 1)
    idx = np.clip(np.digitize(t, edges) - 1, 0, nbin - 1)
    tb, yb, eb = [], [], []
    for b in range(nbin):
        sel = idx == b
        n = int(sel.sum())
        if n:
            tb.append(np.exp(np.log(t[sel]).mean()))
            yb.append(i[sel].mean())
            eb.append(max(sig, 1e-15) / np.sqrt(n))
    return np.array(tb), np.array(yb), np.array(eb)


def traces():
    """(label, t, I, t_min) for every long Array42 relaxation record on disk."""
    out = []
    r = Run(sorted((ROOT / "results").glob("261021-1237_*"))[0])
    for row, it, _ in r.step_traces():
        if row["stage"] == "long" and row["highc"] == "True":
            out.append((f"21 Oct, {row['quadrant']} ({row['pixel_x']},{row['pixel_y']}), "
                        f"-1.00->-0.80 V, 412 s", it[0], it[2], 1.0))
    z = np.load(sorted(ROOT.glob("charatests/261013-2140_*"))[0] / "traces.npz")
    for k in sorted(z.files):
        if k.startswith(("pm_", "v_")):
            continue
        v = k.split("_v")[1][:6]
        if v.startswith("-0.95"):
            continue                       # 0.05 V step: excess below the noise, see text
        out.append((f"13 Oct, TL (124,135), -1.00->{v} V, 21 s", z[k][0], z[k][-1], 0.8))
    z = np.load(sorted(ROOT.glob("demo/q3_step600/261015-2105_*"))[0] / "traces.npz")
    for k in sorted(z.files):
        if k.startswith(("pm_", "v_")) or "_hc1_" not in k:
            continue
        f = k.split("_f")[1][:6]; v = k.split("_v")[1][:6]
        out.append((f"15 Oct, {f}->{v} V, High-C, 610 s", z[k][0], z[k][-1], 1.0))
    return out


# ------------------------------------------------------------------ models
def _wls(cols, y, w):
    """Weighted linear least squares; returns (coefs, chi2, prediction)."""
    A = np.column_stack(cols) * w[:, None]
    coef, *_ = np.linalg.lstsq(A, y * w, rcond=None)
    pred = np.column_stack(cols) @ coef
    return coef, float(np.sum(((y - pred) * w) ** 2)), pred


def fit(t, y, e):
    w = 1.0 / e
    one = np.ones_like(t)
    T = t.max()
    taus = np.geomspace(t.min() / 3, T * 10, 70)
    res = {}

    best = (np.inf,)
    for a in np.linspace(0.05, 3.0, 120):
        c, chi, p = _wls([one, t ** -a], y, w)
        if chi < best[0]: best = (chi, dict(alpha=a), p)
    res["PL"] = best + (3,)

    best = (np.inf,)
    for tau in taus:
        c, chi, p = _wls([one, np.exp(-t / tau)], y, w)
        if chi < best[0]: best = (chi, dict(tau=tau), p)
    res["EXP"] = best + (3,)

    best = (np.inf,)
    for t1, t2 in itertools.combinations(taus[::2], 2):
        c, chi, p = _wls([one, np.exp(-t / t1), np.exp(-t / t2)], y, w)
        if chi < best[0]: best = (chi, dict(tau1=t1, tau2=t2), p)
    res["2EXP"] = best + (5,)

    best = (np.inf,)
    for tau in taus:
        for b in np.linspace(0.08, 1.0, 47):
            c, chi, p = _wls([one, np.exp(-(t / tau) ** b)], y, w)
            if chi < best[0]: best = (chi, dict(tau=tau, beta=b), p)
    res["STR"] = best + (4,)

    best = (np.inf,)
    for a in np.linspace(0.05, 3.0, 60):
        for tau in taus[::2]:
            c, chi, p = _wls([one, t ** -a * np.exp(-t / tau)], y, w)
            if chi < best[0]: best = (chi, dict(alpha=a, tau=tau), p)
    res["PLC"] = best + (4,)
    n = len(t)
    return {k: dict(chi2=v[0], par=v[1], pred=v[2], k=v[3], bic=v[0] + v[3] * np.log(n),
                    red=v[0] / max(1, n - v[3])) for k, v in res.items()}


def fmt(par):
    out = []
    for k, v in par.items():
        out.append(f"{k}={v:.3g}" + (" s" if k.startswith("tau") else ""))
    return ", ".join(out)


# ------------------------------------------------------------------ main
if __name__ == "__main__":
    OUT.mkdir(exist_ok=True)
    rows = traces()
    order = ["PL", "EXP", "2EXP", "STR", "PLC"]
    wins = {k: 0 for k in order}
    fig, axes = plt.subplots(2, 4, figsize=(17, 8.2))
    print(f"{'trace':52} " + " ".join(f"{m:>9}" for m in order) + "   best  (BIC rel. to best; "
          "chi2/dof of best)")
    allres = []
    for idx, (label, t, i, tmin) in enumerate(rows):
        tb, yb, eb = binned(t, i, tmin)
        R = fit(tb, yb, eb)
        b0 = min(R[m]["bic"] for m in order)
        best = min(order, key=lambda m: R[m]["bic"])
        wins[best] += 1
        allres.append((label, R, best))
        print(f"{label[:52]:52} " + " ".join(f"{R[m]['bic'] - b0:9.0f}" for m in order)
              + f"   {best:4}  red chi2 {R[best]['red']:.2f}  [{fmt(R[best]['par'])}]")
        if idx < 4:                         # the four 21 Oct quadrants get a panel each
            a = axes[0][idx]
            a.errorbar(tb, yb * 1e9, yerr=eb * 1e9, fmt=".", ms=3, color="0.25", lw=0.6,
                       label="data, log-binned")
            for m, col, ls in (("PL", "tab:blue", "-"), ("EXP", "tab:orange", "--"),
                               ("STR", "tab:green", "-."), ("2EXP", "tab:red", ":")):
                a.plot(tb, R[m]["pred"] * 1e9, color=col, ls=ls, lw=1.3,
                       label=f"{m}  dBIC {R[m]['bic'] - b0:.0f}")
            a.set_xscale("log")
            a.set_title(label.split(", -1")[0], fontsize=9)
            a.set_xlabel("t after the step  [s, log]")
            a.grid(which="both", alpha=0.25)
            a.legend(fontsize=6.6, loc="lower right")
            if idx == 0:
                a.set_ylabel("I  [nA]")
            # residuals of each model, in units of the bin's standard error
            r = axes[1][idx]
            for m, col, ls in (("PL", "tab:blue", "-"), ("EXP", "tab:orange", "--"),
                               ("STR", "tab:green", "-."), ("2EXP", "tab:red", ":")):
                r.plot(tb, (yb - R[m]["pred"]) / eb, color=col, ls=ls, lw=1.0, label=m)
            r.axhline(0, color="0.3", lw=0.8)
            r.axhspan(-2, 2, color="0.9", zorder=0)
            r.set_xscale("log")
            r.set_ylim(-25, 25)
            r.set_xlabel("t after the step  [s, log]")
            r.grid(which="both", alpha=0.25)
            if idx == 0:
                r.set_ylabel("residual / standard error")
                r.legend(fontsize=7)
    fig.suptitle("Array42, 21 Oct, the four long captures (run 261021-1237): each candidate law "
                 "with I_inf free. Bottom: residuals in standard errors; grey = +/-2.",
                 fontsize=10.5)
    fig.tight_layout()
    fig.savefig(OUT / "fig43_relaxation_models.png", dpi=300)
    plt.close(fig)
    print("\nbest model by BIC, count over", len(rows), "traces:", wins)
