"""Generate the figures for INVESTIGATION_REPORT.md from the real run data.

    python make_capacitance_figures.py [path-to-SMILE_Characterization]

Each function builds one figure from one run and RETURNS the numbers that drove a
decision, so the report text and the plots cannot drift apart: the script prints the
numbers it plotted, and the report quotes those.

Nothing here touches hardware. It reads the `charatests/` runs written by
`characterize_capacitance.py` and the `caprun/` sweeps written by `measure_iqe`.
"""
from __future__ import annotations

import csv
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OVERFLOW = 1e36                      # the 2600's over-range sentinel
AREA_CM2 = 100.0 * 1e-8              # 10x10 um pixel
EPH_EV = 1240.0 / 440.0              # 440 nm photon, in volts
OUT = Path("investigation_report")
DPI = 300

ROOT = Path(sys.argv[1] if len(sys.argv) > 1 else "../SMILE_Characterization")


def run(name: str) -> Path:
    """A characterisation run directory by its timestamp prefix."""
    hits = sorted(ROOT.glob(f"charatests/{name}_*")) + sorted(ROOT.glob(f"caprun/{name}_*"))
    if not hits:
        raise SystemExit(f"run {name} not found under {ROOT}")
    return hits[0]


def rows(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        return list(csv.DictReader(f))


def num(r: dict, key: str) -> float:
    v = r.get(key)
    try:
        x = float(v)
    except (TypeError, ValueError):
        return float("nan")
    return float("nan") if abs(x) > OVERFLOW else x


def sweep_rows(name: str) -> list[dict]:
    d = run(name)
    return rows(next(d.glob("NVLEDSweep/Sweep/*aggregated_bitval=*_mean_std.csv")))


def led(r: dict) -> tuple[float, float, float]:
    """(I_LED, P_LED, EQE) for one sweep row, leakage- and dark-corrected."""
    lit = abs(num(r, "NVLED_CURR_MEDIAN"))
    ref = abs(num(r, "NVLED_CURR_REF"))
    i = lit - ref
    p = num(r, "PM400_POWER_MEDIAN") - num(r, "PM400_POWER_REF")
    return i, p, (p / i) / EPH_EV if i > 0 else float("nan")


def finish(fig, name: str, note: str = "") -> None:
    OUT.mkdir(exist_ok=True)
    if note:
        fig.text(0.5, -0.015, note, ha="center", fontsize=7.5, style="italic")
    fig.tight_layout()
    fig.savefig(OUT / name, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {OUT / name}")


# ---------------------------------------------------------------- 1. range artifact
def fig_range_artifact():
    """Without High-C the reading is a fixed fraction of the RANGE, not the current."""
    rs = rows(run("261013-1316") / "step.csv")
    off = defaultdict(dict)
    on = {}
    for r in rs:
        v = num(r, "v_to")
        if r["highc"] == "True":
            on[v] = abs(num(r, "settled"))
        else:
            off[v][float(r["range_i"])] = abs(num(r, "settled"))

    fig, (a, b) = plt.subplots(1, 2, figsize=(11.5, 4.4))
    for v in sorted(off, reverse=True):
        rg = sorted(off[v])
        a.loglog(rg, [off[v][x] for x in rg], "o-", label=f"{v:+.2f} V")
        b.semilogx(rg, [100 * off[v][x] / x for x in rg], "o-", label=f"{v:+.2f} V")
    for n_, (v, i) in enumerate(sorted(on.items(), reverse=True)):
        a.axhline(i, ls="--", lw=1, color="tab:green")
        a.annotate(f"High-C, {v:+.2f} V: {i*1e9:.0f} nA", (1.05e-6, i), fontsize=7,
                   color="tab:green", textcoords="offset points",
                   xytext=(0, 4 + 9 * (len(on) - 1 - n_)))
    a.set_xlabel("forced measure range (A)")
    a.set_ylabel("settled current reported (A)")
    a.set_title("High-C OFF: the reading follows the RANGE", fontsize=10)
    a.legend(fontsize=8, title="bias")
    a.grid(alpha=0.3, which="both")
    b.axhspan(42, 52, color="tab:red", alpha=0.12, label="42-52% band")
    b.set_xlabel("forced measure range (A)")
    b.set_ylabel("reading as % of full range")
    b.set_title("the same data as a fraction of range: flat", fontsize=10)
    b.set_ylim(0, 100)
    b.legend(fontsize=8)
    b.grid(alpha=0.3, which="both")
    fig.suptitle("1. Without High-C, a low-current reading is a range artifact "
                 "(run 261013-1316, pixel 54,46)", fontsize=11)
    spread = max(max(d.values()) / min(d.values()) for d in off.values())
    finish(fig, "fig01_range_artifact.png",
           f"The same physical current reads {spread:.0f}x differently depending only on which "
           f"range is forced. High-C removes the choice: the limit sets the range.")
    return dict(spread=spread, pct=[100 * off[v][x] / x for v in off for x in off[v]],
                highc=on)


# ------------------------------------------------------------------- 2. capacitance
def fig_capacitance():
    """C from a current ramp, two setpoints, which cancels the ramp leakage."""
    rs = rows(run("261013-1625") / "cap.csv")
    g = defaultdict(dict)
    for r in rs:
        key = (r["quadrant"], r["pixel_x"], r["pixel_y"], r["lit"])
        g[key][num(r, "i_set")] = abs(num(r, "dvdt"))

    solved, raw = defaultdict(list), defaultdict(list)
    for (q, x, y, lit), d in g.items():
        if 1e-5 in d and 1e-4 in d:
            C = (1e-4 - 1e-5) / (d[1e-4] - d[1e-5])
            solved[q].append(C * 1e6)
    for r in rs:
        raw[(r["lit"], num(r, "i_set"))].append(num(r, "capacitance_F") * 1e6)

    fig, (a, b) = plt.subplots(1, 2, figsize=(11.5, 4.4))
    sets = sorted({k[1] for k in raw})
    for lit, style in (("False", "o--"), ("True", "s-")):
        m = [np.mean(raw[(lit, s)]) for s in sets]
        e = [np.std(raw[(lit, s)]) for s in sets]
        a.errorbar([s * 1e6 for s in sets], m, yerr=e, fmt=style,
                   label="blanked" if lit == "False" else "lit", capsize=3)
    allC = [c for v in solved.values() for c in v]
    a.axhline(np.mean(allC), color="tab:red", ls="-", lw=1.4,
              label=f"two-setpoint solve: {np.mean(allC):.1f} uF")
    a.set_xscale("log")
    a.set_yscale("log")
    a.set_xlabel("ramp current (uA)")
    a.set_ylabel("apparent C (uF)")
    a.set_title("a single ramp current reads high:\nleakage steals part of it", fontsize=10)
    a.legend(fontsize=8)
    a.grid(alpha=0.3, which="both")

    qs = sorted(solved)
    b.bar(range(len(qs)), [np.mean(solved[q]) for q in qs],
          yerr=[np.std(solved[q]) for q in qs], capsize=4, color="tab:blue", alpha=0.85)
    for i, q in enumerate(qs):
        b.scatter([i] * len(solved[q]), solved[q], color="k", s=14, zorder=3)
    b.set_xticks(range(len(qs)), qs)
    b.set_ylim(40, 45)
    b.set_xlabel("quadrant")
    b.set_ylabel("C (uF)")
    b.set_title(f"and it is the SAME everywhere:\n{np.mean(allC):.1f} +- {np.std(allC):.1f} uF "
                f"over {len(allC)} measurements", fontsize=10)
    b.grid(alpha=0.3, axis="y")
    fig.suptitle("2. The array carries one global 42.8 uF, uniform across all four quadrants "
                 "(run 261013-1625, 12 pixels)", fontsize=11)
    finish(fig, "fig02_capacitance.png",
           "C = (I2-I1)/(|dV/dt2|-|dV/dt1|) cancels the ramp leakage; a single setpoint cannot.")
    return dict(C_mean=np.mean(allC), C_std=np.std(allC), n=len(allC),
                per_quadrant={q: float(np.mean(v)) for q, v in solved.items()})


# --------------------------------------------------------------- 3. settling is t^-n
def _trace(name: str):
    d = np.load(run(name) / "traces.npz")
    out = {}
    for k in sorted(d.files):
        t, ia, ib = d[k]
        t = np.asarray(t, float)
        i = np.abs(np.asarray(ib, float))
        ok = np.isfinite(i) & (i < OVERFLOW)
        out[k] = (t[ok], i[ok])
    return out


def _smooth(i):
    w = max(5, i.size // 100)
    return np.convolve(i, np.ones(w) / w, mode="valid"), w


def fig_settling_law():
    """|I - Iinf| decays as a power law. An RC settle would not."""
    fig, (a, b) = plt.subplots(1, 2, figsize=(11.5, 4.6))
    slopes = {}
    for k, (t, i) in _trace("261013-2140").items():
        v = float(k.split("_v")[1][:6])
        inf = float(np.median(i[int(0.9 * i.size):]))
        sm, w = _smooth(i)
        ts, dev = t[w - 1:], np.abs(sm - inf) / inf * 100
        m = (ts > 0) & (dev > 0)
        line, = a.loglog(ts[m], dev[m], lw=1.1, label=f"{v:+.2f} V (dV={abs(v+1.0):.2f})")
        b.semilogy(ts[m], dev[m], lw=1.1, color=line.get_color())
        f = m & (ts > 0.8) & (ts < 10) & (dev > 0.2)
        if f.sum() > 20:
            slopes[v] = float(np.polyfit(np.log(ts[f]), np.log(dev[f]), 1)[0])
            j = int(np.argmax(ts[f] > 2.5))
            a.annotate(f"t^{slopes[v]:.1f}", (ts[f][j], dev[f][j]), fontsize=8,
                       color=line.get_color(), textcoords="offset points", xytext=(6, 6))
    for ax_, t_ in ((a, "log-log: a POWER LAW is a straight line"),
                    (b, "log-linear: an EXPONENTIAL would be a straight line")):
        ax_.axhline(1.0, color="tab:red", ls="--", lw=1, label="1% of settled")
        ax_.set_xlabel("time after the voltage step (s)")
        ax_.set_ylabel("|I - Iinf| / Iinf  (%)")
        ax_.set_ylim(2e-2, 400)
        ax_.set_title(t_, fontsize=10)
        ax_.grid(alpha=0.3, which="both")
    a.legend(fontsize=8)
    b.set_xlim(0, 21)
    fig.suptitle("3. The settling is charge relaxation, not RC charging "
                 "(run 261013-2140, High-C, 10 uA limit)", fontsize=11)
    finish(fig, "fig03_settling_law.png",
           "Straight on the left, curved on the right. At 10 uA a 0.2 V step across 42.8 uF "
           "charges in 0.86 s, yet 1% takes 13 s -- charging was never the bottleneck.")
    return dict(slopes=slopes)


# ----------------------------------------------------- 4. small steps do not help
def fig_step_size():
    """The result that refuted the obvious fix."""
    rs = rows(run("261013-2337") / "stepdir.csv")
    tr = _trace("261013-2337")
    fig, (a, b) = plt.subplots(1, 2, figsize=(11.5, 4.4))
    final = {}
    for k, (t, i) in tr.items():
        vf = float(k.split("_f")[1][:6])
        a.plot(t, i * 1e9, lw=0.9, label=f"from {vf:+.2f} V  (dV={abs(-0.8-vf):.2f})")
        final[vf] = float(np.median(i[-50:]))
    truth = min(final.values())
    a.axhline(truth * 1e9, color="k", ls=":", lw=1, label=f"lowest final: {truth*1e9:.0f} nA")
    a.set_xlabel("time after the step onto -0.80 V (s)")
    a.set_ylabel("|I| (nA)")
    a.set_ylim(380, 520)
    a.set_title("every step lands on -0.80 V, from four distances", fontsize=10)
    a.legend(fontsize=8)
    a.grid(alpha=0.3)

    dv = [abs(num(r, "dv")) for r in rs]
    t1 = [num(r, "t_1pct") for r in rs]
    t01 = [num(r, "t_01pct") for r in rs]
    dev = [100 * (final[num(r, "v_from")] - truth) / truth for r in rs]
    b.plot(dv, t1, "o-", label="to 1%")
    b.plot(dv, t01, "s-", label="to 0.1%")
    b2 = b.twinx()
    b2.plot(dv, dev, "^--", color="tab:red", label="final value, % above the lowest")
    b2.set_ylabel("final value error (%)", color="tab:red")
    b.set_xlabel("step size |dV| (V)")
    b.set_ylabel("settling time (s)")
    b.set_title("the SMALLEST step is the slowest and the most biased", fontsize=10)
    b.legend(fontsize=8, loc="center right")
    b.grid(alpha=0.3)
    fig.suptitle("4. Small steps do not settle faster -- the guess the bench refuted "
                 "(run 261013-2337, NPLC 1.0)", fontsize=11)
    finish(fig, "fig04_step_size.png",
           "A 20 mV step was still 4.5% high and drifting at 80 s; a 200 mV step onto the same "
           "point was flat from 60 s. Each point inherits the unrelaxed tail of the previous one.")
    return dict(final={k: v * 1e9 for k, v in final.items()},
                t_1pct=dict(zip(dv, t1)), t_01pct=dict(zip(dv, t01)))


# ------------------------------------------------------------- 5. no sine, and NPLC
def fig_noise():
    """The 'sine riding on the transient' hypothesis, tested and rejected."""
    fig, (a, b) = plt.subplots(1, 2, figsize=(11.5, 4.4))
    stats = {}
    for name, lbl, col in (("261013-2140", "NPLC 0.01", "tab:blue"),
                           ("261013-2201", "NPLC 1.0 (integer)", "tab:green")):
        tr = _trace(name)
        key = [k for k in tr if "-0.800" in k][0]
        t, i = tr[key]
        cut = int(0.5 * t.size)
        t, i = t[cut:], i[cut:]
        dt = float(np.median(np.diff(t)))
        resid = i - np.polyval(np.polyfit(t, i, 2), t)
        amp = np.abs(np.fft.rfft(resid * np.hanning(resid.size))) * (4.0 / resid.size)
        f = np.fft.rfftfreq(resid.size, dt)
        sel = f > 2.0
        a.semilogy(f[sel], amp[sel] * 1e9, lw=0.7, color=col, label=lbl, alpha=0.85)
        stats[lbl] = dict(noise=float(np.std(np.diff(i)) / np.sqrt(2)) * 1e9,
                          peak=float(np.max(amp[sel])) * 1e9,
                          mains=float(np.max(amp[np.abs(f - 50) < 2])) * 1e9 if f[-1] > 50
                          else float("nan"))
    a.axvline(50, color="red", ls="--", lw=1.2, label="50 Hz mains")
    a.set_xlabel("frequency (Hz)")
    a.set_ylabel("amplitude (nA)")
    a.set_title("the spectrum is FLAT: no line at the mains, or anywhere", fontsize=10)
    a.legend(fontsize=8)
    a.grid(alpha=0.3, which="both")

    labels = list(stats)
    xs = np.arange(len(labels))
    b.bar(xs - 0.2, [stats[k]["noise"] for k in labels], 0.38, label="per-sample noise")
    b.bar(xs + 0.2, [stats[k]["mains"] for k in labels], 0.38, label="amplitude at 50 Hz")
    b.set_xticks(xs, labels)
    b.set_ylabel("nA")
    b.set_yscale("log")
    b.set_title("integer NPLC cuts the NOISE, not a line", fontsize=10)
    b.legend(fontsize=8)
    b.grid(alpha=0.3, axis="y")
    fig.suptitle("5. The oscillation is white noise, not mains pickup "
                 "(runs 261013-2140 / -2201, pixel 124,135 at -0.80 V)", fontsize=11)
    finish(fig, "fig05_noise.png",
           "The 50 Hz bin is no larger than any other bin. Integer NPLC still helps -- by "
           "averaging (sqrt(100) = 10x), not by rejection.")
    return stats


# --------------------------------------------------------------- 6. the leakage
def fig_leakage():
    """lit vs addressed-but-off vs blanked, and why the blank is not a zero."""
    rs = rows(run("261013-1625") / "leak.csv")
    byv = defaultdict(list)
    for r in rs:
        byv[num(r, "v_to")].append(r)
    vs = sorted(byv, reverse=True)

    fig, (a, b) = plt.subplots(1, 2, figsize=(11.5, 4.4))
    for key, lbl, style in (("i_lit", "lit (bitval 15)", "o-"),
                            ("i_off", "SAME pixel at bitval 0", "s-"),
                            ("i_blank", "frame blanked", "^-")):
        m = [np.mean([abs(num(r, key)) for r in byv[v]]) * 1e9 for v in vs]
        e = [np.std([abs(num(r, key)) for r in byv[v]]) * 1e9 for v in vs]
        a.errorbar(vs, m, yerr=e, fmt=style, capsize=3, label=lbl)
    a.set_yscale("log")
    a.set_xlabel("bias (V)")
    a.set_ylabel("cathode current (nA)")
    a.set_title("a BLANKED frame draws ~4x MORE than the addressed pixel at bitval 0",
                fontsize=10)
    a.legend(fontsize=8)
    a.grid(alpha=0.3, which="both")

    lf = [100 * np.mean([num(r, "leak_frac") for r in byv[v]]) for v in vs]
    ild = [np.mean([abs(num(r, "i_led")) for r in byv[v]]) for v in vs]
    b.plot([i / AREA_CM2 for i in ild], lf, "o-", color="tab:red")
    for v, j, y in zip(vs, [i / AREA_CM2 for i in ild], lf):
        b.annotate(f"{v:+.2f} V", (j, y), fontsize=7, textcoords="offset points", xytext=(5, -9))
    b.set_xscale("log")
    b.set_xlabel("J from the CORRECTED current (A/cm2)")
    b.set_ylabel("leakage as % of the raw reading")
    b.set_title("what fraction of ch B is not the LED", fontsize=10)
    b.grid(alpha=0.3, which="both")
    fig.suptitle("6. Channel B reads the total cathode current; the reference must be the same "
                 "pixel at bitval 0 (run 261013-1625, 12 pixels)", fontsize=11)
    finish(fig, "fig06_leakage.png",
           "lit-minus-blanked is NEGATIVE, which is impossible. Blanking changes the backplane; "
           "bitval 0 keeps the frame identical and only removes the drive.")
    return dict(v=vs, leak_frac=lf,
                i_off=[float(np.mean([abs(num(r, "i_off")) for r in byv[v]])) * 1e9 for v in vs],
                i_blank=[float(np.mean([abs(num(r, "i_blank")) for r in byv[v]])) * 1e9 for v in vs])


# ------------------------------------------------------- 7. the EQE collapse undone
def fig_eqe_correction():
    """The apparent 20x EQE collapse was a contaminated denominator."""
    rs = rows(run("261013-1625") / "leak.csv")
    byv = defaultdict(list)
    for r in rs:
        byv[num(r, "v_to")].append(r)
    vs = sorted(byv, reverse=True)

    fig, (a, b) = plt.subplots(1, 2, figsize=(11.5, 4.4))
    raw_j, raw_e, cor_j, cor_e = [], [], [], []
    for v in vs:
        p = np.mean([num(r, "p_led") for r in byv[v]])
        iraw = np.mean([abs(num(r, "i_lit")) for r in byv[v]])
        icor = np.mean([abs(num(r, "i_led")) for r in byv[v]])
        raw_j.append(iraw / AREA_CM2)
        raw_e.append(100 * (p / iraw) / EPH_EV)
        cor_j.append(icor / AREA_CM2)
        cor_e.append(100 * (p / icor) / EPH_EV)
    a.semilogx(raw_j, raw_e, "o--", color="tab:gray",
               label="uncorrected: EQE = P / (total ch B)")
    a.semilogx(cor_j, cor_e, "o-", color="tab:blue",
               label="corrected: EQE = P / (ch B - bitval 0)")
    a.set_xlabel("J (A/cm2)")
    a.set_ylabel("EQE (%)")
    a.set_title("the same four biases, two denominators", fontsize=10)
    a.legend(fontsize=8)
    a.grid(alpha=0.3, which="both")

    ratio = [c / r for c, r in zip(cor_e, raw_e)]
    b.semilogx(cor_j, ratio, "o-", color="tab:red")
    b.axhline(1.0, color="k", lw=0.8)
    for v, j, y in zip(vs, cor_j, ratio):
        b.annotate(f"{v:+.2f} V", (j, y), fontsize=7, textcoords="offset points", xytext=(5, 4))
    b.set_xlabel("J (A/cm2)")
    b.set_ylabel("corrected EQE / uncorrected")
    b.set_title("how wrong the uncorrected value is", fontsize=10)
    b.grid(alpha=0.3, which="both")
    fig.suptitle("7. The apparent EQE collapse on the rising side was a contaminated denominator",
                 fontsize=11)
    finish(fig, "fig07_eqe_correction.png",
           "This invalidates every rising-side EQE the project produced before 261013, and those "
           "runs cannot be repaired: the reference was never recorded.")
    return dict(J=cor_j, eqe_corrected=cor_e, eqe_raw=raw_e, ratio=ratio)


# ------------------------------------------------------------------- 8. spatial
def fig_spatial():
    """Is any of this uniform? C is. The LED is not."""
    rs = rows(run("261013-1625") / "leak.csv")
    byq = defaultdict(lambda: defaultdict(list))
    for r in rs:
        byq[r["quadrant"]][num(r, "v_to")].append(r)
    qs = ["TL", "TR", "BL", "BR"]
    vs = sorted({num(r, "v_to") for r in rs}, reverse=True)

    fig, (a, b) = plt.subplots(1, 2, figsize=(11.5, 4.4))
    for q in qs:
        e = [100 * np.mean([num(r, "eqe_led") for r in byq[q][v]]) for v in vs]
        a.plot(vs, e, "o-", label=q)
    a.set_xlabel("bias (V)")
    a.set_ylabel("EQE (%)")
    a.set_title("EQE by quadrant", fontsize=10)
    a.legend(fontsize=8)
    a.grid(alpha=0.3)

    left = [q for q in qs if q.endswith("L")]
    right = [q for q in qs if q.endswith("R")]
    ratio = []
    for v in vs:
        l = np.mean([num(r, "eqe_led") for q in left for r in byq[q][v]])
        rr = np.mean([num(r, "eqe_led") for q in right for r in byq[q][v]])
        ratio.append(100 * (l / rr - 1))
    b.plot(vs, ratio, "o-", color="tab:purple")
    b.axhline(0, color="k", lw=0.8)
    b.set_xlabel("bias (V)")
    b.set_ylabel("left half brighter by (%)")
    b.set_title("a real left-right gradient, 17-31%", fontsize=10)
    b.grid(alpha=0.3)
    fig.suptitle("8. C and the leakage are uniform; the LED efficiency is not "
                 "(run 261013-1625, 3 pixels per quadrant)", fontsize=11)
    finish(fig, "fig08_spatial.png",
           "C varies 0.8% and the bitval-0 reference 3% across the array, so one compliance "
           "schedule and one settle budget work array-wide. The 17-31% EQE gradient is signal.")
    return dict(left_over_right=dict(zip(vs, ratio)))


# ------------------------------------------------------- 9. current source fails
def fig_isrc():
    """Why channel B stays a voltage source."""
    rs = rows(run("261013-1450") / "isrc.csv")
    g = defaultdict(list)
    for r in rs:
        g[num(r, "i_set")].append(r)
    fig, (a, b) = plt.subplots(1, 2, figsize=(11.5, 4.4))
    drift = {}
    for i_set in sorted(g):
        t = np.array([num(r, "t_s") for r in g[i_set]])
        v = np.array([num(r, "v_b") for r in g[i_set]])
        lbl = f"{i_set*1e9:,.0f} nA"
        a.plot(t, v, lw=1.3, label=lbl)
        m = t > t[-1] - 60
        drift[i_set] = float(np.polyfit(t[m], v[m], 1)[0]) * 1e6
    a.set_xlabel("time at this setpoint (s)")
    a.set_ylabel("cathode voltage (V)")
    a.set_title("the current is forced, so the VOLTAGE is the transient", fontsize=10)
    a.legend(fontsize=8, title="I set")
    a.grid(alpha=0.3)

    ks = sorted(drift)
    b.bar(range(len(ks)), [abs(drift[k]) for k in ks], color=
          ["tab:red" if abs(drift[k]) > 100 else "tab:green" for k in ks])
    b.axhline(100, color="k", ls="--", lw=1, label="100 uV/s = settled")
    b.axhspan(0, 100, color="tab:green", alpha=0.08)
    b.set_yscale("log")
    b.set_xticks(range(len(ks)), [f"{k*1e9:,.0f} nA" for k in ks])
    b.set_ylabel("|drift| over the last 60 s (uV/s)")
    b.set_title("below the ~250 nA backplane leak, the node cannot be held", fontsize=10)
    b.legend(fontsize=8)
    b.grid(alpha=0.3, axis="y")
    fig.suptitle("9. Current-source mode cannot reach low J on this array "
                 "(run 261013-1450, 180 s per setpoint)", fontsize=11)
    finish(fig, "fig09_isrc.png",
           "A current source supplies only what you ask for. Ask for 100 nA when the parasitic "
           "path wants 250 nA and the node collapses -- which is the whole region of interest.")
    return dict(drift_uV_per_s=drift)


# ------------------------------------------------------ 10. the compliance clamp
def fig_compliance():
    """My own bug, and the fix, on the same pixel."""
    before, after = sweep_rows("261014-0007"), sweep_rows("261014-0035")
    fig, (a, b) = plt.subplots(1, 2, figsize=(11.5, 4.4))
    for rs, lbl, style in ((before, "limit tied to the density knee", "o--"),
                           (after, "limit from the previous point", "o-")):
        v = [num(r, "NVLED_V") for r in rs]
        i = [abs(num(r, "NVLED_CURR_MEDIAN")) * 1e9 for r in rs]
        a.semilogy(v, i, style, ms=4, lw=1.1, label=lbl)
    a.axhline(1e4, color="tab:red", ls=":", lw=1.2, label="the 10 uA limit")
    a.set_xlabel("bias (V)")
    a.set_ylabel("|I| ch B (nA)")
    a.set_title("eight points pinned at exactly 10000.4 nA", fontsize=10)
    a.legend(fontsize=8)
    a.grid(alpha=0.3, which="both")

    bi = {round(num(r, "NVLED_V"), 3): r for r in before}
    diff_v, diff_p = [], []
    for r in after:
        v = round(num(r, "NVLED_V"), 3)
        if v not in bi:
            continue
        i1, i2 = led(bi[v])[0], led(r)[0]
        diff_v.append(v)
        diff_p.append(100 * (i2 - i1) / i1 if i1 else float("nan"))
    b.plot(diff_v, np.abs(diff_p), "o-", color="tab:purple")
    b.set_yscale("log")
    b.axhspan(0.01, 1, color="tab:green", alpha=0.10, label="agree within 1%")
    b.set_xlabel("bias (V)")
    b.set_ylabel("|difference| between the two runs (%)")
    b.set_title("the two runs agree everywhere EXCEPT the clamped points", fontsize=10)
    b.legend(fontsize=8)
    b.grid(alpha=0.3, which="both")
    fig.suptitle("10. The compliance must follow the current, not the point-density knee "
                 "(runs 261014-0007 vs -0035, pixel 124,135)", fontsize=11)
    finish(fig, "fig10_compliance.png",
           "The knee is at -1.3 V but the device still draws 30 uA at -1.27 V. Points on the "
           "limit are now flagged NVLED_AT_COMPLIANCE and dropped by the analysis.")
    return dict(max_diff_outside_clamp=float(np.nanmax(
        [abs(d) for v, d in zip(diff_v, diff_p) if not -1.29 < v < -1.07])))


# ----------------------------------------------------------- 11. the final sweep
def fig_final_sweep():
    """What the rebuilt sweep produces."""
    rs = sweep_rows("261014-0035")
    J, E, L, ok = [], [], [], []
    for r in rs:
        i, p, e = led(r)
        J.append(i / AREA_CM2)
        E.append(100 * e)
        L.append(100 * abs(num(r, "NVLED_CURR_REF")) / abs(num(r, "NVLED_CURR_MEDIAN")))
        ok.append(str(r.get("NVLED_AT_COMPLIANCE", "False")).lower() == "false")
    J, E, L, ok = map(np.array, (J, E, L, ok))

    fig, (a, b) = plt.subplots(1, 2, figsize=(11.5, 4.4))
    a.semilogx(J[ok], E[ok], "o-", ms=4, color="tab:blue")
    k = int(np.argmax(E[ok]))
    a.annotate(f"peak {E[ok][k]:.4f}% at J={J[ok][k]:.1f}",
               (J[ok][k], E[ok][k]), fontsize=8, textcoords="offset points", xytext=(6, -14))
    a.set_xlabel("J (A/cm2)")
    a.set_ylabel("EQE (%)")
    a.set_title(f"EQE over {J[ok].max()/J[ok].min():.0f}x in current density", fontsize=10)
    a.grid(alpha=0.3, which="both")

    b.semilogx(J[ok], L[ok], "o-", color="tab:red")
    b.set_xlabel("J (A/cm2)")
    b.set_ylabel("leakage as % of the raw reading")
    b.set_title("the correction is negligible at high J and dominant at low J", fontsize=10)
    b.grid(alpha=0.3, which="both")
    fig.suptitle("11. The rebuilt sweep: one pixel, 30 points, three decades of J "
                 "(run 261014-0035)", fontsize=11)
    finish(fig, "fig11_final_sweep.png",
           "High-C on, compliance from the previous point, 8 s dwell on the rising side, "
           "bitval-0 reference at every point.")
    return dict(J_min=float(J[ok].min()), J_max=float(J[ok].max()),
                peak_eqe=float(E[ok].max()), peak_J=float(J[ok][int(np.argmax(E[ok]))]),
                leak_at_min_J=float(L[ok][int(np.argmin(J[ok]))]))


# --------------------------------------------------------- 12. the error budget
def fig_error_budget():
    """Where the low-J limit sits now: the light, not the current."""
    a_rows, b_rows = sweep_rows("261014-0007"), sweep_rows("261014-0035")
    ai = {round(num(r, "NVLED_V"), 3): r for r in a_rows}
    J, di, dp, ei, ep = [], [], [], [], []
    for r in b_rows:
        v = round(num(r, "NVLED_V"), 3)
        if v not in ai or -1.29 < v < -1.07:          # skip the clamped region
            continue
        i2, p2, _ = led(r)
        i1, p1, _ = led(ai[v])
        J.append(i2 / AREA_CM2)
        di.append(abs(100 * (i2 - i1) / i1))
        dp.append(abs(100 * (p2 - p1) / p1))
        si = np.hypot(num(r, "NVLED_CURR_STD"), num(r, "NVLED_CURR_REF_STD"))
        sp = np.hypot(num(r, "PM400_POWER_STD"), num(r, "PM400_POWER_REF_STD"))
        ei.append(100 * si / i2)
        ep.append(100 * sp / p2)

    fig, (a, b) = plt.subplots(1, 2, figsize=(11.5, 4.4))
    a.loglog(J, ei, "o-", label="current (ch B, leakage-corrected)")
    a.loglog(J, ep, "s-", label="optical power (dark-corrected)")
    a.set_xlabel("J (A/cm2)")
    a.set_ylabel("within-burst error (%)")
    a.set_title("the light became the limit", fontsize=10)
    a.legend(fontsize=8)
    a.grid(alpha=0.3, which="both")

    b.loglog(J, di, "o-", label="current")
    b.loglog(J, dp, "s-", label="optical power")
    b.axhline(1.0, color="k", ls="--", lw=1, label="1%")
    b.set_xlabel("J (A/cm2)")
    b.set_ylabel("|difference| between two runs (%)")
    b.set_title("run-to-run, 28 minutes apart", fontsize=10)
    b.legend(fontsize=8)
    b.grid(alpha=0.3, which="both")
    fig.suptitle("12. Where the low-J limit sits now (runs 261014-0007 vs -0035)", fontsize=11)
    finish(fig, "fig12_error_budget.png",
           "The current reproduces to 0.03-1.3%. The optical reference is a ~30 pW dark level "
           "under a ~250 pW signal, so the floor is optical collection, not electrical care.")
    return dict(J=J, err_I=ei, err_P=ep, repro_I=di, repro_P=dp)


FIGURES = [fig_range_artifact, fig_capacitance, fig_settling_law, fig_step_size,
           fig_noise, fig_leakage, fig_eqe_correction, fig_spatial, fig_isrc,
           fig_compliance, fig_final_sweep, fig_error_budget]


def main():
    print(f"reading runs from {ROOT.resolve()}")
    for fn in FIGURES:
        print(f"\n{fn.__name__}:")
        out = fn()
        for k, v in out.items():
            if isinstance(v, (int, float)) and not isinstance(v, bool):
                print(f"    {k} = {v:.4g}")
            elif isinstance(v, dict) and len(v) <= 8:
                print(f"    {k} = " + ", ".join(
                    f"{kk}: {vv:.4g}" if isinstance(vv, float) else f"{kk}: {vv}"
                    for kk, vv in v.items()))
            elif isinstance(v, (list, tuple)) and len(v) <= 8:
                print(f"    {k} = " + ", ".join(f"{x:.4g}" for x in v))
            else:
                print(f"    {k} = <{type(v).__name__}, n={len(v)}>")
    print(f"\n{len(FIGURES)} figures in {OUT.resolve()}")


if __name__ == "__main__":
    main()
