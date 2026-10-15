"""What the 2 s pixel-switch wait costs, measured by the sweep rather than by a capture.

    python plot_switch_deficit.py PIXEL_MAJOR_RUN BIAS_MAJOR_RUN [BIAS_MAJOR_RUN ...]
                                 [--label L ...] [--out DIR]

`bias_major` gives the first pixel at each bias the full 54.47 s dwell with its own
frame lit, and every later pixel only `pixel_switch_settle_s`. That split is the whole
experiment, and it is already in any bias-major run: compare each position against the
pixel-major run on the same pixels.

    switch_deficit.png   left  : deficit against bias, order 0 vs the rest
                         right : deficit against position in the bias's pixel list

On section 13's 16 pixels, median over -1.25..-0.80 V, against the pixel-major run:

    run     switch wait   order 0 dI   order >0 dI   order >0 dP   at matched current
    2308    2 s (blanked)   -2.53 %      -2.42 %       -4.05 %        -1.59 %
    0822    2 s            -0.70 %      -2.99 %       -5.11 %        -1.88 %
    0958    20 s           -0.21 %      -3.25 %       -7.15 %        -3.55 %

**Ten times the switch wait did not reduce the deficit.** It is not a transient. The
third panel is why: at MATCHED CURRENT the 2 s runs agree with pixel-major to -1.9 %,
not -5 %, because the rising side has d(logP)/d(logI) = 1.0-1.2 and a 3 % current
deficit buys a 3-4 % power deficit for free. Expressed as the pixel-major voltage that
would have produced the same current, bias_major sits a near-constant **-4 to -6.6 mV**
off across the whole faint band, at 2 s and at 20 s alike. A constant offset in volts
is a state difference, not a relaxation.

This is worth having as a figure because the dedicated switch experiment bounded the
same quantity at 0.5 % and was wrong by 6x. It watched the TOTAL cathode current, 58 %
of which is backplane leakage at the faint end, and referenced its early samples to its
own 20 s tail -- so a transient longer than the record moves the reference with the
signal. The sweep has neither problem: it subtracts the leakage reference, it records
the optical channel, and its "settled" comparison point is a different run, not a later
part of the same trace.
"""
from __future__ import annotations

import argparse
import datetime as dt
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

FAINT_LO, FAINT_HI = -1.25, -0.80   # the band the deficit is quoted over
EXCLUDE = {(56, 51)}                 # section 15: does not go dark at bitval 0


def started(run: Path) -> dt.datetime | None:
    """Wall-clock start, from the run directory's own YYMMDD-HHMM name."""
    m = re.match(r"(\d{2})(\d{2})(\d{2})-(\d{2})(\d{2})", run.name)
    if not m:
        return None
    yy, mm, dd, HH, MM = (int(g) for g in m.groups())
    return dt.datetime(2000 + yy, mm, dd, HH, MM)


def load(run: Path) -> pd.DataFrame:
    agg = sorted((run / "NVLEDSweep").glob("*/aggregated_bitval=*_mean_std.csv"))
    if not agg:
        sys.exit(f"{run} has no aggregated NVLEDSweep CSV")
    a = pd.read_csv(agg[0])
    a["I"] = (a["NVLED_CURR_MEAN"] - a["NVLED_CURR_REF"]).abs()
    a["P"] = a["PM400_POWER_MEAN"] - a["PM400_POWER_REF"]
    a["E"] = a["P"] / a["I"]
    a = a.sort_values("TIME_START")
    # position in the bias's pixel list. In pixel_major this is meaningless and unused.
    a["order"] = a.groupby("NVLED_V").cumcount()
    a["V"] = a["NVLED_V"].round(4)
    return a


def deficit(ref: pd.DataFrame, run: pd.DataFrame) -> pd.DataFrame:
    """Per shared (pixel, bias): run/ref - 1, in percent, for I, P and EQE."""
    k = ["X", "Y", "V"]
    j = run.merge(ref[k + ["I", "P", "E"]], on=k, suffixes=("", "_ref"))
    j = j[~j[["X", "Y"]].apply(tuple, axis=1).isin(EXCLUDE)]
    for c in ("I", "P", "E"):
        j["d" + c] = 100.0 * (j[c] / j[c + "_ref"] - 1.0)
    return j


def at_matched_current(ref: pd.DataFrame, run: pd.DataFrame) -> pd.DataFrame:
    """Per shared pixel: interpolate the reference's own P(I), and the V that produced
    each current, then evaluate both at the run's current.

    Comparing at matched voltage conflates two things. The rising side has
    d(logP)/d(logI) ~ 1.0-1.2, so a current deficit shows up amplified in the power and
    the EQE looks broken when only the current moved. This asks the question the ABC fit
    actually needs: at the same current, is the light the same?
    """
    out = []
    for (x, y), g in run.groupby(["X", "Y"]):
        if (x, y) in EXCLUDE:
            continue
        r = ref[(ref["X"] == x) & (ref["Y"] == y)]
        r = r[(r["I"] > 0) & (r["P"] > 0)].sort_values("I")
        if len(r) < 4:
            continue
        li, lp, vv = np.log(r["I"].values), np.log(r["P"].values), r["V"].values
        for _, row in g.iterrows():
            if row["P"] <= 0 or not (r["I"].min() <= row["I"] <= r["I"].max()):
                continue   # never extrapolate: the ends are where the floor lives
            out.append((x, y, row["V"], row["order"],
                        100.0 * (row["P"] / np.exp(np.interp(np.log(row["I"]), li, lp)) - 1.0),
                        1000.0 * (row["V"] - np.interp(np.log(row["I"]), li, vv))))
    return pd.DataFrame(out, columns=["X", "Y", "V", "order", "dP_at_I", "dV_mV"])


def summarize(j: pd.DataFrame, label: str) -> None:
    band = j[(j["V"] >= FAINT_LO) & (j["V"] <= FAINT_HI)]
    print(f"\n{label}: {j['V'].nunique()} biases, "
          f"{j.groupby(['X', 'Y']).ngroups} pixels")
    for name, sel in (("order 0 (54.47 s dwell)", band[band["order"] == 0]),
                      ("order > 0 (switch wait)", band[band["order"] > 0])):
        if sel.empty:
            continue
        print(f"  {name:<24} I {sel['dI'].median():+6.2f} %   "
              f"P {sel['dP'].median():+6.2f} %   EQE {sel['dE'].median():+6.2f} %")
    if "dP_at_I" in j:
        m = j[(j["V"] >= FAINT_LO) & (j["V"] <= FAINT_HI) & (j["order"] > 0)]
        if not m.empty:
            print(f"  {'at MATCHED CURRENT':<24} P {m['dP_at_I'].median():+6.2f} %   "
                  f"equivalent offset {m['dV_mV'].median():+5.2f} mV")


def main() -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("reference", type=Path, help="the pixel_major run")
    ap.add_argument("runs", type=Path, nargs="+", help="bias_major run(s)")
    ap.add_argument("--label", nargs="*", default=None)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()
    out = args.out or (args.runs[-1] / "REPORT_assets")
    out.mkdir(parents=True, exist_ok=True)
    labels = args.label or [r.name.split("_")[0] for r in args.runs]

    ref = load(args.reference)
    js = []
    for r, lbl in zip(args.runs, labels):
        a = load(r)
        j = deficit(ref, a)
        mc = at_matched_current(ref, a)
        j = j.merge(mc[["X", "Y", "V", "dP_at_I", "dV_mV"]], on=["X", "Y", "V"], how="left")
        summarize(j, lbl)
        js.append((lbl, j))

    # Every run above is measured against the SAME reference, so elapsed time since
    # that reference is a confound on every number in this script. Print it: the
    # optical channel moves ~2 %/day and the table is the only place that shows it.
    t_ref = started(args.reference)
    if t_ref is not None:
        span = (ref["TIME_START"].max() - ref["TIME_START"].min()) / 7200
        t_ref += dt.timedelta(hours=span)          # its midpoint, not its first point
        print(f"\n{'run':<28}{'h after ref':>12}{'current':>10}{'P at matched I':>17}")
        for (lbl, j), r in zip(js, args.runs):
            t = started(r)
            b = j[(j["V"] >= FAINT_LO) & (j["V"] <= FAINT_HI) & (j["order"] > 0)]
            print(f"{lbl:<28}"
                  f"{((t - t_ref).total_seconds() / 3600) if t else float('nan'):>12.1f}"
                  f"{b['dI'].median():>9.2f} %{b['dP_at_I'].median():>16.2f} %")
        print("  a trend down this table with no change of method is the reference "
              "ageing, not the method.")

    fig, ax = plt.subplots(1, 3, figsize=(18, 4.8))
    for k, (lbl, j) in enumerate(js):
        c = f"C{k}"
        for sel, style, name in ((j[j["order"] == 0], "o-", "order 0: 54.47 s dwell"),
                                 (j[j["order"] > 0], "s--", "order >0: 2 s switch")):
            g = sel.groupby("V")["dI"].median()
            ax[0].plot(-g.index, g.values, style, ms=4, color=c, alpha=1.0 if style[0] == "o" else .6,
                       label=f"{lbl} — {name}")
    ax[0].axhline(0, color="k", lw=.8)
    ax[0].set(xlabel="|bias| (V)", ylabel="current vs pixel_major (%)",
              title="the pixel that holds the frame is right;\nthe ones that do not are 3 % low")
    ax[0].legend(fontsize=7)
    ax[0].grid(alpha=.3)

    for k, (lbl, j) in enumerate(js):
        band = j[(j["V"] >= FAINT_LO) & (j["V"] <= FAINT_HI)]
        for col, m, name in (("dI", "o", "current"), ("dP", "^", "optical power")):
            g = band.groupby("order")[col].median()
            ax[1].plot(g.index, g.values, m + "-", ms=4, color=f"C{k}",
                       alpha=1.0 if col == "dI" else .55, label=f"{lbl} — {name}")
    ax[1].axhline(0, color="k", lw=.8)
    ax[1].set(xlabel="position in the bias's pixel list", ylabel="vs pixel_major (%)",
              title=f"the SAME per-pixel pattern in all three runs, and 20 s\n"
                    f"did not flatten it ({FAINT_LO} to {FAINT_HI} V)")
    ax[1].legend(fontsize=7)
    ax[1].grid(alpha=.3)
    for k, (lbl, j) in enumerate(js):
        g = j[j["order"] > 0].groupby("V")["dP_at_I"].median()
        ax[2].plot(-g.index, g.values, "o-", ms=4, color=f"C{k}", label=f"{lbl}")
    ax[2].axhline(0, color="k", lw=.8)
    ax[2].axvspan(0.80, 0.90, color="k", alpha=.07)
    ax[2].annotate("below the\nleakage floor", (0.85, ax[2].get_ylim()[0]),
                   fontsize=7, ha="center", va="bottom")
    ax[2].set(xlabel="|bias| (V)", ylabel="optical power vs pixel_major (%)",
              title="at MATCHED CURRENT the 2 s runs agree to 2 %;\n"
                    "most of the optical deficit was the current")
    ax[2].legend(fontsize=7)
    ax[2].grid(alpha=.3)

    fig.suptitle("Ten times the switch wait did not reduce the deficit, so it is not a "
                 "transient — and at matched current it was never 5 %", y=1.02)
    fig.tight_layout()
    fig.savefig(out / "switch_deficit.png", dpi=130, bbox_inches="tight")
    plt.close(fig)
    pd.concat([j.assign(run=lbl) for lbl, j in js]).to_csv(
        out / "switch_deficit.csv", index=False)
    print(f"\nwrote {out / 'switch_deficit.png'}")


def demo() -> None:
    """Self-check: deficit() must pair on (pixel, bias) and order must reset per bias."""
    rng = np.random.default_rng(0)
    px = [(x, y) for x in (1, 2) for y in (3, 4)]
    v = [-1.0, -0.9]
    rows = []
    for t, (x, y) in enumerate(px):
        for vv in v:
            rows.append(dict(X=x, Y=y, NVLED_V=vv, TIME_START=t, NVLED_CURR_MEAN=-1e-6,
                             NVLED_CURR_REF=0.0, PM400_POWER_MEAN=1e-7,
                             PM400_POWER_REF=0.0))
    a = pd.DataFrame(rows)
    a["I"] = a["NVLED_CURR_MEAN"].abs(); a["P"] = a["PM400_POWER_MEAN"]
    a["E"] = a["P"] / a["I"]; a["V"] = a["NVLED_V"].round(4)
    a = a.sort_values("TIME_START")
    a["order"] = a.groupby("NVLED_V").cumcount()
    assert sorted(a[a["V"] == -1.0]["order"]) == [0, 1, 2, 3], "order must reset per bias"

    b = a.copy()
    b.loc[b["order"] > 0, "I"] *= 0.97          # exactly the 3 % the run shows
    b["E"] = b["P"] / b["I"]
    j = deficit(a, b)
    assert abs(j[j["order"] == 0]["dI"].median()) < 1e-9
    assert abs(j[j["order"] > 0]["dI"].median() + 3.0) < 1e-9, j["dI"].median()
    # and EQE must move the other way when only the current is cut
    assert j[j["order"] > 0]["dE"].median() > 0

    # a run with the pixels in a different ORDER must still pair correctly
    shuffled = b.sample(frac=1.0, random_state=1)
    assert np.allclose(sorted(deficit(a, shuffled)["dI"]), sorted(j["dI"]))
    # at_matched_current must undo a pure current shift on a power law, and must
    # refuse to extrapolate past the reference's own current range.
    I = np.geomspace(1e-7, 1e-4, 12)
    r = pd.DataFrame(dict(X=1, Y=3, V=-np.linspace(0.8, 2.6, 12), I=I,
                          P=3.0 * I ** 1.4, order=0))
    r["E"] = r["P"] / r["I"]
    q = r.copy()
    q["I"] = r["I"] * 0.97            # 3 % less current, same P(I) curve
    q["P"] = 3.0 * q["I"] ** 1.4
    q["order"] = 1
    m = at_matched_current(r, q)
    assert len(m) < len(q), "must drop the point that falls below the ref's own range"
    assert abs(m["dP_at_I"]).max() < 1e-6, m["dP_at_I"].abs().max()
    # and a real light loss at fixed current must survive
    q2 = q.copy(); q2["P"] *= 0.95
    assert abs(at_matched_current(r, q2)["dP_at_I"].median() + 5.0) < 1e-6
    assert started(Path("261015-0822_Array42_IQE_ROI_(32, 32, 63, 63)")) == \
        dt.datetime(2026, 10, 15, 8, 22)
    assert started(Path("not-a-run")) is None
    print("ok  plot_switch_deficit: pairs on (pixel, bias), order resets per bias, "
          "matched-current cancels a pure current shift, run times parse")


if __name__ == "__main__":
    if "--demo" in sys.argv:
        demo()
    else:
        main()
