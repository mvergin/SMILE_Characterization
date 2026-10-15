"""What the 2 s pixel-switch wait costs, measured by the sweep rather than by a capture.

    python plot_switch_deficit.py PIXEL_MAJOR_RUN BIAS_MAJOR_RUN [BIAS_MAJOR_RUN ...]
                                 [--label L ...] [--out DIR]

`bias_major` gives the first pixel at each bias the full 54.47 s dwell with its own
frame lit, and every later pixel only `pixel_switch_settle_s`. That split is the whole
experiment, and it is already in any bias-major run: compare each position against the
pixel-major run on the same pixels.

    switch_deficit.png   left  : deficit against bias, order 0 vs the rest
                         right : deficit against position in the bias's pixel list

Run 261015-0822 on section 13's 16 pixels, median over -1.25..-0.80 V:

    the pixel that holds the frame through the dwell   -0.70 % I   -0.24 % EQE
    every pixel that gets the 2 s switch wait          -3.0  % I   -1.9  % EQE

and the right panel is flat from position 1 to 15, so it is a transient every switch
resets rather than drift accumulating through the bias.

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
        j = deficit(ref, load(r))
        summarize(j, lbl)
        js.append((lbl, j))

    fig, ax = plt.subplots(1, 2, figsize=(13, 4.8))
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
              title=f"flat from 1 to the end — a per-switch transient,\n"
                    f"not drift ({FAINT_LO} to {FAINT_HI} V)")
    ax[1].legend(fontsize=7)
    ax[1].grid(alpha=.3)
    fig.suptitle("The 2 s pixel-switch wait is too short, and the dedicated switch "
                 "experiment could not see it", y=1.02)
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
    print("ok  plot_switch_deficit: pairs on (pixel, bias), order resets per bias")


if __name__ == "__main__":
    if "--demo" in sys.argv:
        demo()
    else:
        main()
