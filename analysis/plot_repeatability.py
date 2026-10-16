"""How repeatable is each method? Re-measure ONE pixel N times and look at the spread.

    python plot_repeatability.py --old RUN RUN RUN --new RUN RUN RUN [--out DIR]

`compare_runs.py` shows old against new. It cannot tell you whether the difference
beats the noise, because one run of each has no error bar. This does: the spread
within a method is the bar the between-method difference has to clear.

One figure, three panels:

    repeatability.png
      left   EQE(J), every repeat drawn, so the within-method spread is visible
             against the between-method gap
      middle per-bias relative standard deviation of the current -- the precision
             claim, bias by bias
      right  peak IQE per repeat, with the faint-end optical power beside it

The third panel is the one that matters now: once the electrical side is fixed, what
is left of the scatter is the power meter at the faint end, not the SMU.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import analyze_array as aa          # noqa: E402
import compare_runs as cr           # noqa: E402

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

FAINT_V = -0.80         # the bias the whole argument is about


def summarize(runs: list[Path]) -> tuple[pd.DataFrame, list[pd.DataFrame]]:
    """Per-repeat scalars plus the enriched frames."""
    rows, frames = [], []
    for k, r in enumerate(runs, 1):
        df, _cfg = cr.load(r)
        frames.append(df)
        rel = df[df["RELIABLE"]] if "RELIABLE" in df else df
        f = aa.fit_abc(rel["J_LED"].values, rel["EQE_LED"].values)
        at = df[np.isclose(df["NVLED_V"], FAINT_V)]
        rows.append(dict(
            rep=k, run=r.name,
            peak_IQE=100 * f["peak_IQE"] if f else np.nan,
            I_faint_nA=1e9 * float(at["I_LED"].iloc[0]) if len(at) else np.nan,
            P_faint_pW=1e12 * float(at["P_W"].iloc[0]) if len(at) else np.nan,
            J_min=float(rel["J_LED"].min()), n_reliable=int(rel.shape[0]),
            n_points=int(df.shape[0])))
    return pd.DataFrame(rows), frames


def rsd(x) -> float:
    """Relative standard deviation in percent; NaN when it is not defined."""
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if x.size < 2 or not np.mean(x):
        return float("nan")
    return 100.0 * float(np.std(x, ddof=1)) / abs(float(np.mean(x)))


def per_bias(frames: list[pd.DataFrame], col: str):
    """Mean and relative sd of `col` at each bias the repeats share."""
    biases = sorted(set.intersection(*[set(np.round(f["NVLED_V"], 4)) for f in frames]))
    mean, spread = [], []
    for v in biases:
        vals = [float(f[np.isclose(f["NVLED_V"], v)][col].iloc[0]) for f in frames]
        mean.append(np.mean(vals))
        spread.append(rsd(vals))
    return np.array(biases), np.array(mean), np.array(spread)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--old", nargs="+", required=True, type=Path)
    ap.add_argument("--new", nargs="+", required=True, type=Path)
    ap.add_argument("--label-old", default="old method")
    ap.add_argument("--label-new", default="new method")
    ap.add_argument("--out", type=Path, default=None,
                    help="output directory (default: the parent of the first --new run)")
    args = ap.parse_args()

    out = args.out or args.new[0].parent.parent
    out.mkdir(parents=True, exist_ok=True)

    methods = {args.label_old: ("tab:red", args.old), args.label_new: ("tab:blue", args.new)}
    tables, frames = {}, {}
    for lbl, (_col, runs) in methods.items():
        tables[lbl], frames[lbl] = summarize(runs)
        print(f"\n=== {lbl} ({len(runs)} repeats) ===")
        print(tables[lbl].drop(columns=["run"]).to_string(
            index=False, float_format=lambda x: f"{x:.4g}"))
        for c in ("peak_IQE", "I_faint_nA", "P_faint_pW"):
            print(f"    {c:12s} mean {tables[lbl][c].mean():9.4g}"
                  f"  rel.sd {rsd(tables[lbl][c]):6.2f}%")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (a, b, c) = plt.subplots(1, 3, figsize=(15.5, 4.8))

    # 1. every repeat's EQE(J)
    for lbl, (col, _) in methods.items():
        for k, f in enumerate(frames[lbl]):
            r = f[f["RELIABLE"]] if "RELIABLE" in f else f
            r = r.sort_values("J_LED")
            a.semilogx(r["J_LED"], 100 * r["EQE_LED"], "o-", ms=3, lw=1, color=col,
                       alpha=0.75, label=lbl if k == 0 else None)
    a.set_xlabel("current density J (A/cm$^2$)")
    a.set_ylabel("EQE (%)")
    a.set_title("every repeat, same pixel\nthe within-method spread is the error bar",
                fontsize=10)
    a.legend(fontsize=8)
    a.grid(alpha=0.3, which="both")

    # 2. per-bias precision
    for lbl, (col, _) in methods.items():
        v, _m, s = per_bias(frames[lbl], "I_LED")
        b.semilogy(v, s, "o-", ms=4, color=col, label=lbl)
    b.axhline(1.0, color="k", lw=0.8, ls=":")
    b.set_xlabel("bias (V)")
    b.set_ylabel("relative sd of I$_{LED}$ over the repeats (%)")
    b.set_title("precision, bias by bias\nboth methods agree where the device is bright",
                fontsize=10)
    b.legend(fontsize=8)
    b.grid(alpha=0.3, which="both")

    # 3. peak IQE per repeat, and the faint-end power beside it
    w, offs = 0.35, {lbl: i for i, lbl in enumerate(methods)}
    for lbl, (col, _) in methods.items():
        t = tables[lbl]
        c.bar(t["rep"] + (offs[lbl] - 0.5) * w, t["peak_IQE"], w, color=col,
              alpha=0.8, label=lbl)
    c.set_xlabel("repeat")
    c.set_ylabel("peak IQE from the ABC fit (%)")
    c.set_xticks(sorted(tables[args.label_new]["rep"]))
    c2 = c.twinx()
    for lbl, (col, _) in methods.items():
        t = tables[lbl]
        c2.plot(t["rep"], t["P_faint_pW"], "s--", ms=6, color=col, mfc="none", lw=1)
    c2.set_ylabel(f"optical power at {FAINT_V:+.2f} V (pW), dashed")
    c.set_title("peak IQE per repeat (bars) and the faint-end power (dashed)\n"
                "the residual scatter tracks the power meter, not the SMU", fontsize=10)
    c.legend(fontsize=8, loc="lower right")
    c.grid(alpha=0.3, axis="y")

    fig.suptitle(f"Repeatability of one pixel: {len(args.old)} runs per method", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(out / "repeatability.png", dpi=300)
    plt.close(fig)

    lines = [f"# Repeatability: {len(args.old)} runs per method, one pixel", ""]
    for lbl in methods:
        t = tables[lbl]
        lines += [f"**{lbl}**", "",
                  f"- peak IQE: **{t['peak_IQE'].mean():.1f}% +/- "
                  f"{t['peak_IQE'].std(ddof=1):.1f}** ({rsd(t['peak_IQE']):.1f}% rel.sd)",
                  f"- I at {FAINT_V:+.2f} V: **{t['I_faint_nA'].mean():.1f} nA** "
                  f"({rsd(t['I_faint_nA']):.2f}% rel.sd)",
                  f"- P at {FAINT_V:+.2f} V: **{t['P_faint_pW'].mean():.0f} pW** "
                  f"({rsd(t['P_faint_pW']):.1f}% rel.sd)",
                  f"- lowest reliable J: **{t['J_min'].mean():.3f} A/cm2**, "
                  f"{t['n_reliable'].sum()}/{t['n_points'].sum()} points reliable", ""]
    lines += ["![repeatability](repeatability.png)", ""]
    (out / "REPEATABILITY.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"\nwrote {out / 'repeatability.png'} and REPEATABILITY.md")


def demo() -> None:
    """Self-check on the two statistics the figure stands on."""
    assert abs(rsd([100.0, 100.0, 100.0])) < 1e-12
    assert abs(rsd([90.0, 100.0, 110.0]) - 10.0) < 1e-9
    assert np.isnan(rsd([5.0]))                          # one sample, no sd
    assert np.isnan(rsd([1.0, np.nan]))                  # NaN dropped, then one sample
    assert np.isnan(rsd([-1.0, 1.0]))                    # mean 0, no relative sd

    f = lambda v, i: pd.DataFrame({"NVLED_V": v, "I_LED": i})
    v, m, s = per_bias([f([-1.0, -0.8], [10.0, 1.0]), f([-1.0, -0.8], [10.0, 3.0])], "I_LED")
    assert np.allclose(v, [-1.0, -0.8]), v
    assert np.allclose(m, [10.0, 2.0]), m
    assert abs(s[0]) < 1e-12 and abs(s[1] - 70.71) < 0.01, s
    v, _, _ = per_bias([f([-1.0, -0.8], [1.0, 1.0]), f([-1.0, -0.9], [1.0, 1.0])], "I_LED")
    assert np.allclose(v, [-1.0]), v                     # shared biases only
    print("ok  plot_repeatability: rsd and per_bias agree with the hand calculation")


if __name__ == "__main__":
    if "--demo" in sys.argv:
        demo()
    else:
        main()
