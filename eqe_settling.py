"""Does EQE = P/I settle faster than P or I on their own?

    python eqe_settling.py RUN [RUN ...] [--out DIR]

The open question since the dwell was first measured. If the optical power and the
current relax together -- the same charge redistribution moving both -- then their
ratio is settled long before either factor is, and the dwell could be cut to whatever
the ratio needs. If they relax independently, the ratio settles last, not first.

Every swept point in a run with `trace_settling` on records `(t, |I|, P)` through its
own dwell in `settling_traces.npz`. This applies the SAME settling criterion used by
`measure_iqe --calibrate` to all three signals and compares them.

    eqe_settling.png     settle time of I, P and P/I per trace, and the ratio
    EQE_SETTLING.md      the verdict with the numbers behind it
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

# The one definition of "settled". Importing it, rather than copying it, is the whole
# point: a third implementation would drift from the other two the way the second did.
for cand in (Path(__file__).resolve().parent.parent / "SMILE_Characterization",
             Path(r"C:\_D\programs\SMILE_Characterization")):
    if (cand / "measure_iqe" / "calibrate.py").is_file():
        sys.path.insert(0, str(cand))
        break
from measure_iqe.calibrate import settle_time      # noqa: E402


def traces(run: Path):
    """Yield (key, t, I, P) for every dwell trace under a run directory."""
    for npz in sorted(run.glob("**/settling_traces.npz")):
        z = np.load(npz)
        for key in z.files:
            a = np.asarray(z[key], float)
            if a.ndim == 2 and a.shape[0] >= 3 and a.shape[1] >= 8:
                yield key, a[0], np.abs(a[1]), a[2]


DWELLS = (1.0, 2.0, 5.0, 10.0, 20.0)    # candidate shortcuts, in seconds
FAINT_FACTOR = 2.0                      # "the faint end" = within this of the dimmest point


def bias_at(t, y, dwells=DWELLS):
    """Fractional error of `y` if the dwell stopped at each of `dwells`.

    The settling criterion asks "is the drift bigger than the noise". That is the
    wrong question for a shortcut: a 5 % systematic error is still a 5 % error when
    the noise is 8 %. This asks the right one -- how far from its own final value
    the reading would be -- and it is the number that rules a short dwell out.
    """
    fin = float(np.median(y[-max(3, y.size // 5):]))
    if not fin:
        return [float("nan")] * len(dwells)
    return [float(y[int(np.argmin(np.abs(t - d)))] / fin - 1.0) if d <= t[-1]
            else float("nan") for d in dwells]


def row(key: str, t, i, p) -> dict:
    """Settle time of each signal, plus what the signals were."""
    ok = np.isfinite(t) & np.isfinite(i) & np.isfinite(p) & (i > 0)
    t, i, p = t[ok], i[ok], p[ok]
    if t.size < 8:
        return {}
    eqe = p / i
    r = dict(key=key, n=t.size, record_s=float(t[-1]),
             I_nA=1e9 * float(np.median(i[-max(3, i.size // 5):])),
             P_pW=1e12 * float(np.median(p[-max(3, p.size // 5):])),
             t_I=settle_time(t, i), t_P=settle_time(t, p), t_EQE=settle_time(t, eqe))
    for d, bi, be in zip(DWELLS, bias_at(t, i), bias_at(t, eqe)):
        r[f"dI_{d:g}s"], r[f"dEQE_{d:g}s"] = bi, be
    return r


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("runs", nargs="+", type=Path)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()
    out = args.out or args.runs[0]
    out.mkdir(parents=True, exist_ok=True)

    rows = [r for run in args.runs for k, t, i, p in traces(run) if (r := row(k, t, i, p))]
    if not rows:
        sys.exit("no settling traces found - was the run measured with trace_settling on?")
    df = pd.DataFrame(rows)

    # A trace that never settles is a lower bound at the record length, not a missing
    # value. Dropping it would bias every mean towards the traces that settled fast.
    for c in ("t_I", "t_P", "t_EQE"):
        df[c + "_b"] = df[c].fillna(df["record_s"])
    df["ratio"] = df["t_EQE_b"] / df["t_I_b"]

    print(f"{len(df)} dwell traces from {len(args.runs)} run(s), "
          f"record {df['record_s'].min():.1f}-{df['record_s'].max():.1f} s")
    print(f"{'signal':8} {'settled':>9} {'median s':>9} {'mean s':>8} {'worst s':>8}")
    for c, name in (("t_I", "I"), ("t_P", "P"), ("t_EQE", "P/I")):
        print(f"{name:8} {df[c].notna().sum():>4}/{len(df):<4} "
              f"{df[c + '_b'].median():>9.2f} {df[c + '_b'].mean():>8.2f} "
              f"{df[c + '_b'].max():>8.2f}")
    faster = float((df["ratio"] < 1).mean())
    print(f"\nP/I settles sooner than I in {100 * faster:.0f}% of traces "
          f"(median ratio {df['ratio'].median():.2f})")

    # The criterion can say "settled" while a systematic error is still there, if the
    # signal is noisy enough to hide it. Faintest decade only -- that is what sets the
    # dwell, and it is where a shortcut would actually be taken.
    faint = df[df["I_nA"] <= FAINT_FACTOR * df["I_nA"].min()]
    print(f"\nif the dwell stopped early, the error left in the reading "
          f"({len(faint)} faintest traces, |I| <= {FAINT_FACTOR * df['I_nA'].min():.0f} nA):")
    print(f"{'dwell':>7} {'current':>12} {'EQE = P/I':>12}")
    for d in DWELLS:
        ci, ce = faint[f"dI_{d:g}s"].median(), faint[f"dEQE_{d:g}s"].median()
        print(f"{d:>6.0f}s {100 * ci:>11.1f}% {100 * ce:>11.1f}%")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (a, b, ax3) = plt.subplots(1, 3, figsize=(16.2, 4.8))
    lo = max(1e-3, min(df[c + "_b"].min() for c in ("t_I", "t_P", "t_EQE")))
    hi = max(df[c + "_b"].max() for c in ("t_I", "t_P", "t_EQE")) * 1.3
    a.loglog([lo, hi], [lo, hi], "k--", lw=1, label="equal")
    a.loglog(df["t_I_b"], df["t_EQE_b"], "o", ms=5, alpha=0.6, color="tab:blue",
             label="P/I against I")
    a.loglog(df["t_I_b"], df["t_P_b"], "s", ms=4, alpha=0.5, color="tab:orange",
             label="P against I")
    a.set_xlabel("settle time of the current (s)")
    a.set_ylabel("settle time of the other signal (s)")
    a.set_title("below the line settles sooner than the current\n"
                "unsettled traces are plotted at the record length", fontsize=10)
    a.legend(fontsize=8)
    a.grid(alpha=0.3, which="both")

    s = df.sort_values("I_nA")
    for c, name, col in (("t_I_b", "I", "tab:red"), ("t_P_b", "P", "tab:orange"),
                         ("t_EQE_b", "P/I", "tab:blue")):
        b.loglog(s["I_nA"], s[c], "o", ms=4, alpha=0.6, color=col, label=name)
    b.set_xlabel("settled current (nA)")
    b.set_ylabel("settle time (s)")
    b.set_title("the dwell is set by the faint end, whichever signal you watch",
                fontsize=10)
    b.legend(fontsize=8)
    b.grid(alpha=0.3, which="both")

    for sig, col, name in (("dI", "tab:red", "current"), ("dEQE", "tab:blue", "EQE = P/I")):
        med = [100 * abs(faint[f"{sig}_{d:g}s"].median()) for d in DWELLS]
        ax3.semilogy(DWELLS, med, "o-", ms=5, color=col, label=name)
    ax3.axhline(1.0, color="k", lw=0.8, ls=":")
    ax3.set_xlabel("dwell the sweep would stop at (s)")
    ax3.set_ylabel("error left in the reading (%)")
    ax3.set_title(f"what a shortcut actually costs\n{len(faint)} faintest traces, "
                "median absolute error", fontsize=10)
    ax3.legend(fontsize=8)
    ax3.grid(alpha=0.3, which="both")

    fig.suptitle("Does the EQE ratio settle before its two factors?", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(out / "eqe_settling.png", dpi=300)
    plt.close(fig)

    # Two tests, and the shortcut needs both. The criterion can be fooled by a noisy
    # factor; the residual error cannot.
    tol = 0.01
    short = min((d for d in DWELLS
                 if abs(faint[f"dEQE_{d:g}s"].median()) < tol), default=None)
    verdict = ("**the ratio settles no sooner than the current**, so the dwell cannot be "
               "shortened by watching EQE instead of I"
               if df["ratio"].median() >= 0.9 else
               "**the criterion says the ratio settles first**, but see the residual "
               "error below before believing it")
    verdict += (f" -- and no dwell in {DWELLS[0]:g}-{DWELLS[-1]:g} s brings the EQE within "
                f"{100 * tol:g} % of its final value" if short is None else
                f", and a {short:g} s dwell already holds the EQE to {100 * tol:g} %")
    lines = [f"# Does EQE = P/I settle before P or I? ({len(df)} dwell traces)", "",
             verdict, "",
             f"| signal | settled within the record | median | mean | worst |",
             f"|---|---|---|---|---|"]
    for c, name in (("t_I", "I"), ("t_P", "P"), ("t_EQE", "P/I")):
        lines.append(f"| {name} | {df[c].notna().sum()}/{len(df)} | "
                     f"{df[c + '_b'].median():.2f} s | {df[c + '_b'].mean():.2f} s | "
                     f"{df[c + '_b'].max():.2f} s |")
    lines += ["", f"- P/I beats I in **{100 * faster:.0f}%** of traces, median ratio "
                  f"**{df['ratio'].median():.2f}**",
              "- traces that never settled count as the record length, not as missing",
              "",
              f"## What a short dwell would cost ({len(faint)} faintest traces)", "",
              "The criterion above asks whether the drift beats the noise. That is the "
              "wrong question for a shortcut: a systematic error does not stop being one "
              "because the noise is larger. Median error left in the reading:", "",
              "| dwell | current | EQE = P/I |", "|---|---|---|"]
    for d in DWELLS:
        lines.append(f"| {d:g} s | {100 * faint[f'dI_{d:g}s'].median():+.1f} % | "
                     f"{100 * faint[f'dEQE_{d:g}s'].median():+.1f} % |")
    lines += ["", "![settling](eqe_settling.png)", ""]
    (out / "EQE_SETTLING.md").write_text("\n".join(lines), encoding="utf-8")
    df.to_csv(out / "eqe_settling.csv", index=False)
    print(f"wrote {out / 'eqe_settling.png'}, EQE_SETTLING.md and eqe_settling.csv")


def demo() -> None:
    """Self-check: a known-settled and a known-drifting signal come out right."""
    t = np.linspace(0, 20, 400)
    fast = 1e-6 * (1 + 0.5 * np.exp(-t / 0.2))          # settled within a second
    slow = 1e-6 * (1 + 0.5 * (1 + t) ** -0.3)           # still crawling at 20 s
    r = row("x", t, fast, fast * 1e-4)
    assert r and r["t_I"] < 3.0, r
    assert abs(r["I_nA"] - 1000.0) < 5, r
    r = row("y", t, slow, slow * 1e-4)
    assert np.isnan(r["t_I"]) or r["t_I"] > 5.0, r      # drifting: NaN or late

    # a ratio of two identically relaxing signals is flat from the first sample
    r = row("z", t, slow, slow * 1e-4)
    assert r["t_EQE"] < 1.0 or np.isnan(r["t_EQE"]) is False, r
    assert not row("short", t[:4], fast[:4], fast[:4]), "too few samples must be dropped"
    print("ok  eqe_settling: settled, drifting and too-short traces classified correctly")


if __name__ == "__main__":
    if "--demo" in sys.argv:
        demo()
    else:
        main()
