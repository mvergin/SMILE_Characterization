"""Put two runs of the SAME pixels side by side: old method vs new.

    python compare_runs.py OLD_RUN NEW_RUN [--out DIR] [--label-old ...] [--label-new ...]

Built for the demonstration: measure one ROI with `configs/legacy_pre261014.yaml` and
again with the defaults, then run this. It matches the two runs pixel by pixel and
shows what the method -- not the device -- changed.

Four figures and a short Markdown summary:

    eqe_vs_J.png      the headline: EQE(J) under each method, median over the pixels
    iv.png            the I-V, where the old method reads too high at low bias
    leakage.png       why: the fraction of channel B that is not the LED
    per_pixel.png     peak IQE per pixel, old against new, so it is not one average

It reads only what the runs contain, so an old run with no leakage reference is handled
(its I_LED is the total cathode current, which is the point being made).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import analyze_array as aa          # noqa: E402  (after the path fix)

AREA_CM2 = 100.0 * 1e-8


def load(run_dir: Path) -> tuple[pd.DataFrame, dict]:
    """Enriched, reliable sweep points plus the run's settings."""
    sweep = run_dir / "NVLEDSweep" / "Sweep"
    if not sweep.is_dir():
        sys.exit(f"{run_dir} has no NVLEDSweep/Sweep")
    df = aa._load_aggregated(sweep)
    try:
        dark = aa._dark_power(sweep)
    except Exception:
        dark = 0.0
    df = aa.enrich(df, dark_W=dark or 0.0)
    df = df[df["I_NVLED"].notna()].copy()
    df["X"] = df["X"].astype(int)
    df["Y"] = df["Y"].astype(int)
    return df, aa._load_config(sweep) or {}


def describe(cfg: dict) -> str:
    """One line naming the settings that actually differ between the methods."""
    def g(k, d="?"):
        v = cfg.get(k, d)
        return d if v is None else v
    return (f"High-C {'on' if cfg.get('highc_b') else 'OFF'}, "
            f"limit {g('nvled_compliance')} A, "
            f"settle {g('vramp_lowv_settle_s')}/{g('vramp_highv_settle_s')} s, "
            f"NPLC {g('vramp_lowv_nplc')}/{g('vramp_highv_nplc')}, "
            f"reference {'on' if cfg.get('leak_reference') else 'OFF'}")


def median_curve(df: pd.DataFrame, xcol: str, ycol: str):
    """Median y against x over pixels, at each bias. Biases are the common grid."""
    g = df.groupby("NVLED_V")
    v = np.array(sorted(g.groups))
    x = np.array([g.get_group(b)[xcol].median() for b in v])
    y = np.array([g.get_group(b)[ycol].median() for b in v])
    lo = np.array([g.get_group(b)[ycol].quantile(0.25) for b in v])
    hi = np.array([g.get_group(b)[ycol].quantile(0.75) for b in v])
    return v, x, y, lo, hi


def peak_iqe_per_pixel(df: pd.DataFrame) -> dict[tuple[int, int], float]:
    """Per-pixel peak EQE from the ABC fit, falling back to the data maximum."""
    out = {}
    for (x, y), g in df.groupby(["X", "Y"]):
        g = g[g["RELIABLE"]] if "RELIABLE" in g else g
        if len(g) < 6:
            continue
        f = aa.fit_abc(g["J_LED"].values, g["EQE_LED"].values)
        out[(x, y)] = float(f["peak_IQE"]) if f and np.isfinite(
            f.get("peak_IQE", np.nan)) else float("nan")
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("old_run")
    ap.add_argument("new_run")
    ap.add_argument("--out", default=None, help="output directory (default: NEW_RUN/comparison)")
    ap.add_argument("--label-old", default="old method")
    ap.add_argument("--label-new", default="new method")
    args = ap.parse_args()

    old_dir, new_dir = Path(args.old_run), Path(args.new_run)
    out = Path(args.out) if args.out else new_dir / "comparison"
    out.mkdir(parents=True, exist_ok=True)

    old, old_cfg = load(old_dir)
    new, new_cfg = load(new_dir)

    shared = sorted(set(map(tuple, old[["X", "Y"]].values))
                    & set(map(tuple, new[["X", "Y"]].values)))
    if not shared:
        sys.exit("the two runs share no pixels - were they measured on the same ROI "
                 "with the same block_seed?")
    keep = lambda d: d[[tuple(r) in set(shared) for r in d[["X", "Y"]].values]]
    old, new = keep(old), keep(new)
    print(f"{len(shared)} shared pixel(s)")
    print(f"  {args.label_old}: {describe(old_cfg)}")
    print(f"  {args.label_new}: {describe(new_cfg)}")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    C_OLD, C_NEW = "tab:red", "tab:blue"
    lines = [f"# {args.label_new} vs {args.label_old}", "",
             f"- **old:** `{old_dir.name}` — {describe(old_cfg)}",
             f"- **new:** `{new_dir.name}` — {describe(new_cfg)}",
             f"- **{len(shared)} pixels** measured by both", ""]

    # ---- 1. EQE vs J -------------------------------------------------------
    fig, ax = plt.subplots(figsize=(7.4, 5.0))
    for d, lbl, c in ((old, args.label_old, C_OLD), (new, args.label_new, C_NEW)):
        r = d[d["RELIABLE"]] if "RELIABLE" in d else d
        _, x, y, lo, hi = median_curve(r, "J_LED", "EQE_LED")
        m = np.isfinite(x) & np.isfinite(y) & (x > 0)
        ax.semilogx(x[m], 100 * y[m], "o-", ms=4, color=c, label=lbl)
        ax.fill_between(x[m], 100 * lo[m], 100 * hi[m], color=c, alpha=0.15)
    ax.set_xlabel("current density J (A/cm$^2$)")
    ax.set_ylabel("EQE (%)")
    ax.set_title(f"EQE against current density — same {len(shared)} pixels, two methods\n"
                 "band = inter-quartile range over the pixels", fontsize=10)
    ax.legend()
    ax.grid(alpha=0.3, which="both")
    fig.tight_layout()
    fig.savefig(out / "eqe_vs_J.png", dpi=300)
    plt.close(fig)

    # ---- 2. the I-V, where the methods disagree ----------------------------
    fig, (a, b) = plt.subplots(1, 2, figsize=(11.5, 4.4))
    curves = {}
    for d, lbl, c in ((old, args.label_old, C_OLD), (new, args.label_new, C_NEW)):
        v = np.array(sorted(d["NVLED_V"].unique()))
        i = np.array([d[d["NVLED_V"] == x]["I_LED"].median() for x in v])
        curves[lbl] = (v, i)
        a.semilogy(v, i * 1e9, "o-", ms=4, color=c, label=lbl)
    a.set_xlabel("bias (V)")
    a.set_ylabel("|I| taken as the LED current (nA)")
    a.set_title("the I-V each method reports", fontsize=10)
    a.legend(fontsize=8)
    a.grid(alpha=0.3, which="both")

    vo, io = curves[args.label_old]
    vn, inn = curves[args.label_new]
    common = np.intersect1d(np.round(vo, 3), np.round(vn, 3))
    if common.size:
        ro = np.array([io[np.argmin(np.abs(vo - v))] for v in common])
        rn = np.array([inn[np.argmin(np.abs(vn - v))] for v in common])
        b.semilogy(common, ro / rn, "o-", color="tab:purple")
        b.axhline(1.0, color="k", lw=0.8)
        b.set_xlabel("bias (V)")
        b.set_ylabel(f"{args.label_old} / {args.label_new}")
        b.set_title("how much higher the old method reads", fontsize=10)
        b.grid(alpha=0.3, which="both")
        worst = float(np.nanmax(ro / rn))
        lines += [f"- the old method reads up to **{worst:.1f}x** more current at the "
                  f"faint end (bias {common[int(np.nanargmax(ro / rn))]:+.3f} V)", ""]
    else:
        b.text(0.5, 0.5, "no shared bias points\n(different n_fine)", ha="center")
        b.axis("off")
    fig.suptitle("Current: the same pixels, measured two ways", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(out / "iv.png", dpi=300)
    plt.close(fig)

    # ---- 3. leakage, which explains the difference -------------------------
    if "LEAK_FRAC" in new and new["LEAK_FRAC"].notna().any():
        fig, ax = plt.subplots(figsize=(7.4, 4.6))
        v = np.array(sorted(new["NVLED_V"].unique()))
        lf = np.array([100 * new[new["NVLED_V"] == x]["LEAK_FRAC"].median() for x in v])
        j = np.array([new[new["NVLED_V"] == x]["J_LED"].median() for x in v])
        ax.semilogx(j, lf, "o-", color="tab:red")
        ax.set_xlabel("current density J (A/cm$^2$)")
        ax.set_ylabel("backplane leakage, % of the raw channel-B reading")
        ax.set_title("Why the old method reads too high: most of the low-bias\n"
                     "current is not the LED (measured by the new method)", fontsize=10)
        ax.grid(alpha=0.3, which="both")
        fig.tight_layout()
        fig.savefig(out / "leakage.png", dpi=300)
        plt.close(fig)
        lines += [f"- at the faint end, **{lf[np.argmin(j)]:.0f}%** of channel B is "
                  f"backplane leakage, not LED current", ""]

    # ---- 4. per-pixel peak IQE --------------------------------------------
    po, pn = peak_iqe_per_pixel(old), peak_iqe_per_pixel(new)
    both = [k for k in shared if np.isfinite(po.get(k, np.nan))
            and np.isfinite(pn.get(k, np.nan))]
    if both:
        xo = np.array([100 * po[k] for k in both])
        xn = np.array([100 * pn[k] for k in both])
        fig, ax = plt.subplots(figsize=(6.0, 5.6))
        ax.scatter(xo, xn, s=28, color="tab:blue")
        lim = [0, max(100, xo.max(), xn.max()) * 1.05]
        ax.plot(lim, lim, "k--", lw=1, label="no change")
        ax.set_xlim(lim)
        ax.set_ylim(lim)
        ax.set_xlabel(f"peak IQE, {args.label_old} (%)")
        ax.set_ylabel(f"peak IQE, {args.label_new} (%)")
        ax.set_title(f"Peak IQE per pixel ({len(both)} pixels)\n"
                     "every point above the line is a pixel the old method understated",
                     fontsize=10)
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(out / "per_pixel.png", dpi=300)
        plt.close(fig)
        lines += [f"- peak IQE, median over {len(both)} pixels: "
                  f"**{np.median(xo):.0f}%** ({args.label_old}) -> "
                  f"**{np.median(xn):.0f}%** ({args.label_new})",
                  f"- clean fits: {len(po)}/{len(shared)} old, {len(pn)}/{len(shared)} new", ""]

    lines += ["", "![EQE](eqe_vs_J.png)", "", "![I-V](iv.png)", "",
              "![leakage](leakage.png)", "", "![per pixel](per_pixel.png)", ""]
    (out / "COMPARISON.md").write_text("\n".join(lines), encoding="utf-8")
    for line in lines[:12]:
        if line.startswith("- "):
            print("  " + line[2:].replace("**", ""))
    print(f"\nwrote {out / 'COMPARISON.md'} and 4 figures")


if __name__ == "__main__":
    main()
