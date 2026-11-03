"""Pixel classes across biases: between the fast-scan biases, and from the fast scan into
every bias of the run's sweep.

    python analysis/pixel_classes.py "results/<run>"

The classes and their rules are analyze_array's (PIXEL_CLASSES, classify_pixels). This
reads the run's own data, not analyze_array's output, so it runs without it.

REPORT_assets/pixelclassification/
  fastscan/class_transitions_<a>_to_<b>.png/.csv   class at the first fast-scan bias
      (rows) against each other one (columns): count, and share of the row
  sweep/                                           only with NVLEDSweep/
    sweep_classes.csv          per pixel: fast-scan class, then the class at each bias
    summary_by_bias.csv        counts per class and bias; which biases were classified
    fastscan_to_sweep_<v>.png/.csv   the transition matrix at each classified bias
    persistence.png/.csv       share of each fast-scan class still flagged for it
    class_counts.png           pixels per class against bias
    class_curves.png/.csv      median I, P and P/I against bias per fast-scan class

A sweep pixel is classified on its leakage-free current and light (lit minus the
bitval-0 reference), dark below 5 x the light's noise (the OPTICAL_SNR floor of that
point). Only biases where at least half the points are lit are classified: below the
LED threshold every pixel is dark and would read as "open". The overcurrent threshold
is 1.2 x the shorts' median current at that bias, from a transistoriv run on this run's
shorts when one exists (the fast scan's own rule), else 1.2 x the median sweep current.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import analyze_array as aa

CLASSES = list(aa.PIXEL_CLASSES)
LIT_MIN_FRAC = 0.5      # classify a sweep bias only when this share of points is lit


def fastscan_classes(run: Path) -> dict:
    """{bias: classify_pixels frame indexed by (X, Y)}, the classifying bias first."""
    fs_dir = run / "FastScan"
    fs = aa._load_aggregated(fs_dir)
    if fs is None:
        return {}
    fs, extra = aa.split_fastscan_biases(fs, aa._load_config(fs_dir))
    out = {}
    for f in (fs, *extra.values()):
        d = aa.classify_pixels(f)
        out[d.attrs["v"]] = d.set_index(["X", "Y"])
    return out


def load_sweep(run: Path, group: str = "Sweep") -> pd.DataFrame | None:
    """One row per pixel and bias: |I| total, |I| of the bitval-0 reference (NaN where
    none was taken), I_LED = their difference (unclipped), P lit minus reference, and the
    noise of that difference. Only the columns needed: the CSV runs to 1.2 GB."""
    files = sorted((run / "NVLEDSweep" / group).glob("*aggregated_bitval=*.csv"))
    if not files:
        return None
    head = set(pd.read_csv(files[0], nrows=0).columns)

    def pick(base):
        return f"{base}_MEDIAN" if f"{base}_MEDIAN" in head else f"{base}_MEAN"

    i_c, p_c = pick("NVLED_CURR"), pick("PM400_POWER")
    want = {"X", "Y", "QUADRANT", "NVLED_V", "POOL_K", i_c, p_c, "NVLED_CURR_REF",
            "NVLED_CURR_SEM", "PM400_POWER_REF", "PM400_POWER_SEM", "PM400_POWER_REF_SEM"}
    df = pd.concat([pd.read_csv(f, usecols=lambda c: c in want) for f in files],
                   ignore_index=True)
    if "POOL_K" in df:                       # a pooled target is not one pixel
        df = df[df["POOL_K"].fillna(1) <= 1]
    f32 = lambda c: (df[c].astype("float32") if c in df else
                     pd.Series(np.nan, index=df.index, dtype="float32"))
    out = pd.DataFrame({"X": df["X"].astype("int16"), "Y": df["Y"].astype("int16"),
                        "QUADRANT": aa._quadrant_col(df).values,
                        "V": df["NVLED_V"].astype(float).round(4)})
    out["I_tot"] = f32(i_c).abs()
    out["I_ref"] = f32("NVLED_CURR_REF").abs()
    out["I_LED"] = out["I_tot"] - out["I_ref"]
    out["I_sem"] = f32("NVLED_CURR_SEM").abs()
    out["P"] = f32(p_c) - f32("PM400_POWER_REF").fillna(0)
    out["P_noise"] = np.hypot(f32("PM400_POWER_SEM").abs(),
                              f32("PM400_POWER_REF_SEM").abs().fillna(0))
    return out


def shorts_iv(run: Path) -> pd.DataFrame | None:
    """The transistoriv runs, next to this one, whose pixel list came from this run."""
    parts = []
    for d in sorted(run.parent.iterdir()):
        if not (d / "transistoriv.csv").is_file():
            continue
        try:
            src = (json.loads((d / "run_config.json").read_text(encoding="utf-8"))
                   .get("args", {}).get("pixel_list") or "")
        except (OSError, ValueError):
            continue
        if run.name in src:
            t = pd.read_csv(d / "transistoriv.csv")
            t["RUN"] = d.name
            parts.append(t)
    return pd.concat(parts, ignore_index=True) if parts else None


def main_run(tiv: pd.DataFrame) -> pd.DataFrame:
    """The rows of the run with the most shorts. Runs disagree: 261028-2156 started at
    0 V and reads 12.9 uA there at bitval 15, 261029-0836 came down from +1 V and reads
    83.8 uA; pooling them gives a driver curve no run measured."""
    n = tiv.groupby("RUN").apply(lambda g: g[["X", "Y"]].drop_duplicates().shape[0])
    return tiv[tiv["RUN"] == n.idxmax()]


def plot_transitions(a, b, la, lb, out: Path, plt, title=""):
    """a, b: the primary class per pixel, indexed (X, Y). Writes out.csv (counts, with
    totals) and out.png: rows a's class, columns b's; colour = share of the row."""
    both = pd.concat([a.rename(la), b.rename(lb)], axis=1, join="inner").dropna()
    pd.crosstab(both[la], both[lb], margins=True).to_csv(out.with_suffix(".csv"))
    ct = pd.crosstab(both[la], both[lb])
    rows = [c for c in CLASSES if c in ct.index]
    cols = [c for c in CLASSES if c in ct.columns]
    ct = ct.reindex(index=rows, columns=cols, fill_value=0)
    pct = 100 * ct.div(ct.sum(axis=1), axis=0)
    fig, ax = plt.subplots(figsize=(1.05 * len(cols) + 3.5, 0.62 * len(rows) + 2.2))
    im = ax.imshow(pct.values, cmap="Blues", vmin=0, vmax=100, aspect="auto")
    for i in range(len(rows)):
        for j in range(len(cols)):
            if ct.values[i, j]:
                ax.text(j, i, f"{ct.values[i, j]}\n{pct.values[i, j]:.1f} %", ha="center",
                        va="center", fontsize=7,
                        color="white" if pct.values[i, j] > 60 else "black")
    ax.set_xticks(range(len(cols)), cols, rotation=35, ha="right")
    ax.set_yticks(range(len(rows)), [f"{r} ({ct.loc[r].sum()})" for r in rows])
    ax.set_xlabel(f"class at {lb}")
    ax.set_ylabel(f"class at {la}")
    keep = sum(int(ct.loc[c, c]) for c in rows if c in cols)
    n = int(ct.values.sum())
    ax.set_title(f"{title}\n{keep} of {n} pixels ({100 * keep / max(n, 1):.1f} %) keep "
                 f"their primary class", fontsize=9)
    fig.colorbar(im, ax=ax, label="% of the row", shrink=0.8)
    fig.tight_layout()
    fig.savefig(out.with_suffix(".png"), dpi=150)
    plt.close(fig)


def fastscan_transitions(fsc: dict, base: Path, plt, log):
    vs = list(fsc)
    for v in vs[1:]:
        tag = f"{aa._bias_tag(vs[0])}_to_{aa._bias_tag(v)}"
        plot_transitions(fsc[vs[0]]["primary"], fsc[v]["primary"], f"{vs[0]:+.2f} V",
                         f"{v:+.2f} V", base / f"class_transitions_{tag}", plt,
                         f"Fast-scan class at {vs[0]:+.2f} V against {v:+.2f} V")
        log(f"  wrote fastscan/class_transitions_{tag}.png")


def i_max_by_bias(tiv: pd.DataFrame | None, bitval: int = 15) -> dict:
    """1.2 x the shorts' median current per bias, the fast scan's overcurrent rule."""
    if tiv is None:
        return {}
    t = main_run(tiv)
    t = t[t["BITVAL"] == bitval]
    med = t.groupby(t["NVLED_V"].round(4))["NVLED_CURR_MEAN"].apply(lambda s: s.abs().median())
    return {float(v): max(aa.I_MAX_A, aa.SHORT_X * float(i)) for v, i in med.items()}


def sweep_classes(sw: pd.DataFrame, fsc: dict, i_max: dict, base: Path, plt, log):
    base.mkdir(parents=True, exist_ok=True)
    v_fs = next(iter(fsc))
    fs_cls = fsc[v_fs]["primary"]
    px = sw.drop_duplicates(["X", "Y"]).set_index(["X", "Y"])[["QUADRANT"]]
    wide = px.join(fs_cls.rename(f"fastscan_{aa._bias_tag(v_fs)}"))
    fs_in = wide.iloc[:, -1]
    summ, pers, flags = [], {}, {}
    for v in sorted(sw["V"].unique()):
        s = sw[sw["V"] == v]
        floor = aa.OPT_SNR_MIN * s["P_noise"].astype(float)
        lit = float((s["P"] > floor).mean())
        has_ref = bool(s["I_ref"].notna().any())
        if not has_ref or lit < LIT_MIN_FRAC:
            summ.append(dict(v=v, status="no reference" if not has_ref else
                             f"below threshold ({100 * lit:.0f} % lit)", lit_frac=lit))
            continue
        frame = pd.DataFrame({"X": s["X"].values, "Y": s["Y"].values,
                              "QUADRANT": s["QUADRANT"].values,
                              "NVLED_CURR_MEAN": s["I_LED"].clip(lower=0).values,
                              "PM400_POWER_MEAN": s["P"].clip(lower=0).values,
                              "NVLED_V": v})
        # the shorts are measured in 0.1 V steps: interpolate for -0.95 V and the like
        vs = sorted(i_max)
        im = (float(np.interp(v, vs, [i_max[k] for k in vs]))
              if vs and vs[0] <= v <= vs[-1] else None)
        src = "shorts (transistoriv)"
        if im is None:
            im, src = max(aa.I_MAX_A, aa.SHORT_X * float(s["I_LED"].median())), "1.2 x median"
        d = aa.classify_pixels(frame, floor_W=floor.values, i_max_A=im).set_index(["X", "Y"])
        tag = aa._bias_tag(v)
        wide[tag] = d["primary"]
        flags[v] = d
        n = d["primary"].value_counts()
        summ.append(dict(v=v, status="classified", lit_frac=lit,
                         floor_W_median=d.attrs["floor_W"], i_max_A=im, i_max_from=src,
                         **{f"n_{c}": int(n.get(c, 0)) for c in CLASSES}))
        plot_transitions(fs_in, d["primary"], f"fast scan {v_fs:+.2f} V", f"sweep {v:+.2f} V",
                         base / f"fastscan_to_sweep_{tag}", plt,
                         f"Fast-scan class ({v_fs:+.2f} V) against the sweep's at {v:+.2f} V")
        pers[v] = {c: (float(d.reindex(fs_in.index[fs_in == c])[f"is_{c}"].astype(float).mean())
                       if (fs_in == c).any() else np.nan) for c in CLASSES}
        log(f"  sweep {v:+.2f} V: {100 * lit:.0f} % lit, overcurrent > {im * 1e6:.0f} uA "
            f"({src}): " + ", ".join(f"{c} {n[c]}" for c in CLASSES if n.get(c, 0)))
    wide.reset_index().to_csv(base / "sweep_classes.csv", index=False)
    pd.DataFrame(summ).to_csv(base / "summary_by_bias.csv", index=False)
    if not pers:
        log("  no sweep bias lit enough to classify")
        return wide
    p = pd.DataFrame(pers).reindex(CLASSES) * 100
    n_fs = fs_in.value_counts()
    p.insert(0, "n_fastscan", [int(n_fs.get(c, 0)) for c in p.index])
    p.to_csv(base / "persistence.csv", float_format="%.2f")
    _plot_persistence(p, v_fs, base / "persistence.png", plt)
    _plot_counts(pd.DataFrame(summ), fs_in, v_fs, base / "class_counts.png", plt)
    return wide


def _plot_persistence(p, v_fs, out, plt):
    p = p[p["n_fastscan"] > 0]
    vals = p.drop(columns="n_fastscan")
    fig, ax = plt.subplots(figsize=(0.75 * vals.shape[1] + 3.5, 0.6 * len(p) + 2.2))
    im = ax.imshow(vals.values, cmap="viridis", vmin=0, vmax=100, aspect="auto")
    for i in range(vals.shape[0]):
        for j in range(vals.shape[1]):
            x = vals.values[i, j]
            if np.isfinite(x):
                ax.text(j, i, f"{x:.0f}", ha="center", va="center", fontsize=7,
                        color="black" if x > 60 else "white")
    ax.set_xticks(range(vals.shape[1]), [f"{v:+.2f}" for v in vals.columns], rotation=45)
    ax.set_yticks(range(len(p)), [f"{c} ({int(n)})" for c, n in p["n_fastscan"].items()])
    for j, v in enumerate(vals.columns):
        if abs(v - v_fs) < 1e-6:
            ax.add_patch(plt.Rectangle((j - 0.5, -0.5), 1, len(p), fill=False, ec="red", lw=1.5))
    ax.set_xlabel("sweep bias (V); red: the fast scan's bias")
    ax.set_ylabel(f"fast-scan class at {v_fs:+.2f} V (pixels in the sweep)")
    ax.set_title("Share of each fast-scan class still flagged for it in the sweep (%).\n"
                 "Flag, not primary class: a pixel can be in several.", fontsize=9)
    fig.colorbar(im, ax=ax, label="%", shrink=0.8)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def _plot_counts(summ, fs_in, v_fs, out, plt):
    s = summ[summ["status"] == "classified"].sort_values("v")
    n_fs = fs_in.value_counts()
    fig, ax = plt.subplots(figsize=(10, 5.5))
    for c in CLASSES:
        y = s[f"n_{c}"].values
        if not (y > 0).any() and not n_fs.get(c, 0):
            continue
        col = aa.PIXEL_CLASSES[c][0]
        ax.plot(s["v"], np.where(y > 0, y, np.nan), "o-", ms=3, color=col, label=c)
        if n_fs.get(c, 0):
            ax.plot([v_fs], [n_fs[c]], "*", ms=12, color=col, mec="k", mew=0.5)
    ax.set_yscale("log")
    ax.set_xlabel("NVLED bias (V)")
    ax.set_ylabel("pixels, primary class")
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=8, ncol=2)
    ax.set_title(f"Pixels per class against sweep bias (stars: the fast scan at {v_fs:+.2f} V, "
                 f"the same pixels)", fontsize=10)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def class_curves(sw: pd.DataFrame, fs_in: pd.Series, v_fs, base: Path, plt):
    """Median and quartiles of I, P and P/I against bias, per fast-scan class. I is the
    leakage-free I_LED where the bias has a reference, else the total current."""
    cls = fs_in.rename("cls").reset_index()
    m = sw.merge(cls, on=["X", "Y"], how="inner")
    m["has_ref"] = m["I_ref"].notna()
    # P/I only where the light is above its noise: below threshold it is noise / noise
    m["PI"] = (m["P"] / m["I_LED"].where(m["I_LED"] > 0)).where(
        m["P"] > aa.OPT_SNR_MIN * m["P_noise"])
    q = lambda p: (lambda s: s.quantile(p))
    g = m.groupby(["cls", "V"])
    t = pd.DataFrame({"n": g.size(), "has_ref": g["has_ref"].all()})
    for c in ("I_LED", "I_tot", "P", "PI"):
        t[f"{c}_p25"], t[f"{c}_med"], t[f"{c}_p75"] = (g[c].agg(q(.25)), g[c].median(),
                                                       g[c].agg(q(.75)))
    t = t.reset_index()
    t.to_csv(base / "class_curves.csv", index=False)
    fig, ax = plt.subplots(1, 3, figsize=(18, 5.5))
    lims = {0: [], 1: [], 2: []}
    for c in CLASSES:
        tc = t[t["cls"] == c].sort_values("V")
        if tc.empty:
            continue
        col = aa.PIXEL_CLASSES[c][0]
        n = int(tc["n"].max())
        for i, k, sc in ((0, "I_LED", 1e6), (1, "P", 1e9), (2, "PI", 1e3)):
            ok = tc[f"{k}_med"] > 0
            y = tc.loc[ok, f"{k}_med"] * sc
            ax[i].plot(tc.loc[ok, "V"], y, "o-", ms=3, lw=1, color=col, label=f"{c} ({n})")
            ax[i].fill_between(tc.loc[ok, "V"], (tc.loc[ok, f"{k}_p25"] * sc).clip(
                lower=y.min() / 10), tc.loc[ok, f"{k}_p75"] * sc, color=col, alpha=0.15, lw=0)
            lims[i] += list(y)
        ax[0].plot(tc["V"], tc["I_tot_med"] * 1e6, ":", lw=1, color=col)
        lims[0] += list(tc["I_tot_med"] * 1e6)
    for i, (a, lab) in enumerate(zip(ax, (
            "|I| (uA): I_LED = lit - reference (solid), total (dotted)",
            "P lit - reference (nW)", "P / I_LED (nW/uA), light above 5x noise"))):
        a.set_yscale("log")
        y = np.array([v for v in lims[i] if v > 0])
        if y.size:
            a.set_ylim(y.min() / 3, y.max() * 3)
        a.set_xlabel("NVLED bias (V)")
        a.set_ylabel(lab)
        a.grid(alpha=0.3, which="both")
        a.axvline(v_fs, color="red", lw=0.8, ls=":")
    noref = t.loc[~t["has_ref"], "V"]
    if len(noref):
        ax[0].axvspan(noref.min() - 0.05, noref.max() + 0.05, color="0.9", zorder=0,
                      label="no reference: total only")
    ax[0].legend(fontsize=8)
    fig.suptitle(f"Sweep curves per fast-scan class ({v_fs:+.2f} V, red dotted): median and "
                 f"25-75 %", fontsize=11)
    fig.tight_layout()
    fig.savefig(base / "class_curves.png", dpi=150)
    plt.close(fig)


def main(argv=None) -> int:
    run = Path((argv or sys.argv[1:])[0]).resolve()
    log = lambda s: print(s, flush=True)
    plt = aa._plt()
    base = run / "REPORT_assets" / "pixelclassification"
    fsc = fastscan_classes(run)
    if not fsc:
        log("no fast scan")
        return 0
    (base / "fastscan").mkdir(parents=True, exist_ok=True)
    if len(fsc) > 1:
        fastscan_transitions(fsc, base / "fastscan", plt, log)
    if not (run / "NVLEDSweep" / "Sweep").is_dir():
        return 0
    sw = load_sweep(run)
    if sw is None:
        return 0
    log(f"  sweep: {sw[['X', 'Y']].drop_duplicates().shape[0]} pixels, "
        f"{sw['V'].nunique()} biases")
    tiv = shorts_iv(run)
    if tiv is not None:
        log(f"  shorts: {tiv[['X', 'Y']].drop_duplicates().shape[0]} from "
            f"{', '.join(tiv['RUN'].unique())}")
    wide = sweep_classes(sw, fsc, i_max_by_bias(tiv), base / "sweep", plt, log)
    class_curves(sw, wide.iloc[:, 1], next(iter(fsc)), base / "sweep", plt)
    log("  wrote sweep/class_curves.png")
    return 0


if __name__ == "__main__":
    sys.exit(main())
