"""Contract test: this repo can still read a run directory from SMILE_Characterization.

    python test_run_format.py        (no pytest needed; also collects under pytest)

`tests/fixture/` is a real run produced by `python -m measure_iqe --simulate`.
The measurement repo holds a byte-identical copy and asserts it still *writes*
this shape; this file asserts we can still *read* it. If RUN_FORMAT.md changes,
both fixtures and both tests change together — that is the point.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

import analyze_array as aa

FIXTURE = Path(__file__).with_name("tests") / "fixture"
SWEEP = FIXTURE / "NVLEDSweep" / "Sweep"

# The columns RUN_FORMAT.md promises are always there.
REQUIRED = ["X", "Y", "BITVAL", "NVLED_V", "TIME_START", "TIME_END",
            "NVLED_CURR_MEAN", "NVLED_CURR_SEM", "VLED_CURR_MEAN", "VLED_CURR_SEM",
            "PM400_POWER_MEAN", "PM400_POWER_SEM"]


def test_fixture_is_present():
    assert FIXTURE.is_dir(), f"missing fixture at {FIXTURE} — see RUN_FORMAT.md"
    assert (FIXTURE / "FastScan").is_dir()
    assert SWEEP.is_dir()


def test_aggregated_csv_has_the_promised_columns():
    for folder in (FIXTURE / "FastScan", SWEEP):
        files = sorted(folder.glob("*aggregated_bitval=*_mean_std.csv"))
        assert files, f"no aggregated CSV in {folder}"
        df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
        missing = [c for c in REQUIRED if c not in df.columns]
        assert not missing, f"{folder.name} lost required column(s): {missing}"
        assert len(df) > 0


def test_raw_csv_carries_the_dark_rows():
    raw = pd.read_csv(next(iter(sorted((SWEEP / "raw_data").glob("*Sweep.csv")))))
    assert list(raw.columns) == ["X", "Y", "BITVAL", "NVLED_V", "TIME", "TYPE",
                                 "MEAS_VALUE", "MEAS_STD"]
    types = set(raw["TYPE"])
    assert {"PM400", "VLED", "NVLED"} <= types, f"missing channel rows: {types}"
    assert {"DARK_PM400", "DARK_VLED", "DARK_NVLED"} <= types, f"missing dark rows: {types}"


def test_config_json_is_a_settings_mapping():
    cfgs = list((SWEEP / "raw_data").glob("*_config.json"))
    assert cfgs, "no *_config.json in the sweep raw_data"
    settings = json.loads(cfgs[0].read_text(encoding="utf-8"))["settings"]
    assert isinstance(settings, dict) and settings
    for key in ("sample_name", "roi", "pm_wavelength", "pixel_area_um2"):
        assert key in settings, f"settings lost {key}"


def test_loader_and_enrich_accept_the_fixture():
    """The real entry path: analyze_array's own loader, dark subtraction and enrich."""
    df = aa._load_aggregated(SWEEP)
    assert df is not None and len(df) > 0
    enriched = aa.enrich(df, dark_W=aa._dark_power(SWEEP))
    for col in ("P_W", "I_NVLED", "EQE_LED", "J_LED", "RELIABLE"):
        assert col in enriched.columns, f"enrich() no longer produces {col}"
    assert enriched["RELIABLE"].any(), "no reliable points in the fixture"
    assert (enriched["I_NVLED"] >= 0).all(), "LED current must be rectified to positive"


def test_config_loader_reads_the_settings():
    settings = aa._load_config(SWEEP)
    assert settings.get("sample_name") == "FixtureArray"


def test_missing_optional_columns_still_load():
    """Older runs have no *_MEDIAN columns; the loader must fall back to *_MEAN."""
    df = aa._load_aggregated(SWEEP)
    trimmed = df[[c for c in df.columns if not c.endswith(("_MEDIAN", "_SAMPSTD", "_N"))]]
    enriched = aa.enrich(trimmed.copy(), dark_W=0.0)
    assert "EQE_LED" in enriched.columns and len(enriched) == len(trimmed)


def test_a_run_from_before_the_sem_rename_reads_the_same():
    """Runs before 2026-10-24 stored the SEM as *_STD. enrich() must read them unchanged."""
    df = aa._load_aggregated(SWEEP)
    old = df.rename(columns={c: c[:-4] + "_STD" for c in df.columns if c.endswith("_SEM")})
    assert "NVLED_CURR_STD" in old.columns and "NVLED_CURR_SEM" not in old.columns
    new_e, old_e = aa.enrich(df, dark_W=0.0), aa.enrich(old, dark_W=0.0)
    for col in ("EQE_LED_ERR", "RELIABLE"):
        assert new_e[col].equals(old_e[col]), f"{col} differs between the two names"
    # and *_SAMPSTD / *_MADSTD are not mistaken for an old SEM
    assert {"NVLED_CURR_SAMPSTD", "NVLED_CURR_MADSTD"} <= set(aa.legacy_sem(df).columns)


def test_decimation_keeps_both_ends_and_every_kth_bias():
    """--decimation refits on every k-th bias; the lowest and highest always stay."""
    assert aa.decimated_biases(list(range(10)), 3) == [0, 3, 6, 9]
    assert aa.decimated_biases(list(range(11)), 3) == [0, 3, 6, 9, 10]
    assert aa.decimated_biases(list(range(30)), 1) == list(range(30))
    assert not aa.RUN_TRUNCATION and not aa.RUN_DECIMATION, "both stay off by default"


def test_nearest_maps_and_after_the_fact_pooling():
    """Block maps take the nearest measured pixel; a pooled target's fast-scan values are
    the SUM over its members, as the sweep measured it."""
    import numpy as np
    G = aa.nearest_grid([0, 9], [0, 0], [1.0, 2.0], (0, 0, 9, 1))
    assert G.shape == (2, 10)
    assert (G[:, :5] == 1).all() and (G[:, 5:] == 2).all(), G
    fs = pd.DataFrame(dict(X=[1, 2, 3], Y=[1, 1, 1], PM400_POWER_MEAN=[1e-9, 2e-9, 4e-9],
                           NVLED_CURR_MEAN=[-1e-5, -1e-5, -2e-5]))
    tg = pd.DataFrame(dict(X=[1, 3], Y=[1, 1], POOL_MEMBERS=["1:1 2:1", ""], val=[1.0, 1.0]))
    v = aa.fs_target_values(fs, tg)
    assert abs(v["FS_P_W"].iloc[0] - 3e-9) < 1e-18 and abs(v["FS_I_A"].iloc[0] - 2e-5) < 1e-15
    assert v["cx"].iloc[0] == 1.5 and v["K"].iloc[0] == 2 and v["K"].iloc[1] == 1
    scale, n = aa.nearest_scale(fs, tg, "val")
    assert n == 2 and len(scale) == 3
    assert scale[0] == scale[1] != scale[2], "pixels 1 and 2 take the pool's ratio"


def test_leakage_reference_is_subtracted():
    """I_LED must be the corrected current, or the rising side stays wrong."""
    df = aa._load_aggregated(SWEEP)
    for col in ("NVLED_CURR_REF", "NVLED_CURR_REF_SEM", "PM400_POWER_REF"):
        assert col in df.columns, f"{col} missing: the fixture predates the leakage fix"
    enriched = aa.enrich(df.copy(), dark_W=0.0)

    ref = enriched["I_NVLED_REF"]
    assert (ref > 0).any(), "reference is all zero"
    # I_LED = |lit| - |ref|, strictly smaller than the raw current wherever there is leakage
    assert ((enriched["I_LED"] <= enriched["I_NVLED"] + 1e-18).all()), "I_LED exceeds the raw current"
    assert (enriched["I_LED"] < enriched["I_NVLED"]).any(), "nothing was subtracted"
    assert (enriched["LEAK_FRAC"].dropna().between(0, 1)).all()
    # EQE follows the corrected current, so it must be HIGHER than the uncorrected one
    assert (enriched["EQE_LED"] >= enriched["EQE_NVLED"] - 1e-18).all()

    # an old run without the columns keeps the old meaning rather than failing
    old = df.drop(columns=[c for c in df.columns if "_REF" in c])
    old = aa.enrich(old, dark_W=0.0)
    assert (old["I_LED"] == old["I_NVLED"]).all()
    assert old["LEAK_FRAC"].isna().all()


def test_a_cached_reference_does_not_hide_a_contaminated_pixel():
    """REF_CONTAMINATED must be judged on the points that MEASURED their reference.

    Since 17 October 2026 the sweep caches one reference per pixel below
    leak_ref_per_point_above_v (LEAK_REF_CACHED = 1), and that value is a clean
    min-of-k by construction. On the default grid that is 22 of 30 points carrying
    ~0, which would drag the per-pixel median under the gate and let a pixel
    contaminated at every bias it actually read pass as reliable.
    """
    df = aa._load_aggregated(SWEEP)
    if "PM400_POWER_REF" not in df.columns:
        return                                   # nothing to gate on
    px = (df["X"].iloc[0], df["Y"].iloc[0])
    mine = (df["X"] == px[0]) & (df["Y"] == px[1])
    assert mine.sum() >= 4, "need a few biases on one pixel for this to mean anything"

    def flagged(frame):
        return bool(aa.enrich(frame.copy(), dark_W=0.0)
                    .loc[mine.values, "REF_CONTAMINATED"].any())

    # Contaminate the two faintest biases of one pixel and cache the rest -- the shape
    # the sweep now produces, where only the faint side measures its own reference.
    # A MINORITY of the pixel's points on purpose: that is exactly the case the old
    # all-points median misses and the measured-points median catches.
    bad = df.copy()
    faint_v = sorted(bad.loc[mine, "NVLED_V"].unique())[-2:]
    faint = mine & bad["NVLED_V"].isin(faint_v)
    assert 0 < faint.sum() < mine.sum() / 2,         f"need the contaminated points to be a minority: {faint.sum()}/{mine.sum()}"
    bad.loc[faint, "PM400_POWER_REF"] = 0.6 * bad.loc[faint, "PM400_POWER_MEAN"].abs()
    bad["LEAK_REF_CACHED"] = 0
    bad.loc[mine & ~faint, "LEAK_REF_CACHED"] = 1
    assert flagged(bad), (
        "a pixel contaminated at every bias it measured was NOT flagged -- the cached "
        "points diluted the per-pixel median")

    # Same data with the column absent (an older run) must behave as it always did:
    # there every point is a measured one, and the clean half pulls the median down.
    assert not flagged(bad.drop(columns=["LEAK_REF_CACHED"])), (
        "without the column the median runs over all points, so half-clean must pass "
        "-- this is the behaviour the column exists to improve on")


def test_trace_keys_decode():
    """traces.npz keys are part of the contract: the plotter must decode them."""
    import plot_transients as pt

    k = pt.parse_key("000_step_hc1_rNone_f-1.000_v-0.950_5")
    assert (k["what"], k["v"], k["v_from"], k["highc"], k["range"], k["stride"]) ==            ("step", -0.95, -1.0, True, None, "5"), k
    assert pt.label_of("000_step_hc1_rNone_f-1.000_v-0.950_5") ==            "-0.950 V (from -1.000), High-C, range auto"

    k = pt.parse_key("003_step_hc0_r0.001_f-1.000_v-0.800_1")
    assert (k["highc"], k["range"]) == (False, "0.001"), k

    # keys written before the from-voltage was recorded must still decode
    assert pt.parse_key("000_step_hc1_rNone_v-0.950_5")["v_from"] is None

    # leak keys carry a frame state where step carries hc/range
    assert pt.label_of("007_leak_lit_f-0.900_v-0.850_5") == "-0.850 V (from -0.900), lit"

    # an unknown tag must not crash a plot
    assert "wat" in pt.label_of("000_step_wat_v-0.800_1")


def test_run_settings_names_the_limit():
    """The compliance limit must reach the figure: in High-C it selects the range."""
    import json
    import tempfile
    import plot_transients as pt

    with tempfile.TemporaryDirectory() as d:
        run = Path(d)
        (run / "run_config.json").write_text(json.dumps({"config": {}, "args": {
            "v_start": -1.0, "compliance": 1e-5, "charge_limit": 1e-3,
            "nplc": 0.01, "highc": True, "pixels": ["124,135"]}}), encoding="utf-8")
        got = pt.run_settings(run)
        for want in ("-1.000 V", "limit 10 uA", "charged at 1 mA", "NPLC 0.01",
                     "High-C", "124,135"):
            assert want in got, f"{want!r} missing from {got!r}"

        # a run directory without the config must not crash a plot
        (run / "run_config.json").unlink()
        assert pt.run_settings(run) == ""

    assert pt.si(1e-5) == "10 uA" and pt.si(3.2e-7) == "320 nA", pt.si(1e-5)


def test_a_fast_scan_at_several_biases_is_split_not_pooled():
    """fastscan_voltages writes every bias into one CSV, told apart by NVLED_V.

    Pooled, they doubled the pixel count, mixed the alive threshold across biases and
    sent the sweep cross-check to the MEDIAN bias -- -1.975 V for a -1.35/-2.6 V scan,
    which nobody measured. Built from the fixture: the same pixels again at -2.60 V,
    twice as bright, with three of them dark there.
    """
    import shutil
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        run = Path(tmp) / "run"
        shutil.copytree(FIXTURE, run)
        agg = next((run / "FastScan").glob("*aggregated_bitval=*.csv"))
        fs = pd.read_csv(agg)
        v0 = float(fs["NVLED_V"].iloc[0])
        hi = fs.copy()
        hi["NVLED_V"] = -2.6
        hi["PM400_POWER_MEAN"] *= 2.0
        hi["TIME_START"] += 1e4
        dark = fs[["X", "Y"]].iloc[:3]
        hi.loc[hi.index[:3], "PM400_POWER_MEAN"] = 1e-13
        pd.concat([fs, hi]).to_csv(agg, index=False)

        cfg = aa._load_config(run / "FastScan")
        first, extra = aa.split_fastscan_biases(aa._load_aggregated(run / "FastScan"), cfg)
        assert len(first) == len(fs) and set(extra) == {-2.6}, (len(first), set(extra))
        assert round(float(first["NVLED_V"].iloc[0]), 4) == round(v0, 4)

        aa.analyse_run(run, lambda *_a: None)
        report = (run / "REPORT.md").read_text(encoding="utf-8")
        assert f"Pixels scanned: **{len(fs)}**" in report, "the pixel count doubled"
        assert "Fast scan at 2 biases" in report
        row = next(l for l in report.splitlines() if l.startswith("| -2.600 V"))
        assert row.split("|")[3].strip() == str(len(dark)), row
        assert (run / "REPORT_assets" / "fastscan_biases.png").exists()
        # power, current (and IQE when the sweep fits) at EVERY bias, not just the first
        for v in (v0, -2.6):
            assert (run / "REPORT_assets" / f"fastscan_maps_{aa._bias_tag(v)}.png").exists(), v
        assert "Optical power, current and IQE at every fast-scan bias" in report


def test_analyze_finds_runs_and_knows_when_they_are_stale():
    import shutil
    import tempfile

    import analyze

    with tempfile.TemporaryDirectory() as tmp:
        run = Path(tmp) / "261001-0000_X_ROI_(0, 0, 3, 3)"
        shutil.copytree(FIXTURE, run)
        assert analyze.find_runs(Path(tmp)) == [run]
        assert analyze.find_runs(run) == [run]
        h = analyze.inputs_hash(run)
        # what the tools write is not input
        (run / "REPORT_assets").mkdir(exist_ok=True)
        (run / "REPORT_assets" / "x.png").write_bytes(b"0")
        (run / "REPORT.md").write_text("r")
        (run / analyze.STATE).write_text("{}")
        (run / analyze.LOG).write_text("l")
        assert analyze.inputs_hash(run) == h
        agg = next((run / "FastScan").glob("*aggregated*.csv"))
        agg.write_text(agg.read_text() + "\n")
        assert analyze.inputs_hash(run) != h
    ok = dict(code="c", inputs="i", exit=0)
    assert analyze.why_stale("t", ok, "c", "i") is None
    assert analyze.why_stale("t", {}, "c", "i") == "never run"
    assert analyze.why_stale("t", ok, "c2", "i") == "t changed"
    assert analyze.why_stale("t", ok, "c", "i2") == "input files changed"
    assert "failed" in analyze.why_stale("t", dict(ok, exit=1), "c", "i")


def test_density_weighs_current_and_light_by_decade():
    I = np.array([0, 30e-6])
    assert aa.ip_equivalence(I, np.array([0, 40e-9])) == 1e-3     # 1 nW = 1 uA
    assert aa.ip_equivalence(I, np.array([0, 30e-6])) == 1.0      # 1 uW = 1 uA
    assert aa.si_unit(3e-5, "A") == (1e6, "uA")
    assert aa.si_unit(40e-9, "W")[1] == "nW" and aa.si_unit(30e-6, "W")[1] == "uW"


def test_pixel_summary_on_the_fixture():
    run = FIXTURE
    sw = run / "NVLEDSweep" / "Sweep"
    df = aa.enrich(aa._load_aggregated(sw), dark_W=aa._dark_power(sw))
    s = aa.pixel_summary(df, pd.DataFrame())
    assert len(s) == df.groupby(["X", "Y"]).ngroups
    for q in ("nvled_curr", "vled_curr", "pm400_power", "eqe_max", "forward_leakage"):
        for st in ("mean", "median", "std", "mad"):
            assert f"{q}_{st}" in s.columns, (q, st)
    assert {"forward_voltage_mean", "forward_voltage_median", "v_at_eqe_max"} <= set(s.columns)
    # the peak bias is a bias the pixel was measured at
    ok = s["v_at_eqe_max"].notna()
    assert ok.any() and set(s.loc[ok, "v_at_eqe_max"]) <= set(df["NVLED_V"].astype(float))


def test_pixel_summary_interpolates_to_the_fitted_peak():
    # two reliable points at J = 1 and 4 A/cm2; the fit puts the peak at J = 2: halfway
    # in log J, so a power law I ~ J lands at 2 uA, and the bias halfway at -0.95 V
    df = pd.DataFrame(dict(X=0, Y=0, NVLED_V=[-0.9, -1.0], J_LED=[1.0, 4.0], RELIABLE=True))
    for base, v in (("NVLED_CURR", [-1e-6, -4e-6]), ("VLED_CURR", [1e-6, 4e-6]),
                    ("PM400_POWER", [1e-9, 4e-9])):
        for suf in ("MEAN", "MEDIAN", "SAMPSTD", "MADSTD"):
            df[f"{base}_{suf}"] = v if suf in ("MEAN", "MEDIAN") else np.abs(v) * 1e-2
    df["NVLED_CURR_REF"] = df["PM400_POWER_REF"] = 0.0
    fits = pd.DataFrame(dict(X=[0, 1], Y=0, J_at_peak=[2.0, 2.0], clean_fit=[True, False],
                             eta_LEE=1e-3, peak_IQE_raw=0.8, peak_IQE=[0.8, np.nan],
                             R2=0.99))
    summ = pd.DataFrame(dict(pixel_x=[0, 1], pixel_y=0, quadrant="TL", pool_k=1,
                             forward_voltage_median=-0.88))
    si = aa.pixel_summary_interpolated(df, fits, summ).set_index("pixel_x")
    r = si.loc[0]
    assert abs(r["nvled_curr_median"] - 2e-6) < 1e-15, r["nvled_curr_median"]
    assert abs(r["pm400_power_mean"] - 2e-9) < 1e-18
    assert abs(r["v_at_peak"] - (-0.95)) < 1e-12
    assert abs(r["eqe_max_fit"] - 0.8e-3) < 1e-15 and r["forward_voltage_median"] == -0.88
    assert np.isnan(si.loc[1, "v_at_peak"])          # not a clean fit: no values


def test_forward_voltage_interpolates_in_log_j():
    V = np.array([0.0, -0.8, -0.9, -1.0] * 2)
    J = np.array([1e-4, 0.1, 10.0, 100.0,     # crosses 1 A/cm2 halfway (log) -0.8..-0.9
                  1e-4, 1e-3, 1e-2, 0.5])     # never reaches it
    df = pd.DataFrame({"X": [0] * 4 + [1] * 4, "Y": 0, "NVLED_V": V})
    vf = aa._forward_voltage(df, J)
    assert abs(vf[(0, 0)] - (-0.85)) < 1e-12, vf
    assert (1, 0) not in vf.index


def test_block_stats_match_the_per_block_loop():
    rng = np.random.default_rng(1)
    X = rng.integers(3, 70, 400)
    Y = rng.integers(10, 41, 400)
    V = rng.normal(80, 5, 400)
    for S in (2, 8, 32):
        med, spr, cnt = aa._block_stats(X, Y, V, S)
        for j, by in enumerate(range(Y.min() // S, Y.max() // S + 1)):
            for i, bx in enumerate(range(X.min() // S, X.max() // S + 1)):
                v = V[(X // S == bx) & (Y // S == by)]
                assert cnt[j, i] == v.size
                if v.size:
                    assert med[j, i] == np.median(v)
                    want = 1.4826 * np.median(np.abs(v - np.median(v))) if v.size > 1 else 0
                    assert abs(spr[j, i] - want) < 1e-9
                else:
                    assert np.isnan(med[j, i])


def test_pixel_fits_are_cached_and_parallel_fits_match():
    import tempfile

    J = np.logspace(-1, 2, 12)
    rows = []
    for k, beta in enumerate((2.0, 4.0, 8.0)):
        e = aa._abc_eqe_norm(J, beta, 1.0, 0.05)
        rows += [dict(X=k, Y=0, J_LED=j, EQE_LED=v, RELIABLE=True) for j, v in zip(J, e)]
    groups = list(pd.DataFrame(rows).groupby(["X", "Y"]))
    msgs = []
    with tempfile.TemporaryDirectory() as tmp:
        cache = Path(tmp) / "abc_fit_cache.pkl"
        jobs, pmin = aa.FIT_JOBS, aa.FIT_PARALLEL_MIN
        try:
            aa.FIT_JOBS, aa.FIT_PARALLEL_MIN = 2, 1
            par = aa.fit_pixels(groups, cache, msgs.append)
            aa.FIT_JOBS = 1
            again = aa.fit_pixels(groups, cache, msgs.append)
        finally:
            aa.FIT_JOBS, aa.FIT_PARALLEL_MIN = jobs, pmin
    assert "0 cached, 3 to fit in 2 processes" in msgs[0], msgs
    assert "3 cached, 0 to fit" in msgs[1], msgs
    for xy, f in par.items():
        assert abs(f["beta"] - again[xy]["beta"]) == 0
        serial = aa.fit_abc(*[g for (k, g) in groups if k == xy][0][["J_LED", "EQE_LED"]].T.values)
        assert abs(f["peak_IQE"] - serial["peak_IQE"]) < 1e-12, (xy, f["peak_IQE"])


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"ok  {name}")
    print("all contract checks passed")
