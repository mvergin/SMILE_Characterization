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

import pandas as pd

import analyze_array as aa

FIXTURE = Path(__file__).with_name("tests") / "fixture"
SWEEP = FIXTURE / "NVLEDSweep" / "Sweep"

# The columns RUN_FORMAT.md promises are always there.
REQUIRED = ["X", "Y", "BITVAL", "NVLED_V", "TIME_START", "TIME_END",
            "NVLED_CURR_MEAN", "NVLED_CURR_STD", "VLED_CURR_MEAN", "VLED_CURR_STD",
            "PM400_POWER_MEAN", "PM400_POWER_STD"]


def test_fixture_is_present():
    assert FIXTURE.is_dir(), f"missing fixture at {FIXTURE} — see RUN_FORMAT.md"
    assert (FIXTURE / "FastScan").is_dir()
    assert SWEEP.is_dir()


def test_aggregated_csv_has_the_promised_columns():
    for folder in (FIXTURE / "FastScan", SWEEP):
        files = sorted(folder.glob("aggregated_bitval=*_mean_std.csv"))
        assert files, f"no aggregated CSV in {folder}"
        df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
        missing = [c for c in REQUIRED if c not in df.columns]
        assert not missing, f"{folder.name} lost required column(s): {missing}"
        assert len(df) > 0


def test_raw_csv_carries_the_dark_rows():
    raw = pd.read_csv(SWEEP / "raw_data" / "Sweep.csv")
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


def test_leakage_reference_is_subtracted():
    """I_LED must be the corrected current, or the rising side stays wrong."""
    df = aa._load_aggregated(SWEEP)
    for col in ("NVLED_CURR_REF", "NVLED_CURR_REF_STD", "PM400_POWER_REF"):
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
    old = df.drop(columns=[c for c in df.columns if c.endswith(("_REF", "_REF_STD"))])
    old = aa.enrich(old, dark_W=0.0)
    assert (old["I_LED"] == old["I_NVLED"]).all()
    assert old["LEAK_FRAC"].isna().all()


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


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"ok  {name}")
    print("all contract checks passed")
