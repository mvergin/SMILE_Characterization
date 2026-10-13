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


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"ok  {name}")
    print("all contract checks passed")
