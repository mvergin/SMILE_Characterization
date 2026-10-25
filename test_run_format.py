"""Contract test: a run directory still has the shape ``analysis/`` expects.

    python test_run_format.py                 (writer side -- this file)
    python analysis/test_reads_run_format.py  (reader side)

`tests/fixture/` is a real simulated run, committed as the executable form of
RUN_FORMAT.md. This file asserts a fresh run still *writes* that shape, and that the
committed fixture has not drifted from what the code produces.
`analysis/test_reads_run_format.py` asserts the analysis side can still *read* it.

Changing the format intentionally used to mean copying RUN_FORMAT.md and the fixture
into a second repo by hand, and the report audit checked their md5 every time. ROADMAP
item 2 merged the repos, so now: change RUN_FORMAT.md, regenerate the fixture with the
command at the bottom of it, and update both tests. One copy of each, no sync step.
"""

from __future__ import annotations

import csv
import json
import tempfile
from pathlib import Path

from measure_iqe import IQEConfig, run_file, run_iqe_measurement

FIXTURE = Path(__file__).with_name("tests") / "fixture"

# Exactly the settings RUN_FORMAT.md documents for regenerating the fixture.
FIXTURE_CFG = dict(sample_name="FixtureArray", roi=(0, 0, 7, 7), block_size=4,
                   pixels_per_block=1, n_fine=6, warmup_ms=0.0)
# ...plus the timing cuts that keep the test fast; they change no column.
FAST_CFG = dict(thermal_settle_ms=1.0, settle_ms=1.0, fastscan_settle_ms=0.0,
                settle_scale=0.0,   # the sweep settles are SECONDS; a test must not wait
                vramp_lowv_n_pm=3, vramp_highv_n_pm=3)

# The fast scan writes exactly these, in this order. The sweep writes them too, but
# grouped by channel, so a reader finds them by name.
AGG_REQUIRED = ["X", "Y", "BITVAL", "NVLED_V", "TIME_START", "TIME_END",
                "NVLED_CURR_MEAN", "NVLED_CURR_SEM", "VLED_CURR_MEAN", "VLED_CURR_SEM",
                "PM400_POWER_MEAN", "PM400_POWER_SEM"]
# The sweep: one block of burst stats per channel, each followed by its leakage
# reference (section 6), then the reference bookkeeping and the per-point compliance
# limit. Pinned exactly, so a column cannot appear or vanish unnoticed.
_STATS = ["MEAN", "MEDIAN", "SAMPSTD", "MADSTD", "SEM", "N"]
AGG_SWEEP = (AGG_REQUIRED[:6] + ["QUADRANT"]
             + [c for ch in ("NVLED_CURR", "VLED_CURR", "PM400_POWER")
                for c in ([f"{ch}_{s}" for s in _STATS] + [f"{ch}_REF"]
                          + [f"{ch}_REF_{s}" for s in _STATS])]
             + ["LEAK_REF_BITVAL", "LEAK_REF_TRIES", "LEAK_REF_FRAC", "LEAK_REF_CACHED",
                "NVLED_LIMIT_A", "NVLED_AT_COMPLIANCE"])
# Written ONLY when pool_groups is non-empty, which is not the default. Listed apart
# from AGG_SWEEP on purpose: an ordinary run must write the columns above and
# no others, or every existing reader has to learn a column to read a run with no
# pooled rows in it. test_measure_iqe.py pins the "off by default" half.
AGG_POOL = ["POOL_K", "POOL_MEMBERS"]
RAW_COLS = ["X", "Y", "BITVAL", "NVLED_V", "TIME", "TYPE", "MEAS_VALUE", "MEAS_STD"]
SAMPLE_COLS = ["X", "Y", "BITVAL", "NVLED_V", "TIME", "TYPE", "SAMPLE_IDX", "VALUE"]
RAW_TYPES = {"PM400", "VLED", "NVLED"}
DARK_TYPES = {"DARK_PM400", "DARK_VLED", "DARK_NVLED"}


def _fresh_run(tmp):
    cfg = IQEConfig(**{**FIXTURE_CFG, **FAST_CFG})
    return run_iqe_measurement(cfg, simulate=True, run_dir=tmp, log=lambda _m: None).run_dir


def _header(path):
    with open(path, newline="", encoding="utf-8") as f:
        return next(csv.reader(f))


def _check_layout(run_dir: Path, label: str):
    sweep = run_dir / "NVLEDSweep" / "Sweep"
    fast = run_dir / "FastScan"
    assert fast.is_dir(), f"{label}: no FastScan/"
    assert sweep.is_dir(), f"{label}: no NVLEDSweep/Sweep/"

    for folder in (fast, sweep):
        agg = sorted(folder.glob("*aggregated_bitval=*_mean_std.csv"))
        assert agg, f"{label}: no aggregated CSV in {folder.name}"
        missing = set(AGG_REQUIRED) - set(_header(agg[0]))
        assert not missing, f"{label}/{folder.name}: required columns lost -> {missing}"

    fast_agg = _header(sorted(fast.glob("*aggregated_bitval=*_mean_std.csv"))[0])
    # QUADRANT last, so the columns a positional reader counts stay where they were
    assert fast_agg == AGG_REQUIRED + ["QUADRANT"], (
        f"{label}: fast-scan columns changed -> {fast_agg}")
    sweep_agg = _header(sorted(sweep.glob("*aggregated_bitval=*_mean_std.csv"))[0])
    assert sweep_agg in (AGG_SWEEP, AGG_SWEEP + AGG_POOL), \
        f"{label}: sweep columns changed -> {sweep_agg}"

    raw = run_file(run_dir, "NVLEDSweep/Sweep/raw_data/Sweep.csv")
    assert _header(raw) == RAW_COLS, f"{label}: raw columns changed"
    with open(raw, newline="", encoding="utf-8") as f:
        types = {r["TYPE"] for r in csv.DictReader(f)}
    assert RAW_TYPES <= types, f"{label}: missing channel rows {RAW_TYPES - types}"
    assert DARK_TYPES <= types, f"{label}: missing dark rows {DARK_TYPES - types}"

    samples = run_file(run_dir, "NVLEDSweep/Sweep/raw_data/Sweep_samples.csv")
    assert samples.exists() and _header(samples) == SAMPLE_COLS, \
        f"{label}: Sweep_samples.csv columns changed"
    with open(samples, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            float(r["VALUE"])    # a plain float: the sim once wrote "np.float64(...)"

    cfgs = list((sweep / "raw_data").glob("*_config.json"))
    assert cfgs, f"{label}: no *_config.json"
    settings = json.loads(cfgs[0].read_text(encoding="utf-8"))["settings"]
    for key in ("sample_name", "roi", "pm_wavelength", "pixel_area_um2"):
        assert key in settings, f"{label}: settings lost {key}"
    return settings


def test_a_fresh_run_matches_the_documented_format():
    with tempfile.TemporaryDirectory() as tmp:
        settings = _check_layout(_fresh_run(Path(tmp) / "run"), "fresh run")
        assert settings["sample_name"] == "FixtureArray"


def test_the_committed_fixture_matches_the_documented_format():
    assert FIXTURE.is_dir(), f"missing fixture at {FIXTURE} — see RUN_FORMAT.md"
    _check_layout(FIXTURE, "committed fixture")


def test_the_fixture_has_not_drifted_from_what_the_code_writes():
    """Same columns and same config keys as a run made right now.

    Values differ (timings, RNG draws), so only the schema is compared — that is
    what the analysis repo depends on.
    """
    with tempfile.TemporaryDirectory() as tmp:
        fresh = _fresh_run(Path(tmp) / "run")
        for rel in ("FastScan/aggregated_bitval=15_mean_std.csv",
                    "NVLEDSweep/Sweep/aggregated_bitval=15_mean_std.csv",
                    "NVLEDSweep/Sweep/raw_data/Sweep.csv",
                    "NVLEDSweep/Sweep/raw_data/Sweep_samples.csv"):
            assert _header(run_file(fresh, rel)) == _header(run_file(FIXTURE, rel)), \
                f"{rel} drifted"
        cfg_rel = "NVLEDSweep/Sweep/raw_data/Sweep_config.json"
        a = json.loads(run_file(fresh, cfg_rel).read_text(encoding="utf-8"))["settings"]
        b = json.loads(run_file(FIXTURE, cfg_rel).read_text(encoding="utf-8"))["settings"]
        added, removed = sorted(set(a) - set(b)), sorted(set(b) - set(a))
        assert not removed, f"fixture has settings the code no longer writes: {removed}"
        assert not added, f"code writes settings the fixture lacks: {added} — regenerate it"


def test_an_interrupted_run_is_still_readable():
    """RUN_FORMAT.md promises a partial run is valid. Writes are flushed per pixel,
    so a run stopped after the fast scan must still leave a well-formed FastScan."""
    with tempfile.TemporaryDirectory() as tmp:
        run_dir = Path(tmp) / "partial"
        cfg = IQEConfig(**{**FIXTURE_CFG, **FAST_CFG})
        import measure_iqe as m
        pm, smu, smile = m.connect(cfg, lambda _m: None, simulate=True)
        try:
            m.configure_common(pm, smu, cfg, lambda _m: None)
            smu.enable_output("a", True)
            smu.enable_output("b", True)
            run_dir.mkdir(parents=True)
            x1, y1, x2, y2 = cfg.resolve_roi()
            sel = m.run_fast_scan(pm, smu, smile, x1, y1, x2, y2, run_dir, cfg,
                                  lambda _m: None)
        finally:
            m.shutdown(pm, smu, smile, cfg)
        assert sel, "fast scan found no pixels"
        agg = run_file(run_dir, "FastScan/aggregated_bitval=15_mean_std.csv")
        assert _header(agg) == AGG_REQUIRED + ["QUADRANT"]
        with open(agg, newline="", encoding="utf-8") as f:
            assert len(list(csv.DictReader(f))) == 64      # every ROI pixel


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"ok  {name}")
    print("all contract checks passed")
