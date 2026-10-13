"""Self-check for the measure_iqe package — runs the whole flow against the sim rig.

No pytest needed::

    python test_measure_iqe.py

It also collects under pytest if you have it.  Everything writes into a temp
directory, so it touches no measurement data.
"""

from __future__ import annotations

import csv
import dataclasses
import json
import tempfile
from pathlib import Path

from measure_iqe import IQEConfig, run_iqe_measurement
from measure_iqe.cli import main as cli_main
from measure_iqe.cli import parse_override

QUIET = lambda _msg: None                                 # noqa: E731 - tests stay silent

# small but exercises every branch: 2x2 blocks, both ramp segments, raw samples on
FAST = dict(roi=(0, 0, 7, 7), block_size=4, pixels_per_block=1, n_fine=6, n_coarse=4,
            warmup_ms=0.0, thermal_settle_ms=1.0, settle_ms=1.0, fastscan_settle_ms=0.0,
            vramp_lowv_settle_ms=1.0, vramp_highv_settle_ms=1.0,
            vramp_lowv_n_pm=2, vramp_highv_n_pm=2, lowj_n_pm=2, highj_n_pm=2, n_pm_samples=2)

AGG_COLS = ["X", "Y", "BITVAL", "NVLED_V", "TIME_START", "TIME_END",
            "NVLED_CURR_MEAN", "NVLED_CURR_STD", "VLED_CURR_MEAN", "VLED_CURR_STD",
            "PM400_POWER_MEAN", "PM400_POWER_STD"]


def test_config_defaults_are_valid():
    cfg = IQEConfig().validate()
    assert cfg.resolve_roi() == (32, 32, 63, 63)
    assert IQEConfig(roi="TL").resolve_roi() == (0, 0, 255, 255)
    assert abs(cfg.photon_energy_j - 4.514e-19) < 1e-21    # 440 nm photon


def test_config_rejects_bad_input():
    for bad in (dict(source_mode="sideways"), dict(roi="NOWHERE"), dict(bitvals=()),
                dict(vramp_low_frac=2.0), dict(n_fine=1), dict(pixel_area_um2=0),
                dict(roi=(0, 0, 999, 3)), dict(bitvals=(99,)), dict(nvled_range_i=0.003)):
        try:
            IQEConfig(**bad).validate()
        except ValueError:
            continue
        raise AssertionError(f"{bad} should not validate")
    try:
        IQEConfig.from_dict({"not_a_field": 1})
    except ValueError as exc:
        assert "not_a_field" in str(exc)
    else:
        raise AssertionError("unknown keys must be rejected")


def test_config_roundtrips_through_a_file():
    cfg = IQEConfig(n_fine=17, sweep_direction="low_to_high", roi=(1, 2, 3, 4))
    with tempfile.TemporaryDirectory() as tmp:
        for name in ("c.json", "c.yaml"):
            try:
                path = cfg.write_file(Path(tmp) / name)
            except ImportError:
                continue                                   # PyYAML optional for JSON users
            back = IQEConfig.from_file(path)
            assert back == cfg, name
            assert back.roi == (1, 2, 3, 4), "tuple fields must survive the round trip"


def test_example_config_matches_the_defaults():
    """config.example.yaml is documentation — it must stay loadable and in sync."""
    example = Path(__file__).with_name("config.example.yaml")
    if not example.exists():
        return
    try:
        loaded = IQEConfig.from_file(example)
    except ImportError:
        return                                             # PyYAML not installed
    assert loaded == IQEConfig(), "config.example.yaml drifted from IQEConfig defaults"


def test_config_coerces_and_rejects_value_types():
    assert IQEConfig.from_dict({"pm_range": "1e-7"}).pm_range == 1e-7
    assert IQEConfig.from_dict({"vramp_knee_v": None}).vramp_knee_v is None
    for bad in ({"n_fine": "many"}, {"save_raw_samples": "yes"}):
        try:
            IQEConfig.from_dict(bad)
        except ValueError:
            continue
        raise AssertionError(f"{bad} should be rejected")


def test_cli_override_parsing():
    assert parse_override("n_fine=60") == ("n_fine", 60)
    assert parse_override("roi=[0,0,7,7]") == ("roi", [0, 0, 7, 7])
    assert parse_override("sweep_direction=low_to_high") == ("sweep_direction", "low_to_high")
    assert parse_override("save_raw_samples=false") == ("save_raw_samples", False)


def _run(tmp: str, **over) -> "object":
    cfg = IQEConfig(**{**FAST, **over})
    return run_iqe_measurement(cfg, simulate=True, base_dir=tmp, log=QUIET)


def test_voltage_sweep_writes_the_expected_files():
    with tempfile.TemporaryDirectory() as tmp:
        result = _run(tmp)
        assert result.n_pixels == 4 and result.n_points == 24
        agg = result.run_dir / "NVLEDSweep" / "Sweep" / "aggregated_bitval=15_mean_std.csv"
        rows = list(csv.DictReader(agg.open()))
        assert len(rows) == result.n_points
        assert AGG_COLS == list(rows[0])[:len(AGG_COLS)], "analyze_array column contract"
        assert "NVLED_CURR_MEDIAN" in rows[0], "raw-sample stats must be written"
        assert float(rows[0]["PM400_POWER_MEAN"]) > 0

        fast = result.run_dir / "FastScan" / "aggregated_bitval=15_mean_std.csv"
        assert len(list(csv.DictReader(fast.open()))) == 64      # every ROI pixel scanned

        saved = json.loads((result.run_dir / "NVLEDSweep" / "Sweep" / "raw_data"
                            / "Sweep_config.json").read_text(encoding="utf-8"))["settings"]
        assert saved["n_fine"] == 6 and saved["sample_name"] == result.config.sample_name
        assert len(saved) == len(dataclasses.fields(IQEConfig)), "full config is recorded"


def test_every_acquisition_mode_runs():
    modes = [dict(source_mode="current", current_sweep_n=4),
             dict(fine_sweep_mode="current"),
             dict(perpoint_range=True),
             dict(save_raw_samples=False),
             dict(sweep_select="spatial", min_sweep_pixels=2, max_sweep_pixels=2),
             dict(sweep_select="stride", max_sweep_pixels=2),
             dict(vramp_perseg_effort=False),
             dict(sweep_direction="low_to_high")]
    with tempfile.TemporaryDirectory() as tmp:
        for over in modes:
            result = _run(tmp, **over)
            assert result.n_points > 0, over
            assert result.run_dir.is_dir()


def test_warmup_diag_writes_its_own_csv():
    with tempfile.TemporaryDirectory() as tmp:
        cfg = IQEConfig(**FAST, warmup_diag_seconds=0.2, warmup_diag_n_pixels=1)
        result = run_iqe_measurement(cfg, simulate=True, warmup_diag=True,
                                     base_dir=tmp, log=QUIET)
        rows = list(csv.DictReader((result.run_dir / "warmup_diag.csv").open()))
        assert rows and set(rows[0]) == {"pixel_x", "pixel_y", "phase", "t_s", "P_W", "I_A", "EQE"}
        assert {r["phase"] for r in rows} == {"fresh_turnon", "after_conditioning"}


def test_dead_pixel_abort_skips_the_rest_of_the_sweep():
    """A huge dead threshold marks every pixel dead — each keeps exactly one point."""
    with tempfile.TemporaryDirectory() as tmp:
        result = _run(tmp, dead_abort=True, dead_abort_nw=1e9)
        assert result.n_points == result.n_pixels, "one point measured per dead-skipped pixel"


def test_cli_runs_a_simulated_measurement():
    with tempfile.TemporaryDirectory() as tmp:
        assert cli_main(["--print-config"]) == 0
        assert cli_main(["--list-params"]) == 0
        assert cli_main(["--set", "nonsense=1"]) == 2
        argv = ["--simulate", "--out", tmp, "--set", "roi=[0,0,3,3]", "--set", "block_size=4",
                "--set", "pixels_per_block=1", "--set", "n_fine=4", "--set", "warmup_ms=0"]
        assert cli_main(argv) == 0
        assert list(Path(tmp).glob("*/NVLEDSweep/Sweep/*.csv")), "CLI produced no sweep output"


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"ok  {name}")
    print("all checks passed")
