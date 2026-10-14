"""Self-check for the measure_iqe package — runs the whole flow against the sim rig.

No pytest needed::

    python test_measure_iqe.py

It also collects under pytest if you have it.  Everything writes into a temp
directory, so it touches no measurement data.
"""

from __future__ import annotations

import contextlib
import csv
import dataclasses
import io
import json
import tempfile
from pathlib import Path

from measure_iqe import IQEConfig, run_iqe_measurement, select_sweep_pixels
from measure_iqe import acquire
from measure_iqe.cli import main as cli_main
from measure_iqe.cli import parse_override

QUIET = lambda _msg: None                                 # noqa: E731 - tests stay silent

# small but exercises every branch: 2x2 blocks, both ramp segments, raw samples on
FAST = dict(roi=(0, 0, 7, 7), block_size=4, pixels_per_block=1, n_fine=6, n_coarse=4,
            warmup_ms=0.0, thermal_settle_ms=1.0, settle_ms=1.0, fastscan_settle_ms=0.0,
            settle_scale=0.0,   # the real settles are seconds; a test must not wait them
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


def _nearest_neighbour(sel):
    import numpy as np
    p = np.array([[r["X"], r["Y"]] for r in sel], float)
    d = np.abs(p[:, None, :] - p[None, :, :]).max(-1)       # Chebyshev
    np.fill_diagonal(d, np.inf)
    return d.min(axis=1)


def test_blocks_mode_never_selects_touching_pixels():
    """The regression this guards: per-block farthest-point sampling used to run to
    each block's own border, and the block next door did the same, so pairs of
    measured LEDs ended up side by side across the shared edge (at k=4, 87% of
    picks sat on a border and 2/3 of them had an adjacent partner)."""
    alive = [dict(X=x, Y=y) for y in range(64) for x in range(64)]
    for k in (1, 2, 4, 8):
        cfg = IQEConfig(roi=(0, 0, 63, 63), block_size=8, pixels_per_block=k)
        sel = select_sweep_pixels(alive, cfg, QUIET)
        assert len(sel) == 64 * k, f"k={k}: expected equal per-block sampling"
        nn = _nearest_neighbour(sel)
        assert nn.min() >= cfg.min_pixel_separation, f"k={k}: min separation {nn.min()}"


def test_blocks_mode_keeps_equal_per_block_counts():
    from collections import Counter
    alive = [dict(X=x, Y=y) for y in range(64) for x in range(64)]
    cfg = IQEConfig(roi=(0, 0, 63, 63), block_size=8, pixels_per_block=4)
    sel = select_sweep_pixels(alive, cfg, QUIET)
    counts = Counter((r["X"] // 8, r["Y"] // 8) for r in sel)
    assert len(counts) == 64 and set(counts.values()) == {4}


def test_blocks_mode_seed_still_varies_the_sample():
    """block_seed must keep selecting different pixels so repeated runs accumulate
    coverage of the array."""
    alive = [dict(X=x, Y=y) for y in range(64) for x in range(64)]
    sets = []
    for seed in (0, 1, 2):
        cfg = IQEConfig(roi=(0, 0, 63, 63), block_size=8, pixels_per_block=4, block_seed=seed)
        sets.append({(r["X"], r["Y"]) for r in select_sweep_pixels(alive, cfg, QUIET)})
    for a, b in ((0, 1), (0, 2), (1, 2)):
        overlap = len(sets[a] & sets[b]) / len(sets[a])
        assert overlap < 0.5, f"seeds {a}/{b} overlap {overlap:.0%} — seed barely matters"


def test_blocks_mode_takes_fewer_rather_than_touching_when_sparse():
    """A block without k well-separated alive pixels must return fewer, not neighbours."""
    import numpy as np
    rng = np.random.default_rng(7)
    sparse = [dict(X=x, Y=y) for y in range(64) for x in range(64) if rng.random() < 0.08]
    cfg = IQEConfig(roi=(0, 0, 63, 63), block_size=8, pixels_per_block=4)
    lines: list[str] = []
    sel = select_sweep_pixels(sparse, cfg, lines.append)
    assert 0 < len(sel) < 64 * 4, "a sparse ROI cannot fill every block"
    assert _nearest_neighbour(sel).min() >= 2
    assert any("gave fewer than" in ln for ln in lines), "short blocks must be reported"


def test_separation_guard_can_be_disabled():
    alive = [dict(X=x, Y=y) for y in range(64) for x in range(64)]
    cfg = IQEConfig(roi=(0, 0, 63, 63), block_size=8, pixels_per_block=4,
                    min_pixel_separation=1)
    sel = select_sweep_pixels(alive, cfg, QUIET)
    assert len(sel) == 256 and _nearest_neighbour(sel).min() == 1


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
        assert AGG_COLS == list(rows[0])[:len(AGG_COLS)], "SMILE_IQE_Analysis column contract"
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
             dict(perpoint_range=True, highc_b=False),   # the two are incompatible by design
             dict(leak_reference=False),                 # runs from before the leakage fix
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


def test_leakage_reference_is_measured_and_written():
    """The bitval-0 reference is the low-J correction, not an optional extra."""
    with tempfile.TemporaryDirectory() as tmp:
        result = _run(tmp)
        agg = next((result.run_dir / "NVLEDSweep" / "Sweep").glob("aggregated_*.csv"))
        rows = list(csv.DictReader(agg.open()))
        assert rows, "no sweep rows"
        for col in ("NVLED_CURR_REF", "NVLED_CURR_REF_STD", "VLED_CURR_REF",
                    "PM400_POWER_REF", "PM400_POWER_REF_STD", "LEAK_REF_BITVAL"):
            assert col in rows[0], f"{col} missing from the aggregated CSV"
        for r in rows:
            assert int(float(r["LEAK_REF_BITVAL"])) == 0
            # the reference is a real, separate reading: smaller than the lit current,
            # and not a copy of it
            lit, ref = abs(float(r["NVLED_CURR_MEAN"])), abs(float(r["NVLED_CURR_REF"]))
            assert 0 < ref < lit, f"reference {ref} not between 0 and the lit current {lit}"
        assert len({r["NVLED_CURR_REF"] for r in rows}) > 1, "reference never varies"

        # and it must be possible to turn off, for comparison with the older runs
        off = _run(tmp, leak_reference=False)
        agg = next((off.run_dir / "NVLEDSweep" / "Sweep").glob("aggregated_*.csv"))
        assert "NVLED_CURR_REF" not in agg.read_text().splitlines()[0]


def test_fast_scan_does_not_run_at_the_sweep_limit():
    """-1.35 V draws ~30 uA; the sweep's rising-side limit clamps every pixel there."""
    seen = []
    # a value no other setting uses, so seeing it proves the fast scan asked for it
    cfg = IQEConfig(**FAST, fastscan_limit_a=3e-4)
    real = acquire.configure_voltage_source_b

    def spy(smu, c, log=None):
        seen.append(c.nvled_compliance)
        return real(smu, c, log)

    acquire.configure_voltage_source_b = spy
    try:
        with tempfile.TemporaryDirectory() as tmp:
            run_iqe_measurement(cfg, simulate=True, base_dir=tmp, log=QUIET)
    finally:
        acquire.configure_voltage_source_b = real
    assert 3e-4 in seen, f"the fast scan never applied its own limit: {sorted(set(seen))}"
    assert IQEConfig().fastscan_limit_a > IQEConfig().vramp_lowv_limit_a, (
        "the fast-scan limit must clear the ~30 uA that -1.35 V draws")


def test_pick_limit_never_clamps_the_real_iv_curve():
    """Run 261014-0007: a limit tied to the density knee clamped 8 points at 10 uA."""
    from measure_iqe.acquire import pick_limit
    cfg = IQEConfig()
    # the measured curve for pixel 124,135, high current -> low current, amps
    curve = [1.194e-4, 1.161e-4, 1.108e-4, 9.60e-5, 7.12e-5, 4.34e-5, 3.03e-5,
             2.10e-5, 1.40e-5, 8.65e-6, 5.52e-6, 3.16e-6, 1.55e-6, 6.42e-7, 4.24e-7]
    i_last = None
    for i in curve:
        lim = pick_limit(i_last, cfg)
        assert i < 0.995 * lim, f"{i:g} A would clamp at the {lim:g} A limit"
        i_last = i
    # and it must still drop to the fine limit once the current is small, or the
    # whole point of High-C (the limit picks the range) is lost
    assert pick_limit(4.24e-7, cfg) == cfg.vramp_lowv_limit_a
    # the first point knows nothing, so it must take the widest
    assert pick_limit(None, cfg) == cfg.vramp_highv_limit_a
    assert pick_limit(float("nan"), cfg) == cfg.vramp_highv_limit_a


def test_segment_limits_are_applied_not_just_configured():
    """-2.6 V needs ~175 uA and the rising side needs 10 uA: no single limit spans it."""
    seen = []
    cfg = IQEConfig(**FAST)
    real = acquire.configure_voltage_source_b

    def spy(smu, c, log=None):
        seen.append(c.nvled_compliance)
        return real(smu, c, log)

    acquire.configure_voltage_source_b = spy
    try:
        with tempfile.TemporaryDirectory() as tmp:
            run_iqe_measurement(cfg, simulate=True, base_dir=tmp, log=QUIET)
    finally:
        acquire.configure_voltage_source_b = real
    assert cfg.vramp_lowv_limit_a in seen, f"fine limit never applied: {set(seen)}"
    assert cfg.vramp_highv_limit_a in seen, f"wide limit never applied: {set(seen)}"
    assert len(set(seen)) >= 2, "the compliance never changed across the sweep"


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


def _cli(argv) -> tuple[int, str]:
    """Run the CLI with its output captured, returning (exit code, stdout+stderr).

    These assertions are about exit codes and files. Letting the CLI print would
    bury the test results under a full config dump, the parameter table and a run
    log — which reads like a failure even when everything passes. Both streams are
    captured because config errors correctly go to stderr.
    """
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
        code = cli_main(argv)
    return code, buf.getvalue()


def test_cli_reports_a_bad_config_key_and_exits_2():
    """Deliberate negative test: `no_such_field` is not an IQEConfig field, so the
    CLI must refuse it by name rather than ignore it."""
    code, out = _cli(["--set", "no_such_field=1"])
    assert code == 2, "an unknown config key must be a hard error"
    assert "no_such_field" in out, "the error must name the offending key"


def test_cli_runs_a_simulated_measurement():
    with tempfile.TemporaryDirectory() as tmp:
        code, out = _cli(["--print-config"])
        assert code == 0 and '"sample_name"' in out
        code, out = _cli(["--list-params"])
        assert code == 0 and "min_pixel_separation" in out
        argv = ["--simulate", "--out", tmp, "--set", "roi=[0,0,3,3]", "--set", "block_size=4",
                "--set", "pixels_per_block=1", "--set", "n_fine=4", "--set", "warmup_ms=0"]
        assert _cli(argv)[0] == 0
        assert list(Path(tmp).glob("*/NVLEDSweep/Sweep/*.csv")), "CLI produced no sweep output"


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"ok  {name}")
    print("all checks passed")
