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

import numpy as np

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
    """configs/standard.yaml is documentation — it must stay loadable and in sync."""
    example = Path(__file__).parent / "configs" / "standard.yaml"
    if not example.exists():
        return
    try:
        loaded = IQEConfig.from_file(example)
    except ImportError:
        return                                             # PyYAML not installed
    assert loaded == IQEConfig(), (
        "configs/standard.yaml drifted from IQEConfig defaults. That file is a readable "
        "COPY of the defaults, not the source of them: editing it changes nothing at "
        "runtime. To change a default, edit measure_iqe/config.py; for one array use "
        "--array NAME; for one run use --set.")


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


def test_every_swept_pixel_starts_from_a_blank_frame():
    """The first swept pixel must not inherit whatever the fast scan left lit.

    Run 261014-0048 pixel 33,32 (the FIRST measured) had a reference exactly 50% of
    its lit current at all 30 biases: two emitters lit, one still on at bitval 0.
    run_fast_scan used to return without blanking.
    """
    calls = []

    class Spy:
        def __init__(self, inner):
            self._inner = inner

        def set_pixel(self, x, y, bit):
            calls.append(("set", x, y, bit))
            return self._inner.set_pixel(x, y, bit)

        def blank_frame(self):
            calls.append(("blank",))
            return self._inner.blank_frame()

        def __getattr__(self, k):
            return getattr(self._inner, k)

    real_connect = acquire.connect_sim

    def wrapped(cfg, log):
        pm, smu, smile = real_connect(cfg, log)
        return pm, smu, Spy(smile)

    acquire.connect_sim = wrapped
    try:
        with tempfile.TemporaryDirectory() as tmp:
            run_iqe_measurement(IQEConfig(**FAST), simulate=True, base_dir=tmp, log=QUIET)
    finally:
        acquire.connect_sim = real_connect

    # every drive-level write must have a blank somewhere before it with no other
    # pixel left lit in between -- i.e. the call right before each sweep bitval write
    sets = [i for i, c in enumerate(calls) if c[0] == "set" and c[3] != 0]
    assert sets, "no pixels were lit at all"
    for i in sets:
        prev = calls[i - 1]
        assert prev[0] == "blank" or (prev[0] == "set" and prev[1:3] == calls[i][1:3]), (
            f"pixel {calls[i][1:3]} was lit while {prev} was still in the frame")


def test_nothing_touches_the_instruments_after_parking():
    """Ctrl-C parked the hardware, then the poll loop issued one more read:

        !! Ctrl-C: parking the hardware (outputs off, frame blanked)
        read_buffers error: VI_ERROR_INV_OBJECT ... session ... is invalid
    """
    import threading
    import time as _time
    from measure_iqe import safeexit

    calls = []

    def loop():
        for i in range(200):
            if safeexit.parked():
                return
            calls.append(i)
            _time.sleep(0.005)

    with safeexit.safe_exit(lambda: calls.append("PARK")):
        t = threading.Thread(target=loop)
        t.start()
        _time.sleep(0.05)
        safeexit.PARKED.set()               # what a signal handler does first
        t.join(timeout=5)
        assert not t.is_alive(), "the loop ignored the park flag"
    assert calls, "the loop never ran"
    after = calls[calls.index("PARK") + 1:] if "PARK" in calls else []
    assert not after, f"{len(after)} instrument call(s) after parking"


def test_dwell_traces_are_free_and_do_not_shorten_the_dwell():
    """Polling through the dwell must cost no bench time and change no reported value.

    The sweep waits settle_s and then integrates for ~0.5 s: 99% of the dwell was being
    discarded. Keeping it is only worth doing if the wait stays exactly as long.
    """
    import time as _time

    with tempfile.TemporaryDirectory() as tmp:
        # settle_scale=1 makes the dwells real, so keep every one of them small --
        # the leakage reference settles twice per point and dominates otherwise.
        over = dict(FAST, settle_scale=1.0, vramp_lowv_settle_s=0.6,
                    vramp_highv_settle_s=0.6, leak_ref_settle_s=0.02,
                    trace_settling_min_s=0.3, trace_settling_period_s=0.05,
                    n_fine=4, max_sweep_pixels=2, block_size=8, pixels_per_block=1)

        t0 = _time.perf_counter()
        off = run_iqe_measurement(IQEConfig(**over, trace_settling=False),
                                  simulate=True, base_dir=tmp, log=QUIET)
        t_off = _time.perf_counter() - t0

        t0 = _time.perf_counter()
        on = run_iqe_measurement(IQEConfig(**over, trace_settling=True),
                                 simulate=True, base_dir=tmp, log=QUIET)
        t_on = _time.perf_counter() - t0

        npz = on.run_dir / "NVLEDSweep" / "Sweep" / "settling_traces.npz"
        assert npz.exists(), "no traces written"
        import numpy as _np
        with _np.load(npz) as d:
            assert len(d.files) == on.n_points, (len(d.files), on.n_points)
            for k in d.files:
                t, i, p = d[k]
                assert t.size >= 2, f"{k} has {t.size} samples"
                assert t[-1] <= 0.6 + 1e-6, f"{k} polled past the dwell: {t[-1]}"
                assert _np.isfinite(i).all() and _np.isfinite(p).all()

        # the dwell must not have been shortened OR meaningfully lengthened
        assert off.run_dir != on.run_dir, "two runs collided into one directory"
        assert not (off.run_dir / "NVLEDSweep" / "Sweep" / "settling_traces.npz").exists()
        assert t_on >= t_off * 0.9, f"tracing shortened the run: {t_on:.1f}s vs {t_off:.1f}s"
        assert t_on <= t_off * 1.6, f"tracing added too much overhead: {t_on:.1f}s vs {t_off:.1f}s"


def test_calibration_approaches_a_bias_the_way_the_sweep_does():
    """It must take sweep-sized steps from the HIGHER current, not jump 0.5 V.

    Run 261014-1156 jumped -1.30 -> -0.80 V, an 18x larger step than the sweep takes,
    measured 85 s, and produced a 45-hour sweep estimate. It also landed on the 100 uA
    range where the sweep uses 10 uA, which under High-C is a different measurement.
    """
    import os
    from measure_iqe import run_calibration

    seen = []
    real = acquire.configure_voltage_source_b

    class Spy:
        def __init__(self, inner):
            self._inner = inner

        def set_voltage(self, ch, v):
            if ch == "b":
                seen.append(float(v))
            return self._inner.set_voltage(ch, v)

        def __getattr__(self, k):
            return getattr(self._inner, k)

    real_connect = acquire.connect_sim

    def wrapped(cfg, log):
        pm, smu, smile = real_connect(cfg, log)
        return pm, Spy(smu), smile

    acquire.connect_sim = wrapped
    cwd = os.getcwd()
    cfg = IQEConfig(**FAST, calibrate_seconds=0.3, calibrate_approach_s=0.1,
                    calibrate_pixels=1, calibrate_approach_steps=3)
    try:
        with tempfile.TemporaryDirectory() as tmp:
            try:
                os.chdir(tmp)
                run_calibration(cfg, simulate=True, base_dir=tmp, log=QUIET)
            finally:
                os.chdir(cwd)      # Windows cannot delete the cwd
    finally:
        acquire.connect_sim = real_connect

    n_low = max(2, round(cfg.n_fine * cfg.vramp_low_frac))
    step = abs(cfg.nvled_v_coarse_start - cfg.vramp_knee_v) / n_low
    target = cfg.nvled_v_coarse_start                     # the faint end, -0.80
    want = [round(target - k * step, 4) for k in (3, 2, 1)]
    rounded = [round(v, 4) for v in seen]
    for v in want:
        assert v in rounded, f"approach step {v} never set; saw {sorted(set(rounded))}"
    # and every approach step must be MORE negative than the target: the sweep runs
    # high current -> low current, so it arrives from above, not below
    for v in want:
        assert v < target, f"{v} is on the low-current side of {target}"

    # NOTHING may be driven past the sweep's own range. -2.6 V is the first sweep point,
    # so approaching it would mean -2.93 V: harder than the sweep ever drives, straight
    # into compliance, on a real device.
    lo = min(cfg.nvled_v_coarse_start, cfg.nvled_v_coarse_stop)
    assert min(seen) >= lo - 1e-9, (
        f"drove to {min(seen):+.3f} V, past the sweep's {lo:+.3f} V limit")


def test_unsettled_captures_raise_the_recommendation_not_vanish():
    """A capture that never settled is the SLOWEST one; dropping it biases low."""
    import os
    from measure_iqe import run_calibration
    from measure_iqe.calibrate import CALIBRATION_VERSION, load_calibration

    with tempfile.TemporaryDirectory() as tmp:
        cwd = os.getcwd()
        os.chdir(tmp)
        try:
            # a window far too short to resolve anything: every capture is unsettled
            cfg = IQEConfig(**FAST, calibrate_seconds=0.4, calibrate_pixels=1,
                            calibrate_margin=2.0)
            path = run_calibration(cfg, simulate=True, base_dir=tmp, log=QUIET)
            import yaml
            data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
            if not data["never_settled"]:
                return                      # the rig settled instantly; nothing to assert
            assert data["lower_bound"] is True
            rec = load_calibration(path)
            for k in ("vramp_lowv_settle_s", "vramp_highv_settle_s"):
                if k in rec:
                    assert rec[k] >= cfg.calibrate_seconds * cfg.calibrate_margin - 1e-9, (
                        f"{k}={rec[k]} is below the window that failed to settle")
        finally:
            os.chdir(cwd)


def test_array_flag_names_the_run_and_picks_up_its_calibration():
    """--array NAME must set sample_name AND apply that array's measured settings."""
    import os
    from measure_iqe.calibrate import CALIBRATION_VERSION, calibration_path
    from measure_iqe.cli import build_config, build_parser

    with tempfile.TemporaryDirectory() as tmp:
        cwd = os.getcwd()
        os.chdir(tmp)                       # calibration_path is relative to cwd
        try:                                # ... and chdir back before the cleanup
            args = build_parser().parse_args(["--array", "Array42_IQE"])
            cfg = build_config(args)
            assert cfg.sample_name == "Array42_IQE"
            # no calibration yet: the defaults stand, and nothing pretends otherwise
            assert cfg.vramp_lowv_settle_s == IQEConfig().vramp_lowv_settle_s

            path = calibration_path("Array42_IQE")
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps({
                "calibration_version": CALIBRATION_VERSION, "sample_name": "Array42_IQE",
                "recommended": {"vramp_lowv_settle_s": 1.25}}), encoding="utf-8")
            path.rename(path.with_suffix(".yaml")) if False else None

            args = build_parser().parse_args(["--array", "Array42_IQE"])
            cfg = build_config(args)
            assert cfg.sample_name == "Array42_IQE"
            assert cfg.vramp_lowv_settle_s == 1.25, "the calibration was not applied"

            # an explicit --set still wins over the calibration
            args = build_parser().parse_args(
                ["--array", "Array42_IQE", "--set", "vramp_lowv_settle_s=9.0"])
            assert build_config(args).vramp_lowv_settle_s == 9.0
        finally:
            os.chdir(cwd)


def test_calibration_round_trips_and_is_validated():
    """--calibrate measures this array; --calibration applies it. Nothing in between."""
    from measure_iqe import run_calibration
    from measure_iqe.calibrate import CALIBRATION_VERSION, load_calibration

    with tempfile.TemporaryDirectory() as tmp:
        cfg = IQEConfig(**FAST, calibrate_seconds=1.5, calibrate_pixels=1)
        path = run_calibration(cfg, simulate=True, base_dir=tmp, log=QUIET)
        assert path.exists()
        # the transients must be kept: they are the longest records the project has and
        # the only data that can test whether the dwell may be extrapolated
        npz = path.parent.parent / "calibration_traces.npz"
        if not npz.exists():
            npz = next(Path(tmp).rglob("calibration_traces.npz"))
        import numpy as _np
        with _np.load(npz) as d:
            assert len(d.files) >= 3, f"only {len(d.files)} transients kept"
            for k in d.files:
                t, i = d[k]
                assert t.size >= 2 and _np.isfinite(i).all()
        rec = load_calibration(path)
        assert rec, "calibration recommended nothing"
        fields = {f.name for f in dataclasses.fields(IQEConfig)}
        assert set(rec) <= fields, f"unknown keys: {set(rec) - fields}"
        # the floor: a measured 0 means "faster than we sample", not "no wait"
        for k in ("vramp_lowv_settle_s", "vramp_highv_settle_s"):
            if k in rec:
                assert rec[k] >= 2 * cfg.calibrate_period_s, (k, rec[k])
        # it must actually reach the config, and --set must still win over it
        applied = IQEConfig.from_dict({**IQEConfig().to_dict(), **rec})
        for k, v in rec.items():
            assert getattr(applied, k) == v

        # a file that is not a calibration must raise, not apply nothing silently
        bad = Path(tmp) / "nope.json"
        bad.write_text('{"hello": 1}', encoding="utf-8")
        try:
            load_calibration(bad)
        except ValueError:
            pass
        else:
            raise AssertionError("a non-calibration file was accepted")

        # and so must one from a different calibration version
        import json as _json
        stale = Path(tmp) / "stale.json"
        stale.write_text(_json.dumps(
            {"calibration_version": CALIBRATION_VERSION - 1, "recommended": {}}),
            encoding="utf-8")
        try:
            load_calibration(stale)
        except ValueError:
            pass
        else:
            raise AssertionError("a stale calibration version was accepted")


def test_settle_time_criterion():
    """The documented criterion, on traces whose answer is known by construction."""
    from measure_iqe.calibrate import settle_time
    import math

    t = np.linspace(0, 20, 2000)
    assert 3 < settle_time(t, 1e-7 * (1 - np.exp(-t / 1.0)), 0.01) < 7
    assert math.isnan(settle_time(t, 1e-7 * (1 + 0.5 * (1 + t) ** -0.4), 0.01))
    # an already-settled trace reports half a smoothing window, never 0: no measurement
    # can resolve below the interval it smooths over
    from measure_iqe.calibrate import SMOOTH_MAX_S
    assert 0 < settle_time(t, np.full_like(t, 1e-7), 0.01) <= SMOOTH_MAX_S


def test_resume_skips_finished_pixels_and_keeps_their_data():
    """A run killed mid-sweep must continue, not restart and not duplicate."""
    from measure_iqe.acquire import completed_pixels

    with tempfile.TemporaryDirectory() as tmp:
        first = _run(tmp)
        agg = next((first.run_dir / "NVLEDSweep" / "Sweep").glob("aggregated_*.csv"))
        rows = list(csv.DictReader(agg.open()))
        done = completed_pixels(first.run_dir)
        assert done, "no completed pixels found in a finished run"
        assert len(done) == len({(int(float(r["X"])), int(float(r["Y"]))) for r in rows})

        # resume a COMPLETE run: nothing left to do, and nothing lost
        again = run_iqe_measurement(IQEConfig(**FAST), simulate=True, base_dir=tmp,
                                    log=QUIET, run_dir=first.run_dir)
        after = list(csv.DictReader(agg.open()))
        assert again.run_dir == first.run_dir
        assert len(after) == len(rows), f"{len(rows)} rows became {len(after)} on resume"

        # drop one pixel's rows and resume: exactly that pixel is re-measured
        victim = sorted(done)[-1]
        keep = [r for r in rows
                if (int(float(r["X"])), int(float(r["Y"]))) != victim]
        with agg.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=rows[0].keys())
            w.writeheader()
            w.writerows(keep)
        assert victim not in completed_pixels(first.run_dir)
        run_iqe_measurement(IQEConfig(**FAST), simulate=True, base_dir=tmp, log=QUIET,
                            run_dir=first.run_dir)
        final = list(csv.DictReader(agg.open()))
        assert len(final) == len(rows), f"expected {len(rows)} rows, got {len(final)}"
        assert victim in completed_pixels(first.run_dir)


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
