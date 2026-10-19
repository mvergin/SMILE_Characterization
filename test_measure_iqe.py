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
import sys
import tempfile
from pathlib import Path

import numpy as np

from measure_iqe import (IQEConfig, run_file, run_iqe_measurement,
                         select_sweep_pixels)
from measure_iqe import acquire
from measure_iqe.cli import main as cli_main
from measure_iqe.cli import parse_override

QUIET = lambda _msg: None                                 # noqa: E731 - tests stay silent

# small but exercises every branch: 2x2 blocks, both ramp segments, raw samples on
# NOT a real array. write_calibration() deliberately writes to every calibration root,
# including the install directory, so a sweep started from anywhere finds the file. The
# tests inherited the default sample_name -- Array42_IQE -- and so every calibration
# test silently overwrote the bench's measured calibration for the real array with
# simulator output (0.8 s in place of 54.47 s). A test must never name a real array.
TEST_ARRAY = "ZZZ_SimTestArray"

FAST = dict(sample_name=TEST_ARRAY,
            roi=(0, 0, 7, 7), block_size=4, pixels_per_block=1, n_fine=6, n_coarse=4,
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
                dict(roi=(0, 0, 999, 3)), dict(bitvals=(99,)), dict(nvled_range_i=0.003),
                # the retry gate is optical, and a non-zero reference bitval emits:
                # bitval 1 is 11.6% of bitval 15 at -2.60 V (run 261015-2326, section 20)
                dict(leak_ref_bitval=1, leak_ref_tries=3),
                # the dead-pixel test runs at bias index 0, which low_to_high makes the
                # FAINTEST: 0.21 nW against a 2.0 nW floor drops every pixel (section 23)
                dict(sweep_direction="low_to_high", dead_abort=True)):
        try:
            IQEConfig(**bad).validate()
        except ValueError:
            continue
        raise AssertionError(f"{bad} should not validate")
    # ... but a non-zero reference bitval is fine as long as nothing retries on it
    IQEConfig(leak_ref_bitval=1, leak_ref_tries=1).validate()
    # ... and the direction A/B is allowed once the dead-pixel test is off
    IQEConfig(sweep_direction="low_to_high", dead_abort=False).validate()
    try:
        IQEConfig.from_dict({"not_a_field": 1})
    except ValueError as exc:
        assert "not_a_field" in str(exc)
    else:
        raise AssertionError("unknown keys must be rejected")


def test_config_roundtrips_through_a_file():
    cfg = IQEConfig(n_fine=17, sweep_direction="low_to_high", dead_abort=False,
                    roi=(1, 2, 3, 4))
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
    """configs/standard.yaml is documentation — it must stay loadable and in sync.

    Values AND keys. Comparing only the loaded config against the defaults lets a
    MISSING key pass, because from_dict falls back to the default for anything absent:
    `sweep_rotate` and `bias_major_precondition` were both added without this test
    noticing. ROADMAP item 3.
    """
    example = Path(__file__).parent / "configs" / "standard.yaml"
    if not example.exists():
        return
    try:
        import yaml
        raw = yaml.safe_load(example.read_text(encoding="utf-8")) or {}
        loaded = IQEConfig.from_file(example)
    except ImportError:
        return                                             # PyYAML not installed
    assert loaded == IQEConfig(), (
        "configs/standard.yaml drifted from IQEConfig defaults. That file is a readable "
        "COPY of the defaults, not the source of them: editing it changes nothing at "
        "runtime. To change a default, edit measure_iqe/config.py; for one array use "
        "--array NAME; for one run use --set.")
    want = {f.name for f in dataclasses.fields(IQEConfig)}
    missing, extra = sorted(want - set(raw)), sorted(set(raw) - want)
    assert not missing, (
        f"configs/standard.yaml is MISSING {len(missing)} field(s): {missing}. A missing "
        "key silently inherits the default and the equality check above cannot see it, "
        "which is how sweep_rotate and bias_major_precondition got in unannotated. That "
        "file is what a colleague is handed: every field belongs in it, with a comment.")
    assert not extra, (
        f"configs/standard.yaml has {len(extra)} field(s) IQEConfig does not: {extra}")


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
        agg = run_file(result.run_dir,
                       "NVLEDSweep/Sweep/aggregated_bitval=15_mean_std.csv")
        rows = list(csv.DictReader(agg.open()))
        assert len(rows) == result.n_points
        assert AGG_COLS == list(rows[0])[:len(AGG_COLS)], "SMILE_IQE_Analysis column contract"
        assert "NVLED_CURR_MEDIAN" in rows[0], "raw-sample stats must be written"
        assert float(rows[0]["PM400_POWER_MEAN"]) > 0

        fast = run_file(result.run_dir, "FastScan/aggregated_bitval=15_mean_std.csv")
        assert len(list(csv.DictReader(fast.open()))) == 64      # every ROI pixel scanned

        saved = json.loads(run_file(
            result.run_dir,
            "NVLEDSweep/Sweep/raw_data/Sweep_config.json").read_text(
                encoding="utf-8"))["settings"]
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
             dict(sweep_direction="low_to_high", dead_abort=False)]
    with tempfile.TemporaryDirectory() as tmp:
        for over in modes:
            result = _run(tmp, **over)
            assert result.n_points > 0, over
            assert result.run_dir.is_dir()


def test_leakage_reference_is_measured_and_written():
    """The bitval-0 reference is the low-J correction, not an optional extra."""
    with tempfile.TemporaryDirectory() as tmp:
        result = _run(tmp)
        agg = next((result.run_dir / "NVLEDSweep" / "Sweep").glob("*aggregated_*.csv"))
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
        agg = next((off.run_dir / "NVLEDSweep" / "Sweep").glob("*aggregated_*.csv"))
        assert "NVLED_CURR_REF" not in agg.read_text().splitlines()[0]


def test_a_failed_bitval0_write_is_retried_and_then_reported():
    """Section 19: the FPGA does not always write the frame correctly, so the bitval-0
    reference is verified optically and re-written when it fails.

    The evidence that it is the WRITE and not the pixel is intermittency: (56,51)
    referenced clean in 2 of 6 runs and contaminated in 4, and single points flip inside
    one sweep. The gate is optical because a healthy backplane conducts 58 % of the
    reading at -0.80 V without emitting.
    """
    class _Smile:
        def __init__(self):
            self.writes = []

        def set_pixel(self, x, y, bv):
            self.writes.append((x, y, bv))

        def blank_frame(self):
            self.writes.append("blank")

    LIT = 1e-9

    def _stub(ref_powers):
        """Lit reading, then one reference per attempt from `ref_powers`."""
        seq = [LIT] + list(ref_powers)
        def _f(*a, **k):
            w = seq.pop(0) if len(seq) > 1 else seq[0]
            return (w, 0.0, 1e-5, 0.0, -1e-5, 0.0, 0.0, 0.0)
        return _f

    BAD, OK = 0.6e-9, 2e-13          # 60 % of the lit light, and 2e-4 of it
    cases = (
        # ref readings per attempt      tries  warns  what it exercises
        ([OK],                              1, False, "healthy: one write, no retry"),
        ([BAD, OK],                         2, False, "one failure, second write lands"),
        ([BAD, BAD, OK],                    3, False, "two failures, third lands"),
        ([BAD, BAD, BAD],                   3, True,  "never lands: give up and report"),
    )
    real = acquire.acquire_point
    try:
        for refs, want_tries, want_warn, who in cases:
            acquire.acquire_point = _stub(refs)
            acquire._ref_warned.clear()
            cfg = IQEConfig(**dict(FAST, leak_reference=True, perpoint_range=False,
                                   save_raw_samples=False, leak_ref_tries=3,
                                   leak_ref_max_frac=0.25))
            smile, said = _Smile(), []
            row, _ = acquire.measure_point(None, None, smile, 7, 9, 15, -2.6, 1e-3,
                                           8, 8, cfg, 0.0, said.append, restore=False)
            assert row["LEAK_REF_TRIES"] == want_tries, \
                f"{who}: {row['LEAK_REF_TRIES']} tries, wanted {want_tries}"
            # one bitval-0 frame write per attempt, and no more
            zeros = [w for w in smile.writes if w == (7, 9, 0)]
            assert len(zeros) == want_tries, f"{who}: {len(zeros)} writes of bitval 0"
            warned = [m for m in said if "still\ncarries" in m or "still carries" in m]
            assert bool(warned) == want_warn, f"{who}: warned={bool(warned)}"
            # the accepted reference is the LAST one taken, not the first
            assert row["PM400_POWER_REF"] == refs[want_tries - 1], who
            assert abs(row["LEAK_REF_FRAC"] - refs[want_tries - 1] / LIT) < 1e-9, who

        # and it must be switchable off, to reproduce runs taken before it existed
        acquire.acquire_point = _stub([BAD, OK])
        acquire._ref_warned.clear()
        cfg = IQEConfig(**dict(FAST, leak_reference=True, perpoint_range=False,
                               save_raw_samples=False, leak_ref_tries=1))
        row, _ = acquire.measure_point(None, None, _Smile(), 7, 9, 15, -2.6, 1e-3,
                                       8, 8, cfg, 0.0, lambda m: None, restore=False)
        assert row["LEAK_REF_TRIES"] == 1 and row["PM400_POWER_REF"] == BAD
    finally:
        acquire.acquire_point = real
        acquire._ref_warned.clear()


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

        npz = run_file(on.run_dir, "NVLEDSweep/Sweep/settling_traces.npz")
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
        assert not list((off.run_dir / "NVLEDSweep" / "Sweep").glob("*settling_traces.npz"))
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


def _sweep_grid(run_dir):
    """(X, Y, NVLED_V) of every aggregated row, plus the column set."""
    import csv as _csv
    grid, cols = set(), None
    for agg in (Path(run_dir) / "NVLEDSweep" / "Sweep").glob("*aggregated_bitval=*.csv"):
        with agg.open() as f:
            rdr = _csv.DictReader(f)
            cols = tuple(rdr.fieldnames or ())
            for row in rdr:
                grid.add((int(float(row["X"])), int(float(row["Y"])),
                          round(float(row["NVLED_V"]), 4)))
    return grid, cols


def test_a_run_says_what_produced_it():
    """Every output file carries <sample>_<stamp>_, and the run records its commit.

    A CSV copied out of its directory must still name its run, and a run must name the
    code and the calibration behind it -- see measure_iqe/provenance.py.
    """
    import json

    from measure_iqe import run_iqe_measurement

    with tempfile.TemporaryDirectory() as tmp:
        cal = Path(tmp) / "fake_calibration.yaml"
        cal.write_text("sample_name: ZZZ_Prov\nrecommended: {}\n", encoding="utf-8")
        cfg = IQEConfig(**{**FAST, "n_fine": 4, "roi": (0, 0, 3, 3),
                           "sample_name": "ZZZ_Prov",
                           "calibration_source": str(cal)})
        res = run_iqe_measurement(cfg, simulate=True, base_dir=tmp, log=QUIET)

    # Impossible to assert after the TemporaryDirectory closes, so collect inside.
        files = [p for p in res.run_dir.rglob("*") if p.is_file()]
        names = {p.name for p in files}
        stamp = res.run_dir.name.split("_", 1)[0]
        pfx = f"ZZZ_Prov_{stamp}_"

        data = [p for p in files if p.suffix in (".csv", ".npz")]
        assert data, "the run wrote no data files"
        unprefixed = [p.name for p in data if not p.name.startswith(pfx)]
        assert not unprefixed, f"data files without the run prefix: {unprefixed}"

        prov_path = res.run_dir / "RUN_PROVENANCE.json"
        assert prov_path.exists(), f"no RUN_PROVENANCE.json in {names}"
        prov = json.loads(prov_path.read_text(encoding="utf-8"))
        for key in ("commit", "dirty", "started", "host", "sample_name"):
            assert key in prov, f"provenance is missing {key}: {sorted(prov)}"
        assert prov["sample_name"] == "ZZZ_Prov"
        assert isinstance(prov["dirty"], bool)

        copied = [n for n in names if "calibration_applied" in n]
        assert copied, f"the applied calibration was not copied in: {sorted(names)}"
        assert (res.run_dir / copied[0]).read_text(encoding="utf-8") == \
            cal.read_text(encoding="utf-8"), "the copy differs from the calibration"

        # and the saved config must name the file, not just its values
        cfgs = [p for p in files if p.name.endswith("_config.json")]
        assert cfgs, "no saved config"
        saved = json.loads(cfgs[0].read_text(encoding="utf-8"))
        assert saved["settings"]["calibration_source"] == str(cal)
        assert saved["provenance"]["commit"] == prov["commit"]


def test_bias_major_measures_the_same_grid_as_pixel_major():
    """Swapping the loops must change WHEN a point is measured, not WHICH points."""
    from measure_iqe import run_iqe_measurement

    common = dict(FAST, n_fine=5, roi=(0, 0, 7, 7), block_size=4, pixels_per_block=1,
                  block_seed=0, dead_abort=False)
    out = {}
    for order in ("pixel_major", "bias_major"):
        with tempfile.TemporaryDirectory() as tmp:
            cfg = IQEConfig(**common, sweep_order=order)
            res = run_iqe_measurement(cfg, simulate=True, base_dir=tmp, log=QUIET)
            out[order] = _sweep_grid(res.run_dir)
    (g_px, c_px), (g_bm, c_bm) = out["pixel_major"], out["bias_major"]
    assert g_px, "pixel_major measured nothing"
    assert g_bm == g_px, (
        f"different grids: bias_major-only {sorted(g_bm - g_px)[:4]}, "
        f"pixel_major-only {sorted(g_px - g_bm)[:4]}")
    assert c_bm == c_px, f"different columns:\n  {c_px}\n  {c_bm}"


def test_bias_major_precondition_touches_every_pixel_once():
    """The conditioning pass must reach every selected pixel exactly once, hold the
    conditioning voltage while it does, and leave the measured grid untouched."""
    from measure_iqe import run_iqe_measurement

    common = dict(FAST, n_fine=5, roi=(0, 0, 7, 7), block_size=4, pixels_per_block=1,
                  block_seed=0, dead_abort=False, sweep_order="bias_major",
                  warmup_ms=1.0)
    grids, logs = {}, {}
    for pre in (False, True):
        lines = []
        with tempfile.TemporaryDirectory() as tmp:
            cfg = IQEConfig(**common, bias_major_precondition=pre)
            res = run_iqe_measurement(cfg, simulate=True, base_dir=tmp,
                                      log=lines.append)
            grids[pre] = _sweep_grid(res.run_dir)[0]
        logs[pre] = [ln for ln in lines if "preconditioned" in ln]

    assert not logs[False], "preconditioning ran with the flag off"
    assert len(logs[True]) == 1, f"expected one preconditioning line, got {logs[True]}"
    npx = len({(x, y) for x, y, _ in grids[True]})
    assert f"preconditioned {npx} pixel(s)" in logs[True][0], logs[True][0]
    # and the soak must not move the sweep: same points, both ways
    assert grids[True] == grids[False], "preconditioning changed the measured grid"


def _agg_rows(run_dir, phase="NVLEDSweep/Sweep"):
    """Every aggregated row of a phase, as dicts."""
    import csv as _csv
    out = []
    for agg in sorted((Path(run_dir).joinpath(*phase.split("/"))).glob(
            "*aggregated_bitval=*.csv")):
        with agg.open() as f:
            out += list(_csv.DictReader(f))
    return out


def test_the_leakage_reference_is_cached_below_the_threshold():
    """One reference per pixel below ``leak_ref_per_point_above_v``, per point above it.

    The saving is 2100 of 3000 reference reads on the default grid. What makes it safe
    is not that the leak is constant -- it rises 256.5 -> 286.1 nA from -0.80 to
    -2.36 V -- but that the signal rises 728x over the same span, so one fixed
    per-pixel value costs under 0.14 % everywhere it is used.

    This checks the mechanism, which is what can silently invert: that the faint side
    keeps its own reference, that the deep side shares ONE value per pixel, and that
    the column says which.
    """
    from measure_iqe import run_iqe_measurement

    common = dict(FAST, n_fine=8, roi=(0, 0, 7, 7), block_size=4, pixels_per_block=1,
                  block_seed=0, dead_abort=False, leak_reference=True,
                  trace_settling=False)
    for order in ("pixel_major", "bias_major"):
        with tempfile.TemporaryDirectory() as tmp:
            cfg = IQEConfig(**common, sweep_order=order,
                            leak_ref_per_point_above_v=-1.0)
            res = run_iqe_measurement(cfg, simulate=True, base_dir=tmp, log=QUIET)
            rows = _agg_rows(res.run_dir)
        assert rows, f"{order}: no sweep rows"
        per_point = [r for r in rows if r["LEAK_REF_CACHED"] == "0"]
        cached = [r for r in rows if r["LEAK_REF_CACHED"] == "1"]
        assert per_point and cached, (
            f"{order}: expected both kinds, got {len(per_point)} per-point and "
            f"{len(cached)} cached over {len(rows)} points")
        # The split is exactly the threshold, in the sweep's sign convention.
        for r in per_point:
            assert float(r["NVLED_V"]) > -1.0,                 f"{order}: {r['NVLED_V']} V measured per point but is below -1.0 V"
        for r in cached:
            assert float(r["NVLED_V"]) <= -1.0,                 f"{order}: {r['NVLED_V']} V cached but is above -1.0 V"
            assert int(r["LEAK_REF_TRIES"]) == 0,                 "a cached point writes no frame, so TRIES must be 0"
        # One value per pixel, not one per point: the whole saving rests on this.
        by_px = {}
        for r in cached:
            by_px.setdefault((r["X"], r["Y"]), set()).add(r["NVLED_CURR_REF"])
        for px, vals in by_px.items():
            assert len(vals) == 1,                 f"{order}: pixel {px} has {len(vals)} cached reference values, want 1"
        # And the faint side did NOT share: those are separate reads.
        assert len({r["NVLED_CURR_REF"] for r in per_point}) > 1,             f"{order}: every per-point reference is identical -- the cache leaked"

    # Off switch: None reproduces every run made before the cache existed.
    with tempfile.TemporaryDirectory() as tmp:
        cfg = IQEConfig(**common, sweep_order="bias_major",
                        leak_ref_per_point_above_v=None)
        res = run_iqe_measurement(cfg, simulate=True, base_dir=tmp, log=QUIET)
        rows = _agg_rows(res.run_dir)
    assert rows and all(r["LEAK_REF_CACHED"] == "0" for r in rows),         "leak_ref_per_point_above_v=None must reference every point"


def test_the_fast_scan_measures_every_bias_and_judges_on_the_first():
    """``fastscan_voltages`` adds passes; it must not change which pixels get swept.

    -1.35 V was picked as a peak-IQE proxy and the measured peak is at -1.05 V, so a
    scan wants both. The risk in adding biases is that the alive/dead test starts
    comparing a pixel at one bias against the population at another, which would
    change the swept set -- so selection reads nvled_v_fastscan ONLY.
    """
    from measure_iqe import run_iqe_measurement

    common = dict(FAST, n_fine=4, roi=(0, 0, 7, 7), block_size=4, pixels_per_block=1,
                  block_seed=0, dead_abort=False, trace_settling=False)
    picked = {}
    for name, extra in (("one", ()), ("three", (-1.05, -1.65))):
        with tempfile.TemporaryDirectory() as tmp:
            cfg = IQEConfig(**common, fastscan_voltages=extra)
            res = run_iqe_measurement(cfg, simulate=True, base_dir=tmp, log=QUIET)
            fs = _agg_rows(res.run_dir, "FastScan")
            picked[name] = {(r[0], r[1]) for r in _sweep_grid(res.run_dir)[0]}
        want = cfg.fastscan_v_list()
        got = {round(float(r["NVLED_V"]), 4) for r in fs}
        assert got == {round(v, 4) for v in want},             f"{name}: fast scan wrote biases {sorted(got)}, expected {sorted(want)}"
        assert len(fs) == 64 * len(want),             f"{name}: {len(fs)} rows for {len(want)} biases x 64 pixels"
        # Every bias covers every pixel, so the passes are comparable.
        for v in want:
            n = sum(1 for r in fs if round(float(r["NVLED_V"]), 4) == round(v, 4))
            assert n == 64, f"{name}: bias {v} covered {n}/64 pixels"
    assert picked["one"] == picked["three"], (
        f"adding fast-scan biases changed the swept pixels: "
        f"{sorted(picked['one'] ^ picked['three'])}")
    # First entry stays first, whatever order the extras are given in.
    assert IQEConfig(nvled_v_fastscan=-1.35,
                     fastscan_voltages=(-1.65, -1.35, -1.05)).fastscan_v_list() ==         (-1.35, -1.65, -1.05), "nvled_v_fastscan must lead, and must not repeat"


def test_bias_major_pays_each_dwell_once_for_all_pixels():
    """The point of the order: n_pixels x the dwell becomes 1 x the dwell."""
    import time as _time
    from measure_iqe import run_iqe_measurement

    # Distinctive settles, so each sleep is identifiable in the recorded list.
    common = dict(FAST, n_fine=4, roi=(0, 0, 7, 7), block_size=4, pixels_per_block=1,
                  block_seed=0, dead_abort=False, settle_scale=1.0,
                  vramp_lowv_settle_s=0.011, vramp_highv_settle_s=0.013,
                  leak_ref_settle_s=0.017, pixel_switch_settle_s=0.019,
                  leak_reference=True, trace_settling=False, warmup_ms=0.0,
                  # This test counts reference settles to prove bias_major does not
                  # restore the pixel. The per-pixel cache adds a pre-pass and removes
                  # the reference from the biases below its threshold, so it is turned
                  # off here to keep the count about the restore. The cache has its own
                  # test: test_the_leakage_reference_is_cached_below_the_threshold.
                  leak_ref_per_point_above_v=None)
    seen = {}
    real_sleep = _time.sleep
    for order in ("pixel_major", "bias_major"):
        rec: list[float] = []
        _time.sleep = lambda s, _r=rec: (_r.append(round(float(s), 6)), real_sleep(0))[1]
        try:
            with tempfile.TemporaryDirectory() as tmp:
                cfg = IQEConfig(**common, sweep_order=order)
                res = run_iqe_measurement(cfg, simulate=True, base_dir=tmp, log=QUIET)
                npx = len({(x, y) for x, y, _ in _sweep_grid(res.run_dir)[0]})
        finally:
            _time.sleep = real_sleep
        seen[order] = (rec, npx)

    for order, (rec, npx) in seen.items():
        assert npx >= 2, f"{order}: need >1 pixel for this test to mean anything"
    n_dwell = {o: sum(1 for v in rec if v in (0.011, 0.013))
               for o, (rec, _) in seen.items()}
    npx = seen["bias_major"][1]
    # pixel_major pays the dwell on every (pixel, bias); bias_major on every bias.
    assert n_dwell["pixel_major"] == npx * n_dwell["bias_major"], (
        f"dwells: pixel_major {n_dwell['pixel_major']}, bias_major "
        f"{n_dwell['bias_major']}, {npx} pixels - expected a factor of {npx}")
    # and the frame switch replaces it, once per point
    n_switch = sum(1 for v in seen["bias_major"][0] if v == 0.019)
    assert n_switch == npx * n_dwell["bias_major"], (
        f"{n_switch} switch settles for {npx} pixels x {n_dwell['bias_major']} biases")
    assert 0.019 not in seen["pixel_major"][0], "pixel_major must not switch-settle"
    # the reference is restored in pixel_major and not in bias_major: 2 waits vs 1
    n_ref = {o: sum(1 for v in rec if v == 0.017) for o, (rec, _) in seen.items()}
    assert n_ref["pixel_major"] == 2 * n_ref["bias_major"], (
        f"reference settles: {n_ref} - bias_major blanks next, so it must not "
        f"restore the pixel and wait again")


def test_bias_major_serves_the_dwell_with_a_pixel_lit():
    """The dwell must relax in the state the measurement is taken in.

    Run 261014-2308 paid the dwell with the frame BLANKED -- the previous bias ends
    with a blank for its dark reading, and nothing re-lit before the voltage stepped.
    A blanked frame conducts ~4x the bitval-0 level, so the array settled to a
    different charge, and the run read 2.2-2.9% LOW against the pixel-major run on the
    same 16 pixels at every bias from -1.25 to -0.90 V. Nothing in the suite noticed,
    because the grid and the columns were perfect.

    So: record the order of frame operations, voltage steps and sleeps, and assert that
    the long sleep after each voltage step is preceded by a set_pixel, not a blank.
    """
    import time as _time
    import measure_iqe.acquire as acq
    from measure_iqe import run_iqe_measurement

    # The simulated rig is _SimSmile/_SimSMU inside acquire.py, NOT instruments.sim --
    # patching the latter records nothing and the test passes vacuously, which is what
    # the "test is not biting" assertion at the end exists to catch.
    events: list[tuple] = []
    orig_set, orig_blank = acq._SimSmile.set_pixel, acq._SimSmile.blank_frame
    orig_v = acq._SimSMU.set_voltage
    real_sleep = _time.sleep

    def rec_set(self, x, y, b):
        events.append(("pixel", x, y, b))
        return orig_set(self, x, y, b)

    def rec_blank(self):
        events.append(("blank",))
        return orig_blank(self)

    def rec_v(self, ch, v):
        if ch == "b":
            events.append(("volt", round(float(v), 4)))
        return orig_v(self, ch, v)

    DWELL = 0.0213                      # distinctive: no other sleep is this size
    try:
        acq._SimSmile.set_pixel = rec_set
        acq._SimSmile.blank_frame = rec_blank
        acq._SimSMU.set_voltage = rec_v
        _time.sleep = lambda s: (events.append(("sleep", round(float(s), 6))),
                                 real_sleep(0))[1]
        with tempfile.TemporaryDirectory() as tmp:
            cfg = IQEConfig(**{**FAST, "n_fine": 4, "roi": (0, 0, 7, 7),
                               "block_size": 4, "pixels_per_block": 1, "block_seed": 0,
                               "dead_abort": False, "sweep_order": "bias_major",
                               "settle_scale": 1.0, "vramp_lowv_settle_s": DWELL,
                               "vramp_highv_settle_s": DWELL,
                               "leak_ref_settle_s": 0.003,
                               "pixel_switch_settle_s": 0.007,
                               "trace_settling": False, "warmup_ms": 0.0})
            run_iqe_measurement(cfg, simulate=True, base_dir=tmp, log=QUIET)
    finally:
        acq._SimSmile.set_pixel, acq._SimSmile.blank_frame = orig_set, orig_blank
        acq._SimSMU.set_voltage = orig_v
        _time.sleep = real_sleep

    dwells = [k for k, e in enumerate(events) if e == ("sleep", DWELL)]
    assert dwells, f"no dwell sleep of {DWELL}s found in {len(events)} events"
    checked = 0
    for k in dwells:
        # the voltage step this dwell belongs to, then the frame state it stepped in
        volt = next((j for j in range(k - 1, max(-1, k - 4), -1)
                     if events[j][0] == "volt"), None)
        if volt is None:
            continue                    # a dwell not tied to a bias step; not ours
        frame = next((events[j] for j in range(volt - 1, -1, -1)
                      if events[j][0] in ("pixel", "blank")), None)
        checked += 1
        assert frame is not None and frame[0] == "pixel", (
            f"the dwell for {events[volt]} was served with the frame "
            f"{frame[0] if frame else 'unset'}; a blanked frame conducts ~4x bitval 0, "
            f"so the array settles to a charge the measurement never sees. "
            f"Context: {events[max(0, volt - 3):k + 1]}")
    assert checked >= 2, f"only {checked} bias dwell(s) identified - test is not biting"


def test_completed_biases_finds_whole_biases_not_whole_pixels():
    """--resume in bias_major order skips biases; completed_pixels cannot see them."""
    from measure_iqe.acquire import completed_biases, completed_pixels

    with tempfile.TemporaryDirectory() as tmp:
        sweep = Path(tmp) / "NVLEDSweep" / "Sweep"
        sweep.mkdir(parents=True)
        px = [(1, 1), (2, 2), (3, 3)]
        lines = ["X,Y,NVLED_V,NVLED_CURR_MEAN"]
        for v in (-2.6, -1.3):                       # two complete biases
            lines += [f"{x},{y},{v},1e-6" for x, y in px]
        lines += [f"{px[0][0]},{px[0][1]},-0.8,1e-6"]   # one bias, one pixel: partial
        (sweep / "aggregated_bitval=15_mean_std.csv").write_text(
            "\n".join(lines) + "\n", encoding="utf-8")

        done = completed_biases(tmp)
        assert done == {-2.6, -1.3}, done
        assert -0.8 not in done, "a bias missing two pixels is not complete"
        # And the hazard this exists to avoid. On a bias_major file, completed_pixels
        # takes the MODE of the per-pixel row counts -- 2 here -- and calls every pixel
        # that reached it finished. So it reports all three pixels complete when not one
        # has a full curve, and a --resume driven by it would skip the entire run and
        # measure nothing. bias_major must never ask it.
        assert completed_pixels(tmp) == set(px), (
            "completed_pixels no longer over-reports on a bias_major file; re-check "
            "whether _sweep_bias_major still needs completed_biases")
        assert completed_biases(tmp) != set(px), "the two must not be interchangeable"

def test_local_config_overrides_a_shared_one_but_not_the_command_line():
    """ROADMAP item 3. configs/local.yaml describes THIS MACHINE, so it outranks a
    config file shared between machines -- but never --set or --out, which are typed
    per invocation. And every key it applies has to be logged, or it is the same
    invisible-override problem item 0 was about.
    """
    import measure_iqe.cli as cli
    from measure_iqe.cli import build_config, build_parser

    lines = []
    real = cli.local_overrides

    def fake(log=None, data=None):
        if log and data:
            log(f"applied {cli.LOCAL_CONFIG}: {', '.join(sorted(data))}")
        return dict(data or {})

    try:
        # absent: nothing applied, nothing logged
        cli.local_overrides = lambda log=None: fake(log, None)
        cfg = build_config(build_parser().parse_args([]), lines.append)
        assert cfg.base_dir == IQEConfig().base_dir
        assert not lines, f"an absent local.yaml must log nothing, logged {lines}"

        # present: it wins over the defaults AND announces itself
        cli.local_overrides = lambda log=None: fake(log, {"base_dir": "D:/runs",
                                                          "n_fine": 7})
        cfg = build_config(build_parser().parse_args([]), lines.append)
        assert cfg.base_dir == "D:/runs" and cfg.n_fine == 7, cfg.base_dir
        assert any("local.yaml" in x and "base_dir" in x and "n_fine" in x
                   for x in lines), f"every applied key must be named in the log: {lines}"

        # --out and --set are typed for THIS run, so they outrank it
        args = build_parser().parse_args(["--out", "E:/here", "--set", "n_fine=9"])
        cfg = build_config(args, lines.append)
        assert cfg.base_dir == "E:/here", "--out must beat local.yaml"
        assert cfg.n_fine == 9, "--set must beat local.yaml"

        # ... and a config file shared between machines must NOT put this machine's
        # runs where they do not exist, so local.yaml outranks --config
        with tempfile.TemporaryDirectory() as tmp:
            shared = Path(tmp) / "shared.json"
            shared.write_text(json.dumps({"base_dir": "/somebody/elses/disk",
                                          "n_fine": 21}), encoding="utf-8")
            args = build_parser().parse_args(["--config", str(shared)])
            cfg = build_config(args, lines.append)
            assert cfg.base_dir == "D:/runs", "local.yaml must beat --config on paths"
            assert cfg.n_fine == 7, "local.yaml wins where both set a key"
    finally:
        cli.local_overrides = real

    # the real loader, on a real file, through the real path
    with tempfile.TemporaryDirectory() as tmp:
        f = Path(tmp) / "local.yaml"
        f.write_text("base_dir: Z:/elsewhere\n", encoding="utf-8")
        try:
            import yaml
        except ImportError:
            return
        assert yaml.safe_load(f.read_text(encoding="utf-8")) == {
            "base_dir": "Z:/elsewhere"}, "the file format is plain partial YAML"
        # a top-level list is a mistake worth naming, not a silent no-op
        f.write_text("- base_dir\n", encoding="utf-8")
        assert not isinstance(yaml.safe_load(f.read_text(encoding="utf-8")), dict)


def test_array_flag_names_the_run_and_picks_up_its_calibration():
    """--array NAME must set sample_name AND apply that array's measured settings."""
    import os
    from measure_iqe.calibrate import CALIBRATION_VERSION, calibration_path
    from measure_iqe.cli import build_config, build_parser

    # NOT a real array name. calibration_roots() also searches the install directory,
    # so any array that has actually been calibrated -- Array42_IQE has -- would make
    # the "no calibration yet" branch below pass for the wrong reason.
    name = "ZZZ_UncalibratedTestArray"

    with tempfile.TemporaryDirectory() as tmp:
        cwd = os.getcwd()
        os.chdir(tmp)                       # calibration_path is relative to cwd
        try:                                # ... and chdir back before the cleanup
            assert not calibration_path(name).exists(), (
                f"{name} must not exist on disk for this test to mean anything")
            args = build_parser().parse_args(["--array", name])
            cfg = build_config(args)
            assert cfg.sample_name == name
            # no calibration yet: the defaults stand, and nothing pretends otherwise
            assert cfg.vramp_lowv_settle_s == IQEConfig().vramp_lowv_settle_s

            path = calibration_path(name)
            assert path.parent.parent.resolve() == Path(tmp).resolve(), path
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps({
                "calibration_version": CALIBRATION_VERSION, "sample_name": name,
                "recommended": {"vramp_lowv_settle_s": 1.25}}), encoding="utf-8")
            path.rename(path.with_suffix(".yaml")) if False else None

            args = build_parser().parse_args(["--array", name])
            cfg = build_config(args)
            assert cfg.sample_name == name
            assert cfg.vramp_lowv_settle_s == 1.25, "the calibration was not applied"

            # an explicit --set still wins over the calibration
            args = build_parser().parse_args(
                ["--array", name, "--set", "vramp_lowv_settle_s=9.0"])
            assert build_config(args).vramp_lowv_settle_s == 9.0
        finally:
            os.chdir(cwd)


def test_a_calibration_is_found_from_either_directory():
    """--calibrate in one directory and the sweep in another must still meet.

    calibration_path was cwd-relative only, so the two could silently miss each other
    and the sweep would fall back to another array's settle times with one line in the
    log to say so.
    """
    import os
    from measure_iqe.calibrate import CALIB_DIR, calibration_path, calibration_roots

    install = Path(acquire.__file__).resolve().parent.parent
    cwd = os.getcwd()
    with tempfile.TemporaryDirectory() as tmp:
        try:
            os.chdir(tmp)
            roots = calibration_roots()
            assert Path(tmp).resolve() in [r.resolve() for r in roots]
            assert install in [r.resolve() for r in roots], (install, roots)

            # a calibration living beside the package must be found from anywhere
            beside = install / CALIB_DIR / "FoundFromAnywhere.yaml"
            beside.parent.mkdir(parents=True, exist_ok=True)
            beside.write_text("{}", encoding="utf-8")
            try:
                assert calibration_path("FoundFromAnywhere").resolve() == beside.resolve()
            finally:
                beside.unlink()
                if not any(beside.parent.iterdir()):
                    beside.parent.rmdir()
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
                t, i, pw = d[k]          # power too, for the EQE-settling question
                assert t.size >= 2 and _np.isfinite(i).all() and _np.isfinite(pw).all()
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
        agg = next((first.run_dir / "NVLEDSweep" / "Sweep").glob("*aggregated_*.csv"))
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


def _repo_calibrations() -> dict[str, float]:
    """Hash every calibration in the install tree, so the suite can prove it kept them."""
    import hashlib
    from measure_iqe.calibrate import CALIB_DIR
    d = Path(__file__).resolve().parent / CALIB_DIR
    if not d.is_dir():
        return {}
    return {f.name: hashlib.sha256(f.read_bytes()).hexdigest()
            for f in sorted(d.glob("*.yaml"))}


def test_pooling_is_off_by_default_and_adds_no_columns():
    """An ordinary run's files must be byte-for-byte what they were before pooling.

    A flag nobody set must not change the run format, or every existing reader has to
    learn a column to read a run that has no pooled rows in it.
    """
    assert IQEConfig().pool_groups == (), "pooling must default to off"
    with tempfile.TemporaryDirectory() as tmp:
        res = run_iqe_measurement(IQEConfig(**FAST), simulate=True, base_dir=tmp,
                                  log=QUIET)
        rows = _agg_rows(res.run_dir)
    assert rows, "no sweep rows"
    for col in ("POOL_K", "POOL_MEMBERS"):
        assert col not in rows[0], f"{col} written on a run with pool_groups=()"


def test_a_pooled_target_lights_every_member_and_references_them_all():
    """K members lit together, and the reference is the SAME K at bitval 0.

    Section 6: an all-off frame conducts ~4x the bitval-0 level, so a blanked frame is
    not a reference. For a pooled target the only valid reference is the same members
    at zero drive -- same frame structure, same select state, no drive.
    """
    seen = []

    class Spy:
        def __init__(self, inner):
            self._inner = inner

        def set_pixel(self, x, y, bit):
            seen.append(("one", ((int(x), int(y)),), int(bit)))
            return self._inner.set_pixel(x, y, bit)

        def set_pixels(self, coords, bit):
            coords = tuple((int(x), int(y)) for x, y in coords)
            seen.append(("many", coords, int(bit)))
            return self._inner.set_pixels(coords, bit)

        def blank_frame(self):
            return self._inner.blank_frame()

    real = acquire.connect_sim
    acquire.connect_sim = lambda cfg, log: (lambda t: (t[0], t[1], Spy(t[2])))(real(cfg, log))
    try:
        with tempfile.TemporaryDirectory() as tmp:
            cfg = IQEConfig(**{**FAST, "roi": (0, 0, 15, 15), "block_size": 8,
                               "pool_groups": (4,)})
            res = run_iqe_measurement(cfg, simulate=True, base_dir=tmp, log=QUIET)
            # read both here: tmp is gone once the with-block closes
            allrows = _agg_rows(res.run_dir)
            rows = [r for r in allrows if r.get("POOL_K") not in (None, "", "nan")]
    finally:
        acquire.connect_sim = real

    assert rows, "pooling was requested but no pooled row was written"
    # pool_groups is PER BLOCK, so a 16x16 ROI in 8 px blocks gives one group per block.
    groups = {tuple(tuple(int(v) for v in m.split(":")) for m in r["POOL_MEMBERS"].split())
              for r in rows}
    assert len(groups) == 4, f"2x2 blocks at pool_groups=(4,) is 4 groups, got {len(groups)}"
    for r in rows:
        g = tuple(tuple(int(v) for v in m.split(":")) for m in r["POOL_MEMBERS"].split())
        assert len(g) == 4 and int(float(r["POOL_K"])) == 4
        assert (int(r["X"]), int(r["Y"])) == g[0], (
            "a pooled row's X,Y must be its first member, or it has no unique key")
    lit = [c for c in seen if c[0] == "many" and c[2] != 0]
    off = [c for c in seen if c[0] == "many" and c[2] == 0]
    assert lit, "the pooled target was never lit through set_pixels"
    assert off, "the pooled target was never referenced at bitval 0"
    for _t, coords, _b in lit + off:
        assert coords in groups, f"frame lit {coords}, which is no group of {groups}"
    assert {c[1] for c in lit} == groups, "a group was never lit"
    assert {c[1] for c in off} == groups, "a group was never referenced at bitval 0"
    # The fast scan legitimately lights every pixel on its own. What must not happen is
    # a member driven alone AFTER pooling starts -- that would be a different
    # measurement reported under the pooled row's key. With pool_include_singles the
    # SHARED member is a single pixel in its own right, so only the others are barred.
    shared = {(int(r["X"]), int(r["Y"]))
              for r in allrows if r.get("POOL_K") in (None, "", "nan")}
    assert shared, "no single pixels in the run, so the check below proves nothing"
    fresh = {m for g in groups for m in g} - shared
    first_pooled = next(i for i, c in enumerate(seen) if c[0] == "many")
    alone = {c[1][0] for c in seen[first_pooled:] if c[0] == "one" and c[2] != 0}
    assert not (alone & fresh), (
        f"pooled members {alone & fresh} were lit individually during the sweep")


def test_a_pooled_target_does_not_make_single_pixels_look_dead():
    """The dead test compares against the median of the other targets.

    A pooled target emits K x as much, so an unscaled median would rise by roughly K and
    condemn live single pixels. Both walks judge per MEMBER, so the two are comparable.
    """
    bright = dict(PM400_POWER_MEAN=4e-8, POOL_K=4)       # 10 nW per member
    single = dict(PM400_POWER_MEAN=1e-8)                 # 10 nW, the same pixel
    assert acquire.per_member_power(bright) == acquire.per_member_power(single), (
        "a 4-member target at 4x the power must read the same per member")
    assert acquire.per_member_power(dict(PM400_POWER_MEAN=-2e-9)) == 2e-9, (
        "the current is signed; the test is on magnitude")
    # and a run with both kinds must not kill the singles
    with tempfile.TemporaryDirectory() as tmp:
        cfg = IQEConfig(**{**FAST, "roi": (0, 0, 15, 15), "block_size": 8,
                           "pool_groups": (4,), "dead_abort": True})
        res = run_iqe_measurement(cfg, simulate=True, base_dir=tmp, log=QUIET)
        rows = _agg_rows(res.run_dir)
    per_target = {}
    for r in rows:
        per_target.setdefault((r["X"], r["Y"]), []).append(r)
    singles = [k for k, rs in per_target.items()
               if rs[0].get("POOL_K") in (None, "", "nan")]
    assert len(singles) >= 2, f"expected several single pixels, got {len(singles)}"
    full = max(len(rs) for rs in per_target.values())
    short = [k for k in singles if len(per_target[k]) < full]
    assert not short, (
        f"single pixel(s) {short} were dead-aborted in a run with a pooled target; "
        f"the pooled target's power leaked into the live median")


def test_the_pooled_members_column_survives_the_csv():
    """POOL_MEMBERS must carry no comma.

    The aggregated writer joins fields with a comma and does not quote, so "x,y" inside
    a value shifts every later column of that row. The first version did exactly that
    and pandas refused the file.
    """
    with tempfile.TemporaryDirectory() as tmp:
        cfg = IQEConfig(**{**FAST, "roi": (0, 0, 15, 15), "block_size": 8,
                           "pool_groups": (4,)})
        res = run_iqe_measurement(cfg, simulate=True, base_dir=tmp, log=QUIET)
        rows = _agg_rows(res.run_dir)
        n_cols = len(acquire.agg_cols(cfg))
        for agg in sorted((Path(res.run_dir) / "NVLEDSweep" / "Sweep").glob(
                "*aggregated_bitval=*.csv")):
            for i, line in enumerate(agg.read_text().splitlines()):
                assert len(line.split(",")) == n_cols, (
                    f"{agg.name} line {i} has {len(line.split(','))} fields, "
                    f"expected {n_cols}: a value contains a comma")
    pooled = [r for r in rows if r.get("POOL_K") not in (None, "", "nan")]
    assert pooled, "nothing pooled"
    assert "," not in pooled[0]["POOL_MEMBERS"] and "'" not in pooled[0]["POOL_MEMBERS"], (
        f"POOL_MEMBERS is not clean: {pooled[0]['POOL_MEMBERS']!r}")


def test_pooling_is_refused_where_it_cannot_mean_anything():
    for kwargs, why in [
            (dict(pool_groups=(1, 4)), "a group of one"),
            (dict(pool_groups=(4,), source_mode="current"), "a current source"),
            (dict(pool_groups=(4,), sweep_select="stride"), "no blocks to form it in"),
            (dict(pool_groups=(4,), pixels_per_block=0), "no single to share"),
            (dict(pool_pm_range=-1e-7), "a negative range")]:
        try:
            IQEConfig(**kwargs).validate()
        except ValueError:
            continue
        raise AssertionError(f"validate() accepted {why}: {kwargs}")


def test_every_gui_default_round_trips_through_the_cli():
    """What the console types must parse back to the value it showed.

    The console renders every field as text and sends the changed ones as --set, which
    the CLI parses as JSON. A field whose rendering does not round-trip would silently
    change a run the moment the operator touched any OTHER field -- because then it
    would differ from the default and be sent too.
    """
    from measure_iqe.cli import parse_override
    from measure_iqe.gui import _as_text

    base = IQEConfig()
    bad = []
    for f in dataclasses.fields(IQEConfig):
        txt = _as_text(getattr(base, f.name))
        try:
            k, v = parse_override(f"{f.name}={txt}")
            got = getattr(IQEConfig.from_dict({**base.to_dict(), k: v}), f.name)
        except Exception as exc:                      # noqa: BLE001 - reported below
            bad.append(f"{f.name}={txt!r} -> {type(exc).__name__}: {exc}")
            continue
        if got != getattr(base, f.name):
            bad.append(f"{f.name}={txt!r} -> {got!r}, expected {getattr(base, f.name)!r}")
    assert not bad, "fields the console cannot round-trip:\n  " + "\n  ".join(bad)


def test_the_console_only_sends_what_the_operator_changed():
    """An unchanged field must NOT become a --set.

    --set outranks both --config and configs/local.yaml, so a console that sent all
    132 fields every time would silently overwrite the machine's own settings with
    its idea of the defaults -- the exact failure ROADMAP item 3 was written to avoid.
    """
    from measure_iqe.gui import _defaults, build_argv

    d = _defaults()
    assert build_argv(dict(d), d) == [sys.executable, "-m", "measure_iqe"], (
        "an untouched console must produce a bare command")
    v = dict(d)
    v["pool_groups"] = "[2,4,8]"
    v["roi"] = "FULL"
    argv = build_argv(v, d, simulate=True, array="Array42_IQE",
                      extra="# a comment\nn_fine=12\n\n")
    assert argv.count("--set") == 3, argv
    assert "--simulate" in argv and "--array" in argv
    for want in ("roi=FULL", "pool_groups=[2,4,8]", "n_fine=12"):
        assert want in argv, f"{want} missing from {argv}"


def test_the_fast_scan_screen_tiles_the_roi_without_gaps_or_crossings():
    """Every pixel is in exactly one tile or in the leftover list, and no tile crosses 256.

    A gap means a pixel is never tested at all -- the one thing a screen may not do
    quietly. A crossing means the frame lights the near quadrant's mirror image,
    because the pixel-enable bits are per quadrant and live in the frame instruction.
    """
    for roi, k in (((0, 0, 511, 511), 2), ((0, 0, 511, 511), 4), ((0, 0, 511, 511), 8),
                   ((0, 0, 511, 511), 16), ((0, 0, 15, 15), 4), ((32, 32, 63, 63), 4),
                   ((0, 0, 510, 510), 4), ((250, 250, 265, 265), 4)):
        tiles, leftover = acquire.pool_tiles(*roi, k)
        members = [m for _q, _iy, _ix, ms in tiles for m in ms]
        allpx = set(acquire.roi_pixels(*roi))
        assert len(members) == len(set(members)), f"{roi} k={k}: a pixel is in two tiles"
        assert set(members) | set(leftover) == allpx, (
            f"{roi} k={k}: {len(allpx - set(members) - set(leftover))} pixel(s) never tested")
        assert not (set(members) & set(leftover))
        for _q, _iy, _ix, ms in tiles:
            assert len(ms) == k, f"{roi} k={k}: tile of {len(ms)}"
            assert len({(x >= 256, y >= 256) for x, y in ms}) == 1, f"{ms} crosses 256"
    # the whole array divides exactly at every K the screen is sized for
    for k, want in ((2, 131072), (4, 65536), (8, 32768), (16, 16384)):
        tiles, leftover = acquire.pool_tiles(0, 0, 511, 511, k)
        assert (len(tiles), len(leftover)) == (want, 0), (k, len(tiles), len(leftover))

    # local_median is the detrending the recall depends on
    a = np.arange(25.0).reshape(5, 5)
    assert acquire.local_median(a, 3).shape == a.shape
    assert acquire.local_median(a, 3)[2, 2] == 12.0
    assert np.allclose(acquire.local_median(np.ones((7, 7)), 5), 1.0)


def test_the_screen_rescans_what_it_flags_and_says_what_it_skipped():
    """The screen must narrow the per-pixel pass, cover the classification, and shout.

    Three things it may not do: leave a flagged group unmeasured, judge alive/dead on
    the flagged (dim) rows alone -- which would put the percentile threshold in the
    wrong place -- or let the rescan cap bind in silence.
    """
    msgs: list[str] = []
    with tempfile.TemporaryDirectory() as tmp:
        cfg = IQEConfig(**{**FAST, "roi": (0, 0, 31, 31), "block_size": 8,
                           "pixels_per_block": 1, "fastscan_pool": 4})
        res = run_iqe_measurement(cfg, simulate=True, base_dir=tmp, log=msgs.append)
        fs = sorted((Path(res.run_dir) / "FastScan").glob("*screen_bitval=*.csv"))
        assert fs, "the screen wrote no CSV"
        import csv as _csv
        with fs[0].open() as f:
            rows = list(_csv.DictReader(f))
        single = sorted((Path(res.run_dir) / "FastScan").glob(
            "*aggregated_bitval=*.csv"))[0]
        with single.open() as f:
            scanned = {(int(r["X"]), int(r["Y"])) for r in _csv.DictReader(f)}
    assert len(rows) == 256, f"32x32 in 4-pixel tiles is 256 groups, got {len(rows)}"
    assert all(int(r["POOL_K"]) == 4 for r in rows)
    flagged = [r for r in rows if int(r["SCREEN_FLAGGED"])]
    assert flagged, "nothing was flagged, so the rescan path never ran"
    assert len(flagged) < len(rows), "everything was flagged, so the screen saved nothing"
    # every member of a flagged group was rescanned, and nothing else was
    want = {tuple(int(c) for c in m.split(":"))
            for r in flagged for m in r["POOL_MEMBERS"].split()}
    assert want == scanned, (
        f"{len(want - scanned)} flagged member(s) never rescanned, "
        f"{len(scanned - want)} pixel(s) rescanned without being flagged")
    text = " ".join(msgs)
    assert "are NOT measured per pixel" in text, (
        "the screen must say that the unflagged groups were never measured")
    # the simulator kills ~1 pixel in 13, far deader than the array this was sized for,
    # so the cap MUST bind here and MUST say so
    assert "RESCAN CAP IS BINDING" in text, (
        "the cap bound without the run saying so -- a screen that silently ignores "
        "groups it knows are suspicious is the one failure it may not have")
    # and the classification saw the screened estimates, not just the dim rows
    assert "screened estimate(s)" in text


def test_a_set_value_survives_a_shell_that_does_not_strip_quotes():
    """PowerShell and bash strip `roi='[0,0,63,63]'`. cmd.exe does not.

    The value then arrives as "'[0,0,63,63]'", which is not JSON, is kept as a
    plain string, and is rejected with "unknown ROI" -- naming the ROI rather than
    the quoting. Same command, same repo, different shell.
    """
    from measure_iqe.cli import parse_override

    for quoted, bare in ((chr(39), ''), (chr(34), '')):
        for raw, want in ((f'roi={quoted}[0,0,63,63]{quoted}', [0, 0, 63, 63]),
                          (f'pool_groups={quoted}[4,8]{quoted}', [4, 8]),
                          (f'n_fine={quoted}8{quoted}', 8),
                          (f'roi={quoted}FULL{quoted}', 'FULL')):
            k, v = parse_override(raw)
            assert v == want, f'{raw} gave {v!r}, wanted {want!r}'
            # and the unquoted form must give exactly the same thing
            k2, v2 = parse_override(raw.replace(quoted, bare))
            assert (k, v) == (k2, v2), f'{raw} and its bare form disagree'

    # a value that is genuinely a string keeps its characters
    assert parse_override('sample_name=Array42')[1] == 'Array42'
    assert parse_override('sweep_direction=low_to_high')[1] == 'low_to_high'
    # one quote is not a pair and must not be eaten
    assert parse_override("sample_name=it's")[1] == "it's"

def test_the_ramp_never_goes_past_the_stop_it_was_asked_for():
    """A knee outside the ramp cannot split it, and must not extend it.

    Run 261019-1537 asked for -0.8..-1.1 V and measured -0.8..-1.3 V. The split is
    start -> vramp_knee_v -> stop, so with the default knee at -1.3 V the rising leg
    overshot the stop and the droop leg came back up: 8 points over 6 distinct biases
    with -1.1 and -1.2 measured twice. A pooled sweep deliberately stopped short of
    the compliance and range ceilings went straight past both.
    """
    for stop, n in ((-1.1, 8), (-1.2, 8), (-1.4, 8), (-2.6, 30), (-1.3, 8)):
        cfg = IQEConfig(nvled_v_coarse_stop=stop, n_fine=n).validate()
        msgs: list[str] = []
        V = [v for v, _ in acquire.build_fine_voltages(None, None, cfg, msgs.append)]
        lo = min(cfg.nvled_v_coarse_start, stop)
        assert len(V) == n, f'stop={stop}: {len(V)} points, asked for {n}'
        assert min(V) >= lo - 1e-9, f'stop={stop}: swept to {min(V)}, past {lo}'
        assert max(V) <= max(cfg.nvled_v_coarse_start, stop) + 1e-9, V
        assert len(set(V)) == len(V), f'stop={stop}: duplicate biases in {sorted(V)}'
        # and the log says which shape it used, because that changes where the
        # points are spent
        inside = lo < cfg.vramp_knee_v < max(cfg.nvled_v_coarse_start, stop)
        assert ('split:' in msgs[0]) == inside, msgs[0]
        if not inside:
            assert 'OUTSIDE this ramp' in msgs[0], msgs[0]


def test_the_bright_end_estimate_follows_the_sweeps_actual_stop():
    """It scaled a fast-scan reading by a FIXED -2.60 V ratio whatever the stop.

    Run 261019-1537 stopped at -1.1 V and was warned its brightest group would emit
    757 nW against a 470 nW range. The groups measured 188 nW at the -1.3 V the ramp
    bug actually reached, and would have been 72 nW at the -1.1 V requested. The
    message even printed the real stop while using the -2.60 V number.
    """
    # the curve must be usable by np.interp, which needs ascending x. Reversing it
    # was a bug that silently made every scale factor 1.000.
    xs = [c[0] for c in acquire.BIAS_CURVE]
    assert xs == sorted(xs), 'BIAS_CURVE must ascend in V for np.interp'

    # a round trip is exactly 1, and the fast-scan bias is a measured point
    for v in (-2.6, -1.35, -1.1, -0.8):
        p, i = acquire.bias_scale(v, v)
        assert abs(p - 1) < 1e-9 and abs(i - 1) < 1e-9, (v, p, i)

    # monotonic: deeper bias means more light and more current
    prev = None
    for v in (-0.8, -0.9, -1.1, -1.3, -1.65, -2.6):
        p, i = acquire.bias_scale(v, -1.35)
        if prev is not None:
            assert p > prev[0] and i > prev[1], f'not monotonic at {v}'
        prev = (p, i)

    # against the run: members reading 51.6 nW each at -1.35 V, K=4, stop -1.3 V,
    # measured 188.2 nW. 15 % is plenty for a warning threshold.
    p, _ = acquire.bias_scale(-1.3, -1.35)
    pred = 4 * 51.6e-9 * p
    assert abs(pred - 188.2e-9) / 188.2e-9 < 0.15, f'{pred*1e9:.0f} nW vs 188.2 nW'
    # and the old fixed ratio must no longer be reachable at a shallow stop
    p11, _ = acquire.bias_scale(-1.1, -1.35)
    assert 4 * 51.6e-9 * p11 < 150e-9, 'a -1.1 V stop must not predict a bright end'

def test_no_block_ever_straddles_the_quadrant_boundary():
    """The pixel-enable bits are per quadrant and live in the FRAME instruction.

    A block spanning x=256 therefore mixes two quadrants' row drivers into one block
    median, and a pooled group drawn from it would light the near quadrant's mirror
    image instead of the far member. linspace over the whole ROI did straddle it: at
    block_size=102 the edges fell on 0, 102.4, 204.8, 307.2, 409.6, 512, so 9 of 25
    blocks crossed.
    """
    for lo, hi, bs in ((0, 511, 52), (0, 511, 102), (0, 511, 8), (0, 15, 8),
                       (32, 63, 8), (0, 255, 102), (256, 511, 102), (200, 300, 20)):
        e = acquire.block_edges(lo, hi, bs)
        assert e[0] == lo and e[-1] == hi + 1, f"{lo}..{hi} bs={bs}: {e} misses the ROI"
        assert all(a < b for a, b in zip(e, e[1:])), f"{e} is not increasing"
        assert not any(a < 256 < b for a, b in zip(e, e[1:])), f"{e} crosses 256"
    # and every selected pixel's block really is inside one quadrant
    alive = [{"X": x, "Y": y} for x in range(0, 512, 2) for y in range(0, 512, 2)]
    cfg = IQEConfig(roi=(0, 0, 511, 511), sweep_select="blocks", block_size=102,
                    pixels_per_block=1).validate()
    xe = acquire.block_edges(0, 511, 102)
    assert len(xe) - 1 == 6, f"102 px blocks over 512 should give 6 per axis, got {len(xe)-1}"
    assert len(acquire.select_sweep_pixels(alive, cfg, QUIET)) == 36


def test_the_console_presets_describe_runs_that_actually_happen():
    """Each preset must name the number of targets it really produces.

    A preset labelled "36 singles + 50 pooled" that yields 16 and 25 is worse than no
    preset: the operator reads the label, not the selection log.
    """
    from measure_iqe.gui import PRESETS

    names = {f.name for f in dataclasses.fields(IQEConfig)}
    for label, over in PRESETS.items():
        assert set(over) <= names, f"{label}: not config fields: {set(over) - names}"
        IQEConfig(**over).validate()

    alive = [{"X": x, "Y": y} for x in range(512) for y in range(512)]
    sel = acquire.select_sweep_pixels(
        alive, IQEConfig(**PRESETS["Full array, 200 singles + 100 pooled x4"]).validate(),
        QUIET)
    singles = [r for r in sel if "POOL_MEMBERS" not in r]
    groups = [r for r in sel if "POOL_MEMBERS" in r]
    assert len(singles) == 200, f"preset says 200 singles, got {len(singles)}"
    assert len(groups) == 100, f"preset says 100 pooled groups, got {len(groups)}"
    assert all(len(g["POOL_MEMBERS"]) == 4 for g in groups), "preset says groups of 4"

    sel = acquire.select_sweep_pixels(
        alive,
        IQEConfig(**PRESETS["Full array, 36 singles + 36 each of K=2,4,8"]).validate(),
        QUIET)
    singles = [r for r in sel if "POOL_MEMBERS" not in r]
    groups = [r for r in sel if "POOL_MEMBERS" in r]
    assert len(singles) == 36, f"preset says 36 singles, got {len(singles)}"
    by_k = {k: sum(1 for g in groups if len(g["POOL_MEMBERS"]) == k) for k in (2, 4, 8)}
    assert by_k == {2: 36, 4: 36, 8: 36}, f"preset says 36 of each K, got {by_k}"

    cfg100 = IQEConfig(**PRESETS["Full array, 100 singles"]).validate()
    assert len(acquire.select_sweep_pixels(alive, cfg100, QUIET)) == 100
    cfg1k = IQEConfig(**PRESETS["Full array, 1000 singles"]).validate()
    assert len(acquire.select_sweep_pixels(alive, cfg1k, QUIET)) == 1000


def test_a_pooled_group_that_will_clamp_says_so_before_the_sweep():
    """K members draw K x the current, and pick_limit's ladder has a ceiling.

    Measured 261015-0015: 117.7 uA per pixel at -2.60 V against a 1 mA
    vramp_highv_limit_a, so K=8 is the edge and K=16 clamps. A clamped point is the
    instrument's current, not the device's, and the EQE from it is meaningless -- so
    this has to be said at selection and not 30 hours later in NVLED_AT_COMPLIANCE.
    """
    # every pixel at the measured fast-scan current, so the estimate is the real one
    alive = [{"X": x, "Y": y, "NVLED_CURR_MEAN": -33.3e-6}
             for x in range(64) for y in range(64)]
    seen = {}
    for k in (4, 8, 16):
        cfg = IQEConfig(roi=(0, 0, 63, 63), block_size=16, pixels_per_block=1,
                        pool_groups=(k,)).validate()
        msgs: list[str] = []
        sel = acquire.select_sweep_pixels(alive, cfg, msgs.append)
        assert [r for r in sel if "POOL_MEMBERS" in r], (
            f"K={k}: nothing selected, the test measures nothing")
        seen[k] = " ".join(msgs)
    assert "CLAMPING EXPECTED" not in seen[4], (
        "K=4 draws ~471 uA against a 1 mA ceiling and must not warn")
    assert "CLAMPING EXPECTED" not in seen[8] and "close, watch" in seen[8], (
        "K=8 draws ~942 uA: close, but not over")
    assert "CLAMPING EXPECTED" in seen[16], (
        "K=16 draws ~1.9 mA against a 1 mA ceiling and must warn")
    assert "entry <= 8" in seen[16], (
        f"the warning must name a usable K, got: {seen[16]}")


def test_the_warmup_diagnostic_never_runs_on_a_pooled_target():
    """It watches ONE pixel turn on and relax.

    A pooled target reaching it would be driven as its first member alone, which is a
    different measurement reported under the same coordinates.
    """
    with tempfile.TemporaryDirectory() as tmp:
        cfg = IQEConfig(**{**FAST, "roi": (0, 0, 15, 15), "block_size": 8,
                           "pool_groups": (4,),
                           "warmup_diag_n_pixels": 8, "warmup_diag_seconds": 0.01})
        res = run_iqe_measurement(cfg, simulate=True, base_dir=tmp, log=QUIET,
                                  warmup_diag=True)
        csvs = list(Path(res.run_dir).rglob("*armup*.csv"))
        assert csvs, "the warm-up diagnostic wrote nothing"
        import csv as _csv
        with csvs[0].open() as f:
            rows = list(_csv.DictReader(f))
    assert rows, "no warm-up rows"
    # asked for 8 pixels from a selection of 4 singles + 1 pooled target: it must stop
    # at the 4 singles rather than reach into the pooled one
    seen = {(r["pixel_x"], r["pixel_y"]) for r in rows}
    assert len(seen) <= 4, f"the diagnostic touched {len(seen)} targets, expected <= 4"


def test_aggregated_fields_keep_full_float_precision():
    """`agg_field` must not round-trip a float through `str`.

    It writes strings bare so POOL_MEMBERS is usable, and that change must not quietly
    turn every number into a shorter one.
    """
    v = 1.2345678901234567e-09
    assert acquire.agg_field(v) == repr(v) and float(acquire.agg_field(v)) == v
    assert acquire.agg_field(True) == "True"
    assert acquire.agg_field(float("nan")) == "nan"
    assert acquire.agg_field("7:5 5:13") == "7:5 5:13", "a string must not be quoted"


if __name__ == "__main__":
    # write_calibration() writes to every calibration root, the install tree included.
    # That is wanted for a real --calibrate and poison for a test: the suite once
    # replaced the bench's measured Array42_IQE calibration with simulator output and
    # said "all checks passed". The run is not green unless every calibration that was
    # on disk when it started is still there, byte for byte.
    #
    # The guard deliberately knows nothing about TEST_ARRAY. Excluding that name from
    # the snapshot would switch the guard off for whatever the tests happen to be
    # called, which is the mistake being guarded against; deleting the file before the
    # snapshot would destroy a real calibration of that name outright. Both were tried.
    # Instead the suite's own file is removed in the `finally`, on success and on
    # failure alike, so it never survives to appear in the next run's snapshot.
    _cal_dir = Path(__file__).resolve().parent / "calibrations"
    before = _repo_calibrations()
    if f"{TEST_ARRAY}.yaml" in before:
        # Either TEST_ARRAY names a real array -- the mistake this guards against -- or
        # an earlier run died before its cleanup. Cannot tell the two apart from here,
        # and guessing wrong either destroys a measured calibration or hides the bug.
        # So: remove it, say which it might be, and fail. The next run starts clean.
        (_cal_dir / f"{TEST_ARRAY}.yaml").unlink(missing_ok=True)
        raise SystemExit(
            f"calibrations/{TEST_ARRAY}.yaml was already on disk. Either TEST_ARRAY "
            f"names an array a bench has measured (change it -- the suite overwrites "
            f"whatever it names), or an earlier failed run left it behind. It has been "
            f"removed; re-run. If it was a real calibration, restore it from its run "
            f"directory's <array>_calibration.yaml.")
    try:
        for name, fn in sorted(globals().items()):
            if name.startswith("test_") and callable(fn):
                fn()
                print(f"ok  {name}")
    finally:
        after = _repo_calibrations()
        for stray in sorted(set(after) - set(before)):
            (_cal_dir / stray).unlink(missing_ok=True)
            print(f"ok  removed the test calibration the suite wrote: {stray}")
        clobbered = [k for k, v in before.items() if after.get(k) != v]
        assert not clobbered, (
            f"the suite modified or deleted real calibration(s): {clobbered}. A test "
            f"must set sample_name=TEST_ARRAY, and TEST_ARRAY must name no array any "
            f"bench has measured.")
    print("all checks passed")
