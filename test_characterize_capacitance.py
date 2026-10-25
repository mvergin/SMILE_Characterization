"""Self-checks for the switch-deviation metric and the K-pixel cluster primitive.

The metric exists to answer one question at the bench: does a frame switch cost
current that the dwell then has to pay back? Run 261016-0726 is the case the old
version failed on, so it is the case this file pins down. Section 21.

    python test_characterize_capacitance.py
"""
import importlib.util
import pathlib

import numpy as np

from characterize_capacitance import exp_cluster, switch_deviation
from instruments.smile_fpga import SmileFPGA

# smile/__init__.py pulls in termcolor, which an analysis-only machine need not have.
# The bit packing is what is under test, so load that one module off disk rather than
# making this file depend on the whole display package.
_spec = importlib.util.spec_from_file_location(
    "_display", pathlib.Path(__file__).parent / "smile" / "utils" / "display.py")
display = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(display)

FS = 1.0 / 0.0407          # the real sample rate at NPLC 1.0 on two channels
N = 3000                   # --samples 3000, as every switch run uses


def _t(n=N):
    return np.arange(n) / FS


def test_flat_trace_reports_nothing():
    t = _t()
    rng = np.random.default_rng(0)
    y = 455e-9 * (1 + 2e-4 * rng.standard_normal(t.size))
    first, later, res = switch_deviation(t, y)
    assert abs(first / later - 1) < 3 * res / later, (
        f"a flat trace must sit inside its own resolution: "
        f"{1e2 * (first / later - 1):+.3f}% vs {1e2 * res / later:.3f}%")


def test_sign_change_is_not_averaged_away():
    """The 261016-0726 -0.80 V shape: +0.49 % for 0.5 s, then -0.56 % for seconds.

    The mean of second one is ~0, which is exactly why the metric must not take it.
    """
    t = _t()
    rng = np.random.default_rng(1)
    y = 455e-9 * (1 + 2e-4 * rng.standard_normal(t.size))
    y[t < 0.5] *= 1.0049
    y[(t >= 0.5) & (t < 60)] *= 0.9944
    first, later, res = switch_deviation(t, y)
    dev = 100 * (first / later - 1)
    assert dev < -0.4, f"must report the -0.56% plateau, reported {dev:+.3f}%"
    assert abs(dev / (100 * res / later)) > 5, "and it must clear its own resolution"
    # the failure the old version had: the first second alone hides it
    w = int(round(FS))
    assert abs(100 * (y[:w].mean() / later - 1)) < 0.1, (
        "the fixture must reproduce the cancellation, or it tests nothing")


def test_reference_is_the_tail_not_the_whole_record():
    """A 60 s decay leaves most of itself inside any "mean of everything after 2 s"."""
    t = _t()
    y = 7958e-9 * (1 - 0.0045 * np.exp(-t / 25.0))
    first, later, res = switch_deviation(t, y)
    assert abs(later / 7958e-9 - 1) < 5e-4, (
        f"the tail must read the asymptote, read {1e2 * (later / 7958e-9 - 1):+.3f}%")
    naive = float(np.mean([y[k:k + int(FS)].mean()
                           for k in range(2 * int(FS), y.size - int(FS), int(FS))]))
    assert abs(first / later - 1) > abs(first / naive - 1), (
        "anchoring on the tail must report MORE deficit than anchoring on the "
        "whole record, or the change was pointless")


def test_short_record_gets_no_verdict():
    for n in (8, 100, 12 * int(round(FS))):
        assert all(np.isnan(v) for v in switch_deviation(_t(n), np.ones(n))), (
            f"n={n} cannot hold five early seconds plus a disjoint tail")


def test_overflow_samples_are_dropped():
    t = _t()
    y = np.full(t.size, 455e-9)
    y[5:9] = 9.91e37
    assert not any(np.isnan(v) for v in switch_deviation(t, y)), \
        "the over-range marker must be filtered, not propagated"


def test_real_run_reproduces_section_21():
    """The two numbers section 21 quotes, straight off the trace file."""
    import glob
    f = glob.glob(r"demo/q4b_switch/*/traces.npz")
    if not f:
        print("  skip test_real_run_reproduces_section_21 (demo/q4b_switch absent)")
        return
    z = np.load(f[0])
    want = {"-0.800": (-0.70, -0.50), "-1.050": (-0.75, -0.55)}  # -0.604%, -0.666%
    for k in z.files:
        a = z[k]
        first, later, res = switch_deviation(a[0], a[2])
        dev = 100 * (first / later - 1)
        lo, hi = want[k.split("_v")[1][:6]]
        assert lo < dev < hi, f"{k}: {dev:+.3f}% outside [{lo}, {hi}]"
        assert abs(dev) > 5 * 100 * res / later, f"{k}: {dev:+.3f}% inside the noise"


def test_power_trace_starts_at_the_trigger_and_survives_a_dead_meter():
    """PowerTrace runs while the SMU burst blocks the main thread. Two promises:
    t = 0 is the start() call, and a meter that throws kills the thread, not the run.
    """
    import time

    from characterize_capacitance import PowerTrace

    class FakePM:
        def __init__(self, fail_after=None):
            self.n, self.fail_after = 0, fail_after

        def measure(self):
            self.n += 1
            if self.fail_after is not None and self.n > self.fail_after:
                raise RuntimeError("meter went away")
            return 5e-8

    pt = PowerTrace(FakePM(), n=1)
    pt.start()
    time.sleep(0.25)                       # stands in for the blocking burst
    t, pw = pt.stop()
    assert t.size > 10, f"only {t.size} optical samples in 0.25 s"
    assert t.size == pw.size, "t and P must come back the same length"
    # Windows rounds a short sleep to the ~15.6 ms timer tick, so the bound is loose:
    # it only has to prove t=0 is start() and not the top of the record.
    assert t[0] < 0.05, f"first sample at t={t[0]:.4f}s: start() must be t=0"
    assert np.allclose(pw, 5e-8), "the readings must come through untouched"
    assert np.all(np.diff(t) > 0), "the host clock must be monotone"

    # a meter that dies mid-record returns what it had, and raises nothing
    pt = PowerTrace(FakePM(fail_after=5), n=1)
    pt.start()
    time.sleep(0.15)
    t, pw = pt.stop()
    assert t.size == 5, f"kept {t.size} samples, expected the 5 before the failure"

    # never started: empty, not an exception
    assert PowerTrace(FakePM()).stop()[0].size == 0


class _RecordingDev:
    """Captures what would go down the USB link."""

    commands = {"RESET_DIS": 0x1, "USB_2_DIS": 0x2}

    def __init__(self):
        self.sent = []

    def send_instruction(self, instr, data=None):
        self.sent.append((instr, list(data) if data is not None else None))


def _payload(fn, *a, **k):
    dev = _RecordingDev()
    fn(dev, *a, **k)
    frames = [d for _i, d in dev.sent if d is not None]
    assert len(frames) == 1, f"expected exactly one data frame, got {len(frames)}"
    return frames[0]


def test_one_pixel_through_set_pixels_matches_set_pixel():
    """The cluster path must not invent a second addressing convention.

    If these ever diverge, a cluster lights different silicon than the per-pixel
    sweep it is being compared against, and the comparison is meaningless.
    """
    for row, col in [(0, 0), (0, 255), (5, 31), (5, 32), (200, 177), (255, 255)]:
        one = _payload(display.set_pixel, row, col, 11)
        many = _payload(display.set_pixels, [(row, col)], 11)
        assert one == many, f"({row},{col}) packs differently"


def test_k_pixels_are_the_or_of_their_single_pixel_frames():
    """K pixels in one frame, and only those K."""
    coords = [(0, 0), (0, 200), (3, 31), (3, 32), (7, 255)]
    many = _payload(display.set_pixels, coords, 15)
    want = [0] * (8 * 8)                      # rows 0..7, 8 words each
    for row, col in coords:
        for i, w in enumerate(display._pixel_row_constructor(col, 15)):
            want[row * 8 + i] |= w
    assert many == want, "the K-pixel frame is not the OR of the single-pixel frames"
    assert len(many) == 8 * (max(r for r, _ in coords) + 1), (
        "only rows 0..max(row) should be sent -- roi_lrow comes from the payload length")
    # and nothing leaks into a row that holds no member
    assert many[8:24] == [0] * 16, "rows 1 and 2 hold no member and must be zero"


def test_set_pixels_rejects_what_it_cannot_write():
    for bad, why in [([], "empty"), ([(0, 256)], "column out of range"),
                     ([(256, 0)], "row out of range")]:
        try:
            _payload(display.set_pixels, bad, 4)
        except ValueError:
            continue
        raise AssertionError(f"set_pixels accepted {why}")
    try:
        _payload(display.set_pixels, [(0, 0)], 16)
    except ValueError:
        pass
    else:
        raise AssertionError("set_pixels accepted a bitval above 15")


def test_a_cluster_may_not_straddle_a_quadrant_boundary():
    """The pixel-enable bits live in the frame instruction, not in the pixel data.

    Without this guard the far member is silently written into the near quadrant's
    mirror image, so the frame lights a pixel nobody asked for and the pooled
    measurement is wrong in a way no column records.
    """
    fpga = SmileFPGA(None)                 # no device: the guards still run
    fpga.set_pixels([(10, 10), (20, 20)], 15)          # both TL, fine
    for bad in ([(10, 10), (300, 10)], [(10, 10), (10, 300)]):
        try:
            fpga.set_pixels(bad, 15)
        except ValueError as exc:
            assert "quadrant" in str(exc), exc
        else:
            raise AssertionError("straddling cluster accepted")
    for bad in ([(10, 10), (512, 10)], [(-1, 0)]):
        try:
            fpga.set_pixels(bad, 15)
        except ValueError:
            continue
        raise AssertionError("out-of-array coordinate accepted")


class _FakeSmile:
    """Records every frame the experiment asks for."""

    def __init__(self, sim):
        self.frames = []
        self._sim = sim

    def set_pixel(self, x, y, bit, enables=None):
        self.frames.append(("one", ((int(x), int(y)),), int(bit))
                           + (() if enables is None else (enables,)))
        return self._sim.set_pixel(x, y, bit)

    def set_pixels(self, coords, bit):
        coords = [(int(x), int(y)) for x, y in coords]
        self.frames.append(("many", tuple(coords), int(bit)))
        return self._sim.set_pixels(coords, bit)

    def blank_frame(self, enables=None):
        self.frames.append(("blank", (), 0) + (() if enables is None else (enables,)))
        return self._sim.blank_frame()


def _cluster_run(ks, n_members=16):
    """exp_cluster against the built-in sim rig, with every frame recorded."""
    import measure_iqe as m
    from characterize_capacitance import build_parser

    cfg = m.IQEConfig()
    pm, smu, sim = m.connect(cfg, lambda _s: None, simulate=True)
    smile = _FakeSmile(sim)
    args = build_parser().parse_args(
        ["--what", "cluster", "--cluster-k", *[str(k) for k in ks],
         "--cluster-reps", "3", "--cluster-settle-s", "0", "--max-presettle-s", "0",
         "--cluster-n-pm", "2", "--cluster-n-smu", "2",
         "--pixels", *[f"{10 + i},{20 + i}" for i in range(n_members)]])
    rows = exp_cluster(pm, smu, smile, cfg, args, lambda _s: None)
    return rows, smile.frames


def test_the_sample_interval_matches_the_instrument_timestamps():
    """Four bench points, and every record-length claim rests on them.

    Two channels, from the instrument timestamps in charatests/261013-2140 and
    -2155 (NPLC 0.01, 1060 us, decimation stride 5) and -2201 and -2337 (NPLC 1.0,
    40670 us, stride 1). One channel, from results/261021-1051 on Array42 (NPLC
    0.01, 291 us; NPLC 1.0, 20145 us).

    Two models have already been wrong here, so this pins the hardware and not a
    belief about it:

      * max(2*nplc/50, 200e-6) predicted 400 us at NPLC 0.01 against 1060 us
        measured -- 2.65x short, and every record length derived from it with it.
      * then `channels*nplc/50 + 660e-6` predicted 860 us for ONE channel against
        291 us measured, because it assumed the overhead was per sample. It is per
        measure-CALL: 91 us for one, 660 us for two, not 182. An earlier version of
        this very test asserted that one and two channels differ by exactly one
        integration, which is what that wrong model implied -- and the bench says
        the extra channel costs one integration PLUS 569 us.
    """
    from characterize_capacitance import sample_interval_s

    # every measured point, both channel counts
    for ch, nplc, measured_us in ((2, 0.01, 1060.0), (2, 1.0, 40670.0),
                                  (1, 0.01, 291.0), (1, 1.0, 20145.0)):
        got = sample_interval_s(nplc, ch) * 1e6
        assert abs(got - measured_us) / measured_us < 0.01, (
            f'{ch}ch NPLC {nplc}: model {got:.0f} us against measured {measured_us:.0f} us')

    # the default must be the cheap one, or the long stage silently doubles
    from characterize_capacitance import build_parser
    assert build_parser().parse_args(['--what', 'step']).step_channels == 1

    # the integration scales with the channel count; the overhead does NOT
    two, one = sample_interval_s(1.0, 2), sample_interval_s(1.0, 1)
    assert one < two, 'one channel must cost less than two'
    extra = (two - one) - 1.0 / 50.0
    assert extra > 100e-6, (
        'the second channel costs more than one integration -- it adds its own call '
        'overhead, which is why a single fixed constant cannot fit both counts')

    # and the long stage must still outlast the 240 s faint-end settle it exists for
    a = build_parser().parse_args(['--what', 'step'])
    span = a.step_long_samples * sample_interval_s(a.step_long_nplc, a.step_channels)
    assert span > 1.5 * 240.0, (
        f'the long record is {span:.0f} s against a 240 s settle; halving the '
        f'per-sample time without raising --step-long-samples is how that breaks')

    # monotone, and the overhead dominates at the bottom
    assert sample_interval_s(0.001) > 600e-6, 'the fixed overhead must not vanish'
    prev = 0.0
    for nplc in (0.001, 0.01, 0.1, 1.0, 2.0):
        v = sample_interval_s(nplc)
        assert v > prev
        prev = v

def test_the_settling_verdict_weighs_sigma_and_never_guesses_a_dwell():
    """Run 261019-1057's real numbers, and the three branches of the conclusion.

    The first version of this verdict ranked PERCENTAGES. A single pixel emitting
    455 pW deviated 7.70 % at 50 ms -- but that is 35 pW against a 31 pW per-rep
    scatter, 1.1 sigma, nothing. A K=16 group emitting 5863 pW deviated only 2.70 %,
    which is 159 pW against 35.5 pW, 4.5 sigma, real. Ranking by percentage called the
    single pixel best-settled and inverted the answer.
    """
    from characterize_capacitance import settling_verdict

    def row(k, t, dev_frac, p, sd):
        dev = dev_frac * p
        return dict(k=k, probe_s=t, p_settled_W=p, dev_W=dev, dev_frac=dev_frac,
                    dev_sigma=abs(dev) / sd)

    # the measured run: K=1 unresolved, K=16 real at 50 ms
    srows = [row(1, 0.05, 0.0770, 454.7e-12, 31.1e-12),
             row(16, 0.05, 0.0270, 5863.0e-12, 35.5e-12)]
    out = chr(10).join(settling_verdict(srows, {1, 16}, 2.0))
    assert "nothing resolved at K=1" in out, out
    assert "NOT settled at 50 ms" in out and "+2.70%" in out, out
    # and it must not name a dwell it has not seen
    assert "Extend --cluster-probe-s past 2000 ms" in out, out
    assert "Settled by" not in out, "named a dwell from a scan that never settles"

    # a scan that DOES settle: the last significant probe is followed by TWO quiet
    # ones. 0.025 at 200 ms is 4.1 sigma and significant; 0.001 and 0.002 are not.
    # Two are needed because one quiet probe at the end of a scan has nothing after
    # it to confirm -- run 261019-1147's last point looked quiet and was noise.
    srows = [row(16, 0.05, 0.0270, 5863e-12, 35.5e-12),
             row(16, 0.20, 0.0250, 5863e-12, 35.5e-12),
             row(16, 0.50, 0.0010, 5863e-12, 35.5e-12),
             row(16, 1.00, 0.0020, 5863e-12, 35.5e-12)]
    out = chr(10).join(settling_verdict(srows, {16}, 2.0))
    assert "NOT settled at 200 ms" in out, out
    assert "Quiet from 500 ms" in out, out

    # an unordered scan must give the same answer as an ordered one
    assert settling_verdict(srows, {16}, 1.0) == settling_verdict(
        list(reversed(srows)), {16}, 1.0), "the verdict depends on row order"

    # an upper bound is not a zero, and it is quoted
    srows = [row(1, 0.05, 0.005, 454.7e-12, 31.1e-12)]
    out = chr(10).join(settling_verdict(srows, {1}, 2.0))
    assert "NOT shown to be zero" in out and "under ~6.8%" in out, out

    # no scan for a K means no line for it, never a crash
    assert settling_verdict([], {1, 8}, 2.0) == []

    # a scan that ends on its only quiet probe has nothing after it to confirm
    srows = [row(16, 0.05, 0.0270, 5863e-12, 35.5e-12),
             row(16, 0.20, 0.0250, 5863e-12, 35.5e-12)]
    out = chr(10).join(settling_verdict(srows, {16}, 1.0))
    assert "does not reach the dwell" in out, out
    assert "Quiet from" not in out, "named a dwell with no quiet tail"

    # run 261019-1147, K=16: the deviation never decays, so NO dwell may be named.
    # 11 probes from 50 ms to 20 s, P=5800.8 pW, per-rep scatter 20.1 pW. The tail
    # sits +0.77 % from settled at 6.4 sigma on the mean -- drift, not a transient,
    # and a 20 s dwell would fix none of it.
    real = [(0.05, 0.0215), (0.1, 0.0125), (0.2, 0.0110), (0.5, 0.0069),
            (1.0, 0.0110), (2.0, 0.0104), (3.0, 0.0052), (5.0, 0.0050),
            (8.0, 0.0097), (10.0, 0.0110), (20.0, 0.0050)]
    srows = [dict(row(16, t, f, 5800.8e-12, 20.1e-12), settle_ref_s=30.0)
             for t, f in real]
    out = chr(10).join(settling_verdict(srows, {16}, 30.0))
    assert "does NOT decay" in out and "DRIFT" in out, out
    assert "+0.77%" in out, out
    assert "cannot give a settling time" in out, out
    assert "Quiet from" not in out, "named a dwell under a non-decaying offset"


def test_the_cluster_reference_pools_and_is_never_a_blank_frame():
    """Section 6: an all-off frame conducts ~4x the bitval-0 level.

    So the reference for K lit pixels is the SAME K pixels at bitval 0. A blanked
    frame subtracts the wrong number, and the error grows with K.
    """
    rows, frames = _cluster_run([1, 4])
    lit = [f for f in frames if f[0] == "many" and f[2] != 0]
    off = [f for f in frames if f[0] == "many" and f[2] == 0]
    # lit[0] is the charge frame before the first measurement, so pair from lit[1:].
    # Every lit frame after it -- the per-member singles as well as the groups -- owes
    # a reference frame over exactly the same pixels.
    assert len(lit) - 1 == len(off), (
        f"{len(lit)-1} lit frame(s) but {len(off)} reference frame(s)")
    assert len(off) > len(rows), (
        "the per-member singles did not run: without them an additivity loss cannot "
        "be told apart from pixel-to-pixel variation")
    for (_t, members, _b), (_t2, ref, _b2) in zip(lit[1:], off):
        assert members == ref, f"the reference lit {ref}, the measurement lit {members}"
    assert not [f for f in frames[:-1] if f[0] == "blank"], (
        "a blanked frame appeared mid-experiment: it is not a valid reference")


def test_cluster_k_is_nested_so_the_comparison_is_the_same_pixels():
    """K=4 must be the first four members of K=16, never a fresh sample.

    A different sample per K would mix pixel-to-pixel spread into the scatter the
    experiment is trying to attribute to the meter.
    """
    rows, _frames = _cluster_run([1, 4, 16])
    assert [r["k"] for r in rows] == [1, 4, 16]
    sets = [r["members"].split() for r in rows]
    for small, big in zip(sets, sets[1:]):
        assert big[:len(small)] == small, "the K sets are not nested"
    # the verdict column must be the one the log quotes
    for r in rows:
        assert abs(r["rel_scatter_x_sqrt_k"] - r["rel_scatter"] * np.sqrt(r["k"])) < 1e-12



def test_both_highc_states_get_exactly_one_long_record():
    """The de-capped carrier exists to answer "does it settle without High-C".

    Only a long record answers that, and for a while only High-C had one: the guard
    read `if args.highc_both and not highc: return out`, so the non-High-C pass got
    fast captures only. The reason was cost -- non-High-C sweeps --ranges, and five
    long captures is 33 min a pixel -- but the cure threw away the measurement.

    So: one long record per High-C state, and the non-High-C one at a single range.
    """
    from characterize_capacitance import build_parser, step_stages

    a = build_parser().parse_args(["--what", "step", "--highc-both",
                                  "--ranges", "1e-3", "1e-5", "1e-6"])
    longs = [(highc, rng, st[3])
             for highc in (False, True)
             for rng in ([None] if highc else a.ranges)
             for v in a.step_voltages
             for st in step_stages(v, highc, rng, a) if st[0] == "long"]

    assert len(longs) == 2, f"want one long record per High-C state, got {longs}"
    assert {h for h, _, _ in longs} == {False, True}, longs

    (_, rng_off, lim_off), (_, rng_on, lim_on) = sorted(longs, key=lambda r: r[0])
    assert rng_off == 1e-6 and rng_off in a.ranges, longs
    # High-C: the limit IS the range, so it must drop or the ~13 nA tail at 8 s is
    # buried under ~100 nA of 1 mA-range noise.
    assert lim_on == a.step_long_compliance, longs
    # High-C off: the range comes from --ranges, so taking the low limit would only
    # smear the charge edge from 9 ms to 940 ms for nothing.
    assert lim_off is None, longs

    # a range absent from --ranges yields no non-High-C long record, not a crash
    b = build_parser().parse_args(["--what", "step", "--highc-both",
                                   "--ranges", "1e-3", "--step-long-ranges", "1e-9"])
    off = [st for rng in b.ranges for v in b.step_voltages
           for st in step_stages(v, False, rng, b) if st[0] == "long"]
    assert off == [], off

def test_a_worthless_row_says_so_instead_of_printing_plausible_numbers():
    """A long capture is 7 minutes. It must not end in numbers that only look real.

    Both cases are measured, from the Array42 run of 21 October 2026 at -0.80 V on
    the 1 uA range without High-C:

      fast  472.8 nA but 5695/6000 over-range -- `settled` is the median of the 305
            samples that fit, a biased minority
      long   35.6 nA against 27.1 nA of scatter, so settling_metrics widened the 1 %
            band to 13.6 nA and t_1pct fell back to half a smoothing window, 1.017 s.
            Printed bare that reads like fast settling. It means the opposite.

    Neither may fire on the clean High-C long record from the same run.
    """
    from characterize_capacitance import capture_verdict

    noisy = dict(n=20000, n_overflow=0, settled=35.6e-9, t_1pct_tol=13.55e-9)
    v = capture_verdict(noisy, "long", False, 1e-6)
    assert len(v) == 1 and "NOISE-LIMITED" in v[0], v
    assert "38 %" in v[0] and "35.6 nA" in v[0], "say how bad, with the numbers"

    # ...but a band only mildly widened still says something ("settled to within
    # 6 %"), and firing there buried the real case: the simulator alone produced
    # nine of these in a two-pixel run.
    mild = dict(n=6000, n_overflow=0, settled=2629.8e-9, t_1pct_tol=175.76e-9)
    assert capture_verdict(mild, "fast", False, 1e-6) == []

    over = dict(n=6000, n_overflow=5695, settled=472.8e-9, t_1pct_tol=4.7e-9)
    v = capture_verdict(over, "fast", False, 1e-6)
    assert len(v) == 1 and "OVER-RANGE" in v[0], v
    assert "305" in v[0], "name the surviving count, not just the percentage"

    # the clean record this comparison exists to produce must stay quiet
    good = dict(n=20000, n_overflow=18, settled=488.4e-9, t_1pct_tol=4.0e-9)
    assert capture_verdict(good, "long", True, None) == []

    # a row with no reading at all must not crash the logger it is called from
    assert capture_verdict(dict(n=0, n_overflow=0, settled=float("nan"),
                                t_1pct_tol=float("nan")), "fast", True, None) == []


def test_the_switch_happens_inside_the_burst_and_is_placed_on_it():
    """The record must be RUNNING when the frame changes, with a baseline before it.

    Until 2026-10-25 capture_step ran the trigger first and only then started the
    burst, so no switch record contained the switch. This drives the real-hardware
    path with a fake SMU that logs the order of calls: the trigger model is armed,
    started, the frame changes, and only then does the host wait for the buffer.
    """
    import time as _time
    from characterize_capacitance import _capture_with_baseline

    calls = []

    class _Inst:
        timeout = 10_000

        def clear(self):
            calls.append("clear")

    class _SMU:
        inst = _Inst()

        def arm_trigger_burst(self, n, channels=1):
            calls.append(("arm", n, channels))

        def start_trigger_burst_marked(self):
            calls.append("start")
            _time.sleep(0.002)                      # START comes back ~2 ms later
            return _time.perf_counter()

        def wait_trigger_burst(self):
            calls.append("wait")

        def abort_trigger(self):
            calls.append("abort")

        def read_buffer_b(self):
            ts = [i * 191e-6 for i in range(3000)]
            return ts, [5e-7] * 3000

    sync = {}
    t, ia, ib = _capture_with_baseline(
        _SMU(), -0.8, 3000, 0.005, 2000, None, lambda: calls.append("switch"), 1,
        0.05, sync, None)
    order = [c if isinstance(c, str) else c[0] for c in calls]
    assert order == ["arm", "start", "switch", "wait", "abort"], order
    assert calls[0][1:] == (3000, 1), "one record of every sample, ch B only"
    # the switch sits after the baseline on the trace's own axis, sent before acked
    assert 0.05 <= sync["trigger_lo_s"] <= sync["trigger_hi_s"] < 0.5, sync
    assert 0 < sync["first_unc_s"] < 0.05 and not sync["start_ack_late"], sync
    assert t[0] == 0.0 and len(t) == len(ib) == 3000 and np.isnan(ia).all()

    # a source step during the trigger model is unverified: refuse, not guess
    try:
        _capture_with_baseline(_SMU(), -0.8, 10, 0.01, 2000, None, None, 2, 0.1, {}, None)
    except ValueError:
        pass
    else:
        raise AssertionError("pre_s without an external trigger must be refused")


def test_a_simulated_switch_run_records_where_the_switch_is():
    """exp_switch end to end on the sim rig: the CSV row carries the sync columns."""
    import measure_iqe as m
    from characterize_capacitance import build_parser, exp_switch

    cfg = m.IQEConfig()
    pm, smu, sim = m.connect(cfg, lambda _s: None, simulate=True)
    args = build_parser().parse_args(
        ["--what", "switch", "--pixels", "300,40", "--switch-from", "10,20",
         "--voltages", "-1.2", "--switch-hold-s", "0.35", "--max-presettle-s", "0.35",
         "--switch-refresh-s", "0.1", "--samples", "200", "--switch-pre-s", "0.05",
         "--switch-n-pm", "1"])
    smile = _FakeSmile(sim)
    rows = exp_switch(pm, smu, smile, (300, 40), cfg, args, lambda _s: None)
    r = rows[0]
    # the hold pixel is re-written during the hold (0.35 s at 0.1 s, plus once just
    # before the record), then straight to the target: no blank frame in between
    hold = [i for i, f in enumerate(smile.frames) if f == ("one", ((10, 20),), 15)]
    assert len(hold) >= 4, smile.frames
    assert smile.frames[hold[-1] + 1] == ("one", ((300, 40),), 15), smile.frames
    assert all(f[0] == "one" for f in smile.frames[hold[0]:hold[-1] + 2]), smile.frames
    assert (r["from_quadrant"], r["to_quadrant"]) == ("TL", "TR"), r
    assert 0.05 <= r["sync_switch_sent_s"] <= r["sync_switch_ack_s"], r
    assert r["pm_on_trace_axis"] == 1 and r["trace_tag"].startswith("switch_"), r
    assert np.isfinite(r["fpga_refresh_ms"]), "the last pre-record re-write is timed"


def test_the_switch_matrix_runs_every_pair_and_times_every_write():
    """Two pixels per quadrant, four same-quadrant pairs and three cross pairs, and
    every frame write of every capture placed on that capture's own time axis."""
    import measure_iqe as m
    import characterize_capacitance as cc
    from characterize_capacitance import build_parser, exp_switchmatrix

    cfg = m.IQEConfig()
    pm, smu, sim = m.connect(cfg, lambda _s: None, simulate=True)
    args = build_parser().parse_args(
        ["--what", "switchmatrix", "--swm-voltages", "-1.2", "--swm-window", "4",
         "--swm-hold-s", "0.2", "--switch-refresh-s", "0.05", "--max-presettle-s", "0.2",
         "--swm-samples", "100", "--swm-gap-s", "0.01", "--swm-pre-s", "0.02"])
    cc._EXTRA_ROWS.clear()
    smile = _FakeSmile(sim)
    rows = exp_switchmatrix(pm, smu, smile, cfg, args, lambda _s: None)
    pairs = [(r["a_slot"], r["b_slot"]) for r in rows]
    assert pairs == [("p1", "p2"), ("p3", "p4"), ("p5", "p6"), ("p7", "p8"),
                     ("p1", "p3"), ("p1", "p5"), ("p1", "p7")], pairs
    assert [r["kind"] for r in rows] == ["same"] * 4 + ["cross"] * 3
    assert all(r["a_quadrant"] == q for r, q in zip(rows[:4], ("TL", "TR", "BL", "BR")))
    ev = cc._EXTRA_ROWS["switchmatrix_events"]
    for r in rows:
        mine = [e for e in ev if e["trace_tag"] == r["trace_tag"]]
        assert [e["label"] for e in mine] == (["A->B"] + ["B->A", "A->B"] * 3
                                              + ["B off", "B on"]), mine
        assert mine[-2]["bitval"] == 0 and mine[-1]["bitval"] == 15
        t = [x for e in mine for x in (e["sent_s"], e["ack_s"])]
        assert t == sorted(t) and t[0] >= 0.02, "writes in order, after the baseline"
        assert r["n_events"] == 9 and np.isfinite(r["fpga_refresh_ms"]), r
    cc._EXTRA_ROWS.clear()


def test_the_leakage_matrix_measures_every_state_with_its_enables():
    """Three biases x eight frame states, each frame written with the enables it names."""
    import measure_iqe as m
    from characterize_capacitance import LEAKM_STATES, build_parser, exp_leakmatrix

    cfg = m.IQEConfig(highc_b=False)
    pm, smu, sim = m.connect(cfg, lambda _s: None, simulate=True)
    args = build_parser().parse_args(
        ["--what", "leakmatrix", "--pixels", "300,40", "--leakm-settle-s", "0",
         "--leakm-state-settle-s", "0", "--leakm-range-settle-s", "0",
         "--leakm-n-smu", "4", "--leakm-n-pm", "2"])
    smile = _FakeSmile(sim)
    rows = exp_leakmatrix(pm, smu, smile, (300, 40), cfg, args, lambda _s: None)
    assert len(rows) == 3 * len(LEAKM_STATES)
    assert [r["v"] for r in rows[::len(LEAKM_STATES)]] == [0.0, -0.2, -0.4]
    assert {r["enables"] for r in rows} == {"own", "all", "none"}
    assert all(r["range_a"] >= 1e-6 and r["quadrant"] == "TR" for r in rows)
    assert all(r["NVLED_CURR_N"] == 4 for r in rows), "every state has its own burst"
    # the frames carry their enables: own is the plain write, all/none are explicit
    sent = [f for f in smile.frames if f[0] in ("one", "blank")]
    assert ("one", ((300, 40),), 0, "none") in sent and ("blank", (), 0, "none") in sent
    assert ("one", ((300, 40),), 15, "own") in sent


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_"):
            fn()
            print(f"  ok  {name}")
    print("all switch-deviation and cluster checks passed")
