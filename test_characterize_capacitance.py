"""Self-checks for the switch-deviation metric.

The metric exists to answer one question at the bench: does a frame switch cost
current that the dwell then has to pay back? Run 261016-0726 is the case the old
version failed on, so it is the case this file pins down. Section 21.

    python test_characterize_capacitance.py
"""
import numpy as np

from characterize_capacitance import switch_deviation

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


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_"):
            fn()
            print(f"  ok  {name}")
    print("all switch-deviation checks passed")
