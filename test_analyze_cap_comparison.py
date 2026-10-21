"""Checks for analyze_cap_comparison.py.

    uv run python test_analyze_cap_comparison.py

The power-law fit is the one piece of real logic here -- everything else reads a CSV
and draws it -- and it was wrong twice before this test existed, so it gets a synthetic
record whose exponent is known.
"""
from __future__ import annotations

import numpy as np

from analyze_cap_comparison import (OVERFLOW, Run, clean, fit_power_law, log_bin,
                                    noise_sigma)

# the real record: 20000 samples at 20.1 ms, I_inf = 480 nA, 213 nA of excess at 2 s
T = np.arange(1, 20000) * 20.1e-3
I_INF, AMP = 480e-9, 213e-9


def synth(alpha, noise_nA=0.0, seed=7):
    rng = np.random.default_rng(seed)
    return I_INF + AMP * (T / 2.0) ** -alpha + rng.normal(0, noise_nA * 1e-9, T.size)


def test_the_exponent_is_recovered_and_does_not_depend_on_the_noise():
    """Two bugs made an earlier version of this fit wrong, both invisible without it.

    1. It fitted the RAW samples. They are uniform in t, so a 411 s record puts ~95 %
       of its points in the last decade where the excess is smallest, and an unweighted
       log-log fit is dragged toward slope 0 -- with NO noise at all a true 1.6 came
       back as 0.584 and a true 1.2 as 0.787. Log-binning gives each decade of time
       equal weight, which is what a power law needs.
    2. The I_inf grid was 400 points over [0, min(I)], a step of ~1.2 nA. An I_inf
       wrong by 1 nA leaves a constant offset in the tail that flattens the fit, and
       that alone still put a true 1.6 at 1.12. A refinement pass fixes it.

    The bands are the measured worst case over 0, 1 and 3 nA of added noise, not a
    guess. Error grows with the exponent because a steeper law sinks under the noise
    sooner, so there is less of it left to fit:

        alpha  0.6    0.8    1.0    1.2    1.4
        worst  0.001  0.002  0.006  0.013  0.051

    The expected exponent is 1.2 +/- 0.2, where the fit is good to 0.013 -- about 15x
    finer than the spread the physics itself shows.
    """
    bands = {0.6: 0.005, 0.8: 0.005, 1.0: 0.01, 1.2: 0.02, 1.4: 0.05}
    for alpha, band in bands.items():
        for noise in (0.0, 1.0, 3.0):
            got, iinf, r2, nbin = fit_power_law(T, synth(alpha, noise), t_min=1.0)
            assert abs(got - alpha) < band, (
                f"alpha {alpha} with {noise} nA noise came back {got:.3f}")
            assert abs(iinf - I_INF) < 0.02 * I_INF, (
                f"I_inf {iinf * 1e9:.2f} nA against {I_INF * 1e9:.0f}")
            assert r2 > 0.99, f"alpha {alpha}, {noise} nA: r2 only {r2:.4f}"
            assert nbin >= 90, nbin

    # the fit must not be noise-sensitive: 0 and 3 nA may differ by far less than the
    # 0.2 spread the physics itself shows
    spread = [fit_power_law(T, synth(1.2, n), t_min=1.0)[0] for n in (0.0, 1.0, 3.0)]
    # 0.021 measured, against the +/-0.2 the physics itself spreads over
    assert max(spread) - min(spread) < 0.03, spread


def test_the_approach_is_fitted_from_whichever_side_it_comes():
    """Array42 RISES to its asymptote. A one-sided fitter called that alpha = -0.04.

    The first version searched I_inf only BELOW the data, which can only describe a
    decay from above. The real High-C record goes 312 -> 419 -> 474 -> 481 nA, an
    approach from below, and the fit returned alpha = -0.043 at r2 = 0.60 -- a
    straight line through the wrong model, which is worse than no answer. For a rise
    I_inf simply sits above the data, so the grid has to cover both sides.
    """
    for alpha in (0.6, 0.8, 1.0):
        for noise in (0.0, 1.0, 3.0):
            rng = np.random.default_rng(7)
            rising = (I_INF - AMP * (T / 2.0) ** -alpha
                      + rng.normal(0, noise * 1e-9, T.size))
            got, iinf, r2, _ = fit_power_law(T, rising, t_min=1.0)
            assert abs(got - alpha) < 0.01, (alpha, noise, got)
            assert abs(iinf - I_INF) < 0.02 * I_INF, iinf
            assert r2 > 0.99, r2
            # and I_inf must land ABOVE the data it was fitted from
            assert iinf > np.median(rising[T > 1.0]), (iinf, np.median(rising))


def test_a_steep_exponent_degrades_honestly_instead_of_lying():
    """A steeper law sinks under the noise sooner, so there is genuinely less to fit.

    That is physics, not a bug -- but it must show up, or a bad number gets quoted as
    a result. The measured error is 0.005 at alpha 1.0 and 0.13 at alpha 2.0, and r2
    falls with it. The real Array42 records sit at alpha 0.77-1.01, r2 0.97-0.99.
    """
    errs, r2s = [], []
    for alpha in (1.0, 1.4, 2.0):
        worst = max(abs(fit_power_law(T, synth(alpha, n))[0] - alpha)
                    for n in (0.0, 1.0, 3.0))
        errs.append(worst)
        r2s.append(min(fit_power_law(T, synth(alpha, n))[2] for n in (0.0, 1.0, 3.0)))
    assert errs[0] < 0.01 and errs[-1] < 0.15, errs
    assert errs[0] < errs[-1], f"error must grow with the exponent, got {errs}"
    assert r2s[0] > r2s[-1] > 0.98, r2s


def test_a_record_too_short_to_fit_returns_nan_rather_than_a_number():
    """The 40 s smoke-test record has no resolvable power law. Say so, do not guess."""
    t = np.arange(1, 60) * 20.1e-3                  # 1.2 s, nothing to fit
    alpha, iinf, r2, nbin = fit_power_law(t, I_INF + AMP * t ** -1.2, t_min=1.0)
    assert not np.isfinite(alpha) and nbin == 0, (alpha, nbin)


def test_overflow_sentinels_never_reach_the_arithmetic():
    """9.91e37 is the 2600's over-range marker. Left in, it poisons every median."""
    y = np.array([1e-9, 2e-9, 9.91e37, 3e-9, np.nan])
    t = np.arange(y.size, dtype=float)
    tc, yc = clean(t, y)
    assert yc.size == 3 and np.all(np.abs(yc) < OVERFLOW), yc
    assert np.array_equal(tc, [0.0, 1.0, 3.0]), tc


def test_noise_sigma_measures_the_noise_and_not_the_trend():
    """A std over the record measures the relaxation. Consecutive differences do not."""
    clean_trace = I_INF + AMP * (T / 2.0) ** -1.2
    assert noise_sigma(clean_trace) < 0.05e-9, noise_sigma(clean_trace)
    noisy = synth(1.2, 3.0)
    got = noise_sigma(noisy)
    assert abs(got - 3e-9) < 0.4e-9, f"{got * 1e9:.2f} nA against 3 nA put in"
    # and the naive version would be wildly wrong, which is why this helper exists
    assert np.std(clean_trace) > 10 * noise_sigma(clean_trace)


def test_log_bin_keeps_the_shape_it_is_asked_to_summarise():
    tb, yb = log_bin(T, I_INF + AMP * (T / 2.0) ** -1.2, nbin=50)
    assert tb.size == yb.size <= 50 and tb.size > 20, tb.size
    assert np.all(np.diff(tb) > 0), "bins must come out in time order"
    assert np.all(np.diff(yb) < 0), "a decay binned must still decay"


def test_a_run_directory_with_neither_csv_is_not_silently_empty(tmp=None):
    """Run() must load what is there and report what is not, without raising."""
    import tempfile
    from pathlib import Path
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "261021-1200_Nothing_CAPACITANCE"
        p.mkdir()
        r = Run(p)
        assert r.name == "Nothing" and r.stamp == "261021-1200", (r.name, r.stamp)
        assert r.cap == [] and r.step == [] and r.keys == []
        assert r.step_traces() == [] and r.cap_traces() == []


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_"):
            fn()
            print(f"  ok  {name}")
    print("analysis checks passed")
