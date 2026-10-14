"""Configuration for the IQE acquisition.

Every tunable of the measurement lives on :class:`IQEConfig` as a field with a
safe default.  The defaults are the values that were proven on Array38 — change
them by constructing the dataclass, or load a JSON/YAML file with
:meth:`IQEConfig.from_file`.

    >>> from measure_iqe import IQEConfig
    >>> cfg = IQEConfig(roi=(0, 0, 31, 31), sweep_direction="low_to_high")
    >>> cfg = IQEConfig.from_file("iqe_config.yaml")

The inline comments below are the reference documentation for each parameter:
they record *why* a value is what it is (which bench run pinned it, what broke
when it was different).  Keep them with the field.
"""

from __future__ import annotations

import dataclasses
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# Physical constants
Q_E = 1.602176634e-19             # elementary charge (C)
H_PLANCK = 6.62607015e-34         # Planck constant (J s)
C_LIGHT = 2.99792458e8            # speed of light (m/s)

# 2602B current measurement ranges (A), ascending — per-point ranging picks from these
SMU_I_RANGES: tuple[float, ...] = (1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2)

# Named ROI presets accepted by ``roi`` (and by ``--roi`` on the CLI).
NAMED_ROIS: dict[str, tuple[int, int, int, int]] = {
    "FULL": (0, 0, 511, 511), "TL": (0, 0, 255, 255), "TR": (256, 0, 511, 255),
    "BL": (0, 256, 255, 511), "BR": (256, 256, 511, 511),
}


@dataclass
class IQEConfig:
    """All acquisition parameters.  Every field has a working default."""

    # ── Run identity & output ───────────────────────────────────────────
    # base_dir is relative to the working directory unless you pass an absolute
    # path.  Nothing outside it is ever written.
    sample_name: str = "Array38_IQE"
    base_dir: str = "results"          # output root; a timestamped run dir is created inside

    # ── ROI & pixel selection ───────────────────────────────────────────
    roi: tuple[int, int, int, int] | str = (32, 32, 63, 63)  # (x1,y1,x2,y2) or "FULL"/"TL"/"TR"/"BL"/"BR"
    # Which alive pixels get the (expensive) full IQE sweep:
    #   "blocks"  — STRATIFIED: tile the ROI into BLOCK_SIZE×BLOCK_SIZE-pixel blocks and take
    #               PIXELS_PER_BLOCK from EACH block → equal per-block sampling for spatial
    #               (block / row-band / column-band) statistics. Within a block the pixels are
    #               spread by FPS, but each block's FPS STARTS from a RANDOM alive pixel (seeded
    #               by BLOCK_SEED), so the sampled rows/columns are diversified across blocks
    #               instead of landing on the same sub-lattice every time. Change BLOCK_SEED
    #               between runs to sample DIFFERENT pixels (cumulatively cover the array).
    #   "spatial" — fraction of alive pixels by global farthest-point sampling (uniform map).
    #   "stride"  — old every-Nth (aliases columns; kept for compatibility).
    sweep_select: str = "blocks"  # "blocks" | "spatial" | "stride"
    block_size: int = 8  # blocks mode: block edge in pixels (ROI → ~ROI/BLOCK_SIZE blocks/axis)
    pixels_per_block: int = 4  # blocks mode: alive pixels sampled per block
    block_seed: int = 0  # blocks mode: RNG seed (vary across runs to sample new pixels)
    # blocks mode: no two measured pixels may sit closer than this (Chebyshev, so 2 =
    # never sharing an edge or a corner). Without it, each block's farthest-point walk
    # ran to the block's own border and the block next door did the same, so pairs of
    # measured LEDs landed side by side across the shared edge — the sample looked
    # stratified but was really clustered, and neighbouring blocks' statistics blurred
    # into each other. Raise it to spread the sample further (thermal/optical crosstalk
    # between nearby pixels is a physical question, so it is a knob); 1 disables the
    # guard. A block too dead or too small to honour it falls back and says so in the log.
    min_pixel_separation: int = 2
    sweep_fraction: float = 0.10  # spatial mode: fraction of ALIVE pixels to sweep
    min_sweep_pixels: int = 6  # spatial mode: floor on the count (so tiny ROIs still map)
    nth_pixel_sweep: int = 5  # stride mode: sweep every Nth alive pixel
    max_sweep_pixels: int = 40  # hard cap (spatial/stride only; blocks controls its own count)
    fastscan_bitval: int = 15  # bitval for the alive/dead fast scan (brightest)

    # ── Source mode ─────────────────────────────────────────────────────
    # "voltage": ch B sources voltage, adaptive current-targeted sweep. PROVEN on
    #            Array38 — the 10 mA voltage sweep captured the IQE peak (J≈6,
    #            ~59%). USE THIS.
    # "current": ch B sources (sinks) current, measures voltage. Tried on Array38
    #            (run 260624-1310) and it DOES NOT WORK on this device: when ch B is
    #            told to sink a small current (e.g. 0.3 µA), the +1.8 V anode (ch A)
    #            keeps sourcing the full LED current (~60 µA) through a parallel path
    #            that bypasses ch B — so ch B's setpoint is NOT the LED current
    #            (the two channels diverge, leak≈1.0) and the LED current can't be
    #            pulled below ~2 µA. Current-sinking the cathode cannot control this
    #            device's injection. (To pursue current-source you'd have to source on
    #            the anode side or remove the leakage path — needs the pixel schematic.)
    source_mode: str = "voltage"

    # ── Current-source sweep (source_mode = "current") ──────────────────
    current_sweep_low_a: float = 3e-7  # lowest LED current setpoint (1 µA → J ≈ 1)
    current_sweep_high_a: float = 1.2e-4  # highest setpoint (~120 µA; capped by V compliance)
    current_sweep_n: int = 50  # log-spaced current setpoints
    nvled_v_limit: float = 3.2  # |voltage| compliance while sourcing current (V)
    current_sweep_bitval: int = 15  # single bitval — CMOS fully open, we set I directly
    current_source_range_i: float = 1e-3  # fixed source current range (covers 120 µA; gives
                                     # good low-current resolution). Tune on the bench.
    measure_v_range: float = 6.0  # ch B voltage measure range (covers ±3.2 V)

    # ── Voltage-source sweep (source_mode = "voltage", DEFAULT) ─────────
    # Bitval 15's voltage sweep already spans the full J range (low J via turn-off,
    # high J at the saturation knee). Extra bitvals only ADD low-J density — and the
    # bench data showed the peak-IQE estimate depends on that density. So instead of
    # multiple bitvals (3× the runtime), use a SINGLE bitval but bias the fine sweep
    # toward the low-current rising side (LOWJ_WEIGHT) and integrate longer there
    # (LOWJ_NPLC). That samples the rise better than the old [15,4,1] sweep did, faster.
    bitvals: tuple[int, ...] = (15,)  # single bitval — densify low-J via spacing instead
    # Fine-sweep mode:
    #  "voltage" — plain UNIFORM voltage ramp NVLED_V_COARSE_START → STOP (no coarse map). Hits
    #     every voltage including the steep turn-on, so the rising side is densely sampled. Simple,
    #     robust — this is what the old clean runs effectively did. RECOMMENDED.
    #  "current" — coarse pre-sweep maps V→I, then place points at log-spaced CURRENTS. Even log-J
    #     spacing, BUT it skips voltages where dI/dV is steep (the ~−0.9..−1.1 V gap) and depends on
    #     the coarse map (which drove the ranging bugs).
    fine_sweep_mode: str = "voltage"
    nvled_v_coarse_start: float = -0.8  # least-negative sweep voltage (just below turn-on)
    nvled_v_coarse_stop: float = -2.6  # most-negative sweep voltage (below saturation knee)
    n_coarse: int = 26  # coarse pre-sweep points
    coarse_settle_ms: float = 6.0
    coarse_n_smu: int = 4
    # Where the fit actually needs data: across 140 per-pixel fits the device's IQE peak
    # sits at J ≈ 5 A/cm² (≈5 µA) and the trustworthy fits only reach down to J ≈ 1 (≈1 µA)
    # on the rising side. So target J ≈ 1–10, NOT sub-µA: below ~1 µA the data is unneeded
    # AND that is exactly where the fine measure range takes ~seconds to settle. Starting at
    # ~1 µA keeps every point on a fast-settling range and puts the points where the peak is.
    low_current_target_a: float = 1e-6  # lowest LED current the sweep targets (~J 1, the rise foot)
    high_current_frac: float = 0.97
    n_fine: int = 30  # fine voltage-sweep points. Lowered from 45 once the settle time was
                      # measured: the dwell dominates the budget, so fewer points with a
                      # real settle beat more points that never settled. 30 points x 2
                      # drive states x ~8 s is ~10 min/pixel.
    lowj_weight: float = 1.5  # >1 concentrates fine points toward LOW current (the SRH
                                     # rising side J≈1–5 that pins peak IQE). 1.0 = uniform log.
    # Voltage-ramp densification (FINE_SWEEP_MODE="voltage"). The rising side + peak that pin
    # the fit live in a narrow voltage window around turn-on (~−0.85 to −1.3 V), while the droop
    # spans the rest. So split the ramp: put VRAMP_LOW_FRAC of the points in the less-negative
    # segment [START, KNEE] (rising side + peak, dense) and the rest in [KNEE, STOP] (droop).
    vramp_knee_v: float | None = -1.3  # split voltage between the dense (rising) and sparse (droop) halves
    vramp_low_frac: float = 0.6  # fraction of points placed in the rising-side segment
    # Sweep direction (thermal A/B test). "low_to_high" measures the rising side first
    # (coolest, recommended); "high_to_low" starts hot. Run both on the same pixels and
    # compare the low-J EQE to gauge self-heating: if high_to_low reads lower IQE at low
    # J, self-heating is biasing you and low_to_high is the better number.
    sweep_direction: str = "high_to_low"  # "low_to_high" | "high_to_low"

    # ── Pixel warm-up / conditioning ────────────────────────────────────
    # The story took two turns. The −1.0 V diag (260629-0733) showed the LIGHT is flat
    # and conditioning did nothing — because at 5 µA a parasitic is invisible. The
    # −0.85 V diag (260629-0858), at the LOW J that sets peak IQE, revealed the real
    # effect: on a fresh/cold turn-on the measured CURRENT grows to ~1 µA over tens of
    # seconds while the LIGHT stays put — a NON-RADIATIVE parasitic current that only
    # contaminates the EQE denominator. A high-current dwell clears it (conditioned
    # current sits steady at the true ~38 nA). That parasitic is 27× the signal at low J
    # and negligible at high J — exactly the low→high(fresh) vs high→low(conditioned)
    # split. So conditioning is back ON: a short per-pixel high-current dwell makes the
    # low-J current clean and reproducible. (It mostly bit the FIRST pixel after the fast
    # scan, but per-pixel priming is cheap insurance for all of them.)
    warmup_ms: float = 1500.0  # 0 = off. Dwell at WARMUP_NVLED_V before each sweep.
    warmup_nvled_v: float = -2.6  # warm-up bias (high current, clears the parasitic)
    # `--warmup-diag` only: watch P, I and EQE vs time at a fixed operating point on
    # several pixels. Pick the point in the J≈1 region where the A/B split was largest —
    # low enough to stress the current readout, but where the LIGHT is still a few× above
    # the dark floor. (−0.6 V was too low: runs 0759/0847 sat AT the dark floor with the
    # current in the hypersensitive turn-on toe, so neither channel was informative.)
    warmup_diag_v: float = -0.85  # ≈J 1: stresses current readout, light still > dark
    warmup_diag_n_pixels: int = 3  # how many of the selected pixels to characterise
    warmup_diag_seconds: float = 30.0  # log duration per phase (fresh turn-on / conditioned)
    warmup_diag_nplc: float = 1.0
    warmup_diag_n_smu: int = 8
    warmup_diag_n_pm: int = 4

    # ── Adaptive per-point effort by current zone ───────────────────────
    # The warmup-diag (260629-0733) showed the LIGHT is the clean channel (~3% at a
    # fixed point) and the CURRENT is the noisy one. So the effort budget is on the SMU:
    #   • LOW current (rising side, sets peak IQE): pour integration into the CURRENT —
    #     high NPLC + many SMU samples (σ_mean ∝ 1/√N_SMU) — and only a few PM reads since
    #     the light is already stable. Per-point ranging (range_for_current) puts it on a
    #     FINE range — essential, since the settling probe (260629-1130) showed a coarse
    #     range reads low J not just noisily but WRONG (10 mA → −1.9 µA when truth is −38 nA).
    #   • HIGH current (droop): both channels strong → go FAST (short integration).
    # The fine low-J range also settles more slowly, so low-J points get a longer per-point
    # settle (LOWJ_SETTLE_MS) than the fast high-J points (HIGHJ_SETTLE_MS).
    adaptive_threshold_a: float = 1e-5  # 10 µA: boundary between the low-J and high-J zones
    lowj_nplc: float = 2.0
    lowj_n_pm: int = 8
    lowj_n_smu: int = 16  # low J: integrate the CURRENT harder
    highj_nplc: float = 0.5
    highj_n_pm: int = 4
    highj_n_smu: int = 8  # high J: fast
    lowj_settle_ms: float = 200.0
    highj_settle_ms: float = 15.0  # fine low-J range needs more settle time

    # ── Instruments ─────────────────────────────────────────────────────
    pm400_addr: str = "USB0::0x1313::0x8075::P5006526::INSTR"
    smu_addr: str = "USB0::0x05E6::0x2602::4649721::INSTR"

    # ── PM400 ───────────────────────────────────────────────────────────
    pm_wavelength: float = 440.0
    pm_range: float = 1e-7  # W, fixed (no autorange)
    pm_averaging: int = 3  # internal averaging per reading

    # ── VLED (ch A) — constant LED anode drive ──────────────────────────
    vled_voltage: float = 1.8
    vled_compliance: float = 1.0
    vled_range_i: float = 10e-3

    # ── NVLED (ch B) — voltage-mode current range ───────────────────────
    nvled_compliance: float = 1e-5  # current compliance (voltage mode). In High-C this ALSO
                                     # selects the measure range, so it sets the noise floor:
                                     # 10 uA gives ~9 nA/sample at NPLC 0.01, ~1 nA at NPLC 1.
                                     # Measured 261013. The sweep overrides it per segment
                                     # (VRAMP_*_LIMIT_A) because -2.6 V needs ~175 uA.
    nvled_range_i: float = 10e-3  # default fixed measure current range (coarse pre-sweep,
                                     # fast scan, and any point with no current target).
    # Current readout is the dominant error: ~5 µA on the 10 mA range is 0.05% of full
    # scale (and the low-J rise, <1 µA, is ~0.01%) → 20–50% scatter. AUTORANGE was tried
    # (260629-0913/1000) and BROKE the measurement: with the fast zero-delay BURST read
    # (delay=0, autozero off) autorange can't range up in time, so ch B read 100–800× too
    # LOW (157 µA real → 0.2 µA reported; leakage gate then killed every point). So instead
    # of autorange we set the range DETERMINISTICALLY per point from its known target
    # current (the coarse map gives it) — optimal resolution, correct value, no hunting.
    # (There is no autorange knob: hardware autorange is broken with the burst read and the
    # code never turns it on — the range is always set explicitly by set_measure_range_b.)
    perpoint_range: bool = False  # False = fixed NVLED_RANGE_I (10 mA): consistent (offsets and
                                     # all) so the curve SHAPE comes through — what the old clean runs
                                     # used. True = per-point software ranging (accurate value, but
                                     # range offsets/settling corrupted the rising side; see 260630).
    # Picking the range from the COARSE pre-sweep map failed both directions (260630): the
    # coarse map (10 mA range, pre-conditioning) mis-predicts the fine-sweep current, so the
    # range was too low at high bias (→ overflow, 104 pts) and too coarse on the rise (→ stuck
    # 0.8 µA offset). Fix: PROBE the actual current on a safe wide range first, then narrow.
    probe_range_i: float = 1e-3  # wide probe range: covers the device max (~175 µA), never
                                     # overflows, gives the magnitude to pick the real range from
    probe_n_smu: int = 4  # quick samples for the magnitude probe
    range_settle_ms: float = 50.0  # settle after a measure-range change before reading
    # 2602B High-C mode on ch B. The driver guards High-C on B (raises), so we enable it via
    # TSP. High-C adds settling delays (slower) but damps a capacitive load that can make the
    # low current ranges ring/never settle — the reason 10 mA was the safe range. Off by
    # default; sweep_variants.py toggles it to test whether it makes low ranges usable.
    highc_b: bool = True  # MANDATORY, measured 261013. Without High-C every reading below
                          # ~10 uA is a range artifact: it lands at 42-58% of whatever range
                          # is selected, independent of bias, wrong by up to 517x. With it,
                          # three different start voltages agree to 0.6%.

    # SMU autozero for the voltage sweep. The 2600's ADC has a small zero/offset that depends on the
    # integration time (NPLC); the per-segment effort switches NPLC 2↔0.5 at the knee, and if the zero
    # reference is stale across that switch the CURRENT reads with a small NPLC-dependent step → a ~1–3%
    # discontinuity in EQE=P/I at V≈-1.3, seen on every pixel (the voltage path previously left autozero
    # UNMANAGED). "AUTO" re-zeros as needed at each NPLC (removes the step; costs one autozero per
    # NPLC-change burst); "ONCE" zeroes once at the start (fast, but the step returns); "OFF" disables it.
    sweep_autozero: str = "AUTO"  # AUTO | ONCE | OFF

    # Save every individual burst sample (not just the per-point summary) to a raw-samples CSV, and
    # write mean/median/std/SEM (not just one summary) into the aggregated file. Off → old behaviour.
    save_raw_samples: bool = True

    # ── Integration time (NPLC) per phase ───────────────────────────────
    fastscan_nplc: float = 0.05  # Now the fast-scan CURRENT matters: at −1.35 V the fast scan is a
                                     # full-coverage peak-IQE proxy (EQE=P/I on the EQE plateau), so its
                                     # current noise sets the proxy quality. Sweep test (260704-1456/1618/
                                     # 1659): NPLC 0.01→0.05 cut the current relSEM 18%→3% and the proxy's
                                     # per-pixel error 14%→5% (down to the device's ~4% uniformity floor)
                                     # for only +3 s. NPLC 0.1 + N_SMU 8 barely helped (→4.7%) but ~60%
                                     # slower — so 0.05 / N_SMU 4 is the sweet spot.
    fastscan_n_smu: int = 4  # SMU samples/pixel in the fast scan
    coarse_nplc: float = 0.10  # voltage-mode coarse I(V) map: rough, quick
    sweep_nplc: float = 1.0  # the accurate IQE data (raise toward 2–5 for very low J)

    # ── Settle + averaging ──────────────────────────────────────────────
    settle_ms: float = 15.0  # wait after each setpoint before averaging
    fastscan_settle_ms: float = 10.0  # fast scan settle after lighting a pixel. Back to 10 ms (was
                                     # 5) — the 5 ms save wasn't worth risking a not-fully-settled
                                     # optical read on the liveness classifier.
    fastscan_random_order: bool = True  # visit fast-scan pixels in a shuffled order (not a raster). A
                                     # raster correlates any residual/crosstalk with spatial position;
                                     # randomizing turns it into position-independent noise.
    fastscan_order_seed: int = 0  # seed for the shuffle (reproducible)
    fastscan_blank_between: bool = True  # send a black (all-off) frame before lighting each pixel, so
                                     # exactly ONE pixel is ever emitting when the PM reads — guards
                                     # against a lingering previous pixel / accumulated frame inflating
                                     # the optical power (the ~71 nW edge-pixel false-alive, run 1405).

    # ── Dead-pixel early abort (voltage sweep) ──────────────────────────
    # Liveness is gated by the fast scan, which has an optical background floor that can mark a
    # dead edge pixel "alive" (run 1405: 3/4 corner pixels read ~71 nW in the fast scan but 0.05 nW
    # in the sweep). The sweep runs high→low, so its FIRST point is the brightest — if a real pixel
    # would emit strongly there and this one sits at the noise floor, it's dead: skip the remaining
    # points instead of spending the full low-J effort (~40 pts) on it.
    dead_abort: bool = True
    dead_abort_nw: float = 2.0  # absolute floor (nW): first-point power below this ⇒ dead
    dead_abort_frac: float = 0.10  # or below this fraction of the running median of live pixels
    thermal_settle_ms: float = 100.0  # before the fine sweep, sit at the FIRST point this
                                     # long so the pixel reaches its starting-current thermal
                                     # steady state (clears heat left by the coarse pre-sweep)
    n_smu_samples: int = 8  # SMU samples averaged per point
    n_pm_samples: int = 12  # PM400 readings averaged per point

    # ── Per-segment effort (fine_sweep_mode="voltage") ──────────────────
    # The knee (VRAMP_KNEE_V) splits not just point DENSITY but measurement EFFORT. The low-V /
    # rising-side segment (|V| < |knee|) has faint light near the dark floor, so take more PM
    # reads there (σ_light ∝ 1/√N_PM); the high-V droop is bright → go fast.
    # NB (run 1526): raising NPLC (2→5) only halved the CURRENT random noise (7→3.7% at J≈1), which
    # was already well within spec and doesn't matter to the unweighted fit — and it does NOT touch
    # the LIGHT noise (NPLC is SMU-only; light noise is set by N_PM). Same IQE result, more cost.
    # So keep NPLC modest; the low-J limiter is the light SIGNAL near the dark floor (→ optics/dark),
    # not SMU integration. If you must spend, spend on N_PM or optical collection, not NPLC.
    #
    # SETTLE TIMES ARE SECONDS, NOT MILLISECONDS. Measured on the bench 261013: after a
    # voltage step the current approaches its final value as a POWER LAW, |I-Iinf| ~ t^-1.3,
    # not an exponential. 1% takes 5-55 s and 0.1% takes 60-80 s, at EVERY step size from
    # 20 mV to 200 mV. The old 20 ms was wrong by a factor of 250-2750 and is the reason
    # every rising-side EQE this project produced is too high.
    #
    # Small steps do NOT settle faster -- that was the natural guess and the bench refused
    # it. Stepping 20 mV onto -0.80 V left the reading 4.5% high and still drifting at 80 s,
    # while stepping 200 mV onto the same point was flat from 60 s. Each point inherits the
    # unrelaxed tail of every earlier point, and a small step does not escape that, it just
    # makes the per-step excursion small compared with the memory.
    #
    # So the budget is the DWELL, not the step size, and the only free gains are:
    #   * sweep high current -> low current (already the default), which was the flattest
    #   * measure the bitval-0 reference back to back with the lit point, so both carry the
    #     same history and the subtraction cancels most of it
    # Spend the time budget on fewer points with a longer dwell, not more points.
    # -- Per-array calibration (--calibrate) ------------------------------
    # The settle time is a property of the ARRAY, not of the method: 8 s came from
    # Array38 (42.8 uF) and another array may need far more or far less. It does not
    # follow from C, because the settling is a charge relaxation and not an RC charge,
    # so it is measured directly rather than inferred.
    calibrate_pixels: int = 3  # pixels to characterise; the worst case sets the dwell
    calibrate_seconds: float = 40.0  # poll each bias this long. Must exceed the real
                                     # settle time or the answer is a lower bound, which
                                     # the calibration says so explicitly.
    calibrate_n_smu: int = 4  # SMU samples per poll; small, this is a time series
    calibrate_period_s: float = 0.05  # poll interval, and the floor on what the
                                      # calibration can resolve
    calibrate_target: float = 0.01  # how close to the final value counts as settled.
                                    # The settling is a power law, so the dwell scales as
                                    # target^(-1/1.3): 0.03 is ~2.3x faster than 0.01.
                                    # A looser target is a real, statable trade -- the
                                    # residual is common-mode across pixels either way.
    calibrate_approach_steps: int = 3  # sweep-sized steps taken before each measured one,
                                       # so the capture sees the step the sweep takes and
                                       # the range the sweep uses, not a 0.5 V jump
    calibrate_approach_s: float = 5.0  # dwell on each of those approach steps
    calibrate_margin: float = 1.5  # safety factor on the measured worst case, because
                                   # the calibration sees 3 biases and the sweep visits 30

    resume: bool = False  # with --out pointing at an existing run directory, skip the
                          # pixels it already finished and append. The aggregated CSV is
                          # flushed per pixel, so a run killed mid-sweep (Windows Update,
                          # a power cut) keeps every completed pixel and resumes from the
                          # next one. A pixel interrupted part way through its own sweep
                          # is re-measured whole, not stitched.
    settle_scale: float = 1.0  # multiplies every sweep settle. 1 = the measured times.
                               # --simulate sets it to 0 so a dry run is seconds, not
                               # 15 minutes. Lower it on real hardware only to measure what
                               # the shortcut costs -- it costs the rising side.
    vramp_perseg_effort: bool = True
    vramp_lowv_nplc: float = 2.0  # integer NPLC, so the mains line is nulled by integration
    vramp_lowv_n_pm: int = 32
    vramp_lowv_n_smu: int = 12
    vramp_lowv_settle_s: float = 8.0  # rising side: 1% needs 5-55 s. 8 s is the compromise
                                      # that keeps a 30-point sweep under ~10 min/pixel.
    vramp_lowv_limit_a: float = 1e-5  # the rising side never exceeds ~5 uA
    vramp_highv_nplc: float = 1.0  # was 0.5 -- not an integer, so it nulled no mains at all
    vramp_highv_n_pm: int = 8
    vramp_highv_n_smu: int = 8
    vramp_highv_settle_s: float = 2.0  # droop side: large currents, the relaxation is a much
                                       # smaller fraction of the reading
    vramp_highv_limit_a: float = 1e-3  # -2.6 V draws ~175 uA, far above the rising-side limit

    fastscan_limit_a: float = 1e-3  # compliance for the FAST SCAN, which is a separate
                                    # operating point from the sweep. At nvled_v_fastscan
                                    # (-1.35 V) a pixel draws ~30 uA, so the sweep's 10 uA
                                    # rising-side limit clamps EVERY pixel -- measured in
                                    # run 261014-0048, where all 1024 pixels read exactly
                                    # 10.00 uA and the light came out 2.8x low.

    # ── Leakage reference (the low-J correction) ────────────────────────
    # Both source modes read the TOTAL cathode current, so at low bias most of what the SMU
    # reports is not LED injection. Measured 261013 over 12 pixels in 4 quadrants:
    #   -0.95 V:  8% leakage      -0.85 V: 29%
    #   -0.90 V: 14%              -0.80 V: 58%
    # Without this correction the EQE appears to collapse 20x below J ~ 0.4; corrected, the
    # curve is smooth and monotone. This invalidates every rising-side EQE measured before.
    #
    # The reference is the SAME pixel at bitval 0: identical frame structure, identical
    # select state, zero drive. A BLANKED frame is NOT a valid reference -- it draws ~4x
    # MORE (943-956 nA against 236-254 nA), so lit-minus-blanked gives a negative LED
    # current. Measured both ways, twice.
    leak_reference: bool = True  # measure each sweep point lit and at LEAK_REF_BITVAL
    leak_ref_bitval: int = 0  # the reference drive level: addressed, but off
    leak_ref_settle_s: float = 2.0  # after switching drive level. Shorter than the sweep
                                    # settle: the bias does not move, only the drive, and
                                    # the point of measuring back to back is a shared history.

    # ── Dead-pixel threshold (fast scan) ────────────────────────────────
    nvled_v_fastscan: float = -1.35  # single fast-scan bias — does DOUBLE duty. At −1.35 V the pixels
                                     # sit at J≈25 A/cm², which is dead on the (broad, flat) EQE peak
                                     # plateau: EQE is within ~1% of its peak across J≈8–30, so the
                                     # fast-scan EQE=P/I here ≈ each pixel's PEAK EQE. That turns the
                                     # fast scan into a full-coverage peak-IQE proxy for all 65 536
                                     # pixels (calibrated to the sweep in analysis), while still emitting
                                     # ~22 nW for the alive/dead classification — ~1000× the
                                     # MEASURED optical floor (3σ = 23 pW at n_pm 32; see the
                                     # optfloor run of 261013, σ ∝ N^-0.499 from n_pm 4 to 512).
                                     # J≈25 keeps ~5 A/cm² of margin before the droop starts (~J30). The
                                     # current there (~25 µA) is high enough for a fast (NPLC 0.01) read.
                                     # (Was −2.6 V / J≈95 — the droop side at only ~78% of peak.)
    dead_threshold_frac: float = 0.20  # a pixel is alive if P ≥ this × the "typical lit" level
    fastscan_alive_pctl: int = 90  # "typical lit" level = this percentile of the fast-scan power.
                                     # NOT the median: the median collapses to the noise floor on a
                                     # majority-DEAD ROI (then ~everything is mislabelled alive — e.g.
                                     # quadrant 260703-1307 was 57% dark, median 0.08 nW → 84% "alive").
                                     # A high percentile tracks the lit population as long as the lit
                                     # fraction exceeds (100−pctl)% = 10%.

    pixel_area_um2: float = 100.0
    smu_display_off: bool = False

    # ── Derived / helpers ───────────────────────────────────────────────

    @property
    def photon_energy_j(self) -> float:
        """Energy of one emitted photon (J), from ``pm_wavelength``."""
        return H_PLANCK * C_LIGHT / (self.pm_wavelength * 1e-9)

    def resolve_roi(self) -> tuple[int, int, int, int]:
        """``roi`` as (x1, y1, x2, y2), resolving a named preset."""
        if isinstance(self.roi, (tuple, list)) and len(self.roi) == 4:
            return tuple(int(v) for v in self.roi)          # type: ignore[return-value]
        key = str(self.roi).strip().upper()
        if key not in NAMED_ROIS:
            raise ValueError(f"unknown ROI {self.roi!r}; use (x1,y1,x2,y2) or one of {sorted(NAMED_ROIS)}")
        return NAMED_ROIS[key]

    def validate(self) -> IQEConfig:
        """Raise ``ValueError`` on a configuration that cannot be measured."""
        x1, y1, x2, y2 = self.resolve_roi()
        if not all(0 <= v <= 511 for v in (x1, y1, x2, y2)):
            raise ValueError(f"roi {(x1, y1, x2, y2)} outside the 512x512 array")
        for name, allowed in (("source_mode", {"voltage", "current"}),
                              ("sweep_select", {"blocks", "spatial", "stride"}),
                              ("fine_sweep_mode", {"voltage", "current"}),
                              ("sweep_direction", {"low_to_high", "high_to_low"}),
                              ("sweep_autozero", {"AUTO", "ONCE", "OFF"})):
            if getattr(self, name) not in allowed:
                raise ValueError(f"{name}={getattr(self, name)!r}; expected one of {sorted(allowed)}")
        if not self.bitvals or not all(0 <= b <= 15 for b in self.bitvals):
            raise ValueError(f"bitvals={self.bitvals!r}; expected a non-empty tuple of 0..15")
        for name in ("nvled_range_i", "probe_range_i", "current_source_range_i"):
            if getattr(self, name) not in SMU_I_RANGES:
                raise ValueError(f"{name}={getattr(self, name)}; the 2602B only has "
                                 f"the ranges {SMU_I_RANGES}")
        if not 0 < self.pm_wavelength < 1e5:
            raise ValueError(f"pm_wavelength={self.pm_wavelength} nm out of range")
        if self.n_fine < 2 or self.n_coarse < 2:
            raise ValueError("n_fine and n_coarse must be >= 2")
        for name in ("vramp_lowv_settle_s", "vramp_highv_settle_s", "leak_ref_settle_s",
                     "settle_scale"):
            if getattr(self, name) < 0:
                raise ValueError(f"{name} must be >= 0")
        if self.highc_b:
            # In High-C the LIMIT selects the measure range, so a limit too coarse for the
            # segment silently costs resolution rather than failing.
            for name in ("vramp_lowv_limit_a", "vramp_highv_limit_a"):
                if getattr(self, name) > 1e-3:
                    raise ValueError(
                        f"{name}={getattr(self, name)} A with highc_b=True: in High-C the limit "
                        "selects the measure range, so this resolves no better than ~milliamps.")
        if self.leak_reference and self.leak_ref_bitval in self.bitvals:
            raise ValueError(
                f"leak_ref_bitval={self.leak_ref_bitval} is also a swept bitval: the reference "
                "must be a drive level the sweep does not measure as signal.")
        if not 0.0 <= self.vramp_low_frac <= 1.0:
            raise ValueError(f"vramp_low_frac={self.vramp_low_frac} must be in [0, 1]")
        # Measured 261013: in High-C the LIMIT selects the current measure range, so a
        # 0.5 A limit leaves milliamp resolution and a microamp reading is pure noise.
        # The same run showed High-C is what makes low-J readings correct at all, so
        # enabling it with the stock compliance is the one configuration that looks
        # right and measures nothing.
        if self.highc_b and self.nvled_compliance > 1e-3:
            raise ValueError(
                f"highc_b=True with nvled_compliance={self.nvled_compliance} A: in High-C "
                "the limit selects the measure range, so this resolves no better than "
                "~milliamps. Use <=1e-3 (1e-3 still moves the array's ~95 uC in under a "
                "millisecond); go lower for sub-microamp points, at the cost of slower "
                "charging")
        if self.highc_b and self.perpoint_range:
            raise ValueError("highc_b and perpoint_range are incompatible: High-C derives the "
                             "current range from the limit, so per-point range switching "
                             "conflicts with it")
        if self.min_pixel_separation < 1:
            raise ValueError("min_pixel_separation must be >= 1 (1 disables the guard)")
        if not 0 <= self.fastscan_alive_pctl <= 100:
            raise ValueError(f"fastscan_alive_pctl={self.fastscan_alive_pctl} must be in [0, 100]")
        if self.pixel_area_um2 <= 0:
            raise ValueError("pixel_area_um2 must be > 0")
        return self

    # ── Serialisation ───────────────────────────────────────────────────

    def to_dict(self) -> dict[str, Any]:
        """Plain dict of every field — this is what is written to the run's config JSON."""
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> IQEConfig:
        """Build from a mapping, rejecting unknown keys (a typo must not pass silently).

        Values are coerced to the field's declared type, so a JSON/YAML/CLI string
        like ``"1e-6"`` becomes a float instead of quietly staying a string and
        breaking arithmetic deep inside a running measurement.
        """
        defaults = {f.name: f.default for f in dataclasses.fields(cls)}
        unknown = sorted(set(data) - set(defaults))
        if unknown:
            raise ValueError(f"unknown config key(s): {', '.join(unknown)}")
        clean = {}
        for key, value in data.items():
            want = type(defaults[key])
            try:
                if value is None:
                    clean[key] = None          # e.g. vramp_knee_v: null disables the split
                elif want is tuple and isinstance(value, (list, tuple)):
                    clean[key] = tuple(value)
                elif want is bool and not isinstance(value, bool):
                    raise TypeError("expected true/false")
                elif want in (int, float, str) and not isinstance(value, want):
                    clean[key] = want(value)
                else:
                    clean[key] = value
            except (TypeError, ValueError) as exc:
                raise ValueError(f"{key}={value!r}: expected {want.__name__} ({exc})") from None
        return cls(**clean).validate()

    @classmethod
    def from_file(cls, path: str | Path) -> IQEConfig:
        """Load a ``.json``, ``.yaml`` or ``.yml`` config file."""
        path = Path(path)
        text = path.read_text(encoding="utf-8")
        if path.suffix.lower() in (".yaml", ".yml"):
            import yaml                                  # only needed for YAML configs
            data = yaml.safe_load(text)
        else:
            data = json.loads(text)
        if not isinstance(data, dict):
            raise ValueError(f"{path}: expected a mapping at the top level")
        return cls.from_dict(data)

    def write_file(self, path: str | Path) -> Path:
        """Write this config as JSON (or YAML, by extension).  Returns the path."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.suffix.lower() in (".yaml", ".yml"):
            import yaml
            path.write_text(yaml.safe_dump(self.to_dict(), sort_keys=False), encoding="utf-8")
        else:
            path.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")
        return path
