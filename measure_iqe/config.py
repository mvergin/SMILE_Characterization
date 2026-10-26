"""Configuration for the IQE acquisition.

Every tunable of the measurement lives on :class:`IQEConfig` as a field with a
safe default.  The defaults are the values that were proven on Array42 — change
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
    sample_name: str = "Array42_IQE"  # which array is mounted. Set it with --array NAME, which
                                      # also applies calibrations/NAME.yaml. A hardware run that
                                      # leaves this default is REFUSED: it is one real array, so
                                      # any other array would be recorded as it. --simulate is exempt.
    base_dir: str = "results"          # output root; a timestamped run dir is created inside
    analyze_after: bool = True         # run analysis/analyze_array.py on the run directory when
                                       # the run completes (CLI and console). Not after a run
                                       # that was stopped, --calibrate or --warmup-diag. Needs
                                       # the analysis extra (scipy); without it, says so and skips.

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
    bias_major_precondition: bool = False
    # Hold every selected pixel at warmup_nvled_v for warmup_ms ONCE, before the first
    # bias. Costs warmup_ms per pixel (~24 s for 16) and nothing per point.
    #
    # Why: pixel_major conditions every pixel immediately before that pixel's own ramp
    # and _sweep_bias_major conditions none, which is the last standing explanation for
    # the 3.2 % current deficit bias_major shows on the rising side (see
    # pixel_switch_settle_s). Conditioning per pixel per BIAS is not an option -- that
    # is a 1.65 V step into 42.8 uF at every point, i.e. pixel_major with extra steps.
    # Once per run is the only affordable version, and it tests the open half of the
    # question: does the conditioning persist?
    #
    # Two things to know before reading a result from this:
    #   * it drives at vramp_highv_limit_a, not at whatever limit happens to be set.
    #     pixel_major's own warm-up inherits the limit left by the previous pixel's
    #     FAINTEST point, so under high_to_low it is probably current-clamped well
    #     below what -2.6 V would draw. This one is not. They are not identical soaks.
    #   * bias_major already drives every pixel at -2.6 V for ~6 s at its first bias,
    #     harder than that clamped warm-up does. If recency rather than magnitude is
    #     what matters, this flag will not help and the hypothesis is wrong.
    sweep_rotate: int = 0  # rotate the MEASUREMENT ORDER by N, same pixels. Under
                           # bias_major the pixel in slot 0 holds the frame through the
                           # whole bias dwell and reads ~3% higher in current than the
                           # other fifteen -- but it has been the same pixel, (35,36),
                           # in every run so far, so "the slot is clean" and "that pixel
                           # is clean" are indistinguishable. Rotating separates them in
                           # one 33 min run: if the clean pixel follows the slot, loop
                           # position is real; if it follows the pixel, slot 0 was never
                           # special. Section 17, open question 13.
    sweep_fraction: float = 0.10  # spatial mode: fraction of ALIVE pixels to sweep
    min_sweep_pixels: int = 6  # spatial mode: floor on the count (so tiny ROIs still map)
    nth_pixel_sweep: int = 5  # stride mode: sweep every Nth alive pixel
    max_sweep_pixels: int = 40  # hard cap (spatial/stride only; blocks controls its own count)
    fastscan_bitval: int = 15  # bitval for the alive/dead fast scan (brightest)

    # ── Pooled clusters: K pixels lit together, swept as one target ─────
    # OFF by default and deliberately so. A pooled row is not a pixel: it reports the
    # SUM of its members' light and the SUM of their current, so per-pixel quantities
    # drawn from it are averages, not measurements. Turning it on is a choice about
    # what the run is for, never a default the operator did not make.
    #
    # Why it exists: open question 1, the optical floor. The PM400 dark reading wanders
    # 75 pW under a 210 pW signal and faint-end power scatters 8.5-8.9 % under BOTH
    # sweep methods, so the suspect is the meter. K members emit K x the light while
    # the drift stays put. set_pixels writes them in ONE frame instruction, so a pooled
    # point costs exactly what a single point costs -- K pixels per dwell, not one.
    #
    # Pooled EQE = sum(P)/sum(I) is a CURRENT-WEIGHTED mean of the members, which is
    # arguably the right array figure. It is not the mean of the members' EQEs, and the
    # two differ whenever brightness and current are correlated across members.
    #
    # The intended shape is both at once: N single pixels and M pooled groups in the
    # same run. The singles are per-pixel ground truth measured under the same lamp,
    # the same thermal history and the same instrument state, so they GROUND the pooled
    # numbers -- the pooled faint end can be checked against the singles wherever the
    # singles are still above their own noise. Measured on run 261015-0015 the array's
    # alive-pixel brightness spread is 16.4 % (52.21 +- 8.55 nW over 261 441 pixels),
    # which is what a pooled reading has to be interpreted against.
    # Pooled groups are formed INSIDE one block, from that block's own draw. An
    # earlier version drew them from a second array-wide sample: measured, 291 of 300
    # member pairs then landed in different blocks, a median of 152 px apart, so a
    # group averaged over a whole quadrant and threw away the +3.7 % spatial gradient
    # section 16 measures. Block-local keeps the map's resolution at block scale and
    # grounds each group against singles that share its row drivers.
    pool_groups: tuple[int, ...] = ()
                            # per-block pooled group sizes. () = OFF, the default.
                            # (4,) adds one 4-pixel group to every block; (2, 4, 8)
                            # adds three groups of different K to every block, which
                            # gives the K-dependence across the array for free.
                            # Two measured ceilings on any single entry. DETECTION: a
                            # dead or dim member is a 1/K deficit against a sum whose
                            # own spread is 16.4%/sqrt(K) -- 3.0 sigma at K=4, 1.5 at
                            # K=16. COMPLIANCE, harder: K members draw K x the current
                            # and pick_limit's ladder stops at vramp_highv_limit_a, so
                            # at the measured 117.7 uA per pixel at -2.60 V a 1 mA
                            # ceiling clamps at K=8. The selection estimates the second
                            # from the fast scan and says so before the sweep starts.
    pool_include_singles: bool = True
                            # each group takes one of its OWN block's single pixels as
                            # a member. That makes the comparison PAIRED: P(group) -
                            # P(single) is the rest of the group measured on the same
                            # silicon, so an additivity loss is separable from
                            # pixel-to-pixel variation. False keeps the two sets
                            # disjoint, which is the better independent sample but
                            # cannot separate those two. The single is never the
                            # group's FIRST member, so X,Y stays unique per row.
    pool_pm_range_below_v: float | None = None
                            # V. At or below this bias a POOLED target is read on
                            # pool_pm_range instead of pm_range. None = one range for
                            # the whole sweep, which is the default and what every run
                            # before 20 October did.
                            # Why it exists: the PM400 range is fixed (autorange off on
                            # purpose) and the two ends of a pooled sweep want
                            # different scales. MEASURED, runs 261019-1056/-1212: the
                            # floor goes as range^0.74, so the coarse range costs 5.45x
                            # at the faint end where pooling pays -- and MEASURED, runs
                            # 261019-1829/-1833: the gain across the boundary agrees to
                            # 0.150 % (0.56 sigma) and the 1154 pW dark offset cancels
                            # in the lit-minus-off subtraction. So the fine range
                            # belongs on the faint segment and the coarse one at the
                            # bright end, with the switch at a FIXED bias rather than
                            # wherever an autoranging meter decided.
                            # Set it just above where the brightest group leaves the
                            # fine range: at K=4 that is about -1.6 V (section 26.18).
    pm_range_settle_s: float = 0.5
                            # s after a PM400 range change before reading. The meter
                            # switches a transimpedance gain, and nothing here has
                            # measured how long that takes -- 0.5 s is a guess with
                            # margin, and the cross-check below is what would show it
                            # was too short (the two ranges would disagree).
    pm_range_check_targets: int = 8
                            # how many pooled targets are measured at the boundary bias
                            # on BOTH ranges, to verify the 0.150 % in situ. 0 skips
                            # it. Each one costs two extra points, so 8 is about two
                            # minutes and the check is per run rather than per campaign.
    fastscan_pool: int = 0  # K for the fast-scan group-test SCREEN. 0 = off, and off
                            # is the default: the ordinary scan measures every pixel.
                            # With K > 1 the scan lights K pixels at a time, flags the
                            # dimmest fastscan_pool_rescan_frac of groups against a
                            # LOCAL median, and rescans only those members one by one.
                            # Replayed on the real 262 144-pixel scan of run
                            # 261016-2252 (1116 dead, 99.574 % yield) with the rule
                            # this code ships: at K=4 it finds 91.8 % of the dead
                            # pixels for 3.46x less bench time, 2.15 h -> 37 min. K=2
                            # finds 97.5 % for 1.94x. K=8 and K=16 buy 5.71x and 8.89x
                            # but their physical cut selects more groups than the 5 %
                            # cap allows, so recall falls to 75 % and 53 % and the run
                            # says the cap is binding. It is a SCREEN, not a census --
                            # if every dead pixel must be named, leave this at 0.
    fastscan_pool_rescan_frac: float = 0.05
                            # CAP on the fraction of groups rescanned pixel by pixel.
                            # It is not the rule: a group is flagged when its detrended
                            # reading is at or below 1 - 0.5/K, half a member's worth
                            # of deficit, which is physical. The cap only bounds the
                            # bench time, and the run says loudly when it binds --
                            # because a screen that silently ignores groups it knows
                            # are suspicious is the one failure it must not have. At
                            # K=4 on the measured array the cut selects 2545 of 65 536
                            # groups, under the 3277 this cap allows, so it does not
                            # bind.
    fastscan_pool_detrend: int = 15
                            # window, in GROUPS, of the local median the flag is taken
                            # against. The array has a real +3.7 % spatial gradient and
                            # a row-driver signature (section 16), so a GLOBAL median
                            # flags one corner wholesale: measured, detrending lifts
                            # K=4 recall at a 5 % budget from 85.5 % to 93.5 %.
    pool_pm_range: float = 0.0
                            # W, the PM400 range for pooled targets. 0 = use pm_range,
                            # which is almost always right -- read on.
                            # MEASURED, run 261019-1031: pm_range=1e-7 realises as
                            # 470 nW on the S171C, so pm_range is a REQUEST and not a
                            # scale. A single pixel's 257 nW maximum at -2.6 V (run
                            # 261016-2252) sits at 55 % of it, which is why nothing
                            # ever clipped. A K=4 group reaches ~591 nW there and a
                            # K=8 group ~1182 nW, both over.
                            # Raising this is usually the WRONG fix. The additive
                            # optical floor is 30.4 pW on the 470 nW range, 0.0065 %
                            # of full scale; if that fraction is what the front end
                            # fixes, a decade coarser range costs a 10x worse floor
                            # and a K=8 group's faint end would scatter ~9 % -- worse
                            # than one pixel on the fine range, and the whole point of
                            # pooling is gone. Pool over the FAINT segment and keep
                            # the fine range, or sweep K=4 and let the brightest
                            # points over-range and be dropped. Whether the floor
                            # really scales with full scale is open question 22.

    # ── Source mode ─────────────────────────────────────────────────────
    # "voltage": ch B sources voltage, adaptive current-targeted sweep. PROVEN on
    #            Array42 — the 10 mA voltage sweep captured the IQE peak (J≈6,
    #            ~59%). USE THIS.
    # "current": ch B sources (sinks) current, measures voltage. Tried on Array42
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
    sweep_voltages: tuple[float, ...] = ()  # EXPLICIT voltage-sweep biases (V, <= 0).
                            # Empty = the ramp from nvled_v_coarse_start/stop and n_fine.
                            # Set = exactly these, in sweep_direction order, and the start,
                            # stop, n_fine and knee split are ignored for the grid.
    n_coarse: int = 26  # coarse pre-sweep points
    coarse_settle_ms: float = 6.0  # settle before each coarse pre-sweep point (ms)
    coarse_n_smu: int = 4  # SMU readings per coarse pre-sweep point
    # Where the fit actually needs data: across 140 per-pixel fits the device's IQE peak
    # sits at J ≈ 5 A/cm² (≈5 µA) and the trustworthy fits only reach down to J ≈ 1 (≈1 µA)
    # on the rising side. So target J ≈ 1–10, NOT sub-µA: below ~1 µA the data is unneeded
    # AND that is exactly where the fine measure range takes ~seconds to settle. Starting at
    # ~1 µA keeps every point on a fast-settling range and puts the points where the peak is.
    low_current_target_a: float = 1e-6  # lowest LED current the sweep targets (~J 1, the rise foot)
    high_current_frac: float = 0.97  # the fine grid's top target, as a fraction of the coarse pre-sweep's max current
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
    # Sweep direction. "high_to_low" visits the brightest bias first and is the default;
    # "low_to_high" visits the faintest first.
    #
    # The old comment here called low_to_high "recommended" while the default said the
    # opposite, and section 4 said the opposite again. Section 4 wins on evidence: the
    # 200 mV step from the HIGH-current side gave the flattest approach, and a point
    # approached from below overshoots and decays for a whole record without arriving.
    #
    # Two things to know before running the A/B:
    #   * dead_abort has to be off. Both dead-pixel tests judge at bias index 0, which
    #     low_to_high makes the faintest, and validate() refuses the combination.
    #   * no low_to_high sweep exists on disk since June 2026 (Array38, before the
    #     leakage reference and before the settle times were measured). The direction has
    #     never been compared under the current method. Section 23.
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
    warmup_diag_nplc: float = 1.0  # SMU integration (NPLC) for the warm-up diagnostic
    warmup_diag_n_smu: int = 8  # SMU readings per warm-up diagnostic sample
    warmup_diag_n_pm: int = 4  # PM400 readings per warm-up diagnostic sample
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
    lowj_nplc: float = 2.0  # low-J zone (below adaptive_threshold_a): SMU integration (NPLC)
    lowj_n_pm: int = 8  # low-J zone: PM400 readings per point
    lowj_n_smu: int = 16  # low J: integrate the CURRENT harder
    highj_nplc: float = 0.5  # high-J zone (above adaptive_threshold_a): SMU integration (NPLC)
    highj_n_pm: int = 4  # high-J zone: PM400 readings per point
    highj_n_smu: int = 8  # high J: fast
    lowj_settle_ms: float = 200.0  # low-J zone: settle per point (ms); the fine range settles slower
    highj_settle_ms: float = 15.0  # fine low-J range needs more settle time

    # ── Instruments ─────────────────────────────────────────────────────
    pm400_addr: str = "USB0::0x1313::0x8075::P5006526::INSTR"  # VISA address of the Thorlabs PM400 power meter
    smu_addr: str = "USB0::0x05E6::0x2602::4649721::INSTR"  # VISA address of the Keithley 2602B SMU
    # ── PM400 ───────────────────────────────────────────────────────────
    pm_wavelength: float = 440.0  # wavelength (nm) the PM400 uses for its responsivity; also sets the photon energy
    pm_range: float = 1e-7  # W. With pm_autorange off, THE range, fixed. With it on, only
                            # the starting point; the run picks its own ranges.
    pm_autorange: bool = True  # pick the PM400 range per bias (sweep) and per pass (fast
                            # scan), like pick_limit does for the SMU: the smallest
                            # realised range >= pm_range_headroom x the predicted power,
                            # stepping up and re-reading when a point over-ranges. NOT the
                            # meter's own autorange, which switches mid-burst. The first
                            # burst after every change is discarded (set_pm_range). False
                            # = one fixed range, every run before 2026-10-25.
    pm_range_headroom: float = 1.5  # predicted power x this must fit the range
    sweep_read_vled: bool = True  # read ch A (VLED, the anode supply) in every sweep burst.
                            # False reads ch B (the LED current) only: one integration per
                            # sample instead of two, so NPLC 1 costs 20 ms a sample, not
                            # 41. ch A is a supply-rail monitor (RUN_FORMAT.md); its
                            # columns are then NaN.
    leak_ref_skip_above_v: float | None = None  # V. No leakage reference at biases above
                            # this (toward 0 V), where the LED is off: the lit reading is
                            # then the leakage itself and an EQE means nothing. None = a
                            # reference wherever leak_reference asks for one.
    pm_overrange_frac: float = 0.95  # a reading at this fraction of full scale is over
    fastscan_pm_probe_n: int = 256  # pixels read on the widest range before each
                            # fast-scan bias pass; the brightest x headroom sets the range
    pm_averaging: int = 3  # internal averaging per reading

    # ── VLED (ch A) — constant LED anode drive ──────────────────────────
    vled_voltage: float = 1.8  # ch A: constant VLED anode supply (V)
    vled_compliance: float = 1.0  # ch A current limit (A)
    vled_range_i: float = 10e-3  # ch A current measure range (A)
    # ── NVLED (ch B) — voltage-mode current range ───────────────────────
    nvled_compliance: float = 1e-5  # current compliance (voltage mode). In High-C this ALSO
                                     # selects the measure range, so it sets the noise floor:
                                     # 10 uA gives ~9 nA/sample at NPLC 0.01, ~1 nA at NPLC 1.
                                     # Measured 261013. The sweep overrides it per segment
                                     # (VRAMP_*_LIMIT_A) because -2.6 V needs ~175 uA.
    nvled_range_i: float | None = None  # ch B measure range WITHOUT High-C. None (the
                                     # default since 23 Oct 2026): the range follows the
                                     # compliance limit, exactly as High-C does -- the same
                                     # ladder (pick_limit), the same ranges, in either mode.
                                     # A number fixes the range for every point instead, as
                                     # every run before 14 Oct did: the legacy config sets
                                     # 0.01. Ignored under High-C, where the limit IS the range.
                                     # (Was 10e-3 always: a 400 nA reading then sat at
                                     # 0.004 % of full scale.)
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
    fastscan_remeasure_n: int = 100  # after the last pixel of each fast-scan bias, measure
                                     # the FIRST N again in the same loop, with the same
                                     # blank, settle and burst. Written apart, to
                                     # remeasure_bitval=<b>[_v=<V>].csv and
                                     # raw_data/FastScan_remeasure.csv, so no reader of
                                     # the fast scan sees a pixel twice. The pair says how
                                     # far a pixel drifted over the scan. 0 = off.
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
    # -- Settling traces -------------------------------------------------
    # The sweep waits `vramp_*_settle_s` and then integrates for ~0.5 s: at a 65 s dwell
    # that is 99.3% dead time, and the whole transient is currently discarded. Polling
    # through a wait we are already paying for costs NO bench time and no accuracy -- the
    # reported value is still the burst taken at the end -- and it yields, for every
    # point of every pixel, the data needed to answer two open questions offline:
    #   * can the settled value be EXTRAPOLATED from the first few seconds, so the dwell
    #     can be cut? (fit I_inf on the basis {1, t^-n}, n from the calibration)
    #   * does EQE = P/I settle FASTER than either, because both relax with the same
    #     carrier population? If so the dwell only has to fix the J axis.
    # Neither can be answered from existing data: the current is traced, the power is not.
    trace_settling: bool = True
    trace_settling_period_s: float = 0.25  # poll interval during the dwell
    trace_settling_n_pm: int = 2  # PM reads per poll; enough for P(t) once fitted
    trace_settling_n_smu: int = 2  # SMU samples per poll
    trace_settling_min_s: float = 2.0  # skip for dwells shorter than this: nothing to see

    # -- Per-array calibration (--calibrate) ------------------------------
    # The settle time is a property of the ARRAY, not of the method: 8 s came from
    # Array38 (42.8 uF) and another array may need far more or far less. It does not
    # follow from C, because the settling is a charge relaxation and not an RC charge,
    # so it is measured directly rather than inferred.
    # A calibration that samples one corner sets the whole array's dwell from it.
    # Array38's first calibration ran on roi=(32,32,63,63) and picked (35,34), (32,46),
    # (35,54) -- three neighbours inside TL, which the capacitance run had already
    # singled out as the one slow quadrant (108 s against ~1 s in TR, BL and BR). It
    # recommended 15.23 s for the array on that evidence. So sample every quadrant, the
    # way characterize_capacitance does, and from a small window in each rather than a
    # full-ROI fast scan that costs two hours before the first capture.
    calibrate_per_quadrant: int = 1   # pixels per quadrant. 0 = the old single-ROI path
    calibrate_quadrant_window: int = 24  # window scanned at each quadrant centre, px
    calibrate_pixels: int = 3  # pixels to characterise; the worst case sets the dwell
                                     # (single-ROI path only, i.e. calibrate_per_quadrant=0)
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

    calibration_source: str = ""
    # Set by --array/--calibration: the file whose `recommended` block was applied on
    # top of the defaults. Kept in the config (rather than popped) for two reasons: the
    # saved *_config.json then says WHICH calibration produced the run, and the run
    # directory gets a copy of that file. A sweep that believes it is calibrated and
    # cannot say by what is not reproducible.
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
    sweep_order: str = "bias_major"   # "bias_major" (default since 22 Oct 2026) | "pixel_major"
    # WHICH LOOP IS OUTSIDE. This is the single biggest lever on run time.
    #
    #   pixel_major  for pixel: for bias: settle, measure      <- the original default
    #   bias_major   for bias:  settle once; for pixel: measure
    #
    # The dwell is paid for the BIAS change, not the pixel change: the array carries one
    # global 42.8 uF charged by the common cathode, and the frame only selects which
    # row/column is driven. Measured directly (run 261014-2151, --what switch): holding
    # a bias for 300 s and then blanking the frame and lighting a different pixel moves
    # the reading by -0.28% at -1.00 V and -0.87% at -0.85 V, both gone inside a second,
    # and by less than the noise at -0.80 V. Against that, pixel_major's 54 s dwell
    # still leaves ~3.8% at the faintest point (run 261014-1501, 600 s asymptote).
    #
    # So bias_major is both FASTER and MORE ACCURATE at the faint end. For 100 pixels x
    # 16 points the dwell total goes from 15.4 h to 553 s.
    #
    # What it costs:
    #   * the fine voltage list is built ONCE, from the first live pixel, instead of per
    #     pixel. Fine for a voltage ramp (the biases are fixed); not supported for
    #     fine_sweep_mode="current", where each pixel needs its own targets.
    #   * an interrupted run leaves every pixel partially swept instead of some pixels
    #     fully swept, so --resume skips completed BIASES, not completed pixels.
    #   * the per-pixel conditioning dwell is gone. With sweep_direction=high_to_low the
    #     first bias visited IS the brightest, so every pixel is conditioned there anyway.
    #     !! UNVERIFIED, and it is the last standing explanation for the 3.2% deficit
    #     this loop shows against pixel_major (see pixel_switch_settle_s). pixel_major
    #     gives each pixel 1.5s at -2.6V IMMEDIATELY before its own ramp; here a pixel
    #     gets ~6s at -2.6V and then sits dark through most of the next 15 biases. The
    #     cheap test is to condition all selected pixels once before the first bias
    #     (+1.5s each, ~24s for 16) and see whether the deficit collapses. Section 17.
    pixel_switch_settle_s: float = 2.0  # bias_major: the wait after a frame change,
                                       # before measuring the next pixel at a held bias.
                                       # This is the ONLY per-point wait in bias_major,
                                       # so it sets the run time: n_px x n_points x this.
                                       #
                                       # !! DO NOT RAISE THIS. It was measured, twice,
                                       # and it is not the knob it looks like.
                                       #
                                       # bias_major reads ~1 % low in current on the
                                       # FAINT side against pixel_major on the same
                                       # pixels, and nothing above -1.8 V. Runs
                                       # 261015-1421 / -1529, back to back, 68 min
                                       # apart:
                                       #
                                       #   order 0    +0.05 % I
                                       #   order >0   -1.04 % I,  -0.84 % at matched I
                                       #   offset     1.05 mV
                                       #
                                       # It reads ~3 % against a reference from the day
                                       # BEFORE. Two thirds of that was elapsed time,
                                       # not loop order -- see section 18. Cross-run
                                       # drift (~2 %/day) is now the project's largest
                                       # error, so any absolute number needs a SAME-DAY
                                       # pixel_major anchor. Four pixels, 1.8 h.
                                       #
                                       # The deficit looked like an unfinished switch
                                       # transient, so
                                       # run 261015-0958 repeated it at 20 s -- ten times
                                       # the wait, 3.5x the run time:
                                       #
                                       #              2 s (261015-0822)  20 s (0958)
                                       #   order 0      -0.70 % I          -0.21 % I
                                       #   order >0     -2.99 % I          -3.25 % I
                                       #
                                       # No reduction. Three checks agree there is no
                                       # transient: flat against loop position once each
                                       # pixel's own mean is removed, +0.001 %/100 s
                                       # against time-since-the-bias-step over a 15x
                                       # lever, 0.00 %/h against elapsed run time.
                                       #
                                       # The order-0 row above is ONE PIXEL, not a slot.
                                       # Run 261015-1225 rotated the measurement order
                                       # (sweep_rotate=1): the clean pixel followed the
                                       # pixel to slot 15 (-0.22 %), and the pixel moved
                                       # into slot 0 came out worst in the run
                                       # (-3.84 %). There is no privileged position.
                                       #
                                       # What is left, back to back, is ~1 mV and it
                                       # does not decay. It grows as the bias falls
                                       # (-0.2 % at -2.6 V, -1.0 % at -1.0 V, -2.3 % at
                                       # -0.80 V where the leakage floor takes over), so
                                       # it is additive rather than a shifted axis. At
                                       # matched CURRENT the orders agree to 0.84 %, so
                                       # EQE(J), the ABC fit and IQE are fine and EQE(V)
                                       # is ~1 % low on the faint side. Report against J.
                                       #
                                       # Cost of getting this wrong: at 100 px x 30
                                       # points, one second here is 50 min of run time,
                                       # so 2 -> 20 s is 15 hours for nothing.
                                       #
                                       # The conditioning candidate is dead too: run
                                       # 261015-1321 held all 16 pixels at -2.6 V before
                                       # the first bias (bias_major_precondition) and
                                       # read -3.02 % against the unconditioned
                                       # -2.99 %. Conditioning is
                                       # bought with current, not with waiting -- which
                                       # is exactly why 20 s changed nothing. See
                                       # section 17.
    vramp_perseg_effort: bool = True  # true: use the vramp_lowv_* / vramp_highv_* effort per segment (split at vramp_knee_v); false: sweep_nplc, n_pm_samples, n_smu_samples, settle_ms everywhere
    vramp_lowv_nplc: float = 2.0  # integer NPLC, so the mains line is nulled by integration
    vramp_lowv_n_pm: int = 32  # rising segment (above vramp_knee_v): PM400 readings per point
    vramp_lowv_n_smu: int = 12  # rising segment: SMU readings per point
    vramp_lowv_settle_s: float = 8.0  # rising side: 1% needs 5-55 s. 8 s is the compromise
                                      # that keeps a 30-point sweep under ~10 min/pixel.
    vramp_lowv_limit_a: float = 1e-5  # the rising side never exceeds ~5 uA
    vramp_leak_limit_a: float = 1e-6  # the ladder's floor from vramp_leak_above_v to 0 V,
                                      # where the LED is off and the reading is the
                                      # ~100-500 nA leak: a 1 uA range resolves it 10x
                                      # better than 10 uA. WITHOUT High-C only -- in
                                      # High-C the limit is also the charging current,
                                      # and 1 uA charges 45.7 uF at 22 mV/s.
    vramp_leak_above_v: float = -0.80  # biases at or above this (toward 0 V) get the
                                       # vramp_leak_limit_a floor
    vramp_highv_nplc: float = 1.0  # was 0.5 -- not an integer, so it nulled no mains at all
    vramp_highv_n_pm: int = 8  # droop segment (below vramp_knee_v): PM400 readings per point
    vramp_highv_n_smu: int = 8  # droop segment: SMU readings per point
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
    leak_ref_prepass_reads: int = 0  # reads per pixel in the reference PRE-PASS, which
                             # keeps their minimum. 0 = leak_ref_tries, as before. Set it
                             # apart to keep the per-point retry (cheap: only failing
                             # points repeat) without multiplying the pre-pass.
    leak_ref_tries: int = 3  # frame writes allowed for the bitval-0 reference.
                             # The FPGA does not always write the frame correctly, so
                             # the reference is verified optically and re-written when
                             # it fails. Evidence that it is the WRITE and not the
                             # pixel: (56,51) referenced clean in runs 261014-1616 and
                             # 261015-0958 and contaminated in four others, and single
                             # points flip inside one sweep -- one clean bias in an
                             # otherwise contaminated run, and a contiguous block of
                             # four bad biases in an otherwise clean one. A stuck
                             # transistor cannot do that. Section 19.
                             #
                             # Cost is nil: ~1% of points fail, and only those retry.
                             # Set to 1 to reproduce runs taken before this existed.
                             # Requires leak_ref_bitval == 0: the gate is optical, and
                             # bitval 1 emits 11.6% of bitval 15 (section 20).
    leak_ref_max_frac: float = 0.25  # retry while PM400_POWER_REF / PM400_POWER_MEAN
                             # exceeds this. The gate is OPTICAL on purpose: a healthy
                             # backplane conducts 58% of the reading at -0.80 V but does
                             # not emit, so the current cannot tell a failed write from
                             # ordinary leakage and the light can. Over the 3000 points
                             # of run 261015-0015 the healthy pixels sit at 2e-4 and the
                             # failures at 0.38-1.03, with nothing clean above this gate
                             # at any bias -- a 250x margin, so it needs no tuning.
    leak_ref_settle_s: float = 2.0  # after switching drive level. Shorter than the sweep
                                    # settle: the bias does not move, only the drive, and
                                    # the point of measuring back to back is a shared history.

    # Per-point references are the single largest avoidable cost in a sweep: 2 s of
    # settle plus a burst, on every point. Run 261015-0015 spent 2.4 h of its 4.82 h
    # there. Below the threshold the reference is cached PER PIXEL instead, measured
    # once by the pre-pass and reused at every bias.
    #
    # What makes that safe is NOT that the leak is constant -- it is not. It rises
    # smoothly from 256.5 nA at -0.80 V to 286.1 nA at -2.36 V, about 12%, and the
    # range boundaries add no step above that trend (+3.87 and +3.93 nA, inside the
    # ordinary per-step scatter of the ranges they join). What makes it safe is that
    # the signal rises 728x over the same span. One fixed per-pixel value, tested
    # against all 3000 points of 261015-0015, costs:
    #
    #     -0.83 V  0.13%     -1.05 V  0.12%     -1.89 V  0.03%
    #     -0.99 V  0.08%     -1.30 V  0.05%     -2.60 V  0.20%
    #
    # -2.60 V is the worst because the reference jumps to 495-745 nA there in all four
    # direction runs, both directions, same range as its neighbours: the array starting
    # to conduct, not an instrument artifact.
    #
    # Set to None to reference every point, which reproduces every run before this.
    leak_ref_per_point_above_v: float | None = -1.0  # measure per point when v > this
                             # (i.e. on the FAINT side, where the reference is 5-62% of
                             # the reading and the subtraction decides the answer); use
                             # the cached per-pixel value at or below it. Same sign
                             # convention as vramp_knee_v. The default keeps the eight
                             # biases -0.99..-0.80 V per-point and caches the other 22.
    leak_ref_prepass_v: float = -0.90  # bias for the one pre-pass that fills the cache.
                             # It has to be its own pass, not a by-product of the first
                             # bias: under high_to_low the first bias is -2.60 V, whose
                             # reference is the 495-745 nA outlier above, and anchoring
                             # there costs 0.16-0.41% instead of 0.14%. Under
                             # low_to_high the faint end comes first and the pre-pass is
                             # redundant, but it costs 100 points x ~3 s = 5 min against
                             # the 1.69 h it saves, so it is not worth a special case.
                             #
                             # Faint side on purpose: the reference is most precise there
                             # (burst sem 0.13 nA on the 1e-5 range against 9.06 nA on
                             # 1e-3) and that precision carries to every bias that reuses
                             # it.

    # ── Dead-pixel threshold (fast scan) ────────────────────────────────
    nvled_v_fastscan: float = -1.35  # single fast-scan bias — does DOUBLE duty. At −1.35 V the pixels
                                     # sit at J≈33 A/cm², on the droop side of a plateau that is wider
                                     # than it is flat. MEASURED over the 100 swept pixels of
                                     # 261015-0015: median EQE runs 93.7% to 99.7% of peak across
                                     # J≈8–30, a 6.3% spread, and the peak is at -1.05 V (J=7.2).
                                     # -1.35 V sits at ~92% of peak, so EQE=P/I here is a good RANK
                                     # proxy and NOT each pixel's peak EQE. (The earlier "within ~1%
                                     # across J≈8–30" is withdrawn -- investigation report 26.2. Use
                                     # fastscan_voltages to add -1.05 V if the peak is what you want.)
                                     # It still gives a full-coverage proxy for all 65 536
                                     # pixels (calibrated to the sweep in analysis), while still emitting
                                     # ~22 nW for the alive/dead classification — ~1000× the
                                     # MEASURED optical floor (3σ = 23 pW at n_pm 32; see the
                                     # optfloor run of 261013, σ ∝ N^-0.499 from n_pm 4 to 512).
                                     # J≈25 keeps ~5 A/cm² of margin before the droop starts (~J30). The
                                     # current there (~25 µA) is high enough for a fast (NPLC 0.01) read.
                                     # (Was −2.6 V / J≈95 — the droop side at only ~78% of peak.)
    fastscan_only: bool = False      # stop after the fast scan: no pixel selection, no
                                     # sweep. Every bias in fastscan_v_list() is scanned over
                                     # the whole ROI; the first still judges alive/dead in the
                                     # log. The run directory ends in _FASTSCAN. The CLI's
                                     # --fastscan V [V ...] sets this and both bias fields.
    fastscan_voltages: tuple[float, ...] = ()  # EXTRA fast-scan biases. Empty = just
                                     # nvled_v_fastscan, which is the historical behaviour and
                                     # stays the FIRST entry either way -- alive/dead
                                     # classification and sweep-pixel selection are judged on
                                     # that bias alone, so adding biases cannot change which
                                     # pixels get swept. The others are measured on every pixel
                                     # and written to the same CSV, distinguished by NVLED_V.
                                     #
                                     # Why more than one: -1.35 V was chosen as a peak-IQE proxy
                                     # on the assumption that it sits on the flat EQE plateau.
                                     # Over the 100 pixels of run 261015-0015 the peak is at
                                     # -1.05 V (81/100 within -0.99..-1.11), and -1.30 V -- the
                                     # nearest grid bias -- reads 93.6% of peak and ranks pixels
                                     # at Spearman 0.934 against the sweep at -1.30 V,
                                     # best 0.940 at -1.244 V -- it agrees best near its
                                     # OWN bias, and only 0.873 at -1.05 V. What it
                                     # cannot do is SEE the peak: median EQE over those
                                     # 100 pixels is 100% at -1.05 V, 93.7% at -1.30 V
                                     # and ~92% at -1.35 V. (An earlier comment here
                                     # claimed 0.890 and 0.989; neither reproduces and
                                     # both are withdrawn -- investigation report 26.2.)
                                     # A list lets one scan carry both and the analysis compare
                                     # them, instead of re-running the array to move one number.
                                     #
                                     # Cost is linear: each extra bias is another full pass over
                                     # the ROI. fastscan_limit_a (1e-3 A) covers the whole usable
                                     # span -- -2.6 V draws ~175 uA -- so the list shares one
                                     # compliance and one range. At -1.05 V that is ~7 uA read on
                                     # the 1e-3 range, where the range offset is a few nA = 0.05%.
    # --- IV of dead pixels, after the fast scan has classified them -----------------
    # A dead pixel is usually shorted, so it has a RESISTANCE where a working one has a
    # diode. That takes no High-C and no settling: a few sub-turn-on points and a slope.
    # The same sweep runs on an equal number of alive pixels, because a resistance with
    # nothing to compare it against says very little.
    dead_iv_pixels: int = 0          # how many dead pixels to sweep. 0 = off
    dead_iv_dark_sigma: float = 3.0  # "dead dead": keep only pixels whose fast-scan power
                                     # is within this many sigma of dark, judged against the
                                     # median per-pixel burst std. A pixel just under the
                                     # alive threshold is still emitting and would give a
                                     # diode curve, not the short this measurement is for.
    dead_iv_volts: tuple[float, ...] = (0.0, -0.1, -0.2, -0.3, -0.4, -0.5)
                                     # all below turn-on (~-0.8 V), so a healthy pixel draws
                                     # almost nothing and a short draws V/R. Same polarity as
                                     # the sweep: nothing here forward-biases the array.
    dead_iv_limit_a: float = 1e-3    # compliance. A hard short at -0.5 V hits it, which is
                                     # itself the result: R below ~500 ohm.
    dead_iv_nplc: float = 1.0        # quiet reading; the whole sweep is a few seconds
    dead_threshold_frac: float = 0.20  # a pixel is alive if P ≥ this × the "typical lit" level
    fastscan_alive_pctl: int = 90  # "typical lit" level = this percentile of the fast-scan power.
                                     # NOT the median: the median collapses to the noise floor on a
                                     # majority-DEAD ROI (then ~everything is mislabelled alive — e.g.
                                     # quadrant 260703-1307 was 57% dark, median 0.08 nW → 84% "alive").
                                     # A high percentile tracks the lit population as long as the lit
                                     # fraction exceeds (100−pctl)% = 10%.

    pixel_area_um2: float = 100.0  # emitting area of one pixel (um^2), for J = I / area
    smu_display_off: bool = False  # turn the SMU front panel off during a run (fewer display updates on the instrument)
    # ── Derived / helpers ───────────────────────────────────────────────

    @property
    def photon_energy_j(self) -> float:
        """Energy of one emitted photon (J), from ``pm_wavelength``."""
        return H_PLANCK * C_LIGHT / (self.pm_wavelength * 1e-9)

    def fastscan_v_list(self) -> tuple[float, ...]:
        """Every fast-scan bias, ``nvled_v_fastscan`` first.

        First is load-bearing: alive/dead classification and sweep-pixel selection read
        only that bias, so the extra entries cannot change which pixels get swept.
        """
        extra = tuple(float(v) for v in self.fastscan_voltages
                      if float(v) != float(self.nvled_v_fastscan))
        return (float(self.nvled_v_fastscan),) + extra

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
        if self.sweep_voltages:
            if self.fine_sweep_mode != "voltage":
                raise ValueError("sweep_voltages needs fine_sweep_mode='voltage'")
            bad = [v for v in self.sweep_voltages if not (-self.nvled_v_limit <= v <= 0)]
            if bad:
                raise ValueError(
                    f"sweep_voltages {bad}: the cathode bias is negative, 0 at most and "
                    f"no deeper than -nvled_v_limit ({-self.nvled_v_limit:g} V)")
        if not self.bitvals or not all(0 <= b <= 15 for b in self.bitvals):
            raise ValueError(f"bitvals={self.bitvals!r}; expected a non-empty tuple of 0..15")
        if any(v > 0 for v in self.fastscan_v_list()):
            # The NVLED cathode is driven negative. A positive entry is a typo, and it
            # would forward-bias into the compliance instead of measuring anything.
            raise ValueError(f"fastscan_voltages={self.fastscan_voltages!r}: every "
                             f"fast-scan bias must be <= 0 V")
        if (self.leak_ref_per_point_above_v is not None
                and self.leak_ref_per_point_above_v > 0):
            raise ValueError(
                f"leak_ref_per_point_above_v={self.leak_ref_per_point_above_v}: this is "
                f"a bias in volts and the sweep is negative, so a positive value would "
                f"cache the reference at EVERY point, faint side included.")
        if self.pool_groups:
            if self.source_mode == "current":
                raise ValueError(
                    "pool_groups with source_mode='current': a current source sets ONE "
                    "current for the whole frame, so a pooled group would share it "
                    "between its members and the per-member operating point would be "
                    "unknown. Pooling is a voltage-sweep feature.")
            if self.sweep_select != "blocks":
                raise ValueError(
                    f"pool_groups with sweep_select={self.sweep_select!r}: a group is "
                    f"formed inside ONE block, so there must be blocks. Use "
                    f"sweep_select='blocks'.")
            bad = [k for k in self.pool_groups if int(k) < 2]
            if bad:
                raise ValueError(
                    f"pool_groups={self.pool_groups} contains {bad}: a group of one is "
                    f"a single pixel measured through the pooled path, which only hides "
                    f"it from the analysis. Drop the entry.")
            if self.pool_include_singles and self.pixels_per_block < 1:
                raise ValueError(
                    "pool_include_singles needs pixels_per_block >= 1: there has to be "
                    "a single in the block for the group to share. Set "
                    "pool_include_singles=False for disjoint groups.")
        if self.fastscan_pool:
            if self.fastscan_pool < 2:
                raise ValueError(
                    f"fastscan_pool={self.fastscan_pool}: a group of one IS the "
                    f"ordinary scan. Use 0 to turn the screen off.")
            if not 0 < self.fastscan_pool_rescan_frac <= 1:
                raise ValueError(
                    f"fastscan_pool_rescan_frac={self.fastscan_pool_rescan_frac}; "
                    f"expected a fraction in (0, 1]")
            if self.fastscan_pool_detrend < 3 or self.fastscan_pool_detrend % 2 == 0:
                raise ValueError(
                    f"fastscan_pool_detrend={self.fastscan_pool_detrend}; expected an "
                    f"odd window of at least 3 groups")
        if self.pool_pm_range_below_v is not None and self.pm_autorange:
            raise ValueError(
                "pool_pm_range_below_v with pm_autorange: the per-bias ranging already "
                "moves the pooled targets to a coarser range where they need it. Set "
                "pm_autorange=false to use the fixed two-range switch.")
        if self.pool_pm_range_below_v is not None:
            if not self.pool_groups:
                raise ValueError(
                    "pool_pm_range_below_v without pool_groups: the switch is for "
                    "pooled targets, and there are none.")
            if not self.pool_pm_range:
                raise ValueError(
                    "pool_pm_range_below_v needs pool_pm_range: the threshold says "
                    "WHERE to switch and pool_pm_range says WHAT to switch to.")
            if self.pool_pm_range_below_v > 0:
                raise ValueError(
                    f"pool_pm_range_below_v={self.pool_pm_range_below_v}: this is a "
                    f"bias in volts and the sweep is negative.")
            if self.pool_pm_range <= self.pm_range:
                raise ValueError(
                    f"pool_pm_range={self.pool_pm_range:g} is not coarser than "
                    f"pm_range={self.pm_range:g}. Switching to a finer range at the "
                    f"BRIGHT end is backwards -- that is the end that over-ranges.")
            if self.sweep_order != "bias_major":
                raise ValueError(
                    f"pool_pm_range_below_v with sweep_order={self.sweep_order!r}: the "
                    f"range would change at every point instead of once per bias. Use "
                    f"bias_major, where the bias is the outer loop.")
        if self.pm_range_check_targets < 0:
            raise ValueError(f"pm_range_check_targets={self.pm_range_check_targets}; "
                             f"expected >= 0")
        if self.pool_pm_range and self.pool_pm_range <= 0:
            raise ValueError(f"pool_pm_range={self.pool_pm_range}; expected 0 (use "
                             f"pm_range) or a positive power in W")
        for name in ("nvled_range_i", "probe_range_i", "current_source_range_i"):
            if name == "nvled_range_i" and self.nvled_range_i is None:
                continue                     # None: the range follows the limit
            if getattr(self, name) not in SMU_I_RANGES:
                raise ValueError(f"{name}={getattr(self, name)}; the 2602B only has "
                                 f"the ranges {SMU_I_RANGES}")
        if not 0 < self.pm_wavelength < 1e5:
            raise ValueError(f"pm_wavelength={self.pm_wavelength} nm out of range")
        if self.n_fine < 2 or self.n_coarse < 2:
            raise ValueError("n_fine and n_coarse must be >= 2")
        for name in ("vramp_lowv_settle_s", "vramp_highv_settle_s", "leak_ref_settle_s",
                     "settle_scale", "pixel_switch_settle_s"):
            if getattr(self, name) < 0:
                raise ValueError(f"{name} must be >= 0")
        if self.sweep_order not in ("pixel_major", "bias_major"):
            raise ValueError(f"sweep_order={self.sweep_order!r} must be 'pixel_major' "
                             f"or 'bias_major'")
        if self.sweep_order == "bias_major" and self.fine_sweep_mode != "voltage":
            # Current mode picks each pixel's targets from that pixel's own coarse map,
            # so there is no common grid to hold a "bias" at. Fail loudly rather than
            # silently measuring one pixel's targets on every pixel.
            raise ValueError("sweep_order='bias_major' needs fine_sweep_mode='voltage': "
                             "in current mode every pixel has its own setpoints, so "
                             "there is no shared bias to amortise. Add "
                             "--set sweep_order=pixel_major.")
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
        # The retry's gate is the OPTICAL ratio, so it only means "the write did not
        # land" when the reference is DARK. Run 261015-2326 referenced at bitval 1 and
        # found it emits 17.4 nW, 11.6% of bitval 15 at -2.60 V and 84% of it at
        # -0.80 V: the gate fired on 33 of 64 points and burned 66 frame writes on a
        # pixel doing exactly what it was told. Section 20.
        if self.leak_reference and self.leak_ref_tries > 1 and self.leak_ref_bitval != 0:
            raise ValueError(
                f"leak_ref_tries={self.leak_ref_tries} with leak_ref_bitval="
                f"{self.leak_ref_bitval}: the retry gate is optical and a non-zero "
                "reference bitval EMITS, so every point would retry to the limit. Set "
                "leak_ref_tries=1, or reference at bitval 0.")
        # Both dead-abort sites in acquire.py judge the pixel at bias index 0 and say
        # "the first (brightest) point". Under low_to_high index 0 is the FAINTEST.
        # Array42 reads 0.21 nW at -0.80 V against dead_abort_nw = 2.0 (runs 261016-0855
        # and -0911), so the absolute floor would drop every pixel at the first bias and
        # the run would return nothing. This is why no low_to_high sweep exists on disk
        # since June 2026. Section 23.
        if self.sweep_direction == "low_to_high" and self.dead_abort:
            raise ValueError(
                "sweep_direction=low_to_high with dead_abort=True: the dead-pixel test "
                "runs at the FIRST bias, which is the faintest in this direction "
                "(0.21 nW at -0.80 V on Array42 against a 2.0 nW floor), so every pixel "
                "would be dropped. Set dead_abort=false for the direction A/B test.")
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
        if self.calibrate_per_quadrant < 0:
            raise ValueError("calibrate_per_quadrant must be >= 0 (0 = single ROI)")
        if self.calibrate_per_quadrant and self.calibrate_quadrant_window < 2:
            raise ValueError("calibrate_quadrant_window must be >= 2 px")
        if self.dead_iv_pixels < 0:
            raise ValueError("dead_iv_pixels must be >= 0 (0 is off)")
        if self.dead_iv_pixels:
            if not self.dead_iv_volts or len(self.dead_iv_volts) < 3:
                raise ValueError("dead_iv_volts needs at least 3 points to fit a slope")
            if any(v > 0 for v in self.dead_iv_volts):
                raise ValueError(
                    f"dead_iv_volts={self.dead_iv_volts} has a positive entry. The array "
                    f"is driven negative; a positive bias here forward-biases it in a "
                    f"direction nothing else in this program uses.")
            if min(self.dead_iv_volts) < -0.75:
                raise ValueError(
                    f"dead_iv_volts reaches {min(self.dead_iv_volts):g} V, at or past "
                    f"turn-on (~-0.8 V). Then a HEALTHY pixel conducts too and the slope "
                    f"is a diode, not the short this measurement exists to find.")
            if self.dead_iv_dark_sigma <= 0:
                raise ValueError("dead_iv_dark_sigma must be > 0")
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
