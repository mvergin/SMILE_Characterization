# Acquisition techniques tried on this setup

Four techniques exist for getting light + current out of the array. `measure_iqe`
implements one of them. The other three live in `measurement/` as **reference
implementations** — nothing imports them, they are not maintained, and each has
open problems recorded below. They are kept because the *techniques* are sound
and at least two of them are directly relevant to the 50 µF capacitance problem
(see "Why these still matter").

The monolithic driver that used to run them, `auto_measure.py`, was removed. It
is in git history at `2c598d1`:

```bash
git checkout 2c598d1 -- auto_measure.py
```

`measurement/coordinator.py` drives the outer pixel loop and dispatches into a
`MeasurementMode` subclass, so a technique can be revived without reviving the
driver.

---

## 1. Fast Scan — `measurement/modes.py:FastScanMode`

Continuous PM400 reads plus a burst SMU read per point. No waveform.

**Status: superseded.** This is the technique `measure_iqe` uses, and
`measure_iqe` does it better — overlapped SMU/PM acquisition, adaptive per-zone
effort, deterministic ranging, per-pixel conditioning, managed autozero. Use
`measure_iqe`. The copy here is only of historical interest.

---

## 2. Full Transient — `measurement/modes.py:TransientMode`

Array-mode PM400 plus a triggered SMU, capturing the **full optical waveform**
per pixel rather than a settled average.

**Status: incomplete.** From `KNOWN_BUGS.md`:

* Dark acquisition after a pixel only works in Fast Scan. For transient mode the
  dark values would have to be taken from the samples *before* turn-on. There is
  a "Dark Tail" path but its values are wrong.
* The intended fix was to send the dark frame during the recording window.

`measure_iqe` has no equivalent — it captures no time-resolved data at all.
`Sweep_samples.csv` holds the ADC burst at each setpoint, which gives the noise
distribution per point, not turn-on dynamics.

---

## 3. Ultra-Fast — `measurement/ultra_fast.py:UltraFastMode`

One continuous PM400 array capture spanning **many pixels at once**, sliced
afterwards by the FPGA's `set_pixel` ACK timestamps. Per-pixel overhead collapses
to the ACK latency (~5 ms) instead of an arm/fetch cycle per pixel. Roughly 283
pixels per 2 s capture at Δt = 200 µs.

**Status: never fully worked — PM400 limited.** The on-board array buffer is
10 000 samples, which bounds chunk size, and the slicing alignment depends on the
ACK timestamps staying coherent with the PM400's sample clock across a chunk.
`ULTRA_FAST_HW_TEST_PLAN.md` is the test plan that was being worked through when
this was set aside.

Worth keeping as an idea: it is the only technique here that addresses
full-array throughput directly, and 512×512 = 262 144 pixels is the standing
problem.

---

## 4. Experimental hardware-paced sweep — `measurement/experimental.py:ExperimentalSweepMode`

One PM400 1 s array capture per pixel. Inside that capture, a **TSP script
running on the Keithley** walks the whole voltage list and, at each voltage,
rapid-fires K channel-A + channel-B measurement pairs. All three buffers
(chA current, chB current, chB voltage) come back with **per-sample instrument
timestamps**, which are aligned to the PM400's `perf_counter` frame to splice
per-voltage averages and save both waveforms.

**Status: most recent work before the switch** (`16b07ac`, 2026-04-17, "new
experimental nvled sweep method employing full buffered operation"). Not
validated against the physics problems that `measure_iqe` later solved.

From `KNOWN_BUGS.md`: the original NVLED-sweep-plus-transient design was wrong —
it captured the voltage *step* rather than doing a transient measurement at each
voltage, because the voltage cannot easily be changed while the Keithley is
measuring into its buffer. This mode is the separate implementation that was
written to fix exactly that.

---

## Why these still matter: the 50 µF problem

The array carries a ~50 µF capacitor. Moving the cathode bias across the
rising-side span (−0.8 → −1.3 V) moves `C·ΔV = 25 µC` of charge. At the LED
currents in that region (~1–10 µA) that is **seconds per pixel** of settling; at
SMU-limit currents it is **tens of milliseconds**. Getting the SMU to supply that
charge instead of the LED, and knowing when the remainder has settled, is the
whole problem.

Two of the techniques above are the right instruments for it:

* **Modes 2 and 4 give timestamped buffered capture.** Characterising the
  capacitive transient means recording I(t) after a voltage step at high rate,
  per voltage and per range, with and without High-C — which is exactly what the
  TSP-plus-buffer mechanism in `experimental.py` already does. `measure_iqe`
  cannot do this; its `--warmup-diag` samples at roughly 1 Hz.
* **Mode 2's waveform capture enables charge-decay measurement.** Charging the
  capacitor and then watching it discharge through the LED turns `I = C·dV/dt`
  into a current measurement derived from a voltage slope, which the SMU
  measures far more accurately at 1 µA than its 100 nA current range does. That
  needs concurrent PM400 array capture, i.e. mode 2's machinery.

So: debug and strip these to what works, rather than rewriting the buffered
capture from scratch.

---

## Also still here

`smile_postprocess.py` and `smile_heatmap.py` are the post-processing for this
cluster (heatmaps, optical-yield report, on-time estimation). They consume
`auto_measure.py`'s CSV layout, **not** the one in `RUN_FORMAT.md`, so they do
not apply to `measure_iqe` output and nothing imports them any more. They are
kept with the modes they belong to. If a technique above is revived onto
`measure_iqe`, its post-processing belongs in the analysis repo instead.
