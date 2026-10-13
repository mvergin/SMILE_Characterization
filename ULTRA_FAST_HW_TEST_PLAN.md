# Ultra-Fast Mode — Hardware Test Plan

This document is for the first real-hardware bring-up of the Ultra-Fast
measurement mode after the refactor + Step 10 implementation. It exists
because everything so far has only been verified against the simulation
mocks (`instruments/sim.py`) — the sim has zero ACK jitter and a perfect
PM400 array buffer, so the chunk-sizing margins and slicing alignment
have not been stressed.

When you sit down at the rig, walk through the sections in order and
collect the numbers / artefacts at the end of each. Send all of it back
in one batch and we can decide together what (if anything) needs tuning.

---

## 0. Pre-flight (5 min)

Confirm the refactored code still launches and the existing modes still
work on real hardware. We have NOT touched the FPGA driver, but we
have substantially reorganised everything around it.

- [ ] Launch the GUI: `uv run python smile_automatic_gui_2602b.py`
- [ ] Verify the device dropdowns populate with real VISA addresses
      (not just `Sim_PM400` / `Sim_K2602B`).
- [ ] Run a tiny **Full Transient** scan: 1 pixel, 1 bit value, real HW.
      Confirm the CSV is written and looks normal.
- [ ] Run a tiny **Fast Scan**: same pixel, 1 bit value. Confirm CSV.

If either of those fails, **stop** and report the error before going
any further — Ultra-Fast depends on the same instrument drivers, so a
regression here would also break Ultra-Fast.

**Report back:** any error messages, or "pre-flight clean".

---

## 1. Smallest possible Ultra-Fast scan (5 min)

Goal: confirm the wiring is correct end-to-end on real hardware before
we worry about throughput or accuracy.

- [ ] Switch mode dropdown to **Ultra Fast**.
- [ ] Settings:
  - PM Δt: **200 μs**
  - Dark every N: **5**
  - Window Trim: **10 %**
  - ROI: a **5×1** strip somewhere known-good (5 pixels total).
  - Bit values: `0,15` (just two).
  - Dark Acquisition: **on**.
- [ ] Confirm the live label shows something like
      "~283 pixels/chunk, 2.0 s capture (10000 samples @ 200 μs)".
- [ ] Hit Start.

**What to verify in the resulting CSV:**
- [ ] 5 pixels × 2 bit values = 10 `(x, y, bv)` tuples.
- [ ] Each tuple has exactly one `PM400`, one `VLED`, one `NVLED` row.
- [ ] Some `DARK_PM400` rows present (with `X=-1, Y=-1`).
- [ ] PM400 `MEAS_VALUE` values are in the right ballpark (compare to
      what the same pixels gave you in Full Transient pre-flight).

**Report back:**
- The full CSV (it'll be tiny).
- Any GUI log messages from the run, especially the
  "Ultra-Fast: delta_t=… chunk=… pixels/chunk=… dark_every=… trim=…"
  line.
- Any errors / warnings from the PM400 or Keithley.

---

## 2. Single-chunk throughput sanity (10 min)

Goal: figure out the **actual** average per-pixel cost on this rig, so
we can decide if `_AVG_PIXEL_MS = 6.0` (in `measurement/ultra_fast.py`)
is right.

- [ ] Same Ultra-Fast settings as Section 1.
- [ ] ROI sized to fit comfortably inside one chunk: try **100 pixels**
      (e.g. 10×10).
- [ ] Single bit value: `15`.
- [ ] Watch the elapsed time on the GUI status bar / your wristwatch.

**Report back:**
- Wall-clock time from "Start" to "Measurement Finished".
- The CSV (or at least the first ~20 rows + last ~20 rows).
- Note whether the run looked smooth or whether the GUI hung at any
  point.

From the wall-clock + 100 pixels we can compute the real per-pixel cost.
If it's > 6 ms, we'll need to bump `_AVG_PIXEL_MS`. If it's < 4 ms, we
can shrink it and pack more pixels per chunk.

---

## 3. Multi-chunk run (10 min)

Goal: confirm chunk boundaries are clean — no gaps, no double-counted
pixels, no clobbered ACK timestamps when we tear down + re-arm the
PM400 array between chunks.

- [ ] Same Ultra-Fast settings.
- [ ] ROI: **400 pixels** (e.g. 20×20). At 200 μs / Δt, with ~283
      pixels/chunk, this should split into 2 chunks.
- [ ] Bit values: `0,15`.

**What to verify:**
- [ ] Final pixel count in the CSV matches `400 × 2 = 800` `(x, y, bv)`
      tuples (count `PM400`-type rows).
- [ ] No duplicates: `(x, y, bv)` should be unique across the file.
- [ ] No missing pixels at the chunk seam — sort by `(bv, x, y)` and
      eyeball the PM400 column for any obvious discontinuity around
      where the chunks split (roughly pixel 283 of each bit value).

**Report back:**
- Total wall-clock time.
- `PM400` row count.
- Whether you see any anomaly at the seam.
- The full CSV is probably fine to ship — it'll be ~2400 rows.

---

## 4. Δt sweep (10 min)

Goal: verify that the chunk-geometry math holds at the extremes of the
Δt range.

Run the same 100-pixel ROI from Section 2 at three Δt values:

| Δt (μs) | Expected chunk duration | Expected pixels/chunk |
|---------|------------------------:|----------------------:|
| 100     | 1 s                     | ~141                  |
| 500     | 5 s                     | ~708                  |
| 1000    | 10 s                    | ~1416                 |

For each:
- [ ] Run, record wall-clock.
- [ ] Spot-check that PM400 values are stable (mean ± std reasonable).
- [ ] Note whether the run completes cleanly.

**Report back:** the table above with your wall-clock time per Δt
filled in, plus any errors at any Δt.

The Δt=1000 μs run is the most demanding for the PM400 (10 s of
continuous array sampling) — if anything is going to time out, it's
that one.

---

## 5. Cross-validation against Fast Scan (15 min)

Goal: prove that Ultra-Fast values agree with a known-good mode on the
**same** region. This is the only thing that proves the smart-windowing
math is correct.

- [ ] Pick a small ROI (say **5×5** = 25 pixels) somewhere with a mix
      of working and non-working pixels if possible.
- [ ] Run **Fast Scan** with whatever your normal settings are.
      Save the CSV — call it the reference.
- [ ] Run **Ultra Fast** on the *exact same ROI*, *same bit values*,
      *same VLED / NVLED*. Default Ultra-Fast settings.
- [ ] Compare the per-pixel `PM400` `MEAS_VALUE` between the two runs.

**What to look for:**
- They should agree to within ~5–10 % per pixel on bright pixels.
  Larger discrepancies on dim pixels are expected because the
  trimmed-window average is shorter.
- Dim/dark pixels should give roughly the same value in both modes.
- If Ultra-Fast is **systematically** higher or lower (not random), the
  trim window is probably grabbing the wrong slice — increase
  `Window Trim` to 20 % and re-run, see if it converges.

**Report back:**
- Both CSVs.
- Your subjective read on how well they agree.
- If you ran with a higher trim %, mention what value you used.

---

## 6. Stress test (optional, 10 min)

Only if Sections 1–5 all looked clean. This is the "ship it" gate.

- [ ] **Full quadrant** scan (256×256 = 65 536 pixels), single bit
      value `15`, default Ultra-Fast settings.
- [ ] Watch for: GUI freezing, CSV writer keeping up, any pixel-count
      discrepancies at the end.

**Report back:**
- Total wall-clock time.
- Final `PM400` row count (should be 65 536).
- Whether you saw any anomalies.
- A subjective comparison of total scan time vs your typical Fast Scan
  on the same region.

---

## What to send back when you're done

In one message, ideally:

1. Pre-flight result (clean / errors).
2. Section 1 CSV + GUI log line.
3. Section 2 wall-clock + first/last 20 CSV rows.
4. Section 3 pixel count + seam observation + (optionally) full CSV.
5. Section 4 filled-in Δt-sweep table.
6. Section 5 both CSVs + your read on agreement.
7. (Optional) Section 6 wall-clock + row count.
8. Anything weird that didn't fit a section above — instrument errors,
   GUI hangs, surprising values, anything.

From that I can decide whether to:
- Adjust `_AVG_PIXEL_MS` / `_CHUNK_FILL_FRACTION` in
  `measurement/ultra_fast.py` based on real ACK timing.
- Tune the default `Window Trim` if the cross-check shows systematic
  bias.
- Fix any actual bugs that the sim couldn't catch (PM400 array
  re-arming, SMU timestamp alignment drift over a 10 s chunk, etc.).

---

## Things to watch out for / known unknowns

These are the parts of Ultra-Fast I'm least confident about because the
sim cannot exercise them:

- **PM400 array re-arming between chunks.** We call `pm.abort()` then
  `pm.configure_array_mode(...)` then `pm.start_array()` for every
  chunk. The sim accepts this trivially. Real PM400 firmware sometimes
  needs a small delay or an `*OPC?` between abort and re-init.
  Symptom would be: chunk 1 is fine, chunk 2 onwards are empty or
  full of zeros.

- **SMU timestamp drift.** We assume Keithley `nvbuffer1.timestamps`
  are linearly aligned with `perf_counter()` over a chunk. Over 10 s
  the two clocks could drift enough that the windowed slice grabs the
  wrong samples for VLED/NVLED. Symptom: PM400 values look right,
  but VLED/NVLED look noisy or off-by-a-pixel.

- **First-chunk dark reference.** The leading blank-frame at the very
  start of each chunk gives a dark sample, but the LED state from
  *before* the chunk started is unknown — the previous chunk ended
  with a blank, so that's OK in a multi-chunk run, but on chunk 0
  there's no prior state. Probably fine, but worth eyeballing the
  very first PM400 value of a fresh run.

- **`set_pixel` ACK time variance.** We assume ~5 ms ± a little. If the
  real distribution has a long tail (occasional 20 ms ACKs), the
  trim-window math still works (it just gives that one pixel a longer
  averaging window), but the chunk fill-fraction safety margin
  (`_CHUNK_FILL_FRACTION = 0.85`) might be too tight. Symptom:
  occasional chunk where the last few pixels' ACKs land *after* the
  PM400 buffer is already full, so they get lost.

If you see any of those symptoms, mention which one — that tells me
exactly where to look.
