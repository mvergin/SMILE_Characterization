# Known bugs and open defects

Last reviewed 16 October 2026. `§N` refers to `INVESTIGATION_REPORT.md` in the analysis
repo, which is where the evidence for every "fixed" line below lives.

## Open

* ~~Some pixels do not go dark at bitval 0.~~ **Cause found: the FPGA does not always
  write the frame.** It is intermittent in both directions — (56,51) referenced clean in
  two runs and contaminated in four, and single biases flip inside one sweep. The sweep
  now verifies the reference optically and re-writes the frame up to `leak_ref_tries = 3`
  times. §19, open question 9. **Every run taken before 16 October 2026 carries the
  uncorrected rate**, so `REF_CONTAMINATED` stays in the analysis.
* **The per-write frame failure rate is unknown.** Order a few percent, from the runs
  above, but nothing has measured it properly or asked whether it depends on the pixel,
  the bitval or the array position. `LEAK_REF_TRIES` now records it on every point.
* **The backplane leakage has never been recorded as a time series.** Every leakage
  number in the report is a settled burst. Flat from 10 min to 7 h when reconstructed
  from run 261015-0015, unmeasured from one burst to 10 min, and the within-burst SEM is
  1.6 % of the reference — which at −0.80 V is 0.9 % on `I_LED`. `--switch-to-bitval 0`
  records it. §19 Q5, open question 15.
* **A blanked frame conducts ~4× more than an addressed pixel at bitval 0.** Reproducible
  across 12 pixels and two experiments, worked around (the reference is the *addressed*
  pixel, never a blank frame) but not understood. §6, open question 2.
* **Dark acquisition after a pixel only works in the fast scan.** For the transient mode
  the dark values would have to be taken from the samples before turn-on. `Dark Tail`
  exists and is wrong. The original idea — send the dark frame during the recording
  window — was never implemented.
* **The transient mode records the voltage step rather than a transient per voltage.**
  While the Keithley measures into its buffer the voltage cannot be changed freely. There
  are several ways to do this with minimal downtime; it was implemented separately and
  never merged.
* **Cross-run drift of ~2 % per day, unmeasured.** This is the project's largest known
  error. It is bigger than the loop-order effect below, it contaminated five runs' worth
  of conclusions in §15–§17, and nobody knows whether it is the array, the PM400 or the
  collection geometry. Any absolute number needs a same-day `pixel_major` anchor.
  §18, open question 14.
* **`bias_major` reads ~1 % low in current** against `pixel_major`, on the faint side
  only — zero above −1.8 V. Measured back to back (runs 261015-1421 / -1529), worth
  1.05 mV. Not worth chasing at that size, but it is real: three pixels agree to 0.3 pp.
  Report EQE(J), not EQE(V). §18, open question 12 (closed).

  *Earlier this entry read "up to 3 %". Two thirds of that was the drift above.*

## Fixed

* ~~High-C mode does not work: it still tries to change the range and errors.~~ Fixed.
  In High-C the **compliance limit selects the measure range** and autoranging is locked
  — `5069 SMU B: Autorange locked for HighC mode` was the symptom of setting a range
  explicitly. `highc_b: true` is now mandatory: without it every reading below ~10 µA is
  a range artifact, wrong by up to 517×. §1.
* ~~The version name is written into the config so it can be traced later.~~ This was
  lost at some point and is now restored properly: every run writes `RUN_PROVENANCE.json`
  with the commit, the branch, a dirty flag and the uncommitted diff's fingerprint, and
  the same block is embedded in every `*_config.json`. See `measure_iqe/provenance.py`
  and `RUN_FORMAT.md`.
* ~~Post-processing can be switched off and done later in a second tab.~~ Post-processing
  is no longer in this repository at all; the analysis is a separate step that reads a
  run directory. `RUN_FORMAT.md` is the contract.
