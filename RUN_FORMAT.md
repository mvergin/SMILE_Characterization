# Run directory format — the contract between measurement and analysis

This file is the interface between two repositories:

| repo | role |
|---|---|
| **SMILE_Characterization** | talks to the instruments and **writes** a run directory |
| **SMILE_IQE_Analysis** | **reads** a run directory, fits and reports |

They share no code. A run directory on disk is the only thing passed between
them, so this document is the API. **It is kept byte-identical in both repos.**
Change it in one, copy it to the other, and update both fixture tests in the
same breath.

`tests/fixture/` in each repo is a real (simulated) run matching this spec.
Measurement asserts it still writes this shape; analysis asserts it still reads
it. If a change breaks one side, the other side's test fails too.

---

## Layout

```
<YYMMDD-HHMM>_<sample_name>_ROI_(x1, y1, x2, y2)/
├── RUN_PROVENANCE.json                         what produced this run
├── <P>calibration_applied.yaml                 the calibration, copied (if any)
├── FastScan/
│   ├── <P>aggregated_bitval=<b>_mean_std.csv   one row per ROI pixel
│   └── raw_data/
│       ├── <P>FastScan.csv                     long-format per-channel rows
│       └── <P>FastScan_config.json             settings used
└── NVLEDSweep/
    └── Sweep/
        ├── <P>aggregated_bitval=<b>_mean_std.csv  one row per sweep point
        ├── <P>settling_traces.npz              per-point transients (optional)
        └── raw_data/
            ├── <P>Sweep.csv                    long format + dark readings
            ├── <P>Sweep_samples.csv            every burst sample (optional)
            └── <P>Sweep_config.json            settings used
```

### The filename prefix `<P>`

`<P>` is `<sample_name>_<YYMMDD-HHMM>_`, the same stamp as the directory. It is there
so a CSV **copied out of its directory can still name the run it came from** — the
commonest way a measurement loses its identity is someone pasting one file into a
slide deck.

**Readers must accept both spellings.** Every run taken before 16 October 2026 has no
prefix, and those runs are the comparison baseline for the whole investigation. Glob
with a leading `*`:

```python
sorted(run.glob("NVLEDSweep/*/*aggregated_bitval=*_mean_std.csv"))
```

`measure_iqe.run_file(run_dir, "NVLEDSweep/Sweep/Sweep.csv")` does this for you and
raises `FileNotFoundError` naming the directory rather than returning nothing.

### `RUN_PROVENANCE.json`

Written before the first measurement, so an interrupted run is still identified. The
same block is embedded under `"provenance"` in every `*_config.json`.

| key | meaning |
|---|---|
| `commit`, `commit_short`, `branch` | the code that ran |
| `describe` | `git describe --tags --always --dirty` |
| `dirty`, `changed`, `diff_sha256` | uncommitted edits: whether, which files, and a fingerprint of the diff |
| `started`, `host`, `python`, `platform` | when and where |
| `sample_name`, `calibration_source` | what was measured, and with which calibration |

**`dirty: true` means the run cannot be reproduced from `commit` alone.** `changed`
names up to 50 files and `diff_sha256` fingerprints the uncommitted diff, so two dirty
runs with the same `commit` and `diff_sha256` did run identical code.

A git **tag** is deliberately not the mechanism here: it needs someone to remember to
move it, and that discipline fails. Tags remain useful as a human label on top.

`--warmup-diag` runs write `warmup_diag.csv` + `warmup_diag_config.json` at the
run root instead of `NVLEDSweep/`, and are not part of this contract.

### Reader rules

* The directory name is **descriptive only**. Never parse it; read
  `*_config.json` for the sample name and ROI.
* There may be several `aggregated_bitval=*.csv` files, one per bitval.
  Concatenate them; `BITVAL` distinguishes the rows.
* `NVLEDSweep/` may hold more than one sweep group directory. Readers should
  glob its children rather than hardcoding `Sweep`.
* Every file is written incrementally and flushed per pixel. **A run directory
  from an interrupted measurement is valid and must be readable** — it simply
  holds fewer pixels. Never assume a pixel count or a complete voltage ramp.

---

## `aggregated_bitval=<b>_mean_std.csv`

The primary table. One row per measured point.

### Always present, in this order

| column | unit | meaning |
|---|---|---|
| `X`, `Y` | px | pixel coordinates, 0..511 |
| `BITVAL` | — | CMOS drive level, 0..15 |
| `NVLED_V` | V | cathode (channel B) bias, **negative**, rounded to 4 dp. In current-source mode this is the *measured* diode voltage instead of a setpoint. |
| `TIME_START`, `TIME_END` | s | seconds since the phase started, for drift analysis |
| `NVLED_CURR_MEAN` | A | channel B current — **this is the LED current** |
| `NVLED_CURR_STD` | A | **standard error of the mean** (σ/√N), not per-sample scatter |
| `VLED_CURR_MEAN` | A | channel A (anode) current — leakage cross-check only |
| `VLED_CURR_STD` | A | standard error of the mean |
| `PM400_POWER_MEAN` | W | optical power |
| `PM400_POWER_STD` | W | standard error of the mean |

### Present only when `save_raw_samples` is on

| column | unit | meaning |
|---|---|---|
| `NVLED_CURR_MEDIAN`, `VLED_CURR_MEDIAN`, `PM400_POWER_MEDIAN` | A, A, W | median of the burst |
| `NVLED_CURR_SAMPSTD`, `VLED_CURR_SAMPSTD`, `PM400_POWER_SAMPSTD` | A, A, W | population std of the burst |
| `NVLED_CURR_N` | — | samples in the current burst |

A reader must treat these as optional and fall back to the `*_MEAN` columns.

### Present only when `leak_reference` is on (default since 261013)

The same pixel re-measured at `LEAK_REF_BITVAL` (0: addressed, but not driven),
back to back with the lit point at the same bias.

| column | unit | meaning |
|---|---|---|
| `NVLED_CURR_REF` | A | ch B current with the pixel addressed and off |
| `NVLED_CURR_REF_STD` | A | SEM of that |
| `VLED_CURR_REF` | A | ch A current in the same state |
| `PM400_POWER_REF` | W | optical power in the same state: the true local dark |
| `PM400_POWER_REF_STD` | W | SEM of that |
| `LEAK_REF_BITVAL` | - | the drive level used as the reference |

**This is not optional precision, it is the measurement.** Channel B reads the
*total* cathode current. Measured over 12 pixels in 4 quadrants on 261013:

| bias | reference as a fraction of the reading |
|---|---|
| -0.95 V | 8% |
| -0.90 V | 14% |
| -0.85 V | 29% |
| **-0.80 V** | **58%** |

Without the subtraction the EQE appears to collapse ~20x below J = 0.4. With it
the curve is smooth and monotone. **Every rising-side EQE this project produced
before 261013 is wrong**, and uncorrected runs cannot be fixed afterwards,
because the reference was never recorded.

So: `I_LED = |NVLED_CURR_MEAN| - |NVLED_CURR_REF|`, and the error propagates as
`sqrt(sig_lit^2 + sig_ref^2)` against the smaller corrected current. Subtract
`PM400_POWER_REF` rather than the run-wide dark when it is present: it is the
same pixel, the same frame, the same instant.

A reader without these columns must fall back to `I_LED = |NVLED_CURR_MEAN|`
and say so, rather than silently reporting an uncorrected rising side.

### Present for the voltage ramp

| column | unit | meaning |
|---|---|---|
| `NVLED_LIMIT_A` | A | the compliance in force at this point |
| `NVLED_AT_COMPLIANCE` | bool | the reading sat on that limit |

The limit is not constant across a sweep: -2.6 V draws ~120 uA while the rising
side needs a 10 uA limit to resolve at all, and under High-C the limit also
selects the measure range. It is chosen per point from the previous point's
current (the sweep runs high current to low current, so that is an upper bound).

**A point with `NVLED_AT_COMPLIANCE` true is not a measurement of the device.**
Drop it. Run 261014-0007 had eight consecutive points reading exactly 10000.4 nA
because an earlier version tied the limit to the density knee; they look like a
plateau in the I-V and would corrupt any fit through them.

### Conventions that are easy to get wrong

* **`*_STD` is the SEM**, kept under that name for backwards compatibility.
  `*_SAMPSTD` is the per-sample spread. Use `*_STD` for reliability gating,
  because heavier averaging genuinely lowers it.
* **`*_MEAN` is not always the arithmetic mean.** In older runs, and whenever
  `save_raw_samples` is off, it holds a *robust median* (chosen to reject SMU
  autorange glitches). When the `*_MEDIAN` columns exist, `*_MEAN` is the true
  mean and `*_MEDIAN` is the robust value. **Prefer `*_MEDIAN` when present,
  else `*_MEAN`.**
* **Channel B current is negative** (the SMU sinks it). Take `abs()`.
* **Use `NVLED_*` as the LED current, not `VLED_*`.** Channel A can carry other
  current paths. A large `VLED/NVLED` ratio means leakage, not signal. Measured
  261013: channel A sits at ~466 uA of array background with ~29 uA of noise,
  against channel B's 9 nA, so it is a supply-rail monitor and nothing more.
* **A blanked frame is not a valid zero.** It draws ~4x *more* than the addressed
  pixel at bitval 0 (943-956 nA against 236-254 nA), so lit-minus-blanked gives a
  negative LED current. The reference must leave the backplane addressing alone.
* Floats are written with `repr()` for full round-trip precision. Values may be
  `nan`. A dead-skipped pixel contributes exactly one row.

---

## `raw_data/{FastScan,Sweep}.csv`

Long format, one row per channel per point. Columns:

`X, Y, BITVAL, NVLED_V, TIME, TYPE, MEAS_VALUE, MEAS_STD`

`TYPE` is one of:

| `TYPE` | `MEAS_VALUE` | `MEAS_STD` |
|---|---|---|
| `PM400` | optical power (W) | SEM |
| `VLED` | channel A current (A) | SEM |
| `NVLED` | channel B current (A) | SEM |
| `DARK_PM400` | dark optical power (W), pixel blanked | `nan` |
| `DARK_VLED` | dark channel A current (A) | `nan` |
| `DARK_NVLED` | dark channel B current (A) | `nan` |

The `DARK_*` rows are written once per pixel, after its sweep, with the pixel
blanked. **Subtract the median `DARK_PM400` from `PM400_POWER_*` before
computing EQE**, then clip at zero. A dead-skipped pixel has no dark rows.

This file is the fallback when no aggregated CSV exists; a reader that
re-aggregates it must group by `(X, Y, BITVAL, NVLED_V)`.

---

## `raw_data/Sweep_samples.csv`

Every individual burst reading, written only when `save_raw_samples` is on.
Optional — a reader must work without it.

`X, Y, BITVAL, NVLED_V, TIME, TYPE, SAMPLE_IDX, VALUE`

`TYPE` is `PM400`, `VLED` or `NVLED`; `SAMPLE_IDX` counts from 0 within the
point's burst. The current and light bursts are independent, so there is no
per-sample EQE — pair them randomly if you need a noise cloud.

---

## `raw_data/*_config.json`

```json
{ "settings": { "<every IQEConfig field>": <value> } }
```

`settings` is the complete `measure_iqe.IQEConfig` used for the run, so a run is
reproducible from its own output:

```python
from measure_iqe import IQEConfig
cfg = IQEConfig.from_dict(json.load(open(path))["settings"])
```

Fields a reader is most likely to want: `sample_name`, `roi`, `pm_wavelength`
(for the photon energy), `pixel_area_um2` (for current density), `bitvals`,
`source_mode`, `sweep_direction`, `vled_voltage`, `nvled_range_i`, `pm_range`.

**Readers must use `.get()` with a default for every key.** The field set grows
and shrinks as the measurement evolves, and older runs carry an older schema —
including some key names that no longer exist. Missing settings must degrade a
report, never crash it.

---

## Changing this format

Safe without coordination:

* **adding** an optional column at the end of a CSV
* **adding** a key to `settings`

Needs a change on both sides, plus a regenerated fixture in both repos:

* renaming or removing a column, or changing its unit, sign or meaning
* renaming a `TYPE` value
* changing the directory layout

Regenerate the fixture with:

```bash
python -m measure_iqe --simulate --set sample_name=FixtureArray \
  --set roi='[0,0,7,7]' --set block_size=4 --set pixels_per_block=1 \
  --set n_fine=6 --set warmup_ms=0
```

Then strip `RUN_PROVENANCE.json` from the copy: it records the commit and host of
whoever regenerated it, which would make the fixture churn on every machine.
