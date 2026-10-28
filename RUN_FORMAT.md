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

A run from a `pixel_list` is named `<YYMMDD-HHMM>_<sample_name>_LIST-<file stem>/`
instead. Its saved `roi` is the box around the listed pixels, and its FastScan files
hold only those pixels, in list order (2026-10-28+).

```
<YYMMDD-HHMM>_<sample_name>_ROI_(x1, y1, x2, y2)/
├── RUN_PROVENANCE.json                         what produced this run
├── <P>calibration_applied.yaml                 the calibration, copied (if any)
├── FastScan/
│   ├── <P>aggregated_bitval=<b>_mean_std.csv   one row per ROI pixel (one bias)
│   ├── <P>aggregated_bitval=<b>_v=<V>_mean_std.csv  one file per bias, when the
│   │                                           fast scan ran at several (2026-10-25+)
│   ├── <P>remeasure_bitval=<b>[_v=<V>]_mean_std.csv  the first fastscan_remeasure_n
│   │                                           pixels again, at the end of the pass
│   └── raw_data/
│       ├── <P>FastScan.csv                     long-format per-channel rows
│       ├── <P>FastScan_remeasure.csv           the same, for the re-measured pixels
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
* A fast scan at several biases writes one file per bias, named `_v=<V>` (for
  example `aggregated_bitval=15_v=-1.350_mean_std.csv`). Before 2026-10-25 all
  biases went into one file, told apart by `NVLED_V`. Concatenate, then split on
  `NVLED_V`, and both layouts read the same.
* `remeasure_bitval=*.csv` (since 2026-10-25, `fastscan_remeasure_n`, default 100)
  repeats the first N pixels of each fast-scan pass at its end, in the same loop. It
  has the fast-scan columns and is kept out of `aggregated_*`, so a fast-scan reader
  still sees every pixel once. Pair it with the fast scan on `X, Y, NVLED_V`.
* The sweep file holds each pixel's rows as one block, pixels in run order, since
  2026-10-25. `bias_major` writes bias by bias while it runs and reorders the file
  once when the sweep ends, so a run that died mid-sweep is still in bias order.
* `NVLEDSweep/` may hold more than one sweep group directory. Readers should
  glob its children rather than hardcoding `Sweep`.
* Every file is written incrementally and flushed per pixel. **A run directory
  from an interrupted measurement is valid and must be readable** — it simply
  holds fewer pixels. Never assume a pixel count or a complete voltage ramp.

---

## `aggregated_bitval=<b>_mean_std.csv`

The primary table. One row per measured point.

### Always present

The fast scan writes exactly these, in this order. The sweep writes them too, but
grouped by channel (below), so **find columns by name, never by position**.

| column | unit | meaning |
|---|---|---|
| `X`, `Y` | px | pixel coordinates, 0..511 |
| `BITVAL` | — | CMOS drive level, 0..15 |
| `NVLED_V` | V | cathode (channel B) bias, **negative**, rounded to 4 dp. In current-source mode this is the *measured* diode voltage instead of a setpoint. |
| `TIME_START`, `TIME_END` | s | seconds since the phase started, at the start and end of the burst the row reports. Before 2026-10-24 both were taken when the row was built, microseconds apart. |
| `NVLED_CURR_MEAN` | A | channel B current — **this is the LED current** |
| `NVLED_CURR_SEM` | A | **standard error of the mean** (σ/√N), not per-sample scatter |
| `VLED_CURR_MEAN` | A | channel A (anode) current — leakage cross-check only |
| `VLED_CURR_SEM` | A | standard error of the mean |
| `PM400_POWER_MEAN` | W | optical power |
| `PM400_POWER_SEM` | W | standard error of the mean |

**`QUADRANT`** (TL, TR, BL, BR, from `X` and `Y`) is written since 2026-10-25: the LAST column of the fast-scan file, so the older columns keep their positions, and right after `TIME_END` in the sweep file. For older runs, derive it: `X >= 256` is right, `Y >= 256` is bottom.

**`*_SEM` was called `*_STD` before 2026-10-24.** It held the same quantity, so a
reader renames `*_STD` to `*_SEM` when `*_SEM` is absent (`analyze_array.legacy_sem`).
No file written since carries `*_STD`, so a stale reader fails instead of reading an
SEM as a standard deviation.

### The sweep: one stats block per channel

For each channel `NVLED_CURR`, `VLED_CURR`, `PM400_POWER`, in that order, the sweep
writes six columns computed from the raw burst samples:

| suffix | meaning |
|---|---|
| `_MEAN` | arithmetic mean |
| `_MEDIAN` | median — **the centre the analysis uses** |
| `_SAMPSTD` | sample standard deviation (ddof = 1, Excel `STDEV.S`) |
| `_MADSTD` | 1.4826 × median absolute deviation: the robust σ, blind to an odd autorange spike |
| `_SEM` | `_SAMPSTD / √_N`, the uncertainty of the mean |
| `_N` | samples in the burst |

These are computed whether or not `save_raw_samples` is on; that setting decides
only whether `Sweep_samples.csv` is also written. Sweeps before 2026-10-24 had
`_MEDIAN`, `_SAMPSTD` (wrongly documented as the population std) and only
`NVLED_CURR_N`, and only with `save_raw_samples` on. A reader must treat the
non-required columns as optional and fall back to `*_MEAN`.

### Present only when `leak_reference` is on (default since 261013)

The same pixel re-measured at `LEAK_REF_BITVAL` (0: addressed, but not driven),
back to back with the lit point at the same bias. Each channel's block is followed
by its reference: `{ch}_REF`, then `{ch}_REF_MEAN` … `{ch}_REF_N` with the suffixes
above, from the reference's own burst.

| column | unit | meaning |
|---|---|---|
| `NVLED_CURR_REF` | A | ch B current with the pixel addressed and off — the value the analysis subtracts (the burst median, so it equals `NVLED_CURR_REF_MEDIAN`) |
| `VLED_CURR_REF` | A | ch A current in the same state (the burst median) |
| `PM400_POWER_REF` | W | optical power in the same state: the true local dark (the burst mean, so it equals `PM400_POWER_REF_MEAN`) |
| `*_REF_SEM` | A, W | SEM of the reference. Before 2026-10-24: `*_REF_STD`, from a MAD-based σ, and only for NVLED and PM400 |
| `LEAK_REF_BITVAL` | - | the drive level used as the reference |
| `LEAK_REF_TRIES` | - | frame writes it took to get the pixel off (1 = first one landed) |
| `LEAK_REF_FRAC` | - | `PM400_POWER_REF / PM400_POWER_MEAN`, the ratio the retry gated on |
| `LEAK_REF_CACHED` | - | `1` = reused the per-pixel reference, `0` = measured at this point |

A run taken before 2026-10-24 and rewritten in this format saved no reference samples.
Its reference block holds only `*_REF`, the centre it equals (`NVLED_CURR_REF_MEDIAN`,
`VLED_CURR_REF_MEDIAN`, `PM400_POWER_REF_MEAN`) and `NVLED`/`PM400` `*_REF_SEM`. The other
reference columns are `nan`: not recorded, not zero.

`LEAK_REF_CACHED = 1` says the four `*_REF` values came from the pre-pass rather than
from a read taken beside this point, so they are the same at every cached bias of that
pixel and `LEAK_REF_TRIES` is 0. The sweep references per point above
`leak_ref_per_point_above_v` (-1.0 V), where the reference is 5-62 % of the reading,
and caches below it, where one fixed per-pixel value costs under 0.14 % on `I_LED`
(0.20 % at -2.60 V). A reader that ignores the column still gets the right `I_LED`.
**Runs before 17 October 2026 lack it and referenced every point**, which a reader
must treat as `0`, not as unknown.

`LEAK_REF_TRIES > 1` means a frame write did not land and was re-sent. The FPGA does
not always write the frame correctly, at a rate of order a few percent per write, so
the reference is verified optically rather than assumed -- see §19 of the investigation
report. `LEAK_REF_FRAC` above `leak_ref_max_frac` (0.25) means it never landed and the
point is **not usable**: the subtraction removes most of the signal.

**Runs before 16 October 2026 have neither column and carry the uncorrected failure
rate.** A reader must treat their absence as "unknown", not as "the write succeeded",
and keep whatever contamination gate it already has.

### Present only when `pool_groups` is on (never by default)

A **pooled** target is K pixels lit together in one frame and measured as one point.
Its row carries the **sum** of its members' light and the **sum** of their current.

| column | unit | meaning |
|---|---|---|
| `POOL_K` | - | how many pixels this row is the sum of |
| `POOL_MEMBERS` | - | every member, `x:y` pairs separated by spaces |

`X`,`Y` is the **first member**, so the row still has a unique key and every
`groupby(["X","Y"])` keeps working. The members use `:` and not `,` because the
aggregated writer joins fields with a comma and does not quote.

What a reader must do with `POOL_K`:

* **EQE needs no correction.** It is `P/I`, and both are summed by the same K, so a
  pooled row's EQE is already the current-weighted mean of its members' EQEs.
* **Current density does.** `J = I/(K·area)`, or a pooled target reads K times too
  high and lands in the wrong place on every EQE(J) curve.
* **Pixel counts do.** A pooled target is one row and K pixels.

A reader that ignores the column entirely gets the right EQE and a wrong `J` --
the safe direction, but still wrong.

### `PM_RANGE_W`: the PM400 full scale the point was read on (W)

Since 2026-10-25 `pm_autorange` (default on) writes it on every fast-scan and sweep row,
as the REALISED hardware full scale, which the run learns from the meter at start.
The fast scan picks one range per bias pass from a 256-pixel probe; the sweep picks one
per bias from the brightest target of the previous bias, scaled by the measured P(V).
A reading at 95 % of full scale or more is re-read one range up, and the point keeps
that range. The leakage reference is always read on the lit point's range; a cached
reference is cached per (pixel, range). Same caveats as below: values are comparable
across ranges, their noise floor is not (it goes as range^0.74).

#### With `pm_autorange` off: only when `pool_pm_range_below_v` is set

The PM400 range that point was read on, in W, as requested (the realised hardware scale
is coarser -- `1e-7` lands on 470 nW on the S171C). Pooled targets switch to
`pool_pm_range` at or below the threshold bias; single pixels never switch. **A reader
comparing points across the boundary must check this column**: the optical floor goes
as `range^0.74`, so a decade coarser range is 5.45x noisier, and the two segments are
not equally precise. The lit-minus-reference subtraction removes the ranges' 1154 pW
offset difference, so the VALUES are comparable -- only their uncertainty differs.

### `NVLEDSweep/Sweep/<prefix>pm_range_check.csv`, with the same setting

Two rows per checked target, the same pooled group read on each range, with the columns `X, Y, POOL_K, NVLED_V, PM_RANGE_W, PM400_POWER_MEAN,
PM400_POWER_SEM, NVLED_CURR_MEAN, PM400_POWER_REF, NVLED_CURR_REF` (`PM400_POWER_STD`
before 2026-10-24). The ratio of the
two `PM400_POWER_MEAN` values is the gain step across the boundary, and the run logs
its median. It is a per-run verification of the +0.150 % measured once in 261019-1829
and -1833.

**Two groups of rows, told apart by `NVLED_V`.** The pre-flight pair sits at
`pool_pm_range_below_v` itself, on `pm_range_check_targets` targets, measured before the
sweep. The rows at the sweep **grid point** past the threshold are the real
verification: every pooled target, inside the bias loop, with the sweep's own dwell,
NPLC and reference. Use those. The grid-point rows are absent when the fine range could
not hold a pooled target at that bias, in which case the run says so and the switch is
unverified for that run.

### `FastScan/<prefix>screen_bitval=*.csv`, only when `fastscan_pool` is on

The pooled fast-scan **screen** writes one row per group of K pixels, with the same
columns as the fast scan plus `POOL_K`, `POOL_MEMBERS` and `SCREEN_FLAGGED` (1 = its
members were rescanned one by one). Power and current are the **sum** over the members.

**When this file exists, the ordinary `FastScan/<prefix>aggregated_*.csv` no longer
covers every pixel in the ROI.** It covers the rescanned subset: the members of
flagged groups plus any edge pixels no whole tile fit. A pixel in an unflagged group
is never measured on its own, and the only number for it is its group's sum divided by
`POOL_K`. A reader that needs per-pixel coverage must check for this file first and
refuse, or treat the screened members as estimates. `fastscan_pool` is 0 by default,
so an ordinary run has no screen file and the fast scan covers every pixel as before. **Every run before 17 October 2026, and every
run with `pool_groups: []`, lacks both columns, and absence means `POOL_K = 1`,
not unknown.** Pooling is off by default, so an ordinary run's columns are exactly
what they were.

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

* **`*_SEM` is the uncertainty of the mean, not a standard deviation.** It is
  `*_SAMPSTD / √N`. `*_SAMPSTD` and `*_MADSTD` are the per-sample spread. Use
  `*_SEM` for reliability gating, because heavier averaging genuinely lowers it.
  Older runs call it `*_STD`.
* **`*_MEAN` is not always the arithmetic mean.** In the fast scan, in every run
  before 2026-10-24 that had `save_raw_samples` off, and in every run older than
  that switch, the current `*_MEAN` holds a *robust median* (chosen to reject SMU
  autorange glitches). Every sweep since 2026-10-24 has a true `*_MEAN` and a
  separate `*_MEDIAN`. **Prefer `*_MEDIAN` when present, else `*_MEAN`.**
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
