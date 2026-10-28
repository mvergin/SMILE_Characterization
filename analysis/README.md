# Analysis

Fit and report IQE/EQE from the micro-LED array measurement runs that
[`measure_iqe`](../README.md) writes. Nothing here touches hardware and nothing
imports the measurement side — these scripts read run directories off disk.

```
measure_iqe/  ──writes──▶  <run_dir>/  ──reads──▶  analysis/
 (instruments, CLI)         CSV + JSON             (fits, figures)
```

This was its own repo, `SMILE_IQE_Analysis`, until 16 October 2026. It was merged
in with `git subtree`, so its 47 commits are in this repository's history. Run the
scripts from either the repo root or from inside this directory — they resolve
their paths from the file, not the working directory.

That handoff is specified in **[RUN_FORMAT.md](RUN_FORMAT.md)**, which is kept
byte-identical in both repos. Read it before changing anything that parses a run.

---

## Setup

```bash
uv sync                      # or: pip install -r requirements.txt
```

Python 3.13. Dependencies: numpy, pandas, scipy, matplotlib. No instrument
drivers, no PyVISA, no FPGA libraries.

---

## Tools

### `analyze.py` — point it at a folder

```bash
python analysis/analyze.py results                 # every run underneath
python analysis/analyze.py "results/<run>"         # one run
python analysis/analyze.py results --dry-run       # what would run, and why
python analysis/analyze.py results --force         # redo everything
python analysis/analyze.py results --jobs 4        # fit processes for analyze_array
```

It picks the tools from what each run folder holds:

| folder holds | tool |
|---|---|
| `FastScan/` | `analyze_array.py` |
| `calibration_traces.npz` | `plot_calibration.py` |
| `traces.npz`, `switchmatrix.csv` or `isrc.csv` | `plot_transients.py` |
| `leakmatrix.csv` | `plot_leakmatrix.py` |
| `transistoriv.csv` | `plot_transistoriv.py` |
| `droop.csv` | `plot_droop.py` |

A tool runs again only if its source file or the run's input files changed since its
last clean finish. `ANALYSIS_STATE.json` in the run folder records the source hash
(SHA-256), the input fingerprint (path, size, mtime), the exit code and the time.
The output goes to the console and to `ANALYSIS_LOG.txt`. Each tool runs in its own
process, one after the other, so one run's crash or memory leak does not affect the
next. A finished `measure_iqe` run calls this script on itself.

Folders with only `cluster.csv` or `optfloor.csv` have no plotter and are not listed.

### `plot_transistoriv.py`, `plot_droop.py` — the `_PIXELS_` experiments

`plot_transistoriv.py` reads a `--what transistoriv` run: |I_B|(V) through shorted pixels
at bitvals 0, 1, 2, 4, 8, 15. `transistoriv.png` has every curve per bitval with the
median, the medians together, additivity (I15 against I1 + I2 + I4 + I8 − 3 I0), the
spread across pixels, and the current per bitval over bitval 1 against 1:2:4:8:15.
`transistoriv_summary.csv` has n, median, MAD, p10 and p90 per bitval and bias.

`plot_droop.py` reads a `--what droop` run: total current and light with N pixels lit,
against N × one pixel; per-pixel current and light over their N = 1 value; and P/I.
One row per bias, the random and the growing-patch order as two lines.

### `analyze_array.py` — the main report

Loads a run directory, enriches it (EQE, current density, leakage, reliability
mask), fits the ABC model per pixel, and writes a Markdown report with plots.

```bash
python analyze_array.py "<run_dir>"
python analyze_array.py "<results_root>" --all        # every run underneath
python analyze_array.py "<run_dir>" --area 100 --wavelength 440
```

| flag | meaning |
|---|---|
| `path` | a run directory, or a results root with `--all` |
| `--all` | process every run under `path` |
| `--area` | pixel area in µm² (default from the run's config) |
| `--thickness` | active region thickness in nm |
| `--wavelength` | emission wavelength in nm, for the photon energy |
| `--jobs` | processes for the per-pixel ABC fits (default: CPU count − 1; 1 = none) |

The per-pixel fits are cached in `REPORT_assets/IQE_EQE/abc_fit_cache.pkl`. The key
for each pixel is a hash of the fit source (`fit_abc` and its helpers) and that pixel's
exact J and EQE arrays. A plot or report change reuses every fit. A gate or `enrich`
change refits only the pixels whose reliable points it moved. A change to the fit code
refits all of them.

`REPORT_assets/IQE_EQE/pixel_summary.csv` has one row per swept pixel:

| columns | meaning |
|---|---|
| `nvled_curr_*`, `vled_curr_*`, `pm400_power_*` | the burst at the pixel's own EQE peak (`v_at_eqe_max`), as measured |
| `eqe_max_*` | EQE at that bias, leakage- and dark-corrected; std and MAD propagated from light and current |
| `iqe_max`, `iqe_fit_r2` | the per-pixel ABC fit's peak IQE (clean fits only) and its R² |
| `forward_voltage_mean`, `_median` | bias where J_LED first reaches 1 A/cm² from 0 V, interpolated in log J |
| `forward_leakage_*`, `forward_leakage_v` | the reference current (pixel addressed, off) at the sweep bias nearest −1.0 V |

`*` is `mean`, `median`, `std` (sample SD of the burst) and `mad` (1.4826 × MAD).
`REPORT_assets/pixelclassification/fastscan/` sorts every fast-scan pixel (first bias)
into groups:

| group | rule |
|---|---|
| `short` | dark, I > 100 µA |
| `open` | dark, I < 1 µA (leakage only) |
| `dark_conducting` | dark, 1–100 µA |
| `overcurrent` | I > 120 µA, lit or not: more than one pixel's drivers deliver |
| `low_eqe` | lit, P/I < 0.6 × median |
| `neighbour_on` | P ≥ 1.5 × median, P/I within 2 MAD |
| `high_resistance` | lit, P < 0.5 × median, P/I within 2 MAD |
| `typical` | P within median ± 1 MAD |
| `other` | lit, none of the above |

"Dark" is below a noise floor from the same scan. Medians are per quadrant. P/I uses the
current minus the quadrant's leakage (the median open pixel). Groups overlap, so
`pixel_classes.csv` gives every pixel a `primary` group (the first match, in table order)
and an `is_<group>` flag for each. `<group>.csv` lists every member, `summary.csv` the
counts. `classes_IP.png` is current against light by group, `classes_pie.png` the primary
groups. Any of these CSVs works as a `pixel_list` for `measure_iqe` and for the
`transistoriv` experiment.

`pixel_summary_interpolated.csv` has the same columns at each pixel's FITTED peak
(`j_at_peak` from the ABC fit) instead of its best sweep point. Each burst quantity is
interpolated between the two reliable points that bracket it, log-log in J. `v_at_peak`
replaces `v_at_eqe_max`, and `eqe_max_fit` is the model's own peak. Only clean fits
with the peak inside the measured J range get values.

`pixel_matrix_*.png` and `pixel_matrix_interpolated_*.png` plot every quantity against
every other, one point per pixel coloured by `quadrant`, histograms per quadrant on the
diagonal. Current and light are plotted per member: a pooled target (`pool_k` = K)
emits ~K x one pixel. The CSVs keep the target totals. Runs before 2026-10-24 saved no
reference samples, so their `forward_leakage` has only the median.

### `calc_iqe_eqe.py` — single-curve ABC fit

Fits one I–V–L curve without the full array machinery. Useful for a quick check
on a single pixel or an externally supplied curve.

### `benchmark_report.py` — sweep-parameter benchmark report

```bash
python benchmark_report.py "<..._BENCHMARK>"
```

Reads the `benchmark_metrics.csv` and per-variation run directories written by
`benchmark_sweep.py` in the measurement repo, fits each sweep for the peak-IQE
cross-check, detrends against the interleaved baselines, and writes
`benchmark_report/{table.md, benchmark.png}`.

The split is deliberate: measuring a variation needs the hardware, judging it
needs the ABC fit. The measurement repo writes the fit-free precision metrics
and stops; this script does the rest.

### `compare_runs.py` — old method against new, same pixels

```bash
python compare_runs.py OLD_RUN NEW_RUN --label-old "legacy" --label-new "rebuilt"
```

Measure one ROI with `configs/legacy_pre261014.yaml` and again with the defaults,
then run this. It matches the two runs pixel by pixel and writes, into
`NEW_RUN/comparison/`:

* `eqe_vs_J.png` — EQE(J) under each method, median over the pixels with the
  inter-quartile band. The headline figure.
* `iv.png` — the I–V each method reports, and the ratio between them
* `leakage.png` — the fraction of channel B that is not the LED, which is *why*
* `per_pixel.png` — peak IQE per pixel, old against new, so it is not one average
* `COMPARISON.md` — the numbers, with the figures embedded

It reads only what each run contains, so an old run with no leakage reference is
handled: its `I_LED` is the total cathode current, which is the point being made.

### `index_runs.py` — one CSV row per run

```bash
python analysis/index_runs.py                 # results/, demo/, charatests/, caprun/, results_georg/
python analysis/index_runs.py results -o my_index.csv
```

What each run is (sweep, fast scan, calibration, bench), how it was taken (order, dwell,
High-C, reference, pooling, code), the alive count at every fast-scan bias, and the IQE
numbers `analyze_array.py` left in the folder. Reads files only; ~40 s for 190 runs.

### `plot_calibration.py` — a `--calibrate` record at a glance

```bash
python analysis/plot_calibration.py "<run>_CALIBRATION"
```

Every capture against its own final value, current and light, with each pixel's t_target
and the recommended dwell. `measure_iqe --calibrate` runs it on its own record.

### `plot_transients.py` — settling curves and their spectra

```bash
python plot_transients.py "<..._CAPACITANCE>"
```

Reads `traces.npz` and the summary CSVs from `characterize_capacitance.py` in the
measurement repo, and writes into the same run directory:

* `transients.png` — `I(t)` after each voltage step, in seconds, with the settled
  value, the ±1% band and the 1% / 0.1% settling times marked. The y axis is
  clipped to 1.6x the settled value: the first samples overshoot by 10-20x and
  would otherwise set the scale and flatten everything worth seeing.
* `settling.png` — all captures as `|I - I∞| / I∞` on log-log, with the fitted
  exponent. **This is the one to read.** Straight parallel lines mean the approach
  is a power law (t^-1.3 on the bench), not an exponential, and where each crosses
  the 1% line is its settling time.
* `spectra.png` — amplitude spectrum of each trace's settled tail, mains line marked
* `isrc.png` — only for a run with an `isrc.csv`: channel B held as a **current**
  source. The forced quantity is the current, so here the *voltage* is the
  transient. Written from the ~2 Hz poll log, not from `traces.npz`, which `isrc`
  does not produce.

The time axis is linear in `transients.png` and log in `settling.png` on purpose.
An earlier version used `symlog`, which with a 0.5 s `linthresh` invented negative
time decades and made both unreadable.

The spectra answer one specific question: **is a sine riding on the transient?**
The SMU integrates `NPLC × 20 ms` and only an *integer* NPLC nulls the mains line,
so a characterisation run at NPLC 0.01 has no mains rejection at all while its
~943 Hz sample rate puts 50 Hz well inside Nyquist. The measurement script records
`spec_peak_hz`, `spec_peak_amp` and `spec_mains_amp` per capture; this draws the
spectrum behind those numbers, and prints them as a table first.

`traces.npz` holds one `(3, N)` array per capture — time, channel A, channel B —
thinned uniformly so the frequency content is preserved. The key encodes the
settings:

```
000_step_hc1_rNone_f-1.000_v-0.950_5
 |    |    |     |      |       |  |  decimation stride (every 5th sample kept)
 |    |    |     |      |       stepped TO this voltage
 |    |    |     |      stepped FROM this voltage
 |    |    |     measure range, None = not forced (in High-C the limit picks it)
 |    |    High-C: 1 on, 0 off
 |    experiment: step, or leak (which carries lit/off/blank instead)
 capture index, in run order
```

**The plots show channel B only** — `NVLED`, the cathode, which is the LED
current. Channel A is stored but not plotted: it carries ~466 µA of array
background, so the LED current is 0.1–0.8% of it and its per-sample noise is
~29 µA against channel B's 9 nA. Treat channel A as a supply-rail monitor, not
as a second measurement of the LED.

### `make_capacitance_figures.py` — the investigation report's figures

```bash
python analysis/make_capacitance_figures.py   # from the repo root
```

Builds all twelve figures for **`INVESTIGATION_REPORT.md`**, which is
**confidential and deliberately not in this repo** -- it lives only on the
measurement machine, at `C:\_D\programs\SMILE_IQE_Analysis\`. The root
`.gitignore` blocks the filename at any depth so it cannot be committed again.

The figures are gitignored too (`*_report/`), and on the bench machine
`analysis/investigation_report/` is a **directory junction** to the report's own
folder, so a script run from the repo writes its figure straight to where the
report links it. Recreate it after a fresh clone, or the report loses new figures
without saying so:

```
mklink /J analysis\investigation_report C:\_D\programs\SMILE_IQE_Analysis\investigation_report
```

This script reads the `charatests/` and `caprun/` runs **off the bench machine's
disk**. Neither is tracked, so this script only runs where the data was measured. It
prints every number it plotted, so the report text cannot drift from the data. Read the
report first if you are wondering why the sweep settles for seconds, why High-C is
mandatory, or why every rising-side EQE before 261014 is unusable.

### Smaller tools

| script | what it answers |
|---|---|
| `find_knee_jump.py <run_dir>` | which pixels show the NPLC-switch EQE discontinuity at the knee |
| `subsample_convergence.py <run_dir>` | how many measured pixels are needed to recover the array's spatial structure |
| `cleanfit_tracker.py` | one table of clean per-pixel-fit % per run, to watch progress toward the 90% goal |
| `make_investigation_figures.py` | regenerates the figures for the investigation report from real run data |

---

## Tests

```bash
python test_run_format.py
```

Seven checks that this repo can still read a run directory: the promised
aggregated columns, the raw CSV's channel and `DARK_*` rows, the config JSON,
`_load_aggregated` → `_dark_power` → `enrich` end to end, and the fallback when
the optional `*_MEDIAN` columns are absent (older runs).

`tests/fixture/` is a real run from `python -m measure_iqe --simulate`, committed
as the contract. The measurement repo holds the same fixture and asserts it still
*writes* that shape; this repo asserts it can still *read* it. A format change
that breaks one side fails on both.

To regenerate it after an intentional format change, run the command at the
bottom of RUN_FORMAT.md in the measurement repo and copy `tests/fixture/` over.

---

## Conventions worth knowing

These bite everyone once. RUN_FORMAT.md has the full list.

* **`*_SEM` is the standard error of the mean** (σ/√N), not the per-sample
  spread. `*_SAMPSTD` and `*_MADSTD` are the spread. Runs before 2026-10-24 call
  it `*_STD`; `enrich()` renames it (`legacy_sem`).
* **Prefer `*_MEDIAN` over `*_MEAN` when present.** In older runs `*_MEAN`
  already held a robust median; when the `*_MEDIAN` columns exist, `*_MEAN` is
  the true arithmetic mean.
* **`NVLED_*` (channel B) is the LED current**, and it is negative — take
  `abs()`. `VLED_*` (channel A) is a leakage cross-check, not signal: measured
  261013 it sits at ~466 µA of array background with ~29 µA of noise, against
  channel B's 9 nA.
* **Subtract `NVLED_CURR_REF` when it is present.** Channel B reads the *total*
  cathode current, and at −0.80 V 58% of it is backplane leakage, not injection.
  `enrich()` does this and sets `I_LED`, `LEAK_FRAC` and `I_NVLED_REF`. Runs
  without the column keep the old meaning (`I_LED = |I_NVLED|`) and get
  `LEAK_FRAC = NaN` — those runs' rising sides cannot be corrected afterwards,
  because the reference was never recorded.
* **Subtract the median `DARK_PM400`** from the optical power before computing
  EQE, then clip at zero.
* A run from an interrupted measurement is valid and must stay readable; it just
  has fewer pixels.

---

## History

These files were extracted from `SMILE_Characterization`, which had grown to
hold the drivers, the measurement scripts and the analysis together. Their
history before the split lives in that repo.
