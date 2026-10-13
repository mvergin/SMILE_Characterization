# SMILE_IQE_Analysis

Fit and report IQE/EQE from micro-LED array measurement runs produced by
[SMILE_Characterization](../SMILE_Characterization). This repo touches no
hardware and imports nothing from the measurement side — it reads run
directories off disk.

```
SMILE_Characterization  ──writes──▶  <run_dir>/  ──reads──▶  SMILE_IQE_Analysis
     (instruments, CLI)                 CSV + JSON              (fits, reports)
```

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

* **`*_STD` is the standard error of the mean** (σ/√N), not the per-sample
  spread. `*_SAMPSTD` is the spread.
* **Prefer `*_MEDIAN` over `*_MEAN` when present.** In older runs `*_MEAN`
  already held a robust median; when the `*_MEDIAN` columns exist, `*_MEAN` is
  the true arithmetic mean.
* **`NVLED_*` (channel B) is the LED current**, and it is negative — take
  `abs()`. `VLED_*` (channel A) is a leakage cross-check, not signal.
* **Subtract the median `DARK_PM400`** from the optical power before computing
  EQE, then clip at zero.
* A run from an interrupted measurement is valid and must stay readable; it just
  has fewer pixels.

---

## History

These files were extracted from `SMILE_Characterization`, which had grown to
hold the drivers, the measurement scripts and the analysis together. Their
history before the split lives in that repo.
