# SMILE_Characterization

Measure the micro-LED array of the SMILE platform, developed in the EU project
*Scalable Structured Micro Illumination Light Engines*. This repo talks to the
instruments and writes measurement data. **It does not analyse it.**

```
SMILE_Characterization  ──writes──▶  <run_dir>/  ──reads──▶  SMILE_IQE_Analysis
     (instruments, CLI)                CSV + JSON             (fits, reports)
```

Fitting, plotting and reporting live in the separate
[SMILE_IQE_Analysis](../SMILE_IQE_Analysis) repo. The two share no code — a run
directory on disk is the whole interface, specified in
**[RUN_FORMAT.md](RUN_FORMAT.md)** and kept byte-identical in both repos.

| piece | what it is |
|---|---|
| `measure_iqe/` | **the measurement: library + CLI** — fast scan, pixel selection, IQE sweep |
| `smile/`, `instruments/` | SMILE FPGA driver and the PM400 / Keithley 2602B instrument drivers |
| `measurement/`, `data/`, `smile_postprocess.py` | **reference only, unmaintained** — the transient / ultra-fast / hardware-paced-TSP techniques that predate `measure_iqe`. See [MEASUREMENT_MODES.md](MEASUREMENT_MODES.md) |
| `benchmark_sweep.py`, `autozero_bench.py`, `sweep_variants.py` | parameter benchmarks built on the `measure_iqe` API |
| `characterize_capacitance.py` | measures the array's ~50 µF capacitance and what it does to a low-current reading |

---

## Setup

```bash
uv sync                      # or: pip install -r requirements.txt
```

Python **3.13** (`.python-version`). Dependencies are pinned in
`pyproject.toml` and compiled into `requirements.txt`.

Hardware needed for a real run: a Thorlabs PM400 power meter, a Keithley 2602B
SMU and the SMILE FPGA board. **You need none of them to try the tool** — every
path runs against a built-in ABC-model simulation rig:

```bash
python -m measure_iqe --simulate --set roi='[0,0,15,15]'
```

---

## Module usage

The package is importable and has no module-level state: every parameter lives
on an `IQEConfig`, and nothing is read from the environment.

```python
from measure_iqe import IQEConfig, run_iqe_measurement

cfg = IQEConfig(
    sample_name="Array38_IQE",
    roi=(32, 32, 63, 63),        # or "FULL" / "TL" / "TR" / "BL" / "BR"
    base_dir="D:/runs",
    sweep_direction="low_to_high",
)
result = run_iqe_measurement(cfg)        # add simulate=True for no hardware

print(result.run_dir)      # Path to <stamp>_<sample>_ROI_(...)
print(result.n_pixels)     # pixels selected for the full sweep
print(result.n_points)     # sweep points written
print(result.config)       # the IQEConfig actually used
```

Load the configuration from a file instead:

```python
from measure_iqe import IQEConfig, run_iqe_measurement

cfg = IQEConfig.from_file("config.example.yaml")     # .yaml, .yml or .json
run_iqe_measurement(cfg, simulate=True)
```

Change one knob on an existing config without touching the rest — the dataclass
is ordinary, so `dataclasses.replace` works:

```python
import dataclasses
from measure_iqe import IQEConfig

base = IQEConfig.from_file("bench.yaml")
lean = dataclasses.replace(base, n_fine=30, vramp_lowv_n_pm=8)
```

Capture the log instead of printing it:

```python
lines: list[str] = []
result = run_iqe_measurement(cfg, simulate=True, log=lines.append)
```

Drive the phases yourself when you want just one of them — this is what
`benchmark_sweep.py` does:

```python
import measure_iqe as m

cfg = m.IQEConfig(roi="TL")
pm, smu, smile = m.connect(cfg, log=print, simulate=True)
try:
    m.configure_common(pm, smu, cfg, print)
    smu.enable_output("a", True)
    smu.enable_output("b", True)
    selected = m.run_fast_scan(pm, smu, smile, *cfg.resolve_roi(), run_dir, cfg, print)
    n = m.run_sweep_voltage(pm, smu, smile, selected, run_dir, cfg, print)
finally:
    m.shutdown(pm, smu, smile, cfg)
```

### Public API

| name | purpose |
|---|---|
| `IQEConfig` | every measurement parameter, with defaults; `from_file`, `from_dict`, `to_dict`, `write_file`, `resolve_roi`, `validate`, `photon_energy_j` |
| `run_iqe_measurement(config=None, *, simulate, warmup_diag, base_dir, run_dir, log)` | the whole flow; returns `IQEResult` |
| `IQEResult` | `(run_dir, n_pixels, n_points, config)` |
| `connect(cfg, log, simulate=False)` / `shutdown(pm, smu, smile, cfg)` | open / safely close the instruments |
| `configure_common`, `configure_voltage_source_b`, `set_nplc`, `set_measure_range_b` | instrument setup primitives |
| `run_fast_scan`, `run_sweep_voltage`, `run_sweep_current`, `run_warmup_diag` | the individual phases |
| `select_sweep_pixels`, `build_fine_voltages` | pixel selection and sweep-point placement |
| `acquire_point`, `burst_stats` | one overlapped SMU+PM measurement, and burst statistics |
| `make_logger`, `make_run_dir` | encode-safe timestamped logger; timestamped run directory |
| `NAMED_ROIS`, `SMU_I_RANGES` | ROI presets and the 2602B current ranges |

Everything writes CSVs only. Fit and plot them from the analysis repo:

```bash
python analyze_array.py "<run_dir>"        # SMILE_IQE_Analysis
```

---

## CLI usage

```bash
python -m measure_iqe [options]
```

| flag | argument | effect |
|---|---|---|
| `--config` | `PATH` | load a `.json`, `.yaml` or `.yml` config file |
| `--set` | `KEY=VALUE` | override one field; repeatable; applied on top of `--config` |
| `--out` | `DIR` | output root, overriding `base_dir` |
| `--simulate` | — | run against the built-in ABC-model rig, no hardware |
| `--warmup-diag` | — | run the warm-up/conditioning diagnostic instead of the sweep |
| `--dump-config` | `PATH` | write the effective config to `PATH` (`.json`/`.yaml`) and exit |
| `--print-config` | — | print the effective config as JSON and exit |
| `--list-params` | — | list every field with its type and default, and exit |
| `-h`, `--help` | — | usage |

`--set` values are parsed as JSON, so types survive; anything that is not valid
JSON stays a string (`--set sweep_direction=low_to_high` needs no quoting).

```bash
# dry-run the whole flow on a 16x16 ROI, nothing attached
python -m measure_iqe --simulate --set roi='[0,0,15,15]'

# real run from a config file, output elsewhere
python -m measure_iqe --config bench.yaml --out D:/runs

# a config file plus two tweaks
python -m measure_iqe --config bench.yaml --set n_fine=60 --set sweep_direction=low_to_high

# record exactly what a run would use, then run it
python -m measure_iqe --config bench.yaml --set n_fine=60 --dump-config run.yaml
python -m measure_iqe --config run.yaml
```

**Exit codes:** `0` success, `2` bad configuration (the message names the key),
`130` interrupted with Ctrl-C — partial results stay on disk.

### Output

```
<base_dir>/<YYMMDD-HHMM>_<sample_name>_ROI_(x1, y1, x2, y2)/
├── FastScan/
│   ├── aggregated_bitval=<b>_mean_std.csv     one row per ROI pixel
│   └── raw_data/{FastScan.csv, FastScan_config.json}
└── NVLEDSweep/Sweep/
    ├── aggregated_bitval=<b>_mean_std.csv     one row per sweep point
    └── raw_data/{Sweep.csv, Sweep_samples.csv, Sweep_config.json}
```

`*_config.json` holds the complete `IQEConfig` under `"settings"`, so a run is
reproducible from its own output. `--warmup-diag` writes `warmup_diag.csv`
instead of the sweep.

Aggregated columns: `X, Y, BITVAL, NVLED_V, TIME_START, TIME_END,
NVLED_CURR_MEAN, NVLED_CURR_STD, VLED_CURR_MEAN, VLED_CURR_STD,
PM400_POWER_MEAN, PM400_POWER_STD` — plus `*_MEDIAN`, `*_SAMPSTD` and
`NVLED_CURR_N` when `save_raw_samples` is on. `*_STD` is the **standard error of
the mean** (σ/√N), not the per-sample scatter.

Writes are incremental and flushed per pixel, on a background thread for the
voltage sweep, so an interrupted run leaves directly analysable files.

---

## Configuration reference

Every field below is a constructor argument of `IQEConfig`, a key in a
JSON/YAML config file, and a `--set` target. All are optional. Unknown keys and
wrong types are rejected with an error naming the key. `config.example.yaml` is
an annotated copy of these defaults.

Comments in `measure_iqe/config.py` keep the historical UPPERCASE names of some
parameters (`BLOCK_SIZE` is today's `block_size`) because they quote the bench
runs that pinned each value.

#### Run identity & output

| parameter | type | default | meaning |
|---|---|---|---|
| `sample_name` | `str` | `'Array38_IQE'` | base_dir is relative to the working directory unless you pass an absolute path. Nothing outside it is ever written. |
| `base_dir` | `str` | `'results'` | output root; a timestamped run dir is created inside |

#### ROI & pixel selection

| parameter | type | default | meaning |
|---|---|---|---|
| `roi` | `tuple[int, int, int, int] \| str` | `(32, 32, 63, 63)` | (x1,y1,x2,y2) or "FULL"/"TL"/"TR"/"BL"/"BR" |
| `sweep_select` | `str` | `'blocks'` | "blocks" \| "spatial" \| "stride" |
| `block_size` | `int` | `8` | blocks mode: block edge in pixels (ROI → ~ROI/BLOCK_SIZE blocks/axis) |
| `pixels_per_block` | `int` | `4` | blocks mode: alive pixels sampled per block |
| `block_seed` | `int` | `0` | blocks mode: RNG seed (vary across runs to sample new pixels) |
| `min_pixel_separation` | `int` | `2` | blocks mode: no two measured pixels may sit closer than this (Chebyshev, so 2 = never sharing an edge or a corner). Without it, each block's farthest-point walk ran to the block's own border and the block next door … |
| `sweep_fraction` | `float` | `0.1` | spatial mode: fraction of ALIVE pixels to sweep |
| `min_sweep_pixels` | `int` | `6` | spatial mode: floor on the count (so tiny ROIs still map) |
| `nth_pixel_sweep` | `int` | `5` | stride mode: sweep every Nth alive pixel |
| `max_sweep_pixels` | `int` | `40` | hard cap (spatial/stride only; blocks controls its own count) |
| `fastscan_bitval` | `int` | `15` | bitval for the alive/dead fast scan (brightest) |

#### Source mode

| parameter | type | default | meaning |
|---|---|---|---|
| `source_mode` | `str` | `'voltage'` | "voltage": ch B sources voltage, adaptive current-targeted sweep. PROVEN on Array38 — the 10 mA voltage sweep captured the IQE peak (J≈6, ~59%). USE THIS. "current": ch B sources (sinks) current, measures voltage. … |

#### Current-source sweep (source_mode = "current")

| parameter | type | default | meaning |
|---|---|---|---|
| `current_sweep_low_a` | `float` | `3e-07` | lowest LED current setpoint (1 µA → J ≈ 1) |
| `current_sweep_high_a` | `float` | `0.00012` | highest setpoint (~120 µA; capped by V compliance) |
| `current_sweep_n` | `int` | `50` | log-spaced current setpoints |
| `nvled_v_limit` | `float` | `3.2` | \|voltage\| compliance while sourcing current (V) |
| `current_sweep_bitval` | `int` | `15` | single bitval — CMOS fully open, we set I directly |
| `current_source_range_i` | `float` | `0.001` | fixed source current range (covers 120 µA; gives good low-current resolution). Tune on the bench. |
| `measure_v_range` | `float` | `6.0` | ch B voltage measure range (covers ±3.2 V) |

#### Voltage-source sweep (source_mode = "voltage", DEFAULT)

| parameter | type | default | meaning |
|---|---|---|---|
| `bitvals` | `tuple[int, ...]` | `(15,)` | single bitval — densify low-J via spacing instead |
| `fine_sweep_mode` | `str` | `'voltage'` | Fine-sweep mode: "voltage" — plain UNIFORM voltage ramp NVLED_V_COARSE_START → STOP (no coarse map). Hits every voltage including the steep turn-on, so the rising side is densely sampled. Simple, robust — this is … |
| `nvled_v_coarse_start` | `float` | `-0.8` | least-negative sweep voltage (just below turn-on) |
| `nvled_v_coarse_stop` | `float` | `-2.6` | most-negative sweep voltage (below saturation knee) |
| `n_coarse` | `int` | `26` | coarse pre-sweep points |
| `coarse_settle_ms` | `float` | `6.0` |  |
| `coarse_n_smu` | `int` | `4` |  |
| `low_current_target_a` | `float` | `1e-06` | lowest LED current the sweep targets (~J 1, the rise foot) |
| `high_current_frac` | `float` | `0.97` |  |
| `n_fine` | `int` | `45` | fine voltage-sweep points (was 45; more, single bitval) |
| `lowj_weight` | `float` | `1.5` | >1 concentrates fine points toward LOW current (the SRH rising side J≈1–5 that pins peak IQE). 1.0 = uniform log. |
| `vramp_knee_v` | `float \| None` | `-1.3` | split voltage between the dense (rising) and sparse (droop) halves |
| `vramp_low_frac` | `float` | `0.6` | fraction of points placed in the rising-side segment |
| `sweep_direction` | `str` | `'high_to_low'` | "low_to_high" \| "high_to_low" |

#### Pixel warm-up / conditioning

| parameter | type | default | meaning |
|---|---|---|---|
| `warmup_ms` | `float` | `1500.0` | 0 = off. Dwell at WARMUP_NVLED_V before each sweep. |
| `warmup_nvled_v` | `float` | `-2.6` | warm-up bias (high current, clears the parasitic) |
| `warmup_diag_v` | `float` | `-0.85` | ≈J 1: stresses current readout, light still > dark |
| `warmup_diag_n_pixels` | `int` | `3` | how many of the selected pixels to characterise |
| `warmup_diag_seconds` | `float` | `30.0` | log duration per phase (fresh turn-on / conditioned) |
| `warmup_diag_nplc` | `float` | `1.0` |  |
| `warmup_diag_n_smu` | `int` | `8` |  |
| `warmup_diag_n_pm` | `int` | `4` |  |

#### Adaptive per-point effort by current zone

| parameter | type | default | meaning |
|---|---|---|---|
| `adaptive_threshold_a` | `float` | `1e-05` | 10 µA: boundary between the low-J and high-J zones |
| `lowj_nplc` | `float` | `2.0` |  |
| `lowj_n_pm` | `int` | `8` |  |
| `lowj_n_smu` | `int` | `16` | low J: integrate the CURRENT harder |
| `highj_nplc` | `float` | `0.5` |  |
| `highj_n_pm` | `int` | `4` |  |
| `highj_n_smu` | `int` | `8` | high J: fast |
| `lowj_settle_ms` | `float` | `200.0` |  |
| `highj_settle_ms` | `float` | `15.0` | fine low-J range needs more settle time |

#### Instruments

| parameter | type | default | meaning |
|---|---|---|---|
| `pm400_addr` | `str` | `'USB0::0x1313::0x8075::P5006526::INSTR'` |  |
| `smu_addr` | `str` | `'USB0::0x05E6::0x2602::4649721::INSTR'` |  |

#### PM400

| parameter | type | default | meaning |
|---|---|---|---|
| `pm_wavelength` | `float` | `440.0` |  |
| `pm_range` | `float` | `1e-07` | W, fixed (no autorange) |
| `pm_averaging` | `int` | `3` | internal averaging per reading |

#### VLED (ch A) — constant LED anode drive

| parameter | type | default | meaning |
|---|---|---|---|
| `vled_voltage` | `float` | `1.8` |  |
| `vled_compliance` | `float` | `1.0` |  |
| `vled_range_i` | `float` | `0.01` |  |

#### NVLED (ch B) — voltage-mode current range

| parameter | type | default | meaning |
|---|---|---|---|
| `nvled_compliance` | `float` | `0.5` | current compliance (voltage mode) |
| `nvled_range_i` | `float` | `0.01` | default fixed measure current range (coarse pre-sweep, fast scan, and any point with no current target). |
| `perpoint_range` | `bool` | `False` | False = fixed NVLED_RANGE_I (10 mA): consistent (offsets and all) so the curve SHAPE comes through — what the old clean runs used. True = per-point software ranging (accurate value, but range offsets/settling … |
| `probe_range_i` | `float` | `0.001` | wide probe range: covers the device max (~175 µA), never overflows, gives the magnitude to pick the real range from |
| `probe_n_smu` | `int` | `4` | quick samples for the magnitude probe |
| `range_settle_ms` | `float` | `50.0` | settle after a measure-range change before reading |
| `highc_b` | `bool` | `False` | 2602B High-C mode on ch B. The driver guards High-C on B (raises), so we enable it via TSP. High-C adds settling delays (slower) but damps a capacitive load that can make the low current ranges ring/never settle — … |
| `sweep_autozero` | `str` | `'AUTO'` | AUTO \| ONCE \| OFF |
| `save_raw_samples` | `bool` | `True` | Save every individual burst sample (not just the per-point summary) to a raw-samples CSV, and write mean/median/std/SEM (not just one summary) into the aggregated file. Off → old behaviour. |

#### Integration time (NPLC) per phase

| parameter | type | default | meaning |
|---|---|---|---|
| `fastscan_nplc` | `float` | `0.05` | Now the fast-scan CURRENT matters: at −1.35 V the fast scan is a full-coverage peak-IQE proxy (EQE=P/I on the EQE plateau), so its current noise sets the proxy quality. Sweep test (260704-1456/1618/ 1659): NPLC … |
| `fastscan_n_smu` | `int` | `4` | SMU samples/pixel in the fast scan |
| `coarse_nplc` | `float` | `0.1` | voltage-mode coarse I(V) map: rough, quick |
| `sweep_nplc` | `float` | `1.0` | the accurate IQE data (raise toward 2–5 for very low J) |

#### Settle + averaging

| parameter | type | default | meaning |
|---|---|---|---|
| `settle_ms` | `float` | `15.0` | wait after each setpoint before averaging |
| `fastscan_settle_ms` | `float` | `10.0` | fast scan settle after lighting a pixel. Back to 10 ms (was 5) — the 5 ms save wasn't worth risking a not-fully-settled optical read on the liveness classifier. |
| `fastscan_random_order` | `bool` | `True` | visit fast-scan pixels in a shuffled order (not a raster). A raster correlates any residual/crosstalk with spatial position; randomizing turns it into position-independent noise. |
| `fastscan_order_seed` | `int` | `0` | seed for the shuffle (reproducible) |
| `fastscan_blank_between` | `bool` | `True` | send a black (all-off) frame before lighting each pixel, so exactly ONE pixel is ever emitting when the PM reads — guards against a lingering previous pixel / accumulated frame inflating the optical power (the ~71 nW … |

#### Dead-pixel early abort (voltage sweep)

| parameter | type | default | meaning |
|---|---|---|---|
| `dead_abort` | `bool` | `True` | Liveness is gated by the fast scan, which has an optical background floor that can mark a dead edge pixel "alive" (run 1405: 3/4 corner pixels read ~71 nW in the fast scan but 0.05 nW in the sweep). The sweep runs … |
| `dead_abort_nw` | `float` | `2.0` | absolute floor (nW): first-point power below this ⇒ dead |
| `dead_abort_frac` | `float` | `0.1` | or below this fraction of the running median of live pixels |
| `thermal_settle_ms` | `float` | `100.0` | before the fine sweep, sit at the FIRST point this long so the pixel reaches its starting-current thermal steady state (clears heat left by the coarse pre-sweep) |
| `n_smu_samples` | `int` | `8` | SMU samples averaged per point |
| `n_pm_samples` | `int` | `12` | PM400 readings averaged per point |

#### Per-segment effort (fine_sweep_mode="voltage")

| parameter | type | default | meaning |
|---|---|---|---|
| `vramp_perseg_effort` | `bool` | `True` | The knee (VRAMP_KNEE_V) splits not just point DENSITY but measurement EFFORT. The low-V / rising-side segment (\|V\| < \|knee\|) has faint light near the dark floor, so take more PM reads there (σ_light ∝ 1/√N_PM); the … |
| `vramp_lowv_nplc` | `float` | `2.0` |  |
| `vramp_lowv_n_pm` | `int` | `32` |  |
| `vramp_lowv_n_smu` | `int` | `12` |  |
| `vramp_lowv_settle_ms` | `float` | `20.0` |  |
| `vramp_highv_nplc` | `float` | `0.5` |  |
| `vramp_highv_n_pm` | `int` | `8` |  |
| `vramp_highv_n_smu` | `int` | `8` |  |
| `vramp_highv_settle_ms` | `float` | `15.0` |  |

#### Dead-pixel threshold (fast scan)

| parameter | type | default | meaning |
|---|---|---|---|
| `nvled_v_fastscan` | `float` | `-1.35` | single fast-scan bias — does DOUBLE duty. At −1.35 V the pixels sit at J≈25 A/cm², which is dead on the (broad, flat) EQE peak plateau: EQE is within ~1% of its peak across J≈8–30, so the fast-scan EQE=P/I here ≈ … |
| `dead_threshold_frac` | `float` | `0.2` | a pixel is alive if P ≥ this × the "typical lit" level |
| `fastscan_alive_pctl` | `int` | `90` | "typical lit" level = this percentile of the fast-scan power. NOT the median: the median collapses to the noise floor on a majority-DEAD ROI (then ~everything is mislabelled alive — e.g. quadrant 260703-1307 was 57% … |
| `pixel_area_um2` | `float` | `100.0` |  |
| `smu_display_off` | `bool` | `False` |  |

### Valid ranges and constraints

`IQEConfig.validate()` runs before any hardware is touched and rejects:

| field | accepted |
|---|---|
| `roi` | `(x1, y1, x2, y2)` with each value `0..511`, or `"FULL"`, `"TL"`, `"TR"`, `"BL"`, `"BR"` |
| `source_mode`, `fine_sweep_mode` | `"voltage"`, `"current"` |
| `sweep_select` | `"blocks"`, `"spatial"`, `"stride"` |
| `sweep_direction` | `"low_to_high"`, `"high_to_low"` |
| `sweep_autozero` | `"AUTO"`, `"ONCE"`, `"OFF"` |
| `bitvals` | non-empty; each value `0..15` |
| `pm_wavelength` | `> 0` nm |
| `n_fine`, `n_coarse` | `>= 2` |
| `vramp_low_frac` | `0.0 .. 1.0` |
| `min_pixel_separation` | `>= 1` (1 disables the guard) |
| `fastscan_alive_pctl` | `0 .. 100` |
| `pixel_area_um2` | `> 0` |

Physical limits the validator does *not* police, because the hardware does:
`nvled_compliance` is the SMU current compliance, `nvled_v_limit` the voltage
compliance, and `nvled_range_i` / `probe_range_i` must be one of
`SMU_I_RANGES` (1e-7 … 1e-2 A) for the 2602B to accept them.

### The three knobs that matter most

* **`source_mode`** — keep `"voltage"`. `"current"` does not work on this
  device: the +1.8 V anode bypasses channel B through a leakage path, so the
  cathode current setpoint never becomes the LED current.
* **`sweep_direction`** — `"high_to_low"` starts hot and conditioned;
  `"low_to_high"` measures the rising side coolest. Run both on the same pixels
  to gauge self-heating.
* **`vramp_lowv_n_pm` / `vramp_lowv_nplc`** — the low-J precision budget. The
  light is the limiting channel at low current, so spend on `*_n_pm` before
  `*_nplc`.

### How blocks mode samples

`sweep_select="blocks"` tiles the ROI into `block_size`x`block_size` blocks and
takes `pixels_per_block` alive pixels from **each** block, so every block
contributes equally and per-block / row-band / column-band statistics are
comparable. Within a block the picks are spread by farthest-point sampling from
a random start, which `block_seed` varies — change the seed between runs to
measure new pixels and accumulate coverage of the array.

Two rules keep the sample genuinely spatial rather than only nominally
stratified:

* Each block's walk is pushed away from the pixels already chosen in
  **neighbouring** blocks, not just its own, so picks drift towards block
  interiors instead of piling onto the shared edges.
* `min_pixel_separation` (default 2, Chebyshev) is a hard floor: no two measured
  pixels ever share an edge or a corner. A block without that many
  well-separated alive pixels contributes **fewer** rather than a touching pair,
  and the run log says how many blocks fell short. Raise it to spread the sample
  further; set it to 1 to allow neighbours.

The log line reports the realised spacing, so you can see what you got:

```
selected 256 pixels: 8x8 blocks (~8px), 4/block, 64 non-empty blocks, seed 0 ...
spacing: nearest-neighbour min 2 px, median 3 px (target >= 2); 0 touching pixel(s)
```

---

## Tests

```bash
python test_measure_iqe.py       # no pytest needed; also collects under pytest
```

The checks run the whole acquisition against the simulation rig: every source
mode, every pixel-selection mode, the warm-up diagnostic, the dead-pixel abort,
the blocks-mode spacing guarantees, the CLI, and the config round trip.

```bash
python test_run_format.py        # the contract with SMILE_IQE_Analysis
```

This one asserts a fresh run still writes the layout and columns that
RUN_FORMAT.md promises, that the committed `tests/fixture/` has not drifted from
what the code produces, and that an interrupted run stays readable.
`tests/fixture/` is a real simulated run; the analysis repo holds the same
fixture and asserts it can still *read* that shape, so a format change that
breaks one side fails on both.

To change the format: edit RUN_FORMAT.md, copy it to the analysis repo,
regenerate the fixture with the command at the bottom of that file, copy the
fixture across, and update both tests.

---

## The 50 µF problem

The array carries a large capacitor in parallel with the LED. Traversing the
rising-side span (−0.8 → −1.3 V) moves `C·ΔV ≈ 25 µC` of charge. At LED currents
(1–10 µA) that is **seconds per pixel**; at the SMU's compliance it is
microseconds. Which of those supplies the charge decides whether a low-J current
reading is right, and it is the likely cause of:

* the direction hysteresis between `low_to_high` and `high_to_low` (charging vs
  discharging the cap adds to or subtracts from the LED current);
* the fine current ranges reading low J *wrong*, not merely noisily;
* the current oscillating in the SMU buffer at low NPLC (loop instability into a
  capacitive load — which is what High-C mode exists to fix);
* possibly the "slow non-radiative parasitic current" that `--warmup-diag` found,
  since 25 µC at ~1 µA is ~25 s and a high-current dwell settles the cap fast.

```bash
python characterize_capacitance.py --what all --highc-both
python characterize_capacitance.py --what optfloor          # 2 min, no pixels needed
```

**That C is the same everywhere is an assumption**, so every pixel-dependent
experiment runs on `--per-quadrant` pixels (default 3) in each of the four
quadrants. They are found by scanning a window at each quadrant's centre and
handing the alive pixels to `select_sweep_pixels` in blocks mode, so the picks
inherit the minimum-separation guarantee. Each pixel is then checked for light
immediately before its long run, and a dead one is skipped and reported. Pass
`--pixels X,Y …` to characterise specific pixels instead.

| experiment | what it settles |
|---|---|
| `step` | I(t) after a voltage step at the lowest NPLC, per voltage × range × High-C state. Settled value, time to 1% and 0.1%, overshoot, ripple. |
| `cap` | `C` per pixel, from the step transient: integrate the current above its settled level, so `Q/ΔV` is the charge the capacitor took. Run lit *and* blanked, so the array-common part separates from anything local. |
| `isrc` | Channel B sinks a set current and **holds for minutes**. If chA converges toward chB's setpoint as the voltage settles, there is no bypass path and the earlier "current-source mode does not work" conclusion was the capacitor. |
| `stepdir` | One voltage reached from a small and from a large step. Parasitic conduction depends on time since turn-on; a capacitor depends on the charge moved. If settling scales with \|ΔV\|, it is the capacitor. |
| `decay` | Charge the cap, then source 0 A so the node floats and discharges through the LED, logging V(t) and light. `I = C·dV/dt`. Run lit and blanked; the difference is the LED current. See below. |
| `optfloor` | The PM400 noise floor vs averaging, array blanked. Decides how low in J you can go at all. |

`--simulate` exercises the plumbing only — the built-in rig has no capacitance
model, so it reports instant settling and cannot model the decay.

### How low in J can you go?

On a 10×10 µm pixel, `J [A/cm²] = I [A] / 1e-6`, so J = 1 is 1 µA and **J = 0.1
is 100 nA**. The SMU reaches that: 100 nA is the lowest measure range, and the
1 µA range reads it at 10% of full scale.

The light is the binding limit, not the current. Scaling from the −1.35 V
fast-scan point (~22 nW at 25 µA):

| J | I | optical power | …if EQE is down 2× at low J |
|---|---|---|---|
| 1 | 1 µA | ~880 pW | ~440 pW |
| 0.3 | 300 nA | ~264 pW | ~132 pW |
| 0.1 | 100 nA | ~88 pW | ~44 pW |
| 0.03 | 30 nA | ~26 pW | ~13 pW |

**Measured** (`optfloor`, 261013, array blanked, 12 reps per point):

| `n_pm` | offset | repeatability σ | 3σ detectable | min J at 3σ (flat EQE) |
|---|---|---|---|---|
| 4 | −3.3 pW | 36.0 pW | 108 pW | 0.123 |
| 8 | −6.5 pW | 16.3 pW | 48.8 pW | 0.055 |
| 32 | +4.0 pW | 7.7 pW | 23.0 pW | 0.026 |
| 128 | −3.6 pW | 5.7 pW | 17.1 pW | 0.019 |
| 512 | −1.4 pW | 2.5 pW | 7.6 pW | 0.009 |

Two things to take from that. The dark **offset is negligible** — under 7 pW and
sign-random, so there is no pedestal to subtract. And the noise is **ideal white
noise**: a log-log fit gives `σ ∝ N^−0.499` across 128× in `n_pm`, so no
systematic floor has been reached and averaging keeps buying sensitivity.

The `min_j_3sigma` column assumes EQE at the target J equals EQE at the −1.35 V
fast-scan point (J ≈ 25), so it is a **best case**. If EQE at the target is a
factor `f` lower, the detectable J scales by `f`:

| `n_pm` | f = 1 | 2 | 5 | 10 | 20 | 50 |
|---|---|---|---|---|---|---|
| 32 | 0.026 | 0.052 | 0.131 | 0.261 | 0.522 | 1.31 |
| 128 | 0.019 | 0.039 | 0.097 | 0.195 | 0.389 | 0.97 |
| 512 | 0.009 | 0.017 | 0.043 | 0.086 | 0.173 | 0.432 |

So **J = 0.1 is reachable at `n_pm` 512 provided EQE there is within ~11× of its
peak**, and at `n_pm` 32 only if it is within ~4×. `optfloor` also records
`per_point_s`, the time each averaging level costs per measured point, which is
what a sweep has to pay.

On the current side, `decay` is the route below 1 µA. At 100 nA into 50 µF the
decay slope is `I/C` = **2 mV/s**, which the SMU's voltmeter resolves far better
than its 100 nA range resolves 100 nA, and the decay sweeps the operating point
continuously, so one run yields a dense I(V) curve instead of discrete points.
Two caveats it already handles: leakage discharges the cap too, and a 0 A
setpoint is only as good as the source range's regulation — the blanked run
cancels both, and `--decay-source-range` defaults to the finest range.

### High-C mode

`highc_b = True` had never worked, for a concrete reason: in High-C mode the
2602B derives the current range from the **limit**, so writing `measure.rangei`
afterwards conflicts with it. The code wrote the range once at configure time and
again per pixel. It now sets the range *before* enabling High-C and never touches
it again, uses `AUTOZERO_ONCE` as the driver's own notes require, and drains the
SMU error queue so a rejected setting is reported instead of silently
misreading. `highc_b` with `perpoint_range` is rejected by `validate()` — they
cannot both own the range.

---

## Known limitations

* The fast scan holds its rows in memory and writes them once at the end, so an
  interrupt during the scan itself loses that phase. The sweep, which is the
  long one, is flushed per pixel.
* Hardware autorange is broken with the fast zero-delay burst read; the current
  measure range is always set explicitly. Leave `perpoint_range` off unless you
  are reproducing the probe-then-narrow experiment.
* `measurement/`, `data/`, `smile_postprocess.py` and `smile_heatmap.py` are
  unmaintained reference implementations with nothing importing them. They hold
  three acquisition techniques `measure_iqe` does not implement, two of which
  are the right instruments for the 50 µF capacitance problem.
  [MEASUREMENT_MODES.md](MEASUREMENT_MODES.md) records each one's status and open
  bugs. Their monolithic driver `auto_measure.py` was removed; it is in git
  history at `2c598d1`.
* The PyQt GUI was removed in the split. It was an intermediary for checking
  things by hand, and the scripts replaced it. A thin wrapper that shells out to
  `python -m measure_iqe` would be the way to bring that convenience back —
  nothing in this repo needs to change for that. The old GUI is still in git
  history if you want it as a starting point.
