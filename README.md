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
    sample_name="Array42_IQE",
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

cfg = IQEConfig.from_file("configs/standard.yaml")     # .yaml, .yml or .json
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
wrong types are rejected with an error naming the key. `configs/standard.yaml` is
an annotated copy of these defaults.

Comments in `measure_iqe/config.py` keep the historical UPPERCASE names of some
parameters (`BLOCK_SIZE` is today's `block_size`) because they quote the bench
runs that pinned each value.

#### Run identity & output

| parameter | type | default | meaning |
|---|---|---|---|
| `sample_name` | `str` | `'Array42_IQE'` | override per array with --array NAME |
| `base_dir` | `str` | `'results'` | output root; a timestamped run dir is created inside |

#### ROI & pixel selection

| parameter | type | default | meaning |
|---|---|---|---|
| `roi` | `tuple[int, int, int, int] \| str` | `(32, 32, 63, 63)` | (x1,y1,x2,y2) or "FULL"/"TL"/"TR"/"BL"/"BR" Which alive pixels get the (expensive) full IQE sweep: "blocks"  — STRATIFIED: tile the ROI into BLOCK_SIZE×BLOCK_SIZE-pixel blocks and take PIXELS_PER_BLOCK from EACH block → equal per-block sampling for spatial (block / row-band / column-band) statistics. Within a block the pixels are spread by FPS, but each block's FPS STARTS from a RANDOM alive pixel (seeded by BLOCK_SEED), so the sampled rows/columns are diversified across blocks instead of landing on the same sub-lattice every time. Change BLOCK_SEED between runs to sample DIFFERENT pixels (cumulatively cover the array). "spatial" — fraction of alive pixels by global farthest-point sampling (uniform map). "stride"  — old every-Nth (aliases columns; kept for compatibility). |
| `sweep_select` | `str` | `'blocks'` | "blocks" \| "spatial" \| "stride" |
| `block_size` | `int` | `8` | blocks mode: block edge in pixels (ROI → ~ROI/BLOCK_SIZE blocks/axis) |
| `pixels_per_block` | `int` | `4` | blocks mode: alive pixels sampled per block |
| `block_seed` | `int` | `0` | blocks mode: RNG seed (vary across runs to sample new pixels) blocks mode: no two measured pixels may sit closer than this (Chebyshev, so 2 = never sharing an edge or a corner). Without it, each block's farthest-point walk ran to the block's own border and the block next door did the same, so pairs of measured LEDs landed side by side across the shared edge — the sample looked stratified but was really clustered, and neighbouring blocks' statistics blurred into each other. Raise it to spread the sample further (thermal/optical crosstalk between nearby pixels is a physical question, so it is a knob); 1 disables the guard. A block too dead or too small to honour it falls back and says so in the log. |
| `min_pixel_separation` | `int` | `2` |  |
| `sweep_fraction` | `float` | `0.1` | spatial mode: fraction of ALIVE pixels to sweep |
| `min_sweep_pixels` | `int` | `6` | spatial mode: floor on the count (so tiny ROIs still map) |
| `nth_pixel_sweep` | `int` | `5` | stride mode: sweep every Nth alive pixel |
| `max_sweep_pixels` | `int` | `40` | hard cap (spatial/stride only; blocks controls its own count) |
| `fastscan_bitval` | `int` | `15` | bitval for the alive/dead fast scan (brightest) |

#### Source mode

| parameter | type | default | meaning |
|---|---|---|---|
| `source_mode` | `str` | `'voltage'` |  |

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
| `bitvals` | `tuple[int, ...]` | `(15,)` | single bitval — densify low-J via spacing instead Fine-sweep mode: "voltage" — plain UNIFORM voltage ramp NVLED_V_COARSE_START → STOP (no coarse map). Hits every voltage including the steep turn-on, so the rising side is densely sampled. Simple, robust — this is what the old clean runs effectively did. RECOMMENDED. "current" — coarse pre-sweep maps V→I, then place points at log-spaced CURRENTS. Even log-J spacing, BUT it skips voltages where dI/dV is steep (the ~−0.9..−1.1 V gap) and depends on the coarse map (which drove the ranging bugs). |
| `fine_sweep_mode` | `str` | `'voltage'` |  |
| `nvled_v_coarse_start` | `float` | `-0.8` | least-negative sweep voltage (just below turn-on) |
| `nvled_v_coarse_stop` | `float` | `-2.6` | most-negative sweep voltage (below saturation knee) |
| `n_coarse` | `int` | `26` | coarse pre-sweep points |
| `coarse_settle_ms` | `float` | `6.0` |  |
| `coarse_n_smu` | `int` | `4` | Where the fit actually needs data: across 140 per-pixel fits the device's IQE peak sits at J ≈ 5 A/cm² (≈5 µA) and the trustworthy fits only reach down to J ≈ 1 (≈1 µA) on the rising side. So target J ≈ 1–10, NOT sub-µA: below ~1 µA the data is unneeded AND that is exactly where the fine measure range takes ~seconds to settle. Starting at ~1 µA keeps every point on a fast-settling range and puts the points where the peak is. |
| `low_current_target_a` | `float` | `1e-06` | lowest LED current the sweep targets (~J 1, the rise foot) |
| `high_current_frac` | `float` | `0.97` |  |
| `n_fine` | `int` | `30` | fine voltage-sweep points. Lowered from 45 once the settle time was measured: the dwell dominates the budget, so fewer points with a real settle beat more points that never settled. 30 points x 2 drive states x ~8 s is ~10 min/pixel. |
| `lowj_weight` | `float` | `1.5` | >1 concentrates fine points toward LOW current (the SRH rising side J≈1–5 that pins peak IQE). 1.0 = uniform log. Voltage-ramp densification (FINE_SWEEP_MODE="voltage"). The rising side + peak that pin the fit live in a narrow voltage window around turn-on (~−0.85 to −1.3 V), while the droop spans the rest. So split the ramp: put VRAMP_LOW_FRAC of the points in the less-negative segment [START, KNEE] (rising side + peak, dense) and the rest in [KNEE, STOP] (droop). |
| `vramp_knee_v` | `float \| None` | `-1.3` | split voltage between the dense (rising) and sparse (droop) halves |
| `vramp_low_frac` | `float` | `0.6` | fraction of points placed in the rising-side segment Sweep direction (thermal A/B test). "low_to_high" measures the rising side first (coolest, recommended); "high_to_low" starts hot. Run both on the same pixels and compare the low-J EQE to gauge self-heating: if high_to_low reads lower IQE at low J, self-heating is biasing you and low_to_high is the better number. |
| `sweep_direction` | `str` | `'high_to_low'` | "low_to_high" \| "high_to_low" |

#### Pixel warm-up / conditioning

| parameter | type | default | meaning |
|---|---|---|---|
| `warmup_ms` | `float` | `1500.0` | 0 = off. Dwell at WARMUP_NVLED_V before each sweep. |
| `warmup_nvled_v` | `float` | `-2.6` | warm-up bias (high current, clears the parasitic) `--warmup-diag` only: watch P, I and EQE vs time at a fixed operating point on several pixels. Pick the point in the J≈1 region where the A/B split was largest — low enough to stress the current readout, but where the LIGHT is still a few× above the dark floor. (−0.6 V was too low: runs 0759/0847 sat AT the dark floor with the current in the hypersensitive turn-on toe, so neither channel was informative.) |
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
| `nvled_compliance` | `float` | `1e-05` | current compliance (voltage mode). In High-C this ALSO selects the measure range, so it sets the noise floor: 10 uA gives ~9 nA/sample at NPLC 0.01, ~1 nA at NPLC 1. Measured 261013. The sweep overrides it per segment (VRAMP_*_LIMIT_A) because -2.6 V needs ~175 uA. |
| `nvled_range_i` | `float` | `0.01` | default fixed measure current range (coarse pre-sweep, fast scan, and any point with no current target). Current readout is the dominant error: ~5 µA on the 10 mA range is 0.05% of full scale (and the low-J rise, <1 µA, is ~0.01%) → 20–50% scatter. AUTORANGE was tried (260629-0913/1000) and BROKE the measurement: with the fast zero-delay BURST read (delay=0, autozero off) autorange can't range up in time, so ch B read 100–800× too LOW (157 µA real → 0.2 µA reported; leakage gate then killed every point). So instead of autorange we set the range DETERMINISTICALLY per point from its known target current (the coarse map gives it) — optimal resolution, correct value, no hunting. (There is no autorange knob: hardware autorange is broken with the burst read and the code never turns it on — the range is always set explicitly by set_measure_range_b.) |
| `perpoint_range` | `bool` | `False` | False = fixed NVLED_RANGE_I (10 mA): consistent (offsets and all) so the curve SHAPE comes through — what the old clean runs used. True = per-point software ranging (accurate value, but range offsets/settling corrupted the rising side; see 260630). Picking the range from the COARSE pre-sweep map failed both directions (260630): the coarse map (10 mA range, pre-conditioning) mis-predicts the fine-sweep current, so the range was too low at high bias (→ overflow, 104 pts) and too coarse on the rise (→ stuck 0.8 µA offset). Fix: PROBE the actual current on a safe wide range first, then narrow. |
| `probe_range_i` | `float` | `0.001` | wide probe range: covers the device max (~175 µA), never overflows, gives the magnitude to pick the real range from |
| `probe_n_smu` | `int` | `4` | quick samples for the magnitude probe |
| `range_settle_ms` | `float` | `50.0` | settle after a measure-range change before reading 2602B High-C mode on ch B. The driver guards High-C on B (raises), so we enable it via TSP. High-C adds settling delays (slower) but damps a capacitive load that can make the low current ranges ring/never settle — the reason 10 mA was the safe range. Off by default; sweep_variants.py toggles it to test whether it makes low ranges usable. |
| `highc_b` | `bool` | `True` | MANDATORY, measured 261013. Without High-C every reading below ~10 uA is a range artifact: it lands at 42-58% of whatever range is selected, independent of bias, wrong by up to 517x. With it, three different start voltages agree to 0.6%. |
| `sweep_autozero` | `str` | `'AUTO'` | AUTO \| ONCE \| OFF |
| `save_raw_samples` | `bool` | `True` |  |

#### Integration time (NPLC) per phase

| parameter | type | default | meaning |
|---|---|---|---|
| `fastscan_nplc` | `float` | `0.05` | Now the fast-scan CURRENT matters: at −1.35 V the fast scan is a full-coverage peak-IQE proxy (EQE=P/I on the EQE plateau), so its current noise sets the proxy quality. Sweep test (260704-1456/1618/ 1659): NPLC 0.01→0.05 cut the current relSEM 18%→3% and the proxy's per-pixel error 14%→5% (down to the device's ~4% uniformity floor) for only +3 s. NPLC 0.1 + N_SMU 8 barely helped (→4.7%) but ~60% slower — so 0.05 / N_SMU 4 is the sweet spot. |
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
| `fastscan_blank_between` | `bool` | `True` | send a black (all-off) frame before lighting each pixel, so exactly ONE pixel is ever emitting when the PM reads — guards against a lingering previous pixel / accumulated frame inflating the optical power (the ~71 nW edge-pixel false-alive, run 1405). |

#### Dead-pixel early abort (voltage sweep)

| parameter | type | default | meaning |
|---|---|---|---|
| `dead_abort` | `bool` | `True` |  |
| `dead_abort_nw` | `float` | `2.0` | absolute floor (nW): first-point power below this ⇒ dead |
| `dead_abort_frac` | `float` | `0.1` | or below this fraction of the running median of live pixels |
| `thermal_settle_ms` | `float` | `100.0` | before the fine sweep, sit at the FIRST point this long so the pixel reaches its starting-current thermal steady state (clears heat left by the coarse pre-sweep) |
| `n_smu_samples` | `int` | `8` | SMU samples averaged per point |
| `n_pm_samples` | `int` | `12` | PM400 readings averaged per point |

#### Per-segment effort (fine_sweep_mode="voltage")

| parameter | type | default | meaning |
|---|---|---|---|
| `settle_scale` | `float` | `1.0` | multiplies every sweep settle. 1 = the measured times. |
| `vramp_perseg_effort` | `bool` | `True` |  |
| `vramp_lowv_nplc` | `float` | `2.0` | integer NPLC, so the mains line is nulled by integration |
| `vramp_lowv_n_pm` | `int` | `32` |  |
| `vramp_lowv_n_smu` | `int` | `12` |  |
| `vramp_lowv_settle_s` | `float` | `8.0` | rising side: 1% needs 5-55 s. 8 s is the compromise that keeps a 30-point sweep under ~10 min/pixel. |
| `vramp_lowv_limit_a` | `float` | `1e-05` | the rising side never exceeds ~5 uA |
| `vramp_highv_nplc` | `float` | `1.0` | was 0.5 -- not an integer, so it nulled no mains at all |
| `vramp_highv_n_pm` | `int` | `8` |  |
| `vramp_highv_n_smu` | `int` | `8` |  |
| `vramp_highv_settle_s` | `float` | `2.0` | droop side: large currents, the relaxation is a much smaller fraction of the reading |
| `vramp_highv_limit_a` | `float` | `0.001` | -2.6 V draws ~175 uA, far above the rising-side limit |

#### Leakage reference (the low-J correction)

| parameter | type | default | meaning |
|---|---|---|---|
| `fastscan_limit_a` | `float` | `0.001` | compliance for the FAST SCAN, which is a separate operating point from the sweep. At nvled_v_fastscan (-1.35 V) a pixel draws ~30 uA, so the sweep's 10 uA rising-side limit clamps EVERY pixel -- measured in run 261014-0048, where all 1024 pixels read exactly 10.00 uA and the light came out 2.8x low. |
| `leak_reference` | `bool` | `True` | measure each sweep point lit and at LEAK_REF_BITVAL |
| `leak_ref_bitval` | `int` | `0` | the reference drive level: addressed, but off |
| `leak_ref_settle_s` | `float` | `2.0` | after switching drive level. Shorter than the sweep settle: the bias does not move, only the drive, and the point of measuring back to back is a shared history. |

Channel B reads the **total** cathode current. Measured over 12 pixels in 4
quadrants on 261013, the reference is 8% of the reading at −0.95 V and **58% at
−0.80 V**. Without subtracting it the EQE appears to collapse ~20× below J ≈ 0.4.
A *blanked* frame is not a valid reference — it draws ~4× **more** than the
addressed pixel at bitval 0, so lit-minus-blanked is negative.

#### Calibration, resume and safety

| parameter | type | default | meaning |
|---|---|---|---|
| `calibrate_pixels` | `int` | `3` | pixels to characterise; the worst case sets the dwell |
| `calibrate_seconds` | `float` | `40.0` | poll each bias this long. Must exceed the real settle time or the answer is a lower bound, which the calibration says so explicitly. |
| `calibrate_n_smu` | `int` | `4` | SMU samples per poll; small, this is a time series |
| `calibrate_period_s` | `float` | `0.05` | poll interval, and the floor on what the calibration can resolve |
| `calibrate_margin` | `float` | `1.5` | safety factor on the measured worst case, because the calibration sees 3 biases and the sweep visits 30 |
| `resume` | `bool` | `False` | with --out pointing at an existing run directory, skip the pixels it already finished and append. The aggregated CSV is flushed per pixel, so a run killed mid-sweep (Windows Update, a power cut) keeps every completed pixel and resumes from the next one. A pixel interrupted part way through its own sweep is re-measured whole, not stitched. |
#### Dead-pixel threshold (fast scan)

| parameter | type | default | meaning |
|---|---|---|---|
| `nvled_v_fastscan` | `float` | `-1.35` | single fast-scan bias — does DOUBLE duty. At −1.35 V the pixels sit at J≈25 A/cm², which is dead on the (broad, flat) EQE peak plateau: EQE is within ~1% of its peak across J≈8–30, so the fast-scan EQE=P/I here ≈ each pixel's PEAK EQE. That turns the fast scan into a full-coverage peak-IQE proxy for all 65 536 pixels (calibrated to the sweep in analysis), while still emitting ~22 nW for the alive/dead classification — ~1000× the MEASURED optical floor (3σ = 23 pW at n_pm 32; see the optfloor run of 261013, σ ∝ N^-0.499 from n_pm 4 to 512). J≈25 keeps ~5 A/cm² of margin before the droop starts (~J30). The current there (~25 µA) is high enough for a fast (NPLC 0.01) read. (Was −2.6 V / J≈95 — the droop side at only ~78% of peak.) |
| `dead_threshold_frac` | `float` | `0.2` | a pixel is alive if P ≥ this × the "typical lit" level |
| `fastscan_alive_pctl` | `int` | `90` | "typical lit" level = this percentile of the fast-scan power. NOT the median: the median collapses to the noise floor on a majority-DEAD ROI (then ~everything is mislabelled alive — e.g. quadrant 260703-1307 was 57% dark, median 0.08 nW → 84% "alive"). A high percentile tracks the lit population as long as the lit fraction exceeds (100−pctl)% = 10%. |
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

### High-C mode is mandatory for low J — measured

`highc_b = True` had never worked, for a concrete reason: in High-C the 2602B
derives the current range from the **limit**, so writing `measure.rangei`
afterwards conflicts with it. The code wrote the range once at configure time and
again per pixel. It now sets the range *before* enabling High-C and never touches
it again, uses `AUTOZERO_ONCE` as the driver's own notes require, and drains the
SMU error queue after every range change and capture.

With that fixed, the 261013 `step` runs on pixel (54,46) show High-C is not an
optimisation — **without it the low-J readings are wrong, not merely noisy:**

| V | High-C off | High-C on (1 mA limit) | ratio |
|---|---|---|---|
| −0.70 | 351.5 µA | **0.680 µA** | 517× |
| −1.00 | 426.7 µA | **5.628 µA** | 76× |
| −1.30 | 30.48 µA | **30.56 µA** | 1.00× |

The two agree at −1.3 V, where the current is tens of µA, and diverge by up to
517× as the current falls. On the fine ranges without High-C it is worse: 63–87%
of samples over-range, and the surviving median lands at **42–52% of whatever
range is selected, independent of bias** — a range artifact, not a measurement.

With High-C the current is physical, monotonic, and **path-independent**: −0.70 V
reads 0.680 µA arriving from −2.6 V and 0.662 µA arriving from −0.9 V, agreeing to
2.7%. The direction hysteresis that motivated `sweep_direction` is gone.

### The limit sets the resolution: measured

Lowering the High-C limit from 1 mA to 10 µA, same pixel, same points:

| V | 1 mA limit | 10 µA limit | they differ by |
|---|---|---|---|
| −0.85 | 934.2 nA, 79% ripple, 735 nA noise | **925.7 nA, 1% ripple, 9.4 nA noise** | 1% |
| −0.80 | 715.5 nA, 89% ripple, 633 nA noise | **447.5 nA, 2% ripple, 9.3 nA noise** | 60% |
| −0.70 | 661.7 nA, 90% ripple, 598 nA noise | **248.6 nA, 4% ripple, 9.5 nA noise** | 166% |

**63× less noise**, and `settled_ok` is True for the first time (tail drift
0.24–0.93%). The 1 mA readings agree only at −0.85 V and run high below that, so
the 1 mA limit is not usable under about 1 µA.

The resulting curve reaches **J = 0.249** — the sub-unity current density that was
the goal — with an ideality factor of 4.4 over −0.70 to −0.85 V:

| V | I | J | time to settle |
|---|---|---|---|
| −0.70 | 248.6 nA | 0.249 | 15.5 s |
| −0.80 | 447.5 nA | 0.447 | 11.3 s |
| −0.85 | 925.7 nA | 0.926 | 2.7 s |

Settling at low J is **tens of seconds**, exactly as `C·ΔV/I` predicts. A separate
50000-sample run confirmed it: at −0.70 V from −2.6 V, `t_1pct` was 52.3 s of a
53.0 s record, where an 11 s record had reported not-settled.

**Current is now path-independent.** The same three biases reached from −1.00 V,
−0.90 V and −0.75 V agree to 0.6–1.2%, so the capacitive history error is gone:

| V | from −0.90 | from −1.00 | from −0.75 | spread |
|---|---|---|---|---|
| −0.85 | 925.7 nA | 923.6 nA | 917.9 nA | 0.9% |
| −0.80 | 447.5 nA | 443.4 nA | 441.9 nA | 1.2% |
| −0.70 | 248.6 nA | 249.9 nA | 249.2 nA | 0.6% |

### The light runs out before the current does

`exp_step` reads the PM400 after each capture and records `power_W` and
`eqe_ext`. Doing that revealed that the optical signal, not the SMU, now sets the
low-J limit:

| V | I | J | P | P/I | EQE |
|---|---|---|---|---|---|
| −1.35 | 30.5 µA | 30.5 | 60 nW | 1.97 mW/A | 0.070% |
| −0.85 | 923.6 nA | 0.92 | 984 pW | 1.07 mW/A | 0.038% |
| −0.80 | 443.4 nA | 0.44 | 233 pW | 0.53 mW/A | 0.019% |
| −0.70 | 249.9 nA | 0.25 | **6.5 pW** | 0.026 mW/A | 0.001% |

`P/I` halves smoothly down to J = 0.44 and then falls 20× in one step — and the
−0.70 V readings (6.5 pW and 8.0 pW on two runs) sit **on** the measured 7.6 pW
3σ floor. That point is censored by the detector, so whether EQE truly collapses
there cannot be read from this data.

**The practical IQE floor is therefore J ≈ 0.4, not 0.1.** At J = 0.44 the signal
is 233 pW, about 90σ. The earlier estimate of 219 pW at J = 0.25 assumed EQE stays
at its peak value; the real EQE there is ~37× lower, so that estimate was 30× too
optimistic.

Two things follow, both implemented:

* `--step-dark` (on by default) measures the optical dark **at each bias** by
  blanking the pixel. Below ~100 pW the PM400's own ±7 pW sign-random offset is
  the same size as the signal, so a single dark per run is not enough.
* `--pm-range` overrides the PM400 range. **Measured: it makes no difference.**
  Five `optfloor` runs across 1e-7, 1e-8 and 1e-9 (three of them the same 1e-9
  range, one after a PM400 restart) give σ = 2.12 ± 0.46 pW at `n_pm` 512, and the
  three *identical* runs alone span 1.33–2.57 pW. The run-to-run scatter exceeds
  any range effect, so the floor is detector-noise-limited and lowering the range
  buys nothing.

What does work is averaging: the pooled scaling is `σ ∝ N^−0.523`, still ideal.

| `n_pm` | σ | 3σ | s/point | a 5 pW signal is |
|---|---|---|---|---|
| 512 | 2.12 pW | 6.35 pW | 2.1 s | 2.4σ |
| 1024 | 1.50 pW | 4.49 pW | 4.1 s | 3.3σ |
| 2048 | 1.06 pW | 3.17 pW | 8.3 s | 4.7σ |
| 4096 | 0.75 pW | 2.24 pW | 16.6 s | 6.7σ |

So J = 0.25 is reachable at `n_pm` 2048–4096, for 8–17 s per point — affordable on
a sampled subset, and cheaper than reworking the optics. One caveat: the offset at
`n_pm` 512 was **+2.37 pW and positive in all five runs**, so it is a real pedestal
rather than noise. Per-point dark subtraction is not optional at these levels.

Every run now writes `run_config.json` with the effective config and arguments.
Without it the five `optfloor` CSVs were indistinguishable — nothing recorded
which PM range produced them.

**No single limit can span the sweep.** At −2.6 V the LED wants ~175 µA, which a
10 µA limit cannot supply, so it cannot reach the droop side at all; and 1 mA
cannot resolve the rising side. A full sweep needs the limit scheduled per segment,
the way effort already is.

Two further consequences:

* **The limit is both the charging current and the measure range.** 1 mA moves the
  array's ~95 µC in under a millisecond but resolves only ~µA; lower it for
  sub-µA points and charging slows proportionally. `validate()` rejects `highc_b`
  with a compliance above 1 mA, because the stock 0.5 A is the one setting that
  looks correct and measures nothing.
* **Channel A is an array background plus the LED current.** It read 490.6, 491.4
  and 501.2 µA while ch B spanned 166× (60 nA to 10 µA) — but it *tracks*: +9.7 µA
  when ch B went from 1 to 10 µA. So it is a real measurement, not an artifact; the
  per-pixel signal simply sits on a ~490 µA array-level background, which makes the
  `VLED/NVLED` ratio meaningless as a per-pixel leakage check.

### Capacitance and current-source mode: measured

`C = 45.7 ± 3.8 µF` from the 10 µA and 100 µA ramps, confirming the ~50 µF
estimate. The 1 µA setpoint is invalidated by a parallel current, exactly as the
method's own cross-check predicts: decomposing `I_set = C·dV/dt + I_parallel`
gives **866 nA parallel when blanked and 162 nA when lit** at −0.35 V. That
difference is unexplained and matters — it is comparable to the 249 nA measured at
−0.70 V.

**Current-source mode works.** Held at 1 µA, ch B delivers 953.7 nA, the node
fully equilibrates (`dV/dt` → 0.002 mV/s after 180 s) at −0.8568 V, and that
agrees with the voltage-mode measurement at −0.85 V (923.6 nA) to **3.3%**. The
original "ch B as a current source DOES NOT WORK on this device" conclusion is
refuted — it was measured with a 15 ms settle against 46 µF.

At 100 nA it still fails, for a fixable reason: every setpoint was sourced on the
1 mA range, where 100 nA is 0.01% of full scale, and the SMU delivered 60 nA. The
source range is now chosen per setpoint.

### Mains rejection: NPLC 0.01 has none

Worth knowing before trusting any ripple number. The SMU's integration window is
`NPLC × 20 ms` at 50 Hz, and only an **integer** NPLC nulls the mains line:

| NPLC | integrates | mains cycles | rejection |
|---|---|---|---|
| 0.01 | 0.2 ms | 0.01 | **none** |
| 0.1 | 2 ms | 0.1 | none |
| 1.0 | 20 ms | 1.0 | **full** |

Each capture records a spectrum summary — `spec_peak_hz`, `spec_peak_amp`,
`spec_mains_amp`, `spec_mains2_amp`, `spec_rms_ac`, all over the **settled tail** —
and the raw `I(t)` is saved to `traces.npz` (uniformly thinned to `--trace-points`,
so the frequency content survives). `plot_transients.py` in the analysis repo draws
the settling curves and their spectra.

Tail only, because the settling is a power law and a quadratic cannot detrend it.
Measured over the whole trace, `rms_ac` came out at 133–283 nA against a real noise
of 9.3 nA: all residual transient.

### Answered: there is no sine (runs 261013-2140 / -2201)

The tail spectrum is **flat**. The amplitude at the mains line, 1.0–1.35 nA, is no
larger than the biggest random bin, 1.26–1.54 nA, and both match what a flat noise
floor of 9.3 nA rms predicts. There is no line anywhere, at any bias.

The oscillation is therefore white noise, not pickup — and an integer NPLC still
fixes it, by averaging rather than by rejection:

| NPLC | per-sample noise | ripple at −0.80 V | settled value |
|---|---|---|---|
| 0.01 | 9.3 nA | 2.09% | 446.9 nA |
| **1.0** | **1.0–1.4 nA** | **0.22%** | 447.8 nA |

7.8× less noise, consistent with √100 for white noise, and the same answer. Use an
integer NPLC on the measurement points. It costs nothing, because the relaxation
below forces a wait anyway.

### Per-array calibration (do this once per array)

The settle time belongs to the **array**, not to the method. The 8 s default was
measured on one array; another can need far more or far less, and it
does **not** follow from the capacitance, because the settling is a charge relaxation
rather than an RC charge.

```powershell
python -m measure_iqe --calibrate --array Array42_IQE --out D:\cal   # once, ~6 min
python -m measure_iqe --array Array42_IQE                            # every run after
```

`--array NAME` sets `sample_name` **and** picks up `calibrations/NAME.yaml`, which
`--calibrate --array NAME` wrote. One flag per array, no file paths to remember. A run
on an array that has not been characterised says so in the log and uses the defaults
rather than silently inheriting another array's numbers.

`calibrations/` is **gitignored on purpose**: those files are measured facts about one
physical array, and re-copying the repo (the share export runs `git archive`) must
never overwrite them. Anything you edit inside a checkout of the repo *will* be
overwritten by the next copy — keep per-array settings in `calibrations/`, not in
`configs/standard.yaml`.

`--calibrate` steps a few pixels at three biases spanning the sweep, polls the current
until it stops changing (`settle_time`, the criterion documented in the investigation
report), takes the worst case, applies `calibrate_margin` and writes a small YAML. It
records the raw measurements beside the recommendation, and says so explicitly if a
capture never settled within `calibrate_seconds` -- in which case the recommendation
is a lower bound and the run should be repeated with a longer window.

Precedence: defaults, then `--config`, then `--calibration`, then `--set`. An explicit
`--set` always wins, so a deliberate override is never silently replaced.

### The old-vs-new demonstration

`run_comparison_demo.ps1` runs the **measurements** for a head-to-head comparison
unattended: calibrate, transients on one pixel, the sweep with the old method, the
sweep with the new one. ~2-4 h depending on the calibrated dwell.

```powershell
.
un_comparison_demo.ps1 -Simulate              # dry run, no hardware, ~2 min
.
un_comparison_demo.ps1 -Array Array42_IQE -Pixel "35,36"
```

It deliberately does **not** analyse. Analysis needs no hardware, can be repeated and
re-tuned, and a failing figure must never cost bench time. Instead it writes
`<Out>nalyse.ps1` with the run directories already filled in -- run that afterwards.

### Configuration files

| file | what it is |
|---|---|
| `configs/standard.yaml` | an annotated, readable **copy** of the defaults. **Editing it changes nothing** — measure_iqe reads `measure_iqe/config.py`, and only loads this file if you pass `--config`. A test asserts the two stay identical. |
| `configs/legacy_pre261014.yaml` | the old, wrong method, kept so it can be run head to head. Not for measurements. |
| `calibrations/<array>.yaml` | written by `--calibrate --array NAME`, applied by `--array NAME`. Gitignored, so a repo re-copy never clobbers it. |

Full key-by-key reference: **Configuration reference** below.

### What the sweep now does, and why (261013)

> The full write-up, with figures, is **`INVESTIGATION_REPORT.md` in the analysis
> repo** (`SMILE_IQE_Analysis`). Regenerate it with
> `python make_capacitance_figures.py ../SMILE_Characterization`. What follows is
> the summary.


The investigation rewrote the sweep's settings. The short version:

| setting | was | is | because |
|---|---|---|---|
| `highc_b` | False | **True** | below ~10 uA the readings were range artifacts, off by up to 517x |
| `nvled_compliance` | 0.5 A | **10 uA** | in High-C the limit picks the range, so it sets the noise floor |
| `vramp_lowv_settle_ms` | 20 ms | **8 s** (`_settle_s`) | measured: 1% takes 5-55 s |
| `vramp_highv_nplc` | 0.5 | **1.0** | 0.5 is not an integer, so it nulled no mains |
| `n_fine` | 45 | **30** | the dwell dominates; fewer points with a real settle win |
| leakage reference | none | **every point** | at -0.80 V, 58% of the reading is not the LED |

Plus an adaptive compliance, because no single limit spans the sweep: -2.6 V
draws ~120 uA while the rising side needs a 10 uA limit to resolve at all. The
limit is picked per point from the previous point's current, with 50% headroom
(`pick_limit`). Tying it to the density knee instead was the first attempt and it
clamped eight consecutive points at exactly 10 uA in run 261014-0007 -- the knee
is at -1.3 V but the device still draws 30 uA at -1.27 V. Points that do sit on
the limit are now flagged `NVLED_AT_COMPLIANCE` and dropped by the analysis.

**Small steps do not settle faster.** That was the natural guess and the bench
refused it (run 261013-2337). Stepping 20 mV onto -0.80 V left the reading 4.5%
high and still drifting at 80 s; stepping 200 mV onto the same point was flat
from 60 s. Each point inherits the unrelaxed tail of every earlier point, and a
small step does not escape that -- it only makes the per-step excursion small
compared with the memory. So the budget is the **dwell**, not the step size, and
the two things that do help are sweeping high current to low current (already the
default) and measuring the bitval-0 reference back to back with the lit point so
both carry the same history.

Cost: ~30 points x 2 drive states x ~8 s, about 4-5 min of settling per pixel.

### Where the low-J limit now sits (runs 261014-0007 / -0035)

Two full sweeps on pixel 124,135, 28 minutes apart. The **current** reproduces to
0.03-0.6% over almost the whole range and 1.3% at the very last point, and the
sweep agrees with the independent `leak` measurement to 4%. The current is no
longer the problem.

The **light** is. At the dimmest point the signal is ~250 pW and the optical
reference wanders:

| V | P_led | within-burst error | run-to-run |
|---|---|---|---|
| -0.911 | 3343 pW | 1.0% | 0.1% |
| -0.883 | 2139 pW | 1.8% | 1.6% |
| -0.856 | 1180 pW | 3.1% | 0.9% |
| -0.828 | 595 pW | 6.8% | 7.6% |
| **-0.800** | **281 pW** | **11.8%** | **33%** |

So J = 0.2 is reachable but its EQE carries tens of percent. J >= 0.5 (-0.856 V
and below) is good to a few percent. Pushing lower needs optical collection or
far heavier PM averaging -- sigma falls as N^-0.5, so 4x the readings halves it --
not more electrical care.

### Two-phase settling

`--charge-limit` charges the array at a high limit and then drops to the
measurement limit, because the limit is the only current available to move C:

| limit | time to move 95 µC |
|---|---|
| 1 mA | 0.09 s |
| 100 µA | 0.87 s |
| 10 µA | 8.7 s |

So charging at 1 mA and measuring at 10 µA *should* replace a 9–40 s settle with
~0.1 s. It does not, and the reason is worth knowing.

**Verified safe, but only 1.2–1.8× faster** (runs 261013-2140 vs -2155, pixel
124,135, four biases). The range switch does not corrupt the reading — the two
agree within 1.1%, and both land +0.3…+2.6% from the independent `leak` ground
truth:

| V | single limit | two-phase | `leak` i_lit | t_1pct single → two-phase |
|---|---|---|---|---|
| −0.95 | 3610.3 nA | 3617.5 nA | 3582.4 nA | 1.13 s → 0.62 s |
| −0.90 | 1985.5 nA | 1989.3 nA | 1969.0 nA | 4.10 s → 2.89 s |
| −0.85 | 948.9 nA | 957.1 nA | 942.3 nA | 8.04 s → 6.12 s |
| −0.80 | 446.9 nA | 451.7 nA | 440.1 nA | 13.24 s → 10.72 s |

### Why it barely helps: the settling is not RC

At 10 µA, moving 0.2 V across 42.8 µF takes 0.86 s. `t_1pct` at that step is
13.2 s — 15× longer. So charging was never the bottleneck.

The excess current above the asymptote decays as **t^−1.2±0.2**, a power law, not an
exponential, and its amplitude is proportional to the voltage step:

| step from −1.00 V | ΔV | excess at 1 s | at 2 s | at 4 s | at 8 s |
|---|---|---|---|---|---|
| −0.95 | 0.05 V | 40 nA | 12 nA | 0 nA | — |
| −0.90 | 0.10 V | 94 nA | 44 nA | 19 nA | 6 nA |
| −0.85 | 0.15 V | 150 nA | 65 nA | 30 nA | 9 nA |
| −0.80 | 0.20 V | 234 nA | 93 nA | 38 nA | 13 nA |

Excess ∝ ΔV at fixed t is what a relaxation of the displaced charge `Q = C·ΔV`
gives — dielectric absorption or trap emission in the same 42.8 µF structure. No
current limit shortens it, which is exactly why two-phase charging gains so little.

**The free lever is step size, not current.** `t_1pct` tracks ΔV almost linearly,
1.13 s at 0.05 V against 13.24 s at 0.20 V. The 13 s figures are an artifact of
parking at −1.00 V and jumping. A monotone sweep in 0.05 V steps should cost ~1 s
per point, and the relaxation current that contaminates it is 6× smaller. Sweep
monotonically in small steps and never jump back to a park voltage.

### The open question: is the low-J current actually LED current?

Both source modes measure the *total* cathode current, so neither separates LED
injection from leakage. The first attempt used a blanked frame as the reference and
**that is invalid** — measured:

| V | lit | blanked | lit − blanked |
|---|---|---|---|
| −0.85 | 955.5 nA | 962.2 nA | −6.7 nA |
| −0.80 | 455.5 nA | 946.3 nA | −490.8 nA |
| −0.70 | 250.4 nA | 934.3 nA | −683.9 nA |

A negative LED current is impossible, and the blanked current is nearly
bias-independent at 934–962 nA. The `cap` ramps had independently inferred the same
thing — 867 nA parallel when blanked against 164 nA when lit at −0.35 V. Two
experiments agree: **an all-off frame leaves the backplane conducting ~5× more than
a frame with one pixel addressed.** The blanked light was 3.2–4.6 pW, i.e. the
optical floor, so the array really is dark; it is the *leakage* that changes, not
the emission.

So blanking does not hold the leakage fixed, and the reference must leave the
backplane addressing alone. `leak` now measures **three** frame states per bias:

* `lit` — the pixel at its normal bitval
* `off` — the **same pixel addressed at bitval 0**: identical frame structure and
  select state, zero drive. This is the valid reference, and `i_led = lit − off`
* `blank` — no frame, recorded for comparison with the runs above

`led_current_physical` is False when `i_led` comes out ≤ 0, and no EQE is published
from a reference that is still wrong.

### Result: the EQE collapse was a contaminated denominator

With the bitval-0 reference, measured on pixel (54,46):

| V | `i_lit` | `i_off` (leak) | **`i_led`** | leak | EQE raw | **EQE corrected** |
|---|---|---|---|---|---|---|
| −0.85 | 951.9 nA | 260.0 nA | **692.0 nA** | 27% | 0.036% | **0.0498%** |
| −0.80 | 449.8 nA | 244.8 nA | **205.1 nA** | 54% | 0.019% | **0.0410%** |
| −0.70 | 247.2 nA | 234.3 nA | **13.0 nA** | 95% | 0.001% | **0.0256%** |

**The 20× collapse below J ≈ 0.4 is gone.** The corrected EQE falls only
0.050 → 0.041 → 0.026%, a smooth factor of 2, which is what SRH on the rising side
looks like. The collapse was the leakage inflating the denominator: at −0.70 V,
**95% of the cathode current is not injection**, only 13 nA of 247 nA is.

The `off` current is 234–260 nA and mildly bias-dependent, while `blank` stays at
934–957 nA — a constant 3.7–4.0× more, which confirms the backplane-state effect
independently at every bias.

This invalidates every rising-side EQE this project has produced, including the
corrected numbers above wherever the leak fraction is high.

**The new low-J limit is the subtraction, not the optics.** `i_led` is a difference
of two large numbers, so `leak_frac` sets the error:

| V | `i_led` | 1% error on each state → | 2% → |
|---|---|---|---|
| −0.85 | 692 nA | ±1.8% | ±3.5% |
| −0.80 | 205 nA | ±3.4% | ±6.8% |
| −0.70 | 13 nA | **±37%** | **±75%** |

So the usable floor is around **J ≈ 0.2** (−0.80 V, 54% leak), not the J ≈ 0.013
the corrected current at −0.70 V implies. `leak` now reports `i_led_sem` and
`i_led_sigma` so this is visible per point rather than inferred.

One useful side result from the same run: at `step_n_pm` 2048 the optical SEM was
1.8 pW, so the 6.0 pW at −0.70 V is a 3.3σ detection. The averaging extrapolation
held.

`highc_b` with `perpoint_range` is rejected by `validate()`: they cannot both own
the range.

**Both High-C transitions are refusal points.** The 2602B locks the range settings
while High-C is on — error `5069 SMU B: Autorange locked for HighC mode`. So
writing `measure.rangei` *after* enabling High-C is refused, and so is re-entering
`configure_voltage_source_b` while High-C is still on from an earlier call, because
`configure_channel` writes `autorangei` before it touches `highc`. The function
therefore always disables High-C first, sets the range, then re-enables it, which
makes it idempotent whatever state the channel was left in. Any code that sets a
range must skip the call under High-C.

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
