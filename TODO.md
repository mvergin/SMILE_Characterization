# SMILE MicroLED EL Characterisation GUI — Audit & Improvement Plan

**Audited:** 2026-04-01
**Codebase version:** GUI `20260323_v1`, instrumentlib `20260322_v1`, smile_postprocess `20260322_v1`
**Files reviewed:** `smile_automatic_gui_2602b.py` (3105 lines), `instrumentlib.py` (854 lines), `smile_postprocess.py` (857 lines)

---

## 1. ARCHITECTURE & REFACTORING PLAN

### 1.1 Current State

~90% of the operational logic lives in `smile_automatic_gui_2602b.py`:
- **Lines 1–360:** Imports, simulation mock classes, helper functions, ArrayMapWidget
- **Lines 362–490:** Version checking, README writer, `_check_versions()`
- **Lines 492–2081:** `MeasurementWorker(QThread)` — ~1590 lines containing instrument
  connection, scan coordination, all measurement loops, data serialisation, post-processing
  dispatch, transient saving, profiling, plotting, on-time analysis, and cleanup
- **Lines 2083–3105:** `MicroLEDCharGUI(QMainWindow)` — ~1020 lines of UI construction,
  state persistence, widget-to-config marshalling, and the start/stop/postprocess handlers

`instrumentlib.py` is reasonably scoped but contains legacy drivers (Keithley2400, Keithley2611)
that are unused by the GUI and add maintenance burden.

`smile_postprocess.py` is well-separated as a standalone module — this is the best-structured
part of the codebase.

### 1.2 Proposed Module Decomposition

#### Module: `instruments/pm400.py`
**Responsibilities:** All PM400 communication — scalar measurement, continuous mode,
array mode configuration/arm/fetch, display control, config readback.
**Currently lives in:** `instrumentlib.py` (PM400 class) + simulation mock in GUI file
(lines 99–161).
**Interface:**
```python
class PM400:
    def __init__(self, resource_id: str): ...
    def configure_scalar(self, wavelength, range_W, averaging): ...
    def configure_array(self, window_ms, delta_t_us) -> (n_samples, delta_t_us): ...
    def configure_continuous(self, wavelength, range_W, averaging): ...
    def measure_scalar(self) -> float: ...           # MEAS:POW?
    def start_array(self) -> None: ...               # INIT
    def poll_array_complete(self, timeout_s) -> bool: ...
    def fetch_array(self, n, offset) -> list[float]: ...
    def start_continuous(self) -> None: ...           # INIT:CONT
    def fetch_latest(self) -> float: ...              # FETC?
    def abort(self) -> None: ...
    def get_mode(self) -> str: ...                    # "scalar" | "array" | "continuous"
    def get_config_dict(self) -> dict: ...
```
**Key change:** Track internal mode state (`scalar`/`array`/`continuous`) explicitly. Refuse
`measure_scalar()` when in array mode (raise instead of silent corruption). The current code
switches between modes without tracking what state the PM400 is in.

#### Module: `instruments/keithley2602b.py`
**Responsibilities:** All 2602B TSP communication — channel configuration, voltage sourcing,
instant measurement, buffered burst measurement, hardware trigger model, buffer readback,
display control.
**Currently lives in:** `instrumentlib.py` (Keithley2602B class) + simulation mock in GUI
file (lines 164–250).
**Interface:**
```python
class Keithley2602B:
    def configure_channel(self, ch, compliance, nplc, high_c, range_i, zero_delays): ...
    def set_voltage(self, ch, voltage): ...
    def enable_output(self, ch, enable): ...
    def measure_instant(self) -> (float, float): ...
    def setup_buffers(self, timestamps): ...
    def clear_buffers(self): ...
    def configure_hardware_trigger(self, count): ...
    def start_hardware_trigger(self): ...
    def abort_trigger(self): ...
    def measure_burst_fire(self, n): ...
    def measure_burst_join(self): ...
    def read_buffers(self, n) -> (list, list): ...
    def read_buffer_with_timestamps(self) -> (list, list, list, list): ...
    def check_error_queue(self) -> list[str]: ...     # NEW: read errorqueue
```

#### Module: `instruments/smile_fpga.py`
**Responsibilities:** FPGA pixel addressing — coordinate mapping (logical → physical),
quadrant enable configuration, `set_pixel()`, blank frame / turnoff, acknowledgement handling.
**Currently lives in:** `MeasurementWorker` methods `get_smile_config_and_buffer()` (line 572),
`get_direct_config_and_chunks()` (line 596), `_get_pixel_coords()` (line 625),
`_maybe_turnoff_dis()` (line 1051). The coordinate mapping is duplicated three times with
slight variations.
**Interface:**
```python
class SmileFPGA:
    def __init__(self, smile_dev): ...
    def set_pixel(self, log_x, log_y, bit_val) -> float: ...  # returns ack timestamp
    def blank_frame(self) -> float: ...                         # returns ack timestamp
    def map_logical_to_physical(self, log_x, log_y) -> (row, col, quad_cfg): ...
```
**Key change:** Consolidate the three separate coordinate mapping implementations into one
authoritative method. The current `get_smile_config_and_buffer()` and
`get_direct_config_and_chunks()` appear to be legacy and unused (the GUI only calls
`_get_pixel_coords` + `set_pixel`).

#### Module: `instruments/sim.py`
**Responsibilities:** Simulation/mock classes for all instruments.
**Currently lives in:** GUI file lines 99–250.
**Design:** Each mock should implement the same interface as the real driver. Use a factory
function: `create_pm400(resource_id, simulate=False)`.

#### Module: `measurement/modes.py` (or separate files per mode)
**Responsibilities:** Define each measurement mode as a self-contained strategy class.
Each mode declares:
- What instrument state it requires at entry
- Its per-pixel measurement sequence
- Its dark acquisition strategy (`none`, `dark_tail`, `post_pixel_scalar`)
- What data it produces per pixel
- What instrument state it leaves at exit

**Currently lives in:** `MeasurementWorker._loop_transient()` (lines 1131–1568, ~437 lines)
and `MeasurementWorker._loop_fast_scan()` (lines 1570–1747, ~177 lines). These methods
directly read `self.cfg`, call instrument APIs, manage timing, compute statistics, and
push data to the write queue.

**Interface:**
```python
class MeasurementMode(ABC):
    @abstractmethod
    def configure_instruments(self, pm, smu, cfg) -> dict: ...
    @abstractmethod
    def measure_pixel(self, pm, smu, fpga, pixel, bit_val, nv_vol, ...) -> PixelResult: ...
    @abstractmethod
    def dark_strategy(self) -> DarkStrategy: ...  # enum: NONE, DARK_TAIL, POST_PIXEL
    @abstractmethod
    def cleanup_instruments(self, pm, smu): ...

class TransientMode(MeasurementMode): ...
class FastScanMode(MeasurementMode): ...
class UltraFastMode(MeasurementMode): ...       # future
```

#### Module: `measurement/coordinator.py`
**Responsibilities:** The scan loop — iterates over pixels, calls the active measurement
mode, handles dark acquisition dispatch, manages the write queue, computes ETA, emits
progress signals.
**Currently lives in:** The outer loops of `_loop_transient` and `_loop_fast_scan`, plus
the `run()` method of `MeasurementWorker` (lines 1950–2081).
**Key change:** The coordinator should not know the details of any measurement mode. It
calls `mode.measure_pixel(...)` and `mode.dark_strategy()` to decide what to do.

#### Module: `data/writer.py`
**Responsibilities:** CSV writing, HDF5 transient writing, the background writer thread.
**Currently lives in:** `_writer_worker` closure (line 1972), `_save_transient` method
(lines 809–887), CSV header definition (lines 1988–2001).
**Interface:**
```python
class DataWriter:
    def __init__(self, csv_path, hdf5_path=None): ...
    def write_pixel_data(self, rows: list): ...
    def write_transient(self, ...): ...
    def flush(self): ...
    def close(self): ...
```

#### Module: `data/structures.py`
**Responsibilities:** Define `PixelResult`, `TransientResult`, `DarkResult` dataclasses
that all modes produce, replacing the current raw list-of-lists format.

#### Module: `gui/main_window.py`
**Responsibilities:** Only the PyQt6 UI — widget construction, state persistence,
config marshalling into a `MeasurementConfig` dataclass. Never imports `pyvisa`,
never calls instrument methods.
**Currently lives in:** `MicroLEDCharGUI` class (lines 2083–3105).

### 1.3 Implicit GUI State That Must Be Made Explicit

The `start_measurement()` method (line 2979) reads ~45 widget values into a `cfg` dict
at the moment the user clicks START. This is the correct pattern — it snapshots the config.
However, several values are computed from widget state in non-obvious ways:

- `"secondary_storage_enabled"` is `chk_secondary.isChecked() AND mode != "Fast Scan"` (line 3024)
- `"smu_display_off"` is gated on mode (line 3035)
- `"pm_range"` is `float(self.sb_pm_range.currentText())` — string-to-float conversion at
  marshal time (line 3000)

**Recommendation:** Create a `MeasurementConfig` dataclass with validation. The GUI populates
it; the worker receives it as a typed, immutable object. No dict key typos, no implicit
conversions.

### 1.4 Refactoring Order (Safest → Most Invasive)

1. **Extract simulation mocks** to `instruments/sim.py` — pure move, no logic change
2. **Extract `instruments/smile_fpga.py`** — consolidate coordinate mapping, remove the
   three duplicates
3. **Extract `data/writer.py`** — move the writer thread and `_save_transient`
4. **Create `MeasurementConfig` dataclass** — replace the cfg dict
5. **Extract `instruments/pm400.py` and `instruments/keithley2602b.py`** — move from
   instrumentlib, add mode tracking to PM400
6. **Extract measurement modes** — this is the most invasive step; requires the mode
   interface to be stable before moving the loop bodies
7. **Extract `measurement/coordinator.py`** — the outer scan loop
8. **Thin out `MicroLEDCharGUI`** — it should only construct widgets and marshal config

---

## 2. IF/ELSE BLOCK AUDIT

### 2.1 Block: `_loop_transient` — Swept vs Fixed NVLED (lines 1177–1547)

**What it does:** A 370-line `if not is_swept: ... else: ...` block inside the
per-bit-value loop. The non-swept path (Mode A) arms instruments, sends the pixel,
waits, fetches, computes dark tail, computes stats, saves transients. The swept path
(Mode B) sends the pixel once, then loops over NVLED voltages repeating the arm/wait/fetch
sequence per voltage.

**Problem:** The two branches share ~80% of their logic (arm PM400, start Keithley trigger,
wait, abort, fetch, compute stats, dark tail extraction, secondary save). The shared logic
is copy-pasted with minor differences (Mode B has an extra `smu.set_voltage("b", nv_vol)`
and a settle sleep; Mode A records `t_ack_s`). Any bug fix must be applied in both branches.

**Proposed pattern:** Extract a `_measure_one_window()` helper that takes the pre/post
actions as callbacks:

```python
def _measure_one_window(self, pm, smu, instr, profiling, start_time,
                         pre_arm_fn=None, post_fetch_fn=None) -> WindowResult:
    """One arm→wait→fetch→stats cycle. Used by both Mode A and Mode B."""
```

Mode A calls it once per pixel. Mode B calls it in a voltage loop. The `if/else` reduces
to ~20 lines of mode-specific setup.

**Ultra-fast mode fit:** Ultra-fast mode would NOT use `_measure_one_window` at all — it runs
a continuous stream. It slots in as a third `MeasurementMode` subclass with its own
`measure_pixel()` that operates on the continuous buffer. The coordinator doesn't care.

### 2.2 Block: `_maybe_dark_acq` — Mode-Dependent Dark Strategy (lines 1064–1125)

**What it does:** A shared subroutine called by both transient and fast scan modes after
the pixel measurement. It reads a scalar PM400 measurement, an instant SMU measurement,
and appends DARK_* rows.

**Problem:** This subroutine assumes scalar PM400 access. In transient mode, PM400 is in
array mode — calling `pm.measure()` (which sends MEAS:POW? = ABOR + INIT + FETC in scalar
power mode) after CONF:ARR may fail or produce garbage. Fast scan works because it passes
`pm_fn=pm.fetch_latest` explicitly. See Bug 2 analysis for full details.

**Proposed pattern:** Each measurement mode declares its dark acquisition strategy via an
enum. The coordinator dispatches accordingly:

```python
class DarkStrategy(Enum):
    NONE = "none"
    DARK_TAIL = "dark_tail"          # transient: captured in PM400 buffer
    POST_PIXEL_SCALAR = "scalar"     # fast scan: pm.fetch_latest() after blank
    POST_PIXEL_ARRAY = "array"       # future: short array capture after blank
```

No shared `_maybe_dark_acq`. Each mode's `measure_pixel()` handles its own dark logic
internally, producing `DarkResult` as part of its return value.

### 2.3 Block: `run()` — Mode Dispatch (lines 1950–2078)

**What it does:** `if mode == "Fast Scan": ... else: # Full Transient`. Currently only
two branches but each sets up instruments differently.

**Proposed pattern:** The mode dispatch becomes:

```python
mode = MODE_REGISTRY[self.cfg.measurement_mode]  # returns MeasurementMode subclass
mode.configure_instruments(pm, smu, self.cfg)
step_count = coordinator.run_scan(mode, pixel_list, ...)
mode.cleanup_instruments(pm, smu)
```

Adding ultra-fast mode means adding one entry to the registry, not touching the dispatch.

### 2.4 Block: `generate_pixel_sequence` — Quadrant Mapping (lines 528–570)

**What it does:** Maps quadrant name → coordinate ranges, then applies snake scan and
random order.

**This is fine.** It's only ~40 lines and reads clearly. The `if/elif` chain maps 5
quadrant names to coordinate ranges — a dict lookup would be marginally cleaner but
not worth changing in isolation.

---

## 3. BUG ANALYSIS & REMEDIATION PLAN

### Bug 1 — Keithley 2602B High-Capacitance Mode Misconfiguration

#### What the Code Does Now

In `instrumentlib.py:configure_channel()` (lines 464–489):
```python
ch.source.func = ch.OUTPUT_DCVOLTS       # 1. set source function
ch.source.limiti = compliance             # 2. set compliance
ch.measure.nplc = nplc                    # 3. set NPLC
ch.measure.autorangei = ch.AUTORANGE_OFF  # 4. disable autorange
ch.measure.rangei = range_i               # 5. set fixed range
ch.source.highc = ch.ENABLE              # 6. enable high-C
# if zero_delays:
ch.source.delay = 0                       # 7. zero source delay
ch.measure.delay = 0                      # 8. zero measure delay
ch.measure.autozero = ch.AUTOZERO_OFF    # 9. disable autozero
```

The GUI always calls this with `zero_delays=True` (line 916–922).

#### What Correct 2602B High-Cap Mode Requires

Per the Keithley 2602B reference manual (Section 3, "High Capacitance Mode"):

1. **Source function must be voltage** (`OUTPUT_DCVOLTS`) — this IS set correctly.

2. **High-C only works reliably on SMUA.** The 2602B's high-C compensation circuit is
   physically implemented on channel A only on most firmware revisions. Enabling it on SMUB
   may silently fail or behave unpredictably. The current code enables it on BOTH channels
   based on user checkboxes — the NVLED (channel B) high-C checkbox should either be
   removed or display a warning.

3. **Autozero interaction:** When `autozero = AUTOZERO_OFF`, the instrument skips periodic
   reference measurements. This is fine for speed but can cause slow DC drift. The critical
   issue: when high-C is enabled, the source settling loop uses autozero internally for
   stability — disabling it externally may defeat the high-C compensation. **This is the
   most likely root cause.** The manual recommends `AUTOZERO_ONCE` or `AUTOZERO_AUTO`
   when high-C is active.

4. **Source delay:** High-C mode adds internal settling time. Setting `source.delay = 0`
   overrides this, potentially reading before the source has stabilised through the
   high-C compensation network. The 2602B has an `smuX.source.settling` attribute that
   controls the internal settling algorithm — when high-C is on, the default settling
   should be left intact.

5. **Current range vs compliance:** If the fixed measurement range (`rangei`) is smaller
   than the compliance limit (`limiti`), the instrument may try to auto-range during a
   compliance event, even with autorange off. Ensure `rangei >= limiti` or at least that
   the compliance current is reachable within the configured range.

6. **Order of operations:** The manual specifies that `source.highc` should be set
   AFTER `source.func` — the current code does this correctly.

#### Diagnostic Steps

1. **Read the error queue after configuration:**
   ```lua
   print(errorqueue.count)
   -- then read each:
   ec, msg, sev, node = errorqueue.next()
   print(ec, msg)
   ```
   If high-C + AUTOZERO_OFF produces error 4917 ("Settings conflict"), that confirms
   the interaction.

2. **Test with autozero enabled:**
   ```lua
   smua.measure.autozero = smua.AUTOZERO_AUTO
   smua.source.delay = smua.DELAY_AUTO
   smua.source.highc = smua.ENABLE
   ```
   If this works, the root cause is the zero_delays override.

3. **Verify on SMUB:** Enable high-C on SMUB only and check `errorqueue` — likely
   produces error or silent no-op.

4. **Check compliance behaviour:** Set a load that draws near-compliance current and
   measure with high-C on vs off. If the range is too small, you'll see clipping.

#### Proposed Fix

```python
def configure_channel(self, channel, compliance, nplc, high_c, range_i, zero_delays):
    ch = f"smu{channel.lower()}"
    self._write_and_wait(f"{ch}.source.func = {ch}.OUTPUT_DCVOLTS")
    self._write_and_wait(f"{ch}.source.limiti = {compliance}")
    self._write_and_wait(f"{ch}.measure.nplc = {nplc}")
    self._write_and_wait(f"{ch}.measure.autorangei = {ch}.AUTORANGE_OFF")
    self._write_and_wait(f"{ch}.measure.rangei = {range_i}")

    if high_c:
        if channel.lower() != 'a':
            raise ValueError("High-C mode only supported on channel A (SMUA)")
        # High-C requires autozero and source delay to function correctly
        self._write_and_wait(f"{ch}.source.highc = {ch}.ENABLE")
        self._write_and_wait(f"{ch}.measure.autozero = {ch}.AUTOZERO_ONCE")
        self._write_and_wait(f"{ch}.source.delay = {ch}.DELAY_AUTO")
        self._write_and_wait(f"{ch}.measure.delay = {ch}.DELAY_AUTO")
    else:
        self._write_and_wait(f"{ch}.source.highc = {ch}.DISABLE")
        if zero_delays:
            self._write_and_wait(f"{ch}.source.delay = 0")
            self._write_and_wait(f"{ch}.measure.delay = 0")
            self._write_and_wait(f"{ch}.measure.autozero = {ch}.AUTOZERO_OFF")
```

### Bug 2 — Dark Acquisition Control Flow Failure in Transient Mode

#### Tracing the Failure

**Fast Scan path (works):**
1. PM400 is in continuous mode (`INIT:CONT`) — started at line 1605
2. Pixel is measured via `fetch_latest()` (FETC?) calls — lines 1652–1654
3. `_maybe_turnoff_dis()` sends blank frame — line 1713
4. `_maybe_dark_acq()` called with `pm_fn=pm.fetch_latest` — line 1715–1726
5. `pm.fetch_latest()` returns the latest continuous reading (dark) — **works correctly**

**Transient Mode A path (fails when dark_acq=True AND dark_tail_ms=0):**
1. PM400 configured for array mode: `pm.configure_array_mode()` — line 1014
2. Per pixel: `pm.abort()`, `pm.start_array()` (INIT) — lines 1181, 1187
3. Pixel is measured in array mode, then `smu.abort_trigger()`, `pm.abort()` — lines 1224–1225
4. PM400 buffers are fetched — line 1231 or 1236
5. `_maybe_turnoff_dis()` sends blank frame — line 1340
6. `_maybe_dark_acq()` called with **no pm_fn** → defaults to `pm.measure()` — line 1341
7. `pm.measure()` sends `MEAS:POW?` which is `ABOR + INIT + FETC` in **scalar power mode**
8. But PM400 is still configured for `CONF:ARR` — the `INIT` triggers an array capture,
   not a scalar measurement. `FETC` then tries to read a scalar value from an array buffer.
   **This either errors, hangs until timeout, or returns garbage.**

**Transient Mode A path with dark_tail_ms > 0 (partially works, then crashes):**
1. Steps 1–4 as above
2. Dark tail samples are extracted from the PM400 buffer tail — lines 1252–1276. **This
   part works correctly** — dark values come from the already-captured array.
3. BUT: line 1340–1351 is reached regardless. `_maybe_turnoff_dis()` fires (sends a
   SECOND blank frame — redundant, since one was already sent at line 1208). Then
   `_maybe_dark_acq()` is called, which tries `pm.measure()` — **same failure as above**.

The bug is that `_maybe_dark_acq()` runs unconditionally after the dark tail extraction.
The dark tail produces `DARK_PM400` rows (line 1263), and then `_maybe_dark_acq()` tries
to produce MORE `DARK_*` rows via an incompatible measurement path.

**Transient Mode B path (swept NVLED, fails identically):**
Same issue at lines 1536–1547: `_maybe_turnoff_dis()` + `_maybe_dark_acq()` called after
the sweep loop with the same incompatible `pm.measure()` default.

#### Root Cause Summary

1. `_maybe_dark_acq()` is a shared subroutine that assumes scalar PM400 access.
2. Transient mode never passes a compatible `pm_fn` — the default `pm.measure()` is
   incompatible with array mode.
3. The dark tail logic (in-buffer dark capture) works correctly but does not suppress
   the subsequent `_maybe_dark_acq()` call.
4. Even if `_maybe_dark_acq()` were skipped when dark tail > 0, the dark_tail=0 case
   with dark_acq=True would still crash because there's no valid PM400 scalar read
   available in array mode.

#### Proposed Clean Architecture

Each measurement mode explicitly owns its dark acquisition:

```
TransientMode:
  if dark_tail_ms > 0:
    strategy = DARK_TAIL  → extend buffer, turn off pixel mid-capture,
                             extract dark from buffer tail
                             DO NOT call _maybe_dark_acq
  elif dark_acq:
    strategy = POST_PIXEL_REARM → after aborting array capture,
                                    send CONF:POW to return PM400 to scalar mode,
                                    THEN take one pm.measure() for dark
                                    (slower but correct)
  else:
    strategy = NONE

FastScanMode:
  if dark_acq:
    strategy = POST_PIXEL_CONTINUOUS → pm.fetch_latest() after blank frame
  else:
    strategy = NONE

UltraFastMode:
  strategy = INTERLEAVED_DARK_FRAMES → dark frames built into the pixel sequence,
                                        extracted from continuous buffer
```

**Implementation steps:**
1. Remove the unconditional `_maybe_dark_acq()` call from `_loop_transient`
2. For dark_tail > 0: the existing buffer-based dark extraction is correct — keep it
3. For dark_tail = 0 with dark_acq enabled: add `pm.inst.write("CONF:POW")` +
   `time.sleep(0.05)` before calling `pm.measure()` to reset PM400 to scalar mode.
   This adds ~50ms overhead per pixel but is correct.
4. Better yet: make the PM400 driver track its mode and refuse incompatible calls:
   `pm.measure()` raises if `pm.mode == "array"`, forcing the caller to explicitly
   reconfigure.

---

## 4. MEASUREMENT MODE REVIEW & GAPS

### 4.1 Full Transient Mode (Mode A — Fixed NVLED)

**What it does:** Per pixel: abort PM400 → clear Keithley buffers → configure hardware
trigger → arm PM400 array (INIT) → arm Keithley trigger → `set_pixel()` → wait for
capture window → abort both → fetch PM400 array → fetch Keithley buffers with timestamps.

**Instrument state required:**
- PM400: `CONF:ARR` configured (done once in `_build_instr_params`), autorange off,
  filter off
- Keithley: buffers in append mode with timestamps, hardware trigger configured per pixel

**Data produced:** Per pixel per bit value: steady-state mean/std of PM400, VLED, NVLED
(from tail of captured waveforms). Optionally: full transient arrays to HDF5/CSV.

**Correctness concerns:**
- **PM400 state after mode:** `pm.abort()` is called but `CONF:ARR` remains configured.
  If the scan is interrupted and restarted in Fast Scan mode without recycling the PM400
  object, the `CONF:POW` at line 1597 resets it — this works but is fragile.
- **Timestamp alignment:** `T0 = (T_arm_start + T_arm_end) / 2.0` assumes symmetric
  arm latency. PM400 `INIT` and Keithley `trigger.initiate()` have different latencies.
  The ACK timestamp from set_pixel is more reliable as a reference.
- **k_trigger_count estimation:** `int(extended_window_ms / (nplc * 20.0)) * 1.5 + 10`
  is a heuristic. If the actual measurement rate is faster than expected (low NPLC),
  buffers could overflow silently. Should verify `nvbuffer1.n < nvbuffer1.capacity` after
  capture.

### 4.2 Full Transient Mode (Mode B — Swept NVLED)

**What it does:** Per pixel: `set_pixel()` once, then loop over NVLED voltages. For each
voltage: set NVLED, settle, arm PM400 + Keithley, wait, fetch. The pixel stays lit
throughout the sweep.

**Instrument state required:** Same as Mode A, plus NVLED voltage changes per step.

**Data produced:** Same as Mode A, but one set of measurements per (pixel, bit_value,
nvled_voltage) triple.

**Correctness concerns:**
- **NVLED settle time:** The `time.sleep(nvled_settle_ms / 1000.0)` at line 1373 is a
  fixed delay. It does not verify that the NVLED current has actually stabilised. A smarter
  approach would be to take a few SMU readings and check for convergence.
- **No ACK timestamp in Mode B:** `t_ack_s` is set to `None` (line 1530). The pixel was
  already lit before the sweep started, so the ACK from the initial `set_pixel` is not
  recorded per voltage step. This is correct but should be documented.

### 4.3 Fast Scan Mode

**What it does:** PM400 in continuous mode (`INIT:CONT`). Per pixel: `set_pixel()` →
settle → fire Keithley burst (non-blocking) → N× `FETC?` from PM400 → join Keithley burst
→ read Keithley buffers.

**Instrument state required:**
- PM400: `CONF:POW` → settings reconfigured → `INIT:CONT` (done at lines 1595–1605)
- Keithley: buffers in append mode, no timestamps

**Data produced:** Per pixel per bit value per NVLED voltage: N individual PM400 readings
+ N VLED/NVLED readings, each stored as individual rows (not aggregated to mean/std at
capture time — aggregation happens in post-processing).

**Correctness concerns:**
- **PM400 CONF:POW sequence:** Lines 1595–1605 manually write `ABOR`, `CONF:POW`,
  reconfigure wavelength/range/averaging, then `INIT:CONT`. This is necessary because
  entering Fast Scan after a Transient run leaves PM400 in `CONF:ARR` mode. However, if
  Fast Scan is the first mode used, the extra `CONF:POW` is harmless but the 50ms sleep
  is wasted. More importantly, this PM400 mode reset logic lives in the measurement loop
  rather than in the instrument driver — it should be in `PM400.configure_continuous()`.
- **Stale PM400 readings:** Each `FETC?` returns the most recent completed measurement.
  At the PM400's internal measurement rate, multiple `FETC?` calls within the same
  integration period return the SAME value. The N readings are not independent — they
  share temporal correlation. This is documented nowhere and could mislead users into
  thinking they have N independent samples.
- **Buffer clear per pixel:** `smu.clear_buffers_only()` is called per pixel (line 1631),
  but `smu.read_buffers()` reads ALL buffered values (line 1662). If the burst produces
  `n_pts` per voltage × `n_voltages` entries, the indexing at line 1667 `idx = j * n_pts + k`
  assumes they're packed sequentially — this is correct because `clear_buffers_only()` is
  called once per pixel, before the voltage loop.
- **PM400 continuous mode after scan:** `pm.stop_continuous()` is called at line 1746.
  If the scan is interrupted (`is_running = False`), the early break at line 1608 skips
  this call. The `_cleanup()` method (line 1915) calls `pm.stop_continuous()` in a
  try/except, which catches this — but it's an implicit safety net, not explicit control.

### 4.4 Post-Processing Modes (in `smile_postprocess.py`)

These are not measurement modes but run-time analysis passes. They are well-structured.
The on-time estimator uses a global threshold (20% of run-wide peak) which is robust
against dead pixels. The two-pass approach (collect peaks → set threshold → estimate) is
sound.

**Concern:** `post_process_data()` is duplicated — there's a copy in `MeasurementWorker`
(lines 666–807) AND in `smile_postprocess.py` (lines 173–324). The worker copy appears to
be legacy and is never called (the worker calls `_postprocess.run_postprocess()` instead).
**Delete the worker copy.**

---

## 5. ULTRA-FAST SCAN MODE — DESIGN PROPOSAL

### 5.1 Concept

Instead of arming/fetching per pixel, configure both instruments for continuous buffered
capture. Rapid-fire the pixel sequence via FPGA. Use FPGA acknowledgement timestamps as
the ground truth for when each pixel turned on. Post-hoc, slice the continuous buffer
into per-pixel windows.

Target: ~200 px/s raw throughput (including blank/dark frames between pixels), ~100 px/s
effective (measurement-only), vs current ~35 px/s in fast scan.

### 5.2 Instrument Configuration Sequence

#### PM400 Configuration
```
ABOR
CONF:ARR <total_samples>, <delta_t_us>
```
Where:
- `total_samples = (n_pixels × (active_ms + dark_ms) + margin_ms) / (delta_t_us / 1000)`
- `delta_t_us = 100` (10 kHz, PM400 minimum)
- PM400 buffer limit is 10,000 samples per `CONF:ARR` call. For >10k samples, use
  multiple sequential captures with overlapping timestamps, OR implement a rolling
  fetch where Python reads completed chunks while new data is still being captured.
- For 256×256 = 65,536 pixels at 5ms/pixel = ~328 seconds of data = ~3.28M samples.
  This FAR exceeds the 10k buffer. **Strategy:** chunk the pixel sequence into groups
  of ~1000 pixels, each fitting within a 10k-sample PM400 buffer at 100µs resolution
  (= 1 second of data). Between chunks: fetch the buffer, re-arm, continue.

#### Keithley 2602B Configuration
```lua
smua.nvbuffer1.clear() smub.nvbuffer1.clear()
smua.nvbuffer1.appendmode = 1
smub.nvbuffer1.appendmode = 1
smua.nvbuffer1.collecttimestamps = 1
smub.nvbuffer1.collecttimestamps = 1
```
The Keithley 2602B nvbuffer1 can hold up to 60,000 readings. At ~5 readings per pixel
(0.001 NPLC ≈ 0.06ms per reading, 5ms window → ~83 readings/pixel), a 1000-pixel chunk
produces ~83,000 readings — exceeds the buffer. **Reduce to ~700 pixels per chunk**, or
use the trigger model with `trigger.count` set to the chunk size and autonomous operation.

Better approach: use a TSP script that measures in a tight loop and is synchronised with
the Python-side pixel firing:

```lua
-- Keithley runs autonomously:
for i = 1, total_readings do
  smua.measure.i(smua.nvbuffer1)
  smub.measure.i(smub.nvbuffer1)
end
print('DONE')
```

At 0.001 NPLC (20µs per reading), this produces ~50,000 readings/second, well matched to
the PM400's 10,000 samples/second.

#### FPGA Command Sequence
```python
for pixel in pixel_sequence:
    # pixel_sequence includes explicit dark frames: (BLANK, 0, 0) entries
    if pixel.is_dark:
        fpga.blank_frame()     # returns ack_timestamp
        ack_log.append(("DARK", ack_timestamp))
    else:
        fpga.set_pixel(pixel.x, pixel.y, pixel.bit_val)  # returns ack_timestamp
        ack_log.append((pixel, ack_timestamp))
    # NO sleep — fire as fast as FPGA can acknowledge
```

The FPGA ack is the timing reference. The ack arrives when the pixel is activated on the
display controller — this is the moment light appears at the sensor.

### 5.3 Pixel Sequence with Dark Frames

```
[DARK] [px_0] [DARK] [px_1] [DARK] [px_2] ...
```

Or, for efficiency, dark frames every N pixels:

```
[DARK] [px_0] [px_1] ... [px_9] [DARK] [px_10] ... [px_19] [DARK] ...
```

Each dark frame is a blank (all-zero) image. The PM400/Keithley readings during a dark
frame provide the instantaneous dark level, which drifts slowly over the scan. Interpolating
between adjacent dark frames gives per-pixel dark correction.

### 5.4 The Unsort/Alignment Algorithm

**Ground truth:** The FPGA acknowledgement log `ack_log` — a list of
`(pixel_or_dark, ack_timestamp)` tuples. This tells us EXACTLY when each pixel was
activated, regardless of whether it produced any EL signal.

**Step 1 — Define measurement windows:**
For each ack at time `t_ack[i]`, the measurement window is `[t_ack[i], t_ack[i+1])`.
The last pixel's window extends to end-of-buffer.

```python
for i in range(len(ack_log)):
    t_start = ack_timestamps[i]
    t_end = ack_timestamps[i + 1] if i + 1 < len(ack_log) else buffer_end_time
    window = (t_start, t_end)
```

**Step 2 — Slice continuous buffers:**
For the PM400 array: binary-search `t_start` and `t_end` in the PM400 time axis
(which is evenly spaced: `t_pm[j] = T0_pm + j * delta_t_us * 1e-6`). Extract
`pm_values[j_start:j_end]`.

For the Keithley buffers: use the Keithley timestamps to find readings within
`[t_start, t_end)`. The Keithley timestamps are in the instrument's internal clock;
they must be aligned to the PM400 / host clock using the known `T0_keithley` reference.

**Step 3 — Per-pixel extraction:**
```python
for (label, t_ack), (t_start, t_end) in zip(ack_log, windows):
    pm_slice = pm_buffer[t_start:t_end]
    k_slice_a, k_slice_b = keithley_buffer_a[t_start:t_end], ...
    if label == "DARK":
        dark_level = mean(pm_slice)
    else:
        pixel_result = PixelResult(
            pixel=label,
            pm_mean=mean(pm_slice),
            pm_std=std(pm_slice),
            vled_mean=mean(k_slice_a),
            nvled_mean=mean(k_slice_b),
            dark_corrected_pm=mean(pm_slice) - interpolated_dark(t_ack),
            n_samples_pm=len(pm_slice),
            n_samples_k=len(k_slice_a),
        )
```

**Step 4 — Dead pixel detection:**
A dead pixel produces no EL signal. Because we use ack timestamps (not flank detection),
dead pixels are naturally handled — their window contains only dark-level PM400 readings.
Flag pixels where `pm_mean - interpolated_dark < noise_threshold` as dead/dim. Do NOT
treat absence of a flank as a timing error.

**Step 5 — Secondary consistency check (optional):**
Flank detection in the PM400 stream CAN serve as a sanity check:
- For each pixel window, check if there's a rising edge in PM400 power within
  `[t_ack, t_ack + max_on_time]`.
- If a flank is detected, compare its timestamp with `t_ack`. If `|t_flank - t_ack| > threshold`,
  flag a desynchronisation warning.
- If NO flank is detected: check if the pixel is flagged as dead. If not dead (Keithley
  shows current flowing), flag as anomalous (light not reaching sensor — alignment issue?).
- This check is **non-load-bearing**: it never overrides the ack-based window assignment.
  It only adds diagnostic flags to the output.

### 5.5 Chunked Operation

Given PM400's 10k buffer limit:

```
CHUNK_SIZE = 800  # pixels per chunk (including dark frames)
SAMPLES_PER_CHUNK = 10000  # PM400 buffer
CHUNK_DURATION_MS = SAMPLES_PER_CHUNK * 0.1  # = 1000 ms at 100µs

for chunk in chunks(pixel_sequence_with_darks, CHUNK_SIZE):
    pm.abort()
    pm.configure_array_mode(CHUNK_DURATION_MS, 100)
    smu.clear_buffers_only()

    pm.start_array()
    smu_script = fire_tsp_continuous_measurement(len(chunk) * readings_per_pixel)
    T0 = time.perf_counter()

    for pixel in chunk:
        fpga.set_pixel_or_blank(pixel)
        ack_log.append((pixel, time.perf_counter() - T0))

    # Wait for PM400 capture to complete (or timeout)
    pm.poll_array_complete(timeout_s=CHUNK_DURATION_MS/1000 + 1)
    smu.measure_burst_join()  # or abort_trigger

    # Fetch and process
    pm_data = pm.fetch_array(SAMPLES_PER_CHUNK)
    k_data_a, k_data_b, k_ts_a, _ = smu.read_buffer_with_timestamps()
    results.extend(unsort_and_align(ack_log, pm_data, k_data_a, k_data_b, k_ts_a, T0))
    ack_log.clear()
```

### 5.6 Fit into Refactored Module Structure

```
measurement/
  modes/
    ultra_fast.py  → UltraFastMode(MeasurementMode)
      configure_instruments()  — CONF:ARR, buffer setup
      measure_chunk()          — fires pixel chunk, returns raw buffers
      unsort_and_align()       — ack-based window extraction
      dark_strategy = INTERLEAVED_DARK_FRAMES
```

The coordinator calls `mode.measure_chunk()` in a loop instead of `mode.measure_pixel()`
— this is the only coordinator-level difference. The mode interface should accommodate
both per-pixel and per-chunk operation.

### 5.7 Failure Modes and Recovery

| Failure | Detection | Recovery |
|---------|-----------|----------|
| PM400 buffer overflow (chunk too long) | `poll_array_complete()` times out | Reduce chunk size, re-arm, re-measure |
| Keithley buffer overflow | `nvbuffer1.n == nvbuffer1.capacity` | Reduce chunk size |
| FPGA ack timeout (pixel not acknowledged) | `set_pixel()` exceeds timeout | Log as failed pixel, advance to next |
| Stream desynchronisation | Flank check shows systematic offset | Abort chunk, re-align T0, retry |
| Missed ack (ack_log shorter than expected) | `len(ack_log) < len(chunk)` | Map available acks, flag unmatched pixels as "ack missing" |
| PM400/Keithley clock drift | Flank timestamps drift from ack timestamps across chunk | Compute linear drift correction per chunk; warn if >1ms |

### 5.8 Additional Data from Ultra-Fast Mode

Because the PM400 captures a continuous stream at 10 kHz, each pixel's window contains
the full turn-on transient — even though the window may only be 3–5 ms. This enables:

- **Turn-on time estimation** per pixel from the rising edge shape
- **Turn-off tail** visible in the next dark frame's early samples
- **Temporal uniformity mapping** — comparing turn-on times across the array
- **Current transient** from the Keithley stream — VLED inrush current during pixel switch

None of this is available from the current fast scan mode (which returns only point samples).

---

## 6. DATA HANDLING & OUTPUT

### 6.1 Current State

Raw data is written as a flat CSV with columns:
`X, Y, BITVAL, NVLED_V, TIME, TYPE, MEAS_VALUE, MEAS_STD`

Where TYPE ∈ {PM400, VLED, NVLED, DARK_PM400, DARK_VLED, DARK_NVLED}. This means each
"pixel measurement" produces 3–6 rows (3 measurement types + optionally 3 dark types).
Fast scan produces N×3 rows per pixel (no aggregation at capture time).

Transient data is optionally saved as per-pixel CSV files or HDF5 groups.

### 6.2 Recommended Per-Pixel Data Structure

```python
@dataclass
class PixelMeasurement:
    x: int
    y: int
    bitval: int
    nvled_v: float
    timestamp: float                # perf_counter relative to scan start
    mode: str                       # "transient", "fast_scan", "ultra_fast"

    # Primary measurements (steady-state or windowed mean)
    pm400_power_W: float
    pm400_power_std: float
    vled_current_A: float
    vled_current_std: float
    nvled_current_A: float
    nvled_current_std: float

    # Dark measurements (NaN if not acquired)
    dark_pm400_W: float = float('nan')
    dark_pm400_std: float = float('nan')
    dark_vled_A: float = float('nan')
    dark_nvled_A: float = float('nan')

    # Derived
    dark_corrected_pm400_W: float = float('nan')

    # Diagnostics
    n_samples_pm: int = 0           # how many PM400 samples contributed
    n_samples_k: int = 0            # how many Keithley samples contributed
    fpga_ack_ms: float = float('nan')  # ack time relative to arm
    flags: list[str] = field(default_factory=list)  # ["dead", "dim", "ack_timeout", ...]
```

This structure works across all modes. Transient mode fills it from buffer tails.
Fast scan fills it from N-point averages. Ultra-fast fills it from windowed extraction.

### 6.3 Raw CSV Format (Revised)

Keep the flat CSV for compatibility but make each pixel ONE row with all fields:
```
X, Y, BITVAL, NVLED_V, TIME, MODE, PM400_W, PM400_STD, VLED_A, VLED_STD,
NVLED_A, NVLED_STD, DARK_PM400_W, DARK_PM400_STD, N_PM_SAMPLES, N_K_SAMPLES,
ACK_MS, FLAGS
```

This eliminates the need for post-processing aggregation just to get one row per pixel.

### 6.4 Incremental Saving (Crash Recovery)

The current architecture already writes incrementally — the background writer thread
flushes CSV rows as they arrive via the queue. This is good. However:

- The CSV header is written once at the start. If the process crashes, the CSV is valid
  up to the last flushed row.
- HDF5 files may be corrupted on crash (HDF5 does not journal). **Recommendation:** flush
  HDF5 after each pixel group and consider `h5py.File(..., libver='latest', swmr=True)`
  for crash resilience.
- **Add a progress checkpoint file** (JSON) updated every N pixels with the last completed
  pixel coordinates. On restart, offer to resume from the checkpoint.

### 6.5 Metadata Per Scan

Currently saved in `*_config.json`:
- All GUI settings (the `cfg` dict)
- VISA addresses
- PM400 config readback
- SMU config readback
- Software versions

**Missing metadata to add:**
- FPGA firmware version (if queryable from `Smile` object)
- Scan start time (ISO 8601, not just folder timestamp)
- Scan end time
- Total pixels measured / total pixels planned
- Any Keithley error queue entries during the scan
- Any PM400 status warnings
- Host system info (Python version, OS, hostname) — useful for reproducing issues
- Quadrant-to-physical mapping documentation (which quadrant was actually scanned)

---

## 7. OBSERVABILITY & DEBUGGING

### 7.1 Structured Instrument Command Log

Currently, instrument communication is fire-and-forget — commands are sent via
`_write_and_wait()` or `inst.write()`/`inst.query()` with no logging.

**Recommendation:** Add an optional command log to each instrument driver:

```python
class InstrumentLogger:
    def __init__(self, name: str, enabled: bool = False):
        self.name = name
        self.log = []
        self.enabled = enabled

    def record(self, direction: str, cmd: str, response: str = None):
        if self.enabled:
            self.log.append({
                "t": time.perf_counter(),
                "dir": direction,  # "TX" or "RX"
                "cmd": cmd,
                "resp": response
            })
```

Wrap `inst.write()` and `inst.query()` to pass through the logger. Save the log as
`instrument_log.jsonl` in the run directory.

### 7.2 Error Queue Polling

**Keithley 2602B:** Add `check_error_queue()` method:
```lua
print(errorqueue.count)
-- if count > 0:
ec, msg, sev, node = errorqueue.next()
print(ec, msg, sev, node)
```

Call this after `configure_channel()` (catches high-C conflicts), after each
`trigger.initiate()` (catches trigger model errors), and during cleanup.

**PM400:** Add `check_status()` method:
```
SYST:ERR?      → error code + message
STAT:OPER:COND?  → operation condition register
```

### 7.3 Anomaly Flagging

Currently, measurement anomalies are silently recorded as data points. Recommend:

- **Current limit flag:** If Keithley VLED current equals compliance ± 1%, flag the pixel
  as "compliance limited" — the voltage was not fully applied.
- **PM400 range overflow:** If any PM400 reading equals the range maximum, flag as
  "range overflow".
- **Negative power:** Flag PM400 readings < -3σ of noise as "negative power anomaly".
- **FPGA ack timeout:** If `set_pixel()` takes > 100ms, flag as "slow ack".
- **Buffer underrun:** If Keithley buffer has fewer readings than expected, flag as
  "buffer underrun".

Store flags per pixel in the `flags` column of the CSV.

### 7.4 Progress Reporting

Current: ETA string in status bar showing pixel coordinates, percentage, and time remaining.

**Enhancements:**
- Show per-pixel measurement value (PM400 power) in the status bar — this gives immediate
  feedback on whether pixels are alive
- Colour the pixel map by measured value (not just red/green for in-progress/done) — e.g.
  a heatmap that builds up during the scan
- Show instrument health: last Keithley error queue status, PM400 measurement rate
- Log the first and last 5 pixels' values at scan end as a quick sanity check

---

## 8. PRIORITISED ACTION PLAN

### (a) Quick Wins — Immediate Clarity / Safety, Low Risk

1. **Delete duplicate `post_process_data()` from `MeasurementWorker`** (lines 666–807).
   It's never called — the worker uses `_postprocess.run_postprocess()` instead.

2. **Delete unused `get_smile_config_and_buffer()` and `get_direct_config_and_chunks()`**
   from `MeasurementWorker` (lines 572–623). Only `_get_pixel_coords()` + `set_pixel()`
   is used. These methods add confusion about which coordinate mapping is authoritative.

3. **Add `pm.stop_continuous()` to the early-exit path of `_loop_fast_scan`** — currently
   only called if the loop completes normally (line 1746). The cleanup method catches this
   but it's better to be explicit.

4. **Add Keithley error queue check** after `configure_channel()` — even before fixing
   high-C, this will surface configuration errors immediately rather than causing silent
   measurement corruption.

5. **Remove Keithley2400 and Keithley2611 from instrumentlib.py** if they are truly
   unused by this project. They add 190 lines of unmaintained code.

6. **Move simulation mocks** out of the GUI file into a separate `instruments/sim.py`
   (~150 lines freed from the main file).

### (b) Bug Fixes — High-Cap Mode and Dark Acquisition

7. **Fix dark acquisition in transient mode (Bug 2):**
   - Remove the unconditional `_maybe_dark_acq()` call at the end of Mode A (line 1341)
     and Mode B (line 1537)
   - When `dark_tail_ms > 0`: dark data is already extracted from the buffer — done
   - When `dark_tail_ms == 0` and `dark_acq == True`: add PM400 mode reset
     (`CONF:POW` + reconfigure) before the scalar dark read, OR switch to fetching one
     last array capture as the dark measurement
   - Also remove the redundant second `_maybe_turnoff_dis()` at line 1340 (one was
     already sent at line 1208 when dark_tail > 0)

8. **Fix Keithley high-cap mode (Bug 1):**
   - Do not disable autozero when high-C is enabled (use `AUTOZERO_ONCE`)
   - Do not zero source delay when high-C is enabled (use `DELAY_AUTO`)
   - Reject high-C on channel B (SMUB) — warn in GUI or raise in driver
   - Add post-configuration error queue check to verify no conflicts

9. **Add PM400 mode tracking** to the driver class — track whether it's in scalar, array,
   or continuous mode. Raise on incompatible calls (e.g. `measure()` when in array mode).

### (c) Refactoring — Module Decomposition, IF/ELSE Cleanup, GUI Decoupling

10. **Create `MeasurementConfig` dataclass** — replace the 45-key dict with a typed,
    validated config object. This can be done incrementally (create the dataclass, have
    `start_measurement()` build it, pass it to the worker).

11. **Extract instrument drivers** into `instruments/` package — move PM400 and
    Keithley2602B, add mode tracking and error queue methods.

12. **Extract `SmileFPGA` wrapper** — consolidate coordinate mapping, wrap `set_pixel()`
    and blank frame with ack timestamp capture.

13. **Extract `DataWriter`** — move CSV writing, HDF5 transient saving, and the background
    writer thread into `data/writer.py`.

14. **Extract measurement modes** — define the `MeasurementMode` interface, implement
    `TransientMode` and `FastScanMode` as separate classes. Each mode owns its
    instrument setup, per-pixel logic, and dark acquisition strategy.

15. **Extract scan coordinator** — the pixel iteration loop, ETA computation, and progress
    emission become a standalone class that takes a `MeasurementMode` instance.

16. **Thin the GUI** — `MicroLEDCharGUI` should only construct widgets, marshal config,
    and connect signals. No `pyvisa` import, no instrument knowledge.

### (d) New Features — Ultra-Fast Scan Mode

17. **Implement chunked continuous capture** — PM400 array mode with automatic re-arm
    between chunks.

18. **Implement FPGA rapid-fire sequence** — pixel list with interleaved dark frames,
    ack timestamp recording.

19. **Implement unsort/alignment algorithm** — ack-based window slicing, per-pixel
    extraction, dead pixel detection, optional flank-based consistency check.

20. **Implement dark frame interpolation** — linear interpolation between adjacent dark
    frames for per-pixel background correction.

21. **Integrate into mode registry** — `UltraFastMode` class implementing the
    `MeasurementMode` interface with chunk-based operation.

22. **Benchmark and tune** — measure actual throughput, optimise chunk size, verify
    PM400 ↔ Keithley ↔ FPGA clock alignment across long scans.
