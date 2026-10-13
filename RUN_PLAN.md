# Bench run plan — capacitance and low-J investigation

Staged so the cheap decisive runs happen first. Each stage says what to send back.
Do not jump to the long runs: stage 1 decides whether High-C works at all, and
everything after it is shaped by that answer.

Times are estimates. Add slack.

---

## Setup (once, on the measurement PC)

```powershell
cd <wherever you copied SMILETESTS>
uv sync
```

If `uv` is not on that machine:

```powershell
py -3.13 -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

Verify without touching hardware — both of these must pass before you plug anything in:

```powershell
.venv\Scripts\python.exe test_measure_iqe.py
.venv\Scripts\python.exe test_run_format.py
.venv\Scripts\python.exe -m measure_iqe --simulate --set roi='[0,0,15,15]'
```

Then check the instruments answer, and fix the addresses in
`measure_iqe/config.py` (`pm400_addr`, `smu_addr`) if they have changed:

```powershell
.venv\Scripts\python.exe -c "import pyvisa; print(pyvisa.ResourceManager().list_resources())"
```

Pick an output directory outside the repo for everything below, e.g.
`--out D:\caprun`.

---

## Stage 0 — optical floor  (~3 min, no hardware risk)

Decides whether J < 1 is reachable at all. The array is blanked the whole time.

```powershell
.venv\Scripts\python.exe characterize_capacitance.py --what optfloor --out D:\caprun
```

**Send back:** `optfloor.csv`.

What it decides: J = 0.1 (100 nA) emits roughly 44–88 pW. If the 3σ detectable
power at `N_PM = 512` is well under that, J = 0.1 is measurable with heavy
averaging. If it is not, J = 0.3 becomes the realistic floor and the fix is
optical collection, not more averaging. **Do not bother with `decay` until this
says yes.**

---

## Stage 1 — does High-C work?  (~8 min, one pixel)

The single most important run. Pick a pixel you know is good — a clean one from
the earlier work, e.g. `80,15`, or any pixel `find_knee_jump.py` called clean.

```powershell
.venv\Scripts\python.exe characterize_capacitance.py --what step --out D:\caprun ^
  --pixels 80,15 --highc-both ^
  --voltages -0.7 -0.85 -1.0 -1.3 -2.6 ^
  --ranges 1e-3 1e-5 1e-6
```

**Send back:** `step.csv`, plus anything the log printed about
`SMU rejected a ch B setting`.

What to look for yourself, before sending:

* Any line containing `SMU rejected a ch B setting` means High-C is still
  fighting the range. Stop and send me the message.
* Compare `t_1pct` / `t_01pct` for `highc=False` vs `highc=True` at V = −0.85.
  High-C should settle **much** faster. If it does, the whole approach works.
* `ripple` tells you about the oscillation you saw. It should collapse under
  High-C.
* On `highc=False`, check whether the fine ranges (1e-6) have a *different
  settled value*, not just a slower one. A different value is the "reads low J
  WRONG" effect, and that is the thing to kill.

---

## Stage 2 — the actual capacitance  (~6 min, same pixel)

```powershell
.venv\Scripts\python.exe characterize_capacitance.py --what cap --out D:\caprun ^
  --pixels 80,15 --cap-currents 1e-6 1e-5 1e-4 --cap-park -0.40 --cap-v-stop -0.65
```

**Send back:** `cap.csv`.

`cap` measures `C = I/(dV/dt)`: park below turn-on, hand the node to a current
source, time the voltage ramp. It needs only a voltage slope, so it is immune to
the current-range problem. Three setpoints that disagree mean the LED is taking a
share of the ramp current — lower `--cap-v-stop`.

The original version integrated the current transient of a voltage step instead.
It failed on the bench: an accurate current *through* a fast transient is exactly
what the capacitor prevents, and one over-range sample at 9.91e37 poisoned the
integral (it reported C = 1e40 F). Do not resurrect that approach.

`stepdir` is now largely redundant: path independence across three starting
voltages, agreeing to 0.6%, already showed the error was capacitive. Run it only
as a cross-check, and **with `--highc --compliance 1e-5`** — it reads current in
voltage mode and without High-C the numbers are range artifacts.

---

## Stage 3 — the current-source question  (~10 min, same pixel)

This is the one that decides whether current-source mode comes back. It must hold
for minutes; do not shorten `--isrc-seconds`.

```powershell
.venv\Scripts\python.exe characterize_capacitance.py --what isrc --out D:\caprun ^
  --pixels 80,15 --isrc-currents 1e-7 1e-6 1e-5 --isrc-seconds 180
```

**Send back:** `isrc.csv`.

What to look for: in the log tail for each setpoint, compare `chA` against
`chB` and watch `dV/dt`.

* `chA` converging toward `chB` as `dV/dt` → 0 means **no bypass path** — the old
  rejection was the capacitor, and I will build the hybrid low-J current-source
  path.
* `chA` stuck at tens of µA while `dV/dt` is already ≈ 0 means the bypass is
  **real** and current-sourcing the cathode genuinely cannot control this device.

Either answer is useful. The second one closes the question for good.

---

## ⬆ Report back here before going further

Stages 0–3 are about 25 minutes of bench time on one pixel and they settle every
open question. Send the four CSVs and I will tell you what they say and what the
sweep settings should become.

Everything below is the follow-up once the settings are known. Do not run it
first — on 12 pixels it is hours, and the parameters may change.

---

## Stage 3b — is the low-J current actually LED current?  (~3 min, one pixel)

```powershell
.venv\Scripts\python.exe characterize_capacitance.py --what leak --out D:\caprun ^
  --pixels 80,15 --highc --compliance 1e-5 --nplc 0.01 --samples 20000 ^
  --max-presettle-s 90 --v-start -0.9 --voltages -0.85 -0.80 -0.70 --step-n-pm 2048
```

**Send back:** `leak.csv`.

Both source modes read the *total* cathode current. The `cap` ramps implied a few
hundred nA of parallel current near −0.35 V, comparable to the 249 nA measured at
−0.70 V, so the apparent EQE collapse below J ≈ 0.4 may be a contaminated
denominator rather than device physics. `leak` measures each bias lit and blanked;
`i_led` is the difference and `leak_frac` says how much of the cathode current is
not injection.

---

## Stage 4 — spatial: is any of this uniform?  (~60 min, 12 pixels)

Only after stage 1 tells us which ranges and which High-C setting are worth
using. Substitute those below.

Everything up to here has been **one pixel in one quadrant**, so C, the leakage
and the settling times are all still assumed uniform.

```powershell
.venv\Scripts\python.exe characterize_capacitance.py --what cap leak --out D:\caprun ^
  --highc --compliance 1e-5 --per-quadrant 3 --quadrant-window 24 ^
  --nplc 0.01 --samples 20000 --v-start -0.9 --voltages -0.85 -0.80 -0.70 ^
  --step-n-pm 2048 --max-presettle-s 90
```

Finds 3 alive pixels per quadrant by scanning a window at each quadrant centre,
checks each for light, and skips the dead ones. The script prints a wall-clock
estimate before it connects, so a long run is a decision rather than a surprise.
Halve it with `--per-quadrant 2 --quadrant-window 16` and two voltages (~27 min).

**Send back:** `cap.csv`, `leak.csv` — every row is tagged with `pixel_x`,
`pixel_y` and `quadrant`, so the spatial comparison is in the file. What it
decides: whether one compliance schedule and one settle budget work array-wide, or
whether the sweep needs per-region margin.

## Stage 5 — charge decay, only if stage 0 allowed it  (~10 min/pixel)

Use the `capacitance_F` that stage 2 actually measured, not the 50 µF default.

```powershell
.venv\Scripts\python.exe characterize_capacitance.py --what decay --out D:\caprun ^
  --pixels 80,15 --capacitance <measured C> ^
  --decay-from -1.3 --decay-seconds 300 --decay-n-pm 128
```

**Send back:** `decay.csv`. The lit-minus-blanked difference is the LED current;
`j_from_dvdt` is where it lands in J.

---

## If something goes wrong

* `pyvisa` cannot find the instruments → fix `pm400_addr` / `smu_addr` in
  `measure_iqe/config.py`, or pass them with
  `--set pm400_addr=... --set smu_addr=...` to `measure_iqe` (the characterisation
  script reads the config defaults).
* A run dies partway → the CSVs written so far are complete and worth sending.
  Every experiment writes its file at the end of that experiment, so finished
  stages survive.
* The FPGA is unavailable → the log says `SMILE FPGA unavailable … blanking will
  be a no-op`. **Stop.** Without pixel control, every "pixel" measurement is
  actually the whole array.
* Any `SMU rejected a ch B setting` line → send it, it means a TSP setting was
  refused. `5069 Autorange locked for HighC mode` means something wrote a range
  while High-C was on.
* **Every experiment that reads current in voltage mode needs `--highc`.** Without
  it, readings below ~10 µA are not noisy, they are wrong: they land at 42–58% of
  whatever range is selected, independent of bias. The script warns if you omit it.

## Safety notes

* Every script parks the hardware in `finally`: both channels off, frame blanked,
  handles closed. Ctrl-C is safe and keeps the data written so far.
* `nvled_v_limit` is 3.2 V and `nvled_compliance` is 0.5 A. Nothing in this plan
  changes them.
* Stage 3 drives ch B as a current source. It sets the level back to 0 and turns
  the outputs off on exit, including on Ctrl-C.
