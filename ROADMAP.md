# Roadmap

Triage of a brainstorm from 15 October 2026, with the evidence each verdict rests on.
Items 0 and 1 are done; the rest are proposals, not decisions.

| # | item | verdict | effort | status |
|---|---|---|---|---|
| 0 | run provenance: commit SHA, calibration copy, names in filenames | do first | ~2 h | **done** |
| 1 | repo cleanup | yes, but not via an untracked `_obsolete/` | ~1 h | **done** |
| 2 | merge the analysis repo in | yes | ~3 h | **done** |
| 2b | `characterize_capacitance` into `measure_iqe` | **deferred, deliberately** | ~1 h | see below |
| 3 | config: a local override for `base_dir` | yes — but **not** by auto-loading `standard.yaml` | ~1 h | **done** |
| 4 | README | yes, after 1–3 | ~2 h | **done** |
| 5 | cluster / strided scan | pursue — but not for the reason it was proposed | spike first | **done, except the bench run** |
| 6 | GUI | last | 1–2 days | **done** |

---

## 0. Provenance — done

The only item with a deadline: every run already on disk has the problem, and runs
are ongoing. *"I cannot tell directly what measurement routine was actually used."*

* **Commit SHA + dirty flag, written automatically**, not git tags. Tags need
  discipline and discipline fails; a SHA cannot be forgotten. Tags stay as a human
  layer on top (`v0.3 — bias_major`), never as the provenance mechanism.
* **The calibration YAML is copied into the run folder.** 1104 bytes, and otherwise
  there is no way to know which one was applied.
* **Sample name and timestamp in every output filename**, so a stray
  `aggregated_bitval=15_mean_std.csv` can still be traced to its run.

## 1. Cleanup — done, with one change to the original plan

An untracked `_obsolete/` means **deleting the files from git with extra steps**: git
history already preserves everything, and moving a tracked file into an ignored folder
only makes it invisible to `git log --follow`.

So the obsolete files were `git rm`-ed instead. They stay in history, recoverable by
SHA, and `ATTIC.md` records what went and where to find it.

## 2. Merge the analysis repo in — agreed

The reasoning in the brainstorm is the right reasoning: *"without the analysis the
whole measurement doesn't really say much, and the analysis depends heavily on how the
measurement is done, especially with error or estimates."*

It is already biting us:

* `RUN_FORMAT.md` exists in both repos and is kept byte-identical **by hand** — the
  report audit checks its md5 every time.
* The analysis repo has **no git remote**. The entire investigation lives on one disk.
  **This second reason was wrong, and acting on it caused an incident — see below.**

Proposed shape:

```
measure_iqe/
  analysis/        the scripts, importable
  characterize/    characterize_capacitance.py moves here
python -m measure_iqe.analysis <run_dir>
```

`characterize_capacitance` belongs inside `measure_iqe`: same instruments, same
`IQEConfig`, and `--calibrate` is already in there. Its being outside is an accident of
when it was written.

### Done, 16 October 2026 — and the shape changed

`git subtree add --prefix=analysis` rather than a file copy, so all **47** analysis
commits are in the history and `3f4e3e2` is an ancestor of HEAD. The analysis repo was
read-only throughout: it holds the only copy of §18–§23 and nothing was going to touch it.

`analysis/`, **not** `measure_iqe/analysis/`. The scripts are scripts — they are run, not
imported — so a subpackage would mean an `__init__.py`, a `__main__`, and rewriting every
data path for the privilege of typing `python -m measure_iqe.analysis`. Top level solves
both things the merge was for and nothing else needs to move.

What the merge fixed, beyond one remote and one history:

* `RUN_FORMAT.md` was byte-identical in two repos **by hand**, and the report audit
  md5'd it every time. One copy now. The writer-side and reader-side contract tests both
  survive, renamed apart (`test_run_format.py`, `analysis/test_reads_run_format.py`) —
  they were never duplicates, they test opposite directions.
* six scripts hardcoded `MEAS = Path(r"C:\_D\programs\SMILE_Characterization")`. It is
  now `Path(__file__).resolve().parent.parent`, which the merge makes correct.
* **a bug the merge created and the merge had to fix**: nine scripts wrote figures to a
  CWD-relative `investigation_report/`, and the measurement root already had one of its
  own. Run from the root, `plot_followup.py` put figure 35 in the wrong directory. All
  nine now write beside the script.
* duplicated infrastructure removed: `analysis/`'s own `LICENSE`, `pyproject.toml`,
  `requirements.txt`, `uv.lock` and `.python-version`. The dependency delta was exactly
  one package, so `scipy` became an `analysis` extra. `analysis/.gitignore` **stays** —
  git honours nested ignores and it is what keeps 11 MB of figures out.

### The incident, 16 October 2026 — the report is confidential and was pushed

The merge put `INVESTIGATION_REPORT.md` on a **public** GitHub repo. It was never
supposed to leave the measurement machine. The commit message that shipped it reads
*"one repo, with the analysis history intact"* — the history was exactly the problem,
because the subtree carried **15 revisions** of the report, not one.

What the second bullet above got wrong: *"the entire investigation lives on one disk"*
was read as a backup problem to solve with a remote. For the report it is not a
problem, it is the **requirement**. Only the scripts needed the remote.

Fixed:

* `git filter-branch --index-filter` over `--all`, removing both paths the file ever
  had (`INVESTIGATION_REPORT.md` in the 47 imported commits, `analysis/` after the
  merge). 134 commits rewritten, every SHA from the subtree add onward changed.
* the root `.gitignore` blocks the bare filename, so it matches at **any** depth. One
  rule, not one per directory.
* `analysis/README.md` now says where the report actually lives instead of linking it.

**A rewrite is not an un-publish.** The repo was public while the report was on it.
Treat the content as disclosed: GitHub keeps unreachable objects fetchable by SHA until
its own GC, and forks or crawlers are not rewritten at all.

`caprun/` was raised at the same time — real Array38 run data, 24 files, CSVs plus
`REPORT.md` plus eight PNGs, committed `b68295e` on 14 October, before any of this.

**Decided, 16 October 2026: leave the history, stop tracking it.** The old commits stay
public on purpose. `git rm -r --cached caprun` plus a `.gitignore` rule means no further
run ever enters git, which is the rule this file already states two paragraphs down.

The cost is real and accepted: `make_capacitance_figures.py` reads `caprun/` off disk,
so **a fresh clone cannot rebuild the investigation figures**. Only the bench machine
can. The `tests/fixture/` data stays tracked — it is synthetic `FixtureArray` output the
contract tests generate and compare, not device data.

The old report blobs also stay fetchable on GitHub by commit SHA, verified: `main`
returns 404, `072202d` returns 200. Accepted as well. The repo stays public.

### 2b. Moving `characterize_capacitance.py` — deferred, and this is the reason

It is pure tidiness, and the bill is not. Every command in §20, §21, §22 and §23 reads
`python characterize_capacitance.py --what switch ...`, the bench campaign is live, and
the operator has those in shell history. Renaming the entry point invalidates all of it
to move a file that already imports `measure_iqe` and already works.

Revisit when the campaign stops, not during it.

**Measurement data must not live under the repo root.** Gitignore is not a guard
against `git add -A` — on 15 October that command swept in a 5.8 MB PDF from another
agent's working directory without anyone noticing. Not being under the repo root is the
guard. Set `base_dir` to an absolute path outside the repo.

## 3. `standard.yaml` — the need is real, the mechanism proposed is not

Setting `base_dir` once is legitimate. Auto-loading a **tracked** file is not: a run's
behaviour would then depend on an uncommitted local edit to a file git thinks is clean.
That is the same class of problem as item 0.

Proposed instead: load `configs/local.yaml` if it exists — gitignored, and **logged
loudly** at run start (`applied configs/local.yaml: base_dir, sample_name`).
`standard.yaml` stays the documented reference.

### Done, 16 October 2026

Built as proposed, plus one decision the proposal did not make: `local.yaml` sits
**above** `--config` and **below** `--set` and `--out`. It describes this machine, so a
config file shared between machines must not put this one's runs on a disk that does not
exist here — but anything typed per invocation still wins.

Both stated defects are fixed, and the key check found more than the two it was written
for. Adding the key-set assertion immediately failed on **five** absent fields:
`sweep_rotate`, `bias_major_precondition`, `leak_ref_tries`, `leak_ref_max_frac` and
`calibration_source`. All five are in the file now, with comments.

The refuted switch comments are gone, replaced by what §20–§22 measured — including the
part that inverts the sign: the fast component of a frame change is non-radiative, so
EQE at the 2 s wait reads **+0.50 % HIGH**, not 0.5 % low.

Two real defects in that file, independent of the above:

* the sync test compares `IQEConfig.from_file(...) == IQEConfig()`, so a **missing** key
  falls back to the default and passes. `sweep_rotate` and `bias_major_precondition`
  were both added without the test noticing. It only catches wrong values, never
  absent ones.
* its comments still assert the **refuted** switch bound — *"a frame change at a held
  bias costs under 0.5 %"*, *"measured need is under 1 s"*. This is the file a
  colleague is handed.

## 4. README — done, 16 October 2026

1097 lines, and **zero mentions of `sweep_order` or `bias_major`** — the largest change
of the last two days. `TODO.md` is 1071 lines. `RUN_PLAN.md`,
`ULTRA_FAST_HW_TEST_PLAN.md` and `MEASUREMENT_MODES.md` all predate §13–§17. Stale docs
are worse than missing ones. Do this after 1–3, so it is written once.

### Done — and the audit was both better and worse than the item claimed

**Better**: the config reference was not missing `sweep_order` alone. Checked against
`IQEConfig` by field name, it documented **119 of 126** fields. The seven absent were
exactly the ones §15–§21 added: `sweep_order`, `pixel_switch_settle_s`, `sweep_rotate`,
`bias_major_precondition`, `leak_ref_tries`, `leak_ref_max_frac`, `calibration_source`.
All seven are in now, with the measurement behind each, and the check is one line of
Python rather than reading.

**Worse**: `TODO.md`, `RUN_PLAN.md` and `ULTRA_FAST_HW_TEST_PLAN.md` had already gone in
item 1, so the stale-docs problem was smaller than stated — but several statements in
the README were not merely stale, they were **refuted**:

* the two-repo diagram, the `../SMILE_IQE_Analysis` link and *"the two share no code"*,
  all dead as of item 2, plus `python analyze_array.py` with the wrong path
* *"`sweep_direction` — run both on the same pixels to gauge self-heating"*, written
  before §23–§24 measured what the knob does, and before `validate()` made one of its
  two values unusable with `dead_abort`
* the 50 µF suspect list still blamed the capacitor for direction hysteresis, which
  §24 refuted: direction moves the current by **-0.027 %**
* *"is the low-J current actually LED current?"* still headed as an open question
* `--calibration` precedence quoted without `configs/local.yaml`, which item 3 added
  one commit earlier, and the gitignore rationale still described the `git archive`
  share export instead of `git pull`

**Not cut**: the ~550 lines of investigation narrative. The original plan was to delete
it as superseded by the report — but the report left the repo the same day, so those
sections are now the **only** account of the physics a reader of this repo gets. They
were corrected and annotated instead of removed.

**Added**: a `What the numbers carry` table, which is the thing a colleague actually
needs — every known systematic, its size, and whether it cancels in EQE. The
report had the derivations and nothing had the summary.

Also updated in the same pass: `analysis/README.md` (still titled
`# SMILE_IQE_Analysis`), and `KNOWN_BUGS.md`, where two open defects are now closed by
§21 and §25 and one new one is recorded — `--switch-hold-s` is silently truncated by
`--samples`, which has now cost three run lists.

## 5. The cluster / strided scan — right idea, wrong reason

**The speed argument does not survive the real numbers.**
`pseudo_implementation_clusterscan.py` costs a measurement at `integration_time_s=1.0`
and reports `total_measurements / 3600` hours. The actual fast scan is **29.5 ms per
pixel** — 262 144 pixels in 2.15 h, run 261015-0015. The model is **33× pessimistic**.

So the brute-force full-array scan is already 2.2 h and is not the bottleneck: the
100-pixel sweep in the same run took 4.8 h. A 2–3× group-testing win saves ~1.3 h of a
7 h night, and buys it by giving up per-pixel resolution.

**The signal argument is much stronger**, and the brainstorm half-stated it: *"one could
also measure the pooled IQE of 4 pixels or so at once. Increases the current,
potentially allowing a better measurement."*

Open question 1 — the optical floor — is the only real obstacle left in the
investigation: the PM400 dark reading wanders 75 pW under a 210 pW signal, and faint-end
power scatters 8.5–8.9 % under both sweep methods. Lighting K pixels multiplies the
optical signal by K. A 16-pixel cluster turns 218 pW at −0.80 V into ~3.5 nW, clear of
the drift by 50×. Pooled EQE = ΣP/ΣI is a current-weighted mean of the members, which is
arguably the right array figure anyway.

**The hardware allows it today.** `smile/utils/display.py: send_image()` takes an
arbitrary frame and `blank_frame()` already uses it, so `set_pixels(coords, bitval)` is
a small addition on top of `ImageProcessor._chunks_of_32`. Two constraints:

* the quadrant enables (`tl_pixen`/`tr_pixen`/…) are **per frame**, so a cluster that
  straddles a 256×256 boundary needs several bits set. Simplest rule: keep a cluster
  inside one quadrant.
* **the leakage reference must pool too** — the same K pixels at bitval 0, never a
  blanked frame (§6: a blanked frame conducts ~4× the bitval-0 level).

Proposed spike, one afternoon: add `set_pixels`, then measure 1 / 4 / 16 pixels at
−0.90 V and see whether the faint-end scatter falls as √K or stays at 8.5 %. If it
falls, that is the first real progress on open question 1 — and the strided yield scan
comes along free as a second use of the same primitive.

Correct the 1 s → 30 ms cost in `yieldsimu.py` before trusting its break-even maps. It
moves every curve.

### Done, 17 October 2026 — the primitive and the harness, not yet the answer

The spike is built and runs end to end in simulation. The bench run is what is left,
and it is ~2 minutes of instrument time.

* **`smile.utils.display.set_pixels(dev, coords, value, cfg)`** — K pixels of one
  quadrant in **one** `RESET_DIS` + `USB_2_DIS` pair, the same cost as one pixel.
  It reuses `_pixel_row_constructor` and ORs the rows together, so the nibble
  addressing lives in exactly one place, and it hands the payload to `set_row`,
  which already derives `roi_lrow` from the payload length. No second copy of the
  frame builder.
* **`SmileFPGA.set_pixels(coords, bit_val)`** — logical coordinates, and the two
  guards the hardware needs. A cluster that straddles a 256 boundary raises rather
  than silently lighting the near quadrant's mirror image, and an out-of-array
  coordinate raises rather than taking `map_logical_to_physical`'s legacy fall back
  to TL, which for one pixel is harmless and for a cluster collides two members.
* **`characterize_capacitance.py --what cluster`** — nested K (K=4 is the first four
  members of the K=16 set, never a fresh sample), the pooled bitval-0 reference, and
  one verdict column, `rel_scatter_x_sqrt_k`. The bias is charged once and every
  later wait is a frame change at a held bias.

Six checks in `test_characterize_capacitance.py`, three of them on the bit packing:
one pixel through `set_pixels` packs identically to `set_pixel`, K pixels are the OR
of their single-pixel frames and nothing leaks into an empty row, and the guards
reject what they must. Two more drive `exp_cluster` against the sim rig and assert
the reference pools and the K sets nest.

Still open, and deliberately not built: the pooled path is **not** wired into
`measure_iqe`'s sweep. That is only worth doing if the bench run says the scatter
falls, and the run is the cheap half.

### Done, 17 October 2026 — pooling wired in, and the simulator corrected

**`pool_clusters` sweeps pooled targets**, off by default and deliberately so. A
pooled row is not a pixel: it carries `POOL_K` and the **sum** of its members' light
and current. `X`,`Y` is the first member, so every `groupby(["X","Y"])` keeps working.
EQE needs no correction, because both sums carry the same K; **current density does**,
and `analyze_array.py` divides by `POOL_K`.

The shape it is built for is both at once — singles and pooled groups in the same
session, the singles grounding the pooled numbers under the same lamp, thermal history
and instrument state. `pool_seed` is a separate RNG stream so a pooled member is never
also a single pixel, or the grounding comparison would be circular.

One defect this turned up and fixed: **both dead-pixel tests compare a target against
the median of the others**, so an unscaled pooled target (K× the light) would have
raised the median by roughly K and condemned live single pixels. Both now judge per
member, `acquire.per_member_power`.

**`yieldsimu.py` corrected** (it lives in `C:\_D\programs\SmallProjects\SmileYieldSimu`,
outside this repo and not under version control). The 1 s/pixel placeholder is now the
measured 29.56 ms, and the claim in the line above — *"it moves every curve"* — was
**wrong**:

* the **speedup ratio does not move at all**. `speedup = brute_tests / group_tests`,
  and a per-test cost shared by both cancels. Every breakeven in that file — 70.71 %
  uniform, ~50 % blobs, ~64 % lines — is a property of the Dorfman arithmetic and the
  defect geometry, not of time, and all of them stand.
* the **absolute numbers move by 34×**, and that is what kills the motive. Brute force
  is 2.15 h, not 72.8 h. Cases C and D of that file's protocol save minutes.

Two things the correction added that matter more than the arithmetic:

* **the real array is at 99.73 % yield** (261 441 of 262 144 alive at a 2 nW
  threshold, run 261015-0015) — off the top of every chart in that file, which sweeps
  50–99 %.
* **the group test is ANALOG, and that, not Dorfman, caps K.** One dead member is a
  `1/K` deficit against a sum whose own spread is `16.4 %/√K` (the measured
  alive-pixel brightness spread, 52.21 ± 8.55 nW). That is 3.0σ at K = 4 and 1.5σ at
  K = 16, so `max_detectable_k()` returns 4.1 and `pool_size` defaults to 4. The cap
  lifts once the members' individual brightnesses are known — which is exactly what a
  single-pixel pass in the same session buys.

`yieldsimu.py` has a `_selfcheck()` on both claims and runs it before plotting.

## 6. GUI — last

Not a bad idea, but a GUI is where an interface goes to become permanent, so build it
once the run format has stopped moving.

Lazy shape when the time comes: **`tkinter`** (ships with Python, no new dependency),
and **run the CLI as a subprocess** rather than importing `measure_iqe` into the GUI
process. That gives the log window for free, makes a kill button actually work, and
means a crash in the sweep cannot take the window down. The queue is a list of
arg-lists run in sequence; the progress bar parses the `[3/16]` lines the sweep already
prints.

### Done, 17 October 2026 — built exactly to that shape

`measure_iqe/gui.py`, opened with `python -m measure_iqe --gui`. tkinter, the CLI as a
subprocess, a queue of argument lists, and the progress bar parsing `[3/16]`.

What the shape above did not say, and what the operator asked for — **full control
over what is measured, how, and what is kept**:

* **Run shape** holds the ~20 fields that decide a run, four presets, and a free-text
  `--set` box.
* **All parameters** is generated from `dataclasses.fields(IQEConfig)`, so all 132
  fields are reachable and the tab cannot drift from the dataclass.
* **Preview command** prints the argument list. There is nothing the console can reach
  that the CLI cannot, which is the property that keeps the console from becoming a
  second source of truth.

Three decisions worth recording:

* **Stop sends CTRL_BREAK, not a kill.** `safeexit` turns it into outputs off, frame
  blanked, files flushed. A killed process leaves the SMU driving the array. This is
  the one thing in that file that is not a convenience.
* **Only CHANGED fields become `--set`.** An explicit `--set` outranks both `--config`
  and `configs/local.yaml`, so a console that always sent all 132 would silently
  overwrite the machine's own settings with its idea of the defaults — the exact
  failure item 3 was written to avoid.
* **The presets' labels are tested against the sampler**, because an operator reads
  the label, not the selection log. Writing that test is what caught the pooled
  selection asking the sampler for a quarter of the pixels it needed.

The calibration check — *"warn them that no calibration is available"* — is the best
part of that idea and needs no GUI. It belongs in the CLI now: `--array Array43_IQE`
with no `calibrations/Array43_IQE.yaml` should say so loudly before spending 7 hours.

> **Already shipped** (`cli.py`, the `--array` branch). A missing calibration prints
> `NO CALIBRATION found. Looked in: ...` and names the `--calibrate` command that fixes
> it, before any instrument is touched. The rest of item 6 is untouched.
