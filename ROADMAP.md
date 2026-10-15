# Roadmap

Triage of a brainstorm from 15 October 2026, with the evidence each verdict rests on.
Items 0 and 1 are done; the rest are proposals, not decisions.

| # | item | verdict | effort | status |
|---|---|---|---|---|
| 0 | run provenance: commit SHA, calibration copy, names in filenames | do first | ~2 h | **done** |
| 1 | repo cleanup | yes, but not via an untracked `_obsolete/` | ~1 h | **done** |
| 2 | merge the analysis repo in; `characterize_capacitance` into `measure_iqe` | yes | ~3 h | open |
| 3 | config: a local override for `base_dir` | yes — but **not** by auto-loading `standard.yaml` | ~1 h | open |
| 4 | README | yes, after 1–3 | ~2 h | open |
| 5 | cluster / strided scan | pursue — but not for the reason it was proposed | spike first | open |
| 6 | GUI | last | 1–2 days | open |

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

Two real defects in that file, independent of the above:

* the sync test compares `IQEConfig.from_file(...) == IQEConfig()`, so a **missing** key
  falls back to the default and passes. `sweep_rotate` and `bias_major_precondition`
  were both added without the test noticing. It only catches wrong values, never
  absent ones.
* its comments still assert the **refuted** switch bound — *"a frame change at a held
  bias costs under 0.5 %"*, *"measured need is under 1 s"*. This is the file a
  colleague is handed.

## 4. README

1097 lines, and **zero mentions of `sweep_order` or `bias_major`** — the largest change
of the last two days. `TODO.md` is 1071 lines. `RUN_PLAN.md`,
`ULTRA_FAST_HW_TEST_PLAN.md` and `MEASUREMENT_MODES.md` all predate §13–§17. Stale docs
are worse than missing ones. Do this after 1–3, so it is written once.

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

## 6. GUI — last

Not a bad idea, but a GUI is where an interface goes to become permanent, so build it
once the run format has stopped moving.

Lazy shape when the time comes: **`tkinter`** (ships with Python, no new dependency),
and **run the CLI as a subprocess** rather than importing `measure_iqe` into the GUI
process. That gives the log window for free, makes a kill button actually work, and
means a crash in the sweep cannot take the window down. The queue is a list of
arg-lists run in sequence; the progress bar parses the `[3/16]` lines the sweep already
prints.

The calibration check — *"warn them that no calibration is available"* — is the best
part of that idea and needs no GUI. It belongs in the CLI now: `--array Array43_IQE`
with no `calibrations/Array43_IQE.yaml` should say so loudly before spending 7 hours.
