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
| 4 | README | yes, after 1–3 | ~2 h | next |
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

> **Already shipped** (`cli.py`, the `--array` branch). A missing calibration prints
> `NO CALIBRATION found. Looked in: ...` and names the `--calibrate` command that fixes
> it, before any instrument is touched. The rest of item 6 is untouched.
