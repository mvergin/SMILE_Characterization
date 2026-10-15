# Attic — what was removed, and where to find it

Nothing here is lost. Git keeps every version of a tracked file forever, so removing
one from the working tree costs nothing and reading it back is one command:

```bash
git log --oneline --all -- <path>          # when it existed
git show <commit>:<path>                   # read it
git checkout <commit> -- <path>            # bring it back
```

**This is why these files were deleted rather than moved into an untracked
`_obsolete/`.** Moving a tracked file into an ignored folder *removes it from git* and
breaks `git log --follow` at the same time — the history is the archive, and a folder
full of dead files beside the live ones is not.

Removed **16 October 2026**, all of it dead on the day: nothing imported it and no test
covered it. The last commit holding every one of these is the parent of the commit that
removed them.

## Scripts

| file | lines | why |
|---|---|---|
| `debug_experimental_sweep.py` | 261 | nothing referenced it, not even the README |
| `test_pm400_array_timing.py` | 131 | nothing referenced it; not run by either suite |
| `test_pm400_buffer_access.py` | 517 | bring-up for the ultra-fast mode, which `MEASUREMENT_MODES.md` records as unmaintained reference code |
| `autozero_bench.py` | 236 | one-off bench for the autozero setting; the answer is `sweep_autozero: AUTO` in the config |
| `benchmark_sweep.py` | 206 | superseded by `--calibrate`, which measures the dwell instead of timing a guess |
| `sweep_variants.py` | 325 | superseded by `sweep_order` and the per-segment effort knobs |

## Documents

| file | why |
|---|---|
| `RUN_PLAN.md` | the bench plan for the capacitance/low-J investigation. That investigation ran to completion — the result is `INVESTIGATION_REPORT.md` §1–§17 in the analysis repo, which supersedes the plan entirely |
| `ULTRA_FAST_HW_TEST_PLAN.md` | hardware bring-up for the ultra-fast mode, never executed, and the mode is unmaintained reference code |
| `TODO.md` | a 1071-line April audit of `smile_automatic_gui_2602b.py` and `instrumentlib.py` — **neither file is in this repository**. It reviews the predecessor codebase |
| `Ideen.txt` | two lines, both now answered. Kept verbatim below so nothing is dropped |

### `Ideen.txt`, in full

> Niedrig Strom Messen mit den Ranges. Muss ja prob high to low, aber ist die osci nach
> jedem schritt da?
>
> Das low to high nicht geht indiziert ja schon das es ne cap ist

Both were right, and both are now measured rather than suspected:

* the "oscillation" after each step is **white noise from a measure range the compliance
  limit selected**, not ringing — flat spectrum, no mains line (§1, §5);
* the array does carry a capacitance — **42.8 µF on the common cathode** (§2) — but what
  the sweep waits out is **not** an RC charge. It is a power-law charge relaxation,
  `|I − I∞| ∝ t^−1.3`, which is why small steps do not settle faster (§3, §4).

## What was deliberately kept

* **`caprun/`** — 1.8 MB of real run data, and the only measurement data tracked in this
  repository. Runs `261014-0007` and `261014-0035` are the evidence behind §10 and
  §11–§12 of the investigation report, cited by name. Deleting them would strand two
  sections of the report.
* **`measurement/`, `data/`, `smile_postprocess.py`, `smile_heatmap.py`** — the
  transient / ultra-fast / hardware-paced-TSP techniques that predate `measure_iqe`.
  `MEASUREMENT_MODES.md` records the decision to keep them as unmaintained reference
  implementations and says why. That decision stands; this cleanup did not revisit it.
* **`KNOWN_BUGS.md`** — rewritten rather than removed. Its first entry
  (*"High Capacitance Mode geht atm nicht"*) was fixed in §1, and its last
  (*"Er schreibt jetzt auch immer den version namen mit in die config"*) describes a
  provenance record that had been lost and is now restored properly — see
  `measure_iqe/provenance.py`.
