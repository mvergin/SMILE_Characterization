"""Analyse a run folder, or every run under a results folder. It picks the tools from
what the folder holds and skips a run that is already up to date.

    python analysis/analyze.py results                # every run underneath
    python analysis/analyze.py "results/<run>"        # one run
    python analysis/analyze.py results --dry-run      # what would run, and why
    python analysis/analyze.py results --force        # redo everything

| folder holds                                 | tool                 |
|----------------------------------------------|----------------------|
| FastScan/                                    | analyze_array.py     |
| calibration_traces.npz                       | plot_calibration.py  |
| traces.npz, switchmatrix.csv or isrc.csv     | plot_transients.py   |
| leakmatrix.csv                               | plot_leakmatrix.py   |

A tool runs again only when its source or the run's input files changed since its last
clean finish. ANALYSIS_STATE.json in the run folder records which source (SHA-256)
ran on which inputs, and when. The tool's output goes to ANALYSIS_LOG.txt beside it
and to the console. Each tool runs in its own process, one after the other, so a crash
or a memory leak in one run cannot affect the next. The per-pixel ABC fits have their
own cache (analyze_array.py), so a plot-only change replots without refitting.
"""
from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
STATE = "ANALYSIS_STATE.json"
LOG = "ANALYSIS_LOG.txt"
TOOLS = {   # tool: does this run folder need it
    "analyze_array.py": lambda d: (d / "FastScan").is_dir(),
    "plot_calibration.py": lambda d: (d / "calibration_traces.npz").is_file(),
    "plot_transients.py": lambda d: any((d / f).is_file() for f in
                                        ("traces.npz", "switchmatrix.csv", "isrc.csv")),
    "plot_leakmatrix.py": lambda d: (d / "leakmatrix.csv").is_file(),
}
# What the tools write. Everything else in a run folder is input.
OUT_DIRS = {"REPORT_assets", "switch_sync", "switchmatrix", "IQE_EQE", "comparison"}
OUT_SUFFIX = (".png", ".md")
OUT_FILES = {STATE, LOG}


def code_hash(tool: str) -> str:
    # line endings normalised: a CRLF checkout of the same source is the same code
    return hashlib.sha256((HERE / tool).read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def inputs_hash(run: Path) -> str:
    """Path, size and mtime of every input file: changed data, rerun."""
    h = hashlib.sha256()
    for root, dirs, files in os.walk(run):
        dirs[:] = sorted(d for d in dirs if d not in OUT_DIRS)
        for f in sorted(files):
            if f in OUT_FILES or f.endswith(OUT_SUFFIX):
                continue
            st = os.stat(os.path.join(root, f))
            rel = os.path.relpath(os.path.join(root, f), run)
            h.update(f"{rel}|{st.st_size}|{st.st_mtime_ns}\n".encode())
    return h.hexdigest()


def find_runs(path: Path) -> list[Path]:
    """`path` itself when a tool applies to it, else every such folder underneath."""
    runs = []
    for root, dirs, _ in os.walk(path):
        d = Path(root)
        if any(need(d) for need in TOOLS.values()):
            runs.append(d)
            dirs[:] = []           # a run's own subfolders are not runs
        else:
            # FastScan_TL etc. are a calibration's per-quadrant scans, part of its record
            dirs[:] = sorted(x for x in dirs if x not in OUT_DIRS
                             and not x.startswith("FastScan_"))
    return runs


def load_state(run: Path) -> dict:
    try:
        return json.loads((run / STATE).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def why_stale(tool: str, prev: dict, code: str, inputs: str) -> str | None:
    """None when the last clean finish used this code on these inputs."""
    if not prev:
        return "never run"
    if prev.get("exit") != 0:
        return f"last run failed (exit {prev.get('exit')})"
    if prev.get("code") != code:
        return f"{tool} changed"
    if prev.get("inputs") != inputs:
        return "input files changed"
    return None


def run_tool(tool: str, run: Path, extra: list[str], log_f) -> int:
    cmd = [sys.executable, "-X", "utf8", "-u", str(HERE / tool), str(run), *extra]
    with subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                          text=True, encoding="utf-8", errors="replace") as p:
        for line in p.stdout:
            sys.stdout.write("    " + line)
            log_f.write(line)
            log_f.flush()
    return p.returncode


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("path", help="a run folder, or a folder of runs (e.g. results)")
    ap.add_argument("--force", action="store_true", help="rerun even when up to date")
    ap.add_argument("--dry-run", action="store_true", help="list what would run")
    ap.add_argument("--jobs", type=int, default=None,
                    help="processes for the per-pixel ABC fits (analyze_array). "
                         "Default: analyze_array's own")
    args = ap.parse_args(argv)
    for s in (sys.stdout, sys.stderr):
        try:
            s.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass
    root = Path(args.path).resolve()
    if not root.is_dir():
        sys.exit(f"{root} is not a folder")
    runs = find_runs(root)
    print(f"{len(runs)} run folder(s) under {root}")
    codes = {t: code_hash(t) for t in TOOLS}
    table, failed = [], 0
    for i, run in enumerate(runs, 1):
        tools = [t for t, need in TOOLS.items() if need(run)]
        state = load_state(run)
        inputs = inputs_hash(run)
        todo = []
        for t in tools:
            why = "--force" if args.force else why_stale(t, state.get(t, {}), codes[t], inputs)
            if why:
                todo.append((t, why))
            else:
                table.append((run.name, t, "up to date", ""))
        name = run.relative_to(root) if run != root else run.name
        if not todo:
            print(f"[{i}/{len(runs)}] {name}: up to date")
            continue
        for t, why in todo:
            print(f"[{i}/{len(runs)}] {name}: {t} ({why})")
        if args.dry_run:
            table += [(run.name, t, "would run", why) for t, why in todo]
            continue
        with open(run / LOG, "a", encoding="utf-8") as log_f:
            for t, why in todo:
                extra = ([f"--jobs={args.jobs}"]
                         if args.jobs and t == "analyze_array.py" else [])
                stamp = datetime.datetime.now().isoformat(timespec="seconds")
                log_f.write(f"\n===== {stamp} {t} ({why}) =====\n")
                t0 = time.perf_counter()
                rc = run_tool(t, run, extra, log_f)
                dt = time.perf_counter() - t0
                state[t] = dict(code=codes[t], inputs=inputs, exit=rc, seconds=round(dt, 1),
                                finished=datetime.datetime.now().isoformat(timespec="seconds"))
                # written after every tool, so an interrupted batch keeps what finished
                (run / STATE).write_text(json.dumps(state, indent=1), encoding="utf-8")
                status = "done" if rc == 0 else f"FAILED (exit {rc}), see {run / LOG}"
                failed += rc != 0
                print(f"[{i}/{len(runs)}] {name}: {t} {status} in {dt:.0f} s")
                table.append((run.name, t, "done" if rc == 0 else f"FAILED {rc}", f"{dt:.0f} s"))
    if table:
        w = max(len(r[0]) for r in table)
        print("\nSummary")
        for r in table:
            print(f"  {r[0]:<{w}}  {r[1]:<20} {r[2]:<11} {r[3]}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
