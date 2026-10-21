"""What produced a run: the commit, the working tree's cleanliness, the machine.

Written automatically into every run directory as ``RUN_PROVENANCE.json`` and folded
into the saved ``*_config.json``. The point is stated once, here:

    A git TAG is the wrong mechanism for this. It needs someone to remember to move it
    when a default changes, and that discipline fails within a month -- after which
    every run carries a tag that is quietly a lie. The commit SHA cannot be forgotten
    because nobody has to do anything. Tags stay useful as a human label on top
    ("v0.3 -- bias_major"), never as the record of what ran.

``dirty`` is the field that matters most. A run taken against uncommitted edits cannot
be reproduced from the SHA alone, and ``changed`` names the files, so at least the blast
radius is known. ``diff_sha256`` fingerprints the uncommitted diff itself: two runs with
the same SHA and the same ``diff_sha256`` really did run identical code.
"""
from __future__ import annotations

import datetime
import functools
import hashlib
import pathlib
import json
import platform
import re
import socket
import subprocess
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parent.parent


def _git(*args: str, cwd: Path | None = None) -> str | None:
    """Run a git command, or return None if git/the repo is unavailable."""
    try:
        # errors="replace", because `git diff` of a run directory carries CSV bytes that
        # are not valid in the console's codepage. Without it Python decodes in a reader
        # THREAD, the exception never reaches this frame, and stdout silently comes back
        # as None -- which is how this first showed up.
        out = subprocess.run(("git",) + args, cwd=str(cwd or REPO), capture_output=True,
                             text=True, encoding="utf-8", errors="replace", timeout=10)
    except (OSError, subprocess.SubprocessError):
        return None
    if out.returncode != 0 or out.stdout is None:
        return None
    return out.stdout.strip()


def _changed_paths(status: str | None) -> tuple[list[str], list[str]]:
    """``(tracked_edits, untracked)`` out of ``git status --porcelain``.

    Slicing ``ln[3:]`` is the obvious way and it is wrong: it silently eats the first
    character of a path whenever the prefix is not exactly three columns, and the only
    symptom is a plausible-looking filename (``RUN_FORMAT.md`` arriving as
    ``UN_FORMAT.md``). Split on whitespace after the status columns instead, and take
    the destination of a rename, which is the file that actually differs.

    The two kinds are kept apart because only one of them can change what ran. An
    UNTRACKED file -- status "??" -- is a file git has never seen: a scratch
    ``run.bat``, an editor backup, a stray CSV. It cannot affect the code that
    executed, so a run taken beside one is still reproducible from its commit. Lumping
    the two together made every run from a bench machine with a scratch file announce
    "this run cannot be reproduced from the commit alone", which is both false and the
    fastest way to teach someone to ignore the warning that matters.
    """
    edits, untracked = [], []
    for ln in (status or "").splitlines():
        m = re.match(r"^(..)\s+(.*)$", ln)
        if not m:
            continue
        path = m.group(2).strip()
        if " -> " in path:                       # "R  old -> new"
            path = path.split(" -> ", 1)[1]
        path = path.strip('"')                   # git quotes paths with odd characters
        if path:
            (untracked if m.group(1) == "??" else edits).append(path)
    return edits, untracked


@functools.lru_cache(maxsize=1)
def _snapshot() -> tuple[tuple[str, Any], ...]:
    """The git part, computed once per process.

    A run calls provenance() four times (the stamp plus each saved config) and the
    commit cannot change underneath it, so this would otherwise be ~24 git invocations
    -- and `git diff HEAD` on a dirty tree is the slow one. Cached as a tuple of pairs
    because lru_cache needs the result to be hashable; callers get a fresh dict.
    """
    status = _git("status", "--porcelain")
    changed, untracked = _changed_paths(status)
    diff = _git("diff", "HEAD") if changed else None
    out: dict[str, Any] = {
        "commit": _git("rev-parse", "HEAD"),
        "commit_short": _git("rev-parse", "--short", "HEAD"),
        "branch": _git("rev-parse", "--abbrev-ref", "HEAD"),
        "committed": _git("log", "-1", "--format=%cI"),
        # --always so a commit with no tag behind it still yields something
        "describe": _git("describe", "--tags", "--always", "--dirty"),
        # Only edits to TRACKED files make a run unreproducible. Untracked ones are
        # recorded beside them, because knowing a stray file was on disk is sometimes
        # useful, but they never set this flag.
        "dirty": bool(changed),
        "changed": sorted(changed)[:50],
        "untracked": sorted(untracked)[:50],
        "diff_sha256": hashlib.sha256(diff.encode()).hexdigest()[:16] if diff else None,
        "started": datetime.datetime.now().astimezone().isoformat(timespec="seconds"),
        "host": socket.gethostname(),
        "python": platform.python_version(),
        "platform": platform.platform(),
    }
    return tuple(out.items())


def provenance(extra: dict[str, Any] | None = None) -> dict[str, Any]:
    """Everything needed to identify the code and machine behind a run."""
    out = dict(_snapshot())
    if extra:
        out.update(extra)
    return out


def write_provenance(run_dir: str | Path, extra: dict[str, Any] | None = None) -> Path:
    """Write ``RUN_PROVENANCE.json`` into a run directory and return the path."""
    p = Path(run_dir) / "RUN_PROVENANCE.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(provenance(extra), indent=2), encoding="utf-8")
    return p


def run_slug(sample_name: str, run_dir: str | Path) -> str:
    """``<sample>_<stamp>``, the prefix every output file in a run carries.

    Taken from the run directory's own name so a resumed run keeps the prefix of the
    run it is resuming, rather than stamping the files with the resume's clock.
    """
    stamp = Path(run_dir).name.split("_", 1)[0]
    safe = "".join(c if (c.isalnum() or c in "-._") else "_" for c in str(sample_name))
    return f"{safe}_{stamp}" if stamp else safe


def run_file(run_dir: str | Path, relpath: str) -> Path:
    """Find one of a run's files by its name WITHOUT the ``<sample>_<stamp>_`` prefix.

    Runs taken before 16 October 2026 carry no prefix, and they are still the comparison
    baseline for every result in the investigation report, so every reader has to accept
    both spellings. Rather than scatter ``*`` through a dozen globs, say it once here.

        run_file(d, "NVLEDSweep/Sweep/aggregated_bitval=15_mean_std.csv")

    Raises FileNotFoundError naming the directory, because a reader that silently finds
    nothing produces an empty analysis rather than an error.
    """
    rel = pathlib.PurePosixPath(relpath)
    folder = Path(run_dir).joinpath(*rel.parts[:-1])
    hits = sorted(folder.glob(f"*{rel.name}"))
    exact = [h for h in hits if h.name == rel.name]
    if exact:
        return exact[0]
    if not hits:
        raise FileNotFoundError(f"no *{rel.name} in {folder}")
    return hits[0]


def demo() -> None:
    """Self-check: the slug must survive odd sample names and odd run dirs."""
    assert run_slug("Array42_IQE", "261015-1321_Array42_IQE_ROI_(32, 32, 63, 63)") == \
        "Array42_IQE_261015-1321"
    # a name with separators in it must not create directories or break a glob
    assert run_slug("A/B 42", "261015-1321_x") == "A_B_42_261015-1321"
    # second-resolution stamps (two runs in one minute) keep their own stamp
    assert run_slug("X", "261015-132159_X_ROI") == "X_261015-132159"
    # a directory with no stamp at all still yields a usable prefix
    assert run_slug("X", "somewhere") == "X_somewhere"

    import tempfile
    with tempfile.TemporaryDirectory() as d:
        sweep = Path(d) / "NVLEDSweep" / "Sweep"
        sweep.mkdir(parents=True)
        old = sweep / "aggregated_bitval=15_mean_std.csv"
        old.write_text("x", encoding="utf-8")
        rel = "NVLEDSweep/Sweep/aggregated_bitval=15_mean_std.csv"
        assert run_file(d, rel) == old, "an unprefixed (pre-16-Oct) run must still resolve"
        new = sweep / "Array42_261015-1321_aggregated_bitval=15_mean_std.csv"
        new.write_text("x", encoding="utf-8")
        # both present: the exact name wins, so a mixed directory is unambiguous
        assert run_file(d, rel) == old
        old.unlink()
        assert run_file(d, rel) == new, "a prefixed run must resolve too"
        try:
            run_file(d, "NVLEDSweep/Sweep/nothing.csv")
        except FileNotFoundError:
            pass
        else:
            raise AssertionError("a missing file must raise, not return nothing")

    # the porcelain parse: the bug this replaced ate the first character of a path
    got = _changed_paths(" M RUN_FORMAT.md\n?? ROADMAP.md\nA  a/b.py\n")
    assert got == (["RUN_FORMAT.md", "a/b.py"], ["ROADMAP.md"]), got
    assert _changed_paths("R  old/x.py -> new/y.py") == (["new/y.py"], [])
    assert _changed_paths('A  "odd name.py"') == (["odd name.py"], [])
    assert _changed_paths(None) == ([], []) and _changed_paths("") == ([], [])

    # An UNTRACKED file must not make a run "unreproducible". Run 261020-1454 was
    # flagged dirty over a scratch run.bat: git had never seen the file, the stored
    # diff_sha256 was None, and the run was reproducible from its commit all along.
    # A warning that fires on every bench run with a scratch file beside it teaches
    # people to ignore the warning that matters.
    edits, stray = _changed_paths("?? run.bat\n?? notes.txt\n")
    assert edits == [] and stray == ["run.bat", "notes.txt"], (edits, stray)
    assert _changed_paths(" M measure_iqe/acquire.py\n?? run.bat\n")[0] == [
        "measure_iqe/acquire.py"], "a real edit must still be caught"

    # _git must never return something other than str|None, whatever the diff holds
    for args in (("rev-parse", "HEAD"), ("diff", "HEAD"), ("not-a-git-command",)):
        got = _git(*args)
        assert got is None or isinstance(got, str), (args, type(got))

    p = provenance()
    assert set(p) >= {"commit", "dirty", "started", "host"}
    # `extra` must not leak into the cached snapshot shared by every other caller
    assert provenance({"sample_name": "X"})["sample_name"] == "X"
    assert "sample_name" not in provenance()
    assert isinstance(p["dirty"], bool)
    assert p["commit"] is None or len(p["commit"]) == 40, p["commit"]
    # dirty and changed must agree -- a dirty tree with no named files is useless
    assert p["dirty"] == bool(p["changed"]) or len(p["changed"]) == 50
    assert isinstance(p["untracked"], list)
    assert not (p["untracked"] and not p["changed"] and p["dirty"]), (
        "an untracked file alone must never set dirty")
    print(f"ok  provenance: {p['describe']} on {p['branch']}, "
          f"dirty={p['dirty']} ({len(p['changed'])} edit(s), "
          f"{len(p['untracked'])} untracked)")


if __name__ == "__main__":
    demo()
