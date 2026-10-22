"""Operator console for ``measure_iqe`` and ``characterize_capacitance`` -- tkinter, stdlib only.

    python -m measure_iqe --gui

ROADMAP item 6. Three decisions it is built on, all of which cost nothing and buy a
lot:

* **tkinter**, because it ships with Python. A measurement machine that needs a pip
  install before it can run a sweep is a machine that will not run the sweep.
* **the CLI as a subprocess**, never ``import measure_iqe`` into this process to run
  it. The log window is then free (it is the child's stdout), the Stop button actually
  works, and a crash inside a sweep cannot take the window down with it. Everything
  this window can do, it does by building an argument list a person could have typed
  -- and it shows that list, so they can.
* **generated, not hand-built**: the command-line options of both tools come from
  their own argparse parsers, every config field from the IQEConfig dataclass, and
  every tooltip from the parser's help or the field's comment in config.py. A new
  option or field appears here without touching this file, and a test fails if one
  has no explanation.

Stop sends **CTRL_BREAK**, not a kill. ``measure_iqe.safeexit`` catches it in both tools
and parks the hardware: outputs off, frame blanked. A killed process leaves the SMU
driving the array. That is the one thing in this file that is not a convenience.

The queue is a list of argument lists run in sequence, so an operator can leave a
night's work and come back to it. Jobs of both tools can share one queue.
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import os
import queue
import re
import shlex
import signal
import subprocess
import sys
import threading
from pathlib import Path

import tkinter as tk
from tkinter import filedialog, messagebox, ttk

from .config import IQEConfig

REPO = Path(__file__).resolve().parent.parent

# measure_iqe prints "  [3/16] pixel (x,y)" or "  [2/30] -1.300 V"; characterize_capacitance
# prints "PIXEL 2/4 (x,y)". One regex for the progress bar.
_PROG = re.compile(r"(?:\[|PIXEL )(\d+)/(\d+)")


# ----------------------------------------------------------------------
#  What the operator sets, and what they can still reach
# ----------------------------------------------------------------------
# The fields that shape a run, in the order an operator decides them. Everything else
# stays reachable on the All parameters tab, which is generated from the dataclass and
# therefore cannot drift from it.
SHAPE: list[tuple[str, str, str]] = [
    ("sample_name", "Sample", "goes in the run directory name and every filename"),
    ("roi", "ROI", 'FULL / TL / TR / BL / BR, or [x1,y1,x2,y2]'),
    ("sweep_select", "Selection", "blocks | spatial | stride"),
    ("block_size", "Block edge (px)", "ROI/block_size blocks per axis"),
    ("pixels_per_block", "Singles per block", "single pixels measured per block"),
    ("block_seed", "Single seed", "change between runs to cover new pixels"),
    ("pool_groups", "Pooled groups per block",
     "[] = off. e.g. [4] or [2,4,8]. One group per entry, inside each block"),
    ("pool_include_singles", "Pool shares a single",
     "a group takes one of its block's singles -> paired comparison"),
    ("pool_pm_range", "Pooled PM400 range (W)", "0 = same as pm_range"),
    ("nvled_v_fastscan", "Fast-scan bias (V)", "the ONLY bias that judges alive/dead"),
    ("fastscan_voltages", "Extra fast-scan biases", "e.g. [-1.05,-1.65]"),
    ("fastscan_only", "Fast scan only", "true = scan every bias above, no sweep"),
    ("n_fine", "Sweep points", "biases per target"),
    ("nvled_v_coarse_start", "Sweep from (V)", ""),
    ("nvled_v_coarse_stop", "Sweep to (V)", ""),
    ("sweep_order", "Order", "bias_major pays each dwell once for all targets"),
    ("sweep_direction", "Direction", "high_to_low | low_to_high"),
    ("leak_reference", "Leakage reference", "subtract the same target at bitval 0"),
    ("leak_ref_per_point_above_v", "Reference per point above (V)",
     "below this a cached per-pixel reference is used. null = every point"),
    ("save_raw_samples", "Keep raw bursts", "every sample, not just the summary"),
    ("trace_settling", "Keep dwell traces", "poll through the dwell instead of sleeping"),
]

# One click each. The second is the shape the operator asked for by name.
PRESETS: dict[str, dict] = {
    "Full array, 100 singles": dict(
        roi="FULL", sweep_select="blocks", block_size=52, pixels_per_block=1,
        sweep_order="bias_major"),
    # One group per block, so the counts follow the block grid: blocks never cross the
    # quadrant edge at 256, so each axis takes 2*round(256/block_size) blocks. 52 px
    # gives 10 per axis and 100 blocks -> 200 singles and 100 groups of 4.
    "Full array, 200 singles + 100 pooled x4": dict(
        roi="FULL", sweep_select="blocks", block_size=52, pixels_per_block=2,
        pool_groups=(4,), sweep_order="bias_major"),
    # Mixed K in one run: every block gets a 2, a 4 and an 8, so the K-dependence of
    # the pooled faint end comes out of the array itself, not only the bench rig.
    "Full array, 36 singles + 36 each of K=2,4,8": dict(
        roi="FULL", sweep_select="blocks", block_size=102, pixels_per_block=1,
        pool_groups=(2, 4, 8), sweep_order="bias_major"),
    "Full array, 1000 singles": dict(
        roi="FULL", sweep_select="blocks", block_size=52, pixels_per_block=10,
        sweep_order="bias_major"),
    # --fastscan in the CLI. Add biases in "Extra fast-scan biases".
    "Fast scan only, full array": dict(roi="FULL", fastscan_only=True),
    "Quick check, 16x16 ROI": dict(
        roi=(0, 0, 15, 15), sweep_select="blocks", block_size=8, pixels_per_block=1,
        n_fine=6),
}


def _as_text(v) -> str:
    """A field value as the operator should see and type it."""
    if v is None:
        return "null"
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, (tuple, list)):
        return "[" + ",".join(_as_text(x) for x in v) + "]"
    return str(v)


def _defaults() -> dict[str, str]:
    d = IQEConfig()
    return {f.name: _as_text(getattr(d, f.name)) for f in dataclasses.fields(IQEConfig)}


_FIELD = re.compile(r"^    (\w+)\s*:\s*[^=#]+=.*?(?:#\s?(.*))?$")
_COMMENT = re.compile(r"^(\s+)#\s?(.*)$")


def field_docs() -> dict[str, str]:
    """Every IQEConfig field's explanation, read from its comment in config.py.

    The comments ARE the documentation of this project -- configs/standard.yaml copies
    them -- so the tooltips read them rather than keep a second copy that would drift.
    Three layouts occur, tried in this order: an inline comment plus its deeper-indented
    continuation lines; a comment block directly above the field; a comment block
    directly below it. Section rulers (``# ── … ──``) are skipped.
    """
    src = (Path(__file__).with_name("config.py")).read_text(encoding="utf-8").splitlines()
    names = {f.name for f in dataclasses.fields(IQEConfig)}
    docs: dict[str, str] = {}
    for k, line in enumerate(src):
        m = _FIELD.match(line)
        if not m or m.group(1) not in names or m.group(1) in docs:
            continue
        text = [m.group(2)] if m.group(2) else []
        j = k + 1
        while j < len(src) and (c := _COMMENT.match(src[j])) and len(c.group(1)) > 4:
            text.append(c.group(2)); j += 1
        if not text:                                   # block above
            j = k - 1
            while j >= 0 and (c := _COMMENT.match(src[j])) and len(c.group(1)) == 4:
                text.insert(0, c.group(2)); j -= 1
        if not text:                                   # block below
            j = k + 1
            while j < len(src) and (c := _COMMENT.match(src[j])) and len(c.group(1)) == 4:
                text.append(c.group(2)); j += 1
        docs[m.group(1)] = " ".join(t.strip() for t in text if t.strip() and "──" not in t)
    return docs


# ----------------------------------------------------------------------
#  Command-line options, generated from a tool's own parser
# ----------------------------------------------------------------------
@dataclasses.dataclass(frozen=True)
class Opt:
    """One argparse option as the console shows it."""
    dest: str
    flags: tuple[str, ...]
    kind: str            # "flag" (store_true), "bool" (--x/--no-x), "value"
    nargs: object
    default: str         # as typed; "" = no default
    choices: tuple = ()
    help: str = ""
    metavar: str = ""

    @property
    def tip(self) -> str:
        t = self.help or ""
        if self.choices:
            t += f"\nchoices: {' '.join(map(str, self.choices))}"
        if self.kind == "value":
            t += f"\ndefault: {self.default or '(none)'}"
        return f"{' / '.join(self.flags)}\n{t}".strip()


def _opt_text(v) -> str:
    if v is None:
        return ""
    if isinstance(v, (list, tuple)):
        return " ".join(_opt_text(x) for x in v)
    return str(v)


def option_specs(parser: argparse.ArgumentParser,
                 skip: tuple[str, ...] = ("--help", "--set", "--gui")) -> list[Opt]:
    """Every option of ``parser`` except ``skip``, in the parser's order."""
    out = []
    for a in parser._actions:                      # noqa: SLF001 -- the public way is --help text
        if not a.option_strings or set(a.option_strings) & set(skip):
            continue
        if isinstance(a, argparse.BooleanOptionalAction):
            kind = "bool"
        elif isinstance(a, argparse._StoreTrueAction):  # noqa: SLF001
            kind = "flag"
        else:
            kind = "value"
        out.append(Opt(dest=a.dest, flags=tuple(a.option_strings), kind=kind,
                       nargs=a.nargs,
                       default=("true" if a.default else "false") if kind != "value"
                       else _opt_text(a.default),
                       choices=tuple(a.choices or ()),
                       help=(a.help or "").replace("%%", "%"),
                       metavar=a.metavar if isinstance(a.metavar, str) else ""))
    return out


def options_argv(specs: list[Opt], values: dict[str, str]) -> list[str]:
    """The options the operator changed, as argv. Unchanged ones are left to the tool."""
    out: list[str] = []
    for o in specs:
        v = values.get(o.dest, o.default).strip()
        if v == o.default:
            continue
        if o.kind == "flag":
            out += [o.flags[0]] if v == "true" else []
        elif o.kind == "bool":
            out += [o.flags[0] if v == "true" else o.flags[1]]
        elif v:
            multi = o.nargs in ("+", "*") or (isinstance(o.nargs, int) and o.nargs > 1)
            out += [o.flags[0], *(v.split() if multi else [v])]
    return out


def build_argv(values: dict[str, str], defaults: dict[str, str], *,
               opts: list[str] | tuple[str, ...] = (), extra: str = "") -> list[str]:
    """The measure_iqe command the window would run. Exactly what a person could type.

    ``opts`` are the command-line options (``options_argv`` of the Command tab). Only
    config fields the operator CHANGED become ``--set``. Sending all of them every time
    would bury the two that matter and make every run log unreadable -- and it would
    also defeat ``--config`` and ``configs/local.yaml``, because an explicit ``--set``
    outranks both, so a window that always sent everything would silently override the
    machine's own settings with this window's idea of the defaults.

    Separated from the widgets so it can be tested without a display.
    """
    a = [sys.executable, "-m", "measure_iqe", *opts]
    for k in values:
        v = values[k].strip()
        if v != defaults[k]:
            a += ["--set", f"{k}={v}"]
    for line in extra.splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            a += ["--set", line]
    return a


def capacitance_parser() -> argparse.ArgumentParser | None:
    """characterize_capacitance's parser, or None when it cannot be imported here."""
    if str(REPO) not in sys.path:
        sys.path.insert(0, str(REPO))
    try:
        import characterize_capacitance as cc
    except Exception:                                  # noqa: BLE001 - shown in the tab
        return None
    return cc.build_parser()


# ----------------------------------------------------------------------
#  The runner: one subprocess at a time, its output on a queue
# ----------------------------------------------------------------------
class Runner:
    """Runs argument lists in sequence, streaming stdout onto a queue.

    Owns no tkinter: the window polls ``out`` on a timer. That keeps every widget call
    on the main thread, which is the one rule tkinter actually enforces.
    """

    def __init__(self) -> None:
        self.out: "queue.Queue[tuple[str, object]]" = queue.Queue()
        self.proc: subprocess.Popen | None = None
        self._stop = False
        self._thread: threading.Thread | None = None

    @property
    def busy(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def start(self, jobs: list[list[str]]) -> None:
        if self.busy:
            raise RuntimeError("a run is already in progress")
        self._stop = False
        self._thread = threading.Thread(target=self._run_all, args=(jobs,), daemon=True)
        self._thread.start()

    def stop(self) -> None:
        """Ask the child to park the hardware, then give it time to do it.

        CTRL_BREAK rather than kill: measure_iqe.safeexit turns it into an orderly
        shutdown -- outputs off, frame blanked, files flushed. A killed process leaves
        the SMU driving the array, which is how an array gets cooked overnight.
        """
        self._stop = True
        p = self.proc
        if p is None or p.poll() is not None:
            return
        try:
            if os.name == "nt":
                p.send_signal(signal.CTRL_BREAK_EVENT)
            else:
                p.send_signal(signal.SIGINT)
        except Exception:
            p.terminate()

    def _run_all(self, jobs: list[list[str]]) -> None:
        for i, argv in enumerate(jobs, 1):
            if self._stop:
                self.out.put(("line", f"--- stopped before job {i}/{len(jobs)}"))
                break
            self.out.put(("job", (i, len(jobs), argv)))
            rc = self._run_one(argv)
            self.out.put(("line", f"--- job {i}/{len(jobs)} exit code {rc}"))
            if rc != 0 and not self._stop:
                self.out.put(("line", "--- nonzero exit: the queue stops here, so a "
                                      "misconfigured job does not burn the night"))
                break
        self.out.put(("done", None))

    def _run_one(self, argv: list[str]) -> int:
        kw: dict = dict(stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                        text=True, bufsize=1, cwd=str(REPO))
        if os.name == "nt":
            # Its own process group, or CTRL_BREAK would hit this window too.
            kw["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
        else:
            kw["start_new_session"] = True
        try:
            self.proc = subprocess.Popen(argv, **kw)
        except OSError as exc:
            self.out.put(("line", f"--- could not start: {exc}"))
            return 127
        assert self.proc.stdout is not None
        for line in self.proc.stdout:
            self.out.put(("line", line.rstrip("\n")))
        self.proc.wait()
        rc = self.proc.returncode
        self.proc = None
        return rc


# ----------------------------------------------------------------------
#  Widgets
# ----------------------------------------------------------------------
class Tip:
    """A tooltip: the text appears below the widget while the pointer is on it."""

    def __init__(self, widget: tk.Widget, text: str) -> None:
        self.w, self.text, self.top = widget, text[:1500], None
        widget.bind("<Enter>", self.show, add="+")
        widget.bind("<Leave>", self.hide, add="+")

    def show(self, _e=None) -> None:
        if self.top or not self.text:
            return
        x = self.w.winfo_rootx() + 12
        y = self.w.winfo_rooty() + self.w.winfo_height() + 4
        self.top = tk.Toplevel(self.w)
        self.top.wm_overrideredirect(True)
        self.top.wm_geometry(f"+{x}+{y}")
        tk.Label(self.top, text=self.text, justify="left", background="#ffffe0",
                 relief="solid", borderwidth=1, wraplength=560, padx=4, pady=2).pack()

    def hide(self, _e=None) -> None:
        if self.top:
            self.top.destroy()
            self.top = None


def _scrolled(master) -> tuple[ttk.Frame, ttk.Frame]:
    """(outer, inner): a vertically scrolling frame. The wheel scrolls whichever is
    under the pointer, so several of them can coexist."""
    outer = ttk.Frame(master)
    canvas = tk.Canvas(outer, highlightthickness=0)
    sb = ttk.Scrollbar(outer, orient="vertical", command=canvas.yview)
    inner = ttk.Frame(canvas)
    inner.bind("<Configure>", lambda _e: canvas.configure(scrollregion=canvas.bbox("all")))
    canvas.create_window((0, 0), window=inner, anchor="nw")
    canvas.configure(yscrollcommand=sb.set)
    canvas.pack(side="left", fill="both", expand=True)
    sb.pack(side="right", fill="y")
    wheel = lambda e: canvas.yview_scroll(int(-e.delta / 120), "units")  # noqa: E731
    canvas.bind("<Enter>", lambda _e: canvas.bind_all("<MouseWheel>", wheel))
    canvas.bind("<Leave>", lambda _e: canvas.unbind_all("<MouseWheel>"))
    return outer, inner


class OptionForm(ttk.Frame):
    """One row per command-line option: the flag, a widget, and its help as a tooltip."""

    PATHS = {"PATH": "file", "FILE": "file", "DIR": "dir", "RUN_DIR": "dir"}

    def __init__(self, master, specs: list[Opt]) -> None:
        super().__init__(master)
        self.specs = specs
        self.vars: dict[str, tk.StringVar] = {}
        outer, inner = _scrolled(self)
        outer.pack(fill="both", expand=True)
        for r, o in enumerate(specs):
            var = self.vars[o.dest] = tk.StringVar(value=o.default)
            lab = ttk.Label(inner, text=" / ".join(o.flags), anchor="e")
            lab.grid(row=r, column=0, sticky="e", padx=(4, 6), pady=1)
            if o.kind in ("flag", "bool"):
                w = ttk.Checkbutton(inner, variable=var, onvalue="true", offvalue="false")
            elif o.choices and o.nargs in (None, "?"):
                w = ttk.Combobox(inner, textvariable=var, width=30, state="readonly",
                                 values=[str(c) for c in o.choices])
            else:
                w = ttk.Entry(inner, textvariable=var, width=34)
            w.grid(row=r, column=1, sticky="w", pady=1)
            pick = self.PATHS.get(o.metavar)
            if pick:
                ttk.Button(inner, text="...", width=3,
                           command=lambda v=var, k=pick: self._pick(v, k)
                           ).grid(row=r, column=2, sticky="w")
            short = o.help.split(". ")[0][:90]
            hint = ttk.Label(inner, text=short, foreground="#666")
            hint.grid(row=r, column=3, sticky="w", padx=(6, 4))
            for widget in (lab, w, hint):
                Tip(widget, o.tip)

    @staticmethod
    def _pick(var: tk.StringVar, kind: str) -> None:
        p = filedialog.askdirectory() if kind == "dir" else filedialog.askopenfilename()
        if p:
            var.set(p)

    def values(self) -> dict[str, str]:
        return {k: v.get() for k, v in self.vars.items()}

    def argv(self) -> list[str]:
        return options_argv(self.specs, self.values())

    def reset(self) -> None:
        for o in self.specs:
            self.vars[o.dest].set(o.default)


# ----------------------------------------------------------------------
#  The window
# ----------------------------------------------------------------------
class App(ttk.Frame):
    def __init__(self, master: tk.Tk) -> None:
        super().__init__(master, padding=8)
        master.title("measure_iqe - operator console")
        master.geometry("1280x880")
        self.pack(fill="both", expand=True)

        from .cli import build_parser
        self.defaults = _defaults()
        self.docs = field_docs()
        self.vars: dict[str, tk.StringVar] = {
            k: tk.StringVar(value=v) for k, v in self.defaults.items()}
        self.iqe_specs = option_specs(build_parser())
        self.cc_parser = capacitance_parser()
        self.jobs: list[list[str]] = []
        self.runner = Runner()

        self._build()
        self.after(100, self._pump)

    # ---------------------------------------------------------------- layout
    def _build(self) -> None:
        nb = ttk.Notebook(self)
        nb.pack(fill="both", expand=True)
        self.iqe_form = OptionForm(nb, self.iqe_specs)
        nb.add(self.iqe_form, text="measure_iqe: command")
        nb.add(self._tab_shape(nb), text="measure_iqe: run shape")
        nb.add(self._tab_all(nb), text="measure_iqe: all parameters")
        nb.add(self._tab_cc(nb), text="characterize_capacitance")
        nb.add(self._tab_queue(nb), text="Queue")

        bar2 = ttk.Frame(self)
        bar2.pack(fill="x", pady=4)
        ttk.Label(bar2, text="measure_iqe:").pack(side="left")
        for text, cmd, tip in (
                ("Preview command", self._preview, "show the command, run nothing"),
                ("Check config", self._check,
                 "build the effective config with --print-config: no hardware touched"),
                ("Add to queue", self._enqueue, "append this command to the Queue tab"),
                ("Run now", self._run_now, "run this command alone, now")):
            b = ttk.Button(bar2, text=text, command=cmd)
            b.pack(side="left", padx=2)
            Tip(b, tip)
        ttk.Separator(bar2, orient="vertical").pack(side="left", fill="y", padx=8)
        self.btn_start = ttk.Button(bar2, text="Start queue", command=self._start)
        self.btn_start.pack(side="left")
        Tip(self.btn_start, "run every queued job in order; a nonzero exit stops the queue")
        self.btn_stop = ttk.Button(bar2, text="Stop (park hardware)",
                                   command=self._stop, state="disabled")
        self.btn_stop.pack(side="left", padx=4)
        Tip(self.btn_stop, "CTRL_BREAK to the running job: it parks the hardware "
                           "(outputs off, frame blanked) and writes what it has")

        self.progress = ttk.Progressbar(self, mode="determinate", maximum=100)
        self.progress.pack(fill="x", pady=(4, 2))
        self.status = ttk.Label(self, text="idle", anchor="w")
        self.status.pack(fill="x")

        self.log = tk.Text(self, height=16, wrap="none", background="#101014",
                           foreground="#d8d8d8", insertbackground="#d8d8d8")
        self.log.pack(fill="both", expand=True, pady=(4, 0))
        sb = ttk.Scrollbar(self, orient="horizontal", command=self.log.xview)
        sb.pack(fill="x")
        self.log.configure(xscrollcommand=sb.set)

    def _field_tip(self, key: str) -> str:
        return f"{key}\n{self.docs.get(key, '')}\ndefault: {self.defaults[key]}"

    def _tab_shape(self, master) -> ttk.Frame:
        f = ttk.Frame(master, padding=8)
        p = ttk.LabelFrame(f, text="Presets", padding=6)
        p.pack(fill="x")
        for name in PRESETS:
            b = ttk.Button(p, text=name, command=lambda n=name: self._preset(n))
            b.pack(side="left", padx=3)
            Tip(b, "reset every field, then set: " + ", ".join(
                f"{k}={_as_text(v)}" for k, v in PRESETS[name].items()))
        ttk.Button(p, text="Reset to defaults", command=self._reset).pack(side="right")

        g = ttk.Frame(f)
        g.pack(fill="both", expand=True, pady=(8, 0))
        for i, (key, label, hint) in enumerate(SHAPE):
            r, c = i % 11, (i // 11) * 3
            lab = ttk.Label(g, text=label)
            lab.grid(row=r, column=c, sticky="e", padx=(8, 4), pady=2)
            ent = ttk.Entry(g, textvariable=self.vars[key], width=20)
            ent.grid(row=r, column=c + 1, sticky="w", pady=2)
            h = ttk.Label(g, text=hint, foreground="#666")
            h.grid(row=r, column=c + 2, sticky="w", padx=(6, 12))
            for w in (lab, ent, h):
                Tip(w, self._field_tip(key))
        n = ttk.LabelFrame(f, text="Extra --set overrides (one per line, KEY=VALUE)",
                           padding=6)
        n.pack(fill="x", pady=(10, 0))
        self.extra_box = tk.Text(n, height=4, wrap="none")
        self.extra_box.pack(fill="x")
        Tip(self.extra_box, "anything the tabs do not show, or a value to force even when "
                            "it equals the default. Lines starting with # are ignored")
        return f

    def _tab_all(self, master) -> ttk.Frame:
        f = ttk.Frame(master, padding=8)
        top = ttk.Frame(f)
        top.pack(fill="x")
        ttk.Label(top, text="Filter:").pack(side="left")
        self.filter = tk.StringVar()
        e = ttk.Entry(top, textvariable=self.filter, width=30)
        e.pack(side="left", padx=4)
        e.bind("<KeyRelease>", lambda _e: self._refilter())
        ttk.Label(top, text="every IQEConfig field, generated from the dataclass so it "
                            "cannot drift. Hover a field for its explanation.",
                  foreground="#666").pack(side="left", padx=8)

        outer, self.allframe = _scrolled(f)
        outer.pack(fill="both", expand=True, pady=(6, 0))
        self.rows: list[tuple[str, ttk.Label, ttk.Entry, ttk.Label]] = []
        for fld in dataclasses.fields(IQEConfig):
            lab = ttk.Label(self.allframe, text=fld.name, width=30, anchor="e")
            ent = ttk.Entry(self.allframe, textvariable=self.vars[fld.name], width=26)
            doc = self.docs.get(fld.name, "")
            typ = ttk.Label(self.allframe, text=f"{fld.type}   {doc.split('. ')[0][:80]}",
                            foreground="#666")
            for w in (lab, ent, typ):
                Tip(w, self._field_tip(fld.name))
            self.rows.append((fld.name, lab, ent, typ))
        self._refilter()
        return f

    def _tab_cc(self, master) -> ttk.Frame:
        f = ttk.Frame(master, padding=8)
        if self.cc_parser is None:
            ttk.Label(f, text="characterize_capacitance.py could not be imported from "
                              f"{REPO}. Run the console from the repository.").pack()
            self.cc_form = None
            return f
        b = ttk.Frame(f)
        b.pack(fill="x", pady=(0, 6))
        ttk.Label(b, text="characterize_capacitance.py:").pack(side="left")
        for text, cmd in (("Preview command", self._cc_preview),
                          ("Add to queue", self._cc_enqueue), ("Run now", self._cc_run),
                          ("Reset", lambda: self.cc_form.reset())):
            ttk.Button(b, text=text, command=cmd).pack(side="left", padx=2)
        ttk.Label(b, text="--sample-name is required for a hardware run",
                  foreground="#a00").pack(side="left", padx=10)
        self.cc_form = OptionForm(f, option_specs(self.cc_parser))
        self.cc_form.pack(fill="both", expand=True)
        return f

    def _tab_queue(self, master) -> ttk.Frame:
        f = ttk.Frame(master, padding=8)
        self.qlist = tk.Listbox(f, height=14)
        self.qlist.pack(fill="both", expand=True)
        b = ttk.Frame(f)
        b.pack(fill="x", pady=6)
        ttk.Button(b, text="Remove selected", command=self._dequeue).pack(side="left")
        ttk.Button(b, text="Clear", command=self._clear_queue).pack(side="left", padx=4)
        ttk.Label(b, text="jobs run in order; a nonzero exit stops the queue",
                  foreground="#666").pack(side="left", padx=10)
        return f

    def _refilter(self) -> None:
        q = self.filter.get().strip().lower()
        r = 0
        for name, lab, ent, typ in self.rows:
            if q and q not in name.lower() and q not in self.docs.get(name, "").lower():
                lab.grid_remove(); ent.grid_remove(); typ.grid_remove()
                continue
            lab.grid(row=r, column=0, sticky="e", padx=4, pady=1)
            ent.grid(row=r, column=1, sticky="w", pady=1)
            typ.grid(row=r, column=2, sticky="w", padx=6)
            r += 1

    # ---------------------------------------------------------------- actions
    def _preset(self, name: str) -> None:
        self._reset()
        for k, v in PRESETS[name].items():
            self.vars[k].set(_as_text(v))
        self._say(f"preset: {name}")

    def _reset(self) -> None:
        for k, v in self.defaults.items():
            self.vars[k].set(v)

    def argv(self) -> list[str]:
        return build_argv({k: v.get() for k, v in self.vars.items()}, self.defaults,
                          opts=self.iqe_form.argv(),
                          extra=self.extra_box.get("1.0", "end"))

    def cc_argv(self) -> list[str]:
        return [sys.executable, "characterize_capacitance.py", *self.cc_form.argv()]

    def _show(self, argv: list[str]) -> None:
        self._say("$ " + " ".join(shlex.quote(x) for x in argv))

    def _preview(self) -> None:
        self._show(self.argv())

    def _cc_preview(self) -> None:
        self._show(self.cc_argv())

    def _check(self) -> None:
        """Validate without touching hardware: --print-config does the whole build."""
        argv = self.argv() + ["--print-config"]
        try:
            r = subprocess.run(argv, capture_output=True, text=True, timeout=120,
                               cwd=str(REPO))
        except Exception as exc:
            self._say(f"!! could not run the check: {exc}")
            return
        if r.returncode != 0:
            self._say("!! " + (r.stderr or r.stdout).strip())
            return
        try:
            d = json.loads(r.stdout[r.stdout.index("{"):])
        except Exception:
            self._say(r.stdout.strip()[:400])
            return
        pg = tuple(d.get("pool_groups") or ())
        self._say(f"config OK: roi={d.get('roi')} select={d.get('sweep_select')} "
                  f"block={d.get('block_size')} singles/block={d.get('pixels_per_block')} "
                  f"pool_groups={pg or '()'} n_fine={d.get('n_fine')} "
                  f"order={d.get('sweep_order')} dir={d.get('sweep_direction')}"
                  + (f" FAST SCAN ONLY at {d.get('nvled_v_fastscan')} "
                     f"{d.get('fastscan_voltages')}" if d.get("fastscan_only") else ""))
        if pg:
            self._say(f"   every block adds {len(pg)} group(s) of {list(pg)}, formed "
                      f"inside that block. Pooled rows carry POOL_K and the SUM of "
                      f"their members: EQE is unaffected, per-pixel current is not. "
                      f"The {d.get('pixels_per_block')}/block singles are measured too "
                      f"and are what grounds them"
                      + (", and each group shares one of them so the comparison is "
                         "paired." if d.get("pool_include_singles") else "."))

    def _add(self, argv: list[str]) -> None:
        self.jobs.append(argv)
        self.qlist.insert("end", " ".join(argv[1:]))
        self._say(f"queued job {len(self.jobs)}")

    def _enqueue(self) -> None:
        self._add(self.argv())

    def _cc_enqueue(self) -> None:
        self._add(self.cc_argv())

    def _dequeue(self) -> None:
        for i in reversed(self.qlist.curselection()):
            self.qlist.delete(i)
            del self.jobs[i]

    def _clear_queue(self) -> None:
        self.qlist.delete(0, "end")
        self.jobs.clear()

    def _run_now(self) -> None:
        self._launch([self.argv()])

    def _cc_run(self) -> None:
        self._launch([self.cc_argv()])

    def _start(self) -> None:
        if not self.jobs:
            messagebox.showinfo("Queue empty", "Add a job first, or use Run now.")
            return
        self._launch(list(self.jobs))

    def _launch(self, jobs: list[list[str]]) -> None:
        if self.runner.busy:
            messagebox.showwarning("Busy", "A run is already in progress.")
            return
        self.progress.configure(value=0)
        self.btn_start.configure(state="disabled")
        self.btn_stop.configure(state="normal")
        self.runner.start(jobs)

    def _stop(self) -> None:
        self._say("--- Stop: sending CTRL_BREAK so the child parks the hardware "
                  "(outputs off, frame blanked). Give it a few seconds.")
        self.runner.stop()

    # ---------------------------------------------------------------- plumbing
    def _say(self, text: str) -> None:
        self.log.insert("end", text + "\n")
        self.log.see("end")

    def _pump(self) -> None:
        """Drain the runner's queue onto the widgets. Main thread only."""
        try:
            while True:
                kind, payload = self.runner.out.get_nowait()
                if kind == "line":
                    self._say(str(payload))
                    m = _PROG.search(str(payload))
                    if m:
                        done, total = int(m.group(1)), max(1, int(m.group(2)))
                        self.progress.configure(value=100.0 * done / total)
                        self.status.configure(text=f"{done}/{total}")
                elif kind == "job":
                    i, n, argv = payload
                    self.status.configure(text=f"job {i}/{n}")
                    self._say(f"=== job {i}/{n}: " +
                              " ".join(shlex.quote(x) for x in argv))
                elif kind == "done":
                    self.btn_start.configure(state="normal")
                    self.btn_stop.configure(state="disabled")
                    self.status.configure(text="idle")
                    self._say("=== all jobs finished")
        except queue.Empty:
            pass
        except Exception as exc:                      # noqa: BLE001
            # One bad line must not stop the log for the rest of the session: the pump
            # is the only thing draining the runner's queue, and a dead pump looks
            # exactly like a hung measurement.
            self.log.insert("end", f"!! console error while logging: {exc!r}\n")
        self.after(100, self._pump)


def main(argv: list[str] | None = None) -> int:
    root = tk.Tk()
    App(root)
    root.mainloop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
