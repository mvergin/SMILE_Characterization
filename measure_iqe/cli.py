"""Command-line front end for :func:`measure_iqe.run_iqe_measurement`.

    python -m measure_iqe --simulate
    python -m measure_iqe --config iqe_config.yaml --out D:/runs
    python -m measure_iqe --set n_fine=60 --set sweep_direction=low_to_high

Parsing and printing live here; the measurement itself lives in
:mod:`measure_iqe.acquire` and never touches ``argv``.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import sys
from pathlib import Path
from typing import Any, Sequence

from .acquire import make_logger, run_calibration, run_iqe_measurement
from .calibrate import calibration_path, load_calibration
from .config import IQEConfig


def parse_override(text: str) -> tuple[str, Any]:
    """``"n_fine=60"`` -> ``("n_fine", 60)``.

    The value is read as JSON so numbers, booleans and lists keep their type;
    anything that is not valid JSON stays a plain string (so ``mode=voltage``
    works without quoting).
    """
    key, sep, raw = text.partition("=")
    if not sep:
        raise argparse.ArgumentTypeError(f"--set expects KEY=VALUE, got {text!r}")
    key = key.strip()
    # A value wrapped in matching quotes is shell leakage, not data. PowerShell and bash
    # strip `roi='[0,0,63,63]'`; cmd.exe passes the quotes through, so the same command
    # fails on one machine with "unknown ROI" and works on another. Two shapes, and the
    # second is the one that hides: a DOUBLE-quoted value is valid JSON for the string
    # inside it, so `roi="[0,0,63,63]"` parses happily into the string "[0,0,63,63]"
    # and never looks wrong until the config rejects it.
    quoted = len(raw) >= 2 and raw[0] == raw[-1] and raw[0] in "\"'"
    try:
        val = json.loads(raw)
    except json.JSONDecodeError:
        if quoted:                       # 'FULL' or '[0,0,63,63]' from cmd.exe
            try:
                return key, json.loads(raw[1:-1])
            except json.JSONDecodeError:
                return key, raw[1:-1]
        return key, raw                  # mode=voltage: a bare word stays a string
    if quoted and isinstance(val, str):  # "[0,0,63,63]" parsed as a string
        try:
            return key, json.loads(val)
        except json.JSONDecodeError:
            pass                         # "Array42" really is the string Array42
    return key, val


LOCAL_CONFIG = "configs/local.yaml"


def local_overrides(log=None) -> dict[str, Any]:
    """``configs/local.yaml`` if it exists, as a partial override dict.

    This file is for what is true of THIS MACHINE and nothing else -- ``base_dir``
    above all. It is gitignored on purpose: ROADMAP item 3 rejected auto-loading the
    tracked ``configs/standard.yaml`` because a run's behaviour would then depend on an
    uncommitted edit to a file git calls clean, which is the provenance problem item 0
    was about.

    Every key it applies is logged, because an override nobody can see is the same
    problem wearing a different hat.

    ponytail: no schema beyond IQEConfig's own coercion, and no check that the file is
    actually gitignored. If someone commits machine paths, add `git check-ignore` here.
    """
    path = Path(__file__).resolve().parent.parent / LOCAL_CONFIG
    if not path.exists():
        return {}
    import yaml                                   # only needed when the file is there
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(data, dict):
        raise ValueError(f"{path}: expected a mapping at the top level")
    if log and data:
        log(f"applied {LOCAL_CONFIG}: {', '.join(sorted(data))}")
        for k, v in sorted(data.items()):
            log(f"  {k} = {v!r}")
    return data


def build_config(args: argparse.Namespace, log=None) -> IQEConfig:
    """Config file (or defaults) with ``--set`` overrides applied on top."""
    base = IQEConfig.from_file(args.config) if args.config else IQEConfig()
    data = base.to_dict()
    # Precedence, lowest first: defaults, --config, configs/local.yaml, --calibration,
    # --set, --out.
    # local.yaml sits ABOVE --config on purpose: it describes the machine, and a config
    # file shared between machines must not put this one's runs somewhere that does not
    # exist here. Anything per-invocation (--set, --out) still wins over both.
    data.update(local_overrides(log))
    # The calibration is measured fact about this array; an explicit --set still wins,
    # so a deliberate override is never silently replaced by the calibration.
    if getattr(args, "array", None):
        data["sample_name"] = args.array
    cal = getattr(args, "calibration", None)
    if cal is None and getattr(args, "array", None) and not args.calibrate:
        # The whole point of --array: one flag, and this array's measured settings
        # follow. Absent is fine and silent-but-reported, not an error -- a run before
        # the array is characterised is legitimate, it just uses the defaults.
        auto = calibration_path(args.array)
        cal = str(auto) if auto.exists() else None
    if cal:
        data.update(load_calibration(cal))
        # Kept in the config, not popped: the saved *_config.json then names the file
        # that produced the run, and acquire.stamp_run copies it into the run directory.
        # Set for an explicit --calibration too, not only the --array lookup.
        data["calibration_source"] = str(cal)
    overrides = dict(args.set or [])
    data.update(overrides)
    if args.simulate and "settle_scale" not in overrides:
        # A simulated run should take seconds, not the real multi-second dwells -- but
        # only when the user has not asked for them. This is the one place that can tell
        # "the default 1.0" from "the user typed 1.0".
        data["settle_scale"] = 0.0
    if args.out:
        data["base_dir"] = args.out
    return IQEConfig.from_dict(data)


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        prog="python -m measure_iqe",
        description="IQE-focused acquisition for the SMILE micro-LED array.",
        epilog="Every IQEConfig field can be overridden with --set; see --list-params.")
    ap.add_argument("--config", metavar="PATH",
                    help="JSON or YAML config file (see configs/standard.yaml)")
    ap.add_argument("--set", metavar="KEY=VALUE", type=parse_override, action="append",
                    help="override one config field; repeatable")
    ap.add_argument("--out", metavar="DIR",
                    help="output root, overriding base_dir from the config")
    ap.add_argument("--array", metavar="NAME",
                    help="which physical array this is. Sets sample_name, and picks up "
                         "calibrations/NAME.yaml if --calibrate has been run for it")
    ap.add_argument("--calibrate", action="store_true",
                    help="measure THIS array's settling and write a calibration file, "
                         "instead of sweeping. Run once per array (~6 min)")
    ap.add_argument("--calibration", metavar="FILE",
                    help="apply a calibration file written by --calibrate")
    ap.add_argument("--resume", metavar="RUN_DIR",
                    help="continue an interrupted run: skip the pixels this directory "
                         "already finished and append to its files")
    ap.add_argument("--simulate", action="store_true",
                    help="run against the built-in ABC-model rig, no hardware")
    ap.add_argument("--warmup-diag", action="store_true",
                    help="run the warm-up/conditioning diagnostic instead of the sweep")
    ap.add_argument("--dump-config", metavar="PATH",
                    help="write the effective config to PATH (.json/.yaml) and exit")
    ap.add_argument("--print-config", action="store_true",
                    help="print the effective config as JSON and exit")
    ap.add_argument("--list-params", action="store_true",
                    help="list every config field with its default and exit")
    ap.add_argument("--gui", action="store_true",
                    help="open the tkinter operator console instead of running")
    return ap


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point.  Returns the process exit code."""
    args = build_parser().parse_args(argv)

    if args.gui:
        # Imported here, not at module scope: a headless measurement machine has no
        # display and importing tkinter there would break every ordinary CLI run.
        from .gui import main as gui_main
        return gui_main()

    if args.list_params:
        for f in dataclasses.fields(IQEConfig):
            print(f"{f.name:<28} {str(f.type):<28} default={f.default!r}")
        return 0

    # The logger is built BEFORE the config so configs/local.yaml can announce itself
    # at the top of the run rather than after the instruments are already connected.
    log = make_logger()
    try:
        cfg = build_config(args, log)
    except (ValueError, OSError) as exc:
        print(f"config error: {exc}", file=sys.stderr)
        return 2

    if args.print_config:
        print(json.dumps(cfg.to_dict(), indent=2))
        return 0
    if args.dump_config:
        print(f"wrote {cfg.write_file(args.dump_config)}")
        return 0

    if args.array and not args.calibrate and not args.calibration:
        auto = calibration_path(args.array)
        if auto.exists():
            # Say WHICH calibration and WHEN it was measured. Several --calibrate runs
            # leave several record folders, but only one file is ever consulted: this
            # one, overwritten by the most recent run. Printing the timestamp and the
            # values makes it impossible to analyse a sweep against the wrong one.
            try:
                import yaml
                d = yaml.safe_load(auto.read_text(encoding="utf-8")) or {}
            except Exception:
                d = {}
            log(f"array {args.array}: using {auto}")
            log(f"  measured {d.get('measured_utc', '?')} on pixels {d.get('pixels', '?')}, "
                f"target {d.get('target', '?')}, window {d.get('seconds_per_capture', '?')}s"
                + (" (LOWER BOUND)" if d.get("lower_bound") else ""))
            log(f"  applying {d.get('recommended', {})}")
        else:
            from .calibrate import CALIB_DIR, calibration_roots
            looked = ", ".join(str(d / CALIB_DIR) for d in calibration_roots())
            log(f"array {args.array}: NO CALIBRATION found. Looked in: {looked}")
            log(f"  the sweep will use the DEFAULT settle times, which were measured on "
                f"a different array. Run --calibrate --array {args.array} to fix that.")
    if args.calibrate:
        try:
            run_calibration(cfg, simulate=args.simulate, log=log)
        except KeyboardInterrupt:
            log("interrupted - hardware parked")
            return 130
        return 0
    try:
        result = run_iqe_measurement(cfg, simulate=args.simulate,
                                     warmup_diag=args.warmup_diag, log=log,
                                     run_dir=args.resume)
    except KeyboardInterrupt:
        log("interrupted — partial results are on disk")
        return 130
    log(f"run directory: {result.run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
