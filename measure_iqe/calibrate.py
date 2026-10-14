"""Measure one array's settling behaviour and save it as a reusable calibration.

    python -m measure_iqe --calibrate --out D:\\cal        # once per array, ~6 min
    python -m measure_iqe --calibration D:\\cal\\<...>_calibration.yaml

Why this exists: **the settle time is a property of the array, not of the method.**
A dwell measured on one array does not transfer to another. An array with 7 uF may well
need far less -- or not, because the settling is a charge relaxation and not an RC
charge, so it does not have to scale with C. Guessing either way is wrong. Measure it.

What it does, per pixel, at three biases spanning the sweep:

  1. step the bias and poll the current for `--calibrate-seconds`
  2. find when the reading stops changing, by the criterion in `settle_time()`
  3. take the worst case over the pixels, add margin, and write it out

The output is a small YAML that `--calibration` merges into the config, so a
characterised array needs one flag instead of ten hand-typed overrides.

Polling, not the SMU buffer: the thing being measured takes seconds, and a ~20 Hz
poll resolves that with room to spare. It needs no instrument code the sweep does not
already have.
"""
from __future__ import annotations

import datetime
import json
import time
from pathlib import Path
from typing import Any, Callable

import numpy as np

from .config import SMU_I_RANGES, IQEConfig

Logger = Callable[[str], None]

CALIBRATION_VERSION = 2

# Beyond this fraction of the record, a settle time is reporting the window, not the
# device (see settle_time). Such a capture is treated exactly like one that never
# settled: it raises the worst case and marks the result a lower bound.
RECORD_LIMITED_FRAC = 0.5

# Longest smoothing window, in seconds. Caps the time resolution of a settle time so it
# does not degrade with record length (see settle_time).
SMOOTH_MAX_S = 2.0


# ---------------------------------------------------------------------------
# The settling criterion. One implementation, used by the calibration, quoted by
# INVESTIGATION_REPORT.md, and mirrored in the analysis repo's plot_transients.py.
# ---------------------------------------------------------------------------
def settle_time(t, i, frac: float = 0.01) -> float:
    """When did the current stop changing, to within `frac` of its final value?

    Returns the **last** time the (smoothed) trace was outside the band, because a
    power-law tail re-enters and leaves it several times -- "first time inside" would
    report the first lucky sample. Returns NaN if it never settles, never 0.0: the
    original version returned 0.0 in exactly that case, so a trace that never settled
    scored as perfectly settled.

    The band is `frac` of the final value, or 5 sigma of the averaged noise if that is
    wider, so at low current the answer is not set by noise.

    It also checks that the tail has stopped DRIFTING. Without that, a slow power law
    is scored as settled: "final" is taken from the end of the record, and a trace that
    is still falling is always close to its own last value. The drift check compares
    the last tenth of the record against the 60-80% window and fails if they differ by
    more than `frac`.
    """
    t = np.asarray(t, float)
    i = np.asarray(i, float)
    ok = np.isfinite(t) & np.isfinite(i)
    t, i = t[ok], i[ok]
    if t.size < 20:
        return float("nan")
    final = float(np.median(i[int(0.8 * i.size):]))
    if final == 0:
        return float("nan")
    # Smooth over a fixed TIME, not a fixed fraction of the record. n//20 looks
    # harmless until the record is long: a 600 s capture gave a 30 s window, so the
    # first smoothed point sat 15 s in and "settled at t = 0" actually meant "settled
    # by 15 s, we cannot see sooner". The droop side was recommended a 0.1 s dwell on
    # that basis. The 5 sigma / sqrt(w) term in the tolerance already adapts to the
    # smaller window, so capping costs noise rejection only where it was free.
    dt = float(np.median(np.diff(t))) or 1e-3
    w = int(np.clip(i.size // 20, 5, max(5, round(SMOOTH_MAX_S / dt))))
    smooth = np.convolve(i, np.ones(w) / w, mode="valid")
    ts = t[w - 1:]
    noise = float(np.std(np.diff(i[int(0.8 * i.size):]))) / np.sqrt(2)
    tol = max(frac * abs(final), 5.0 * noise / np.sqrt(w))
    early = float(np.median(i[int(0.6 * i.size):int(0.8 * i.size)]))
    if abs(early - final) > max(tol, abs(frac * final)):
        return float("nan")              # the tail is still moving: not settled at all
    bad = np.nonzero(np.abs(smooth - final) > tol)[0]
    if bad.size == 0:
        # In band at the first point we can resolve -- which is half a smoothing window
        # in, not zero. Report that bound rather than a zero the data cannot support.
        return float(ts[0]) / 2.0
    if bad[-1] + 1 >= smooth.size:
        return float("nan")              # still outside the band at the end
    answer = float(ts[bad[-1]])
    # RECORD-LIMITED. "final" is the median of the last 20% of the record, so a trace
    # that is still decaying is always close to its own last value and the measured
    # settle time scales with however long you happened to watch. Measured on one array:
    #   21 s record -> t_1pct 13.2 s (62% of it)
    #   81 s record -> t_1pct 54.2 s (67%)
    #  120 s record -> t_1pct 74.0 s (62%)
    # all of the same device. An answer past half the record is not a measurement of the
    # device, it is a measurement of the window, so say so rather than return it.
    if answer > RECORD_LIMITED_FRAC * float(t[-1]):
        return float("nan")
    return answer


def settle_curve(pm, smu, cfg: IQEConfig, v_to: float, seconds: float,
                 period: float = 0.05, n_pm: int = 0):
    """Step ch B to `v_to` and poll for `seconds`. Returns (t, |I|, P).

    `n_pm > 0` also records the optical power, which costs a little time per poll and
    makes the trace able to answer whether EQE = P/I settles faster than either factor.
    """
    from .acquire import acquire_point
    from .safeexit import parked

    smu.set_voltage("b", float(v_to))
    t0 = time.perf_counter()
    ts: list[float] = []
    iv: list[float] = []
    pv: list[float] = []
    while (dt := time.perf_counter() - t0) < seconds:
        if parked():                 # Ctrl-C or a shutdown: the session is gone
            break
        p, _, _, _, inv, _, _, _ = acquire_point(pm, smu, cfg.calibrate_n_smu, n_pm)
        ts.append(dt)
        iv.append(abs(inv))
        pv.append(p)
        rest = period - (time.perf_counter() - t0 - dt)
        if rest > 0:
            time.sleep(rest)
    return np.asarray(ts), np.asarray(iv), np.asarray(pv)


def _ladder(lo: float, hi: float) -> list[float]:
    return [r for r in SMU_I_RANGES if lo <= r <= hi] or [hi]


def calibrate(pm, smu, smile, pixels: list[tuple[int, int]], cfg: IQEConfig,
              log: Logger, out_dir: str | Path | None = None) -> dict[str, Any]:
    """Measure the settling behaviour of this array. Returns the calibration dict.

    When `out_dir` is given, the raw I(t) of every capture is written beside it as
    `calibration_traces.npz`.
    """
    from .acquire import configure_voltage_source_b, pick_limit, set_nplc

    # Three biases spanning the sweep: the start, the density knee, the faint end.
    # The faint end is the one that sets the budget, so it is measured on every pixel.
    knee = cfg.vramp_knee_v if cfg.vramp_knee_v is not None else (
        (cfg.nvled_v_coarse_start + cfg.nvled_v_coarse_stop) / 2)
    biases = [cfg.nvled_v_coarse_stop, float(knee), cfg.nvled_v_coarse_start]
    n_low = max(2, round(cfg.n_fine * cfg.vramp_low_frac))
    step = abs(cfg.nvled_v_coarse_start - knee) / n_low
    droop_step = abs(knee - cfg.nvled_v_coarse_stop) / max(1, cfg.n_fine - n_low)
    log(f"CALIBRATION — {len(pixels)} pixel(s), biases {biases}, {cfg.calibrate_seconds:g}s "
        f"each, target {cfg.calibrate_target*100:g}%; the sweep steps {step*1e3:.0f} mV on the "
        f"rising side and {droop_step*1e3:.0f} mV on the droop side, and each capture is "
        f"approached by {cfg.calibrate_approach_steps} such step(s)")

    rows: list[dict[str, Any]] = []
    traces: dict[str, np.ndarray] = {}
    for (x, y) in pixels:
        smile.blank_frame()
        smile.set_pixel(x, y, cfg.bitvals[0])
        prev_i = None
        for v in biases:
            # APPROACH THE BIAS THE WAY THE SWEEP DOES. The sweep arrives at -0.80 V by
            # eighteen 28 mV steps from a nearly-settled neighbour; an earlier version of
            # this jumped straight from -1.30 V, a 0.5 V step, and measured 85 s where the
            # sweep may need far less (run 261014-1156 -> a 45 h sweep). The pre-steps also
            # bring prev_i down to the ADJACENT point's current, so pick_limit lands on the
            # same range the sweep will use -- the jump version measured -0.80 V on the
            # 100 uA range while the sweep uses 10 uA, and under High-C that is a different
            # measurement, not a detail.
            seg_step = step if v > knee else droop_step
            # Never drive past the sweep's own range. -2.6 V is the FIRST point of the
            # sweep, so there is nothing before it, and stepping "before" it would mean
            # -2.93 V -- harder than the sweep ever drives, and straight into compliance.
            lo = min(cfg.nvled_v_coarse_start, cfg.nvled_v_coarse_stop)
            n_approach = cfg.calibrate_approach_steps
            while n_approach and v - n_approach * seg_step < lo:
                n_approach -= 1
            for k in range(n_approach, 0, -1):
                # MORE negative, i.e. from the higher current: the sweep runs
                # high_to_low, so the point before v is v - step, every time. Getting
                # this sign wrong approaches from the low-current side, which is the
                # opposite memory condition to the one being measured.
                vk = v - k * seg_step
                limit = pick_limit(prev_i, cfg)
                configure_voltage_source_b(
                    smu, __import__("dataclasses").replace(cfg, nvled_compliance=limit), log)
                set_nplc(smu, cfg.vramp_lowv_nplc if v > knee else cfg.vramp_highv_nplc)
                tk, ik, _ = settle_curve(pm, smu, cfg, vk, cfg.calibrate_approach_s,
                                         cfg.calibrate_period_s)
                if ik.size:
                    prev_i = float(np.median(ik[int(0.8 * ik.size):]))
            limit = pick_limit(prev_i, cfg)
            configure_voltage_source_b(
                smu, __import__("dataclasses").replace(cfg, nvled_compliance=limit), log)
            set_nplc(smu, cfg.vramp_lowv_nplc if v > knee else cfg.vramp_highv_nplc)
            t, i, p = settle_curve(pm, smu, cfg, v, cfg.calibrate_seconds,
                                   cfg.calibrate_period_s, cfg.trace_settling_n_pm)
            # KEEP the trace. These are the longest and cleanest transients the project
            # has: cal3 polled three 600 s captures and discarded all three, which are
            # exactly the records needed to test whether the settled value can be
            # extrapolated from the first seconds.
            traces[f"{x}_{y}_v{v:+.4f}"] = np.vstack(
                [t.astype(np.float32), i.astype(np.float32), p.astype(np.float32)])
            prev_i = float(np.median(i[int(0.8 * i.size):])) if i.size else None
            r = dict(pixel_x=x, pixel_y=y, v_to=v, limit_a=limit,
                     settled_a=prev_i, step_v=round(seg_step, 5),
                     approach_steps=n_approach,
                     t_target=settle_time(t, i, cfg.calibrate_target),
                     t_1pct=settle_time(t, i, 0.01),
                     t_01pct=settle_time(t, i, 0.001),
                     n_samples=int(i.size))
            rows.append(r)
            log(f"  ({x},{y}) {v:+.3f} V  I={(prev_i or 0)*1e9:9.1f} nA  limit {limit:g} A  "
                f"t_{cfg.calibrate_target*100:g}%={r['t_target']:.2f}s  "
                f"(1%={r['t_1pct']:.2f}s  0.1%={r['t_01pct']:.2f}s)")
    smile.blank_frame()

    def worst(key: str, keep) -> tuple[float, bool]:
        """(worst case over the segment, is it only a lower bound?).

        A capture that never settled is the SLOWEST one, so dropping it would bias the
        recommendation low by exactly the data that matters -- which is what the first
        version of this did. An unsettled capture is still outside the band at the end
        of the window, so it contributes at least `calibrate_seconds`, and the result
        is flagged as a lower bound.
        """
        seg = [r for r in rows if keep(r)]
        vals = [r[key] for r in seg if np.isfinite(r[key])]
        unsettled = len(seg) - len(vals)
        if unsettled:
            vals.append(cfg.calibrate_seconds)
        return (float(max(vals)) if vals else float("nan")), bool(unsettled)

    rising, rising_lb = worst("t_target", lambda r: r["v_to"] > knee)
    droop, droop_lb = worst("t_target", lambda r: r["v_to"] <= knee)
    never = [r for r in rows if not np.isfinite(r["t_target"])]
    currents = [r["settled_a"] for r in rows if r["settled_a"]]

    # Margin, because the calibration sees three biases and the sweep visits thirty.
    m = cfg.calibrate_margin
    # A floor of two poll periods: the calibration cannot resolve a settle shorter than
    # it samples, so a measured 0.0 means "faster than we can see", not "no wait needed".
    floor = 2 * cfg.calibrate_period_s
    rec: dict[str, Any] = {}
    if np.isfinite(rising):
        rec["vramp_lowv_settle_s"] = round(max(float(rising) * m, floor), 2)
    if np.isfinite(droop):
        rec["vramp_highv_settle_s"] = round(max(float(droop) * m, floor), 2)
    if currents:
        lad = _ladder(cfg.vramp_lowv_limit_a, cfg.vramp_highv_limit_a)
        need_hi = 1.5 * max(currents)
        rec["vramp_highv_limit_a"] = next((r for r in lad if r >= need_hi), lad[-1])

    out = dict(
        calibration_version=CALIBRATION_VERSION,
        sample_name=cfg.sample_name,
        measured_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
        pixels=[list(p) for p in pixels],
        seconds_per_capture=cfg.calibrate_seconds,
        target=cfg.calibrate_target,
        approach_steps=cfg.calibrate_approach_steps,
        margin=m,
        rising_step_v=round(step, 5),
        recommended=rec,
        never_settled=[{k: r[k] for k in ("pixel_x", "pixel_y", "v_to")} for r in never],
        lower_bound=False,          # set below once both segments are known
        measurements=rows,
    )
    out["lower_bound"] = bool(rising_lb or droop_lb)
    if never:
        log(f"  !! {len(never)} of {len(rows)} capture(s) did not settle within "
            f"{cfg.calibrate_seconds:g}s, so the worst case is at least that long and the "
            f"recommendation is a LOWER BOUND.")
        for r in never:
            log(f"       ({r['pixel_x']},{r['pixel_y']}) at {r['v_to']:+.3f} V")
        log(f"     (either still outside the band at the end, or settling past half the "
            f"record, which measures the window and not the device.)")
        log(f"     Re-run with --set calibrate_seconds={max(300, 3 * cfg.calibrate_seconds):g} "
            f"to get a real number. The sweep will work with this value, but the low-J "
            f"points carry more unrelaxed tail than the margin assumes.")
    if out_dir is not None and traces:
        npz = Path(out_dir) / "calibration_traces.npz"
        np.savez_compressed(npz, **traces)
        out["traces_file"] = npz.name
        log(f"  wrote {npz} ({len(traces)} transients, "
            f"{npz.stat().st_size/1e6:.1f} MB)")
    log(f"  recommended: " + ", ".join(f"{k}={v}" for k, v in rec.items()))
    return out


CALIB_DIR = "calibrations"      # gitignored, so a share re-export never clobbers it


def calibration_roots(root: str | Path | None = None) -> list[Path]:
    """Where a calibration may live, most specific first.

    The current directory, then the directory holding the measure_iqe package. Being
    cwd-relative alone is a trap: --calibrate run from one directory and the sweep run
    from another would silently not find each other, and the sweep would fall back to
    the defaults with only a line in the log to say so.
    """
    here = Path(root) if root is not None else Path.cwd()
    install = Path(__file__).resolve().parent.parent
    seen, out = set(), []
    for d in (here, install):
        d = d.resolve()
        if d not in seen:
            seen.add(d)
            out.append(d)
    return out


def calibration_path(sample_name: str, root: str | Path | None = None) -> Path:
    """Where `--array NAME` looks for a calibration.

    Returns the first existing file; if there is none, the preferred location to write.
    """
    roots = calibration_roots(root)
    for d in roots:
        cand = d / CALIB_DIR / f"{sample_name}.yaml"
        if cand.exists():
            return cand
    return roots[0] / CALIB_DIR / f"{sample_name}.yaml"


def _dump(path: Path, data: dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        import yaml
        path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    except ImportError:
        path = path.with_suffix(".json")
        path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    return path


def write_calibration(out_dir: str | Path, data: dict[str, Any],
                      root: str | Path = ".") -> tuple[Path, Path]:
    """Write the calibration twice: with the run, and in the per-array location.

    The copy in the run directory is the record of what was measured, timestamped
    alongside its own fast scan. The copy in ``calibrations/<sample>.yaml`` is the one
    ``--array`` picks up, so a characterised array needs one flag and no file paths.
    """
    name = str(data.get("sample_name") or "array")
    record = _dump(Path(out_dir) / f"{name}_calibration.yaml", data)
    # Write to every root, so a sweep started from either directory finds it. They are
    # small, and a stale copy in the other root would be worse than a duplicate.
    stable = None
    for d in calibration_roots(root if root != "." else None):
        stable = _dump(d / CALIB_DIR / f"{name}.yaml", data)
    return record, stable


def load_calibration(path: str | Path) -> dict[str, Any]:
    """Read a calibration file and return its `recommended` overrides.

    Raises on a file that is not a calibration, rather than silently applying nothing:
    a sweep that believes it is calibrated and is not is worse than one that stops.
    """
    path = Path(path)
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() in (".yaml", ".yml"):
        import yaml
        data = yaml.safe_load(text)
    else:
        data = json.loads(text)
    if not isinstance(data, dict) or "recommended" not in data:
        raise ValueError(f"{path} is not a calibration file (no 'recommended' key)")
    if data.get("calibration_version") != CALIBRATION_VERSION:
        raise ValueError(
            f"{path} is calibration version {data.get('calibration_version')}, "
            f"this build writes {CALIBRATION_VERSION}. Re-run --calibrate.")
    rec = data["recommended"]
    unknown = sorted(set(rec) - {f.name for f in __import__("dataclasses").fields(IQEConfig)})
    if unknown:
        raise ValueError(f"{path} recommends unknown config key(s): {', '.join(unknown)}")
    return dict(rec)


def demo() -> None:
    """Self-check for settle_time, the one piece with real logic."""
    t = np.linspace(0, 20, 2000)

    # settles at ~5 s: an exponential with tau = 1 s is inside 1% by t = 4.6 s
    i = 100e-9 * (1 - np.exp(-t / 1.0))
    ts = settle_time(t, i, 0.01)
    assert 3 < ts < 7, ts

    # never settles: a power law still 5% out at the end
    i = 100e-9 * (1 + 0.5 * (1 + t) ** -0.4)
    assert np.isnan(settle_time(t, i, 0.01)), settle_time(t, i, 0.01)

    # already settled: flat plus noise -> half a smoothing window, never NaN, and
    # never 0.0, because no measurement can resolve below what it smooths over
    i = 100e-9 + np.random.default_rng(0).normal(0, 1e-12, t.size)
    assert 0 < settle_time(t, i, 0.01) <= SMOOTH_MAX_S, settle_time(t, i, 0.01)

    # noise-aware: a 1% band on a tiny current would be set by noise alone, so a
    # trace that is flat-but-noisy must still count as settled
    i = 1e-9 + np.random.default_rng(1).normal(0, 5e-10, t.size)
    assert 0 < settle_time(i=i, t=t, frac=0.01) <= SMOOTH_MAX_S

    # the resolution must NOT degrade with record length: the same fast settle on a
    # 600 s record must still report ~seconds, not "0" meaning "sooner than 15 s"
    tl = np.linspace(0, 600, 1835)
    il = 100e-9 * (1 - np.exp(-tl / 0.5))
    assert settle_time(tl, il, 0.03) <= 4.0, settle_time(tl, il, 0.03)

    # last-time-outside: a trace that dips back out at 5 s reports ~5 s, not ~1 s.
    # (Kept well inside half the record, or the record-limited guard fires instead --
    # which it should: an answer at 10 s of a 20 s record is not a measurement.)
    i = 100e-9 * (1 - np.exp(-t / 0.2))
    i[(t > 4) & (t < 5)] *= 1.05
    ts = settle_time(t, i, 0.01)
    assert 4 < ts < 6.5, ts

    # the record-limited guard: the same trace dipping out at 10 s of a 20 s record
    i = 100e-9 * (1 - np.exp(-t / 0.2))
    i[(t > 9) & (t < 10)] *= 1.05
    assert np.isnan(settle_time(t, i, 0.01))

    print("ok  calibrate.settle_time: settles / never settles / already settled / "
          "noise-aware / last-time-outside / resolution independent of record length")


if __name__ == "__main__":
    demo()
