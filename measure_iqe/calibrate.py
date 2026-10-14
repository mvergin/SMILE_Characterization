"""Measure one array's settling behaviour and save it as a reusable calibration.

    python -m measure_iqe --calibrate --out D:\\cal        # once per array, ~6 min
    python -m measure_iqe --calibration D:\\cal\\<...>_calibration.yaml

Why this exists: **the settle time is a property of the array, not of the method.**
The 8 s default came from one array (Array38, 42.8 uF). An array with 7 uF may well
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

CALIBRATION_VERSION = 1


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
    w = max(5, i.size // 20)
    smooth = np.convolve(i, np.ones(w) / w, mode="valid")
    ts = t[w - 1:]
    noise = float(np.std(np.diff(i[int(0.8 * i.size):]))) / np.sqrt(2)
    tol = max(frac * abs(final), 5.0 * noise / np.sqrt(w))
    early = float(np.median(i[int(0.6 * i.size):int(0.8 * i.size)]))
    if abs(early - final) > max(tol, abs(frac * final)):
        return float("nan")              # the tail is still moving: not settled at all
    bad = np.nonzero(np.abs(smooth - final) > tol)[0]
    if bad.size == 0:
        return 0.0                       # settled before the first smoothed sample
    if bad[-1] + 1 >= smooth.size:
        return float("nan")              # still outside the band at the end
    return float(ts[bad[-1]])


def settle_curve(pm, smu, cfg: IQEConfig, v_to: float, seconds: float,
                 period: float = 0.05) -> tuple[np.ndarray, np.ndarray]:
    """Step ch B to `v_to` and poll |I| for `seconds`. Returns (t, |I|)."""
    from .acquire import acquire_point

    smu.set_voltage("b", float(v_to))
    t0 = time.perf_counter()
    ts: list[float] = []
    iv: list[float] = []
    while (dt := time.perf_counter() - t0) < seconds:
        _, _, _, _, inv, _, _, _ = acquire_point(pm, smu, cfg.calibrate_n_smu, 0)
        ts.append(dt)
        iv.append(abs(inv))
        rest = period - (time.perf_counter() - t0 - dt)
        if rest > 0:
            time.sleep(rest)
    return np.asarray(ts), np.asarray(iv)


def _ladder(lo: float, hi: float) -> list[float]:
    return [r for r in SMU_I_RANGES if lo <= r <= hi] or [hi]


def calibrate(pm, smu, smile, pixels: list[tuple[int, int]], cfg: IQEConfig,
              log: Logger) -> dict[str, Any]:
    """Measure the settling behaviour of this array. Returns the calibration dict."""
    from .acquire import configure_voltage_source_b, pick_limit, set_nplc

    # Three biases spanning the sweep: the start, the density knee, the faint end.
    # The faint end is the one that sets the budget, so it is measured on every pixel.
    knee = cfg.vramp_knee_v if cfg.vramp_knee_v is not None else (
        (cfg.nvled_v_coarse_start + cfg.nvled_v_coarse_stop) / 2)
    biases = [cfg.nvled_v_coarse_stop, float(knee), cfg.nvled_v_coarse_start]
    step = abs(cfg.nvled_v_coarse_start - knee) / max(
        1, round(cfg.n_fine * cfg.vramp_low_frac))
    log(f"CALIBRATION — {len(pixels)} pixel(s), biases {biases}, "
        f"{cfg.calibrate_seconds:g}s each; sweep step on the rising side is {step*1e3:.0f} mV")

    rows: list[dict[str, Any]] = []
    for (x, y) in pixels:
        smile.blank_frame()
        smile.set_pixel(x, y, cfg.bitvals[0])
        prev_i = None
        for v in biases:
            limit = pick_limit(prev_i, cfg)
            configure_voltage_source_b(
                smu, __import__("dataclasses").replace(cfg, nvled_compliance=limit), log)
            set_nplc(smu, cfg.vramp_lowv_nplc if v > knee else cfg.vramp_highv_nplc)
            # Approach each bias the way the sweep does -- from the previous, larger
            # current -- so the measured time is the time the sweep will actually need.
            t, i = settle_curve(pm, smu, cfg, v, cfg.calibrate_seconds,
                                cfg.calibrate_period_s)
            prev_i = float(np.median(i[int(0.8 * i.size):])) if i.size else None
            r = dict(pixel_x=x, pixel_y=y, v_to=v, limit_a=limit,
                     settled_a=prev_i,
                     t_1pct=settle_time(t, i, 0.01),
                     t_01pct=settle_time(t, i, 0.001),
                     n_samples=int(i.size))
            rows.append(r)
            log(f"  ({x},{y}) {v:+.3f} V  I={(prev_i or 0)*1e9:9.1f} nA  "
                f"limit {limit:g} A  t_1%={r['t_1pct']:.2f}s  t_0.1%={r['t_01pct']:.2f}s")
    smile.blank_frame()

    def worst(key: str, keep) -> float:
        vals = [r[key] for r in rows if keep(r) and np.isfinite(r[key])]
        return float(max(vals)) if vals else float("nan")

    rising = worst("t_1pct", lambda r: r["v_to"] > knee)
    droop = worst("t_1pct", lambda r: r["v_to"] <= knee)
    never = [r for r in rows if not np.isfinite(r["t_1pct"])]
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
        margin=m,
        rising_step_v=round(step, 5),
        recommended=rec,
        never_settled=[{k: r[k] for k in ("pixel_x", "pixel_y", "v_to")} for r in never],
        measurements=rows,
    )
    if never:
        log(f"  !! {len(never)} capture(s) never settled within {cfg.calibrate_seconds:g}s. "
            f"The recommendation is a LOWER BOUND — re-run with a longer "
            f"--set calibrate_seconds.")
    log(f"  recommended: " + ", ".join(f"{k}={v}" for k, v in rec.items()))
    return out


CALIB_DIR = "calibrations"      # gitignored, so a share re-export never clobbers it


def calibration_path(sample_name: str, root: str | Path = ".") -> Path:
    """Where `--array NAME` looks for, and `--calibrate` saves, a calibration."""
    return Path(root) / CALIB_DIR / f"{sample_name}.yaml"


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
    stable = _dump(calibration_path(name, root), data)
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

    # already settled: flat plus noise -> 0.0, not NaN
    i = 100e-9 + np.random.default_rng(0).normal(0, 1e-12, t.size)
    assert settle_time(t, i, 0.01) == 0.0

    # noise-aware: a 1% band on a tiny current would be set by noise alone, so a
    # trace that is flat-but-noisy must still count as settled
    i = 1e-9 + np.random.default_rng(1).normal(0, 5e-10, t.size)
    assert settle_time(i=i, t=t, frac=0.01) == 0.0

    # last-time-outside: a trace that dips back out at 10 s reports ~10 s, not ~1 s
    i = 100e-9 * (1 - np.exp(-t / 0.2))
    i[(t > 9) & (t < 10)] *= 1.05
    ts = settle_time(t, i, 0.01)
    assert 9 < ts < 11, ts

    print("ok  calibrate.settle_time: settles / never settles / already settled / "
          "noise-aware / last-time-outside")


if __name__ == "__main__":
    demo()
