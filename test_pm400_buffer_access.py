"""PM400 raw-buffer access probe.

The manual states the internal scope buffer always stores 10000 raw samples
at 10 kHz and that "data is processed and averaged during data access later".
The user-visible configuration is ``CONF:ARR <n_samples>, <delta_t>`` where
``delta_t/100 * n_samples <= 10000``.

Open question (from conversation): when delta_t > 100 (e.g. delta_t=500,
n_samples=2000), are indices 2000..9999 accessible via ``FETC? <i>`` /
``FETC:ARR? i, N``?  If yes, do they return:

    (a) zeros / NaN / error — the hardware drops them
    (b) the raw 10 kHz samples — decimation happens at read time
    (c) stale data from a previous capture
    (d) repeated copies of the decimated output

Determining (b) would let us always fetch the full 10k raw buffer
regardless of ``delta_t``, giving us the best of both worlds — coarse
indexing for fast throughput AND the raw data for detailed analysis.

This script runs 6 experiments against a real PM400 and prints the
evidence.  It requires a stable (ideally steady-state) light source
during the run so that decimation artefacts are distinguishable from
real signal variation.  A 1 mW laser into the sensor works fine.

Run with:  uv run python test_pm400_buffer_access.py
Edit PM400_RESOURCE below to match your VISA address (or None for auto).
"""

from __future__ import annotations

import struct
import sys
import time

# ── Configuration ──────────────────────────────────────────────────
PM400_RESOURCE = None  # Set to e.g. "USB0::0x1313::0x8079::..." or None
POWER_RANGE_W = 1e-7
WAVELENGTH_NM = 550.0

# The two configurations we compare.  BASELINE captures every internal
# sample; DECIMATED returns fewer samples by averaging blocks of 5.
BASELINE_N, BASELINE_DT = 10000, 100  # full 10 kHz
DECIMATED_N, DECIMATED_DT = 2000, 500  # 2 kHz after decimation

# Test read ranges. Covers "in-bounds" (< n_samples), "at the edge"
# (n_samples ± a few), and "deep past end" (near 10000).
TEST_INDICES = [0, 10, 100, 1999, 2000, 2001, 2500, 5000, 9000, 9999]
# Chunk size for FETC:ARR? (manual says max 40, driver uses 40).
CHUNK = 40
# ───────────────────────────────────────────────────────────────────


def _auto_detect_pm400():
    import pyvisa

    rm = pyvisa.ResourceManager()
    for addr in rm.list_resources():
        try:
            dev = rm.open_resource(addr)
            dev.timeout = 3000
            dev.read_termination = "\n"
            dev.write_termination = "\n"
            idn = dev.query("*IDN?").strip()
            dev.close()
            if "PM400" in idn:
                rm.close()
                return addr
        except Exception:
            pass
    rm.close()
    return None


def _open_raw(addr):
    """Open a raw pyvisa handle — we deliberately bypass our driver
    so we can experiment with SCPI commands not exposed in pm400.py."""
    import pyvisa

    rm = pyvisa.ResourceManager()
    inst = rm.open_resource(addr)
    inst.timeout = 5000
    inst.read_termination = "\n"
    inst.write_termination = "\n"
    idn = inst.query("*IDN?").strip()
    print(f"Connected: {idn}")
    inst.write("*CLS")
    return rm, inst


def _prep_sensor(inst):
    """Put the meter into a clean state for array captures."""
    inst.write("ABOR")
    time.sleep(0.05)
    inst.write(f"SENS:CORR:WAV {WAVELENGTH_NM}")
    inst.write("SENS:POW:UNIT W")
    inst.write("SENS:POW:RANG:AUTO 0")
    inst.write(f"SENS:POW:RANG {POWER_RANGE_W}")
    inst.write("SENS:AVER 1")
    inst.write("INP:FILT 0")
    time.sleep(0.2)


def _drain_errors(inst, label=""):
    """Drain the SCPI error queue and return any errors found."""
    found = []
    for _ in range(20):
        try:
            resp = inst.query("SYST:ERR?").strip()
        except Exception as e:
            found.append(f"<query-error: {e}>")
            break
        head = resp.split(",", 1)[0]
        if head in ("0", "+0"):
            break
        found.append(resp)
    if found:
        print(f"  [errors after {label}]: {found}")
    return found


def _configure_array(inst, n_samples, delta_t_us):
    inst.write("ABOR")
    time.sleep(0.05)
    inst.write(f"CONF:ARR {n_samples},{delta_t_us}")
    time.sleep(0.05)


def _start_and_wait(inst, timeout_s=3.0):
    inst.write("INIT")
    deadline = time.perf_counter() + timeout_s
    while time.perf_counter() < deadline:
        try:
            state = int(inst.query("FETC:STAT?").strip())
            if state == 1:
                return True
        except Exception:
            pass
        time.sleep(0.005)
    return False


def _fetch_single(inst, index):
    """ASCII single-sample read via ``FETC? <index>``.

    Returns (value_or_None, error_string_or_None).
    """
    try:
        resp = inst.query(f"FETC? {index}").strip()
        try:
            return float(resp), None
        except ValueError:
            return None, f"non-numeric: {resp!r}"
    except Exception as e:
        return None, f"visa-error: {e}"


def _fetch_binary(inst, offset, count):
    """Binary block read via FETC:ARR? <offset>, <count>.

    Returns (list_of_floats, error_string_or_None). The returned list
    may be shorter than ``count`` if the device declined to provide
    that many samples.
    """
    old_term = inst.read_termination
    try:
        inst.read_termination = None
        inst.write(f"FETC:ARR? {offset},{count}")
        raw = inst.read_raw()
    except Exception as e:
        inst.read_termination = old_term
        return [], f"visa-error: {e}"
    inst.read_termination = old_term

    try:
        comma_pos = raw.index(b",")
    except ValueError:
        return [], f"no-comma: {raw[:40]!r}"
    # Header: "<digit-count><count>,".  We only need the payload.
    payload = raw[comma_pos + 1 :].rstrip(b"\r\n")
    n_floats = len(payload) // 4
    if n_floats == 0:
        return [], f"empty-payload: header={raw[:comma_pos+1]!r}"
    vals = struct.unpack(f"<{n_floats}f", payload[: n_floats * 4])
    return list(vals), None


def _fetch_full_buffer_binary(inst, n_samples):
    """Fetch indices [0, n_samples) in 40-sample chunks."""
    values = []
    offset = 0
    while offset < n_samples:
        count = min(CHUNK, n_samples - offset)
        chunk, err = _fetch_binary(inst, offset, count)
        if err is not None:
            print(f"    !! fetch failed at offset={offset}: {err}")
            break
        values.extend(chunk)
        if len(chunk) < count:
            print(
                f"    !! short read at offset={offset}: "
                f"got {len(chunk)} expected {count}"
            )
            break
        offset += count
    return values


def _stat(values, n_head=5):
    """Summarise a list of floats."""
    if not values:
        return "<empty>"
    n = len(values)
    mn = min(values)
    mx = max(values)
    mean = sum(values) / n
    head = ", ".join(f"{v:.3e}" for v in values[:n_head])
    return (
        f"n={n}  min={mn:.3e}  max={mx:.3e}  mean={mean:.3e}  "
        f"head=[{head}]"
    )


# ═══════════════════════════════════════════════════════════════════
# Experiments
# ═══════════════════════════════════════════════════════════════════


def experiment_1_probe_indices(inst):
    """Configure CONF:ARR 2000, 500 — read specific indices beyond 2000.

    This is the headline question: can we reach indices 2000..9999 at all?
    If yes, we will analyse the contents in experiments 2/3.
    """
    print("\n[1] Probe indices at CONF:ARR 2000, 500")
    print("    (configured output=2000 samples, raw internal buffer still 10k)")
    _configure_array(inst, DECIMATED_N, DECIMATED_DT)
    ok = _start_and_wait(inst)
    if not ok:
        print("    !! INIT did not complete within timeout")
        return

    print("    Single-sample ASCII reads (FETC? <i>):")
    for i in TEST_INDICES:
        v, err = _fetch_single(inst, i)
        tag = f"{v:.3e}" if v is not None else f"FAIL({err})"
        print(f"      index {i:>5d}: {tag}")
    _drain_errors(inst, "single-sample reads")

    print("    Binary block reads (FETC:ARR? <i>, 40):")
    for i in TEST_INDICES:
        vals, err = _fetch_binary(inst, i, CHUNK)
        if err is not None:
            print(f"      offset {i:>5d}: FAIL ({err})")
        else:
            head = ", ".join(f"{x:.3e}" for x in vals[:3])
            print(
                f"      offset {i:>5d}: got {len(vals)} values; "
                f"head=[{head}]"
            )
    _drain_errors(inst, "binary block reads")


def experiment_2_extended_capture(inst):
    """Test whether CONF:ARR 10000, <delta_t> captures for n*delta_t seconds.

    An old Ultra-Fast diagnostic .npz from 2026-04-13 shows a chunk with
    ``pm_arr.shape = (10000,)`` at ``delta_t = 500 us`` — i.e. a 5-second
    capture.  This directly contradicts the manual's stated constraint
    ``delta_t * n_samples <= 10000`` (which would cap it at 1 s).  The
    PM400 apparently accepts the "illegal" config and captures for the
    full duration.

    This experiment re-runs that case and several longer ones to confirm
    current firmware behaviour:

        CONF:ARR 10000, 100   → 1 s   (baseline, always works)
        CONF:ARR 10000, 500   → 5 s   (the old .npz case)
        CONF:ARR 10000, 1000  → 10 s
        CONF:ARR 10000, 2000  → 20 s  (probes the real limit)

    For each: time the INIT→FETC:STAT=1 transition, fetch all 10000,
    and verify the returned values look like real shot-noise data (not
    repeats, not zeros, not -inf).
    """
    print("\n[2] Extended captures: CONF:ARR 10000, <delta_t>")
    print("    (manual says illegal; old Ultra-Fast diagnostics say it works)")

    configs = [
        (10000, 100, 1.0),    # baseline
        (10000, 500, 5.0),    # old .npz config
        (10000, 1000, 10.0),  # 10 s
        (10000, 2000, 20.0),  # 20 s — probe the actual limit
    ]

    # Guard with a reasonable safety ceiling: if poll hasn't completed
    # in expected_s + 5, we give up on that config.
    for n_samples, delta_t_us, expected_s in configs:
        print(f"\n    Trying CONF:ARR {n_samples}, {delta_t_us}  "
              f"(expected ~{expected_s:.0f}s capture)")
        _configure_array(inst, n_samples, delta_t_us)
        errs = _drain_errors(inst, f"CONF:ARR accept")
        if errs:
            print(f"      → PM400 REJECTED config: {errs}")
            continue

        t0 = time.perf_counter()
        ok = _start_and_wait(inst, timeout_s=expected_s + 5.0)
        t_poll = time.perf_counter() - t0
        if not ok:
            print(f"      → Poll TIMEOUT after {t_poll:.2f}s")
            # Try to salvage what we can
            inst.write("ABOR")
            time.sleep(0.05)
            continue
        print(f"      → Capture complete in {t_poll:.2f}s "
              f"(expected {expected_s:.1f}s; "
              f"{'MATCH' if abs(t_poll - expected_s) < 0.5 else 'MISMATCH'})")

        vals = _fetch_full_buffer_binary(inst, n_samples)
        print(f"      → Fetched {len(vals)}/{n_samples} samples")
        if not vals:
            continue

        # Distribution sanity: all finite? any -inf/zero runs?
        import math
        finite = sum(1 for v in vals if math.isfinite(v))
        neg_inf = sum(1 for v in vals if v == float("-inf"))
        near_zero = sum(1 for v in vals if abs(v) < 1e-15)
        print(f"      → finite={finite}/{len(vals)}, "
              f"-inf={neg_inf}, near-zero={near_zero}")
        print(f"      → {_stat(vals)}")

        # Spot-check the tail (would-be out-of-range region per the
        # manual's constraint) to confirm it looks like real data.
        if len(vals) >= 9950:
            tail = vals[9900:10000]
            print(f"      → tail [9900:10000]: {_stat(tail, n_head=3)}")

        _drain_errors(inst, f"after fetch {n_samples},{delta_t_us}")

    print("\n    Interpretation:")
    print("      If all four configs completed with ~expected durations and")
    print("      finite data throughout, the manual's delta_t*n_samples<=10000")
    print("      rule is a soft guideline.  The clamp in pm400.py")
    print("      configure_array_mode() can be relaxed to allow operators to")
    print("      pick any (n_samples<=10000, delta_t>=100us) pair and capture")
    print("      for n_samples*delta_t_us microseconds.")


def experiment_3_buffer_persistence(inst):
    """Does the 10k buffer persist across INIT without ABOR?

    If yes, we could in principle interleave two distinct captures
    (useful if we ever want free-running drift reference alongside
    a gated capture). If no, every INIT starts a clean buffer.
    """
    print("\n[3] Does the internal buffer persist across INIT?")

    # Capture a known reference
    print("    Capture A: CONF:ARR 10000, 100")
    _configure_array(inst, BASELINE_N, BASELINE_DT)
    if not _start_and_wait(inst, timeout_s=3.0):
        print("    !! Capture A failed")
        return
    ref = _fetch_full_buffer_binary(inst, BASELINE_N)
    if len(ref) < 100:
        print("    !! Capture A returned too few samples")
        return
    ref_head_mean = sum(ref[:100]) / 100
    print(f"      Capture A head mean: {ref_head_mean:.3e}")

    # Cover the sensor (or change light) BETWEEN A and B to make
    # the difference obvious.  We can't control the light from here,
    # so just pause and note the time — operator may manually change.
    print("    (If possible, change the light level now — pausing 2 s)")
    time.sleep(2.0)

    # Capture B — but fetch before it completes?  Instead, just
    # reconfigure to a different n_samples and see if the buffer
    # retains old data.
    print("    Capture B: CONF:ARR 10000, 100 (new INIT, no ABOR first)")
    inst.write(f"CONF:ARR {BASELINE_N},{BASELINE_DT}")
    time.sleep(0.05)
    if not _start_and_wait(inst, timeout_s=3.0):
        print("    !! Capture B failed")
        return
    now = _fetch_full_buffer_binary(inst, BASELINE_N)
    if len(now) < 100:
        print("    !! Capture B returned too few samples")
        return
    now_head_mean = sum(now[:100]) / 100
    print(f"      Capture B head mean: {now_head_mean:.3e}")

    delta_rel = abs(now_head_mean - ref_head_mean) / max(
        abs(ref_head_mean), 1e-15
    )
    print(
        f"      Relative delta: {delta_rel:.3f} "
        f"({'DIFFERENT' if delta_rel > 0.05 else 'SAME'} — within 5%)"
    )
    _drain_errors(inst, "experiment 3")


def experiment_4_ascii_vs_binary(inst):
    """Do FETC? <i> (ASCII) and FETC:ARR? <i>, 1 (binary) agree?

    Sanity check — they should, but worth confirming since we've been
    mixing them up through this probe.  If they disagree, the fault
    lies in our binary decoder.
    """
    print("\n[4] ASCII (FETC? <i>) vs binary (FETC:ARR? <i>, 1) parity")
    _configure_array(inst, BASELINE_N, BASELINE_DT)
    if not _start_and_wait(inst, timeout_s=3.0):
        print("    !! INIT failed")
        return
    for i in [0, 100, 5000, 9999]:
        v_ascii, err_a = _fetch_single(inst, i)
        v_bin, err_b = _fetch_binary(inst, i, 1)
        ascii_str = f"{v_ascii:.6e}" if v_ascii is not None else f"ERR({err_a})"
        if err_b is not None:
            bin_str = f"ERR({err_b})"
        elif not v_bin:
            bin_str = "<empty>"
        else:
            bin_str = f"{v_bin[0]:.6e}"
        agree = "—"
        if v_ascii is not None and v_bin:
            rel = abs(v_ascii - v_bin[0]) / max(abs(v_ascii), 1e-15)
            agree = "OK" if rel < 1e-3 else f"DIFFER rel={rel:.2e}"
        print(f"      idx {i:>5d}: ascii={ascii_str}  binary={bin_str}  {agree}")
    _drain_errors(inst, "experiment 4")


def experiment_5_no_init_fetch(inst):
    """What does FETC? return BEFORE any INIT?

    Probes whether the buffer comes up in a defined state (e.g. zeros)
    or whether it retains whatever was there before.
    """
    print("\n[5] Fetch from buffer without a fresh INIT")
    inst.write("ABOR")
    time.sleep(0.1)
    # Re-configure but don't INIT
    _configure_array(inst, BASELINE_N, BASELINE_DT)
    state = None
    try:
        state = inst.query("FETC:STAT?").strip()
    except Exception as e:
        state = f"error: {e}"
    print(f"      FETC:STAT? (no INIT): {state}")
    for i in [0, 5000, 9999]:
        v, err = _fetch_single(inst, i)
        tag = f"{v:.3e}" if v is not None else f"FAIL({err})"
        print(f"      FETC? {i}: {tag}")
    _drain_errors(inst, "experiment 5")


def experiment_6_partial_init_abort(inst):
    """If we ABOR mid-capture, can we still FETC? samples that were taken?

    The array capture takes 1 s regardless of delta_t.  If we start
    one and abort at 500 ms, presumably half the buffer is populated;
    the question is whether we can still read it out.
    """
    print("\n[6] Fetch after mid-capture ABOR")
    _configure_array(inst, BASELINE_N, BASELINE_DT)
    inst.write("INIT")
    time.sleep(0.5)  # half a capture
    inst.write("ABOR")
    time.sleep(0.1)
    try:
        state = inst.query("FETC:STAT?").strip()
    except Exception as e:
        state = f"error: {e}"
    print(f"      FETC:STAT? after abort: {state}")
    for i in [0, 2000, 4000, 6000, 9999]:
        v, err = _fetch_single(inst, i)
        tag = f"{v:.3e}" if v is not None else f"FAIL({err})"
        print(f"      FETC? {i}: {tag}")
    _drain_errors(inst, "experiment 6")


# ═══════════════════════════════════════════════════════════════════


def main():
    addr = PM400_RESOURCE or _auto_detect_pm400()
    if addr is None:
        print("ERROR: No PM400 found on VISA bus.")
        sys.exit(1)
    print(f"Using PM400 @ {addr}\n")

    rm, inst = _open_raw(addr)
    try:
        _prep_sensor(inst)
        experiment_1_probe_indices(inst)
        experiment_2_extended_capture(inst)
        experiment_3_buffer_persistence(inst)
        experiment_4_ascii_vs_binary(inst)
        experiment_5_no_init_fetch(inst)
        experiment_6_partial_init_abort(inst)
        print("\nDone.")
    finally:
        try:
            inst.write("ABOR")
            inst.close()
        except Exception:
            pass
        try:
            rm.close()
        except Exception:
            pass


if __name__ == "__main__":
    main()
