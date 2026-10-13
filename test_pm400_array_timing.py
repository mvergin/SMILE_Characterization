"""PM400 array-mode timing sanity check.

The PM400 always captures 10 000 internal samples at 100 us (= 1.0 s).
The delta_t parameter is a decimation factor: each output sample
averages delta_t/100 internal samples.  The constraint is:

    (delta_t / 100) * n_samples <= 10000

This script verifies that:
  - poll_array_complete() always takes ~1 s regardless of delta_t
  - The returned sample count matches 10000 / (delta_t / 100)
  - The fetched data length matches the requested count

Run with:  uv run python test_pm400_array_timing.py

Edit PM400_RESOURCE below to match your VISA address.
"""

import time
import sys

# ── Configuration ──────────────────────────────────────────────────
PM400_RESOURCE = None  # Set to e.g. "USB0::0x1313::0x8079::..." or None for auto-detect

DELTA_T_VALUES = [100, 200, 500, 1000]  # µs
EXPECTED_CAPTURE_S = 1.0  # always 1 s
# ───────────────────────────────────────────────────────────────────


def find_pm400():
    """Auto-detect the first PM400 on the VISA bus."""
    import pyvisa
    rm = pyvisa.ResourceManager()
    for addr in rm.list_resources():
        try:
            dev = rm.open_resource(addr)
            dev.timeout = 3000
            dev.read_termination = "\n"
            dev.write_termination = "\n"
            idn = dev.query("*IDN?").strip()
            if "PM400" in idn:
                print(f"Found PM400: {idn}")
                print(f"Address:     {addr}")
                dev.close()
                rm.close()
                return addr
            dev.close()
        except Exception:
            pass
    rm.close()
    return None


def main():
    from instruments.pm400 import PM400

    addr = PM400_RESOURCE
    if addr is None:
        print("Auto-detecting PM400...")
        addr = find_pm400()
        if addr is None:
            print("ERROR: No PM400 found on VISA bus.")
            sys.exit(1)
    print()

    pm = PM400(addr)
    pm.set_power_unit("W")
    pm.set_auto_range(False)
    pm.set_range(1e-3)
    pm.set_averaging(1)

    print(f"  delta_t   decimation  exp_samples  act_samples   poll_time   fetch_len   match?  notes")
    print("-" * 100)

    for delta_t_us in DELTA_T_VALUES:
        decimation = delta_t_us // 100
        expected_samples = 10000 // decimation
        # Pass window_ms=1000 (1 s) — driver will clamp n_samples
        # to the hardware max for this delta_t.
        window_ms = 1000.0

        # Configure and arm
        pm.abort()
        time.sleep(0.05)
        n_actual, dt_actual = pm.configure_array_mode(window_ms, delta_t_us)

        notes = []
        if n_actual != expected_samples:
            notes.append(f"n_samples={n_actual} (expected {expected_samples})")
        if dt_actual != delta_t_us:
            notes.append(f"actual_dt={dt_actual}us (requested {delta_t_us}us)")

        pm.start_array()
        t_start = time.perf_counter()

        # Poll until complete (generous timeout)
        poll_ok = pm.poll_array_complete(timeout_s=EXPECTED_CAPTURE_S + 5.0)
        t_poll = time.perf_counter() - t_start

        if not poll_ok:
            notes.append("POLL TIMEOUT")

        # Fetch data
        pm.abort()
        data = pm.fetch_array(n_actual)
        fetch_len = len(data)

        if fetch_len != n_actual:
            notes.append(f"fetch got {fetch_len}")

        match = "OK" if abs(t_poll - EXPECTED_CAPTURE_S) < 0.3 else "MISMATCH"
        if not poll_ok:
            match = "TIMEOUT"

        print(
            f"  {delta_t_us:>6d}us  "
            f"{decimation:>10d}x  "
            f"{expected_samples:>11d}  "
            f"{n_actual:>11d}  "
            f"{t_poll:>9.3f}s  "
            f"{fetch_len:>9d}  "
            f"{match:>8s}  "
            f"{'; '.join(notes) if notes else ''}"
        )

    pm.close()
    print("\nDone.")


if __name__ == "__main__":
    main()
