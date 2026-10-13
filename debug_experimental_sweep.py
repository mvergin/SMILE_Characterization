"""Standalone 2602B debug script for the experimental NVLED sweep.

Run this with the Keithley plugged in (no PM400 / no FPGA required). It
walks the same sequence the real measurement does, but drains the error
queue after every step and prints the response of every query so you can
see exactly where the -285 / -420 pair originates.

    uv run python debug_experimental_sweep.py

Edit ``SMU_ADDR`` below if auto-detect doesn't find your unit. The sweep
runs with OUTPUTS ON into whatever is on the terminals — make sure
that's safe (or leave the outputs disconnected; the script still prints
the same TSP errors).
"""

from __future__ import annotations

import sys
import time

import pyvisa


# =========================  CONFIG  ==================================
SMU_ADDR = "USB0::0x05E6::0x2602::4649721::INSTR"  # None = auto-detect

# Same shape as a small phase-2 sweep — keep tiny to make logs readable.
V_LIST = [-1.0, -1.5, -2.0]
VLED = 1.8
SETTLE_S = 0.05  # auto 60/40 path
K = 20  # points per voltage

ENABLE_OUTPUTS = True  # set False to run TSP-only without driving anything
PROGRESSIVE = False  # True = send prep statements one at a time
#                        (pinpoints which line causes -285 if it's still broken)
# =====================================================================


def drain_errors(inst, label):
    """Print + clear the Keithley error queue, return the list."""
    errs = []
    try:
        n = int(float(inst.query("print(errorqueue.count)").strip()))
    except Exception as e:
        print(f"  [{label}] errorqueue.count query failed: {e}")
        return errs
    for _ in range(min(n, 20)):
        try:
            errs.append(inst.query("print(errorqueue.next())").strip())
        except Exception as e:
            errs.append(f"<read failed: {e}>")
    if errs:
        print(f"  [{label}] ERRORS ({n}): {errs}")
    else:
        print(f"  [{label}] error queue clean")
    return errs


def step(label):
    print(f"\n--- {label} ---")


def auto_detect():
    rm = pyvisa.ResourceManager()
    for addr in rm.list_resources():
        try:
            d = rm.open_resource(addr)
            d.timeout = 2000
            d.read_termination = "\n"
            d.write_termination = "\n"
            if "2602" in d.query("*IDN?").upper():
                d.close()
                return addr
            d.close()
        except Exception:
            pass
    return None


def main():
    addr = SMU_ADDR or auto_detect()
    if addr is None:
        print("ERROR: no 2602B found")
        sys.exit(1)

    rm = pyvisa.ResourceManager()
    inst = rm.open_resource(addr)
    inst.timeout = 10000
    inst.read_termination = "\n"
    inst.write_termination = "\n"

    step("Connect + reset")
    print("IDN:", inst.query("*IDN?").strip())
    inst.clear()
    inst.write("errorqueue.clear()")
    inst.write("smua.reset() smub.reset()")
    time.sleep(1.0)
    drain_errors(inst, "after reset")

    step("Configure channels (mirror of auto_measure)")
    cfg_lines = [
        "smua.source.func = smua.OUTPUT_DCVOLTS",
        "smua.source.limiti = 0.1",
        "smua.measure.nplc = 0.001",
        "smua.measure.autorangei = smua.AUTORANGE_OFF",
        "smua.measure.rangei = 10e-3",
        "smua.source.delay = 0",
        "smua.measure.delay = 0",
        "smua.measure.autozero = smua.AUTOZERO_OFF",
        "smub.source.func = smub.OUTPUT_DCVOLTS",
        "smub.source.limiti = 0.5",
        "smub.measure.nplc = 0.001",
        "smub.measure.autorangei = smub.AUTORANGE_OFF",
        "smub.measure.rangei = 10e-3",
        "smub.source.delay = 0",
        "smub.measure.delay = 0",
        "smub.measure.autozero = smub.AUTOZERO_OFF",
    ]
    inst.write("\n".join(cfg_lines))
    inst.query("print('CFG_DONE')")
    drain_errors(inst, "after configure")

    if ENABLE_OUTPUTS:
        step("Enable outputs")
        inst.write(
            f"smua.source.levelv = {VLED} "
            f"smub.source.levelv = {V_LIST[0]} "
            "smua.source.output = smua.OUTPUT_ON "
            "smub.source.output = smub.OUTPUT_ON"
        )
        inst.query("print('OUT_ON')")
        drain_errors(inst, "after enable outputs")

    step("Prep buffers + stash globals")
    v_str = "{" + ", ".join(f"{v:.6f}" for v in V_LIST) + "}"
    prep_lines = [
        "smua.nvbuffer1.clear()",
        "smub.nvbuffer1.clear()",
        "smub.nvbuffer2.clear()",
        "smua.nvbuffer1.appendmode = 1",
        "smub.nvbuffer1.appendmode = 1",
        "smub.nvbuffer2.appendmode = 1",
        "smua.nvbuffer1.collecttimestamps = 1",
        "smub.nvbuffer1.collecttimestamps = 1",
        f"smua.source.levelv = {VLED}",
        f"_exp_v = {v_str}",
        f"_exp_settle = {SETTLE_S}",
        f"_exp_k = {K}",
    ]

    if PROGRESSIVE:
        print("PROGRESSIVE=True — sending prep statements one by one:")
        for i, line in enumerate(prep_lines, 1):
            print(f"  [{i:2d}] {line}")
            inst.write(line)
            resp = inst.query("print('OK')").strip()
            errs = drain_errors(inst, f"after line {i}")
            if errs or resp != "OK":
                print(
                    f"  >>> FAILED at line {i} — probe returned {resp!r}, errors={errs}"
                )
                print("  aborting progressive prep; remaining lines skipped.")
                inst.close()
                rm.close()
                sys.exit(2)
    else:
        prep = "\n".join(prep_lines)
        print("TSP prep payload (newline-joined):")
        for ln in prep.splitlines():
            print(f"  | {ln}")
        inst.write(prep)
        resp = inst.query(
            "print('PREP', type(_exp_v), _exp_settle, _exp_k, table.getn(_exp_v))"
        ).strip()
        print(f"  prep probe -> {resp!r}")
        drain_errors(inst, "after prep")

    step("Capacity check (nvbuffer1/2)")
    for chan in ("smua", "smub"):
        for buf in ("nvbuffer1", "nvbuffer2"):
            try:
                cap = inst.query(f"print({chan}.{buf}.capacity)").strip()
                print(f"  {chan}.{buf}.capacity = {cap}")
            except Exception as e:
                print(f"  {chan}.{buf}.capacity query failed: {e}")
    drain_errors(inst, "after capacity check")

    step("Run TSP sweep (wrapped in loadandrunscript + pcall)")
    script = "\n".join(
        [
            "loadandrunscript",
            "local ok, err = pcall(function()",
            "  for i = 1, table.getn(_exp_v) do",
            "    smub.source.levelv = _exp_v[i]",
            "    if _exp_settle > 0 then delay(_exp_settle) end",
            "    for j = 1, _exp_k do",
            "      smub.measure.iv(smub.nvbuffer1, smub.nvbuffer2)",
            "      smua.measure.i(smua.nvbuffer1)",
            "    end",
            "  end",
            "end)",
            "if ok then",
            "  print('EXP_DONE')",
            "else",
            "  print('EXP_ERR:' .. tostring(err))",
            "end",
            "endscript",
        ]
    )
    print("TSP script payload:")
    print("  " + script)
    t0 = time.perf_counter()
    inst.write(script)
    inst.timeout = 30000
    try:
        resp = inst.read().strip()
    except Exception as e:
        print(f"  read failed after {time.perf_counter() - t0:.2f}s: {e}")
        resp = "<read timeout>"
    else:
        print(f"  sweep finished in {time.perf_counter() - t0:.2f}s: {resp!r}")
    drain_errors(inst, "after sweep")

    if resp == "EXP_DONE":
        step("Read buffers")
        for chan, buf, name in [
            ("smua", "nvbuffer1", "chA current"),
            ("smub", "nvbuffer1", "chB current"),
            ("smub", "nvbuffer2", "chB voltage"),
        ]:
            try:
                n = int(float(inst.query(f"print({chan}.{buf}.n)").strip()))
                print(f"  {chan}.{buf}.n = {n}  ({name})")
                if n > 0:
                    sample = inst.query(
                        f"printbuffer(1, {min(n, 5)}, {chan}.{buf})"
                    ).strip()
                    print(f"    first samples: {sample}")
            except Exception as e:
                print(f"  {chan}.{buf} read failed: {e}")
        drain_errors(inst, "after buffer read")

    step("Teardown")
    try:
        inst.write(
            "smua.source.output = smua.OUTPUT_OFF "
            "smub.source.output = smub.OUTPUT_OFF "
            "_exp_v = nil _exp_settle = nil _exp_k = nil"
        )
        inst.query("print('OFF')")
    except Exception as e:
        print(f"  teardown failed: {e}")
    drain_errors(inst, "after teardown")

    inst.close()
    rm.close()
    print("\nDone.")


if __name__ == "__main__":
    main()
