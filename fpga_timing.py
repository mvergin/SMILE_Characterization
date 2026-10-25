"""How long does one FPGA frame write take, and does it depend on how long the link idled?

    python fpga_timing.py [--pixel X Y] [--idle 0.5 2 5 10 30 60]

The switch test on Array38 (261025-1211, -1222, -1224) measured ~1 s from sending
set_pixel to its ACK, and the display changed only at the END of that second. The fast
scan writes frames every ~30 ms, so the FPGA is not that slow when it is kept busy. Every
switch write follows a 60 s hold with no FPGA traffic. This times set_pixel back to back,
then after each idle period, to see whether the first write after an idle is the slow one.

FPGA only: the SMU and PM400 are not opened, so the array is unpowered and nothing lights.
"""
from __future__ import annotations

import argparse
import statistics
import time

from smile.core import Smile
from instruments.smile_fpga import SmileFPGA


def timed(fn, *a) -> float:
    t = time.perf_counter()
    fn(*a)
    return 1e3 * (time.perf_counter() - t)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--pixel", nargs=2, type=int, default=(100, 100))
    ap.add_argument("--idle", nargs="+", type=float, default=[0.5, 2, 5, 10, 30, 60])
    args = ap.parse_args()
    smile = SmileFPGA(Smile(timeout=5000, debug_lvl=0))
    x, y = args.pixel
    try:
        b2b = [timed(smile.set_pixel, x, y, 15) for _ in range(20)]
        print(f"set_pixel back to back : median {statistics.median(b2b):7.1f} ms, "
              f"max {max(b2b):7.1f} ms (n=20)")
        bl = [timed(smile.blank_frame) for _ in range(5)]
        print(f"blank_frame back to back: median {statistics.median(bl):7.1f} ms (n=5)")
        for idle in args.idle:
            time.sleep(idle)
            first = timed(smile.set_pixel, x, y, 15)
            second = timed(smile.set_pixel, x, y, 15)
            print(f"after {idle:5.1f} s idle   : first {first:7.1f} ms, second {second:7.1f} ms")
    finally:
        smile.blank_frame()
        smile.close()


if __name__ == "__main__":
    main()
