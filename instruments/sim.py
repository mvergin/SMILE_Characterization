"""Simulation mocks for PM400, Keithley 2602B, and SMILE FPGA.

Each class mirrors the interface of its real counterpart. Used when the
corresponding `sim_*` checkbox is ticked in the GUI, or when the real
driver module is unavailable (import failure, no hardware).
"""

import random
import time


class PM400Sim:
    """Mock PM400 power meter supporting scalar, continuous, and array modes."""

    def __init__(self, resource_id):
        print(f"[SIM] PM400 @ {resource_id}")
        self._arr_n_samples = 0
        self._arr_delta_t_us = 100
        self._mode = "idle"

    def set_wavelength(self, wl):
        pass

    def set_power_unit(self, unit="W"):
        pass

    def set_auto_range(self, enable):
        pass

    def set_range(self, upper_limit):
        pass

    def set_averaging(self, count):
        pass

    def measure(self):
        return random.random() * 1e-6

    def display_off(self):
        pass

    def display_on(self):
        pass

    def start_continuous(self):
        self._mode = "continuous"

    def stop_continuous(self):
        self._mode = "idle"

    def fetch_latest(self):
        return random.random() * 1e-6

    def abort(self):
        self._mode = "idle"

    def configure_scalar(self, wavelength=None, range_W=None, averaging=None):
        self._mode = "scalar"

    def configure_array_mode(self, window_ms, delta_t_us=100):
        delta_t_us = max(100, int(round(delta_t_us / 100.0)) * 100)
        decimation = delta_t_us // 100
        max_samples = 10000 // decimation
        n = max(1, min(int(window_ms / (delta_t_us / 1000.0)), max_samples))
        self._arr_n_samples = n
        self._arr_delta_t_us = delta_t_us
        self._mode = "array"
        return n, delta_t_us

    def start_array(self):
        pass

    def poll_array_complete(self, timeout_s=5.0):
        return True

    def fetch_array(self, n_samples, start_offset=0):
        return [random.random() * 1e-6 for _ in range(n_samples)]

    def get_mode(self):
        return self._mode

    def get_config_dict(self):
        return {"sim": True}

    def check_error_queue(self):
        return []

    def close(self):
        pass


class Keithley2602BSim:
    """Mock Keithley 2602B dual-channel SMU."""

    def __init__(self, resource_id):
        print(f"[SIM] Keithley2602B @ {resource_id}")
        self._n_pts = 10
        self._buf_a = []
        self._buf_b = []

    def configure_channel(
        self,
        ch,
        compliance_current=0.1,
        nplc=1.0,
        high_c=False,
        zero_delays=False,
        range_i=1e-3,
    ):
        pass

    def set_voltage(self, ch, voltage):
        pass

    def enable_output(self, ch, enable):
        pass

    def display_off(self):
        pass

    def display_on(self):
        pass

    def setup_buffers(self, timestamps=False):
        self._buf_a = []
        self._buf_b = []

    def clear_buffers(self):
        self._buf_a = []
        self._buf_b = []

    def clear_buffers_only(self):
        self._buf_a = []
        self._buf_b = []

    def measure_both_to_buffer(self):
        self._buf_a.append(random.gauss(100e-6, 1e-6))
        self._buf_b.append(random.gauss(10e-3, 0.1e-3))

    def measure_burst(self, n):
        for _ in range(n):
            self._buf_a.append(random.gauss(100e-6, 1e-6))
            self._buf_b.append(random.gauss(10e-3, 0.1e-3))

    def measure_burst_fire(self, n):
        # sim: no async, just fill immediately
        self.measure_burst(n)

    def measure_burst_join(self):
        pass

    def read_buffers(self, n=None):
        a = self._buf_a[:n] if n is not None else list(self._buf_a)
        b = self._buf_b[:n] if n is not None else list(self._buf_b)
        return a, b

    def configure_hardware_trigger(self, count):
        self._n_pts = count

    def start_hardware_trigger(self):
        pass

    def abort_trigger(self):
        pass

    def measure_instant(self):
        return random.gauss(100e-6, 1e-6), random.gauss(10e-3, 0.1e-3)

    def read_buffer_with_timestamps(self):
        n = max(1, self._n_pts)
        ia = [random.gauss(100e-6, 1e-6) for _ in range(n)]
        ib = [random.gauss(10e-3, 0.1e-3) for _ in range(n)]
        ta = [i * 0.001 for i in range(n)]
        tb = [i * 0.001 for i in range(n)]
        return ia, ib, ta, tb

    def get_config_dict(self):
        return {"sim": True}

    def check_error_queue(self):
        return []

    def close(self):
        pass


class SmileFPGASim:
    """Mock FPGA display controller.

    Mirrors the interface of the real SmileFPGA wrapper: set_pixel() and
    blank_frame() both return an ACK timestamp (perf_counter).
    """

    def __init__(self, resource_id=None):
        print("[SIM] SMILE FPGA")

    def set_pixel(self, log_x, log_y, bit_val):
        # Simulate the ~5 ms ACK latency of a real set_pixel call.
        time.sleep(0.005)
        return time.perf_counter()

    def blank_frame(self):
        time.sleep(0.005)
        return time.perf_counter()

    def close(self):
        pass
