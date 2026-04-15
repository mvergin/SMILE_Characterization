"""Keithley 2602B dual-channel SMU driver (TSP).

Extracted from the pre-refactor `instrumentlib.py`. Behavioural changes
over the original:

  * **Bug 1 fix — High-C mode configuration.** The old driver
    unconditionally applied `zero_delays=True` (source/measure delays = 0
    and autozero OFF) regardless of the `high_c` flag. On capacitive
    loads this produces noisy first samples and, on channel B of some
    2602B firmware revisions, silently misreports current. When
    `high_c=True` we now use `DELAY_AUTO` + `AUTOZERO_ONCE`, and we
    reject the combination on channel B entirely — high-C mode on
    channel B is unreliable on the 2602B and should use channel A.

  * `check_error_queue()` — drains the Keithley error queue via
    `errorqueue.next()` and returns the list of error strings.
"""

from __future__ import annotations

import time

import pyvisa


class Keithley2602B:
    """Dual-channel Keithley 2602B (TSP). smua=VLED, smub=NVLED."""

    SETTLE_TIME = 0.1

    def __init__(self, resource_id):
        self.rm = pyvisa.ResourceManager()
        try:
            self.inst = self.rm.open_resource(resource_id)
            self.inst.clear()
            time.sleep(0.5)
            self.inst.timeout = 10000
            self.inst.read_termination = "\n"
            self.inst.write_termination = "\n"

            idn = self.inst.query("*IDN?").strip()
            print(f"Connected to Dual-Channel K2602B: {idn}")
            if "26" not in idn:
                print("WARNING: Device IDN does not look like a Keithley 26xx.")

            self.inst.write("smua.reset() smub.reset()")
            time.sleep(1.0)

        except pyvisa.VisaIOError as e:
            print(f"Error connecting to K2602B ({resource_id}): {e}")
            raise

    def display_off(self):
        """Switch both display rows to static text in one TSP write."""
        self.inst.write(
            'display.clear() display.setcursor(1,1) '
            'display.settext("Measuring Smile") display.setcursor(2,1) '
            'display.settext("Powered by Georg and Max")'
        )

    def display_on(self):
        self.inst.write("display.screen = display.SMUA_SMUB")

    def close(self):
        if self.inst:
            self._write_and_wait("smua.source.output = smua.OUTPUT_OFF")
            self._write_and_wait("smub.source.output = smub.OUTPUT_OFF")
            self.inst.close()
        print("K2602B Connection closed.")

    def _write_and_wait(self, cmd):
        self.inst.write(cmd)
        time.sleep(self.SETTLE_TIME)

    # ------------------------------------------------------------------
    # Channel configuration
    # ------------------------------------------------------------------

    def configure_channel(
        self,
        channel,
        compliance_current=0.1,
        nplc=1.0,
        high_c=False,
        zero_delays=False,
        range_i=1e-3,
    ):
        """Configure one SMU channel.

        Parameters
        ----------
        channel : str
            "a" or "b".
        compliance_current : float
            Current limit in amps.
        nplc : float
            Integration time in NPLCs.
        high_c : bool
            Enable High-Capacitance mode. Only valid on channel A —
            raises ValueError on channel B.
        zero_delays : bool
            Request zero source/measure delays + autozero OFF for
            fastest throughput. Silently ignored when `high_c=True`
            (high-C mode requires settling delays to work).
        range_i : float
            Fixed current measurement range in amps.
        """
        ch_letter = channel.lower()
        if high_c and ch_letter == "b":
            raise ValueError(
                "high_c=True is not supported on channel B of the 2602B; "
                "use channel A for high-capacitance loads."
            )
        ch = f"smu{ch_letter}"

        self._write_and_wait(f"{ch}.source.func = {ch}.OUTPUT_DCVOLTS")
        self._write_and_wait(f"{ch}.source.limiti = {compliance_current}")
        self._write_and_wait(f"{ch}.measure.nplc = {nplc}")
        self._write_and_wait(f"{ch}.measure.autorangei = {ch}.AUTORANGE_OFF")
        self._write_and_wait(f"{ch}.measure.rangei = {range_i}")

        highc_state = f"{ch}.ENABLE" if high_c else f"{ch}.DISABLE"
        self._write_and_wait(f"{ch}.source.highc = {highc_state}")

        if high_c:
            # Bug 1 fix: high-C mode requires DELAY_AUTO + AUTOZERO_ONCE
            # to give the capacitor time to charge before each measurement.
            # Zero delays here will yield unreliable readings.
            self._write_and_wait(f"{ch}.source.delay = {ch}.DELAY_AUTO")
            self._write_and_wait(f"{ch}.measure.delay = {ch}.DELAY_AUTO")
            self._write_and_wait(f"{ch}.measure.autozero = {ch}.AUTOZERO_ONCE")
        elif zero_delays:
            self._write_and_wait(f"{ch}.source.delay = 0")
            self._write_and_wait(f"{ch}.measure.delay = 0")
            self._write_and_wait(f"{ch}.measure.autozero = {ch}.AUTOZERO_OFF")

    def set_voltage(self, channel, voltage):
        ch = f"smu{channel.lower()}"
        self._write_and_wait(f"{ch}.source.levelv = {voltage}")

    def enable_output(self, channel, enable: bool):
        ch = f"smu{channel.lower()}"
        state = f"{ch}.OUTPUT_ON" if enable else f"{ch}.OUTPUT_OFF"
        self._write_and_wait(f"{ch}.source.output = {state}")

    def measure_instant(self):
        """Single immediate measurement on both channels, no buffer."""
        resp = self.inst.query("print(smua.measure.i(), smub.measure.i())")
        parts = resp.strip().split()
        ia = float(parts[0]) if len(parts) > 0 else float("nan")
        ib = float(parts[1]) if len(parts) > 1 else float("nan")
        return ia, ib

    # ------------------------------------------------------------------
    # Buffer / burst
    # ------------------------------------------------------------------

    def setup_buffers(self, timestamps=False):
        self._write_and_wait("smua.nvbuffer1.clear() smub.nvbuffer1.clear()")
        self._write_and_wait("smua.nvbuffer1.appendmode = 1")
        self._write_and_wait("smub.nvbuffer1.appendmode = 1")
        if timestamps:
            self._write_and_wait("smua.nvbuffer1.collecttimestamps = 1")
            self._write_and_wait("smub.nvbuffer1.collecttimestamps = 1")
        # Flush the TSP queue so all writes above have executed before the
        # first trigger arm.
        self.inst.query("print('SETUP_DONE')")

    def clear_buffers(self):
        self.setup_buffers(timestamps=False)

    def clear_buffers_only(self):
        """Clear buffer contents only — preserves timestamp / append mode."""
        self.inst.write("smua.nvbuffer1.clear() smub.nvbuffer1.clear()")

    def measure_both_to_buffer(self):
        self.inst.write(
            "smua.measure.i(smua.nvbuffer1) smub.measure.i(smub.nvbuffer1)"
        )

    def run_tsp_and_wait(self, tsp_code):
        self.inst.write(tsp_code)
        res = self.inst.query("print('DONE')").strip()
        if res != "DONE":
            self.inst.clear()
            raise RuntimeError(f"TSP queue desync: expected 'DONE', got '{res[:40]}'")

    def configure_hardware_trigger(self, count):
        self.inst.write(
            "smua.trigger.measure.i(smua.nvbuffer1) "
            "smub.trigger.measure.i(smub.nvbuffer1)"
        )
        self.inst.write(f"smua.trigger.count = {count}")
        self.inst.write(f"smub.trigger.count = {count}")
        self.inst.write(
            "smua.trigger.measure.action = smua.ENABLE "
            "smub.trigger.measure.action = smub.ENABLE"
        )

    def start_hardware_trigger(self):
        self.inst.write("smua.trigger.initiate() smub.trigger.initiate()")

    def measure_burst(self, n):
        self.inst.query(
            f"for i=1,{n} do "
            f"smua.measure.i(smua.nvbuffer1) "
            f"smub.measure.i(smub.nvbuffer1) "
            f"end print('DONE')"
        )

    def measure_burst_fire(self, n):
        self.inst.write(
            f"for i=1,{n} do "
            f"smua.measure.i(smua.nvbuffer1) "
            f"smub.measure.i(smub.nvbuffer1) "
            f"end print('DONE')"
        )

    def measure_burst_join(self):
        resp = self.inst.read().strip()
        if resp != "DONE":
            self.inst.clear()
            raise RuntimeError(f"TSP burst sync error: expected 'DONE', got '{resp[:40]}'")

    def read_buffers(self, n=None):
        try:
            n_a = int(float(self.inst.query("print(smua.nvbuffer1.n)")))
            n_b = int(float(self.inst.query("print(smub.nvbuffer1.n)")))
            count = min(n_a, n_b)
            if n is not None:
                count = min(count, n)
            if count == 0:
                return [], []
            ia_str = self.inst.query(f"printbuffer(1, {count}, smua.nvbuffer1)")
            ib_str = self.inst.query(f"printbuffer(1, {count}, smub.nvbuffer1)")
            return (
                [float(x) for x in ia_str.split(",")],
                [float(x) for x in ib_str.split(",")],
            )
        except Exception as e:
            print(f"read_buffers error: {e}")
            return [], []

    def abort_trigger(self):
        self.inst.write("smua.abort() smub.abort()")
        self.inst.write(
            "smua.trigger.measure.action = smua.DISABLE "
            "smub.trigger.measure.action = smub.DISABLE"
        )
        time.sleep(0.05)

    def read_buffer_with_timestamps(self):
        try:
            n_a = int(float(self.inst.query("print(smua.nvbuffer1.n)")))
            n_b = int(float(self.inst.query("print(smub.nvbuffer1.n)")))
            n = min(n_a, n_b)
            if n == 0:
                return [], [], [], []
            self.inst.write("format.data = format.ASCII")
            ia_str = self.inst.query(f"printbuffer(1, {n}, smua.nvbuffer1)")
            ib_str = self.inst.query(f"printbuffer(1, {n}, smub.nvbuffer1)")
            ta_str = self.inst.query(
                f"printbuffer(1, {n}, smua.nvbuffer1.timestamps)"
            )
            tb_str = self.inst.query(
                f"printbuffer(1, {n}, smub.nvbuffer1.timestamps)"
            )
            ia_vals = [float(x) for x in ia_str.split(",")]
            ib_vals = [float(x) for x in ib_str.split(",")]
            ta_vals = [float(x) for x in ta_str.split(",")]
            tb_vals = [float(x) for x in tb_str.split(",")]
            return ia_vals, ib_vals, ta_vals, tb_vals
        except Exception as e:
            print(f"Buffer read with timestamps error: {e}")
            return [], [], [], []

    # ------------------------------------------------------------------
    # Error queue
    # ------------------------------------------------------------------

    def check_error_queue(self):
        """Drain the Keithley error queue and return the list of messages.

        An empty list means the queue was clean.
        """
        errors = []
        try:
            count = int(float(self.inst.query("print(errorqueue.count)").strip()))
            for _ in range(min(count, 20)):
                resp = self.inst.query("print(errorqueue.next())").strip()
                errors.append(resp)
        except Exception:
            pass
        return errors

    def get_config_dict(self):
        config = {}
        try:
            config["idn"] = self.inst.query("*IDN?").strip()
            config["chA_voltage"] = float(
                self.inst.query("print(smua.source.levelv)")
            )
            config["chA_comp"] = float(
                self.inst.query("print(smua.source.limiti)")
            )
            config["chA_highc"] = int(
                float(self.inst.query("print(smua.source.highc)"))
            )
            config["chA_nplc"] = float(
                self.inst.query("print(smua.measure.nplc)")
            )
            config["chA_output"] = int(
                float(self.inst.query("print(smua.source.output)"))
            )

            config["chB_voltage"] = float(
                self.inst.query("print(smub.source.levelv)")
            )
            config["chB_comp"] = float(
                self.inst.query("print(smub.source.limiti)")
            )
            config["chB_highc"] = int(
                float(self.inst.query("print(smub.source.highc)"))
            )
            config["chB_nplc"] = float(
                self.inst.query("print(smub.measure.nplc)")
            )
            config["chB_output"] = int(
                float(self.inst.query("print(smub.source.output)"))
            )
        except Exception as e:
            config["error"] = str(e)
        return config
