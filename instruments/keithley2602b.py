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
    # Experimental hardware-swept NVLED (listv on ch B, co-triggered ch A)
    # ------------------------------------------------------------------

    def configure_experimental_sweep(
        self,
        v_list,
        vled_voltage,
        settle_time_s,
        points_per_voltage,
    ):
        """Prepare buffers + stash the sweep parameters for the scripted
        hardware-paced NVLED sweep.

        The actual sweep is a TSP script (see ``initiate_experimental_sweep``)
        that walks ``v_list`` on ch B and, per voltage, rapid-fires
        ``points_per_voltage`` chA+chB measurements into the instrument
        buffers. Timestamps on both buffers give per-sample absolute time
        (from the instrument's internal clock, which we align to the
        PM400's perf_counter on the host side).

        We use a script rather than the nested trigger model because the
        chA measure stimulus can't easily fire K times per listv step on
        the 2602B without a trigger-blender chain — the TSP ``for`` loop is
        simpler and also produces back-to-back (∼tens of µs apart) chA/chB
        pairs, which is well below the PM400 100 µs sample period.

        Reuses the existing ``configure_channel`` setup for compliance,
        NPLC, and range (caller is expected to have done that already).
        """
        n = len(v_list)
        if n < 1:
            raise ValueError("v_list must contain at least one voltage")
        K = int(points_per_voltage)
        if K < 1:
            raise ValueError("points_per_voltage must be >= 1")
        v_str = "{" + ", ".join(f"{float(v):.6f}" for v in v_list) + "}"

        # Hard-reset VISA I/O state before we push a long TSP payload.
        # Fast Scan's burst uses the same `print('DONE')` / read() pattern
        # and any partial leftover in the output queue would misalign our
        # next query → -420 "Query UNTERMINATED" on the first read.
        try:
            self.inst.clear()
        except Exception:
            pass
        # Drain any lingering error-queue entries so later error checks
        # only report fresh problems.
        try:
            self.inst.write("errorqueue.clear()")
        except Exception:
            pass

        # Clear buffers, enable timestamps + appendmode. Stash the script
        # parameters as Lua globals so initiate() is a fixed-size message.
        # Statements MUST be separated by newlines — the 2602B's TSP
        # parser emits -285 "Syntax error" when a table constructor is
        # on the same line as subsequent assignments with only spaces.
        tsp = "\n".join([
            "smua.nvbuffer1.clear()",
            "smub.nvbuffer1.clear()",
            "smub.nvbuffer2.clear()",
            "smua.nvbuffer1.appendmode = 1",
            "smub.nvbuffer1.appendmode = 1",
            "smub.nvbuffer2.appendmode = 1",
            "smua.nvbuffer1.collecttimestamps = 1",
            "smub.nvbuffer1.collecttimestamps = 1",
            f"smua.source.levelv = {float(vled_voltage)}",
            f"_exp_v = {v_str}",
            f"_exp_settle = {float(settle_time_s)}",
            f"_exp_k = {K}",
        ])
        self.inst.write(tsp)
        # Flush TSP queue before caller fires the sweep, and verify the
        # globals survived (nil here == the setup write silently failed).
        resp = self.inst.query(
            "print('PREP', type(_exp_v), _exp_settle, _exp_k)"
        ).strip()
        if "table" not in resp:
            errs = self.check_error_queue()
            raise RuntimeError(
                f"Experimental sweep setup failed — expected 'PREP table ...', "
                f"got '{resp}'. Keithley errors: {errs}"
            )
        return n, K

    def initiate_experimental_sweep(self):
        """Fire-and-forget: start the scripted sweep on the instrument.

        Wrapped in ``pcall`` so that any runtime failure (nil global,
        buffer overflow, invalid source value, ...) is captured and
        returned as ``EXP_ERR:<message>`` — otherwise a TSP runtime error
        just leaves the output queue empty and the host sees the generic
        -420 "Query UNTERMINATED" when it tries to join.
        """
        # MUST be wrapped in loadandrunscript/endscript — the 2602B
        # evaluates each incoming line as a standalone Lua chunk in its
        # default interactive mode, so multi-line control flow
        # (do/end, if/end, function/end) gets shredded and bare lines
        # like `smub.measure.iv(...)` run out of context. The
        # loadandrunscript ... endscript markers switch the parser into
        # "accumulate this whole block as one chunk, then run" mode.
        script = "\n".join([
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
        ])
        # Hard-clear any stale output sitting in the VISA queue from
        # prior phases — a leftover `DONE` from Fast Scan would make our
        # join() return instantly and swallow the real EXP_DONE.
        try:
            self.inst.clear()
        except Exception:
            pass
        self.inst.write(script)

    def join_experimental_sweep(self, timeout_s):
        """Block until the script prints ``EXP_DONE``.

        Raises ``RuntimeError`` with the Keithley error queue content if
        the script returned ``EXP_ERR:...`` or the read timed out.
        """
        prev = self.inst.timeout
        self.inst.timeout = int(timeout_s * 1000) + 5000
        try:
            try:
                resp = self.inst.read().strip()
            except Exception as read_err:
                errs = self.check_error_queue()
                raise RuntimeError(
                    f"Experimental sweep read timed out ({read_err}). "
                    f"Keithley error queue: {errs}"
                )
        finally:
            self.inst.timeout = prev
        if resp == "EXP_DONE":
            return
        errs = self.check_error_queue()
        try:
            self.inst.clear()
        except Exception:
            pass
        raise RuntimeError(
            f"Experimental sweep failed: response='{resp[:200]}', "
            f"Keithley errors={errs}"
        )

    def read_experimental_sweep_buffers(self):
        """Read (ia, ib, vb, ta, tb) from the experimental-sweep buffers.

        ``ta`` / ``tb`` are the chA / chB buffer timestamps in the
        instrument's internal clock (referenced to the first sample =
        0.0s). All lists have the same length on success; empty on
        failure.
        """
        try:
            n_a = int(float(self.inst.query("print(smua.nvbuffer1.n)")))
            n_b = int(float(self.inst.query("print(smub.nvbuffer1.n)")))
            n_v = int(float(self.inst.query("print(smub.nvbuffer2.n)")))
            n = min(n_a, n_b, n_v)
            if n == 0:
                return [], [], [], [], []
            ia_str = self.inst.query(f"printbuffer(1, {n}, smua.nvbuffer1)")
            ib_str = self.inst.query(f"printbuffer(1, {n}, smub.nvbuffer1)")
            vb_str = self.inst.query(f"printbuffer(1, {n}, smub.nvbuffer2)")
            ta_str = self.inst.query(
                f"printbuffer(1, {n}, smua.nvbuffer1.timestamps)"
            )
            tb_str = self.inst.query(
                f"printbuffer(1, {n}, smub.nvbuffer1.timestamps)"
            )
            return (
                [float(x) for x in ia_str.split(",")],
                [float(x) for x in ib_str.split(",")],
                [float(x) for x in vb_str.split(",")],
                [float(x) for x in ta_str.split(",")],
                [float(x) for x in tb_str.split(",")],
            )
        except Exception as e:
            print(f"read_experimental_sweep_buffers error: {e}")
            return [], [], [], [], []

    def cleanup_experimental_sweep(self):
        """Drop the stashed globals. No trigger-model state to reset —
        the script path doesn't touch trigger.source/measure.action."""
        try:
            self.inst.write("_exp_v = nil _exp_settle = nil _exp_k = nil")
        except Exception:
            pass

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
