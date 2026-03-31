# 16.03.2026
# v3 - Array mode / hardware trigger additions
VERSION = "20260322_v1"
import pyvisa
import time
import struct


class PM400:
    """
    Class to control Thorlabs PM400 Power Meter via SCPI commands.
    """

    # Delay to allow device to settle after writing configuration settings
    SETTLE_TIME = 0.1

    def __init__(self, resource_id):
        """
        Connects to the PM400 and verifies the identity.

        Args:
            resource_id (str): The VISA resource string (e.g., 'USB0::0x1313::...')
        """
        self.rm = pyvisa.ResourceManager()
        self.inst = None

        try:
            self.inst = self.rm.open_resource(resource_id)
            self.inst.timeout = 5000  # Set timeout to 5 seconds
            self.inst.read_termination = "\n"
            self.inst.write_termination = "\n"

            # 1. Check Device Identity (The Meter itself)
            idn = self.inst.query("*IDN?").strip()
            print(f"Connected to Device: {idn}")

            if "PM400" not in idn:
                print("WARNING: Device IDN does not match expected 'PM400'.")

            # 2. Check Sensor Identity (The connected head)
            # The response format you provided: "S171C","1925471","11-MAR-2024",1,2,33
            self.sensor_idn = self.inst.query("SYST:SENS:IDN?").strip()
            print(f"Connected Sensor: {self.sensor_idn}")

            # Optional: Clear status registers on connect
            self.inst.write("*CLS")

        except pyvisa.VisaIOError as e:
            print(f"Error connecting to {resource_id}: {e}")
            raise

    def close(self):
        """Closes the connection to the device."""
        if self.inst:
            try:
                self.inst.close()
            except:
                pass
        if self.rm:
            try:
                self.rm.close()
            except:
                pass
        print("Connection closed.")

    def _write_and_wait(self, cmd):
        """Helper to write a command and wait for the device to process/settle."""
        self.inst.write(cmd)
        time.sleep(self.SETTLE_TIME)

    # ==========================================
    # Configuration Methods
    # ==========================================

    def set_wavelength(self, wavelength_nm):
        """Sets the wavelength correction in nm."""
        self._write_and_wait(f"SENS:CORR:WAV {wavelength_nm}")

    def set_auto_range(self, enable: bool):
        """Enables (True) or Disables (False) Auto-Ranging."""
        state = "1" if enable else "0"
        self._write_and_wait(f"SENS:POW:RANG:AUTO {state}")

    def set_power_unit(self, unit="W"):
        """Sets the power unit. Options: 'W' (Watts) or 'DBM'."""
        if unit.upper() not in ["W", "DBM"]:
            raise ValueError("Unit must be 'W' or 'DBM'")
        self._write_and_wait(f"SENS:POW:UNIT {unit.upper()}")

    def set_averaging(self, count):
        """
        Sets the averaging rate (1 to 1000).
        Rate = 1000 / count (Hz).
        """
        self._write_and_wait(f"SENS:AVER {int(count)}")

    def set_range(self, upper_limit):
        """
        Sets a manual range (automatically disables auto-range).
        Args:
            upper_limit (float): Expected power in Watts.
        """
        self._write_and_wait(f"SENS:POW:RANG {upper_limit}")

    def zero_device(self):
        """
        Performs the zeroing procedure.
        Note: The sensor must be covered/dark before calling this.
        This function blocks until zeroing is complete.
        """
        print("Zeroing device... (Ensure sensor is covered)")
        self.inst.write("SENS:CORR:COLL:ZERO:INIT")

        # Poll the status until zeroing is done
        while True:
            # Check status bit 7 (Zeroing running) in Operation Condition Register
            status = int(self.inst.query("STAT:OPER:COND?"))
            if not (status & 128):  # Bit 7 is 128
                break
            time.sleep(0.2)
        print("Zeroing complete.")

    # ==========================================
    # Measurement Methods
    # ==========================================

    def display_off(self):
        """Set display brightness to 0 to reduce instrument CPU load (DISP:BRIG 0)."""
        try:
            self._saved_brightness = float(self.inst.query("DISP:BRIG?"))
        except Exception:
            self._saved_brightness = 1.0
        try:
            self.inst.write("DISP:BRIG 0")
        except Exception:
            pass

    def display_on(self):
        """Restore display brightness saved by display_off()."""
        brightness = getattr(self, "_saved_brightness", 1.0)
        try:
            self.inst.write(f"DISP:BRIG {brightness}")
        except Exception:
            pass

    def start_continuous(self):
        """Start continuous trigger mode (INIT:CONT).
        The PM400 measures at its configured rate autonomously.
        Use fetch_latest() to read the most recent result without ABOR+INIT overhead."""
        self.inst.write("INIT:CONT")
        time.sleep(0.05)   # allow first measurement to complete

    def stop_continuous(self):
        """Stop continuous trigger mode."""
        self.inst.write("ABOR")
        time.sleep(0.01)

    def fetch_latest(self):
        """Return the latest power measurement in continuous mode (FETC?).
        Faster than measure() because it skips ABOR+INIT. Call start_continuous() first."""
        try:
            return float(self.inst.query("FETC?"))
        except (ValueError, Exception):
            return float("nan")

    def measure(self):
        """
        Triggered single measurement (MEAS:POW? = ABOR + INIT + FETC).
        Stops any active INIT:CONT. Use fetch_latest() in continuous mode.
        Returns:
            float: Power in the currently selected unit (W or dBm).
        """
        try:
            val = self.inst.query("MEAS:POW?")
            return float(val)
        except ValueError:
            return float("nan")

    # ==========================================
    # Config Extraction
    # ==========================================

    def get_config_dict(self):
        """
        Queries all relevant settings from the device and returns them as a dictionary.
        Useful for saving measurement metadata.
        """
        config = {}

        try:
            # Static Device Info
            config["meter_idn"] = self.inst.query("*IDN?").strip()
            config["sensor_idn"] = self.inst.query("SYST:SENS:IDN?").strip()
            config["calibration_date"] = self.inst.query("CAL:STR?").strip()

            # Wavelength
            config["wavelength_nm"] = float(self.inst.query("SENS:CORR:WAV?"))

            # Power Settings
            config["power_unit"] = self.inst.query("SENS:POW:UNIT?").strip()

            # Ranging
            is_auto = int(self.inst.query("SENS:POW:RANG:AUTO?"))
            config["auto_range_enabled"] = bool(is_auto)

            # The current upper range limit (in Watts usually)
            config["current_range_max"] = float(self.inst.query("SENS:POW:RANG?"))

            # Averaging
            config["averaging_count"] = int(self.inst.query("SENS:AVER?"))

            # Beam Diameter (relevant for power density)
            config["beam_diameter_mm"] = float(self.inst.query("SENS:CORR:BEAM?"))

            # Attenuation (if set)
            config["attenuation_db"] = float(self.inst.query("SENS:CORR:LOSS?"))

            # Bandwidth filter
            config["bandwidth_filter"] = self.inst.query("INP:FILT?").strip()

        except Exception as e:
            print(f"Error retrieving config: {e}")
            config["error"] = str(e)

        return config

    # ==========================================
    # Array Mode Methods
    # ==========================================

    def abort(self):
        """Aborts any ongoing measurement."""
        self.inst.write("ABOR")
        time.sleep(0.01)

    def configure_array_mode(self, window_ms, delta_t_us=100):
        """
        Prepares PM400 for array mode capture.
        - delta_t_us: sampling interval in µs (must be multiple of 100, min 100)
        - window_ms: desired capture window in ms
        Returns (n_samples, delta_t_us) actually configured.
        """
        delta_t_us = max(100, int(round(delta_t_us / 100.0)) * 100)
        n_samples = int(window_ms / (delta_t_us / 1000.0))
        n_samples = max(1, min(n_samples, 10000))
        self.inst.write("ABOR")
        time.sleep(0.05)
        self.inst.write("SENS:POW:RANG:AUTO 0")
        self.inst.write("INP:FILT 0")
        self.inst.write(f"CONF:ARR {n_samples},{delta_t_us}")
        time.sleep(0.05)
        self._arr_n_samples = n_samples
        self._arr_delta_t_us = delta_t_us
        return n_samples, delta_t_us

    def start_array(self):
        """Non-blocking: fires INIT so PM400 starts capturing array immediately."""
        self.inst.write("INIT")

    def poll_array_complete(self, timeout_s=5.0):
        """Polls FETC:STAT? until capture is complete. Returns True on success."""
        deadline = time.perf_counter() + timeout_s
        while time.perf_counter() < deadline:
            try:
                state = int(self.inst.query("FETC:STAT?").strip())
                if state == 1:
                    return True
            except Exception:
                pass
            time.sleep(0.005)
        return False

    def fetch_array(self, n_samples, start_offset=0):
        """
        Downloads array buffer using FETC:ARR? binary protocol.
        PM400 response format: "<count>,<binary_data>" where binary_data is
        little-endian single-precision floats (4 bytes each), up to 40 per query.
        start_offset: first sample index to fetch (default 0 = beginning of buffer).
        Returns list of float values.
        """
        values = []
        offset = 0
        chunk_size = 40
        old_term = self.inst.read_termination
        while offset < n_samples:
            count = min(chunk_size, n_samples - offset)
            self.inst.read_termination = None
            self.inst.write(f"FETC:ARR? {start_offset + offset},{count}")
            raw = self.inst.read_raw()
            self.inst.read_termination = old_term
            # Format: "<n>,<binary_LE_float32>" — find comma, skip ASCII count prefix
            try:
                comma_pos = raw.index(b',')
                payload = raw[comma_pos + 1:].rstrip(b'\r\n')
            except ValueError:
                # No comma: response is "0" (no data yet) or unexpected format
                offset += count
                continue
            n_floats = len(payload) // 4
            if n_floats > 0:
                vals = struct.unpack(f"<{n_floats}f", payload[:n_floats * 4])
                values.extend(vals)
            offset += count
        return values


# =============================================================================
# Keithley 2400 (Legacy SCPI Style)
# =============================================================================
class Keithley2400:
    """
    Class to control Keithley 2400 SourceMeter via SCPI commands.
    Mode: Source Voltage, Measure Current.
    """

    SETTLE_TIME = 0.1

    def __init__(self, resource_id):
        self.rm = pyvisa.ResourceManager()
        try:
            self.inst = self.rm.open_resource(resource_id)
            self.inst.timeout = 5000
            self.inst.read_termination = "\n"
            self.inst.write_termination = "\n"

            # Identity Check
            idn = self.inst.query("*IDN?").strip()
            print(f"Connected to K2400: {idn}")
            if "MODEL 24" not in idn:
                print("WARNING: Device IDN does not look like a Keithley 24xx.")

            # Reset to defaults and clear status
            self.inst.write("*RST")
            self.inst.write("*CLS")
            time.sleep(3.0)  # Reset takes time

            # Basic setup for Source V, Measure I
            self._write_and_wait(":ROUT:TERM FRON")
            self._write_and_wait(":SOUR:FUNC VOLT")
            self._write_and_wait(":SENS:FUNC 'CURR'")
            self._write_and_wait(
                ":FORM:ELEM CURR"
            )  # Only return Current string to make parsing easy

        except pyvisa.VisaIOError as e:
            print(f"Error connecting to K2400 ({resource_id}): {e}")
            raise

    def close(self):
        if self.inst:
            try:
                self.inst.write(":OUTP OFF")  # Safety: Turn off output
                self.inst.close()
            except:
                pass
        print("K2400 Connection closed.")

    def _write_and_wait(self, cmd):
        self.inst.write(cmd)
        time.sleep(self.SETTLE_TIME)

    def configure_source(self, compliance_current=0.1, nplc=1.0):
        """
        Sets compliance and integration speed.
        """
        # Set Current Compliance
        self._write_and_wait(f":SENS:CURR:PROT {compliance_current}")
        # Set Speed (NPLC: 0.01=Fast, 1=Normal, 10=Slow/HighAcc)
        self._write_and_wait(f":SENS:CURR:NPLC {nplc}")
        # Fixed range for measurement
        self._write_and_wait(":SENS:CURR:RANG 1E-3")

    def set_voltage(self, voltage):
        """Sets the source voltage level."""
        self._write_and_wait(f":SOUR:VOLT:LEV {voltage}")

    def enable_output(self, enable: bool):
        state = "ON" if enable else "OFF"
        self._write_and_wait(f"OUTPUT {state}")

    def measure_current(self):
        """Triggers and reads current."""
        try:
            # :READ? triggers a measurement and returns formatted data
            val = self.inst.query(":READ?")
            return float(val)
        except Exception as e:
            print(f"K2400 Measure Error: {e}")
            return float("nan")

    def get_config_dict(self):
        """Queries current settings for metadata storage."""
        config = {}
        try:
            config["idn"] = self.inst.query("*IDN?").strip()
            config["source_mode"] = self.inst.query(":SOUR:FUNC?").strip()
            config["voltage_setpoint"] = float(self.inst.query(":SOUR:VOLT:LEV?"))
            config["current_compliance"] = float(self.inst.query(":SENS:CURR:PROT?"))
            config["nplc"] = float(self.inst.query(":SENS:CURR:NPLC?"))
            config["output_state"] = int(self.inst.query(":OUTP?"))
            config["range_auto"] = int(self.inst.query(":SENS:CURR:RANG:AUTO?"))
            config["autozero"] = self.inst.query(":SYST:AZER:STAT?").strip()
        except Exception as e:
            config["error"] = str(e)
        return config


# =============================================================================
# Keithley 2602B (Dual-Channel TSP)
# =============================================================================
class Keithley2602B:
    """
    Class to control Keithley 2602B (Dual Channel) via TSP commands.
    Assumes Channel A (smua) is VLED and Channel B (smub) is NVLED.
    """

    SETTLE_TIME = 0.1

    def __init__(self, resource_id):
        self.rm = pyvisa.ResourceManager()
        try:
            self.inst = self.rm.open_resource(resource_id)
            self.inst.clear()
            time.sleep(0.5)
            self.inst.timeout = 10000  # Extended timeout for buffered readouts
            self.inst.read_termination = "\n"
            self.inst.write_termination = "\n"

            idn = self.inst.query("*IDN?").strip()
            print(f"Connected to Dual-Channel K2602B: {idn}")
            if "26" not in idn:
                print("WARNING: Device IDN does not look like a Keithley 26xx.")

            # Reset both channels to safe defaults
            self.inst.write("smua.reset() smub.reset()")
            time.sleep(1.0)

        except pyvisa.VisaIOError as e:
            print(f"Error connecting to K2602B ({resource_id}): {e}")
            raise

    def display_off(self):
        """Switch both display rows to static text in one TSP write.
        2602B settext format: first 20 chars → top row (smua), next 20 → bottom row (smub).
        Single write avoids the visible blank-frame flash between two separate writes."""
        line = "Measuring SMILE"   # exactly 20 chars
        self.inst.write(f'display.clear() display.setcursor(1,1) display.settext("Measuring Smile") display.setcursor(2,1) display.settext("Powered by Georg and Max")')

    def display_on(self):
        """Restore normal dual-channel display."""
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

    def configure_channel(self, channel, compliance_current=0.1, nplc=1.0,
                          high_c=False, zero_delays=False, range_i=1e-3):
        """
        Sets compliance, speed, and High Capacitance mode for a specific channel.
        Args:
            channel (str): 'a' or 'b'
            zero_delays (bool): When True, sets source/measure delays to 0 and
                                disables autozero for fastest throughput.
            range_i (float): Fixed current measurement range in Amps.
        """
        ch = f"smu{channel.lower()}"

        self._write_and_wait(f"{ch}.source.func = {ch}.OUTPUT_DCVOLTS")
        self._write_and_wait(f"{ch}.source.limiti = {compliance_current}")
        self._write_and_wait(f"{ch}.measure.nplc = {nplc}")
        self._write_and_wait(f"{ch}.measure.autorangei = {ch}.AUTORANGE_OFF")
        self._write_and_wait(f"{ch}.measure.rangei = {range_i}")

        # Enable High Capacitance Mode if needed (helps stabilize capacitive loads)
        highc_state = f"{ch}.ENABLE" if high_c else f"{ch}.DISABLE"
        self._write_and_wait(f"{ch}.source.highc = {highc_state}")

        if zero_delays:
            self._write_and_wait(f"{ch}.source.delay = 0")
            self._write_and_wait(f"{ch}.measure.delay = 0")
            self._write_and_wait(f"{ch}.measure.autozero = {ch}.AUTOZERO_OFF")

    def set_voltage(self, channel, voltage):
        """Sets the source voltage level on the specified channel."""
        ch = f"smu{channel.lower()}"
        self._write_and_wait(f"{ch}.source.levelv = {voltage}")

    def enable_output(self, channel, enable: bool):
        """Turns the specified channel output ON/OFF."""
        ch = f"smu{channel.lower()}"
        state = f"{ch}.OUTPUT_ON" if enable else f"{ch}.OUTPUT_OFF"
        self._write_and_wait(f"{ch}.source.output = {state}")

    def measure_instant(self):
        """Single immediate measurement on both channels, no buffer. Returns (ia_A, ib_A)."""
        resp = self.inst.query("print(smua.measure.i(), smub.measure.i())")
        parts = resp.strip().split()
        ia = float(parts[0]) if len(parts) > 0 else float("nan")
        ib = float(parts[1]) if len(parts) > 1 else float("nan")
        return ia, ib

    # --- BUFFERED MEASUREMENT METHODS ---
    def setup_buffers(self, timestamps=False):
        """Clears buffers, sets them to append mode, optionally enables timestamps."""
        self._write_and_wait("smua.nvbuffer1.clear() smub.nvbuffer1.clear()")
        self._write_and_wait("smua.nvbuffer1.appendmode = 1")
        self._write_and_wait("smub.nvbuffer1.appendmode = 1")
        if timestamps:
            self._write_and_wait("smua.nvbuffer1.collecttimestamps = 1")
            self._write_and_wait("smub.nvbuffer1.collecttimestamps = 1")
        # Sync: flush the TSP queue so all writes above have executed before
        # the first trigger arm. On fresh power-on the TSP JIT is still warming
        # up and fire-and-forget writes can lag behind trigger.initiate().
        self.inst.query("print('SETUP_DONE')")

    def clear_buffers(self):
        """Clears buffers and sets them to append mode (legacy alias)."""
        self.setup_buffers(timestamps=False)

    def clear_buffers_only(self):
        """Clears buffer data only — does not change collecttimestamps or other settings."""
        self.inst.write("smua.nvbuffer1.clear() smub.nvbuffer1.clear()")

    def measure_both_to_buffer(self):
        """
        Triggers a measurement on both channels that immediately stores into the internal buffer.
        This command executes very quickly without blocking Python with string parsing.
        """
        self.inst.write("smua.measure.i(smua.nvbuffer1) smub.measure.i(smub.nvbuffer1)")

    def run_tsp_and_wait(self, tsp_code):
        """Executes a TSP script and blocks until it completes. Raises on queue desync."""
        self.inst.write(tsp_code)
        res = self.inst.query("print('DONE')").strip()
        if res != "DONE":
            self.inst.clear()
            raise RuntimeError(f"TSP queue desync: expected 'DONE', got '{res[:40]}'")

    def configure_hardware_trigger(self, count):
        """
        Sets up the hardware trigger model on both channels to capture `count`
        measurements each. Call setup_buffers() first.
        """
        self.inst.write("smua.trigger.measure.i(smua.nvbuffer1) smub.trigger.measure.i(smub.nvbuffer1)")
        self.inst.write(f"smua.trigger.count = {count}")
        self.inst.write(f"smub.trigger.count = {count}")
        self.inst.write("smua.trigger.measure.action = smua.ENABLE smub.trigger.measure.action = smub.ENABLE")

    def start_hardware_trigger(self):
        """
        Non-blocking: fires trigger.initiate() on both channels.
        The Keithley then measures autonomously in the background.
        """
        self.inst.write("smua.trigger.initiate() smub.trigger.initiate()")

    def measure_burst(self, n):
        """Take n current measurements on both channels in a single TSP round-trip.
        Results stored in nvbuffer1 (appendmode must be 1, call setup_buffers first).
        Blocks until all n measurements are complete."""
        self.inst.query(
            f"for i=1,{n} do "
            f"smua.measure.i(smua.nvbuffer1) "
            f"smub.measure.i(smub.nvbuffer1) "
            f"end print('DONE')"
        )

    def measure_burst_fire(self, n):
        """Fire n-measurement TSP burst without waiting for completion.
        The Keithley executes the loop autonomously; call measure_burst_join()
        to collect the DONE response before reading buffers."""
        self.inst.write(
            f"for i=1,{n} do "
            f"smua.measure.i(smua.nvbuffer1) "
            f"smub.measure.i(smub.nvbuffer1) "
            f"end print('DONE')"
        )

    def measure_burst_join(self):
        """Wait for a burst started by measure_burst_fire() to complete.
        Reads and verifies the DONE response from the TSP queue."""
        resp = self.inst.read().strip()
        if resp != "DONE":
            self.inst.clear()
            raise RuntimeError(f"TSP burst sync error: expected 'DONE', got '{resp[:40]}'")

    def read_buffers(self, n=None):
        """Read current readings from both channel nvbuffer1 buffers.
        n: if given, read at most n entries (uses min(n, actual_count)).
        Returns (ia_list, ib_list) of floats."""
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
            return [float(x) for x in ia_str.split(",")], [float(x) for x in ib_str.split(",")]
        except Exception as e:
            print(f"read_buffers error: {e}")
            return [], []

    def abort_trigger(self):
        """Aborts any ongoing hardware trigger measurement and disables trigger actions."""
        self.inst.write("smua.abort() smub.abort()")
        self.inst.write("smua.trigger.measure.action = smua.DISABLE smub.trigger.measure.action = smub.DISABLE")
        time.sleep(0.05)

    def read_buffer_with_timestamps(self):
        """
        Downloads all stored current data + hardware timestamps from both channels.
        Returns: (vled_current_list, nvled_current_list, vled_timestamps_list, nvled_timestamps_list)
        Timestamps are in the Keithley's internal time reference (seconds since reset).
        """
        try:
            n_a = int(float(self.inst.query("print(smua.nvbuffer1.n)")))
            n_b = int(float(self.inst.query("print(smub.nvbuffer1.n)")))
            n = min(n_a, n_b)
            if n == 0:
                return [], [], [], []
            self.inst.write("format.data = format.ASCII")
            ia_str = self.inst.query(f"printbuffer(1, {n}, smua.nvbuffer1)")
            ib_str = self.inst.query(f"printbuffer(1, {n}, smub.nvbuffer1)")
            ta_str = self.inst.query(f"printbuffer(1, {n}, smua.nvbuffer1.timestamps)")
            tb_str = self.inst.query(f"printbuffer(1, {n}, smub.nvbuffer1.timestamps)")
            ia_vals = [float(x) for x in ia_str.split(",")]
            ib_vals = [float(x) for x in ib_str.split(",")]
            ta_vals = [float(x) for x in ta_str.split(",")]
            tb_vals = [float(x) for x in tb_str.split(",")]
            return ia_vals, ib_vals, ta_vals, tb_vals
        except Exception as e:
            print(f"Buffer read with timestamps error: {e}")
            return [], [], [], []

    def get_config_dict(self):
        config = {}
        try:
            config["idn"] = self.inst.query("*IDN?").strip()
            config["chA_voltage"] = float(self.inst.query("print(smua.source.levelv)"))
            config["chA_comp"] = float(self.inst.query("print(smua.source.limiti)"))
            config["chA_highc"] = int(float(self.inst.query("print(smua.source.highc)")))
            config["chA_nplc"] = float(self.inst.query("print(smua.measure.nplc)"))
            config["chA_output"] = int(float(self.inst.query("print(smua.source.output)")))

            config["chB_voltage"] = float(self.inst.query("print(smub.source.levelv)"))
            config["chB_comp"] = float(self.inst.query("print(smub.source.limiti)"))
            config["chB_highc"] = int(float(self.inst.query("print(smub.source.highc)")))
            config["chB_nplc"] = float(self.inst.query("print(smub.measure.nplc)"))
            config["chB_output"] = int(float(self.inst.query("print(smub.source.output)")))
        except Exception as e:
            config["error"] = str(e)
        return config


# =============================================================================
# Keithley 2611 (TSP/Lua Style)
# =============================================================================
class Keithley2611:
    """
    Class to control Keithley 2600 Series (2611) via TSP commands.
    Assumes Channel A (smua).
    """

    SETTLE_TIME = 0.1

    def __init__(self, resource_id):
        self.rm = pyvisa.ResourceManager()
        try:
            self.inst = self.rm.open_resource(resource_id)
            self.inst.timeout = 5000
            self.inst.read_termination = "\n"
            # TSP devices usually don't strictly require write termination change
            # but usually work best with \n

            # Identity Check
            idn = self.inst.query("*IDN?").strip()
            print(f"Connected to K2611: {idn}")
            if "26" not in idn:
                print("WARNING: Device IDN does not look like a Keithley 26xx.")

            # Reset smua to defaults
            self.inst.write("smua.reset()")
            time.sleep(1.0)

            # Set to Source Voltage, Measure Current Mode
            self._write_and_wait("smua.source.func = smua.OUTPUT_DCVOLTS")

        except pyvisa.VisaIOError as e:
            print(f"Error connecting to K2611 ({resource_id}): {e}")
            raise

    def close(self):
        if self.inst:
            self._write_and_wait("smua.source.output = smua.OUTPUT_OFF")
            self.inst.close()
        print("K2611 Connection closed.")

    def _write_and_wait(self, cmd):
        self.inst.write(cmd)
        time.sleep(self.SETTLE_TIME)

    def configure_source(self, compliance_current=0.1, nplc=1.0):
        """Sets compliance and measurement speed."""
        self._write_and_wait(f"smua.source.limiti = {compliance_current}")
        self._write_and_wait(f"smua.measure.nplc = {nplc}")
        self._write_and_wait("smua.measure.autorangei = smua.AUTORANGE_OFF")
        self._write_and_wait("smua.measure.rangei = 0.1")

    def set_voltage(self, voltage):
        """Sets the source voltage level."""
        self._write_and_wait(f"smua.source.levelv = {voltage}")

    def set_voltage(self, voltage, channel="a"):
        """Sets the source voltage level."""
        self._write_and_wait(f"smu{channel}.source.levelv = {voltage}")

    def enable_output(self, enable: bool):
        state = "smua.OUTPUT_ON" if enable else "smua.OUTPUT_OFF"
        self._write_and_wait(f"smua.source.output = {state}")

    def measure_current(self):
        """
        Measurements in TSP require an explicit 'print' command to
        send data back to the PC.
        """
        try:
            val = self.inst.query("print(smua.measure.i())")
            return float(val)
        except Exception as e:
            print(f"K2611 Measure Error: {e}")
            return float("nan")

    def get_config_dict(self):
        """Queries settings using Lua print statements."""
        config = {}
        try:
            config["idn"] = self.inst.query("*IDN?").strip()
            # Note: 0=Volts, 1=Amps in TSP enums, logic below assumes DCVOLTS was set
            config["source_func_enum"] = int(
                float(self.inst.query("print(smua.source.func)"))
            )
            config["voltage_setpoint"] = float(
                self.inst.query("print(smua.source.levelv)")
            )
            config["current_compliance"] = float(
                self.inst.query("print(smua.source.limiti)")
            )
            config["nplc"] = float(self.inst.query("print(smua.measure.nplc)"))
            config["output_state_enum"] = int(
                float(self.inst.query("print(smua.source.output)"))
            )
            config["autorange_i_enum"] = int(
                float(self.inst.query("print(smua.measure.autorangei)"))
            )
            config["range_i"] = float(self.inst.query("print(smua.measure.rangei)"))
        except Exception as e:
            config["error"] = str(e)
        return config


# ==========================================
# Example Usage
# ==========================================
if __name__ == "__main__":
    # IMPORTANT: Replace with your specific Resource ID found in NI-MAX or Device Manager
    # Example format: "USB0::0x1313::0x8078::P0000000::INSTR"
    RESOURCE_ID = "USB0::0x1313::0x8075::P5006526::INSTR"

    pm = None
    try:
        # Initialize
        pm = PM400(RESOURCE_ID)

        # --- Setup Device ---
        print("Configuring device...")
        pm.set_wavelength(532)  # Set to 532 nm
        pm.set_power_unit("W")  # Measure in Watts
        pm.set_auto_range(True)  # Enable Auto-range
        pm.set_averaging(100)  # Set averaging filter

        # --- Export Configuration ---
        # Get the config dictionary to save with your data later
        measurement_metadata = pm.get_config_dict()
        print("\n--- Measurement Configuration to Save ---")
        for key, value in measurement_metadata.items():
            print(f"{key}: {value}")
        print("-----------------------------------------\n")

        # --- Measure Loop ---
        print("Starting Measurement...")
        for i in range(5):
            power = pm.measure()
            print(f"Reading {i+1}: {power:.4e} W")
            time.sleep(0.2)

    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        if pm:
            pm.close()
    # Replace with your actual Resource IDs
    K2400_ADDR = "GPIB0::24::INSTR"
    K2611_ADDR = "USB0::0x05E6::0x2611::123456::INSTR"

    # --- Test Keithley 2400 ---
    try:
        print("\n--- Testing K2400 ---")
        k24 = Keithley2400(K2400_ADDR)
        k24.configure_source(compliance_current=0.01, nplc=1.0)

        k24.enable_output(True)
        k24.set_voltage(1.5)

        # Save Config
        print("K2400 Config:", k24.get_config_dict())

        # Measure
        curr = k24.measure_current()
        print(f"K2400 Current: {curr:.4e} A")

        k24.close()
    except Exception as e:
        print(f"Skipping K2400 test: {e}")

    # --- Test Keithley 2611 ---
    try:
        print("\n--- Testing K2611 ---")
        k26 = Keithley2611(K2611_ADDR)
        k26.configure_source(compliance_current=0.01, nplc=1.0)

        k26.enable_output(True)
        k26.set_voltage(1.5)

        # Save Config
        print("K2611 Config:", k26.get_config_dict())

        # Measure
        curr = k26.measure_current()
        print(f"K2611 Current: {curr:.4e} A")

        k26.close()
    except Exception as e:
        print(f"Skipping K2611 test: {e}")
