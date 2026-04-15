"""Thorlabs PM400 power meter driver.

Extracted from the pre-refactor `instrumentlib.py`. The only behavioural
additions over the original are:

  * Internal `_mode` tracking (`"idle"`, `"scalar"`, `"continuous"`,
    `"array"`). The PM400 has three mutually incompatible measurement
    modes; mixing them silently corrupts the capture (e.g. calling
    `measure()` while an array capture is armed aborts the capture and
    returns whatever's in the FETC register). The mode tracker lets
    callers assert intent at runtime.

  * `configure_scalar()` — explicit transition out of array mode back
    to a plain single-shot configuration. Used by the transient loop
    to take a "real" dark reference between array captures without
    leaving the PM400 stuck in `CONF:ARR`.

  * `check_error_queue()` — drains `SYST:ERR?` and returns the list of
    error strings. Empty list means the queue was clean.
"""

from __future__ import annotations

import struct
import time

import pyvisa


class PM400:
    """Thorlabs PM400 controller (SCPI)."""

    SETTLE_TIME = 0.1
    MODE_IDLE = "idle"
    MODE_SCALAR = "scalar"
    MODE_CONTINUOUS = "continuous"
    MODE_ARRAY = "array"

    def __init__(self, resource_id):
        self.rm = pyvisa.ResourceManager()
        self.inst = None
        self._mode = self.MODE_IDLE
        self._arr_n_samples = 0
        self._arr_delta_t_us = 100

        try:
            self.inst = self.rm.open_resource(resource_id)
            self.inst.timeout = 5000
            self.inst.read_termination = "\n"
            self.inst.write_termination = "\n"

            idn = self.inst.query("*IDN?").strip()
            print(f"Connected to Device: {idn}")
            if "PM400" not in idn:
                print("WARNING: Device IDN does not match expected 'PM400'.")

            self.sensor_idn = self.inst.query("SYST:SENS:IDN?").strip()
            print(f"Connected Sensor: {self.sensor_idn}")
            self.inst.write("*CLS")

        except pyvisa.VisaIOError as e:
            print(f"Error connecting to {resource_id}: {e}")
            raise

    def close(self):
        if self.inst:
            try:
                self.inst.close()
            except Exception:
                pass
        if self.rm:
            try:
                self.rm.close()
            except Exception:
                pass
        print("Connection closed.")

    def _write_and_wait(self, cmd):
        self.inst.write(cmd)
        time.sleep(self.SETTLE_TIME)

    # ==========================================
    # Configuration
    # ==========================================

    def set_wavelength(self, wavelength_nm):
        self._write_and_wait(f"SENS:CORR:WAV {wavelength_nm}")

    def set_auto_range(self, enable: bool):
        state = "1" if enable else "0"
        self._write_and_wait(f"SENS:POW:RANG:AUTO {state}")

    def set_power_unit(self, unit="W"):
        if unit.upper() not in ["W", "DBM"]:
            raise ValueError("Unit must be 'W' or 'DBM'")
        self._write_and_wait(f"SENS:POW:UNIT {unit.upper()}")

    def set_averaging(self, count):
        self._write_and_wait(f"SENS:AVER {int(count)}")

    def set_range(self, upper_limit):
        self._write_and_wait(f"SENS:POW:RANG {upper_limit}")

    def zero_device(self):
        print("Zeroing device... (Ensure sensor is covered)")
        self.inst.write("SENS:CORR:COLL:ZERO:INIT")
        while True:
            status = int(self.inst.query("STAT:OPER:COND?"))
            if not (status & 128):
                break
            time.sleep(0.2)
        print("Zeroing complete.")

    def configure_scalar(self, wavelength=None, range_W=None, averaging=None):
        """Put the meter into scalar (single-shot) mode.

        Issues CONF:POW which cancels any pending array capture and leaves
        the meter ready for `measure()` calls. Optionally re-applies
        wavelength/range/averaging in the same call for convenience.
        """
        self.inst.write("ABOR")
        time.sleep(0.02)
        self.inst.write("CONF:POW")
        time.sleep(0.02)
        if wavelength is not None:
            self.set_wavelength(wavelength)
        if range_W is not None:
            self.set_auto_range(False)
            self.set_range(range_W)
        if averaging is not None:
            self.set_averaging(averaging)
        self._mode = self.MODE_SCALAR

    # ==========================================
    # Measurement
    # ==========================================

    def display_off(self):
        try:
            self._saved_brightness = float(self.inst.query("DISP:BRIG?"))
        except Exception:
            self._saved_brightness = 1.0
        try:
            self.inst.write("DISP:BRIG 0")
        except Exception:
            pass

    def display_on(self):
        brightness = getattr(self, "_saved_brightness", 1.0)
        try:
            self.inst.write(f"DISP:BRIG {brightness}")
        except Exception:
            pass

    def start_continuous(self):
        """Start continuous trigger mode (INIT:CONT). Use fetch_latest()."""
        self.inst.write("INIT:CONT")
        time.sleep(0.05)
        self._mode = self.MODE_CONTINUOUS

    def stop_continuous(self):
        self.inst.write("ABOR")
        time.sleep(0.01)
        self._mode = self.MODE_IDLE

    def fetch_latest(self):
        """Latest power in continuous mode (FETC?)."""
        try:
            return float(self.inst.query("FETC?"))
        except (ValueError, Exception):
            return float("nan")

    def measure(self):
        """Triggered single measurement (ABOR + INIT + FETC).

        Raises if the meter is currently in array mode — calling
        MEAS:POW? would wipe the pending array capture.
        """
        if self._mode == self.MODE_ARRAY:
            raise RuntimeError(
                "PM400.measure() called while in array mode; "
                "call configure_scalar() first."
            )
        try:
            val = self.inst.query("MEAS:POW?")
            return float(val)
        except ValueError:
            return float("nan")

    def get_mode(self):
        """Return the current internal mode label."""
        return self._mode

    # ==========================================
    # Error queue
    # ==========================================

    def check_error_queue(self):
        """Drain SYST:ERR? and return the list of error strings.

        An empty list means the queue was clean at call time.
        """
        errors = []
        try:
            for _ in range(20):  # safety cap
                resp = self.inst.query("SYST:ERR?").strip()
                # PM400 returns "0,"No error"" when empty.
                head = resp.split(",", 1)[0]
                if head in ("0", "+0"):
                    break
                errors.append(resp)
        except Exception:
            pass
        return errors

    # ==========================================
    # Config extraction
    # ==========================================

    def get_config_dict(self):
        config = {}
        try:
            config["meter_idn"] = self.inst.query("*IDN?").strip()
            config["sensor_idn"] = self.inst.query("SYST:SENS:IDN?").strip()
            config["calibration_date"] = self.inst.query("CAL:STR?").strip()
            config["wavelength_nm"] = float(self.inst.query("SENS:CORR:WAV?"))
            config["power_unit"] = self.inst.query("SENS:POW:UNIT?").strip()
            is_auto = int(self.inst.query("SENS:POW:RANG:AUTO?"))
            config["auto_range_enabled"] = bool(is_auto)
            config["current_range_max"] = float(self.inst.query("SENS:POW:RANG?"))
            config["averaging_count"] = int(self.inst.query("SENS:AVER?"))
            config["beam_diameter_mm"] = float(self.inst.query("SENS:CORR:BEAM?"))
            config["attenuation_db"] = float(self.inst.query("SENS:CORR:LOSS?"))
            config["bandwidth_filter"] = self.inst.query("INP:FILT?").strip()
        except Exception as e:
            print(f"Error retrieving config: {e}")
            config["error"] = str(e)
        return config

    # ==========================================
    # Array mode
    # ==========================================

    def abort(self):
        self.inst.write("ABOR")
        time.sleep(0.01)
        self._mode = self.MODE_IDLE

    def configure_array_mode(self, window_ms, delta_t_us=100):
        """Configure an array-mode capture.

        The PM400 always captures into a 10 000-sample internal buffer
        at 100 µs (10 kHz).  ``delta_t_us`` is a decimation factor:
        each output sample averages ``delta_t_us / 100`` internal
        samples.  The hardware constraint is::

            (delta_t_us / 100) * n_samples <= 10000

        so the maximum output sample count decreases with delta_t.
        The capture duration is always 1.0 s regardless of delta_t.
 
        ``window_ms`` is accepted for backward compatibility but only
        used if it would further limit n_samples.
        """
        delta_t_us = max(100, int(round(delta_t_us / 100.0)) * 100)
        decimation = delta_t_us // 100
        max_samples = 10000 // decimation
        n_from_window = int(window_ms / (delta_t_us / 1000.0))
        n_samples = max(1, min(n_from_window, max_samples))
        self.inst.write("ABOR")
        time.sleep(0.05)
        self.inst.write("INP:FILT 0")
        self.inst.write(f"CONF:ARR {n_samples},{delta_t_us}")
        time.sleep(0.05)
        self._arr_n_samples = n_samples
        self._arr_delta_t_us = delta_t_us
        self._mode = self.MODE_ARRAY
        return n_samples, delta_t_us

    def start_array(self):
        """Non-blocking INIT — PM400 begins array capture immediately."""
        self.inst.write("INIT")

    def poll_array_complete(self, timeout_s=5.0):
        """Poll FETC:STAT? until capture is complete."""
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
        """Download array buffer via FETC:ARR? binary protocol."""
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
            try:
                comma_pos = raw.index(b",")
                payload = raw[comma_pos + 1 :].rstrip(b"\r\n")
            except ValueError:
                offset += count
                continue
            n_floats = len(payload) // 4
            if n_floats > 0:
                vals = struct.unpack(f"<{n_floats}f", payload[: n_floats * 4])
                values.extend(vals)
            offset += count
        return values
