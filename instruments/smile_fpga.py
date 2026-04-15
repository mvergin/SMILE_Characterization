"""SmileFPGA wrapper — thin abstraction over the raw smile library.

Consolidates the logical→physical coordinate mapping into a single
authoritative method, and wraps set_pixel() / blank_frame() so both
operations return the perf_counter() ACK timestamp that the measurement
loop uses for timing analysis.
"""

import time


class SmileFPGA:
    """Wraps a real `smile.core.Smile` device handle.

    Parameters
    ----------
    smile_dev : smile.core.Smile
        Opened device handle. May be None if the FPGA is unavailable,
        in which case `set_pixel` and `blank_frame` are no-ops that
        still return a perf_counter timestamp (useful so callers do
        not need to special-case the no-hardware path).
    """

    BLANK_FRAME_CHUNK_COUNT = 256 * 8  # 2048 chunks

    def __init__(self, smile_dev):
        self._dev = smile_dev
        # Lazy import so the module can be imported without the smile library installed.
        if smile_dev is not None:
            from smile.utils.display import send_image, set_pixel

            self._send_image = send_image
            self._set_pixel = set_pixel
        else:
            self._send_image = None
            self._set_pixel = None

    # ------------------------------------------------------------------
    # Coordinate mapping
    # ------------------------------------------------------------------
    @staticmethod
    def map_logical_to_physical(log_x, log_y):
        """Map logical (log_x, log_y) to (local_row, local_col, quad_cfg).

        The 512x512 logical grid is split into four 256x256 quadrants.
        The TR, BL, BR quadrants are mirrored relative to TL, which is
        why this mapping is not a straight modulo.
        """
        quad_cfg = {"tl_pixen": 0, "tr_pixen": 0, "bl_pixen": 0, "br_pixen": 0}
        if 0 <= log_x < 256 and 0 <= log_y < 256:
            quad_cfg["tl_pixen"] = 1
            return log_y, log_x, quad_cfg
        elif 256 <= log_x < 512 and 0 <= log_y < 256:
            quad_cfg["tr_pixen"] = 1
            return log_y, 255 - (log_x - 256), quad_cfg
        elif 0 <= log_x < 256 and 256 <= log_y < 512:
            quad_cfg["bl_pixen"] = 1
            return 255 - (log_y - 256), log_x, quad_cfg
        elif 256 <= log_x < 512 and 256 <= log_y < 512:
            quad_cfg["br_pixen"] = 1
            return 255 - (log_y - 256), 255 - (log_x - 256), quad_cfg
        # Out-of-range: fall back to TL (matches legacy behaviour).
        quad_cfg["tl_pixen"] = 1
        return log_y, log_x, quad_cfg

    # ------------------------------------------------------------------
    # Actions
    # ------------------------------------------------------------------
    def set_pixel(self, log_x, log_y, bit_val):
        """Turn on a single logical pixel at (log_x, log_y) with intensity bit_val.

        Returns
        -------
        float
            `time.perf_counter()` taken immediately after the FPGA ACK.
        """
        row, col, quad_cfg = self.map_logical_to_physical(log_x, log_y)
        if self._dev is not None:
            self._set_pixel(self._dev, row, col, bit_val, cfg=quad_cfg)
        return time.perf_counter()

    def blank_frame(self):
        """Send an all-zero frame (turns every pixel off).

        Returns
        -------
        float
            `time.perf_counter()` taken immediately after the FPGA ACK.
        """
        if self._dev is not None:
            self._send_image(self._dev, [0] * self.BLANK_FRAME_CHUNK_COUNT)
        return time.perf_counter()

    def close(self):
        if self._dev is not None:
            try:
                self._dev.close()
            except Exception:
                pass
            self._dev = None
