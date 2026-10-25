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
            from smile.utils.display import send_image, set_pixel, set_pixels

            self._send_image = send_image
            self._set_pixel = set_pixel
            self._set_pixels = set_pixels
        else:
            self._send_image = None
            self._set_pixel = None
            self._set_pixels = None

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
    # The four quadrant enables in the frame instruction. With sel_*_pixen = 0 (the
    # default) the frame's own bits are the enable source (smile set-pixen).
    _ENABLES = ("tl_pixen", "tr_pixen", "bl_pixen", "br_pixen")

    def _enable_cfg(self, quad_cfg, enables):
        """Frame config for ``enables``: "own" (None) = the pixel's quadrant only,
        "all" = all four quadrants, "none" = no quadrant."""
        if enables in (None, "own"):
            return quad_cfg
        if enables not in ("all", "none"):
            raise ValueError(f"enables must be own, all or none, not {enables!r}")
        return {k: int(enables == "all") for k in self._ENABLES}

    def set_pixel(self, log_x, log_y, bit_val, enables=None):
        """Turn on a single logical pixel at (log_x, log_y) with intensity bit_val.

        ``enables`` sets the quadrant enables of the frame: None/"own" enables the
        pixel's own quadrant (every caller before 2026-10-25), "all" or "none".

        Returns
        -------
        float
            `time.perf_counter()` taken immediately after the FPGA ACK.
        """
        row, col, quad_cfg = self.map_logical_to_physical(log_x, log_y)
        if self._dev is not None:
            self._set_pixel(self._dev, row, col, bit_val,
                            cfg=self._enable_cfg(quad_cfg, enables))
        return time.perf_counter()

    def set_pixels(self, coords, bit_val):
        """Turn on several logical pixels at once, all in ONE quadrant.

        `coords` is an iterable of (log_x, log_y). The quadrant enable bits
        (`tl_pixen`/`tr_pixen`/...) live in the frame instruction, not in the
        pixel data, so a cluster that straddles a 256 boundary cannot be written
        as one frame. Rather than silently light the mirror image of half the
        cluster, this raises.

        Returns
        -------
        float
            `time.perf_counter()` taken immediately after the FPGA ACK.
        """
        coords = [(int(x), int(y)) for x, y in coords]
        if not coords:
            raise ValueError("coords is empty")
        # map_logical_to_physical falls back to TL for out-of-range input, which is
        # harmless for one pixel and silently collides two cluster members here.
        bad = [c for c in coords if not (0 <= c[0] < 512 and 0 <= c[1] < 512)]
        if bad:
            raise ValueError(f"logical coordinates outside the 512x512 array: {bad[:4]}")
        mapped = [self.map_logical_to_physical(x, y) for x, y in coords]
        quad_cfg = mapped[0][2]
        if any(m[2] != quad_cfg for m in mapped):
            quads = sorted({k for _, _, q in mapped for k, v in q.items() if v})
            raise ValueError(
                f"a cluster must stay inside one quadrant, got {quads}: the pixel-enable "
                f"bits are per frame")
        if self._dev is not None:
            self._set_pixels(self._dev, [(r, c) for r, c, _ in mapped], bit_val,
                             cfg=quad_cfg)
        return time.perf_counter()

    def blank_frame(self, enables=None):
        """Send an all-zero frame (turns every pixel off).

        With no ``enables`` the frame carries the library default, which is ALL FOUR
        quadrants enabled -- what every blank frame before 2026-10-25 sent. "none"
        sends the same zeros with every quadrant disabled.

        Returns
        -------
        float
            `time.perf_counter()` taken immediately after the FPGA ACK.
        """
        if self._dev is not None:
            cfg = None if enables in (None, "all") else self._enable_cfg({}, enables)
            self._send_image(self._dev, [0] * self.BLANK_FRAME_CHUNK_COUNT, cfg)
        return time.perf_counter()

    def close(self):
        if self._dev is not None:
            try:
                self._dev.close()
            except Exception:
                pass
            self._dev = None
