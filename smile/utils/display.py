"""Helpers for displaying images on the SMILE microdisplay.

This module provides utility functions used by the CLI and examples for
generating patterns and streaming data to the FPGA.  Display frames are
interpreted as the top-left 256\u00d7256 quadrant and are mirrored by the
hardware into the remaining three quadrants.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, TYPE_CHECKING
import numpy as np

# Prevent circular imports: only import Smile for type checking
if TYPE_CHECKING:
    from ..core import Smile

logger = logging.getLogger(__name__)

# Default configuration values used when constructing USB_2_DIS instructions.
DEFAULT_CFG: Dict[str, Any] = {
    "roi_fcol": 0,
    "roi_lcol": 7,
    "roi_frow": 0,
    "roi_lrow": None,
    "pwm_dutyc": 0,
    "pwm_ticks": 0,
    "fr_ticks": 0,
    "sel_tl_pixen": 0,
    "sel_tr_pixen": 0,
    "sel_bl_pixen": 0,
    "sel_br_pixen": 0,
    "tl_pixen": 1,
    "tr_pixen": 1,
    "bl_pixen": 1,
    "br_pixen": 1,
    "tl_pwm_en": 0,
    "tr_pwm_en": 0,
    "bl_pwm_en": 0,
    "br_pwm_en": 0,
    "enable_pwm": 0,
}


def row_constructor(value: int) -> List[int]:
    """Create a single display row with all 256 pixels set to `value`."""
    subrow = 0
    for i in range(32):
        subrow |= value << (i * 4)
    return [subrow] * 8


def _pixel_row_constructor(col: int, value: int) -> List[int]:
    """Create a row with only the pixel at `col` set to `value`; all others 0.

    Layout: 8 subwords × 32 pixels × 4 bits = 256 pixels per row.
    Column groups are stored in reverse order (group 0 → subword 7) and nibbles
    are MSB-first within each subword (col%32=0 → nibble 31). This matches the
    addressing used by get_direct_config_and_chunks in the GUI.
    """
    group       = col // 32
    subword_idx = 7 - group
    nibble_pos  = 31 - (col % 32)
    words = [0] * 8
    words[subword_idx] = value << (nibble_pos * 4)
    return words


def create_spot(
    rows: int, cols: int, intensity: int, start_row: int, start_col: int
) -> List[int]:
    """Generate a small illuminated block."""
    if not (0 <= intensity <= 15):
        raise ValueError("Intensity must be between 0 and 15")
    if rows > 256 or cols > 256:
        raise ValueError("Spot size exceeds display size")
    if not (0 <= start_row < 256) or not (0 <= start_col < 256):
        raise ValueError("Start position out of range")
    if start_row + rows > 256 or start_col + cols > 256:
        raise ValueError("Spot exceeds display boundaries")

    spot = [[0] * 256 for _ in range(256)]
    for r in range(start_row, start_row + rows):
        for c in range(start_col, start_col + cols):
            spot[r][c] = intensity
    return [pixel for row in spot for pixel in row]


def send_image(
    dev: Smile, data: List[int], cfg: Optional[Dict[str, Any]] = None
) -> None:
    """Send raw image words to the display."""
    dev.send_instruction(instr=(dev.commands["RESET_DIS"] << 120))

    config = {**DEFAULT_CFG, **(cfg or {})}

    par_or_ser = 0x01  # Force serial interface
    pwm_ticks = int(config["pwm_ticks"])
    pwm_dutyc = int(config["pwm_dutyc"])
    fr_ticks = int(config["fr_ticks"])
    roi_fcol = int(config["roi_fcol"])
    roi_lcol = int(config["roi_lcol"])
    roi_frow = int(config["roi_frow"])
    enable_pwm = int(config["enable_pwm"])
    sel_tl_pixen = int(config["sel_tl_pixen"])
    sel_tr_pixen = int(config["sel_tr_pixen"])
    sel_bl_pixen = int(config["sel_bl_pixen"])
    sel_br_pixen = int(config["sel_br_pixen"])
    tl_pixen = int(config["tl_pixen"])
    tr_pixen = int(config["tr_pixen"])
    bl_pixen = int(config["bl_pixen"])
    br_pixen = int(config["br_pixen"])
    tl_pwm_en = int(config["tl_pwm_en"])
    tr_pwm_en = int(config["tr_pwm_en"])
    bl_pwm_en = int(config["bl_pwm_en"])
    br_pwm_en = int(config["br_pwm_en"])
    roi_lrow = config.get("roi_lrow")
    if roi_lrow is None:
        roi_lrow = (len(data) >> 3) - 1
    roi_lrow = int(roi_lrow)

    instr = (
        dev.commands["USB_2_DIS"] << 120
        | par_or_ser << 119
        | sel_tl_pixen << 78
        | sel_tr_pixen << 77
        | sel_bl_pixen << 76
        | sel_br_pixen << 75
        | tl_pixen << 74
        | tr_pixen << 73
        | bl_pixen << 72
        | br_pixen << 71
        | tl_pwm_en << 70
        | tr_pwm_en << 69
        | bl_pwm_en << 68
        | br_pwm_en << 67
        | enable_pwm << 66
        | roi_fcol << 62
        | roi_lcol << 58
        | roi_frow << 49
        | roi_lrow << 40
        | pwm_dutyc << 30
        | pwm_ticks << 20
        | fr_ticks
    )
    dev.send_instruction(instr, data)


def generate_vstripe_matrix(
    size: int = 256,
    stripe_width: int = 4,
    start: int = 0,
    intensity: int = 15,
) -> List[int]:
    """Return a flattened vertical striped pattern."""
    if start < 0 or start >= stripe_width * 2:
        raise ValueError("start must be within the stripe period")
    if intensity < 0 or intensity > 15:
        raise ValueError("intensity must be between 0 and 15")

    matrix = np.zeros((size, size), dtype=int)
    for col in range(start, size, stripe_width * 2):
        matrix[:, col : col + stripe_width] = intensity

    return [int(val) for val in matrix.flatten().tolist()]


def generate_hstripe_matrix(
    size: int = 256,
    stripe_height: int = 4,
    start: int = 0,
    intensity: int = 15,
) -> List[int]:
    """Return a flattened horizontal striped pattern."""
    if start < 0 or start >= stripe_height * 2:
        raise ValueError("start must be within the stripe period")
    if intensity < 0 or intensity > 15:
        raise ValueError("intensity must be between 0 and 15")

    matrix = np.zeros((size, size), dtype=int)
    for row in range(start, size, stripe_height * 2):
        matrix[row : row + stripe_height, :] = intensity

    return [int(val) for val in matrix.flatten().tolist()]


def set_row(
    dev: Smile,
    row: int,
    value: int,
    default: int = 0,
    cfg: Optional[Dict[str, Any]] = None,
    *,
    _pixel_data: Optional[List[int]] = None,
) -> None:
    """Write a single row using USB_2_DIS.

    _pixel_data: if provided, used as the 8-word row payload instead of
    row_constructor(value). Internal parameter used by set_pixel.
    """
    pixels = _pixel_data if _pixel_data is not None else row_constructor(value)
    
    dev.send_instruction(instr=(dev.commands["RESET_DIS"] << 120))

    config = {**DEFAULT_CFG, **(cfg or {})}

    par_or_ser = 0x01
    pwm_ticks = int(config["pwm_ticks"])
    pwm_dutyc = int(config["pwm_dutyc"])
    fr_ticks = int(config["fr_ticks"])
    roi_fcol = int(config["roi_fcol"])
    roi_lcol = int(config["roi_lcol"])
    roi_frow = int(config["roi_frow"])
    enable_pwm = int(config["enable_pwm"])
    sel_tl_pixen = int(config["sel_tl_pixen"])
    sel_tr_pixen = int(config["sel_tr_pixen"])
    sel_bl_pixen = int(config["sel_bl_pixen"])
    sel_br_pixen = int(config["sel_br_pixen"])
    tl_pixen = int(config["tl_pixen"])
    tr_pixen = int(config["tr_pixen"])
    bl_pixen = int(config["bl_pixen"])
    br_pixen = int(config["br_pixen"])
    tl_pwm_en = int(config["tl_pwm_en"])
    tr_pwm_en = int(config["tr_pwm_en"])
    bl_pwm_en = int(config["bl_pwm_en"])
    br_pwm_en = int(config["br_pwm_en"])

    default_rows = 0
    for i in range(32):
        default_rows |= default << (i * 4)
    data = [default_rows] * 8 * row + pixels
    
    roi_lrow_cfg = config.get("roi_lrow")
    roi_lrow = (len(data) >> 3) - 1 if roi_lrow_cfg is None else int(roi_lrow_cfg)

    instr = (
        dev.commands["USB_2_DIS"] << 120
        | par_or_ser << 119
        | sel_tl_pixen << 78
        | sel_tr_pixen << 77
        | sel_bl_pixen << 76
        | sel_br_pixen << 75
        | tl_pixen << 74
        | tr_pixen << 73
        | bl_pixen << 72
        | br_pixen << 71
        | tl_pwm_en << 70
        | tr_pwm_en << 69
        | bl_pwm_en << 68
        | br_pwm_en << 67
        | enable_pwm << 66
        | roi_fcol << 62
        | roi_lcol << 58
        | roi_frow << 49
        | roi_lrow << 40
        | pwm_dutyc << 30
        | pwm_ticks << 20
        | fr_ticks
    )
    dev.send_instruction(instr, data)


def set_pixel(
    dev: Smile, row: int, col: int, value: int, cfg: Optional[Dict[str, Any]] = None
) -> None:
    """Write a single pixel using one RESET_DIS + one USB_2_DIS.

    Sends data for rows 0..row only (8*(row+1) words). For early rows this is
    substantially less than the 2048 words required by send_image. RESET_DIS
    blanks the entire display before writing, so no separate clear pass is needed.
    """
    set_row(dev, row, value=0, default=0, cfg=cfg,
            _pixel_data=_pixel_row_constructor(col, value))


def set_col(
    dev: Smile,
    col: int,
    value: int,
    default: int = 0,
    cfg: Optional[Dict[str, Any]] = None,
) -> None:
    """Write a single column using USB_2_DIS."""
    # Import locally to avoid circular import
    from .image_processor import ImageProcessor

    frame = create_spot(256, 1, value, 0, col)
    if default != 0:
        frame = [p if p != 0 else default for p in frame]
    processor = ImageProcessor()
    data = processor._chunks_of_32(frame)
    send_image(dev, data, cfg)


# --- MISSING FUNCTIONS ADDED AS STUBS ---

def generate_scan_frames(*args, **kwargs):
    """
    Placeholder: This function was missing in the original source file.
    """
    raise NotImplementedError("generate_scan_frames is not implemented in this version.")

def scan_pixels(*args, **kwargs):
    """
    Placeholder: This function was missing in the original source file.
    """
    raise NotImplementedError("scan_pixels is not implemented in this version.")


__all__ = [
    "DEFAULT_CFG",
    "row_constructor",
    "create_spot",
    "set_row",
    "set_col",
    "set_pixel",
    "send_image",
    "generate_vstripe_matrix",
    "generate_hstripe_matrix",
    "generate_scan_frames",
    "scan_pixels",
]