"""Utility helpers for the SMILE package.

This subpackage groups FPGA programming helpers and display utilities used by
the CLI and higher level APIs.
"""

from .fpga import progfpga
from .display import (
    DEFAULT_CFG,
    row_constructor,
    create_spot,
    send_image,
    generate_vstripe_matrix,
    generate_hstripe_matrix,
    generate_scan_frames,
    scan_pixels,
)

__all__ = [
    "progfpga",
    "DEFAULT_CFG",
    "row_constructor",
    "create_spot",
    "send_image",
    "generate_vstripe_matrix",
    "generate_hstripe_matrix",
    "generate_scan_frames",
    "scan_pixels",
]
