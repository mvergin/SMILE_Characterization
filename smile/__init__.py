"""High-level API for the SMILE microdisplay.

The package exposes the :class:`~smile.core.Smile` USB driver along with
utility classes for converting images into the 4-bit format expected by the
hardware.
"""

from .utils.image_processor import ImageProcessor
from .core import Smile

__version__ = "0.1.0"

__all__ = ["Smile", "ImageProcessor"]
