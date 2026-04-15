"""Backward-compatibility shim for the pre-refactor flat instrumentlib.

The real drivers now live in the `instruments/` package — see
`instruments/pm400.py` and `instruments/keithley2602b.py`. This module
only exists so that existing `import instrumentlib` call sites and the
VERSION-gate in the GUI continue to work while the refactor lands.

New code should import from `instruments` directly.
"""

from instruments.keithley2602b import Keithley2602B
from instruments.pm400 import PM400

VERSION = "20260415_v1"

__all__ = ["PM400", "Keithley2602B", "VERSION"]
