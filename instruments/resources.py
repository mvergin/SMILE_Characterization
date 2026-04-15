"""VISA resource enumeration helper.

Isolates the ``pyvisa`` dependency from the GUI layer. The GUI only
needs the list of connected instrument resource strings to populate its
device-selector combo boxes; it does not need a live ``ResourceManager``
handle.
"""

from __future__ import annotations

from typing import List


def list_visa_resources() -> List[str]:
    """Return the current VISA resource strings (or an empty list).

    Never raises: any import error (``pyvisa`` missing) or backend error
    (no VISA runtime installed) is swallowed and reported as an empty
    list, letting the caller fall back to simulation placeholders.
    """
    try:
        import pyvisa
    except Exception:
        return []
    try:
        rm = pyvisa.ResourceManager()
        try:
            return list(rm.list_resources())
        finally:
            try:
                rm.close()
            except Exception:
                pass
    except Exception:
        return []
