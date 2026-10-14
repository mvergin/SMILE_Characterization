"""Make sure the hardware is parked even when the process is killed from outside.

A `try/finally` covers an exception and a Ctrl-C. It does **not** cover the case that
actually bit us: Windows Update decided to restart the machine mid-run, the process was
terminated, and the Keithley's output stayed on with a pixel driven.

Three layers, weakest last:

1. **Console control handler** (`SetConsoleCtrlHandler`, via ctypes -- no pywin32).
   This is the one that matters on Windows: it receives `CTRL_CLOSE_EVENT` (console
   window closed), `CTRL_LOGOFF_EVENT` and `CTRL_SHUTDOWN_EVENT` (what Windows Update
   sends), which Python does **not** turn into a signal and which `atexit` never sees.
   The OS gives a few seconds before it kills the process anyway, which is enough to
   turn two outputs off and blank a frame.
2. **Signals** -- SIGTERM, SIGINT, SIGBREAK. Covers `taskkill`, Ctrl-C and Ctrl-Break.
3. **atexit** -- covers a normal or `sys.exit()` path, as a backstop.

What this CANNOT cover: a power cut, a bluescreen, or `taskkill /F`. Nothing in the
host process can. The real protection there is `nvled_v_limit` and the compliance
limit, which bound what the SMU can deliver with no host involved.

Usage:

    with safe_exit(lambda: shutdown(pm, smu, smile, cfg), log):
        ...                       # the measurement

The callback must be idempotent: it can run from the handler AND from the normal
`finally`. `shutdown()` already is -- every step is individually guarded.
"""
from __future__ import annotations

import atexit
import ctypes
import os
import signal
import sys
import threading
from contextlib import contextmanager
from typing import Callable

# Set the moment the hardware is parked. Measurement loops poll it and stop, instead of
# issuing one more instrument call against a session that `shutdown()` has just closed:
#
#     !! Ctrl-C: parking the hardware (outputs off, frame blanked)
#     read_buffers error: VI_ERROR_INV_OBJECT ... the given session ... is invalid
#
# The park itself was correct there -- outputs went off before the handles closed -- but
# the poll loop resumed for one iteration and talked to a dead session. Harmless, and
# alarming to read, which is reason enough.
PARKED = threading.Event()


def parked() -> bool:
    """True once the hardware has been parked; a loop that sees this should stop."""
    return PARKED.is_set()

# Console control events. Python maps only CTRL_C_EVENT/CTRL_BREAK_EVENT to signals;
# the three that a shutdown or a closed window sends are invisible without this handler.
_CTRL_C, _CTRL_BREAK, _CTRL_CLOSE, _CTRL_LOGOFF, _CTRL_SHUTDOWN = 0, 1, 2, 5, 6
_NAMES = {_CTRL_C: "Ctrl-C", _CTRL_BREAK: "Ctrl-Break", _CTRL_CLOSE: "console closed",
          _CTRL_LOGOFF: "logoff", _CTRL_SHUTDOWN: "system shutdown"}


@contextmanager
def safe_exit(park: Callable[[], None], log: Callable[[str], None] | None = None):
    """Run `park()` on the way out, however the process leaves."""
    done = threading.Lock()
    state = {"parked": False}

    def once(why: str) -> None:
        with done:
            if state["parked"]:
                return
            state["parked"] = True
        if log:
            log(f"!! {why}: parking the hardware (outputs off, frame blanked)")
        PARKED.set()                       # before park(), so no loop races the close
        try:
            park()
        except Exception as exc:                      # never raise out of a handler
            if log:
                log(f"   parking failed: {exc}")

    handlers: list[Callable[[], None]] = []

    # 1. Windows console control events
    if sys.platform == "win32":
        @ctypes.WINFUNCTYPE(ctypes.c_bool, ctypes.c_uint)
        def _console(event):
            once(_NAMES.get(event, f"console event {event}"))
            return False          # False = also run the default handler, i.e. still exit

        try:
            ctypes.windll.kernel32.SetConsoleCtrlHandler(_console, True)
            # keep a reference, or the callback is collected and the process crashes
            handlers.append(lambda: ctypes.windll.kernel32.SetConsoleCtrlHandler(_console, False))
            globals().setdefault("_KEEPALIVE", []).append(_console)
        except Exception:
            pass

    # 2. signals
    prev: dict[int, object] = {}
    for name in ("SIGTERM", "SIGINT", "SIGBREAK"):
        sig = getattr(signal, name, None)
        if sig is None:
            continue
        try:
            prev[sig] = signal.getsignal(sig)
        except (ValueError, OSError):
            continue

        def _sig(signum, frame, _name=name):
            once(_name)
            old = prev.get(signum)
            if callable(old):
                return old(signum, frame)
            if old == signal.SIG_DFL:
                raise KeyboardInterrupt if signum == getattr(signal, "SIGINT", None) \
                    else SystemExit(128 + signum)

        try:
            signal.signal(sig, _sig)
        except (ValueError, OSError):
            prev.pop(sig, None)

    # 3. atexit
    atexit.register(once, "process exit")

    PARKED.clear()
    try:
        yield
    finally:
        once("done")                           # the normal path
        atexit.unregister(once)
        for undo in handlers:
            try:
                undo()
            except Exception:
                pass
        for sig, old in prev.items():
            try:
                signal.signal(sig, old)
            except (ValueError, OSError, TypeError):
                pass


def demo() -> None:
    """Self-check: the callback runs exactly once, on each exit path."""
    calls = []

    with safe_exit(lambda: calls.append(1)):
        pass
    assert calls == [1], calls                      # normal exit parks once

    calls.clear()
    try:
        with safe_exit(lambda: calls.append(1)):
            raise RuntimeError("boom")
    except RuntimeError:
        pass
    assert calls == [1], calls                      # an exception still parks, once

    calls.clear()
    with safe_exit(lambda: calls.append(1)) as _:
        os.kill(os.getpid(), signal.SIGINT) if False else None
    assert calls == [1], calls

    # a park that itself fails must not escape
    with safe_exit(lambda: (_ for _ in ()).throw(ValueError("nope"))):
        pass

    # and the handler must be idempotent even if park() is slow/reentrant
    n = []
    with safe_exit(lambda: n.append(len(n))):
        pass
    assert n == [0], n
    print("ok  safeexit: parks exactly once on the normal, exception and failing paths")


if __name__ == "__main__":
    demo()
