import typer
from rich import print

from ...core import Smile


def set_pwm(ctx: typer.Context, period: int, duty: int) -> None:
    """Configure PWM: PERIOD DUTY.

    The current per-quadrant and global enable bits are preserved by
    reading the display configuration register first.
    """
    if duty > period:
        duty = period
    if ctx.obj.get("dry_run"):
        print("[yellow]Dry-run: set PWM[/yellow]")
        return
    dev = Smile(timeout=5000, debug_lvl=0)
    try:
        pwm_ticks = (duty << 10) | period
        current = dev.read_register(0x01)

        tl = (current >> 70) & 1
        tr = (current >> 69) & 1
        bl = (current >> 68) & 1
        br = (current >> 67) & 1
        enable = (current >> 66) & 1

        instr = (
            dev.commands["SET_PWM"] << 120
            | tl << 119
            | tr << 118
            | bl << 117
            | br << 116
            | enable << 115
            | pwm_ticks << 20
        )
        dev.send_instruction(instr)
        print("[green]PWM configured[/green]")
    finally:
        dev.close()

