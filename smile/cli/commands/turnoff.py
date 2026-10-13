import typer
from rich import print

from ...core import Smile


def turnoff(ctx: typer.Context) -> None:
    """Turn off the display (no arguments)."""
    if ctx.obj.get("dry_run"):
        print("[yellow]Dry-run: turn off display[/yellow]")
        return
    dev = Smile(timeout=5000, debug_lvl=0)
    try:
        dev.send_instruction(instr=(dev.commands["TURNOFF_DIS"] << 120))
        print("[green]Display turned off[/green]")
    finally:
        dev.close()

