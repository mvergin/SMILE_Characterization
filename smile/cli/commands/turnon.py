import typer
from rich import print

from ...core import Smile


def turnon(ctx: typer.Context) -> None:
    """Turn on the display (no arguments)."""
    if ctx.obj.get("dry_run"):
        print("[yellow]Dry-run: turn on display[/yellow]")
        return
    dev = Smile(timeout=5000, debug_lvl=0)
    try:
        dev.send_instruction(instr=(dev.commands["TURNON_DIS"] << 120))
        print("[green]Display turned on[/green]")
    finally:
        dev.close()

