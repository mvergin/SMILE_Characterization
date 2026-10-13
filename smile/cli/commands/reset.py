import typer
from rich import print

from ...core import Smile
from ...utils.display import send_image


def reset(ctx: typer.Context) -> None:
    """Reset the display and clear the frame memory (no arguments)."""
    if ctx.obj.get("dry_run"):
        print("[yellow]Dry-run: reset display[/yellow]")
        return
    dev = Smile(timeout=5000, debug_lvl=0)
    try:
        dev.send_instruction(instr=(dev.commands["RESET_DIS"] << 120))
        blank_frame = [0] * (256 * 8)
        send_image(dev, blank_frame)
        print("[green]Display reset[/green]")
    finally:
        dev.close()

