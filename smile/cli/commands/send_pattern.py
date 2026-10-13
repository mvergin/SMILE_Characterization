import typer
from rich import print

from ...core import Smile
from ...utils.image_processor import ImageProcessor
from ...utils.display import send_image


def send_pattern(ctx: typer.Context) -> None:
    """Send the internal test pattern (no arguments)."""
    if ctx.obj.get("dry_run"):
        print("[yellow]Dry-run: send pattern[/yellow]")
        return
    dev = Smile(timeout=5000, debug_lvl=0)
    try:
        processor = ImageProcessor()
        send_image(dev, processor.test_pattern)
        print("[green]Pattern sent[/green]")
    finally:
        dev.close()

