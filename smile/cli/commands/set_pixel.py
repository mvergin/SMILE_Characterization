import json
import typer
from rich import print

from ...core import Smile
from ...utils.display import set_pixel as display_set_pixel


def set_pixel(
    ctx: typer.Context,
    row: int,
    col: int,
    value: int,
    config: str = typer.Option(None, help="Optional JSON or YAML config file"),
) -> None:
    """Set a pixel: ROW COL VALUE [--config FILE]."""
    cfg = {}
    if config:
        with open(config, "r", encoding="utf-8") as f:
            if config.lower().endswith((".yml", ".yaml")):
                import yaml
                cfg = yaml.safe_load(f)
            else:
                cfg = json.load(f)
    if ctx.obj.get("dry_run"):
        print(f"[yellow]Dry-run: set pixel {row},{col} to {value}[/yellow]")
        return
    dev = Smile(timeout=5000, debug_lvl=0)
    try:
        display_set_pixel(dev, row, col, value, cfg)
        print("[green]Pixel written[/green]")
    finally:
        dev.close()

