import json
import typer
from rich import print

from ...core import Smile
from ...utils import DEFAULT_CFG
from ...utils.display import set_col as display_set_col


def set_col(
    ctx: typer.Context,
    col: int,
    value: int,
    default: int = typer.Option(0, help="Default pixel value"),
    config: str = typer.Option(None, help="Optional JSON or YAML config file"),
) -> None:
    """Set a column: COL VALUE [--default DEFAULT] [--config FILE]."""
    cfg = {}
    if config:
        with open(config, "r", encoding="utf-8") as f:
            if config.lower().endswith((".yml", ".yaml")):
                import yaml
                cfg = yaml.safe_load(f)
            else:
                cfg = json.load(f)
    cfg = {**cfg}
    if ctx.obj.get("dry_run"):
        print(f"[yellow]Dry-run: set col {col} to {value}[/yellow]")
        return
    dev = Smile(timeout=5000, debug_lvl=0)
    try:
        display_set_col(dev, col, value, default, cfg)
        print("[green]Column written[/green]")
    finally:
        dev.close()
