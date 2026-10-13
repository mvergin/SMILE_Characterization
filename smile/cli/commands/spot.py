import json
import typer
from rich import print

from ...core import Smile
from ...utils.display import create_spot
from ...utils.image_processor import ImageProcessor
from ...utils.display import send_image


def spot(
    ctx: typer.Context,
    rows: int,
    cols: int,
    intensity: int,
    start_row: int,
    start_col: int,
    config: str = typer.Option(None, help="Optional JSON or YAML config file"),
) -> None:
    """Send a spot: ROWS COLS INTENSITY START_ROW START_COL [--config FILE]."""
    cfg = {}
    if config:
        with open(config, "r", encoding="utf-8") as f:
            if config.lower().endswith((".yml", ".yaml")):
                import yaml
                cfg = yaml.safe_load(f)
            else:
                cfg = json.load(f)
    rows = int(cfg.get("rows", rows))
    cols = int(cfg.get("cols", cols))
    intensity = int(cfg.get("intensity", intensity))
    start_row = int(cfg.get("start_row", start_row))
    start_col = int(cfg.get("start_col", start_col))
    pattern = create_spot(rows, cols, intensity, start_row, start_col)
    processor = ImageProcessor()
    data = processor._chunks_of_32(pattern)
    if ctx.obj.get("dry_run"):
        print("[yellow]Dry-run: send spot pattern[/yellow]")
        return
    dev = Smile(timeout=5000, debug_lvl=0)
    try:
        send_image(dev, data, cfg)
        print("[green]Spot sent[/green]")
    finally:
        dev.close()

