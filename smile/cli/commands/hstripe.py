import json
import typer
from rich import print

from ...core import Smile
from ...utils.display import generate_hstripe_matrix
from ...utils.image_processor import ImageProcessor
from ...utils.display import send_image


def hstripe(
    ctx: typer.Context,
    intensity: int,
    start: int,
    height: int = typer.Option(4, help="Stripe height in pixels"),
    config: str = typer.Option(None, help="Optional JSON or YAML config file"),
) -> None:
    """Send horizontal stripes: INTENSITY START [--height N] [--config FILE]."""
    cfg = {}
    if config:
        with open(config, "r", encoding="utf-8") as f:
            if config.lower().endswith((".yml", ".yaml")):
                import yaml
                cfg = yaml.safe_load(f)
            else:
                cfg = json.load(f)
    intensity = int(cfg.get("intensity", intensity))
    start = int(cfg.get("start", start))
    height = int(cfg.get("height", height))
    pattern = generate_hstripe_matrix(stripe_height=height, start=start, intensity=intensity)
    processor = ImageProcessor()
    data = processor._chunks_of_32(pattern)
    if ctx.obj.get("dry_run"):
        print("[yellow]Dry-run: send horizontal stripe pattern[/yellow]")
        return
    dev = Smile(timeout=5000, debug_lvl=0)
    try:
        send_image(dev, data, cfg)
        print("[green]Horizontal stripe pattern sent[/green]")
    finally:
        dev.close()
