import json
import typer
from rich import print

from ...core import Smile
from ...utils.image_processor import ImageProcessor
from ...utils.display import send_image


def send_image_cmd(
    ctx: typer.Context,
    path: str,
    gray: int = typer.Option(16, help="Grayscale levels"),
    config: str = typer.Option(None, help="Optional JSON or YAML config file"),
    full: bool = typer.Option(
        False, "--full", help="Use full 512x512 resolution (disable mirroring)"
    ),
) -> None:
    """Send an image: PATH [--gray LEVELS] [--config FILE]."""
    cfg = {}
    if config:
        with open(config, "r", encoding="utf-8") as f:
            if config.lower().endswith((".yml", ".yaml")):
                import yaml

                cfg = yaml.safe_load(f)
            else:
                cfg = json.load(f)
    canvas_size = 256

    if full:
        canvas_size = 512
        print("[blue]Mode: Full 512x512 (Mirroring Disabled)[/blue]")

        # These settings tell the FPGA to treat quadrants individually
        # instead of broadcasting one quadrant to all four.
        full_res_overrides = {
            "sel_tl_pixen": 1,
            "sel_tr_pixen": 1,
            "sel_bl_pixen": 1,
            "sel_br_pixen": 1,
            # Ensure quadrants are actually enabled
            "tl_pixen": 1,
            "tr_pixen": 1,
            "bl_pixen": 1,
            "br_pixen": 1,
        }
        cfg.update(full_res_overrides)
    if ctx.obj.get("dry_run"):
        print(f"[yellow]Dry-run: send image {path}[/yellow]")
        return
    dev = Smile(timeout=5000, debug_lvl=0)
    try:
        processor = ImageProcessor(path)
        processor.process(
            grayscale_levels=gray, canvas_width=canvas_size, canvas_height=canvas_size
        )
        send_image(dev, processor.display_data, cfg)
        print("[green]Image sent[/green]")
    finally:
        dev.close()
