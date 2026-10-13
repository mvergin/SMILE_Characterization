import json
from typing import Optional

import typer
from rich import print

from ...core import Smile


def set_pixen(
    ctx: typer.Context,
    sel_tl: Optional[int] = typer.Option(
        None, help="Top-left source: 1 external, 0 internal"
    ),
    sel_tr: Optional[int] = typer.Option(
        None, help="Top-right source: 1 external, 0 internal"
    ),
    sel_bl: Optional[int] = typer.Option(
        None, help="Bottom-left source: 1 external, 0 internal"
    ),
    sel_br: Optional[int] = typer.Option(
        None, help="Bottom-right source: 1 external, 0 internal"
    ),
    tl: Optional[int] = typer.Option(
        None, help="Top-left internal enable (1 on, 0 off)"
    ),
    tr: Optional[int] = typer.Option(
        None, help="Top-right internal enable (1 on, 0 off)"
    ),
    bl: Optional[int] = typer.Option(
        None, help="Bottom-left internal enable (1 on, 0 off)"
    ),
    br: Optional[int] = typer.Option(
        None, help="Bottom-right internal enable (1 on, 0 off)"
    ),
    config: str = typer.Option(None, help="Optional JSON or YAML config file"),
) -> None:
    """Configure pixel enable sources and internal states.

    Unspecified options preserve their current value by reading the
    display configuration register before applying changes.
    """
    cfg = {}
    if config:
        with open(config, "r", encoding="utf-8") as f:
            if config.lower().endswith((".yml", ".yaml")):
                import yaml
                cfg = yaml.safe_load(f)
            else:
                cfg = json.load(f)
    if ctx.obj.get("dry_run"):
        print("[yellow]Dry-run: set pixel enable[/yellow]")
        return

    dev = Smile(timeout=5000, debug_lvl=0)
    try:
        current = dev.read_register(0x01)

        def _bit(pos: int) -> int:
            return (current >> pos) & 1

        sel_tl_val = int(
            cfg.get("sel_tl_pixen")
            if "sel_tl_pixen" in cfg
            else sel_tl
            if sel_tl is not None
            else _bit(78)
        )
        sel_tr_val = int(
            cfg.get("sel_tr_pixen")
            if "sel_tr_pixen" in cfg
            else sel_tr
            if sel_tr is not None
            else _bit(77)
        )
        sel_bl_val = int(
            cfg.get("sel_bl_pixen")
            if "sel_bl_pixen" in cfg
            else sel_bl
            if sel_bl is not None
            else _bit(76)
        )
        sel_br_val = int(
            cfg.get("sel_br_pixen")
            if "sel_br_pixen" in cfg
            else sel_br
            if sel_br is not None
            else _bit(75)
        )
        tl_val = int(
            cfg.get("tl_pixen")
            if "tl_pixen" in cfg
            else tl
            if tl is not None
            else _bit(74)
        )
        tr_val = int(
            cfg.get("tr_pixen")
            if "tr_pixen" in cfg
            else tr
            if tr is not None
            else _bit(73)
        )
        bl_val = int(
            cfg.get("bl_pixen")
            if "bl_pixen" in cfg
            else bl
            if bl is not None
            else _bit(72)
        )
        br_val = int(
            cfg.get("br_pixen")
            if "br_pixen" in cfg
            else br
            if br is not None
            else _bit(71)
        )

        instr = (
            dev.commands["SET_PIXEN"] << 120
            | sel_tl_val << 119
            | sel_tr_val << 118
            | sel_bl_val << 117
            | sel_br_val << 116
            | tl_val << 115
            | tr_val << 114
            | bl_val << 113
            | br_val << 112
        )
        dev.send_instruction(instr)
        print("[green]Pixel enable configured[/green]")
    finally:
        dev.close()
