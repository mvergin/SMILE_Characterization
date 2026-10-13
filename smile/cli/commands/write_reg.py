import typer
from rich import print

from ...core import Smile


def write_reg(ctx: typer.Context, reg_id: int, value: int) -> None:
    """Write VALUE to configuration register REG_ID."""
    if ctx.obj.get("dry_run"):
        print(f"[yellow]Dry-run: write 0x{value:032X} to register 0x{reg_id:02X}[/yellow]")
        return
    dev = Smile(timeout=5000, debug_lvl=0)
    try:
        dev.write_register(reg_id, value)
        print(f"Wrote 0x{value:032X} to register 0x{reg_id:02X}")
    finally:
        dev.close()

