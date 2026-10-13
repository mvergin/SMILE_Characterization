import typer
from rich import print

from ...utils import progfpga


def program(ctx: typer.Context) -> None:
    """Program the FPGA bitstream using Impact (no arguments)."""
    if ctx.obj.get("dry_run"):
        print("[yellow]Dry-run: program FPGA[/yellow]")
        return
    progfpga()
    print("[green]FPGA programmed[/green]")

