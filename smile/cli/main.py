import typer
from rich import print

from ..utils.logging_utils import setup_logging

app = typer.Typer(help="SMILE command line tool", no_args_is_help=True)


@app.callback()
def main(
    ctx: typer.Context,
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable debug output"),
    dry_run: bool = typer.Option(False, "--dry-run", help="Run commands without talking to the device"),
) -> None:
    """Global options."""
    ctx.obj = {"dry_run": dry_run}
    setup_logging(verbose)

# Commands will be registered below




from .commands.program import program
from .commands.reset import reset
from .commands.turnon import turnon
from .commands.turnoff import turnoff
from .commands.set_row import set_row
from .commands.set_col import set_col
from .commands.set_pixel import set_pixel
from .commands.send_image import send_image_cmd
from .commands.send_pattern import send_pattern
from .commands.set_pwm import set_pwm
from .commands.vstripe import vstripe
from .commands.hstripe import hstripe
from .commands.spot import spot
from .commands.set_pixen import set_pixen
from .commands.read_reg import read_reg
from .commands.write_reg import write_reg

app.command()(program)
app.command()(reset)
app.command()(turnon)
app.command()(turnoff)
app.command()(set_row)
app.command()(set_col)
app.command()(set_pixel)
app.command(name="send-image")(send_image_cmd)
app.command(name="send-pattern")(send_pattern)
app.command(name="set-pwm")(set_pwm)
app.command()(vstripe)
app.command()(hstripe)
app.command()(spot)
app.command(name="set-pixen")(set_pixen)
app.command(name="read-reg")(read_reg)
app.command(name="write-reg")(write_reg)

if __name__ == "__main__":
    app()

