import typer
from rich import print
from rich.table import Table

from ...core import Smile


DIS_REG_FIELDS = [
    ("reset", 82, 82),
    ("serial_mode", 81, 81),
    ("enable", 80, 80),
    ("master", 79, 79),
    ("sel_tl_pixen", 78, 78),
    ("sel_tr_pixen", 77, 77),
    ("sel_bl_pixen", 76, 76),
    ("sel_br_pixen", 75, 75),
    ("tl_pixen", 74, 74),
    ("tr_pixen", 73, 73),
    ("bl_pixen", 72, 72),
    ("br_pixen", 71, 71),
    ("tl_no_pwm", 70, 70),
    ("tr_no_pwm", 69, 69),
    ("bl_no_pwm", 68, 68),
    ("br_no_pwm", 67, 67),
    ("enable_pwm", 66, 66),
    ("roi_fcol", 65, 62),
    ("roi_lcol", 61, 58),
    ("roi_frow", 57, 49),
    ("roi_lrow", 48, 40),
    ("pwm_ticks", 39, 20),
    ("fr_delay", 19, 0),
]

MEM_REG_FIELDS = [
    ("instr_valid", 89, 89),
    ("ddr_flow", 88, 88),
    ("instr_code", 87, 80),
    ("start_addr", 79, 48),
    ("num_words", 45, 32),
    ("end_addr", 31, 0),
]

CAL_REG_FIELDS = [
    ("enable", 17, 17),
    ("master", 16, 16),
    ("delay_ticks", 15, 0),
]

SCAN_REG_FIELDS = [
    ("en_tl", 11, 11),
    ("en_tr", 10, 10),
    ("en_bl", 9, 9),
    ("en_br", 8, 8),
    ("sck_ticks", 7, 0),
]

TC_REG_FIELDS = [
    ("pix_out_en_n", 5, 5),
    ("sram_we", 4, 4),
    ("sram_data", 3, 0),
]

REG_MAP = {
    0x01: DIS_REG_FIELDS,
    0x02: MEM_REG_FIELDS,
    0x03: TC_REG_FIELDS,
    0x04: CAL_REG_FIELDS,
    # 0x05 has no documented fields
}


def _decode_bits(reg_id: int, value: int) -> list[tuple[str, int]]:
    fields = REG_MAP.get(reg_id)
    if not fields:
        return []
    decoded = []
    for name, hi, lo in fields:
        mask = (1 << (hi - lo + 1)) - 1
        decoded.append((name, (value >> lo) & mask))
    return decoded


def read_reg(
    ctx: typer.Context,
    reg_id: int,
    show_bits: bool = typer.Option(False, "--show-bits"),
) -> None:
    """Read a configuration register: REG_ID."""
    if ctx.obj.get("dry_run"):
        print(f"[yellow]Dry-run: read register 0x{reg_id:02X}[/yellow]")
        return
    dev = Smile(timeout=5000, debug_lvl=0)
    try:
        value = dev.read_register(reg_id)
        print(f"Register 0x{reg_id:02X} = 0x{value:032X}")
        if show_bits:
            fields = _decode_bits(reg_id, value)
            if fields:
                table = Table(show_header=True, header_style="bold")
                table.add_column("Field")
                table.add_column("Value", justify="right")
                for name, val in fields:
                    table.add_row(name, str(val))
                print(table)
            else:
                print("[red]No bit definitions for this register[/red]")
    finally:
        dev.close()

