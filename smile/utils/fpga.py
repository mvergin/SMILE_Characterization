"""Program the FPGA bitstream using Xilinx Impact.

The :func:`progfpga` helper runs the Impact batch file bundled with the
repository.  By default it expects ``hardware/ise/scripts/impact.cmd``
and executes from ``hardware/ise/syn`` with the Xilinx binaries located
under ``/opt/Xilinx/14.7/ISE_DS/ISE/bin/lin64/``.
"""

from pathlib import Path
from os import system
from termcolor import colored

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]  # puja 2 nivells fins al root del projecte

def progfpga(
    impact_file: str = ROOT / "hardware/ise/scripts/impact.cmd",
    working_dir: str = ROOT / "hardware/ise/syn",
    xilinx_dir: str = "/opt/Xilinx/14.7/ISE_DS/ISE/bin/lin64/",
) -> None:
    IMPACT_OPTS = f" -batch {impact_file} >> impact.log"
    command = f"cd {working_dir} && {xilinx_dir}impact {IMPACT_OPTS}"
    ack = system(command)
    if ack == 256:
        raise Exception(colored("ERROR!! Bitstream file not found.", "red"))
