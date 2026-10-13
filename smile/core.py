"""
Created on 25/10/2022
@author: jcanals

Used to control the SMILE microdisplay through CalistoS6-USBC
Notes:
    USB transmission is in chuncks of 32 bits = 4 Bytes. --> sending first the LSByte
    The implemented protocol uses words of 128 bits. -->  expecting first the LSB
    BE CAREFUL in the .to_byte byteorder='xxx'
       (1).to_byte(4, byteorder = 'big')    -->  b'\x00\x00\x00\x01'
       (1).to_byte(4, byteorder = 'little') -->  b'\x01\x00\x00\x00'
"""

from time import strftime
from functools import reduce
from sys import platform
import logging

logger = logging.getLogger(__name__)
from typing import List, Tuple
from termcolor import colored
import ftd3xx


class Smile:
    """USB driver for the SMILE microdisplay controller."""

    # constants
    ACK = 0xF50F05F50F05F50F05F50F05F50F05F5
    NACK = 0x0F5F05F50F5F05F50F5F05F50F5F05F5
    commands = {
        "SOFTRESET": 0xFF,
        "STOP_PRCS": 0xF0,
        "USB_2_MEM": 0x90,
        "MEM_2_USB": 0x99,
        "MEM_RFSH": 0x14,
        "MEM_2_DIS": 0x39,
        "USB_2_DIS": 0xA1,
        "CAL_2_USB": 0xE9,
        "CAL_2_MEM": 0x79,
        "SCAN_COLS": 0x08,
        "TURNOFF_DIS": 0x20,
        "TURNON_DIS": 0x21,
        "SEL_PI": 0x22,
        "SEL_SI": 0x23,
        "SET_ROI": 0x24,
        "SET_FR": 0x25,
        "SET_PWM": 0x26,
        "ENABLE_PWM": 0x27,
        "DISABLE_PWM": 0x28,
        "RESET_DIS": 0x29,
        "TC_PIX_ON": 0x2A,
        "TC_PIX_OFF": 0x2B,
        "TC_PIX_WR": 0x2C,
        "VLED_ON": 0x2D,
        "VLED_OFF": 0x2E,
        "SET_PIXEN": 0x2F,
        "REG_RD": 0x30,
        "REG_WR": 0x31,
    }

    def __init__(self, timeout: int = 4000, debug_lvl: int = 0) -> None:
        """Create a connection to the microdisplay controller.

        Args:
            timeout: USB communication timeout in milliseconds.
            debug_lvl: Verbosity level for debug output.
        """
        self._dev = None
        self._debug_lvl = debug_lvl
        self.open(timeout)
        #  self._reset()
        self._test()

    def _reset(self) -> None:
        """Reset the controller and verify communication.

        Raises:
            Exception: If a negative acknowledgement is received.
        """
        if self._debug_lvl > 1:
            logger.debug(f"[{strftime('%H:%M:%S')}] Rest System")
        instr = self.commands["SOFTRESET"].to_bytes(16, "big")
        self._dev.send(instr)
        ack = self.wait_ack()
        if not ack:
            self._dev.close()
            raise Exception(colored("ERROR!! NACK", "red"))

    def _test(self) -> None:
        """Verify USB communication with the device.

        Raises:
            Exception: If no acknowledgement is received.
        """
        if self._debug_lvl > 0:
            logger.info(f"[{strftime('%H:%M:%S')}] Checking Comunications")
        instr = self.commands["STOP_PRCS"].to_bytes(16, "big")
        self._dev.send(instr)
        ack = self.wait_ack()
        if not ack:
            self._dev.close()
            raise Exception(colored("ERROR!! NACK", "red"))
        if self._debug_lvl > 0 and ack:
            logger.info("Ok!")

    def open(self, timeout: int = 4000) -> None:
        """Open the USB device.

        Args:
            timeout: Timeout in milliseconds for USB operations.
        """
        if self._debug_lvl > 0:
            logger.info(f"[{strftime('%H:%M:%S')}] Opening Comunications")
        self._dev = USB_FT60X_sync245mode(timeout)
        if self._debug_lvl > 0:
            logger.info("Ok!")

    def close(self) -> None:
        """Close the USB driver."""
        if self._debug_lvl > 0:
            logger.info(f"[{strftime('%H:%M:%S')}] Closing Comunications")
        self._dev.close()
        if self._debug_lvl > 0:
            logger.info("Ok!")

    def wait_ack(self) -> bool:
        """Wait for an acknowledgement from the device.

        Returns:
            bool: ``True`` if :data:`ACK` is received, ``False`` for :data:`NACK`.

        Raises:
            Exception: If an unknown ACK code is received.
        """
        rdata = int.from_bytes(self._dev.recv(16), "little")
        if rdata == self.ACK:
            return True
        if rdata == self.NACK:
            return False
        self._dev.close()
        raise Exception(colored(f"ERROR!! Unknown ACK CODE: 0x{rdata:016x}", "red"))

    def send_data(self, data: List[int]) -> None:
        """Send raw data words to the device.

        Args:
            data: List of 128-bit words to transmit.
        """
        # selects the valid data words and transform to bytes and concatenates them
        bstring = reduce(
            lambda x, y: x + y.to_bytes(16, "little"),
            data[1:],
            data[0].to_bytes(16, "little"),
        )
        if self._debug_lvl > 0:
            logger.info(f"[{strftime('%H:%M:%S')}] Sending data")
        self._dev.send(bstring)

    def send_instruction(self, instr: int, payload: List[int] = []) -> None:
        """Send a command followed by optional payload words.

        Args:
            instr: Encoded instruction word.
            payload: Optional list of data words.

        Raises:
            Exception: If the device replies with a NACK.
        """
        # build write instruction
        instr = instr.to_bytes(16, "little")
        if payload:
            # selects the valid data words and transform to bytes and concatenates them
            bstring = instr + reduce(
                lambda x, y: x + y.to_bytes(16, "little"),
                payload[1:],
                payload[0].to_bytes(16, "little"),
            )
        else:
            bstring = instr
        if self._debug_lvl > 0:
            logger.info(f"[{strftime('%H:%M:%S')}] Sending data")
        self._dev.send(bstring)
        ack = self.wait_ack()
        if not ack:
            self._dev.close()
            raise Exception(colored("ERROR!! NACK", "red"))
        if self._debug_lvl > 0 and ack:
            logger.info("Ok!")

    def read_data(self, instr: int, nwords: int) -> List[int]:
        """Read a number of data words from the device.

        Args:
            instr: Instruction to send before reading.
            nwords: Number of 128-bit words to read.

        Returns:
            List of integers containing the data words.

        Raises:
            Exception: If the device replies with a NACK or an unknown code.
        """
        # send instruction
        instr = instr.to_bytes(16, "little")
        if self._debug_lvl > 0:
            logger.info(f"[{strftime('%H:%M:%S')}] Reading data")
        self._dev.send(instr)
        bstring = self._dev.recv((nwords) * 16)  # *16 because each word = 16 bytes
        if self._debug_lvl > 2:
            logger.debug(bstring)
        # spliting bstring in a list of words
        rdata = [
            int.from_bytes(bstring[i : i + 16], "little")
            for i in range(0, len(bstring), 16)
        ]
        ack = self.wait_ack()
        if self._debug_lvl > 0 and ack:
            logger.info("Ok!")
        if not ack:
            self._dev.close()
            raise Exception(colored("ERROR!! NACK", "red"))
        return rdata

    def write_sram(self, addr: int, dwords: List[int]) -> None:
        """Write a sequence of words to the controller SRAM.

        Args:
            addr: Start address in bytes.
            dwords: Data words to write.
        """
        # caluclates the valid start addr, number of words and end address
        arange = self.addr_range(addr, len(dwords))
        # build write instruction
        instr = (
            self.commands["USB_2_MEM"] << 120
            | arange[0] << 48
            | arange[1] << 32
            | arange[2]
        )
        # selects the valid data words
        payload = dwords[0 : arange[1] + 1]
        # send instruction packet
        self.send_instruction(instr, payload)

    def read_sram(self, addr: int, nwords: int) -> List[int]:
        """Read words from the controller SRAM.

        Args:
            addr: Start address in bytes.
            nwords: Number of 128-bit words to read.

        Returns:
            List of integers with the words read.
        """
        # caluclates the valid start addr, number of words and end address
        arange = self.addr_range(addr, nwords)
        # build read instruction
        instr = (
            self.commands["MEM_2_USB"] << 120
            | arange[0] << 48
            | arange[1] << 32
            | arange[2]
        )
        # read data
        rdata = self.read_data(instr, arange[1] + 1)
        return rdata

    def write_register(self, reg_id: int, value: int) -> None:
        """Write a configuration register."""
        word = (
            self.commands["REG_WR"] << 120
            | (reg_id & 0xFF) << 112
            | (value & ((1 << 112) - 1))
        )
        self.send_instruction(word)

    def read_register(self, reg_id: int) -> int:
        """Read a configuration register."""
        word = (self.commands["REG_RD"] << 120) | ((reg_id & 0xFF) << 112)
        instr = word.to_bytes(16, "little")
        self._dev.send(instr)
        response = int.from_bytes(self._dev.recv(16), "little")
        self.wait_ack()
        return response

    def addr_range(self, addr: int, nwords: int) -> Tuple[int, int, int]:
        """Calculate a valid memory burst range.

        Args:
            addr: Starting address in bytes.
            nwords: Number of 128-bit words in the burst.

        Returns:
            Tuple ``(start, words, end)`` describing the sanitized range.
        """
        # force start address alignament and in range
        saddr = addr & 0x03FFFFFF0
        # check if burst will fit in memory
        tmp = saddr + ((nwords - 1) << 4)
        overrun = bool(tmp > 0x03FFFFFF0)
        eaddr = 0x03FFFFFF0 if overrun else tmp
        nwords = (eaddr - saddr) >> 4
        if self._debug_lvl > 1:
            if (addr & 0xF) != 0:
                logger.debug(
                    f"\r[{strftime('%H:%M:%S')}] Start address not aligned (0x{addr:08x})"
                )
            if addr > 0x03FFFFFFF:
                logger.debug(
                    f"\r[{strftime('%H:%M:%S')}] Start address out of range (0x){addr:08x})"
                )
            if addr <= 0x03FFFFFF0:
                logger.debug(
                    f"\r[{strftime('%H:%M:%S')}] Start address is valid (0x{addr:08x})"
                )
            if overrun:
                logger.debug(
                    f"\r[{strftime('%H:%M:%S')}] Burst Length too long, reduced to {nwords+1} words"
                )
        if self._debug_lvl > 1:
            logger.debug(f"saddr = 0x{saddr:08x}")
            logger.debug(f"nwords= 0x{nwords:04x}")
            logger.debug(f"eaddr = 0x{eaddr:08x}")
        return saddr, nwords, eaddr


class USB_FT60X_sync245mode:
    """Wrapper around the FTDI FT600/FT601 USB FIFO interface."""

    def __init__(self, timeout: int = 2000):
        """Initialize the USB communication channel.

        Args:
            timeout: Timeout in milliseconds for USB transfers.
        """
        usb = None

        if platform == "win32":
            usb = ftd3xx.create(0, ftd3xx._ftd3xx_win32.FT_OPEN_BY_INDEX)
        elif platform == "linux":
            usb = ftd3xx.create(0, ftd3xx._ftd3xx_linux.FT_OPEN_BY_INDEX)

        if usb is None:
            raise Exception("Could not find or open Device!")

        if platform == "win32" and usb.getDriverVersion() < 0x01020006:
            usb.close()
            raise Exception("Old kernel driver version. Please update driver!")

        if usb.getDeviceDescriptor().bcdUSB < 0x300:
            logger.debug(
                "*** Warning: Device is NOT connected using USB3.0 cable or port!"
            )

        cfg = usb.getChipConfiguration()
        n_channel = [4, 2, 1, 0, 0][cfg.ChannelConfig]
        if n_channel != 1:
            usb.close()
            raise Exception(f"Number of Channels invalid! (numChannel={n_channel})")

        if platform == "win32":
            usb.setPipeTimeout(0x02, timeout)
            usb.setPipeTimeout(0x82, timeout)

        usb.flushPipe(0)
        self._usb = usb
        self._chunk = 1024 * 16
        self._timeout = timeout

    def close(self) -> None:
        """Close the USB connection and release resources."""
        self._usb.flushPipe(0)
        self._usb.resetDevicePort()
        self._usb.close()

    def send(self, data: bytes) -> int:
        """Send a chunk of bytes over USB.

        Args:
            data: Bytes to transmit.

        Returns:
            Number of bytes successfully written.
        """
        txlen = 0
        for i in range(0, len(data), self._chunk):
            chunk = data[i : min(i + self._chunk, len(data))]
            if platform == "win32":
                txlen_once = self._usb.writePipe(0x02, chunk, len(chunk))
            else:
                txlen_once = self._usb.writePipe(0, chunk, len(chunk), self._timeout)
            txlen += txlen_once
            if txlen_once < len(chunk):
                break
        return txlen

    def recv(self, rxlen: int) -> bytes:
        """Receive a number of bytes from USB.

        Args:
            rxlen: Number of bytes to read.

        Returns:
            Bytes read from the device.
        """
        data = b""
        for i in range(rxlen, 0, -self._chunk):
            chunk = bytes(min(self._chunk, i))
            if platform == "win32":
                rxlen_once = self._usb.readPipe(0x82, chunk, len(chunk))
            else:
                rxlen_once = self._usb.readPipe(0, chunk, len(chunk), self._timeout)
            if rxlen_once < len(chunk):
                data += chunk[0:rxlen_once]
                break
            data += chunk
        return data
