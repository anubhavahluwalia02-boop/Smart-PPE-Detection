"""
Alerts and hardware integration module for buzzer, LED, and ESP32 microcontroller communication.
"""
import sys
import logging
from typing import Optional, Dict, Any

import config

logger = logging.getLogger(__name__)

# Attempt to import serial safely
try:
    import serial
    SERIAL_AVAILABLE = True
except ImportError:
    SERIAL_AVAILABLE = False
    serial = None

class AlertManager:
    """
    Handles audio beeps and serial transmission to external microcontrollers (e.g., ESP32 / Arduino).
    """

    def __init__(
        self,
        port: str = config.SERIAL_PORT,
        baud: int = config.SERIAL_BAUD,
        enable_serial: bool = config.ENABLE_SERIAL,
        enable_audio: bool = config.ENABLE_AUDIO_ALERT
    ):
        self.port = port
        self.baud = baud
        self.enable_serial = enable_serial
        self.enable_audio = enable_audio
        self.serial_conn: Optional[Any] = None
        self.last_sent_state = ""

        if self.enable_serial:
            self.connect_serial()

    def connect_serial(self) -> bool:
        """Attempts to open the serial port to the ESP32."""
        if not SERIAL_AVAILABLE:
            logger.warning("pyserial package is not installed. ESP32 communication disabled.")
            return False

        try:
            self.serial_conn = serial.Serial(self.port, self.baud, timeout=1)
            logger.info(f"Connected to ESP32 on port {self.port} at {self.baud} baud.")
            return True
        except Exception as e:
            logger.warning(f"Could not connect to serial port {self.port}: {e}")
            self.serial_conn = None
            return False

    def trigger_audio_alert(self):
        """Emits an audible warning sound."""
        if not self.enable_audio:
            return

        try:
            if sys.platform.startswith("win"):
                import winsound
                winsound.Beep(1800, 250)  # 1800 Hz for 250ms
            else:
                # Terminal bell on Linux / macOS
                sys.stdout.write("\a")
                sys.stdout.flush()
        except Exception:
            pass

    def send_raw_command(self, cmd: str) -> bool:
        """Sends an explicit raw ASCII string command to the ESP32."""
        if not self.serial_conn or not getattr(self.serial_conn, "is_open", False):
            if self.enable_serial:
                self.connect_serial()
            if not self.serial_conn or not getattr(self.serial_conn, "is_open", False):
                return False

        try:
            if not cmd.endswith("\n"):
                cmd += "\n"
            self.serial_conn.write(cmd.encode("utf-8"))
            self.serial_conn.flush()
            return True
        except Exception as e:
            logger.warning(f"Error sending serial command '{cmd.strip()}': {e}")
            self.serial_conn = None
            return False

    def ping(self) -> bool:
        """Sends a heartbeat ping to the ESP32 and checks for a response."""
        if not self.serial_conn or not getattr(self.serial_conn, "is_open", False):
            if not self.connect_serial():
                return False

        try:
            self.serial_conn.reset_input_buffer()
            self.serial_conn.write(b"PING\n")
            self.serial_conn.flush()
            response = self.serial_conn.readline().decode("utf-8", errors="ignore").strip()
            return "PONG" in response or "OK" in response
        except Exception as e:
            logger.debug(f"ESP32 ping heartbeat failed: {e}")
            return False

    def send_hardware_state(self, status: str) -> bool:
        """
        Sends formatted commands to ESP32 microcontroller.
        Commands supported by ESP32 firmware:
            'SAFE\\n'               -> Green LED (GPIO 18) ON, Red (GPIO 19) OFF, Buzzer (GPIO 21) OFF
            'VIOLATION\\n'          -> Red LED (GPIO 19) ON, Green (GPIO 18) OFF, Buzzer (GPIO 21) BEEP/ON
            'P18:1,P19:0,P21:0\\n'  -> Direct hardware pin control
        """
        # Audio alert for violations
        if "VIOLATION" in status:
            self.trigger_audio_alert()

        if status == self.last_sent_state:
            return True
        self.last_sent_state = status

        if not self.enable_serial:
            return False

        if not self.serial_conn or not getattr(self.serial_conn, "is_open", False):
            # Attempt automatic reconnect
            if not self.connect_serial():
                return False

        # Determine primary command and explicit pin states
        if "COMPLIANT" in status or status == "SAFE":
            primary_cmd = "SAFE\n"
            pin_cmd = f"P{config.ESP32_PIN_GREEN}:1,P{config.ESP32_PIN_RED}:0,P{config.ESP32_PIN_BUZZER}:0\n"
        elif "VIOLATION" in status:
            primary_cmd = "VIOLATION\n"
            pin_cmd = f"P{config.ESP32_PIN_GREEN}:0,P{config.ESP32_PIN_RED}:1,P{config.ESP32_PIN_BUZZER}:1\n"
        else:
            primary_cmd = "IDLE\n"
            pin_cmd = f"P{config.ESP32_PIN_GREEN}:0,P{config.ESP32_PIN_RED}:0,P{config.ESP32_PIN_BUZZER}:0\n"

        try:
            # Send standard SAFE / VIOLATION keyword
            self.serial_conn.write(primary_cmd.encode("utf-8"))
            # Send dual-protocol pin command for maximum firmware compatibility
            self.serial_conn.write(pin_cmd.encode("utf-8"))
            self.serial_conn.flush()
            logger.info(f"[ESP32 DISPATCH] Sent: {primary_cmd.strip()} & {pin_cmd.strip()}")
            return True
        except Exception as e:
            logger.warning(f"Error sending serial data to ESP32: {e}")
            self.serial_conn = None
            return False

    def close(self):
        """Releases the serial connection."""
        if self.serial_conn and hasattr(self.serial_conn, "close"):
            try:
                self.serial_conn.close()
            except Exception:
                pass
            self.serial_conn = None

    def get_hardware_status(self) -> Dict[str, Any]:
        """Returns the current status of the serial hardware interface."""
        return {
            "serial_available": SERIAL_AVAILABLE,
            "enabled": self.enable_serial,
            "port": self.port,
            "baud": self.baud,
            "connected": self.serial_conn is not None and getattr(self.serial_conn, "is_open", False),
            "last_state": self.last_sent_state or "IDLE"
        }
