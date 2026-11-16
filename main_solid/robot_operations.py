# Author Nils Wikström niwi0007
"""
Allt som har med roboten att göra:
  RobotFrame    vinkel i markörens frame -> RZ (hur mycket roboten ska vrida biten)
  pick_command  texten (protokollet) som RAPID-koden på roboten läser
  RobotLink     TCP-servern som roboten (ABB) ansluter till
"""
import socket
from typing import Dict

from detection import DetectedShape
from settings import MARKER_TO_ROBOT_ROT_DEG, MARKER_Y_FLIPPED, SLOT_ANGLE_ROBOT_DEG


def wrap_period(deg: float, period: float) -> float:
    """Viker en vinkel till intervallet [-period/2, period/2). Ex: kvadrat (90), 100° -> 10°."""
    return (deg + period / 2.0) % period - period / 2.0


class RobotFrame:
    def __init__(self, symmetry_deg: Dict[str, float], slot_angles: Dict[str, float] = SLOT_ANGLE_ROBOT_DEG,
                 y_flipped: bool = MARKER_Y_FLIPPED, rot_offset_deg: float = MARKER_TO_ROBOT_ROT_DEG):
        self.symmetry_deg = symmetry_deg        # former utan symmetri (cirkel) har ingen rotation
        self.slot_angles = slot_angles
        self.y_flipped = y_flipped
        self.rot_offset_deg = rot_offset_deg

    def robot_angle(self, marker_angle_deg: float) -> float:
        """Vinkel i markörens frame -> vinkel i robotens tray-frame (spegling + extra vridning)."""
        angle = -marker_angle_deg if self.y_flipped else marker_angle_deg
        return angle + self.rot_offset_deg

    def rz(self, shape: str, marker_angle_deg: float) -> float:
        """Hur mycket roboten ska vrida: fackets vinkel - bitens vinkel, kortaste vägen tack vare symmetrin."""
        if shape not in self.symmetry_deg:
            return 0.0
        rz = self.slot_angles[shape] - self.robot_angle(marker_angle_deg)
        return round(wrap_period(rz, float(self.symmetry_deg[shape])), 1)


def pick_command(piece: DetectedShape) -> str:
    """Texten som RAPID-koden på roboten läser från socketen."""
    sym = {"Circle": 360.0, "Hexagon": 60.0, "Square": 90.0, "Star": 72.0, "Triangle": 120.0}.get(piece.shape, 360.0)
    rz = wrap_period(-piece.rz_deg, sym)
    return f"SHAPE={piece.shape};X={piece.x_mm:.1f};Y={piece.y_mm:.1f};RZ={rz:.1f};END\r\n"


class RobotLink:
    def __init__(self, host: str = "0.0.0.0", port: int = 5000):
        self.host = host
        self.port = port
        self.serverSocket = None
        self.clientSocket = None
        self.clientIp = None

    # Read-only property: True om roboten är ansluten
    @property
    def connected(self):
        return self.clientSocket is not None

    def start(self):
        self.serverSocket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

        # Låter servern använda samma adress och port direkt efter en omstart
        self.serverSocket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)

        self.serverSocket.bind((self.host, self.port))
        self.serverSocket.listen(1)

        print(f"waiting for the robot to connect to {self.host}:{self.port}")
        try:
            self.clientSocket, self.clientIp = self.serverSocket.accept()
        except OSError:                              # socketen stängdes medan vi väntade
            return
        print(f"Robot at address {self.clientIp} connected")

    def send(self, message):
        if not self.connected:
            print("robot is not connected")
            return
        self.clientSocket.sendall(message.encode("utf-8"))
        print(f"sent {message} to robot")

    def close(self):
        for sock in (self.clientSocket, self.serverSocket):
            if sock is not None:
                try:
                    sock.close()
                except OSError:
                    pass
        self.clientSocket = None
        self.serverSocket = None
