"""TCP link to the ABB robot.

The robot runs as a *client* (see RobotCode/server.txt) and connects to this
Python process, which acts as a TCP *server*. Once connected we push pick
commands as plain text, e.g.

    SHAPE=Star;X=86.0;Y=-53.2;RZ=32.2;END\\r\\n

The RAPID side parses the key=value tokens in RobotCode/parser.txt.

Run this file directly for a manual test::

    python socket_link.py            # 0.0.0.0:5000, type commands by hand
"""

from __future__ import annotations

import socket
from typing import Optional

DEFAULT_HOST = "0.0.0.0"
DEFAULT_PORT = 5000


class RobotLink:
    """Server-side socket link to the robot (ABB)."""

    def __init__(self, host: str = DEFAULT_HOST, port: int = DEFAULT_PORT):
        self.host = host
        self.port = port
        self.server_socket: Optional[socket.socket] = None
        self.client_socket: Optional[socket.socket] = None
        self.client_ip: Optional[str] = None

    # -- state -----------------------------------------------------------------
    @property
    def connected(self) -> bool:
        return self.client_socket is not None

    # -- lifecycle -------------------------------------------------------------
    def start(self, timeout: Optional[float] = None) -> "RobotLink":
        """Bind, listen and wait for the robot to connect.

        ``timeout`` (seconds) makes the accept non-blocking-ish: if nobody
        connects in time this returns without a client, so a GUI can keep
        running and the caller can retry later.
        """
        self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        # Lets us rebind the same port immediately after a restart.
        self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.server_socket.bind((self.host, self.port))
        self.server_socket.listen(1)
        self.server_socket.settimeout(timeout)

        print(f"[SOCKET] Waiting for the robot on {self.host}:{self.port} ...")
        try:
            self.client_socket, self.client_ip = self.server_socket.accept()
        except (OSError, socket.timeout):
            self.server_socket.settimeout(None)
            print("[SOCKET] No robot connected yet.")
            return self

        self.server_socket.settimeout(None)
        print(f"[SOCKET] Robot at {self.client_ip} connected.")
        return self

    def send(self, message: str, read_reply: bool = False) -> Optional[str]:
        """Send a text payload to the robot.

        Fire-and-forget by default so a live loop is never blocked. Pass
        ``read_reply=True`` to wait (1 s) for a single reply line, e.g. when
        sending the ``exit`` command.
        """
        if not self.connected:
            print("[SOCKET] Robot not connected - nothing sent.")
            return None
        self.client_socket.sendall(message.encode("utf-8"))
        print(f"[SOCKET] Sent: {message.strip()}")

        if read_reply:
            try:
                self.client_socket.settimeout(1.0)
                reply = self.client_socket.recv(4096).decode("utf-8", errors="replace")
                print(f"[SOCKET] Reply: {reply.strip()}")
                return reply
            except (OSError, socket.timeout):
                return None
            finally:
                self.client_socket.settimeout(None)
        return None

    def close(self) -> None:
        """Close both sockets safely (never raises)."""
        for sock in (self.client_socket, self.server_socket):
            if sock is not None:
                try:
                    sock.close()
                except OSError:
                    pass
        self.client_socket = None
        self.server_socket = None
        print("[SOCKET] Closed.")


if __name__ == "__main__":
    link = RobotLink()
    link.start()
    try:
        while True:
            text = input("send> ").strip()
            if text in ("", "q", "quit"):
                break
            link.send(text if text.endswith("\r\n") else text + "\r\n")
    except (EOFError, KeyboardInterrupt):
        pass
    finally:
        link.close()
