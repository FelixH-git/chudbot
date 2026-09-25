# Author Nils Wikström niwi0007
import socket

class RobotLink:
    def __init__(self, host: str = "0.0.0.0", port: int = 5000):
        self.host = host
        self.port = port
        self.serverSocket = None
        self.clientSocket = None
        self.clientIp = None

    # create a read only property calle connected, so this fucntion is a prop we can use to see if the robot is conncted
    @property
    def connected(self):
        return self.clientSocket is not None

    def start(self):
        self.serverSocket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        # Allows the server to reuse the same address and port shortly after restarting.
        self.serverSocket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.serverSocket.bind((self.host, self.port))
        self.serverSocket.listen(1)
        print(f"Waiting for the robot to connect to {self.host}:{self.port} ...")
        self.accept_client(timeout=1.0)
        return self

    def accept_client(self, timeout: float = 1.0):
        """Accepts a client connection if none is currently connected."""
        if self.serverSocket is None or self.connected:
            return
        self.serverSocket.settimeout(timeout)
        try:
            self.clientSocket, self.clientIp = self.serverSocket.accept()
            # Set client socket back to blocking mode with a timeout for sends
            self.clientSocket.settimeout(5.0)
            print(f"Robot at address {self.clientIp} connected")
        except socket.timeout:
            pass
        except OSError as e:
            pass

    def send(self, message: str):
        if not self.connected:
            print("Robot is not connected - command queued or dropped.")
            return
        try:
            self.clientSocket.sendall(message.encode("utf-8"))
            print(f"Sent '{message}' to robot")
        except Exception as e:
            print(f"Error sending to robot ({e}), marking disconnected.")
            if self.clientSocket is not None:
                try:
                    self.clientSocket.close()
                except OSError:
                    pass
            self.clientSocket = None

    def close(self):
        for sock in (self.clientSocket, self.serverSocket):
            if sock is not None:
                try:
                    sock.close()
                except OSError:
                    pass
        self.clientSocket = None
        self.serverSocket = None
 



