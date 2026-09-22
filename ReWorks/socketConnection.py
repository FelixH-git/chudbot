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
        self.serverSocket = socket.socket(socket.AF_INET,socket.SOCK_STREAM)

        # Allows the server to reuse the same address and port shortly after restarting.
        self.serverSocket.setsockopt(socket.SOL_SOCKET, 
                                     socket.SO_REUSEADDR,
                                     1)
        
        self.serverSocket.bind((self.host,self.port))
        self.serverSocket.listen(1)

        print(f"waiting for the robot to connect to {self.host}:{self.port}")
        try:
            self.clientSocket, self.clientIp = self.serverSocket.accept()
        except OSError:
            return self
        print(f"Robot at address {self.clientIp} connected")


    def send(self,message):
        if not self.connected:
            print("robot is not connected")
            return
        self.clientSocket.sendall(message.encode("utf-8"))
        print(f"sent {message} to robot")

    def close(self):
            for sock in (self.clientSocket,self.serverSocket):
                if sock is not None:
                    try:
                        sock.close()
                    except OSError:
                        pass
            self.clientSocket = None
            self.serverSocketr = None
 



