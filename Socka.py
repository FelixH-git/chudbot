import socket

# All the exact target coordinates you provided stored in a Python dictionary
targets = {
    "phexa": [[43.07, 156.38, 4.75], [0.00892356, 0.157212, -0.987515, -0.00424525], [0, 0, 0, 0], [9e09, 9e09, 9e09, 9e09, 9e09, 9e09]],
    "pstar": [[89.73, 95.53, 6.17], [0.00894014, 0.157212, -0.987515, -0.00422388], [0, 0, -1, 0], [9e09, 9e09, 9e09, 9e09, 9e09, 9e09]],
    "pcircle": [[148.24, 170.09, 6.39], [0.00908413, 0.157235, -0.987511, -0.00408367], [0, 0, 0, 0], [9e09, 9e09, 9e09, 9e09, 9e09, 9e09]],
    "psquare": [[147.44, 38.19, 5.31], [0.00826666, 0.157305, -0.987506, -0.00431963], [0, 0, -1, 0], [9e09, 9e09, 9e09, 9e09, 9e09, 9e09]],
    "phome": [[-25.31, 281.57, 234.72], [0.0145349, 0.160659, -0.986901, -0.00193449], [0, 0, -1, 0], [9e09, 9e09, 9e09, 9e09, 9e09, 9e09]],
    "papproach": [[116.45, -263.21, 168.99], [0.00810578, 0.157473, -0.987481, -0.00431433], [-1, 0, -1, 0], [9e09, 9e09, 9e09, 9e09, 9e09, 9e09]],
    "phome10": [[56.33, 485.14, 249.87], [0.0148255, 0.160884, -0.98686, -0.00205593], [0, 0, -1, 0], [9e09, 9e09, 9e09, 9e09, 9e09, 9e09]],
    "phome20": [[102.02, 105.35, 168.09], [0.00799293, 0.157506, -0.987477, -0.00426628], [0, 0, -1, 0], [9e09, 9e09, 9e09, 9e09, 9e09, 9e09]]
}

def sendToRobot(sendData):
    client_socket.send(sendData.encode("UTF-8"))
    print("Sent data to robot!")
    
    if client_socket.recv:
        client_message = client_socket.recv(4094).decode("latin-1")
        print("!!", client_message)
    return client_message

if __name__ == '__main__':
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.bind(('192.168.125.201', 5000))
    server_socket.listen()

    (client_socket, client_ip) = server_socket.accept()
    print(f"Robot at address {client_ip} connected.")
    
    while True:
        shape = input("Enter shape (pcircle, psquare, phexa, pstar, or exit): ").strip()
        
        if shape == "exit":
            sendToRobot("exit")
            print("exiting.....")
            break
            
        if shape in targets:
            # Extract the X and Y coordinates (index 0 of the target array contains [X, Y, Z])
            x_val = targets[shape][0][0]
            y_val = targets[shape][0][1]
            
            # Send format: "shape,X,Y"
            payload = f"{shape},{x_val},{y_val}"
            sendToRobot(payload)
        else:
            print("Unknown shape! Try again.")