# Raspberry Pi 4B Vision Node for ABB Robot
# - Basler Camera image grab (pypylon)
# - ArUco marker real-world homography (pixels -> mm)
# - 3-Head ONNX Neural Network (Shape + Angle + Coordinates in 1 pass)
# - ABB Robot Socket TCP Server

import cv2
import numpy as np
import onnxruntime as ort
from pypylon import pylon
import socket
import time

ONNX_MODEL_PATH = "shape_3head_model.onnx"
MARKER_SIZE_MM  = 100.0
ROBOT_PORT      = 5000

CLASSES = ["Circle", "Hexagon", "Square", "Star", "Triangle"]
SYMMETRY = {"Circle": 360.0, "Hexagon": 60.0, "Triangle": 120.0, "Square": 90.0, "Star": 72.0}

# 1. Initialize ONNX Engine (uses all 4 Pi 4B cores)
sess_opt = ort.SessionOptions()
sess_opt.intra_op_num_threads = 4
session = ort.InferenceSession(ONNX_MODEL_PATH, sess_opt, providers=["CPUExecutionProvider"])
input_name = session.get_inputs()[0].name

# 2. Connect Basler Camera via pypylon
tl_factory = pylon.TlFactory.GetInstance()
devices = tl_factory.EnumerateDevices()
if len(devices) == 0:
    raise RuntimeError("No Basler camera found! Plug in USB3 or Ethernet cable.")
camera = pylon.InstantCamera(tl_factory.CreateDevice(devices[0]))
camera.Open()
camera.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)

converter = pylon.ImageFormatConverter()
converter.OutputPixelFormat = pylon.PixelType_BGR8packed

# 3. ArUco Marker Setup
ARUCO_DICT = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_100)
ARUCO_PARAMS = cv2.aruco.DetectorParameters()

def get_homography(img_bgr):
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    corners, ids, _ = cv2.aruco.detectMarkers(gray, ARUCO_DICT, parameters=ARUCO_PARAMS)
    if ids is None or len(ids) == 0:
        return None
    pts_px = corners[0].reshape(4, 2).astype(np.float32)
    pts_mm = np.array([[0,0], [MARKER_SIZE_MM, 0], [MARKER_SIZE_MM, MARKER_SIZE_MM], [0, MARKER_SIZE_MM]], dtype=np.float32)
    H, _ = cv2.findHomography(pts_px, pts_mm)
    return H

# 4. Main Robot Vision Loop
print(f"Starting Vision Node on Raspberry Pi. Waiting for ABB Robot on port {ROBOT_PORT}...")
server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
server.bind(("0.0.0.0", ROBOT_PORT))
server.listen(1)

try:
    client, addr = server.accept()
    print(f"Connected to ABB Robot controller at {addr}!")
except Exception:
    client = None

mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
std  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

while camera.IsGrabbing():
    grab_res = camera.RetrieveResult(5000, pylon.TimeoutHandling_ThrowException)
    if not grab_res.GrabSucceeded():
        continue
    frame = converter.Convert(grab_res).GetArray()
    grab_res.Release()

    H = get_homography(frame)
    if H is None:
        continue

    # Detect shapes
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, 30, 255, cv2.THRESH_BINARY)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for cnt in contours:
        if cv2.contourArea(cnt) < 2500:
            continue
        x, y, w, h = cv2.boundingRect(cnt)
        roi = frame[max(0, y-10):y+h+10, max(0, x-10):x+w+10]
        if roi.size == 0:
            continue

        # ONE-SHOT 3-HEAD INFERENCE
        rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(rgb, (128, 128)).astype(np.float32) / 255.0
        norm = (resized - mean) / std
        tensor = np.expand_dims(np.transpose(norm, (2, 0, 1)), axis=0)

        out = session.run(None, {input_name: tensor})
        shape_idx = int(np.argmax(out[0][0]))
        shape = CLASSES[shape_idx]

        # Only process our target shapes
        if shape not in ["Hexagon", "Triangle", "Circle"]:
            continue

        # Angle & robot fix
        sin_v, cos_v = float(out[1][0][0]), float(out[1][0][1])
        angle = float(np.degrees(np.arctan2(sin_v, cos_v)))
        sym = SYMMETRY[shape]
        if shape == "Circle":
            fix_angle = 0.0
        else:
            delta = (-angle) % sym
            if delta > sym / 2.0:
                delta -= sym
            fix_angle = round(float(delta), 1)

        # Centroid coordinates -> mm
        cx_px = x + float(out[2][0][0]) * w
        cy_px = y + float(out[2][0][1]) * h
        pt_mm = H @ np.array([cx_px, cy_px, 1.0], dtype=np.float32)
        x_mm = round(float(pt_mm[0] / pt_mm[2]), 1)
        y_mm = round(float(pt_mm[1] / pt_mm[2]), 1)

        # Send to ABB Robot
        msg = f"SHAPE={shape};X={x_mm};Y={y_mm};RZ={fix_angle};END\r\n"
        print("Sending to Robot:", msg.strip())
        if client:
            client.sendall(msg.encode("ascii"))
            time.sleep(1.0)
