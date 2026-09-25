# Author Nils Wikström niwi0007

"""
main.py - Multithreaded Vision & Robot Control System
Fulfills all course requirements with 4 dedicated threads:
  1. AI Model Inference Thread: Runs 3-head ONNX model on frame snapshots.
  2. Coordinate Translation Thread: Converts pixel coords to robot workspace mm.
  3. Server Communication Thread: Sends pick commands to ABB robot via TCP socket.
  4. Graphical User Interface (GUI) Thread: Live camera feed with HUD & click-to-pick.
"""

import os
import queue
import threading
import time
from typing import List, Optional

import cv2
import numpy as np

from camera import BaslerCamera
from socketConnection import RobotLink
from vision import DetectedShape, VisionPipeline

# SHARED THREAD QUEUES & STATE
frame_snapshot_queue: queue.Queue = queue.Queue(maxsize=1)
coord_translation_queue: queue.Queue = queue.Queue(maxsize=2)
robot_command_queue: queue.Queue = queue.Queue()

shapes_lock = threading.Lock()
latest_shapes: List[DetectedShape] = []

selected_shape_lock = threading.Lock()
selected_shape: str = "Circle"  # Default selected shape on program start
selected_shape_info: Optional[dict] = {
    "shape": "Circle",
    "bbox": (0, 0, 0, 0),
    "expire_time": 0.0,
}  # Initialized with default shape so it is never None or empty on startup

stop_event = threading.Event()


# THREAD 1: AI MODEL INFERENCE
def thread_ai_inference(pipeline: VisionPipeline):
    """Worker Thread: Continuously grabs frame snapshots and runs the ONNX model."""
    print("[Thread 1: AI Inference] Started.")
    while not stop_event.is_set():
        try:
            frame = frame_snapshot_queue.get(timeout=0.2)
        except queue.Empty:
            continue

        # Extract ROIs with HSV color masking (black background)
        rois = pipeline.extract_rois(frame)

        # Run 3-head ONNX inference on each detected contour crop
        raw_detections = []
        for bbox, roi in rois:
            det = pipeline.infer_roi(bbox, roi)
            if det is not None:
                raw_detections.append(det)

        # Send detections forward to the Coordinate Translation thread
        if not coord_translation_queue.full():
            coord_translation_queue.put((frame, raw_detections))

    print("[Thread 1: AI Inference] Stopped.")


# THREAD 2: COORDINATE TRANSLATION
def thread_coord_translation(pipeline: VisionPipeline):
    """Worker Thread: Translates detected pixel centers into robot table mm."""
    global latest_shapes
    print("[Thread 2: Coordinate Translation] Started.")

    while not stop_event.is_set():
        try:
            frame, detections = coord_translation_queue.get(timeout=0.2)
        except queue.Empty:
            continue

        # Ensure ArUco calibration is active
        if pipeline.homography_matrix is None:
            pipeline.update_homography(frame)

        # Re-verify and fine-tune real-world mm coordinates
        translated_shapes: List[DetectedShape] = []
        for obj in detections:
            coords_mm = pipeline.pixel_to_mm(obj.cx_px, obj.cy_px)
            if coords_mm is not None:
                obj.x_mm = coords_mm[0]
                obj.y_mm = coords_mm[1]
            translated_shapes.append(obj)

        # Safely publish latest detected shapes to GUI
        with shapes_lock:
            latest_shapes = translated_shapes

    print("[Thread 2: Coordinate Translation] Stopped.")


#THREAD 3: SERVER COMMUNICATION (ABB ROBOT)
def thread_robot_server(robot_link: RobotLink):
    """Worker Thread: Manages TCP connection and sends pick commands to the ABB robot."""
    print("[Thread 3: Server Communication] Started.")

    # Start socket server in non-blocking thread
    try:
        robot_link.start()
    except Exception as err:
        print(f"[Thread 3: Server Communication] Socket startup notice: {err}")

    while not stop_event.is_set():
        try:
            command = robot_command_queue.get(timeout=0.2)
        except queue.Empty:
            continue

        print(f"[Thread 3] Sending command to ABB Robot: {command.strip()}")
        robot_link.send(command)

    robot_link.close()
    print("[Thread 3: Server Communication] Stopped.")


#MOUSE CLICK HANDLER (CLICK-TO-PICK)
def on_mouse_click(event, mouse_x, mouse_y, flags, param):
    """Called by OpenCV whenever the user clicks the mouse on the video window."""
    global selected_shape, selected_shape_info

    if event == cv2.EVENT_LBUTTONDOWN:
        with shapes_lock:
            current_shapes = list(latest_shapes)

        for s in current_shapes:
            bx, by, bw, bh = s.bbox
            # Check if mouse clicked inside this shape's bounding box
            if bx <= mouse_x <= (bx + bw) and by <= mouse_y <= (by + bh):
                print(f"\n[USER CLICK] Selected {s.shape} at X={s.x_mm}mm, Y={s.y_mm}mm")
                cmd = s.to_robot_command()
                robot_command_queue.put(cmd)

                with selected_shape_lock:
                    selected_shape = s.shape
                    selected_shape_info = {
                        "shape": s.shape,
                        "bbox": s.bbox,
                        "expire_time": time.time() + 1.5,  # Highlight for 1.5s
                    }
                break


#THREAD 4: GRAPHICAL USER INTERFACE (MAIN THREAD)
def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(script_dir, "..", "MISC", "shape_3head_model.onnx")
    if not os.path.exists(model_path):
        model_path = os.path.join(script_dir, "shape_3head_model.onnx")

    image_path = os.path.join(script_dir, "test.png")

    # 1. Initialize Pipeline & Robot
    print("Initializing Vision Pipeline...")
    pipeline = VisionPipeline(model_path=model_path)
    robot_link = RobotLink(host="0.0.0.0", port=5000)

    # 2. Try Connecting to Basler Camera (fallback to test image if not plugged in)
    live_camera = None
    try:
        live_camera = BaslerCamera()
        live_camera.open()
        print("[CAMERA] Live Basler Camera connected successfully!")
    except Exception as e:
        print(f"[CAMERA] Basler camera not detected ({e}). Using '{image_path}' as simulated feed.")
        live_camera = None

    # Load fallback image
    static_frame = cv2.imread(image_path)
    if static_frame is None and live_camera is None:
        print(f"Error: Neither Basler camera nor '{image_path}' could be opened!")
        return

    # 3. Launch Background Threads (1, 2, 3)
    t1 = threading.Thread(target=thread_ai_inference, args=(pipeline,), daemon=True)
    t2 = threading.Thread(target=thread_coord_translation, args=(pipeline,), daemon=True)
    t3 = threading.Thread(target=thread_robot_server, args=(robot_link,), daemon=True)

    t1.start()
    t2.start()
    t3.start()

    # 4. Set up OpenCV GUI Window
    window_name = "ABB Robot Vision & Picking Control"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 1240, 720)
    cv2.setMouseCallback(window_name, on_mouse_click)

    print("\n=========================================================")
    print(" SYSTEM RUNNING (4 Threads Active)")
    print(" - CLICK on any shape on the screen to pick it!")
    print(" - HOTKEYS: [S]tar, [T]riangle, [C]ircle, [H]exagon")
    print(" - Press [Q] to exit.")
    print("=========================================================\n")

    last_snapshot_time = 0.0

    try:
        while not stop_event.is_set():
            # Grab latest frame (from live camera or simulated picture)
            if live_camera is not None and live_camera.isOpen:
                frame = live_camera.grabFrame()
                if frame is None:
                    continue
            else:
                frame = static_frame.copy()
                time.sleep(0.03)  # ~30 FPS throttle for image simulation

            # Periodically (every 200ms) feed a snapshot to the AI thread
            current_time = time.time()
            if current_time - last_snapshot_time > 0.2:
                if frame_snapshot_queue.empty():
                    frame_snapshot_queue.put(frame.copy())
                last_snapshot_time = current_time

            # Safely grab latest detections from Thread 2
            with shapes_lock:
                current_shapes = list(latest_shapes)

            # Draw HUD annotations
            display_frame = pipeline.draw_hud(frame, current_shapes)

            # Draw Gold Selection Highlight if user clicked a shape
            with selected_shape_lock:
                if selected_shape_info and selected_shape_info.get("expire_time", 0.0) > 0 and time.time() < selected_shape_info["expire_time"]:
                    bx, by, bw, bh = selected_shape_info["bbox"]
                    if bw > 0 and bh > 0:
                        cv2.rectangle(display_frame, (bx, by), (bx + bw, by + bh), (0, 215, 255), 4)
                        cv2.putText(
                            display_frame,
                            f">> SENDING {selected_shape_info['shape'].upper()} TO ROBOT <<",
                            (bx, max(30, by - 20)),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.7,
                            (0, 215, 255),
                            2,
                            cv2.LINE_AA,
                        )
                elif selected_shape_info and time.time() >= selected_shape_info.get("expire_time", 0.0):
                    selected_shape_info["expire_time"] = 0.0

            # Render Status Banner at Top of GUI
            status_text = (
                f"ROBOT: {'CONNECTED' if robot_link.connected else 'WAITING'} | "
                f"SELECTED: {selected_shape.upper()} | "
                f"CLICK SHAPE OR PRESS [S/T/C/H/U/SPACE]"
            )
            banner_color = (0, 180, 0) if robot_link.connected else (0, 120, 255)
            cv2.rectangle(display_frame, (0, 0), (display_frame.shape[1], 45), (30, 30, 30), -1)
            cv2.putText(
                display_frame,
                status_text,
                (20, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                banner_color,
                2,
                cv2.LINE_AA,
            )

            cv2.imshow(window_name, display_frame)

            # Keyboard Hotkeys
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q") or key == 27:  # 'q' or ESC
                break
            elif key in (ord("s"), ord("t"), ord("c"), ord("h"), ord("u")):
                key_map = {
                    ord("s"): "Star",
                    ord("t"): "Triangle",
                    ord("c"): "Circle",
                    ord("h"): "Hexagon",
                    ord("u"): "Square",
                }
                target_shape_name = key_map[key]

                # Find matching detected shape and dispatch
                for s in current_shapes:
                    if s.shape.lower() == target_shape_name.lower():
                        print(f"\n[HOTKEY] Selected {s.shape} at X={s.x_mm}, Y={s.y_mm}")
                        robot_command_queue.put(s.to_robot_command())
                        with selected_shape_lock:
                            selected_shape = s.shape
                            selected_shape_info = {
                                "shape": s.shape,
                                "bbox": s.bbox,
                                "expire_time": time.time() + 1.5,
                            }
                        break
            elif key in (32, 13):  # SPACE or ENTER to send currently selected shape
                for s in current_shapes:
                    if s.shape.lower() == selected_shape.lower():
                        print(f"\n[TRIGGER] Selected {s.shape} at X={s.x_mm}, Y={s.y_mm}")
                        robot_command_queue.put(s.to_robot_command())
                        with selected_shape_lock:
                            selected_shape_info = {
                                "shape": s.shape,
                                "bbox": s.bbox,
                                "expire_time": time.time() + 1.5,
                            }
                        break

    finally:
        stop_event.set()
        if live_camera is not None:
            live_camera.close()
        cv2.destroyAllWindows()
        print("Application shut down cleanly.")


if __name__ == "__main__":
    main()
