"""Main program: live vision + picking for the ABB robot.

Four threads:
  1. AI thread      - runs the 3-head ONNX model on frames from the camera.
  2. Coord thread   - converts shape centres from pixels to mm on the table.
  3. Server thread  - TCP socket server, sends pick commands to the ABB robot.
  4. GUI (main)     - live view with HUD; click a shape to pick it.

Falls back to ``test.png`` when no Basler camera is attached, so the GUI and
model can be exercised on a laptop.

    python main.py
"""

from __future__ import annotations

import os
import queue
import threading
import time
from typing import List, Optional

import cv2

from camera import BaslerCamera
from socket_link import RobotLink
from vision import DetectedShape, VisionPipeline

# ── Shared state between threads ─────────────────────────────────────────────
frame_snapshot_queue: "queue.Queue" = queue.Queue(maxsize=1)
coord_translation_queue: "queue.Queue" = queue.Queue(maxsize=2)
robot_command_queue: "queue.Queue" = queue.Queue()

shapes_lock = threading.Lock()
latest_shapes: List[DetectedShape] = []

selected_shape_lock = threading.Lock()
selected_shape_info: Optional[dict] = None  # shape highlighted gold after a click

stop_event = threading.Event()


# ── Thread 1: AI inference ───────────────────────────────────────────────────
def thread_ai_inference(pipeline: VisionPipeline) -> None:
    print("[Thread 1: AI] Started.")
    while not stop_event.is_set():
        try:
            frame = frame_snapshot_queue.get(timeout=0.2)
        except queue.Empty:
            continue

        raw_detections = []
        for bbox, roi in pipeline.extract_rois(frame):
            det = pipeline.infer_roi(bbox, roi)
            if det is not None:
                raw_detections.append(det)

        if not coord_translation_queue.full():
            coord_translation_queue.put((frame, raw_detections))
    print("[Thread 1: AI] Stopped.")


# ── Thread 2: coordinates (pixels -> mm) ─────────────────────────────────────
def thread_coord_translation(pipeline: VisionPipeline) -> None:
    global latest_shapes
    print("[Thread 2: Coordinates] Started.")
    while not stop_event.is_set():
        try:
            frame, detections = coord_translation_queue.get(timeout=0.2)
        except queue.Empty:
            continue

        if pipeline.homography_matrix is None:
            pipeline.update_homography(frame)

        translated: List[DetectedShape] = []
        for obj in detections:
            coords = pipeline.pixel_to_mm(obj.cx_px, obj.cy_px)
            if coords is not None:
                obj.x_mm, obj.y_mm = coords
            translated.append(obj)

        with shapes_lock:
            latest_shapes = translated
    print("[Thread 2: Coordinates] Stopped.")


# ── Thread 3: robot server ───────────────────────────────────────────────────
def thread_robot_server(robot_link: RobotLink) -> None:
    print("[Thread 3: Server] Started.")
    try:
        robot_link.start()
    except Exception as err:
        print(f"[Thread 3: Server] Socket startup notice: {err}")

    while not stop_event.is_set():
        try:
            command = robot_command_queue.get(timeout=0.2)
        except queue.Empty:
            continue
        print(f"[Thread 3] Sending to ABB robot: {command.strip()}")
        robot_link.send(command)

    robot_link.close()
    print("[Thread 3: Server] Stopped.")


# ── Mouse click (click to pick) ──────────────────────────────────────────────
def on_mouse_click(event, mouse_x, mouse_y, flags, param) -> None:
    global selected_shape_info
    if event != cv2.EVENT_LBUTTONDOWN:
        return

    with shapes_lock:
        current = list(latest_shapes)

    for s in current:
        bx, by, bw, bh = s.bbox
        if bx <= mouse_x <= bx + bw and by <= mouse_y <= by + bh:
            print(f"\n[CLICK] Selected {s.shape} at X={s.x_mm}mm, Y={s.y_mm}mm")
            robot_command_queue.put(s.to_robot_command())
            with selected_shape_lock:
                selected_shape_info = {"shape": s.shape, "bbox": s.bbox, "expire_time": time.time() + 1.5}
            break


# ── Thread 4: GUI (main) ─────────────────────────────────────────────────────
def main() -> None:
    script_dir = os.path.dirname(os.path.abspath(__file__))
    image_path = os.path.join(script_dir, "test.png")
    if not os.path.exists(image_path):
        image_path = os.path.join(script_dir, "..", "ReWorks", "test.png")

    print("Initializing vision pipeline...")
    pipeline = VisionPipeline()
    robot_link = RobotLink(host="0.0.0.0", port=5000)

    # Try the live camera, otherwise use the static test image.
    live_camera = None
    try:
        live_camera = BaslerCamera()
        live_camera.open()
    except Exception as err:
        print(f"[CAMERA] No live camera ({err}). Using '{image_path}' as a simulated feed.")
        live_camera = None

    static_frame = cv2.imread(image_path) if os.path.exists(image_path) else None
    if static_frame is None and live_camera is None:
        print(f"Error: neither a camera nor '{image_path}' could be opened!")
        return

    for target, args in (
        (thread_ai_inference, (pipeline,)),
        (thread_coord_translation, (pipeline,)),
        (thread_robot_server, (robot_link,)),
    ):
        threading.Thread(target=target, args=args, daemon=True).start()

    window = "ABB Robot Vision & Picking Control"
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window, 1240, 720)
    cv2.setMouseCallback(window, on_mouse_click)

    print("\n=========================================================")
    print(" SYSTEM RUNNING (4 threads)")
    print(" - CLICK a shape to pick it.  HOTKEYS: [S/T/C/H]")
    print(" - Press [Q] to exit.")
    print("=========================================================\n")

    last_snapshot = 0.0
    try:
        while not stop_event.is_set():
            if live_camera is not None and live_camera.isOpen:
                frame = live_camera.grabFrame()
                if frame is None:
                    continue
            else:
                frame = static_frame.copy()
                time.sleep(0.03)

            now = time.time()
            if now - last_snapshot > 0.2:
                if frame_snapshot_queue.empty():
                    frame_snapshot_queue.put(frame.copy())
                last_snapshot = now

            with shapes_lock:
                current_shapes = list(latest_shapes)

            display = pipeline.draw_hud(frame, current_shapes)

            # Gold frame around the last clicked shape.
            with selected_shape_lock:
                if selected_shape_info and time.time() < selected_shape_info["expire_time"]:
                    bx, by, bw, bh = selected_shape_info["bbox"]
                    cv2.rectangle(display, (bx, by), (bx + bw, by + bh), (0, 215, 255), 4)
                    cv2.putText(
                        display,
                        f">> SENDING {selected_shape_info['shape'].upper()} TO ROBOT <<",
                        (bx, max(30, by - 20)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 215, 255), 2, cv2.LINE_AA,
                    )
                elif selected_shape_info and time.time() >= selected_shape_info["expire_time"]:
                    selected_shape_info = None

            status = (
                f"ROBOT: {'CONNECTED' if robot_link.connected else 'WAITING'} | "
                f"MODE: {'LIVE CAMERA' if live_camera else 'TEST IMAGE'} | "
                f"CLICK OR PRESS [S/T/C/H] TO PICK"
            )
            banner = (0, 180, 0) if robot_link.connected else (0, 120, 255)
            cv2.rectangle(display, (0, 0), (display.shape[1], 45), (30, 30, 30), -1)
            cv2.putText(display, status, (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.65, banner, 2, cv2.LINE_AA)

            cv2.imshow(window, display)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if key in (ord("s"), ord("t"), ord("c"), ord("h")):
                wanted = {ord("s"): "Star", ord("t"): "Triangle", ord("c"): "Circle", ord("h"): "Hexagon"}[key]
                for s in current_shapes:
                    if s.shape.lower() == wanted.lower():
                        print(f"\n[HOTKEY] Selected {s.shape} at X={s.x_mm}, Y={s.y_mm}")
                        robot_command_queue.put(s.to_robot_command())
                        with selected_shape_lock:
                            selected_shape_info = {"shape": s.shape, "bbox": s.bbox, "expire_time": time.time() + 1.5}
                        break
    finally:
        stop_event.set()
        if live_camera is not None:
            live_camera.close()
        cv2.destroyAllWindows()
        print("Application shut down cleanly.")


if __name__ == "__main__":
    main()
