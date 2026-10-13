# Author Nils Wikström niwi0007

"""
main.py - första versionen (vision.py + den gamla modellen). Den nya är main_v2.py.
4 threads:
  1. AI-thread: kör nätverket (3 heads) på bilder från kameran.
  2. Koordinat-thread: räknar om pixlar till mm på bordet.
  3. Server-thread: skickar pick-kommandon till ABB-roboten via TCP-socket.
  4. GUI (main thread): visar kameran live med HUD, klicka på en form för att plocka den.
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

# KÖER OCH DATA SOM DELAS MELLAN THREADS
frame_snapshot_queue: queue.Queue = queue.Queue(maxsize=1)
coord_translation_queue: queue.Queue = queue.Queue(maxsize=2)
robot_command_queue: queue.Queue = queue.Queue()

shapes_lock = threading.Lock()
latest_shapes: List[DetectedShape] = []

selected_shape_lock = threading.Lock()
selected_shape_info: Optional[dict] = None  # formen som ska lysa guld i GUI:t efter ett klick

stop_event = threading.Event()


# THREAD 1: AI (kör nätverket)
def thread_ai_inference(pipeline: VisionPipeline):
    """Hämtar bilder från kön hela tiden och kör ONNX-modellen på dem."""
    print("[Thread 1: AI Inference] Started.")
    while not stop_event.is_set():
        try:
            frame = frame_snapshot_queue.get(timeout=0.2)
        except queue.Empty:
            continue

        # Klipp ut en crop per form (HSV-mask, svart bakgrund)
        rois = pipeline.extract_rois(frame)

        # Kör nätverket på varje crop
        raw_detections = []
        for bbox, roi in rois:
            det = pipeline.infer_roi(bbox, roi)
            if det is not None:
                raw_detections.append(det)

        # Skicka vidare till koordinat-threaden
        if not coord_translation_queue.full():
            coord_translation_queue.put((frame, raw_detections))

    print("[Thread 1: AI Inference] Stopped.")


# THREAD 2: KOORDINATER (pixlar -> mm)
def thread_coord_translation(pipeline: VisionPipeline):
    """Räknar om formernas centre från pixlar till mm på bordet."""
    global latest_shapes
    print("[Thread 2: Coordinate Translation] Started.")

    while not stop_event.is_set():
        try:
            frame, detections = coord_translation_queue.get(timeout=0.2)
        except queue.Empty:
            continue

        # Se till att ArUco-kalibreringen finns
        if pipeline.homography_matrix is None:
            pipeline.update_homography(frame)

        # Räkna om X/Y i mm
        translated_shapes: List[DetectedShape] = []
        for obj in detections:
            coords_mm = pipeline.pixel_to_mm(obj.cx_px, obj.cy_px)
            if coords_mm is not None:
                obj.x_mm = coords_mm[0]
                obj.y_mm = coords_mm[1]
            translated_shapes.append(obj)

        # Ge de senaste formerna till GUI:t (med lock)
        with shapes_lock:
            latest_shapes = translated_shapes

    print("[Thread 2: Coordinate Translation] Stopped.")


# THREAD 3: SERVER (ABB-roboten)
def thread_robot_server(robot_link: RobotLink):
    """Sköter TCP-anslutningen och skickar pick-kommandon till ABB-roboten."""
    print("[Thread 3: Server Communication] Started.")

    # Starta socket-servern (väntar på att roboten ansluter)
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


# MUSKLICK (klicka för att plocka)
def on_mouse_click(event, mouse_x, mouse_y, flags, param):
    """Anropas av OpenCV när man klickar i videofönstret."""
    global selected_shape_info

    if event == cv2.EVENT_LBUTTONDOWN:
        with shapes_lock:
            current_shapes = list(latest_shapes)

        for s in current_shapes:
            bx, by, bw, bh = s.bbox
            # Klickade man inuti den här formens bounding box?
            if bx <= mouse_x <= (bx + bw) and by <= mouse_y <= (by + bh):
                print(f"\n[USER CLICK] Selected {s.shape} at X={s.x_mm}mm, Y={s.y_mm}mm")
                cmd = s.to_robot_command()
                robot_command_queue.put(cmd)

                with selected_shape_lock:
                    selected_shape_info = {
                        "shape": s.shape,
                        "bbox": s.bbox,
                        "expire_time": time.time() + 1.5,  # lys i 1.5 s
                    }
                break


# THREAD 4: GUI (main thread)
def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(script_dir, "..", "MISC", "shape_3head_model.onnx")
    if not os.path.exists(model_path):
        model_path = os.path.join(script_dir, "shape_3head_model.onnx")

    image_path = os.path.join(script_dir, "test.png")

    # 1. Starta vision pipeline och robot-länken
    print("Initializing Vision Pipeline...")
    pipeline = VisionPipeline(model_path=model_path)
    robot_link = RobotLink(host="0.0.0.0", port=5000)

    # 2. Försök ansluta Basler-kameran (annars används test.png)
    live_camera = None
    try:
        live_camera = BaslerCamera()
        live_camera.open()
        print("[CAMERA] Live Basler Camera connected successfully!")
    except Exception as e:
        print(f"[CAMERA] Basler camera not detected ({e}). Using '{image_path}' as simulated feed.")
        live_camera = None

    # Ladda test-bilden
    static_frame = cv2.imread(image_path)
    if static_frame is None and live_camera is None:
        print(f"Error: Neither Basler camera nor '{image_path}' could be opened!")
        return

    # 3. Starta threads 1, 2 och 3 i bakgrunden
    t1 = threading.Thread(target=thread_ai_inference, args=(pipeline,), daemon=True)
    t2 = threading.Thread(target=thread_coord_translation, args=(pipeline,), daemon=True)
    t3 = threading.Thread(target=thread_robot_server, args=(robot_link,), daemon=True)

    t1.start()
    t2.start()
    t3.start()

    # 4. Skapa OpenCV-fönstret
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
            # Hämta senaste framen (kameran eller test-bilden)
            if live_camera is not None and live_camera.isOpen:
                frame = live_camera.grabFrame()
                if frame is None:
                    continue
            else:
                frame = static_frame.copy()
                time.sleep(0.03)  # ca 30 FPS när test-bilden används

            # Var 200:e ms: skicka en bild till AI-threaden
            current_time = time.time()
            if current_time - last_snapshot_time > 0.2:
                if frame_snapshot_queue.empty():
                    frame_snapshot_queue.put(frame.copy())
                last_snapshot_time = current_time

            # Hämta de senaste formerna från thread 2 (med lock)
            with shapes_lock:
                current_shapes = list(latest_shapes)

            # Rita HUD
            display_frame = pipeline.draw_hud(frame, current_shapes)

            # Guldram runt formen man klickade på
            with selected_shape_lock:
                if selected_shape_info and time.time() < selected_shape_info["expire_time"]:
                    bx, by, bw, bh = selected_shape_info["bbox"]
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
                elif selected_shape_info and time.time() >= selected_shape_info["expire_time"]:
                    selected_shape_info = None

            # Statusrad överst i fönstret
            status_text = (
                f"ROBOT: {'CONNECTED' if robot_link.connected else 'WAITING'} | "
                f"MODE: {'LIVE CAMERA' if live_camera else 'TEST IMAGE'} | "
                f"CLICK SHAPE OR PRESS [S/T/C/H] TO PICK"
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

            # Tangenter
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q") or key == 27:  # 'q' eller ESC
                break
            elif key in (ord("s"), ord("t"), ord("c"), ord("h")):
                key_map = {ord("s"): "Star", ord("t"): "Triangle", ord("c"): "Circle", ord("h"): "Hexagon"}
                target_shape_name = key_map[key]

                # Hitta första formen av den sorten och skicka den
                for s in current_shapes:
                    if s.shape.lower() == target_shape_name.lower():
                        print(f"\n[HOTKEY] Selected {s.shape} at X={s.x_mm}, Y={s.y_mm}")
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
