# Author Nils Wikström niwi0007

"""
main.py - vision och robotstyrning med 4 threads. Starta med:  python main.py
  1. AI-thread: kör nätverket (4 heads: class, rotation, centre, colour) på bilder från kameran.
  2. Koordinat-thread: räknar om pixlar till mm på bordet.
  3. Server-thread: skickar pick-kommandon till ABB-roboten via TCP-socket.
  4. GUI (main thread, Tkinter): kameran live med HUD och knappar.
     - Free pick: klicka på valfri bit för att plocka den.
     - Colour pick: välj en färg med knapparna; bara bitar i den färgen visas och kan plockas.
       "Queue all" sparar alla bitar i färgen i en kö (med nummer i bilden). Start plockar dem i ordning,
       "Pick now" (eller klick på en bit) flyttar en bit först i kön.
  För våran variant så måste OK knapp tryckas på med
"""

import os
import queue
import threading
import time
from typing import List, Optional

import tkinter as tk

import cv2
import numpy as np
from PIL import Image, ImageTk

from camera import BaslerCamera
from robot_link import RobotLink
from vision import DetectedShape, VisionPipeline

# KÖER OCH DATA SOM DELAS MELLAN THREADS
frame_snapshot_queue: queue.Queue = queue.Queue(maxsize=1)
coord_translation_queue: queue.Queue = queue.Queue(maxsize=2)
robot_command_queue: queue.Queue = queue.Queue()

shapes_lock = threading.Lock()
latest_shapes: List[DetectedShape] = []

stop_event = threading.Event()
latest_inference_time_ms: float = 0.0


# THREAD 1: AI (kör nätverket)
def thread_ai_inference(pipeline: VisionPipeline):
    """Hämtar bilder från kön hela tiden och kör ONNX-modellen på dem."""
    global latest_inference_time_ms
    print("[Thread 1: AI Inference] Started.")
    while not stop_event.is_set():
        try:
            frame = frame_snapshot_queue.get(timeout=0.2)
        except queue.Empty:
            continue

        t_start = time.perf_counter()
        # Klipp ut en crop per form (HSV-mask, svart bakgrund)
        rois = pipeline.extract_rois(frame)

        # Kör nätverket på varje crop
        raw_detections = []
        for bbox, roi in rois:
            det = pipeline.infer_roi(bbox, roi)
            if det is not None:
                raw_detections.append(det)

        latest_inference_time_ms = (time.perf_counter() - t_start) * 1000.0

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


# Kön plockas en bit i taget. Roboten svarar inte när den är klar, så vi väntar så här länge
# mellan kommandona (en pick tar ungefär så lång tid). Kan ändras i GUI:t.
PICK_ALL_DELAY_S = 20.0

COLOURS = ["red", "green", "blue"]                                  # samma ordning som colour-headen
BUTTON_COLOURS = {"red": "#dc2626", "green": "#16a34a", "blue": "#2563eb"}
DISPLAY_SIZE = (1000, 820)                                          # kamerabilden får högst vara så här stor i fönstret (px)


def report_selection(s: DetectedShape, source: str, pipeline: VisionPipeline) -> None:
    """Skriver ut den valda formen och kommandot som skickas, och alla candidates igen nästa frame."""
    print(f"\n[{source}] Selected {s.colour} ({s.colour_confidence*100:.0f}%) {s.shape} ({s.confidence*100:.0f}%) "
          f"at X={s.x_mm} mm, Y={s.y_mm} mm, RZ={s.rz_deg} deg")
    print(f"[{source}] Command: {s.to_robot_command().strip()}")
    pipeline.log_next_frame()


class PickerApp:
    """Tkinter-fönstret: kamerabilden till vänster, knappar, kö och inställningar till höger."""

    def __init__(self, root, pipeline, robot_link, live_camera, static_frame):
        self.root = root
        self.pipeline = pipeline
        self.robot_link = robot_link
        self.live_camera = live_camera
        self.static_frame = static_frame

        self.mode = tk.StringVar(value="free")      # "free" eller "colour"
        self.colour: Optional[str] = None           # vald färg i colour-läget
        self.queue: List[DetectedShape] = []        # sparade bitar, plockas uppifrån och ner
        self.running = False                        # plockar kön just nu?
        self.next_pick_time = 0.0                   # tidigast när nästa kommando får skickas
        self.highlight: Optional[tuple] = None      # (bbox, shape, när den slutar lysa)
        self.shapes: List[DetectedShape] = []       # det som visas just nu
        self.scale = 1.0                            # fönsterpixlar per kamerapixel
        self.last_snapshot = 0.0
        self.photo = None                           # Tkinter måste ha kvar en referens till bilden

        root.title("ABB Robot Vision & Picking Control")
        self.video = tk.Label(root, bg="black", cursor="hand2")
        self.video.grid(row=0, column=0, padx=6, pady=6)
        self.video.bind("<Button-1>", self.on_click)
        panel = tk.Frame(root, padx=10, pady=6)
        panel.grid(row=0, column=1, sticky="n")

        # Läge
        tk.Label(panel, text="Mode", font=("", 12, "bold")).pack(anchor="w")
        for value, text in (("free", "Free pick (click any piece)"), ("colour", "Colour pick (queue)")):
            tk.Radiobutton(panel, text=text, variable=self.mode, value=value,
                           command=self.on_mode_change).pack(anchor="w")

        # Färgknappar
        tk.Label(panel, text="Colour", font=("", 12, "bold")).pack(anchor="w", pady=(10, 0))
        row = tk.Frame(panel); row.pack(anchor="w")
        self.colour_buttons = {}
        for c in COLOURS:
            b = tk.Button(row, text=c.capitalize(), width=7, bg=BUTTON_COLOURS[c], fg="white",
                          activebackground=BUTTON_COLOURS[c], command=lambda c=c: self.select_colour(c))
            b.pack(side="left", padx=2)
            self.colour_buttons[c] = b

        # Kön
        tk.Label(panel, text="Queue", font=("", 12, "bold")).pack(anchor="w", pady=(10, 0))
        self.queue_button = tk.Button(panel, text="Queue all", width=30, command=self.queue_all)
        self.queue_button.pack(anchor="w", pady=(0, 3))
        self.queue_list = tk.Listbox(panel, height=7, width=40, font=("Courier", 9), activestyle="none")
        self.queue_list.pack(anchor="w")
        self.queue_list.bind("<Double-Button-1>", lambda e: self.pick_now())
        row = tk.Frame(panel); row.pack(anchor="w", pady=(3, 0))
        self.start_button = tk.Button(row, text="Start", width=6, command=self.start)
        self.pick_now_button = tk.Button(row, text="Pick now", width=8, command=self.pick_now)
        self.stop_button = tk.Button(row, text="Stop", width=5, command=self.stop)
        self.clear_button = tk.Button(row, text="Clear", width=5, command=self.clear)
        for b in (self.start_button, self.pick_now_button, self.stop_button, self.clear_button):
            b.pack(side="left", padx=1)
        self.queue_controls = [self.queue_button, self.start_button, self.pick_now_button,
                               self.stop_button, self.clear_button]

        # Inställningar som beror på uppställningen
        tk.Label(panel, text="Settings", font=("", 12, "bold")).pack(anchor="w", pady=(10, 0))
        self.min_area = self._slider(panel, "Min piece area (px)", 5000, 250000, 1000, pipeline.min_area,
                                     lambda v: setattr(pipeline, "min_area", int(float(v))))
        self.max_area = self._slider(panel, "Max piece area (px)", 5000, 250000, 1000, pipeline.max_area,
                                     lambda v: setattr(pipeline, "max_area", int(float(v))))
        self.min_circle = self._slider(panel, "Min circle area (px, magnets are smaller)", 5000, 250000, 1000,
                                       pipeline.min_circle_area,
                                       lambda v: setattr(pipeline, "min_circle_area", int(float(v))))
        self.delay = self._slider(panel, "Queue: seconds between picks", 5, 90, 1, PICK_ALL_DELAY_S, None)

        # Status
        tk.Label(panel, text="Status", font=("", 12, "bold")).pack(anchor="w", pady=(10, 0))
        self.status = tk.Label(panel, justify="left", anchor="w", font=("", 10), width=34)
        self.status.pack(anchor="w")
        self.last_command = tk.Label(panel, justify="left", anchor="w", font=("Courier", 9), width=40,
                                     wraplength=260, fg="#444")
        self.last_command.pack(anchor="w", pady=(6, 0))

        root.protocol("WM_DELETE_WINDOW", self.quit)
        root.bind("<Escape>", lambda e: self.quit())
        self.on_mode_change()
        self.tick()

    @staticmethod
    def _slider(parent, text, lo, hi, step, value, command):
        tk.Label(parent, text=text).pack(anchor="w")
        scale = tk.Scale(parent, from_=lo, to=hi, resolution=step, orient="horizontal", length=260,
                         command=command)
        scale.set(value)
        scale.pack(anchor="w")
        return scale

    # Vad som syns och får plockas
    def visible(self, s: DetectedShape) -> bool:
        """Free pick: alla. Colour pick: bara vald färg (alla tills en färg är vald)."""
        return self.mode.get() == "free" or self.colour is None or s.colour == self.colour

    def on_mode_change(self):
        colour_mode = self.mode.get() == "colour"
        for b in self.colour_buttons.values():
            b.config(state="normal" if colour_mode else "disabled")
        if not colour_mode:
            self.colour = None
            self.clear()
        self.update_buttons()

    def select_colour(self, colour: str):
        if colour != self.colour:
            self.clear()                            # ny färg = ny kö, så inga bitar av den gamla färgen skickas
        self.colour = colour
        print(f"\n[COLOUR] Now showing: {colour}")
        self.update_buttons()

    def update_buttons(self):
        for c, b in self.colour_buttons.items():
            b.config(relief="sunken" if c == self.colour else "raised",
                     text=f"[{c.capitalize()}]" if c == self.colour else c.capitalize())
        ready = self.mode.get() == "colour" and self.colour is not None
        for b in self.queue_controls:
            b.config(state="normal" if ready else "disabled")
        self.queue_button.config(text=f"Queue all {self.colour}" if ready else "Queue all (choose a colour)")
        self.start_button.config(relief="sunken" if self.running else "raised")

    # Kön
    def refresh_list(self):
        self.queue_list.delete(0, "end")
        for i, s in enumerate(self.queue):
            self.queue_list.insert("end", f"{i + 1}. {s.shape:<8} X{s.x_mm:7.1f} Y{s.y_mm:7.1f} RZ{s.rz_deg:+6.1f}")
        if self.queue:
            self.queue_list.itemconfig(0, bg="#fde68a")          # gul = nästa bit
        self.update_buttons()

    def queue_all(self):
        # Spara alla synliga bitar (= vald färg) med sina positioner. Bitarna ligger still, så positionerna gäller.
        self.queue = [s for s in self.shapes if self.visible(s)]
        self.running = False
        print(f"\n[QUEUE] {len(self.queue)} {self.colour} saved: " + ", ".join(s.shape for s in self.queue))
        self.refresh_list()

    def start(self):
        if self.queue:
            self.running = True
            print(f"\n[QUEUE] Started, {len(self.queue)} to pick")
        self.update_buttons()

    def stop(self):
        if self.running:
            print(f"\n[QUEUE] Paused, {len(self.queue)} left in the queue")
        self.running = False
        self.update_buttons()

    def clear(self):
        if self.queue:
            print(f"\n[QUEUE] Cleared, {len(self.queue)} not sent")
        self.queue, self.running = [], False
        self.refresh_list()

    def jump(self, s: DetectedShape):
        """Flyttar biten först i kön och startar kön. Den skickas så fort pausen efter förra biten har gått."""
        if s in self.queue:
            self.queue.remove(s)
        self.queue.insert(0, s)
        self.running = True
        wait = max(0.0, self.next_pick_time - time.time())
        print(f"\n[QUEUE] {s.colour} {s.shape} jumps the queue" + (f", sent in {wait:.0f} s" if wait else ", sent now"))
        self.refresh_list()

    def pick_now(self):
        sel = self.queue_list.curselection()
        if sel:
            self.jump(self.queue[sel[0]])

    # Plocka
    def send(self, s: DetectedShape, source: str) -> bool:
        # Utan ArUco-markör är X/Y pixlar och RZ = 0: skicka aldrig det till roboten
        if not s.has_mm or self.pipeline.homography_matrix is None:
            print(f"\n[{source}] NOT SENT: no ArUco marker, {s.colour} {s.shape} has no mm position")
            self.last_command.config(text="NOT SENT: no ArUco marker found", fg="#dc2626")
            return False
        report_selection(s, source, self.pipeline)
        cmd = s.to_robot_command()
        robot_command_queue.put(cmd)
        self.last_command.config(text="Sent: " + cmd.strip().replace(";", "; "), fg="#444")
        self.highlight = (s.bbox, s.shape, time.time() + 1.5)
        return True

    def on_click(self, event):
        x, y = event.x / self.scale, event.y / self.scale         # fönsterpixlar -> kamerapixlar
        for s in self.shapes:
            bx, by, bw, bh = s.bbox
            if not (bx <= x <= bx + bw and by <= y <= by + bh):
                continue
            if self.mode.get() == "free":
                self.send(s, "USER CLICK")                         # free pick: skicka direkt
            elif self.colour is not None:
                # Colour pick: klickad bit först i kön (samma bit om den redan är sparad, annars läggs den till)
                queued = min(self.queue, key=lambda q: np.hypot(q.cx_px - s.cx_px, q.cy_px - s.cy_px), default=None)
                near = queued is not None and np.hypot(queued.cx_px - s.cx_px, queued.cy_px - s.cy_px) < max(bw, bh) / 2
                self.jump(queued if near else s)
            return

    # En frame
    def tick(self):
        if stop_event.is_set():
            return
        if self.live_camera is not None and self.live_camera.isOpen:
            frame = self.live_camera.grabFrame()
        else:
            frame = self.static_frame.copy()
        if frame is not None:
            self.show(frame)
        self.root.after(15, self.tick)

    def show(self, frame):
        # Var 200:e ms: skicka en bild till AI-threaden
        now = time.time()
        if now - self.last_snapshot > 0.2 and frame_snapshot_queue.empty():
            frame_snapshot_queue.put(frame.copy())
            self.last_snapshot = now

        with shapes_lock:
            all_shapes = list(latest_shapes)
        self.shapes = [s for s in all_shapes if self.visible(s)]   # andra färger: ingen kontur, går inte att klicka

        # Kön: skicka nästa bit när pausen efter förra har gått
        if self.running and self.queue and now >= self.next_pick_time:
            if not self.send(self.queue[0], f"QUEUE ({len(self.queue) - 1} left)"):
                self.stop()                         # ingen markör: pausa, biten ligger kvar först i kön
                return
            self.queue.pop(0)
            self.next_pick_time = now + float(self.delay.get())
            if not self.queue:
                self.running = False
                print("\n[QUEUE] Done")
            self.refresh_list()

        display = self.pipeline.draw_hud(frame, self.shapes)
        for i, s in enumerate(self.queue):                          # könummer på varje sparad bit
            p = (int(s.cx_px) - 30, int(s.cy_px) + 70)
            cv2.putText(display, f"#{i + 1}", p, cv2.FONT_HERSHEY_SIMPLEX, 2.0, (0, 0, 0), 10, cv2.LINE_AA)
            cv2.putText(display, f"#{i + 1}", p, cv2.FONT_HERSHEY_SIMPLEX, 2.0,
                        (0, 215, 255) if i == 0 else (255, 255, 255), 4, cv2.LINE_AA)
        if self.highlight and now < self.highlight[2]:
            (bx, by, bw, bh), shape, _ = self.highlight
            cv2.rectangle(display, (bx, by), (bx + bw, by + bh), (0, 215, 255), 6)
            cv2.putText(display, f">> SENDING {shape.upper()} TO ROBOT <<", (bx, max(40, by - 25)),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 215, 255), 3, cv2.LINE_AA)

        # Till Tkinter: krymp, BGR -> RGB, PhotoImage
        self.scale = min(DISPLAY_SIZE[0] / display.shape[1], DISPLAY_SIZE[1] / display.shape[0])
        small = cv2.resize(display, None, fx=self.scale, fy=self.scale, interpolation=cv2.INTER_AREA)
        self.photo = ImageTk.PhotoImage(Image.fromarray(cv2.cvtColor(small, cv2.COLOR_BGR2RGB)))
        self.video.config(image=self.photo)

        counts = {c: sum(s.colour == c for s in all_shapes) for c in COLOURS}
        if not self.queue:
            queue_text = "empty"
        elif self.running:
            wait = max(0.0, self.next_pick_time - now)
            queue_text = f"{len(self.queue)} left, next in {wait:.0f} s" if wait else f"{len(self.queue)} left, sending"
        else:
            queue_text = f"{len(self.queue)} saved, paused (Start)"
        self.status.config(text=(
            f"Robot:   {'CONNECTED' if self.robot_link.connected else 'waiting...'}\n"
            f"Infer:   {latest_inference_time_ms:.1f} ms\n"
            f"Camera:  {'live' if self.live_camera else 'test image'}\n"
            f"Marker:  {'found' if self.pipeline.homography_matrix is not None else 'NOT FOUND'}\n"
            f"In view: {counts['red']} red, {counts['green']} green, {counts['blue']} blue\n"
            f"Queue:   {queue_text}"))

    def quit(self):
        stop_event.set()
        if self.live_camera is not None:
            self.live_camera.close()
        self.root.destroy()
        print("Application shut down cleanly.")


# THREAD 4: GUI (main thread)
def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(script_dir, "models", "crpnet.onnx")
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

    static_frame = cv2.imread(image_path)
    if static_frame is None and live_camera is None:
        print(f"Error: Neither Basler camera nor '{image_path}' could be opened!")
        return

    # 3. Starta threads 1, 2 och 3 i bakgrunden
    workers = [threading.Thread(target=thread_ai_inference, args=(pipeline,), daemon=True),
               threading.Thread(target=thread_coord_translation, args=(pipeline,), daemon=True)]
    for t in workers:
        t.start()
    threading.Thread(target=thread_robot_server, args=(robot_link,), daemon=True).start()

    # 4. Tkinter-fönstret (main thread)
    root = tk.Tk()
    PickerApp(root, pipeline, robot_link, live_camera, static_frame)
    print("\nSYSTEM RUNNING (4 threads). Free pick: click a piece. Colour pick: choose a colour, Queue all, Start; click a piece or Pick now to jump the queue.")
    root.mainloop()

    # Vänta tills AI- och koordinat-threaden har slutat, annars kan programmet krascha när det stängs
    # mitt i en körning av modellen. (Server-threaden kan vänta på roboten för evigt, den väntar vi inte på.)
    for t in workers:
        t.join(timeout=3)


if __name__ == "__main__":
    main()
