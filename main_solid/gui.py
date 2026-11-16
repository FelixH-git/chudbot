# Author Nils Wikström niwi0007
"""
PickerApp: Tkinter-fönstret. Kamerabilden till vänster, knappar, kö och inställningar till höger.
Bara GUI: vad som syns och vad ett klick gör bestäms av läget, kön av PickQueue och sändningen
av PickSender (alla i pick_operations), ritandet av HudRenderer.
"""
import queue
import threading
import time
from typing import List, Optional, Tuple

import tkinter as tk

import cv2
from PIL import Image, ImageTk

from detection import DetectedShape
from frame_source import FrameSource
from hud import HudRenderer
from pick_operations import MODES, FreePick, PickMode, PickQueue, PickSender
from pipeline import VisionPipeline
from robot_operations import RobotLink
from settings import COLOUR_BUTTONS, DISPLAY_SIZE
from workers import LatestShapes


class PickerApp:
    def __init__(self, root: tk.Tk, source: FrameSource, pipeline: VisionPipeline, hud: HudRenderer,
                 sender: PickSender, robot_link: RobotLink, frames: queue.Queue, latest: LatestShapes,
                 stop: threading.Event, colours: List[str]):
        self.root = root
        self.source = source
        self.pipeline = pipeline
        self.hud = hud
        self.sender = sender
        self.robot_link = robot_link
        self.frames = frames                        # bilder till thread 1
        self.latest = latest                        # bitar från thread 2
        self.stop_event = stop

        self.mode: PickMode = FreePick()
        self.queue = PickQueue()
        self.highlight: Optional[Tuple[DetectedShape, float]] = None   # (bit, när den slutar lysa)
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
        self._heading(panel, "Mode", top=0)
        self.mode_var = tk.StringVar(value="free")
        for key, mode_cls in MODES.items():
            tk.Radiobutton(panel, text=mode_cls.label, variable=self.mode_var, value=key,
                           command=self.on_mode_change).pack(anchor="w")

        # Färgknappar (samma ordning som colour-headen)
        self._heading(panel, "Colour")
        row = tk.Frame(panel); row.pack(anchor="w")
        self.colour_buttons = {}
        for c in colours:
            b = tk.Button(row, text=c.capitalize(), width=7, bg=COLOUR_BUTTONS[c], fg="white",
                          activebackground=COLOUR_BUTTONS[c], command=lambda c=c: self.select_colour(c))
            b.pack(side="left", padx=2)
            self.colour_buttons[c] = b

        # Kön
        self._heading(panel, "Queue")
        self.queue_button = tk.Button(panel, text="Queue all", width=30, command=self.queue_all)
        self.queue_button.pack(anchor="w", pady=(0, 3))
        self.queue_list = tk.Listbox(panel, height=7, width=40, font=("Courier", 9), activestyle="none")
        self.queue_list.pack(anchor="w")
        self.queue_list.bind("<Double-Button-1>", lambda e: self.pick_now())
        row = tk.Frame(panel); row.pack(anchor="w", pady=(3, 0))
        buttons = [tk.Button(row, text=text, width=width, command=command) for text, width, command in
                   (("Start", 6, self.start), ("Pick now", 8, self.pick_now), ("Stop", 5, self.stop),
                    ("Clear", 5, self.clear))]
        for b in buttons:
            b.pack(side="left", padx=1)
        self.start_button = buttons[0]
        self.queue_controls = [self.queue_button] + buttons      # bara aktiva när en färg är vald

        # Inställningar som beror på uppställningen
        seg = pipeline.segmenter
        self._heading(panel, "Settings")
        self._slider(panel, "Min piece area (px)", 5000, 250000, 1000, seg.min_area,
                     lambda v: setattr(seg, "min_area", int(float(v))))
        self._slider(panel, "Max piece area (px)", 5000, 250000, 1000, seg.max_area,
                     lambda v: setattr(seg, "max_area", int(float(v))))
        self._slider(panel, "Min circle area (px, magnets are smaller)", 5000, 250000, 1000,
                     pipeline.min_circle_area, lambda v: setattr(pipeline, "min_circle_area", int(float(v))))
        self._slider(panel, "Queue: seconds between picks", 5, 90, 1, self.queue.delay_s,
                     lambda v: setattr(self.queue, "delay_s", float(v)))

        # Status
        self._heading(panel, "Status")
        self.status = tk.Label(panel, justify="left", anchor="w", font=("", 10), width=34)
        self.status.pack(anchor="w")
        self.last_command = tk.Label(panel, justify="left", anchor="w", font=("Courier", 9), width=40,
                                     wraplength=260, fg="#444")
        self.last_command.pack(anchor="w", pady=(6, 0))

        # Quit-knapp
        self.quit_button = tk.Button(panel, text="Quit Application", font=("", 10, "bold"),
                                     bg="#ef4444", fg="white", activebackground="#dc2626", activeforeground="white",
                                     width=28, command=self.quit)
        self.quit_button.pack(anchor="w", pady=(14, 0))

        root.protocol("WM_DELETE_WINDOW", self.quit)
        root.bind("<Escape>", lambda e: self.quit())
        self.on_mode_change()
        self.tick()

    @staticmethod
    def _heading(parent, text, top=10):
        tk.Label(parent, text=text, font=("", 12, "bold")).pack(anchor="w", pady=(top, 0))

    @staticmethod
    def _slider(parent, text, lo, hi, step, value, command):
        tk.Label(parent, text=text).pack(anchor="w")
        scale = tk.Scale(parent, from_=lo, to=hi, resolution=step, orient="horizontal", length=260, command=command)
        scale.set(value)
        scale.pack(anchor="w")

    # Läge och färg
    def on_mode_change(self):
        mode_cls = MODES[self.mode_var.get()]
        if type(self.mode) is not mode_cls:
            self.mode = mode_cls()
            self.clear()                            # nytt läge = ny kö
        for b in self.colour_buttons.values():
            b.config(state="normal" if self.mode.picks_colour else "disabled")
        self.refresh_list()

    def select_colour(self, colour: str):
        if colour != self.mode.colour:
            self.clear()                            # ny färg = ny kö, så inga bitar av den gamla färgen skickas
        self.mode.colour = colour
        print(f"\n[COLOUR] Now showing: {colour}")
        self.update_buttons()

    def update_buttons(self):
        for c, b in self.colour_buttons.items():
            b.config(relief="sunken" if c == self.mode.colour else "raised",
                     text=f"[{c.capitalize()}]" if c == self.mode.colour else c.capitalize())
        ready = self.mode.uses_queue
        for b in self.queue_controls:
            b.config(state="normal" if ready else "disabled")
        self.queue_button.config(text=f"Queue all {self.mode.colour}" if ready else "Queue all (choose a colour)")
        self.start_button.config(relief="sunken" if self.queue.running else "raised")

    # Kön
    def refresh_list(self):
        self.queue_list.delete(0, "end")
        for i, s in enumerate(self.queue.items):
            self.queue_list.insert("end", f"{i + 1}. {s.shape:<8} X{s.x_mm:7.1f} Y{s.y_mm:7.1f} RZ{s.rz_deg:+6.1f}")
        if self.queue.items:
            self.queue_list.itemconfig(0, bg="#fde68a")          # gul = nästa bit
        self.update_buttons()

    def queue_all(self):
        self.queue.fill(self.shapes)                             # self.shapes = bara synliga bitar (vald färg)
        print(f"\n[QUEUE] {len(self.queue)} {self.mode.colour} saved: " + ", ".join(s.shape for s in self.queue.items))
        self.refresh_list()

    def start(self):
        if self.queue.start():
            print(f"\n[QUEUE] Started, {len(self.queue)} to pick")
        self.update_buttons()

    def stop(self):
        if self.queue.running:
            print(f"\n[QUEUE] Paused, {len(self.queue)} left in the queue")
        self.queue.stop()
        self.update_buttons()

    def clear(self):
        left = self.queue.clear()
        if left:
            print(f"\n[QUEUE] Cleared, {left} not sent")
        self.refresh_list()

    def jump(self, piece: DetectedShape):
        """Flyttar biten först i kön och startar kön. Den skickas så fort pausen efter förra biten har gått."""
        self.queue.jump(piece)
        wait = self.queue.wait_left(time.time())
        print(f"\n[QUEUE] {piece.colour} {piece.shape} jumps the queue" + (f", sent in {wait:.0f} s" if wait else ", sent now"))
        self.refresh_list()

    def pick_now(self):
        sel = self.queue_list.curselection()
        if sel:
            self.jump(self.queue.items[sel[0]])

    # Plocka
    def send(self, piece: DetectedShape, source: str) -> bool:
        cmd = self.sender.send(piece, source)
        if cmd is None:
            self.last_command.config(text="NOT SENT: no ArUco marker found", fg="#dc2626")
            return False
        self.last_command.config(text="Sent: " + cmd.strip().replace(";", "; "), fg="#444")
        self.highlight = (piece, time.time() + 1.5)
        return True

    def on_click(self, event):
        x, y = event.x / self.scale, event.y / self.scale         # fönsterpixlar -> kamerapixlar
        for s in self.shapes:
            bx, by, bw, bh = s.bbox
            if bx <= x <= bx + bw and by <= y <= by + bh:
                self.mode.on_click(s, self)                         # läget bestämmer: skicka direkt eller köa
                return

    # En frame
    def tick(self):
        if self.stop_event.is_set():
            return
        frame = self.source.read()
        if frame is not None:
            self.show(frame)
        self.root.after(15, self.tick)

    def show(self, frame):
        # Var 200:e ms: skicka en bild till AI-threaden
        now = time.time()
        if now - self.last_snapshot > 0.2 and self.frames.empty():
            self.frames.put(frame.copy())
            self.last_snapshot = now

        all_shapes = self.latest.get()
        self.shapes = [s for s in all_shapes if self.mode.visible(s)]   # andra färger: ingen kontur, går inte att klicka

        # Kön: skicka nästa bit när pausen efter förra har gått
        piece = self.queue.due(now)
        if piece is not None:
            if not self.send(piece, f"QUEUE ({len(self.queue) - 1} left)"):
                self.stop()                         # ingen markör: pausa, biten ligger kvar först i kön
                return
            if self.queue.mark_sent(now):
                print("\n[QUEUE] Done")
            self.refresh_list()

        display = self.hud.draw(frame, self.shapes)
        self.hud.draw_queue(display, self.queue.items)
        if self.highlight and now < self.highlight[1]:
            self.hud.draw_sending(display, self.highlight[0])

        # Till Tkinter: krymp, BGR -> RGB, PhotoImage
        self.scale = min(DISPLAY_SIZE[0] / display.shape[1], DISPLAY_SIZE[1] / display.shape[0])
        small = cv2.resize(display, None, fx=self.scale, fy=self.scale, interpolation=cv2.INTER_AREA)
        self.photo = ImageTk.PhotoImage(Image.fromarray(cv2.cvtColor(small, cv2.COLOR_BGR2RGB)))
        self.video.config(image=self.photo)
        self.status.config(text=self.status_text(all_shapes, now))

    def status_text(self, all_shapes: List[DetectedShape], now: float) -> str:
        counts = ", ".join(f"{sum(s.colour == c for s in all_shapes)} {c}" for c in self.colour_buttons)
        if not self.queue.items:
            queue_text = "empty"
        elif self.queue.running:
            wait = self.queue.wait_left(now)
            queue_text = f"{len(self.queue)} left, next in {wait:.0f} s" if wait else f"{len(self.queue)} left, sending"
        else:
            queue_text = f"{len(self.queue)} saved, paused (Start)"
        return (f"Robot:   {'CONNECTED' if self.robot_link.connected else 'waiting...'}\n"
                f"Camera:  {'live' if self.source.live else 'test image'}\n"
                f"Marker:  {'found' if self.pipeline.calibration.found else 'NOT FOUND'}\n"
                f"In view: {counts}\n"
                f"Queue:   {queue_text}")

    def quit(self):
        self.stop_event.set()
        self.source.close()
        self.robot_link.close()
        self.root.destroy()
        print("Application shut down cleanly.")
