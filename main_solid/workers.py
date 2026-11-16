# Author Nils Wikström niwi0007
"""
De tre bakgrunds-threadsen. Allt de delar (köer, senaste bitarna, stopp-signalen) skickas in som parametrar.
  1. ai_worker:          bild -> bitar i pixlar   (VisionPipeline.detect)
  2. coordinate_worker:  pixlar -> mm och RZ       (VisionPipeline.locate)
  3. robot_worker:       kommandon -> roboten      (RobotLink)
"""
import queue
import threading
import time
from typing import List

from detection import DetectedShape
from pipeline import VisionPipeline
from robot_operations import RobotLink


class LatestShapes:
    """De senaste bitarna från thread 2, läses av GUI:t. Lock så att listan inte byts mitt i en läsning."""

    def __init__(self):
        self._lock = threading.Lock()
        self._shapes: List[DetectedShape] = []

    def set(self, shapes: List[DetectedShape]) -> None:
        with self._lock:
            self._shapes = shapes

    def get(self) -> List[DetectedShape]:
        with self._lock:
            return list(self._shapes)


def ai_worker(pipeline: VisionPipeline, frames: queue.Queue, detections: queue.Queue, stop: threading.Event):
    """Hämtar bilder från kön hela tiden och kör nätverket på dem."""
    print("[Thread 1: AI Inference] Started.")
    while not stop.is_set():
        try:
            frame = frames.get(timeout=0.2)
        except queue.Empty:
            continue
        t0 = time.perf_counter()
        shapes = pipeline.detect(frame)
        dt_ms = (time.perf_counter() - t0) * 1000.0
        print(f"[AI Inference] Time: {dt_ms:.1f} ms | Detected {len(shapes)} shapes")
        if not detections.full():
            detections.put((frame, shapes))
    print("[Thread 1: AI Inference] Stopped.")


def coordinate_worker(pipeline: VisionPipeline, detections: queue.Queue, latest: LatestShapes,
                      stop: threading.Event):
    """Räknar om bitarnas centre från pixlar till mm på bordet, och RZ."""
    print("[Thread 2: Coordinate Translation] Started.")
    while not stop.is_set():
        try:
            frame, shapes = detections.get(timeout=0.2)
        except queue.Empty:
            continue
        latest.set(pipeline.locate(frame, shapes))
    print("[Thread 2: Coordinate Translation] Stopped.")


def robot_worker(link: RobotLink, commands: queue.Queue, stop: threading.Event):
    """Sköter TCP-anslutningen och skickar pick-kommandon till ABB-roboten."""
    print("[Thread 3: Server Communication] Started.")
    try:
        link.start()                                # väntar på att roboten ansluter
    except Exception as err:
        print(f"[Thread 3: Server Communication] Socket startup notice: {err}")

    while not stop.is_set():
        try:
            command = commands.get(timeout=0.2)
        except queue.Empty:
            continue
        print(f"[Thread 3] Sending command to ABB Robot: {command.strip()}")
        link.send(command)

    link.close()
    print("[Thread 3: Server Communication] Stopped.")
