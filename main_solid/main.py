# Author Nils Wikström niwi0007
"""
main.py - vision och robotstyrning med 4 threads. Starta med:  python main.py
Här skapas alla delar och kopplas ihop; ingen egen logik. Vad varje fil gör:
  settings.py      alla inställningar (markör, tray, segmentering, filter)
  segmenter.py     hittar bitarna i bilden (HSV-mask) och klipper ut crops
  shape_model.py   CRPNet (ONNX): class, rotation, centre, colour för en crop
  calibration.py   ArUco-markören: pixlar <-> mm
  pipeline.py      kör delarna i rätt ordning (detect = thread 1, locate = thread 2)
  robot_operations.py  RZ för roboten, kommando-texten och TCP-servern
  pick_operations.py   Free pick / Colour pick, kön, och sändningen (aldrig utan markör)
  workers.py       threads 1-3
  hud.py           ritar på bilden
  frame_source.py  Basler-kameran eller test.png
  gui.py           Tkinter-fönstret (thread 4)
"""
import os
os.environ["GDK_BACKEND"] = "x11"
os.environ["QT_QPA_PLATFORM"] = "xcb"
import queue
import threading

import tkinter as tk

from calibration import MarkerCalibration
from frame_source import open_source
from gui import PickerApp
from hud import HudRenderer
from pick_operations import PickSender
from pipeline import CandidateLog, VisionPipeline
from robot_operations import RobotFrame, RobotLink
from segmenter import PieceSegmenter
from settings import MODEL_INFO_PATH, MODEL_PATH, ROBOT_HOST, ROBOT_PORT, TEST_IMAGE_PATH
from shape_model import ModelInfo, ShapeModel
from workers import LatestShapes, ai_worker, coordinate_worker, robot_worker


def build_pipeline(info: ModelInfo) -> VisionPipeline:
    return VisionPipeline(segmenter=PieceSegmenter(), model=ShapeModel(MODEL_PATH, info),
                          calibration=MarkerCalibration(), robot_frame=RobotFrame(info.symmetry_deg),
                          log=CandidateLog())


def main():
    # 1. Vision pipeline och robot-länken
    print("Initializing Vision Pipeline...")
    info = ModelInfo.load(MODEL_INFO_PATH)
    pipeline = build_pipeline(info)
    robot_link = RobotLink(ROBOT_HOST, ROBOT_PORT)

    # 2. Basler-kameran (annars test.png)
    source = open_source(TEST_IMAGE_PATH)
    if source is None:
        return

    # 3. Det som delas mellan threads
    frames = queue.Queue(maxsize=1)          # GUI -> thread 1
    detections = queue.Queue(maxsize=2)      # thread 1 -> thread 2
    commands = queue.Queue()                 # GUI -> thread 3
    latest = LatestShapes()                  # thread 2 -> GUI
    stop = threading.Event()

    # 4. Threads 1, 2 och 3 i bakgrunden
    workers = [threading.Thread(target=ai_worker, args=(pipeline, frames, detections, stop), daemon=True),
               threading.Thread(target=coordinate_worker, args=(pipeline, detections, latest, stop), daemon=True)]
    for t in workers:
        t.start()
    threading.Thread(target=robot_worker, args=(robot_link, commands, stop), daemon=True).start()

    # 5. Tkinter-fönstret (main thread)
    root = tk.Tk()
    hud = HudRenderer(pipeline.calibration, rotating_shapes=info.symmetry_deg)
    sender = PickSender(commands, pipeline.calibration, pipeline.log)
    PickerApp(root, source, pipeline, hud, sender, robot_link, frames, latest, stop, info.colours)
    print("\nSYSTEM RUNNING (4 threads). Free pick: click a piece. Colour pick: choose a colour, Queue all, Start; click a piece or Pick now to jump the queue.")
    root.mainloop()

    # Vänta tills AI- och koordinat-threaden har slutat, annars kan programmet krascha när det stängs
    # mitt i en körning av modellen. (Server-threaden kan vänta på roboten för evigt, den väntar vi inte på.)
    for t in workers:
        t.join(timeout=3)


if __name__ == "__main__":
    main()
