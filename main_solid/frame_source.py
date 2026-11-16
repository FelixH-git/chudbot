# Author Nils Wikström niwi0007
"""Var bilderna kommer ifrån: Basler-kameran eller en sparad bild. GUI:t vet inte vilken det är."""
from abc import ABC, abstractmethod
from typing import Optional

import cv2
import numpy as np

from camera import BaslerCamera


class FrameSource(ABC):
    live = False                     # True = riktig kamera (visas i GUI:ts status)

    @abstractmethod
    def read(self) -> Optional[np.ndarray]:
        """Nästa bild (BGR), eller None om ingen bild fanns just nu."""

    def close(self) -> None:
        pass


class CameraSource(FrameSource):
    live = True

    def __init__(self, camera: BaslerCamera):
        self.camera = camera

    def read(self) -> Optional[np.ndarray]:
        return self.camera.grabFrame()               # None om kameran inte är öppen

    def close(self) -> None:
        self.camera.close()


class ImageSource(FrameSource):
    """Samma bild hela tiden (simulerad kamera när ingen Basler är inkopplad)."""

    def __init__(self, frame: np.ndarray):
        self.frame = frame

    def read(self) -> Optional[np.ndarray]:
        return self.frame.copy()


def open_source(image_path: str) -> Optional[FrameSource]:
    """Basler-kameran om den finns, annars bilden image_path. None om ingen av dem går att öppna."""
    try:
        camera = BaslerCamera()
        camera.open()
        print("[CAMERA] Live Basler Camera connected successfully!")
        return CameraSource(camera)
    except Exception as e:
        print(f"[CAMERA] Basler camera not detected ({e}). Using '{image_path}' as simulated feed.")

    frame = cv2.imread(image_path)
    if frame is None:
        print(f"Error: Neither Basler camera nor '{image_path}' could be opened!")
        return None
    return ImageSource(frame)
