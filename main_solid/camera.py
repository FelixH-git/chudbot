"""Basler-kameran. (Ett eget kameratest finns i main/camera.py: python camera.py)"""
from typing import Optional

import numpy as np

try:
    from pypylon import pylon
    PYPYLON_AVAILABLE = True
except ImportError:
    PYPYLON_AVAILABLE = False


class BaslerCamera:
    """Ansluter till Basler-kameran, ställer in den och hämtar frames."""

    def __init__(self, timeoutMs: int = 5000):
        if not PYPYLON_AVAILABLE:
            raise ImportError("pypylon not installed")
        self.timeoutMs = timeoutMs
        self.camera: Optional[pylon.InstantCamera] = None
        self.converter = pylon.ImageFormatConverter()
        self.converter.OutputPixelFormat = pylon.PixelType_BGR8packed
        self.converter.OutputBitAlignment = pylon.OutputBitAlignment_MsbAligned

    @property
    def isOpen(self) -> bool:
        return self.camera is not None and self.camera.IsOpen()

    def open(self) -> None:
        """Hittar och ansluter till den första Basler-kameran."""
        if self.isOpen:
            return
        tl_factory = pylon.TlFactory.GetInstance()
        if not tl_factory.EnumerateDevices():
            raise RuntimeError("NO CAMERA FOUND!!!!!!!!!!!")
        self.camera = pylon.InstantCamera(tl_factory.CreateFirstDevice())
        self.camera.Open()
        self.camera.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)
        print(f"[CAMERA] connected to Basler Camera {self.camera.GetDeviceInfo().GetModelName()}")

    def grabFrame(self) -> Optional[np.ndarray]:
        """Den senaste framen som en BGR numpy-array (OpenCV-formatet). None om det tar för lång tid (timeout)."""
        if not self.isOpen or not self.camera.IsGrabbing():
            return None
        try:
            grabRes = self.camera.RetrieveResult(self.timeoutMs, pylon.TimeoutHandling_ThrowException)
            frame = self.converter.Convert(grabRes).GetArray() if grabRes.GrabSucceeded() else None
            grabRes.Release()
            return frame
        except Exception as e:
            print(f"[CAMERA] error grabbing frame {e}")
            return None

    def close(self) -> None:
        """Stoppar kameran på ett säkert sätt."""
        if self.camera is not None:
            if self.camera.IsGrabbing():
                self.camera.StopGrabbing()
            if self.camera.IsOpen():
                self.camera.Close()
            self.camera = None
            print("[Camera] Disconnected safely.")
