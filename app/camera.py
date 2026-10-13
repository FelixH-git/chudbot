"""Basler camera wrapper (via pypylon).

Importing this module never fails when pypylon is missing; the error is only
raised when you actually try to open the camera. That keeps ``main.py`` able to
fall back to a static test image on machines without a Basler installed.

Run this file directly for a standalone camera test::

    python camera.py     # [q] to quit, [s] to save a snapshot
"""

from __future__ import annotations

from typing import Optional

import numpy as np

try:
    from pypylon import pylon

    PYPYLON_AVAILABLE = True
except ImportError:  # pypylon / the Basler pylon SDK is not installed
    PYPYLON_AVAILABLE = False


class BaslerCamera:
    """Connects to the first Basler camera found and grabs BGR frames."""

    def __init__(self, timeout_ms: int = 5000):
        if not PYPYLON_AVAILABLE:
            raise ImportError(
                "pypylon is not installed. Install the Basler pylon SDK, then "
                "`pip install pypylon`."
            )
        self.timeout_ms = timeout_ms
        self.camera = None
        self.converter = pylon.ImageFormatConverter()
        self.converter.OutputPixelFormat = pylon.PixelType_BGR8packed
        self.converter.OutputBitAlignment = pylon.OutputBitAlignment_MsbAligned

    # -- state -----------------------------------------------------------------
    @property
    def isOpen(self) -> bool:
        return self.camera is not None and self.camera.IsOpen()

    # -- lifecycle -------------------------------------------------------------
    def open(self) -> None:
        """Find and connect to the first Basler camera, then start grabbing."""
        if self.isOpen:
            return

        tl_factory = pylon.TlFactory.GetInstance()
        devices = tl_factory.EnumerateDevices()
        if not devices:
            raise RuntimeError("No Basler camera found.")

        self.camera = pylon.InstantCamera(tl_factory.CreateFirstDevice())
        self.camera.Open()
        self.camera.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)
        print(f"[CAMERA] Connected to {self.camera.GetDeviceInfo().GetModelName()}")

    def grabFrame(self) -> Optional[np.ndarray]:
        """Return the latest frame as a BGR numpy array, or None on timeout."""
        if not self.isOpen or not self.camera.IsGrabbing():
            return None
        try:
            result = self.camera.RetrieveResult(
                self.timeout_ms, pylon.TimeoutHandling_ThrowException
            )
            if result.GrabSucceeded():
                frame = self.converter.Convert(result).GetArray()
                result.Release()
                return frame
            result.Release()
        except Exception as err:  # pylon raises its own exception type
            print(f"[CAMERA] Error grabbing frame: {err}")
        return None

    def close(self) -> None:
        """Stop grabbing and release the camera safely."""
        if self.camera is not None:
            if self.camera.IsGrabbing():
                self.camera.StopGrabbing()
            if self.camera.IsOpen():
                self.camera.Close()
            self.camera = None
            print("[CAMERA] Disconnected.")

    # -- context manager -------------------------------------------------------
    def __enter__(self) -> "BaslerCamera":
        self.open()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()


if __name__ == "__main__":
    import time

    import cv2

    print("--- Basler camera test ---")
    print("Press 'q' to quit, 's' to save 'test_frame.png'.")
    try:
        with BaslerCamera() as cam:
            prev = time.time()
            fps = 0.0
            while True:
                frame = cam.grabFrame()
                if frame is None:
                    continue
                now = time.time()
                fps = 0.9 * fps + 0.1 * (1.0 / max(now - prev, 1e-6))
                prev = now
                h, w = frame.shape[:2]
                cv2.putText(
                    frame, f"Res: {w}x{h} | FPS: {fps:.1f}", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2,
                )
                cv2.imshow("Basler Live Camera Test", frame)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break
                if key == ord("s"):
                    cv2.imwrite("test_frame.png", frame)
                    print("[CAMERA] Saved test_frame.png")
        cv2.destroyAllWindows()
    except Exception as err:
        print(f"[CAMERA] Test failed: {err}")
