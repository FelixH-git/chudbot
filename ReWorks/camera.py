"""
This file can be run on its own "python camera.py"
"""

from typing import Optional
import numpy as np 

try:
    from pypylon import pylon
    PYPYLON_AVAILABLE = True
except ImportError:
    PYPYLON_AVAILABLE = False

class BaslerCamera:
    """Manages connection and configuareion and frame grabbing from the basler camera"""
    def __init__(self,timeoutMs:int =5000):
        if not PYPYLON_AVAILABLE:
            raise ImportError(
                "pypylon not wokring"
            )

        self.timeoutMs = timeoutMs
        self.camera: Optional[pylon.InstantCamera] = None
        self.converter = pylon.ImageFormatConverter()
        self.converter.OutputPixelFormat = pylon.PixelType_BGR8packed
        self.converter.OutputBitAlignment = pylon.OutputBitAlignment_MsbAligned


    @property
    def isOpen(self) -> bool:
        return self.camera is not None and self.camera.IsOpen()
    
    def open(self) -> None:
        """finds and connects to the first available basler camera"""
        if self.isOpen:
            return

        tl_factory = pylon.TlFactory.GetInstance()
        devices = tl_factory.EnumerateDevices()
        if not devices: 
            raise RuntimeError("NO CAMERA FOUND!!!!!!!!!!!")

        # connect to first detected camera
        self.camera = pylon.InstantCamera(tl_factory.CreateFirstDevice())
        self.camera.Open()

        self.camera.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)
        modelName = self.camera.GetDeviceInfo().GetModelName()
        print(F"[CAMERA] connected to Basler Camera {modelName}")


    def grabFrame(self) -> Optional[np.ndarray]:
        """
        Grabs the latest frame as an RGB numpy array (openCV format)
        return : None if grabbing is times out
        """

        if not self.isOpen or not self.camera.IsGrabbing():
            return None

        try:
            grabRes = self.camera.RetrieveResult(
                self.timeoutMs,
                pylon.TimeoutHandling_ThrowException
            )
            if grabRes.GrabSucceeded():
                image = self.converter.Convert(grabRes)
                frame = image.GetArray()
                grabRes.Release()
                return frame
            grabRes.Release()

        except Exception as e:
            print(f"[CAMERA] errror grabbing frame {e}")

    def close(self) -> None:
        """Safely stops grabbing and releases the camera hardware."""
        if self.camera is not None:
            if self.camera.IsGrabbing():
                self.camera.StopGrabbing()
            if self.camera.IsOpen():
                self.camera.Close()
            self.camera = None
            print("[Camera] Disconnected safely.")

    def __enter__(self):
        self.open()
        return self
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()




if __name__ == "__main__":
    import cv2
    import time
    print("--- Running Standalone Camera Test ---")
    print("Press 'q' in the window to quit, or 's' to save a test snapshot.")
    try:
        with BaslerCamera() as cam:
            prev_time = time.time()
            fps = 0.0
            while True:
                frame = cam.grabFrame()
                if frame is None:
                    continue
                # Calculate real-time FPS
                current_time = time.time()
                fps = 0.9 * fps + 0.1 * (1.0 / (current_time - prev_time))
                prev_time = current_time
                # Display resolution and FPS on the image
                h, w = frame.shape[:2]
                cv2.putText(
                    frame,
                    f"Res: {w}x{h} | FPS: {fps:.1f}",
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.0,
                    (0, 255, 0),
                    2,
                )
                cv2.imshow("Basler Live Camera Test", frame)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break
                elif key == ord("s"):
                    cv2.imwrite("test_frame.png", frame)
                    print("[Camera] Saved test snapshot as 'test_frame.png'")
            cv2.destroyAllWindows()
    except Exception as err:
        print(f"[Camera Test Failed]: {err}")