"""
capture_basler.py

Captures frames from a Basler camera (via pypylon) on a Raspberry Pi, for
building the shape-detection training dataset described earlier.

Two capture modes:
  --mode manual : live preview window, press SPACE to save a frame, 'q' to quit.
                  Best when you're moving shapes by hand and want full control
                  over exactly which frames get saved.
  --mode auto   : saves a frame automatically every --interval seconds, no
                  keypresses needed. Best paired with continuously shuffling
                  the shapes around while it runs unattended.

Optional: --require-aruco skips saving any frame where your calibration tag
isn't detected, so you never end up with images you can't compute robot-frame
coordinates for.

SETUP (Raspberry Pi)
---------------------
1. Install the Basler pylon SDK for ARM (Basler provides an ARM64/armhf
   build on their website — install that .deb/.tar first, matching your Pi's
   OS architecture).
2. pip install pypylon opencv-python --break-system-packages
3. Connect the camera (USB3 Basler cameras are the common case on a Pi).

USAGE
-----
# Manual capture, save into ./raw_images, press SPACE to grab a frame
python capture_basler.py --output-dir ./raw_images --mode manual

# Unattended capture, one frame every 1.5 seconds, only if the ArUco tag
# (5x5 dict, id 4) is visible
python capture_basler.py --output-dir ./raw_images --mode auto \
    --interval 1.5 --require-aruco --aruco-dict DICT_5X5_50 --aruco-id 4
"""

import argparse
import time
from datetime import datetime
from pathlib import Path

import cv2

try:
    from pypylon import pylon
except ImportError as e:
    raise SystemExit(
        "pypylon not found. Install the Basler pylon SDK (ARM build) first, "
        "then: pip install pypylon --break-system-packages"
    ) from e


def open_camera():
    tl_factory = pylon.TlFactory.GetInstance()
    devices = tl_factory.EnumerateDevices()
    if not devices:
        raise SystemExit("No Basler camera detected. Check the USB/GigE connection.")
    camera = pylon.InstantCamera(tl_factory.CreateFirstDevice())
    camera.Open()
    print(f"Connected to: {camera.GetDeviceInfo().GetModelName()}")
    return camera


def make_converter():
    converter = pylon.ImageFormatConverter()
    converter.OutputPixelFormat = pylon.PixelType_BGR8packed
    converter.OutputBitAlignment = pylon.OutputBitAlignment_MsbAligned
    return converter


def tag_visible(frame_bgr, aruco_dict_name, aruco_id):
    aruco_dict = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, aruco_dict_name))
    parameters = cv2.aruco.DetectorParameters()
    detector = cv2.aruco.ArucoDetector(aruco_dict, parameters)
    corners, ids, _ = detector.detectMarkers(frame_bgr)
    if ids is None:
        return False
    return aruco_id in ids.flatten()


def save_frame(frame_bgr, output_dir: Path, prefix="frame"):
    ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    path = output_dir / f"{prefix}_{ts}.jpg"
    cv2.imwrite(str(path), frame_bgr)
    return path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--mode", choices=["manual", "auto"], default="manual")
    ap.add_argument("--interval", type=float, default=1.0, help="seconds between auto-captures")
    ap.add_argument("--exposure-us", type=float, default=None, help="override exposure time (microseconds)")
    ap.add_argument("--preview", action="store_true", default=True)
    ap.add_argument("--no-preview", dest="preview", action="store_false", help="headless, no display window")
    ap.add_argument("--require-aruco", action="store_true", help="only save frames where the tag is visible")
    ap.add_argument("--aruco-dict", default="DICT_5X5_50")
    ap.add_argument("--aruco-id", type=int, default=0)
    args = ap.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    camera = open_camera()

    if args.exposure_us is not None:
        try:
            camera.ExposureTime.SetValue(args.exposure_us)
        except Exception as e:
            print(f"Warning: could not set exposure time ({e}); using camera default.")

    converter = make_converter()
    camera.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)

    saved_count = 0
    skipped_count = 0
    last_capture_time = 0.0

    print(f"Mode: {args.mode} | Saving to: {output_dir.resolve()}")
    if args.mode == "manual":
        print("Press SPACE to save a frame, 'q' to quit.")
    else:
        print(f"Auto-capturing every {args.interval}s. Ctrl+C to stop.")

    try:
        while camera.IsGrabbing():
            grab_result = camera.RetrieveResult(5000, pylon.TimeoutHandling_ThrowException)
            if not grab_result.GrabSucceeded():
                grab_result.Release()
                continue

            image = converter.Convert(grab_result)
            frame = image.GetArray()
            grab_result.Release()

            if args.preview:
                cv2.imshow("Basler capture (SPACE=save, q=quit)", frame)
                key = cv2.waitKey(1) & 0xFF
            else:
                key = -1

            should_save = False
            if args.mode == "manual":
                should_save = key == ord(" ")
                if key == ord("q"):
                    break
            else:  # auto mode
                now = time.time()
                if now - last_capture_time >= args.interval:
                    should_save = True
                    last_capture_time = now

            if should_save:
                if args.require_aruco and not tag_visible(frame, args.aruco_dict, args.aruco_id):
                    skipped_count += 1
                    print(f"Skipped (tag not visible). Saved so far: {saved_count}, skipped: {skipped_count}")
                else:
                    path = save_frame(frame, output_dir)
                    saved_count += 1
                    print(f"Saved {path.name} (total saved: {saved_count})")

    except KeyboardInterrupt:
        print("\nStopped by user.")
    finally:
        camera.StopGrabbing()
        camera.Close()
        if args.preview:
            cv2.destroyAllWindows()
        print(f"\nDone. {saved_count} frames saved, {skipped_count} skipped (no tag) to {output_dir.resolve()}")


if __name__ == "__main__":
    main()
