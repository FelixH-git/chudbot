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
import json
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

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


def preprocess_frame(
    frame_bgr,
    aruco_dict_name="DICT_5X5_100",
    aruco_id=0,
    marker_size_mm=50.0,
    min_object_area=150.0,
):
    """Detect coloured objects and convert their centres to workspace mm."""
    annotated = frame_bgr.copy()
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)

    aruco_dict = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, aruco_dict_name))
    parameters = cv2.aruco.DetectorParameters()
    if hasattr(cv2.aruco, "ArucoDetector"):
        detector = cv2.aruco.ArucoDetector(aruco_dict, parameters)
        marker_corners, marker_ids, _ = detector.detectMarkers(gray)
    else:
        marker_corners, marker_ids, _ = cv2.aruco.detectMarkers(
            gray, aruco_dict, parameters=parameters
        )

    homography = None
    if marker_ids is not None:
        matching = np.flatnonzero(marker_ids.flatten() == aruco_id)
        if len(matching):
            image_corners = marker_corners[matching[0]][0].astype(np.float32)
            world_corners = np.array(
                [
                    [0, 0],
                    [marker_size_mm, 0],
                    [marker_size_mm, marker_size_mm],
                    [0, marker_size_mm],
                ],
                dtype=np.float32,
            )
            homography, _ = cv2.findHomography(image_corners, world_corners)
            cv2.polylines(
                annotated,
                [image_corners.astype(np.int32)],
                True,
                (255, 0, 0),
                2,
            )

    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    saturation = hsv[:, :, 1]
    value = hsv[:, :, 2]
    object_mask = cv2.inRange(
        cv2.merge((saturation, value, value)),
        np.array([45, 35, 35], dtype=np.uint8),
        np.array([255, 255, 255], dtype=np.uint8),
    )
    object_mask = cv2.morphologyEx(
        object_mask,
        cv2.MORPH_OPEN,
        np.ones((5, 5), dtype=np.uint8),
    )

    contours, _ = cv2.findContours(
        object_mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )
    objects = []

    for contour in contours:
        area = cv2.contourArea(contour)
        if area < min_object_area:
            continue

        moments = cv2.moments(contour)
        if moments["m00"] == 0:
            continue

        center_px = np.array(
            [
                moments["m10"] / moments["m00"],
                moments["m01"] / moments["m00"],
            ],
            dtype=np.float32,
        )
        rectangle = cv2.minAreaRect(contour)
        angle_deg = float(rectangle[2])
        corners = len(cv2.approxPolyDP(contour, 0.04 * cv2.arcLength(contour, True), True))
        if corners == 3:
            shape_name = "triangle"
        elif corners == 4:
            shape_name = "square_or_rectangle"
        else:
            shape_name = "circle_or_other"

        center_mm = None
        if homography is not None:
            point = center_px.reshape(1, 1, 2)
            center_mm = cv2.perspectiveTransform(point, homography)[0, 0]
            center_mm = [float(center_mm[0]), float(center_mm[1])]

        object_data = {
            "shape": shape_name,
            "center_px": [float(center_px[0]), float(center_px[1])],
            "center_mm": center_mm,
            "rotation_deg": angle_deg,
            "area_px": float(area),
        }
        objects.append(object_data)

        center_for_text = tuple(np.round(center_px).astype(int))
        cv2.drawContours(annotated, [contour], -1, (0, 255, 0), 2)
        cv2.circle(annotated, center_for_text, 5, (255, 0, 0), -1)
        coordinate_text = (
            f"({center_mm[0]:.1f}, {center_mm[1]:.1f}) mm"
            if center_mm is not None
            else f"({center_px[0]:.1f}, {center_px[1]:.1f}) px"
        )
        cv2.putText(
            annotated,
            f"{shape_name}: {coordinate_text}",
            (center_for_text[0] + 8, center_for_text[1] - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            annotated,
            f"rot {angle_deg:.1f} deg",
            (center_for_text[0] + 8, center_for_text[1] + 12),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

    objects.sort(key=lambda item: item["center_px"][1])
    return annotated, objects


def save_preprocessed_data(annotated_frame, objects, raw_path: Path):
    processed_path = raw_path.with_name(f"{raw_path.stem}_processed.jpg")
    labels_path = raw_path.with_suffix(".json")
    cv2.imwrite(str(processed_path), annotated_frame)
    labels_path.write_text(
        json.dumps({"image": raw_path.name, "objects": objects}, indent=2),
        encoding="utf-8",
    )
    return processed_path, labels_path


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
    ap.add_argument("--aruco-dict", default="DICT_5X5_100")
    ap.add_argument("--aruco-id", type=int, default=0)
    ap.add_argument("--marker-size-mm", type=float, default=50.0)
    ap.add_argument("--min-object-area", type=float, default=150.0)
    ap.add_argument(
        "--preprocess-every",
        type=int,
        default=20,
        help="run preprocessing once every N camera frames (default: 20)",
    )
    args = ap.parse_args()
    if args.preprocess_every < 1:
        ap.error("--preprocess-every must be at least 1")

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
    frame_count = 0
    last_capture_time = 0.0
    processed_frame = None
    objects = []

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
            frame_count += 1

            processed_this_frame = frame_count % args.preprocess_every == 0
            if processed_this_frame:
                processed_frame, objects = preprocess_frame(
                    frame,
                    aruco_dict_name=args.aruco_dict,
                    aruco_id=args.aruco_id,
                    marker_size_mm=args.marker_size_mm,
                    min_object_area=args.min_object_area,
                )

            if args.preview:
                preview_frame = processed_frame if processed_frame is not None else frame
                cv2.imshow("Preprocessed shapes (SPACE=save, q=quit)", preview_frame)
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
                    if processed_this_frame:
                        processed_path, labels_path = save_preprocessed_data(
                            processed_frame,
                            objects,
                            path,
                        )
                        print(
                            f"Saved {path.name}, {processed_path.name}, "
                            f"{labels_path.name} (objects: {len(objects)}, "
                            f"frame: {frame_count}, total saved: {saved_count})"
                        )
                    else:
                        print(
                            f"Saved {path.name} (raw only; preprocessing runs "
                            f"every {args.preprocess_every} frames, frame: "
                            f"{frame_count}, total saved: {saved_count})"
                        )

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