"""
capture_dataset.py
==================
Capture a labelled dataset for training an AI model on the sorted-shape task.

For every capture it saves:
  * the RAW camera frame            (dataset/raw/<ts>_full.png)
  * the FILTERED / MASKED frame     (dataset/masked/<ts>_full.png)
  * the binary silhouette mask      (dataset/maskbin/<ts>_full.png)
  * per-object crops                (dataset/objects/<shape>/..., dataset/objects_raw/<shape>/...,
                                     dataset/objects_mask/<shape>/...)
  * one row per object in labels.csv with the object's CLASSIFICATION (shape)
    and its ROTATION (measured in the ArUco-marker frame, folded through the
    shape's symmetry so it is a clean regression target in [0, period) deg).

The masking / shape / orientation code mirrors test_basler.py so the labels
match what the live pick-and-place script uses.

Keys:
  c / Space  : capture current frame now (all detected objects + frames)
  a          : toggle AUTO mode - captures a new sample automatically whenever
               the set of (shape, rotation, position) on the table changes
               (handy for sweeping an object through many rotations by hand)
  q          : quit

Run on the robot Pi:  python3 capture_dataset.py
"""

import csv
import os
import time

import cv2
import numpy as np
from pypylon import pylon

# ----------------------------------------------------------------------
# Configuration (adjust before running)
# ----------------------------------------------------------------------
DATASET_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dataset")
ARUCO_LENGTH_MM = 100.0     # physical ArUco side length (mm)

# HSV range that keeps only the coloured parts (same as test_basler.py)
HSV_LOWER = np.array([0, 50, 50])
HSV_UPPER = np.array([179, 255, 255])

MIN_OBJECT_AREA = 300       # ignore contours smaller than this (resized px)
PAD_FRAC = 0.25             # extra border around each crop (fraction of bbox max dim)
SAVE_FULL_SIZE_RAW = True   # also store the full-resolution raw frame (bigger files)
SAVE_FEED_RAW = True        # store the resized raw feed too (pairs 1:1 with masked)

AUTO_CHANGE_THRESHOLD_DEG = 5.0   # auto mode: min rotation change to re-capture
AUTO_DEBOUNCE_SEC = 0.5           # auto mode: min seconds between captures

RETRIEVE_TIMEOUT_MS = 5000
MAX_CONSECUTIVE_TIMEOUTS = 3
MAX_GRAB_RESTARTS = 3

# ----------------------------------------------------------------------
# Shape detection (same logic as test_basler.py)
# ----------------------------------------------------------------------
# Rotational symmetry of each shape (deg). The measured feature angle is
# folded into [0, period) so the label is a clean target for the AI model.
SHAPE_SYMMETRY_DEG = {
    "Square": 90.0,
    "Rectangle": 180.0,
    "Hexagon": 60.0,
    "Triangle": 120.0,  # assumes equilateral
    "Star": 72.0,       # 5-fold star
    "Circle": 360.0,
}

# Map detected shape names to robot keys (kept for reference / grouping)
def shape_key(shape_name):
    mapping = {
        "Circle": "pcircle",
        "Square": "psquare",
        "Rectangle": "prectangle",
        "Hexagon": "phexa",
        "Star": "pstar",
        "Triangle": "ptriangle",
    }
    return mapping.get(shape_name, shape_name.lower())


def detect_shape(cnt):
    """Classify a contour as square, rectangle, triangle, hexagon, star, circle, etc."""
    peri = cv2.arcLength(cnt, True)
    if peri == 0:
        return "Unknown"
    approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
    vertices = len(approx)
    area = cv2.contourArea(cnt)
    circularity = 4 * np.pi * area / (peri * peri)

    if vertices == 3:
        return "Triangle"
    elif vertices == 4:
        x, y, w, h = cv2.boundingRect(cnt)
        aspect = w / float(h) if h != 0 else 0
        if 0.9 <= aspect <= 1.1:
            return "Square"
        return "Rectangle"
    elif vertices == 6:
        return "Hexagon"
    elif 8 <= vertices <= 12 and circularity < 0.8:
        return "Star"
    elif circularity > 0.85:
        return "Circle"
    return f"Polygon({vertices})"


# ----------------------------------------------------------------------
# Orientation helpers (same as test_basler.py)
# ----------------------------------------------------------------------
def _poly_points(approx):
    return approx.reshape(-1, 2).astype(np.float64)


def longest_edge_vector(poly_pts):
    """Direction vector (image frame) of the polygon's longest edge."""
    n = len(poly_pts)
    best_len, best_vec = -1.0, None
    for i in range(n):
        d = poly_pts[(i + 1) % n] - poly_pts[i]
        ln = np.hypot(d[0], d[1])
        if ln > best_len:
            best_len, best_vec = ln, d
    return best_vec


def shape_feature_vector(cnt, shape, cx, cy):
    """2D vector (image frame) defining "0 deg" for this shape:
      - Square/Rectangle/Triangle/Hexagon: along its longest edge.
      - Star: from the centre toward the point between its two "feet".
      - Circle: no feature - any rotation fits the slot.
    """
    if shape == "Circle":
        return None
    peri = cv2.arcLength(cnt, True)
    if peri == 0:
        return None
    if shape == "Star":
        approx = cv2.approxPolyDP(cnt, 0.01 * peri, True)  # keeps all 10 corners
        pts = _poly_points(approx)
        if len(pts) < 3:
            return None
        order = np.argsort(pts[:, 1])[::-1]          # two lowest vertices = "feet"
        feet_mid = pts[order[:2]].mean(axis=0)
        up = np.array([cx, cy], dtype=np.float64) - feet_mid
        norm = np.hypot(up[0], up[1])
        return up / norm if norm > 1e-6 else np.array([0.0, -1.0])
    approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
    pts = _poly_points(approx)
    if len(pts) < 2:
        return None
    vec = longest_edge_vector(pts)
    norm = np.hypot(vec[0], vec[1])
    return vec / norm if norm > 1e-6 else np.array([1.0, 0.0])


def angle_in_image_frame(vec):
    """Degrees of a direction vector measured from the image X axis (right = 0)."""
    return np.degrees(np.arctan2(vec[1], vec[0]))


def angle_in_marker_frame(vec, x_unit, y_unit):
    """Degrees of a direction vector measured from the marker X axis, in the
    marker's own frame (marker X = 0 deg)."""
    return np.degrees(np.arctan2(np.dot(vec, y_unit), np.dot(vec, x_unit)))


def fold_angle(angle_deg, period_deg):
    """Fold an angle into [0, period) using the shape's rotational symmetry."""
    if period_deg and period_deg > 0:
        return angle_deg % period_deg
    return angle_deg


def wrap_angle(angle_deg):
    """Wrap into (-180, 180]."""
    return ((angle_deg + 180.0) % 360.0) - 180.0


# ----------------------------------------------------------------------
# ArUco / frame helpers
# ----------------------------------------------------------------------
aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_100)
aruco_params = cv2.aruco.DetectorParameters()
aruco_detector = cv2.aruco.ArucoDetector(aruco_dict, aruco_params)


def find_marker_frame(corners):
    """From detected marker corners (in resized image space) return the
    reference corner and the marker X/Y unit axes."""
    if corners is None or len(corners) == 0:
        return None
    marker_pts = corners[0][0]
    tl = marker_pts[0]
    tr = marker_pts[1]
    bl = marker_pts[3]
    x_axis = tr - tl
    y_axis = bl - tl
    x_len = np.linalg.norm(x_axis)
    y_len = np.linalg.norm(y_axis)
    x_unit = x_axis / x_len if x_len > 0 else np.array([1.0, 0.0])
    y_unit = y_axis / y_len if y_len > 0 else np.array([0.0, 1.0])
    return {"ref": tl, "x_unit": x_unit, "y_unit": y_unit}


def measure_object(cnt, shape, frame):
    """Return a dict describing one detected object, or None if degenerate."""
    M = cv2.moments(cnt)
    if M["m00"] == 0:
        return None
    cx = M["m10"] / M["m00"]
    cy = M["m01"] / M["m00"]
    period = SHAPE_SYMMETRY_DEG.get(shape, 360.0)

    feat = shape_feature_vector(cnt, shape, cx, cy)
    if feat is None:
        angle_image = 0.0
        angle_marker = None
    else:
        angle_image = wrap_angle(angle_in_image_frame(feat))
        angle_marker = None
        if frame is not None:
            angle_marker = wrap_angle(angle_in_marker_frame(feat, frame["x_unit"], frame["y_unit"]))

    # Primary label: rotation in the marker frame (falls back to image frame)
    base = angle_marker if angle_marker is not None else angle_image
    rotation_deg = fold_angle(base, period)

    # Millimetre position relative to the marker corner (marker frame)
    x_mm = y_mm = None
    if frame is not None:
        obj_vec = np.array([cx, cy]) - frame["ref"]
        x_mm = np.dot(obj_vec, frame["x_unit"]) / pixel_cm_ratio
        y_mm = np.dot(obj_vec, frame["y_unit"]) / pixel_cm_ratio

    bx, by, bw, bh = cv2.boundingRect(cnt)
    return {
        "cx": cx,
        "cy": cy,
        "bbox": (int(bx), int(by), int(bw), int(bh)),
        "shape": shape,
        "shape_key": shape_key(shape),
        "symmetry_deg": period,
        "rotation_deg": float(rotation_deg),      # canonical label (folded)
        "angle_image_deg": float(angle_image),    # image-frame angle
        "angle_marker_deg": (None if angle_marker is None else float(angle_marker)),
        "x_mm": x_mm,
        "y_mm": y_mm,
    }


def padded_crop(img, box, pad_frac=PAD_FRAC):
    """Crop img to box expanded by a fraction of the larger box side."""
    x, y, w, h = box
    pad = int(max(w, h) * pad_frac)
    H, W = img.shape[:2]
    x0, y0 = max(0, x - pad), max(0, y - pad)
    x1, y1 = min(W, x + w + pad), min(H, y + h + pad)
    return img[y0:y1, x0:x1]


# ----------------------------------------------------------------------
# Dataset writing
# ----------------------------------------------------------------------
_csv_path = None
_csv_file = None
_csv_writer = None
_saved_counts = {}        # shape -> number of object samples saved
_total_samples = 0

CSV_HEADER = [
    "timestamp", "mode", "marker", "object_index", "shape", "shape_key",
    "symmetry_deg", "rotation_deg", "angle_image_deg", "angle_marker_deg",
    "x_mm", "y_mm", "bbox_x", "bbox_y", "bbox_w", "bbox_h",
    "raw_full_path", "raw_feed_path", "masked_full_path", "maskbin_full_path",
    "object_masked_path", "object_raw_path", "object_mask_path",
]


def _init_dataset():
    global _csv_path, _csv_file, _csv_writer
    for sub in ("raw", "raw_feed", "masked", "maskbin",
                "objects", "objects_raw", "objects_mask"):
        os.makedirs(os.path.join(DATASET_DIR, sub), exist_ok=True)
    _csv_path = os.path.join(DATASET_DIR, "labels.csv")
    new_file = not os.path.exists(_csv_path)
    _csv_file = open(_csv_path, "a", newline="", encoding="utf-8")
    _csv_writer = csv.writer(_csv_file)
    if new_file:
        _csv_writer.writerow(CSV_HEADER)
    print(f"Dataset directory: {DATASET_DIR}")


def _rel_path(*parts):
    """Return a POSIX-style path relative to DATASET_DIR for the CSV."""
    return os.path.relpath(os.path.join(DATASET_DIR, *parts), DATASET_DIR).replace("\\", "/")


def _close_dataset():
    global _csv_file
    if _csv_file is not None:
        _csv_file.flush()
        _csv_file.close()
        _csv_file = None


def save_capture(frame_full, img_resized, filtered_img, mask_bin, objects, mode, marker):
    """Save raw + masked + binary frames and one crop-set + CSV row per object."""
    global _total_samples
    if _csv_writer is None:
        return

    ts = time.time()
    stamp = time.strftime("%Y%m%d_%H%M%S", time.localtime(ts)) + f"_{int(ts % 1 * 1000):03d}"

    # --- full-frame images -------------------------------------------
    if SAVE_FULL_SIZE_RAW:
        raw_full_name = _rel_path("raw", f"{stamp}_full.png")
        cv2.imwrite(os.path.join(DATASET_DIR, "raw", f"{stamp}_full.png"), frame_full)
    else:
        raw_full_name = ""
    if SAVE_FEED_RAW:
        raw_feed_name = _rel_path("raw_feed", f"{stamp}_full.png")
        cv2.imwrite(os.path.join(DATASET_DIR, "raw_feed", f"{stamp}_full.png"), img_resized)
    else:
        raw_feed_name = ""
    masked_full_name = _rel_path("masked", f"{stamp}_full.png")
    maskbin_full_name = _rel_path("maskbin", f"{stamp}_full.png")
    cv2.imwrite(os.path.join(DATASET_DIR, "masked", f"{stamp}_full.png"), filtered_img)
    cv2.imwrite(os.path.join(DATASET_DIR, "maskbin", f"{stamp}_full.png"), mask_bin)

    # --- per-object crops + CSV rows ---------------------------------
    for i, o in enumerate(objects):
        shape_dir = o["shape"].replace("/", "_").replace(" ", "_")
        # masked colour crop, raw crop and binary silhouette crop
        masked_crop = padded_crop(filtered_img, o["bbox"])
        raw_crop = padded_crop(img_resized, o["bbox"])
        mask_crop = padded_crop(mask_bin, o["bbox"])

        masked_path = _rel_path("objects", shape_dir, f"{stamp}_obj{i}.png")
        raw_path = _rel_path("objects_raw", shape_dir, f"{stamp}_obj{i}.png")
        mask_path = _rel_path("objects_mask", shape_dir, f"{stamp}_obj{i}.png")

        for sub, crop, name in (("objects", masked_crop, masked_path),
                                ("objects_raw", raw_crop, raw_path),
                                ("objects_mask", mask_crop, mask_path)):
            os.makedirs(os.path.join(DATASET_DIR, sub, shape_dir), exist_ok=True)
            cv2.imwrite(os.path.join(DATASET_DIR, name), crop)

        _csv_writer.writerow([
            stamp, mode, "yes" if marker else "no", i, o["shape"], o["shape_key"],
            f"{o['symmetry_deg']:.1f}", f"{o['rotation_deg']:.2f}",
            f"{o['angle_image_deg']:.2f}",
            "" if o["angle_marker_deg"] is None else f"{o['angle_marker_deg']:.2f}",
            "" if o["x_mm"] is None else f"{o['x_mm']:.2f}",
            "" if o["y_mm"] is None else f"{o['y_mm']:.2f}",
            o["bbox"][0], o["bbox"][1], o["bbox"][2], o["bbox"][3],
            raw_full_name, raw_feed_name, masked_full_name, maskbin_full_name,
            masked_path, raw_path, mask_path,
        ])
        _saved_counts[o["shape"]] = _saved_counts.get(o["shape"], 0) + 1
        _total_samples += 1

    _csv_file.flush()
    return stamp


def frame_signature(objects):
    """Hash of what is currently on the table - used by AUTO mode to detect
    when the arrangement (shape + rotation + rough position) has changed.
    Rotation is binned to AUTO_CHANGE_THRESHOLD_DEG so small jitter does
    not trigger repeated captures."""
    return tuple(sorted(
        (o["shape"],
         round(o["rotation_deg"] / AUTO_CHANGE_THRESHOLD_DEG),
         int(o["cx"] // 150),
         int(o["cy"] // 150))
        for o in objects
    ))


def draw_orientation(img, o, color=(0, 255, 255)):
    """Draw the measured 0-deg feature direction + text for sanity checking."""
    cx, cy = int(o["cx"]), int(o["cy"])
    cv2.circle(img, (cx, cy), 4, (0, 0, 255), -1)
    # Arrow along the TRUE image-space feature direction (what you see on screen)
    rad = np.radians(o["angle_image_deg"])
    r = max(o["bbox"][2], o["bbox"][3])
    x2 = int(cx + r * 0.6 * np.cos(rad))
    y2 = int(cy + r * 0.6 * np.sin(rad))
    cv2.arrowedLine(img, (cx, cy), (x2, y2), color, 2, tipLength=0.3)
    # Label shows the canonical marker-frame rotation (the value logged as rotation_deg)
    cv2.putText(img, f"{o['shape']} {o['rotation_deg']:.0f}d",
                (cx + 6, cy - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)


# ----------------------------------------------------------------------
# Main capture loop
# ----------------------------------------------------------------------
def main():
    global pixel_cm_ratio, _saved_counts, _total_samples

    pixel_cm_ratio = 0.0  # set every frame a marker is visible

    converter = pylon.ImageFormatConverter()
    converter.OutputPixelFormat = pylon.PixelType_BGR8packed
    converter.OutputBitAlignment = pylon.OutputBitAlignment_MsbAligned

    _init_dataset()

    auto_mode = False
    last_sig = None
    last_auto_capture = 0.0

    with pylon.InstantCamera(pylon.FirstFound) as camera:
        print("Using device:", camera.DeviceInfo.ModelName)
        camera.Width.TrySetToMaximum()
        cv2.namedWindow("Dataset Capture")
        camera.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)

        consecutive_timeouts = 0
        grab_restarts = 0

        while camera.IsGrabbing():
            try:
                with camera.RetrieveResult(
                    RETRIEVE_TIMEOUT_MS, pylon.TimeoutHandling_ThrowException
                ) as grab_result:
                    if not grab_result.GrabSucceeded():
                        continue
                    consecutive_timeouts = 0

                    image = converter.Convert(grab_result)
                    img = image.GetArray()            # full-resolution BGR
                    img_resized = cv2.resize(img, (0, 0), fx=0.5, fy=0.5)

                    # color mask (same pipeline as test_basler.py)
                    hsv = cv2.cvtColor(img_resized, cv2.COLOR_BGR2HSV)
                    mask = cv2.inRange(hsv, HSV_LOWER, HSV_UPPER)
                    kernel = np.ones((5, 5), np.uint8)
                    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
                    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
                    filtered_img = cv2.bitwise_and(img_resized, img_resized, mask=mask)

                    # overlay copy (draw on this, don't save it)
                    display = filtered_img.copy()

                    # ArUco marker -> reference frame for rotation + mm
                    corners, ids, _ = aruco_detector.detectMarkers(img)
                    frame = None
                    if corners is not None and len(corners) > 0:
                        scaled_corners = [c * 0.5 for c in corners]
                        cv2.aruco.drawDetectedMarkers(display, scaled_corners, ids)
                        perimeter = cv2.arcLength(scaled_corners[0], True)
                        pixel_cm_ratio = perimeter / (ARUCO_LENGTH_MM * 4)
                        frame = find_marker_frame(scaled_corners)
                        tl = frame["ref"].astype(int)
                        cv2.circle(display, tuple(tl), 5, (255, 0, 0), -1)
                        cv2.putText(display, f"ratio {pixel_cm_ratio:.3f} px/mm",
                                    (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)

                    # detect objects
                    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    objects = []
                    for cnt in contours:
                        if cv2.contourArea(cnt) <= MIN_OBJECT_AREA:
                            continue
                        shape = detect_shape(cnt)
                        o = measure_object(cnt, shape, frame)
                        if o is None:
                            continue
                        objects.append(o)
                        draw_orientation(display, o)

                    # ----- capture triggers ---------------------------------
                    now = time.time()
                    do_capture = False
                    mode = "auto" if auto_mode else "manual"

                    if auto_mode:
                        sig = frame_signature(objects)
                        changed = sig != last_sig and (now - last_auto_capture) >= AUTO_DEBOUNCE_SEC
                        last_sig = sig
                        if changed:
                            do_capture = True
                            last_auto_capture = now
                    # (manual capture handled via key below)

                    # status / HUD
                    total = sum(_saved_counts.values())
                    cv2.putText(display, f"Saved samples: {total} | shapes: {dict(_saved_counts)}",
                                (30, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
                    cv2.putText(display,
                                f"AUTO: {'ON ' if auto_mode else 'OFF'} | 'c' capture | 'a' auto | 'q' quit",
                                (30, display.shape[0] - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                                (0, 255, 0) if auto_mode else (0, 165, 255), 1, cv2.LINE_AA)
                    cv2.imshow("Dataset Capture", display)

                    key = cv2.waitKey(1) & 0xFF
                    if key == ord("q"):
                        break
                    elif key in (ord("c"), ord(" ")):
                        stamp = save_capture(img, img_resized, filtered_img, mask, objects, "manual", frame is not None)
                        if stamp:
                            print(f"manual capture {stamp}: {len(objects)} object(s) saved "
                                  f"(total {sum(_saved_counts.values())})")
                    elif key == ord("a"):
                        auto_mode = not auto_mode
                        last_sig = frame_signature(objects)
                        print(f"Auto capture: {'ON' if auto_mode else 'OFF'}")
                    elif do_capture and objects:
                        save_capture(img, img_resized, filtered_img, mask, objects, "auto", frame is not None)

            except Exception as e:
                is_timeout = type(e).__name__ == "TimeoutException" or "grab timed out" in str(e).lower()
                if not is_timeout:
                    print(f"Error caught: {e}")
                    continue
                consecutive_timeouts += 1
                print(f"[Timeout {consecutive_timeouts}/{MAX_CONSECUTIVE_TIMEOUTS}] No frame: {e}")
                if consecutive_timeouts >= MAX_CONSECUTIVE_TIMEOUTS:
                    grab_restarts += 1
                    if grab_restarts > MAX_GRAB_RESTARTS:
                        print("Giving up - camera not streaming.")
                        break
                    try:
                        camera.StopGrabbing()
                    except Exception as stop_err:
                        print(f"  StopGrabbing failed: {stop_err}")
                    try:
                        camera.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)
                        consecutive_timeouts = 0
                        print(f"  Grab engine restarted ({grab_restarts}/{MAX_GRAB_RESTARTS})")
                    except Exception as start_err:
                        print(f"  StartGrabbing failed: {start_err}")

        camera.StopGrabbing()
        cv2.destroyAllWindows()

    _close_dataset()
    print("\nDone. Dataset summary per shape:")
    for shape, n in sorted(_saved_counts.items()):
        print(f"  {shape:>12s}: {n}")
    print(f"TOTAL object samples: {_total_samples}")
    print(f"Labels written to: {_csv_path}")


if __name__ == "__main__":
    main()
