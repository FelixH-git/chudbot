import threading

import cv2
import numpy as np
from pypylon import pylon

from Socka import RobotLink

aruco_length = 100  # mm

# Initialize ArUco detector
aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_100)
parameters = cv2.aruco.DetectorParameters()
aruco_detector = cv2.aruco.ArucoDetector(aruco_dict, parameters)


def detect_shape(cnt):
    """Classify a contour as square, rectangle, triangle, hexagon, star, circle, etc."""
    peri = cv2.arcLength(cnt, True)
    if peri == 0:
        return "Unknown"

    # Approximate the contour to get its main vertices
    approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
    vertices = len(approx)

    # Circularity: 1.0 = perfect circle
    area = cv2.contourArea(cnt)
    circularity = 4 * np.pi * area / (peri * peri)

    if vertices == 3:
        return "Triangle"
    elif vertices == 4:
        # A square has roughly equal width and height
        x, y, w, h = cv2.boundingRect(cnt)
        aspect = w / float(h) if h != 0 else 0
        if 0.9 <= aspect <= 1.1:
            return "Square"
        return "Rectangle"
    elif vertices == 6:
        return "Hexagon"
    elif 8 <= vertices <= 12 and circularity < 0.8:
        # A 5-pointed star approximates to ~10 sharp corners
        return "Star"
    elif circularity > 0.85:
        return "Circle"
    return f"Polygon({vertices})"


# ---- Map detected shape names to the robot's known target keys ----
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


# =====================================================================
# ---- Placement-orientation control (ADJUST THESE TO CALIBRATE) ----
# =====================================================================
# Why this exists: the robot picks every part with its wrist rotated so the
# part's "alignment feature" is gripped the same way each time, then drops it
# at a FIXED ai* target in wobjAiPlaceTray. Because the wrist rotation sent to
# the robot equals the object's measured rotation, every part is seated at the
# SAME orientation and fits its placement point (the place WObj) every time.
#
# Angles are measured in the ArUco-marker frame (marker X axis = "0 deg") -
# the same frame the X/Y mm coordinates are computed in, which maps 1:1 to
# wobjAruco in the robot.

# Rotational symmetry of each shape (deg). The measured feature angle is
# folded into [0, period) so the robot only needs a small wrist rotation
# to grip the part "straight" (symmetric parts fit the same either way).
SHAPE_SYMMETRY_DEG = {
    "Square": 90.0,
    "Rectangle": 180.0,
    "Hexagon": 60.0,
    "Triangle": 120.0,  # assumes equilateral
    "Star": 72.0,       # 5-fold star
    "Circle": 360.0,
}

# Per-shape fixed offset (deg) added before sending. Leave at 0 unless one
# shape needs its own correction (e.g. star reference choice).
ANGLE_BIAS_DEG = {
    "Square": 0.0,
    "Rectangle": 0.0,
    "Hexagon": 0.0,
    "Triangle": 0.0,
    "Star": 0.0,
}

# MASTER calibration knob (deg). Absorbs the offset between the marker frame
# and the robot's wrist-zero / place frame. Adjust it on the bench - or live
# while running with the '+'/'-' keys (see on-screen HUD) - until the parts
# seat cleanly in the place WObj, then write the final number back here.
ANGLE_MASTER_OFFSET_DEG = 0.0

# If the robot turns the wrong way (camera/marker frame is mirrored relative
# to the robot frame), enable this or press 'f' while the program runs.
ANGLE_FLIP_SIGN = False

# Live-tunable state (changed with the keyboard - see HUD / keys below)
_angle_offset = ANGLE_MASTER_OFFSET_DEG  # degrees added to every angle
_angle_flip = ANGLE_FLIP_SIGN            # mirror the angle sign
_angle_step = 1.0                        # degrees per '+'/'-' press


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
    """2D vector (image frame) defining "0 deg" for this shape, so the robot
    can grip it in a canonical orientation:
      - Square/Rectangle/Triangle/Hexagon: along its longest edge.
      - Star: from the centre toward the point between its two "feet"
        (the two lowest vertices) - i.e. the star standing upright.
      - Circle: no feature - any rotation fits the slot.
    """
    if shape == "Circle":
        return None
    peri = cv2.arcLength(cnt, True)
    if peri == 0:
        return None
    if shape == "Star":
        approx = cv2.approxPolyDP(cnt, 0.01 * peri, True)  # fine: keeps all 10 corners
        pts = _poly_points(approx)
        if len(pts) < 3:
            return None
        # Two vertices with the largest image-y are the "feet" the star rests on
        order = np.argsort(pts[:, 1])[::-1]
        feet_mid = pts[order[:2]].mean(axis=0)
        up = np.array([cx, cy], dtype=np.float64) - feet_mid
        return up if np.hypot(up[0], up[1]) > 1e-6 else np.array([0.0, -1.0])
    approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
    pts = _poly_points(approx)
    return longest_edge_vector(pts) if len(pts) >= 2 else None


def angle_in_marker_frame(vec, x_unit, y_unit):
    """Degrees of a direction vector measured from the marker X axis, in the
    marker's own frame (marker X = 0 deg). Mirrors how x_mm/y_mm are built."""
    return np.degrees(np.arctan2(np.dot(vec, y_unit), np.dot(vec, x_unit)))


def wrap_angle(angle_deg):
    """Wrap into (-180, 180] for a tidy number to send to the robot."""
    return ((angle_deg + 180.0) % 360.0) - 180.0


# ---- Socket link to the ABB robot (PC = server, robot connects) ----
robot_link = RobotLink()
threading.Thread(target=robot_link.start, daemon=True).start()

# ---- Click-to-select state (point clicked in the resized frame) ----
_click_lock = threading.Lock()
_click_pos = None          # (x, y) just clicked, waiting to be processed
_pending = None            # chosen object {'payload', 'bbox'}, waiting to be sent
_last_sent = None          # last payload successfully sent to the robot


def on_mouse(event, x, y, flags, param):
    global _click_pos
    if event == cv2.EVENT_LBUTTONDOWN:
        with _click_lock:
            _click_pos = (x, y)


def draw_axis_arrow(img, p_start, p_end, color, label, center=None):
    """Draw an arrowed line and place its label off the line, away from `center`."""
    start = (int(p_start[0]), int(p_start[1]))
    end = (int(p_end[0]), int(p_end[1]))
    cv2.arrowedLine(img, start, end, color, 3, tipLength=0.15)

    # Place the label slightly beside the line, on the side facing away from the marker
    mid = (p_start + p_end) / 2.0
    vec = p_end - p_start
    norm = np.linalg.norm(vec)
    if norm > 0:
        perp = np.array([-vec[1], vec[0]]) / norm
        if center is not None and np.dot(perp, mid - center) < 0:
            perp = -perp
    else:
        perp = np.array([0.0, -1.0])

    label_pos = mid + perp * 22
    cv2.putText(
        img,
        label,
        (int(label_pos[0]), int(label_pos[1])),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        color,
        2,
        cv2.LINE_AA,
    )


# Converter to turn raw Bayer sensor data into a color BGR array
converter = pylon.ImageFormatConverter()
converter.OutputPixelFormat = pylon.PixelType_BGR8packed
converter.OutputBitAlignment = pylon.OutputBitAlignment_MsbAligned

with pylon.InstantCamera(pylon.FirstFound) as camera:
    print("Using device:", camera.DeviceInfo.ModelName)

    camera.Width.TrySetToMaximum()

    cv2.namedWindow("Basler Camera Feed")
    cv2.setMouseCallback("Basler Camera Feed", on_mouse)

    camera.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)

    # ---- Timeout / error handling state ----
    RETRIEVE_TIMEOUT_MS = 5000
    MAX_CONSECUTIVE_TIMEOUTS = 3   # restart grabbing after this many timeouts in a row
    MAX_GRAB_RESTARTS = 3          # give up entirely after this many failed restarts
    consecutive_timeouts = 0       # number of retrieve timeouts in a row
    grab_restarts = 0              # how many times the grab engine was restarted

    while camera.IsGrabbing():
        try:
            with camera.RetrieveResult(
                RETRIEVE_TIMEOUT_MS, pylon.TimeoutHandling_ThrowException
            ) as grab_result:
                if grab_result.GrabSucceeded():
                    consecutive_timeouts = 0  # a frame arrived - reset the timeout streak
                    # Convert raw frame to color BGR numpy array
                    image = converter.Convert(grab_result)
                    img = image.GetArray()

                    # 1. Detect ArUco markers on the original full-size image
                    corners, ids, _ = aruco_detector.detectMarkers(img)

                    # 2. Resize the image down
                    img_resized = cv2.resize(img, (0, 0), fx=0.5, fy=0.5)

                    # 3. Apply the color mask
                    hsv = cv2.cvtColor(img_resized, cv2.COLOR_BGR2HSV)
                    lower_color = np.array([0, 50, 50])
                    upper_color = np.array([179, 255, 255])
                    mask = cv2.inRange(hsv, lower_color, upper_color)

                    kernel = np.ones((5, 5), np.uint8)
                    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
                    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

                    filtered_img = cv2.bitwise_and(img_resized, img_resized, mask=mask)

                    # 4. Draw markers, text, and object distances after the mask is applied
                    if len(corners) > 0:
                        # Scale full-size corners down by 0.5 to match the resized image
                        scaled_corners = [c * 0.5 for c in corners]
                        cv2.aruco.drawDetectedMarkers(filtered_img, scaled_corners, ids)
                        
                        aruco_perimeter = cv2.arcLength(scaled_corners[0], True)
                        pixel_cm_ratio = aruco_perimeter / (aruco_length * 4)

                        # Marker corners: [top-left, top-right, bottom-right, bottom-left]
                        marker_pts = scaled_corners[0][0]
                        tl = marker_pts[0]
                        tr = marker_pts[1]
                        bl = marker_pts[3]
                        marker_center = np.mean(marker_pts, axis=0)

                        # Marker-local X/Y axes along its top (tl->tr) and left (tl->bl) edges
                        x_axis = tr - tl
                        y_axis = bl - tl
                        x_len = np.linalg.norm(x_axis)
                        y_len = np.linalg.norm(y_axis)
                        x_unit = x_axis / x_len if x_len > 0 else np.array([1.0, 0.0])
                        y_unit = y_axis / y_len if y_len > 0 else np.array([0.0, 1.0])

                        # Use the top-left marker corner as the reference origin
                        aruco_ref = tl
                        aruco_ref_int = (int(aruco_ref[0]), int(aruco_ref[1]))
                        
                        # Draw the marker reference (corner) point
                        cv2.circle(filtered_img, aruco_ref_int, 5, (255, 0, 0), -1)

                        # X/Y direction arrows along the marker sides (X = red, Y = green)
                        draw_axis_arrow(filtered_img, tl, tr, (0, 0, 255), "X", marker_center)
                        draw_axis_arrow(filtered_img, tl, bl, (0, 255, 0), "Y", marker_center)

                        # Find contours of colored objects separated by black background
                        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                        object_id = 0  # Running ID for each detected color region
                        detected_objects = []
                        for cnt in contours:
                            # Filter out small noise areas
                            if cv2.contourArea(cnt) > 300:
                                object_id += 1
                                # Find the center (centroid) of the object using image moments
                                M = cv2.moments(cnt)
                                if M["m00"] != 0:
                                    obj_cx = int(M["m10"] / M["m00"])
                                    obj_cy = int(M["m01"] / M["m00"])

                                    # Classify the shape of the colored object
                                    shape = detect_shape(cnt)

                                    # ---- Object rotation, measured in the marker frame ----
                                    feat_vec = shape_feature_vector(cnt, shape, obj_cx, obj_cy)
                                    if feat_vec is None:
                                        angle_deg = 0.0  # circle / unknown - fits anyway
                                    else:
                                        raw = angle_in_marker_frame(feat_vec, x_unit, y_unit)
                                        period = SHAPE_SYMMETRY_DEG.get(shape, 360.0)
                                        # Fold through the shape's symmetry so the wrist only
                                        # needs a small rotation to grip the part "straight"
                                        angle_deg = raw % period if period > 0 else raw
                                        angle_deg += ANGLE_BIAS_DEG.get(shape, 0.0)
                                        angle_deg += _angle_offset
                                        if _angle_flip:
                                            angle_deg = -angle_deg
                                        angle_deg = wrap_angle(angle_deg)
                                    
                                    # Offset of the object center from the ArUco reference corner
                                    obj_vec = np.array([obj_cx, obj_cy]) - aruco_ref

                                    # Project onto the marker's X (top edge) and Y (left edge)
                                    # axes, then convert from pixels to millimeters so the
                                    # coordinates are truly relative to the ArUco marker
                                    x_mm = np.dot(obj_vec, x_unit) / pixel_cm_ratio
                                    y_mm = np.dot(obj_vec, y_unit) / pixel_cm_ratio
                                    total_dist_mm = np.hypot(x_mm, y_mm)

                                    # Record this object so a click can select & send it
                                    bx, by, bw, bh = cv2.boundingRect(cnt)
                                    detected_objects.append(
                                        {
                                            "id": object_id,
                                            "cx": obj_cx,
                                            "cy": obj_cy,
                                            "bbox": (bx, by, bw, bh),
                                            "x_mm": x_mm,
                                            "y_mm": y_mm,
                                            "shape": shape,
                                            "angle_deg": angle_deg,
                                        }
                                    )

                                    # Draw line from the ArUco corner to the object center
                                    cv2.line(filtered_img, aruco_ref_int, (obj_cx, obj_cy), (255, 255, 0), 2)
                                    # Draw object center point
                                    cv2.circle(filtered_img, (obj_cx, obj_cy), 5, (0, 0, 255), -1)
                                    # Draw the contour outline
                                    cv2.drawContours(filtered_img, [cnt], -1, (0, 255, 0), 2)

                                    # Display the object ID above the center point
                                    cv2.putText(
                                        filtered_img,
                                        f"ID: {object_id}",
                                        (obj_cx - 10, obj_cy - 55),
                                        cv2.FONT_HERSHEY_SIMPLEX,
                                        0.7,
                                        (0, 255, 255),
                                        2,
                                        cv2.LINE_AA,
                                    )

                                    # Display the detected shape below the center point
                                    cv2.putText(
                                        filtered_img,
                                        shape,
                                        (obj_cx - 10, obj_cy + 25),
                                        cv2.FONT_HERSHEY_SIMPLEX,
                                        0.6,
                                        (0, 255, 255),
                                        2,
                                        cv2.LINE_AA,
                                    )

                                    # Display X/Y position relative to the ArUco corner (mm)
                                    cv2.putText(
                                        filtered_img,
                                        f"X:{x_mm:.1f} Y:{y_mm:.1f} mm",
                                        (obj_cx - 10, obj_cy + 45),
                                        cv2.FONT_HERSHEY_SIMPLEX,
                                        0.5,
                                        (255, 255, 255),
                                        1,
                                        cv2.LINE_AA,
                                    )

                                    # Display the rotation angle that will be sent to the robot
                                    cv2.putText(
                                        filtered_img,
                                        f"Rot: {angle_deg:.1f} deg",
                                        (obj_cx - 10, obj_cy + 65),
                                        cv2.FONT_HERSHEY_SIMPLEX,
                                        0.5,
                                        (255, 255, 0),
                                        1,
                                        cv2.LINE_AA,
                                    )

                                    # Display distance text near the object
                                    dist_text = f"{total_dist_mm:.1f} mm"
                                    cv2.putText(
                                        filtered_img,
                                        dist_text,
                                        (obj_cx + 10, obj_cy - 25),
                                        cv2.FONT_HERSHEY_SIMPLEX,
                                        0.6,
                                        (255, 255, 255),
                                        2,
                                        cv2.LINE_AA,
                                    )

                        # ---- Click-to-select: choose an object and send its X/Y to the robot ----
                        with _click_lock:
                            click = _click_pos
                            _click_pos = None

                        if click is not None and detected_objects:
                            # Prefer the object whose bounding box contains the click
                            chosen = None
                            for obj in detected_objects:
                                bx, by, bw, bh = obj["bbox"]
                                if bx <= click[0] <= bx + bw and by <= click[1] <= by + bh:
                                    chosen = obj
                                    break
                            # Otherwise pick the object with the nearest center (within 50 px)
                            if chosen is None:
                                chosen = min(
                                    detected_objects,
                                    key=lambda o: (o["cx"] - click[0]) ** 2
                                    + (o["cy"] - click[1]) ** 2,
                                )
                                if np.hypot(chosen["cx"] - click[0], chosen["cy"] - click[1]) > 50:
                                    chosen = None

                            if chosen is not None:
                                _pending = {
                                    "payload": (
                                        f"{shape_key(chosen['shape'])}"
                                        f",{chosen['y_mm']:.2f},{chosen['x_mm']:.2f}"
                                        f",{chosen['angle_deg']:.2f}"
                                    ),
                                    "bbox": chosen["bbox"],
                                }

                        # Show the pending selection and send it once the robot is connected
                        if _pending is not None:
                            px, py, pw, ph = _pending["bbox"]
                            cv2.rectangle(filtered_img, (px, py), (px + pw, py + ph), (255, 0, 255), 2)
                            cv2.putText(
                                filtered_img,
                                "Selected",
                                (px, py - 8),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.6,
                                (255, 0, 255),
                                2,
                                cv2.LINE_AA,
                            )
                            if robot_link.connected:
                                robot_link.send(_pending["payload"])
                                _last_sent = _pending["payload"]
                                _pending = None
                                cv2.putText(
                                    filtered_img,
                                    f"Sent: {_last_sent}",
                                    (px, py - 30),
                                    cv2.FONT_HERSHEY_SIMPLEX,
                                    0.6,
                                    (0, 255, 0),
                                    2,
                                    cv2.LINE_AA,
                                )

                        text = f"Ratio: {pixel_cm_ratio:.4f} px/mm"
                        cv2.putText(
                            filtered_img,
                            text,
                            (30, 50),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            1,
                            (0, 255, 0),
                            2,
                            cv2.LINE_AA,
                        )
                    else:
                        cv2.putText(
                            filtered_img,
                            "No ArUco Marker Detected",
                            (30, 50),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            1,
                            (0, 0, 255),
                            2,
                            cv2.LINE_AA,
                        )

                    # Bottom-left status bar
                    status = "Robot: connected" if robot_link.connected else "Robot: waiting..."
                    if _last_sent is not None:
                        status += f" | Last sent: {_last_sent}"
                    cv2.putText(
                        filtered_img,
                        status,
                        (30, filtered_img.shape[0] - 12),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (0, 255, 0) if robot_link.connected else (0, 165, 255),
                        1,
                        cv2.LINE_AA,
                    )

                    # Rotation-calibration HUD (live tuning state)
                    flip_str = "ON" if _angle_flip else "OFF"
                    cv2.putText(
                        filtered_img,
                        f"Angle: {_angle_offset:+.1f} deg | sign flip: {flip_str}",
                        (30, filtered_img.shape[0] - 45),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (0, 255, 255),
                        1,
                        cv2.LINE_AA,
                    )
                    cv2.putText(
                        filtered_img,
                        "'+'/'-' adjust angle | 'f' flip sign | 'r' reset | 'q' quit",
                        (30, filtered_img.shape[0] - 25),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (200, 200, 200),
                        1,
                        cv2.LINE_AA,
                    )

                    cv2.imshow("Basler Camera Feed", filtered_img)

                    key = cv2.waitKey(1) & 0xFF
                    if key == ord("q"):
                        break
                    elif key in (ord("+"), ord("=")):
                        _angle_offset += _angle_step
                        print(f"Angle offset: {_angle_offset:+.1f} deg")
                    elif key == ord("-"):
                        _angle_offset -= _angle_step
                        print(f"Angle offset: {_angle_offset:+.1f} deg")
                    elif key == ord("f"):
                        _angle_flip = not _angle_flip
                        print(f"Angle sign flip: {'ON' if _angle_flip else 'OFF'}")
                    elif key == ord("r"):
                        _angle_offset = ANGLE_MASTER_OFFSET_DEG
                        _angle_flip = ANGLE_FLIP_SIGN
                        print("Angle tuning reset to the configured defaults.")

        except Exception as e:
            # TimeoutException may not be exposed by every pypylon build, so also
            # detect it by name/text. Either way, keep the frame loop alive.
            is_timeout = type(e).__name__ == "TimeoutException" or "grab timed out" in str(e).lower()

            if not is_timeout:
                # Non-timeout errors (camera lost, internal errors, etc.)
                print(f"Error caught: {e}")
                continue

            consecutive_timeouts += 1
            print(
                f"[Timeout {consecutive_timeouts}/{MAX_CONSECUTIVE_TIMEOUTS}] "
                f"No frame within {RETRIEVE_TIMEOUT_MS / 1000:.0f}s: {e}"
            )

            # A single missed frame is often a hiccup - keep grabbing.
            if consecutive_timeouts < MAX_CONSECUTIVE_TIMEOUTS:
                continue

            # Too many consecutive timeouts: the grab engine is likely stuck.
            grab_restarts += 1
            if grab_restarts > MAX_GRAB_RESTARTS:
                print(
                    f"Giving up after {MAX_GRAB_RESTARTS} restart(s) - "
                    "camera is not streaming."
                )
                break

            print(
                f"Restarting grab engine (attempt {grab_restarts}/"
                f"{MAX_GRAB_RESTARTS})..."
            )
            try:
                camera.StopGrabbing()
            except Exception as stop_err:
                print(f"  StopGrabbing failed: {stop_err}")
            try:
                camera.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)
                consecutive_timeouts = 0  # try a fresh streak after a restart
                print("  Grab engine restarted.")
            except Exception as start_err:
                print(f"  StartGrabbing failed: {start_err}")

    camera.StopGrabbing()
    cv2.destroyAllWindows()
    robot_link.close()