"""AI / computer-vision pipeline for the shape-picking robot.

One file that does everything between "a camera frame" and "a robot command":

  1. Calibrate pixels -> millimetres with an ArUco marker (OpenCV homography).
  2. Find the coloured shapes on the black table (HSV mask + contours).
  3. Crop each shape and run the 3-head ONNX model (class / rotation / centre).
  4. Turn the prediction into robot coordinates (X, Y in mm) and the rotation
     RZ the robot must apply so the piece fits its slot in the tray.

The ONNX model outputs (see ReWorks/neural_net/model.py):
    class_logits     (1, 5)     score per shape class
    rotation_sincos  (1, 5, 2)  [sin(k*angle), cos(k*angle)] per class
    centre           (1, 2)     (cx, cy) as 0..1 of the 128x128 input

Keep the constants below in sync with the training package
(ReWorks/neural_net/config.py) - changing them here alone will desync the model.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np
import onnxruntime as ort

# ── Paths ────────────────────────────────────────────────────────────────────
APP_DIR = Path(__file__).resolve().parent
ROOT_DIR = APP_DIR.parent

# First model that exists wins. crpnet.onnx is the newest export.
MODEL_CANDIDATES = [
    ROOT_DIR / "ReWorks" / "neural_net" / "models" / "crpnet.onnx",
    ROOT_DIR / "MISC" / "shape_3head_v2.onnx",
    APP_DIR / "crpnet.onnx",
    APP_DIR / "shape_3head_v2.onnx",
]


def default_model_path() -> str:
    for path in MODEL_CANDIDATES:
        if path.exists():
            return str(path)
    raise FileNotFoundError(
        "No ONNX model found. Looked in:\n  " + "\n  ".join(str(p) for p in MODEL_CANDIDATES)
    )


# ── Calibration (must match how the dataset was captured) ────────────────────
# Which marker corner is (0, 0). OpenCV corner order: 0=TL, 1=TR, 2=BR, 3=BL
# in the marker's own frame. 2 = the left-most corner in the camera image.
ORIGIN_CORNER = 2

# The angle each piece must have in its tray slot (wobjAiPlaceTray, degrees CCW
# from above). Measured from the GT_angle photos.
SLOT_ANGLE_ROBOT_DEG = {"Circle": 0.0, "Hexagon": -5.4, "Square": -1.7, "Star": 32.2, "Triangle": 58.4}

# The marker frame (Y pointing "down" like the image) is mirrored versus the
# robot frame (Z up), so angles flip sign marker -> robot.
MARKER_Y_FLIPPED = True
# Extra rotation between the marker X axis and the robot X axis.
MARKER_TO_ROBOT_ROT_DEG = 0.0

# Where X/Y come from: "ai" = network head 3, "cv2" = centre of the mask.
CENTER_SOURCE = "ai"

IMG_SIZE = 128
CANVAS_MARGIN = 0.15  # black border around the crop before resize (as in training)

# Robot key per class, matching the tray targets in RobotCode/MainModule.mod
# (move_to_shape switches on these).
ROBOT_SHAPE_KEY = {
    "Circle": "pcircle",
    "Hexagon": "phexa",
    "Square": "psquare",
    "Star": "pstar",
    "Triangle": "ptriangle",
}


def wrap_period(deg: float, period: float) -> float:
    """Fold an angle into [-period/2, period/2). E.g. square (90): 100 -> 10."""
    return (deg + period / 2.0) % period - period / 2.0


@dataclass
class DetectedShape:
    """Everything about one detected shape, ready to send to the ABB robot."""

    shape: str
    x_mm: float
    y_mm: float
    rz_deg: float
    confidence: float
    cx_px: float = 0.0
    cy_px: float = 0.0
    bbox: Tuple[int, int, int, int] = (0, 0, 0, 0)
    theta_img_deg: float = 0.0                    # piece angle in the image (for drawing)
    contour_px: Optional[np.ndarray] = None       # full-image contour (for the ghost outline)

    def to_robot_command(self) -> str:
        """Text the RAPID code reads from the socket (see RobotCode/ServerComm.mod)."""
        key = ROBOT_SHAPE_KEY.get(self.shape, self.shape.lower())
        return f"SHAPE={key};X={self.x_mm:.1f};Y={self.y_mm:.1f};RZ={self.rz_deg:.1f};END\r\n"


@dataclass
class Prediction:
    """Raw network output for one preprocessed crop."""

    shape: str
    class_idx: int
    confidence: float
    angle_img_deg: float          # rotation in the image frame (0 for circles)
    centre_norm: Tuple[float, float]  # (cx, cy) as 0..1 of the square canvas


class VisionPipeline:
    CLASSES = ["Circle", "Hexagon", "Square", "Star", "Triangle"]
    SYMMETRY = {"Circle": 360.0, "Hexagon": 60.0, "Triangle": 120.0, "Square": 90.0, "Star": 72.0}

    def __init__(
        self,
        model_path: Optional[str] = None,
        marker_size_mm: float = 100.0,
        target_shapes: Optional[List[str]] = None,
        pad_frac: float = 0.25,
    ):
        self.model_path = model_path or default_model_path()
        self.marker_size_mm = marker_size_mm
        self.target_shapes = target_shapes or list(self.CLASSES)
        self.pad_frac = pad_frac
        self.homography_matrix: Optional[np.ndarray] = None

        # ImageNet normalisation used during training.
        self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        self.std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

        # ArUco detection (DICT_5X5_100) with sub-pixel corner refinement.
        self.aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_100)
        self.aruco_params = cv2.aruco.DetectorParameters()
        self.aruco_params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
        self.aruco_detector = (
            cv2.aruco.ArucoDetector(self.aruco_dict, self.aruco_params)
            if hasattr(cv2.aruco, "ArucoDetector")
            else None
        )

        # Load the ONNX model.
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = 4
        opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.session = ort.InferenceSession(
            self.model_path, sess_options=opts, providers=["CPUExecutionProvider"]
        )
        self.input_name = self.session.get_inputs()[0].name

        angle_shape = self.session.get_outputs()[1].shape
        if len(angle_shape) != 3:
            raise ValueError(
                f"{self.model_path} is an old (v1) model. Export a v2 3-head model instead."
            )
        print(f"[VISION] Loaded {self.model_path}")

    # ── Calibration: pixels -> mm ────────────────────────────────────────────
    def detect_marker(self, frame_bgr: np.ndarray):
        """Detect every ArUco marker in the frame; returns OpenCV (corners, ids)."""
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        if self.aruco_detector is not None:
            corners, ids, _ = self.aruco_detector.detectMarkers(gray)
        else:
            corners, ids, _ = cv2.aruco.detectMarkers(gray, self.aruco_dict, parameters=self.aruco_params)
        return corners, ids

    def _homography_from_corner(self, corner_px: np.ndarray) -> np.ndarray:
        """Pixel -> mm homography for one marker, with (0, 0) at ORIGIN_CORNER."""
        pts_px = np.roll(corner_px.reshape(4, 2), -ORIGIN_CORNER, axis=0).astype(np.float64)
        s = self.marker_size_mm
        pts_mm = np.array([[0, 0], [s, 0], [s, s], [0, s]], dtype=np.float64)
        H, _ = cv2.findHomography(pts_px, pts_mm)
        return H

    def update_homography(self, frame_bgr: np.ndarray) -> bool:
        """Find the ArUco marker and compute the pixel -> mm homography.

        The origin (0, 0) is the marker corner ORIGIN_CORNER. Returns True if a
        marker was found.
        """
        corners, ids = self.detect_marker(frame_bgr)
        if ids is None or len(ids) == 0:
            return False
        self.homography_matrix = self._homography_from_corner(corners[0])
        return True

    # Alias kept for backwards compatibility with the old scripts.
    updateHomography = update_homography

    def pixel_to_mm(self, px: float, py: float) -> Optional[Tuple[float, float]]:
        """Pixel (x, y) -> mm on the table, measured from the marker origin."""
        if self.homography_matrix is None:
            return None
        pt = np.array([[[px, py]]], dtype=np.float64)
        x_mm, y_mm = cv2.perspectiveTransform(pt, self.homography_matrix)[0, 0]
        return round(float(x_mm), 1), round(float(y_mm), 1)

    def image_angle_to_marker_deg(self, cx: float, cy: float, theta_img_deg: float) -> float:
        """A direction in the image at (cx, cy) -> angle in the marker frame."""
        r = np.radians(theta_img_deg)
        pts = np.array([[[cx, cy]], [[cx + 40.0 * np.cos(r), cy + 40.0 * np.sin(r)]]], dtype=np.float64)
        a, b = cv2.perspectiveTransform(pts, self.homography_matrix).reshape(2, 2)
        return float(np.degrees(np.arctan2(b[1] - a[1], b[0] - a[0])))

    def robot_rz(self, shape: str, marker_angle_deg: float) -> float:
        """Rotation (deg, CCW from above, around the robot Z axis) = slot angle -
        piece angle, both in the robot frame, folded by the shape's symmetry."""
        if shape == "Circle":
            return 0.0
        piece_robot = -marker_angle_deg if MARKER_Y_FLIPPED else marker_angle_deg
        piece_robot += MARKER_TO_ROBOT_ROT_DEG
        rz = SLOT_ANGLE_ROBOT_DEG[shape] - piece_robot
        return round(wrap_period(rz, self.SYMMETRY[shape]), 1)

    # ── Shape extraction (HSV mask, same look as the dataset) ────────────────
    def extract_rois(
        self,
        frame_bgr: np.ndarray,
        min_area: int = 15000,
        max_area: int = 70000,
    ) -> List[Tuple[Tuple[int, int, int, int], np.ndarray]]:
        """Black out the background with an HSV colour mask and crop each shape
        with padding, returning [(bbox, crop), ...]."""
        h_img, w_img = frame_bgr.shape[:2]

        hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, np.array([0, 50, 50]), np.array([179, 255, 255]))

        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        filtered = cv2.bitwise_and(frame_bgr, frame_bgr, mask=mask)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        rois: List[Tuple[Tuple[int, int, int, int], np.ndarray]] = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < min_area or area > max_area:
                continue
            x, y, w, h = cv2.boundingRect(cnt)
            # Ignore the beams at the top and the robot arm at the bottom.
            if y < int(h_img * 0.05) or (y + h) > int(h_img * 0.85):
                continue
            # Real shapes are roughly square; drop long streaks / cables.
            aspect = w / float(h) if h > 0 else 0
            if aspect < 0.45 or aspect > 2.2:
                continue

            x1, y1 = self._crop_origin((x, y, w, h))
            pad = int(max(w, h) * self.pad_frac)
            x2 = min(w_img, x + w + pad)
            y2 = min(h_img, y + h + pad)

            roi = filtered[y1:y2, x1:x2]
            if roi.size > 0:
                rois.append(((x, y, w, h), roi))
        return rois

    def _crop_origin(self, bbox: Tuple[int, int, int, int]) -> Tuple[int, int]:
        """Top-left corner of the padded crop in the full image."""
        x, y, w, h = bbox
        pad = int(max(w, h) * self.pad_frac)
        return max(0, x - pad), max(0, y - pad)

    # ── Preprocessing + inference ────────────────────────────────────────────
    def preprocess(self, crop_bgr: np.ndarray):
        """Centre a crop on a black square (no squashing), resize to 128,
        normalise. Returns (tensor (1,3,128,128), (x0, y0, side))."""
        h, w = crop_bgr.shape[:2]
        side = int(round(max(h, w) * (1.0 + 2.0 * CANVAS_MARGIN)))
        canvas = np.zeros((side, side, 3), dtype=crop_bgr.dtype)
        x0, y0 = (side - w) // 2, (side - h) // 2
        canvas[y0:y0 + h, x0:x0 + w] = crop_bgr

        small = cv2.resize(canvas, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)
        rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        normalized = (rgb - self.mean) / self.std
        tensor = np.expand_dims(np.transpose(normalized, (2, 0, 1)), axis=0).astype(np.float32)
        return tensor, (x0, y0, side)

    def predict(self, tensor: np.ndarray) -> Prediction:
        """Run the model on a preprocessed tensor and parse the three heads."""
        logits, rotation_out, centre_out = self.session.run(None, {self.input_name: tensor})

        logits = logits[0]
        class_idx = int(np.argmax(logits))
        shape_name = self.CLASSES[class_idx]

        exp = np.exp(logits - np.max(logits))
        confidence = float(exp[class_idx] / np.sum(exp))

        # Rotation for the predicted class (row class_idx of the (5, 2) head).
        angle_img = 0.0
        if shape_name != "Circle":
            k = 360.0 / self.SYMMETRY[shape_name]
            sin_v, cos_v = (float(v) for v in rotation_out[0][class_idx])
            angle_img = float(np.degrees(np.arctan2(sin_v, cos_v)) / k)

        cx_norm, cy_norm = float(centre_out[0][0]), float(centre_out[0][1])
        return Prediction(shape_name, class_idx, confidence, angle_img, (cx_norm, cy_norm))

    def infer_roi(
        self, bbox: Tuple[int, int, int, int], roi_bgr: np.ndarray, min_confidence: float = 0.85
    ) -> Optional[DetectedShape]:
        """Run the model on one crop and turn the answer into a DetectedShape."""
        x, y, w, h = bbox
        tensor, (x0, y0, side) = self.preprocess(roi_bgr)
        pred = self.predict(tensor)

        if pred.shape not in self.target_shapes:
            return None
        print(f"  Candidate at ({x},{y}): {pred.shape} | Conf: {pred.confidence*100:.1f}% | Area: {w*h}px")
        if pred.confidence < min_confidence:
            return None

        # Centre in full-image pixels.
        crop_x, crop_y = self._crop_origin(bbox)
        if CENTER_SOURCE == "cv2":
            gray = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)
            m = cv2.moments((gray > 0).astype(np.uint8), binaryImage=True)
            cx_roi, cy_roi = m["m10"] / m["m00"], m["m01"] / m["m00"]
        else:
            cx_roi = pred.centre_norm[0] * side - x0
            cy_roi = pred.centre_norm[1] * side - y0
        cx_px, cy_px = crop_x + cx_roi, crop_y + cy_roi

        # Contour of the shape in full-image pixels (for the ghost outline).
        roi_mask = (cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY) > 0).astype(np.uint8)
        contours, _ = cv2.findContours(roi_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contour_px = (
            max(contours, key=cv2.contourArea).reshape(-1, 2) + [crop_x, crop_y]
            if contours
            else None
        )

        # mm on the table + RZ (without a marker we fall back to pixels and RZ = 0).
        coords_mm = self.pixel_to_mm(cx_px, cy_px)
        if coords_mm is None:
            coords_mm = (round(float(cx_px), 1), round(float(cy_px), 1))
            rz = 0.0
        else:
            rz = self.robot_rz(pred.shape, self.image_angle_to_marker_deg(cx_px, cy_px, pred.angle_img_deg))

        return DetectedShape(
            shape=pred.shape,
            x_mm=coords_mm[0],
            y_mm=coords_mm[1],
            rz_deg=rz,
            confidence=round(pred.confidence, 2),
            cx_px=cx_px,
            cy_px=cy_px,
            bbox=bbox,
            theta_img_deg=pred.angle_img_deg,
            contour_px=contour_px,
        )

    def process_frame(self, frame_bgr: np.ndarray) -> List[DetectedShape]:
        """A whole frame: calibrate (first time), find shapes, run the network,
        convert to mm."""
        if self.homography_matrix is None:
            self.update_homography(frame_bgr)

        detections: List[DetectedShape] = []
        for bbox, roi in self.extract_rois(frame_bgr):
            obj = self.infer_roi(bbox, roi)
            if obj is not None:
                detections.append(obj)
        return detections

    def draw_marker_frame(
        self,
        frame_bgr: np.ndarray,
        axis_len_mm: float = 80.0,
        grid_step_mm: float = 50.0,
    ) -> np.ndarray:
        """Overlay the marker frame the robot's wobjAruco must match.

        Shows, so the frame can be taught by eye:
          * all four marker corners labelled with their OpenCV index 0..3
            (0=TL, 1=TR, 2=BR, 3=BL in the marker's own frame),
          * ORIGIN_CORNER highlighted = the point to teach as wobjAruco's origin,
          * +X (green) and +Y (blue) arrows = the wobjAruco axis directions,
          * a light 50 mm grid for scale.

        The corner -> mm mapping matches update_homography exactly, so what is
        drawn is what the pipeline actually uses for X/Y in mm.
        """
        annotated = frame_bgr.copy()
        s = self.marker_size_mm

        corners, ids = self.detect_marker(frame_bgr)
        if ids is None or len(ids) == 0:
            cv2.putText(
                annotated, "ArUco marker: NOT FOUND", (20, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2, cv2.LINE_AA,
            )
            return annotated

        pts = corners[0].reshape(4, 2).astype(np.float64)

        # Marker mm -> pixels, using the same convention as update_homography.
        H_inv = np.linalg.inv(self._homography_from_corner(corners[0]))
        o_px = cv2.perspectiveTransform(np.array([[[0.0, 0.0]]]), H_inv).reshape(2)
        x_px = cv2.perspectiveTransform(np.array([[[axis_len_mm, 0.0]]]), H_inv).reshape(2)
        y_px = cv2.perspectiveTransform(np.array([[[0.0, axis_len_mm]]]), H_inv).reshape(2)

        # Light reference grid inside the marker square.
        if grid_step_mm and grid_step_mm > 0:
            for i in range(int(round(s / grid_step_mm)) + 1):
                mm = i * grid_step_mm
                e = cv2.perspectiveTransform(
                    np.array([[[mm, 0.0]], [[mm, s]], [[0.0, mm]], [[s, mm]]]), H_inv
                ).reshape(4, 2).astype(np.int32)
                cv2.line(annotated, tuple(e[0]), tuple(e[1]), (170, 170, 170), 1, cv2.LINE_AA)
                cv2.line(annotated, tuple(e[2]), tuple(e[3]), (170, 170, 170), 1, cv2.LINE_AA)

        # Marker outline + corner index labels.
        cv2.polylines(annotated, [pts.astype(np.int32)], True, (255, 200, 0), 2, cv2.LINE_AA)
        for i, (px, py) in enumerate(pts):
            is_origin = i == ORIGIN_CORNER
            color = (0, 215, 255) if is_origin else (255, 255, 255)
            cv2.circle(annotated, (int(px), int(py)), 7 if is_origin else 4, color, -1)
            cv2.putText(
                annotated, str(i), (int(px) + 9, int(py) - 9),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA,
            )

        # Marker-frame axes = wobjAruco X and Y.
        o = tuple(o_px.astype(int))
        cv2.arrowedLine(annotated, o, tuple(x_px.astype(int)), (0, 255, 0), 3, cv2.LINE_AA, 0, 0.15)
        cv2.arrowedLine(annotated, o, tuple(y_px.astype(int)), (255, 0, 0), 3, cv2.LINE_AA, 0, 0.15)
        cv2.putText(annotated, f"+X ({axis_len_mm:.0f}mm)", (int(x_px[0]) + 10, int(x_px[1])),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.putText(annotated, f"+Y ({axis_len_mm:.0f}mm)", (int(y_px[0]) + 10, int(y_px[1])),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2, cv2.LINE_AA)
        cv2.putText(
            annotated,
            f"ORIGIN = corner {ORIGIN_CORNER} -> teach as wobjAruco (0,0). Marker {s:.0f}mm",
            (int(o_px[0]) + 12, int(o_px[1]) + 28),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 215, 255), 2, cv2.LINE_AA,
        )
        return annotated

    # ── Drawing ──────────────────────────────────────────────────────────────
    def ghost_outline(self, obj: DetectedShape) -> Optional[np.ndarray]:
        """The piece's contour after the robot rotates it by RZ, in full-image
        pixels (the magenta "ghost")."""
        if obj.contour_px is None or self.homography_matrix is None:
            return None
        H = self.homography_matrix
        pts_mm = cv2.perspectiveTransform(obj.contour_px.reshape(-1, 1, 2).astype(np.float64), H).reshape(-1, 2)
        c_mm = cv2.perspectiveTransform(np.array([[[obj.cx_px, obj.cy_px]]], dtype=np.float64), H).reshape(2)
        r = np.radians(-obj.rz_deg if MARKER_Y_FLIPPED else obj.rz_deg)
        R = np.array([[np.cos(r), -np.sin(r)], [np.sin(r), np.cos(r)]])
        ghost_mm = (pts_mm - c_mm) @ R.T + c_mm
        ghost_px = cv2.perspectiveTransform(ghost_mm.reshape(-1, 1, 2), np.linalg.inv(H)).reshape(-1, 2)
        return ghost_px.astype(np.int32)

    def draw_hud(self, frame_bgr: np.ndarray, detections: List[DetectedShape]) -> np.ndarray:
        """Draw bbox, ghost, centre, angle arrow and a label for every shape."""
        annotated = frame_bgr.copy()
        for obj in detections:
            x, y, w, h = obj.bbox
            cv2.rectangle(annotated, (x, y), (x + w, y + h), (0, 255, 0), 2)

            ghost = self.ghost_outline(obj)
            if ghost is not None:
                cv2.polylines(annotated, [ghost], True, (255, 0, 255), 3, cv2.LINE_AA)

            cx, cy = int(obj.cx_px), int(obj.cy_px)
            cv2.circle(annotated, (cx, cy), 6, (0, 0, 255), -1)

            if obj.shape != "Circle":
                r = np.radians(obj.theta_img_deg)
                end = (int(cx + 45 * np.cos(r)), int(cy + 45 * np.sin(r)))
                cv2.arrowedLine(annotated, (cx, cy), end, (255, 50, 50), 3, tipLength=0.3)

            label = f"{obj.shape} ({int(obj.confidence*100)}%): X={obj.x_mm}, Y={obj.y_mm}, RZ={obj.rz_deg} deg"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            badge_y1 = max(0, y - th - 12)
            badge_y2 = max(th + 12, y)
            cv2.rectangle(annotated, (x, badge_y1), (x + tw + 10, badge_y2), (0, 0, 0), -1)
            cv2.putText(annotated, label, (x + 5, badge_y2 - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2, cv2.LINE_AA)
        return annotated


if __name__ == "__main__":
    pipeline = VisionPipeline()
    print("VisionPipeline initialised OK.")
    print("Inputs:", [i.name for i in pipeline.session.get_inputs()])
    print("Outputs:", [o.name for o in pipeline.session.get_outputs()])
