# Author Nils Wikström niwi0007
"""
Vison pipeline for shape detection and robot corindate extraction
arUco tag detection for real world calibraton (pixels -> mm), origin = ORIGIN_CORNER of the marker
3 - head ONNX Neural Network v2 (class + rotation + centre)
RZ = how much the robot must turn the piece so it fits its slot in the tray
"""

from dataclasses import dataclass
from typing import List, Optional, Tuple
import cv2
import numpy as np
import onnxruntime as ort


# ── Calibration constants (must match the notebook Shapeclassifer(2).ipynb) ───
# Which marker corner is (0, 0). OpenCV order: 0=TL, 1=TR, 2=BR, 3=BL (marker's own frame).
# 2 = the leftmost corner in the camera view. X points to the next corner, Y to the previous one.
ORIGIN_CORNER = 2

# Angle each piece must have in its tray slot, in the robot's tray frame
# (wobjAiPlaceTray, degrees, counter-clockwise seen from above). From the GT_angle photos.
SLOT_ANGLE_ROBOT_DEG = {"Circle": 0.0, "Hexagon": 25.2, "Square": 43.4, "Star": 29.9, "Triangle": 58.6}

# The marker frame (Y "down" like the image) is mirrored compared with the robot frame (Z up),
# so angles change sign marker -> robot. Proven by the GT photos (see notebook).
MARKER_Y_FLIPPED = True
# Extra turn between the marker's X axis and the robot's X axis (0 = wobjAruco X along marker X).
MARKER_TO_ROBOT_ROT_DEG = 0.0

# Where X/Y comes from: "ai" = head 3 of the network, "cv2" = centroid of the shape's mask
CENTER_SOURCE = "ai"

IMG_SIZE = 128
CANVAS_MARGIN = 0.15   # black border added before resizing (same as training)


def wrap_period(deg: float, period: float) -> float:
    """Wrap an angle into [-period/2, period/2)."""
    return (deg + period / 2.0) % period - period / 2.0


@dataclass
class DetectedShape:
    """Stores all information about a detected shape ready for the ABB robot."""
    shape: str
    x_mm: float
    y_mm: float
    rz_deg: float
    confidence: float
    cx_px: float = 0.0
    cy_px: float = 0.0
    bbox: Tuple[int, int, int, int] = (0, 0, 0, 0)
    theta_img_deg: float = 0.0        # piece angle in the image (for drawing)
    contour_px: Optional[np.ndarray] = None   # outline in full-frame pixels (for the ghost)

    def to_robot_command(self) -> str:
        """Return the string expected by the ABB RAPID socket parser."""
        return f"SHAPE={self.shape};X={self.x_mm:.1f};Y={self.y_mm:.1f};RZ={self.rz_deg:.1f};END\r\n"

    def toRobortCommand(self) -> str:
        return self.to_robot_command()



class VisionPipeline:
    CLASSES =["Circle","Hexagon","Square","Star","Triangle"]
    SYMMETRY = {
        "Circle": 360.0,
        "Hexagon": 60.0,
        "Triangle": 120.0,
        "Square": 90.0,
        "Star": 72.0,
    }
    def __init__(
        self,
        model_path: str = "shape_3head_v2.onnx",
        marker_size_mm: float = 100.0,
        target_shapes: Optional[List[str]] = None,
        pad_frac: float = 0.25,
    ):
        self.marker_size_mm = marker_size_mm
        self.target_shapes = target_shapes or ["Hexagon", "Triangle", "Circle", "Square", "Star"]
        self.homography_matrix: Optional[np.ndarray] = None
        self.pad_frac = pad_frac

        # Normalize constants used during model training
        self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        self.std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

        # ArUco marker detection (DICT_5X5_100) with sub-pixel corners
        self.aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_100)
        self.aruco_params = cv2.aruco.DetectorParameters()
        self.aruco_params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
        if hasattr(cv2.aruco, "ArucoDetector"):
            self.aruco_detector = cv2.aruco.ArucoDetector(self.aruco_dict, self.aruco_params)
        else:
            self.aruco_detector = None

        # ONNX inference
        session_opts = ort.SessionOptions()
        session_opts.intra_op_num_threads = 4
        session_opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        session_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        self.session = ort.InferenceSession(
            model_path, sess_options=session_opts, providers=["CPUExecutionProvider"]
        )
        self.input_name = self.session.get_inputs()[0].name

        # v2 model: angle output is one [sin kθ, cos kθ] pair per class -> shape (1, 5, 2)
        angle_shape = self.session.get_outputs()[1].shape
        if len(angle_shape) != 3:
            raise ValueError(
                f"{model_path} is an old (v1) model. Export shape_3head_v2.onnx from the notebook."
            )

    def update_homography(self, frame_bgr: np.ndarray) -> bool:
        """
        Detects ArUco marker and computes the homography matrix (pixels -> mm).
        Origin (0, 0) is marker corner ORIGIN_CORNER.
        Returns True if marker is detected.
        """
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        if self.aruco_detector is not None:
            corners, ids, _ = self.aruco_detector.detectMarkers(gray)
        else:
            corners, ids, _ = cv2.aruco.detectMarkers(
                gray, self.aruco_dict, parameters=self.aruco_params
            )
        if ids is None or len(ids) == 0:
            return False

        pts_px = np.roll(corners[0].reshape(4, 2), -ORIGIN_CORNER, axis=0).astype(np.float64)
        pts_mm = np.array(
            [
                [0.0, 0.0],
                [self.marker_size_mm, 0.0],
                [self.marker_size_mm, self.marker_size_mm],
                [0.0, self.marker_size_mm],
            ],
            dtype=np.float64,
        )
        self.homography_matrix, _ = cv2.findHomography(pts_px, pts_mm)
        return True

    def updateHomography(self, frame_bgr: np.ndarray) -> bool:
        return self.update_homography(frame_bgr)


    def pixel_to_mm(self,px:float,py:float,) -> Optional[Tuple[float,float]]:
        """
        Transform a pixel x,y cord to ORL workspace mm (measured from the ArUco origin corner)
        """

        if self.homography_matrix is None:
            return None

        pt = np.array([[[px, py]]], dtype=np.float64)
        x_mm, y_mm = cv2.perspectiveTransform(pt, self.homography_matrix)[0, 0]

        return round(float(x_mm),1), round(float(y_mm),1)

    def image_angle_to_marker_deg(self, cx: float, cy: float, theta_img_deg: float) -> float:
        """Maps a direction at (cx, cy) in the image through the homography -> angle in the marker frame."""
        r = np.radians(theta_img_deg)
        pts = np.array([[[cx, cy]], [[cx + 40.0 * np.cos(r), cy + 40.0 * np.sin(r)]]], dtype=np.float64)
        a, b = cv2.perspectiveTransform(pts, self.homography_matrix).reshape(2, 2)
        return float(np.degrees(np.arctan2(b[1] - a[1], b[0] - a[0])))

    def robot_rz(self, shape: str, marker_angle_deg: float) -> float:
        """
        Rotation difference the robot must apply (degrees, CCW seen from above, about the robot Z axis):
        slot angle - piece angle, both in the robot frame, smallest turn using the shape's symmetry.
        """
        if shape == "Circle":
            return 0.0
        piece_robot = -marker_angle_deg if MARKER_Y_FLIPPED else marker_angle_deg
        piece_robot += MARKER_TO_ROBOT_ROT_DEG
        rz = SLOT_ANGLE_ROBOT_DEG[shape] - piece_robot
        return round(wrap_period(rz, self.SYMMETRY[shape]), 1)


    def extract_rois(
        self,
        frame_bgr: np.ndarray,
        min_area: int = 15000,
        max_area: int = 70000,
    ) -> List[Tuple[Tuple[int, int, int, int], np.ndarray]]:
        """
        Applies HSV color masking to black-out the background (matching dataset/objects),
        filters out workspace boundaries / robot arm, and crops ROIs with 25% padding.
        """
        h_img, w_img = frame_bgr.shape[:2]

        # 1. HSV color filter: keeps colored shapes, sets table/shadows to black
        hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
        lower_color = np.array([0, 50, 50])
        upper_color = np.array([179, 255, 255])
        mask = cv2.inRange(hsv, lower_color, upper_color)

        # 2. Smooth out noise
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        # 3. Mask out the background to pure black (EXACTLY like dataset images)
        filtered_img = cv2.bitwise_and(frame_bgr, frame_bgr, mask=mask)

        # 4. Find shape contours from the mask
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        rois = []

        for cnt in contours:
            area = cv2.contourArea(cnt)
            # Filter 1: Size limits (reject noise < 15,000px and giant background blobs > 70,000px)
            if area < min_area or area > max_area:
                continue

            x, y, w, h = cv2.boundingRect(cnt)

            # Filter 2: Workspace boundaries (ignore top ceiling beams and bottom robot arm)
            if y < int(h_img * 0.05) or (y + h) > int(h_img * 0.85):
                continue

            # Filter 3: Aspect ratio (real shapes are roughly 1:1, reject long streaks/cables)
            aspect = w / float(h) if h > 0 else 0
            if aspect < 0.45 or aspect > 2.2:
                continue

            # 5. Proportional padding (25% of max dimension, matching capture_dataset.py)
            x1, y1 = self._crop_origin((x, y, w, h))
            pad = int(max(w, h) * self.pad_frac)
            y2 = min(h_img, y + h + pad)
            x2 = min(w_img, x + w + pad)

            # 6. Crop from the MASKED image (black background)
            roi = filtered_img[y1:y2, x1:x2]
            if roi.size > 0:
                rois.append(((x, y, w, h), roi))

        return rois

    def _crop_origin(self, bbox: Tuple[int, int, int, int]) -> Tuple[int, int]:
        """Top-left corner of the padded crop in the full frame."""
        x, y, w, h = bbox
        pad = int(max(w, h) * self.pad_frac)
        return max(0, x - pad), max(0, y - pad)

    def _preprocess(self, roi_bgr: np.ndarray):
        """Pads the crop to a centred square (so it isn't squashed), resizes to 128, normalizes."""
        h, w = roi_bgr.shape[:2]
        side = int(round(max(h, w) * (1.0 + 2.0 * CANVAS_MARGIN)))
        canvas = np.zeros((side, side, 3), dtype=roi_bgr.dtype)
        x0, y0 = (side - w) // 2, (side - h) // 2
        canvas[y0:y0 + h, x0:x0 + w] = roi_bgr

        small = cv2.resize(canvas, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)
        rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        normalized = (rgb - self.mean) / self.std
        tensor = np.expand_dims(np.transpose(normalized, (2, 0, 1)), axis=0).astype(np.float32)
        return tensor, (x0, y0, side)

    def infer_roi(
        self, bbox: Tuple[int, int, int, int], roi_bgr: np.ndarray, min_confidence: float = 0.85
    ) -> Optional[DetectedShape]:
        """Runs the 3-head ONNX model on a single cropped shape."""
        x, y, w, h = bbox

        tensor, (x0, y0, side) = self._preprocess(roi_bgr)

        # Run model inference
        outputs = self.session.run(None, {self.input_name: tensor})

        # Head 1: Shape Classification
        logits = outputs[0][0]
        class_idx = int(np.argmax(logits))
        shape_name = self.CLASSES[class_idx]

        if shape_name not in self.target_shapes:
            return None

        # Confidence calculation
        exp_logits = np.exp(logits - np.max(logits))
        confidence = float(exp_logits[class_idx] / np.sum(exp_logits))

        print(f"  Candidate at ({x},{y}): {shape_name} | Conf: {confidence*100:.1f}% | Area: {w*h}px")

        # Filter 4: Reject low-confidence predictions
        if confidence < min_confidence:
            return None

        # Head 3: centre, normalized on the square canvas -> full frame pixels
        crop_x, crop_y = self._crop_origin(bbox)
        roi_mask = (cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY) > 0).astype(np.uint8)
        contours, _ = cv2.findContours(roi_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contour_px = (max(contours, key=cv2.contourArea).reshape(-1, 2) + [crop_x, crop_y]) if contours else None
        if CENTER_SOURCE == "cv2":
            gray = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)
            M = cv2.moments((gray > 0).astype(np.uint8), binaryImage=True)
            cx_roi, cy_roi = M["m10"] / M["m00"], M["m01"] / M["m00"]
        else:
            cx_roi = float(outputs[2][0][0]) * side - x0
            cy_roi = float(outputs[2][0][1]) * side - y0
        cx_px = crop_x + cx_roi
        cy_px = crop_y + cy_roi

        # Head 2: angle in the image, using the row of the predicted class: θ = atan2(s, c) / k
        sym = self.SYMMETRY[shape_name]
        theta_img = 0.0
        if shape_name != "Circle":
            k = 360.0 / sym
            sin_v, cos_v = (float(v) for v in outputs[1][0][class_idx])
            theta_img = float(np.degrees(np.arctan2(sin_v, cos_v)) / k)

        # Translate to robot table mm + rotation difference (fallback to pixels if no ArUco marker)
        coords_mm = self.pixel_to_mm(cx_px, cy_px)
        if coords_mm is None:
            coords_mm = (round(float(cx_px), 1), round(float(cy_px), 1))
            rz = 0.0
        else:
            rz = self.robot_rz(shape_name, self.image_angle_to_marker_deg(cx_px, cy_px, theta_img))

        return DetectedShape(
            shape=shape_name,
            x_mm=coords_mm[0],
            y_mm=coords_mm[1],
            rz_deg=rz,
            confidence=round(confidence, 2),
            cx_px=cx_px,
            cy_px=cy_px,
            bbox=bbox,
            theta_img_deg=theta_img,
            contour_px=contour_px,
        )

    def process_frame(self, frame_bgr: np.ndarray) -> List[DetectedShape]:
        """
        Processes a frame: calibrates, detects, infers, and translates coords.
        """
        if self.homography_matrix is None:
            self.update_homography(frame_bgr)

        detections: List[DetectedShape] = []

        for bbox, roi in self.extract_rois(frame_bgr):
            obj = self.infer_roi(bbox, roi)
            if obj is not None:
                detections.append(obj)

        return detections

    def ghost_outline(self, obj: DetectedShape) -> Optional[np.ndarray]:
        """
        The piece's outline after the robot turns it by RZ, in full-frame pixels.
        The robot turns +RZ (CCW from above) = -RZ in the mirrored marker frame, so the
        outline is rotated by -RZ in mm around the centre and mapped back to pixels.
        """
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

    def draw_hud(
        self, frame_bgr: np.ndarray, detections: List[DetectedShape]
    ) -> np.ndarray:
        """
        Draws bounding boxes, shape labels, millimeter coordinates,
        and orientation vectors on the frame with high-contrast text badges.
        """
        annotated = frame_bgr.copy()

        for obj in detections:
            x, y, w, h = obj.bbox

            # Draw bounding box
            cv2.rectangle(annotated, (x, y), (x + w, y + h), (0, 255, 0), 2)

            # Draw ghost: the piece after the robot turns it by RZ (magenta)
            ghost = self.ghost_outline(obj)
            if ghost is not None:
                cv2.polylines(annotated, [ghost], True, (255, 0, 255), 3, cv2.LINE_AA)

            # Draw center point
            cx, cy = int(obj.cx_px), int(obj.cy_px)
            cv2.circle(annotated, (cx, cy), 6, (0, 0, 255), -1)

            # Draw orientation arrow (the piece's measured angle in the image)
            if obj.shape != "Circle":
                angle_rad = np.radians(obj.theta_img_deg)
                arrow_len = 45
                end_x = int(cx + arrow_len * np.cos(angle_rad))
                end_y = int(cy + arrow_len * np.sin(angle_rad))
                cv2.arrowedLine(annotated, (cx, cy), (end_x, end_y), (255, 50, 50), 3, tipLength=0.3)

            # High-contrast text label badge
            label = f"{obj.shape} ({int(obj.confidence*100)}%): X={obj.x_mm}, Y={obj.y_mm}, RZ={obj.rz_deg} deg"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)

            badge_y1 = max(0, y - th - 12)
            badge_y2 = max(th + 12, y)
            cv2.rectangle(annotated, (x, badge_y1), (x + tw + 10, badge_y2), (0, 0, 0), -1)
            cv2.putText(
                annotated,
                label,
                (x + 5, badge_y2 - 6),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 255),
                2,
                cv2.LINE_AA,
            )

        return annotated
if __name__ == "__main__":
    print("Testing VisionPipeline initialization...")
    pipeline = VisionPipeline(model_path="shape_3head_v2.onnx")
    print("VisionPipeline initialized successfully!")
