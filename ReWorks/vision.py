# Author Nils Wikström niwi0007
"""
Vison pipeline for shape detection and robot corindate extraction
arUco tag detection for real world calibraton (pixels -> mm )
3 - head ONNX Neural Network (class + rotation + centroid)
symetry awareness pick angle for robot gripper
"""

from dataclasses import dataclass
from typing import List, Optional, Tuple
import cv2
import numpy as np
import onnxruntime as ort


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

    def to_robot_command(self) -> str:
        """Return the string expected by the ABB RAPID socket parser."""
        return f"SHAPE={self.shape};X={self.x_mm};Y={self.y_mm};RZ={self.rz_deg};END\r\n"

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
        model_path: str = "shape_3head_model.onnx",
        marker_size_mm: float = 100.0,
        target_shapes: Optional[List[str]] = None,
    ):
        self.marker_size_mm = marker_size_mm
        self.target_shapes = target_shapes or ["Hexagon", "Triangle", "Circle", "Square", "Star"]
        self.homography_matrix: Optional[np.ndarray] = None

        # Normalize constants used during model training
        self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        self.std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

        # ArUco marker detection (DICT_5X5_100)
        self.aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_100)
        self.aruco_params = cv2.aruco.DetectorParameters()
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

    def update_homography(self, frame_bgr: np.ndarray) -> bool:
        """
        Detects ArUco marker and computes the homography matrix (pixels -> mm).
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

        pts_px = corners[0].reshape(4, 2).astype(np.float32)
        pts_mm = np.array(
            [
                [0.0, 0.0],
                [self.marker_size_mm, 0.0],
                [self.marker_size_mm, self.marker_size_mm],
                [0.0, self.marker_size_mm],
            ],
            dtype=np.float32,
        )
        self.homography_matrix, _ = cv2.findHomography(pts_px, pts_mm)
        return True

    def updateHomography(self, frame_bgr: np.ndarray) -> bool:
        return self.update_homography(frame_bgr)


    def pixel_to_mm(self,px:float,py:float,) -> Optional[Tuple[float,float]]:
        """
        Transform a pixel x,y cord to ORL workspace mm
        """

        if self.homography_matrix is None:
            return None

        pt_px = np.array([px,py,1.0],dtype=np.float32)
        pt_mm = self.homography_matrix @ pt_px

        x_mm = pt_mm[0] / pt_mm[2]
        y_mm = pt_mm[1] / pt_mm[2]

        return round(float(x_mm),1), round(float(y_mm),1)


    def extract_rois(
        self,
        frame_bgr: np.ndarray,
        min_area: int = 15000,
        max_area: int = 70000,
        pad_frac: float = 0.25,
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
            pad = int(max(w, h) * pad_frac)
            y1 = max(0, y - pad)
            y2 = min(h_img, y + h + pad)
            x1 = max(0, x - pad)
            x2 = min(w_img, x + w + pad)

            # 6. Crop from the MASKED image (black background)
            roi = filtered_img[y1:y2, x1:x2]
            if roi.size > 0:
                rois.append(((x, y, w, h), roi))

        return rois

    def _calculate_symmetry_angle(self, shape: str, sin_v: float, cos_v: float) -> float:
        """Calculates optimal gripper angle in degrees considering rotational symmetry."""
        raw_angle = float(np.degrees(np.arctan2(sin_v, cos_v)))
        if shape == "Circle":
            return 0.0

        sym = self.SYMMETRY.get(shape, 360.0)
        delta = (-raw_angle) % sym
        if delta > sym / 2.0:
            delta -= sym
        return round(float(delta), 1)

    def infer_roi(
        self, bbox: Tuple[int, int, int, int], roi_bgr: np.ndarray, min_confidence: float = 0.85
    ) -> Optional[DetectedShape]:
        """Runs the 3-head ONNX model on a single cropped shape."""
        x, y, w, h = bbox

        # Preprocess: Resize (128x128), RGB, normalize, NCHW layout
        rgb = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(rgb, (128, 128)).astype(np.float32) / 255.0
        normalized = (resized - self.mean) / self.std
        tensor = np.expand_dims(np.transpose(normalized, (2, 0, 1)), axis=0)

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

        # Head 2: Orientation Angle
        sin_v, cos_v = float(outputs[1][0][0]), float(outputs[1][0][1])
        rz = self._calculate_symmetry_angle(shape_name, sin_v, cos_v)

        # Head 3: Center position in pixels
        cx_px = x + float(outputs[2][0][0]) * w
        cy_px = y + float(outputs[2][0][1]) * h

        # Translate coordinates to robot table mm (fallback to pixel coords if no ArUco marker)
        coords_mm = self.pixel_to_mm(cx_px, cy_px)
        if coords_mm is None:
            coords_mm = (round(float(cx_px), 1), round(float(cy_px), 1))

        return DetectedShape(
            shape=shape_name,
            x_mm=coords_mm[0],
            y_mm=coords_mm[1],
            rz_deg=rz,
            confidence=round(confidence, 2),
            cx_px=cx_px,
            cy_px=cy_px,
            bbox=bbox,
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

            # Draw center point
            cx, cy = int(obj.cx_px), int(obj.cy_px)
            cv2.circle(annotated, (cx, cy), 6, (0, 0, 255), -1)

            # Draw orientation arrow (gripper rotation angle)
            angle_rad = np.radians(obj.rz_deg)
            arrow_len = 45
            end_x = int(cx + arrow_len * np.cos(angle_rad))
            end_y = int(cy - arrow_len * np.sin(angle_rad))
            cv2.arrowedLine(annotated, (cx, cy), (end_x, end_y), (255, 50, 50), 3, tipLength=0.3)

            # High-contrast text label badge
            label = f"{obj.shape} ({int(obj.confidence*100)}%): X={obj.x_mm}, Y={obj.y_mm}, RZ={obj.rz_deg}\xc2\xb0"
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
    pipeline = VisionPipeline(model_path="shape_3head_model.onnx")
    print("VisionPipeline initialized successfully!")








