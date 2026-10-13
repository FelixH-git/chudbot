# Author Nils Wikström niwi0007
"""
Vision pipeline: hittar formerna i kamerabilden och räknar ut vad roboten ska göra.
- ArUco-markören kalibrerar bilden (pixlar -> mm). Origo = hörnet ORIGIN_CORNER på markören.
- ONNX-nätverk med 3 heads: class, rotation och centre.
- RZ = hur mycket roboten ska vrida biten så att den passar i sitt fack i trayen.
"""

import os
from dataclasses import dataclass
from typing import List, Optional, Tuple
import cv2
import numpy as np
import onnxruntime as ort


# ── Kalibrering (måste stämma med training) ──────────────────────────────────
# Vilket hörn på markören som är (0, 0). OpenCV-ordning: 0=TL, 1=TR, 2=BR, 3=BL (markörens egen frame).
# 2 = hörnet längst till vänster i kamerabilden. X pekar mot nästa hörn, Y mot det föregående.
ORIGIN_CORNER = 2

# Vinkeln varje bit ska ha i sitt fack, i robotens tray-frame
# (wobjAiPlaceTray, grader, moturs sett ovanifrån). Uppmätt från GT_angle-fotona.
SLOT_ANGLE_ROBOT_DEG = {"Circle": 0.0, "Hexagon": -5.4, "Square": -1.7, "Star": 32.2, "Triangle": 58.4}

# Markörens frame (Y "nedåt" som i bilden) är spegelvänd mot robotens frame (Z uppåt),
# så vinklar byter tecken markör -> robot. Bekräftat med GT-fotona.
MARKER_Y_FLIPPED = True
# Extra vridning mellan markörens X-axel och robotens X-axel (0 = wobjAruco X längs markörens X).
MARKER_TO_ROBOT_ROT_DEG = 0.0

# Var X/Y kommer ifrån: "ai" = head 3 i nätverket, "cv2" = mitten av formens mask
CENTER_SOURCE = "ai"

IMG_SIZE = 128
CANVAS_MARGIN = 0.15   # svart kant runt cropen innan resize (samma som i training)

# Modellen som tränas och exporteras av neural_net/train.py
MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "neural_net", "models", "crpnet.onnx")


def wrap_period(deg: float, period: float) -> float:
    """Viker en vinkel till intervallet [-period/2, period/2). Ex: kvadrat (90), 100° -> 10°."""
    return (deg + period / 2.0) % period - period / 2.0


@dataclass
class DetectedShape:
    """All information om en hittad form, redo att skickas till ABB-roboten."""
    shape: str
    x_mm: float
    y_mm: float
    rz_deg: float
    confidence: float
    cx_px: float = 0.0
    cy_px: float = 0.0
    bbox: Tuple[int, int, int, int] = (0, 0, 0, 0)
    theta_img_deg: float = 0.0        # bitens vinkel i bilden (för att rita)
    contour_px: Optional[np.ndarray] = None   # konturen i pixlar i hela bilden (för ghost)

    def to_robot_command(self) -> str:
        """Texten som RAPID-koden på roboten läser från socketen."""
        return f"SHAPE={self.shape};X={self.x_mm:.1f};Y={self.y_mm:.1f};RZ={self.rz_deg:.1f};END\r\n"

    def toRobortCommand(self) -> str:
        return self.to_robot_command()


class VisionPipeline:
    CLASSES = ["Circle", "Hexagon", "Square", "Star", "Triangle"]
    SYMMETRY = {
        "Circle": 360.0,
        "Hexagon": 60.0,
        "Triangle": 120.0,
        "Square": 90.0,
        "Star": 72.0,
    }

    def __init__(
        self,
        model_path: str = MODEL_PATH,
        marker_size_mm: float = 100.0,
        target_shapes: Optional[List[str]] = None,
        pad_frac: float = 0.25,
    ):
        self.marker_size_mm = marker_size_mm
        self.target_shapes = target_shapes or ["Hexagon", "Triangle", "Circle", "Square", "Star"]
        self.homography_matrix: Optional[np.ndarray] = None
        self.pad_frac = pad_frac

        # Normaliseringen som användes i training (ImageNet)
        self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        self.std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

        # ArUco-detektering (DICT_5X5_100) med sub-pixel-hörn
        self.aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_100)
        self.aruco_params = cv2.aruco.DetectorParameters()
        self.aruco_params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
        if hasattr(cv2.aruco, "ArucoDetector"):
            self.aruco_detector = cv2.aruco.ArucoDetector(self.aruco_dict, self.aruco_params)
        else:
            self.aruco_detector = None

        # Ladda ONNX-modellen
        session_opts = ort.SessionOptions()
        session_opts.intra_op_num_threads = 4
        session_opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        session_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        self.session = ort.InferenceSession(
            model_path, sess_options=session_opts, providers=["CPUExecutionProvider"]
        )
        self.input_name = self.session.get_inputs()[0].name

        # Rotation-outputen ska vara ett [sin kθ, cos kθ]-par per class -> formen (1, 5, 2)
        angle_shape = self.session.get_outputs()[1].shape
        if len(angle_shape) != 3:
            raise ValueError(
                f"{model_path} is an old (v1) model. Export crpnet.onnx with neural_net/train.py."
            )

    def update_homography(self, frame_bgr: np.ndarray) -> bool:
        """
        Hittar ArUco-markören och räknar ut homografin (pixlar -> mm).
        Origo (0, 0) är markörens hörn ORIGIN_CORNER.
        Returnerar True om markören hittades.
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

    def pixel_to_mm(self, px: float, py: float,) -> Optional[Tuple[float, float]]:
        """
        Pixel (x, y) -> mm på bordet, mätt från markörens origo-hörn.
        """

        if self.homography_matrix is None:
            return None

        pt = np.array([[[px, py]]], dtype=np.float64)
        x_mm, y_mm = cv2.perspectiveTransform(pt, self.homography_matrix)[0, 0]

        return round(float(x_mm), 1), round(float(y_mm), 1)

    def image_angle_to_marker_deg(self, cx: float, cy: float, theta_img_deg: float) -> float:
        """En riktning i bilden vid (cx, cy) -> vinkel i markörens frame (via homografin)."""
        r = np.radians(theta_img_deg)
        pts = np.array([[[cx, cy]], [[cx + 40.0 * np.cos(r), cy + 40.0 * np.sin(r)]]], dtype=np.float64)
        a, b = cv2.perspectiveTransform(pts, self.homography_matrix).reshape(2, 2)
        return float(np.degrees(np.arctan2(b[1] - a[1], b[0] - a[0])))

    def robot_rz(self, shape: str, marker_angle_deg: float) -> float:
        """
        Hur mycket roboten ska vrida (grader, moturs sett ovanifrån, runt robotens Z-axel):
        fackets vinkel - bitens vinkel, båda i robotens frame. Kortaste vridningen tack vare symmetrin.
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
        Gör bakgrunden svart med en HSV-färgmask (som i datasetet), tar bort saker vid kanterna
        (balkar / robotarm) och klipper ut en crop per form med 25 % padding.
        """
        h_img, w_img = frame_bgr.shape[:2]

        # 1. HSV-färgfilter: behåller färgade former, bord och skuggor blir svarta
        hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
        lower_color = np.array([0, 50, 50])
        upper_color = np.array([179, 255, 255])
        mask = cv2.inRange(hsv, lower_color, upper_color)

        # 2. Ta bort brus
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        # 3. Gör bakgrunden helt svart (EXAKT som bilderna i datasetet)
        filtered_img = cv2.bitwise_and(frame_bgr, frame_bgr, mask=mask)

        # 4. Hitta formernas konturer i masken
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        rois = []

        for cnt in contours:
            area = cv2.contourArea(cnt)
            # Filter 1: storlek (släng brus < 15 000 px och stora blobs i bakgrunden > 70 000 px)
            if area < min_area or area > max_area:
                continue

            x, y, w, h = cv2.boundingRect(cnt)

            # Filter 2: kanterna (ignorera balkarna upptill och robotarmen nertill)
            if y < int(h_img * 0.05) or (y + h) > int(h_img * 0.85):
                continue

            # Filter 3: proportioner (riktiga former är ungefär 1:1, släng långa streck / kablar)
            aspect = w / float(h) if h > 0 else 0
            if aspect < 0.45 or aspect > 2.2:
                continue

            # 5. Padding: 25 % av längsta sidan (samma som när datasetet togs)
            x1, y1 = self._crop_origin((x, y, w, h))
            pad = int(max(w, h) * self.pad_frac)
            y2 = min(h_img, y + h + pad)
            x2 = min(w_img, x + w + pad)

            # 6. Croppa från den MASKADE bilden (svart bakgrund)
            roi = filtered_img[y1:y2, x1:x2]
            if roi.size > 0:
                rois.append(((x, y, w, h), roi))

        return rois

    def _crop_origin(self, bbox: Tuple[int, int, int, int]) -> Tuple[int, int]:
        """Övre vänstra hörnet av cropen (med padding) i hela bilden."""
        x, y, w, h = bbox
        pad = int(max(w, h) * self.pad_frac)
        return max(0, x - pad), max(0, y - pad)

    def _preprocess(self, roi_bgr: np.ndarray):
        """Lägger cropen mitt i en svart kvadrat (så den inte trycks ihop), resize till 128, normaliserar."""
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
        """Kör nätverket (3 heads) på en crop och gör om svaret till en DetectedShape."""
        x, y, w, h = bbox

        tensor, (x0, y0, side) = self._preprocess(roi_bgr)

        # Kör modellen
        outputs = self.session.run(None, {self.input_name: tensor})

        # Head 1: class
        logits = outputs[0][0]
        class_idx = int(np.argmax(logits))
        shape_name = self.CLASSES[class_idx]

        if shape_name not in self.target_shapes:
            return None

        # Confidence: softmax gör om poängen till sannolikheter
        exp_logits = np.exp(logits - np.max(logits))
        confidence = float(exp_logits[class_idx] / np.sum(exp_logits))

        print(f"  Candidate at ({x},{y}): {shape_name} | Conf: {confidence*100:.1f}% | Area: {w*h}px")

        # Filter 4: släng gissningar med låg confidence
        if confidence < min_confidence:
            return None

        # Head 3: centre, 0..1 på kvadraten -> pixlar i hela bilden
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

        # Head 2: vinkel i bilden, från raden för den gissade classen: θ = atan2(s, c) / k
        sym = self.SYMMETRY[shape_name]
        theta_img = 0.0
        if shape_name != "Circle":
            k = 360.0 / sym
            sin_v, cos_v = (float(v) for v in outputs[1][0][class_idx])
            theta_img = float(np.degrees(np.arctan2(sin_v, cos_v)) / k)

        # Till mm på bordet + RZ (om ingen ArUco-markör finns: pixlar och RZ = 0)
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
        En hel frame: kalibrera (första gången), hitta former, kör nätverket, räkna om till mm.
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
        Bitens kontur efter att roboten vridit den med RZ, i pixlar i hela bilden (ghost).
        Roboten vrider +RZ (moturs ovanifrån) = -RZ i markörens spegelvända frame, så konturen
        vrids -RZ i mm runt centre och räknas sedan tillbaka till pixlar.
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
        Ritar bounding box, ghost, centre, vinkelpil och en textruta (form, X, Y, RZ) för varje form.
        """
        annotated = frame_bgr.copy()

        for obj in detections:
            x, y, w, h = obj.bbox

            # Bounding box
            cv2.rectangle(annotated, (x, y), (x + w, y + h), (0, 255, 0), 2)

            # Ghost: biten efter att roboten vridit den med RZ (magenta)
            ghost = self.ghost_outline(obj)
            if ghost is not None:
                cv2.polylines(annotated, [ghost], True, (255, 0, 255), 3, cv2.LINE_AA)

            # Centre
            cx, cy = int(obj.cx_px), int(obj.cy_px)
            cv2.circle(annotated, (cx, cy), 6, (0, 0, 255), -1)

            # Vinkelpil (bitens vinkel i bilden)
            if obj.shape != "Circle":
                angle_rad = np.radians(obj.theta_img_deg)
                arrow_len = 45
                end_x = int(cx + arrow_len * np.cos(angle_rad))
                end_y = int(cy + arrow_len * np.sin(angle_rad))
                cv2.arrowedLine(annotated, (cx, cy), (end_x, end_y), (255, 50, 50), 3, tipLength=0.3)

            # Textruta med svart bakgrund
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
    pipeline = VisionPipeline()
    print("VisionPipeline initialized successfully!")
