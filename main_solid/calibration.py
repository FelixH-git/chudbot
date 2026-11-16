# Author Nils Wikström niwi0007
"""MarkerCalibration: hittar ArUco-markören och räknar om mellan pixlar och mm på bordet (homografi)."""
from typing import Optional, Tuple

import cv2
import numpy as np

from settings import MARKER_SIZE_MM, ORIGIN_CORNER


class MarkerCalibration:
    def __init__(self, marker_size_mm: float = MARKER_SIZE_MM, origin_corner: int = ORIGIN_CORNER):
        self.marker_size_mm = marker_size_mm
        self.origin_corner = origin_corner
        self.homography: Optional[np.ndarray] = None
        # ArUco-detektering (DICT_5X5_100) med sub-pixel-hörn
        params = cv2.aruco.DetectorParameters()
        params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
        self.detector = cv2.aruco.ArucoDetector(cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_100), params)

    @property
    def found(self) -> bool:
        return self.homography is not None

    def update(self, frame_bgr: np.ndarray) -> bool:
        """Hittar markören och räknar ut homografin (pixlar -> mm), origo i hörnet origin_corner. False = ingen markör."""
        corners, ids, _ = self.detector.detectMarkers(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY))
        if ids is None or len(ids) == 0:
            return False
        pts_px = np.roll(corners[0].reshape(4, 2), -self.origin_corner, axis=0).astype(np.float64)
        s = self.marker_size_mm
        pts_mm = np.array([[0, 0], [s, 0], [s, s], [0, s]], dtype=np.float64)
        self.homography, _ = cv2.findHomography(pts_px, pts_mm)
        return True

    @staticmethod
    def _transform(points, h: np.ndarray) -> np.ndarray:
        return cv2.perspectiveTransform(np.asarray(points, dtype=np.float64).reshape(-1, 1, 2), h).reshape(-1, 2)

    def to_mm(self, points_px) -> np.ndarray:
        """Punkter (N, 2) i pixlar -> mm på bordet."""
        return self._transform(points_px, self.homography)

    def to_px(self, points_mm) -> np.ndarray:
        """Punkter (N, 2) i mm -> pixlar."""
        return self._transform(points_mm, np.linalg.inv(self.homography))

    def pixel_to_mm(self, px: float, py: float) -> Optional[Tuple[float, float]]:
        """En pixel -> mm (avrundat till 0.1 mm). None om markören inte hittats."""
        if not self.found:
            return None
        x_mm, y_mm = self.to_mm([[px, py]])[0]
        return round(float(x_mm), 1), round(float(y_mm), 1)

    def image_angle_to_marker_deg(self, cx: float, cy: float, theta_img_deg: float) -> float:
        """En riktning i bilden vid (cx, cy) -> vinkel i markörens frame (en 40 px lång pil räknas om till mm)."""
        r = np.radians(theta_img_deg)
        a, b = self.to_mm([[cx, cy], [cx + 40.0 * np.cos(r), cy + 40.0 * np.sin(r)]])
        return float(np.degrees(np.arctan2(b[1] - a[1], b[0] - a[0])))
