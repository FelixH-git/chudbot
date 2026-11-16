# Author Nils Wikström niwi0007
"""HudRenderer: ritar på kamerabilden (bbox, ghost, centre, vinkelpil, etikett, skala, könummer). Räknar inget själv."""
from typing import Iterable, List, Optional

import cv2
import numpy as np

from calibration import MarkerCalibration
from detection import DetectedShape
from settings import MARKER_Y_FLIPPED

YELLOW = (0, 215, 255)


class HudRenderer:
    def __init__(self, calibration: MarkerCalibration, rotating_shapes: Iterable[str],
                 y_flipped: bool = MARKER_Y_FLIPPED):
        self.calibration = calibration
        self.rotating_shapes = set(rotating_shapes)      # former som får en vinkelpil (inte cirkeln)
        self.y_flipped = y_flipped

    def ghost_outline(self, obj: DetectedShape) -> Optional[np.ndarray]:
        """
        Bitens kontur efter att roboten vridit den med RZ, i pixlar i hela bilden (ghost).
        Roboten vrider +RZ (moturs ovanifrån) = -RZ i markörens spegelvända frame, så konturen
        vrids -RZ i mm runt centre och räknas sedan tillbaka till pixlar.
        """
        if obj.contour_px is None or not self.calibration.found:
            return None
        pts_mm = self.calibration.to_mm(obj.contour_px)
        c_mm = self.calibration.to_mm([[obj.cx_px, obj.cy_px]])[0]
        r = np.radians(-obj.rz_deg if self.y_flipped else obj.rz_deg)
        R = np.array([[np.cos(r), -np.sin(r)], [np.sin(r), np.cos(r)]])
        ghost_mm = (pts_mm - c_mm) @ R.T + c_mm
        return self.calibration.to_px(ghost_mm).astype(np.int32)

    def draw_scale_bar(self, img: np.ndarray, length_mm: float = 10.0) -> None:
        """Ritar ett |---| nere till vänster som är length_mm lång i pixlar (mätt vid bildens mitt)."""
        if not self.calibration.found:
            return
        h_img, w_img = img.shape[:2]

        # Bildens mitt -> mm, gå length_mm i X, och tillbaka till pixlar
        x_mm, y_mm = self.calibration.to_mm([[w_img / 2, h_img / 2]])[0]
        end = self.calibration.to_px([[x_mm + length_mm, y_mm]])[0]
        length_px = float(np.hypot(end[0] - w_img / 2, end[1] - h_img / 2))

        label = f"{length_mm:g} mm = {length_px:.0f} px"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
        x0, y = 50, h_img - 50
        x1 = x0 + int(round(length_px))

        cv2.rectangle(img, (x0 - 20, y - th - 25), (x1 + tw + 40, y + 20), (0, 0, 0), -1)
        cv2.line(img, (x0, y), (x1, y), (255, 255, 255), 3)             # själva linjen
        cv2.line(img, (x0, y - 10), (x0, y + 10), (255, 255, 255), 3)   # ändstreck
        cv2.line(img, (x1, y - 10), (x1, y + 10), (255, 255, 255), 3)
        cv2.putText(img, label, (x1 + 20, y + th // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)

    def draw(self, frame_bgr: np.ndarray, shapes: List[DetectedShape]) -> np.ndarray:
        """En kopia av bilden med bounding box, ghost, centre, vinkelpil och en textruta för varje bit."""
        annotated = frame_bgr.copy()
        for obj in shapes:
            x, y, w, h = obj.bbox
            cv2.rectangle(annotated, (x, y), (x + w, y + h), (0, 255, 0), 2)

            # Ghost: biten efter att roboten vridit den med RZ (blå)
            ghost = self.ghost_outline(obj)
            if ghost is not None:
                cv2.polylines(annotated, [ghost], True, (255, 0, 0), 3, cv2.LINE_AA)

            # Centre
            cx, cy = int(obj.cx_px), int(obj.cy_px)
            cv2.circle(annotated, (cx, cy), 6, (0, 0, 255), -1)

            # Vinkelpil (bitens vinkel i bilden)
            if obj.shape in self.rotating_shapes:
                angle_rad = np.radians(obj.theta_img_deg)
                end = (int(cx + 45 * np.cos(angle_rad)), int(cy + 45 * np.sin(angle_rad)))
                cv2.arrowedLine(annotated, (cx, cy), end, (255, 50, 50), 3, tipLength=0.3)

            # Textruta med svart bakgrund
            label = f"{obj.colour} {obj.shape} ({int(obj.confidence*100)}%): X={obj.x_mm}, Y={obj.y_mm}, RZ={obj.rz_deg} deg"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            badge_y1 = max(0, y - th - 12)
            badge_y2 = max(th + 12, y)
            cv2.rectangle(annotated, (x, badge_y1), (x + tw + 10, badge_y2), (0, 0, 0), -1)
            cv2.putText(annotated, label, (x + 5, badge_y2 - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2,
                        cv2.LINE_AA)

        self.draw_scale_bar(annotated)
        return annotated

    @staticmethod
    def draw_queue(img: np.ndarray, queue: List[DetectedShape]) -> None:
        """Könummer (#1, #2, ...) på varje sparad bit. Nästa bit (#1) är gul."""
        for i, s in enumerate(queue):
            p = (int(s.cx_px) - 30, int(s.cy_px) + 70)
            cv2.putText(img, f"#{i + 1}", p, cv2.FONT_HERSHEY_SIMPLEX, 2.0, (0, 0, 0), 10, cv2.LINE_AA)
            cv2.putText(img, f"#{i + 1}", p, cv2.FONT_HERSHEY_SIMPLEX, 2.0,
                        YELLOW if i == 0 else (255, 255, 255), 4, cv2.LINE_AA)

    @staticmethod
    def draw_sending(img: np.ndarray, piece: DetectedShape) -> None:
        """Gul ruta runt biten som just skickades till roboten."""
        bx, by, bw, bh = piece.bbox
        cv2.rectangle(img, (bx, by), (bx + bw, by + bh), YELLOW, 6)
        cv2.putText(img, f">> SENDING {piece.shape.upper()} TO ROBOT <<", (bx, max(40, by - 25)),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, YELLOW, 3, cv2.LINE_AA)
