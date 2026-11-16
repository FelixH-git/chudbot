# Author Nils Wikström niwi0007
"""PieceSegmenter: hittar de färgade bitarna i kamerabilden och klipper ut en crop per bit (ingen AI)."""
from dataclasses import dataclass
from typing import List, Optional, Tuple

import cv2
import numpy as np

from settings import ASPECT_RANGE, BOTTOM_LIMIT, HSV_LOWER, HSV_UPPER, MAX_AREA, MIN_AREA, PAD_FRAC, TOP_LIMIT


@dataclass
class Crop:
    bbox: Tuple[int, int, int, int]           # bitens bounding box (x, y, w, h) i hela bilden
    origin: Tuple[int, int]                   # var cropens övre vänstra hörn ligger i hela bilden
    image: np.ndarray                         # cropen, svart bakgrund (som bilderna i datasetet)
    contour_px: Optional[np.ndarray]          # bitens kontur i hela bilden


class PieceSegmenter:
    def __init__(self, min_area: int = MIN_AREA, max_area: int = MAX_AREA, pad_frac: float = PAD_FRAC):
        # min_area och max_area beror på kamerans höjd; GUI:t ändrar dem live
        self.min_area = min_area
        self.max_area = max_area
        self.pad_frac = pad_frac

    def find(self, frame_bgr: np.ndarray) -> List[Crop]:
        h_img, w_img = frame_bgr.shape[:2]

        # 1. HSV-färgfilter: behåller färgade former, bord och skuggor blir svarta
        hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, np.array(HSV_LOWER), np.array(HSV_UPPER))
        # 2. Ta bort brus
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        # 3. Gör bakgrunden helt svart (EXAKT som bilderna i datasetet)
        filtered = cv2.bitwise_and(frame_bgr, frame_bgr, mask=mask)

        crops = []
        for cnt in cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]:
            area = cv2.contourArea(cnt)
            if area < self.min_area or area > self.max_area:              # storlek
                continue
            x, y, w, h = cv2.boundingRect(cnt)
            if y < int(h_img * TOP_LIMIT) or (y + h) > int(h_img * BOTTOM_LIMIT):   # kanterna
                continue
            aspect = w / float(h) if h > 0 else 0
            if aspect < ASPECT_RANGE[0] or aspect > ASPECT_RANGE[1]:     # långa streck / kablar
                continue

            pad = int(max(w, h) * self.pad_frac)
            x1, y1 = max(0, x - pad), max(0, y - pad)
            roi = filtered[y1:min(h_img, y + h + pad), x1:min(w_img, x + w + pad)]
            if roi.size > 0:
                crops.append(Crop((x, y, w, h), (x1, y1), roi, self._outline(roi, (x1, y1))))
        return crops

    @staticmethod
    def _outline(roi_bgr: np.ndarray, origin: Tuple[int, int]) -> Optional[np.ndarray]:
        """Konturen av den största blobben i cropen, flyttad till hela bildens pixlar."""
        roi_mask = (cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY) > 0).astype(np.uint8)
        contours, _ = cv2.findContours(roi_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None
        return max(contours, key=cv2.contourArea).reshape(-1, 2) + list(origin)
