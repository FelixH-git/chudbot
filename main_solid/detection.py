# Author Nils Wikström niwi0007
"""DetectedShape: all information om en hittad bit. Bara data, ingen logik."""
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np


@dataclass
class DetectedShape:
    # Från modellen (pixlar i kamerabilden)
    shape: str
    confidence: float
    colour: str
    colour_confidence: float
    cx_px: float
    cy_px: float
    theta_img_deg: float                      # bitens vinkel i bilden
    bbox: Tuple[int, int, int, int]           # (x, y, w, h) i hela bilden
    contour_px: Optional[np.ndarray] = None   # konturen i hela bilden (för ghost)
    # Från kalibreringen (mm på bordet, robotens vridning)
    x_mm: float = 0.0
    y_mm: float = 0.0
    rz_deg: float = 0.0
    has_mm: bool = False                      # False = ingen ArUco-markör: X/Y är pixlar och RZ = 0, får inte skickas
