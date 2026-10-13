"""Gör om en crop av en bit till nätverkets input. Används både vid training och på roboten."""
import cv2
import numpy as np

from .config import CANVAS_MARGIN, IMG_SIZE, NORM_MEAN, NORM_STD


def square_canvas(crop_bgr):
    """
    Lägger cropen i mitten av en svart kvadrat med marginal runt.
    Returnerar (canvas, (x0, y0, side)): var cropens övre vänstra hörn hamnar, och kvadratens storlek.
    """
    h, w = crop_bgr.shape[:2]
    side = int(round(max(h, w) * (1 + 2 * CANVAS_MARGIN)))   # längsta sidan + 15 % marginal på varje sida
    canvas = np.zeros((side, side, 3), dtype=np.uint8)       # helt svart kvadrat

    # Centrera cropen i kvadraten (rader = y först, sedan kolumner = x)
    x0, y0 = (side - w) // 2, (side - h) // 2
    canvas[y0:y0 + h, x0:x0 + w] = crop_bgr
    return canvas, (x0, y0, side)


def to_input(canvas):
    """Kvadraten -> float32-array med formen (3, 128, 128), färger normaliserade som ImageNet."""
    small = cv2.resize(canvas, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)   # krymp till 128 x 128
    rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0         # BGR -> RGB, 0..255 -> 0..1
    normalized = (rgb - np.array(NORM_MEAN, np.float32)) / np.array(NORM_STD, np.float32)
    return normalized.transpose(2, 0, 1)                                            # (höjd, bredd, färg) -> (färg, höjd, bredd)
