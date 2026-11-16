# Author Nils Wikström niwi0007
"""
VisionPipeline: kopplar ihop delarna. Ingen egen bildbehandling, bara ordningen:
  detect(): PieceSegmenter (crops) -> ShapeModel (class, colour, centre, vinkel) -> filter   [pixlar]
  locate(): MarkerCalibration (mm) -> RobotFrame (RZ)                                         [robotens värld]
"""
import time
from typing import List

import cv2

from calibration import MarkerCalibration
from detection import DetectedShape
from robot_operations import RobotFrame
from segmenter import PieceSegmenter
from settings import LOG_INTERVAL_S, MIN_CIRCLE_AREA, MIN_CONFIDENCE
from shape_model import ShapeModel


class CandidateLog:
    """Skriver ut alla candidates (även de som filtreras bort) högst en gång per interval_s."""

    def __init__(self, interval_s: float = LOG_INTERVAL_S):
        self.interval_s = interval_s
        self._last = 0.0
        self.active = True

    def begin_frame(self) -> None:
        now = time.monotonic()
        self.active = now - self._last >= self.interval_s
        if self.active:
            self._last = now

    def next_frame(self) -> None:
        """Skriv ut nästa frame oavsett intervallet (t.ex. direkt efter ett val)."""
        self._last = float("-inf")

    def candidate(self, bbox, shape: str, confidence: float) -> None:
        if self.active:
            x, y, w, h = bbox
            print(f"  Candidate at ({x},{y}): {shape} | Conf: {confidence*100:.1f}% | Area: {w*h}px")


class VisionPipeline:
    def __init__(self, segmenter: PieceSegmenter, model: ShapeModel, calibration: MarkerCalibration,
                 robot_frame: RobotFrame, log: CandidateLog, min_confidence: float = MIN_CONFIDENCE,
                 min_circle_area: int = MIN_CIRCLE_AREA):
        self.segmenter = segmenter
        self.model = model
        self.calibration = calibration
        self.robot_frame = robot_frame
        self.log = log
        self.min_confidence = min_confidence
        self.min_circle_area = min_circle_area     # magnet-filtret, beror på kamerans höjd (slider i GUI:t)

    def detect(self, frame_bgr) -> List[DetectedShape]:
        """Alla bitar i bilden: class, colour, centre och vinkel, i pixlar."""
        self.log.begin_frame()
        shapes = []
        for crop in self.segmenter.find(frame_bgr):
            pred = self.model.predict(crop.image)
            self.log.candidate(crop.bbox, pred.shape, pred.confidence)
            if pred.confidence < self.min_confidence:                     # osäker gissning
                continue
            if pred.shape == "Circle" and (crop.contour_px is None or
                                           cv2.contourArea(crop.contour_px.astype("int32")) < self.min_circle_area):
                continue                                                   # för liten "Circle" = magnet
            cx, cy = pred.centre
            shapes.append(DetectedShape(
                shape=pred.shape, confidence=round(pred.confidence, 2),
                colour=pred.colour, colour_confidence=round(pred.colour_confidence, 2),
                cx_px=crop.origin[0] + cx, cy_px=crop.origin[1] + cy, theta_img_deg=pred.theta_img_deg,
                bbox=crop.bbox, contour_px=crop.contour_px,
            ))
        return shapes

    def locate(self, frame_bgr, shapes: List[DetectedShape]) -> List[DetectedShape]:
        """Fyller i mm och RZ. Utan markör: X/Y blir pixlar, RZ = 0 och has_mm = False (skickas aldrig)."""
        if not self.calibration.found:
            self.calibration.update(frame_bgr)
        for s in shapes:
            mm = self.calibration.pixel_to_mm(s.cx_px, s.cy_px)
            if mm is None:
                s.x_mm, s.y_mm = round(float(s.cx_px), 1), round(float(s.cy_px), 1)
                continue
            s.x_mm, s.y_mm = mm
            s.rz_deg = self.robot_frame.rz(s.shape, self.calibration.image_angle_to_marker_deg(
                s.cx_px, s.cy_px, s.theta_img_deg))
            s.has_mm = True
        return shapes

    def process_frame(self, frame_bgr) -> List[DetectedShape]:
        """Hela kedjan på en bild (för tester; i programmet gör thread 1 detect och thread 2 locate)."""
        return self.locate(frame_bgr, self.detect(frame_bgr))
