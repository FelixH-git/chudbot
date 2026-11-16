# Author Nils Wikström niwi0007
"""ShapeModel: kör CRPNet (ONNX) på en crop. Klasser, färger och bildstorlek läses från modellens json-fil."""
import json
from dataclasses import dataclass
from typing import Dict, List, Tuple

import cv2
import numpy as np
import onnxruntime as ort


@dataclass
class ModelInfo:
    """Det som hör ihop med en tränad modell (json bredvid .onnx), så robot-programmet inte har egna kopior."""
    classes: List[str]
    colours: List[str]
    symmetry_deg: Dict[str, float]     # former som har en rotation (cirkeln saknas)
    img_size: int
    canvas_margin: float
    norm_mean: List[float]
    norm_std: List[float]
    outputs: List[str]

    @classmethod
    def load(cls, path: str) -> "ModelInfo":
        with open(path) as f:
            return cls(**json.load(f))


@dataclass
class Prediction:
    shape: str
    confidence: float
    colour: str
    colour_confidence: float
    centre: Tuple[float, float]        # (cx, cy) i cropens pixlar
    theta_img_deg: float               # vinkel i bilden (0 för cirklar)


def softmax(logits: np.ndarray) -> np.ndarray:
    e = np.exp(logits - np.max(logits))
    return e / e.sum()


class ShapeModel:
    def __init__(self, model_path: str, info: ModelInfo):
        self.info = info
        self.mean = np.array(info.norm_mean, dtype=np.float32)
        self.std = np.array(info.norm_std, dtype=np.float32)

        opts = ort.SessionOptions()
        opts.intra_op_num_threads = 4
        opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.session = ort.InferenceSession(model_path, sess_options=opts, providers=["CPUExecutionProvider"])
        self.input_name = self.session.get_inputs()[0].name
        if [o.name for o in self.session.get_outputs()] != info.outputs:
            raise ValueError(f"{model_path} does not have the outputs {info.outputs}. Export it with neural_net/train.py.")

    def preprocess(self, roi_bgr: np.ndarray):
        """Cropen mitt i en svart kvadrat (så den inte trycks ihop), resize, normalisera. Samma som i training."""
        h, w = roi_bgr.shape[:2]
        side = int(round(max(h, w) * (1.0 + 2.0 * self.info.canvas_margin)))
        canvas = np.zeros((side, side, 3), dtype=roi_bgr.dtype)
        x0, y0 = (side - w) // 2, (side - h) // 2
        canvas[y0:y0 + h, x0:x0 + w] = roi_bgr

        small = cv2.resize(canvas, (self.info.img_size, self.info.img_size), interpolation=cv2.INTER_AREA)
        rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        tensor = np.transpose((rgb - self.mean) / self.std, (2, 0, 1))[None].astype(np.float32)
        return tensor, (x0, y0, side)

    def predict(self, roi_bgr: np.ndarray) -> Prediction:
        tensor, (x0, y0, side) = self.preprocess(roi_bgr)
        class_logits, rotation, centre, colour_logits = self.session.run(None, {self.input_name: tensor})

        # Head 1: class (softmax -> sannolikheter)
        p = softmax(class_logits[0])
        idx = int(np.argmax(p))
        shape = self.info.classes[idx]

        # Head 2: vinkel i bilden, från raden för den gissade classen: θ = atan2(s, c) / k
        theta = 0.0
        if shape in self.info.symmetry_deg:
            k = 360.0 / self.info.symmetry_deg[shape]
            sin_v, cos_v = (float(v) for v in rotation[0][idx])     # float64, annars blir atan2 float32
            theta = float(np.degrees(np.arctan2(sin_v, cos_v)) / k)

        # Head 3: centre, 0..1 på kvadraten -> cropens pixlar
        cx, cy = (float(v) * side for v in centre[0])

        # Head 4: colour
        pc = softmax(colour_logits[0])
        return Prediction(shape, float(p[idx]), self.info.colours[int(np.argmax(pc))], float(np.max(pc)),
                          (cx - x0, cy - y0), theta)
