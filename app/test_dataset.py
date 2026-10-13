"""Test the ONNX model on the local dataset.

Walks ``dataset/dataset/labels.csv`` and runs the same preprocessing as the
robot (``vision.VisionPipeline.preprocess``) on every masked object crop, then
compares the model output with the dataset's own labels:

  * class    - predicted shape vs the ``shape`` column (accuracy + confusion).
  * rotation - predicted image-frame angle vs the rule-based ground truth
               (same rule the model was trained on, see neural_net/labels.py),
               folded by the shape's symmetry.
  * centre   - predicted centre vs the mask centroid, in crop pixels.

Usage (from the app/ folder):
    python test_dataset.py
    python test_dataset.py --model ../MISC/shape_3head_v2.onnx
    python test_dataset.py --save previews        # write annotated crops
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from vision import ROOT_DIR, VisionPipeline, wrap_period

# Candidate dataset roots (labels.csv + objects/... live here).
DATASET_CANDIDATES = [
    ROOT_DIR / "dataset" / "dataset",
    ROOT_DIR / "dataset",
]


# ── Rule-based ground truth (copied from ReWorks/neural_net/labels.py) ───────
def get_mask(img: np.ndarray):
    """Shape = the largest bright blob on the black background. Bool mask or None."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    mask = cv2.morphologyEx((gray > 25).astype(np.uint8), cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    n, blobs, stats, _ = cv2.connectedComponentsWithStats(mask)
    if n < 2:
        return None
    return blobs == 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])


def rule_angle(mask: np.ndarray, shape: str, symmetry: float) -> float:
    """The hard rule used to label the training data: direction to a tip
    (triangle / star) or straight out from a flat edge (hexagon / square),
    in image degrees (x right, y down)."""
    k = int(round(360 // symmetry))
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    outline = max(contours, key=cv2.contourArea)
    corners = cv2.approxPolyDP(outline, 0.02 * cv2.arcLength(outline, True), True).reshape(-1, 2).astype(float)

    if shape in ("Triangle", "Star"):
        m = cv2.moments(outline)
        d = corners - [m["m10"] / m["m00"], m["m01"] / m["m00"]]
        tips = d[np.argsort(np.hypot(d[:, 0], d[:, 1]))[-k:]]
        angles, weights, extra = np.arctan2(tips[:, 1], tips[:, 0]), np.ones(len(tips)), 0
    else:
        edges = np.roll(corners, -1, axis=0) - corners
        angles, weights, extra = np.arctan2(edges[:, 1], edges[:, 0]), np.hypot(edges[:, 0], edges[:, 1]), 90

    z = np.sum(weights * np.exp(1j * k * angles))
    return wrap_period(np.degrees(np.angle(z)) / k + extra, symmetry)


# ── Helpers ──────────────────────────────────────────────────────────────────
def find_dataset_root(explicit: str | None) -> Path:
    if explicit:
        path = Path(explicit)
        if (path / "labels.csv").exists():
            return path
        raise FileNotFoundError(f"No labels.csv in {path}")
    for path in DATASET_CANDIDATES:
        if (path / "labels.csv").exists():
            return path
    raise FileNotFoundError("No dataset found. Pass --dataset <folder with labels.csv>.")


def print_confusion(matrix: np.ndarray, classes) -> None:
    width = max(len(c) for c in classes) + 1
    print("\nConfusion matrix (rows = truth, cols = predicted):")
    print(" " * width + "".join(f"{c[:8]:>9}" for c in classes))
    for i, c in enumerate(classes):
        row = "".join(f"{matrix[i, j]:>9d}" for j in range(len(classes)))
        print(f"{c:>{width}}" + row)


# ── Main ─────────────────────────────────────────────────────────────────────
def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate the ONNX shape model on the dataset.")
    parser.add_argument("--dataset", default=None, help="folder containing labels.csv")
    parser.add_argument("--model", default=None, help="path to the ONNX model")
    parser.add_argument("--limit", type=int, default=0, help="only test the first N crops")
    parser.add_argument("--save", default=None, help="folder to write annotated preview crops")
    args = parser.parse_args()

    root = find_dataset_root(args.dataset)
    print(f"Dataset: {root}")

    pipeline = VisionPipeline(model_path=args.model)
    classes = pipeline.CLASSES
    class_index = {c: i for i, c in enumerate(classes)}

    df = pd.read_csv(root / "labels.csv")
    df = df[df["shape"].isin(classes)].reset_index(drop=True)
    if args.limit:
        df = df.head(args.limit)
    print(f"Crops labelled with a trained class: {len(df)}")

    save_dir = Path(args.save) if args.save else None
    if save_dir:
        save_dir.mkdir(parents=True, exist_ok=True)

    confusion = np.zeros((len(classes), len(classes)), dtype=int)
    correct = total = skipped = 0
    centre_errors: list[float] = []
    rot_errors: dict[str, list[float]] = defaultdict(list)
    rot_bias: dict[str, list[float]] = defaultdict(list)
    confidences: list[float] = []

    for _, row in df.iterrows():
        path = root / row["object_masked_path"]
        img = cv2.imread(str(path))
        if img is None:
            skipped += 1
            continue

        shape = row["shape"]
        tensor, (x0, y0, side) = pipeline.preprocess(img)
        pred = pipeline.predict(tensor)

        total += 1
        confidences.append(pred.confidence)
        if pred.shape == shape:
            correct += 1
        confusion[class_index[shape], class_index.get(pred.shape, 0)] += 1

        mask = get_mask(img)
        if mask is not None:
            ys, xs = np.nonzero(mask)
            # Centre error in crop pixels.
            cx_crop = pred.centre_norm[0] * side - x0
            cy_crop = pred.centre_norm[1] * side - y0
            centre_errors.append(float(np.hypot(cx_crop - (xs.mean() + 0.5), cy_crop - (ys.mean() + 0.5))))

            # Rotation error vs the rule-based ground truth (ignore circles).
            if shape in pipeline.SYMMETRY and shape != "Circle":
                gt = rule_angle(mask, shape, pipeline.SYMMETRY[shape])
                diff = wrap_period(pred.angle_img_deg - gt, pipeline.SYMMETRY[shape])
                rot_errors[shape].append(abs(diff))
                rot_bias[shape].append(diff)

        if save_dir:
            vis = img.copy()
            color = (0, 200, 0) if pred.shape == shape else (0, 0, 255)
            cx = int(pred.centre_norm[0] * side - x0)
            cy = int(pred.centre_norm[1] * side - y0)
            cv2.circle(vis, (cx, cy), 5, (0, 0, 255), -1)
            text = f"T:{shape} P:{pred.shape} {pred.confidence*100:.0f}%"
            cv2.putText(vis, text, (4, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
            cv2.imwrite(str(save_dir / f"{shape}_{Path(row['object_masked_path']).stem}.png"), vis)

    # ── Report ───────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print(f"CLASSIFICATION  {correct}/{total} = {correct / max(total,1) * 100:.1f}%")
    print(f"Confidence      mean {np.mean(confidences)*100:.1f}%  (min {np.min(confidences)*100:.1f}%)")
    if centre_errors:
        print(f"CENTRE          mean {np.mean(centre_errors):.1f} px  median {np.median(centre_errors):.1f} px in the crop")

    print("\nPer-class recall:")
    for c in classes:
        idx = class_index[c]
        n = confusion[idx].sum()
        print(f"  {c:<9} {confusion[idx, idx]:>4d}/{n:<4d}  {confusion[idx, idx] / n * 100 if n else 0:5.1f}%")

    print("\nROTATION error vs rule-based ground truth (image frame, folded):")
    for c in classes:
        if c == "Circle" or not rot_errors[c]:
            continue
        errs = np.array(rot_errors[c])
        bias = np.mean(rot_bias[c])
        print(f"  {c:<9} mean {errs.mean():5.2f}deg  median {np.median(errs):5.2f}deg  bias {bias:+5.2f}deg  (n={len(errs)})")

    print_confusion(confusion, classes)
    if skipped:
        print(f"\nSkipped {skipped} rows (image missing on disk).")
    if save_dir:
        print(f"\nAnnotated previews written to {save_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
