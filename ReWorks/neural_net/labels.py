"""Labels för training: class, centre och rotation för varje crop, utan de crops som skulle lära ut nonsens."""
import cv2
import numpy as np
import pandas as pd

from .config import CLASSES, SYMMETRY_DEG, dataset_root


def wrap(angle, symmetry_deg):
    """Viker en vinkel till intervallet [-symmetry_deg/2, symmetry_deg/2). Ex: kvadrat, 100° -> 10°."""
    return (angle + symmetry_deg / 2) % symmetry_deg - symmetry_deg / 2


def get_mask(img):
    """Formen = den största ljusa blobben på den svarta bakgrunden. True/False per pixel, eller None."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    mask = cv2.morphologyEx((gray > 25).astype(np.uint8), cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))  # ta bort brus
    n, blobs, stats, _ = cv2.connectedComponentsWithStats(mask)   # dela upp i separata blobs (0 = bakgrund)
    if n < 2:
        return None
    return blobs == 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])     # behåll den största blobben


def rule_angle(mask, shape):
    """
    Den hårda regeln. Triangel / stjärna: riktningen mot en spets. Hexagon / kvadrat: rakt ut från en platt kant.
    Grader i bilden (x höger, y nedåt), alltså -90 = uppåt. Returnerar också hur bra formen passar (0..1).
    """
    k = 360 // SYMMETRY_DEG[shape]                                   # antal lika riktningar (t.ex. 6 för hexagon)
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    outline = max(contours, key=cv2.contourArea)
    # Förenkla konturen till raka linjer
    corners = cv2.approxPolyDP(outline, 0.02 * cv2.arcLength(outline, True), True).reshape(-1, 2).astype(float)

    if shape in ("Triangle", "Star"):
        # Spetsarna = de k hörn som ligger längst från mitten
        m = cv2.moments(outline)
        d = corners - [m["m10"] / m["m00"], m["m01"] / m["m00"]]
        tips = d[np.argsort(np.hypot(d[:, 0], d[:, 1]))[-k:]]
        angles, weights, extra = np.arctan2(tips[:, 1], tips[:, 0]), np.ones(len(tips)), 0
    else:
        # Kanterna = från varje hörn till nästa, långa kanter väger mer. +90° = rakt ut från kanten
        edges = np.roll(corners, -1, axis=0) - corners
        angles, weights, extra = np.arctan2(edges[:, 1], edges[:, 0]), np.hypot(edges[:, 0], edges[:, 1]), 90

    # Medelvärde på k-cirkeln: vinkel * k gör alla spetsar/kanter till samma riktning, sedan / k tillbaka
    z = np.sum(weights * np.exp(1j * k * angles))
    return wrap(np.degrees(np.angle(z)) / k + extra, SYMMETRY_DEG[shape]), abs(z) / weights.sum()


def load_crops():
    """Alla användbara crops med labels: class, centre (pixlar i cropen) och rotation-vinkel."""
    root = dataset_root()
    rows = []
    for _, r in pd.read_csv(root / "labels.csv").iterrows():
        shape = r["shape"]
        if shape not in CLASSES:                 # hoppa över extra former (Rectangle, Polygon ...)
            continue
        img = cv2.imread(str(root / r["object_masked_path"]))
        if img is None:                          # bilden saknas på disken
            continue
        mask = get_mask(img)
        if mask is None:                         # ingen bit hittades
            continue
        ys, xs = np.nonzero(mask)                # koordinaterna för bitens pixlar
        angle, fit = rule_angle(mask, shape) if shape in SYMMETRY_DEG else (0.0, 1.0)   # cirkel: ingen rotation
        rows.append({
            "path": str(root / r["object_masked_path"]),
            "shape": shape,
            "cx": xs.mean() + 0.5,               # centre = medel av pixlarna (+0.5 = mitten av pixeln)
            "cy": ys.mean() + 0.5,
            "angle": angle,
            "fit": fit,
            "area": mask.sum(),
            "colour": cv2.cvtColor(img, cv2.COLOR_BGR2HSV)[..., 1][mask].mean(),   # mättnad: papper är gråaktigt
            "cut_off": mask[0].any() or mask[-1].any() or mask[:, 0].any() or mask[:, -1].any(),   # rör kanten?
        })
    crops = pd.DataFrame(rows)

    # Släng dåliga crops: grå (papper), fel storlek (kluster / fel bit), avklippta, eller formen passar inte
    size = crops["area"] / crops.groupby("shape")["area"].transform("median")
    good = (crops["colour"] > 120) & size.between(0.6, 1.5) & ~crops["cut_off"] & (crops["fit"] > 0.8)
    return crops[good].reset_index(drop=True)
