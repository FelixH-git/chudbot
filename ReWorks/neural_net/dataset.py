"""Gör om tabellen från labels.py till training-bilder. Bilderna vrids slumpmässigt och alla labels följer med."""
import cv2
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset

from .config import CLASSES, SYMMETRY_DEG
from .preprocess import square_canvas, to_input


class CRPDataset(Dataset):
    """
    Ger (bild, class, centre, rotation, har_rotation) för varje exempel.
      bild:         (3, 128, 128)
      class:        index i CLASSES
      centre:       (cx, cy) som 0..1 av kvadraten
      rotation:     [sin(k*vinkel), cos(k*vinkel)]
      har_rotation: 1, eller 0 för cirklar (då räknas rotationen inte i training)
    """

    def __init__(self, crops, train, copies=4):
        self.crops = crops.reset_index(drop=True)
        self.train = train
        self.copies = copies                     # varje crop används 4 gånger per epoch, vriden olika
        # Läs in alla bilder en gång och lägg dem i kvadraten direkt
        self.canvases = [square_canvas(cv2.imread(path)) for path in self.crops["path"]]

    def __len__(self):
        return len(self.crops) * self.copies

    def __getitem__(self, i):
        j = i % len(self.crops)
        row = self.crops.iloc[j]
        canvas, (x0, y0, side) = self.canvases[j]
        # Training: ny slump varje gång. Validation: samma vridning varje epoch (seed = i)
        rng = np.random.default_rng(None if self.train else i)

        # Vrid (och vid training: skala + flytta lite) runt kvadratens mitt
        turn = rng.uniform(-180, 180)
        scale = rng.uniform(0.9, 1.1) if self.train else 1.0
        M = cv2.getRotationMatrix2D((side / 2, side / 2), turn, scale)
        if self.train:
            M[:, 2] += rng.uniform(-0.04, 0.04, size=2) * side
        img = cv2.warpAffine(canvas, M, (side, side))

        # Labels följer med vridningen
        cx, cy = M @ [row["cx"] + x0, row["cy"] + y0, 1]   # centre flyttas med samma matris som bilden
        angle = row["angle"] - turn                         # cv2 vrider moturs på skärmen = vinkeln minskar

        if self.train:
            img = np.clip(img * rng.uniform(0.75, 1.25), 0, 255).astype(np.uint8)   # ljusstyrka
            if rng.random() < 0.5:
                img = img[..., rng.permutation(3)].copy()                          # byt färgkanaler

        if row["shape"] in SYMMETRY_DEG:
            k = 360 // SYMMETRY_DEG[row["shape"]]
            rotation = [np.sin(np.radians(k * angle)), np.cos(np.radians(k * angle))]
            has_rotation = 1.0
        else:
            rotation, has_rotation = [0.0, 1.0], 0.0         # cirkel: värdet spelar ingen roll

        return (
            torch.from_numpy(to_input(img)),
            CLASSES.index(row["shape"]),
            torch.tensor([cx / side, cy / side], dtype=torch.float32),
            torch.tensor(rotation, dtype=torch.float32),
            torch.tensor(has_rotation, dtype=torch.float32),
        )


def make_loaders(crops, batch_size=32):
    """Delar crops 80 / 20 i training och validation (samma andel av varje form) och gör DataLoaders."""
    train_crops, val_crops = train_test_split(crops, test_size=0.2, random_state=42, stratify=crops["shape"])
    train_loader = DataLoader(CRPDataset(train_crops, train=True), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(CRPDataset(val_crops, train=False), batch_size=64)
    return train_loader, val_loader
