"""Tränar CRPNet och exporterar den till ONNX. Kör från ReWorks-mappen:  python -m neural_net.train"""
import cv2
import numpy as np
import torch
from torch import nn

from .config import CLASSES, IMG_SIZE, MODELS_DIR, ONNX_PATH, SYMMETRY_DEG, WEIGHTS_PATH
from .dataset import make_loaders
from .labels import load_crops, wrap
from .model import CRPNet
from .preprocess import square_canvas, to_input

EPOCHS = 60
W_ROTATION, W_CENTRE = 2.0, 10.0     # hur mycket rotation och centre väger jämfört med class
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# k per form (cirkel = 1 så vi aldrig delar med 0)
K = torch.tensor([360 // SYMMETRY_DEG.get(c, 360) for c in CLASSES], dtype=torch.float32, device=DEVICE)


def rotation_pair(rotation_out, cls):
    """Plockar ut (sin, cos)-paret för rätt form ur (batch, 5, 2)."""
    return rotation_out[torch.arange(len(cls)), cls]


def compute_losses(outputs, cls, centre, rotation, has_rotation):
    """En loss per head. Lägre = bättre."""
    class_out, rotation_out, centre_out = outputs

    # Head 1: cross entropy, straffar låg sannolikhet för rätt form
    class_loss = nn.functional.cross_entropy(class_out, cls)

    # Head 2: 1 - cos(skillnaden) på k-cirkeln, + håller paret nära längd 1. Cirklar räknas inte
    pair = rotation_pair(rotation_out, cls)
    per_item = 1 - torch.cosine_similarity(pair, rotation) + 0.1 * (pair.norm(dim=1) - 1) ** 2
    rotation_loss = (per_item * has_rotation).sum() / has_rotation.sum().clamp(min=1)

    # Head 3: avståndet mellan gissad och rätt centre
    centre_loss = nn.functional.smooth_l1_loss(centre_out, centre, beta=0.02)

    return class_loss, rotation_loss, centre_loss


@torch.no_grad()
def evaluate(model, loader):
    """Rätt class (andel), centre-fel (pixlar i 128-bilden) och rotation-fel (grader) per form."""
    model.eval()
    correct, total, centre_errors, rot_errors, rot_classes = 0, 0, [], [], []
    for x, cls, centre, rotation, has_rotation in loader:
        x, cls, centre, rotation = x.to(DEVICE), cls.to(DEVICE), centre.to(DEVICE), rotation.to(DEVICE)
        class_out, rotation_out, centre_out = model(x)

        correct += (class_out.argmax(dim=1) == cls).sum().item()
        total += len(cls)
        centre_errors.append(((centre_out - centre).norm(dim=1) * IMG_SIZE).cpu())

        # Vinkelskillnad på k-cirkeln, sedan / k = riktiga grader
        pair = rotation_pair(rotation_out, cls)
        diff = torch.atan2(pair[:, 0], pair[:, 1]) - torch.atan2(rotation[:, 0], rotation[:, 1])
        diff = torch.atan2(torch.sin(diff), torch.cos(diff))
        keep = has_rotation > 0
        rot_errors.append((torch.rad2deg(diff.abs()) / K[cls]).cpu()[keep])
        rot_classes.append(cls.cpu()[keep])

    rot_errors, rot_classes = torch.cat(rot_errors), torch.cat(rot_classes)
    return {
        "class": correct / total,
        "centre_px": torch.cat(centre_errors).mean().item(),
        "rotation": {s: rot_errors[rot_classes == CLASSES.index(s)].mean().item() for s in SYMMETRY_DEG},
    }


@torch.no_grad()
def final_check(model, crops):
    """Samma kontroll som i notebooken: ovridna crops, centre i crop-pixlar, rotation mot regeln."""
    model.eval()
    correct, centre_errors, rot_errors = 0, [], {s: [] for s in SYMMETRY_DEG}
    for _, row in crops.iterrows():
        canvas, (x0, y0, side) = square_canvas(cv2.imread(row["path"]))
        class_out, rotation_out, centre_out = model(torch.from_numpy(to_input(canvas))[None].to(DEVICE))
        shape = CLASSES[class_out.argmax().item()]
        correct += shape == row["shape"]
        cx, cy = centre_out[0].tolist()
        centre_errors.append(np.hypot(cx * side - x0 - row["cx"], cy * side - y0 - row["cy"]))
        if row["shape"] in SYMMETRY_DEG:
            k = 360 // SYMMETRY_DEG[row["shape"]]
            s, c = rotation_out[0, CLASSES.index(row["shape"])].tolist()
            angle = np.degrees(np.arctan2(s, c)) / k
            rot_errors[row["shape"]].append(abs(wrap(angle - row["angle"], SYMMETRY_DEG[row["shape"]])))
    print(f"class:    {correct}/{len(crops)} rätt")
    print(f"centre:   {np.mean(centre_errors):.1f} px fel i snitt (crop-pixlar)")
    for s, e in rot_errors.items():
        print(f"rotation: {s:<9} {np.mean(e):.2f}° fel i snitt")


def export_onnx(model):
    """Sparar modellen som ONNX, det roboten använder (behöver inte PyTorch)."""
    model.eval().cpu()
    torch.onnx.export(
        model, torch.randn(1, 3, IMG_SIZE, IMG_SIZE), str(ONNX_PATH),
        input_names=["input"], output_names=["class_logits", "rotation_sincos", "centre"],
        dynamic_axes={"input": {0: "batch"}}, opset_version=17,
    )
    print(f"sparade {ONNX_PATH}")


def main():
    torch.manual_seed(0)
    MODELS_DIR.mkdir(exist_ok=True)

    crops = load_crops()
    train_loader, val_loader = make_loaders(crops)
    print(f"{len(train_loader.dataset.crops)} crops training, {len(val_loader.dataset.crops)} validation, {DEVICE}")

    model = CRPNet().to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, EPOCHS)   # learning rate sjunker mjukt mot 0

    best = float("inf")
    for epoch in range(1, EPOCHS + 1):
        model.train()
        for x, cls, centre, rotation, has_rotation in train_loader:
            x, cls, centre = x.to(DEVICE), cls.to(DEVICE), centre.to(DEVICE)
            rotation, has_rotation = rotation.to(DEVICE), has_rotation.to(DEVICE)

            class_loss, rotation_loss, centre_loss = compute_losses(model(x), cls, centre, rotation, has_rotation)
            loss = class_loss + W_ROTATION * rotation_loss + W_CENTRE * centre_loss

            optimizer.zero_grad()      # nollställ gamla gradients
            loss.backward()            # backpropagation: hur varje weight påverkar lossen
            optimizer.step()           # optimizer: ändra alla weights lite åt rätt håll
        scheduler.step()

        # Ett enda tal för "hur bra": rotation-fel + centre-fel + 1 per procent fel class. Spara den bästa
        result = evaluate(model, val_loader)
        rotation_error = np.mean(list(result["rotation"].values()))
        score = rotation_error + result["centre_px"] + 100 * (1 - result["class"])
        saved = ""
        if score < best:
            best = score
            torch.save(model.state_dict(), WEIGHTS_PATH)
            saved = "(sparad)"
        print(f"epoch {epoch:2d} | class {result['class']:.0%} | centre {result['centre_px']:4.2f} px | "
              f"rotation {rotation_error:4.2f}° {saved}")

    # Bästa modellen: slutkontroll på validation-crops, sedan export
    model.load_state_dict(torch.load(WEIGHTS_PATH, map_location=DEVICE))
    print("\nSlutkontroll (validation):")
    final_check(model, val_loader.dataset.crops)
    export_onnx(model)


if __name__ == "__main__":
    main()
