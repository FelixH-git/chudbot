"""Inställningar som delas av training och roboten. Ändra här, ingen annanstans."""
from pathlib import Path

# Formerna, i samma ordning som nätverkets outputs. Ändra INTE ordningen .
CLASSES = ["Circle", "Hexagon", "Square", "Star", "Triangle"]

# Efter så här många grader ser formen likadan ut igen. Cirklar har ingen rotation.
SYMMETRY_DEG = {"Hexagon": 60, "Square": 90, "Star": 72, "Triangle": 120}

# Nätverkets input
IMG_SIZE = 128                        # crops görs om till 128 x 128
CANVAS_MARGIN = 0.15                  # svart kant runt cropen innan resize

# Normalisering: pixel = (pixel - NORM_MEAN) / NORM_STD, per färgkanal (R, G, B), pixlar på skalan 0..1.
# Det flyttar värdena så att de ligger runt 0 med spridning ca ±1, vilket nätverk lär sig lättare av.
#
# Var siffrorna kommer ifrån: medelvärde (mean) och spridning (std) för alla pixlar i ImageNets
# ~1,2 miljoner training-bilder. Vår backbone (MobileNetV3-small, weights IMAGENET1K_V1)
# är pretrained med exakt denna normalisering, så vi måste använda samma siffror för att den ska
# känna igen kanter och former lika bra. Kolla själv:
#     from torchvision.models import MobileNet_V3_Small_Weights
#     print(MobileNet_V3_Small_Weights.IMAGENET1K_V1.transforms())
# Byter man till en backbone med andra pretrained weights: ta siffrorna från dess .transforms().
NORM_MEAN = (0.485, 0.456, 0.406)     # R, G, B
NORM_STD = (0.229, 0.224, 0.225)      # R, G, B

# Var filerna ligger
PACKAGE_DIR = Path(__file__).parent
MODELS_DIR = PACKAGE_DIR / "models"
WEIGHTS_PATH = MODELS_DIR / "crpnet.pth"     # PyTorch-weights (training)
ONNX_PATH = MODELS_DIR / "crpnet.onnx"       # exporterad modell (roboten)

DATASET_CANDIDATES = [
    Path(r"C:/Users/Nils/Downloads/dataset/dataset"),             # Windows PC
    Path.home() / "Desktop/PhysicalAIBinPicking/chudbot/dataset",  # Linux-datorn
]


def dataset_root():
    """Den första dataset-mappen som finns på den här datorn."""
    for path in DATASET_CANDIDATES:
        if (path / "labels.csv").exists():
            return path
    raise FileNotFoundError("No dataset found. Add its folder to DATASET_CANDIDATES in config.py")
