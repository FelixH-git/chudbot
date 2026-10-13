"""CRPNet: ett nätverk med tre heads. Class, rotation och centre från en och samma bild."""
from torch import nn
from torchvision.models import MobileNet_V3_Small_Weights, mobilenet_v3_small

from .config import CLASSES, IMG_SIZE


class CRPNet(nn.Module):
    """
    In:  bilder (batch, 3, 128, 128)
    Ut:  class    (batch, 5)     poäng per form, störst = gissningen
         rotation (batch, 5, 2)  ett [sin(k*vinkel), cos(k*vinkel)]-par per form, använd raden för formen
         centre   (batch, 2)     (cx, cy) som 0..1 av bilden
    """

    def __init__(self, pretrained=True):
        super().__init__()
        n = len(CLASSES)
        grid = IMG_SIZE // 32                    # backbone krymper bilden 32 gånger: 128 -> 4x4 grid

        # Backbone: MobileNetV3-small, pretrained på ImageNet. Bild -> 576 features i ett 4x4 grid
        weights = MobileNet_V3_Small_Weights.IMAGENET1K_V1 if pretrained else None
        self.backbone = mobilenet_v3_small(weights=weights).features

        # Head 1, class: medelvärde över hela griden räcker (VAR biten ligger spelar ingen roll)
        self.class_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
            nn.Linear(576, 128), nn.Hardswish(), nn.Dropout(0.2),
            nn.Linear(128, n),
        )

        # Head 2 och 3 behöver veta VAR saker är, så de får hela 4x4-griden (krympt till 64 channels)
        self.spatial = nn.Sequential(
            nn.Conv2d(576, 64, kernel_size=1), nn.BatchNorm2d(64), nn.Hardswish(), nn.Flatten(),
        )

        # Head 2, rotation: ett (sin, cos)-par per form
        self.rotation_head = nn.Sequential(
            nn.Linear(64 * grid * grid, 128), nn.Hardswish(), nn.Dropout(0.1),
            nn.Linear(128, n * 2),
        )

        # Head 3, centre: (cx, cy). Sigmoid håller svaret mellan 0 och 1
        self.centre_head = nn.Sequential(
            nn.Linear(64 * grid * grid, 64), nn.Hardswish(),
            nn.Linear(64, 2), nn.Sigmoid(),
        )

    def forward(self, x):
        features = self.backbone(x)              # (batch, 576, 4, 4)
        spatial = self.spatial(features)         # (batch, 64*4*4)
        classes = self.class_head(features)
        rotation = self.rotation_head(spatial).view(-1, len(CLASSES), 2)
        centre = self.centre_head(spatial)
        return classes, rotation, centre
