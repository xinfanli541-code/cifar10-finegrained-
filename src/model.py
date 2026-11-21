"""Models for the CIFAR-10 comparison experiments.

SimpleCNN: hand-written small CNN (~600k params).
ResNet18-CIFAR: torchvision ResNet18 adapted for 32x32 inputs
(3x3 stride-1 first conv, no early maxpool), ~11M params.
"""

import torch.nn as nn
from torchvision import models


class SimpleCNN(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.features = nn.Sequential(
            # 32x32
            nn.Conv2d(3, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.MaxPool2d(2), nn.Dropout(0.25),
            # 16x16
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.MaxPool2d(2), nn.Dropout(0.25),
            # 8x8
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.MaxPool2d(2), nn.Dropout(0.25),
        )
        # 4x4
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 256), nn.ReLU(inplace=True), nn.Dropout(0.5),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


def build_resnet18_cifar(num_classes=10, pretrained=False):
    """ResNet18 with a CIFAR-friendly stem. If pretrained=True, load
    ImageNet weights for all layers except the replaced stem/fc
    (skipped silently if the download fails)."""
    weights = None
    if pretrained:
        try:
            weights = models.ResNet18_Weights.DEFAULT
        except Exception as e:
            print(f'pretrained weights unavailable ({e}); training from scratch')
    model = models.resnet18(weights=weights)
    model.conv1 = nn.Conv2d(3, 64, 3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    if pretrained and weights is not None:
        # reload, ignoring the mismatched stem/fc keys
        try:
            sd = models.resnet18(weights=weights).state_dict()
            own = model.state_dict()
            loaded = {k: v for k, v in sd.items()
                      if k in own and v.shape == own[k].shape}
            own.update(loaded)
            model.load_state_dict(own)
            print(f'loaded {len(loaded)}/{len(own)} pretrained tensors')
        except Exception as e:
            print(f'pretrained weight load failed ({e}); training from scratch')
    return model


def build_model(name, num_classes=10, pretrained=False):
    if name == 'simplecnn':
        return SimpleCNN(num_classes)
    if name == 'resnet18':
        return build_resnet18_cifar(num_classes, pretrained)
    raise ValueError(f'unknown model: {name}')


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
