"""Feature-Net: ResNet-18 face embedder."""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


class FeatureNet(nn.Module):
    # Indices of layer1..layer4 inside `features` (conv1, bn1, relu, maxpool, layer1..4, avgpool)
    MASKED_STAGES = ("4", "5", "6", "7")

    def __init__(self, n_classes: int, pretrained: bool = True):
        super().__init__()
        weights = models.ResNet18_Weights.DEFAULT if pretrained else None
        net = models.resnet18(weights=weights)
        self.features = nn.Sequential(
            *[m for name, m in net.named_children() if name not in ("fc",)]
        )
        self.fc = nn.Linear(512, 512)
        self.bn = nn.BatchNorm1d(512)
        self.classifier = nn.Linear(512, n_classes)

    def embed(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """With `mask` (B,1,H,W), the regulariser of the paper: the input-layer mask is also
        applied to the intermediate feature maps after each residual stage."""
        for name, layer in self.features.named_children():
            x = layer(x)
            if mask is not None and name in self.MASKED_STAGES:
                x = x * F.interpolate(mask, size=x.shape[-2:], mode="bilinear", align_corners=False)
        x = self.bn(self.fc(torch.flatten(x, 1)))
        return F.normalize(x, p=2, dim=1)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None):
        emb = self.embed(x, mask=mask)
        return self.classifier(emb), emb
