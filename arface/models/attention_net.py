"""Attention-Net: predicts spatial attention logits over a coarse grid."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class AttentionNet(nn.Module):
    """
    CNN over the face image → (B, 1, Gh, Gw) logits.
    Soft mask via sigmoid; hard mask via Bernoulli for REINFORCE.
    """

    def __init__(self, grid: int = 7):
        super().__init__()
        self.grid = grid
        self.backbone = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((grid, grid)),
            nn.Conv2d(128, 1, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)

    def soft_mask(self, x: torch.Tensor, out_size: int) -> torch.Tensor:
        logits = self.forward(x)
        m = torch.sigmoid(logits)
        return F.interpolate(m, size=(out_size, out_size), mode="bilinear", align_corners=False)

    def sample_mask(self, x: torch.Tensor, out_size: int):
        """Return hard mask, log_prob, soft probs for regularization."""
        logits = self.forward(x)
        probs = torch.sigmoid(logits)
        dist = torch.distributions.Bernoulli(probs=probs.clamp(1e-4, 1 - 1e-4))
        hard = dist.sample()
        log_prob = dist.log_prob(hard).sum(dim=(1, 2, 3))
        hard_up = F.interpolate(hard, size=(out_size, out_size), mode="nearest")
        soft_up = F.interpolate(probs, size=(out_size, out_size), mode="bilinear", align_corners=False)
        return hard_up, log_prob, soft_up, probs
