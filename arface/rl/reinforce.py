"""REINFORCE helpers for Attention-Net."""

from __future__ import annotations

import torch


def reinforce_loss(log_prob: torch.Tensor, reward: torch.Tensor, baseline: float = 0.0) -> torch.Tensor:
    """Maximize expected reward → minimize -(R - b) * log π."""
    adv = reward.detach() - baseline
    return -(adv * log_prob).mean()


def sparsity_reg(probs: torch.Tensor, target: float = 0.5) -> torch.Tensor:
    """Encourage mean attention near target (paper-style regularization)."""
    return (probs.mean() - target).pow(2)
