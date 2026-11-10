"""Attention-Net, REINFORCE and the mask regulariser, on synthetic inputs."""

from __future__ import annotations

import inspect
from pathlib import Path

import numpy as np
import pytest
import torch

from arface.eval import best_acc_eer
from arface.models.attention_net import AttentionNet
from arface.models.feature_net import FeatureNet
from arface.rl.reinforce import reinforce_loss

ROOT = Path(__file__).resolve().parents[1]


def test_attention_masks_are_probabilities_at_image_size():
    torch.manual_seed(0)
    net = AttentionNet(grid=7).eval()
    x = torch.rand(2, 3, 112, 112)
    soft = net.soft_mask(x, 112)
    hard, log_prob, _, probs = net.sample_mask(x, 112)
    assert soft.shape == hard.shape == (2, 1, 112, 112)
    assert ((soft >= 0) & (soft <= 1)).all()
    assert set(hard.unique().tolist()) <= {0.0, 1.0}
    assert log_prob.shape == (2,)


def test_reinforce_raises_the_probability_of_rewarded_masks():
    """Abstract: the Attention-Net is trained with reinforcement learning to maximise accuracy."""
    logit = torch.zeros(1, requires_grad=True)
    log_prob = torch.distributions.Bernoulli(logits=logit).log_prob(torch.ones(1))
    reinforce_loss(log_prob, torch.ones(1), baseline=0.0).backward()
    assert logit.grad.item() < 0  # gradient descent increases the chosen action's logit


def test_reward_is_recognition_accuracy():
    source = (ROOT / "arface" / "train.py").read_text(encoding="utf-8")
    assert "argmax(dim=1) == labels" in source


def test_mask_is_also_applied_to_intermediate_feature_maps():
    """Abstract: the regulariser applies the input-layer mask to intermediate feature maps, so
    the Feature-Net has to accept the mask and use it beyond the input."""
    assert "mask" in inspect.signature(FeatureNet.forward).parameters
    torch.manual_seed(0)
    net = FeatureNet(5, pretrained=False).eval()
    x = torch.rand(2, 3, 112, 112)
    mask = torch.ones(2, 1, 112, 112)
    mask[:, :, :, :56] = 0
    with torch.no_grad():
        plain = net(x * mask)[0]
        regularised = net(x * mask, mask=mask)[0]
    assert not torch.allclose(plain, regularised)


def test_eer_is_where_false_match_and_false_non_match_rates_meet():
    """The EER is the error rate where FMR equals FNMR, not the lowest half total error rate."""
    scores = np.array([0.6, 0.7, 0.8, 0.9, 0.0, 0.1, 0.2, 0.65])
    labels = np.array([1, 1, 1, 1, 0, 0, 0, 0])
    assert best_acc_eer(scores, labels)[1] == pytest.approx(0.25, abs=0.01)
