"""REPORT.md, the committed output of the full run, against the paper's claim."""

from __future__ import annotations

import re

import pytest


def accuracy(report: str, protocol: str) -> float:
    match = re.search(rf"\| {protocol} \| ([\d.]+)% \|", report)
    assert match, f"{protocol} row missing from REPORT.md"
    return float(match.group(1))


def test_recogniser_works(report, targets):
    assert accuracy(report, "lfw_attn") >= targets["clean_lfw_accuracy_floor"]


@pytest.mark.parametrize("dataset", ["lfw", "calfw", "cplfw"])
def test_attention_improves_recognition(report, dataset):
    """Abstract: attention-aware patch selection improves recognition over the plain network."""
    assert accuracy(report, f"{dataset}_attn") > accuracy(report, f"{dataset}_full")
