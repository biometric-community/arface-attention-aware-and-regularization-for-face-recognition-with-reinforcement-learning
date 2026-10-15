"""Figures + REPORT.md."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import yaml

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from arface.plot_style import FIGSIZE, LINEWIDTH_MAIN, MULTI_SERIES_COLORS, apply_rcparams, apply_style


def plot_train(hist, out: Path) -> None:
    apply_rcparams()
    fig, ax = plt.subplots(figsize=FIGSIZE)
    epochs = [h["epoch"] for h in hist]
    ce = [h["ce_loss"] for h in hist]
    ax.plot(epochs, ce, color=MULTI_SERIES_COLORS[0], linewidth=LINEWIDTH_MAIN, label="CE")
    rew = [(h["reward"] if h["reward"] is not None else np.nan) for h in hist]
    if any(np.isfinite(rew)):
        ax2 = ax.twinx()
        ax2.plot(epochs, rew, color=MULTI_SERIES_COLORS[1], linewidth=LINEWIDTH_MAIN, label="reward")
        ax2.set_ylabel("Batch accuracy reward")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("CE loss")
    ax.set_title("ARFace training (CelebA)")
    apply_style(ax)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out, format="svg")
    plt.close(fig)


def plot_verify(eval_res: dict, out: Path) -> None:
    apply_rcparams()
    fig, ax = plt.subplots(figsize=FIGSIZE)
    ver = eval_res.get("verification", {})
    keys = [k for k in ("lfw_full", "lfw_attn", "calfw_full", "calfw_attn", "cplfw_full", "cplfw_attn") if k in ver]
    vals = [ver[k]["accuracy"] * 100 for k in keys]
    x = np.arange(len(keys))
    ax.bar(x, vals, color=MULTI_SERIES_COLORS[0], edgecolor="#002080")
    ax.set_xticks(x)
    ax.set_xticklabels(keys, rotation=20, ha="right", fontsize=9)
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Verification (full vs attention-masked)")
    apply_style(ax, grid_axis="y")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out, format="svg")
    plt.close(fig)


def write_report(cfg, train_s, eval_res, fig_dir: Path, path: Path) -> None:
    protocol = cfg["train"].get("protocol", "full")
    lines = [
        "# REPORT — ARFace (Attention-Aware FR with RL)",
        "",
        f"**Protocol:** `{protocol}`",
        "",
        "Lab twin of Zhang et al. (TBIOM): Feature-Net + Attention-Net trained with",
        "REINFORCE (batch accuracy reward) and sparsity regularization.",
        "",
        "## Training",
        "",
        f"- Epochs: {train_s.get('epochs')} (warmup {train_s.get('warmup_epochs')})",
        f"- Train images: {train_s.get('n_train')}",
        f"- Identities: {train_s.get('num_classes')}",
        f"- Elapsed (s): {train_s.get('elapsed_sec', 0):.1f}",
        "",
        f"![Training]({fig_dir.as_posix()}/train_curves.svg)",
        "",
        "## Verification",
        "",
        "| Protocol | Accuracy | EER | N |",
        "|----------|----------|-----|---|",
    ]
    for key, v in sorted(eval_res.get("verification", {}).items()):
        lines.append(f"| {key} | {v['accuracy']*100:.2f}% | {v['eer']*100:.2f}% | {v['n_pairs']} |")
    lines += [
        "",
        f"![Verification]({fig_dir.as_posix()}/verification.svg)",
        "",
        "## Notes",
        "",
        "- Paper trains on VGGFace2 / MegaFace-scale data; this rebuild uses CelebA + LFW/CALFW/CPLFW.",
        "- Landmark-guided patch selection is approximated by a learned spatial grid Attention-Net.",
        "- See `DEVIATIONS.md`.",
        "",
    ]
    path.write_text("\n".join(lines))
    print(f"Wrote {path}")


def generate(cfg: dict) -> None:
    log_dir = Path(cfg["paths"]["log_dir"])
    fig_dir = Path(cfg["paths"]["figure_dir"])
    fig_dir.mkdir(parents=True, exist_ok=True)
    train_s = json.loads((log_dir / "train_summary.json").read_text())
    eval_res = json.loads((log_dir / "eval.json").read_text())
    plot_train(train_s["history"], fig_dir / "train_curves.svg")
    plot_verify(eval_res, fig_dir / "verification.svg")
    write_report(cfg, train_s, eval_res, fig_dir, Path("REPORT.md"))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    args = ap.parse_args()
    generate(yaml.safe_load(Path(args.config).read_text()))


if __name__ == "__main__":
    main()
