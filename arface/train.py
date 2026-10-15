"""Train Feature-Net + Attention-Net (ARFace lab twin)."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from arface.data.celeba import CelebAIdDataset, resolve_celeba_root
from arface.models.attention_net import AttentionNet
from arface.models.feature_net import FeatureNet
from arface.rl.reinforce import reinforce_loss, sparsity_reg


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def _device(cfg: dict, override: Optional[str] = None) -> torch.device:
    name = override or cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu")
    if str(name).startswith("cuda") and not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(name)


def train(cfg: dict, device_override: Optional[str] = None) -> Dict:
    device = _device(cfg, device_override)
    root = resolve_celeba_root(cfg)
    seed = int(cfg["train"].get("seed", 42))
    torch.manual_seed(seed)
    np.random.seed(seed)

    max_n = cfg["data"].get("max_images")
    max_n = int(max_n) if max_n is not None else None
    image_size = int(cfg["model"].get("image_size", 112))
    train_ds = CelebAIdDataset(
        root,
        split="train",
        image_size=image_size,
        train_ratio=float(cfg["data"].get("train_ratio", 0.7)),
        max_n=max_n,
        seed=seed,
        train=True,
    )
    loader = DataLoader(
        train_ds,
        batch_size=int(cfg["train"].get("batch_size", 64)),
        shuffle=True,
        num_workers=int(cfg["train"].get("num_workers", 4)),
        pin_memory=device.type == "cuda",
        drop_last=True,
    )

    feat = FeatureNet(train_ds.num_classes, pretrained=True).to(device)
    attn = AttentionNet(grid=int(cfg["model"].get("attn_grid", 7))).to(device)
    opt_f = torch.optim.SGD(feat.parameters(), lr=float(cfg["train"].get("lr", 0.05)), momentum=0.9, weight_decay=5e-4)
    opt_a = torch.optim.Adam(attn.parameters(), lr=float(cfg["train"].get("attn_lr", 1e-3)))

    epochs = int(cfg["train"].get("epochs", 8))
    warmup = int(cfg["train"].get("warmup_epochs", 2))
    lambda_sp = float(cfg["train"].get("sparsity_weight", 0.1))
    target_sp = float(cfg["train"].get("sparsity_target", 0.6))
    use_soft = bool(cfg["train"].get("soft_mask_for_feature", True))

    hist = []
    reward_ema = 0.0
    t0 = time.time()
    for epoch in range(1, epochs + 1):
        feat.train()
        attn.train()
        losses_ce, losses_rl, rewards, spars = [], [], [], []
        pbar = tqdm(loader, desc=f"ep{epoch}/{epochs}", leave=False)
        for images, labels in pbar:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            if epoch <= warmup:
                logits, _ = feat(images)
                loss_ce = F.cross_entropy(logits, labels)
                opt_f.zero_grad(set_to_none=True)
                loss_ce.backward()
                opt_f.step()
                losses_ce.append(float(loss_ce.detach()))
                continue

            hard, log_prob, soft, probs = attn.sample_mask(images, image_size)
            # Feature-Net: soft mask detached so CE does not backprop into Attention-Net
            mask_f = soft.detach() if use_soft else hard.detach()
            logits, _ = feat(images * mask_f)
            loss_ce = F.cross_entropy(logits, labels)
            opt_f.zero_grad(set_to_none=True)
            loss_ce.backward()
            opt_f.step()

            # Attention-Net: REINFORCE with hard-mask accuracy reward (Feature frozen in graph)
            with torch.no_grad():
                logits_r, _ = feat(images * hard)
                reward = (logits_r.argmax(dim=1) == labels).float()
                rewards.append(float(reward.mean()))
            reward_ema = 0.9 * reward_ema + 0.1 * float(reward.mean())
            loss_rl = reinforce_loss(log_prob, reward, baseline=reward_ema)
            loss_sp = sparsity_reg(probs, target=target_sp)
            loss_attn = loss_rl + lambda_sp * loss_sp
            opt_a.zero_grad(set_to_none=True)
            loss_attn.backward()
            opt_a.step()

            losses_ce.append(float(loss_ce.detach()))
            losses_rl.append(float(loss_rl.detach()))
            spars.append(float(probs.mean().detach()))
            pbar.set_postfix(ce=f"{losses_ce[-1]:.3f}", r=f"{rewards[-1]:.2f}")

        row = {
            "epoch": epoch,
            "ce_loss": float(np.mean(losses_ce)) if losses_ce else None,
            "rl_loss": float(np.mean(losses_rl)) if losses_rl else None,
            "reward": float(np.mean(rewards)) if rewards else None,
            "attn_mean": float(np.mean(spars)) if spars else None,
            "warmup": epoch <= warmup,
        }
        hist.append(row)
        print(
            f"[train] epoch={epoch} ce={row['ce_loss']} rl={row['rl_loss']} "
            f"reward={row['reward']} attn_mean={row['attn_mean']}"
        )

    elapsed = time.time() - t0
    ckpt_dir = Path(cfg["paths"]["checkpoint_dir"])
    log_dir = Path(cfg["paths"]["log_dir"])
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "feature": feat.state_dict(),
            "attention": attn.state_dict(),
            "num_classes": train_ds.num_classes,
            "config": cfg,
        },
        ckpt_dir / "last.pt",
    )
    summary = {
        "protocol": cfg["train"].get("protocol", "full"),
        "epochs": epochs,
        "warmup_epochs": warmup,
        "n_train": len(train_ds),
        "num_classes": train_ds.num_classes,
        "elapsed_sec": elapsed,
        "history": hist,
        "checkpoint": str(ckpt_dir / "last.pt"),
    }
    with open(log_dir / "train_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
        f.write("\n")
    print(f"Wrote {log_dir / 'train_summary.json'} elapsed={elapsed:.1f}s")
    return summary


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--device", default=None)
    args = ap.parse_args()
    train(load_config(args.config), device_override=args.device)


if __name__ == "__main__":
    main()
