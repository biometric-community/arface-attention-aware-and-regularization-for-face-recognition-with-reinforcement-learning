"""Embed one image with optional Attention-Net soft mask."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
import yaml
from PIL import Image
from torchvision import transforms

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from arface.models.attention_net import AttentionNet
from arface.models.feature_net import FeatureNet


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--image", required=True)
    ap.add_argument("--attn", action="store_true")
    ap.add_argument("--device", default=None)
    args = ap.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text())
    device_name = args.device or cfg.get("device", "cpu")
    device = torch.device(
        device_name if not str(device_name).startswith("cuda") or torch.cuda.is_available() else "cpu"
    )
    blob = torch.load(Path(cfg["paths"]["checkpoint_dir"]) / "last.pt", map_location=device, weights_only=False)
    feat = FeatureNet(int(blob["num_classes"]), pretrained=False).to(device)
    feat.load_state_dict(blob["feature"])
    feat.eval()
    size = int(cfg["model"].get("image_size", 112))
    tf = transforms.Compose(
        [
            transforms.Resize((size, size)),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ]
    )
    x = tf(Image.open(args.image).convert("RGB")).unsqueeze(0).to(device)
    if args.attn:
        attn = AttentionNet(grid=int(cfg["model"].get("attn_grid", 7))).to(device)
        attn.load_state_dict(blob["attention"])
        attn.eval()
        with torch.no_grad():
            x = x * attn.soft_mask(x, size)
    with torch.no_grad():
        emb = feat.embed(x).cpu().numpy()[0]
    out_dir = Path(cfg["paths"]["pred_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / "embedding.json"
    with open(out, "w") as f:
        json.dump({"attn": args.attn, "embedding": emb.tolist()}, f)
        f.write("\n")
    print(f"Wrote {out} dim={len(emb)}")


if __name__ == "__main__":
    main()
