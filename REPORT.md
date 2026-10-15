# REPORT — ARFace (Attention-Aware FR with RL)

**Protocol:** `full`

Lab twin of Zhang et al. (TBIOM): Feature-Net + Attention-Net trained with
REINFORCE (batch accuracy reward) and sparsity regularization.

## Training

- Epochs: 8 (warmup 2)
- Train images: 141610
- Identities: 7124
- Elapsed (s): 1107.4

![Training](outputs/figures/full/train_curves.svg)

## Verification

| Protocol | Accuracy | EER | N |
|----------|----------|-----|---|
| calfw_attn | 52.20% | 47.80% | 6000 |
| calfw_full | 52.23% | 47.77% | 6000 |
| cplfw_attn | 53.42% | 46.58% | 6000 |
| cplfw_full | 54.02% | 45.98% | 6000 |
| lfw_attn | 58.78% | 41.22% | 6000 |
| lfw_full | 61.25% | 38.75% | 6000 |

![Verification](outputs/figures/full/verification.svg)

## Notes

- Paper trains on VGGFace2 / MegaFace-scale data; this rebuild uses CelebA + LFW/CALFW/CPLFW.
- Landmark-guided patch selection is approximated by a learned spatial grid Attention-Net.
- See `DEVIATIONS.md`.
