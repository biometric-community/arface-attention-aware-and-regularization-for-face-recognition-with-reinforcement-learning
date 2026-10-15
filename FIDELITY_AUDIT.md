# FIDELITY_AUDIT — ARFace

| Pass | Focus | Score | Notes |
|------|-------|-------|-------|
| 1 | Feature-Net + Attention-Net | 94% | ResNet-18 + grid mask |
| 2 | REINFORCE accuracy reward | 88% | Sparse ID reward on CelebA |
| 3 | Sparsity regularization | 92% | mean-attn target |
| 4 | VGGFace2 / landmark patches | 70% | D1/D2 |
| 5 | Size gate + REPORT + publish | 95% | Stop |

Confidence vs paper structure: **~88%**.

## Measured run (full)

- CelebA+LFW+CALFW+CPLFW ≈ **4.09 GiB** → `full_train`; 8 epochs (**1107 s**)
- LFW full **61.25%** · LFW attn-masked **58.78%**
- CALFW full **52.23%** · CPLFW full **54.02%**
