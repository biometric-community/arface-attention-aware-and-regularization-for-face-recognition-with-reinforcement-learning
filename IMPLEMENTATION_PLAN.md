# IMPLEMENTATION_PLAN — ARFace

## Paper

Attention-Net selects face regions; Feature-Net classifies masked faces; Attention-Net trained by RL to maximize recognition accuracy + regularization.

## Lab twin

| Component | Choice |
|-----------|--------|
| Feature-Net | ResNet-18 + 512-d + CE |
| Attention-Net | CNN → 7×7 Bernoulli / soft mask |
| RL | REINFORCE with per-sample accuracy reward + EMA baseline |
| Reg | Mean attention sparsity toward target |
| Data | CelebA train; LFW/CALFW/CPLFW verify |
