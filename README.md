# ARFace: Attention-Aware and Regularization for Face Recognition With Reinforcement Learning

[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-ee4c2c.svg)](https://pytorch.org/)
[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)
[![License: CC BY 4.0](https://img.shields.io/badge/License-CC%20BY%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by/4.0/)

Lab twin of **ARFace**: Feature-Net + Attention-Net, with REINFORCE on recognition accuracy and sparsity regularization.

**Repository:** https://github.com/biometric-community/arface-attention-aware-and-regularization-for-face-recognition-with-reinforcement-learning

**Paper:** [10.1109/TBIOM.2021.3104014](https://doi.org/10.1109/TBIOM.2021.3104014)

## Setup

```bash
bash scripts/setup_env.sh
```

## Train / Eval / Report

```bash
bash scripts/smoke.sh
bash scripts/train_full.sh
```

See [REPORT.md](./REPORT.md) and [DEVIATIONS.md](./DEVIATIONS.md).

## License

[CC BY 4.0](./LICENSE)
