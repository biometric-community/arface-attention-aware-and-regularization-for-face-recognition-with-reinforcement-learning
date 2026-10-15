# DEVIATIONS — ARFace

| ID | Item | Paper | Ours | Reason |
|----|------|-------|------|--------|
| D1 | Train corpus | VGGFace2 / MegaFace | CelebA | Local public data |
| D2 | Patch selection | Landmark-guided patches | Spatial grid Attention-Net | No landmark detector required |
| D3 | Backbone | Deeper FR nets | ResNet-18 | Lab twin compute |
| D4 | Benchmarks | IJB / MegaFace / YTF | LFW/CALFW/CPLFW | Available locally |
| D5 | RL details | Paper-specific reward schedule | REINFORCE + EMA baseline | Faithful spirit, simplified |
