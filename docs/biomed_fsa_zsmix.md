# BiomedCoOp — FSA + BiomedCLIP-ZS-Mix (K=1, 4, 16)

> Few-shot classification on the 11 BiomedCoOp medical datasets using
> DINOv3-B (vision) + BiomedCLIP text encoder
> (`hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224`).
> Prompt: `"a biomedical image of {}."`.

## Method

- **FSA (baseline)**: train the flow adapter on K-shot support; at inference,
  tune (α, t_end) on val — after-tune test accuracy reported.
- **FSA + BiomedCLIP-ZS-Mix**: at inference, mix FSA's MT logits with
  pre-computed BiomedCLIP zero-shot logits on the same test set:

  ```
  logits = (1 − α) · MT + α · BiomedCLIP_ZS
  ```

  The external ZS logits are rescaled to match FSA's logit scale before
  mixing (`external_zs / 100` when model.logit_scale = 0).
  `t_end` is fixed to the per-dataset value picked by the FSA training-time
  tuner (seed 42). α is **fixed** per shot: **α=0.5 for K=1** and **α=0.25
  for K=4 and K=16**. No per-dataset α tuning.

---

## Per-dataset results

FSA is 3-seed mean ± std over seeds {42, 1, 2}. ZS-mix is seed 42 only.

| dataset | FSA K=1 | +ZS-Mix K=1 (α=0.5) | FSA K=4 | +ZS-Mix K=4 (α=0.25) | FSA K=16 | +ZS-Mix K=16 (α=0.25) |
|:---|---:|---:|---:|---:|---:|---:|
| BUSI | 39.27 ± 5.60 | 37.29 | 51.69 ± 2.10 | 47.88 | 74.58 ± 0.92 | **75.00** |
| KneeXray | 28.30 ± 1.30 | **35.21** | 31.82 ± 2.18 | **38.29** | 35.93 ± 1.21 | **37.14** |
| CHMNIST | 64.03 ± 9.47 | 52.46 | 77.75 ± 1.54 | **77.86** | 87.54 ± 1.54 | **88.10** |
| BTMRI | 53.07 ± 0.34 | **62.19** | 72.56 ± 0.81 | **74.68** | 79.27 ± 2.82 | **84.07** |
| COVID_19 | 52.41 ± 9.84 | **67.75** | 62.79 ± 2.38 | **72.10** | 70.92 ± 2.23 | **74.07** |
| CTKidney | 37.89 ± 3.40 | **38.66** | 52.43 ± 7.40 | **54.83** | 75.73 ± 2.90 | **78.63** |
| DermaMNIST | 41.93 ± 17.68 | **49.33** | 52.95 ± 2.70 | 50.70 | 57.93 ± 0.99 | 56.69 |
| Kvasir | 53.58 ± 2.97 | **61.83** | 75.00 ± 3.61 | **80.83** | 85.50 ± 0.88 | **85.92** |
| LungColon | 83.40 ± 3.73 | 82.84 | 90.51 ± 0.94 | 89.39 | 94.80 ± 0.41 | 94.17 |
| OCTMNIST | 46.15 ± 4.33 | 26.08 | 60.33 ± 2.93 | 56.97 | 70.95 ± 1.10 | **73.28** |
| RETINA | 58.86 ± 3.14 | 58.47 | 64.35 ± 1.85 | 62.03 | 74.39 ± 0.88 | 72.47 |


## BiomedCLIP zero-shot baselines (for reference)

| dataset | BiomedCLIP ZS | dataset | BiomedCLIP ZS |
|:---|---:|:---|---:|
| BUSI | 38.14 | DermaMNIST | 21.24 |
| KneeXray | 33.51 | Kvasir | 50.50 |
| CHMNIST | 34.38 | LungColon | 39.96 |
| BTMRI | 63.36 | OCTMNIST | 19.96 |
| COVID_19 | 62.97 | RETINA | 28.56 |
| CTKidney | 38.29 |  |  |

## Notes

- **Seeds.** FSA is 3-seed (42, 1, 2). ZS-Mix is **seed 42 only**; multi-seed
  confirmation is a follow-up.
- **α selection.** α=0.25 and α=0.5 are picked from the K × dataset sweep
  mean curves (see internal logs), not tuned per-dataset. This matches our
  ImageNet CLIP-ZS-Mix protocol where a single α worked well across datasets.
- **Training recipe.** 200 epochs, lr=5e-5, wd=1e-3, batch_size=32, DINOv3-B +
  BiomedCLIP text. Prompt `"a biomedical image of {}."`.
