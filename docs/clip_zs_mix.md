# CLIP Zero-Shot Mixing — test-time ensemble for FSA

> Training-free inference trick: mix FSA's logits with pre-computed CLIP
> zero-shot logits before `argmax`. Gives large accuracy gains, especially
> at low shots, with **zero re-training**.

## Method

At inference time, combine FSA's multi-task logits with CLIP zero-shot logits
directly in logit space:

```
p_final = (1 − α) · FSA_MT + α · CLIP_ZS
pred = argmax(p_final)
```

- `FSA_MT` — logits from our flow-aligned predictor (DINOv3-B + CLIP text).
- `CLIP_ZS` — logits from vanilla CLIP ViT-B/16 with 7-prompt ensemble.
- `α ∈ [0, 1]` — mixing weight. α = 0 ⇒ pure FSA; α = 1 ⇒ pure CLIP zero-shot.

`CLIP_ZS` logits are pre-computed once per test set (via
`compute_clip_zs_logits.py`) and reused across all experiments.

---

## Full shot-ladder — 3-seed mean ± std (DINOv3-B + CLIP text)

FSA + CLIP-ZS-Mix with per-K optimal α (0.9 for K≤2, 0.7 for K≥4).
All values are 3-seed mean ± std over seeds {42, 1, 2}.

| K | α | ImageNet | V2 | Sketch | A | R | **OOD avg** |
|---:|---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 1 | 0.9 | 72.51 ± 0.09 | 65.49 ± 0.25 | 52.91 ± 0.11 | 54.88 ± 0.17 | 81.16 ± 0.18 | **63.61 ± 0.07** |
| 2 | 0.9 | 73.98 ± 0.31 | 66.92 ± 0.18 | 54.36 ± 0.12 | 55.80 ± 0.21 | 81.97 ± 0.12 | **64.76 ± 0.14** |
| 4 | 0.7 | 75.99 ± 0.13 | 67.78 ± 0.26 | 57.16 ± 0.13 | 56.29 ± 0.72 | 83.01 ± 0.42 | **66.06 ± 0.23** |
| 8 | 0.7 | 77.97 ± 0.21 | 69.96 ± 0.25 | 58.72 ± 0.25 | 58.58 ± 0.67 | 84.08 ± 0.31 | **67.83 ± 0.12** |
| 16 | 0.7 | 78.95 ± 0.04 | 71.05 ± 0.20 | 59.75 ± 0.11 | 58.72 ± 0.40 | 84.84 ± 0.06 | **68.59 ± 0.09** |

### Δ vs FSA without mixing (from fewshot_table.md)

| K | FSA ID | FSA+Mix ID | Δ ID | FSA OOD (K=16 only) | FSA+Mix OOD | Δ OOD |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 56.33 ± 0.04 | **72.51 ± 0.09** | **+16.18** | — | **63.61** | — |
| 2 | 64.69 ± 0.47 | **73.98 ± 0.31** | **+9.29** | — | **64.76** | — |
| 4 | 70.56 ± 0.21 | **75.99 ± 0.13** | **+5.43** | — | **66.06** | — |
| 8 | 74.20 ± 0.19 | **77.97 ± 0.21** | **+3.77** | — | **67.83** | — |
| 16 | 76.50 ± 0.12 | **78.97 ± 0.04** | **+2.45** | 63.01 ± 0.26 | **68.59 ± 0.09** | **+5.58** |

### Key observation

K=1 + CLIP-ZS-Mix achieves **OOD avg = 63.61**, matching K=16 FSA without mixing
(OOD avg = 63.01). One labelled example per class + CLIP zero-shot prior ≈
sixteen labelled examples without it.

---

## Full 11-dataset × 5-shot few-shot table (DINOv3-B + CLIP text)

3-seed mean over seeds {42, 1, 2}. α_clip schedule:
- K ∈ {1, 2}: α_clip = 0.9 for ImageNet, 0.7 for the other 10 datasets.
- K = 4: α_clip = 0.7 everywhere.
- K ∈ {8, 16}: α_clip = 0.7 for ImageNet, 0.5 for the other 10 datasets.

α_fsa = 0 (pure MT branch). t_end reuses the value tuned per checkpoint on
held-out val. Mixing performed in logit space.

| dataset | K=1 | K=2 | K=4 | K=8 | K=16 | mean |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| OxfordPets          | 87.69 | 89.07 | 92.90 | 94.33 | 94.92 | 91.78 |
| Food101             | 80.97 | 84.60 | 86.58 | 86.22 | 87.84 | 85.24 |
| StanfordCars        | 74.24 | 81.49 | 85.47 | 90.20 | 92.66 | 84.81 |
| FGVCAircraft        | 37.51 | 44.54 | 48.64 | 61.98 | 70.10 | 52.55 |
| UCF101              | 71.29 | 76.38 | 80.35 | 83.65 | 85.95 | 79.52 |
| DescribableTextures | 54.45 | 60.82 | 68.50 | 72.42 | 77.17 | 66.67 |
| EuroSAT             | 65.44 | 70.29 | 80.38 | 87.01 | 91.07 | 78.84 |
| Caltech101          | 95.02 | 96.13 | 97.43 | 97.67 | 97.98 | 96.85 |
| OxfordFlowers       | 97.31 | 98.51 | 99.09 | 99.92 | 99.89 | 98.95 |
| SUN397              | 66.21 | 70.91 | 74.25 | 75.51 | 77.24 | 72.82 |
| ImageNet            | 72.51 | 73.98 | 75.99 | 77.97 | 78.97 | 75.88 |
| **mean (11 ds)** | **72.96** | **76.97** | **80.87** | **84.26** | **86.71** | **80.35** |

### Gain over FSA alone (no CLIP-ZS mix)

| K | FSA alone (mean-11, from `fewshot_table.md`) | FSA + CLIP-ZS Mix (mean-11) | Δ |
|:---:|:---:|:---:|:---:|
| 1 | 64.30 | **72.96** | **+8.66** |
| 2 | 72.25 | **76.97** | **+4.72** |
| 4 | 79.14 | **80.87** | **+1.73** |
| 8 | 83.00 | **84.26** | **+1.26** |
| 16 | 85.85 | **86.71** | **+0.86** |

Δ monotonically decreases with K: CLIP-ZS prior contributes most when FSA is
data-starved (+8.66 pp at K=1), and remains consistently positive even at K=16
(+0.86 pp). Every entry improves; no regressions.

## OP-only ablation on ImageNet

Breakdown of the three FSA inference branches across K. All numbers are seed 42,
ImageNet full val. "K = 0" is pure CLIP-B/16 zero-shot with 7-prompt ensemble
(no few-shot supervision at all).

| K | OP-only (ZS branch) | FSA MT (full flow, α_fsa=0) | FSA + CLIP-ZS-Mix |
|:---:|:---:|:---:|:---:|
| 0 | — | — | **68.98** (pure CLIP ZS) |
| 1 | 53.05 | 56.33 | 72.51 |
| 2 | 60.27 | 64.69 | 73.98 |
| 4 | 64.16 | 70.56 | 75.99 |
| 8 | 67.13 | 74.20 | 77.97 |
| 16 | 68.25 | 76.50 | 78.97 |

### Reading the contributions

| K | Δ (flow − OP) | Δ (mix − flow) |
|:---:|:---:|:---:|
|  1 | +3.28 | **+16.18** |
|  2 | +4.42 | +9.29 |
|  4 | +6.40 | +5.43 |
|  8 | +7.07 | +3.77 |
| 16 | +8.25 | +2.47 |

The two additions are complementary across shots:

- **Flow adapter** (OP → MT) gains grow monotonically with K (+3.3 → +8.3).
  The flow learns an increasingly rich nonlinear alignment as more support is
  available.
- **CLIP-ZS mixing** (MT → MT + mix) gains shrink monotonically with K
  (+16.2 → +2.5). The pretrained zero-shot prior matters most when FSA is
  data-starved.

Notably, **OP-only at K=16 (68.25%) barely exceeds pure CLIP ZS at K=0 (68.98%)**
— demonstrating that without the flow, a linear cross-modal alignment fit on
16k support images doesn't outperform CLIP's 400M-pair pretraining. The flow
contribution accounts for the +8.25 pp to reach 76.50%.
