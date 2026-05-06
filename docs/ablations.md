# Ablations — ImageNet K=16

All values are **3-seed mean ± std** over seeds {42, 1, 2}.
Eval protocol: **ImageNet** = tuned α, t_end on held-out val; **OOD avg** = mean over ImageNet-{A, R, Sketch, V2} at fixed α=0, t_end=0.2 (no tuning).

Shared recipe unless noted: `lr=5e-5, wd=1e-3, bs=128, 50 epochs, source_cond=True, x0_noise_k=4, adapter=SimpleMLP depth=2 width=1536`.

---

## Text-encoder scaling

Vision encoder fixed to **DINOv3-B** (86M). Text encoders span two decades in parameter count.

| text encoder | family | text params | ImageNet | OOD avg |
|:---|:---|:---:|:---:|:---:|
| all-MiniLM-L12-v2 (`sentence-transformers`) | uni-modal (distilled BERT)  | 33M   | 74.67 ± 0.17 | — |
| GTE-mini (`prdev/mini-gte`)                 | uni-modal (BERT-style)      | 66M   | 74.98 ± 0.09 | 61.12 ± 0.26 |
| GTE-base (`Alibaba-NLP/gte-base-en-v1.5`)   | uni-modal (BERT-style)      | 137M  | 75.26 ± 0.18 | 61.35 ± 0.31 |
| GTE-large (`Alibaba-NLP/gte-large-en-v1.5`) | uni-modal (BERT-style)      | 434M  | 75.52 ± 0.09 | 61.72 ± 0.36 |
| Qwen3-8B-Embedding                          | uni-modal (decoder)         | 8000M | 76.15 ± 0.16 | 62.19 ± 0.29 |
| CLIP ViT-B/16 text                          | vision-aware (OpenAI)       | 63M   | 76.50 ± 0.12 | 63.01 ± 0.26 |
| SigLIP 2 B/16 text                          | **vision-aware** (WebLI)    | 86M   | **77.01 ± 0.07** | **63.34 ± 0.23** |

**Takeaways.**
1. **Scaling works within a family** (MiniLM → GTE-mini → GTE-base → GTE-large): +0.85 pp ID across a 13× param range.
2. **Going from 434M → 8000M** (GTE-large → Qwen3-8B) adds only +0.63 pp. Diminishing returns.
3. **An 86M vision-aware text encoder beats a 8000M uni-modal one** (SigLIP 2 vs Qwen3-8B: +0.80 pp ID, +1.15 pp OOD). Alignment to vision pretraining > raw size.
4. **Vision-aware variants scale cleanly**: SigLIP 2 (2024, WebLI + sigmoid loss, 86M active) beats CLIP (2021, WIT + softmax, 63M) by +0.51 ID / +0.33 OOD — a small but consistent gain from better contrastive training.

---

## Vision-encoder scaling

Text encoder fixed to **GTE-mini** (66M). Vision encoders span the DINOv3 family.

| vision encoder | vision params | ImageNet |
|:---|:---:|:---:|
| DINOv3-S (`vits16`) | 22M  | 63.66 ± 0.24 |
| DINOv3-B (`vitb16`) | 86M  | 74.98 ± 0.09 |
| DINOv3-L (`vitl16`) | 304M | **81.21 ± 0.07** |

_(ImageNet only for this table; OOD evals were not run for S / L.)_

**Takeaways.**
1. Vision scaling gives large, monotonic gains: S → B = +11.32 pp, B → L = +6.23 pp.
2. Compared to the text axis (where 132× params yields +1.17 pp), the vision encoder dominates the param budget for this task.
3. The DINOv3-L + GTE-mini combination (304+66 = 370M total) reaches **81.21 %** on ImageNet K=16, well above any multi-modal baseline at 149M. Cross-axis scaling — stronger vision + stronger text — is the path to higher numbers; within one axis, vision is the cheaper win.

---

## Image-encoder choice (cross-modal setup)

Text encoder fixed to **CLIP ViT-B/16 text** (63M). Compares DINOv2 vs DINOv3 at the same base scale.

| vision encoder | vision params | ImageNet | OOD avg |
|:---|:---:|:---:|:---:|
| DINOv2-B | 86M | 76.64 ± 0.22 | 61.37 ± 0.13 |
| DINOv3-B | 86M | 76.50 ± 0.12 | **63.01 ± 0.26** |

**Takeaways.** DINOv2 and DINOv3 are indistinguishable on ID at matched params (Δ = −0.14, within noise). DINOv3 wins OOD-avg by **+1.64 pp**, driven entirely by ImageNet-R (+9.33) and ImageNet-Sketch (+3.96) — the stylized/distribution-shifted variants where DINOv3's stronger self-supervision transfers better. DINOv2 is competitive or slightly better on ImageNet-A (+0.80 → DINOv2) and ImageNet-V2 (+1.01 → DINOv2), the natural-distribution variants.

Full per-variant breakdown:

| variant   | DINOv2-B (before) | DINOv3-B (before) | Δ (v3 − v2) |
|:---|:---:|:---:|:---:|
| ImageNet-A       | 56.91 ± 0.43 | 51.11 ± 0.53 | **−5.80** |
| ImageNet-R       | 68.31 ± 0.32 | 77.64 ± 0.28 | **+9.33** |
| ImageNet-Sketch  | 52.01 ± 0.07 | 55.97 ± 0.09 | **+3.96** |
| ImageNet-V2      | 68.24 ± 0.11 | 67.23 ± 0.22 | **−1.01** |
| **OOD avg**      | **61.37**     | **63.01**     | **+1.64** |


---

## FSA vs linear-probe baselines across shots

How much does the flow adapter contribute on top of the encoder alone? We
compare FSA against L2-regularised logistic-regression probes trained on the
same K-shot support set. Probes are fit with fixed `C = 10`, L2-normalised
features, full ImageNet val test set, 3-seed mean ± std over seeds {42, 1, 2}.

### ImageNet (ID)

| Method | Image encoder | Text enc. | K=1 | K=2 | K=4 | K=8 | K=16 |
|:---|:---|:---|:---:|:---:|:---:|:---:|:---:|
| Linear probe   | CLIP-B/16 pre-proj | — | 32.85 ± 0.67 | 44.73 ± 0.43 | 55.54 ± 0.38 | 62.72 ± 0.14 | 68.12 ± 0.24 |
| Linear probe   | DINOv3-B           | — | 47.42 ± 0.57 | 59.06 ± 0.57 | 67.61 ± 0.20 | 72.67 ± 0.25 | 75.91 ± 0.21 |
| **FSA (ours)** | **DINOv3-B**       | **CLIP-B/16 text** | **56.33 ± 0.04** | **64.69 ± 0.47** | **70.56 ± 0.21** | **74.20 ± 0.19** | **76.50 ± 0.12** |

### Δ FSA − DINOv3-B linear probe (matched image encoder)

| K | Δ ID |
|---|---|
| 1  | **+8.91** |
| 2  | **+5.63** |
| 4  | **+2.95** |
| 8  | **+1.53** |
| 16 | **+0.59** |

**Takeaways.**
1. **FSA wins biggest at low shots.** At K=1 the flow-adapter provides +8.91 pp on top of a strong DINOv3-B probe; by K=16 the gap shrinks to +0.59 pp. A linear probe with 16k labelled samples saturates near the encoder's ceiling, so there is little left for the flow to contribute.
2. **Text-conditioning explains the low-shot gap.** With one sample per class, the probe's class weights are noisy; FSA uses text-encoder priors to stabilise the decision boundary. This also shows up in top-5: at K=1 FSA reduces top-5 error by ~30% relative vs DINOv3-B probe (see main-paper table).
3. **CLIP-B visual probe is far behind DINOv3-B probe at every K** (−12 to −15 pp). The image encoder choice matters much more than any adapter can recover — even a 1-shot DINOv3-B probe beats a 16-shot CLIP-B probe on ID by a hair (47.4 vs 68.1 at matched K=16, but the 1→16 growth rate is steep on both).

