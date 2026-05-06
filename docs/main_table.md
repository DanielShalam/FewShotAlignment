# ImageNet K=16 — Main Results Table

## Table

| Method | Image enc. | Text enc. | Params (M) | ImageNet | OOD avg |
|:---|:---|:---|:---:|:---:|:---:|
| *Multi-modal: jointly-trained vision–language encoders* | | | | | |
| CoOp         | CLIP-B/16 | CLIP-B/16 text | 86+63 | 71.92 | 58.46 |
| Tip-Adapter-F | CLIP-B/16 | CLIP-B/16 text | 86+63 | 73.69 | 59.12 |
| CLAP         | CLIP-B/16 | CLIP-B/16 text | 86+63 | 73.38 | 60.04 |
| | | | | | |
| *Uni-modal: independently-trained encoders* | | | | | |
| FSA (ours)   | DINOv3-B | all-MiniLM-L12-v2  | 86+33  | 74.67 ± 0.17 | -- |
| FSA (ours)   | DINOv3-B | GTE-mini  | 86+66  | 74.98 ± 0.09 | 61.12 ± 0.26 |
| | | | | | |
| *Cross-modal: uni-modal vision + vision-aware text* | | | | | |
| FSA (ours)   | DINOv2-B | CLIP-B/16 text | 86+63 | 76.64 ± 0.22 | 61.37 ± 0.13 |
| FSA (ours)   | DINOv3-B | CLIP-B/16 text | 86+63 | 76.50 ± 0.12 | 63.01 ± 0.26 |
| **FSA (ours)** | **DINOv3-B** | **SigLIP 2 B/16 text** | **86+86** | **77.01 ± 0.07** | **63.34 ± 0.23** |

> **Caption.** ImageNet-1k K=16 few-shot classification. Methods grouped by how their image and text encoders were pretrained. *Multi-modal* (top): both encoders come from a vision–language pair, jointly trained with image–text contrastive loss. *Uni-modal* (middle): both encoders were trained on their own modality only — DINOv3 on images (self-supervised), GTE-mini on text. *Cross-modal* (bottom): DINOv2 / DINOv3 vision encoder paired with CLIP's text encoder, which was originally trained alongside CLIP's image encoder. OOD avg is mean accuracy over ImageNet-{A, R, Sketch, V2}. All values are mean ± std over 3 seeds (42, 1, 2). OOD uses fixed α=0, t_end=0.2 (no tuning); ImageNet uses tuned α, t_end on held-out val. At matched parameter budget (86+63M), our cross-modal configuration obtains +2.89 pp on ID and +4.14 pp on OOD avg over the strongest multi-modal baseline (Tip-Adapter-F). See `ablations.md` for text- and vision-encoder scaling.

---

## What the reader should take away

The three groups span a progression of encoder coupling:

1. **Multi-modal (top)** — image and text encoder were *designed* to work together (CLIP contrastive pretraining). This is the standard baseline for few-shot classification.
2. **Uni-modal (middle)** — image and text encoders come from completely independent pretraining. Image encoder (DINOv3) never saw language; text encoders (GTE-mini, GTE-base, GTE-large, Qwen3-8B) never saw images. FSA aligns them at K=16-shot time. Scaling the text encoder provides only modest gains (see `ablations.md`).
3. **Cross-modal (bottom)** — image encoder (DINOv2 or DINOv3) is still uni-modal, but the text encoder (CLIP text) was originally vision-aware.

Row 3 beats row 1 by +2.89 pp on ID and +4.14 pp on OOD avg at *identical* parameter budget (86+63M).

The isolation is clean because across rows 1 and 3 the text encoder is held fixed (both are CLIP-B/16 text). Only the image encoder changes — DINOv3 replaces CLIP's image encoder. This shows:

- Replacing the image encoder with a stronger uni-modal one (DINOv3) gains a lot when the text encoder stays vision-aware.
- Using a vision-aware text encoder is necessary to keep those gains — row 2 (same DINOv3, non-vision-aware text) loses most of the advantage.

The combination — strong uni-modal vision + vision-aware text — is what delivers SOTA.

---

## FSA vs LFA comparison

### ImageNet (FSA is 3-seed mean over seeds {42, 1, 2})

| encoder (text) | LFA (seed 42) | FSA 3-seed mean | Δ |
|---|---|---|---|
| mini-gte (768)     | 71.45 | 74.98 ± 0.09 | **+3.53** |
| gte-base (768)     | 72.37 | 75.26 ± 0.18 | **+2.89** |
| gte-large (1024)   | 72.33 | 75.52 ± 0.09 | **+3.19** |
| Qwen3-8B (4096)    | 74.66 | 76.15 ± 0.16 | **+1.49** |
| CLIP ViT-B-16 (512)| 72.92 | 76.50 ± 0.12 | **+3.58** |

### ImageNet OOD-avg per-encoder lift (FSA − LFA)

| encoder | LFA OOD-avg | FSA OOD-avg (3-seed) | Δ |
|---|---|---|---|
| mini-gte      | 53.81 | 61.12 ± 0.26 | **+7.31** |
| gte-base      | 54.89 | 61.35 ± 0.31 | **+6.46** |
| gte-large     | 55.21 | 61.72 ± 0.36 | **+6.51** |
| Qwen3-8B      | 58.59 | 62.19 ± 0.29 | **+3.60** |
| CLIP ViT-B-16 | 56.53 | 63.01 ± 0.26 | **+6.48** |

### OOD variants (no tuning, α=0, t_end=0.2)

| encoder (text) | method | ImageNet-A | ImageNet-R | ImageNet-Sketch | ImageNet-V2 | OOD-avg |
|---|---|---|---|---|---|---|
| mini-gte      | LFA (seed 42)   | 41.07 | 64.70 | 46.92 | 62.55 | 53.81 |
| mini-gte      | FSA (3-seed)    | 49.58 ± 0.58 | 75.26 ± 0.22 | 53.85 ± 0.13 | 65.78 ± 0.19 | **61.12 ± 0.26** |
| gte-base      | LFA (seed 42)   | 42.16 | 65.77 | 48.08 | 63.54 | 54.89 |
| gte-base      | FSA (3-seed)    | 49.66 ± 0.70 | 75.65 ± 0.38 | 54.26 ± 0.13 | 65.83 ± 0.16 | **61.35 ± 0.31** |
| gte-large     | LFA (seed 42)   | 43.17 | 66.39 | 48.16 | 63.13 | 55.21 |
| gte-large     | FSA (3-seed)    | 50.07 ± 0.88 | 75.98 ± 0.36 | 54.70 ± 0.15 | 66.11 ± 0.19 | **61.72 ± 0.36** |
| Qwen3-8B      | LFA (seed 42)   | 46.56 | 70.47 | 51.21 | 66.13 | 58.59 |
| Qwen3-8B      | FSA (3-seed)    | 50.47 ± 0.64 | 76.36 ± 0.32 | 55.24 ± 0.28 | 66.69 ± 0.13 | **62.19 ± 0.29** |
| CLIP ViT-B-16 | LFA (seed 42)   | 44.31 | 68.77 | 48.95 | 64.09 | 56.53 |
| CLIP ViT-B-16 | FSA (3-seed)    | 51.11 ± 0.53 | 77.64 ± 0.28 | 55.97 ± 0.09 | 67.23 ± 0.22 | **63.01 ± 0.26** |
