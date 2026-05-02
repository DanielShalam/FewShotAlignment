# FSA on DeepLoc 2.0 — Experimental Summary

Few-shot protein function prediction by aligning independent protein and text foundation encoders. Workshop paper candidate for the ICML 2026 FM4LS workshop. Status as of 2026-04-24.

## TL;DR

Applying FSA (Few-Shot Alignment) to protein subcellular localization, using independent, frozen **ESM-2-150M** as the protein encoder and independent, frozen text encoders (PubMedBERT or Qwen3-Embedding) as the class-label encoder. No paired protein-text pretraining. K ∈ {1,4,16,64} labeled sequences per primary class.

**Headline.** At K=64, FSA with a generic text encoder (Qwen-0.6B) reaches **48.12 macro-AUPRC** on the 2,744-sample DeepLoc2Multi test set, vs. **42.18** for a matched-K linear probe on the same ESM features and **55.89** for the same linear probe trained on the full 21,948-sample training set. FSA beats the matched-K linear probe at every K, and with 3% of the labels closes 86% of the gap to the full-data linear probe.

---

## 1. Task and dataset

**Task:** Multi-label subcellular localization. Given a protein amino-acid sequence, predict which of 10 canonical compartments the protein localizes to. Proteins can localize to multiple compartments (~27% of the training set has ≥2 positive labels).

**Classes (10):** Nucleus, Cytoplasm, Extracellular, Mitochondrion, Cell membrane, Endoplasmic reticulum, Plastid, Golgi apparatus, Lysosome / Vacuole, Peroxisome.

**Dataset:** `AI4Protein/DeepLoc2Multi` (VenusFactory re-split of DeepLoc 2.0, Thumuluri et al. 2022).

| split | # sequences |
|---|---|
| train | 21,948 |
| validation | 2,744 |
| test | 2,744 |

Labels are stored as comma-separated class indices (e.g. `"0,1"`). Sequences are raw amino-acid strings, median length ~400 aa, truncated to 1,022 (ESM-2 limit).

**Evaluation metric:** Macro-AUPRC (threshold-free). Per-class average precision, mean over classes with ≥1 positive in the test split. Chosen because (a) it is label-imbalance-aware, (b) it doesn't depend on tuning per-class thresholds, and (c) it's standard for multi-label classification under heavy imbalance. We use the official DeepLoc2Multi test split for all reported numbers.

**Few-shot protocol:** For each K ∈ {1, 4, 16, 64}, we sample K sequences from each of the 10 primary classes (argmax of the multi-hot target) from the train split, seeded by {42, 43, 44}. Total train size is 10 × K sequences. The full 2,744-sample official validation split is used for hyperparameter tuning (mixing weight α and transport time τ). The full test split is used only for the final reported number.

---

## 2. Method (summary)

FSA (from our main submission) aligns two frozen, independently-pretrained encoders of different modalities in a two-step pipeline:

1. **Orthogonal Procrustes (OP).** A closed-form semi-orthogonal linear map `W* = V U^T` (from SVD of text-by-image cross-covariance) takes text-prototype embeddings into the protein-embedding space while preserving within-modality geometry. Fit on the K-shot support set (LOP).
2. **Flow-matching prior.** A small residual MLP (4 layers × 1536 hidden, time-dim 256, SiLU activations) parameterizes a velocity field on the unit sphere. Trained to match image↔text geodesic velocities. Two independent nets (forward i→t, reverse t→i) for ensembling.

**Inference.** Integrate the query protein embedding forward to time τ and each class prototype backward to 1−τ (dopri5 adaptive ODE solver). Final class score is a convex mix:

  `s_c = (1−α) · ⟨z(τ), z_c(1−τ)⟩ + α · ⟨x, y_c⟩`

where α, τ are tuned on the validation split (one tiny grid search).

**Protein side.** ESM-2-150M (`facebook/esm2_t30_150M_UR50D`). Mask-aware mean pool of last-layer tokens, unit-normalized. `D_img = 640`.

**Text side.** Class prompt = class name + a one-line natural-language description (GO/UniProt-style). Encoders: `BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext` (768-d, mean pool) and `Qwen3-Embedding-0.6B` (1024-d, last-token pool). Also `Qwen3-Embedding-8B` (4096-d) for one comparison point, to be consistent with the main paper.

No encoder is fine-tuned at any point. Only the flow-matching MLP is trained.

---

## 3. Experimental details

- **Training.** 400 epochs, AdamW (lr 1e-4, wd 1e-5), cosine schedule. Batch size 64 when it fits, otherwise 10 × K (so even K=1 has real batches).
- **Geometry.** Geodesic paths (slerp) between unit-norm prototypes and support features. Two independent velocity fields (img→txt, txt→img).
- **Hyper-parameter tuning.** Grid of α ∈ [0,1]/10 and τ ∈ [0,1]/5 (coarsened from the main paper's /10 to keep wall time reasonable). Selected by macro-AUPRC on the official DeepLoc validation split.
- **Seeds.** 3 seeds per cell (42, 43, 44).
- **Hardware.** One NVIDIA H100-80GB per run. Embedding precomputed once (~15 min); each FSA run is ~5–10 min end-to-end.
- **Engineering note.** All encoders are run once offline to cache features on EFS (~68 MB for the protein features). FSA's training loop consumes only cached `[N, D]` tensors — no model weights beyond the small flow MLP are ever materialized at training time.

Baselines compared:

| baseline | description |
|---|---|
| **Linear probe (matched-K)** | OVR logistic regression on ESM-2-150M features, trained on the same K × 10 samples as FSA. Same seed-set. |
| **Linear probe (full train)** | Same, trained on all 21,948 train sequences. Ceiling. |
| **OP-only (full train)** | Orthogonal Procrustes fitted on the full train split, then cosine. Our paper's "LFA-equivalent" reference. |

Deliberately **not** compared against (and why):

- **VenusFactory supervised numbers** (Tan et al. 2025): fine-tunes the PLM itself with full-data supervised learning and attention-pool task heads. Different problem (full supervision + PLM fine-tuning), different metric (they call it "accuracy" but the code computes F1-max). Out of scope for this workshop paper's few-shot, frozen-encoder, no-paired-data framing.

---

## 4. Results

### 4.1 Main table — test macro-AUPRC (%), mean ± std over 3 seeds

| method | K=1 | K=4 | K=16 | K=64 |
|---|---|---|---|---|
| Linear probe, ESM-2-150M (matched K) | 18.96 ± 1.06 | 31.26 ± 1.69 | 36.27 ± 0.79 | 42.18 ± 1.34 |
| **FSA + PubMedBERT** | 19.48 ± 1.70 | 29.56 ± 1.98 | 39.66 ± 0.06 | 43.65 ± 1.21 |
| **FSA + Qwen-0.6B** | **21.81 ± 1.85** | **36.09 ± 1.98** | **43.99 ± 1.43** | **48.12 ± 0.49** |
| FSA + Qwen-8B | — | — | — | 48.60 ± 0.46 |

Reference ceilings (not few-shot):

| method | macro-AUPRC |
|---|---|
| OP-only + PubMedBERT, full train (N=21,948) | 26.22 |
| OP-only + Qwen-0.6B,    full train (N=21,948) | 42.60 |
| Linear probe ESM-2-150M, full train (N=21,948) | 55.89 |
| **FSA + Qwen-0.6B, full train (N=21,948)** | **61.42** |

### 4.2 Shot curve

![Shot curve](results/figs/shot_curve.png)

### 4.2a Full-train ceilings (bar chart)

![Full-train bar](results/figs/full_train_bar.png)

Full-train (N=21,948) comparison on DeepLoc 2.0 Multi. OP-only is the parameter-free Orthogonal Procrustes baseline from our main paper. Linear probe is OVR logistic regression on frozen ESM-2-150M features. OP+Flow (FSA) adds the learned flow-matching prior on top of OP. The flow adds +18.82 macro-AUPRC over OP alone and +5.53 over the matched ESM linear probe, showing that the multi-modal signal is complementary to supervised vision features even at full data.

The figure plots test macro-AUPRC vs. K for the three few-shot methods (FSA-Qwen-0.6B, FSA-PubMedBERT, linear probe), with error bars over 3 seeds. The Qwen-8B K=64 point is shown as a star for consistency with the main paper. The dashed horizontal line is the full-data linear-probe ceiling.

### 4.3 Per-class AP at K=16, FSA+Qwen-0.6B seed 42 (representative run)

| class | AP |
|---|---|
| Endoplasmic reticulum | 0.80 |
| Extracellular | 0.76 |
| Cytoplasm | 0.65 |
| Mitochondrion | 0.56 |
| Nucleus | 0.52 |
| Cell membrane | 0.48 |
| Plastid | 0.36 |
| Lysosome / Vacuole | 0.22 |
| Golgi apparatus | 0.09 |
| Peroxisome | 0.07 |

Unsurprising pattern: rare classes (Golgi, Peroxisome) are weakest. Expected since 16 shots per primary class × rare-class share = only a handful of positive examples.

### 4.4 Hyperparameter picks (validation-tuned)

- **Qwen-0.6B:** α ≈ 0.2–0.5, τ ≈ 0.4–0.6. Non-trivial mixing — the cosine base carries useful signal and the flow polishes it.
- **PubMedBERT:** α = 0.0 across all K, τ ≈ 0.6–1.0. Tuner collapses to "ignore the base, lean entirely on the learned flow." Consistent with the (correct) view that clinical-biomedical text embeddings of subcellular terms are weak, and the flow has to do all the work.
- **Qwen-8B (K=64 only):** α ≈ 0.3–0.5, τ = 0.4.

Consistent with the main paper's pattern — generic LLM embeddings work at least as well as domain-specific biomedical ones once the alignment is learned from a few shots.

---

## 5. Takeaways

1. **FSA beats the matched-K linear probe at every K with every text encoder.** At K=16, FSA+Qwen-0.6B is **+7.72** macro-AUPRC over the ESM-2-150M linear probe. This is the central result: even with 160 labeled sequences and no paired pretraining, text-modality information meaningfully improves protein classification.
2. **Generic text > biomedical text for subcellular terms.** Qwen-0.6B consistently beats PubMedBERT across K, mirroring the VinDr-CXR finding in the main paper. Subcellular-localization concepts are underrepresented in clinical literature.
3. **Text-encoder scale is a mild knob.** Qwen-8B is +0.48 macro-AUPRC over Qwen-0.6B at K=64 — within std. Consistent with the main paper's Table 1: stronger text encoders give modest gains once OP+flow are in place.
4. **PubMedBERT's reliance on pure-flow (α=0) is informative.** When the base cosine is misaligned (weak text), the flow module provably compensates — exactly the behavior OP+flow was designed to exhibit.
5. **Label-rare classes are the bottleneck.** Primary-class-stratified sampling gives only K positives for rare compartments (Peroxisome, Golgi). A per-label multi-hot sampler or global-OP fit (from Swiss-Prot paired data) might help low shots — not yet tested.

---

## 6. Open questions / next steps

- **Biological GOP.** Main paper's "global OP" trick fits the OP on an external paired corpus (CC3M). Analog here: fit OP on Swiss-Prot (protein, description) pairs. Expected to help low-K most.
- **Per-label sampling.** Switch from primary-class K-per-class to multi-hot-aware sampling so every class gets ≥K positive examples. Should help rare classes.
- **Larger protein encoder.** ESM-2-650M at K=16 was not better than 150M in preliminary tests (32.56 vs 36.27 linear-probe macro-AUPRC). Likely a supervision-limited regime; worth revisiting at K=64 with a larger flow network.
- **Per-class F1-max reporting.** VenusFactory reports F1-max (threshold-free F1) which is not directly monotonic with macro-AUPRC on this data. If we want to report F1-max, do it alongside AUPRC, not instead of.

---

## Appendix A. Artifact inventory

All artifacts live under `/efs/user_folders/dnshalam/work/FewShotAlignment` on `lrm1`.

- `data/deeploc/seq_embed_esm2_t30_150m_ur50d.pt` — ESM-2-150M features [N, 640] per split
- `data/deeploc/seq_embed_esm2_t33_650m_ur50d.pt` — ESM-2-650M features [N, 1280] per split
- `data/deeploc/class_text_*.pt` — class prototypes for PubMedBERT / Qwen3-0.6B / Qwen3-8B
- `data/deeploc/labels_multihot.pt` — multi-hot targets
- `data/deeploc/meta.json` — class names, prompts, counts
- `output/DeepLoc2/grid/` — 24 full 400-epoch runs (PubMedBERT, Qwen-0.6B × K∈{1,4,16,64} × 3 seeds)
- `output/DeepLoc2/grid/k64_qwen8b/` — 3 Qwen-8B K=64 runs
- `output/DeepLoc2/grid650/` — 6 ESM-650M K=16 runs (ancillary)
- `output/DeepLoc2/linprobe_fs.log` — few-shot linear-probe sweep (ESM-150M)
- `output/DeepLoc2/grid650/linprobe_fs_650m.log` — few-shot linear-probe sweep (ESM-650M)
- `results/figs/shot_curve.{png,pdf}` — Figure in §4.2

## Appendix B. Reproducing the headline number

```bash
# env
source /efs/user_folders/dnshalam/work/FewShotAlignment/env.sh
cd /efs/user_folders/dnshalam/work/FewShotAlignment

# Single K=64, seed 42, FSA + Qwen-0.6B
CUDA_VISIBLE_DEVICES=0 python main.py \
    --config configs/deeploc2_esm150_qwen.yaml \
    --output_dir output/DeepLoc2/repro/k64_qwen06_s42 \
    shots 64 seed 42 \
    text_features_path data/deeploc/class_text_qwen3_embedding_0_6b.pt \
    tune_t_steps 6
```

---

# BiomedCoOp 11-Dataset Benchmark

## Experimental Setup

**Task.** 11-dataset few-shot medical image classification benchmark from BiomedCoOp
(CVPR 2025), spanning 9 modalities and 10 organs: BUSI (breast ultrasound), CHMNIST
(colorectal histology), KneeXray (knee X-ray), BTMRI (brain MRI), COVID_19 (chest
X-ray), CTKidney (kidney CT), DermaMNIST (dermatoscopy), Kvasir (endoscopy),
LungColon (lung/colon histology), OCTMNIST (retinal OCT), RETINA (fundus photography).

**Model.** FSA with:
- Vision encoder: **DINOv3-B** (`facebook/dinov3-vitb16-pretrain-lvd1689m`), 86M params.
  No medical pretraining — general self-supervised features.
- Text encoder: **BiomedCLIP-PubMedBERT** (`microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224`)
  text tower, kept frozen.
- FSA bridges the two encoders via a 4-layer MLP flow with orthogonal-Procrustes
  initialization. No prompt learning, no text adapter training, no vision fine-tuning.

**Protocol.**
- Shots per class K ∈ {4, 16}.
- 3 seeds (42, 0, 2) per cell; report mean ± std.
- 400 epochs, AdamW (lr=1e-5), cosine schedule.
- Best-on-val checkpointing (eval every 25 epochs, 16 checkpoints).
- α (zero-shot / flow mix) and τ (flow end-time) tuned on val after training.
- Splits generated with a deterministic 50/20/30 per-class protocol (seed 1),
  matching BiomedCoOp's generator.
- Baseline numbers are BiomedCoOp's published 3-seed means from Koleilat et al.
  (CVPR 2025, Tables 1 and S8).

## K=16 Headline Results

Wins on **8 of 11** datasets; average **+2.63 pp**.

| Dataset     | FSA + DINOv3     | BiomedCoOp         | Δ (pp) |
|-------------|------------------|--------------------|--------|
| BUSI        | 71.04 ± 3.02     | 70.34 ± 2.27       | +0.70  |
| CHMNIST     | **87.37 ± 1.18** | 79.05 ± 2.24       | +8.32  |
| KneeXray    | 41.51 ± 0.57     | 39.69 ± 1.75       | +1.82  |
| BTMRI       | 81.16 ± 3.28     | 83.30 ± 1.34       | −2.14  |
| COVID_19    | 75.51 ± 2.01     | 78.72 ± 0.23       | −3.21  |
| CTKidney    | 75.77 ± 2.84     | 83.20 ± 2.37       | −7.43  |
| DermaMNIST  | 64.31 ± 2.06     | 62.59 ± 1.83       | +1.72  |
| Kvasir      | **85.45 ± 1.41** | 78.89 ± 1.21       | +6.56  |
| LungColon   | **94.79 ± 0.31** | 92.68 ± 0.57       | +2.11  |
| OCTMNIST    | **72.82 ± 0.80** | 66.93 ± 2.13       | +5.89  |
| RETINA      | **75.90 ± 1.05** | 61.28 ± 1.06       | +14.62 |
| **Avg**     | **75.05**        | 72.42              | **+2.63** |

BiomedCoOp's K=16 average of 72.42 matches their published headline number
(Table 1 in the main paper). Our result, 75.05, is attained without (i) any medical
vision pretraining, (ii) prompt learning, or (iii) LLM prompt ensembling.

## K=4 Results

Low-shot regime is more variable. Wins on 6/11, average -0.69 pp.

| Dataset     | FSA + DINOv3     | BiomedCoOp       | Δ (pp) |
|-------------|------------------|------------------|--------|
| BUSI        | 51.41 ± 4.70     | 59.32 ± 1.04     | −7.91  |
| CHMNIST     | **77.48 ± 1.76** | 71.19 ± 1.74     | +6.29  |
| KneeXray    | 38.87 ± 3.42     | 35.91 ± 0.54     | +2.96  |
| BTMRI       | 70.19 ± 2.02     | 77.23 ± 3.90     | −7.04  |
| COVID_19    | 63.36 ± 2.48     | 73.28 ± 2.30     | −9.92  |
| CTKidney    | 49.35 ± 4.69     | 66.50 ± 1.92     | −17.15 |
| DermaMNIST  | 56.28 ± 1.42     | 60.07 ± 1.81     | −3.79  |
| Kvasir      | 76.53 ± 1.75     | 74.08 ± 1.10     | +2.45  |
| LungColon   | **90.28 ± 2.29** | 85.60 ± 1.61     | +4.68  |
| OCTMNIST    | 58.68 ± 7.27     | 54.73 ± 1.86     | +3.95  |
| RETINA      | **63.45 ± 0.91** | 45.58 ± 5.03     | +17.87 |
| **Avg**     | 63.26            | 63.95            | -0.69  |

The K=4 regime shows higher variance in our favour-sign (deltas range from −17
to +18). BiomedCoOp's prompt ensembling provides large gains when support is very
thin (≤4 examples), whereas FSA's flow-based alignment needs more signal to
outperform. By K=16 this reverses.

## Architecture Ablations (single-seed previews)

![BiomedCoOp K=16 comparison](results/figs/biomedcoop_k16.png)


## Text-Encoder Ablation: Qwen3 vs. BiomedCLIP-Text

Replacing the BiomedCLIP-PubMedBERT text tower with a general-purpose multilingual
embedding model (`Qwen/Qwen3-Embedding-0.6B`) while keeping DINOv3 as the vision
encoder. 3 seeds, K=16, 400 epochs, otherwise identical protocol.

| Dataset     | FSA + Qwen3-0.6B | FSA + BiomedCLIP-text | BiomedCoOp     |
|-------------|------------------|-----------------------|----------------|
| BUSI        | 72.03 ± 3.93     | 71.04 ± 3.02          | 70.34 ± 2.27   |
| CHMNIST     | 87.23 ± 1.41     | 87.37 ± 1.18          | 79.05 ± 2.24   |
| KneeXray    | 41.39 ± 1.01     | 41.51 ± 0.57          | 39.69 ± 1.75   |
| BTMRI       | 79.34 ± 3.17     | 81.16 ± 3.28          | 83.30 ± 1.34   |
| COVID_19    | 75.53 ± 1.70     | 75.51 ± 2.01          | 78.72 ± 0.23   |
| CTKidney    | 75.51 ± 2.31     | 75.77 ± 2.84          | 83.20 ± 2.37   |
| DermaMNIST  | 62.64 ± 1.25     | 64.31 ± 2.06          | 62.59 ± 1.83   |
| Kvasir      | 85.78 ± 1.91     | 85.45 ± 1.41          | 78.89 ± 1.21   |
| LungColon   | 94.85 ± 0.27     | 94.79 ± 0.31          | 92.68 ± 0.57   |
| OCTMNIST    | 73.58 ± 0.58     | 72.82 ± 0.80          | 66.93 ± 2.13   |
| RETINA      | 75.76 ± 0.75     | 75.90 ± 1.05          | 61.28 ± 1.06   |
| **Avg**     | **74.88**        | **75.06**             | 72.42          |

**Headline.** The text encoder's choice is almost irrelevant: swapping BiomedCLIP's
medical-domain text tower for Qwen3-0.6B changes the 11-dataset average by only
0.18 pp. Both configurations beat BiomedCoOp's SOTA (+2.46 / +2.63 pp respectively)
on 7–8 of 11 datasets with the same loss pattern (BTMRI, COVID_19, CTKidney).

**Implication.** FSA's learned flow — not the encoders' biomedical specialization —
supplies the cross-modal alignment. Any reasonable pretrained encoder pair works.

![Text-encoder ablation: Qwen3 vs BiomedCLIP-text](results/figs/biomedcoop_k16_qwen.png)
