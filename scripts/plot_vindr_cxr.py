#!/usr/bin/env python
"""VinDr-CXR figure (Option 1): x-axis = number of labeled training images.

FSA + RAD-DINO + Qwen3 and RAD-DINO few-shot linear probe are curves over
increasing image budgets. Full-train linear-probe ceilings (DINOv2, BiomedCLIP,
CheXzero, MRM, RAD-DINO) are single points at N=1500.

All values are macro-AUPRC (%).
"""
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Shared image-budget x values for the few-shot sweeps
#   K=4: 28, K=8: 56, K=16: 112, K=32: 224, K=128: 488
# (K=64 / N=338 omitted from display to keep x-axis uncluttered on log scale.)
N_IMG = np.array([28, 56, 112, 224, 488])

fs  = np.array([26.72, 27.60, 33.20, 38.51, 42.62])  # RAD-DINO few-shot linear probe
fsa = np.array([33.1,  41.0,  47.65, 51.28, 53.61])  # FSA + RAD-DINO + Qwen3

N_FULL = 1500
RAD_DINO_FULL = 52.80
FULL_BASELINES = [
    # (label, AUPRC, color, marker)
    ("DINOv2 (full train)",     31.6,  "#7f7f7f", "D"),
    ("BiomedCLIP (full train)", 35.9,  "#2ca02c", "s"),
    ("CheXzero (full train)",   40.0,  "#ff7f0e", "v"),
    ("MRM (full train)",        51.3,  "#9467bd", "^"),
    ("RAD-DINO (full train)",   52.80, "#c0392b", "*"),
]

FSA_COLOR     = "#2b7bba"
RADDINO_COLOR = "#c0392b"

fig, ax = plt.subplots(figsize=(8.5, 4.6))

# --- Few-shot curves ---
ax.plot(N_IMG, fs, marker="s", linewidth=1.8, markersize=7,
        color=RADDINO_COLOR, label="RAD-DINO (few-shot)", zorder=2)

ax.plot(N_IMG, fsa, marker="o", linewidth=2.2, markersize=8,
        color=FSA_COLOR, label="FSA + RAD-DINO + Qwen3 (few-shot, ours)", zorder=3)

# Annotate FSA values (above markers)
for n, v in zip(N_IMG, fsa):
    ax.annotate(f"{v:.1f}", (n, v), textcoords="offset points",
                xytext=(0, 8), ha="center", fontsize=8.5,
                color=FSA_COLOR, fontweight="bold")

# Annotate few-shot linear probe values (below markers)
for n, v in zip(N_IMG, fs):
    ax.annotate(f"{v:.1f}", (n, v), textcoords="offset points",
                xytext=(0, -13), ha="center", fontsize=8.5,
                color=RADDINO_COLOR)

# --- Full-train ceilings as single points at x=1500 ---
for label, val, color, marker in FULL_BASELINES:
    ax.scatter([N_FULL], [val], marker=marker, s=110, color=color,
               edgecolor="black", linewidth=0.6, zorder=4,
               label=f"{label} ({val:.2f})")
# Annotate RAD-DINO full-train value next to its marker (the key reference)
ax.annotate(f"{RAD_DINO_FULL:.2f}",
            (N_FULL, RAD_DINO_FULL),
            textcoords="offset points", xytext=(10, 2),
            ha="left", va="center", fontsize=9,
            color=RADDINO_COLOR, fontweight="bold")

# Reference vertical guide at N_FULL
ax.axvline(N_FULL, color="black", linestyle=":", linewidth=0.8, alpha=0.35, zorder=1)
ax.text(N_FULL * 1.02, 24, f"N={N_FULL}", fontsize=8.5, color="black",
        alpha=0.6, ha="left", va="bottom")

# --- Shaded band showing where FSA exceeds the RAD-DINO full-train ceiling ---
cross_mask = fsa >= RAD_DINO_FULL
if cross_mask.any():
    n_cross = N_IMG[cross_mask].min()
    ax.axvspan(n_cross, N_FULL * 1.25, color=FSA_COLOR, alpha=0.06, zorder=0)

# --- Axis formatting ---
ax.set_xscale("log")
xticks_major = list(N_IMG) + [N_FULL]
ax.set_xticks(xticks_major)
ax.set_xticklabels([str(n) for n in xticks_major], fontsize=9)

ax.set_xlabel("Number of labeled training images", fontsize=10)
ax.set_ylabel("Macro-AUPRC (%)", fontsize=10)
ax.set_ylim(22, 60)
ax.set_xlim(22, N_FULL * 1.25)
ax.grid(alpha=0.3, linewidth=0.5, which="major")

# Legend outside (reversed: strongest on top)
handles, labels = ax.get_legend_handles_labels()
ax.legend(handles[::-1], labels[::-1],
          loc="center left", bbox_to_anchor=(1.02, 0.5),
          frameon=True, fontsize=8.5, framealpha=0.95, edgecolor="gray",
          borderaxespad=0.0)

plt.tight_layout()
outdir = Path("/efs/user_folders/dnshalam/work/FewShotAlignment/results/figs")
outdir.mkdir(parents=True, exist_ok=True)
for ext in ("pdf", "png"):
    plt.savefig(outdir / f"vindr_cxr_fsa.{ext}", dpi=200, bbox_inches="tight")
print("Saved:", outdir / "vindr_cxr_fsa.pdf", outdir / "vindr_cxr_fsa.png")
