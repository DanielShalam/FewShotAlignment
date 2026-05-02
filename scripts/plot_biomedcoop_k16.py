#!/usr/bin/env python
"""Paper-grade figure for the BiomedCoOp comparison at K=16."""
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Per-dataset K=16 numbers (mean, std)
data = [
    ("BUSI",       71.04, 3.02, 70.34, 2.27),
    ("CHMNIST",    87.37, 1.18, 79.05, 2.24),
    ("KneeXray",   41.51, 0.57, 39.69, 1.75),
    ("BTMRI",      81.16, 3.28, 83.30, 1.34),
    ("COVID_19",   75.51, 2.01, 78.72, 0.23),
    ("CTKidney",   75.77, 2.84, 83.20, 2.37),
    ("DermaMNIST", 64.31, 2.06, 62.59, 1.83),
    ("Kvasir",     85.45, 1.41, 78.89, 1.21),
    ("LungColon",  94.79, 0.31, 92.68, 0.57),
    ("OCTMNIST",   72.82, 0.80, 66.93, 2.13),
    ("RETINA",     75.90, 1.05, 61.28, 1.06),
]
names = [r[0] for r in data]
ours  = np.array([r[1] for r in data]); ours_std  = np.array([r[2] for r in data])
them  = np.array([r[3] for r in data]); them_std  = np.array([r[4] for r in data])
deltas = ours - them

# Append average column
names.append("Avg")
ours = np.append(ours, ours.mean());   ours_std = np.append(ours_std, ours_std.mean())
them = np.append(them, them.mean());   them_std = np.append(them_std, them_std.mean())
deltas = np.append(deltas, (ours[-1] - them[-1]))

BMC_COLOR  = "#6c6c6c"   # neutral gray for BiomedCoOp
WIN_COLOR  = "#2b7bba"   # blue for our wins
LOSS_COLOR = "#c0392b"   # red for our losses
AVG_ACCENT = "#111111"   # dark border for Avg

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.2),
                               gridspec_kw={"width_ratios": [2.3, 1.0]})

# --- Panel A: grouped bars ---
x = np.arange(len(names))
w = 0.38
ours_colors = [WIN_COLOR if d >= 0 else LOSS_COLOR for d in deltas]

b1 = ax1.bar(x - w/2, them, w, yerr=them_std, color=BMC_COLOR,
             edgecolor="black", linewidth=0.4, label="BiomedCoOp (SOTA)",
             error_kw={"lw": 0.6, "capsize": 2})
b2 = ax1.bar(x + w/2, ours, w, yerr=ours_std, color=ours_colors,
             edgecolor="black", linewidth=0.4, label="FSA + DINOv3 (ours)",
             error_kw={"lw": 0.6, "capsize": 2})

# Highlight Avg column with thicker border
b1[-1].set_edgecolor(AVG_ACCENT); b1[-1].set_linewidth(1.3)
b2[-1].set_edgecolor(AVG_ACCENT); b2[-1].set_linewidth(1.3)

ax1.set_xticks(x)
ax1.set_xticklabels(names, rotation=30, ha="right", fontsize=9)
ax1.set_ylabel("Accuracy (%)", fontsize=10)
ax1.set_title("Per-dataset accuracy, $K$=16 (3 seeds)", fontsize=11)
ax1.set_ylim(30, 100)
ax1.grid(axis="y", alpha=0.25, linewidth=0.5)
ax1.axvline(x[-1] - 0.5, color="gray", linestyle="--", linewidth=0.6, alpha=0.6)
# Legend with dummy color patches
from matplotlib.patches import Patch
legend_items = [
    Patch(facecolor=BMC_COLOR, edgecolor="black", label="BiomedCoOp (SOTA)"),
    Patch(facecolor=WIN_COLOR, edgecolor="black", label="FSA+DINOv3 — win"),
    Patch(facecolor=LOSS_COLOR, edgecolor="black", label="FSA+DINOv3 — loss"),
]
ax1.legend(handles=legend_items, loc="lower right", frameon=False, fontsize=8)

# --- Panel B: delta bars, sorted ---
order = np.argsort(deltas[:-1])[::-1]  # excl Avg, sort best → worst
names_s  = [names[i] for i in order]
deltas_s = deltas[order]
colors_s = [WIN_COLOR if d >= 0 else LOSS_COLOR for d in deltas_s]

y = np.arange(len(names_s))[::-1]
ax2.barh(y, deltas_s, color=colors_s, edgecolor="black", linewidth=0.4)
ax2.axvline(0, color="black", linewidth=0.8)
# Annotate with numeric delta
for yi, d in zip(y, deltas_s):
    off = 0.4 if d >= 0 else -0.4
    ax2.text(d + off, yi, f"{d:+.1f}", va="center",
             ha="left" if d >= 0 else "right", fontsize=8)
ax2.set_yticks(y); ax2.set_yticklabels(names_s, fontsize=9)
ax2.set_xlabel("Δ accuracy vs. BiomedCoOp (pp)", fontsize=10)
ax2.set_title("Gain/loss (sorted)", fontsize=11)
ax2.grid(axis="x", alpha=0.25, linewidth=0.5)
# Add avg delta as annotation
avg_delta = deltas[-1]
ax2.text(0.98, 0.02, f"Avg gain = {avg_delta:+.2f} pp  (win: 8/11)",
         transform=ax2.transAxes, ha="right", va="bottom",
         fontsize=9, fontweight="bold",
         bbox=dict(facecolor="white", edgecolor="gray", boxstyle="round,pad=0.3"))

plt.tight_layout()
outdir = Path("/efs/user_folders/dnshalam/work/FewShotAlignment/results/figs")
outdir.mkdir(parents=True, exist_ok=True)
for ext in ("pdf", "png"):
    plt.savefig(outdir / f"biomedcoop_k16.{ext}", dpi=200, bbox_inches="tight")
print("Saved:", outdir / "biomedcoop_k16.pdf", outdir / "biomedcoop_k16.png")
