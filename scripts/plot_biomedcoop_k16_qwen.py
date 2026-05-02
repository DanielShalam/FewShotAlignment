#!/usr/bin/env python
"""Paper-grade figure for the BiomedCoOp K=16 comparison, with Qwen3 ablation."""
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# (ds, FSA+BMC mean/std, FSA+Qwen mean/std, BiomedCoOp mean/std)
data = [
    ("BUSI",       71.04, 3.02, 72.03, 3.93, 70.34, 2.27),
    ("CHMNIST",    87.37, 1.18, 87.23, 1.41, 79.05, 2.24),
    ("KneeXray",   41.51, 0.57, 41.39, 1.01, 39.69, 1.75),
    ("BTMRI",      81.16, 3.28, 79.34, 3.17, 83.30, 1.34),
    ("COVID_19",   75.51, 2.01, 75.53, 1.70, 78.72, 0.23),
    ("CTKidney",   75.77, 2.84, 75.51, 2.31, 83.20, 2.37),
    ("DermaMNIST", 64.31, 2.06, 62.64, 1.25, 62.59, 1.83),
    ("Kvasir",     85.45, 1.41, 85.78, 1.91, 78.89, 1.21),
    ("LungColon",  94.79, 0.31, 94.85, 0.27, 92.68, 0.57),
    ("OCTMNIST",   72.82, 0.80, 73.58, 0.58, 66.93, 2.13),
    ("RETINA",     75.90, 1.05, 75.76, 0.75, 61.28, 1.06),
]
names = [r[0] for r in data]
bmc   = np.array([r[1] for r in data]); bmc_s   = np.array([r[2] for r in data])  # FSA+BMC-text
qwen  = np.array([r[3] for r in data]); qwen_s  = np.array([r[4] for r in data])  # FSA+Qwen3
bio   = np.array([r[5] for r in data]); bio_s   = np.array([r[6] for r in data])  # BiomedCoOp

# Append average column
names.append("Avg")
bmc   = np.append(bmc,  bmc.mean());   bmc_s  = np.append(bmc_s,  bmc_s.mean())
qwen  = np.append(qwen, qwen.mean());  qwen_s = np.append(qwen_s, qwen_s.mean())
bio   = np.append(bio,  bio.mean());   bio_s  = np.append(bio_s,  bio_s.mean())

# Colors
COLOR_BMC_BASE    = "#6c6c6c"   # BiomedCoOp (baseline)
COLOR_FSA_BMCTEXT = "#2b7bba"   # FSA + BMC-text
COLOR_FSA_QWEN    = "#27ae60"   # FSA + Qwen3
AVG_ACCENT        = "#111111"

# One big panel: 3 grouped bars per dataset
fig, ax = plt.subplots(figsize=(13, 4.2))
x = np.arange(len(names))
w = 0.27

b1 = ax.bar(x - w, bio,  w, yerr=bio_s,  color=COLOR_BMC_BASE,
            edgecolor="black", linewidth=0.4, label="BiomedCoOp (SOTA)",
            error_kw={"lw": 0.6, "capsize": 2})
b2 = ax.bar(x,      bmc,  w, yerr=bmc_s,  color=COLOR_FSA_BMCTEXT,
            edgecolor="black", linewidth=0.4, label="FSA + DINOv3 + BiomedCLIP-text",
            error_kw={"lw": 0.6, "capsize": 2})
b3 = ax.bar(x + w,  qwen, w, yerr=qwen_s, color=COLOR_FSA_QWEN,
            edgecolor="black", linewidth=0.4, label="FSA + DINOv3 + Qwen3-Embed-0.6B",
            error_kw={"lw": 0.6, "capsize": 2})

# Highlight Avg column
for b in [b1, b2, b3]:
    b[-1].set_edgecolor(AVG_ACCENT); b[-1].set_linewidth(1.3)

ax.set_xticks(x)
ax.set_xticklabels(names, rotation=30, ha="right", fontsize=9)
ax.set_ylabel("Accuracy (%)", fontsize=10)
ax.set_title("Per-dataset accuracy at $K$=16 (3 seeds)", fontsize=11)
ax.set_ylim(30, 100)
ax.grid(axis="y", alpha=0.25, linewidth=0.5)
ax.axvline(x[-1] - 0.5, color="gray", linestyle="--", linewidth=0.6, alpha=0.6)
ax.legend(loc="lower right", frameon=False, fontsize=8)

# Annotate averages above Avg bars
avg_x = x[-1]
for i, (vals, col) in enumerate([(bio, COLOR_BMC_BASE), (bmc, COLOR_FSA_BMCTEXT), (qwen, COLOR_FSA_QWEN)]):
    ax.text(avg_x + (i-1)*w, vals[-1] + 1.2, f"{vals[-1]:.2f}",
            ha="center", fontsize=8.5, fontweight="bold", color=col)

plt.tight_layout()
outdir = Path("/efs/user_folders/dnshalam/work/FewShotAlignment/results/figs")
for ext in ("pdf", "png"):
    plt.savefig(outdir / f"biomedcoop_k16_qwen.{ext}", dpi=200, bbox_inches="tight")
print("Saved:", outdir / "biomedcoop_k16_qwen.png")
