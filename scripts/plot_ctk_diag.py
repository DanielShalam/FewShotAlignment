#!/usr/bin/env python
"""Plot training trajectory: val + test accuracy per eval, plus final tuned acc.
Compares the original run vs the rerun."""
import re, matplotlib.pyplot as plt
from pathlib import Path

OUT = Path("/efs/user_folders/dnshalam/work/FewShotAlignment/output")
runs = {
    "original (dv3+bmc k16 s42)": OUT / "BiomedCoOp_focused" / "CTKidney_dinov3_k16_s42.log",
    "rerun (dv3+bmc k16 s42)":    OUT / "CTK_redo_dv3_bmc_k16_s42.log",
    "BMC vision + BMC text (k16 s42)": OUT / "CTK_fsa_bmc_bmc_k16_s42.log",
}
EVAL_FREQ = 25  # from config

fig, ax = plt.subplots(figsize=(10, 5))
colors = {"val": ["#2b7bba", "#27ae60", "#8e44ad"],
          "test": ["#1f4e79", "#196f3d", "#5b2c6f"]}
final = {}

for i, (label, p) in enumerate(runs.items()):
    if not p.exists():
        print(f"missing: {p}"); continue
    text = p.read_text()
    vals = [float(x) for x in re.findall(r'Val Accuracy: ([0-9.]+)', text)]
    tests = [float(x) for x in re.findall(r'Test Accuracy: ([0-9.]+)', text)]
    n = min(len(vals), len(tests))
    if n == 0:
        print(f"no evals yet: {label}"); continue
    epochs = [EVAL_FREQ * (j+1) for j in range(n)]
    ax.plot(epochs, vals[:n],  '-o', color=colors["val"][i],  label=f"{label} — val",  markersize=4, linewidth=1.5, alpha=0.9)
    ax.plot(epochs, tests[:n], '--s', color=colors["test"][i], label=f"{label} — test", markersize=4, linewidth=1.2, alpha=0.7)
    after = re.findall(r'Test Accuracy \(After tuning\): ([0-9.]+)', text)
    best_val_loaded = re.findall(r'Loaded best-val checkpoint \(val acc = ([0-9.]+)%\)', text)
    if after:
        final[label] = (float(after[-1]), float(best_val_loaded[-1]) if best_val_loaded else None)

ax.axhline(83.20, color='#c0392b', linestyle=':', linewidth=1.5, alpha=0.8, label="BiomedCoOp SOTA = 83.20")
ax.set_xlabel("Epoch")
ax.set_ylabel("Accuracy (%)")
ax.set_title("CTKidney K=16 seed=42: val+test vs epoch")
ax.grid(alpha=0.3)
ax.legend(fontsize=8, loc="lower right")
ax.set_ylim(35, 95)

# Annotate final tuned accuracy
info_lines = ["After-tune test | best-val ckpt:"]
for k,(a,b) in final.items():
    info_lines.append(f"  {k}: {a:.2f}  (val @ ckpt: {b:.1f}%)")
txt = "\n".join(info_lines)
ax.text(0.02, 0.98, txt, transform=ax.transAxes, fontsize=8,
        va="top", ha="left",
        bbox=dict(facecolor="white", edgecolor="gray", boxstyle="round,pad=0.4"))

out_png = OUT.parent / "results" / "figs" / "ctkidney_diag.png"
plt.tight_layout()
plt.savefig(out_png, dpi=180, bbox_inches="tight")
print("saved:", out_png)
