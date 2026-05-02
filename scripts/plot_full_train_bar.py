"""Bar chart: full-train ceilings (OP-only, Linear probe, OP+Flow) on DeepLoc2Multi."""
import matplotlib.pyplot as plt

labels = ["OP-only\n(Qwen-0.6B)", "Linear probe\n(ESM-2-150M)", "OP+Flow (FSA)\n(Qwen-0.6B)"]
values = [42.60, 55.89, 61.42]
colors = ["#9ecae1", "#2ca02c", "#d62728"]

fig, ax = plt.subplots(figsize=(5.2, 3.8))
bars = ax.bar(labels, values, color=colors, edgecolor="black", linewidth=0.6, width=0.6)
for b, v in zip(bars, values):
    ax.text(b.get_x() + b.get_width()/2, v + 0.7, f"{v:.2f}", ha="center", fontsize=10, fontweight="bold")
ax.set_ylabel("Test macro-AUPRC (%)")
ax.set_ylim(0, max(values) + 8)
ax.set_title("Full-train DeepLoc 2.0 Multi (N=21,948)")
ax.grid(axis="y", alpha=.25)
plt.tight_layout()
plt.savefig("results/figs/full_train_bar.png", dpi=180)
plt.savefig("results/figs/full_train_bar.pdf")
print("ok")
