"""Generate the shot-curve figure (macro-AUPRC vs K)."""
import matplotlib.pyplot as plt
import numpy as np

K = np.array([1, 4, 16, 64])
# (mean, std) per K
fsa_pubmed  = np.array([(19.48,1.70),(29.56,1.98),(39.66,0.06),(43.65,1.21)])
fsa_qwen06  = np.array([(21.81,1.85),(36.09,1.98),(43.99,1.43),(48.12,0.49)])
lp_esm150   = np.array([(18.96,1.06),(31.26,1.69),(36.27,0.79),(42.18,1.34)])
qwen8b_k64  = (48.60, 0.46)
lp_full     = 55.89   # ESM-150M linear probe, full train

fig, ax = plt.subplots(figsize=(6.5, 4.2))
for arr, lbl, mk, c in [
    (fsa_qwen06, "FSA + Qwen-0.6B",  "o", "#d62728"),
    (fsa_pubmed, "FSA + PubMedBERT", "s", "#1f77b4"),
    (lp_esm150,  "Linear probe (ESM-2-150M)", "^", "#2ca02c"),
]:
    m, s = arr[:,0], arr[:,1]
    ax.errorbar(K, m, yerr=s, marker=mk, lw=1.7, ms=6, capsize=3, label=lbl, color=c)
ax.errorbar([64], [qwen8b_k64[0]], yerr=[qwen8b_k64[1]], marker="*", ms=13,
            color="#d62728", mec="black", mew=0.8, label="FSA + Qwen-8B (K=64)")
ax.axhline(lp_full, ls="--", lw=1.2, color="gray", label=f"Linear probe, full train (N=21,948) = {lp_full:.1f}")

ax.set_xscale("log", base=2)
ax.set_xticks(K); ax.set_xticklabels([str(k) for k in K])
ax.set_xlabel("Shots per primary class (K)")
ax.set_ylabel("Test macro-AUPRC (%)")
ax.set_title("DeepLoc 2.0 Multi — FSA vs. linear probe (mean ± std over 3 seeds)")
ax.grid(alpha=.25); ax.legend(loc="lower right", fontsize=9)
plt.tight_layout()
plt.savefig("results/figs/shot_curve.png", dpi=180)
plt.savefig("results/figs/shot_curve.pdf")
print("ok")
