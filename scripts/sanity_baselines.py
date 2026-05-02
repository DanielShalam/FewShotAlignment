"""Minimal sanity baselines on DeepLoc-2 Multi embeddings.

1) Full-train linear probe (one-vs-rest) on ESM features -> test macro-AUPRC.
2) Zero-shot cosine between class-text prototype and ESM feature (no alignment).

Prints one summary table.
"""
import argparse, json
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import average_precision_score

def macro_auprc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    # class-wise average precision, mean over classes that have positives
    aps = []
    for c in range(y_true.shape[1]):
        if y_true[:, c].sum() == 0:
            continue
        aps.append(average_precision_score(y_true[:, c], y_score[:, c]))
    return float(np.mean(aps))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="data/deeploc")
    ap.add_argument("--esm_file", default="seq_embed_esm2_t30_150m_ur50d.pt")
    args = ap.parse_args()

    d = Path(args.data_dir)
    seq = torch.load(d / args.esm_file, weights_only=False)
    lab = torch.load(d / "labels_multihot.pt", weights_only=False)
    meta = json.loads((d / "meta.json").read_text())

    X_tr, Y_tr = seq["train"].numpy(), lab["train"].numpy()
    X_te, Y_te = seq["test"].numpy(),  lab["test"].numpy()
    print(f"train: {X_tr.shape}, test: {X_te.shape}, D_seq={X_tr.shape[1]}")

    # ---- (1) Linear probe, full train ----
    clf = OneVsRestClassifier(LogisticRegression(max_iter=2000, C=1.0, n_jobs=-1), n_jobs=-1)
    clf.fit(X_tr, Y_tr)
    scores = np.zeros_like(Y_te, dtype=np.float32)
    for c, est in enumerate(clf.estimators_):
        # decision_function exists for LR
        scores[:, c] = est.decision_function(X_te)
    lp = macro_auprc(Y_te, scores)

    # ---- (2) Zero-shot text cosine ----
    seq_te = F.normalize(seq["test"], dim=-1)        # already normalized, idempotent
    rows = [("linear_probe_esm(full)", lp)]
    for f in sorted(d.glob("class_text_*.pt")):
        proto = torch.load(f, weights_only=False)     # [10, D_text]
        # naive cross-dim cosine is ill-defined; pad or project to compare
        # => we follow FSA's standard "no alignment" baseline: match by shared prefix dim if equal,
        # otherwise *minimum* valid baseline is random; here we project via PCA to the lower dim.
        # Simpler, honest baseline: use OP on full train -> zero-shot cosine on test (LFA zero-shot).
        # Without OP, text and protein live in different spaces of different dims.
        Dt = proto.shape[1]; Dp = seq_te.shape[1]
        # Use the pseudo-inverse mapping from train: best linear map P s.t. T @ P ~ X (Procrustes)
        T_tr = proto[np.argmax(Y_tr, axis=1)]          # per-train-sample text pick (first pos class)
        X_tr_t = torch.as_tensor(X_tr, dtype=torch.float32)
        # closed-form orthogonal Procrustes: W = U V^T from SVD of T^T X
        M = (T_tr.T.float() @ X_tr_t).float()
        U, S, Vh = torch.linalg.svd(M, full_matrices=False)
        W = (U @ Vh)                                   # [Dt, Dp]
        proto_in_prot = F.normalize(proto @ W, dim=-1)  # [10, Dp]
        s = seq_te @ proto_in_prot.T                    # [N, 10]
        ap_score = macro_auprc(Y_te, s.numpy())
        rows.append((f"OP_zeroshot::{f.stem.replace('class_text_','')}", ap_score))

    print()
    print(f"{'baseline':70s}  macro-AUPRC")
    print("-" * 90)
    for k, v in rows:
        print(f"{k:70s}  {v*100:6.2f}")

if __name__ == "__main__":
    main()
