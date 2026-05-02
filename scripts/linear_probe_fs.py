"""Few-shot linear probe on ESM features.

For each K in shots, draw K primary-class samples per class from the train split (seeded),
fit OVR logistic regression, evaluate macro-AUPRC on the test split.
"""
import argparse, json, random
from collections import defaultdict
from pathlib import Path
import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import average_precision_score

def macro_auprc(Y, S):
    aps = [average_precision_score(Y[:, c], S[:, c]) for c in range(Y.shape[1]) if Y[:, c].sum() > 0]
    return float(np.mean(aps))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="data/deeploc")
    ap.add_argument("--esm_file", default="seq_embed_esm2_t30_150m_ur50d.pt")
    ap.add_argument("--shots", nargs="+", type=int, default=[1, 4, 16, 64])
    ap.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    args = ap.parse_args()

    d = Path(args.data_dir)
    seq = torch.load(d / args.esm_file, weights_only=False)
    lab = torch.load(d / "labels_multihot.pt", weights_only=False)
    X_tr_all, Y_tr_all = seq["train"].numpy(), lab["train"].numpy()
    X_te, Y_te = seq["test"].numpy(), lab["test"].numpy()
    C = Y_te.shape[1]

    # group train by primary (argmax)
    primary = Y_tr_all.argmax(axis=1)
    per_class = defaultdict(list)
    for i, p in enumerate(primary):
        per_class[int(p)].append(i)

    print(f"{'K':>3s} {'seed':>5s}  macro-AUPRC")
    results = defaultdict(list)
    for K in args.shots:
        for s in args.seeds:
            rng = random.Random(s)
            idx = []
            for c in range(C):
                items = per_class[c][:]
                rng.shuffle(items)
                idx.extend(items[:K])
            X, Y = X_tr_all[idx], Y_tr_all[idx]
            clf = OneVsRestClassifier(LogisticRegression(max_iter=2000, C=1.0))
            clf.fit(X, Y)
            scores = np.stack([e.decision_function(X_te) for e in clf.estimators_], axis=1)
            m = macro_auprc(Y_te, scores)
            results[K].append(m)
            print(f"{K:3d} {s:5d}  {m*100:6.2f}")
    print()
    print(f"{'K':>3s}  {'mean':>7s} {'std':>6s}")
    for K, vs in sorted(results.items()):
        v = np.array(vs) * 100
        print(f"{K:3d}  {v.mean():7.2f} {v.std(ddof=0):6.2f}")

if __name__ == "__main__":
    main()
