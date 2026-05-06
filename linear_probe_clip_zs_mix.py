"""Mix linear probe (DINOv3) logits with CLIP zero-shot logits.

Fair baseline comparison for the CLIP-ZS-Mix idea:
- Fit log-reg on cached DINOv3 features (support).
- Load precomputed CLIP ZS logits for the same test set.
- Convert both to softmax probabilities and blend:
      p_final = (1 - alpha) * p_probe + alpha * p_clip
- Report top-1 accuracy across alpha grid for multiple shots.

Compares against the FSA+CLIP-ZS-Mix numbers at matched K, alpha.
"""
import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression


def load_feats(path):
    d = torch.load(path, map_location="cpu", weights_only=True)
    return d["feats"].float(), d["labels"].long()


def load_clip_zs(path):
    d = torch.load(path, map_location="cpu", weights_only=False)
    return d["logits"].float(), d["labels"].long()


def fit_probe(X_tr, y_tr, C=10.0):
    X = F.normalize(X_tr, dim=-1).numpy()
    y = y_tr.numpy()
    clf = LogisticRegression(C=C, max_iter=1000, solver="lbfgs", verbose=0)
    clf.fit(X, y)
    return clf


def probe_logits(clf, X):
    X = F.normalize(X, dim=-1).numpy()
    return torch.from_numpy(clf.decision_function(X)).float()  # [N, C]


def mix_probs(logits_a, logits_b, alpha):
    """Mix in softmax space. alpha in [0, 1]."""
    p_a = F.softmax(logits_a, dim=-1)
    p_b = F.softmax(logits_b, dim=-1)
    return (1 - alpha) * p_a + alpha * p_b


def evaluate(logits_a, logits_b, alpha, labels):
    p = mix_probs(logits_a, logits_b, alpha)
    pred = p.argmax(dim=-1)
    return float((pred == labels).float().mean())


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--feat_dir", default="output/linear_probe/features")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--shots", type=int, nargs="+", default=[1, 4, 16])
    p.add_argument("--alphas", nargs="+", default=["0.0", "0.25", "0.5", "0.7", "0.9", "1.0"])
    p.add_argument("--C", type=float, default=10.0)
    p.add_argument("--datasets", nargs="+",
                   default=["ImageNet", "ImageNetV2", "ImageNetSketch", "ImageNetA", "ImageNetR"])
    p.add_argument("--root", default="/efs/user_folders/dnshalam/datasets")
    p.add_argument("--out", default="output/linear_probe/clip_zs_mix_comparison.json")
    args = p.parse_args()
    alphas = [float(a) for a in args.alphas]

    # For ImageNet-A / ImageNet-R we need a 200-way subset of the 1000-way probe.
    def get_subset_indices(ds_name):
        from linear_probe import get_imagenet_to_ood_subset
        cfg = {"root": args.root, "shots": 16, "seed": args.seed, "subsample_classes": "all"}
        return get_imagenet_to_ood_subset(cfg, ds_name)

    # Clip-zs file map
    zs_files = {
        "ImageNet": "output/clip_zs/imagenet.pth",
        "ImageNetV2": "output/clip_zs/imagenetv2.pth",
        "ImageNetSketch": "output/clip_zs/imagenetsketch.pth",
        "ImageNetA": "output/clip_zs/imageneta.pth",
        "ImageNetR": "output/clip_zs/imagenetr.pth",
    }

    results = []
    for shots in args.shots:
        tr_path = os.path.join(
            args.feat_dir, f"dinov3b__ImageNet__train_support__k{shots}__seed{args.seed}.pth"
        )
        if not os.path.exists(tr_path):
            print(f"  MISSING: {tr_path}")
            continue

        print(f"\n=== K={shots} seed={args.seed} ===")
        tr_feats, tr_labels = load_feats(tr_path)
        print(f"  support: {tr_feats.shape}")

        clf = fit_probe(tr_feats, tr_labels, C=args.C)

        for ds in args.datasets:
            # Load test features + CLIP ZS logits
            val_feats, val_labels = load_feats(os.path.join(args.feat_dir, f"dinov3b__{ds}__test.pth"))
            clip_logits, clip_labels = load_clip_zs(zs_files[ds])
            assert clip_logits.size(0) == val_feats.size(0)
            assert torch.equal(clip_labels, val_labels)

            # Probe logits — for ImageNetA / ImageNetR restrict to the 200 relevant columns
            probe_val_logits = probe_logits(clf, val_feats)  # [N, 1000]
            if ds in ("ImageNetA", "ImageNetR"):
                subset = get_subset_indices(ds)
                probe_val_logits = probe_val_logits[:, subset]  # [N, 200]

            row = {"shots": shots, "seed": args.seed, "dataset": ds}
            for alpha in alphas:
                acc = evaluate(probe_val_logits, clip_logits, alpha, val_labels)
                row[f"alpha_{alpha}"] = acc
            results.append(row)
            # compact print: best alpha
            best_alpha = max(alphas, key=lambda a: row[f"alpha_{a}"])
            print(f"    {ds:<18} best α={best_alpha:<4} acc={row[f'alpha_{best_alpha}']*100:.2f}%")

    Path(os.path.dirname(args.out)).mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[*] Saved -> {args.out}")

    print("\n=== SUMMARY ===")
    for shots in args.shots:
        print(f"\n-- K={shots} --")
        hdr = f'{"dataset":<18}  ' + '  '.join(f'a={a:<4}' for a in alphas)
        print(hdr)
        for r in results:
            if r["shots"] != shots: continue
            accs = '  '.join(f'{r[f"alpha_{a}"]*100:>5.2f}' for a in alphas)
            print(f'{r["dataset"]:<18}  {accs}')


if __name__ == "__main__":
    main()
