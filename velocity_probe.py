"""Velocity-probe experiment.

At a given (x, t), evaluate the flow's velocity field v(x, t) and use it as
(optionally concatenated) features for an L2 log-reg classifier.

Variants:
    * velocity_only:   v(x, 0.0)            -- 768-d
    * concat_v:        [x ; v(x, 0.0)]      -- 1536-d
    * sweep t values for v(x, t)

Usage:
    python velocity_probe.py --ckpt ... --shots 1 --seed 42 --t_vals 0.0,0.2,0.5
"""
import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from src.model import FlowAdapter
from src.utils import load_checkpoint, set_seed

from concat_probe import load_fsa_model, fit_logreg, eval_probe


@torch.no_grad()
def compute_velocity(model, feats, t_val=0.0, chunk_size=512):
    """Return v(x, t_val) for each feature in ``feats``.

    Uses the adapter (image→text) with source-conditioning set to x itself.
    feats: [N, D], assumed L2-normalized image features.
    """
    device = next(model.parameters()).device
    net = model.adapter  # image-to-text flow
    outs = []
    for i in range(0, feats.size(0), chunk_size):
        x = feats[i:i + chunk_size].to(device)
        t = torch.full((x.size(0),), float(t_val), device=device)
        # SimpleMLP signature: forward(t, x, y)
        v = net(t, x, y=x)
        outs.append(v.cpu())
    return torch.cat(outs, dim=0)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default="output/ImageNet/enc_clipvitb16/model_best.pth")
    p.add_argument("--config", default="configs/imagenet_dinov3_qwen3.yaml")
    p.add_argument("--feat_dir", default="output/linear_probe/features")
    p.add_argument("--t_vals", default="0.0,0.2,0.5",
                   help="Comma-separated t values to probe the velocity at.")
    p.add_argument("--C", type=float, default=10.0)
    p.add_argument("--out", default="output/linear_probe/velocity_probe_result.json")
    p.add_argument("--root", default="/efs/user_folders/dnshalam/datasets")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--shots", type=int, default=16)
    p.add_argument("--normalize_v", action="store_true",
                   help="L2-normalize velocity before probe (velocity is not on unit sphere).")
    args = p.parse_args()

    t_vals = [float(x) for x in args.t_vals.split(",")]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    set_seed(args.seed)

    print(f"[*] Loading FSA checkpoint: {args.ckpt}")
    model, cfg = load_fsa_model(args.ckpt, args.config, device=device)

    def load_feats(tag):
        path = os.path.join(args.feat_dir, tag)
        d = torch.load(path, map_location="cpu", weights_only=True)
        return d["feats"].float(), d["labels"].long()

    print("[*] Loading cached features...")
    tr_feats, tr_labels = load_feats(
        f"dinov3b__ImageNet__train_support__k{args.shots}__seed{args.seed}.pth")
    val_feats, val_labels = load_feats("dinov3b__ImageNet__test.pth")
    ood = {}
    for ood_name in ["ImageNetV2", "ImageNetSketch", "ImageNetA", "ImageNetR"]:
        ood[ood_name] = load_feats(f"dinov3b__{ood_name}__test.pth")

    tr_feats = F.normalize(tr_feats, dim=-1)
    val_feats = F.normalize(val_feats, dim=-1)
    for name in ood:
        ood[name] = (F.normalize(ood[name][0], dim=-1), ood[name][1])

    from linear_probe import get_imagenet_to_ood_subset
    cfg_root = {"root": args.root, "shots": args.shots, "seed": args.seed, "subsample_classes": "all"}

    all_results = []
    for t in t_vals:
        print(f"\n===========  t = {t}  ===========")
        print(f"[*] Computing velocities at t={t}...")
        tr_v = compute_velocity(model, tr_feats, t_val=t)
        val_v = compute_velocity(model, val_feats, t_val=t)
        ood_v = {name: compute_velocity(model, ood[name][0], t_val=t) for name in ood}

        # Report magnitude statistics
        tr_norms = tr_v.norm(dim=-1)
        print(f"  tr_v norms: mean={tr_norms.mean():.3f} std={tr_norms.std():.3f} "
              f"min={tr_norms.min():.3f} max={tr_norms.max():.3f}")

        if args.normalize_v:
            tr_v = F.normalize(tr_v, dim=-1)
            val_v = F.normalize(val_v, dim=-1)
            for name in ood_v:
                ood_v[name] = F.normalize(ood_v[name], dim=-1)

        results = {"ckpt": args.ckpt, "t": t, "C": args.C, "seed": args.seed, "shots": args.shots}

        for probe_name, make_feat in [
            ("raw", lambda tr, v: tr),
            ("v_only", lambda tr, v: v),
            ("raw+v", lambda tr, v: torch.cat([tr, v], dim=-1)),
        ]:
            if probe_name == "raw" and t != t_vals[0]:
                for k in ["raw__ImageNet", "raw__ImageNetV2", "raw__ImageNetSketch",
                         "raw__ImageNetA", "raw__ImageNetR", "raw__ood_mean"]:
                    results[k] = all_results[0][k]
                continue
            X_tr = make_feat(tr_feats, tr_v)
            print(f"[probe={probe_name}] shape={tuple(X_tr.shape)}")
            clf = fit_logreg(X_tr, tr_labels, C=args.C)

            X_val = make_feat(val_feats, val_v)
            in_acc = eval_probe(clf, X_val, val_labels)
            results[f"{probe_name}__ImageNet"] = in_acc

            ood_scores = []
            for name, (feats, labels) in ood.items():
                v = ood_v[name]
                X = make_feat(feats, v)
                if name in ["ImageNetA", "ImageNetR"]:
                    subset = get_imagenet_to_ood_subset(cfg_root, name)
                    acc = eval_probe(clf, X, labels, class_subset=subset)
                else:
                    acc = eval_probe(clf, X, labels)
                results[f"{probe_name}__{name}"] = acc
                ood_scores.append(acc)
            results[f"{probe_name}__ood_mean"] = sum(ood_scores) / len(ood_scores)
            print(f"  IN={in_acc*100:.2f}  OOD-avg={results[f'{probe_name}__ood_mean']*100:.2f}")
        all_results.append(results)

    Path(os.path.dirname(args.out)).mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n[*] Saved -> {args.out}")

    print("\n=== SUMMARY ===")
    print(f"{'t':>6} | {'raw IN':>7} {'v IN':>7} {'rv IN':>7} | {'raw OOD':>8} {'v OOD':>8} {'rv OOD':>8}")
    for r in all_results:
        print(f"{r['t']:>6.2f} | {r['raw__ImageNet']*100:>7.2f} {r['v_only__ImageNet']*100:>7.2f} {r['raw+v__ImageNet']*100:>7.2f} | {r['raw__ood_mean']*100:>8.2f} {r['v_only__ood_mean']*100:>8.2f} {r['raw+v__ood_mean']*100:>8.2f}")


if __name__ == "__main__":
    main()
