"""Concat-probe experiment: [z_img ; z_flow] linear classifier.

Loads a trained FSA checkpoint (DINOv3-B + CLIP text, ImageNet K=16 seed 42),
re-encodes image features cached in linear_probe.py's feature cache, passes them
through the OP + flow adapter to get transported features, concatenates with
the raw image features, and fits an L2 log-reg classifier.

Evaluates on ImageNet full val + 4 OOD variants.
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


def load_fsa_model(ckpt_path, config_path, device="cuda"):
    import yaml
    import copy
    from src.datasets.base_dataset import build_dataset
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    cfg.update({
        "dataset": "ImageNet",
        "batch_size": 128,
        "text_adapter": True,
        "flow_adapter": "simple",
        "ada_depth": 2,
        "ada_dim": 1536,
        "source_cond": True,
        "img_src": "HF",
        "img_model": "facebook/dinov3-vitb16-pretrain-lvd1689m",
        "txt_src": "OC",
        "txt_model": "ViT-B-16",
        "txt_pretrained": "openai",
        "shots": 16,
        "seed": 42,
    })
    # Need classnames so the text encoder knows what to encode.
    ds = build_dataset(copy.deepcopy(cfg))
    cfg["classnames"] = ds.classnames
    model = FlowAdapter(cfg).to(device)
    load_checkpoint(model, ckpt_path)
    model.eval()
    return model, cfg


@torch.no_grad()
def compute_transported(model, feats, t_end=0.5, solver="midpoint", steps=2, chunk_size=512):
    """Apply OP + flow to get transported features z_flow.

    feats: [N, D_img] raw DINOv3-B features (normalized).
    Returns: z_flow in image-aligned space (same dim).
    """
    device = next(model.parameters()).device
    # OP was fit during create_bank; assume model has been "warmed up" with support
    z_list = []
    for i in range(0, feats.size(0), chunk_size):
        f = feats[i:i + chunk_size].to(device)
        # Source-cond flow needs an explicit conditioning vector; use the raw
        # feats themselves (consistent with training).
        y_img = f
        i2t = model._solve_ode(f, t_end=t_end, method=solver, steps=steps, y=y_img)
        z_list.append(i2t.cpu())
    return torch.cat(z_list, dim=0)


def fit_logreg(X_tr, y_tr, C=10.0):
    from sklearn.linear_model import LogisticRegression
    clf = LogisticRegression(C=C, max_iter=1000, solver="lbfgs", verbose=0)
    clf.fit(X_tr.numpy(), y_tr.numpy())
    return clf


def eval_probe(clf, feats, labels, class_subset=None):
    X = feats.numpy()
    y = labels.numpy()
    if class_subset is not None:
        subset = np.asarray(class_subset)
        df = clf.decision_function(X)
        df = df[:, subset]
        pred = df.argmax(axis=-1)
    else:
        pred = clf.predict(X)
    return float((pred == y).mean())


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default="output/ImageNet/enc_clipvitb16/model_best.pth")
    p.add_argument("--config", default="configs/imagenet_dinov3_qwen3.yaml")
    p.add_argument("--feat_dir", default="output/linear_probe/features")
    p.add_argument("--t_ends", type=str, default="0.5",
                   help="Comma-separated list of t_end values, e.g. '0.0,0.2,0.5,0.8,1.0'")
    p.add_argument("--C", type=float, default=10.0)
    p.add_argument("--out", default="output/linear_probe/concat_probe_result.json")
    p.add_argument("--root", default="/efs/user_folders/dnshalam/datasets")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--shots", type=int, default=16)
    args = p.parse_args()

    t_ends = [float(x) for x in args.t_ends.split(",")]
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
    val_feats, val_labels = load_feats(f"dinov3b__ImageNet__test.pth")
    ood = {}
    for ood_name in ["ImageNetV2", "ImageNetSketch", "ImageNetA", "ImageNetR"]:
        ood[ood_name] = load_feats(f"dinov3b__{ood_name}__test.pth")

    # Normalize feats
    tr_feats = F.normalize(tr_feats, dim=-1)
    val_feats = F.normalize(val_feats, dim=-1)
    for name in ood:
        ood[name] = (F.normalize(ood[name][0], dim=-1), ood[name][1])

    from linear_probe import get_imagenet_to_ood_subset
    cfg_root = {"root": args.root, "shots": args.shots, "seed": args.seed, "subsample_classes": "all"}

    all_results = []
    for t_end in t_ends:
        print(f"\n===========  t_end = {t_end}  ===========")
        if t_end == 0.0:
            # No flow at all: z_flow == z_img.
            tr_flow = tr_feats.clone()
            val_flow = val_feats.clone()
            ood_flow = {name: ood[name][0].clone() for name in ood}
        else:
            print(f"[*] Transporting features (t_end={t_end})...")
            tr_flow = compute_transported(model, tr_feats, t_end=t_end)
            val_flow = compute_transported(model, val_feats, t_end=t_end)
            ood_flow = {name: compute_transported(model, ood[name][0], t_end=t_end) for name in ood}
            tr_flow = F.normalize(tr_flow, dim=-1)
            val_flow = F.normalize(val_flow, dim=-1)
            for name in ood_flow:
                ood_flow[name] = F.normalize(ood_flow[name], dim=-1)

        results = {
            "ckpt": args.ckpt, "t_end": t_end, "C": args.C,
            "seed": args.seed, "shots": args.shots,
        }

        for probe_name, make_feat in [
            ("raw", lambda tr, fl: tr),
            ("flow_only", lambda tr, fl: fl),
            ("concat", lambda tr, fl: torch.cat([tr, fl], dim=-1)),
        ]:
            # raw probe doesn't depend on t_end - skip recomputing for t_end>0.0 after first
            if probe_name == "raw" and t_end != t_ends[0]:
                results["raw__ImageNet"] = all_results[0]["raw__ImageNet"]
                results["raw__ImageNetV2"] = all_results[0]["raw__ImageNetV2"]
                results["raw__ImageNetSketch"] = all_results[0]["raw__ImageNetSketch"]
                results["raw__ImageNetA"] = all_results[0]["raw__ImageNetA"]
                results["raw__ImageNetR"] = all_results[0]["raw__ImageNetR"]
                results["raw__ood_mean"] = all_results[0]["raw__ood_mean"]
                continue
            X_tr = make_feat(tr_feats, tr_flow)
            print(f"[probe={probe_name}] features shape={tuple(X_tr.shape)}; fitting log-reg C={args.C}")
            clf = fit_logreg(X_tr, tr_labels, C=args.C)

            X_val = make_feat(val_feats, val_flow)
            in_acc = eval_probe(clf, X_val, val_labels)
            results[f"{probe_name}__ImageNet"] = in_acc

            ood_scores = []
            for name, (feats, labels) in ood.items():
                fl = ood_flow[name]
                X = make_feat(feats, fl)
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
    print(f"{'t_end':>6} | {'raw IN':>8} {'flow IN':>8} {'cat IN':>8} | {'raw OOD':>9} {'flow OOD':>9} {'cat OOD':>9}")
    for r in all_results:
        print(f"{r['t_end']:>6.2f} | {r['raw__ImageNet']*100:>8.2f} {r['flow_only__ImageNet']*100:>8.2f} {r['concat__ImageNet']*100:>8.2f} | {r['raw__ood_mean']*100:>9.2f} {r['flow_only__ood_mean']*100:>9.2f} {r['concat__ood_mean']*100:>9.2f}")


if __name__ == "__main__":
    main()
