"""Linear probe baseline for ImageNet K=16 + OOD variants.

Encodes images with a frozen visual encoder (CLIP ViT-B/16 pre-projection or
DINOv3-B), fits an L2-regularized logistic regression on the K-shot support
set, and evaluates on ImageNet val + {ImageNet-V2, -Sketch, -A, -R}.

Features are cached to disk per (encoder, dataset) so that multiple seeds of the
K=16 support only require one full encoding pass.

Convention:
    - For ImageNet/V2/Sketch: 1000-way classification.
    - For ImageNet-A/R: 200-way classification. The probe head is trained on
      1000 classes and sliced to the 200 columns present in each OOD dataset.

Usage:
    python linear_probe.py --encoder dinov3b --seed 42 --shots 16
    python linear_probe.py --encoder clip_b16_preproj --seed 42 --shots 16
"""
import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms as T
from tqdm import tqdm

from src.datasets.base_dataset import build_dataset, DatasetWrapper
from src.datasets.imagenet import ImageNet


# --------------------------
# Encoders
# --------------------------

class CLIPVisualPreProj(torch.nn.Module):
    """CLIP ViT-B/16 visual tower, returning the 768-d pre-projection features
    (before the text-aligned `visual.proj` matmul)."""
    def __init__(self, device="cuda"):
        super().__init__()
        import open_clip
        model, _, preprocess = open_clip.create_model_and_transforms(
            "ViT-B-16", pretrained="openai", force_quick_gelu=True
        )
        model.eval().to(device)
        for p in model.parameters():
            p.requires_grad_(False)
        self.visual = model.visual
        self.preprocess = preprocess
        self.device = device
        self.dim = self.visual.proj.shape[0]  # 768
        self.img_size = 224

    @torch.no_grad()
    def encode(self, pixel_values):
        # Save, zero out, restore proj to get pre-projection features via
        # model.visual() API without modifying the returned interface.
        proj = self.visual.proj
        self.visual.proj = None
        try:
            feats = self.visual(pixel_values)
        finally:
            self.visual.proj = proj
        return feats  # [B, 768]


class DINOv3B(torch.nn.Module):
    """facebook/dinov3-vitb16-pretrain-lvd1689m via HuggingFace AutoModel.
    Returns 768-d [CLS] token feature."""
    def __init__(self, device="cuda"):
        super().__init__()
        from transformers import AutoModel, AutoProcessor
        self.name = "facebook/dinov3-vitb16-pretrain-lvd1689m"
        self.model = AutoModel.from_pretrained(self.name).eval().to(device)
        for p in self.model.parameters():
            p.requires_grad_(False)
        self.processor = AutoProcessor.from_pretrained(self.name, use_fast=True)
        self.device = device
        self.dim = self.model.config.hidden_size  # 768
        self.img_size = 224
        mean = self.processor.image_mean
        std = self.processor.image_std
        self.preprocess = T.Compose([
            T.Resize(256, interpolation=T.InterpolationMode.BICUBIC),
            T.CenterCrop(224),
            T.ToTensor(),
            T.Normalize(mean=mean, std=std),
        ])

    @torch.no_grad()
    def encode(self, pixel_values):
        out = self.model(pixel_values=pixel_values)
        return out.last_hidden_state[:, 0, :]  # [CLS]


class DINOv2B(torch.nn.Module):
    """facebook/dinov2-base via HuggingFace AutoModel. 768-d [CLS]."""
    def __init__(self, device="cuda"):
        super().__init__()
        from transformers import AutoModel, AutoImageProcessor
        self.name = "facebook/dinov2-base"
        self.model = AutoModel.from_pretrained(self.name).eval().to(device)
        for p in self.model.parameters():
            p.requires_grad_(False)
        self.processor = AutoImageProcessor.from_pretrained(self.name, use_fast=True)
        self.device = device
        self.dim = self.model.config.hidden_size  # 768
        self.img_size = 224
        mean = self.processor.image_mean
        std = self.processor.image_std
        self.preprocess = T.Compose([
            T.Resize(256, interpolation=T.InterpolationMode.BICUBIC),
            T.CenterCrop(224),
            T.ToTensor(),
            T.Normalize(mean=mean, std=std),
        ])

    @torch.no_grad()
    def encode(self, pixel_values):
        out = self.model(pixel_values=pixel_values)
        return out.last_hidden_state[:, 0, :]


ENCODER_REGISTRY = {
    "clip_b16_preproj": CLIPVisualPreProj,
    "dinov3b": DINOv3B,
    "dinov2b": DINOv2B,
}


# --------------------------
# Dataset -> DataLoader
# --------------------------

def make_loader(cfg, dataset_name, split, preprocess, batch_size=256, num_workers=8):
    """Build a DataLoader over the requested split of the requested dataset.

    split in {"train_support", "test"}.
    """
    c = dict(cfg)
    c["dataset"] = dataset_name
    d = build_dataset(c)
    if split == "train_support":
        items = d.train_x
    elif split == "test":
        items = d.test
    else:
        raise ValueError(split)
    wrapper = DatasetWrapper(c, items, transform=preprocess, is_train=False)
    loader = DataLoader(
        wrapper,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=False, drop_last=False, pin_memory=True,
    )
    return loader, d


@torch.no_grad()
def encode_split(encoder, loader, device="cuda", desc=""):
    feats = []
    labels = []
    for batch in tqdm(loader, desc=desc, dynamic_ncols=True):
        imgs = batch["img"].to(device, non_blocking=True)
        y = batch["label"]
        z = encoder.encode(imgs)
        feats.append(z.float().cpu())
        labels.append(y.long())
    feats = torch.cat(feats, dim=0)
    labels = torch.cat(labels, dim=0)
    return feats, labels


def cache_path(cache_dir, encoder_name, dataset_name, split, seed=None, shots=None):
    tag = f"{encoder_name}__{dataset_name}__{split}"
    if shots is not None and split == "train_support":
        tag += f"__k{shots}"
    if seed is not None:
        tag += f"__seed{seed}"
    return os.path.join(cache_dir, tag + ".pth")


def ensure_features(encoder, encoder_name, cfg, dataset_name, split, cache_dir,
                    seed=None, shots=None, batch_size=256, num_workers=8, device="cuda"):
    path = cache_path(cache_dir, encoder_name, dataset_name, split, seed=seed, shots=shots)
    if os.path.exists(path):
        d = torch.load(path, map_location="cpu", weights_only=True)
        return d["feats"], d["labels"]
    # ensure dataset reads the right seed/shots when drawing few-shot support
    cfg = dict(cfg)
    if seed is not None:
        cfg["seed"] = seed
    if shots is not None:
        cfg["shots"] = shots
    loader, _ = make_loader(
        cfg, dataset_name, split, encoder.preprocess,
        batch_size=batch_size, num_workers=num_workers,
    )
    t0 = time.time()
    feats, labels = encode_split(encoder, loader, device=device,
                                 desc=f"encode {encoder_name}/{dataset_name}/{split}")
    dt = time.time() - t0
    print(f"[+] {dataset_name}/{split}: {feats.shape} labels={labels.shape} in {dt:.1f}s")
    Path(os.path.dirname(path)).mkdir(parents=True, exist_ok=True)
    torch.save({"feats": feats, "labels": labels}, path)
    return feats, labels


# --------------------------
# Linear probe (L2 log-reg)
# --------------------------

def fit_logreg(feats_train, labels_train, feats_val, labels_val,
               Cs=None, normalize=True, fixed_C=None):
    """Fit L2-regularized logistic regression.

    If ``fixed_C`` is provided, just fit once at that C and return. Otherwise
    sweep ``Cs`` and pick the best on (feats_val, labels_val).

    Returns (clf, C, val_acc). val_acc is None when fixed_C is used (no tuning
    set is consumed).
    """
    from sklearn.linear_model import LogisticRegression

    if normalize:
        feats_train = F.normalize(feats_train, dim=-1)

    Xtr = feats_train.numpy()
    ytr = labels_train.numpy()

    if fixed_C is not None:
        clf = LogisticRegression(
            C=fixed_C, max_iter=1000, solver="lbfgs", verbose=0,
        )
        clf.fit(Xtr, ytr)
        return clf, fixed_C, None

    # sweep
    if Cs is None:
        Cs = [0.316, 1.0, 3.16, 10.0, 31.6, 100.0]
    if normalize:
        feats_val = F.normalize(feats_val, dim=-1)
    Xva = feats_val.numpy()
    yva = labels_val.numpy()
    best = None
    for C in Cs:
        clf = LogisticRegression(C=C, max_iter=1000, solver="lbfgs", verbose=0)
        clf.fit(Xtr, ytr)
        pred = clf.predict(Xva)
        acc = float((pred == yva).mean())
        print(f"  C={C:>7.3f}  val_acc={acc*100:.2f}%")
        if best is None or acc > best[2]:
            best = (clf, C, acc)
    clf, C, acc = best
    return clf, C, acc


def evaluate_probe(clf, feats, labels, class_subset=None, normalize=True, topk=(1, 5)):
    """Evaluate probe on features. Returns dict {k: acc@k} for k in topk.

    If class_subset is provided (list of 1000-way class indices), restrict the
    decision function to those columns and re-map labels 0..N.
    """
    if normalize:
        feats = F.normalize(feats, dim=-1)
    X = feats.numpy()
    y = labels.numpy()
    df = clf.decision_function(X)
    if class_subset is not None:
        subset = np.asarray(class_subset)
        df = df[:, subset]
    # For each sample, get top-k predictions
    out = {}
    for k in topk:
        if k == 1:
            pred = df.argmax(axis=-1)
            out[1] = float((pred == y).mean())
        else:
            # partition for top-k (faster than full sort)
            topk_idx = np.argpartition(df, -k, axis=-1)[:, -k:]
            hit = (topk_idx == y[:, None]).any(axis=-1)
            out[k] = float(hit.mean())
    return out


# --------------------------
# Main
# --------------------------

def get_imagenet_to_ood_subset(cfg, ood_dataset_name):
    """Return a 1000-d index array: class_subset[i] = ImageNet-1k class index
    that OOD-dataset's class i corresponds to.
    """
    root = os.path.abspath(os.path.expanduser(cfg["root"]))
    in1k_classnames = ImageNet.read_classnames(
        os.path.join(root, "imagenet", "classnames.txt"))
    c = dict(cfg); c["dataset"] = ood_dataset_name
    ood = build_dataset(c)
    ood_classnames_in_order = list(ood.classnames)
    name_to_wnid = {v: k for k, v in in1k_classnames.items()}
    in1k_wnid_to_idx = {w: i for i, w in enumerate(in1k_classnames.keys())}
    # classnames in ood are full-string classnames (like "goldfish"); map via wnid
    # by looking up in 1k's name->wnid
    ood_to_in1k = []
    for cn in ood_classnames_in_order:
        wnid = name_to_wnid[cn]
        ood_to_in1k.append(in1k_wnid_to_idx[wnid])
    return ood_to_in1k


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--encoder", required=True, choices=list(ENCODER_REGISTRY.keys()))
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--shots", type=int, default=16)
    p.add_argument("--root", default="/efs/user_folders/dnshalam/datasets")
    p.add_argument("--cache_dir", default="output/linear_probe/features")
    p.add_argument("--results_dir", default="output/linear_probe/results")
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--num_workers", type=int, default=8)
    p.add_argument("--eval_only", action="store_true",
                   help="Skip probe fitting, assume cached clf from a previous run.")
    p.add_argument("--fixed_C", type=float, default=None,
                   help="If set, use this C for L2 log-reg (no held-out val sweep). "
                        "Evaluates on the full test set.")
    p.add_argument("--dataset", type=str, default="ImageNet",
                   help="Dataset name (default: ImageNet runs the full OOD sweep; "
                        "any other dataset does a single support→test evaluation).")
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    cfg = {
        "root": args.root,
        "seed": args.seed,
        "shots": args.shots,
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "subsample_classes": "all",
    }

    print(f"[*] encoder={args.encoder}, seed={args.seed}, shots={args.shots}")
    encoder = ENCODER_REGISTRY[args.encoder](device=device)
    print(f"[*] encoder dim={encoder.dim}, img_size={encoder.img_size}")

    # ---- 1. Encode support ----
    tr_feats, tr_labels = ensure_features(
        encoder, args.encoder, cfg, args.dataset, "train_support",
        args.cache_dir, seed=args.seed, shots=args.shots,
        batch_size=args.batch_size, num_workers=args.num_workers, device=device,
    )

    # ---- 2. Encode test (seed-independent for ImageNet; for others, test == the full test split) ----
    val_feats, val_labels = ensure_features(
        encoder, args.encoder, cfg, args.dataset, "test",
        args.cache_dir, seed=None,
        batch_size=args.batch_size, num_workers=args.num_workers, device=device,
    )

    # ---- 3. Fit linear probe ----
    if args.fixed_C is not None:
        print(f"[*] Fitting logreg on {tr_feats.shape[0]} support at fixed C={args.fixed_C} "
              f"(no val tuning; evaluating on full test)")
        clf, best_C, val_acc = fit_logreg(
            tr_feats, tr_labels, None, None, fixed_C=args.fixed_C,
        )
    else:
        # Tune regularization on a small held-out slice of val (5k) and
        # evaluate on the remaining 45k.
        rng = np.random.RandomState(args.seed)
        n_val = val_feats.shape[0]
        perm = rng.permutation(n_val)
        n_tune = min(5000, n_val // 10)
        tune_idx = perm[:n_tune]
        eval_idx_subset = perm[n_tune:]
        val_feats_tune = val_feats[tune_idx]
        val_labels_tune = val_labels[tune_idx]
        print(f"[*] Fitting logreg on {tr_feats.shape[0]} support, tuning on {n_tune}")
        clf, best_C, val_acc = fit_logreg(
            tr_feats, tr_labels, val_feats_tune, val_labels_tune,
        )
        print(f"[+] best C={best_C}, tune val_acc={val_acc*100:.2f}%")

    # ---- 4. Evaluate on test ----
    if args.fixed_C is not None:
        in_accs = evaluate_probe(clf, val_feats, val_labels)
        print(f"[+] {args.dataset} (full test, {val_feats.shape[0]} images): "
              f"top1={in_accs[1]*100:.2f}%  top5={in_accs[5]*100:.2f}%")
    else:
        in_accs = evaluate_probe(clf, val_feats[eval_idx_subset], val_labels[eval_idx_subset])
        print(f"[+] {args.dataset} (held-out, {len(eval_idx_subset)} images): "
              f"top1={in_accs[1]*100:.2f}%  top5={in_accs[5]*100:.2f}%")

    results = {
        "encoder": args.encoder,
        "dataset": args.dataset,
        "seed": args.seed,
        "shots": args.shots,
        "best_C": best_C,
        "fixed_C": args.fixed_C,
        "val_acc_tune": val_acc,
        "test_acc": in_accs[1],
        "test_acc_top5": in_accs[5],
    }

    # ---- 5. OOD variants (only for ImageNet) ----
    if args.dataset == "ImageNet":
        results["ImageNet_val_heldout"] = in_accs[1]  # legacy key
        results["ImageNet_val_heldout_top5"] = in_accs[5]
        for ood_name, needs_subset in [
            ("ImageNetV2", False),
            ("ImageNetSketch", False),
            ("ImageNetA", True),
            ("ImageNetR", True),
        ]:
            feats, labels = ensure_features(
                encoder, args.encoder, cfg, ood_name, "test",
                args.cache_dir, seed=None,
                batch_size=args.batch_size, num_workers=args.num_workers, device=device,
            )
            if needs_subset:
                subset = get_imagenet_to_ood_subset(cfg, ood_name)
                accs = evaluate_probe(clf, feats, labels, class_subset=subset)
            else:
                accs = evaluate_probe(clf, feats, labels)
            print(f"[+] {ood_name}: top1={accs[1]*100:.2f}%  top5={accs[5]*100:.2f}%")
            results[ood_name] = accs[1]
            results[f"{ood_name}_top5"] = accs[5]

        results["ood_mean"] = sum(results[k] for k in
            ["ImageNetV2", "ImageNetSketch", "ImageNetA", "ImageNetR"]) / 4.0
        results["ood_mean_top5"] = sum(results[f"{k}_top5"] for k in
            ["ImageNetV2", "ImageNetSketch", "ImageNetA", "ImageNetR"]) / 4.0

    Path(args.results_dir).mkdir(parents=True, exist_ok=True)
    out = os.path.join(args.results_dir,
                       f"{args.encoder}_{args.dataset}_shots{args.shots}_seed{args.seed}.json")
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"[*] Saved results -> {out}")

    if args.dataset == "ImageNet":
        print("\n=== SUMMARY ===")
        print(f"top1:  IN={results['test_acc']*100:.2f}  V2={results['ImageNetV2']*100:.2f}  "
              f"SK={results['ImageNetSketch']*100:.2f}  A={results['ImageNetA']*100:.2f}  "
              f"R={results['ImageNetR']*100:.2f}  OOD-avg={results['ood_mean']*100:.2f}")
        print(f"top5:  IN={results['test_acc_top5']*100:.2f}  V2={results['ImageNetV2_top5']*100:.2f}  "
              f"SK={results['ImageNetSketch_top5']*100:.2f}  A={results['ImageNetA_top5']*100:.2f}  "
              f"R={results['ImageNetR_top5']*100:.2f}  OOD-avg={results['ood_mean_top5']*100:.2f}")
    else:
        print(f"\n=== SUMMARY ===\n{args.encoder} {args.dataset} K={args.shots} seed={args.seed}: "
              f"top1={in_accs[1]*100:.2f}%  top5={in_accs[5]*100:.2f}%")


if __name__ == "__main__":
    main()
