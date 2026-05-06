"""Evaluate FSA checkpoint top-1 AND top-5 on ImageNet val + OOD variants.

Usage:
    python eval_fsa_topk.py \
        --resume output/ImageNet/enc_clipvitb16/model_best.pth \
        --alpha 0.1 --t_end 0.4 \
        --datasets ImageNet ImageNetV2 ImageNetSketch ImageNetA ImageNetR \
        txt_src OC txt_model ViT-B-16 txt_pretrained openai ...
"""
import argparse
import copy
import json
from pathlib import Path

import torch
import torch.nn.functional as F
import yaml
from tqdm import tqdm

from src.datasets.base_dataset import build_dataset, DatasetWrapper
from src.model import FlowAdapter
from src.utils import load_checkpoint, set_seed, setup_logger


def load_cfg(config_path, opts):
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    if opts:
        for i in range(0, len(opts), 2):
            k = opts[i].lstrip("-")
            v = opts[i+1]
            try:
                cfg[k] = yaml.safe_load(v)
            except Exception:
                cfg[k] = v
    return cfg


@torch.no_grad()
def evaluate_topk(model, loader, device, alpha=0.0, t_end=0.2,
                  solver="midpoint", steps=2, topk=(1, 5), logit_mode="mt"):
    """Evaluate top-k accuracy for the FSA model on a given dataset."""
    model.eval()
    correct_k = {k: 0 for k in topk}
    total = 0
    for batch in tqdm(loader, desc="eval", dynamic_ncols=True):
        x = batch["img"].to(device, non_blocking=True)
        y = batch["label"].to(device)
        out = model(x, t_end=t_end, solver=solver, steps=steps)
        if logit_mode == "mt":
            logits = (1 - alpha) * out["MT"] + alpha * out["ZS"]
        elif logit_mode == "zs":
            logits = out["ZS"]
        else:
            raise ValueError(logit_mode)
        for k in topk:
            _, topk_idx = logits.topk(k, dim=-1)  # [B, k]
            correct_k[k] += (topk_idx == y[:, None]).any(dim=-1).sum().item()
        total += y.size(0)
    return {k: 100.0 * correct_k[k] / total for k in topk}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/imagenet_dinov3_qwen3.yaml")
    p.add_argument("--resume", required=True)
    p.add_argument("--datasets", nargs="+",
                   default=["ImageNet", "ImageNetV2", "ImageNetSketch", "ImageNetA", "ImageNetR"])
    p.add_argument("--alpha", type=float, default=0.0)
    p.add_argument("--t_end", type=float, default=0.2)
    p.add_argument("--output", default=None)
    p.add_argument("opts", nargs=argparse.REMAINDER)
    args = p.parse_args()

    device = "cuda"
    set_seed(42)

    base_cfg = load_cfg(args.config, args.opts)

    results = {"resume": args.resume, "alpha": args.alpha, "t_end": args.t_end}
    for ds_name in args.datasets:
        cfg = copy.deepcopy(base_cfg)
        cfg["dataset"] = ds_name
        dataset = build_dataset(cfg)
        cfg["classnames"] = dataset.classnames
        print(f"\n=== {ds_name} ({len(dataset.test)} images, {len(dataset.classnames)} classes) ===")

        # Rebuild model with the correct classname set (fixes OOD subsets like ImageNet-A/R)
        model = FlowAdapter(cfg).to(device)
        load_checkpoint(model, args.resume)

        loader = torch.utils.data.DataLoader(
            DatasetWrapper(cfg, dataset.test, transform=model.eval_tfm, is_train=False),
            batch_size=cfg.get("batch_size", 128),
            num_workers=cfg.get("num_workers", 8),
            drop_last=False, pin_memory=True,
        )
        accs = evaluate_topk(model, loader, device, alpha=args.alpha, t_end=args.t_end)
        print(f"  alpha={args.alpha} t_end={args.t_end}  top1={accs[1]:.2f}%  top5={accs[5]:.2f}%")
        results[f"{ds_name}_top1"] = accs[1]
        results[f"{ds_name}_top5"] = accs[5]

    # OOD mean over V2/Sketch/A/R (top1 + top5)
    ood_ds = [d for d in args.datasets if d in ("ImageNetV2", "ImageNetSketch", "ImageNetA", "ImageNetR")]
    if ood_ds:
        results["ood_mean_top1"] = sum(results[f"{d}_top1"] for d in ood_ds) / len(ood_ds)
        results["ood_mean_top5"] = sum(results[f"{d}_top5"] for d in ood_ds) / len(ood_ds)

    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\n[*] saved -> {args.output}")

    print("\n=== SUMMARY ===")
    for d in args.datasets:
        print(f"  {d:18s}  top1={results[f'{d}_top1']:.2f}  top5={results[f'{d}_top5']:.2f}")
    if ood_ds:
        print(f"  {'OOD-avg':18s}  top1={results['ood_mean_top1']:.2f}  top5={results['ood_mean_top5']:.2f}")


if __name__ == "__main__":
    main()
