"""Run FSA + CLIP-ZS-Mix across a list of α values for one checkpoint+dataset.

Efficient variant: encodes each test image once, then evaluates multiple α
values from the already-computed MT logits + precomputed CLIP ZS logits.
"""
import argparse
import copy
import json
from pathlib import Path

import torch
import yaml

from src.datasets.base_dataset import build_dataset, DatasetWrapper
from src.engine import evaluate_multi_alpha
from src.model import FlowAdapter
from src.utils import load_checkpoint, set_seed


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/imagenet_dinov3_qwen3.yaml")
    p.add_argument("--resume", required=True)
    p.add_argument("--dataset", required=True)
    p.add_argument("--clip_zs_file", required=True)
    p.add_argument("--alphas", nargs="+", default=["0.0", "0.25", "0.5", "0.7", "0.9", "1.0"])
    p.add_argument("--mix_space", default="softmax", choices=["softmax", "logit"])
    p.add_argument("--base", default="MT", choices=["MT", "ZS"])
    p.add_argument("--t_end", type=float, default=0.2)
    p.add_argument("--output", required=True,
                   help="Path to save JSON with per-alpha accuracies.")
    p.add_argument("opts", nargs=argparse.REMAINDER)
    return p.parse_args()


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


def main():
    args = get_args()
    alphas = [float(a) for a in args.alphas]

    cfg = load_cfg(args.config, args.opts)
    cfg["dataset"] = args.dataset
    set_seed(cfg.get("seed", 42))
    device = "cuda"

    dataset = build_dataset(copy.deepcopy(cfg))
    cfg["classnames"] = dataset.classnames
    print(f"[*] {args.dataset}: {len(dataset.test)} images, {len(dataset.classnames)} classes")

    model = FlowAdapter(cfg).to(device)
    load_checkpoint(model, args.resume)

    loader = torch.utils.data.DataLoader(
        DatasetWrapper(cfg, dataset.test, transform=model.eval_tfm, is_train=False),
        batch_size=cfg.get("batch_size", 128),
        num_workers=cfg.get("num_workers", 8),
        drop_last=False, pin_memory=True,
    )

    zs = torch.load(args.clip_zs_file, map_location="cpu", weights_only=False)
    clip_logits = zs["logits"].float()
    print(f"[*] CLIP ZS logits: {clip_logits.shape}")

    accs = evaluate_multi_alpha(
        model, loader, device, alphas,
        t_end=args.t_end, external_zs_logits=clip_logits,
        mix_space=args.mix_space,
        base=args.base,
    )

    out = {
        "resume": args.resume, "dataset": args.dataset, "shots": cfg.get("shots"),
        "seed": cfg.get("seed"), "t_end": args.t_end, "mix_space": args.mix_space, "base": args.base,
        "accs": accs,
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(out, f, indent=2)

    print("\n=== SUMMARY ===")
    for a in alphas:
        print(f"  alpha={a:<4}  acc={accs[a]:.2f}%")


if __name__ == "__main__":
    main()
