"""Precompute CLIP zero-shot logits for a dataset.

For a given CLIP model (default: ViT-B/16 OpenAI), encodes the test images and
the class-name prompt ensemble, and saves the unnormalized logit matrix
[N, C] along with labels and class ordering to a .pth file.

Designed to be combined with FSA's MT logits at inference time as a stabilizing
zero-shot prior:
    logits_final = (1 - α) * MT + α * clip_zs_logits

Usage:
    python compute_clip_zs_logits.py --dataset ImageNet \
        --out output/clip_zs/imagenet.pth
"""
import argparse
import copy
import os
from pathlib import Path

import open_clip
import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.datasets.base_dataset import build_dataset, DatasetWrapper


# Same prompt ensemble as FSA's TextEncoder uses (prompt_ensembling n=7).
PROMPTS_IMAGENET = [
    "itap of a {}.",
    "a bad photo of the {}.",
    "a origami {}.",
    "a photo of the large {}.",
    "a {} in a video game.",
    "art of the {}.",
    "a photo of the small {}.",
]

PROMPTS_BIOMED = [
    "a medical image of {}.",
    "a histopathology image of {}.",
    "a radiology image showing {}.",
    "a clinical photograph of {}.",
    "{}",
]

PROMPTS_SIMPLE = ["{}"]

PROMPT_STYLES = {
    "imagenet": PROMPTS_IMAGENET,
    "biomed": PROMPTS_BIOMED,
    "simple": PROMPTS_SIMPLE,
}


@torch.no_grad()
def encode_text_ensemble(model, tokenizer, classnames, device, prompt_templates):
    """Return [C, D] normalized text features averaged over prompt_templates."""
    feats = []
    for cn in classnames:
        prompts = [p.format(cn) for p in prompt_templates]
        tokens = tokenizer(prompts).to(device)
        f = model.encode_text(tokens)
        f = F.normalize(f.float(), dim=-1)
        f = f.mean(dim=0)
        f = F.normalize(f, dim=-1)
        feats.append(f)
    return torch.stack(feats, dim=0)  # [C, D]


@torch.no_grad()
def encode_images_and_logits(model, loader, text_feats, logit_scale, device):
    """Return (logits [N,C], labels [N])."""
    logits_list = []
    labels_list = []
    for batch in tqdm(loader, desc="encode imgs", dynamic_ncols=True):
        x = batch["img"].to(device, non_blocking=True)
        y = batch["label"]
        img_feats = F.normalize(model.encode_image(x).float(), dim=-1)
        logits = logit_scale * (img_feats @ text_feats.t())
        logits_list.append(logits.cpu())
        labels_list.append(y)
    return torch.cat(logits_list, dim=0), torch.cat(labels_list, dim=0)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True,
                   help="ImageNet | ImageNetV2 | ImageNetSketch | ImageNetA | ImageNetR | ...")
    p.add_argument("--clip_model", default="ViT-B-16")
    p.add_argument("--clip_pretrained", default="openai")
    p.add_argument("--config", default="configs/imagenet_dinov3_qwen3.yaml")
    p.add_argument("--root", default="/efs/user_folders/dnshalam/datasets")
    p.add_argument("--out", required=True)
    p.add_argument("--prompt_style", default="imagenet",
                   choices=["imagenet", "biomed", "simple"])
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--num_workers", type=int, default=8)
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"[*] Loading CLIP {args.clip_model} / {args.clip_pretrained}")
    model, _, preprocess = open_clip.create_model_and_transforms(
        args.clip_model, pretrained=args.clip_pretrained,
        force_quick_gelu=args.clip_pretrained == "openai",
    )
    model = model.eval().to(device)
    tokenizer = open_clip.get_tokenizer(args.clip_model)
    logit_scale = model.logit_scale.exp().item() if hasattr(model, "logit_scale") else 100.0
    print(f"[*] logit_scale = {logit_scale:.2f}")

    with open(args.config) as f:
        base_cfg = yaml.safe_load(f)
    cfg = copy.deepcopy(base_cfg)
    cfg["dataset"] = args.dataset
    cfg["root"] = args.root
    cfg["shots"] = 16  # shots value doesn't affect test set
    cfg["seed"] = 42
    cfg["subsample_classes"] = "all"
    cfg["batch_size"] = args.batch_size
    cfg["num_workers"] = args.num_workers

    dataset = build_dataset(cfg)
    classnames = list(dataset.classnames)
    print(f"[+] {args.dataset}: {len(dataset.test)} test images, {len(classnames)} classes")

    # Encode text
    print("[*] Encoding text prompts...")
    prompt_templates = PROMPT_STYLES[args.prompt_style]
    print(f"[*] Using {len(prompt_templates)} prompt templates ({args.prompt_style})")
    text_feats = encode_text_ensemble(model, tokenizer, classnames, device, prompt_templates)
    print(f"[+] text feats shape: {text_feats.shape}")

    # Encode images + compute logits
    loader = DataLoader(
        DatasetWrapper(cfg, dataset.test, transform=preprocess, is_train=False),
        batch_size=args.batch_size, num_workers=args.num_workers,
        shuffle=False, drop_last=False, pin_memory=True,
    )
    logits, labels = encode_images_and_logits(model, loader, text_feats, logit_scale, device)

    # Quick sanity: top-1 accuracy
    pred = logits.argmax(dim=-1)
    top1 = (pred == labels).float().mean().item()
    print(f"[+] CLIP ZS top-1 on {args.dataset}: {top1*100:.2f}%")

    out = {
        "dataset": args.dataset,
        "clip_model": args.clip_model,
        "clip_pretrained": args.clip_pretrained,
        "logit_scale": logit_scale,
        "logits": logits,        # [N, C]
        "labels": labels,        # [N]
        "classnames": classnames,  # ordered same as logits columns
        "top1_zs": top1,
    }
    Path(os.path.dirname(args.out)).mkdir(parents=True, exist_ok=True)
    torch.save(out, args.out)
    print(f"[*] Saved -> {args.out}")


if __name__ == "__main__":
    main()
