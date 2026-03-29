#!/usr/bin/env python3
import argparse, os, sys, math, json, csv
import glob
from pathlib import Path
from typing import Optional, Tuple, List, Dict, Iterator
import numpy as np
from tqdm import tqdm
import PIL
from PIL import Image
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import Dataset, DataLoader
import webdataset as wds
import torchvision.transforms as T
import timm
from transformers import AutoModel, AutoProcessor

# Assuming these exist in your src folder based on imports
from src.pretrained_encoders import clip
from src.pretrained_encoders.hf_encoders import HFImageEncoder, HFTextEncoder
from src.pretrained_encoders.openai_encoders import OpenAICLIPEncoders
from src.pretrained_encoders.open_clip_encoders import OpenCLIPEncoders


# --------------------------
# Dataset: TSV (image_path \t caption)
# --------------------------
class CC3MTSV(Dataset):
    def __init__(self, tsv_path: str, max_samples: Optional[int] = None, skip_broken: bool = True):
        self.rows = []
        with open(tsv_path, "r", encoding="utf-8") as f:
            reader = csv.reader(f, delimiter="\t")
            for i, row in enumerate(reader):
                if len(row) < 2:
                    continue
                img, cap = row[0], row[1]
                self.rows.append((img, cap))
                if max_samples and len(self.rows) >= max_samples:
                    break
        self.skip_broken = skip_broken

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        caption, img_path = self.rows[idx]
        try:
            img = Image.open(img_path).convert("RGB")
        except Exception as e:
            # print(e) # Optional: reduce verbosity
            if self.skip_broken:
                return None
            raise
        return {"image": img, "caption": caption, "path": img_path}


def collate_skip_nones(batch):
    batch = [b for b in batch if b is not None]
    if len(batch) == 0:
        return None
    return {
        "image": [b["image"] for b in batch],
        "caption": [b["caption"] for b in batch],
        "path": [b["path"] for b in batch],
    }


# ========== WebDataset loader ==========
def make_wds_iterator(
        pattern: str,
        batch_size: int,
        caption_kind: str = "auto",  # "auto" | "txt" | "json"
        json_field: str = "caption",
        image_exts: str = "jpg jpeg png webp",
        num_workers: int = 8,
        shardshuffle: int = 1000,
):
    """
    Yields dicts: {"image": [PIL,...], "caption": [str,...], "key":[...]} batched.
    """
    import json as _json
    shards = sorted(glob.glob(pattern))
    if not shards:
        raise FileNotFoundError(f"No shards match {pattern}")

    exts = set(image_exts.split())

    def pick(sample):
        # 1) pick first image
        img = None
        for k, v in sample.items():
            ext = k.split(".")[-1].lower()
            if isinstance(v, Image.Image) and ext in exts:
                img = v
                break
        if img is None:
            raise wds.SkipSample("no image")

        # 2) pick caption
        cap = None
        if caption_kind in ("auto", "txt") and "txt" in sample:
            cap = sample["txt"]
            if isinstance(cap, bytes):
                cap = cap.decode("utf-8", errors="ignore")
        if cap is None and caption_kind in ("auto", "json") and "json" in sample:
            js = sample["json"]
            if isinstance(js, bytes):
                js = _json.loads(js.decode("utf-8", errors="ignore"))
            if isinstance(js, dict) and json_field in js and isinstance(js[json_field], str):
                cap = js[json_field]
        if cap is None:
            raise wds.SkipSample("no caption")

        return {"image": img, "caption": cap, "key": sample.get("__key__", "")}

    def collate_dicts(samples):
        # samples: list[dict] from pick()
        return {
            "image": [s["image"] for s in samples],
            "caption": [s["caption"] for s in samples],
            "key": [s.get("key", "") for s in samples],
        }

    ds = (
        wds.WebDataset(
            shards,
            handler=wds.ignore_and_continue,
            shardshuffle=shardshuffle,
        )
        .decode("pil")
        .map(pick)
        .batched(batch_size, partial=False, collation_fn=collate_dicts)
    )
    return iter(wds.WebLoader(ds, num_workers=num_workers, batch_size=None))


# --------------------------
# Helpers
# --------------------------
def get_accuracy(logits):
    return (logits.argmax(-1) == torch.arange(logits.size(0)).to(logits.device)).float().mean()


def make_timm_preprocess(model_name: str, img_size: Optional[int] = None):
    # Create target encoder to get its default cfg for transforms
    tmp = timm.create_model(model_name, pretrained=True)
    cfg = tmp.default_cfg
    size = img_size or (cfg.get("input_size", (3, 224, 224))[1])
    # timm's create_transform picks good defaults for the model
    from timm.data.transforms_factory import create_transform
    transform = create_transform(
        input_size=size,
        is_training=False,
        mean=cfg.get("mean", (0.485, 0.456, 0.406)),
        std=cfg.get("std", (0.229, 0.224, 0.225)),
        interpolation=cfg.get("interpolation", "bicubic"),
        crop_pct=cfg.get("crop_pct", 0.9),
    )
    del tmp
    print(transform)
    return transform


class TargetImageEncoder(nn.Module):
    def __init__(self, model_name: str, device: torch.device):
        super().__init__()
        self.model = timm.create_model(model_name, pretrained=True)
        self.model.eval().to(device)
        self.cfg = self.model.default_cfg

        # Try to find a feature extractor head
        if hasattr(self.model, "forward_features"):
            self.forward_features = self.model.forward_features
            self.pool = getattr(self.model, "global_pool", None)
        else:
            self.forward_features = None
            self.pool = None
        print("Pool:", self.pool)

        self.img_size: int = self.cfg.get("input_size", (3, 224, 224))[1]
        # Infer output dim
        with torch.no_grad():
            dummy = torch.zeros(1, 3, self.img_size, self.img_size, device=device)
            feat = self.get_feats(dummy)
        self.dim = feat.shape[-1]

    @torch.no_grad()
    def get_feats(self, x: torch.Tensor) -> torch.Tensor:
        # Generic: forward -> global pooled feature
        if self.forward_features is not None:
            f = self.forward_features(x)
            f = self.model.forward_head(f, pre_logits=True)
        else:
            f = self.model(x)
            if f.dim() > 2:
                f = f.mean(dim=(-2, -1))
        return F.normalize(f, dim=-1)


# ========== Accumulation ==========
@torch.inference_mode()
def extract_batch_features(
        batch,
        text_encoder,
        image_encoder,
        text_pre,
        image_pre,
        device: torch.device):
    if batch is None:
        return [], []

    imgs_pil: List[Image.Image] = batch["image"]
    caps: List[str] = batch["caption"]
    if len(imgs_pil) == 0:
        return [], []

    if isinstance(text_encoder, HFTextEncoder):
        text_feat = text_encoder.encode_text(text_encoder.tokenize(caps))  # [B, D_t]
    else:
        text_feat = text_encoder.encode_text(caps)  # [B, D_t]

    if isinstance(image_encoder, HFImageEncoder):
        image_feat = image_encoder(imgs_pil)  # [B, D_x]
    else:
        # Fixed: using image_pre instead of undefined tgt_pre
        x_tensors = torch.stack([image_pre(img) for img in imgs_pil], 0).to(device)
        if hasattr(image_encoder, "encode_image"):
            image_feat = image_encoder.encode_image(x_tensors)  # [B, D_x]
        else:
            image_feat = image_encoder.get_feats(x_tensors)  # [B, D_x]

    return F.normalize(image_feat, dim=-1), F.normalize(text_feat, dim=-1)


@torch.inference_mode()
def accumulate_from_iterator(
        iterator: Iterator[Dict[str, List]],
        text_encoder,
        image_encoder,
        text_pre,
        image_pre,
        device: torch.device,
        log_every: int = 50,
):
    # Fixed: Use passed encoder objects instead of global clip_enc/tgt_enc
    D_t = text_encoder.text_dim
    D_x = image_encoder.dim

    M = torch.zeros(D_t, D_x, device=device)

    num_pairs = 0

    for bi, batch in enumerate(iterator):
        if batch is None:
            continue
        imgs_pil: List[Image.Image] = batch["image"]
        caps: List[str] = batch["caption"]
        if len(imgs_pil) == 0:
            continue

        # text encoding
        if isinstance(text_encoder, HFTextEncoder):
            text_feat = text_encoder.encode_text(text_encoder.tokenize(caps))  # [B, D_t]
        else:
            text_feat = text_encoder.encode_text(caps)  # [B, D_t]

        # image encodings
        if isinstance(image_encoder, HFImageEncoder):
            img_feat = image_encoder(imgs_pil)  # [B, D_x]
        else:
            # Fixed: Use image_pre instead of undefined tgt_pre
            x_tensors = torch.stack([image_pre(img) for img in imgs_pil], 0).to(device)
            if hasattr(image_encoder, "encode_image"):
                img_feat = image_encoder.encode_image(x_tensors)  # [B, D_x]
            else:
                img_feat = image_encoder.get_feats(x_tensors)  # [B, D_x]

        # Fixed: dim=-1 (was dim-1)
        text_feat = F.normalize(text_feat, dim=-1)
        img_feat = F.normalize(img_feat, dim=-1)

        M += text_feat.t() @ img_feat
        # Fixed: Use actual batch size instead of undefined 'x'
        num_pairs += text_feat.size(0)

    # Removed referencing undefined 'weight_mode', 'weight_tau', 'mean_w_accum'
    stats = {
        "num_pairs": num_pairs,
    }
    return M, stats


def evaluate_proj_text(test_data, proj):
    img_features = test_data["image"]
    txt_features = test_data["text"] @ proj.float().to(test_data["image"].device)
    logits = 100. * img_features @ txt_features.t()
    score = get_accuracy(logits)
    print(f"Validation Accuracy={score * 100.:.2f}%")
    return


def rectangular_op_from_cov(M: torch.Tensor) -> torch.Tensor:
    """
    Solve min ||T W - X||_F with W having orthonormal columns when possible.
    For rectangular case, W = U V^T with SVD(M) = U Σ V^T.
    """
    U, S, Vh = torch.linalg.svd(M, full_matrices=False)  # U: [D_t, r], Vh: [r, D_x]
    W = U @ Vh  # [D_t, D_x]
    return W


# --------------------------
# Main
# --------------------------

def main():
    p = argparse.ArgumentParser("Global Orthogonal Procrustes (GOP) from CC3M-like TSV")
    # webdataset path
    p.add_argument("--wds_pattern", type=str, default='/root/datasets/cc3m/cc3m/00*.tar',
                   help="Glob pattern for webdataset shards, e.g. /data/cc3m/00*.tar")
    p.add_argument("--wds_caption_kind", type=str, default="auto", choices=["auto", "txt", "json"])
    p.add_argument("--wds_json_field", type=str, default="caption")
    p.add_argument("--image_exts", type=str, default="jpg jpeg png webp")
    p.add_argument("--tsv", type=str, required=False, help="TSV file: <image_path>\\t<caption>")
    p.add_argument("--out", type=str, required=True, help="Path to save .pth (contains W and metadata)")
    p.add_argument("--text_model", type=str, default="openai::ViT-B/16", help="Text Encoder")
    p.add_argument("--image_model", type=str, default="hf::facebook/dinov2-base", help="Image Encoder")
    p.add_argument("--batch_size", type=int, default=1024)
    p.add_argument("--num_workers", type=int, default=8)
    p.add_argument("--max_samples", type=int, default=None, help="Optional cap on dataset size")
    p.add_argument("--max_batches", type=int, default=None, help="Optional cap on num batches")
    p.add_argument("--val_batches", type=int, default=0, help="Optional validation batches")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    # Encoders + preprocess
    print("Loading encoders...")

    # Selecting text encoder
    txt_src, args.text_model = args.text_model.split('::')
    txt_src = txt_src.lower()
    assert txt_src in ["openai", "hf", "openclip"]

    if txt_src == "openai":
        print(f"Loading (OpenAI) CLIP: {args.text_model}...")
        text_encoder = OpenAICLIPEncoders(args.text_model, device=device)
    elif txt_src == "hf":
        text_encoder = HFTextEncoder(args.text_model, device=device)
    elif txt_src == "openclip":
        pre = 'laion2b_s32b_b82k' if args.text_model == 'ViT-L-14' else 'openai'
        print(f"Loading (Open-CLIP) CLIP: {args.text_model} ({pre})...")
        text_encoder = OpenCLIPEncoders(args.text_model, pretrained=pre, device=device)
    else:
        raise ValueError("Unknown text src:", txt_src)
    text_pre = text_encoder.preprocess

    # Selecting image encoder
    img_src, args.image_model = args.image_model.split('::')
    img_src = img_src.lower()
    assert img_src in ["openai", "hf", "openclip", "timm"]

    if img_src == "openai":
        image_encoder = OpenAICLIPEncoders(args.image_model, device=device)
        image_encoder.dim = image_encoder.image_dim
        image_pre = image_encoder.preprocess
    elif img_src == "hf":
        image_encoder = HFImageEncoder(args.image_model).to(device)
        image_pre = image_encoder.image_processor
    elif img_src == "openclip":
        image_encoder = OpenCLIPEncoders(args.image_model, pretrained='laion2b_s32b_b82k', device=device)
        if hasattr(image_encoder, 'image_dim'):
            image_encoder.dim = image_encoder.image_dim
        else:
            # fallback or trust it exists
            image_encoder.dim = 1024 if 'L-14' in args.image_model else 512
        image_pre = image_encoder.preprocess
    elif img_src == "timm":
        image_encoder = TargetImageEncoder(args.image_model, device=device)
        image_pre = make_timm_preprocess(args.image_model)
    else:
        raise ValueError("Unknown image src:", img_src)

    img_dim = image_encoder.dim
    txt_dim = text_encoder.text_dim
    print(f"Text Encoder Dim: {txt_dim}, Image Encoder Dim: {img_dim}")

    # Iterator over data
    if args.wds_pattern is not None:
        it = make_wds_iterator(
            args.wds_pattern, args.batch_size,
            caption_kind=args.wds_caption_kind,
            json_field=args.wds_json_field,
            image_exts=args.image_exts,
            num_workers=args.num_workers,
        )
    elif args.tsv is not None:
        # Minimal local-TSV fallback (paths to local files)
        def tsv_iterator(tsv_path, bs):
            with open(tsv_path, "r", encoding="utf-8") as f:
                rows = []
                for line in f:
                    if "\t" not in line:
                        continue
                    path, cap = line.rstrip("\n").split("\t", 1)
                    try:
                        img = Image.open(path).convert("RGB")
                    except Exception:
                        continue
                    rows.append({"image": img, "caption": cap})
                    if len(rows) == bs:
                        yield {"image": [r["image"] for r in rows],
                               "caption": [r["caption"] for r in rows]}
                        rows = []
                if rows:
                    yield {"image": [r["image"] for r in rows],
                           "caption": [r["caption"] for r in rows]}

        it = tsv_iterator(args.tsv, args.batch_size)
    else:
        print("Provide either --wds-pattern (recommended) or --tsv.", file=sys.stderr)
        sys.exit(1)

    # Accumulate cross-covariance
    val_data = {"image": [], "text": []}
    M = torch.zeros(txt_dim, img_dim, device=device)

    seen = 0
    for bi, batch in enumerate(tqdm(it, desc="Accumulating M...")):
        if args.max_batches is not None and bi >= args.max_batches > 0:
            if bi - args.max_batches < args.val_batches:
                img_f, txt_f = extract_batch_features(
                    batch, text_encoder, image_encoder, text_pre, image_pre, device
                )
                val_data["image"].extend(img_f.cpu())
                val_data["text"].extend(txt_f.cpu())
                continue
            else:
                break

        # reuse the accumulator for a single batch at a time
        Mi, stats = accumulate_from_iterator(
            [batch], text_encoder, image_encoder, text_pre, image_pre, device, log_every=1
        )
        M += Mi
        seen += stats["num_pairs"]
        if (bi + 1) % 20 == 0:
            print(f"[progress] shards/batches processed: {bi + 1}, pairs={seen}")

    print(f"Total pairs used: {seen}")

    # Solve OP
    print("Solving rectangular OP ...")
    W = rectangular_op_from_cov(M)

    if args.val_batches > 1 and len(val_data["image"]) > 0:
        torch.cuda.empty_cache()
        print(f"Validation (batches={args.val_batches})...")
        val_data = {k: torch.stack(v) for k, v in val_data.items()}
        evaluate_proj_text(val_data, proj=W)

    # Save
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "W": W.cpu(),
        "t_dim": int(W.shape[0]),
        "x_dim": int(W.shape[1]),
        "num_pairs": int(seen),
        "image_model": args.image_model,
        "text_model": args.text_model,
        "seed": args.seed,
    }
    torch.save(payload, out)
    print(f"[✓] Saved W to {out}")
    print("Meta:", json.dumps({k: v for k, v in payload.items() if k != 'W'}, indent=2))


if __name__ == "__main__":
    main()