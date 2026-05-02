#!/usr/bin/env python
"""Shard-parallel feature extraction for PMC-OA.

Usage per rank:
  python prep_pmc_oa_gop.py --rank R --world W --n_samples 500000 --stage {img,txt_bmc,txt_qwen}

After all shards for a stage complete, run `--stage merge` (single process) to
concatenate and (for non-img stages) fit the GOP.
"""
import argparse, json, os, time
from pathlib import Path
import torch
import torch.nn.functional as F
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
from torch.utils.data import Dataset, DataLoader

DATA_ROOT = Path("/efs/user_folders/dnshalam/datasets/pmc_oa")
IMG_ZIP   = DATA_ROOT / "images.zip"
IMG_SUBDIR_IN_ZIP = "caption_T060_filtered_top4_sep_v0_subfigures"
OUT_ROOT  = Path("/efs/user_folders/dnshalam/work/FewShotAlignment/data/pmc_oa")

def load_rows(n_samples):
    rows = []
    with open(DATA_ROOT / "pmc_oa.jsonl") as f:
        for line in f:
            rows.append(json.loads(line))
            if len(rows) >= n_samples: break
    return rows

def shard(rows, rank, world):
    per = (len(rows) + world - 1) // world
    lo, hi = rank * per, min((rank+1) * per, len(rows))
    return lo, hi, rows[lo:hi]

# ---------- Image stage (zip-reading + torchvision transforms, fast path) ----------
import zipfile, io
import torchvision.transforms as _T

# DINOv3 pretraining used ImageNet-style mean/std at 224
_IMG_TF = _T.Compose([
    _T.Resize(256, antialias=True),
    _T.CenterCrop(224),
    _T.ToTensor(),
    _T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

class PmcImg(Dataset):
    def __init__(self, rows, _proc_ignored=None):
        self.rows = rows
        self._zf = None
    def _zip(self):
        if self._zf is None:
            self._zf = zipfile.ZipFile(IMG_ZIP, 'r')
        return self._zf
    def __len__(self): return len(self.rows)
    def __getitem__(self, i):
        name = f"{IMG_SUBDIR_IN_ZIP}/{self.rows[i]['image']}"
        try:
            with self._zip().open(name) as f:
                data = f.read()
            im = Image.open(io.BytesIO(data)).convert('RGB')
            return _IMG_TF(im)
        except Exception:
            return torch.zeros(3, 224, 224)

def stage_img(args):
    from transformers import AutoModel
    device = torch.device("cuda")
    lo, hi, rows = shard(load_rows(args.n_samples), args.rank, args.world)
    model = AutoModel.from_pretrained("facebook/dinov3-vitb16-pretrain-lvd1689m").to(device).eval()
    dl = DataLoader(PmcImg(rows), batch_size=128, num_workers=8, pin_memory=True)
    feats = []
    t0 = time.time()
    with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
        for i, x in enumerate(dl):
            x = x.to(device, non_blocking=True)
            out = model(x).last_hidden_state[:, 0, :]
            feats.append(F.normalize(out.float().cpu(), dim=-1))
            if i % 20 == 0:
                n = (i+1) * 128
                print(f"[img rank={args.rank}] {n}/{len(rows)} ({n/(time.time()-t0):.0f} img/s)", flush=True)
    V = torch.cat(feats, 0)[:len(rows)]
    out_fp = OUT_ROOT / f"img_dv3_n{args.n_samples}_r{args.rank}of{args.world}.pt"
    torch.save(V, out_fp)
    print(f"[save] {out_fp} {V.shape}")

# ---------- Text stage: BMC ----------
def stage_txt_bmc(args):
    import open_clip
    device = torch.device("cuda")
    lo, hi, rows = shard(load_rows(args.n_samples), args.rank, args.world)
    model, _, _ = open_clip.create_model_and_transforms(
        "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224")
    tok_fn = open_clip.get_tokenizer("hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224")
    model = model.to(device).eval()
    BS = 256
    feats = []
    t0 = time.time()
    with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
        for i in range(0, len(rows), BS):
            caps = [r["caption"][:512] for r in rows[i:i+BS]]
            tok = tok_fn(caps).to(device)
            out = model.encode_text(tok)
            feats.append(F.normalize(out.float().cpu(), dim=-1))
            if (i // BS) % 40 == 0:
                n = i + len(caps)
                print(f"[bmc rank={args.rank}] {n}/{len(rows)} ({n/(time.time()-t0):.0f} txt/s)", flush=True)
    T = torch.cat(feats, 0)
    out_fp = OUT_ROOT / f"txt_bmc_n{args.n_samples}_r{args.rank}of{args.world}.pt"
    torch.save(T, out_fp)
    print(f"[save] {out_fp} {T.shape}")

# ---------- Text stage: Qwen3 ----------
def stage_txt_qwen(args):
    from transformers import AutoTokenizer, AutoModel
    device = torch.device("cuda")
    lo, hi, rows = shard(load_rows(args.n_samples), args.rank, args.world)
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-Embedding-0.6B")
    model = AutoModel.from_pretrained("Qwen/Qwen3-Embedding-0.6B",
                                      torch_dtype=torch.bfloat16).to(device).eval()
    def last_token_pool(h, mask):
        left = (mask[:, -1].sum() == mask.shape[0])
        if left: return h[:, -1]
        lens = mask.sum(dim=1) - 1
        return h[torch.arange(h.shape[0], device=h.device), lens]
    BS = 32
    feats = []
    t0 = time.time()
    with torch.no_grad():
        for i in range(0, len(rows), BS):
            caps = [r["caption"][:512] for r in rows[i:i+BS]]
            tok = tokenizer(caps, padding=True, truncation=True,
                            max_length=512, return_tensors="pt").to(device)
            out = model(**tok).last_hidden_state
            pooled = last_token_pool(out, tok["attention_mask"])
            feats.append(F.normalize(pooled.float().cpu(), dim=-1))
            if (i // BS) % 40 == 0:
                n = i + len(caps)
                print(f"[qwen rank={args.rank}] {n}/{len(rows)} ({n/(time.time()-t0):.0f} txt/s)", flush=True)
    T = torch.cat(feats, 0)
    out_fp = OUT_ROOT / f"txt_qwen_n{args.n_samples}_r{args.rank}of{args.world}.pt"
    torch.save(T, out_fp)
    print(f"[save] {out_fp} {T.shape}")

# ---------- Merge + GOP fit ----------
def fit_gop(V, T):
    """W* = argmin ||V W - T|| s.t. W orthogonal, from SVD of V^T T."""
    M = V.t() @ T  # [dV, dT]
    U, S, Vh = torch.linalg.svd(M, full_matrices=False)
    W = U @ Vh
    print(f"[gop] singular values: mean={S.mean():.3f} max={S.max():.3f} "
          f"min={S.min():.3f}")
    return W

def stage_merge(args):
    world = args.world
    # Images (required)
    V_parts = []
    for r in range(world):
        fp = OUT_ROOT / f"img_dv3_n{args.n_samples}_r{r}of{world}.pt"
        if not fp.exists():
            print(f"[miss] {fp}"); return
        V_parts.append(torch.load(fp))
    V = torch.cat(V_parts, 0)
    print(f"[merged] img {V.shape}")
    torch.save(V, OUT_ROOT / f"img_dv3_n{args.n_samples}.pt")

    # Text variants (optional)
    for tag in ("bmc", "qwen"):
        T_parts = []
        for r in range(world):
            fp = OUT_ROOT / f"txt_{tag}_n{args.n_samples}_r{r}of{world}.pt"
            if not fp.exists():
                print(f"[skip {tag}] {fp} missing"); T_parts = None; break
            T_parts.append(torch.load(fp))
        if T_parts is None: continue
        T = torch.cat(T_parts, 0)
        print(f"[merged] txt_{tag} {T.shape}")
        torch.save(T, OUT_ROOT / f"txt_{tag}_n{args.n_samples}.pt")
        W = fit_gop(V, T)
        gop_fp = OUT_ROOT / f"gop_dv3_to_{tag}_n{args.n_samples}.pt"
        torch.save(W, gop_fp)
        print(f"[save] {gop_fp} {W.shape}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["img", "txt_bmc", "txt_qwen", "merge"], required=True)
    ap.add_argument("--rank", type=int, default=0)
    ap.add_argument("--world", type=int, default=8)
    ap.add_argument("--n_samples", type=int, default=500000)
    args = ap.parse_args()
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    {"img": stage_img, "txt_bmc": stage_txt_bmc,
     "txt_qwen": stage_txt_qwen, "merge": stage_merge}[args.stage](args)

if __name__ == "__main__":
    main()
