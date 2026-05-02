"""Precompute UNI image features + text prototypes for NCT-CRC-HE-100K (Raw/NONORM).

Outputs under --out_dir:
  labels.pt                     {'train': [N], 'test': [N], 'classnames': [9]}
  names.json                    {'train': [filename...], 'test': [...]}
  img_embed_uni.pt              {'train': [N, 1024], 'test': [N, 1024]}  (unit-norm)
  class_text_<text_tag>.pt      [9, D_text] unit-norm
  meta.json
"""
import argparse, json, os, re
from pathlib import Path
from collections import defaultdict

import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm
import timm
from torchvision import transforms as T
from transformers import AutoTokenizer, AutoModel

# CRC-100K 9 canonical classes, order matches CONCH Table 2 / Kather et al.
CLASS_CODES = ["ADI", "BACK", "DEB", "LYM", "MUC", "MUS", "NORM", "STR", "TUM"]
CLASS_NAMES = [
    "adipose",
    "background",
    "debris",
    "lymphocyte",
    "mucus",
    "smooth muscle",
    "normal colon mucosa",
    "cancer-associated stroma",
    "colorectal adenocarcinoma epithelium",
]
# CONCH-style prompts (one-line, name + short description)
CLASS_PROMPTS = [
    "adipose tissue (fat cells).",
    "background: slide background with no tissue.",
    "debris: cellular debris and necrotic tissue fragments.",
    "lymphocytes: dense aggregates of immune cells.",
    "mucus: extracellular mucus secretion.",
    "smooth muscle tissue of the bowel wall.",
    "normal colon mucosa: non-neoplastic colonic epithelium.",
    "cancer-associated stroma: desmoplastic reactive stromal tissue around tumor.",
    "colorectal adenocarcinoma epithelium: malignant glandular epithelial tumor.",
]

def _slug(s): return re.sub(r"[^a-zA-Z0-9]+", "_", s).strip("_").lower()

def list_split(root: Path):
    """Files in a flat NCT-style dir; class = prefix before first '-'."""
    files = sorted(p for p in root.iterdir() if p.suffix.lower() == ".png")
    labels, names = [], []
    c2i = {c: i for i, c in enumerate(CLASS_CODES)}
    for p in files:
        code = p.name.split("-", 1)[0]
        if code not in c2i:
            continue
        labels.append(c2i[code]); names.append(p.name)
    return files, labels, names

def iter_tar(tar_path: str):
    """Yield (name, PIL.Image) for each .png member, filtering to valid class codes."""
    import tarfile, io
    c2i = {c: i for i, c in enumerate(CLASS_CODES)}
    with tarfile.open(tar_path, "r:gz") as tf:
        for m in tf:
            if not m.isfile() or not m.name.lower().endswith(".png"):
                continue
            nm = m.name.rsplit("/", 1)[-1]
            code = nm.split("-", 1)[0]
            if code not in c2i:
                continue
            f = tf.extractfile(m)
            if f is None: continue
            img = Image.open(io.BytesIO(f.read())).convert("RGB")
            yield nm, c2i[code], img

@torch.inference_mode()
def embed_stream(model, tfm, items_iter, total, device, batch=128):
    """items_iter yields (name, label, PIL); returns (emb [N,D], labels, names)."""
    model.eval()
    embs, names, labels = [], [], []
    buf = []
    def flush():
        if not buf: return
        x = torch.stack([tfm(im) for _, _, im in buf]).to(device)
        with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
            f = model(x)
        embs.append(F.normalize(f.float(), dim=-1).cpu())
        for nm, lb, _ in buf:
            names.append(nm); labels.append(lb)
        buf.clear()
    pbar = tqdm(total=total, desc="UNI")
    for nm, lb, im in items_iter:
        buf.append((nm, lb, im))
        if len(buf) == batch:
            flush(); pbar.update(batch)
    flush(); pbar.update(len(buf)); pbar.close()
    return torch.cat(embs, 0), torch.tensor(labels), names

@torch.inference_mode()
def embed_images(model, tfm, files, device, batch=128):
    """Embed from a list of file paths (used for the extracted val split)."""
    def gen():
        c2i = {c: i for i, c in enumerate(CLASS_CODES)}
        for p in files:
            code = p.name.split("-", 1)[0]
            if code in c2i:
                yield p.name, c2i[code], Image.open(p).convert("RGB")
    return embed_stream(model, tfm, gen(), len(files), device, batch)

@torch.inference_mode()
def embed_texts(model_name, prompts, device):
    tok = AutoTokenizer.from_pretrained(model_name)
    mdl = AutoModel.from_pretrained(model_name).to(device).eval()
    batch = tok(prompts, padding=True, truncation=True, max_length=512, return_tensors="pt").to(device)
    out = mdl(**batch)
    if "Qwen" in model_name:
        lengths = batch["attention_mask"].sum(dim=1) - 1
        emb = out.last_hidden_state[torch.arange(lengths.size(0), device=device), lengths]
    else:
        mask = batch["attention_mask"].unsqueeze(-1).float()
        emb = (out.last_hidden_state * mask).sum(1) / mask.sum(1).clamp(min=1)
    return F.normalize(emb.float(), dim=-1).cpu()

def build_uni(device):
    """Standard UNI loader per README: timm ViT-L/16-224, reg4, pos embed interp."""
    model = timm.create_model(
        "vit_large_patch16_224", img_size=224, patch_size=16,
        init_values=1e-5, num_classes=0, dynamic_img_size=True,
    )
    from huggingface_hub import hf_hub_download
    tok = os.environ.get("HF_TOKEN") or open("/efs/user_folders/dnshalam/hf_cache/token").read().strip()
    path = hf_hub_download("MahmoodLab/UNI", "pytorch_model.bin", token=tok)
    sd = torch.load(path, map_location="cpu")
    model.load_state_dict(sd, strict=True)
    model = model.to(device).eval()
    tfm = T.Compose([
        T.Resize(224, interpolation=T.InterpolationMode.BICUBIC),
        T.CenterCrop(224),
        T.ToTensor(),
        T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ])
    return model, tfm

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_tar", required=True, help="train tar.gz (streamed)")
    ap.add_argument("--test_dir",  required=True, help="extracted val dir with flat PNGs")
    ap.add_argument("--out_dir",   required=True)
    ap.add_argument("--text_models", nargs="+", default=[
        "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext",
        "Qwen/Qwen3-Embedding-0.6B",
    ])
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--train_total", type=int, default=100000, help="progress-bar total; extras are fine")
    args = ap.parse_args()

    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Index + labels for test split
    test_files, y_te_list, n_te = list_split(Path(args.test_dir))
    print(f"test: {len(test_files)}")

    # Load UNI once
    model, tfm = build_uni(device)

    # Train: stream from tar (no disk writes for 100k small files)
    tr_emb, y_tr, n_tr = embed_stream(
        model, tfm, iter_tar(args.train_tar), args.train_total, device, batch=args.batch)

    # Test: read from already-extracted dir
    te_emb, y_te, _ = embed_images(model, tfm, test_files, device, batch=args.batch)

    torch.save({"train": tr_emb, "test": te_emb}, out / "img_embed_uni.pt")
    torch.save({"train": y_tr, "test": y_te, "classnames": CLASS_NAMES}, out / "labels.pt")
    (out / "names.json").write_text(json.dumps({"train": n_tr, "test": n_te}))

    del model

    # Text prototypes
    for tm in args.text_models:
        tag = _slug(tm.split("/")[-1])
        proto = embed_texts(tm, CLASS_PROMPTS, device)
        torch.save(proto, out / f"class_text_{tag}.pt")

    meta = {
        "dataset": "NCT-CRC-HE-100K (NONORM) + CRC-VAL-HE-7K",
        "classes": CLASS_NAMES, "codes": CLASS_CODES, "prompts": CLASS_PROMPTS,
        "image_encoder": "MahmoodLab/UNI (ViT-L/16, 1024-d)",
        "text_models": args.text_models,
        "counts": {"train": int(len(n_tr)), "test": int(len(n_te))},
    }
    (out / "meta.json").write_text(json.dumps(meta, indent=2))
    print("done ->", out)

if __name__ == "__main__":
    main()
