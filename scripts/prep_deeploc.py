"""
Precompute DeepLoc 2.0 (multi-label) embeddings and class-text prototypes.

Outputs under --out_dir:
  meta.json
  labels_multihot.pt                 {'train'|'validation'|'test': [N,10] float, 'names': {...}}
  seq_embed_<esm_tag>.pt             {'train'|'validation'|'test': [N,D] float (unit-norm), 'names': {...}}
  class_text_<text_tag>.pt           [10, D_text] float (unit-norm)
"""
import argparse, json, os, re
from pathlib import Path
import torch
import torch.nn.functional as F
from tqdm import tqdm
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModel

DEEPLOC_CLASSES = [
    "Nucleus", "Cytoplasm", "Extracellular", "Mitochondrion", "Cell membrane",
    "Endoplasmic reticulum", "Plastid", "Golgi apparatus",
    "Lysosome / Vacuole", "Peroxisome",
]

# Short natural-language prompts (name + one-line definition).
CLASS_PROMPTS = [
    "Nucleus: the organelle containing the cell's chromosomal DNA and the site of transcription.",
    "Cytoplasm: the aqueous interior of the cell excluding the nucleus and membrane-bound organelles.",
    "Extracellular: secreted or located outside the cell, including the extracellular matrix.",
    "Mitochondrion: the organelle that carries out oxidative phosphorylation and ATP production.",
    "Cell membrane: the plasma membrane bilayer enclosing the cell.",
    "Endoplasmic reticulum: the membrane network for protein folding and lipid synthesis.",
    "Plastid: a plant-cell organelle such as a chloroplast that performs photosynthesis or storage.",
    "Golgi apparatus: the organelle that modifies, sorts and packages proteins for secretion.",
    "Lysosome or vacuole: the acidic organelle for degradation and storage.",
    "Peroxisome: the organelle for fatty-acid oxidation and hydrogen-peroxide metabolism.",
]

def _slug(s: str) -> str:
    return re.sub(r"[^a-zA-Z0-9]+", "_", s).strip("_").lower()

def parse_multihot(label_str: str, num_classes: int = 10) -> torch.Tensor:
    y = torch.zeros(num_classes, dtype=torch.float32)
    for tok in str(label_str).split(","):
        tok = tok.strip()
        if tok == "":
            continue
        y[int(tok)] = 1.0
    return y

@torch.inference_mode()
def embed_sequences(model, tokenizer, seqs, device, batch_size=8, max_len=1022):
    model.eval()
    out = []
    for i in tqdm(range(0, len(seqs), batch_size), desc="esm"):
        batch = seqs[i:i+batch_size]
        tok = tokenizer(batch, padding=True, truncation=True, max_length=max_len, return_tensors="pt").to(device)
        h = model(**tok).last_hidden_state                  # [B,L,D]
        mask = tok["attention_mask"].unsqueeze(-1).float()
        emb = (h * mask).sum(1) / mask.sum(1).clamp(min=1)  # mean-pool, mask-aware
        emb = F.normalize(emb.float(), dim=-1)
        out.append(emb.cpu())
    return torch.cat(out, dim=0)

@torch.inference_mode()
def embed_texts(model_name, prompts, device):
    tok = AutoTokenizer.from_pretrained(model_name)
    mdl = AutoModel.from_pretrained(model_name, torch_dtype=torch.float32).to(device).eval()
    batch = tok(prompts, padding=True, truncation=True, max_length=512, return_tensors="pt").to(device)
    out = mdl(**batch)
    if "Qwen" in model_name:
        # last-token pool (the HF encoder class in FSA uses this for Qwen)
        lengths = batch["attention_mask"].sum(dim=1) - 1
        emb = out.last_hidden_state[torch.arange(lengths.size(0), device=device), lengths]
    else:
        # mean pool
        mask = batch["attention_mask"].unsqueeze(-1).float()
        emb = (out.last_hidden_state * mask).sum(1) / mask.sum(1).clamp(min=1)
    emb = F.normalize(emb.float(), dim=-1)
    return emb.cpu()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--esm_model", default="facebook/esm2_t30_150M_UR50D")
    ap.add_argument("--text_models", nargs="+",
                    default=["microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext",
                             "Qwen/Qwen3-Embedding-8B"])
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    device = args.device if torch.cuda.is_available() else "cpu"

    # ---- data ----
    ds = load_dataset("AI4Protein/DeepLoc2Multi")
    splits = {"train": ds["train"], "validation": ds["validation"], "test": ds["test"]}

    # ---- labels ----
    labels = {k: torch.stack([parse_multihot(e["label"]) for e in d]) for k, d in splits.items()}
    names  = {k: [e["name"] for e in d] for k, d in splits.items()}
    torch.save({**labels, "names": names}, out / "labels_multihot.pt")

    # ---- protein embeddings ----
    esm_tag = _slug(args.esm_model.split("/")[-1])
    esm_tok = AutoTokenizer.from_pretrained(args.esm_model)
    esm_mdl = AutoModel.from_pretrained(args.esm_model).to(device).eval()
    seq_out = {}
    for k, d in splits.items():
        seq_out[k] = embed_sequences(esm_mdl, esm_tok, list(d["aa_seq"]), device, batch_size=args.batch_size)
    seq_out["names"] = names
    torch.save(seq_out, out / f"seq_embed_{esm_tag}.pt")
    del esm_mdl

    # ---- class text prototypes ----
    for tm in args.text_models:
        tag = _slug(tm.split("/")[-1])
        proto = embed_texts(tm, CLASS_PROMPTS, device)    # [10, D]
        torch.save(proto, out / f"class_text_{tag}.pt")

    meta = {
        "dataset": "AI4Protein/DeepLoc2Multi",
        "classes": DEEPLOC_CLASSES,
        "class_prompts": CLASS_PROMPTS,
        "num_classes": 10,
        "esm_model": args.esm_model,
        "text_models": args.text_models,
        "counts": {k: len(v) for k, v in splits.items()},
    }
    (out / "meta.json").write_text(json.dumps(meta, indent=2))
    print("done ->", out)

if __name__ == "__main__":
    main()
