from typing import List, Tuple

import PIL
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer, AutoProcessor, AutoImageProcessor
import torchvision.transforms as T
from torchvision.transforms import InterpolationMode


def _resolve_sizes(proc) -> Tuple[int, int]:
    """Return (resize_short, crop_size) from a HF image processor."""
    # HF processors vary: some have size={"shortest_edge": X}, some size=(H,W), some crop_size=...
    resize_short = None
    crop = None

    if hasattr(proc, "size"):
        sz = proc.size
        if isinstance(sz, dict):
            resize_short = sz.get("shortest_edge", sz.get("height", None))
            if resize_short is None and "width" in sz:
                resize_short = sz["width"]
        elif isinstance(sz, (tuple, list)):
            # (height, width)
            resize_short = max(sz)
        elif isinstance(sz, int):
            resize_short = sz

    if hasattr(proc, "crop_size"):
        cs = proc.crop_size
        crop = cs.get("height", None) if isinstance(cs, dict) else (cs if isinstance(cs, int) else None)

    # Sensible fallbacks
    if crop is None:
        crop = 224
    if resize_short is None:
        resize_short = int(round(crop / proc.crop_pct)) if hasattr(proc, "crop_pct") else 256

    return int(resize_short), int(crop)

def make_transforms_hf(model_name: str, rrc_scale: Tuple[float,float]=(0.5, 1.0), hflip_p: float=0.5):
    proc = AutoProcessor.from_pretrained(model_name)
    resize_short, crop = _resolve_sizes(proc)
    mean = getattr(proc, "image_mean", (0.485, 0.456, 0.406))
    std  = getattr(proc, "image_std",  (0.229, 0.224, 0.225))

    # Mild augmentation, but preserve HF normalization + geometry
    train_tfm = T.Compose([
        T.RandomResizedCrop(size=224, scale=rrc_scale, interpolation=T.InterpolationMode.BICUBIC),
        T.RandomHorizontalFlip(p=hflip_p),
        T.ToTensor(),
        T.Normalize(mean=mean, std=std),
    ])
    eval_tfm = T.Compose([
        T.Resize(resize_short, interpolation=InterpolationMode.BICUBIC, antialias=True),
        T.CenterCrop(crop),
        T.ToTensor(),
        T.Normalize(mean=mean, std=std),
    ])
    return train_tfm, eval_tfm


# Text encoder
class HFTextEncoder(nn.Module):
    """
    HFTextEncoder class.
    Load pretrained models via Huggingface,
    and extract features for different modalities.
    """

    def __init__(self, txt_model: str, device: torch.device = torch.device("cuda")):
        super().__init__()
        self.device = device
        self.txt_model_name = txt_model
        try:
            self.txt_model = AutoModel.from_pretrained(txt_model)
            self.tokenizer = AutoTokenizer.from_pretrained(txt_model)
        except:
            from sentence_transformers import SentenceTransformer
            self.txt_model = SentenceTransformer(txt_model)
            self.tokenizer = None
            print("=> SentenceTransformer Text FeatureExtractor loaded.")

        self.txt_model.to(self.device)
        # Freeze
        for p in self.parameters():
            p.requires_grad_(False)

    def tokenize(self, text: List[str]):
        tokenizer_out = self.tokenizer(text, padding=True, truncation=True, return_tensors="pt").to(self.device)
        return tokenizer_out

    def encode_text(self, tokenizer_out, normalize=False):
        device = torch.device("cuda")
        if self.tokenizer is None:
            text_features = self.txt_model.encode(tokenizer_out)
            text_features = torch.from_numpy(text_features).float().to(device)
            return F.normalize(text_features, dim=1) if normalize else text_features

        if hasattr(self.txt_model, 'get_text_features'):
            text_features = self.txt_model.get_text_features(**tokenizer_out)
        elif 'Qwen' in self.txt_model_name:
            def last_token_pool(last_hidden_states, attention_mask):
                left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
                if left_padding:
                    return last_hidden_states[:, -1]
                else:
                    sequence_lengths = attention_mask.sum(dim=1) - 1
                    batch_size = last_hidden_states.shape[0]
                    return last_hidden_states[
                        torch.arange(batch_size, device=last_hidden_states.device), sequence_lengths]
            model_output = self.txt_model(**tokenizer_out)
            text_features = last_token_pool(model_output.last_hidden_state, tokenizer_out['attention_mask'])
        else:
            # Mean Pooling function
            def average_pool(last_hidden_states, attention_mask):
                last_hidden = last_hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)
                return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]
            model_output = self.txt_model(**tokenizer_out)
            text_features = average_pool(model_output.last_hidden_state, tokenizer_out['attention_mask'])

        return F.normalize(text_features, dim=1) if normalize else text_features

    @property
    def text_dim(self):
        # CLIP text head dim equals model.ln_final.weight.shape[0] for ViTs; simpler: probe
        return self.encode_text(self.tokenize(["a"]).to(self.device)).shape[-1]


# Image encoder
class HFImageEncoder(nn.Module):
    """
    Load pretrained models via Huggingface, and extract features for different modalities.
    """
    def __init__(self, img_model: str = "facebook/dinov2-base", device: torch.device = torch.device("cuda")):
        super().__init__()
        self.device = device
        self.img_model_name = img_model
        try:
            self.img_model = AutoModel.from_pretrained(img_model)   # Different image extractor (e.g. DINO)
            self.image_processor = AutoProcessor.from_pretrained(img_model, use_fast=True)
            if 'mae' in img_model:
                print(self.img_model)
                self.img_model.config.mask_ratio = 0.
                self.img_model.embeddings.config.mask_ratio = 0.
                print(self.img_model.config)
                print(self.img_model.embeddings.config)
        except:
            from sentence_transformers import SentenceTransformer
            self.img_model = SentenceTransformer(img_model)
            self.image_processor = None
        config = self.img_model.config
        try: config = config.text_config
        except: pass

        if hasattr(config, "hidden_size"):
            self.dim = config.hidden_size
        else:
            self.dim = config.hidden_sizes[-1]
        # Freeze
        for p in self.parameters():
            p.requires_grad_(False)

    @property
    def config(self):
        return self.img_model.config

    def forward(self, pixel_values, normalize=False):
        # maybe preprocess
        if isinstance(pixel_values[0], PIL.Image.Image):
            pixel_values = self.image_processor(pixel_values, return_tensors="pt").pixel_values.to(self.device)

        if hasattr(self.img_model, 'get_image_features'):
            image_features = self.img_model.get_image_features(pixel_values)
        elif hasattr(self.img_model, 'encode'):
            image_features = self.img_model.encode(pixel_values)
        else:
            image_features = self.img_model(pixel_values)[0][:, 0, :]     # [CLS] token embedding

            # model_out = self.img_model(pixel_values, output_hidden_states=True)
            # image_features = model_out[0][:, 0, :]     # [CLS] token embedding
            # for i in range(2, 5):
            #     image_features += self.img_model.layernorm(model_out.hidden_states[-i][:, 0, :])
            # image_features /= 4.

            # cls_token = F.normalize(model_out[0][:, 0, :], dim=-1)
            # if True or self.training:
            #     patch_tokens = F.normalize(model_out[0][:, 1:, :].mean(1), dim=-1)  # Mean pool patch tokens
            #     image_features = (F.softmax(cls_token @ patch_tokens.t() * 10., dim=-1) @ cls_token)  # Attention pooling of patch tokens, weighted by CLS-token attention scores
            # else:
            #     patch_tokens = F.normalize(model_out[0][:, 1:, :], dim=-1)
            #     attn = torch.einsum('bd,bnd->bn', cls_token, patch_tokens) * 10.
            #     attn = F.softmax(attn, dim=-1)                        # [B, N]
            #     image_features = torch.einsum('bn,bnd->bd', attn, patch_tokens)  # [B, D]

        return F.normalize(image_features, dim=1) if normalize else image_features

