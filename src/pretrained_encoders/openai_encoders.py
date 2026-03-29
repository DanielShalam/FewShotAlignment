from typing import List

import torch
import torch.nn.functional as F

from . import clip


class OpenAICLIPEncoders:
    def __init__(self, clip_model_name: str, device: torch.device):
        self.model, self.preprocess = clip.load(clip_model_name, device=device, jit=False)
        self.model.eval().to(device)
        self.device = device
        self.img_size = self.model.visual.input_resolution

    @torch.no_grad()
    def encode_text(self, texts: List[str]) -> torch.Tensor:
        tokens = clip.tokenize(texts, truncate=True).to(self.device)
        zt = self.model.encode_text(tokens)
        return F.normalize(zt.float(), dim=-1)

    @torch.no_grad()
    def encode_image(self, img_tensor: torch.Tensor) -> torch.Tensor:
        zi = self.model.encode_image(img_tensor)
        return F.normalize(zi.float(), dim=-1)

    @property
    def text_dim(self):
        # CLIP text head dim equals model.ln_final.weight.shape[0] for ViTs; simpler: probe
        return self.model.encode_text(clip.tokenize(["a"]).to(self.device)).shape[-1]

    @property
    def image_dim(self):
        return self.model.encode_image(torch.zeros(1,3,self.img_size,self.img_size, device=self.device)).shape[-1]
