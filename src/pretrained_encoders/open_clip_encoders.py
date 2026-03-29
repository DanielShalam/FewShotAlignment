from typing import List

import torch
import torch.nn.functional as F
import open_clip


class OpenCLIPEncoders:
    def __init__(self, clip_model_name: str, pretrained: str = 'laion2b_s32b_b82k',
                 device: torch.device = torch.device('cuda')):

        self.model, self.train_preprocess, self.preprocess = open_clip.create_model_and_transforms(
            clip_model_name, pretrained=pretrained, force_quick_gelu=pretrained == "openai")
        self.tokenizer = open_clip.get_tokenizer(clip_model_name)
        self.model.eval().to(device)
        self.device = device
        print(self.model.visual.preprocess_cfg)
        self.img_size = self.model.visual.preprocess_cfg['size']
        if isinstance(self.img_size, tuple):
            self.img_size = self.img_size[-1]

    @torch.no_grad()
    def encode_text(self, texts: List[str]) -> torch.Tensor:
        tokens = self.tokenizer(texts).to(self.device)
        zt = self.model.encode_text(tokens)
        return F.normalize(zt.float(), dim=-1)

    @torch.no_grad()
    def encode_image(self, img_tensor: torch.Tensor) -> torch.Tensor:
        zi = self.model.encode_image(img_tensor)
        return F.normalize(zi.float(), dim=-1)

    @property
    def text_dim(self):
        # CLIP text head dim equals model.ln_final.weight.shape[0] for ViTs; simpler: probe
        return self.model.encode_text(self.tokenizer(["a"]).to(self.device)).shape[-1]

    @property
    def image_dim(self):
        return self.model.encode_image(torch.zeros(1,3,self.img_size,self.img_size, device=self.device)).shape[-1]
