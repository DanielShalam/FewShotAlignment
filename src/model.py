from __future__ import annotations
import copy
import os
import os.path as osp
import math
from functools import partial
from pathlib import Path
from typing import Callable, Tuple, Union, List, Dict, Optional

import numpy as np
import open_clip
import requests
import torch
import torch.nn as nn
from matplotlib import pyplot as plt
from torch import Tensor
from torch.nn import functional as F
import torchvision.transforms as T
from torchvision.transforms import InterpolationMode
from transformers import AutoImageProcessor
from transformers import AutoConfig, AutoProcessor
from sklearn.metrics import average_precision_score, roc_auc_score
from torch.utils.data import DataLoader
from tqdm import tqdm
from PIL import Image, ImageOps

# flow_matching
from flow_matching.path import GeodesicProbPath, CondOTProbPath
from flow_matching.path.scheduler import CondOTScheduler
from flow_matching.solver import ODESolver, RiemannianODESolver
from flow_matching.utils import ModelWrapper
from flow_matching.utils.manifolds import Sphere, Manifold

from src.imagenet_templates import IMAGENET_TEMPLATES, IMAGENET_TEMPLATES_SELECT
from src.utils import load_checkpoint
from src.pretrained_encoders.clip import clip

os.environ["TOKENIZERS_PARALLELISM"] = "false"

CUSTOM_TEMPLATES = {
    'OxfordPets': 'a photo of a {}, a type of pet.',
    'OxfordFlowers': 'a photo of a {}, a type of flower.',
    'FGVCAircraft': 'a photo of a {}, a type of aircraft.',
    'DescribableTextures': '{} texture.',
    'EuroSAT': 'a centered satellite photo of {}.',
    'StanfordCars': 'a photo of a {}.',
    'Food101': 'a photo of {}, a type of food.',
    'SUN397': 'a photo of a {}.',
    'Caltech101': 'a photo of a {}.',
    'UCF101': 'a photo of a person doing {}.',
    'ImageNet': 'a photo of a {}.',
    'ImageNetSketch': 'a photo of a {}.',
    'ImageNetV2': 'a photo of a {}.',
    'ImageNetA': 'a photo of a {}.',
    'ImageNetR': 'a photo of a {}.',
    'ChestX14': 'a chest X-ray showing {}.',
    'VinDrCXR': 'a chest X-ray showing {}.',
}

## Helpers
def load_clip_to_cpu(backbone_name):
    url = clip._MODELS[backbone_name]
    model_path = clip._download(url)
    try:
        # loading JIT archive
        model = torch.jit.load(model_path, map_location='cpu').eval()
        state_dict = None
    except RuntimeError:
        state_dict = torch.load(model_path, map_location='cpu')
    model = clip.build_model(state_dict or model.state_dict())
    return model
def slerp(u: torch.Tensor, v: torch.Tensor, lam: torch.Tensor, eps: float = 1e-7) -> torch.Tensor:
    """
    Spherical interpolation between u and v (both L2-normalized).
    u, v: [B, D]
    lam:  [B, 1] in [0,1], 1 means using v, 0 using u
    returns: [B, D], L2-normalized
    """
    # Cosine of angle between u and v
    dot = (u * v).sum(dim=-1, keepdim=True).clamp(-1.0 + eps, 1.0 - eps)
    omega = torch.acos(dot)
    sin_omega = torch.sin(omega)

    # For nearly colinear vectors, fall back to linear then renorm
    near = (sin_omega < 1e-4).squeeze(-1)
    out = torch.empty_like(u)

    if (~near).any():
        lam_ = lam[~near]
        u_ = u[~near]
        v_ = v[~near]
        omega_ = omega[~near]
        sin_omega_ = sin_omega[~near]
        a = torch.sin((1.0 - lam_) * omega_) / sin_omega_
        b = torch.sin(lam_ * omega_) / sin_omega_
        out[~near] = a * u_ + b * v_

    if near.any():
        lam_ = lam[near]
        u_ = u[near]
        v_ = v[near]
        out[near] = (1.0 - lam_) * u_ + lam_ * v_

    return F.normalize(out, p=2, dim=-1)
@torch.inference_mode()
def make_support_bank(visual, support_loader):
    visual.eval()
    feats, labels = [], []
    for b in support_loader:
        f = F.normalize(visual(b['img'].cuda()), dim=-1)
        feats.append(f.cpu())
        labels.append(b['label'])
    return torch.cat(feats).cuda(), torch.cat(labels).cuda()  # [Ns,D], [Ns]
@torch.no_grad()
def build_class_prototypes(feats, labels, num_classes):
    """
    Per-class L2-normalized mean in feature space of feats.
    """
    D = feats.size(1)
    protos = torch.zeros(num_classes, D, device=feats.device, dtype=feats.dtype)
    counts = torch.zeros(num_classes, device=feats.device, dtype=feats.dtype)
    for c in range(num_classes):
        mask = (labels == c)
        if mask.any():
            protos[c] = feats[mask].mean(dim=0)
            counts[c] = mask.float().sum()
    protos = F.normalize(protos, dim=-1)
    return protos
@torch.no_grad()
def generalized_orthogonal_procrustes(T, P, labels, *, eps=1e-6):
    """
    Solve:   min_W || T W - P ||_F
    where T:[C, Dt], P:[C, Di].
    If Dt >= Di, constraint is W^T W = I (semi-orthogonal columns).
    If Dt <  Di, we flip the problem to keep a valid semi-orthogonal constraint.

    Returns W with shape [Dt, Di].
    """
    Dt, Di = T.shape[1], P.shape[1]
    num_classes = labels.max() + 1
    num_shot = labels.shape[0] // num_classes
    arr = torch.arange(P.shape[0]).to(T.device)
    cls_to_indexes = []
    for c in range(num_classes):
        cls_to_indexes.append(arr[labels == c])
    cls_to_indexes = torch.stack(cls_to_indexes)

    arr_list = []
    for s in range(num_shot):
        arr_list.append(P[cls_to_indexes[:, s]])

    W = None
    for arr in arr_list:
        # Classic rectangular Procrustes: W^T W = I, solution W = U V^T
        M = (T.T @ arr).float()
        U, S, Vh = torch.linalg.svd(M, full_matrices=False)  # U:[Dt,Di], Vh:[Di,Di]
        if W is None:
            W = U @ Vh  # [Dt,Di]
        else:
            W += U @ Vh
    W /= num_shot
    return W # [Dt,Di]
@torch.no_grad()
def orthogonal_procrustes(T, P, labels, *, r=None, eps=1e-6):
    """
    Solve:   min_W || T W - P ||_F
    where T:[C, Dt], P:[C, Di].
    If Dt >= Di, constraint is W^T W = I (semi-orthogonal columns).
    If Dt <  Di, we flip the problem to keep a valid semi-orthogonal constraint.

    Returns W with shape [Dt, Di].
    """
    Dt, Di = T.shape[1], P.shape[1]

    # Classic rectangular Procrustes: W^T W = I, solution W = U V^T
    M = (T.T @ P).float()

    U, S, Vh = torch.linalg.svd(M, full_matrices=False)  # U:[Dt,Di], Vh:[Di,Di]

    if r is not None:
        # truncated rank
        U, Vh = U[:, :r], Vh[:r, :]

    W = U @ Vh  # [Dt,Di]
    # sign fix ONLY if square; otherwise skip (no determinant for rectangular)
    if Dt == Di:
        if torch.det(W) < 0:
            U[:, -1] = -U[:, -1]
            W = U @ Vh
    return W # [Dt,Di]

# Text Encoder
class TextEncoder(nn.Module):
    templates = IMAGENET_TEMPLATES_SELECT

    def __init__(self, cfg, classnames, text_model):
        super().__init__()
        self.cfg = cfg
        self.classnames = classnames
        self.text_model = text_model.eval()
        # self.dtype = text_model.dtype

        # add custom-made prompt
        if "ImageNet" not in cfg["dataset"]:
            self.templates = [CUSTOM_TEMPLATES[cfg["dataset"]]]

        num_temp = len(self.templates)
        print(f"Prompt ensembling (n={num_temp})")

        mean_text_features = 0

        for i, temp in enumerate(self.templates):
            prompts = [temp.format(c.replace("_", " ")) for c in classnames]
            prompts = self.tokenize_prompts(prompts, device="cpu")
            with torch.no_grad():
                text_features = self.text_model.encode_text(prompts)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            mean_text_features = mean_text_features + text_features
        mean_text_features = mean_text_features / num_temp
        mean_text_features = mean_text_features / mean_text_features.norm(dim=-1, keepdim=True)

        self.text_features = mean_text_features.cpu()
        self.dim = self.text_features.shape[-1]
        print("Text Features shape:", self.text_features.shape)

    def tokenize_prompts(self, prompts: List[str], device="cuda"):
        if self.cfg["txt_src"] == "HF":
            enc_prompts = self.text_model.tokenize(prompts).to(device)
        elif self.cfg["txt_src"] == "OC":
            enc_prompts = open_clip.get_tokenizer(self.cfg["txt_model"])(prompts).to(device)
        else:
            enc_prompts = torch.cat([clip.tokenize(p) for p in prompts]).to(device)
        return enc_prompts

    def encode_text(self, text_tokens, normalize=False):
        text_features = self.text_model.encode_text(text_tokens)
        return F.normalize(text_features, dim=-1) if normalize else text_features

    def forward(self):
        return self.text_features.cuda()


# Flow-model
def zero_module(module):
    """
    Zero out the parameters of a module and return it.
    """
    for p in module.parameters():
        p.detach().zero_()
    return module
def timestep_embedding(timesteps, dim, max_period=10000):
    """Create sinusoidal timestep embeddings.

    :param timesteps: a 1-D Tensor of N indices, one per batch element. These may be fractional.
    :param dim: the dimension of the output.
    :param max_period: controls the minimum frequency of the embeddings.
    :return: an [N x dim] Tensor of positional embeddings.
    """
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period)
        * torch.arange(start=0, end=half, dtype=torch.float32, device=timesteps.device)
        / half
    )
    args = timesteps[:, None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding
class ResBlock(nn.Module):
    """
    A residual block with AdaLN (Adaptive Layer Normalization) for timestep and context conditioning.
    Supports dense skip connections for U-Net style architectures.
    """

    def __init__(
            self,
            channels,
            mid_channels,
            emb_channels,
            dropout,
            use_context=False,
            context_channels=512,
            use_skip=False
    ):
        super().__init__()
        self.channels = channels
        self.emb_channels = emb_channels
        self.dropout = dropout
        self.use_skip = use_skip

        # Input dimension for first linear layer
        input_dim = channels * 2 if use_skip else channels

        # We keep the norm layers out of the sequential so we can manually modulate them.
        self.norm1 = nn.LayerNorm(channels, elementwise_affine=False)
        self.act1 = nn.SiLU()
        self.linear1 = nn.Linear(input_dim, mid_channels, bias=True)

        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(emb_channels, 2 * channels + 2 * mid_channels, bias=True) 
        )

        self.norm2 = nn.LayerNorm(mid_channels, elementwise_affine=False)
        self.act2 = nn.SiLU()
        self.drop = nn.Dropout(p=dropout)
        self.out_proj = zero_module(nn.Linear(mid_channels, channels, bias=True))

        self.use_context = use_context
        if use_context:
            self.context_layers = nn.MultiheadAttention(
                embed_dim=mid_channels,
                num_heads=8,
                kdim=context_channels,
                vdim=context_channels,
                batch_first=True
            )

    def forward(self, x, emb, context=None, skip=None):
        # 1. Compute scale/shift parameters from time-embedding
        modulation_params = self.adaLN_modulation(emb)
        
        # Split into gamma/beta for both norms
        gamma1, beta1, gamma2, beta2 = torch.split(
            modulation_params,
            [self.channels, self.channels, self.norm2.normalized_shape[0], self.norm2.normalized_shape[0]],
            dim=-1
        )

        # 2. First block: Norm -> Modulate -> Act -> Linear
        h = self.norm1(x)
        h = h * (1. + gamma1) + beta1
        h = self.act1(h)
        
        # Concatenate skip connection if available
        if self.use_skip and skip is not None:
            h = torch.cat([h, skip], dim=-1)
        
        h = self.linear1(h)

        # context
        if self.use_context and context is not None:
            B = h.shape[0]
            # h comes in as [B, mid_channels], need [B, 1, mid_channels] for Q
            h_q = h.unsqueeze(1)
            # context comes in as [C, D], broadcast to [B, C, D]
            ctx = context.unsqueeze(0).expand(B, -1, -1)
            # context_out is [B, 1, mid_channels]
            context_out, _ = self.context_layers(h_q, ctx, ctx)
            h = h + context_out.squeeze(1)

        # 3. Second block: Norm -> Modulate -> Act -> Drop -> Linear
        h = self.norm2(h)
        h = h * (1. + gamma2) + beta2
        h = self.act2(h)
        h = self.drop(h)
        h = self.out_proj(h)

        # 4. Residual connection
        return x + h

class GaussianFourierProjection(nn.Module):
    """Continuous Gaussian Fourier features for encoding time steps."""
    def __init__(self, embed_dim, scale=16.0):
        super().__init__()
        # Fixed random frequencies for better continuous-time resolution without adding parameters
        self.register_buffer('W', torch.randn(embed_dim // 2) * scale)

    def forward(self, t):
        t_proj = t.view(-1, 1) * self.W[None, :] * 2 * math.pi
        return torch.cat([torch.cos(t_proj), torch.sin(t_proj)], dim=-1)


def modulate(x, shift, scale):
    return x * (1 + scale) + shift


class FinalLayer(nn.Module):
    """DiT-style conditioned final layer for vector outputs."""

    def __init__(self, hidden_size, out_channels, embed_channels):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = zero_module(nn.Linear(hidden_size, out_channels, bias=True))
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(embed_channels, 2 * hidden_size, bias=True)
        )

    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        x = modulate(self.norm_final(x), shift, scale)
        x = self.linear(x)
        return x


class SimpleMLP(nn.Module):
    """
    The full UNet model with attention and timestep embedding.
    Uses dense skip connections: input projection passed to all ResBlocks.
    :param in_channels: channels in the input Tensor.
    :param model_channels: base channel count for the model.
    :param out_channels: channels in the output Tensor.
    :param num_res_blocks: number of residual blocks per downsample.
    """

    def __init__(
            self,
            in_channels,
            time_embed_dim,
            model_channels,
            bottleneck_channels,
            out_channels,
            num_res_blocks,
            dropout=0,
            use_context=False,
            context_channels=512,
            use_final_layer_head=False,
    ):
        super().__init__()

        self.image_size = 1
        self.in_channels = in_channels
        self.model_channels = model_channels
        self.out_channels = out_channels
        self.num_res_blocks = num_res_blocks
        self.dropout = dropout
        self.use_final_layer_head = use_final_layer_head

        # Use continuous random fourier features instead of fixed uniform stepping
        self.time_encoder = GaussianFourierProjection(self.model_channels)
        self.time_embed = nn.Sequential(
            nn.Linear(self.model_channels, time_embed_dim),
            nn.SiLU(),
            nn.Linear(time_embed_dim, time_embed_dim),
        )

        self.input_proj = nn.Linear(in_channels, model_channels)

        res_blocks = []
        for i in range(num_res_blocks):
            res_blocks.append(ResBlock(
                model_channels,
                bottleneck_channels,
                time_embed_dim,
                dropout,
                use_context=use_context,
                context_channels=context_channels,
                use_skip=False,
            ))
        self.res_blocks = nn.ModuleList(res_blocks)

        if self.use_final_layer_head:
            self.out = FinalLayer(model_channels, out_channels, time_embed_dim)
        else:
            self.out = nn.Sequential(
                nn.LayerNorm(model_channels, eps=1e-6),
                nn.SiLU(),
                zero_module(nn.Linear(model_channels, out_channels, bias=True)),
            )

    def forward(self, t, x, y=None):
        """
        Apply the model to an input batch.
        :param x: an [N x C x ...] Tensor of inputs.
        :param t: a 1-D batch of timesteps.
        :param y: conditioning plugged in via crossattn
        :return: an [N x C x ...] Tensor of outputs.
        """
        x = self.input_proj(x)

        if t.ndim == 2:
            t = t.squeeze(-1)
        if t.dim() == 0:
            t = t.repeat(x.shape[0])

        # Generate continuous timestep embeddings
        emb = self.time_embed(self.time_encoder(t))

        for block in self.res_blocks:
            x = block(x, emb, y)

        if self.use_final_layer_head:
            x = self.out(x, emb)
        else:
            x = self.out(x)

        return x

class ProjectToTangent(torch.nn.Module):
    """Projects a vector field onto the tangent plane at the input."""

    def __init__(self, vecfield, manifold, metric_normalize):
        super().__init__()
        self.vecfield = vecfield
        self.manifold = manifold
        self.metric_normalize = metric_normalize

    def forward(self, t, x, **kwargs):
        x = self.manifold.projx(x)
        v = self.vecfield(t, x, **kwargs)
        v = self.manifold.proju(x, v)

        if self.metric_normalize and hasattr(self.manifold, "metric_normalized"):
            v = self.manifold.metric_normalized(x, v)
        return v

# OP
class OrthogonalProcrustes(nn.Module):
    def __init__(self, enable: bool, mode: str, zero_padding: bool, pretrained: str = None):
        super().__init__()
        assert mode in ["centroid", "generalized"], f"mode must be 'centroid' or 'generalized', got {mode}"
        self.enable = enable
        self.mode = mode
        self.zero_padding = zero_padding

        self._compute_procrustes = orthogonal_procrustes if mode == "centroid" else generalized_orthogonal_procrustes
        self.W = None
        if pretrained is not None and pretrained != "":
            print("[OP] Loading pretrained W...")
            W_state_dict = torch.load(pretrained, weights_only=True)
            self.W = W_state_dict['W'].float()

    def apply_zero_padding(self, x: Tensor, new_dim: int):
        if x.size(-1)>=new_dim or not (self.enable or self.zero_padding):
            return x
        return F.pad(x, (0, new_dim-x.shape[-1]), 'constant', 0.)

    def fit(self, x: Tensor, y: Tensor, labels: Tensor = None, beta: float = None):
        if not self.enable or self.W is not None:
            return x, y
        dx, dy = x.shape[-1], y.shape[-1]
        # zero-padding (optional)
        if self.zero_padding and dx != dy:
            if dx<dy: x = self.apply_zero_padding(x, dy)
            else: y = self.apply_zero_padding(y, dx)

        # compute W from inputs
        W = self._compute_procrustes(x, y, labels)
        print("[OP] W shape, beta=:", W.shape, beta)

        # beta-regularized (optional)
        if beta and beta is not None:
            identity = torch.zeros_like(W)
            identity.fill_diagonal_(1.)
            W = W - (W - identity) * beta

        self.W = W.float().to(x.device)
        return F.normalize(x @ self.W, dim=-1), y

    def transform(self, x: Tensor, y: Tensor):
        """
        Compute the optimal Orthogonal (or semi) Procrustes transform.
        Aligns matrix x [b, dx] to a matrix y [b, dy].
        """
        if not self.enable:
            return x, y
        dx, dy = x.shape[-1], y.shape[-1]
        # zero-padding (optional)
        if self.zero_padding and dx != dy:
            if dx<dy: x = self.apply_zero_padding(x, dy)
            else: y = self.apply_zero_padding(y, dx)
        return F.normalize(x @ self.W.float().to(x.device), dim=-1), y


# Flow-Adapter
class FlowAdapter(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = copy.deepcopy(cfg)
        self.train_time_samples = int(cfg.get("train_time_samples", 1))

        self.bank_feat = None; self.bank_labels = None; self.bank_prototypes = None

        # 1. Load Uni-Modal Encoders (Frozen)
        text_encoder, self.image_encoder, self.train_tfm, self.eval_tfm = self._build_pretrained_encoders()
        self.text_encoder = TextEncoder(cfg, cfg["classnames"], text_encoder)
        self.logit_scale = text_encoder.logit_scale if hasattr(text_encoder, 'logit_scale') else torch.tensor(0.)

        # 2. Initialize Orthogonal Procrustes
        zero_padding = False
        self.OP = OrthogonalProcrustes(
            enable=cfg["use_op"], mode="centroid", zero_padding=zero_padding, pretrained=cfg["pretrained_op"],
        )

        # 3. Create Adapters
        velocity_dim = max(int(self.image_encoder.dim), int(self.text_encoder.dim)) if self.OP.zero_padding \
            else int(self.image_encoder.dim)

        self.use_context = cfg.get("use_context", False)
        ada_kwargs = dict(
            in_channels=velocity_dim, out_channels=velocity_dim,
            num_res_blocks=cfg["ada_depth"], time_embed_dim=cfg["ada_t_dim"],
            model_channels=cfg["ada_dim"], bottleneck_channels=cfg["ada_dim"],
            use_context=self.use_context, context_channels=velocity_dim,
            use_final_layer_head=cfg.get("use_final_layer_head", False),
        )

        # I2T adapter
        self.adapter = SimpleMLP(**ada_kwargs)

        # T2I adapter (optional)
        if cfg["text_adapter"]:
            self.t_adapter = SimpleMLP(**ada_kwargs)    # text to image adapter

        # 4. Flow Matching Setup
        if cfg["fm_type"] == "geodesic":
            print("Initialized geodesic FM....")
            manifold = Sphere()
            self.path = GeodesicProbPath(scheduler=CondOTScheduler(), manifold=manifold)

            # wrap adapters with tangent projection
            self.adapter = ProjectToTangent(self.adapter, manifold=manifold, metric_normalize=False)
            if cfg["text_adapter"]:
                self.t_adapter = ProjectToTangent(self.t_adapter, manifold=manifold, metric_normalize=False)

        elif cfg["fm_type"] == "linear":
            print("Initialized linear FM....")
            self.path = CondOTProbPath()

        else:
            raise NotImplementedError

        print("Model initialized...")

    def _build_pretrained_encoders(self):
        from src.pretrained_encoders.hf_encoders import HFTextEncoder, HFImageEncoder, make_transforms_hf
        from src.pretrained_encoders.bam_encoder import vit_base, load_bam_pretrained_weights

        cfg = self.cfg
        # placeholder for image transform
        train_tfm, eval_tfm = None, None

        # load text-encoder
        text_source = cfg['txt_src']; text_model = cfg['txt_model']
        if text_source == "HF":                 # huggingface/sentence-transformers
            print(f'-> Loading HuggingFace Text encoder... ({text_model})')
            text_encoder = HFTextEncoder(text_model)
        elif text_source == "OC":               # open-clip
            print(f'-> Loading Open-Clip Text encoder... ({text_model})')
            try:
                text_encoder, train_tfm, eval_tfm = open_clip.create_model_and_transforms(
                    text_model, pretrained='laion2b_s32b_b82k')
            except:
                text_encoder, train_tfm, eval_tfm = open_clip.create_model_and_transforms(
                    text_model, pretrained='openai', force_quick_gelu=True)
        else:                                   # OpenAI-clip
            print(f'-> Loading Clip Text encoder... ({text_model})')
            text_encoder = load_clip_to_cpu(text_model)

        # load image-encoder
        img_source = cfg['img_src']; img_model = cfg['img_model']
        if img_source == "HF":      # huggingface/sentence-transformers
            print(f'-> Loading HuggingFace image encoder... ({img_model})')
            image_encoder = HFImageEncoder(img_model)
            train_tfm, eval_tfm = make_transforms_hf(img_model, rrc_scale=(0.5, 1.0),  hflip_p=0.5)
        elif img_source == "BAM":
            # download from https://github.com/DanielShalam/BAM
            image_encoder, train_tfm, eval_tfm = load_bam_pretrained_weights(vit_base(num_classes=0), img_model)
        else:
            if img_source == "OC":    # open-clip
                print(f'-> Loading Open-Clip  (Visual) as image encoder... ({img_model})')
                try:
                    temp_, train_tfm, eval_tfm = open_clip.create_model_and_transforms(
                        img_model, pretrained='laion2b_s32b_b82k')
                except:
                    temp_, train_tfm, eval_tfm = open_clip.create_model_and_transforms(
                        img_model, pretrained='openai', force_quick_gelu=True)
            else:                                   # clip
                print(f'-> Loading Clip  (Visual) as image encoder... ({img_model})')
                temp_, preprocess = clip.load(img_model, jit=False)
                train_tfm = T.Compose([
                    T.RandomResizedCrop(size=224, scale=(0.5, 1), interpolation=T.InterpolationMode.BICUBIC),
                    T.RandomHorizontalFlip(p=0.5),
                    T.ToTensor(),
                    T.Normalize(mean=(0.48145466, 0.4578275, 0.40821073), std=(0.26862954, 0.26130258, 0.27577711))
                ])
                eval_tfm = preprocess

            image_encoder = temp_.visual
            if isinstance(temp_.text_projection, torch.nn.Parameter):
                image_encoder.dim = temp_.text_projection.shape[-1]
            else:
                image_encoder.dim = temp_.text_projection.weight.shape[-1]

        text_encoder = text_encoder.eval().float()
        image_encoder = image_encoder.eval().float()

        return text_encoder, image_encoder, train_tfm, eval_tfm

    @torch.inference_mode()
    def create_bank(self, loader, op_beta=None):
        """
        support_items: the EXACT k-shot training items (per class) you will train on.
                       (No val items here to keep protocol clean.)
        """

        # compute support features in the SAME space you'll use at train/eval
        self.bank_feat, self.bank_labels = make_support_bank(visual=self.image_encoder, support_loader=loader)

        #  class prototypes (image space)
        num_classes = self.text_encoder.text_features.size(0)
        self.bank_prototypes = build_class_prototypes(self.bank_feat, self.bank_labels, num_classes)  # [C, D_img]

        # rectangular OP: align text prototypes to image space
        if self.cfg["use_op"]:
            #    (Assumes text_encoder.text_features is [C, Dt] and L2-normalized)
            text_feats = F.normalize(self.text_encoder(), dim=-1)
            y = self.bank_feat if self.OP.mode == "generalized" else self.bank_prototypes
            text_aligned, image_aligned = self.OP.fit(x=text_feats, y=y, labels=self.bank_labels, beta=op_beta)

            print(f"[create_bank] support feats: {self.bank_feat.shape}, "
                  f"protos: {self.bank_prototypes.shape}, "
                  f"image aligned: {image_aligned.shape}, "
                  f"text aligned: {text_aligned.shape}")

    def forward(
        self,
        images: Tensor,
        labels: Tensor = None,
        t_end: float = 0.5,
        solver: str = "dopri5",
        steps: Optional[int] = None,
    ):
        # 1. Encode
        img_feats = F.normalize(self.image_encoder(images), dim=-1)
        txt_feats = F.normalize(self.text_encoder().clone().detach(), dim=-1)

        # 2. Apply OP
        txt_feats, img_feats = self.OP.transform(x=txt_feats, y=img_feats)
        ctx = txt_feats if self.use_context else None

        # 3. Training Mode
        if self.training and labels is not None:
            # conditional-flow-matching loss
            def cfm_loss(v_model, x_0, x_1, y=None):
                k = max(1, self.train_time_samples)
                if k > 1:
                    x_0 = x_0.repeat_interleave(k, dim=0)
                    x_1 = x_1.repeat_interleave(k, dim=0)
                t = torch.rand(x_0.shape[0], device=x_0.device)  # sample timestep(s)
                p_s = self.path.sample(t=t, x_0=x_0, x_1=x_1)  # sample path
                vt = v_model(p_s.t, p_s.x_t, y=y)
                return torch.pow(vt - p_s.dx_t, 2).mean()

            # image -> text loss
            loss = cfm_loss(v_model=self.adapter, x_0=img_feats, x_1=txt_feats[labels], y=ctx)

            # (optional) text -> image loss
            if self.cfg["text_adapter"]:
                loss += cfm_loss(v_model=self.t_adapter, x_0=txt_feats[labels], x_1=img_feats, y=ctx)

            return loss

        # 4. Inference
        else:
            # 2-sided integration
            i2t_feats = self._solve_ode(img_feats, t_end=t_end, method=solver, steps=steps, y=ctx)

            inv_fm_type = self.cfg["inv_fm_type"]
            if self.cfg["text_adapter"] and inv_fm_type != "none":
                t2i_feats = self._solve_ode(
                    txt_feats, 
                    t_end=(1 - t_end), 
                    method=solver,
                    steps=steps,
                    net_t=inv_fm_type not in ["reverse", "proto"],
                    reverse=inv_fm_type == "reverse",
                    use_protos=inv_fm_type == "proto",
                    y=ctx,
                    )
            else:
                t2i_feats = txt_feats

            # compute logits
            logit_scale = self.logit_scale.exp()
            logits_zs = logit_scale * (img_feats @ txt_feats.t())
            logits_mt = logit_scale * (i2t_feats @ t2i_feats.t())

            return {"ZS": logits_zs, "MT": logits_mt}

    @torch.inference_mode()
    def _solve_ode(
            self, x0: torch.Tensor, y: torch.Tensor = None, steps: Optional[int] = None, method: str = "dopri5",
            t_end: float = 1., net_t: bool = False, reverse: bool = False, use_protos: bool = False,
            return_intermediates=False) -> torch.Tensor:
        """Generate samples via an ODE solver from *torchdiffeq* (if available).

        Args:
            x0:         source data.
            y:          conditioning data.
            steps:      number of integration steps. only used for non-adaptive solvers ("euler", "rk4", ...).
            method:     ODE method string accepted by *torchdiffeq* ("rk4", "dopri5", …).
        """
        if t_end == 0. or method is None:
            return x0

        if use_protos:
            return F.normalize(slerp(x0, self.bank_prototypes, lam=torch.ones_like(x0)[:, 0:1] * t_end), dim=-1)

        device = x0.device
        class WrappedModel(ModelWrapper):
            def forward(self, t: torch.Tensor, x: torch.Tensor, **extras):
                return self.model(x=x, t=t, **extras)
        net = copy.deepcopy(self.adapter) if not net_t else copy.deepcopy(self.t_adapter)
        net = net.eval()

        if method == "dopri5":
            solver = ODESolver(velocity_model=WrappedModel(net))  # create an ODESolver class
            time_grid = torch.tensor([0.0, t_end]).to(device=device)
            sol = solver.sample(
                x_init=x0,
                step_size=None,
                method=method,
                return_intermediates=return_intermediates,
                time_grid=time_grid if not reverse else torch.flip(time_grid, dims=[0]),
                y=y if self.use_context else None,
            )
        else:
            # not an adaptive-step solver
            solver = RiemannianODESolver(velocity_model=WrappedModel(net), manifold=Sphere())  # create an ODESolver class
            n_steps = int(steps) if steps is not None else 1
            n_steps = max(1, n_steps)
            time_grid = torch.linspace(0.0, float(t_end), n_steps + 1, device=x0.device)
            # For steps=1, using step_size=t_end violates strict interval>step_size checks in solver internals.
            step_size = float(t_end) / float(max(n_steps, 2))
            sol = solver.sample(
                x_init=x0,
                method=method,
                step_size=step_size,
                return_intermediates=return_intermediates,
                time_grid=time_grid if not reverse else torch.flip(time_grid, dims=[0]),
                y=y if self.use_context else None,
            )

        return F.normalize(sol, dim=-1)

    @torch.inference_mode()
    def tune_hyperparameters(self, loader, device, alphas=None, t_end_list=None, solver: str = "dopri5"):
        if alphas is None: alphas = np.arange(11) / 10  # 0..1
        if t_end_list is None: t_end_list = np.arange(0, 11) / 10
        best, best_params = -1.0, (0.0, 1.0, 1.0)
        accuracy_list = []
        self.eval()
        for t_end in t_end_list:
            for temperature in [1.]:
                correct = torch.zeros(len(alphas), device=device)
                total = 0
                for batch in loader:
                    imgs = batch['img'].to(device)
                    labels = batch['label'].to(device)
                    logits = self.forward(imgs, t_end=t_end, solver=solver)  # raw logits
                    for i, a in enumerate(alphas):
                        mix_logits = (1 - a) * (logits["MT"]*temperature) + a * logits["ZS"]
                        pred = mix_logits.argmax(dim=1)
                        correct[i] += (pred == labels).sum()
                    total += labels.size(0)
                    
                accuracy_list.append([acc.item() for acc in correct / total])
                acc = (correct / total).max().item()
                a_star = alphas[int((correct / total).argmax().item())]
                if acc > best:
                    best, best_params = acc, (a_star, t_end, temperature)

        return best, best_params  # (alpha*, T*)


class MultiLabelFlowAdapter(FlowAdapter):

    def _text_mixture(self, y_multi: torch.Tensor, text_features: torch.Tensor, eps: float = 1e-6):
        # y_multi: [B,C] in {0,1}; text_features: [C,D]
        if y_multi.ndim != 2:
            raise ValueError(f"Expected multi-label targets with shape [B, C], got {tuple(y_multi.shape)}")
        y_multi = y_multi.to(device=text_features.device, dtype=text_features.dtype)
        w = y_multi / (y_multi.sum(dim=1, keepdim=True) + eps)  # normalize per-sample if multiple positives
        x_txt_mix = w @ text_features  # [B, D]
        return F.normalize(x_txt_mix, dim=-1)

    def forward(
        self,
        images: Tensor,
        labels: Tensor = None,
        t_end: float = 0.5,
        solver: str = "dopri5",
        steps: Optional[int] = None,
    ):
        # 1. Encode
        img_feats = F.normalize(self.image_encoder(images), dim=-1)
        txt_feats = F.normalize(self.text_encoder().clone().detach(), dim=-1)

        # 2. Apply OP
        txt_feats, img_feats = self.OP.transform(x=txt_feats, y=img_feats)

        # 3. Training Mode
        if self.training and labels is not None:
            # conditional-flow-matching loss
            def cfm_loss(v_model, x_0, x_1):
                k = max(1, self.train_time_samples)
                if k > 1:
                    x_0 = x_0.repeat_interleave(k, dim=0)
                    x_1 = x_1.repeat_interleave(k, dim=0)
                t = torch.rand(x_0.shape[0], device=x_0.device)  # sample timestep(s)
                p_s = self.path.sample(t=t, x_0=x_0, x_1=x_1)  # sample path
                vt = v_model(p_s.t, p_s.x_t)
                return torch.pow(vt - p_s.dx_t, 2).mean()

            txt_feats_mixture = self._text_mixture(labels, txt_feats)

            # image -> text loss
            loss = cfm_loss(v_model=self.adapter, x_0=img_feats, x_1=txt_feats_mixture)

            # (optional) text -> image loss
            if self.cfg["text_adapter"]:
                loss += cfm_loss(v_model=self.t_adapter, x_0=txt_feats_mixture, x_1=img_feats)

            return loss

        # 4. Testing Mode
        else:
            # 2-sided integration
            i2t_feats = self._solve_ode(img_feats, t_end=t_end, method=solver, steps=steps)

            inv_fm_type = self.cfg["inv_fm_type"]
            if self.cfg["text_adapter"] and inv_fm_type != "none":
                t2i_feats = self._solve_ode(
                    txt_feats, t_end=(1 - t_end), method=solver, steps=steps,
                    net_t=inv_fm_type not in ["reverse", "proto"],
                    reverse=inv_fm_type == "reverse", use_protos=inv_fm_type == "proto")
            else:
                t2i_feats = txt_feats

            # compute logits
            logit_scale = self.logit_scale.exp()
            logits_zs = logit_scale * (img_feats @ txt_feats.t())
            logits_mt = logit_scale * (i2t_feats @ t2i_feats.t())

            return {"ZS": logits_zs, "MT": logits_mt}

    @torch.inference_mode()
    def tune_hyperparameters(self, loader, multi_map, device, alphas=None, t_end_list=None, solver: str = "dopri5"):
        if alphas is None: alphas = np.linspace(0.0, 1.0, 11)  # 0..1
        if t_end_list is None: t_end_list = np.linspace(0., 1., 11)
        best, best_pair = -1.0, (0.5, 0.6)
        self.eval()
        for t_end in t_end_list:
            # accumulate all val scores/targets for this t_end
            all_scores = []
            all_targets = []
            for batch in loader:
                imgs = batch['img'].to(device)
                impaths = batch["impath"]  # make sure your Dataset returns this
                y_multi = torch.stack([multi_map[p] for p in impaths], 0).to(device)
                logits = self.forward(imgs, t_end=t_end, solver=solver)  # returns {"ZS_raw","MT_raw"}
                ZS = logits["ZS"].detach().cpu().numpy()
                MT = logits["MT"].detach().cpu().numpy()
                all_scores.append((ZS, MT))
                all_targets.append(y_multi.cpu().numpy())

            Y = np.concatenate([t for _, t in zip(all_scores, all_targets)], axis=0)  # [N,6]
            ZS_all = np.concatenate([z for (z, m) in all_scores], axis=0)
            MT_all = np.concatenate([m for (z, m) in all_scores], axis=0)

            for a in alphas:
                S = (1 - a) * MT_all + a * ZS_all  # raw scores
                # per-class AP; ignore classes with no positives in Y
                ap_list = []
                for c in range(Y.shape[1]):
                    if Y[:, c].sum() < 1:  # skip empty class
                        continue
                    ap = average_precision_score(Y[:, c], S[:, c])
                    ap_list.append(ap)
                if len(ap_list) == 0:
                    continue
                macro_ap = float(np.mean(ap_list))
                # print(macro_ap)
                if macro_ap > best:
                    best, best_pair = macro_ap, (float(a), float(t_end))
        return best, best_pair



class ResBlockNoConditioning(nn.Module):
    """
    A residual block with AdaLN (Adaptive Layer Normalization) for timestep and context conditioning.
    Supports dense skip connections for U-Net style architectures.
    """

    def __init__(
            self,
            channels,
            mid_channels,
            emb_channels,
            dropout,
            use_context=False,
            context_channels=512,
            use_skip=False
    ):
        super().__init__()
        self.channels = channels
        self.emb_channels = emb_channels
        self.dropout = dropout
        self.use_skip = use_skip

        # Input dimension for first linear layer
        input_dim = channels * 2 if use_skip else channels
        model_channels = input_dim
        # We keep the norm layers out of the sequential so we can manually modulate them.
        self.norm1 = nn.LayerNorm(model_channels, eps=1e-6)
        self.act1 = nn.SiLU()
        self.linear1 = nn.Linear(input_dim, mid_channels, bias=True)

        self.norm2 = nn.LayerNorm(model_channels, eps=1e-6)
        self.act2 = nn.SiLU()
        self.drop = nn.Dropout(p=dropout)
        self.out_proj = nn.Linear(mid_channels, channels, bias=True)

        self.use_context = use_context
        if use_context:
            self.context_layers = nn.MultiheadAttention(
                embed_dim=mid_channels,
                num_heads=8,
                kdim=context_channels,
                vdim=context_channels,
                batch_first=True
            )

    def forward(self, x):
        # 2. First block: Norm -> Modulate -> Act -> Linear
        h = self.norm1(x)
        h = self.act1(h)
        h = self.linear1(h)

        # 3. Second block: Norm -> Modulate -> Act -> Drop -> Linear
        h = self.norm2(h)
        h = self.act2(h)
        h = self.drop(h)
        h = self.out_proj(h)

        # 4. Residual connection
        return x + h


class SimpleMLPNoConditioning(nn.Module):
    """
    The full UNet model with attention and timestep embedding.
    Uses dense skip connections: input projection passed to all ResBlocks.
    :param in_channels: channels in the input Tensor.
    :param model_channels: base channel count for the model.
    :param out_channels: channels in the output Tensor.
    :param num_res_blocks: number of residual blocks per downsample.
    """

    def __init__(
            self,
            in_channels,
            time_embed_dim,
            model_channels,
            bottleneck_channels,
            out_channels,
            num_res_blocks,
            dropout=0,
            use_context=False,
            context_channels=512,
            use_final_layer_head=False,
    ):
        super().__init__()

        self.image_size = 1
        self.in_channels = in_channels
        self.model_channels = model_channels
        self.out_channels = out_channels
        self.num_res_blocks = num_res_blocks
        self.dropout = dropout
        self.use_final_layer_head = use_final_layer_head

        self.input_proj = nn.Linear(in_channels, model_channels)

        res_blocks = []
        for i in range(num_res_blocks):
            res_blocks.append(ResBlockNoConditioning(
                model_channels,
                bottleneck_channels,
                time_embed_dim,
                dropout,
                use_context=use_context,
                context_channels=context_channels,
                use_skip=False,
            ))
        self.res_blocks = nn.ModuleList(res_blocks)

        self.out = nn.Sequential(
            nn.LayerNorm(model_channels, eps=1e-6),
            nn.SiLU(),
            nn.Linear(model_channels, out_channels, bias=True),
        )

    def forward(self, x):
        """
        Apply the model to an input batch.
        :param x: an [N x C x ...] Tensor of inputs.
        :param t: a 1-D batch of timesteps.
        :param y: conditioning plugged in via crossattn
        :return: an [N x C x ...] Tensor of outputs.
        """
        x = self.input_proj(x)
        for block in self.res_blocks:
            x = block(x)
        x = self.out(x)
        return x
class ContrastiveMLPAdapter(FlowAdapter):
    def __init__(self, cfg):
        nn.Module.__init__(self)
        self.cfg = copy.deepcopy(cfg)

        self.bank_feat = None; self.bank_labels = None; self.bank_prototypes = None

        text_encoder, self.image_encoder, self.train_tfm, self.eval_tfm = self._build_pretrained_encoders()
        self.text_encoder = TextEncoder(cfg, cfg["classnames"], text_encoder)
        self.logit_scale = text_encoder.logit_scale if hasattr(text_encoder, 'logit_scale') else torch.tensor(0.)

        self.OP = OrthogonalProcrustes(
            enable=cfg["use_op"], mode="centroid", zero_padding=False, pretrained=cfg["pretrained_op"],
        )

        img_dim = int(self.image_encoder.dim)
        txt_dim = int(self.text_encoder.dim)
        
        txt_in_dim = img_dim if self.OP.enable else txt_dim
        img_in_dim = img_dim

        if not cfg.get("text_adapter", True) and not self.OP.enable:
            out_dim = txt_in_dim
        else:
            out_dim = img_in_dim

        hidden_dim = cfg.get("mlp_hidden_dim", out_dim * 2)

        ada_kwargs = dict(
            in_channels=out_dim, out_channels=out_dim,
            num_res_blocks=cfg["ada_depth"], time_embed_dim=cfg["ada_t_dim"],
            model_channels=cfg["ada_dim"], bottleneck_channels=cfg["ada_dim"],
            use_context=False,
            use_final_layer_head=cfg.get("use_final_layer_head", False),
        )
        self.adapter = SimpleMLPNoConditioning(**ada_kwargs)
        # self.adapter = nn.Sequential(
        #     nn.Linear(img_in_dim, hidden_dim),
        #     nn.LayerNorm(hidden_dim),
        #     nn.ReLU(inplace=True),
        #     nn.Linear(hidden_dim, out_dim)
        # )
        
        if cfg.get("text_adapter", True):
            self.t_adapter = nn.Sequential(
                nn.Linear(txt_in_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(inplace=True),
                nn.Linear(hidden_dim, out_dim)
            )

    def forward(self, images: Tensor, labels: Tensor = None, **kwargs):
        img_feats = self.image_encoder(images)
        img_feats = F.normalize(img_feats, dim=-1)

        txt_feats = self.text_encoder().to(img_feats.device)
        txt_feats = F.normalize(txt_feats, dim=-1)

        txt_feats, img_feats = self.OP.transform(x=txt_feats, y=img_feats)
        
        if img_feats.shape[-1] == txt_feats.shape[-1]:
            logits_zs = self.logit_scale.exp() * img_feats @ txt_feats.T
        else:
            logits_zs = None

        img_mt = self.adapter(img_feats)
        img_mt = F.normalize(img_mt, dim=-1)

        if self.cfg.get("text_adapter", True):
            txt_mt = self.t_adapter(txt_feats)
            txt_mt = F.normalize(txt_mt, dim=-1)
        else:
            txt_mt = txt_feats

        logits_mt = self.logit_scale.exp() * img_mt @ txt_mt.T

        if self.training and labels is not None:
            loss = F.cross_entropy(logits_mt, labels)
            return loss

        return {"ZS": logits_zs, "MT": logits_mt}

    @torch.inference_mode()
    def tune_hyperparameters(self, loader, device, alphas=None, **kwargs):
        if alphas is None: alphas = np.arange(11) / 10
        self.eval()
        correct = torch.zeros(len(alphas), device=device)
        total = 0
        for batch in loader:
            imgs = batch['img'].to(device)
            labels = batch['label'].to(device)
            logits = self.forward(imgs)
            for i, a in enumerate(alphas):
                if not self.OP.enable:
                    mix_logits = logits["MT"]
                else:
                    mix_logits = (1 - a) * logits["MT"] + a * logits["ZS"]
                pred = mix_logits.argmax(dim=1)
                correct[i] += (pred == labels).sum()
            total += labels.size(0)

        accs = correct / total
        best_acc = accs.max().item()
        best_alpha = alphas[accs.argmax().item()]
        
        if not self.OP.enable:
            best_alpha = 0.0

        return best_acc, (best_alpha, 0.5)

