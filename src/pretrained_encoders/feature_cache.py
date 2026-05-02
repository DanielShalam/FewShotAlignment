"""Stub encoders that serve precomputed features; no model loading at runtime."""
import torch
import torch.nn as nn
import torch.nn.functional as F


class IdentityImageEncoder(nn.Module):
    """Passes through precomputed unit-normalized embeddings."""
    def __init__(self, dim: int):
        super().__init__()
        self.dim = int(dim)
        # preserve the eval()/train() API used by FSA
        self.dummy = nn.Parameter(torch.zeros(1), requires_grad=False)

    def forward(self, x, normalize: bool = False):
        # x is already [B, D] float; unit-norm is applied by the caller with F.normalize
        return x


class CachedTextEncoder(nn.Module):
    """Holds class-text prototypes as a fixed [C, D] tensor.

    Mirrors just enough of TextEncoder to satisfy FlowAdapter:
      - callable: returns [C, D]
      - .text_features attribute
      - .dim (= D)
      - exposes a no-op .eval() / .float()
    """
    def __init__(self, text_features: torch.Tensor):
        super().__init__()
        # register as buffer so .to(device) works
        self.register_buffer("text_features", F.normalize(text_features.float(), dim=-1))

    @property
    def dim(self) -> int:
        return int(self.text_features.shape[-1])

    def forward(self):
        return self.text_features

    def __call__(self):
        return self.forward()
