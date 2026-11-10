"""Root-mean-square normalization, with no mean subtraction."""

import torch
from torch import nn


class RMSNorm(nn.Module):
    def __init__(self, width: int, eps: float = 1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(width))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        original_dtype = x.dtype
        values = x.float()
        values = values * torch.rsqrt(values.square().mean(dim=-1, keepdim=True) + self.eps)
        return values.to(original_dtype) * self.weight.to(original_dtype)
