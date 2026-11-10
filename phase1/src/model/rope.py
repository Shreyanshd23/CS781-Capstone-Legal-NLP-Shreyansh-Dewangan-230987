"""Rotary position embedding applied to even/odd pairs of Q and K."""

import torch
from torch import nn


class RotaryEmbedding(nn.Module):
    def __init__(self, head_dim: int, max_positions: int, theta: float = 10000.0):
        super().__init__()
        if head_dim % 2:
            raise ValueError("head_dim must be even")
        inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2).float() / head_dim))
        positions = torch.arange(max_positions, dtype=torch.float32)
        phases = torch.outer(positions, inv_freq)
        self.register_buffer("cos_cache", phases.cos(), persistent=False)
        self.register_buffer("sin_cache", phases.sin(), persistent=False)

    def rotate(self, x: torch.Tensor, position_ids: torch.Tensor) -> torch.Tensor:
        # x: [batch, heads, time, head_dim]; position_ids: [batch, time]
        cos = self.cos_cache[position_ids].unsqueeze(1).to(x.dtype)
        sin = self.sin_cache[position_ids].unsqueeze(1).to(x.dtype)
        even, odd = x[..., 0::2], x[..., 1::2]
        return torch.stack((even * cos - odd * sin,
                            even * sin + odd * cos), dim=-1).flatten(-2)

    def forward(self, q: torch.Tensor, k: torch.Tensor,
                position_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.rotate(q, position_ids), self.rotate(k, position_ids)
