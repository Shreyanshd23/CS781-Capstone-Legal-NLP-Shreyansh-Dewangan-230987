"""Pre-Norm causal GQA and SwiGLU residual block."""

import torch
from torch import nn

from .attention import GroupedQueryAttention
from .config import LegalLMConfig
from .rmsnorm import RMSNorm
from .swiglu import SwiGLU


class LegalBlock(nn.Module):
    def __init__(self, config: LegalLMConfig):
        super().__init__()
        self.attn_norm = RMSNorm(config.d_model, config.rmsnorm_eps)
        self.attn = GroupedQueryAttention(config)
        self.ffn_norm = RMSNorm(config.d_model, config.rmsnorm_eps)
        self.ffn = SwiGLU(config.d_model, config.d_ff)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor, position_ids: torch.Tensor,
                attention_mask: torch.Tensor | None = None,
                past_key_value: tuple[torch.Tensor, torch.Tensor] | None = None,
                use_cache: bool = False) -> tuple[torch.Tensor, tuple | None]:
        attended, cache = self.attn(self.attn_norm(x), position_ids,
                                    attention_mask, past_key_value, use_cache)
        x = x + self.dropout(attended)
        x = x + self.dropout(self.ffn(self.ffn_norm(x)))
        return x, cache
