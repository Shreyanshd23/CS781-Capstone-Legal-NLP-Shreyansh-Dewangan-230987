"""Full causal grouped-query attention using PyTorch SDPA."""

import torch
from torch import nn
from torch.nn import functional as F

from .config import LegalLMConfig
from .rope import RotaryEmbedding


class GroupedQueryAttention(nn.Module):
    def __init__(self, config: LegalLMConfig):
        super().__init__()
        self.n_query_heads = config.n_query_heads
        self.n_kv_heads = config.n_kv_heads
        self.head_dim = config.head_dim
        self.q_proj = nn.Linear(config.d_model,
                                config.n_query_heads * config.head_dim, bias=False)
        self.k_proj = nn.Linear(config.d_model,
                                config.n_kv_heads * config.head_dim, bias=False)
        self.v_proj = nn.Linear(config.d_model,
                                config.n_kv_heads * config.head_dim, bias=False)
        self.o_proj = nn.Linear(config.d_model, config.d_model, bias=False)
        self.rope = RotaryEmbedding(config.head_dim, config.context_length,
                                    config.rope_theta)
        self.bidirectional = False

    def forward(self, x: torch.Tensor, position_ids: torch.Tensor,
                attention_mask: torch.Tensor | None = None,
                past_key_value: tuple[torch.Tensor, torch.Tensor] | None = None,
                use_cache: bool = False) -> tuple[torch.Tensor, tuple | None]:
        batch, time, _ = x.shape
        q = self.q_proj(x).view(batch, time, self.n_query_heads,
                                self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch, time, self.n_kv_heads,
                                self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch, time, self.n_kv_heads,
                                self.head_dim).transpose(1, 2)
        q, k = self.rope(q, k, position_ids)
        if past_key_value is not None:
            old_k, old_v = past_key_value
            k = torch.cat((old_k, k), dim=-2)
            v = torch.cat((old_v, v), dim=-2)
        key_length = k.shape[-2]
        if key_length > self.rope.cos_cache.shape[0]:
            raise ValueError("Sequence exceeds configured 4096-token context")
        if self.bidirectional and (past_key_value is not None or use_cache):
            raise ValueError("Bidirectional attention does not support KV caching")

        if attention_mask is None and past_key_value is None:
            mask = None
            causal = not self.bidirectional
        else:
            causal = False
            if attention_mask is None:
                valid_keys = torch.ones((batch, key_length), dtype=torch.bool,
                                        device=x.device)
            else:
                if attention_mask.shape != (batch, key_length):
                    raise ValueError("attention_mask must cover all cached keys")
                valid_keys = attention_mask.bool()
            mask = valid_keys[:, None, None, :]
            if time > 1 and not self.bidirectional:
                past_length = key_length - time
                q_positions = torch.arange(time, device=x.device) + past_length
                k_positions = torch.arange(key_length, device=x.device)
                causal_keys = k_positions[None, :] <= q_positions[:, None]
                mask = mask & causal_keys[None, None, :, :]
        attended = F.scaled_dot_product_attention(
            q, k, v, attn_mask=mask, is_causal=causal,
            dropout_p=0.0, enable_gqa=True)
        merged = attended.transpose(1, 2).contiguous().view(batch, time, -1)
        return self.o_proj(merged), ((k, v) if use_cache else None)
