"""From-scratch encoder-decoder Transformer for legal text continuation."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
from torch import nn
from torch.nn import functional as F

from .rmsnorm import RMSNorm
from .rope import RotaryEmbedding
from .swiglu import SwiGLU


@dataclass(frozen=True)
class LegalSeq2SeqConfig:
    vocab_size: int
    d_model: int = 768
    n_heads: int = 12
    n_encoder_layers: int = 8
    n_decoder_layers: int = 8
    d_ff: int = 2304
    encoder_context: int = 1024
    decoder_context: int = 768
    dropout: float = 0.05
    pad_token_id: int = 0
    eos_token_id: int = 2
    bos_token_id: int = 2

    def __post_init__(self):
        if min(self.vocab_size, self.d_model, self.n_heads, self.n_encoder_layers,
               self.n_decoder_layers, self.d_ff, self.encoder_context,
               self.decoder_context) < 1:
            raise ValueError("Model dimensions must be positive")
        if self.d_model % self.n_heads or (self.d_model // self.n_heads) % 2:
            raise ValueError("Attention head dimension must be even")
        if not 0 <= self.dropout < 1:
            raise ValueError("dropout must be in [0,1)")

    def to_dict(self):
        return asdict(self)


class SelfAttention(nn.Module):
    def __init__(self, cfg: LegalSeq2SeqConfig, causal: bool):
        super().__init__()
        self.heads = cfg.n_heads
        self.dim = cfg.d_model // cfg.n_heads
        self.causal = causal
        self.q = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.k = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.v = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.o = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.rope = RotaryEmbedding(self.dim,
                                   cfg.decoder_context if causal else cfg.encoder_context)

    def forward(self, x: torch.Tensor, past: tuple | None = None,
                cache: bool = False):
        batch, length, width = x.shape
        offset = 0 if past is None else past[0].shape[-2]
        if offset + length > self.rope.cos_cache.shape[0]:
            raise ValueError("Sequence exceeds attention context")
        positions = torch.arange(offset, offset + length, device=x.device).expand(batch, -1)
        q = self.q(x).view(batch, length, self.heads, self.dim).transpose(1, 2)
        k = self.k(x).view(batch, length, self.heads, self.dim).transpose(1, 2)
        v = self.v(x).view(batch, length, self.heads, self.dim).transpose(1, 2)
        q, k = self.rope(q, k, positions)
        if past is not None:
            k = torch.cat((past[0], k), dim=-2)
            v = torch.cat((past[1], v), dim=-2)
        attended = F.scaled_dot_product_attention(
            q, k, v, is_causal=self.causal and past is None, dropout_p=0.0)
        out = self.o(attended.transpose(1, 2).contiguous().view(batch, length, width))
        return out, ((k, v) if cache else None)


class CrossAttention(nn.Module):
    def __init__(self, cfg: LegalSeq2SeqConfig):
        super().__init__()
        self.heads = cfg.n_heads
        self.dim = cfg.d_model // cfg.n_heads
        self.q = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.k = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.v = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.o = nn.Linear(cfg.d_model, cfg.d_model, bias=False)

    def forward(self, x: torch.Tensor, memory: torch.Tensor,
                projected_memory: tuple | None = None):
        batch, length, width = x.shape
        q = self.q(x).view(batch, length, self.heads, self.dim).transpose(1, 2)
        if projected_memory is None:
            source_length = memory.shape[1]
            k = self.k(memory).view(batch, source_length, self.heads, self.dim).transpose(1, 2)
            v = self.v(memory).view(batch, source_length, self.heads, self.dim).transpose(1, 2)
        else:
            k, v = projected_memory
        out = F.scaled_dot_product_attention(q, k, v, dropout_p=0.0)
        return self.o(out.transpose(1, 2).contiguous().view(batch, length, width)), (k, v)


class EncoderBlock(nn.Module):
    def __init__(self, cfg: LegalSeq2SeqConfig):
        super().__init__()
        self.attn_norm = RMSNorm(cfg.d_model)
        self.attn = SelfAttention(cfg, causal=False)
        self.ffn_norm = RMSNorm(cfg.d_model)
        self.ffn = SwiGLU(cfg.d_model, cfg.d_ff)
        self.dropout = nn.Dropout(cfg.dropout)

    def forward(self, x: torch.Tensor):
        attended, _ = self.attn(self.attn_norm(x))
        x = x + self.dropout(attended)
        return x + self.dropout(self.ffn(self.ffn_norm(x)))


class DecoderBlock(nn.Module):
    def __init__(self, cfg: LegalSeq2SeqConfig):
        super().__init__()
        self.self_norm = RMSNorm(cfg.d_model)
        self.self_attn = SelfAttention(cfg, causal=True)
        self.cross_norm = RMSNorm(cfg.d_model)
        self.cross_attn = CrossAttention(cfg)
        self.ffn_norm = RMSNorm(cfg.d_model)
        self.ffn = SwiGLU(cfg.d_model, cfg.d_ff)
        self.dropout = nn.Dropout(cfg.dropout)

    def forward(self, x: torch.Tensor, memory: torch.Tensor,
                past: tuple | None = None, cross_kv: tuple | None = None,
                cache: bool = False):
        attended, new_past = self.self_attn(self.self_norm(x), past, cache)
        x = x + self.dropout(attended)
        crossed, new_cross_kv = self.cross_attn(self.cross_norm(x), memory, cross_kv)
        x = x + self.dropout(crossed)
        x = x + self.dropout(self.ffn(self.ffn_norm(x)))
        return x, new_past, new_cross_kv


class LegalSeq2Seq(nn.Module):
    def __init__(self, config: LegalSeq2SeqConfig):
        super().__init__()
        self.config = config
        self.token_embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.embedding_dropout = nn.Dropout(config.dropout)
        self.encoder = nn.ModuleList(EncoderBlock(config) for _ in range(config.n_encoder_layers))
        self.decoder = nn.ModuleList(DecoderBlock(config) for _ in range(config.n_decoder_layers))
        self.encoder_norm = RMSNorm(config.d_model)
        self.decoder_norm = RMSNorm(config.d_model)
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def encode(self, input_ids: torch.Tensor):
        if input_ids.ndim != 2 or input_ids.shape[1] > self.config.encoder_context:
            raise ValueError("Invalid encoder input length")
        x = self.embedding_dropout(self.token_embedding(input_ids))
        for block in self.encoder:
            x = block(x)
        return self.encoder_norm(x)

    def decode(self, decoder_ids: torch.Tensor, memory: torch.Tensor,
               past: tuple | None = None, cross_kv: tuple | None = None,
               cache: bool = False):
        if decoder_ids.ndim != 2:
            raise ValueError("decoder_ids must be [batch, time]")
        x = self.embedding_dropout(self.token_embedding(decoder_ids))
        next_past, next_cross = [], []
        for i, block in enumerate(self.decoder):
            x, p, c = block(x, memory, None if past is None else past[i],
                            None if cross_kv is None else cross_kv[i], cache)
            if cache:
                next_past.append(p)
                next_cross.append(c)
        logits = F.linear(self.decoder_norm(x), self.token_embedding.weight)
        return logits, (tuple(next_past) if cache else None), (tuple(next_cross) if cache else None)

    def forward(self, input_ids: torch.Tensor, labels: torch.Tensor):
        if labels.ndim != 2 or labels.shape[0] != input_ids.shape[0]:
            raise ValueError("labels must be [batch, target_time]")
        if labels.shape[1] > self.config.decoder_context:
            raise ValueError("Target exceeds decoder context")
        bos = torch.full_like(labels[:, :1], self.config.bos_token_id)
        decoder_ids = torch.cat((bos, labels[:, :-1]), dim=1)
        memory = self.encode(input_ids)
        logits, _, _ = self.decode(decoder_ids, memory)
        loss = F.cross_entropy(logits.float().reshape(-1, self.config.vocab_size),
                               labels.reshape(-1), ignore_index=-100)
        return logits, loss

    @torch.inference_mode()
    def generate(self, input_ids: torch.Tensor, max_new_tokens: int,
                 temperature: float = 0.9, top_p: float = 0.9,
                 top_k: int = 50, repetition_penalty: float = 1.1,
                 suppress_tokens: tuple[int, ...] = ()):
        if input_ids.shape[1] > self.config.encoder_context:
            input_ids = input_ids[:, -self.config.encoder_context:]
        if max_new_tokens > self.config.decoder_context:
            raise ValueError("max_new_tokens exceeds decoder context")
        memory = self.encode(input_ids)
        out = torch.empty((input_ids.shape[0], 0), dtype=torch.long, device=input_ids.device)
        current = torch.full((input_ids.shape[0], 1), self.config.bos_token_id,
                             dtype=torch.long, device=input_ids.device)
        past = cross_kv = None
        for _ in range(max_new_tokens):
            logits, past, cross_kv = self.decode(current, memory, past, cross_kv, cache=True)
            scores = logits[:, -1, :].float()
            if repetition_penalty != 1:
                for row in range(out.shape[0]):
                    ids = torch.cat((input_ids[row], out[row])).unique()
                    values = scores[row, ids]
                    scores[row, ids] = torch.where(values < 0, values * repetition_penalty,
                                                   values / repetition_penalty)
            if suppress_tokens:
                scores[:, list(suppress_tokens)] = -float("inf")
            if temperature == 0:
                current = scores.argmax(-1, keepdim=True)
            else:
                scores = scores / temperature
                if top_k and top_k < scores.shape[-1]:
                    cutoff = scores.topk(top_k, dim=-1).values[:, -1:]
                    scores.masked_fill_(scores < cutoff, -float("inf"))
                if top_p < 1:
                    ordered, indices = scores.sort(descending=True)
                    remove = (ordered.softmax(-1).cumsum(-1) - ordered.softmax(-1)) >= top_p
                    ordered.masked_fill_(remove, -float("inf"))
                    scores = torch.full_like(scores, -float("inf")).scatter(-1, indices, ordered)
                current = torch.multinomial(scores.softmax(-1), 1)
            out = torch.cat((out, current), dim=1)
        return out
