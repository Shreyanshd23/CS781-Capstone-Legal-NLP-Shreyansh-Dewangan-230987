"""From-scratch decoder-only legal language model with tied output weights."""

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F

from .block import LegalBlock
from .config import LegalLMConfig
from .rmsnorm import RMSNorm


@dataclass
class LMOutput:
    logits: torch.Tensor
    loss: torch.Tensor | None = None
    past_key_values: tuple | None = None


class LegalLM(nn.Module):
    def __init__(self, config: LegalLMConfig):
        super().__init__()
        self.config = config
        self.token_embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.embedding_dropout = nn.Dropout(config.dropout)
        self.blocks = nn.ModuleList(LegalBlock(config) for _ in range(config.n_layers))
        self.final_norm = RMSNorm(config.d_model, config.rmsnorm_eps)
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    @property
    def lm_head_weight(self) -> torch.Tensor:
        return self.token_embedding.weight

    def forward(self, input_ids: torch.Tensor,
                labels: torch.Tensor | None = None,
                attention_mask: torch.Tensor | None = None,
                past_key_values: tuple | None = None,
                use_cache: bool = False) -> LMOutput:
        batch, time = input_ids.shape
        past_length = 0 if past_key_values is None else past_key_values[0][0].shape[-2]
        if time < 1 or past_length + time > self.config.context_length:
            raise ValueError("Input exceeds configured context length")
        if past_key_values is not None and len(past_key_values) != len(self.blocks):
            raise ValueError("One KV cache entry is needed per block")
        if attention_mask is None:
            position_ids = torch.arange(past_length, past_length + time,
                                        device=input_ids.device).expand(batch, -1)
        else:
            if attention_mask.shape != (batch, past_length + time):
                raise ValueError("attention_mask must include every cached position")
            position_ids = (attention_mask.long().cumsum(-1) - 1).clamp_min(0)[:, -time:]
        hidden = self.embedding_dropout(self.token_embedding(input_ids))
        caches = []
        for index, block in enumerate(self.blocks):
            past = None if past_key_values is None else past_key_values[index]
            hidden, cache = block(hidden, position_ids, attention_mask, past, use_cache)
            if use_cache:
                caches.append(cache)
        hidden = self.final_norm(hidden)
        logits = F.linear(hidden, self.token_embedding.weight)
        loss = None
        if labels is not None:
            if labels.shape != input_ids.shape:
                raise ValueError("labels must match input_ids")
            loss = F.cross_entropy(logits[:, :-1].contiguous().float().view(-1, self.config.vocab_size),
                                   labels[:, 1:].contiguous().view(-1), ignore_index=-100)
        return LMOutput(logits, loss, tuple(caches) if use_cache else None)

    @torch.inference_mode()
    def generate(self, input_ids: torch.Tensor, max_new_tokens: int,
                 eos_token_id: int | None = None, pad_token_id: int | None = None,
                 temperature: float = 1.0, top_p: float = 1.0,
                 top_k: int = 0, repetition_penalty: float = 1.0,
                 attention_mask: torch.Tensor | None = None,
                 suppress_tokens: list[int] | None = None,
                 do_sample: bool = True, **_: object) -> torch.Tensor:
        if (max_new_tokens < 0 or temperature < 0 or not 0 < top_p <= 1
                or top_k < 0 or repetition_penalty < 1):
            raise ValueError("Invalid generation setting")
        output = input_ids
        mask = attention_mask
        cache = None
        finished = torch.zeros(input_ids.shape[0], dtype=torch.bool, device=input_ids.device)
        for _step in range(max_new_tokens):
            if output.shape[1] >= self.config.context_length:
                break
            current = output if cache is None else output[:, -1:]
            prediction = self(current, attention_mask=mask,
                              past_key_values=cache, use_cache=True)
            cache = prediction.past_key_values
            logits = prediction.logits[:, -1].float()
            if repetition_penalty != 1:
                for row in range(output.shape[0]):
                    seen = output[row] if mask is None else output[row, mask[row].bool()]
                    ids = seen.unique()
                    values = logits[row, ids]
                    logits[row, ids] = torch.where(values < 0, values * repetition_penalty,
                                                     values / repetition_penalty)
            if suppress_tokens:
                logits[:, suppress_tokens] = -float("inf")
            if temperature == 0 or not do_sample:
                next_id = logits.argmax(-1, keepdim=True)
            else:
                logits = logits / temperature
                if top_k > 0 and top_k < logits.shape[-1]:
                    cutoff = logits.topk(top_k, dim=-1).values[:, -1:]
                    logits = logits.masked_fill(logits < cutoff, -float("inf"))
                if top_p < 1.0:
                    sorted_logits, indices = logits.sort(descending=True)
                    probabilities = sorted_logits.softmax(-1)
                    remove = (probabilities.cumsum(-1) - probabilities) >= top_p
                    sorted_logits.masked_fill_(remove, -float("inf"))
                    logits = torch.full_like(logits, -float("inf")).scatter(-1, indices, sorted_logits)
                next_id = torch.multinomial(logits.softmax(-1), 1)
            if eos_token_id is not None:
                replacement = eos_token_id if pad_token_id is None else pad_token_id
                next_id = torch.where(finished[:, None], replacement, next_id)
            output = torch.cat((output, next_id), -1)
            if mask is not None:
                mask = torch.cat((mask, torch.ones_like(next_id, dtype=mask.dtype)), -1)
            if eos_token_id is not None:
                finished |= next_id[:, 0].eq(eos_token_id)
            if finished.all() and eos_token_id is not None:
                break
        return output
