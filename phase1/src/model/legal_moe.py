"""Sparse Top-1 extension of the proven LegalLM decoder.

Only FFNs in zero-based blocks 3, 7, 11, and 15 are replaced. All
attention, normalization, positional encoding, and generation code is reused.
"""

from __future__ import annotations

import torch
from torch import nn

from .config import LegalLMConfig
from .legal_lm import LegalLM, LMOutput
from .swiglu import SwiGLU


MOE_INDICES = (3, 7, 11, 15)


class SparseTop1SwiGLU(nn.Module):
    def __init__(self, width: int, hidden: int, experts: int = 4):
        super().__init__()
        self.router = nn.Linear(width, experts, bias=False)
        self.experts = nn.ModuleList(SwiGLU(width, hidden) for _ in range(experts))
        self.expert_count = experts
        self.aux_loss: torch.Tensor | None = None
        self.last_counts: torch.Tensor | None = None
        nn.init.normal_(self.router.weight, std=0.01)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shape = x.shape
        flat = x.reshape(-1, shape[-1])
        # Autocast must not round the router softmax to BF16.
        with torch.autocast(device_type=x.device.type, enabled=False):
            logits = self.router(flat.float())
            probabilities = logits.softmax(dim=-1)
            chosen = probabilities.argmax(dim=-1)
            selected_probability = probabilities.gather(1, chosen[:, None]).squeeze(1)
            counts = torch.bincount(chosen, minlength=self.expert_count)
            fraction = counts.float().detach() / max(flat.shape[0], 1)
            self.aux_loss = (self.expert_count * fraction * probabilities.mean(0)).sum()
            self.last_counts = counts.detach()

        output = torch.zeros_like(flat)
        for expert_id, expert in enumerate(self.experts):
            indices = torch.nonzero(chosen == expert_id, as_tuple=True)[0]
            if indices.numel():
                values = expert(flat.index_select(0, indices))
                weighted = values * selected_probability.index_select(0, indices)[:, None].to(values.dtype)
                output = output.index_add(0, indices, weighted.to(output.dtype))
        return output.reshape(shape)


class LegalMoELM(LegalLM):
    architecture_name = "Legal-MoE-251M Top-1 decoder-only RoPE GQA SwiGLU Transformer"

    def __init__(self, config: LegalLMConfig, router_aux_weight: float = 0.01):
        if config.n_layers != 16:
            raise ValueError("Legal-MoE requires the existing 16-layer backbone")
        super().__init__(config)
        self.router_aux_weight = router_aux_weight
        for index in MOE_INDICES:
            self.blocks[index].ffn = SparseTop1SwiGLU(config.d_model, config.d_ff)

    def forward(self, *args, **kwargs) -> LMOutput:
        result = super().forward(*args, **kwargs)
        if result.loss is not None:
            aux = torch.stack([self.blocks[i].ffn.aux_loss for i in MOE_INDICES]).mean()
            result.loss = result.loss + self.router_aux_weight * aux
        return result

    @torch.no_grad()
    def initialize_from_dense(self, state: dict[str, torch.Tensor]) -> None:
        """Copy the trained dense model, expanding each replaced FFN to four experts.

        Scaling each expert's output projection by four compensates for an
        approximately uniform initial router probability of one quarter.
        """
        current = self.state_dict()
        for key in current:
            if any(key.startswith(f"blocks.{i}.ffn.") for i in MOE_INDICES):
                continue
            if key not in state or current[key].shape != state[key].shape:
                raise ValueError(f"Dense checkpoint mismatch: {key}")
            current[key] = state[key]
        for index in MOE_INDICES:
            prefix = f"blocks.{index}.ffn."
            for expert_id in range(4):
                for name in ("gate.weight", "up.weight", "down.weight"):
                    source = state[prefix + name]
                    target = f"{prefix}experts.{expert_id}.{name}"
                    current[target] = source * 4 if name == "down.weight" else source
        self.load_state_dict(current, strict=True)
