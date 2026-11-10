"""Validated configuration for the from-scratch legal continuation model."""

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class LegalLMConfig:
    vocab_size: int
    d_model: int = 896
    n_layers: int = 16
    n_query_heads: int = 14
    n_kv_heads: int = 2
    head_dim: int = 64
    d_ff: int = 2560
    context_length: int = 4096
    dropout: float = 0.05
    rmsnorm_eps: float = 1e-5
    rope_theta: float = 10000.0
    pad_token_id: int = 0
    eos_token_id: int = 2
    bos_token_id: int = 2

    def __post_init__(self) -> None:
        if self.d_model != self.n_query_heads * self.head_dim:
            raise ValueError("d_model must equal query heads times head_dim")
        if self.n_query_heads % self.n_kv_heads:
            raise ValueError("Query heads must divide evenly across KV heads")
        if self.head_dim % 2:
            raise ValueError("RoPE requires an even head_dim")
        if min(self.vocab_size, self.n_layers, self.d_ff, self.context_length) < 1:
            raise ValueError("Vocabulary and model sizes must be positive")
        if not 0 <= self.dropout < 1:
            raise ValueError("Dropout must be in [0,1)")

    def to_dict(self) -> dict:
        return asdict(self)
