"""Randomly initialized T5-style legal encoder-decoder with UL2 objectives.

The installed Transformers implementation creates separate encoder and decoder
embeddings when the LM head is untied. Explicitly reattach both to ``shared``
to satisfy the requested shared input embedding plus separate output head.
"""

from __future__ import annotations

from transformers import T5Config, T5ForConditionalGeneration


def build_legal_ul2(vocab_size: int, pad_id: int, eos_id: int,
                    small_debug: bool = False) -> T5ForConditionalGeneration:
    if small_debug:
        width, layers, heads, head_dim, d_ff = 64, 2, 4, 16, 128
    else:
        width, layers, heads, head_dim, d_ff = 768, 12, 12, 64, 2048
    config = T5Config(vocab_size=vocab_size, d_model=width, d_kv=head_dim,
                      d_ff=d_ff, num_layers=layers, num_decoder_layers=layers,
                      num_heads=heads, feed_forward_proj="gated-gelu",
                      relative_attention_num_buckets=32,
                      relative_attention_max_distance=128,
                      dropout_rate=0.05, layer_norm_epsilon=1e-6,
                      pad_token_id=pad_id, eos_token_id=eos_id,
                      decoder_start_token_id=pad_id, use_cache=True)
    config.tie_word_embeddings = False
    config._attn_implementation = "sdpa"
    model = T5ForConditionalGeneration(config)
    model.encoder.set_input_embeddings(model.shared)
    model.decoder.set_input_embeddings(model.shared)
    if model.encoder.embed_tokens.weight is not model.decoder.embed_tokens.weight:
        raise RuntimeError("Encoder and decoder input embeddings are not shared")
    if model.lm_head.weight is model.shared.weight:
        raise RuntimeError("The LM head must be separate")
    if model.config.tie_word_embeddings:
        raise RuntimeError("The LM head is configured as tied")
    return model
