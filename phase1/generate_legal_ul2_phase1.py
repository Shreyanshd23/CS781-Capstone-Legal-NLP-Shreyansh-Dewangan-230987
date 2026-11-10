#!/usr/bin/env python3
"""Generate exact 128-word legal continuations from a trained LEGAL-UL2 model."""

from __future__ import annotations

import argparse
from contextlib import nullcontext
import json
import os
from pathlib import Path
import time

import torch
from tokenizers import Tokenizer
from tqdm import tqdm

from src.model.legal_ul2 import build_legal_ul2
from train_seq2seq_ddp import sha256


PROFILES = {
    "balanced": {"temperature": 0.8, "top_p": 0.90, "top_k": 50,
                 "repetition_penalty": 1.10},
    "diverse": {"temperature": 0.9, "top_p": 0.95, "top_k": 50,
                "repetition_penalty": 1.10},
    "low_penalty": {"temperature": 0.9, "top_p": 0.90, "top_k": 50,
                    "repetition_penalty": 1.05},
    "high_penalty": {"temperature": 0.9, "top_p": 0.90, "top_k": 50,
                     "repetition_penalty": 1.15},
    "warm": {"temperature": 1.0, "top_p": 0.95, "top_k": 50,
             "repetition_penalty": 1.10},
}


def repeat_trigram_fraction(words):
    triples = [tuple(words[i:i + 3]) for i in range(len(words) - 2)]
    return 1 - len(set(triples)) / len(triples) if triples else 0.0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path,
                        default=Path("data/legal_ul2_full_20261006/tokenizer.json"))
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--profile", choices=PROFILES, required=True)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--target-words", type=int, default=128)
    parser.add_argument("--max-new-tokens", type=int, default=384)
    parser.add_argument("--seed", type=int, default=781)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    if args.output.exists():
        parser.error(f"Output already exists: {args.output}")
    if (args.batch_size < 1 or args.target_words < 1 or args.max_new_tokens < 1 or
            args.num_shards < 1 or not 0 <= args.shard_index < args.num_shards):
        parser.error("Invalid generation argument")
    checkpoint_file = args.checkpoint / "state.pt" if args.checkpoint.is_dir() else args.checkpoint
    state = torch.load(checkpoint_file, map_location="cpu", mmap=True, weights_only=False)
    run = state["run_config"]
    if not torch.cuda.is_available() and not run.get("debug_small", False):
        parser.error("A CUDA GPU is required for full-size LEGAL-UL2 generation")
    if run["architecture"] != "LEGAL-UL2-SEQ2SEQ":
        parser.error("Checkpoint is not LEGAL-UL2")
    if sha256(args.tokenizer) != run["tokenizer_sha256"]:
        parser.error("Tokenizer does not match checkpoint")
    tokenizer = Tokenizer.from_file(str(args.tokenizer))
    if tokenizer.get_vocab_size() != run["vocab_size"]:
        parser.error("Tokenizer vocabulary differs from checkpoint")
    model = build_legal_ul2(run["vocab_size"], tokenizer.token_to_id("<pad>"),
                            tokenizer.token_to_id("<eos>"),
                            small_debug=run.get("debug_small", False))
    model.load_state_dict(state["model"], strict=True)
    checkpoint_step = int(state["step"])
    del state
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()
    torch.manual_seed(args.seed + args.shard_index)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed + args.shard_index)
    records = json.loads(args.input.read_text(encoding="utf-8"))
    if not isinstance(records, dict) or not records:
        parser.error("Input must be a nonempty JSON mapping IDs to prompts")
    items = list(records.items())[args.shard_index::args.num_shards]
    if args.limit is not None:
        items = items[:args.limit]
    if not items:
        parser.error("No prompts in this shard")
    mode_id = tokenizer.token_to_id("<legal_continuation>")
    if mode_id is None:
        parser.error("Continuation mode token is missing")
    suppressed = [tokenizer.token_to_id(name) for name in
                  ("<pad>", "<eos>", "<unk>", "<legal_continuation>",
                   "<legal_span>", "<legal_extreme>")]
    suppressed += [tokenizer.token_to_id(f"<extra_id_{i}>") for i in range(128)]
    suppressed = sorted({value for value in suppressed if value is not None})
    profile = PROFILES[args.profile]
    autocast = (torch.autocast("cuda", dtype=torch.bfloat16)
                if torch.cuda.is_bf16_supported() else nullcontext())
    results = {}
    repeats = []
    truncated = 0
    extended = 0
    started = time.monotonic()
    for offset in tqdm(range(0, len(items), args.batch_size),
                       desc=f"LEGAL-UL2 {args.profile}", unit="batch"):
        batch = items[offset:offset + args.batch_size]
        texts = []
        for identifier, row in batch:
            if not isinstance(identifier, str) or not isinstance(row, dict) or not isinstance(row.get("text"), str):
                raise ValueError(f"Invalid prompt record {identifier!r}")
            texts.append(row["text"])
        encoded = tokenizer.encode_batch(texts, add_special_tokens=False)
        source_ids = []
        for item in encoded:
            ids = item.ids
            if not ids:
                raise ValueError("Prompt tokenized to an empty sequence")
            if len(ids) + 1 > 2048:
                truncated += 1
            source_ids.append([mode_id] + ids[-2047:])
        width = max(len(ids) for ids in source_ids)
        pad_id = tokenizer.token_to_id("<pad>")
        inputs = torch.tensor([ids + [pad_id] * (width - len(ids))
                               for ids in source_ids], dtype=torch.long, device=device)
        mask = torch.tensor([[1] * len(ids) + [0] * (width - len(ids))
                             for ids in source_ids], dtype=torch.long, device=device)
        settings = {"do_sample": True, "max_new_tokens": args.max_new_tokens,
                    "suppress_tokens": suppressed, "pad_token_id": pad_id,
                    **profile}
        with torch.inference_mode(), autocast:
            generated = model.generate(input_ids=inputs, attention_mask=mask,
                                       **settings)
        for row_index, (identifier, _row) in enumerate(batch):
            words = tokenizer.decode(generated[row_index].tolist(),
                                     skip_special_tokens=True).split()
            if len(words) < args.target_words:
                retry_ids = source_ids[row_index]
                retry_input = torch.tensor([retry_ids], dtype=torch.long, device=device)
                retry_mask = torch.ones_like(retry_input)
                for budget, temperature, penalty, no_repeat in (
                    (768, max(1.0, profile["temperature"]),
                     max(1.15, profile["repetition_penalty"]), 0),
                    (1024, 1.15, max(1.20, profile["repetition_penalty"]), 4),
                    (1536, 1.35, max(1.25, profile["repetition_penalty"]), 3),
                ):
                    retry_settings = dict(settings)
                    retry_settings.update(max_new_tokens=budget,
                                          temperature=temperature,
                                          repetition_penalty=penalty,
                                          no_repeat_ngram_size=no_repeat)
                    if budget == 1536:
                        retry_settings.update(top_p=1.0, top_k=0)
                    with torch.inference_mode(), autocast:
                        retry = model.generate(input_ids=retry_input,
                                               attention_mask=retry_mask,
                                               **retry_settings)
                    words = tokenizer.decode(retry[0].tolist(),
                                             skip_special_tokens=True).split()
                    extended += 1
                    if len(words) >= args.target_words:
                        break
            if len(words) < args.target_words:
                raise RuntimeError(f"{identifier}: only {len(words)} words after 1536 tokens")
            words = words[:args.target_words]
            results[identifier] = {"text": " ".join(words)}
            repeats.append(repeat_trigram_fraction(words))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(results, ensure_ascii=False, indent=2) + "\n",
                         encoding="utf-8")
    os.replace(temporary, args.output)
    metadata = {"checkpoint": str(checkpoint_file.resolve()),
                "checkpoint_step": checkpoint_step, "profile": args.profile,
                "settings": profile, "input": str(args.input.resolve()),
                "output": str(args.output.resolve()), "examples": len(results),
                "target_words": args.target_words,
                "truncated_prompts": truncated,
                "extended_generations": extended,
                "mean_repeated_trigram_fraction": sum(repeats) / len(repeats),
                "generation_seconds": time.monotonic() - started,
                "seed": args.seed, "shard_index": args.shard_index,
                "num_shards": args.num_shards}
    args.output.with_suffix(".metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(f"Saved {len(results)} exact-length continuations to {args.output}; "
          f"truncated={truncated}; repeated trigrams={metadata['mean_repeated_trigram_fraction']:.3f}",
          flush=True)


if __name__ == "__main__":
    main()
