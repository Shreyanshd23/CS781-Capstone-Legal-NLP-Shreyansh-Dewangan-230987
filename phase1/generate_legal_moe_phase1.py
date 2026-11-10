#!/usr/bin/env python3
"""Generate Phase-1 continuations from a locally trained custom RoPE + GQA checkpoint.

Input/output are JSON objects mapping exact IDs to {"text": ...}. An optional
shard processes IDs at positions congruent to --shard-index modulo
--num-shards, allowing four independent GPU processes to generate in parallel.
"""

from __future__ import annotations

import argparse
from contextlib import nullcontext
import hashlib
import json
import os
from pathlib import Path
import time

import torch
from tokenizers import Tokenizer
from tqdm import tqdm
from src.model import LegalLMConfig
from src.model.legal_moe import LegalMoELM


DECODING = {
    "base": {"do_sample": True, "temperature": 0.1,
             "top_p": 0.95, "top_k": 50, "repetition_penalty": 1.0},
    "starter": {"do_sample": True, "temperature": 0.1,
                "top_p": 0.95, "top_k": 50, "repetition_penalty": 1.0},
    "balanced": {"do_sample": True, "temperature": 0.7,
                 "top_p": 0.90, "top_k": 50, "repetition_penalty": 1.05},
    "diverse": {"do_sample": True, "temperature": 0.9,
                "top_p": 0.95, "top_k": 50, "repetition_penalty": 1.10},
}

# Controlled sweep around the previous diverse decoder. Keep top-p and top-k
# fixed so temperature and repetition penalty are the only changing factors.
for _temperature in (0.80, 0.85, 0.90):
    for _penalty in (1.10, 1.15):
        DECODING[f"t{round(_temperature * 100):03d}_r{round(_penalty * 100):03d}"] = {
            "do_sample": True, "temperature": _temperature,
            "top_p": 0.95, "top_k": 50,
            "repetition_penalty": _penalty,
        }

# Second sweep: broader sampling controls, with the prior private-test winner
# t090_r110 included as a fixed baseline on the fresh validation documents.
DECODING.update({
    "t095_p095_k50_r110": {"do_sample": True, "temperature": 0.95,
                            "top_p": 0.95, "top_k": 50, "repetition_penalty": 1.10},
    "t100_p095_k50_r110": {"do_sample": True, "temperature": 1.00,
                            "top_p": 0.95, "top_k": 50, "repetition_penalty": 1.10},
    "t090_p090_k50_r110": {"do_sample": True, "temperature": 0.90,
                            "top_p": 0.90, "top_k": 50, "repetition_penalty": 1.10},
    "t090_p098_k50_r110": {"do_sample": True, "temperature": 0.90,
                            "top_p": 0.98, "top_k": 50, "repetition_penalty": 1.10},
    "t090_p095_k00_r110": {"do_sample": True, "temperature": 0.90,
                            "top_p": 0.95, "top_k": 0, "repetition_penalty": 1.10},
    "t090_p095_k50_r105": {"do_sample": True, "temperature": 0.90,
                            "top_p": 0.95, "top_k": 50, "repetition_penalty": 1.05},
})

RESCUE = {"do_sample": True, "temperature": 1.1,
          "top_p": 0.98, "top_k": 50, "repetition_penalty": 1.20}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_model(checkpoint: Path, tokenizer_path: Path, device: torch.device):
    checkpoint_file = checkpoint / "state.pt" if checkpoint.is_dir() else checkpoint
    state = torch.load(checkpoint_file, map_location="cpu", mmap=True,
                       weights_only=False)
    run = state["run_config"]
    if run["tokenizer_sha256"] != sha256(tokenizer_path):
        raise ValueError("Checkpoint and tokenizer hashes differ")
    special = run["special_token_ids"]
    if run.get("architecture") != LegalMoELM.architecture_name:
        raise ValueError("Expected a Legal-MoE checkpoint")
    config = LegalLMConfig(**run["model_config"])
    model = LegalMoELM(config, router_aux_weight=run.get("router_aux_weight", 0.01))
    model.load_state_dict(state["model"], strict=True)
    model.to(device).eval()
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    if tokenizer.get_vocab_size() != run["vocab_size"]:
        raise ValueError("Tokenizer vocabulary size differs from checkpoint")
    return model, tokenizer, run, int(state["step"]), special


def repeat_trigram_fraction(words: list[str]) -> float:
    trigrams = [tuple(words[i:i + 3]) for i in range(len(words) - 2)]
    return 1 - len(set(trigrams)) / len(trigrams) if trigrams else 0.0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--decoding", choices=DECODING, required=True)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--target-words", type=int, default=128)
    parser.add_argument("--max-new-tokens", type=int, default=384)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--limit", type=int, help="For a small pipeline smoke check")
    args = parser.parse_args()
    if (args.batch_size < 1 or args.target_words < 1 or args.max_new_tokens < 1 or
            args.num_shards < 1 or not 0 <= args.shard_index < args.num_shards):
        parser.error("Invalid batch, length, or shard argument")
    if args.output.exists():
        parser.error(f"Output already exists: {args.output}")
    if not torch.cuda.is_available():
        parser.error("CUDA is required for practical checkpoint generation")

    inputs = json.loads(args.input.read_text(encoding="utf-8"))
    if not isinstance(inputs, dict) or not inputs:
        parser.error("Input must be a nonempty JSON object")
    items = []
    for index, (identifier, record) in enumerate(inputs.items()):
        if not isinstance(identifier, str) or not isinstance(record, dict) or not isinstance(record.get("text"), str):
            parser.error(f"Invalid prompt record: {identifier!r}")
        if index % args.num_shards == args.shard_index:
            items.append((identifier, record["text"]))
    if args.limit is not None:
        items = items[:args.limit]
    if not items:
        parser.error("No prompts belong to this shard")

    torch.manual_seed(args.seed + args.shard_index)
    torch.cuda.manual_seed_all(args.seed + args.shard_index)
    device = torch.device("cuda:0")
    model, tokenizer, run, step, special = load_model(args.checkpoint, args.tokenizer, device)
    max_input_tokens = run["block_size"] - args.max_new_tokens
    if max_input_tokens < 1:
        parser.error("Generation budget exceeds model context")
    generation_settings = dict(DECODING[args.decoding])
    generation_settings.update({
        "max_new_tokens": args.max_new_tokens,
        "pad_token_id": special["<pad>"],
        "eos_token_id": special["<eos>"],
        # Each answer must reach 128 words. Suppress the tokenizer's special
        # symbols so EOS cannot cut off the continuation early.
        "suppress_tokens": [special["<pad>"], special["<unk>"], special["<eos>"]],
        "use_cache": True,
    })
    answers = {}
    truncated_prompts = 0
    repetitions = []
    short_initial = 0
    fallback_profiles = {}
    started = time.monotonic()
    autocast = (torch.autocast("cuda", dtype=torch.bfloat16)
                if torch.cuda.is_bf16_supported() else nullcontext())
    for offset in tqdm(range(0, len(items), args.batch_size),
                       desc=f"{args.decoding} shard {args.shard_index}", unit="batch"):
        batch = items[offset:offset + args.batch_size]
        encodings = tokenizer.encode_batch([prompt for _, prompt in batch],
                                           add_special_tokens=False)
        token_ids = []
        for encoding in encodings:
            ids = encoding.ids
            if not ids:
                raise ValueError("Prompt tokenizer returned no IDs")
            if len(ids) > max_input_tokens:
                truncated_prompts += 1
            token_ids.append(ids[-max_input_tokens:])
        width = max(map(len, token_ids))
        pad = special["<pad>"]
        input_ids = torch.tensor([[pad] * (width - len(ids)) + ids
                                  for ids in token_ids], dtype=torch.long, device=device)
        attention_mask = torch.tensor([[0] * (width - len(ids)) + [1] * len(ids)
                                       for ids in token_ids], dtype=torch.long, device=device)
        with torch.inference_mode(), autocast:
            generated = model.generate(input_ids=input_ids,
                                       attention_mask=attention_mask,
                                       **generation_settings)
        for row_index, (identifier, _) in enumerate(batch):
            # The generated suffix begins after the *padded batch width*, not
            # after the unpadded prompt length.
            suffix = generated[row_index, width:].tolist()
            words = tokenizer.decode(suffix, skip_special_tokens=True).split()
            if len(words) < args.target_words:
                short_initial += 1
                # A legal citation can consume far more than three BPE tokens
                # per whitespace word. Retry from the original prompt with a
                # larger budget, then progressively stronger anti-looping
                # settings. Apply the same policy to held-out and test inputs.
                original_ids = token_ids[row_index]
                extended_budget = min(768, run["block_size"] - len(original_ids))
                retry_profiles = [(args.decoding, DECODING[args.decoding])]
                retry_profiles += [(name, DECODING[name]) for name in
                                   ("balanced", "diverse") if name != args.decoding]
                retry_profiles.append(("rescue", RESCUE))
                for profile_name, profile in retry_profiles:
                    if extended_budget <= args.max_new_tokens:
                        break
                    retry_settings = dict(generation_settings)
                    retry_settings.update(profile)
                    retry_settings["max_new_tokens"] = extended_budget
                    retry_input = torch.tensor([original_ids], dtype=torch.long,
                                               device=device)
                    with torch.inference_mode(), autocast:
                        retry_output = model.generate(input_ids=retry_input,
                                                      **retry_settings)
                    retry_suffix = retry_output[0, len(original_ids):].tolist()
                    retry_words = tokenizer.decode(retry_suffix,
                                                   skip_special_tokens=True).split()
                    if len(retry_words) >= args.target_words:
                        words = retry_words
                        fallback_profiles[profile_name] = fallback_profiles.get(profile_name, 0) + 1
                        break
                if len(words) < args.target_words:
                    raise RuntimeError(
                        f"{identifier}: only {len(words)} words after {args.max_new_tokens} "
                        f"tokens; extended budget {extended_budget} also failed")
            words = words[:args.target_words]
            answers[identifier] = {"text": " ".join(words)}
            repetitions.append(repeat_trigram_fraction(words))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(answers, ensure_ascii=False, indent=2) + "\n",
                         encoding="utf-8")
    os.replace(temporary, args.output)
    metadata = {
        "checkpoint": str(args.checkpoint.resolve()), "checkpoint_step": step,
        "tokenizer": str(args.tokenizer.resolve()), "tokenizer_sha256": sha256(args.tokenizer),
        "input": str(args.input.resolve()), "input_sha256": sha256(args.input),
        "output": str(args.output.resolve()), "decoding": args.decoding,
        "settings": generation_settings, "seed": args.seed,
        "shard_index": args.shard_index, "num_shards": args.num_shards,
        "examples": len(answers), "target_words": args.target_words,
        "truncated_prompts": truncated_prompts,
        "short_initial_generations": short_initial,
        "fallback_profiles": fallback_profiles,
        "mean_repeated_trigram_fraction": sum(repetitions) / len(repetitions),
        "generation_seconds": time.monotonic() - started,
    }
    args.output.with_suffix(".metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(f"Saved {len(answers)} continuations to {args.output}; "
          f"truncated prompts={truncated_prompts}; "
          f"mean repeated trigram fraction={metadata['mean_repeated_trigram_fraction']:.3f}",
          flush=True)


if __name__ == "__main__":
    main()
