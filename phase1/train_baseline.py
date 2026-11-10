"""Train a randomly initialized GPT-2-style Phase-1 causal LM on packed legal text.

Default run: one pass over 200M sampled train tokens, 5M held-out validation
tokens, 2048 context, 8 layers, 8 heads, width 576. Width 576 with our 32k
tokenizer to clear the competition's 50M parameter minimum.
"""

from __future__ import annotations

import argparse
from contextlib import nullcontext
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import random
import time

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Sampler
from transformers import GPT2Config, GPT2LMHeadModel


class TokenBlocks(Dataset):
    def __init__(self, path: Path, dtype: str, block_size: int, max_tokens: int):
        self.data = np.memmap(path, dtype=np.dtype(dtype), mode="r")
        self.block_size = block_size
        self.available_blocks = len(self.data) // block_size
        self.blocks = min(self.available_blocks, max_tokens // block_size)
        if self.blocks < 1:
            raise ValueError(f"Need at least {block_size} tokens in {path}")

    def __len__(self) -> int:
        return self.blocks

    def __getitem__(self, index: int) -> torch.Tensor:
        # Spread a capped run across the entire split, not its first N tokens.
        physical_block = index * self.available_blocks // self.blocks
        start = physical_block * self.block_size
        # Copy away from read-only mmap so PyTorch workers can safely use it.
        return torch.from_numpy(np.array(self.data[start:start + self.block_size],
                                         dtype=np.int64, copy=True))


class FixedPermutationBatches(Sampler[list[int]]):
    """Stable order that can restart at a saved microbatch index."""

    def __init__(self, block_count: int, batch_size: int, seed: int, start_batch: int):
        self.indices = np.random.default_rng(seed).permutation(block_count)
        self.batch_size = batch_size
        self.start_batch = start_batch

    def __iter__(self):
        for start in range(self.start_batch * self.batch_size,
                           len(self.indices), self.batch_size):
            yield self.indices[start:start + self.batch_size].tolist()

    def __len__(self) -> int:
        return max(0, math.ceil(len(self.indices) / self.batch_size) - self.start_batch)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def save_checkpoint(path: Path, raw_model, optimizer, scheduler, run_config,
                    step: int, next_microbatch: int, best_loss: float,
                    elapsed_seconds: float, run_started_utc: str) -> None:
    state = {
        "model": raw_model.state_dict(), "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(), "run_config": run_config,
        "step": step, "next_microbatch": next_microbatch,
        "best_periodic_validation_loss": best_loss,
        "elapsed_seconds": elapsed_seconds,
        "run_started_utc": run_started_utc,
        "torch_rng": torch.get_rng_state(),
        "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        "numpy_rng": np.random.get_state(), "python_rng": random.getstate(),
    }
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(state, temporary)
    os.replace(temporary, path)


def restore_rng(state: dict) -> None:
    torch.set_rng_state(state["torch_rng"])
    if state["cuda_rng"] is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda_rng"])
    np.random.set_state(state["numpy_rng"])
    random.setstate(state["python_rng"])


def evaluate(model, dataset: TokenBlocks, batch_size: int, device: torch.device) -> float:
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                        num_workers=0, pin_memory=device.type == "cuda")
    model.eval()
    weighted_loss = 0.0
    targets = 0
    autocast = (torch.autocast(device_type="cuda", dtype=torch.bfloat16)
                if device.type == "cuda" else nullcontext())
    with torch.inference_mode(), autocast:
        for batch in loader:
            batch = batch.to(device, non_blocking=True)
            result = model(input_ids=batch, labels=batch, use_cache=False)
            count = batch.shape[0] * (batch.shape[1] - 1)
            weighted_loss += float(result.loss) * count
            targets += count
    model.train()
    return weighted_loss / targets


def log_jsonl(path: Path, record: dict) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, separators=(",", ":")) + "\n")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def main() -> None:
    session_start = time.monotonic()
    session_started_utc = utc_now()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tokens-dir", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--train-tokens", type=int, default=200_000_000)
    parser.add_argument("--val-tokens", type=int, default=5_000_000)
    parser.add_argument("--periodic-val-tokens", type=int, default=1_000_000)
    parser.add_argument("--block-size", type=int, default=2048)
    parser.add_argument("--n-embd", type=int, default=576)
    parser.add_argument("--n-layer", type=int, default=8)
    parser.add_argument("--n-head", type=int, default=8)
    parser.add_argument("--micro-batch-size", type=int, default=4)
    parser.add_argument("--grad-accum", type=int, default=2)
    parser.add_argument("--eval-batch-size", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--warmup-steps", type=int, default=500)
    parser.add_argument("--eval-every", type=int, default=1000)
    parser.add_argument("--save-every", type=int, default=1000)
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--max-steps", type=int, default=None,
                        help="Stop early for benchmarking; scheduler remains sized for full planned run")
    parser.add_argument("--data-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--allow-cpu", action="store_true")
    parser.add_argument("--debug-allow-under50m", action="store_true")
    args = parser.parse_args()
    if any(value < 1 for value in (args.train_tokens, args.val_tokens,
                                  args.periodic_val_tokens, args.block_size,
                                  args.n_embd, args.n_layer, args.n_head,
                                  args.micro_batch_size, args.grad_accum,
                                  args.eval_batch_size, args.eval_every,
                                  args.save_every, args.log_every)):
        parser.error("Count and size arguments must be positive")
    if args.n_embd % args.n_head:
        parser.error("n-embd must be divisible by n-head")
    if args.data_workers < 0 or args.warmup_steps < 0:
        parser.error("data-workers and warmup-steps cannot be negative")
    if args.periodic_val_tokens > args.val_tokens:
        parser.error("periodic-val-tokens cannot exceed val-tokens")
    if args.max_steps is not None and args.max_steps < 1:
        parser.error("max-steps must be positive")
    if not torch.cuda.is_available() and not args.allow_cpu:
        parser.error("CUDA is unavailable. Use --allow-cpu only for a tiny debug run")
    if args.output_dir.exists() and any(args.output_dir.iterdir()) and not args.resume:
        parser.error(f"Output directory is not empty: {args.output_dir}")

    token_manifest_path = args.tokens_dir / "token_manifest.json"
    if not token_manifest_path.is_file() or not args.tokenizer.is_file():
        parser.error("Missing token manifest or tokenizer")
    token_manifest = json.loads(token_manifest_path.read_text(encoding="utf-8"))
    tokenizer_hash = sha256(args.tokenizer)
    if tokenizer_hash != token_manifest["tokenizer_sha256"]:
        parser.error("Tokenizer differs from the one used to encode the corpus")
    dtype = token_manifest["dtype"]
    train_set = TokenBlocks(args.tokens_dir / "train.bin", dtype,
                            args.block_size, args.train_tokens)
    full_val_set = TokenBlocks(args.tokens_dir / "validation.bin", dtype,
                               args.block_size, args.val_tokens)
    periodic_val_set = TokenBlocks(args.tokens_dir / "validation.bin", dtype,
                                   args.block_size, args.periodic_val_tokens)
    microbatches = math.ceil(len(train_set) / args.micro_batch_size)
    total_steps = math.ceil(microbatches / args.grad_accum)
    if total_steps < 1:
        parser.error("Not enough training blocks")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
        torch.set_float32_matmul_precision("high")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = GPT2Config(
        vocab_size=token_manifest["vocab_size"],
        n_positions=args.block_size, n_ctx=args.block_size,
        n_embd=args.n_embd, n_layer=args.n_layer, n_head=args.n_head,
        bos_token_id=token_manifest["eos_id"],
        eos_token_id=token_manifest["eos_id"],
        pad_token_id=token_manifest["pad_id"], use_cache=False,
    )
    raw_model = GPT2LMHeadModel(config)
    parameters = sum(p.numel() for p in raw_model.parameters())
    print(f"Freshly initialized model: {parameters:,} parameters; "
          f"attention={raw_model.config._attn_implementation}", flush=True)
    if parameters > 300_000_000 or (parameters < 50_000_000 and
                                    not args.debug_allow_under50m):
        parser.error(f"Model has {parameters:,} parameters; valid range is 50M–300M")
    raw_model.to(device)

    decay = [p for p in raw_model.parameters() if p.requires_grad and p.ndim >= 2]
    no_decay = [p for p in raw_model.parameters() if p.requires_grad and p.ndim < 2]
    optimizer = torch.optim.AdamW(
        [{"params": decay, "weight_decay": args.weight_decay},
         {"params": no_decay, "weight_decay": 0.0}],
        lr=args.learning_rate, fused=device.type == "cuda")

    def lr_factor(step: int) -> float:
        if step < args.warmup_steps:
            return (step + 1) / max(1, args.warmup_steps)
        progress = (step - args.warmup_steps) / max(1, total_steps - args.warmup_steps)
        return 0.5 * (1 + math.cos(math.pi * min(1.0, progress)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_factor)
    run_config = {
        "tokenizer_sha256": tokenizer_hash,
        "token_manifest_sha256": sha256(token_manifest_path),
        "vocab_size": token_manifest["vocab_size"],
        "eos_id": token_manifest["eos_id"],
        "pad_id": token_manifest["pad_id"],
        "block_size": args.block_size, "n_embd": args.n_embd,
        "n_layer": args.n_layer, "n_head": args.n_head,
        "parameters": parameters, "train_tokens_requested": args.train_tokens,
        "train_tokens_used": len(train_set) * args.block_size,
        "val_tokens_requested": args.val_tokens,
        "val_tokens_used": len(full_val_set) * args.block_size,
        "periodic_val_tokens_used": len(periodic_val_set) * args.block_size,
        "micro_batch_size": args.micro_batch_size,
        "grad_accum": args.grad_accum,
        "effective_batch_tokens": args.micro_batch_size * args.grad_accum * args.block_size,
        "learning_rate": args.learning_rate, "weight_decay": args.weight_decay,
        "warmup_steps": args.warmup_steps, "total_steps": total_steps,
        "eval_every": args.eval_every, "save_every": args.save_every,
        "log_every": args.log_every,
        "seed": args.seed, "torch_version": torch.__version__,
        "block_sampling": "uniform_across_full_split_v1",
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    config_path = args.output_dir / "run_config.json"
    if args.resume:
        state = torch.load(args.resume, map_location="cpu", weights_only=False)
        if state["run_config"] != run_config:
            parser.error("Checkpoint run configuration differs from current arguments/data")
        raw_model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        step = state["step"]
        next_microbatch = state["next_microbatch"]
        best_loss = state["best_periodic_validation_loss"]
        run_started_utc = state.get("run_started_utc", session_started_utc)
        prior_elapsed = float(state.get("elapsed_seconds", 0.0))
        summary_path = args.output_dir / "run_summary.json"
        if summary_path.is_file():
            previous_summary = json.loads(summary_path.read_text(encoding="utf-8"))
            if previous_summary.get("step") == step:
                prior_elapsed = max(prior_elapsed,
                                    float(previous_summary.get("elapsed_seconds", 0.0)))
        print(f"Resuming from optimizer step {step}, microbatch {next_microbatch}", flush=True)
    else:
        step = 0
        next_microbatch = 0
        best_loss = float("inf")
        run_started_utc = session_started_utc
        prior_elapsed = 0.0
        config_path.write_text(json.dumps(run_config, indent=2) + "\n", encoding="utf-8")
    session_start_step = step

    def elapsed_seconds() -> float:
        return prior_elapsed + time.monotonic() - session_start

    def timed_record(record: dict) -> dict:
        return {"timestamp_utc": utc_now(), "elapsed_seconds": round(elapsed_seconds(), 3),
                **record}
    if args.compile:
        model = torch.compile(raw_model)
    else:
        model = raw_model
    model.train()
    if args.resume:
        restore_rng(state)
    print(f"Train blocks={len(train_set):,}; validation blocks={len(full_val_set):,}; "
          f"planned optimizer steps={total_steps:,}; device={device}", flush=True)

    sampler = FixedPermutationBatches(len(train_set), args.micro_batch_size,
                                      args.seed, next_microbatch)
    loader_kwargs = {"num_workers": args.data_workers,
                     "pin_memory": device.type == "cuda",
                     "generator": torch.Generator().manual_seed(args.seed)}
    if args.data_workers:
        loader_kwargs.update(persistent_workers=True, prefetch_factor=2)
    loader = DataLoader(train_set, batch_sampler=sampler, **loader_kwargs)
    autocast = (torch.autocast(device_type="cuda", dtype=torch.bfloat16)
                if device.type == "cuda" else nullcontext())
    optimizer.zero_grad(set_to_none=True)
    interval_start = time.monotonic()
    interval_tokens = 0
    running_loss = 0.0
    running_batches = 0
    total_microbatches = microbatches
    next_position = next_microbatch

    for offset, batch in enumerate(loader):
        microbatch = next_microbatch + offset
        group_start = (microbatch // args.grad_accum) * args.grad_accum
        group_size = min(args.grad_accum, total_microbatches - group_start)
        batch = batch.to(device, non_blocking=True)
        with autocast:
            result = model(input_ids=batch, labels=batch, use_cache=False)
            loss = result.loss
        if not torch.isfinite(loss):
            raise RuntimeError(f"Non-finite training loss at microbatch {microbatch}")
        (loss / group_size).backward()
        running_loss += float(loss.detach())
        running_batches += 1
        interval_tokens += batch.numel()
        at_group_end = ((microbatch + 1) % args.grad_accum == 0 or
                        microbatch + 1 == total_microbatches)
        if not at_group_end:
            continue

        torch.nn.utils.clip_grad_norm_(raw_model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad(set_to_none=True)
        step += 1
        next_position = microbatch + 1
        if step % args.log_every == 0 or step == 1:
            if device.type == "cuda":
                torch.cuda.synchronize()
            seconds = max(time.monotonic() - interval_start, 1e-9)
            session_steps = step - session_start_step
            session_seconds = time.monotonic() - session_start
            record = timed_record({"step": step,
                      "train_tokens_seen": min(next_position * args.micro_batch_size,
                                               len(train_set)) * args.block_size,
                      "train_loss": running_loss / running_batches,
                      "lr": optimizer.param_groups[0]["lr"],
                      "interval_seconds": round(seconds, 3),
                      "train_tokens_per_second": interval_tokens / seconds,
                      "estimated_remaining_seconds": round(
                          (total_steps - step) * session_seconds / session_steps, 1),
                      "gpu_peak_gib": (torch.cuda.max_memory_allocated() / 1024 ** 3
                                       if device.type == "cuda" else None)})
            log_jsonl(args.output_dir / "metrics.jsonl", record)
            print(json.dumps(record), flush=True)
            interval_start = time.monotonic()
            interval_tokens = 0
            running_loss = 0.0
            running_batches = 0
        if step % args.eval_every == 0:
            eval_start = time.monotonic()
            valid_loss = evaluate(model, periodic_val_set, args.eval_batch_size, device)
            record = timed_record({"step": step, "periodic_val_loss": valid_loss,
                      "periodic_val_perplexity": math.exp(min(valid_loss, 30)),
                      "eval_seconds": round(time.monotonic() - eval_start, 3)})
            log_jsonl(args.output_dir / "metrics.jsonl", record)
            print(json.dumps(record), flush=True)
            if valid_loss < best_loss:
                best_loss = valid_loss
                save_checkpoint(args.output_dir / "checkpoint-best.pt", raw_model,
                                optimizer, scheduler, run_config, step,
                                next_position, best_loss, elapsed_seconds(), run_started_utc)
            interval_start = time.monotonic()
            interval_tokens = 0
        if step % args.save_every == 0:
            save_checkpoint(args.output_dir / "checkpoint-latest.pt", raw_model,
                            optimizer, scheduler, run_config, step,
                            next_position, best_loss, elapsed_seconds(), run_started_utc)
        if args.max_steps is not None and step >= args.max_steps:
            print(f"Stopped at requested benchmark step {step}", flush=True)
            break

    save_checkpoint(args.output_dir / "checkpoint-latest.pt", raw_model,
                    optimizer, scheduler, run_config, step, next_position,
                    best_loss, elapsed_seconds(), run_started_utc)
    final_loss = None
    if step == total_steps:
        eval_start = time.monotonic()
        final_loss = evaluate(model, full_val_set, args.eval_batch_size, device)
        record = timed_record({"step": step, "full_val_loss": final_loss,
                  "full_val_perplexity": math.exp(min(final_loss, 30)),
                  "eval_seconds": round(time.monotonic() - eval_start, 3)})
        log_jsonl(args.output_dir / "metrics.jsonl", record)
        print(json.dumps(record), flush=True)
        save_checkpoint(args.output_dir / "checkpoint-final.pt", raw_model,
                        optimizer, scheduler, run_config, step,
                        next_position, best_loss, elapsed_seconds(), run_started_utc)
    summary = timed_record({
        "status": "complete" if step == total_steps else "benchmark_stopped",
        "run_started_utc": run_started_utc,
        "session_started_utc": session_started_utc,
        "session_seconds": round(time.monotonic() - session_start, 3),
        "step": step, "total_steps": total_steps,
        "train_tokens_seen": min(next_position * args.micro_batch_size,
                                 len(train_set)) * args.block_size,
        "train_tokens_planned": len(train_set) * args.block_size,
        "full_val_loss": final_loss,
    })
    (args.output_dir / "run_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    log_jsonl(args.output_dir / "metrics.jsonl", {"event": "session_end", **summary})
    print(json.dumps({"event": "session_end", **summary}), flush=True)
    print(f"Latest checkpoint: {args.output_dir / 'checkpoint-latest.pt'}", flush=True)


if __name__ == "__main__":
    main()
