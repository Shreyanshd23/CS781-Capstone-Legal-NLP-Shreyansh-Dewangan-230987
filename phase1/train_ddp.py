"""Four-GPU DDP trainer for the CS-781 Phase-1 A1 decoder baseline.

The long run must be preceded by gpu_preflight.py and a four-GPU smoke/resume
test. This script never picks physical GPUs automatically.
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
import shutil
import signal
import socket
import subprocess
import sys
import tempfile
import time

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset, Sampler, Subset
from tokenizers import Tokenizer
from tqdm import tqdm
from transformers import GPT2Config, GPT2LMHeadModel


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class TokenBlocks(Dataset):
    """Evenly sample nonoverlapping context blocks across a packed split."""

    def __init__(self, path: Path, dtype: str, block_size: int, count: int):
        self.path = path
        self.dtype = dtype
        self.data = np.memmap(path, dtype=np.dtype(dtype), mode="r")
        self.block_size = block_size
        self.available = len(self.data) // block_size
        self.count = count
        if not 0 < count <= self.available:
            raise ValueError(f"Requested {count} blocks from {self.available} in {path}")

    def __len__(self) -> int:
        return self.count

    def __getitem__(self, index: int) -> torch.Tensor:
        physical = index * self.available // self.count
        start = physical * self.block_size
        return torch.from_numpy(np.array(self.data[start:start + self.block_size],
                                         dtype=np.int64, copy=True))

    def __getstate__(self) -> dict:
        state = self.__dict__.copy()
        state["data"] = None  # Workers reopen the mmap instead of pickling 1 GiB.
        return state

    def __setstate__(self, state: dict) -> None:
        self.__dict__.update(state)
        self.data = np.memmap(self.path, dtype=np.dtype(self.dtype), mode="r")


class ExactRankBatches(Sampler[list[int]]):
    """One global permutation, disjoint rank slices, no padded duplicates."""

    def __init__(self, block_count: int, rank: int, world_size: int,
                 batch_size: int, seed: int, start_microbatch: int):
        permutation = np.random.default_rng(seed).permutation(block_count)
        self.rank_indices = permutation[rank::world_size]
        self.batch_size = batch_size
        self.start_microbatch = start_microbatch
        if len(self.rank_indices) % batch_size:
            raise ValueError("Training block count must divide the global microbatch")

    def __iter__(self):
        for start in range(self.start_microbatch * self.batch_size,
                           len(self.rank_indices), self.batch_size):
            yield self.rank_indices[start:start + self.batch_size].tolist()

    def __len__(self) -> int:
        return len(self.rank_indices) // self.batch_size - self.start_microbatch


def model_breakdown(model: GPT2LMHeadModel) -> dict[str, int]:
    token = sum(p.numel() for p in model.transformer.wte.parameters())
    position = sum(p.numel() for p in model.transformer.wpe.parameters())
    blocks = sum(p.numel() for block in model.transformer.h
                 for p in block.parameters())
    final_norm = sum(p.numel() for p in model.transformer.ln_f.parameters())
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    tied = model.lm_head.weight.data_ptr() == model.transformer.wte.weight.data_ptr()
    if not tied or total != token + position + blocks + final_norm:
        raise RuntimeError("Unexpected model parameter accounting or untied LM head")
    return {"total": total, "trainable": trainable, "token_embedding": token,
            "position_embedding": position, "transformer_blocks": blocks,
            "final_layer_norm": final_norm, "lm_head_unique": 0,
            "lm_head_shared_with_token_embedding": token}


def check_causal_mask(model: GPT2LMHeadModel, vocab_size: int,
                      device: torch.device) -> float:
    model.eval()
    short = torch.arange(3, 11, device=device).unsqueeze(0) % vocab_size
    long = torch.arange(3, 15, device=device).unsqueeze(0) % vocab_size
    with torch.inference_mode():
        a = model(input_ids=short, use_cache=False).logits
        b = model(input_ids=long, use_cache=False).logits[:, :short.shape[1]]
    error = (a - b).abs().max().item()
    model.train()
    if not torch.allclose(a, b, atol=1e-4, rtol=1e-4):
        raise RuntimeError(f"Causal-mask prefix test failed: max error={error}")
    return error


def rng_state() -> dict:
    return {"python": random.getstate(), "numpy": np.random.get_state(),
            "torch_cpu": torch.get_rng_state(),
            "torch_cuda": torch.cuda.get_rng_state() if torch.cuda.is_initialized() else None}


def restore_rng(state: dict) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"])
    if state["torch_cuda"] is not None and torch.cuda.is_initialized():
        torch.cuda.set_rng_state(state["torch_cuda"])


def atomic_json(path: Path, payload: dict) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def update_link(folder: Path, name: str, target: str) -> None:
    temporary = folder / f".{name}.tmp"
    if temporary.is_symlink():
        temporary.unlink()
    temporary.symlink_to(target, target_is_directory=True)
    os.replace(temporary, folder / name)


def save_checkpoint(folder: Path, model: GPT2LMHeadModel, optimizer,
                    scheduler, run_config: dict, step: int, elapsed_seconds: float,
                    started_utc: str, best_loss: float, rank: int,
                    world_size: int, keep_last_n: int, is_best: bool,
                    validation_loss: float | None) -> Path:
    states = [None] * world_size
    dist.all_gather_object(states, rng_state())
    target = folder / f"step_{step:08d}"
    if rank == 0:
        folder.mkdir(parents=True, exist_ok=True)
        revision = 2
        while target.exists():
            target = folder / f"step_{step:08d}_r{revision}"
            revision += 1
        temporary = Path(tempfile.mkdtemp(prefix=".checkpoint_tmp_", dir=folder))
        try:
            torch.save({"model": model.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "scheduler": scheduler.state_dict(),
                        "run_config": run_config, "step": step,
                        "next_microbatch": step * run_config["grad_accum"],
                        "tokens_processed": step * run_config["tokens_per_update"],
                        "epoch": int(step == run_config["total_steps"]),
                        "rng_states": states,
                        "elapsed_seconds": elapsed_seconds,
                        "run_started_utc": started_utc,
                        "best_validation_loss": best_loss},
                       temporary / "state.pt")
            atomic_json(temporary / "metadata.json",
                        {"step": step,
                         "tokens_processed": step * run_config["tokens_per_update"],
                         "validation_loss": validation_loss,
                         "validation_perplexity": (
                             math.exp(min(validation_loss, 30))
                             if validation_loss is not None else None),
                         "saved_utc": utc_now(),
                         "elapsed_seconds": elapsed_seconds})
            os.replace(temporary, target)
        finally:
            if temporary.exists():
                shutil.rmtree(temporary)
        update_link(folder, "latest", target.name)
        if is_best:
            update_link(folder, "best", target.name)
        protected = set()
        if (folder / "best").is_symlink():
            protected.add((folder / "best").resolve())
        steps = sorted(folder.glob("step_*"))
        for old in steps[:-keep_last_n]:
            if old.resolve() not in protected:
                shutil.rmtree(old)
    dist.barrier()
    return target


def evaluate(model: GPT2LMHeadModel, dataset: TokenBlocks,
             batch_size: int, rank: int, world_size: int,
             device: torch.device) -> float:
    shard = Subset(dataset, range(rank, len(dataset), world_size))
    loader = DataLoader(shard, batch_size=batch_size, shuffle=False,
                        num_workers=0, pin_memory=device.type == "cuda",
                        generator=torch.Generator().manual_seed(0))
    model.eval()
    sums = torch.zeros(2, device=device, dtype=torch.float64)
    context = (torch.autocast("cuda", dtype=torch.bfloat16)
               if device.type == "cuda" else nullcontext())
    with torch.inference_mode(), context:
        for batch in loader:
            batch = batch.to(device, non_blocking=True)
            loss = model(input_ids=batch, labels=batch, use_cache=False).loss
            count = batch.shape[0] * (batch.shape[1] - 1)
            sums[0] += loss.detach().double() * count
            sums[1] += count
    dist.all_reduce(sums, op=dist.ReduceOp.SUM)
    model.train()
    return (sums[0] / sums[1]).item()


def gpu_metrics(device: torch.device, rank: int, world_size: int) -> list[dict]:
    local = {"rank": rank, "allocated_gib": None, "peak_allocated_gib": None,
             "reserved_gib": None}
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        local.update(allocated_gib=torch.cuda.memory_allocated(device) / 1024**3,
                     peak_allocated_gib=torch.cuda.max_memory_allocated(device) / 1024**3,
                     reserved_gib=torch.cuda.memory_reserved(device) / 1024**3)
    results = [None] * world_size
    dist.all_gather_object(results, local)
    return results


def append_metric(path: Path, record: dict) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, separators=(",", ":")) + "\n")


def host_gpu_utilization(physical_ids: list[int]) -> dict:
    try:
        query = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,utilization.gpu,memory.used",
             "--format=csv,noheader,nounits"], capture_output=True, text=True,
            check=False)
    except FileNotFoundError:
        return {"error": "nvidia-smi unavailable; utilization unknown"}
    if query.returncode:
        return {"error": query.stderr.strip() or "nvidia-smi failed"}
    result = {}
    for line in query.stdout.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if (len(parts) == 3 and parts[0].isdigit() and
                int(parts[0]) in physical_ids):
            result[parts[0]] = {"utilization_percent": int(parts[1]),
                                "used_mib": int(parts[2])}
    return result


def run_training(args, rank: int, world_size: int, device: torch.device,
                 selection: dict | None, visible_ids: list[int], manifest: dict,
                 special_ids: dict, tokenizer: Tokenizer, tokenizer_hash: str,
                 token_manifest_path: Path, train_set: TokenBlocks,
                 full_val_set: TokenBlocks, periodic_val_set: TokenBlocks,
                 raw_model: GPT2LMHeadModel, breakdown: dict, total_steps: int,
                 warmup_steps: int, tokens_per_update: int,
                 effective_sequences: int, session_start: float,
                 session_started_utc: str) -> None:
    decay = [p for p in raw_model.parameters() if p.requires_grad and p.ndim >= 2]
    no_decay = [p for p in raw_model.parameters() if p.requires_grad and p.ndim < 2]
    optimizer = torch.optim.AdamW(
        [{"params": decay, "weight_decay": args.weight_decay},
         {"params": no_decay, "weight_decay": 0.0}],
        lr=args.learning_rate, betas=(0.9, 0.95),
        fused=device.type == "cuda")

    def lr_factor(index: int) -> float:
        if index < warmup_steps:
            return (index + 1) / max(warmup_steps, 1)
        progress = (index - warmup_steps) / max(total_steps - warmup_steps, 1)
        return 0.5 * (1 + math.cos(math.pi * min(1.0, progress)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_factor)
    run_config = {
        "architecture": "GPT2-style decoder-only pre-LN causal Transformer",
        "tokenizer_path": str(args.tokenizer.resolve()),
        "tokenizer_sha256": tokenizer_hash,
        "token_manifest_sha256": sha256(token_manifest_path),
        "special_token_ids": special_ids,
        "bos_model_id": (special_ids["<bos>"] if special_ids["<bos>"] is not None
                         else special_ids["<eos>"]),
        "vocab_size": manifest["vocab_size"],
        "block_size": args.block_size, "n_embd": args.n_embd,
        "n_layer": args.n_layer, "n_head": args.n_head,
        "n_inner": 4 * args.n_embd, "dropout": 0.1,
        "layer_norm_epsilon": 1e-5, "activation": "gelu",
        "model_parameters": breakdown,
        "train_tokens_requested": args.train_tokens,
        "train_tokens_planned": len(train_set) * args.block_size,
        "val_tokens_planned": len(full_val_set) * args.block_size,
        "periodic_val_tokens_planned": len(periodic_val_set) * args.block_size,
        "world_size": world_size, "per_gpu_batch": args.per_gpu_batch,
        "distributed_backend": dist.get_backend(),
        "grad_accum": args.grad_accum,
        "global_sequence_batch": effective_sequences,
        "tokens_per_update": tokens_per_update,
        "total_steps": total_steps,
        "learning_rate": args.learning_rate, "weight_decay": args.weight_decay,
        "betas": [0.9, 0.95], "warmup_ratio": args.warmup_ratio,
        "warmup_steps": warmup_steps, "schedule": "cosine",
        "grad_clip_norm": 1.0, "precision": "bf16" if device.type == "cuda" else "fp32-debug",
        "eval_every": args.eval_every, "save_every": args.save_every,
        "seed": args.seed, "block_sampling": "uniform_full_split_no_pad_v1",
        "torch_version": torch.__version__,
    }

    if rank == 0:
        if args.resume is None:
            args.output_dir.mkdir(parents=True, exist_ok=False)
            atomic_json(args.output_dir / "run_config.json", run_config)
            atomic_json(args.output_dir / "tokenizer_metadata.json",
                        {"path": str(args.tokenizer.resolve()),
                         "sha256": tokenizer_hash,
                         "vocab_size": manifest["vocab_size"],
                         "special_token_ids": special_ids,
                         "bos_model_id": run_config["bos_model_id"]})
            if selection is not None:
                atomic_json(args.output_dir / "gpu_selection.json", selection)
        git_result = subprocess.run(["git", "rev-parse", "HEAD"],
                                    capture_output=True, text=True, check=False)
        environment = {
            "session_started_utc": session_started_utc,
            "python": sys.version,
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "nccl": (torch.cuda.nccl.version() if device.type == "cuda" else None),
            "backend": dist.get_backend(),
            "bf16_supported": (torch.cuda.is_bf16_supported()
                               if device.type == "cuda" else False),
            "physical_gpu_ids": visible_ids,
            "local_gpu_names": ([torch.cuda.get_device_name(i)
                                  for i in range(world_size)]
                                 if device.type == "cuda" else []),
            "git_commit": (git_result.stdout.strip()
                           if git_result.returncode == 0 else None),
            "config_file": str((args.output_dir / "run_config.json").resolve()),
        }
        append_metric(args.output_dir / "environment_sessions.jsonl", environment)
    dist.barrier()

    state = None
    if args.resume is not None:
        checkpoint_file = args.resume / "state.pt" if args.resume.is_dir() else args.resume
        state = torch.load(checkpoint_file, map_location="cpu", weights_only=False)
        if state["run_config"] != run_config:
            raise RuntimeError("Checkpoint training configuration or token data changed")
        if len(state["rng_states"]) != world_size:
            raise RuntimeError("Checkpoint RNG rank count differs from world size")
        raw_model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        step = int(state["step"])
        next_microbatch = int(state["next_microbatch"])
        best_loss = float(state["best_validation_loss"])
        run_started_utc = state["run_started_utc"]
        prior_elapsed = float(state["elapsed_seconds"])
        summary_path = args.output_dir / "run_summary.json"
        if summary_path.is_file():
            previous = json.loads(summary_path.read_text(encoding="utf-8"))
            if previous.get("step") == step:
                prior_elapsed = max(prior_elapsed,
                                    float(previous.get("elapsed_seconds", 0.0)))
        if next_microbatch != step * args.grad_accum:
            raise RuntimeError("Checkpoint microbatch position is inconsistent")
    else:
        step = 0
        next_microbatch = 0
        best_loss = float("inf")
        run_started_utc = session_started_utc
        prior_elapsed = 0.0
    initial_step = step

    def elapsed() -> float:
        return prior_elapsed + time.monotonic() - session_start

    def timed_record(record: dict) -> dict:
        return {"timestamp_utc": utc_now(), "elapsed_seconds": round(elapsed(), 3),
                **record}

    sampler = ExactRankBatches(len(train_set), rank, world_size,
                               args.per_gpu_batch, args.seed, next_microbatch)
    first = sampler.rank_indices[
        next_microbatch * args.per_gpu_batch:
        next_microbatch * args.per_gpu_batch + 8].tolist()
    first_by_rank = [None] * world_size
    dist.all_gather_object(first_by_rank, first)
    flattened = [value for row in first_by_rank for value in row]
    if len(flattened) != len(set(flattened)):
        raise RuntimeError("Duplicate training block IDs across DDP ranks")
    if rank == 0:
        print(f"DDP shard sample IDs: {first_by_rank}; no overlap", flush=True)
        append_metric(args.output_dir / "metrics.jsonl",
                      timed_record({"event": "ddp_shard_check",
                                    "sample_ids_by_rank": first_by_rank,
                                    "no_duplicates": True}))
    loader_options = {"num_workers": args.data_workers,
                      "pin_memory": device.type == "cuda",
                      "generator": torch.Generator().manual_seed(args.seed + rank)}
    if args.data_workers:
        loader_options.update(persistent_workers=True, prefetch_factor=2,
                              multiprocessing_context="spawn")
    loader = DataLoader(train_set, batch_sampler=sampler, **loader_options)
    model = DDP(raw_model,
                device_ids=[device.index] if device.type == "cuda" else None,
                broadcast_buffers=False, find_unused_parameters=False,
                gradient_as_bucket_view=True)
    if state is not None:
        restore_rng(state["rng_states"][rank])
    model.train()
    optimizer.zero_grad(set_to_none=True)
    amp = (torch.autocast("cuda", dtype=torch.bfloat16)
           if device.type == "cuda" else nullcontext())
    termination = {"requested": False}

    def handle_signal(_number, _frame):
        termination["requested"] = True

    signal.signal(signal.SIGTERM, handle_signal)
    signal.signal(signal.SIGINT, handle_signal)
    progress_file = None
    if rank == 0 and not args.no_progress:
        try:
            progress_file = open("/dev/tty", "w", encoding="utf-8")
        except OSError:
            progress_file = sys.stdout
    progress = (tqdm(total=total_steps, initial=step, desc="A1 DDP",
                     unit="step", mininterval=10, dynamic_ncols=True,
                     file=progress_file) if rank == 0 and progress_file else None)
    interval_start = time.monotonic()
    interval_step_start = step
    local_loss = torch.zeros(2, dtype=torch.float64, device=device)
    last_grad_norm = None
    stop_reason = None
    last_validation_loss = None
    final_validation_loss = None
    try:
        for offset, batch in enumerate(loader):
            microbatch = next_microbatch + offset
            sync_gradients = (microbatch + 1) % args.grad_accum == 0
            batch = batch.to(device, non_blocking=True)
            context = nullcontext() if sync_gradients else model.no_sync()
            with context:
                with amp:
                    loss = model(input_ids=batch, labels=batch, use_cache=False).loss
                if not torch.isfinite(loss):
                    raise RuntimeError(f"Non-finite loss on rank {rank}, microbatch {microbatch}")
                (loss / args.grad_accum).backward()
            target_count = batch.shape[0] * (batch.shape[1] - 1)
            local_loss[0] += loss.detach().double() * target_count
            local_loss[1] += target_count
            if not sync_gradients:
                continue

            grad_norm = torch.nn.utils.clip_grad_norm_(raw_model.parameters(), 1.0)
            if not torch.isfinite(grad_norm):
                raise RuntimeError(f"Non-finite gradients on rank {rank}, step {step}")
            last_grad_norm = float(grad_norm)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            step += 1
            if progress is not None:
                progress.update(1)

            if step == 1 or step % args.log_every == 0:
                dist.all_reduce(local_loss, op=dist.ReduceOp.SUM)
                per_gpu = gpu_metrics(device, rank, world_size)
                if rank == 0:
                    seconds = max(time.monotonic() - interval_start, 1e-9)
                    steps_in_interval = step - interval_step_start
                    session_steps = step - initial_step
                    session_seconds = max(time.monotonic() - session_start, 1e-9)
                    record = timed_record({
                        "event": "train", "global_step": step,
                        "tokens_processed": step * tokens_per_update,
                        "train_loss_global": (local_loss[0] / local_loss[1]).item(),
                        "learning_rate": optimizer.param_groups[0]["lr"],
                        "gradient_norm": last_grad_norm,
                        "interval_seconds": round(seconds, 3),
                        "mean_step_seconds": seconds / steps_in_interval,
                        "tokens_per_second_global": (
                            steps_in_interval * tokens_per_update / seconds),
                        "estimated_remaining_seconds": (
                            (total_steps - step) * session_seconds /
                            max(session_steps, 1)),
                        "per_gpu_memory": per_gpu,
                        "physical_gpu_utilization": (
                            host_gpu_utilization(visible_ids)
                            if device.type == "cuda" else None),
                    })
                    append_metric(args.output_dir / "metrics.jsonl", record)
                    print(json.dumps(record), flush=True)
                    if progress is not None:
                        progress.set_postfix(loss=f"{record['train_loss_global']:.3f}",
                                             tok_s=f"{record['tokens_per_second_global']:.0f}")
                local_loss.zero_()
                interval_start = time.monotonic()
                interval_step_start = step

            improved = False
            if step % args.eval_every == 0:
                eval_start = time.monotonic()
                last_validation_loss = evaluate(raw_model, periodic_val_set,
                                                args.eval_batch, rank, world_size,
                                                device)
                improved = last_validation_loss < best_loss
                if improved:
                    best_loss = last_validation_loss
                if rank == 0:
                    record = timed_record({
                        "event": "periodic_validation", "global_step": step,
                        "tokens_processed": step * tokens_per_update,
                        "validation_loss_global": last_validation_loss,
                        "perplexity": math.exp(min(last_validation_loss, 30)),
                        "validation_seconds": time.monotonic() - eval_start,
                        "best_so_far": improved})
                    append_metric(args.output_dir / "metrics.jsonl", record)
                    print(json.dumps(record), flush=True)
            if improved:
                save_checkpoint(args.output_dir / "checkpoints", raw_model,
                                optimizer, scheduler, run_config, step, elapsed(),
                                run_started_utc, best_loss, rank, world_size,
                                args.keep_last_n, True, last_validation_loss)
            if step % args.save_every == 0 and not improved:
                save_checkpoint(args.output_dir / "checkpoints", raw_model,
                                optimizer, scheduler, run_config, step, elapsed(),
                                run_started_utc, best_loss, rank, world_size,
                                args.keep_last_n, False, last_validation_loss)

            stop_flag = torch.tensor(int(termination["requested"]), device=device)
            dist.all_reduce(stop_flag, op=dist.ReduceOp.MAX)
            if stop_flag.item():
                stop_reason = "signal"
                break
            if args.max_steps is not None and step >= args.max_steps:
                stop_reason = "smoke_stopped" if args.smoke else "max_steps_stopped"
                break

        # A smoke run validates the held-out reduction even before eval_every.
        if args.smoke and step < total_steps:
            eval_start = time.monotonic()
            last_validation_loss = evaluate(raw_model, periodic_val_set,
                                            args.eval_batch, rank, world_size,
                                            device)
            improved = last_validation_loss < best_loss
            if improved:
                best_loss = last_validation_loss
            if rank == 0:
                record = timed_record({"event": "smoke_validation",
                    "global_step": step, "validation_loss_global": last_validation_loss,
                    "perplexity": math.exp(min(last_validation_loss, 30)),
                    "validation_seconds": time.monotonic() - eval_start})
                append_metric(args.output_dir / "metrics.jsonl", record)
                print(json.dumps(record), flush=True)
            if improved:
                save_checkpoint(args.output_dir / "checkpoints", raw_model,
                                optimizer, scheduler, run_config, step, elapsed(),
                                run_started_utc, best_loss, rank, world_size,
                                args.keep_last_n, True, last_validation_loss)

        if step == total_steps:
            if step % args.eval_every != 0:
                periodic_start = time.monotonic()
                last_validation_loss = evaluate(raw_model, periodic_val_set,
                                                args.eval_batch, rank, world_size,
                                                device)
                improved = last_validation_loss < best_loss
                if improved:
                    best_loss = last_validation_loss
                if rank == 0:
                    record = timed_record({"event": "final_periodic_validation",
                        "global_step": step,
                        "validation_loss_global": last_validation_loss,
                        "perplexity": math.exp(min(last_validation_loss, 30)),
                        "validation_seconds": time.monotonic() - periodic_start,
                        "best_so_far": improved})
                    append_metric(args.output_dir / "metrics.jsonl", record)
                    print(json.dumps(record), flush=True)
                if improved:
                    save_checkpoint(args.output_dir / "checkpoints", raw_model,
                                    optimizer, scheduler, run_config, step, elapsed(),
                                    run_started_utc, best_loss, rank, world_size,
                                    args.keep_last_n, True, last_validation_loss)
            eval_start = time.monotonic()
            final_validation_loss = evaluate(raw_model, full_val_set,
                                             args.eval_batch, rank, world_size,
                                             device)
            if rank == 0:
                record = timed_record({"event": "final_validation",
                    "global_step": step, "validation_loss_global": final_validation_loss,
                    "perplexity": math.exp(min(final_validation_loss, 30)),
                    "validation_seconds": time.monotonic() - eval_start})
                append_metric(args.output_dir / "metrics.jsonl", record)
                print(json.dumps(record), flush=True)

        save_checkpoint(args.output_dir / "checkpoints", raw_model,
                        optimizer, scheduler, run_config, step, elapsed(),
                        run_started_utc, best_loss, rank, world_size,
                        args.keep_last_n, False,
                        final_validation_loss or last_validation_loss)
        final_memory = gpu_metrics(device, rank, world_size)
        generation = None
        if args.smoke and rank == 0:
            raw_model.eval()
            prompt = torch.tensor(np.asarray(full_val_set.data[:8],
                                             dtype=np.int64), device=device).unsqueeze(0)
            with torch.inference_mode():
                generated = raw_model.generate(prompt, max_new_tokens=16,
                                               do_sample=False,
                                               pad_token_id=special_ids["<pad>"],
                                               eos_token_id=special_ids["<eos>"])
            generation = tokenizer.decode(generated[0].tolist())[:400]
            raw_model.train()
            print(f"Smoke generation preview: {generation!r}", flush=True)
        dist.barrier()
        if rank == 0:
            summary = timed_record({
                "status": ("complete" if step == total_steps else
                           stop_reason or "stopped"),
                "run_started_utc": run_started_utc,
                "session_started_utc": session_started_utc,
                "session_seconds": round(time.monotonic() - session_start, 3),
                "step": step, "total_steps": total_steps,
                "tokens_processed": step * tokens_per_update,
                "train_tokens_planned": len(train_set) * args.block_size,
                "best_validation_loss": best_loss,
                "final_validation_loss": final_validation_loss,
                "per_gpu_memory": final_memory,
                "smoke_generation": generation})
            atomic_json(args.output_dir / "run_summary.json", summary)
            append_metric(args.output_dir / "metrics.jsonl",
                          {"event": "session_end", **summary})
            print(json.dumps({"event": "session_end", **summary}), flush=True)
            if args.smoke:
                metric_lines = (args.output_dir / "metrics.jsonl").read_text(
                    encoding="utf-8").splitlines()
                last_train = next((json.loads(line) for line in reversed(metric_lines)
                                   if '"event":"train"' in line), None)
                report = {
                    "gpu_selection": selection,
                    "software": json.loads((args.output_dir / "environment_sessions.jsonl")
                                           .read_text().splitlines()[-1]),
                    "model": breakdown,
                    "distributed": {"world_size": world_size,
                                    "backend": dist.get_backend(),
                                    "sample_ids_by_rank": first_by_rank,
                                    "duplicate_sample_ids": False,
                                    "global_sequence_batch": effective_sequences,
                                    "tokens_per_update": tokens_per_update},
                    "performance": {"per_gpu_batch": args.per_gpu_batch,
                                    "per_gpu_memory": final_memory,
                                    "tokens_per_second_global": (
                                        last_train.get("tokens_per_second_global")
                                        if last_train else None),
                                    "mean_step_seconds": (
                                        last_train.get("mean_step_seconds")
                                        if last_train else None),
                                    "physical_gpu_utilization": (
                                        last_train.get("physical_gpu_utilization")
                                        if last_train else None)},
                    "numerical_health": {"finite_loss_and_gradients": True,
                                         "last_gradient_norm": last_grad_norm,
                                         "last_train_loss_global": (
                                             last_train.get("train_loss_global")
                                             if last_train else None)},
                    "checkpoint": {"path": str((args.output_dir / "checkpoints" /
                                                  "latest" / "state.pt").resolve()),
                                   "saved": True, "resume_loaded": state is not None,
                                   "optimizer_restored": state is not None,
                                   "scheduler_restored": state is not None,
                                   "rng_restored": state is not None,
                                   "resumed_from_step": initial_step,
                                   "current_step": step},
                    "validation": {"loss": last_validation_loss,
                                   "perplexity": (math.exp(min(last_validation_loss, 30))
                                                  if last_validation_loss is not None else None),
                                   "distributed_all_reduce": True},
                    "generation_preview": generation,
                    "risks": ([{"severity": "CRITICAL",
                                "detail": "CPU Gloo test is not a four-GPU NCCL preflight"}]
                              if device.type == "cpu" else
                              ([{"severity": "IMPORTANT",
                                 "detail": "Four-GPU checkpoint resume has not been tested yet"}]
                               if state is None else []) +
                              ([{"severity": "CRITICAL",
                                 "detail": "NCCL is unavailable; CUDA Gloo smoke is diagnostic only"}]
                               if dist.get_backend() != "nccl" else []) +
                              ([{"severity": "IMPORTANT",
                                 "detail": "Selected GPUs had other active jobs; throughput and contention may vary"}]
                               if selection.get("shared_gpu_override") else []) +
                              [{"severity": "MINOR",
                                "detail": "Tokenizer has no BOS; model uses EOS ID as BOS"}])}
                atomic_json(args.output_dir / "preflight_report.json", report)
    finally:
        if progress is not None:
            progress.close()
        if progress_file is not None and progress_file is not sys.stdout:
            progress_file.close()


def main() -> None:
    session_start = time.monotonic()
    session_started_utc = utc_now()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tokens-dir", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--gpu-selection", type=Path)
    parser.add_argument("--resume", type=Path,
                        help="Checkpoint directory (latest/best/step_N) or state.pt")
    parser.add_argument("--train-tokens", type=int, default=200_000_000)
    parser.add_argument("--val-tokens", type=int, default=5_000_000)
    parser.add_argument("--periodic-val-tokens", type=int, default=1_000_000)
    parser.add_argument("--block-size", type=int, default=2048)
    parser.add_argument("--n-embd", type=int, default=768)
    parser.add_argument("--n-layer", type=int, default=12)
    parser.add_argument("--n-head", type=int, default=12)
    parser.add_argument("--per-gpu-batch", type=int, default=2)
    parser.add_argument("--grad-accum", type=int, default=1)
    parser.add_argument("--eval-batch", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--warmup-ratio", type=float, default=0.02)
    parser.add_argument("--eval-every", type=int, default=1000)
    parser.add_argument("--save-every", type=int, default=1000)
    parser.add_argument("--log-every", type=int, default=25)
    parser.add_argument("--keep-last-n", type=int, default=3)
    parser.add_argument("--data-workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--smoke", action="store_true",
                        help="Stop at 100 steps unless --max-steps is specified")
    parser.add_argument("--cpu-ddp-smoke", action="store_true",
                        help="Tiny two-rank Gloo code test; never a qualifying run")
    parser.add_argument("--cuda-backend", choices=("nccl", "gloo"), default="nccl",
                        help="Gloo on CUDA is allowed only for a diagnostic smoke test")
    parser.add_argument("--debug-allow-under50m", action="store_true")
    parser.add_argument("--no-progress", action="store_true")
    args = parser.parse_args()

    positive = (args.train_tokens, args.val_tokens, args.periodic_val_tokens,
                args.block_size, args.n_embd, args.n_layer, args.n_head,
                args.per_gpu_batch, args.grad_accum, args.eval_batch,
                args.eval_every, args.save_every, args.log_every, args.keep_last_n)
    if any(value < 1 for value in positive) or args.data_workers < 0:
        parser.error("Counts and sizes must be positive; data-workers can be zero")
    if args.n_embd % args.n_head or not 0 <= args.warmup_ratio < 1:
        parser.error("n-embd must divide n-head; warmup-ratio must be in [0,1)")
    if args.periodic_val_tokens > args.val_tokens:
        parser.error("periodic-val-tokens cannot exceed val-tokens")
    if args.smoke and args.max_steps is None:
        args.max_steps = 100
    if args.cuda_backend == "gloo" and not args.smoke:
        parser.error("CUDA Gloo is diagnostic only; production requires NCCL")
    if args.smoke and (args.max_steps < 1 or args.max_steps > 200):
        parser.error("Smoke test must run 1–200 optimizer steps")
    if args.max_steps is not None and args.max_steps < 1:
        parser.error("max-steps must be positive")
    if "WORLD_SIZE" not in os.environ or "LOCAL_RANK" not in os.environ:
        parser.error("Launch with torchrun; one process per GPU")
    world_size = int(os.environ["WORLD_SIZE"])
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    if args.cpu_ddp_smoke:
        if world_size != 2 or not args.debug_allow_under50m or not args.smoke:
            parser.error("CPU code test requires 2 ranks, --smoke, and debug size flag")
        device = torch.device("cpu")
        selection = None
        visible_ids = []
        backend = "gloo"
    else:
        if world_size != 4 or not torch.cuda.is_available():
            parser.error("Production/smoke CUDA DDP requires exactly four visible GPUs")
        if args.gpu_selection is None or not args.gpu_selection.is_file():
            parser.error("A fresh gpu_preflight.py selection report is required")
        selection = json.loads(args.gpu_selection.read_text(encoding="utf-8"))
        if selection.get("host", {}).get("hostname") != socket.gethostname():
            parser.error("GPU selection report was made on a different host")
        visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
        try:
            visible_ids = [int(part) for part in visible.split(",")]
        except ValueError:
            parser.error("CUDA_VISIBLE_DEVICES must list four physical integer GPU IDs")
        if (len(visible_ids) != 4 or len(set(visible_ids)) != 4 or
                visible_ids != selection.get("selected_physical_ids")):
            parser.error("CUDA_VISIBLE_DEVICES differs from the GPU selection report")
        selected_at = datetime.fromisoformat(selection["timestamp_utc"])
        age_seconds = (datetime.now(timezone.utc) - selected_at).total_seconds()
        if age_seconds < 0 or age_seconds > 10 * 60:
            parser.error("GPU selection report is older than 10 minutes; inspect GPUs again")
        if torch.cuda.device_count() != 4:
            parser.error("Expected exactly four CUDA devices after CUDA_VISIBLE_DEVICES")
        torch.cuda.set_device(local_rank)
        if not torch.cuda.is_bf16_supported():
            parser.error("Selected GPU does not support BF16")
        device = torch.device("cuda", local_rank)
        backend = args.cuda_backend
        torch.set_float32_matmul_precision("high")
    if args.output_dir.exists() and not args.resume:
        parser.error(f"Run directory already exists: {args.output_dir}")
    if args.resume and not args.output_dir.is_dir():
        parser.error("Run directory does not exist for resume")

    token_manifest_path = args.tokens_dir / "token_manifest.json"
    if not token_manifest_path.is_file() or not args.tokenizer.is_file():
        parser.error("Missing packed token manifest or tokenizer")
    manifest = json.loads(token_manifest_path.read_text(encoding="utf-8"))
    tokenizer_hash = sha256(args.tokenizer)
    if tokenizer_hash != manifest["tokenizer_sha256"]:
        parser.error("Tokenizer hash does not match packed training data")
    tokenizer = Tokenizer.from_file(str(args.tokenizer))
    if tokenizer.get_vocab_size() != manifest["vocab_size"]:
        parser.error("Tokenizer vocabulary size differs from manifest")
    special_ids = {name: tokenizer.token_to_id(name)
                   for name in ("<unk>", "<bos>", "<eos>", "<pad>")}
    if (special_ids["<unk>"] is None or special_ids["<eos>"] != manifest["eos_id"]
            or special_ids["<pad>"] != manifest["pad_id"]):
        parser.error("Tokenizer special IDs differ from packed-data manifest")
    bos_id = (special_ids["<bos>"] if special_ids["<bos>"] is not None
              else special_ids["<eos>"])
    dtype = manifest["dtype"]
    available_train = len(np.memmap(args.tokens_dir / "train.bin", dtype=dtype,
                                    mode="r")) // args.block_size
    available_val = len(np.memmap(args.tokens_dir / "validation.bin", dtype=dtype,
                                  mode="r")) // args.block_size
    effective_sequences = args.per_gpu_batch * world_size * args.grad_accum
    requested_blocks = min(args.train_tokens // args.block_size, available_train)
    train_blocks = (requested_blocks // effective_sequences) * effective_sequences
    val_blocks = min(args.val_tokens // args.block_size, available_val)
    periodic_blocks = min(args.periodic_val_tokens // args.block_size, available_val)
    if min(train_blocks, val_blocks, periodic_blocks) < 1:
        parser.error("Insufficient token blocks for requested run")
    train_set = TokenBlocks(args.tokens_dir / "train.bin", dtype,
                            args.block_size, train_blocks)
    full_val_set = TokenBlocks(args.tokens_dir / "validation.bin", dtype,
                               args.block_size, val_blocks)
    periodic_val_set = TokenBlocks(args.tokens_dir / "validation.bin", dtype,
                                   args.block_size, periodic_blocks)
    total_steps = train_blocks // effective_sequences
    warmup_steps = math.ceil(total_steps * args.warmup_ratio)
    tokens_per_update = effective_sequences * args.block_size

    random.seed(args.seed + rank)
    np.random.seed(args.seed + rank)
    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed(args.seed + rank)
    config = GPT2Config(
        vocab_size=manifest["vocab_size"], n_positions=args.block_size,
        n_ctx=args.block_size, n_embd=args.n_embd, n_layer=args.n_layer,
        n_head=args.n_head, n_inner=4 * args.n_embd,
        activation_function="gelu", layer_norm_epsilon=1e-5,
        resid_pdrop=0.1, embd_pdrop=0.1, attn_pdrop=0.1,
        bos_token_id=bos_id, eos_token_id=special_ids["<eos>"],
        pad_token_id=special_ids["<pad>"], use_cache=False)
    raw_model = GPT2LMHeadModel(config)
    breakdown = model_breakdown(raw_model)
    if not args.debug_allow_under50m and not 50_000_000 <= breakdown["total"] <= 300_000_000:
        parser.error(f"Model parameter count {breakdown['total']:,} is out of range")
    raw_model.to(device)

    dist.init_process_group(backend=backend)
    try:
        if dist.get_world_size() != world_size:
            raise RuntimeError("Distributed world size mismatch")
        max_token_id = torch.tensor([0], dtype=torch.long, device=device)
        if rank == 0:
            max_token_id[0] = max(int(train_set.data.max()),
                                  int(full_val_set.data.max()))
        dist.broadcast(max_token_id, src=0)
        if max_token_id.item() >= manifest["vocab_size"]:
            raise RuntimeError("Packed token ID exceeds model vocabulary")
        causal_error = check_causal_mask(raw_model, manifest["vocab_size"], device)
        if rank == 0:
            print("MODEL PARAMETER BREAKDOWN: " + json.dumps(breakdown), flush=True)
            print(f"world_size={world_size}; physical GPU IDs={visible_ids}; "
                  f"per_gpu_batch={args.per_gpu_batch}; grad_accum={args.grad_accum}; "
                  f"global_sequence_batch={effective_sequences}; "
                  f"tokens_per_update={tokens_per_update}; total_steps={total_steps}; "
                  f"causal_mask_max_error={causal_error:.3g}", flush=True)
            if special_ids["<bos>"] is None:
                print("Tokenizer has no <bos>; model BOS uses <eos> ID.", flush=True)
        run_training(args, rank, world_size, device, selection, visible_ids,
                     manifest, special_ids, tokenizer, tokenizer_hash,
                     token_manifest_path, train_set, full_val_set,
                     periodic_val_set, raw_model, breakdown, total_steps,
                     warmup_steps, tokens_per_update, effective_sequences,
                     session_start, session_started_utc)
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
