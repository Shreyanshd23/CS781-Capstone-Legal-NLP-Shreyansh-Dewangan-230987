#!/usr/bin/env python3
"""Four-GPU BF16 DDP trainer for the full official Legal-UL2 corpus."""

from __future__ import annotations

import argparse
from contextlib import nullcontext
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import random
import shutil
import tempfile
import time

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Sampler, Subset
from tqdm import tqdm

from legal_ul2_data import LegalUL2Examples, collate_ul2
from src.model.legal_ul2 import build_legal_ul2
from train_seq2seq_ddp import sha256


BUCKET_LIMITS = (256, 512, 1024, 1536, 2048, 2558)
PER_GPU_BATCH = (64, 32, 16, 8, 4, 4)


def utc_now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def append_jsonl(path: Path, value: dict):
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps({"timestamp_utc": utc_now(), **value}) + "\n")


class BucketedRankBatches(Sampler[list[int]]):
    """Deterministic shuffled length buckets, with every record seen once."""

    def __init__(self, records: np.ndarray, rank: int, world_size: int,
                 seed: int, start_step: int = 0):
        self.rank = rank
        self.world_size = world_size
        self.start_step = start_step
        rng = np.random.default_rng(seed)
        lengths = records["length"]
        modes = records["mode"]
        bins = np.searchsorted(BUCKET_LIMITS, lengths, side="left")
        if np.any(bins >= len(BUCKET_LIMITS)):
            raise ValueError("Record exceeds longest bucket")
        keys = modes.astype(np.int16) * len(BUCKET_LIMITS) + bins
        self.buckets = {}
        plan = []
        self.mode_records = [int(np.count_nonzero(modes == mode)) for mode in range(3)]
        for key in np.unique(keys):
            key = int(key)
            indices = np.flatnonzero(keys == key).astype(np.int32)
            rng.shuffle(indices)
            self.buckets[key] = indices
            batch_per_gpu = PER_GPU_BATCH[key % len(BUCKET_LIMITS)]
            global_batch = batch_per_gpu * world_size
            plan.extend((key, offset) for offset in range(0, len(indices), global_batch))
        rng.shuffle(plan)
        self.plan = plan
        if not 0 <= start_step <= len(plan):
            raise ValueError("Invalid resume step")

    def __len__(self):
        return len(self.plan) - self.start_step

    @property
    def total_steps(self):
        return len(self.plan)

    def __iter__(self):
        for key, offset in self.plan[self.start_step:]:
            bucket = self.buckets[key]
            per_gpu = PER_GPU_BATCH[key % len(BUCKET_LIMITS)]
            global_batch = per_gpu * self.world_size
            ids = bucket[offset:offset + global_batch]
            if len(ids) < global_batch:
                ids = np.resize(ids, global_batch)
            yield ids[self.rank * per_gpu:(self.rank + 1) * per_gpu].tolist()


@torch.inference_mode()
def evaluate(model, dataset, rank: int, world_size: int,
             device: torch.device, full: bool, max_examples: int = 1024):
    model.eval()
    if full:
        chosen = np.arange(len(dataset), dtype=np.int32)
    else:
        chosen = np.linspace(0, len(dataset) - 1,
                             min(max_examples, len(dataset)), dtype=np.int32)
    local_ids = chosen[rank::world_size].tolist()
    loader = DataLoader(Subset(dataset, local_ids), batch_size=4, shuffle=False,
                        num_workers=0,
                        collate_fn=lambda batch: collate_ul2(batch, dataset.pad_id))
    totals = torch.zeros(2, dtype=torch.float64, device=device)
    for source, mask, labels in loader:
        source = source.to(device, non_blocking=True)
        mask = mask.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            output = model(input_ids=source, attention_mask=mask,
                           labels=labels, use_cache=False)
        token_count = labels.ne(-100).sum()
        totals[0] += output.loss.double() * token_count
        totals[1] += token_count
    dist.all_reduce(totals, op=dist.ReduceOp.SUM)
    model.train()
    return float(totals[0] / totals[1]), int(totals[1].item())


def rng_state():
    return {"python": random.getstate(), "numpy": np.random.get_state(),
            "torch_cpu": torch.get_rng_state(), "torch_cuda": torch.cuda.get_rng_state()}


def restore_rng(state):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"])
    torch.cuda.set_rng_state(state["torch_cuda"])


def save_checkpoint(run: Path, model, optimizer, scheduler, run_config: dict,
                    step: int, best_loss: float, validation_loss: float | None,
                    elapsed: float, target_tokens_seen: int, rank: int, world_size: int):
    states = [None] * world_size
    dist.all_gather_object(states, rng_state())
    folder = run / "checkpoints"
    if rank == 0:
        folder.mkdir(exist_ok=True)
        target = folder / f"step_{step:07d}"
        if target.exists():
            raise RuntimeError(f"Checkpoint already exists: {target}")
        temporary = Path(tempfile.mkdtemp(prefix=".checkpoint_", dir=folder))
        try:
            torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                        "scheduler": scheduler.state_dict(), "run_config": run_config,
                        "step": step, "best_validation_loss": best_loss,
                        "validation_loss": validation_loss, "elapsed_seconds": elapsed,
                        "target_tokens_seen": target_tokens_seen,
                        "rng_states": states}, temporary / "state.pt")
            os.replace(temporary, target)
        finally:
            if temporary.exists():
                shutil.rmtree(temporary)
        for name in ("latest", "best" if validation_loss is not None and
                     validation_loss <= best_loss else None):
            if name is None:
                continue
            link = folder / name
            replacement = folder / f".{name}.tmp"
            replacement.unlink(missing_ok=True)
            replacement.symlink_to(target.name, target_is_directory=True)
            os.replace(replacement, link)
        protected = {p.resolve() for p in (folder / "latest", folder / "best") if p.exists()}
        older = sorted(folder.glob("step_*"))
        for old in older[:-3]:
            if old.resolve() not in protected:
                shutil.rmtree(old)
        append_jsonl(run / "metrics.jsonl", {"event": "checkpoint", "step": step,
                                           "path": str(target.resolve()),
                                           "validation_loss": validation_loss,
                                           "elapsed_seconds": elapsed})
    dist.barrier()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared-dir", type=Path,
                        default=Path("data/legal_ul2_full_20261006"))
    parser.add_argument("--tokens-dir", type=Path,
                        default=Path("data/tokens_bpe_full_15p95gib"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--warmup-ratio", type=float, default=0.02)
    parser.add_argument("--min-lr-ratio", type=float, default=0.1)
    parser.add_argument("--eval-every", type=int, default=5000)
    parser.add_argument("--save-every", type=int, default=5000)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=781)
    parser.add_argument("--stop-after", type=int,
                        help="Diagnostic stop with a resumable checkpoint")
    args = parser.parse_args()
    if min(args.eval_every, args.save_every, args.log_every) < 1 or args.workers < 0:
        parser.error("Invalid logging or worker count")
    if not 0 <= args.warmup_ratio < 1 or not 0 <= args.min_lr_ratio <= 1:
        parser.error("Invalid schedule ratio")
    if int(os.environ.get("WORLD_SIZE", "0")) != 4 or not torch.cuda.is_available():
        parser.error("Run with torchrun --nproc_per_node=4 and four CUDA GPUs")
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "7,4,2,1":
        parser.error("This run is pinned to physical GPUs 7,4,2,1")
    if torch.cuda.device_count() != 4 or not torch.cuda.is_bf16_supported():
        parser.error("Expected four BF16-capable visible GPUs")
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    if rank == 0 and not args.resume and args.output_dir.exists():
        parser.error(f"Run directory exists: {args.output_dir}")
    if args.resume and not args.output_dir.is_dir():
        parser.error("Resume run directory is missing")
    torch.cuda.set_device(local_rank)
    torch.set_float32_matmul_precision("high")
    device = torch.device("cuda", local_rank)
    dist.init_process_group("nccl", device_id=device)
    session_start = time.monotonic()
    try:
        metadata_file = args.prepared_dir / "manifest.json"
        metadata = json.loads(metadata_file.read_text())
        token_manifest = json.loads((args.tokens_dir / "token_manifest.json").read_text())
        if metadata["token_manifest_sha256"] != sha256(args.tokens_dir / "token_manifest.json"):
            raise RuntimeError("Packed corpus changed after UL2 indexing")
        if metadata["tokenizer_sha256"] != sha256(args.prepared_dir / "tokenizer.json"):
            raise RuntimeError("UL2 tokenizer changed after indexing")
        train_set = LegalUL2Examples(args.tokens_dir / "train.bin",
                                      args.prepared_dir / "train_records.npy",
                                      token_manifest["dtype"], metadata_file, args.seed)
        val_set = LegalUL2Examples(args.tokens_dir / "validation.bin",
                                    args.prepared_dir / "validation_records.npy",
                                    token_manifest["dtype"], metadata_file, args.seed + 1)
        sampler = BucketedRankBatches(train_set.records, rank, 4, args.seed)
        total_steps = sampler.total_steps
        warmup_steps = max(1, int(total_steps * args.warmup_ratio))
        run_config = {"architecture": "LEGAL-UL2-SEQ2SEQ",
                      "model": {"encoder_layers": 12, "decoder_layers": 12,
                                "d_model": 768, "heads": 12, "head_dim": 64,
                                "d_ff": 2048, "ffn": "GEGLU", "norm": "pre-RMSNorm",
                                "relative_position_buckets": 32,
                                "shared_input_embedding": True,
                                "separate_lm_head": True},
                      "preparation_sha256": sha256(metadata_file),
                      "tokenizer_sha256": metadata["tokenizer_sha256"],
                      "vocab_size": metadata["vocab_size"],
                      "train_records": len(train_set), "validation_records": len(val_set),
                      "train_non_eos_tokens_covered_once": metadata["train"]["non_eos_tokens_covered_once"],
                      "objective_mix_by_tokens": metadata["train"]["mode_tokens"],
                      "bucket_limits": BUCKET_LIMITS,
                      "per_gpu_batch_by_bucket": PER_GPU_BATCH,
                      "total_steps": total_steps, "world_size": 4,
                      "learning_rate": args.learning_rate,
                      "warmup_steps": warmup_steps,
                      "min_lr_ratio": args.min_lr_ratio,
                      "seed": args.seed, "precision": "bf16",
                      "optimizer": "AdamW", "grad_clip": 1.0,
                      "transformers_version": __import__("transformers").__version__}
        torch.manual_seed(args.seed)
        np.random.seed(args.seed + rank)
        random.seed(args.seed + rank)
        torch.cuda.manual_seed(args.seed + rank)
        model = build_legal_ul2(metadata["vocab_size"], metadata["pad_id"],
                                metadata["eos_id"]).to(device)
        parameter_count = sum(p.numel() for p in model.parameters())
        if not 240_000_000 <= parameter_count <= 260_000_000:
            raise RuntimeError(f"Unexpected parameter count {parameter_count:,}")
        run_config["parameter_count"] = parameter_count
        decay = [p for p in model.parameters() if p.requires_grad and p.ndim >= 2]
        no_decay = [p for p in model.parameters() if p.requires_grad and p.ndim < 2]
        optimizer = torch.optim.AdamW(
            [{"params": decay, "weight_decay": 0.1},
             {"params": no_decay, "weight_decay": 0}],
            lr=args.learning_rate, betas=(0.9, 0.95), fused=True)

        def factor(index: int):
            if index < warmup_steps:
                return (index + 1) / warmup_steps
            fraction = min(1.0, (index - warmup_steps) /
                           max(1, total_steps - warmup_steps))
            return (args.min_lr_ratio + (1 - args.min_lr_ratio) *
                    (1 + math.cos(math.pi * fraction)) / 2)

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, factor)
        step = 0
        best_loss = float("inf")
        previous_elapsed = 0.0
        target_tokens_seen = 0
        state = None
        if args.resume:
            checkpoint = args.resume / "state.pt" if args.resume.is_dir() else args.resume
            state = torch.load(checkpoint, map_location="cpu", weights_only=False)
            if state["run_config"] != run_config:
                raise RuntimeError("Checkpoint does not match this run configuration")
            model.load_state_dict(state["model"], strict=True)
            optimizer.load_state_dict(state["optimizer"])
            scheduler.load_state_dict(state["scheduler"])
            step = int(state["step"])
            best_loss = float(state["best_validation_loss"])
            previous_elapsed = float(state["elapsed_seconds"])
            target_tokens_seen = int(state["target_tokens_seen"])
        if rank == 0 and state is None:
            args.output_dir.mkdir(parents=True, exist_ok=False)
            (args.output_dir / "run_config.json").write_text(json.dumps(run_config, indent=2) + "\n")
            print(f"LEGAL-UL2: {parameter_count:,} parameters, {len(train_set):,} "
                  f"examples, {total_steps:,} updates, four H100 GPUs", flush=True)
        dist.barrier()
        sampler.start_step = step
        loader_options = {"num_workers": args.workers, "pin_memory": True,
                          "collate_fn": lambda batch: collate_ul2(batch, train_set.pad_id)}
        if args.workers:
            # Dataset workers only read the memory maps; example augmentation is
            # deterministic from the saved record index.
            loader_options.update(persistent_workers=True, prefetch_factor=2)
        loader = DataLoader(train_set, batch_sampler=sampler, **loader_options)
        wrapped = DDP(model, device_ids=[local_rank], broadcast_buffers=False,
                      gradient_as_bucket_view=True, find_unused_parameters=False)
        if state is not None:
            restore_rng(state["rng_states"][rank])
        wrapped.train()
        optimizer.zero_grad(set_to_none=True)
        progress = tqdm(total=total_steps, initial=step, desc="LEGAL-UL2",
                        unit="update", mininterval=5, dynamic_ncols=True,
                        disable=rank != 0)
        recent = torch.zeros(3, dtype=torch.float64, device=device)
        interval_start = time.monotonic()
        interval_step = step
        stop_step = min(total_steps, args.stop_after or total_steps)
        if stop_step < step:
            raise RuntimeError("Requested stop is earlier than resume step")
        for source, mask, labels in loader:
            source = source.to(device, non_blocking=True)
            mask = mask.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                output = wrapped(input_ids=source, attention_mask=mask,
                                 labels=labels, use_cache=False)
                loss = output.loss
            if not torch.isfinite(loss):
                raise RuntimeError(f"Non-finite loss at step {step}")
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            if not torch.isfinite(norm):
                raise RuntimeError(f"Non-finite gradient at step {step}")
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            step += 1
            progress.update(1)
            token_count = labels.ne(-100).sum()
            recent[0] += loss.detach().double() * token_count
            recent[1] += token_count
            recent[2] += source.shape[0]
            if step == 1 or step % args.log_every == 0 or step == stop_step:
                dist.all_reduce(recent, op=dist.ReduceOp.SUM)
                interval_seconds = max(time.monotonic() - interval_start, 1e-9)
                target_tokens_seen += int(recent[1].item())
                if rank == 0:
                    append_jsonl(args.output_dir / "metrics.jsonl", {
                        "event": "train", "step": step, "total_steps": total_steps,
                        "train_loss": float(recent[0] / recent[1]),
                        "target_tokens_seen": target_tokens_seen,
                        "examples_in_interval": int(recent[2].item()),
                        "target_tokens_per_second": recent[1].item() / interval_seconds,
                        "steps_per_second": (step - interval_step) / interval_seconds,
                        "learning_rate": optimizer.param_groups[0]["lr"],
                        "gradient_norm": float(norm),
                        "peak_gpu_allocated_gib": torch.cuda.max_memory_allocated(device) / 2**30,
                        "elapsed_seconds": previous_elapsed + time.monotonic() - session_start,
                    })
                    progress.set_postfix(loss=f"{recent[0]/recent[1]:.3f}",
                                         tok_s=f"{recent[1].item()/interval_seconds:.0f}")
                recent.zero_()
                interval_start = time.monotonic()
                interval_step = step
            validation_loss = None
            if step % args.eval_every == 0 or step == total_steps:
                validation_loss, validation_tokens = evaluate(
                    model, val_set, rank, 4, device, full=step == total_steps)
                improved = validation_loss < best_loss
                best_loss = min(best_loss, validation_loss)
                if rank == 0:
                    append_jsonl(args.output_dir / "metrics.jsonl", {
                        "event": "validation", "step": step, "loss": validation_loss,
                        "perplexity": math.exp(min(validation_loss, 30)),
                        "target_tokens": validation_tokens, "best": improved,
                        "full": step == total_steps,
                        "elapsed_seconds": previous_elapsed + time.monotonic() - session_start,
                    })
            if step % args.save_every == 0 or validation_loss is not None or step == stop_step:
                save_checkpoint(args.output_dir, model, optimizer, scheduler,
                                run_config, step, best_loss, validation_loss,
                                previous_elapsed + time.monotonic() - session_start,
                                target_tokens_seen, rank, 4)
            if step >= stop_step:
                break
        progress.close()
        if rank == 0:
            summary = {"status": "complete" if step == total_steps else "diagnostic_stop",
                       "step": step, "total_steps": total_steps,
                       "parameters": parameter_count,
                       "train_non_eos_tokens_covered_once": metadata["train"]["non_eos_tokens_covered_once"],
                       "target_tokens_seen": target_tokens_seen,
                       "best_validation_loss": (best_loss if math.isfinite(best_loss) else None),
                       "elapsed_seconds": previous_elapsed + time.monotonic() - session_start,
                       "checkpoint": str(args.output_dir / "checkpoints/best")}
            (args.output_dir / "run_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
            print(json.dumps(summary), flush=True)
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
