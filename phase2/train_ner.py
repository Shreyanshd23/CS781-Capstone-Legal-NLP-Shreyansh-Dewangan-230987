#!/usr/bin/env python3
"""Fine-tune the Phase-1 dense decoder for exact-span Phase-2 legal NER."""

from __future__ import annotations

import argparse
from collections import Counter
from contextlib import nullcontext
import hashlib
import json
import math
import os
from pathlib import Path
import random
import time

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
from tokenizers import Tokenizer
from tqdm import tqdm

from ner_core import (DecoderNER, LABELS, TAGS, collate, decode_tags,
                      read_folds, strict_macro_f1, tokenize_document, windows)
from src.model import LegalLMConfig


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def split_documents(rows: dict, seed: int, val_per_fold: int = 5):
    validation = []
    for fold in (1, 2):
        ids = sorted(k for k, v in rows.items() if v['fold'] == fold)
        ids.sort(key=lambda k: hashlib.sha256(f'{seed}:{k}'.encode()).hexdigest())
        validation.extend(ids[:val_per_fold])
    validation = sorted(validation)
    training = sorted(set(rows) - set(validation))
    return training, validation


@torch.inference_mode()
def predict_documents(model, documents, length, stride, device):
    model.eval()
    predictions = {}
    amp = torch.autocast('cuda', dtype=torch.bfloat16) if device.type == 'cuda' else nullcontext()
    for doc in documents:
        chosen = [-1] * len(doc.ids)
        scores = [-1] * len(doc.ids)
        for row in windows(doc, length, stride):
            ids = torch.tensor(row['ids'], dtype=torch.long, device=device)[None]
            with amp:
                logits = model(ids)
            tags = logits[0].argmax(-1).tolist()
            for local, tag in enumerate(tags):
                global_index = row['start'] + local
                # Prefer the prediction with the longest causal prefix.
                if local > scores[global_index]:
                    chosen[global_index] = tag
                    scores[global_index] = local
        predictions[doc.doc_id] = decode_tags(chosen, doc.offsets, doc.text)
    model.train()
    return predictions


def save_checkpoint(path: Path, model, config, meta):
    temporary = path.with_suffix('.tmp')
    torch.save({'model': model.state_dict(), 'model_config': (config.to_dict()
                if hasattr(config, 'to_dict') else config),
                'tags': TAGS, 'meta': meta}, temporary)
    os.replace(temporary, path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--tokenizer', type=Path, required=True)
    parser.add_argument('--parent-checkpoint', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--epochs', type=int, default=4)
    parser.add_argument('--length', type=int, default=1024)
    parser.add_argument('--stride', type=int, default=768)
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--backbone-lr', type=float, default=2e-5)
    parser.add_argument('--head-lr', type=float, default=5e-4)
    parser.add_argument('--outside-weight', type=float, default=0.2)
    parser.add_argument('--begin-weight', type=float, default=2.0)
    parser.add_argument('--inside-weight', type=float, default=1.0)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--method', choices=('full', 'partial', 'lora',
                                              'lorma_plus', 'adapter', 'norm',
                                              'encoder_full'),
                        default='full')
    args = parser.parse_args()
    if args.output_dir.exists():
        parser.error(f'Output already exists: {args.output_dir}')
    if args.epochs < 1 or args.length < 1 or not 0 < args.stride <= args.length:
        parser.error('Invalid epochs, length, or stride')
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_float32_matmul_precision('high')
    device = torch.device(args.device)
    if device.type == 'cuda' and not torch.cuda.is_bf16_supported():
        parser.error('BF16 CUDA device required')
    tokenizer = Tokenizer.from_file(str(args.tokenizer))
    parent_file = args.parent_checkpoint / 'state.pt'
    parent = torch.load(parent_file, map_location='cpu', mmap=True, weights_only=False)
    run = parent['run_config']
    if run['architecture'] == 'Custom decoder-only pre-RMSNorm RoPE GQA SwiGLU Transformer':
        backbone_kind = 'dense'
    elif run['architecture'] == 'Legal-MoE-251M Top-1 decoder-only RoPE GQA SwiGLU Transformer':
        backbone_kind = 'moe'
    elif run['architecture'] == 'GPT2-style decoder-only pre-LN causal Transformer':
        backbone_kind = 'gpt2'
    elif run['architecture'] == 'LEGAL-UL2-SEQ2SEQ':
        backbone_kind = 'ul2'
    else:
        parser.error('Expected a trained Phase-1 legal decoder or MoE checkpoint')
    if run['tokenizer_sha256'] != sha256(args.tokenizer):
        parser.error('Tokenizer differs from Phase-1 checkpoint')
    if backbone_kind in ('dense', 'moe'):
        config = LegalLMConfig(**run['model_config'])
        context_length = config.context_length
    elif backbone_kind == 'gpt2':
        config = run
        context_length = run['block_size']
    else:
        config = {'vocab_size': run['vocab_size'],
                  'pad_id': tokenizer.token_to_id('<pad>'),
                  'eos_id': tokenizer.token_to_id('<eos>')}
        context_length = 2048
    if args.length > context_length:
        parser.error('Window exceeds Phase-1 model context')
    model = DecoderNER(config, backbone_kind, args.method)
    model.backbone.load_state_dict(parent['model'], strict=True)
    del parent
    model.configure_adaptation()
    model.to(device)
    rows = read_folds(args.data)
    train_ids, val_ids = split_documents(rows, args.seed)
    documents = {doc_id: tokenize_document(doc_id, row['text'], tokenizer,
                                           row['entities'], row['fold'])
                 for doc_id, row in rows.items()}
    train_windows = [window for doc_id in train_ids
                     for window in windows(documents[doc_id], args.length, args.stride)]
    val_docs = [documents[doc_id] for doc_id in val_ids]
    aligned = sum(doc.aligned for doc in documents.values())
    total = sum(doc.total for doc in documents.values())
    print(f'Train docs={len(train_ids)}, validation docs={len(val_ids)}, '
          f'windows={len(train_windows)}, aligned gold spans={aligned}/{total}', flush=True)
    generator = torch.Generator().manual_seed(args.seed)
    loader = DataLoader(train_windows, batch_size=args.batch_size, shuffle=True,
                        collate_fn=collate, generator=generator, num_workers=0)
    counts = Counter(tag for row in train_windows for tag in row['tags'] if tag >= 0)
    weights = torch.ones(len(TAGS), dtype=torch.float32, device=device)
    weights[0] = args.outside_weight
    for index, tag in enumerate(TAGS):
        if tag.startswith('B-'):
            weights[index] = args.begin_weight
        elif tag.startswith('I-'):
            weights[index] = args.inside_weight
    optimizer = torch.optim.AdamW([
        {'params': [p for p in model.backbone.parameters() if p.requires_grad],
         'lr': args.backbone_lr},
        {'params': model.classifier.parameters(), 'lr': args.head_lr}],
        weight_decay=0.01, fused=device.type == 'cuda')
    total_steps = args.epochs * len(loader)
    def decay(step):
        warmup = max(1, int(total_steps * 0.05))
        if step < warmup:
            return (step + 1) / warmup
        progress = (step - warmup) / max(1, total_steps - warmup)
        return 0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * progress))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, decay)
    args.output_dir.mkdir(parents=True)
    provenance = {'parent_checkpoint': str(parent_file.resolve()),
                  'parent_sha256': sha256(parent_file),
                  'tokenizer': str(args.tokenizer.resolve()),
                  'tokenizer_sha256': sha256(args.tokenizer),
                  'training_ids': train_ids, 'validation_ids': val_ids,
                  'alignment_rate': aligned / total,
                  'tag_counts': counts, 'method': args.method,
                  'backbone_kind': backbone_kind,
                  'parameters': sum(p.numel() for p in model.parameters()),
                  'trainable_parameters': sum(p.numel() for p in model.parameters()
                                              if p.requires_grad),
                  'arguments': vars(args)}
    provenance['arguments'] = {k: str(v) if isinstance(v, Path) else v
                               for k, v in provenance['arguments'].items()}
    (args.output_dir / 'config.json').write_text(json.dumps(provenance, indent=2) + '\n')
    amp = torch.autocast('cuda', dtype=torch.bfloat16) if device.type == 'cuda' else nullcontext()
    best = -1.0
    start_time = time.monotonic()
    metrics_file = args.output_dir / 'metrics.jsonl'
    for epoch in range(1, args.epochs + 1):
        model.train()
        loss_sum = 0.0
        for ids, tags, _ in tqdm(loader, desc=f'NER epoch {epoch}', unit='batch'):
            ids, tags = ids.to(device, non_blocking=True), tags.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with amp:
                logits = model(ids)
                loss = F.cross_entropy(logits.float().reshape(-1, len(TAGS)),
                                       tags.reshape(-1), weight=weights, ignore_index=-100)
            if not torch.isfinite(loss):
                raise RuntimeError('Non-finite NER loss')
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            loss_sum += float(loss.detach())
        prediction = predict_documents(model, val_docs, args.length, args.stride, device)
        gold = {doc_id: rows[doc_id]['entities'] for doc_id in val_ids}
        score = strict_macro_f1(prediction, gold)
        summary = {'epoch': epoch, 'train_loss': loss_sum / len(loader),
                   'validation_macro_f1': score['macro_f1'],
                   'method': args.method, 'backbone_kind': backbone_kind,
                   'per_label': score['per_label'],
                   'elapsed_seconds': time.monotonic() - start_time}
        with metrics_file.open('a') as handle:
            handle.write(json.dumps(summary) + '\n')
        print(json.dumps({k: v for k, v in summary.items() if k != 'per_label'}), flush=True)
        save_checkpoint(args.output_dir / 'latest.pt', model, config, summary)
        if score['macro_f1'] > best:
            best = score['macro_f1']
            save_checkpoint(args.output_dir / 'best.pt', model, config, summary)
            (args.output_dir / 'validation_predictions.json').write_text(
                json.dumps(prediction, ensure_ascii=False) + '\n')
    print(f'Best held-out strict macro F1={best:.6f}', flush=True)


if __name__ == '__main__':
    main()
