#!/usr/bin/env python3
"""Train and package CRF, boundary, or biaffine legal NER experiments."""
from __future__ import annotations

import argparse
from contextlib import nullcontext
import json
import math
import os
from pathlib import Path
import random
import time
from zipfile import ZipFile, ZIP_DEFLATED

import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
from tokenizers import Tokenizer
from tqdm import tqdm

from ner_core import LABELS, TAGS, collate, read_folds, strict_macro_f1, tokenize_document, windows
from structured_ner import BILOU, StructuredNER, bilou_tags, decode_bilou, gold_spans, nonoverlap
from train_ner import sha256, split_documents
from src.model import LegalLMConfig


def parent_config(path, tokenizer):
    parent = torch.load(path, map_location='cpu', mmap=True, weights_only=False)
    run = parent['run_config']
    architecture = run['architecture']
    kinds = {
        'Custom decoder-only pre-RMSNorm RoPE GQA SwiGLU Transformer': 'dense',
        'Legal-MoE-251M Top-1 decoder-only RoPE GQA SwiGLU Transformer': 'moe',
        'GPT2-style decoder-only pre-LN causal Transformer': 'gpt2',
        'LEGAL-UL2-SEQ2SEQ': 'ul2',
    }
    kind = kinds[architecture]
    if run['tokenizer_sha256'] != sha256(tokenizer):
        raise ValueError('Parent tokenizer SHA256 mismatch')
    if kind in ('dense', 'moe'):
        config = LegalLMConfig(**run['model_config'])
    elif kind == 'gpt2':
        config = run
    else:
        tok = Tokenizer.from_file(str(tokenizer))
        config = {'vocab_size': run['vocab_size'], 'pad_id': tok.token_to_id('<pad>'),
                  'eos_id': tok.token_to_id('<eos>')}
    return parent, kind, config


def candidate_indices(tags, lengths, structure, max_span):
    """Aligned positives and bounded sampled negatives for span training."""
    batch, starts, ends, labels = [], [], [], []
    for b, length in enumerate(lengths):
        spans = gold_spans(tags[b, :length].tolist())
        positive = {(i, j) for i, j, _ in spans}
        for i, j, label in spans:
            batch.append(b); starts.append(i); ends.append(j); labels.append(label)
        if structure != 'biaffine':
            continue
        target = max(48, 3 * len(spans))
        negatives = set()
        for _ in range(target * 12):
            if len(negatives) >= target:
                break
            i = random.randrange(length)
            j = min(length - 1, i + random.randrange(max_span))
            if (i, j) not in positive:
                negatives.add((i, j))
        for i, j in negatives:
            batch.append(b); starts.append(i); ends.append(j); labels.append(0)
    return batch, starts, ends, labels


def training_loss(model, ids, tags, lengths, structure, max_span, hidden=None,
                  minority_weight=1.):
    if hidden is None:
        hidden = model.encode(ids)
    if structure == 'bilou_crf':
        converted = torch.tensor([bilou_tags(tags[b, :n].tolist()) + [-100] * (tags.shape[1] - n)
                                  for b, n in enumerate(lengths)], device=ids.device)
        emissions = model.emit(hidden, lengths).float()
        weights = torch.ones(len(BILOU), device=ids.device)
        weights[0] = .12
        for index, tag in enumerate(BILOU):
            if tag.endswith(('-APP', '-RESP', '-AUTH')):
                weights[index] = minority_weight
        return model.crf(emissions, converted, lengths) + .6 * F.cross_entropy(
            emissions.reshape(-1, len(BILOU)), converted.reshape(-1),
            weight=weights, ignore_index=-100)
    start_target = torch.zeros_like(tags)
    end_target = torch.zeros_like(tags)
    start_target[tags == -100] = -100
    end_target[tags == -100] = -100
    for b, n in enumerate(lengths):
        for i, j, label in gold_spans(tags[b, :n].tolist()):
            start_target[b, i] = label if structure == 'start_end' else 1
            end_target[b, j] = label if structure == 'start_end' else 1
    start_logits, end_logits = model.boundary_logits(hidden)
    nclass = len(LABELS) + 1 if structure == 'start_end' else 2
    weights = torch.ones(nclass, device=ids.device)
    weights[0] = .08
    loss = F.cross_entropy(start_logits.float().reshape(-1, nclass),
                           start_target.reshape(-1), weight=weights, ignore_index=-100)
    loss += F.cross_entropy(end_logits.float().reshape(-1, nclass),
                            end_target.reshape(-1), weight=weights, ignore_index=-100)
    b, i, j, labels = candidate_indices(tags, lengths, structure, max_span)
    if labels:
        logits = model.span_logits(hidden, b, i, j).float()
        target = torch.tensor(labels, device=ids.device)
        if structure == 'start_end':
            loss += .3 * F.cross_entropy(logits, target - 1)
        else:
            weights = torch.ones(nclass + 11, device=ids.device)
            weights[0] = .4
            loss += .5 * F.cross_entropy(logits, target, weight=weights)
    return loss


def distillation_loss(student, teacher, ids, tags, student_hidden, temperature=2.):
    """Match the trained teacher's token distributions at aligned BPE positions."""
    valid = tags != -100
    with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16):
        teacher_hidden = teacher.encode(ids)
        teacher_heads = teacher.boundary_logits(teacher_hidden)
    student_heads = student.boundary_logits(student_hidden)
    if not isinstance(student_heads, tuple):
        student_heads, teacher_heads = (student_heads,), (teacher_heads,)
    total = 0.
    for student_logits, teacher_logits in zip(student_heads, teacher_heads):
        total = total + F.kl_div(
            F.log_softmax(student_logits[valid].float() / temperature, -1),
            F.softmax(teacher_logits[valid].float() / temperature, -1),
            reduction='batchmean') * temperature**2
    return total / len(student_heads)


@torch.inference_mode()
def predict_documents(model, docs, length, stride, device, max_span, threshold):
    model.eval()
    amp = torch.autocast('cuda', dtype=torch.bfloat16) if device.type == 'cuda' else nullcontext()
    predictions = {}
    for doc in docs:
        if model.structure == 'bilou_crf':
            chosen = [0] * len(doc.ids)
            priority = [-1] * len(doc.ids)
        else:
            candidates = {}
        for row in windows(doc, length, stride):
            ids = torch.tensor(row['ids'], device=device)[None]
            with amp:
                hidden = model.encode(ids)
                if model.structure == 'bilou_crf':
                    emission = model.emit(hidden)[0]
                else:
                    s_logits, e_logits = model.boundary_logits(hidden)
            if model.structure == 'bilou_crf':
                tags = model.crf.decode(emission)
                for local, tag in enumerate(tags):
                    global_i = row['start'] + local
                    if local > priority[global_i]:
                        chosen[global_i] = tag
                        priority[global_i] = local
                continue
            s = s_logits[0].float().softmax(-1)
            e = e_logits[0].float().softmax(-1)
            if model.structure == 'start_end':
                s_score, s_label = s[:, 1:].max(-1)
                e_score, e_label = e[:, 1:].max(-1)
            else:
                s_score, e_score = s[:, 1], e[:, 1]
            ns = min(128, len(row['ids']))
            ne = min(128, len(row['ids']))
            start_top = s_score.topk(ns).indices.tolist()
            end_top = e_score.topk(ne).indices.tolist()
            s_score, e_score = s_score.cpu().tolist(), e_score.cpu().tolist()
            if model.structure == 'start_end':
                s_label, e_label = s_label.cpu().tolist(), e_label.cpu().tolist()
            start_set = sorted(x for x in start_top if s_score[x] > .08)
            end_set = sorted(x for x in end_top if e_score[x] > .08)
            pairs = []
            for i in start_set:
                for j in end_set:
                    if j < i or j - i >= max_span:
                        continue
                    if model.structure == 'start_end' and s_label[i] != e_label[j]:
                        continue
                    pairs.append((i, j))
            if not pairs:
                continue
            with amp:
                span_logits = model.span_logits(hidden, [0] * len(pairs),
                                                [x[0] for x in pairs],
                                                [x[1] for x in pairs]).float()
            probabilities = span_logits.softmax(-1).cpu().numpy()
            for p, (i, j) in enumerate(pairs):
                if model.structure == 'start_end':
                    label_id = s_label[i] + 1
                    type_p = float(probabilities[p, label_id - 1])
                else:
                    label_id = int(probabilities[p, 1:].argmax()) + 1
                    type_p = float(probabilities[p, label_id])
                    if type_p <= float(probabilities[p, 0]):
                        continue
                score = s_score[i] * e_score[j] * type_p
                if score < threshold:
                    continue
                a, b = doc.offsets[row['start'] + i][0], doc.offsets[row['start'] + j][1]
                if not 0 <= a < b <= len(doc.text):
                    continue
                key = (a, b, LABELS[label_id - 1])
                candidates[key] = max(score, candidates.get(key, 0.))
        if model.structure == 'bilou_crf':
            predictions[doc.doc_id] = decode_bilou(chosen, doc.offsets, doc.text)
        else:
            predictions[doc.doc_id] = nonoverlap([(*key, score) for key, score in candidates.items()])
    model.train()
    return predictions


def boundary_f1(predictions, gold):
    p = {(d, a, b) for d, rows in predictions.items() for a, b, _ in rows}
    g = {(d, a, b) for d, rows in gold.items() for a, b, *_ in rows}
    return 2 * len(p & g) / (len(p) + len(g)) if p or g else 0.


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--data', type=Path, required=True)
    ap.add_argument('--tokenizer', type=Path, required=True)
    ap.add_argument('--parent-checkpoint', type=Path, required=True)
    ap.add_argument('--output-dir', type=Path, required=True)
    ap.add_argument('--submission-dir', type=Path, required=True)
    ap.add_argument('--structure', choices=('bilou_crf', 'start_end', 'biaffine'), required=True)
    ap.add_argument('--method', choices=('full', 'partial', 'lora', 'lorma_plus', 'adapter'), required=True)
    ap.add_argument('--epochs', type=int, default=5)
    ap.add_argument('--length', type=int, default=512)
    ap.add_argument('--stride', type=int, default=384)
    ap.add_argument('--batch-size', type=int, default=4)
    ap.add_argument('--max-span', type=int, default=32)
    ap.add_argument('--threshold', type=float, default=.025)
    ap.add_argument('--backbone-lr', type=float, default=1e-3)
    ap.add_argument('--head-lr', type=float, default=5e-4)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--minority-weight', type=float, default=1.,
                    help='Cross-entropy weight for APP, RESP, and AUTH BILOU tags')
    ap.add_argument('--bidir-head', action='store_true',
                    help='Add a bidirectional GRU above the pretrained decoder states')
    ap.add_argument('--transformer-head', action='store_true',
                    help='Add a two-layer bidirectional Transformer tagging head')
    ap.add_argument('--split-seed', type=int,
                    help='Use a fixed document split while varying the training seed')
    ap.add_argument('--teacher-checkpoint', type=Path,
                    help='Train an eight-layer student by distillation from this structured NER teacher')
    ap.add_argument('--distill-weight', type=float, default=.5)
    args = ap.parse_args()
    if args.minority_weight <= 0:
        ap.error('--minority-weight must be positive')
    if args.bidir_head and args.structure != 'bilou_crf':
        ap.error('--bidir-head requires BILOU-CRF')
    if args.transformer_head and args.structure != 'bilou_crf':
        ap.error('--transformer-head requires BILOU-CRF')
    if args.transformer_head and args.bidir_head:
        ap.error('Choose one bidirectional head')
    if args.output_dir.exists() or args.submission_dir.exists():
        ap.error('Output already exists')
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_float32_matmul_precision('high')
    device = torch.device('cuda:0')
    tokenizer = Tokenizer.from_file(str(args.tokenizer))
    parent_path = args.parent_checkpoint / 'state.pt'
    parent, kind, config = parent_config(parent_path, args.tokenizer)
    if kind in ('gpt2', 'ul2') and args.method not in ('full',):
        ap.error('This adaptation method requires dense/MoE blocks')
    if args.teacher_checkpoint:
        if kind != 'dense':
            ap.error('Structured student distillation currently requires the dense Phase-1 model')
        compact = config.to_dict()
        compact['n_layers'] = 8
        config = LegalLMConfig(**compact)
    model = StructuredNER(config, kind, args.method, args.structure,
                          bidir_head=args.bidir_head,
                          transformer_head=args.transformer_head)
    source_weights = parent['model']
    if args.teacher_checkpoint:
        source_weights = {k: v for k, v in source_weights.items()
                          if not k.startswith('blocks.') or int(k.split('.')[1]) < 8}
    model.encoder.backbone.load_state_dict(source_weights, strict=True)
    del parent
    model.configure_adaptation()
    model.to(device)
    teacher = None
    if args.teacher_checkpoint:
        state = torch.load(args.teacher_checkpoint, map_location='cpu', weights_only=False)
        teacher_meta = state['meta']
        if teacher_meta['structure'] != args.structure:
            ap.error('Teacher and student NER structures differ')
        teacher_config = LegalLMConfig(**state['model_config'])
        teacher = StructuredNER(teacher_config, teacher_meta['backbone_kind'],
                                teacher_meta['method'], args.structure)
        teacher.configure_adaptation()
        teacher.load_state_dict(state['model'], strict=True)
        teacher.to(device).eval().requires_grad_(False)
        del state
    rows = read_folds(args.data)
    training_ids, validation_ids = split_documents(
        rows, args.seed if args.split_seed is None else args.split_seed)
    docs = {k: tokenize_document(k, r['text'], tokenizer, r['entities'], r['fold'])
            for k, r in rows.items()}
    train_windows = [w for k in training_ids for w in windows(docs[k], args.length, args.stride)]
    loader = DataLoader(train_windows, batch_size=args.batch_size, shuffle=True,
                        collate_fn=collate, generator=torch.Generator().manual_seed(args.seed))
    groups = [
        {'params': [p for p in model.encoder.parameters() if p.requires_grad], 'lr': args.backbone_lr},
        {'params': [p for name, p in model.named_parameters() if not name.startswith('encoder.') and p.requires_grad], 'lr': args.head_lr},
    ]
    opt = torch.optim.AdamW(groups, weight_decay=.01, fused=True)
    steps = args.epochs * len(loader)
    warmup = max(1, steps // 20)
    scheduler = torch.optim.lr_scheduler.LambdaLR(opt, lambda step: (step + 1) / warmup if step < warmup else .1 + .45 * (1 + math.cos(math.pi * (step - warmup) / max(1, steps - warmup))))
    args.output_dir.mkdir(parents=True)
    provenance = {'parent_checkpoint': str(parent_path.resolve()), 'parent_sha256': sha256(parent_path),
                  'tokenizer': str(args.tokenizer.resolve()), 'tokenizer_sha256': sha256(args.tokenizer),
                  'training_ids': training_ids, 'validation_ids': validation_ids,
                  'method': 'distillation' if teacher else args.method,
                  'structure': args.structure, 'backbone_kind': kind,
                  'bidir_head': args.bidir_head,
                  'transformer_head': args.transformer_head,
                  'parameters': sum(p.numel() for p in model.parameters()),
                  'trainable_parameters': sum(p.numel() for p in model.parameters() if p.requires_grad),
                  'arguments': {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}}
    if teacher:
        provenance['teacher_checkpoint'] = str(args.teacher_checkpoint.resolve())
        provenance['teacher_sha256'] = sha256(args.teacher_checkpoint)
        provenance['student_layers'] = 8
    (args.output_dir / 'config.json').write_text(json.dumps(provenance, indent=2) + '\n')
    gold = {k: rows[k]['entities'] for k in validation_ids}
    val_docs = [docs[k] for k in validation_ids]
    best = -1.
    started = time.monotonic()
    for epoch in range(1, args.epochs + 1):
        model.train()
        cumulative = 0.
        for ids, tags, lengths in tqdm(loader, desc=f'{args.structure} epoch {epoch}', unit='batch'):
            ids, tags = ids.to(device), tags.to(device)
            opt.zero_grad(set_to_none=True)
            with torch.autocast('cuda', dtype=torch.bfloat16):
                if teacher:
                    student_hidden = model.encode(ids)
                    supervised = training_loss(model, ids, tags, lengths,
                                               args.structure, args.max_span, student_hidden,
                                               args.minority_weight)
                    soft = distillation_loss(model, teacher, ids, tags, student_hidden)
                    loss = (1 - args.distill_weight) * supervised + args.distill_weight * soft
                else:
                    loss = training_loss(model, ids, tags, lengths,
                                         args.structure, args.max_span,
                                         minority_weight=args.minority_weight)
            if not torch.isfinite(loss):
                raise RuntimeError('Non-finite structured NER loss')
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
            opt.step(); scheduler.step()
            cumulative += float(loss.detach())
        prediction = predict_documents(model, val_docs, args.length, args.stride, device,
                                       args.max_span, args.threshold)
        scored = strict_macro_f1(prediction, gold)
        metric = {'epoch': epoch, 'train_loss': cumulative / len(loader),
                  'validation_macro_f1': scored['macro_f1'],
                  'boundary_f1': boundary_f1(prediction, gold),
                  'per_label': scored['per_label'],
                  'method': 'distillation' if teacher else args.method,
                  'backbone_kind': kind, 'structure': args.structure,
                  'bidir_head': args.bidir_head,
                  'transformer_head': args.transformer_head,
                  'elapsed_seconds': time.monotonic() - started}
        with (args.output_dir / 'metrics.jsonl').open('a') as file:
            file.write(json.dumps(metric) + '\n')
        print(json.dumps({k: v for k, v in metric.items() if k != 'per_label'}), flush=True)
        state = {'model': model.state_dict(),
                 'model_config': config.to_dict() if hasattr(config, 'to_dict') else config,
                 'meta': metric}
        torch.save(state, args.output_dir / 'latest.pt.tmp')
        os.replace(args.output_dir / 'latest.pt.tmp', args.output_dir / 'latest.pt')
        if metric['validation_macro_f1'] > best:
            best = metric['validation_macro_f1']
            torch.save(state, args.output_dir / 'best.pt.tmp')
            os.replace(args.output_dir / 'best.pt.tmp', args.output_dir / 'best.pt')
            (args.output_dir / 'validation_predictions.json').write_text(json.dumps(prediction) + '\n')
    print(f'Best held-out strict macro F1={best:.6f}', flush=True)
    state = torch.load(args.output_dir / 'best.pt', map_location='cpu', weights_only=False)
    model.load_state_dict(state['model'], strict=True)
    test = json.loads((args.data / 'Test/fold3-text.json').read_text())
    test_docs = [tokenize_document(k, v, tokenizer) for k, v in test.items()]
    prediction = predict_documents(model, test_docs, args.length, args.stride, device,
                                   args.max_span, args.threshold)
    if set(prediction) != set(test):
        raise RuntimeError('Test ID mismatch')
    args.submission_dir.mkdir(parents=True)
    output = args.submission_dir / 'prediction.json'
    output.write_text(json.dumps(prediction, ensure_ascii=False, indent=2) + '\n')
    with ZipFile(args.submission_dir / 'submission.zip', 'w', ZIP_DEFLATED, compresslevel=9) as archive:
        archive.write(output, arcname='prediction.json')
    (args.submission_dir / 'manifest.json').write_text(json.dumps({
        'checkpoint': str((args.output_dir / 'best.pt').resolve()),
        'checkpoint_sha256': sha256(args.output_dir / 'best.pt'),
        'tokenizer_sha256': sha256(args.tokenizer),
        'test_sha256': sha256(args.data / 'Test/fold3-text.json'),
        'prediction_sha256': sha256(output),
        'submission_zip_sha256': sha256(args.submission_dir / 'submission.zip'),
        'documents': len(prediction), 'predicted_entities': sum(map(len, prediction.values())),
        'labels': LABELS, 'structure': args.structure}, indent=2) + '\n')
    print(f'Saved {len(prediction)} test documents in {args.submission_dir}', flush=True)


if __name__ == '__main__':
    main()
