#!/usr/bin/env python3
"""Refit the measured MoE/LoRA/BILOU-CRF recipe on all Phase-2 training docs.

This final refit deliberately reports no held-out score: every labeled document
is used for training. The separate fixed-split E2/E7 run selects the recipe.
"""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import random
import time
from zipfile import ZIP_DEFLATED, ZipFile

import torch
from torch.utils.data import DataLoader
from tokenizers import Tokenizer
from tqdm import tqdm

from ner_core import LABELS, collate, read_folds, tokenize_document, windows
from structured_ner import StructuredNER
from sweep_crf_inference import NumpyCRF, decode_record, emission_bias, make_cache
from train_ner import sha256
from train_structured_ner import parent_config, training_loss


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--data', type=Path, required=True)
    ap.add_argument('--tokenizer', type=Path, required=True)
    ap.add_argument('--parent-checkpoint', type=Path, required=True)
    ap.add_argument('--output-dir', type=Path, required=True)
    ap.add_argument('--submission-dir', type=Path, required=True)
    ap.add_argument('--epochs', type=int, default=4)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--length', type=int, default=512)
    ap.add_argument('--stride', type=int, default=384)
    ap.add_argument('--batch-size', type=int, default=4)
    ap.add_argument('--inference-length', type=int, default=1024)
    ap.add_argument('--inference-stride', type=int, default=768)
    ap.add_argument('--inference-mode', choices=('latest','stitch'), default='latest')
    ap.add_argument('--recall-bias', type=float, default=.5)
    ap.add_argument('--bidir-head', action='store_true')
    args = ap.parse_args()
    if args.output_dir.exists() or args.submission_dir.exists():
        ap.error('Output directory exists')
    if not 0 < args.stride <= args.length or not 0 < args.inference_stride <= args.inference_length:
        ap.error('Invalid window or stride')
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_float32_matmul_precision('high')
    device = torch.device('cuda:0')
    tokenizer = Tokenizer.from_file(str(args.tokenizer))
    parent_path = args.parent_checkpoint / 'state.pt'
    parent, kind, config = parent_config(parent_path, args.tokenizer)
    if kind != 'moe':
        ap.error('Expected the MoE Phase-1 parent')
    model = StructuredNER(config, kind, 'lora', 'bilou_crf',
                          bidir_head=args.bidir_head)
    model.encoder.backbone.load_state_dict(parent['model'], strict=True)
    del parent
    model.configure_adaptation()
    model.to(device)
    rows = read_folds(args.data)
    training_ids = sorted(rows)
    docs = {key: tokenize_document(key, row['text'], tokenizer, row['entities'], row['fold'])
            for key, row in rows.items()}
    batches = [part for key in training_ids
               for part in windows(docs[key], args.length, args.stride)]
    loader = DataLoader(batches, batch_size=args.batch_size, shuffle=True,
                        collate_fn=collate, generator=torch.Generator().manual_seed(args.seed))
    opt = torch.optim.AdamW([
        {'params': [p for p in model.encoder.parameters() if p.requires_grad], 'lr': .001},
        {'params': [p for name,p in model.named_parameters()
                    if not name.startswith('encoder.') and p.requires_grad], 'lr': .0005},
    ], weight_decay=.01, fused=True)
    steps = args.epochs * len(loader)
    warmup = max(1, steps // 20)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda step: (step+1)/warmup if step < warmup else
        .1 + .45*(1+math.cos(math.pi*(step-warmup)/max(1,steps-warmup))))
    args.output_dir.mkdir(parents=True)
    config_record = {
        'parent_checkpoint': str(parent_path.resolve()), 'parent_sha256': sha256(parent_path),
        'tokenizer': str(args.tokenizer.resolve()), 'tokenizer_sha256': sha256(args.tokenizer),
        'training_ids': training_ids, 'validation_ids': [], 'method': 'lora',
        'structure': 'bilou_crf', 'backbone_kind': 'moe',
        'bidir_head': args.bidir_head,
        'parameters': sum(p.numel() for p in model.parameters()),
        'trainable_parameters': sum(p.numel() for p in model.parameters() if p.requires_grad),
        'selection_basis': 'E2/E7 fixed-split validation; no local validation for this full-data refit',
        'arguments': {k: str(v) if isinstance(v, Path) else v for k,v in vars(args).items()},
    }
    (args.output_dir / 'config.json').write_text(json.dumps(config_record, indent=2)+'\n')
    started = time.monotonic()
    for epoch in range(1, args.epochs+1):
        model.train()
        total = 0.
        for ids,tags,lengths in tqdm(loader, desc=f'full-data CRF epoch {epoch}', unit='batch'):
            ids,tags = ids.to(device),tags.to(device)
            opt.zero_grad(set_to_none=True)
            with torch.autocast('cuda', dtype=torch.bfloat16):
                loss = training_loss(model,ids,tags,lengths,'bilou_crf',32)
            if not torch.isfinite(loss):
                raise RuntimeError('Non-finite CRF training loss')
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
            opt.step();scheduler.step()
            total += float(loss.detach())
        metric = {'epoch':epoch,'train_loss':total/len(loader),
                  'elapsed_seconds':time.monotonic()-started,
                  'validation_macro_f1':None}
        with (args.output_dir/'metrics.jsonl').open('a') as stream:
            stream.write(json.dumps(metric)+'\n')
        print(json.dumps(metric),flush=True)
    model.eval()
    state = {'model':model.state_dict(), 'model_config':config.to_dict(),
             'meta':{'structure':'bilou_crf','backbone_kind':'moe','method':'lora',
                     'epoch':args.epochs,'validation_macro_f1':None,
                     'bidir_head':args.bidir_head}}
    torch.save(state,args.output_dir/'final.pt.tmp')
    os.replace(args.output_dir/'final.pt.tmp',args.output_dir/'final.pt')
    test = json.loads((args.data/'Test/fold3-text.json').read_text())
    test_docs = [tokenize_document(key,text,tokenizer) for key,text in test.items()]
    cache = make_cache(model,test_docs,args.inference_length,args.inference_stride,device)
    crf = NumpyCRF(model.crf)
    bias_spec = {name:args.recall_bias for name in ('APP','RESP','AUTH')}
    bias = emission_bias(bias_spec)
    prediction = {key:decode_record(row,crf,args.inference_mode,bias)
                  for key,row in cache.items()}
    if set(prediction) != set(test):
        raise RuntimeError('Official test ID mismatch')
    for key,spans in prediction.items():
        seen=set()
        for span in spans:
            if (len(span)!=3 or type(span[0]) is not int or type(span[1]) is not int
                    or not 0<=span[0]<span[1]<=len(test[key]) or span[2] not in LABELS
                    or tuple(span) in seen):
                raise RuntimeError(f'Invalid official test span: {key} {span}')
            seen.add(tuple(span))
    args.submission_dir.mkdir(parents=True)
    output = args.submission_dir/'prediction.json'
    output.write_text(json.dumps(prediction,ensure_ascii=False,indent=2)+'\n')
    archive = args.submission_dir/'submission.zip'
    with ZipFile(archive,'w',ZIP_DEFLATED,compresslevel=9) as stream:
        stream.write(output,arcname='prediction.json')
    with ZipFile(archive) as stream:
        if stream.namelist()!=['prediction.json'] or stream.testzip() is not None:
            raise RuntimeError('Invalid archive')
    manifest = {'checkpoint':str((args.output_dir/'final.pt').resolve()),
                'checkpoint_sha256':sha256(args.output_dir/'final.pt'),
                'tokenizer_sha256':sha256(args.tokenizer),
                'test_sha256':sha256(args.data/'Test/fold3-text.json'),
                'prediction_sha256':sha256(output),'submission_zip_sha256':sha256(archive),
                'documents':len(prediction),'predicted_entities':sum(map(len,prediction.values())),
                'labels':LABELS,'inference_variant':{'length':args.inference_length,
                    'stride':args.inference_stride,'mode':args.inference_mode,
                    'bias':bias_spec},
                'validation_macro_f1':None}
    (args.submission_dir/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(f'Saved {len(prediction)} official IDs to {archive}',flush=True)


if __name__=='__main__':
    main()
