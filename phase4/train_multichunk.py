#!/usr/bin/env python3
"""Fine-tune the Phase-1 dense decoder on three judgment sections."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

import numpy as np
import torch
from torch import nn
from tqdm import tqdm

from phase1.src.model.legal_lm import LegalLM
from phase2.train_structured_ner import parent_config

from .data import ROOT, cases, validate_prediction
from .extract_dense import PARENT, TOKENIZER
from .train_lorma import scores


class ThreeChunkCourt(nn.Module):
    def __init__(self, backbone: LegalLM):
        super().__init__()
        self.backbone = backbone
        hidden = backbone.config.d_model
        self.head = nn.Sequential(nn.LayerNorm(3*hidden), nn.Dropout(.15),
                                  nn.Linear(3*hidden, 256), nn.GELU(),
                                  nn.Dropout(.15), nn.Linear(256, 1))

    def forward(self, ids: torch.Tensor, end: torch.Tensor) -> torch.Tensor:
        batch, chunks, width = ids.shape
        flat = ids.reshape(batch*chunks, width)
        hidden = self.backbone.embedding_dropout(self.backbone.token_embedding(flat))
        position = torch.arange(width, device=ids.device).expand(batch*chunks, -1)
        for block in self.backbone.blocks:
            hidden, _ = block(hidden, position)
        hidden = self.backbone.final_norm(hidden)
        selected = hidden[torch.arange(batch*chunks, device=ids.device), end.reshape(-1).long()]
        return self.head(selected.reshape(batch, chunks*selected.shape[-1]).float()).squeeze(-1)


def predict(model: ThreeChunkCourt, ids: torch.Tensor, end: torch.Tensor,
            batch_size: int) -> np.ndarray:
    model.eval()
    values = []
    with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16):
        for start in range(0, len(ids), batch_size):
            values.append(model(ids[start:start+batch_size],
                                end[start:start+batch_size]).float().sigmoid().cpu())
    return torch.cat(values).numpy()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--id', default='P4_E32_dense_three_chunk_full')
    parser.add_argument('--gpu', type=int, default=1)
    parser.add_argument('--batch-size', type=int, default=6)
    parser.add_argument('--epochs', type=int, default=1)
    parser.add_argument('--lr', type=float, default=2e-5)
    parser.add_argument('--min-chars', type=int, default=0)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--initial-run', type=Path,
                        help='Phase-4 checkpoint descended from the Phase-1 dense model')
    args = parser.parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.set_num_threads(8)
    torch.backends.cuda.matmul.allow_tf32 = True
    device = torch.device(f'cuda:{args.gpu}')
    token_dir = ROOT/'tokens/three_chunks_1024'
    rows = {split: cases(split) for split in ('train', 'dev', 'test')}
    arrays = {}
    for split, examples in rows.items():
        expected = [r['id'] for r in examples]
        if json.loads((token_dir/f'{split}_ids.json').read_text()) != expected:
            raise RuntimeError(f'{split} token IDs mismatch')
        arrays[split] = (torch.as_tensor(np.load(token_dir/f'{split}_tokens.npy'),
                                              device=device, dtype=torch.long),
                         torch.as_tensor(np.load(token_dir/f'{split}_positions.npy'),
                                         device=device, dtype=torch.long))
    y_train = np.array([r['label'] for r in rows['train']])
    y_dev = np.array([r['label'] for r in rows['dev']])
    chosen = np.array([i for i,r in enumerate(rows['train'])
                       if int(r['id'][:4]) < 2000 and len(r['text']) >= args.min_chars])
    index = torch.as_tensor(chosen, device=device)
    labels = torch.as_tensor(y_train, device=device, dtype=torch.float32)
    positive = int(y_train[chosen].sum())
    balance = torch.tensor((len(chosen)-positive)/positive, device=device)
    parent, kind, config = parent_config(PARENT, TOKENIZER)
    if kind != 'dense':
        raise RuntimeError('The selected Phase-1 parent is not dense LegalLM')
    backbone = LegalLM(config)
    backbone.load_state_dict(parent['model'], strict=True)
    del parent
    if args.initial_run is not None:
        previous = torch.load(args.initial_run, map_location='cpu', weights_only=False)
        adapted = {name.removeprefix('backbone.'): value
                   for name,value in previous['trainable'].items()
                   if name.startswith('backbone.')}
        if set(adapted) != set(backbone.state_dict()):
            raise RuntimeError('Initial run does not contain a full compatible backbone')
        backbone.load_state_dict(adapted, strict=True)
    model = ThreeChunkCourt(backbone).to(device)
    body = list(model.backbone.parameters())
    head = list(model.head.parameters())
    optimizer = torch.optim.AdamW([{'params': body, 'lr': args.lr},
                                   {'params': head, 'lr': args.lr*50}], weight_decay=.01)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=balance)
    total_steps = args.epochs*math.ceil(len(index)/args.batch_size)
    warmup = min(100, total_steps//10)
    run = ROOT/'runs'/args.id
    submission = ROOT/'submissions'/args.id
    if run.exists() or submission.exists():
        raise RuntimeError('Experiment ID exists')
    run.mkdir(parents=True)
    started = time.monotonic()
    logs = []
    best = None
    step = 0
    for epoch in range(1, args.epochs+1):
        model.train()
        order = index[torch.randperm(len(index), device=device)]
        losses = []
        for start in tqdm(range(0,len(order),args.batch_size), desc=f'{args.id} epoch {epoch}', unit='step'):
            ids = order[start:start+args.batch_size]
            step += 1
            factor = (step/max(warmup,1) if step <= warmup
                      else .1+.9*(1+math.cos(math.pi*(step-warmup)/max(total_steps-warmup,1)))/2)
            optimizer.param_groups[0]['lr'] = args.lr*factor
            optimizer.param_groups[1]['lr'] = args.lr*50*factor
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast('cuda', dtype=torch.bfloat16):
                logits = model(arrays['train'][0][ids], arrays['train'][1][ids])
                loss = loss_fn(logits.float(), labels[ids])
            loss.backward()
            nn.utils.clip_grad_norm_(body+head, 1.)
            optimizer.step()
            losses.append(float(loss.detach()))
        dev_prob = predict(model, *arrays['dev'], batch_size=args.batch_size*2)
        record = {'epoch': epoch, 'train_loss': round(float(np.mean(losses)),6),
                  **scores(y_dev,dev_prob),
                  'elapsed_seconds': round(time.monotonic()-started,1),
                  'gpu_peak_gib': round(torch.cuda.max_memory_allocated(device)/1024**3,2)}
        logs.append(record)
        (run/'metrics.jsonl').write_text('\n'.join(json.dumps(x) for x in logs)+'\n')
        print(json.dumps(record), flush=True)
        if best is None or record['macro_f1'] > best['macro_f1']:
            best = record
            test_prob = predict(model, *arrays['test'], batch_size=args.batch_size*2)
            np.savez_compressed(run/'probabilities.npz',
                                dev=dev_prob.astype(np.float32), test=test_prob.astype(np.float32),
                                dev_ids=np.array([r['id'] for r in rows['dev']]),
                                test_ids=np.array([r['id'] for r in rows['test']]))
            torch.save({'trainable': {name:p.detach().cpu() for name,p in model.named_parameters()},
                        'parent': str(PARENT), 'config': vars(args), 'dev_metrics': best}, run/'best.pt')
    saved = np.load(run/'probabilities.npz')
    prediction = [{'id': row['id'], 'pred': int(prob>=best['threshold'])}
                  for row,prob in zip(rows['test'],saved['test'])]
    validate_prediction(prediction)
    submission.mkdir(parents=True)
    json_path = submission/'prediction.json'
    json_path.write_text(json.dumps(prediction,indent=2)+'\n')
    zip_path = submission/'submission.zip'
    with ZipFile(zip_path,'w',ZIP_DEFLATED,compresslevel=9) as archive:
        archive.write(json_path,'prediction.json')
    (submission/'manifest.json').write_text(json.dumps({
        'id':args.id,'method':'three_chunk_full','parent':str(PARENT),
        'dev_metrics':best,'train_cases':len(chosen),'test_cases':len(prediction),
        'prediction_sha256':hashlib.sha256(json_path.read_bytes()).hexdigest(),
        'zip_sha256':hashlib.sha256(zip_path.read_bytes()).hexdigest()},indent=2)+'\n')
    print(json.dumps({'best':best,'submission':str(zip_path)},indent=2),flush=True)


if __name__ == '__main__':
    main()
