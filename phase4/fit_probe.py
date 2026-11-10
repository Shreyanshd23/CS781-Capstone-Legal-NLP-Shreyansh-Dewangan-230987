#!/usr/bin/env python3
"""Fit a binary head on frozen Phase-1 decoder features and package CJPE predictions."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import random
import time
from zipfile import ZipFile, ZIP_DEFLATED

import numpy as np
import torch
from torch import nn
from sklearn.metrics import f1_score, precision_score, recall_score
from tqdm import tqdm

from .data import ROOT, cases, validate_prediction


def checksum(path: Path) -> str:
    h=hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(4*1024*1024),b''):
            h.update(block)
    return h.hexdigest()


def metrics(y: np.ndarray, prob: np.ndarray):
    rows=[]
    for threshold in np.arange(.3,.701,.01):
        pred=(prob>=threshold).astype(np.uint8)
        rows.append({'threshold':round(float(threshold),3),
                     'macro_f1':float(f1_score(y,pred,average='macro')),
                     'macro_precision':float(precision_score(y,pred,average='macro',zero_division=0)),
                     'macro_recall':float(recall_score(y,pred,average='macro',zero_division=0))})
    return max(rows,key=lambda item:(item['macro_f1'],-abs(item['threshold']-.5)))


class Head(nn.Module):
    def __init__(self,width:int):
        super().__init__()
        self.network=nn.Sequential(nn.LayerNorm(width),nn.Dropout(.2),
            nn.Linear(width,256),nn.GELU(),nn.Dropout(.3),nn.Linear(256,1))

    def forward(self,x):return self.network(x).squeeze(-1)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--feature-dir',type=Path,
                        default=ROOT/'features/dense_head192_tail448')
    parser.add_argument('--id',default='P4_E1_dense_frozen_head_tail')
    parser.add_argument('--gpu',type=int,default=1)
    parser.add_argument('--epochs',type=int,default=30)
    parser.add_argument('--batch-size',type=int,default=512)
    parser.add_argument('--seed',type=int,default=42)
    args=parser.parse_args()
    random.seed(args.seed);np.random.seed(args.seed);torch.manual_seed(args.seed)
    torch.set_num_threads(8)
    torch.backends.cuda.matmul.allow_tf32=True
    device=torch.device(f'cuda:{args.gpu}')
    path=args.feature_dir
    rows={split:cases(split) for split in ('train','dev','test')}
    arrays={split:np.load(path/f'{split}.npy').astype(np.float32)
            for split in rows}
    for split in rows:
        ids=json.loads((path/f'{split}_ids.json').read_text())
        if ids!=[row['id'] for row in rows[split]] or len(arrays[split])!=len(ids):
            raise RuntimeError(f'{split} feature alignment failed')
    mean=arrays['train'].mean(axis=0);std=arrays['train'].std(axis=0)
    std=np.maximum(std,.05)
    x={split:torch.from_numpy((value-mean)/std).to(device)
       for split,value in arrays.items()}
    y_train=np.asarray([row['label'] for row in rows['train']],dtype=np.uint8)
    y_dev=np.asarray([row['label'] for row in rows['dev']],dtype=np.uint8)
    y=torch.from_numpy(y_train.astype(np.float32)).to(device)
    model=Head(x['train'].shape[1]).to(device)
    optimizer=torch.optim.AdamW(model.parameters(),lr=1e-3,weight_decay=.02)
    positive_weight=torch.tensor(float((y_train==0).sum()/(y_train==1).sum()),device=device)
    criterion=nn.BCEWithLogitsLoss(pos_weight=positive_weight)
    folder=ROOT/'runs'/args.id;folder.mkdir(parents=True,exist_ok=True)
    submission=ROOT/'submissions'/args.id
    if submission.exists():raise RuntimeError(f'Submission already exists: {submission}')
    logs=[];best=None;start=time.monotonic();patience=0
    for epoch in range(1,args.epochs+1):
        model.train();order=torch.randperm(len(y),device=device)
        losses=[]
        for offset in tqdm(range(0,len(y),args.batch_size),desc=f'epoch {epoch}',unit='batch'):
            index=order[offset:offset+args.batch_size]
            optimizer.zero_grad(set_to_none=True)
            logit=model(x['train'][index])
            loss=criterion(logit,y[index]);loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
            optimizer.step();losses.append(float(loss.detach()))
        model.eval()
        with torch.inference_mode():
            p_dev=model(x['dev']).sigmoid().cpu().numpy()
            p_test=model(x['test']).sigmoid().cpu().numpy()
        result=metrics(y_dev,p_dev)
        record={'epoch':epoch,'train_loss':round(float(np.mean(losses)),6),
                **result,'elapsed_seconds':round(time.monotonic()-start,2),
                'gpu_peak_gib':round(torch.cuda.max_memory_allocated(device)/1024**3,2)}
        logs.append(record);print(json.dumps(record),flush=True)
        if best is None or result['macro_f1']>best['macro_f1']+1e-5:
            best={**record};patience=0
            np.savez_compressed(folder/'probabilities.npz',
                dev=p_dev.astype(np.float32),test=p_test.astype(np.float32),
                dev_ids=np.array([row['id'] for row in rows['dev']]),
                test_ids=np.array([row['id'] for row in rows['test']]))
            torch.save({'model':model.state_dict(),'mean':mean,'std':std,
                        'feature_manifest':json.loads((path/'manifest.json').read_text()),
                        'metrics':best},folder/'best.pt')
        else:
            patience+=1
        (folder/'metrics.jsonl').write_text('\n'.join(json.dumps(r) for r in logs)+'\n')
        if patience>=7:break
    saved=np.load(folder/'probabilities.npz')
    probabilities=saved['test'];threshold=best['threshold']
    prediction=[{'id':row['id'],'pred':int(prob>=threshold)}
                for row,prob in zip(rows['test'],probabilities)]
    validate_prediction(prediction)
    submission.mkdir(parents=True)
    json_path=submission/'prediction.json'
    json_path.write_text(json.dumps(prediction,ensure_ascii=False,indent=2)+'\n')
    zip_path=submission/'submission.zip'
    with ZipFile(zip_path,'w',ZIP_DEFLATED,compresslevel=9) as archive:
        archive.write(json_path,'prediction.json')
    with ZipFile(zip_path) as archive:
        if archive.namelist()!=['prediction.json'] or archive.testzip() is not None:
            raise RuntimeError('Invalid submission ZIP')
    (submission/'manifest.json').write_text(json.dumps({
        'id':args.id,'phase1_parent':str(json.loads((path/'manifest.json').read_text())['parent']),
        'method':'frozen_phase1_dense_head_tail_probe',
        'dev_metrics':best,'test_cases':len(prediction),
        'prediction_sha256':checksum(json_path),'zip_sha256':checksum(zip_path)},indent=2)+'\n')
    print(json.dumps({'best':best,'submission':str(zip_path)},indent=2),flush=True)


if __name__=='__main__':main()
