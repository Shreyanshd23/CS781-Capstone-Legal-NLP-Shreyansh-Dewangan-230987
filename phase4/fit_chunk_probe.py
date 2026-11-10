#!/usr/bin/env python3
"""Fit small classifiers on frozen beginning/middle/end LegalLM features."""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path
from zipfile import ZIP_DEFLATED,ZipFile

import numpy as np
import torch
from torch import nn
from sklearn.metrics import f1_score

from .data import ROOT,cases,validate_prediction


def main() -> None:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--features',type=Path,default=ROOT/'runs/chunk_features_e23')
    parser.add_argument('--id',default='P4_E42_frozen_e23_multiview_probe')
    parser.add_argument('--gpu',type=int,default=1)
    parser.add_argument('--hidden',type=int,default=256)
    parser.add_argument('--epochs',type=int,default=35)
    parser.add_argument('--seed',type=int,default=42)
    args=parser.parse_args()
    torch.manual_seed(args.seed);np.random.seed(args.seed)
    torch.set_num_threads(8)
    device=torch.device(f'cuda:{args.gpu}')
    arrays={}
    rows={split:cases(split) for split in ('train','dev','test')}
    old=[r for r in rows['train'] if int(r['id'][:4])<2000]
    expected={'train':old,'dev':rows['dev'],'test':rows['test']}
    for split,examples in expected.items():
        if json.loads((args.features/f'{split}_ids.json').read_text())!=[r['id'] for r in examples]:
            raise RuntimeError(f'{split} features have wrong IDs')
        data=np.load(args.features/f'{split}.npy').astype(np.float32)
        arrays[split]=data.reshape(len(data),-1)
    mean=arrays['train'].mean(0)
    std=arrays['train'].std(0).clip(min=.1)
    for split in arrays:arrays[split]=torch.as_tensor(
        ((arrays[split]-mean)/std).clip(-8,8),device=device)
    y=np.array([r['label'] for r in old],dtype=np.float32)
    yd=np.array([r['label'] for r in rows['dev']],dtype=np.uint8)
    labels=torch.as_tensor(y,device=device)
    balance=torch.tensor((len(y)-y.sum())/y.sum(),device=device)
    run=ROOT/'runs'/args.id
    submission=ROOT/'submissions'/args.id
    if run.exists() or submission.exists():raise RuntimeError('Experiment ID exists')
    run.mkdir(parents=True)
    started=time.monotonic()
    models={
        'linear':nn.Sequential(nn.Dropout(.15),nn.Linear(arrays['train'].shape[1],1)),
        'nonlinear':nn.Sequential(nn.LayerNorm(arrays['train'].shape[1]),nn.Dropout(.2),
                                  nn.Linear(arrays['train'].shape[1],args.hidden),nn.GELU(),
                                  nn.Dropout(.3),nn.Linear(args.hidden,1))}
    records=[];best=None
    for name,model in models.items():
        model=model.to(device)
        optimizer=torch.optim.AdamW(model.parameters(),lr=3e-4 if name=='nonlinear' else 1e-3,
                                     weight_decay=.05)
        loss_fn=nn.BCEWithLogitsLoss(pos_weight=balance)
        for epoch in range(1,args.epochs+1):
            model.train();permutation=torch.randperm(len(y),device=device)
            loss_values=[]
            for indices in permutation.split(256):
                optimizer.zero_grad(set_to_none=True)
                logits=model(arrays['train'][indices]).squeeze(-1)
                loss=loss_fn(logits,labels[indices])
                loss.backward();optimizer.step();loss_values.append(float(loss.detach()))
            model.eval()
            with torch.inference_mode():
                dev_prob=model(arrays['dev']).squeeze(-1).sigmoid().cpu().numpy()
            candidate=max(({'threshold':round(float(t),3),
                            'macro_f1':float(f1_score(yd,dev_prob>=t,average='macro'))}
                           for t in np.arange(.30,.701,.01)),key=lambda r:r['macro_f1'])
            record={'model':name,'epoch':epoch,'train_loss':round(float(np.mean(loss_values)),6),
                    **candidate,'elapsed_seconds':round(time.monotonic()-started,1)}
            records.append(record)
            if best is None or record['macro_f1']>best['macro_f1']:
                best=record
                with torch.inference_mode():
                    test_prob=model(arrays['test']).squeeze(-1).sigmoid().cpu().numpy()
                np.savez_compressed(run/'probabilities.npz',dev=dev_prob.astype(np.float32),
                                    test=test_prob.astype(np.float32),
                                    dev_ids=np.array([r['id'] for r in rows['dev']]),
                                    test_ids=np.array([r['id'] for r in rows['test']]))
                torch.save({'head':model.state_dict(),'mean':mean,'std':std,'args':vars(args),
                            'best':best},run/'best.pt')
            if epoch%5==0 or epoch==1:print(json.dumps(record),flush=True)
        del model
    (run/'metrics.jsonl').write_text('\n'.join(json.dumps(x) for x in records)+'\n')
    prob=np.load(run/'probabilities.npz')['test']
    prediction=[{'id':r['id'],'pred':int(p>=best['threshold'])}
                for r,p in zip(rows['test'],prob)]
    validate_prediction(prediction)
    submission.mkdir(parents=True)
    json_path=submission/'prediction.json'
    json_path.write_text(json.dumps(prediction,indent=2)+'\n')
    zip_path=submission/'submission.zip'
    with ZipFile(zip_path,'w',ZIP_DEFLATED,compresslevel=9) as archive:
        archive.write(json_path,'prediction.json')
    (submission/'manifest.json').write_text(json.dumps({'best':best,'train_cases':len(y),
        'features':str(args.features),'prediction_sha256':hashlib.sha256(json_path.read_bytes()).hexdigest(),
        'zip_sha256':hashlib.sha256(zip_path.read_bytes()).hexdigest()},indent=2)+'\n')
    print(json.dumps({'best':best,'zip':str(zip_path)},indent=2),flush=True)


if __name__=='__main__':main()
