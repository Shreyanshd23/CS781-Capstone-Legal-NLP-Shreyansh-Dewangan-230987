#!/usr/bin/env python3
"""Adapt the Phase-1 dense LegalLM for court outcomes."""
from __future__ import annotations

import argparse
from dataclasses import replace
import hashlib
import json
import math
import time
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

import numpy as np
import torch
from torch import nn
from sklearn.metrics import f1_score, precision_score, recall_score
from tqdm import tqdm

from .data import ROOT, cases, validate_prediction
from .extract_dense import PARENT, TOKENIZER
from phase2.ner_core import LowRankLinear, AdapterFFN
from phase2.ner_core import DecoderNER
from phase2.train_structured_ner import parent_config
from phase1.src.model.legal_lm import LegalLM
from phase1.src.model.legal_moe import LegalMoELM
from phase1.src.model.rmsnorm import RMSNorm

UL2_TOKENIZER=ROOT.parent/'phase1/data/legal_ul2_full_20261006/tokenizer.json'


class CourtModel(nn.Module):
    def __init__(self, backbone: nn.Module, method: str, kind: str,
                 pooling: str = 'first_last', bidirectional: bool = False,
                 position_scale: float = 1.0):
        super().__init__()
        if method not in ('lorma_plus','lora','partial','norm','adapter','full','encoder_full','distillation'):
            raise ValueError(method)
        self.backbone=backbone
        self.kind=kind
        self.pooling=pooling
        self.bidirectional=bidirectional
        if position_scale < 1:
            raise ValueError('RoPE position scale must be at least one')
        if pooling not in ('first_last','attention'):
            raise ValueError(f'Unknown pooling: {pooling}')
        if bidirectional:
            if kind not in ('dense','moe'):
                raise ValueError('Bidirectional mode requires a LegalLM backbone')
            for block in self.backbone.blocks:block.attn.bidirectional=True
        if position_scale != 1:
            if kind not in ('dense','moe'):
                raise ValueError('RoPE interpolation requires a LegalLM backbone')
            for block in self.backbone.blocks:
                rope=block.attn.rope
                length=rope.cos_cache.shape[0]
                dimension=block.attn.head_dim
                inv_freq=1.0/(backbone.config.rope_theta **
                              (torch.arange(0,dimension,2).float()/dimension))
                phases=torch.outer(torch.arange(length).float()/position_scale,
                                   inv_freq)
                rope.cos_cache=phases.cos()
                rope.sin_cache=phases.sin()
        width=backbone.config.n_embd if kind=='gpt2' else backbone.config.d_model
        if method not in ('full','distillation'):self.backbone.requires_grad_(False)
        if method in ('lorma_plus','lora'):
            if kind=='gpt2':raise ValueError('GPT2 LoRMA/LoRA not configured')
            for block in self.backbone.blocks:
                for name in ('q_proj','v_proj'):
                    setattr(block.attn,name,LowRankLinear(getattr(block.attn,name),
                                                           method,rank=8,alpha=16))
                if all(hasattr(block.ffn,name) for name in ('gate','up','down')):
                    for name in ('gate','up','down'):
                        setattr(block.ffn,name,LowRankLinear(getattr(block.ffn,name),
                                                              method,rank=8,alpha=16))
        elif method=='partial':
            if kind=='gpt2':
                for block in self.backbone.transformer.h[-4:]:block.requires_grad_(True)
                self.backbone.transformer.ln_f.requires_grad_(True)
            else:
                for block in self.backbone.blocks[-4:]:block.requires_grad_(True)
                self.backbone.final_norm.requires_grad_(True)
        elif method=='norm':
            for module in self.backbone.modules():
                if isinstance(module,RMSNorm):module.requires_grad_(True)
        elif method=='adapter':
            if kind not in ('dense','moe'):raise ValueError('Adapter requires LegalLM blocks')
            for block in self.backbone.blocks:
                block.ffn=AdapterFFN(block.ffn,width,bottleneck=64)
        elif method=='encoder_full':
            if kind!='ul2':raise ValueError('Encoder-only adaptation requires UL2')
            self.backbone.encoder.requires_grad_(True)
            self.backbone.shared.requires_grad_(True)
        if pooling=='attention':self.pool_score=nn.Linear(width,1,bias=False)
        input_width=width*(3 if pooling=='attention' else 2)
        self.head=nn.Sequential(nn.LayerNorm(input_width),nn.Dropout(.15),
                                nn.Linear(input_width,256),nn.GELU(),
                                nn.Dropout(.15),nn.Linear(256,1))

    def forward(self,ids:torch.Tensor,places:torch.Tensor)->torch.Tensor:
        batch,width=ids.shape
        if self.kind=='gpt2':
            hidden=self.backbone.transformer(input_ids=ids,use_cache=False,
                                             return_dict=True).last_hidden_state
        elif self.kind=='ul2':
            valid=torch.arange(width,device=ids.device)[None,:] <= places[:,1,None]
            hidden=self.backbone.encoder(input_ids=ids,attention_mask=valid,
                                         return_dict=True).last_hidden_state
        else:
            positions=torch.arange(width,device=ids.device).expand(batch,-1)
            hidden=self.backbone.embedding_dropout(self.backbone.token_embedding(ids))
            for block in self.backbone.blocks:hidden,_=block(hidden,positions)
            hidden=self.backbone.final_norm(hidden)
        batch_index=torch.arange(batch,device=ids.device)
        if self.kind=='ul2':
            valid=(torch.arange(width,device=ids.device)[None,:] <= places[:,1,None])
            first=(hidden*valid[:,:,None]).sum(dim=1)/valid.sum(dim=1)[:,None]
        else:first=hidden[batch_index,places[:,0].long()]
        last=hidden[batch_index,places[:,1].long()]
        features=[first,last]
        if self.pooling=='attention':
            valid=torch.arange(width,device=ids.device)[None,:] <= places[:,1,None]
            weight=self.pool_score(hidden.float()).squeeze(-1)
            weight=weight.masked_fill(~valid,float('-inf')).softmax(dim=1)
            pooled=torch.bmm(weight[:,None,:],hidden.float()).squeeze(1)
            features.append(pooled)
        return self.head(torch.cat(features,dim=-1).float()).squeeze(-1)


def scores(y,p):
    options=[]
    for threshold in np.arange(.35,.651,.01):
        decision=(p>=threshold).astype(int)
        options.append({'threshold':round(float(threshold),3),
                        'macro_f1':float(f1_score(y,decision,average='macro')),
                        'macro_precision':float(precision_score(y,decision,average='macro')),
                        'macro_recall':float(recall_score(y,decision,average='macro'))})
    return max(options,key=lambda r:r['macro_f1'])


def predict(model,ids,places,batch_size):
    model.eval();output=[]
    with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
        for start in range(0,len(ids),batch_size):
            stop=min(start+batch_size,len(ids))
            output.append(model(ids[start:stop],places[start:stop]).float().sigmoid().cpu())
    return torch.cat(output).numpy()


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--gpu',type=int,default=2)
    ap.add_argument('--epochs',type=int,default=3)
    ap.add_argument('--batch-size',type=int,default=16)
    ap.add_argument('--lr',type=float,default=2e-4)
    ap.add_argument('--seed',type=int,default=42)
    ap.add_argument('--head-multiplier',type=float,default=50.)
    ap.add_argument('--pooling',choices=['first_last','attention'],default='first_last')
    ap.add_argument('--bidirectional',action='store_true')
    ap.add_argument('--extend-context',action='store_true',
                    help='Extend LegalLM RoPE cache to the token-cache width')
    ap.add_argument('--position-scale',type=float,default=1.0,
                    help='Divide RoPE positions by this factor for interpolation')
    ap.add_argument('--method',choices=['lorma_plus','lora','partial','norm','adapter','full','encoder_full','distillation'],
                    default='lorma_plus')
    ap.add_argument('--id',default='P4_E4_dense_lorma_era_matched')
    ap.add_argument('--parent',type=Path,default=PARENT)
    ap.add_argument('--tokenizer',type=Path,default=TOKENIZER)
    ap.add_argument('--token-dir',type=Path,
                    default=ROOT/'tokens/head128_tail384')
    ap.add_argument('--train-era',choices=['pre2000','all'],default='pre2000')
    ap.add_argument('--max-year',type=int,
                    help='Optional exclusive case-year cutoff within the chosen training era')
    ap.add_argument('--old-era-weight',type=float,default=1.0,
                    help='Training loss weight for cases before 2000 when using all eras')
    ap.add_argument('--min-chars',type=int,default=0)
    ap.add_argument('--teacher-probabilities',type=Path)
    ap.add_argument('--initial-run',type=Path,
                    help='Continue a compatible Phase-1-descended Phase-4 checkpoint')
    args=ap.parse_args()
    if args.old_era_weight < 1:
        ap.error('--old-era-weight must be at least one')
    if args.method=='distillation' and args.teacher_probabilities is None:
        ap.error('Distillation requires --teacher-probabilities')
    torch.manual_seed(args.seed);np.random.seed(args.seed)
    torch.set_num_threads(8)
    torch.backends.cuda.matmul.allow_tf32=True
    device=torch.device(f'cuda:{args.gpu}')
    token_dir=args.token_dir
    rows={split:cases(split) for split in ('train','dev','test')}
    arrays={}
    for split in rows:
        if json.loads((token_dir/f'{split}_ids.json').read_text())!=[r['id'] for r in rows[split]]:
            raise RuntimeError(f'{split} token alignment mismatch')
        arrays[split]=(torch.as_tensor(np.load(token_dir/f'{split}_tokens.npy'),
                                       device=device,dtype=torch.long),
                       torch.as_tensor(np.load(token_dir/f'{split}_positions.npy'),
                                       device=device,dtype=torch.long))
    y_train=np.array([r['label'] for r in rows['train']])
    y_dev=np.array([r['label'] for r in rows['dev']])
    train_index=np.array([i for i,r in enumerate(rows['train'])
                          if (args.train_era=='all' or int(r['id'][:4])<2000)
                          and (args.max_year is None or int(r['id'][:4])<args.max_year)
                          and len(r['text'])>=args.min_chars],dtype=np.int64)
    train_index=torch.as_tensor(train_index,device=device)
    labels=torch.as_tensor(y_train,device=device,dtype=torch.float32)
    sample_weights=torch.as_tensor(
        [args.old_era_weight if int(r['id'][:4])<2000 else 1.0
         for r in rows['train']],device=device,dtype=torch.float32)
    chosen=train_index.cpu().numpy()
    selected_weights=sample_weights[train_index].cpu().numpy()
    positive=float((selected_weights*y_train[chosen]).sum())
    negative=float((selected_weights*(1-y_train[chosen])).sum())
    parent,kind,config=parent_config(args.parent,args.tokenizer)
    if kind not in ('dense','moe','gpt2','ul2'):
        raise RuntimeError('Unknown Phase-1 parent')
    if args.extend_context:
        if kind not in ('dense','moe'):
            raise ValueError('Context extension requires a LegalLM backbone')
        config=replace(config,context_length=arrays['train'][0].shape[1])
    if kind=='dense':backbone=LegalLM(config)
    elif kind=='moe':backbone=LegalMoELM(config)
    else:backbone=DecoderNER(config,kind,'full').backbone
    backbone.load_state_dict(parent['model'],strict=True)
    del parent
    backbone.to(device=device,dtype=torch.float32 if args.method in ('full','encoder_full','partial','norm','distillation')
                else torch.bfloat16)
    model=CourtModel(backbone,args.method,kind,args.pooling,args.bidirectional,
                     args.position_scale).to(device)
    if args.initial_run is not None:
        previous=torch.load(args.initial_run,map_location='cpu',weights_only=False)
        missing,unexpected=model.load_state_dict(previous['trainable'],strict=False)
        if missing or unexpected:
            raise RuntimeError(f'Initial checkpoint mismatch: missing={missing[:3]}, '
                               f'unexpected={unexpected[:3]}')
    head_params=list(model.head.parameters())
    if args.pooling=='attention':head_params+=list(model.pool_score.parameters())
    adapter_params=[p for name,p in model.named_parameters()
                    if p.requires_grad and not name.startswith(('head.','pool_score.'))]
    head_multiplier=(args.head_multiplier if args.method in ('full','encoder_full','distillation')
                     else 3)
    optimizer=torch.optim.AdamW([{'params':adapter_params,'lr':args.lr},
                                 {'params':head_params,'lr':args.lr*head_multiplier}],
                                weight_decay=.01)
    total_steps=args.epochs*math.ceil(len(train_index)/args.batch_size)
    warmup=min(100,total_steps//10)
    balance=torch.tensor(negative/max(positive,1),device=device)
    loss_fn=nn.BCEWithLogitsLoss(pos_weight=balance,reduction='none')
    teacher=None
    if args.method=='distillation':
        cache=np.load(args.teacher_probabilities)
        if not np.array_equal(cache['train_ids'],
                              np.array([r['id'] for r in rows['train']])):
            raise RuntimeError('Teacher train IDs are not aligned')
        teacher=torch.as_tensor(cache['train'],device=device,dtype=torch.float32)
        teacher=torch.logit(teacher.clamp(1e-4,1-1e-4)).div(2).sigmoid()
    run=ROOT/'runs'/args.id
    submission=ROOT/'submissions'/args.id
    if run.exists() or submission.exists():raise RuntimeError('Experiment ID exists')
    run.mkdir(parents=True)
    started=time.monotonic();global_step=0;best=None;logs=[]
    for epoch in range(1,args.epochs+1):
        model.train();order=train_index[torch.randperm(len(train_index),device=device)]
        losses=[]
        for start in tqdm(range(0,len(order),args.batch_size),
                          desc=f'{args.id} epoch {epoch}',unit='step'):
            index=order[start:start+args.batch_size]
            global_step+=1
            if global_step<=warmup:factor=global_step/max(warmup,1)
            else:factor=.1+.9*(1+math.cos(math.pi*(global_step-warmup)/
                                           max(total_steps-warmup,1)))/2
            optimizer.param_groups[0]['lr']=args.lr*factor
            optimizer.param_groups[1]['lr']=args.lr*head_multiplier*factor
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast('cuda',dtype=torch.bfloat16):
                logits=model(arrays['train'][0][index],arrays['train'][1][index])
                hard=(loss_fn(logits.float(),labels[index])*sample_weights[index]).sum()/sample_weights[index].sum()
                if teacher is None:loss=hard
                else:
                    soft=nn.functional.binary_cross_entropy_with_logits(
                        logits.float()/2,teacher[index])*4
                    loss=.6*hard+.4*soft
            loss.backward()
            nn.utils.clip_grad_norm_(adapter_params+head_params,1.)
            optimizer.step();losses.append(float(loss.detach()))
        dp=predict(model,*arrays['dev'],batch_size=args.batch_size*2)
        result=scores(y_dev,dp)
        record={'epoch':epoch,'train_loss':round(float(np.mean(losses)),6),
                **result,'elapsed_seconds':round(time.monotonic()-started,1),
                'gpu_peak_gib':round(torch.cuda.max_memory_allocated(device)/1024**3,2)}
        logs.append(record)
        (run/'metrics.jsonl').write_text('\n'.join(json.dumps(r) for r in logs)+'\n')
        print(json.dumps(record),flush=True)
        if best is None or record['macro_f1']>best['macro_f1']:
            best=record
            tp=predict(model,*arrays['test'],batch_size=args.batch_size*2)
            np.savez_compressed(run/'probabilities.npz',dev=dp.astype(np.float32),
                                test=tp.astype(np.float32),
                                dev_ids=np.array([r['id'] for r in rows['dev']]),
                                test_ids=np.array([r['id'] for r in rows['test']]))
            np.savez_compressed(run/f'probabilities_epoch{epoch}.npz',
                                dev=dp.astype(np.float32),test=tp.astype(np.float32),
                                dev_ids=np.array([r['id'] for r in rows['dev']]),
                                test_ids=np.array([r['id'] for r in rows['test']]))
            torch.save({'trainable':{name:p.detach().cpu() for name,p in model.named_parameters()
                                      if p.requires_grad},'parent':str(args.parent),
                        'config':{**vars(args),'parent':str(args.parent),
                                  'tokenizer':str(args.tokenizer),
                                  'token_dir':str(args.token_dir),
                                  'teacher_probabilities':str(args.teacher_probabilities)},
                        'dev_metrics':best},run/'best.pt')
    saved=np.load(run/'probabilities.npz')
    decision=[{'id':row['id'],'pred':int(p>=best['threshold'])}
              for row,p in zip(rows['test'],saved['test'])]
    validate_prediction(decision)
    submission.mkdir(parents=True)
    json_path=submission/'prediction.json'
    json_path.write_text(json.dumps(decision,indent=2)+'\n')
    zip_path=submission/'submission.zip'
    with ZipFile(zip_path,'w',ZIP_DEFLATED,compresslevel=9) as archive:
        archive.write(json_path,'prediction.json')
    (submission/'manifest.json').write_text(json.dumps({'id':args.id,
        'method':args.method,'parent':str(args.parent),'dev_metrics':best,
        'train_cases':len(train_index),'test_cases':len(decision),
        'prediction_sha256':hashlib.sha256(json_path.read_bytes()).hexdigest(),
        'zip_sha256':hashlib.sha256(zip_path.read_bytes()).hexdigest()},indent=2)+'\n')
    print(json.dumps({'best':best,'submission':str(zip_path)},indent=2),flush=True)


if __name__=='__main__':main()
