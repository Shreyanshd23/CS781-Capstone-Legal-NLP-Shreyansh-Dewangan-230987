#!/usr/bin/env python3
"""Extract frozen three-section representations from a Phase-4 adapted Phase-1 model."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from phase1.src.model.legal_lm import LegalLM
from phase2.train_structured_ner import parent_config

from .data import ROOT, cases
from .extract_dense import PARENT, TOKENIZER


def main() -> None:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',type=Path,default=ROOT/'runs/P4_E23_dense_full_4096_no_overlap/best.pt')
    parser.add_argument('--output',type=Path,default=ROOT/'runs/chunk_features_e23')
    parser.add_argument('--gpu',type=int,default=1)
    parser.add_argument('--batch-size',type=int,default=12,
                        help='Flattened chunks per GPU batch')
    args=parser.parse_args()
    started=time.monotonic()
    torch.set_num_threads(8)
    device=torch.device(f'cuda:{args.gpu}')
    parent,kind,config=parent_config(PARENT,TOKENIZER)
    if kind!='dense':raise RuntimeError('Expected dense Phase-1 parent')
    model=LegalLM(config)
    model.load_state_dict(parent['model'],strict=True)
    del parent
    adapted=torch.load(args.run,map_location='cpu',weights_only=False)
    weights={name.removeprefix('backbone.'):value
             for name,value in adapted['trainable'].items()
             if name.startswith('backbone.')}
    if set(weights)!=set(model.state_dict()):
        raise RuntimeError('Adapted checkpoint is not a full dense backbone')
    model.load_state_dict(weights,strict=True)
    del adapted,weights
    model.to(device=device,dtype=torch.bfloat16).eval()
    token_dir=ROOT/'tokens/three_chunks_1024'
    args.output.mkdir(parents=True,exist_ok=False)
    manifest={'parent':str(PARENT),'adapted_run':str(args.run),
              'token_dir':str(token_dir),'gpu':args.gpu,'batch_size':args.batch_size,'splits':{}}
    for split in ('train','dev','test'):
        rows=cases(split)
        if split=='train':rows=[r for r in rows if int(r['id'][:4])<2000]
        all_ids=json.loads((token_dir/f'{split}_ids.json').read_text())
        lookup={value:i for i,value in enumerate(all_ids)}
        indices=np.array([lookup[r['id']] for r in rows])
        tokens=np.load(token_dir/f'{split}_tokens.npy',mmap_mode='r')
        ends=np.load(token_dir/f'{split}_positions.npy',mmap_mode='r')
        if tokens.shape[1:]!=(3,1024):raise RuntimeError('Unexpected token shape')
        features=np.zeros((len(rows),3,config.d_model*2),dtype=np.float16)
        with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
            for start in tqdm(range(0,len(rows),max(1,args.batch_size//3)),
                              desc=f'{split} features',unit='batch'):
                take=indices[start:start+max(1,args.batch_size//3)]
                ids=torch.as_tensor(np.asarray(tokens[take]),device=device,dtype=torch.long)
                stop=torch.as_tensor(np.asarray(ends[take]),device=device,dtype=torch.long)
                b,c,w=ids.shape
                flat=ids.reshape(b*c,w)
                hidden=model.embedding_dropout(model.token_embedding(flat))
                position=torch.arange(w,device=device).expand(b*c,-1)
                for block in model.blocks:hidden,_=block(hidden,position)
                hidden=model.final_norm(hidden)
                last=hidden[torch.arange(b*c,device=device),stop.reshape(-1)]
                # Valid-token mean contributes a different view from final EOS.
                valid=torch.arange(w,device=device)[None,:]<=stop.reshape(-1,1)
                mean=(hidden.float()*valid[:,:,None]).sum(1)/valid.sum(1)[:,None]
                packed=torch.cat((last.float(),mean),-1).reshape(b,c,-1)
                features[start:start+b]=packed.cpu().numpy().astype(np.float16)
        np.save(args.output/f'{split}.npy',features)
        (args.output/f'{split}_ids.json').write_text(json.dumps([r['id'] for r in rows]))
        if split!='test':np.save(args.output/f'{split}_labels.npy',
                                 np.array([r['label'] for r in rows],dtype=np.uint8))
        manifest['splits'][split]={'cases':len(rows),'shape':list(features.shape)}
        (args.output/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    manifest['seconds']=round(time.monotonic()-started,2)
    (args.output/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps(manifest,indent=2),flush=True)


if __name__=='__main__':main()
