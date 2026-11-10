#!/usr/bin/env python3
"""Cache head/end representations from the frozen Phase-1 dense decoder."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
import time

import numpy as np
import torch
from tokenizers import Tokenizer
from tqdm import tqdm

from .data import ROOT, cases
from phase2.train_structured_ner import parent_config
from phase1.src.model.legal_lm import LegalLM


TOKENIZER = ROOT.parent/'phase1/artifacts/tokenizers_512m_32k/bpe/tokenizer.json'
PARENT = ROOT.parent/'phase1/runs/A3_legal_full_20261005/checkpoints/best/state.pt'


def selected_ids(text: str, tokenizer: Tokenizer, head: int, tail: int, eos: int):
    # Limit tokenization cost on extreme documents. Character windows are
    # substantially wider than the retained token windows.
    if len(text) <= 12000:
        tokens = tokenizer.encode(text, add_special_tokens=False).ids
        if len(tokens) <= head+tail:
            ids = tokens + [eos]
            split = min(head, len(tokens))
        else:
            ids = tokens[:head]+[eos]+tokens[-tail:]+[eos]
            split = head
    else:
        first = tokenizer.encode(text[:max(3500, head*16)],
                                 add_special_tokens=False).ids[:head]
        last = tokenizer.encode(text[-max(7500, tail*16):],
                                add_special_tokens=False).ids[-tail:]
        ids = first+[eos]+last+[eos]
        split = len(first)
    if not ids:
        ids = [eos]
    return ids, split


def main() -> None:
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--gpu',type=int,default=1)
    ap.add_argument('--batch-size',type=int,default=24)
    ap.add_argument('--head-tokens',type=int,default=192)
    ap.add_argument('--tail-tokens',type=int,default=448)
    ap.add_argument('--splits',nargs='+',choices=['train','dev','test'],
                    default=['train','dev','test'])
    ap.add_argument('--output-dir',type=Path,default=ROOT/'features/dense_head192_tail448')
    args=ap.parse_args()
    if args.batch_size<1 or args.head_tokens<1 or args.tail_tokens<1:
        ap.error('Batch and window sizes must be positive')
    random.seed(42);torch.manual_seed(42)
    torch.set_num_threads(8)
    torch.backends.cuda.matmul.allow_tf32=True
    device=torch.device(f'cuda:{args.gpu}')
    if not torch.cuda.is_available() or args.gpu>=torch.cuda.device_count():
        raise RuntimeError('Requested CUDA GPU unavailable')
    tokenizer=Tokenizer.from_file(str(TOKENIZER))
    eos=tokenizer.token_to_id('<eos>')
    if eos is None:raise RuntimeError('BPE tokenizer has no <eos>')
    parent,kind,config=parent_config(PARENT,TOKENIZER)
    if kind!='dense':raise RuntimeError('Wrong Phase-1 parent')
    model=LegalLM(config)
    model.load_state_dict(parent['model'],strict=True)
    del parent
    model.to(device=device,dtype=torch.bfloat16).eval()
    for p in model.parameters():p.requires_grad_(False)
    args.output_dir.mkdir(parents=True,exist_ok=True)
    metadata={'parent':str(PARENT),'tokenizer':str(TOKENIZER),
              'head_tokens':args.head_tokens,'tail_tokens':args.tail_tokens,
              'device':str(device),'batch_size':args.batch_size,
              'feature_dimension':config.d_model*2,'splits':{}}
    for split in args.splits:
        started=time.monotonic();rows=cases(split)
        ids_list=[];head_positions=[];lengths=[]
        for row in tqdm(rows,desc=f'{split} tokenize',unit='case'):
            ids,head_pos=selected_ids(row['text'],tokenizer,
                                      args.head_tokens,args.tail_tokens,eos)
            ids_list.append(ids);head_positions.append(head_pos)
            lengths.append(len(ids))
        features=np.empty((len(rows),config.d_model*2),dtype=np.float16)
        with torch.inference_mode():
            for start in tqdm(range(0,len(rows),args.batch_size),
                              desc=f'{split} encode',unit='batch'):
                stop=min(start+args.batch_size,len(rows))
                sequences=ids_list[start:stop]
                width=max(map(len,sequences))
                batch=torch.full((len(sequences),width),eos,dtype=torch.long,device=device)
                for offset,seq in enumerate(sequences):
                    batch[offset,:len(seq)]=torch.tensor(seq,device=device)
                with torch.autocast('cuda',dtype=torch.bfloat16):
                    positions=torch.arange(width,device=device).expand(len(sequences),-1)
                    hidden=model.embedding_dropout(model.token_embedding(batch))
                    for block in model.blocks:hidden,_=block(hidden,positions)
                    hidden=model.final_norm(hidden)
                indices=torch.arange(len(sequences),device=device)
                a=hidden[indices,torch.tensor(head_positions[start:stop],device=device)]
                b=hidden[indices,torch.tensor(lengths[start:stop],device=device)-1]
                features[start:stop]=torch.cat((a,b),dim=1).float().cpu().numpy().astype(np.float16)
        np.save(args.output_dir/f'{split}.npy',features)
        (args.output_dir/f'{split}_ids.json').write_text(json.dumps([r['id'] for r in rows]))
        if split!='test':
            np.save(args.output_dir/f'{split}_labels.npy',
                    np.asarray([r['label'] for r in rows],dtype=np.uint8))
        metadata['splits'][split]={'examples':len(rows),'seconds':round(time.monotonic()-started,2),
                                   'median_tokens':int(np.median(lengths)),
                                   'truncated_cases':sum(n>args.head_tokens+args.tail_tokens+2 for n in lengths)}
        (args.output_dir/'manifest.json').write_text(json.dumps(metadata,indent=2)+'\n')
        del features,rows,ids_list
    print(json.dumps(metadata,indent=2))


if __name__=='__main__':main()
