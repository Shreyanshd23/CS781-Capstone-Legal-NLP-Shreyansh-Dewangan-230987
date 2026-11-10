#!/usr/bin/env python3
"""Cache fixed head/tail BPE windows without using test labels."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
from tokenizers import Tokenizer
from tqdm import tqdm

from .data import ROOT, cases
from .extract_dense import TOKENIZER, selected_ids


def selected_ids_no_overlap(text,tokenizer,head,tail,eos):
    """Keep the full document when it fits and avoid overlapping head/tail slices."""
    if len(text)<=100000:
        tokens=tokenizer.encode(text,add_special_tokens=False).ids
        if len(tokens)<=head+tail:
            return tokens+[eos],min(head,len(tokens))
        return tokens[:head]+[eos]+tokens[-tail:]+[eos],head
    first_chars=min(max(3500,head*16),len(text)//2)
    last_chars=min(max(7500,tail*16),len(text)-first_chars)
    first=tokenizer.encode(text[:first_chars],add_special_tokens=False).ids[:head]
    last=tokenizer.encode(text[-last_chars:],add_special_tokens=False).ids[-tail:]
    return first+[eos]+last+[eos],len(first)


def restore_substitutions(text: str) -> str:
    """Undo the corpus-wide literal replacements of 'co' and 'no'."""
    return text.replace('company', 'co').replace('number', 'no')


def main() -> None:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--tokenizer',type=Path,default=TOKENIZER)
    parser.add_argument('--output-dir',type=Path,
                        default=ROOT/'tokens/head128_tail384')
    parser.add_argument('--head',type=int,default=128)
    parser.add_argument('--tail',type=int,default=384)
    parser.add_argument('--no-overlap',action='store_true')
    parser.add_argument('--restore-substitutions',action='store_true',
                        help="Undo literal company→co and number→no replacements")
    args=parser.parse_args()
    if args.head<1 or args.tail<1:parser.error('Head/tail lengths must be positive')
    started=time.monotonic()
    tokenizer=Tokenizer.from_file(str(args.tokenizer))
    eos=tokenizer.token_to_id('<eos>')
    if eos is None:raise RuntimeError('No EOS token')
    head,tail=args.head,args.tail
    width=head+tail+2
    folder=args.output_dir;folder.mkdir(parents=True,exist_ok=True)
    manifest={'tokenizer':str(args.tokenizer),'head':head,'tail':tail,'width':width,
              'selection':'no_overlap_full_if_100k' if args.no_overlap else 'original',
              'restore_substitutions':args.restore_substitutions,
              'splits':{}}
    for split in ('train','dev','test'):
        rows=cases(split)
        ids=np.full((len(rows),width),eos,dtype=np.int32)
        positions=np.zeros((len(rows),2),dtype=np.int16)
        for index,row in enumerate(tqdm(rows,desc=f'{split} tokenize',unit='doc')):
            selector=selected_ids_no_overlap if args.no_overlap else selected_ids
            source=(restore_substitutions(row['text']) if args.restore_substitutions
                    else row['text'])
            seq,first=selector(source,tokenizer,head,tail,eos)
            if len(seq)>width:raise RuntimeError('Window exceeds width')
            ids[index,:len(seq)]=seq
            positions[index]=first,len(seq)-1
        np.save(folder/f'{split}_tokens.npy',ids)
        np.save(folder/f'{split}_positions.npy',positions)
        if split!='test':np.save(folder/f'{split}_labels.npy',
                                 np.array([row['label'] for row in rows],dtype=np.uint8))
        (folder/f'{split}_ids.json').write_text(json.dumps([row['id'] for row in rows]))
        manifest['splits'][split]={'cases':len(rows)}
        (folder/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    manifest['seconds']=round(time.monotonic()-started,2)
    (folder/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps(manifest,indent=2),flush=True)


if __name__=='__main__':main()
