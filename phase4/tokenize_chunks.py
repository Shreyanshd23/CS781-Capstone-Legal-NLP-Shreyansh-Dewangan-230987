#!/usr/bin/env python3
"""Cache three document views for a hierarchical Phase-1 classifier."""
from __future__ import annotations

import json
import time

import numpy as np
from tokenizers import Tokenizer
from tqdm import tqdm

from .data import ROOT, cases
from .extract_dense import TOKENIZER


def main() -> None:
    started = time.monotonic()
    tokenizer = Tokenizer.from_file(str(TOKENIZER))
    eos = tokenizer.token_to_id('<eos>')
    if eos is None:
        raise RuntimeError('Tokenizer has no EOS token')
    width = 1024
    folder = ROOT/'tokens/three_chunks_1024'
    folder.mkdir(parents=True, exist_ok=True)
    manifest = {'tokenizer': str(TOKENIZER), 'chunks': 3, 'width': width,
                'selection': 'first, middle, last 1023 BPE tokens', 'splits': {}}
    for split in ('train', 'dev', 'test'):
        rows = cases(split)
        tokens = np.full((len(rows), 3, width), eos, dtype=np.int32)
        positions = np.zeros((len(rows), 3), dtype=np.int16)
        for index, row in enumerate(tqdm(rows, desc=f'{split} chunks', unit='doc')):
            text = row['text']
            midpoint = len(text)//2
            snippets = (text[:16000],
                        text[max(0, midpoint-8000):min(len(text), midpoint+8000)],
                        text[-16000:])
            for chunk, passage in enumerate(snippets):
                encoded = tokenizer.encode(passage, add_special_tokens=False).ids
                selected = (encoded[-(width-1):] if chunk == 2
                            else encoded[:width-1] if chunk == 0
                            else encoded[max(0, (len(encoded)-(width-1))//2):][:width-1])
                sequence = selected + [eos]
                tokens[index, chunk, :len(sequence)] = sequence
                positions[index, chunk] = len(selected)
        np.save(folder/f'{split}_tokens.npy', tokens)
        np.save(folder/f'{split}_positions.npy', positions)
        (folder/f'{split}_ids.json').write_text(json.dumps([r['id'] for r in rows]))
        manifest['splits'][split] = {'cases': len(rows)}
    manifest['seconds'] = round(time.monotonic()-started, 2)
    (folder/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == '__main__':
    main()
