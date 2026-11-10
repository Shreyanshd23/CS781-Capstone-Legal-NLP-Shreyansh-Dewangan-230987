#!/usr/bin/env python3
"""Evaluate CRF window merging and conservative entity-logit calibration."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import numpy as np
import torch
from tokenizers import Tokenizer
from tqdm import tqdm

from ner_core import LABELS, read_folds, strict_macro_f1, tokenize_document, windows
from structured_ner import BILOU, StructuredNER, decode_bilou
from train_structured_ner import boundary_f1
from src.model import LegalLMConfig


class NumpyCRF:
    def __init__(self, crf):
        self.transitions = np.where(crf.allowed.cpu().numpy(),
                                    crf.transitions.detach().float().cpu().numpy(), -1e4)
        self.initial = np.where(crf.start_allowed.cpu().numpy(),
                                crf.initial.detach().float().cpu().numpy(), -1e4)
        self.final = np.where(crf.end_allowed.cpu().numpy(),
                              crf.final.detach().float().cpu().numpy(), -1e4)

    def decode(self, emissions):
        score = self.initial + emissions[0]
        back = []
        indices = np.arange(len(score))
        for row in emissions[1:]:
            candidates = score[:, None] + self.transitions
            previous = candidates.argmax(axis=0)
            score = candidates[previous, indices] + row
            back.append(previous)
        tag = int(np.argmax(score + self.final))
        path = [tag]
        for previous in reversed(back):
            tag = int(previous[tag])
            path.append(tag)
        return path[::-1]


def load_model(checkpoint, device):
    state = torch.load(checkpoint, map_location='cpu', weights_only=False)
    meta = state['meta']
    if meta['structure'] != 'bilou_crf':
        raise ValueError('Expected a BILOU-CRF checkpoint')
    config = LegalLMConfig(**state['model_config'])
    model = StructuredNER(config, meta['backbone_kind'], meta['method'],
                          'bilou_crf', bidir_head=meta.get('bidir_head', False),
                          transformer_head=meta.get('transformer_head', False))
    model.configure_adaptation()
    model.load_state_dict(state['model'], strict=True)
    model.to(device).eval()
    return model


@torch.inference_mode()
def make_cache(model, docs, length, stride, device):
    records = {}
    for doc in tqdm(docs, desc=f'Emission cache {length}/{stride}', unit='doc'):
        out = []
        for row in windows(doc, length, stride):
            ids = torch.tensor(row['ids'], device=device)[None]
            with torch.autocast('cuda', dtype=torch.bfloat16):
                emission = model.emit(model.encode(ids))[0].float()
            out.append((row['start'], emission.cpu().numpy()))
        records[doc.doc_id] = {'text': doc.text, 'offsets': doc.offsets,
                               'windows': out}
    return records


def emission_bias(spec):
    bias = np.zeros(len(BILOU), dtype=np.float32)
    for label, value in spec.items():
        for i, tag in enumerate(BILOU):
            if tag.endswith('-' + label):
                bias[i] = value
    return bias


def decode_record(record, crf, mode, bias):
    n = len(record['offsets'])
    if n == 0:
        return []
    if mode == 'stitch':
        chosen = [0] * n
        priority = [-1] * n
        for start, emissions in record['windows']:
            tags = crf.decode(emissions + bias)
            for local, tag in enumerate(tags):
                index = start + local
                if local > priority[index]:
                    chosen[index], priority[index] = tag, local
    else:
        if mode == 'latest':
            merged = np.zeros((n, len(BILOU)), dtype=np.float32)
            priority = np.full(n, -1, dtype=np.int32)
            for start, emissions in record['windows']:
                stop = start + len(emissions)
                local = np.arange(len(emissions), dtype=np.int32)
                take = local > priority[start:stop]
                merged[start:stop][take] = emissions[take]
                priority[start:stop][take] = local[take]
        else:
            merged = np.zeros((n, len(BILOU)), dtype=np.float32)
            weights = np.zeros(n, dtype=np.float32)
            for start, emissions in record['windows']:
                stop = start + len(emissions)
                if mode == 'mean':
                    weight = np.ones(len(emissions), dtype=np.float32)
                elif mode == 'causal_weighted':
                    weight = np.arange(1, len(emissions) + 1, dtype=np.float32)
                    weight /= len(emissions)
                else:
                    raise ValueError(mode)
                merged[start:stop] += emissions * weight[:, None]
                weights[start:stop] += weight
            merged /= weights[:, None]
        chosen = crf.decode(merged + bias)
    return decode_bilou(chosen, record['offsets'], record['text'])


def score_predictions(predictions, gold, folds):
    overall = strict_macro_f1(predictions, gold)['macro_f1']
    output = {'macro_f1': overall, 'boundary_f1': boundary_f1(predictions, gold),
              'entities': sum(map(len, predictions.values()))}
    for fold in (1, 2):
        ids = [k for k in predictions if folds[k] == fold]
        output[f'fold{fold}_macro_f1'] = strict_macro_f1(
            {k: predictions[k] for k in ids}, {k: gold[k] for k in ids})['macro_f1']
    return output


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--checkpoint', type=Path, required=True)
    ap.add_argument('--tokenizer', type=Path, required=True)
    ap.add_argument('--data', type=Path, required=True)
    ap.add_argument('--output-dir', type=Path, required=True)
    ap.add_argument('--lengths', type=int, nargs='+', default=[512, 768, 1024])
    args = ap.parse_args()
    if args.output_dir.exists():
        ap.error('Output directory already exists')
    args.output_dir.mkdir(parents=True)
    config = json.loads((args.checkpoint.parent / 'config.json').read_text())
    rows = read_folds(args.data)
    ids = config['validation_ids']
    tokenizer = Tokenizer.from_file(str(args.tokenizer))
    docs = [tokenize_document(k, rows[k]['text'], tokenizer) for k in ids]
    gold = {k: rows[k]['entities'] for k in ids}
    folds = {k: rows[k]['fold'] for k in ids}
    device = torch.device('cuda:0')
    model = load_model(args.checkpoint, device)
    crf = NumpyCRF(model.crf)
    results = []
    started = time.monotonic()
    for length in args.lengths:
        stride = length * 3 // 4
        cache = make_cache(model, docs, length, stride, device)
        for mode in ('stitch', 'latest', 'mean', 'causal_weighted'):
            candidate = {'length': length, 'stride': stride, 'mode': mode, 'bias': {}}
            predictions = {k: decode_record(v, crf, mode, emission_bias({}))
                           for k, v in cache.items()}
            candidate.update(score_predictions(predictions, gold, folds))
            results.append(candidate)
            print(json.dumps(candidate), flush=True)
        # The bias sweep is deliberately small to limit tuning on ten documents.
        for amount in (.25, .5, 1.0):
            bias = {'APP': amount, 'RESP': amount, 'AUTH': amount}
            for mode in ('stitch', 'latest'):
                candidate = {'length': length, 'stride': stride,
                             'mode': mode, 'bias': bias}
                predictions = {k: decode_record(v, crf, mode, emission_bias(bias))
                               for k, v in cache.items()}
                candidate.update(score_predictions(predictions, gold, folds))
                results.append(candidate)
                print(json.dumps(candidate), flush=True)
    report = {'checkpoint': str(args.checkpoint.resolve()),
              'validation_ids': ids, 'elapsed_seconds': time.monotonic() - started,
              'results': sorted(results, key=lambda x: x['macro_f1'], reverse=True)}
    (args.output_dir / 'sweep.json').write_text(json.dumps(report, indent=2) + '\n')
    print('BEST', json.dumps(report['results'][0]), flush=True)


if __name__ == '__main__':
    main()
