#!/usr/bin/env python3
"""Apply a training-derived sentence-position prior to CRF emissions."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch
from tokenizers import Tokenizer

from .src.data import ROOT, documents, labels
from .src.evaluate import evaluate
from .src.model import LinearCRF, RoleModel
from .train import PARENT_ROOT, TOKENIZERS, encode_documents, predict, save_submission, sha256


def training_prior(bins: int) -> list[list[float]]:
    vocab = labels()
    counts = [[1.] * len(vocab) for _ in range(bins)]
    total = [1.] * len(vocab)
    for row in documents('train'):
        for index, label in enumerate(row['labels']):
            bucket = min(bins-1, int(index/max(1,len(row['text']))*bins))
            j = vocab.index(label)
            counts[bucket][j] += 1
            total[j] += 1
    overall = [n/sum(total) for n in total]
    return [[math.log(row[j]/sum(row)) - math.log(overall[j])
             for j in range(len(vocab))] for row in counts]


def correct(rows: list[dict], probabilities: dict, crf: LinearCRF,
            prior: list[list[float]], strength: float) -> dict:
    vocab = labels(); bins = len(prior)
    output = {}
    for row in rows:
        key = row['id']; n = len(row['text'])
        if len(probabilities[key]) != n:
            raise ValueError(f'Probability count mismatch for {key}')
        if not n:
            output[key] = []; continue
        emissions = []
        for index, probabilities_at_sentence in enumerate(probabilities[key]):
            bucket = min(bins-1,int(index/n*bins))
            emissions.append([
                math.log(max(1e-8,probabilities_at_sentence[j]))
                + strength*prior[bucket][j] for j in range(len(vocab))])
        output[key] = [vocab[j] for j in crf.decode(
            torch.tensor(emissions, device=crf.start.device))]
    return output


def main() -> None:
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--id',default='E7_dense_lorma_position_crf_20261014')
    ap.add_argument('--source-run',type=Path,
                    default=ROOT/'runs/E4_dense_lorma_s4_20261014')
    ap.add_argument('--bins',type=int,default=10)
    ap.add_argument('--strength',type=float,default=1.)
    args=ap.parse_args()
    output_dir=ROOT/'runs'/args.id;submission_dir=ROOT/'submissions'/args.id
    if output_dir.exists() or submission_dir.exists():ap.error('Output already exists')
    meta=json.loads((args.source_run/'config.json').read_text())
    if meta['structure']!='s4':ap.error('CRF source required')
    tokenizer=Tokenizer.from_file(str(TOKENIZERS[meta['kind']]))
    dev=encode_documents(documents('dev'),tokenizer,'s4')
    test=encode_documents(documents('test'),tokenizer,'s4')
    device=torch.device('cuda:0')
    model=RoleModel(meta['kind'],meta['method'],'s4',PARENT_ROOT,
                    TOKENIZERS[meta['kind']],meta.get('student_layers',0),
                    use_position=meta.get('position_features',False)).to(device)
    checkpoint=torch.load(args.source_run/'best.pt',map_location=device,weights_only=False)
    model.load_state_dict(checkpoint['model'],strict=True)
    baseline_dev,prob_dev=predict(model,dev,device,meta['chunk'],meta['batch_size'])
    saved=json.loads((args.source_run/'dev_predictions.json').read_text())
    if baseline_dev!=saved:raise RuntimeError('Source checkpoint reload changed dev predictions')
    _,prob_test=predict(model,test,device,meta['chunk'],meta['batch_size'])
    prior=training_prior(args.bins)
    updated_dev=correct(dev,prob_dev,model.crf,prior,args.strength)
    updated_test=correct(test,prob_test,model.crf,prior,args.strength)
    dev_metrics=evaluate(dev,updated_dev)
    baseline_metrics=evaluate(dev,baseline_dev)
    output_dir.mkdir(parents=True)
    config={'id':args.id,'source_run':str(args.source_run.resolve()),
            'source_best_sha256':sha256(args.source_run/'best.pt'),
            'method':'training_position_prior_crf_decoding',
            'architecture':meta['kind'],'adaptation':meta['method'],
            'structure':'s4','bins':args.bins,'prior_strength':args.strength,
            'position_prior':prior,'development_ids':[r['id'] for r in dev],
            'test_ids':[r['id'] for r in test]}
    (output_dir/'config.json').write_text(json.dumps(config,indent=2)+'\n')
    (output_dir/'dev_metrics.json').write_text(json.dumps(dev_metrics,indent=2)+'\n')
    (output_dir/'dev_predictions.json').write_text(json.dumps(updated_dev,indent=2)+'\n')
    (output_dir/'metrics.jsonl').write_text(json.dumps({
        'epoch':0,'macro_f1':dev_metrics['overall']['macro_f1'],
        'cl_macro_f1':dev_metrics['domains']['CL']['macro_f1'],
        'it_macro_f1':dev_metrics['domains']['IT']['macro_f1'],
        'baseline_macro_f1':baseline_metrics['overall']['macro_f1']})+'\n')
    prediction_path,archive_path=save_submission(updated_test,submission_dir)
    (submission_dir/'manifest.json').write_text(json.dumps({
        'source_run':str(args.source_run.resolve()),
        'run_config':str((output_dir/'config.json').resolve()),
        'dev_macro_f1':dev_metrics['overall']['macro_f1'],
        'baseline_macro_f1':baseline_metrics['overall']['macro_f1'],
        'prediction_sha256':sha256(prediction_path),
        'zip_sha256':sha256(archive_path),
        'official_score':None},indent=2)+'\n')
    print(json.dumps({'baseline':baseline_metrics['overall']['macro_f1'],
                      'candidate':dev_metrics['overall']['macro_f1'],
                      'CL':dev_metrics['domains']['CL']['macro_f1'],
                      'IT':dev_metrics['domains']['IT']['macro_f1'],
                      'zip':str(archive_path)}))


if __name__=='__main__':main()
