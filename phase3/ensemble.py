#!/usr/bin/env python3
"""Blend two Phase-3 classifiers by aligned document/sentence probabilities."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import warnings

import torch
from tokenizers import Tokenizer

from .position_prior import training_prior
from .src.data import ROOT, documents, labels
from .src.evaluate import evaluate
from .src.model import RoleModel
from .train import PARENT_ROOT, TOKENIZERS, encode_documents, predict, save_submission, sha256


def infer(run: Path, split: str, device, chunk_override: int | None = None) -> tuple[dict,dict]:
    meta=json.loads((run/'config.json').read_text())
    torch.set_float32_matmul_precision(meta.get('matmul_precision','high'))
    tokenizer=Tokenizer.from_file(str(TOKENIZERS[meta['kind']]))
    rows=encode_documents(documents(split),tokenizer,meta['structure'],
                          meta.get('target_tokens'),meta.get('neighbor_tokens',36),
                          meta.get('court_prefix',False))
    model=RoleModel(meta['kind'],meta['method'],meta['structure'],PARENT_ROOT,
                    TOKENIZERS[meta['kind']],meta.get('student_layers',0),
                    use_position=meta.get('position_features',False),
                    train_last_layers=meta.get('train_last_layers',4),
                    pooling=meta.get('pooling','last'),
                    input_dropout=meta.get('input_dropout',0.),
                    classifier_dropout=meta.get('classifier_dropout',.1)).to(device)
    checkpoint=torch.load(run/'best.pt',map_location=device,weights_only=False)
    model.load_state_dict(checkpoint['model'],strict=True)
    prediction,probabilities=predict(model,rows,device,chunk_override or meta['chunk'],meta['batch_size'])
    if split=='dev' and chunk_override is None:
        saved=json.loads((run/'dev_predictions.json').read_text())
        if set(saved)!=set(prediction) or any(len(saved[k])!=len(prediction[k]) for k in saved):
            raise RuntimeError(f'Checkpoint reload changed dev alignment for {run.name}')
        mismatches=sum(a!=b for k in saved for a,b in zip(saved[k],prediction[k]))
        total=sum(map(len,saved.values()))
        if mismatches/total>.005:
            raise RuntimeError(f'Checkpoint reload changed {mismatches}/{total} dev labels for {run.name}')
        if mismatches:
            warnings.warn(f'{run.name}: {mismatches}/{total} dev labels differ on checkpoint reload; '
                          'ensemble evaluation uses fresh inference probabilities')
    del model;torch.cuda.empty_cache()
    return prediction,probabilities


def combine(rows,first,second,weight,prior,prior_strength):
    vocab=labels();bins=len(prior)
    if set(first)!=set(second):raise ValueError('Model document IDs differ')
    output={}
    for row in rows:
        key=row['id'];size=len(row['text']);a=first[key];b=second[key]
        if len(a)!=size or len(b)!=size:raise ValueError(f'Sentence alignment failed: {key}')
        output[key]=[]
        for index in range(size):
            bucket=min(bins-1,int(index/max(1,size)*bins))
            fused=[(1-weight)*a[index][j]+weight*b[index][j]
                   for j in range(len(vocab))]
            label=max(range(len(vocab)),key=lambda j:
                math.log(max(1e-8,fused[j]))+prior_strength*prior[bucket][j])
            output[key].append(vocab[label])
    return output


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--id',default='E8_lorma_distilled_ensemble_20261014')
    ap.add_argument('--first',type=Path,default=ROOT/'runs/E4_dense_lorma_s4_20261014')
    ap.add_argument('--second',type=Path,default=ROOT/'runs/E6b_dense_student8_partial_kd_s3_20261014')
    ap.add_argument('--second-weight',type=float,default=.4)
    ap.add_argument('--position-weight',type=float,default=.5)
    ap.add_argument('--bins',type=int,default=10)
    ap.add_argument('--inference-chunk',type=int)
    args=ap.parse_args()
    run=ROOT/'runs'/args.id;submission=ROOT/'submissions'/args.id
    if run.exists() or submission.exists():ap.error('Output already exists')
    device=torch.device('cuda:0')
    rows_dev=documents('dev');rows_test=documents('test')
    if args.inference_chunk is not None and args.inference_chunk<=4:ap.error('Inference chunk must exceed overlap')
    _,a_dev=infer(args.first,'dev',device,args.inference_chunk)
    _,b_dev=infer(args.second,'dev',device,args.inference_chunk)
    _,a_test=infer(args.first,'test',device,args.inference_chunk)
    _,b_test=infer(args.second,'test',device,args.inference_chunk)
    prior=training_prior(args.bins)
    dev_pred=combine(rows_dev,a_dev,b_dev,args.second_weight,prior,args.position_weight)
    test_pred=combine(rows_test,a_test,b_test,args.second_weight,prior,args.position_weight)
    metrics=evaluate(rows_dev,dev_pred)
    run.mkdir(parents=True)
    (run/'config.json').write_text(json.dumps({
        'id':args.id,'method':'aligned_probability_ensemble',
        'first_run':str(args.first.resolve()),'second_run':str(args.second.resolve()),
        'first_checkpoint_sha256':sha256(args.first/'best.pt'),
        'second_checkpoint_sha256':sha256(args.second/'best.pt'),
        'second_weight':args.second_weight,'position_weight':args.position_weight,
        'inference_chunk':args.inference_chunk,
        'bins':args.bins,'position_prior':prior},indent=2)+'\n')
    (run/'dev_predictions.json').write_text(json.dumps(dev_pred,indent=2)+'\n')
    (run/'dev_metrics.json').write_text(json.dumps(metrics,indent=2)+'\n')
    (run/'metrics.jsonl').write_text(json.dumps({
        'epoch':0,'macro_f1':metrics['overall']['macro_f1'],
        'cl_macro_f1':metrics['domains']['CL']['macro_f1'],
        'it_macro_f1':metrics['domains']['IT']['macro_f1']})+'\n')
    prediction_path,archive=save_submission(test_pred,submission)
    (submission/'manifest.json').write_text(json.dumps({
        'experiment':args.id,'dev_macro_f1':metrics['overall']['macro_f1'],
        'prediction_sha256':sha256(prediction_path),
        'zip_sha256':sha256(archive),'official_score':None},indent=2)+'\n')
    print(json.dumps({'macro_f1':metrics['overall']['macro_f1'],
                      'CL':metrics['domains']['CL']['macro_f1'],
                      'IT':metrics['domains']['IT']['macro_f1'],
                      'zip':str(archive)}))


if __name__=='__main__':main()
