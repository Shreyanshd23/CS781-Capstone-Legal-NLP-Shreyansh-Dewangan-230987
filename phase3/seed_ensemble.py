#!/usr/bin/env python3
"""Blend independent full fine-tune seeds with the established partial model."""
from __future__ import annotations

import json
import math

import numpy as np

from .position_prior import training_prior
from .src.data import ROOT, documents, labels
from .src.evaluate import evaluate
from .train import save_submission, sha256


SOURCES={
    'E22_dense_partial_position_s4_20261014':.50,
    'E54_dense_full_low_lr_20261016':.25,
    'E73_full_seed17_20261016':.125,
    'E74_full_seed123_20261016':.125,
}
RUN_ID='E77_full_seed_ensemble_20261016'


def load(name:str,split:str)->dict:
    if name.startswith('E22_'):
        path=ROOT/'prediction_cache/e25_chunk32_20261016'/f'{name}_{split}.json'
    else:
        path=ROOT/'runs'/name/f'{split}_probabilities.json'
    return json.loads(path.read_text())


def predict(rows:list[dict], probabilities:dict[str,dict], position_prior:list[list[float]])->dict:
    order=labels();output={}
    for row in rows:
        key=row['id'];n=len(row['text'])
        fused=sum(weight*np.asarray(probabilities[name][key],dtype=np.float64)
                  for name,weight in SOURCES.items())
        if fused.shape!=(n,len(order)):raise ValueError(f'Probability shape mismatch: {key}')
        bucket=np.minimum(9,(np.arange(n)/max(1,n)*10).astype(int))
        scores=np.log(np.maximum(fused,1e-8))+.25*np.asarray(position_prior)[bucket]
        output[key]=[order[int(j)] for j in scores.argmax(axis=1)]
    return output


def main()->None:
    run=ROOT/'runs'/RUN_ID;submission=ROOT/'submissions'/RUN_ID
    if run.exists() or submission.exists():raise FileExistsError(RUN_ID)
    position_prior=training_prior(10)
    outputs={}
    for split in ('dev','test'):
        rows=documents(split)
        probabilities={name:load(name,split) for name in SOURCES}
        outputs[split]=predict(rows,probabilities,position_prior)
    metric=evaluate(documents('dev'),outputs['dev'])
    run.mkdir()
    (run/'config.json').write_text(json.dumps({'id':RUN_ID,'method':'full_seed_probability_ensemble',
        'structure':'ensemble','sources':SOURCES,'position_prior_strength':.25,
        'source_checkpoint_sha256':{name:sha256(ROOT/'runs'/name/'best.pt') for name in SOURCES}},indent=2)+'\n')
    (run/'dev_predictions.json').write_text(json.dumps(outputs['dev'],indent=2)+'\n')
    (run/'dev_metrics.json').write_text(json.dumps(metric,indent=2)+'\n')
    (run/'metrics.jsonl').write_text(json.dumps({'epoch':0,
        'macro_f1':metric['overall']['macro_f1'],
        'cl_macro_f1':metric['domains']['CL']['macro_f1'],
        'it_macro_f1':metric['domains']['IT']['macro_f1']})+'\n')
    path,archive=save_submission(outputs['test'],submission)
    (submission/'manifest.json').write_text(json.dumps({'experiment':RUN_ID,
        'dev_macro_f1':metric['overall']['macro_f1'],
        'prediction_sha256':sha256(path),'zip_sha256':sha256(archive),
        'official_score':None},indent=2)+'\n')
    print(json.dumps({'experiment':RUN_ID,'dev_macro_f1':metric['overall']['macro_f1'],
                      'zip':str(archive)}))


if __name__=='__main__':main()
