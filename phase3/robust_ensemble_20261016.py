#!/usr/bin/env python3
"""Fuse the established E25 ensemble with weight-averaged and local-context models."""
from __future__ import annotations

import argparse
import json

import numpy as np

from .position_prior import training_prior
from .src.data import ROOT, documents, labels
from .src.evaluate import evaluate
from .train import save_submission, sha256


BASE=('E22_dense_partial_position_s4_20261014',
      'E11_dense_partial_s4_20261014')
SWA='E52_dense_swa_20261016'
LOCAL='E28_dense_local_hierarchy4_20261015'
WEIGHTS={'E25_base':.50,SWA:.35,LOCAL:.15}


def probabilities(split):
    base_dir=ROOT/'prediction_cache/e25_chunk32_20261016'
    base=[json.loads((base_dir/f'{name}_{split}.json').read_text()) for name in BASE]
    swa=json.loads((ROOT/'runs'/SWA/f'{split}_probabilities.json').read_text())
    local=json.loads((ROOT/'prediction_cache/refinement_20261015_chunk64'/f'{LOCAL}.json').read_text())[split]
    return base,swa,local


def predict(rows,components,position_weight):
    base,swa,local=components
    order=labels();prior=np.asarray(training_prior(10))
    result={}
    for row in rows:
        key=row['id'];n=len(row['text'])
        p=(WEIGHTS['E25_base']*.5*(np.asarray(base[0][key])+np.asarray(base[1][key]))
           +WEIGHTS[SWA]*np.asarray(swa[key])
           +WEIGHTS[LOCAL]*np.asarray(local[key]))
        assert p.shape==(n,len(order))
        bins=np.minimum(9,(np.arange(n)/n*10).astype(int))
        score=np.log(np.maximum(1e-8,p))+position_weight*prior[bins]
        result[key]=[order[j] for j in score.argmax(1)]
    return result


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--id',default='E56_shared_swa_local_20261016')
    ap.add_argument('--position-weight',type=float,default=.5)
    args=ap.parse_args()
    name=args.id
    run=ROOT/'runs'/name;submission=ROOT/'submissions'/name
    if run.exists() or submission.exists():raise FileExistsError(name)
    dev=documents('dev');test=documents('test')
    dp=predict(dev,probabilities('dev'),args.position_weight)
    tp=predict(test,probabilities('test'),args.position_weight)
    metric=evaluate(dev,dp)
    run.mkdir()
    config={'id':name,'method':'swa_and_local_context_probability_ensemble',
            'sources':[str(ROOT/'runs'/s) for s in (*BASE,SWA,LOCAL)],
            'checkpoint_sha256':{s:sha256(ROOT/'runs'/s/'best.pt')
                                 for s in (*BASE,SWA,LOCAL)},
            'weights':WEIGHTS,'position_prior_strength':args.position_weight,
            'development_labels_used_for':'candidate selection only'}
    (run/'config.json').write_text(json.dumps(config,indent=2)+'\n')
    (run/'dev_predictions.json').write_text(json.dumps(dp,indent=2)+'\n')
    (run/'dev_metrics.json').write_text(json.dumps(metric,indent=2)+'\n')
    (run/'metrics.jsonl').write_text(json.dumps({
        'epoch':0,'macro_f1':metric['overall']['macro_f1'],
        'cl_macro_f1':metric['domains']['CL']['macro_f1'],
        'it_macro_f1':metric['domains']['IT']['macro_f1']})+'\n')
    path,archive=save_submission(tp,submission)
    (submission/'manifest.json').write_text(json.dumps({
        'experiment':name,'dev_macro_f1':metric['overall']['macro_f1'],
        'prediction_sha256':sha256(path),'zip_sha256':sha256(archive),
        'official_score':None},indent=2)+'\n')
    print(json.dumps({'experiment':name,'macro_f1':metric['overall']['macro_f1'],
                      'CL':metric['domains']['CL']['macro_f1'],
                      'IT':metric['domains']['IT']['macro_f1'],
                      'zip':str(archive)}))


if __name__=='__main__':main()
