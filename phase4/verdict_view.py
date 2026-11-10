#!/usr/bin/env python3
"""Train a train-only text model on likely disposition passages."""
from __future__ import annotations

import json
import re
import time

import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score

from .data import ROOT, cases


KEYWORDS=re.compile(r'\b(?:appeals?|petitions?|allowed|dismissed|set\s+aside|upheld|'
                    r'convictions?|acquitted|disposed\s+of|succeed|rejected|'
                    r'impugned|judg(?:e)?ment|relief|quashed|granted|denied)\b',re.I)
SENTENCE=re.compile(r'(?<=[.!?])\s+|\n+')


def view(text:str)->str:
    # Use only the tail half for disposition cues; cap extreme cases first.
    tail=text[-20000:]
    chunks=SENTENCE.split(tail)
    found=[part.strip()[:800] for part in chunks if KEYWORDS.search(part)]
    return text[:600]+' HEAD_END '+text[-600:]+' VERDICT_CUES '+' '.join(found[-12:])


def main():
    start=time.monotonic()
    train,dev,test=(cases(split) for split in ('train','dev','test'))
    old=[r for r in train if int(r['id'][:4])<2000]
    vectorizer=TfidfVectorizer(ngram_range=(1,2),min_df=3,max_features=160000,
                               sublinear_tf=True,strip_accents='unicode',dtype=np.float32)
    x_train=vectorizer.fit_transform(view(r['text']) for r in old)
    x_dev=vectorizer.transform(view(r['text']) for r in dev)
    x_test=vectorizer.transform(view(r['text']) for r in test)
    y=np.array([r['label'] for r in old]);truth=np.array([r['label'] for r in dev])
    folder=ROOT/'runs/verdict_view';folder.mkdir(parents=True,exist_ok=True)
    results=[]
    for c in (.3,1.,3.):
        model=LogisticRegression(C=c,class_weight='balanced',solver='liblinear',
                                 max_iter=400,random_state=42)
        model.fit(x_train,y)
        dp=model.predict_proba(x_dev)[:,1]
        tp=model.predict_proba(x_test)[:,1]
        best=max(({'C':c,'threshold':round(float(t),3),
                   'macro_f1':float(f1_score(truth,dp>=t,average='macro'))}
                  for t in np.arange(.35,.651,.01)),key=lambda r:r['macro_f1'])
        results.append(best);print(json.dumps(best),flush=True)
        np.savez_compressed(folder/f'C{c}_probabilities.npz',dev=dp.astype(np.float32),
                            test=tp.astype(np.float32),
                            dev_ids=np.array([r['id'] for r in dev]),
                            test_ids=np.array([r['id'] for r in test]))
        joblib.dump(model,folder/f'C{c}_model.joblib',compress=3)
    joblib.dump(vectorizer,folder/'vectorizer.joblib',compress=3)
    (folder/'metrics.json').write_text(json.dumps({'scores':results,
        'train_cases':len(old),'features':x_train.shape[1],
        'seconds':round(time.monotonic()-start,2)},indent=2)+'\n')


if __name__=='__main__':main()
