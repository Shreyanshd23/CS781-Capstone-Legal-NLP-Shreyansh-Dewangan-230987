#!/usr/bin/env python3
"""Correct a bounded quotation of an appellate tax authority's directions."""
from __future__ import annotations

import argparse
import json
import re

from .src.data import ROOT, documents, validate_prediction
from .src.evaluate import evaluate
from .train import save_submission, sha256


LOWER = 'RulingByLowerCourt'
INTRO = re.compile(r'\bIn\s+the\s+said\s+order\b.{0,80}'
                   r'\bfollowing\s+directions\s+were\s+issued\b', re.I)
NARRATIVE_RETURN = re.compile(r'^(?:Pursuant\s+thereto|Subsequently,?\s+the\s+assessee)', re.I)


def correct(row: dict, prediction: list[str]) -> tuple[list[str], list[dict]]:
    if len(row['text']) != len(prediction):
        raise ValueError('Sentence count mismatch')
    output = prediction.copy()
    changes = []
    for start, sentence in enumerate(row['text']):
        if not INTRO.search(sentence):
            continue
        end = start+1
        limit = min(len(output), start+18)
        while end < limit and not NARRATIVE_RETURN.search(row['text'][end]):
            end += 1
        if end == limit:
            continue
        for index in range(start,end):
            if output[index] != LOWER:
                changes.append({'index':index,'from':output[index],
                                'to':LOWER,'anchor':start,'end':end})
                output[index] = LOWER
    return output,changes


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--id',required=True)
    parser.add_argument('--source',required=True)
    args = parser.parse_args()
    run = ROOT/'runs'/args.id
    submission = ROOT/'submissions'/args.id
    if run.exists() or submission.exists():
        parser.error('Output already exists')
    source_run = ROOT/'runs'/args.source
    source_submission = ROOT/'submissions'/args.source
    outputs = {}; changes = {}
    for split in ('dev','test'):
        rows=documents(split)
        source=(source_run/'dev_predictions.json' if split=='dev'
                else source_submission/'prediction.json')
        base=json.loads(source.read_text())
        if set(base)!={row['id'] for row in rows}:
            raise ValueError(f'{split} ID mismatch')
        outputs[split]={};changes[split]=[]
        for row in rows:
            pred,log=correct(row,base[row['id']])
            outputs[split][row['id']]=pred
            for item in log:
                record={'document':row['id'],**item}
                if split=='dev':record['gold']=row['labels'][item['index']]
                changes[split].append(record)
    validate_prediction(outputs['test'])
    metrics=evaluate(documents('dev'),outputs['dev'])
    run.mkdir(parents=True)
    config={'id':args.id,'source':args.source,
            'method':'bounded_quoted_cit_order_correction',
            'source_prediction_sha256':sha256(source_submission/'prediction.json'),
            'dev_changes':len(changes['dev']),'test_changes':len(changes['test'])}
    (run/'config.json').write_text(json.dumps(config,indent=2)+'\n')
    (run/'dev_predictions.json').write_text(json.dumps(outputs['dev'],indent=2)+'\n')
    (run/'dev_metrics.json').write_text(json.dumps(metrics,indent=2)+'\n')
    (run/'changes.json').write_text(json.dumps(changes,indent=2)+'\n')
    (run/'metrics.jsonl').write_text(json.dumps({
        'epoch':0,'macro_f1':metrics['overall']['macro_f1'],
        'cl_macro_f1':metrics['domains']['CL']['macro_f1'],
        'it_macro_f1':metrics['domains']['IT']['macro_f1']})+'\n')
    path,archive=save_submission(outputs['test'],submission)
    (submission/'manifest.json').write_text(json.dumps({
        'experiment':args.id,'source':args.source,
        'dev_macro_f1':metrics['overall']['macro_f1'],
        'prediction_sha256':sha256(path),'zip_sha256':sha256(archive),
        'official_score':None},indent=2)+'\n')
    print(json.dumps({'id':args.id,'dev_macro_f1':metrics['overall']['macro_f1'],
                      'dev_changes':len(changes['dev']),
                      'test_changes':len(changes['test']),'zip':str(archive)}))


if __name__=='__main__':
    main()
