#!/usr/bin/env python3
"""Combine two audited CRF outputs by entity label and validate the submission."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

from ner_core import LABELS, read_folds
from sweep_crf_inference import score_predictions
from train_ner import sha256


def merge(primary, specialist, labels):
    if set(primary) != set(specialist):
        raise ValueError('Document IDs differ between primary and specialist')
    out = {}
    for key in primary:
        keep = [row for row in primary[key] if row[2] not in labels]
        for row in specialist[key]:
            if row[2] not in labels:
                continue
            # The primary retains an overlapping span of another label.
            if any(row[0] < existing[1] and existing[0] < row[1] for existing in keep):
                continue
            keep.append(row)
        out[key] = sorted(keep, key=lambda row: (row[0], row[1], row[2]))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--primary-run', type=Path, required=True)
    ap.add_argument('--primary-submission', type=Path, required=True)
    ap.add_argument('--specialist-run', type=Path, required=True)
    ap.add_argument('--specialist-submission', type=Path, required=True)
    ap.add_argument('--data', type=Path, required=True)
    ap.add_argument('--run-dir', type=Path, required=True)
    ap.add_argument('--submission-dir', type=Path, required=True)
    ap.add_argument('--labels', nargs='+', choices=LABELS, default=['AUTH','CASENO'])
    args = ap.parse_args()
    if args.run_dir.exists() or args.submission_dir.exists():
        ap.error('Output directory exists')
    first_config = json.loads((args.primary_run / 'config.json').read_text())
    second_config = json.loads((args.specialist_run / 'config.json').read_text())
    ids = first_config['validation_ids']
    if set(ids) != set(second_config['validation_ids']):
        raise ValueError('Different validation documents')
    primary_val = json.loads((args.primary_run / 'validation_predictions.json').read_text())
    specialist_val = json.loads((args.specialist_run / 'validation_predictions.json').read_text())
    validation = merge(primary_val, specialist_val, set(args.labels))
    rows = read_folds(args.data)
    gold = {key: rows[key]['entities'] for key in ids}
    folds = {key: rows[key]['fold'] for key in ids}
    metrics = score_predictions(validation, gold, folds)
    metrics.update({'epoch': 0, 'method': 'label_router', 'backbone_kind': 'moe',
                    'structure': 'bilou_crf', 'validation_macro_f1': metrics['macro_f1']})
    print('Validation', json.dumps(metrics), flush=True)
    test = json.loads((args.data / 'Test/fold3-text.json').read_text())
    primary_test = json.loads((args.primary_submission / 'prediction.json').read_text())
    specialist_test = json.loads((args.specialist_submission / 'prediction.json').read_text())
    prediction = merge(primary_test, specialist_test, set(args.labels))
    if set(prediction) != set(test):
        raise RuntimeError('Official test IDs differ')
    for key, spans in prediction.items():
        seen = set()
        for span in spans:
            if (len(span) != 3 or type(span[0]) is not int or type(span[1]) is not int
                    or not 0 <= span[0] < span[1] <= len(test[key])
                    or span[2] not in LABELS or tuple(span) in seen):
                raise RuntimeError(f'Invalid or duplicate span: {key} {span}')
            seen.add(tuple(span))
    args.run_dir.mkdir(parents=True)
    args.submission_dir.mkdir(parents=True)
    provenance = {'primary_run': str(args.primary_run.resolve()),
                  'primary_submission': str(args.primary_submission.resolve()),
                  'specialist_run': str(args.specialist_run.resolve()),
                  'specialist_submission': str(args.specialist_submission.resolve()),
                  'primary_prediction_sha256': sha256(args.primary_submission/'prediction.json'),
                  'specialist_prediction_sha256': sha256(args.specialist_submission/'prediction.json'),
                  'training_ids': first_config['training_ids'], 'validation_ids': ids,
                  'specialist_labels': args.labels, 'overlap_priority': 'primary_other_label',
                  'method': 'label_router', 'backbone_kind': 'moe', 'structure': 'bilou_crf'}
    (args.run_dir/'config.json').write_text(json.dumps(provenance,indent=2)+'\n')
    (args.run_dir/'metrics.jsonl').write_text(json.dumps(metrics)+'\n')
    (args.run_dir/'validation_predictions.json').write_text(
        json.dumps(validation,ensure_ascii=False)+'\n')
    output = args.submission_dir/'prediction.json'
    output.write_text(json.dumps(prediction,ensure_ascii=False,indent=2)+'\n')
    archive = args.submission_dir/'submission.zip'
    with ZipFile(archive,'w',ZIP_DEFLATED,compresslevel=9) as stream:
        stream.write(output,arcname='prediction.json')
    with ZipFile(archive) as stream:
        if stream.namelist()!=['prediction.json'] or stream.testzip() is not None:
            raise RuntimeError('Invalid submission ZIP')
    manifest = {'prediction_sha256':sha256(output),'submission_zip_sha256':sha256(archive),
                'test_sha256':sha256(args.data/'Test/fold3-text.json'),
                'documents':len(prediction),'predicted_entities':sum(map(len,prediction.values())),
                'labels':LABELS,'specialist_labels':args.labels}
    (args.submission_dir/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(f'Saved {len(prediction)} official IDs to {archive}',flush=True)


if __name__=='__main__':
    main()
