#!/usr/bin/env python3
"""Combine saved Phase-4 probabilities and package an ID-checked submission."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
from zipfile import ZIP_DEFLATED, ZipFile

import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score

from .data import ROOT, cases, validate_prediction


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--id', required=True)
    parser.add_argument('--source', nargs=2, action='append', metavar=('PATH', 'WEIGHT'), required=True)
    parser.add_argument('--threshold', type=float, required=True)
    parser.add_argument('--near-match-threshold', type=float, default=.7)
    args = parser.parse_args()
    if not 0 < args.threshold < 1:
        parser.error('Threshold must be in (0, 1)')
    weights = np.array([float(weight) for _, weight in args.source], dtype=np.float64)
    if np.any(weights < 0) or not np.isclose(weights.sum(), 1.0, atol=1e-6):
        parser.error('Source weights must be nonnegative and sum to one')
    rows = {split: cases(split) for split in ('dev', 'test')}
    expected = {split: np.array([r['id'] for r in values]) for split, values in rows.items()}
    sources = []
    for filename, _ in args.source:
        path = Path(filename)
        source = np.load(path)
        for split in rows:
            if not np.array_equal(source[f'{split}_ids'], expected[split]):
                raise RuntimeError(f'Source {path} has misaligned {split} IDs')
        sources.append(source)
    near = {split: np.load(ROOT/f'runs/near_duplicate_diagnostic/{split}.npz')
            for split in rows}
    for split in rows:
        if not np.array_equal(near[split]['ids'], expected[split]):
            raise RuntimeError(f'Near-match {split} IDs are misaligned')

    def predict(split: str) -> np.ndarray:
        probability = sum(weight * source[split]
                          for weight, source in zip(weights, sources))
        label = (probability >= args.threshold).astype(np.uint8)
        matched = near[split]['similarity'] >= args.near_match_threshold
        label[matched] = near[split]['labels'][matched]
        return label

    truth = np.array([r['label'] for r in rows['dev']])
    dev_prediction = predict('dev')
    metrics = {'macro_f1': float(f1_score(truth, dev_prediction, average='macro')),
               'macro_precision': float(precision_score(truth, dev_prediction, average='macro')),
               'macro_recall': float(recall_score(truth, dev_prediction, average='macro')),
               'threshold': args.threshold,
               'near_match_threshold': args.near_match_threshold,
               'sources': [{'path': path, 'weight': float(weight)}
                           for (path, _), weight in zip(args.source, weights)],
               'dev_cases': len(truth), 'test_cases': len(rows['test'])}
    prediction = [{'id': row['id'], 'pred': int(value)}
                  for row, value in zip(rows['test'], predict('test'))]
    validate_prediction(prediction)
    folder = ROOT/'submissions'/args.id
    folder.mkdir(parents=True, exist_ok=False)
    source_dir = folder/'sources'
    source_dir.mkdir()
    for index, source in enumerate(metrics['sources']):
        copy = source_dir/f'source_{index}.npz'
        shutil.copy2(source['path'], copy)
        source['snapshot'] = str(copy.relative_to(folder))
        source['sha256'] = hashlib.sha256(copy.read_bytes()).hexdigest()
    for split in rows:
        shutil.copy2(ROOT/f'runs/near_duplicate_diagnostic/{split}.npz',
                     source_dir/f'near_{split}.npz')
    json_path = folder/'prediction.json'
    json_path.write_text(json.dumps(prediction, indent=2)+'\n')
    zip_path = folder/'submission.zip'
    with ZipFile(zip_path, 'w', ZIP_DEFLATED, compresslevel=9) as archive:
        archive.write(json_path, 'prediction.json')
    with ZipFile(zip_path) as archive:
        if archive.namelist() != ['prediction.json'] or archive.testzip() is not None:
            raise RuntimeError('Invalid submission ZIP')
        if json.loads(archive.read('prediction.json')) != prediction:
            raise RuntimeError('Submission ZIP contents differ from prediction.json')
    metrics['prediction_sha256'] = hashlib.sha256(json_path.read_bytes()).hexdigest()
    metrics['zip_sha256'] = hashlib.sha256(zip_path.read_bytes()).hexdigest()
    (folder/'manifest.json').write_text(json.dumps(metrics, indent=2)+'\n')
    print(json.dumps({'archive': str(zip_path), 'validation': metrics}, indent=2))


if __name__ == '__main__':
    main()
