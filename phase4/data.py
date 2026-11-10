"""Load official CJPE examples while excluding test-only metadata and labels."""
from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent
DATA = ROOT / 'data/official/Phase_4_CJPE'


def cases(split: str) -> list[dict]:
    if split not in ('train', 'dev', 'test'):
        raise ValueError(split)
    part = 'Test' if split == 'test' else 'Train'
    raw = json.loads((DATA / part / f'{split}.json').read_text())
    if not isinstance(raw, list) or not raw:
        raise ValueError(f'Invalid {split} split')
    rows = []
    seen = {}
    for item in raw:
        key, text = item.get('id'), item.get('text')
        if not isinstance(key, str) or not key:
            raise ValueError(f'Invalid case ID: {key!r}')
        if not isinstance(text, str) or not text.strip():
            raise ValueError(f'Empty case text: {key!r}')
        # The official test file currently contains a label field. It is
        # intentionally ignored and cannot enter training or validation.
        row = {'id': key, 'text': text}
        if split != 'test':
            value = item.get('label')
            if type(value) is not int or value not in (0, 1):
                raise ValueError(f'Invalid training label: {key!r}')
            row['label'] = value
        if key in seen:
            if split == 'test' or row != seen[key]:
                raise ValueError(f'Conflicting or duplicate test case ID: {key!r}')
            continue  # Exact duplicate in official train/dev split.
        seen[key] = row
        rows.append(row)
    return rows


def validate_prediction(prediction: list[dict]) -> None:
    expected = [row['id'] for row in cases('test')]
    if not isinstance(prediction, list) or len(prediction) != len(expected):
        raise ValueError('Wrong number of test predictions')
    for index, (item, key) in enumerate(zip(prediction, expected)):
        if (not isinstance(item, dict) or set(item) != {'id', 'pred'}
                or item['id'] != key or type(item['pred']) is not int
                or item['pred'] not in (0, 1)):
            raise ValueError(f'Invalid test prediction at position {index}')
