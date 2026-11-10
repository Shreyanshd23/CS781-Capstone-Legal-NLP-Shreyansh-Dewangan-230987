"""Official Phase-3 data loading and exact submission validation."""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'data/official/Phase_3_RR'


def labels() -> list[str]:
    vocab = json.loads((DATA / 'Train/label_vocab.json').read_text())
    if sorted(vocab.values()) != list(range(len(vocab))):
        raise ValueError('Noncontiguous official label vocabulary')
    return [name for name, _ in sorted(vocab.items(), key=lambda item: item[1])]


def documents(split: str) -> list[dict]:
    result = []
    seen = set()
    valid = set(labels())
    for domain in ('CL', 'IT'):
        folder = 'Test' if split == 'test' else 'Train'
        path = DATA / folder / f'{domain}_{split}.json'
        raw = json.loads(path.read_text())
        if split == 'test':
            if not isinstance(raw, dict):
                raise ValueError(f'{path} should map IDs to sentence lists')
            rows = [{'id': key, 'text': value} for key, value in raw.items()]
        else:
            if not isinstance(raw, list):
                raise ValueError(f'{path} should contain document records')
            rows = raw
        for row in rows:
            key = row['id']
            sentences = row['text']
            if key in seen or not isinstance(key, str) or not isinstance(sentences, list):
                raise ValueError(f'Duplicate/invalid document ID: {key}')
            seen.add(key)
            if any(not isinstance(s, str) for s in sentences):
                raise ValueError(f'Invalid sentence in {key}')
            if split != 'test':
                tags = row['labels']
                if len(tags) != len(sentences) or any(tag not in valid for tag in tags):
                    raise ValueError(f'Invalid label sequence in {key}')
            result.append({**row, 'domain': domain})
    return result


def validate_prediction(prediction: dict, test: list[dict] | None = None) -> None:
    test = documents('test') if test is None else test
    if not isinstance(prediction, dict) or set(prediction) != {row['id'] for row in test}:
        raise ValueError('Prediction document IDs do not match official test')
    valid = set(labels())
    for row in test:
        result = prediction[row['id']]
        if (not isinstance(result, list) or len(result) != len(row['text'])
                or any(not isinstance(tag, str) or tag not in valid for tag in result)):
            raise ValueError(f'Invalid label sequence in {row["id"]}')
