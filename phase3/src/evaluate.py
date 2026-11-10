"""Sentence-level Macro-F1 evaluation with fixed official label order."""
from __future__ import annotations

from collections import defaultdict
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support

from .data import labels


def evaluate(rows: list[dict], prediction: dict[str, list[str]]) -> dict:
    order = labels()
    if set(prediction) != {row['id'] for row in rows}:
        raise ValueError('Prediction/ground-truth document mismatch')
    gold, pred, domains, positions, lengths = [], [], [], [], []
    for row in rows:
        target = row['labels']
        output = prediction[row['id']]
        if len(target) != len(output) or any(tag not in order for tag in output):
            raise ValueError(f'Invalid prediction for {row["id"]}')
        gold.extend(target); pred.extend(output)
        domains.extend([row['domain']] * len(target))
        positions.extend([(i + .5) / max(1, len(target)) for i in range(len(target))])
        lengths.extend([len(target)] * len(target))

    def metrics(indices):
        if not indices:
            return {'macro_f1': 0., 'macro_precision': 0., 'macro_recall': 0.,
                    'sentences': 0, 'per_class': {
                        tag: {'precision': 0., 'recall': 0., 'f1': 0., 'support': 0}
                        for tag in order}}
        g = [gold[i] for i in indices]; p = [pred[i] for i in indices]
        precision, recall, f1, support = precision_recall_fscore_support(
            g, p, labels=order, zero_division=0)
        return {'macro_f1': float(f1.mean()),
                'macro_precision': float(precision.mean()),
                'macro_recall': float(recall.mean()),
                'sentences': len(indices),
                'per_class': {tag: {'precision': float(precision[j]),
                                    'recall': float(recall[j]),
                                    'f1': float(f1[j]), 'support': int(support[j])}
                              for j, tag in enumerate(order)}}

    all_ids = list(range(len(gold)))
    result = {'overall': metrics(all_ids),
              'domains': {domain: metrics([i for i in all_ids if domains[i] == domain])
                          for domain in ('CL', 'IT')},
              'confusion_matrix': confusion_matrix(gold, pred, labels=order).tolist(),
              'label_order': order,
              'aggregation': 'Unweighted mean of 13 per-class F1 values over all CL+IT sentences; zero_division=0',
              'position_quartiles': {}, 'document_length': {}}
    for j in range(4):
        result['position_quartiles'][str(j + 1)] = metrics(
            [i for i in all_ids if min(3, int(positions[i] * 4)) == j])['macro_f1']
    for name, condition in [('short', lambda n:n<100),
                            ('medium',lambda n:100<=n<300),
                            ('long',lambda n:n>=300)]:
        result['document_length'][name] = metrics(
            [i for i in all_ids if condition(lengths[i])])['macro_f1']
    return result
