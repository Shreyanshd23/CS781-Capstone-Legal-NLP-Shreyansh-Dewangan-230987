#!/usr/bin/env python3
"""Package test predictions from a completed structured Phase-2 checkpoint."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED

import torch
from tokenizers import Tokenizer

from ner_core import LABELS, tokenize_document
from structured_ner import StructuredNER
from train_ner import sha256
from train_structured_ner import predict_documents
from src.model import LegalLMConfig


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--checkpoint', type=Path, required=True)
    ap.add_argument('--tokenizer', type=Path, required=True)
    ap.add_argument('--test', type=Path, required=True)
    ap.add_argument('--output-dir', type=Path, required=True)
    args = ap.parse_args()
    if args.output_dir.exists():
        ap.error('Output already exists')
    state = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    meta = state['meta']
    kind = meta['backbone_kind']
    config = (LegalLMConfig(**state['model_config']) if kind in ('dense', 'moe')
              else state['model_config'])
    model = StructuredNER(config, kind, meta['method'], meta['structure'])
    model.configure_adaptation()
    model.load_state_dict(state['model'], strict=True)
    del state
    model.cuda().eval()
    tokenizer = Tokenizer.from_file(str(args.tokenizer))
    test = json.loads(args.test.read_text())
    docs = [tokenize_document(k, v, tokenizer) for k, v in test.items()]
    prediction = predict_documents(model, docs, 512, 384, torch.device('cuda:0'), 32, .025)
    if set(prediction) != set(test):
        raise RuntimeError('Test IDs differ')
    for doc_id, spans in prediction.items():
        for span in spans:
            if (len(span) != 3 or not 0 <= span[0] < span[1] <= len(test[doc_id])
                    or span[2] not in LABELS):
                raise RuntimeError(f'Invalid span {doc_id}: {span}')
    args.output_dir.mkdir(parents=True)
    output = args.output_dir / 'prediction.json'
    output.write_text(json.dumps(prediction, ensure_ascii=False, indent=2) + '\n')
    archive = args.output_dir / 'submission.zip'
    with ZipFile(archive, 'w', ZIP_DEFLATED, compresslevel=9) as z:
        z.write(output, arcname='prediction.json')
    (args.output_dir / 'manifest.json').write_text(json.dumps({
        'checkpoint': str(args.checkpoint.resolve()),
        'checkpoint_sha256': sha256(args.checkpoint),
        'tokenizer_sha256': sha256(args.tokenizer),
        'test_sha256': sha256(args.test),
        'prediction_sha256': sha256(output),
        'submission_zip_sha256': sha256(archive),
        'documents': len(prediction),
        'predicted_entities': sum(map(len, prediction.values())),
        'labels': LABELS,
        'structure': meta['structure'],
    }, indent=2) + '\n')
    print(f'Saved {len(prediction)} test documents to {archive}', flush=True)


if __name__ == '__main__':
    main()
