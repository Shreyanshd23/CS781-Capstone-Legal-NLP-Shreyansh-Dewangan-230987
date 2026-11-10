#!/usr/bin/env python3
"""Generate exact-ID Phase-2 NER predictions and a Codabench ZIP."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

import torch
from tokenizers import Tokenizer

from ner_core import DecoderNER, LABELS, TAGS, tokenize_document
from train_ner import predict_documents
from src.model import LegalLMConfig


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--tokenizer', type=Path, required=True)
    parser.add_argument('--test', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--length', type=int, default=1024)
    parser.add_argument('--stride', type=int, default=768)
    args = parser.parse_args()
    if args.output_dir.exists():
        parser.error(f'Output exists: {args.output_dir}')
    state = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    if tuple(state['tags']) != TAGS:
        parser.error('Checkpoint label vocabulary changed')
    meta = state['meta']
    config = (LegalLMConfig(**state['model_config'])
              if meta.get('backbone_kind', 'dense') in ('dense', 'moe')
              else state['model_config'])
    model = DecoderNER(config, meta.get('backbone_kind', 'dense'),
                       meta.get('method', 'full'))
    model.configure_adaptation()
    model.load_state_dict(state['model'], strict=True)
    del state
    model.cuda().eval()
    tokenizer = Tokenizer.from_file(str(args.tokenizer))
    test = json.loads(args.test.read_text())
    if not isinstance(test, dict) or not test:
        parser.error('Expected nonempty ID-to-text test object')
    if any(not isinstance(k, str) or not isinstance(v, str) for k, v in test.items()):
        parser.error('Invalid test record')
    documents = [tokenize_document(k, v, tokenizer) for k, v in test.items()]
    prediction = predict_documents(model, documents, args.length, args.stride,
                                   torch.device('cuda:0'))
    if set(prediction) != set(test):
        raise RuntimeError('Predicted test IDs differ')
    for doc_id, entities in prediction.items():
        text = test[doc_id]
        for record in entities:
            if (len(record) != 3 or not 0 <= record[0] < record[1] <= len(text)
                    or record[2] not in LABELS):
                raise RuntimeError(f'Invalid span in {doc_id}: {record}')
    args.output_dir.mkdir(parents=True)
    destination = args.output_dir / 'prediction.json'
    destination.write_text(json.dumps(prediction, ensure_ascii=False, indent=2) + '\n')
    archive = args.output_dir / 'submission.zip'
    with ZipFile(archive, 'w', compression=ZIP_DEFLATED, compresslevel=9) as z:
        z.write(destination, arcname='prediction.json')
    with ZipFile(archive) as z:
        if z.namelist() != ['prediction.json'] or z.testzip() is not None:
            raise RuntimeError('Invalid submission ZIP')
    def digest(path):
        return hashlib.sha256(path.read_bytes()).hexdigest()
    manifest = {'checkpoint': str(args.checkpoint.resolve()),
                'checkpoint_sha256': digest(args.checkpoint),
                'tokenizer_sha256': digest(args.tokenizer),
                'test_sha256': digest(args.test),
                'prediction_sha256': digest(destination),
                'submission_zip_sha256': digest(archive),
                'documents': len(prediction),
                'predicted_entities': sum(map(len, prediction.values())),
                'labels': LABELS,
                'format': 'document_id -> [[start,end,label], ...]'}
    (args.output_dir / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(f'Saved {len(prediction)} documents to {destination} and {archive}', flush=True)


if __name__ == '__main__':
    main()
