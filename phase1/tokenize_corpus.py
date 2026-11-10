"""Encode prepared Phase-1 train/validation JSONL as packed token streams.

The files contain token IDs plus an explicit EOS after each document. Whole
documents were split in prepare_corpus.py, so no document crosses train and
validation. The official test set is never read here.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np
from tokenizers import Tokenizer


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def encode_split(tokenizer: Tokenizer, source: Path, destination: Path,
                 dtype: np.dtype, eos_id: int, batch_documents: int) -> dict:
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    documents = 0
    text_bytes = 0
    tokens = 0
    batch: list[str] = []

    def flush(output) -> None:
        nonlocal documents, tokens
        encodings = tokenizer.encode_batch(batch, add_special_tokens=False)
        for encoding in encodings:
            ids = encoding.ids
            if not ids:
                raise ValueError(f"Tokenizer returned no IDs for a document in {source}")
            np.asarray(ids + [eos_id], dtype=dtype).tofile(output)
            documents += 1
            tokens += len(ids) + 1
        batch.clear()

    with source.open("r", encoding="utf-8", buffering=8 * 1024 * 1024) as input_file, \
         temporary.open("wb", buffering=8 * 1024 * 1024) as output:
        for line in input_file:
            row = json.loads(line)
            text = row["text"]
            if not isinstance(text, str) or not text:
                raise ValueError(f"Invalid text in {source}")
            text_bytes += row["text_bytes"] if "text_bytes" in row else len(text.encode("utf-8"))
            batch.append(text)
            if len(batch) >= batch_documents:
                flush(output)
                if documents % 10000 < batch_documents:
                    print(f"{source.name}: {documents:,} documents, {tokens:,} tokens", flush=True)
        if batch:
            flush(output)
    os.replace(temporary, destination)
    expected_bytes = tokens * dtype.itemsize
    if destination.stat().st_size != expected_bytes:
        raise RuntimeError(f"Unexpected size for {destination}")
    return {"source": str(source.resolve()), "file": str(destination.resolve()),
            "documents": documents, "text_bytes": text_bytes,
            "tokens_including_eos": tokens}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus-dir", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--batch-documents", type=int, default=64)
    parser.add_argument("--cpu-threads", type=int, default=8)
    args = parser.parse_args()
    if args.batch_documents < 1 or args.cpu_threads < 1:
        parser.error("Batch documents and CPU threads must be positive")
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        parser.error(f"Output directory is not empty: {args.output_dir}")
    for split in ("train", "validation"):
        if not (args.corpus_dir / f"{split}.jsonl").is_file():
            parser.error(f"Missing {split}.jsonl in {args.corpus_dir}")
    if not args.tokenizer.is_file():
        parser.error(f"Missing tokenizer: {args.tokenizer}")

    os.environ["RAYON_NUM_THREADS"] = str(args.cpu_threads)
    os.environ["TOKENIZERS_PARALLELISM"] = "true"
    tokenizer = Tokenizer.from_file(str(args.tokenizer))
    vocab_size = tokenizer.get_vocab_size()
    eos_id = tokenizer.token_to_id("<eos>")
    pad_id = tokenizer.token_to_id("<pad>")
    if eos_id is None or pad_id is None:
        parser.error("Tokenizer needs <eos> and <pad> tokens")
    dtype = np.dtype("<u2" if vocab_size <= 65536 else "<u4")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    splits = {}
    for split in ("train", "validation"):
        splits[split] = encode_split(tokenizer, args.corpus_dir / f"{split}.jsonl",
                                     args.output_dir / f"{split}.bin", dtype,
                                     eos_id, args.batch_documents)
    manifest = {"tokenizer": str(args.tokenizer.resolve()),
                "tokenizer_sha256": sha256(args.tokenizer),
                "vocab_size": vocab_size, "eos_id": eos_id, "pad_id": pad_id,
                "dtype": dtype.str, "splits": splits}
    (args.output_dir / "token_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"Saved {args.output_dir / 'token_manifest.json'}", flush=True)


if __name__ == "__main__":
    main()
