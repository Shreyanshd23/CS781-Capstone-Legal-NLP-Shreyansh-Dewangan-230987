#!/usr/bin/env python3
"""Index every official BPE document for a 70/20/10 legal UL2 mixture."""

from __future__ import annotations

import argparse
from array import array
import json
from pathlib import Path

import numpy as np
from tokenizers import Tokenizer

from train_seq2seq_ddp import sha256


MODE_TOKENS = ("<legal_continuation>", "<legal_span>", "<legal_extreme>")
SENTINELS = tuple(f"<extra_id_{i}>" for i in range(128))
RECORD_DTYPE = np.dtype([("start", "<u4"), ("length", "<u2"), ("mode", "u1")])
MAX_TEXT = (2558, 2048, 896)


def document_ends(tokens: np.memmap, eos_id: int) -> np.ndarray:
    parts = []
    chunk_size = 50_000_000
    for offset in range(0, len(tokens), chunk_size):
        part = np.asarray(tokens[offset:offset + chunk_size])
        parts.append(np.flatnonzero(part == eos_id).astype(np.int64) + offset)
    ends = np.concatenate(parts)
    if int(tokens[-1]) != eos_id:
        raise ValueError("Packed corpus does not end with a document EOS")
    return ends


def index_split(tokens_path: Path, dtype: str, eos_id: int, output: Path,
                seed: int, validation: bool):
    tokens = np.memmap(tokens_path, dtype=dtype, mode="r")
    ends = document_ends(tokens, eos_id)
    assignments = (np.zeros(len(ends), dtype=np.uint8) if validation else
                   np.random.default_rng(seed).integers(0, 100, size=len(ends),
                                                       dtype=np.uint8))
    starts = array("I")
    lengths = array("H")
    modes = array("B")
    mode_documents = [0, 0, 0]
    mode_tokens = [0, 0, 0]
    covered = 0
    previous = 0
    nonempty_documents = 0
    for index, end in enumerate(ends):
        begin = previous
        previous = int(end) + 1
        doc_length = int(end) - begin
        if doc_length == 0:
            continue
        nonempty_documents += 1
        assignment = int(assignments[index])
        mode = 0 if validation or assignment < 70 else (1 if assignment < 90 else 2)
        mode_documents[mode] += 1
        mode_tokens[mode] += doc_length
        covered += doc_length
        width = MAX_TEXT[mode]
        for start in range(begin, int(end), width):
            length = min(width, int(end) - start)
            starts.append(start)
            lengths.append(length)
            modes.append(mode)
    records = np.empty(len(starts), dtype=RECORD_DTYPE)
    records["start"] = np.frombuffer(starts, dtype=np.uint32)
    records["length"] = np.frombuffer(lengths, dtype=np.uint16)
    records["mode"] = np.frombuffer(modes, dtype=np.uint8)
    np.save(output, records)
    expected = len(tokens) - len(ends)
    if covered != expected or sum(records["length"].astype(np.uint64)) != expected:
        raise RuntimeError("Record index did not cover every non-EOS token")
    result = {"packed_file": str(tokens_path.resolve()), "packed_tokens_including_eos": len(tokens),
              "eos_documents": len(ends), "nonempty_documents": nonempty_documents,
              "non_eos_tokens_covered_once": covered, "records": len(records),
              "mode_documents": dict(zip(("continuation", "span", "extreme"), mode_documents)),
              "mode_tokens": dict(zip(("continuation", "span", "extreme"), mode_tokens)),
              "records_sha256": sha256(output)}
    print(output, json.dumps(result), flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tokens-dir", type=Path,
                        default=Path("data/tokens_bpe_full_15p95gib"))
    parser.add_argument("--base-tokenizer", type=Path,
                        default=Path("artifacts/tokenizers_512m_32k/bpe/tokenizer.json"))
    parser.add_argument("--output-dir", type=Path,
                        default=Path("data/legal_ul2_full_20261006"))
    parser.add_argument("--seed", type=int, default=781)
    args = parser.parse_args()
    manifest_path = args.tokens_dir / "token_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if sha256(args.base_tokenizer) != manifest["tokenizer_sha256"]:
        parser.error("Base BPE differs from packed official corpus")
    if args.output_dir.exists():
        parser.error(f"Output directory already exists: {args.output_dir}")
    args.output_dir.mkdir(parents=True)
    tokenizer = Tokenizer.from_file(str(args.base_tokenizer))
    added = tokenizer.add_special_tokens(list(MODE_TOKENS + SENTINELS))
    if added != len(MODE_TOKENS) + len(SENTINELS):
        raise RuntimeError(f"Expected 131 new special tokens, added {added}")
    tokenizer_path = args.output_dir / "tokenizer.json"
    tokenizer.save(str(tokenizer_path))
    train = index_split(args.tokens_dir / "train.bin", manifest["dtype"],
                        manifest["eos_id"], args.output_dir / "train_records.npy",
                        args.seed, validation=False)
    validation = index_split(args.tokens_dir / "validation.bin", manifest["dtype"],
                             manifest["eos_id"], args.output_dir / "validation_records.npy",
                             args.seed + 1, validation=True)
    metadata = {"seed": args.seed, "base_tokenizer_sha256": sha256(args.base_tokenizer),
                "tokenizer_sha256": sha256(tokenizer_path),
                "token_manifest_sha256": sha256(manifest_path),
                "vocab_size": tokenizer.get_vocab_size(),
                "mode_ids": {name: tokenizer.token_to_id(name) for name in MODE_TOKENS},
                "sentinel_ids": [tokenizer.token_to_id(name) for name in SENTINELS],
                "pad_id": manifest["pad_id"], "eos_id": manifest["eos_id"],
                "objectives": {"continuation": 0.70, "span_corruption": 0.20,
                               "extreme_corruption": 0.10},
                "max_encoder_tokens": 2048, "max_decoder_tokens": 512,
                "train": train, "validation": validation}
    (args.output_dir / "manifest.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(f"Prepared all official BPE documents; vocab={metadata['vocab_size']}", flush=True)


if __name__ == "__main__":
    main()
