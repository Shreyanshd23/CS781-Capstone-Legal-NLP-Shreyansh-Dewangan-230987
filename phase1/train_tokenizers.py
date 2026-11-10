"""Train comparable byte-level BPE and Unigram tokenizers on official train text.

Both tokenizers see the same prepared training sample. Validation text is used
only for diagnostics, never to learn vocabulary or merges.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
import multiprocessing
import os
from pathlib import Path
import statistics
import time


SPECIAL_TOKENS = ["<pad>", "<unk>", "<eos>"]


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def evaluate(tokenizer, path: Path, max_bytes: int) -> dict:
    total_bytes = 0
    words = 0
    tokens = 0
    unknowns = 0
    checked = 0
    roundtrip_failures = 0
    token_counts = []
    batch = []

    def consume(texts: list[str]) -> None:
        nonlocal words, tokens, unknowns, checked, roundtrip_failures
        unk_id = tokenizer.token_to_id("<unk>")
        for text, encoding in zip(texts, tokenizer.encode_batch(texts, add_special_tokens=False)):
            checked += 1
            words += len(text.split())
            tokens += len(encoding.ids)
            unknowns += encoding.ids.count(unk_id)
            token_counts.append(len(encoding.ids))
            if tokenizer.decode(encoding.ids, skip_special_tokens=False) != text:
                roundtrip_failures += 1

    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            size = len(line.encode("utf-8"))
            if total_bytes + size > max_bytes and checked:
                break
            total_bytes += size
            batch.append(line.removesuffix("\n"))
            if len(batch) == 128:
                consume(batch)
                batch = []
        if batch:
            consume(batch)
    return {"documents": checked, "raw_bytes": total_bytes,
            "words": words, "tokens": tokens,
            "tokens_per_word": tokens / words if words else None,
            "bytes_per_token": total_bytes / tokens if tokens else None,
            "unknown_tokens": unknowns,
            "roundtrip_failures": roundtrip_failures,
            "median_tokens_per_document": statistics.median(token_counts) if token_counts else None}


def train_one(algorithm: str, train_path: str, valid_path: str, output_dir: str,
              vocab_size: int, rayon_threads: int, evaluation_mib: int,
              sample_sha256: str) -> dict:
    # Set before importing tokenizers so each process gets its own Rayon pool.
    os.environ["RAYON_NUM_THREADS"] = str(rayon_threads)
    os.environ["TOKENIZERS_PARALLELISM"] = "true"
    import tokenizers
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers

    alphabet = pre_tokenizers.ByteLevel.alphabet()
    if algorithm == "bpe":
        tokenizer = Tokenizer(models.BPE(unk_token="<unk>"))
        trainer = trainers.BpeTrainer(
            vocab_size=vocab_size, min_frequency=2, initial_alphabet=alphabet,
            special_tokens=SPECIAL_TOKENS, max_token_length=24)
    elif algorithm == "unigram":
        tokenizer = Tokenizer(models.Unigram())
        trainer = trainers.UnigramTrainer(
            vocab_size=vocab_size, unk_token="<unk>", initial_alphabet=alphabet,
            special_tokens=SPECIAL_TOKENS, max_piece_length=24)
    else:
        raise ValueError(algorithm)
    # Keep case, punctuation and original Unicode. ByteLevel provides fallback
    # coverage without a pretrained tokenizer or outside text.
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()

    start = time.monotonic()
    tokenizer.train([train_path], trainer)
    elapsed = time.monotonic() - start
    folder = Path(output_dir) / algorithm
    folder.mkdir(parents=True, exist_ok=False)
    tokenizer_path = folder / "tokenizer.json"
    tokenizer.save(str(tokenizer_path))
    checks = {token: tokenizer.token_to_id(token) for token in SPECIAL_TOKENS}
    if any(value is None for value in checks.values()):
        raise RuntimeError(f"Missing special token in {algorithm}: {checks}")
    evaluation = evaluate(tokenizer, Path(valid_path), evaluation_mib * 1024 ** 2)
    if evaluation["roundtrip_failures"]:
        raise RuntimeError(f"{algorithm} failed {evaluation['roundtrip_failures']} round trips")
    metadata = {
        "algorithm": algorithm,
        "tokenizers_version": tokenizers.__version__,
        "vocabulary_size": tokenizer.get_vocab_size(),
        "requested_vocabulary_size": vocab_size,
        "special_token_ids": checks,
        "pre_tokenizer": "ByteLevel(add_prefix_space=False)",
        "decoder": "ByteLevel",
        "normalizer": None,
        "automatic_bos_or_eos": False,
        "training_sample_sha256": sample_sha256,
        "training_seconds": round(elapsed, 2),
        "validation_sample": evaluation,
    }
    (folder / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n",
                                          encoding="utf-8")
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--vocab-size", type=int, default=32000)
    parser.add_argument("--cpus", type=int, default=min(os.cpu_count() or 2, 16),
                        help="Total CPU threads across the two concurrent trainers")
    parser.add_argument("--evaluation-mib", type=int, default=16)
    args = parser.parse_args()
    if args.vocab_size < 300 or args.cpus < 1 or args.evaluation_mib < 1:
        parser.error("vocab-size >= 300, cpus >= 1 and evaluation-mib >= 1 required")
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        parser.error(f"Output directory is not empty: {args.output_dir}")
    train_path = args.corpus_dir / "train_tokenizer_sample.txt"
    valid_path = args.corpus_dir / "validation_tokenizer_sample.txt"
    for path in (train_path, valid_path):
        if not path.is_file() or path.stat().st_size == 0:
            parser.error(f"Missing or empty sample: {path}")
    sample_hash = file_sha256(train_path)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    algorithms = ("bpe", "unigram")
    workers = min(2, args.cpus)
    threads_per_worker = max(1, args.cpus // workers)
    with ProcessPoolExecutor(max_workers=workers,
                             mp_context=multiprocessing.get_context("spawn")) as pool:
        futures = [pool.submit(train_one, algorithm, str(train_path), str(valid_path),
                               str(args.output_dir), args.vocab_size,
                               threads_per_worker, args.evaluation_mib, sample_hash)
                   for algorithm in algorithms]
        results = [future.result() for future in futures]
    summary = {result["algorithm"]: result for result in results}
    manifest_path = args.corpus_dir / "manifest.json"
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        for result in results:
            bytes_per_token = result["validation_sample"]["bytes_per_token"]
            if bytes_per_token:
                result["estimated_corpus_tokens_from_validation_sample"] = {
                    split: round(sum(source["text_bytes"] for source in manifest[split]["sources"].values())
                                 / bytes_per_token)
                    for split in ("train", "validation")
                }
    (args.output_dir / "comparison.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    for result in results:
        val = result["validation_sample"]
        print(f"{result['algorithm']}: vocab={result['vocabulary_size']:,}, "
              f"validation tokens/word={val['tokens_per_word']:.3f}, "
              f"unknowns={val['unknown_tokens']}, train={result['training_seconds']:.1f}s")
        if "estimated_corpus_tokens_from_validation_sample" in result:
            estimate = result["estimated_corpus_tokens_from_validation_sample"]
            print(f"  Approximate train/validation tokens: "
                  f"{estimate['train']:,} / {estimate['validation']:,}")
    print(f"Saved comparison: {args.output_dir / 'comparison.json'}")


if __name__ == "__main__":
    main()
