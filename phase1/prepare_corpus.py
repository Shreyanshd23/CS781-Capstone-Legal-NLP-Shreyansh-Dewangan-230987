"""Fetch and prepare only the official CS-781 Phase-1 training files.

The starter notebook shows that each source is a JSON array of {"text": ...}
records. This script streams those arrays, splits whole documents, samples a
bounded corpus, and interleaves the four sources. It never reads the test set.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import random
import shutil
import sys
from typing import Iterator


REPO = "Exploration-Lab/CS-781-Capstone"
SOURCE_FILES = {
    "acts": "train_acts.json",
    "high_courts": "train_high_courts.json",
    "supreme_court": "train_supreme_court.json",
    "tribunals": "train_tribunals.json",
}
DEFAULT_WEIGHTS = {"acts": 0.10, "high_courts": 0.35,
                   "supreme_court": 0.40, "tribunals": 0.15}
MIB = 1024 ** 2
GIB = 1024 ** 3


def source_path(data_dir: Path, source: str) -> Path:
    filename = SOURCE_FILES[source]
    candidates = [
        data_dir / "Phase_1_Pretraining" / "Train" / filename,
        data_dir / "Train" / filename,
        data_dir / filename,
    ]
    for path in candidates:
        if path.is_file():
            return path
    raise FileNotFoundError(f"Missing {filename}; looked in: {candidates}")


def iter_json_array(path: Path) -> Iterator[dict]:
    """Stream a top-level JSON array; ijson is faster when installed.

    The fallback uses the C-backed stdlib JSON decoder and keeps only the
    current document in memory. It makes local smoke tests possible without
    installing packages.
    """
    try:
        import ijson
    except ImportError:
        ijson = None
    if ijson is not None:
        with path.open("rb", buffering=4 * MIB) as handle:
            yield from ijson.items(handle, "item")
        return

    decoder = json.JSONDecoder()
    with path.open("r", encoding="utf-8", buffering=4 * MIB) as handle:
        buffer = ""
        position = 0
        eof = False

        def read_more() -> None:
            nonlocal buffer, position, eof
            chunk = handle.read(4 * MIB)
            buffer = buffer[position:] + chunk
            position = 0
            eof = not chunk

        read_more()
        while True:
            while position < len(buffer) and buffer[position].isspace():
                position += 1
            if position < len(buffer):
                break
            if eof:
                raise ValueError(f"Empty JSON file: {path}")
            read_more()
        if buffer[position] != "[":
            raise ValueError(f"Expected a top-level JSON array in {path}")
        position += 1
        expect_value = True
        seen_record = False
        while True:
            while position < len(buffer) and buffer[position].isspace():
                position += 1
            if position == len(buffer):
                if eof:
                    raise ValueError(f"Unexpected end of JSON array: {path}")
                read_more()
                continue
            char = buffer[position]
            if char == "]":
                if expect_value and seen_record:
                    raise ValueError(f"Trailing comma in JSON array: {path}")
                position += 1
                if buffer[position:].strip() or handle.read().strip():
                    raise ValueError(f"Extra data after JSON array: {path}")
                return
            if not expect_value:
                if char != ",":
                    raise ValueError(f"Expected ',' in {path}")
                position += 1
                expect_value = True
                continue
            try:
                record, end = decoder.raw_decode(buffer, position)
            except json.JSONDecodeError as exc:
                if eof:
                    raise ValueError(f"Invalid JSON in {path}: {exc}") from exc
                read_more()
                continue
            position = end
            expect_value = False
            seen_record = True
            yield record
            if position > 4 * MIB:
                buffer = buffer[position:]
                position = 0


def record_text(record: object, path: Path, index: int) -> str | None:
    if not isinstance(record, dict):
        raise ValueError(f"{path}: record {index} is not an object")
    value = record.get("text")
    if value is None or value == "":
        return None
    if not isinstance(value, str):
        raise ValueError(f"{path}: record {index} has non-string text")
    return value if value.strip() else None


def scan_source(source: str, filename: str) -> dict:
    path = Path(filename)
    result = {"source": source, "file": str(path), "records": 0,
              "nonempty_records": 0, "empty_records": 0, "text_bytes": 0,
              "example_fields": None}
    for index, record in enumerate(iter_json_array(path)):
        result["records"] += 1
        if result["example_fields"] is None and isinstance(record, dict):
            result["example_fields"] = sorted(record.keys())
        text = record_text(record, path, index)
        if text is None:
            result["empty_records"] += 1
            continue
        result["nonempty_records"] += 1
        result["text_bytes"] += len(text.encode("utf-8"))
    return result


def allocate_bytes(target: int, weights: dict[str, float], capacity: dict[str, float]) -> dict[str, float]:
    if target > sum(capacity.values()):
        raise ValueError(f"Requested {target / GIB:.2f} GiB, but only "
                         f"{sum(capacity.values()) / GIB:.2f} GiB is available")
    allocation = {name: 0.0 for name in weights}
    remaining = set(weights)
    while remaining:
        budget = target - sum(allocation.values())
        if budget <= 0:
            break
        total_weight = sum(weights[name] for name in remaining)
        if total_weight <= 0:
            break
        saturated = set()
        for name in remaining:
            share = budget * weights[name] / total_weight
            if share >= capacity[name]:
                allocation[name] = capacity[name]
                saturated.add(name)
        if not saturated:
            for name in remaining:
                allocation[name] = budget * weights[name] / total_weight
            break
        remaining -= saturated
    return allocation


def split_numbers(text: str, seed: int) -> tuple[float, float, str]:
    raw = hashlib.blake2b(text.encode("utf-8"), digest_size=16,
                          key=seed.to_bytes(8, "little")).digest()
    denominator = 2 ** 64
    return (int.from_bytes(raw[:8], "big") / denominator,
            int.from_bytes(raw[8:], "big") / denominator, raw.hex())


def write_source(source: str, filename: str, parts_dir: str, seed: int,
                 validation_pool: float, train_rate: float, valid_rate: float) -> dict:
    path = Path(filename)
    folder = Path(parts_dir) / source
    folder.mkdir(parents=True, exist_ok=True)
    result = {"source": source, "train_records": 0, "train_bytes": 0,
              "validation_records": 0, "validation_bytes": 0}
    with (folder / "train.jsonl").open("w", encoding="utf-8", buffering=MIB) as train_out, \
         (folder / "validation.jsonl").open("w", encoding="utf-8", buffering=MIB) as valid_out:
        for index, record in enumerate(iter_json_array(path)):
            text = record_text(record, path, index)
            if text is None:
                continue
            split_u, pick_u, digest = split_numbers(text, seed)
            split = None
            if split_u < validation_pool:
                if pick_u < valid_rate:
                    split = "validation"
                    output = valid_out
            elif pick_u < train_rate:
                split = "train"
                output = train_out
            if split is None:
                continue
            count = len(text.encode("utf-8"))
            document_id = record.get("id", record.get("doc_id", index))
            row = {"source": source, "id": str(document_id),
                   "text": text, "text_bytes": count, "fingerprint": digest}
            output.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
            result[f"{split}_records"] += 1
            result[f"{split}_bytes"] += count
    return result


def buffered_rows(path: Path, rng: random.Random, buffer_size: int) -> Iterator[dict]:
    with path.open("r", encoding="utf-8", buffering=MIB) as handle:
        buffer = []
        for _ in range(buffer_size):
            line = handle.readline()
            if not line:
                break
            buffer.append(json.loads(line))
        while buffer:
            index = rng.randrange(len(buffer))
            row = buffer[index]
            line = handle.readline()
            if line:
                buffer[index] = json.loads(line)
            else:
                buffer[index] = buffer[-1]
                buffer.pop()
            yield row


def mix_split(split: str, parts_dir: Path, output_dir: Path, selected: dict,
              seed: int, buffer_size: int, sample_target: int) -> dict:
    rng = random.Random(seed + (1 if split == "validation" else 0))
    iterators = {name: buffered_rows(parts_dir / name / f"{split}.jsonl", rng, buffer_size)
                 for name in SOURCE_FILES}
    remaining = {name: selected[name][f"{split}_bytes"] for name in SOURCE_FILES}
    total = sum(remaining.values())
    sample_rate = min(1.0, sample_target / total) if total else 0.0
    counts = {name: {"records": 0, "text_bytes": 0} for name in SOURCE_FILES}
    sample_bytes = 0
    mixed_path = output_dir / f"{split}.jsonl"
    sample_path = output_dir / f"{split}_tokenizer_sample.txt"
    with mixed_path.open("w", encoding="utf-8", buffering=MIB) as mixed, \
         sample_path.open("w", encoding="utf-8", buffering=MIB) as sample:
        active = {name for name in SOURCE_FILES if remaining[name] > 0}
        while active:
            names = sorted(active)
            name = rng.choices(names, weights=[max(1, remaining[n]) for n in names])[0]
            try:
                row = next(iterators[name])
            except StopIteration:
                active.remove(name)
                continue
            mixed.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
            size = row["text_bytes"]
            remaining[name] -= size
            counts[name]["records"] += 1
            counts[name]["text_bytes"] += size
            # Independent deterministic selection; validation text never trains a tokenizer.
            h = hashlib.blake2b(bytes.fromhex(row["fingerprint"]), digest_size=8,
                                key=b"tok-samp").digest()
            if int.from_bytes(h, "big") / 2 ** 64 < sample_rate:
                line = row["text"].replace("\r", " ").replace("\n", " ") + "\n"
                sample.write(line)
                sample_bytes += len(line.encode("utf-8"))
    return {"file": str(mixed_path), "tokenizer_sample": str(sample_path),
            "sample_bytes": sample_bytes, "sources": counts}


def parse_weights(raw: str) -> dict[str, float]:
    values = [float(item) for item in raw.split(",")]
    if len(values) != 4 or any(x < 0 for x in values) or sum(values) <= 0:
        raise ValueError("--mix needs four nonnegative weights with a positive sum")
    return dict(zip(SOURCE_FILES, values))


def fetch(args: argparse.Namespace) -> None:
    from huggingface_hub import HfApi, hf_hub_download
    args.data_dir.mkdir(parents=True, exist_ok=True)
    revision = HfApi().dataset_info(REPO, revision=args.revision).sha
    def download(name: str) -> tuple[str, str]:
        filename = f"Phase_1_Pretraining/Train/{SOURCE_FILES[name]}"
        path = hf_hub_download(repo_id=REPO, filename=filename, repo_type="dataset",
                               revision=revision, local_dir=args.data_dir)
        return name, path
    downloaded = {}
    with ThreadPoolExecutor(max_workers=min(args.workers, 4)) as pool:
        for name, path in pool.map(download, SOURCE_FILES):
            print(f"{name}: {path}")
            downloaded[name] = {"path": path, "file_bytes": Path(path).stat().st_size}
    (args.data_dir / "download_manifest.json").write_text(
        json.dumps({"repo": REPO, "revision": revision, "files": downloaded}, indent=2)
        + "\n", encoding="utf-8")
    print(f"Pinned official dataset revision: {revision}")


def prepare(args: argparse.Namespace) -> None:
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {args.output_dir}")
    if not 0 < args.validation_pool < 1:
        raise ValueError("--validation-pool must be between 0 and 1")
    paths = {name: str(source_path(args.data_dir, name)) for name in SOURCE_FILES}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    parts_dir = args.output_dir / "_parts"
    parts_dir.mkdir()
    with ProcessPoolExecutor(max_workers=min(args.workers, 4)) as pool:
        scans = list(pool.map(scan_source, SOURCE_FILES, paths.values()))
    stats = {item["source"]: item for item in scans}
    for name, item in stats.items():
        print(f"{name}: {item['nonempty_records']:,} documents, "
              f"{item['text_bytes'] / GIB:.2f} GiB text; fields={item['example_fields']}",
              flush=True)
    weights = parse_weights(args.mix)
    train_target = round(args.train_gib * GIB)
    valid_target = round(args.validation_mib * MIB)
    train_quota = allocate_bytes(train_target, weights, {
        name: item["text_bytes"] * (1 - args.validation_pool)
        for name, item in stats.items()})
    valid_quota = allocate_bytes(valid_target, weights, {
        name: item["text_bytes"] * args.validation_pool
        for name, item in stats.items()})
    jobs = []
    for name in SOURCE_FILES:
        total = stats[name]["text_bytes"]
        train_rate = train_quota[name] / (total * (1 - args.validation_pool)) if total else 0
        valid_rate = valid_quota[name] / (total * args.validation_pool) if total else 0
        jobs.append((name, paths[name], str(parts_dir), args.seed,
                     args.validation_pool, train_rate, valid_rate))
    with ProcessPoolExecutor(max_workers=min(args.workers, 4)) as pool:
        selections = list(pool.map(lambda_unpack_write_source, jobs))
    selected = {item["source"]: item for item in selections}
    train = mix_split("train", parts_dir, args.output_dir, selected, args.seed,
                      args.shuffle_buffer, round(args.tokenizer_sample_mib * MIB))
    validation = mix_split("validation", parts_dir, args.output_dir, selected,
                           args.seed, args.shuffle_buffer,
                           round(args.validation_tokenizer_sample_mib * MIB))
    manifest = {"repo": REPO, "seed": args.seed, "mix_weights": weights,
                "validation_pool": args.validation_pool,
                "targets": {"train_text_bytes": train_target,
                            "validation_text_bytes": valid_target},
                "input": stats, "selected": selected,
                "train": train, "validation": validation}
    download_manifest = args.data_dir / "download_manifest.json"
    if download_manifest.is_file():
        manifest["download"] = json.loads(download_manifest.read_text(encoding="utf-8"))
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if not args.keep_shards:
        shutil.rmtree(parts_dir)
    print(f"Wrote {args.output_dir / 'manifest.json'}")
    print(f"Selected train {sum(x['train_bytes'] for x in selections) / GIB:.2f} GiB; "
          f"validation {sum(x['validation_bytes'] for x in selections) / MIB:.1f} MiB")


def lambda_unpack_write_source(job: tuple) -> dict:
    return write_source(*job)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    fetch_cmd = commands.add_parser("fetch", help="Download the four official train files")
    fetch_cmd.add_argument("--data-dir", type=Path, required=True)
    fetch_cmd.add_argument("--revision", default="main")
    fetch_cmd.add_argument("--workers", type=int, default=4)
    prep = commands.add_parser("prepare", help="Split and mix the training corpus")
    prep.add_argument("--data-dir", type=Path, required=True)
    prep.add_argument("--output-dir", type=Path, required=True)
    prep.add_argument("--train-gib", type=float, default=2.0)
    prep.add_argument("--validation-mib", type=float, default=48.0)
    prep.add_argument("--tokenizer-sample-mib", type=float, default=128.0)
    prep.add_argument("--validation-tokenizer-sample-mib", type=float, default=16.0)
    prep.add_argument("--mix", default="0.10,0.35,0.40,0.15",
                      help="Acts, High Courts, Supreme Court, Tribunals byte weights")
    prep.add_argument("--validation-pool", type=float, default=0.05)
    prep.add_argument("--seed", type=int, default=42)
    prep.add_argument("--workers", type=int, default=4)
    prep.add_argument("--shuffle-buffer", type=int, default=128)
    prep.add_argument("--keep-shards", action="store_true")
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be positive")
    try:
        if args.command == "fetch":
            fetch(args)
        else:
            if args.shuffle_buffer < 1:
                parser.error("--shuffle-buffer must be positive")
            prepare(args)
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise


if __name__ == "__main__":
    main()
