"""Document-contained continuation and span corruption examples for Legal-UL2."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from prepare_legal_ul2 import MODE_TOKENS


def corruption(tokens: np.ndarray, mode_id: int, sentinel_ids: list[int],
               eos_id: int, seed: int, extreme: bool):
    rng = np.random.default_rng(seed)
    n = len(tokens)
    mask_count = min(n, max(1, int(round(n * (0.50 if extreme else 0.15)))))
    mean_span = 32 if extreme else 5
    span_count = min(len(sentinel_ids) - 1, mask_count,
                     max(1, int(round(mask_count / mean_span))))
    if span_count == 1:
        masked_lengths = np.array([mask_count], dtype=np.int32)
    else:
        cuts = np.sort(rng.choice(np.arange(1, mask_count), span_count - 1,
                                  replace=False))
        masked_lengths = np.diff(np.concatenate(([0], cuts, [mask_count])))
    gaps = rng.multinomial(n - mask_count,
                           np.full(span_count + 1, 1 / (span_count + 1)))
    source = [mode_id]
    target = []
    cursor = 0
    for i, masked_length in enumerate(masked_lengths):
        gap = int(gaps[i])
        source.extend(int(x) for x in tokens[cursor:cursor + gap])
        cursor += gap
        sentinel = sentinel_ids[i]
        source.append(sentinel)
        target.append(sentinel)
        target.extend(int(x) for x in tokens[cursor:cursor + int(masked_length)])
        cursor += int(masked_length)
    source.extend(int(x) for x in tokens[cursor:])
    target.extend((sentinel_ids[span_count], eos_id))
    if len(source) > 2048 or len(target) > 512:
        raise RuntimeError(f"Corruption exceeds limits: {len(source)} / {len(target)}")
    return source, target


class LegalUL2Examples(Dataset):
    def __init__(self, tokens_file: Path, records_file: Path, dtype: str,
                 metadata_file: Path, seed: int = 781):
        self.tokens_file = tokens_file
        self.records_file = records_file
        self.dtype = dtype
        self.metadata_file = metadata_file
        self.seed = seed
        self.metadata = json.loads(metadata_file.read_text())
        self.tokens = np.memmap(tokens_file, dtype=dtype, mode="r")
        self.records = np.load(records_file, mmap_mode="r")
        self.mode_ids = [self.metadata["mode_ids"][name] for name in MODE_TOKENS]
        self.sentinel_ids = self.metadata["sentinel_ids"]
        self.eos_id = self.metadata["eos_id"]
        self.pad_id = self.metadata["pad_id"]

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        record = self.records[index]
        start = int(record["start"])
        length = int(record["length"])
        mode = int(record["mode"])
        tokens = self.tokens[start:start + length]
        if mode == 0:
            target_length = min(511, max(1, int(round(length * 0.20))))
            source = [self.mode_ids[0]] + [int(x) for x in tokens[:-target_length]]
            target = [int(x) for x in tokens[-target_length:]] + [self.eos_id]
        else:
            source, target = corruption(tokens, self.mode_ids[mode],
                                        self.sentinel_ids, self.eos_id,
                                        self.seed + int(index) * 1000003,
                                        extreme=mode == 2)
        if not 0 < len(source) <= 2048 or not 0 < len(target) <= 512:
            raise RuntimeError(f"Invalid example lengths {len(source)}, {len(target)}")
        return source, target

    def __getstate__(self):
        state = self.__dict__.copy()
        state["tokens"] = state["records"] = None
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self.tokens = np.memmap(self.tokens_file, dtype=self.dtype, mode="r")
        self.records = np.load(self.records_file, mmap_mode="r")


def collate_ul2(batch, pad_id: int):
    batch_size = len(batch)
    source_width = min(2048, ((max(len(pair[0]) for pair in batch) + 7) // 8) * 8)
    target_width = min(512, ((max(len(pair[1]) for pair in batch) + 7) // 8) * 8)
    source = torch.full((batch_size, source_width), pad_id, dtype=torch.long)
    mask = torch.zeros((batch_size, source_width), dtype=torch.long)
    labels = torch.full((batch_size, target_width), -100, dtype=torch.long)
    for row, (input_ids, target_ids) in enumerate(batch):
        source[row, :len(input_ids)] = torch.tensor(input_ids, dtype=torch.long)
        mask[row, :len(input_ids)] = 1
        labels[row, :len(target_ids)] = torch.tensor(target_ids, dtype=torch.long)
    return source, mask, labels
