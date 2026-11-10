"""Exact-offset legal NER using the Phase-1 decoder's hidden states."""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F
from tokenizers import Tokenizer

from src.model import LegalLM, LegalLMConfig
from src.model.legal_moe import LegalMoELM
from src.model.legal_ul2 import build_legal_ul2
from src.model.rmsnorm import RMSNorm
from transformers import GPT2Config, GPT2LMHeadModel

LABELS = ('APP', 'RESP', 'A.COUNSEL', 'R.COUNSEL', 'JUDGE', 'WIT',
          'AUTH', 'COURT', 'STAT', 'PREC', 'DATE', 'CASENO')
TAGS = ('O',) + tuple(f'{prefix}-{label}' for label in LABELS for prefix in ('B', 'I'))
TAG_TO_ID = {tag: i for i, tag in enumerate(TAGS)}


def read_folds(root: Path, folds=(1, 2)):
    rows = {}
    for fold in folds:
        train = root / 'Train'
        texts = json.loads((train / f'fold{fold}-text.json').read_text())
        gold = json.loads((train / f'fold{fold}-gold.json').read_text())
        if set(texts) != set(gold):
            raise ValueError(f'Fold {fold} text and gold IDs differ')
        for doc_id, text in texts.items():
            entities = gold[doc_id]
            for start, end, label, entity_text in entities:
                if not 0 <= start < end <= len(text) or text[start:end] != entity_text:
                    raise ValueError(f'Bad annotation in {doc_id}: {start}:{end}')
                if label not in LABELS:
                    raise ValueError(f'Unknown label {label}')
            rows[doc_id] = {'text': text, 'entities': entities, 'fold': fold}
    return rows


def normalized_offsets(text: str, offsets):
    result = []
    for start, end in offsets:
        while start < end and text[start].isspace():
            start += 1
        while start < end and text[end - 1].isspace():
            end -= 1
        result.append((start, end))
    return result


@dataclass
class DocumentTokens:
    doc_id: str
    text: str
    ids: list[int]
    offsets: list[tuple[int, int]]
    tags: list[int] | None
    fold: int | None = None
    aligned: int = 0
    total: int = 0


def tokenize_document(doc_id: str, text: str, tokenizer: Tokenizer,
                      entities=None, fold=None) -> DocumentTokens:
    encoding = tokenizer.encode(text, add_special_tokens=False)
    offsets = normalized_offsets(text, encoding.offsets)
    tags = [TAG_TO_ID['O']] * len(encoding.ids) if entities is not None else None
    aligned = 0
    if entities is not None:
        starts = {start: index for index, (start, end) in enumerate(offsets)
                  if start < end}
        ends = {end: index for index, (start, end) in enumerate(offsets)
                if start < end}
        for start, end, label, _ in entities:
            first, last = starts.get(start), ends.get(end)
            if first is None or last is None or first > last:
                # These boundary cases cannot be represented by BPE tags.
                # Ignore intersecting tokens in the training loss.
                for index, (a, b) in enumerate(offsets):
                    if a < end and b > start:
                        tags[index] = -100
                continue
            aligned += 1
            tags[first] = TAG_TO_ID[f'B-{label}']
            for index in range(first + 1, last + 1):
                tags[index] = TAG_TO_ID[f'I-{label}']
    return DocumentTokens(doc_id, text, encoding.ids, offsets, tags,
                          fold, aligned, len(entities or []))


def windows(document: DocumentTokens, length: int = 1024,
            stride: int = 768):
    if length < 1 or stride < 1 or stride > length:
        raise ValueError('Invalid window length or stride')
    size = len(document.ids)
    for start in range(0, size, stride):
        end = min(start + length, size)
        yield {'doc_id': document.doc_id, 'start': start,
               'ids': document.ids[start:end],
               'tags': None if document.tags is None else document.tags[start:end]}
        if end == size:
            break


def collate(batch, pad_id: int = 0):
    width = max(len(row['ids']) for row in batch)
    ids = torch.full((len(batch), width), pad_id, dtype=torch.long)
    tags = torch.full((len(batch), width), -100, dtype=torch.long)
    for index, row in enumerate(batch):
        n = len(row['ids'])
        ids[index, :n] = torch.tensor(row['ids'])
        if row['tags'] is not None:
            tags[index, :n] = torch.tensor(row['tags'])
    return ids, tags, [len(row['ids']) for row in batch]


class LowRankLinear(nn.Module):
    """Frozen base projection with additive LoRA or multiplicative LoRMA+."""

    def __init__(self, base: nn.Linear, method: str, rank: int = 8, alpha: int = 16):
        super().__init__()
        if method not in ('lora', 'lorma_plus'):
            raise ValueError(method)
        self.base = base
        self.method = method
        self.scale = alpha / rank
        width = base.in_features if method == 'lora' else base.out_features
        self.a = nn.Linear(width, rank, bias=False)
        self.b = nn.Linear(rank, base.out_features, bias=False)
        nn.init.kaiming_uniform_(self.a.weight, a=math.sqrt(5))
        nn.init.zeros_(self.b.weight)
        self.base.requires_grad_(False)

    def forward(self, x):
        base_output = self.base(x)
        adapter_input = x if self.method == 'lora' else base_output
        return base_output + self.b(self.a(adapter_input)) * self.scale


class AdapterFFN(nn.Module):
    """Frozen SwiGLU plus a learned bottleneck residual adapter."""

    def __init__(self, base: nn.Module, width: int, bottleneck: int = 64):
        super().__init__()
        self.base = base.requires_grad_(False)
        self.down = nn.Linear(width, bottleneck, bias=True)
        self.up = nn.Linear(bottleneck, width, bias=True)
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)

    def forward(self, x):
        return self.base(x) + self.up(F.gelu(self.down(x)))


class DecoderNER(nn.Module):
    def __init__(self, config: LegalLMConfig | dict, backbone_kind: str = 'dense',
                 method: str = 'full'):
        super().__init__()
        if backbone_kind not in ('dense', 'moe', 'gpt2', 'ul2'):
            raise ValueError(backbone_kind)
        if backbone_kind == 'dense':
            self.backbone = LegalLM(config)
            width = config.d_model
        elif backbone_kind == 'moe':
            self.backbone = LegalMoELM(config)
            width = config.d_model
        elif backbone_kind == 'gpt2':
            hf_config = GPT2Config(
                vocab_size=config['vocab_size'], n_positions=config['block_size'],
                n_ctx=config['block_size'], n_embd=config['n_embd'],
                n_layer=config['n_layer'], n_head=config['n_head'],
                n_inner=config['n_inner'], activation_function=config['activation'],
                layer_norm_epsilon=config['layer_norm_epsilon'],
                resid_pdrop=config['dropout'], embd_pdrop=config['dropout'],
                attn_pdrop=config['dropout'], bos_token_id=config['bos_model_id'],
                eos_token_id=config['special_token_ids']['<eos>'],
                pad_token_id=config['special_token_ids']['<pad>'],
                use_cache=False)
            self.backbone = GPT2LMHeadModel(hf_config)
            width = hf_config.n_embd
        else:
            self.backbone = build_legal_ul2(config['vocab_size'],
                                            config['pad_id'], config['eos_id'])
            width = self.backbone.config.d_model
        self.classifier = nn.Linear(width, len(TAGS))
        self.backbone_kind = backbone_kind
        self.method = method

    def configure_adaptation(self):
        method = self.method
        if method in ('full', 'distillation'):
            return
        self.backbone.requires_grad_(False)
        if method == 'encoder_full' and self.backbone_kind == 'ul2':
            self.backbone.encoder.requires_grad_(True)
            self.backbone.shared.requires_grad_(True)
        elif method == 'partial':
            if self.backbone_kind == 'gpt2':
                for block in self.backbone.transformer.h[-4:]:
                    block.requires_grad_(True)
                self.backbone.transformer.ln_f.requires_grad_(True)
            elif self.backbone_kind == 'ul2':
                for block in self.backbone.encoder.block[-4:]:
                    block.requires_grad_(True)
                self.backbone.encoder.final_layer_norm.requires_grad_(True)
            else:
                for block in self.backbone.blocks[-4:]:
                    block.requires_grad_(True)
                self.backbone.final_norm.requires_grad_(True)
        elif method in ('lora', 'lorma_plus'):
            for block in self.backbone.blocks:
                for name in ('q_proj', 'v_proj'):
                    setattr(block.attn, name,
                            LowRankLinear(getattr(block.attn, name), method))
                if self.backbone_kind == 'dense':
                    for name in ('gate', 'up', 'down'):
                        setattr(block.ffn, name,
                                LowRankLinear(getattr(block.ffn, name), method))
        elif method == 'adapter':
            for block in self.backbone.blocks:
                block.ffn = AdapterFFN(block.ffn, self.backbone.config.d_model)
        elif method == 'norm':
            for module in self.backbone.modules():
                if isinstance(module, RMSNorm):
                    module.requires_grad_(True)
        else:
            raise ValueError(method)

    def encode(self, ids: torch.Tensor) -> torch.Tensor:
        if self.backbone_kind == 'gpt2':
            return self.backbone.transformer(
                input_ids=ids, use_cache=False, return_dict=True).last_hidden_state
        if self.backbone_kind == 'ul2':
            return self.backbone.encoder(
                input_ids=ids, attention_mask=ids.ne(0),
                return_dict=True).last_hidden_state
        batch, length = ids.shape
        positions = torch.arange(length, device=ids.device).expand(batch, -1)
        hidden = self.backbone.embedding_dropout(self.backbone.token_embedding(ids))
        for block in self.backbone.blocks:
            hidden, _ = block(hidden, positions)
        hidden = self.backbone.final_norm(hidden)
        return hidden

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.encode(ids))


def decode_tags(tags: list[int], offsets: list[tuple[int, int]], text: str):
    if len(tags) != len(offsets):
        raise ValueError('Tags and offsets differ')
    entities = []
    active = None
    for tag_id, (start, end) in zip(tags, offsets):
        tag = TAGS[tag_id]
        if start == end or tag == 'O':
            if active is not None:
                entities.append(active)
                active = None
            continue
        prefix, label = tag.split('-', 1)
        if active is None or prefix == 'B' or active[2] != label:
            if active is not None:
                entities.append(active)
            active = [start, end, label]
        else:
            active[1] = end
    if active is not None:
        entities.append(active)
    return [[a, b, label] for a, b, label in entities
            if 0 <= a < b <= len(text) and text[a:b].strip()]


def strict_macro_f1(predictions: dict, gold: dict):
    values = {}
    for label in LABELS:
        p = {(doc_id, a, b) for doc_id, rows in predictions.items()
             for a, b, name in rows if name == label}
        g = {(doc_id, a, b) for doc_id, rows in gold.items()
             for a, b, name, *_ in rows if name == label}
        tp = len(p & g)
        precision = tp / len(p) if p else 0.0
        recall = tp / len(g) if g else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        values[label] = {'precision': precision, 'recall': recall, 'f1': f1,
                         'predicted': len(p), 'gold': len(g), 'true_positive': tp}
    return {'macro_f1': sum(v['f1'] for v in values.values()) / len(values),
            'per_label': values}
