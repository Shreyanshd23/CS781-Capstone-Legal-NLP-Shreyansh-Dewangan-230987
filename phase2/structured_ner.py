"""Three genuinely different NER heads on the existing Phase-1 backbone."""
from __future__ import annotations

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from ner_core import DecoderNER, LABELS, TAGS

BILOU = ('O',) + tuple(f'{p}-{label}' for label in LABELS for p in ('B', 'I', 'L', 'U'))
BILOU_ID = {s: i for i, s in enumerate(BILOU)}
LABEL_ID = {s: i + 1 for i, s in enumerate(LABELS)}


def gold_spans(tags):
    """Extract BPE-aligned spans from one BIO window; end index is inclusive."""
    result = []
    i = 0
    while i < len(tags):
        tag = int(tags[i])
        if tag <= 0 or not TAGS[tag].startswith('B-'):
            i += 1
            continue
        label = TAGS[tag][2:]
        j = i
        while j + 1 < len(tags) and int(tags[j + 1]) >= 0 and TAGS[int(tags[j + 1])] == f'I-{label}':
            j += 1
        result.append((i, j, LABEL_ID[label]))
        i = j + 1
    return result


def bilou_tags(tags):
    output = [0 if int(t) >= 0 else -100 for t in tags]
    for i, j, label_id in gold_spans(tags):
        label = LABELS[label_id - 1]
        if i == j:
            output[i] = BILOU_ID[f'U-{label}']
        else:
            output[i] = BILOU_ID[f'B-{label}']
            for k in range(i + 1, j):
                output[k] = BILOU_ID[f'I-{label}']
            output[j] = BILOU_ID[f'L-{label}']
    return output


def transition_masks():
    k = len(BILOU)
    allowed = torch.zeros(k, k, dtype=torch.bool)
    start = torch.zeros(k, dtype=torch.bool)
    end = torch.zeros(k, dtype=torch.bool)
    for i, a in enumerate(BILOU):
        pa, la = ('O', '') if a == 'O' else a.split('-', 1)
        start[i] = pa in ('O', 'B', 'U')
        end[i] = pa in ('O', 'L', 'U')
        for j, b in enumerate(BILOU):
            pb, lb = ('O', '') if b == 'O' else b.split('-', 1)
            allowed[i, j] = (pb in ('I', 'L') and pa in ('B', 'I') and la == lb) or (pb in ('O', 'B', 'U') and pa in ('O', 'L', 'U'))
    return allowed, start, end


class ConstrainedCRF(nn.Module):
    def __init__(self):
        super().__init__()
        k = len(BILOU)
        self.transitions = nn.Parameter(torch.zeros(k, k))
        self.initial = nn.Parameter(torch.zeros(k))
        self.final = nn.Parameter(torch.zeros(k))
        allowed, start, end = transition_masks()
        self.register_buffer('allowed', allowed)
        self.register_buffer('start_allowed', start)
        self.register_buffer('end_allowed', end)

    def forward(self, emissions, targets, lengths):
        emissions = emissions.float()
        trans = self.transitions.masked_fill(~self.allowed, -1e4)
        init = self.initial.masked_fill(~self.start_allowed, -1e4)
        final = self.final.masked_fill(~self.end_allowed, -1e4)
        target = targets.clamp_min(0)
        batch, length, _ = emissions.shape
        alpha = init + emissions[:, 0]
        score = init[target[:, 0]] + emissions[:, 0].gather(1, target[:, 0, None]).squeeze(1)
        for t in range(1, length):
            valid = torch.as_tensor([t < n for n in lengths], device=emissions.device)
            nxt = torch.logsumexp(alpha[:, :, None] + trans[None], dim=1) + emissions[:, t]
            alpha = torch.where(valid[:, None], nxt, alpha)
            edge = trans[target[:, t - 1], target[:, t]]
            emit = emissions[:, t].gather(1, target[:, t, None]).squeeze(1)
            score += torch.where(valid, edge + emit, 0.)
        score += final[target[torch.arange(batch, device=emissions.device), torch.as_tensor(lengths, device=emissions.device) - 1]]
        partition = torch.logsumexp(alpha + final, dim=1)
        return (partition - score).sum() / sum(lengths)

    @torch.no_grad()
    def decode(self, emissions):
        emissions = emissions.float().cpu().numpy()
        trans = self.transitions.float().masked_fill(~self.allowed, -1e4).cpu().numpy()
        score = (self.initial.float().masked_fill(~self.start_allowed, -1e4)
                 .cpu().numpy() + emissions[0])
        back = []
        for t in range(1, len(emissions)):
            candidates = score[:, None] + trans
            previous = candidates.argmax(axis=0)
            score = candidates[previous, np.arange(len(previous))] + emissions[t]
            back.append(previous)
        score += self.final.float().masked_fill(~self.end_allowed, -1e4).cpu().numpy()
        path = [int(score.argmax())]
        for previous in reversed(back):
            path.append(int(previous[path[-1]]))
        return path[::-1]


def decode_bilou(tags, offsets, text):
    found, active = [], None
    for tag_id, (a, b) in zip(tags, offsets):
        tag = BILOU[int(tag_id)]
        if a == b or tag == 'O':
            active = None
            continue
        prefix, label = tag.split('-', 1)
        if prefix == 'U':
            found.append([a, b, label])
            active = None
        elif prefix == 'B':
            active = [a, b, label]
        elif prefix == 'I' and active is not None and active[2] == label:
            active[1] = b
        elif prefix == 'L' and active is not None and active[2] == label:
            active[1] = b
            found.append(active)
            active = None
    return [x for x in found if text[x[0]:x[1]].strip()]


class StructuredNER(nn.Module):
    def __init__(self, config, backbone_kind, method, structure, bidir_head=False,
                 transformer_head=False):
        super().__init__()
        if bidir_head and transformer_head:
            raise ValueError('Choose one bidirectional context head')
        self.encoder = DecoderNER(config, backbone_kind, method)
        self.structure = structure
        self.bidir_head = bidir_head
        self.transformer_head = transformer_head
        width = self.encoder.classifier.in_features
        self.encoder.classifier = nn.Identity()
        if structure == 'bilou_crf':
            if bidir_head:
                self.contextualizer = nn.GRU(width, 256, batch_first=True,
                                             bidirectional=True)
                emission_width = 512
            elif transformer_head:
                layer = nn.TransformerEncoderLayer(
                    d_model=256, nhead=4, dim_feedforward=1024, dropout=.1,
                    activation='gelu', batch_first=True, norm_first=True)
                self.contextualizer = nn.Sequential(
                    nn.Linear(width, 256), nn.LayerNorm(256),
                    nn.TransformerEncoder(layer, num_layers=2,
                                          enable_nested_tensor=False))
                emission_width = 256
            else:
                self.contextualizer = None
                emission_width = width
            self.emission = nn.Linear(emission_width, len(BILOU))
            self.crf = ConstrainedCRF()
        elif structure == 'start_end':
            self.start = nn.Linear(width, len(LABELS) + 1)
            self.end = nn.Linear(width, len(LABELS) + 1)
            self.type_head = nn.Sequential(nn.Linear(2 * width, 256), nn.GELU(), nn.Linear(256, len(LABELS)))
        elif structure == 'biaffine':
            self.start = nn.Linear(width, 2)
            self.end = nn.Linear(width, 2)
            self.start_proj = nn.Linear(width, 128)
            self.end_proj = nn.Linear(width, 128)
            self.biaffine = nn.Parameter(torch.zeros(len(LABELS) + 1, 129, 129))
            nn.init.normal_(self.biaffine, std=0.02)
        else:
            raise ValueError(structure)

    def configure_adaptation(self):
        self.encoder.configure_adaptation()

    def encode(self, ids):
        return self.encoder.encode(ids)

    def emit(self, hidden, lengths=None):
        if self.structure != 'bilou_crf':
            raise ValueError('emit is only defined for BILOU-CRF')
        if self.transformer_head:
            hidden = self.contextualizer[0](hidden)
            hidden = self.contextualizer[1](hidden)
            mask = None
            if lengths is not None:
                length_tensor = torch.as_tensor(lengths, device=hidden.device)
                mask = torch.arange(hidden.shape[1], device=hidden.device)[None] >= length_tensor[:, None]
            hidden = self.contextualizer[2](hidden, src_key_padding_mask=mask)
        elif self.contextualizer is not None:
            if lengths is None:
                hidden, _ = self.contextualizer(hidden)
            else:
                packed = nn.utils.rnn.pack_padded_sequence(
                    hidden, lengths, batch_first=True, enforce_sorted=False)
                contextual, _ = self.contextualizer(packed)
                hidden, _ = nn.utils.rnn.pad_packed_sequence(
                    contextual, batch_first=True, total_length=hidden.shape[1])
        return self.emission(hidden)

    def boundary_logits(self, hidden):
        if self.structure == 'bilou_crf':
            return self.emission(hidden)
        return self.start(hidden), self.end(hidden)

    def span_logits(self, hidden, batch_index, starts, ends):
        if self.structure == 'start_end':
            pair = torch.cat((hidden[batch_index, starts], hidden[batch_index, ends]), dim=-1)
            return self.type_head(pair)
        s = self.start_proj(hidden[batch_index, starts]).float()
        e = self.end_proj(hidden[batch_index, ends]).float()
        s = F.pad(s, (0, 1), value=1)
        e = F.pad(e, (0, 1), value=1)
        return torch.einsum('ni,kij,nj->nk', s, self.biaffine.float(), e)


def nonoverlap(candidates):
    """Greedy exact-offset conflict resolution, highest confidence first."""
    result = []
    for a, b, label, score in sorted(candidates, key=lambda r: r[3], reverse=True):
        if not any(a < y and b > x for x, y, _ in result):
            result.append((a, b, label))
    return [[a, b, label] for a, b, label in sorted(result)]
