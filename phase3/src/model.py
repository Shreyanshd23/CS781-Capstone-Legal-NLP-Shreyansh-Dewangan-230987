"""Phase-1-derived rhetorical-role models and distinct adaptation methods."""
from __future__ import annotations

import math
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

from peft import AdaLoraConfig, IA3Config, LoraConfig, inject_adapter_in_model

from ner_core import DecoderNER
from train_structured_ner import parent_config
from src.model import LegalLMConfig  # provided by Phase-1 package when on PYTHONPATH


class LormaPlusLinear(nn.Module):
    """LoRMA+: (I + BA) W0 x, with B=0 at initialization."""
    def __init__(self, base: nn.Linear, rank: int = 8, scale: float = 2.):
        super().__init__()
        self.base = base.requires_grad_(False)
        self.a = nn.Linear(base.out_features, rank, bias=False)
        self.b = nn.Linear(rank, base.out_features, bias=False)
        nn.init.kaiming_uniform_(self.a.weight, a=math.sqrt(5))
        nn.init.zeros_(self.b.weight)
        self.scale = scale

    def forward(self, x):
        output = self.base(x)
        return output + self.scale * self.b(self.a(output))


class LinearCRF(nn.Module):
    """Unconstrained linear-chain CRF over sentence labels."""
    def __init__(self, labels: int):
        super().__init__()
        self.transition = nn.Parameter(torch.zeros(labels, labels))
        self.start = nn.Parameter(torch.zeros(labels))
        self.end = nn.Parameter(torch.zeros(labels))

    def loss(self, emissions: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        emissions = emissions.float(); n = labels.numel()
        alpha = self.start + emissions[0]
        for step in range(1, n):
            alpha = torch.logsumexp(alpha[:, None] + self.transition, dim=0) + emissions[step]
        log_z = torch.logsumexp(alpha + self.end, dim=0)
        score = self.start[labels[0]] + emissions[0, labels[0]]
        for step in range(1, n):
            score = score + self.transition[labels[step-1], labels[step]] + emissions[step, labels[step]]
        return (log_z - score) / n

    def decode(self, emissions: torch.Tensor) -> list[int]:
        emissions = emissions.float()
        score = self.start + emissions[0]
        history = []
        for row in emissions[1:]:
            candidate = score[:, None] + self.transition
            best, index = candidate.max(dim=0)
            history.append(index)
            score = best + row
        last = int((score + self.end).argmax())
        path = [last]
        for index in reversed(history):
            last = int(index[last]); path.append(last)
        return list(reversed(path))


PARENTS = {
    'gpt2': 'A1_bpe_2b_7421',
    'dense': 'A3_legal_full_20261005',
    'ul2': 'A5_legal_ul2_full_20261006',
    'moe': 'A6_legal_moe_full_20261006',
}


class RoleModel(nn.Module):
    def __init__(self, kind: str, method: str, structure: str,
                 parent_root: Path, tokenizer_path: Path, student_layers: int = 0,
                 total_steps: int = 1000, use_position: bool = False,
                 train_last_layers: int = 4, pooling: str = 'last',
                 input_dropout: float = 0., classifier_dropout: float = .1):
        super().__init__()
        self.kind, self.method, self.structure = kind, method, structure
        self.input_dropout=nn.Dropout(input_dropout)
        source = parent_root / PARENTS[kind] / 'checkpoints/best/state.pt'
        parent, real_kind, config = parent_config(source, tokenizer_path)
        if real_kind != kind:
            raise ValueError(f'Wrong Phase-1 architecture: {source}')
        if student_layers:
            if kind != 'dense':
                raise ValueError('Compact student requires the dense checkpoint')
            raw = config.to_dict(); raw['n_layers'] = student_layers
            config = LegalLMConfig(**raw)
        self.encoder = DecoderNER(config, kind, method='full')
        weights = parent['model']
        if student_layers:
            weights = {k:v for k,v in weights.items()
                       if not k.startswith('blocks.') or int(k.split('.')[1]) < student_layers}
        self.encoder.backbone.load_state_dict(weights, strict=True)
        del parent
        self.encoder.backbone.requires_grad_(False)
        self.encoder.classifier.requires_grad_(False)
        if method in ('lora','rslora','adalora'):
            if kind == 'gpt2':
                targets = ['c_attn', 'c_proj']
            elif kind == 'ul2':
                targets = [name for name, module in self.encoder.backbone.named_modules()
                           if name.startswith('encoder.block.')
                           and name.endswith(('.q', '.v'))
                           and isinstance(module, nn.Linear)]
            else:
                targets = ['q_proj','v_proj']
            if method == 'adalora':
                peft_config = AdaLoraConfig(init_r=12, target_r=4,
                    tinit=max(1,min(100,total_steps//10)),
                    tfinal=max(1,min(100,total_steps//10)),
                    deltaT=20, total_step=max(10,total_steps),
                    lora_alpha=16, target_modules=targets)
            else:
                peft_config = LoraConfig(r=8, lora_alpha=16,
                    use_rslora=method=='rslora', target_modules=targets)
            inject_adapter_in_model(peft_config, self.encoder.backbone)
            self.peft_config = peft_config
        elif method == 'lorma':
            if kind not in ('dense','moe'):
                raise ValueError('LoRMA+ currently targets legal decoder linear layers')
            for block in self.encoder.backbone.blocks:
                for name in ('q_proj','v_proj'):
                    setattr(block.attn, name, LormaPlusLinear(getattr(block.attn,name)))
        elif method == 'ia3':
            if kind not in ('dense','moe'):
                raise ValueError('IA3 currently targets legal decoder linear layers')
            peft_config = IA3Config(target_modules=['k_proj','v_proj','down'],
                                     feedforward_modules=['down'])
            inject_adapter_in_model(peft_config, self.encoder.backbone)
            self.peft_config = peft_config
        elif method == 'distillation_partial':
            if kind != 'dense' or not student_layers:
                raise ValueError('Partial distillation needs a compact dense student')
            for block in self.encoder.backbone.blocks[-4:]:
                block.requires_grad_(True)
            self.encoder.backbone.final_norm.requires_grad_(True)
        elif method == 'partial':
            if kind not in ('dense','moe'):
                raise ValueError('Partial adaptation requires the legal decoder')
            if not 1 <= train_last_layers <= len(self.encoder.backbone.blocks):
                raise ValueError('Invalid number of trainable backbone layers')
            for block in self.encoder.backbone.blocks[-train_last_layers:]:
                block.requires_grad_(True)
            self.encoder.backbone.final_norm.requires_grad_(True)
        elif method == 'full':
            if kind not in ('dense','moe'):
                raise ValueError('Full decoder adaptation requires the legal decoder')
            self.encoder.backbone.requires_grad_(True)
        elif method == 'encoder_full':
            if kind != 'ul2':
                raise ValueError('Encoder-full adaptation requires UL2')
            self.encoder.backbone.encoder.requires_grad_(True)
            self.encoder.backbone.shared.requires_grad_(True)
        elif method not in ('distillation','frozen'):
            raise ValueError(method)
        width = config['n_embd'] if kind=='gpt2' else 768 if kind=='ul2' else config.d_model
        if pooling not in ('last','mean_last','last_attention'):raise ValueError(pooling)
        self.attn_pool=nn.Linear(width,1,bias=False) if pooling=='last_attention' else None
        self.pool_projection = nn.Linear(2*width,width,bias=False) if pooling!='last' else None
        if self.pool_projection is not None:
            with torch.no_grad():
                if pooling=='mean_last':
                    weights=torch.cat([torch.eye(width)*.5,torch.eye(width)*.5],dim=1)
                else:
                    weights=torch.cat([torch.eye(width),torch.zeros(width,width)],dim=1)
                self.pool_projection.weight.copy_(weights)
        self.position_embedding = nn.Embedding(20,width) if use_position else None
        if self.position_embedding is not None:
            nn.init.normal_(self.position_embedding.weight,std=.02)
        self.sentence_projection = nn.Linear(width,256) if structure=='s6' else None
        if structure=='s6':
            layer=nn.TransformerEncoderLayer(256,4,1024,dropout=.1,activation='gelu',
                                             batch_first=True,norm_first=True)
            self.context=nn.TransformerEncoder(layer,2,enable_nested_tensor=False)
            for block in self.context.layers:
                for parameter in block.parameters():
                    if parameter.ndim>1:nn.init.xavier_uniform_(parameter)
            self.classifier=nn.Sequential(nn.LayerNorm(256),nn.Dropout(classifier_dropout),nn.Linear(256,13))
        elif structure in ('s3','s4','s5'):
            self.context = nn.GRU(width, 256, batch_first=True,
                                  bidirectional=True)
            self.classifier = nn.Sequential(nn.LayerNorm(512), nn.Dropout(classifier_dropout),
                                            nn.Linear(512, 13))
        else:
            self.context = None
            self.classifier = nn.Sequential(nn.LayerNorm(width), nn.Dropout(classifier_dropout),
                                            nn.Linear(width, 13))
        self.crf = LinearCRF(13) if structure in ('s4','s5','s6') else None

    def encode(self, ids: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        hidden = self.encoder.encode(ids)
        last=hidden[torch.arange(ids.shape[0],device=ids.device),lengths-1]
        if self.pool_projection is not None:
            mask=torch.arange(ids.shape[1],device=ids.device)[None,:]<lengths[:,None]
            if self.attn_pool is None:
                pooled=(hidden*mask.unsqueeze(-1)).sum(1)/lengths[:,None].clamp_min(1)
            else:
                attention=self.attn_pool(hidden).squeeze(-1).float()
                attention=attention.masked_fill(~mask,-1e9).softmax(-1)
                pooled=(hidden*attention.unsqueeze(-1)).sum(1)
            return self.pool_projection(torch.cat([last,pooled],dim=-1))
        if self.kind == 'ul2':
            mask = torch.arange(ids.shape[1],device=ids.device)[None,:]<lengths[:,None]
            return (hidden*mask.unsqueeze(-1)).sum(1)/lengths[:,None].clamp_min(1)
        return last

    def forward(self, ids: torch.Tensor, lengths: torch.Tensor,
                positions: torch.Tensor | None = None) -> torch.Tensor:
        representations = self.encode(ids,lengths)
        if self.position_embedding is not None:
            if positions is None:
                raise ValueError('Sentence position is required by this model')
            representations = representations + self.position_embedding(positions)
        representations=self.input_dropout(representations)
        if self.structure=='s6':
            representations=self.sentence_projection(representations)
            position=torch.arange(len(representations),device=ids.device).float()[:,None]
            frequency=torch.exp(torch.arange(0,256,2,device=ids.device).float()*(-math.log(10000.)/256))
            positional=torch.zeros_like(representations)
            positional[:,0::2]=torch.sin(position*frequency)
            positional[:,1::2]=torch.cos(position*frequency)
            representations=self.context((representations+positional).unsqueeze(0)).squeeze(0)
        elif self.context is not None:
            representations = self.context(representations.unsqueeze(0))[0].squeeze(0)
        return self.classifier(representations)
