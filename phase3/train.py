#!/usr/bin/env python3
"""Train a Phase-1-derived rhetorical-role classifier and package predictions."""
from __future__ import annotations

import argparse
from collections import Counter
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import random
import time
from zipfile import ZipFile, ZIP_DEFLATED

import torch
from torch import nn
from torch.nn import functional as F
from tokenizers import Tokenizer
from tqdm import tqdm

from .src.data import DATA, ROOT, documents, labels, validate_prediction
from .src.evaluate import evaluate
from .src.model import PARENTS, RoleModel


PARENT_ROOT = ROOT.parent / 'phase1/runs'
TOKENIZERS = {
    'gpt2': ROOT.parent / 'phase1/artifacts/tokenizers_512m_32k/bpe/tokenizer.json',
    'dense': ROOT.parent / 'phase1/artifacts/tokenizers_512m_32k/bpe/tokenizer.json',
    'moe': ROOT.parent / 'phase1/artifacts/tokenizers_512m_32k/bpe/tokenizer.json',
    'ul2': ROOT.parent / 'phase1/data/legal_ul2_full_20261006/tokenizer.json',
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(4*1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def encode_documents(rows: list[dict], tokenizer: Tokenizer,
                     structure: str, target_tokens: int | None = None,
                     neighbor_tokens: int = 36,
                     court_prefix: bool = False) -> list[dict]:
    # Every sentence yields exactly one sequence. Truncation is tracked in
    # the run config; no sentence or document is dropped.
    def encode(text):
        return tokenizer.encode(text, add_special_tokens=False).ids
    eos = tokenizer.token_to_id('<eos>')
    if eos is None:
        eos = tokenizer.token_to_id('</s>')
    if eos is None:
        eos = 2
    output = []
    truncated = 0
    limit = target_tokens or (112 if structure=='s2' else 160)
    for row in rows:
        sentences = row['text']
        original = [encode(text) for text in sentences]
        prefix = (encode(f" Court: {row['id'].split('_', 1)[0]}. Domain: {row['domain']}. Sentence: ")
                  if court_prefix else [])
        ids = []
        for i, target in enumerate(original):
            if structure in ('s2','s5'):
                previous = original[i-1][-neighbor_tokens:] if i and neighbor_tokens else []
                next_ = original[i+1][:neighbor_tokens] if i+1 < len(original) else []
                marker = encode(' Classify target sentence:')
                if structure=='s5' and len(target)>limit:
                    head=int(limit*.7)
                    target_ids=target[:head]+target[-(limit-head):]
                else:
                    target_ids = target[:limit]
                truncated += len(target) > limit
                tokens = prefix + previous + encode(' Target:') + target_ids + encode(' Next:') + next_ + marker + [eos]
                if structure=='s2' and target_tokens is None and neighbor_tokens==36:
                    tokens = tokens[-224:]
            else:
                truncated += len(target) > limit
                head=int(limit*.7)
                tokens = prefix + (target[:head]+target[-(limit-head):] if len(target)>limit else target) + [eos]
            ids.append(tokens if tokens else [eos])
        output.append({**row, 'ids':ids})
    return output


def make_units(rows: list[dict], structure: str, chunk: int = 16,
               overlap: int = 4, training: bool = False,
               training_overlap: int = 0) -> list[tuple[str,list[int]]]:
    units = []
    for row in rows:
        size = len(row['ids'])
        if structure in ('s1','s2'):
            units.extend((row['id'],[i]) for i in range(size))
            continue
        step = chunk-training_overlap if training else chunk-overlap
        for start in range(0,size,step):
            indices = list(range(start,min(start+chunk,size)))
            if indices: units.append((row['id'],indices))
            if start+chunk>=size:break
    return units


def batch_inputs(rows_by_id: dict, unit: tuple[str,list[int]], device):
    key, indices = unit
    sequences = [rows_by_id[key]['ids'][i] for i in indices]
    width = max(map(len,sequences))
    ids = torch.zeros((len(sequences),width),dtype=torch.long,device=device)
    for j,seq in enumerate(sequences):ids[j,:len(seq)] = torch.tensor(seq,device=device)
    lengths = torch.tensor([len(seq) for seq in sequences],device=device)
    return ids,lengths


@torch.inference_mode()
def predict(model: RoleModel, rows: list[dict], device, chunk: int = 16,
            batch_size: int = 16) -> tuple[dict,dict]:
    model.eval()
    by_id = {row['id']:row for row in rows}
    accum = {row['id']:torch.zeros((len(row['ids']),13)) for row in rows}
    counts = {row['id']:torch.zeros(len(row['ids'])) for row in rows}
    units = make_units(rows,model.structure,chunk=chunk,training=False)
    if model.structure in ('s1','s2'):
        groups = [units[i:i+batch_size] for i in range(0,len(units),batch_size)]
    else: groups = [[unit] for unit in units]
    for group in groups:
        seqs = [(key,index) for key,indices in group for index in indices]
        sequences = [by_id[k]['ids'][i] for k,i in seqs]
        width=max(map(len,sequences))
        ids=torch.zeros((len(sequences),width),dtype=torch.long,device=device)
        for j,seq in enumerate(sequences):ids[j,:len(seq)]=torch.tensor(seq,device=device)
        lengths=torch.tensor([len(seq) for seq in sequences],device=device)
        positions=torch.tensor([min(19,int(index/max(1,len(by_id[key]['ids']))*20))
                                for key,index in seqs],device=device)
        with torch.autocast('cuda',dtype=torch.bfloat16):
            logits=model(ids,lengths,positions)
        logits=logits.float().cpu()
        for (key,index),logit in zip(seqs,logits):
            accum[key][index]+=logit; counts[key][index]+=1
    output={};probabilities={}
    vocab=labels()
    for row in rows:
        key=row['id']; scores=accum[key]/counts[key].clamp_min(1)[:,None]
        indices=(model.crf.decode(scores.to(device)) if model.crf is not None
                 else scores.argmax(-1).tolist())
        output[key]=[vocab[i] for i in indices]
        probabilities[key]=scores.softmax(-1).tolist()
    return output,probabilities


def save_submission(prediction: dict, folder: Path) -> tuple[Path,Path]:
    validate_prediction(prediction)
    folder.mkdir(parents=True,exist_ok=True)
    path=folder/'prediction.json'
    path.write_text(json.dumps(prediction,ensure_ascii=False,indent=2)+'\n')
    zip_path=folder/'submission.zip'
    with ZipFile(zip_path,'w',ZIP_DEFLATED,compresslevel=9) as archive:
        archive.write(path,arcname='prediction.json')
    with ZipFile(zip_path) as archive:
        if (archive.namelist()!=['prediction.json'] or archive.testzip() is not None
                or json.loads(archive.read('prediction.json'))!=prediction):
            raise RuntimeError('Invalid submission ZIP')
    return path,zip_path


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--id',required=True)
    ap.add_argument('--kind',choices=PARENTS,required=True)
    ap.add_argument('--method',choices=['lora','rslora','adalora','lorma','ia3','partial','full','encoder_full','frozen','distillation','distillation_partial'],required=True)
    ap.add_argument('--structure',choices=['s1','s2','s3','s4','s5','s6'],required=True)
    ap.add_argument('--epochs',type=int,default=3)
    ap.add_argument('--batch-size',type=int,default=16)
    ap.add_argument('--chunk',type=int,default=16)
    ap.add_argument('--learning-rate',type=float,default=1.5e-4)
    ap.add_argument('--head-learning-rate',type=float)
    ap.add_argument('--target-tokens',type=int)
    ap.add_argument('--neighbor-tokens',type=int,default=36)
    ap.add_argument('--court-prefix',action='store_true',
                    help='Prefix each sentence with the available court and CL/IT domain')
    ap.add_argument('--train-last-layers',type=int,default=4)
    ap.add_argument('--pooling',choices=['last','mean_last','last_attention'],default='last')
    ap.add_argument('--input-dropout',type=float,default=0.)
    ap.add_argument('--classifier-dropout',type=float,default=.1)
    ap.add_argument('--schedule',choices=['constant','cosine'],default='constant')
    ap.add_argument('--patience',type=int,default=2)
    ap.add_argument('--max-units-per-document',type=int,
                    help='Randomly cap training chunks from each document per epoch')
    ap.add_argument('--train-domain',choices=['CL','IT'],
                    help='Train on one official domain; still evaluate both domains')
    ap.add_argument('--train-fraction',type=float,default=1.,
                    help='Fraction of training documents sampled per domain without replacement')
    ap.add_argument('--holdout-ids-file',type=Path,
                    help='JSON list of official training document IDs to use as validation')
    ap.add_argument('--audit-ids-file',type=Path,
                    help='Exclude listed training IDs and score them only after selecting on official dev')
    ap.add_argument('--training-overlap',type=int,default=0,
                    help='Overlap adjacent training chunks for structured models')
    ap.add_argument('--save-epoch-checkpoints',action='store_true',
                    help='Retain model-only snapshots for checkpoint averaging')
    ap.add_argument('--seed',type=int,default=42)
    ap.add_argument('--student-layers',type=int,default=0)
    ap.add_argument('--teacher-run',type=Path)
    ap.add_argument('--position-features',action='store_true')
    ap.add_argument('--aux-ce',type=float,default=0.,
                    help='Auxiliary class-weighted CE added to CRF loss')
    ap.add_argument('--smoke',action='store_true')
    args=ap.parse_args()
    if args.aux_ce<0:ap.error('--aux-ce must be nonnegative')
    if args.target_tokens is not None and args.target_tokens<2:ap.error('--target-tokens must be at least 2')
    if args.neighbor_tokens<0 or args.chunk<=4 or args.patience<1:ap.error('Invalid context/chunk/patience')
    if args.max_units_per_document is not None and args.max_units_per_document<1:
        ap.error('--max-units-per-document must be positive')
    if not 0 <= args.training_overlap < args.chunk:
        ap.error('--training-overlap must be between zero and chunk size')
    if not 0 <= args.input_dropout < 1 or not 0 <= args.classifier_dropout < 1:
        ap.error('Dropout probabilities must be in [0,1)')
    if not 0 < args.train_fraction <= 1:
        ap.error('--train-fraction must be in (0,1]')
    if args.holdout_ids_file and args.audit_ids_file:
        ap.error('--holdout-ids-file and --audit-ids-file are mutually exclusive')
    torch.manual_seed(args.seed);random.seed(args.seed)
    torch.set_float32_matmul_precision('high')
    device=torch.device('cuda:0')
    run=ROOT/'runs'/args.id;submission=ROOT/'submissions'/args.id
    if run.exists() or submission.exists():ap.error('Run/submission already exists')
    run.mkdir(parents=True)
    tokenizer_path=TOKENIZERS[args.kind]
    tokenizer=Tokenizer.from_file(str(tokenizer_path))
    training_rows=documents('train')
    audit_rows=[]
    if args.audit_ids_file:
        audit_ids=json.loads(args.audit_ids_file.read_text())
        if not isinstance(audit_ids,list) or len(audit_ids)!=len(set(audit_ids)):
            ap.error('--audit-ids-file must contain a unique JSON ID list')
        unknown=set(audit_ids)-{row['id'] for row in training_rows}
        if unknown or not audit_ids:
            ap.error(f'Invalid audit IDs: {sorted(unknown)[:3]}')
        audit_rows=[row for row in training_rows if row['id'] in set(audit_ids)]
        training_rows=[row for row in training_rows if row['id'] not in set(audit_ids)]
    if args.holdout_ids_file:
        holdout_ids=json.loads(args.holdout_ids_file.read_text())
        if not isinstance(holdout_ids,list) or len(holdout_ids)!=len(set(holdout_ids)):
            ap.error('--holdout-ids-file must contain a unique JSON ID list')
        unknown=set(holdout_ids)-{row['id'] for row in training_rows}
        if unknown or not holdout_ids:
            ap.error(f'Invalid holdout IDs: {sorted(unknown)[:3]}')
        development_rows=[row for row in training_rows if row['id'] in set(holdout_ids)]
        training_rows=[row for row in training_rows if row['id'] not in set(holdout_ids)]
    else:
        development_rows=documents('dev')
    if args.train_domain:
        training_rows=[row for row in training_rows if row['domain']==args.train_domain]
    if args.train_fraction<1:
        rng=random.Random(args.seed)
        groups={domain:[item for item in training_rows if item['domain']==domain]
                for domain in ('CL','IT')}
        training_rows=[row for group in groups.values() if group
                       for row in rng.sample(group,max(1,round(args.train_fraction*len(group))))]
    train=encode_documents(training_rows,tokenizer,args.structure,args.target_tokens,args.neighbor_tokens,args.court_prefix)
    dev=encode_documents(development_rows,tokenizer,args.structure,args.target_tokens,args.neighbor_tokens,args.court_prefix)
    test=encode_documents(documents('test'),tokenizer,args.structure,args.target_tokens,args.neighbor_tokens,args.court_prefix)
    audit=encode_documents(audit_rows,tokenizer,args.structure,args.target_tokens,args.neighbor_tokens,args.court_prefix) if audit_rows else []
    if args.smoke:
        train=[{**train[0], 'text':train[0]['text'][:16],
                'labels':train[0]['labels'][:16], 'ids':train[0]['ids'][:16]}]
        dev=train
    planned_units=make_units(train,args.structure,args.chunk,training=True,
                             training_overlap=args.training_overlap)
    if args.max_units_per_document is not None:
        per_document=Counter(key for key,_ in planned_units)
        planned_count=sum(min(count,args.max_units_per_document)
                          for count in per_document.values())
    else:
        planned_count=len(planned_units)
    planned_steps=(math.ceil(planned_count/args.batch_size)
                   if args.structure in ('s1','s2') else planned_count)*args.epochs
    model=RoleModel(args.kind,args.method,args.structure,PARENT_ROOT,
                    tokenizer_path,args.student_layers,planned_steps,
                    args.position_features,args.train_last_layers,args.pooling,
                    args.input_dropout,args.classifier_dropout).to(device)
    if args.method=='adalora':
        from peft.tuners.adalora import RankAllocator
        allocator=RankAllocator(model.encoder.backbone,model.peft_config,'default')
    else:allocator=None
    teacher=None;teacher_probs=None
    if args.teacher_run:
        meta=json.loads((args.teacher_run/'config.json').read_text())
        if set(meta['training_ids'])!={r['id'] for r in documents('train')}:
            raise ValueError('Teacher used a different training split')
        teacher=RoleModel(meta['kind'],meta['method'],meta['structure'],PARENT_ROOT,
                          TOKENIZERS[meta['kind']],meta.get('student_layers',0),
                          use_position=meta.get('position_features',False),
                          train_last_layers=meta.get('train_last_layers',4),
                          pooling=meta.get('pooling','last')).to(device)
        ck=torch.load(args.teacher_run/'best.pt',map_location=device,weights_only=False)
        teacher.load_state_dict(ck['model'],strict=True);teacher.eval().requires_grad_(False)
        teacher_tok=Tokenizer.from_file(str(TOKENIZERS[meta['kind']]))
        teacher_train=encode_documents(documents('train'),teacher_tok,meta['structure'],
                                       meta.get('target_tokens'),meta.get('neighbor_tokens',36))
        _,teacher_probs=predict(teacher,teacher_train,device,meta['chunk'],meta['batch_size'])
        del teacher;torch.cuda.empty_cache()
    count=sum(p.numel() for p in model.parameters())
    trainable=sum(p.numel() for p in model.parameters() if p.requires_grad)
    parent=PARENT_ROOT/PARENTS[args.kind]/'checkpoints/best/state.pt'
    config={'id':args.id,'kind':args.kind,'method':args.method,'structure':args.structure,
            'parent_checkpoint':str(parent.resolve()),'parent_sha256':sha256(parent),
            'tokenizer':str(tokenizer_path.resolve()),'tokenizer_sha256':sha256(tokenizer_path),
            'training_ids':[r['id'] for r in training_rows],
            'train_domain':args.train_domain,
            'train_fraction':args.train_fraction,
            'development_ids':[r['id'] for r in development_rows],
            'holdout_ids_file':str(args.holdout_ids_file) if args.holdout_ids_file else None,
            'audit_ids_file':str(args.audit_ids_file) if args.audit_ids_file else None,
            'audit_ids':[r['id'] for r in audit_rows],
            'parameters':count,'trainable_parameters':trainable,
            'epochs':args.epochs,'batch_size':args.batch_size,'chunk':args.chunk,
            'learning_rate':args.learning_rate,'seed':args.seed,
            'head_learning_rate':args.head_learning_rate,
            'target_tokens':args.target_tokens,'neighbor_tokens':args.neighbor_tokens,
            'court_prefix':args.court_prefix,
            'train_last_layers':args.train_last_layers,'schedule':args.schedule,
            'pooling':args.pooling,
            'input_dropout':args.input_dropout,
            'classifier_dropout':args.classifier_dropout,
            'patience':args.patience,'matmul_precision':'high',
            'max_units_per_document':args.max_units_per_document,
            'training_overlap':args.training_overlap,
            'save_epoch_checkpoints':args.save_epoch_checkpoints,
            'student_layers':args.student_layers,
            'position_features':args.position_features,
            'aux_ce':args.aux_ce,
            'teacher_run':str(args.teacher_run) if args.teacher_run else None}
    (run/'config.json').write_text(json.dumps(config,indent=2)+'\n')
    classes={tag:i for i,tag in enumerate(labels())}
    frequencies=Counter(tag for row in training_rows for tag in row['labels'])
    weights=torch.tensor([min(3.,max(.6,(sum(frequencies.values())/(13*max(1,frequencies[tag])))**.25))
                          for tag in labels()],device=device)
    parameter_groups=[{'params':[p for n,p in model.named_parameters()
                                 if p.requires_grad and n.startswith('encoder.backbone.')],
                       'lr':args.learning_rate},
                      {'params':[p for n,p in model.named_parameters()
                                 if p.requires_grad and not n.startswith('encoder.backbone.')],
                       'lr':args.head_learning_rate or args.learning_rate}]
    optimizer=torch.optim.AdamW(parameter_groups,weight_decay=.01,fused=True)
    units=make_units(train,args.structure,args.chunk,training=True,
                     training_overlap=args.training_overlap)
    unit_groups={}
    for unit in units:unit_groups.setdefault(unit[0],[]).append(unit)
    def epoch_units():
        if args.max_units_per_document is None:
            selected=list(units)
        else:
            selected=[unit for group in unit_groups.values()
                      for unit in random.sample(group,min(len(group),args.max_units_per_document))]
        random.shuffle(selected)
        return selected
    by_id={row['id']:row for row in train}
    if args.structure in ('s1','s2'):
        def batches():
            selected=epoch_units()
            for i in range(0,len(selected),args.batch_size):
                yield selected[i:i+args.batch_size]
        steps=math.ceil(planned_count/args.batch_size)*args.epochs
    else:
        def batches():
            for unit in epoch_units():yield [unit]
        steps=planned_count*args.epochs
    global_step=0;best=-1;started=time.monotonic();best_epoch=0
    warmup=max(1,int(.05*steps))
    def lr_factor(step):
        if args.schedule=='constant':return 1.
        if step<warmup:return max(.01,step/warmup)
        fraction=min(1.,(step-warmup)/max(1,steps-warmup))
        return .1+.9*.5*(1+math.cos(math.pi*fraction))
    scheduler=torch.optim.lr_scheduler.LambdaLR(optimizer,lr_factor)
    for epoch in range(1,args.epochs+1):
        model.train();total_loss=0;n_batches=0
        for group in tqdm(batches(),total=math.ceil(planned_count/args.batch_size) if args.structure in ('s1','s2') else planned_count,
                          desc=f'{args.id} epoch {epoch}',unit='batch'):
            seqs=[(key,index) for key,indices in group for index in indices]
            sequences=[by_id[key]['ids'][index] for key,index in seqs]
            width=max(map(len,sequences))
            ids=torch.zeros((len(sequences),width),dtype=torch.long,device=device)
            for j,seq in enumerate(sequences):ids[j,:len(seq)]=torch.tensor(seq,device=device)
            lengths=torch.tensor([len(seq) for seq in sequences],device=device)
            positions=torch.tensor([min(19,int(index/max(1,len(by_id[key]['ids']))*20))
                                    for key,index in seqs],device=device)
            gold=torch.tensor([classes[by_id[key]['labels'][index]] for key,index in seqs],device=device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast('cuda',dtype=torch.bfloat16):
                logits=model(ids,lengths,positions)
                if model.crf is not None:
                    loss=model.crf.loss(logits,gold)
                    if args.aux_ce:
                        loss=loss+args.aux_ce*F.cross_entropy(
                            logits.float(),gold,weight=weights)
                else:
                    loss=F.cross_entropy(logits.float(),gold,weight=weights)
                if teacher_probs is not None:
                    soft=torch.tensor([teacher_probs[key][index] for key,index in seqs],device=device)
                    temperature=2.
                    kd=F.kl_div(F.log_softmax(logits.float()/temperature,-1),
                                F.softmax(soft.clamp_min(1e-8).log()/temperature,-1),
                                reduction='batchmean')*temperature**2
                    loss=.65*loss+.35*kd
            loss.backward()
            nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],1.)
            optimizer.step();scheduler.step();global_step+=1
            if allocator is not None and global_step<=model.peft_config.total_step:
                allocator.update_and_allocate(model.encoder.backbone,global_step)
            total_loss+=float(loss.detach());n_batches+=1
            if args.smoke and global_step>=20:break
        dev_pred,dev_probs=predict(model,dev,device,args.chunk,args.batch_size)
        metrics=evaluate(dev,dev_pred)
        record={'epoch':epoch,'step':global_step,'train_loss':total_loss/max(1,n_batches),
                'macro_f1':metrics['overall']['macro_f1'],
                'cl_macro_f1':metrics['domains']['CL']['macro_f1'],
                'it_macro_f1':metrics['domains']['IT']['macro_f1'],
                'elapsed_seconds':time.monotonic()-started,
                'gpu_peak_gib':torch.cuda.max_memory_allocated()/2**30}
        with (run/'metrics.jsonl').open('a') as handle:handle.write(json.dumps(record)+'\n')
        state={'model':model.state_dict(),'epoch':epoch,'metrics':record,
               'optimizer':optimizer.state_dict(),'scheduler':scheduler.state_dict(),
               'global_step':global_step}
        torch.save(state,run/'latest.pt.tmp');os.replace(run/'latest.pt.tmp',run/'latest.pt')
        if args.save_epoch_checkpoints:
            torch.save({'model':model.state_dict(),'epoch':epoch},
                       run/f'epoch_{epoch:02d}.pt.tmp')
            os.replace(run/f'epoch_{epoch:02d}.pt.tmp',run/f'epoch_{epoch:02d}.pt')
        if record['macro_f1']>best:
            best=record['macro_f1'];best_epoch=epoch
            torch.save(state,run/'best.pt.tmp');os.replace(run/'best.pt.tmp',run/'best.pt')
            (run/'dev_predictions.json').write_text(json.dumps(dev_pred,indent=2)+'\n')
            (run/'dev_probabilities.json').write_text(json.dumps(dev_probs)+'\n')
            (run/'dev_metrics.json').write_text(json.dumps(metrics,indent=2)+'\n')
        print(json.dumps(record),flush=True)
        if args.smoke:break
        if epoch-best_epoch>=args.patience:break
    # Verify full checkpoint reload before official test inference.
    checkpoint=torch.load(run/'best.pt',map_location=device,weights_only=False)
    model.load_state_dict(checkpoint['model'],strict=True)
    repeat,_=predict(model,dev,device,args.chunk,args.batch_size)
    if repeat!=json.loads((run/'dev_predictions.json').read_text()):
        raise RuntimeError('Best checkpoint reload changed development predictions')
    if args.smoke:
        print(f'SMOKE FINISHED {args.id}',flush=True)
        return
    if audit:
        audit_pred,audit_probs=predict(model,audit,device,args.chunk,args.batch_size)
        (run/'audit_predictions.json').write_text(json.dumps(audit_pred,indent=2)+'\n')
        (run/'audit_probabilities.json').write_text(json.dumps(audit_probs)+'\n')
        (run/'audit_metrics.json').write_text(json.dumps(evaluate(audit,audit_pred),indent=2)+'\n')
    test_pred,test_probs=predict(model,test,device,args.chunk,args.batch_size)
    (run/'test_probabilities.json').write_text(json.dumps(test_probs)+'\n')
    path,zip_path=save_submission(test_pred,submission)
    (submission/'manifest.json').write_text(json.dumps({
        'experiment':args.id,'config':str((run/'config.json').resolve()),
        'best_dev_macro_f1':best,'best_epoch':best_epoch,
        'prediction_sha256':sha256(path),'zip_sha256':sha256(zip_path),
        'official_score':None,'documents':len(test_pred),
        'sentences':sum(map(len,test_pred.values()))},indent=2)+'\n')
    print(f'FINISHED {args.id} dev_macro_f1={best:.6f} zip={zip_path}',flush=True)


if __name__=='__main__':main()
