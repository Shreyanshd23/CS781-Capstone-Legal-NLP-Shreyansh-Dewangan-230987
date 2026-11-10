#!/usr/bin/env python3
"""Package a cross-validated exact-mention consistency variant of E26.

Only official Phase-2 texts and existing Phase-1-derived model predictions are
used. Validation labels are read solely to report held-out scores.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import re
from zipfile import ZIP_DEFLATED, ZipFile

from ner_core import LABELS, read_folds
from package_crf_label_router import merge
from sweep_crf_inference import score_predictions
from train_ner import sha256


PROPAGATE_LABELS = frozenset({'WIT', 'AUTH', 'STAT', 'JUDGE', 'COURT'})
MIN_MENTION_LENGTH = 4
PARTY_ALIAS_STOPWORDS = frozenset({
    'state', 'union', 'government', 'india', 'court', 'justice', 'the',
    'national', 'new', 'civil', 'delhi', 'indian', 'shri', 'smt', 'mr',
    'mrs', 'm/s', 'limited', 'private', 'commissioner', 'inspector',
    'trustees', 'university', 'called',
})


def propagate_mentions(predictions: dict, texts: dict[str, str]) -> dict:
    """Add nonoverlapping exact-case repeats of selected predicted mentions."""
    if set(predictions) != set(texts):
        raise ValueError('Prediction and text IDs differ')
    result = {}
    for doc_id, spans in predictions.items():
        text = texts[doc_id]
        chosen = [list(span) for span in spans]
        terms = Counter((text[a:b], label) for a, b, label in spans
                        if label in PROPAGATE_LABELS and b - a >= MIN_MENTION_LENGTH)
        for (surface, label), _ in terms.most_common():
            pattern = re.compile(r'(?<!\w)' + re.escape(surface) + r'(?!\w)')
            for match in pattern.finditer(text):
                start, end = match.span()
                if not any(start < old_end and end > old_start
                           for old_start, old_end, _ in chosen):
                    chosen.append([start, end, label])
        result[doc_id] = sorted(chosen)
    return result


def caption_party_correction(predictions: dict, texts: dict[str, str]) -> dict:
    """Resolve a predicted party's role from its explicit case-caption side.

    Only multiword alphabetic names appearing within 50 characters of a
    caption's Vs./Versus marker are changed. Ambiguous names are left alone.
    """
    result = {}
    for doc_id, spans in predictions.items():
        text = texts[doc_id]
        caption = text[:1800]
        boundary = re.search(r'\b(?:JUDGMENT|HEADNOTE)\b', caption, re.I)
        if boundary and boundary.start() > 100:
            caption = caption[:boundary.start()]
        markers = list(re.finditer(r'\b(?:vs\.?|versus|v\.)\b', caption, re.I))
        corrected = []
        for start, end, label in spans:
            new_label = label
            surface = text[start:end]
            if (label in ('APP', 'RESP') and len(surface) >= 6
                    and len(surface.split()) >= 2
                    and re.fullmatch(r"[A-Za-z .'-]+", surface)):
                pattern = re.compile(r'(?<!\w)' + re.escape(surface) + r'(?!\w)', re.I)
                left = right = 0
                for marker in markers:
                    left += bool(pattern.search(caption[max(0, marker.start() - 50):marker.start()]))
                    right += bool(pattern.search(caption[marker.end():marker.end() + 50]))
                if left and not right:
                    new_label = 'APP'
                elif right and not left:
                    new_label = 'RESP'
            corrected.append([start, end, new_label])
        result[doc_id] = corrected
    return result


def caption_party_aliases(predictions: dict, texts: dict[str, str],
                          alias_types: frozenset[str]) -> dict:
    """Propagate caption-anchored party names and their short forms.

    A seed must already be predicted as APP/RESP next to a Vs./Versus marker.
    An alias is used only when it appears at least twice in the same document.
    Generic role and government terms are excluded as short-form seeds. No
    other predicted entity type is overwritten.
    """
    result = {}
    for doc_id, spans in predictions.items():
        text = texts[doc_id]
        caption = text[:1800]
        boundary = re.search(r'\b(?:JUDGMENT|HEADNOTE)\b', caption, re.I)
        if boundary and boundary.start() > 100:
            caption = caption[:boundary.start()]
        markers = list(re.finditer(r'\b(?:vs\.?|versus|v\.)\b', caption, re.I))
        seeds = []
        for start, end, label in spans:
            if label not in ('APP', 'RESP') or end > len(caption):
                continue
            if any((label == 'APP' and end <= marker.start()
                    and marker.start() - end <= 60)
                   or (label == 'RESP' and start >= marker.end()
                       and start - marker.end() <= 60)
                   for marker in markers):
                seeds.append((text[start:end], label))
        aliases = []
        for surface, label in seeds:
            words = re.findall(r'[A-Za-z]+', surface)
            if ('full' in alias_types and len(surface) >= 6 and words
                    and words[0].lower() not in PARTY_ALIAS_STOPWORDS):
                aliases.append((surface.strip(), label))
            if 'acronym' in alias_types and len(words) >= 3:
                acronym = ''.join(word[0].upper() for word in words)
                if 3 <= len(acronym) <= 8:
                    aliases.append((acronym, label))
            if ('first' in alias_types and words and len(words[0]) >= 5
                    and words[0].lower() not in PARTY_ALIAS_STOPWORDS):
                aliases.append((words[0], label))
            if ('first2' in alias_types and len(words) >= 2
                    and len(words[0] + words[1]) >= 8
                    and words[0].lower() not in PARTY_ALIAS_STOPWORDS):
                aliases.append((' '.join(words[:2]), label))
        chosen = [list(span) for span in spans]
        for surface, label in aliases:
            pattern = re.compile(r'(?<!\w)' + re.escape(surface) + r'(?!\w)', re.I)
            matches = list(pattern.finditer(text))
            if len(matches) < 2:
                continue
            for match in matches:
                start, end = match.span()
                overlapping = [(index, span) for index, span in enumerate(chosen)
                               if start < span[1] and end > span[0]]
                if not overlapping:
                    chosen.append([start, end, label])
                elif (len(overlapping) == 1
                      and chosen[overlapping[0][0]][:2] == [start, end]
                      and chosen[overlapping[0][0]][2] in ('APP', 'RESP')
                      and chosen[overlapping[0][0]][2] != label):
                    chosen[overlapping[0][0]][2] = label
        result[doc_id] = sorted(chosen)
    return result


def complete_corporate_party_spans(predictions: dict,
                                   texts: dict[str, str]) -> dict:
    """Recover full corporate party names and omitted final dots in Ltd."""
    output = {}
    for doc_id, spans in predictions.items():
        text = texts[doc_id]
        caption = text[:1800]
        boundary = re.search(r'\b(?:JUDGMENT|HEADNOTE)\b', caption, re.I)
        if boundary and boundary.start() > 100:
            caption = caption[:boundary.start()]
        markers = list(re.finditer(r'\b(?:vs\.?|versus|v\.)\b', caption, re.I))
        seeds = set()
        for start, end, label in spans:
            if label not in ('APP', 'RESP') or end > len(caption):
                continue
            near_marker = any(
                (label == 'APP' and end <= marker.start()
                 and marker.start() - end <= 60)
                or (label == 'RESP' and start >= marker.end()
                    and start - marker.end() <= 60)
                for marker in markers)
            if near_marker:
                words = re.findall(r'[A-Za-z]+', text[start:end])
                if (words and len(words[0]) >= 5
                        and words[0].lower() not in PARTY_ALIAS_STOPWORDS):
                    seeds.add((words[0], label))
        chosen = [span.copy() for span in spans]
        for name, label in sorted(seeds):
            pattern = re.compile(
                r'(?<!\w)' + re.escape(name)
                + r'(?:\s+[A-Z][A-Za-z&.]+){1,4}\s+'
                  r'(?:Pvt\.?\s+)?(?:Ltd\.?|Limited)(?!\w)', re.I)
            for match in pattern.finditer(text):
                start, end = match.span()
                overlapping = [span for span in chosen
                               if start < span[1] and end > span[0]]
                if all(span[2] == label and start <= span[0]
                       and span[1] <= end for span in overlapping):
                    chosen = [span for span in chosen if span not in overlapping]
                    chosen.append([start, end, label])
        for span in chosen:
            start, end, label = span
            if (label in ('APP', 'RESP') and end < len(text)
                    and text[start:end].lower().endswith('ltd')
                    and text[end] == '.'
                    and not any(end < other[1] and end + 1 > other[0]
                                for other in chosen if other is not span)):
                span[1] += 1
        output[doc_id] = sorted(chosen)
    return output


def complete_short_party_prefixes(predictions: dict,
                                  texts: dict[str, str]) -> dict:
    """Expand one-word party aliases to short caption-confirmed names."""
    output = {}
    for doc_id, spans in predictions.items():
        text = texts[doc_id]
        caption = text[:1800]
        boundary = re.search(r'\b(?:JUDGMENT|HEADNOTE)\b', caption, re.I)
        if boundary and boundary.start() > 100:
            caption = caption[:boundary.start()]
        markers = list(re.finditer(r'\b(?:vs\.?|versus|v\.)\b', caption, re.I))
        seeds = set()
        for start, end, label in spans:
            if label not in ('APP', 'RESP') or end > len(caption):
                continue
            if any((label == 'APP' and end <= marker.start()
                    and marker.start() - end <= 60)
                   or (label == 'RESP' and start >= marker.end()
                       and start - marker.end() <= 60)
                   for marker in markers):
                words = tuple(re.findall(r'[A-Za-z]+', text[start:end]))
                if (2 <= len(words) <= 3
                        and words[0].lower() not in PARTY_ALIAS_STOPWORDS):
                    seeds.add((label, words))
        chosen = [span.copy() for span in spans]
        for span in chosen:
            start, end, label = span
            surface = text[start:end]
            if (label not in ('APP', 'RESP')
                    or len(re.findall(r'[A-Za-z]+', surface)) != 1):
                continue
            best_end = end
            for seed_label, words in seeds:
                if seed_label != label or surface.lower() != words[0].lower():
                    continue
                for count in range(2, len(words) + 1):
                    pattern = re.compile(
                        r'(?<!\w)' + r'\s+'.join(map(re.escape, words[:count]))
                        + r'(?!\w)', re.I)
                    match = pattern.match(text, start)
                    if match:
                        best_end = max(best_end, match.end())
            if (best_end > end and not any(start < other[1]
                    and best_end > other[0] for other in chosen
                    if other is not span)):
                span[1] = best_end
        output[doc_id] = sorted(chosen)
    return output


def add_nonoverlapping_judges(primary: dict, bigru: dict) -> dict:
    """Keep both models' JUDGE spans when they do not overlap an entity."""
    if set(primary) != set(bigru):
        raise ValueError('Judge source IDs differ')
    output = {}
    for doc_id, spans in primary.items():
        chosen = [span.copy() for span in spans if span[2] != 'JUDGE']
        judges = {tuple(span) for span in spans + bigru[doc_id]
                  if span[2] == 'JUDGE'}
        for start, end, label in sorted(judges):
            if not any(start < row[1] and end > row[0] for row in chosen):
                chosen.append([start, end, label])
        output[doc_id] = sorted(chosen)
    return output


def correct_repeated_witness_roles(predictions: dict, texts: dict[str, str],
                                   min_witness_votes: int = 5) -> dict:
    """Use repeated exact WIT tags to resolve contradictory party tags."""
    output = {}
    for doc_id, spans in predictions.items():
        text = texts[doc_id]
        votes = Counter(text[start:end] for start, end, label in spans
                        if label == 'WIT')
        output[doc_id] = [[start, end,
                           'WIT' if label in ('APP', 'RESP')
                           and votes[text[start:end]] >= min_witness_votes
                           else label]
                          for start, end, label in spans]
    return output


def improved(primary: dict, bigru: dict, texts: dict[str, str],
             long: dict | None = None, caption_party: bool = False,
             alias_types: frozenset[str] = frozenset(),
             judge_union: bool = False,
             witness_consistency: bool = False,
             corporate_boundaries: bool = False,
             party_prefixes: bool = False) -> dict:
    source = add_nonoverlapping_judges(primary, bigru) if judge_union else primary
    result = propagate_mentions(merge(source, bigru, {'CASENO'}), texts)
    if long is not None:
        if set(result) != set(long):
            raise ValueError('Long-context prediction IDs differ')
        for doc_id, spans in result.items():
            supported_dates = {tuple(span) for span in long[doc_id]
                               if span[2] == 'DATE'}
            result[doc_id] = [span for span in spans
                              if span[2] != 'DATE' or tuple(span) in supported_dates]
    if caption_party:
        result = caption_party_correction(result, texts)
    if alias_types:
        result = caption_party_aliases(result, texts, alias_types)
    if corporate_boundaries:
        result = complete_corporate_party_spans(result, texts)
    if party_prefixes:
        result = complete_short_party_prefixes(result, texts)
    if witness_consistency:
        result = correct_repeated_witness_roles(result, texts)
    return result


def validate(predictions: dict, texts: dict[str, str]) -> None:
    if set(predictions) != set(texts):
        raise ValueError('Official test IDs differ')
    for doc_id, spans in predictions.items():
        seen = set()
        for span in spans:
            if (not isinstance(span, list) or len(span) != 3
                    or type(span[0]) is not int or type(span[1]) is not int
                    or not 0 <= span[0] < span[1] <= len(texts[doc_id])
                    or span[2] not in LABELS or tuple(span) in seen):
                raise ValueError(f'Invalid span in {doc_id}: {span!r}')
            seen.add(tuple(span))


def validation_report(data_dir: Path, date_consensus: bool = False,
                      caption_party: bool = False,
                      alias_types: frozenset[str] = frozenset(),
                      judge_union: bool = False,
                      witness_consistency: bool = False,
                      corporate_boundaries: bool = False,
                      party_prefixes: bool = False) -> dict:
    rows = read_folds(data_dir)
    variants = [
        ('42',
         json.loads(Path('runs/E26_moe_bigru_label_router_20261011/validation_predictions.json').read_text()),
         json.loads(Path('runs/E25_moe_bigru_ctx1536_20261011/validation_predictions.json').read_text()),
         json.loads(Path('runs/E9_moe_lora_crf_train1024_20261011/validation_predictions.json').read_text())),
    ]
    for split in ('43', '49'):
        candidates = json.loads(Path(
            f'artifacts/E23_bigru_inference_sweep_20261011/split{split}_final.predictions.json'
        ).read_text())
        variants.append((split, candidates['final_aggressive'],
                         candidates['bigru'], candidates['long']))
    report = {}
    for split, primary, bigru, long in variants:
        ids = set(primary)
        if set(bigru) != ids:
            raise ValueError(f'Different validation IDs in split {split}')
        texts = {doc_id: rows[doc_id]['text'] for doc_id in ids}
        gold = {doc_id: rows[doc_id]['entities'] for doc_id in ids}
        folds = {doc_id: rows[doc_id]['fold'] for doc_id in ids}
        output = improved(primary, bigru, texts,
                          long if date_consensus else None, caption_party,
                          alias_types, judge_union, witness_consistency,
                          corporate_boundaries, party_prefixes)
        validate(output, texts)
        report[split] = {
            'validation_ids': sorted(ids),
            'primary': score_predictions(primary, gold, folds),
            'improved': score_predictions(output, gold, folds),
        }
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path,
                        default=Path('data/official/Phase_2_LNER'))
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--run-dir', type=Path,
                        help='Provenance directory; defaults to runs/<output name>')
    parser.add_argument('--date-consensus', action='store_true',
                        help='Keep a DATE only when the independent 1024-token model agrees')
    parser.add_argument('--caption-party', action='store_true',
                        help='Correct multiword APP/RESP names identified in the case caption')
    parser.add_argument('--caption-aliases', nargs='*', default=[],
                        choices=('first', 'first2', 'acronym', 'full'),
                        help='Add caption-anchored short party names')
    parser.add_argument('--judge-union', action='store_true',
                        help='Add nonoverlapping JUDGE spans from the BiGRU model')
    parser.add_argument('--witness-consistency', action='store_true',
                        help='Relabel contradictory party mentions with at least five WIT votes')
    parser.add_argument('--corporate-boundaries', action='store_true',
                        help='Complete caption-derived Ltd/ Limited party spans')
    parser.add_argument('--party-prefixes', action='store_true',
                        help='Complete short party mentions using two- or three-word caption names')
    args = parser.parse_args()
    run_dir = args.run_dir or Path('runs') / args.output_dir.name
    if args.output_dir.exists():
        parser.error('Output directory already exists')
    if run_dir.exists():
        parser.error('Run directory already exists')
    primary_path = Path('submissions/E26_moe_bigru_label_router_20261011/prediction.json')
    bigru_path = Path('submissions/E25_moe_bigru_ctx1536_20261011/prediction.json')
    long_path = Path('submissions/E9_moe_lora_crf_train1024_20261011/prediction.json')
    test_path = args.data_dir / 'Test/fold3-text.json'
    primary = json.loads(primary_path.read_text())
    bigru = json.loads(bigru_path.read_text())
    long = json.loads(long_path.read_text()) if args.date_consensus else None
    texts = json.loads(test_path.read_text())
    alias_types = frozenset(args.caption_aliases)
    prediction = improved(primary, bigru, texts, long, args.caption_party,
                          alias_types, args.judge_union, args.witness_consistency,
                          args.corporate_boundaries, args.party_prefixes)
    validate(prediction, texts)
    report = validation_report(args.data_dir, args.date_consensus,
                               args.caption_party, alias_types, args.judge_union,
                               args.witness_consistency, args.corporate_boundaries,
                               args.party_prefixes)
    args.output_dir.mkdir(parents=True)
    output = args.output_dir / 'prediction.json'
    output.write_text(json.dumps(prediction, ensure_ascii=False, indent=2) + '\n')
    archive = args.output_dir / 'submission.zip'
    with ZipFile(archive, 'w', ZIP_DEFLATED, compresslevel=9) as stream:
        stream.write(output, arcname='prediction.json')
    with ZipFile(archive) as stream:
        if stream.namelist() != ['prediction.json'] or stream.testzip() is not None:
            raise RuntimeError('Invalid submission archive')
    manifest = {
        'method': 'E26 with BiGRU case-number route and within-document exact mention propagation',
        'primary_prediction': str(primary_path.resolve()),
        'primary_sha256': sha256(primary_path),
        'bigru_prediction': str(bigru_path.resolve()),
        'bigru_sha256': sha256(bigru_path),
        'date_consensus': args.date_consensus,
        'caption_party_correction': args.caption_party,
        'caption_aliases': sorted(alias_types),
        'judge_union': args.judge_union,
        'witness_consistency': args.witness_consistency,
        'minimum_witness_votes': 5 if args.witness_consistency else None,
        'corporate_boundaries': args.corporate_boundaries,
        'party_prefixes': args.party_prefixes,
        'long_prediction': str(long_path.resolve()) if args.date_consensus else None,
        'long_sha256': sha256(long_path) if args.date_consensus else None,
        'test_sha256': sha256(test_path),
        'prediction_sha256': sha256(output),
        'submission_zip_sha256': sha256(archive),
        'source_code_sha256': sha256(Path(__file__)),
        'documents': len(prediction),
        'predicted_entities': sum(map(len, prediction.values())),
        'propagate_labels': sorted(PROPAGATE_LABELS),
        'minimum_mention_length': MIN_MENTION_LENGTH,
        'local_validation': report,
        'official_score': None,
    }
    (args.output_dir / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    run_dir.mkdir(parents=True)
    source_config = json.loads(Path(
        'runs/E26_moe_bigru_label_router_20261011/config.json').read_text())
    (run_dir / 'config.json').write_text(json.dumps({
        'method': 'exact_mention_consistency_and_label_routing',
        'backbone_kind': 'moe',
        'structure': 'bilou_crf',
        'training_ids': source_config['training_ids'],
        'validation_ids': source_config['validation_ids'],
        'source_manifest': str((args.output_dir / 'manifest.json').resolve()),
        'propagate_labels': sorted(PROPAGATE_LABELS),
        'minimum_mention_length': MIN_MENTION_LENGTH,
        'date_consensus': args.date_consensus,
        'caption_party_correction': args.caption_party,
        'caption_aliases': sorted(alias_types),
        'judge_union': args.judge_union,
        'witness_consistency': args.witness_consistency,
        'minimum_witness_votes': 5 if args.witness_consistency else None,
        'corporate_boundaries': args.corporate_boundaries,
        'party_prefixes': args.party_prefixes,
    }, indent=2) + '\n')
    (run_dir / 'metrics.jsonl').write_text(json.dumps({
        'epoch': 0,
        'method': 'exact_mention_consistency_and_label_routing',
        'validation_macro_f1': report['42']['improved']['macro_f1'],
        'alternate_split_macro_f1': {
            split: report[split]['improved']['macro_f1'] for split in ('43', '49')},
    }) + '\n')
    print(json.dumps({split: {name: row[name]['macro_f1'] for name in ('primary', 'improved')}
                      for split, row in report.items()}, indent=2), flush=True)
    print(f'Saved {len(prediction)} official test IDs to {archive}', flush=True)


if __name__ == '__main__':
    main()
