"""Strict input boundary for development fixtures and later reviewed outcomes."""
import hashlib
import json

OUTCOMES = {'match': 0, 'material_mismatch': 1, 'clarification_or_abstention': 0}

def validate(rows, *, development=False):
    ids, groups, texts = set(), {}, {}
    for row in rows:
        if row['id'] in ids:
            raise ValueError('Duplicate case ID')
        ids.add(row['id'])
        split = row['split']
        if split not in {'train', 'validation', 'calibration', 'test'}:
            raise ValueError('Unknown split')
        if groups.setdefault(row['group_id'], split) != split:
            raise ValueError('Group crosses splits')
        key = (row['context'].strip().lower(), row['request'].strip().lower())
        if texts.setdefault(key, split) != split:
            raise ValueError('Duplicate features cross splits')
        if not row['request'].strip():
            raise ValueError('Empty request')
        if development:
            if row['source'] != 'synthetic_pipeline_fixture' or row['label'] not in (0, 1):
                raise ValueError('Development mode accepts marked fixtures only')
        else:
            if row.get('source') == 'synthetic_pipeline_fixture':
                raise ValueError('Fixtures cannot become research labels')
            for field in ('source', 'license', 'assistant_revision', 'prompt_sha256',
                          'decoding_revision', 'intent_recorded_at', 'trial_recorded_at',
                          'hidden_intent', 'observed_interpretation'):
                if not row.get(field):
                    raise ValueError(f'Missing provenance: {field}')
            reviews = row.get('reviews', [])
            if len({r['reviewer_id'] for r in reviews if r.get('human') is True}) < 2:
                raise ValueError('Two independent human reviews required')
            if row.get('outcome') not in OUTCOMES:
                raise ValueError('Unjudgeable or unknown outcomes cannot be forced to zero')
            if any(r.get('outcome') != row['outcome'] for r in reviews) and not row.get('adjudication'):
                raise ValueError('Unresolved review disagreement')
            if row.get('label') != OUTCOMES[row['outcome']]:
                raise ValueError('Outcome/label mismatch')
    for split in ('train', 'validation', 'calibration'):
        if {r['label'] for r in rows if r['split'] == split} != {0, 1}:
            raise ValueError(f'{split} must contain both outcomes for this pilot')
    return rows

def features(row):
    # Never concatenate intent, assistant output, reviews, or later corrections.
    return row['context'], row['request']

def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()

def encode(tokenizer, rows, max_length=256):
    contexts, requests = zip(*(features(r) for r in rows))
    # Fail closed rather than silently dropping relevant context or the request.
    raw = tokenizer(list(contexts), list(requests), truncation=False)
    if any(len(x) > max_length for x in raw['input_ids']):
        raise ValueError('Unsupported long input; shorten through a reviewed policy')
    return tokenizer(list(contexts), list(requests), padding=True,
                     truncation=False, return_tensors='pt')
