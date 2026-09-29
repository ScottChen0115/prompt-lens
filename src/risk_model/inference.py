"""Read-only, local inference for the experimental bilingual checkpoint."""
import json
import math
from pathlib import Path
import time

from .quality import LABELS

ROOT = Path(__file__).resolve().parents[2]
MAX_CHARACTERS = 12000


class InputError(ValueError):
    def __init__(self, code, **details):
        self.code, self.details = code, details
        super().__init__(code)


def validate_input(payload):
    if not isinstance(payload, dict):
        raise InputError('invalid_input')
    request, context = payload.get('request'), payload.get('context', '')
    if not isinstance(request, str) or not isinstance(context, str):
        raise InputError('invalid_input')
    if not request.strip():
        raise InputError('empty_request')
    if len(request) + len(context) > MAX_CHARACTERS:
        raise InputError('too_many_characters', limit=MAX_CHARACTERS)
    return context, request


class QualityPredictor:
    def __init__(self, root=ROOT):
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        self.torch = torch
        torch.set_num_threads(4)
        root = Path(root)
        run = Path(json.loads((root/'outputs/development/latest-bilingual.json').read_text())['run_path']).resolve()
        if not run.is_relative_to((root/'outputs/runs').resolve()):
            raise ValueError('Model pointer must refer to a project run')
        report = json.loads((run/'report.json').read_text())
        if report['status'] != 'completed_development_only':
            raise ValueError('A successfully verified checkpoint is required')
        model_dir = run/'model'
        policy = json.loads((model_dir/'quality-policy.json').read_text())
        if tuple(policy['label_order']) != LABELS:
            raise ValueError('Unexpected label order')
        temperatures = policy.get('temperature', [1.0] * len(LABELS))
        if len(temperatures) != len(LABELS) or any(not isinstance(t, (int, float)) or not math.isfinite(t) or t <= 0 for t in temperatures):
            raise ValueError('Invalid calibration temperatures')
        self.temperature = torch.tensor(temperatures, dtype=torch.float32)
        self.max_length = report['config']['max_length']
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True, trust_remote_code=False)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_dir, local_files_only=True, trust_remote_code=False)
        if tuple(self.model.config.id2label[i] for i in range(len(LABELS))) != LABELS:
            raise ValueError('Checkpoint label order mismatch')
        self.model.eval().requires_grad_(False)
        self.metadata = {'type': 'ready', 'model': 'XLM-RoBERTa', 'parameters': report['parameters'],
                         'run_id': report['run_id'], 'max_tokens': self.max_length, 'device': 'cpu',
                         'experimental': True}

    def predict(self, payload):
        context, request = validate_input(payload)
        start = time.monotonic()
        batch = self.tokenizer(context, request, truncation=False, return_tensors='pt')
        count = batch['input_ids'].shape[1]
        if count > self.max_length:
            raise InputError('too_many_tokens', tokens=count, limit=self.max_length)
        with self.torch.inference_mode():
            values = (self.model(**batch).logits.squeeze(0) / self.temperature).sigmoid().tolist()
        if len(values) != len(LABELS) or any(not 0 <= v <= 1 for v in values):
            raise RuntimeError('Invalid model output')
        return {'type': 'result', 'signals': [{'id': label, 'value': value} for label, value in zip(LABELS, values)],
                'tokens': count, 'elapsed_ms': round((time.monotonic()-start)*1000),
                'experimental': True, 'quality_score': None}
