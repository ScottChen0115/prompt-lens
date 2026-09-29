"""Pretrained fine-tuning diagnostic. Never produces valid misunderstanding probabilities."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import platform
import random
import sys
import time
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from risk_model.data import validate, encode, digest
import numpy as np
import torch
from scipy.optimize import minimize_scalar
from scipy.special import expit
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss, average_precision_score
from transformers import BertForSequenceClassification, BertTokenizerFast


def fixtures():
    rows = []
    # These labels deliberately describe fixture patterns, NOT observed assistant errors.
    verbs = ['Move', 'Reschedule', 'Shift', 'Change', 'Rearrange', 'Postpone', 'Update', 'Reset']
    objects = ['meeting', 'appointment', 'reminder', 'task', 'call', 'review', 'visit', 'session']
    for family, (verb, obj) in enumerate((v, o) for v in verbs for o in objects):
        split = ('train' if family < 40 else 'validation' if family < 48 else
                 'calibration' if family < 56 else 'test')
        for i in range(8):
            label = i % 2
            rows.append({'id': f'fixture-{family:02d}-{i}', 'group_id': f'family-{family:02d}',
                'context': f'{obj.capitalize()} A is at {8+i}:00. '+
                    (f'{obj.capitalize()} B is at {9+i}:30.' if label else 'Only one item exists.'),
                'request': f'{verb} '+('it' if label else f'{obj} A')+f' to {15+i}:00.',
                'label': label, 'split': split, 'source': 'synthetic_pipeline_fixture'})
    return rows


def metrics(y, p):
    return {'brier': float(brier_score_loss(y, p)),
            'log_loss': float(log_loss(y, p, labels=[0, 1])),
            'average_precision': float(average_precision_score(y, p))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--device', choices=['cpu', 'mps'], default='cpu')
    ap.add_argument('--epochs', type=int, default=4)
    ap.add_argument('--seed', type=int, default=42)
    args = ap.parse_args()
    if args.epochs < 1:
        ap.error('epochs must be positive')
    started = time.monotonic()
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    torch.set_num_threads(4)
    if args.device == 'mps' and not torch.backends.mps.is_available():
        raise RuntimeError('MPS unavailable; no silent device fallback')
    run_id = datetime.now(timezone.utc).strftime('development-%Y%m%dT%H%M%SZ')
    out = ROOT / 'outputs/runs' / run_id
    out.mkdir(parents=True, exist_ok=False)
    rows = validate(fixtures(), development=True)
    (out/'fixtures.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
    checkpoint = json.loads((ROOT/'configs/checkpoint.json').read_text())
    path = checkpoint['local_path']
    tokenizer = BertTokenizerFast(vocab=str(Path(path)/'vocab.txt'), do_lower_case=True)
    assert len(tokenizer) == 30522
    model = BertForSequenceClassification.from_pretrained(path, num_labels=1,
                 local_files_only=True).to(args.device)
    initial = model.classifier.weight.detach().cpu().clone()
    subsets = {s: [r for r in rows if r['split'] == s] for s in ('train','validation','calibration')}
    # Held-out test features/labels are not fed to training, selection, or calibration.
    encoded = {s: encode(tokenizer, rs) for s, rs in subsets.items()}
    labels = {s: np.array([r['label'] for r in rs]) for s, rs in subsets.items()}
    def logits(split):
        model.eval()
        with torch.no_grad():
            return model(**{k:v.to(args.device) for k,v in encoded[split].items()}).logits.flatten().cpu().numpy()
    def text(rs):
        return [r['context']+' [SEP] '+r['request'] for r in rs]
    vectorizer = TfidfVectorizer(ngram_range=(1,2))
    x = vectorizer.fit_transform(text(subsets['train']))
    baseline = LogisticRegression(C=1, random_state=args.seed).fit(x, labels['train'])
    baseline_metrics = {
        'constant': metrics(labels['validation'], np.full(len(labels['validation']), labels['train'].mean())),
        'tfidf': metrics(labels['validation'], baseline.predict_proba(vectorizer.transform(text(subsets['validation'])))[:,1])}
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=.01)
    history, best, best_state = [], float('inf'), None
    for epoch in range(args.epochs):
        model.train(); losses=[]
        for indices in torch.randperm(len(subsets['train'])).split(32):
            batch={k:v[indices].to(args.device) for k,v in encoded['train'].items()}
            target=torch.tensor(labels['train'][indices.numpy()], dtype=torch.float32, device=args.device)
            optimizer.zero_grad()
            loss=torch.nn.functional.binary_cross_entropy_with_logits(model(**batch).logits.flatten(), target)
            if not torch.isfinite(loss): raise RuntimeError('Nonfinite loss')
            loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),1.0, error_if_nonfinite=True)
            optimizer.step(); losses.append(loss.item())
        val=metrics(labels['validation'],expit(logits('validation')))
        history.append({'epoch':epoch+1,'train_loss':float(np.mean(losses)), 'validation':val})
        print(json.dumps(history[-1]),flush=True)
        if val['log_loss'] < best:
            best=val['log_loss']; best_state=copy.deepcopy({k:v.cpu() for k,v in model.state_dict().items()})
    assert not torch.equal(initial, model.classifier.weight.detach().cpu())
    model.load_state_dict(best_state)
    cal_logits = logits('calibration')
    fit=minimize_scalar(lambda logt:log_loss(labels['calibration'],expit(cal_logits/np.exp(logt)),labels=[0,1]),bounds=(-3,3),method='bounded')
    if not fit.success: raise RuntimeError('Calibration optimizer failed')
    temperature=float(np.exp(fit.x))
    # Calibration itself is an infrastructure exercise on artificial data.
    model.cpu().eval()
    saved=out/'model'; model.save_pretrained(saved, safe_serialization=True); tokenizer.save_pretrained(saved)
    reloaded=BertForSequenceClassification.from_pretrained(saved,local_files_only=True).eval()
    tok2=BertTokenizerFast.from_pretrained(saved,local_files_only=True)
    reencoded=encode(tok2,subsets['validation'])
    with torch.no_grad():
        expected=model(**encoded['validation']).logits
        actual=reloaded(**reencoded).logits
    torch.testing.assert_close(expected,actual)
    manifest={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in
              [ROOT/'scripts/train_development.py',ROOT/'src/risk_model/data.py',ROOT/'uv.lock']}
    report={'run_id':run_id,'status':'completed_development_only','valid_risk_probability':False,
        'warning':'Synthetic fixture labels are not observed assistant misunderstanding. No research gates passed.',
        'device':args.device,'platform':platform.platform(),'seconds':time.monotonic()-started,
        'seed':args.seed,'checkpoint':checkpoint,'parameters':sum(p.numel() for p in model.parameters()),
        'splits':{s:sum(r['split']==s for r in rows) for s in ('train','validation','calibration','test')},
        'test_evaluated':False,'dataset_sha256':digest(rows),'code_sha256':manifest,'history':history,
        'baseline_validation_diagnostics':baseline_metrics,'temperature_development_only':temperature,
        'reload_parity':'passed','weights_changed':True,'package_bytes':sum(p.stat().st_size for p in saved.iterdir() if p.is_file())}
    (out/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    (saved/'risk-policy.json').write_text(json.dumps({'status':'development_only','probability':None,
        'temperature':temperature,'threshold':None,'deployment_allowed':False},indent=2)+'\n')
    (saved/'README.md').write_text('# Development checkpoint — not a risk estimator\n\nFine-tuned on deliberately artificial fixtures. Do not use the sigmoid output as a probability of misunderstanding.\n\nBase: '+checkpoint['repo']+' @ '+checkpoint['revision']+' (Apache-2.0).\n\nNo observed target-assistant outcomes, independent annotation pilot, deployment calibration, or locked evaluation exists.\n')
    (out/'Report.md').write_text('# Development training completed\n\nThis run verifies pretrained fine-tuning and artifact handling only. Synthetic losses and metrics cannot establish misunderstanding prediction quality.\n\n'+f'- Device: {args.device}\n- Parameters: {report["parameters"]:,}\n- Epochs: {args.epochs}\n- Train / validation / calibration / untouched test fixtures: 320 / 64 / 64 / 64\n- Elapsed: {report["seconds"]:.1f} seconds\n- Weights changed; tokenizer and model save/reload parity passed.\n- Full reproducibility metadata and diagnostic metrics: `report.json`.\n- Research blocker: frozen target-assistant trials and independently reviewed outcome labels.\n')
    (ROOT/'outputs/development/latest.json').write_text(json.dumps({'run_path':str(out)},indent=2)+'\n')
    print(str(out),flush=True)

if __name__ == '__main__': main()
