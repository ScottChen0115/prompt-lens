"""Full-parameter XLM-R development fine-tuning. No gold labels or deployment scores."""
import argparse
import gc
import hashlib
import json
import platform
import random
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
import numpy as np
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from risk_model.data import digest, encode
from risk_model.quality import LABELS, fixtures, validate_development


RUN_PATH = None

def json_default(value):
    if isinstance(value, set):
        return sorted(value)
    raise TypeError(f'Unsupported JSON type: {type(value).__name__}')

def evaluate_metrics(targets, logits):
    t = torch.tensor(targets, dtype=torch.float32)
    z = torch.tensor(logits, dtype=torch.float32)
    return {'mean_binary_cross_entropy': torch.nn.functional.binary_cross_entropy_with_logits(z,t).item(),
            'mean_brier': torch.mean((z.sigmoid()-t)**2).item()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--device', choices=['cpu','mps'], default='cpu')
    ap.add_argument('--config', default='configs/training-bilingual.json')
    args = ap.parse_args()
    cfg = json.loads((ROOT/args.config).read_text())
    assert cfg['development_only'] is True and tuple(cfg['labels']) == LABELS
    assert cfg['epochs'] > 0 and cfg['micro_batch_size'] > 0 and cfg['gradient_accumulation_steps'] > 0
    checkpoint = json.loads((ROOT/cfg['checkpoint']).read_text())
    torch.set_num_threads(4)
    torch.manual_seed(cfg['seed']); np.random.seed(cfg['seed']); random.seed(cfg['seed'])
    if args.device == 'mps' and not torch.backends.mps.is_available():
        raise RuntimeError('MPS unavailable; no silent CPU fallback')
    run_id = datetime.now(timezone.utc).strftime('bilingual-development-%Y%m%dT%H%M%S%fZ')
    out = ROOT/'outputs/runs'/run_id
    out.mkdir(parents=True, exist_ok=False)
    global RUN_PATH
    RUN_PATH = out
    start = time.monotonic()
    def write(name, obj):
        (out/name).write_text(json.dumps(obj, ensure_ascii=False, indent=2, default=json_default)+'\n')
    write('status.json', {'status':'running','run_id':run_id})
    rows = validate_development(fixtures())
    (out/'fixtures.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows))
    write('training-config.json', cfg)
    tok = AutoTokenizer.from_pretrained(checkpoint['local_path'], local_files_only=True, trust_remote_code=False)
    # Probe both writing systems before spending compute on optimizer steps.
    probes = ['Please summarize document A.', '请总结甲文档。', '請總結甲文件。', '请review这份draft。']
    probe_tokens = [tok(p, add_special_tokens=False)['input_ids'] for p in probes]
    if any(not ids or tok.unk_token_id in ids for ids in probe_tokens):
        raise RuntimeError('Tokenizer probe failed for supported languages')
    model, loading = AutoModelForSequenceClassification.from_pretrained(
        checkpoint['local_path'], local_files_only=True, trust_remote_code=False,
        num_labels=len(LABELS), id2label=dict(enumerate(LABELS)),
        label2id={label:i for i,label in enumerate(LABELS)},
        problem_type='multi_label_classification', output_loading_info=True)
    missing = loading.get('missing_keys', [])
    if any(not key.startswith('classifier.') for key in missing) or loading.get('mismatched_keys') or loading.get('error_msgs'):
        raise RuntimeError(f'Unexpected pretrained backbone load failure: {loading}')
    count = sum(p.numel() for p in model.parameters())
    if not cfg['parameter_range'][0] <= count <= cfg['parameter_range'][1]:
        raise ValueError(f'Parameter budget violated: {count}')
    model.to(args.device)
    # Small snapshots verify BOTH backbone and output head were trained.
    probes_before = {'backbone': model.roberta.encoder.layer[0].attention.self.query.weight.detach().cpu().clone(),
                     'head': model.classifier.out_proj.weight.detach().cpu().clone()}
    subsets = {s:[r for r in rows if r['split']==s] for s in ('train','validation')}
    batches = {s:encode(tok,rs,cfg['max_length']) for s,rs in subsets.items()}
    y = {s:np.array([r['labels'] for r in rs]) for s,rs in subsets.items()}
    def logits(split):
        model.eval(); result=[]
        with torch.no_grad():
            for i in range(0,len(subsets[split]),cfg['micro_batch_size']):
                b={k:v[i:i+cfg['micro_batch_size']].to(args.device) for k,v in batches[split].items()}
                result.append(model(**b).logits.float().cpu())
        return torch.cat(result).numpy()
    tfidf = TfidfVectorizer(analyzer='char', ngram_range=(2,4), max_features=20000)
    text = lambda rs:[r['context']+'\n'+r['request'] for r in rs]
    baseline = OneVsRestClassifier(LogisticRegression(C=1,random_state=cfg['seed']))
    baseline.fit(tfidf.fit_transform(text(subsets['train'])), y['train'])
    base_p = baseline.predict_proba(tfidf.transform(text(subsets['validation'])))
    base_z = np.log(np.clip(base_p,1e-7,1-1e-7)/(1-np.clip(base_p,1e-7,1-1e-7)))
    baseline_result = evaluate_metrics(y['validation'],base_z)
    initial_metrics = evaluate_metrics(y['validation'],logits('validation'))
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg['learning_rate'],weight_decay=cfg['weight_decay'])
    best, best_state, best_epoch = float('inf'), None, None
    history=[]; steps=0
    print(json.dumps({'run_id':run_id,'parameters':count,'device':args.device,
                      'rows':{k:len(v) for k,v in subsets.items()},'initial_validation':initial_metrics}),flush=True)
    for epoch in range(cfg['epochs']):
        model.train(); losses=[]
        indices = torch.randperm(len(subsets['train']))
        effective = cfg['micro_batch_size']*cfg['gradient_accumulation_steps']
        for group in indices.split(effective):
            optimizer.zero_grad(set_to_none=True)
            for micro in group.split(cfg['micro_batch_size']):
                b = {k:v[micro].to(args.device) for k,v in batches['train'].items()}
                targets = torch.tensor(y['train'][micro.numpy()],dtype=torch.float32,device=args.device)
                loss = torch.nn.functional.binary_cross_entropy_with_logits(model(**b).logits,targets)
                if not torch.isfinite(loss): raise RuntimeError('Nonfinite training loss')
                (loss*(len(micro)/len(group))).backward()
                losses.append(loss.item())
            torch.nn.utils.clip_grad_norm_(model.parameters(),cfg['gradient_clip'],error_if_nonfinite=True)
            optimizer.step(); steps+=1
            print(json.dumps({'epoch':epoch+1,'optimizer_step':steps,'recent_micro_loss':losses[-1]}),flush=True)
        z=logits('validation')
        metrics=evaluate_metrics(y['validation'],z)
        per_lang={language:evaluate_metrics(y['validation'][mask],z[mask])
                  for language in ('en','zh')
                  for mask in [np.array([r['language']==language for r in subsets['validation']])]}
        history.append({'epoch':epoch+1,'train_loss':float(np.mean(losses)),'validation':metrics,'by_language':per_lang})
        print(json.dumps(history[-1]),flush=True)
        write('history.json',history)
        if metrics['mean_binary_cross_entropy'] < best:
            best=metrics['mean_binary_cross_entropy']; best_epoch=epoch+1
            best_state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
    changed = {
        'backbone': not torch.equal(probes_before['backbone'],model.roberta.encoder.layer[0].attention.self.query.weight.detach().cpu()),
        'head': not torch.equal(probes_before['head'],model.classifier.out_proj.weight.detach().cpu())}
    assert all(changed.values())
    del optimizer
    model.cpu(); model.load_state_dict(best_state); del best_state
    gc.collect()
    if args.device=='mps': torch.mps.empty_cache()
    model.eval()
    # Reload parity on both languages, plus tokenizer Chinese variant coverage.
    parity_rows = [next(r for r in subsets['validation'] if r['language']==language) for language in ('en','zh')]
    parity_batch=encode(tok,parity_rows,cfg['max_length'])
    with torch.no_grad(): expected=model(**parity_batch).logits.clone()
    saved=out/'model'
    model.save_pretrained(saved,safe_serialization=True); tok.save_pretrained(saved)
    del model; gc.collect()
    reload_tok=AutoTokenizer.from_pretrained(saved,local_files_only=True,trust_remote_code=False)
    reload_model=AutoModelForSequenceClassification.from_pretrained(saved,local_files_only=True,trust_remote_code=False).eval()
    actual_batch=encode(reload_tok,parity_rows,cfg['max_length'])
    assert all(torch.equal(parity_batch[k],actual_batch[k]) for k in parity_batch)
    with torch.no_grad(): actual=reload_model(**actual_batch).logits
    torch.testing.assert_close(expected,actual)
    policy={'status':'development_only','quality_score':None,'issue_thresholds':None,'deployment_allowed':False,
            'label_order':list(LABELS),'languages':['en','zh'],
            'warning':'Unreviewed artificial fixtures; raw sigmoid outputs are not validated quality scores or misunderstanding probabilities.'}
    (saved/'quality-policy.json').write_text(json.dumps(policy,indent=2)+'\n')
    (saved/'README.md').write_text('# Bilingual request-quality development model\n\n'+policy['warning']+'\n\n'+
        f'Base: {checkpoint["repo"]} @ {checkpoint["revision"]}; MIT license.\n\n'+
        f'{count:,} parameters, five issue outputs. Full-parameter fine-tuning; classifier head newly initialized.\n\n'+
        'English/Chinese fixtures share generation templates, including translations. Separate language diagnostics do not establish generalization. Calibration and test fixtures were not evaluated. No free-form generation, overall quality score, action thresholds or GPT-6 Sol performance improvement is validated.\n')
    code_files=['scripts/train_bilingual.py','src/risk_model/quality.py','src/risk_model/data.py',args.config,cfg['checkpoint'],'uv.lock']
    report={'run_id':run_id,'status':'completed_development_only','config':cfg,'checkpoint':checkpoint,
            'parameters':count,'trainable_parameters':sum(p.numel() for p in reload_model.parameters() if p.requires_grad),
            'device':args.device,'platform':platform.platform(),'seconds':time.monotonic()-start,'optimizer_steps':steps,
            'base_loading_info':loading,'tokenizer_probe_lengths':dict(zip(['en','zh-Hans','zh-Hant','mixed'],map(len,probe_tokens))),
            'unknown_tokens_in_probes':0,'dataset_sha256':digest(rows),
            'split_counts':{s:{lang:sum(r['split']==s and r['language']==lang for r in rows) for lang in ('en','zh')}
                            for s in ('train','validation','calibration','test')},
            'calibration_evaluated':False,'test_evaluated':False,'quality_score_validated':False,
            'baseline_validation_diagnostics':baseline_result,'initial_validation':initial_metrics,'history':history,
            'selected_epoch':best_epoch,'weights_changed':changed,'tokenizer_reload_parity':'passed','model_reload_parity':'passed',
            'code_sha256':{f:hashlib.sha256((ROOT/f).read_bytes()).hexdigest() for f in code_files},
            'package_bytes':sum(p.stat().st_size for p in saved.iterdir() if p.is_file())}
    write('report.json',report)
    (out/'Report.md').write_text('# Bilingual development training completed\n\n'+
        f'- Model: {checkpoint["repo"]}\n- Parameters: {count:,} ({count/1e9:.3f}B), all trainable.\n'+
        f'- Device: {args.device}; epochs: {cfg["epochs"]}; optimizer steps: {steps}.\n'+
        '- Data: 144 unreviewed artificial English/Chinese fixtures; 96 train, 24 validation, 12 calibration, 12 test.\n'+
        '- Calibration and test were not evaluated. Translation families remained in the same split.\n'+
        '- Five issue outputs: unclear goal, missing information, unresolved reference, conflicting constraints, unclear deliverable.\n'+
        '- Backbone and head updated; saved tokenizer/model reload parity passed on English and Chinese.\n'+
        f'- Elapsed run time: {report["seconds"]:.1f}s; saved package: {report["package_bytes"]/1e9:.3f} GB.\n\n'+
        'This proves the bilingual architecture can be trained locally, not that it can reliably assess user requests. Fixtures are formulaic and unreviewed. Real quality, useful clarification and downstream GPT-6 Sol improvement remain unverified. See `report.json` for reproducibility and separate language diagnostics.\n')
    write('status.json',{'status':'completed_development_only','run_id':run_id})
    (ROOT/'outputs/development/latest-bilingual.json').write_text(json.dumps({'run_path':str(out)},indent=2)+'\n')
    print(str(out),flush=True)

if __name__=='__main__':
    try:
        main()
    except Exception as exc:
        if RUN_PATH is not None:
            (RUN_PATH/'status.json').write_text(json.dumps({'status':'failed','error':str(exc)},indent=2)+'\n')
        raise
