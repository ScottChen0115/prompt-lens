"""Authorized local fine-tuning with supplied weak labels. Locked tests are never loaded."""
import os
os.environ['HF_HUB_OFFLINE']='1';os.environ['TOKENIZERS_PARALLELISM']='false'
from pathlib import Path
import sys,json,time,hashlib,math,gc,random
from datetime import datetime,timezone
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
import numpy as np
import torch
from scipy.special import expit,logit
from scipy.optimize import minimize_scalar
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from transformers import AutoModelForSequenceClassification,AutoTokenizer,get_linear_schedule_with_warmup
from risk_model.learning import targets,masked_loss_sum,metrics,slices
from risk_model.quality import LABELS
RUN=None


def write(path,value):
    temp=path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(value,ensure_ascii=False,indent=2,default=lambda x:sorted(x) if isinstance(x,set) else str(x))+'\n')
    temp.replace(path)


def main():
    global RUN
    cfg=json.loads((ROOT/'configs/training-supplied.json').read_text())
    run_id=datetime.now(timezone.utc).strftime('supplied-%Y%m%dT%H%M%SZ')
    RUN=ROOT/'outputs/runs'/run_id;RUN.mkdir(parents=True,exist_ok=False)
    write(RUN/'status.json',{'status':'running','phase':'prepare','run_id':run_id})
    write(ROOT/'outputs/dataset-audit/active-run.json',{'run_path':str(RUN)})
    write(RUN/'training-config.json',cfg)
    start=time.monotonic();random.seed(cfg['seed']);np.random.seed(cfg['seed']);torch.manual_seed(cfg['seed']);torch.set_num_threads(4)
    if not torch.backends.mps.is_available():raise RuntimeError('MPS required for this full training run')
    device='mps';data={};data_hashes={}
    for split in ('train','validation','calibration'):
        data[split]=[]
        for source in cfg['sources']:
            f=ROOT/cfg['data_root']/source/(split+'.jsonl')
            data_hashes[str(f.relative_to(ROOT))]=hashlib.sha256(f.read_bytes()).hexdigest()
            data[split].extend(json.loads(l) for l in f.read_text().splitlines())
    checkpoint=json.loads((ROOT/cfg['checkpoint']).read_text())
    tok=AutoTokenizer.from_pretrained(checkpoint['local_path'],local_files_only=True,trust_remote_code=False)
    enc={};y={};known={}
    for split,rows in data.items():
        enc[split]=tok([r['context'] for r in rows],[r['request'] for r in rows],truncation=False)
        if any(len(ids)>cfg['max_length'] for ids in enc[split]['input_ids']):raise ValueError('Length gate failed')
        y[split],known[split]=targets(rows)
    def batch(split,indices):
        return {k:v.to(device) for k,v in tok.pad([{k:enc[split][k][int(i)] for k in enc[split]} for i in indices],padding=True,return_tensors='pt').items()}
    def predict(model,split):
        model.eval();arr=[]
        with torch.inference_mode():
            for i in range(0,len(data[split]),16):
                ids=range(i,min(i+16,len(data[split])))
                arr.append(model(**batch(split,ids)).logits.float().cpu().numpy())
        return np.concatenate(arr)
    # Cheap baseline fitted on train only, with unknowns masked independently.
    vectorizer=TfidfVectorizer(analyzer='char',ngram_range=(2,4),max_features=50000,sublinear_tf=True)
    text=lambda rs:[r['context']+'\n'+r['request'] for r in rs]
    xtrain=vectorizer.fit_transform(text(data['train']));xval=vectorizer.transform(text(data['validation']))
    baseline_p=np.zeros(y['validation'].shape)
    for j in range(5):
        classifier=LogisticRegression(C=1,max_iter=400,random_state=cfg['seed'])
        classifier.fit(xtrain[known['train'][:,j]],y['train'][known['train'][:,j],j])
        baseline_p[:,j]=classifier.predict_proba(xval)[:,1]
    baseline=slices(data['validation'],baseline_p)
    write(RUN/'baseline-validation.json',baseline)
    np.savez_compressed(RUN/'baseline-validation-predictions.npz',ids=[r['id'] for r in data['validation']],probabilities=baseline_p)
    del xtrain,xval,vectorizer,classifier;gc.collect()
    print(json.dumps({'phase':'baseline','validation':baseline['overall']}),flush=True)
    # Start from the pinned pretrained encoder, not from the former fixture-trained head.
    model,loading=AutoModelForSequenceClassification.from_pretrained(checkpoint['local_path'],local_files_only=True,
        trust_remote_code=False,num_labels=5,id2label=dict(enumerate(LABELS)),label2id={s:i for i,s in enumerate(LABELS)},
        problem_type='multi_label_classification',output_loading_info=True)
    if any(not k.startswith('classifier.') for k in loading.get('missing_keys',[])) or loading.get('mismatched_keys'):
        raise RuntimeError('Unexpected backbone load mismatch')
    count=sum(p.numel() for p in model.parameters());assert 100000000<=count<=300000000
    model.to(device)
    before={'head':model.classifier.out_proj.weight.detach().cpu().clone(),
            'backbone':model.roberta.encoder.layer[0].attention.self.query.weight.detach().cpu().clone()}
    optimizer=torch.optim.AdamW(model.parameters(),lr=cfg['learning_rate'],weight_decay=cfg['weight_decay'])
    effective=cfg['micro_batch_size']*cfg['gradient_accumulation_steps']
    steps_per_epoch=math.ceil(len(data['train'])/effective)
    schedule=get_linear_schedule_with_warmup(optimizer,int(cfg['warmup_fraction']*steps_per_epoch*cfg['epochs']),steps_per_epoch*cfg['epochs'])
    best=float('inf');best_epoch=0;bad=0;history=[];step=0
    for epoch in range(cfg['epochs']):
        model.train();total_loss=0.;total_known=0
        order=np.random.permutation(len(data['train']))
        for offset in range(0,len(order),effective):
            group=order[offset:offset+effective];denominator=int(known['train'][group].sum())
            if denominator==0:continue
            optimizer.zero_grad(set_to_none=True)
            for micro_start in range(0,len(group),cfg['micro_batch_size']):
                ids=group[micro_start:micro_start+cfg['micro_batch_size']]
                target=torch.tensor(y['train'][ids],device=device)
                mask=torch.tensor(known['train'][ids],dtype=torch.float32,device=device)
                loss_sum=masked_loss_sum(model(**batch('train',ids)).logits,target,mask)
                if not torch.isfinite(loss_sum):raise RuntimeError('Nonfinite loss')
                (loss_sum/denominator).backward();total_loss+=loss_sum.item();total_known+=int(mask.sum().item())
            torch.nn.utils.clip_grad_norm_(model.parameters(),cfg['gradient_clip'],error_if_nonfinite=True)
            optimizer.step();schedule.step();step+=1
            if step%25==0:
                status={'status':'running','phase':'training','epoch':epoch+1,'max_epochs':cfg['epochs'],
                        'step':step,'steps_per_epoch':steps_per_epoch,'train_loss':total_loss/max(1,total_known),
                        'elapsed_seconds':time.monotonic()-start}
                write(RUN/'status.json',status);print(json.dumps(status),flush=True)
        z=predict(model,'validation');result=slices(data['validation'],expit(z))
        history.append({'epoch':epoch+1,'train_loss':total_loss/total_known,'validation':result})
        write(RUN/'history.json',history)
        print(json.dumps({'phase':'epoch_complete','epoch':epoch+1,'validation':result['overall']}),flush=True)
        if result['overall']['mean_log_loss']<best:
            best=result['overall']['mean_log_loss'];best_epoch=epoch+1;bad=0
            model.save_pretrained(RUN/'model',safe_serialization=True);tok.save_pretrained(RUN/'model')
            np.savez_compressed(RUN/'validation-best.npz',ids=[r['id'] for r in data['validation']],logits=z)
        else:
            bad+=1
            if bad>=cfg['early_stopping_patience']:break
    changed={'head':not torch.equal(before['head'],model.classifier.out_proj.weight.detach().cpu()),
        'backbone':not torch.equal(before['backbone'],model.roberta.encoder.layer[0].attention.self.query.weight.detach().cpu())}
    assert all(changed.values())
    del optimizer,schedule,model;gc.collect();torch.mps.empty_cache()
    model=AutoModelForSequenceClassification.from_pretrained(RUN/'model',local_files_only=True).to(device).eval()
    zval=predict(model,'validation');saved_z=np.load(RUN/'validation-best.npz')['logits']
    np.testing.assert_allclose(zval,saved_z,rtol=2e-4,atol=2e-4)
    # Operating thresholds selected only on validation before calibration is opened for fitting.
    raw_thresholds=[]
    for j in range(5):
        yy=y['validation'][known['validation'][:,j],j];pp=expit(zval[known['validation'][:,j],j])
        candidates=np.linspace(.05,.95,91)
        scores=[f1_score(yy,pp>=t,zero_division=0) for t in candidates]
        best_i=max(range(len(candidates)),key=lambda i:(scores[i],candidates[i]))
        raw_thresholds.append(float(candidates[best_i]))
    zcal=predict(model,'calibration');temperatures=[]
    for j in range(5):
        zz=zcal[known['calibration'][:,j],j];yy=y['calibration'][known['calibration'][:,j],j]
        fit=minimize_scalar(lambda lt:float(np.mean(np.logaddexp(0,zz/np.exp(lt))-yy*zz/np.exp(lt))),bounds=(-2,2),method='bounded')
        if not fit.success:raise RuntimeError('Calibration fit failed')
        temperatures.append(float(np.exp(fit.x)))
    thresholds=expit(logit(np.array(raw_thresholds))/np.array(temperatures)).tolist()
    validation=slices(data['validation'],expit(zval/np.array(temperatures)),thresholds)
    calibration=slices(data['calibration'],expit(zcal/np.array(temperatures)),thresholds)
    uncalibrated=slices(data['validation'],expit(zval),raw_thresholds)
    old=json.loads((ROOT/'outputs/dataset-audit/current-model-validation.json').read_text())
    old_cells={source+'/'+lang:np.average([v['log_loss'] for v in old['results'][source]['by_language'][lang].values()], weights=[v['eligible'] for v in old['results'][source]['by_language'][lang].values()])
        for source in cfg['sources'] for lang in ('en','zh')}
    # Compare same uncalibrated metric to incumbent and cheap baseline; calibration cannot rescue a poor encoder.
    gate={'better_than_old_each_source_language':all(uncalibrated['source_language'][k]['mean_log_loss']<v for k,v in old_cells.items()),
          'better_than_tfidf_log_loss':uncalibrated['overall']['mean_log_loss']<baseline['overall']['mean_log_loss']}
    policy={'status':'development_only','quality_score':None,'issue_thresholds':thresholds,'temperature':temperatures,
        'deployment_allowed':False,'label_order':list(LABELS),'languages':['en','zh'],
        'supervision':'machine-generated labels; no independent human gold',
        'warning':'Experimental weak-label classifier. Calibration and thresholds describe synthetic rubric labels only, not actual user quality or GPT-6 Sol error probability.'}
    write(RUN/'model/quality-policy.json',policy)
    code=['scripts/train_supplied.py','src/risk_model/learning.py','src/risk_model/supplied_data.py',
          'scripts/audit_supplied_data.py','configs/training-supplied.json','uv.lock']
    report={'run_id':run_id,'status':'completed_development_only','config':cfg,'checkpoint':checkpoint,'parameters':count,
        'selected_epoch':best_epoch,'epochs_completed':len(history),'optimizer_steps':step,'weights_changed':changed,
        'validation':validation,'uncalibrated_validation':uncalibrated,'calibration_fit_diagnostics':calibration,
        'baseline_validation':baseline,'promotion_gate':gate,'old_validation_cell_log_loss':old_cells,
        'dataset_counts':{k:len(v) for k,v in data.items()},'dataset_hashes':data_hashes,
        'code_sha256':{f:hashlib.sha256((ROOT/f).read_bytes()).hexdigest() for f in code},
        'model_reload_parity':'passed','test_evaluated':False,'quality_score_validated':False,
        'device':device,'seconds':time.monotonic()-start}
    write(RUN/'report.json',report)
    np.savez_compressed(RUN/'calibration-logits.npz',ids=[r['id'] for r in data['calibration']],logits=zcal)
    (RUN/'model/README.md').write_text('# Experimental bilingual request-quality model\n\n'+policy['warning']+
        '\n\nXLM-RoBERTa-base, MIT base model, 278,047,493 parameters. Source datasets remain user-provided; no redistribution license is inferred.\n')
    hashes={f.name:hashlib.sha256(f.read_bytes()).hexdigest() for f in (RUN/'model').iterdir() if f.is_file()}
    write(RUN/'frozen-candidate.json',{'created_at':datetime.now(timezone.utc).isoformat(),'model_hashes':hashes,
        'thresholds':thresholds,'temperatures':temperatures,'gate':gate,'status':'frozen_before_test'})
    write(RUN/'status.json',{'status':'completed_development_only','phase':'frozen_for_evaluation','run_id':run_id,'gate':gate})
    print(json.dumps({'run_path':str(RUN),'selected_epoch':best_epoch,'gate':gate,'validation':validation['overall']}),flush=True)

if __name__=='__main__':
    try:main()
    except BaseException as exc:
        if RUN:write(RUN/'status.json',{'status':'failed','error':str(exc)})
        raise
