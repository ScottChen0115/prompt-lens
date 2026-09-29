"""Read-only evaluation of current checkpoint on development validation, never locked tests."""
import os
os.environ['HF_HUB_OFFLINE']='1'
os.environ['TOKENIZERS_PARALLELISM']='false'
import json,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
import numpy as np
import torch
from sklearn.metrics import average_precision_score
from transformers import AutoTokenizer,AutoModelForSequenceClassification
from risk_model.quality import LABELS


def metrics(rows, probs):
    result={}
    for j,label in enumerate(LABELS):
        mask=np.array([r['labels'][j] is not None for r in rows])
        y=np.array([r['labels'][j] for r in rows if r['labels'][j] is not None],dtype=float)
        p=np.clip(probs[mask,j],1e-7,1-1e-7)
        result[label]={'eligible':len(y),'positives':int(y.sum()),'unknown':int((~mask).sum()),
            'brier':float(np.mean((p-y)**2)),'log_loss':float(-np.mean(y*np.log(p)+(1-y)*np.log(1-p))),
            'average_precision':float(average_precision_score(y,p)) if y.sum()>0 else None}
    return result


def main():
    torch.set_num_threads(4)
    run=Path(json.loads((ROOT/'outputs/development/latest-bilingual.json').read_text())['run_path'])
    tok=AutoTokenizer.from_pretrained(run/'model',local_files_only=True)
    model=AutoModelForSequenceClassification.from_pretrained(run/'model',local_files_only=True).eval().requires_grad_(False)
    report={'checkpoint':str(run),'evaluation_role':'development_validation_only','training_performed':False,
        'human_ground_truth':False,'threshold_selected':False,'locked_tests_scored':False,'results':{}}
    start=time.monotonic()
    out=ROOT/'outputs/dataset-audit'
    for source in ('brq','rq'):
        rows=[json.loads(l) for l in (ROOT/'data/processed/supplied-v1'/source/'validation.jsonl').read_text().splitlines()]
        values=[]
        for i in range(0,len(rows),16):
            batch=rows[i:i+16]
            encoded=tok([r['context'] for r in batch],[r['request'] for r in batch],padding=True,truncation=False,return_tensors='pt')
            if encoded['input_ids'].shape[1]>256:raise ValueError('Unsupported input escaped audit')
            with torch.inference_mode():values.extend(model(**encoded).logits.sigmoid().tolist())
        predictions=np.array(values)
        report['results'][source]={'rows':len(rows),'overall':metrics(rows,predictions),'by_language':{}}
        for lang in ('en','zh'):
            mask=np.array([r['language']==lang for r in rows])
            report['results'][source]['by_language'][lang]=metrics([r for r in rows if r['language']==lang],predictions[mask])
        (out/(source+'-current-validation-predictions.jsonl')).write_text(''.join(json.dumps({'id':r['id'],'signals':v.tolist()})+'\n' for r,v in zip(rows,predictions)))
        print(source,json.dumps(report['results'][source]['overall']),flush=True)
    report['seconds']=time.monotonic()-start
    (out/'current-model-validation.json').write_text(json.dumps(report,indent=2)+'\n')
    print('Read-only validation complete',flush=True)

if __name__=='__main__':main()
