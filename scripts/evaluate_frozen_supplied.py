"""One held-out evaluation of a frozen experimental model; never tunes model or policy."""
import os
os.environ['HF_HUB_OFFLINE']='1';os.environ['TOKENIZERS_PARALLELISM']='false'
from pathlib import Path
import sys,json,hashlib,zipfile,time
import numpy as np
import torch
from scipy.special import expit
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
from transformers import AutoTokenizer,AutoModelForSequenceClassification
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from risk_model.supplied_data import normalize
from risk_model.learning import slices
from risk_model.inference import validate_input,InputError

def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def readlines(text):return [json.loads(line) for line in text.splitlines() if line.strip()]
def main():
    run=Path(json.loads((ROOT/'outputs/dataset-audit/active-run.json').read_text())['run_path'])
    frozen=json.loads((run/'frozen-candidate.json').read_text())
    assert frozen['status']=='frozen_before_test'
    assert all(digest(run/'model'/name)==value for name,value in frozen['model_hashes'].items())
    output=run/'held-out-evaluation.json'
    repair='--repair-loader' in sys.argv
    if output.exists() and not repair:raise RuntimeError('Final evaluation already exists; do not reuse held-out tests for tuning')
    if repair:
        previous=json.loads(output.read_text())
        assert previous['frozen_manifest_sha256']==digest(run/'frozen-candidate.json')
        (run/'held-out-evaluation-before-loader-repair.json').write_text(json.dumps(previous,indent=2)+'\n')
    (run/'evaluation-started.json').write_text(json.dumps({'started_at_unix':time.time(),'frozen_manifest_sha256':digest(run/'frozen-candidate.json'),'purpose':'Final held-out evaluation; repairs may change loader only, never tune the candidate.'},indent=2)+'\n')
    tok=AutoTokenizer.from_pretrained(run/'model',local_files_only=True)
    torch.set_num_threads(4)
    model=AutoModelForSequenceClassification.from_pretrained(run/'model',local_files_only=True).eval().requires_grad_(False)
    # CPU is the exact application runtime.
    temperatures=np.array(frozen['temperatures']);thresholds=frozen['thresholds']
    datasets={};source_hashes={}
    for split in ('test','challenge'):
        p=ROOT/'test set/brq-v1'/('locked_test/test.jsonl' if split=='test' else 'data/challenge.jsonl')
        source_hashes[str(p.relative_to(ROOT))]=digest(p)
        datasets['brq/'+split]=readlines(p.read_text())
    zpath=ROOT/'test set/request_quality_v0.1_locked_test_RESTRICTED.zip'
    source_hashes[str(zpath.relative_to(ROOT))]=digest(zpath)
    with zipfile.ZipFile(zpath) as archive:
        blobs={Path(name).name:archive.read(name).decode() for name in archive.namelist() if name.endswith('.jsonl')}
    challenge=ROOT/'test set/request_quality_v0.1_challenge_RESTRICTED'
    for split in ('test','challenge'):
        parts=[]
        for kind in ('inputs','gold','metadata'):
            name=f'{split}_{kind}.jsonl'
            if split=='test':records=readlines(blobs[name])
            else:
                p=challenge/name;source_hashes[str(p.relative_to(ROOT))]=digest(p);records=readlines(p.read_text())
            parts.append({r['id']:r for r in records})
        assert set(parts[0])==set(parts[1])==set(parts[2])
        datasets['rq/'+split]=[{**parts[2][i],**parts[1][i],**parts[0][i]} for i in parts[0]]
    seen=set();development=[]
    canonical=lambda r:' '.join((r['context']+'\n'+r['request']).casefold().split())
    for source in ('brq','rq'):
        for split in ('train','validation','calibration'):
            for r in readlines((ROOT/f'data/processed/supplied-v1/{source}/{split}.jsonl').read_text()):
                seen.add(canonical(r));development.append(r)
    vectorizer=TfidfVectorizer(analyzer='char',ngram_range=(3,5),max_features=30000)
    development_vectors=vectorizer.fit_transform([r['context']+'\n'+r['request'] for r in development])
    nearest=NearestNeighbors(n_neighbors=1,metric='cosine',algorithm='brute',n_jobs=2).fit(development_vectors)
    report={'evaluation_role':'single_frozen_held_out_evaluation','human_ground_truth':False,
        'runtime':'CPU float32, same calibration and tokenizer as Prompt Lens','source_hashes':source_hashes,
        'frozen_manifest_sha256':digest(run/'frozen-candidate.json'),'results':{},
        'limitations':['Machine-generated labels, not independent human evaluation.',
            'Exact full-input leakage checked; semantic near-duplicate independence unverified.',
            'Challenge input limits are judged against actual 12,000-character / 256-token app contract.']}
    start=time.monotonic()
    for key,raw in datasets.items():
        source,split=key.split('/');rows=[];enc=[];rejected=[];collisions=[]
        for original in raw:
            try:
                # Mixed-language challenge examples can be scored by the actual app.
                language=original.get('language')
                row=normalize({**original,'split':split,'language':'en' if language=='mixed' else language},source,split)
                row['language']=language
                validate_input(row)
                encoded=tok(row['context'],row['request'],truncation=False)
                if len(encoded['input_ids'])>256:raise InputError('too_many_tokens')
            except (ValueError,KeyError,TypeError) as exc:
                rejected.append({'id':original.get('id'),'reason':exc.code if isinstance(exc,InputError) else str(exc)})
                continue
            if canonical(row) in seen:collisions.append(row['id'])
            rows.append(row);enc.append(encoded)
        prediction_path=run/(key.replace('/','-')+'-predictions.jsonl')
        cached={}
        if repair and prediction_path.exists():
            cached={r['id']:r['probabilities'] for r in readlines(prediction_path.read_text())}
        missing=[i for i,r in enumerate(rows) if r['id'] not in cached]
        with torch.inference_mode():
            for offset in range(0,len(missing),16):
                indices=missing[offset:offset+16]
                batch=tok.pad([enc[i] for i in indices],padding=True,return_tensors='pt')
                probabilities=expit(model(**batch).logits.numpy()/temperatures)
                for i,value in zip(indices,probabilities):cached[rows[i]['id']]=value.tolist()
        p=np.array([cached[r['id']] for r in rows])
        result={'total_rows':len(raw),'supported_rows':len(rows),'rejected':rejected,
            'exact_development_collisions':collisions,'metrics':slices(rows,p,thresholds)}
        distance,_=nearest.kneighbors(vectorizer.transform([r['context']+'\n'+r['request'] for r in rows]))
        similarity=1-distance[:,0]
        result['near_development_similarity_at_least_0.9']=int((similarity>=.9).sum())
        distinct=similarity<.9
        if distinct.any():result['similarity_below_0.9_metrics']=slices([r for r,k in zip(rows,distinct) if k],p[distinct],thresholds)
        clean=np.array([r['id'] not in set(collisions) for r in rows])
        if collisions and clean.any():result['no_exact_collision_metrics']=slices([r for r,k in zip(rows,clean) if k],p[clean],thresholds)
        report['results'][key]=result
        (run/(key.replace('/','-')+'-predictions.jsonl')).write_text(''.join(json.dumps({'id':r['id'],'probabilities':v.tolist()})+'\n' for r,v in zip(rows,p)))
        print(json.dumps({'dataset':key,'supported':len(rows),'rejected':len(rejected),'metrics':result['metrics']['overall']}),flush=True)
    report['seconds']=time.monotonic()-start
    if repair:report['loader_repair']='Included mixed-language challenge inputs supported by the app; all previously scored examples reused cached predictions. Model and settings unchanged.'
    assert all(digest(run/'model'/name)==value for name,value in frozen['model_hashes'].items())
    output.write_text(json.dumps(report,indent=2,ensure_ascii=False)+'\n')
if __name__=='__main__':main()
