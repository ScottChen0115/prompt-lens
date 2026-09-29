"""Development-only similarity diagnostic, not a semantic equivalence judgment."""
import json,sys
from pathlib import Path
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
ROOT=Path(__file__).resolve().parents[1]
data={}
for split in ('train','validation','calibration'):
 data[split]=[]
 for source in ('brq','rq'):
  data[split]+=[json.loads(l) for l in (ROOT/f'data/processed/supplied-v1/{source}/{split}.jsonl').read_text().splitlines()]
texts=lambda rows:[r['context']+'\n'+r['request'] for r in rows]
v=TfidfVectorizer(analyzer='char',ngram_range=(3,5),max_features=30000)
x=v.fit_transform(texts(data['train']))
n=NearestNeighbors(n_neighbors=1,metric='cosine',algorithm='brute',n_jobs=2).fit(x)
out={'method':'Nearest training neighbor by character 3–5 gram TF-IDF cosine similarity. Diagnostic only; not semantic equivalence. Shared templates can inflate test performance.','results':{}}
for split in ('validation','calibration'):
 distances,indices=n.kneighbors(v.transform(texts(data[split])))
 values=1-distances[:,0]
 out['results'][split]={'rows':len(values),'similarity_at_least_0.9':int((values>=.9).sum()),'similarity_at_least_0.95':int((values>=.95).sum()),'maximum':float(values.max()),'top_50':[{'id':data[split][i]['id'],'training_id':data['train'][int(indices[i,0])]['id'],'similarity':float(values[i])} for i in np.argsort(values)[-50:][::-1]]}
(ROOT/'outputs/dataset-audit/near-duplicate-diagnostic.json').write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps({k:{a:b for a,b in val.items() if a!='top_50'} for k,val in out['results'].items()}))
