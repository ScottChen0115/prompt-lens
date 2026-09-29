"""Audit only public development splits; retain originals and keep locked tests sealed."""
from pathlib import Path
import sys,json,hashlib,re,unicodedata
from collections import Counter,defaultdict
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from risk_model.supplied_data import normalize
from risk_model.quality import LABELS
from transformers import AutoTokenizer


def canonical(text):
    return re.sub(r'\s+',' ',unicodedata.normalize('NFKC',text)).strip().casefold()


def key(r):
    return hashlib.sha256(json.dumps([canonical(r['context']),canonical(r['request'])],ensure_ascii=False).encode()).hexdigest()


def main():
    checkpoint=json.loads((ROOT/'configs/checkpoint-bilingual.json').read_text())
    tok=AutoTokenizer.from_pretrained(checkpoint['local_path'],local_files_only=True,trust_remote_code=False)
    sources={'brq':ROOT/'test set/brq-v1','rq':ROOT/'test set/request_quality_v0.1_dev'}
    all_rows=[]; report={'label_order':list(LABELS),'tokenizer_revision':checkpoint['revision'],
        'max_tokens':256,'review_status':'both sources machine-labeled; no human gold established',
        'locked_tests_inspected_or_scored':False,'sources':{}}
    for source,base in sources.items():
        manifest=json.loads((base/'manifest.json').read_text())
        entries={item['path']:item for item in manifest['files']}
        source_report={};ids=set();relations=defaultdict(set)
        for split in ('train','validation','calibration'):
            rel=('data/' if source=='brq' else '')+split+'.jsonl'
            path=base/rel
            sha=hashlib.sha256(path.read_bytes()).hexdigest()
            if sha != entries[rel]['sha256']:raise ValueError(f'Source checksum mismatch: {path}')
            rows=[normalize(json.loads(line),source,split) for line in path.read_text().splitlines()]
            for row in rows:
                if row['id'] in ids:raise ValueError('Duplicate ID')
                ids.add(row['id'])
                for field,value in row['relations'].items():relations[(field,value)].add(split)
            encoded=tok([r['context'] for r in rows],[r['request'] for r in rows],truncation=False,verbose=False)
            counts=[];accepted=[];quarantined=[]
            for row,tokens in zip(rows,encoded['input_ids']):
                row['token_count']=len(tokens);counts.append(len(tokens));row['feature_sha256']=key(row)
                if len(tokens)>256:
                    quarantined.append({'id':row['id'],'reason':'over_256_tokens','tokens':len(tokens)})
                else:accepted.append(row)
            dest=ROOT/'data/processed/supplied-v1'/source
            dest.mkdir(parents=True,exist_ok=True)
            (dest/(split+'.jsonl')).write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in accepted))
            (dest/(split+'-quarantine.json')).write_text(json.dumps(quarantined,indent=2)+'\n')
            source_report[split]={'source_file':str(path),'sha256':sha,'checksum_verified':True,
                'rows':len(rows),'accepted':len(accepted),'overlength':len(quarantined),
                'language':dict(Counter(r['language'] for r in rows)),
                'accepted_language':dict(Counter(r['language'] for r in accepted)),
                'positive_labels':{label:sum(r['labels'][i]==1 for r in rows) for i,label in enumerate(LABELS)},
                'unknown_labels':{label:sum(r['labels'][i] is None for r in rows) for i,label in enumerate(LABELS)},
                'tokens':{'min':min(counts),'max':max(counts),'mean':sum(counts)/len(counts)}}
            all_rows.extend(rows)
        source_report['relation_split_violations']=[{'field':field,'value':value,'splits':sorted(splits)}
            for (field,value),splits in relations.items() if len(splits)>1]
        report['sources'][source]=source_report
    by_features=defaultdict(list)
    for row in all_rows:by_features[row['feature_sha256']].append(row)
    duplicates=[]
    for sha,group in by_features.items():
        if len(group)<2:continue
        cross=len({r['split'] for r in group})>1
        conflicts=[LABELS[i] for i in range(5) if len({r['labels'][i] for r in group if r['labels'][i] is not None})>1]
        if cross or conflicts:
            duplicates.append({'feature_sha256':sha,'rows':[{'id':r['id'],'split':r['split']} for r in group],
                               'cross_split':cross,'conflicting_labels':conflicts})
    report['exact_feature_conflicts_or_leaks']=duplicates
    report['summary']={'development_rows':len(all_rows),
        'supported_rows':sum(r['token_count']<=256 for r in all_rows),
        'cross_split_exact_groups':sum(d['cross_split'] for d in duplicates),
        'contradictory_label_groups':sum(bool(d['conflicting_labels']) for d in duplicates)}
    out=ROOT/'outputs/dataset-audit';out.mkdir(parents=True,exist_ok=True)
    (out/'audit.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k not in ['exact_feature_conflicts_or_leaks','sources']},indent=2))
    for source,data in report['sources'].items():
        print(source,json.dumps(data,ensure_ascii=False))

if __name__=='__main__':main()
