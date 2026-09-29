"""Masked five-label training and evaluation; unknowns are never negatives."""
import numpy as np
import torch
from sklearn.metrics import average_precision_score, precision_recall_fscore_support
from .quality import LABELS


def targets(rows):
    known=np.array([[v is not None for v in r['labels']] for r in rows],dtype=bool)
    y=np.array([[0 if v is None else v for v in r['labels']] for r in rows],dtype=np.float32)
    return y,known


def masked_loss_sum(logits, labels, mask):
    return (torch.nn.functional.binary_cross_entropy_with_logits(logits,labels,reduction='none')*mask).sum()


def metrics(rows, probabilities, thresholds=None):
    y,known=targets(rows); p=np.clip(np.asarray(probabilities),1e-7,1-1e-7)
    threshold=np.array(thresholds if thresholds is not None else [.5]*5)
    result={}; losses=[];aps=[];f1s=[]
    for j,label in enumerate(LABELS):
        yy=y[known[:,j],j];pp=p[known[:,j],j]
        if not len(yy):result[label]={'eligible':0};continue
        loss=-(yy*np.log(pp)+(1-yy)*np.log(1-pp))
        precision,recall,f1,_=precision_recall_fscore_support(yy,pp>=threshold[j],average='binary',zero_division=0)
        ap=float(average_precision_score(yy,pp)) if yy.sum() else None
        result[label]={'eligible':len(yy),'positives':int(yy.sum()),'unknown':int((~known[:,j]).sum()),
            'log_loss':float(loss.mean()),'brier':float(((pp-yy)**2).mean()),'average_precision':ap,
            'precision':float(precision),'recall':float(recall),'f1':float(f1),'threshold':float(threshold[j])}
        losses.extend(loss.tolist());f1s.append(float(f1))
        if ap is not None:aps.append(ap)
    clear=known.all(axis=1)&(y.sum(axis=1)==0)
    flags=(p>=threshold).any(axis=1)
    return {'rows':len(rows),'mean_log_loss':float(np.mean(losses)) if losses else None,
            'macro_average_precision':float(np.mean(aps)) if aps else None,
            'macro_f1':float(np.mean(f1s)) if f1s else None,'all_clear_count':int(clear.sum()),
            'all_clear_false_flag_rate':float(flags[clear].mean()) if clear.any() else None,
            'labels':result}


def slices(rows,p,thresholds=None):
    out={'overall':metrics(rows,p,thresholds)}
    for field in ('source','language'):
        out[field]={}
        for value in sorted({r[field] for r in rows}):
            mask=np.array([r[field]==value for r in rows])
            out[field][value]=metrics([r for r in rows if r[field]==value],p[mask],thresholds)
    out['source_language']={}
    for source,lang in sorted({(r['source'],r['language']) for r in rows}):
        mask=np.array([r['source']==source and r['language']==lang for r in rows])
        out['source_language'][source+'/'+lang]=metrics([r for r in rows if r['source']==source and r['language']==lang],p[mask],thresholds)
    return out
