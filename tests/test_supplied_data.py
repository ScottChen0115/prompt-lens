import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import pytest
import torch
from risk_model.supplied_data import normalize,render_context,BRQ_LABELS
from risk_model.learning import masked_loss_sum,targets


def test_unknown_remains_unknown_and_label_names_are_mapped():
    row={'id':'1','split':'train','language':'en','context':[],'request':'Do it',
         'labels':{name:{'value':'unknown' if i==1 else str(i%2)} for i,name in enumerate(BRQ_LABELS)}}
    r=normalize(row,'brq','train')
    assert r['labels']==[0,None,0,1,0]


def test_hidden_attachment_content_not_leaked():
    s=render_context([{'type':'attachment','available':False,'name':'x','text':'SECRET'}])
    assert 'SECRET' not in s and 'content not available' in s


def test_reasons_and_metadata_are_never_features():
    row={'id':'1','split':'train','language':'en','context':'visible','request':'hello',
         'labels':{name:{'value':'0','reason':'SECRET'} for name in BRQ_LABELS},'meta':{'recipe':'SECRET'}}
    r=normalize(row,'brq','train')
    assert r['context']=='visible' and r['request']=='hello' and 'meta' not in r


def test_unknown_cell_has_zero_loss_and_gradient():
    z=torch.tensor([[1.,2.]],requires_grad=True)
    loss=masked_loss_sum(z,torch.tensor([[1.,0.]]),torch.tensor([[1.,0.]]))
    loss.backward()
    assert z.grad[0,1]==0 and z.grad[0,0]!=0
    expected=torch.nn.functional.binary_cross_entropy_with_logits(z[0,0],torch.tensor(1.))
    assert loss.item()==pytest.approx(expected.item())


def test_masked_batch_normalization_matches_full_batch():
    z=torch.tensor([[1.,2.],[3.,4.]],requires_grad=True)
    target=torch.tensor([[1.,0.],[0.,1.]])
    mask=torch.tensor([[1.,0.],[1.,1.]])
    full=masked_loss_sum(z,target,mask)/mask.sum()
    micro=sum(masked_loss_sum(z[i:i+1],target[i:i+1],mask[i:i+1])/mask.sum() for i in range(2))
    assert micro.item()==pytest.approx(full.item())
