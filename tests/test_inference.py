"""Read-only inference tests; no optimizer, training call or checkpoint writes."""
import json
import sys
from pathlib import Path
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from risk_model.inference import InputError, QualityPredictor, validate_input


@pytest.mark.parametrize('payload,code', [({},'invalid_input'),({'request':''},'empty_request'),
    ({'request':'hello','context':3},'invalid_input'),
    ({'request':'x'*12001},'too_many_characters')])
def test_input_boundary(payload, code):
    with pytest.raises(InputError) as exc:
        validate_input(payload)
    assert exc.value.code == code


@pytest.fixture(scope='module')
def predictor():
    return QualityPredictor()


def test_read_only_english_and_chinese_outputs(predictor):
    for request in ['Summarize this report in three bullet points.', '请把这份报告总结为三个要点。']:
        result = predictor.predict({'request':request,'context':''})
        assert len(result['signals']) == 5
        assert all(0 <= s['value'] <= 1 for s in result['signals'])
        assert result['quality_score'] is None and result['experimental'] is True
    assert not predictor.model.training
    assert not any(p.requires_grad for p in predictor.model.parameters())


def test_input_is_not_silently_truncated(predictor):
    with pytest.raises(InputError) as exc:
        predictor.predict({'request':'Hello world. '*400})
    assert exc.value.code == 'too_many_tokens'
    assert exc.value.details['tokens'] > 256


def test_gui_inference_matches_saved_model(predictor):
    import torch
    payload={'request':'比较这两份文档，把它缩短。','context':'有甲、乙两份文档。'}
    actual=predictor.predict(payload)
    batch=predictor.tokenizer(payload['context'],payload['request'],truncation=False,return_tensors='pt')
    with torch.inference_mode(): expected=(predictor.model(**batch).logits[0] / predictor.temperature).sigmoid().tolist()
    assert [s['value'] for s in actual['signals']] == pytest.approx(expected,abs=1e-7)
