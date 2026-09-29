import copy
import sys
from pathlib import Path
import pytest
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src'))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
from risk_model.data import validate, features, encode
from train_development import fixtures

def test_fixture_cannot_become_gold():
    with pytest.raises(ValueError, match='Fixtures'):
        validate(fixtures())

def test_group_leakage_rejected():
    rows=fixtures(); rows[0]['split']='validation'
    with pytest.raises(ValueError, match='Group crosses'):
        validate(rows,development=True)

def test_duplicate_features_rejected():
    rows=fixtures(); rows[-1].update(context=rows[0]['context'],request=rows[0]['request'])
    with pytest.raises(ValueError, match='Duplicate features'):
        validate(rows,development=True)

def test_label_only_fields_never_enter_features():
    row=fixtures()[0]; before=features(row)
    row.update(hidden_intent='SECRET',observed_interpretation='SECRET',reviews=['SECRET'],label=99)
    assert features(row)==before

def test_no_forced_unjudgeable_negative():
    rows=fixtures()
    for r in rows:
        r.update(source='authored',license='project',assistant_revision='frozen',prompt_sha256='abc',
            decoding_revision='greedy',intent_recorded_at='2026-09-27',trial_recorded_at='2026-09-28',
            hidden_intent={'action':'move'},observed_interpretation={'action':'move'},outcome='unjudgeable',
            reviews=[{'reviewer_id':x,'human':True,'outcome':'unjudgeable'} for x in ['a','b']])
    with pytest.raises(ValueError,match='Unjudgeable'):
        validate(rows)

def test_valid_fixture_splits():
    assert len(validate(fixtures(),development=True))==512

def test_long_input_fails_closed():
    class FakeTokenizer:
        def __call__(self, *args, **kwargs):
            return {'input_ids': [[0]*257]}
    with pytest.raises(ValueError,match='Unsupported long input'):
        encode(FakeTokenizer(), fixtures()[:1])

def test_single_reviewer_rejected():
    rows=fixtures()
    for r in rows:
        r.update(source='authored',license='project',assistant_revision='frozen',prompt_sha256='abc',
            decoding_revision='greedy',intent_recorded_at='2026-09-27',trial_recorded_at='2026-09-28',
            hidden_intent={'action':'move'},observed_interpretation={'action':'move'},
            reviews=[{'reviewer_id':'a','human':True},{'reviewer_id':'a','human':True}])
    with pytest.raises(ValueError,match='Two independent human reviews'):
        validate(rows)
