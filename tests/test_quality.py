import copy
import sys
from pathlib import Path
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from risk_model.quality import fixtures, validate_development


def test_both_languages_and_all_labels_exercised():
    rows=validate_development(fixtures())
    assert len(rows)==144
    assert {r['language'] for r in rows}=={'en','zh'}


def test_translation_split_leakage_rejected_even_with_different_group():
    rows=fixtures(); rows[1]['split']='test'; rows[1]['group_id']='different'
    with pytest.raises(ValueError,match='Cross-split leakage'):
        validate_development(rows)


def test_research_labels_cannot_enter_fixture_trainer():
    rows=fixtures(); rows[0]['source']='reviewed_user_data'
    with pytest.raises(ValueError,match='fixtures only'):
        validate_development(rows)


def test_single_output_schema_not_accepted():
    rows=fixtures(); rows[0]['labels']=[1]
    with pytest.raises(ValueError,match='five binary'):
        validate_development(rows)


def test_missing_language_slice_rejected():
    rows=[r for r in fixtures() if not (r['language']=='zh' and r['split']=='validation')]
    with pytest.raises(ValueError,match='Each language/split'):
        validate_development(rows)


def test_transformers_set_metadata_serializes():
    import json
    sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
    from train_bilingual import json_default
    encoded=json.dumps({'missing_keys':{'classifier.bias','classifier.weight'}},default=json_default)
    assert json.loads(encoded)['missing_keys']==['classifier.bias','classifier.weight']
