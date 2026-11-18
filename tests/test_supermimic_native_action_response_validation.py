import json
from pathlib import Path

import pytest

from scripts.validate_supermimic_native_action_response import (
    _load_bounds, _response_errors, build_receipt, validate_native_sample_ids,
)


def test_bounds_require_predeclared_disjoint_episodes(tmp_path):
    path=tmp_path/'bounds.json'
    path.write_text(json.dumps(dict(schema='contact-validation-bounds-v1',
        thresholds_declared_before_validation=True,
        limits=dict(native_repeat_max=1e-6,native_response_min=.001,
                    response_rmse=.2,response_cosine_error_max=.2,
                    response_relative_error_max=.5),
        event_limits={'contact_rmse': .2},
        minimum_paired_samples=1, minimum_independent_validation_episodes=1,
        training_episode_ids=['train'],validation_episode_ids=['valid'])))
    assert _load_bounds(path)['validation_episode_ids']==['valid']
    path.write_text(json.dumps(dict(schema='contact-validation-bounds-v1',
        thresholds_declared_before_validation=True,
        limits={'response_rmse':.2},event_limits={'contact_rmse':.2},
        minimum_paired_samples=1,minimum_independent_validation_episodes=1,
        training_episode_ids=['same'],validation_episode_ids=['same'])))
    with pytest.raises(ValueError,match='disjoint'):_load_bounds(path)


def test_response_errors_keep_direction_and_repeatability_metrics():
    result=_response_errors(dict(native_repeat_max=.0001,response_rmse=.04,
        response_cosine=[.9,.8],relative_error=[.1,.2],count=2))
    assert result['response_cosine_error_max']==pytest.approx(.2)
    assert result['response_relative_error_max']==pytest.approx(.2)


def test_receipt_contains_hashed_evidence_and_declared_bounds(tmp_path):
    probe=tmp_path/'probe.pt';probe.write_bytes(b'probe')
    events=tmp_path/'events.json';events.write_text(json.dumps({'event_aligned_errors':{'contact_rmse':.1}}))
    models={}
    for name in ('physical','ball','task'):
        path=tmp_path/(name+'.pt');path.write_bytes(name.encode());models[name]=path
    bounds=dict(schema='contact-validation-bounds-v1',thresholds_declared_before_validation=True,
        limits=dict(native_repeat_max=.01,native_response_min=.001,response_rmse=.2,
                    response_cosine_error_max=.3,response_relative_error_max=.5),
        event_limits={'contact_rmse':.2},training_episode_ids=['train'],validation_episode_ids=['valid'])
    measured=dict(native_repeat_max=.001,response_rmse=.05,response_cosine=[.9],
                  relative_error=[.2],count=1,sample_ids=['valid'],passed=True,failures=[])
    bounds['minimum_paired_samples']=1
    bounds['minimum_independent_validation_episodes']=1
    receipt=build_receipt(bounds=bounds,event_errors={'contact_rmse':.1},
        response_errors=_response_errors(measured),models=models,
        contract={'asset':{'sha256':'asset'}},probe_path=probe,event_path=events,
        measured=measured,identity={'schema':'runtime'})
    assert receipt['schema']=='supermimic-contact-validation-v1'
    assert receipt['native_action_probe']['same_prefix_verified']
    assert set(receipt['upper_bounds']['action_response_errors'])=={
        'native_repeat_max','response_rmse','response_cosine_error_max','response_relative_error_max'}


def test_receipt_rejects_response_outside_declared_bound(tmp_path):
    probe=tmp_path/'probe.pt';probe.write_bytes(b'probe')
    events=tmp_path/'events.json';events.write_text('{}')
    model=tmp_path/'model.pt';model.write_bytes(b'model')
    bounds=dict(schema='contact-validation-bounds-v1',thresholds_declared_before_validation=True,
        limits=dict(native_repeat_max=.01,native_response_min=.001,response_rmse=.02,
                    response_cosine_error_max=.3,response_relative_error_max=.5),
        event_limits={'contact_rmse':.2},minimum_paired_samples=1,
        minimum_independent_validation_episodes=1,training_episode_ids=['train'],validation_episode_ids=['valid'])
    measured=dict(native_repeat_max=.001,response_rmse=.05,response_cosine=[.9],
                  relative_error=[.2],count=1,sample_ids=['valid'],passed=True,failures=[])
    with pytest.raises(ValueError,match='exceeds bound'):
        build_receipt(bounds=bounds,event_errors={'contact_rmse':.1},
            response_errors=_response_errors(measured),models={'physical':model},
            contract={'asset':{'sha256':'asset'}},probe_path=probe,event_path=events,
            measured=measured,identity={'schema':'runtime'})


def test_native_samples_must_belong_to_declared_validation_set(tmp_path):
    probe=tmp_path/'probe.pt'; probe.write_bytes(b'probe')
    events=tmp_path/'events.json'; events.write_text(json.dumps({'event_aligned_errors': {'contact_rmse': .1}}))
    model=tmp_path/'model.pt'; model.write_bytes(b'model')
    bounds=dict(schema='contact-validation-bounds-v1', thresholds_declared_before_validation=True,
        limits=dict(native_repeat_max=.01,native_response_min=.001,response_rmse=1.,
                    response_cosine_error_max=1.,response_relative_error_max=2.),
        event_limits={'contact_rmse':1.}, minimum_paired_samples=1,
        minimum_independent_validation_episodes=1, training_episode_ids=['train'],
        validation_episode_ids=['valid'])
    measured=dict(native_repeat_max=.001,response_rmse=.05,response_cosine=[.9],
                  relative_error=[.2],count=1,sample_ids=['other'],passed=True,failures=[])
    with pytest.raises(ValueError, match='validation episode set'):
        validate_native_sample_ids(measured['sample_ids'], bounds['validation_episode_ids'])
