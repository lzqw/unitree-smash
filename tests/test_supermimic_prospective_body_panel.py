import json

import pytest

from scripts.evaluate_supermimic_fresh_wm_panel import check_panel, digest
from scripts.rescore_supermimic_fresh_wm_panel import body_forward_check


def test_body_comparison_rejects_changed_weights_and_reused_launch(tmp_path):
    def write(name, data):
        path=tmp_path/name
        path.write_text(json.dumps(data))
        return path
    def case(name,x):
        return dict(case_id=name,launch=dict(position_m=[x,0,0],velocity_mps=[1,0,0]))
    suite=write('suite.json',dict(cases=[case('reserved',1)]))
    protocol=write('protocol.json',dict(training_suite=dict(path=suite.name,sha256=digest(suite)),formal_suites={}))
    body=write('body',1)
    closure=write('impulse',2)
    previous=write('earlier.json',dict(cases=[case('earlier',2)]))
    panel=dict(protocol_sha256=digest(protocol),policy_training=False,cases=[case('new',3)],weights={},
               prospective_body_comparison=dict(bodies={str(body):digest(body)},calibration_path=str(closure),
               calibration_sha256=digest(closure),previous_panels={str(previous):digest(previous)}))
    check_panel(panel,protocol,tmp_path,[])
    panel['cases']=[case('renamed',2)]
    with pytest.raises(ValueError,match='Overlapping'):
        check_panel(panel,protocol,tmp_path,[])
    panel['cases']=[case('new',3)]
    body.write_text('updated')
    with pytest.raises(ValueError,match='checkpoint changed'):
        check_panel(panel,protocol,tmp_path,[])


def test_body_gate_checks_each_horizon_and_requires_measured_support():
    def metrics(value, count):
        return dict(uniform=dict(body_rmse=value,windows=count))
    controls=dict(anchor={1:metrics(1,2),64:metrics(1,2)},candidate={1:metrics(.9,2),64:metrics(1,2)})
    assert body_forward_check(controls,[1,64],['body_rmse'])['passed']
    for value, count in ((1.01,2),(.2,0),(.2,1),(float('nan'),2)):
        controls['candidate'][64]=metrics(value,count)
        assert not body_forward_check(controls,[1,64],['body_rmse'])['passed']


def test_ball_calibration_is_specific_to_weights_and_training_split():
    from quadsmash.rl.supermimic_ball_residual_v2 import verify_event_calibration
    parent = dict(packed_sha256='archive-a',train_lanes=[0,3],excluded_lanes=[1,2])
    row = dict(parent,source_ball_sha256='ball-a',fitted_body_sha256='body-b')
    verify_event_calibration({'event_calibration':row},'ball-a','body-b',{'calibration':parent})
    for update in (dict(train_lanes=[0,1]),dict(fitted_body_sha256='body-c'),dict(source_ball_sha256='ball-b')):
        with pytest.raises(ValueError):
            verify_event_calibration({'event_calibration':dict(row,**update)},'ball-a','body-b',{'calibration':parent})
