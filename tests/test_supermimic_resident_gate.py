import hashlib
import json

import pytest

from quadsmash.rl.supermimic_native_event_coverage import coverage_deficits, coverage_targets, native_event_coverage
from quadsmash.rl.supermimic_resident_gate import await_native_decision, publish_boundary


def test_resident_boundary_is_immutable_and_bound_to_decision(tmp_path):
    latest = tmp_path / 'latest.pt'
    latest.write_bytes(b'original checkpoint')
    directory = tmp_path / 'gate'
    receipt = publish_boundary(directory, latest, 40, {'free_bytes': 1024})
    next_checkpoint = tmp_path / 'next.pt'
    next_checkpoint.write_bytes(b'replacement checkpoint')
    next_checkpoint.replace(latest)
    assert (directory / 'policy.pt').read_bytes() == b'original checkpoint'
    assert json.loads((directory / 'ready.json').read_text()) == receipt
    assert await_native_decision(directory, receipt, 0, lambda: None) == 'stop'

    paired = b'{"result": "training-panel-pair"}'
    (directory / 'paired.json').write_bytes(paired)
    decision = dict(update=40, checkpoint_sha256=receipt['checkpoint_sha256'],
                    action='continue', paired_result_sha256=hashlib.sha256(paired).hexdigest())
    (directory / 'decision.json').write_text(json.dumps(decision))
    assert await_native_decision(directory, receipt, 1, lambda: None) == 'continue'
    (directory / 'decision.json').write_text(json.dumps(dict(decision, update=48)))
    with pytest.raises(ValueError, match='does not match'):
        await_native_decision(directory, receipt, 1, lambda: None)


def test_warm_coverage_uses_independent_unperturbed_labelled_starts():
    base = dict(episode_index=0, arm_identification_episode=False, numerical_failures=0,
                qualified_airborne_string_contact=True, outgoing_speed_mps=4.6, rejection_reasons=[],
                ball_side_complete_except_recovery=True,
                native_contact=dict(frame_hit=False, string_hit=True, signed_string_margin_m=.008))
    records = [
        dict(base, env_id=0, case_id='near'),
        dict(base, env_id=1, case_id='low', outgoing_speed_mps=4.3, rejection_reasons=['speed_threshold'],
             ball_side_complete_except_recovery=False),
        dict(base, env_id=2, case_id='frame', qualified_airborne_string_contact=False,
             native_contact=dict(frame_hit=True, string_hit=False), ball_side_complete_except_recovery=False),
        dict(base, env_id=3, case_id='near', episode_index=1),
        dict(base, env_id=4, case_id='holdout'),
        dict(base, env_id=5, case_id='perturbed', arm_identification_episode=True),
        dict(base, env_id=6, case_id='numerical', numerical_failures=1),
    ]
    baseline = native_event_coverage(records, {4}, 180.)
    target = coverage_targets(baseline)
    assert target == dict(settled_cases=3, qualified_string=2, low_speed=1, frame=1,
                          near_edge_under_10_mm=2, marginal_success=1)
    partial = native_event_coverage(records, {4}, 90., indexed_episodes={(0, 0), (2, 0), (3, 1), (4, 0), (5, 0)})
    assert coverage_deficits(partial['counts'], target) == {
        'settled_cases': 1, 'qualified_string': 1, 'low_speed': 1, 'near_edge_under_10_mm': 1,
    }
    assert partial['per_collection_hour']['settled_cases'] == 80
    complete = native_event_coverage(records, {4}, 120., indexed_episodes={(i, 0) for i in range(7)})
    assert not coverage_deficits(complete['counts'], target)
