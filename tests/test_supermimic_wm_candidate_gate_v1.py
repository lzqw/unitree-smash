import pytest

from quadsmash.rl.supermimic_wm_candidate_gate_v1 import compare_candidate_to_anchor


def record(contact, pre=1.0, post=.1):
    return {'event_aligned': {
        '-1': {'1': {'ball_velocity_rmse': pre}},
        '0': {'1': {'ball_velocity_rmse': contact,
                    'ball_forward_sign_error': .1},
               '4': {'ball_velocity_rmse': .5}},
        '1': {'1': {'ball_velocity_rmse': post}},
    }}


def test_candidate_must_improve_contact_and_preserve_pre_post():
    decision = compare_candidate_to_anchor(record(1.0), record(.8, pre=1.0, post=.1))
    assert decision['passed']
    assert decision['contact_edge_improved']


def test_contact_degradation_rejects_even_when_other_metric_improves():
    decision = compare_candidate_to_anchor(record(1.0), record(1.3, pre=.8, post=.1))
    assert not decision['passed']
    assert any(row.get('status') == 'degraded' for row in decision['failures'])


def test_missing_protected_metric_is_not_treated_as_zero():
    candidate = record(.8)
    del candidate['event_aligned']['1']['1']['ball_velocity_rmse']
    decision = compare_candidate_to_anchor(record(1.0), candidate)
    assert not decision['passed']
    assert any(row.get('status') == 'missing' for row in decision['failures'])


def test_contact_gain_with_adjacent_impulse_is_rejected():
    decision = compare_candidate_to_anchor(record(1.0, pre=1.0, post=.1),
                                           record(.8, pre=1.0, post=.100001))
    assert not decision['passed']
    assert any(row['metric'] == 'event_aligned/1/1/ball_velocity_rmse'
               for row in decision['failures'])
