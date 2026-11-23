import copy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from quadsmash.tasks.stand import jump_metrics
from quadsmash.tasks.stand.contact_flight_recovery_v5 import ContactFlightRecoveryV5, independent_recovery
from quadsmash.tasks.stand.contact_recovery_origin0_v5 import PROTOCOL as REVISED, validate_recovery_protocol


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = json.loads((ROOT / 'configs/evaluation_protocol_v2.json').read_text())


def state():
    jump = SimpleNamespace()
    jump_metrics.initialize(jump, 2, 'cpu')
    return jump, ContactFlightRecoveryV5(2, 'cpu', PROTOCOL)


def flight(jump, tracker, start, qualified=True, bind=True):
    jump.takeoff_time[:] = start
    jump.flight_candidate[:] = True
    jump.flight_qualified[:] = qualified
    jump.valid_jump_landed[:] = False
    jump.flight_apex_rise[:] = .12 if qualified else .01
    jump.flight_clear_time[:] = .2 if qualified else .001
    tracker.metric(jump, torch.zeros(2, dtype=torch.bool), .001)
    if bind:
        tracker.contact(jump, legal_air_strings_touching=torch.ones(2, dtype=torch.bool),
                        time=torch.full((2,), start + .2), rally_contact_time=torch.full((2,), start + .2))


def land(jump, tracker, time, qualified=True):
    jump.flight_candidate[:] = False
    jump.valid_jump_landed[:] = qualified
    jump.flight_qualified[:] = qualified
    jump.flight_clear_time[:] = 0
    jump.landing_time[:] = time
    tracker.metric(jump, torch.zeros(2, dtype=torch.bool), .001)


def settle(jump, tracker, seconds):
    tracker.metric(jump, torch.ones(2, dtype=torch.bool), seconds)


def test_later_hop_preserves_contact_landing_but_restarts_stable_window():
    jump, tracker = state()
    flight(jump, tracker, 2.)
    land(jump, tracker, 2.5)
    settle(jump, tracker, .5)
    flight(jump, tracker, 2.6, qualified=False, bind=False)
    land(jump, tracker, 2.601, qualified=False)
    settle(jump, tracker, .799)
    assert not tracker.success(jump).any()
    settle(jump, tracker, .001)
    assert tracker.success(jump).all()
    assert not jump.valid_jump_landed.any()
    jump.body_ground_seen[1] = True
    assert tracker.success(jump).tolist() == [True, False]
    tracker.reset(torch.tensor([0]))
    assert tracker.bound.tolist() == [False, True]
    assert not tracker.success(jump).any()


def test_relaxed_window_reports_resets_and_cannot_complete_after_deadline():
    from quadsmash.rl.supermimic_terminal_scoring import relaxed_scoring_protocol
    from quadsmash.tasks.stand.contact_flight_recovery_v6 import ContactFlightRecoveryV6

    jump, _ = state()
    jump.jump_clock = torch.zeros(2)
    tracker = ContactFlightRecoveryV6(2, 'cpu', PROTOCOL, relaxed_scoring_protocol())
    flight(jump, tracker, 2.)
    land(jump, tracker, 2.5)
    failures = torch.zeros((2, 6), dtype=torch.bool)
    for t in (2.6, 2.7, 2.8):
        jump.jump_clock[:] = t
        tracker.relaxed_metric(jump, failures, .1)
    failures[1, 0] = True
    jump.jump_clock[:] = 2.9
    tracker.relaxed_metric(jump, failures, .1)
    assert tracker.relaxed_success.tolist() == [True, False]
    assert tracker.relaxed_record(1)['first_reset_reasons'] == ['support']
    failures.zero_()
    for t in (4.3, 4.4, 4.5, 4.6):
        jump.jump_clock[:] = t
        tracker.relaxed_metric(jump, failures, .1)
    assert tracker.relaxed_success.tolist() == [True, False]
    assert tracker.relaxed_record(1)['reset_counts']['deadline'] == 1


def test_earlier_qualified_flight_does_not_qualify_later_contact_hop():
    jump, tracker = state()
    flight(jump, tracker, 1., bind=False)
    land(jump, tracker, 1.5)
    flight(jump, tracker, 2., qualified=False)
    land(jump, tracker, 2.201, qualified=False)
    settle(jump, tracker, 2.)
    assert tracker.bound.all()
    assert not tracker.qualified_landing.any()
    assert not tracker.success(jump).any()


def test_later_unrelated_qualified_flight_cannot_repair_contact_hop():
    jump, tracker = state()
    flight(jump, tracker, 1., qualified=False)
    land(jump, tracker, 1.201, qualified=False)
    flight(jump, tracker, 2., qualified=True, bind=False)
    land(jump, tracker, 2.5)
    settle(jump, tracker, 2.)
    assert not tracker.qualified_landing.any()
    assert not tracker.success(jump).any()


def test_grounded_or_illegal_strings_cannot_bind():
    jump, tracker = state()
    tracker.contact(jump, legal_air_strings_touching=torch.ones(2, dtype=torch.bool),
                    time=torch.ones(2), rally_contact_time=torch.ones(2))
    assert not tracker.bound.any()
    flight(jump, tracker, 1., bind=False)
    tracker.contact(jump, legal_air_strings_touching=torch.zeros(2, dtype=torch.bool),
                    time=torch.ones(2), rally_contact_time=torch.ones(2))
    land(jump, tracker, 1.5)
    settle(jump, tracker, 2.)
    assert not tracker.success(jump).any()


def test_independent_check_rejects_time_threshold_and_body_mismatch():
    jump, tracker = state()
    flight(jump, tracker, 2.)
    land(jump, tracker, 2.5)
    settle(jump, tracker, .8)
    record = tracker.record(jump, 0, .323)
    row = dict(contact_time_s=record['rally_contact_time_s'],
               jump=dict(body_landing=False, terminated=False, maximum_body_ground_force_n=0.))
    assert independent_recovery(record, row, PROTOCOL, .0000625)
    for name, value in [('rally_contact_time_s', 1.), ('contact_flight_clear_s', .099),
                        ('recovery_s', .799), ('body_landing', True),
                        ('terminal_root_height_m', .1)]:
        changed = copy.deepcopy(record)
        changed[name] = value
        assert not independent_recovery(changed, row, PROTOCOL, .0000625)
    with pytest.raises(ValueError):
        independent_recovery({}, row, PROTOCOL, .0000625)


def test_protocol_revision_cannot_change_physics_cases_target_or_thresholds():
    revised = json.loads((ROOT / REVISED).read_text())
    original = json.loads((ROOT / 'configs/g07_full_racket_eval_v4_r3_origin0.json').read_text())
    assert validate_recovery_protocol(revised) == original
    for key, value in [('episode_seconds', 9.), ('physics_dt_s', .001),
                       ('thresholds', dict(complete_success_rate=.1)), ('training_suite', {})]:
        changed = copy.deepcopy(revised)
        changed[key] = value
        with pytest.raises(ValueError):
            validate_recovery_protocol(changed)
