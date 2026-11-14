"""CPU contract counterexamples; these are not native execution tests."""
from types import SimpleNamespace

import pytest
import torch

from quadsmash.rl.supermimic_imagination_contract_v1 import (
    ENVIRONMENT_FIELDS, TIMING_FIELDS, RECOVERY_FIELDS,
    capture_auxiliary_state, physical_start_issues, require_physical_start)
from quadsmash.tasks.stand.jump_metrics import FLOAT_FIELDS, BOOL_FIELDS
from quadsmash.tasks.stand.contact_flight_recovery_v5 import ContactFlightRecoveryV5


def test_same_observed_recovery_flag_does_not_determine_next_recovery():
    # Both lanes can have identical physical55, public794 and paid-event8,
    # while one has spent .799 s in the same stable window, the other 0 s.
    recovery = ContactFlightRecoveryV5(2, 'cpu',
        {'recovery': {'continuous_stable_window_s': .8}})
    recovery.bound[:] = True
    recovery.qualified_landing[:] = True
    recovery.flight_id[:] = 1
    recovery.contact_flight_id[:] = 1
    recovery.stable_run[0] = .799
    assert recovery.stable_seen.tolist() == [False, False]
    jump = SimpleNamespace(
        flight_candidate=torch.zeros(2, dtype=torch.bool), takeoff_time=torch.ones(2),
        flight_apex_rise=torch.full((2,), .1), flight_clear_time=torch.full((2,), .2),
        valid_jump_landed=torch.ones(2, dtype=torch.bool),
        flight_qualified=torch.ones(2, dtype=torch.bool), landing_time=torch.ones(2),
        body_ground_seen=torch.zeros(2, dtype=torch.bool))
    recovery.metric(jump, torch.ones(2, dtype=torch.bool), .001)
    assert recovery.stable_seen.tolist() == [True, False]


def test_old_archive_rejected_without_zero_filling_missing_task_state():
    execution = dict(object_types={'controller':
        'quadsmash.control.full_racket_rally_controller_v4.FullRacketRallyControllerV4'},
        omitted={'controller/logs': 'diagnostic logs'})
    state = {'recovery': {'rally_recovery_success': torch.tensor([False])}}
    issues = physical_start_issues(state, execution)
    assert any(i.get('group') == 'contact_recovery' and 'stable_run' in i['fields'] for i in issues)
    assert any(i['code'] == 'omitted_causal_plan_log' for i in issues)
    with pytest.raises(ValueError, match='Incomplete physical imagination start'):
        require_physical_start(state, execution)
    assert 'auxiliary' not in state


def test_auxiliary_capture_is_read_only_and_does_not_claim_solver_cache():
    # Shape-only fixture for capture ownership; not a replacement for Isaac.
    env = SimpleNamespace()
    for name in set(FLOAT_FIELDS + BOOL_FIELDS + ENVIRONMENT_FIELDS):
        setattr(env, name, torch.ones(2))
    for name in TIMING_FIELDS:
        setattr(env, name, 17)
    env.contact_recovery = SimpleNamespace(**{name: torch.ones(2) for name in RECOVERY_FIELDS})
    env.contact_events = SimpleNamespace(**{name: torch.ones(2) for name in ('active', 'count', 'clear_frames')})
    env.safety_body_ids = [0, 1]
    env._sm_residual_limits = torch.ones(18, 2)
    env.robot = SimpleNamespace(data=SimpleNamespace(
        body_pos_w=torch.ones(2, 3, 3), projected_gravity_b=torch.ones(2, 3),
        root_com_lin_vel_w=torch.ones(2, 3), root_com_ang_vel_w=torch.ones(2, 3),
        applied_torque=torch.ones(2, 18)))
    snapshot = capture_auxiliary_state(env)
    env.contact_recovery.stable_run.zero_()
    env.robot.data.applied_torque.zero_()
    assert snapshot['contact_recovery']['stable_run'].eq(1).all()
    assert snapshot['reward_inputs']['applied_torque_all'].eq(1).all()
    assert 'PhysX contact solver cache' in snapshot['not_captured']
    assert not physical_start_issues({'auxiliary': snapshot}, {'object_types': {}, 'omitted': {}})
    del env.contact_recovery.stable_run
    with pytest.raises(AttributeError):
        capture_auxiliary_state(env)

