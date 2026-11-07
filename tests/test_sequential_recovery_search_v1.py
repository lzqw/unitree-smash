"""Protect physical rejection while permitting internal search trajectories."""
import copy
import json
from pathlib import Path
import numpy as np
from test_differentiated_sequence_v1 import setup_profile
from scripts.landing_sequential_recovery_search_v1 import rollout_for_search, restore_feasibility
from scripts.landing_joint_sequence_proposal_v1 import verify_sequence, propose_sequence


def test_search_rollout_reproduces_original_nominal_without_mutating_history():
    controller,state,profile=setup_profile()
    saved=copy.deepcopy(state)
    previous=controller.qp.interface.previous_command.copy()
    commands=np.stack([n['command'] for n in profile])
    result=rollout_for_search(controller,state,commands,state.joint_position)
    assert result['native_execution'] is False
    assert result['physically_feasible']
    for original,rebuilt in zip(profile,result['profile']):
        np.testing.assert_allclose(original['following'].q,rebuilt['following'].q,atol=1e-10)
        np.testing.assert_allclose(original['following'].v,rebuilt['following'].v,atol=1e-10)
        np.testing.assert_array_equal(original['following'].contacts,rebuilt['following'].contacts)
        assert rebuilt['internal_search_only']
    np.testing.assert_array_equal(state.q,saved.q)
    np.testing.assert_array_equal(state.v,saved.v)
    np.testing.assert_array_equal(controller.qp.interface.previous_command,previous)


def test_infeasible_search_trajectory_cannot_pass_independent_verification():
    controller,state,profile=setup_profile()
    commands=np.stack([n['command'] for n in profile])
    commands[:,0]+=1.5
    searched=rollout_for_search(controller,state,commands,state.joint_position)
    assert searched['infeasibility']>0
    assert not searched['physically_feasible']
    assert any(x['physical']['motor_torque_nm']>0 for x in searched['checks'])
    checked=verify_sequence(controller,state,commands)
    assert not checked['accepted']
    assert checked['native_execution'] is False


def test_minimum_increment_restoration_preserves_a_feasible_nominal():
    controller,state,profile=setup_profile()
    nominal=np.stack([n['command'] for n in profile])
    result=propose_sequence(controller,state,state.joint_position,knots=2,
        nominal_profile=profile,differentiate_dynamics=True,
        solver_backend='active_set',feasibility_restoration=True)
    assert result['linear_proposal_solved']
    assert result['accepted'] is False  # A QP is still not execution authority.
    np.testing.assert_allclose(result['commands'],nominal,atol=1e-9)
    assert result['linear_constraint_violation']<1e-9
    assert result['constraint_rows']>1000
    assert verify_sequence(controller,state,result['commands'])['accepted']==verify_sequence(controller,state,nominal)['accepted']


def test_recorded_circular_cone_pass_does_not_bypass_polygon_constraint():
    from scripts.parallel_landing_solver_v1 import build_controller
    data=json.loads((Path(__file__).parent/'fixtures/landing_restoration_friction_boundary.json').read_text())
    cfg=data['configuration'];c=build_controller(cfg);limits=np.asarray(cfg['interface']['limits'])
    state=c.qp.model.state_from_public(np.asarray(data['public']),np.asarray(cfg['posture_target']))
    actions=np.asarray(data['actions'],dtype=np.float32)
    state.previous_command_target=limits[:,0]+.5*(actions[0]+1.)*np.diff(limits).ravel()
    reference=limits[:,0]+.5*(actions[1]+1.)*np.diff(limits).ravel()
    commands=np.asarray(data['commands'])
    assert verify_sequence(c,state,commands)['accepted']
    before=rollout_for_search(c,state,commands,reference)
    assert not before['physically_feasible']
    repaired=restore_feasibility(c,state,reference,commands,iterations=1)
    assert not repaired['accepted'] or repaired['physically_feasible']
    if repaired['accepted']:
        assert repaired['maximum_physical_violation']<=c.qp.settings.residual_tolerance
        assert verify_sequence(c,state,repaired['commands'])['accepted']
