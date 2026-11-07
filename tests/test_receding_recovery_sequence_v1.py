"""Replanning must use new observations and never return failed search iterates."""
from types import SimpleNamespace
import copy
import numpy as np
import scripts.receding_recovery_sequence_v1 as module
import scripts.landing_sequential_recovery_search_v1 as search


def fixture():
    interface=SimpleNamespace(limits=np.tile([-2.,2.],(18,1)),arm_rate=10.)
    controller=SimpleNamespace(sequence_geometry=object(),qp=SimpleNamespace(interface=interface),
        settings=SimpleNamespace(horizon_steps=40,control_dt_s=.01))
    state=SimpleNamespace(q=np.zeros(25),v=np.zeros(24),contacts=np.ones(4,dtype=bool),
        previous_command_target=np.zeros(18))
    return controller,state


def test_local_search_does_not_mutate_frozen_or_other_solver_bindings():
    before=(search.propose_sequence,search.rollout_for_search)
    task,repair=module.local_search();task2,repair2=module.local_search()
    assert before==(search.propose_sequence,search.rollout_for_search)
    assert task.__globals__ is repair.__globals__
    assert task.__globals__ is not task2.__globals__
    assert task.__globals__['propose_sequence'] is not search.propose_sequence
    assert task.__kwdefaults__==search.sequential_recovery_search.__kwdefaults__


def test_shift_uses_actual_history_and_executed_count():
    c,s=fixture();commands=np.zeros((40,18));commands[:,0]=np.arange(40)/100
    commands[:,12]=1.;s.previous_command_target[12]=-.4
    shifted=module.shifted_seed(commands,10,40,s.previous_command_target,c.qp.interface,.01)
    assert shifted[0,0]==.1
    np.testing.assert_allclose(shifted[0,12],-.3)
    assert np.max(abs(np.diff(np.r_[s.previous_command_target[12],shifted[:,12]])))<=.10000001
    np.testing.assert_allclose(commands[:,12],1.)
    assert s.previous_command_target[12]==-.4


def test_new_observation_reverified_and_failed_plan_has_no_action(monkeypatch):
    c,s=fixture();planner=module.RecedingRecoverySequenceV1(c);seen=[]
    def check(controller,state,commands,reference):
        seen.append(copy.deepcopy(state))
        return dict(accepted=bool(state.v[0]==0)),state
    monkeypatch.setattr(module,'characterize',check)
    monkeypatch.setattr(module,'future_equilibrium_reference',lambda *args:dict(solved=False))
    planner.search=lambda *args,**kwargs:dict(commands=None,final_search_commands=np.ones((40,18)))
    planner.restore=lambda *args,**kwargs:dict(commands=None)
    warm=np.zeros((40,18))
    first=planner.solve(s,np.zeros(18),warm);assert first['accepted']
    s.v[0]=.04;s.previous_command_target[12]=.2
    second=planner.solve(s,np.zeros(18),warm)
    assert not second['accepted'] and second['commands'] is None
    assert seen[-1].v[0]==.04 and seen[-1].previous_command_target[12]==.2
    assert first['initial_v'][0]==0
    np.testing.assert_array_equal(warm,np.zeros((40,18)))


def test_no_support_does_not_reuse_previous_feasible_plan():
    c,s=fixture();s.contacts[:]=False
    result=module.RecedingRecoverySequenceV1(c).solve(s,np.zeros(18),np.zeros((40,18)))
    assert not result['accepted'] and result['commands'] is None
    assert 'No current support' in result['reason']
