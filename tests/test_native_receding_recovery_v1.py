"""Feedback scheduling and actual-command history, without native physics."""
from types import SimpleNamespace
import copy
import numpy as np
from scripts.native_receding_recovery_v1 import NativeRecedingRecoveryV1


def fixture():
    c=NativeRecedingRecoveryV1.__new__(NativeRecedingRecoveryV1)
    c.limits=np.tile([-2.,2.],(18,1));c.dt=.01;c.arm_rate=10.;c.start=0
    c.commands=np.zeros((100,18));c.plan=None;c.plan_start=None;c.plan_contacts=None
    c.planning_failed=False;c.plans=[];c.teacher_steps=0;c.fallback_steps=0
    c.controller=SimpleNamespace(qp=SimpleNamespace(interface=SimpleNamespace(limits=c.limits,arm_rate=10.)))
    s=SimpleNamespace(q=np.zeros(25),v=np.zeros(24),contacts=np.ones(4,dtype=bool),measured_force_world=np.zeros((4,3)))
    calls=[]
    def solve(state,reference,warm):
        calls.append(copy.deepcopy(state));commands=warm.copy();commands[:,0]=.1
        return dict(accepted=True,commands=commands)
    c.planner=SimpleNamespace(solve=solve)
    return c,s,calls


def test_new_measurement_and_actual_history_reach_scheduled_replan():
    c,s,calls=fixture();prior=np.zeros(18)
    for tick in range(11):
        s.v[0]=tick*.01
        target,mode=c.feedback_target(tick,s,np.zeros(18),prior)
        prior=target.copy()
    assert len(calls)==2 and calls[-1].v[0]==.1
    np.testing.assert_array_equal(calls[-1].previous_command_target,prior)
    assert c.teacher_steps==11 and c.fallback_steps==0


def test_contact_change_replans_before_period():
    c,s,calls=fixture();prior=np.zeros(18)
    c.feedback_target(0,s,prior,prior);s.contacts[0]=False
    c.feedback_target(1,s,prior,prior)
    assert len(calls)==2 and not calls[-1].contacts[0]
    assert c.plans[-1]['contact_changed']


def test_failure_does_not_execute_stale_plan_and_labels_fallback():
    c,s,calls=fixture();prior=np.zeros(18)
    c.feedback_target(0,s,prior,prior);s.contacts[:]=False
    c.planner.solve=lambda *args:dict(accepted=False,commands=None)
    target,mode=c.feedback_target(1,s,np.ones(18),prior)
    assert 'not_certified' in mode and c.planning_failed
    np.testing.assert_allclose(target[12:],.1)
    assert c.teacher_steps==1 and c.fallback_steps==1
    assert target[0]!=c.plan[1,0]
    c.feedback_target(2,s,np.ones(18),target)
    assert len(c.plans)==2 and c.fallback_steps==2


def test_generated_native_action_executes_feedback_and_preserves_other_lanes(tmp_path):
    import torch
    from test_native_recovery_sequence_replay_v1 import setup_replay
    c,public,student,finished,state=setup_replay(tmp_path)
    c.__class__=NativeRecedingRecoveryV1;c.steps=40
    c.plan=None;c.plan_start=None;c.plan_contacts=None;c.planning_failed=False
    c.plans=[];c.teacher_steps=0;c.fallback_steps=0
    c.controller=SimpleNamespace(qp=SimpleNamespace(interface=SimpleNamespace(limits=c.limits,arm_rate=10.)))
    def solve(observed,reference,warm):
        commands=warm.copy();commands[:,0]+=.001
        return dict(accepted=True,commands=commands)
    c.planner=SimpleNamespace(solve=solve);public[15,775]=.02
    result=c.action(2,public,student,finished,state(0))
    np.testing.assert_allclose(c.decode(result[15])[0],c.commands[0,0]+.001,atol=5e-7)
    torch.testing.assert_close(result[:15],student[:15],rtol=0,atol=0)
    torch.testing.assert_close(result[16:],student[16:],rtol=0,atol=0)
    assert c.records[-1]['mode']=='receding_support_sequence' and c.query_check['passed']
