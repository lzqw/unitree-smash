import numpy as np
from scripts.nonlinear_push_correction_v1 import bounded_response_step


def inputs():return np.zeros(18),np.tile([-1.,1.],(18,1)),np.ones(6)*.1,np.ones(6)


def test_joint_responses_are_independent_and_arm_command_is_preserved():
    command,limits,desired,scales=inputs();command[12:]=.25
    original=command.copy();calls=[]
    A=np.tile(np.eye(6),(1,2))
    def evaluate(u):
        calls.append(u.copy())
        return dict(valid=True,response=A@u[:12])
    result=bounded_response_step(command,limits,desired,scales,evaluate)
    assert result['accepted'] and len(calls)==14
    assert result['cost_after']<result['cost_before']
    assert np.max(abs(result['command'][:12]-command[:12]))<=.03
    for u in calls:np.testing.assert_array_equal(u[12:],original[12:])
    np.testing.assert_array_equal(command,original)
    np.testing.assert_allclose(result['linear_predicted_response'],A@result['command'][:12],atol=1e-12)


def test_unverified_proposal_is_not_adopted_or_replaced_by_mean():
    command,limits,desired,scales=inputs()
    def evaluate(u):
        return dict(valid=bool(np.linalg.norm(u[:12])<.01),response=u[:6],rejection='fixture physical constraint')
    result=bounded_response_step(command,limits,desired,scales,evaluate)
    assert not result['accepted']
    assert result['evaluations'][-1]['kind']=='proposal' and not result['evaluations'][-1]['valid']
    np.testing.assert_array_equal(result['command'],command)


def test_missing_or_nonfinite_baseline_never_creates_reference():
    command,limits,desired,scales=inputs()
    for response,valid in [(np.full(6,np.nan),True),(np.zeros(6),False)]:
        result=bounded_response_step(command,limits,desired,scales,lambda u:dict(response=response,valid=valid))
        assert not result['accepted'] and len(result['evaluations'])==1
        assert result['rejection']=='BASELINE_PHYSICALLY_UNVERIFIED'


def test_input_bounds_and_repeated_calls_do_not_share_history():
    command,limits,desired,scales=inputs();command[0]=1.
    fn=lambda u:dict(response=u[:6],valid=True)
    one=bounded_response_step(command,limits,desired,scales,fn)
    two=bounded_response_step(command,limits,desired,scales,fn)
    np.testing.assert_array_equal(one['command'],two['command'])
    assert all(np.max(x['command'])<=1 for x in one['evaluations'])
    one['command'][:]=99
    assert np.max(two['command'])<=1
