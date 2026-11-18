import copy
import numpy as np
import pytest
from scripts.full_prefix_tracking_v1 import arm_tracking_response,ARM_TRACKING_LIMIT_RAD
from scripts.terminal_push_refinement_v1 import refine_terminal


def trace():
    rows=[]
    for k in range(3):
        e=np.array([.2,-.1,.05,-.03,0,.01])*(k+1)/3
        rows.append(dict(node=k,q=np.r_[np.zeros(7),e],planned_arm_reference_rad=np.zeros(6),arm_tracking_error_rad=float(np.max(abs(e)))))
    return dict(path_valid=True,trajectory=rows)


def test_signed_responses_keep_each_node_joint_and_copy_storage():
    r=trace();y,lo,hi=arm_tracking_response(r,np.arange(7,13),3)
    assert y.shape==(18,) and ARM_TRACKING_LIMIT_RAD==.25
    np.testing.assert_allclose(y.reshape(3,6)[-1],[.2,-.1,.05,-.03,0,.01])
    np.testing.assert_array_equal(lo,np.full(18,-.25));np.testing.assert_array_equal(hi,np.full(18,.25))
    before=r['trajectory'][-1]['q'].copy();y[:]=99
    np.testing.assert_array_equal(r['trajectory'][-1]['q'],before)


def test_incomplete_nonfinite_or_inconsistent_history_fails_closed():
    for mode in ('partial','nan','record','node','limit'):
        r=trace()
        if mode=='partial':r['trajectory'].pop()
        if mode=='nan':r['trajectory'][1]['q'][8]=np.nan
        if mode=='record':r['trajectory'][1]['arm_tracking_error_rad']=0
        if mode=='node':r['trajectory'][1]['node']=2
        if mode=='limit':
            r['trajectory'][1]['q'][8]=.251;r['trajectory'][1]['arm_tracking_error_rad']=.251
        with pytest.raises(ValueError):arm_tracking_response(r,np.arange(7,13),3)


def test_path_constraints_shape_the_step_not_only_reject_afterward():
    def evaluate(a):
        return dict(path_valid=True,normalized_residual=np.r_[100*(a[0]-.04),np.zeros(8)],tracking=np.array([.2+2*a[0],-.23-a[1]]))
    def response(r):return r['tracking'],np.full(2,-.25),np.full(2,.25)
    r=refine_terminal(evaluate,path_response_fn=response,backtracking=True)
    assert r['improved'] and r['path_response_constraint_rows']==2
    selected=r['trials'][r['selected_index']]
    assert selected['path_response_valid'] and selected['amplitude_rad'][0]<=.025+1e-10
    assert np.all(abs(selected['actual_path_response'])<=.25)
    assert not r['valid_teacher_reference']


def test_nonlinear_path_violation_is_not_admitted_by_feasible_linear_model():
    def evaluate(a):
        y=.2+2*a[0]+(.1 if abs(a[0])>.002 else 0)
        return dict(path_valid=True,normalized_residual=np.r_[100*(a[0]-.04),np.zeros(8)],tracking=np.array([y]))
    def response(r):return r['tracking'],np.array([-.25]),np.array([.25])
    r=refine_terminal(evaluate,path_response_fn=response,backtracking=True)
    assert not r['improved'] and r['selected_index']==0
    assert all(not row['valid'] for row in r['trials'][-3:])
    assert r['predicted_path_response'][0]<=.25+1e-10
