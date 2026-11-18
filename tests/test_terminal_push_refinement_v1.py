import numpy as np
from scripts.terminal_push_refinement_v1 import refine_terminal


def test_proposal_uses_complete_terminal_response_not_one_step_cost():
    calls=[]
    A=np.zeros((9,12));A[:9,:9]=np.eye(9)
    def evaluate(a):
        calls.append(a.copy())
        return dict(path_valid=True,normalized_residual=A@a-.02)
    result=refine_terminal(evaluate)
    assert result['improved'] and len(calls)==14
    selected=result['trials'][result['selected_index']]
    assert selected['label']=='proposal' and selected['cost']<result['trials'][0]['cost']
    assert np.max(abs(selected['amplitude_rad']))<=.03
    np.testing.assert_allclose(result['predicted_residual'],selected['result']['normalized_residual'],atol=1e-12)
    assert not result['valid_teacher_reference'] and not result['native_execution']


def test_failed_full_prefix_is_not_replaced_by_short_prefix_or_average():
    def evaluate(a):return dict(path_valid=bool(np.linalg.norm(a)<.01),normalized_residual=np.full(9,.1)-a[:9])
    result=refine_terminal(evaluate)
    assert not result['improved'] and result['selected_index']==0
    assert not result['trials'][-1]['valid']


def test_nonfinite_baseline_rejected_without_spending_search_budget():
    result=refine_terminal(lambda a:dict(path_valid=True,normalized_residual=np.full(9,np.nan)))
    assert not result['improved'] and len(result['trials'])==1


def test_derivative_failure_cannot_admit_proposal():
    def evaluate(a):
        if np.any(a):raise ValueError('contact solver failed')
        return dict(path_valid=True,normalized_residual=np.ones(9))
    result=refine_terminal(evaluate)
    assert result['rejection']=='NO_VALID_LOCAL_RESPONSE' and not result['improved']


def test_boundary_derivatives_use_available_direction_without_extra_samples():
    bounds=np.tile([-.03,0.],(12,1));seen=[]
    def evaluate(a):
        seen.append(a.copy())
        assert np.all(a<=0), 'must not probe beyond original upper command limit'
        return dict(path_valid=True,normalized_residual=np.ones(9)*.02+a[:9])
    result=refine_terminal(evaluate,amplitude_bounds=bounds)
    assert result['improved'] and len(seen)==14
    assert all(seen[j+1][j]<0 for j in range(12))


def test_fixed_parameter_is_not_perturbed_or_granted_an_artificial_interval():
    bounds=np.tile([-.03,.03],(12,1));bounds[0]=0
    def evaluate(a):
        assert a[0]==0
        return dict(path_valid=True,normalized_residual=np.ones(9)*.02+a[:9])
    result=refine_terminal(evaluate,amplitude_bounds=bounds)
    assert 0 not in result['usable_columns']
    assert len(result['trials'])==13


def test_backtracking_revalidates_entire_scaled_sequence():
    seen=[]
    def evaluate(a):
        seen.append(a.copy())
        return dict(path_valid=bool(np.max(abs(a))<.015),
                    normalized_residual=100*(np.full(9,.02)-a[:9]))
    result=refine_terminal(evaluate,backtracking=True)
    assert result['improved'] and result['accepted_step_scale']==.5
    assert len(seen)==15 and not result['trials'][-2]['valid']
    np.testing.assert_allclose(seen[-1],seen[-2]*.5)
    assert result['selected_index']==14
    assert not result['valid_teacher_reference']


def test_backtracking_exhaustion_keeps_verified_baseline():
    def evaluate(a):
        return dict(path_valid=bool(np.max(abs(a))<=.0011),
                    normalized_residual=100*(np.full(9,.02)-a[:9]))
    result=refine_terminal(evaluate,backtracking=True)
    assert len(result['trials'])==16 and result['selected_index']==0
    assert not result['improved']
    assert all(not t['valid'] for t in result['trials'][-3:])
