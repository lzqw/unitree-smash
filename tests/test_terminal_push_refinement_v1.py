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


def test_task_constraints_prevent_position_gain_buying_normal_regression():
    def evaluate(a):
        y=np.zeros(9);y[0]=10-100*a[0];y[3]=1+10*a[0];y[6]=1
        return dict(path_valid=True,normalized_residual=y)
    unconstrained=refine_terminal(evaluate)
    constrained=refine_terminal(evaluate,protect_terminal_tasks=True,backtracking=True)
    assert unconstrained['improved']
    selected=constrained['trials'][constrained['selected_index']]['result']['normalized_residual']
    assert selected[3]<=1+1e-9
    assert constrained['selected_index']==0 or constrained['trials'][constrained['selected_index']]['terminal_task_nonregression']


def test_full_rollout_rejects_nonlinear_regression_even_when_local_model_passes():
    def evaluate(a):
        y=np.zeros(9);y[0]=10-100*a[0];y[3]=1+(0.1 if abs(a[0])>.002 else 0);y[6]=1
        return dict(path_valid=True,normalized_residual=y)
    result=refine_terminal(evaluate,protect_terminal_tasks=True,backtracking=True)
    assert not result['improved'] and result['selected_index']==0
    proposals=result['trials'][-3:]
    assert all(t['valid'] and not t['terminal_task_nonregression'] for t in proposals)
    assert all(t['cost']<result['trials'][0]['cost'] for t in proposals)


def test_separate_tasks_can_all_improve_without_changing_teacher_gate():
    def evaluate(a):
        return dict(path_valid=True,normalized_residual=100*(np.full(9,.02)-a[:9]))
    result=refine_terminal(evaluate,protect_terminal_tasks=True,backtracking=True)
    assert result['improved']
    selected=result['trials'][result['selected_index']]
    assert selected['terminal_task_nonregression']
    assert np.all(selected['terminal_task_norms']<result['protected_task_norms'])
    assert not result['valid_teacher_reference']


def test_velocity_objective_preserves_position_and_normal_and_changes_choice():
    def evaluate(a):
        # Both choices reduce all tasks. Total least squares prefers position,
        # while explicit velocity priority uses the second command direction.
        y=np.zeros(9);y[0]=2-20*a[0]-a[1];y[3]=.3;y[6]=1-a[0]-10*a[1]
        return dict(path_valid=True,normalized_residual=y)
    constraint=(np.array([[1.,1.]]),np.array([-1.]),np.array([.03]))
    args=dict(parameter_radii=np.array([.03,.03]),linear_constraints=constraint,protect_terminal_tasks=True)
    total=refine_terminal(evaluate,**args)
    speed=refine_terminal(evaluate,objective='velocity',**args)
    ts=total['trials'][total['selected_index']]['result']['normalized_residual']
    vs=speed['trials'][speed['selected_index']]['result']['normalized_residual']
    assert speed['improved'] and vs[6]<ts[6]-.1
    assert vs[0]<=2 and vs[3]<=.3 and vs[6]<1
    assert speed['trials'][-1]['terminal_task_nonregression']


def test_position_objective_preserves_normal_and_velocity():
    def evaluate(a):
        y=np.zeros(9);y[0]=1-10*a[0];y[3]=.2-.1*a[0];y[6]=.3-.1*a[0]
        return dict(path_valid=True,normalized_residual=y)
    r=refine_terminal(evaluate,objective='position',parameter_radii=np.array([.03]),protect_terminal_tasks=True)
    assert r['improved'] and r['trials'][r['selected_index']]['result']['normalized_residual'][0] < 1
    assert r['trials'][-1]['terminal_task_nonregression']


def test_velocity_mode_cannot_disable_task_protection():
    import pytest
    with pytest.raises(ValueError,match='protection'):
        refine_terminal(lambda _:None,objective='velocity')


def test_velocity_proposal_cannot_trade_actual_position_for_speed():
    def evaluate(a):
        y=np.zeros(9);y[0]=1;y[3]=.2;y[6]=1-10*a[0]
        if abs(a[0])>.002:y[0]=1.001 # nonlinear response absent from local derivative
        return dict(path_valid=True,normalized_residual=y)
    r=refine_terminal(evaluate,parameter_radii=np.array([.03]),objective='velocity',protect_terminal_tasks=True)
    assert not r['improved'] and r['selected_index']==0
    assert not r['trials'][-1]['terminal_task_nonregression']
