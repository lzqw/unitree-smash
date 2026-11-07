import numpy as np
from scipy import sparse
from scripts.dense_active_set_qp_v1 import DenseActiveSetQP


def test_equality_and_unilateral_constraints_have_correct_signs():
    solver=DenseActiveSetQP()
    # min ||x||^2 / 2, x0+x1=1 and x0>=.8 => (.8,.2).
    solver.setup(P=sparse.eye(2,format='csc'),q=np.zeros(2),
        A=sparse.csc_matrix([[1.,1.],[1.,0.]]),l=np.array([1.,.8]),u=np.array([1.,np.inf]))
    result=solver.solve()
    assert result.info.status_val==1,solver.last_diagnostics
    np.testing.assert_allclose(result.x,[.8,.2],atol=1e-12)
    assert solver.last_diagnostics['stationarity']<1e-12


def test_inconsistent_constraints_are_not_softened():
    solver=DenseActiveSetQP()
    solver.setup(P=sparse.eye(1,format='csc'),q=np.zeros(1),
        A=sparse.csc_matrix([[1.],[1.]]),l=np.array([1.,-np.inf]),u=np.array([np.inf,0.]))
    result=solver.solve()
    assert result.info.status=='primal infeasible'
    assert result.info.status_val!=1


def test_nonpositive_hessian_is_rejected_without_regularization():
    solver=DenseActiveSetQP()
    solver.setup(P=sparse.csc_matrix([[-1.]]),q=np.zeros(1),
        A=sparse.csc_matrix([[1.]]),l=np.array([-1.]),u=np.array([1.]))
    assert solver.solve().info.status_val!=1


def test_loaded_interval_controller_retains_independent_physical_checks():
    from scripts.landing_active_set_interval_mpc_v1 import LandingActiveSetIntervalMPCV1
    from scripts.landing_support_mpc_v2 import LandingMPCSettingsV2
    from test_admitted_arm_support_qp_v1 import fixture
    _,interface,state=fixture();state.previous_command_target=state.joint_position.copy()
    mpc=LandingActiveSetIntervalMPCV1(interface,settings=LandingMPCSettingsV2(horizon_steps=3),
                                   arm_target_rate_rad_s=10.)
    mpc.set_posture_target(state.joint_position)
    plan=mpc.solve(state,state.joint_position,force=True,lane=0,control_tick=0)
    assert len(plan['horizon'])==3,plan.get('rejection')
    assert all(s['accepted'] for s in plan['horizon']),plan.get('rejection')
    assert all(max(s['residuals'].values())<3e-4 for s in plan['horizon'])
    assert mpc.contract()['qp_backend']=='quadprog'
    assert mpc.contract()['physical_constraints_relaxed'] is False
    terminal,_=mpc.replay_horizon(state,plan['horizon'])
    assert terminal.q[2]>=.12
    np.testing.assert_allclose(terminal.previous_command_target,
        plan['horizon'][-1]['joint_position_target'],atol=1e-10)
