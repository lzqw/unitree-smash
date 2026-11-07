import numpy as np
import pytest
from scripts.implicit_command_support_qp_v1 import ImplicitCommandSupportQP, implicit_command_map
from scripts.implicit_support_prediction_v1 import predict_implicit_support
from scripts.whole_body_support_qp_v1 import SupportSettings
from test_admitted_arm_support_qp_v1 import fixture


def test_command_map_solves_end_step_pd_not_initial_pd():
    dt=.01; mass=.2; q=.4; v=.3; kp=80.; kd=4.; a=2.
    matrix,bias=implicit_command_map(np.array([[mass]]),np.array([.7]),
        np.array([q]),np.array([v]),np.array([kp]),np.array([kd]),np.ones((1,1)),dt)
    u=(matrix@np.array([a])+bias)[0]
    next_v=v+dt*a;next_q=q+dt*next_v
    assert kp*(u-next_q)-kd*next_v==pytest.approx(mass*a+.7)
    assert abs(kp*(u-q)-kd*v-(mass*a+.7))>1e-2


@pytest.mark.parametrize('preserve_arm',[True,False])
def test_qp_matches_independent_implicit_step_and_preserves_limits(preserve_arm):
    _,interface,state=fixture()
    state.previous_command_target=state.joint_position.copy()
    qp=ImplicitCommandSupportQP(interface,SupportSettings(),preserve_arm_command=preserve_arm,
                                arm_target_rate_rad_s=10.)
    plan=qp.solve(state,state.joint_position)
    assert plan['accepted'],plan
    prediction=predict_implicit_support(qp.model,interface,state,plan['joint_position_target'],substeps=1)
    np.testing.assert_allclose(prediction['state'].joint_velocity,plan['predicted_joint_velocity'],atol=1e-6)
    np.testing.assert_allclose(prediction['state'].joint_position,plan['predicted_joint_position'],atol=1e-8)
    np.testing.assert_allclose((prediction['state'].v-state.v)/.01,plan['acceleration'],atol=1e-4)
    assert np.max(abs(plan['torque'])-interface.effort)<1e-5
    assert max(prediction['diagnostics'][k] for k in ['friction_violation_n','normal_force_violation_n',
        'joint_velocity_violation_rad_s','joint_position_violation_rad'])<1e-5
    if preserve_arm:
        np.testing.assert_allclose(plan['joint_position_target'][12:],state.joint_position[12:],atol=1e-7)


def test_unobserved_feet_have_no_reaction_force():
    _,interface,state=fixture();state.contacts[:]=False
    state.previous_command_target=state.joint_position.copy()
    qp=ImplicitCommandSupportQP(interface,SupportSettings(),preserve_arm_command=False)
    plan=qp.solve(state,state.joint_position)
    assert plan['accepted'],plan
    np.testing.assert_array_equal(plan['force_world'],np.zeros((4,3)))
    assert plan['residuals']['dynamics']<1e-5


def test_missing_history_is_not_substituted_by_measured_positions():
    _,interface,state=fixture();state.previous_command_target=None
    qp=ImplicitCommandSupportQP(interface,SupportSettings())
    with pytest.raises(ValueError,match='previous final'):
        qp.solve(state,state.joint_position)


def test_rollout_and_terminal_replay_use_same_implicit_response():
    from scripts.landing_implicit_command_mpc_v1 import LandingImplicitCommandMPCV1
    from scripts.landing_support_mpc_v2 import LandingMPCSettingsV2
    _,interface,state=fixture();state.previous_command_target=state.joint_position.copy()
    mpc=LandingImplicitCommandMPCV1(interface,settings=LandingMPCSettingsV2(horizon_steps=3),
                                  preserve_arm_command=True,arm_target_rate_rad_s=10.)
    mpc.set_posture_target(state.joint_position)
    plan=mpc.solve(state,state.joint_position,force=True,lane=7,control_tick=0)
    assert len(plan['horizon'])==3,plan.get('rejection')
    assert all(s['accepted'] for s in plan['horizon']),plan.get('rejection')
    rolled=state
    for step in plan['horizon']:
        rolled=mpc._integrate(rolled,step['acceleration'],step)
        rolled,_=mpc._after_prediction(rolled)
    terminal,_=mpc.replay_horizon(state,plan['horizon'])
    np.testing.assert_allclose(terminal.q,rolled.q,atol=1e-12)
    np.testing.assert_allclose(terminal.v,rolled.v,atol=1e-12)
    np.testing.assert_allclose(terminal.previous_command_target,rolled.previous_command_target,atol=1e-12)
    assert max(x['qp_prediction_acceleration_disagreement'] for x in mpc.prediction_checks)<1e-3
