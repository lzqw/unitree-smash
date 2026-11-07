import numpy as np
import pytest
from scripts.interval_command_support_qp_v1 import IntervalCommandSupportQP
from scripts.implicit_support_prediction_v1 import predict_implicit_support
from scripts.whole_body_support_qp_v1 import SupportSettings
from test_admitted_arm_support_qp_v1 import fixture


@pytest.mark.parametrize('preserve_arm',[True,False])
def test_optimized_command_passes_independent_entire_interval(preserve_arm):
    _,interface,state=fixture();state.previous_command_target=state.joint_position.copy()
    qp=IntervalCommandSupportQP(interface,SupportSettings(),preserve_arm_command=preserve_arm,
        arm_target_rate_rad_s=10.)
    plan=qp.solve(state,state.joint_position)
    assert plan['accepted'],{k:plan.get(k) for k in ['solver_status','rejection','residuals']}
    predicted=predict_implicit_support(qp.model,interface,state,plan['joint_position_target'],substeps=16)
    np.testing.assert_allclose(predicted['state'].joint_position,plan['predicted_joint_position'],atol=1e-10)
    assert max(v for k,v in predicted['diagnostics'].items() if k!='maximum_active_set_iterations')<2e-4
    assert plan['constraint_rows']>16*18*3
    if preserve_arm:np.testing.assert_allclose(plan['joint_position_target'][12:],state.joint_position[12:],atol=1e-7)


def test_airborne_model_never_provides_ground_reaction():
    _,interface,state=fixture();state.contacts[:]=False;state.previous_command_target=state.joint_position.copy()
    qp=IntervalCommandSupportQP(interface,SupportSettings(),preserve_arm_command=False,arm_target_rate_rad_s=10.)
    plan=qp.solve(state,state.joint_position)
    assert plan['accepted'],plan.get('rejection')
    np.testing.assert_array_equal(plan['force_world'],np.zeros((4,3)))
    assert plan['acceleration'][2]<-9.


def test_original_termination_height_cannot_be_accepted_as_recovery():
    _,interface,state=fixture();state.q[2]=.11;state.root_position_world[2]=.11
    state.previous_command_target=state.joint_position.copy()
    qp=IntervalCommandSupportQP(interface,SupportSettings(),arm_target_rate_rad_s=10.)
    plan=qp.solve(state,state.joint_position)
    assert not plan['accepted']


def test_rollout_and_terminal_replay_keep_interval_dynamics_and_command_history():
    from scripts.landing_interval_command_mpc_v1 import LandingIntervalCommandMPCV1
    from scripts.landing_support_mpc_v2 import LandingMPCSettingsV2
    _,interface,state=fixture();state.previous_command_target=state.joint_position.copy()
    mpc=LandingIntervalCommandMPCV1(interface,settings=LandingMPCSettingsV2(horizon_steps=3),
                                  arm_target_rate_rad_s=10.)
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
    # Quiet velocity and a large support polygon must not mask task termination.
    state.q[2]=.119
    assert not mpc._terminal_ok(state,mpc.risk(state))
