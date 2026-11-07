import numpy as np
from scripts.landing_reference_shooting_mpc_v1 import (
    reference_acceleration,LandingReferenceShootingMPCV1,ReferenceSearchSettings,ShootingReferenceQP)
from scripts.landing_joint_damper_mpc_v1 import LandingJointDamperMPCV1,JointDamperIntervalQP
from scripts.landing_support_mpc_v2 import LandingMPCSettingsV2
from scripts.whole_body_support_qp_v1 import SupportSettings
from test_admitted_arm_support_qp_v1 import fixture


def test_smooth_reference_endpoints_and_convex_bounds():
    knots=np.array([[1.,-2.,3.],[-2.,3.,-4.],[3.,1.,2.]])
    np.testing.assert_array_equal(reference_acceleration(knots,0.),knots[0])
    np.testing.assert_array_equal(reference_acceleration(knots,1.),knots[-1])
    for t in np.linspace(0.,1.,31):
        value=reference_acceleration(knots,t)
        assert np.all(value>=knots.min(0)) and np.all(value<=knots.max(0))


def test_failed_search_does_not_replace_fallback_or_advance_live_memory():
    _,interface,state=fixture()
    state.previous_command_target=state.joint_position.copy()
    settings=LandingMPCSettingsV2(horizon_steps=2,minimum_support_feet=1)
    old=LandingJointDamperMPCV1(interface,qp_settings=SupportSettings(),settings=settings,arm_target_rate_rad_s=10.)
    candidate=LandingReferenceShootingMPCV1(interface,qp_settings=SupportSettings(),settings=settings,
        arm_target_rate_rad_s=10.,search_settings=ReferenceSearchSettings(population=3,elites=2,generations=1))
    # Violate original height so a reduced outer cost cannot confer acceptance.
    state.q[2]=.11;state.root_position_world[2]=.11
    a=old.solve(state,state.joint_position,force=True,control_tick=0)
    b=candidate.solve(state,state.joint_position,force=True,control_tick=0)
    assert not a['accepted'] and not b['accepted']
    assert not b['reference_shooting']['selected_new_plan']
    assert b['reference_shooting']['candidate_count']==4
    assert b['rejection']==a['rejection']
    assert candidate.qp.shooting_knots is None
    for key in ('reference_position','reference_velocity','support_anchors','previous_contacts'):
        np.testing.assert_array_equal(old._live_reference[0][key],candidate._live_reference[0][key])
    assert not hasattr(state,'_shooting_stage')


def test_zero_shooting_reference_keeps_loaded_qp_and_actual_history_unchanged():
    _,interface,state=fixture()
    state.previous_command_target=state.joint_position.copy()
    old=JointDamperIntervalQP(interface,SupportSettings(),arm_target_rate_rad_s=10.)
    new=ShootingReferenceQP(interface,SupportSettings(),arm_target_rate_rad_s=10.)
    new.shooting_knots=np.zeros((3,3))
    state._shooting_stage=17
    a=old.solve(state,state.joint_position);b=new.solve(state,state.joint_position)
    assert a['accepted'] and b['accepted']
    np.testing.assert_array_equal(a['joint_position_target'],b['joint_position_target'])
    np.testing.assert_array_equal(state.previous_command_target,state.joint_position)


def test_free_arm_variant_keeps_drive_and_slew_constraints():
    from scripts.landing_free_arm_reference_shooting_v1 import LandingFreeArmReferenceShootingMPCV1
    _,interface,state=fixture()
    reference=state.joint_position.copy();reference[17]+=.4
    state.previous_command_target=state.joint_position.copy();state.previous_command_target[17]+=.3
    controller=LandingFreeArmReferenceShootingMPCV1(interface,qp_settings=SupportSettings(),arm_target_rate_rad_s=10.)
    result=controller.qp.solve(state,reference)
    assert result['accepted'] and not controller.qp.preserve_arm_command
    assert not result['arm_feasibility_fallback_attempted']
    assert max(result['residuals'].values())<3e-4
    assert np.max(abs(result['joint_position_target'][12:]-state.previous_command_target[12:]))<=.1+1e-10
    assert not controller.contract()['preserve_arm_position_command']
