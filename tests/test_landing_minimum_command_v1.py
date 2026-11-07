import numpy as np
import pytest
from scripts.landing_minimum_command_mpc_v1 import (
    command_change_cost, LandingMinimumCommandMPCV1)
from scripts.landing_full_command_interval_mpc_v1 import LandingFullCommandIntervalMPCV1
from scripts.whole_body_support_qp_v1 import SupportSettings
from test_admitted_arm_support_qp_v1 import fixture


def test_scaled_cost_matches_physical_command_difference():
    rng=np.random.default_rng(7)
    center=rng.normal(size=18);student=rng.normal(size=18)
    scale=rng.uniform(.2,2.,18);x=rng.normal(size=18);weight=1.3
    diagonal,linear=command_change_cost(center,scale,student,weight)
    constant=.5*weight*np.sum((center-student)**2)
    assert np.isclose(.5*np.sum(diagonal*x*x)+linear@x+constant,
                      .5*weight*np.sum((center+scale*x-student)**2),rtol=1e-13)
    np.testing.assert_allclose(diagonal*x+linear,
                               weight*scale*(center+scale*x-student),rtol=1e-13)


def test_zero_weight_preserves_existing_full_command_qp():
    _,interface,state=fixture();state.previous_command_target=state.joint_position.copy()
    old=LandingFullCommandIntervalMPCV1(interface,qp_settings=SupportSettings(),arm_target_rate_rad_s=10.)
    new=LandingMinimumCommandMPCV1(interface,qp_settings=SupportSettings(),arm_target_rate_rad_s=10.,command_change_weight=0.)
    a=old.qp.solve(state,state.joint_position);b=new.qp.solve(state,state.joint_position)
    assert a['accepted'] and b['accepted']
    np.testing.assert_array_equal(a['joint_position_target'],b['joint_position_target'])
    assert a['constraint_rows']==b['constraint_rows']
    assert a['residuals']==b['residuals']


def test_tracking_cost_reduces_change_without_changing_hard_limits():
    _,interface,state=fixture();state.previous_command_target=state.joint_position.copy()
    reference=state.joint_position.copy();student=reference.copy();student[:12]+=.08
    outputs=[]
    for weight in [0.,1.]:
        m=LandingMinimumCommandMPCV1(interface,qp_settings=SupportSettings(),arm_target_rate_rad_s=10.,command_change_weight=weight)
        m.qp.student_command_reference=student.copy()
        out=m.qp.solve(state,reference)
        assert out['accepted']
        assert max(out['residuals'].values())<=m.qp.settings.residual_tolerance
        outputs.append(out)
        np.testing.assert_array_equal(m.qp.interface.limits,interface.limits)
        np.testing.assert_array_equal(m.qp.interface.effort,interface.effort)
        assert m.qp._tracking_context is None
    assert outputs[1]['constraint_rows']==outputs[0]['constraint_rows']
    assert outputs[1]['student_command_change_l2_rad']<outputs[0]['student_command_change_l2_rad']
    np.testing.assert_array_equal(outputs[1]['command_change_reference'],student)


def test_invalid_weight_rejected():
    with pytest.raises(ValueError):
        command_change_cost(np.zeros(18),np.ones(18),np.zeros(18),-1.)
