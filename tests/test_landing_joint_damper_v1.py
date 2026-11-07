import numpy as np
import pytest
from scripts.landing_joint_damper_mpc_v1 import joint_damper_constraints, JointDamperIntervalQP
from scripts.landing_arm_feasibility_mpc_v1 import ArmFeasibilityIntervalQP
from scripts.whole_body_support_qp_v1 import SupportSettings
from scripts.implicit_support_prediction_v1 import predict_implicit_support
from test_admitted_arm_support_qp_v1 import fixture


def test_position_alone_can_pass_while_outward_velocity_is_rejected():
    # q(u)=0.99 is within [-1,1], but v(u)=u must be <=0.2 rad/s.
    stage = dict(velocity_map=np.ones((1,1)),velocity_bias=np.zeros(1),
                 joint_position_map=np.zeros((1,1)),joint_position_bias=np.array([.99]))
    A,l,u = joint_damper_constraints([stage], np.eye(1), np.array([[-1.,1.]]),20.)
    assert (A@np.array([1.]) > u).all()
    assert (A@np.array([.1]) <= u).all()
    assert (A@np.array([-.1]) >= l).all()
    with pytest.raises(ValueError):
        joint_damper_constraints([stage],np.eye(1),np.array([[-1.,1.]]),0.)


def test_loaded_solution_preserves_physical_bounds_and_independent_prediction():
    _, interface, state = fixture()
    state.previous_command_target = state.joint_position.copy()
    candidate = JointDamperIntervalQP(interface,SupportSettings(),arm_target_rate_rad_s=10.)
    reference = state.joint_position.copy()
    result = candidate.solve(state,reference)
    assert result['accepted']
    assert result['joint_damper_constraint_rows']==16*18
    prediction=predict_implicit_support(candidate.model,interface,state,result['joint_position_target'],substeps=16)
    q,v=prediction['state'].joint_position,prediction['state'].joint_velocity
    assert np.all(v+20.*q >= 20.*interface.limits[:,0]-3e-4)
    assert np.all(v+20.*q <= 20.*interface.limits[:,1]+3e-4)
    assert max(result['residuals'].values()) < 3e-4
    assert candidate._damper_context is None


def test_inactive_envelope_does_not_change_old_command_or_reference_clock():
    _,interface,state=fixture()
    state.previous_command_target=state.joint_position.copy()
    old=ArmFeasibilityIntervalQP(interface,SupportSettings(),arm_target_rate_rad_s=10.)
    new=JointDamperIntervalQP(interface,SupportSettings(),arm_target_rate_rad_s=10.)
    a=old.solve(state,state.joint_position);b=new.solve(state,state.joint_position)
    assert a['accepted'] and b['accepted']
    np.testing.assert_allclose(a['joint_position_target'],b['joint_position_target'],atol=1e-9,rtol=0)
    for key in ('reference_position','reference_velocity','support_anchors','previous_contacts'):
        np.testing.assert_array_equal(old.reference_state()[key],new.reference_state()[key])
