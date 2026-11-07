"""Check arm torque prescription against the original command interface."""
import numpy as np
import pinocchio as pin
import pytest

from scripts.admitted_arm_support_qp_v1 import AdmittedArmSupportQP
from scripts.loaded_pinocchio_model_v1 import PublicState
from scripts.whole_body_support_qp_v1 import PositionPDInterface, SupportSettings


def fixture():
    interface = PositionPDInterface(np.tile([-4., 4.], (18, 1)),
        np.r_[np.full(12, 25.), np.full(6, 80.)],
        np.r_[np.full(12, .5), np.full(6, 4.)],
        np.r_[np.full(12, 23.5), np.full(6, 100.)],
        np.r_[np.full(12, 30.), np.full(6, 5.)], .65)
    qp = AdmittedArmSupportQP(interface, SupportSettings())
    q = pin.neutral(qp.model.model); q[:3] = [0., 0., .35]
    joints = np.array([0.,0.,0.,0.,.8,.8,.8,.8,-1.5,-1.5,-1.5,-1.5,
                       .08,1.2,-2.58,-.15,-1.05,.13])
    q[qp.model.q_indices] = joints
    v = np.zeros(24); v[qp.model.v_indices[12:]] = [.2,-.3,.4,-.5,.6,-.7]
    state = PublicState(q,v,np.eye(3),q[:3].copy(),joints.copy(),
                        v[qp.model.v_indices].copy(),np.ones(4,dtype=bool),0.)
    return qp, interface, state


def test_prescribed_arm_torque_is_inside_actual_admission_interval():
    qp, interface, state = fixture()
    target = state.joint_position.copy()
    target[12:] += [.9,-.8,.7,-.6,.5,-.4]
    original = target.copy()
    plan = qp.solve(state,target)
    low, high = interface.torque_bounds(state.joint_position,state.joint_velocity,.01)
    prescribed_low, prescribed_high = qp.interface.torque_bounds(
        state.joint_position,state.joint_velocity,.01)
    assert np.all(prescribed_low[12:] >= low[12:] - 1e-10)
    assert np.all(prescribed_high[12:] <= high[12:] + 1e-10)
    np.testing.assert_array_equal(prescribed_low[12:],prescribed_high[12:])
    np.testing.assert_allclose(plan['admitted_arm_reference'],
        state.joint_position[12:] + [.05,-.05,.05,-.05,.05,-.05],atol=1e-12)
    np.testing.assert_array_equal(target,original)
    if plan['accepted']:
        np.testing.assert_allclose(plan['joint_position_target'][12:],
                                   plan['admitted_arm_reference'],atol=1e-12)


def test_invalid_posture_is_rejected_instead_of_hidden_by_admission():
    qp, _, state = fixture()
    target=state.joint_position.copy();target[12]=10.
    with pytest.raises(ValueError,match='outside original bounds'):
        qp.solve(state,target)
