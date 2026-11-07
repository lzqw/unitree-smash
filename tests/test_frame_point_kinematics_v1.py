"""Independent finite-difference checks on the loaded robot model (CPU)."""
import numpy as np
import pinocchio as pin

from scripts.loaded_pinocchio_model_v1 import LoadedPinocchioModelV1
from scripts.frame_point_kinematics_v1 import frame_point_kinematics


def test_point_jacobians_and_bias_match_finite_difference_after_reference_fk():
    model = LoadedPinocchioModelV1()
    q = pin.neutral(model.model)
    q[:3] = [.7, -.2, .35]
    q[3:7] = pin.Quaternion(pin.rpy.rpyToMatrix(.2, -.3, .4)).coeffs()
    q[model.q_indices] = [.1,-.1,.1,-.1,.8,.8,1.,1.,-1.5,-1.5,-1.5,-1.5,
                           .08,1.2,-2.58,-.15,-1.05,.13]
    v = np.random.default_rng(17).normal(0., .4, model.model.nv)
    names = ['FL_thigh','FR_thigh','RL_thigh','RR_thigh','FL_foot']
    # Reproduce the reference writer using a different root and posture.
    qref = q.copy(); qref[:3] = 0.; qref[model.q_indices] += .1
    model.poses(qref)
    actual = frame_point_kinematics(model,q,v,names)
    eps=1e-6
    plus=frame_point_kinematics(model,pin.integrate(model.model,q,eps*v),v,names)
    minus=frame_point_kinematics(model,pin.integrate(model.model,q,-eps*v),v,names)
    for name in names:
        finite_velocity=(plus[name]['position']-minus[name]['position'])/(2*eps)
        finite_jdot=(plus[name]['jacobian']-minus[name]['jacobian'])/(2*eps)
        np.testing.assert_allclose(actual[name]['velocity'],finite_velocity,atol=2e-8,rtol=0.)
        np.testing.assert_allclose(actual[name]['jacobian_dot'],finite_jdot,atol=2e-8,rtol=0.)
    # Returned snapshots cannot alias later FK operations on model.data.
    saved={name:actual[name]['position'].copy() for name in names}
    model.poses(qref)
    for name in names:np.testing.assert_array_equal(actual[name]['position'],saved[name])


def test_clearance_qp_reads_measured_state_after_posture_reference_writes():
    from scripts.reduced_whole_body_support_qp_v1 import ReducedWholeBodySupportQP
    from scripts.loaded_pinocchio_model_v1 import PublicState
    from scripts.whole_body_support_qp_v1 import PositionPDInterface, SupportSettings
    interface=PositionPDInterface(np.tile([-4.,4.],(18,1)),np.full(18,25.),
                                  np.full(18,.5),np.full(18,23.5),np.full(18,30.),.65)
    qp=ReducedWholeBodySupportQP(interface,SupportSettings(nonfoot_clearance_weight=5.))
    q=pin.neutral(qp.model.model);q[:3]=[.7,-.2,.35]
    joints=np.array([.1,-.1,.1,-.1,.8,.8,1.,1.,-1.5,-1.5,-1.5,-1.5,.08,1.2,-2.58,-.15,-1.05,.13])
    q[qp.model.q_indices]=joints;v=np.zeros(24)
    state=PublicState(q,v,np.eye(3),q[:3].copy(),joints.copy(),np.zeros(18),np.ones(4,dtype=bool),0.)
    expected=qp.model.snapshot(q,v)['body_positions_world'].copy()
    plan=qp.solve(state,joints)
    assert len(plan['nonfoot_clearance_points_world'])==4
    for name,position in plan['nonfoot_clearance_points_world'].items():
        np.testing.assert_allclose(position,expected[qp.model.names.index(name)],atol=1e-12,rtol=0.)
