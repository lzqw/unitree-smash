"""Coordinate checks prevent an orientation probe silently adding velocity."""
from types import SimpleNamespace
import numpy as np
import pinocchio as pin
from scripts.perturb_early_support_domain_v1 import perturb_state


def state():
    rotation=pin.exp3(np.array([.2,-.4,.7]));q=np.zeros(25);q[3:7]=pin.Quaternion(rotation).coeffs()
    return SimpleNamespace(q=q,v=np.arange(24,dtype=float)*.1,root_rotation_world=rotation,
        previous_command_target=np.arange(18,dtype=float))


def test_rotation_probe_keeps_world_twist_and_command_history():
    original=state();q=original.q.copy();v=original.v.copy()
    altered=perturb_state(original,dict(kind='orientation',world_axis=0,amount=.01))
    for part in [slice(0,3),slice(3,6)]:
        np.testing.assert_allclose(altered.root_rotation_world@altered.v[part],original.root_rotation_world@original.v[part],atol=1e-14)
    np.testing.assert_array_equal(original.q,q);np.testing.assert_array_equal(original.v,v)
    np.testing.assert_array_equal(altered.previous_command_target,original.previous_command_target)
    np.testing.assert_array_equal(altered.q[7:],original.q[7:])
    assert np.linalg.norm(pin.log3(altered.root_rotation_world@original.root_rotation_world.T))>.009


def test_world_velocity_probe_changes_only_declared_component():
    original=state()
    for kind,part,other in [('linear_velocity',slice(0,3),slice(3,24)),('angular_velocity',slice(3,6),slice(0,3))]:
        altered=perturb_state(original,dict(kind=kind,world_axis=1,amount=-.02))
        np.testing.assert_allclose(original.root_rotation_world@(altered.v[part]-original.v[part]),[0,-.02,0],atol=1e-14)
        np.testing.assert_array_equal(altered.q,original.q);np.testing.assert_array_equal(altered.v[other],original.v[other])
        np.testing.assert_array_equal(altered.v[6:],original.v[6:])
