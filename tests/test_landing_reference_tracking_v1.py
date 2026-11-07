from types import SimpleNamespace
import numpy as np
from scripts.landing_reference_tracking_v1 import LandingReferenceTrackingV1


def setup():
    q=np.zeros(25);q[6]=1.;v=np.zeros(24)
    jac=np.zeros((4,3,24))
    for i in range(4):
        jac[i,:,:3]=np.eye(3);jac[i,:,6+3*i:9+3*i]=np.eye(3)
    model=SimpleNamespace(model=SimpleNamespace(nv=24),v_indices=np.arange(6,24),snapshot=lambda q,v:dict(foot_jacobian=jac))
    base=SimpleNamespace(model=model,interface=SimpleNamespace(limits=np.tile([-1.,1.],(18,1))),action=lambda o,s,p:(np.zeros(18),'base'))
    feedback=LandingReferenceTrackingV1(base,np.tile(q,(520,1)),np.tile(v,(520,1)))
    state=SimpleNamespace(q=q.copy(),v=v.copy(),root_rotation_world=np.eye(3),contacts=np.ones(4,bool))
    return feedback,state


def test_nominal_and_no_support_do_not_change_command():
    f,s=setup()
    np.testing.assert_array_equal(f.action(100,s,np.zeros(18))[0],np.zeros(18))
    s.q[0]=.01;s.contacts[:]=False
    np.testing.assert_array_equal(f.action(100,s,np.zeros(18))[0],np.zeros(18))
    s.contacts[:]=True
    np.testing.assert_array_equal(f.action(80,s,np.zeros(18))[0],np.zeros(18))


def test_root_shift_sign_support_mask_and_command_bounds():
    f,s=setup();s.q[0]=.01;s.contacts[1:]=False
    target,_=f.action(100,s,np.zeros(18))
    # Root should move -x; a fixed foot needs the opposing +x leg shift.
    assert 0<target[0]<.0041
    np.testing.assert_array_equal(target[3:],np.zeros(15))
    s.q[0]=100.;s.v[0]=100.
    target,_=f.action(100,s,np.zeros(18))
    assert np.max(abs(target))<=.08
    assert np.linalg.norm(f.last_diagnostic['requested_virtual_root_delta_local'][:3])<=.025+1e-12
