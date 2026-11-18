import numpy as np
import pinocchio as pin
import pytest
from scripts.contact_terminal_attribution_v1 import contact_kinematics,decompose_position
from scripts.loaded_pinocchio_model_v1 import LoadedPinocchioModelV1


def test_vector_decomposition_does_not_add_lengths():
    d=decompose_position([1,0,0],[1,1,0],[0,0,0])
    np.testing.assert_allclose(d['actual_minus_target_world_m'],
        d['actual_minus_reference_world_m']+d['reference_minus_target_world_m'])
    assert d['vector_identity_error_m']==0
    assert d['actual_error_m']<d['reference_error_m']+d['tracking_displacement_m']
    with pytest.raises(ValueError):decompose_position([np.nan,0,0],[0,0,0],[0,0,0])


def test_contact_jacobians_match_finite_difference_and_do_not_alias():
    model=LoadedPinocchioModelV1();q=pin.neutral(model.model)
    q[model.q_indices[12:]]=[.2,.4,-.8,.3,.5,-.2]
    offset=np.array([.03,.01,-.02]);p,n,jp,jn=contact_kinematics(model,q,offset)
    saved=[x.copy() for x in (p,n,jp,jn)]
    for i in list(range(6))+list(model.v_indices[12:]):
        tangent=np.zeros(model.model.nv);tangent[i]=1e-7
        pp,nn,_,_=contact_kinematics(model,pin.integrate(model.model,q,tangent),offset)
        np.testing.assert_allclose((pp-p)/1e-7,jp[:,i],atol=2e-7)
        np.testing.assert_allclose((nn-n)/1e-7,jn[:,i],atol=2e-7)
    for before,after in zip(saved,(p,n,jp,jn)):np.testing.assert_array_equal(before,after)


def test_world_pose_transport_and_base_velocity_identity():
    model=LoadedPinocchioModelV1();q=pin.neutral(model.model);q[model.q_indices[12:]]=[.2,.4,-.8,.3,.5,-.2]
    offset=np.array([.03,.01,-.02]);p,n,jp,jn=contact_kinematics(model,q,offset)
    R=pin.exp3(np.array([.4,-.3,.2]));shift=np.array([.7,-.2,.5]);moved=q.copy()
    moved[:3]=shift;moved[3:7]=pin.Quaternion(R).coeffs()
    mp,mn,mjp,mjn=contact_kinematics(model,moved,offset)
    np.testing.assert_allclose(mp,shift+R@p,atol=1e-12)
    np.testing.assert_allclose(mn,R@n,atol=1e-12)
    v=np.zeros(model.model.nv);v[:6]=[.2,-.1,.3,.5,-.2,.1]
    np.testing.assert_allclose(mjp@v,R@v[:3]+np.cross(R@v[3:6],mp-shift),atol=1e-12)


def test_loaded_model_matches_geometry_used_for_conservative_reach_bound():
    from quadsmash.control.full_racket_geometry_v4 import FullRacketGeometryV4
    model=LoadedPinocchioModelV1();geometry=FullRacketGeometryV4()
    offset=np.array([.0256,.003,.07])
    for angles in ([.2,.4,-.8,.3,.5,-.2],[-.7,1.1,-1.2,.8,.9,1.7]):
        q=pin.neutral(model.model);q[model.q_indices[12:]]=angles
        p,n,_,_=contact_kinematics(model,q,offset)
        center,rotation=geometry.forward(angles,base_position=(0,0,0))
        # Loaded USD transforms and source URDF transforms have distinct
        # numeric serialization. This checks micrometre-level model agreement,
        # not a relaxed contact/physical admission gate.
        np.testing.assert_allclose(p,center+rotation@offset,atol=1e-6,rtol=0)
        np.testing.assert_allclose(n,rotation[:,0],atol=1e-6,rtol=0)
