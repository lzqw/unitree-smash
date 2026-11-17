import numpy as np
import pinocchio as pin
from scripts.loaded_pinocchio_model_v1 import LoadedPinocchioModelV1
from scripts.decompose_prediction_error_v1 import attribution,momentum


def test_attribution_reconstructs_error_and_does_not_modify_states():
    model=LoadedPinocchioModelV1();qn=pin.neutral(model.model);qp=qn.copy()
    qp[:3]=[.1,.2,.3];qp[3:7]=pin.Quaternion(pin.exp3(np.array([.1,.2,-.1]))).coeffs()
    qp[model.q_indices[12:]]=[.1,.5,-.6,.2,.1,-.1]
    saved=qp.copy();r=attribution(model,qp,qn,[.0256,0,.07])
    assert r['reconstruction_error_m']<1e-14
    assert r['base_effect_norm_m']>0 and r['joint_effect_norm_m']>0
    np.testing.assert_array_equal(qp,saved)
    base=attribution(model,qn,qn,[.0256,0,.07]);assert base['total_error_norm_m']==0.


def test_momentum_is_detached_from_pinocchio_workspace():
    model=LoadedPinocchioModelV1();q=pin.neutral(model.model);v=np.zeros(model.model.nv)
    first=momentum(model,q,v);saved=first.copy();v[:3]=[1,2,3]
    second=momentum(model,q,v)
    np.testing.assert_array_equal(first,saved)
    assert np.linalg.norm(second[:3])>0
