"""Offline initialization for a different point on the unchanged string bed."""
import numpy as np
import torch
from scipy.optimize import least_squares

from .full_racket_geometry_v4 import FullRacketGeometryV4
from .full_racket_kinematics_v4 import FullRacketKinematicsV4


def retarget_contact_uv(joints, old_uv, new_uv, lower_delta, upper_delta, face=1):
    """Fit one bounded joint offset over measured TRAIN arm configurations.

    Preserve the former control point and face direction while moving that
    point toward the center of the existing strings. This is kinematics only:
    there is no simulated state write or claim of executable dynamics.
    """
    joints=np.asarray(joints,dtype=float)
    if joints.ndim!=2 or joints.shape[1]!=6 or len(joints)==0 or not np.isfinite(joints).all():
        raise ValueError('Require finite measured arm configurations')
    model=FullRacketGeometryV4()
    old_uv,new_uv=[np.asarray(v,dtype=float) for v in [old_uv,new_uv]]
    for uv in [old_uv,new_uv]:
        if uv.shape!=(2,) or not np.isfinite(uv).all() or np.sum((uv/model.half_extents[1:])**2)>1:
            raise ValueError('Control point must lie inside the unchanged strings ellipse')
    if face not in [-1,1]:
        raise ValueError('Require an existing racket face')
    lo,hi=[np.asarray(v,dtype=float) for v in [lower_delta,upper_delta]]
    if any(v.shape!=(6,) or not np.isfinite(v).all() for v in [lo,hi]) or np.any(lo>=hi):
        raise ValueError('Require finite ordered joint-offset bounds')
    kin=FullRacketKinematicsV4().double()
    q=torch.tensor(joints,dtype=torch.float64)
    old_local=torch.tensor([face*.0256,*old_uv],dtype=torch.float64)
    new_local=torch.tensor([face*.0256,*new_uv],dtype=torch.float64)
    with torch.inference_mode():
        target,normal,*_=kin(q,old_local)
        before,*_=kin(q,new_local)
    if np.array_equal(old_uv,new_uv):
        return dict(delta_rad=np.zeros(6),position_error_before_m=np.zeros(len(q)),
                    position_error_after_m=np.zeros(len(q)),normal_error_after_rad=np.zeros(len(q)))
    # A single shared offset must respect every measured configuration and
    # the caller's original nominal-command parameter limits.
    lo=np.maximum(lo,np.max(model.bounds[:,0]-joints,axis=0))+1e-8
    hi=np.minimum(hi,np.min(model.bounds[:,1]-joints,axis=0))-1e-8
    if np.any(lo>=hi):
        raise ValueError('No common offset satisfies the original joint bounds')
    def residual(delta):
        with torch.inference_mode():
            point,n,*_=kin(q+torch.tensor(delta),new_local)
        return np.r_[5*(point-target).numpy().ravel(),.3*(n-normal).numpy().ravel(),.01*delta]
    fit=least_squares(residual,np.clip(np.zeros(6),lo,hi),bounds=(lo,hi),max_nfev=300,
                      ftol=1e-10,xtol=1e-10,gtol=1e-10)
    with torch.inference_mode():
        point,n,*_=kin(q+torch.tensor(fit.x),new_local)
    return dict(delta_rad=fit.x.copy(),position_error_before_m=(before-target).norm(dim=1).numpy(),
                position_error_after_m=(point-target).norm(dim=1).numpy(),
                normal_error_after_rad=torch.acos((n*normal).sum(1).clamp(-1,1)).numpy())
