"""Project a virtual COM reference into observed support before leg tracking.

This 2-D reference QP is not a force controller or stability certificate.
Finite command authority, ramps and real dynamics can violate its virtual
geometric margin; full-remainder dynamics and native execution decide utility.
"""
import copy

import numpy as np
import pinocchio as pin

from scripts.landing_support_mpc_v2 import _hull, _support_margin
from scripts.landing_reference_tracking_v1 import LandingReferenceTrackingV1
from scripts.timed_whole_body_recovery_v1 import timed_factory
from scripts.current_state_recovery_planner_v3 import build_feedback_plan


def project_support(point, feet, margin=.01):
    point=np.asarray(point,float);feet=np.asarray(feet,float)
    if point.shape!=(2,) or feet.ndim!=2 or feet.shape[1]!=2 or not np.isfinite(point).all() or not np.isfinite(feet).all() or not 0<=margin<=.05:
        raise ValueError('Finite horizontal point/support geometry required')
    hull=_hull(feet)
    if len(hull)<3:return None
    edge=np.roll(hull,-1,axis=0)-hull
    lengths=np.linalg.norm(edge,axis=1)
    if np.any(lengths<1e-10):return None
    normal=np.stack([-edge[:,1],edge[:,0]],axis=1)/lengths[:,None]
    bound=np.sum(normal*hull,axis=1)+margin
    def feasible(p):return bool(np.all(normal@p>=bound-1e-10))
    if feasible(point):return point.copy()
    candidates=[]
    for n,b in zip(normal,bound):
        candidate=point+(b-n@point)*n
        if feasible(candidate):candidates.append(candidate)
    for i in range(len(hull)):
        for j in range(i):
            matrix=normal[[i,j]]
            if abs(np.linalg.det(matrix))<1e-10:continue
            candidate=np.linalg.solve(matrix,bound[[i,j]])
            if feasible(candidate):candidates.append(candidate)
    if not candidates:return None
    return min(candidates,key=lambda p:np.sum((p-point)**2)).copy()


class SupportProjectedTrackingV1(LandingReferenceTrackingV1):
    def action(self,offset,state,previous):
        step=offset-80
        if type(step) is not int or not 0<=step<520:raise ValueError('Original clock required')
        # Retain the protected early control path bit-for-bit.
        if step<35 or not np.asarray(state.contacts,bool).any():
            return super().action(offset,state,previous)
        target,mode=self.base.action(offset,state,previous)
        contacts=np.asarray(state.contacts,bool)
        rq=self.reference_q[step];rv=self.reference_v[step]
        R=state.root_rotation_world;ref_R=pin.Quaternion(rq[3:7]).matrix()
        dp=rq[:3]-state.q[:3];dv=ref_R@rv[:3]-R@state.v[:3]
        dr=pin.log3(ref_R@R.T);dw=ref_R@rv[3:6]-R@state.v[3:6]
        translation=self.bounded_vector(self.gains[0]*dp+self.gains[1]*dv,self.limits[0])
        angular=self.gains[2]*dr+self.gains[3]*dw;angular[2]=0.
        angular=self.bounded_vector(angular,self.limits[1])
        snapshot=self.model.snapshot(state.q,state.v)
        ids=[self.model.names.index(n) for n in self.model.feet]
        feet=snapshot['body_positions_world'][ids][contacts]
        com=snapshot['com_world'];requested=com[:2]+translation[:2]
        projected=project_support(requested,feet[:,:2],.01)
        projection_applied=projected is not None
        if projection_applied:
            u=float(np.clip((step*.01-.35)/.20,0.,1.));blend=u*u*(3.-2.*u)
            translation[:2]=(1.-blend)*translation[:2]+blend*(projected-com[:2])
            translation=self.bounded_vector(translation,self.limits[0])
        desired=np.r_[R.T@translation,R.T@angular]
        jac=snapshot['foot_jacobian'][contacts].reshape(-1,self.model.model.nv)
        legs=jac[:,self.model.v_indices[:12]];rhs=-jac[:,:6]@desired
        delta=legs.T@np.linalg.solve(legs@legs.T+.003**2*np.eye(len(rhs)),rhs)
        delta=np.clip(delta,-self.limits[2],self.limits[2])
        proposed=target.copy();proposed[:12]=np.clip(target[:12]+delta,self.interface.limits[:12,0],self.interface.limits[:12,1])
        self.last_diagnostic=dict(active=True,projection_available=projection_applied,reference_step=step,
            current_contacts=np.flatnonzero(contacts).tolist(),virtual_requested_com_xy=requested.tolist(),
            virtual_projected_com_xy=None if projected is None else projected.tolist(),
            requested_virtual_root_delta_local=desired.tolist(),
            virtual_margin_after_ramp_and_bound_m=float(_support_margin(com+translation,feet)),
            physical_stability_certified=False)
        return proposed,'support_projected_reference_tracking_candidate'


def build_support_projected_plan(worker,plan):
    if plan.get('contract')!='support_projected_timed_recovery_v1':raise ValueError('Explicit support-projection contract required')
    keys=worker['factory'].keywords
    if set(keys)!={'reference_q','reference_v','gains'}:raise ValueError('Declared reference factory required')
    def factory(model,interface,initial,knots):
        original=timed_factory(model,interface,initial,knots,**keys,timing=plan['early_timing_s'])
        return SupportProjectedTrackingV1(original.base,keys['reference_q'],keys['reference_v'],**keys['gains'])
    local=dict(worker,factory=factory);underlying=copy.deepcopy(plan)
    underlying['contract']='foot_space_current_state_plan_v1'
    return build_feedback_plan(local,underlying)
