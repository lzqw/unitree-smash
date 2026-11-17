"""Offline local pose/rate refinement, with exact position/normal admission.

Velocity support bounds ignore torque/acceleration and are diagnostics only.
Neither optimizer convergence nor these bounds certify a full trajectory.
"""
import time
import numpy as np
from scipy.optimize import minimize


def velocity_support(jacobian, base_velocity, requested, speed):
    J=np.asarray(jacobian,dtype=float);base=np.asarray(base_velocity,dtype=float)
    target=np.asarray(requested,dtype=float);cap=np.asarray(speed,dtype=float)
    if (J.shape!=(3,6) or base.shape!=(3,) or target.shape!=(3,) or cap.shape!=(6,)
            or not all(np.isfinite(x).all() for x in (J,base,target,cap)) or np.any(cap<=0)):
        raise ValueError('Finite velocity support inputs required')
    relative=target-base;size=np.linalg.norm(relative)
    d=relative/size if size>1e-12 else np.zeros(3)
    maximum=float(d@base+np.sum(cap*np.abs(d@J)))
    return dict(direction=d.tolist(),required_projection_m_s=float(d@target),
        optimistic_projection_upper_m_s=maximum,
        optimistic_shortfall_m_s=max(0.,float(d@target)-maximum),
        physical_feasibility_verified=False)


def refine_velocity_pose(kinematics,q,qd,lower,upper,speed,position,normal,requested,
                         base_velocity_fn=None,position_budget_m=.0005,
                         normal_tolerance=.12,max_iterations=40):
    start=time.perf_counter()
    q=np.array(q,dtype=float,copy=True);qd=np.array(qd,dtype=float,copy=True)
    lo=np.asarray(lower,dtype=float);hi=np.asarray(upper,dtype=float);cap=np.asarray(speed,dtype=float)
    pstar=np.asarray(position,dtype=float);nstar=np.asarray(normal,dtype=float);vstar=np.asarray(requested,dtype=float)
    if (any(x.shape!=(6,) for x in (q,qd,lo,hi,cap)) or any(x.shape!=(3,) for x in (pstar,nstar,vstar))
            or not all(np.isfinite(x).all() for x in (q,qd,lo,hi,cap,pstar,nstar,vstar))
            or np.any(lo>=hi) or np.any(cap<=0) or not 0<position_budget_m<=.0005
            or not 0<normal_tolerance<=.12 or np.linalg.norm(nstar)<1e-12):
        raise ValueError('Invalid velocity-aware pose contract')
    nstar=nstar/np.linalg.norm(nstar)
    def sample(x):
        p,n,J,Jn=(np.asarray(v,dtype=float) for v in kinematics(x[:6]))
        base=np.zeros(3) if base_velocity_fn is None else np.asarray(base_velocity_fn(p),dtype=float)
        if any(v.shape!=shape or not np.isfinite(v).all() for v,shape in
               ((p,(3,)),(n,(3,)),(J,(3,6)),(Jn,(3,6)),(base,(3,)))):
            raise ValueError('Invalid velocity-aware FK')
        return p,n,J,Jn,base,base+J@x[6:]
    best=[None,float('inf')];evaluations=0
    def consider(x):
        nonlocal evaluations
        evaluations+=1
        p,n,J,Jn,base,v=sample(x)
        ok=(np.all(x[:6]>=lo-1e-10) and np.all(x[:6]<=hi+1e-10)
            and np.all(np.abs(x[6:])<=cap+1e-10)
            and np.linalg.norm(p-pstar)<=position_budget_m+1e-10
            and np.linalg.norm(n-nstar)<=normal_tolerance+1e-10)
        error=float(np.linalg.norm(v-vstar))
        if ok and error<best[1]:best[:]=[x.copy(),error]
        return error
    initial=np.r_[q,qd];initial_error=consider(initial)
    if best[0] is None:raise ValueError('Initial pose/rate must satisfy original bounds and submillimetre pose contract')
    def objective(x):
        error=consider(x)
        return .5*error**2
    def gradient(x):
        *_,J,Jn,base,v=sample(x)
        error=v-vstar;g=np.empty(12)
        for j in range(6):
            plus=x.copy();minus=x.copy()
            plus[j]=min(hi[j],x[j]+1e-4);minus[j]=max(lo[j],x[j]-1e-4)
            g[j]=error@(sample(plus)[-1]-sample(minus)[-1])/(plus[j]-minus[j])
        g[6:]=J.T@error
        return g
    def constraints(x):
        p,n,*_=sample(x)
        return np.array([1.-np.sum((p-pstar)**2)/position_budget_m**2,
                         1.-np.sum((n-nstar)**2)/normal_tolerance**2])
    def constraint_jac(x):
        p,n,J,Jn,*_=sample(x)
        return np.c_[np.vstack((-2*(p-pstar)@J/position_budget_m**2,
                               -2*(n-nstar)@Jn/normal_tolerance**2)),np.zeros((2,6))]
    fit=minimize(objective,initial,jac=gradient,method='SLSQP',
        bounds=list(zip(np.r_[lo,-cap],np.r_[hi,cap])),
        constraints=[dict(type='ineq',fun=constraints,jac=constraint_jac)],
        options=dict(maxiter=max_iterations,ftol=1e-9,disp=False))
    consider(fit.x);x=best[0];p,n,J,Jn,base,v=sample(x)
    return dict(q_rad=x[:6].copy(),qd_rad_s=x[6:].copy(),position=p,normal=n,velocity=v,
        requested_velocity=vstar.copy(),position_error_m=float(np.linalg.norm(p-pstar)),
        normal_error=float(np.linalg.norm(n-nstar)),velocity_error_m_s=best[1],
        original_pose_velocity_error_m_s=initial_error,
        improved=best[1]<initial_error-1e-8,solver_success=bool(fit.success),solver_status=int(fit.status),
        solver_message=str(fit.message),iterations=int(fit.nit),evaluations=evaluations,
        saturated_joints=np.flatnonzero(np.abs(x[6:])>=cap-1e-5).tolist(),
        velocity_support=velocity_support(J,base,vstar,cap),seconds=time.perf_counter()-start,
        physical_feasibility_verified=False)
