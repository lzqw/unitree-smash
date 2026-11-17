"""Offline coupled pose/rate/string-point max-min task search.

Contact time is enumerated by the unchanged outer event window. No case labels,
future truth, rate-only search, or execution authority enter this optimizer.
"""
import time
import numpy as np
from scipy.optimize import minimize
from .task_constrained_impact_goal_v2 import task_margins


def optimize_task_hit(kinematics,q,qd,lower,upper,speed,position,local,strip,
                      base_velocity_fn,rotation,displacement,metadata,impact,
                      court,incoming_planning,max_iterations=32):
    start_time=time.perf_counter()
    q=np.array(q,copy=True);qd=np.asarray(qd);speed=np.asarray(speed)
    lower=np.asarray(lower);upper=np.asarray(upper);position=np.asarray(position);local=np.asarray(local)
    R=np.asarray(rotation);displacement=np.asarray(displacement);incoming=np.asarray(incoming_planning)
    if (q.shape!=(6,) or qd.shape!=(6,) or np.any(speed<=0) or np.any(speed>5.) or
        not all(np.isfinite(x).all() for x in (q,qd,lower,upper,speed,position,local,R,incoming))):
        raise ValueError('Finite original-bounded six-axis inputs required')
    center=np.asarray(strip['center']);half=.5*np.asarray(strip['size'])
    uvlo=center[1:]-half[1:]+1e-8;uvhi=center[1:]+half[1:]-1e-8
    if np.any(uvlo>=uvhi):raise ValueError('Nonempty authored string strip required')
    lo=np.r_[lower,-speed,uvlo,-1000.];hi=np.r_[upper,speed,uvhi,2.]
    x0=np.clip(np.r_[q,qd,local[1:],-1.],lo,hi)
    cache={};best=None;best_failed=None
    def evaluate(x):
        nonlocal best,best_failed
        key=np.asarray(x[:14],dtype='<f8').tobytes()
        if key in cache:return cache[key]
        contact=np.r_[local[0],x[12:14]]
        p,n,jp,_=kinematics(x[:6],contact);p=np.asarray(p);n=np.asarray(n);n=n/np.linalg.norm(n)
        velocity=base_velocity_fn(p)+jp@x[6:12]
        out=impact(dict(metadata,contact_point_local=contact,
            achieved_position_current_frame_m=displacement+R@p,
            achieved_normal_current_frame=R@n,achieved_velocity_current_frame_m_s=R@velocity))
        m=task_margins(out,court)
        # The task normal moves with the achieved outgoing goal, rather than
        # remaining locked to a rejected inverse goal. Friction is still in
        # the forward model; this normal agreement remains an extra gate.
        actual=np.asarray(out.get('outgoing_velocity'),dtype=float)
        basis=np.asarray(metadata['planning_to_world_rotation'])
        delta=basis.T@actual-incoming if actual.shape==(3,) and np.isfinite(actual).all() else np.zeros(3)
        goal_n=delta/np.linalg.norm(delta) if np.linalg.norm(delta)>1e-10 else R@n
        en=float(np.linalg.norm(R@n-goal_n));ep=float(np.linalg.norm(p-position))
        geometrical=bool(ep<=.0005 and en<=.12 and np.all(x[:6]>=lower) and np.all(x[:6]<=upper) and np.all(abs(x[6:12])<=speed))
        all_margins=dict(zip(('speed','downward','forward','net','court','target'),m['normalized'].tolist()))
        all_margins.update(position=1-ep/.0005,normal=1-en/.12)
        row=dict(limiting_feasibility_constraint=min(all_margins,key=all_margins.get),constraint_margins=all_margins,q_rad=x[:6].copy(),qd_rad_s=x[6:12].copy(),contact_point_local=contact,
            position=p,normal=n,velocity=velocity,task_target_normal_current=goal_n,
            position_error_m=ep,normal_error=en,task_margins=m,impact_outcome=out,
            POSITION_FEASIBLE=ep<=.0005,NORMAL_FEASIBLE=en<=.12,
            ANALYTIC_GOAL_VALID=m['valid'],TASK_FEASIBLE_RACKET_VELOCITY=bool(geometrical and m['valid']),
            saturated_joints=np.flatnonzero(abs(x[6:12])>=speed-1e-5).tolist())
        if geometrical and m['valid'] and (best is None or m['gamma']>best['task_margins']['gamma']):best=row
        violation=max(ep/.0005-1,en/.12-1,-m['gamma'])
        if best_failed is None or violation<best_failed[0]:best_failed=(violation,row)
        cache[key]=row
        return row
    def constraints(x):
        row=evaluate(x)
        return np.r_[row['task_margins']['normalized']-x[14],
                     1-row['position_error_m']/.0005,1-row['normal_error']/.12]
    # Absolute FD increments required by the production float32 FK.
    steps=np.r_[np.full(6,1e-4),np.full(6,1e-3),np.full(2,1e-5),1e-4]
    def jac(x):
        cols=[]
        for j,h in enumerate(steps):
            xp=x.copy();xm=x.copy();xp[j]=min(hi[j],x[j]+h);xm[j]=max(lo[j],x[j]-h)
            cols.append((constraints(xp)-constraints(xm))/(xp[j]-xm[j]))
        return np.column_stack(cols)
    x0[14]=max(-1000.,evaluate(x0)['task_margins']['gamma'])
    fit=minimize(lambda x:-x[14],x0,jac=lambda x:np.r_[np.zeros(14),-1.],method='SLSQP',bounds=list(zip(lo,hi)),
        constraints=[dict(type='ineq',fun=constraints,jac=jac)],options=dict(maxiter=max_iterations,ftol=1e-7))
    evaluate(fit.x)
    chosen=best if best is not None else best_failed[1]
    return dict(accepted=best is not None,selected=chosen,best_failed=best_failed[1],
        solver_success=bool(fit.success),solver_status=int(fit.status),solver_message=str(fit.message),
        iterations=int(fit.nit),queries=len(cache),seconds=time.perf_counter()-start_time,
        position_budget_m=.0005,normal_tolerance=.12,physical_feasibility_verified=False,
        anchor_q_preserved=q,contact_time_mode='unchanged_outer_discrete_window',
        optimization_variables='q6,qd6,string_uv2,task_margin_gamma; outgoing derived from forward collision')
