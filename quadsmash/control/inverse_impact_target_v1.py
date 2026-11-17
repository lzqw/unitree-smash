"""Bounded analytic task-target inversion; never a robot-feasibility certificate."""
import numpy as np
from scipy.optimize import least_squares
from .initial_intercept import propagate_ball_state


class UnreachableImpactTarget(ValueError):
    """One contact time has no verified analytic task target."""


def inverse_impact_target(impact_model, position, incoming, contact_local):
    p=np.asarray(position,dtype=float);v=np.asarray(incoming,dtype=float)
    if p.shape!=(3,) or v.shape!=(3,) or not np.isfinite([p,v]).all():
        raise ValueError('Finite world position and incoming velocity required')
    court=impact_model.court;physics=impact_model.physics
    # One deterministic interior target, no case lookup or parameter scan.
    landing=np.array([court['target_x_limits_m'][1]-.1,0.,physics['collision_radius_m']])
    vz=-1.01
    def residual(x):
        end,_=propagate_ball_state(p,[x[0],x[1],vz],x[2],physics)
        return (end-landing)/.01
    fit=least_squares(residual,[8.,0.,.4],bounds=([.1,-20.,.05],[30.,20.,2.]),
                      max_nfev=60,ftol=1e-8,xtol=1e-8,gtol=1e-8)
    desired=np.array([fit.x[0],fit.x[1],vz])
    delta=desired-v;size=np.linalg.norm(delta)
    if size<1e-9:return dict(accepted=False,rejection_reason='zero_impact_change')
    normal=delta/size
    # Match incoming tangential velocity: relative tangential speed and hence
    # friction impulse are zero. Forward model independently verifies this.
    racket_velocity=v+size/(1+physics['restitution'])*normal
    surface=p-physics['collision_radius_m']*normal
    outcome=impact_model.evaluate(p,v,surface,racket_velocity,normal,contact_local)
    valid=bool(fit.success and np.linalg.norm(residual(fit.x))<=.1 and
               outcome.get('legal_return_predicted') and outcome.get('attack5_predicted') and
               outcome.get('target_landing_predicted'))
    return dict(accepted=valid,rejection_reason=None if valid else 'predicted_outgoing_unreachable',
        normal_world=normal.tolist(),racket_velocity_world_m_s=racket_velocity.tolist(),
        incoming_velocity_world_m_s=v.tolist(),desired_outgoing_velocity_world_m_s=desired.tolist(),
        target_landing_world_m=landing.tolist(),flight_time_s=float(fit.x[2]),
        optimizer_nfev=int(fit.nfev),forward_outcome=outcome,
        robot_feasibility_verified=False,model='infinite_mass_zero_relative_tangent_seed')


def optimize_reachable_impact(impact_model,position,incoming,contact_local,kinematics,
                              q0,lower,upper,speed,base_velocity_fn=None,max_nfev=24,
                              allow_contact_motion=True):
    """Joint continuous outgoing/arm/contact optimization, one deterministic seed.

    `kinematics(q, local)` returns the ball-centre proxy and Jacobians in WORLD.
    The UV box is one actual authored string strip, never its convex hull.
    Final analytic and robot gates are separate; neither is inferred from cost.
    """
    from scipy.optimize import lsq_linear
    import time
    started=time.perf_counter()
    p=np.asarray(position);incoming=np.asarray(incoming);local=np.asarray(contact_local)
    seed=inverse_impact_target(impact_model,p,incoming,local)
    if 'desired_outgoing_velocity_world_m_s' not in seed:return seed
    strips=impact_model.strips
    strip=min(strips,key=lambda s:np.linalg.norm(np.maximum(np.abs(local-np.asarray(s['center']))-.5*np.asarray(s['size']),0)))
    center=np.asarray(strip['center']);half=.5*np.asarray(strip['size'])
    uvlow=center[1:]-half[1:]+1e-7;uvhigh=center[1:]+half[1:]-1e-7
    if not allow_contact_motion:
        uvlow=local[1:]-1e-10;uvhigh=local[1:]+1e-10
    q0=np.clip(q0,lower,upper)
    initial=np.r_[q0,seed['desired_outgoing_velocity_world_m_s'],seed['flight_time_s'],np.clip(local[1:],uvlow,uvhigh)]
    lo=np.r_[lower,.1,-20.,-12.,.05,uvlow];hi=np.r_[upper,30.,20.,-1.,2.,uvhigh]
    initial=np.clip(initial,lo+1e-12,hi-1e-12)
    court=impact_model.court;physics=impact_model.physics
    def evaluate(x):
        q=x[:6];out=x[6:9];t=x[9];uv=x[10:12];contact=np.r_[local[0],uv]
        delta=out-incoming;n=delta/max(np.linalg.norm(delta),1e-12)
        vr=incoming+delta/(1+physics['restitution'])
        point,normal,jp,jn=kinematics(q,contact)
        base=np.zeros(3) if base_velocity_fn is None else base_velocity_fn(point)
        rate=lsq_linear(np.vstack((jp,.04*np.eye(6))),np.r_[vr-base,np.zeros(6)],bounds=(-speed,speed),max_iter=30)
        landing,_=propagate_ball_state(p,out,t,physics)
        outcome=impact_model.evaluate(p,incoming,p-physics['collision_radius_m']*n,vr,n,contact)
        return contact,n,vr,point,normal,jp,base+jp@rate.x,landing,outcome,rate.x
    def residual(x):
        contact,n,vr,point,normal,jp,actual,landing,outcome,rate=evaluate(x)
        # All terms normalized; penalties guide search, never replace hard gates.
        cx=sum(court['target_x_limits_m'])/2
        xlo,xhi=court['target_x_limits_m'];ymax=court['target_y_half_width_m']
        clearance=outcome.get('net_clearance');clearance=-1. if clearance is None else clearance
        return np.r_[(point-p)/.035,(normal-n)/.12,(actual-vr)/.75,
            (landing[2]-physics['collision_radius_m'])/.01,
            max(xlo+.1-landing[0],0)/.1,max(landing[0]-(xhi-.1),0)/.1,
            max(abs(landing[1])-(ymax-.1),0)/.1,max(.05-clearance,0)/.05,
            max(5.1-np.linalg.norm(x[6:9]),0)/.5,max(x[8]+1.1,0)/.1,
            .05*(landing[0]-cx),.05*landing[1],.02*(x[:6]-q0)]
    # Absolute steps remain meaningful at zero-angle joints with float32 FK.
    # A relative diff_step falls back to machine epsilon at exactly zero.
    def jacobian(x):
        steps=np.r_[np.full(6,1e-4),np.full(3,1e-3),1e-4,np.full(2,1e-5)]
        columns=[]
        for j,h in enumerate(steps):
            plus=x.copy();minus=x.copy()
            plus[j]=min(hi[j],x[j]+h);minus[j]=max(lo[j],x[j]-h)
            columns.append((residual(plus)-residual(minus))/(plus[j]-minus[j]))
        return np.column_stack(columns)
    fit=least_squares(residual,initial,jac=jacobian,bounds=(lo,hi),max_nfev=max_nfev)
    contact,n,vr,point,normal,jp,actual,landing,outcome,rate=evaluate(fit.x)
    analytic=bool(outcome.get('legal_return_predicted') and outcome.get('attack5_predicted') and outcome.get('target_landing_predicted'))
    ep=float(np.linalg.norm(point-p));en=float(np.linalg.norm(normal-n));ev=float(np.linalg.norm(actual-vr))
    return dict(accepted=analytic,accepted_semantics='analytic_ball_side_only',
        analytic_target_valid=analytic,robot_contact_kinematic_valid=bool(ep<=.003 and en<=.12 and ev<=.75),
        rejection_reason=None if analytic else 'predicted_outgoing_unreachable',
        normal_world=n.tolist(),racket_velocity_world_m_s=vr.tolist(),contact_point_local=contact.tolist(),
        desired_outgoing_velocity_world_m_s=fit.x[6:9].tolist(),forward_outcome=outcome,
        original_target=seed,optimized_q_rad=fit.x[:6].tolist(),optimized_qd_rad_s=rate.tolist(),
        position_error_m=ep,normal_vector_error=en,velocity_error_m_s=ev,
        kinematic_feasible=bool(ep<=.035 and en<=.12 and ev<=.75),
        robot_feasibility_verified=False,solver_success=bool(fit.success),solver_status=int(fit.status),
        optimizer_nfev=int(fit.nfev),seconds=time.perf_counter()-started,
        contact_box_lower=uvlow.tolist(),contact_box_upper=uvhigh.tolist(),
        model='bounded_reachable_infinite_mass_seed')
