"""Experimental time-of-impact split of the existing loaded contact model.

Positive-gap speculative constraints can end a step with a closing velocity.
This candidate resolves a newly crossed foot plane inside the step, then
solves the remaining constrained motion. It is not claimed equivalent to TGS.
The original numerical path is retained exactly when no new foot crosses.
"""
import copy
import numpy as np
import pinocchio as pin

from scripts.audit_sphere_touchdown_predictions_v1 import sphere_gaps
from scripts.predict_speed_limited_spheres_v1 import surface_jacobians
from scripts.frictional_speed_limited_step_v1 import speed_limited_velocity_step


def event_resolved_foot_step(model,interface,state,target,spheres,dt,*,depth=0):
    if not np.isfinite(dt) or not 0<dt<=.01 or depth>8:
        raise ValueError('Bounded positive physical interval required')
    target=np.asarray(target,float)
    if target.shape!=(18,) or not np.isfinite(target).all():raise ValueError('Invalid target')
    q,v=state.q,state.v;gap=sphere_gaps(model,q,spheres)
    snap=model.snapshot(q,v);jac=surface_jacobians(model,q,spheres)
    selection=np.zeros((24,18));selection[model.v_indices,np.arange(18)]=1.

    def velocity_step(duration,mask,normal=None):
        ids=np.flatnonzero(mask)
        requested=(-gap[ids]/np.where(gap[ids]>0,duration,.01) if normal is None else normal)
        result=speed_limited_velocity_step(snap['mass_matrix'],v,snap['nonlinear'],selection,
            interface.kp,interface.kd,target-q[model.q_indices],interface.effort,duration,
            jac[ids].reshape(-1,24),requested,interface.friction,interface.velocity)
        impulse=np.zeros((4,3));impulse[ids]=result['impulse'].reshape(-1,3)
        return result,impulse

    def following(configuration,velocity,impulse):
        result=copy.deepcopy(state);result.q=configuration;result.v=velocity
        result.root_position_world=configuration[:3].copy()
        result.root_rotation_world=pin.Quaternion(configuration[3:7]).matrix()
        result.joint_position=configuration[model.q_indices].copy()
        result.joint_velocity=velocity[model.v_indices].copy()
        result.contacts=impulse[:,2]>1e-8
        return result

    original,original_impulse=velocity_step(dt,np.ones(4,dtype=bool))
    original_q=pin.integrate(model.model,q,dt*original['velocity'])
    # Numerical geometric tolerance only; it does not alter native contacts,
    # physics, force thresholds, or task/clearance acceptance criteria.
    positive=gap>1e-6
    newly_loaded=positive & (original_impulse[:,2]>1e-10)
    fallback=dict(state=following(original_q,original['velocity'],original_impulse),
        impulse=original_impulse,events=[],momentum_residual=original['momentum_residual'],
        original_step_exact=True,subintervals=1)
    if not newly_loaded.any():return fallback
    active=~positive
    free,free_impulse=velocity_step(dt,active)
    free_q=pin.integrate(model.model,q,dt*free['velocity'])
    end_gap=sphere_gaps(model,free_q,spheres)
    crossing=positive & (end_gap<=0.)
    if not crossing.any():return fallback

    # Earliest geometric zero among potential crossings. The physical step
    # is 62.5 us in the native audit; no constant-velocity teleport is used.
    lo,hi=0.,dt
    hit_result,hit_impulse,hit_q=free,free_impulse,free_q
    for _ in range(18):
        mid=(lo+hi)/2
        trial,trial_impulse=velocity_step(mid,active)
        trial_q=pin.integrate(model.model,q,mid*trial['velocity'])
        trial_gap=sphere_gaps(model,trial_q,spheres)
        if np.min(trial_gap[crossing])<=0.:
            hi=mid;hit_result,hit_impulse,hit_q=trial,trial_impulse,trial_q
        else:lo=mid
    hit_gap=sphere_gaps(model,hit_q,spheres)
    hit_feet=np.flatnonzero(positive & (hit_gap<=1e-6)).tolist()
    if not hit_feet:raise ValueError('No actual geometric event after bracketing')
    at_hit=following(hit_q,hit_result['velocity'],hit_impulse)
    remaining=dt-hi
    if remaining<=1e-12:
        # A vanishing remaining duration still carries an inelastic contact
        # impulse. The finite minimum is only used in the implicit drive solve.
        remaining=1e-12
    tail=event_resolved_foot_step(model,interface,at_hit,target,spheres,remaining,depth=depth+1)
    events=[dict(time_from_interval_start_s=hi,feet=hit_feet,
        preimpact_gap_m=hit_gap.tolist(),bracket_width_s=hi-lo)]
    events += [dict(e,time_from_interval_start_s=hi+e['time_from_interval_start_s']) for e in tail['events']]
    return dict(state=tail['state'],impulse=hit_impulse+tail['impulse'],events=events,
        momentum_residual=max(hit_result['momentum_residual'],tail['momentum_residual']),
        original_step_exact=False,subintervals=1+tail['subintervals'])
