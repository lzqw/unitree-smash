"""Causal recovery eligibility and original native command coordinates.

These utilities define a supervised command dataset, not a PPO rollout. Outcome
labels select supervision after execution; eligibility itself never sees them.
"""
import numpy as np


def recovery_eligibility(public):
    public=np.asarray(public)
    if public.ndim!=2 or public.shape[1]!=794 or not len(public) or not np.isfinite(public).all():
        raise ValueError('Finite complete public observation sequence required')
    clock=public[:,775]
    if abs(float(clock[0]))>1e-6 or np.any(abs(np.diff(clock)-.01)>2e-6):
        raise ValueError('A single contiguous ground-start episode is required')
    active=np.zeros(len(public),dtype=bool);elapsed=np.zeros(len(public),dtype=np.float32)
    observed_hit=False;onset=None
    for tick,obs in enumerate(public):
        if onset is None and observed_hit and obs[774]>.5:
            onset=tick
        if obs[773]>.5:observed_hit=True
        if onset is not None:
            active[tick]=True;elapsed[tick]=.01*(tick-onset)
    return active,elapsed


def native_commands(actions,limits):
    actions=np.asarray(actions,dtype=np.float64);limits=np.asarray(limits,dtype=np.float64)
    if (actions.ndim!=2 or actions.shape[1]!=18 or limits.shape!=(18,2) or
            not np.isfinite(actions).all() or not np.isfinite(limits).all() or
            np.any(limits[:,0]>=limits[:,1]) or np.max(abs(actions),initial=0.)>1.+1e-6):
        raise ValueError('Require recorded normalized native commands and original joint bounds')
    normalized=np.clip(actions,-1.,1.)
    targets=limits[:,0]+.5*(normalized+1.)*(limits[:,1]-limits[:,0])
    return normalized.astype(np.float32),targets


def previous_commands(targets):
    targets=np.asarray(targets,dtype=np.float64)
    if targets.ndim!=2 or targets.shape[1]!=18 or not np.isfinite(targets).all():
        raise ValueError('Finite native joint target sequence required')
    previous=np.full_like(targets,np.nan)
    previous[1:]=targets[:-1]
    known=np.ones(len(targets),dtype=bool);known[:1]=False
    return previous,known


def compose_native_recovery_delta(base,delta,previous,active,limits,arm_rate=10.,dt=.01):
    """Represent total native targets while retaining original bounds and slew.

    Delta is in [-1,1] and spans twice the normalized position range, so every
    admissible native target is expressible from any base command. This is a
    NEW software action coordinate, not the legacy +/-0.4 rad residual. This
    NumPy reference is for dataset validation, not a deployed learned policy.
    """
    base=np.asarray(base,dtype=np.float64);delta=np.asarray(delta,dtype=np.float64)
    previous=np.asarray(previous,dtype=np.float64);active=np.asarray(active,dtype=bool)
    limits=np.asarray(limits,dtype=np.float64)
    if (base.ndim!=2 or base.shape[1]!=18 or delta.shape!=base.shape or previous.shape!=base.shape or
            active.shape!=(len(base),) or limits.shape!=(18,2) or
            not all(np.isfinite(x).all() for x in (base,delta,previous,limits)) or
            np.max(abs(base),initial=0.)>1.+1e-6 or np.max(abs(previous),initial=0.)>1.+1e-6 or
            np.max(abs(delta),initial=0.)>1.+1e-6 or np.any(limits[:,0]>=limits[:,1]) or
            not np.isfinite(arm_rate) or arm_rate<=0 or not np.isfinite(dt) or dt<=0):
        raise ValueError('Valid native command/delta/history and original limits required')
    candidate=np.clip(base+2.*delta,-1.,1.)
    normalized_step=2.*arm_rate*dt/(limits[12:,1]-limits[12:,0])
    candidate[:,12:]=np.clip(candidate[:,12:],previous[:,12:]-normalized_step,
                            previous[:,12:]+normalized_step)
    candidate=np.clip(candidate,-1.,1.)
    return np.where(active[:,None],candidate,base)
