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
