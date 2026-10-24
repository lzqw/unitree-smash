"""Optional arm-search coordinates and bounded causal preparation corrections."""
import numpy as np
import torch
from scipy.optimize import lsq_linear
from .full_racket_contact_curriculum_v4 import measured_recovery_leg_crossovers

VELOCITY_PARAMETER_NAMES = [f'arm_joint_{i}_velocity_seed_rad_s' for i in range(1,7)]


def normal_preserving_position_inverse(position_jacobian,normal_jacobian):
    position_jacobian,normal_jacobian=[np.asarray(x,dtype=float) for x in (position_jacobian,normal_jacobian)]
    if position_jacobian.shape!=(3,6) or normal_jacobian.shape!=(3,6):
        raise ValueError('Require two3x6 kinematic Jacobians')
    if not np.isfinite(position_jacobian).all() or not np.isfinite(normal_jacobian).all():
        raise ValueError('Finite kinematic Jacobians required')
    task=np.r_[position_jacobian,.3*normal_jacobian]
    return (task.T@np.linalg.inv(task@task.T+.04**2*np.eye(6)))[:,:3]


def extend_arm_velocity_parameters(parameters, seed_velocity):
    if parameters.ndim != 1 or len(parameters) not in (26,32) or not torch.isfinite(parameters).all():
        raise ValueError('Expected a finite26/32-vector')
    if len(parameters) == 32:
        if (parameters[26:32].abs()>5.).any():
            raise ValueError('Velocity seeds exceed original5rad/s bounds')
        return parameters.clone()
    velocity = torch.as_tensor(seed_velocity,dtype=parameters.dtype,device=parameters.device)
    if velocity.shape != (6,) or not torch.isfinite(velocity).all() or (velocity.abs()>5.).any():
        raise ValueError('Six finite velocity seeds within original5rad/s required')
    return torch.cat((parameters,velocity))


def arm_velocity(parameters, legacy_seed):
    if parameters.shape[1] == 32:
        return parameters[:,26:32] * parameters[:,7:8]
    if parameters.shape[1] > 26:
        raise ValueError('Unknown arm parameter layout')
    return legacy_seed[None] * parameters[:,7:8]


def arm_velocity_grid(reference, bank, lower, upper):
    reference, bank, lower, upper = [np.asarray(x,dtype=float) for x in (reference,bank,lower,upper)]
    if reference.shape!=(32,) or bank.shape!=(4,6) or lower.shape!=(32,) or upper.shape!=(32,):
        raise ValueError('Require32-parameter reference and four6D velocity proposals')
    if not np.isfinite(bank).all() or (np.abs(bank)>5.).any():
        raise ValueError('Velocity bank exceeds original5rad/s seed bounds')
    result = []
    for velocity in bank:
        for lead in [-.03,0.,.03]:
            candidate = reference.copy()
            candidate[14] = np.clip(reference[14]+lead,lower[14],upper[14])
            candidate[26:32] = velocity
            result.append(candidate)
    return np.asarray(result)


def interception_velocity_grid(reference, bank, lower, upper):
    """Probe three public interception planes, with arm timing and velocity.

    Planes are policy choices on the unchanged TRAIN trajectories. They do
    not move the shuttle, net, robot, or any task success threshold.
    """
    grid = arm_velocity_grid(reference, bank, lower, upper)
    result = []
    for plane in [1.10, 1.25, 1.40]:
        candidates = grid.copy()
        candidates[:,24] = np.clip(plane, lower[24], upper[24])
        result.append(candidates)
    return np.concatenate(result)


def body_crossovers_preserving_velocity(reference, recovered, lower, upper):
    reference, lower, upper = [np.asarray(x,dtype=float) for x in (reference,lower,upper)]
    if reference.shape not in [(26,),(32,)] or lower.shape!=reference.shape or upper.shape!=reference.shape:
        raise ValueError('Require matching26/32 parameter bounds')
    body = measured_recovery_leg_crossovers(reference[:26],recovered,lower[:26],upper[:26])
    result = np.broadcast_to(reference,(len(body),len(reference))).copy()
    result[:,:26] = body
    return result


def bounded_prepose_step(delta_body,position_jacobian,normal_jacobian,lower,upper):
    lower,upper=np.asarray(lower),np.asarray(upper)
    if lower.shape!=(6,) or upper.shape!=(6,) or not np.isfinite(lower).all() or not np.isfinite(upper).all() or np.any(upper<=lower):
        raise ValueError('Require finite strictly ordered six-joint correction bounds')
    inverse=normal_preserving_position_inverse(position_jacobian,normal_jacobian)
    unconstrained=inverse@delta_body
    if np.all(unconstrained>=lower) and np.all(unconstrained<=upper):
        return unconstrained
    matrix=np.r_[position_jacobian,.3*normal_jacobian,.04*np.eye(6)]
    target=np.r_[delta_body,np.zeros(9)]
    fit=lsq_linear(matrix,target,bounds=(lower,upper),tol=1e-10,max_iter=200)
    if not fit.success:raise RuntimeError('Bounded causal preparation solve failed')
    return fit.x


def causal_prepose_delta(target_delta_world,public,position_jacobians,normal_jacobians,valid,gain,lower,upper):
    """Use a current public forecast and world-axis observations, never case IDs."""
    if not 0. <= gain <= 1.:
        raise ValueError('Causal preparation gain must be in[0,1]')
    target_delta_world = np.asarray(target_delta_world,dtype=float)
    public,valid=np.asarray(public),np.asarray(valid,dtype=bool)
    position_jacobians,normal_jacobians,lower,upper=[np.asarray(x) for x in (position_jacobians,normal_jacobians,lower,upper)]
    count=len(public)
    if public.ndim!=2 or target_delta_world.shape!=(count,3) or public.shape[1]!=794 or valid.shape!=(count,):
        raise ValueError('Expected N3 targets and N794 public observations')
    if position_jacobians.shape!=(count,3,6) or normal_jacobians.shape!=(count,3,6) or lower.shape!=(count,6) or upper.shape!=(count,6):
        raise ValueError('Expected per-lane kinematic Jacobians and six-joint bounds')
    if gain == 0.:
        return np.zeros((len(public),6))
    forward, up = public[:,763:766], -public[:,6:9]
    left = np.cross(up,forward)
    ready = valid & np.isfinite(target_delta_world).all(1)
    ready &= np.isfinite(forward).all(1) & np.isfinite(up).all(1)
    ready &= (np.linalg.norm(forward,axis=1)>.5) & (np.linalg.norm(up,axis=1)>.5)
    ready &= np.linalg.norm(left,axis=1)>.2
    result = np.zeros((len(public),6))
    if ready.any():
        delta = target_delta_world[ready]
        body = delta[:,:1]*forward[ready] + delta[:,1:2]*left[ready] + delta[:,2:3]*up[ready]
        for index,body_delta in zip(np.flatnonzero(ready),body):
            result[index]=bounded_prepose_step(gain*body_delta,position_jacobians[index],normal_jacobians[index],lower[index],upper[index])
    return result
