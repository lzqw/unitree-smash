"""Read-only actuator contract and causal inputs for offline robot identification."""
import torch

from .supermimic_physical_observation_v1 import JOINT_NAMES


def actuator_contract(env, task_contract):
    ids = list(map(int, env.joint_ids))
    names = [env.robot.joint_names[j] for j in ids]
    if tuple(names) != JOINT_NAMES:
        raise ValueError('Actuator contract and WM joint order differ')
    # A randomized scene can have more than one drive profile. Keep its exact lane mapping.
    columns = [getattr(env.robot.data, name)[:, ids].reshape(env.num_envs, -1)
               for name in ('joint_stiffness', 'joint_damping', 'joint_effort_limits',
                            'joint_vel_limits', 'joint_pos_limits')]
    joined = torch.cat(columns, -1).detach().cpu()
    if not torch.isfinite(joined).all():
        raise ValueError('Nonfinite limits in native drive contract')
    profiles, lane_index = torch.unique(joined, dim=0, return_inverse=True)
    width = len(names)
    entries = []
    for row in profiles:
        kp, kd, effort, velocity, limits = row.split([width]*4+[2*width])
        entries.append(dict(kp=kp.tolist(), kd=kd.tolist(), effort_limit=effort.tolist(),
                            velocity_limit=velocity.tolist(), position_limits=limits.reshape(width, 2).tolist()))
    return dict(schema='native-joint-drive-v1', asset_sha256=task_contract['asset']['sha256'],
                joint_names=names, joint_indices=ids, all_joint_names=list(env.robot.joint_names),
                control_dt_s=float(env.step_dt), physics_dt_s=float(env.cfg.sim.dt),
                hold_physics_steps=int(env.cfg.decimation), lane_profile=lane_index.tolist(),
                profiles=entries, actuator_types={k:type(v).__name__ for k,v in env.robot.actuators.items()},
                torque_source='articulation.data.applied_torque; implicit actuator estimate',
                command_boundary='final joint target assigned before physics integration')


def drive_features(state, target, previous_target, previous_torque, hold_age, profile):
    """All inputs are known at the control boundary. Current-step torque is a label."""
    q, qd = state[...,13:31], state[...,31:49]
    error = target - q
    p = profile['kp'] * error
    d = profile['kd'] * qd
    limit = profile['effort_limit']
    demand = p-d
    torque = torch.maximum(torch.minimum(demand, limit), -limit)
    features = torch.stack((error, qd/profile['velocity_limit'], p/limit, d/limit,
                            torque/limit, target-previous_target, previous_torque/limit,
                            (demand.abs() >= limit).to(q), torch.log1p(hold_age.to(q))), dim=-1)
    return torque, features
