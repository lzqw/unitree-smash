"""Propose a two-stage unloaded-foot placement, then four-foot preload.

IK and future force balance are only seed generators. They grant no current
support to an airborne foot. The unchanged frictional rollout must validate
the whole transition, including contact establishment and actuator limits.
"""
import copy
import numpy as np
import pinocchio as pin
from scripts.frictional_static_reference_v1 import sphere_static_reference


def foot_center(model, q, sphere):
    pin.computeJointJacobians(model.model, model.data, q)
    pin.updateFramePlacements(model.model, model.data)
    fid = model.body_frames[sphere['body']]
    pose = model.data.oMf[fid]
    offset = pose.rotation@np.asarray(sphere['center_body_m'])
    jac = pin.getFrameJacobian(model.model, model.data, fid, pin.ReferenceFrame.LOCAL_WORLD_ALIGNED)
    return pose.translation+offset, jac[:3]-pin.skew(offset)@jac[3:]


def propose_recontact(controller, state, spheres, horizon=40):
    model = controller.qp.model
    interface = controller.qp.interface
    if horizon != 40:
        raise ValueError('Seed timing is declared for 0.4 s')
    missing = np.flatnonzero(~state.contacts)
    if not len(missing) or not state.contacts.any():
        return dict(seeds=[], reason='Requires both current support and an unloaded foot')
    present = sphere_static_reference(controller, state, spheres)
    if not present['solved']:
        return dict(seeds=[], reason='No current static preload proposal')
    goal = copy.deepcopy(state)
    diagnostics = []
    for foot in missing:
        indices = np.array([foot, foot+4, foot+8])
        original, _ = foot_center(model, goal.q, spheres[foot])
        target = original.copy()
        target[2] = spheres[foot]['radius_m']
        for _ in range(60):
            point, jac = foot_center(model, goal.q, spheres[foot])
            error = target-point
            if np.linalg.norm(error) < 1e-6:
                break
            local = jac[:, model.v_indices[indices]]
            delta = local.T@np.linalg.solve(local@local.T+1e-5*np.eye(3), error)
            delta *= min(1., .1/max(np.max(abs(delta)), 1e-12))
            goal.q[model.q_indices[indices]] = np.clip(goal.q[model.q_indices[indices]]+delta,
                interface.limits[indices, 0], interface.limits[indices, 1])
        point, _ = foot_center(model, goal.q, spheres[foot])
        error = float(np.linalg.norm(target-point))
        diagnostics.append(dict(foot=spheres[foot]['body'], initial_gap_m=float(original[2]-spheres[foot]['radius_m']),
            target_world=target.tolist(), ik_error_m=error))
        if error > 1e-5:
            return dict(seeds=[], reason='Foot plane target unreachable by bounded IK', diagnostics=diagnostics)
    goal.joint_position = goal.q[model.q_indices].copy()
    goal.v[:] = 0.
    goal.joint_velocity[:] = 0.
    # This is explicitly a hypothetical future mode, used only for a seed.
    goal.contacts[:] = True
    future = sphere_static_reference(controller, goal, spheres)
    if not future['solved']:
        return dict(seeds=[], reason='No four-foot future equilibrium', diagnostics=diagnostics)
    phase1 = np.asarray(present['target']).copy()
    for foot in missing:
        indices = np.array([foot, foot+4, foot+8])
        phase1[indices] = goal.joint_position[indices]
    phase2 = np.asarray(future['target'])
    seeds = []
    for arrival in [6, 12, 20]:
        commands = []
        previous = state.previous_command_target.copy()
        for tick in range(horizon):
            if tick < arrival:
                weight = (tick+1)/arrival
                command = (1-weight)*state.previous_command_target+weight*phase1
            else:
                weight = min(1., (tick-arrival+1)/10)
                command = (1-weight)*phase1+weight*phase2
            command = np.clip(command, interface.limits[:, 0], interface.limits[:, 1])
            command[12:] = np.clip(command[12:], previous[12:]-.01*interface.arm_rate,
                previous[12:]+.01*interface.arm_rate)
            commands.append(command)
            previous = command
        seeds.append(dict(name=f'recontact_then_preload_{arrival*10}ms', commands=np.asarray(commands)))
    return dict(seeds=seeds, diagnostics=diagnostics, future_mode_only=True,
        current_contact_state_unchanged=True, valid_teacher=False)
