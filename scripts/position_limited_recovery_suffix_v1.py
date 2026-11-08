"""Predict a recovery suffix including coupled speed and position-stop reactions.

The caller supplies a policy with the ORIGINAL reference origin. Prediction
starts at the current state, not the reference state, and ends at tick 800.
This primitive is for subsequent sequence optimization; it is not an MPC
solver, native execution, or a guarantee of recursive feasibility.
"""
import copy
import time

import numpy as np

from scripts.predict_position_limited_spheres_v1 import predict_position_limited_spheres as predict_frictional_spheres
from scripts.feedback_tail_rollout_v1 import (
    terminal_measures, terminal_violations, kinematically_stable,
)


def evaluate_suffix(controller, observed_state, policy, spheres, geometry, protocol, *,
                    control_tick, substeps, deadline, preceding_stable_ticks=0, save=True):
    if (type(control_tick) is not int or not 280 <= control_tick < 800
            or type(preceding_stable_ticks) is not int
            or not 0 <= preceding_stable_ticks <= control_tick
            or substeps not in (16, 32, 160) or not np.isfinite(deadline)):
        raise ValueError('Explicit original clock, history and bounded integration required')
    expected = dict(continuous_stable_window_s=.8, maximum_abs_roll_pitch_deg=15.,
        maximum_horizontal_root_speed_m_s=.3, maximum_root_angular_speed_rad_s=1.,
        maximum_abs_joint_velocity_rad_s=1.)
    if any(protocol[k] != value for k, value in expected.items()):
        raise ValueError('Recovery contract changed')
    if controller.settings.terminal_support_margin_m != .01:
        raise ValueError('Original terminal support margin required')
    state = copy.deepcopy(observed_state)
    interface = controller.qp.interface
    previous = np.asarray(state.previous_command_target)
    if (previous.shape != (18,) or not np.isfinite(previous).all()
            or not np.isfinite(state.q).all() or not np.isfinite(state.v).all()):
        raise ValueError('Finite measured state and actual dispatched command required')
    if preceding_stable_ticks and (state.q[2] < .12 or
            not kinematically_stable(terminal_measures(controller, state), protocol)):
        raise ValueError('Stable history contradicts the current state')
    initial_q = state.q.copy(); initial_v = state.v.copy()
    initial_contacts = state.contacts.copy(); initial_command = previous.copy()
    began = time.monotonic(); length = 800-control_tick
    rows = []; failure = None; cost = 0.; completed = 0; end = None
    trailing = longest = preceding_stable_ticks
    for step in range(length):
        if time.monotonic() >= deadline:
            failure = dict(step=step, reason='declared_compute_budget'); break
        tick = control_tick+step
        previous = state.previous_command_target.copy()
        # Global reference clock is never reset at a replanning boundary.
        target, mode = policy.action(80+tick-280, state, previous)
        target = np.asarray(target, float)
        if (target.shape != (18,) or not np.isfinite(target).all()
                or np.any(target < interface.limits[:, 0]-1e-10)
                or np.any(target > interface.limits[:, 1]+1e-10)
                or np.max(abs(target[12:]-previous[12:])) > .01*interface.arm_rate+1e-10):
            failure = dict(step=step, reason='actual_command_interface_violation'); break
        try:
            pred = predict_frictional_spheres(controller.qp.model, interface, state, target,
                spheres, geometry=geometry, substeps=substeps)
        except (ValueError, np.linalg.LinAlgError) as error:
            collision = getattr(geometry, 'last_violation', None)
            failure = dict(step=step,
                reason='authored_hip_mount_constraint' if collision is not None else 'model_solver_rejected',
                error=repr(error))
            if collision is not None: failure['collision'] = collision
            break
        bad = {k:v for k,v in pred['diagnostics'].items() if
            (k == 'nonfoot_ground_violation_m' and v > 1e-6) or
            (k in ['joint_position_violation_rad', 'joint_velocity_violation_rad_s',
                   'motor_torque_violation_nm'] and v > 3e-4)}
        if bad:
            failure = dict(step=step, reason='path_constraints', violations=bad,
                first_ground_violation=pred['first_ground_violation']); break
        state = pred['state']; state.previous_command_target = target.copy(); completed += 1
        end = terminal_measures(controller, state)
        trailing = trailing+1 if kinematically_stable(end, protocol) and state.q[2] >= .12 else 0
        longest = max(longest, trailing)
        cost += .01*(20*(end['roll']**2+end['pitch']**2)+5*end['omega']**2+
            5*end['horizontal']**2+.05*np.sum(state.joint_velocity**2)+.1*np.sum((target-previous)**2))
        if save:
            rows.append(dict(control_tick=tick, mode=mode, target=target.copy(),
                q=state.q.copy(), v=state.v.copy(), contacts=state.contacts.copy()))
    violations = terminal_violations(end, .01) if end else {'no_endpoint':True}
    if trailing < 80: violations['trailing_window_s'] = trailing*.01
    if state.q[2] < .12: violations['root_height_m'] = float(state.q[2])
    accepted = failure is None and completed == length and not violations
    fraction = 0.
    if (failure and failure.get('first_ground_violation') and
            set(failure.get('violations', {})) == {'nonfoot_ground_violation_m'}):
        fraction = failure['first_ground_violation']['micro']/substeps
    return dict(accepted=accepted, completed_model_steps=completed, required_model_steps=length,
        rank=[0 if accepted else 1, length-completed-fraction,
              0 if accepted else max(0, 80-trailing), cost/max(1, completed)],
        failure=failure, terminal=end, terminal_violations=violations,
        trailing_model_window_s=trailing*.01, longest_model_window_s=longest*.01,
        cost=float(cost), substeps=substeps, wall_seconds=time.monotonic()-began,
        trajectory=rows if save else None, control_tick=control_tick,
        original_episode_end_tick=800, preceding_stable_ticks=preceding_stable_ticks,
        initial_q=initial_q, initial_v=initial_v, initial_contacts=initial_contacts,
        initial_previous_command=initial_command, native_execution=False, valid_expert_labels=False)
