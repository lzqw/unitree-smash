"""Check a short command sequence followed by an explicit feedback continuation.

The short-prefix endpoint is diagnostic, not a rest constraint. All original
path limits remain; the extended endpoint retains the old MPC rest/support
requirements and must follow a full trailing recovery-kinematics window.
This finite model check is neither a native success nor an invariant-set proof.
"""
import copy
import time
import numpy as np

from scripts.predict_frictional_spheres_v1 import predict_frictional_spheres


def terminal_measures(controller, state):
    rotation = state.root_rotation_world
    linear = rotation @ state.v[:3]
    margin = float(controller.support_margin(state))
    return dict(roll=float(np.arctan2(rotation[2, 1], rotation[2, 2])),
        pitch=float(np.arcsin(np.clip(-rotation[2, 0], -1., 1.))),
        horizontal=float(np.linalg.norm(linear[:2])), vertical_speed=float(abs(linear[2])),
        omega=float(np.linalg.norm(state.v[3:6])),
        joint_speed=float(np.max(abs(state.joint_velocity))),
        support_feet=int(state.contacts.sum()),
        support_margin_m=margin if np.isfinite(margin) else None)


def terminal_violations(values, support_margin):
    """Same numerical endpoint thresholds as the existing nonlinear MPC."""
    result = {}
    for name, bound in [('roll', np.deg2rad(15)), ('pitch', np.deg2rad(15)),
                        ('horizontal', .3), ('vertical_speed', .1), ('omega', 1.), ('joint_speed', 1.)]:
        violation = abs(values[name])-bound
        if not np.isfinite(violation) or violation > 1e-10:
            result[name] = float(violation) if np.isfinite(violation) else 'nonfinite'
    margin = values['support_margin_m']
    if margin is None or margin < support_margin-1e-12:
        result['support_margin_m'] = None if margin is None else support_margin-margin
    if values['support_feet'] < 1:
        result['support_feet'] = 'no_support'
    return result


def kinematically_stable(values, protocol):
    return bool(values['support_feet'] > 0
        and max(abs(values['roll']), abs(values['pitch'])) <= np.deg2rad(protocol['maximum_abs_roll_pitch_deg'])
        and values['horizontal'] <= protocol['maximum_horizontal_root_speed_m_s']
        and values['omega'] <= protocol['maximum_root_angular_speed_rad_s']
        and values['joint_speed'] <= protocol['maximum_abs_joint_velocity_rad_s'])


def evaluate_feedback_tail(controller, initial, feedback, spheres, geometry, protocol, *,
                           control_tick, prefix_commands=None, prefix_steps=80,
                           substeps=16, time_limit_s=180.):
    if (type(control_tick) is not int or not 460 <= control_tick < 800
            or prefix_steps != 80 or substeps not in (16, 160) or not 0 < time_limit_s <= 600):
        raise ValueError('Bounded original-phase 80-step prefix and explicit integration required')
    if controller.settings.terminal_support_margin_m != .01:
        raise ValueError('Original terminal support margin required')
    expected_protocol = dict(continuous_stable_window_s=.8, maximum_abs_roll_pitch_deg=15.,
        maximum_horizontal_root_speed_m_s=.3, maximum_root_angular_speed_rad_s=1.,
        maximum_abs_joint_velocity_rad_s=1.)
    if any(protocol[k] != value for k, value in expected_protocol.items()):
        raise ValueError('Task recovery thresholds changed')
    if prefix_commands is not None:
        prefix_commands = np.asarray(prefix_commands, float)
        if prefix_commands.shape != (80, 18) or not np.isfinite(prefix_commands).all():
            raise ValueError('Complete finite absolute prefix required')
    model, interface = controller.qp.model, controller.qp.interface
    state = copy.deepcopy(initial)
    if np.asarray(state.previous_command_target).shape != (18,) or not np.isfinite(state.previous_command_target).all():
        raise ValueError('Actual command history required')
    # Preserve the original episode and always check at least 0.8 seconds
    # beyond the short prefix. Future-only prediction does not extend scoring.
    length = max(800-control_tick, prefix_steps+80)
    began = time.monotonic(); commands = []; states = []; records = []
    failure = None; prefix_terminal = None; final = None; trailing = longest = 0
    path_maxima = {}; cost = 0.
    for step in range(length):
        if time.monotonic()-began > time_limit_s:
            failure = dict(step=step, reason='continuation_budget_exceeded'); break
        previous = state.previous_command_target.copy()
        if step < prefix_steps and prefix_commands is not None:
            target = prefix_commands[step].copy(); mode = 'supplied_short_prefix'
        else:
            target, mode = feedback.action(min(control_tick+step-460, 339), state, previous)
        target = np.asarray(target, float)
        if (target.shape != (18,) or not np.isfinite(target).all()
                or np.any(target < interface.limits[:, 0]-1e-10)
                or np.any(target > interface.limits[:, 1]+1e-10)
                or np.max(abs(target[12:]-previous[12:])) > .01*interface.arm_rate+1e-10):
            failure = dict(step=step, reason='actual_command_interface_violation'); break
        try:
            prediction = predict_frictional_spheres(model, interface, state, target, spheres,
                                                   geometry=geometry, substeps=substeps)
        except (ValueError, np.linalg.LinAlgError) as error:
            failure = dict(step=step, reason='prediction_failed', error=repr(error)); break
        diagnostics = prediction['diagnostics']
        for name, value in diagnostics.items():
            path_maxima[name] = max(path_maxima.get(name, 0.), value)
        bad = {k: v for k, v in diagnostics.items() if
               (k == 'nonfoot_ground_violation_m' and v > 1e-6) or
               (k in ['joint_position_violation_rad', 'joint_velocity_violation_rad_s', 'motor_torque_violation_nm'] and v > 3e-4)}
        if bad:
            failure = dict(step=step, reason='path_constraints', violations=bad); break
        state = prediction['state']; state.previous_command_target = target.copy()
        final = terminal_measures(controller, state)
        trailing = trailing+1 if kinematically_stable(final, protocol) else 0
        longest = max(longest, trailing)
        cost += (20*(final['roll']**2+final['pitch']**2)+5*final['omega']**2+
                 5*final['horizontal']**2+.05*np.sum(state.joint_velocity**2)+.1*np.sum((target-previous)**2))*.01
        commands.append(target.copy())
        states.append(dict(q=state.q.copy(), v=state.v.copy(), contacts=state.contacts.copy()))
        records.append(dict(step=step, mode=mode, diagnostics=diagnostics, terminal=final))
        if step == prefix_steps-1:
            prefix_terminal = dict(values=final, old_endpoint_violations=terminal_violations(final, .01))
    violations = terminal_violations(final, .01) if final is not None else {'no_endpoint': True}
    if trailing < 80:
        violations['trailing_kinematic_window_s'] = trailing*.01
    accepted = failure is None and len(states) == length and not violations
    return dict(accepted=accepted, failure=failure, commands=np.asarray(commands) if accepted else None,
        predicted_states=states, records=records, completed_model_steps=len(states), required_model_steps=length,
        prediction_end_control_tick=control_tick+length, original_episode_end_control_tick=800,
        prefix_terminal=prefix_terminal, terminal=final, terminal_violations=violations,
        trailing_model_kinematic_window_s=trailing*.01, longest_model_kinematic_window_s=longest*.01,
        path_diagnostics=path_maxima, cost=float(cost), wall_seconds=time.monotonic()-began,
        integration_substeps=substeps, finite_model_continuation_only=True,
        formal_recovery_success=False, native_execution=False, valid_expert_labels=False)
