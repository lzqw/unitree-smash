"""Fail-closed prerequisites for physical imagination, not a learner adapter.

Passing this inventory is necessary, not sufficient: execution replay,
reward parity and native action sensitivity still require separate checks.
No missing state is synthesized from zeros or a terminal success label.
"""
import torch
from quadsmash.tasks.stand.jump_metrics import FLOAT_FIELDS, BOOL_FIELDS


AUXILIARY_SCHEMA = 'native-task-auxiliary-state-v1'
TIMING_FIELDS = ('_sim_step_counter', '_observation_physics_frame',
                 '_jump_physics_frame', '_jump_metric_previous_frame',
                 '_last_physics_frame', '_jump_metric_stride')
ENVIRONMENT_FIELDS = (
    '_observed_air_time', 'jump_clock', 'stable_recovery', 'clear_flight',
    'launch_done', 'launch_frame', 'launch_signals_ready', 'ball_clock',
    'previous_ball_position', 'previous_ball_velocity', 'previous_tool_position',
    'previous_tool_quaternion', 'pending_sweep_age', 'sweep_ready',
    'first_tool_normal', 'first_tool_velocity', 'first_string_time',
    'action_delta', 'last_action')
RECOVERY_FIELDS = ('flight_id', 'contact_flight_id', 'bound',
    'qualified_landing', 'stable_seen', 'last_takeoff', 'rally_contact_time',
    'string_time', 'takeoff_time', 'landing_time', 'rise', 'clear_time',
    'stable_run', 'longest_stable')


def _tensor_fields(obj, names):
    # Required fields deliberately raise instead of substituting defaults.
    return {name: getattr(obj, name).detach().clone() for name in names}


@torch.no_grad()
def capture_auxiliary_state(env):
    """Read-only capture of native task memory and exact reward inputs.

    Does not claim to capture inaccessible PhysX contact solver caches.
    This is a control-boundary snapshot, not a substep event trace.
    """
    if env.contact_recovery is None:
        raise ValueError('Physical task capture requires declared contact-flight recovery')
    d = env.robot.data
    return dict(schema=AUXILIARY_SCHEMA,
        jump=_tensor_fields(env, FLOAT_FIELDS + BOOL_FIELDS),
        environment=_tensor_fields(env, ENVIRONMENT_FIELDS),
        timing={name: int(getattr(env, name)) for name in TIMING_FIELDS},
        contact_recovery=_tensor_fields(env.contact_recovery, RECOVERY_FIELDS),
        contact_episodes=_tensor_fields(env.contact_events, ('active', 'count', 'clear_frames')),
        reward_inputs=dict(
            safety_body_height=d.body_pos_w[:, env.safety_body_ids, 2].detach().clone(),
            projected_gravity_b=d.projected_gravity_b.detach().clone(),
            root_com_lin_vel_w=d.root_com_lin_vel_w.detach().clone(),
            root_com_ang_vel_w=d.root_com_ang_vel_w.detach().clone(),
            applied_torque_all=d.applied_torque.detach().clone()),
        command_limits=env._sm_residual_limits.detach().clone(),
        not_captured=['PhysX contact solver cache', 'within-tick event sequence'])


def physical_start_issues(state, execution):
    """List concrete missing evidence, without certifying a complete model."""
    issues = []
    auxiliary = state.get('auxiliary', {})
    required = {
        'contact_recovery': RECOVERY_FIELDS,
        'jump': FLOAT_FIELDS + BOOL_FIELDS,
        'environment': ENVIRONMENT_FIELDS,
        'timing': TIMING_FIELDS,
        'contact_episodes': ('active', 'count', 'clear_frames'),
        'reward_inputs': ('safety_body_height', 'projected_gravity_b',
                         'root_com_lin_vel_w', 'root_com_ang_vel_w', 'applied_torque_all'),
    }
    for group, names in required.items():
        missing = [name for name in names if name not in auxiliary.get(group, {})]
        if missing:
            issues.append(dict(code='missing_auxiliary_state', group=group, fields=missing))
    if 'command_limits' not in auxiliary:
        issues.append(dict(code='missing_command_limits'))
    # These controllers read reference.logs[plan_cursor:] to update valid_plan.
    # A name such as "logs" is not evidence that the contents are diagnostic.
    for path, kind in execution['object_types'].items():
        if kind == 'quadsmash.control.full_racket_rally_controller_v4.FullRacketRallyControllerV4':
            log_path = path + '/logs'
            if log_path in execution['omitted']:
                issues.append(dict(code='omitted_causal_plan_log', path=log_path,
                                   old_classification=execution['omitted'][log_path]))
    return issues


def require_physical_start(state, execution):
    issues = physical_start_issues(state, execution)
    if issues:
        raise ValueError('Incomplete physical imagination start: ' + repr(issues))

