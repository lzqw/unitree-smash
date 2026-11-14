"""Opt-in, pre-reset physical transitions for model development.

Never restores state or supplies actions. This archive is not yet a training
replay: unknown executor objects are explicitly listed, not silently treated
as a complete Markov state. Tensors are copied before native buffers change.
"""
from dataclasses import fields
import hashlib
import json
from pathlib import Path
import types

import numpy as np
import torch

from .supermimic_physical_observation_v1 import capture_native_inputs, JOINT_NAMES
from .supermimic_imagination_contract_v1 import capture_auxiliary_state, AUXILIARY_SCHEMA

# Names from the hash-verified historical source factories. Do not traverse
# arbitrary library instances or modules reachable from a controller attribute.
SOURCE_STATE_TYPES = frozenset((
    'checked_shared_target_core.SharedTargetController',
    'checked_public_stance_core.StanceRecoveryController',
    'cpu_checked_momentum_strike_core.MomentumStrikeController',
    'checked_air_countermotion_core.AirCountermotionController',
    'air_preserved_actual_recovery_prior.RecoveryPriorController',
    'preserved_windowed_swing_factory.WindowedSwingController',
    'window_preserved_lateral_factory.EarlyLateralController',
    'preserved_measured_swing_factory.MeasuredSwingController',
    'checked_air_countermotion_core.LegCountermotionRequest',
    'countermotion_known_model.KnownModel',
    'cpu_checked_momentum_strike_core.PublicStrikeRequest',
    'public_strike_known_model.KnownModel',
    'checked_public_stance_core.StanceLegRequest',
    'stance_known_model.KnownModel',
))


def tensor_tree_copy(value, *, cpu=False):
    if isinstance(value, torch.Tensor):
        return value.detach().to('cpu').clone() if cpu else value.detach().clone()
    if isinstance(value, dict):
        return {k: tensor_tree_copy(v, cpu=cpu) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [tensor_tree_copy(v, cpu=cpu) for v in value]
    return value


def capture_controller(controller):
    """Inspect attributes, not properties; preserve alias and omission evidence.

    No deepcopy of native handles, no state_dict/forward call, and no mutation.
    Network weights live in the pinned source checkpoint. Their presence is
    recorded but they are not duplicated at every control tick.
    """
    tensors, values, aliases, omitted, object_types = {}, {}, {}, {}, {}
    seen = {}

    def visit(obj, path):
        # isinstance may consult a proxy's __class__. Only inspect its actual
        # Python type here; never invoke arbitrary __getattribute__ hooks.
        cls = type(obj)
        qualified = cls.__module__ + '.' + cls.__name__
        if issubclass(cls, torch.Tensor):
            tensors[path] = obj.detach().clone()
        elif cls in (type(None), bool, int, float, str):
            values[path] = obj
        elif issubclass(cls, np.ndarray):
            tensors[path] = torch.from_numpy(obj.copy())
        elif issubclass(cls, np.generic):
            values[path] = obj.item()
        elif issubclass(cls, (Path, torch.device)):
            values[path] = str(obj)
        elif issubclass(cls, torch.nn.Module):
            omitted[path] = 'network:' + cls.__name__
        elif issubclass(cls, (types.ModuleType, type, types.FunctionType,
                             types.MethodType, types.BuiltinFunctionType)):
            omitted[path] = 'code:' + qualified
        elif id(obj) in seen:
            aliases[path] = seen[id(obj)]
        else:
            seen[id(obj)] = path
            if cls is dict:
                for key, item in obj.items():
                    if not isinstance(key, str):
                        omitted[path] = 'non-string dictionary keys'
                        return
                    visit(item, path + '/' + key)
            elif cls in (list, tuple):
                for i, item in enumerate(obj):
                    visit(item, path + '/' + str(i))
            elif (cls is types.SimpleNamespace or cls.__module__.startswith('quadsmash.control.')
                  or qualified in SOURCE_STATE_TYPES):
                object_types[path] = qualified
                for key, item in object.__getattribute__(obj, '__dict__').items():
                    if key == 'logs' and qualified != ('quadsmash.control.full_racket_rally_controller_v4.'
                                                       'FullRacketRallyControllerV4'):
                        omitted[path + '/' + key] = 'diagnostic logs'
                    else:
                        visit(item, path + '/' + key)
            else:
                omitted[path] = qualified

    visit(controller, 'controller')
    return dict(tensors=tensors, values=values, aliases=aliases,
                omitted=omitted, object_types=object_types)


@torch.no_grad()
def capture_state(env):
    physical = capture_native_inputs(env)
    d = env.robot.data
    # Capture exact native task tensors for schema design; no inferred False
    # labels and no use of post-reset tensors to fill terminal transitions.
    task = {name: value.detach().clone() for name, value in vars(env.rally).items()
            if isinstance(value, torch.Tensor)}
    recovery_names = ('rally_recovery_success', 'body_ground_seen', 'flight_qualified',
                      'observed_airborne', 'observed_airborne_seen', 'contact_seen')
    recovery = {name: getattr(env, name).detach().clone() for name in recovery_names}
    return dict(physical={f.name: getattr(physical, f.name) for f in fields(physical)},
                history=env.policy_history.frames.detach().clone(),
                history_tick=env.policy_history.last_tick.detach().clone(),
                episode_index=env.episode_index.detach().clone(),
                elapsed_tick=env.episode_length_buf.detach().clone(),
                control_tick=int(env.common_step_counter),
                body_position_w=d.body_link_pos_w.detach().clone(),
                body_quaternion_w=d.body_link_quat_w.detach().clone(),
                body_linear_velocity_w=d.body_link_lin_vel_w.detach().clone(),
                body_angular_velocity_w=d.body_link_ang_vel_w.detach().clone(),
                applied_torque=d.applied_torque[:, env.joint_ids].detach().clone(),
                foot_net_force_w=env.feet_contact.data.net_forces_w.detach().clone(),
                shuttle_state_w=env.shuttle.data.root_state_w.detach().clone(),
                task=task, recovery=recovery,
                event_state=env.supermimic_event_state().detach().clone(),
                auxiliary=capture_auxiliary_state(env))


class PhysicalTransitionArchive:
    schema = 'supermimic-native-physical-transition-v1'

    def __init__(self, directory, *, metadata, chunk_ticks=32):
        if chunk_ticks < 1:
            raise ValueError('chunk_ticks must be positive')
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=False)
        self.metadata = dict(metadata, schema=self.schema,
                             joint_names=list(JOINT_NAMES),
                             scope='native data only; executor memory completeness not certified',
                             substep_contact_traces=False,
                             auxiliary_schema=AUXILIARY_SCHEMA,
                             causal_plan_logs_captured=True)
        self.chunk_ticks = chunk_ticks
        self.pending = None
        self.rows = []
        self.chunks = []
        self.ticks = self.transitions = 0
        self.task_sequence = None
        self._manifest()

    def attach_task_sequence(self, env):
        from .supermimic_native_sequence_v1 import NativeTaskSequenceV1
        self.task_sequence = NativeTaskSequenceV1(env)
        self.metadata['substep_contact_traces'] = True
        self.metadata['within_tick_event_sequence'] = NativeTaskSequenceV1.schema
        self.metadata['task_replay_config'] = dict(court=env.rally.court,
            radius=env.rally.radius, recovery_protocol=env.contact_recovery.protocol,
            contact_threshold=env.contact_events.threshold, contact_separation=env.contact_events.separation)
        self._manifest()

    def _manifest(self):
        value = dict(self.metadata, control_ticks=self.ticks, transitions=self.transitions,
                     chunks=self.chunks, pending_transition=self.pending is not None)
        temp = self.directory/'manifest.json.tmp'
        temp.write_text(json.dumps(value, indent=2) + '\n')
        temp.replace(self.directory/'manifest.json')

    def begin(self, env, request, public):
        state = capture_state(env)
        if self.task_sequence is not None:
            state['auxiliary']['not_captured'] = ['PhysX contact solver cache']
        self.begin_record(state, request, public,
                          capture_controller(env._sm_residual_controller))
        if self.task_sequence is not None:
            self.task_sequence.begin()

    def begin_record(self, state, request, public, memory):
        if self.pending is not None:
            raise RuntimeError('Previous physical transition has no pre-reset end')
        self.pending = tensor_tree_copy(dict(before=state, requested_action=request,
                                            observation=public, executor_before=memory))

    def finish(self, env, reward):
        self.pending['base_action'] = env._sm_base_action.detach().clone()
        self.pending['reward_components'] = tensor_tree_copy(env._sm_reward_components)
        if self.task_sequence is not None:
            self.pending['task_sequence'] = self.task_sequence.finish()
        state = capture_state(env)
        if self.task_sequence is not None:
            state['auxiliary']['not_captured'] = ['PhysX contact solver cache']
        self.finish_record(state, env._sm_final_action, env._sm_joint_targets,
                           reward, env.reset_terminated, env.reset_time_outs,
                           capture_controller(env._sm_residual_controller))

    def finish_record(self, state, final_action, targets, reward, terminated, timeout, memory):
        if self.pending is None:
            raise RuntimeError('No physical transition begin')
        before = self.pending['before']
        if not torch.equal(before['episode_index'], state['episode_index']):
            raise RuntimeError('Physical transition crossed reset; refusing reset-state target')
        if state['control_tick'] != before['control_tick'] + 1:
            raise RuntimeError('Nonconsecutive native control ticks')
        if final_action.shape != targets.shape or targets.shape[-1] != 18:
            raise ValueError('Expected final18-joint command')
        row = dict(self.pending, after=state, final_action=final_action,
                   final_joint_targets=targets, reward=reward,
                   terminated=terminated, timeout=timeout, executor_after=memory)
        self.rows.append(tensor_tree_copy(row))
        self.pending = None
        self.ticks += 1
        self.transitions += len(reward)
        if len(self.rows) >= self.chunk_ticks:
            self.flush()

    def flush(self):
        if self.rows:
            name = f'chunk_{len(self.chunks):05d}.pt'
            temp = self.directory/(name + '.tmp')
            torch.save(dict(schema=self.schema, rows=tensor_tree_copy(self.rows, cpu=True)), temp)
            path = self.directory/name
            temp.replace(path)
            self.chunks.append(dict(file=name, ticks=len(self.rows),
                                    sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
            self.rows.clear()
        self._manifest()
