"""Experimental branch-owned G07 execution graph, gated by command/gradient tests.

Static networks are shared read-only. Mutable Python/tensor state and bound
callbacks are cloned with alias preservation. Native execution is never patched.
The original discrete NumPy interception planner is stop-gradient; subsequent
Torch feedback and command slew remain differentiable. This module alone is
NOT a certified PhysicalImaginationAdapter or a training entry point.
"""
import ast
import inspect
import math
import textwrap
import types
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from quadsmash.control.full_racket_rally_controller_v4 import FullRacketRallyControllerV4
from quadsmash.control.full_racket_arm_target_rate_v4 import ArmTargetRateV4
from .supermimic_execution_preview_v1 import MixerMemory, ResidualExecutionPreviewV1


def _planner_only():
    # Retain the actual source planner verbatim, excluding the final teacher
    # call. Reimplementing its many calibration/validity conditions would risk
    # changing the frozen execution contract. Source provenance is auditable.
    original = inspect.unwrap(FullRacketRallyControllerV4.action)
    tree = ast.parse(textwrap.dedent(inspect.getsource(original)))
    fn = tree.body[0]
    if not isinstance(fn.body[-1], ast.Return):
        raise RuntimeError('G07 planner source shape changed')
    fn.body = fn.body[:-1]
    fn.decorator_list = []
    fn.name = '_plan_current_state'
    namespace = dict(original.__globals__)
    exec(compile(ast.fix_missing_locations(tree), inspect.getfile(original), 'exec'), namespace)
    return namespace[fn.name]


_PLAN = _planner_only()


def _rally_action(self, observation, default, limits):
    # Only interception/validity decisions use a detached observation. No
    # future archive actions/events enter this computation.
    with torch.no_grad():
        _PLAN(self, {**observation, 'policy': observation['policy'].detach()}, default, limits)
    return self.teacher.action(observation, self.parameters, default, limits,
                              event_time=self.event_time, target_offset_world=self.target_offset_world)


def _arm_rate(self, desired, observed, time):
    if desired.shape != self.target.shape or observed.shape != desired.shape or time.shape != self.clock.shape:
        raise ValueError('Arm rate shape mismatch')
    if not all(torch.isfinite(x).all() for x in (desired, observed, time)):
        raise ValueError('Nonfinite arm rate input')
    restarted = (~self.initialized) | (time < self.clock-1e-6)
    before = torch.where(restarted[:, None], observed, self.target)
    dt = torch.where(restarted, torch.full_like(time, self.control_dt),
                     (time-self.clock).clamp(0., self.control_dt))
    maximum = self.rate*dt[:, None]
    self.target = before+(desired-before).clamp(-maximum, maximum)
    self.clock = time.clone()
    self.initialized = torch.ones_like(restarted)
    return self.target.clone()


_CLASSES = {}


def _learning_class(cls):
    if cls in _CLASSES:
        return _CLASSES[cls]
    methods = {}
    for name, value in vars(cls).items():
        if isinstance(value, types.FunctionType) and hasattr(value, '__wrapped__'):
            methods[name] = inspect.unwrap(value)
    if cls is FullRacketRallyControllerV4:
        methods['action'] = _rally_action
    if cls is ArmTargetRateV4:
        methods['__call__'] = _arm_rate
    derived = type(cls.__name__+'BranchPreview', (cls,), methods)
    _CLASSES[cls] = derived
    _CLASSES[derived] = derived
    return derived


def clone_execution_graph(value, *, learning=False, memo=None):
    """Clone only Python execution state; refuse unhandled native objects."""
    memo = {} if memo is None else memo
    if id(value) in memo:
        return memo[id(value)]
    if isinstance(value, torch.Tensor):
        result = value.clone()  # preserve input gradients and tensor aliases
    elif isinstance(value, torch.nn.Module):
        if any(p.requires_grad for p in value.parameters()):
            raise ValueError('Execution source networks must be frozen')
        result = value
    elif isinstance(value, np.ndarray):
        result = value.copy()
    elif value is None or isinstance(value, (str, bool, int, float, np.generic, Path, torch.device)):
        result = value
    elif isinstance(value, (types.ModuleType, type, types.FunctionType, types.BuiltinFunctionType)):
        result = value
    elif isinstance(value, types.MethodType):
        owner = clone_execution_graph(value.__self__, learning=learning, memo=memo)
        fn = inspect.unwrap(value.__func__) if learning else value.__func__
        result = types.MethodType(fn, owner)
    elif type(value) is list and value and type(value[0]) is float and all(type(x) is float for x in value):
        # Public plan observations are long, mutable lists of immutable floats.
        # Give the branch its own list while preserving the exact scalar objects.
        result = value.copy()
    elif isinstance(value, (list, tuple, dict)):
        result = {} if isinstance(value, dict) else []
        memo[id(value)] = result
        if isinstance(value, dict):
            result.update({k: clone_execution_graph(v, learning=learning, memo=memo) for k,v in value.items()})
        else:
            result.extend(clone_execution_graph(v, learning=learning, memo=memo) for v in value)
            if isinstance(value, tuple):
                result = tuple(result)
    elif hasattr(value, '__dict__'):
        cls = _learning_class(type(value)) if learning else type(value)
        result = (types.SimpleNamespace.__new__(cls) if isinstance(value, types.SimpleNamespace)
                  else object.__new__(cls))
        memo[id(value)] = result
        for key, item in vars(value).items():
            setattr(result, key, clone_execution_graph(item, learning=learning, memo=memo))
    else:
        raise TypeError('Unhandled execution state '+str(type(value)))
    memo[id(value)] = result
    return result


def _child(obj, key):
    return obj[int(key)] if isinstance(obj, (list, tuple)) else obj[key] if isinstance(obj, dict) else getattr(obj, key)


def restore_captured_execution(template, snapshot, *, device='cpu'):
    """Restore all recorded leaves and causal plan logs; no zero-filled state."""
    root = clone_execution_graph(template)
    def locate(path):
        obj = root
        for key in path.split('/')[1:]:
            obj = _child(obj, key)
        return obj
    def set_leaf(path, value):
        keys = path.split('/')[1:]
        obj = root
        for i, key in enumerate(keys[:-1]):
            following = keys[i+1]
            if isinstance(obj, list):
                while len(obj) <= int(key): obj.append(None)
                if obj[int(key)] is None: obj[int(key)] = [] if following.isdigit() else {}
            elif isinstance(obj, dict) and key not in obj:
                obj[key] = [] if following.isdigit() else {}
            elif not isinstance(obj, (list, dict)) and getattr(obj, key, None) is None:
                # Lazy diagnostics/plans may be None in the reset template;
                # their actual recorded descendants define the container.
                setattr(obj, key, [] if following.isdigit() else {})
            obj = _child(obj, key)
        key = keys[-1]
        if isinstance(obj, list):
            while len(obj) <= int(key): obj.append(None)
            obj[int(key)] = value
        elif isinstance(obj, dict): obj[key] = value
        else: setattr(obj, key, value)
    for path, value in snapshot['values'].items():
        try: old = locate(path)
        except (KeyError, IndexError, AttributeError): old = None
        if isinstance(old, torch.device): value = torch.device(device)
        elif isinstance(old, Path): value = Path(value)
        set_leaf(path, value)
    for path, value in snapshot['tensors'].items():
        try: old = locate(path)
        except (KeyError, IndexError, AttributeError): old = None
        restored = value.cpu().numpy().copy() if isinstance(old, np.ndarray) else value.to(device).clone()
        set_leaf(path, restored)
    for path, target in snapshot['aliases'].items():
        set_leaf(path, locate(target))
    return root


@dataclass
class G07BatchMemory:
    """Independent controller clocks and plan histories sharing a tensor rollout."""
    branches: tuple
    lane_counts: tuple

    @property
    def reference_controller(self):
        controllers = [branch.reference_controller for branch in self.branches]
        first = controllers[0]
        return types.SimpleNamespace(
            parameters=torch.cat([branch.parameters for branch in controllers]),
            event_time=torch.cat([branch.event_time for branch in controllers]),
            planned=torch.cat([branch.planned for branch in controllers]),
            nominal_target=first.nominal_target, physics=first.physics, height_intercept=first.height_intercept)


_LANE_TENSORS = frozenset({
    'clock', 'initialized', 'event_time', 'planned', 'parameters', 'population', 'runtime_parameters',
    'last_raw', 'reached', 'standing_ticks', 'started', 'start_time', 'start_leg',
    'target', 'target_offset_world', 'nonzero_seen', 'valid_plan', 'ever_active',
    'hit_seen', 'air_hit_seen', 'first_public_hit_clock', 'effective_velocity', 'original_velocity',
    'center_s', 'half_window_s', 'requests', 'last_request', 'last_requested', 'last_actual',
    'last_base_action', 'last_arm_targets_before_slew', 'last_arm_targets_after_slew',
})


class _IncompatibleExecution(Exception):
    pass


def pack_planned_execution(branches, lane_counts, *, required=False):
    """Merge consumed G07 plans and independent tensor memories for one rollout."""
    if len(branches) == 1:
        return branches[0]
    if not all(bool(branch.reference_controller.planned.all()) for branch in branches):
        return G07BatchMemory(branches, lane_counts)
    memo = {}

    def merge(values, path=()):
        first = values[0]
        name = path[-1] if path else ''
        key = tuple(map(id, values))
        if name in ('count', 'num_envs') and all(
                type(v) is int and v == n for v, n in zip(values, lane_counts, strict=True)):
            return sum(lane_counts)
        if name == 'tick' and len(path) > 1 and path[-2] in ('walk', 'standing'):
            device = branches[0].reference_controller.parameters.device
            return torch.cat([torch.as_tensor(v, device=device).expand(n)
                              for v, n in zip(values, lane_counts, strict=True)])
        if name == 'logs':
            return []
        if name in ('last_diagnostic', 'last_diagnostics'):
            return None
        if key in memo:
            return memo[key]
        if any(type(v) is not type(first) for v in values):
            raise _IncompatibleExecution(('controller type differs', path))
        if isinstance(first, torch.Tensor):
            per_lane = name in _LANE_TENSORS or 'runtime_matrices' in path
            if per_lane and all(v.ndim and len(v) == n for v, n in zip(values, lane_counts, strict=True)):
                result = torch.cat(values)
            elif all(v.shape == first.shape and (v is first or torch.allclose(
                    v, first, atol=0., rtol=0., equal_nan=True)) for v in values):
                result = first.clone()
            else:
                raise _IncompatibleExecution(('unrecognized tensor memory', path))
        elif isinstance(first, torch.nn.Module):
            if any(v is not first for v in values):
                raise _IncompatibleExecution(('different frozen source networks', path))
            result = first
        elif isinstance(first, np.ndarray):
            if any(not np.array_equal(v, first, equal_nan=True) for v in values):
                raise _IncompatibleExecution(('different static arrays', path))
            result = first.copy()
        elif isinstance(first, types.MethodType):
            if any(v.__func__ is not first.__func__ for v in values):
                raise _IncompatibleExecution(('different controller callbacks', path))
            result = types.MethodType(first.__func__, merge([v.__self__ for v in values], path + ('owner',)))
        elif isinstance(first, dict):
            if any(v.keys() != first.keys() for v in values):
                raise _IncompatibleExecution(('different execution fields', path))
            result = {}
            memo[key] = result
            result.update((k, merge([v[k] for v in values], path + (k,))) for k in first)
        elif isinstance(first, (list, tuple)):
            if any(len(v) != len(first) for v in values):
                raise _IncompatibleExecution(('different execution records', path))
            result = type(first)(merge(items, path + (str(i),))
                                 for i, items in enumerate(zip(*values, strict=True)))
        elif hasattr(first, '__dict__') and not isinstance(first, (type, types.FunctionType, types.ModuleType)):
            if any(vars(v).keys() != vars(first).keys() for v in values):
                raise _IncompatibleExecution(('different execution memory fields', path))
            cls = type(first)
            result = types.SimpleNamespace.__new__(cls) if isinstance(first, types.SimpleNamespace) else object.__new__(cls)
            memo[key] = result
            for attr in vars(first):
                if attr == 'plan_cursor':
                    bypass = type(first).__name__ == 'MomentumStrikeController' and all(
                        v.requester.gain == 0. for v in values)
                    if not bypass and any(v.plan_cursor < len(v.reference_controller.logs if hasattr(v, 'reference_controller')
                                                              else v.reference.logs) for v in values):
                        raise _IncompatibleExecution(('unconsumed controller plans', path))
                    setattr(result, attr, 0)
                else:
                    setattr(result, attr, merge([getattr(v, attr) for v in values], path + (attr,)))
        else:
            if any(v is not first and v != first and not
                   (isinstance(first, float) and math.isnan(first) and math.isnan(v)) for v in values):
                raise _IncompatibleExecution(('different static control context', path))
            result = first
        memo[key] = result
        return result

    try:
        return merge(branches)
    except _IncompatibleExecution:
        if required:
            raise
        return G07BatchMemory(branches, lane_counts)


class G07MemoryPreviewV1:
    schema = 'g07-branch-owned-execution-preview-experimental-v1'

    def __init__(self, source_controller):
        self.template = source_controller
        mixer = source_controller.mixer
        self.mixer = ResidualExecutionPreviewV1(mixer.scale.tolist(), mixer.arm_rate.rate,
                                               max(mixer.scale.tolist()), mixer.arm_rate.control_dt)

    def _base_step(self, observation, default, limits, branch):
        if isinstance(branch, G07BatchMemory):
            inputs = observation.split(branch.lane_counts, dim=0)
            bases, memories = zip(*(self._base_step(obs, default, limits, memory)
                                    for obs, memory in zip(inputs, branch.branches, strict=True)))
            state = MixerMemory(*(torch.cat([getattr(memory, key) for memory in memories])
                                for key in ('arm_target', 'arm_clock', 'arm_initialized', 'nonzero_seen')))
            return torch.cat(bases), state
        base = branch.base.action({'policy': observation}, default, limits)
        state = MixerMemory(branch.mixer.arm_rate.target, branch.mixer.arm_rate.clock,
                            branch.mixer.arm_rate.initialized, branch.mixer.nonzero_seen)
        return base, state

    @torch.no_grad()
    def prepare_fixed_command(self, observation, default, limits, memory):
        """Fixed native start only. The actor request enters after the G07 base step."""
        return self._base_step(observation, default, limits, clone_execution_graph(memory, learning=True))

    def forward(self, observation, residual, default, limits, memory):
        return self._advance(observation, residual, default, limits, clone_execution_graph(memory, learning=True))

    def private_scoring_step(self, memory):
        """Advance one candidate's private controller until the end of its score."""
        branch = clone_execution_graph(memory, learning=True)

        def step(observation, residual, default, limits):
            if torch.is_grad_enabled():
                raise RuntimeError('Private candidate scoring requires disabled gradients')
            return self._advance(observation, residual, default, limits, branch)

        return step

    def _advance(self, observation, residual, default, limits, branch):
        base, state = self._base_step(observation, default, limits, branch)
        preview = self.mixer(base, residual, observation, default, limits, state)
        next_state = preview.next_memory
        if isinstance(branch, G07BatchMemory):
            lanes = branch.lane_counts
            states = zip(next_state.arm_target.split(lanes), next_state.arm_clock.split(lanes),
                         next_state.arm_initialized.split(lanes), next_state.nonzero_seen.split(lanes), strict=True)
            for memory, values in zip(branch.branches, states, strict=True):
                self._advance_mixer(memory, MixerMemory(*values))
        else:
            self._advance_mixer(branch, next_state)
        return preview, branch, base

    @staticmethod
    def _advance_mixer(branch, next_state):
        branch.mixer.arm_rate.target = next_state.arm_target
        branch.mixer.arm_rate.clock = next_state.arm_clock
        branch.mixer.arm_rate.initialized = next_state.arm_initialized
        branch.mixer.nonzero_seen = next_state.nonzero_seen
