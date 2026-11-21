"""Experimental branch-owned G07 execution graph, gated by command/gradient tests.

Static networks are shared read-only. Mutable Python/tensor state and bound
callbacks are cloned with alias preservation. Native execution is never patched.
The original discrete NumPy interception planner is stop-gradient; subsequent
Torch feedback and command slew remain differentiable. This module alone is
NOT a certified PhysicalImaginationAdapter or a training entry point.
"""
import ast
import inspect
import textwrap
import types
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


class G07MemoryPreviewV1:
    schema = 'g07-branch-owned-execution-preview-experimental-v1'

    def __init__(self, source_controller):
        self.template = source_controller
        mixer = source_controller.mixer
        self.mixer = ResidualExecutionPreviewV1(mixer.scale.tolist(), mixer.arm_rate.rate,
                                               max(mixer.scale.tolist()), mixer.arm_rate.control_dt)

    def forward(self, observation, residual, default, limits, memory):
        branch = clone_execution_graph(memory, learning=True)
        base = branch.base.action({'policy': observation}, default, limits)
        state = MixerMemory(branch.mixer.arm_rate.target, branch.mixer.arm_rate.clock,
                            branch.mixer.arm_rate.initialized, branch.mixer.nonzero_seen)
        preview = self.mixer(base, residual, observation, default, limits, state)
        branch.mixer.arm_rate.target = preview.next_memory.arm_target
        branch.mixer.arm_rate.clock = preview.next_memory.arm_clock
        branch.mixer.arm_rate.initialized = preview.next_memory.arm_initialized
        branch.mixer.nonzero_seen = preview.next_memory.nonzero_seen
        return preview, branch, base
