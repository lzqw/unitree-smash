"""Inspect reachable known learned action sources without executing a policy.

This is a rejection guard, not a proof that arbitrary Python code is actor-free.
It recognizes the actual transferred walk/standing actor and learned PWA strike
classes used by the rally controller, including nested frozen policy wrappers.
"""
from collections.abc import Mapping


def learned_action_sources(controller):
    from quadsmash.control.go2_walk_transfer import Go2WalkTransfer
    from quadsmash.control.piecewise_affine import PiecewiseAffinePolicy
    found, visited = [], set()

    def visit(value, path):
        if id(value) in visited:
            return
        visited.add(id(value))
        if isinstance(value, Go2WalkTransfer):
            found.append(dict(path=path+'.actor', kind='transferred_learned_action_actor',
                owner=type(value).__module__+'.'+type(value).__qualname__))
            return
        if isinstance(value, PiecewiseAffinePolicy):
            found.append(dict(path=path, kind='learned_piecewise_affine_action_policy',
                owner=type(value).__module__+'.'+type(value).__qualname__))
            return
        if isinstance(value, Mapping):
            children = value.items()
        elif isinstance(value, (tuple, list)):
            children = enumerate(value)
        else:
            children = getattr(value, '__dict__', {}).items()
        for name, child in children:
            # Configuration/weight data cannot contain a live controller.
            if isinstance(child, (str, bytes, int, float, bool, type(None))):
                continue
            visit(child, path+'.'+str(name))
    visit(controller, 'controller')
    return found


def require_no_known_learned_action_sources(controller):
    sources = learned_action_sources(controller)
    if sources:
        paths = ', '.join(row['path'] for row in sources)
        raise RuntimeError('Teacher-only rejected: reachable learned action sources: '+paths)
    return sources


class FrozenBaseInferenceAudit:
    """Count actual frozen base calls, preserving each original module output."""
    def __init__(self, teacher):
        import hashlib
        self._hashlib = hashlib
        self.counts = {}
        self.modules = {}
        audit = self
        class Counted:
            def __init__(self, name, module):
                self.name, self.module = name, module
            def __call__(self, *args, **kwargs):
                audit.counts[self.name] += 1
                return self.module(*args, **kwargs)
            def __getattr__(self, name):
                module=self.__dict__.get('module')
                if module is None:raise AttributeError(name)
                return getattr(module, name)
            def __deepcopy__(self, memo):
                from copy import deepcopy
                return deepcopy(self.module,memo)
        for name, owner, attr in (('walk',teacher.walk,'actor'),
                                   ('standing',teacher.standing,'actor'),
                                   ('strike',teacher,'strike')):
            module=getattr(owner,attr)
            module.eval()
            for parameter in module.parameters(): parameter.requires_grad_(False)
            self.modules[name]=module
            self.counts[name]=0
            setattr(owner,attr,Counted(name,module))
        self.before=self.hashes()

    def hashes(self):
        result={}
        for name,module in self.modules.items():
            digest=self._hashlib.sha256()
            for key,tensor in sorted(module.state_dict().items()):
                digest.update(key.encode());digest.update(str(tensor.dtype).encode())
                digest.update(str(tuple(tensor.shape)).encode())
                digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
            result[name]=digest.hexdigest()
        return result

    def report(self):
        after=self.hashes()
        if after != self.before: raise RuntimeError('Frozen base weights/buffers changed')
        return dict(controller_type='frozen_base_assisted_programmatic_teacher',
            frozen_base_forward_counts=dict(self.counts), frozen_state_sha256=after,
            frozen_state_unchanged=True,
            approach_learning_head='not instantiated in this probe',
            recovery_learning_head='not instantiated in this probe',
            policy_optimizer='not instantiated in this probe')
