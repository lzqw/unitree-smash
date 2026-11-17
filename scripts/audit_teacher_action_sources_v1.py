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
