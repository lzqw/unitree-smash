"""One versioned actor request path for native training, evaluation and deployment.

This does not perform the G07 mixer/PD transform: callers retain the original
executor and its history. Task latches must be current, causal, per lane.
"""
from collections.abc import Mapping
import torch

POSTHIT = 'causal-contact-release-recovery-v1'
LEGACY = 'exported-phase-recovery-v1'
PHYSICAL_SCHEMA = 'g07-physical-task-observation-loop-v1'


def contract(mode=POSTHIT):
    if mode not in (POSTHIT, LEGACY):
        raise ValueError('Unknown policy execution mode: '+str(mode))
    return dict(schema='supermimic-policy-execution-v1', recovery_gate=mode,
                request='original-normalized-residual', final_transform='unchanged-G07-mixer-and-native-limiter')


def resolve_contract(checkpoint):
    if checkpoint is None:
        return contract(LEGACY), 'initial-export-default; historical contract'
    declared=checkpoint.get('metadata',{}).get('policy_execution')
    if declared is not None:
        expected=contract(declared.get('recovery_gate'))
        if declared!=expected:
            raise ValueError('Policy execution contract fields differ')
        return expected,'explicit-checkpoint-contract'
    # This exact physical learner has used the causal latch since its first
    # version. Old evaluation scores used a different entry point; do not rewrite them.
    if checkpoint.get('adapter_schema')==PHYSICAL_SCHEMA:
        return contract(POSTHIT),'legacy-physical-checkpoint training contract inferred; prior evaluation scores unchanged'
    return contract(LEGACY),'legacy-nonphysical-export'


def recovery_gate(rally, observation):
    values=[]
    for name in ('had_contact','released'):
        value=rally.get(name) if isinstance(rally,Mapping) else getattr(rally,name,None)
        if not torch.is_tensor(value) or value.numel()!=len(observation):
            raise ValueError('Missing or wrong-sized causal latch: '+name)
        value=value.to(device=observation.device).reshape(-1)
        if value.is_floating_point() and not torch.isfinite(value).all():
            raise ValueError('Nonfinite causal latch: '+name)
        values.append(value>.5)
    return values[0]&values[1]


def policy_forward(actor,observation,rally,*,mode=POSTHIT,sample=False):
    if mode==LEGACY:
        return actor(observation,sample=sample)
    if mode!=POSTHIT:
        raise ValueError('Unknown policy execution mode: '+str(mode))
    if not hasattr(actor,'forward_with_recovery_gate'):
        raise ValueError('Causal recovery contract requires explicit gate forward; no silent fallback')
    return actor.forward_with_recovery_gate(observation,recovery_gate(rally,observation),sample=sample)
