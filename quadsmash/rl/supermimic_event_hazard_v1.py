"""Consistent one-shot event hazard supervision and soft latch evolution.

Native labels must be binary, within one episode, and monotone. This fixes
probability semantics, not event precision or one-shot reward computation.
"""
import torch
from torch.nn import functional as F

SEMANTICS = 'conditional-event-hazard-v1'


def event_hazard_loss(logits, native_before, native_after):
    if logits.shape != native_before.shape or logits.shape != native_after.shape:
        raise ValueError('Event hazard labels must match logits')
    eligible = 1. - native_before
    # Only a not-yet-paid event has a meaningful first-occurrence hazard.
    # Positive absolute labels after occurrence must not train hazard=1.
    loss = F.binary_cross_entropy_with_logits(logits, native_after, reduction='none')
    return (loss*eligible).sum()/eligible.sum().clamp_min(1.)


def advance_event_latches(state, logits):
    if state.shape != logits.shape:
        raise ValueError('Event latch state must match logits')
    return state + (1.-state)*torch.sigmoid(logits)
