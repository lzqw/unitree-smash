"""Native-label supervision for an explicitly enabled ball-gate repair.

These labels are fit targets, never model/Actor inputs. A free-flight-consistent
transition does not certify absence of every physical contact. Unexplained
non-free transitions remain unknown, rather than becoming false free labels.
The residual tolerances match response_mode_labels in ball_residual_v2.
"""
import torch
from torch.nn import functional as F


def observed_ball_mode_targets(residual, racket_edge, present):
    if residual.ndim != 2 or residual.shape[1] != 6:
        raise ValueError('Expected native ball residual [batch,6]')
    if racket_edge.shape != residual.shape[:1] or present.shape != residual.shape[:1]:
        raise ValueError('Native ball mode label batch mismatch')
    if not torch.isfinite(residual).all():
        raise ValueError('Nonfinite native ball residual')
    if racket_edge.dtype != torch.bool or present.dtype != torch.bool:
        raise ValueError('Native edge/presence labels must be boolean')
    free = ((residual[:, 3:].norm(dim=-1) <= .001)
            & (residual[:, :3].norm(dim=-1) <= .00001) & present)
    target = torch.full_like(present, -1, dtype=torch.long)
    target[free] = 0
    target[racket_edge & present] = 1
    return target


def observed_ball_mode_loss(logits, targets):
    if logits.ndim != 2 or logits.shape[1] != 5 or targets.shape != logits.shape[:1]:
        raise ValueError('Expected five ball modes and aligned native labels')
    if targets.dtype != torch.long or not ((targets >= -1) & (targets <= 1)).all():
        raise ValueError('Only known free/racket labels or explicit unknown are supported')
    if not torch.isfinite(logits).all():
        raise FloatingPointError('Nonfinite ball gate logits')
    known = targets >= 0
    loss = F.cross_entropy(logits[known], targets[known]) if known.any() else logits.sum()*0.
    return loss, dict(free=int((targets == 0).sum()), racket=int((targets == 1).sum()),
                      unknown=int((~known).sum()), supervised=int(known.sum()))
