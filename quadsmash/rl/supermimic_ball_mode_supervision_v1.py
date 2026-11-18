"""Native-label supervision for an explicitly enabled ball-gate repair.

These labels are fit targets, never model/Actor inputs. A free-flight-consistent
transition does not certify absence of every physical contact. Unexplained
non-free transitions remain unknown, rather than becoming false free labels.
The residual tolerances match response_mode_labels in ball_residual_v2.
"""
import torch
from torch.nn import functional as F


def observed_ball_mode_targets(residual, racket_edge, present, frame_edge=None):
    if residual.ndim != 2 or residual.shape[1] != 6:
        raise ValueError('Expected native ball residual [batch,6]')
    if racket_edge.shape != residual.shape[:1] or present.shape != residual.shape[:1]:
        raise ValueError('Native ball mode label batch mismatch')
    if frame_edge is not None and frame_edge.shape != residual.shape[:1]:
        raise ValueError('Native frame label batch mismatch')
    if not torch.isfinite(residual).all():
        raise ValueError('Nonfinite native ball residual')
    if racket_edge.dtype != torch.bool or present.dtype != torch.bool:
        raise ValueError('Native edge/presence labels must be boolean')
    if frame_edge is not None and frame_edge.dtype != torch.bool:
        raise ValueError('Native frame labels must be boolean')
    free = ((residual[:, 3:].norm(dim=-1) <= .001)
            & (residual[:, :3].norm(dim=-1) <= .00001) & present)
    target = torch.full_like(present, -1, dtype=torch.long)
    target[free] = 0
    target[racket_edge & present] = 1
    # Native frame hits are a distinct observed response mode.  They are
    # still supervision-only labels; the deployed forward receives only its
    # causal state and learned mode probabilities.
    if frame_edge is not None:
        target[frame_edge & present] = 4
    return target


def observed_ball_mode_loss(logits, targets, *, allow_extended=False):
    if logits.ndim != 2 or logits.shape[1] != 5 or targets.shape != logits.shape[:1]:
        raise ValueError('Expected five ball modes and aligned native labels')
    upper = 4 if allow_extended else 1
    if targets.dtype != torch.long or not ((targets >= -1) & (targets <= upper)).all():
        raise ValueError('Only known free/racket labels or explicit unknown are supported')
    if not torch.isfinite(logits).all():
        raise FloatingPointError('Nonfinite ball gate logits')
    known = targets >= 0
    loss = F.cross_entropy(logits[known], targets[known]) if known.any() else logits.sum()*0.
    counts = dict(free=int((targets == 0).sum()), racket=int((targets == 1).sum()),
                  unknown=int((~known).sum()), supervised=int(known.sum()))
    if allow_extended:
        counts['frame'] = int((targets == 4).sum())
    return loss, counts
