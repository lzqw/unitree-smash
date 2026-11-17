"""Opt-in, supervised action-conditioned racket-sweep correction.

Uses the predicted post-command robot pose (not future/native labels) and
native racket geometry to estimate a soft collision support. Legacy ball
predictions remain byte-for-byte unchanged unless a verified candidate is
explicitly loaded. This is a differentiable surrogate, not a PhysX solver.
"""
import torch
from torch import nn
from .supermimic_ball_residual_v1 import ball_geometry_features
from .supermimic_ball_geometry_preview_v1 import BallGeometryPreviewV1
from .supermimic_v1 import MLP


def preimpact_hard_negatives(contact_edge, episode, terminated, timeout):
    """Offline replay label for the transition immediately before a hit edge.

    Never consumed by Actor/WM forward; does not cross a reset boundary.
    """
    if not (contact_edge.shape==episode.shape==terminated.shape==timeout.shape):
        raise ValueError('Preimpact label shape mismatch')
    out=torch.zeros_like(contact_edge,dtype=torch.bool)
    out[:-1]=contact_edge[1:].bool()&(episode[:-1]==episode[1:])&~(
        terminated[:-1].bool()|timeout[:-1].bool())
    return out&~contact_edge.bool()


class ActionConditionedBallV1(nn.Module):
    schema = 'action-conditioned-racket-sweep-v1'
    feature_dim = 48

    def __init__(self, *, preview=None, hidden=(128, 128), softness_m=0.008):
        super().__init__()
        if not 0 < softness_m <= 0.05:
            raise ValueError('Racket sweep support width must be in (0, 0.05] m')
        self.preview = preview if preview is not None else BallGeometryPreviewV1()
        self.softness_m = float(softness_m)
        self.head = MLP(self.feature_dim, hidden, 6)
        nn.init.zeros_(self.head.net[-1].weight)
        nn.init.zeros_(self.head.net[-1].bias)
        one = [3.]*3 + [12.]*3 + [.2]*3 + [12.]*3 + [1.]*3 + [8.]*3 + [15.]*3
        self.register_buffer('input_scale', torch.tensor(one+one+[.03]*3+[.1]*3))
        self.register_buffer('output_scale', torch.tensor([.05]*3+[10.]*3))

    def forward(self, state, predicted_robot, free_ball, geometry, present, *, details=False):
        """Predict [B,6] additional ball correction from current to next pose.

        state is the current 55D physical state, predicted_robot is 55D after
        the learned robot dynamics for the current final joint target, and
        free_ball is [B,6] from the causal free-flight model. No future native
        observations, events, or contact labels are consumed.
        """
        if state.ndim != 2 or state.shape[1] != 55 or predicted_robot.shape != state.shape:
            raise ValueError('Sweep requires [batch,55] current and predicted robot states')
        if free_ball.shape != (len(state),6) or present.numel() != len(state):
            raise ValueError('Sweep free-flight/ball-presence batch mismatch')
        current = ball_geometry_features(state, geometry)
        future = ball_geometry_features(torch.cat((predicted_robot[:, :49], free_ball),-1), geometry)
        first = current[:, 6:9]
        last = future[:, 6:9]
        middle = (first + last) * .5
        gaps = torch.stack((self.preview.racket_gap(first),
                            self.preview.racket_gap(middle),
                            self.preview.racket_gap(last)),-1)
        # Soft minimum keeps the impulse pathway differentiable near grazing
        # trajectories; the original collision geometry is not enlarged.
        smallest = -self.softness_m * torch.logsumexp(-gaps/self.softness_m,-1)
        support = torch.sigmoid(-smallest/self.softness_m)
        features = torch.cat((current[:, :21],future[:, :21],gaps,last-first),-1)
        normalized = (features/self.input_scale).clamp(-30.,30.)
        residual = (self.head(normalized)*self.output_scale)*support[:,None]*present.reshape(-1,1).to(state.dtype)
        if not torch.isfinite(residual).all():
            raise FloatingPointError('Nonfinite action-conditioned ball response')
        if details:
            return residual, dict(support=support, gaps=gaps,
                                  local_displacement=last-first)
        return residual
