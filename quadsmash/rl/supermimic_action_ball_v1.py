"""Opt-in, supervised action-conditioned racket-sweep correction.

Uses the predicted post-command robot pose (not future/native labels) and
native racket geometry to estimate a soft collision support. Legacy ball
predictions remain byte-for-byte unchanged unless a verified candidate is
explicitly loaded. This is a differentiable surrogate, not a PhysX solver.
"""
import math
import torch
from torch import nn
from .supermimic_ball_residual_v1 import ball_geometry_features
from .supermimic_ball_geometry_preview_v1 import BallGeometryPreviewV1
from .supermimic_v1 import MLP


def normalized_softmin(values, softness):
    """Smooth minimum without a sampling-count-dependent bias."""
    if values.shape[-1] < 1:
        raise ValueError('Soft minimum requires at least one sample')
    return -softness * (
        torch.logsumexp(-values / softness, dim=-1)
        - math.log(values.shape[-1])
    )


def hermite_sweep_samples(first, last, first_velocity, last_velocity,
                          dt=0.01, count=5):
    """Sample a causal local racket path with endpoint velocity tangents.

    The endpoint states come from the current state and the predicted
    post-command robot state.  This is a differentiable motion approximation,
    not a native contact solve.
    """
    if count < 3 or count % 2 == 0:
        raise ValueError('Sweep sample count must be an odd integer >= 3')
    if not isinstance(dt, (int, float)) or not math.isfinite(float(dt)) or dt <= 0:
        raise ValueError('Sweep interpolation duration must be finite and positive')
    alpha = torch.linspace(0., 1., count, device=first.device, dtype=first.dtype)
    a = alpha.view(1, -1, 1)
    a2 = a * a
    a3 = a2 * a
    p0 = first[:, None]
    p1 = last[:, None]
    # alpha is normalized time.  Convert the physical endpoint velocities
    # from metres/second to the metre displacement over this control step.
    # Omitting dt makes the Hermite tangent term dimensionally incorrect and
    # enlarges the inferred within-tick sweep by roughly 1/dt.
    v0 = (float(dt) * first_velocity)[:, None]
    v1 = (float(dt) * last_velocity)[:, None]
    return ((2 * a3 - 3 * a2 + 1) * p0
            + (a3 - 2 * a2 + a) * v0
            + (-2 * a3 + 3 * a2) * p1
            + (a3 - a2) * v1)


def sweep_gap_summary(preview, first, last, first_velocity, last_velocity,
                      softness, dt=0.01):
    samples = hermite_sweep_samples(
        first, last, first_velocity, last_velocity, dt=dt)
    sampled_gaps = preview.racket_gap(samples.reshape(-1, 3)).reshape(samples.shape[:2])
    smallest = normalized_softmin(sampled_gaps, softness)
    # Preserve the established 3-feature checkpoint shape while using all
    # trajectory samples for support and the middle summary feature.
    gaps = torch.stack((sampled_gaps[..., 0], smallest,
                        sampled_gaps[..., -1]), -1)
    return gaps, smallest, sampled_gaps


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


def postimpact_hard_negatives(contact_edge, episode, terminated, timeout):
    """Observed transition after an edge, excluding another edge or reset.

    This is an offline sampling index, not a claim that the ball has separated
    and not an extra input to the model. Native next state remains the label.
    """
    if not (contact_edge.shape == episode.shape == terminated.shape == timeout.shape):
        raise ValueError('Postimpact label shape mismatch')
    out = torch.zeros_like(contact_edge, dtype=torch.bool)
    out[1:] = contact_edge[:-1].bool() & (episode[1:] == episode[:-1]) & ~(
        terminated[:-1].bool() | timeout[:-1].bool())
    return out & ~contact_edge.bool()


def contact_neighborhood_pools(contact_edge, episode, terminated, timeout, eligible):
    """Disjoint contact/pre/post/background pools; future labels stay offline."""
    if eligible.shape != contact_edge.shape:
        raise ValueError('Contact eligibility shape mismatch')
    pre = preimpact_hard_negatives(contact_edge, episode, terminated, timeout)
    post = postimpact_hard_negatives(contact_edge, episode, terminated, timeout)
    # Consecutive impacts may place a row in both neighborhoods; retain its
    # previous-impact association so post-impact coverage remains explicit.
    pre = pre & ~post
    masks = dict(contact=contact_edge.bool(), pre_contact=pre, post_contact=post,
                 background=~(contact_edge.bool() | pre | post))
    return {key: (mask & eligible.bool()).nonzero() for key, mask in masks.items()}


class ActionConditionedBallV1(nn.Module):
    schema = 'action-conditioned-racket-sweep-v1'
    feature_dim = 48

    def __init__(self, *, preview=None, hidden=(128, 128), softness_m=0.008,
                 sweep_dt_s=0.01):
        super().__init__()
        if not 0 < softness_m <= 0.05:
            raise ValueError('Racket sweep support width must be in (0, 0.05] m')
        self.preview = preview if preview is not None else BallGeometryPreviewV1()
        self.softness_m = float(softness_m)
        if not math.isfinite(float(sweep_dt_s)) or sweep_dt_s <= 0:
            raise ValueError('Sweep control duration must be finite and positive')
        self.sweep_dt_s = float(sweep_dt_s)
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
        gaps, smallest, sampled_gaps = sweep_gap_summary(
            self.preview, first, last, current[:, 9:12], future[:, 9:12],
            self.softness_m, dt=self.sweep_dt_s)
        # Soft minimum keeps the impulse pathway differentiable near grazing
        # trajectories; the original collision geometry is not enlarged.
        support = torch.sigmoid(-smallest/self.softness_m)
        features = torch.cat((current[:, :21],future[:, :21],gaps,last-first),-1)
        normalized = (features/self.input_scale).clamp(-30.,30.)
        residual = (self.head(normalized)*self.output_scale)*support[:,None]*present.reshape(-1,1).to(state.dtype)
        if not torch.isfinite(residual).all():
            raise FloatingPointError('Nonfinite action-conditioned ball response')
        if details:
            return residual, dict(support=support, gaps=gaps,
                                  sampled_gaps=sampled_gaps,
                                  predicted_impact_probability=support,
                                  local_displacement=last-first)
        return residual


class CausalActionConditionedBallV2(ActionConditionedBallV1):
    """V1 response with a causal entry/separation gate.

    The original V1 support is symmetric in the swept segment.  That lets a
    learned correction remain active when the racket is already moving away
    from the shuttle after impact.  This candidate keeps V1's learned
    features and parameter shapes, then gates them using only current and
    predicted-endpoint geometry.  A response is active when the current
    racket is inside the contact envelope or when the sweep is closing on it.
    It never consumes a future native event label.

    This is a differentiable surrogate and requires native paired-action
    validation before it can enable contact gradients for the actor.
    """
    schema = 'causal-action-conditioned-racket-sweep-v2'

    def __init__(self, *, preview=None, hidden=(128, 128), softness_m=0.008,
                 closing_softness_m=0.004, contact_margin_m=0.003):
        super().__init__(preview=preview, hidden=hidden, softness_m=softness_m)
        if closing_softness_m <= 0 or closing_softness_m > 0.05:
            raise ValueError('Closing gate width must be in (0, 0.05] m')
        if contact_margin_m < 0 or contact_margin_m > 0.05:
            raise ValueError('Contact gate margin must be in [0, 0.05] m')
        self.closing_softness_m = float(closing_softness_m)
        self.contact_margin_m = float(contact_margin_m)

    def forward(self, state, predicted_robot, free_ball, geometry, present, *,
                details=False):
        if state.ndim != 2 or state.shape[1] != 55 or predicted_robot.shape != state.shape:
            raise ValueError('Sweep requires [batch,55] current and predicted robot states')
        if free_ball.shape != (len(state), 6) or present.numel() != len(state):
            raise ValueError('Sweep free-flight/ball-presence batch mismatch')
        current = ball_geometry_features(state, geometry)
        future = ball_geometry_features(torch.cat((predicted_robot[:, :49], free_ball), -1), geometry)
        first = current[:, 6:9]
        last = future[:, 6:9]
        gaps, smallest, sampled_gaps = sweep_gap_summary(
            self.preview, first, last, current[:, 9:12], future[:, 9:12],
            self.softness_m)
        support = torch.sigmoid(-smallest / self.softness_m)
        current_gap = self.preview.racket_gap(first)
        future_gap = self.preview.racket_gap(last)
        closing_gate = torch.sigmoid((current_gap - future_gap) / self.closing_softness_m)
        inside_gate = torch.sigmoid((self.contact_margin_m - current_gap) /
                                    self.closing_softness_m)
        # Smooth union.  Inside-contact states remain active even when the
        # endpoint is separating, while post-impact separated states decay.
        causal_gate = closing_gate * (closing_gate + inside_gate * (1. - closing_gate))
        features = torch.cat((current[:, :21], future[:, :21], gaps, last - first), -1)
        normalized = (features / self.input_scale).clamp(-30., 30.)
        residual = ((self.head(normalized) * self.output_scale) * support[:, None] *
                    causal_gate[:, None] * present.reshape(-1, 1).to(state.dtype))
        if not torch.isfinite(residual).all():
            raise FloatingPointError('Nonfinite causal action-conditioned ball response')
        if details:
            return residual, dict(support=support, gaps=gaps,
                                  sampled_gaps=sampled_gaps,
                                  local_displacement=last - first,
                                  closing_gate=closing_gate,
                                  inside_gate=inside_gate,
                                  causal_gate=causal_gate,
                                  predicted_impact_probability=support * causal_gate)
        return residual


class CausalRacketModeActionBallV3(CausalActionConditionedBallV2):
    """Causal sweep response with separate current-mode and impact gates.

    The mode probability is predicted from the current physical state by the
    object-separated ball model. It is not a future contact label. Combining
    The current mode remains a diagnostic input. The response itself is gated
    by the predicted impact geometry so a near-but-separating state cannot
    activate a contact residual one control tick early.
    """
    schema = 'causal-racket-mode-action-sweep-v3'
    requires_racket_probability = True
    uses_predicted_impact_probability = True

    def forward(self, state, predicted_robot, free_ball, geometry, present, *,
                racket_probability, details=False):
        if racket_probability.shape != (len(state),):
            raise ValueError('Racket mode probability must be [batch]')
        if (not torch.isfinite(racket_probability).all() or
                not ((racket_probability >= 0) & (racket_probability <= 1)).all()):
            raise ValueError('Invalid predicted racket mode probability')
        residual, info = super().forward(state, predicted_robot, free_ball,
                                         geometry, present, details=True)
        if details:
            return residual, dict(info, racket_probability=racket_probability,
                                  current_racket_probability=racket_probability,
                                  effective_support=info['predicted_impact_probability'])
        return residual


class TemporalCausalRacketModeActionBallV7(CausalRacketModeActionBallV3):
    """V3 with the previous executed command in the causal sweep features.

    A lead-time native probe can apply a request before the measured impact
    tick.  V3 only sees the current predicted endpoint, so it cannot represent
    actuator memory that survives into the impact transition.  V7 adds the
    previous *executed* command as an explicit causal input.  It is initialized
    by a strict shape-checked migration from V3 and remains an opt-in model.
    """
    schema = 'native-temporal-causal-racket-mode-action-sweep-v7'
    requires_previous_action = True

    def __init__(self, *, preview=None, hidden=(128,128), softness_m=.008,
                 closing_softness_m=.004, contact_margin_m=.003):
        super().__init__(preview=preview, hidden=hidden, softness_m=softness_m,
                         closing_softness_m=closing_softness_m,
                         contact_margin_m=contact_margin_m)
        old_scale=self.input_scale.detach().clone()
        del self._buffers['input_scale']
        self.register_buffer('input_scale',torch.cat((old_scale,old_scale.new_ones(18))))
        self.head=MLP(66,hidden,6)
        nn.init.zeros_(self.head.net[-1].weight)
        nn.init.zeros_(self.head.net[-1].bias)

    def forward(self,state,predicted_robot,free_ball,geometry,present,*,
                racket_probability,previous_action,details=False):
        if previous_action.shape != (len(state),18):
            raise ValueError('Previous executed action must be [batch,18]')
        current=ball_geometry_features(state,geometry)
        future=ball_geometry_features(torch.cat((predicted_robot[:,:49],free_ball),-1),geometry)
        first=current[:,6:9];last=future[:,6:9]
        gaps,smallest,sampled_gaps=sweep_gap_summary(self.preview, first, last,
            current[:,9:12], future[:,9:12], self.softness_m)
        support=torch.sigmoid(-smallest/self.softness_m)
        current_gap=self.preview.racket_gap(first);future_gap=self.preview.racket_gap(last)
        closing_gate=torch.sigmoid((current_gap-future_gap)/self.closing_softness_m)
        inside_gate=torch.sigmoid((self.contact_margin_m-current_gap)/self.closing_softness_m)
        causal_gate=closing_gate*(closing_gate+inside_gate*(1.-closing_gate))
        features=torch.cat((current[:,:21],future[:,:21],gaps,last-first,previous_action),-1)
        normalized=(features/self.input_scale).clamp(-30.,30.)
        residual=(self.head(normalized)*self.output_scale)*support[:,None]*causal_gate[:,None]*present.reshape(-1,1).to(state.dtype)
        if not torch.isfinite(residual).all():raise FloatingPointError('Nonfinite temporal action-conditioned ball response')
        if details:return residual,dict(support=support,sampled_gaps=sampled_gaps,
            predicted_impact_probability=support*causal_gate,
            closing_gate=closing_gate,inside_gate=inside_gate,causal_gate=causal_gate,
            racket_probability=racket_probability,current_racket_probability=racket_probability)
        return residual


def migrate_temporal_v7(source):
    """Explicitly migrate a V3 checkpoint into V7 without strict=False."""
    if source.get('schema') != CausalRacketModeActionBallV3.schema:
        raise ValueError('Temporal V7 migration requires a V3 checkpoint')
    old=action_ball_for_schema(source['schema'],closing_softness_m=float(source.get('closing_softness_m',.004)),contact_margin_m=float(source.get('contact_margin_m',.003)))
    old.load_state_dict(source['model'],strict=True)
    new=TemporalCausalRacketModeActionBallV7(closing_softness_m=old.closing_softness_m,contact_margin_m=old.contact_margin_m)
    target=new.state_dict();saved=old.state_dict()
    for key,value in saved.items():
        if key not in target:raise ValueError('V3 migration field missing: '+key)
        if key == 'head.net.0.weight':
            if target[key].shape[0]!=value.shape[0] or target[key].shape[1]<value.shape[1]:raise ValueError('V3 migration head shape mismatch')
            target[key][:,:value.shape[1]]=value
        elif key == 'input_scale':
            if target[key].shape[0] < value.shape[0]:raise ValueError('V3 migration input-scale shape mismatch')
            target[key][:value.shape[0]]=value
        elif target[key].shape==value.shape:
            target[key]=value
        else:raise ValueError('V3 migration tensor shape mismatch: '+key)
    new.load_state_dict(target,strict=True)
    return new


class TemporalCausalRacketModeTargetActionBallV8(TemporalCausalRacketModeActionBallV7):
    """V7 with the current executed joint target as a causal input.

    A lead-time probe can leave almost identical predicted robot geometry while
    changing the executable target that produces the impact.  V8 exposes that
    target directly, together with V7's previous executed command.  Both are
    current-time quantities and are safe for rollout and BPTT.  The migration
    starts from a V3 checkpoint with the new columns initialized to zero.
    """
    schema = 'native-temporal-causal-racket-mode-target-action-sweep-v8'
    requires_current_target = True

    def __init__(self, *, preview=None, hidden=(128,128), softness_m=.008,
                 closing_softness_m=.004, contact_margin_m=.003):
        super().__init__(preview=preview, hidden=hidden, softness_m=softness_m,
                         closing_softness_m=closing_softness_m,
                         contact_margin_m=contact_margin_m)
        old_scale = self.input_scale.detach().clone()
        del self._buffers['input_scale']
        # V7 already contains 48 geometry + 18 previous-action features.
        self.register_buffer('input_scale', torch.cat((old_scale, old_scale.new_full((18,), 3.0))))
        old_head = self.head
        self.head = MLP(84, hidden, 6)
        nn.init.zeros_(self.head.net[-1].weight)
        nn.init.zeros_(self.head.net[-1].bias)

    def forward(self, state, predicted_robot, free_ball, geometry, present, *,
                racket_probability, previous_action, current_target, details=False):
        if previous_action.shape != (len(state), 18):
            raise ValueError('Previous executed action must be [batch,18]')
        if current_target.shape != (len(state), 18):
            raise ValueError('Current executed target must be [batch,18]')
        current = ball_geometry_features(state, geometry)
        future = ball_geometry_features(torch.cat((predicted_robot[:, :49], free_ball), -1), geometry)
        first = current[:, 6:9]
        last = future[:, 6:9]
        gaps, smallest, sampled_gaps = sweep_gap_summary(
            self.preview, first, last, current[:, 9:12], future[:, 9:12],
            self.softness_m)
        support = torch.sigmoid(-smallest / self.softness_m)
        current_gap = self.preview.racket_gap(first)
        future_gap = self.preview.racket_gap(last)
        closing_gate = torch.sigmoid((current_gap - future_gap) / self.closing_softness_m)
        inside_gate = torch.sigmoid((self.contact_margin_m - current_gap) /
                                    self.closing_softness_m)
        causal_gate = closing_gate * (closing_gate + inside_gate * (1. - closing_gate))
        features = torch.cat((current[:, :21], future[:, :21], gaps,
                              last - first, previous_action, current_target), -1)
        normalized = (features / self.input_scale).clamp(-30., 30.)
        residual = ((self.head(normalized) * self.output_scale) * support[:, None] *
                    causal_gate[:, None] *
                    present.reshape(-1, 1).to(state.dtype))
        if not torch.isfinite(residual).all():
            raise FloatingPointError('Nonfinite temporal target-conditioned ball response')
        if details:
            return residual, dict(support=support, sampled_gaps=sampled_gaps,
                                  predicted_impact_probability=support * causal_gate,
                                  closing_gate=closing_gate,
                                  inside_gate=inside_gate, causal_gate=causal_gate,
                                  racket_probability=racket_probability,
                                  current_racket_probability=racket_probability)
        return residual


class TemporalCausalRacketModeTargetActionBallV9(TemporalCausalRacketModeTargetActionBallV8):
    """V8 with a bounded geometry-gate floor for lead-time calibration.

    Native lead-time probes can reach a real impact while the learned robot
    model remains a few millimetres outside the soft racket envelope.  V9
    keeps the geometry and mode gates as diagnostics, but gives the supervised
    residual a small causal floor so the calibration can learn that mismatch.
    This is still an uncertified candidate and cannot enable actor gradients
    without a native receipt.
    """
    schema = 'native-temporal-causal-racket-mode-target-floor-action-sweep-v9'
    support_floor = .05
    mode_floor = .05

    def forward(self, state, predicted_robot, free_ball, geometry, present, *,
                racket_probability, previous_action, current_target, details=False):
        residual, info = super().forward(
            state, predicted_robot, free_ball, geometry, present,
            racket_probability=racket_probability, previous_action=previous_action,
            current_target=current_target, details=True)
        # V8 has already applied both gates. Recompute its raw head output and
        # replace only the two multiplicative gates with bounded floors.
        current = ball_geometry_features(state, geometry)
        future = ball_geometry_features(torch.cat((predicted_robot[:, :49], free_ball), -1), geometry)
        first, last = current[:, 6:9], future[:, 6:9]
        gaps, smallest, sampled_gaps = sweep_gap_summary(
            self.preview, first, last, current[:, 9:12], future[:, 9:12],
            self.softness_m)
        support = torch.sigmoid(-smallest / self.softness_m)
        current_gap = self.preview.racket_gap(first)
        future_gap = self.preview.racket_gap(last)
        closing = torch.sigmoid((current_gap - future_gap) / self.closing_softness_m)
        inside = torch.sigmoid((self.contact_margin_m - current_gap) / self.closing_softness_m)
        causal = closing * (closing + inside * (1. - closing))
        features = torch.cat((current[:, :21], future[:, :21], gaps, last - first,
                              previous_action, current_target), -1)
        raw = self.head((features / self.input_scale).clamp(-30., 30.)) * self.output_scale
        gate = (self.support_floor + (1. - self.support_floor) * support)
        # The current learned mode remains an observed diagnostic.  The
        # response gate is determined by the predicted impact geometry.
        result = raw * gate[:, None] * causal[:, None] * present.reshape(-1, 1).to(state.dtype)
        if not torch.isfinite(result).all():
            raise FloatingPointError('Nonfinite temporal floor-conditioned ball response')
        if details:
            return result, dict(info, support=support, sampled_gaps=sampled_gaps,
                                causal_gate=causal,
                                predicted_impact_probability=gate * causal,
                                current_racket_probability=racket_probability,
                                effective_support=gate * causal)
        return result


class TemporalCausalRacketModeTargetLinearActionBallV10(TemporalCausalRacketModeTargetActionBallV9):
    """V9 plus an explicit executable-command response channel.

    The nonlinear geometry head can be locally insensitive when the learned
    robot model misses the impact by a few millimetres.  This linear channel
    gives the calibration a bounded first-order map from the current target
    change to ball correction, while retaining V9's causal gates.
    """
    schema = 'native-temporal-causal-racket-mode-target-linear-floor-action-sweep-v10'

    def __init__(self, *, preview=None, hidden=(128,128), softness_m=.008,
                 closing_softness_m=.004, contact_margin_m=.003):
        super().__init__(preview=preview, hidden=hidden, softness_m=softness_m,
                         closing_softness_m=closing_softness_m,
                         contact_margin_m=contact_margin_m)
        self.command_head = nn.Linear(18, 6)
        nn.init.zeros_(self.command_head.weight)
        nn.init.zeros_(self.command_head.bias)

    def forward(self, state, predicted_robot, free_ball, geometry, present, *,
                racket_probability, previous_action, current_target, details=False):
        result, info = super().forward(
            state, predicted_robot, free_ball, geometry, present,
            racket_probability=racket_probability, previous_action=previous_action,
            current_target=current_target, details=True)
        command_delta = (current_target - previous_action) / current_target.new_tensor(3.0)
        current = ball_geometry_features(state, geometry)
        future = ball_geometry_features(torch.cat((predicted_robot[:, :49], free_ball), -1), geometry)
        first, last = current[:, 6:9], future[:, 6:9]
        gaps, smallest, sampled_gaps = sweep_gap_summary(
            self.preview, first, last, current[:, 9:12], future[:, 9:12],
            self.softness_m)
        support = torch.sigmoid(-smallest / self.softness_m)
        current_gap = self.preview.racket_gap(first)
        future_gap = self.preview.racket_gap(last)
        closing = torch.sigmoid((current_gap - future_gap) / self.closing_softness_m)
        inside = torch.sigmoid((self.contact_margin_m - current_gap) / self.closing_softness_m)
        causal = closing * (closing + inside * (1. - closing))
        gate = (self.support_floor + (1. - self.support_floor) * support)
        gate = gate * causal
        result = result + self.command_head(command_delta) * self.output_scale * gate[:, None] * present.reshape(-1, 1).to(state.dtype)
        if details:
            return result, dict(info, command_delta=command_delta,
                                sampled_gaps=sampled_gaps,
                                predicted_impact_probability=gate,
                                current_racket_probability=racket_probability,
                                effective_support=gate)
        return result


class TemporalCausalRacketModeNativeResponseBallV11(TemporalCausalRacketModeTargetLinearActionBallV10):
    """V10 with a causal state/command response head for native calibration.

    The V10 command-only branch cannot represent lane-dependent impact impulse
    directions and is still multiplied by a learned geometry/mode gate.  The
    native head uses only current geometry, predicted current-step robot state,
    previous executed action and current target.  It has no future event input.
    It is initialized to zero and remains an uncertified candidate until a
    held-out native direction receipt passes.
    """
    schema = 'native-temporal-causal-racket-mode-native-response-v11'

    def __init__(self, *, preview=None, hidden=(128,128), softness_m=.008,
                 closing_softness_m=.004, contact_margin_m=.003):
        super().__init__(preview=preview, hidden=hidden, softness_m=softness_m,
                         closing_softness_m=closing_softness_m,
                         contact_margin_m=contact_margin_m)
        self.native_response_head = MLP(84, hidden, 6)
        nn.init.zeros_(self.native_response_head.net[-1].weight)
        nn.init.zeros_(self.native_response_head.net[-1].bias)

    def forward(self, state, predicted_robot, free_ball, geometry, present, *,
                racket_probability, previous_action, current_target, details=False):
        result, info = super().forward(
            state, predicted_robot, free_ball, geometry, present,
            racket_probability=racket_probability, previous_action=previous_action,
            current_target=current_target, details=True)
        current = ball_geometry_features(state, geometry)
        future = ball_geometry_features(torch.cat((predicted_robot[:, :49], free_ball), -1), geometry)
        first, last = current[:, 6:9], future[:, 6:9]
        gaps, _, sampled_gaps = sweep_gap_summary(
            self.preview, first, last, current[:, 9:12], future[:, 9:12],
            self.softness_m)
        features = torch.cat((current[:, :21], future[:, :21], gaps,
                              last - first, previous_action, current_target), -1)
        # The native head is causal and bounded only by the declared output
        # units.  It is trained with native paired labels and supervised
        # non-contact negatives before it can be used for Actor gradients.
        native = self.native_response_head((features / self.input_scale).clamp(-30., 30.))
        native = native * self.output_scale * info['predicted_impact_probability'][:, None]
        native = native * present.reshape(-1, 1).to(state.dtype)
        result = result + native
        if not torch.isfinite(result).all():
            raise FloatingPointError('Nonfinite native response head output')
        if details:
            return result, dict(info, sampled_gaps=sampled_gaps,
                                native_response=native)
        return result


def migrate_temporal_v11(source):
    """Migrate V3 through V10 and add a zero-initialized native head."""
    schema = source.get('schema')
    if schema == TemporalCausalRacketModeTargetLinearActionBallV10.schema:
        # The calibration candidate is commonly already V10.  Reconstructing
        # it through the older migration chain would reinterpret its widened
        # feature tensors and silently lose the learned command channel.
        base = action_ball_from_checkpoint(source)
    elif schema in (TemporalCausalRacketModeTargetActionBallV9.schema,
                    TemporalCausalRacketModeTargetActionBallV8.schema,
                    TemporalCausalRacketModeActionBallV7.schema):
        # These checkpoints already contain their widened temporal tensors.
        # Load the exact class before adding only the new V11 head.
        base = action_ball_from_checkpoint(source)
    elif schema == 'causal-racket-mode-action-sweep-v3':
        base = migrate_temporal_v10(source)
    else:
        raise ValueError(f'Cannot migrate temporal action-ball schema to V11: {schema}')
    new = TemporalCausalRacketModeNativeResponseBallV11(
        closing_softness_m=base.closing_softness_m,
        contact_margin_m=base.contact_margin_m)
    target, saved = new.state_dict(), base.state_dict()
    for key, value in saved.items():
        if key not in target or target[key].shape != value.shape:
            raise ValueError('V11 migration tensor mismatch: ' + key)
        target[key] = value
    new.load_state_dict(target, strict=True)
    return new


def migrate_temporal_v10(source):
    """Migrate V3 to V10 with the explicit linear channel zero initialized."""
    base = migrate_temporal_v9(source)
    new = TemporalCausalRacketModeTargetLinearActionBallV10(
        closing_softness_m=base.closing_softness_m,
        contact_margin_m=base.contact_margin_m)
    target, saved = new.state_dict(), base.state_dict()
    for key, value in saved.items():
        if key not in target:
            raise ValueError('V10 migration field missing: ' + key)
        if key == 'head.net.0.weight':
            target[key][:, :value.shape[1]] = value
        elif key == 'input_scale':
            target[key][:value.shape[0]] = value
        elif target[key].shape == value.shape:
            target[key] = value
        else:
            raise ValueError('V10 migration tensor shape mismatch: ' + key)
    new.load_state_dict(target, strict=True)
    return new


def migrate_temporal_v9(source):
    """Migrate V3 weights into V9 with explicit zero-initialized new columns."""
    base = migrate_temporal_v8(source)
    new = TemporalCausalRacketModeTargetActionBallV9(
        closing_softness_m=base.closing_softness_m,
        contact_margin_m=base.contact_margin_m)
    target, saved = new.state_dict(), base.state_dict()
    for key, value in saved.items():
        if key not in target:
            raise ValueError('V9 migration field missing: ' + key)
        if key == 'head.net.0.weight':
            target[key][:, :value.shape[1]] = value
        elif key == 'input_scale':
            target[key][:value.shape[0]] = value
        elif target[key].shape == value.shape:
            target[key] = value
        else:
            raise ValueError('V9 migration tensor shape mismatch: ' + key)
    new.load_state_dict(target, strict=True)
    return new


def migrate_temporal_v8(source):
    """Migrate V3 weights to V8 with strict shape checks and zero new columns."""
    if source.get('schema') != CausalRacketModeActionBallV3.schema:
        raise ValueError('Temporal V8 migration requires a V3 checkpoint')
    old = action_ball_for_schema(source['schema'],
                                  closing_softness_m=float(source.get('closing_softness_m', .004)),
                                  contact_margin_m=float(source.get('contact_margin_m', .003)))
    old.load_state_dict(source['model'], strict=True)
    new = TemporalCausalRacketModeTargetActionBallV8(
        closing_softness_m=old.closing_softness_m,
        contact_margin_m=old.contact_margin_m)
    target, saved = new.state_dict(), old.state_dict()
    for key, value in saved.items():
        if key not in target:
            raise ValueError('V3 migration field missing: ' + key)
        if key == 'head.net.0.weight':
            if target[key].shape[0] != value.shape[0] or target[key].shape[1] < value.shape[1]:
                raise ValueError('V3 migration head shape mismatch')
            target[key][:, :value.shape[1]] = value
        elif key == 'input_scale':
            if target[key].shape[0] < value.shape[0]:
                raise ValueError('V3 migration input-scale shape mismatch')
            target[key][:value.shape[0]] = value
        elif target[key].shape == value.shape:
            target[key] = value
        else:
            raise ValueError('V3 migration tensor shape mismatch: ' + key)
    new.load_state_dict(target, strict=True)
    return new


class RacketModeActionBallV2(ActionConditionedBallV1):
    """Action correction belongs to the predicted racket mixture component.

    V1 adds its correction even when the shared ball model predicts free
    flight. This explicit schema revision uses that model's CURRENT soft
    racket probability, never a native/future event label. Geometry, command
    authority and learned parameter shapes are unchanged. This remains a
    candidate model, not certified contact or a native execution change.
    """
    schema = 'racket-mode-conditioned-action-sweep-v2'
    requires_racket_probability = True

    def forward(self, state, predicted_robot, free_ball, geometry, present, *,
                racket_probability, details=False):
        if racket_probability.shape != (len(state),):
            raise ValueError('Racket mode probability must be [batch]')
        if (not torch.isfinite(racket_probability).all() or
                not ((racket_probability >= 0) & (racket_probability <= 1)).all()):
            raise ValueError('Invalid predicted racket mode probability')
        residual, info = super().forward(state, predicted_robot, free_ball,
                                         geometry, present, details=True)
        residual = residual * racket_probability[:, None]
        if details:
            return residual, dict(info, racket_probability=racket_probability,
                                  effective_support=info['support']*racket_probability)
        return residual


class ImpactTimeActionBallV3(RacketModeActionBallV2):
    """Explicit candidate using causal impact-time motion, not tick-end motion."""
    schema='impact-time-racket-mode-action-sweep-v3'

    def forward(self,state,predicted_robot,free_ball,geometry,present,*,racket_probability,details=False):
        from .supermimic_impact_timing_v1 import causal_sweep_time,interpolate_physical_state
        current=ball_geometry_features(state,geometry)
        endpoint=torch.cat((predicted_robot[:,:49],free_ball),-1)
        future=ball_geometry_features(endpoint,geometry)
        time,info=causal_sweep_time(current[:,6:9],current[:,9:12],future[:,6:9],self.preview)
        at_impact=interpolate_physical_state(state,endpoint,time,self.preview.times[-1])
        result=super().forward(state,at_impact,at_impact[:,49:],geometry,present,
                               racket_probability=racket_probability,details=details)
        if details:
            residual,metadata=result
            return residual,dict(metadata,impact_time_s=time,**info)
        return result


class MaterialInertiaActionBallV4(ImpactTimeActionBallV3):
    """Learned impact correction conditioned on measured component materials.

    Articulated inertia and normal-impact velocity scale are causal features,
    not an executed collision solver. Tangential/spin/multi-contact response
    remains supervised. In particular no missing ball-spin state is filled
    with a fictitious zero. Within-part gradients are continuous; nearest
    primitive and crossing-bin selection remain explicit discrete choices.
    """
    schema='native-material-articulated-impact-action-v4'

    def __init__(self, *, material_contract, **kwargs):
        from .supermimic_racket_material_contract_v1 import validate_contract
        from .supermimic_articulated_inertia_v1 import ArticulatedInertiaV1
        super().__init__(**kwargs)
        self.material_contract=validate_contract(material_contract,material_contract['asset_sha256'])
        self.inertia=ArticulatedInertiaV1(material_contract['asset_sha256'],dtype=torch.float32)
        self.register_buffer('component_material',torch.tensor(material_contract['effective'],dtype=torch.float32))
        self.register_buffer('mechanical_scale',torch.tensor([1.]*3+[12.]*6+[50.]*9+[1.]*3+[15.]*3))
        self.mechanical_head=MLP(24,(128,128),6)
        nn.init.zeros_(self.mechanical_head.net[-1].weight)
        nn.init.zeros_(self.mechanical_head.net[-1].bias)
        from .supermimic_ball_impulse_prior_v1 import BallImpulsePriorV1
        self.impulse_geometry=BallImpulsePriorV1()
        import json
        from pathlib import Path
        self.ball_mass=float(json.loads((Path(__file__).resolve().parents[2]/'configs/shuttle_physics_v2.json').read_text())['mass_kg'])

    def _load_from_state_dict(self,state_dict,prefix,local_metadata,strict,missing_keys,unexpected_keys,error_msgs):
        # Parent adapter.load_state_dict recursively calls this hook; overriding
        # only public load_state_dict would leave resumed mechanical buffers unchecked.
        expected=self.state_dict()
        for key,value in expected.items():
            if key.startswith(('inertia.','impulse_geometry.')) or key in ('component_material','mechanical_scale'):
                saved=state_dict.get(prefix+key)
                if saved is None or not torch.equal(value.detach().cpu(),saved.detach().cpu()):
                    raise ValueError('Immutable mechanical contract differs: '+prefix+key)
        super()._load_from_state_dict(state_dict,prefix,local_metadata,strict,missing_keys,unexpected_keys,error_msgs)

    def mechanical_features(self, impact, geometry):
        from .supermimic_physical_dataset_v1 import decode_physical
        if geometry.asset_sha256!=self.material_contract['asset_sha256']:
            raise ValueError('Impact inertia/physical geometry asset mismatch')
        fields=decode_physical(impact)
        f=ball_geometry_features(impact,geometry)
        local_normal,part=self.impulse_geometry.racket_surface(f[:,6:9])
        local_point=f[:,6:9]-self.preview.radius*local_normal
        inverse,info=self.inertia(fields['joint_position'],fields['root_quaternion_w'],local_point)
        bodies=geometry(fields['joint_position'],fields['joint_velocity'],fields['root_position_w'],fields['root_quaternion_w'],fields['root_link_velocity_w'],fields['root_angular_velocity_w'])
        rid=geometry.ids['Racket'];normal=(bodies['rotation'][:,rid]@local_normal[...,None]).squeeze(-1)
        generalized=torch.cat((fields['root_link_velocity_w'],fields['root_angular_velocity_w'],fields['joint_velocity']),-1)
        surface=(info['point_jacobian']@generalized[...,None]).squeeze(-1)
        relative=fields['ball_velocity_w']-surface
        material=self.component_material[part]
        vn=(relative*normal).sum(-1)
        normal_inverse=(normal*(inverse@normal[...,None]).squeeze(-1)).sum(-1)
        normal_dv=-(1+material[:,2])*vn.clamp_max(0)/(1+self.ball_mass*normal_inverse)
        features=torch.cat((normal,relative,surface,inverse.flatten(1),material,normal_dv[:,None]*normal),-1)
        if features.shape!=(len(impact),24) or not torch.isfinite(features).all():
            raise FloatingPointError('Invalid causal articulated material features')
        return features,part

    def forward(self,state,predicted_robot,free_ball,geometry,present,*,racket_probability,details=False):
        from .supermimic_impact_timing_v1 import causal_sweep_time,interpolate_physical_state
        current=ball_geometry_features(state,geometry)
        endpoint=torch.cat((predicted_robot[:,:49],free_ball),-1)
        future=ball_geometry_features(endpoint,geometry)
        time,timing=causal_sweep_time(current[:,6:9],current[:,9:12],future[:,6:9],self.preview)
        impact=interpolate_physical_state(state,endpoint,time,self.preview.times[-1])
        # Evaluate the same V2 correction on the same causal impact state as V3.
        legacy,metadata=RacketModeActionBallV2.forward(self,state,impact,impact[:,49:],geometry,present,
            racket_probability=racket_probability,details=True)
        features,part=self.mechanical_features(impact,geometry)
        extra=self.mechanical_head((features/self.mechanical_scale).clamp(-30.,30.))*self.output_scale
        extra=extra*metadata['effective_support'][:,None]*present.reshape(-1,1).to(state.dtype)
        result=legacy+extra
        if not torch.isfinite(result).all():raise FloatingPointError('Nonfinite material-conditioned ball correction')
        if details:return result,dict(metadata,impact_time_s=time,material_part=part,mechanical_features=features,**timing)
        return result


class FrameAwareMaterialInertiaActionBallV5(MaterialInertiaActionBallV4):
    """V4 response network with an explicit frame-contact input contract.

    V4 checkpoints retain their historical string-only probability semantics.
    V5 opts into passing the causal sum of learned string and frame impact
    probabilities, while keeping the network tensors and physical geometry
    contract unchanged.
    """
    schema='native-material-frame-aware-impact-action-v5'
    requires_frame_probability=True


class FrameAwareMaterialInertiaActionBallV6(FrameAwareMaterialInertiaActionBallV5):
    """V5 with a fixed causal articulated-impact impulse prior.

    The prior uses the predicted impact state, authored material and asset
    inertia.  The learned head remains a residual; this is a differentiable
    approximation and does not replace or modify PhysX.
    """
    schema='native-material-frame-aware-impact-action-v6'
    requires_analytic_impact_prior=True

    def forward(self,state,predicted_robot,free_ball,geometry,present,*,racket_probability,details=False):
        from .supermimic_impact_timing_v1 import causal_sweep_time,interpolate_physical_state
        current=ball_geometry_features(state,geometry)
        endpoint=torch.cat((predicted_robot[:,:49],free_ball),-1)
        future=ball_geometry_features(endpoint,geometry)
        time,timing=causal_sweep_time(current[:,6:9],current[:,9:12],future[:,6:9],self.preview)
        impact=interpolate_physical_state(state,endpoint,time,self.preview.times[-1])
        legacy,metadata=RacketModeActionBallV2.forward(self,state,impact,impact[:,49:],geometry,present,
            racket_probability=racket_probability,details=True)
        features,part=self.mechanical_features(impact,geometry)
        support=metadata['effective_support']
        normal_dv=features[:,-3:]
        remaining=(self.impulse_geometry.dt-time).clamp_min(0.)
        prior=torch.cat((normal_dv*remaining[:,None],normal_dv),-1)*support[:,None]
        extra=self.mechanical_head((features/self.mechanical_scale).clamp(-30.,30.))*self.output_scale
        extra=extra*support[:,None]*present.reshape(-1,1).to(state.dtype)
        result=legacy+prior+extra
        if not torch.isfinite(result).all():raise FloatingPointError('Nonfinite frame-aware analytic-prior response')
        if details:return result,dict(metadata,impact_time_s=time,material_part=part,mechanical_features=features,analytic_impact_prior=prior,**timing)
        return result


def migrate_material_inertia_v4(source,material_contract):
    """Explicit V3 migration. Check every legacy tensor; never strict=False."""
    if type(source) is not ImpactTimeActionBallV3:
        raise ValueError('Native-material migration requires explicit V3 source')
    candidate=MaterialInertiaActionBallV4(material_contract=material_contract,
        softness_m=source.softness_m).to(next(source.parameters()).device)
    old=source.state_dict();complete=candidate.state_dict()
    for key,value in old.items():
        if key not in complete or complete[key].shape!=value.shape:
            raise ValueError('Legacy migration field mismatch: '+key)
        complete[key]=value
    candidate.load_state_dict(complete,strict=True)
    return candidate


def action_ball_from_checkpoint(checkpoint):
    kwargs={}
    if checkpoint.get('schema') in (CausalActionConditionedBallV2.schema,
                                    CausalRacketModeActionBallV3.schema,
                                    TemporalCausalRacketModeActionBallV7.schema,
                                    TemporalCausalRacketModeTargetActionBallV8.schema,
                                    TemporalCausalRacketModeTargetActionBallV9.schema,
                                    TemporalCausalRacketModeTargetLinearActionBallV10.schema,
                                    TemporalCausalRacketModeNativeResponseBallV11.schema):
        kwargs.update(closing_softness_m=float(checkpoint.get('closing_softness_m', .004)),
                      contact_margin_m=float(checkpoint.get('contact_margin_m', .003)))
    if checkpoint.get('schema') in (MaterialInertiaActionBallV4.schema,
                                    FrameAwareMaterialInertiaActionBallV5.schema,
                                    FrameAwareMaterialInertiaActionBallV6.schema):
        if 'material_contract' not in checkpoint:raise ValueError('Missing native material receipt in V4 checkpoint')
        kwargs['material_contract']=checkpoint['material_contract']
    return action_ball_for_schema(checkpoint.get('schema'),**kwargs)


def action_ball_for_schema(schema, **kwargs):
    """Explicit dispatch: old checkpoints retain their original computation."""
    classes = {cls.schema: cls for cls in (ActionConditionedBallV1, CausalActionConditionedBallV2,
        CausalRacketModeActionBallV3, RacketModeActionBallV2, ImpactTimeActionBallV3, MaterialInertiaActionBallV4,
        FrameAwareMaterialInertiaActionBallV5, FrameAwareMaterialInertiaActionBallV6,
        TemporalCausalRacketModeActionBallV7, TemporalCausalRacketModeTargetActionBallV8,
        TemporalCausalRacketModeTargetActionBallV9, TemporalCausalRacketModeTargetLinearActionBallV10)}
    classes[TemporalCausalRacketModeNativeResponseBallV11.schema] = TemporalCausalRacketModeNativeResponseBallV11
    if schema not in classes:
        raise ValueError('Unknown action-ball checkpoint schema: '+str(schema))
    return classes[schema](**kwargs)
