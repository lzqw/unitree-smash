"""Adapt two learned residual layers without changing their weights or actuators.

Gains multiply the actors' bounded requests before their existing command mixers.
The intact source checkpoint remains embedded. Every original rate, face and
net-volume guard remains in the source controller. This is policy adaptation,
not a physical-model change or evidence of improved task performance.
"""
import torch

from .full_racket_prepose_branch_v4 import make_rally_controller as make_source

SCHEMA = 'full-racket-frozen-source-residual-gain-v4'


def validate_gains(values):
    gains = torch.as_tensor(values, dtype=torch.float64)
    if gains.shape != (2, 2) or not torch.isfinite(gains).all() or (gains < 0).any() or (gains > 1).any():
        raise ValueError('Require two outer-to-inner layers, each with leg/arm gains in [0,1]')
    return gains.tolist()


def gain_candidates():
    """Predeclare a layer/group ablation, including the exact measured source."""
    definitions = [
        ('unchanged_source', [[1., 1.], [1., 1.]]),
        ('inner_leg_off', [[1., 1.], [0., 1.]]),
        ('outer_leg_off', [[0., 1.], [1., 1.]]),
        ('both_leg_off', [[0., 1.], [0., 1.]]),
        ('inner_layer_off', [[1., 1.], [0., 0.]]),
        ('outer_layer_off', [[0., 0.], [1., 1.]]),
        ('both_layers_off', [[0., 0.], [0., 0.]]),
        ('both_leg_half', [[.5, 1.], [.5, 1.]]),
    ]
    return [dict(candidate_id=i, name=name, gains=validate_gains(gains))
            for i, (name, gains) in enumerate(definitions)]


class ScaledResidualActor(torch.nn.Module):
    def __init__(self, actor, gains, device):
        super().__init__()
        self.original_actor = actor
        self.identity = gains == [1., 1.]
        self.register_buffer('gain', torch.tensor([gains[0]]*12+[gains[1]]*6,
                                                dtype=torch.float32, device=device))

    def forward(self, public):
        request = self.original_actor(public)
        # The same original mixer clamps requests first. Multiplying the
        # unbounded actor output would change saturation instead of scaling
        # the actual bounded correction.
        return request if self.identity else request.clamp(-1., 1.)*self.gain


class ResidualGainControllerV4:
    def __init__(self, source, gains, count, device):
        if not source.get('native_execution'):
            raise ValueError('Require an actually executed source policy')
        if source.get('schema') == SCHEMA:
            raise ValueError('Do not recursively compound gain adaptations')
        self.gains = validate_gains(gains)
        self.source = make_source(source, count, device)
        self.reference = self.source.reference_controller
        node, actors = self.source, []
        while True:
            if hasattr(node, 'actor'):
                actors.append(node)
                node = node.base
            elif hasattr(node, 'source'):
                node = node.source
            else:
                break
        if len(actors) != 2:
            raise ValueError('Require exactly the two measured source residual layers')
        self.layers = actors
        for layer, gain in zip(actors, self.gains):
            for parameter in layer.actor.parameters():
                parameter.requires_grad_(False)
            layer.actor = ScaledResidualActor(layer.actor, gain, device).eval()
        teacher = self.reference.teacher
        if teacher.arm_target_rate is None or teacher.joint_preparation is None or teacher.face_feedback is None:
            raise ValueError('Preserve source rate, net-volume and face guards')

    @property
    def logs(self):
        return self.source.logs

    @logs.setter
    def logs(self, value):
        self.source.logs = value

    @property
    def reference_controller(self):
        return self.reference

    def reset(self, ids=None):
        self.source.reset(ids)

    @torch.inference_mode()
    def action(self, observation, default, limits):
        return self.source.action(observation, default, limits)


def make_rally_controller(checkpoint, count, device):
    if checkpoint.get('schema') != SCHEMA:
        return make_source(checkpoint, count, device)
    source = checkpoint['source_public_policy']
    if checkpoint['asset_sha256'] != source['asset_sha256'] or not torch.equal(checkpoint['parameters'], source['parameters']):
        raise ValueError('Gain policy changed the embedded physical/source contract')
    return ResidualGainControllerV4(source, checkpoint['residual_gains_outer_to_inner'], count, device)
