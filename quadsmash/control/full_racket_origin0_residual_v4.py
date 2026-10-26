"""A fresh trainable whole-body residual over an actually measured origin0 base."""
import io
import torch

from .full_racket_residual_v4 import ResidualActionMixerV4, baseline_mode
from .full_racket_gain_body_adaptation_v4 import make_rally_controller as make_source

SCHEMA = 'full-racket-origin0-residual-ppo-v4'
BASE_SCHEMAS = {'full-racket-frozen-source-prepose-branch-v4',
                'full-racket-frozen-source-gain-body-adaptation-v4'}


def validate_frozen_baseline(checkpoint):
    if checkpoint.get('schema') not in BASE_SCHEMAS or not checkpoint.get('native_execution'):
        raise ValueError('Require one actually executed origin0 preparation or gain/body policy')
    numerical = checkpoint.get('numerical_execution', {})
    if numerical.get('variant') != 'r3-origin0' or numerical.get('environment_spacing_m') != 0.:
        raise ValueError('Source must explicitly belong to the admitted origin0 layout')
    parameters = checkpoint.get('parameters')
    if not isinstance(parameters, torch.Tensor) or parameters.shape != (38,) or not torch.isfinite(parameters).all():
        raise ValueError('Require one fixed finite38-coordinate policy')
    node = checkpoint
    while 'seed' not in node:
        if 'source_public_policy' not in node:
            raise ValueError('Missing original analytic seed in immutable source tree')
        child = node['source_public_policy']
        if node['asset_sha256'] != child['asset_sha256']:
            raise ValueError('Source tree changed physical assets')
        node = child
    if node['seed'].get('arm_target_rate_limit_rad_s') != 5.:
        raise ValueError('Preserve the measured5rad/s arm target limiter')


def frozen_origin0_controller(checkpoint, count, device):
    validate_frozen_baseline(checkpoint)
    controller = make_source(checkpoint, count, device)
    node, actors = controller, []
    while True:
        if hasattr(node, 'actor'):
            actors.append(node.actor); node = node.base
        elif hasattr(node, 'source'):
            node = node.source
        else:
            break
    if len(actors) != 2:
        raise ValueError('This fresh course requires the two measured source residual layers')
    for actor in actors:
        for parameter in actor.parameters():
            parameter.requires_grad_(False)
    return controller


class Origin0ResidualControllerV4:
    def __init__(self, checkpoint, count, device):
        if checkpoint.get('schema') != SCHEMA or checkpoint.get('residual_base_mode') != 'frozen_public_policy':
            raise ValueError('Explicit origin0 learned-residual schema/base mode required')
        base = checkpoint['base_controller']
        if checkpoint['asset_sha256'] != base['asset_sha256'] or not torch.equal(checkpoint['parameters'], base['parameters']):
            raise ValueError('Residual altered its immutable source parameters or physical asset')
        self.base = frozen_origin0_controller(base, count, device)
        self.actor = torch.jit.load(io.BytesIO(checkpoint['residual_actor_jit']), map_location=device).eval()
        self.mixer = ResidualActionMixerV4(count, device, checkpoint['residual_joint_scale_rad'])

    @property
    def logs(self):
        return self.base.logs

    @logs.setter
    def logs(self, value):
        self.base.logs = value

    @property
    def reference_controller(self):
        return self.base.reference_controller

    def reset(self, ids=None):
        self.base.reset(ids); self.mixer.reset(ids)

    @torch.inference_mode()
    def action(self, observation, default, limits):
        residual = self.actor(observation['policy'])
        base = self.base.action(observation, default, limits)
        return self.mixer.action(base, residual, observation['policy'], default, limits)


def make_rally_controller(checkpoint, count, device):
    if checkpoint.get('schema') == SCHEMA:
        return Origin0ResidualControllerV4(checkpoint, count, device)
    return make_source(checkpoint, count, device)
