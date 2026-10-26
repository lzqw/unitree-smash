"""One new residual over the measured body teacher, preserving both fixed actors."""
import io
import torch

from .full_racket_residual_v4 import ResidualActionMixerV4, baseline_mode
from .full_racket_fixed_actor_body_v4 import SCHEMA as BODY_SCHEMA
from .full_racket_fixed_actor_body_v4 import make_rally_controller as make_body_controller

SCHEMA='full-racket-body-residual-ppo-v4'


def validate_frozen_baseline(checkpoint):
    if checkpoint.get('schema') != BODY_SCHEMA or not checkpoint.get('native_execution'):
        raise ValueError('Require the actually executed explicit body-policy family')
    source=checkpoint['source_public_policy']
    if checkpoint['asset_sha256'] != source['asset_sha256']:
        raise ValueError('Body/source physical asset changed')
    if checkpoint['parameters'].shape != (38,) or not torch.isfinite(checkpoint['parameters']).all():
        raise ValueError('Require one measured fixed38-coordinate body teacher')
    if source['seed'].get('arm_target_rate_limit_rad_s') != 5.:
        raise ValueError('Preserve the original arm target limiter')


def frozen_body_controller(checkpoint,count,device):
    validate_frozen_baseline(checkpoint)
    # The existing explicit body factory freezes the two actual source JIT
    # actors and preserves their normalizers, scales and per-lane reset state.
    return make_body_controller(checkpoint,count,device)


class BodyResidualControllerV4:
    def __init__(self,checkpoint,count,device):
        if checkpoint.get('schema') != SCHEMA:
            raise ValueError('Unknown body-residual policy schema')
        base=checkpoint['base_controller']
        if checkpoint['asset_sha256'] != base['asset_sha256'] or not torch.equal(checkpoint['parameters'],base['parameters']):
            raise ValueError('Body residual changed its measured fixed teacher')
        if checkpoint.get('residual_base_mode') != 'frozen_public_policy':
            raise ValueError('Require explicit frozen measured body base mode')
        self.base=frozen_body_controller(base,count,device)
        self.actor=torch.jit.load(io.BytesIO(checkpoint['residual_actor_jit']),map_location=device).eval()
        self.mixer=ResidualActionMixerV4(count,device,checkpoint['residual_joint_scale_rad'])

    @property
    def logs(self):return self.base.logs
    @logs.setter
    def logs(self,value):self.base.logs=value
    @property
    def reference_controller(self):return self.base.reference_controller

    def reset(self,ids=None):
        self.base.reset(ids);self.mixer.reset(ids)

    @torch.inference_mode()
    def action(self,observation,default,limits):
        residual=self.actor(observation['policy'])
        base=self.base.action(observation,default,limits)
        return self.mixer.action(base,residual,observation['policy'],default,limits)


def make_rally_controller(checkpoint,count,device):
    if checkpoint.get('schema') == SCHEMA:
        return BodyResidualControllerV4(checkpoint,count,device)
    return make_body_controller(checkpoint,count,device)
