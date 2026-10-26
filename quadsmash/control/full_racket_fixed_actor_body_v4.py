"""Search analytic whole-body parameters while preserving both measured actors.

The explicit new family keeps the source policy intact and overrides only the
reference controller's declared38 coordinates after normal resets. Outputs are
ordinary bounded joint commands through the existing5rad/s arm limiters.
"""
import torch

from .full_racket_residual_v4 import SCHEMA as SOURCE_SCHEMA
from .full_racket_residual_v4 import make_rally_controller as make_source_controller

SCHEMA = 'full-racket-frozen-actors-body-search-v4'


class FixedActorBodyControllerV4:
    def __init__(self,source,parameters,count,device):
        if (source.get('schema') != SOURCE_SCHEMA or not source.get('native_execution')
                or source.get('residual_base_mode') != 'frozen_public_policy'):
            raise ValueError('Require the actual measured outer policy and its intact learned base')
        if source['parameters'].shape != (38,):
            raise ValueError('This explicit family requires the existing38-coordinate controller')
        self.source = make_source_controller(source,count,device)
        self.reference = self.source.reference_controller
        node = self.source
        while hasattr(node,'actor'):
            for parameter in node.actor.parameters(): parameter.requires_grad_(False)
            node = node.base
        self.count,self.device = count,device
        value = torch.as_tensor(parameters,dtype=torch.float32,device=device)
        self.population = value[None].repeat(count,1) if value.ndim == 1 else value.clone()
        self._validate(self.population)
        self.reset()

    def _validate(self,parameters):
        if parameters.shape != (self.count,38) or not torch.isfinite(parameters).all():
            raise ValueError('Require one finite38-coordinate vector per environment')
        if ((parameters[:,24] < .8) | (parameters[:,24] > 1.45)).any():
            raise ValueError('Preserve the existing interception-plane range')
        if (parameters[:,37].abs() > .1200001).any():
            raise ValueError('Preserve the existing desired-normal algorithm range')

    @property
    def logs(self): return self.source.logs

    @logs.setter
    def logs(self,value): self.source.logs = value

    @property
    def reference_controller(self): return self.reference

    def set_parameters(self,parameters):
        self._validate(parameters)
        if self.reference.planned.any():
            raise ValueError('Install a new population immediately after a normal reset')
        self.population.copy_(parameters)
        self.reference.parameters.copy_(self.population)

    def reset(self,ids=None):
        self.source.reset(ids)
        if ids is None: self.reference.parameters.copy_(self.population)
        else: self.reference.parameters[ids] = self.population[ids]

    @torch.inference_mode()
    def action(self,observation,default,limits):
        return self.source.action(observation,default,limits)


def make_rally_controller(checkpoint,count,device):
    if checkpoint.get('schema') != SCHEMA:
        return make_source_controller(checkpoint,count,device)
    source = checkpoint['source_public_policy']
    if checkpoint['asset_sha256'] != source['asset_sha256']:
        raise ValueError('Physical asset changed across fixed-actor body search')
    return FixedActorBodyControllerV4(source,checkpoint['parameters'],count,device)
