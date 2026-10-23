"""Learned parameter choice from one current public observation, before strike.

Training may use completed development outcomes. Runtime receives no labels,
case identifiers, simulator state, or future observations.
"""
import torch


def selection_features(public, feature_set='full-state-v1'):
    ball = public[:, 61:67]
    ball = ball.sign()*torch.log1p(ball.abs())
    if feature_set=='full-state-v1':
        return torch.cat((ball, public[:, :45], public[:, 760:766]), 1)
    if feature_set=='body-ball-v1':
        return torch.cat((ball, public[:, :9], public[:, 760:766]), 1)
    if feature_set=='body-ball-arm-v1':
        return torch.cat((ball, public[:, :9], public[:, 21:27], public[:, 39:45], public[:, 760:766]), 1)
    raise ValueError('Unknown public selection feature set')


class CausalParameterSelector:
    def __init__(self, checkpoint, device):
        if checkpoint['schema'] != 'g07-public-rbf-parameter-selector-v1':
            raise ValueError('Unknown causal parameter selector')
        self.mean = checkpoint['mean'].to(device)
        self.scale = checkpoint['scale'].to(device)
        self.support = checkpoint['support'].to(device)
        self.alpha = checkpoint['alpha'].to(device)
        self.intercept = checkpoint['intercept'].to(device)
        self.width = checkpoint['kernel_width']
        self.time_s = checkpoint['time_s']
        self.feature_set = checkpoint.get('feature_set', 'full-state-v1')
        self.tables = {k: v.to(device) for k, v in checkpoint['parameter_tables'].items()}

    def choose(self, public):
        query = (selection_features(public, self.feature_set)-self.mean)/self.scale
        distance = (query.square().sum(1)[:, None, None]
                    + self.support.square().sum(2)[None]
                    - 2*torch.einsum('nd,ctd->nct', query, self.support)).clamp(min=0)
        score = torch.einsum('nct,ct->nc', torch.exp(-distance/self.width), self.alpha)+self.intercept
        return score.argmax(1), score
