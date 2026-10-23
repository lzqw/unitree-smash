"""Learned clock-partitioned affine feedback policy, without a teacher prefix."""
import torch

FEATURES = [775, 6, *range(27,39)]
COMPOSED_FEATURES = [775, 6, 7, 0, 1, *range(27,39)]


def policy_features(observation, indices, transform='none'):
    x = observation.index_select(1, indices)
    if transform == 'body-pitch-clock-v1':
        clock = observation[:, 775:776]
        pitch = torch.atan2(observation[:, 6:7], -observation[:, 8:9])
        x = torch.cat((x, pitch, clock.square(), clock*pitch, clock.square()*pitch), 1)
    elif transform != 'none':
        raise ValueError('Unknown learned feature transform')
    return x


class PiecewiseAffinePolicy(torch.nn.Module):
    def __init__(self, checkpoint):
        super().__init__()
        if checkpoint['schema'] != 'learned-pwa-actions-v1':
            raise ValueError('Unknown learned action policy schema')
        for name in ('thresholds','mean','scale','weights'):
            self.register_buffer(name,checkpoint[name].float())
        self.register_buffer('indices',torch.tensor(checkpoint['features'],dtype=torch.long))
        self.feature_transform = checkpoint.get('feature_transform', 'none')

    def forward(self, observation):
        x=policy_features(observation,self.indices,self.feature_transform)
        x=(x-self.mean)/self.scale
        x=torch.cat((torch.ones_like(x[:,:1]),x),dim=1)
        leaf=torch.bucketize(observation[:,775].contiguous(),self.thresholds)
        return torch.bmm(x[:,None,:],self.weights[leaf]).squeeze(1).clamp(-1,1)
