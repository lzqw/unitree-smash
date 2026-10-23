"""Learned clock-partitioned affine feedback policy, without a teacher prefix."""
import torch

FEATURES = [775, 6, *range(27,39)]


class PiecewiseAffinePolicy(torch.nn.Module):
    def __init__(self, checkpoint):
        super().__init__()
        if checkpoint['schema'] != 'learned-pwa-actions-v1':
            raise ValueError('Unknown learned action policy schema')
        for name in ('thresholds','mean','scale','weights'):
            self.register_buffer(name,checkpoint[name].float())
        self.register_buffer('indices',torch.tensor(checkpoint['features'],dtype=torch.long))

    def forward(self, observation):
        x=observation.index_select(1,self.indices)
        x=(x-self.mean)/self.scale
        x=torch.cat((torch.ones_like(x[:,:1]),x),dim=1)
        leaf=torch.bucketize(observation[:,775].contiguous(),self.thresholds)
        return torch.bmm(x[:,None,:],self.weights[leaf]).squeeze(1).clamp(-1,1)
