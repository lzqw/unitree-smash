"""Object-separated ball responses with causal one-tick approach features.

Mode labels are recorded task events / residual proxies, not complete native
substep contact truth. No future labels or case identifiers enter forward().
"""
import torch
from torch import nn
from .supermimic_v1 import MLP


MODE_NAMES = ('free', 'racket', 'ground', 'net', 'other_residual')


def approach_features(features, dt=.01):
    """Constant-relative-velocity geometry descriptors, not a contact solver."""
    r, v = features[:, 6:9], features[:, 9:12]
    closest_t = (-(r*v).sum(-1, keepdim=True) / v.square().sum(-1, keepdim=True).clamp_min(1e-6)).clamp(0., dt)
    closest = r + closest_t*v
    end = r + dt*v
    # Signed plane time, bounded so near-parallel motion remains finite.
    normal_v = v[:, :1]
    denominator = torch.where(normal_v >= 0., normal_v.clamp_min(1e-4), normal_v.clamp_max(-1e-4))
    plane_t = (-r[:, :1]/denominator).clamp(-dt, 2*dt)
    plane_point = r[:, 1:] + plane_t.clamp(0., dt)*v[:, 1:]
    radial = (plane_point / features.new_tensor([.0915, .1165])).square().sum(-1, keepdim=True)
    ground_end = features[:, 21:22] + dt*features[:, 5:6] - .5*9.81*dt*dt
    net_end = features[:, 22:25] + dt*features[:, 3:6]
    return torch.cat((features, closest_t/dt, closest, end, plane_t/dt,
                      plane_point, radial, ground_end, net_end), -1)


def response_mode_labels(masks, residual, present):
    """Supervision only. Repeated contacts use explicitly named residual proxies.

    Disjoint priority: racket > ground > net > other > free. The response is
    the *whole* measured transition residual; simultaneous impacts aren't
    claimed to have been decomposed into independent impulses.
    """
    nonfree = ((residual[:, 3:].norm(dim=-1) > .001) |
               (residual[:, :3].norm(dim=-1) > .00001)) & present.bool()
    labels = nonfree.long()*4
    labels = torch.where((masks['new_net'] | (masks['after_net'] & nonfree)) & present, 3, labels)
    labels = torch.where((masks['new_ground'] | (masks['after_ground'] & nonfree)) & present, 2, labels)
    labels = torch.where(masks['new_racket'] & present, 1, labels)
    return labels


class BallResidualV2(nn.Module):
    schema = 'object-separated-causal-ball-response-v2'
    feature_dim = 40

    def __init__(self, gate_hidden=(128,128), response_hidden=(512,512)):
        super().__init__()
        self.gate = MLP(self.feature_dim, gate_hidden, 5)
        self.response = MLP(self.feature_dim, response_hidden, 24)
        self.register_buffer('mean', torch.zeros(self.feature_dim))
        self.register_buffer('scale', torch.ones(self.feature_dim))
        self.register_buffer('residual_scale', torch.ones(4,6))

    @torch.no_grad()
    def normalize_from_training(self, features, residual, labels):
        x = approach_features(features)
        self.mean.copy_(x.mean(0)); self.scale.copy_(x.std(0,unbiased=False).clamp_min(.01))
        for mode in range(1,5):
            values = residual[labels == mode]
            if len(values):
                self.residual_scale[mode-1].copy_(values.square().mean(0).sqrt().clamp_min(.001))

    def normalized(self, features):
        return (approach_features(features)-self.mean)/self.scale

    def components(self, features):
        x = self.normalized(features)
        return self.gate(x).softmax(-1), self.response(x).reshape(-1,4,6)*self.residual_scale

    def forward(self, features, present):
        probability, response = self.components(features)
        delta = (probability[:,1:,None]*response).sum(1)
        return present[:,None].to(delta.dtype)*delta, 1.-probability[:,0]
