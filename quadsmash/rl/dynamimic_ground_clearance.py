"""World-vertical surface clearance of the monitored Go2 base and leg colliders."""
import hashlib
import json
from pathlib import Path

import torch
from torch import nn


SHAPES = Path(__file__).resolve().parents[2] / 'configs/go2_piper_authored_ground_geometry_v1.json'


class Go2GroundClearance(nn.Module):
    def __init__(self, geometry, path=SHAPES):
        super().__init__()
        payload = path.read_bytes()
        if hashlib.sha256(payload).hexdigest() != 'e39c41e9b69e481aeac8907c2dfacb9f593cf54e00c8fb822696638ef083167b':
            raise ValueError('Unexpected authored ground geometry export')
        record = json.loads(payload)
        if geometry.asset_sha256 not in record['asset_sha256'].values():
            raise ValueError('Authored ground geometry belongs to another asset')
        names = {'base', *(f'{leg}_{part}' for leg in ('FL','FR','RL','RR') for part in ('hip','thigh','calf'))}
        shapes = [s for s in record['shapes'] if s['body'] in names]
        if {s['body'] for s in shapes} != names or {s['kind'] for s in shapes} != {'Cube','Cylinder'}:
            raise ValueError('Unexpected Go2 base and leg collision shapes')
        self.names = tuple(s['body'] for s in shapes)
        self.left_rear_thigh = self.names.index('RL_thigh')
        self.register_buffer('links', torch.tensor([geometry.ids[s['body']] for s in shapes]))
        transforms = torch.tensor([s['transform'] for s in shapes], dtype=torch.float32)
        self.register_buffer('linear', transforms[:, :3, :3])
        self.register_buffer('offset', transforms[:, 3, :3])
        self.register_buffer('half_box', torch.tensor([s.get('size', 0.)/2 for s in shapes]))
        self.register_buffer('radius', torch.tensor([s.get('radius', 0.) for s in shapes]))
        self.register_buffer('half_height', torch.tensor([s.get('height', 0.)/2 for s in shapes]))
        axes = torch.tensor([s.get('axis', 0) for s in shapes])
        self.register_buffer('axis', torch.nn.functional.one_hot(axes, 3).float())

    def forward(self, positions, rotations):
        # USD row-vector transforms map authored shape coordinates into each link.
        vertical = rotations[:, self.links, 2, :]
        direction = (self.linear * vertical.unsqueeze(-2)).sum(-1)
        z = positions[:, self.links, 2] + (vertical * self.offset).sum(-1)
        box = self.half_box * direction.abs().sum(-1)
        axial = self.half_height * (direction * self.axis).sum(-1).abs()
        radial = self.radius * (direction * (1-self.axis)).norm(dim=-1)
        return z - box - axial - radial

    def monitored_gaps(self, geometry, q, v):
        from .dynamimic_lagrangian_v1 import mv, rotation
        rot = rotation(q[:, 3:7])
        fk = geometry(q[:, 7:], v[:, 6:], q[:, :3], q[:, 3:7],
                      mv(rot, v[:, :3]), mv(rot, v[:, 3:6]))
        gaps = self(fk['position'], fk['rotation'])
        return torch.stack((gaps.amin(-1), gaps[:, self.left_rear_thigh]), -1)

    def thigh_margin_reward(self, gaps):
        # Native R13 training cases 16 ticks before a late fall have median
        # thigh clearance 41 mm. Clean relaxed milestones have a tenth percentile
        # of 75 mm at that lead. There is no bonus for additional height above it.
        x = (gaps[:, self.left_rear_thigh] / .075).clamp(0., 1.)
        return .06 * x.square() * (3 - 2*x)


def ground_survival(previous, gaps):
    # Smooth first entry into the final centimetre, then latch it across the rollout.
    x = ((gaps.amin(-1) - .001) / .009).clamp(0.,1.)
    return torch.minimum(previous, x.square() * (3 - 2*x))
