"""Post-release robot dynamics in floating-base local tangent coordinates."""
import math

import torch
from torch import nn
from torch.nn import functional as F

from .dynamimic_lagrangian_v1 import DynaMimicLagrangianV1, mv, rotation
from .supermimic_physical_observation_v1 import JOINT_NAMES


def local_robot_state(physical):
    """Extract q and local v from a native 55D world-frame record."""
    if physical.shape[-1] != 55:
        raise ValueError('Expected native 55D state')
    rot_t = rotation(physical[:, 3:7]).mT
    q = torch.cat((physical[:, :7], physical[:, 13:31]), dim=-1)
    v = torch.cat((mv(rot_t, physical[:, 7:10]), mv(rot_t, physical[:, 10:13]),
                   physical[:, 31:49]), dim=-1)
    return q, v


def integrate_configuration(q, v, dt):
    """Right-compose local angular motion, advance link origin in world coordinates."""
    quat = q[:, 3:7]
    half = v[:, 3:6] * (dt / 2)
    angle = half.norm(dim=-1, keepdim=True)
    delta = half * torch.sinc(angle / math.pi)
    w, xyz = quat[:, :1], quat[:, 1:]
    new_quat = F.normalize(torch.cat((
        w * angle.cos() - (xyz * delta).sum(-1, keepdim=True),
        xyz * angle.cos() + w * delta + torch.linalg.cross(xyz, delta)), dim=-1))
    return torch.cat((q[:, :3] + dt * mv(rotation(quat), v[:, :3]),
                      new_quat, q[:, 7:] + dt * v[:, 6:]), dim=-1)


class ContactForces(nn.Module):
    def __init__(self):
        super().__init__()
        # 9 base + 36 joints + 4 heights + 12 foot velocities + 18 targets
        self.network = nn.Sequential(nn.Linear(79, 128), nn.SiLU(), nn.Linear(128, 128),
                                     nn.SiLU(), nn.Linear(128, 41))

    def forward(self, features, foot_z):
        output = self.network(features)
        contact = output[:, :16].reshape(-1, 4, 4)
        gate = torch.sigmoid((0.035 - foot_z) / 0.008)
        probability = gate * contact[..., 0].sigmoid()
        # Each loaded foot supports a nominal fraction of robot weight.
        normal = probability * 100 * F.softplus(contact[..., 3])
        tangent = contact[..., 1:3]
        direction = tangent / tangent.norm(dim=-1, keepdim=True).clamp_min(1)
        force = torch.cat((0.9 * normal.unsqueeze(-1) * direction, normal.unsqueeze(-1)), -1)
        residual = 5 * output[:, 16:40].tanh()
        return force, probability, residual, output[:, 40]


class DynaMimicRecoveryWorldModel(nn.Module):
    """One member per rollout, with shared measured rigid-body and actuator parameters."""
    def __init__(self, asset, contract, substeps=4):
        super().__init__()
        self.body = DynaMimicLagrangianV1(asset).float()
        if (contract['schema'] != 'native-joint-drive-v1' or
                contract['asset_sha256'] != self.body.asset_sha256 or
                tuple(contract['joint_names']) != JOINT_NAMES or
                contract['control_dt_s'] != 0.01):
            raise ValueError('Native drive contract differs from the recovery dynamics')
        if substeps not in (2, 3, 4):
            raise ValueError('Expected 2 to 4 physics substeps per control tick')
        self.substeps = substeps
        self.dt = contract['control_dt_s'] / substeps
        for key in ('kp', 'kd', 'effort_limit'):
            values = torch.tensor([profile[key] for profile in contract['profiles']], dtype=torch.float32)
            if (values.ndim != 2 or values.shape[1] != 18 or not values.isfinite().all()
                    or (values < 0).any() or (key == 'effort_limit' and (values == 0).any())):
                raise ValueError('Invalid actuator contract field ' + key)
            self.register_buffer(key, values)
        self.members = nn.ModuleList(ContactForces() for _ in range(3))

    def forward(self, q, v, target, member, profile=None):
        if profile is None:
            if self.kp.shape[0] != 1:
                raise ValueError('Multiple native drive profiles require per-lane profile indices')
            profile = 0
        kp, kd, limit = self.kp[profile], self.kd[profile], self.effort_limit[profile]
        for _ in range(self.substeps):
            mass, bias, feet, foot_v, jac = self.body(q, v)
            foot_z = feet[..., 2]
            gravity = -rotation(q[:, 3:7])[:, 2]
            features = torch.cat((v[:, :6], gravity, q[:, 7:], v[:, 6:], foot_z,
                                  foot_v.flatten(1), target), -1)
            force, prob, residual, ground_logit = self.members[member](features, foot_z)
            external = mv(jac.mT, force).sum(1) + residual - bias
            regularized = mass + 1e-6 * torch.eye(24, device=q.device, dtype=q.dtype)
            # Use end-of-substep position/velocity for the implicit native drives.
            # An explicit derivative term excites the low-inertia wrist at 2.5 ms.
            damping = self.dt * (kd + self.dt * kp)
            drive = kp * (target - q[:, 7:] - self.dt * v[:, 6:]) - kd * v[:, 6:]
            implicit_mass = regularized + torch.diag_embed(F.pad(damping, (6, 0)))
            prediction = torch.linalg.solve(implicit_mass, (external + F.pad(drive, (6, 0))).unsqueeze(-1)).squeeze(-1)
            torque = limit * ((drive - damping * prediction[:, 6:]) / limit).tanh()
            net = external + F.pad(torque, (6, 0))
            acceleration = torch.linalg.solve(regularized, net.unsqueeze(-1)).squeeze(-1)
            v = v + self.dt * acceleration
            q = integrate_configuration(q, v, self.dt)
        return q, v, dict(foot_contact=prob, contact_force=force, residual_force=residual,
                          body_ground_logit=ground_logit)
