"""Bounded causal interception proposals; no simulator state writes.

The public observation supplies the actual string-center pose and relative
ball/tool velocities. A short linear time-to-moving-plane estimate chooses a
tangent correction; the existing controller still supplies every native
joint target, actuator force, position limit and 5rad/s target slew bound.
"""
import math

import torch

from quadsmash.control.full_racket_geometry_v4 import FullRacketGeometryV4
from quadsmash.control.full_racket_kinematics_v4 import FullRacketKinematicsV4


class PublicTangentCorrection(torch.nn.Module):
    def __init__(self, default_arm, gain, maximum_delta_rad=.1, horizon_s=.20):
        super().__init__()
        if not 0 < gain <= 1 or not 0 < maximum_delta_rad <= .15 or not .03 < horizon_s <= .25:
            raise ValueError('Require bounded finite public interception settings')
        if not all(math.isfinite(v) for v in [gain, maximum_delta_rad, horizon_s]):
            raise ValueError('Finite settings required')
        kin = FullRacketKinematicsV4().float()
        for name in ['arm_origin', 'offsets', 'rotations', 'tool_offset', 'tool_rotation']:
            self.register_buffer(name, getattr(kin, name).clone())
        bounds = torch.tensor(FullRacketGeometryV4().bounds, dtype=torch.float32)
        self.register_buffer('default_arm', torch.as_tensor(default_arm, dtype=torch.float32).clone())
        self.register_buffer('lower', bounds[:, 0])
        self.register_buffer('upper', bounds[:, 1])
        self.register_buffer('eye', torch.eye(6))
        self.gain = float(gain)
        self.maximum_delta = float(maximum_delta_rad)
        self.horizon = float(horizon_s)

    def kinematics(self, joints):
        count = joints.shape[0]
        orientation = torch.eye(3, device=joints.device, dtype=joints.dtype).unsqueeze(0).expand(count, 3, 3)
        position = self.arm_origin.unsqueeze(0).expand(count, 3)
        origins = torch.jit.annotate(list[torch.Tensor], [])
        axes = torch.jit.annotate(list[torch.Tensor], [])
        for i in range(6):
            position = position + torch.einsum('nij,j->ni', orientation, self.offsets[i])
            orientation = orientation @ self.rotations[i]
            origins.append(position)
            axes.append(orientation[:, :, 2])
            cosine, sine = joints[:, i].cos(), joints[:, i].sin()
            zero, one = torch.zeros_like(cosine), torch.ones_like(cosine)
            turn = torch.stack([cosine, -sine, zero, sine, cosine, zero, zero, zero, one], 1).reshape(count, 3, 3)
            orientation = orientation @ turn
        center = position + torch.einsum('nij,j->ni', orientation, self.tool_offset)
        tool_rotation = orientation @ self.tool_rotation
        normal = tool_rotation[:, :, 0]
        axis_stack, origin_stack = torch.stack(axes, 1), torch.stack(origins, 1)
        jp = torch.linalg.cross(axis_stack, center.unsqueeze(1)-origin_stack, dim=2).transpose(1, 2)
        jn = torch.linalg.cross(axis_stack, normal.unsqueeze(1).expand(-1, 6, -1), dim=2).transpose(1, 2)
        return center, normal, jp, jn

    @torch.jit.export
    def diagnostics(self, public):
        joints = (self.default_arm.unsqueeze(0)+public[:, 21:27]).clamp(self.lower, self.upper)
        _, _, jp, jn = self.kinematics(joints)
        normal = public[:, 48:51]
        normal = normal / normal.norm(p=2, dim=1, keepdim=True).clamp_min(1e-6)
        relative = public[:, 61:64]-public[:, 45:48]
        relative_velocity = public[:, 64:67]-public[:, 51:54]
        normal_rate = torch.linalg.cross(public[:, 54:57], normal, dim=1)
        distance = (normal*relative).sum(1)-.0256
        closing = (normal*relative_velocity).sum(1)+(normal_rate*relative).sum(1)
        tau = distance/(-closing).clamp_min(.25)
        prediction_time = tau.clamp(0., self.horizon)
        predicted_relative = relative+relative_velocity*prediction_time.unsqueeze(1)
        # The scalar plane derivative includes actual tool rotation. For this
        # short local proposal tangent translation holds the current normal.
        error = predicted_relative-normal*(normal*predicted_relative).sum(1, keepdim=True)
        error *= (.14/error.norm(p=2, dim=1, keepdim=True).clamp_min(.14))
        jacobian = torch.cat([jp, .3*jn], dim=1)
        rhs = torch.cat([error, torch.zeros_like(error)], dim=1)
        solved = torch.linalg.solve(jacobian@jacobian.transpose(1, 2)+.04**2*self.eye.unsqueeze(0), rhs.unsqueeze(2))
        delta = (jacobian.transpose(1, 2)@solved).squeeze(2)
        # A scalar trust region preserves the solved direction before joint
        # limits. The final native mixer separately limits position and slew.
        delta *= self.gain
        delta *= (self.maximum_delta/delta.abs().amax(1, keepdim=True).clamp_min(self.maximum_delta))
        delta = (joints+delta).clamp(self.lower, self.upper)-joints
        forward = public[:, 763:766]
        root_x = 1.10-(public[:, 760:763]*forward).sum(1)
        proposed_center, _, _, _ = self.kinematics(joints+delta)
        proposed_world_x = root_x+(proposed_center*forward).sum(1)
        gate = ((public[:, 67] > .5) & (public[:, 68] > .5) & (distance > 0.)
            & (closing < -.25) & (tau > 0.) & (tau < self.horizon)
            & (public[:, 775] > 2.2) & (public[:, 775] < 3.3)
            & (public[:, 773] < .5) & (public[:, 774] < .5)
            & (proposed_world_x < 1.45) & (relative.norm(p=2, dim=1) < 3.))
        ramp = ((self.horizon-tau)/.04).clamp(0., 1.)
        correction = torch.where(gate.unsqueeze(1), delta*ramp.unsqueeze(1), torch.zeros_like(delta))
        return correction, tau, gate, error, distance, closing

    def forward(self, public):
        return self.diagnostics(public)[0]


class ResidualWithTangent(torch.nn.Module):
    def __init__(self, actor, correction, joint_scale_rad):
        super().__init__()
        self.actor = actor
        self.correction = correction
        scale = torch.as_tensor(joint_scale_rad, dtype=torch.float32)
        if scale.shape != (18,) or not torch.isfinite(scale).all() or (scale <= 0).any():
            raise ValueError('Require the original 18 finite positive residual scales')
        self.register_buffer('scale', scale[12:].clone())

    def forward(self, public):
        original = self.actor(public)
        arm = (original[:, 12:]+self.correction(public)/self.scale).clamp(-1., 1.)
        return torch.cat([original[:, :12], arm], dim=1)
