"""Shared TRAIN-derived preparation targets over intact measured source actors.

Only instantiated analytic controller fields are changed. All forces still come
from the existing native actuators, and every preparation/face/rate guard acts.
An empty override is an exact reference path, including reset behavior.
"""
import numpy as np
import torch

from .full_racket_contact_aim_v4 import make_rally_controller as make_source
from .full_racket_geometry_v4 import FullRacketGeometryV4

SCHEMA = 'full-racket-frozen-source-prepose-branch-v4'
FIELDS = {'hit_joint_position_rad', 'swing_joint_velocity_rad_s',
          'walk_waypoint_delta_m', 'intercept_plane_x_m',
          'desired_normal_world', 'desired_point_velocity_world_mps',
          'nominal_projected_pitch_rad'}


def validate_overrides(overrides):
    if not isinstance(overrides, dict) or set(overrides)-FIELDS:
        raise ValueError('Unknown preparation override')
    result = {}
    for key, value in overrides.items():
        shape = (6,) if key in ('hit_joint_position_rad', 'swing_joint_velocity_rad_s') else (
            (3,) if key in ('desired_normal_world', 'desired_point_velocity_world_mps') else ())
        a = np.asarray(value, dtype=float)
        if a.shape != shape or not np.isfinite(a).all():
            raise ValueError('Invalid finite override: '+key)
        result[key] = a.tolist() if shape else float(a)
    model = FullRacketGeometryV4()
    if 'hit_joint_position_rad' in result:
        q = np.asarray(result['hit_joint_position_rad'])
        if np.any(q < model.bounds[:, 0]) or np.any(q > model.bounds[:, 1]):
            raise ValueError('Preparation target outside original arm joint bounds')
    if 'swing_joint_velocity_rad_s' in result and max(abs(np.asarray(result['swing_joint_velocity_rad_s']))) > 5.:
        raise ValueError('Preserve the existing velocity-seed search bounds')
    if abs(result.get('walk_waypoint_delta_m', 0.)) > .20:
        raise ValueError('Walking target offset outside declared search')
    if not .8 <= result.get('intercept_plane_x_m', 1.08) <= 1.45:
        raise ValueError('Preserve the original interception-plane range')
    if abs(result.get('nominal_projected_pitch_rad', 0.)) > .5:
        raise ValueError('Preparation pitch outside declared search')
    if 'desired_normal_world' in result and abs(np.linalg.norm(result['desired_normal_world'])-1.) > 1e-6:
        raise ValueError('Require a unit world normal')
    if 'desired_point_velocity_world_mps' in result and np.linalg.norm(result['desired_point_velocity_world_mps']) > 5.:
        raise ValueError('Desired point speed outside declared search')
    return result


class PreposeBranchControllerV4:
    def __init__(self, source, overrides, count, device):
        if not source.get('native_execution'):
            raise ValueError('Require an actually executed measured source')
        self.overrides = validate_overrides(overrides)
        self.source = make_source(source, count, device)
        self.reference = self.source.reference_controller
        self.count, self.device = count, device
        teacher = self.reference.teacher
        if teacher.arm_target_rate is None or teacher.joint_preparation is None or teacher.face_feedback is None:
            raise ValueError('Require and preserve measured rate, net-volume and face guards')
        self.runtime_parameters = self.reference.parameters.detach().clone()
        values = self.overrides
        if 'hit_joint_position_rad' in values:
            teacher.arm = teacher.arm.new_tensor(values['hit_joint_position_rad'])
            self.runtime_parameters[:, 8:14] = 0.
            model = FullRacketGeometryV4()
            uv = teacher.contact_offset[1:].detach().cpu().numpy()
            q = np.asarray(values['hit_joint_position_rad'])
            teacher.pitch_basis = teacher.arm.new_tensor(model.pitch_compensation_basis(q, uv=uv, face=teacher.face_sign))
            self.reference.jacobian = model.contact_jacobian(q, uv=uv, face=teacher.face_sign, base_position=(0, 0, 0))
            j = self.reference.jacobian
            self.reference.inverse = j.T @ np.linalg.inv(j @ j.T + .04**2*np.eye(3))
        if 'swing_joint_velocity_rad_s' in values:
            self.runtime_parameters[:, 7] = 1.
            self.runtime_parameters[:, 26:32] = teacher.arm.new_tensor(values['swing_joint_velocity_rad_s'])
        self.runtime_parameters[:, 1] += values.get('walk_waypoint_delta_m', 0.)
        if not bool(((self.runtime_parameters[:, 1] >= .45) & (self.runtime_parameters[:, 1] <= 1.20)).all()):
            raise ValueError('Walking target outside existing algorithm bounds')
        if 'intercept_plane_x_m' in values:
            self.runtime_parameters[:, 24] = values['intercept_plane_x_m']
        if 'desired_normal_world' in values:
            # This field already specifies the complete world direction.
            teacher.normal_world = teacher.arm.new_tensor(values['desired_normal_world'])
            self.runtime_parameters[:, 37] = 0.
        if 'desired_point_velocity_world_mps' in values:
            teacher.velocity_world = teacher.arm.new_tensor(values['desired_point_velocity_world_mps'])
        if 'nominal_projected_pitch_rad' in values:
            teacher.nominal_pitch = values['nominal_projected_pitch_rad']
        if values:
            self.reference.parameters.copy_(self.runtime_parameters)

    @property
    def logs(self):
        return self.source.logs

    @logs.setter
    def logs(self, value):
        self.source.logs = value

    @property
    def reference_controller(self):
        return self.reference

    def reset(self, ids=None):
        self.source.reset(ids)
        if self.overrides:
            if ids is None:
                self.reference.parameters.copy_(self.runtime_parameters)
            else:
                self.reference.parameters[ids] = self.runtime_parameters[ids]

    @torch.inference_mode()
    def action(self, observation, default, limits):
        return self.source.action(observation, default, limits)


class GroupedPreposeControllerV4:
    """One independent source controller per equal contiguous case group."""
    def __init__(self, source, candidates, case_count, device):
        self.case_count = case_count
        self.controllers = [PreposeBranchControllerV4(source, candidate['overrides'], case_count, device)
                            for candidate in candidates]

    def reset(self, ids=None):
        for group, controller in enumerate(self.controllers):
            if ids is None:
                controller.reset()
            else:
                selected = ids[(ids >= group*self.case_count) & (ids < (group+1)*self.case_count)]
                if len(selected):
                    controller.reset(selected-group*self.case_count)

    @property
    def logs(self):
        return [dict(entry, env_ids=[lane+group*self.case_count for lane in entry['env_ids']], candidate_id=group)
                for group, controller in enumerate(self.controllers) for entry in controller.logs]

    @torch.inference_mode()
    def action(self, observation, default, limits):
        return torch.cat([controller.action({'policy': observation['policy'][group*self.case_count:(group+1)*self.case_count]},
                                            default, limits)
                          for group, controller in enumerate(self.controllers)])


def make_rally_controller(checkpoint, count, device):
    if checkpoint.get('schema') != SCHEMA:
        return make_source(checkpoint, count, device)
    source = checkpoint['source_public_policy']
    if checkpoint['asset_sha256'] != source['asset_sha256'] or not torch.equal(checkpoint['parameters'], source['parameters']):
        raise ValueError('Source policy or physical asset changed')
    return PreposeBranchControllerV4(source, checkpoint['runtime_overrides'], count, device)
