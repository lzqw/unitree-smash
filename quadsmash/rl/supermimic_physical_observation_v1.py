"""Differentiable physical-state -> existing 794D policy observation.

Not a dynamics model and not yet connected to the learner. Racket geometry
comes from the pinned URDF. Contact/event state is an explicit input, never
inferred from proximity. Executor and rally memory must still be modelled
before this can serve as a complete imagination state.
"""
from dataclasses import dataclass

import torch
from torch import nn

from quadsmash.control.full_racket_kinematics_v4 import FullRacketKinematicsV4
from quadsmash.physics.native import quat_rotate
from quadsmash.tasks.stand.observation_contract import inverse_rotate, relative_velocity


JOINT_NAMES = tuple(
    [f"{leg}_{part}_joint" for part in ("hip", "thigh", "calf")
     for leg in ("FL", "FR", "RL", "RR")]
    + [f"joint{i}" for i in range(1, 7)])


@dataclass(frozen=True)
class PhysicalPolicyInputs:
    root_position_w: torch.Tensor
    root_quaternion_w: torch.Tensor  # unit wxyz, link frame
    root_link_velocity_w: torch.Tensor
    root_angular_velocity_w: torch.Tensor
    joint_position: torch.Tensor  # JOINT_NAMES order
    joint_velocity: torch.Tensor
    joint_default: torch.Tensor
    foot_contact: torch.Tensor
    ball_position_w: torch.Tensor
    ball_velocity_w: torch.Tensor
    ball_present: torch.Tensor
    airborne: torch.Tensor
    airborne_seen: torch.Tensor
    hit_seen: torch.Tensor
    goal_position_w: torch.Tensor
    goal_normal_w: torch.Tensor
    goal_velocity_w: torch.Tensor
    goal_time_s: torch.Tensor
    goal_enabled: torch.Tensor
    elapsed_s: torch.Tensor
    final_action: torch.Tensor  # actual normalized final command, NOT residual


class PhysicalPolicyObservationV1(nn.Module):
    def __init__(self):
        super().__init__()
        self.kinematics = FullRacketKinematicsV4()

    def forward(self, state: PhysicalPolicyInputs, history_before: torch.Tensor):
        """Insert exactly one new frame; never mutate branch inputs/history.

        Caller owns tick/reset bookkeeping. Boolean event inputs match native
        inference. Differentiable contact/event surrogates need an explicit
        separate design rather than silently changing the original phase rule.
        """
        s = state
        count = s.joint_position.shape[0]
        if s.joint_position.shape != (count, 18) or history_before.shape != (count, 11, 69):
            raise ValueError("Expected named 18-joint state and [batch,11,69] history")
        if s.final_action.shape != (count, 18):
            raise ValueError("final_action must be the 18D final normalized command")
        q, root = s.root_quaternion_w, s.root_position_w
        tool, normal, jacobian, _, _, axes, _ = self.kinematics(
            s.joint_position[:, 12:], return_pose=True)
        tool_velocity_b = torch.einsum("bij,bj->bi", jacobian, s.joint_velocity[:, 12:])
        tool_omega_b = torch.einsum("bji,bj->bi", axes, s.joint_velocity[:, 12:])
        gravity_w = torch.zeros_like(root)
        gravity_w[:, 2] = -1
        present = s.ball_present[:, None].to(root.dtype)
        frame = torch.cat((
            inverse_rotate(q, s.root_link_velocity_w),
            inverse_rotate(q, s.root_angular_velocity_w),
            inverse_rotate(q, gravity_w),
            s.joint_position - s.joint_default, s.joint_velocity,
            tool, normal, tool_velocity_b, tool_omega_b,
            s.foot_contact.to(root.dtype),
            inverse_rotate(q, s.ball_position_w-root) * present,
            relative_velocity(q, s.ball_position_w, s.ball_velocity_w, root,
                              s.root_link_velocity_w, s.root_angular_velocity_w) * present,
            present, torch.ones_like(present)), -1)
        history = torch.cat((frame[:, None], history_before[:, :-1]), 1)
        enabled = s.goal_enabled[:, None].to(root.dtype)
        command = torch.cat((
            (s.goal_time_s-s.elapsed_s)[:, None],
            inverse_rotate(q, s.goal_position_w-root), inverse_rotate(q, s.goal_normal_w),
            relative_velocity(q, s.goal_position_w, s.goal_velocity_w, root,
                              s.root_link_velocity_w, s.root_angular_velocity_w),
            torch.ones_like(present)), -1) * enabled
        phase = torch.zeros(count, dtype=torch.long, device=root.device)
        phase = torch.where(~s.airborne_seen & ~s.airborne & (s.root_link_velocity_w[:, 2] > .05), 1, phase)
        phase = torch.where(s.airborne & ~s.hit_seen, 2, phase)
        phase = torch.where(s.airborne & s.hit_seen, 3, phase)
        phase = torch.where(s.airborne_seen & ~s.airborne & s.foot_contact.bool().any(1), 4, phase)
        observation = torch.cat((history.flatten(1), command,
            torch.nn.functional.one_hot(phase, 5).to(root.dtype),
            s.elapsed_s[:, None], s.final_action), -1)
        return observation, history


@torch.no_grad()
def capture_native_inputs(env):
    """Read-only snapshot for a later native reconstruction audit.

    Does not call observe/reset/step or alter controller memory. Returns copies
    so later native updates cannot silently overwrite the captured evidence.
    """
    d = env.robot.data
    names = tuple(env.robot.joint_names[i] for i in env.joint_ids)
    if names != JOINT_NAMES:
        raise ValueError(f"Unexpected native policy joint order: {names}")
    q = d.root_link_quat_w
    omega = d.root_com_ang_vel_w
    velocity = d.root_com_lin_vel_w - torch.linalg.cross(omega, quat_rotate(q, d.body_com_pos_b[:, 0]))
    present = getattr(env, "ball_observation_enabled", None)
    if present is None:
        present = torch.ones(env.num_envs, dtype=torch.bool, device=env.device)
    hit = getattr(env, "contact_seen", None)
    if hit is None:
        hit = env.virtual_event_seen
    values = dict(root_position_w=d.root_link_pos_w, root_quaternion_w=q,
        root_link_velocity_w=velocity, root_angular_velocity_w=omega,
        joint_position=d.joint_pos[:, env.joint_ids], joint_velocity=d.joint_vel[:, env.joint_ids],
        joint_default=d.default_joint_pos[:, env.joint_ids], foot_contact=env.observed_foot_contact,
        ball_position_w=env.shuttle.data.root_pos_w, ball_velocity_w=env.shuttle.data.root_lin_vel_w,
        ball_present=present, airborne=env.observed_airborne, airborne_seen=env.observed_airborne_seen,
        hit_seen=hit, goal_position_w=env.goal_position_w, goal_normal_w=env.goal_normal_w,
        goal_velocity_w=env.goal_velocity_w, goal_time_s=env.goal_time_s,
        goal_enabled=torch.full((env.num_envs,), env.cfg.event_goal_enabled, device=env.device, dtype=torch.bool),
        elapsed_s=env.episode_length_buf.float()*env.step_dt, final_action=env.last_action)
    return PhysicalPolicyInputs(**{k: v.detach().clone() for k, v in values.items()})
