"""CPU reconstruction checks; synthetic states are NOT native performance."""
from dataclasses import replace
from types import SimpleNamespace as NS

import numpy as np
import pytest
import torch
from scipy.spatial.transform import Rotation

from quadsmash.control.full_racket_geometry_v4 import FullRacketGeometryV4
from quadsmash.rl.supermimic_physical_observation_v1 import (
    JOINT_NAMES, PhysicalPolicyObservationV1, capture_native_inputs)
from quadsmash.tasks.stand.observation_contract import PolicyHistory, observe_native


def fixture_env(enabled=True):
    rng = np.random.default_rng(18301)
    n = 5
    model = FullRacketGeometryV4()
    arm = rng.uniform(model.bounds[:, 0]+.05, model.bounds[:, 1]-.05, (n, 6))
    joint = np.concatenate((rng.normal(size=(n, 12))*.2, arm), -1)
    speed = rng.normal(size=(n, 18))*.4
    root = rng.normal(size=(n, 3)); root[:, 2] += 1.2
    r = Rotation.from_rotvec(rng.normal(size=(n, 3))*.3)
    rmat = r.as_matrix(); xyzw = r.as_quat()
    quat = np.concatenate((xyzw[:, 3:], xyzw[:, :3]), -1)
    velocity = rng.normal(size=(n, 3))*.2
    velocity[:, 2] = [0., .2, .5, -.5, 0.]
    omega = rng.normal(size=(n, 3))*.2
    centers, orientations, vbody, wbody = [], [], [], []
    # Independent NumPy URDF FK and central differences, no Torch Jacobian.
    for q, dq in zip(arm, speed[:, 12:]):
        p, rot = model.forward(q, base_position=(0, 0, 0))
        hi, rh = model.forward(q+1e-6*dq, base_position=(0, 0, 0))
        lo, rl = model.forward(q-1e-6*dq, base_position=(0, 0, 0))
        angular = ((rh-rl)/2e-6) @ rot.T
        centers.append(p); orientations.append(rot); vbody.append((hi-lo)/2e-6)
        wbody.append([angular[2, 1], angular[0, 2], angular[1, 0]])
    centers, orientations = np.array(centers), np.array(orientations)
    offset = np.einsum('bij,bj->bi', rmat, centers)
    tool_v = velocity+np.cross(omega, offset)+np.einsum('bij,bj->bi', rmat, vbody)
    tool_w = omega+np.einsum('bij,bj->bi', rmat, wbody)
    tq = Rotation.from_matrix(rmat@orientations).as_quat()
    tq = np.concatenate((tq[:, 3:], tq[:, :3]), -1)
    tensor = lambda x: torch.tensor(np.array(x), dtype=torch.float32)
    com = np.tile([.02, -.01, .04], (n, 1))
    data = NS(root_link_quat_w=tensor(quat), root_link_pos_w=tensor(root),
        root_com_ang_vel_w=tensor(omega),
        root_com_lin_vel_w=tensor(velocity+np.cross(omega, np.einsum('bij,bj->bi', rmat, com))),
        body_com_pos_b=tensor(com)[:, None],
        projected_gravity_b=tensor(r.inv().apply(np.tile([0., 0., -1.], (n, 1)))),
        joint_pos=tensor(joint), joint_vel=tensor(speed), default_joint_pos=tensor(joint*.1),
        body_link_pos_w=tensor(root+offset)[:, None], body_link_quat_w=tensor(tq)[:, None],
        body_link_lin_vel_w=tensor(tool_v)[:, None], body_link_ang_vel_w=tensor(tool_w)[:, None])
    history = PolicyHistory(n, 'cpu')
    history.frames.copy_(tensor(rng.normal(size=(n, 11, 69))))
    env = NS(robot=NS(data=data, joint_names=JOINT_NAMES), num_envs=n, device='cpu',
        joint_ids=list(range(18)), racket_body_id=0, policy_history=history,
        observed_airborne=torch.tensor([False, False, True, True, False]),
        observed_airborne_seen=torch.tensor([False, False, True, True, True]),
        observed_foot_contact=torch.tensor([[1, 1, 1, 1], [1, 0, 0, 1], [0, 0, 0, 0],
                                            [0, 0, 0, 0], [1, 0, 1, 0]], dtype=torch.bool),
        contact_seen=torch.tensor([False, False, False, True, True]),
        shuttle=NS(data=NS(root_pos_w=tensor(root+rng.normal(size=(n, 3))),
                           root_lin_vel_w=tensor(rng.normal(size=(n, 3))))),
        ball_observation_enabled=torch.tensor([False, True, True, True, False]),
        goal_position_w=tensor(rng.normal(size=(n, 3))), goal_normal_w=tensor(np.tile([1, 0, 0], (n, 1))),
        goal_velocity_w=tensor(rng.normal(size=(n, 3))), goal_time_s=tensor(np.full(n, 2.75)),
        cfg=NS(event_goal_enabled=enabled), common_step_counter=12,
        episode_length_buf=torch.full((n,), 12), step_dt=.01,
        last_action=tensor(rng.uniform(-1, 1, (n, 18))))
    return env


@pytest.mark.parametrize('enabled', [True, False])
def test_reconstruction_matches_original_observation_builder(enabled):
    env = fixture_env(enabled)
    before = env.policy_history.frames.clone()
    state = capture_native_inputs(env)
    model = PhysicalPolicyObservationV1().float()
    actual, history = model(state, before)
    reference = observe_native(env)['policy']
    torch.testing.assert_close(actual, reference, atol=2e-6, rtol=0)
    assert actual[:, 770:775].argmax(1).tolist() == [0, 1, 2, 3, 4]
    torch.testing.assert_close(actual[:, 776:], state.final_action, atol=0, rtol=0)
    torch.testing.assert_close(history[:, 1:], before[:, :-1], atol=0, rtol=0)
    assert not torch.count_nonzero(actual[[0, 4], 61:68])
    # A second candidate is independent and cannot modify the first history.
    saved = before.clone()
    model(replace(state, joint_position=state.joint_position+.01), before)
    torch.testing.assert_close(saved, before, atol=0, rtol=0)


def test_physical_reconstruction_preserves_state_gradients():
    env = fixture_env()
    state = capture_native_inputs(env)
    joints = state.joint_position.clone().requires_grad_()
    speed = state.joint_velocity.clone().requires_grad_()
    model = PhysicalPolicyObservationV1().float()
    obs, _ = model(replace(state, joint_position=joints, joint_velocity=speed), env.policy_history.frames)
    # Only derived tool channels: passing this cannot rely on copying q/qd.
    loss = obs[:, 45:57].square().sum()
    gq, gv = torch.autograd.grad(loss, (joints, speed))
    assert torch.isfinite(gq).all() and torch.isfinite(gv).all()
    assert gq[:, 12:].abs().sum() > 0 and gv[:, 12:].abs().sum() > 0


def test_snapshot_isolation_and_wrong_joint_order_rejection():
    env = fixture_env()
    state = capture_native_inputs(env)
    saved = state.joint_position.clone()
    env.robot.data.joint_pos.add_(1)
    torch.testing.assert_close(state.joint_position, saved, atol=0, rtol=0)
    env.robot.joint_names = tuple(reversed(JOINT_NAMES))
    with pytest.raises(ValueError, match='joint order'):
        capture_native_inputs(env)
